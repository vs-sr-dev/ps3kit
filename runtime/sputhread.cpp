// ps3kit runtime — lv2's SPU threads: SPU images, thread groups and their
// threads, each thread's mailboxes and event ports, and its local store as
// the PPU reaches it (sys_spu_thread_read_ls / write_ls).
//
// An SPU thread runs its image's recompiled program (spu.h) on a host thread
// of its own, in a local store of its own. The image is found by the address
// of the SPU ELF it was imported from: sys_spu_image_import records it
// against the segment table it writes, and sys_spu_thread_initialize reads
// that table back. The thread talks to lv2 the way SPU code does:
//   - the inbound mailbox (4 deep) is written by sys_spu_thread_write_spu_mb
//     and read by the SPU, which waits on it;
//   - a write to the outbound interrupt mailbox with a port number below 64
//     is sys_spu_thread_send_event (data1 from the outbound mailbox, the
//     result back in the inbound one); 64 to 127, sys_spu_thread_throw_event,
//     with no result;
//   - stop 0x110 / 0x111 receives from the event queue bound to the number in
//     the outbound mailbox (sys_spu_thread_bind_queue), and the result and
//     the event's three data words come back in the inbound mailbox;
//   - stop 0x101 ends the group, stop 0x102 the thread, with the status in
//     the outbound mailbox; stop 0x100 yields.
// A terminated group's threads end at their next channel access or stop: a
// thread that spins on registers alone is not interrupted.
#include "rt.h"
#include "spu.h"
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr uint32_t CELL_EBUSY = 0x8001000Au, CELL_ESTAT = 0x8001000Fu, CELL_ENOTCONN = 0x80010016u, CELL_EISCONN = 0x80010015u;
constexpr uint64_t SYS_SPU_THREAD_EVENT_USER_KEY = 0xFFFFFFFF53505501ull;
constexpr uint32_t JOIN_GROUP_EXIT = 1, JOIN_ALL_THREADS_EXIT = 2, JOIN_TERMINATED = 4;
constexpr uint32_t SEG_COPY = 1, SEG_FILL = 2, SEG_INFO = 4;
constexpr size_t kHostStack = 64u << 20;

std::mutex g_mx;
std::condition_variable g_cv;
std::map<uint32_t, uint32_t> g_image_elf;          // segment table -> the SPU ELF it came from

struct Seg { uint32_t type, ls, size, addr; };

struct Group;
struct Thread;
std::map<uint32_t, std::unique_ptr<Group>> g_groups;
std::map<uint32_t, std::unique_ptr<Thread>> g_threads;
uint32_t g_next_group = 0x04000100, g_next_thread = 0x02000000;

struct Killed {};                                  // unwinds a thread of a terminated group

struct Group {
    uint32_t id, num, prio;
    std::string name;
    std::vector<uint32_t> threads;                 // by SPU number; 0 where none
    bool started = false;
    int running = 0;
    uint32_t cause = 0, status = 0;
    std::atomic<bool> kill{false};
    uint32_t joins = 0;                            // how many times the group has ended
};

struct Thread : SPUHost {
    uint32_t id, group, num;
    std::string name;
    uint32_t elf = 0, entry = 0;
    std::vector<Seg> segs;
    uint64_t args[4] = {};
    SPUContext s{};
    std::unique_ptr<uint8_t[]> ls{new uint8_t[SPU_LS_SIZE]()};
    std::deque<uint32_t> in_mbox, out_mbox;
    uint32_t spup[64] = {};                        // event port -> event queue
    std::map<uint32_t, uint32_t> spuq;             // spuq number -> event queue
    uint64_t cfg = 0;

    Group& grp() { return *g_groups.at(group); }
    void check_kill() { if (grp().kill) throw Killed{}; }

    bool rdch(SPUContext&, uint32_t ch, uint32_t& v) override {
        if (ch != 29) return false;                // SPU_RdInMbox
        std::unique_lock<std::mutex> lk(g_mx);
        while (in_mbox.empty()) {
            if (grp().kill) throw Killed{};
            g_cv.wait_for(lk, std::chrono::milliseconds(50));
        }
        v = in_mbox.front();
        in_mbox.pop_front();
        g_cv.notify_all();
        return true;
    }
    bool rchcnt(SPUContext&, uint32_t ch, uint32_t& n) override {
        std::lock_guard<std::mutex> lk(g_mx);
        if (grp().kill) throw Killed{};
        switch (ch) {
        case 29: n = (uint32_t)in_mbox.size(); return true;              // SPU_RdInMbox
        case 28: n = out_mbox.empty() ? 1 : 0; return true;              // SPU_WrOutMbox
        case 30: n = 1; return true;                                      // SPU_WrOutIntrMbox
        default: return false;
        }
    }
    bool wrch(SPUContext&, uint32_t ch, uint32_t v) override {
        if (ch == 28) {                            // SPU_WrOutMbox
            std::lock_guard<std::mutex> lk(g_mx);
            if (grp().kill) throw Killed{};
            out_mbox.clear();                      // one entry; nobody else drains it
            out_mbox.push_back(v);
            return true;
        }
        if (ch != 30) {                            // SPU_WrOutIntrMbox
            check_kill();
            return false;
        }
        uint32_t code = v >> 24;
        if (code >= 128) {
            rt_log("sputhread %08X: interrupt mailbox %08X (code %u) not handled", id, v, code);
            return true;
        }
        uint32_t port = code & 63, data = 0, q;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            if (!out_mbox.empty()) { data = out_mbox.front(); out_mbox.pop_front(); }
            q = spup[port];
        }
        uint32_t res = !q ? CELL_ENOTCONN
                          : lv2_event_queue_push(q, SYS_SPU_THREAD_EVENT_USER_KEY, id,
                                                 (uint64_t)port << 32 | (v & 0xFFFFFF), data)
                                ? CELL_OK : CELL_EBUSY;
        if (code < 64) {                           // send_event answers; throw_event does not
            std::lock_guard<std::mutex> lk(g_mx);
            in_mbox.push_back(res);
            g_cv.notify_all();
        } else if (res != CELL_OK) {
            rt_log("sputhread %08X: throw_event on port %u: %08X", id, port, res);
        }
        return true;
    }
    bool stop(SPUContext&, uint32_t pc, uint32_t code) override {
        std::unique_lock<std::mutex> lk(g_mx);
        if (grp().kill) throw Killed{};
        uint32_t out = out_mbox.empty() ? 0 : out_mbox.front();
        switch (code) {
        case 0x100:                                // yield
            lk.unlock();
            std::this_thread::yield();
            return false;
        case 0x101: {                              // sys_spu_thread_group_exit
            out_mbox.clear();
            Group& g = grp();
            if (!g.cause) { g.cause = JOIN_GROUP_EXIT; g.status = out; }
            g.kill = true;
            g_cv.notify_all();
            return true;
        }
        case 0x102:                                // sys_spu_thread_exit
            out_mbox.clear();
            return true;
        case 0x110: case 0x111: {                  // sys_spu_thread_(try)receive_event
            out_mbox.clear();
            auto it = spuq.find(out);
            if (it == spuq.end()) { in_mbox.push_back(CELL_EINVAL); g_cv.notify_all(); return false; }
            uint32_t q = it->second;
            lk.unlock();
            uint64_t ev[4];
            uint32_t r = lv2_event_queue_pop(q, ev, code == 0x110, &grp().kill);
            if (grp().kill) throw Killed{};
            lk.lock();
            in_mbox.push_back(r);
            if (r == CELL_OK)
                for (int i = 1; i < 4; ++i) in_mbox.push_back((uint32_t)ev[i]);
            g_cv.notify_all();
            return false;
        }
        default:
            rt_log("sputhread %08X \"%s\": stop %X at %05X, not a system stop; it goes on", id, name.c_str(), code, pc);
            return false;
        }
    }

    void load() {                                  // the image into local store, the registers
        std::memset(ls.get(), 0, SPU_LS_SIZE);
        for (const Seg& g : segs) {
            if (g.ls >= SPU_LS_SIZE || g.ls + g.size > SPU_LS_SIZE) {
                rt_log("sputhread %08X: segment at LS %X+%X does not fit", id, g.ls, g.size);
                continue;
            }
            if (g.type == SEG_COPY) std::memcpy(ls.get() + g.ls, g_mem + g.addr, g.size);
            else if (g.type == SEG_FILL)
                for (uint32_t i = 0; i + 4 <= g.size; i += 4) {
                    uint32_t w = PPU_BSWAP32(g.addr);
                    std::memcpy(ls.get() + g.ls + i, &w, 4);
                }
        }
        s = SPUContext{};
        s.ls = ls.get();
        s.id = num;
        s.host = this;
        s.raddr = ~0u;
        s.r[1] = spu_slot(0x3FFF0);
        for (int i = 0; i < 4; ++i) s.r[3 + i].u64[1] = args[i];
        in_mbox.clear();
        out_mbox.clear();
    }
};

void* thread_main(void* p) {
    Thread* t = static_cast<Thread*>(p);
    const SPUProgram* prog = spu_program_of(t->elf);
    if (!prog) {
        rt_log("sputhread %08X \"%s\": the SPU ELF at %08X was not recompiled; the thread ends at once", t->id,
               t->name.c_str(), t->elf);
    } else {
        try {
            uint32_t pc = spu_run(t->s, prog, t->entry);
            if (pc != SPU_STOPPED)
                rt_log("sputhread %08X \"%s\": branched out of its program to %05X; the thread ends", t->id,
                       t->name.c_str(), pc);
        } catch (const Killed&) {
        }
    }
    std::lock_guard<std::mutex> lk(g_mx);
    Group& g = t->grp();
    if (--g.running == 0) {
        if (!g.cause) g.cause = JOIN_ALL_THREADS_EXIT;
        g.started = false;
        ++g.joins;
    }
    g_cv.notify_all();
    return nullptr;
}

Thread* thread_of(uint32_t id) {
    auto it = g_threads.find(id);
    return it == g_threads.end() ? nullptr : it->second.get();
}
Group* group_of(uint32_t id) {
    auto it = g_groups.find(id);
    return it == g_groups.end() ? nullptr : it->second.get();
}
}  // namespace

// ---- images -------------------------------------------------------------------------
// sys_spu_image_t: u32 type, u32 entry_point, u32 segs, s32 nsegs.
// sys_spu_segment_t: s32 type, u32 ls, u32 size, u32 addr or fill value, u32 pad.
static uint32_t image_from_elf(uint32_t img, uint32_t src) {
    if (ld32(src) != 0x7F454C46u || ld16(src + 18) != 23) return CELL_EINVAL;
    uint32_t entry = ld32(src + 24), phoff = ld32(src + 28);
    uint32_t phnum = ld16(src + 44);
    std::vector<Seg> segs;
    for (uint32_t i = 0; i < phnum; ++i) {
        uint32_t ph = src + phoff + 32 * i;
        uint32_t type = ld32(ph), off = ld32(ph + 4), vaddr = ld32(ph + 8), filesz = ld32(ph + 16),
                 memsz = ld32(ph + 20);
        if (type == 1) {
            if (filesz) segs.push_back({SEG_COPY, vaddr, filesz, src + off});
            if (memsz > filesz) segs.push_back({SEG_FILL, vaddr + filesz, memsz - filesz, 0});
        } else if (type == 4) {
            segs.push_back({SEG_INFO, 0, filesz, src + off});
        }
    }
    uint32_t table = hle_alloc((uint32_t)segs.size() * 0x18, 16);
    for (size_t i = 0; i < segs.size(); ++i) {
        uint32_t a = table + 0x18 * (uint32_t)i;
        st32(a, segs[i].type); st32(a + 4, segs[i].ls); st32(a + 8, segs[i].size);
        st32(a + 12, segs[i].addr); st32(a + 16, 0); st32(a + 20, 0);
    }
    st32(img, 0);                                  // SYS_SPU_IMAGE_TYPE_USER
    st32(img + 4, entry);
    st32(img + 8, table);
    st32(img + 12, (uint32_t)segs.size());
    std::lock_guard<std::mutex> lk(g_mx);
    g_image_elf[table] = src;
    return CELL_OK;
}

PS3_HLE(sys_spu_image_import) {                    // (sys_spu_image_t*, const void* src, u32 type)
    uint32_t r = image_from_elf(arg_u32(c, 0), arg_u32(c, 1));
    if (r) rt_log("sputhread: sys_spu_image_import of %08X: not an SPU ELF", arg_u32(c, 1));
    c.r[3] = r;
}

PS3_HLE(sys_spu_image_close) {                     // (sys_spu_image_t*): the table is the runtime's, kept
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(169, sys_spu_initialize) {             // (max usable SPUs, max raw SPUs)
    c.r[3] = CELL_OK;
}

// ---- thread groups ---------------------------------------------------------------------
// attribute: u32 nsize, u32 name, s32 type, u32 container
LV2_SYSCALL(170, sys_spu_thread_group_create) {    // (u32* id, u32 num, s32 prio, attribute*)
    uint32_t num = arg_u32(c, 1), attr = arg_u32(c, 3);
    if (!num || num > 6) { c.r[3] = CELL_EINVAL; return; }
    auto g = std::make_unique<Group>();
    g->num = num;
    g->prio = arg_u32(c, 2);
    if (attr && ld32(attr + 4)) g->name = mem_cstr(ld32(attr + 4), 128);
    g->threads.assign(num, 0);
    std::lock_guard<std::mutex> lk(g_mx);
    g->id = g_next_group++;
    st32(arg_u32(c, 0), g->id);
    rt_log("sputhread: group %08X \"%s\", %u threads, prio %d", g->id, g->name.c_str(), num, (int)g->prio);
    g_groups[g->id] = std::move(g);
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(171, sys_spu_thread_group_destroy) {   // (id)
    std::lock_guard<std::mutex> lk(g_mx);
    Group* g = group_of(arg_u32(c, 0));
    if (!g) { c.r[3] = CELL_ESRCH; return; }
    if (g->running) { c.r[3] = CELL_EBUSY; return; }
    for (uint32_t t : g->threads) g_threads.erase(t);
    g_groups.erase(g->id);
    c.r[3] = CELL_OK;
}

// attribute: u32 name, u32 nsize, u32 option; argument: u64 arg1..arg4
LV2_SYSCALL(172, sys_spu_thread_initialize) {      // (u32* thread, group, spu num, image*, attribute*, argument*)
    uint32_t gid = arg_u32(c, 1), num = arg_u32(c, 2), img = arg_u32(c, 3), attr = arg_u32(c, 4),
             arg = arg_u32(c, 5);
    std::lock_guard<std::mutex> lk(g_mx);
    Group* g = group_of(gid);
    if (!g) { c.r[3] = CELL_ESRCH; return; }
    if (num >= g->num) { c.r[3] = CELL_EINVAL; return; }
    auto t = std::make_unique<Thread>();
    t->group = gid;
    t->num = num;
    if (attr && ld32(attr)) t->name = mem_cstr(ld32(attr), 128);
    t->entry = ld32(img + 4);
    uint32_t table = ld32(img + 8), n = ld32(img + 12);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t a = table + 0x18 * i;
        t->segs.push_back({ld32(a), ld32(a + 4), ld32(a + 8), ld32(a + 12)});
    }
    auto it = g_image_elf.find(table);
    t->elf = it == g_image_elf.end() ? 0 : it->second;
    if (arg)
        for (int i = 0; i < 4; ++i) t->args[i] = ld64(arg + 8 * i);
    t->id = g_next_thread++;
    t->s.ls = t->ls.get();
    st32(arg_u32(c, 0), t->id);
    rt_log("sputhread: thread %08X \"%s\" of group %08X, SPU %u: ELF %08X, entry %05X, %u segments", t->id,
           t->name.c_str(), gid, num, t->elf, t->entry, n);
    g->threads[num] = t->id;
    g_threads[t->id] = std::move(t);
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(173, sys_spu_thread_group_start) {     // (id)
    std::vector<Thread*> start;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        Group* g = group_of(arg_u32(c, 0));
        if (!g) { c.r[3] = CELL_ESRCH; return; }
        if (g->started) { c.r[3] = CELL_ESTAT; return; }
        g->started = true;
        g->kill = false;
        g->cause = g->status = 0;
        for (uint32_t id : g->threads)
            if (Thread* t = thread_of(id)) { t->load(); start.push_back(t); }
        g->running = (int)start.size();
        if (!g->running) { g->started = false; g->cause = JOIN_ALL_THREADS_EXIT; ++g->joins; }
    }
    for (Thread* t : start) {
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, kHostStack);
        pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
        pthread_t h;
        if (pthread_create(&h, &a, thread_main, t)) {
            rt_log("sputhread: cannot start a host thread for %08X", t->id);
            std::abort();
        }
        pthread_attr_destroy(&a);
    }
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(174, sys_spu_thread_group_suspend) {   // (id): not done; the group runs on
    rt_log("sputhread: sys_spu_thread_group_suspend(%08X) is not done; the group runs on", arg_u32(c, 0));
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(177, sys_spu_thread_group_terminate) { // (id, value)
    std::lock_guard<std::mutex> lk(g_mx);
    Group* g = group_of(arg_u32(c, 0));
    if (!g) { c.r[3] = CELL_ESRCH; return; }
    if (!g->started) { c.r[3] = CELL_ESTAT; return; }
    g->cause = JOIN_TERMINATED;
    g->status = arg_u32(c, 1);
    g->kill = true;
    g_cv.notify_all();
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(178, sys_spu_thread_group_join) {      // (id, u32* cause, u32* status)
    std::unique_lock<std::mutex> lk(g_mx);
    Group* g = group_of(arg_u32(c, 0));
    if (!g) { c.r[3] = CELL_ESRCH; return; }
    uint32_t id = g->id;
    if (!g->started && !g->joins) { c.r[3] = CELL_ESTAT; return; }
    g_cv.wait(lk, [&] { Group* x = group_of(id); return !x || !x->started; });
    if (!(g = group_of(id))) { c.r[3] = CELL_ESRCH; return; }
    if (arg_u32(c, 1)) st32(arg_u32(c, 1), g->cause);
    if (arg_u32(c, 2)) st32(arg_u32(c, 2), g->status);
    c.r[3] = CELL_OK;
}

// ---- a thread's local store, mailbox and events, from the PPU ------------------------------
LV2_SYSCALL(181, sys_spu_thread_write_ls) {        // (thread, lsa, u64 value, type: 1, 2, 4 or 8)
    std::lock_guard<std::mutex> lk(g_mx);
    Thread* t = thread_of(arg_u32(c, 0));
    uint32_t lsa = arg_u32(c, 1), type = arg_u32(c, 3);
    if (!t) { c.r[3] = CELL_ESRCH; return; }
    if (lsa >= SPU_LS_SIZE || (type != 1 && type != 2 && type != 4 && type != 8) || lsa % type) {
        c.r[3] = CELL_EINVAL;
        return;
    }
    uint64_t v = arg_u64(c, 2);
    for (uint32_t i = 0; i < type; ++i) t->ls[lsa + i] = (uint8_t)(v >> (8 * (type - 1 - i)));
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(182, sys_spu_thread_read_ls) {         // (thread, lsa, u64* value, type)
    std::lock_guard<std::mutex> lk(g_mx);
    Thread* t = thread_of(arg_u32(c, 0));
    uint32_t lsa = arg_u32(c, 1), type = arg_u32(c, 3);
    if (!t) { c.r[3] = CELL_ESRCH; return; }
    if (lsa >= SPU_LS_SIZE || (type != 1 && type != 2 && type != 4 && type != 8) || lsa % type) {
        c.r[3] = CELL_EINVAL;
        return;
    }
    uint64_t v = 0;
    for (uint32_t i = 0; i < type; ++i) v = v << 8 | t->ls[lsa + i];
    st64(arg_u32(c, 2), v);
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(187, sys_spu_thread_set_spu_cfg) {     // (thread, u64 value): the signal notification modes
    std::lock_guard<std::mutex> lk(g_mx);
    Thread* t = thread_of(arg_u32(c, 0));
    if (!t) { c.r[3] = CELL_ESRCH; return; }
    t->cfg = arg_u64(c, 1);
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(190, sys_spu_thread_write_spu_mb) {    // (thread, u32 value)
    std::unique_lock<std::mutex> lk(g_mx);
    Thread* t = thread_of(arg_u32(c, 0));
    if (!t) { c.r[3] = CELL_ESRCH; return; }
    uint32_t id = t->id;
    g_cv.wait(lk, [&] { Thread* x = thread_of(id); return !x || x->in_mbox.size() < 4; });
    if (!(t = thread_of(id))) { c.r[3] = CELL_ESRCH; return; }
    t->in_mbox.push_back(arg_u32(c, 1));
    g_cv.notify_all();
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(191, sys_spu_thread_connect_event) {   // (thread, event queue, event type, spup)
    std::lock_guard<std::mutex> lk(g_mx);
    Thread* t = thread_of(arg_u32(c, 0));
    uint32_t q = arg_u32(c, 1), et = arg_u32(c, 2), port = arg_u32(c, 3);
    if (!t) { c.r[3] = CELL_ESRCH; return; }
    if (et != 1 || port > 63) {                    // SYS_SPU_THREAD_EVENT_USER only
        rt_log("sputhread %08X: connect_event type %u port %u not handled", t->id, et, port);
        c.r[3] = CELL_EINVAL;
        return;
    }
    if (t->spup[port]) { c.r[3] = CELL_EISCONN; return; }
    t->spup[port] = q;
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(192, sys_spu_thread_disconnect_event) {    // (thread, event type, spup)
    std::lock_guard<std::mutex> lk(g_mx);
    Thread* t = thread_of(arg_u32(c, 0));
    uint32_t port = arg_u32(c, 2);
    if (!t) { c.r[3] = CELL_ESRCH; return; }
    if (port > 63) { c.r[3] = CELL_EINVAL; return; }
    if (!t->spup[port]) { c.r[3] = CELL_ENOTCONN; return; }
    t->spup[port] = 0;
    c.r[3] = CELL_OK;
}

LV2_SYSCALL(193, sys_spu_thread_bind_queue) {      // (thread, event queue, spuq number)
    std::lock_guard<std::mutex> lk(g_mx);
    Thread* t = thread_of(arg_u32(c, 0));
    if (!t) { c.r[3] = CELL_ESRCH; return; }
    uint32_t num = arg_u32(c, 2);
    if (t->spuq.count(num)) { c.r[3] = CELL_EISCONN; return; }
    t->spuq[num] = arg_u32(c, 1);
    c.r[3] = CELL_OK;
}

// (group, event queue, u64 bitmask of the ports it may take, u8* the port taken)
LV2_SYSCALL(251, sys_spu_thread_group_connect_event_all_threads) {
    std::lock_guard<std::mutex> lk(g_mx);
    Group* g = group_of(arg_u32(c, 0));
    if (!g) { c.r[3] = CELL_ESRCH; return; }
    uint64_t req = arg_u64(c, 2);
    for (uint32_t port = 0; port < 64; ++port) {
        if (!(req >> port & 1)) continue;
        bool free = true;
        for (uint32_t id : g->threads)
            if (Thread* t = thread_of(id)) free &= !t->spup[port];
        if (!free) continue;
        for (uint32_t id : g->threads)
            if (Thread* t = thread_of(id)) t->spup[port] = arg_u32(c, 1);
        st8(arg_u32(c, 3), (uint8_t)port);
        c.r[3] = CELL_OK;
        return;
    }
    c.r[3] = CELL_EBUSY;
}

LV2_SYSCALL(252, sys_spu_thread_group_disconnect_event_all_threads) {  // (group, u8 port)
    std::lock_guard<std::mutex> lk(g_mx);
    Group* g = group_of(arg_u32(c, 0));
    uint32_t port = arg_u32(c, 1);
    if (!g) { c.r[3] = CELL_ESRCH; return; }
    if (port > 63) { c.r[3] = CELL_EINVAL; return; }
    for (uint32_t id : g->threads)
        if (Thread* t = thread_of(id)) t->spup[port] = 0;
    c.r[3] = CELL_OK;
}
