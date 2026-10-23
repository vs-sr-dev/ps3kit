// ps3kit runtime — cellSpurs: SPURS instances, task sets and their tasks,
// task exit codes, and event flags.
//
// The kernel here is the runtime's own, and it does not schedule: each task
// runs on a host thread of its own from the moment it is created, in a local
// store of its own, as many at once as the game creates (the system runs
// them on at most as many SPUs as the instance was given). A task is an SPU
// ELF, loaded from 0x3000 up as the taskset policy module loads it, with the
// module's context at 0x2700 and the kernel's at 0x100 filled in where the
// task's library reads them (the layouts are the system's, as RPCS3's
// cellSpurs.h describes them):
//   kernel  0x1C0 the instance (u64)  0x1C8 SPU number  0x1CC DMA tag
//           0x1D0 the workload (the taskset, u64)  0x1DC workload id
//   module  0x2700 the taskset's first 128 bytes  0x2780 the task's info
//           0x27B8 the taskset (u64)  0x27C0 the kernel context's address
//           0x27C4 the syscall entry  0x27CC SPU number  0x27D0 DMA tag
//           0x27D4 task id  0x2840 "SPURSTASK MODULE"  0x2FB8 0x2700
//           0x2FBC the task's lowest load address  0x2FD0 the exit code
// The task starts at its ELF's entry with the task argument in $3 and, in
// $4, the taskset's argument (preferred doubleword) and the instance. It
// calls the module through the syscall entry, 0xA70, with the number in $3
// and its argument in $4, and is resumed at $0 with the result in $3:
//   0 exit (the exit code is the module context's), 1 yield,
//   2 wait for a signal (the taskset's `signalled` bit for the task, which is
//     polled and cleared), 3 poll (nothing is ever found), 4 the workload
//     flag (reported).
// An event a task sends through the interrupt mailbox goes to the event
// queue attached to that SPU port of the instance, as an SPU thread's would.
//
// Event flags keep the system's 128-byte layout, because the SPU side of the
// protocol is the tasks' own code: a PPU waiter blocks on the queue attached
// to the flag, a waiting task is woken by its signal.
//
// Lock-free queues keep the system's layout too (CellSyncLFQueue, 128 bytes),
// for the same reason: the tasks pop them with their own code. The PPU pushes;
// a task that found the queue empty has left its signal id in the queue, and
// the push that fills it signals the task.
//
// Queues keep the layout the tasks' own code reads too; the PPU pops and
// pushes them, and wakes the tasks waiting on them.
//
// Barriers are not implemented yet: they are reported and answer CELL_OK
// having done nothing.
#include "rt.h"
#include "spu.h"
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr uint32_t TASK_ERROR_AGAIN = 0x80410901u, TASK_ERROR_INVAL = 0x80410902u, TASK_ERROR_STAT = 0x8041090Fu,
                   TASK_ERROR_BUSY = 0x8041090Au, TASK_ERROR_PERM = 0x80410909u, TASK_ERROR_SRCH = 0x80410905u,
                   TASK_ERROR_NULL_POINTER = 0x80410911u;
constexpr uint32_t kSyscall = 0xA70, kTaskTop = 0x3000;
constexpr size_t kHostStack = 64u << 20;
constexpr uint64_t SYS_SPU_THREAD_EVENT_USER_KEY = 0xFFFFFFFF53505501ull;

std::mutex g_mx;
std::condition_variable g_cv;
const bool g_trace = std::getenv("PS3KIT_SPURS") != nullptr;

struct Instance {
    uint32_t addr, nspus = 0;
    std::string prefix;
    uint32_t next_wid = 0;
    std::map<uint32_t, uint32_t> port_queue;       // SPU port -> event queue
};
std::map<uint32_t, Instance> g_spurs;               // by its CellSpurs address

struct Taskset {
    uint32_t addr, spurs, wid;
    std::string name;
    bool shutdown = false;
    int live = 0;
    bool used[128] = {};                           // task ids taken; an exited task frees its id
};
std::map<uint32_t, Taskset> g_tasksets;

// CellSpursTaskExitCode, the runtime's layout: +0 u32 1 once the task has
// exited, +4 s32 its exit code.
void exit_code_set(uint32_t container, uint32_t code) {
    if (!container) return;
    std::lock_guard<std::mutex> lk(g_mx);
    st32(container + 4, code);
    st32(container, 1);
    g_cv.notify_all();
}

uint32_t get32(const uint8_t* ls, uint32_t a);

// The taskset's bitsets: bit `id` is (0x80000000 >> id % 32) of word id / 32.
uint32_t bit_word(uint32_t bitset, uint32_t id) { return bitset + 4 * (id / 32); }
uint32_t bit_mask(uint32_t id) { return 0x80000000u >> (id % 32); }
constexpr uint32_t TS_RUNNING = 0x00, TS_READY = 0x10, TS_ENABLED = 0x30, TS_SIGNALLED = 0x40, TS_WAITING = 0x50;

void bit_set(uint32_t taskset, uint32_t set, uint32_t id, bool on) {
    std::lock_guard<std::mutex> lk(spu_atomic_mutex());
    uint32_t a = bit_word(taskset + set, id), v = ld32(a);
    st32(a, on ? v | bit_mask(id) : v & ~bit_mask(id));
}

struct Task : SPUHost {
    uint32_t id, taskset, spurs, elf, exit_container;
    uint8_t arg[16];
    SPUContext s{};
    std::unique_ptr<uint8_t[]> ls{new uint8_t[SPU_LS_SIZE]()};
    std::vector<uint32_t> in_mbox;
    uint32_t out_mbox = 0;
    bool abandoned = false;                        // left waiting when its taskset was shut down
    bool out_full = false;

    bool rdch(SPUContext&, uint32_t ch, uint32_t& v) override {
        if (ch != 29 || in_mbox.empty()) {         // SPU_RdInMbox: only the answers the kernel posts
            if (ch == 29) rt_log("spurs task %u: reads its inbound mailbox with nothing posted", id);
            return false;
        }
        v = in_mbox.front();
        in_mbox.erase(in_mbox.begin());
        return true;
    }
    bool rchcnt(SPUContext&, uint32_t ch, uint32_t& n) override {
        if (ch == 29) { n = (uint32_t)in_mbox.size(); return true; }
        if (ch == 28) { n = out_full ? 0 : 1; return true; }
        return false;
    }
    bool wrch(SPUContext& s, uint32_t ch, uint32_t v) override {
        if (ch == 28) { out_mbox = v; out_full = true; return true; }
        if (ch != 30) return false;
        uint32_t code = v >> 24, port = code & 63, q = 0;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            auto it = g_spurs.find(spurs);
            if (it != g_spurs.end() && it->second.port_queue.count(port)) q = it->second.port_queue[port];
        }
        uint32_t data = out_full ? out_mbox : 0;
        out_full = false;
        if (port == 1 && !q) {                     // spu_printf, with no server to print it
            rt_log("spurs task %u: spu_printf: %s", id, spu_printf_text(s, data).c_str());
            if (g_trace) {
                uint32_t b = data & 0x3FFF0;
                for (uint32_t k = 0; k < 64; k += 16)
                    rt_log("   %05X: %08X %08X %08X %08X", b + k, get32(s.ls, b + k), get32(s.ls, b + k + 4),
                           get32(s.ls, b + k + 8), get32(s.ls, b + k + 12));
            }
            if (code < 64) in_mbox.push_back(CELL_OK);
            return true;
        }
        bool ok = q && lv2_event_queue_push(q, SYS_SPU_THREAD_EVENT_USER_KEY, id, (uint64_t)port << 32 | (v & 0xFFFFFF), data);
        if (g_trace || !ok)
            rt_log("spurs task %u: event on SPU port %u (%06X, %08X) %s", id, port, v & 0xFFFFFF, data,
                   ok ? "sent" : q ? "not sent, the queue is full" : "not sent, no queue attached");
        if (code < 64) in_mbox.push_back(ok ? CELL_OK : 0x80010016u);
        return true;
    }
    bool stop(SPUContext&, uint32_t pc, uint32_t code) override {
        rt_log("spurs task %u: stop %X at %05X; the task ends", id, code, pc);
        return true;
    }

    // A shut-down workload is no longer scheduled: a task that yields or
    // waits is never resumed, and ends there without an exit code.
    bool shut_down() {
        std::lock_guard<std::mutex> lk(g_mx);
        auto it = g_tasksets.find(taskset);
        abandoned = it == g_tasksets.end() || it->second.shutdown;
        return abandoned;
    }

    // The module's answer to a syscall; false when the task has exited or
    // been abandoned.
    bool syscall() {
        uint32_t num = s.r[3].u32[3], arg4 = s.r[4].u32[3];
        if (g_trace) rt_log("spurs task %u: syscall %X (%08X)", id, num, arg4);
        switch (num & 0x0F) {
        case 0:                                    // exit
            return false;
        case 1:                                    // yield
            if (shut_down()) return false;
            std::this_thread::yield();
            s.r[3] = spu_slot(0);
            return true;
        case 2:                                    // wait for a signal
            bit_set(taskset, TS_WAITING, id, true);
            for (;;) {
                {
                    std::lock_guard<std::mutex> lk(spu_atomic_mutex());
                    uint32_t a = bit_word(taskset + TS_SIGNALLED, id), w = ld32(a);
                    if (w & bit_mask(id)) {
                        st32(a, w & ~bit_mask(id));
                        uint32_t wa = bit_word(taskset + TS_WAITING, id);
                        st32(wa, ld32(wa) & ~bit_mask(id));
                        break;
                    }
                }
                if (shut_down()) return false;
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            s.r[3] = spu_slot(0);
            return true;
        case 3:                                    // poll
            s.r[3] = spu_slot(0);
            return true;
        default:
            rt_log("spurs task %u: syscall %X (%08X) not handled; it answers 0", id, num, arg4);
            s.r[3] = spu_slot(0);
            return true;
        }
    }
};

// The task's ELF into its local store; the entry point and the lowest load
// address, or false if it is not an SPU ELF.
bool load_elf(uint8_t* ls, uint32_t elf, uint32_t& entry, uint32_t& lowest) {
    if (ld32(elf) != 0x7F454C46u || ld16(elf + 18) != 23) return false;
    entry = ld32(elf + 24);
    lowest = SPU_LS_SIZE;
    uint32_t phoff = ld32(elf + 28), phnum = ld16(elf + 44);
    for (uint32_t i = 0; i < phnum; ++i) {
        uint32_t ph = elf + phoff + 32 * i;
        if (ld32(ph) != 1) continue;
        uint32_t off = ld32(ph + 4), vaddr = ld32(ph + 8), filesz = ld32(ph + 16), memsz = ld32(ph + 20);
        if (vaddr + memsz > SPU_LS_SIZE) return false;
        std::memcpy(ls + vaddr, g_mem + elf + off, filesz);
        std::memset(ls + vaddr + filesz, 0, memsz - filesz);
        if (vaddr < lowest) lowest = vaddr;
    }
    return true;
}

void put64(uint8_t* ls, uint32_t a, uint64_t v) { for (int i = 0; i < 8; ++i) ls[a + i] = (uint8_t)(v >> (56 - 8 * i)); }
void put32(uint8_t* ls, uint32_t a, uint32_t v) { for (int i = 0; i < 4; ++i) ls[a + i] = (uint8_t)(v >> (24 - 8 * i)); }
uint32_t get32(const uint8_t* ls, uint32_t a) { return (uint32_t)ls[a] << 24 | ls[a + 1] << 16 | ls[a + 2] << 8 | ls[a + 3]; }

void* task_main(void* p) {
    std::unique_ptr<Task> t(static_cast<Task*>(p));
    const SPUProgram* prog = spu_program_of(t->elf);
    uint32_t exit_code = 0;
    uint32_t entry = 0, lowest = 0;
    uint8_t* ls = t->ls.get();
    if (!prog || !load_elf(ls, t->elf, entry, lowest)) {
        rt_log("spurs task %u: the SPU ELF at %08X %s; the task exits at once", t->id, t->elf,
               prog ? "does not load" : "was not recompiled");
    } else {
        // the kernel's context and the module's
        put64(ls, 0x1C0, t->spurs);
        put32(ls, 0x1C8, 0);
        put32(ls, 0x1CC, 31);
        put64(ls, 0x1D0, t->taskset);
        put32(ls, 0x1DC, g_tasksets[t->taskset].wid);
        ls[0x1E8] = 'T'; ls[0x1E9] = 'K';
        std::memcpy(ls + 0x2700, g_mem + t->taskset, 0x80);
        std::memcpy(ls + 0x2780, g_mem + t->taskset + 0x80 + 0x30 * t->id, 0x30);
        put64(ls, 0x27B8, t->taskset);
        put32(ls, 0x27C0, 0x100);
        put32(ls, 0x27C4, kSyscall);
        put32(ls, 0x27CC, 0);
        put32(ls, 0x27D0, 31);
        put32(ls, 0x27D4, t->id);
        std::memcpy(ls + 0x2840, "SPURSTASK MODULE", 16);
        put32(ls, 0x2FB8, 0x2700);
        put32(ls, 0x2FBC, lowest);
        put32(ls, 0x2FD0, 0);
        SPUContext& s = t->s;
        s.ls = ls;
        s.id = t->id;
        s.host = t.get();
        s.raddr = ~0u;
        s.r[1] = spu_slot(0x3FFF0);
        for (int i = 0; i < 4; ++i)
            s.r[3].u32[3 - i] = (uint32_t)t->arg[4 * i] << 24 | t->arg[4 * i + 1] << 16 | t->arg[4 * i + 2] << 8 | t->arg[4 * i + 3];
        s.r[4].u64[1] = ld64(t->taskset + 0x68);
        s.r[4].u64[0] = t->spurs;
        if (g_trace) rt_log("spurs task %u of %08X: %s, entry %05X", t->id, t->taskset, prog->name, entry);
        uint32_t pc = entry;
        for (;;) {
            pc = spu_run(s, prog, pc);
            if (pc == SPU_STOPPED) break;
            if (pc == kSyscall) {
                if (!t->syscall()) { exit_code = get32(ls, 0x2FD0); break; }
                pc = s.r[0].u32[3] & 0x3FFFC;
                continue;
            }
            rt_log("spurs task %u (%s): branched out of its program to %05X; the task ends", t->id, prog->name, pc);
            break;
        }
    }
    bit_set(t->taskset, TS_RUNNING, t->id, false);
    bit_set(t->taskset, TS_ENABLED, t->id, false);
    if (t->abandoned) rt_log("spurs task %u of %08X: its taskset is shut down; the task is not resumed", t->id, t->taskset);
    else exit_code_set(t->exit_container, exit_code);
    if (g_trace && !t->abandoned) rt_log("spurs task %u of %08X: exited with %d", t->id, t->taskset, (int)exit_code);
    std::lock_guard<std::mutex> lk(g_mx);
    Taskset& ts = g_tasksets[t->taskset];
    ts.used[t->id] = false;
    --ts.live;
    g_cv.notify_all();
    return nullptr;
}

uint32_t create_task(uint32_t taskset, uint32_t id_out, uint32_t elf, uint32_t context, uint32_t size,
                     uint32_t ls_pattern, uint32_t argument, uint32_t exit_container) {
    auto t = new Task;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        auto it = g_tasksets.find(taskset);
        if (it == g_tasksets.end()) { delete t; return TASK_ERROR_SRCH; }
        Taskset& ts = it->second;
        uint32_t id = 0;
        while (id < 128 && ts.used[id]) ++id;
        if (id == 128) { delete t; return TASK_ERROR_AGAIN; }
        ts.used[id] = true;
        t->id = id;
        t->spurs = ts.spurs;
        ++ts.live;
    }
    t->taskset = taskset;
    t->elf = elf & ~7u;
    t->exit_container = exit_container;
    if (argument) std::memcpy(t->arg, g_mem + argument, 16);
    else std::memset(t->arg, 0, 16);
    // the task's info in the taskset, as the system's library fills it: its
    // argument, its ELF, its context save area with the number of 2 KB local
    // store blocks it can hold in the low bits, and the pattern of the blocks
    // to save. A task that blocks checks them (cellSpursEventFlagWait answers
    // CELL_SPURS_TASK_ERROR_STAT without a context that covers its stack).
    uint32_t info = taskset + 0x80 + 0x30 * t->id;
    uint32_t blocks = !context ? 0 : size > 0x3D400 ? 0x7A : (size - 0x400) >> 11;
    std::memcpy(g_mem + info, t->arg, 16);
    st64(info + 0x10, elf);
    st64(info + 0x18, context | blocks);
    if (ls_pattern) std::memcpy(g_mem + info + 0x20, g_mem + ls_pattern, 16);
    else std::memset(g_mem + info + 0x20, 0, 16);
    bit_set(taskset, TS_ENABLED, t->id, true);
    bit_set(taskset, TS_RUNNING, t->id, true);
    if (id_out) st32(id_out, t->id);
    rt_log("spurs: task %u of taskset %08X, ELF %08X%s", t->id, taskset, t->elf,
           spu_program_of(t->elf) ? "" : " (not recompiled)");
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, kHostStack);
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    pthread_t h;
    if (pthread_create(&h, &a, task_main, t)) {
        rt_log("spurs: cannot start a host thread for task %u", t->id);
        std::abort();
    }
    pthread_attr_destroy(&a);
    return CELL_OK;
}

// Signal a task: the taskset's `signalled` bit, which a waiting task polls.
void send_signal(uint32_t taskset, uint32_t task) {
    bit_set(taskset, TS_SIGNALLED, task, true);
}
}  // namespace

// ---- the instance ---------------------------------------------------------------------
// CellSpursAttribute, the runtime's layout: +0 revision, +4 sdk version,
// +8 SPUs, +0xC SPU thread priority, +0x10 PPU priority, +0x14 exit if no
// work, +0x20 the name prefix (15 + NUL), +0x30 its length, +0x34 flags.
PS3_HLE(_cellSpursAttributeInitialize) {   // (attr, revision, sdk, nSpus, spuPriority, ppuPriority, exitIfNoWork)
    uint32_t a = arg_u32(c, 0);
    std::memset(g_mem + a, 0, 0x200);
    for (int i = 1; i <= 6; ++i) st32(a + 4 * (i - 1), arg_u32(c, i));
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursAttributeSetNamePrefix) {  // (attr, prefix, size)
    uint32_t a = arg_u32(c, 0), n = arg_u32(c, 2);
    if (n > 15) { c.r[3] = 0x80410702u; return; }   // CELL_SPURS_CORE_ERROR_INVAL
    std::memset(g_mem + a + 0x20, 0, 16);
    std::memcpy(g_mem + a + 0x20, g_mem + arg_u32(c, 1), n);
    st32(a + 0x30, n);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursAttributeEnableSpuPrintfIfAvailable) {   // (attr): no SPU printf server here
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursInitializeWithAttribute) {  // (spurs, attr)
    uint32_t sp = arg_u32(c, 0), a = arg_u32(c, 1);
    Instance in;
    in.addr = sp;
    in.nspus = ld32(a + 8);
    in.prefix = mem_cstr(a + 0x20, 16);
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_spurs[sp] = in;
    }
    rt_log("spurs: instance %08X \"%s\", %u SPUs", sp, in.prefix.c_str(), in.nspus);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursFinalize) {                // (spurs)
    std::lock_guard<std::mutex> lk(g_mx);
    g_spurs.erase(arg_u32(c, 0));
    c.r[3] = CELL_OK;
}

// ---- task sets -------------------------------------------------------------------------
// CellSpursTasksetAttribute, the runtime's layout: +0 revision, +4 sdk,
// +8 argument (u64), +0x10 priorities (8 bytes), +0x18 max contention,
// +0x1C name, +0x20 taskset size.
PS3_HLE(_cellSpursTasksetAttributeInitialize) {  // (attr, revision, sdk, args u64, priority*, maxContention)
    uint32_t a = arg_u32(c, 0);
    std::memset(g_mem + a, 0, 0x80);
    st32(a, arg_u32(c, 1));
    st32(a + 4, arg_u32(c, 2));
    st64(a + 8, arg_u64(c, 3));
    if (arg_u32(c, 4)) std::memcpy(g_mem + a + 0x10, g_mem + arg_u32(c, 4), 8);
    st32(a + 0x18, arg_u32(c, 5));
    st32(a + 0x20, 6400);                           // sizeof(CellSpursTaskset)
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursTasksetAttributeSetName) {      // (attr, name)
    st32(arg_u32(c, 0) + 0x1C, arg_u32(c, 1));
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursTasksetAttributeSetTasksetSize) {   // (attr, size)
    st32(arg_u32(c, 0) + 0x20, arg_u32(c, 1));
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursCreateTasksetWithAttribute) {   // (spurs, taskset, attr)
    uint32_t sp = arg_u32(c, 0), tsa = arg_u32(c, 1), a = arg_u32(c, 2);
    uint32_t size = ld32(a + 0x20);
    std::memset(g_mem + tsa, 0, size < 6400 ? 6400 : size);
    st64(tsa + 0x60, sp);
    st64(tsa + 0x68, ld64(a + 8));
    st32(tsa + 0x1890, size);
    Taskset ts;
    ts.addr = tsa;
    ts.spurs = sp;
    if (ld32(a + 0x1C)) ts.name = mem_cstr(ld32(a + 0x1C), 64);
    {
        std::lock_guard<std::mutex> lk(g_mx);
        ts.wid = g_spurs.count(sp) ? g_spurs[sp].next_wid++ : 0;
        g_tasksets[tsa] = ts;
    }
    st32(tsa + 0x74, ts.wid);
    // the workload's argument in the instance (wklInfo1[wid].arg), where a task
    // finds the taskset of a workload it signals
    if (ts.wid < 16) st64(sp + 0xB00 + 0x20 * ts.wid + 8, tsa);
    rt_log("spurs: taskset %08X \"%s\" on %08X, workload %u", tsa, ts.name.c_str(), sp, ts.wid);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursShutdownTaskset) {              // (taskset)
    std::lock_guard<std::mutex> lk(g_mx);
    auto it = g_tasksets.find(arg_u32(c, 0));
    if (it == g_tasksets.end()) { c.r[3] = TASK_ERROR_SRCH; return; }
    it->second.shutdown = true;
    g_cv.notify_all();
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursJoinTaskset) {                  // (taskset): until it is shut down and its tasks have exited
    uint32_t tsa = arg_u32(c, 0);
    std::unique_lock<std::mutex> lk(g_mx);
    if (!g_tasksets.count(tsa)) { c.r[3] = TASK_ERROR_SRCH; return; }
    g_cv.wait(lk, [&] { Taskset& t = g_tasksets[tsa]; return t.shutdown && t.live == 0; });
    g_tasksets.erase(tsa);
    c.r[3] = CELL_OK;
}

// ---- tasks -----------------------------------------------------------------------------
// CellSpursTaskAttribute, the runtime's layout: +0 revision, +4 sdk,
// +8 ELF, +0xC context, +0x10 context size, +0x14 exit code container,
// +0x20 LS pattern (16 bytes), +0x30 argument (16 bytes).
// (attr, revision, sdk, elf, saveConfig*, argument*). The signature is read
// off a call site (the SDK's inline cellSpursTaskAttributeInitialize passes
// these); saveConfig is {context, its size, the LS pattern's address}.
PS3_HLE(_cellSpursTaskAttributeInitialize) {
    uint32_t a = arg_u32(c, 0), save = arg_u32(c, 4), arg = arg_u32(c, 5);
    std::memset(g_mem + a, 0, 0x80);
    st32(a, arg_u32(c, 1));
    st32(a + 4, arg_u32(c, 2));
    st32(a + 8, arg_u32(c, 3));
    if (save) {
        st32(a + 0xC, ld32(save));
        st32(a + 0x10, ld32(save + 4));
        if (ld32(save + 8)) std::memcpy(g_mem + a + 0x20, g_mem + ld32(save + 8), 16);
    }
    if (arg) std::memcpy(g_mem + a + 0x30, g_mem + arg, 16);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursTaskAttributeSetExitCodeContainer) {   // (attr, container)
    st32(arg_u32(c, 0) + 0x14, arg_u32(c, 1));
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursCreateTaskWithAttribute) {      // (taskset, u32* id, attr)
    uint32_t a = arg_u32(c, 2);
    c.r[3] = create_task(arg_u32(c, 0), arg_u32(c, 1), ld32(a + 8), ld32(a + 0xC), ld32(a + 0x10), a + 0x20, a + 0x30,
                         ld32(a + 0x14));
}

PS3_HLE(cellSpursCreateTask) {   // (taskset, u32* id, elf, context, size, lsPattern*, argument*)
    c.r[3] = create_task(arg_u32(c, 0), arg_u32(c, 1), arg_u32(c, 2), arg_u32(c, 3), arg_u32(c, 4), arg_u32(c, 5),
                         arg_u32(c, 6), 0);
}

PS3_HLE(cellSpursTaskExitCodeInitialize) {       // (container)
    std::lock_guard<std::mutex> lk(g_mx);
    std::memset(g_mem + arg_u32(c, 0), 0, 16);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursTaskExitCodeGet) {              // (container, u32* code): once the task has exited
    uint32_t box = arg_u32(c, 0);
    std::unique_lock<std::mutex> lk(g_mx);
    g_cv.wait(lk, [&] { return ld32(box) != 0; });
    if (arg_u32(c, 1)) st32(arg_u32(c, 1), ld32(box + 4));
    c.r[3] = CELL_OK;
}

// ---- event flags -----------------------------------------------------------------------
// CellSpursEventFlag (128 bytes): +0 events (u16), +2 spuTaskPendingRecv,
// +4 ppuWaitMask, +6 ppuWaitSlotAndMode, +7 ppuPendingRecv, +8
// spuTaskUsedWaitSlots, +0xA spuTaskWaitMode, +0xC spuPort, +0xD isIwl,
// +0xE direction, +0xF clearMode, +0x10 spuTaskWaitMask[16], +0x30
// pendingRecvTaskEvents[16], +0x50 waitingTaskId[16], +0x60
// waitingTaskWklId[16], +0x70 the taskset or instance (u64), +0x78 event
// port, +0x7C event queue.
namespace {
constexpr uint32_t EF_SPU2SPU = 0, EF_SPU2PPU = 1, EF_PPU2SPU = 2, EF_ANY2ANY = 3;
constexpr uint32_t EF_OR = 0, EF_CLEAR_AUTO = 0;
}

PS3_HLE(_cellSpursEventFlagInitialize) {    // (spurs, taskset, eventFlag, clearMode, direction)
    uint32_t sp = arg_u32(c, 0), ts = arg_u32(c, 1), ef = arg_u32(c, 2);
    if ((!ts && !sp) || !ef) { c.r[3] = TASK_ERROR_NULL_POINTER; return; }
    if (arg_u32(c, 4) > EF_ANY2ANY || arg_u32(c, 3) > 1) { c.r[3] = TASK_ERROR_INVAL; return; }
    std::lock_guard<std::mutex> lk(spu_atomic_mutex());
    std::memset(g_mem + ef, 0, 128);
    st8(ef + 0xE, (uint8_t)arg_u32(c, 4));
    st8(ef + 0xF, (uint8_t)arg_u32(c, 3));
    st8(ef + 0xC, 0xFF);
    if (ts) st64(ef + 0x70, ts);
    else { st8(ef + 0xD, 1); st64(ef + 0x70, sp); }
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursEventFlagAttachLv2EventQueue) {   // (eventFlag): a queue for the PPU waiter
    uint32_t ef = arg_u32(c, 0);
    uint32_t dir = ld8(ef + 0xE);
    if (dir != EF_SPU2PPU && dir != EF_ANY2ANY) { c.r[3] = TASK_ERROR_PERM; return; }
    if (ld8(ef + 0xC) != 0xFF) { c.r[3] = TASK_ERROR_STAT; return; }
    uint32_t q = lv2_event_queue_new(127);
    uint32_t owner = (uint32_t)ld64(ef + 0x70);
    uint32_t sp = ld8(ef + 0xD) ? owner : (uint32_t)ld64(owner + 0x60);
    uint32_t port = 0xFF;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        Instance& in = g_spurs[sp];
        for (uint32_t p = 16; p < 64 && port == 0xFF; ++p)   // the low ports are the system's
            if (!in.port_queue.count(p)) port = p;
        if (port != 0xFF) in.port_queue[port] = q;
    }
    if (port == 0xFF) { c.r[3] = TASK_ERROR_AGAIN; return; }
    std::lock_guard<std::mutex> lk(spu_atomic_mutex());
    st8(ef + 0xC, (uint8_t)port);
    st32(ef + 0x78, 0);
    st32(ef + 0x7C, q);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursEventFlagDetachLv2EventQueue) {   // (eventFlag)
    uint32_t ef = arg_u32(c, 0);
    uint32_t port = ld8(ef + 0xC);
    if (port == 0xFF) { c.r[3] = TASK_ERROR_STAT; return; }
    uint32_t owner = (uint32_t)ld64(ef + 0x70);
    uint32_t sp = ld8(ef + 0xD) ? owner : (uint32_t)ld64(owner + 0x60);
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_spurs[sp].port_queue.erase(port);
    }
    std::lock_guard<std::mutex> lk(spu_atomic_mutex());
    st8(ef + 0xC, 0xFF);
    st32(ef + 0x7C, 0);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursEventFlagSet) {            // (eventFlag, u16 bits)
    uint32_t ef = arg_u32(c, 0);
    uint16_t bits = (uint16_t)arg_u32(c, 1);
    uint32_t dir = ld8(ef + 0xE);
    if (dir != EF_PPU2SPU && dir != EF_ANY2ANY) { c.r[3] = TASK_ERROR_PERM; return; }
    bool send = false;
    uint16_t pending = 0, pending_events[16] = {}, ppu_events = 0;
    uint32_t ppu_slot = 0;
    {
        std::lock_guard<std::mutex> lk(spu_atomic_mutex());
        uint16_t events = ld16(ef), spu_pending = ld16(ef + 2), ppu_mask = ld16(ef + 4);
        uint8_t ppu_slot_mode = ld8(ef + 6);
        uint16_t used = ld16(ef + 8), modes = ld16(ef + 0xA);
        uint16_t clear = 0;
        if (dir == EF_ANY2ANY && ppu_mask) {
            uint16_t rel = (events | bits) & ppu_mask;
            if (!(ppu_mask & ~rel) || ((ppu_slot_mode & 0xF) == EF_OR && rel)) {
                st8(ef + 7, 1);
                st16(ef + 4, 0);
                ppu_events = rel;
                clear = rel;
                ppu_slot = ppu_slot_mode >> 4;
                send = true;
            }
        }
        uint16_t slots = used & ~spu_pending;
        for (int i = 15, j = 0; slots; slots >>= 1, --i, ++j) {
            if (!(slots & 1)) continue;
            uint16_t mask = ld16(ef + 0x10 + 2 * i), rel = (events | bits) & mask;
            if (!(mask & ~rel) || (((modes >> j) & 1) == EF_OR && rel)) {
                clear |= rel;
                pending |= 1 << j;
                pending_events[j] = rel;
            }
        }
        events |= bits;
        if (ld8(ef + 0xF) == EF_CLEAR_AUTO) events &= ~clear;
        st16(ef, events);
        st16(ef + 2, spu_pending | pending);
        if (send) st16(ef + 0x30 + 2 * ppu_slot, ppu_events);
        for (int i = 0; i < 16; ++i)
            if (pending & (0x8000 >> i)) st16(ef + 0x30 + 2 * i, pending_events[i]);
    }
    if (send) lv2_event_queue_push(ld32(ef + 0x7C), 0, 0, 0, 0);
    for (int i = 0; i < 16; ++i)
        if (pending & (0x8000 >> i)) {
            uint32_t owner = (uint32_t)ld64(ef + 0x70);
            if (ld8(ef + 0xD)) rt_log("spurs: event flag %08X wakes a task by workload, not done", ef);
            else send_signal(owner, ld8(ef + 0x50 + i));
        }
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursEventFlagWait) {           // (eventFlag, u16* mask, mode)
    uint32_t ef = arg_u32(c, 0), maskp = arg_u32(c, 1), mode = arg_u32(c, 2);
    uint32_t dir = ld8(ef + 0xE);
    if (dir != EF_SPU2PPU && dir != EF_ANY2ANY) { c.r[3] = TASK_ERROR_PERM; return; }
    if (ld8(ef + 0xC) == 0xFF) { c.r[3] = TASK_ERROR_STAT; return; }
    uint16_t mask = ld16(maskp);
    bool wait = false;
    {
        std::lock_guard<std::mutex> lk(spu_atomic_mutex());
        if (ld16(ef + 4) || ld8(ef + 7)) { c.r[3] = TASK_ERROR_BUSY; return; }
        uint16_t events = ld16(ef), rel = events & mask;
        if (!(mask & ~rel) || (mode == EF_OR && rel)) {
            if (ld8(ef + 0xF) == EF_CLEAR_AUTO) st16(ef, events & ~rel);
            st16(maskp, rel);
        } else {
            uint8_t slot_mode = 0;
            if (dir == EF_ANY2ANY) {
                uint16_t used = ld16(ef + 8);
                int i = 0;
                while (used & 1) { used >>= 1; ++i; }
                if (i == 16) { c.r[3] = TASK_ERROR_BUSY; return; }
                slot_mode = (uint8_t)((15 - i) << 4);
            }
            st8(ef + 6, slot_mode | (uint8_t)mode);
            st16(ef + 4, mask);
            wait = true;
        }
    }
    if (wait) {
        uint64_t ev[4];
        lv2_event_queue_pop(ld32(ef + 0x7C), ev, true);
        std::lock_guard<std::mutex> lk(spu_atomic_mutex());
        uint32_t slot = dir == EF_ANY2ANY ? ld8(ef + 6) >> 4 : 0;
        st16(maskp, ld16(ef + 0x30 + 2 * slot));
        st8(ef + 7, 0);
    }
    c.r[3] = CELL_OK;
}

// ---- not yet: barriers -----------------------------------------------------------------
#define SPURS_LATER(name)                                                                   \
    PS3_HLE(name) {                                                                         \
        rt_log("spurs: " #name "(%08X, %08X, %08X, %08X) is not implemented; it answers CELL_OK", \
               arg_u32(c, 0), arg_u32(c, 1), arg_u32(c, 2), arg_u32(c, 3));                \
        c.r[3] = CELL_OK;                                                                   \
    }
SPURS_LATER(cellSpursBarrierInitialize)

// ---- lock-free queues ------------------------------------------------------------------
// CellSpursLFQueue is a CellSyncLFQueue: +0 pop pointers (u16 each: committed,
// completion bitmap, wait flag, reserved), +8 push pointers (the same four),
// +0x10 element size, +0x14 depth, +0x18 buffer (u64, bit 0 set for ANY2ANY),
// +0x20 SPU ports (4 bytes), +0x24 direction, +0x28 v1, +0x2C init, +0x30 the
// pack of tasks waiting to pop, +0x32 their signal ids (15 u16), +0x50 the
// pack of tasks waiting to push, +0x52 their ids, +0x70 where a signal goes
// (the taskset, or the instance with bit 0 set), +0x78 v2, +0x7C event queue.
// The pointers run modulo twice the depth. A pack is three 5-bit counters
// modulo 30: the next waiter to wake (bits 10-14), the woken waiters that have
// taken their element (5-9), and the next free waiting place (0-4). In an
// ANY2ANY queue the packs live in the pop and push wait flags (+4 and +0xC)
// and +0x30 and +0x50 hold 0xFFFF: Edge zlib's inflate task, whose own code
// pops the queue, reads them there. A signal id is (workload << 8) | task.
//
// The PPU pushes with the queue's line under the SPUs' atomic lock, so a
// task's GETLLAR/PUTLLC on the queue sees each step whole. A PPU that finds
// the queue full polls it instead of waiting on the attached event queue: it
// never leaves itself among the push waiters.
namespace {
constexpr uint32_t LFQ_PPU2SPU = 2, LFQ_ANY2ANY = 3;
constexpr uint32_t SPURS_SYNC_AGAIN = 0x80410901u, SPURS_SYNC_INVAL = 0x80410902u, SPURS_SYNC_PERM = 0x80410909u,
                   SPURS_SYNC_STAT = 0x8041090Fu, SPURS_SYNC_ALIGN = 0x80410910u, SPURS_SYNC_NULL = 0x80410911u;

// v, a step off the range [0, n), brought back into it
int32_t ring(int32_t v, int32_t n) { return v < 0 ? v + n : v >= n ? v - n : v; }

// Signal the task whose id a waiter left: in the queue's taskset, or, for a
// queue of the instance, in the taskset that runs that workload.
void lfq_signal(uint32_t q, uint32_t id) {
    uint64_t to = ld64(q + 0x70);
    uint32_t wid = id >> 8, task = id & 0xFF, taskset = (uint32_t)to;
    if (to & 15) {
        uint32_t sp = (uint32_t)to & ~15u;
        taskset = 0;
        std::lock_guard<std::mutex> lk(g_mx);
        for (auto& [a, ts] : g_tasksets)
            if (ts.spurs == sp && ts.wid == wid) taskset = a;
    }
    if (g_trace) rt_log("spurs: lock-free queue %08X signals task %u of workload %u (taskset %08X)", q, task, wid, taskset);
    if (!taskset) { rt_log("spurs: lock-free queue %08X: no taskset runs workload %u; the signal is lost", q, wid); return; }
    send_signal(taskset, task);
}
}  // namespace

PS3_HLE(_cellSpursLFQueueInitialize) {   // (tasksetOrSpurs, queue, buffer, size, depth, direction)
    uint32_t to = arg_u32(c, 0), q = arg_u32(c, 1), buf = arg_u32(c, 2), size = arg_u32(c, 3), depth = arg_u32(c, 4),
             dir = arg_u32(c, 5);
    if (!q || (size && !buf)) { c.r[3] = SPURS_SYNC_NULL; return; }
    if (size > 0x4000 || size % 16 || !depth || depth > 0x7FFF || dir > LFQ_ANY2ANY) { c.r[3] = SPURS_SYNC_INVAL; return; }
    if (q % 128 || buf % 16) { c.r[3] = SPURS_SYNC_ALIGN; return; }
    {
        std::lock_guard<std::mutex> lk(spu_atomic_mutex());
        std::memset(g_mem + q, 0, 128);
        st32(q + 0x10, size);
        st32(q + 0x14, depth);
        st64(q + 0x18, dir == LFQ_ANY2ANY ? buf | 1 : buf);
        st32(q + 0x20, dir == LFQ_ANY2ANY ? 0xFFFF0000u : 0xFFFFFFFFu);
        st32(q + 0x24, dir);
        st32(q + 0x28, dir == LFQ_ANY2ANY ? 0xFFFFFFFFu : 0);
        st16(q + 0x30, dir == LFQ_ANY2ANY ? 0xFFFF : 0);
        st16(q + 0x50, dir == LFQ_ANY2ANY ? 0xFFFF : 0);
        st64(q + 0x70, to);
    }
    rt_log("spurs: lock-free queue %08X, %u elements of %u bytes at %08X, direction %u, signals to %08X", q, depth, size,
           buf, dir, to);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursLFQueueAttachLv2EventQueue) {   // (queue): for a PPU that waits, which this one never does
    uint32_t q = arg_u32(c, 0);
    if (ld32(q + 0x7C)) { c.r[3] = SPURS_SYNC_STAT; return; }
    st32(q + 0x7C, lv2_event_queue_new(127));
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursLFQueueDetachLv2EventQueue) {   // (queue)
    uint32_t q = arg_u32(c, 0);
    if (!ld32(q + 0x7C)) { c.r[3] = SPURS_SYNC_STAT; return; }
    st32(q + 0x7C, 0);
    c.r[3] = CELL_OK;
}

PS3_HLE(_cellSpursLFQueuePushBody) {   // (queue, element, isBlocking)
    uint32_t q = arg_u32(c, 0), elem = arg_u32(c, 1), blocking = arg_u32(c, 2);
    if (!q || !elem) { c.r[3] = SPURS_SYNC_NULL; return; }
    if (q % 128 || elem % 16) { c.r[3] = SPURS_SYNC_ALIGN; return; }
    uint32_t dir = ld32(q + 0x24);
    if (dir != LFQ_PPU2SPU && dir != LFQ_ANY2ANY) { c.r[3] = SPURS_SYNC_PERM; return; }
    const int32_t depth = (int32_t)ld32(q + 0x14), d2 = 2 * depth;
    const uint32_t pop_pack = dir == LFQ_ANY2ANY ? q + 4 : q + 0x30, push_pack = dir == LFQ_ANY2ANY ? q + 0xC : q + 0x50;
    // a place: free, not owed to a woken pusher, with no pusher waiting first
    // and fewer than 16 pushes under way (the completion bitmap's width)
    int32_t pos;
    for (bool polled = false;; polled = true) {
        {
            std::lock_guard<std::mutex> lk(spu_atomic_mutex());
            int32_t h1 = ld16(q), h5 = ld16(q + 8), h8 = (int16_t)ld16(q + 0xE);
            int32_t pack = ld16(push_pack), a = pack >> 10 & 31, b = pack >> 5 & 31, w = pack & 31;
            int32_t used = ring(h8 - h1, d2), owed = ring(a - b, 30), busy = ring(h8 - h5, d2);
            if (depth - used > owed && busy + owed <= 15 && a == w) {
                pos = h8;
                st16(q + 0xE, (uint16_t)(h8 + 1 >= d2 ? 0 : h8 + 1));
                break;
            }
        }
        if (!blocking) { c.r[3] = SPURS_SYNC_AGAIN; return; }
        if (!polled && g_trace) rt_log("spurs: lock-free queue %08X is full; the PPU polls it", q);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    uint32_t size = ld32(q + 0x10), buf = (uint32_t)ld64(q + 0x18) & ~1u;
    std::memcpy(g_mem + buf + size * (uint32_t)(pos >= depth ? pos - depth : pos), g_mem + elem, size);
    // complete it: the committed pointer moves over every completed place in
    // order, and the first task waiting to pop is woken, unless 16 pops would
    // then be under way or owed
    int32_t wake = -1;
    {
        std::lock_guard<std::mutex> lk(spu_atomic_mutex());
        int32_t h1 = ld16(q), h4 = (int16_t)ld16(q + 6), h5 = ld16(q + 8);
        uint32_t done = ld16(q + 0xA) | 0x8000u >> ring(pos - h5, d2);
        int32_t n = 0;
        while (n < 16 && (done << n & 0x8000u)) ++n;
        st16(q + 0xA, (uint16_t)(n >= 16 ? 0 : done << n));
        st16(q + 8, (uint16_t)ring(h5 + n, d2));
        int32_t pack = ld16(pop_pack), a = pack >> 10 & 31, b = pack >> 5 & 31, w = pack & 31;
        if (a != w && ring(h4 - h1, d2) + ring(a - b, 30) <= 15) {
            wake = ld16(q + 0x32 + 2 * (a >= 15 ? a - 15 : a));
            st16(pop_pack, (uint16_t)((pack & 0x83FF) | (a == 29 ? 0 : a + 1) << 10));
        }
    }
    if (g_trace) rt_log("spurs: lock-free queue %08X: pushed at %d%s", q, pos, wake >= 0 ? ", a task waiting" : "");
    if (wake >= 0) lfq_signal(q, (uint32_t)wake);
    c.r[3] = CELL_OK;
}

// ---- queues ----------------------------------------------------------------------------
// CellSpursQueue (128 bytes), as Bullet's narrow-phase task pops and pushes
// it with its own code: +0 the pop pointer, +4 the push pointer (s32 each,
// modulo twice the depth, stored complemented while a pop or a push is under
// way), +8 element size, +0xC depth, +0x10 buffer (u64), +0x18 the SPU port
// of the event queue attached for a PPU that waits (above 63: none), +0x1C
// direction, +0x20 the number of tasks waiting to pop and +0x21 their task
// ids, +0x30 the same for tasks waiting to push, +0x40 the number again and
// +0x41 the waiting poppers' workloads, +0x50 the pushers' workloads, +0x60
// the taskset (u64, 0 for a queue of the instance). A task that finishes a pop
// wakes the next popper and the first pusher; one that finishes a push, the
// first popper.
//
// The PPU pops and pushes each element whole under the SPUs' atomic lock, so
// it never leaves a pointer complemented; it wakes the waiters as a task
// would. A PPU that would wait polls instead, every 50 microseconds, and the
// event queue it attaches is never sent to.
namespace {
constexpr uint32_t Q_SPU2PPU = 1, Q_PPU2SPU = 2, Q_ANY2ANY = 3;
constexpr uint32_t SPURS_SYNC_BUSY = 0x8041090Au;

struct QueueOwner { uint32_t spurs, eq; };
std::map<uint32_t, QueueOwner> g_queues;            // by the queue's address

int32_t q_ptr(int32_t v) { return v < 0 ? ~v : v; }

struct Waiter { uint32_t task, wid; };

// The first task of a waiting list (+0x20 poppers, +0x30 pushers), if any.
void q_first(uint32_t q, uint32_t list, std::vector<Waiter>& wake) {
    if (ld8(q + list)) wake.push_back({ld8(q + list + 1), ld8(q + list + 0x21)});
}

// Wake a waiting task: by the queue's taskset, or by its workload.
void q_wake(uint32_t q, Waiter w) {
    uint32_t task = w.task, wid = w.wid, taskset = (uint32_t)ld64(q + 0x60);
    if (!taskset) {
        std::lock_guard<std::mutex> lk(g_mx);
        uint32_t sp = g_queues.count(q) ? g_queues[q].spurs : 0;
        for (auto& [a, ts] : g_tasksets)
            if (ts.spurs == sp && ts.wid == wid) taskset = a;
    }
    if (g_trace) rt_log("spurs: queue %08X wakes task %u of taskset %08X", q, task, taskset);
    if (taskset) send_signal(taskset, task);
    else rt_log("spurs: queue %08X: no taskset runs workload %u; the signal is lost", q, wid);
}

// One element in or out, whole; CELL_OK, or the error to answer or wait on.
uint32_t q_move(uint32_t q, uint32_t elem, bool push, std::vector<Waiter>& wake) {
    std::lock_guard<std::mutex> lk(spu_atomic_mutex());
    int32_t w0 = (int32_t)ld32(q), w1 = (int32_t)ld32(q + 4), depth = (int32_t)ld32(q + 0xC), d2 = 2 * depth;
    int32_t n0 = q_ptr(w0), n1 = q_ptr(w1), count = ring(n1 - n0, d2);
    uint32_t size = ld32(q + 8), buf = (uint32_t)ld64(q + 0x10);
    if (push) {
        if (count >= depth) return SPURS_SYNC_AGAIN;
        if (w1 < 0 || ld8(q + 0x30)) return SPURS_SYNC_BUSY;
        std::memcpy(g_mem + buf + size * (uint32_t)(n1 >= depth ? n1 - depth : n1), g_mem + elem, size);
        st32(q + 4, (uint32_t)(n1 + 1 >= d2 ? 0 : n1 + 1));
        q_first(q, 0x20, wake);
        if (count + 1 < depth) q_first(q, 0x30, wake);
    } else {
        if (count == 0) return SPURS_SYNC_AGAIN;
        if (w0 < 0 || ld8(q + 0x20)) return SPURS_SYNC_BUSY;
        std::memcpy(g_mem + elem, g_mem + buf + size * (uint32_t)(n0 >= depth ? n0 - depth : n0), size);
        st32(q, (uint32_t)(n0 + 1 >= d2 ? 0 : n0 + 1));
        if (count > 1) q_first(q, 0x20, wake);
        q_first(q, 0x30, wake);
    }
    return CELL_OK;
}

void q_body(PPUContext& c, bool push) {   // (queue, element, isBlocking)
    uint32_t q = arg_u32(c, 0), elem = arg_u32(c, 1), blocking = arg_u32(c, 2);
    if (!q || !elem) { c.r[3] = SPURS_SYNC_NULL; return; }
    if (q % 128 || elem % 16) { c.r[3] = SPURS_SYNC_ALIGN; return; }
    uint32_t dir = ld32(q + 0x1C);
    if (dir != Q_ANY2ANY && dir != (push ? Q_PPU2SPU : Q_SPU2PPU)) { c.r[3] = SPURS_SYNC_PERM; return; }
    for (bool polled = false;; polled = true) {
        std::vector<Waiter> wake;
        uint32_t r = q_move(q, elem, push, wake);
        for (const Waiter& w : wake) q_wake(q, w);
        if (r == CELL_OK || !blocking) { c.r[3] = r; return; }
        if (!polled && g_trace) rt_log("spurs: queue %08X is %s; the PPU polls it", q, push ? "full" : "empty");
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}
}  // namespace

PS3_HLE(_cellSpursQueueInitialize) {   // (spurs, taskset, queue, buffer, size, depth, direction)
    uint32_t sp = arg_u32(c, 0), ts = arg_u32(c, 1), q = arg_u32(c, 2), buf = arg_u32(c, 3), size = arg_u32(c, 4),
             depth = arg_u32(c, 5), dir = arg_u32(c, 6);
    if (!q || !buf || (!sp && !ts)) { c.r[3] = SPURS_SYNC_NULL; return; }
    if (!size || size % 16 || !depth || depth > 0x7FFF || dir > Q_ANY2ANY) { c.r[3] = SPURS_SYNC_INVAL; return; }
    if (q % 128 || buf % 16) { c.r[3] = SPURS_SYNC_ALIGN; return; }
    {
        std::lock_guard<std::mutex> lk(spu_atomic_mutex());
        std::memset(g_mem + q, 0, 128);
        st32(q + 8, size);
        st32(q + 0xC, depth);
        st64(q + 0x10, buf);
        st32(q + 0x18, 0xFF);
        st32(q + 0x1C, dir);
        st64(q + 0x60, ts);
    }
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_queues[q] = {ts ? (uint32_t)ld64(ts + 0x60) : sp, 0};
    }
    if (g_trace) rt_log("spurs: queue %08X, %u elements of %u bytes at %08X, direction %u", q, depth, size, buf, dir);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursQueueAttachLv2EventQueue) {   // (queue): for a PPU that waits, which this one never does
    uint32_t q = arg_u32(c, 0);
    std::lock_guard<std::mutex> lk(g_mx);
    auto it = g_queues.find(q);
    if (it == g_queues.end()) { c.r[3] = SPURS_SYNC_STAT; return; }
    if (it->second.eq) { c.r[3] = SPURS_SYNC_STAT; return; }
    it->second.eq = lv2_event_queue_new(127);
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursQueueDetachLv2EventQueue) {   // (queue)
    uint32_t q = arg_u32(c, 0);
    std::lock_guard<std::mutex> lk(g_mx);
    auto it = g_queues.find(q);
    if (it == g_queues.end() || !it->second.eq) { c.r[3] = SPURS_SYNC_STAT; return; }
    it->second.eq = 0;
    c.r[3] = CELL_OK;
}

PS3_HLE(cellSpursQueuePushBody) { q_body(c, true); }   // (queue, element, isBlocking)
PS3_HLE(cellSpursQueuePopBody) { q_body(c, false); }   // (queue, buffer, isBlocking)
