// ps3kit runtime — sceNpTrophy: the game's trophy set, read from the disc's
// TROPHY.TRP, and what the player has unlocked, kept in
// /dev_hdd0/home/00000001/trophy/<NP communication id>.
//
// A context names a set by its communication id; registering it installs the
// set, reporting each step to the game's status callback on the calling
// thread, as the system does: first whether it was installed already, then
// setup, progress, finalize and complete. Once installed, the game asks for a
// trophy's details and whether it is unlocked, and unlocks it; the platinum is
// never unlocked by the game, but by the system, when the last trophy it
// stands on is.
//
// The set's configuration is the package's TROP_01.SFM (English, the language
// cellSysutil reports) or TROP.SFM, plain XML. The unlocks are kept in
// PS3KIT.TROPUSR, the runtime's own record, one "id tick" per line; the
// system's TROPUSR.DAT and TROPTROP.DAT are not made. A disc without the
// package (one extracted without it) gets an empty set that installs and
// unlocks by number alone, so that a game still gets past its trophy checks.
#include "rt.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
extern std::string g_dev_bdvd, g_dev_hdd0;

namespace {
constexpr uint32_t ERR_NOT_INITIALIZED = 0x80022902u, ERR_CONTEXT_NOT_REGISTERED = 0x80022904u,
                   ERR_INVALID_ARGUMENT = 0x80022906u, ERR_UNKNOWN_CONTEXT = 0x8002290Au,
                   ERR_PROCESSING_ABORTED = 0x8002290Fu, ERR_UNKNOWN_HANDLE = 0x80022911u,
                   ERR_HIDDEN = 0x80022913u, ERR_CANNOT_UNLOCK_PLATINUM = 0x80022914u,
                   ERR_ALREADY_UNLOCKED = 0x80022915u, ERR_INVALID_TROPHY_ID = 0x80022920u;
enum Status : uint32_t { NOT_INSTALLED = 1, INSTALLED = 3, SETUP = 5, PROGRESS = 6, FINALIZE = 7, COMPLETE = 8 };
enum Grade : uint32_t { PLATINUM = 1, GOLD = 2, SILVER = 3, BRONZE = 4 };
constexpr uint32_t kInvalidId = 0xFFFFFFFFu;
constexpr uint64_t kEpochTicks = 62135596800ull * 1000000;    // CellRtcTick of 1970-01-01: µs since year 1

struct Trophy {
    uint32_t grade = BRONZE;
    bool hidden = false;
    int pid = -1;                        // the platinum this one counts towards
    std::string name, detail;
};

struct Context {
    std::string commid;                  // "NPWR01180_00"
    std::vector<Trophy> set;             // empty when the disc has no package
    bool has_conf = false;
    bool registered = false;
    std::map<uint32_t, uint64_t> unlocked;    // id -> CellRtcTick
};

std::mutex g_mx;
bool g_init;
uint32_t g_next_ctx = 1, g_next_handle = 1;
std::map<uint32_t, Context> g_ctx;
std::map<uint32_t, bool> g_handles;
const bool g_trace = std::getenv("PS3KIT_TROPHY") != nullptr;

uint64_t rtc_now() {
    using namespace std::chrono;
    return kEpochTicks + (uint64_t)duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

std::string user_dir(const Context& x) { return g_dev_hdd0 + "/home/00000001/trophy/" + x.commid; }
std::string user_file(const Context& x) { return user_dir(x) + "/PS3KIT.TROPUSR"; }

// One member of the TRP package: a 0x40-byte header, then entries of
// name[32], u64 offset, u64 size, and padding, big-endian.
bool trp_member(const std::string& path, const char* want, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto be32 = [&](size_t o) { return (uint32_t)(uint8_t)b[o] << 24 | (uint32_t)(uint8_t)b[o + 1] << 16 |
                                       (uint32_t)(uint8_t)b[o + 2] << 8 | (uint8_t)b[o + 3]; };
    auto be64 = [&](size_t o) { return (uint64_t)be32(o) << 32 | be32(o + 4); };
    if (b.size() < 0x40 || be32(0) != 0xDCA24D00u) return false;
    uint32_t n = be32(0x10), esz = be32(0x14);
    for (uint32_t i = 0; i < n; ++i) {
        size_t e = 0x40 + (size_t)i * esz;
        if (e + 0x30 > b.size()) return false;
        if (std::strncmp(b.c_str() + e, want, 32) != 0) continue;
        uint64_t off = be64(e + 32), size = be64(e + 40);
        if (off + size > b.size()) return false;
        out = b.substr(off, size);
        return true;
    }
    return false;
}

std::string attr(const std::string& tag, const char* key) {
    std::string k = std::string(" ") + key + "=\"";
    size_t p = tag.find(k);
    if (p == std::string::npos) return "";
    p += k.size();
    return tag.substr(p, tag.find('"', p) - p);
}

std::string element(const std::string& body, const char* name) {
    std::string open = std::string("<") + name + ">", close = std::string("</") + name + ">";
    size_t p = body.find(open);
    if (p == std::string::npos) return "";
    p += open.size();
    size_t q = body.find(close, p);
    return q == std::string::npos ? "" : body.substr(p, q - p);
}

// The <trophy> elements of an SFM, in id order.
std::vector<Trophy> parse_sfm(const std::string& x) {
    std::vector<Trophy> set;
    for (size_t p = 0; (p = x.find("<trophy ", p)) != std::string::npos;) {
        size_t te = x.find('>', p);
        if (te == std::string::npos) break;
        std::string tag = x.substr(p, te - p);
        size_t be = x.find("</trophy>", te);
        std::string body = (tag.back() != '/' && be != std::string::npos) ? x.substr(te + 1, be - te - 1) : "";
        uint32_t id = (uint32_t)std::strtoul(attr(tag, "id").c_str(), nullptr, 10);
        if (id >= set.size()) set.resize(id + 1);
        Trophy& t = set[id];
        std::string g = attr(tag, "ttype");
        t.grade = g == "P" ? PLATINUM : g == "G" ? GOLD : g == "S" ? SILVER : BRONZE;
        t.hidden = attr(tag, "hidden") == "yes";
        t.pid = std::atoi(attr(tag, "pid").c_str());
        t.name = element(body, "name");
        t.detail = element(body, "detail");
        p = te;
    }
    return set;
}

void load_set(Context& x) {
    std::string trp = g_dev_bdvd + "/PS3_GAME/TROPDIR/" + x.commid + "/TROPHY.TRP", sfm;
    if (trp_member(trp, "TROP_01.SFM", sfm) || trp_member(trp, "TROP.SFM", sfm)) {
        x.set = parse_sfm(sfm);
        x.has_conf = true;
    } else {
        rt_log("trophy: no trophy set at %s; an empty one stands in", trp.c_str());
    }
}

bool installed(const Context& x) {
    std::error_code ec;
    return fs::exists(user_file(x), ec);
}

void load_unlocks(Context& x) {
    std::ifstream f(user_file(x));
    uint32_t id;
    uint64_t tick;
    while (f >> id >> tick) x.unlocked[id] = tick;
}

void save_unlocks(const Context& x) {
    std::error_code ec;
    fs::create_directories(user_dir(x), ec);
    std::ofstream f(user_file(x), std::ios::trunc);
    for (auto& [id, tick] : x.unlocked) f << id << ' ' << tick << '\n';
}

Context* context(uint32_t id) {
    auto it = g_ctx.find(id);
    return it == g_ctx.end() ? nullptr : &it->second;
}

const char* name_of(const Context& x, uint32_t id) {
    return id < x.set.size() && !x.set[id].name.empty() ? x.set[id].name.c_str() : "?";
}
}  // namespace

PS3_HLE(sceNpTrophyInit) {               // (pool, poolSize, containerId, options)
    std::lock_guard<std::mutex> lk(g_mx);
    g_init = true;
    c.r[3] = CELL_OK;
}

PS3_HLE(sceNpTrophyTerm) {
    std::lock_guard<std::mutex> lk(g_mx);
    g_init = false;
    g_ctx.clear();
    g_handles.clear();
    c.r[3] = CELL_OK;
}

// (u32* context, const SceNpCommunicationId* commId, const SceNpCommunicationSignature*, u64 options)
// SceNpCommunicationId: char data[9], char term, u8 num, char dummy.
PS3_HLE(sceNpTrophyCreateContext) {
    std::lock_guard<std::mutex> lk(g_mx);
    if (!g_init) { c.r[3] = ERR_NOT_INITIALIZED; return; }
    uint32_t out = arg_u32(c, 0), comm = arg_u32(c, 1);
    if (!out || !comm) { c.r[3] = ERR_INVALID_ARGUMENT; return; }
    char id[16];
    std::snprintf(id, sizeof id, "%s_%02u", mem_cstr(comm, 9).c_str(), g_mem[comm + 10]);
    uint32_t n = g_next_ctx++;
    Context& x = g_ctx[n];
    x.commid = id;
    load_set(x);
    st32(out, n);
    if (g_trace) rt_log("trophy: context %u for %s, %zu trophies", n, id, x.set.size());
    c.r[3] = CELL_OK;
}

PS3_HLE(sceNpTrophyDestroyContext) {
    std::lock_guard<std::mutex> lk(g_mx);
    c.r[3] = g_ctx.erase(arg_u32(c, 0)) ? CELL_OK : ERR_UNKNOWN_CONTEXT;
}

PS3_HLE(sceNpTrophyCreateHandle) {       // (u32* handle)
    std::lock_guard<std::mutex> lk(g_mx);
    if (!g_init) { c.r[3] = ERR_NOT_INITIALIZED; return; }
    if (!arg_u32(c, 0)) { c.r[3] = ERR_INVALID_ARGUMENT; return; }
    uint32_t n = g_next_handle++;
    g_handles[n] = true;
    st32(arg_u32(c, 0), n);
    c.r[3] = CELL_OK;
}

PS3_HLE(sceNpTrophyDestroyHandle) {
    std::lock_guard<std::mutex> lk(g_mx);
    c.r[3] = g_handles.erase(arg_u32(c, 0)) ? CELL_OK : ERR_UNKNOWN_HANDLE;
}

// (context, handle, u64* reqspace, u64 options): what installing would take,
// nothing once installed.
PS3_HLE(sceNpTrophyGetRequiredDiskSpace) {
    std::lock_guard<std::mutex> lk(g_mx);
    Context* x = context(arg_u32(c, 0));
    if (!x) { c.r[3] = ERR_UNKNOWN_CONTEXT; return; }
    if (!g_handles.count(arg_u32(c, 1))) { c.r[3] = ERR_UNKNOWN_HANDLE; return; }
    std::error_code ec;
    uint64_t size = fs::file_size(g_dev_bdvd + "/PS3_GAME/TROPDIR/" + x->commid + "/TROPHY.TRP", ec);
    if (ec) size = 0;
    if (arg_u32(c, 2)) st64(arg_u32(c, 2), installed(*x) ? 0 : (size + 0xFFFF) & ~0xFFFFull);
    c.r[3] = CELL_OK;
}

// (context, handle, statusCb, arg, u64 options); statusCb(context, status,
// completed, total, arg) returns a negative value to abort.
PS3_HLE(sceNpTrophyRegisterContext) {
    uint32_t ctx = arg_u32(c, 0), cb = arg_u32(c, 2), arg = arg_u32(c, 3);
    Context* x;
    bool was;
    size_t n;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        x = context(ctx);
        if (!x) { c.r[3] = ERR_UNKNOWN_CONTEXT; return; }
        if (!g_handles.count(arg_u32(c, 1))) { c.r[3] = ERR_UNKNOWN_HANDLE; return; }
        was = installed(*x);
        n = x->set.size();
    }
    auto report = [&](uint32_t status, uint32_t done, uint32_t total) {
        if (g_trace) rt_log("trophy: status %u (%u of %u)", status, done, total);
        return !cb || (int32_t)ppu_call_guest(c, cb, ctx, status, done, total, arg) >= 0;
    };
    if (!report(was ? INSTALLED : NOT_INSTALLED, 0, 0)) { c.r[3] = ERR_PROCESSING_ABORTED; return; }
    const std::pair<uint32_t, uint32_t> steps[] = {{SETUP, 3}, {PROGRESS, (uint32_t)n}, {FINALIZE, 4}};
    for (auto [status, total] : steps)
        for (uint32_t i = 0; i <= total; ++i)
            if (!report(status, i, total)) { c.r[3] = ERR_PROCESSING_ABORTED; return; }
    {
        std::lock_guard<std::mutex> lk(g_mx);
        load_unlocks(*x);
        if (!was) save_unlocks(*x);
        x->registered = true;
        rt_log("trophy: %s registered, %zu trophies, %zu unlocked%s", x->commid.c_str(), n, x->unlocked.size(),
               was ? "" : " (installed now)");
    }
    report(COMPLETE, 0, 0);
    c.r[3] = CELL_OK;
}

// (context, handle, trophyId, SceNpTrophyDetails*, SceNpTrophyData*)
// SceNpTrophyDetails: s32 trophyId, u32 trophyGrade, char name[128],
// char description[1024], b8 hidden, u8 reserved[3].
// SceNpTrophyData: u64 timestamp, s32 trophyId, b8 unlocked, u8 reserved[3].
PS3_HLE(sceNpTrophyGetTrophyInfo) {
    std::lock_guard<std::mutex> lk(g_mx);
    Context* x = context(arg_u32(c, 0));
    uint32_t id = arg_u32(c, 2), det = arg_u32(c, 3), dat = arg_u32(c, 4);
    if (!x) { c.r[3] = ERR_UNKNOWN_CONTEXT; return; }
    if (!g_handles.count(arg_u32(c, 1))) { c.r[3] = ERR_UNKNOWN_HANDLE; return; }
    if (!x->registered) { c.r[3] = ERR_CONTEXT_NOT_REGISTERED; return; }
    if (!det && !dat) { c.r[3] = ERR_INVALID_ARGUMENT; return; }
    if (x->has_conf && id >= x->set.size()) { c.r[3] = ERR_INVALID_TROPHY_ID; return; }
    Trophy t = id < x->set.size() ? x->set[id] : Trophy();
    auto u = x->unlocked.find(id);
    bool unlocked = u != x->unlocked.end();
    if (t.hidden && !unlocked) { c.r[3] = ERR_HIDDEN; return; }
    if (det) {
        std::memset(g_mem + det, 0, 0x48C);
        st32(det, id);
        st32(det + 4, t.grade);
        mem_write(det + 8, t.name.c_str(), std::min<size_t>(t.name.size(), 127));
        mem_write(det + 0x88, t.detail.c_str(), std::min<size_t>(t.detail.size(), 1023));
        g_mem[det + 0x488] = t.hidden;
    }
    if (dat) {
        std::memset(g_mem + dat, 0, 0x10);
        st64(dat, unlocked ? u->second : 0);
        st32(dat + 8, id);
        g_mem[dat + 12] = unlocked;
    }
    c.r[3] = CELL_OK;
}

// (context, handle, trophyId, s32* platinumId): the platinum the unlock
// completes, if any, is unlocked with it.
PS3_HLE(sceNpTrophyUnlockTrophy) {
    std::lock_guard<std::mutex> lk(g_mx);
    Context* x = context(arg_u32(c, 0));
    uint32_t id = arg_u32(c, 2), plat = arg_u32(c, 3);
    if (!x) { c.r[3] = ERR_UNKNOWN_CONTEXT; return; }
    if (!g_handles.count(arg_u32(c, 1))) { c.r[3] = ERR_UNKNOWN_HANDLE; return; }
    if (!x->registered) { c.r[3] = ERR_CONTEXT_NOT_REGISTERED; return; }
    if (x->has_conf && id >= x->set.size()) { c.r[3] = ERR_INVALID_TROPHY_ID; return; }
    if (id < x->set.size() && x->set[id].grade == PLATINUM) { c.r[3] = ERR_CANNOT_UNLOCK_PLATINUM; return; }
    if (x->unlocked.count(id)) { c.r[3] = ERR_ALREADY_UNLOCKED; return; }
    uint64_t now = rtc_now();
    x->unlocked[id] = now;
    rt_log("trophy: unlocked %03u \"%s\"", id, name_of(*x, id));
    uint32_t won = kInvalidId;
    int pid = id < x->set.size() ? x->set[id].pid : -1;
    if (pid >= 0 && (size_t)pid < x->set.size() && !x->unlocked.count(pid)) {
        bool all = true;
        for (size_t i = 0; i < x->set.size() && all; ++i)
            if (x->set[i].pid == pid && !x->unlocked.count((uint32_t)i)) all = false;
        if (all) {
            x->unlocked[pid] = now;
            won = (uint32_t)pid;
            rt_log("trophy: unlocked %03d \"%s\", the platinum", pid, name_of(*x, pid));
        }
    }
    save_unlocks(*x);
    if (plat) st32(plat, won);
    c.r[3] = CELL_OK;
}
