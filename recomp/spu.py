"""SPU programs to C++.

An SPU program is translated whole into one function, compiled to sit at one
local-store address (its base): every instruction gets a label, a direct
branch inside the program is a goto, and an indirect branch (bi, a return,
a jump table) goes through a switch over every instruction of the program.
That needs no function discovery at all, which suits SPU code: it has no
call instruction distinct from a branch and its returns are indirect jumps.
A branch out of the program returns the target to the caller (the runtime's
job manager), which knows what lives there.

Programs are found by what marks them. A SPURS job binary built with the
SDK's job runtime carries "JOBCRT Ver" at +0x30, its image size at +0x3C,
and its entry at +0x10; the code is position-independent and relocates
itself, so any base serves as long as the runtime loads it there.

An SPU ELF (machine 23) embedded in the executable is what an SPU thread
group or a SPURS task runs: its executable segment is the program, compiled
at the address its program header loads it to. An ELF that carries
relocations (`.rela.text`) is a module some other SPU program loads where it
chooses (FMOD's codecs and effects); its address is not in the file. Where the
loader puts it is learnt from a run and given as a bias (local store = its
address + the bias): the module is then translated there, every instruction a
possible target, and the runtime runs it when its loader branches into a copy
of it in local store (an overlay). Without a bias it is listed, not translated.
"""
import re
import struct
from ..spu import decode, _sext

JOBCRT_MAGIC = b"JOBCRT Ver"
JOB_BASE = 0x1000          # where the runtime's job manager loads a job binary


class Program:
    def __init__(self, ea, size, base, name, elf=0, module=False):
        self.ea, self.size, self.base, self.name = ea, size, base, name
        self.elf = elf                      # the SPU ELF it comes from, or 0 for a job
        self.module = module                # a relocatable module, placed at a bias

    @property
    def fname(self):
        return f"spu_{self.ea:08X}_{self.base:05X}" if self.module else f"spu_{self.ea:08X}"


def find_jobs(img):
    """Every SPURS job binary in the image's data, by its JOBCRT header."""
    out = []
    for seg in img.segments:
        data = seg.data
        at = data.find(JOBCRT_MAGIC)
        while at >= 0:
            start = at - 0x30
            if start >= 0 and start % 16 == 0:
                size = int.from_bytes(data[start + 0x3C:start + 0x40], "big")
                if 0x60 < size <= 0x3F000 and start + size <= len(data):
                    ea = seg.vaddr + start
                    blob = data[start:start + size]
                    name = _job_name(blob) or f"job_{ea:08X}"
                    out.append(Program(ea, size, JOB_BASE, name))
            at = data.find(JOBCRT_MAGIC, at + 1)
    return out


SPU_MACHINE = 23
PF_X = 1


def _spu_name(blob, shoff, shnum, shstrndx):
    """(the ELF's `.note.spu_name`, whether it has relocation sections)."""
    if not shnum or shoff + shnum * 40 > len(blob):
        return None, False
    secs = [struct.unpack_from(">10I", blob, shoff + 40 * i) for i in range(shnum)]
    stro = secs[shstrndx][4] if shstrndx < shnum else 0

    def name(off):
        end = blob.find(b"\0", stro + off)
        return blob[stro + off:end].decode("latin1") if end >= 0 else ""
    note, rela = None, False
    for sh in secs:
        n = name(sh[0])
        rela |= n.startswith(".rela")
        if n == ".note.spu_name" and sh[5] >= 12:
            namesz, descsz = struct.unpack_from(">II", blob, sh[4])
            d = sh[4] + 12 + ((namesz + 3) & ~3)
            note = blob[d:d + descsz].split(b"\0")[0].decode("latin1") or None
    return note, rela


def find_elfs(img):
    """(programs, relocatable modules): every SPU ELF in the image, by its
    header. A program is the ELF's executable segment, at its load address;
    a module is (ea, name) and is not translated."""
    progs, modules = [], []
    for seg in img.segments:
        data = seg.data
        at = data.find(b"\x7fELF\x01\x02\x01")
        while at >= 0:
            if len(data) - at >= 52 and struct.unpack_from(">H", data, at + 18)[0] == SPU_MACHINE:
                ea = seg.vaddr + at
                phoff, shoff = struct.unpack_from(">II", data, at + 28)
                phnum, shnum, shstrndx = struct.unpack_from(">HxxHH", data, at + 44)
                blob = data[at:]
                note, rela = _spu_name(blob, shoff, shnum, shstrndx)
                name = note or f"spuelf_{ea:08X}"
                if rela:
                    modules.append((ea, name))
                else:
                    for i in range(phnum):
                        p_type, off, vaddr, _, filesz, _, flags, _ = struct.unpack_from(">8I", blob, phoff + 32 * i)
                        if p_type == 1 and flags & PF_X and filesz:
                            progs.append(Program(ea + off, filesz & ~3, vaddr, name, elf=ea))
            at = data.find(b"\x7fELF\x01\x02\x01", at + 1)
    return progs, modules


def place_modules(img, modules, biases):
    """The programs of relocatable modules, each one's executable segment at
    its address plus every bias given."""
    out = []
    for ea, name in modules:
        phoff, phnum = img.u32(ea + 28), img.u32(ea + 44) >> 16
        for i in range(phnum):
            p_type, off, vaddr, _, filesz, _, flags, _ = struct.unpack(">8I", img.read(ea + phoff + 32 * i, 32))
            if p_type == 1 and flags & PF_X and filesz:
                for bias in biases:
                    out.append(Program(ea + off, filesz & ~3, vaddr + bias, name, elf=ea, module=True))
    return out


def _job_name(blob):
    """The middleware's own banner, when the binary carries one."""
    for marker in (b"SpursJob ",):
        i = blob.find(marker)
        if i >= 0:
            j = i + len(marker)
            k = j
            while k < len(blob) and 0x21 <= blob[k] < 0x7F and blob[k] != ord("/"):
                k += 1
            return blob[j:k].decode("ascii")
    return None


class Unsupported(Exception):
    pass


def _r(n):
    return f"s.r[{n}]"


_BINARY = {     # rt = f(ra, rb)
    "a": "{ra}.w + {rb}.w", "sf": "{rb}.w - {ra}.w", "ah": "{ra}.h + {rb}.h", "sfh": "{rb}.h - {ra}.h",
    "and": "{ra}.w & {rb}.w", "andc": "{ra}.w & ~{rb}.w", "or": "{ra}.w | {rb}.w", "orc": "{ra}.w | ~{rb}.w",
    "xor": "{ra}.w ^ {rb}.w", "nand": "~({ra}.w & {rb}.w)", "nor": "~({ra}.w | {rb}.w)", "eqv": "~({ra}.w ^ {rb}.w)",
    "fa": "{ra}.f + {rb}.f", "fs": "{ra}.f - {rb}.f", "fm": "{ra}.f * {rb}.f",
    "dfa": "{ra}.df + {rb}.df", "dfs": "{ra}.df - {rb}.df", "dfm": "{ra}.df * {rb}.df",
}
_BINARY_FIELD = {"ah": "h", "sfh": "h", "fa": "f", "fs": "f", "fm": "f", "dfa": "df", "dfs": "df", "dfm": "df"}

_COMPARE = {    # rt = mask(ra OP rb), by element view
    "ceq": ("sw", "==", "sw"), "ceqh": ("sh", "==", "sh"), "ceqb": ("sb", "==", "sb"),
    "cgt": ("sw", ">", "sw"), "cgth": ("sh", ">", "sh"), "cgtb": ("sb", ">", "sb"),
    "clgt": ("w", ">", "sw"), "clgth": ("h", ">", "sh"), "clgtb": ("b", ">", "sb"),
    "fceq": ("f", "==", "sw"), "fcgt": ("f", ">", "sw"),
    "dfceq": ("df", "==", "sd"), "dfcgt": ("df", ">", "sd"),
}

_CALL2 = {      # rt = spu_op(ra, rb)
    "cg": "spu_cg", "bg": "spu_bg", "mpy": "spu_mpy", "mpyu": "spu_mpyu", "mpyh": "spu_mpyh",
    "mpys": "spu_mpys", "mpyhh": "spu_mpyhh", "mpyhhu": "spu_mpyhhu", "avgb": "spu_avgb",
    "absdb": "spu_absdb", "sumb": "spu_sumb", "shl": "spu_shl", "shlh": "spu_shlh", "rot": "spu_rot",
    "roth": "spu_roth", "rotm": "spu_rotm", "rothm": "spu_rothm", "rotma": "spu_rotma",
    "rotmah": "spu_rotmah", "fcmeq": "spu_fcmeq", "fcmgt": "spu_fcmgt",
}
_CALL1 = {      # rt = spu_op(ra)
    "clz": "spu_clz", "cntb": "spu_cntb", "fsmb": None, "fsmh": "spu_fsmh", "fsm": "spu_fsm",
    "gbb": "spu_gbb", "gbh": "spu_gbh", "gb": "spu_gb", "xsbh": "spu_xsbh", "xshw": "spu_xshw",
    "xswd": "spu_xswd", "orx": "spu_orx", "frest": "spu_frest", "frsqest": "spu_frsqest",
    "fesd": "spu_fesd", "frds": "spu_frds",
}
_CALL_T = {     # rt = spu_op(ra, rb, rt)
    "addx": "spu_addx", "cgx": "spu_cgx", "sfx": "spu_sfx", "bgx": "spu_bgx",
    "dfma": "spu_dfma", "dfms": "spu_dfms", "dfnms": "spu_dfnms", "dfnma": "spu_dfnma",
}
_RRR_CALL = {"selb": "spu_selb", "shufb": "spu_shufb", "mpya": "spu_mpya", "fma": "spu_fma",
             "fms": "spu_fms", "fnms": "spu_fnms"}
_QSHIFT_REG = {  # rt = spu_op_n(ra, count from rb's preferred slot)
    "shlqbi": ("spu_shlqbi_n", ""), "rotqbi": ("spu_rotqbi_n", ""), "rotqmbi": ("spu_rotqmbi_n", ""),
    "shlqby": ("spu_shlqby_n", ""), "rotqby": ("spu_rotqby_n", ""), "rotqmby": ("spu_rotqmby_n", ""),
    "shlqbybi": ("spu_shlqby_n", " >> 3"), "rotqbybi": ("spu_rotqby_n", " >> 3"),
    "rotqmbybi": ("spu_rotqmby_n", " >> 3"),
}
_QSHIFT_IMM = {"shlqbii": "spu_shlqbi_n", "rotqbii": "spu_rotqbi_n", "rotqmbii": "spu_rotqmbi_n",
               "shlqbyi": "spu_shlqby_n", "rotqbyi": "spu_rotqby_n", "rotqmbyi": "spu_rotqmby_n"}
_SHIFT_IMM = {"shli": "spu_shli", "shlhi": "spu_shlhi", "roti": "spu_roti", "rothi": "spu_rothi",
              "rotmi": "spu_rotmi", "rothmi": "spu_rothmi", "rotmai": "spu_rotmai", "rotmahi": "spu_rotmahi"}
_INSERT = {"cbd": 1, "chd": 2, "cwd": 4, "cdd": 8, "cbx": 1, "chx": 2, "cwx": 4, "cdx": 8}
_INSERT_ALIGN = {1: 0xF, 2: 0xE, 4: 0xC, 8: 0x8}

_IMM10 = {      # rt = ra OP imm, by element view; imm from I10
    "ai": ("w", "+", "w"), "ahi": ("h", "+", "h"), "andi": ("w", "&", "w"), "andhi": ("h", "&", "h"),
    "andbi": ("b", "&", "b"), "ori": ("w", "|", "w"), "orhi": ("h", "|", "h"), "orbi": ("b", "|", "b"),
    "xori": ("w", "^", "w"), "xorhi": ("h", "^", "h"), "xorbi": ("b", "^", "b"),
}
_IMM10_SUB = {"sfi": "w", "sfhi": "h"}
_CMP10 = {      # rt = mask(ra OP imm)
    "ceqi": ("sw", "==", "w", "sw"), "ceqhi": ("sh", "==", "h", "sh"), "ceqbi": ("sb", "==", "b", "sb"),
    "cgti": ("sw", ">", "w", "sw"), "cgthi": ("sh", ">", "h", "sh"), "cgtbi": ("sb", ">", "b", "sb"),
    "clgti": ("w", ">", "w", "sw"), "clgthi": ("h", ">", "h", "sh"), "clgtbi": ("b", ">", "b", "sb"),
}
_HALT = {"heq": "==", "hgt": ">", "hlgt": ">", "heqi": "==", "hgti": ">", "hlgti": ">"}
_NOPS = {"lnop", "nop", "sync", "dsync", "hbr", "hbra", "hbrr", "mtspr", "fscrwr"}


def _splat(view, value):
    return {"w": f"spu_w((uint32_t){value})", "h": f"spu_h((uint16_t){value})",
            "b": f"spu_b((uint8_t){value})"}[view]


def emit(ins, off, prog):
    """C++ statements for the instruction at offset off; a list of lines."""
    op, f = ins.op, ins.f
    pc = prog.base + off
    rt, ra, rb = f.get("rt"), f.get("ra"), f.get("rb")
    R = _r

    def goto(target):
        t = (target - prog.base) & 0xFFFFFFFF
        if t < prog.size and t % 4 == 0:
            return f"goto L_{t:05X};"
        return f"{{ pc = 0x{target & 0x3FFFF:05X}u; goto dispatch; }}"

    def indirect(reg):
        return f"{{ pc = {R(reg)}.u32[3] & 0x3FFFCu; goto dispatch; }}"

    if op in _NOPS:
        return []
    if op == "illegal":
        return [f"spu_illegal(s, 0x{pc:05X}u, 0x{ins.word:08X}u);"]
    if op in ("stop", "stopd"):
        return [f"if (spu_stop(s, 0x{pc:05X}u, 0x{f.get('code', 0):X}u)) return SPU_STOPPED;"]

    # loads and stores
    if op == "lqd":
        return [f"{R(rt)} = spu_lq(s, {R(ra)}.u32[3] + 0x{(f['i10'] << 4) & 0xFFFFFFFF:X}u);"]
    if op == "stqd":
        return [f"spu_stq(s, {R(ra)}.u32[3] + 0x{(f['i10'] << 4) & 0xFFFFFFFF:X}u, {R(rt)});"]
    if op == "lqx":
        return [f"{R(rt)} = spu_lq(s, {R(ra)}.u32[3] + {R(rb)}.u32[3]);"]
    if op == "stqx":
        return [f"spu_stq(s, {R(ra)}.u32[3] + {R(rb)}.u32[3], {R(rt)});"]
    if op in ("lqa", "lqr", "stqa", "stqr"):
        i = _sext(f["i16"], 16) << 2
        a = (pc + i if op in ("lqr", "stqr") else i) & 0x3FFF0
        if op[0] == "l":
            return [f"{R(rt)} = spu_lq(s, 0x{a:05X}u);"]
        return [f"spu_stq(s, 0x{a:05X}u, {R(rt)});"]
    if op in _INSERT:
        n = _INSERT[op]
        if op.endswith("d"):
            t = f"({R(ra)}.u32[3] + 0x{f['i7'] & 0xFFFFFFFF:X}u)"
        else:
            t = f"({R(ra)}.u32[3] + {R(rb)}.u32[3])"
        return [f"{R(rt)} = spu_insert_mask({t} & 0x{_INSERT_ALIGN[n]:X}u, {n});"]

    # constants
    if op == "il":
        return [f"{R(rt)} = spu_w(0x{_sext(f['i16'], 16) & 0xFFFFFFFF:X}u);"]
    if op == "ilh":
        return [f"{R(rt)} = spu_h(0x{f['i16']:X}u);"]
    if op == "ilhu":
        return [f"{R(rt)} = spu_w(0x{f['i16'] << 16:X}u);"]
    if op == "iohl":
        return [f"{R(rt)}.w |= 0x{f['i16']:X}u;"]
    if op == "ila":
        return [f"{R(rt)} = spu_w(0x{f['i18']:X}u);"]
    if op == "fsmbi":
        return [f"{R(rt)} = spu_fsmb16(0x{f['i16']:X}u);"]
    if op == "fsmb":
        return [f"{R(rt)} = spu_fsmb16({R(ra)}.u32[3] & 0xFFFFu);"]

    # branches
    if op in ("br", "bra"):
        return [goto(_target(ins, pc))]
    if op in ("brsl", "brasl"):
        return [f"{R(rt)} = spu_slot(0x{pc + 4:05X}u);", goto(_target(ins, pc))]
    if op in ("brz", "brnz", "brhz", "brhnz"):
        v = f"{R(rt)}.u32[3]" if op in ("brz", "brnz") else f"({R(rt)}.u32[3] & 0xFFFFu)"
        cond = "==" if op in ("brz", "brhz") else "!="
        return [f"if ({v} {cond} 0) {goto(_target(ins, pc))}"]
    if op in ("bi", "iret"):
        return [indirect(ra)]
    if op == "bisl":
        return [f"{{ uint32_t t = {R(ra)}.u32[3] & 0x3FFFCu; {R(rt)} = spu_slot(0x{pc + 4:05X}u); pc = t; goto dispatch; }}"]
    if op == "bisled":
        raise Unsupported("bisled")
    if op in ("biz", "binz", "bihz", "bihnz"):
        v = f"{R(rt)}.u32[3]" if op in ("biz", "binz") else f"({R(rt)}.u32[3] & 0xFFFFu)"
        cond = "==" if op in ("biz", "bihz") else "!="
        return [f"if ({v} {cond} 0) {indirect(ra)}"]

    # halts
    if op in _HALT:
        cmp = _HALT[op]
        if op.endswith("i"):
            imm = f["i10"]
            b = f"(int32_t){imm}" if op != "hlgti" else f"0x{imm & 0xFFFFFFFF:X}u"
        else:
            b = f"(int32_t){R(rb)}.u32[3]" if op != "hlgt" else f"{R(rb)}.u32[3]"
        a = f"(int32_t){R(ra)}.u32[3]" if op not in ("hlgt", "hlgti") else f"{R(ra)}.u32[3]"
        return [f"if ({a} {cmp} {b}) spu_halt(s, 0x{pc:05X}u);"]

    # channels
    if op == "rdch":
        return [f"{R(rt)} = spu_slot(spu_rdch(s, {ra}));"]
    if op == "rchcnt":
        return [f"{R(rt)} = spu_slot(spu_rchcnt(s, {ra}));"]
    if op == "wrch":
        return [f"spu_wrch(s, {ra}, {R(rt)}.u32[3]);"]
    if op in ("mfspr", "fscrrd"):
        return [f"{R(rt)} = spu_w(0);"]

    # arithmetic and logic
    if op in _BINARY:
        v = _BINARY_FIELD.get(op, "w")
        return [f"{R(rt)}.{v} = " + _BINARY[op].format(ra=R(ra), rb=R(rb)) + ";"]
    if op in _COMPARE:
        av, cmp, mv = _COMPARE[op]
        bv = av
        return [f"{R(rt)}.{mv} = ({R(ra)}.{av} {cmp} {R(rb)}.{bv});"]
    if op in _CALL2:
        return [f"{R(rt)} = {_CALL2[op]}({R(ra)}, {R(rb)});"]
    if op in _CALL1:
        return [f"{R(rt)} = {_CALL1[op]}({R(ra)});"]
    if op in _CALL_T:
        return [f"{R(rt)} = {_CALL_T[op]}({R(ra)}, {R(rb)}, {R(rt)});"]
    if op in ("mpyhha", "mpyhhau"):
        call = "spu_mpyhh" if op == "mpyhha" else "spu_mpyhhu"
        return [f"{R(rt)}.w += {call}({R(ra)}, {R(rb)}).w;"]
    if op in _RRR_CALL:
        return [f"{R(rt)} = {_RRR_CALL[op]}({R(ra)}, {R(rb)}, {R(f['rc'])});"]
    if op in _QSHIFT_REG:
        fn, sh = _QSHIFT_REG[op]
        return [f"{R(rt)} = {fn}({R(ra)}, {R(rb)}.u32[3]{sh});"]
    if op in _QSHIFT_IMM:
        return [f"{R(rt)} = {_QSHIFT_IMM[op]}({R(ra)}, 0x{f['i7'] & 0xFFFFFFFF:X}u);"]
    if op in _SHIFT_IMM:
        return [f"{R(rt)} = {_SHIFT_IMM[op]}({R(ra)}, {f['i7']});"]
    if op in _IMM10:
        view, o, _ = _IMM10[op]
        imm = f["i10"]
        if view == "b":
            imm &= 0xFF
        return [f"{R(rt)}.{view} = {R(ra)}.{view} {o} {_splat(view, imm)}.{view};"]
    if op in _IMM10_SUB:
        view = _IMM10_SUB[op]
        imm = f["i10"]
        return [f"{R(rt)}.{view} = {_splat(view, imm)}.{view} - {R(ra)}.{view};"]
    if op in _CMP10:
        av, cmp, sv, mv = _CMP10[op]
        imm = f["i10"]
        if sv == "b":
            imm &= 0xFF
        return [f"{R(rt)}.{mv} = ({R(ra)}.{av} {cmp} {_splat(sv, imm)}.{av});"]
    if op == "mpyi":
        return [f"{R(rt)} = spu_mpyi({R(ra)}, {f['i10']});"]
    if op == "mpyui":
        return [f"{R(rt)} = spu_mpyui({R(ra)}, {f['i10']});"]
    if op in ("cflts", "cfltu", "csflt", "cuflt"):
        return [f"{R(rt)} = spu_{op}({R(ra)}, {f['i8']}u);"]
    if op == "fi":
        return [f"{R(rt)} = {R(rb)};"]
    raise Unsupported(op)


def _target(ins, pc):
    from ..spu import target
    return target(ins, pc)


_LINKING = {"brsl", "brasl", "bisl", "bisled"}
_GOTO = re.compile(r"goto L_([0-9A-F]{5});")


def dispatch_targets(img, prog):
    """The addresses an indirect branch of an SPU ELF's program can reach, as
    offsets into it: its entry, every return address (after a branch that
    links), and every code address the program holds as a constant, in an
    immediate load (`ila`, `il`, `ilhu` with `iohl`) or as a word of any of
    the ELF's loaded segments (jump tables, function pointers). A job
    relocates itself and computes its addresses, so a job keeps every
    instruction as a target (None), and so does a module, whose loader may
    rewrite the addresses it holds."""
    if not prog.elf or prog.module:
        return None
    lo, n = prog.base, prog.size
    out = set()

    def add(a):
        if 0 <= a - lo < n and a % 4 == 0:
            out.add(a - lo)
    add(img.u32(prog.elf + 24))
    hi_part = {}
    for off in range(0, n, 4):
        ins = decode(img.u32(prog.ea + off))
        op, f = ins.op, ins.f
        if op in _LINKING:
            add(lo + off + 4)
        elif op == "ila":
            add(f["i18"])
        elif op == "il":
            add(_sext(f["i16"], 16) & 0xFFFFFFFF)
        elif op == "ilhu":
            hi_part[f["rt"]] = f["i16"] << 16
        elif op == "iohl" and f["rt"] in hi_part:
            add(hi_part.pop(f["rt"]) | f["i16"])
    phoff, phnum = img.u32(prog.elf + 28), img.u32(prog.elf + 44) >> 16
    for i in range(phnum):
        ph = prog.elf + phoff + 32 * i
        if img.u32(ph) != 1:
            continue
        off, filesz = img.u32(ph + 4), img.u32(ph + 16)
        for k in range(0, filesz & ~3, 4):
            add(img.u32(prog.elf + off + k))
    return out


def translate(img, prog, comments=False, step=False):
    """The C++ function for one program, and the operations it could not translate."""
    unsupported = {}
    targets = dispatch_targets(img, prog)
    body, used = [], set()
    for off in range(0, prog.size, 4):
        w = img.u32(prog.ea + off)
        ins = decode(w)
        try:
            code = emit(ins, off, prog)
        except Unsupported as ex:
            unsupported[str(ex)] = unsupported.get(str(ex), 0) + 1
            code = [f"spu_illegal(s, 0x{prog.base + off:05X}u, 0x{w:08X}u);"]
        for x in code:
            for m in _GOTO.finditer(x):
                used.add(int(m.group(1), 16))
        body.append((off, ins, code))
    cases = range(0, prog.size, 4) if targets is None else sorted(targets)
    used.update(cases)
    lines = [f"// {prog.name}: {prog.size:#x} bytes from {prog.ea:#010x}, at local store {prog.base:#x}"
             + ("" if targets is None else f", {len(targets)} indirect-branch targets"),
             f"uint32_t {prog.fname}(SPUContext& s, uint32_t pc) {{",
             "dispatch:",
             "    SPU_DISPATCH(s, pc);",
             f"    switch ((pc - 0x{prog.base:X}u) >> 2) {{"]
    for off in cases:
        lines.append(f"    case {off >> 2}: goto L_{off:05X};")
    if targets is None:
        lines.append("    default: return pc;")
    else:
        lines.append(f"    default: return pc - 0x{prog.base:X}u < 0x{prog.size:X}u ? spu_missed(s, pc) : pc;")
    lines.append("    }")
    for off, ins, code in body:
        if off in used:
            label = f"L_{off:05X}:"
            if comments:
                label += f"  // {ins.op}"
            lines.append(label)
        if step:
            lines.append(f"    SPU_STEP(s, 0x{prog.base + off:05X}u);")
        lines.extend("    " + x for x in code)
    lines.append(f"    pc = 0x{prog.base + prog.size:05X}u;")
    lines.append("    goto dispatch;")
    lines.append("}\n")
    return "\n".join(lines), unsupported
