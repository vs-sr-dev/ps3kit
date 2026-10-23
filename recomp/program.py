"""The program model a recompiler needs: units, entry points, labels, switch tables.

A PS3 executable is stripped, but it is not silent: every function whose
address is ever taken has a descriptor in the `.opd`, and that is nearly all
of them (69,691 on Ragnarok Odyssey ACE). Discovery starts there:

- **Units** begin at every descriptor's entry and at every `bl` target (the
  static functions no pointer reaches: 6,957 more on that game), and run to
  the next unit. The code ends where the last unit reachable this way ends:
  the import stubs, `.sceStub.text`, are called by `bl` and close it.
- **Entry points** are the addresses a C++ function must exist for: every
  unit start, every `b`/`bc` target outside its own unit, and (to a fixed
  point) every backward branch out of such an extra entry's body. An extra
  entry gets its own C++ function from the entry to the end of its unit.
- **Switch tables.** The compiler's pattern is `lwz rT, off(r2)` (the table's
  address, from the TOC), `lwzx` or `lwax` of the scaled index, `add` of the
  table base, `mtctr`, `bctr`: entries are offsets from the table. The TOC
  a unit runs with is its descriptor's; a unit no descriptor names inherits
  its first caller's. The bound is the `cmplwi` on the index before it. A
  case may lie outside the unit that holds the `bctr` (the compiler shares
  blocks between functions): such a target becomes an entry, reached by a
  tail call.

Discovery works on raw words, never on a decoded program held in memory:
two and a half million instructions do not fit as Python objects.
"""
import bisect
from array import array

from .. import ppu

_TERMINATORS = {0x4E800020, 0x4E800420}           # blr, bctr


class Unit:
    __slots__ = ("start", "end", "toc")

    def __init__(self, start, end, toc):
        self.start, self.end, self.toc = start, end, toc

    def __contains__(self, a):
        return self.start <= a < self.end


class Program:
    def __init__(self, img, log=print):
        self.img, self.log = img, log
        seg = img.text_segments()[0]
        self.base = seg.vaddr
        self.w = array("I", seg.data[: len(seg.data) & ~3])
        self.w.byteswap()
        opd = img.opd()
        self.opd_toc = {}
        for e, t in opd:
            self.opd_toc.setdefault(e, t)
        self.lo = min(self.opd_toc)
        self.hi, starts, callers = self._extent(set(self.opd_toc))
        self.units = self._units(starts, callers)
        self._starts = [u.start for u in self.units]
        self.tables = {}                  # bctr address -> [targets]
        self.unresolved_tables = []
        self._resolve_tables()
        self.entries = self._entries()

    # --- words -----------------------------------------------------------------------

    def word(self, a):
        return self.w[(a - self.base) >> 2]

    def decode(self, a):
        return ppu.decode(self.word(a))

    def ins(self, start, end):
        return [(a, ppu.decode(self.word(a))) for a in range(start, end, 4)]

    def _scan_end(self, a):
        """The address after the first unconditional terminator at or after `a`."""
        while True:
            w = self.word(a)
            a += 4
            if w in _TERMINATORS or (w >> 26 == 18 and not w & 1):
                return a

    @staticmethod
    def _rel(w, a):
        if w >> 26 == 18:
            li = w & 0x03FFFFFC
            if li & 0x02000000:
                li -= 0x04000000
            return li if w & 2 else (a + li) & 0xFFFFFFFF
        bd = w & 0xFFFC
        if bd & 0x8000:
            bd -= 0x10000
        return bd if w & 2 else (a + bd) & 0xFFFFFFFF

    # --- the code range and the units --------------------------------------------------

    def _extent(self, starts):
        """(end of code, unit starts, {bl target: first caller}): grown until
        every call target is inside the code range."""
        callers = {}
        hi = self._scan_end(max(starts))
        done = self.lo
        while True:
            for a in range(done, hi, 4):
                w = self.word(a)
                if w >> 26 == 18 and w & 1:                   # bl
                    t = self._rel(w, a)
                    if t not in starts:
                        starts.add(t)
                        callers[t] = a
            done = hi
            top = max(starts)
            if top < hi:
                return hi, starts, callers
            hi = self._scan_end(top)

    def _units(self, starts, callers):
        order = sorted(s for s in starts if self.lo <= s < self.hi)
        units = []
        for i, s in enumerate(order):
            end = order[i + 1] if i + 1 < len(order) else self.hi
            units.append(Unit(s, end, self.opd_toc.get(s)))
        # a static function runs with its first caller's TOC
        starts_ = [u.start for u in units]
        for _ in range(4):
            for u in units:
                if u.toc is None and u.start in callers:
                    k = bisect.bisect_right(starts_, callers[u.start]) - 1
                    if k >= 0:
                        u.toc = units[k].toc
        for u in units:                                   # trailing zero padding
            while u.end > u.start + 4 and self.word(u.end - 4) == 0:
                u.end -= 4
        return units

    def unit_at(self, a):
        i = bisect.bisect_right(self._starts, a) - 1
        return self.units[i] if i >= 0 and a in self.units[i] else None

    # --- switch tables -----------------------------------------------------------------

    def _resolve_tables(self):
        img = self.img
        for u in self.units:
            for a in range(u.start, u.end, 4):
                if self.word(a) != 0x4E800420:
                    continue
                back = [(a - 4 * k, self.decode(a - 4 * k)) for k in range(1, 13)
                        if a - 4 * k >= u.start]
                load = self._entry_load(back)
                if load is None:
                    continue
                base_reg, toc_off, scaled = load
                relative = any(i.op == "add" and base_reg in (i.f["A"], i.f["B"]) for _, i in back)
                bound = self._bound(a, u, self._index_regs(back, scaled))
                targets = None
                if base_reg is not None and u.toc is not None:
                    try:
                        table = img.u32((u.toc + toc_off) & 0xFFFFFFFF)
                        targets = self._read_table(table, relative, bound, u)
                    except KeyError:
                        targets = None
                if targets:
                    self.tables[a] = targets
                else:
                    self.unresolved_tables.append(a)

    @staticmethod
    def _entry_load(back):
        """(table base register, its TOC offset, the scaled index registers)
        for the load of a switch table's entry, or None if `back` (the words
        before a `bctr`, nearest first) loads none. Two shapes: `lwzx`/`lwax`
        of the scaled index off the base, and the unoptimised one, `add` of
        the two and then `lwz 0(sum)`. The base is a `lwz` off r2."""
        def toc_base(regs, after):
            for _, i in back[after:]:
                if i.op == "lwz" and i.f["A"] == 2 and i.f["D"] in regs:
                    return i.f["D"], i.f["d"]
            return None, None
        for k, (_, i) in enumerate(back):
            if i.op in ("lwzx", "lwax"):
                pair = {i.f["A"], i.f["B"]}
                base, off = toc_base(pair, k + 1)
                return base, off, pair - {base}
            if (i.op == "lwz" and i.f["d"] == 0 and i.f["A"] != 1
                    and any(x.op == "add" and i.f["D"] in (x.f["A"], x.f["B"])
                            for _, x in back[:k])):       # an offset, added to a base
                for j in range(k + 1, len(back)):
                    add = back[j][1]
                    if add.op == "add" and add.f["D"] == i.f["A"]:
                        pair = {add.f["A"], add.f["B"]}
                        base, off = toc_base(pair, j + 1)
                        if base is not None:
                            return base, off, pair - {base}
                        break
        return None

    def _index_regs(self, back, scaled):
        """The registers that hold the switch index: the one the entry load
        scales, and what it was scaled and zero-extended from (`rldic`,
        `rldicr`, `rlwinm`, `rldicl`, `mr`)."""
        regs = set(scaled)
        for _, i in back:                       # nearest first, in this block
            if self._block_end(i.word, i):
                break
            if i.op in ("rldic", "rldicr", "rlwinm", "rldicl") and i.f["A"] in regs:
                regs.add(i.f["S"])
            elif i.op == "or" and i.f["S"] == i.f["B"] and i.f["A"] in regs:
                regs.add(i.f["S"])
        return regs

    def _block_end(self, w, i):
        """Does this word end the straight-line block that runs into the next
        one? An unconditional branch, a return, a `bctr`, or data."""
        return (w in _TERMINATORS or (w >> 26 == 18 and not w & 1)
                or (i.op in ("bclr", "bcctr") and i.f["BO"] & 0x14 == 0x14)
                or i.op == "illegal")

    def _cmp_back(self, a, lo, regs):
        """(bound, block start) walking back from `a` inside its own block: the
        `cmplwi` on one of `regs` (any `cmplwi` when `regs` is None). A compare
        of a register masked or zero-extended from one of them counts too
        (`clrlwi r0, r9, 24; cmplwi r0, 45`, the index tested in a copy)."""
        pending = {}                            # compared register -> bound
        for k in range(1, 40):
            b = a - 4 * k
            if b < lo:
                return None, None
            w = self.word(b)
            i = ppu.decode(w)
            if self._block_end(w, i):
                return None, b + 4
            if i.op == "cmpli" and i.f["L"] == 0:
                if regs is None or i.f["A"] in regs:
                    return i.f["imm"] + 1, None
                pending.setdefault(i.f["A"], i.f["imm"] + 1)
            elif (regs is not None and i.op in ("rlwinm", "rldicl") and i.f["SH"] == 0
                  and i.f["A"] in pending and i.f["S"] in regs):
                return pending[i.f["A"]], None
        return None, None

    def _bound(self, a, u, regs):
        """How many entries the table at this `bctr` has: the bound the index is
        checked against before it. The check is in the `bctr`'s own block, or,
        when that block is reached by a branch (the compiler moves dispatch
        blocks after a return), before a branch to it. None if not found."""
        bound, start = self._cmp_back(a, u.start, regs)
        if bound is not None:
            return bound
        if start is not None:
            for b in range(u.start, u.end, 4):
                w = self.word(b)
                if w >> 26 in (16, 18) and not w & 1 and self._rel(w, b) == start:
                    got, _ = self._cmp_back(b + 4, u.start, regs)
                    if got is not None:
                        return got
        bound, _ = self._cmp_back(a, u.start, None)
        return bound

    def _read_table(self, table, relative, bound, u):
        out = []
        n = bound if bound is not None else 4096
        for k in range(n):
            try:
                v = self.img.u32(table + 4 * k)
            except KeyError:
                break
            t = (table + (v - (1 << 32) if v & 0x80000000 else v)) & 0xFFFFFFFF if relative else v
            if not self.lo <= t < self.hi or t % 4 or self.unit_at(t) is None:
                if bound is not None:
                    return None
                break
            out.append(t)
        return out or None

    # --- entry points -----------------------------------------------------------------

    def _entries(self):
        entries = {u.start for u in self.units}
        self.bad_targets = []
        new = set()
        for u in self.units:
            for a in range(u.start, u.end, 4):
                w = self.word(a)
                op = w >> 26
                if op not in (16, 18):
                    continue
                t = self._rel(w, a)
                if w & 1:                                 # bl / bcl
                    if op == 16 and t == a + 4:
                        continue                          # bcl to next: reads the PC
                    if t not in entries:
                        self.bad_targets.append((a, t))
                    continue
                if t in u:
                    continue
                if self.unit_at(t) is None:
                    self.bad_targets.append((a, t))
                elif t not in entries:
                    new.add(t)
        for a, ts in self.tables.items():
            u = self.unit_at(a)
            new.update(t for t in ts if t not in u and t not in entries)
        done = set()
        while new - entries:
            entries |= new
            more = set()
            for e in sorted(new - done):
                done.add(e)
                u = self.unit_at(e)
                if u is None or e == u.start:
                    continue
                for a in range(e, u.end, 4):
                    w = self.word(a)
                    if w >> 26 in (16, 18) and not w & 1:
                        t = self._rel(w, a)
                        if t in u and t < e and t not in entries:
                            more.add(t)
            for a, t in self._table_back_targets(entries):
                more.add(t)
            new = more
        return sorted(entries)

    def _table_back_targets(self, entries):
        """(bctr, target) for switch-table targets that lie before an extra
        entry whose body holds the bctr: from that C++ function they are
        tail calls, so they need entries of their own."""
        out = []
        extra = sorted(e for e in entries if self.unit_at(e) and self.unit_at(e).start != e)
        for a, ts in self.tables.items():
            u = self.unit_at(a)
            lo = bisect.bisect_left(extra, u.start)
            hi = bisect.bisect_right(extra, a)
            if lo == hi:
                continue
            first = extra[hi - 1]                         # the latest extra entry before the bctr
            out.extend((a, t) for t in ts if t < first and t not in entries)
        return out

    def body(self, entry):
        """(unit, end) of the C++ function for this entry."""
        u = self.unit_at(entry)
        return u, u.end

    def labels(self, entry, ins):
        """Addresses inside this entry's body that are jumped to."""
        u = self.unit_at(entry)
        out = set()
        for a, i in ins:
            t = ppu.branch_target(i, a)
            if t is not None and not i.f.get("LK") and entry <= t < u.end:
                out.add(t)
            if a in self.tables:
                out.update(x for x in self.tables[a] if entry <= x < u.end)
        return out


def code_ranges(img):
    p = Program(img, log=lambda *a: None)
    return [(u.start, u.end) for u in p.units]
