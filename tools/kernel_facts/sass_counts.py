"""Instruction counts of a kernel's K loop, read off its SASS (#613 PR-3). No GPU.

The register-MMA kernel (generic_gemm.cuh, rmma_kernel) spends its time in one loop: the
double-buffered K loop whose body fetches the next K step's weights and activations while the
tensor cores consume the current one (`for (step ...; step += 2 * WK)`: two K steps per
iteration, a `break` between them for an odd count). The price wants what one iteration issues,
per pipe: the counts are the same on every card of an architecture and nearly the same across
architectures (the 05.11 census found the per-MMA counts of the 5070 Ti and the 5090 equal),
while the cycles each pipe takes are the card's facts.

How the loop is found (`loop_counts`):
1. The function's instructions are parsed from `cuobjdump -sass` text: offset, guard predicate,
   opcode, operands. A branch's target is the trailing hex offset of its operands.
2. Basic blocks and their edges: a block ends at a branch (BRA, BRA.U, BRA.DIV), an EXIT or a
   RET; a guarded or predicated branch also falls through (one on PT / UPT does not, one on !PT
   is never taken); CALL falls through (the callee is outside the loop's count, see below); an
   indirect branch (BRX, JMX, their uniform forms) makes the control flow unknown and the counts
   `loops 0`. Blocks the entry does not reach (a callee, the trailing self-branch) take no part.
3. Dominators over the blocks; a back edge is an edge into a block that dominates its source;
   the natural loop of a back edge is its header plus every block that reaches the source
   without passing the header; back edges to one header are one loop.
4. The K loop is the innermost loop holding tensor-core MMAs (HMMA, IMMA, QMMA, ...): the one
   whose body holds MMAs and no MMA-bearing loop inside it. Where the build has no MMA at all
   (the sm_70 build multiplies with FFMAs, NINFERNO_FMA_MMA) the innermost loop holding FFMAs
   stands in. Several such loops (the compiler kept two versions of the body) are reported by
   their number, and the one whose iteration is heaviest (step 5) is counted.
5. One iteration executes one path through the body, from the header to a block that branches
   back to it. The counts are those of the heaviest path (the most MMAs, then the most
   instructions; `longest_iteration`): the full iteration, which fetches both slots and loads
   every row (the guards skip only rows past the call's end and the last fetches), with the
   longest of the compiler's alternative versions of a step where it kept several (the
   group-tail path of the 32-value formats: the same MMAs and loads, a few more checks; Volta's
   emulated MMA over its lane-local path). Predicated instructions count: they issue. A CALL's
   callee (a slow path) is not counted; an MMA-free loop nested in the body counts once.

What is counted per iteration (`LoopCounts`): the MMAs and their shape; of them the min-term
MMAs, whose A operand is a register holding the 1.0 pair (0x3f803f80 BF16, 0x3c003c00 F16: the
activation sums of a Q4_K / Q5_K / Q4_1 / Q5_1 group, mma_step), followed through the function in
address order; the loads and stores by space, with the bytes a warp moves (32 lanes times the
access width; LDG global, LD generic, LDGSTS the asynchronous global-to-shared copy, LDS, LDSM
and STS shared, LDL and STL local, the spills); SHFL; the integer and the floating-point ALU
instructions; the uniform-datapath instructions; the rest (branches, barriers, convergence,
constant loads, global stores, NOP); and the total. `k_iter`, the K values one iteration covers
per warp, follows from the weight MMAs (the min-term ones advance no K): the sum of m*n*k over
them, divided by the 16x8 tile and the tiles the warp holds (m_tiles * n_tiles * row sets). It
is 0 where there is no tensor MMA, a weight MMA's shape is unknown, or the sum is not a whole
number of K; it equals 2 * 4 * kRun of the decoder where the loop was found right, which
`kernel_facts.py extract` checks across every row of a decoder.
"""

import re
from collections import Counter

INSTRUCTION = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
FUNCTION = re.compile(r"Function : (\S+)")
TARGET = re.compile(r"0x([0-9a-f]+)\s*$")
REGISTER = re.compile(r"^(U?R)(\d+)")

MMA_OPCODES = ("HMMA", "IMMA", "QMMA", "OMMA", "DMMA", "BMMA")
# m, n, k of an MMA shape as the opcode spells it (HMMA.16816.F32.BF16 is m16n8k16).
MMA_SHAPES = {
    "16816": (16, 8, 16),
    "1688": (16, 8, 8),
    "1684": (16, 8, 4),
    "884": (8, 8, 4),
    "16832": (16, 8, 32),
    "8816": (8, 8, 16),
    "16864": (16, 8, 64),
}
ONES = ("0x3f803f80", "0x3c003c00")
# The same pairs as HFMA2 spells them when it materialises a constant (two f16 halves: 0x3f80 is
# 1.875, 0x3c00 is 1).
ONES_HALVES = (("1.875", "1.875"), ("1", "1"))
INT_ALU = {
    "IADD", "IADD3", "IADD32I", "IMAD", "IMUL", "IMNMX", "ISETP", "LOP", "LOP3", "LOP32I", "LEA",
    "SHF", "SHL", "SHR", "PRMT", "SEL", "POPC", "FLO", "BFE", "BFI", "IABS", "INEG", "ICMP",
    "I2I", "I2IP", "ISCADD", "VABSDIFF", "VIMNMX", "VIADD", "VIADDMNMX", "IDP", "IDP4A", "MOV",
    "MOV32I", "P2R", "R2P", "PLOP3", "BMSK", "SGXT", "BREV",
}
FP_ALU = {
    "FFMA", "FFMA32I", "FMUL", "FMUL32I", "FADD", "FADD32I", "FMNMX", "FSETP", "FSEL", "FSET",
    "FCHK", "FRND", "MUFU", "F2F", "F2FP", "F2I", "I2F", "I2FP", "HADD2", "HMUL2", "HFMA2",
    "HSETP2", "HSET2", "HMNMX2", "FSWZADD", "DADD", "DMUL", "DFMA", "DSETP", "FFMA2", "FMUL2",
    "FADD2", "FMNMX2", "HADD2_32I", "HFMA2_32I",
}
LOADS = {"LDG": "ldg", "LD": "ld", "LDGSTS": "ldgsts", "LDS": "lds", "LDSM": "lds", "STS": "sts",
         "LDL": "local", "STL": "local"}
WIDTHS = ((".256", 32), (".128", 16), (".64", 8), (".U8", 1), (".S8", 1), (".U16", 2), (".S16", 2))

COUNT_FIELDS = ("mma", "mma_ones", "ldg", "ldg_bytes", "ld", "ld_bytes", "ldgsts",
                "ldgsts_bytes", "lds", "lds_bytes", "sts", "sts_bytes", "local", "local_bytes",
                "shfl", "int", "fp", "uniform", "other", "total")
# The critical path of one iteration (#613 PR-7b): the instructions on the longest chain of the
# body's register dataflow (a read depends on the latest earlier writer of the register within
# the iteration; the loads of the next step, consumed an iteration later, carry no edge), as
# counts per latency class along it. Which chain is the longest depends on the clocks of each
# class, so the path is chosen with the latencies its price uses: the measured ISA facts of the
# snapshot's architecture (chain_weights of its row in src/core/data/isa_facts.json, the L1 load
# at one resident warp), and with nominal ones (CP_NOMINAL) only where that row lacks a fact; the
# same chain_clocks then prices the counts. The L1 latency moves a little with the resident warps
# (w2, w4, w8), where another chain may be the longest: critical_path reports by how many clocks.
CP_FIELDS = ("cp_mma", "cp_ld", "cp_lds", "cp_shfl", "cp_int", "cp_fp", "cp_cvt", "cp_other")
CP_NOMINAL = {"cp_mma": 33.0, "cp_ld": 48.0, "cp_lds": 25.0, "cp_shfl": 27.0, "cp_int": 4.5,
              "cp_fp": 4.1, "cp_cvt": 6.0, "cp_other": 2.0}
CP_WARPS = (1, 2, 4, 8)  # the resident warps the ISA facts measure the L1 latency at
CVT_OPCODES = {"I2F", "F2F", "F2I", "I2FP", "F2FP", "I2I", "F2IP"}
# The snapshot's columns for these facts, in order.
COLUMNS = ("loops", "k_iter", "mma", "mma_ones", "mma_shape") + COUNT_FIELDS[2:] + CP_FIELDS


def iter_functions(lines, wanted=None):
    """(mangled name, [(offset, guarded, opcode, operands)]) for each Function block of
    `cuobjdump -sass` text (an iterable of lines), as the text is read; `wanted(name)` selects
    the functions parsed, the others are skipped line by line."""
    name, body = None, None
    for line in lines:
        if "Function" in line:
            m = FUNCTION.search(line)
            if m:
                if body is not None:
                    yield name, body
                name = m.group(1)
                body = [] if wanted is None or wanted(name) else None
                continue
        if body is None:
            continue
        m = INSTRUCTION.search(line)
        if m:
            guard = (m.group(2) or "").strip()
            body.append((int(m.group(1), 16), guard not in ("", "@PT"), m.group(3),
                         m.group(4).strip()))
    if body is not None:
        yield name, body


def parse_functions(lines, wanted=None):
    """{mangled name: body} of every function iter_functions yields."""
    return dict(iter_functions(lines, wanted))


def base_opcode(opcode):
    return opcode.split(".", 1)[0]


def access_bytes(opcode):
    """Bytes one lane moves in a load or store: the width suffix, 4 without one; an LDSM's
    matrices (.1, .2, .4: 4 bytes each)."""
    if opcode.startswith("LDSM"):
        return 16 if ".4" in opcode else 8 if ".2" in opcode else 4
    for suffix, width in WIDTHS:
        if suffix in opcode:
            return width
    return 4


def branch_predicate(operands):
    """The predicate operand of a branch ("!UP0", "UPT", "PT", ...), "" for none."""
    head = operands.split("0x")[0].replace(",", " ").split()
    return head[0] if head and "P" in head[0] else ""


def branch_edges(guarded, opcode, operands):
    """(takes the target, may fall through) of a branch: a guard or a predicate operand makes it
    conditional, PT / UPT unconditional, !PT / !UPT never taken, BRA.DIV divergent."""
    if opcode.startswith("BRA.DIV"):
        return True, True
    predicate = branch_predicate(operands)
    if predicate in ("!PT", "!UPT"):
        return False, True
    conditional = guarded or (predicate != "" and predicate not in ("PT", "UPT"))
    return True, conditional


def successors(body):
    """The control-flow successors of every instruction (indices into `body`), or None where the
    flow is unknown (an indirect branch, a branch to an offset outside the function)."""
    at = {offset: i for i, (offset, _, _, _) in enumerate(body)}
    out = []
    for i, (_, guarded, opcode, operands) in enumerate(body):
        nexts = []
        base = base_opcode(opcode)
        if base.startswith(("BRX", "JMX")):
            return None
        if base in ("BRA", "JMP"):
            m = TARGET.search(operands)
            if not m or int(m.group(1), 16) not in at:
                return None
            taken, falls = branch_edges(guarded, opcode, operands)
            if taken:
                nexts.append(at[int(m.group(1), 16)])
            if falls:
                nexts.append(i + 1)
        elif base in ("EXIT", "RET", "KILL") and not guarded:
            pass
        else:
            nexts.append(i + 1)
        out.append([n for n in nexts if n < len(body)])
    return out


def basic_blocks(body, succ):
    """[(first index, last index)] in address order and the edges between them (by block index).
    A leader is the entry, a successor of a branch, or the instruction after a block ender."""
    leaders = {0}
    for i, nexts in enumerate(succ):
        if base_opcode(body[i][2]) in ("BRA", "JMP", "EXIT", "RET", "KILL"):
            leaders.add(i + 1)
            leaders.update(nexts)
    starts = sorted(l for l in leaders if l < len(body))
    block_of = {}
    blocks = []
    for b, s in enumerate(starts):
        e = (starts[b + 1] if b + 1 < len(starts) else len(body)) - 1
        blocks.append((s, e))
        for i in range(s, e + 1):
            block_of[i] = b
    edges = [sorted({block_of[n] for n in succ[e]}) for _, e in blocks]
    return blocks, edges


def reachable(edges, root=0):
    """The blocks the entry reaches."""
    seen, stack = {root}, [root]
    while stack:
        for s in edges[stack.pop()]:
            if s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


def dominators(edges, live):
    """dom[b]: the blocks dominating b (b included) as a bit set, by the iterative data-flow
    form, over the reachable blocks `live` (an unreachable block dominates nothing and keeps the
    full set). Also the predecessor lists, within `live`."""
    n = len(edges)
    preds = [[] for _ in range(n)]
    for b in live:
        for s in edges[b]:
            preds[s].append(b)
    full = (1 << n) - 1
    dom = [full] * n
    dom[0] = 1
    changed = True
    while changed:
        changed = False
        for b in sorted(live):
            if b == 0:
                continue
            new = full
            for p in preds[b]:
                new &= dom[p]
            new |= 1 << b
            if new != dom[b]:
                dom[b] = new
                changed = True
    return dom, preds


def is_back_edge(dom, u, h):
    return bool((dom[u] >> h) & 1)


def natural_loops(edges, dom, preds, live):
    """{header: set of blocks} of every natural loop (back edges to one header merged)."""
    loops = {}
    for u in live:
        for h in edges[u]:
            if not is_back_edge(dom, u, h):
                continue
            members = loops.setdefault(h, {h})
            stack = [u]
            while stack:
                b = stack.pop()
                if b in members:
                    continue
                members.add(b)
                stack.extend(preds[b])
    return loops


def longest_iteration(edges, dom, header, members, weight):
    """The blocks of one iteration of the loop `members` (headed by `header`) on its heaviest
    path: from the header, over the loop's forward edges (the back edges dropped), to a block
    that branches back to the header, maximising the sum of `weight(block)` (a tuple, compared
    lexicographically). The body without its back edges is a DAG for the code nvcc emits; an
    MMA-free loop nested in the body is walked once."""
    forward = {b: [s for s in edges[b] if s in members and not is_back_edge(dom, b, s)]
               for b in members}
    order, seen, stack = [], {header}, [(header, iter(forward[header]))]
    while stack:  # iterative depth-first walk, post-order
        b, it = stack[-1]
        s = next(it, None)
        if s is None:
            order.append(stack.pop()[0])
        elif s not in seen:
            seen.add(s)
            stack.append((s, iter(forward[s])))
    best, choice = {}, {}
    for b in order:  # post-order: every forward successor is settled before its block
        own = weight(b)
        if header in edges[b]:
            best[b], choice[b] = own, None
        candidates = [(best[s], s) for s in forward[b] if s in best]
        if candidates:
            value, s = max(candidates)
            total = tuple(x + y for x, y in zip(own, value))
            if b not in best or total > best[b]:
                best[b], choice[b] = total, s
    if header not in best:
        return []
    path, b = [], header
    while b is not None:
        path.append(b)
        b = choice[b]
    return path


def mma_shape(opcode):
    """(m, n, k) of an MMA opcode, or None for a shape not in the table."""
    for part in opcode.split(".")[1:]:
        if part in MMA_SHAPES:
            return MMA_SHAPES[part]
    return None


def classify(opcode):
    """The count field of one instruction: mma, a load/store space, shfl, int, fp, uniform, other."""
    base = base_opcode(opcode)
    if base in MMA_OPCODES:
        return "mma"
    if base in LOADS:
        return LOADS[base]
    if base == "SHFL":
        return "shfl"
    if base.startswith("U"):
        return "uniform"
    if base in INT_ALU:
        return "int"
    if base in FP_ALU:
        return "fp"
    return "other"


OPERAND_REGISTER = re.compile(r"^[-~!|]*(U?R(?:Z|\d+))(?:\.\w+)*\|?$")
PREDICATE = re.compile(r"^!?U?P(?:T|\d+)$")
# Instructions that write no general register although a register may lead their operands: whole
# base opcodes (RED is a reduction to memory; REDUX, which writes a uniform register, is not one)
# and the prefixes of the store, branch and barrier families.
NO_REGISTER_WRITE = {"RED", "ISETP", "FSETP", "HSETP2", "DSETP", "LDGSTS", "EXIT", "RET", "CALL",
                     "BSSY", "BSYNC", "WARPSYNC", "DEPBAR", "NOP", "PREEXIT", "ACQBULK", "KILL",
                     "JMP", "PLOP3"}
NO_REGISTER_WRITE_PREFIXES = ("ST", "BRA", "BRX", "JMX", "BAR")
# Ops that write a 64-bit float pair whatever their suffix; F2F / I2F do when their result type
# (the first suffix) is F64.
FP64_WRITERS = {"DADD", "DFMA", "DMUL", "DMNMX"}
# A register operand written plain (no -, ~ or | modifier, no suffix but .reuse): a copy source.
PLAIN_REGISTER = re.compile(r"^U?R\d+(?:\.reuse)?$")


def operand_tokens(operands):
    """The comma-separated operands, each trimmed; a register loses its modifiers and suffixes
    (`R52.reuse`, `-RZ`, `R2.64` become R52, RZ, R2), anything else (an immediate, a constant
    bank, an address) stays as written."""
    out = []
    for f in operands.split(","):
        f = f.strip()
        if not f:
            continue
        m = OPERAND_REGISTER.match(f)
        out.append(m.group(1) if m else f)
    return out


def destination(opcode, tokens):
    """The register an instruction writes (after any predicate it also writes: `SHFL.IDX PT, R4`,
    `LOP3.LUT P0, R6`), or None."""
    base = base_opcode(opcode)
    if base in NO_REGISTER_WRITE or base.startswith(NO_REGISTER_WRITE_PREFIXES):
        return None
    for t in tokens:
        if PREDICATE.match(t):
            continue
        return t if REGISTER.match(t) and t not in ("RZ", "URZ") else None
    return None


def written_registers(opcode, operands):
    """The registers an instruction writes: its destination and the pair, quad or MMA accumulator
    it extends to; none for a store, a branch or a compare."""
    tokens = operand_tokens(operands)
    dest = destination(opcode, tokens)
    if dest is None:
        return set()
    base = base_opcode(opcode)
    raw = operands.split(",")[0]
    if base in MMA_OPCODES:
        width = 2 if ".F16" in opcode and ".F32" not in opcode else 4
    elif base == "LDSM":
        width = 4 if ".4" in opcode else 2 if ".2" in opcode else 1
    elif ".128" in opcode or ".256" in opcode:
        width = 4
    elif ".64" in opcode or ".WIDE" in opcode or ".64" in raw:
        width = 2
    elif base in FP64_WRITERS or (base in ("F2F", "I2F") and opcode.split(".")[1:2] == ["F64"]):
        width = 2
    else:
        width = 1
    m = REGISTER.match(dest)
    return {m.group(1) + str(int(m.group(2)) + k) for k in range(width)}


def ones_definition(guarded, opcode, tokens, operands=""):
    """What an instruction makes of its destination as a 1.0-pair holder: ("const", Rd) for an
    unguarded move of the constant (MOV / UMOV / MOV32I Rd, imm; IMAD.MOV.U32 Rd, RZ, RZ, imm as
    Ampere and Ada spell it; HFMA2 Rd, -RZ, RZ, 1.875, 1.875: the two f16 halves, compared as
    written), ("copy", Rd, Rs) for an unguarded register copy in those forms whose source is
    written plain (`-Rs`, `~Rs` or `|Rs|` make something else of the pair), else None. A guarded
    definition is no definition here: written_registers clears it. `operands` is the raw operand
    text the tokens came from; without it a copy is taken on the tokens alone."""
    if guarded or len(tokens) < 2:
        return None
    base = base_opcode(opcode)
    move = base in ("MOV", "MOV32I", "UMOV") or opcode.startswith("IMAD.MOV")
    if move and all(t == "RZ" for t in tokens[1:-1]):
        if tokens[-1] in ONES:
            return ("const", tokens[0])
        source = operands.split(",")[-1].strip() if operands else tokens[-1]
        if PLAIN_REGISTER.match(source) and REGISTER.match(tokens[-1]):
            return ("copy", tokens[0], tokens[-1])
    if base == "HFMA2" and len(tokens) == 5 and tokens[1:3] == ["RZ", "RZ"] and \
            tuple(tokens[3:5]) in ONES_HALVES:
        return ("const", tokens[0])
    return None


TOP, IS_ONES, NOT_ONES = 0, 1, 2


def meet(a, b):
    if a == TOP:
        return b
    if b == TOP or a == b:
        return a
    return NOT_ONES


def ones_mma_indices(body, succ):
    """Indices of the MMAs whose A operand holds the 1.0 pair on every path that reaches them: a
    reaching-definitions pass over the control flow (`succ`, successors()) for the registers a
    ones constant is ever moved into, and their copies. A register is ONES after an unguarded
    constant move or a copy of a ONES register, NOT after any other write (a guarded one too) and
    at the entry, and the meet of two paths is ONES only where both are; a redefinition later in
    the loop body reaches the loop's top through the back edge."""
    tokens = [operand_tokens(operands) for _, _, _, operands in body]
    defs = [ones_definition(g, op, t, operands) for (_, g, op, operands), t in zip(body, tokens)]
    tracked = {d[1] for d in defs if d and d[0] == "const"}
    grown = True
    while grown:  # the copies of tracked registers, transitively
        grown = False
        for d in defs:
            if d and d[0] == "copy" and d[2] in tracked and d[1] not in tracked:
                tracked.add(d[1])
                grown = True
    if not tracked:
        return set()
    preds = [[] for _ in body]
    for i, nexts in enumerate(succ):
        for n in nexts:
            preds[n].append(i)
    written = [written_registers(op, operands) & tracked for _, _, op, operands in body]
    order = sorted(tracked)
    index = {r: k for k, r in enumerate(order)}
    state_in = [[TOP] * len(order) for _ in body]
    state_out = [[TOP] * len(order) for _ in body]
    changed = True
    while changed:
        changed = False
        for i in range(len(body)):
            inc = [NOT_ONES] * len(order) if i == 0 else [TOP] * len(order)
            for p in preds[i]:
                inc = [meet(a, b) for a, b in zip(inc, state_out[p])]
            out = list(inc)
            for r in written[i]:
                out[index[r]] = NOT_ONES
            d = defs[i]
            if d and d[1] in tracked:
                if d[0] == "const":
                    out[index[d[1]]] = IS_ONES
                elif d[2] in tracked:
                    out[index[d[1]]] = inc[index[d[2]]]
            if inc != state_in[i] or out != state_out[i]:
                state_in[i], state_out[i] = inc, out
                changed = True
    found = set()
    for i, (_, _, opcode, _) in enumerate(body):
        if base_opcode(opcode) in MMA_OPCODES and len(tokens[i]) >= 2:
            a = tokens[i][1]
            if a in tracked and state_in[i][index[a]] == IS_ONES:
                found.add(i)
    return found


class LoopCounts:
    """The K loop's counts of one kernel (the module docstring). `loops` is the number of
    candidate loops (0: none found, the counts are then 0); `mma_shape` the MMA opcode, `-`
    without one, `a:n+b:m` where shapes mix."""

    def __init__(self):
        self.loops = 0
        self.k_iter = 0
        self.mma_shape = "-"
        # Clocks by which the critical path's chain falls short of the longest one under another
        # of the weight sets it was checked against (critical_path); not a column.
        self.cp_short = 0.0
        for f in COUNT_FIELDS + CP_FIELDS:
            setattr(self, f, 0)

    def row(self):
        """The snapshot's columns (COLUMNS), as strings."""
        return [str(getattr(self, c)) for c in COLUMNS]

    @classmethod
    def from_row(cls, fields):
        """The counts of a snapshot row's loop columns; ValueError on a field that is no count."""
        # A schema 2 row stops before the critical path's columns (CP_FIELDS), which read 0.
        if len(fields) not in (len(COLUMNS), len(COLUMNS) - len(CP_FIELDS)):
            raise ValueError(f"{len(fields)} loop fields, {len(COLUMNS)} columns")
        c = cls()
        for name, value in zip(COLUMNS, fields):
            setattr(c, name, value if name == "mma_shape" else int(value))
        return c

    def __eq__(self, other):
        return isinstance(other, LoopCounts) and self.row() == other.row()

    def __repr__(self):
        return "LoopCounts(" + " ".join(f"{c}={v}" for c, v in zip(COLUMNS, self.row())) + ")"


def loop_counts(body, tiles=1, weights=(CP_NOMINAL,)):
    """The K loop's LoopCounts of a function body (iter_functions); `tiles` the m16n8 tiles one
    warp accumulates (m_tiles * n_tiles * row sets), for k_iter; `weights` the clocks per latency
    class the critical path is chosen with and checked against (critical_path)."""
    out = LoopCounts()
    if not body:
        return out
    succ = successors(body)
    if succ is None:
        return out
    blocks, edges = basic_blocks(body, succ)
    live = reachable(edges)
    dom, preds = dominators(edges, live)
    loops = natural_loops(edges, dom, preds, live)
    has_mma = any(base_opcode(op) in MMA_OPCODES for _, _, op, _ in body)
    marker = (lambda op: base_opcode(op) in MMA_OPCODES) if has_mma else \
        (lambda op: base_opcode(op) == "FFMA")

    def indices(members):
        for b in sorted(members):
            s, e = blocks[b]
            yield from range(s, e + 1)

    def weight(b):
        s, e = blocks[b]
        return (sum(1 for i in range(s, e + 1) if marker(body[i][2])), e - s + 1)

    marked = {h: sum(weight(b)[0] for b in m) for h, m in loops.items()}
    candidates = [h for h, n in marked.items() if n > 0]
    innermost = [h for h in candidates
                 if not any(o != h and loops[o] < loops[h] for o in candidates)]
    iterations = {h: longest_iteration(edges, dom, h, loops[h], weight) for h in innermost}
    iterations = {h: p for h, p in iterations.items() if p}
    out.loops = len(iterations)
    if not iterations:
        return out
    path = max(iterations.values(),
               key=lambda p: tuple(sum(w) for w in zip(*(weight(b) for b in p))))
    ones = ones_mma_indices(body, succ) if has_mma else set()
    shapes = Counter()
    mnk_sum = 0
    unknown_shape = False
    for i in indices(path):
        opcode = body[i][2]
        kind = classify(opcode)
        out.total += 1
        if kind == "mma":
            out.mma += 1
            shapes[opcode] += 1
            if i in ones:
                out.mma_ones += 1
            else:
                shape = mma_shape(opcode)
                if shape:
                    mnk_sum += shape[0] * shape[1] * shape[2]
                else:
                    unknown_shape = True
        elif kind in ("ldg", "ld", "ldgsts", "lds", "sts", "local"):
            setattr(out, kind, getattr(out, kind) + 1)
            setattr(out, kind + "_bytes", getattr(out, kind + "_bytes") + 32 * access_bytes(opcode))
        else:
            setattr(out, kind, getattr(out, kind) + 1)
    if shapes:
        if len(shapes) == 1:
            out.mma_shape = next(iter(shapes))
        else:
            out.mma_shape = "+".join(f"{s}:{n}" for s, n in sorted(shapes.items()))
        per_tile = 16 * 8 * tiles
        if mnk_sum and not unknown_shape and mnk_sum % per_tile == 0:
            out.k_iter = mnk_sum // per_tile
    counts, out.cp_short = critical_path(body, list(indices(path)), weights)
    for name, n in counts.items():
        setattr(out, name, n)
    return out


def cp_class(opcode):
    """The latency class (CP_FIELDS) of an instruction."""
    base = base_opcode(opcode)
    if base in MMA_OPCODES:
        return "cp_mma"
    if base in CVT_OPCODES:
        return "cp_cvt"
    kind = classify(opcode)
    if kind in ("ldg", "ld", "local"):
        return "cp_ld"
    if kind == "lds":
        return "cp_lds"
    if kind == "shfl":
        return "cp_shfl"
    if kind == "int":
        return "cp_int"
    if kind == "fp":
        return "cp_fp"
    return "cp_other"


READ_REGISTER = re.compile(r"\bU?R\d+\b")


def chain_weights(isa, warps=1):
    """The clocks per latency class (CP_FIELDS) of an architecture's measured ISA facts (`isa`:
    {fact: value}, a row of src/core/data/isa_facts.json) at `warps` resident warps a scheduler
    (CP_WARPS): the MMA accumulator at the MMA's dependent latency, global and local loads at the
    L1 hit latency, shared loads at the shared-memory one, shuffles, the integer decode at the
    IMAD latency (LOP3 / PRMT / SHF are equal, IADD3 one clock more), the float math at the HFMA2
    latency, converts at the I2F / F2I one, the rest (MOV, branches, uniform) at the ALU floor.
    None where the row lacks one of them: no facts, never another architecture's."""
    try:
        return {"cp_mma": isa["mma_dependent_latency"], "cp_ld": isa[f"l1_latency_w{warps}"],
                "cp_lds": isa["shared_latency"], "cp_shfl": isa["lat_shfl"],
                "cp_int": isa["lat_imad"], "cp_fp": isa["lat_hfma2"],
                "cp_cvt": isa["lat_cvt_i2f_f2i"], "cp_other": isa["lat_imad"]}
    except KeyError:
        return None


def chain_clocks(counts, weights):
    """The clocks of a chain: its instructions per latency class (`counts`, any mapping with the
    CP_FIELDS) at `weights` clocks each."""
    return sum(counts[name] * weights[name] for name in CP_FIELDS)


def dependence_graph(body, order):
    """(classes, preds) of the instructions `order` (body indices, in iteration order): the
    latency class of each and the positions in `order` it depends on, ascending. Instruction i
    depends on the latest earlier instruction that wrote a register it reads (RZ / URZ aside; the
    destination is not a read, an accumulator is read as its own operand); a dependence on a
    register no earlier instruction wrote is a value from the previous iteration or the prologue
    and carries no edge."""
    classes, preds = [], []
    last_writer = {}
    for i in order:
        _, _, opcode, operands = body[i]
        tokens = operand_tokens(operands)
        written = written_registers(opcode, operands)
        dest = destination(opcode, tokens)
        sources = list(tokens)
        if dest is not None:
            sources.remove(dest)
        reads = {r for t in sources for r in READ_REGISTER.findall(t) if r not in ("RZ", "URZ")}
        k = len(classes)
        classes.append(cp_class(opcode))
        preds.append(sorted({last_writer[r] for r in reads if r in last_writer}))
        for r in written:
            last_writer[r] = k
    return classes, preds


def longest_chain(classes, preds, weights):
    """(clocks, counts per latency class) of the longest chain of a dependence_graph with each
    class at `weights` clocks; among equally long chains the one through the earliest
    instructions (the same on every run)."""
    dist, best_from = [], []
    for k, cls in enumerate(classes):
        pred, longest = -1, 0.0
        for p in preds[k]:
            if dist[p] > longest:
                pred, longest = p, dist[p]
        dist.append(longest + weights[cls])
        best_from.append(pred)
    counts = {name: 0 for name in CP_FIELDS}
    if not classes:
        return 0.0, counts
    k = max(range(len(classes)), key=lambda j: dist[j])
    clocks = dist[k]
    while k >= 0:
        counts[classes[k]] += 1
        k = best_from[k]
    return clocks, counts


def critical_path(body, order, weights=(CP_NOMINAL,)):
    """(counts, short): the counts per latency class (CP_FIELDS) along the longest chain of
    register dependences (dependence_graph) through the instructions `order`, each class at
    `weights[0]` clocks; and the most clocks by which that chain falls short of the longest one
    under any other of `weights` (0.0: it is the longest, or as long as it, under all of them)."""
    classes, preds = dependence_graph(body, order)
    _, counts = longest_chain(classes, preds, weights[0])
    short = 0.0
    for other in weights[1:]:
        short = max(short, longest_chain(classes, preds, other)[0] - chain_clocks(counts, other))
    return counts, (short if short > 1e-9 else 0.0)
