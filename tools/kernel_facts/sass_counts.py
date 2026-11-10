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
# The snapshot's columns for these facts, in order.
COLUMNS = ("loops", "k_iter", "mma", "mma_ones", "mma_shape") + COUNT_FIELDS[2:]


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
# Instructions that write no general register although a register may lead their operands.
NO_REGISTER_WRITE = ("ST", "RED", "BRA", "ISETP", "FSETP", "HSETP2", "DSETP", "LDGSTS", "EXIT",
                     "RET", "CALL", "BSSY", "BSYNC", "WARPSYNC", "BAR", "DEPBAR", "NOP", "PREEXIT",
                     "ACQBULK", "KILL", "BRX", "JMX", "JMP", "PLOP3")


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
    if base_opcode(opcode).startswith(NO_REGISTER_WRITE):
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
    else:
        width = 1
    m = REGISTER.match(dest)
    return {m.group(1) + str(int(m.group(2)) + k) for k in range(width)}


def ones_definition(guarded, opcode, tokens):
    """What an instruction makes of its destination as a 1.0-pair holder: ("const", Rd) for an
    unguarded move of the constant (MOV / UMOV / MOV32I Rd, imm; IMAD.MOV.U32 Rd, RZ, RZ, imm as
    Ampere and Ada spell it; HFMA2 Rd, -RZ, RZ, 1.875, 1.875: the two f16 halves, compared as
    written), ("copy", Rd, Rs) for an unguarded register copy in those forms, else None. A guarded
    definition is no definition here: written_registers clears it."""
    if guarded or len(tokens) < 2:
        return None
    base = base_opcode(opcode)
    move = base in ("MOV", "MOV32I", "UMOV") or opcode.startswith("IMAD.MOV")
    if move and all(t == "RZ" for t in tokens[1:-1]):
        if tokens[-1] in ONES:
            return ("const", tokens[0])
        if REGISTER.match(tokens[-1]) and tokens[-1] not in ("RZ", "URZ"):
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
    defs = [ones_definition(g, op, t) for (_, g, op, _), t in zip(body, tokens)]
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
        for f in COUNT_FIELDS:
            setattr(self, f, 0)

    def row(self):
        """The snapshot's columns (COLUMNS), as strings."""
        return [str(getattr(self, c)) for c in COLUMNS]

    @classmethod
    def from_row(cls, fields):
        """The counts of a snapshot row's loop columns; ValueError on a field that is no count."""
        if len(fields) != len(COLUMNS):
            raise ValueError(f"{len(fields)} loop fields, {len(COLUMNS)} columns")
        c = cls()
        for name, value in zip(COLUMNS, fields):
            setattr(c, name, value if name == "mma_shape" else int(value))
        return c

    def __eq__(self, other):
        return isinstance(other, LoopCounts) and self.row() == other.row()

    def __repr__(self):
        return "LoopCounts(" + " ".join(f"{c}={v}" for c, v in zip(COLUMNS, self.row())) + ")"


def loop_counts(body, tiles=1):
    """The K loop's LoopCounts of a function body (iter_functions); `tiles` the m16n8 tiles one
    warp accumulates (m_tiles * n_tiles * row sets), for k_iter."""
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
    return out
