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
# body's dataflow through registers and predicates (a read depends on the earlier writers of the
# value within the iteration that may reach it: dependence_graph. Every register of an MMA
# fragment or a 64-bit pair counts, as do a guard, a compare's predicate and a carry; the loads
# of the next step, consumed an iteration later, carry no edge), as counts per latency class
# along it. Which chain is the longest depends on the clocks of each
# class, so the path is chosen with the latencies its price uses: the measured ISA facts of the
# snapshot's architecture (chain_weights of its row in src/core/data/isa_facts.json, the L1 load
# at one resident warp), and with nominal ones (CP_NOMINAL) only where that row lacks a fact; the
# same chain_clocks then prices the counts. The L1 latency moves a little with the resident warps
# (w2, w4, w8), where another chain may be the longest: critical_path reports by how many clocks.
CP_FIELDS = ("cp_mma", "cp_ld", "cp_lds", "cp_shfl", "cp_imad", "cp_iadd", "cp_lop3", "cp_prmt",
             "cp_shf", "cp_hfma2", "cp_ffma", "cp_cvt", "cp_f2f", "cp_other")
# One class per measured latency fact, so that no instruction is priced at another's clocks (on
# the Ampere and Ada cards an F2F takes 5 clocks where an I2F or an F2I takes 11.5, an IADD3 one
# clock more than an IMAD). The opcodes with a fact of their own:
CP_OPCODES = {
    "cp_iadd": {"IADD", "IADD3", "IADD32I"},
    "cp_lop3": {"LOP", "LOP3", "LOP32I"},
    "cp_prmt": {"PRMT"},
    "cp_shf": {"SHF", "SHL", "SHR"},
    "cp_cvt": {"I2F", "F2I", "I2FP", "F2IP", "I2I"},
    "cp_f2f": {"F2F", "F2FP"},
}
# The nominal clocks keep one value per family (integer, float, convert): no fact tells its
# members apart where they apply.
CP_NOMINAL = {"cp_mma": 33.0, "cp_ld": 48.0, "cp_lds": 25.0, "cp_shfl": 27.0, "cp_imad": 4.5,
              "cp_iadd": 4.5, "cp_lop3": 4.5, "cp_prmt": 4.5, "cp_shf": 4.5, "cp_hfma2": 4.1,
              "cp_ffma": 4.1, "cp_cvt": 6.0, "cp_f2f": 6.0, "cp_other": 2.0}
CP_WARPS = (1, 2, 4, 8)  # the resident warps the ISA facts measure the L1 latency at
# The snapshot's columns for these facts, in order.
COLUMNS = ("loops", "k_iter", "mma", "mma_ones", "mma_shape") + COUNT_FIELDS[2:] + CP_FIELDS


def iter_functions(lines, wanted=None):
    """(mangled name, [(offset, guard, opcode, operands)]) for each Function block of
    `cuobjdump -sass` text (an iterable of lines), as the text is read; `wanted(name)` selects
    the functions parsed, the others are skipped line by line. The guard is the predicate an
    instruction runs under as written, without its `@` (`P0`, `!UP1`); "" for none or PT, so it
    reads as a flag too."""
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
            guard = (m.group(2) or "").strip().lstrip("@")
            body.append((int(m.group(1), 16), "" if guard == "PT" else guard, m.group(3),
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


def writes_predicates_only(base):
    """Whether a base opcode writes predicates and no register: the compares (ISETP, FSETP,
    DSETP, HSETP2), FCHK and the predicate logic op PLOP3, with their uniform forms (UISETP,
    UPLOP3)."""
    core = base[1:] if base.startswith("U") else base
    return core.endswith(("SETP", "SETP2")) or core in ("PLOP3", "FCHK")


def writes_no_register(base):
    return (base in NO_REGISTER_WRITE or base.startswith(NO_REGISTER_WRITE_PREFIXES)
            or writes_predicates_only(base))


def destination(opcode, tokens):
    """The register an instruction writes (after any predicate it also writes: `SHFL.IDX PT, R4`,
    `LOP3.LUT P0, R6`), or None."""
    base = base_opcode(opcode)
    if writes_no_register(base):
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
    elif base == "CS2R" and ".32" not in opcode:
        width = 2  # the 64-bit special-register read: `CS2R R112, SRZ` zeroes R112 and R113
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
    """The latency class (CP_FIELDS) of an instruction: the measured fact that prices it. An
    integer op with no fact of its own counts with the IMAD (the ALU latency LOP3, PRMT and SHF
    share on every measured card), an FP16x2 op with the HFMA2, any other float op with the
    FFMA."""
    base = base_opcode(opcode)
    if base in MMA_OPCODES:
        return "cp_mma"
    for name, opcodes in CP_OPCODES.items():
        if base in opcodes:
            return name
    kind = classify(opcode)
    if kind in ("ldg", "ld", "local"):
        return "cp_ld"
    if kind == "lds":
        return "cp_lds"
    if kind == "shfl":
        return "cp_shfl"
    if kind == "int":
        return "cp_imad"
    if kind == "fp":
        return "cp_hfma2" if base.startswith("H") else "cp_ffma"
    return "cp_other"


# A register inside an operand, with the suffixes it is written with (`R2.64` in an address).
READ_REGISTER = re.compile(r"\b(U?R)(\d+)((?:\.\w+)*)")
# The bits of one element of an MMA's A and B operands, by opcode; a type suffix overrides it.
MMA_ELEMENT_BITS = {"HMMA": 16, "IMMA": 8, "QMMA": 8, "OMMA": 4, "DMMA": 64, "BMMA": 1}
MMA_TYPE_BITS = {"TF32": 32, "S4": 4, "U4": 4}
WIDE_SUFFIXES = {"64": 2, "U64": 2, "S64": 2, "128": 4}
PREDICATE_REGISTERS = tuple(f"P{n}" for n in range(7))  # P7 is PT


def mma_fragments(opcode):
    """(A, B): the registers one lane holds of an MMA's A and B operands, which the instruction
    names by their first register only. A is m x k elements and B n x k over the 32 lanes of a
    warp (m16n8k16 of f16 or bf16: 4 and 2; m16n8k32 of s8: 4 and 2; m16n8k8 of tf32: 4 and 2);
    Volta's m8n8k4 keeps every element on four lanes, 2 and 2. (1, 1), the named registers
    alone, for a shape not in the table."""
    shape = mma_shape(opcode)
    if shape is None:
        return 1, 1
    parts = opcode.split(".")
    if parts[0] == "HMMA" and shape == (8, 8, 4):
        return 2, 2
    bits = MMA_ELEMENT_BITS[parts[0]]
    for part in parts[1:]:
        bits = MMA_TYPE_BITS.get(part, bits)
    m, n, k = shape
    return max(1, -(-m * k * bits // 1024)), max(1, -(-n * k * bits // 1024))


def immediate(field):
    """The value of a hexadecimal immediate operand (`0x20`), or None of anything else."""
    try:
        return int(field, 16) if field.startswith("0x") else None
    except ValueError:
        return None


def masked_predicates(mask):
    """The predicates the whole predicate register PR stands for in a P2R or an R2P: those whose
    bit its last operand, the mask, holds (`P2R R0, PR, RZ, 0x20` reads P5 alone); all seven
    where the mask is not an immediate."""
    bits = immediate(mask)
    if bits is None:
        return PREDICATE_REGISTERS
    return tuple(p for n, p in enumerate(PREDICATE_REGISTERS) if bits >> n & 1)


def source_width(opcode, position, count, accumulator):
    """The registers the register operand at `position` of `count` operands stands for when it
    is read: an MMA's fragments (mma_fragments; the accumulator operand as wide as the
    destination, `accumulator`), the pairs and quads of a 64- or 128-bit instruction (a store's
    data, IADD.64, SEL.64, an ISETP on U64; the addend of an IMAD.WIDE; the operands of the
    FP64 ops; the source of a convert from a 64-bit type), else the one named. SHF names both
    halves of its 64-bit operand itself."""
    base = base_opcode(opcode)
    parts = opcode.split(".")[1:]
    if base in MMA_OPCODES:
        a, b = mma_fragments(opcode)
        return {1: a, 2: b, 3: accumulator}.get(position, 1)
    if base in ("SHF", "SHL", "SHR"):
        return 1
    if base in CP_OPCODES["cp_cvt"] or base in CP_OPCODES["cp_f2f"]:
        return 2 if parts[1:2] in (["F64"], ["S64"], ["U64"]) else 1
    if base in FP64_WRITERS or base == "DSETP":
        return 2
    if "WIDE" in parts:
        return 2 if position == count - 1 else 1
    return max([WIDE_SUFFIXES[p] for p in parts if p in WIDE_SUFFIXES], default=1)


def reads_and_writes(guard, opcode, operands):
    """(reads, writes): the registers and predicates one instruction reads and writes, by name
    (R4, UR8, P0, UP1; never RZ, URZ, PT or UPT, the constants).

    Registers: the destination (written_registers) is written; every other register is read,
    inside an address too, as many as it stands for (source_width; `R2.64` in an address is R2
    and R3). An accumulator is read as its own operand.

    Predicates: a compare, FCHK or PLOP3 writes its leading predicates (two at most: PLOP3's
    other three are its sources); an instruction with a register destination writes the
    predicates before it (`LOP3.LUT P0, R6, ...`) and those right after it (the carries out of
    `IADD3 R4, P0, P1, ...`, `LEA R4, P0, ...`); any other predicate operand is read (the carry
    into `LEA.HI.X ..., P0`, SEL's selector, a branch's condition, the last operand of a
    compare). P2R reads and R2P writes the predicates of its mask (masked_predicates).

    A guard is read. Guarded by !PT the instruction does nothing."""
    if guard in ("!PT", "!UPT"):
        return set(), set()
    base = base_opcode(opcode)
    fields = [f.strip() for f in operands.split(",") if f.strip()]
    tokens = operand_tokens(operands)
    reads, writes = set(), written_registers(opcode, operands)
    lead = 0
    while lead < len(tokens) and PREDICATE.match(tokens[lead]):
        lead += 1
    if writes_predicates_only(base):
        written_at = set(range(min(lead, 2)))
    elif (not writes_no_register(base) and lead < len(tokens)
          and OPERAND_REGISTER.match(tokens[lead])):
        written_at = set(range(lead))
        j = lead + 1
        while j < len(tokens) and PREDICATE.match(tokens[j]):
            written_at.add(j)
            j += 1
    else:
        written_at = set()
    has_destination = destination(opcode, tokens) is not None
    accumulator = len(writes)
    for j, (field, token) in enumerate(zip(fields, tokens)):
        if PREDICATE.match(token):
            name = token.lstrip("!")
            if name not in ("PT", "UPT"):
                (writes if j in written_at else reads).add(name)
            continue
        if token == "PR":
            (writes if base == "R2P" else reads).update(masked_predicates(fields[-1]))
            continue
        if j == lead and has_destination:
            continue
        width = 1
        if OPERAND_REGISTER.match(token):
            width = source_width(opcode, j - lead, len(tokens) - lead, accumulator)
        for prefix, number, suffixes in READ_REGISTER.findall(field):
            n = max(width, 2 if ".64" in suffixes else 1)
            reads.update(prefix + str(int(number) + k) for k in range(n))
    if guard and guard.lstrip("!") not in ("PT", "UPT"):
        reads.add(guard.lstrip("!"))
    return reads, writes


def chain_weights(isa, warps=1):
    """The clocks per latency class (CP_FIELDS) of an architecture's measured ISA facts (`isa`:
    {fact: value}, a row of src/core/data/isa_facts.json) at `warps` resident warps a scheduler
    (CP_WARPS): the MMA accumulator at the MMA's dependent latency, global and local loads at the
    L1 hit latency, shared loads at the shared-memory one, and every other class at the lat_*
    fact of its instruction (cp_class): IMAD, IADD3, LOP3, PRMT, SHF, HFMA2, FFMA, SHFL, the
    I2F / F2I converts and the F32 / F16 ones; the rest (branches, uniform) at the IMAD's, the
    ALU floor. None where the row lacks one of them: no facts, never another architecture's."""
    try:
        return {"cp_mma": isa["mma_dependent_latency"], "cp_ld": isa[f"l1_latency_w{warps}"],
                "cp_lds": isa["shared_latency"], "cp_shfl": isa["lat_shfl"],
                "cp_imad": isa["lat_imad"], "cp_iadd": isa["lat_iadd"],
                "cp_lop3": isa["lat_lop3"], "cp_prmt": isa["lat_prmt"], "cp_shf": isa["lat_shf"],
                "cp_hfma2": isa["lat_hfma2"], "cp_ffma": isa["lat_ffma"],
                "cp_cvt": isa["lat_cvt_i2f_f2i"], "cp_f2f": isa["lat_cvt_f32_f16"],
                "cp_other": isa["lat_imad"]}
    except KeyError:
        return None


def chain_clocks(counts, weights):
    """The clocks of a chain: its instructions per latency class (`counts`, any mapping with the
    CP_FIELDS) at `weights` clocks each."""
    return sum(counts[name] * weights[name] for name in CP_FIELDS)


def single_predicate(field):
    """The predicate a one-bit immediate mask names (0x20: P5), or None."""
    bits = immediate(field)
    if bits is None or bits & (bits - 1) or not 0 < bits < 1 << len(PREDICATE_REGISTERS):
        return None
    return PREDICATE_REGISTERS[bits.bit_length() - 1]


class PredicateValues:
    """Which value each predicate holds while a body is walked in order, so that a guard the
    compiler saved and restored is known as the guard it was. A value is (identity, inverted):
    the predicate with the positions of the writers that may have made it, or the value it is
    a copy of, and whether it is that value's negation.

    The compiler keeps a predicate past its register in two ways. `P2R Rd, PR, RZ, 0x20` puts
    P5 alone into Rd, and `ISETP.NE.AND Pn, PT, Rd, RZ, PT` reads it back. Or one bit of a
    word: `LOP3.LUT R, R, ~bit, RZ, 0xc0` clears it, `@P LOP3.LUT R, R, bit, RZ, 0xfc` sets it
    where P holds (under @!P it keeps the negation), and `LOP3.LUT Pn, RZ, R, bit, RZ, 0xc0`
    reads it back. Any other write of the register or of the predicate ends what was known."""

    def __init__(self):
        self.copies = {}  # predicate: the value it was restored as
        self.held = {}    # register: {bit, or None for the whole register: value}
        self.zero = {}    # register: the bits known clear

    def value(self, predicate, writers):
        return self.copies.get(predicate) or (
            (predicate, tuple(w[0] for w in writers.get(predicate, ()))), False)

    def guard(self, guard, writers):
        """(identity, polarity) of a guard, (None, None) of none."""
        if not guard:
            return None, None
        identity, inverted = self.value(guard.lstrip("!"), writers)
        return identity, guard.startswith("!") != inverted

    def step(self, guard, opcode, tokens, written, writers):
        """Take one instruction in: `written` its reads_and_writes writes, `writers` the
        reaching writers before it."""
        base = base_opcode(opcode)
        restored = held = None
        target = tokens[0] if tokens else ""
        if base == "P2R" and not guard and tokens[1:3] == ["PR", "RZ"] and len(tokens) == 4:
            saved = single_predicate(tokens[3])
            if saved:
                held = {None: self.value(saved, writers)}
                self.zero.pop(target, None)
        elif (opcode in ("ISETP.NE.AND", "ISETP.NE.U32.AND") and not guard
              and tokens[1:2] == ["PT"] and tokens[3:] == ["RZ", "PT"]):
            restored = self.held.get(tokens[2], {}).get(None)
        elif opcode == "LOP3.LUT" and len(tokens) == 7 and not guard and tokens[1:] == [
                "RZ", tokens[2], tokens[3], "RZ", "0xc0", "!PT"] and PREDICATE.match(target):
            restored = self.held.get(tokens[2], {}).get(immediate(tokens[3]))
        elif (opcode == "LOP3.LUT" and len(tokens) == 6 and tokens[1] == target
              and tokens[3:] == ["RZ", "0xc0" if not guard else "0xfc", "!PT"]
              and immediate(tokens[2]) is not None):
            bits = immediate(tokens[2])
            known = self.held.get(target, {})
            clear = self.zero.get(target, 0)
            if not guard:  # R &= bits
                held = {b: v for b, v in known.items() if b is not None and b & bits}
                self.zero[target] = clear | (~bits & 0xffffffff)
            else:  # R |= bits where the guard holds
                held = {b: v for b, v in known.items() if b is not None and not b & bits}
                if bits & (bits - 1) == 0 and clear & bits:
                    held[bits] = self.guard(guard, writers)
                self.zero[target] = clear & ~bits
        for name in written:
            if name in self.copies:
                del self.copies[name]
            if held is None:
                self.held.pop(name, None)
                self.zero.pop(name, None)
        if held is not None:
            self.held[target] = held
        if restored and target in written:
            self.copies[target] = restored


def dependence_graph(body, order):
    """(classes, preds) of the instructions `order` (body indices, in iteration order): the
    latency class of each and the positions in `order` it depends on, ascending. Instruction i
    depends on the earlier instructions whose value of a register or a predicate it reads
    (reads_and_writes: every register of a fragment or a pair, the guard, the carries) may
    reach it: the latest writer, and where that one is guarded (it writes only the lanes its
    guard holds on, the others keep the value) the writers before it back to the latest
    unguarded one. The guarded writer itself does not wait for the value it replaces. A reader
    under the same guard as a writer (the same value of the predicate: not written in between,
    or saved and restored, PredicateValues) reads that writer's value on all its lanes and
    nothing older; under the opposite guard it reads none of it. Where the two guards cannot be
    told to be one value the older writers stay, so the chain is never shorter than the true
    one. A dependence on a value no earlier instruction wrote is one from the previous
    iteration or the prologue and carries no edge."""
    classes, preds = [], []
    writers = {}  # name: [(position, the guard's identity or None, its polarity)]
    values = PredicateValues()

    for i in order:
        _, guard, opcode, operands = body[i]
        reads, written = reads_and_writes(guard, opcode, operands)
        if guard in ("UPT", "PT"):
            guard = ""
        identity, polarity = values.guard(guard, writers)
        k = len(classes)
        deps = set()
        for r in reads:
            for position, other, other_polarity in reversed(writers.get(r, ())):
                same_value = identity is not None and other == identity
                if same_value and other_polarity != polarity:
                    continue
                deps.add(position)
                if other is None or same_value:
                    break
        classes.append(cp_class(opcode))
        preds.append(sorted(deps))
        values.step(guard, opcode, operand_tokens(operands), written, writers)
        for r in written:
            entry = (k, identity, polarity)
            writers[r] = writers.get(r, []) + [entry] if guard else [entry]
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
