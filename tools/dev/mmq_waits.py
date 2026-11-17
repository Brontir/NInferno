#!/usr/bin/env python3
"""Scoreboard waits of mul_mat_q's counted iteration, by what they wait for (#613 PR-5).

  mmq_waits.py SASS[.gz] > waits_sm_<arch>.csv

Reads `cuobjdump -sass` text of a build (ggml's mul_mat_q kernels, ggml_prec 30), takes each
kernel's counted K-loop iteration the way sass_counts.loop_counts does (the innermost MMA loop's
heaviest path), and walks it twice in address order (a load issued late in one iteration and
waited at the top of the next counts once). Each instruction's control code names the scoreboard
it sets on its result (LDG / LD: global, LDS / LDSM: shared, anything else: other) or on its
sources' release, and the scoreboards it waits on; a wait is classed by the kinds it retires,
global over shared over the rest. Written per kernel: type, J, the global loads, the waits on
global, on shared and on the rest, the stall sum and the instruction count. No GPU, no timing.
"""
import csv
import gzip
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernel_facts"))
import sass_counts as sc  # noqa: E402

NAME = re.compile(r"^_Z9mul_mat_qIL9ggml_type(\d+)ELi(\d+)ELb0EL9ggml_prec30E")


def iteration(body):
    succ = sc.successors(body)
    if succ is None:
        return []
    blocks, edges = sc.basic_blocks(body, succ)
    live = sc.reachable(edges)
    dom, preds = sc.dominators(edges, live)
    loops = sc.natural_loops(edges, dom, preds, live)

    def weight(b):
        s, e = blocks[b]
        return (sum(1 for i in range(s, e + 1) if sc.base_opcode(body[i][2]) in sc.MMA_OPCODES),
                e - s + 1)

    marked = {h: sum(weight(b)[0] for b in m) for h, m in loops.items()}
    cand = [h for h, n in marked.items() if n > 0]
    inner = [h for h in cand if not any(o != h and loops[o] < loops[h] for o in cand)]
    its = {h: sc.longest_iteration(edges, dom, h, loops[h], weight) for h in inner}
    its = {h: p for h, p in its.items() if p}
    if not its:
        return []
    path = max(its.values(), key=lambda p: tuple(sum(w) for w in zip(*(weight(b) for b in p))))
    return [i for b in path for i in range(blocks[b][0], blocks[b][1] + 1)]


def waits(body, it):
    pend = [set() for _ in range(8)]
    out = {"ldg": 0, "w_global": 0, "w_shared": 0, "w_other": 0, "stall": 0, "instr": len(it)}
    for walk in (0, 1):
        for i in it:
            _, _, op, _, hw = body[i]
            c = sc.Control(hw)
            kinds = set()
            for b in range(6):
                if c.wait_mask >> b & 1 and pend[b]:
                    kinds |= pend[b]
                    pend[b] = set()
            base = sc.base_opcode(op)
            if walk:
                if kinds:
                    key = "w_global" if "g" in kinds else "w_shared" if "s" in kinds else "w_other"
                    out[key] += 1
                out["stall"] += c.stall
                out["ldg"] += base in ("LDG", "LD")
            kind = "g" if base in ("LDG", "LD") else "s" if base in ("LDS", "LDSM") else "o"
            if c.write_barrier != sc.NO_BARRIER:
                pend[c.write_barrier].add(kind)
            if c.read_barrier != sc.NO_BARRIER:
                pend[c.read_barrier].add("o")
    return out


def main():
    path = sys.argv[1]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        fns = sc.parse_functions(f, lambda n: bool(NAME.match(n)))
    rows = []
    for name, body in fns.items():
        it = iteration(body)
        if not it or body[it[0]][4] is None:
            continue
        m = NAME.match(name)
        rows.append((int(m.group(1)), int(m.group(2)), waits(body, it)))
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["type", "J", "ldg", "w_global", "w_shared", "w_other", "stall", "instr"])
    for t, J, o in sorted(rows):
        w.writerow([t, J, o["ldg"], o["w_global"], o["w_shared"], o["w_other"], o["stall"], o["instr"]])


if __name__ == "__main__":
    main()
