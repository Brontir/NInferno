#!/usr/bin/env python3
"""The counted MMQ column-tile law (#613 PR-5, step 5b), from the committed kernel-facts snapshots.

  mmq_law.py [--arch 75 86 89 120a] [--out-dir tests/fixtures/mmq_law]

Per architecture and ggml type, the mmq section of src/ops/generic/data/kernel_facts/sm_<arch>.txt
holds one iteration of mul_mat_q's counted loop (a 128-row x J-column tile over 256 K values) for
each column tile J. A tile step is T_step(J) = F + V J: F the part that does not depend on J (the
decode of the 128 weight rows), V the per-column part (the activation tile loads and the dot
products). F and V are the least-squares line through the instances J = 16, 32, 64, 128 (the
counts, not timings), once over a warp's instructions (issue) and once over its stall sum (the
static schedule: the clocks one warp alone takes). The fixed share is f = F / (F + 128 V), and the
per-width ratio a forced-width run measures is T(J) / T(128) = f + (1 - f) J / 128.

Written to <out-dir>/predicted_sm_<arch>.csv: type, name, F and V (issue), F and V (stall),
f_issue, f_stall and the ratio at J = 16, 32, 64 under each. No GPU; nothing measured is read.
"""

import argparse
import csv
from pathlib import Path

NAMES = {2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 10: "q2_k", 11: "q3_k",
         12: "q4_k", 13: "q5_k", 14: "q6_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
         19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 29: "iq1_m"}
WIDTHS = (16, 32, 64, 128)


def line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    v = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - v * mx, v


def rows_of(snapshot: Path):
    cols, rows = None, {}
    for text in snapshot.read_text().splitlines():
        c = text.split()
        if c and c[0] == "family_columns" and "J" in c:
            cols = c[1:]
            continue
        if not c or c[0] != "mmq" or cols is None:
            continue
        r = dict(zip(cols, c))
        if int(r["fallback"]) or int(r["loops"]) == 0 or int(r["J"]) not in WIDTHS:
            continue
        rows.setdefault(int(r["type"]), {})[int(r["J"])] = (int(r["total"]), int(r["stall_sum"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["75", "86", "89", "120a"])
    ap.add_argument("--out-dir", default="tests/fixtures/mmq_law")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    for arch in a.arch:
        rows = rows_of(root / f"src/ops/generic/data/kernel_facts/sm_{arch}.txt")
        out = root / a.out_dir / f"predicted_sm_{arch}.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["type", "name", "F_issue", "V_issue", "F_stall", "V_stall", "f_issue",
                        "f_stall", "r16_issue", "r32_issue", "r64_issue", "r16_stall", "r32_stall",
                        "r64_stall"])
            for t, by_j in sorted(rows.items()):
                if len(by_j) < 3:
                    continue
                js = sorted(by_j)
                fi, vi = line(js, [by_j[j][0] for j in js])
                fs, vs = line(js, [by_j[j][1] for j in js])
                share_i, share_s = fi / (fi + 128 * vi), fs / (fs + 128 * vs)
                ratio = lambda s, j: s + (1 - s) * j / 128
                w.writerow([t, NAMES.get(t, str(t)), f"{fi:.1f}", f"{vi:.2f}", f"{fs:.1f}", f"{vs:.2f}",
                            f"{share_i:.3f}", f"{share_s:.3f}"] +
                           [f"{ratio(share_i, j):.3f}" for j in (16, 32, 64)] +
                           [f"{ratio(share_s, j):.3f}" for j in (16, 32, 64)])
        print(out)


if __name__ == "__main__":
    main()
