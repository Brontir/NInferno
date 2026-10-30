#!/usr/bin/env python3
"""The routed banks' share of the DRAM rate in an MoE's decode (kMoeRoutedEta, src/product/choice/
choice_constants.h, DECISIONS 65) from the step rows of tests/fixtures/choice/speed_measured.json
(mbatch.sh: the server's windows whose mean batch is C rows, eight different tasks).

A step of C rows takes (choice.cpp speed_of)

  t = W_rest / (eta_rows(C) BW) + R (1 - (1 - k/E)^C) / (eta_routed(C) BW) + C row_bytes / (eta BW)
      + N_gemv t_gemv + N_small t_small

with every term but eta_routed from the dense model's facts (DECISIONS 60: kRowsEta's K-quant rows,
kKernelCost, kDecodeLaunches). eta_routed(C) is the measured step less those, per generation, the
mean over cards and quantizations at 2, 4 and 8 rows; then every step row's error under them.

    python3 fit_moe_decode.py [speed_measured.json]
"""
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ETA = 0.946
GEN = {(7, 0): "volta", (7, 5): "turing", (8, 6): "ga10x", (8, 9): "ada", (12, 0): "gb20x"}
# choice_constants.h: kRowsEta's K-quant rows (an MoE's weights are WeightClass::Other: the K-quants'),
# kKernelCost, kDecodeLaunches' MoE rows (weight-reading, small).
ROWS_KQ = {"ada": [(2, 0.904), (4, 0.877), (8, 0.794)], "gb20x": [(2, 0.922), (4, 0.795), (8, 0.732)]}
COST = {"ada": (2.34e-6, 1.93e-6), "gb20x": (1.87e-6, 2.29e-6)}
LAUNCHES = {"qwen3_5_moe_text": (311, 267), "moe_decoder": (97, 199), "gemma4_text": (181, 219)}
# Families whose rows check the table but do not make it: Gemma 4 26B-A4B's routed eta rises with
# the rows (0.81 at one row to 0.94 at eight, the others' falls), not yet explained (DECISIONS 65).
CHECK_ONLY = {"gemma4_text"}
# kKvReadFactor (Blackwell takes Ada's: the transfer generation).
KV_READ = {"bf16": 1.06, "fp8": 1.055, "k8v4": 1.025, "nvfp4": 1.235}


def interp(points, rows, one=ETA):
    pts = [(1, one)] + points
    for (r0, e0), (r1, e1) in zip(pts, pts[1:]):
        if rows <= r1:
            return e0 + math.log2(rows / r0) / math.log2(r1 / r0) * (e1 - e0)
    return pts[-1][1]


def terms(m, q, r, bw, gen):
    """The step's time but the routed reads, and the routed bytes C rows read, rows per expert."""
    c = r["concurrency"]
    table = q.get("input_table_bytes") or m["input_table_params"] * q["bits_per_weight"] / 8
    routed = q["routed_expert_bytes"]
    rest = q["weight_bytes"] - table - routed
    pick = m["experts_per_token"] / m["experts"]
    apart = 1 - (1 - pick) ** c
    # A row's own reads: its KV over the context, its recurrent state, its sliding layers' ring.
    window = min(r["context"], m.get("window", 0))
    ring = m.get("ring_bytes_per_token", {}).get(r.get("ring", "bf16"), 0)
    row = (m["kv_bytes_per_token"][r["kv"]] * r["context"] * KV_READ.get(r["kv"], 1.0)
           + 2 * m["state_bytes"][r["state"]] + window * ring)
    ng, ns = LAUNCHES[m["family"]]
    tg, ts = COST[gen]
    other = rest / (interp(ROWS_KQ[gen], c) * bw) + c * row / (ETA * bw) + ng * tg + ns * ts
    return other, routed * apart, max(1.0, c * pick / apart)


def main():
    fx = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "tests/fixtures/choice/speed_measured.json").read_text())
    rows = [r for r in fx["rows"] if r.get("step")]
    eta = {}
    for r in rows:
        card = fx["cards"][r["card"]]
        gen, bw = GEN[(card["major"], card["minor"])], card["dram_bytes_per_s"]
        m = fx["models"][r["model"]]
        other, read, _ = terms(m, m["quants"][r["quant"]], r, bw, gen)
        e = read / (bw * (1 / r["decode"] - other))
        print(f"{r['card']:12} {r['quant']:11} x{r['concurrency']}: routed {read / 1e9:.2f} GB, eta {e:.3f}")
        if r["concurrency"] in (2, 4, 8) and m["family"] not in CHECK_ONLY:
            eta.setdefault((gen, r["concurrency"]), []).append((r["card"], e))
    table = {}
    for (gen, c), v in sorted(eta.items()):
        table.setdefault(gen, []).append((c, statistics.mean(x[1] for x in v)))
        print(f"  {gen} x{c}: {statistics.mean(x[1] for x in v):.3f} ({min(x[1] for x in v):.3f}..{max(x[1] for x in v):.3f}, "
              f"{len(v)} points, {len({x[0] for x in v})} card(s))")
    err = []
    for r in rows:
        card = fx["cards"][r["card"]]
        gen, bw = GEN[(card["major"], card["minor"])], card["dram_bytes_per_s"]
        m = fx["models"][r["model"]]
        other, read, _ = terms(m, m["quants"][r["quant"]], r, bw, gen)
        e = interp(table[gen], r["concurrency"]) if r["concurrency"] > 1 else ETA
        p = 1 / (other + read / (e * bw))
        err.append(math.log(p / r["decode"]))
        print(f"{r['card']:12} {r['quant']:11} x{r['concurrency']}: predicted {p:.1f}, measured {r['decode']}")
    print(f"rms {math.sqrt(sum(x * x for x in err) / len(err)):.4f}, worst {max(map(abs, err)):.3f} ({len(err)} points)")


if __name__ == "__main__":
    main()
