#!/usr/bin/env python3
"""The dense decode constants of src/product/choice/choice_constants.h from the campaign's points
(tests/fixtures/choice/decode_measured.json; sm_run.sh measured them with sm_bench.py):

  t0 (kLaunchCost)    per generation, the mean over the models of (step - bytes / (eta_dram BW)) / L
                      at one stream, L = kDecodeLaunchesPerLayer x layers;
  eta(C) (kRowsEta)   per generation and row count, the mean over cards and models of the weights'
                      share of the DRAM rate once the rows' own bytes and the launches are taken off.

    python3 fit_decode.py [fixture]
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ETA = 0.946
PER_LAYER = 10
GEN = {(7, 0): "volta", (7, 5): "turing", (8, 6): "ga10x", (8, 9): "ada", (12, 0): "gb20x"}
# Left out of t0: the RTX 4070 Ti's Q8_0 T=1 GEMV (86 % of the DRAM rate, a kernel deficit).
SKIP_T0 = {("RTX 4070 Ti", "Q8_0")}


def main():
    fx = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "tests/fixtures/choice/decode_measured.json").read_text())
    pts = {}
    for r in fx["rows"]:
        pts.setdefault((r["card"], r["model"], r["quant"]), {})[r["concurrency"]] = r
    by_gen = {}
    for (card, model, quant), rows in pts.items():
        c, m = fx["cards"][card], fx["models"][model]
        q = m["quants"][quant]
        w = q["weight_bytes"] - q["input_table_bytes"] - q["speculation_bytes"]
        def row_bytes(r):
            kv = m["kv_bytes_per_token"][r["kv"]] * r["context"]
            st = 2 * m["state_bytes"].get(r["state"], 0) if m.get("recurrent", True) else 0
            ring = m.get("ring_bytes_per_token", {}).get(r["ring"], 0) * min(r["context"], m.get("window", 0))
            return kv + st + ring
        bw, launches = c["dram_bytes_per_s"], PER_LAYER * m["layers"]
        by_gen.setdefault(GEN[(c["major"], c["minor"])], []).append((card, quant, w, bw, launches, rows, row_bytes))
    for gen, items in by_gen.items():
        t0s = [(1 / rows[1]["decode"] - (w + row_bytes(rows[1])) / (ETA * bw)) / launches
               for card, quant, w, bw, launches, rows, row_bytes in items if (card, quant) not in SKIP_T0]
        t0 = statistics.mean(t0s)
        print(f"{gen}: t0 {t0 * 1e6:.2f} us ({len({i[0] for i in items})} cards; {min(t0s) * 1e6:.2f}..{max(t0s) * 1e6:.2f})")
        for n in (2, 4, 8):
            etas = [w / (bw * (1 / rows[n]["decode"] - n * row_bytes(rows[n]) / (ETA * bw) - launches * t0))
                    for card, quant, w, bw, launches, rows, row_bytes in items if n in rows]
            print(f"  eta({n}) {statistics.mean(etas):.3f}  ({min(etas):.3f}..{max(etas):.3f})")


if __name__ == "__main__":
    main()
