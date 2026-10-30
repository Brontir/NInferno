#!/usr/bin/env python3
"""The dense decode constants of src/product/choice/choice_constants.h from the campaign's points
(tests/fixtures/choice/decode_measured.json, sm_run.sh with sm_bench.py) and the nsys census of one
step (tests/fixtures/choice/decode_census.json, census.sh with ana2.py):

  kDecodeLaunches   the census's kernel counts (facts of each family's graph; the same on every card
                    at one row);
  kKernelCost       per generation, the mean over the census's one-row steps of what a weight-reading
                    kernel takes beyond its weights at eta_dram, and a small one beyond the state and
                    KV bytes the step's rows read (the census ran at a ~160-token context);
  kRowsEta          per generation, weight class and row count, the mean over cards and models of the
                    weights' share of the DRAM rate once the rows' own bytes and the launches
                    (kDecodeLaunches x kKernelCost) are taken off the measured step;
  kDecodePredictionError  per generation, the rms of ln(predicted / measured) over every point.

    python3 fit_decode.py [decode_measured.json [decode_census.json]]
"""
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ETA = 0.946
CENSUS_CONTEXT = 160
GEN = {(7, 0): "volta", (7, 5): "turing", (8, 6): "ga10x", (8, 9): "ada", (12, 0): "gb20x"}
CLASS = {"Q8_0": "q8", "IQ4_XS": "iq4", "IQ4_NL": "iq4"}  # else K-quants


def weights(m, quant):
    q = m["quants"][quant]
    return q["weight_bytes"] - q["input_table_bytes"] - q["speculation_bytes"]


def row_bytes(m, r):
    kv = m["kv_bytes_per_token"][r["kv"]] * r["context"]
    st = 2 * m["state_bytes"].get(r["state"], 0) if m.get("recurrent", True) else 0
    ring = m.get("ring_bytes_per_token", {}).get(r["ring"], 0) * min(r["context"], m.get("window", 0))
    return kv + st + ring


def main():
    fx = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "tests/fixtures/choice/decode_measured.json").read_text())
    cen = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "tests/fixtures/choice/decode_census.json").read_text())
    gen_of = {name: GEN[(c["major"], c["minor"])] for name, c in fx["cards"].items()}
    # The kernel counts by (model, quant, rows): the census's modal count on the reference generation's
    # cards (Ada; the 5-8-row routes vary by card).
    counts = {}
    for r in cen["rows"]:
        if gen_of[r["card"]] != "ada":
            continue
        counts.setdefault((r["model"], r["quant"], r["streams"]), []).append((r["gemv_kernels"], r["small_kernels"]))
    count = {k: statistics.mode(v) for k, v in counts.items()}
    print("kernels a step (gemv, small):", {f"{k[0]} {k[1]} x{k[2]}": v for k, v in sorted(count.items())})

    def kernels(model, quant, rows):
        return count[(model, quant, 1 if rows == 1 else 2 if rows <= 4 else 8)]

    # kKernelCost from the one-row census.
    cost = {}
    for r in cen["rows"]:
        if r["streams"] != 1:
            continue
        m, c = fx["models"][r["model"]], fx["cards"][r["card"]]
        bw = c["dram_bytes_per_s"]
        # The census ran the decode campaign's storages (its one-row point) at a ~160-token context.
        ctx = dict(next(x for x in fx["rows"] if x["card"] == r["card"] and x["model"] == r["model"]
                        and x["quant"] == r["quant"] and x["concurrency"] == 1), context=CENSUS_CONTEXT)
        tg = (r["gemv_us"] * 1e-6 - weights(m, r["quant"]) / (ETA * bw)) / r["gemv_kernels"]
        ts = (r["small_us"] * 1e-6 - row_bytes(m, ctx) / (ETA * bw)) / r["small_kernels"]
        cost.setdefault(gen_of[r["card"]], []).append((r["card"], tg, ts))
    t_gemv, t_small = {}, {}
    for gen, v in cost.items():
        t_gemv[gen] = statistics.mean(x[1] for x in v)
        t_small[gen] = statistics.mean(x[2] for x in v)
        print(f"{gen}: t_gemv {t_gemv[gen] * 1e6:.2f} us ({min(x[1] for x in v) * 1e6:.2f}..{max(x[1] for x in v) * 1e6:.2f}), "
              f"t_small {t_small[gen] * 1e6:.2f} us ({min(x[2] for x in v) * 1e6:.2f}..{max(x[2] for x in v) * 1e6:.2f}), "
              f"{len({x[0] for x in v})} card(s)")

    # kRowsEta per generation, class and rows.
    eta = {}
    for r in fx["rows"]:
        gen, n = gen_of[r["card"]], r["concurrency"]
        if n == 1 or gen not in t_gemv:
            continue
        m, bw = fx["models"][r["model"]], fx["cards"][r["card"]]["dram_bytes_per_s"]
        ng, ns = kernels(r["model"], r["quant"], n)
        t = 1 / r["decode"] - n * row_bytes(m, r) / (ETA * bw) - ng * t_gemv[gen] - ns * t_small[gen]
        eta.setdefault((gen, CLASS.get(r["quant"], "kq"), n), []).append(weights(m, r["quant"]) / (bw * t))
    for k in sorted(eta):
        v = eta[k]
        print(f"  eta {k}: {statistics.mean(v):.3f} ({min(v):.3f}..{max(v):.3f}, {len(v)} points)")

    # kDecodePredictionError: every point under the rows above.
    err = {}
    for r in fx["rows"]:
        gen, n = gen_of[r["card"]], r["concurrency"]
        if gen not in t_gemv:
            continue
        m, bw = fx["models"][r["model"]], fx["cards"][r["card"]]["dram_bytes_per_s"]
        cls = CLASS.get(r["quant"], "kq")
        e = ETA
        if n > 1:
            pts = [(1, ETA)] + [(c, statistics.mean(eta[(gen, cls, c)])) for c in (2, 4, 8) if (gen, cls, c) in eta]
            e = pts[-1][1]
            for (r0, e0), (r1, e1) in zip(pts, pts[1:]):
                if n <= r1:
                    e = e0 + math.log2(n / r0) / math.log2(r1 / r0) * (e1 - e0)
                    break
        ng, ns = kernels(r["model"], r["quant"], n)
        t = weights(m, r["quant"]) / (e * bw) + n * row_bytes(m, r) / (ETA * bw) + ng * t_gemv[gen] + ns * t_small[gen]
        err.setdefault(gen, []).append(math.log((1 / t) / r["decode"]))
    for gen, v in err.items():
        print(f"{gen}: prediction error rms {math.sqrt(sum(x * x for x in v) / len(v)):.4f}, "
              f"worst {max(map(abs, v)):.3f} ({len(v)} points)")


if __name__ == "__main__":
    main()
