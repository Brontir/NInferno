#!/usr/bin/env python3
"""The prefill model's constants (src/product/choice/choice_constants.h: kPrefillEta, kPrefillAttentionEta)
from data/prefill/rows.json: per card, model and prompt length n, the measured prefill (perf_matrix.py
--no-prefix-reuse server, --chars 8000 / 32000 / 95940: ~2K / 7K / 21K tokens).

Per generation, t(n) = tail + n 2P / (eta_lin[kind] peak_s8) + F_att(n) / (eta_att peak_bf16):
P the active parameters, F_att(n) the attention flops of a prompt of n tokens (full layers over every
earlier token, sliding ones over the window). An MoE's linears run at its chunk's share of the saturated
rate (kMoeChunkShare, by the tokens each expert takes from the chunk the server ran): eta_lin is the
saturated one. tail is a fact, not fitted (DECISIONS 62): the chat template's generation prompt splits
the prefill, each piece after the last message a whole-model call that reads a decode step's weights,
tail_calls x the active share of weight_bytes / DRAM bandwidth (the product prices it from its decode
rate). Linear in 1/eta_lin and 1/eta_att; relative least squares over t(n) - tail.

    python fit_prefill.py data/prefill/rows.json
"""
import json
import sys

import numpy as np


# kMoeChunkShare (choice_constants.h): tokens per expert -> share of the saturated rate.
CHUNK_SHARE = [(8, 0.409), (16, 0.635), (32, 0.813), (64, 0.952), (128, 1.0)]


def chunk_share(model, chunk):
    if not model["moe"]:
        return 1.0
    x = chunk * model["experts_per_token"] / model["experts"]
    if x <= CHUNK_SHARE[0][0]:
        return CHUNK_SHARE[0][1]
    for (x0, s0), (x1, s1) in zip(CHUNK_SHARE, CHUNK_SHARE[1:]):
        if x <= x1:
            return s0 + np.log2(x / x0) / np.log2(x1 / x0) * (s1 - s0)
    return 1.0


def att_flops(n, full, sliding, window):
    s = n * n / 2 if window <= 0 or n <= window else n * window - window * window / 2
    return 2 * (full * n * n / 2 + sliding * s)


def main():
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    cards, models, rows = data["cards"], data["models"], data["rows"]
    for gen in sorted({c["generation"] for c in cards.values()}):
        rs = [r for r in rows if cards[r["card"]]["generation"] == gen]
        if not rs:
            continue
        on = sorted({r["card"] for r in rs})
        kinds = sorted({models[r["model"]]["moe"] for r in rs})
        A, y, tails = [], [], []
        for r in rs:
            m, c = models[r["model"]], cards[r["card"]]
            clock = c["sms"] * c["boost_mhz"] * 1e6
            x = [0.0] * (len(kinds) + 1)
            x[kinds.index(m["moe"])] = r["n"] * 2 * m["active_params"] / (clock * c["s8"] * chunk_share(m, r["chunk"]))
            x[len(kinds)] = att_flops(r["n"], m["full_width"], m["sliding_width"], m["window"]) / (clock * c["bf16"])
            tails.append(m["tail_calls"] * m["weight_bytes"] * m["active_params"] / m["params"] / (c["dram_gbs"] * 1e9))
            A.append(x)
            y.append(r["n"] / r["prefill"])
        A, y, tails = np.array(A), np.array(y), np.array(tails)
        coef, *_ = np.linalg.lstsq(A / y[:, None], 1 - tails / y, rcond=None)
        etas = "  ".join(f"eta_lin[{'moe' if k else 'dense'}] {1 / coef[i]:.3f}" for i, k in enumerate(kinds))
        print(f"{gen} ({', '.join(on)}): {etas}  eta_att {1 / coef[len(kinds)]:.3f}")
        for r, t, tail in zip(rs, A @ coef + tails, tails):
            print(f"   {r['card']:7s} {r['model']:36s} n {r['n']:6d}  measured {r['prefill']:7.0f}"
                  f"  model {100 * (r['n'] / t / r['prefill'] - 1):+5.1f} %  (tail {1e3 * tail:4.1f} ms)")


if __name__ == "__main__":
    main()
