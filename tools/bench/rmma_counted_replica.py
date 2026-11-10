#!/usr/bin/env python3
"""The register-MMA price on the lab's route sweeps, today's and with the counted terms of
#613 PR-7a (ops/generic/rmma_counted_terms.h), predicted against measured. No GPU.

    rmma_counted_replica.py --repo DIR --sweep "GeForce RTX 4090=DIR" [--sweep ...]
                            [--law today ones ones_spill issue_max issue_nofill] [--tmin 17]
                            [--dump out.csv]

A sweep directory holds `ninferno_route_tune --routes-out` CSVs (format, compute, epilogue, n,
k, T, route, median_us, model_us, registers, local_bytes, ...); the board is named as
src/core/data/gpu_facts.json spells it. For every register-MMA row the script prices the route
as generic_cost.cpp's `cost` does (the chip from gpu_facts.json and the architecture's residency
limits, the kernel instance from the committed kernel-facts snapshot of the board's release
engine, matched by the registers and spill the sweep recorded), then with the counted terms:

  today        the shipped price (the plain form outside sm_86 / sm_120, the heavy form there)
  ones         the tensor term from the loop's MMAs, the min-term ones included
  ones_spill   ones, and the spill from the loop's LDL / STL bytes (what the engine prices under
               NINFERNO_RMMA_COUNTED_TERMS=1)
  issue_max    ones_spill with the issue bound beside the bytes-plus-tensor sum
  issue_nofill max(bytes, tensor, issue) + reduce + chain, no fitted fill at all

and reports, per board and T band, predicted / measured (geomean, p5..p95) and the ranking
regret (the measured time of the law's pick over the measured best, summed per band), plus the
picks the counted law changes with the measured gain or loss. The plain-form boards reproduce
the CSV's own model_us bit for bit; the heavy-form boards' CSVs predate the last refit of the
constants, so their model_us is not a check. This is a study tool: the engine's price is
generic_cost.cpp, and this file follows it by hand.
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import re

CC = {"GeForce RTX 2080 Ti": 75, "GeForce RTX 3060": 86, "GeForce RTX 3090": 86,
      "GeForce RTX 4060 Ti": 89, "GeForce RTX 4070 Ti": 89, "GeForce RTX 4090": 89,
      "GeForce RTX 5070 Ti": 120, "GeForce RTX 5090": 120}
ARCH = {75: "75", 86: "86", 89: "89", 120: "120a"}
DECODERS = {"bf16": ["Bf16Decoder"], "q4k": ["GgmlQ45KQuadDecoder<4>"],
            "q5k": ["GgmlQ45KQuadDecoder<5>"], "q6k": ["GgmlQ6KDecoder", "GgmlQ6KWideDecoder"]}
EPILOGUE = {"store": 0, "add": 1, "swiglu": 2}
HEAVY_FORMATS = {"q4k", "q5k", "q6k"}  # has_subblock_scales among the swept formats
BPW = {"bf16": 2.0, "q4k": 144 / 256, "q5k": 176 / 256, "q6k": 210 / 256}  # payload bytes a value
BF16_RATE = {75: 1536.0, 86: 512.0, 89: 512.0, 120: 512.0}  # core/generation.h
A16_SCALE = 1.11
# route_cost_constants.h
K = dict(rmma_step_cycles=150.0, rmma_step_k=16.0, rmma_saturating_warps=11.0,
         rmma_heavy_saturating_warps=8.0, l2_ga10x=32.0, l2_gb20x=24.0, rmma_heavy_step_k=128.0,
         spill_accesses_per_step=2.0, l1_bytes_per_sm_clock=128.0, reg_alloc_per_warp=256.0,
         cluster_reduce_cycles=30.0, warp_reduce_cycles=2.75, min_warp_fill=0.05)
ROUTE = re.compile(r"rmma\.m(\d+)\.n(\d+)\.w(\d+)x(\d+)(?:\.c(\d+))?")
MMA_FLOPS = {"16816": 4096, "1688": 2048, "1684": 1024}
LAWS = ("today", "ones", "ones_spill", "issue_max", "issue_nofill")


def chip_of(repo, name):
    cc = CC[name]
    gpus = json.load(open(os.path.join(repo, "src/core/data/gpu_facts.json")))["gpus"]
    row = next(g for g in gpus if g["name"] == name)
    return dict(
        cc=cc, arch=ARCH[cc], sms=row["sms"], clk=row["boost_mhz"] * 1e6,
        bw=row["bus_bits"] / 8.0 * row["gbps"] * 1e9, tensor=BF16_RATE[cc],
        max_thr=1024 if cc == 75 else 1536,
        max_cta=16 if cc in (75, 86) else 24 if cc == 89 else 32,
        regs_sm=65536.0, l2=K["l2_gb20x"] if cc == 120 else K["l2_ga10x"],
        heavy_ok=cc in (86, 120),
        # rmma_counted_terms.h: INT32 2 clocks a warp instruction, FP32 2 on Turing, 1 after
        int_cycles=2.0, fp_cycles=2.0 if cc == 75 else 1.0, schedulers=4.0)


def load_snapshot(repo, arch):
    path = os.path.join(repo, "src/ops/generic/data/kernel_facts", f"sm_{arch}.txt")
    cols, rows = None, {}
    for line in open(path):
        if line.startswith("columns "):
            cols = line.split()[1:]
        elif line.startswith("rmma "):
            d = dict(zip(cols, line.split()))
            key = (d["decoder"], int(d["warps_m"]), int(d["warps_k"]), int(d["m_tiles"]),
                   int(d["n_tiles"]), int(d["cluster"]), int(d["epilogue"]))
            inst = {k: (int(v) if k not in ("family", "decoder", "mma_shape") else v)
                    for k, v in d.items()}
            rows.setdefault(key, []).append(inst)
    return rows


def find_instance(snap, fmt, sched, epilogue, registers, local_bytes):
    """The snapshot instance (plain, latency or add-sums; the wide decoder for Q6_K) whose
    registers and spill equal what route_tune described; the plain one where none does (the
    sweep's build was another commit)."""
    mt, nt, wm, wk, c = sched
    cands = []
    for dec in DECODERS[fmt]:
        cands += snap.get((dec, wm, wk, mt, nt, c, epilogue), [])
    if not cands:
        return None, "noinst"
    order = lambda i: (i["latency"], i["add_sums"], i["decoder"] != DECODERS[fmt][0])
    exact = sorted((i for i in cands if i["registers"] == registers and
                    i["stack"] + i["local"] == local_bytes), key=order)
    if exact:
        return exact[0], "ok" if len(exact) == 1 else "ambiguous"
    return sorted(cands, key=order)[0], "fallback"


def price(chip, call, inst, sched, law):
    """Seconds of the register-MMA tile under `law` and its terms."""
    n, k, tokens, sets, bpw, heavy = (call[x] for x in ("n", "k", "T", "sets", "bpw", "heavy"))
    mt, nt, wm, wk, cluster = sched
    rows, cols = 16 * mt * wm, 8 * nt
    threads = 32.0 * wm * wk
    clk, sms, bw = chip["clk"], chip["sms"], chip["bw"]
    peak_sm = chip["tensor"] * clk
    l2_sm = chip["l2"] * clk
    limit = min(math.floor(chip["max_thr"] / threads), chip["max_cta"])
    if heavy and inst:
        unit = K["reg_alloc_per_warp"]
        per_cta = math.ceil(inst["registers"] * 32.0 / unit) * unit * (threads / 32.0)
        occ = max(1.0, min(limit, math.floor(chip["regs_sm"] / per_cta)))
    else:
        occ = limit
    local_bytes = (inst["stack"] + inst["local"]) if (heavy and inst) else 0.0
    tiles = math.ceil(tokens / cols)
    gx = math.ceil(n / rows)
    kc = k / cluster
    q = math.ceil(gx * tiles * cluster / sms)
    warps = min(q, max(occ, 1.0)) * threads / 32
    mem = max(q * rows * kc * bpw / (bw / sms), n * k * bpw / bw)
    mma = q * 2.0 * rows * sets * cols * kc / peak_sm
    rounds = math.ceil(q / max(occ, 1.0))
    dram = n * k * bpw / bw
    l2_w = q * kc * rows * bpw / l2_sm
    per_round = min(tiles, max(occ, 1.0), q, max(1.0, math.ceil(occ * sms / gx)))
    l2_x = rounds * per_round * kc * (tokens / tiles) * 2.0 / l2_sm
    spill = (q * threads * local_bytes * K["spill_accesses_per_step"] *
             math.ceil(kc / (K["rmma_heavy_step_k"] * wk)) / (K["l1_bytes_per_sm_clock"] * clk))
    reduce = (q * (cluster - 1) * rows * sets * cols * K["cluster_reduce_cycles"] / clk +
              rounds * (wk - 1.0) * rows * sets * cols * K["warp_reduce_cycles"] / clk)
    chain = rounds * math.ceil(kc / (K["rmma_step_k"] * wk)) * K["rmma_step_cycles"] / clk
    fill_warps = K["rmma_heavy_saturating_warps"] if heavy else K["rmma_saturating_warps"]
    fill = min(1.0, max(K["min_warp_fill"], warps / fill_warps))
    t = dict(occ=occ, q=q, warps=warps, rounds=rounds, mem=mem, dram=dram, l2_w=l2_w, l2_x=l2_x,
             spill=spill, mma=mma, reduce=reduce, chain=chain, fill=fill, iters=0.0,
             mma_counted=mma, spill_counted=spill, issue=0.0, int_pipe=0.0, fp_pipe=0.0)
    # The counted terms (rmma_counted_terms.h) per warp-iteration of the K loop.
    usable = inst is not None and inst.get("loops", 0) == 1 and inst.get("k_iter", 0) > 0 \
        and inst.get("mma", 0) > inst.get("mma_ones", 0)
    if usable:
        iters = (kc / wk) / inst["k_iter"]
        wi = q * (wm * wk) * iters
        shape = inst["mma_shape"].split(".")[1] if "." in inst["mma_shape"] else "16816"
        flops = MMA_FLOPS.get(shape, 4096)
        per_slot = 1.0 / (chip["schedulers"] * clk)
        t.update(iters=iters,
                 mma_counted=wi * inst["mma"] * flops / peak_sm,
                 spill_counted=wi * inst["local_bytes"] / (K["l1_bytes_per_sm_clock"] * clk),
                 int_pipe=wi * inst["int"] * chip["int_cycles"] * per_slot,
                 fp_pipe=wi * inst["fp"] * chip["fp_cycles"] * per_slot)
        t["issue"] = max(wi * inst["total"] * per_slot, t["int_pipe"], t["fp_pipe"])
    heavy_bytes = dram + l2_w + l2_x
    if law == "today":
        total = ((heavy_bytes + spill if heavy else mem) + mma + reduce) / fill + chain
    elif law == "ones":
        total = ((heavy_bytes + spill if heavy else mem) + t["mma_counted"] + reduce) / fill + chain
    elif law == "ones_spill":
        total = ((heavy_bytes + t["spill_counted"] if heavy else mem) + t["mma_counted"] + reduce) / fill + chain
    elif law == "issue_max":
        total = max((heavy_bytes + t["spill_counted"] if heavy else mem) + t["mma_counted"] + reduce, t["issue"]) / fill + chain
    elif law == "issue_nofill":
        total = max(heavy_bytes + t["spill_counted"] if heavy else mem, t["mma_counted"], t["issue"]) + reduce + chain
    else:
        raise ValueError(law)
    return total * A16_SCALE, t


def load_rows(directory):
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)):
        if "selected" in os.path.basename(path) or os.path.basename(path).startswith("sel"):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                m = ROUTE.match(r.get("route", ""))
                if not m or r["format"] not in DECODERS:
                    continue
                out.append(dict(fmt=r["format"], epi=r["epilogue"], n=int(r["n"]), k=int(r["k"]),
                                T=int(r["T"]), route=r["route"],
                                sched=tuple(int(v) if v else 1 for v in m.groups()),
                                median=float(r["median_us"]),
                                model=float(r["model_us"]) if r.get("model_us") else None,
                                registers=int(r.get("registers") or 0),
                                local=int(r.get("local_bytes") or 0)))
    return out


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def band(T):
    return "17-32" if T <= 32 else "33-128" if T <= 128 else "129+"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sweep", action="append", required=True, help='"<board name>=<dir>"')
    ap.add_argument("--law", nargs="+", default=list(LAWS))
    ap.add_argument("--tmin", type=int, default=17)
    ap.add_argument("--dump")
    ap.add_argument("--changes", type=int, default=5, help="changed picks to list per board")
    a = ap.parse_args()
    dump = csv.writer(open(a.dump, "w", newline="")) if a.dump else None
    term_names = ("occ", "warps", "iters", "fill", "dram", "l2_w", "l2_x", "spill", "spill_counted",
                  "mma", "mma_counted", "issue", "int_pipe", "fp_pipe", "chain")
    if dump:
        dump.writerow(["board", "fmt", "epi", "n", "k", "T", "route", "median_us", "model_us"]
                      + [f"{l}_us" for l in a.law] + list(term_names))
    for spec in a.sweep:
        name, _, directory = spec.partition("=")
        chip = chip_of(a.repo, name)
        snap = load_snapshot(a.repo, chip["arch"])
        rows = [r for r in load_rows(directory) if r["T"] >= a.tmin]
        status = collections.Counter()
        ratio = {l: collections.defaultdict(list) for l in a.law}
        today_vs_model = []
        problems = collections.defaultdict(list)
        issue_binds = 0
        for r in rows:
            inst, st = find_instance(snap, r["fmt"], r["sched"], EPILOGUE[r["epi"]],
                                     r["registers"], r["local"])
            status[st] += 1
            sets = 2.0 if r["epi"] == "swiglu" else 1.0
            call = dict(n=r["n"], k=r["k"], T=r["T"], sets=sets, bpw=BPW[r["fmt"]] * sets,
                        heavy=r["fmt"] in HEAVY_FORMATS and chip["heavy_ok"])
            preds, terms = {}, {}
            for l in a.law:
                seconds, terms = price(chip, call, inst, r["sched"], l)
                preds[l] = seconds * 1e6
                ratio[l][band(r["T"])].append(preds[l] / r["median"])
            if r["model"] and "today" in preds:
                today_vs_model.append(preds["today"] / r["model"])
            if terms["issue"] > terms["dram"] + terms["l2_w"] + terms["l2_x"] + terms["spill_counted"] + terms["mma_counted"]:
                issue_binds += 1
            problems[(r["fmt"], r["epi"], r["n"], r["k"], r["T"])].append((r["median"], preds, r["route"]))
            if dump:
                scale = lambda x: terms[x] * (1e6 if x not in ("occ", "warps", "iters", "fill") else 1)
                dump.writerow([name, r["fmt"], r["epi"], r["n"], r["k"], r["T"], r["route"],
                               r["median"], r["model"]] + [f"{preds[l]:.3f}" for l in a.law]
                              + [f"{scale(x):.5g}" for x in term_names])
        print(f"\n== {name} (sm_{chip['arch']}, heavy form {'on' if chip['heavy_ok'] else 'off'}): "
              f"{len(rows)} register-MMA rows at T >= {a.tmin}; instances {dict(status)}; "
              f"the issue bound above the bytes-plus-tensor sum on {issue_binds} rows")
        if today_vs_model:
            xs = sorted(today_vs_model)
            print(f"  today's replica / the CSV's model_us: geomean {geomean(xs):.4f}, "
                  f"within 1 %: {sum(1 for x in xs if abs(x - 1) < 0.01) / len(xs):.1%}")
        print(f"  {'law':13s}" + "".join(f"{b + ' pred/meas':>24s}" for b in ("17-32", "33-128", "129+"))
              + "   regret 17-32 33-128 129+")
        for l in a.law:
            cells, regret = [], []
            for b in ("17-32", "33-128", "129+"):
                xs = sorted(ratio[l].get(b, []))
                cells.append(f"{geomean(xs):6.3f} [{xs[len(xs) // 20]:.2f}..{xs[-max(1, len(xs) // 20)]:.2f}]" if xs else "-")
                picked = best = 0.0
                for key, cands in problems.items():
                    if band(key[4]) != b:
                        continue
                    picked += min(cands, key=lambda c: c[1][l])[0]
                    best += min(c[0] for c in cands)
                regret.append(picked / best if best else float("nan"))
            print(f"  {l:13s}" + "".join(f"{c:>24s}" for c in cells) + "   " + " ".join(f"{x:.3f}" for x in regret))
        if "today" in a.law and "ones_spill" in a.law:
            changed = []
            for key, cands in problems.items():
                old = min(cands, key=lambda c: c[1]["today"])
                new = min(cands, key=lambda c: c[1]["ones_spill"])
                if old[2] != new[2]:
                    changed.append((key, old[2], old[0], new[2], new[0]))
            gains = [z / o for _, _, o, _, z in changed]
            print(f"  picks changed by ones_spill: {len(changed)} of {len(problems)}"
                  + (f"; new / old measured geomean {geomean(gains):.3f}, worst {max(gains):.2f}, best {min(gains):.2f}" if gains else ""))
            for key, orute, om, nrute, nm in sorted(changed, key=lambda c: -abs(math.log(c[4] / c[2])))[:a.changes]:
                print(f"    {key[0]} {key[1]} {key[2]}x{key[3]} T={key[4]}: {orute} {om:.1f} us -> {nrute} {nm:.1f} us ({nm / om:.2f})")


if __name__ == "__main__":
    main()
