"""The MoE expert-call law (#619 step 11): rank a grouped MoE layer's expert plans (gate/up schedule | down schedule,
the 12 x 12 plans of `ninferno_moe_bench --plans sweep`) on one board from facts alone. No GPU, no fitted constant.

A stage (gate/up or down) of a plan takes
    t = max(t_dram, waves x c, t_issue) + c
  - t_dram: the touched experts' weight bytes (E (1 - (1 - k/E)^T) experts, a whole row span each, at the format's bits
    per weight) over min(the published bus rate, active SMs x one SM's rate at the bytes its resident warps keep in
    flight). The SM rate is the warp_inflight probe's (src/core/data/warp_inflight, placed k = 1 rows); a warp keeps the
    bytes one K-loop iteration loads (ldg + ldgsts bytes, counted from the SASS).
  - c, one CTA's chain: its K-loop iterations (at least one) x (the iteration's static stall cycles over the reference
    clock + one fetch latency, the probe's one-warp ns_fetch), plus the static stalls outside the loop (index setup, the
    K split's reduction, the epilogue), plus, for an R8 kernel that quantizes its input itself (the down stage, pre 0),
    its quantizer loop's stalls and global waits.
  - waves: the grid's CTAs over SMs x the CTAs an SM holds (registers, static shared memory, threads, the CTA limit).
  - t_issue: every warp instruction of the busiest SM's CTAs over its four schedulers.
The trailing c is the fill and drain no bus hides. The counted facts are the kernel-facts snapshot's moegemm and moer8
sections (src/ops/generic/data/kernel_facts/sm_<arch>.txt). The grid geometry follows moe.cpp (rows a CTA,
r8_lanes, jobs as the touched experts at T <= 8).

    moe_expert_law.py score FACTS GRID_DIR BOARD           the law's pick against a measured moe11 grid
    moe_expert_law.py predict FACTS BOARD PLANS_CSV E,K,H,F OUT.csv
                                                           every plan's price and the pick, for a sealed grid
"""
import collections, csv, math, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE_DIR = os.path.join(ROOT, 'src', 'core', 'data', 'warp_inflight')
NT_OF = [1]
BOARDS = {  # sms, DRAM GB/s (published), reference clock MHz, fetch latency ns (warp_inflight chain1 w1), max CTAs/SM, threads/SM
    "rtx4090": (128, 1008, 2520, 253, 24, 1536),
    "rtx4060ti": (34, 288, 2535, None, 24, 1536),
    "rtx5090": (170, 1792, 2407, None, 32, 1536),
    "rtx3090": (82, 936, 1695, None, 16, 1536),
    "rtx5070ti": (70, 896, 2452, None, 32, 1536),
    "rtx4070ti": (60, 504, 2610, None, 24, 1536),
    "rtx3060": (28, 360, 1777, None, 16, 1536),
    "rtx2080ti": (68, 616, 1545, None, 16, 1024),
}
BPW = {"q3_k": 3.4375, "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625}
GEMM_DEC = {"q3_k": "GgmlQ3KDecoder", "q4_k": "GgmlQ45KDecoder<4>", "q5_k": "GgmlQ45KDecoder<5>", "q6_k": "GgmlQ6KDecoder"}
R8_CODEC = {"q3_k": "W4a8Q3K", "q4_k": "W4a8Q45K<4>", "q5_k": "W4a8Q45K<5>", "q6_k": "W4a8Q6K"}
R8_PER_LOAD = {"q3_k": 4, "q4_k": 2, "q5_k": 2, "q6_k": 4}
# The candidates are the plans the call may run (moe.cpp r8_admits): an R8 stage only up to the R8 token limit,
# moe_regime_rules.h's records (the generation's: eight on Ada and consumer Blackwell, four elsewhere; a non-frag
# Q6_K bank four everywhere). The law prices the admitted plans; admission itself is the runtime's.
GENERATION = {"rtx4090": "ada", "rtx4060ti": "ada", "rtx4070ti": "ada", "rtx5090": "gb20x", "rtx5070ti": "gb20x",
              "rtx3090": "ga10x", "rtx3060": "ga10x", "rtx2080ti": "turing"}


def r8_token_limit(board_name, fmt):
    if fmt == "q6_k":
        return 4
    return 8 if GENERATION[board_name] in ("ada", "gb20x") else 4


def admitted(board_name, fmt, plan, tokens):
    return all(not s.startswith("r8") or tokens <= r8_token_limit(board_name, fmt) for s in plan.split("|"))


def r8_lanes(fmt, k):
    """moe.cpp r8_lanes: the lanes a row takes, from the loads a lane makes over the row (K / 32 / per load)."""
    loads = k // 32 // R8_PER_LOAD[fmt]
    return 8 if loads <= 24 else 16 if loads <= 48 else 32


def load_facts(path):
    fams, cols, rows = {}, {}, collections.defaultdict(list)
    fam = None
    for line in open(path):
        if line.startswith("family_columns"):
            cols[fam] = line.split()[1:]
        elif line.startswith("family "):
            fam = line.split()[1]
        elif fam in ("moegemm", "moer8") and line.startswith(fam + " "):
            rows[fam].append(line.split())
    out = {}
    for fam in ("moegemm", "moer8"):
        c = cols[fam]
        nkey = c.index("registers")
        for r in rows[fam]:
            # a decoder name may hold spaces? (no: template args have none)
            d = dict(zip(c, r))
            out[(fam,) + tuple(r[1:nkey])] = d
    return out


PROBE = {"rtx4090": "warp_inflight_89.txt", "rtx4060ti": "warp_inflight_89_rtx4060ti.txt",
         "rtx5090": "warp_inflight_120a.txt", "rtx3090": "warp_inflight_86.txt",
         "rtx5070ti": "warp_inflight_120a_rtx5070ti.txt", "rtx3060": "warp_inflight_86_rtx3060.txt",
         "rtx2080ti": "warp_inflight_75.txt", "rtx4070ti": "warp_inflight_89_rtx4070ti.txt"}
_curves = {}


def sm_rate(board_name, bytes_inflight):
    """One SM's DRAM rate (B/s) at `bytes_inflight` bytes in flight: the warp_inflight probe's placed k=1 rows
    (bytes = loads in flight a warp x warps x 512), the best rate at or below those bytes, linear between rows."""
    if board_name not in _curves:
        pts = {}
        for line in open(os.path.join(PROBE_DIR, PROBE[board_name])):
            f = line.split()
            if f[:3] == ["row", "placed", "1"]:
                b = int(f[4]) * int(f[5]) * 512; g = float(f[6]) * 1e9
                pts[b] = max(pts.get(b, 0), g)
        xs = sorted(pts); best = 0; curve = []
        for x in xs: best = max(best, pts[x]); curve.append((x, best))
        _curves[board_name] = curve
    c = _curves[board_name]
    if bytes_inflight <= c[0][0]: return c[0][1] * bytes_inflight / c[0][0]
    for (x0, y0), (x1, y1) in zip(c, c[1:]):
        if bytes_inflight <= x1: return y0 + (y1 - y0) * (bytes_inflight - x0) / (x1 - x0)
    return c[-1][1]


def probe_latency(board_name):
    """The probe's fetch latency (s): placed k=1 chain1 at one warp (ns_fetch)."""
    for line in open(os.path.join(PROBE_DIR, PROBE[board_name])):
        f = line.split()
        if f[:4] == ["row", "placed", "1", "chain1"] and f[5] == "1": return float(f[9]) * 1e-9
    raise SystemExit("no latency row")


def occupancy(regs, smem, threads, max_ctas, max_threads):
    by_regs = (64 * 1024) // max(1, regs * threads) if regs else max_ctas
    by_smem = (100 * 1024) // smem if smem else max_ctas
    return max(0, min(max_ctas, by_regs, by_smem, max_threads // threads))


def stage_time(facts, board, fmt, sched, stage, T, jobs, H, F):
    sms, gbs, mhz, lat_ns, max_ctas, max_thr = board
    lat = probe_latency(BOARD_NAME)
    clock = mhz * 1e6
    rows_out = 2 * F if stage == 1 else H  # gate/up: the gate and up rows of the touched experts
    k = H if stage == 1 else F
    if sched.startswith("rmma"):
        m = int(sched.split(".m")[1].split(".")[0]); n = int(sched.split(".n")[1].split(".")[0])
        wm, wk = (int(x) for x in sched.split(".w")[1].split("x"))
        key = ("moegemm", GEMM_DEC[fmt], str(wm), str(wk), str(m), str(n), "1", str(stage))
        f = facts.get(key)
        if f is None: return None, key
        warps = wm * wk; rows_cta = wm * m * 16; sub = 1
        ctas = math.ceil((F if stage == 1 else H) / rows_cta) * jobs * sub
        iters = (k / wk) / max(1, int(f["k_iter"])); NT_OF[0] = 0
    else:
        w = int(sched.split(".w")[1].split("r")[0]); r = int(sched.split("r")[-1])
        nt = min(8, max(1, math.ceil(T * 8 / 256 * 256 / max(1, jobs))))  # tokens a job holds (moe11: T x 8 / jobs)
        cand = [kk for kk in facts if kk[0] == "moer8" and kk[1] == R8_CODEC[fmt] and kk[2] == str(w) and kk[3] == str(r)
                and kk[5] == str(stage)]
        # pre 1: the kernel reads codes the router's CTAs wrote (gate/up of an unpadded x); pre 0: it quantizes its
        # BF16 input itself (the down stage).
        want_pre = "1" if stage == 1 else "0"
        cand = [kk for kk in cand if kk[6] == want_pre] or cand
        if not cand: return None, ("moer8", R8_CODEC[fmt], w, r, stage)
        nts = sorted({int(kk[4]) for kk in cand})
        ntk = min([x for x in nts if x >= nt] or [nts[-1]])
        f = facts[[kk for kk in cand if int(kk[4]) == ntk][0]]; NT_OF[0] = nt
        warps = w; rows_cta = w * r * (32 // r8_lanes(fmt, k))
        ctas = math.ceil((F if stage == 1 else H) / rows_cta) * jobs
        iters = k / max(1, int(f["k_iter"]))
    threads = warps * 32
    res = occupancy(int(f["registers"]), int(f["shared"]), threads, max_ctas, max_thr)
    if res == 0: return None, "occupancy"
    conc = min(res, math.ceil(ctas / sms))
    active = min(sms, ctas)
    bw_warp = int(f["ldg_bytes"]) + int(f["ldgsts_bytes"])
    per_sm = sm_rate(BOARD_NAME, conc * warps * bw_warp)
    rate = min(gbs * 1e9, active * per_sm)
    touched_rows = jobs * rows_out  # jobs ~ the touched experts at T <= 8
    bytes_ = touched_rows * k * BPW[fmt] / 8
    t_dram = bytes_ / rate
    waves = math.ceil(ctas / (sms * res))
    # One CTA's chain: each iteration's static stalls plus one DRAM latency for its loads (the first wave's fill
    # and the last wave's drain are not hidden by the bus).
    c = max(1.0, iters) * (int(f["stall_sum"]) / clock + lat)
    # The code outside the K loop (index setup, the K split's reduction, the epilogue): its static stalls.
    c += int(f["outside_stall"]) / clock
    if f.get("pro_loops", "0") not in ("0", None):
        # The R8 kernel's activation quantizer (pro_ columns): one 32-value block a thread and pass over the CTA's
        # tokens x K, before the dp4a loop; its global-load waits each at the fetch latency.
        pro_iters = math.ceil(NT_OF[0] * k / 32 / threads)
        c += pro_iters * int(f["pro_stall"]) / clock + int(f["pro_glb_waits"]) * lat
    t_chain = waves * c
    # Issue: every warp instruction of the busiest SM's CTAs over its four schedulers (one a clock each).
    t_issue = math.ceil(ctas / sms) * warps * iters * int(f["total"]) / (4 * clock)
    return max(t_dram, t_chain, t_issue) + c, (t_dram, t_chain, waves, conc, res, t_issue)


def main():
    global BOARD_NAME
    facts = load_facts(sys.argv[1]); D = sys.argv[2]; board = BOARDS[sys.argv[3]]; BOARD_NAME = sys.argv[3]
    for fmt in BPW:
        meas = collections.defaultdict(list); auto = collections.defaultdict(list); geo = None
        for p in (1, 2, 3):
            for r in csv.DictReader(open(f"{D}/sw_{fmt}_p{p}.csv")):
                try: meas[(int(r["T"]), r["plan"])].append(float(r["median_us"])); geo = (int(r["E"]), int(r["K"]), int(r["H"]), int(r["F"]))
                except (KeyError, ValueError): pass
            for r in csv.DictReader(open(f"{D}/au_{fmt}_p{p}.csv")):
                try: auto[int(r["T"])].append(r["plan"])
                except (KeyError, ValueError): pass
        E, K, H, F = geo
        regrets, aregrets, miss = [], [], 0
        for T in sorted({t for t, _ in meas}):
            plans = {pl: st.median(v) for (t, pl), v in meas.items() if t == T}
            best = min(plans.values())
            jobs = round(E * (1 - (1 - K / E) ** T))
            pred = {}
            for pl in plans:
                if not admitted(BOARD_NAME, fmt, pl, T): continue
                gu, dn = pl.split("|")
                a, _ = stage_time(facts, board, fmt, gu, 1, T, jobs, H, F)
                b, _ = stage_time(facts, board, fmt, dn, 2, T, jobs, H, F)
                if a is None or b is None: miss += 1; continue
                pred[pl] = a + b
            if not pred: continue
            pick = min(pred, key=pred.get)
            ap = collections.Counter(auto[T]).most_common(1)[0][0] if auto[T] else None
            regrets.append(plans[pick] / best - 1)
            if ap in plans: aregrets.append(plans[ap] / best - 1)
            print(f"  {fmt} T={T} law {pick} {plans[pick]:.1f} ({100*(plans[pick]/best-1):.1f} %)  auto {ap} "
                  f"{plans.get(ap, float('nan')):.1f}  best {best:.1f}  pred {pred[pick]*1e6:.1f} us")
        print(f"{fmt}: law regret max {100*max(regrets):.1f} % mean {100*st.mean(regrets):.1f} %;"
              f" auto max {100*max(aregrets or [0]):.1f} % mean {100*st.mean(aregrets or [0]):.1f} %; unpriced {miss}")



def predict(facts_path, plans_from, geo, out_path):
    """The frozen form: every plan's predicted us and the pick, per format and T, for a board whose grid is
    sealed. The plan list is read from a training grid's sweep (the same 144 plans every board runs)."""
    global BOARD_NAME
    facts = load_facts(facts_path); board = BOARDS[BOARD_NAME]
    plans = sorted({r["plan"] for r in csv.DictReader(open(plans_from)) if r.get("plan")})
    E, K, H, F = geo
    with open(out_path, "w", newline="\n") as out:
        out.write("board,format,E,K,H,F,T,plan,pred_us,pick\n")
        for fmt in BPW:
            for T in range(1, 9):
                jobs = round(E * (1 - (1 - K / E) ** T)); pred = {}
                for pl in plans:
                    if not admitted(BOARD_NAME, fmt, pl, T): continue
                    gu, dn = pl.split("|")
                    a, _ = stage_time(facts, board, fmt, gu, 1, T, jobs, H, F)
                    b, _ = stage_time(facts, board, fmt, dn, 2, T, jobs, H, F)
                    if a is not None and b is not None: pred[pl] = a + b
                pick = min(pred, key=pred.get)
                for pl in plans:
                    if pl in pred:
                        out.write(f"{BOARD_NAME},{fmt},{E},{K},{H},{F},{T},{pl},{pred[pl]*1e6:.3f},{int(pl == pick)}\n")


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        BOARD_NAME = sys.argv[3]
        predict(sys.argv[2], sys.argv[4], tuple(int(x) for x in sys.argv[5].split(",")), sys.argv[6])
    elif sys.argv[1] == "score":
        sys.argv = sys.argv[1:]
        main()
    else:
        raise SystemExit(__doc__)
