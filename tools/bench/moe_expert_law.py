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

Version 2 (#987): a warp's registers in 256-register units (defect 5), and the issue bound counts the code outside
the K loop (the K split's reduction, the epilogue) and the R8 quantizer's loop (defect 7).
Known defects (#987, DECISIONS 94), all pricing R8 low: the R8 kernel of the mean tokens a job
(the runtime takes the next power of two at or above T); 32 lanes a row assumed (the loop steps by
r8_lanes); R8's dynamic shared memory left out of the occupancy; the quantizer's global waits
counted once, not per iteration. The frozen CSVs record the law with them.

    moe_expert_law.py score FACTS GRID_DIR BOARD           the law's pick against a measured moe11 grid
    moe_expert_law.py predict FACTS BOARD PLANS_CSV E,K,H,F OUT.csv [GU:DOWN,...]
                                                           every plan's price and the pick, for a sealed grid
                                                           (format pairs of a mixed layer: q4_k:q6_k)
    moe_expert_law.py pick [--check] [--rebands]           write (check) routes/moe_pick/<class>.json; a run that
                                                           moves a sealed layer's bands is refused without --rebands
"""
import collections, csv, json, math, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE_DIR = os.path.join(ROOT, 'src', 'core', 'data', 'warp_inflight')
NT_OF = [1]
BOARDS = {  # sms, DRAM GB/s (published), reference clock MHz, max CTAs/SM, threads/SM, compute capability
    "rtx4090": (128, 1008, 2520, 24, 1536, 89),
    "rtx4060ti": (34, 288, 2535, 24, 1536, 89),
    "rtx5090": (170, 1792, 2407, 32, 1536, 120),
    "rtx3090": (82, 936, 1695, 16, 1536, 86),
    "rtx5070ti": (70, 896, 2452, 32, 1536, 120),
    "rtx4070ti": (60, 504, 2610, 24, 1536, 89),
    "rtx3060": (28, 360, 1777, 16, 1536, 86),
    "rtx2080ti": (68, 616, 1545, 16, 1024, 75),
}
# Shared memory an SM holds and the driver's reserve per block, per compute capability: the arch rows of
# src/core/data/gpu_facts.json (64 KiB and no reserve on Turing, 100 KiB and 1 KiB a block from sm_80 on).
_ARCHS = {a["cc"]: a for a in json.load(open(os.path.join(ROOT, "src", "core", "data", "gpu_facts.json")))["archs"]}


def smem_facts(cc):
    a = _ARCHS[cc]
    return a["smem_per_sm_kib"] * 1024, a["smem_reserved_per_block"]
BPW = {"q3_k": 3.4375, "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625}
GEMM_DEC = {"q3_k": "GgmlQ3KDecoder", "q4_k": "GgmlQ45KDecoder<4>", "q5_k": "GgmlQ45KDecoder<5>", "q6_k": "GgmlQ6KDecoder"}
R8_CODEC = {"q3_k": "W4a8Q3K", "q4_k": "W4a8Q45K<4>", "q5_k": "W4a8Q45K<5>", "q6_k": "W4a8Q6K"}
R8_PER_LOAD = {"q3_k": 4, "q4_k": 2, "q5_k": 2, "q6_k": 4}
# The same-format layers a grid and a pick file price by default.
FORMATS = ("q3_k", "q4_k", "q5_k", "q6_k")
# "q4_kf": a Q4_K bank in fragment order (ggml_frag_v1), as convert's expert_bank_layout Auto stores Q4_K
# experts (ninferno_moe_bench --frag 1): its own decoder and R8 codec rows in the snapshot, Q4_K's bytes a
# weight and loads a lane (moe.cpp r8_lanes reads the format, not the layout).
BPW["q4_kf"] = BPW["q4_k"]
GEMM_DEC["q4_kf"] = "GgmlQ4KFragDecoder"
R8_CODEC["q4_kf"] = "ninferno::ops::detail::moe::W4a8Q4KFrag"
R8_PER_LOAD["q4_kf"] = R8_PER_LOAD["q4_k"]
# The candidates are the plans the call may run (moe.cpp r8_admits): an R8 stage only up to the R8 token limit,
# moe_regime_rules.h's records (the generation's: eight on Ada and consumer Blackwell, four elsewhere; a non-frag
# Q6_K bank four on every tensor mma.sync build; a Q3_K, Q5_K or published IQ2_XXS bank eight on GA10x, ga10x_q3k_r8_wide /
# ga10x_q5k_r8_wide / ga10x_iq2xxs_r8_wide, #1166). The law prices the admitted plans; admission itself is the runtime's.
GENERATION = {"rtx4090": "ada", "rtx4060ti": "ada", "rtx4070ti": "ada", "rtx5090": "gb20x", "rtx5070ti": "gb20x",
              "rtx3090": "ga10x", "rtx3060": "ga10x", "rtx2080ti": "turing"}


def r8_token_limit(board_name, fmt):
    if fmt == "q6_k":
        return 4
    if fmt in ("q3_k", "q5_k", "iq2_xxs") and GENERATION[board_name] == "ga10x":
        return 8
    return 8 if GENERATION[board_name] in ("ada", "gb20x") else 4


def admitted(board_name, fmt, plan, tokens, down_fmt=None):
    """Whether the plan's R8 stages run at `tokens`: each stage by its own bank's format (gate/up `fmt`, down
    `down_fmt`, the same where not given: a mixed layer such as Q4_K_M's q4_k gate/up over a q6_k down)."""
    fmts = (fmt, down_fmt or fmt)
    return all(not s.startswith("r8") or tokens <= r8_token_limit(board_name, f) for s, f in zip(plan.split("|"), fmts))


def price_plans(facts, board, board_name, fmt, down_fmt, T, E, K, H, F, plans):
    """{plan: predicted s} of every admitted plan the law prices, a layer of gate/up format `fmt` over a down
    bank of `down_fmt` at T tokens."""
    jobs = round(E * (1 - (1 - K / E) ** T)); pred = {}
    for pl in plans:
        if not admitted(board_name, fmt, pl, T, down_fmt): continue
        gu, dn = pl.split("|")
        a, _ = stage_time(facts, board, fmt, gu, 1, T, jobs, H, F, K)
        b, _ = stage_time(facts, board, down_fmt, dn, 2, T, jobs, H, F, K)
        if a is not None and b is not None: pred[pl] = a + b
    return pred


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
    (bytes = loads in flight a lane x 32 lanes x the load's bytes x warps: 16, or 8 / 4 for probe 7's `_b8` / `_b4`
    chains, warp_inflight.load_bytes_of), the best rate at or below those bytes, linear between rows."""
    if board_name not in _curves:
        pts = {}
        for line in open(os.path.join(PROBE_DIR, PROBE[board_name])):
            f = line.split()
            if f[:3] == ["row", "placed", "1"]:
                load = 8 if f[3].endswith("_b8") else 4 if f[3].endswith("_b4") else 16
                b = int(f[4]) * 32 * load * int(f[5]); g = float(f[6]) * 1e9
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


def occupancy(regs, smem, threads, max_ctas, max_threads, cc):
    # A warp's registers are allocated in units of 256 (#987 defect 5; register_mma_occupancy rounds the same way).
    warps = max(1, threads // 32)
    per_warp = math.ceil(regs * 32 / 256) * 256 if regs else 0
    by_regs = (64 * 1024) // (per_warp * warps) if regs else max_ctas
    smem_sm, reserved = smem_facts(cc)
    by_smem = smem_sm // (smem + reserved) if smem else max_ctas
    return max(0, min(max_ctas, by_regs, by_smem, max_threads // threads))


def stage_time(facts, board, fmt, sched, stage, T, jobs, H, F, K=8):
    sms, gbs, mhz, max_ctas, max_thr, cc = board
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
        nt = min(8, max(1, math.ceil(T * K / max(1, jobs))))  # tokens a job holds (T x top-k / jobs; #987: the runtime's NT differs)
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
    res = occupancy(int(f["registers"]), int(f["shared"]), threads, max_ctas, max_thr, cc)
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
    # Issue: every warp instruction of the busiest SM's CTAs over its four schedulers (one a clock each): the K
    # loop's, the code outside it (index setup, the K split's cross-warp reduction through shared memory, the
    # epilogue: outside_total) and, for an R8 kernel that quantizes its input, the quantizer's (#987 defect 7).
    per_warp = iters * int(f["total"]) + int(f.get("outside_total", 0) or 0)
    if f.get("pro_loops", "0") not in ("0", None):
        per_warp += math.ceil(NT_OF[0] * k / 32 / threads) * int(f.get("pro_total", 0) or 0)
    t_issue = math.ceil(ctas / sms) * warps * per_warp / (4 * clock)
    return max(t_dram, t_chain, t_issue) + c, (t_dram, t_chain, waves, conc, res, t_issue)


def main():
    global BOARD_NAME
    facts = load_facts(sys.argv[1]); D = sys.argv[2]; board = BOARDS[sys.argv[3]]; BOARD_NAME = sys.argv[3]
    for fmt in FORMATS:
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
                a, _ = stage_time(facts, board, fmt, gu, 1, T, jobs, H, F, K)
                b, _ = stage_time(facts, board, fmt, dn, 2, T, jobs, H, F, K)
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



def predict(facts_path, plans_from, geo, out_path, pairs=None):
    """The frozen form: every plan's predicted us and the pick, per format and T, for a board whose grid is
    sealed. The plan list is read from a training grid's sweep (the same 144 plans every board runs)."""
    global BOARD_NAME
    facts = load_facts(facts_path); board = BOARDS[BOARD_NAME]
    plans = sorted({r["plan"] for r in csv.DictReader(open(plans_from)) if r.get("plan")})
    E, K, H, F = geo
    with open(out_path, "w", newline="\n") as out:
        out.write("board,format,E,K,H,F,T,plan,pred_us,pick" + (",down_format" if pairs else "") + "\n")
        for fmt, down_fmt in pairs or [(f, f) for f in FORMATS]:
            for T in range(1, 9):
                pred = price_plans(facts, board, BOARD_NAME, fmt, down_fmt, T, E, K, H, F, plans)
                if not pred:
                    raise SystemExit(f"{facts_path}: no plan of {fmt}/{down_fmt} T={T} is priced (no moer8 / moegemm section?)")
                pick = min(pred, key=pred.get)
                for pl in plans:
                    if pl in pred:
                        out.write(f"{BOARD_NAME},{fmt},{E},{K},{H},{F},{T},{pl},{pred[pl]*1e6:.3f},{int(pl == pick)}"
                                  + (f",{down_fmt}" if pairs else "") + "\n")


# The stage schedules a plan pairs: every compiled grouped schedule (moe.cpp NINFERNO_MOE_SCHEDULES and
# NINFERNO_MOE_R8_SCHEDULES), the 12 x 12 plans `ninferno_moe_bench --plans sweep` times.
SWEEP_STAGES = ("rmma.m1.n1.w1x4", "rmma.m1.n1.w1x8", "rmma.m1.n1.w2x2", "rmma.m1.n1.w4x1", "rmma.m1.n4.w1x4",
                "rmma.m1.n4.w1x8", "rmma.m1.n4.w4x1", "rmma.m1.n8.w4x1", "r8.w2r1", "r8.w4r1", "r8.w4r2", "r8.w8r1")
# The boards and layer geometries (E, K, H, F) whose picks ship (routes/moe_pick/<class>.json, #987): each one
# passed a sealed grid against production auto under the rule mean < auto and max <= auto + 2 points (the frozen
# grids and their scores: the issue's comments). A board or a geometry joins only after its own sealed grid passes.
# "class" is the board's device class id (DeviceClass::id(), the route lookup's route_gpu_class), so a cut-down
# chip names its SM count; the file name spells its ':' as '-' (route_table_file_name), as routes/pick does.
# The product's mixed expert banks (gate/up : down) of the K-quant mixes, Q4_K in fragment order (q4_kf).
MIXED_FRAG = (("q3_k", "q4_kf"), ("q4_kf", "q6_k"), ("q5_k", "q6_k"))
PICKS = {
    "rtx3090": {"class": "ga102:82sm", "pci_id": "0x2204", "arch": "86",
                # Granite 3.0 MoE (E40): v2's own sealed grid. OLMoE (E64) passed sealed under v1 only (v2 was
                # scored on it after unsealing), so it waits for a grid of its own.
                # Qwen3.6 35B-A3B (E256), the four K-quants with both banks in the published layout; and the
                # product's mixed banks (Q3_K_M / Q4_K_M / Q5_K_M layers, Q4_K in fragment order as convert's
                # Auto layout stores it): sealed grid 5.
                "layers": ((40, 8, 1536, 512), (256, 8, 2048, 512), (256, 8, 2048, 512, MIXED_FRAG))},
    "rtx3060": {"class": "ga106:28sm", "pci_id": "0x2504", "arch": "86",
                "layers": ((256, 8, 2048, 512, MIXED_FRAG),)},  # Qwen3.6 35B-A3B, mixed banks: sealed grid 5
    "rtx4060ti": {"class": "ad106:34sm", "pci_id": "0x2805", "arch": "89",
                  "layers": ((64, 8, 2048, 1024),)},  # OLMoE (E64): v1 = v2 on this board, sealed
}
PICK_SCHEMA = "ninferno-moe-pick/1"
PICK_DIR = os.path.join(ROOT, "routes", "moe_pick")


def _sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sha256_text(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _facts_sections(path):
    """The snapshot's moegemm and moer8 sections, the only ones the law reads (a change elsewhere in the
    snapshot leaves the picks in step)."""
    keep, out = False, []
    for line in open(path, encoding="utf-8"):
        if line.startswith("family "):
            keep = line.split()[1] in ("moegemm", "moer8")
        if keep:
            out.append(line)
    return "".join(out)


def _arch_rows():
    """gpu_facts.json's archs array, the only part of it the law reads (smem_facts)."""
    facts = json.load(open(os.path.join(ROOT, "src", "core", "data", "gpu_facts.json"), encoding="utf-8"))
    return json.dumps(facts["archs"], sort_keys=True)


def pick_document(board_name):
    """The routes/moe_pick file of one board: per layer geometry and format, the law's cheapest plan at each
    decode width T 1-8, as route-table moe bands ([t_min, t_max, "<gate/up>|<down>"], equal neighbours joined)."""
    global BOARD_NAME
    BOARD_NAME = board_name
    spec = PICKS[board_name]
    facts_path = os.path.join(ROOT, "src", "ops", "generic", "data", "kernel_facts", f"sm_{spec['arch']}.txt")
    facts = load_facts(facts_path)
    board = BOARDS[board_name]
    plans = [f"{a}|{b}" for a in SWEEP_STAGES for b in SWEEP_STAGES]
    entries = []
    for layer in spec["layers"]:
        E, K, H, F = layer[:4]
        for fmt, down_fmt in (layer[4] if len(layer) > 4 else [(f, f) for f in FORMATS]):
            picks = []
            for T in range(1, 9):
                pred = price_plans(facts, board, board_name, fmt, down_fmt, T, E, K, H, F, plans)
                if not pred:
                    raise SystemExit(f"{facts_path}: no plan of {fmt}/{down_fmt} T={T} is priced (no moer8 / moegemm section?)")
                picks.append(min(sorted(pred), key=pred.get))
            bands = []
            for T, pl in enumerate(picks, start=1):
                if bands and bands[-1][2] == pl: bands[-1][1] = T
                else: bands.append([T, T, pl])
            spell = lambda f: f[:-1].replace("_", "") if f.endswith("f") else f.replace("_", "")
            entry = {"format": spell(fmt)}
            if spell(down_fmt) != spell(fmt): entry["down_format"] = spell(down_fmt)
            frag = [bank for bank, f in (("gate_up", fmt), ("down", down_fmt)) if f.endswith("f")]
            if frag: entry["frag"] = frag
            entry.update({"experts": E, "top_k": K, "hidden": H, "inter": F, "bands": bands})
            entries.append(entry)
    rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/")
    probe = os.path.join(PROBE_DIR, PROBE[board_name])
    inputs = {rel(os.path.abspath(__file__)): _sha256(os.path.abspath(__file__)),
              rel(facts_path) + " moegemm+moer8": _sha256_text(_facts_sections(facts_path)),
              rel(probe): _sha256(probe),
              "src/core/data/gpu_facts.json archs": _sha256_text(_arch_rows())}
    return {"schema": PICK_SCHEMA, "gpu": {"class": spec["class"], "pci_id": spec["pci_id"], "board": board_name},
            "source": "tools/bench/moe_expert_law.py pick (the expert-call law v2, #987): priced, not measured",
            "inputs": inputs, "moe": entries}


def pick_text(board_name):
    """The file's text: the header indented, one line a layer (its key and bands), as a reviewer reads it."""
    doc = pick_document(board_name)
    head = json.dumps({k: v for k, v in doc.items() if k != "moe"}, indent=1)
    rows = ",\n".join("  " + json.dumps(e) for e in doc["moe"])
    return head[:-2] + ',\n "moe": [\n' + rows + "\n ]\n}\n"


def _bands(text):
    """{(format, down_format, frag, E, K, H, F): bands} of a pick file's text. Two entries of one key would
    collapse into one and compare wrongly, so a file that holds one twice is refused (#1115)."""
    out = {}
    for e in json.loads(text)["moe"]:
        key = (e["format"], e.get("down_format", e["format"]), tuple(sorted(e.get("frag", ()))), e["experts"],
               e["top_k"], e["hidden"], e["inter"])
        if key in out: raise ValueError(f"routes/moe_pick: layer {key} appears twice")
        out[key] = e["bands"]
    return out


def write_picks(check, rebands=False):
    """Writes (or, `check`, compares) every board's routes/moe_pick/<class>.json; exit 1 on a stale file.
    The shipped plans are the sealed grids' frozen picks: a run whose inputs moved restamps a file only
    where every layer it held keeps its bands and its key. Bands that moved (a changed law, or kernel facts
    that moved the prices) or a layer whose key is gone (rekeyed or dropped) are refused unless `rebands`
    (--rebands), after a new sealed grid of those layers. A new layer's key passes."""
    stale, moved = [], {}
    want = {PICKS[b]["class"].replace(":", "-") + ".json": pick_text(b) for b in PICKS}
    have = set(os.listdir(PICK_DIR)) if os.path.isdir(PICK_DIR) else set()
    for name, text in sorted(want.items()):
        path = os.path.join(PICK_DIR, name)
        old = open(path, encoding="utf-8").read() if name in have else None
        if old == text: continue
        stale.append(name)
        if old is not None:
            before, after = _bands(old), _bands(text)
            # A sealed layer whose key changed (its geometry, format pair or fragment layout) leaves
            # its old key behind: a removed key moved as much as changed bands. New keys are new
            # layers and pass.
            changed = sorted(str(k) for k in before.keys() & after.keys() if before[k] != after[k])
            changed += sorted(f"{k} (removed)" for k in before.keys() - after.keys())
            if changed: moved[name] = changed
        if not check and (rebands or name not in moved):
            os.makedirs(PICK_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f: f.write(text)
    extra = sorted(n for n in have if n.endswith(".json") and n not in want)
    if moved:
        print("routes/moe_pick: the bands of sealed layers moved or were removed: " +
              "; ".join(f"{n}: {', '.join(k)}" for n, k in moved.items()) +
              (" (written: --rebands)" if rebands and not check else
               ": refused; a sealed grid of these layers first, then --rebands"))
        if check or not rebands:
            sys.exit(1)
    if check:
        if stale or extra:
            print(f"routes/moe_pick: stale {stale}, not generated {extra}: run tools/bench/moe_expert_law.py pick")
            sys.exit(1)
        print(f"routes/moe_pick: {len(want)} boards, in step with the law and the facts")
    else:
        print(f"routes/moe_pick: wrote {stale or 'nothing (in step)'}" + (f"; remove {extra}" if extra else ""))


if __name__ == "__main__":
    if sys.argv[1] == "pick":
        write_picks("--check" in sys.argv[2:], "--rebands" in sys.argv[2:])
    elif sys.argv[1] == "predict":
        BOARD_NAME = sys.argv[3]
        # an optional 7th argument: gate/up:down format pairs, "q3_k:q4_k,q4_k:q6_k"
        pairs = [tuple(x.split(":")) for x in sys.argv[7].split(",")] if len(sys.argv) > 7 else None
        predict(sys.argv[2], sys.argv[4], tuple(int(x) for x in sys.argv[5].split(",")), sys.argv[6], pairs)
    elif sys.argv[1] == "score":
        sys.argv = sys.argv[1:]
        main()
    else:
        raise SystemExit(__doc__)
