"""A speculation round's cost from facts (src/spec/round_cost.h) against the lab's measurements.

    python fit_eta.py [--hard DIR]

Reads the shipped registry's pairs (their round_facts and lab_board, tools/spec/round_facts.py) and
the PCI-ID table (src/core/data/gpu_facts.json). Prints:
  1. eta per generation (round_cost.h kSmallTEta): the share of the int8 tensor peak that makes
     R(facts) fit the measured rounds of Qwen targets (qwen3_5, not Q8_0) verifying 5-8 rows
     (k 4..7), least squares on log(R measured / R facts);
  2. R(facts) against R measured (a round's seconds over plain decode's seconds a token, the mean
     over the sections) per card, family and depth class;
  3. per generation of two lab cards or more, each card's rounds against R(facts) at the eta
     fitted on the generation's other cards only (a hold-out of the card), and the configuration
     (k <= 7) a plan picks for a target the other card measured: the other card's tokens a round
     over this card's R(facts), each section weighted as the mix (section_weight), against the
     configuration this card's own tune measured fastest (k <= 7);
  4. with --hard DIR (hard50 results: DIR/<stand>/<tag>/{plain,mtp:K,dflash:K}.json), the speedup
     E / R(facts) on each card with that card left out of its generation's eta fit, against the
     measured speedup (E counted as tokens - 1 - accepted - plain steps rounds).
"""
import argparse
import collections
import glob
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REG = json.loads((ROOT / "src/spec/data/drafter_registry.json").read_text(encoding="utf-8"))
BOARDS = {g["id"].upper(): g for g in json.loads((ROOT / "src/core/data/gpu_facts.json").read_text(encoding="utf-8"))["gpus"]}
GEN = {70: "volta", 75: "turing", 86: "ga10x", 89: "ada", 120: "gb20x"}
SHARE = 0.30
STANDS = {"l4070": "0x2782", "l4090": "0x2684", "l4060": "0x2805", "l5070": "0x2C05", "l5090": "0x2B85",
          "l3060": "0x2503", "l2080": "0x1E04", "lgv100": "0x1DBA"}
HARD_TAGS = {"q9b_q4km": "unsloth/Qwen3.5-9B-MTP-GGUF@9716a636ee4bddc3fed678220b7a33dd2a4160ae Q4_K_M",
             "g12_q4km": "unsloth/gemma-4-12b-it-GGUF@fc034cfff751157913579611efad8462ac1be606 Q4_K_M",
             "oss20": "ggml-org/gpt-oss-20b-GGUF@15dcfc7e7c608c6161507211ed43c5b590801b4b MXFP4",
             "oss20_m": "ggml-org/gpt-oss-20b-GGUF@15dcfc7e7c608c6161507211ed43c5b590801b4b MXFP4",
             "q36_q4km": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d UD-IQ3_S"}


def board(pci):
    r = BOARDS[pci.upper()]
    s8 = 512 if r["cc"] == 70 else 2048  # core/generation.h s8; Volta dp4a's 512
    return dict(name=r["name"], gen=GEN[r["cc"]], bw=r["bus_bits"] / 8 * r["gbps"] * 1e9,
                peak=s8 * r["sms"] * r["boost_mhz"] * 1e6)


def frac(p, rows):
    if not p.get("experts"):
        return 1.0
    pick = p["experts_used"] / p["experts"]
    return pick + SHARE * ((1 - (1 - pick) ** rows) - pick)


def cost(p, rows, kappa):
    act = p["experts_used"] / p["experts"] if p.get("experts") else 1.0
    return (p["bytes"] + p.get("expert_bytes", 0) * frac(p, rows)
            + kappa * 2 * rows * (p["params"] + p.get("expert_params", 0) * act))


def R(rf, mode, k, kappa):
    d = rf["drafters"].get(mode)
    if d is None:
        return None
    n = k if d["per_draft"] else 1
    head = cost(d["head"], k, kappa) if "head" in d else 0.0  # DFlash: the head over the k drafts
    return (cost(rf["target"], k + 1, kappa) + n * cost(d, d["rows"], kappa) + head) / cost(rf["target"], 1, kappa)


def measured(pair):
    plain = collections.defaultdict(lambda: [0.0, 0.0])
    rr = collections.defaultdict(lambda: [0.0, 0.0])
    for m in pair["tune"]["measurements"]:
        if m["config"] == "plain":
            plain[m["family"]][0] += m["seconds"]; plain[m["family"]][1] += m["tokens"]
        elif m["rounds"]:
            a = rr[(m["config"], m["family"])]; a[0] += m["seconds"]; a[1] += m["rounds"]
    out = collections.defaultdict(list)
    for (c, f), (s, r) in rr.items():
        if f in plain and plain[f][0] > 0:
            out[c].append((s / r) / (plain[f][0] / plain[f][1]))
    return {c: sum(v) / len(v) for c, v in out.items()}


def family(p):
    k = p["target_repo"]
    return ("gemma" if p["engine"] == "gemma4" else "nemotron" if p["engine"] == "nemotron_h"
            else "gpt-oss" if "oss" in k else "muse" if "Muse" in k else "qwen")


def rows_of_registry():
    out = []
    for p in REG["pairs"]:
        if "round_facts" not in p or "lab_board" not in p:
            continue
        b = board(p["lab_board"])
        for c, v in measured(p).items():
            mode, k = c.split(":")
            if mode in p["round_facts"]["drafters"]:
                out.append(dict(p=p, gpu=p["tune"]["key"]["gpu"], board=b, mode=mode, k=int(k), meas=v,
                                fam=family(p), q8="Q8_0" in p["target_repo"],
                                moe=bool(p["round_facts"]["target"].get("experts"))))
    return out


def kappa(b, eta):
    return b["bw"] / (eta * b["peak"])


def fit(rows):
    best = None
    for i in range(400):
        eta = 0.05 * 1.01 ** i
        s = sum(math.log(r["meas"] / R(r["p"]["round_facts"], r["mode"], r["k"], kappa(r["board"], eta))) ** 2
                for r in rows)
        if best is None or s < best[0]:
            best = (s, eta)
    return best[1]


WEIGHT = {"code": 7, "math": 4, "prose": 8, "thinking": 1}  # spec/profile.h section_weight


def sections(pair):
    """{config: {family: (seconds a token, tokens a round)}} of a pair's tune."""
    out = collections.defaultdict(dict)
    for m in pair["tune"]["measurements"]:
        if m["tokens"]:
            tpr = (m["accepted"] + m["rounds"]) / m["rounds"] if m["rounds"] else 1.0
            out[m["config"]][m["family"]] = (m["seconds"] / m["tokens"], tpr)
    return out


def mix_speedup(spt, plain):
    """The mix's speedup from seconds a token by section (plain's sections the reference)."""
    fams = [f for f in plain if f in spt and WEIGHT.get(f)]
    return sum(WEIGHT[f] * plain[f] for f in fams) / sum(WEIGHT[f] * spt[f] for f in fams) if fams else 0.0


def cross(rows):
    """Item 3: a hold-out of each card of a generation with two lab cards or more."""
    print("\n| generation | held-out card | eta (others) | rounds k<=7 | R measured / R facts | rms |"
          "\n|---|---|---|---|---|---|")
    plans = []
    for g in sorted({r["board"]["gen"] for r in rows}):
        names = sorted({r["board"]["name"] for r in rows if r["board"]["gen"] == g and fitted(r)})
        if len(names) < 2:
            continue
        for name in names:
            eta = fit([r for r in rows if r["board"]["gen"] == g and fitted(r) and r["board"]["name"] != name])
            own = [r for r in rows if r["board"]["name"] == name and r["fam"] == "qwen" and not r["q8"] and r["k"] <= 7]
            e = [r["meas"] / R(r["p"]["round_facts"], r["mode"], r["k"], kappa(r["board"], eta)) - 1 for r in own]
            print(f"| {g} | {name} | {eta:.3f} | {len(e)} | {100 * sum(e) / len(e):+.1f} % | {100 * (sum(x * x for x in e) / len(e)) ** .5:.1f} % |")
            # The plan for a target the generation's other card measured, priced for this card.
            for p in REG["pairs"]:
                if "round_facts" not in p or "lab_board" not in p or board(p["lab_board"])["name"] != name:
                    continue
                for q in REG["pairs"]:
                    if (q is p or q.get("target") != p["target"] or "lab_board" not in q
                            or board(q["lab_board"])["gen"] != g or board(q["lab_board"])["name"] == name):
                        continue
                    b = board(p["lab_board"]); own_s = sections(p); other = sections(q)
                    plain = {f: v[0] for f, v in own_s["plain"].items()}
                    best = {}
                    for c in own_s:
                        if c == "plain" or int(c.split(":")[1]) > 7 or c not in other:
                            continue
                        mode, k = c.split(":")[0], int(c.split(":")[1])
                        rp = R(p["round_facts"], mode, k, kappa(b, eta))
                        if rp is None:
                            continue
                        # spec/pairs.cpp round_cost_estimate: the other card's measured excess
                        # over its own R(facts) raises the estimate (only up, beyond 5 %).
                        qb = board(q["lab_board"]); qm = measured(q).get(c)
                        if qm:
                            excess = qm / R(q["round_facts"], mode, k, kappa(qb, eta))
                            rp *= excess if excess > 1.05 else 1.0
                        pred = {f: plain[f] * rp / other[c][f][1] for f in plain if f in other[c]}
                        meas = {f: v[0] for f, v in own_s[c].items()}
                        best[c] = (mix_speedup(pred, plain), mix_speedup(meas, plain))
                    if best:
                        pc = max(best, key=lambda c: best[c][0]); mc = max(best, key=lambda c: best[c][1])
                        plans.append((name, p["target_repo"].split("@")[0] + " " + p["target_repo"].split(" ")[-1],
                                      board(q["lab_board"])["name"], pc, best[pc][0], best[pc][1], mc, best[mc][1]))
    print("\n| card | target | E from | plan picks | predicted x | measured x of it | measured best | its x |"
          "\n|---|---|---|---|---|---|---|---|")
    for t in plans:
        print(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]:.2f} | {t[5]:.2f} | {t[6]} | {t[7]:.2f} |")


def fitted(r):
    return r["fam"] == "qwen" and not r["q8"] and 4 <= r["k"] <= 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard")
    args = ap.parse_args()
    rows = rows_of_registry()
    gens = sorted({r["board"]["gen"] for r in rows})
    etas = {}
    print("| generation | eta | cards |\n|---|---|---|")
    for g in gens:
        sel = [r for r in rows if r["board"]["gen"] == g and fitted(r)]
        etas[g] = fit(sel)
        print(f"| {g} | {etas[g]:.3f} | {', '.join(sorted({r['board']['name'] for r in sel}))} |")
    print("\n| card | family | k | rounds | R measured / R facts: mean | rms |\n|---|---|---|---|---|---|")
    grp = collections.defaultdict(list)
    for r in rows:
        key = (r["board"]["name"], r["fam"] + (" MoE" if r["moe"] else "") + (" Q8_0" if r["q8"] else ""),
               "8-16" if r["k"] > 7 else "1-7")
        grp[key].append(r["meas"] / R(r["p"]["round_facts"], r["mode"], r["k"], kappa(r["board"], etas[r["board"]["gen"]])) - 1)
    for (name, fam, ks), e in sorted(grp.items()):
        print(f"| {name} | {fam} | {ks} | {len(e)} | {100 * sum(e) / len(e):+.1f} % | {100 * (sum(x * x for x in e) / len(e)) ** .5:.1f} % |")
    cross(rows)
    if args.hard:
        print("\n| held-out card | model | config | E | measured x | predicted x | error |\n|---|---|---|---|---|---|---|")
        pairs = {p["target_repo"]: p for p in REG["pairs"] if "round_facts" in p}
        for d in sorted(glob.glob(os.path.join(args.hard, "*", "*"))):
            stand, tag = Path(d).parts[-2:]
            key = HARD_TAGS.get(tag)
            if key not in pairs or stand not in STANDS or not os.path.exists(os.path.join(d, "plain.json")):
                continue
            b = board(STANDS[stand])
            sel = [r for r in rows if r["board"]["gen"] == b["gen"] and fitted(r) and r["board"]["name"] != b["name"]]
            if not sel:
                continue  # its generation has no other card
            eta = fit(sel)
            plain = json.load(open(os.path.join(d, "plain.json")))["summary"]
            for f in sorted(glob.glob(os.path.join(d, "*.json"))):
                c = os.path.basename(f)[:-5].replace("_", ":")
                if c == "plain" or int(c.split(":")[1]) > 7:
                    continue
                run = json.load(open(f))
                acc = sum(x["accepted"] for x in run["rows"])
                rounds = sum(max(x["tokens"] - 1 - x["accepted"] - x.get("plain_steps", 0), 0) for x in run["rows"])
                rp = R(pairs[key]["round_facts"], c.split(":")[0], int(c.split(":")[1]), kappa(b, eta))
                if not rounds or rp is None:
                    continue
                E = (acc + rounds) / rounds
                S = run["summary"]["tok_s"] / plain["tok_s"]
                print(f"| {b['name']} | {tag} | {c} | {E:.2f} | {S:.2f} | {E / rp:.2f} | {100 * (E / rp / S - 1):+.0f} % |")


if __name__ == "__main__":
    main()
