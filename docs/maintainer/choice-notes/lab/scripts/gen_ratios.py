"""generation_round_costs of the shipped registry, as src/spec/pairs.cpp computes it."""
import json, statistics, sys
from collections import defaultdict


def costs(tune):
    """As src/spec/pairs.cpp round_costs: per section (repeats pooled) a round's seconds over plain
    decode's seconds per token, then the mean over the sections."""
    plain = defaultdict(lambda: [0.0, 0.0])
    rounds = defaultdict(lambda: [0.0, 0.0])
    for m in tune["measurements"]:
        if m["config"] == "plain":
            if m["tokens"] > 0:
                plain[m["family"]][0] += m["seconds"]; plain[m["family"]][1] += m["tokens"]
        elif m.get("rounds", 0) > 0:
            r = rounds[(m["config"], m["family"])]
            r[0] += m["seconds"]; r[1] += m["rounds"]
    each = defaultdict(list)
    for (c, f), r in rounds.items():
        if f in plain and plain[f][0] > 0:
            each[c].append((r[0] / r[1]) / (plain[f][0] / plain[f][1]))
    return {c: sum(v) / len(v) for c, v in each.items()}


r = json.load(open(sys.argv[1], encoding="utf-8"))["pairs"]
for a_gen, b_gen in (("sm89", "sm120"), ("sm120", "sm89"), ("sm89", "sm86")):
    ratios = defaultdict(list); targets = set()
    for a in r:
        if a["generation"] != a_gen: continue
        ca = costs(a["tune"])
        for b in r:
            if b["generation"] != b_gen or b["target"] != a["target"] or b["engine"] != a["engine"]: continue
            targets.add(b["target_repo"])
            for c, v in costs(b["tune"]).items():
                if c in ca and ca[c] > 0: ratios[c].append(v / ca[c])
    print(a_gen, "->", b_gen, len(targets), "targets")
    for c in sorted(ratios, key=lambda c: (c.split(":")[0], int(c.split(":")[1].split("+")[0]) if ":" in c else 0)):
        print(f"   {c:14s} median {statistics.median(ratios[c]):.3f}  n={len(ratios[c])}  range {min(ratios[c]):.2f}..{max(ratios[c]):.2f}")
