"""generation_round_costs of the shipped registry, as src/spec/pairs.cpp computes it."""
import json, statistics, sys
from collections import defaultdict


def costs(tune):
    plain = {m["family"]: m["seconds"] / m["tokens"] for m in tune["measurements"]
             if m["config"] == "plain" and m["tokens"] > 0}
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for m in tune["measurements"]:
        if m["config"] == "plain" or m.get("rounds", 0) <= 0 or m["family"] not in plain:
            continue
        s = sums[m["config"]]
        s[0] += m["seconds"]; s[1] += m["rounds"]; s[2] += plain[m["family"]]; s[3] += 1
    return {c: (s[0] / s[1]) / (s[2] / s[3]) for c, s in sums.items() if s[2] > 0}


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
