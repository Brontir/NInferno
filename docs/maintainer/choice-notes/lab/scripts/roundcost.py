"""Per configuration of a spec-tune: tokens per round and a round's cost in plain decode steps
(seconds per round over plain seconds per token, the mean over the task families both measured)."""
import json, sys
from collections import defaultdict


def table(measurements, only_non_thinking=True):
    plain = {m["family"]: m["seconds"] / m["tokens"] for m in measurements if m["config"] == "plain"}
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, []])
    for m in measurements:
        if m["config"] == "plain" or not m.get("rounds"):
            continue
        if only_non_thinking and "think" in m["family"]:
            continue
        a = agg[m["config"]]
        a[0] += m["seconds"]; a[1] += m["rounds"]; a[2] += m["tokens"]; a[3].append(m["family"])
    out = {}
    for c, (s, r, t, fams) in agg.items():
        p = sum(plain[f] for f in fams if f in plain) / max(1, sum(1 for f in fams if f in plain))
        out[c] = (t / r, (s / r) / p)
    return out


def load(path, target=None):
    d = json.load(open(path, encoding="utf-8"))
    if "pairs" in d:
        for p in d["pairs"]:
            if p["target_repo"] == target and p["tune"]["key"]["gpu"].startswith(sys.argv[3]):
                return p["tune"]["measurements"]
        raise SystemExit("no pair")
    return d["measurements"]


if __name__ == "__main__":
    a = table(load(sys.argv[1]))
    reg = json.load(open(sys.argv[2], encoding="utf-8"))
    b = table(load(sys.argv[2], sys.argv[4]))
    print(f"{'config':10s} {'tpr new':>8s} {'tpr ref':>8s} {'cost new':>9s} {'cost ref':>9s} {'ratio':>6s}")
    for c in sorted(set(a) & set(b)):
        print(f"{c:10s} {a[c][0]:8.2f} {b[c][0]:8.2f} {a[c][1]:9.2f} {b[c][1]:9.2f} {a[c][1]/b[c][1]:6.3f}")
