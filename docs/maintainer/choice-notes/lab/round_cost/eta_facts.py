"""kSmallTEta against two fact hypotheses (choice-notes DECISIONS 54).

    python eta_facts.py

1. The small-T compute runs at the dp4a rate (512 int8 ops a SM-clock, CUDA C++ Programming Guide;
   the generation row's s8 peak is 2048 on Turing..Blackwell consumer, Volta's peak is dp4a's):
   the share of it each card's own eta implies.
2. A round costs a fixed time instead of kappa F: with kappa = 0, the time a round leaves over its
   bytes, (R measured - R bytes) x the measured plain seconds a token, per pair against k.
Reads the shipped registry through fit_eta.py (beside it).
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_eta as F  # noqa: E402


def plain_seconds(pair):
    s = t = 0.0
    for m in pair["tune"]["measurements"]:
        if m["config"] == "plain":
            s += m["seconds"]; t += m["tokens"]
    return s / t if t else None


def main():
    rows = [r for r in F.rows_of_registry() if F.fitted(r)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["board"]["name"]].append(r)
    print("| card | rounds | eta | dp4a share | fixed ms a round per pair and mode: k -> ms |")
    print("|---|---|---|---|---|")
    for name, rs in sorted(by.items()):
        b = rs[0]["board"]
        eta = F.fit(rs)
        share = eta * (1 if b["gen"] == "volta" else 4)
        series = collections.defaultdict(list)
        for r in rs:
            tp = plain_seconds(r["p"])
            if not tp:
                continue
            rb = F.R(r["p"]["round_facts"], r["mode"], r["k"], 0.0)
            series[(r["p"]["target_repo"].split("/")[-1][:24], r["mode"])].append(
                (r["k"], (r["meas"] - rb) * tp * 1e3))
        cells = []
        for (t, mode), v in sorted(series.items()):
            v.sort()
            cells.append(f"{t} {mode}: " + " ".join(f"{k}:{ms:.2f}" for k, ms in v))
        print(f"| {name} | {len(rs)} | {eta:.3f} | {share:.2f} | {'; '.join(cells)} |")


if __name__ == "__main__":
    main()
