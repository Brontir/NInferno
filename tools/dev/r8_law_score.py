#!/usr/bin/env python3
"""Score the R8 schedule law's frozen prediction against a measured r8 grid (#613 PR-10, step 10c).

  r8_law_score.py --predicted tests/fixtures/r8_law/predicted_<board>.csv \
                  --grid PASS_1.txt [--grid PASS_2.txt ...] [--band 0.02]

The prediction must be frozen before the grid is read: the CSV has to be committed, unchanged
since, in a commit that is an ancestor of HEAD (the script refuses otherwise). The grid is
ninferno_r8_schedule_bench's output (one `r8 <format> <epilogue> n=N k=K t=1 r8.wWrR <us> us ...`
line per schedule, `picked` on the rules' choice); a schedule's time is its fastest pass.

Per problem the law's pick is the schedule of least rank_us among those the grid measured (the
prediction may cover schedules the grid does not time); schedules of equal rank are a tie the law
does not break, and the first in the CSV's order is its pick. It is set against the fastest
measured schedule among those the prediction covers. Output: one line per problem
where the law misses by more than the band, then the summary: problems scored, picks within the
band, the law's and the rules' mean and worst regret (t(pick) / t(best) - 1), and abstentions.
Exit 0 on a score (whatever it is), 2 on usage, 3 when the prediction is not frozen.
"""

import argparse
import collections
import csv
import re
import subprocess
import sys
from pathlib import Path

LINE = re.compile(
    r"^r8 (\S+) (\S+) n=(\d+) k=(\d+) t=1 r8\.w(\d+)r(\d+) ([0-9.]+) us .*?(picked)?(?: rules)?\s+h="
)


def frozen(path: Path) -> str:
    """The commit that froze the prediction, or an error string."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path.parent,
                             capture_output=True, text=True, check=True).stdout.strip()
        rel = path.resolve().relative_to(Path(top).resolve()).as_posix()
        if subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=top,
                          capture_output=True, text=True, check=True).stdout.strip():
            return f"error: {rel} has uncommitted changes"
        sha = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=top,
                             capture_output=True, text=True, check=True).stdout.strip()
        if not sha:
            return f"error: {rel} is not committed"
        if subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=top).returncode:
            return f"error: {sha} is not an ancestor of HEAD"
        return sha
    except (subprocess.CalledProcessError, ValueError) as e:
        return f"error: {e}"


def read_prediction(path: Path):
    ranks, none = collections.defaultdict(dict), []  # key -> schedule -> rank (CSV order)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["format"], row["epilogue"], int(row["n"]), int(row["k"]))
            if row["schedule"] == "none":
                none.append(key)
                continue
            ranks[key][row["schedule"]] = float(row["rank_us"])
    return ranks, none


def read_grid(paths):
    times = collections.defaultdict(dict)  # key -> schedule -> fastest pass
    rules = {}
    for p in paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            fmt, epi, n, k, w, r, us, picked = m.groups()
            key, sched = (fmt, epi, int(n), int(k)), f"w{w}r{r}"
            t = float(us)
            if sched not in times[key] or t < times[key][sched]:
                times[key][sched] = t
            if picked or " picked " in line:
                rules[key] = sched
    return times, rules


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predicted", required=True, type=Path)
    ap.add_argument("--grid", required=True, action="append")
    ap.add_argument("--band", type=float, default=0.02)
    a = ap.parse_args()
    sha = frozen(a.predicted)
    if sha.startswith("error"):
        print(f"r8_law_score: the prediction is not frozen: {sha}", file=sys.stderr)
        return 3
    ranks, none = read_prediction(a.predicted)
    times, rules = read_grid(a.grid)
    law_regret, rule_regret, within, tie_hits, scored, ties = [], [], 0, 0, 0, 0
    for key in sorted(ranks):
        measured = {s: t for s, t in times.get(key, {}).items() if s in ranks[key]}
        if not measured:
            continue
        rank = {s: r for s, r in ranks[key].items() if s in measured}
        pick = min(rank, key=rank.get)  # the first of equal ranks in the CSV's order
        tied = {s for s, r in rank.items() if r == rank[pick]}
        ties += len(tied) > 1
        scored += 1
        best_s = min(measured, key=measured.get)
        best = measured[best_s]
        reg = measured[pick] / best - 1
        law_regret.append(reg)
        if reg <= a.band:
            within += 1
        elif any(measured[s] / best - 1 <= a.band for s in tied - {pick}):
            tie_hits += 1
        if reg > a.band:
            print(f"miss {' '.join(map(str, key))}: law {pick} {measured[pick]:.2f} us, "
                  f"best {best_s} {best:.2f} us ({100 * reg:+.1f} %)")
        if key in rules and rules[key] in measured:
            rule_regret.append(measured[rules[key]] / best - 1)
    if not scored:
        print("r8_law_score: no problem of the prediction is in the grid", file=sys.stderr)
        return 2
    mean = lambda v: 100 * sum(v) / len(v) if v else float("nan")
    print(f"prediction {sha[:12]} ({a.predicted.name}), {len(a.grid)} pass(es), band {100 * a.band:g} %")
    print(f"scored {scored}, law within the band {within} ({100 * within / scored:.1f} %), "
          f"ties {ties} (another of the tie within the band {tie_hits}), abstained {len(none)}")
    print(f"law regret: mean {mean(law_regret):.2f} %, worst {100 * max(law_regret):.1f} %")
    if rule_regret:
        print(f"rules regret ({len(rule_regret)} problems): mean {mean(rule_regret):.2f} %, "
              f"worst {100 * max(rule_regret):.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
