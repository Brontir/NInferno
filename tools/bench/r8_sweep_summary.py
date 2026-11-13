#!/usr/bin/env python3
"""Paste-ready tables of one card's ninferno_r8_schedule_bench sweeps (#850).

    python3 tools/bench/r8_sweep_summary.py SWEEP [SWEEP ...]

Every SWEEP is a complete bench output of one card (its "done" line), read with the rules of
tools/bench/r8_decode_table.py: same=1 on every T = 1 R8 line, or the file is refused. The files
may time different problems (run_850_r8_sweep.sh writes the down projections and the rest of the
step apart); a problem's figures are over the files that timed it.

Per problem: each R8 schedule's median over the sweeps; the shape-class rules' choice (the "rules"
mark: master's pick where the card has no table entry for it); the best schedule and its distance
to the rules'; and whether r8_decode_table.py would write a table entry for it (the best is not the
rules', at least --min-gain faster in the median and faster in every sweep; an add problem is
looked up as its store problem, so only a store row writes one where both were swept).

Then two verdicts, both frozen in docs/maintainer/kernel-proposals.md (entry P) before any sweep:
- the card's prediction for the Q6_K down projection 4096x12288: the candidate schedule's median
  against the rules', inside or outside the band, against the acceptance edge;
- the rule check: on every K-quant problem at K >= 8192 that is not a SwiGLU call, w8r1 against
  w8r2, the two schedules the K rule's change would swap.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8_decode_table as tool  # noqa: E402  (the reader of the merge tool, its regexes)

# Entry P of docs/maintainer/kernel-proposals.md, frozen 13.11 before any sweep: per board class,
# the card, the candidate schedule (warps, rows), the predicted change of the Q6_K 4096x12288 call
# in us (lower edge, upper edge: the upper the larger gain) and the acceptance edge (half the lower).
FROZEN = {
    "ad102:128sm": ("RTX 4090", (8, 1), -2.0, -4.0, -1.0),
    "gb202:170sm": ("RTX 5090", (8, 1), -0.8, -2.2, -0.4),
    "ga102:82sm": ("RTX 3090", (8, 1), -3.0, -7.0, -1.5),
    "ga106:28sm": ("RTX 3060", (8, 1), -1.0, -4.0, -0.5),
    "tu102:68sm": ("RTX 2080 Ti", (4, 1), -1.0, -4.0, -0.5),
    "ad104:60sm": ("RTX 4070 Ti", (8, 1), -0.5, -2.0, -0.25),
    "gb203:70sm": ("RTX 5070 Ti", (8, 1), -0.5, -2.0, -0.25),
}
TARGET = ("q6k", 4096, 12288)
KQUANTS = ("q2k", "q4k", "q5k", "q6k")


def read_sweep(path):
    """(device line, class id, board PCI id, {(fmt, epi, n, k): {(warps, rows): us}}, {key: rules
    schedule}). Exits on an incomplete sweep, unequal bytes or a file with no T = 1 R8 line."""
    times, rules = {}, {}
    device, gpu, board, done = None, None, None, False
    with open(path, errors="replace", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == "done":
            done = True
            continue
        m = tool.DEVICE.match(line)
        if m:
            device, gpu = line.strip(), m.group(1)
            continue
        m = tool.BOARD.match(line)
        if m:
            board = tool.pci_text(m.group(5)) if m.group(5) else None
            continue
        m = tool.ROW.match(line)
        if not m:
            continue
        fmt, epi, n, k, tokens, warps, rows, us, same, pick, rule = m.groups()
        if tokens not in (None, "1"):
            continue
        if same != "1":
            sys.exit(f"{path}: schedules differ in bytes: {line.strip()}")
        key = (fmt, epi, int(n), int(k))
        times.setdefault(key, {})[(int(warps), int(rows))] = float(us)
        if rule or (pick and key not in rules):
            rules[key] = (int(warps), int(rows))
    if not done:
        sys.exit(f"{path}: no 'done' line (an incomplete sweep)")
    if not times:
        sys.exit(f"{path}: no T = 1 R8 schedule line")
    return device, gpu, board, times, rules


def name(schedule):
    return f"w{schedule[0]}r{schedule[1]}"


def add_check(problems, entries, median):
    """The add call against each store entry. A model step runs its down projection as an add call
    (ops::linear_add), which looks the store entry up, so an entry stays only where its schedule
    also beats the add call's rules schedule in the median; one whose add call was not swept rests
    on store alone."""
    if not entries:
        return
    print("**Add check (an add call takes the store entry; it stays only where its schedule also "
          "wins on the add call):**")
    print()
    for (fmt, n, k), best in sorted(entries.items()):
        key = (fmt, "add", n, k)
        if key not in problems:
            print(f"- {fmt} {n}x{k}: store entry {name(best)}; the add call was not swept: the "
                  "entry rests on store alone")
            continue
        rule = problems[key]["rules"]
        t_rule, t_best = (median(key, rule) if rule else None), median(key, best)
        if t_rule is None or t_best is None:
            print(f"- {fmt} {n}x{k}: store entry {name(best)}; the add call's rules or entry "
                  "schedule was not timed")
            continue
        d = t_best - t_rule
        every = all(r[best] < r[rule] for r in problems[key]["runs"] if best in r and rule in r)
        if best == rule:
            verdict = "keep (the add call's rules already pick it)"
        elif d < 0:
            verdict = "keep"
        else:
            verdict = "DROP (slower on the add call)"
        print(f"- {fmt} {n}x{k}: store entry {name(best)}; add {name(rule)} {t_rule:.2f} -> "
              f"{name(best)} {t_best:.2f} us ({d:+.2f}, faster in every sweep: "
              f"{'yes' if every else 'no'}): {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("sweeps", nargs="+")
    args = parser.parse_args()

    runs, device, gpu, board = [], None, None, None
    for path in args.sweeps:
        d, g, b, times, rules = read_sweep(path)
        if gpu is not None and g != gpu:
            sys.exit(f"{path}: device class {g}, the other sweeps {gpu}: one card per call")
        device, gpu, board = d, g, board or b
        runs.append((times, rules))

    problems = {}
    for times, rules in runs:
        for key, by_schedule in times.items():
            entry = problems.setdefault(key, {"runs": [], "rules": rules.get(key)})
            if rules.get(key) != entry["rules"]:
                sys.exit(f"{key}: another rules choice in another sweep")
            entry["runs"].append(by_schedule)
    schedules = sorted({s for p in problems.values() for r in p["runs"] for s in r})
    order = {"store": 0, "add": 1, "swiglu": 2}
    keys = sorted(problems, key=lambda k: (k[3], k[2], k[0], order.get(k[1], 9)))
    stores = {(f, n, k) for f, e, n, k in problems if e == "store"}

    def median(key, s):
        values = [r[s] for r in problems[key]["runs"] if s in r]
        return statistics.median(values) if values else None

    print(f"### {gpu}, pci {board or 'unknown'}: {len(args.sweeps)} sweep files")
    print(f"`{device}`")
    print()
    head = " | ".join(name(s) for s in schedules)
    print(f"| problem | sweeps | rules | {head} | best | best - rules | entry if merged |")
    print("|" + "---|" * (len(schedules) + 6))
    entries = {}  # (fmt, n, k) -> the schedule a store row would write
    for key in keys:
        fmt, epi, n, k = key
        p = problems[key]
        rule = p["rules"]
        med = {s: median(key, s) for s in schedules}
        timed = {s: t for s, t in med.items() if t is not None}
        best = min(timed, key=timed.get)
        cells = " | ".join("-" if med[s] is None else f"{med[s]:.2f}" for s in schedules)
        if rule is None or rule not in timed:
            delta, entry = "-", "no rules mark"
        else:
            d = timed[best] - timed[rule]
            delta = f"{d:+.2f} us ({100 * d / timed[rule]:+.1f} %)"
            every = all(r[best] < r[rule] for r in p["runs"] if best in r and rule in r)
            gain = best != rule and timed[best] <= timed[rule] * (1 - args.min_gain) and every
            entry = "yes" if gain else "no"
            if gain and epi == "store":
                entries[(fmt, n, k)] = best
            if gain and epi == "add" and (fmt, n, k) in stores:
                entry = "no (looked up as store)"
        rules_name = name(rule) if rule else "-"
        print(f"| {fmt} {epi} {n}x{k} | {len(p['runs'])} | {rules_name} | {cells} | "
              f"{name(best)} | {delta} | {entry} |")
    print()
    print("Only the down files are merged (run_850_r8_sweep.sh prints their argument); a \"yes\" on "
          "another row is a finding for another entry, not merged by #850 (a).")
    print()
    add_check(problems, entries, median)

    frozen = FROZEN.get(gpu)
    print("**Frozen prediction (kernel-proposals P), Q6_K 4096x12288:**", end=" ")
    if frozen is None:
        print(f"no frozen band for {gpu}.")
    else:
        card, cand, lower, upper, accept = frozen
        print(f"{card}, candidate {name(cand)}, band {lower:+.1f}..{upper:+.1f} us, "
              f"acceptance <= {accept:+.2f} us.")
        print()
        for epi in ("store", "add"):
            key = (TARGET[0], epi, TARGET[1], TARGET[2])
            if key not in problems:
                print(f"- {epi}: not swept")
                continue
            rule = problems[key]["rules"]
            t_rule, t_cand = median(key, rule), median(key, cand)
            if t_rule is None or t_cand is None:
                print(f"- {epi}: the rules' or the candidate's schedule was not timed")
                continue
            d = t_cand - t_rule
            if rule == cand:
                verdict = "the rules already pick the candidate here (no change to judge)"
            elif d < upper:
                verdict = "over the upper edge: a finding (a law term is wrong)"
            elif d <= lower:
                verdict = "inside the band: accepted"
            elif d <= accept:
                verdict = "under the lower edge, within the acceptance: accepted"
            else:
                verdict = "under half the lower edge: refuted on this card"
            others = ", ".join(f"{name(s)} {median(key, s) - t_rule:+.2f}" for s in schedules
                               if s not in (rule, cand) and median(key, s) is not None)
            print(f"- {epi}: {name(rule)} {t_rule:.2f} -> {name(cand)} {t_cand:.2f} us, "
                  f"{d:+.2f} us ({100 * d / t_rule:+.1f} %): {verdict} (others vs rules: {others})")
    print()

    print("**Rule check (K-quants, K >= 8192, not SwiGLU): w8r1 against w8r2.**")
    print()
    rows, worst = [], None
    for key in keys:
        fmt, epi, n, k = key
        if fmt not in KQUANTS or epi == "swiglu" or k < 8192:
            continue
        a, b = median(key, (8, 1)), median(key, (8, 2))
        if a is None or b is None:
            continue
        pairs = [(r[(8, 1)], r[(8, 2)]) for r in problems[key]["runs"]
                 if (8, 1) in r and (8, 2) in r]
        every = all(x < y for x, y in pairs)
        pct = 100 * (a - b) / b
        worst = pct if worst is None else max(worst, pct)
        rows.append(f"| {fmt} {epi} {n}x{k} | {b:.2f} | {a:.2f} | {a - b:+.2f} ({pct:+.1f} %) | "
                    f"{'yes' if every else 'no'} |")
    if not rows:
        print("no such problem swept")
        return
    print("| problem | w8r2 | w8r1 | w8r1 - w8r2 us | w8r1 faster in every sweep |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))
    print()
    holds = worst <= 0.0
    print(f"On this card w8r1 is {'never slower' if holds else 'slower somewhere'} than w8r2 "
          f"(worst row {worst:+.1f} %); the rule's criterion (entry P): at least 4 boards with "
          "native BF16 MMA (sm_80 and up; Turing and Volta keep their own branch) where it is "
          "never slower, and none where a row is slower by more than 0.5 % "
          f"({'met' if worst <= 0.5 else 'NOT met'} here).")


if __name__ == "__main__":
    main()
