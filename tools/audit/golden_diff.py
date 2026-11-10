#!/usr/bin/env python3
"""Compact summary of what changed between two selector golden trees (#620 step 8).

The goldens (tests/ops/linear/selector_golden/<build set>/<board>[.<facts digest>].txt) are written
by tests/ops/linear/test_selector_golden.cpp (dump_body / dump_board). A decision change moves
thousands of lines; this prints, per build set and board, how many decisions changed (one decision
= one sampled token width T of one (format, policy, epilogue, shape, profile) problem), grouped by problem
with the T ranges collapsed and by (old route -> new route).

  golden_diff.py BASE_DIR HEAD_DIR            two golden trees (the selector_golden directories)
  golden_diff.py --base REV --head REV        two git revisions, files read with `git show`
Exit 0 always, unless --fail-on-change (1 when anything changed). An unknown line is an error (2).

Line grammar (see dump_body):
  "# <board> <device class id> <arch>"                      first line
  "# facts <digest>"                                        facts-priced boards only
  "<format> <N>x<K> not packed: <reason>"                   a format the test packer cannot build
  "<format> p<policy> <epilogue> <rows>x<k> <profile> T=<a>[..<b>] <route text>"
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

GOLDEN_REL = "tests/ops/linear/selector_golden"
INDEX_NAME = "facts_index.txt"

DECISION = re.compile(
    r"^(?P<fmt>\S+) p(?P<policy>\d+) (?P<epi>\S+) (?P<rows>\d+)x(?P<k>\d+) (?P<profile>\S+)"
    r" T=(?P<lo>\d+)(?:\.\.(?P<hi>\d+))? (?P<route>\S.*)$"
)
NOT_PACKED = re.compile(r"^(?P<fmt>\S+) (?P<n>\d+)x(?P<k>\d+) not packed: (?P<why>.*)$")
FACTS = re.compile(r"^# facts (\S+)$")
HEADER = re.compile(r"^# (\S+) (\S+) (\S+)$")


class GoldenParseError(Exception):
    pass


class Golden:
    """One parsed golden file."""

    def __init__(self):
        self.header = None  # "# board id arch" line, without "# "
        self.facts = None
        self.not_packed = {}  # (fmt, n, k) -> reason
        self.runs = defaultdict(list)  # (fmt, policy, epi, shape, profile) -> [(lo, hi, route)]


def parse_golden(text, name="<golden>"):
    g = Golden()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        where = f"{name}:{number}"
        m = FACTS.match(line)
        if m:
            g.facts = m.group(1)
            continue
        if line.startswith("#"):
            m = HEADER.match(line)
            if not m or g.header is not None:
                raise GoldenParseError(f"{where}: unknown comment line: {line!r}")
            g.header = " ".join(m.groups())
            continue
        m = DECISION.match(line)
        if m:
            lo = int(m.group("lo"))
            hi = int(m.group("hi")) if m.group("hi") else lo
            if hi < lo:
                raise GoldenParseError(f"{where}: empty width range: {line!r}")
            key = (
                m.group("fmt"),
                "p" + m.group("policy"),
                m.group("epi"),
                f"{m.group('rows')}x{m.group('k')}",
                m.group("profile"),
            )
            g.runs[key].append((lo, hi, m.group("route")))
            continue
        m = NOT_PACKED.match(line)
        if m:
            g.not_packed[(m.group("fmt"), m.group("n"), m.group("k"))] = m.group("why")
            continue
        raise GoldenParseError(f"{where}: unknown line: {line!r}")
    for key, runs in g.runs.items():
        runs.sort()
        for (_, hi, _), (lo, _, _) in zip(runs, runs[1:]):
            if lo <= hi:
                raise GoldenParseError(f"{name}: overlapping widths for {' '.join(key)}")
    return g


def grid_of(*goldens):
    """The sampled widths (kTokens in the test): the dump covers only these, and a run T=a..b
    stands for the sampled widths between a and b. Taken as every run end seen in the files."""
    points = set()
    for g in goldens:
        for runs in g.runs.values():
            for lo, hi, _ in runs:
                points.update((lo, hi))
    return sorted(points)


def diff_runs(old, new, grid):
    """Segments (first T, last T, old route or None, new route or None) over the sampled widths
    where the two run lists differ; adjacent sampled widths with the same pair are merged."""

    def route_at(runs, t):
        for lo, hi, route in runs:
            if lo <= t <= hi:
                return route
        return None

    out, count = [], []
    for t in grid:
        r_old, r_new = route_at(old, t), route_at(new, t)
        if r_old == r_new:
            continue
        i = grid.index(t)
        if out and out[-1][2] == r_old and out[-1][3] == r_new and out[-1][4] == i - 1:
            out[-1] = (out[-1][0], t, r_old, r_new, i)
            count[-1] += 1
        else:
            out.append((t, t, r_old, r_new, i))
            count.append(1)
    return [(lo, hi, o, n, c) for (lo, hi, o, n, _), c in zip(out, count)]


def diff_golden(old, new):
    """Changes between two parsed files: dict with meta lines and segments
    {problem key: [(lo, hi, old, new, sampled widths)]}."""
    res = {"meta": [], "segments": {}}
    if old.header != new.header:
        res["meta"].append(f"header: {old.header} -> {new.header}")
    if old.facts != new.facts:
        res["meta"].append(f"facts digest: {old.facts} -> {new.facts}")
    for k in sorted(set(old.not_packed) | set(new.not_packed)):
        a, b = old.not_packed.get(k), new.not_packed.get(k)
        if a != b:
            res["meta"].append(f"not packed {k[0]} {k[1]}x{k[2]}: {a} -> {b}")
    grid = grid_of(old, new)
    for key in sorted(set(old.runs) | set(new.runs)):
        segs = diff_runs(old.runs.get(key, []), new.runs.get(key, []), grid)
        if segs:
            res["segments"][key] = segs
    return res


def widths(segs):
    return sum(c for *_, c in segs)


def t_text(lo, hi):
    return f"T {lo}" if lo == hi else f"T {lo}..{hi}"


def show(route):
    return "(absent)" if route is None else route


def split_name(name):
    """'rtx3090.0ae580162b29.txt' -> ('rtx3090', '0ae580162b29'); 'v100.txt' -> ('v100', None)."""
    stem = name[:-4] if name.endswith(".txt") else name
    board, dot, digest = stem.rpartition(".")
    if dot and re.fullmatch(r"[0-9a-f]{12}", digest):
        return board, digest
    return stem, None


def pair_files(old_names, new_names):
    """Pair files of one set: same name, else the lone old and lone new file of one board
    (a facts digest moved)."""
    pairs = [(n, n) for n in sorted(set(old_names) & set(new_names))]
    only_old = sorted(set(old_names) - set(new_names))
    only_new = sorted(set(new_names) - set(old_names))
    by_board_old, by_board_new = defaultdict(list), defaultdict(list)
    for n in only_old:
        by_board_old[split_name(n)[0]].append(n)
    for n in only_new:
        by_board_new[split_name(n)[0]].append(n)
    for board in sorted(set(by_board_old) | set(by_board_new)):
        o, n = by_board_old.get(board, []), by_board_new.get(board, [])
        if len(o) == 1 and len(n) == 1:
            pairs.append((o[0], n[0]))
        else:
            pairs.extend((x, None) for x in o)
            pairs.extend((None, x) for x in n)
    return pairs


def load_dir(root):
    """{set: {file name: text}} from a selector_golden directory."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"golden_diff: not a directory: {root}")
    tree = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        tree[sub.name] = {
            f.name: f.read_text(encoding="utf-8") for f in sorted(sub.glob("*.txt"))
        }
    return tree


def git(args, repo):
    return subprocess.run(
        ["git", "-C", str(repo)] + args, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def load_rev(rev, repo):
    names = git(["ls-tree", "-r", "--name-only", rev, "--", GOLDEN_REL], repo).splitlines()
    tree = {}
    for path in names:
        parts = path[len(GOLDEN_REL) + 1 :].split("/")
        if len(parts) != 2 or not parts[1].endswith(".txt"):
            continue  # facts_index.txt and anything else outside <set>/<file>.txt
        tree.setdefault(parts[0], {})[parts[1]] = git(["show", f"{rev}:{path}"], repo)
    return tree


def summarize(old_tree, new_tree, detail, out):
    """Prints the summary; returns the number of changed widths + meta changes."""
    total = 0
    sets = sorted(set(old_tree) | set(new_tree))
    for s in sets:
        old_files, new_files = old_tree.get(s, {}), new_tree.get(s, {})
        for a, b in pair_files(old_files.keys(), new_files.keys()):
            label = f"{s}/{b or a}" + (f" (was {a})" if a and b and a != b else "")
            if a is None or b is None:
                text = (new_files if a is None else old_files)[b or a]
                g = parse_golden(text, f"{s}/{b or a}")
                n = sum(len(runs) for runs in g.runs.values())
                out.append(f"{label}: {'ADDED' if a is None else 'REMOVED'} file ({n} runs)")
                total += max(n, 1)
                continue
            res = diff_golden(parse_golden(old_files[a], f"{s}/{a}"), parse_golden(new_files[b], f"{s}/{b}"))
            changed = sum(widths(v) for v in res["segments"].values())
            if not changed and not res["meta"]:
                continue
            total += changed + len(res["meta"])
            out.append(f"{label}: {changed} decisions changed in {len(res['segments'])} problems")
            for m in res["meta"]:
                out.append(f"  {m}")
            transitions = defaultdict(lambda: [0, set()])
            for key, segs in res["segments"].items():
                for lo, hi, r_old, r_new, c in segs:
                    t = transitions[(show(r_old), show(r_new))]
                    t[0] += c
                    t[1].add(key)
            for (r_old, r_new), (n, keys) in sorted(transitions.items(), key=lambda kv: -kv[1][0]):
                out.append(f"  {n:6d} T in {len(keys):4d} problems: {r_old}  ->  {r_new}")
            shown = 0
            for key, segs in res["segments"].items():
                if detail is not None and shown >= detail:
                    out.append(f"  ... {len(res['segments']) - shown} more problems (--detail all)")
                    break
                shown += 1
                fmt, policy, epi, shape, profile = key
                out.append(f"  {fmt} {policy} {epi} {shape} {profile}")
                for lo, hi, r_old, r_new, _ in segs:
                    out.append(f"    {t_text(lo, hi)}: {show(r_old)}  ->  {show(r_new)}")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", nargs="?", help="base selector_golden directory")
    ap.add_argument("head_dir", nargs="?", help="head selector_golden directory")
    ap.add_argument("--base", help="git revision of the base")
    ap.add_argument("--head", help="git revision of the head")
    ap.add_argument("--repo", default=".", help="repository for --base/--head (default: cwd)")
    ap.add_argument("--detail", default="20", help="problems listed per board, or 'all' (default 20)")
    ap.add_argument("--fail-on-change", action="store_true", help="exit 1 when anything changed")
    args = ap.parse_args(argv)
    detail = None if args.detail == "all" else int(args.detail)
    try:
        if args.base or args.head:
            if not (args.base and args.head) or args.base_dir or args.head_dir:
                ap.error("use either BASE_DIR HEAD_DIR or both --base and --head")
            old, new = load_rev(args.base, args.repo), load_rev(args.head, args.repo)
        else:
            if not (args.base_dir and args.head_dir):
                ap.error("give BASE_DIR HEAD_DIR, or --base REV --head REV")
            old, new = load_dir(args.base_dir), load_dir(args.head_dir)
        out = []
        total = summarize(old, new, detail, out)
    except (GoldenParseError, subprocess.CalledProcessError) as error:
        print(f"golden_diff: {getattr(error, 'stderr', None) or error}", file=sys.stderr)
        return 2
    print("\n".join(out) if out else "no golden changes")
    if total:
        print(f"total: {total} changed decisions/lines")
    return 1 if (args.fail_on_change and total) else 0


if __name__ == "__main__":
    sys.exit(main())
