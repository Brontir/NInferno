#!/usr/bin/env python3
"""Compact summary of what changed between two selector golden trees (#620 step 8).

The goldens (tests/ops/linear/selector_golden/<build set>/<board>[.<facts digest>].txt) are written
by tests/ops/linear/test_selector_golden.cpp (dump_body / dump_board). A decision change moves
thousands of lines; this prints, per build set and board, how many decisions changed (one decision
= one sampled token width T of one (format, policy, epilogue, shape, profile) problem), grouped by problem
with the T ranges collapsed and by (old route -> new route).

  golden_diff.py BASE_DIR HEAD_DIR            two golden trees (the selector_golden directories)
  golden_diff.py --base REV --head REV        two git revisions, files read with `git show`
  golden_diff.py --range A...B                a branch's own changes: base = `git merge-base A B`
                                              (a plain --base master also shows master's changes)
Boards are compared by the file holding the release's choices: with facts_index.txt (before #629)
the file of the row of the `release` nvcc, so a repointed row is a change and several digest files
of one board are not "removed + added". Added/removed boards are counted apart from decisions.
A speed-c1 width the dump omits (it answers as speed) is read as the speed route.
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


# The sampled widths of the dump (kTokens in test_selector_golden.cpp; a unit test compares them).
KTOKENS = (1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048)
C1_MAX = 16  # speed-c1 is dumped for T <= 16 only; a width that answers as speed is left out


def check_widths(g, name):
    for runs in g.runs.values():
        for lo, hi, _ in runs:
            for t in (lo, hi):
                if t not in KTOKENS:
                    raise GoldenParseError(
                        f"{name}: width {t} is not a sampled width (kTokens changed? update KTOKENS)"
                    )


def decisions(g):
    """{problem key: {T: route}} over the sampled widths, as written in the file."""
    table = {}
    for key, runs in g.runs.items():
        row = {}
        for lo, hi, route in runs:
            for t in KTOKENS:
                if lo <= t <= hi:
                    row[t] = route
        table[key] = row
    return table


def effective_c1(table, key):
    """The speed-c1 row of a problem: a width the dump omits (T <= 16, it answers as speed) is
    the same problem's speed route, also when the file has no speed-c1 key at all."""
    row = dict(table.get(key, {}))
    speed = table.get(key[:4] + ("speed",), {})
    for t in KTOKENS:
        if t <= C1_MAX and t not in row and t in speed:
            row[t] = speed[t]
    return row


def c1_rows(key, d_old, d_new):
    """Old and new effective speed-c1 rows with the widths that are no change of their own
    blanked: unchanged ones, and ones that merely follow a changed speed route (the pair equals
    the speed pair at that width, and is counted there)."""
    so = d_old.get(key[:4] + ("speed",), {})
    sn = d_new.get(key[:4] + ("speed",), {})
    ro, rn = effective_c1(d_old, key), effective_c1(d_new, key)
    for t in KTOKENS:
        if ro.get(t) == rn.get(t) or (ro.get(t), rn.get(t)) == (so.get(t), sn.get(t)):
            ro.pop(t, None)
            rn.pop(t, None)
    return ro, rn


def diff_rows(old, new):
    """Segments (first T, last T, old route or None, new route or None, sampled widths) where
    two {T: route} rows differ; adjacent sampled widths with the same pair are merged."""
    out = []
    last_i = None
    for i, t in enumerate(KTOKENS):
        r_old, r_new = old.get(t), new.get(t)
        if r_old == r_new:
            continue
        if out and out[-1][2] == r_old and out[-1][3] == r_new and last_i == i - 1:
            out[-1] = (out[-1][0], t, r_old, r_new, out[-1][4] + 1)
        else:
            out.append((t, t, r_old, r_new, 1))
        last_i = i
    return out


def diff_golden(old, new):
    """Changes between two parsed files: dict with meta lines and segments
    {problem key: [(lo, hi, old, new, sampled widths)]}."""
    res = {"meta": [], "info": [], "segments": {}}
    if old.header != new.header:
        res["meta"].append(f"header: {old.header} -> {new.header}")
    if old.facts != new.facts:
        res["info"].append(f"facts digest: {old.facts} -> {new.facts}")
    for k in sorted(set(old.not_packed) | set(new.not_packed)):
        a, b = old.not_packed.get(k), new.not_packed.get(k)
        if a != b:
            res["meta"].append(f"not packed {k[0]} {k[1]}x{k[2]}: {a} -> {b}")
    d_old, d_new = decisions(old), decisions(new)
    for key in sorted(set(d_old) | set(d_new)):
        if key[4] == "speed-c1":
            ro, rn = c1_rows(key, d_old, d_new)
        else:
            ro, rn = d_old.get(key, {}), d_new.get(key, {})
        segs = diff_rows(ro, rn)
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


def parse_index(text):
    """facts_index.txt: (release nvcc, rows {set/board: [(facts digest, file digest, nvcc)]})."""
    release, rows = None, defaultdict(list)
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "release" and len(parts) == 2:
            release = parts[1]
        elif len(parts) == 4:
            rows[parts[0]].append((parts[1], parts[2], parts[3]))
        else:
            raise GoldenParseError(f"facts_index.txt:{number}: unknown line: {line!r}")
    return release, rows


def resolve(tree, index_text):
    """{(set, board): (file name, text, note)}: the file holding the release's choices of each
    board. With a facts index that is the file of the row of the release nvcc; a board without
    a row has one file. A board with several files and no row is an error."""
    release, rows = parse_index(index_text) if index_text is not None else (None, {})
    out = {}
    for s, files in tree.items():
        by_board = defaultdict(list)
        for name in files:
            by_board[split_name(name)[0]].append(name)
        for board, names in by_board.items():
            note = ""
            entries = rows.get(f"{s}/{board}", [])
            if entries:
                hit = [e for e in entries if e[2] == release]
                if not hit:
                    raise GoldenParseError(f"facts_index.txt: {s}/{board} has no row of release {release}")
                if len(hit) > 1:
                    raise GoldenParseError(
                        f"facts_index.txt: {s}/{board} has {len(hit)} rows of release {release}"
                    )
                chosen = f"{board}.{hit[0][1]}.txt"
                if chosen not in files:
                    raise GoldenParseError(f"facts_index.txt: {s}/{chosen} is listed but missing")
                note = f"release {release} -> {hit[0][1]}"
            elif len(names) == 1:
                chosen = names[0]
            else:
                raise GoldenParseError(
                    f"{s}/{board}: several files ({', '.join(sorted(names))}) and no facts_index row"
                )
            out[(s, board)] = (chosen, files[chosen], note)
    return out


def parse_checked(text, name):
    g = parse_golden(text, name)
    check_widths(g, name)
    return g


def summarize(old_tree, new_tree, detail, out, old_index=None, new_index=None):
    """Prints the summary; returns (changed decisions, meta changes, board files added/removed)."""
    changed_total = meta_total = files_total = 0
    old_r, new_r = resolve(old_tree, old_index), resolve(new_tree, new_index)
    for sb in sorted(set(old_r) | set(new_r)):
        s, board = sb
        a, b = old_r.get(sb), new_r.get(sb)
        if a is None or b is None:
            g = parse_checked((a or b)[1], f"{s}/{(a or b)[0]}")
            n = sum(len(r) for r in g.runs.values())
            out.append(f"{s}/{board}: {'ADDED' if a is None else 'REMOVED'} board file ({n} runs)")
            files_total += 1
            continue
        label = f"{s}/{board}" + (f" (file {a[0]} -> {b[0]})" if a[0] != b[0] else "")
        res = diff_golden(parse_checked(a[1], f"{s}/{a[0]}"), parse_checked(b[1], f"{s}/{b[0]}"))
        if a[2] != b[2]:
            res["info"].append(f"facts_index: {a[2] or '-'} -> {b[2] or '-'}")
        changed = sum(widths(v) for v in res["segments"].values())
        if not changed and not res["meta"]:
            if res["info"]:
                out.append(f"{label}: no decision changes ({'; '.join(res['info'])})")
            continue
        changed_total += changed
        meta_total += len(res["meta"])
        out.append(f"{label}: {changed} decisions changed in {len(res['segments'])} problems")
        for m in res["meta"] + res["info"]:
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
    return changed_total, meta_total, files_total


def load_dir(root):
    """(tree {set: {file name: text}}, facts_index text or None) from a selector_golden dir."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"golden_diff: not a directory: {root}")
    tree = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        tree[sub.name] = {f.name: f.read_text(encoding="utf-8") for f in sorted(sub.glob("*.txt"))}
    index = root / INDEX_NAME
    return tree, (index.read_text(encoding="utf-8") if index.is_file() else None)


def git(args, repo):
    return subprocess.run(
        ["git", "-C", str(repo)] + args, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def load_rev(rev, top):
    names = git(["ls-tree", "-r", "--name-only", rev, "--", GOLDEN_REL], top).splitlines()
    tree, index = {}, None
    for path in names:
        parts = path[len(GOLDEN_REL) + 1 :].split("/")
        if parts == [INDEX_NAME]:
            index = git(["show", f"{rev}:{path}"], top)
        elif len(parts) == 2 and parts[1].endswith(".txt"):
            tree.setdefault(parts[0], {})[parts[1]] = git(["show", f"{rev}:{path}"], top)
    return tree, index


def detail_arg(text):
    if text == "all":
        return None
    try:
        n = int(text)
    except ValueError:
        n = -1
    if n < 0:
        raise argparse.ArgumentTypeError("a number or 'all'")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", nargs="?", help="base selector_golden directory")
    ap.add_argument("head_dir", nargs="?", help="head selector_golden directory")
    ap.add_argument("--base", help="git revision of the base")
    ap.add_argument("--head", help="git revision of the head")
    ap.add_argument("--range", help="A...B: base = merge-base of A and B, head = B (a branch's own changes)")
    ap.add_argument("--repo", default=".", help="repository for revisions (default: cwd)")
    ap.add_argument("--detail", type=detail_arg, default=20, help="problems listed per board, or 'all' (default 20)")
    ap.add_argument("--fail-on-change", action="store_true", help="exit 1 when anything changed")
    args = ap.parse_args(argv)
    try:
        if args.range:
            if args.base or args.head or args.base_dir or args.head_dir or "..." not in args.range:
                ap.error("--range A...B is used alone")
            a, b = args.range.split("...", 1)
            top = git(["rev-parse", "--show-toplevel"], args.repo).strip()
            args.base, args.head = git(["merge-base", a, b], top).strip(), b
        if args.base or args.head:
            if not (args.base and args.head) or args.base_dir or args.head_dir:
                ap.error("use either BASE_DIR HEAD_DIR, both --base and --head, or --range A...B")
            top = git(["rev-parse", "--show-toplevel"], args.repo).strip()
            (old, old_ix), (new, new_ix) = load_rev(args.base, top), load_rev(args.head, top)
        else:
            if not (args.base_dir and args.head_dir):
                ap.error("give BASE_DIR HEAD_DIR, or --base REV --head REV")
            (old, old_ix), (new, new_ix) = load_dir(args.base_dir), load_dir(args.head_dir)
        out = []
        changed, meta, files = summarize(old, new, args.detail, out, old_ix, new_ix)
    except (GoldenParseError, subprocess.CalledProcessError) as error:
        print(f"golden_diff: {getattr(error, 'stderr', None) or error}", file=sys.stderr)
        return 2
    print("\n".join(out) if out else "no golden changes")
    anything = changed or meta or files
    if anything:
        print(f"total: {changed} decisions changed, {meta} meta changes, {files} board files added/removed")
    return 1 if (args.fail_on_change and anything) else 0


if __name__ == "__main__":
    sys.exit(main())
