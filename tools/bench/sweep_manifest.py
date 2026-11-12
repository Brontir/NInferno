#!/usr/bin/env python3
"""A manifest of the route sweep archive: which route_tune CSVs exist, on which card, and what they hold.

The sweeps the cost-model work is scored on (regret_bands.py, selection_regret.py) live on the lab
machines, not in the repository. This lists them so a replica or a regret run can name its inputs:

  sweep_manifest.py scan ROOT --card rtx4090 > rtx4090.json   # on the machine: one card's files
  sweep_manifest.py merge rtx4090.json rtx3060.json ... > tools/bench/data/sweep_archive.json

`scan` walks ROOT for CSVs whose header is route_tune's `--routes-out` or `--selected-out` (other
CSVs are skipped; build and worktree directories are not walked; a directory that cannot be
listed is recorded with its error) and records, per file: its path
relative to ROOT, size, SHA-256 and modification date; the rows; the formats, compute classes,
problems and widths it covers; its run, device and timing protocol where the file names them (files
from before route_tune wrote those columns say "unnamed"); for a sweep, whether it carries the cost
model's price (`model_us`) and with how many significant digits (six before route_tune printed 17,
too few to tell a tie, see regret_bands.py); a last row cut short (a sweep killed mid-write) is left
out and its line recorded as `cut_last_row`; and the `--selected-out` files of the same directory,
the pairs a regret run needs. A file that cannot be read is listed with its error, not dropped.

`merge` joins the per-card scans into one document sorted by card and path, refusing one card
scanned twice. `summary` renders a merged manifest as markdown, one row per card and directory
(docs/maintainer/sweep-archive.md is one). Nothing here measures, selects or fits.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import re
import sys
from pathlib import Path

SCHEMA = "ninferno-sweep-manifest/1"
KEY = ("format", "compute", "epilogue", "n", "k", "T")
# route_tune's two CSV headers start so (src/ops/generic/route_tune_tool.cpp); the columns after
# these grew over time (run, device, protocol, authority, evidence), so a prefix tells the kind.
ROUTES_PREFIX = KEY + ("route", "median_us")
SELECTED_PREFIX = KEY + ("asked", "selected")
# Build trees (the lab's build_TAG_ARCH, CMake's build, build-*, cmake-build-*) and worktrees hold
# copies or generated files, not archive measurements.
SKIP_DIRS = re.compile(r"^(build$|build[_-]|cmake-build-|wt_|\.git$|__pycache__$)")
CARD = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
UNNAMED = "unnamed"


def kind_of(header):
    columns = tuple(c.strip() for c in header)
    if columns[:len(ROUTES_PREFIX)] == ROUTES_PREFIX:
        return "routes"
    if columns[:len(SELECTED_PREFIX)] == SELECTED_PREFIX:
        return "selected"
    return None


def significant_digits(text):
    """The significant digits of a decimal as written ("12.3456" -> 6, "1.5e-05" -> 2)."""
    mantissa = text.strip().lower().split("e")[0].lstrip("+-").replace(".", "").lstrip("0")
    return len(mantissa)


def named(rows, column, first_word=False):
    """The sorted values a file's rows give `column`, or [UNNAMED] when the file has no such column
    or leaves it empty somewhere. `first_word` keeps the value's first word only: a device is
    recorded by its class id ("ad102:128sm"), not the GPU UUID route_tune may write after it."""
    if not rows or column not in rows[0]:
        return [UNNAMED]
    values = {(r.get(column) or "").strip() for r in rows}
    if first_word:
        values = {v.split()[0] if v else v for v in values}
    return sorted(UNNAMED if v == "" else v for v in values)


def describe(path, root):
    """One file's record, or None when it is no route_tune CSV. The header is read first, so an
    unrelated CSV is not read whole; a UTF-8 byte-order mark before it is allowed."""
    rel = path.relative_to(root).as_posix()
    try:
        with path.open("rb") as f:
            first = f.readline(1 << 16)
            line = first.decode("utf-8-sig", errors="replace")
            try:
                header = next(csv.reader(io.StringIO(line, newline="")), None)
            except csv.Error:  # no CSV header (a bare CR, a NUL): not route_tune's
                return None
            kind = kind_of(header) if header else None
            if kind is None:
                return None
            data = first + f.read()
        modified = path.stat().st_mtime
    except OSError as error:  # unreadable, or removed while the scan ran
        return {"path": rel, "error": str(error)}
    text = data.decode("utf-8-sig", errors="replace")
    record = {
        "path": rel,
        "kind": kind,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "modified": datetime.datetime.fromtimestamp(modified,
                                                    datetime.timezone.utc).strftime("%Y-%m-%d"),
    }

    def short(r):
        return any(r.get(c) is None for c in header) or None in r

    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        lines, rows = [], []
        for r in reader:
            lines.append(reader.line_num)
            rows.append(r)
        # A sweep killed mid-write ends in a cut row: route_tune ends every row with a newline, so
        # a file without one, or whose last row stops short, ends in a cut row. It is left out and
        # said so (`cut_last_row`, its line). A short or long row anywhere else makes the file
        # unreadable.
        if rows and (not data.endswith(b"\n") or (short(rows[-1]) and None not in rows[-1])):
            record["cut_last_row"] = lines.pop()
            rows.pop()
        for line_num, r in zip(lines, rows):
            if short(r):
                raise ValueError(f"line {line_num} has "
                                 f"{'too few' if None not in r else 'too many'} columns")
        widths = sorted({int(r["T"]) for r in rows})
        record.update({
            "rows": len(rows),
            "formats": sorted({r["format"].strip() for r in rows}),
            "computes": sorted({r["compute"].strip() for r in rows}),
            "problems": len({tuple(r[c].strip() for c in KEY[:5]) for r in rows}),
            "t_min": widths[0] if widths else None,
            "t_max": widths[-1] if widths else None,
            "widths": len(widths),
            "runs": named(rows, "run"),
            "devices": named(rows, "device", first_word=True),
        })
        if kind == "routes":
            record["protocols"] = named(rows, "protocol")
            record["routes"] = len({r["route"].strip() for r in rows})
            priced = [r["model_us"] for r in rows if (r.get("model_us") or "").strip()]
            record["model_us_rows"] = len(priced)
            record["model_us_digits"] = max((significant_digits(v) for v in priced), default=0)
        else:
            record["authority_column"] = "authority" in rows[0] if rows else False
    except (KeyError, ValueError, TypeError, AttributeError, csv.Error) as error:  # malformed
        record["error"] = f"unreadable rows: {error}"
    return record


def walk(root, unreadable):
    """The CSV files under root. A directory that cannot be listed (permission denied, gone) is
    appended to `unreadable` as a record with its error, so the manifest says what it did not see."""
    stack = [root]
    while stack:
        here = stack.pop()
        try:
            entries = sorted(here.iterdir())
        except OSError as error:
            unreadable.append({"path": here.relative_to(root).as_posix(), "error": f"unlisted: {error}"})
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if not SKIP_DIRS.match(entry.name):
                    stack.append(entry)
            elif entry.suffix == ".csv":
                yield entry


def scan(root, card):
    if not CARD.match(card):
        raise ValueError(f"card {card!r}: expected a card name such as rtx4090 (lowercase, no spaces)")
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"{root}: not a directory")
    unreadable = []
    files = [r for r in (describe(p, root) for p in walk(root, unreadable)) if r is not None]
    files += unreadable
    files.sort(key=lambda r: r["path"])
    # The selections a sweep pairs with: the --selected-out files of its directory.
    selected = {}
    for r in files:
        # Only a selection that reads: an unreadable one is no pair a regret run can use.
        if r.get("kind") == "selected" and "error" not in r:
            selected.setdefault(str(Path(r["path"]).parent), []).append(r["path"])
    for r in files:
        if r.get("kind") == "routes":
            r["selected_in_dir"] = selected.get(str(Path(r["path"]).parent), [])
    return {"schema": SCHEMA, "card": card, "files": files}


def merge(scans):
    cards = {}
    for name, doc in scans:
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"{name}: not a {SCHEMA} scan")
        card = doc.get("card")
        if not isinstance(card, str) or not CARD.match(card):
            raise ValueError(f"{name}: no card name")
        if card in cards:
            raise ValueError(f"{name}: card {card} is scanned twice")
        cards[card] = doc["files"]
    files = [dict(f, card=c) for c in sorted(cards) for f in cards[c]]
    # Every card scanned has its totals, a card with no route_tune CSV too (all zero).
    totals = {c: {"routes": 0, "selected": 0, "errors": 0, "bytes": 0} for c in cards}
    for f in files:
        t = totals[f["card"]]
        if "error" in f:
            t["errors"] += 1
        if f.get("kind") in ("routes", "selected"):
            t[f["kind"]] += 1
        t["bytes"] += f.get("bytes", 0)
    return {"schema": SCHEMA, "cards": sorted(cards), "totals": totals, "files": files}


def span(values):
    values = sorted(set(values))
    return "" if not values else str(values[0]) if len(values) == 1 else f"{values[0]}..{values[-1]}"


def summary(document):
    """Markdown: per card, one row per directory of its sweeps and selections."""
    if document.get("schema") != SCHEMA or "cards" not in document:
        raise ValueError(f"not a merged {SCHEMA} manifest")
    lines = []
    for card in document["cards"]:
        dirs = {}
        for f in document["files"]:
            if f["card"] == card:
                # A directory that could not be listed is its own row, not its parent's.
                unlisted = f.get("error", "").startswith("unlisted:")
                where = f["path"] if unlisted else Path(f["path"]).parent.as_posix()
                dirs.setdefault(where, []).append(f)
        t = document["totals"][card]
        lines += [f"### {card}", "",
                  f"{t['routes']} sweeps, {t['selected']} selections, {t['bytes'] / 1e6:.1f} MB"
                  + (f", {t['errors']} unreadable" if t["errors"] else "") + ".", "",
                  "| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for d in sorted(dirs):
            fs = dirs[d]
            ok = [f for f in fs if "error" not in f]
            sweeps = [f for f in ok if f["kind"] == "routes"]
            protocols = sorted({p for f in sweeps for p in f["protocols"]} - {UNNAMED})
            digits = sorted({f["model_us_digits"] for f in sweeps if f["model_us_rows"]})
            named_files = sum(1 for f in ok if UNNAMED not in f["runs"] + f["devices"])
            t_lo = [f["t_min"] for f in ok if f["t_min"] is not None]
            t_hi = [f["t_max"] for f in ok if f["t_max"] is not None]
            cells = [
                d, str(len(sweeps)), str(sum(1 for f in ok if f["kind"] == "selected")),
                str(sum(f["rows"] for f in ok)),
                " ".join(sorted({x for f in ok for x in f["formats"]})),
                f"{min(t_lo)}..{max(t_hi)}" if t_lo else "",
                " ".join(protocols) or "unnamed",
                span(digits) or "none",
                f"{named_files}/{len(ok)}",
                span(f["modified"] for f in fs if "modified" in f),
            ]
            notes = [f"{len(fs) - len(ok)} unreadable"] if len(ok) != len(fs) else []
            cut = sum(1 for f in ok if "cut_last_row" in f)
            notes += [f"{cut} cut"] if cut else []
            if notes:
                cells[0] += f" ({', '.join(notes)})"
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="one card's route_tune CSVs under a directory")
    s.add_argument("root", type=Path)
    s.add_argument("--card", required=True, help="the card the files were measured on, e.g. rtx4090")
    m = sub.add_parser("merge", help="join per-card scans into the archive manifest")
    m.add_argument("scans", type=Path, nargs="+")
    r = sub.add_parser("summary", help="a merged manifest as markdown, per card and directory")
    r.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "summary":
            print(summary(json.loads(args.manifest.read_text(encoding="utf-8"))))
            return 0
        if args.command == "scan":
            document = scan(args.root, args.card)
        else:
            document = merge([(str(p), json.loads(p.read_text(encoding="utf-8")))
                              for p in args.scans])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sweep_manifest: {error}", file=sys.stderr)
        return 2
    print(json.dumps(document, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
