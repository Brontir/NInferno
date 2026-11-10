#!/usr/bin/env python3
"""Forced-width grid and A/B helpers for the MMQ column-tile work table (DECISIONS 74, #524).

Used by tools/bench/run_524_turing_grid.sh with a bench-only hooked build (never merged into the
kernel, see the script's header).

measure  one running ninferno-serve: for each prompt size N, `--reps` non-streaming chat requests
         with `enable_thinking` false and max_tokens 1, each of exactly N prompt tokens. The metric is
         the server's own prefill time, `timings.prompt_ms` (src/serve/openai_chat_response.cpp),
         and the size the server counted, `usage.prompt_tokens` = `timings.prompt_n` +
         `timings.cache_n`; a request whose count is not N, or that hit the prefix cache, is
         rejected. Writes one JSON line {"label", "n", "ms": [...], "prompt_n": [...]} per size to
         --out, which must not exist or be empty (one file per visit: stale rows never mix in).
grid     reads the per-width files of `measure` (label "J<w>") and prints the regret of the model's
         picks against the best width for each f of --fs (0.43 = DECISIONS 69, "old" = fewest tiles).
ab       reads files labelled old/new and prints the per-size change (medians with the range of the
         visits) and the mix's.

Both readers refuse empty files, malformed lines, rows whose prompt_n differs from n, and a cell
measured twice in `grid` (stale rows), and print the number of requests behind every label.

The regret of a pick = (time at the picked width - least time over all widths measured) / least,
averaged over the sizes; the pick is the production rule (core/mmq_columns.h pick: least
ceil(N / J) x (f + (1 - f) J / 128) among 16 32 64 80 96 128, the narrower on a tie).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

SIZES = [16, 32, 48, 64, 80, 96, 112, 128, 136, 144, 160, 176, 192, 208, 224, 256, 288, 320, 384, 448, 512]
AB_SIZES = [96, 128, 136, 144, 160, 176, 192, 208, 224, 240, 256, 264, 288, 320, 384, 448, 512]
MIX = [144, 160, 176, 192, 208, 224, 240, 256]
WIDTHS = [16, 32, 48, 64, 80, 96, 112, 128]
PRODUCTION = [16, 32, 64, 80, 96, 128]
OLD = [16, 32, 64, 128]
FS = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.43]
SALT_DIGITS = 6  # a fixed-width salt: every request spends the same number of tokens on it


class Fail(Exception):
    """A readable error: main() prints it and exits 1, no traceback."""


def pick(n: int, f: float | None, widths: list[int]) -> int:
    best, least = 0, 1e300
    for j in widths:
        t = -(-n // j)
        cost = t if f is None else t * (f + (1 - f) * j / 128)
        if cost < least - 1e-12:
            best, least = j, cost
    return best


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:400]
        raise Fail(
            f"the server answered HTTP {e.code}: {text}\n"
            "  a forced width without a tile configuration for a weight format fails with 'ggml mmq: no "
            "tile fits this device'; all eight widths exist only for Q4_K and Q6_K, so use a Q4_K_M "
            "model (Qwen3-8B Q4_K_M), and see the server log"
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise Fail(f"cannot reach the server at {url}: {e}") from e


def request(base: str, model: str, words: int, salt: int) -> dict:
    """One chat completion of `words` words behind a fixed-width salt word, thinking off."""
    # a distinct salt per request: no prefix is shared (the server runs --no-prefix-reuse too)
    content = f"w{salt % 10**SALT_DIGITS:0{SALT_DIGITS}d} " + "a " * words
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1,
            "temperature": 0, "enable_thinking": False}
    t0 = time.perf_counter()
    r = post(base + "/v1/chat/completions", body)
    r["_wall_ms"] = (time.perf_counter() - t0) * 1e3
    return r


def counted(r: dict) -> int:
    """The prompt size the server counted: usage.prompt_tokens, checked against the timings."""
    n = r["usage"]["prompt_tokens"]
    t = r.get("timings") or {}
    if "prompt_ms" not in t or "prompt_n" not in t:
        raise Fail("the response has no timings.prompt_ms / prompt_n: is this ninferno-serve?")
    if t.get("cache_n", 0) != 0 or t["prompt_n"] != n:
        raise Fail(f"the prefix cache took part (cache_n {t.get('cache_n')}, prompt_n {t['prompt_n']}, "
                   f"prompt_tokens {n}): start the server with --no-prefix-reuse")
    return n


def words_for(base: str, model: str, n: int, salt: int) -> int:
    """Words that make the salted prompt exactly n tokens (calibrated on the salted prompt)."""
    words = max(1, n - 8)
    for _ in range(6):
        got = counted(request(base, model, words, salt))
        if got == n:
            return words
        words += n - got
        if words < 0:
            break
    raise Fail(f"cannot build a prompt of exactly {n} tokens (the chat template's own tokens leave no room?)")


def measure(a: argparse.Namespace) -> int:
    out = Path(a.out)
    if out.exists() and out.stat().st_size > 0:
        raise Fail(f"{out} is not empty: one file per visit, stale rows would be pooled; "
                   "remove it or use another --out")
    base = f"http://{a.host}:{a.port}"
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
            model = json.load(r)["data"][0]["id"]
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
        raise Fail(f"cannot read {base}/v1/models: {e}") from e
    sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else SIZES
    salt = int(time.time()) % 1000 * 1000
    for rep in range(a.warmup):
        request(base, model, 120, salt + rep)
    rejected = 0
    rows = 0
    for n in sizes:
        words = words_for(base, model, n, salt)
        ms, ns, wall = [], [], []
        for i in range(a.reps):
            r = request(base, model, words, salt + 10 * n + i + 1)
            got = counted(r)
            if got != n:
                rejected += 1
                print(f"{a.label} N={n}: request {i} had {got} prompt tokens, rejected", file=sys.stderr)
                continue
            ms.append(r["timings"]["prompt_ms"])
            ns.append(got)
            wall.append(round(r["_wall_ms"], 3))
        if len(ms) < a.reps:
            print(f"{a.label} N={n}: only {len(ms)} of {a.reps} requests kept, size not written", file=sys.stderr)
            continue
        with out.open("a") as f:
            f.write(json.dumps({"label": a.label, "n": n, "ms": ms, "prompt_n": ns, "wall_ms": wall}) + "\n")
        rows += 1
        print(f"{a.label} N={n} prompt_n {n} median {statistics.median(ms):.2f} ms", flush=True)
    print(f"{a.label}: {rows} of {len(sizes)} sizes written to {out}, {rejected} requests rejected")
    return 0 if rows == len(sizes) else 1


def load(paths: list[str]) -> dict[str, dict[int, list[list[float]]]]:
    """label -> n -> one list of per-request ms for each visit (row), in file order."""
    d: dict[str, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise Fail(f"{p}: no such file")
        lines = [(i, s) for i, s in enumerate(path.read_text().splitlines(), 1) if s.strip()]
        if not lines:
            raise Fail(f"{p}: empty (the measurement wrote no rows)")
        for i, line in lines:
            try:
                r = json.loads(line)
                label, n, ms = r["label"], int(r["n"]), [float(x) for x in r["ms"]]
                pn = r.get("prompt_n")
            except (ValueError, KeyError, TypeError) as e:
                raise Fail(f"{p}:{i}: not a measurement row ({e.__class__.__name__}: {e}); a partial file?") from e
            if not ms:
                raise Fail(f"{p}:{i}: no timings in the row")
            if pn is None:
                raise Fail(f"{p}:{i}: no prompt_n: rows of an older kit measured unverified prompt sizes; "
                           "measure again")
            if len(pn) != len(ms) or any(x != n for x in pn):
                raise Fail(f"{p}:{i}: label {label} N={n} has prompt_n {sorted(set(pn))}: "
                           "the request was not N tokens")
            d[label][n].append(ms)
    return d


def cell_counts(d, label: str) -> str:
    cells = d[label]
    return (f"{label}: {len(cells)} sizes, {sum(len(v) for v in cells.values())} visits, "
            f"{sum(len(r) for v in cells.values() for r in v)} requests")


def flat(rows: list[list[float]]) -> list[float]:
    return [x for r in rows for x in r]


def grid(a: argparse.Namespace) -> None:
    d = load(a.files)
    labels = [f"J{j}" for j in WIDTHS if f"J{j}" in d]
    if not labels:
        raise Fail("no J<width> labels in the files")
    unknown = sorted(set(d) - set(labels))
    if unknown:
        raise Fail(f"labels {unknown} are not forced widths: the files of the A/B? (use `ab`)")
    for lab in labels:
        print(cell_counts(d, lab))
        twice = sorted(n for n, v in d[lab].items() if len(v) > 1)
        if twice:
            raise Fail(f"{lab}: sizes {twice} were measured more than once: stale rows from an earlier run; "
                       "the grid takes one visit per width")
    med = {int(lab[1:]): {n: statistics.median(flat(d[lab][n])) for n in d[lab]} for lab in labels}
    sizes = sorted(set.intersection(*[set(m) for m in med.values()]))
    if not sizes:
        raise Fail("the widths share no prompt size")
    missing = sorted({n for m in med.values() for n in m} - set(sizes))
    if missing:
        print(f"warning: sizes {missing} are missing at some width and dropped")
    best = {n: min(m[n] for m in med.values()) for n in sizes}
    print("N  best_width  " + " ".join(f"J{j:<5}" for j in med))
    for n in sizes:
        row = " ".join(f"{med[j][n]:6.2f}" for j in med)
        print(f"{n:3d} J{min(med, key=lambda j: med[j][n]):<3d}  {row}")

    def regret(p) -> float:
        usable = [n for n in sizes if p(n) in med]
        if not usable:
            raise Fail("none of the picked widths was measured")
        return 100 * statistics.mean((med[p(n)][n] - best[n]) / best[n] for n in usable)

    print("\nregret of the picks against the best width (mean over sizes, %):")
    print(f"  fewest tiles (16 32 64 128): {regret(lambda n: pick(n, None, OLD)):.2f}")
    res = []
    for f in a.fs:
        r = regret(lambda n, f=f: pick(n, f, PRODUCTION))
        res.append((r, f))
        print(f"  f {f:.2f}: {r:.2f}")
    least = min(r for r, _ in res)
    ok = [f for r, f in res if r <= least + 0.05]
    print(f"\nleast {least:.2f} % at f {', '.join(f'{f:.2f}' for f in ok)} (within 0.05 of the least)")


def ab(a: argparse.Namespace) -> None:
    d = load(a.files)
    for lab in ("old", "new"):
        if lab not in d:
            raise Fail(f"no rows labelled {lab}")
    unknown = sorted(set(d) - {"old", "new"})
    if unknown:
        raise Fail(f"labels {unknown} are not old/new: the files of the grid? (use `grid`)")
    old, new = d["old"], d["new"]
    print(cell_counts(d, "old"))
    print(cell_counts(d, "new"))
    sizes = sorted(set(old) & set(new))
    gone = sorted(set(old) ^ set(new))
    if not sizes:
        raise Fail("old and new share no prompt size")
    if gone:
        print(f"warning: sizes {gone} are in only one of old / new and dropped")
    for n in sizes:
        if len(old[n]) != len(new[n]):
            raise Fail(f"N={n}: {len(old[n])} old visits against {len(new[n])} new: an interrupted or stale run")

    def spread(rows: list[list[float]]) -> str:
        m = [statistics.median(r) for r in rows]
        return f"{statistics.median(flat(rows)):7.2f} [{min(m):.2f}-{max(m):.2f}]"

    print("size  old median [range of the visits' medians]  new median [range]  change")
    for n in sizes:
        o, w = statistics.median(flat(old[n])), statistics.median(flat(new[n]))
        print(f"N={n:3d} old {spread(old[n])}  new {spread(new[n])} ms {100 * (w / o - 1):+6.1f} %")
    mix = [n for n in MIX if n in sizes]
    if not mix:
        raise Fail(f"none of the mix sizes {MIX} was measured")
    visits = len(old[mix[0]])
    per = []
    for v in range(visits):
        mo = statistics.mean(statistics.median(old[n][v]) for n in mix)
        mw = statistics.mean(statistics.median(new[n][v]) for n in mix)
        per.append(100 * (mw / mo - 1))
    mo = statistics.mean(statistics.median(flat(old[n])) for n in mix)
    mw = statistics.mean(statistics.median(flat(new[n])) for n in mix)
    print(f"mix {mix}: old {mo:.2f} new {mw:.2f} ms {100 * (mw / mo - 1):+.1f} %  "
          f"(per visit {min(per):+.1f} .. {max(per):+.1f} %, {visits} visits)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    m = sub.add_parser("measure")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=18524)
    m.add_argument("--label", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--sizes", help="comma list (default: the 21 grid sizes)")
    m.add_argument("--reps", type=int, default=4)
    m.add_argument("--warmup", type=int, default=3)
    m.set_defaults(fn=measure)
    g = sub.add_parser("grid")
    g.add_argument("files", nargs="+")
    g.add_argument("--fs", type=float, nargs="+", default=FS)
    g.set_defaults(fn=grid)
    b = sub.add_parser("ab")
    b.add_argument("files", nargs="+")
    b.set_defaults(fn=ab)
    a = ap.parse_args()
    try:
        return a.fn(a) or 0
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
