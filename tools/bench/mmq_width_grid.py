#!/usr/bin/env python3
"""Forced-width grid and A/B helpers for the MMQ column-tile work table (DECISIONS 74, #524).

Used by tools/bench/run_524_turing_grid.sh with a bench-only hooked build (never merged into the
kernel, see the script's header).

SIZES. Every size N is a PROMPT TOKEN count. The kernel sees columns = tokens - k: the prefill is split
at execution boundaries, so an N-token prompt runs a body call of N - k columns plus small tail calls
(src/models/qwen3_5/frontend/prompt_layout.cpp). k is a parameter (--k, default 7, an ASSUMPTION that
nobody has measured; `kprobe` measures it). Picks, tile counts, predictions and the printed
"columns" all use N - k; k is written into every row and the readers refuse files with different k.
The older grids' N were columns (DECISIONS 74): compare their rows with this kit's columns, not tokens.
`prompt_ms` also holds a width-independent part (tail calls, the first token's head and sampling, host
commit), which dilutes every percentage below.

measure  one running ninferno-serve: for each prompt size N, `--reps` non-streaming chat requests
         with `enable_thinking` false and max_tokens 1, each of exactly N prompt tokens. The metric is
         the server's own prefill time, `timings.prompt_ms` (src/serve/openai_chat_response.cpp),
         and the size the server counted, `usage.prompt_tokens` = `timings.prompt_n` +
         `timings.cache_n`; a request whose count is not N, or that hit the prefix cache, is
         rejected. A size that cannot be built (smaller than the template + salt, or not converging) is
         skipped, reported, and recorded as a "skipped" row. Writes one JSON line {"label", "n", "k",
         "ms": [...], "prompt_n": [...]} per size to --out, which must not exist or be empty (one file
         per visit: stale rows never mix in). `--sizes ab` / `mix` take the A/B / mix size lists.
grid     reads the per-width files of `measure` (label "J<w>"; all eight widths, the same sizes at
         every width, else it refuses) and prints, with the assumption f 0.20 beside the measurement:
         the f buckets (one table of picks each, with the sizes that decide each edge), the regret per
         bucket and at f 0.20, the least, the difference, the spread, the per-size predicted width
         against the measured best, the mismatch count and the verdict of the criteria that were
         fixed before the run (CRITERIA below).
ab       reads files labelled old/new (the same sizes, the same number of visits at every size, the
         whole mix) and prints the per-size change (medians with the range of the visits) beside the
         predicted range, the mismatch count, and the mix change beside its predicted range.
kprobe   reads one `measure` file made with the width forced to 128 and consecutive sizes (130..146)
         and prints the interval of k that the biggest step in time implies (columns cross 128 there).

Both readers refuse empty files, malformed lines, rows whose prompt_n differs from n, rows without k,
and a cell measured twice in `grid` (stale rows), and print the number of requests behind every label.

The regret of a pick = (time at the picked width - least time over all widths measured) / least,
averaged over the sizes; the pick is the production rule (core/mmq_columns.h pick: least
ceil(c / J) x (f + (1 - f) J / 128) among 16 32 64 80 96 128, the narrower on a tie), c = N - k.
Nothing is tuned to the data: every predicted value is computed from the rule or taken from the
constants above before the data is read, and is printed beside the measured one. The only constants
are the thresholds of CRITERIA, TOL and the group edges of PRED_AB, all set a priori and labelled so.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

K_ASSUMED = 7  # ASSUMED, not measured: tokens - columns of the body call (see "columns" below); `kprobe` measures it
# Grid and A/B sizes are PROMPT TOKENS; the kernel sees columns = tokens - k. Sizes were moved off the
# zero-margin spots (136 -> 142, 264 -> 270): every size keeps at least `MARGIN_WARN` columns of
# distance from a tile edge of the widths 64 80 96 128 for k = 7 (`margins`), so a k that is a few off
# does not change a tile count.
SIZES = [24, 32, 48, 64, 80, 96, 112, 128, 142, 144, 160, 176, 192, 208, 224, 256, 288, 320, 384, 448, 512]
AB_SIZES = [96, 128, 142, 144, 160, 176, 192, 208, 224, 240, 256, 270, 288, 320, 384, 448, 512]
MIX = [144, 160, 176, 192, 208, 224, 240, 256]
WIDTHS = [16, 32, 48, 64, 80, 96, 112, 128]
PRODUCTION = [16, 32, 64, 80, 96, 128]
OLD = [16, 32, 64, 128]
EDGE_WIDTHS = [64, 80, 96, 128]
MARGIN_WARN = 4
ASSUMED_F = 0.20  # the assumption under test (an assumption, not a derivation)
F_ONE = 0.43  # DECISIONS 69, shown as the representative of the top bucket when it lies in it
SALT_DIGITS = 6  # a fixed-width salt: every request spends the same number of tokens on it
TOL = 2.0  # percent: a predicted width slower than the best by more than this is a material mismatch
# Fixed BEFORE any data exists (not fitted); thresholds are printed with the verdicts.
R1_PP = 0.5  # percentage points of regret that the assumed f may lose to the least bucket
R2_SHARE = 0.25  # share of sizes with a material mismatch at the assumed f
R4_PP = 2.0  # percentage points the A/B mix may lie outside the predicted mix range
# GA10x / Ada bracket (DECISIONS 74), by columns; the edges 168 / 200 are the midpoints between the
# sample sizes 160|176 and 192|208 of the old rows, not fitted: (first col, last col, lo %, hi %)
PRED_AB = [(129, 168, -19.0, -14.0), (169, 200, -13.0, -9.0), (201, 260, -0.5, 0.5)]
CRITERIA = f"""REFUTATION CRITERIA for f = {ASSUMED_F:.2f} (fixed before the run; the thresholds are not fitted):
  R1 grid: the regret at f {ASSUMED_F:.2f} exceeds the least regret over the f buckets by more than {R1_PP} pp.
  R2 grid: more than {int(R2_SHARE * 100)} % of the sizes have a material mismatch: the width picked at f {ASSUMED_F:.2f} is more
     than {TOL:g} % slower than the best measured width.
  R3 grid: the regret at f {ASSUMED_F:.2f} is not below the regret of the fewest tiles (the table gives no gain).
  R4 ab (f {ASSUMED_F:.2f}): the mix change lies more than {R4_PP:g} pp outside the predicted mix range.
Any one met refutes f = {ASSUMED_F:.2f} for this card; none met leaves it standing (it is not proved by one visit)."""


class Fail(Exception):
    """A readable error: main() prints it and exits 1, no traceback."""


class Unbuildable(Fail):
    """A prompt size that cannot be built: measure() skips it and reports it."""


def margin(c: int, j: int) -> int:
    """How many columns the count may be off, either way, before ceil(c / j) changes."""
    r = (c - 1) % j + 1
    return min(r - 1, j - r)


def margins(n: int, k: int) -> int:
    return min(margin(n - k, j) for j in EDGE_WIDTHS)


def pick(n: int, f: float | None, widths: list[int]) -> int:
    best, least = 0, 1e300
    for j in widths:
        t = -(-n // j)
        cost = t if f is None else t * (f + (1 - f) * j / 128)
        if cost < least - 1e-12:
            best, least = j, cost
    return best


def work(c: int, j: int, f: float) -> float:
    return -(-c // j) * (f + (1 - f) * j / 128)


def pick_table(cols: list[int], f: float | None, widths: list[int] = PRODUCTION) -> tuple[int, ...]:
    return tuple(pick(c, f, widths) for c in cols)


def crossings(cols: list[int], widths: list[int]) -> list[float]:
    """The f values in (0, 1) where two widths cost the same at some column count: all places where
    the table of picks can change (from t1 (a1 + f (1 - a1)) = t2 (a2 + f (1 - a2)), a = J / 128)."""
    bps = set()
    for c in cols:
        for i, j1 in enumerate(widths):
            for j2 in widths[i + 1:]:
                t1, t2, a1, a2 = -(-c // j1), -(-c // j2), j1 / 128, j2 / 128
                den = t1 * (1 - a1) - t2 * (1 - a2)
                if den == 0:
                    continue
                f = (t2 * a2 - t1 * a1) / den
                if 0 < f < 1:
                    bps.add(round(f, 9))
    return sorted(bps)


def buckets(cols: list[int], assumed: float) -> list[dict]:
    """The f intervals with one table of picks each, one representative f per bucket (the assumed f for
    the bucket it falls in), and the column counts whose pick changes at the lower edge. The first
    bucket is the one below the lowest breakpoint. A tie goes to the narrower width, so an edge belongs
    to the bucket below it."""
    edges = [0.0] + crossings(cols, PRODUCTION) + [1.0]
    merged: list[list] = []
    for lo, hi in zip(edges, edges[1:]):
        if hi - lo < 1e-9:
            continue
        t = pick_table(cols, (lo + hi) / 2)
        if merged and merged[-1][2] == t:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi, t])
    ta = pick_table(cols, assumed)
    out = []
    for i, (lo, hi, t) in enumerate(merged):
        mine = t == ta
        rep = assumed if mine else (F_ONE if lo < F_ONE <= hi else (lo + hi) / 2)
        deciding = [] if i == 0 else [x for x in range(len(cols)) if t[x] != merged[i - 1][2][x]]
        out.append({"lo": lo, "hi": hi, "rep": rep, "table": t, "assumed": mine, "deciding": deciding})
    if not any(b["assumed"] for b in out):
        raise Fail(f"internal: the table at f {assumed} is in no bucket")
    return out


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
    t = r.get("timings") or {}
    if "prompt_ms" not in t or "prompt_n" not in t or "prompt_tokens" not in (r.get("usage") or {}):
        raise Fail("the response has no usage.prompt_tokens / timings.prompt_ms / prompt_n: is this ninferno-serve?")
    n = r["usage"]["prompt_tokens"]
    if t.get("cache_n", 0) != 0 or t["prompt_n"] != n:
        raise Fail(f"the prefix cache took part (cache_n {t.get('cache_n')}, prompt_n {t['prompt_n']}, "
                   f"prompt_tokens {n}): start the server with --no-prefix-reuse")
    if not t["prompt_ms"] > 0:
        raise Fail(f"timings.prompt_ms is {t['prompt_ms']}: the server reports no first token")
    return n


def words_for(base: str, model: str, n: int, salts) -> int:
    """Words that make the salted prompt exactly n tokens (calibrated on the salted prompt: the
    template, the salt word and the trailing space are measured with an empty body first)."""
    floor = counted(request(base, model, 0, next(salts)))
    words = n - floor
    if words < 0:
        raise Unbuildable(f"cannot build a prompt of {n} tokens: the template, the salt word and the "
                          f"trailing space alone are {floor}")
    for _ in range(6):
        got = counted(request(base, model, words, next(salts)))
        if got == n:
            return words
        words += n - got
        if words < 0:
            break
    raise Unbuildable(f"cannot build a prompt of exactly {n} tokens: the count does not converge "
                      "(a word is not one token here?)")


def sizes_of(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return default
    named = {"grid": SIZES, "ab": AB_SIZES, "mix": MIX}
    return named[text] if text in named else [int(s) for s in text.split(",")]


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
    sizes = sizes_of(a.sizes, SIZES)
    # one salt counter for every request of the visit: no two requests share a salt
    salts = itertools.count(int(time.time() * 1000) % 10**SALT_DIGITS)
    for _ in range(a.warmup):
        request(base, model, 120, next(salts))
    meta = {"k": a.k}
    if a.f is not None:
        meta["f"] = a.f
    rejected = rows = 0
    skipped: list[int] = []
    for n in sizes:
        try:
            words = words_for(base, model, n, salts)
        except Unbuildable as e:
            skipped.append(n)
            print(f"{a.label} N={n}: skipped: {e}", file=sys.stderr)
            with out.open("a") as f:
                f.write(json.dumps({"label": a.label, "n": n, "skipped": str(e), **meta}) + "\n")
            continue
        ms, ns, wall = [], [], []
        for i in range(a.reps):
            r = request(base, model, words, next(salts))
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
            f.write(json.dumps({"label": a.label, "n": n, "ms": ms, "prompt_n": ns, "wall_ms": wall, **meta}) + "\n")
        rows += 1
        print(f"{a.label} N={n} (columns {n - a.k}) prompt_n {n} median {statistics.median(ms):.2f} ms", flush=True)
    print(f"{a.label}: {rows} of {len(sizes)} sizes written to {out}, {len(skipped)} skipped {skipped}, "
          f"{rejected} requests rejected")
    return 0 if rows and rows + len(skipped) == len(sizes) else 1


class Data:
    """What the readers take from the files: label -> n -> one list of per-request ms per visit (row)."""

    def __init__(self) -> None:
        self.d: dict[str, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
        self.skipped: dict[str, dict[int, str]] = defaultdict(dict)
        self.ks: set[int] = set()
        self.fs: dict[str, set[float]] = defaultdict(set)

    def k(self) -> int:
        if len(self.ks) != 1:
            raise Fail(f"the files were measured with different k {sorted(self.ks)}: one k per campaign")
        return next(iter(self.ks))


def load(paths: list[str]) -> Data:
    data = Data()
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
                label, n = r["label"], int(r["n"])
                k = r.get("k")
                if k is None:
                    raise KeyError("k")
                data.ks.add(int(k))
                if "f" in r:
                    data.fs[label].add(float(r["f"]))
                if "skipped" in r:
                    data.d[label]  # the label exists even if no size was measured
                    data.skipped[label][n] = str(r["skipped"])
                    continue
                ms = [float(x) for x in r["ms"]]
                pn = r.get("prompt_n")
            except (ValueError, KeyError, TypeError) as e:
                raise Fail(f"{p}:{i}: not a measurement row ({e.__class__.__name__}: {e}); a partial file or "
                           "a file of an older kit (no k)?") from e
            if not ms:
                raise Fail(f"{p}:{i}: no timings in the row")
            if pn is None:
                raise Fail(f"{p}:{i}: no prompt_n: rows of an older kit measured unverified prompt sizes; "
                           "measure again")
            if len(pn) != len(ms) or any(x != n for x in pn):
                raise Fail(f"{p}:{i}: label {label} N={n} has prompt_n {sorted(set(pn))}: "
                           "the request was not N tokens")
            data.d[label][n].append(ms)
    return data


def cell_counts(d, label: str) -> str:
    cells = d[label]
    return (f"{label}: {len(cells)} sizes, {sum(len(v) for v in cells.values())} visits, "
            f"{sum(len(r) for v in cells.values() for r in v)} requests")


def flat(rows: list[list[float]]) -> list[float]:
    return [x for r in rows for x in r]


def same_sizes(data: Data, labels: list[str]) -> list[int]:
    """The sizes, identical at every label; anything else is an incomplete campaign and is refused."""
    sets = {lab: set(data.d[lab]) for lab in labels}
    empty = [lab for lab in labels if not sets[lab]]
    if empty:
        raise Fail(f"{empty} measured no size at all (skipped: {dict(data.skipped.get(empty[0], {}))})")
    union = set.union(*sets.values())
    lacking = {lab: sorted(union - s) for lab, s in sets.items() if union - s}
    if lacking:
        raise Fail(f"incomplete campaign, sizes missing per label: {lacking}; scoring the rest would "
                   "compare different workloads. Measure the missing visits again (skipped sizes: "
                   f"{ {lab: sorted(v) for lab, v in data.skipped.items() if v} })")
    return sorted(union)


def grid(a: argparse.Namespace) -> None:
    data = load(a.files)
    d = data.d
    k = data.k()
    labels = [f"J{j}" for j in WIDTHS if f"J{j}" in d]
    unknown = sorted(set(d) - set(labels))
    if unknown:
        raise Fail(f"labels {unknown} are not forced widths: the files of the A/B? (use `ab`)")
    absent = [f"J{j}" for j in WIDTHS if f"J{j}" not in d]
    if absent:
        raise Fail(f"no rows for {absent}: all eight forced widths are needed before any regret is scored "
                   "(a pick may land on any of them)")
    print(f"ASSUMPTION under test: f = {ASSUMED_F:.2f} (not derived). columns = tokens - k, k = {k} "
          f"({'the default ASSUMED value, not measured' if k == K_ASSUMED else 'as given'}).")
    print(CRITERIA)
    print()
    for lab in labels:
        print(cell_counts(d, lab))
        twice = sorted(n for n, v in d[lab].items() if len(v) > 1)
        if twice:
            raise Fail(f"{lab}: sizes {twice} were measured more than once: stale rows from an earlier run; "
                       "the grid takes one visit per width")
    skipped = sorted({n for v in data.skipped.values() for n in v})
    if skipped:
        print(f"skipped sizes (could not be built): {skipped}")
    sizes = same_sizes(data, labels)
    cols = [n - k for n in sizes]
    if min(cols) < 1:
        raise Fail(f"size {sizes[cols.index(min(cols))]} has no column with k {k}")
    med = {j: {n: statistics.median(flat(d[f"J{j}"][n])) for n in sizes} for j in WIDTHS}
    best = {n: min(m[n] for m in med.values()) for n in sizes}
    low = [(n, margins(n, k)) for n in sizes if margins(n, k) < MARGIN_WARN]
    if low:
        print(f"warning: sizes within {MARGIN_WARN} columns of a tile edge at k {k} (a k that far off "
              f"changes a tile count): {low}")
    cells = [(max(v) - min(v)) / statistics.median(v) for lab in labels for rows in d[lab].values()
             for v in rows if len(v) > 1]
    noise = 100 * statistics.median(cells) if cells else float("nan")

    bk = buckets(cols, ASSUMED_F)
    ta = pick_table(cols, ASSUMED_F)

    def regret(table: tuple[int, ...]) -> float:
        return 100 * statistics.mean((med[j][n] - best[n]) / best[n] for j, n in zip(table, sizes))

    print("\nper size: the prediction at f %.2f beside the measurement (ms = median of the visit)" % ASSUMED_F)
    print("  N  cols  predicted  best   ms@pred  ms@best  loss   " + " ".join(f"J{j:<5}" for j in WIDTHS))
    strict = material = 0
    for n, c, jp in zip(sizes, cols, ta):
        jb = min(WIDTHS, key=lambda j: med[j][n])
        loss = 100 * (med[jp][n] / best[n] - 1)
        strict += med[jp][n] > best[n]
        material += loss > TOL
        flag = "  MISMATCH" if loss > TOL else ("  differs" if med[jp][n] > best[n] else "")
        print(f"{n:4d} {c:5d}  J{jp:<8d} J{jb:<4d} {med[jp][n]:8.2f} {best[n]:8.2f} {loss:5.1f}%  "
              + " ".join(f"{med[j][n]:6.2f}" for j in WIDTHS) + flag)
    print(f"width picked at f {ASSUMED_F:.2f} differs from the measured best in {strict} of {len(sizes)} sizes "
          f"(slower at all); more than {TOL:g} % slower (material mismatch): {material}")

    print("\nf buckets (one table of picks each; an edge belongs to the bucket below it) and regret:")
    res = []
    for b in bk:
        r = regret(b["table"])
        res.append(r)
        dec = ", ".join(f"{sizes[x]}t/{cols[x]}c:J{bk[bk.index(b) - 1]['table'][x]}->J{b['table'][x]}"
                        for x in b["deciding"]) or "-"
        print(f"  f ({b['lo']:.3f}, {b['hi']:.3f}]  rep {b['rep']:.3f}  regret {r:6.2f} %  "
              f"{'<- ASSUMED f ' + format(ASSUMED_F, '.2f') if b['assumed'] else ''}  lower edge decided by: {dec}")
    r_old = regret(pick_table(cols, None, OLD))
    r_a = next(r for r, b in zip(res, bk) if b["assumed"])
    least = min(res)
    at_least = [f"({b['lo']:.3f}, {b['hi']:.3f}]" for r, b in zip(res, bk) if r <= least + 1e-9]
    spread = max(res) - least
    print(f"  fewest tiles (16 32 64 128): regret {r_old:.2f} %")
    print(f"\nregret at the assumed f {ASSUMED_F:.2f}: {r_a:.2f} %   least: {least:.2f} % in {', '.join(at_least)}   "
          f"difference {r_a - least:.2f} pp   spread over the buckets {spread:.2f} pp   "
          f"noise (median range of one cell's requests) {noise:.1f} %")
    print("\nverdict on the criteria fixed before the run:")
    r1, r2, r3 = r_a - least > R1_PP, material > R2_SHARE * len(sizes), r_a >= r_old
    for name, hit, text in (
            ("R1", r1, f"difference {r_a - least:.2f} pp against {R1_PP} pp"),
            ("R2", r2, f"{material} of {len(sizes)} sizes against {R2_SHARE * len(sizes):.1f}"),
            ("R3", r3, f"regret {r_a:.2f} % at f {ASSUMED_F:.2f} against {r_old:.2f} % fewest tiles")):
        print(f"  {name}: {'MET (refutes f %.2f)' % ASSUMED_F if hit else 'not met'}  ({text})")
    print("  " + ("f %.2f is REFUTED for this card by the criteria above" % ASSUMED_F if r1 or r2 or r3
                  else "f %.2f stands (one visit of four requests a cell; not a proof)" % ASSUMED_F))


def pred_range(c: int) -> tuple[float, float] | None:
    for lo_c, hi_c, lo, hi in PRED_AB:
        if lo_c <= c <= hi_c:
            return lo, hi
    return None


def ab(a: argparse.Namespace) -> None:
    data = load(a.files)
    d = data.d
    k = data.k()
    for lab in ("old", "new"):
        if lab not in d:
            raise Fail(f"no rows labelled {lab}")
    unknown = sorted(set(d) - {"old", "new"})
    if unknown:
        raise Fail(f"labels {unknown} are not old/new: the files of the grid? (use `grid`)")
    old, new = d["old"], d["new"]
    print(cell_counts(d, "old"))
    print(cell_counts(d, "new"))
    sizes = same_sizes(data, ["old", "new"])
    for n in sizes:
        if len(old[n]) != len(new[n]):
            raise Fail(f"N={n}: {len(old[n])} old visits against {len(new[n])} new: an interrupted or stale run")
    if len({len(old[n]) for n in sizes}) != 1:
        raise Fail("the sizes were measured a different number of times: an interrupted or stale run")
    lack = [n for n in MIX if n not in sizes]
    if lack:
        raise Fail(f"the mix sizes {lack} were not measured at old and new: the mix would change its workload")
    fs = data.fs.get("new", set())
    if len(fs) > 1:
        raise Fail(f"the new rows were measured with different f {sorted(fs)}")
    f = next(iter(fs)) if fs else ASSUMED_F
    applies = abs(f - ASSUMED_F) < 1e-9
    print(f"columns = tokens - k, k = {k}. new arm: work table at f {f:.2f}"
          f"{'' if fs else ' (not recorded in the rows: the assumed value is taken)'}; predicted ranges are the "
          f"GA10x / Ada bracket for f {ASSUMED_F:.2f}"
          f"{'' if applies else ' and do NOT apply to this run: no verdict'}.")
    print("The new arm does not move the host cost model: route choice stays on the production table.")
    print(next(ln for ln in CRITERIA.splitlines() if ln.strip().startswith("R4")))

    def spread(rows: list[list[float]]) -> str:
        m = [statistics.median(r) for r in rows]
        return f"{statistics.median(flat(rows)):7.2f} [{min(m):.2f}-{max(m):.2f}]"

    print("size cols widths(old->new)  old median [range of the visits' medians]  new median [range]  "
          "change | model work | predicted range")
    outside = judged = 0
    for n in sizes:
        c = n - k
        o, w = statistics.median(flat(old[n])), statistics.median(flat(new[n]))
        jo, jn = pick(c, None, OLD), pick(c, f, PRODUCTION)
        model = 100 * (work(c, jn, f) / work(c, jo, f) - 1)
        ch = 100 * (w / o - 1)
        pr = pred_range(c)
        if pr is None:
            ptxt = "-"
        else:
            judged += 1
            bad = not pr[0] - 1e-9 <= ch <= pr[1] + 1e-9
            outside += bad
            ptxt = f"{pr[0]:+.1f}..{pr[1]:+.1f} %{'  OUTSIDE' if bad else ''}"
        print(f"N={n:3d} {c:4d} J{jo:<3d}->J{jn:<3d} old {spread(old[n])}  new {spread(new[n])} ms {ch:+6.1f} % "
              f"| {model:+6.1f} % | {ptxt}")
    print(f"measured change outside the predicted range: {outside} of {judged} sizes that have a prediction")
    mix = [n for n in MIX if n in sizes]
    visits = len(old[mix[0]])
    per = []
    for v in range(visits):
        mo = statistics.mean(statistics.median(old[n][v]) for n in mix)
        mw = statistics.mean(statistics.median(new[n][v]) for n in mix)
        per.append(100 * (mw / mo - 1))
    mo = statistics.mean(statistics.median(flat(old[n])) for n in mix)
    mw = statistics.mean(statistics.median(flat(new[n])) for n in mix)
    mix_ch = 100 * (mw / mo - 1)
    print(f"mix {mix}: old {mo:.2f} new {mw:.2f} ms {mix_ch:+.1f} %  "
          f"(per visit {min(per):+.1f} .. {max(per):+.1f} %, {visits} visits)")
    prs = [r for r in (pred_range(n - k) for n in mix) if r]
    if prs:
        lo, hi = statistics.mean(r[0] for r in prs), statistics.mean(r[1] for r in prs)
        print(f"predicted mix change (the mean of the per-size ranges of the {len(prs)} mix sizes that have "
              f"one): {lo:+.1f} .. {hi:+.1f} %")
        if applies:
            hit = mix_ch < lo - R4_PP or mix_ch > hi + R4_PP
            print(f"  R4: {'MET (refutes f %.2f)' % ASSUMED_F if hit else 'not met'} "
                  f"(measured {mix_ch:+.1f} % against {lo - R4_PP:+.1f} .. {hi + R4_PP:+.1f} %)")


def kprobe(a: argparse.Namespace) -> None:
    data = load([a.file])
    if len(data.d) != 1:
        raise Fail("kprobe wants one label")
    cells = next(iter(data.d.values()))
    e = a.width
    near = sorted(n for n in cells if e <= n <= e + 24)
    if len(near) < 3:
        raise Fail(f"kprobe needs at least three sizes in {e}..{e + 24} (measure the sizes {e}..{e + 18} "
                   f"with the width forced to {e})")
    m = {n: statistics.median(flat(cells[n])) for n in near}
    jumps = sorted(((100 * (m[b] / m[x] - 1), x, b) for x, b in zip(near, near[1:])), reverse=True)
    top, x, b = jumps[0]
    print("sizes " + " ".join(f"{n}:{m[n]:.2f}" for n in near))
    print(f"biggest step {top:+.1f} % between N={x} and N={b} (next: {jumps[1][0]:+.1f} %): the call crosses "
          f"{e} columns there, so k lies in [{x - e}, {b - e - 1}]"
          f"{' (exact: consecutive sizes)' if b - x == 1 else ''}")
    if top < 3 * max(jumps[1][0], 0.1):
        print("warning: the step is not clearly above the other steps: do not trust this interval")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    m = sub.add_parser("measure")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=18524)
    m.add_argument("--label", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--sizes", help="comma list of prompt tokens, or grid / ab / mix (default: the 21 grid sizes)")
    m.add_argument("--reps", type=int, default=4)
    m.add_argument("--warmup", type=int, default=3)
    m.add_argument("--k", type=int, default=K_ASSUMED,
                   help="tokens - columns of the body call (ASSUMED %d unless `kprobe` measured it)" % K_ASSUMED)
    m.add_argument("--f", type=float, help="the f of the run, recorded in the rows (the A/B's new arm)")
    m.set_defaults(fn=measure)
    g = sub.add_parser("grid")
    g.add_argument("files", nargs="+")
    g.set_defaults(fn=grid)
    b = sub.add_parser("ab")
    b.add_argument("files", nargs="+")
    b.set_defaults(fn=ab)
    kp = sub.add_parser("kprobe")
    kp.add_argument("file")
    kp.add_argument("--width", type=int, default=128)
    kp.set_defaults(fn=kprobe)
    a = ap.parse_args()
    try:
        return a.fn(a) or 0
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
