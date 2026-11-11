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
grid     reads the per-width files of `measure` (label "J<w>"; one file per visit of a width, every
         width the same number of visits and every visit all the sizes, else it refuses) and prints,
         with the assumption f 0.20 beside the measurement: f MEASURED by a least-squares fit of the
         work law over all cells, with a bootstrap interval over the width visits and a lack-of-fit
         ratio (FIT below), the per-size predicted width against the measured best of the six
         production widths, the mismatch count, what the two extra widths 48 and 112 would gain
         (information), the f buckets (one table of picks each, with the sizes that decide each edge),
         the regret per bucket and at f 0.20 (a report), and the verdict of the criteria that were
         fixed before the run (CRITERIA below). The script measures every width twice, 16..128 then
         128..16, so a drift in time does not line up with J. Last, the grid read with the launch
         law of DECISIONS 78 (THE LAUNCH LAW below): with the tree's constants (no refit) and
         refitted, the law's pick per size beside the measured best, with each linear's launch mode.
ab       reads files labelled old/new (the same sizes, the same number of visits at every size, the
         whole mix) and prints the per-size change (medians with the range of the visits) beside the
         predicted range, the mismatch count, and the mix change beside its predicted range (the
         per-size ranges weighted by the old medians, as the measured mix change of means is).
kprobe   reads one `measure` file made with the width forced to 128 and consecutive sizes (130..146)
         and prints the interval of k that the biggest step in time implies (columns cross 128 there),
         ending with one machine-readable line `KPROBE k_lo=<a> k_hi=<b> trusted=<0|1>` that the
         script's `all` reads (it refuses to go on with a K outside [a, b], and takes K = a if a = b).

Both readers refuse empty files, malformed lines, rows whose prompt_n differs from n, rows without k,
and in `grid` a size measured twice in one visit or missing from one (stale rows), and print the
number of requests behind every label.

The regret of a pick = (time at the picked width - least time over the six production widths) /
that least, averaged over the sizes; the pick is the production rule (core/mmq_columns.h pick: least
ceil(c / J) x (f + (1 - f) J / 128) among 16 32 64 80 96 128, the narrower on a tie), c = N - k.
J 48 and J 112 are measured too (they follow the same law and show what more widths would give), but
the table cannot pick them, so no criterion compares a pick with them: they win some sizes by padding
alone (41 and 105 columns, for example) whatever f is.

FIT. The work law the table rests on, T(N, J) = C_N + a x tiles x (1 - J / 128) + b x tiles x J / 128
(tiles = ceil(c / J); one constant C_N per size, because the width-independent part grows with N), is
fitted by least squares over the cells (size x width visit, the median of each cell), so f = a / b is
MEASURED, not scored bucket by bucket; with C_N per size only the differences between the widths at
one size carry it. Each cell is weighted by 1 / s_N^2, s_N the median time at its size: the request
noise is a share of the time, so the residuals are relative and a 512-token cell does not outweigh a
64-token one tenfold. The interval is a bootstrap over the width visits (one block = one server run of
one width, all its sizes together: a drift of that run moves all its cells at once, so the cell is not
the unit; BOOT_N draws, fixed seed BOOT_SEED: the output is deterministic). The lack-of-fit ratio is
sqrt(residual variance of the cells about the fit / the variance of a cell median expected from its own
requests); near 1 the law describes the card to the noise, above LOF_MAX it does not and the fit's f is
no measurement of f.
RUN OFFSETS. The request noise is the noise WITHIN one server run, but every width visit is a server
start of its own, and a card that boosts differently from one start to the next (1-2 % is ordinary)
moves all the cells of a run together. Without a term for that, the drift between runs sits in the
residual and raises the ratio as a misfit of the law would (the law at f 0.20 exactly read "no verdict"
in 7 of 12 seeds at +-2 % per run). So the fit carries one free offset D_r per server run, in ms
(T = C_N + D_r + the work law), identified through the size constants C_N that all runs share. They are
FITTED NUISANCE TERMS, not facts about the card, and the output labels them so. An additive offset, not
a scale, because the runs are nested in the widths (one width a run): a per-run scale x T would also
take any change of the tile cost that is the same for every size of one width (an occupancy step, an f
that depends on J) and hide exactly those misfits, while an offset in ms does not grow with the tiles.
It takes a scale drift less than fully (a +-2 % scale leaves a ratio about 1.6, well below LOF_MAX). It
does take a per-width launch constant (f then reads 0.199 against a true 0.20): that is right, a launch
constant is not a share of the tile work, and f is the share. The degrees of freedom are counted from
the cells: cells - size constants - (runs - the constant runs and sizes share) - 2 (a, b). The pure
error of the replicate visits is not used as the denominator: it holds the run drift too, so it would
over-count the noise and lose the power against misfits. Beside the ratio the output prints the drift
it saw (OBSERVED, not fitted): per width the median over the sizes of visit 2 / visit 1, and the median
and range over the widths, and the ratio without the run offsets for comparison. It also prints the
residual beyond the request noise, the misfit's own size as a share of the time: the ratio grows as the
request noise falls, this does not, so the candidate cause a no-verdict names is chosen on it (WAVE_RMS).
The regret per bucket (R1) stays as a report: the table of picks is the same for
every f in (0.111, 0.200] and its neighbours differ at one to three sizes, so the regret cannot
separate f 0.10 from 0.30.
Nothing is tuned to the data: every predicted value is computed from the rule or taken from the
constants above before the data is read, and is printed beside the measured one. The only constants
are the thresholds of CRITERIA, TOL, the predicted range, the interval level, LOF_MAX and the group
edges of PRED_AB, all set a priori and labelled so.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
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
# distance from a tile edge of the widths 64 80 96 128 and from the MMQ gate (MMQ_GATE) for k = 7
# (`margins`), so a k that is a few off changes neither a tile count nor the route. The smallest size
# is 32 (25 columns), not 24: 24 - 7 = 17 columns sits on the gate, and a k of 8 would take that cell
# off ggml's matrix kernel at every width. 40 keeps the grid at 21 sizes.
SIZES = [32, 40, 48, 64, 80, 96, 112, 128, 142, 144, 160, 176, 192, 208, 224, 256, 288, 320, 384, 448, 512]
AB_SIZES = [96, 128, 142, 144, 160, 176, 192, 208, 224, 240, 256, 270, 288, 320, 384, 448, 512]
MIX = [144, 160, 176, 192, 208, 224, 240, 256]
WIDTHS = [16, 32, 48, 64, 80, 96, 112, 128]
PRODUCTION = [16, 32, 64, 80, 96, 128]
OLD = [16, 32, 64, 128]
EDGE_WIDTHS = [64, 80, 96, 128]
# ggml's matrix kernel takes a prefill call from 17 columns (src/ops/generic/route_authorities.cpp,
# mmq_prefill_gate: a call of fewer tokens goes elsewhere); a fact of the route, not a threshold
MMQ_GATE = 17
MARGIN_WARN = 4
ASSUMED_F = 0.20  # the assumption under test (an assumption, not a derivation)
F_ONE = 0.43  # DECISIONS 69, shown as the representative of the top bucket when it lies in it
SALT_DIGITS = 6  # a fixed-width salt: every request spends the same number of tokens on it
TOL = 2.0  # percent: a predicted width slower than the best by more than this is a material mismatch
# Fixed BEFORE any data exists (not fitted); thresholds are printed with the verdicts.
R1_PP = 0.5  # percentage points of regret that the assumed f may lose to the least bucket
R2_SHARE = 0.25  # share of sizes with a material mismatch at the assumed f
R4_PP = 2.0  # percentage points the A/B mix may lie outside the predicted mix range
# The prediction R5 tests is the stated range, not the point: "f 0.20, best range 0.15-0.25" (the
# script's header; DECISIONS 74's measured cards are best at 0.15-0.20 or 0.25-0.30). A priori.
PRED_LO, PRED_HI = 0.15, 0.25
FIT_LEVEL = 0.95  # chosen a priori: the two-sided level of the bootstrap interval of the fitted f
BOOT_N = 1000  # bootstrap draws over the width visits (a priori; enough for a 95 % percentile interval)
BOOT_SEED = 524  # fixed: the same files print the same interval
FIT_EPS = 1e-6  # float rounding only (an exact law gives an interval of zero width), not a margin
# Lack of fit (after the run offsets, FIT): above this ratio the law does not describe the card and R5
# gives no verdict on f. Chosen a priori, from synthetic grids at 2 % request noise and two visits (none
# of them a measurement; the ratio without the run offsets in brackets): the pure law gives about 1.0
# (1.0), a per-run scale of +-1 % 1.2 (1.9), of +-2 % 1.6 (3.3), a linear drift of 3 % over the 16 runs
# 1.7 (3.1); a per-width launch constant is taken by the offsets (1.0, f 0.199); 68-SM wave
# quantisation 2.4, an occupancy step x1.08 at J >= 80 2.8, a per-tile tail of 0.5 % a tile 3.0, an f
# that depends on J (0.10..0.30) 3.3, a cost step x1.25 at N > 256 (L2) 5.5, an occupancy step x1.3 9,
# a per-tile tail of 2 % a tile 11. The ratio scales with 1 / request noise: at 1 % a misfit's ratio
# about doubles (wave 4.6), at 0.5 % it about quadruples (wave 9.0). A run drift leaves a residual
# beyond the noise of about 0.24 % rms at +-1 % per run, so at 0.5 % request noise the pure law with
# that drift reads about 2.7.
LOF_MAX = 3.0
# The misfit's own size does not scale with the noise: the residual beyond the request noise,
# sqrt(residual variance - request variance), a share of the size's time. The 68-SM wave quantisation
# of the stream-k launch (64 row blocks over 68 SMs, the law's tile cost per wave) gives 0.80-0.90 %
# at 0.5, 1 and 2 % request noise, with and without a run drift; an occupancy step x1.08 1.0, an f that
# depends on J or a 0.5 % tail 1.1, a cost step x1.18 at N > 256 1.5, x1.25 2.0, an occupancy step x1.3
# 3.4, a 2 % tail 4.3 (synthetic grids, none a measurement). A priori bound: up to this share the
# no-verdict text names the wave quantisation as one candidate cause (a misfit larger than that is more
# than the wave effect of the law's own tile cost can give).
WAVE_RMS = 0.0125
MEDIAN_VAR = 1.5708  # pi / 2: the variance of a median of m requests ~ (pi / 2) s^2 / m (normal theory)
# GA10x / Ada bracket (DECISIONS 74), by columns; the edges 168 / 200 are the midpoints between the
# sample sizes 160|176 and 192|208 of the old rows, not fitted: (first col, last col, lo %, hi %)
PRED_AB = [(129, 168, -19.0, -14.0), (169, 200, -13.0, -9.0), (201, 260, -0.5, 0.5)]
CRITERIA = f"""REFUTATION CRITERIA for f = {ASSUMED_F:.2f} (fixed before the run; the thresholds are not fitted):
  R5 grid: f fitted to the work law over all cells (FIT), with its {int(FIT_LEVEL * 100)} % bootstrap interval over the width
     visits, against the predicted range {PRED_LO:.2f}..{PRED_HI:.2f}: the whole interval outside it REFUTES, the whole
     interval inside it stands, an interval across an edge is UNDECIDED. When the law does not fit (the
     lack-of-fit ratio above {LOF_MAX:g} after one fitted offset per server run, or b <= 0), R5 gives no verdict
     on f: the fit's f is not a measurement.
  R2 grid: more than {int(R2_SHARE * 100)} % of the sizes have a material mismatch: the width picked at f {ASSUMED_F:.2f} is more
     than {TOL:g} % slower than the best of the six production widths (J 48 / J 112 are not pickable).
  R3 grid: the regret at f {ASSUMED_F:.2f} is not below the regret of the fewest tiles (the table gives no gain).
  R4 ab (f {ASSUMED_F:.2f}): the mix change lies more than {R4_PP:g} pp outside the predicted mix range.
  R1 grid is a REPORT, not a verdict: the regret at f {ASSUMED_F:.2f} against the least over the f buckets
     ({R1_PP} pp shown for scale). Neighbouring buckets' tables differ at one to three sizes, so the
     regret cannot separate f 0.10 from 0.30; the fit (R5) can.
Any one of R2..R5 met refutes f = {ASSUMED_F:.2f} for this card; none met leaves it standing (one campaign is not a proof)."""


class Fail(Exception):
    """A readable error: main() prints it and exits 1, no traceback."""


class Unbuildable(Fail):
    """A prompt size that cannot be built: measure() skips it and reports it."""


def margin(c: int, j: int) -> int:
    """How many columns the count may be off, either way, before ceil(c / j) changes."""
    r = (c - 1) % j + 1
    return min(r - 1, j - r)


def margins(n: int, k: int) -> int:
    """Columns the count may be off before a tile count of the edge widths changes or the call leaves
    ggml's matrix kernel (MMQ_GATE); negative when the call is below the gate already."""
    c = n - k
    return min(min(margin(c, j) for j in EDGE_WIDTHS), c - MMQ_GATE)


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


def law_terms(c: int, j: int) -> tuple[float, float]:
    """The two regressors of the work law at c columns and width j: tiles x (1 - J/128), tiles x J/128."""
    t = -(-c // j)
    return t * (1 - j / 128), t * j / 128


def _centre(vecs: list[list[float]], w: list[float], key: list) -> dict:
    """Subtract from every vector its w-weighted mean within each group of `key`, in place; returns the
    means subtracted from the last vector (group -> mean)."""
    tot: dict = defaultdict(float)
    for wi, g in zip(w, key):
        tot[g] += wi
    last: dict = {}
    for v in vecs:
        m: dict = defaultdict(float)
        for wi, g, x in zip(w, key, v):
            m[g] += wi * x
        for g in m:
            m[g] /= tot[g]
        for i, g in enumerate(key):
            v[i] -= m[g]
        last = m
    return last


def _components(sizes: list, runs: list) -> int:
    """Connected components of the bipartite graph size - run (an edge per cell): the size constants and
    the run offsets together have len(sizes) + len(runs) - components free parameters."""
    parent: dict = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, r in zip(sizes, runs):
        parent[root(("s", s))] = root(("r", r))
    return len({root(x) for x in list(parent)})


def fit_law(cells: list[tuple[int, float, float, float]], weights: list[float] | None = None,
            runs: list | None = None):
    """Weighted least squares of y = C_N + D_r + a x1 + b x2 over cells (N, x1, x2, y), one free constant
    C_N per size N and, when `runs` is given, one free offset D_r (ms) per server run (runs[i] = the run of
    cell i; a FITTED nuisance term, see FIT), cell i weighted by `weights[i]` (a zero weight drops the
    cell). The two sets of constants are taken out by alternating weighted centring (by size, then by
    run, until nothing moves: one sweep is exact when the weights are a size factor times a run factor
    and every run holds every size, as in `grid`). Returns (a, b, weighted residual sum of squares, sum of
    the weights, the number of free constants, run -> D_r with weighted mean 0) or None when the cells do
    not determine a and b (every size's widths on one line, or the run offsets take all of it)."""
    keep = [i for i in range(len(cells)) if (1.0 if weights is None else weights[i])]
    w = [1.0 if weights is None else weights[i] for i in keep]
    size = [cells[i][0] for i in keep]
    x1, x2, y = ([cells[i][c] for i in keep] for c in (1, 2, 3))
    _centre([x1, x2, y], w, size)
    # what the size constants leave of the regressors: a regressor the run offsets take (nearly) all of
    # is not determined, whatever rounding leaves of it
    q11, q22 = (sum(wi * e * e for wi, e in zip(w, v)) for v in (x1, x2))
    offsets: dict = {}
    nconst = len(set(size))
    if runs is not None:
        run = [runs[i] for i in keep]
        nconst = len(set(size)) + len(set(run)) - _components(size, run)
        scale = [1e-12 * (1 + max(map(abs, v), default=0.0)) for v in (x1, x2, y)]
        dy: dict = defaultdict(float)  # the run means taken from y (with those from x1, x2 they give D_r)
        d1: dict = defaultdict(float)
        d2: dict = defaultdict(float)
        for _ in range(200):
            moved = 0.0
            for k, acc in enumerate((d1, d2, dy)):
                v = (x1, x2, y)[k]
                m = _centre([v], w, run)
                for r, x in m.items():
                    acc[r] += x
                moved = max(moved, max(map(abs, m.values())) / scale[k])
                ms = _centre([v], w, size)
                moved = max(moved, max(map(abs, ms.values())) / scale[k])
            if moved <= 1.0:
                break
    s11 = s12 = s22 = s1y = s2y = syy = 0.0
    for wi, e1, e2, ey in zip(w, x1, x2, y):
        s11, s12, s22 = s11 + wi * e1 * e1, s12 + wi * e1 * e2, s22 + wi * e2 * e2
        s1y, s2y, syy = s1y + wi * e1 * ey, s2y + wi * e2 * ey, syy + wi * ey * ey
    det = s11 * s22 - s12 * s12
    if s11 <= 1e-9 * q11 or s22 <= 1e-9 * q22 or det <= 1e-9 * s11 * s22:
        return None
    a = (s1y * s22 - s2y * s12) / det
    b = (s2y * s11 - s1y * s12) / det
    rss = syy - a * s1y - b * s2y
    if runs is not None:
        offsets = {r: dy[r] - a * d1[r] - b * d2[r] for r in dy}
    # below rounding (an exact law): zero, so that an exact law reads as an exact fit
    return a, b, (rss if rss > 1e-12 * syy else 0.0), sum(w), nconst, offsets


def lack_of_fit(rss: float, dof: int, rel: list[float], within: list[float] | None) -> float | None:
    """sqrt(residual variance per degree of freedom / the mean relative variance of a cell median expected
    from its own requests); None without request spread information or degrees of freedom."""
    if within is None or dof <= 0:
        return None
    noise = statistics.mean(r * v for r, v in zip(rel, within))
    resid = rss / dof
    if noise > 0:
        return (resid / noise) ** 0.5
    return 0.0 if resid <= 1e-18 else float("inf")  # requests without any spread: only an exact law fits


def beyond_noise(rss: float, dof: int, rel: list[float], within: list[float] | None) -> float | None:
    """sqrt(max(residual variance per degree of freedom - the mean relative variance of a cell median
    expected from its requests, 0)): the misfit's own size, a share of the size's time (with rel = 1 /
    s_N^2), which unlike the ratio does not grow as the request noise falls; None as lack_of_fit."""
    if within is None or dof <= 0:
        return None
    return max(rss / dof - statistics.mean(r * v for r, v in zip(rel, within)), 0.0) ** 0.5


def fit_f(cells: list[tuple[int, float, float, float]], blocks: list, rel: list[float],
          within: list[float] | None = None, per_run: bool = True) -> dict:
    """f = a / b of the work law over `cells` (N, x1, x2, median ms), cell i weighted by rel[i]
    (1 / s_N^2: relative residuals), with a bootstrap interval over the blocks (blocks[i] = the width
    visit of cell i: BOOT_N draws of as many blocks with replacement, BOOT_SEED) and, from within[i] (the
    variance of cell i's median expected from its requests, ms^2), the lack-of-fit ratio. The blocks are
    the server runs: the fit carries one FITTED offset per block (fit_law `runs`, see FIT); `per_run`
    False fits without them (printed beside, for comparison)."""
    runs = blocks if per_run else None
    full = fit_law(cells, rel, runs)
    if full is None:
        return {"ok": False, "why": "the sizes do not determine the law (every width one tile?)"}
    a, b, rss, wsum, nconst, offsets = full
    # b at rounding level is no growth with the width (times that do not depend on J at all)
    bmin = 1e-9 * statistics.median(abs(c[3]) for c in cells)
    if not b > bmin:
        return {"ok": False, "why": f"b = {b:.4g} ms is not positive: the time does not grow with the "
                                    "tile's width, the work law does not describe this card"}
    # the counts behind the degrees of freedom: cells - (size constants + run offsets - the constants the
    # two share, fit_law) - a, b
    dof = len(cells) - nconst - 2
    # rel = 1 / s_N^2, so rss is a sum of squared relative residuals: rms is a share of the size's time
    out = {"ok": True, "f": a / b, "a": a, "b": b, "rms": (rss / len(cells)) ** 0.5, "lof": None, "dof": dof,
           "cells": len(cells), "sizes": len({c[0] for c in cells}), "runs": len(set(blocks)) if per_run else 0,
           "nconst": nconst, "offsets": offsets}
    out["lof"] = lack_of_fit(rss, dof, rel, within)
    out["beyond"] = beyond_noise(rss, dof, rel, within)
    ids = sorted(set(blocks), key=str)
    index = {x: i for i, x in enumerate(ids)}
    rng = random.Random(BOOT_SEED)
    draws, bad = [], 0
    for _ in range(BOOT_N):
        count = [0] * len(ids)
        for _ in ids:
            count[rng.randrange(len(ids))] += 1
        r = fit_law(cells, [count[index[bl]] * w for bl, w in zip(blocks, rel)], runs)
        if r is None or not r[1] > bmin:
            bad += 1  # the draw does not determine f (one width only, say): dropped, not counted as a value
            continue
        draws.append(r[0] / r[1])
    if not draws:
        return {"ok": False, "why": "no bootstrap draw determines the law"}
    draws.sort()
    q = (1 - FIT_LEVEL) / 2
    out.update(lo=draws[round(q * (len(draws) - 1))], hi=draws[round((1 - q) * (len(draws) - 1))],
               bad=bad, draws=len(draws), dmin=draws[0], dmax=draws[-1], blocks=len(ids))
    return out

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
        # label -> n -> the index of the file each row of d came from (a file is one visit)
        self.src: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        self.files: dict[str, set[int]] = defaultdict(set)  # label -> the files (visits) it has rows in

    def k(self) -> int:
        if len(self.ks) != 1:
            raise Fail(f"the files were measured with different k {sorted(self.ks)}: one k per campaign")
        return next(iter(self.ks))


def load(paths: list[str]) -> Data:
    data = Data()
    for fi, p in enumerate(paths):
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
            data.src[label][n].append(fi)
            data.files[label].add(fi)
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


# --- THE LAUNCH LAW (DECISIONS 78) ---------------------------------------------------------------
# The Turing row's law the tree ranks the widths by (core/mmq_columns.h launch_of / launch_seconds,
# core/generation.h kTuringMmqLaw; tests/test_mmq_width_grid.py holds these to the tree): a product's
# time is its longest CTA's units times a unit's cost, plus the stream-k fixup (its partial tiles'
# bytes at the DRAM rate and one dependent launch) where the launch splits tiles. The `grid` reader
# prices every cell with it beside the tile law: with the law's constants as they stand (no refit;
# one C_N per size and one D_r per run fitted as in FIT), and refitted (units by launch mode and the
# fixup launches free), naming each pick's launch modes.
# Facts of the board and the model the kit measures (the RTX 2080 Ti's facts row: 68 SMs, a 1545 MHz
# reference clock, 352 bits at 14 Gbps; Qwen3-8B's linears on the MMQ path, routes/tu102.json and
# generic_gemm.cpp: gate and up two products, down K 12288; 36 layers):
LAW_SMS = 68
LAW_CLOCK_HZ = 1545e6
LAW_DRAM_BYTES_PER_S = 616e9
LAW_LINEARS = [(5120, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (12288, 4096), (4096, 12288)]
LAW_LAYERS = 36
# kTuringMmqLaw (MEASURED; core/generation.h names the source of each value)
LAW_UNIT_CLOCKS = {16: 5256.0, 32: 7047.0, 64: 9698.0, 80: 11170.0, 96: 13134.0, 128: 17920.0}
LAW_STREAM_K_UNIT = 1.18
LAW_LAUNCH_S = 3.3e-6
LAW_CELL_TOL = 1.0  # percent, a priori: a cell predicted within this of its median (the run drift is 0-0.9 %)


def launch_of(rows: int, k: int, cols: int, j: int, sms: int = LAW_SMS) -> dict:
    """core/mmq_columns.h launch_of: the launch select_launch makes (a tile a CTA where the last wave is
    90 % full, else stream-k over min(SMs, units) CTAs; the fixup where the CTAs do not divide the tiles)."""
    tiles = -(-rows // 128) * -(-cols // j)
    steps = max(1, -(-k // 256))
    waves = -(-tiles // sms)
    if 100 * tiles // (sms * waves) >= 90:
        return {"tiles": tiles, "blocks": tiles, "units": waves * steps, "split": False, "fixup": False,
                "partials": 0, "split_tiles": 0}
    work = tiles * steps
    blocks = min(sms, work)
    ends = [b * work // blocks for b in range(1, blocks)]
    inside = [e for e in ends if e % steps]
    return {"tiles": tiles, "blocks": blocks, "units": -(-work // blocks), "split": True,
            "fixup": tiles % blocks != 0, "partials": len(inside), "split_tiles": len({e // steps for e in inside})}


def law_seconds(rows: int, k: int, cols: int, j: int) -> float | None:
    """core/mmq_columns.h launch_seconds with the Turing row's law; None for a width the law has no unit of."""
    if j not in LAW_UNIT_CLOCKS:
        return None
    lo = launch_of(rows, k, cols, j)
    unit = LAW_UNIT_CLOCKS[j] * (LAW_STREAM_K_UNIT if lo["split"] else 1.0) / LAW_CLOCK_HZ
    s = lo["units"] * unit
    if lo["fixup"]:
        s += (lo["partials"] + 2.0 * lo["split_tiles"]) * 128 * j * 4 / LAW_DRAM_BYTES_PER_S + LAW_LAUNCH_S
    return s


def law_ms(cols: int, j: int) -> float | None:
    """The model's MMQ linears of one prefill body call at the law (ms), None off the law's widths."""
    parts = [law_seconds(r, kk, cols, j) for r, kk in LAW_LINEARS]
    return None if None in parts else 1e3 * LAW_LAYERS * sum(parts)


def law_terms_of(cols: int, j: int) -> dict:
    """The launch law's regressors of a cell over the model's linears: units on the longest CTA by mode
    (and times J / 128) and the fixup launches; and each linear's launch mode, for the output."""
    t = {"ut": 0.0, "utj": 0.0, "us": 0.0, "usj": 0.0, "fix": 0.0, "modes": []}
    for r, kk in LAW_LINEARS:
        lo = launch_of(r, kk, cols, j)
        key = "us" if lo["split"] else "ut"
        t[key] += LAW_LAYERS * lo["units"]
        t[key + "j"] += LAW_LAYERS * lo["units"] * j / 128
        t["fix"] += LAW_LAYERS * (1 if lo["fixup"] else 0)
        t["modes"].append("sk+fix" if lo["fixup"] else ("sk" if lo["split"] else "tile"))
    return t


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None for a singular system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    top = max((abs(x) for row in a for x in row), default=0.0)
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) <= 1e-12 * max(top, 1e-300):
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c] / m[c][c]
                for q in range(c, n + 1):
                    m[r][q] -= f * m[c][q]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_offsets_and(y: list[float], xs: list[list[float]], w: list[float], size: list, run: list):
    """Weighted least squares of y = C_N + D_r + sum_i beta_i x_i, one free constant per size and one per
    run, taken out by alternating weighted centring (fit_law's); returns (betas, residuals) or None."""
    vecs = [v[:] for v in xs] + [y[:]]
    for v in vecs:
        top = 1e-12 * (1 + max(map(abs, v), default=0.0))
        for _ in range(500):
            moved = 0.0
            for key in (run, size):
                m = _centre([v], w, key)
                moved = max(moved, max(map(abs, m.values()), default=0.0))
            if moved <= top:
                break
    *ex, ey = vecs
    if not ex:
        return [], ey
    a = [[sum(wi * p * q for wi, p, q in zip(w, xi, xj)) for xj in ex] for xi in ex]
    b = [sum(wi * p * q for wi, p, q in zip(w, xi, ey)) for xi in ex]
    beta = _solve(a, b)
    if beta is None:
        return None
    return beta, [e - sum(bi * xi[c] for bi, xi in zip(beta, ex)) for c, e in enumerate(ey)]


def launch_law_report(d, data, sizes: list[int], cols: list[int], med: dict, best: dict) -> dict:
    """The grid read with the launch law (no refit, then refitted) and the launch modes of its picks."""
    print("\nLAUNCH LAW (DECISIONS 78; core/mmq_columns.h): a product's time = its longest CTA's units x a unit's "
          f"cost + the stream-k fixup; Qwen3-8B's linears on {LAW_SMS} SMs (the RTX 2080 Ti's modes and constants)")
    cells, size, run, within = [], [], [], []
    for j in WIDTHS:
        for n, c in zip(sizes, cols):
            for v, fi in zip(d[f"J{j}"][n], data.src[f"J{j}"][n]):
                cells.append((n, c, j, statistics.median(v)))
                run.append((j, fi))
                size.append(n)
                within.append(MEDIAN_VAR * statistics.variance(v) / len(v) if len(v) > 1 else 0.0)
    scale = {n: statistics.median(c[3] for c in cells if c[0] == n) for n in sizes}
    w = [1 / scale[n] ** 2 for n in size]
    out: dict = {}

    def rms_beyond(idx: list[int], res: list[float], free: int) -> tuple[float, float]:
        ss = sum(w[i] * e * e for i, e in zip(idx, res))
        dof = max(len(idx) - len({size[i] for i in idx}) - len({run[i] for i in idx}) + 1 - free, 1)
        noise = statistics.mean(w[i] * within[i] for i in idx)
        return (ss / len(idx)) ** 0.5, max(ss / dof - noise, 0.0) ** 0.5

    # 1. no refit: the tree's constants, only the size constants and the run offsets fitted
    idx = [i for i, c in enumerate(cells) if c[2] in LAW_UNIT_CLOCKS]
    got = fit_offsets_and([cells[i][3] - law_ms(cells[i][1], cells[i][2]) for i in idx], [],
                          [w[i] for i in idx], [size[i] for i in idx], [run[i] for i in idx])
    _, res0 = got
    rms0, beyond0 = rms_beyond(idx, res0, 0)
    pred: dict = defaultdict(list)
    for i, e in zip(idx, res0):
        pred[cells[i][0], cells[i][2]].append(cells[i][3] - e)
    off = {key: 100 * (statistics.mean(p) / med[key[1]][key[0]] - 1) for key, p in pred.items()}
    near = sum(1 for v in off.values() if abs(v) <= LAW_CELL_TOL)
    print(f"  no refit (the tree's constants; C_N and D_r fitted): residual rms {100 * rms0:.2f} % of the size's "
          f"time (beyond the request noise {100 * beyond0:.2f} %) over {len(idx)} cells of the six production "
          f"widths; {near} of {len(off)} size x width medians within {LAW_CELL_TOL:g} %")
    print("  per size: the law's pick and its launch modes (q+k, v, o, gate, up, down) beside the measured best of "
          "the six production widths; the cell's prediction against its median")
    regret = []
    for n, c in zip(sizes, cols):
        jl = min(PRODUCTION, key=lambda j: (law_ms(c, j), j))
        jb = min(PRODUCTION, key=lambda j: med[j][n])
        loss = 100 * (med[jl][n] / best[n] - 1)
        regret.append(loss)
        modes = "/".join(law_terms_of(c, jl)["modes"])
        print(f"{n:4d} {c:5d}  law J{jl:<4d} {modes:<34s} best J{jb:<4d} {med[jl][n]:8.2f} {best[n]:8.2f} "
              f"{loss:5.1f}%   cell {off[n, jl]:+.1f} %" + ("  MISMATCH" if loss > TOL else ""))
    hits = sum(1 for r in regret if r <= 0)
    print(f"  the law's picks: {hits} of {len(sizes)} sizes at the measured best, mean regret "
          f"{statistics.mean(regret):.2f} %")
    out.update(rms=rms0, beyond=beyond0, near=near, cells=len(off), hits=hits, regret=statistics.mean(regret),
               off=off)

    # 2. refit: units by launch mode (a + b J / 128 each) and the fixup launches free
    terms = [law_terms_of(c, j) for _, c, j, _ in cells]
    xs = [[t[key] for t in terms] for key in ("ut", "utj", "us", "usj", "fix")]
    got = fit_offsets_and([c[3] for c in cells], xs, w, size, run)
    if got is None:
        print("  refit: the cells do not determine the law's terms")
        return out
    beta, res = got
    rms, beyond = rms_beyond(list(range(len(cells))), res, len(beta))
    print(f"  refit (FITTED, a diagnosis, not the tree's constants): a tile-a-CTA unit {1e3 * beta[0]:.2f} + "
          f"{1e3 * beta[1]:.2f} J/128 us, a stream-k unit {1e3 * beta[2]:.2f} + {1e3 * beta[3]:.2f} J/128 us, "
          f"{1e3 * beta[4]:.1f} us a fixup launch; residual rms {100 * rms:.2f} % (beyond the request noise "
          f"{100 * beyond:.2f} %) over {len(cells)} cells, all eight widths")
    out.update(refit=beta, refit_rms=rms)
    return out


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
    skipped = sorted({n for v in data.skipped.values() for n in v})
    if skipped:
        print(f"skipped sizes (could not be built): {skipped}")
    sizes = same_sizes(data, labels)
    for lab in labels:
        files = sorted(data.files[lab])
        off = sorted(n for n, src in data.src[lab].items() if sorted(src) != files)
        if off:
            raise Fail(f"{lab}: sizes {off} are not measured exactly once in each of its {len(files)} visit files: "
                       "stale rows from an earlier run or an incomplete visit; one file per visit")
    nvis = {lab: len(data.files[lab]) for lab in labels}
    if len(set(nvis.values())) != 1:
        raise Fail(f"the widths were measured a different number of visits {nvis}: an interrupted or stale run")
    visits = nvis[labels[0]]
    print(f"visits per width: {visits}"
          + (" (the script runs them 16..128, then 128..16)" if visits > 1 else
             "; warning: ONE visit per width, so a drift between the widths' server runs lines up with J and only "
             "the 8 widths carry the interval; the script measures every width twice, counterbalanced"))
    cols = [n - k for n in sizes]
    if min(cols) < 1:
        raise Fail(f"size {sizes[cols.index(min(cols))]} has no column with k {k}")
    med = {j: {n: statistics.median(flat(d[f"J{j}"][n])) for n in sizes} for j in WIDTHS}
    # the best the table can reach: J 48 / J 112 are measured but not pickable (their gains are shown apart)
    best = {n: min(med[j][n] for j in PRODUCTION) for n in sizes}
    low = [(n, margins(n, k)) for n in sizes if margins(n, k) < MARGIN_WARN]
    if low:
        print(f"warning: sizes within {MARGIN_WARN} columns of a tile edge or of the {MMQ_GATE}-column MMQ gate "
              f"at k {k} (a k that far off changes a tile count or the route): {low}")
    cells = [(max(v) - min(v)) / statistics.median(v) for lab in labels for rows in d[lab].values()
             for v in rows if len(v) > 1]
    noise = 100 * statistics.median(cells) if cells else float("nan")

    bk = buckets(cols, ASSUMED_F)
    ta = pick_table(cols, ASSUMED_F)

    def regret(table: tuple[int, ...]) -> float:
        return 100 * statistics.mean((med[j][n] - best[n]) / best[n] for j, n in zip(table, sizes))

    # one fit cell per size x width visit: its median, its block (the width's server run), its weight
    # 1 / s_N^2 (relative residuals) and the variance of its median expected from its own requests
    fcells, blocks, within = [], [], []
    for j in WIDTHS:
        for n, c in zip(sizes, cols):
            for v, fi in zip(d[f"J{j}"][n], data.src[f"J{j}"][n]):
                fcells.append((n, *law_terms(c, j), statistics.median(v)))
                blocks.append((j, fi))
                within.append(MEDIAN_VAR * statistics.variance(v) / len(v) if len(v) > 1 else 0.0)
    scale = {n: statistics.median(y for m, _, _, y in fcells if m == n) for n in sizes}
    rel = [1 / scale[m] ** 2 for m, _, _, _ in fcells]
    one_request = all(len(v) < 2 for lab in labels for rows in d[lab].values() for v in rows)
    wvar = None if one_request else within
    fit = fit_f(fcells, blocks, rel, wvar)
    print(f"\nFIT of the work law T = C_N + D_r + a tiles (1 - J/128) + b tiles J/128 over {len(fcells)} cells "
          f"({len(sizes)} sizes x {len(WIDTHS)} widths x {visits} visits; one C_N per size, one fitted offset D_r "
          "per server run; residuals relative to the size's time):")
    if visits > 1:
        # OBSERVED, not fitted: what each width's later visit took against its first, size by size
        drift = {j: [statistics.median(statistics.median(d[f"J{j}"][n][v]) / statistics.median(d[f"J{j}"][n][0])
                                       for n in sizes) for v in range(1, visits)] for j in WIDTHS}
        allr = [x for j in WIDTHS for x in drift[j]]
        print("  run drift (observed, not fitted): per width the median over the sizes of "
              + ", ".join(f"visit {v + 1} / visit 1" for v in range(1, visits)) + ": "
              + ", ".join(f"J{j} " + "/".join(f"{x:.3f}" for x in drift[j]) for j in WIDTHS)
              + f"; over the widths median {statistics.median(allr):.3f}, range {min(allr):.3f}..{max(allr):.3f}")
    if fit["ok"]:
        print(f"  measured f = a / b = {fit['f']:.3f}, {int(FIT_LEVEL * 100)} % bootstrap interval "
              f"[{fit['lo']:.3f}, {fit['hi']:.3f}] ({BOOT_N} draws over the {fit['blocks']} width visits, seed "
              f"{BOOT_SEED}{', %d undetermined draws dropped' % fit['bad'] if fit['bad'] else ''})   "
              f"predicted f {ASSUMED_F:.2f}, range {PRED_LO:.2f}..{PRED_HI:.2f}")
        bare = fit_law(fcells, rel)  # the same fit without the run offsets, for comparison only
        bare_lof = None if bare is None else lack_of_fit(bare[2], len(fcells) - bare[4] - 2, rel, wvar)
        lof = ("not computed (one request a cell)" if fit["lof"] is None else
               f"{fit['lof']:.2f} (the residual after the run offsets against the cells' own request noise; above "
               f"{LOF_MAX:g} the law does not fit)"
               + ("" if bare_lof is None else f"; without the run offsets {bare_lof:.2f}"))
        print(f"  a = {fit['a']:.3f} ms, b = {fit['b']:.3f} ms (a 128-column tile's step over the linears), "
              f"residual rms {100 * fit['rms']:.2f} % of the size's time, lack-of-fit ratio {lof}")
        if fit["beyond"] is not None:
            print(f"  residual beyond the request noise {100 * fit['beyond']:.2f} % rms of the size's time (the "
                  "misfit's own size: unlike the ratio it does not grow as the request noise falls)")
        free = fit["nconst"] - fit["sizes"]
        print(f"  degrees of freedom {fit['dof']} = {fit['cells']} cells - {fit['sizes']} size constants - {free} run "
              f"offsets ({fit['runs']} runs, {fit['runs'] - free} shared with the size constants) - 2 (a, b)")
        off = sorted(fit["offsets"].values())
        if off:
            mid = statistics.median(off)
            typical = statistics.median(c[3] for c in fcells)
            print(f"  run offsets D_r (FITTED nuisance terms, not facts about the card; they absorb a drift of a "
                  f"server run and any per-width constant): {off[0] - mid:+.3f} .. {off[-1] - mid:+.3f} ms around "
                  f"the median run ({100 * (off[0] - mid) / typical:+.1f} .. {100 * (off[-1] - mid) / typical:+.1f} % "
                  "of the median cell)")
        for what, fv in (("the measured f", fit["f"]), ("the interval's low edge", fit["lo"]),
                         ("the interval's high edge", fit["hi"])):
            tf = pick_table(cols, fv)
            moved = [f"{n}t/{c}c:J{x}->J{y}" for n, c, x, y in zip(sizes, cols, ta, tf) if x != y]
            print(f"  the table of picks at {what} {fv:.3f} against f {ASSUMED_F:.2f}: "
                  + (f"differs at {len(moved)} sizes ({', '.join(moved)}), regret {regret(tf):.2f} % against "
                     f"{regret(ta):.2f} %" if moved else "the same at every size"))
    else:
        print(f"  no f: {fit['why']}")

    print("\nper size: the prediction at f %.2f beside the measurement (ms = median of the visits; best = the "
          "best of the six production widths)" % ASSUMED_F)
    print("  N  cols  predicted  best   ms@pred  ms@best  loss   " + " ".join(f"J{j:<5}" for j in WIDTHS)
          + "  J48/J112 gain")
    strict = material = 0
    extra = []
    for n, c, jp in zip(sizes, cols, ta):
        jb = min(PRODUCTION, key=lambda j: med[j][n])
        loss = 100 * (med[jp][n] / best[n] - 1)
        strict += med[jp][n] > best[n]
        material += loss > TOL
        flag = "  MISMATCH" if loss > TOL else ("  differs" if med[jp][n] > best[n] else "")
        jx = min((48, 112), key=lambda j: med[j][n])
        gain = 100 * (1 - med[jx][n] / best[n])
        if gain > 0:
            extra.append((n, c, jx, gain))
        print(f"{n:4d} {c:5d}  J{jp:<8d} J{jb:<4d} {med[jp][n]:8.2f} {best[n]:8.2f} {loss:5.1f}%  "
              + " ".join(f"{med[j][n]:6.2f}" for j in WIDTHS)
              + (f"  J{jx} {gain:.1f}%" if gain > 0 else "  -") + flag)
    print(f"width picked at f {ASSUMED_F:.2f} differs from the measured best in {strict} of {len(sizes)} sizes "
          f"(slower at all); more than {TOL:g} % slower (material mismatch): {material}")
    print("information, no verdict (the table cannot pick them): J 48 / J 112 beat the best production width at "
          + (f"{len(extra)} sizes: " + ", ".join(f"{n}t/{c}c J{j} {g:.1f} %" for n, c, j, g in extra)
             + f"; mean gain over all {len(sizes)} sizes {sum(e[3] for e in extra) / len(sizes):.2f} %"
             if extra else "no size"))

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
    r2, r3 = material > R2_SHARE * len(sizes), r_a >= r_old
    why5 = ""
    if not fit["ok"]:
        s5, t5 = "nofit", f"no f: {fit['why']}"
        why5 = "the work law does not describe this card"
    elif fit["lof"] is not None and fit["lof"] > LOF_MAX:
        why5 = ("the residual exceeds the request noise after the run drift is taken out: the work law does not "
                "fit this card (see the visit ratios for the drift)")
        if fit["beyond"] is not None and fit["beyond"] <= WAVE_RMS:
            why5 += (f"; the residual beyond the request noise is {100 * fit['beyond']:.2f} % of the time, within the "
                     f"{100 * WAVE_RMS:g} % (a priori) that the wave quantisation of the stream-k launch over the card's "
                     "SMs (68 on the RTX 2080 Ti) can give, which the law does not hold: one candidate cause")
        s5, t5 = "nofit", (f"ratio {fit['lof']:.2f} above {LOF_MAX:g}: {why5}; the fit's f {fit['f']:.3f} is not a "
                           "measurement of f")
    else:
        lo, hi = fit["lo"], fit["hi"]
        if hi < PRED_LO - FIT_EPS or lo > PRED_HI + FIT_EPS:
            s5 = "met"
        elif lo >= PRED_LO - FIT_EPS and hi <= PRED_HI + FIT_EPS:
            s5 = "inside"
        else:
            s5 = "undecided"
        t5 = (f"measured f {fit['f']:.3f} [{lo:.3f}, {hi:.3f}] against the predicted range {PRED_LO:.2f}..{PRED_HI:.2f}: "
              + {"met": "the interval lies wholly outside", "inside": "the interval lies wholly inside",
                 "undecided": "the interval crosses an edge"}[s5])
    print(f"  R1 (report, not a verdict): difference {r_a - least:.2f} pp to the least bucket ({R1_PP} pp for "
          "scale; it cannot separate f 0.10 from 0.30)")
    for name, state, text in (
            ("R2", "met" if r2 else "no", f"{material} of {len(sizes)} sizes against {R2_SHARE * len(sizes):.1f}"),
            ("R3", "met" if r3 else "no", f"regret {r_a:.2f} % at f {ASSUMED_F:.2f} against {r_old:.2f} % fewest tiles"),
            ("R5", s5, t5)):
        word = {"met": "MET (refutes f %.2f)" % ASSUMED_F, "undecided": "UNDECIDED",
                "nofit": "NO VERDICT on f"}.get(state, "not met")
        print(f"  {name}: {word}  ({text})")
    if r2 or r3 or s5 == "met":
        verdict = "f %.2f is REFUTED for this card by the criteria above" % ASSUMED_F
    elif s5 == "inside":
        verdict = "f %.2f stands (one campaign; not a proof)" % ASSUMED_F
    elif s5 == "undecided":
        verdict = (f"f {ASSUMED_F:.2f} is UNDECIDED: the interval of the measured f crosses an edge of "
                   f"{PRED_LO:.2f}..{PRED_HI:.2f}; measure more visits before deciding")
    else:
        verdict = (f"no verdict on f {ASSUMED_F:.2f}: {why5}; R2 and R3 find no "
                   f"fault with the table at f {ASSUMED_F:.2f}")
    print("  " + verdict)
    launch_law_report(d, data, sizes, cols, med, best)


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
    # the measured mix change is a change of mean times, i.e. the per-size changes weighted by the old
    # times; the predicted range is weighted the same way (by the old per-size medians)
    prs = [(r, statistics.median(flat(old[n]))) for r, n in ((pred_range(n - k), n) for n in mix) if r]
    if prs:
        wsum = sum(w for _, w in prs)
        lo, hi = (sum(r[i] * w for r, w in prs) / wsum for i in (0, 1))
        print(f"predicted mix change ({len(prs)} mix sizes with a range, the ranges weighted by the old medians as "
              f"the measured change of the mean is): {lo:+.1f} .. {hi:+.1f} %")
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
    # 3x the next step: chosen a priori (not fitted), a step that is no outlier among the others is no edge
    trusted = top >= 3 * max(jumps[1][0], 0.1)
    if not trusted:
        print("warning: the step is not clearly above the other steps: do not trust this interval")
    # one machine-readable line, read by the script's `all` (keep it the last line and this format)
    print(f"KPROBE k_lo={x - e} k_hi={b - e - 1} trusted={int(trusted)}")


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
