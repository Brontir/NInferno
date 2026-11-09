#!/usr/bin/env python3
"""Forced-width grid and A/B helpers for the MMQ column-tile work table (DECISIONS 74, #524).

measure  one running ninferno-serve: for each prompt size N, `--reps` requests of exactly N prompt
         tokens (chat, max_tokens 1, stream, time to the first content chunk), the median in ms;
         appends one JSON line {"label", "n", "ms": [...]} per size to --out.
grid     reads the per-width files of `measure` (label "J<w>") and prints the regret of the model's
         picks against the best width for each f of --fs (0.43 = DECISIONS 69, "old" = fewest tiles).
ab       reads files labelled old/new and prints the per-size change and the mix's.

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
import urllib.request
from collections import defaultdict
from pathlib import Path

SIZES = [16, 32, 48, 64, 80, 96, 112, 128, 136, 144, 160, 176, 192, 208, 224, 256, 288, 320, 384, 448, 512]
AB_SIZES = [96, 128, 136, 144, 160, 176, 192, 208, 224, 240, 256, 264, 288, 320, 384, 448, 512]
MIX = [144, 160, 176, 192, 208, 224, 240, 256]
WIDTHS = [16, 32, 48, 64, 80, 96, 112, 128]
PRODUCTION = [16, 32, 64, 80, 96, 128]
OLD = [16, 32, 64, 128]


def pick(n: int, f: float | None, widths: list[int]) -> int:
    best, least = 0, 1e300
    for j in widths:
        t = -(-n // j)
        cost = t if f is None else t * (f + (1 - f) * j / 128)
        if cost < least - 1e-12:
            best, least = j, cost
    return best


def post(url: str, body: dict) -> urllib.request.addinfourl:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def prompt_tokens(base: str, model: str, words: int) -> int:
    body = {"model": model, "messages": [{"role": "user", "content": "a " * words}], "max_tokens": 1}
    with post(base + "/v1/chat/completions", body) as r:
        return json.load(r)["usage"]["prompt_tokens"]


def ttft_ms(base: str, model: str, words: int, salt: int) -> float:
    # a distinct first word per request: no prefix is shared (the server runs --no-prefix-reuse too)
    content = f"w{salt} " + "a " * words
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1, "stream": True}
    t0 = time.perf_counter()
    with post(base + "/v1/chat/completions", body) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line:
                dt = (time.perf_counter() - t0) * 1e3
                r.read()
                return dt
    raise RuntimeError("no content chunk")


def measure(a: argparse.Namespace) -> None:
    base = f"http://{a.host}:{a.port}"
    model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=60))["data"][0]["id"]
    overhead = prompt_tokens(base, model, 1) - 1  # template tokens around the words
    sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else SIZES
    out = Path(a.out)
    salt = int(time.time()) % 100000 * 1000
    for rep in range(a.warmup):
        ttft_ms(base, model, 128, salt + rep)
    for n in sizes:
        words = n - overhead - 1  # "wNNN" is one more token
        got = prompt_tokens(base, model, words)
        if abs(got - n) > 1:
            words += n - got
        times = [ttft_ms(base, model, words, salt + 10 * n + i) for i in range(a.reps)]
        with out.open("a") as f:
            f.write(json.dumps({"label": a.label, "n": n, "ms": times}) + "\n")
        print(f"{a.label} N={n} median {statistics.median(times):.2f} ms", flush=True)


def load(paths: list[str]) -> dict[str, dict[int, list[float]]]:
    d: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            d[r["label"]][r["n"]] += r["ms"]
    return d


def grid(a: argparse.Namespace) -> None:
    d = load(a.files)
    med = {j: {n: statistics.median(d[f"J{j}"][n]) for n in d[f"J{j}"]} for j in WIDTHS if f"J{j}" in d}
    sizes = sorted(set.intersection(*[set(m) for m in med.values()]))
    best = {n: min(m[n] for m in med.values()) for n in sizes}
    print("N  best_width  " + " ".join(f"J{j:<4}" for j in med))
    for n in sizes:
        row = " ".join(f"{med[j][n]:6.2f}" for j in med)
        print(f"{n:3d} J{min(med, key=lambda j: med[j][n]):<3d}  {row}")

    def regret(p) -> float:
        usable = [n for n in sizes if p(n) in med]
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
    old, new = d["old"], d["new"]
    for n in sorted(old):
        o, w = statistics.median(old[n]), statistics.median(new[n])
        print(f"N={n:3d} old {o:7.2f} new {w:7.2f} ms {100 * (w / o - 1):+6.1f} %")
    mo = statistics.mean(statistics.median(old[n]) for n in MIX if n in old)
    mw = statistics.mean(statistics.median(new[n]) for n in MIX if n in new)
    print(f"mix {MIX}: old {mo:.2f} new {mw:.2f} ms {100 * (mw / mo - 1):+.1f} %")


def main() -> None:
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
    g.add_argument("--fs", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.43])
    g.set_defaults(fn=grid)
    b = sub.add_parser("ab")
    b.add_argument("files", nargs="+")
    b.set_defaults(fn=ab)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
