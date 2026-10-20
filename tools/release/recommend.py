#!/usr/bin/env python3
"""recommend.py --out FILE: the model choices ninferno-start offers, from the measured performance data.

One tab-separated line per measured GPU x model pair, from docs/performance/data (the data the
performance pages are generated from):

    gpu  gpu_name  nvidia_smi_names  sm  vram_gb  model  single  multi  concurrency  command  command_multi

single and multi are tokens per second (one stream; all streams at that concurrency) at their best
settings; command and command_multi start the server at those settings. ninferno-start shows the
lines of the user's GPU, or of the measured GPU of the same architecture closest in memory.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "docs"))
import gen_performance as perf  # noqa: E402  (the same selection as the pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gpus, models, rows, problems = perf.load()
    for p in problems:
        print(p, file=sys.stderr)
    speed, _, _ = perf.collect(rows)
    lines = []
    for g in perf.sort_present(gpus, {k[0] for k in speed}):
        for m in perf.sort_present(models, {k[1] for k in speed if k[0] == g}):
            settings = speed[(g, m)]
            s1, _ = perf.best_single(settings)
            sm, _ = perf.best_multi(settings)
            if s1 is None and sm is None:
                continue
            info = gpus[g]
            lines.append("\t".join([
                g, info["name"], "|".join(info["match"]), str(info["sm"]), str(info["vram_gb"]), models[m]["name"],
                perf.fmt(s1["value"]) if s1 else "-", perf.fmt(sm["value"]) if sm else "-",
                str(sm["concurrency"]) if sm else "-",
                perf.command(s1, 1) if s1 else "-",
                perf.command(sm, sm["concurrency"]) if sm else "-"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(lines)} choices -> {args.out}")


if __name__ == "__main__":
    main()
