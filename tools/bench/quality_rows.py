#!/usr/bin/env python3
"""Measure what the faster activation modes cost in quality, the way docs/performance/ shows it.

    python3 tools/bench/quality_rows.py --artifact MODEL.ninferno --model ORG/NAME \
        --recipe "org/repo --quant LABEL" --jsonl docs/performance/data/bench.jsonl

Runs ninferno-perplexity on the fixed corpus (eval/corpora/perplexity-1m, --quick by default) once per
--activations mode. The first mode (16 by default) is the reference: it records its top-K tokens
per position, and every other mode is scored against them. Rows appended per mode:

- metric "perplexity": the overall perplexity of that mode;
- metric "kld" (not for the reference): the mean KL divergence from the reference per scored
  position, with the share of positions whose top-1 token agrees ("top1_agreement").

The model is the same converted file in every mode, so the rows isolate the activation precision.
"""
import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_matrix import machine  # noqa: E402  (the same machine record as the speed rows)


def run(binary, artifact, mode, args, out_dir, kld_out=None, kld=None):
    cmd = [str(binary), str(artifact), "--corpus", str(args.corpus), "--context", str(args.context),
           "--stride", str(args.stride), "--activations", mode, "--device", str(args.device),
           "--output", str(out_dir)]
    if args.quick:
        cmd.append("--quick")
    if args.kv_dtype:
        cmd += ["--kv-dtype", args.kv_dtype]
    if kld_out:
        cmd += ["--kld-out", str(kld_out)]
    if kld:
        cmd += ["--kld", str(kld)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads((out_dir / "report.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=Path, required=True, help="the converted .ninferno file")
    ap.add_argument("--perplexity", type=Path, default=ROOT / "build/apps/ninferno-perplexity")
    ap.add_argument("--modes", default="16,auto,8,4", help="--activations modes; the first is the reference")
    ap.add_argument("--corpus", type=Path, default=ROOT / "eval/corpora/perplexity-1m/manifest.json")
    ap.add_argument("--full", dest="quick", action="store_false", help="every corpus stream, not one per domain")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--kv-dtype", help="the KV format; default: ninferno-perplexity's")
    ap.add_argument("--device", type=int, default=0)
    rows = ap.add_argument_group("bench.jsonl rows")
    rows.add_argument("--jsonl", type=Path, required=True, help="append the rows to this file")
    rows.add_argument("--model", required=True, help="the Hugging Face checkpoint the file was converted from")
    rows.add_argument("--recipe", required=True, help="the exact source, as in the speed rows")
    rows.add_argument("--version", help="the NInferno release or public commit")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if len(set(modes)) != len(modes) or not modes:
        ap.error("--modes needs distinct activation modes")

    reports = {}
    with tempfile.TemporaryDirectory(prefix="ninferno-quality-") as tmp:
        tmp = Path(tmp)
        reference = tmp / "reference.kld"
        for i, mode in enumerate(modes):
            out = tmp / f"report-{mode}"
            out.mkdir()
            reports[mode] = run(args.perplexity, args.artifact, mode, args, out,
                                kld_out=reference if i == 0 else None, kld=reference if i else None)

    first = reports[modes[0]]
    common = {"machine": machine(args.device), "model": args.model, "recipe": args.recipe,
              "engine": "ninferno", "corpus": first["corpus"]["id"] + ("/quick" if args.quick else ""),
              "context": args.context, "stride": args.stride, "kv": first["execution"]["kv_dtype"],
              "unit": "",
              "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if args.version:
        common["version"] = args.version
    out_rows = []
    for mode in modes:
        rep = reports[mode]
        out_rows.append({**common, "metric": "perplexity", "activations": mode,
                         "value": rep["overall"]["perplexity"], "scored_tokens": rep["overall"]["scored_tokens"]})
        if mode != modes[0]:
            k = rep["kld"]["overall"]
            out_rows.append({**common, "metric": "kld", "activations": mode, "reference": modes[0],
                             "value": k["mean_kld"], "top1_agreement": k["top1_agreement"],
                             "p99_kld": k["p99_kld"]})
    with args.jsonl.open("a", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    for row in out_rows:
        extra = f" top-1 {row['top1_agreement'] * 100:.2f} %" if row["metric"] == "kld" else ""
        print(f"{row['activations']:>6} {row['metric']:>10} {row['value']:.6g}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
