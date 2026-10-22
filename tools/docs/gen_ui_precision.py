#!/usr/bin/env python3
"""Generate src/ui/data/precision.json: what each --activations mode measured in the lab.

ninferno-ui shows the speed and the quality of each --activations mode (auto, 16, 8, 4) for the
selected card and quantization, and only where a lab measurement says so; everything else is
"no estimate". This file is that measurement, taken from docs/performance/data/bench.jsonl:

  quality   metric "kld" rows (tools/bench/quality_rows.py): the mean KL divergence of the mode's
            next-token distribution from the reference mode's (16 bits), its top-1 agreement and
            p99, and the "perplexity" rows; the reference mode has perplexity only
  speed     metric "decode" (one request after a long prompt) and "prefill" rows at concurrency
            1 (tools/bench/perf_matrix.py), under the speculation setting every measured mode of
            the pair shares (the fastest such one), else each mode's fastest

One row per (model, GPU, quantization, mode), keyed as the catalog keys them: the model by its
catalog id (the Hugging Face name, lower case), the GPU without its "NVIDIA GeForce " prefix, the
quantization by the recipe's --quant label. Each row lists the PCI device ids of its board ("pci"),
looked up by the exact board name in src/core/data/gpu_facts.json ("RTX 4090" is 0x2684 only: not
the RTX 4090 D nor the laptop chip of that name); the page matches a row to the user's card by
these ids, never by name, and a row whose board the table lacks matches no card. Older rows spell
the modes exact, fast and int8: they are 16, auto and 8.

--check exits 1 (printing a diff) when the file on disk differs from what the data generates.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs" / "performance" / "data" / "bench.jsonl"
OUT = ROOT / "src" / "ui" / "data" / "precision.json"
FACTS = ROOT / "src" / "core" / "data" / "gpu_facts.json"

MODES = {"exact": "16", "fast": "auto", "int8": "8", "auto": "auto", "16": "16", "8": "8", "4": "4"}
ORDER = {"auto": 0, "16": 1, "8": 2, "4": 3}


def mode_of(row: dict) -> str | None:
    return MODES.get(str(row.get("activations", "auto")))


def gpu_of(row: dict) -> str:
    name = row["machine"]["gpu_name"]
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def board_ids(facts: dict) -> dict[str, list[str]]:
    """Board name as gpu_of spells it ("RTX 4090") -> its PCI device ids, by exact name."""
    out: dict[str, list[str]] = defaultdict(list)
    for gpu in facts["gpus"]:
        name = gpu["name"]
        for prefix in ("NVIDIA GeForce ", "NVIDIA ", "GeForce "):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        out[name].append(gpu["id"].upper().replace("0X", "0x"))
    return {name: sorted(ids) for name, ids in out.items()}


def quant_of(row: dict) -> str:
    recipe = row.get("recipe", "")
    return recipe.split("--quant", 1)[1].strip() if "--quant" in recipe else ""


def build(lines: list[str], boards: dict[str, list[str]]) -> dict:
    # (model, gpu, quant) -> mode -> rows
    pairs: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("engine", "ninferno") != "ninferno":
            continue
        mode = mode_of(row)
        quant = quant_of(row)
        if mode is None or not quant:
            continue
        key = (row["model"].split("/")[-1].lower(), gpu_of(row), quant)
        pairs[key][mode].append(row)
    out = []
    for key in sorted(pairs):
        modes = pairs[key]
        quality = {m: [r for r in rows if r["metric"] in ("kld", "perplexity")] for m, rows in modes.items()}
        # Only pairs with a quality measurement of more than one mode compare modes at all.
        if sum(1 for rows in quality.values() if rows) < 2:
            continue
        speed = {m: [r for r in rows if r["metric"] in ("decode", "prefill") and r.get("concurrency", 1) == 1]
                 for m, rows in modes.items()}
        timed = [m for m in speed if any(r["metric"] == "decode" for r in speed[m])]
        shared = set.intersection(*[{r.get("spec", "none") for r in speed[m] if r["metric"] == "decode"} for m in timed]) \
            if len(timed) > 1 else set()

        def best(rows: list[dict], metric: str, spec: str | None) -> dict | None:
            pick = [r for r in rows if r["metric"] == metric and (spec is None or r.get("spec", "none") == spec)]
            return max(pick, key=lambda r: r["value"]) if pick else None

        spec = None
        if shared:
            spec = max(shared, key=lambda s: max((r["value"] for m in timed for r in speed[m]
                                                  if r["metric"] == "decode" and r.get("spec", "none") == s), default=0))
        for mode in sorted(modes, key=lambda m: ORDER[m]):
            entry: dict = {"model": key[0], "gpu": key[1], "pci": boards.get(key[1], []),
                           "quant": key[2], "mode": mode}
            dates = [r["date"][:10] for r in modes[mode] if r.get("date")]
            kld = next((r for r in quality.get(mode, []) if r["metric"] == "kld"), None)
            ppl = next((r for r in quality.get(mode, []) if r["metric"] == "perplexity"), None)
            if kld:
                entry["kld"] = round(kld["value"], 6)
                entry["reference"] = MODES.get(str(kld.get("reference", "exact")), "16")
                if "top1_agreement" in kld:
                    entry["top1"] = round(kld["top1_agreement"], 4)
                if "p99_kld" in kld:
                    entry["p99_kld"] = round(kld["p99_kld"], 6)
            if ppl:
                entry["perplexity"] = round(ppl["value"], 4)
                entry["corpus"] = ppl.get("corpus", "")
            for metric in ("decode", "prefill"):
                row = best(speed.get(mode, []), metric, spec) or (None if spec else best(speed.get(mode, []), metric, None))
                if row:
                    entry[metric] = round(row["value"], 1)
                    entry[metric + "_spec"] = row.get("spec", "none")
                    entry[metric + "_prompt"] = row.get("prompt_len", 0)
            entry["run"] = max(dates) if dates else ""
            out.append(entry)
    return {
        "schema": 1,
        "note": "Generated by tools/docs/gen_ui_precision.py from docs/performance/data/bench.jsonl; "
                "do not edit. Lab measurements of each --activations mode per model, GPU and "
                "quantization: quality (kld against the reference mode, top-1 agreement, perplexity) "
                "and speed (decode and prefill of one request). ninferno-ui labels a row "
                "'verified' only on a card whose PCI device id is in the row's pci list, with the "
                "same quantization, and shows 'no estimate' elsewhere.",
        "rows": out,
    }


def render(document: dict) -> str:
    head = {k: v for k, v in document.items() if k != "rows"}
    text = json.dumps(head, indent=2, ensure_ascii=False)[:-2]
    rows = ",\n".join("    " + json.dumps(r, ensure_ascii=False) for r in document["rows"])
    return text + ',\n  "rows": [\n' + rows + "\n  ]\n}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail when the file on disk is stale")
    args = ap.parse_args()
    boards = board_ids(json.loads(FACTS.read_text(encoding="utf-8")))
    text = render(build(BENCH.read_text(encoding="utf-8").splitlines(), boards))
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            sys.stdout.writelines(difflib.unified_diff(current.splitlines(True), text.splitlines(True),
                                                       str(OUT), "generated"))
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
