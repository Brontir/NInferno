#!/usr/bin/env python3
"""The facts table and the architecture limits against the driver, on this machine (#612 step 2).

Runs ninferno-facts-probe (the engine's own device query) on every visible GPU, compares each
device with its gpu_facts.json row and the synthetic class (tools/audit/facts_compare.py), prints a
table and writes a JSON summary. Exit 0: no mismatch; 1: a mismatch; 2: the probe did not run.

  python tools/audit/facts_vs_driver.py --probe build/apps/ninferno-facts-probe
  python tools/audit/facts_vs_driver.py --probe-json canned.json      # no GPU: a recorded output

Build the probe: cmake --build <build> --target ninferno-facts-probe (NINFERNO_BUILD_APPS=ON).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facts_compare  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "src" / "core" / "data" / "gpu_facts.json"


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        gpus = json.load(f)["gpus"]
    for g in gpus:
        facts_compare.check_variants(g)
    return {int(g["id"], 16): g for g in gpus}


def run(probe_output, rows):
    """(summary, text) for a probe's parsed JSON."""
    results = []
    for device in probe_output["devices"]:
        q = device["queried"]
        row = rows.get(q["pci_device_id"])
        results.append((device["index"], q["name"], facts_compare.compare_device(device, row)))
    return facts_compare.summarize(results), "\n\n".join(
        facts_compare.table_text(i, n, r) for i, n, r in results)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", help="path of ninferno-facts-probe")
    ap.add_argument("--probe-json", help="a recorded probe output instead of running it")
    ap.add_argument("--facts", default=str(FACTS))
    ap.add_argument("--json-out", help="write the JSON summary here (always printed to stdout "
                    "after the table)")
    ap.add_argument("--device", type=int, help="one CUDA ordinal")
    args = ap.parse_args(argv)
    try:
        if args.probe_json:
            probe_output = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
        elif args.probe:
            command = [args.probe] + (["--device", str(args.device)] if args.device is not None else [])
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode != 0:
                print(f"facts_vs_driver: {args.probe} failed ({done.returncode}): {done.stderr.strip()}",
                      file=sys.stderr)
                return 2
            probe_output = json.loads(done.stdout)
        else:
            ap.error("give --probe or --probe-json")
        summary, text = run(probe_output, load_rows(args.facts))
    except (OSError, ValueError, KeyError) as error:
        print(f"facts_vs_driver: {error}", file=sys.stderr)
        return 2
    print(text)
    out = json.dumps({"ok": summary["ok"], "devices": [
        {"index": d["index"], "name": d["name"], "counts": d["counts"],
         "mismatches": d["mismatches"]} for d in summary["devices"]]}, indent=1)
    print(out)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
