#!/usr/bin/env python3
"""The facts table and the architecture limits against the driver, on this machine (#612 step 2).

Runs ninferno-facts-probe (the engine's own device query) on every visible GPU, compares each
device with its gpu_facts.json row and the synthetic class (tools/audit/facts_compare.py), prints a
table and writes a JSON summary. Exit 0: no mismatch; 1: a mismatch, or with --strict a field the
driver did not answer (listed as "unverified" in the summary); 2: the probe did not run or its
output is invalid. facts_compare.py's docstring says what is and is not compared.

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
PROBE_TIMEOUT_S = 300


def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:  # a BOM (a Windows redirect) is accepted
        document = json.load(f)
    gpus = document.get("gpus") if isinstance(document, dict) else None
    if not isinstance(gpus, list) or not gpus:
        raise ValueError(f"gpu facts: {path} is not an object with a nonempty \"gpus\" list")
    rows = {}
    for g in gpus:
        facts_compare.check_row(g)
        key = int(g["id"], 16)
        if key in rows:
            raise ValueError(f"gpu facts: id {g['id']} appears twice")
        rows[key] = g
    return rows


def run(probe_output, rows):
    """(summary, text) for a probe's parsed JSON."""
    facts_compare.validate_probe(probe_output)  # ValueError: exit 2, never a clean audit
    results = []
    for device in probe_output["devices"]:
        q = device.get("queried", {})  # validate_probe: an object (an error device too)
        row = None if "error" in device else rows.get(q.get("pci_device_id"))
        name = q.get("name", "?")
        if device.get("pci_bus_id"):
            name += f" [{device['pci_bus_id']}]"
        results.append((device["index"], name, facts_compare.compare_device(device, row)))
    return facts_compare.summarize(results), "\n\n".join(
        facts_compare.table_text(i, n, r) for i, n, r in results)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", help="path of ninferno-facts-probe")
    ap.add_argument("--probe-json", help="a recorded probe output instead of running it")
    ap.add_argument("--facts", default=str(FACTS))
    ap.add_argument("--json-out", help="write the JSON summary here (always printed to stdout "
                    "after the table)")
    ap.add_argument("--device", type=int, help="one CUDA ordinal (with --probe)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when a field is unverified (the driver gave no value)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        # A device name outside the console's code page must not end the run with a traceback.
        try:
            stream.reconfigure(errors="backslashreplace")
        except AttributeError:  # a captured stream without reconfigure
            pass
    if bool(args.probe) == bool(args.probe_json):
        ap.error("give --probe or --probe-json, not both and not neither")
    if args.device is not None and not args.probe:
        ap.error("--device goes with --probe")
    try:
        if args.probe_json:
            probe_output = json.loads(Path(args.probe_json).read_text(encoding="utf-8-sig"))
        else:
            command = [args.probe] + (["--device", str(args.device)] if args.device is not None else [])
            try:
                done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                      timeout=PROBE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(f"facts_vs_driver: {args.probe} did not finish in {PROBE_TIMEOUT_S} s",
                      file=sys.stderr)
                return 2
            if done.returncode != 0:
                print(f"facts_vs_driver: {args.probe} failed ({done.returncode}): {done.stderr.strip()}",
                      file=sys.stderr)
                return 2
            probe_output = json.loads(done.stdout)
        summary, text = run(probe_output, load_rows(args.facts))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, RecursionError) as error:
        # TypeError, KeyError: a backstop for a shape validate_probe does not name; ArithmeticError:
        # a number too large for a float; RecursionError: deeply nested JSON
        print(f"facts_vs_driver: {error}", file=sys.stderr)
        return 2
    print(text)
    out = json.dumps({"ok": summary["ok"], "unverified": summary["unverified"], "devices": [
        {"index": d["index"], "name": d["name"], "counts": d["counts"],
         "mismatches": d["mismatches"]} for d in summary["devices"]]}, indent=1)
    print(out)
    if summary["unverified"]:
        print(f"facts_vs_driver: {len(summary['unverified'])} fields unverified (the driver gave "
              "no value)", file=sys.stderr)
    if not summary["ok"]:
        return 1
    return 1 if args.strict and summary["unverified"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
