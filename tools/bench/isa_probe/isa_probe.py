#!/usr/bin/env python3
"""Build and run isa_probe on this card, or parse a saved run, and write its facts into the ISA
facts table (src/core/data/isa_facts.json), one row per architecture, provenance Measured.

    isa_probe.py run   --arch 89 [--nvcc /usr/local/cuda-13.4/bin/nvcc] [--out isa_probe_89.txt]
    isa_probe.py merge --input isa_probe_89.txt [--table src/core/data/isa_facts.json]
    isa_probe.py show  [--table ...]

`run` compiles tools/bench/isa_probe/isa_probe.cu for sm_<arch> into a temporary directory, runs it
(about 20 s, the GPU otherwise idle) and saves its output; `merge` turns an output into the table's
row for the run's architecture (replacing an older row of the same architecture), recording the
card, the probe version, nvcc, driver and runtime and the date, so a fact's origin is on the row.
No fact is reasoned here: the table holds what the card said, in SM clocks.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_TABLE = os.path.join(ROOT, "src", "core", "data", "isa_facts.json")
SCHEMA = 1
# Every fact a run reports, in SM clocks unless the unit says otherwise (test_isa_facts pins it).
FACTS = (
    "resident_ctas_per_sm",
    "mma_cycles_per_sm", "mma_cycles_per_partition", "tensor_flops_per_sm_clock",
    "mma_dependent_latency", "fp32_cycles_per_warp_instruction",
    "int32_cycles_per_warp_instruction", "l1_latency_w1", "l1_latency_w2", "l1_latency_w4",
    "l1_latency_w8", "l2_latency_w1", "l2_latency_w2", "l2_latency_w4", "l2_latency_w8",
    "dram_latency_w1", "dram_latency_w2", "dram_latency_w4", "dram_latency_w8",
    # Beyond the SM the latency is wall time: the L2 and DRAM figures in ns too, and the SM clock
    # the chases ran at (so a clock that moved shows as such, not as a latency change).
    "l2_latency_ns_w1", "l2_latency_ns_w2", "l2_latency_ns_w4", "l2_latency_ns_w8",
    "dram_latency_ns_w1", "dram_latency_ns_w2", "dram_latency_ns_w4", "dram_latency_ns_w8",
    # DRAM with the TLB warm (16 MB a warp: the lines untouched, the pages mapped), against the
    # `dram` figures whose 256 MB a warp miss the TLB too.
    "dram_tlbwarm_latency_w1", "dram_tlbwarm_latency_w2", "dram_tlbwarm_latency_w4",
    "dram_tlbwarm_latency_w8", "dram_tlbwarm_latency_ns_w1", "dram_tlbwarm_latency_ns_w2",
    "dram_tlbwarm_latency_ns_w4", "dram_tlbwarm_latency_ns_w8",
    "sm_clock_mhz_chase", "shared_latency",
)
UNITS = {"tensor_flops_per_sm_clock": "flop", "resident_ctas_per_sm": "count",
         "sm_clock_mhz_chase": "mhz"}


def unit_of(name):
    """The unit a fact is reported in: SM clocks unless listed, wall ns for the `_ns_` facts."""
    return UNITS.get(name, "ns" if "_ns_" in name else "clk")
HEADER = ("probe", "card", "cc", "sms", "l2_bytes", "driver", "runtime", "nvcc", "shape")


def parse_run(text):
    """The header fields and the facts of one isa_probe output; ValueError when a line is odd or a
    fact is missing."""
    head, facts, ended, failures, checked = {}, {}, False, [], False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "fact":
            if len(parts) != 4:
                raise ValueError(f"odd fact line: {line!r}")
            facts[parts[1]] = {"value": float(parts[2]), "unit": parts[3]}
        elif parts[0] == "end":
            ended = True
        elif parts[0] == "selfcheck":
            checked = True
            if parts[1:2] != ["ok"]:
                failures.append(" ".join(parts[2:]))
        elif parts[0] in HEADER:
            head[parts[0]] = " ".join(parts[1:])
        else:
            raise ValueError(f"unknown line: {line!r}")
    missing = [f for f in FACTS if f not in facts]
    if missing or not ended or not checked:
        raise ValueError(f"incomplete run: missing {missing}, ended={ended}, selfcheck={checked}")
    if failures:
        raise ValueError("the probe's self-check failed: " + "; ".join(failures))
    for k in ("card", "cc", "nvcc", "probe", "shape"):
        if k not in head:
            raise ValueError(f"no {k} line")
    return head, facts


def row_of(head, facts):
    shape, flops = head["shape"].split()
    return {
        "provenance": "Measured",
        "card": head["card"],
        "cc": int(head["cc"]),
        "sms": int(head.get("sms", 0)),
        "l2_bytes": int(head.get("l2_bytes", 0)),
        "mma_shape": shape,
        "mma_flops": int(flops),
        "tool": {"probe": head["probe"], "nvcc": head["nvcc"], "driver": head.get("driver", ""),
                 "runtime": head.get("runtime", "")},
        "date": datetime.date.today().isoformat(),
        "facts": {name: facts[name] for name in FACTS},
    }


def load_table(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"schema": SCHEMA, "note": "", "arches": {}}


def save_table(path, table):
    with open(path, "w", newline="\n") as f:
        json.dump(table, f, indent=2, sort_keys=False)
        f.write("\n")


def cmd_run(a):
    arch = a.arch
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "isa_probe")
        src = os.path.join(HERE, "isa_probe.cu")
        cmd = [a.nvcc, "-O3", "-std=c++17", f"-arch=sm_{arch}", "-o", exe, src]
        print("+", " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
        # The probe's stderr (CUDA errors) passes through; its stdout is kept whole, so a failed
        # or refused run still shows what it measured before stopping.
        run = subprocess.run([exe], stdout=subprocess.PIPE, text=True)
        out = run.stdout
    path = a.out or f"isa_probe_{arch}.txt"
    with open(path, "w", newline="\n") as f:
        f.write(out)
    sys.stdout.write(out)
    if run.returncode != 0:
        print(f"isa_probe: the probe exited {run.returncode}; {path} holds its output, not a row "
              "to merge", file=sys.stderr)
        return run.returncode
    try:
        parse_run(out)
    except ValueError as e:
        print(f"isa_probe: {e}; {path} is not a row to merge", file=sys.stderr)
        return 3
    print(f"isa_probe: wrote {path}")
    return 0


def cmd_merge(a):
    with open(a.input) as f:
        head, facts = parse_run(f.read())
    row = row_of(head, facts)
    table = load_table(a.table)
    if table.get("schema") != SCHEMA:
        print(f"isa_probe: table schema {table.get('schema')} is not {SCHEMA}", file=sys.stderr)
        return 1
    key = arch_key(row["cc"], head["shape"])
    table["arches"][key] = row
    table["arches"] = dict(sorted(table["arches"].items(), key=lambda kv: kv[1]["cc"]))
    save_table(a.table, table)
    print(f"isa_probe: {a.table} now holds sm_{key} from {row['card']}")
    return 0


def arch_key(cc, shape):
    """The architecture label of a row: the compute capability, `120a` for the Blackwell
    accelerated set (the probe's MMA shape tells the two apart from cc alone: not needed yet)."""
    del shape
    return "120a" if cc == 120 else str(cc)


def cmd_show(a):
    table = load_table(a.table)
    for key, row in table["arches"].items():
        print(f"sm_{key}: {row['card']} ({row['mma_shape']}, {row['tool']['nvcc']}, {row['date']})")
        for name in FACTS:
            v = row["facts"][name]
            print(f"  {name:36s} {v['value']:10.3f} {v['unit']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arch", required=True, help="sm_<arch> to compile for (70, 75, 86, 89, 120a)")
    r.add_argument("--nvcc", default="nvcc")
    r.add_argument("--out")
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("merge")
    m.add_argument("--input", required=True)
    m.add_argument("--table", default=DEFAULT_TABLE)
    m.set_defaults(fn=cmd_merge)
    s = sub.add_parser("show")
    s.add_argument("--table", default=DEFAULT_TABLE)
    s.set_defaults(fn=cmd_show)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
