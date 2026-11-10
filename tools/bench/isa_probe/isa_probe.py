#!/usr/bin/env python3
"""Build and run isa_probe on this card, or parse a saved run, and write its facts into the ISA
facts table (src/core/data/isa_facts.json), one row per architecture, provenance Measured.

    isa_probe.py run        --arch 89 [--nvcc /usr/local/cuda-13.4/bin/nvcc] [--out isa_probe_89.txt]
    isa_probe.py merge      --input isa_probe_89.txt [--inputs src/core/data/isa_probe] [--table ...]
    isa_probe.py regenerate [--inputs src/core/data/isa_probe] [--table src/core/data/isa_facts.json]
    isa_probe.py show       [--table ...]

`run` compiles tools/bench/isa_probe/isa_probe.cu for sm_<arch> into a temporary directory, runs it
(about a minute, the GPU otherwise idle) and saves its output; `merge` copies that output into the
committed inputs directory as isa_probe_<arch>.txt (replacing an older run of the same
architecture) and regenerates the table from every file there; `regenerate` alone rebuilds the
table from the inputs. Adding an architecture is one file plus a regenerate. A row records its
source file, the card, the probe version, nvcc, driver and runtime, so a fact's origin is on the
row. No fact is reasoned here: the table holds what the card said.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_TABLE = os.path.join(ROOT, "src", "core", "data", "isa_facts.json")
DEFAULT_INPUTS = os.path.join(ROOT, "src", "core", "data", "isa_probe")
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
    # Dependent (issue-to-use) latencies of the decode chains' ops, one warp a chain; the
    # conversion pairs per op.
    "lat_imad", "lat_iadd", "lat_lop3", "lat_prmt", "lat_shf", "lat_hfma2", "lat_ffma", "lat_shfl",
    "lat_cvt_i2f_f2i", "lat_cvt_f32_f16",
)
# Facts a row may lack: those a later probe added (the dependent ALU latencies, probe 11), so the
# rows measured by earlier probes stay valid until their card is probed again.
OPTIONAL_FACTS = frozenset(f for f in FACTS if f.startswith("lat_"))
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
        elif parts[0] == "note":
            # Information the probe adds (the SM clock's range over the chases): kept, not a fact.
            head.setdefault("notes", []).append(" ".join(parts[1:]))
        elif parts[0] in HEADER:
            head[parts[0]] = " ".join(parts[1:])
        else:
            raise ValueError(f"unknown line: {line!r}")
    missing = [f for f in FACTS if f not in facts and f not in OPTIONAL_FACTS]
    if missing or not ended or not checked:
        raise ValueError(f"incomplete run: missing {missing}, ended={ended}, selfcheck={checked}")
    if failures:
        raise ValueError("the probe's self-check failed: " + "; ".join(failures))
    for k in ("card", "cc", "nvcc", "probe", "shape"):
        if k not in head:
            raise ValueError(f"no {k} line")
    return head, facts


# Flops per warp instruction of each MMA shape the probe issues: 2 x m x n x k, and on Volta four
# of them (one mma.sync.m8n8k4 performs four independent 8x8x4 products, one per quad pair: PTX
# ISA). The row reads this table, so a probe output that printed another figure (probes before 10
# printed 512 for the Volta shape) still gives the right rate; the measured cycles per MMA stay.
SHAPE_FLOPS = {"hmma_884_f32_f16": 2048, "hmma_1688_f32_f16": 2048, "hmma_16816_f32_bf16": 4096}


def row_of(head, facts, source):
    """The table's row of one run; `source` names the committed probe output it came from. The
    tensor rate is derived here: the shape's flops over the measured cycles per MMA per SM."""
    shape, printed = head["shape"].split()
    flops = SHAPE_FLOPS.get(shape, int(printed))
    facts = dict(facts)
    facts["tensor_flops_per_sm_clock"] = {
        "value": round(flops / facts["mma_cycles_per_sm"]["value"], 3), "unit": "flop"}
    return {
        "provenance": "Measured",
        "source": source,
        "card": head["card"],
        "cc": int(head["cc"]),
        "sms": int(head.get("sms", 0)),
        "l2_bytes": int(head.get("l2_bytes", 0)),
        "mma_shape": shape,
        "mma_flops": flops,
        "tool": {"probe": head["probe"], "nvcc": head["nvcc"], "driver": head.get("driver", ""),
                 "runtime": head.get("runtime", "")},
        "notes": head.get("notes", []),
        "facts": {name: facts[name] for name in FACTS if name in facts},
    }


NOTE = ("ISA facts per architecture, regenerated by `tools/bench/isa_probe/isa_probe.py "
        "regenerate` from the committed probe outputs in src/core/data/isa_probe/ (one "
        "isa_probe_<arch>.txt per architecture, written by `isa_probe.py run` on one card of it; "
        "never edited by hand). In SM clocks unless the unit says otherwise: tensor pipe cycles "
        "per MMA at the SM and per partition, the dependent MMA latency, INT32 / FP32 cycles per "
        "warp instruction, dependent L1 / L2 / DRAM load latencies alone and under 2, 4, 8 chasing "
        "warps per SM (the L2 and DRAM ones in wall ns too, with the SM clock of the chase; DRAM "
        "with the TLB warm and past it), shared-memory latency. Every row names its source file, "
        "the card, the probe version, nvcc, driver and runtime; provenance is Measured for all. "
        "tests/test_isa_facts.py pins the schema and that the table equals its regeneration.")


def build_table(inputs_dir):
    """The whole table from the probe outputs in `inputs_dir` (isa_probe_*.txt), rows in order of
    compute capability; ValueError names a file that is no row."""
    arches = {}
    for name in sorted(os.listdir(inputs_dir)):
        if not (name.startswith("isa_probe_") and name.endswith(".txt")):
            continue
        with open(os.path.join(inputs_dir, name)) as f:
            try:
                head, facts = parse_run(f.read())
            except ValueError as e:
                raise ValueError(f"{name}: {e}") from None
        row = row_of(head, facts, name)
        key = arch_key(row["cc"], head["shape"])
        if key in arches:
            raise ValueError(f"{name}: sm_{key} is already given by {arches[key]['source']}")
        arches[key] = row
    arches = dict(sorted(arches.items(), key=lambda kv: kv[1]["cc"]))
    return {"schema": SCHEMA, "note": NOTE, "arches": arches}


def load_table(path):
    with open(path) as f:
        return json.load(f)


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


def cmd_regenerate(a):
    try:
        table = build_table(a.inputs)
    except ValueError as e:
        print(f"isa_probe: {e}", file=sys.stderr)
        return 1
    save_table(a.table, table)
    print(f"isa_probe: {a.table} regenerated from {a.inputs}: "
          + ", ".join(f"sm_{k} ({r['card']})" for k, r in table["arches"].items()))
    return 0


def cmd_merge(a):
    """One file in, one regenerate: the run's output is copied into the inputs directory under its
    architecture's name (replacing an older run of the same architecture) and the table is
    regenerated from all of them."""
    with open(a.input) as f:
        head, facts = parse_run(f.read())
    key = arch_key(int(head["cc"]), head["shape"])
    dest = os.path.join(a.inputs, f"isa_probe_{key}.txt")
    with open(a.input) as src, open(dest, "w", newline="\n") as out:
        out.write(src.read())
    print(f"isa_probe: {a.input} -> {dest}")
    return cmd_regenerate(a)


def arch_key(cc, shape):
    """The architecture label of a row: the compute capability, `120a` for the Blackwell
    accelerated set (the probe's MMA shape tells the two apart from cc alone: not needed yet)."""
    del shape
    return "120a" if cc == 120 else str(cc)


def cmd_show(a):
    table = load_table(a.table)
    for key, row in table["arches"].items():
        print(f"sm_{key}: {row['card']} ({row['mma_shape']}, nvcc {row['tool']['nvcc']}, "
              f"{row['source']})")
        for name in FACTS:
            v = row["facts"].get(name)
            if v is None:
                print(f"  {name:36s} {'-':>10s} (not measured by this row's probe)")
                continue
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
    m.add_argument("--inputs", default=DEFAULT_INPUTS)
    m.add_argument("--table", default=DEFAULT_TABLE)
    m.set_defaults(fn=cmd_merge)
    g = sub.add_parser("regenerate")
    g.add_argument("--inputs", default=DEFAULT_INPUTS)
    g.add_argument("--table", default=DEFAULT_TABLE)
    g.set_defaults(fn=cmd_regenerate)
    s = sub.add_parser("show")
    s.add_argument("--table", default=DEFAULT_TABLE)
    s.set_defaults(fn=cmd_show)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
