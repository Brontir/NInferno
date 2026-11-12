#!/usr/bin/env python3
"""The counted clocks of one MMQ unit per architecture (#785 PR-B), from the kernel-facts
snapshots' mmq sections (kernel_facts.py; no GPU), against the measured units.

A unit is one iteration of mul_mat_q's K loop: a 128-row x J-column tile over 256 K values
(core/generation.h MmqLaunchLaw). The kernel runs 8 warps on the SM's 4 schedulers, two warps a
scheduler, so a scheduler issues every counted instruction of two warps per unit. Per scheduler
and unit, in SM clocks:
  issue    2 x total: one instruction a clock
  tensor   2 x IMMA x the clocks one IMMA holds the partition's tensor core: its MACs over the
           generation's int8 rate (core/generation.h s8, flops an SM-clock, a quarter of them per
           partition): m8n8k16 4 on Turing; m16n8k32 16 (m16n8k16 8) on GA10x, Ada and GB20x; half
           those on GA100, Hopper and GB100
  alu      the INT32 and FP32 datapaths, 2 x: an INT32 instruction 2 clocks on every generation
           (16 lanes; isa_facts int32_cycles_per_warp_instruction 2.0 on sm_75, 86, 89, 120a), an
           FP32-pipe one (FFMA, FMUL, FADD, I2FP and the rest of the float class but I2F) 2 clocks
           on Turing and GA100, 1 on the others (isa_facts fp32_cycles_per_warp_instruction);
           where the INT32 lanes are half of the FP32 ones (GA10x, Ada, GB20x: 16 FP32 lanes
           beside 16 FP32 / INT32 lanes) both share a datapath, max(2 int, int + fp); elsewhere
           max(2 int, fp x its clocks)
  xu       2 x I2F x 8: I2F (not I2FP) on the multi-function path, 16 results an SM-clock (the CUDA
           C++ Programming Guide's throughput table, conversions); a listed figure, not measured
  shared   per SM: 8 warps x (LDS + STS bytes) / 128 bytes an SM-clock
  bound    the largest of them (the pipes overlap); 2 x stall: the two warps' static schedules
           (ptxas's stall counts, stall_sum) issued one after the other
  DRAM     a measured board's weight bytes of a unit (128 rows x 256 values of the format: Q4_K
           18432, Q5_K 22528, Q6_K 26880) at its bus (src/core/data/gpu_facts.json: bus bits x
           Gbps) over its SMs and the clock the unit is given at; the activation tile (J x 288
           bytes) is the L2's
The measured units (SM clocks at the clock the board ran): the RTX 2080 Ti's (DECISIONS 78, a tile
a CTA on 64 CTAs: 4.56 us at J 32, 11.60 at J 128, at the 1.93-1.97 GHz the board ran, 1.95 GHz
taken; core/generation.h kTuringMmqLaw keeps them at the 1545 MHz reference clock, 7047 and 17920);
the RTX 4070 Ti's at full load (#785 plan: swiglu 15360x3840, 60 tile-a-CTA units in about 160 us
at 2.61 GHz, J 32: about 6960); the RTX 3090's at full load (5600-7500 at J 32, #785 plan, given at
its 1695 MHz boost clock).

    mmq_units.py [--facts DIR] [--types 12,13,14] [--widths 16,32,64,128]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_facts  # noqa: E402

FACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "ops",
                     "generic", "data", "kernel_facts")
TYPES = {12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
WEIGHT_BYTES = {12: 128 * 144, 13: 128 * 176, 14: 128 * 210}  # a unit's 128 rows x 256 values
# Per build label: the generation, the int8 MMA rate (core/generation.h s8, flops an SM-clock),
# FP32 clocks a warp instruction, whether the INT32 lanes share the FP32 datapath.
ARCHS = {
    "75": ("Turing", 2048, 2, False),
    "80": ("GA100", 4096, 2, False),
    "86": ("GA10x", 2048, 1, True),
    "89": ("Ada", 2048, 1, True),
    "90": ("Hopper", 4096, 1, False),
    "100": ("GB100", 4096, 1, False),
    "120a": ("GB20x", 2048, 1, True),
}
SHAPE_MACS = {"IMMA.8816": 8 * 8 * 16, "IMMA.16816": 16 * 8 * 16, "IMMA.16832": 16 * 8 * 32}
I2F_CLOCKS = 8
INT_CLOCKS = 2
SHARED_BYTES_PER_CLOCK = 128
WARPS = 8
WARPS_PER_SCHEDULER = 2
# The measured units, Q4_K: (label, J, the board's PCI id in src/core/data/gpu_facts.json (its
# name, SMs and bus), what was measured, SM clocks low, high, the clock the unit is given at).
MEASURED = (
    ("75", 32, "0x1E04", "64 CTAs", 4.56e-6 * 1.95e9, None, 1.95e9),
    ("75", 128, "0x1E04", "64 CTAs", 11.60e-6 * 1.95e9, None, 1.95e9),
    ("89", 32, "0x2782", "full load", 160e-6 / 60 * 2.61e9, None, 2.61e9),
    ("86", 32, "0x2204", "full load", 5600, 7500, 1.695e9),
)
GPU_FACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "core",
                         "data", "gpu_facts.json")


def board(pci_id):
    """(name, SMs, bus bytes a second) of a board of the registry."""
    with open(GPU_FACTS) as f:
        for g in json.load(f)["gpus"]:
            if g["id"].lower() == pci_id.lower():
                return g["name"], g["sms"], g["bus_bits"] * g["gbps"] * 1e9 / 8
    raise KeyError(pci_id)


def mma_clocks(shape, s8):
    """Partition clocks one IMMA of `shape` holds the tensor core at the int8 rate `s8`."""
    for prefix, macs in SHAPE_MACS.items():
        if shape.startswith(prefix + "."):
            return macs / (s8 / 2 / 4)
    return float("nan")


def unit(label, loop):
    """{term: SM clocks per scheduler a unit} of one MMQ row's loop counts (module docstring)."""
    _, s8, fp_clocks, shared_datapath = ARCHS[label]
    w = WARPS_PER_SCHEDULER
    fp = loop.fp - loop.i2f  # the FP32-pipe instructions: FFMA, FMUL, FADD, I2FP, FSEL, ...
    if shared_datapath:
        alu = max(INT_CLOCKS * loop.int, loop.int + fp * fp_clocks)
    else:
        alu = max(INT_CLOCKS * loop.int, fp * fp_clocks)
    out = {
        "issue": w * loop.total,
        "tensor": w * loop.mma * mma_clocks(loop.mma_shape, s8),
        "alu": w * alu,
        "xu": w * loop.i2f * I2F_CLOCKS,
        "shared": WARPS * (loop.lds_bytes + loop.sts_bytes) / SHARED_BYTES_PER_CLOCK,
        "stall": w * loop.stall_sum,
    }
    out["bound"] = max(out[k] for k in ("issue", "tensor", "alu", "xu", "shared"))
    return out


def snapshots(facts):
    """{label: (mmq section nvcc, {key: row})} of the snapshots with an mmq section."""
    out = {}
    for path in sorted(os.listdir(facts)):
        if not (path.startswith("sm_") and path.endswith(".txt")):
            continue
        with open(os.path.join(facts, path)) as f:
            header, sections = kernel_facts.parse_sections(f.read())
        if "mmq" in sections and header["arch"] in ARCHS:
            out[header["arch"]] = (sections["mmq"]["nvcc"], sections["mmq"]["rows"])
    return out


def loop_of(rows, t, J):
    v = rows.get((t, J, 0, 30))
    return v[4] if v is not None and v[4] is not None and v[4].loops > 0 else None


def counted_table(snaps, types, widths):
    yield ("| arch | nvcc | format | J | registers | instr / warp | IMMA | INT | FP32 pipe | I2F | "
           "stall_sum | issue | tensor | alu | xu | shared | bound | 2 x stall |")
    yield "|---" * 18 + "|"
    order = [a for a in ARCHS if a in snaps]
    for t in types:
        for J in widths:
            for label in order:
                nvcc, rows = snaps[label]
                loop = loop_of(rows, t, J)
                if loop is None:
                    continue
                u = unit(label, loop)
                regs = rows[(t, J, 0, 30)][0]
                yield (f"| sm_{label} {ARCHS[label][0]} | {nvcc} | {TYPES.get(t, t)} | {J} | "
                       f"{regs} | {loop.total} | {loop.mma} {loop.mma_shape.split('.')[1]} | "
                       f"{loop.int} | {loop.fp - loop.i2f} | {loop.i2f} | {loop.stall_sum} | "
                       f"{u['issue']:.0f} | {u['tensor']:.0f} | {u['alu']:.0f} | {u['xu']:.0f} | "
                       f"{u['shared']:.0f} | **{u['bound']:.0f}** | {u['stall']:.0f} |")


def ratio(lo, hi, base):
    return f"{lo / base:.2f}" if hi is None else f"{lo / base:.2f}-{hi / base:.2f}"


def decision_table(snaps):
    yield ("| board (arch) | J | measured unit | counted bound | measured / bound | 2 x stall | "
           "measured / 2 x stall | DRAM (weights) | bound + DRAM | measured / (bound + DRAM) | "
           "measured / (2 x stall + DRAM) |")
    yield "|---" * 11 + "|"
    for label, J, pci_id, what, lo, hi, clock in MEASURED:
        if label not in snaps:
            continue
        name, sms, bus = board(pci_id)
        card = f"{name.replace('GeForce ', '')}, {what}"
        loop = loop_of(snaps[label][1], 12, J)
        if loop is None:
            continue
        u = unit(label, loop)
        dram = WEIGHT_BYTES[12] / (bus / sms / clock)
        meas = f"{lo:.0f}" if hi is None else f"{lo:.0f}-{hi:.0f}"
        yield (f"| {card} (sm_{label}) | {J} | {meas} | {u['bound']:.0f} | "
               f"{ratio(lo, hi, u['bound'])} | {u['stall']:.0f} | {ratio(lo, hi, u['stall'])} | "
               f"{dram:.0f} | {u['bound'] + dram:.0f} | {ratio(lo, hi, u['bound'] + dram)} | "
               f"{ratio(lo, hi, u['stall'] + dram)} |")


W4A8_TILE = (32, 128, 2, 4, 0)  # block_m, block_n, warps_m, warps_n, the store epilogue
W4A8_CODECS = (("Q4_K", "W4a8Q45K<4>"), ("Q5_K", "W4a8Q45K<5>"), ("Q6_K", "W4a8Q6K"))


def w4a8_table(facts):
    """The W4A8 32 x 128 tile's counted 128-value K tile per format (the plan's format term:
    measured, a Q6_K CTA takes about 1.5 x a Q4_K one's time per K on the RTX 4070 Ti)."""
    yield ("| arch | format | registers | instr / warp | IMMA | INT | FP32 | stall_sum | "
           "instr vs Q4_K | stall vs Q4_K |")
    yield "|---" * 10 + "|"
    for label in ARCHS:
        path = os.path.join(facts, f"sm_{label}.txt")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            _, sections = kernel_facts.parse_sections(f.read())
        if "w4a8" not in sections:
            continue
        rows = sections["w4a8"]["rows"]
        base = None
        for name, codec in W4A8_CODECS:
            found = [v for k, v in rows.items() if k[0] == codec and k[1:6] == W4A8_TILE and
                     k[7:] == (0, 0) and v[4] is not None and v[4].loops == 1]
            if not found:
                continue
            regs, c = found[0][0], found[0][4]
            base = base or c
            yield (f"| sm_{label} {ARCHS[label][0]} | {name} | {regs} | {c.total} | {c.mma} | "
                   f"{c.int} | {c.fp} | {c.stall_sum} | {c.total / base.total:.2f} | "
                   f"{c.stall_sum / base.stall_sum:.2f} |")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--facts", default=FACTS, help="the snapshot directory")
    p.add_argument("--types", default="12,13,14", help="ggml_type values (12 Q4_K, 14 Q6_K)")
    p.add_argument("--widths", default="16,32,64,128", help="column tiles")
    args = p.parse_args(argv)
    snaps = snapshots(args.facts)
    types = [int(v) for v in args.types.split(",")]
    widths = [int(v) for v in args.widths.split(",")]
    print("Counted, per scheduler and unit (SM clocks):\n")
    print("\n".join(counted_table(snaps, types, widths)))
    print("\nAgainst the measured units (Q4_K):\n")
    print("\n".join(decision_table(snaps)))
    print("\nThe W4A8 32 x 128 tile, one 128-value K tile (store epilogue):\n")
    print("\n".join(w4a8_table(args.facts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
