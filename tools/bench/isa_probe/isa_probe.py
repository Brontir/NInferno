#!/usr/bin/env python3
"""Build and run isa_probe on this card, or parse a saved run, and write its facts into the ISA
facts table (src/core/data/isa_facts.json), one row per architecture, provenance Measured.

    isa_probe.py run        --arch 89 [--nvcc /usr/local/cuda-13.4/bin/nvcc] [--out isa_probe_89.txt]
    isa_probe.py merge      --input isa_probe_89.txt [--inputs src/core/data/isa_probe] [--table ...]
    isa_probe.py regenerate [--inputs src/core/data/isa_probe] [--table src/core/data/isa_facts.json]
    isa_probe.py show       [--table ...]

`run` compiles tools/bench/isa_probe/isa_probe.cu for sm_<arch> into a temporary directory, runs it
(10 to 20 s, the GPU otherwise idle) and saves its output, headed by the source's sha256, the
build's -arch and the SASS MMA instructions of one mma.sync in the build (cuobjdump), and followed
by notes of the compute processes on the probe's GPU before, during and after (a run with another
one is no row) and of the clocks nvidia-smi saw during the run; a refused run's output goes to
<out>.refused instead, which `merge` does not take, and from probe 17 `merge` refuses a run whose
notes do not show its GPU idle wherever the file came from; `merge` copies that output into the
committed inputs directory as isa_probe_<arch>.txt (replacing an older run of the same
architecture) and regenerates the table from every file there; `regenerate` alone rebuilds the
table from the inputs. Adding an architecture is one file plus a regenerate. A row records its
source file, the card, the probe version, nvcc, driver and runtime, so a fact's origin is on the
row. No fact is reasoned here: the table holds what the card said.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
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
    # The tensor pipe, per PTX mma.sync of the row's mma_shape (one warp instruction of the PTX):
    # on sm_70 one mma.sync.m8n8k4 is four SASS HMMA.884 STEPs, elsewhere one HMMA, as the row's
    # sass_per_mma says (read from the build's SASS by `run`). A price per SASS MMA is these
    # cycles over sass_per_mma; tensor_flops_per_sm_clock is the same in either unit.
    "mma_cycles_per_sm", "mma_cycles_per_partition", "tensor_flops_per_sm_clock",
    # The same tensor loop with one CTA of 4, 8, 16 warps on the SM (1, 2, 4 warps a scheduler
    # when the SM spreads them evenly, which the probe does not see).
    "mma_cycles_per_sm_w4", "mma_cycles_per_sm_w8", "mma_cycles_per_sm_w16",
    # The period of a chain of mma.sync through the accumulators (on sm_70: of a group of four
    # STEPs, each chaining to the same STEP of the next mma.sync, not one HMMA's latency).
    "mma_dependent_latency",
    # FP32: the span holds the loop's own instructions (counter, compare, branch: 3 to 5 to 256
    # FFMAs from probe 17), which on sm_86 and later share a datapath with half the FP32 lanes:
    # the figure is the floor (1 clk from Ampere on, 2 before) plus that loop overhead, not taken
    # out.
    "fp32_cycles_per_warp_instruction",
    "int32_cycles_per_warp_instruction", "l1_latency_w1", "l1_latency_w2", "l1_latency_w4",
    "l1_latency_w8", "l2_latency_w1", "l2_latency_w2", "l2_latency_w4", "l2_latency_w8",
    "dram_latency_w1", "dram_latency_w2", "dram_latency_w4", "dram_latency_w8",
    # Beyond the SM the latency is wall time: the L2 and DRAM figures in ns too, and the SM clock
    # the chases ran at (so a clock that moved shows as such, not as a latency change).
    "l2_latency_ns_w1", "l2_latency_ns_w2", "l2_latency_ns_w4", "l2_latency_ns_w8",
    "dram_latency_ns_w1", "dram_latency_ns_w2", "dram_latency_ns_w4", "dram_latency_ns_w8",
    # DRAM with every warp in a 512 KB range of its own (the lines untouched, the pages walked by
    # the warm-up links), against the `dram` figures whose links fall anywhere in 2 GiB.
    "dram_tlbwarm_latency_w1", "dram_tlbwarm_latency_w2", "dram_tlbwarm_latency_w4",
    "dram_tlbwarm_latency_w8", "dram_tlbwarm_latency_ns_w1", "dram_tlbwarm_latency_ns_w2",
    "dram_tlbwarm_latency_ns_w4", "dram_tlbwarm_latency_ns_w8",
    # shared_latency: a link is a bare LDS R, [R] on every architecture from probe 17 (up to 16 a
    # node held an index, which sm_70 scaled with a shift before every LDS).
    "sm_clock_mhz_chase", "shared_latency",
    # Dependent (issue-to-use) latencies of the decode chains' ops, one warp a chain, one op a
    # link. lat_add_chain is the add ptxas makes of a dependent doubling: IADD3 and IMAD.IADD
    # alternating (the mean of the two) up to sm_89, IADD on sm_120a. The four conversions each
    # alone (probe 17): int -> float, float -> int, f32 -> f16, f16 -> f32.
    "lat_imad", "lat_add_chain", "lat_lop3", "lat_prmt", "lat_shf", "lat_hfma2", "lat_ffma",
    "lat_shfl", "lat_cvt_i2f", "lat_cvt_f2i", "lat_cvt_f32_to_f16", "lat_cvt_f16_to_f32",
    # The SM clock under the tensor loop, once it has settled (clock64 over %globaltimer).
    "sm_clock_mhz_mma",
)
_WARPS = (1, 2, 4, 8)
# The DRAM chase with all 32 lanes of every warp on chains of their own: 32 x W lines in flight an
# SM, the warp's link as long as its slowest line.
FACTS += tuple(f"dram_lanes_latency{u}_w{w}" for u in ("", "_ns") for w in _WARPS)
# The same chase on half the SMs: as many lines in flight on each of those SMs, half as many on
# the chip, so the pair says whether a link waits on its SM's lines or on the chip's.
FACTS += tuple(f"dram_lanes_half_latency{u}_w{w}" for u in ("", "_ns") for w in _WARPS)
# Every level again beside one warp of FFMAs on each SM (`_held`): the card may clock an SM whose
# only warp waits on memory differently from a working one, and these are the chases at the
# working clock (sm_clock_mhz_chase_held against sm_clock_mhz_chase).
_BEYOND_L1 = ("l2", "dram_tlbwarm", "dram", "dram_lanes", "dram_lanes_half")
FACTS += tuple(f"{lv}_held_latency_w{w}" for lv in ("l1",) + _BEYOND_L1 for w in _WARPS)
FACTS += tuple(f"{lv}_held_latency_ns_w{w}" for lv in _BEYOND_L1 for w in _WARPS)
FACTS += ("sm_clock_mhz_chase_held",)


# The facts probe 17 split or renamed: the four conversions alone (until then two pairs, each the
# mean of two ops on different pipes) and the add chain under a name that is not IADD's.
PROBE_17_FACTS = frozenset({"lat_add_chain", "lat_cvt_i2f", "lat_cvt_f2i", "lat_cvt_f32_to_f16",
                            "lat_cvt_f16_to_f32"})


def first_probe(name):
    """The probe version that first printed a fact (0: every version a row may come from). A run
    of probe N carries every fact of the versions up to N and none of a later one, so a row of an
    earlier probe still parses and a run that lost a fact of its own version does not. An earlier
    row is stale all the same: test_isa_facts fails while a row's probe is not the checked-in
    one."""
    if name in PROBE_17_FACTS:
        return 17
    if name.startswith("dram_lanes_half"):
        return 16
    if "_held" in name or name.startswith("dram_lanes") or name == "sm_clock_mhz_mma":
        return 15
    if name.startswith("mma_cycles_per_sm_w"):
        return 14
    if name.startswith("lat_"):
        return 11
    return 0


# Facts earlier probes printed that are not facts of the card under their name, with the last
# probe that printed each: read past in a run of that probe or an earlier one, never in a row,
# refused in a later run. resident_ctas_per_sm (up to 13) was the launch's own CTA count per SM;
# lat_iadd (up to 16) is the add chain, not IADD alone (lat_add_chain); lat_cvt_i2f_f2i and
# lat_cvt_f32_f16 (up to 16) were each the mean of two conversions on different pipes.
LEGACY_FACTS = {"resident_ctas_per_sm": 13, "lat_iadd": 16, "lat_cvt_i2f_f2i": 16,
                "lat_cvt_f32_f16": 16}
UNITS = {"tensor_flops_per_sm_clock": "flop"}


def unit_of(name):
    """The unit a fact is reported in: SM clocks unless listed, wall ns for the `_ns_` facts, MHz
    for the SM clocks."""
    if name.startswith("sm_clock_mhz"):
        return "mhz"
    return UNITS.get(name, "ns" if "_ns_" in name else "clk")


HEADER = ("probe", "card", "cc", "sms", "l2_bytes", "driver", "runtime", "nvcc", "shape")
# Header fields from a probe version on, with that version: `mem_clock_khz` the probe prints (the
# memory clock the card is rated at); `source_sha256` (of the isa_probe.cu the run built), `arch`
# (the -arch it was built for) and `sass_per_mma` (SASS MMA instructions of one mma.sync in that
# build) are written by `run` before the probe's output.
HEADER_SINCE = {"mem_clock_khz": 17, "source_sha256": 17, "arch": 17, "sass_per_mma": 17}
# Header fields that are positive integers. mem_clock_khz is a count too, but 0 is the device's
# way of not giving it (some virtualised setups): the row then says so in a note.
HEADER_COUNTS = ("cc", "sms", "l2_bytes", "driver", "runtime", "sass_per_mma")
MEM_CLOCK_NOT_GIVEN = "the rated memory clock is not given (the device attribute reads 0)"
# The notes `run` writes of the compute processes on the probe's GPU before, during and after the
# run (from probe 17 a row owes all three, each once): a count, or "not read" with the reason.
PROCESS_NOTE = re.compile(r"compute processes on (the probe's GPU|the GPU|any GPU \(the probe's "
                          r"not known\)) (before|during|after) the run(?: besides the probe "
                          r"\(nvidia-smi every 200 ms\))?: (.+)")
PROCESS_COUNT = re.compile(r"(\d+)(?: \(pid [^)]*\))?")
WAIVED = "the idle check was waived (--idle-unchecked)"
PROBE_VALUE = re.compile(r"isa_probe (\d+)")
PROBE_LINE = re.compile(r'printf\("probe (isa_probe \d+)\\n"\)')
SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE = os.path.join(HERE, "isa_probe.cu")


def checked_in_probe():
    """The `probe` header value the checked-in isa_probe.cu prints (`isa_probe <version>`)."""
    with open(SOURCE) as f:
        m = PROBE_LINE.search(f.read())
    if not m:
        raise ValueError("isa_probe.cu prints no probe version")
    return m.group(1)


def source_sha256(path=SOURCE):
    """The sha256 of a probe source as `run` records it: of its bytes with CRLF read as LF, so a
    checkout's line endings do not change it."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def parse_run(text):
    """The header fields and the facts of one isa_probe output; ValueError when a line is odd, a
    fact of the run's probe version is missing, a fact is unknown, of a later version, given
    twice, not a positive finite number or in another unit than its own, a fact the version no
    longer prints is there, a header field the version owes is missing or no count, the source
    hash is no sha256, the build's -arch is not the card's architecture, or a fact is outside
    what the architecture and the probe's own arithmetic allow (fact_bound_refusal)."""
    head, facts, ended, failures, checked = {}, {}, False, [], False
    legacy = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "fact":
            if len(parts) != 4:
                raise ValueError(f"odd fact line: {line!r}")
            name = parts[1]
            if name in LEGACY_FACTS:
                legacy.append(name)
                continue
            if name not in FACTS:
                raise ValueError(f"unknown fact: {line!r}")
            if name in facts:
                raise ValueError(f"fact given twice: {name}")
            try:
                value = float(parts[2])
            except ValueError:
                raise ValueError(f"odd fact line: {line!r}") from None
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} is not a positive finite number: {line!r}")
            if parts[3] != unit_of(name):
                raise ValueError(f"{name} is in {unit_of(name)}, not {parts[3]}: {line!r}")
            facts[name] = {"value": value, "unit": parts[3]}
        elif parts[0] == "end":
            ended = True
        elif parts[0] == "selfcheck":
            checked = True
            if parts[1:2] != ["ok"]:
                failures.append(" ".join(parts[2:]))
        elif parts[0] == "note":
            # Information the probe adds (the SM clock's range over the chases): kept, not a fact.
            head.setdefault("notes", []).append(" ".join(parts[1:]))
        elif parts[0] == REFUSED_LINE:
            # `run` refused this run and said why in the file itself: no row, whatever its name.
            raise ValueError(f"`run` refused this run: {' '.join(parts[1:])}")
        elif parts[0] in HEADER or parts[0] in HEADER_SINCE:
            if parts[0] in head:
                raise ValueError(f"{parts[0]} line given twice")
            head[parts[0]] = " ".join(parts[1:])
        else:
            raise ValueError(f"unknown line: {line!r}")
    # The version decides which facts the run owes; a run without its probe line owes them all.
    version = PROBE_VALUE.fullmatch(head.get("probe", ""))
    owed = int(version.group(1)) if version else max(first_probe(f) for f in FACTS)
    missing = [f for f in FACTS if f not in facts and first_probe(f) <= owed]
    if missing or not ended or not checked:
        raise ValueError(f"incomplete run: missing {missing}, ended={ended}, selfcheck={checked}")
    if failures:
        raise ValueError("the probe's self-check failed: " + "; ".join(failures))
    for k in HEADER + tuple(k for k, since in HEADER_SINCE.items() if since <= owed):
        if not head.get(k):
            raise ValueError(f"no {k} line")
    if not version:
        raise ValueError(f"odd probe line: {head['probe']!r}")
    # A version past the checked-in probe's owes facts and header lines this parser cannot know.
    if owed > int(checked_in_probe().split()[1]):
        raise ValueError(f"{head['probe']} is newer than the checked-in probe "
                         f"({checked_in_probe()})")
    early = [f for f in facts if first_probe(f) > owed]
    if early:
        raise ValueError(f"facts of a later probe than {head['probe']}: {early}")
    late = [f for f in legacy if LEGACY_FACTS[f] < owed]
    if late:
        raise ValueError(f"facts {head['probe']} no longer prints: {late}")
    for k in HEADER_COUNTS:
        if k in head and not (head[k].isdigit() and int(head[k]) > 0):
            raise ValueError(f"{k} is no positive count: {head[k]!r}")
    if "mem_clock_khz" in head and not head["mem_clock_khz"].isdigit():
        raise ValueError(f"mem_clock_khz is no count (kHz, 0 for not given): "
                         f"{head['mem_clock_khz']!r}")
    # From probe 17 a row is a run shown idle: the three process notes are there and none saw
    # another process, so a refused run's file merged by hand is still no row.
    if owed >= 17:
        refusal = idle_refusal(head.get("notes", []))
        if refusal:
            raise ValueError(refusal)
    if "source_sha256" in head and not SHA256.fullmatch(head["source_sha256"]):
        raise ValueError(f"odd source_sha256 line: {head['source_sha256']!r}")
    shape = head["shape"].split()
    if len(shape) != 2 or not shape[1].isdigit():
        raise ValueError(f"odd shape line: {head['shape']!r}")
    if shape[0] not in SHAPE_FLOPS:
        raise ValueError(f"unknown shape: {head['shape']!r}")
    if shape[0] != shape_of(int(head["cc"])):
        raise ValueError(f"shape {shape[0]} is not the one of cc {head['cc']} "
                         f"({shape_of(int(head['cc']))})")
    # The build's architecture is the card's own: a cubin of an earlier minor (sm_80 on a cc 8.6
    # card) or Blackwell without its accelerated set (sm_120 for sm_120a) runs, but is not the
    # code the row's facts claim to be of.
    if "arch" in head:
        own = "sm_" + arch_key(int(head["cc"]), head["shape"])
        if head["arch"] != own:
            raise ValueError(f"built for {head['arch']}, not {own} (cc {head['cc']})")
    # The SASS MMAs of one mma.sync, each of the shape's own m x n x k (sass_counts' table), are
    # the shape's flops: what `run` checked on the opcodes, held here on the count it wrote.
    if "sass_per_mma" in head:
        mnk = _sass_counts().MMA_SHAPES.get(shape[0].split("_")[1])
        if mnk is None or int(head["sass_per_mma"]) * 2 * mnk[0] * mnk[1] * mnk[2] != \
                SHAPE_FLOPS[shape[0]]:
            raise ValueError(f"sass_per_mma {head['sass_per_mma']} SASS MMAs are not the "
                             f"{SHAPE_FLOPS[shape[0]]} flops of one {shape[0]}")
    refusal = fact_bound_refusal(shape[0], int(shape[1]), facts)
    if refusal:
        raise ValueError(refusal)
    return head, facts


# isa_probe.cu prints every fact with three decimals (fact(): "%.3f"), so a printed value lies
# within half its last place of the value the probe computed.
PRINTED_HALF_PLACE = 0.0005
# The probe's own self-check of the saturated tensor rate (isa_probe.cu, after the
# tensor_flops_per_sm_clock fact): no SM it knows issues a dense MMA above 2048 flop a clock (the
# GA100 / GH100 class) or below one m8n8k4 (512 flop) in 32 clocks. A run outside it fails that
# check, so a file outside it is no output of a probe that passed.
TENSOR_RATE_RANGE = (16.0, 2048.0)
MMA_CYCLE_FACTS = ("mma_cycles_per_sm", "mma_cycles_per_sm_w4", "mma_cycles_per_sm_w8",
                   "mma_cycles_per_sm_w16")


def fact_bound_refusal(shape, printed_flops, facts):
    """Why a run's facts are outside what the architecture and the probe's own arithmetic allow,
    or None. `shape` is the run's MMA shape, `printed_flops` the flops its shape line printed.

    - The tensor rate (the shape's flops over mma_cycles_per_sm, as row_of derives it) is inside
      the probe's self-check range; the 4, 8 and 16 warp loops, the same MMAs with fewer warps,
      do not beat the pipe's 2048 flop a clock either. This keeps the derived row finite and
      positive.
    - What the probe computes from mma_cycles_per_sm agrees with it to the printed places:
      mma_cycles_per_partition is 4 x it, tensor_flops_per_sm_clock is the printed flops over it,
      and mma_dependent_latency is at least half the issue interval (2 x it), the probe's check.
    - A dependent latency in clocks is at least one clock: the link that uses a result issues
      after the one that made it.
    """
    h = PRINTED_HALF_PLACE
    low, high = TENSOR_RATE_RANGE
    flops = SHAPE_FLOPS[shape]
    sm = facts["mma_cycles_per_sm"]["value"]
    if not low <= flops / sm <= high:
        return (f"mma_cycles_per_sm {sm} gives {flops / sm:.6g} flop a clock for {shape}, outside "
                f"the probe's {low:g}..{high:g}")
    for name in MMA_CYCLE_FACTS[1:]:
        if name in facts and flops / facts[name]["value"] > high:
            return (f"{name} {facts[name]['value']} gives {flops / facts[name]['value']:.6g} flop "
                    f"a clock for {shape}, above the pipe's {high:g}")
    partition = facts["mma_cycles_per_partition"]["value"]
    if abs(partition - 4.0 * sm) > 5 * h:
        return f"mma_cycles_per_partition {partition} is not 4 x mma_cycles_per_sm {sm}"
    rate = facts["tensor_flops_per_sm_clock"]["value"]
    if not printed_flops / (sm + h) - h <= rate <= printed_flops / (sm - h) + h:
        return (f"tensor_flops_per_sm_clock {rate} is not the shape line's {printed_flops} flops "
                f"over mma_cycles_per_sm {sm}")
    latency = facts["mma_dependent_latency"]["value"]
    if latency < 2.0 * sm - 3 * h:
        return (f"mma_dependent_latency {latency} is under half the issue interval "
                f"(2 x mma_cycles_per_sm {sm})")
    for name, fact in facts.items():
        if fact["unit"] == "clk" and ("latency" in name or name.startswith("lat_")) and \
                fact["value"] < 1.0:
            return f"{name} {fact['value']} is under one clock"
    return None


def idle_refusal(notes):
    """Why the process notes of a run (its `note` lines without the word) do not show its GPU
    idle, or None: each of before, during and after is noted once; a count other than 0 is
    another process there (never waived); a list not read needs the --idle-unchecked note, and
    that note needs a list not read (`run` writes it only then)."""
    seen, unread = {}, []
    for n in notes:
        m = PROCESS_NOTE.fullmatch(n)
        if not m:
            continue
        where, when, value = m.groups()
        if when in seen:
            return f"the compute processes {when} the run are noted twice"
        seen[when] = value
        if value.startswith("not read"):
            unread.append(f"{when}: {value}")
            continue
        count = PROCESS_COUNT.fullmatch(value)
        if not count:
            return f"odd compute-process note: {n!r}"
        # With the probe's GPU not known, a process on any GPU may be on the probe's: refused
        # like one seen there; none on any GPU is a list that cannot say which GPU, not read.
        if int(count.group(1)) != 0:
            return (f"another compute process on {where} {when} the run: {value}")
        if where.startswith("any GPU"):
            unread.append(f"{when}: 0 on {where}")
    missing = [w for w in ("before", "during", "after") if w not in seen]
    if missing:
        return f"no note of the compute processes {' / '.join(missing)} the run"
    waived = any(n.startswith(WAIVED) for n in notes)
    if unread and not waived:
        return (f"the compute processes were not read ({'; '.join(unread)}); `run "
                "--idle-unchecked` accepts an unread list and says so in the row")
    if waived and not unread:
        return "the idle check is noted as waived, yet every process list was read"
    return None


# Flops per warp instruction of each MMA shape the probe issues: 2 x m x n x k, and on Volta four
# of them (one mma.sync.m8n8k4 performs four independent 8x8x4 products, one per quad pair: PTX
# ISA). The row reads this table, so a probe output that printed another figure (probes before 10
# printed 512 for the Volta shape) still gives the right rate; the measured cycles per MMA stay.
SHAPE_FLOPS = {"hmma_884_f32_f16": 2048, "hmma_1688_f32_f16": 2048, "hmma_16816_f32_bf16": 4096}


def shape_of(cc):
    """The MMA shape the probe issues on compute capability `cc` (isa_probe.cu, where the device
    code picks it and the host checks it against the card): Volta's m8n8k4, Turing's m16n8k8 of
    f16, m16n8k16 of bf16 from Ampere on."""
    return ("hmma_16816_f32_bf16" if cc >= 80 else
            "hmma_1688_f32_f16" if cc >= 75 else "hmma_884_f32_f16")


def row_of(head, facts, source):
    """The table's row of one run (parse_run's head and facts); `source` names the committed
    probe output it came from. The tensor rate is derived here: the shape's flops (the table's,
    never the figure the run printed) over the measured cycles per MMA per SM."""
    shape = head["shape"].split()[0]
    flops = SHAPE_FLOPS[shape]
    facts = dict(facts)
    facts["tensor_flops_per_sm_clock"] = {
        "value": round(flops / facts["mma_cycles_per_sm"]["value"], 3), "unit": "flop"}
    row = {
        "provenance": "Measured",
        "source": source,
        "card": head["card"],
        "cc": int(head["cc"]),
        "sms": int(head["sms"]),
        "l2_bytes": int(head["l2_bytes"]),
        "mma_shape": shape,
        "mma_flops": flops,
    }
    # From probe 17: SASS MMA instructions per mma.sync of mma_shape in the run's build (the unit
    # of the MMA facts is the mma.sync), and the memory clock the card is rated at (a note where
    # the device does not give it).
    notes = list(head.get("notes", []))
    if "sass_per_mma" in head:
        row["sass_per_mma"] = int(head["sass_per_mma"])
    if "mem_clock_khz" in head:
        if int(head["mem_clock_khz"]) > 0:
            row["mem_clock_khz"] = int(head["mem_clock_khz"])
        else:
            notes.append(MEM_CLOCK_NOT_GIVEN)
    row["tool"] = {"probe": head["probe"], "nvcc": head["nvcc"], "driver": head["driver"],
                   "runtime": head["runtime"]}
    for k in ("arch", "source_sha256"):
        if k in head:
            row["tool"][k] = head[k]
    row["notes"] = notes
    row["facts"] = {name: facts[name] for name in FACTS if name in facts}
    return row


NOTE = ("ISA facts per architecture, regenerated by `tools/bench/isa_probe/isa_probe.py "
        "regenerate` from the committed probe outputs in src/core/data/isa_probe/ (one "
        "isa_probe_<arch>.txt per architecture, written by `isa_probe.py run` on one card of it; "
        "never edited by hand). In SM clocks unless the unit says otherwise: tensor pipe cycles "
        "per MMA at the SM and per partition (and with one CTA of 4, 8, 16 warps on the SM) and "
        "the dependent MMA latency, all per PTX mma.sync of the row's mma_shape (on sm_70 one "
        "mma.sync.m8n8k4 is four SASS HMMA.884 STEPs, elsewhere one HMMA: the row's sass_per_mma, "
        "read from the build's SASS, from probe 17), INT32 / FP32 cycles per warp instruction "
        "(the FP32 figure with the loop's counter, compare and branch in it), the dependent "
        "latencies of the decode chains' ALU ops (lat_add_chain the add ptxas makes of a doubling, "
        "IADD3 and IMAD.IADD alternating up to sm_89; each conversion alone from probe 17), "
        "dependent L1 / L2 / DRAM load latencies alone and under 2, "
        "4, 8 chasing warps per SM (the L2 and DRAM ones in wall ns too, with the SM clock of the "
        "chase; DRAM in a small range per warp, over the whole buffer, and with all 32 lanes of a "
        "warp chasing, on every SM and on half of them), shared-memory latency. The load "
        "latencies are given twice: on a chip that "
        "runs nothing else, and beside one warp of FFMAs on every SM (`_held`), where the SM "
        "clock is the working one (sm_clock_mhz_chase_held; sm_clock_mhz_mma under the tensor "
        "loop). "
        "The DRAM ns figures step with the ~1 us %globaltimer over 512 (one lane) or 256 (32 "
        "lanes) links: about 2 and 4 ns. "
        "Every row names its source file, the card, the probe version, nvcc, driver and runtime, "
        "and from probe 17 the sha256 of the isa_probe.cu it was built from, the build's -arch, "
        "the memory clock the card is rated at (a note where the device does not give it) and, "
        "in its notes, the compute processes on its GPU before, during and after the run (none "
        "besides the probe, or not read and the check waived) and the clocks nvidia-smi saw "
        "during it; "
        "provenance is Measured for all. A row is what its own probe version measured: up to probe 13 a link of "
        "the L1 / L2 / DRAM chains carried the chain's index-to-address ops beside the load, up "
        "to probe 16 the sm_70 shared link a shift beside the LDS. tests/test_isa_facts.py pins "
        "the schema, that the table equals its regeneration, and that every row comes from the "
        "checked-in probe source.")


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


def _sass_counts():
    """tools/kernel_facts/sass_counts.py, the SASS reader the kernel facts count MMAs with."""
    path = os.path.join(ROOT, "tools", "kernel_facts", "sass_counts.py")
    spec = importlib.util.spec_from_file_location("sass_counts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MMA_UNIT = "isa_probe_mma_unit"


def sass_mma_unit(sass):
    """[(opcode, (m, n, k) or None)] of the MMA instructions of isa_probe_mma_unit (one mma.sync of
    the build's shape and nothing else of the tensor pipe) in `cuobjdump -sass` text, read with
    sass_counts' own parser and shape table; ValueError without that kernel or an MMA in it."""
    sc = _sass_counts()
    bodies = [body for name, body in sc.iter_functions(sass.splitlines(), lambda n: MMA_UNIT in n)]
    if len(bodies) != 1:
        raise ValueError(f"{len(bodies)} {MMA_UNIT} kernels in the build's SASS")
    mmas = [(op, sc.mma_shape(op)) for _, _, op, _ in bodies[0]
            if sc.base_opcode(op) in sc.MMA_OPCODES]
    if not mmas:
        raise ValueError(f"no MMA in {MMA_UNIT}")
    return mmas


def _bus(text):
    """A PCI bus id as CUDA (0000:01:00.0) and nvidia-smi (00000000:01:00.0) spell it, alike."""
    parts = text.strip().lower().split(":")
    if len(parts) != 3:
        return text.strip().lower()
    return f"{int(parts[0], 16):x}:{parts[1]}:{parts[2]}"


def _smi(args):
    """nvidia-smi's CSV lines (no header, no units), or the reason there are none."""
    try:
        r = subprocess.run(["nvidia-smi"] + args + ["--format=csv,noheader,nounits"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"nvidia-smi failed: {type(e).__name__}"
    if r.returncode != 0:
        return None, f"nvidia-smi exited {r.returncode}"
    return [[f.strip() for f in line.split(",")] for line in r.stdout.splitlines()
            if line.strip()], None


def _compute_apps():
    return _smi(["--query-compute-apps=gpu_bus_id,pid"])


def _apps_note(when, apps, bus):
    """The note line of the compute processes on the probe's GPU (pids only: a process's name may
    be a path of the machine it ran on)."""
    rows, why = apps
    if rows is None:
        return f"note compute processes on the GPU {when} the run: not read ({why})"
    mine = [r for r in rows if bus is None or _bus(r[0]) == bus]
    where = "the probe's GPU" if bus is not None else "any GPU (the probe's not known)"
    pids = ", ".join(r[1] for r in mine if len(r) > 1)
    return f"note compute processes on {where} {when} the run: {len(mine)}" + \
        (f" (pid {pids})" if mine else "")


# How nvidia-smi prints a pid it cannot give.
UNNAMED = "[N/A]"


def _during_note(during, bus, own):
    """The note line of the compute processes nvidia-smi listed on the probe's GPU while the probe
    (pid `own`) ran; `during` is rows of (timestamp, bus id, pid), one timestamp a sample.

    The probe is one of the listed processes itself. Where its pid is listed, every other entry
    (another pid, or one printed as `[N/A]`) is another process. Where it is not (the sampler gave
    nothing in time, or this nvidia-smi prints pids of another namespace or none), the probe may
    be any one entry, never two: a list with no entry, or with the same one entry throughout, was
    not read (the waiver may cover it); a list with two entries in one sample or two distinct
    entries over the run had at least one process besides the probe, which is never waived."""
    head = "note compute processes on the probe's GPU during the run"
    if during is None:
        return f"{head}: not read (nvidia-smi not run)"
    if bus is None:
        return f"{head}: not read (the probe's GPU not known)"
    # A sample's entries: its distinct pids (a pid listed twice in one sample, or in two samples
    # printed with one timestamp, is one process, so the probe never counts against itself), and
    # each `[N/A]` row on its own (unnamed rows cannot be told apart: two may be two processes).
    samples = {}
    for r in during:
        if _bus(r[1]) == bus:
            pids, unnamed = samples.setdefault(r[0], (set(), []))
            if r[2] == UNNAMED:
                unnamed.append(r[2])
            else:
                pids.add(r[2])
    entries = sorted({p for pids, unnamed in samples.values() for p in pids | set(unnamed)})
    busy = f"{head} besides the probe (nvidia-smi every 200 ms)"
    if own in entries:
        others = [p for p in entries if p != own]
        beside = max(len(pids - {own}) + len(unnamed) for pids, unnamed in samples.values())
        n = max(len(others), beside)
        return f"{busy}: {n}" + (f" (pid {', '.join(others)})" if n else "")
    if not entries:
        return f"{head}: not read (no process listed, the probe's own pid {own} neither)"
    most = max(len(pids) + len(unnamed) for pids, unnamed in samples.values())
    if most == 1 and len(entries) == 1:
        return (f"{head}: not read (one process listed throughout, pid {entries[0]}, not the "
                f"probe's own pid {own}: it may be the probe under another pid namespace)")
    # The probe is at most one of them.
    n = max(most, len(entries)) - 1
    return (f"{busy}: {n} (pid {', '.join(entries)}; the probe's own pid {own} not among them, "
            f"up to {most} a sample)")


# The suffix of a refused run's file (never an isa_probe_<arch>.txt, so never merged as a row),
# and the line `run` ends it with, which parse_run refuses under any name.
REFUSED = ".refused"
REFUSED_LINE = "refused"


def cmd_run(a):
    arch = a.arch
    src = SOURCE
    # Lines `run` adds before the probe's output: which source the build was of, its -arch, and
    # how many SASS MMA instructions one mma.sync is in it (the unit of the MMA facts).
    head = [f"source_sha256 {source_sha256(src)}", f"arch sm_{arch}"]
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "isa_probe.exe" if os.name == "nt" else "isa_probe")
        cmd = [a.nvcc, "-O3", "-std=c++17", f"-arch=sm_{arch}", "-o", exe, src]
        print("+", " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
        cuobjdump = a.cuobjdump
        if cuobjdump is None:
            # Beside the nvcc of the build where the toolkit has one, else from PATH.
            nvcc = shutil.which(a.nvcc) or a.nvcc
            near = os.path.join(os.path.dirname(nvcc), "cuobjdump" + (".exe" if os.name == "nt"
                                                                     else ""))
            cuobjdump = near if os.path.exists(near) else "cuobjdump"
        sass = subprocess.run([cuobjdump, "-sass", exe], stdout=subprocess.PIPE, text=True,
                              check=True).stdout
        try:
            mmas = sass_mma_unit(sass)
        except ValueError as e:
            print(f"isa_probe: {e}; no run", file=sys.stderr)
            return 3
        head.append(f"sass_per_mma {len(mmas)}")
        before = _compute_apps()

        def sampler(query):
            # nvidia-smi every 200 ms while the probe runs, or None without it.
            try:
                return subprocess.Popen(
                    ["nvidia-smi", query, "--format=csv,noheader,nounits", "-lms", "200"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except OSError:
                return None

        def rows_of(proc, fields):
            if proc is None:
                return None
            proc.terminate()
            text, _ = proc.communicate()
            return [[f.strip() for f in line.split(",")] for line in text.splitlines()
                    if line.count(",") == fields - 1]

        # The clocks (the memory clock is the P-state's, which the device API does not give) and
        # the compute processes, sampled through the run.
        clocks = sampler("--query-gpu=pci.bus_id,pstate,clocks.sm,clocks.mem")
        apps = sampler("--query-compute-apps=timestamp,gpu_bus_id,pid")
        # The probe's stderr (CUDA errors) passes through; its stdout is kept whole, so a failed
        # or refused run still shows what it measured before stopping.
        probe = subprocess.Popen([exe], stdout=subprocess.PIPE, text=True)
        stdout, _ = probe.communicate()
        run = subprocess.CompletedProcess([exe], probe.returncode, stdout, None)
        samples = rows_of(clocks, 4) or []
        during = rows_of(apps, 3)
        after = _compute_apps()
    m = re.search(r"^note device 0 is the GPU at PCI (\S+)$", run.stdout, re.M)
    bus = _bus(m.group(1)) if m else None
    tail = [_apps_note("before", before, bus), _apps_note("after", after, bus),
            _during_note(during, bus, str(probe.pid))]
    mine = [s for s in samples if bus is not None and _bus(s[0]) == bus]
    if mine:
        def span(col):
            vals = sorted(int(s[col]) for s in mine if s[col].isdigit())
            return f"{vals[0]}..{vals[-1]}" if vals else "not read"
        tail.append(f"note nvidia-smi over the run ({len(mine)} samples): memory clock "
                    f"{span(3)} MHz, SM clock {span(2)} MHz, P-state "
                    f"{'/'.join(sorted({s[1] for s in mine}))}")
    else:
        tail.append("note nvidia-smi over the run: no clock sample of the probe's GPU")
    # Another compute process on the probe's GPU before, during or after the run (a time-sliced
    # GPU stretches every clock64 span while the clock reads right), or a run whose GPU could not
    # be shown idle at all three: no row (parse_run's idle_refusal). --idle-unchecked accepts a
    # run whose process list could not be read (and the row says so), never one where another
    # process was seen; the note is written only when it is used.
    if a.idle_unchecked and any(
            m and (m.group(3).startswith("not read") or
                   (m.group(1).startswith("any GPU") and m.group(3).startswith("0")))
            for m in (PROCESS_NOTE.fullmatch(n[5:]) for n in tail)):
        tail.append("note " + WAIVED)
    body = run.stdout if run.stdout.endswith("\n") or not run.stdout else run.stdout + "\n"
    out = "\n".join(head) + "\n" + body + "\n".join(tail) + "\n"
    refusal = None
    if run.returncode != 0:
        refusal = f"the probe exited {run.returncode}"
    else:
        try:
            parsed, _ = parse_run(out)
        except ValueError as e:
            refusal = str(e)
        else:
            # The SASS MMAs of one mma.sync are its flops, by sass_counts' shape table: so a
            # count of SASS MMAs priced at these facts over sass_per_mma is the same work.
            shape = parsed["shape"].split()[0]
            if any(s is None for _, s in mmas) or \
                    sum(2 * s[0] * s[1] * s[2] for _, s in mmas) != SHAPE_FLOPS[shape]:
                refusal = (f"the SASS of one {shape} ({', '.join(op for op, _ in mmas)}) is not "
                           f"its {SHAPE_FLOPS[shape]} flops by sass_counts' shapes")
    # A refused run is kept for its numbers, under a name that is no row's: `--out` is written
    # only by a run that is one, so "merge the first run" can never pick up a refused one.
    path = a.out or f"isa_probe_{arch}.txt"
    if refusal:
        # The reason goes into the file too, so a copy under a row's name is refused by merge.
        path += REFUSED
        out += f"{REFUSED_LINE} {' '.join(refusal.split())}\n"
    with open(path, "w", newline="\n") as f:
        f.write(out)
    sys.stdout.write(out)
    if refusal:
        print(f"isa_probe: {refusal}; wrote {path}, which is no row to merge", file=sys.stderr)
        return run.returncode or 3
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
    regenerated from all of them. A merge the table cannot be built from (another input is
    malformed, or gives the same architecture under another name) changes nothing: the
    destination is put back as it was."""
    if a.input.endswith(REFUSED):
        print(f"isa_probe: {a.input} is a refused run's output; nothing merged", file=sys.stderr)
        return 1
    # Read whole before the destination opens: the input may be the destination itself.
    with open(a.input) as f:
        text = f.read()
    try:
        head, _ = parse_run(text)
    except ValueError as e:
        print(f"isa_probe: {a.input}: {e}; nothing merged", file=sys.stderr)
        return 1
    key = arch_key(int(head["cc"]), head["shape"])
    dest = os.path.join(a.inputs, f"isa_probe_{key}.txt")
    before = None
    if os.path.exists(dest):
        with open(dest, "rb") as f:
            before = f.read()
    with open(dest, "w", newline="\n") as out:
        out.write(text)
    try:
        build_table(a.inputs)
    except ValueError as e:
        if before is None:
            os.remove(dest)
        else:
            with open(dest, "wb") as out:
                out.write(before)
        print(f"isa_probe: {e}; nothing merged ({dest} is as it was)", file=sys.stderr)
        return 1
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
    r.add_argument("--cuobjdump", help="default: the one beside --nvcc, else from PATH")
    r.add_argument("--out", help="the run's output when it is a row; a refused run's goes to "
                   "<out>" + REFUSED)
    r.add_argument("--idle-unchecked", action="store_true",
                   help="accept a run whose compute-process list nvidia-smi could not give")
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
