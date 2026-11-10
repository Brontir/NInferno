#!/usr/bin/env python3
"""The ISA facts table (src/core/data/isa_facts.json), its committed probe outputs
(src/core/data/isa_probe/) and the probe's parser (#613 PR-2): the schema is pinned (every row
Measured with its source file, every fact present with a positive value in its unit, the card and
tool named, the levels in order), the table equals its regeneration from the inputs, every row is
from the checked-in probe (its version and the sha256 of its source), a probe output round-trips
into a row, and odd or incomplete outputs are refused. No GPU, no CUDA toolkit."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "isa_probe", os.path.join(ROOT, "tools", "bench", "isa_probe", "isa_probe.py"))
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)
SC = ip._sass_counts()


def generation_bf16_rate(cc):
    """The bf16 tensor rate per SM and clock of the generation row for compute capability `cc`
    (src/core/generation.h: `{major, minor, <lent>, "name", bf16, ...}`), or None without one."""
    with open(os.path.join(ROOT, "src", "core", "generation.h")) as f:
        text = f.read()
    # A row covers its major from its minor up to the next row's (ga10x is {8, 1, ...}: 8.1-8.8).
    best = None
    for m in re.finditer(r'\{\s*(\d+),\s*(\d+),\s*\d+,\s*"[^"]+",\s*([\d.]+)', text):
        major, minor, rate = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if major == cc // 10 and minor <= cc % 10 and (best is None or minor >= best[0]):
            best = (minor, rate)
    return best[1] if best else None

# A run in the checked-in probe's format (its version and source hash read from isa_probe.cu), as
# `run` writes it: its own header lines, then the probe's output, then its notes.
SAMPLE = "source_sha256 " + ip.source_sha256() + """
arch sm_89
sass_per_mma 1
probe """ + ip.checked_in_probe() + """
card NVIDIA GeForce RTX 4090
cc 89
sms 128
l2_bytes 75497472
mem_clock_khz 10501000
driver 13040
runtime 13040
nvcc 13.4.92
shape hmma_16816_f32_bf16 4096
fact mma_cycles_per_sm 8.012 clk
fact mma_cycles_per_partition 32.048 clk
fact tensor_flops_per_sm_clock 511.2 flop
fact mma_cycles_per_sm_w4 8.3 clk
fact mma_cycles_per_sm_w8 8.1 clk
fact mma_cycles_per_sm_w16 8.0 clk
fact mma_dependent_latency 33.1 clk
fact fp32_cycles_per_warp_instruction 1.004 clk
fact int32_cycles_per_warp_instruction 2.01 clk
""" + "".join(
    # Every level on the chip alone (at 2520 MHz) and beside the FFMA kernel (`_held`, 1890 MHz).
    f"fact {lv}{tag}_latency_w{w} {v + 3 * w:.1f} clk\n" +
    (f"fact {lv}{tag}_latency_ns_w{w} {(v + 3 * w) * 1e3 / mhz:.1f} ns\n" if lv != "l1" else "")
    for tag, mhz in (("", 2520.0), ("_held", 1890.0))
    for lv, v in (("l1", 34.0), ("l2", 260.0), ("dram", 560.0), ("dram_tlbwarm", 520.0),
                  ("dram_lanes", 590.0), ("dram_lanes_half", 575.0))
    for w in (1, 2, 4, 8)) + \
    "fact sm_clock_mhz_chase 2520.0 mhz\nfact sm_clock_mhz_chase_held 1890.0 mhz\n" + \
    "fact sm_clock_mhz_mma 1900.0 mhz\nfact shared_latency 29.5 clk\n" + \
    "".join(f"fact lat_{op} {v:.1f} clk\n" for op, v in (
        ("imad", 4.5), ("add_chain", 4.0), ("lop3", 4.2), ("prmt", 4.4), ("shf", 4.5),
        ("hfma2", 5.0), ("ffma", 4.1), ("shfl", 24.0), ("cvt_i2f", 4.5), ("cvt_f2i", 12.0),
        ("cvt_f32_to_f16", 6.5), ("cvt_f16_to_f32", 4.5))) + \
    "note the SM clock ranged 2400..2535 MHz over the L2 / DRAM chases\nselfcheck ok\nend\n" + \
    "note compute processes on the probe's GPU before the run: 0\n" + \
    "note compute processes on the probe's GPU after the run: 0\n" + \
    "note compute processes on the probe's GPU during the run besides the probe (nvidia-smi " + \
    "every 200 ms): 0\n"
DURING_IDLE = ("note compute processes on the probe's GPU during the run besides the probe "
               "(nvidia-smi every 200 ms): 0\n")
# The lines `run` writes around the probe's own output (its header lines before, its process
# notes after): what a faked run's probe binary does not print.
RUN_LINES = ("source_sha256 ", "arch ", "sass_per_mma ", "note compute processes")


def during_of(samples):
    """nvidia-smi --query-compute-apps=timestamp,gpu_bus_id,pid -lms 200 output: one sample per
    list, each entry a pid on the GPU at 01 (or "<bus>:<pid>" on another one)."""
    lines = []
    for i, pids in enumerate(samples):
        for p in pids:
            bus, pid = p.split(":") if ":" in p else ("01", p)
            lines.append(f"2026/11/10 12:00:{i // 5:02d}.{i % 5 * 200:03d}, 00000000:{bus}:00.0, "
                         f"{pid}")
    return "".join(line + "\n" for line in lines)


# The SASS of isa_probe_mma_unit for sm_89: one HMMA.16816 is one m16n8k16.
SASS_89 = """
\t\tFunction : _Z18isa_probe_mma_unitPfjj
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0100*/                   HMMA.16816.F32.BF16 R4, R12, R14, R4 ;
        /*0200*/                   EXIT ;
"""


def earlier_run(version):
    """SAMPLE as a run of an earlier probe `version` would read: without the facts and the header
    lines later versions added."""
    keep = []
    for line in SAMPLE.splitlines():
        word = line.split()[0]
        if word == "fact" and ip.first_probe(line.split()[1]) > version:
            continue
        if ip.HEADER_SINCE.get(word, 0) > version:
            continue
        keep.append(line)
    return "\n".join(keep).replace("probe " + ip.checked_in_probe(),
                                   f"probe isa_probe {version}") + "\n"


# The SASS of isa_probe_mma_unit as `cuobjdump -sass` prints it for sm_70 (nvcc 12.8): one
# mma.sync.m8n8k4 is four HMMA.884 STEPs.
SASS_70 = """
\t\tFunction : _Z18isa_probe_mma_unitPfjj
\t.headerflags\t@"EF_CUDA_SM70 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM70)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ; /* 0x00000a00ff017624 */
        /*0010*/              @!PT SHFL.IDX PT, RZ, RZ, RZ, RZ ;     /* 0x000000fffffff389 */
        /*0100*/                   HMMA.884.F32.F32.STEP0 R4, R12.reuse.ROW, R14.reuse.COL, R4 ;
        /*0110*/                   HMMA.884.F32.F32.STEP1 R6, R12.reuse.ROW, R14.reuse.COL, R6 ;
        /*0140*/                   HMMA.884.F32.F32.STEP2 R8, R12.reuse.ROW, R14.reuse.COL, R8 ;
        /*0170*/                   HMMA.884.F32.F32.STEP3 R10, R12.ROW, R14.COL, R10 ;
        /*0200*/                   EXIT ;
\t\tFunction : _Z11mma_latencyv
        /*0000*/                   HMMA.884.F32.F32.STEP0 R4, R12.reuse.ROW, R14.reuse.COL, R4 ;
"""


class IsaFacts(unittest.TestCase):
    def test_parse_and_row(self):
        head, facts = ip.parse_run(SAMPLE)
        self.assertEqual(head["card"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(set(facts), set(ip.FACTS))
        row = ip.row_of(head, facts, "isa_probe_89.txt")
        self.assertEqual((row["provenance"], row["source"]), ("Measured", "isa_probe_89.txt"))
        self.assertEqual((row["cc"], row["sms"], row["mma_shape"], row["mma_flops"]),
                         (89, 128, "hmma_16816_f32_bf16", 4096))
        self.assertEqual(row["tool"]["nvcc"], "13.4.92")
        self.assertEqual((row["tool"]["arch"], row["tool"]["source_sha256"]),
                         ("sm_89", ip.source_sha256()))
        self.assertEqual((row["sass_per_mma"], row["mem_clock_khz"]), (1, 10501000))
        self.assertEqual(row["notes"],
                         ["the SM clock ranged 2400..2535 MHz over the L2 / DRAM chases",
                          "compute processes on the probe's GPU before the run: 0",
                          "compute processes on the probe's GPU after the run: 0",
                          DURING_IDLE[5:-1]])
        self.assertAlmostEqual(row["facts"]["dram_latency_w8"]["value"], 584.0)
        # The tensor rate is derived from the parser's flops table over the measured cycles, not
        # read from the probe: a Volta run that printed 512 for m8n8k4 (four 8x8x4 products a
        # warp instruction: 2048) still gives V100's ~1024 flop/clk/SM.
        self.assertAlmostEqual(row["facts"]["tensor_flops_per_sm_clock"]["value"],
                               round(4096 / 8.012, 3))
        volta = SAMPLE.replace("shape hmma_16816_f32_bf16 4096", "shape hmma_884_f32_f16 512") \
            .replace("cc 89", "cc 70").replace("fact mma_cycles_per_sm 8.012 clk",
                                               "fact mma_cycles_per_sm 2.014 clk") \
            .replace("arch sm_89", "arch sm_70").replace("sass_per_mma 1", "sass_per_mma 4")
        head, facts = ip.parse_run(volta)
        row = ip.row_of(head, facts, "isa_probe_70.txt")
        self.assertEqual(row["mma_flops"], 2048)
        self.assertAlmostEqual(row["facts"]["tensor_flops_per_sm_clock"]["value"],
                               round(2048 / 2.014, 3))
        self.assertEqual(ip.arch_key(120, "hmma_16816_f32_bf16 4096"), "120a")
        self.assertEqual(ip.arch_key(75, "hmma_1688_f32_f16 2048"), "75")
        self.assertEqual([ip.unit_of(n) for n in ("l1_latency_w1", "dram_latency_ns_w2",
                                                  "sm_clock_mhz_chase", "mma_cycles_per_sm_w16",
                                                  "tensor_flops_per_sm_clock",
                                                  "dram_lanes_held_latency_ns_w8",
                                                  "l1_held_latency_w4", "sm_clock_mhz_chase_held",
                                                  "sm_clock_mhz_mma")],
                         ["clk", "ns", "mhz", "clk", "flop", "ns", "clk", "mhz", "mhz"])
        # What probes up to 13 printed as resident_ctas_per_sm was the launch's own CTA count, no
        # fact of the card, and what probes up to 16 printed as lat_iadd, lat_cvt_i2f_f2i and
        # lat_cvt_f32_f16 were not the facts their names say: read past in a run of those
        # probes, never in a row, refused in a later probe's run.
        for legacy, last in ip.LEGACY_FACTS.items():
            line = f"fact {legacy} 4.000 {'count' if legacy.startswith('resident') else 'clk'}\n"
            run = earlier_run(last)
            head, facts = ip.parse_run(run + line)
            self.assertNotIn(legacy, ip.row_of(head, facts, "x")["facts"])
            with self.assertRaises(ValueError, msg=legacy) as e:
                ip.parse_run(SAMPLE + line)
            self.assertIn("no longer prints", str(e.exception))

    def test_refused_values_and_headers(self):
        # merge is where a saved run enters the table: a value that is no positive finite number,
        # another unit than the fact's own, an unknown or repeated fact and a missing or odd
        # header field are all refused, each named.
        shared = "fact shared_latency 29.5 clk\n"
        for bad, said in (
                ("fact shared_latency nan clk\n", "positive finite"),
                ("fact shared_latency inf clk\n", "positive finite"),
                ("fact shared_latency -29.5 clk\n", "positive finite"),
                ("fact shared_latency 0 clk\n", "positive finite"),
                ("fact shared_latency 29.5 ns\n", "is in clk"),
                ("fact shared_latency twelve clk\n", "odd fact line"),
                (shared + shared, "given twice"),
                (shared + "fact shared_latency_w9 29.5 clk\n", "unknown fact")):
            with self.assertRaises(ValueError, msg=bad) as e:
                ip.parse_run(SAMPLE.replace(shared, bad))
            self.assertIn(said, str(e.exception), bad)
        for field, value in (("sms", "128"), ("l2_bytes", "75497472"), ("driver", "13040"),
                             ("runtime", "13040"), ("nvcc", "13.4.92"),
                             ("card", "NVIDIA GeForce RTX 4090"), ("cc", "89")):
            line = f"{field} {value}\n"
            self.assertIn(line, SAMPLE)
            with self.assertRaises(ValueError, msg=field) as e:
                ip.parse_run(SAMPLE.replace(line, ""))
            self.assertIn(f"no {field} line", str(e.exception))
        for line, bad in (("sms 128\n", "sms 0\n"), ("sms 128\n", "sms many\n"),
                          ("l2_bytes 75497472\n", "l2_bytes -1\n"),
                          ("sms 128\n", "sms 128\nsms 128\n"),
                          ("shape hmma_16816_f32_bf16 4096\n", "shape hmma_16816_f32_bf16\n")):
            with self.assertRaises(ValueError, msg=bad):
                ip.parse_run(SAMPLE.replace(line, bad))
        # A shape the probe does not issue, or another compute capability's, is refused: its
        # flops would give the row a rate of an instruction the card did not run.
        shape = "shape hmma_16816_f32_bf16 4096\n"
        for bad, said in (("shape bogus 0\n", "unknown shape"),
                          ("shape hmma_1688_f32_f16 2048\n", "is not the one of cc 89"),
                          ("shape hmma_884_f32_f16 2048\n", "is not the one of cc 89")):
            with self.assertRaises(ValueError, msg=bad) as e:
                ip.parse_run(SAMPLE.replace(shape, bad))
            self.assertIn(said, str(e.exception), bad)
        self.assertEqual([ip.shape_of(cc) for cc in (70, 75, 80, 86, 89, 120)],
                         ["hmma_884_f32_f16", "hmma_1688_f32_f16"] + ["hmma_16816_f32_bf16"] * 4)
        # From probe 17 a run owes the lines `run` writes (the source's hash, the build's -arch,
        # the SASS MMAs per mma.sync) and the rated memory clock; each must be well formed, and a
        # build for another architecture than the card's (sm_80 code on a cc 8.9 card, sm_120
        # without the accelerated set on Blackwell) is refused.
        for field in ip.HEADER_SINCE:
            line = next(l for l in SAMPLE.splitlines() if l.startswith(field + " ")) + "\n"
            with self.assertRaises(ValueError, msg=field) as e:
                ip.parse_run(SAMPLE.replace(line, ""))
            self.assertIn(f"no {field} line", str(e.exception))
        for line, bad, said in (
                ("arch sm_89\n", "arch sm_80\n", "built for sm_80, not sm_89"),
                ("sass_per_mma 1\n", "sass_per_mma 0\n", "no positive count"),
                ("sass_per_mma 1\n", "sass_per_mma 2\n", "are not the 4096 flops"),
                ("selfcheck ok\n", "selfcheck ok\nrefused the SASS of one x\n",
                 "`run` refused this run: the SASS of one x"),
                ("sass_per_mma 1\n", "sass_per_mma four\n", "no positive count"),
                ("mem_clock_khz 10501000\n", "mem_clock_khz -1\n", "mem_clock_khz is no count"),
                ("mem_clock_khz 10501000\n", "mem_clock_khz fast\n", "mem_clock_khz is no count"),
                ("source_sha256 " + ip.source_sha256() + "\n", "source_sha256 abc\n",
                 "odd source_sha256")):
            with self.assertRaises(ValueError, msg=bad) as e:
                ip.parse_run(SAMPLE.replace(line, bad))
            self.assertIn(said, str(e.exception), bad)
        blackwell = SAMPLE.replace("cc 89", "cc 120").replace("arch sm_89", "arch sm_120")
        with self.assertRaises(ValueError) as e:
            ip.parse_run(blackwell)
        self.assertIn("built for sm_120, not sm_120a", str(e.exception))
        ip.parse_run(blackwell.replace("arch sm_120", "arch sm_120a"))
        # A rated memory clock the device does not give (the attribute reads 0) is a note of
        # the row, not a refusal of the run.
        head, facts = ip.parse_run(SAMPLE.replace("mem_clock_khz 10501000\n", "mem_clock_khz 0\n"))
        row = ip.row_of(head, facts, "isa_probe_89.txt")
        self.assertNotIn("mem_clock_khz", row)
        self.assertIn(ip.MEM_CLOCK_NOT_GIVEN, row["notes"])

    def test_a_row_is_a_run_shown_idle(self):
        # From probe 17 a row is a run whose notes show its GPU idle before, during and after,
        # wherever the file came from: a refused run's output merged by hand is still no row.
        busy = SAMPLE.replace(DURING_IDLE, DURING_IDLE.replace(": 0\n", ": 1 (pid 4242)\n"))
        before = "note compute processes on the probe's GPU before the run: 0\n"
        after = "note compute processes on the probe's GPU after the run: 0\n"
        unread = "note compute processes on the probe's GPU during the run: not read (x)\n"
        waived = "note " + ip.WAIVED + "\n"
        for text, said in (
                (busy, "another compute process on the probe's GPU during the run: 1"),
                (SAMPLE.replace(before, before.replace(": 0", ": 2 (pid 1, 2)")),
                 "before the run: 2"),
                (SAMPLE.replace(after, after.replace(": 0", ": 1 (pid 9)")), "after the run: 1"),
                (SAMPLE.replace(before, ""), "no note of the compute processes before"),
                (SAMPLE.replace(DURING_IDLE, ""), "no note of the compute processes during"),
                (SAMPLE + after, "noted twice"),
                (SAMPLE.replace(DURING_IDLE, unread), "--idle-unchecked` accepts an unread list"),
                (SAMPLE + waived, "noted as waived, yet every process list was read"),
                (busy + waived, "another compute process"),
                (SAMPLE.replace(before, before.replace(": 0", ": some")), "odd compute-process")):
            with self.assertRaises(ValueError, msg=said) as e:
                ip.parse_run(text)
            self.assertIn(said, str(e.exception))
        # A list not read is accepted with the waiver, and the row says so.
        head, _ = ip.parse_run(SAMPLE.replace(DURING_IDLE, unread) + waived)
        self.assertIn(ip.WAIVED, head["notes"])
        anygpu = "note compute processes on any GPU (the probe's not known) after the run: 0\n"
        ip.parse_run(SAMPLE.replace(after, anygpu) + waived)
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace(after, anygpu))
        # A process on some GPU when the probe's is not known may be on the probe's: refused,
        # with the waiver too.
        with self.assertRaises(ValueError) as e:
            ip.parse_run(SAMPLE.replace(after, anygpu.replace(": 0", ": 1 (pid 9)")) + waived)
        self.assertIn("another compute process on any GPU", str(e.exception))

    def test_during_list_without_the_probe_pid(self):
        # In a container whose nvidia-smi prints host-namespace pids, the probe's own pid is not
        # listed: one entry may be the probe (not read, waivable); two never are both the probe
        # (another process seen, never waived).
        bus = ip._bus("0000:01:00.0")
        waived = ["compute processes on the probe's GPU before the run: 0",
                  "compute processes on the probe's GPU after the run: 0", ip.WAIVED]

        def refusal(samples):
            note = ip._during_note([r.split(", ") for r in during_of(samples).splitlines()],
                                   bus, "4321")
            return ip.idle_refusal(waived + [note[5:]])
        self.assertIsNone(refusal([["4242"]]))
        self.assertIsNone(refusal([["[N/A]"]] * 3))
        self.assertIsNone(refusal([]))
        for samples in ([["4242", "5555"]], [["4242"], ["5555"]], [["[N/A]", "[N/A]"]],
                        [["4321", "4242"]], [["4321", "[N/A]"]], [["[N/A]"], ["4242"]]):
            self.assertIn("another compute process", refusal(samples) or "", samples)

        # Two samples printed with one timestamp, or a pid listed twice in one sample, are one
        # process: the probe alone stays idle, and one foreign pid stays not read.
        def one_stamp(pids):
            return [["2026/11/10 12:00:00.000", "00000000:01:00.0", p] for p in pids]

        def note_of(rows, own="4321"):
            return ip._during_note(rows, bus, own)
        self.assertTrue(note_of(one_stamp(["4321"] * 5)).endswith(": 0"))
        self.assertTrue(note_of(one_stamp(["4321", "4321"])).endswith(": 0"))
        self.assertIn("not read", note_of(one_stamp(["4242"] * 5)))
        # Unnamed rows cannot be told apart: two in one sample may be two processes.
        self.assertIn("not read", note_of(one_stamp(["[N/A]"])))
        self.assertIn(": 2 (pid 4242, [N/A];", note_of(one_stamp(["[N/A]", "[N/A]", "4242"])))
        # A count other than 0 always names what it counted.
        for rows in (one_stamp(["4321", "[N/A]", "[N/A]"]), one_stamp(["4321", "4242", "4242"])):
            note = note_of(rows)
            self.assertRegex(note, r": [1-9]\d* \(pid [^)]+\)$", rows)
        # An earlier probe's run carries no such notes and owes none.
        ip.parse_run(earlier_run(16))

    def run_faked(self, d, *, before="", after="", during=None, probe_rc=0, mem_khz=None,
                  waive=False, sass=SASS_89, no_bus=False):
        """isa_probe.py run with nvcc, cuobjdump, nvidia-smi and the probe binary faked; the
        exit code, the --out path and stderr."""
        pid = 4321
        if during is None:
            during = during_of([[str(pid)]] * 20)
        lines = [l for l in SAMPLE.splitlines() if not l.startswith(RUN_LINES)]
        probe_out = "\n".join(lines) + "\n"
        if not no_bus:
            probe_out = probe_out.replace(
                "nvcc 13.4.92", "nvcc 13.4.92\nnote device 0 is the GPU at PCI 0000:01:00.0")
        if mem_khz is not None:
            probe_out = probe_out.replace("mem_clock_khz 10501000", f"mem_clock_khz {mem_khz}")
        if probe_rc:
            probe_out = probe_out.replace("selfcheck ok", "selfcheck FAIL tensor rate")
        apps = iter((before, after))

        class Proc:
            def __init__(self, text, rc, pid):
                self.text, self.returncode, self.pid = text, rc, pid

            def terminate(self):
                pass

            def communicate(self):
                return self.text, None

        def run(cmd, **kw):
            exe = os.path.basename(cmd[0])
            if exe == "nvcc":
                return ip.subprocess.CompletedProcess(cmd, 0)
            if exe == "cuobjdump":
                return ip.subprocess.CompletedProcess(cmd, 0, sass, "")
            self.assertEqual((exe, cmd[1]), ("nvidia-smi", "--query-compute-apps=gpu_bus_id,pid"))
            return ip.subprocess.CompletedProcess(cmd, 0, next(apps), "")

        def popen(cmd, **kw):
            if os.path.basename(cmd[0]) != "nvidia-smi":
                return Proc(probe_out, probe_rc, pid)
            if cmd[1].startswith("--query-gpu"):
                return Proc("00000000:01:00.0, P0, 2520, 10501\n" * 20, 0, 1)
            self.assertEqual(cmd[1], "--query-compute-apps=timestamp,gpu_bus_id,pid")
            return Proc(during, 0, 2)

        out = os.path.join(d, "isa_probe_89_1.txt")
        args = argparse.Namespace(arch="89", nvcc="nvcc", cuobjdump="cuobjdump", out=out,
                                  idle_unchecked=waive)
        err = io.StringIO()
        with mock.patch.object(ip.subprocess, "run", run), \
                mock.patch.object(ip.subprocess, "Popen", popen), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = ip.cmd_run(args)
        return rc, out, err.getvalue()

    def test_run_writes_a_refused_run_under_another_name(self):
        # `run` writes --out only for a run that is a row; a refused run's output goes to
        # <out>.refused, which merge does not take, so "merge the first run" cannot merge it.
        other = during_of([["4321", "4242", "02:5555"]] * 5)
        na = during_of([["[N/A]"]] * 5)
        for kw, rc_want, said in (
                ({}, 0, None),
                ({"during": other}, 3, "during the run: 1 (pid 4242)"),
                ({"before": "00000000:01:00.0, 999\n"}, 3, "before the run: 1 (pid 999)"),
                ({"after": "00000000:01:00.0, 999\n"}, 3, "after the run: 1 (pid 999)"),
                ({"probe_rc": 3}, 3, "the probe exited 3"),
                # A process on another GPU is not the probe's GPU's.
                ({"before": "00000000:02:00.0, 999\n"}, 0, None),
                # During the run the probe is one of the listed processes. A list without its own
                # pid that shows nothing, or one entry throughout (a foreign-namespace pid, or
                # [N/A]: maybe the probe), was not read: refused, accepted with the waiver.
                ({"during": ""}, 3, "no process listed"),
                ({"during": na}, 3, "one process listed throughout, pid [N/A]"),
                ({"during": during_of([["4242"]] * 5)}, 3, "one process listed throughout"),
                ({"during": "", "waive": True}, 0, None),
                ({"during": na, "waive": True}, 0, None),
                ({"during": during_of([["4242"]] * 5), "waive": True}, 0, None),
                # Two entries where the probe can be at most one, or an entry beside the probe's
                # own pid, are another process seen: refused, waiver or not.
                ({"during": other, "waive": True}, 3, "during the run: 1 (pid 4242)"),
                ({"during": during_of([["4321", "[N/A]"]]), "waive": True}, 3,
                 "during the run: 1 (pid [N/A])"),
                ({"during": during_of([["4242", "5555"]] * 3), "waive": True}, 3,
                 "1 (pid 4242, 5555; the probe's own pid 4321 not among them, up to 2"),
                ({"during": during_of([["4242"], ["5555"]]), "waive": True}, 3,
                 "1 (pid 4242, 5555; the probe's own pid 4321 not among them, up to 1"),
                ({"during": during_of([["[N/A]", "[N/A]"]] * 3), "waive": True}, 3,
                 "1 (pid [N/A]; the probe's own pid 4321 not among them, up to 2"),
                # The probe's GPU not known (no bus id printed): a process on any GPU may be on
                # the probe's, refused; none anywhere is a list that cannot say which GPU, waived.
                ({"no_bus": True, "before": "00000000:02:00.0, 999\n", "waive": True}, 3,
                 "on any GPU (the probe's not known) before the run: 1 (pid 999)"),
                ({"no_bus": True}, 3, "the probe's GPU not known"),
                ({"no_bus": True, "waive": True}, 0, None),
                ({"mem_khz": 0}, 0, None),
                # The SASS of one mma.sync not the shape's flops (an opcode of no known shape):
                # refused by `run`, and the file says so, so a copy under the row's name is too.
                ({"sass": SASS_89.replace("HMMA.16816.F32.BF16", "HMMA.F32.BF16")}, 3,
                 "is not its 4096 flops")):
            with tempfile.TemporaryDirectory() as d:
                rc, out, err = self.run_faked(d, **kw)
                self.assertEqual(rc, rc_want, (kw, err))
                refused = out + ip.REFUSED
                if rc_want == 0:
                    self.assertTrue(os.path.exists(out), kw)
                    self.assertFalse(os.path.exists(refused), kw)
                    with open(out) as f:
                        text = f.read()
                    head, facts = ip.parse_run(text)
                    self.assertEqual(ip.WAIVED in head["notes"], kw.get("waive", False), kw)
                    if kw.get("mem_khz") == 0:
                        self.assertIn(ip.MEM_CLOCK_NOT_GIVEN,
                                      ip.row_of(head, facts, "x")["notes"])
                    continue
                self.assertFalse(os.path.exists(out), kw)
                self.assertTrue(os.path.exists(refused), kw)
                self.assertIn(said, err, kw)
                self.assertIn("no row to merge", err, kw)
                inputs = os.path.join(d, "inputs")
                os.mkdir(inputs)
                for path in (refused, out):
                    if path == out:
                        shutil.copy(refused, out)
                    args = argparse.Namespace(input=path, inputs=inputs,
                                              table=os.path.join(d, "t.json"))
                    with contextlib.redirect_stderr(io.StringIO()), \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(ip.cmd_merge(args), 1, (kw, path))
                    self.assertEqual(os.listdir(inputs), [], kw)

    def test_sass_per_mma(self):
        # One mma.sync of the build is the MMAs of isa_probe_mma_unit, read with sass_counts'
        # parser and shapes; on sm_70 four HMMA.884 STEPs, whose 8x8x4 products are the
        # m8n8k4's 2048 flops, as `run` checks before it calls the output a row.
        mmas = ip.sass_mma_unit(SASS_70)
        self.assertEqual(len(mmas), 4)
        self.assertEqual(sum(2 * m * n * k for _, (m, n, k) in mmas),
                         ip.SHAPE_FLOPS["hmma_884_f32_f16"])
        for bad in (SASS_70.replace("isa_probe_mma_unit", "other"),
                    SASS_70.replace("HMMA.884", "FFMA.884")):
            with self.assertRaises(ValueError):
                ip.sass_mma_unit(bad)

    def test_merge_refuses_without_writing(self):
        # A run merge cannot parse leaves the inputs directory and the table as they were, and the
        # command says why and returns 1 (no traceback).
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "run.txt")
            with open(bad, "w", newline="\n") as f:
                f.write(SAMPLE.replace("fact shared_latency 29.5 clk\n",
                                       "fact shared_latency nan clk\n"))
            inputs = os.path.join(d, "inputs")
            os.mkdir(inputs)
            args = argparse.Namespace(input=bad, inputs=inputs, table=os.path.join(d, "t.json"))
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ip.cmd_merge(args), 1)
            self.assertIn("positive finite", err.getvalue())
            self.assertEqual(os.listdir(inputs), [])
            self.assertFalse(os.path.exists(args.table))

    def test_merge_restores_when_the_table_cannot_be_built(self):
        # A valid run next to inputs the table cannot be built from (here the same architecture
        # under another file name, then a malformed neighbour) merges nothing: an older run of
        # the architecture keeps its text, a new destination is not left behind, and the table
        # is not written.
        older = SAMPLE.replace("fact shared_latency 29.5 clk\n", "fact shared_latency 30.5 clk\n")
        for neighbour, text, said in (("isa_probe_ada.txt", SAMPLE, "already given"),
                                      ("isa_probe_86.txt", "garbage line\n", "isa_probe_86.txt")):
            for before in (older, None):
                with tempfile.TemporaryDirectory() as d:
                    run = os.path.join(d, "run.txt")
                    inputs = os.path.join(d, "inputs")
                    os.mkdir(inputs)
                    dest = os.path.join(inputs, "isa_probe_89.txt")
                    for path, body in ((run, SAMPLE), (os.path.join(inputs, neighbour), text),
                                       (dest, before)):
                        if body is not None:
                            with open(path, "w", newline="\n") as f:
                                f.write(body)
                    args = argparse.Namespace(input=run, inputs=inputs,
                                              table=os.path.join(d, "t.json"))
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(ip.cmd_merge(args), 1)
                    self.assertIn(said, err.getvalue())
                    self.assertIn("nothing merged", err.getvalue())
                    self.assertFalse(os.path.exists(args.table))
                    if before is None:
                        self.assertEqual(os.listdir(inputs), [neighbour])
                    else:
                        with open(dest) as f:
                            self.assertEqual(f.read(), older)

    def test_rows_come_from_the_checked_in_probe(self):
        # A row is evidence for the probe that produced it. When the probe changes what it times
        # (which CTAs count, what a link is, which lines a chase reads), the rows of the earlier
        # source no longer say what the checked-in method measures: rerun the probe on that
        # architecture's card and merge. A row is from the checked-in probe when its version and
        # the sha256 of the source it was built from (written by `run`) are the checked-in
        # file's: a change of the .cu without a version bump makes every row stale too. This
        # fails until every row is from the checked-in source, naming each stale row and the
        # commands that re-take it.
        with open(ip.DEFAULT_TABLE) as f:
            table = json.load(f)
        now, sha = ip.checked_in_probe(), ip.source_sha256()
        stale = []
        for key, row in table["arches"].items():
            probe, theirs = row["tool"]["probe"], row["tool"].get("source_sha256")
            if probe == now and theirs == sha:
                continue
            why = (f"{probe}, " if probe != now else "") + \
                (f"source {theirs[:12]}" if theirs else "no source hash")
            nvcc = "<the nvcc of CUDA 12.8>" if key == "70" else "<nvcc>"
            stale.append(
                f"  sm_{key} ({row['card']}; {why}): on an otherwise idle {row['card']}, run "
                f"n = 1, 2, 3 and merge the first:\n"
                f"    python3 tools/bench/isa_probe/isa_probe.py run --arch {key} --nvcc {nvcc} "
                f"--out isa_probe_{key}_<n>.txt\n"
                f"    python3 tools/bench/isa_probe/isa_probe.py merge --input "
                f"isa_probe_{key}_1.txt")
        if stale:
            self.fail(f"rows not measured by the checked-in probe ({now}, isa_probe.cu sha256 "
                      f"{sha[:12]}); re-take each:\n" + "\n".join(stale))

    def test_refusals(self):
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace("end\n", ""))
        # A run whose self-check failed, or that reports none, is no row.
        with self.assertRaises(ValueError) as failed:
            ip.parse_run(SAMPLE.replace("selfcheck ok\n", "selfcheck FAIL DRAM latency not above "
                                                             "1.2x L2 at w2\nselfcheck FAIL shape "
                                                             "hmma_884_f32_f16 is not the card's\n"))
        self.assertIn("DRAM latency", str(failed.exception))
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace("selfcheck ok\n", ""))
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace("fact shared_latency 29.5 clk\n", ""))
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE + "garbage line\n")
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace("fact mma_cycles_per_sm 8.012 clk", "fact mma_cycles_per_sm"))

    def test_a_run_owes_the_facts_of_its_probe_version(self):
        # A run carries every fact its own probe version prints and none a later version added:
        # an earlier probe's saved run still parses without the later facts, the checked-in
        # probe's run does not, and a run that names no version owes them all.
        now = ip.checked_in_probe()
        newest = max(ip.first_probe(f) for f in ip.FACTS)
        self.assertEqual(int(now.split()[1]), newest, "a fact names a probe that is not checked in")
        held = "fact dram_held_latency_ns_w4 302.6 ns\n"
        self.assertIn(held, SAMPLE)
        with self.assertRaises(ValueError) as e:
            ip.parse_run(SAMPLE.replace(held, ""))
        self.assertIn("dram_held_latency_ns_w4", str(e.exception))
        for version in sorted({ip.first_probe(f) for f in ip.FACTS} - {0}):
            # Without the facts and the header lines of `version` and later (a probe-16 run has
            # no source hash, -arch, sass_per_mma or memory clock).
            head, facts = ip.parse_run(earlier_run(version - 1))
            self.assertEqual(set(facts),
                             {f for f in ip.FACTS if ip.first_probe(f) < version}, version)
            # The same run with one fact of the next version is not that version's output.
            with self.assertRaises(ValueError, msg=version) as e:
                ip.parse_run(SAMPLE.replace("probe " + now, f"probe isa_probe {version - 1}"))
            self.assertIn("later probe", str(e.exception))
        for odd in ("probe isa_probe", "probe isa_probe fifteen", "probe other 15"):
            with self.assertRaises(ValueError, msg=odd):
                ip.parse_run(SAMPLE.replace("probe " + now, odd))
        # A version past the checked-in probe's owes what this parser cannot know: refused.
        with self.assertRaises(ValueError) as e:
            ip.parse_run(SAMPLE.replace("probe " + now, f"probe isa_probe {newest + 1}"))
        self.assertIn("newer than the checked-in probe", str(e.exception))

    def test_merge_in_place(self):
        # Merging the file that already sits in the inputs directory under its architecture's
        # name keeps its text (the destination is not opened before the input is read).
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "isa_probe_89.txt")
            with open(path, "w", newline="\n") as f:
                f.write(SAMPLE)
            args = argparse.Namespace(input=path, inputs=d, table=os.path.join(d, "table.json"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ip.cmd_merge(args), 0)
            with open(path) as f:
                self.assertEqual(f.read(), SAMPLE)
            with open(args.table) as f:
                self.assertEqual(list(json.load(f)["arches"]), ["89"])

    def test_table_is_its_regeneration(self):
        # The committed table is exactly what the committed probe outputs give: a new architecture
        # is one file in src/core/data/isa_probe/ plus `isa_probe.py regenerate`.
        with open(ip.DEFAULT_TABLE) as f:
            table = json.load(f)
        self.assertEqual(table, ip.build_table(ip.DEFAULT_INPUTS))
        self.assertGreaterEqual(len(table["arches"]), 1, "no architecture measured yet")

    def test_table_schema(self):
        with open(ip.DEFAULT_TABLE) as f:
            table = json.load(f)
        self.assertEqual(table["schema"], ip.SCHEMA)
        self.assertIsInstance(table["note"], str)
        for key, row in table["arches"].items():
            self.assertEqual(row["provenance"], "Measured", key)
            self.assertEqual(key, ip.arch_key(row["cc"], row["mma_shape"]), key)
            self.assertTrue(os.path.exists(os.path.join(ip.DEFAULT_INPUTS, row["source"])), key)
            for field in ("card", "cc", "sms", "l2_bytes", "mma_shape", "mma_flops", "tool",
                          "facts"):
                self.assertIn(field, row, f"{key} lacks {field}")
            for field in ("probe", "nvcc", "driver", "runtime"):
                self.assertIn(field, row["tool"], f"{key} tool lacks {field}")
            # Every fact the row's own probe version prints (and the derived tensor rate), none
            # of a later one: a row keeps lacking those until its card is probed again.
            probe = int(row["tool"]["probe"].split()[1])
            self.assertEqual(set(row["facts"]),
                             {f for f in ip.FACTS if ip.first_probe(f) <= probe}, key)
            for field in ("sms", "l2_bytes"):
                self.assertGreater(row[field], 0, f"{key} {field}")
            for field in ("driver", "runtime"):
                self.assertTrue(row["tool"][field].isdigit(), f"{key} tool {field}")
            # From probe 17: the build's -arch is the row's architecture, the source hash is
            # recorded, the rated memory clock is there, and the SASS MMAs of one mma.sync are
            # the shape's flops by sass_counts' table (the unit the kernel facts count MMAs in):
            # the row's MMA cycles over sass_per_mma are the price of one counted SASS MMA.
            if probe >= 17:
                self.assertEqual(row["tool"]["arch"], f"sm_{key}", key)
                self.assertRegex(row["tool"]["source_sha256"], ip.SHA256, key)
                if "mem_clock_khz" in row:
                    self.assertGreater(row["mem_clock_khz"], 0, key)
                else:
                    self.assertIn(ip.MEM_CLOCK_NOT_GIVEN, row["notes"], key)
                token = row["mma_shape"].split("_")[1]
                m, n, k = SC.MMA_SHAPES[token]
                self.assertEqual(row["sass_per_mma"] * 2 * m * n * k, row["mma_flops"],
                                 f"{key}: {row['sass_per_mma']} SASS {token} MMAs per mma.sync")
            # The measured tensor rate against the generation row's bf16 rate (core/generation.h)
            # where both are the same operation: the native BF16 m16n8k16 with FP32 accumulate,
            # cc 80 and up. Before that the row is a fitted aggregate of an emulated path (Volta's
            # FP32 FMAs, Turing's f16 k8 pairs) and the probe's native shape is another number.
            if row["cc"] >= 80:
                rate = generation_bf16_rate(row["cc"])
                self.assertIsNotNone(rate, f"{key}: no generation row")
                measured = row["facts"]["tensor_flops_per_sm_clock"]["value"]
                self.assertLess(abs(measured - rate) / rate, 0.05,
                                f"{key}: measured {measured} flop/clk/SM, the generation row says "
                                f"{rate}; the row is what moves")
            for name, v in row["facts"].items():
                self.assertGreater(v["value"], 0.0, f"{key} {name}")
                self.assertEqual(v["unit"], ip.unit_of(name), f"{key} {name}")
            # The levels in order, and loaded latency never under the lone warp's: in clocks for
            # the L1, in wall ns beyond the SM (what the probe's own self-check holds).
            fa = row["facts"]
            self.assertLess(fa["l1_latency_w1"]["value"], fa["l2_latency_w1"]["value"], key)
            for lv in ("dram", "dram_tlbwarm"):
                self.assertGreater(fa[f"{lv}_latency_ns_w1"]["value"],
                                   1.2 * fa["l2_latency_ns_w1"]["value"], f"{key} {lv}")
            for w in (2, 4, 8):
                self.assertGreaterEqual(fa[f"l1_latency_w{w}"]["value"] * 1.05,
                                        fa["l1_latency_w1"]["value"], f"{key} l1 w{w}")
                for lv in ("l2", "dram", "dram_tlbwarm"):
                    self.assertGreaterEqual(fa[f"{lv}_latency_ns_w{w}"]["value"] * 1.05,
                                            fa[f"{lv}_latency_ns_w1"]["value"], f"{key} {lv} w{w}")
            # A link anywhere in the buffer is never the shorter of the two DRAM levels (the
            # probe's own self-check from version 14 on, where no two links share a line; the rows
            # of earlier probes, whose links did, are named by the version test above instead).
            if probe >= 14:
                for w in (1, 2, 4, 8):
                    self.assertGreaterEqual(fa[f"dram_latency_ns_w{w}"]["value"],
                                            0.97 * fa[f"dram_tlbwarm_latency_ns_w{w}"]["value"],
                                            f"{key} dram under dram_tlbwarm at w{w}")
            # From probe 15 on every level is measured again beside a working warp on each SM,
            # and DRAM with 32 chasing lanes a warp: the same order holds there, a 32-lane warp
            # waits at least as long as a one-lane warp at the same warp count, and the SM clock
            # of the held chases is the tensor loop's settled clock within 10 % (both are an SM
            # at work; the lone chases' clock may be anything).
            # (The same chase on half the SMs, from probe 16 on, is held to the same two bounds;
            # how it stands against the whole chip's is the fact it is there for.)
            if probe >= 15:
                for lv in ("dram_lanes",) + (("dram_lanes_half",) if probe >= 16 else ()):
                    for tag in ("", "_held"):
                        for w in (1, 2, 4, 8):
                            lanes = fa[f"{lv}{tag}_latency_ns_w{w}"]["value"]
                            self.assertGreaterEqual(
                                lanes, 0.95 * fa[f"dram{tag}_latency_ns_w{w}"]["value"],
                                f"{key} {lv}{tag} under dram at w{w}")
                            self.assertGreaterEqual(
                                lanes * 1.05, fa[f"{lv}{tag}_latency_ns_w1"]["value"],
                                f"{key} {lv}{tag} w{w}")
                self.assertLess(fa["l1_held_latency_w1"]["value"],
                                fa["l2_held_latency_w1"]["value"], key)
                for w in (1, 2, 4, 8):
                    self.assertGreater(fa[f"dram_held_latency_ns_w{w}"]["value"],
                                       1.2 * fa[f"l2_held_latency_ns_w{w}"]["value"],
                                       f"{key} held dram at w{w}")
                    self.assertGreaterEqual(
                        fa[f"dram_held_latency_ns_w{w}"]["value"],
                        0.97 * fa[f"dram_tlbwarm_held_latency_ns_w{w}"]["value"],
                        f"{key} held dram under dram_tlbwarm at w{w}")
                held, mma = fa["sm_clock_mhz_chase_held"]["value"], fa["sm_clock_mhz_mma"]["value"]
                self.assertLess(abs(held - mma) / mma, 0.10,
                                f"{key}: held chases at {held} MHz, the tensor loop at {mma}")


if __name__ == "__main__":
    unittest.main()
