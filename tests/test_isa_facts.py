#!/usr/bin/env python3
"""The ISA facts table (src/core/data/isa_facts.json), its committed probe outputs
(src/core/data/isa_probe/) and the probe's parser (#613 PR-2): the schema is pinned (every row
Measured with its source file, every fact present with a positive value in its unit, the card and
tool named, the levels in order), the table equals its regeneration from the inputs, every row is
from the checked-in probe version, a probe output round-trips into a row, and odd or incomplete
outputs are refused. No GPU, no CUDA toolkit."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "isa_probe", os.path.join(ROOT, "tools", "bench", "isa_probe", "isa_probe.py"))
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)


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

# A run in the checked-in probe's format (its version read from isa_probe.cu).
SAMPLE = "probe " + ip.checked_in_probe() + """
card NVIDIA GeForce RTX 4090
cc 89
sms 128
l2_bytes 75497472
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
        ("imad", 4.5), ("iadd", 4.0), ("lop3", 4.2), ("prmt", 4.4), ("shf", 4.5), ("hfma2", 5.0),
        ("ffma", 4.1), ("shfl", 24.0), ("cvt_i2f_f2i", 6.0), ("cvt_f32_f16", 5.5))) + \
    "note the SM clock ranged 2400..2535 MHz over the L2 / DRAM chases\nselfcheck ok\nend\n"


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
        self.assertEqual(row["notes"], ["the SM clock ranged 2400..2535 MHz over the L2 / DRAM chases"])
        self.assertAlmostEqual(row["facts"]["dram_latency_w8"]["value"], 584.0)
        # The tensor rate is derived from the parser's flops table over the measured cycles, not
        # read from the probe: a Volta run that printed 512 for m8n8k4 (four 8x8x4 products a
        # warp instruction: 2048) still gives V100's ~1024 flop/clk/SM.
        self.assertAlmostEqual(row["facts"]["tensor_flops_per_sm_clock"]["value"],
                               round(4096 / 8.012, 3))
        volta = SAMPLE.replace("shape hmma_16816_f32_bf16 4096", "shape hmma_884_f32_f16 512") \
            .replace("cc 89", "cc 70").replace("fact mma_cycles_per_sm 8.012 clk",
                                               "fact mma_cycles_per_sm 2.014 clk")
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
        # fact of the card: read past, never in a row.
        head, facts = ip.parse_run(SAMPLE.replace(
            "fact mma_cycles_per_sm 8.012 clk\n",
            "fact resident_ctas_per_sm 4.000 count\nfact mma_cycles_per_sm 8.012 clk\n"))
        self.assertNotIn("resident_ctas_per_sm", ip.row_of(head, facts, "x")["facts"])

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
        # version no longer say what the checked-in method measures: rerun the probe on that
        # architecture's card and merge. This fails until every row is from the checked-in probe.
        with open(ip.DEFAULT_TABLE) as f:
            table = json.load(f)
        now = ip.checked_in_probe()
        stale = {key: row["tool"]["probe"] for key, row in table["arches"].items()
                 if row["tool"]["probe"] != now}
        self.assertEqual(stale, {}, f"rows not measured by the checked-in probe ({now}): rerun "
                                    "`isa_probe.py run` on a card of each and `merge` the output")

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
            earlier = "".join(
                line + "\n" for line in SAMPLE.splitlines()
                if not (line.startswith("fact ") and ip.first_probe(line.split()[1]) >= version))
            earlier = earlier.replace("probe " + now, f"probe isa_probe {version - 1}")
            head, facts = ip.parse_run(earlier)
            self.assertEqual(set(facts),
                             {f for f in ip.FACTS if ip.first_probe(f) < version}, version)
            # The same run with one fact of the next version is not that version's output.
            with self.assertRaises(ValueError, msg=version) as e:
                ip.parse_run(SAMPLE.replace("probe " + now, f"probe isa_probe {version - 1}"))
            self.assertIn("later probe", str(e.exception))
        for odd in ("probe isa_probe", "probe isa_probe fifteen", "probe other 15"):
            with self.assertRaises(ValueError, msg=odd):
                ip.parse_run(SAMPLE.replace("probe " + now, odd))

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
