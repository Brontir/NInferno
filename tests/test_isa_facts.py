#!/usr/bin/env python3
"""The ISA facts table (src/core/data/isa_facts.json), its committed probe outputs
(src/core/data/isa_probe/) and the probe's parser (#613 PR-2): the schema is pinned (every row
Measured with its source file, every fact present with a positive value in its unit, the card and
tool named, the levels in order), the table equals its regeneration from the inputs, a probe output
round-trips into a row, and odd or incomplete outputs are refused. No GPU, no CUDA toolkit."""

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

SAMPLE = """probe isa_probe 7
card NVIDIA GeForce RTX 4090
cc 89
sms 128
l2_bytes 75497472
driver 13040
runtime 13040
nvcc 13.4.92
shape hmma_16816_f32_bf16 4096
fact resident_ctas_per_sm 4.000 count
fact mma_cycles_per_sm 8.012 clk
fact mma_cycles_per_partition 32.048 clk
fact tensor_flops_per_sm_clock 511.2 flop
fact mma_dependent_latency 33.1 clk
fact fp32_cycles_per_warp_instruction 1.004 clk
fact int32_cycles_per_warp_instruction 2.01 clk
""" + "".join(f"fact {lv}_latency_w{w} {v + 3 * w:.1f} clk\n"
              for lv, v in (("l1", 34.0), ("l2", 260.0), ("dram", 560.0), ("dram_tlbwarm", 520.0))
              for w in (1, 2, 4, 8)) + \
    "".join(f"fact {lv}_latency_ns_w{w} {(v + 3 * w) / 2.52:.1f} ns\n"
            for lv, v in (("l2", 260.0), ("dram", 560.0), ("dram_tlbwarm", 520.0))
            for w in (1, 2, 4, 8)) + \
    "fact sm_clock_mhz_chase 2520.0 mhz\nfact shared_latency 29.5 clk\n" \
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
                                                  "sm_clock_mhz_chase", "resident_ctas_per_sm",
                                                  "tensor_flops_per_sm_clock")],
                         ["clk", "ns", "mhz", "count", "flop"])

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
            # Every fact of the table's list, except those a later probe added (the ALU
            # latencies), which a row keeps lacking until its card is probed again.
            self.assertTrue(set(ip.FACTS) - ip.OPTIONAL_FACTS <= set(row["facts"]), key)
            self.assertTrue(set(row["facts"]) <= set(ip.FACTS), key)
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


if __name__ == "__main__":
    unittest.main()
