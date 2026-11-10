#!/usr/bin/env python3
"""The ISA facts table (src/core/data/isa_facts.json) and its probe's parser (#613 PR-2): the
schema is pinned (every row Measured, every fact present with a positive value in SM clocks, the
card and tool named), a probe output round-trips into a row, and odd or incomplete outputs are
refused. No GPU, no CUDA toolkit."""

import importlib.util
import json
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "isa_probe", os.path.join(ROOT, "tools", "bench", "isa_probe", "isa_probe.py"))
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)

SAMPLE = """probe isa_probe 1
card GeForce RTX 4090
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
fact mma_dependent_latency 33.1 clk
fact fp32_cycles_per_warp_instruction 1.004 clk
fact int32_cycles_per_warp_instruction 2.01 clk
""" + "".join(f"fact {lv}_latency_w{w} {v + 3 * w:.1f} clk\n"
              for lv, v in (("l1", 34.0), ("l2", 260.0), ("dram", 560.0)) for w in (1, 2, 4, 8)) + \
    "fact shared_latency 29.5 clk\nend\n"


class IsaFacts(unittest.TestCase):
    def test_parse_and_row(self):
        head, facts = ip.parse_run(SAMPLE)
        self.assertEqual(head["card"], "GeForce RTX 4090")
        self.assertEqual(set(facts), set(ip.FACTS))
        row = ip.row_of(head, facts)
        self.assertEqual(row["provenance"], "Measured")
        self.assertEqual((row["cc"], row["sms"], row["mma_shape"], row["mma_flops"]),
                         (89, 128, "hmma_16816_f32_bf16", 4096))
        self.assertEqual(row["tool"]["nvcc"], "13.4.92")
        self.assertAlmostEqual(row["facts"]["dram_latency_w8"]["value"], 584.0)
        self.assertEqual(ip.arch_key(120, "hmma_16816_f32_bf16 4096"), "120a")
        self.assertEqual(ip.arch_key(75, "hmma_1688_f32_f16 2048"), "75")

    def test_refusals(self):
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace("end\n", ""))
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace("fact shared_latency 29.5 clk\n", ""))
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE + "garbage line\n")
        with self.assertRaises(ValueError):
            ip.parse_run(SAMPLE.replace("fact mma_cycles_per_sm 8.012 clk", "fact mma_cycles_per_sm"))

    def test_table_schema(self):
        with open(os.path.join(ROOT, "src", "core", "data", "isa_facts.json")) as f:
            table = json.load(f)
        self.assertEqual(table["schema"], ip.SCHEMA)
        self.assertIsInstance(table["note"], str)
        for key, row in table["arches"].items():
            self.assertEqual(row["provenance"], "Measured", key)
            self.assertEqual(key, ip.arch_key(row["cc"], row["mma_shape"]), key)
            for field in ("card", "cc", "sms", "mma_shape", "mma_flops", "tool", "date", "facts"):
                self.assertIn(field, row, f"{key} lacks {field}")
            for field in ("probe", "nvcc", "driver", "runtime"):
                self.assertIn(field, row["tool"], f"{key} tool lacks {field}")
            self.assertEqual(set(row["facts"]), set(ip.FACTS), key)
            for name, v in row["facts"].items():
                self.assertGreater(v["value"], 0.0, f"{key} {name}")
                self.assertEqual(v["unit"], "flop" if name == "tensor_flops_per_sm_clock" else "clk",
                                 f"{key} {name}")
            # Loaded latency never falls under the lone warp's; DRAM is further than L2 than L1.
            fa = row["facts"]
            for lv in ("l1", "l2", "dram"):
                self.assertGreaterEqual(fa[f"{lv}_latency_w8"]["value"] * 1.05,
                                        fa[f"{lv}_latency_w1"]["value"], f"{key} {lv}")
            self.assertLess(fa["l1_latency_w1"]["value"], fa["l2_latency_w1"]["value"], key)
            self.assertLess(fa["l2_latency_w1"]["value"], fa["dram_latency_w1"]["value"], key)


if __name__ == "__main__":
    unittest.main()
