#!/usr/bin/env python3
"""The per-SM DRAM bytes-in-flight probe (tools/bench/warp_inflight, Refs #690, #735): its parser
and derivation. A run in the checked-in probe's format round-trips into rows with b_w = rate x L0
/ W and P = the largest rate; the probe source and the parser agree on the variants, the grid
and the columns; odd, incomplete, foreign or busy runs are refused. No GPU, no CUDA toolkit."""

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
    "warp_inflight", os.path.join(ROOT, "tools", "bench", "warp_inflight", "warp_inflight.py"))
wi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wi)

with open(wi.SOURCE) as f:
    CU = f.read()

# A made-up card that follows the law exactly: L = 256 ns a fetch, P = 90 GB/s an SM, 2800 MHz.
L_NS, P_GBPS, MHZ = 256.0, 90.0, 2800.0
HEAD = {"sms": "60", "max_warps_per_sm": "48"}
NO_ROW = {("ring4", 48)}


def law(variant, w):
    return min(P_GBPS, w * wi.inflight_of(variant) * wi.WARP_LOAD / L_NS)


def row_line(mode, k, v, w, gbps=None):
    g = law(v, w) if gbps is None else gbps
    return (f"row {mode} {k} {v} {wi.inflight_of(v)} {w} {g:.4f} {g * 0.99:.4f} {g * 1.01:.4f} "
            f"{L_NS:.1f} {MHZ:.0f} {w:.3f}\n")


ROWS = sorted(wi.expected_points(HEAD, NO_ROW),
              key=lambda p: (wi.MODES.index(p[0]), p[1], list(wi.VARIANTS).index(p[2]), p[3]))
SAMPLE = (
    f"source_sha256 {wi.source_sha256()}\narch sm_89\nsass_ldg128 " +
    " ".join(f"{v}:{wi.sass_loads_of(v)}" for v in wi.VARIANTS) + "\n"
    f"probe {wi.checked_in_probe()}\ncard NVIDIA GeForce RTX 4070 Ti\ncc 89\nsms 60\n"
    "l2_bytes 50331648\nmem_clock_khz 10501000\nmax_warps_per_sm 48\nsmem_per_sm 102400\n"
    "driver 13040\nruntime 13020\nnvcc 13.2.86\nnote device 0 is the GPU at PCI 0000:01:00.0\n"
    "columns " + " ".join(wi.COLUMNS) + "\nnote warm-up: 300 ms of every SM streaming\n"
    "note placed ring4 w48: 1 of 2 CTAs fit an SM (registers), no row\n" +
    "".join(row_line(*p) for p in ROWS) +
    "span_ns 1762862400000000000 1762862404000000000\nselfcheck ok\nend\n"
    "note compute processes on the probe's GPU before the run: 0\n"
    "note compute processes on the probe's GPU after the run: 0\n"
    "note compute processes on the probe's GPU during the run besides the probe (nvidia-smi "
    "every 200 ms): 0\n")
# The clocks over the measured span: steady, P0, at the rated 10501 MHz.
CLOCKS = ("note clocks over the measured span (38 samples, nvidia-smi every 100 ms): memory "
          "10501..10501 MHz, SM 2805..2820 MHz, P-state P0\n")
SAMPLE += CLOCKS


class WarpInflight(unittest.TestCase):
    def test_round_trip(self):
        head, rows = wi.parse_run(SAMPLE)
        self.assertEqual(len(rows), len(ROWS))
        self.assertEqual(head["card"], "NVIDIA GeForce RTX 4070 Ti")
        s = wi.derive(rows, L_NS)
        # Little's law on the law's own rows: b_w is the bytes in flight while the rate is linear.
        for mode in wi.MODES:
            for v in wi.VARIANTS:
                d = s["variants"][mode].get(v)
                if v == "ring4" or d is None:
                    continue
                self.assertAlmostEqual(d["b_w_w1"], wi.inflight_of(v) * 512, delta=1.0)
                self.assertEqual(d["warp_bytes_in_flight"], wi.inflight_of(v) * 512)
        # chain1 runs 2 GB/s a warp: linear up to 45 warps, so through W = 32 free and, with 90
        # of 96 GB/s at W = 48 (94 %), through 48 placed. chain4 saturates past 11 warps (W = 12
        # still keeps 94 %).
        self.assertEqual(s["variants"]["free"]["chain1"]["linear_to_w"], 32)
        self.assertEqual(s["variants"]["placed"]["chain1"]["linear_to_w"], 48)
        self.assertEqual(s["variants"]["free"]["chain4"]["linear_to_w"], 12)
        self.assertAlmostEqual(s["p_gbps"]["free"], P_GBPS, delta=1e-3)
        self.assertAlmostEqual(s["p_bytes_per_clk"]["placed"], P_GBPS * 1e3 / MHZ, delta=0.01)
        row = next(r for r in rows if r["mode"] == "free" and r["variant"] == "chain2"
                   and r["w"] == 4 and r["k"] == 1)
        self.assertAlmostEqual(row["b_w_bytes"], row["gbps"] * L_NS / 4, delta=0.1)

    def test_rows_file_carries_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "warp_inflight_89.txt")
            with open(path, "w", newline="\n") as f:
                f.write(SAMPLE)
            table = wi.table_of([path], wi.DEFAULT_ISA_TABLE)
        run = table["runs"][0]
        self.assertEqual(table["schema"], wi.SCHEMA)
        self.assertEqual(run["provenance"], "Measured")
        self.assertEqual(run["source"], "warp_inflight_89.txt")
        self.assertEqual(run["tool"]["source_sha256"], wi.source_sha256())
        self.assertEqual(run["tool"]["probe"], wi.checked_in_probe())
        # L0 is the run's own fetch time; the ISA table's sm_89 L0 is named beside it, and gives
        # no b_w, since that row was measured on another card (an RTX 4090).
        with open(wi.DEFAULT_ISA_TABLE) as f:
            isa = json.load(f)["arches"]["89"]
        l0 = run["l0"]
        self.assertEqual(l0["own_ns"], L_NS)
        self.assertEqual(l0["table_ns"], isa["facts"]["dram_tlbwarm_latency_ns_w1"]["value"])
        self.assertIn("dram_tlbwarm_latency_ns_w1", l0["table_source"])
        self.assertEqual(l0["table_card"], isa["card"])
        self.assertFalse(l0["table_is_this_card"])
        r = run["rows"][5]
        self.assertAlmostEqual(r["b_w_bytes"], r["gbps"] * L_NS / r["w"], delta=0.1)
        self.assertIsNone(r["b_w_table_bytes"])
        json.dumps(table, allow_nan=False)

    def test_table_b_w_only_on_the_tables_card(self):
        with open(wi.DEFAULT_ISA_TABLE) as f:
            isa = json.load(f)["arches"]["89"]
        table_l0 = isa["facts"]["dram_tlbwarm_latency_ns_w1"]["value"]
        self.assertNotEqual(table_l0, L_NS)
        head, rows = wi.parse_run(SAMPLE.replace("card NVIDIA GeForce RTX 4070 Ti",
                                                 f"card {isa['card']}"))
        run = wi.run_of(head, rows, "x.txt", wi.DEFAULT_ISA_TABLE)
        self.assertTrue(run["l0"]["table_is_this_card"])
        r = run["rows"][5]
        self.assertAlmostEqual(r["b_w_bytes"], r["gbps"] * L_NS / r["w"], delta=0.1)
        self.assertAlmostEqual(r["b_w_table_bytes"], r["gbps"] * table_l0 / r["w"], delta=0.1)
        # The summary reads the run's own L0, whichever card the table row is of.
        self.assertAlmostEqual(run["summary"]["variants"]["free"]["chain1"]["b_w_w1"], 512.0,
                               delta=1.0)

    def test_clocks(self):
        def with_clocks(note, extra=""):
            return SAMPLE.replace(CLOCKS, note + extra)
        head = ("note clocks over the measured span ({n} samples, nvidia-smi every 100 ms): "
                "memory {lo}..{hi} MHz, SM 2805..2820 MHz, P-state {p}\n")
        steady_p2 = head.format(n=30, lo=10251, hi=10251, p="P2")
        waived = "note " + wi.CLOCK_WAIVED + "\n"
        self.refused(with_clocks(head.format(n=30, lo=405, hi=10501, p="P0")), "memory clock moved")
        self.refused(with_clocks(head.format(n=30, lo=10501, hi=10501, p="P0/P8")),
                     "P-state moved")
        self.refused(with_clocks(head.format(n=2, lo=10501, hi=10501, p="P0")), "2 clock samples")
        self.refused(with_clocks(steady_p2), "--clocks-waived")
        self.refused(with_clocks(head.format(n=30, lo=9001, hi=9001, p="P0")), "under the rated")
        self.refused(with_clocks(""), "0 notes of the clocks")
        self.refused(with_clocks("note clocks over the measured span: no nvidia-smi sample of "
                                 "the probe's GPU\n"), "not read")
        # A moved clock is never waived; a steady P2 is, with the waiver noted, and only then.
        self.refused(with_clocks(head.format(n=30, lo=405, hi=10251, p="P2"), waived), "moved")
        self.refused(SAMPLE + waived, "noted as waived")
        wi.parse_run(with_clocks(steady_p2, waived))
        self.refused(SAMPLE.replace("span_ns 1762862400000000000 1762862404000000000",
                                    "span_ns 5 4"), "span_ns")

    def test_clock_note_keeps_the_span(self):
        t0 = wi.smi_time_ns("2026/11/11 12:00:00.000")
        rows = [[f"2026/11/11 12:00:0{i}.000", "00000000:01:00.0", "P8" if i in (0, 5) else "P2",
                 "2805", "405" if i in (0, 5) else "10251"] for i in range(6)]
        rows.append(["2026/11/11 12:00:02.000", "00000000:02:00.0", "P0", "2805", "9000"])
        note = wi.clock_note(rows, (t0 + 500_000_000, t0 + 4_500_000_000), wi.IP._bus("0:01:00.0"))
        self.assertEqual(note, "note clocks over the measured span (4 samples, nvidia-smi every "
                               "100 ms): memory 10251..10251 MHz, SM 2805..2805 MHz, P-state P2")
        self.assertEqual(wi.clock_refusal([note[5:]], 10501000)[1], True)

    def test_no_isa_row_gives_no_table_b_w(self):
        l0, why, card = wi.isa_l0(80, wi.DEFAULT_ISA_TABLE)
        self.assertIsNone(l0)
        self.assertIsNone(card)
        self.assertIn("sm_80", why)
        _, rows = wi.parse_run(SAMPLE)
        s = wi.derive(rows, None)
        self.assertIsNone(rows[0]["b_w_bytes"])
        self.assertEqual(s["variants"]["free"], {})
        self.assertAlmostEqual(s["p_gbps"]["free"], P_GBPS, delta=1e-3)

    def test_source_and_parser_agree(self):
        self.assertTrue(wi.PROBE_VALUE.fullmatch(wi.checked_in_probe()))
        variants = re.findall(r'\{"(\w+)", (\d), (true|false), stream<(\d), (true|false)>\}', CU)
        self.assertEqual({n: (int(f), r == "true") for n, f, r, _, _ in variants}, wi.VARIANTS)
        for _, f, r, tf, tr in variants:
            self.assertEqual((f, r), (tf, tr))
        grid = re.search(r"w_list\[\] = \{([^}]*)\}", CU).group(1)
        self.assertEqual(tuple(int(x) for x in grid.split(",")), wi.W_GRID)
        ks = re.search(r"k_list\[\] = \{([^}]*)\}", CU).group(1)
        self.assertEqual(tuple(int(x) for x in ks.split(",")), wi.K_GRID)
        cols = re.search(r'printf\("columns ([^\\]*)\\n"\)', CU).group(1)
        self.assertEqual(tuple(cols.split()), wi.COLUMNS)
        self.assertRegex(CU, r"constexpr int kWarpLoad\s+= 32 \* kLoadBytes;")
        self.assertIn("ld.global.cs.v4.u32", CU)

    def test_sass_loads(self):
        def fn(f, ring, loads):
            name = f"_ZN45_GLOBAL__N__x_6streamILi{f}ELb{int(ring)}EEEvPKhyyiPKijPjPNS_3RecES5_"
            body = "".join(f"        /*{i:04x}*/                   LDG.E.EF.128 R4, [R4.64] ;\n"
                           for i in range(loads))
            return (f"\t\tFunction : {name}\n        /*0000*/              @P0 LDG.E R4, "
                    f"[R4.64] ;\n{body}        /*0ff0*/                   EXIT ;\n")
        sass = "".join(fn(f, ring, wi.sass_loads_of(v)) for v, (f, ring) in wi.VARIANTS.items())
        self.assertEqual(wi.sass_loads(sass), {v: wi.sass_loads_of(v) for v in wi.VARIANTS})
        with self.assertRaises(ValueError):
            wi.sass_loads(fn(1, False, 1))

    def refused(self, text, why):
        with self.assertRaises(ValueError) as cm:
            wi.parse_run(text)
        self.assertIn(why, str(cm.exception))

    def test_refusals(self):
        first = row_line(*ROWS[0])
        self.refused(SAMPLE.replace("end\n", ""), "incomplete run")
        self.refused(SAMPLE.replace("selfcheck ok", "selfcheck FAIL free k1 chain1 w1: x"),
                     "self-check failed")
        self.refused(SAMPLE.replace("columns mode", "columns k mode"), "columns")
        self.refused(SAMPLE.replace(first, ""), "rows missing")
        self.refused(SAMPLE.replace(first, first + first), "given twice")
        self.refused(SAMPLE.replace(first, first.replace(" chain1 1 ", " chain1 2 ")),
                     "in flight")
        self.refused(SAMPLE.replace(first, row_line("free", 1, "chain1", 1, gbps=-1.0)),
                     "positive")
        lo_hi = first.split()
        lo_hi[6] = str(float(lo_hi[6]) * 2)
        self.refused(SAMPLE.replace(first, " ".join(lo_hi) + "\n"), "range")
        self.refused(SAMPLE.replace("arch sm_89", "arch sm_86"), "built for")
        self.refused(SAMPLE.replace(wi.source_sha256(), "0" * 64), "source_sha256")
        self.refused(SAMPLE.replace(f"probe {wi.checked_in_probe()}", "probe warp_inflight 999"),
                     "checked-in probe")
        self.refused(SAMPLE.replace("ring4:12", "ring4:8"), "16-byte loads")
        self.refused(SAMPLE.replace("after the run: 0", "after the run: 1 (pid 42)"),
                     "another compute process")
        self.refused(SAMPLE.replace("note compute processes on the probe's GPU before the run: 0\n",
                                    ""), "no note of the compute processes")
        self.refused(SAMPLE + "refused the probe exited 3\n", "refused")
        self.refused(SAMPLE.replace("selfcheck ok", "bogus 1\nselfcheck ok"), "unknown line")
        self.refused(SAMPLE.replace("selfcheck ok", row_line("free", 1, "chain1", 48).strip() +
                                    "\nselfcheck ok"), "outside the grid")
        self.refused(SAMPLE.replace("selfcheck ok", first.replace(" chain1 ", " chain3 ") +
                                    "selfcheck ok"), "unknown variant")
        # Without the registers' note the placed ring4 W = 48 row is owed.
        self.refused(SAMPLE.replace("note placed ring4 w48: 1 of 2 CTAs fit an SM (registers), "
                                    "no row\n", ""), "rows missing")

    def test_cli_rows_refuses_a_bad_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.txt")
            with open(path, "w") as f:
                f.write(SAMPLE.replace("end\n", ""))
            args = type("A", (), {"input": [path], "isa_table": wi.DEFAULT_ISA_TABLE,
                                  "out": None})()
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(wi.cmd_rows(args), 1)
            self.assertIn("bad.txt", err.getvalue())


if __name__ == "__main__":
    unittest.main()
