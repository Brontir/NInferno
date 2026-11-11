#!/usr/bin/env python3
"""The per-SM DRAM bytes-in-flight probe (tools/bench/warp_inflight, Refs #690, #735): its parser
and derivation. A run in the checked-in probe's format round-trips into rows with b_w = rate x L0
/ W and P = the largest rate; the probe source and the parser agree on the variants, the grid
and the columns; odd, incomplete, foreign or busy runs are refused. The SM rate facts table
(src/core/data/sm_rate_facts.json) equals its regeneration from the committed runs, every one of
the checked-in probe; merge and regenerate name a refused run and leave its architecture out. No
GPU, no CUDA toolkit."""

import argparse
import contextlib
import datetime
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
# The warm-up held the probe until that memory clock settled.
SETTLE = ("note warm-up: the memory clock held {mhz} MHz for 1.0 s before the measured span "
          "(settled 2.4 s into the warm-up)\n")
SAMPLE += SETTLE.format(mhz=10501)
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
        # The reading at the top of the linear region: chain4 at W = 12 holds 90 x 256 / 12 B.
        self.assertAlmostEqual(s["variants"]["free"]["chain4"]["b_w_at_linear_w"],
                               P_GBPS * L_NS / 12, delta=0.5)
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
        self.assertIn("by construction", l0["own_source"])
        self.assertIs(run["clocks_waived"], False)
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
        p2 = with_clocks(steady_p2, waived).replace(SETTLE.format(mhz=10501),
                                                    SETTLE.format(mhz=10251))
        head, rows = wi.parse_run(p2)
        self.assertIs(wi.run_of(head, rows, "x.txt", wi.DEFAULT_ISA_TABLE)["clocks_waived"], True)
        self.refused(SAMPLE.replace("span_ns 1762862400000000000 1762862404000000000",
                                    "span_ns 5 4"), "span_ns")

    def test_settle_note_is_owed(self):
        self.refused(SAMPLE.replace(SETTLE.format(mhz=10501), ""), "settled before the measured")
        # Settled at one clock, measured at another: the span is not the settled one.
        self.refused(SAMPLE.replace(SETTLE.format(mhz=10501), SETTLE.format(mhz=10251)),
                     "settled at 10251 MHz")

    def ramp(self, t0, points, period_ms=100, bus="00000000:01:00.0"):
        """A fake `nvidia-smi --query-gpu=timestamp,pci.bus_id,pstate,clocks.sm,clocks.mem -lms
        100` stream: (until ms, memory MHz) steps from t0, one row every period_ms, in P8 at the
        idle 405 MHz (an idle clock that holds is no settled one) and P2 above it."""
        rows, t, i = [], 0, 0
        while i < len(points):
            if t >= points[i][0]:
                i += 1
                continue
            stamp = datetime.datetime.fromtimestamp((t0 + t * 1_000_000) / 1e9)
            rows.append((t0 + t * 1_000_000, [stamp.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3], bus,
                                              "P8" if points[i][1] == 405 else "P2", "2805",
                                              str(points[i][1])]))
            t += period_ms
        return rows

    def warm(self, stream, t0, begins_ms=300, timeout_ns=wi.SETTLE_TIMEOUT_NS):
        """warm_until_settled on a fake clock: the probe prints its warm-up line at begins_ms."""
        now = [t0]
        sent = []
        probe = ["note device 0 is the GPU at PCI 0000:01:00.0"]

        def lines():
            if now[0] >= t0 + begins_ms * 1_000_000 and probe[-1] != "note warm-up: begins":
                probe.append("note warm-up: begins")
            return probe

        def sleep(s):
            now[0] += int(s * 1e9)

        note, why = wi.warm_until_settled(
            lines, lambda: [r for t, r in stream if t <= now[0]], sent.append, lambda: now[0],
            sleep, lambda: not sent, timeout_ns)
        return note, why, sent, (now[0] - t0) / 1e9

    def test_warm_up_waits_for_the_ramp(self):
        t0 = wi.smi_time_ns("2026/11/11 12:00:00.000")
        # Idle 405 MHz for 1.5 s (the 0.3 s warm-up of probe 2 measured inside it), then a step
        # at 5001 for 0.6 s, then 10251 for good.
        stream = self.ramp(t0, [(1500, 405), (2100, 5001), (20000, 10251)])
        note, why, sent, at = self.warm(stream, t0)
        self.assertIsNone(why)
        self.assertEqual(sent, ["go"])
        # Not before 10251 has held for 1 s (from 2.1 s): go at about 3.1 s, never in the ramp.
        self.assertGreaterEqual(at, 3.1)
        self.assertLess(at, 3.3)
        m = wi.SETTLE_NOTE.fullmatch(note[5:])
        self.assertTrue(m)
        self.assertEqual(m.group(1), "10251")
        # The note is the one parse_run owes.
        wi.parse_run(SAMPLE.replace(SETTLE.format(mhz=10501), note.replace("10251", "10501") +
                                    "\n"))

    def test_hold_begins_in_a_performance_pstate(self):
        t0 = wi.smi_time_ns("2026/11/11 12:00:00.000")
        bus = wi.IP._bus("0:01:00.0")
        # The memory clock reads 10251 MHz in P8 for 2 s, then the same value in P2: the hold
        # counts from the P2 rows only.
        rows = [r for _, r in self.ramp(t0, [(4000, 10251)])]
        for r in rows[:20]:
            r[2] = "P8"
        self.assertIsNone(wi.settled(rows[:25], bus, t0))  # 0.4 s in P2
        s = wi.settled(rows, bus, t0)
        self.assertIsNotNone(s)
        self.assertEqual(s[0], 10251)
        self.assertAlmostEqual(s[1] / 1e9, 1.9, places=6)
        # Still in P8 at the end: not settled, whatever the hold.
        self.assertIsNone(wi.settled(rows[:20], bus, t0))

    def test_warm_up_refuses_a_clock_that_never_settles(self):
        t0 = wi.smi_time_ns("2026/11/11 12:00:00.000")
        # 405 and 10251 alternating every 0.5 s: never 1 s at one value.
        steps = [(500 * (i + 1), 405 if i % 2 else 10251) for i in range(40)]
        note, why, sent, at = self.warm(self.ramp(t0, steps), t0)
        self.assertIsNone(note)
        self.assertEqual(sent, ["stop"])
        self.assertIn("did not settle (its peak held 1 s, or a lower clock 3 s) within 10 s", why)
        self.assertIn("405..10251 MHz", why)
        self.assertLess(at, 10.6)
        # A clock that dropped from its peak has not settled either, until it held there
        # SETTLE_LOWER_HOLD_NS (a power-limited board).
        bus = wi.IP._bus("0:01:00.0")
        self.assertIsNone(wi.settled([r for _, r in self.ramp(t0, [(1000, 10251), (3000, 9501)])],
                                     bus, t0))
        lower = wi.settled([r for _, r in self.ramp(t0, [(1000, 10251), (4200, 9501)])], bus, t0)
        self.assertIsNotNone(lower)
        self.assertEqual(lower[0], 9501)

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

    # ---- The SM rate facts table (src/core/data/sm_rate_facts.json) ----

    def test_facts_table_is_its_regeneration(self):
        # The committed table is exactly what the committed runs give, and every committed run is
        # one of the checked-in probe (a run the parser refuses, a stale probe version or source
        # hash among them, would give no row: re-take it on its card and merge).
        with open(wi.DEFAULT_TABLE) as f:
            table = json.load(f)
        built, refused = wi.build_facts(wi.DEFAULT_INPUTS)
        self.assertEqual(refused, [], "committed runs the parser refuses")
        self.assertEqual(table, built)
        self.assertGreaterEqual(len(table["arches"]), 1)

    def test_facts_schema(self):
        with open(wi.DEFAULT_TABLE) as f:
            table = json.load(f)
        self.assertEqual(table["schema"], wi.FACTS_SCHEMA)
        self.assertIsInstance(table["note"], str)
        for key, row in table["arches"].items():
            self.assertEqual(row["provenance"], "Measured", key)
            self.assertEqual(key, wi.IP.arch_key(row["cc"], ""), key)
            self.assertEqual(row["source"], f"warp_inflight_{key}.txt", key)
            self.assertTrue(os.path.exists(os.path.join(wi.DEFAULT_INPUTS, row["source"])), key)
            self.assertEqual(row["tool"]["probe"], wi.checked_in_probe(), key)
            self.assertEqual(row["tool"]["source_sha256"], wi.source_sha256(), key)
            self.assertEqual(row["tool"]["arch"], f"sm_{key}", key)
            self.assertEqual(set(row["facts"]), set(wi.FACT_UNITS), key)
            for name, fact in row["facts"].items():
                self.assertEqual(fact["unit"], wi.FACT_UNITS[name], f"{key} {name}")
                self.assertGreater(fact["value"], 0, f"{key} {name}")
            # b_w_max is the largest lone-warp b_w of the row's variants.
            self.assertGreaterEqual(row["facts"]["max_warp_bytes_in_flight"]["value"],
                                    row["facts"]["ring4_b_w_w1_free"]["value"], key)
            # The transfer checks: other cards of the architecture, by their card's file name,
            # and the verdict is the bounds' over their ratios.
            for c in row["transfer"]["runs"]:
                self.assertEqual(c["source"], f"warp_inflight_{key}_{wi.card_slug(c['card'])}.txt")
                self.assertNotEqual(c["card"], row["card"], key)
                self.assertEqual(c["tool"]["source_sha256"], wi.source_sha256(), key)
            self.assertEqual(row["transfer"]["refuted"],
                             any(not c["within_bounds"] for c in row["transfer"]["runs"]), key)
            for mode in wi.MODES:
                for rings in ("ring2_rows", "ring4_rows"):
                    ws = [w for w, _, _ in row[rings][mode]]
                    self.assertEqual(ws, sorted(set(ws)), f"{key} {mode}")
                    self.assertEqual(ws[0], 1, f"{key} {mode}")
                # P is per SM clock: the rate over the clock it was read at.
                f = row["facts"]
                self.assertAlmostEqual(
                    f[f"sm_dram_bytes_per_clock_{mode}"]["value"],
                    f[f"sm_dram_gbps_{mode}"]["value"] * 1e3 / f[f"sm_mhz_{mode}"]["value"],
                    delta=0.01)
            # No host name of the machine the run was taken on: the card names it.
            self.assertNotIn("llm", json.dumps(row).lower(), key)

    def test_facts_row_of_a_run(self):
        head, rows = wi.parse_run(SAMPLE)
        row = wi.facts_row(head, rows, "warp_inflight_89.txt")
        f = row["facts"]
        self.assertEqual(f["dram_stream_latency_ns"], {"value": L_NS, "unit": "ns"})
        self.assertAlmostEqual(f["sm_dram_gbps_free"]["value"], P_GBPS, delta=1e-3)
        self.assertAlmostEqual(f["sm_dram_bytes_per_clock_placed"]["value"], P_GBPS * 1e3 / MHZ,
                               delta=0.01)
        # The made-up card follows the law: ring2 keeps its counted 2048 B at W = 1.
        self.assertAlmostEqual(f["ring2_b_w_w1_free"]["value"], 2048, delta=2)
        self.assertEqual([w for w, _, _ in row["ring2_rows"]["free"]],
                         [w for w in wi.W_GRID if w <= 32])
        # The 4-load ring keeps its 4096 B too: that is the most a warp of this card holds.
        self.assertAlmostEqual(f["max_warp_bytes_in_flight"]["value"], 4096, delta=4)
        # The law of the row against its own ring rows: the made-up card follows it exactly.
        ratios = wi.law_ratios(row, {v: row[f"{v}_rows"] for v in wi.CHECKED_RINGS})
        self.assertTrue(all(abs(r - 1) < 1e-3 for *_, r in ratios), ratios[:3])
        json.dumps(row, allow_nan=False)

    def test_transfer_checks(self):
        # Another card of the architecture under its card's name is a check of the row: one that
        # follows the law is within the bounds, one 20 % faster refutes the row.
        faster = re.sub(r"^(row \S+ \d+ \S+ \d+ \d+) (\S+) (\S+) (\S+)",
                        lambda m: m.group(1) + " " + " ".join(f"{float(x) * 1.2:.4f}"
                                                               for x in m.groups()[1:]),
                        SAMPLE, flags=re.M)
        for text, refuted in ((SAMPLE, False), (faster, True)):
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "warp_inflight_89.txt"), "w", newline="\n") as f:
                    f.write(SAMPLE)
                other = text.replace("card NVIDIA GeForce RTX 4070 Ti",
                                     "card NVIDIA GeForce RTX 4060 Ti")
                with open(os.path.join(d, "warp_inflight_89_rtx4060ti.txt"), "w",
                          newline="\n") as f:
                    f.write(other)
                table, refused = wi.build_facts(d)
                self.assertEqual(refused, [])
                t = table["arches"]["89"]["transfer"]
                self.assertEqual([c["card"] for c in t["runs"]], ["NVIDIA GeForce RTX 4060 Ti"])
                self.assertIs(t["refuted"], refuted)
                if refuted:
                    self.assertAlmostEqual(t["runs"][0]["law_over_measured"]["lowest"], 1 / 1.2,
                                           delta=1e-3)
                # Under another card's name, or the reference card's own, the run is refused.
                os.rename(os.path.join(d, "warp_inflight_89_rtx4060ti.txt"),
                          os.path.join(d, "warp_inflight_89_rtx4090.txt"))
                with self.assertRaises(ValueError):
                    wi.build_facts(d)
        self.assertEqual(wi.card_slug("NVIDIA GeForce RTX 5070 Ti"), "rtx5070ti")
        self.assertEqual(wi.card_slug("Quadro GV100"), "gv100")

    def test_regenerate_names_a_refused_run_and_leaves_its_arch_out(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "warp_inflight_89.txt"), "w", newline="\n") as f:
                f.write(SAMPLE)
            stale = SAMPLE.replace(f"probe {wi.checked_in_probe()}", "probe warp_inflight 2")
            stale = stale.replace("cc 89", "cc 75").replace("arch sm_89", "arch sm_75")
            with open(os.path.join(d, "warp_inflight_75.txt"), "w", newline="\n") as f:
                f.write(stale)
            table, refused = wi.build_facts(d)
            self.assertEqual(list(table["arches"]), ["89"])
            self.assertEqual([n for n, _ in refused], ["warp_inflight_75.txt"])
            self.assertIn("not the checked-in probe", refused[0][1])
            self.assertIn("sm_75 (NVIDIA GeForce RTX 2080 Ti)", wi.absent_arches(table))
            args = argparse.Namespace(inputs=d, table=os.path.join(d, "t.json"))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(wi.cmd_regenerate(args), 0)
            self.assertIn("warp_inflight_75.txt is no row", err.getvalue())
            self.assertRegex(out.getvalue(), r"legacy DRAM law prices them: .*sm_75 ")
            with open(args.table) as f:
                self.assertEqual(list(json.load(f)["arches"]), ["89"])

    def test_a_run_under_another_arch_name_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "warp_inflight_86.txt"), "w", newline="\n") as f:
                f.write(SAMPLE)
            with self.assertRaises(ValueError):
                wi.build_facts(d)

    def test_merge(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "run.txt")
            with open(src, "w", newline="\n") as f:
                f.write(SAMPLE)
            inputs = os.path.join(d, "in")
            os.mkdir(inputs)
            args = argparse.Namespace(input=src, inputs=inputs, table=os.path.join(d, "t.json"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(wi.cmd_merge(args), 0)
            self.assertEqual(os.listdir(inputs), ["warp_inflight_89.txt"])
            with open(args.table) as f:
                self.assertEqual(list(json.load(f)["arches"]), ["89"])
            # A refused run merges nothing.
            bad = os.path.join(d, "bad.txt")
            with open(bad, "w", newline="\n") as f:
                f.write(SAMPLE.replace("end\n", ""))
            args.input = bad
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(wi.cmd_merge(args), 1)
            args.input = src + wi.REFUSED
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(wi.cmd_merge(args), 1)
            with open(os.path.join(inputs, "warp_inflight_89.txt")) as f:
                self.assertEqual(f.read(), SAMPLE)

    def test_sampler_drains_as_it_goes(self):
        # A sampler that writes far more than a pipe holds (about 4 KB on Windows, 64 KB on Linux)
        # and then waits, as nvidia-smi -lms does: drained as it writes, every row is read and
        # stopping it does not hang (#741).
        import sys
        import time
        chatty = ("import sys, time\n"
                  "for i in range(20000):\n"
                  "    sys.stdout.write('2026/11/11 12:00:00.000, 00000000:01:00.0, %d' % i + chr(10))\n"
                  "sys.stdout.flush()\n"
                  "time.sleep(60)\n")
        s = wi.IP.Sampler(None, None, argv=[sys.executable, "-c", chatty])
        deadline = time.time() + 20
        while len(s.lines.lines) < 20000 and time.time() < deadline:
            time.sleep(0.05)
        rows = s.rows(3)
        self.assertEqual(len(rows), 20000)
        self.assertEqual(rows[-1][2], "19999")


if __name__ == "__main__":
    unittest.main()
