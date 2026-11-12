#!/usr/bin/env python3
"""The launch law of ggml's matrix kernel from the microbench (tools/bench/mmq_launch_law.py,
#785): a card's law derived from its runs, a generation's row fitted on two cards with each card
predicted from the other, and a card checked against a row of core/generation.h, on synthetic
cards that follow a known law. The microbench source and the script agree on the schema and the
grids; the script's generations are core/generation.h's work-table rows. No GPU, no CUDA toolkit."""

import contextlib
import copy
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
    "mmq_launch_law", os.path.join(ROOT, "tools", "bench", "mmq_launch_law.py"))
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

with open(ml.SOURCE) as f:
    CU = f.read()

# A made-up law (kTuringMmqLaw's values) and two made-up cards of one generation that follow it.
LAW = {"unit_clocks": [5256.0, 7047.0, 9698.0, 11170.0, 13134.0, 17920.0], "stream_k_unit": 1.18,
       "launch_seconds": 3.3e-6}
CARD_A = {"name": "Card A", "cc": "8.9", "sms": 60, "reference_clock_hz": 2.61e9,
          "dram_bytes_per_s": 504e9}
CARD_B = {"name": "Card B", "cc": "8.9", "sms": 128, "reference_clock_hz": 2.52e9,
          "dram_bytes_per_s": 1008e9}
INTERCEPT = 4e-6  # the launch's fixed part of every call: not in the law, taken off by the slope


def partition(rows, k, columns, j, blocks, sms):
    """The microbench's partition_of (mmq_launch_microbench.cu)."""
    tiles = rows // 128 * -(-columns // j)
    steps = k // 256
    p = {"tiles": tiles, "steps": steps, "blocks": blocks, "split": False, "fixup": False,
         "partials": 0, "split_tiles": 0, "fixup_bytes": 0.0}
    if blocks == tiles:
        p.update(units=-(-tiles // sms) * steps)
        return p
    work = tiles * steps
    p.update(units=-(-work // blocks), split=True, fixup=tiles % blocks != 0)
    last = -1
    for b in range(1, blocks):
        end = b * work // blocks
        if end % steps == 0:
            continue
        p["partials"] += 1
        if end // steps != last:
            p["split_tiles"] += 1
            last = end // steps
    if p["fixup"]:
        p["fixup_bytes"] = (p["partials"] + 2.0 * p["split_tiles"]) * 128 * j * 4.0
    return p


def grid(name, sms):
    half = max(1, sms // 2)
    return {"law": (half, 2, 2 * half), "wave1": (sms, 1, sms), "full": (2 * sms, 2, 4 * sms),
            "streamk": (half, 1, sms), "streamk_wide": (max(sms + 1, 3 * sms // 2), 1, sms),
            "halfwave": (half, 1, half)}[name]


def card_doc(info, law=LAW, formats=("q4_K", "q6_K"), scale=None, format_scale=1.5,
             fixup_rate=None, stream_k=None, launch=None):
    """A microbench document of a card whose calls take exactly the law's time plus INTERCEPT;
    `scale` multiplies its units by J (a card off the law). q6_K's units are format_scale x. The
    fixup takes its bytes at `fixup_rate` (default the DRAM rate) plus `launch` (default the
    law's), in a call and alone; `stream_k` gives each stream-k grid its own ratio."""
    configs, alone = [], []
    rate = fixup_rate or info["dram_bytes_per_s"]
    fixed = law["launch_seconds"] if launch is None else launch
    for f in formats:
        for i, j in enumerate(ml.WIDTHS):
            unit = law["unit_clocks"][i] / info["reference_clock_hz"]
            unit *= (scale or {}).get(j, 1.0) * (format_scale if f == "q6_K" else 1.0)
            for g in ml.GRIDS:
                rt, ct, blocks = grid(g, info["sms"])
                for k in ((4096, 8192) if g == "full" else (2048, 4096, 8192)):
                    p = partition(rt * 128, k, ct * j, j, blocks, info["sms"])
                    s = (stream_k or {}).get(g, law["stream_k_unit"])
                    t = INTERCEPT + p["units"] * unit * (s if p["split"] else 1)
                    if p["fixup"]:
                        t += p["fixup_bytes"] / rate + fixed
                    us = t * 1e6
                    configs.append(dict(p, format=f, grid=g, j=j, k=k, rows=rt * 128,
                                        columns=ct * j, resident_ctas_per_sm=1, copies=3,
                                        calls=100, t_begin=[100.0, 200.0], t_end=[101.0, 201.0],
                                        weight_bytes=1.0,
                                        samples_us=[[us * 1.01, us, us * 0.99] * 3,
                                                    [us * 1.002, us, us * 0.998] * 3]))
                    if p["fixup"] and k == 8192:
                        a = (p["fixup_bytes"] / rate + fixed) * 1e6
                        alone.append({"format": f, "j": j, "k": k, "grid": g,
                                      "fixup_bytes": p["fixup_bytes"], "calls": 100,
                                      "samples_us": [[a] * 9, [a] * 9]})
    card = dict(info, chip="x", l2_bytes=48 << 20, global_bytes=12 << 30, pci_bus_id="0000:01:00.0",
                driver_sm_clock_khz=0, driver_memory_clock_khz=0, driver=13000, runtime=13020)
    return {"schema": ml.SCHEMA, "card": card, "protocol": {"samples": 9, "passes": 2},
            "configs": configs, "fixup_alone": alone,
            "launch_node": [{"kind": k, "launches": 1000, "samples_us": [1.5] * 9}
                            for k in ml.NODE_KINDS],
            # What `mmq_launch_law.py run` attaches: the idle-process notes and the clock samples.
            "notes": ["note compute processes on the probe's GPU before the run: 0"],
            "clocks": {"samples": 0, "offset_s": 0.0, "sm_mhz": None, "memory_mhz": None,
                       "pstates": []}}


def quiet(fn, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = fn(*args)
    return code, out.getvalue()


class Derive(unittest.TestCase):
    def test_recovers_the_law(self):
        d = ml.derive(card_doc(CARD_A))
        for got, want in zip(d["law"]["unit_clocks"], LAW["unit_clocks"]):
            self.assertAlmostEqual(got / want, 1.0, places=9)
        self.assertAlmostEqual(d["law"]["stream_k_unit"], LAW["stream_k_unit"], places=9)
        self.assertAlmostEqual(d["law"]["launch_seconds"] / LAW["launch_seconds"], 1.0, places=6)
        self.assertAlmostEqual(d["law"]["fixup_rate_ratio"], 1.0, places=6)
        for g in ml.STREAM_K_GRIDS:
            self.assertAlmostEqual(d["stream_k_by_grid"][g], LAW["stream_k_unit"], places=9)
        self.assertEqual(d["missing_widths"], [])
        self.assertEqual(d["resident_flags"], [])
        self.assertIsNone(d["clock_ratio"])  # no clocks recorded
        self.assertEqual(d["generation"], "ada")
        row = d["per_format"]["q4_K"][32]
        self.assertAlmostEqual(row["intercept_s"] / INTERCEPT, 1.0, places=6)
        for ratio in ("wave1_ratio", "full_ratio", "halfwave_ratio"):
            self.assertAlmostEqual(row[ratio], 1.0, places=9)
        self.assertAlmostEqual(d["launch_node_s"]["graph"], 1.5e-6)
        # The other format's units beside the row's.
        q6 = ml.derive(card_doc(CARD_A), "q6_K")
        self.assertAlmostEqual(q6["law"]["unit_clocks"][1] / LAW["unit_clocks"][1], 1.5, places=9)

    def test_median_over_passes(self):
        cfg = {"samples_us": [[1.0, 2.0, 30.0], [3.0, 4.0, 5.0]]}
        self.assertAlmostEqual(ml.time_s(cfg), 3e-6)  # medians 2 and 4
        self.assertAlmostEqual(ml.pass_spread(cfg), 2.0 / 3.0)

    def test_refuses(self):
        doc = card_doc(CARD_A)
        doc["card"]["reference_clock_hz"] = 0
        with self.assertRaises(ValueError):
            ml.derive(doc)
        with self.assertRaises(ValueError):
            ml.derive(card_doc(CARD_A), "q5_K")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json")
            bad = card_doc(CARD_A)
            bad["refused"] = "another compute process"
            with open(path, "w") as f:
                json.dump(bad, f)
            with self.assertRaises(ValueError):
                ml.load_card(path)
            raw = card_doc(CARD_A)  # the microbench's own output, never checked by `run`
            del raw["notes"], raw["clocks"]
            with open(path, "w") as f:
                json.dump(raw, f)
            with self.assertRaises(ValueError):
                ml.load_card(path)

    def test_a_missing_width_leaves_the_row_incomplete(self):
        doc = card_doc(CARD_A)
        for c in doc["configs"]:
            if c["j"] == 80:
                c["skipped"] = "no tile of this width on this device"
                del c["samples_us"]
        d = ml.derive(doc)
        self.assertEqual(d["missing_widths"], [80])
        self.assertIsNone(ml.cpp_row("kX", d["law"]))
        # Neither a fit nor a check passes with a width left unmeasured.
        self.assertFalse(ml.fit([d, ml.derive(card_doc(CARD_B))])["within_tolerance"])
        self.assertTrue(ml.check(ml.derive(card_doc(CARD_A)), LAW)["within_tolerance"])
        self.assertFalse(ml.check(d, LAW)["within_tolerance"])


    def test_fixup_from_the_kernel_alone(self):
        # The fixup reads its partials from the L2 at twice the DRAM rate: the fit over the
        # widths finds the rate and L; the in-situ difference at the DRAM rate is biased.
        d = ml.derive(card_doc(CARD_A, fixup_rate=2 * CARD_A["dram_bytes_per_s"]))
        self.assertAlmostEqual(d["law"]["launch_seconds"] / LAW["launch_seconds"], 1.0, places=6)
        self.assertAlmostEqual(d["law"]["fixup_rate_ratio"], 2.0, places=6)
        self.assertAlmostEqual(d["law"]["stream_k_unit"], LAW["stream_k_unit"], places=9)
        insitu = d["per_format"]["q4_K"][128]["insitu_fixup_beyond_bytes_s"]
        self.assertLess(insitu, LAW["launch_seconds"] - 1e-7)

    def test_a_negative_launch_gives_no_row(self):
        d = ml.derive(card_doc(CARD_A, launch=-1e-6))
        self.assertLess(d["law"]["launch_seconds"], 0)
        self.assertIsNone(ml.cpp_row("kX", d["law"]))
        self.assertIn("negative", ml.row_problem(d["law"], d["missing_widths"]))

    def test_each_stream_k_grid_apart(self):
        d = ml.derive(card_doc(CARD_A, stream_k={"streamk": 1.25, "streamk_wide": 1.05}))
        self.assertAlmostEqual(d["stream_k_by_grid"]["streamk"], 1.25, places=9)
        self.assertAlmostEqual(d["stream_k_by_grid"]["streamk_wide"], 1.05, places=9)
        self.assertAlmostEqual(d["law"]["stream_k_unit"], 1.15, places=9)

    def test_flags_resident_ctas_and_moving_clocks(self):
        doc = card_doc(CARD_A)
        for c in doc["configs"]:
            if c["j"] == 16:
                c["resident_ctas_per_sm"] = 2
            c["sm_mhz"] = [2700, 2760]
        doc["configs"][0]["sm_mhz"] = [2700, 2800]  # 3.7 %
        d = ml.derive(doc)
        self.assertTrue(d["resident_flags"])
        self.assertTrue(all(" J 16 " in f for f in d["resident_flags"]))
        self.assertEqual(len(d["clock_flags"]), 1)
        self.assertAlmostEqual(d["clock_ratio"], 2730 / 2610)
        _, text = quiet(ml.print_derived, d)
        self.assertIn("RESIDENT", text)
        self.assertIn("CLOCK", text)


class Fit(unittest.TestCase):
    def test_two_cards_of_one_law(self):
        r = ml.fit([ml.derive(card_doc(CARD_A)), ml.derive(card_doc(CARD_B))])
        self.assertTrue(r["within_tolerance"])
        self.assertEqual(r["name"], "kAdaMmqLaw")
        self.assertEqual(len(r["holdout"]), 2)
        self.assertEqual(r["holdout"][0]["from"], ["Card B"])
        for h in r["holdout"]:
            kinds = {ml.kind_of(row["what"]) for row in h["rows"]}
            self.assertEqual(kinds, {"unit", "streamk unit", "streamk_wide unit",
                                     "launch_seconds", "call law", "call full", "call streamk",
                                     "call streamk_wide"})
            # Every measured call of the row's format, predicted end to end.
            self.assertEqual(sum(row["what"].startswith("call ") for row in h["rows"]),
                             len(ml.WIDTHS) * 11)
            for row in h["rows"]:
                self.assertLess(abs(row["error"]), 1e-6)
        self.assertEqual(ml.cpp_row(r["name"], r["law"]),
                         "inline constexpr MmqLaunchLaw kAdaMmqLaw{{5256, 7047, 9698, 11170, "
                         "13134, 17920}, 1.18, 3.3e-06};")

    def test_the_holdout_flags_a_card_off_the_law(self):
        off = card_doc(CARD_B, scale={64: 1.25, 128: 1.25})
        r = ml.fit([ml.derive(card_doc(CARD_A)), ml.derive(off)])
        self.assertFalse(r["within_tolerance"])
        flagged = {(h["card"], row["what"], row["j"]) for h in r["holdout"] for row in h["rows"]
                   if row["outside"]}
        # B measured 25 % above A's prediction at J 64 and 128 (A 20 % below B's).
        self.assertIn(("Card B", "unit", 64), flagged)
        self.assertIn(("Card A", "unit", 128), flagged)
        self.assertNotIn(("Card B", "unit", 32), flagged)
        # The row is the two cards' mean.
        self.assertAlmostEqual(r["law"]["unit_clocks"][2] / LAW["unit_clocks"][2], 1.125)
        self.assertAlmostEqual(r["law"]["unit_clocks"][1] / LAW["unit_clocks"][1], 1.0)

    def test_launch_seconds_is_held_out_on_its_own(self):
        # B's fixup costs 3x A's L: diluted in the calls, it is outside on its own.
        r = ml.fit([ml.derive(card_doc(CARD_A)),
                    ml.derive(card_doc(CARD_B, launch=3 * LAW["launch_seconds"]))])
        flagged = {(h["card"], row["what"]) for h in r["holdout"] for row in h["rows"]
                   if row["outside"]}
        self.assertIn(("Card B", "launch_seconds"), flagged)
        self.assertIn(("Card A", "launch_seconds"), flagged)
        self.assertFalse(r["within_tolerance"])

    def test_names_cards_at_different_clock_shares(self):
        a, b = card_doc(CARD_A), card_doc(CARD_B)
        for c in a["configs"]:
            c["sm_mhz"] = [2800, 2810]
        for c in b["configs"]:
            c["sm_mhz"] = [2500, 2510]
        r = ml.fit([ml.derive(a), ml.derive(b)])
        self.assertIsNotNone(r["clock_note"])
        self.assertIsNone(ml.fit([ml.derive(card_doc(CARD_A)),
                                  ml.derive(card_doc(CARD_B))])["clock_note"])

    def test_refuses_one_card_and_mixed_generations(self):
        with self.assertRaises(ValueError):
            ml.fit([ml.derive(card_doc(CARD_A))])
        with self.assertRaises(ValueError):  # one card given twice is not a second card
            ml.fit([ml.derive(card_doc(CARD_A)), ml.derive(card_doc(CARD_A))])
        ga10x = dict(CARD_B, name="Card C", cc="8.6")
        with self.assertRaises(ValueError):
            ml.fit([ml.derive(card_doc(CARD_A)), ml.derive(card_doc(ga10x))])
        hopper = dict(CARD_B, cc="9.0")
        self.assertIsNone(ml.derive(card_doc(hopper))["generation"])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (os.path.join(tmp, n) for n in ("a.json", "b.json", "c.json"))
            for path, doc in ((a, card_doc(CARD_A)), (b, card_doc(CARD_B)),
                              (c, card_doc(CARD_B, scale={16: 0.8}))):
                with open(path, "w") as f:
                    json.dump(doc, f)
            out = os.path.join(tmp, "law.json")
            code, text = quiet(ml.main, ["fit", "--input", a, "--input", b, "--out", out])
            self.assertEqual(code, 0)
            self.assertIn("kAdaMmqLaw{{5256, 7047", text)
            with open(out) as f:
                saved = json.load(f)
            self.assertEqual(saved["schema"], ml.LAW_SCHEMA)
            self.assertTrue(saved["cpp"].startswith("inline constexpr MmqLaunchLaw kAdaMmqLaw"))
            code, text = quiet(ml.main, ["fit", "--input", a, "--input", c])
            self.assertEqual(code, 4)
            self.assertIn("OUTSIDE", text)
            code, _ = quiet(ml.main, ["derive", "--input", a])
            self.assertEqual(code, 0)


class Check(unittest.TestCase):
    def test_generation_rows(self):
        laws = ml.generation_laws()
        self.assertIn("kTuringMmqLaw", laws)
        self.assertEqual(laws["kTuringMmqLaw"], LAW)

    def test_a_card_against_kturingmmqlaw(self):
        turing = dict(CARD_A, cc="7.5", sms=68, reference_clock_hz=1.545e9,
                      dram_bytes_per_s=616e9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.json")
            with open(path, "w") as f:
                json.dump(card_doc(turing), f)
            code, text = quiet(ml.main, ["check", "--input", path])
            self.assertEqual(code, 0)
            self.assertIn("against kTuringMmqLaw", text)
            with open(path, "w") as f:
                json.dump(card_doc(turing, scale={32: 1.2}), f)
            code, text = quiet(ml.main, ["check", "--input", path])
            self.assertEqual(code, 4)

    def test_generations_are_generation_h_rows(self):
        with open(ml.GENERATION_H) as f:
            text = f.read()
        rows = {m.group(4): ((int(m.group(1)), int(m.group(2))), (int(m.group(1)),
                                                                  int(m.group(3))))
                for m in re.finditer(r'\{(\d+), (\d+), (\d+), "(\w+)"', text)}
        for name, (lo, hi, _) in ml.GENERATIONS.items():
            self.assertEqual(rows[name], (lo, hi), name)
        # Every row with the work table is one of them (mmq_work_tiles: the 13th field).
        work = {m.group(1) for m in re.finditer(
            r'\{\d+, \d+, \d+, "(\w+)", [^,]+, [^,]+, [^,]+, \w+, [^,]+, [^,]+, \w+, \w+, true',
            text)}
        self.assertEqual(work, set(ml.GENERATIONS))


class Source(unittest.TestCase):
    def test_schema_and_grids(self):
        self.assertIn(f'kSchema = "{ml.SCHEMA}"', CU)
        m = re.search(r"kGrids\[\] = \{([^}]*)\}", CU)
        self.assertEqual(tuple(re.findall(r'"(\w+)"', m.group(1))), ml.GRIDS)
        m = re.search(r"std::vector<int> widths\{([^}]*)\}", CU)
        self.assertEqual(tuple(int(v) for v in m.group(1).split(",")), ml.WIDTHS)


class Clocks(unittest.TestCase):
    def test_each_config_gets_its_window(self):
        doc = card_doc(CARD_A, formats=("q4_K",))
        doc["configs"] = doc["configs"][:2]
        base = 1.78e9  # a real epoch: a naive datetime near 1970 fails on some platforms
        doc["configs"][0].update(t_begin=[base + 10.0], t_end=[base + 11.0])
        doc["configs"][1].update(t_begin=[base + 20.0], t_end=[base + 21.0])
        doc["fixup_alone"] = []

        def stamp(t):
            return datetime.datetime.fromtimestamp(t).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        samples = [[stamp(base + 10.5), "00000000:01:00.0", "P2", "2700", "10501"],
                   [stamp(base + 10.7), "00000000:01:00.0", "P2", "2650", "10501"],
                   [stamp(base + 20.5), "00000000:01:00.0", "P2", "2500", "10501"],
                   [stamp(base + 20.6), "00000000:02:00.0", "P2", "1000", "10501"]]
        idle = [stamp(base + 5.0), "00000000:01:00.0", "P8", "210", "405"]
        ml.attach_clocks(doc, samples + [idle], ml._bus("0000:01:00.0"))
        self.assertEqual(doc["configs"][0]["sm_mhz"], [2650, 2700])
        self.assertEqual(doc["configs"][1]["sm_mhz"], [2500, 2500])
        self.assertEqual(doc["clocks"]["samples"], 3)  # the idle sample before the run left out
        self.assertEqual(doc["clocks"]["pstates"], ["P2"])
        # nvidia-smi in another time zone (WSL): its stamps two hours ahead, the offset read once.
        self.assertEqual(ml.smi_offset(base + 0.4, stamp(base + 7200)), -7200.0)
        doc2 = card_doc(CARD_A, formats=("q4_K",))
        doc2["configs"] = copy.deepcopy(doc["configs"])
        doc2["fixup_alone"] = []
        shifted = [[stamp(base + 7200 + t), *row[1:]] for t, row in
                   zip((10.5, 10.7, 20.5, 20.6), samples)]
        ml.attach_clocks(doc2, shifted, ml._bus("0000:01:00.0"), -7200.0)
        self.assertEqual(doc2["configs"][0]["sm_mhz"], [2650, 2700])


if __name__ == "__main__":
    unittest.main()
