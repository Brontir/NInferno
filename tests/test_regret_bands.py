import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import regret_bands as bands

ROUTES = "format,compute,epilogue,n,k,T,route,median_us,model_us\n"
SELECTED = "format,compute,epilogue,n,k,T,asked,selected\n"


def case(case_id, board, selected, times, workload="route_tune synthetic"):
    return {"id": case_id, "board": board, "model": "m", "workload": workload,
            "numerical_contract": "c", "selected": selected,
            "candidates": [{"id": r, "status": "ok", "samples_ms": {"p1": t, "p2": t}}
                           for r, t in times.items()]}


def document(cases):
    return {"schema": "ninferno-selection-regret/1", "selector_revision": "r", "build_id": "b",
            "holdout": "board", "fit_boards": [], "fit_models": [], "cases": cases}


class Bands(unittest.TestCase):
    def test_edges(self):
        got = {t: bands.band_of(t) for t in (1, 8, 9, 16, 17, 32, 33, 128, 129, 4096)}
        self.assertEqual(got, {1: "decode", 8: "decode", 9: "verify", 16: "verify", 17: "narrow",
                               32: "narrow", 33: "wide", 128: "wide", 129: "batch129",
                               4096: "batch129"})
        with self.assertRaises(ValueError):
            bands.band_of(0)

    def test_family_and_t(self):
        self.assertEqual(bands.family_of("ggml_mmq.x64"), "ggml_mmq")
        self.assertEqual(bands.family_of("tiled"), "tiled")
        self.assertEqual(bands.tokens_of({"id": "q4k.a16.store.n8.k8.T24"}), 24)
        self.assertEqual(bands.tokens_of({"id": "x", "workload": "route_tune n=8 T=130, policy"}), 130)
        with self.assertRaises(ValueError):
            bands.tokens_of({"id": "x", "workload": "none"})


class Report(unittest.TestCase):
    def rows(self):
        doc = document([
            # Two decode cases of one band: weighted by time, the long case dominates.
            case("q4k.a16.store.n1.k1.T1", "b1", "rmma.a", {"rmma.a": 11, "rmma.b": 10}),
            case("q4k.a16.store.n1.k1.T2", "b1", "rmma.a", {"rmma.a": 100, "rmma.b": 100}),
            case("q6k.a16.store.n1.k1.T64", "b1", "tiled", {"tiled": 30, "rmma.b": 20}),
            case("q6k.a16.store.n1.k1.T64b2", "b2", "tiled", {"tiled": 20, "rmma.b": 20},
                 workload="route_tune T=64"),
        ])
        return bands.rows_from_document(doc)

    def test_time_weighted_not_per_case(self):
        rows, skipped = self.rows()
        self.assertEqual(skipped, [])
        rep = bands.report(rows)
        decode = rep["overall"]["decode"]
        self.assertEqual(decode["cases"], 2)
        # (11 + 100) / (10 + 100) - 1, not the mean of 10% and 0%.
        self.assertAlmostEqual(decode["weighted_regret"], 111 / 110 - 1)
        self.assertAlmostEqual(decode["worst_regret"], 0.1)
        # sqrt(1.1 * 1.0) - 1, not the mean of the two regrets.
        self.assertAlmostEqual(decode["geomean_regret"], 1.1 ** 0.5 - 1)
        self.assertEqual((rep["cases"], rep["cases_skipped"]), (4, 0))
        self.assertEqual(decode["cases_over_5pct"], 1)
        self.assertNotIn("batch129", rep["overall"])

    def test_groups(self):
        rows, _ = self.rows()
        rep = bands.report(rows)
        self.assertEqual(sorted(rep["by_board"]), ["b1", "b2"])
        self.assertAlmostEqual(rep["by_board"]["b1"]["wide"]["weighted_regret"], 0.5)
        self.assertAlmostEqual(rep["by_board"]["b2"]["wide"]["weighted_regret"], 0.0)
        self.assertEqual(sorted(rep["by_format"]), ["q4k", "q6k"])
        self.assertEqual(sorted(rep["by_family"]), ["rmma", "tiled"])
        self.assertAlmostEqual(rep["by_family"]["tiled"]["wide"]["weighted_regret"], 50 / 40 - 1)
        self.assertIn("| b1 |", bands.markdown(rep))

    def test_band_weights(self):
        rows, _ = self.rows()
        rep = bands.report(rows, band_weights={"decode": 1, "wide": 3})
        b1 = rep["by_board"]["b1"]
        expect = (b1["decode"]["weighted_regret"] * 1 + b1["wide"]["weighted_regret"] * 3) / 4
        self.assertAlmostEqual(b1["weighted"], expect)
        with self.assertRaises(ValueError):
            bands.report(rows, band_weights={"nope": 1})
        with self.assertRaises(ValueError):
            bands.report(rows, band_weights={"decode": 0})

    def test_no_mean_without_weights_or_coverage(self):
        rows, _ = self.rows()
        self.assertNotIn("weighted", bands.report(rows)["by_board"]["b1"])
        # A weighted band the group has no case in: no mean, not a mean of the rest.
        rep = bands.report(rows, band_weights={"decode": 1, "batch129": 1})
        self.assertIsNone(rep["by_board"]["b1"]["weighted"])
        for bad in ([1, 2], {"decode": "x"}, {"decode": float("nan")}, {}, {"decode": True}):
            with self.assertRaises(ValueError):
                bands.report(rows, band_weights=bad)

    def test_incomplete_cases_are_counted_not_scored(self):
        bad = case("q4k.a16.store.n1.k1.T3", "b1", "rmma.a", {"rmma.a": 1, "rmma.b": 1})
        bad["candidates"].append({"id": "rmma.c", "status": "unmeasured", "reason": "timeout"})
        rows, skipped = bands.rows_from_document(document([bad]))
        self.assertEqual(rows, [])
        self.assertEqual([(r['id'], r['t'], r['board'], r['format'], r['reason']) for r in skipped],
                         [('q4k.a16.store.n1.k1.T3', 3, 'b1', 'q4k', 'document case incomplete')])


class Csv(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def counts(result):
        rows, skipped = result
        return len(rows), len(skipped)

    def selection_of(self, paths, name="model_sel.csv"):
        """A selection CSV listing every key of the passes, the key's first route as executed."""
        keys = {}
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines()[1:]:
                fields = line.split(",")
                keys.setdefault(",".join(fields[:6]), fields[6])
        return self.write(name, SELECTED + "".join(f"{k},{r},{r}\n" for k, r in keys.items()))

    def model(self, paths, excluded=None):
        """The model pick of the passes, every key of which is a speed-loop width."""
        return bands.rows_from_csv(paths, self.selection_of(paths), "b", pick="model",
                                   excluded=excluded)

    def test_legacy_csv_without_run_columns(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,10,40\n"
                        "q4k,a16,store,64,64,4,ggml_mmq.x,5,1\n")
        p2 = self.write("p2.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,22,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,12,40\n"
                        "q4k,a16,store,64,64,4,ggml_mmq.x,5,1\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        rows, skipped = bands.rows_from_csv([p1, p2], sel, "boardX")
        self.assertEqual(skipped, [])
        row = rows[0]
        # The a8 route is not an a16 alternative; medians are over the two passes.
        self.assertEqual((row["best"], row["board"], row["t"]), ("rmma.b", "boardX", 4))
        self.assertAlmostEqual(row["selected_ms"], 0.021)
        self.assertAlmostEqual(row["best_ms"], 0.011)
        # The model ranks only from 17 tokens: a T=4 case is not scored by it.
        self.assertEqual(self.counts(bands.rows_from_csv([p1, p2], sel, "boardX", pick="model")),
                         (0, 1))
        # Either pick needs the selection: its widths are the speed profile's.
        with self.assertRaises(ValueError):
            bands.rows_from_csv([p1, p2], None, "boardX", pick="model")

    def test_median_over_three_passes_not_mean(self):
        # a: 20, 22, 100 -> median 22 (mean 47.3); b: 10, 12, 14 -> median 12.
        passes = [self.write(f"p{i}.csv", ROUTES + f"q4k,a16,store,64,64,4,rmma.a,{a},30\n"
                             f"q4k,a16,store,64,64,4,rmma.b,{b},40\n")
                  for i, (a, b) in enumerate(((20, 10), (22, 12), (100, 14)))]
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        rows, skipped = bands.rows_from_csv(passes, sel, "b")
        self.assertEqual(skipped, [])
        self.assertAlmostEqual(rows[0]["selected_ms"], 0.022)
        self.assertAlmostEqual(rows[0]["best_ms"], 0.012)

    def test_model_pick_edge_at_t16_and_t17(self):
        self.assertEqual(bands.MODEL_MIN_T, 17)
        lines = "".join(f"q4k,a16,store,64,64,{t},{r},{us},{m}\n"
                        for t in (16, 17) for r, us, m in (("rmma.a", 20, 30), ("rmma.b", 10, 40)))
        p1 = self.write("p1.csv", ROUTES + lines)
        rows, skipped = self.model([p1])
        self.assertEqual([r["t"] for r in rows], [17])
        self.assertEqual([(r["t"], r["reason"]) for r in skipped],
                         [(16, "the model ranks only from T 17")])

    def test_model_pick_skips_unpriced_routes(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,40,rmma.b,10,40\n"
                        "q4k,a16,store,64,64,40,rmma.c,8,\n")
        rows, skipped = self.model([p1])
        self.assertEqual((skipped, rows[0]["selected"], rows[0]["best"]), ([], "rmma.a", "rmma.c"))

    def test_a_route_one_pass_missed_skips_the_case(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,10,40\n")
        p2 = self.write("p2.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        self.assertEqual(self.counts(bands.rows_from_csv([p1, p2], sel, "b")), (0, 1))

    def test_a_selection_without_sweep_rows_is_skipped(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30" + chr(10)
                        + "q4k,a16,store,64,64,4,rmma.b,10,40" + chr(10))
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a" + chr(10)
                         + "q4k,a16,store,64,64,8,rmma.a,rmma.a" + chr(10))
        rows, skipped = bands.rows_from_csv([p1], sel, "b")
        self.assertEqual((len(rows), len(skipped)), (1, 1))

    def test_bad_timings_and_duplicate_selections_are_refused(self):
        zero = self.write("z.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,0,30\n")
        with self.assertRaises(ValueError):
            self.model([zero])
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n")
        twice = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n"
                           "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        with self.assertRaises(ValueError):
            bands.rows_from_csv([p1], twice, "b")

    def test_one_class_candidate_is_no_comparison_not_zero_regret(self):
        # An a16 case: the pick plus only an a8 route leaves one candidate of its class.
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,40,ggml_mmq.x64,10,40\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n")
        for pick in ("selected", "model"):
            rows, skipped = bands.rows_from_csv([p1], sel, "b", pick=pick)
            self.assertEqual(rows, [])
            self.assertEqual([r["reason"] for r in skipped],
                             ["fewer than two candidates of the case's compute class"])

    def tied_pick(self, tied_rows, other_rows, fmt="q4k,a16", long=True,
                  longest="1.2345678901234567"):
        """The model pick of the T=40 key whose tied rows are written as given, beside other keys.

        With `long`, a row of another problem carries the price `longest` (by default with the
        17 significant digits route_tune writes now).
        """
        def keyed(t, rows):
            return "".join(f"{fmt},store,64,64,{t},{route},{us},{price}\n"
                           for route, us, price in rows)
        long_price = f"q4k,a16,store,65,64,50,rmma.z,10,{longest}\n"
        p1 = self.write("p1.csv", ROUTES + keyed(40, tied_rows) + "".join(
            keyed(t, rows) for t, rows in other_rows) + (long_price if long else ""))
        rows, skipped = self.model([p1])
        at40 = [r for r in rows if r["t"] == 40]
        return (at40[0]["selected"], None) if at40 else (
            None, [r["reason"] for r in skipped if r["t"] == 40])

    def test_a_same_family_tie_takes_the_route_the_problem_offers_first(self):
        # Production keeps the first route offered on an equal price. The offer order is the one
        # the other speed-loop widths of the problem write (schedule index), whatever the order of
        # the tied key's own rows.
        a, b, c = ("rmma.a", 20, 30), ("rmma.b", 10, 30), ("rmma.c", 15, 40)
        other = [(48, [a, b, c]), (64, [a, b, c])]
        for tied, first in (([a, b, c], "rmma.a"), ([b, a, c], "rmma.a"), ([c, b, a], "rmma.a")):
            self.assertEqual(self.tied_pick(tied, other), (first, None))
        other = [(48, [b, a, c]), (64, [b, a, c])]
        for tied in ([a, b, c], [b, a, c], [c, b, a]):
            self.assertEqual(self.tied_pick(tied, other), ("rmma.b", None))

    def test_a_tie_the_other_widths_do_not_settle_is_a_skip(self):
        a, b, c = ("rmma.a", 20, 30), ("rmma.b", 10, 30), ("rmma.c", 15, 40)
        out_of_order = ["the pass wrote the tied rows out of offer order"]
        # No other key of the problem.
        self.assertEqual(self.tied_pick([b, a, c], []), (None, out_of_order))
        # The other keys do not hold both tied routes.
        self.assertEqual(self.tied_pick([b, a, c], [(48, [a, c]), (64, [b, c])]),
                         (None, out_of_order))
        # The other keys disagree, one each.
        self.assertEqual(self.tied_pick([b, a, c], [(48, [a, b, c]), (64, [b, a, c])]),
                         (None, out_of_order))
        # Another problem (n or k) is no vote.
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.b,10,30\n"
                        "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,128,64,48,rmma.a,20,30\n"
                        "q4k,a16,store,128,64,48,rmma.b,10,30\n"
                        "q4k,a16,store,64,128,48,rmma.a,20,30\n"
                        "q4k,a16,store,64,128,48,rmma.b,10,30\n"
                        "q4k,a16,store,65,64,50,rmma.z,10,1.2345678901234567\n")
        rows, skipped = self.model([p1])
        self.assertEqual([r["reason"] for r in skipped if r["t"] == 40], out_of_order)

    def test_a_tie_in_a_six_digit_model_us_is_a_skip(self):
        # Before route_tune printed 17 digits, prices that differ in the seventh digit read as a
        # tie, and production takes the cheaper: the tool cannot tell the two, so it skips.
        a, b, c = ("rmma.a", 20, 30), ("rmma.b", 10, 30), ("rmma.c", 15, 40)
        other = [(48, [a, b, c])]
        self.assertEqual(self.tied_pick([b, a, c], other, long=False),
                         (None, ["model_us has six significant digits: an equal price may be a "
                                 "rounding"]))
        self.assertEqual(self.tied_pick([b, a, c], other), ("rmma.a", None))
        self.assertEqual([bands.significant_digits(t) for t in ("30", "0.012300", "1.5e+06",
                                                                 "12.345678", "-4.2")],
                         [1, 3, 2, 8, 2])

    def test_six_significant_digits_skip_and_seven_resolve(self):
        # The edge itself (LEGACY_MODEL_DIGITS = 6): a file whose longest model_us has exactly six
        # significant digits is a legacy file, seven is not. A `<` in place of `<=` fails here.
        a, b, c = ("rmma.a", 20, 30), ("rmma.b", 10, 30), ("rmma.c", 15, 40)
        other = [(48, [a, b, c])]
        legacy = (None, ["model_us has six significant digits: an equal price may be a rounding"])
        for longest, expect in (("1.23456", legacy), ("123456", legacy), ("0.00123456", legacy),
                                ("1.234567", ("rmma.a", None)), ("1234567", ("rmma.a", None))):
            self.assertEqual(self.tied_pick([b, a, c], other, longest=longest), expect, longest)

    def test_a_tie_reads_its_offer_order_from_speed_widths_only(self):
        # The prefill loop writes the prefill route first (rmma.b here) at its widths, which the
        # selection does not list: they are no vote, so the speed width 48 settles the tie.
        a, b = ("rmma.a", 20, 30), ("rmma.b", 10, 30)

        def keyed(t, rows):
            return "".join(f"q4k,a16,store,64,64,{t},{r},{us},{m}\n" for r, us, m in rows)
        p1 = self.write("p1.csv", ROUTES + keyed(40, [a, b]) + keyed(48, [a, b])
                        + keyed(2048, [b, a]) + keyed(4096, [b, a])
                        + "q4k,a16,store,65,64,50,rmma.z,10,1.2345678901234567\n")
        sel = self.write("s.csv", SELECTED + "".join(
            f"q4k,a16,store,{n},64,{t},rmma.a,rmma.a\n" for n, t in ((64, 40), (64, 48), (65, 50))))
        rows, _ = bands.rows_from_csv([p1], sel, "b", pick="model")
        self.assertEqual({r["t"]: r["selected"] for r in rows if r["id"].startswith("q4k.a16.store.n64")},
                         {40: "rmma.a", 48: "rmma.a"})

    def test_a_case_whose_i8_schedules_are_unswept_is_not_scored_by_the_model(self):
        # cost_route prices the I8 tiles for an int8 q4 / q5 / q6 case with K a multiple of 512;
        # route_tune sweeps none of them, so the model's pick may be a route the file lacks.
        def key_rows(fmt, compute, k):
            return "".join(f"{fmt},{compute},store,64,{k},40,{r},{us},{m}\n"
                           for r, us, m in (("w4a8.a", 20, 30), ("w4a8.b", 10, 40)))
        p1 = self.write("p1.csv", ROUTES + key_rows("q4", "a8", 512) + key_rows("q6", "a8", 1024)
                        + key_rows("q5", "a8", 576) + key_rows("q4k", "a8", 512)
                        + key_rows("q4", "a16", 512).replace("w4a8", "rmma"))
        rows, skipped = self.model([p1])
        reason = "the model also prices the I8 schedules, which route_tune does not sweep"
        self.assertEqual(sorted((r["id"], r["reason"]) for r in skipped),
                         [("q4.a8.store.n64.k512.T40", reason),
                          ("q6.a8.store.n64.k1024.T40", reason)])
        self.assertEqual(sorted(r["id"] for r in rows),
                         ["q4.a16.store.n64.k512.T40", "q4k.a8.store.n64.k512.T40",
                          "q5.a8.store.n64.k576.T40"])
        # The recorded selection is still scored.
        rows, skipped = bands.rows_from_csv([p1], self.selection_of([p1]), "b")
        self.assertEqual((len(rows), skipped), (5, []))

    def test_nvfp4_is_not_scored_by_the_model(self):
        # route_authorities.cpp cost_model declines NVFP4; route_tune writes its model_us anyway.
        text = "".join(f"{fmt},a16,store,64,64,40,{r},{us},{m}\n" for fmt in ("nvfp4", "q4k")
                       for r, us, m in (("rmma.a", 20, 30), ("rmma.b", 10, 40)))
        p1 = self.write("p1.csv", ROUTES + text)
        rows, skipped = self.model([p1])
        self.assertEqual([r["format"] for r in rows], ["q4k"])
        self.assertEqual([(r["format"], r["reason"]) for r in skipped],
                         [("nvfp4", "the model does not rank NVFP4 routes")])
        # The recorded selection of an NVFP4 case is still scored.
        rows, skipped = bands.rows_from_csv([p1], self.selection_of([p1]), "b")
        self.assertEqual((len(rows), skipped), (2, []))

    def test_prefill_profile_rows_are_neither_cases_nor_candidates(self):
        # route_tune's prefill loop writes after its speed loop, into the same CSV with no profile
        # column: a width the selection does not list (1024) and, at a listed width, a route the
        # speed loop does not time there (R8 above 4 tokens) are prefill-profile rows.
        p1 = self.write("p1.csv", ROUTES + "q4k,a8,store,64,64,40,w4a8.a,20,30\n"
                        "q4k,a8,store,64,64,40,w4a8.b,10,40\n"
                        "q4k,a8,store,64,64,40,r8.w4r1,5,1\n"
                        "q4k,a8,store,64,64,1024,w4a8.a,200,300\n"
                        "q4k,a8,store,64,64,1024,w4a8.b,100,400\n")
        sel = self.write("s.csv", SELECTED + "q4k,a8,store,64,64,40,w4a8.a,w4a8.a\n")
        for pick in ("selected", "model"):
            excluded = []
            rows, skipped = bands.rows_from_csv([p1], sel, "b", pick=pick, excluded=excluded)
            self.assertEqual(skipped, [], pick)
            self.assertEqual([(r["t"], r["selected"], r["best"]) for r in rows],
                             [(40, "w4a8.a", "w4a8.b")], pick)
            self.assertEqual(sorted(excluded), [("q4k.a8.store.n64.k64.T1024", "w4a8.a"),
                                                ("q4k.a8.store.n64.k64.T1024", "w4a8.b"),
                                                ("q4k.a8.store.n64.k64.T40", "r8.w4r1")], pick)
        out = self.root / "o.json"
        self.assertEqual(bands.main(["--routes", str(p1), "--selected", str(sel), "--board", "b",
                                     "--qualified", "q", "--json", str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())["prefill_profile_rows_excluded"], 3)
        # The executed selection is a speed-loop row at any width (R8 runs verify widths).
        p2 = self.write("p2.csv", ROUTES + "q4k,a8,store,64,64,8,r8.w4r1,5,\n"
                        "q4k,a8,store,64,64,8,w4a8.a,20,\n")
        sel2 = self.write("s2.csv", SELECTED + "q4k,a8,store,64,64,8,r8.w4r1,r8.w4r1\n")
        rows, skipped = bands.rows_from_csv([p2], sel2, "b")
        self.assertEqual([(r["selected"], r["best"]) for r in rows], [("r8.w4r1", "r8.w4r1")])

    def test_speed_loop_widths_of_each_family(self):
        # route_tune_tool.cpp: R8 to kR8MaxTokens (4), w4a4 / staged / nvfp4t / tiled from 17.
        got = {(route, t): bands.speed_loop_times(route, t, None)
               for route in ("r8.w4r1", "tiled.64x64", "staged.64x32.w4x1.s2", "nvfp4t.c1.64x64",
                             "w4a4.64x64", "rmma.m1.n1.w1x8", "ggml_mmq")
               for t in (4, 5, 16, 17)}
        self.assertEqual({k for k, v in got.items() if not v},
                         {("r8.w4r1", 5), ("r8.w4r1", 16), ("r8.w4r1", 17)}
                         | {(r, t) for r in ("tiled.64x64", "staged.64x32.w4x1.s2",
                                             "nvfp4t.c1.64x64", "w4a4.64x64") for t in (4, 5, 16)})
        self.assertTrue(bands.speed_loop_times("tiled.64x64", 16, "tiled.64x64"))

    def test_w4a8ws_is_one_family_with_w4a8(self):
        # route_name spells the producer-warp W4A8 schedule w4a8ws; it is offered after the
        # w4a8 schedules, so a tie of the two is settled by the order, not skipped as two families.
        a, ws, c = ("w4a8.64x64", 20, 30), ("w4a8ws.128x64.p4", 10, 30), ("ggml_mmq.x", 15, 40)
        other = [(48, [a, ws, c])]
        self.assertEqual(self.tied_pick([ws, a, c], other, fmt="q4k,a8"), ("w4a8.64x64", None))
        self.assertEqual(bands.offer_family("w4a8ws.128x64.p4"), "w4a8")
        self.assertEqual(bands.offer_family("w4a8.64x64"), "w4a8")
        self.assertEqual(bands.offer_family("rmma.m1.n8.w4x1"), "rmma")

    def test_a_tie_across_families_is_a_skip_whatever_the_row_order(self):
        # tiled and staged are written and offered in other orders: the pick is unknown.
        a = "q4k,a16,store,64,64,40,tiled.a,20,30\n"
        b = "q4k,a16,store,64,64,40,staged.b,10,30\n"
        c = "q4k,a16,store,64,64,40,rmma.c,15,40\n"
        for text in (a + b + c, b + a + c, c + b + a):
            p1 = self.write("p1.csv", ROUTES + text)
            rows, skipped = self.model([p1])
            self.assertEqual((rows, [r["reason"] for r in skipped]),
                             ([], ["the model prices the best candidates of two families equally"]))
        # A tie above the lowest price does not matter.
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,40,rmma.b,10,40\n"
                        "q4k,a16,store,64,64,40,rmma.c,15,40\n")
        rows, skipped = self.model([p1])
        self.assertEqual((skipped, rows[0]["selected"]), ([], "rmma.a"))

    def test_untimed_selection_is_skipped(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,10,40\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.z,rmma.z\n")
        self.assertEqual(self.counts(bands.rows_from_csv([p1], sel, "b")), (0, 1))
        with self.assertRaises(ValueError):
            bands.rows_from_csv([p1], None, "b")

    def test_main_writes_both_reports(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,40,rmma.b,10,40\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n")
        out_json, out_md = self.root / "r.json", self.root / "r.md"
        code = bands.main(["--routes", str(p1), "--selected", str(sel), "--board", "b", "--qualified", "oracle tests",
                           "--json", str(out_json), "--md", str(out_md)])
        self.assertEqual(code, 0)
        rep = json.loads(out_json.read_text())
        self.assertAlmostEqual(rep["overall"]["wide"]["weighted_regret"], 1.0)
        self.assertIn("wide (n / skipped)", out_md.read_text())
        self.assertEqual(bands.main(["--routes", str(p1), "--board", "b", "--qualified", "x"]), 2)


class Skipped(unittest.TestCase):
    """A band must say what it did not score; skipped cases never read as zero regret."""

    def rows_and_skipped(self):
        # Ten narrow cases: two picks are the best route (0 % regret), eight picks were never timed.
        rows, skipped = [], []
        for i in range(10):
            key = f"q4k.a16.store.n{i + 1}.k1.T20"
            if i < 2:
                rows.append({"id": key, "board": "b1", "format": "q4k", "t": 20,
                             "selected": "rmma.a", "best": "rmma.a", "selected_ms": 1.0,
                             "best_ms": 1.0})
            else:
                skipped.append(bands.skip_row(key, "b1", "q4k", 20, "tiled.z",
                                              "the pick was not timed among the candidates"))
        return rows, skipped

    def test_reviewer_reproduction_is_visible(self):
        rows, skipped = self.rows_and_skipped()
        rep = bands.report(rows, skipped, band_weights={"narrow": 1})
        narrow = rep["overall"]["narrow"]
        self.assertEqual((narrow["cases"], narrow["skipped"]), (2, 8))
        self.assertEqual(narrow["weighted_regret"], 0.0)
        self.assertEqual(rep["cases_skipped"], 8)
        # Every group carries the same count: board, format and the family of the unswept pick.
        self.assertEqual(rep["by_board"]["b1"]["narrow"]["skipped"], 8)
        self.assertEqual(rep["by_format"]["q4k"]["narrow"]["skipped"], 8)
        self.assertEqual(rep["by_family"]["rmma"]["narrow"]["skipped"], 0)
        self.assertEqual(rep["by_family"]["tiled"]["narrow"], bands.stats([], 8))
        self.assertIn("+0.0% (2 / 8)", bands.markdown(rep))
        self.assertIn("- (0 / 8)", bands.markdown(rep))

    def test_skip_rows_keep_t_board_format_and_reason(self):
        rows, skipped = self.rows_and_skipped()
        listed = bands.report(rows, skipped)["skipped_cases"]
        self.assertEqual(len(listed), 8)
        self.assertEqual(listed[0], {"id": "q4k.a16.store.n3.k1.T20", "board": "b1",
                                     "format": "q4k", "t": 20, "selected": "tiled.z",
                                     "family": "tiled",
                                     "reason": "the pick was not timed among the candidates"})

    def test_weighted_is_none_when_a_weighted_band_has_skips(self):
        rows, skipped = self.rows_and_skipped()
        rep = bands.report(rows, skipped, band_weights={"narrow": 1})
        self.assertIsNone(rep["overall"]["weighted"])
        self.assertEqual(rep["overall"]["weighted_incomplete"], ["narrow"])
        self.assertIsNone(rep["by_board"]["b1"]["weighted"])
        # The same cases without the skips do get a mean; a skip in an unweighted band changes nothing.
        clean = bands.report(rows, [], band_weights={"narrow": 1})
        self.assertEqual(clean["overall"]["weighted"], 0.0)
        self.assertNotIn("weighted_incomplete", clean["overall"])
        other = [bands.skip_row("q4k.a16.store.n1.k1.T3", "b1", "q4k", 3, "tiled.z", "x")]
        self.assertEqual(bands.report(rows, other, band_weights={"narrow": 1})["overall"]["weighted"],
                         0.0)

    def test_a_band_of_only_skipped_cases_is_present_and_unscored(self):
        rows, _ = self.rows_and_skipped()
        only = [bands.skip_row("q4k.a16.store.n1.k1.T3", "b1", "q4k", 3, "tiled.z", "x")]
        rep = bands.report(rows, only, band_weights={"decode": 1})
        decode = rep["overall"]["decode"]
        self.assertEqual((decode["cases"], decode["skipped"], decode["weighted_regret"]),
                         (0, 1, None))
        self.assertIsNone(rep["overall"]["weighted"])
        self.assertIn("- (0 / 1)", bands.markdown(rep))

    def test_no_pick_family_for_a_model_skip(self):
        rows, _ = self.rows_and_skipped()
        none = [bands.skip_row("q4k.a16.store.n1.k1.T3", "b1", "q4k", 3, None, "no price")]
        self.assertEqual(bands.report(rows, none)["by_family"]["(no pick)"]["decode"]["skipped"], 1)

    def test_document_skips_are_rows(self):
        bad = case("q4k.a16.store.n1.k1.T3", "b9", "rmma.a", {"rmma.a": 1, "rmma.b": 1})
        bad["candidates"].append({"id": "rmma.c", "status": "unmeasured", "reason": "timeout"})
        ok = case("q4k.a16.store.n1.k1.T40", "b9", "rmma.a", {"rmma.a": 2, "rmma.b": 1})
        rows, skipped = bands.rows_from_document(document([bad, ok]))
        rep = bands.report(rows, skipped)
        self.assertEqual((rep["cases"], rep["cases_skipped"]), (1, 1))
        self.assertEqual(rep["by_board"]["b9"]["decode"]["skipped"], 1)
        self.assertEqual(rep["by_board"]["b9"]["wide"]["skipped"], 0)


class Cli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def doc(self, name, revision, build, board="b1"):
        d = document([case("q4k.a16.T40", board, "rmma.a", {"rmma.a": 2.0, "rmma.b": 1.0})])
        d["selector_revision"], d["build_id"] = revision, build
        path = self.root / name
        path.write_text(json.dumps(d), encoding="utf-8")
        return path

    def run_main(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            return bands.main(argv), err.getvalue()

    def test_docs_of_different_provenance_are_refused(self):
        a, b = self.doc("a.json", "r1", "b1"), self.doc("b.json", "r2", "b1")
        code, err = self.run_main(["--doc", str(a), "--doc", str(b)])
        self.assertEqual(code, 2)
        self.assertIn("selector_revision", err)
        self.assertNotIn("build_id", err.split("report them")[0].split("differ in")[1].split("(")[0])

    def csv_pair(self, selected_row="rmma.a,rmma.a"):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n", encoding="utf-8")
        sel = self.root / "s.csv"
        sel.write_text(SELECTED + "q4k,a16,store,64,64,40," + selected_row + "\n", encoding="utf-8")
        return p1, sel

    def test_doc_and_routes_cannot_be_mixed(self):
        a = self.doc("a.json", "r1", "b1")
        p1, sel = self.csv_pair()
        out = self.root / "o.json"
        code, err = self.run_main(["--doc", str(a), "--routes", str(p1), "--selected", str(sel),
                                   "--board", "x", "--qualified", "oracle tests", "--json", str(out)])
        self.assertEqual(code, 2)
        self.assertIn("--doc and --routes cannot be merged", err)
        self.assertFalse(out.exists())

    def test_pick_model_refuses_doc_and_needs_selected(self):
        a = self.doc("a.json", "r1", "b1")
        p1, sel = self.csv_pair()
        code, err = self.run_main(["--pick", "model", "--doc", str(a)])
        self.assertEqual((code, "cannot be combined with --doc" in err), (2, True))
        code, err = self.run_main(["--pick", "model", "--routes", str(p1), "--board", "x",
                                   "--qualified", "q"])
        self.assertEqual((code, "one --selected per --routes group" in err), (2, True))

    def selection(self, keys):
        sel = self.root / "s.csv"
        sel.write_text(SELECTED + "".join(f"q4k,a16,store,64,64,{t},rmma.a,rmma.a\n"
                                          for t in keys), encoding="utf-8")
        return sel

    def test_a_case_two_inputs_hold_is_refused(self):
        # Merged twice, a case would count and weigh twice in its band.
        a = self.doc("a.json", "r1", "b1")
        code, err = self.run_main(["--doc", str(a), "--doc", str(a)])
        self.assertEqual(code, 2)
        self.assertIn("both hold case q4k.a16.T40 of board b1, model m", err)
        # The same id on another board, or of another model, is another case.
        for field, value in (("board", "b2"), ("model", "m2")):
            other = json.loads(a.read_text())
            other["cases"][0][field] = value
            b = self.root / f"{field}.json"
            b.write_text(json.dumps(other), encoding="utf-8")
            self.assertEqual(self.run_main(["--doc", str(a), "--doc", str(b)])[0], 0, field)
        # Two --routes groups of one board holding one case, scored or skipped.
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,40,rmma.b,10,40\n", encoding="utf-8")
        sel = self.root / "s.csv"
        sel.write_text(SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n", encoding="utf-8")
        group = ["--routes", str(p1), "--selected", str(sel), "--qualified", "q"]
        code, err = self.run_main(group + group[:4] + ["--board", "x", "--board", "x"])
        self.assertEqual(code, 2)
        self.assertIn("both hold case q4k.a16.store.n64.k64.T40 of board x:", err)
        self.assertEqual(self.run_main(group + group[:4] + ["--board", "x", "--board", "y"])[0], 0)

    def test_too_many_selected_names_the_selected(self):
        p1, sel = self.csv_pair()
        code, err = self.run_main(["--routes", str(p1), "--selected", str(sel), "--selected",
                                   str(sel), "--board", "x", "--qualified", "q"])
        self.assertEqual(code, 2)
        self.assertIn("at most one --selected per --routes group (2 selected, 1 groups)", err)

    def test_skipped_cases_reach_the_json_report(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,40,rmma.b,10,40\n"
                      "q4k,a16,store,64,64,41,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,41,rmma.b,10,40\n", encoding="utf-8")
        sel = self.root / "s.csv"
        sel.write_text(SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n"
                       "q4k,a16,store,64,64,41,rmma.a,rmma.z\n", encoding="utf-8")
        out = self.root / "o.json"
        code, _ = self.run_main(["--routes", str(p1), "--selected", str(sel), "--board", "x",
                                 "--qualified", "q", "--json", str(out),
                                 "--band-weights", '{"wide": 1}'])
        self.assertEqual(code, 0)
        rep = json.loads(out.read_text())
        self.assertEqual((rep["cases"], rep["cases_skipped"]), (1, 1))
        self.assertEqual(rep["overall"]["wide"]["skipped"], 1)
        self.assertIsNone(rep["overall"]["weighted"])
        self.assertEqual(rep["skipped_cases"][0]["id"], "q4k.a16.store.n64.k64.T41")
        self.assertEqual(rep["skipped_cases"][0]["selected"], "rmma.z")

    def test_document_skips_reach_the_json_report(self):
        bad = case("q4k.a16.store.n1.k1.T3", "b9", "rmma.a", {"rmma.a": 1, "rmma.b": 1})
        bad["candidates"].append({"id": "rmma.c", "status": "unmeasured", "reason": "timeout"})
        ok = case("q4k.a16.store.n1.k1.T4", "b9", "rmma.a", {"rmma.a": 2, "rmma.b": 1})
        path = self.root / "d.json"
        path.write_text(json.dumps(document([bad, ok])), encoding="utf-8")
        out = self.root / "o.json"
        code, _ = self.run_main(["--doc", str(path), "--json", str(out),
                                 "--band-weights", '{"decode": 1}'])
        self.assertEqual(code, 0)
        rep = json.loads(out.read_text())
        self.assertEqual((rep["cases"], rep["cases_skipped"]), (1, 1))
        self.assertEqual(rep["overall"]["decode"]["skipped"], 1)
        self.assertIsNone(rep["overall"]["weighted"])

    def test_model_pricing_nothing_is_a_skip_row(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,40,rmma.b,10,40\n"
                      "q4k,a16,store,64,64,41,rmma.a,20,\n"
                      "q4k,a16,store,64,64,41,rmma.b,10,\n", encoding="utf-8")
        out = self.root / "o.json"
        code, _ = self.run_main(["--routes", str(p1), "--pick", "model", "--board", "x",
                                 "--selected", str(self.selection((40, 41))),
                                 "--qualified", "q", "--json", str(out)])
        self.assertEqual(code, 0)
        rep = json.loads(out.read_text())
        self.assertEqual(rep["cases_skipped"], 1)
        self.assertEqual(rep["skipped_cases"][0]["reason"], "the model prices none of the candidates")

    def test_skip_reasons_name_a_missing_selection_and_another_compute_class(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,40,rmma.b,10,40\n"
                      "q4k,a16,store,64,64,41,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,41,ggml_mmq.x64,10,30\n"
                      "q4k,a16,store,65,64,41,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,42,rmma.a,20,30\n", encoding="utf-8")
        sel = self.root / "s.csv"
        sel.write_text(SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n"
                       "q4k,a16,store,64,64,41,rmma.a,ggml_mmq.x64\n", encoding="utf-8")
        out = self.root / "o.json"
        code, _ = self.run_main(["--routes", str(p1), "--selected", str(sel), "--board", "x",
                                 "--qualified", "q", "--json", str(out)])
        self.assertEqual(code, 0)
        rep = json.loads(out.read_text())
        reasons = {r["id"]: r["reason"] for r in rep["skipped_cases"]}
        # n65 at a width the selection lists for another problem: its selection is missing. T=42
        # is a width the selection never lists, a prefill-profile row: not a case.
        self.assertEqual(reasons, {
            "q4k.a16.store.n64.k64.T41": "the pick is not of the case's compute class",
            "q4k.a16.store.n65.k64.T41": "no selection recorded for this case"})
        self.assertEqual(rep["prefill_profile_rows_excluded"], 1)

    def test_no_scored_case_is_refused(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,41,rmma.a,20,\n", encoding="utf-8")
        out, md = self.root / "o.json", self.root / "o.md"
        code, err = self.run_main(["--routes", str(p1), "--pick", "model", "--board", "x",
                                   "--selected", str(self.selection((41,))), "--qualified", "q",
                                   "--json", str(out), "--md", str(md)])
        self.assertEqual(code, 2)
        self.assertIn("no scored cases (1 skipped; the report lists them)", err)
        # The report is still written: it is where the skipped cases and their reasons are.
        rep = json.loads(out.read_text())
        self.assertEqual((rep["cases"], rep["cases_skipped"]), (0, 1))
        self.assertEqual(rep["skipped_cases"][0]["reason"], "the model prices none of the candidates")
        self.assertIn("- (0 / 1)", md.read_text())

    def test_every_document_case_skipped_still_writes_the_report(self):
        bad = case("q4k.a16.store.n1.k1.T3", "b9", "rmma.a", {"rmma.a": 1, "rmma.b": 1})
        bad["candidates"].append({"id": "rmma.c", "status": "unmeasured", "reason": "timeout"})
        path = self.root / "d.json"
        path.write_text(json.dumps(document([bad])), encoding="utf-8")
        out = self.root / "o.json"
        code, err = self.run_main(["--doc", str(path), "--json", str(out)])
        self.assertEqual(code, 2)
        self.assertIn("no scored cases", err)
        self.assertEqual(json.loads(out.read_text())["skipped_cases"][0]["id"],
                         "q4k.a16.store.n1.k1.T3")

    def test_an_unwritable_report_is_an_error_not_a_traceback(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,40,rmma.b,10,40\n", encoding="utf-8")
        argv = ["--routes", str(p1), "--selected", str(self.selection((40,))), "--board", "x",
                "--qualified", "q"]
        missing = self.root / "no such directory"
        for flag in ("--json", "--md"):
            code, err = self.run_main(argv + [flag, str(missing / "r")])
            self.assertEqual(code, 2, flag)
            self.assertIn("regret_bands: ", err)
            self.assertIn("no such directory", err)

    def test_docs_of_equal_provenance_record_it(self):
        a, b = self.doc("a.json", "r1", "b1"), self.doc("b.json", "r1", "b1", board="b2")
        out = self.root / "o.json"
        code, _ = self.run_main(["--doc", str(a), "--doc", str(b), "--json", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())["provenance"],
                         {"selector_revision": "r1", "build_id": "b1", "holdout": "board",
                          "fit_boards": [], "fit_models": []})

    def test_docs_of_another_evaluation_scope_are_refused(self):
        # One selector_revision is one frozen selector with one fit inventory: documents that
        # hold out another identity or name other fit boards or models are not one evaluation.
        def scoped(name, board, **fields):
            path = self.doc(name, "r1", "b1", board=board)
            d = json.loads(path.read_text())
            d.update(fields)
            path.write_text(json.dumps(d), encoding="utf-8")
            return path

        a = scoped("a.json", "b1", fit_boards=["f1", "f2"], fit_models=["m0"])
        for field, value in (("holdout", "both"), ("fit_boards", ["f1"]),
                             ("fit_models", ["m0", "m9"])):
            b = scoped("b.json", "b2", **{"fit_boards": ["f1", "f2"], "fit_models": ["m0"],
                                          field: value})
            code, err = self.run_main(["--doc", str(a), "--doc", str(b)])
            self.assertEqual(code, 2, field)
            self.assertEqual(err.split("differ in ")[1].split(" (")[0], field)
        # The same sets in another order are the same scope.
        b = scoped("b.json", "b2", fit_boards=["f2", "f1"], fit_models=["m0"])
        out = self.root / "o.json"
        self.assertEqual(self.run_main(["--doc", str(a), "--doc", str(b), "--json", str(out)])[0], 0)
        self.assertEqual(json.loads(out.read_text())["provenance"]["fit_boards"], ["f1", "f2"])

    def test_selected_pick_needs_one_selection_per_group(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n", encoding="utf-8")
        sel = self.root / "s.csv"
        sel.write_text(SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n", encoding="utf-8")
        argv = ["--routes", str(p1), "--routes", str(p1), "--board", "x", "--board", "y",
                "--selected", str(sel), "--qualified", "oracle tests"]
        code, err = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("one --selected per --routes group", err)

    def test_legacy_routes_need_a_qualification_statement(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                      "q4k,a16,store,64,64,40,rmma.b,10,40\n", encoding="utf-8")
        sel = str(self.selection((40,)))
        code, err = self.run_main(["--routes", str(p1), "--pick", "model", "--board", "b",
                                   "--selected", sel])
        self.assertEqual((code, "--qualified" in err), (2, True))
        out = self.root / "o.json"
        code, _ = self.run_main(["--routes", str(p1), "--pick", "model", "--board", "b",
                                 "--selected", sel, "--qualified", "oracle tests",
                                 "--json", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())["legacy_qualification"], "oracle tests")


if __name__ == "__main__":
    unittest.main()
