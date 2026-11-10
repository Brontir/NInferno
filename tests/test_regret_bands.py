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
        self.assertEqual(skipped, 0)
        rep = bands.report(rows)
        decode = rep["overall"]["decode"]
        self.assertEqual(decode["cases"], 2)
        # (11 + 100) / (10 + 100) - 1, not the mean of 10% and 0%.
        self.assertAlmostEqual(decode["weighted_regret"], 111 / 110 - 1)
        self.assertAlmostEqual(decode["worst_regret"], 0.1)
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
        for bad in ([1, 2], {"decode": "x"}, {"decode": float("nan")}):
            with self.assertRaises(ValueError):
                bands.report(rows, band_weights=bad)

    def test_incomplete_cases_are_counted_not_scored(self):
        bad = case("q4k.a16.store.n1.k1.T3", "b1", "rmma.a", {"rmma.a": 1, "rmma.b": 1})
        bad["candidates"].append({"id": "rmma.c", "status": "unmeasured", "reason": "timeout"})
        rows, skipped = bands.rows_from_document(document([bad]))
        self.assertEqual((rows, skipped), ([], 1))


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

    def test_legacy_csv_without_run_columns(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,10,40\n"
                        "q4k,a16,store,64,64,4,ggml_mmq.x,5,1\n")
        p2 = self.write("p2.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,22,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,12,40\n"
                        "q4k,a16,store,64,64,4,ggml_mmq.x,5,1\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        rows, skipped = bands.rows_from_csv([p1, p2], sel, "boardX")
        self.assertEqual(skipped, 0)
        row = rows[0]
        # The a8 route is not an a16 alternative; medians are over the two passes.
        self.assertEqual((row["best"], row["board"], row["t"]), ("rmma.b", "boardX", 4))
        self.assertAlmostEqual(row["selected_ms"], 0.021)
        self.assertAlmostEqual(row["best_ms"], 0.011)
        # The model ranks only from 17 tokens: a T=4 case is not scored by it.
        self.assertEqual(bands.rows_from_csv([p1, p2], None, "boardX", pick="model"), ([], 1))

    def test_model_pick_skips_unpriced_routes(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,40,rmma.b,10,40\n"
                        "q4k,a16,store,64,64,40,rmma.c,8,\n")
        rows, skipped = bands.rows_from_csv([p1], None, "b", pick="model")
        self.assertEqual((skipped, rows[0]["selected"], rows[0]["best"]), (0, "rmma.a", "rmma.c"))

    def test_a_route_one_pass_missed_skips_the_case(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,10,40\n")
        p2 = self.write("p2.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        self.assertEqual(bands.rows_from_csv([p1, p2], sel, "b"), ([], 1))

    def test_bad_timings_and_duplicate_selections_are_refused(self):
        zero = self.write("z.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,0,30\n")
        with self.assertRaises(ValueError):
            bands.rows_from_csv([zero], None, "b", pick="model")
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n")
        twice = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n"
                           "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        with self.assertRaises(ValueError):
            bands.rows_from_csv([p1], twice, "b")

    def test_untimed_selection_is_skipped(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,10,40\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.z,rmma.z\n")
        self.assertEqual(bands.rows_from_csv([p1], sel, "b"), ([], 1))
        with self.assertRaises(ValueError):
            bands.rows_from_csv([p1], None, "b")

    def test_main_writes_both_reports(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,40,rmma.b,10,40\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n")
        out_json, out_md = self.root / "r.json", self.root / "r.md"
        code = bands.main(["--routes", str(p1), "--selected", str(sel), "--board", "b",
                           "--json", str(out_json), "--md", str(out_md)])
        self.assertEqual(code, 0)
        rep = json.loads(out_json.read_text())
        self.assertAlmostEqual(rep["overall"]["wide"]["weighted_regret"], 1.0)
        self.assertIn("wide (n)", out_md.read_text())
        self.assertEqual(bands.main(["--routes", str(p1), "--board", "b"]), 2)


if __name__ == "__main__":
    unittest.main()
