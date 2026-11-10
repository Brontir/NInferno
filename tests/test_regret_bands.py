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
        self.assertEqual(self.counts(bands.rows_from_csv([p1, p2], None, "boardX", pick="model")), (0, 1))

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
        rows, skipped = bands.rows_from_csv([p1], None, "b", pick="model")
        self.assertEqual([r["t"] for r in rows], [17])
        self.assertEqual([(r["t"], r["reason"]) for r in skipped],
                         [(16, "the model ranks only from T 17")])

    def test_model_pick_skips_unpriced_routes(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,40,rmma.b,10,40\n"
                        "q4k,a16,store,64,64,40,rmma.c,8,\n")
        rows, skipped = bands.rows_from_csv([p1], None, "b", pick="model")
        self.assertEqual((skipped, rows[0]["selected"], rows[0]["best"]), ([], "rmma.a", "rmma.c"))

    def test_a_route_one_pass_missed_skips_the_case(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n"
                        "q4k,a16,store,64,64,4,rmma.b,10,40\n")
        p2 = self.write("p2.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30\n")
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a\n")
        self.assertEqual(self.counts(bands.rows_from_csv([p1, p2], sel, "b")), (0, 1))

    def test_a_selection_without_sweep_rows_is_skipped(self):
        p1 = self.write("p1.csv", ROUTES + "q4k,a16,store,64,64,4,rmma.a,20,30" + chr(10))
        sel = self.write("s.csv", SELECTED + "q4k,a16,store,64,64,4,rmma.a,rmma.a" + chr(10)
                         + "q4k,a16,store,64,64,8,rmma.a,rmma.a" + chr(10))
        rows, skipped = bands.rows_from_csv([p1], sel, "b")
        self.assertEqual((len(rows), len(skipped)), (1, 1))

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

    def doc(self, name, revision, build):
        d = document([case("q4k.a16.T40", "b1", "rmma.a", {"rmma.a": 2.0, "rmma.b": 1.0})])
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

    def test_pick_model_refuses_selected_and_doc(self):
        a = self.doc("a.json", "r1", "b1")
        p1, sel = self.csv_pair()
        code, err = self.run_main(["--pick", "model", "--routes", str(p1), "--selected", str(sel),
                                   "--board", "x", "--qualified", "q"])
        self.assertEqual((code, "cannot be combined with --selected" in err), (2, True))
        code, err = self.run_main(["--pick", "model", "--doc", str(a)])
        self.assertEqual((code, "cannot be combined with --doc" in err), (2, True))

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
                      "q4k,a16,store,64,64,42,rmma.a,20,30\n", encoding="utf-8")
        sel = self.root / "s.csv"
        sel.write_text(SELECTED + "q4k,a16,store,64,64,40,rmma.a,rmma.a\n"
                       "q4k,a16,store,64,64,41,rmma.a,ggml_mmq.x64\n", encoding="utf-8")
        out = self.root / "o.json"
        code, _ = self.run_main(["--routes", str(p1), "--selected", str(sel), "--board", "x",
                                 "--qualified", "q", "--json", str(out)])
        self.assertEqual(code, 0)
        reasons = {r["t"]: r["reason"] for r in json.loads(out.read_text())["skipped_cases"]}
        self.assertEqual(reasons, {41: "the pick is not of the case's compute class",
                                   42: "no selection recorded for this case"})

    def test_no_scored_case_is_refused(self):
        p1 = self.root / "p1.csv"
        p1.write_text(ROUTES + "q4k,a16,store,64,64,41,rmma.a,20,\n", encoding="utf-8")
        code, err = self.run_main(["--routes", str(p1), "--pick", "model", "--board", "x",
                                   "--qualified", "q"])
        self.assertEqual(code, 2)
        self.assertIn("no scored cases", err)

    def test_docs_of_equal_provenance_record_it(self):
        a, b = self.doc("a.json", "r1", "b1"), self.doc("b.json", "r1", "b1")
        out = self.root / "o.json"
        code, _ = self.run_main(["--doc", str(a), "--doc", str(b), "--json", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())["provenance"],
                         {"selector_revision": "r1", "build_id": "b1"})

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
        p1.write_text(ROUTES + "q4k,a16,store,64,64,40,rmma.a,20,30\n", encoding="utf-8")
        code, err = self.run_main(["--routes", str(p1), "--pick", "model", "--board", "b"])
        self.assertEqual((code, "--qualified" in err), (2, True))
        out = self.root / "o.json"
        code, _ = self.run_main(["--routes", str(p1), "--pick", "model", "--board", "b",
                                 "--qualified", "oracle tests", "--json", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())["legacy_qualification"], "oracle tests")


if __name__ == "__main__":
    unittest.main()
