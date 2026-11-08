import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import regret_collect as collect
import selection_regret as regret

HEAD = "format,compute,epilogue,n,k,T,route,median_us\n"
SELECTED_HEAD = "format,compute,epilogue,n,k,T,asked,selected\n"


class Collect(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def run_collect(self, passes, selected, qualified="generic linear and route class tests"):
        sel = [self.write(f"sel{i}.csv", SELECTED_HEAD + selected) for i in range(len(passes))]
        routes = [self.write(f"p{i}.csv", HEAD + text) for i, text in enumerate(passes, 1)]
        return collect.collect(sel, routes, "held-board", "held-model", "rev", "build", "both",
                               ["fit-board"], ["fit-model"], qualified)

    def test_regret_of_the_selected_route(self):
        rows = "q8,a16,store,4096,4096,4,rmma.a,12\nq8,a16,store,4096,4096,4,rmma.b,10\n"
        doc = self.run_collect([rows, rows], "q8,a16,store,4096,4096,4,rmma.a,rmma.a\n")
        report = regret.evaluate(json.loads(json.dumps(doc)))
        case = report["cases"][0]
        self.assertEqual(case["status"], "complete")
        self.assertEqual(case["best_observed"], "rmma.b")
        self.assertAlmostEqual(case["regret"], 0.2)
        self.assertEqual(case["passes"], 2)
        self.assertEqual(doc["qualification"], "generic linear and route class tests")

    def test_route_missing_from_a_pass_is_unmeasured(self):
        full = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        part = "q8,a16,store,64,64,1,rmma.a,5\n"
        doc = self.run_collect([full, part], "q8,a16,store,64,64,1,rmma.a,rmma.a\n")
        b = [c for c in doc["cases"][0]["candidates"] if c["id"] == "rmma.b"][0]
        self.assertEqual(b["status"], "unmeasured")
        self.assertEqual(regret.evaluate(doc)["cases"][0]["status"], "incomplete")

    def test_untimed_selection_is_listed(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        doc = self.run_collect([rows], "q8,a16,store,64,64,1,ggml_vec,ggml_vec\n")
        ids = {c["id"]: c["status"] for c in doc["cases"][0]["candidates"]}
        self.assertEqual(ids["ggml_vec"], "unmeasured")
        self.assertEqual(regret.evaluate(doc)["cases"][0]["status"], "incomplete")

    def test_without_qualification_nothing_is_ok(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        doc = self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n", qualified="")
        self.assertTrue(all(c["status"] == "unmeasured" for c in doc["cases"][0]["candidates"]))
        self.assertNotIn("qualification", doc)
        doc = self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n", qualified="  ")
        self.assertTrue(all(c["status"] == "unmeasured" for c in doc["cases"][0]["candidates"]))

    def test_other_compute_class_is_ineligible(self):
        rows = ("q4k,a16,store,64,64,12,rmma.a,5\nq4k,a16,store,64,64,12,rmma.b,6\n"
                "q4k,a16,store,64,64,12,ggml_mmq,3\n")
        doc = self.run_collect([rows], "q4k,a16,store,64,64,12,rmma.a,rmma.a\n")
        mmq = [c for c in doc["cases"][0]["candidates"] if c["id"] == "ggml_mmq"][0]
        self.assertEqual(mmq["status"], "ineligible")
        case = regret.evaluate(doc)["cases"][0]
        self.assertEqual(case["status"], "complete")
        self.assertEqual(case["best_observed"], "rmma.a")

    def test_conflicting_selections_refused(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\n"
        with self.assertRaises(ValueError):
            self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n"
                                     "q8,a16,store,64,64,1,rmma.b,rmma.b\n")

    def test_route_timed_twice_in_a_pass_refused(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.a,4\n"
        with self.assertRaises(ValueError):
            self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n")

    def test_passes_with_different_selections_refused(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        a = self.write("sa.csv", SELECTED_HEAD + "q8,a16,store,64,64,1,rmma.a,rmma.a\n")
        b = self.write("sb.csv", SELECTED_HEAD + "q8,a16,store,64,64,1,rmma.a,rmma.b\n")
        with self.assertRaises(ValueError):
            collect.collect([a, b], [self.write("p1.csv", HEAD + rows)], "b", "m", "r", "i",
                            "both", [], [], "q")
        with self.assertRaises(ValueError):  # one selection for two passes
            collect.collect([a], [self.write("p1.csv", HEAD + rows), self.write("p2.csv", HEAD + rows)],
                            "b", "m", "r", "i", "both", [], [], "q")
        doc = collect.collect([a, a], [self.write("p1.csv", HEAD + rows),
                                       self.write("p2.csv", HEAD + rows)],
                              "b", "m", "r", "i", "both", [], [], "q")
        self.assertIn("A16Only", doc["cases"][0]["numerical_contract"])
        self.assertIn("speed route profile", doc["cases"][0]["workload"])

    # #518: route_tune names each row's run, board and timing protocol.
    NEW_HEAD = "format,compute,epilogue,n,k,T,route,median_us,run,device,protocol,calls,samples\n"
    NEW_SELECTED_HEAD = "format,compute,epilogue,n,k,T,asked,selected,run,device\n"

    def new_pass(self, i, run, rows, device="ga106:28sm"):
        routes = "".join(f"q8,a16,store,64,64,{t},{r},{us},{run},{device},{p},{c},15\n"
                         for t, r, us, p, c in rows)
        sel = "".join(f"q8,a16,store,64,64,{t},rmma.a,rmma.a,{run},{device}\n"
                      for t in sorted({row[0] for row in rows}))
        return (self.write(f"nsel{i}.csv", self.NEW_SELECTED_HEAD + sel),
                self.write(f"np{i}.csv", self.NEW_HEAD + routes))

    def collect_new(self, pairs):
        return collect.collect([s for s, _ in pairs], [r for _, r in pairs], "b", "m", "r", "i",
                               "both", [], [], "q")

    def test_protocol_is_reported_per_case(self):
        rows = [(4, "rmma.a", 5, "copies", 64), (4, "rmma.b", 6, "copies", 64),
                (24, "rmma.a", 50, "flushed", 1), (24, "rmma.b", 40, "flushed", 1)]
        doc = self.collect_new([self.new_pass(1, "x-1", rows), self.new_pass(2, "x-2", rows)])
        work = {c["id"]: c["workload"] for c in doc["cases"]}
        self.assertIn("64 consecutive calls over weight copies", work["q8.a16.store.n64.k64.T4"])
        self.assertIn("one call after an L2 flush", work["q8.a16.store.n64.k64.T24"])
        self.assertEqual(doc["device"], "ga106:28sm")

    def test_passes_with_different_protocols_refused(self):
        rotated = [(4, "rmma.a", 5, "copies", 64)]
        flushed = [(4, "rmma.a", 7, "flushed", 1)]
        with self.assertRaisesRegex(ValueError, "different protocols"):
            self.collect_new([self.new_pass(1, "x-1", rotated), self.new_pass(2, "x-2", flushed)])

    def test_files_of_different_runs_refused(self):
        rows = [(4, "rmma.a", 5, "copies", 64)]
        sel, _ = self.new_pass(1, "x-1", rows)
        _, routes = self.new_pass(2, "x-2", rows)
        with self.assertRaisesRegex(ValueError, "different runs"):
            self.collect_new([(sel, routes)])

    def test_passes_of_different_boards_refused(self):
        rows = [(4, "rmma.a", 5, "copies", 64)]
        with self.assertRaisesRegex(ValueError, "more than one board"):
            self.collect_new([self.new_pass(1, "x-1", rows),
                              self.new_pass(2, "x-2", rows, device="ad106:34sm")])

    def test_old_files_report_an_unknown_protocol(self):
        rows = "q8,a16,store,64,64,4,rmma.a,5\n"
        doc = self.run_collect([rows], "q8,a16,store,64,64,4,rmma.a,rmma.a\n")
        self.assertIn("timing protocol unknown", doc["cases"][0]["workload"])
        self.assertNotIn("L2 flushed", doc["cases"][0]["workload"])

    def test_short_row_exits_2(self):
        sel = self.write("sel.csv", SELECTED_HEAD + "q8,a16\n")
        routes = self.write("p1.csv", HEAD + "q8,a16,store,64,64,1,rmma.a,5\n")
        args = ["--selected", str(sel), "--routes", str(routes), "--board", "b", "--model", "m",
                "--selector-revision", "r", "--build-id", "i", "--holdout", "both"]
        self.assertEqual(collect.main(args), 2)

    def test_missing_column_refused(self):
        sel = self.write("sel.csv", "format,compute\nq8,a16\n")
        with self.assertRaises(ValueError):
            collect.collect(sel, [self.write("p1.csv", HEAD)], "b", "m", "r", "i", "both", [], [],
                            "q")


if __name__ == "__main__":
    unittest.main()
