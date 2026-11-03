import contextlib
import copy
import io
import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import selection_regret as regret


def measured(key, times):
    return {"id": key, "status": "ok", "samples_ms": dict(zip(("a", "b", "c"), times))}


def document():
    return {
        "schema": "ninferno-selection-regret/1", "selector_revision": "test-revision",
        "build_id": "synthetic-only", "holdout": "both",
        "fit_boards": ["fit-board"], "fit_models": ["fit-model"],
        "cases": [{"id": "decode-t4", "board": "held-board", "model": "held-model",
                   "workload": "synthetic fixed shape, policy, layout and memory budget",
                   "numerical_contract": "qualified-a16", "selected": "selected",
                   "candidates": [measured("selected", [12, 12, 12]),
                                  measured("best", [10, 10, 10])]}],
    }


class SelectionRegretTest(unittest.TestCase):
    def test_known_slowdown_and_threshold(self):
        report = regret.evaluate(document(), 0.1)
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["cases"][0]["regret"], 0.2)
        self.assertEqual(report["cases"][0]["best_observed"], "best")
        self.assertAlmostEqual(report["summary"]["geomean_slowdown"], 1.2)
        self.assertTrue(regret.evaluate(document(), 0.21)["passed"])

    def test_threshold_equality_and_real_excess(self):
        data = document()
        data["cases"][0]["candidates"][0] = measured("selected", [11, 11, 11])
        self.assertTrue(regret.evaluate(data, 0.1)["passed"])
        data["cases"][0]["candidates"][0] = measured("selected", [11.001, 11.001, 11.001])
        self.assertFalse(regret.evaluate(data, 0.1)["passed"])

    def test_exact_tie_has_zero_regret_in_either_order(self):
        data = document()
        data["cases"][0]["candidates"][0] = measured("selected", [10, 10, 10])
        expected = regret.evaluate(data, 0)
        self.assertTrue(expected["passed"])
        self.assertEqual(expected["cases"][0]["regret"], 0)
        data["cases"][0]["candidates"].reverse()
        self.assertEqual(regret.evaluate(data, 0), expected)

    def test_nonselected_wrong_output_cannot_be_the_fastest(self):
        data = document()
        data["cases"][0]["candidates"].append(
            {"id": "wrong", "status": "oracle_failed", "reason": "independent oracle mismatch"})
        report = regret.evaluate(data)
        self.assertTrue(report["passed"])
        self.assertEqual(report["cases"][0]["best_observed"], "best")
        self.assertEqual(report["cases"][0]["excluded"]["wrong"]["status"], "oracle_failed")

    def test_invalid_selected_is_a_failure_not_an_omitted_case(self):
        for status in ("oracle_failed", "ineligible", "unmeasured"):
            with self.subTest(status=status):
                data = document()
                data["cases"][0]["candidates"][0] = {
                    "id": "selected", "status": status, "reason": "test failure"}
                report = regret.evaluate(data)
                self.assertFalse(report["passed"])
                self.assertEqual(report["summary"]["cases"], 1)
                self.assertEqual(report["summary"]["scored_cases"], 0)
                self.assertIsNone(report["summary"]["max_regret"])

    def test_missing_alternative_does_not_report_zero_regret(self):
        data = document()
        data["cases"][0]["candidates"][1] = {
            "id": "best", "status": "unmeasured", "reason": "benchmark crashed"}
        report = regret.evaluate(data)
        self.assertFalse(report["passed"])
        self.assertEqual(report["cases"][0]["status"], "incomplete")
        self.assertNotIn("regret", report["cases"][0])

    def test_only_one_qualified_candidate_is_not_evidence_of_good_selection(self):
        data = document()
        data["cases"][0]["candidates"] = [measured("selected", [1, 1, 1])]
        report = regret.evaluate(data)
        self.assertFalse(report["passed"])
        self.assertEqual(report["cases"][0]["status"], "no_comparison")
        self.assertIsNone(report["summary"]["median_regret"])

    def test_holdout_checks_every_case_and_both_axes(self):
        for axis, field in (("board", "board"), ("model", "model"), ("both", "board"),
                            ("both", "model")):
            with self.subTest(axis=axis, field=field):
                data = document()
                data["holdout"] = axis
                leaked = copy.deepcopy(data["cases"][0])
                leaked["id"] = "leaked"
                leaked[field] = "fit-" + field
                data["cases"].append(leaked)
                with self.assertRaisesRegex(ValueError, "held-out"):
                    regret.evaluate(data)

    def test_single_axis_holdout_allows_other_axis_overlap(self):
        for axis, overlap in (("board", "model"), ("model", "board")):
            data = document()
            data["holdout"] = axis
            data["cases"][0][overlap] = "fit-" + overlap
            self.assertTrue(regret.evaluate(data)["passed"])

    def test_best_is_fixed_across_passes_not_hindsight_per_pass(self):
        data = document()
        data["cases"][0]["candidates"] = [measured("selected", [10, 10, 10]),
                                           measured("other", [1, 100, 100])]
        row = regret.evaluate(data)["cases"][0]
        self.assertEqual(row["best_observed"], "selected")
        self.assertEqual(row["regret"], 0)
        self.assertEqual(row["paired_regret_range"], [0, 0])

    def test_summary_is_case_weighted_and_keeps_groups(self):
        data = document()
        other = copy.deepcopy(data["cases"][0])
        other.update(id="other", board="held-board-2", model="held-model-2")
        other["candidates"][0] = measured("selected", [20, 20, 20])
        data["cases"].append(other)
        report = regret.evaluate(data)
        self.assertAlmostEqual(report["summary"]["geomean_slowdown"], math.sqrt(1.2 * 2))
        self.assertEqual(report["summary"]["p95_regret"], 1)
        self.assertEqual(report["by_board"]["held-board-2"]["max_regret"], 1)
        self.assertEqual(len(report["by_model"]), 2)

    def test_candidate_case_and_pass_order_do_not_change_report(self):
        data = document()
        data["cases"].append(copy.deepcopy(data["cases"][0]))
        data["cases"][1]["id"] = "other"
        expected = regret.evaluate(data)
        for order in itertools.permutations(range(2)):
            changed = copy.deepcopy(data)
            changed["cases"] = [changed["cases"][i] for i in order]
            for case in changed["cases"]:
                case["candidates"].reverse()
                for row in case["candidates"]:
                    row["samples_ms"] = dict(reversed(list(row["samples_ms"].items())))
            self.assertEqual(regret.evaluate(changed), expected)

    def test_matched_passes_required(self):
        data = document()
        del data["cases"][0]["candidates"][1]["samples_ms"]["c"]
        with self.assertRaisesRegex(ValueError, "same measurement passes"):
            regret.evaluate(data)

    def test_bad_measurements_rejected(self):
        for value in (0, -1, True, "10", None, float("nan"), float("inf")):
            with self.subTest(value=value):
                data = document()
                data["cases"][0]["candidates"][0]["samples_ms"]["a"] = value
                with self.assertRaises(ValueError):
                    regret.evaluate(data)

    def test_no_measurements_for_failed_candidates(self):
        data = document()
        data["cases"][0]["candidates"][0].update(status="oracle_failed", reason="oracle")
        with self.assertRaisesRegex(ValueError, "non-ok"):
            regret.evaluate(data)

    def test_missing_identity_and_duplicates_rejected(self):
        for field in ("selector_revision", "build_id", "fit_boards", "fit_models"):
            data = document()
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                regret.evaluate(data)
        for field in ("id", "board", "model", "workload", "numerical_contract", "selected"):
            data = document()
            del data["cases"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                regret.evaluate(data)
        for kind in ("case", "candidate"):
            data = document()
            rows = data["cases"] if kind == "case" else data["cases"][0]["candidates"]
            rows.append(copy.deepcopy(rows[0]))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "duplicate"):
                regret.evaluate(data)

    def test_empty_inventory_and_absent_selection_rejected(self):
        for candidates in ([], [measured("different", [1, 1, 1])]):
            data = document()
            data["cases"][0]["candidates"] = candidates
            with self.assertRaises(ValueError):
                regret.evaluate(data)

    def test_cli_exit_codes_and_input_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(document()), encoding="utf-8")
            for threshold, expected in (("0.1", 1), ("0.3", 0), ("nan", 2), ("-1", 2)):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = regret.main([str(path), "--max-regret", threshold])
                self.assertEqual(code, expected)
                if expected != 2:
                    self.assertEqual(len(json.loads(out.getvalue())["input_sha256"]), 64)
            path.write_text('{"cases": [], "cases": []}', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(regret.main([str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
