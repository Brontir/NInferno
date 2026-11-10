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

GENERATION_HEADER = Path(__file__).resolve().parents[1] / "src" / "core" / "generation.h"


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

    def test_mismatched_passes_are_invalid_whatever_the_status(self):
        data = document()
        data["cases"][0]["candidates"][1]["samples_ms"] = {"x": 10}
        data["cases"][0]["candidates"].append(
            {"id": "missing", "status": "unmeasured", "reason": "benchmark crashed"})
        with self.assertRaisesRegex(ValueError, "same measurement passes"):
            regret.evaluate(data)

    def test_whitespace_identities_rejected(self):
        for field in ("board", "model", "id", "selected"):
            data = document()
            data["cases"][0][field] = data["cases"][0][field] + " "
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "whitespace"):
                regret.evaluate(data)
        data = document()
        data["fit_boards"] = ["held-board "]
        with self.assertRaisesRegex(ValueError, "whitespace"):
            regret.evaluate(data)

    def test_case_ids_unique_across_boards(self):
        data = document()
        other = copy.deepcopy(data["cases"][0])
        other["board"] = "held-board-2"
        data["cases"].append(other)
        with self.assertRaisesRegex(ValueError, "duplicate case identity"):
            regret.evaluate(data)

    def test_single_pass_cases_are_counted(self):
        data = document()
        for row in data["cases"][0]["candidates"]:
            row["samples_ms"] = {"a": row["samples_ms"]["a"]}
        self.assertEqual(regret.evaluate(data)["summary"]["single_pass_cases"], 1)
        self.assertEqual(regret.evaluate(document())["summary"]["single_pass_cases"], 0)

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
            path.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(regret.main([str(path)]), 2)

    def test_regret_by_board_regime_and_authority(self):
        def case(t, authority, evidence, selected_ms, best_ms, board="held-board"):
            c = {"id": f"q4k.a16.store.n1.k1.T{t}", "board": board, "model": "held-model",
                 "workload": "w", "numerical_contract": "c", "selected": "selected",
                 "candidates": [measured("selected", [selected_ms] * 3),
                                measured("best", [best_ms] * 3)]}
            if authority is not None:
                c["authority"], c["evidence"] = authority, evidence
            return c
        data = document()
        data["cases"] = [case(1, "r8-table", "measured", 11, 10),
                         case(4, "r8-table", "measured", 12, 10),
                         case(8, "cost-model", "ranker", 10, 10),
                         case(64, "cost-model", "ranker", 15, 10),
                         case(512, "route-table", "borrowed", 13, 10),
                         case(513, None, None, 10, 10)]  # an old row: no authority, no evidence
        data["board_facts"] = {"held-board": {"cc": 120, "sms": 170, "dram_gbps": 1792.0,
                                              "boost_mhz": 2407}}
        ridges = regret.ridges_of(data, GENERATION_HEADER)
        # s8 2048 x 170 SMs x 2407 MHz x 0.5625 B/value / (2 x 1792 GB/s) = 131.5
        self.assertAlmostEqual(ridges["held-board"], 2048 * 170 * 2407e6 * 4.5 / 8 / (2 * 1792e9),
                               places=6)
        self.assertAlmostEqual(ridges["held-board"], 131.54, places=1)
        report = regret.evaluate(data, None, ridges)
        groups = {(g["regime"], g["authority"]): g for g in report["by_board_regime_authority"]}
        self.assertEqual(set(groups), {("decode T<=4", "r8-table"),
                                       ("verify T5-16", "cost-model"),
                                       ("T>=17 below ridge", "cost-model"),
                                       ("T>=17 above ridge", "route-table"),
                                       ("T>=17 above ridge", "unknown")})
        self.assertEqual(groups[("decode T<=4", "r8-table")]["cases"], 2)
        self.assertAlmostEqual(groups[("decode T<=4", "r8-table")]["max_regret"], 0.2)
        self.assertEqual(groups[("decode T<=4", "r8-table")]["evidence"], {"measured": 2})
        self.assertEqual(groups[("T>=17 above ridge", "route-table")]["evidence"],
                         {"borrowed": 1})
        self.assertEqual(groups[("T>=17 above ridge", "unknown")]["evidence"], {"unknown": 1})
        self.assertEqual(groups[("T>=17 above ridge", "unknown")]["ridge_t"], ridges["held-board"])
        # Sorted by board, then regime order, then authority.
        order = [(g["regime"], g["authority"]) for g in report["by_board_regime_authority"]]
        self.assertEqual(order[0], ("decode T<=4", "r8-table"))
        self.assertEqual(order[-1], ("T>=17 above ridge", "unknown"))

    def test_without_authority_or_ridge_old_data_still_reports(self):
        data = document()  # no authority, no evidence, no t, no board_facts
        report = regret.evaluate(data, None, regret.ridges_of(data, GENERATION_HEADER))
        self.assertEqual(report["cases"][0]["authority"], "unknown")
        self.assertEqual(report["cases"][0]["t"], 4)  # from the id "decode-t4"
        (group,) = report["by_board_regime_authority"]
        self.assertEqual((group["regime"], group["authority"]), ("decode T<=4", "unknown"))
        self.assertIsNone(group["ridge_t"])
        # Wide calls without a ridge are said so, not placed on either side of it.
        self.assertEqual(regret.regime_of(64, None), "T>=17 ridge unknown")
        self.assertEqual(regret.regime_of(None, 100.0), "unknown width")
        self.assertEqual(regret.regime_of(16, 100.0), "verify T5-16")
        self.assertEqual(regret.regime_of(99, 100.0), "T>=17 below ridge")
        self.assertEqual(regret.regime_of(100, 100.0), "T>=17 above ridge")

    def test_bad_authority_or_width_rejected(self):
        for field, value in (("authority", " "), ("evidence", ""), ("t", 0), ("t", "4")):
            data = document()
            data["cases"][0][field] = value
            with self.assertRaises(ValueError):
                regret.evaluate(data)

    def test_malformed_board_facts_are_invalid_input_not_a_crash(self):
        good = {"cc": 120, "sms": 170, "dram_gbps": 1792.0, "boost_mhz": 2407}
        for bad in ("a string", {"cc": 120}, dict(good, sms="170"), dict(good, dram_gbps=0),
                    dict(good, cc=12.0), dict(good, boost_mhz=float("nan"))):
            data = document()
            data["board_facts"] = {"held-board": bad}
            with self.assertRaises(ValueError):
                regret.ridges_of(data, GENERATION_HEADER)
        # A value that is not an object is refused, the falsy ones too: only an absent field (or
        # an empty object) means "no facts".
        for bad in ("x", None, False, 0, []):
            data = document()
            data["board_facts"] = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                regret.ridges_of(data, GENERATION_HEADER)
        data = document()
        self.assertEqual(regret.ridges_of(data, GENERATION_HEADER), {})
        data["board_facts"] = {}
        self.assertEqual(regret.ridges_of(data, GENERATION_HEADER), {})
        with tempfile.TemporaryDirectory() as directory:  # the CLI exits 2, not a traceback
            data = document()
            data["board_facts"] = {"held-board": {"cc": 120}}
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(regret.main([str(path)]), 2)

    def test_a_board_without_tensor_cores_has_no_tensor_ridge(self):
        # A GTX 1660 Ti (sm75, 24 SMs, 1770 MHz, 288 GB/s): Turing's int8 tensor rate would give a
        # ridge of 84.96, but the board has no tensor cores.
        facts = {"cc": 75, "sms": 24, "dram_gbps": 288.0, "boost_mhz": 1770}
        data = document()
        data["cases"][0]["id"] = "wide.T64"
        data["board_facts"] = {"held-board": dict(facts)}
        self.assertAlmostEqual(regret.ridges_of(data, GENERATION_HEADER)["held-board"], 84.96,
                               places=2)
        data["board_facts"] = {"held-board": dict(facts, mma_slow=False)}
        self.assertIn("held-board", regret.ridges_of(data, GENERATION_HEADER))
        data["board_facts"] = {"held-board": dict(facts, mma_slow=True)}
        self.assertEqual(regret.ridges_of(data, GENERATION_HEADER), {})
        report = regret.evaluate(data, None, regret.ridges_of(data, GENERATION_HEADER))
        group = report["by_board_regime_authority"][0]
        self.assertEqual((group["regime"], group["ridge_t"]), ("T>=17 ridge unknown", None))
        for bad in (0, 1, "true", None):
            data["board_facts"] = {"held-board": dict(facts, mma_slow=bad)}
            with self.assertRaises(ValueError, msg=repr(bad)):
                regret.ridges_of(data, GENERATION_HEADER)
        with tempfile.TemporaryDirectory() as directory:  # and the CLI exits 2
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(regret.main([str(path)]), 2)

    def test_a_compute_capability_without_a_generation_row_has_no_ridge(self):
        data = document()
        data["board_facts"] = {"held-board": {"cc": 20, "sms": 10, "dram_gbps": 100.0,
                                              "boost_mhz": 1000}}
        self.assertEqual(regret.ridges_of(data, GENERATION_HEADER), {})
        data["cases"][0]["id"] = "wide.T64"
        report = regret.evaluate(data, None, regret.ridges_of(data, GENERATION_HEADER))
        self.assertEqual(report["by_board_regime_authority"][0]["regime"], "T>=17 ridge unknown")

    def test_one_ridge_formula(self):
        import mmq_transfer_audit as audit
        gens = audit.generations(GENERATION_HEADER)
        gb20x = audit.generation_of(gens, 120)
        self.assertAlmostEqual(audit.ridge_t(gb20x, 170, 2407, 1792.0), 131.54, places=1)
        self.assertIsNone(audit.ridge_t(gb20x, 170, None, 1792.0))
        self.assertIsNone(audit.ridge_t(audit.generation_of(gens, 70), 80, 1500, 900.0))  # no s8

    def test_t_and_the_id_suffix_must_agree(self):
        data = document()
        data["cases"][0]["id"] = "decode.T4"
        data["cases"][0]["t"] = 4
        self.assertEqual(regret.evaluate(data)["cases"][0]["t"], 4)
        data["cases"][0]["t"] = 5
        with self.assertRaisesRegex(ValueError, "ends in T4"):
            regret.evaluate(data)


if __name__ == "__main__":
    unittest.main()
