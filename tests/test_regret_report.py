import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import regret_report as rr
import selection_regret as regret

BOARD = "rtx-test"
META = {"card": "RTX Test", "revision": "0123456789abcdef0123456789abcdef01234567",
        "date": "2026-11-12T20:00:00Z", "nvcc": "12.8.93", "driver": "580.95.05",
        "qualification": "synthetic", "sweep": {"passes": 3, "samples": 7}}
HAS_TABLE_EXIT = hasattr(regret, "table_exit")  # selection_regret.py of #616 step 9


def ok(key, ms):
    return {"id": key, "status": "ok", "samples_ms": {"p1": ms, "p2": ms, "p3": ms}}


def case(t, authority, evidence, selected_ms, best_ms, ranker_ms=None, shape="n64.k64"):
    candidates = [ok("picked", selected_ms), ok("best", best_ms)]
    out = {"id": f"q4k.a8.store.{shape}.T{t}", "board": BOARD, "model": "synthetic",
           "workload": "synthetic", "numerical_contract": "a8", "selected": "picked",
           "candidates": candidates, "t": t, "authority": authority, "evidence": evidence}
    if ranker_ms is not None:
        candidates.append(ok("ranked", ranker_ms))
        out["ranker"] = "ranked"
    return out


def document(decode_ms=1.1, verify_ranker_ms=1.01):
    """Four cases: a vector-rule decode case 10 % off; two measured route-table verify cases, one
    whose ranker is 10 % slower (holds) and one whose ranker is `verify_ranker_ms` (caught up at
    1.01); a cost-model wide case on the best route."""
    return {"schema": "ninferno-selection-regret/1", "selector_revision": "synthetic",
            "build_id": "synthetic", "holdout": "board", "fit_boards": [], "fit_models": [],
            "device": "ad104:60sm",
            "cases": [case(1, "vec-rule", "ranker", decode_ms, 1.0),
                      case(8, "route-table", "measured", 1.0, 1.0, ranker_ms=1.1),
                      case(12, "route-table", "measured", 1.0, 1.0, ranker_ms=verify_ranker_ms),
                      case(32, "cost-model", "ranker", 2.0, 2.0)]}


def bands(decode):
    return {"overall": {"decode": {"weighted_regret": decode}, "wide": {"weighted_regret": 0.0}}}


def selection_report(doc):
    return regret.evaluate(doc)


def a5_summary(report, decode_band):
    """An #790 summary file: the report without its per-case rows, and the band tables."""
    trimmed = {k: v for k, v in report.items() if k != "cases"}
    return {"board": BOARD, "card": "RTX Test", "source_sha": "f24d0079" + "0" * 32,
            "regret_report": trimmed, "selected_bands": bands(decode_band)}


class BuildTest(unittest.TestCase):
    def test_first_run_has_no_delta_and_passes_its_schema(self):
        doc = rr.build(selection_report(document()), BOARD, META, bands(0.1))
        self.assertEqual(doc["schema"], rr.SCHEMA)
        self.assertEqual(doc["delta"], {"previous": None, "threshold": rr.THRESHOLD})
        self.assertEqual(doc["device"], "ad104:60sm")
        self.assertEqual(doc["summary"]["cases"], 4)
        self.assertEqual({(g["regime"], g["authority"]) for g in doc["by_board_regime_authority"]},
                         {("decode T<=4", "vec-rule"), ("verify T5-16", "route-table"),
                          ("T>=17 ridge unknown", "cost-model")})
        self.assertEqual(doc["bands"]["decode"]["weighted_regret"], 0.1)
        self.assertEqual(rr.check(doc), [])

    def test_regression_against_an_earlier_run(self):
        before = rr.build(selection_report(document()), BOARD, META, bands(0.1))
        after = rr.build(selection_report(document(decode_ms=1.3)), BOARD, META, bands(0.3),
                         previous=before, previous_path="runs/rtx-test/a.json")
        d = after["delta"]
        self.assertEqual(d["previous"], {"path": "runs/rtx-test/a.json",
                                         "revision": META["revision"]})
        moved = {(e["scope"], e.get("regime"), e.get("band"), e["metric"]) for e in d["regressions"]}
        self.assertIn(("regime_authority", "decode T<=4", None, "geomean_slowdown"), moved)
        self.assertIn(("regime_authority", "decode T<=4", None, "p95"), moved)
        self.assertIn(("band", None, "decode", "weighted"), moved)
        self.assertIn(("board", None, None, "p95"), moved)
        decode = next(e for e in d["regressions"]
                      if e.get("regime") == "decode T<=4" and e["metric"] == "p95")
        self.assertAlmostEqual(decode["previous"], 1.1)
        self.assertAlmostEqual(decode["current"], 1.3)
        self.assertAlmostEqual(decode["change"], 1.3 / 1.1 - 1)
        self.assertEqual(decode["cases"], [1, 1])
        self.assertEqual(d["improvements"], [])
        self.assertEqual(rr.check(after), [])
        # Back again: the same moves are improvements.
        back = rr.build(selection_report(document()), BOARD, META, bands(0.1), previous=after,
                        previous_path="b")
        self.assertEqual(back["delta"]["regressions"], [])
        self.assertEqual(len(back["delta"]["improvements"]), len(d["regressions"]))
        self.assertEqual(rr.check(back), [])

    def test_threshold_hides_a_small_move(self):
        before = rr.build(selection_report(document()), BOARD, META)
        after = rr.build(selection_report(document(decode_ms=1.12)), BOARD, META,
                         previous=before, previous_path="a")
        self.assertEqual(after["delta"]["regressions"], [])
        tight = rr.build(selection_report(document(decode_ms=1.12)), BOARD, META,
                         previous=before, previous_path="a", threshold=0.01)
        self.assertTrue(tight["delta"]["regressions"])

    def test_new_and_gone_groups(self):
        before = rr.build(selection_report(document()), BOARD, META)
        doc = document()
        doc["cases"][3]["authority"] = "pin"
        after = rr.build(selection_report(doc), BOARD, META, previous=before, previous_path="a")
        self.assertEqual(after["delta"]["new_groups"],
                         [{"regime": "T>=17 ridge unknown", "authority": "pin", "cases": 1}])
        self.assertEqual(after["delta"]["gone_groups"],
                         [{"regime": "T>=17 ridge unknown", "authority": "cost-model", "cases": 1}])

    def test_against_an_a5_summary(self):
        report = selection_report(document())
        previous = a5_summary(report, 0.1)
        after = rr.build(selection_report(document(decode_ms=1.3)), BOARD, META, bands(0.3),
                         previous=previous, previous_path="tools/bench/data/route_regret/x.json")
        d = after["delta"]
        self.assertEqual(d["previous"]["revision"], previous["source_sha"])
        self.assertTrue(any(e.get("band") == "decode" for e in d["regressions"]))
        self.assertIsNone(d["new_holds"])  # the summary keeps no per-case rows
        self.assertIsNone(d["new_caught_up"])
        self.assertEqual(rr.check(after), [])

    def test_refuses_another_board_and_other_documents(self):
        report = selection_report(document())
        with self.assertRaises(ValueError):
            rr.build(report, "other", META)
        other = rr.build(report, BOARD, META)
        other["board"] = "other"
        with self.assertRaises(ValueError):
            rr.build(report, BOARD, META, previous=other, previous_path="a")
        with self.assertRaises(ValueError):
            rr.build(report, BOARD, META, previous={"schema": "x"}, previous_path="a")

    @unittest.skipUnless(HAS_TABLE_EXIT, "selection_regret.py has no table_exit yet (#616 step 9)")
    def test_override_states_and_new_holds(self):
        before = rr.build(selection_report(document()), BOARD, META)
        self.assertEqual(before["table_exit"]["caught_up"], 1)
        self.assertEqual(before["table_exit"]["holds"], 1)
        self.assertEqual(before["overrides"]["holds"], ["q4k.a8.store.n64.k64.T8"])
        self.assertEqual(before["overrides"]["caught_up"], ["q4k.a8.store.n64.k64.T12"])
        # The T12 override's ranker falls behind: it now holds.
        after = rr.build(selection_report(document(verify_ranker_ms=1.2)), BOARD, META,
                         previous=before, previous_path="a")
        d = after["delta"]
        self.assertEqual(d["new_holds"], ["q4k.a8.store.n64.k64.T12"])
        self.assertEqual(d["new_caught_up"], [])
        self.assertEqual(d["table_exit"]["holds"], [1, 2])
        self.assertEqual(rr.check(after), [])

    @unittest.skipIf(HAS_TABLE_EXIT, "selection_regret.py has table_exit")
    def test_without_table_exit_the_fields_are_null(self):
        doc = rr.build(selection_report(document()), BOARD, META)
        self.assertIsNone(doc["table_exit"])
        self.assertIsNone(doc["overrides"])
        self.assertEqual(rr.check(doc), [])


class CheckTest(unittest.TestCase):
    def setUp(self):
        before = rr.build(selection_report(document()), BOARD, META, bands(0.1))
        self.good = rr.build(selection_report(document(decode_ms=1.3)), BOARD, META, bands(0.3),
                             previous=before, previous_path="a")
        self.assertEqual(rr.check(self.good), [])

    def broken(self, mutate):
        doc = copy.deepcopy(self.good)
        mutate(doc)
        return rr.check(doc)

    def test_each_break_is_named(self):
        def drop(key):
            return lambda d: d.pop(key)

        def flip_sign(d):
            e = d["delta"]["regressions"][0]
            e["previous"], e["current"] = e["current"], e["previous"]

        cases = {
            "schema": lambda d: d.update(schema="ninferno-regret-run/0"),
            "date": lambda d: d.update(date="2026-11-12"),
            "revision": lambda d: d.update(revision="master"),
            "card": lambda d: d.update(card=" padded"),
            "summary": lambda d: d["summary"].pop("p95_regret"),
            "by_board_regime_authority[0]": lambda d: d["by_board_regime_authority"][0].update(
                board="other"),
            "unknown regime": lambda d: d["by_board_regime_authority"][0].update(regime="T 1"),
            "overrides": drop("overrides"),
            "wrong sign": flip_sign,
            "unknown scope": lambda d: d["delta"]["regressions"][0].update(scope="global"),
            "delta.new_holds": lambda d: d["delta"].pop("new_holds"),
            "bands.decode": lambda d: d["bands"]["decode"].pop("weighted_regret"),
            "delta.threshold": lambda d: d["delta"].update(threshold=-1),
        }
        for what, mutate in cases.items():
            with self.subTest(what=what):
                problems = self.broken(mutate)
                self.assertTrue(any(what in p for p in problems), problems)

    def test_override_ids_must_match_the_counts(self):
        def add_exit(d):
            d["table_exit"] = {"cases": 4, "answered_by": {}, "overrides": 2, "caught_up": 1,
                               "holds": 1, "unscored": 0, "exit_share": 0.5, "resolution": 0.05}
            d["overrides"] = {"resolution": 0.05, "caught_up": ["a"], "holds": [], "unscored": []}
        self.assertTrue(any("overrides.holds" in p for p in self.broken(add_exit)))

    def test_no_previous_holds_nothing_else(self):
        doc = rr.build(selection_report(document()), BOARD, META)
        doc["delta"]["regressions"] = []
        self.assertTrue(rr.check(doc))


class CliTest(unittest.TestCase):
    def test_build_then_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "report.json").write_text(json.dumps(selection_report(document())))
            (tmp / "bands.json").write_text(json.dumps(bands(0.1)))
            (tmp / "shapes.txt").write_text("q4k a16 store 64 64\n")
            args = ["build", "--selection-report", str(tmp / "report.json"),
                    "--bands", str(tmp / "bands.json"), "--board", BOARD,
                    "--shapes", str(tmp / "shapes.txt"), "--passes", "3", "--samples", "7",
                    "--previous", "none", "--repo", str(tmp), "--out", str(tmp / "run.json"),
                    "--md", str(tmp / "run.md")]
            for key in ("card", "revision", "date", "nvcc", "driver"):
                args += [f"--{key}", META[key]]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rr.main(args), 0)
            run = json.loads((tmp / "run.json").read_text())
            self.assertEqual(run["sweep"]["shapes"], "shapes.txt")  # relative to the repo
            self.assertIn("RTX Test", (tmp / "run.md").read_text())
            self.assertEqual(rr.main(["check", str(tmp / "run.json")]), 0)
            run["schema"] = "x"
            (tmp / "bad.json").write_text(json.dumps(run))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(rr.main(["check", str(tmp / "bad.json")]), 2)

    def test_markdown_lists_the_regressions(self):
        before = rr.build(selection_report(document()), BOARD, META, bands(0.1))
        after = rr.build(selection_report(document(decode_ms=1.3)), BOARD, META, bands(0.3),
                         previous=before, previous_path="a")
        text = rr.markdown(after)
        self.assertIn("decode T<=4 / vec-rule", text)
        self.assertIn("regressions", text)


class NamesTest(unittest.TestCase):
    def test_board_of_matches_the_committed_names(self):
        for name, want in (("GeForce RTX 4070 Ti", ("rtx4070ti", "RTX 4070 Ti")),
                           ("NVIDIA GeForce RTX 3090", ("rtx3090", "RTX 3090")),
                           ("Quadro GV100", ("gv100", "Quadro GV100"))):
            with self.subTest(name=name):
                self.assertEqual(rr.board_of(name), want)
        data = Path(__file__).resolve().parents[1] / "tools/bench/data/route_regret"
        for path in data.glob("rtx*.json"):
            committed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(rr.board_of("GeForce " + committed["card"]),
                             (committed["board"], committed["card"]))

    def test_pick_previous(self):
        d = rr.DATA
        names = [f"{d}/rtx3090.json", f"{d}/shapes.txt", f"{d}/runs/rtx3090/20261112-aaaa.json",
                 f"{d}/runs/rtx3090/20261119-bbbb.json", f"{d}/runs/rtx30900/20261201-cccc.json"]
        self.assertEqual(rr.pick_previous(names, "rtx3090"), f"{d}/runs/rtx3090/20261119-bbbb.json")
        self.assertEqual(rr.pick_previous(names[:2], "rtx3090"), f"{d}/rtx3090.json")
        self.assertIsNone(rr.pick_previous(names, "rtx4090"))

    def test_the_committed_summaries_read_as_previous(self):
        data = Path(__file__).resolve().parents[1] / "tools/bench/data/route_regret"
        for path in sorted(data.glob("*.json")):
            committed = json.loads(path.read_text(encoding="utf-8"))
            if "board" not in committed:
                continue
            with self.subTest(board=committed["board"]):
                view = rr.previous_view(committed, committed["board"])
                self.assertTrue(view[2])  # its regime x authority groups
                self.assertIn("decode", view[3])


if __name__ == "__main__":
    unittest.main()
