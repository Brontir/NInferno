"""route_table_prune.py (#614, PR-F3): the pruning plan on synthetic regret reports, and the
repository's evidence check: every measured route table entry records evidence_us with a gain above
the resolution, except the gaps in tests/route_evidence_gaps.json, which may only shrink."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import route_table_prune as prune

ALLOWLIST = Path(__file__).resolve().parent / "route_evidence_gaps.json"


def entry(fmt="q4k", n=1024, k=4096, bands=((1, 4, "vec"), (5, 16, "rmma.a")), prefill="mmq",
          measured=True):
    e = {"format": fmt, "compute": "a16", "epilogue": "store", "n": n, "k": k,
         "bands": [list(b) for b in bands]}
    if prefill:
        e["prefill"] = prefill
    if measured:
        e["tuned_on"] = "ad102:128sm"
    return e


def case(t, slowdown, fmt="q4k", n=1024, k=4096, authority="route-table", evidence="measured",
         status="complete", with_ranker=True):
    c = {"id": f"{fmt}.a16.store.n{n}.k{k}.T{t}", "board": "board", "t": t,
         "authority": authority, "evidence": evidence, "status": status}
    if with_ranker and slowdown is not None:
        c["ranker"] = "cost"
        c["ranker_vs_selected"] = slowdown
    return c


def report(*cases):
    return {"schema": "ninferno-selection-regret-report/1", "cases": list(cases)}


def table(*entries):
    return {"schema": "ninferno-route-table/1", "gpu": {"class": "x", "name": "x"},
            "entries": list(entries)}


def decided(plan):
    return {key: prune for _, key, prune, _ in plan}


class Prune(unittest.TestCase):
    KEY = ("q4k", "a16", "store", 1024, 4096)

    def entry(self, **kw):
        """A speed-only measured entry (no prefill override) unless asked."""
        kw.setdefault("prefill", None)
        return entry(**kw)

    def test_pruned_only_when_every_band_is_within_the_resolution(self):
        t = table(self.entry(bands=((1, 4, "vec"), (5, 16, "rmma.a"))))
        within = report(case(1, 0.02), case(4, 0.0), case(8, 0.049))
        self.assertTrue(decided(prune.plan_table(within, t, 0.05))[self.KEY])
        worse = report(case(1, 0.02), case(4, 0.0), case(8, 0.2))
        plan = prune.plan_table(worse, t, 0.05)
        self.assertFalse(decided(plan)[self.KEY])
        self.assertIn("T 5..16: ranker 20.0 % slower somewhere", plan[0][3])
        # the resolution is a parameter
        self.assertTrue(decided(prune.plan_table(worse, t, 0.25))[self.KEY])

    def test_a_prefill_override_is_kept_on_speed_evidence_alone(self):
        # the reproducer: bands up to 16777216, prefill_tuned_on and prefill_bands, T = 1 / 8 / 17
        e = self.entry(bands=((1, 4, "vec"), (5, 16, "rmma.a"), (17, 16777216, "mmq")))
        e["prefill"] = "mmq"
        e["prefill_bands"] = [[17, 128, "w4a8.64x64"]]
        e["prefill_tuned_on"] = "ad102:128sm"
        plan = prune.plan_table(report(case(1, 0.0), case(8, 0.0), case(17, 0.0)), table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("prefill" in why for why in plan[0][3]), plan[0][3])
        # prefill-profile cases (none are written yet) clear it
        spoken = report(case(1, 0.0), case(8, 0.0), case(17, 0.0),
                        dict(case(64, 0.01), profile="prefill"))
        self.assertTrue(prune.plan_table(spoken, table(e), 0.05)[0][2])
        slow = report(case(1, 0.0), case(8, 0.0), case(17, 0.0),
                      dict(case(64, 0.3), profile="prefill"))
        self.assertFalse(prune.plan_table(slow, table(e), 0.05)[0][2])

    def test_every_band_needs_a_case_inside_its_range(self):
        bands = ((1, 1, "a"), (2, 4, "b"), (5, 16, "c"), (17, 16777216, "d"))
        t = table(self.entry(bands=bands))
        sparse = report(case(1, 0.0), case(5, 0.0), case(17, 0.0))
        plan = prune.plan_table(sparse, t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 2..4: no case", plan[0][3])
        full = report(case(1, 0.0), case(3, 0.0), case(5, 0.0), case(17, 0.0))
        self.assertTrue(prune.plan_table(full, t, 0.05)[0][2])

    def test_a_band_spanning_the_ridge_needs_a_case_on_each_side(self):
        t = table(self.entry(bands=((1, 16, "a"), (17, 16777216, "d"))))
        cases = [case(1, 0.0), case(5, 0.0), case(17, 0.0)]
        with_ridge = dict(report(*cases), by_board_regime_authority=[{"ridge_t": 131.5}])
        plan = prune.plan_table(with_ridge, t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 132..16777216: no case", plan[0][3])
        both = dict(report(*cases, case(512, 0.0)), by_board_regime_authority=[{"ridge_t": 131.5}])
        self.assertTrue(prune.plan_table(both, t, 0.05)[0][2])
        # without a ridge the wide band is one segment
        self.assertTrue(prune.plan_table(report(*cases), t, 0.05)[0][2])
        self.assertEqual(prune.segments(17, 16777216, 131.5), [(17, 131), (132, 16777216)])
        self.assertEqual(prune.segments(1, 16, None), [(1, 4), (5, 16)])
        self.assertEqual(prune.segments(6, 9, 131.5), [(6, 9)])

    def test_a_case_without_a_finite_ranker_time_keeps_the_entry(self):
        t = table(self.entry(bands=((1, 16, "a"),)))
        for bad in (float("nan"), float("inf"), None, "0.0", True):
            c = case(1, 0.0)
            c["ranker_vs_selected"] = bad
            plan = prune.plan_table(report(c, case(8, 0.0)), t, 0.05)
            self.assertFalse(plan[0][2], bad)
            self.assertTrue(any("without a ranker time" in why for why in plan[0][3]), bad)
        incomplete = report(case(1, 0.0, status="incomplete"), case(8, 0.0))
        self.assertFalse(prune.plan_table(incomplete, t, 0.05)[0][2])

    def test_other_authorities_and_other_evidence_do_not_speak(self):
        t = table(self.entry(bands=((1, 16, "a"),)))
        r = report(case(1, 0.0, authority="cost-model", evidence="ranker"),
                   case(8, 0.0, evidence="borrowed"))
        plan = prune.plan_table(r, t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertEqual(prune.plan_table(report(case(1, 0.0, n=7)), t, 0.05)[0][3], ["no case"])
        self.assertEqual(prune.plan_table(report(), table(entry(measured=False)), 0.05), [])

    def test_dense_tune_problems_of_the_class(self):
        dense = {"schema": "ninferno-dense-tune/4", "problems": {
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20": {
                "routes": {"1": "vec", "2": None, "8": "rmma.a"}, "prefill": {}},
            "ad102:128sm|q4k|a16|store|n2048|k4096|s20": {"routes": {"1": None}, "prefill": {}},
            "ga106:28sm|q4k|a16|store|n1024|k4096|s20": {"routes": {"1": "vec"}, "prefill": {}}}}
        cases = [case(1, 0.0, authority="dense-tune"), case(8, 0.01, authority="dense-tune")]
        plan = prune.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)
        # the all-null problem and the other class's are not listed
        self.assertEqual([(p[1], p[2]) for p in plan], [(self.KEY, True)])
        # a routed width in a segment without a case keeps it
        plan = prune.plan_dense(report(cases[0]), dense, "ad102:128sm", 0.05)
        self.assertFalse(plan[0][2])
        # a prefill route is kept on speed cases alone
        dense["problems"]["ad102:128sm|q4k|a16|store|n1024|k4096|s20"]["prefill"] = {"64": "w4a8"}
        self.assertFalse(prune.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)[0][2])
        cases[1] = case(8, 0.3, authority="dense-tune")
        self.assertFalse(prune.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)[0][2])

    def test_dense_keys_that_collide_after_parsing_are_refused(self):
        dense = {"problems": {
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20.13": {"routes": {"1": "vec"}, "prefill": {}},
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20.13|mmq": {"routes": {"1": "x"},
                                                                  "prefill": {}}}}
        with self.assertRaisesRegex(ValueError, "collide"):
            prune.plan_dense(report(case(1, 0.0, authority="dense-tune")), dense, "ad102:128sm", 0.05)

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = prune.main(argv)
        return code, out.getvalue(), err.getvalue()

    def write(self, root, name, doc):
        path = root / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def test_dry_run_writes_nothing_and_apply_deletes_only_the_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = self.entry(n=2048, bands=((1, 16, "a"),))
            t = table(self.entry(bands=((1, 16, "a"),)), keep,
                      entry(fmt="q8", measured=False))
            t["gpu"]["class"] = "ad102"
            rep = report(case(1, 0.0), case(8, 0.0), case(1, 0.3, n=2048), case(8, 0.3, n=2048))
            rep["cases"] = [dict(c, board="ad102") for c in rep["cases"]]
            report_path = self.write(root, "report.json", rep)
            path = root / "table.json"
            path.write_text(prune.canonical(t), encoding="utf-8", newline="")
            before = path.read_bytes()
            code, out, _ = self.run_cli([str(report_path), "--table", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("board ad102, table class ad102", out)
            self.assertIn("1 of 2 measured entries", out)
            self.assertIn("dry run", out)
            self.assertEqual(path.read_bytes(), before)
            code, out, _ = self.run_cli([str(report_path), "--table", str(path), "--apply"])
            self.assertEqual(code, 0)
            after = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(after["entries"], [keep, entry(fmt="q8", measured=False)])
            # a file not in the canonical form is refused, not reformatted
            path.write_text(json.dumps(t, indent=2), encoding="utf-8")
            code, _, err = self.run_cli([str(report_path), "--table", str(path), "--apply"])
            self.assertEqual(code, 2)
            self.assertIn("canonical", err)

    def test_the_report_the_table_and_the_class_must_name_one_board(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t = table(self.entry(bands=((1, 16, "a"),)))
            t["gpu"]["class"] = "ad102"
            table_path = self.write(root, "t.json", t)
            for board, argv, why in (
                    ("ga106", [], "the table of class ad102"),
                    ("ad102", ["--class", "gb202:170sm"], "--class gb202:170sm differs from the "
                                                         "report's board ad102"),
                    ("ad102:128sm", ["--class", "ad104:60sm"], "differs")):
                rep = report(dict(case(1, 0.0), board=board))
                report_path = self.write(root, "r.json", rep)
                code, _, err = self.run_cli([str(report_path), "--table", str(table_path)] + argv)
                self.assertEqual(code, 2, board)
                self.assertIn(why, err)
            # the chip of a full class id agrees (ad102 with ad102:128sm), a table class too
            rep = report(dict(case(1, 0.0), board="ad102:128sm"))
            code, out, _ = self.run_cli([str(self.write(root, "r.json", rep)), "--table",
                                         str(table_path), "--class", "ad102:128sm"])
            self.assertEqual(code, 0)
            self.assertIn("board ad102:128sm, table class ad102", out)

    def test_bad_input_exits_2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "r.json").write_text(json.dumps({"schema": "other", "cases": []}), encoding="utf-8")
            self.assertEqual(self.run_cli([str(root / "r.json"), "--table", str(root / "t")])[0], 2)
            self.assertEqual(self.run_cli([str(root / "r.json")])[0], 2)
            two = report(case(1, 0.0), dict(case(1, 0.0), board="other"))
            two["cases"][0]["board"] = "board"
            (root / "two.json").write_text(json.dumps(two), encoding="utf-8")
            (root / "t.json").write_text(prune.canonical(table()), encoding="utf-8")
            code, _, err = self.run_cli([str(root / "two.json"), "--table", str(root / "t.json")])
            self.assertEqual(code, 2)
            self.assertIn("one board", err)


class EvidenceGaps(unittest.TestCase):
    def test_gap_kinds(self):
        good = entry()
        good["evidence_us"] = [[7, 10.0, 8.0]]
        weak = entry(n=2)
        weak["evidence_us"] = [[7, 10.0, 9.8]]
        decode = entry(n=3)
        decode["evidence_decode_us"] = [[1, 10.0, 5.0]]
        gaps = prune.evidence_gaps(table(good, weak, decode, entry(n=4), entry(n=5, measured=False)))
        self.assertEqual(gaps, {("q4k", "a16", "store", 2, 4096): "weak",
                                ("q4k", "a16", "store", 4, 4096): "missing"})

    def test_repository_tables_record_their_evidence_except_the_listed_gaps(self):
        allowed = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
        self.assertEqual(allowed["schema"], "ninferno-route-evidence-gaps/1")
        found = {}
        for path in sorted((ROOT / "routes").glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for key, kind in prune.evidence_gaps(doc).items():
                found[f"{path.name}:{prune.label(key)}"] = kind
        listed = allowed["gaps"]
        new = {k: v for k, v in found.items() if listed.get(k) != v}
        self.assertFalse(new, f"measured entries without evidence_us (or below the resolution) "
                              f"that are not in {ALLOWLIST.name}: {sorted(new.items())[:5]} ...")
        stale = sorted(k for k in listed if k not in found or found[k] != listed[k])
        self.assertFalse(stale, f"{ALLOWLIST.name} lists gaps that are closed (or changed): remove "
                                f"them, the list only shrinks: {stale[:5]} ...")


if __name__ == "__main__":
    unittest.main()
