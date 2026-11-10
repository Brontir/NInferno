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


RIDGE = [{"ridge_t": 131.5}]


def with_ridge(r):
    return dict(r, by_board_regime_authority=RIDGE)


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
        self.assertTrue(decided(prune.plan_table(worse, t, 0.25))[self.KEY])

    def test_a_prefill_override_is_kept_on_speed_evidence_alone(self):
        # the reproducer: bands up to 16777216, prefill_tuned_on and prefill_bands, T = 1 / 8 / 17
        e = self.entry(bands=((1, 4, "vec"), (5, 16, "rmma.a"), (17, 16777216, "mmq")))
        e["prefill"] = "mmq"
        e["prefill_bands"] = [[17, 128, "w4a8.64x64"]]
        e["prefill_tuned_on"] = "ad102:128sm"
        plan = prune.plan_table(with_ridge(report(case(1, 0.0), case(8, 0.0), case(17, 0.0))),
                                table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("prefill" in why for why in plan[0][3]), plan[0][3])
        speed = [case(1, 0.0), case(8, 0.0), case(17, 0.0), case(512, 0.0)]
        # prefill-profile cases (none are written yet) clear it, in every prefill range: the band
        # 17..128 and the route past it (17.. cut at the ridge)
        pre = [dict(case(t, 0.01), profile="prefill") for t in (64, 17, 200, 512)]
        self.assertTrue(prune.plan_table(with_ridge(report(*speed, *pre)), table(e), 0.05)[0][2])
        lone = [dict(case(64, 0.01), profile="prefill")]
        plan = prune.plan_table(with_ridge(report(*speed, *lone)), table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any(why.startswith("prefill T ") for why in plan[0][3]), plan[0][3])
        slow = [dict(case(t, 0.3 if t == 512 else 0.0), profile="prefill") for t in (64, 17, 200, 512)]
        self.assertFalse(prune.plan_table(with_ridge(report(*speed, *slow)), table(e), 0.05)[0][2])

    def test_every_band_needs_a_case_inside_its_range(self):
        bands = ((1, 1, "a"), (2, 4, "b"), (5, 16, "c"), (17, 16777216, "d"))
        t = table(self.entry(bands=bands))
        sparse = with_ridge(report(case(1, 0.0), case(5, 0.0), case(17, 0.0), case(512, 0.0)))
        plan = prune.plan_table(sparse, t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 2..4: no case", plan[0][3])
        full = with_ridge(report(case(1, 0.0), case(3, 0.0), case(5, 0.0), case(17, 0.0),
                                 case(512, 0.0)))
        self.assertTrue(prune.plan_table(full, t, 0.05)[0][2])

    def test_a_wide_band_needs_the_ridge_and_a_case_on_each_side(self):
        t = table(self.entry(bands=((1, 16, "a"), (17, 16777216, "d"))))
        cases = [case(1, 0.0), case(5, 0.0), case(17, 0.0)]
        plan = prune.plan_table(with_ridge(report(*cases)), t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 132..16777216: no case", plan[0][3])
        self.assertTrue(prune.plan_table(with_ridge(report(*cases, case(512, 0.0))), t, 0.05)[0][2])
        # without a ridge in the report a wide band cannot be split: kept
        plan = prune.plan_table(report(*cases, case(512, 0.0)), t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("no ridge" in why for why in plan[0][3]), plan[0][3])
        # a decode-only entry needs none
        self.assertTrue(prune.plan_table(report(case(1, 0.0), case(5, 0.0)),
                                         table(self.entry(bands=((1, 16, "a"),))), 0.05)[0][2])
        self.assertEqual(prune.segments(17, 16777216, 131.5), [(17, 131), (132, 16777216)])
        self.assertEqual(prune.segments(1, 16, None), [(1, 4), (5, 16)])
        self.assertEqual(prune.segments(6, 9, 131.5), [(6, 9)])

    def test_r8_bands_must_lie_inside_the_measured_segments(self):
        e = self.entry(bands=((5, 16, "c"),))
        e["r8_bands"] = [[1, 1, "r8.w8r2"], [3, 4, "r8.w4r2"]]
        sparse = report(case(8, 0.0))
        plan = prune.plan_table(sparse, table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 1..1: no case", plan[0][3])
        self.assertIn("T 3..4: no case", plan[0][3])
        self.assertTrue(prune.plan_table(report(case(8, 0.0), case(1, 0.0), case(3, 0.0)), table(e),
                                         0.05)[0][2])
        only_r8 = self.entry(bands=())
        only_r8["r8_bands"] = [[3, 4, "r8.w4r2"]]
        self.assertTrue(prune.plan_table(report(case(3, 0.0)), table(only_r8), 0.05)[0][2])

    def test_a_claim_keeps_the_entry(self):
        e = self.entry(bands=((1, 16, "a"),))
        e["claim"] = True
        plan = prune.plan_table(report(case(1, 0.0), case(8, 0.0)), table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("claims" in why for why in plan[0][3]), plan[0][3])

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

    def test_the_cases_t_field_decides_the_width(self):
        t = table(self.entry(bands=((5, 16, "c"),)))
        moved = dict(case(8, 0.0), t=2)  # the id says T8, the field T2: outside the band
        self.assertFalse(prune.plan_table(report(moved), t, 0.05)[0][2])
        self.assertTrue(prune.plan_table(report(dict(case(2, 0.0), t=8)), t, 0.05)[0][2])

    def test_other_authorities_and_other_evidence_do_not_speak(self):
        t = table(self.entry(bands=((1, 16, "a"),)))
        r = report(case(1, 0.0, authority="cost-model", evidence="ranker"),
                   case(8, 0.0, evidence="borrowed"))
        self.assertFalse(prune.plan_table(r, t, 0.05)[0][2])
        self.assertEqual(prune.plan_table(report(case(1, 0.0, n=7)), t, 0.05)[0][3], ["no case"])
        self.assertEqual(prune.plan_table(report(), table(entry(measured=False)), 0.05), [])

    def test_a_key_another_table_holds_or_may_borrow_is_kept(self):
        e = self.entry(bands=((1, 16, "a"),))
        t = table(e)
        r = report(case(1, 0.0), case(8, 0.0))
        lender = table(entry(measured=True, bands=((1, 16, "b"),)))
        lacker = table(entry(n=7))
        # a table that holds the key: the board would borrow its override
        plan = prune.plan_table(r, t, 0.05, {"gb206": lender})
        self.assertFalse(plan[0][2])
        self.assertTrue(any("holds this key (gb206)" in why for why in plan[0][3]), plan[0][3])
        # a table that lacks it may borrow ours
        plan = prune.plan_table(r, t, 0.05, {"ga106": lacker})
        self.assertFalse(plan[0][2])
        self.assertTrue(any("may borrow it (ga106)" in why for why in plan[0][3]), plan[0][3])
        self.assertTrue(prune.plan_table(r, t, 0.05, {"ga106": lacker}, allow_borrowers=True)[0][2])
        # allowing borrowers does not allow a lender
        self.assertFalse(prune.plan_table(r, t, 0.05, {"gb206": lender}, allow_borrowers=True)[0][2])
        self.assertTrue(prune.plan_table(r, t, 0.05, {})[0][2])

    def test_dense_tune_problems_of_the_class(self):
        dense = {"schema": "ninferno-dense-tune/4", "problems": {
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20": {
                "routes": {"1": "vec", "2": None, "8": "rmma.a"}, "prefill": {}},
            "ad102:128sm|q4k|a16|store|n2048|k4096|s20": {"routes": {"1": None}, "prefill": {}},
            "ga106:28sm|q4k|a16|store|n1024|k4096|s20": {"routes": {"1": "vec"}, "prefill": {}}}}
        cases = [case(1, 0.0, authority="dense-tune"), case(8, 0.01, authority="dense-tune")]
        plan = prune.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)
        self.assertEqual([(p[1], p[2]) for p in plan], [(self.KEY, True)])
        plan = prune.plan_dense(report(cases[0]), dense, "ad102:128sm", 0.05)
        self.assertFalse(plan[0][2])
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

    def table_of(self, cls, sms, *entries):
        t = table(*entries)
        t["gpu"] = {"class": cls, "name": "x", "board": {"sms": sms}}
        return t

    def test_dry_run_writes_nothing_and_apply_deletes_only_the_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = self.entry(n=2048, bands=((1, 16, "a"),))
            t = self.table_of("ad102", 128, self.entry(bands=((1, 16, "a"),)), keep,
                              entry(fmt="q8", measured=False))
            rep = report(case(1, 0.0), case(8, 0.0), case(1, 0.3, n=2048), case(8, 0.3, n=2048))
            rep["cases"] = [dict(c, board="ad102:128sm") for c in rep["cases"]]
            report_path = self.write(root, "report.json", rep)
            path = root / "table.json"
            path.write_text(prune.canonical(t), encoding="utf-8", newline="")
            before = path.read_bytes()
            code, out, _ = self.run_cli([str(report_path), "--table", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("board ad102:128sm, table class ad102", out)
            self.assertIn("1 of 2 measured entries", out)
            self.assertIn("dry run", out)
            self.assertEqual(path.read_bytes(), before)
            code, out, _ = self.run_cli([str(report_path), "--table", str(path), "--apply"])
            self.assertEqual(code, 0)
            after = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(after["entries"], [keep, entry(fmt="q8", measured=False)])
            self.assertEqual(list(root.glob("*.tmp")), [])
            # a file not in the canonical form is refused, not reformatted
            path.write_text(json.dumps(t, indent=2), encoding="utf-8")
            code, _, err = self.run_cli([str(report_path), "--table", str(path), "--apply"])
            self.assertEqual(code, 2)
            self.assertIn("canonical", err)

    def test_the_other_tables_of_the_directory_decide_the_lending_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mine = self.table_of("ad102", 128, self.entry(bands=((1, 16, "a"),)))
            rep = report(dict(case(1, 0.0), board="ad102:128sm"),
                         dict(case(8, 0.0), board="ad102:128sm"))
            report_path = self.write(root, "report.json", rep)
            path = root / "ad102.json"
            path.write_text(prune.canonical(mine), encoding="utf-8", newline="")
            other = self.table_of("ga106", 28, entry(n=7))
            (root / "ga106.json").write_text(prune.canonical(other), encoding="utf-8")
            code, out, _ = self.run_cli([str(report_path), "--table", str(path)])
            self.assertIn("may borrow it (ga106)", out)
            code, out, _ = self.run_cli([str(report_path), "--table", str(path), "--allow-borrowers"])
            self.assertIn("prune q4k", out)
            lender = self.table_of("gb206", 36, self.entry(bands=((1, 16, "b"),)))
            (root / "gb206.json").write_text(prune.canonical(lender), encoding="utf-8")
            code, out, _ = self.run_cli([str(report_path), "--table", str(path), "--allow-borrowers"])
            self.assertIn("holds this key (gb206)", out)
            self.assertNotIn("prune q4k", out)

    def test_both_files_are_written_or_neither(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a.json", root / "b.json"
            a.write_text(prune.canonical({"x": 1}), encoding="utf-8", newline="")
            b.write_text(prune.canonical({"y": 1}), encoding="utf-8", newline="")
            before = (a.read_bytes(), b.read_bytes())
            real = prune.os.replace
            calls = []

            def failing(src, dst):
                calls.append(dst)
                if len(calls) == 2:
                    raise OSError("disk full")
                real(src, dst)
            prune.os.replace = failing
            try:
                with self.assertRaises(OSError):
                    prune.write_both([(a, {"x": 2}), (b, {"y": 2})])
            finally:
                prune.os.replace = real
            self.assertEqual((a.read_bytes(), b.read_bytes()), before)
            self.assertEqual(list(root.glob("*.tmp")), [])
            prune.write_both([(a, {"x": 2}), (b, {"y": 2})])
            self.assertEqual(json.loads(a.read_text(encoding="utf-8")), {"x": 2})
            self.assertEqual(json.loads(b.read_text(encoding="utf-8")), {"y": 2})

    def test_the_report_the_table_and_the_class_must_name_one_board(self):
        # 5080 (gb203, bare, 84 SMs) and 5070 Ti (gb203:70sm), L4 / 4070 Ti (ad104), 3090 / A10
        bare84 = self.table_of("gb203", 84)
        full70 = self.table_of("gb203:70sm", 70)
        self.assertTrue(prune.board_matches_table("gb203", bare84))
        self.assertTrue(prune.board_matches_table("gb203:84sm", bare84))
        self.assertFalse(prune.board_matches_table("gb203:70sm", bare84))
        self.assertFalse(prune.board_matches_table("gb203", full70))
        self.assertTrue(prune.board_matches_table("gb203:70sm", full70))
        self.assertFalse(prune.board_matches_table("ad104:60sm", self.table_of("ad104", 58)))
        self.assertTrue(prune.board_matches_table("ad104:58sm", self.table_of("ad104", 58)))
        self.assertFalse(prune.board_matches_table("ga102:72sm", self.table_of("ga102", 82)))
        self.assertFalse(prune.board_matches_table("ga106", bare84))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table_path = self.write(root, "gb203.json", bare84)
            for board, argv, why in (
                    ("gb203:70sm", [], "not the table's board"),
                    ("ga106:28sm", [], "not the table's board"),
                    ("gb203:84sm", ["--class", "gb203"], "not a full class id"),
                    ("gb203:84sm", ["--class", "gb203:70sm"], "differs from the report's board")):
                rep = report(dict(case(1, 0.0), board=board))
                report_path = self.write(root, "r.json", rep)
                code, _, err = self.run_cli([str(report_path), "--table", str(table_path)] + argv)
                self.assertEqual(code, 2, (board, argv))
                self.assertIn(why, err)
            rep = report(dict(case(1, 0.0), board="gb203:84sm"))
            code, out, _ = self.run_cli([str(self.write(root, "r.json", rep)), "--table",
                                         str(table_path), "--class", "gb203:84sm"])
            self.assertEqual(code, 0)
            self.assertIn("board gb203:84sm, table class gb203", out)

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
