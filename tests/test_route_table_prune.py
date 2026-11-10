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

    def test_pruned_only_when_every_regime_is_within_the_resolution(self):
        t = table(entry())
        within = report(case(1, 0.02), case(8, 0.0), case(64, 0.049))
        self.assertTrue(decided(prune.plan_table(within, t, 0.05))[self.KEY])
        worse = report(case(1, 0.02), case(8, 0.0), case(64, 0.2))
        plan = prune.plan_table(worse, t, 0.05)
        self.assertFalse(decided(plan)[self.KEY])
        self.assertIn("wide T>=17: ranker 20.0 % slower somewhere", plan[0][3])
        # the resolution is a parameter: 25 % keeps nothing of it
        self.assertTrue(decided(prune.plan_table(worse, t, 0.25))[self.KEY])

    def test_a_regime_without_a_case_or_a_ranker_time_keeps_the_entry(self):
        t = table(entry())
        for r, why in ((report(case(1, 0.0), case(8, 0.0)), "wide T>=17: no case"),
                       (report(case(1, 0.0), case(8, 0.0), case(64, None)),
                        "a case without a ranker time"),
                       (report(case(1, 0.0), case(8, 0.0), case(64, 0.0, status="incomplete")),
                        "a case without a ranker time")):
            plan = prune.plan_table(r, t, 0.05)
            self.assertFalse(plan[0][2])
            self.assertTrue(any(why in reason for reason in plan[0][3]), plan[0][3])

    def test_only_the_regimes_an_entry_has_routes_in_are_asked(self):
        decode_only = entry(bands=((1, 4, "vec"),), prefill=None)
        self.assertEqual(prune.regimes_of_entry(decode_only), {"decode T<=4"})
        t = table(decode_only)
        self.assertTrue(decided(prune.plan_table(report(case(2, 0.0)), t, 0.05))[self.KEY])
        wide = entry(bands=((1, 4, "vec"), (17, 64, "mmq")), prefill=None)
        self.assertEqual(prune.regimes_of_entry(wide), {"decode T<=4", "wide T>=17"})

    def test_other_authorities_and_other_evidence_do_not_speak(self):
        t = table(entry())
        r = report(case(1, 0.0), case(8, 0.0), case(64, 0.0, authority="cost-model", evidence="ranker"),
                   case(128, 0.0, evidence="borrowed"))
        plan = prune.plan_table(r, t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("wide T>=17: no case", plan[0][3])
        # nor does an entry no case names; unmeasured entries (no tuned_on) are not listed at all
        self.assertEqual(prune.plan_table(report(case(1, 0.0, n=7)), table(entry()), 0.05)[0][3],
                         ["no case"])
        self.assertEqual(prune.plan_table(report(), table(entry(measured=False)), 0.05), [])

    def test_dense_tune_problems_of_the_class(self):
        dense = {"schema": "ninferno-dense-tune/4", "problems": {
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20": {
                "routes": {"1": "vec", "2": None, "8": "rmma.a"}, "prefill": {"64": "mmq"}},
            "ad102:128sm|q4k|a16|store|n2048|k4096|s20": {"routes": {"1": None}, "prefill": {}},
            "ga106:28sm|q4k|a16|store|n1024|k4096|s20": {"routes": {"1": "vec"}, "prefill": {}}}}
        cases = [case(1, 0.0, authority="dense-tune"), case(8, 0.01, authority="dense-tune"),
                 case(64, 0.01, authority="dense-tune")]
        plan = prune.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)
        # the all-null problem and the other class's are not listed
        self.assertEqual([(p[1], p[2]) for p in plan], [(self.KEY, True)])
        cases[2] = case(64, 0.3, authority="dense-tune")
        self.assertFalse(prune.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)[0][2])

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = prune.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_writes_nothing_and_apply_deletes_only_the_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = entry(n=2048)
            t = table(entry(), keep, entry(fmt="q8", measured=False))
            rep = report(case(1, 0.0), case(8, 0.0), case(64, 0.0),
                         case(1, 0.3, n=2048), case(8, 0.3, n=2048), case(64, 0.3, n=2048))
            rep["cases"] = [dict(c, board="board") for c in rep["cases"]]
            (root / "report.json").write_text(json.dumps(rep), encoding="utf-8")
            path = root / "table.json"
            path.write_text(prune.canonical(t), encoding="utf-8", newline="")
            before = path.read_bytes()
            code, out, _ = self.run_cli([str(root / "report.json"), "--table", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("1 of 2 measured entries", out)
            self.assertIn("dry run", out)
            self.assertEqual(path.read_bytes(), before)
            code, out, _ = self.run_cli([str(root / "report.json"), "--table", str(path), "--apply"])
            self.assertEqual(code, 0)
            after = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(after["entries"], [keep, entry(fmt="q8", measured=False)])
            # a file not in the canonical form is refused, not reformatted
            path.write_text(json.dumps(t, indent=2), encoding="utf-8")
            code, _, err = self.run_cli([str(root / "report.json"), "--table", str(path), "--apply"])
            self.assertEqual(code, 2)
            self.assertIn("canonical", err)

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
