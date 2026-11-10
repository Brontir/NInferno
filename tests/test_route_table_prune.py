"""route_table_prune.py (#614, PR-F3): the pruning plan on synthetic regret reports, and the
repository's evidence check: every measured route table entry records evidence_us with a gain above
the resolution, except the gaps in tests/route_evidence_gaps.json (the tool's --gaps output: no new
entry may join it, and a change to routes/ that closes or changes a gap regenerates it)."""
import contextlib
import copy
import io
import json
import re
import subprocess
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
         status="complete", with_ranker=True, selected=None):
    """A report case; `selected` (the route it measured) is filled from the table or the dense
    problem it is planned against (Prune.plan_table, plan_dense) unless given."""
    c = {"id": f"{fmt}.a16.store.n{n}.k{k}.T{t}", "board": "board", "t": t,
         "authority": authority, "evidence": evidence, "status": status}
    if selected is not None:
        c["selected"] = selected
    if with_ranker and slowdown is not None:
        c["ranker"] = "cost"
        c["ranker_vs_selected"] = slowdown
    return c


def named_route(ranges, t):
    """The route of the first [lo, hi, route] holding t, or None (the test's own lookup)."""
    for lo, hi, route in ranges:
        if lo <= t <= hi:
            return route
    return None


def measured_against(rep, entries=(), dense=None):
    """`rep` as if collected against `entries` (and `dense`'s problems): every case without a
    `selected` gets the route its entry names at its width (bands, then r8_bands; prefill-profile
    cases: prefill_bands, then the prefill route), so it speaks for that entry."""
    rep = copy.deepcopy(rep)
    for c in rep["cases"]:
        if "selected" in c:
            continue
        t = c["t"]
        for e in entries:
            if c["id"].startswith(prune.label(prune.entry_key(e)) + ".T"):
                if c.get("profile") == "prefill":
                    route = named_route(e.get("prefill_bands", []), t) or e.get("prefill")
                else:
                    route = named_route(e.get("bands", []) + e.get("r8_bands", []), t)
                c["selected"] = route or "unrouted"
        for text, problem in (dense or {}).get("problems", {}).items():
            if c["id"].startswith(prune.label(prune.dense_key(text)[1]) + ".T"):
                c.setdefault("selected", problem["routes"].get(str(t)) or "unrouted")
    return rep


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

    def plan_table(self, rep, t, *args, **kw):
        """prune.plan_table on `rep` as collected against `t` (measured_against)."""
        return prune.plan_table(measured_against(rep, t["entries"]), t, *args, **kw)

    def plan_dense(self, rep, dense, *args, **kw):
        return prune.plan_dense(measured_against(rep, dense=dense), dense, *args, **kw)

    def test_pruned_only_when_every_band_is_within_the_resolution(self):
        t = table(self.entry(bands=((1, 4, "vec"), (5, 16, "rmma.a"))))
        within = report(case(1, 0.02), case(4, 0.0), case(8, 0.049))
        self.assertTrue(decided(self.plan_table(within, t, 0.05))[self.KEY])
        worse = report(case(1, 0.02), case(4, 0.0), case(8, 0.2))
        plan = self.plan_table(worse, t, 0.05)
        self.assertFalse(decided(plan)[self.KEY])
        self.assertIn("T 5..16: ranker 20.0 % slower somewhere", plan[0][3])
        self.assertTrue(decided(self.plan_table(worse, t, 0.25))[self.KEY])

    def test_a_prefill_override_is_kept_on_speed_evidence_alone(self):
        # the reproducer: bands up to 16777216, prefill_tuned_on and prefill_bands, T = 1 / 8 / 17
        e = self.entry(bands=((1, 4, "vec"), (5, 16, "rmma.a"), (17, 16777216, "mmq")))
        e["prefill"] = "mmq"
        e["prefill_bands"] = [[17, 128, "w4a8.64x64"]]
        e["prefill_tuned_on"] = "ad102:128sm"
        plan = self.plan_table(with_ridge(report(case(1, 0.0), case(8, 0.0), case(17, 0.0))),
                                table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("prefill" in why for why in plan[0][3]), plan[0][3])
        speed = [case(1, 0.0), case(8, 0.0), case(17, 0.0), case(512, 0.0)]
        # prefill-profile cases (none are written yet) clear it, in every prefill range: the band
        # 17..128 and the route past it (129.., cut at the ridge: 129..131 and 132..)
        pre = [dict(case(t, 0.01), profile="prefill") for t in (64, 17, 130, 200, 512)]
        self.assertTrue(self.plan_table(with_ridge(report(*speed, *pre)), table(e), 0.05)[0][2])
        lone = [dict(case(64, 0.01), profile="prefill")]
        plan = self.plan_table(with_ridge(report(*speed, *lone)), table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any(why.startswith("prefill T ") for why in plan[0][3]), plan[0][3])
        slow = [dict(case(t, 0.3 if t == 512 else 0.0), profile="prefill")
                for t in (64, 17, 130, 200, 512)]
        self.assertFalse(self.plan_table(with_ridge(report(*speed, *slow)), table(e), 0.05)[0][2])

    def test_every_band_needs_a_case_inside_its_range(self):
        bands = ((1, 1, "a"), (2, 4, "b"), (5, 16, "c"), (17, 16777216, "d"))
        t = table(self.entry(bands=bands))
        sparse = with_ridge(report(case(1, 0.0), case(5, 0.0), case(17, 0.0), case(512, 0.0)))
        plan = self.plan_table(sparse, t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 2..4: no case", plan[0][3])
        full = with_ridge(report(case(1, 0.0), case(3, 0.0), case(5, 0.0), case(17, 0.0),
                                 case(512, 0.0)))
        self.assertTrue(self.plan_table(full, t, 0.05)[0][2])

    def test_a_wide_band_needs_the_ridge_and_a_case_on_each_side(self):
        t = table(self.entry(bands=((1, 16, "a"), (17, 16777216, "d"))))
        cases = [case(1, 0.0), case(5, 0.0), case(17, 0.0)]
        plan = self.plan_table(with_ridge(report(*cases)), t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 132..16777216: no case", plan[0][3])
        self.assertTrue(self.plan_table(with_ridge(report(*cases, case(512, 0.0))), t, 0.05)[0][2])
        # without a ridge in the report a wide band cannot be split: kept
        plan = self.plan_table(report(*cases, case(512, 0.0)), t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("no ridge" in why for why in plan[0][3]), plan[0][3])
        # a decode-only entry needs none
        self.assertTrue(self.plan_table(report(case(1, 0.0), case(5, 0.0)),
                                         table(self.entry(bands=((1, 16, "a"),))), 0.05)[0][2])
        self.assertEqual(prune.segments(17, 16777216, 131.5), [(17, 131), (132, 16777216)])
        self.assertEqual(prune.segments(1, 16, None), [(1, 4), (5, 16)])
        self.assertEqual(prune.segments(6, 9, 131.5), [(6, 9)])

    def test_r8_bands_must_lie_inside_the_measured_segments(self):
        e = self.entry(bands=((5, 16, "c"),))
        e["r8_bands"] = [[1, 1, "r8.w8r2"], [3, 4, "r8.w4r2"]]
        sparse = report(case(8, 0.0))
        plan = self.plan_table(sparse, table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 1..1: no case", plan[0][3])
        self.assertIn("T 3..4: no case", plan[0][3])
        self.assertTrue(self.plan_table(report(case(8, 0.0), case(1, 0.0), case(3, 0.0)), table(e),
                                         0.05)[0][2])
        only_r8 = self.entry(bands=())
        only_r8["r8_bands"] = [[3, 4, "r8.w4r2"]]
        self.assertTrue(self.plan_table(report(case(3, 0.0)), table(only_r8), 0.05)[0][2])
        # an r8 band inside a band: a case that measured the band's route says nothing of the R8
        # route, so cases at T 1, 3 and 8 that all ran "a" leave the R8 band unmeasured
        overlap = self.entry(bands=((1, 16, "a"),))
        overlap["r8_bands"] = [[3, 4, "r8.w4r2"]]
        ran_a = report(*[case(t, 0.0, selected="a") for t in (1, 3, 8)])
        plan = prune.plan_table(ran_a, table(overlap), 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 3..4: no case", plan[0][3])
        both = report(*[case(t, 0.0, selected="a") for t in (1, 3, 8)],
                      case(4, 0.0, selected="r8.w4r2"))
        self.assertTrue(prune.plan_table(both, table(overlap), 0.05)[0][2])

    def test_a_prefill_fallback_past_its_bands_needs_its_own_case(self):
        # prefill_bands 17..1000 "A" and the prefill route "B", which answers only past 1000
        # (table_route_answer): cases that ran "A" at T 64 and 512 do not measure "B"
        e = self.entry(bands=())
        e["prefill"] = "B"
        e["prefill_bands"] = [[17, 1000, "A"]]
        e["prefill_tuned_on"] = "ad102:128sm"
        pre = [dict(case(t, 0.0, selected="A"), profile="prefill") for t in (64, 512)]
        plan = prune.plan_table(with_ridge(report(*pre)), table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any(why.startswith("prefill T 1001..") and why.endswith("no case")
                            for why in plan[0][3]), plan[0][3])
        self.assertEqual(prune.entry_ranges(e)[1], [(17, 1000, "A"), (1001, prune.INF, "B")])
        past = dict(case(2000, 0.0, selected="B"), profile="prefill")
        self.assertTrue(prune.plan_table(with_ridge(report(*pre, past)), table(e), 0.05)[0][2])
        # the fallback fills every gap the bands leave, below them too
        gap = dict(e, prefill_bands=[[64, 1000, "A"]])
        self.assertEqual(prune.entry_ranges(gap)[1],
                         [(64, 1000, "A"), (17, 63, "B"), (1001, prune.INF, "B")])

    def test_a_claim_keeps_the_entry(self):
        e = self.entry(bands=((1, 16, "a"),))
        e["claim"] = True
        plan = self.plan_table(report(case(1, 0.0), case(8, 0.0)), table(e), 0.05)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("claims" in why for why in plan[0][3]), plan[0][3])

    def test_a_case_without_a_finite_ranker_time_keeps_the_entry(self):
        t = table(self.entry(bands=((1, 16, "a"),)))
        for bad in (float("nan"), float("inf"), None, "0.0", True):
            c = case(1, 0.0)
            c["ranker_vs_selected"] = bad
            plan = self.plan_table(report(c, case(8, 0.0)), t, 0.05)
            self.assertFalse(plan[0][2], bad)
            self.assertTrue(any("without a ranker time" in why for why in plan[0][3]), bad)
        incomplete = report(case(1, 0.0, status="incomplete"), case(8, 0.0))
        self.assertFalse(self.plan_table(incomplete, t, 0.05)[0][2])

    def test_the_cases_t_field_decides_the_width(self):
        t = table(self.entry(bands=((5, 16, "c"),)))
        moved = dict(case(8, 0.0), t=2)  # the id says T8, the field T2: outside the band
        self.assertFalse(self.plan_table(report(moved), t, 0.05)[0][2])
        self.assertTrue(self.plan_table(report(dict(case(2, 0.0), t=8)), t, 0.05)[0][2])

    def test_other_authorities_and_other_evidence_do_not_speak(self):
        t = table(self.entry(bands=((1, 16, "a"),)))
        r = report(case(1, 0.0, authority="cost-model", evidence="ranker"),
                   case(8, 0.0, evidence="borrowed"))
        self.assertFalse(self.plan_table(r, t, 0.05)[0][2])
        self.assertEqual(self.plan_table(report(case(1, 0.0, n=7)), t, 0.05)[0][3], ["no case"])
        self.assertEqual(self.plan_table(report(), table(entry(measured=False)), 0.05), [])

    def test_a_key_another_table_holds_or_may_borrow_is_kept(self):
        e = self.entry(bands=((1, 16, "a"),))
        t = table(e)
        r = report(case(1, 0.0), case(8, 0.0))
        lender = table(entry(measured=True, bands=((1, 16, "b"),)))
        lacker = table(entry(n=7))
        # a table that holds the key: the board would borrow its override
        plan = self.plan_table(r, t, 0.05, {"gb206": lender})
        self.assertFalse(plan[0][2])
        self.assertTrue(any("holds this key (gb206)" in why for why in plan[0][3]), plan[0][3])
        # a table that lacks it may borrow ours
        plan = self.plan_table(r, t, 0.05, {"ga106": lacker})
        self.assertFalse(plan[0][2])
        self.assertTrue(any("may borrow it (ga106)" in why for why in plan[0][3]), plan[0][3])
        self.assertTrue(self.plan_table(r, t, 0.05, {"ga106": lacker}, allow_borrowers=True)[0][2])
        # allowing borrowers does not allow a lender
        self.assertFalse(self.plan_table(r, t, 0.05, {"gb206": lender}, allow_borrowers=True)[0][2])
        self.assertTrue(self.plan_table(r, t, 0.05, {})[0][2])

    def test_dense_tune_problems_of_the_class(self):
        dense = {"schema": "ninferno-dense-tune/4", "problems": {
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20": {
                "routes": {"1": "vec", "2": None, "4": "vec.w4", "8": "rmma.a"}, "prefill": {}},
            "ad102:128sm|q4k|a16|store|n2048|k4096|s20": {"routes": {"1": None}, "prefill": {}},
            "ga106:28sm|q4k|a16|store|n1024|k4096|s20": {"routes": {"1": "vec"}, "prefill": {}}}}
        cases = [case(t, 0.01, authority="dense-tune") for t in (1, 4, 8)]
        plan = self.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)
        self.assertEqual([(p[1], p[2]) for p in plan], [(self.KEY, True)])
        plan = self.plan_dense(report(cases[0]), dense, "ad102:128sm", 0.05)
        self.assertFalse(plan[0][2])
        # the runtime takes a dense route by the exact width (tuned_dense_route): T = 1 says
        # nothing of the T = 4 route, though both are decode widths
        plan = self.plan_dense(report(cases[0], cases[2]), dense, "ad102:128sm", 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 4..4: no case", plan[0][3])
        # a case at the unrouted T = 2 (the heuristic's width) does not stand in for T = 4 either
        plan = self.plan_dense(report(cases[0], case(2, 0.0, authority="dense-tune"), cases[2]),
                               dense, "ad102:128sm", 0.05)
        self.assertFalse(plan[0][2])
        dense["problems"]["ad102:128sm|q4k|a16|store|n1024|k4096|s20"]["prefill"] = {"64": "w4a8"}
        self.assertFalse(self.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)[0][2])
        cases[2] = case(8, 0.3, authority="dense-tune")
        self.assertFalse(self.plan_dense(report(*cases), dense, "ad102:128sm", 0.05)[0][2])

    def test_a_dense_prefill_route_needs_a_case_in_its_own_band(self):
        # prefill bands of generic_tune.cpp kPrefillBandTokens: "64" takes T 33..64, "2048" every
        # width from 1025
        dense = {"problems": {"ad102:128sm|q4k|a16|store|n1024|k4096|s20": {
            "routes": {}, "prefill": {"64": "w4a8.64x64", "2048": "w4a8.128x128", "32": None}}}}
        speed, prefill = prune.dense_ranges(dense["problems"][next(iter(dense["problems"]))])
        self.assertEqual((speed, prefill), ([], [(33, 64, "w4a8.64x64"),
                                                 (1025, prune.INF, "w4a8.128x128")]))
        rep = with_ridge(report(*[dict(case(t, 0.0, authority="dense-tune", selected=route),
                                       profile="prefill")
                                  for t, route in ((40, "w4a8.64x64"), (4096, "w4a8.128x128"))]))
        self.assertTrue(prune.plan_dense(rep, dense, "ad102:128sm", 0.05)[0][2])
        # a case at T = 20 (band "32", which keeps the heuristic) covers neither routed band
        rep["cases"][0] = dict(case(20, 0.0, authority="dense-tune", selected="w4a8.64x64"),
                               profile="prefill")
        plan = prune.plan_dense(rep, dense, "ad102:128sm", 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("prefill T 33..64: no case", plan[0][3])
        # the selector names a dense prefill band's answer dense-tune-prefill-band
        # (route_authorities.cpp): those cases speak for the problem too
        named = with_ridge(report(*[dict(case(t, 0.0, authority="dense-tune-prefill-band",
                                              selected=route), profile="prefill")
                                    for t, route in ((40, "w4a8.64x64"),
                                                     (4096, "w4a8.128x128"))]))
        self.assertTrue(prune.plan_dense(named, dense, "ad102:128sm", 0.05)[0][2])

    def test_the_dense_prefill_bands_are_the_tuners(self):
        # DENSE_PREFILL_BANDS mirrors generic_tune.cpp's kPrefillBandTokens and the prefill keys
        # of the release's routes/tuner/dense_tune.json
        source = (ROOT / "src/ops/generic/generic_tune.cpp").read_text(encoding="utf-8")
        found = re.search(r"kPrefillBandTokens\s*=\s*\{([^}]*)\}", source)
        self.assertIsNotNone(found)
        literal = tuple(int(v) for v in found[1].replace("\n", " ").split(",") if v.strip())
        self.assertEqual(prune.DENSE_PREFILL_BANDS, literal)
        dense = json.loads((ROOT / "routes/tuner/dense_tune.json").read_text(encoding="utf-8"))
        keys = {int(t) for p in dense["problems"].values() for t in p.get("prefill", {})}
        self.assertTrue(keys)
        self.assertLessEqual(keys, set(prune.DENSE_PREFILL_BANDS))

    def test_a_case_that_measured_another_route_keeps_the_entry(self):
        # a report collected before the entry changed: its cases timed the old route
        t = table(self.entry(bands=((1, 16, "a"),)))
        old = report(case(1, 0.0, selected="b"), case(8, 0.0, selected="b"))
        plan = prune.plan_table(old, t, 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 1: the case measured b, the entry routes a there", plan[0][3])
        self.assertTrue(prune.plan_table(report(case(1, 0.0, selected="a"),
                                                case(8, 0.0, selected="a")), t, 0.05)[0][2])
        # one stale case is enough to keep it, whatever the others measured
        mixed = report(case(1, 0.0, selected="a"), case(8, 0.0, selected="a"),
                       case(9, 0.0, selected="b"))
        self.assertFalse(prune.plan_table(mixed, t, 0.05)[0][2])
        # a case without a selected route speaks for nothing
        self.assertFalse(prune.plan_table(report(case(1, 0.0), case(8, 0.0)), t, 0.05)[0][2])
        # the dense tuner's problems alike
        dense = {"problems": {"ad102:128sm|q4k|a16|store|n1024|k4096|s20": {
            "routes": {"1": "vec"}, "prefill": {}}}}
        stale = report(case(1, 0.0, authority="dense-tune", selected="rmma.a"))
        plan = prune.plan_dense(stale, dense, "ad102:128sm", 0.05)
        self.assertFalse(plan[0][2])
        self.assertIn("T 1: the case measured rmma.a, the entry routes vec there", plan[0][3])
        fresh = report(case(1, 0.0, authority="dense-tune", selected="vec"))
        self.assertTrue(prune.plan_dense(fresh, dense, "ad102:128sm", 0.05)[0][2])

    def test_a_table_entry_the_dense_tuner_also_routes_is_kept(self):
        # once the entry is gone the tuner's route answers, not the ranker whose time the case has
        t = table(self.entry(bands=((1, 16, "a"),)))
        r = report(case(1, 0.0), case(8, 0.0))
        dense = {"problems": {
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20": {"routes": {"1": "vec"}, "prefill": {}},
            "ad102:128sm|q4k|a16|store|n2048|k4096|s20": {"routes": {"1": None}, "prefill": {}},
            "ga106:28sm|q4k|a16|store|n4096|k4096|s20": {"routes": {"1": "vec"}, "prefill": {}}}}
        tuned = prune.dense_routed(dense, "ad102:128sm")
        self.assertEqual(tuned, {self.KEY})
        plan = self.plan_table(r, t, 0.05, tuned=tuned)
        self.assertFalse(plan[0][2])
        self.assertTrue(any("dense tuner routes this problem" in why for why in plan[0][3]))
        self.assertTrue(self.plan_table(r, t, 0.05, tuned=prune.dense_routed(dense, "x:1sm"))[0][2])

    def test_apply_keeps_a_dense_problem_whose_t4_route_was_never_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "ad102:128sm|q4k|a16|store|n1024|k4096|s20"
            dense = {"schema": "ninferno-dense-tune/4", "problems": {
                key: {"routes": {"1": "vec", "2": None, "4": "vec.w4"}, "prefill": {}}}}
            path = root / "dense_tune.json"
            path.write_text(prune.canonical(dense), encoding="utf-8", newline="")
            rep = report(dict(case(1, 0.0, authority="dense-tune", selected="vec"),
                              board="ad102:128sm"))
            rep["device"] = "ad102:128sm GPU-0"
            report_path = self.write(root, "report.json", rep)
            code, out, _ = self.run_cli([str(report_path), "--dense-tune", str(path), "--class",
                                         "ad102:128sm", "--apply"])
            self.assertEqual(code, 0)
            self.assertIn("T 4..4: no case", out)
            self.assertIn(key, json.loads(path.read_text(encoding="utf-8"))["problems"])

    def test_dense_keys_that_collide_after_parsing_are_refused(self):
        dense = {"problems": {
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20.13": {"routes": {"1": "vec"}, "prefill": {}},
            "ad102:128sm|q4k|a16|store|n1024|k4096|s20.13|mmq": {"routes": {"1": "x"},
                                                                  "prefill": {}}}}
        with self.assertRaisesRegex(ValueError, "collide"):
            self.plan_dense(report(case(1, 0.0, authority="dense-tune")), dense, "ad102:128sm", 0.05)

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
            rep = measured_against(report(case(1, 0.0), case(8, 0.0), case(1, 0.3, n=2048),
                                          case(8, 0.3, n=2048)), t["entries"])
            rep["cases"] = [dict(c, board="ad102:128sm") for c in rep["cases"]]
            rep["device"] = "ad102:128sm GPU-0"
            report_path = self.write(root, "report.json", rep)
            path = root / "table.json"
            path.write_text(prune.canonical(t), encoding="utf-8", newline="")
            before = path.read_bytes()
            code, out, _ = self.run_cli([str(report_path), "--table", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("board ad102:128sm, device ad102:128sm, table class ad102", out)
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
            rep = report(dict(case(1, 0.0, selected="a"), board="ad102:128sm"),
                         dict(case(8, 0.0, selected="a"), board="ad102:128sm"))
            rep["device"] = "ad102:128sm GPU-0"
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
                    ("gb203:84sm", ["--class", "gb203:70sm"], "differs from the report's device")):
                rep = dict(report(dict(case(1, 0.0), board="label")), device=board + " GPU-1")
                report_path = self.write(root, "r.json", rep)
                code, _, err = self.run_cli([str(report_path), "--table", str(table_path)] + argv)
                self.assertEqual(code, 2, (board, argv))
                self.assertIn(why, err)
            rep = dict(report(dict(case(1, 0.0), board="RTX 5080")), device="gb203:84sm GPU-1")
            code, out, _ = self.run_cli([str(self.write(root, "r.json", rep)), "--table",
                                         str(table_path), "--class", "gb203:84sm"])
            self.assertEqual(code, 0)
            self.assertIn("board RTX 5080, device gb203:84sm, table class gb203", out)

    def test_the_measured_device_not_the_label_names_the_board(self):
        # a 3060 Ti's report (ga104) labelled as the 3060's (ga106) is refused against ga106.json:
        # the label is the user's, the device is route_tune's own
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table_path = self.write(root, "ga106.json", self.table_of("ga106", 28))
            rep = dict(report(dict(case(1, 0.0), board="ga106:28sm")), device="ga104:38sm GPU-2")
            code, _, err = self.run_cli([str(self.write(root, "r.json", rep)), "--table",
                                         str(table_path)])
            self.assertEqual(code, 2)
            self.assertIn("the report measured ga104:38sm", err)
            code, _, err = self.run_cli([str(self.write(root, "r.json", rep)), "--dense-tune",
                                         str(table_path), "--class", "ga106:28sm"])
            self.assertEqual(code, 2)
            self.assertIn("differs from the report's device ga104:38sm", err)
            # a report that names no device (selection_regret.py before it passed it on)
            code, _, err = self.run_cli([str(self.write(root, "r.json", report(case(1, 0.0)))),
                                         "--table", str(table_path)])
            self.assertEqual(code, 2)
            self.assertIn("names no measured device", err)

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
        text = ALLOWLIST.read_text(encoding="utf-8")
        allowed = json.loads(text)
        self.assertEqual(allowed["schema"], "ninferno-route-evidence-gaps/1")
        regenerated = prune.gaps_document(ROOT / "routes")
        found, listed = regenerated["gaps"], allowed["gaps"]
        new = sorted(k for k in found if k not in listed)
        self.assertFalse(new, f"measured entries without evidence_us (or below the resolution) "
                              f"that are not in {ALLOWLIST.name}: record their evidence "
                              f"(route_tune); no new entry may join the list: {new[:5]} ...")
        # a gap closed, or changed in kind by a change to routes/ (here or on master): the list is
        # the tool's output, regenerated
        self.assertEqual(text, prune.canonical(regenerated),
                         f"{ALLOWLIST.name} is not the tables' gaps: regenerate it with "
                         f"tools/bench/route_table_prune.py --gaps routes --gaps-out "
                         f"tests/route_evidence_gaps.json")

    def test_gaps_cli_writes_the_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weak = entry(n=2)
            weak["evidence_us"] = [[7, 10.0, 9.8]]
            (root / "x.json").write_text(json.dumps(table(weak, entry(n=4))), encoding="utf-8")
            out_path = root / "gaps.json"
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                self.assertEqual(prune.main(["--gaps", str(root), "--gaps-out", str(out_path)]), 0)
                self.assertEqual(prune.main(["--gaps", str(root), "--apply"]), 2)
            self.assertEqual(json.loads(out_path.read_text(encoding="utf-8"))["gaps"],
                             {"x.json:q4k.a16.store.n2.k4096": "weak",
                              "x.json:q4k.a16.store.n4.k4096": "missing"})
            self.assertNotIn(b"\r\n", out_path.read_bytes())

    def test_the_tool_runs_isolated(self):
        # `python -I` puts no script directory on sys.path: the tool finds selection_regret itself
        script = ROOT / "tools" / "bench" / "route_table_prune.py"
        with tempfile.TemporaryDirectory() as directory:
            done = subprocess.run([sys.executable, "-I", "-B", str(script), "--gaps",
                                   str(ROOT / "routes")], cwd=directory, capture_output=True,
                                  text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["schema"], "ninferno-route-evidence-gaps/1")


if __name__ == "__main__":
    unittest.main()
