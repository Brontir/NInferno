"""tools/kit/validate.py: the user measurement kit's stated rejection rules and verdicts, no GPU
(docs/maintainer/user-measurement-kit.md)."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "kit"))
import gpu  # noqa: E402
import measure  # noqa: E402
import validate  # noqa: E402

P = "clocks_event_reasons"


def telemetry(util=1.0, reason=None, at="decode", mem_growth=0.0):
    """Samples every 0.5 s over 12 s: idle 0..5, prefill 5..7, decode 7..12."""
    t = [i * 0.5 for i in range(25)]
    cols = {"utilization.gpu": [util if x <= 5 else 99.0 for x in t],
            "memory.used": [1000.0 + (mem_growth if x >= 5 else 0.0) for x in t]}
    for r in validate.REJECT_REASONS + validate.RECORD_REASONS + ("applications_clocks_setting",):
        cols[f"{P}.{r}"] = ["Not Active"] * len(t)
    if reason:
        lo, hi = (5, 7) if at == "prefill" else (7, 12)
        cols[f"{P}.{reason}"] = ["Active" if lo <= x <= hi else "Not Active" for x in t]
    return {"t": t, "columns": cols, "reason_fields": P,
            "marks": [{"name": "idle", "start": 0, "end": 5}, {"name": "prefill 0", "start": 5, "end": 7},
                      {"name": "decode 0", "start": 7, "end": 12}]}


def one_pass(decode=50.0, prefill=3000.0, spread=0.02, tokens=8192, prompt=2048, **tel):
    return {"served": {"fingerprint": "nf1-aa", "choice": "KV int8, activations fast, state f16 | ..."},
            "processes_before": {"listed": 1, "others": 0, "informative": True},
            "telemetry": telemetry(**tel),
            "measured": {"decode": {"median": decode, "min": decode * (1 - spread / 2),
                                    "max": decode * (1 + spread / 2), "prompt_tokens": tokens},
                         "prefill": {"median": prefill, "min": prefill * 0.99, "max": prefill * 1.01,
                                     "prompt_tokens": prompt}}}


def doc(passes=None, smi=None):
    card = {"name": "NVIDIA GeForce RTX 4060 Ti", "power.limit": 165.0, "power.default_limit": 165.0,
            "reason_fields": P}
    card.update(smi or {})
    return {"schema": "ninferno-user-run/1.0", "host": {"on_battery": False},
            "card": {"smi": card, "plan_device": {"name": "NVIDIA GeForce RTX 4060 Ti"}},
            "runs": [{"id": "q9", "repo": "org/m", "quant": "Q4_K_M", "mode": "plain",
                      "plan": {"chosen": {"decode_tok_s_at_depth": 48.0, "prefill_tok_s": 3400.0,
                                          "depth": 8192, "kv": "int8", "activations": "fast"},
                               "prompt": 2048},
                      "passes": passes if passes is not None else [one_pass(), one_pass(decode=50.5)]}]}


def metric(report, name):
    return next(m for m in report["runs"][0]["metrics"] if m["metric"] == name)


class Rules(unittest.TestCase):
    def test_clean_run_confirms_within_tolerance(self):
        r = validate.validate(doc())
        self.assertEqual(r["file"], "accepted")
        d = metric(r, "decode")
        self.assertEqual(d["status"], "accepted")
        self.assertTrue(d["verdict"].startswith("confirms"))
        p = metric(r, "prefill")
        self.assertTrue(p["verdict"].startswith("contradicts"))  # 3000 against 3400: 12 % off

    def test_clock_and_power_of_each_metric(self):
        def loaded(**kw):
            p = one_pass(**kw)
            t = p["telemetry"]["t"]
            p["telemetry"]["columns"]["clocks.sm"] = [2400.0 if 5 <= x < 7 else 2800.0 for x in t]
            p["telemetry"]["columns"]["power.draw"] = [570.0 if 5 <= x < 7 else 300.0 for x in t]
            return p
        r = validate.validate(doc(passes=[loaded(), loaded(decode=50.5)]))
        self.assertEqual((metric(r, "prefill")["sm_clock_mhz"], metric(r, "prefill")["power_w"]), (2400, 570))
        self.assertEqual((metric(r, "decode")["sm_clock_mhz"], metric(r, "decode")["power_w"]), (2800, 300))
        self.assertIn("at 2400 MHz, 570 W", validate.render(r))
        # The instant draw, where the driver reads it, over the one-second average.
        p = loaded()
        p["telemetry"]["columns"]["power.draw.instant"] = [575.0 if 5 <= x < 7 else 290.0
                                                          for x in p["telemetry"]["t"]]
        r = validate.validate(doc(passes=[p, loaded(decode=50.5)]))
        self.assertEqual(metric(r, "prefill")["power_w"], 572)  # median of 575 and 570 samples
        # An excluded request's window (a reused prefix) is not the speed's: its samples are left out.
        p = loaded()
        p["requests"] = [{"kind": "prefill", "excluded": "a reused prefix (cache_n > 0)"}]
        q = loaded(decode=50.5)
        q["telemetry"]["columns"]["clocks.sm"] = [2600.0 if 5 <= x < 7 else 2800.0
                                                  for x in q["telemetry"]["t"]]
        self.assertEqual(validate.window_load([p, q], "prefill")[0], 2600)
        # No clock column (an old file): nothing is recorded or printed.
        self.assertNotIn("sm_clock_mhz", metric(validate.validate(doc()), "prefill"))

    def test_schema(self):
        bad = doc()
        bad["schema"] = "ninferno-user-run/2.0"
        self.assertTrue(validate.validate(bad)["file"].startswith("rejected: R0"))
        bad["schema"] = "ninferno-user-run/10.0"
        self.assertTrue(validate.validate(bad)["file"].startswith("rejected: R0"))
        bad["schema"] = "ninferno-user-run/1.3"
        self.assertEqual(validate.validate(bad)["file"], "accepted")

    def test_missing_names(self):
        bad = doc()
        bad["card"]["smi"].pop("name")
        self.assertTrue(validate.validate(bad)["file"].startswith("rejected: R1 unknown"))

    def test_identity(self):
        bad = doc()
        bad["card"]["plan_device"]["name"] = "NVIDIA GeForce RTX 4090"
        self.assertTrue(validate.validate(bad)["file"].startswith("rejected: R1"))
        other = doc(passes=[one_pass(), one_pass()])
        other["runs"][0]["passes"][1]["served"]["choice"] = "KV fp8, activations fast | ..."
        self.assertTrue(any(n.startswith("R1") for n in validate.validate(other)["runs"][0]["notes"]))

    def test_other_load_rejects_the_pass(self):
        r = validate.validate(doc(passes=[one_pass(util=40.0), one_pass()]))
        self.assertIn("R2", " ".join(r["runs"][0]["notes"]))
        self.assertTrue(metric(r, "decode")["status"].startswith("rejected: R6"))
        r = validate.validate(doc(passes=[one_pass(mem_growth=600.0), one_pass(), one_pass()]))
        self.assertEqual(metric(r, "decode")["status"], "accepted")

    def test_thermal_rejects_only_its_metric(self):
        r = validate.validate(doc(passes=[one_pass(reason="hw_thermal_slowdown", at="decode"), one_pass(),
                                          one_pass()]))
        self.assertEqual(metric(r, "decode")["status"], "accepted")  # two clean passes remain
        r = validate.validate(doc(passes=[one_pass(reason="hw_slowdown", at="decode"), one_pass()]))
        self.assertTrue(metric(r, "decode")["status"].startswith("rejected: R6"))
        self.assertEqual(metric(r, "prefill")["status"], "accepted")

    def test_power_cap_is_recorded_not_rejected(self):
        r = validate.validate(doc(passes=[one_pass(reason="sw_power_cap", at="prefill"), one_pass()]))
        self.assertEqual(metric(r, "prefill")["status"], "accepted")
        self.assertIn("power cap", " ".join(metric(r, "prefill")["notes"]))

    def test_caps_keep_but_never_confirm(self):
        r = validate.validate(doc(smi={"power.limit": 120.0}))
        self.assertTrue(r["capped"])
        self.assertEqual(metric(r, "decode")["verdict"], "not used: the card is capped")

    def test_repeats(self):
        r = validate.validate(doc(passes=[one_pass(decode=50.0), one_pass(decode=55.0)]))
        self.assertTrue(metric(r, "decode")["status"].startswith("rejected: R5"))
        r = validate.validate(doc(passes=[one_pass(spread=0.3), one_pass()]))
        self.assertTrue(metric(r, "decode")["status"].startswith("rejected: R5"))

    def test_spread_required(self):
        d = doc()
        d["runs"][0]["passes"][0]["measured"]["decode"].pop("min")
        self.assertTrue(metric(validate.validate(d), "decode")["status"].startswith("rejected: R5"))

    def test_depth(self):
        r = validate.validate(doc(passes=[one_pass(tokens=4000), one_pass(tokens=4000)]))
        self.assertTrue(metric(r, "decode")["status"].startswith("rejected: R7"))
        r = validate.validate(doc(passes=[one_pass(prompt=8192), one_pass(prompt=8192)]))
        self.assertTrue(metric(r, "prefill")["status"].startswith("rejected: R7"))

    def test_wddm_process_list_is_not_load(self):
        p = [one_pass(), one_pass()]
        for x in p:
            x["processes_before"] = {"listed": 20, "others": 19, "informative": False}
        self.assertEqual(metric(validate.validate(doc(passes=p)), "decode")["status"], "accepted")
        p[0]["processes_before"]["informative"] = True
        self.assertIn("R2", " ".join(validate.validate(doc(passes=p))["runs"][0]["notes"]))

    def test_unknown_reasons_are_never_a_pass_silently(self):
        d = doc()
        for p in d["runs"][0]["passes"]:
            p["telemetry"]["reason_fields"] = None
        r = validate.validate(d)
        self.assertIn("R3 unknown", " ".join(r["runs"][0]["notes"]))
        self.assertTrue(metric(r, "decode")["verdict"].startswith("not used: R3 unknown"))

    def test_incomplete_files_are_r0(self):
        for bad in ({"schema": "ninferno-user-run/1.0", "card": None, "runs": []},
                    {"schema": "ninferno-user-run/1.0", "card": {}, "runs": None},
                    {"schema": "ninferno-user-run/1.0", "card": {}, "runs": [None]}):
            self.assertTrue(validate.validate(bad)["file"].startswith("rejected: R0"))

    def test_unreported_choice_gives_no_verdict(self):
        d = doc()
        for p in d["runs"][0]["passes"]:
            p["served"]["choice"] = None
        r = validate.validate(d)
        self.assertTrue(metric(r, "decode")["verdict"].startswith("not used: R1 unknown"))

    def test_lengths_per_request(self):
        d = doc()
        for p in d["runs"][0]["passes"]:
            p["measured"]["decode"]["prompt_tokens_range"] = [7600, 8800]
        self.assertTrue(metric(validate.validate(d), "decode")["status"].startswith("rejected: R7"))

    def test_na_reasons_are_unknown(self):
        d = doc()
        for p in d["runs"][0]["passes"]:
            cols = p["telemetry"]["columns"]
            for k in list(cols):
                if k.startswith(P):
                    cols[k] = [None] * len(cols[k])
        self.assertTrue(metric(validate.validate(d), "decode")["verdict"].startswith("not used: R3 unknown"))

    def test_unknown_power_source(self):
        d = doc()
        d["host"]["on_battery"] = None
        self.assertTrue(metric(validate.validate(d), "decode")["verdict"].startswith("not used: the power"))

    def test_render(self):
        text = validate.render(validate.validate(copy.deepcopy(doc())))
        self.assertIn("decode: accepted", text)


class Privacy(unittest.TestCase):
    def test_scrub_paths(self):
        for text in ("/opt/private/model.gguf", "\\\\server\\share\\x.json", "D:\\data\\x"):
            self.assertNotIn("private", measure.scrub(text))
            self.assertIn("<path>", measure.scrub(text))
        for text in ("12 tok/s", "org/repo", "https://huggingface.co/org/repo"):
            self.assertEqual(measure.scrub(text), text)
        self.assertEqual(measure.scrub_doc({"artifact": "/x/y", "a": ["/home/u/f"]}), {"a": ["<path>"]})
        for text in ("/srv/users/Alice Smith/model.gguf", "D:/data/alice/model.gguf", "//server/share/Alice/f",
                     "/Alice", "/srv/users/O'Alice/model.gguf", '/srv/"Alice"/x'):
            self.assertNotIn("lice", measure.scrub(text))


class Gpu(unittest.TestCase):
    def test_help_parse_takes_every_alias(self):
        text = ('List of valid properties to query for the switch "--query-gpu":\n\n'
                '"name" or "gpu_name"\nOfficial product name.\n\n'
                '"clocks_event_reasons.active" or "clocks_throttle_reasons.active"\nBitmask.\n'
                '"power.limit"\nThe software power limit.\n')
        known = gpu.parse_help(text)
        self.assertTrue({"name", "gpu_name", "clocks_event_reasons.active",
                         "clocks_throttle_reasons.active", "power.limit"} <= known)
        self.assertNotIn("--query-gpu", known)  # the heading line does not start with a quote

    def test_values(self):
        self.assertIsNone(gpu._value(" [N/A]"))
        self.assertIsNone(gpu._value("[Requested functionality has been deprecated]"))
        self.assertEqual(gpu._value(" 285.00"), 285.0)
        self.assertEqual(gpu._value("WDDM"), "WDDM")


# --- Speculation (S1-S6) and hard50 (H1-H3) -----------------------------------------------------

SHA = "ab" * 32
SECTIONS = ("math", "code", "prose", "thinking")
SETTINGS = "max context 4096, prefill chunk 1024, KV bf16, greedy, wait spin"


def measurements(t=3.5, plain_s=0.02, round_cost=1.5, configs=("mtp:3",), plain_scale=None):
    """A tune's measurements: plain decode at `plain_s` seconds a token in every section, each
    configuration `t` tokens a round at `round_cost` plain steps a round."""
    out = []
    for i, f in enumerate(SECTIONS):
        s = plain_s * ((plain_scale or {}).get(f, 1.0))
        out.append({"config": "plain", "family": f, "prompts": 8, "tokens": 1000, "seconds": 1000 * s,
                    "rounds": 0, "accepted": 0, "tok_s": 1 / s})
        for c in configs:
            rounds = 300
            out.append({"config": c, "family": f, "prompts": 8, "tokens": int(rounds * t),
                        "rounds": rounds, "accepted": int(rounds * (t - 1)),
                        "seconds": rounds * round_cost * s})
    return out


def lab_pair(gpu="ad104:60sm", t=3.5, drafters=None, settings=SETTINGS, engine="ninferno-spec-1"):
    return {"target": SHA, "target_repo": "org/m@" + "1" * 40 + " Q4_K_M", "engine": "qwen3_5",
            "generation": "sm89", "drafters": drafters or {},
            "tune": {"key": {"gpu": gpu, "engine": engine}, "settings": settings,
                     "measurements": measurements(t=t)}}


def spec_run(t=3.5, round_cost=1.5, predicted=1.5, source="registry pair org/m (2 lab card(s)); rounds "
             "priced from the weights' bytes and this card's rates (ada, eta 0.148)", **extra):
    run = {"id": "q9", "repo": "org/m", "quant": "Q4_K_M", "mode": "spec",
           "plan": {"speculation": {"basis": "measured", "source": source,
                                    "round_cost": {"mtp:3": predicted}}},
           "manifest": {"target": {"engine": "qwen3_5"},
                        "drafters": [{"name": "mtp", "kind": "mtp", "status": "embedded"}]},
           "target": {"revision": "1" * 40, "weights": [{"sha256": SHA, "bytes": 1}]},
           "profile": {"key": {"gpu": "ad106:34sm", "engine": "ninferno-spec-1"}, "settings": SETTINGS,
                       "drafters": [{"name": "mtp", "status": "accepted"}],
                       "measurements": measurements(t=t, round_cost=round_cost)},
           "publish": {"measured_on": "NVIDIA GeForce RTX 4060 Ti user-kit"},
           "processes_before": {"listed": 1, "others": 0, "informative": True},
           "telemetry": {"t": [0.0, 1.0], "reason_fields": P, "marks": [{"name": "spec-tune", "start": 0, "end": 2}],
                         "columns": {f"{P}.{r}": ["Not Active"] * 2 for r in validate.REJECT_REASONS}}}
    run.update(extra)
    return run


def spec_doc(run, pairs=None):
    d = doc(passes=[])
    d["runs"] = [run]
    d["card"]["smi"]["pci.device_id"] = "0x280510DE"
    return d, {"pairs": pairs if pairs is not None else [lab_pair(), lab_pair("ad102:128sm", t=3.45)]}


def judge(run, pairs=None):
    d, registry = spec_doc(run, pairs)
    out = {"id": run["id"], "mode": "spec", "metrics": [], "notes": []}
    validate.judge_spec(d, run, out, [], registry, validate.TOLERANCE)
    return out


def row(out, name):
    return next(m for m in out["metrics"] if m["metric"] == name)


class Spec(unittest.TestCase):
    def test_clean_tune_confirms_both_and_is_a_candidate(self):
        out = judge(spec_run())
        a, r = row(out, "acceptance mtp:3"), row(out, "round cost mtp:3")
        self.assertTrue(a["verdict"].startswith("confirms the registry pairs' acceptance"), a)
        self.assertTrue(r["verdict"].startswith("confirms the facts' R(k)"), r)
        self.assertAlmostEqual(r["measured"], 1.5, places=3)
        self.assertEqual(out["candidate"], "yes")

    def test_round_cost_mirrors_the_engine(self):
        # Per section a round's seconds over plain's seconds a token, the mean over the sections
        # (tests/test_spec_pairs.cpp holds the C++ to the same numbers).
        m = [{"config": "plain", "family": "math", "tokens": 100, "seconds": 10.0},
             {"config": "plain", "family": "code", "tokens": 100, "seconds": 5.0},
             {"config": "mtp:3", "family": "math", "rounds": 20, "seconds": 4.0},
             {"config": "mtp:3", "family": "code", "rounds": 20, "seconds": 3.0},
             {"config": "mtp:3", "family": "prose", "rounds": 20, "seconds": 3.0}]
        self.assertEqual(validate.round_costs(m), {"mtp:3": 2.5})

    def test_acceptance_off_contradicts_and_blocks_the_candidate(self):
        out = judge(spec_run(t=2.9))
        self.assertTrue(row(out, "acceptance mtp:3")["verdict"].startswith("contradicts"))
        self.assertEqual(out["candidate"], "no: an acceptance contradicts the lab pairs")

    def test_round_cost_off_contradicts_the_facts(self):
        out = judge(spec_run(round_cost=1.8))
        self.assertTrue(row(out, "round cost mtp:3")["verdict"].startswith("contradicts the facts' R(k)"))
        self.assertEqual(out["candidate"], "yes")  # speed is evidence about the model, not the entry

    def test_prediction_kinds_are_said(self):
        own = judge(spec_run(source="registry pair org/m (2 lab card(s))"),
                    pairs=[lab_pair(), lab_pair("ad106:34sm")])
        self.assertIn("a lab measurement, not the facts", row(own, "round cost mtp:3")["verdict"])
        raised = judge(spec_run(source="registry pair org/m; rounds priced from the weights' bytes and this "
                                       "card's rates (ada, eta 0.148); conservative where the lab card "
                                       "measured rounds dearer than their bytes: mtp:3 +12 %"))
        self.assertIn("raised to a lab card's measurement", row(raised, "round cost mtp:3")["verdict"])
        other = judge(spec_run(source="registry sibling org/m Q8_0 (2 lab card(s))"))
        self.assertIn("registry sibling", row(other, "round cost mtp:3")["verdict"])

    def test_throttle_rejects_round_cost_not_acceptance(self):
        run = spec_run()
        run["telemetry"]["columns"][f"{P}.hw_thermal_slowdown"] = ["Active", "Not Active"]
        out = judge(run)
        self.assertTrue(row(out, "round cost mtp:3")["status"].startswith("rejected: S3"))
        self.assertTrue(row(out, "acceptance mtp:3")["verdict"].startswith("confirms"))

    def test_other_process_and_unsteady_plain(self):
        out = judge(spec_run(processes_after={"listed": 2, "others": 1, "informative": True}))
        self.assertTrue(row(out, "round cost mtp:3")["status"].startswith("rejected: S3"))
        run = spec_run()
        run["profile"]["measurements"] = measurements(plain_scale={"code": 1.08})
        self.assertTrue(row(judge(run), "round cost mtp:3")["status"].startswith("rejected: S4"))

    def test_unknown_reasons_keep_round_cost_without_verdict(self):
        run = spec_run()
        run["telemetry"]["columns"] = {f"{P}.{r}": [None, None] for r in validate.REJECT_REASONS}
        r = row(judge(run), "round cost mtp:3")
        self.assertEqual(r["status"], "accepted")
        self.assertTrue(r["verdict"].startswith("not used: S3 unknown"))

    def test_no_reference_cases(self):
        # Another target, another engine key, other settings, one lab card, a +head configuration.
        self.assertIn("no lab pair", " ".join(judge(spec_run(), pairs=[]).get("notes")))
        out = judge(spec_run(), pairs=[lab_pair(engine="ninferno-spec-2"), lab_pair("ad102:128sm", engine="x")])
        self.assertTrue(row(out, "acceptance mtp:3")["verdict"].startswith("no reference"))
        out = judge(spec_run(), pairs=[lab_pair(settings="max context 8192, greedy"), lab_pair("ad102:128sm", settings="x")])
        self.assertTrue(row(out, "acceptance mtp:3")["verdict"].startswith("no reference"))
        self.assertTrue(out["candidate"].startswith("no: the tune's settings"))
        self.assertTrue(row(judge(spec_run(), pairs=[lab_pair()]), "acceptance mtp:3")["verdict"]
                        .startswith("no reference: fewer than 2 lab cards"))
        run = spec_run()
        run["profile"]["measurements"] = measurements(configs=("mtp:3+head",))
        self.assertIn("proposal head", row(judge(run), "acceptance mtp:3+head")["verdict"])

    def test_lab_pairs_of_other_menus_are_left_out(self):
        # The registry's pairs of one target need not share a menu: one without the label is no
        # reference for it (and no crash).
        third = lab_pair("gb203:70sm")
        third["tune"]["measurements"] = measurements(configs=("dflash:7",))
        out = judge(spec_run(), pairs=[lab_pair(), lab_pair("ad102:128sm", t=3.45), third])
        self.assertIn("lab cards 2", " ".join(row(out, "acceptance mtp:3")["notes"]))
        lone = judge(spec_run(), pairs=[lab_pair(), third])
        self.assertIn("fewer than 2 lab cards", row(lone, "acceptance mtp:3")["verdict"])
        none = judge(spec_run(), pairs=[third, third])
        self.assertIn("the lab measured no mtp:3", row(none, "acceptance mtp:3")["verdict"])

    def test_a_configuration_without_rounds_is_no_candidate(self):
        run = spec_run()
        for m in run["profile"]["measurements"]:
            if m["config"] == "mtp:3" and m["family"] == "prose":
                m["rounds"] = m["accepted"] = 0
        out = judge(run)
        self.assertTrue(row(out, "acceptance mtp:3")["status"].startswith("rejected: S1"))
        self.assertEqual(out["candidate"], "no: a configuration without a round in a section")

    def test_registry_is_a_parameter(self):
        d, registry = spec_doc(spec_run())
        r = validate.validate(d, registry=registry)
        self.assertEqual(r["runs"][0]["candidate"], "yes")
        self.assertTrue(r["runs"][0]["metrics"][0]["verdict"].startswith("confirms"))

    def test_wait_mode_is_not_an_acceptance_setting(self):
        out = judge(spec_run(), pairs=[lab_pair(settings=SETTINGS.replace("spin", "block")),
                                       lab_pair("ad102:128sm", settings=SETTINGS.replace("spin", "block"))])
        self.assertTrue(row(out, "acceptance mtp:3")["verdict"].startswith("confirms"))

    def test_separate_drafter_must_be_the_pairs(self):
        run = spec_run()
        run["manifest"]["drafters"].append({"name": "dflash", "kind": "dflash", "status": "embedded",
                                            "repo": "z/D", "revision": "r1"})
        run["profile"]["measurements"] = measurements(configs=("dflash:7",))
        run["plan"]["speculation"]["round_cost"] = {"dflash:7": 1.5}
        good = [lab_pair(drafters={"dflash": "z/D@r1"}), lab_pair("ad102:128sm", drafters={"dflash": "z/D@r1"})]
        for p in good:
            p["tune"]["measurements"] = measurements(configs=("dflash:7",))
        self.assertTrue(row(judge(run, good), "acceptance dflash:7")["verdict"].startswith("confirms"))
        for p in good:
            p["drafters"] = {"dflash": "z/D@r2"}
        self.assertTrue(row(judge(run, good), "acceptance dflash:7")["verdict"].startswith("no reference"))

    def test_gemma_assistant_is_the_mtp_labels_drafter(self):
        run = spec_run()
        run["manifest"]["drafters"] = [{"name": "assistant", "kind": "gemma4_assistant", "status": "attached",
                                        "repo": "g/A", "revision": "r1"}]
        pairs = [lab_pair(drafters={"gemma4_assistant": "g/A@r1"}),
                 lab_pair("ad102:128sm", drafters={"gemma4_assistant": "g/A@r1"})]
        self.assertTrue(row(judge(run, pairs), "acceptance mtp:3")["verdict"].startswith("confirms"))
        for p in pairs:
            p["drafters"] = {"gemma4_assistant": "g/A@r2"}
        self.assertIn("another drafter", row(judge(run, pairs), "acceptance mtp:3")["verdict"])
        run["manifest"]["drafters"][0]["repo"] = "other"
        self.assertTrue(judge(run)["candidate"].startswith("no: a measured drafter the registry does not declare"))

    def test_partial_or_unidentified_tunes_are_no_candidates(self):
        run = spec_run()
        run["profile"]["skipped"] = ["dflash:7"]
        self.assertEqual(judge(run)["candidate"], "no: partial tune")
        run = spec_run()
        run["profile"]["drafters"].append({"name": "dflash", "status": "untested"})
        self.assertEqual(judge(run)["candidate"], "no: partial tune")
        self.assertTrue(judge(spec_run(target=None))["candidate"].startswith("no: the target's identity"))

    def test_failed_tune_through_validate(self):
        d, _ = spec_doc({"id": "q9", "mode": "spec", "error": "spec-tune failed"})
        r = validate.validate(d)
        self.assertEqual(r["runs"][0]["candidate"], "no: the run failed")
        d, _ = spec_doc({"id": "q9", "mode": "spec"})
        self.assertIn("S1", validate.render(validate.validate(d)))


REF = {"set_sha256": "s" * 64, "models": {"org/m Q4_K_M": [
    {"card": f"c{i}", "correct": c, "wrong": w} for i, (c, w) in enumerate(
        [(27, ["a", "b"]), (26, ["a", "c"]), (27, ["b", "d"]), (25, ["a", "b", "c"])])]}}


def hard(correct=27, wrong=("a", "b"), **extra):
    run = {"id": "q9", "repo": "org/m", "quant": "Q4_K_M", "mode": "hard50", "set_sha256": "s" * 64,
           "hard50": {"questions": 50, "correct": correct, "errors": 0, "wrong": list(wrong)}}
    run.update(extra)
    out = {"metrics": [], "notes": []}
    validate.judge_hard50(run, out, REF)
    return out["metrics"][0]


class Hard50(unittest.TestCase):
    def test_within_the_lab(self):
        self.assertTrue(hard()["verdict"].startswith("consistent with the lab's hard50"))

    def test_count_far_below(self):
        # lab 25..27, median 26.5: p 0.53, sd 3.53, below the lab under 17.9
        self.assertTrue(hard(correct=17)["verdict"].startswith("below"))
        self.assertTrue(hard(correct=18)["verdict"].startswith("consistent"))

    def test_items_every_lab_card_got_right(self):
        # A lab card misses at most one question the others all got right (the third: d): m = 1,
        # so more than 2 m + 2 = 4 such questions here is below the lab.
        self.assertTrue(hard(wrong=("a", "x1", "x2", "x3", "x4"))["verdict"].startswith("consistent"))
        self.assertTrue(hard(wrong=("a", "x1", "x2", "x3", "x4", "x5"))["verdict"].startswith("below"))

    def test_rejections_and_no_reference(self):
        self.assertTrue(hard(hard50={"questions": 50, "correct": 20, "errors": 2})["status"].startswith("rejected: H1"))
        self.assertTrue(hard(hard50={"questions": 49, "correct": 20, "errors": 0})["status"].startswith("rejected: H1"))
        self.assertTrue(hard(set_sha256="t" * 64)["verdict"].startswith("no reference: H2"))
        self.assertTrue(hard(repo="org/other")["verdict"].startswith("no reference"))

    def test_reference_file_is_well_formed(self):
        ref = validate.load_json(validate.HARD_REFERENCE)
        self.assertEqual(ref["set_sha256"], measure.sha256_of(measure.HARD_SET))
        for key, runs in ref["models"].items():
            self.assertEqual(len(key.split(" ")), 2, key)
            for r in runs:
                self.assertEqual(r["correct"], 50 - len(r["wrong"]), (key, r["card"]))


class Kit(unittest.TestCase):
    def test_lab_configs_are_the_pairs_menu(self):
        import json
        registry = json.loads(measure.REGISTRY.read_text(encoding="utf-8"))
        pair = next(p for p in registry["pairs"] if p["target_repo"].endswith(" Q4_K_M")
                    and "Qwen3.5-9B-MTP-GGUF@" in p["target_repo"])
        labels = measure.lab_configs({"repo": "unsloth/Qwen3.5-9B-MTP-GGUF", "quant": "Q4_K_M"})
        self.assertTrue({m["config"] for m in pair["tune"]["measurements"]} - {"plain"} <= set(labels))
        self.assertEqual(len(labels), len(set(labels)))
        newest = max((p for p in registry["pairs"] if p["target_repo"] == pair["target_repo"]),
                     key=lambda p: p["tune"]["created"])
        first = [m["config"] for m in newest["tune"]["measurements"] if m["config"] != "plain"]
        self.assertEqual(labels[:len(set(first))], list(dict.fromkeys(first)))
        self.assertEqual(measure.lab_configs({"repo": "nobody/none", "quant": "Q4_K_M"}), [])

    def test_target_identity_hashes_the_snapshot_file(self):
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            rev = "c" * 40
            model = Path(root) / "models--org--m"
            (model / "ninferno" / rev).mkdir(parents=True)
            (model / "snapshots" / rev).mkdir(parents=True)
            (model / "snapshots" / rev / "M-Q4_K_M.gguf").write_bytes(b"weights")
            (model / "snapshots" / rev / "M-Q8_0.gguf").write_bytes(b"other")
            (model / "snapshots" / rev / "mmproj-M-Q4_K_M.gguf").write_bytes(b"vision")
            art = model / "ninferno" / rev / "M-Q4_K_M.qwen3_5_bf16.ninferno"
            got = measure.target_identity(str(art))
            self.assertEqual(got["revision"], rev)
            self.assertEqual([w["sha256"] for w in got["weights"]], [hashlib.sha256(b"weights").hexdigest()])
            self.assertIsNone(measure.target_identity(str(Path(root) / "x.ninferno")))
            # A split model: every part, by the artifact named after the model or its first part.
            for i in (1, 2):
                (model / "snapshots" / rev / f"B-Q4_K_M-0000{i}-of-00002.gguf").write_bytes(b"p%d" % i)
            for name in ("B-Q4_K_M.qwen3_5_bf16.ninferno", "B-Q4_K_M-00001-of-00002.qwen3_5_bf16.ninferno"):
                got = measure.target_identity(str(model / "ninferno" / rev / name))
                self.assertEqual(len(got["weights"]), 2, name)

if __name__ == "__main__":
    unittest.main()
