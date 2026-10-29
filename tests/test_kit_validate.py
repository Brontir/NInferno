"""tools/kit/validate.py: the user measurement kit's stated rejection rules and verdicts, no GPU
(docs/maintainer/user-measurement-kit.md)."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "kit"))
import gpu  # noqa: E402
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
    return {"schema": "ninferno-user-run/1.0", "host": {},
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

    def test_schema(self):
        bad = doc()
        bad["schema"] = "ninferno-user-run/2.0"
        self.assertTrue(validate.validate(bad)["file"].startswith("rejected: R0"))
        bad["schema"] = "ninferno-user-run/10.0"
        self.assertTrue(validate.validate(bad)["file"].startswith("rejected: R0"))
        bad["schema"] = "ninferno-user-run/1.3"
        self.assertEqual(validate.validate(bad)["file"], "accepted")

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

    def test_render(self):
        text = validate.render(validate.validate(copy.deepcopy(doc())))
        self.assertIn("decode: accepted", text)


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


if __name__ == "__main__":
    unittest.main()
