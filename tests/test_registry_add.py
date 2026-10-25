"""tools/spec/registry_add.py: the "*" fallback it writes and the load table warnings (no GPU).

    python3 tests/test_registry_add.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "spec"))
import registry_add  # noqa: E402


def choice(config: str, drafter: str = "assistant") -> dict:
    return {"config": config, "drafter": drafter, "floor_tokens_per_round": 1.5}


def published(default: str, portable: str | None = None, **extra) -> dict:
    out = {"gpu_class": "gb203", "gpu": "RTX 5080", "applies_to": ["org/model"], "measured_on": "t",
           "default": choice(default), "non_thinking": choice(default), "thinking": choice(default)}
    if portable:
        out["portable"] = {"default": choice(portable), "non_thinking": choice(portable),
                           "thinking": choice(portable)}
    out.update(extra)
    return out


STAR_LOAD = [{"lanes": 2, "depth": 3}, {"lanes": 5, "depth": 0}]


def registry(*profiles: dict) -> dict:
    return {"models": [{"id": "m", "drafters": [{"name": "assistant"}], "profiles": list(profiles)}]}


class RegistryAdd(unittest.TestCase):
    def run_main(self, reg: dict, pub: dict, *flags: str) -> tuple[dict, str]:
        with tempfile.TemporaryDirectory() as tmp:
            reg_path, pub_path = Path(tmp) / "registry.json", Path(tmp) / "pub.json"
            reg_path.write_text(json.dumps(reg), encoding="utf-8")
            pub_path.write_text(json.dumps(pub), encoding="utf-8")
            err = io.StringIO()
            argv = sys.argv
            sys.argv = ["registry_add.py", "m", str(pub_path), "--registry", str(reg_path), *flags]
            try:
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    registry_add.main()
            finally:
                sys.argv = argv
            return json.loads(reg_path.read_text(encoding="utf-8")), err.getvalue()

    @staticmethod
    def star(reg: dict) -> dict:
        return next(p for p in reg["models"][0]["profiles"] if p["gpu_class"] == "*")

    def test_fallback_takes_portable_choices_and_load_marks(self):
        star = registry_add.fallback_of(published("mtp:6", "mtp:3", load=STAR_LOAD, load_partial=True,
                                                  load_measured_on="cells"))
        self.assertEqual(star["default"]["config"], "mtp:3")
        self.assertEqual(star["load_measured_on"], "fallback for unmeasured GPUs: RTX 5080 (gb203), cells")
        self.assertEqual(star["thinking"]["config"], "mtp:3")
        self.assertEqual(star["load"], STAR_LOAD)
        self.assertTrue(star["load_partial"])
        self.assertNotIn("served", star)

    def test_retune_keeps_the_profile_in_its_place(self):
        first = published("mtp:6")
        other = dict(published("mtp:6"), gpu_class="ad102", applies_to=["org/other"])
        reg, _ = self.run_main(registry(first, other), published("mtp:4"))
        profiles = reg["models"][0]["profiles"]
        self.assertEqual([p["gpu_class"] for p in profiles], ["gb203", "ad102"])
        self.assertEqual(profiles[0]["default"]["config"], "mtp:4")

    def test_fallback_keeps_the_old_star_load_of_the_same_kind(self):
        old_star = {"gpu_class": "*", "applies_to": [], "default": choice("mtp:7"), "load": STAR_LOAD,
                    "load_measured_on": "5090", "load_partial": True}
        reg, err = self.run_main(registry(old_star), published("mtp:6", "mtp:3"), "--fallback")
        star = self.star(reg)
        self.assertEqual(star["default"]["config"], "mtp:3")
        self.assertEqual(star["load"], STAR_LOAD)
        self.assertEqual(star["load_measured_on"], "5090")
        self.assertTrue(star["load_partial"])
        self.assertIn("borrows the *", err)

    def test_fallback_drops_an_old_star_load_of_another_kind(self):
        old_star = {"gpu_class": "*", "applies_to": [], "default": choice("dflash:7", "dflash"),
                    "load": STAR_LOAD}
        reg = registry(old_star)
        reg["models"][0]["drafters"].append({"name": "dflash"})
        reg, err = self.run_main(reg, published("mtp:6", "mtp:3"), "--fallback")
        self.assertNotIn("load", self.star(reg))
        self.assertIn("dropped", err)
        self.assertIn("runs unbounded", err)

    def test_warning_names_a_same_class_lender_first(self):
        other = dict(published("mtp:5"), applies_to=["org/other"], load=[{"lanes": 2, "depth": 2}])
        star = {"gpu_class": "*", "applies_to": [], "default": choice("mtp:7"), "load": STAR_LOAD}
        _, err = self.run_main(registry(other, star), published("mtp:6"))
        self.assertIn("borrows the gb203", err)
        self.assertIn("'depth': 2", err)

    def test_undeclared_drafter_in_portable_choices_is_refused(self):
        pub = published("mtp:6", "mtp:3")
        pub["portable"]["thinking"] = choice("dflash:7", "dflash")
        with self.assertRaises(SystemExit):
            self.run_main(registry(), pub, "--fallback")

    def test_load_table_draft_widths(self):
        registry_add.check_load([{"lanes": 2, "depth": 3, "draft_width": "block"},
                                 {"lanes": 4, "depth": 2, "draft_width": "drafts"}, {"lanes": 6, "depth": 0}])
        for bad in ({"lanes": 2, "depth": 0, "draft_width": "block"},
                    {"lanes": 2, "depth": 3, "draft_width": "wide"}):
            with self.assertRaises(SystemExit):
                registry_add.check_load([bad])

    def test_retune_keeps_the_load_marks(self):
        old = dict(published("mtp:6"), load=STAR_LOAD, load_partial=True)
        reg, _ = self.run_main(registry(old), published("mtp:5"))
        (kept,) = [p for p in reg["models"][0]["profiles"] if p["gpu_class"] == "gb203"]
        self.assertEqual(kept["default"]["config"], "mtp:5")
        self.assertEqual(kept["load"], STAR_LOAD)
        self.assertTrue(kept["load_partial"])

    def test_retune_drops_a_load_table_of_another_kind(self):
        old = dict(published("mtp:6"), load=STAR_LOAD)
        reg = registry(old)
        reg["models"][0]["drafters"].append({"name": "dflash2"})
        pub = published("dflash2:7")
        for key in ("default", "non_thinking", "thinking"):
            pub[key]["drafter"] = "dflash2"
        reg, err = self.run_main(reg, pub, "--fallback")
        (new,) = [p for p in reg["models"][0]["profiles"] if p["gpu_class"] == "gb203"]
        self.assertNotIn("load", new)
        self.assertNotIn("load", self.star(reg))
        self.assertIn("dropped", err)

    def test_variant_profile_requires_its_drafter_beside_the_published_one(self):
        reg = registry(published("mtp:3"))
        reg["models"][0]["drafters"][0]["variants"] = ["q8"]
        pub = published("mtp:8")
        for key in ("default", "non_thinking", "thinking"):
            pub[key]["drafter"] = "assistant.q8"
        reg, _ = self.run_main(reg, pub)
        profiles = reg["models"][0]["profiles"]
        self.assertEqual([(p.get("requires"), p["default"]["config"]) for p in profiles],
                         [(None, "mtp:3"), (["assistant.q8"], "mtp:8")])

    def test_undeclared_variant_refused(self):
        pub = published("mtp:8")
        pub["default"]["drafter"] = "assistant.nvfp4"
        with self.assertRaises(SystemExit):
            self.run_main(registry(), pub)


class Stamps(unittest.TestCase):
    def test_a_changed_profile_is_dated_and_the_version_bumped(self):
        reg = {"schema": 1, "version": 3, "published": "2026-01-01",
               "models": [{"id": "m", "profiles": [published("mtp:6"), published("mtp:4", gpu_class="gb202")]}]}
        registry_add.stamp_registry(reg, None, "2026-02-02")
        self.assertEqual(reg["version"], 3)
        kept = json.loads(json.dumps(reg))
        reg["models"][0]["profiles"][1]["default"]["config"] = "mtp:5"
        self.assertTrue(registry_add.stamp_registry(reg, kept, "2026-03-03"))
        first, second = reg["models"][0]["profiles"]
        self.assertEqual((reg["version"], reg["published"]), (4, "2026-03-03"))
        self.assertEqual(first["measured_at"], "2026-02-02")
        self.assertEqual(first["revision"], kept["models"][0]["profiles"][0]["revision"])
        self.assertEqual(second["measured_at"], "2026-03-03")
        self.assertNotEqual(second["revision"], kept["models"][0]["profiles"][1]["revision"])
        self.assertEqual(list(second)[-1], "revision")
        self.assertEqual(list(second)[list(second).index("measured_on") + 1], "measured_at")
        again = json.loads(json.dumps(reg))
        self.assertFalse(registry_add.stamp_registry(reg, again, "2026-04-04"))
        self.assertEqual(reg, again)

    def test_the_compiled_in_registry_is_stamped(self):
        path = Path(__file__).resolve().parents[1] / "src/spec/data/drafter_registry.json"
        reg = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(reg.get("version", 0), 1)
        for m in reg["models"]:
            for p in m["profiles"]:
                self.assertEqual(p.get("revision"), registry_add.revision_of(p),
                                 f"{m['id']} {p['gpu_class']}: edited without the tool "
                                 "(python tools/spec/registry_add.py --restamp)")
                self.assertRegex(p.get("measured_at", ""), r"^\d{4}-\d{2}-\d{2}$")


class PairTest(unittest.TestCase):
    """--pair: a lab measurement of one target and its drafters as the registry's pair."""

    tune = {"key": {"engine": "ninferno-spec-1", "gpu": "ad104:60sm"},
            "measurements": [{"config": "plain"}, {"config": "dflash:7"}]}
    manifest = {"target": {"engine": "qwen3_5"},
                "drafters": [{"kind": "mtp", "repo": None, "status": "embedded"},
                             {"kind": "dflash", "repo": "z-lab/Qwen3.5-9B-DFlash",
                              "revision": "5fc3b3d4", "status": "embedded"}]}

    def test_pair_of_a_tune(self):
        pair = registry_add.pair_of(self.tune, self.manifest, "org/m@abc", "UD-Q4_K_XL",
                                    ["B" * 64, "a" * 64])
        self.assertEqual(pair["target"], "a" * 64 + "," + "b" * 64)
        self.assertEqual(pair["generation"], "sm89")
        self.assertEqual(pair["target_bits"], 4)
        self.assertEqual(pair["engine"], "qwen3_5")
        self.assertEqual(pair["drafters"], {"dflash": "z-lab/Qwen3.5-9B-DFlash@5fc3b3d4"})
        self.assertEqual(pair["target_repo"], "org/m@abc UD-Q4_K_XL")

    def test_labels_and_generations(self):
        self.assertEqual(registry_add.label_bits("TQ1_0"), 1)
        self.assertEqual(registry_add.label_bits("IQ2_M"), 2)
        self.assertEqual(registry_add.label_bits(""), 16)
        self.assertEqual(registry_add.generation_of("gb203:70sm"), "sm120")

    def test_a_tune_without_a_drafter_is_refused(self):
        bare = {"key": self.tune["key"], "measurements": [{"config": "plain"}]}
        with self.assertRaises(SystemExit):
            registry_add.pair_of(bare, self.manifest, "org/m@abc", "Q4_K_M", ["a" * 64])

    def test_bad_weights_are_refused(self):
        with self.assertRaises(SystemExit):
            registry_add.pair_of(self.tune, self.manifest, "org/m@abc", "Q4_K_M", ["abc"])

    def test_a_pair_replaces_its_own(self):
        pair = registry_add.pair_of(self.tune, self.manifest, "org/m@abc", "Q4_K_M", ["a" * 64])
        again = json.loads(json.dumps(pair))
        again["target_repo"] = "org/m@def Q4_K_M"
        other = json.loads(json.dumps(pair))
        other["tune"]["key"]["gpu"] = "ad102:128sm"
        pairs = registry_add.put_pair(registry_add.put_pair([pair], other), again)
        self.assertEqual(len(pairs), 2)
        self.assertIn("org/m@def Q4_K_M", [p["target_repo"] for p in pairs])


if __name__ == "__main__":
    unittest.main()
