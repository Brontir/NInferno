"""The fusion sites' manifest table and regret format against the registry, no GPU (#617 step H).

  - src/ops/common/fusion_site_table.h is the registry's generated form and is current;
  - every registry site has a row there, hence a manifest line (the emitter reads the table);
  - tools/bench/site_regret.py refuses a site the registry does not have, a decided branch that is
    not a candidate, a bad time, and fills the qk_norm_rope rows from the rope crossover CSV.
"""
import copy
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import site_regret  # noqa: E402

REGISTRY = ROOT / "tools" / "audit" / "fusion_sites.json"
TABLE = ROOT / "src" / "ops" / "common" / "fusion_site_table.h"
GEN = ROOT / "tools" / "audit" / "gen_fusion_site_table.py"
MANIFEST = ROOT / "src" / "ops" / "common" / "fusion_site_manifest.h"


class SiteManifest(unittest.TestCase):
    def test_table_is_current(self):
        done = subprocess.run([sys.executable, "-B", str(GEN), "--check"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_every_registry_site_has_a_row(self):
        ids = [s["id"] for s in json.loads(REGISTRY.read_text(encoding="utf-8"))["sites"]]
        rows = re.findall(r'^    \{R"~~\(([a-z0-9_]+)\)~~"', TABLE.read_text(encoding="utf-8"), re.M)
        self.assertEqual(rows, ids, "the table lists the registry's sites in its order")

    def test_evaluated_sites_exist_in_the_registry(self):
        ids = {s["id"] for s in json.loads(REGISTRY.read_text(encoding="utf-8"))["sites"]}
        evaluated = set(re.findall(r'id == "([a-z0-9_]+)"', MANIFEST.read_text(encoding="utf-8")))
        self.assertTrue(evaluated, "the emitter evaluates some site")
        self.assertFalse(evaluated - ids, f"emitter names sites the registry lacks: {evaluated - ids}")


def doc():
    return {"schema": site_regret.SCHEMA, "board": "b", "run": "r", "protocol": "p",
            "rows": [{"site": "qk_norm_rope", "case": "32+2", "T": 128, "decided": "fused",
                      "candidates": [{"branch": "fused", "median_us": 4.0},
                                     {"branch": "decomposed", "median_us": 5.0}]}]}


class SiteRegret(unittest.TestCase):
    ids = site_regret.registry_ids()

    def test_valid_document_and_regret(self):
        d = doc()
        site_regret.validate(d, self.ids)
        self.assertEqual(site_regret.regret_of(d["rows"][0]), 0.0)
        d["rows"][0]["decided"] = "decomposed"
        self.assertAlmostEqual(site_regret.regret_of(d["rows"][0]), 0.25)
        self.assertEqual(site_regret.summarize(d)["qk_norm_rope"]["rows"], 1)

    def test_refusals(self):
        def refuse(mutate):
            d = copy.deepcopy(doc())
            mutate(d)
            with self.assertRaises(ValueError):
                site_regret.validate(d, self.ids)
        refuse(lambda d: d["rows"][0].update(site="not_a_site"))
        refuse(lambda d: d["rows"][0].update(decided="elsewhere"))
        refuse(lambda d: d["rows"][0]["candidates"][0].update(median_us=0))
        refuse(lambda d: d["rows"][0]["candidates"][0].update(median_us=float("nan")))
        refuse(lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])))
        refuse(lambda d: d["rows"][0]["candidates"].pop())

    def test_rope_crossover_csv(self):
        text = ("# device=x\ngeometry,T,fused_us,split_us,fused_minus_split_us\n"
                "32+2,128,4.0,5.0,-1.0\n32+2,512,9.0,8.0,1.0\n# 32+2: crossover_T=300\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text(text, encoding="utf-8")
            d = site_regret.from_rope_crossover(path, "b", "r", 256)
        site_regret.validate(d, self.ids)
        self.assertEqual([r["decided"] for r in d["rows"]], ["fused", "decomposed"])
        self.assertAlmostEqual(site_regret.regret_of(d["rows"][1]), 0.0)
        self.assertAlmostEqual(site_regret.regret_of(d["rows"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
