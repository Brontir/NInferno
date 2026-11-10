import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import mmq_transfer_audit as audit


def card(generation, ridge, bands, cc=89, sms=60, dram=500.0, l2=48.0):
    return {"name": "x", "generation": generation, "ridge_t": ridge,
            "board": {"cc": cc, "sms": sms, "dram_gbps": dram, "l2_mib": l2},
            "entries": {("q4k", "a16", "store", 1, 1): {"bands": bands, "prefill_bands": bands}}}


class TableAudit(unittest.TestCase):
    def test_prefill_widths_read_the_prefill_fields(self):
        entry = {"bands": [[1, 99999, "ggml_mmq"]], "prefill_bands": [[17, 64, "rmma"]],
                 "prefill": "staged"}
        self.assertEqual(audit.route_at(entry, 16), "ggml_mmq")
        self.assertEqual(audit.route_at(entry, 17), "rmma")
        self.assertEqual(audit.route_at(entry, 65), "staged")

    def test_regimes_and_disagreement(self):
        a = card("g", 100, [[1, 4, "vec"], [5, 40, "ggml_mmq"], [41, 99999, "rmma"]])
        b = card("g", 300, [[1, 4, "vec"], [5, 16, "rmma"], [17, 99999, "ggml_mmq"]])
        (pair,) = audit.table_audit({"a": a, "b": b})
        r = pair["regimes"]
        self.assertEqual(r["decode T<=4"]["mmq_agree"], r["decode T<=4"]["points"])
        self.assertEqual(r["verify T5-16"]["mmq_agree"], 0)  # mmq on A only
        self.assertEqual(r["verify T5-16"]["mmq_a"], r["verify T5-16"]["points"])
        # ridges 100 and 300: 17..99 memory-bound on both, 100..299 straddle, 300+ compute-bound
        self.assertGreater(r["prefill straddles ridge"]["points"], 0)
        self.assertEqual(audit.regime(150, [100, 300]), "prefill straddles ridge")
        self.assertEqual(audit.regime(2048, [100, 300]), "prefill compute-bound (both)")

    def test_other_generation_is_not_paired(self):
        pairs = audit.table_audit({"a": card("g1", 100, [[1, 9, "x"]]),
                                   "b": card("g2", 100, [[1, 9, "x"]])})
        self.assertEqual(pairs, [])


class GateAudit(unittest.TestCase):
    def test_regret_is_time_not_throughput(self):
        paired = {"rows": [{"set": "s", "model": "m", "format": "f", "card": "b",
                            "mmq": 80.0, "w4a8": 100.0},
                           {"set": "s", "model": "m", "format": "f", "card": "a",
                            "mmq": 120.0, "w4a8": 100.0}]}
        labels = {"a": {"pays": True}, "b": {"pays": False}}
        gens = {"a": "g", "b": "g"}
        g = next(x for x in audit.gate_audit(paired, labels, gens)
                 if x["label_of"] == "a")  # a's label (mmq) applied on b
        self.assertAlmostEqual(g["max_regret"], 0.25)  # 100/80 - 1
        self.assertEqual(g["wrong"], 1)

    def test_repository_numbers(self):
        root = ROOT
        gens = audit.generations(root / "src/core/generation.h")
        self.assertEqual(audit.generation_of(gens, 89)[3], "ada")
        measured = audit.measured_boards(root / "src/core/measured_boards.h")
        self.assertGreaterEqual(len(measured), 8)
        import json
        facts = json.loads((root / "src/core/data/gpu_facts.json").read_text(encoding="utf-8"))
        cards = audit.load_cards(root / "routes", gens, facts)
        lab = {v: cards[k] for k, v in audit.PAIRED_CARDS.items()}
        labels = audit.gate_labels(measured, lab)
        # issue #373: the 4070 Ti takes the L4 at 0.797, the 5070 Ti the 5060 Ti's false
        self.assertEqual(labels["rtx4070ti"]["donor"], "kL4")
        self.assertAlmostEqual(labels["rtx4070ti"]["distance"], 0.797371, places=5)
        self.assertEqual(labels["rtx5070ti"]["donor"], "kRtx5060Ti")
        self.assertFalse(labels["rtx5070ti"]["pays"])
        paired = json.loads((root / "tools/bench/data/mmq_paired_prefill.json").read_text(
            encoding="utf-8"))
        gates = audit.gate_audit(paired, labels, {v: cards[k]["generation"]
                                                  for k, v in audit.PAIRED_CARDS.items()})
        ada = [g for g in gates if g["generation"] == "ada"]
        self.assertTrue(ada and all(g["wrong"] == 0 for g in ada))
        five = {(g["label_of"], g["applied_on"]): g for g in gates if g["generation"] == "gb20x"}
        self.assertAlmostEqual(five[("rtx5070ti", "rtx5090")]["max_regret"], 14500 / 13300 - 1)
        self.assertAlmostEqual(five[("rtx5090", "rtx5070ti")]["max_regret"], 5860 / 5475 - 1)


if __name__ == "__main__":
    unittest.main()
