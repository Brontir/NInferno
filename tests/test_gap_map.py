"""tools/bench/gap_map.py: the two laws, the binding law, no-law rows, evidence attribution and the
CLI on synthetic inputs, plus checks of the facts it parses out of the committed doc and header.
No GPU."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import gap_map as gm

DOC = """
| `ggml_q4_k` | 256 K values, 144 bytes | 4-bit | x |
| `ggml_q4_0` | 32 K values, 18 bytes | 4-bit | binary16 d |
| `q8_g32_fp16` | 8 | 32 | `[-127, 127]` | one binary16 scale/group | 8.50 |
| `nvfp4` | E2M1, 4 bits/weight | 16 | one E4M3FN word/group | one positive FP32 |
| `mxfp4` | E2M1, 4 bits/weight | 32 | one E8M0 exponent/group | none (OCP MX) |
"""
HEADER = """
    {8, 9, 9, "ada", 512.0, 2048.0, false, 1.0, 1.11, true},
    {7, 0, 4, "volta", 1536.0, 0.0, false, 1.0, 1.11, true},
"""
FACTS = {"gpus": [
    {"chip": "ad999", "sms": 100, "cc": 89, "boost_mhz": 2000, "bus_bits": 256, "gbps": 16.0,
     "name": "Test Board"},
    {"chip": "ad999", "sms": 100, "cc": 89, "boost_mhz": 2000, "bus_bits": 128, "gbps": 16.0,
     "name": "Test Board Half"},
    {"chip": "tw", "sms": 20, "cc": 89, "boost_mhz": 1500, "bus_bits": 128, "gbps": 8.0, "name": "Twin A"},
    {"chip": "tw", "sms": 20, "cc": 89, "boost_mhz": 1800, "bus_bits": 128, "gbps": 8.0, "name": "Twin B"},
    {"chip": "gx", "sms": 10, "cc": 70, "boost_mhz": 1000, "bus_bits": 128, "gbps": 8.0,
     "name": "Volta Test"},
]}  # Test Board: 512 GB/s; Half: 256 GB/s


def table(**entry):
    base = {"format": "q4k", "compute": "a16", "epilogue": "store", "n": 4096, "k": 4096,
            "evidence_us": [[1, 99.0, 50.0], [64, 9.0, 80.0]]}
    base.update(entry)
    return {"gpu": {"class": "ad999", "name": "Test Board", "board": {"sms": 100, "dram_gbps": 512.0}},
            "entries": [base]}


class Facts(unittest.TestCase):
    def test_bits_from_doc(self):
        b = gm.load_format_bits(DOC)
        self.assertEqual(b["ggml_q4_k"], 4.5)
        self.assertEqual(b["ggml_q4_0"], 4.5)
        self.assertEqual(b["q8_g32_fp16"], 8.5)
        self.assertEqual(b["nvfp4"], 4.5)
        self.assertEqual(b["mxfp4"], 4.25)
        self.assertEqual(b["bf16"], 16.0)

    def test_real_doc_covers_every_route_format(self):
        repo = Path(__file__).resolve().parents[1]
        bits = gm.load_format_bits((repo / "docs/maintainer/tensor-formats.md").read_text(encoding="utf-8"))
        missing = [r for r, c in gm.ROUTE_FORMATS.items() if c not in bits and not r.startswith(("u4zp", "q3g"))]
        self.assertEqual(missing, [])

    def test_real_doc_bits_values(self):
        repo = Path(__file__).resolve().parents[1]
        b = gm.load_format_bits((repo / "docs/maintainer/tensor-formats.md").read_text(encoding="utf-8"))
        # The qmax table (7, 15, 31) must not overwrite the bits/weight table.
        self.assertEqual([b["q4_g64_fp16"], b["q5_g64_fp16"], b["q6_g64_fp16"], b["q8_g32_fp16"]],
                         [4.25, 5.25, 6.25, 8.5])
        self.assertEqual(b["ggml_q4_k"], 4.5)
        self.assertEqual(b["ggml_q6_k"], 210 * 8 / 256)
        self.assertEqual(b["nvfp4"], 4.5)

    def test_qmax_table_does_not_overwrite(self):
        qmax = "| `q4_g64_fp16` | 4 | `0x8` | `[-8, 7]` | 7 |"
        bits = "| `q4_g64_fp16` | 4 | 64 | `[-8, 7]` | one binary16 scale/group | 4.25 |"
        doc = chr(10).join([bits, DOC, qmax, ""])
        self.assertEqual(gm.load_format_bits(doc)["q4_g64_fp16"], 4.25)

    def test_generations_and_board(self):
        g = gm.load_generations(HEADER)
        self.assertEqual(gm.generation_of(g, 89)["name"], "ada")
        self.assertIsNone(gm.generation_of(g, 120))
        self.assertEqual(gm.board_from_facts(FACTS, "ad999", 100, 512.0)["name"], "Test Board")
        self.assertEqual(gm.board_from_facts(FACTS, "ad999", 100, 256.0)["name"], "Test Board Half")
        self.assertIsNone(gm.board_from_facts(FACTS, "ad999", 100, 900.0))
        self.assertIsNone(gm.board_from_facts(FACTS, "ad999", 99, 512.0))

    def test_real_generation_header_parses(self):
        repo = Path(__file__).resolve().parents[1]
        g = gm.load_generations((repo / "src/core/generation.h").read_text(encoding="utf-8"))
        self.assertIn("gb20x", [r["name"] for r in g])
        self.assertEqual(gm.generation_of(g, 75)["name"], "turing")


class Laws(unittest.TestCase):
    board = {"sms": 100, "boost_mhz": 2000.0, "dram_gbs": 500.0, "cc": 89}

    def test_dram_binds_at_one_token(self):
        gen = gm.generation_of(gm.load_generations(HEADER), 89)
        law = gm.law_linear(self.board, gen, 4.5, "a16", 4096, 4096, 1)
        nbytes = 4096 * 4096 * 4.5 / 8 + 4096 * 2 + 4096 * 2
        self.assertAlmostEqual(law["bytes"], nbytes)
        self.assertAlmostEqual(law["dram_us"], nbytes / 500e9 * 1e6)
        self.assertEqual(law["binds"], "dram")

    def test_tensor_binds_at_wide_prefill(self):
        gen = gm.generation_of(gm.load_generations(HEADER), 89)
        law = gm.law_linear(self.board, gen, 4.5, "a16", 4096, 4096, 4096)
        rate = 100 * 2000e6 * 512.0
        self.assertAlmostEqual(law["tensor_us"], 2 * 4096 ** 3 / rate * 1e6)
        self.assertEqual(law["binds"], "tensor")
        self.assertAlmostEqual(law["law"], law["tensor_us"])

    def test_swiglu_reads_gate_and_up(self):
        gen = gm.generation_of(gm.load_generations(HEADER), 89)
        one = gm.law_linear(self.board, gen, 4.5, "a16", 4096, 4096, 1)
        two = gm.law_linear(self.board, gen, 4.5, "a16", 4096, 4096, 1, sets=2)
        self.assertAlmostEqual(two["bytes"] - one["bytes"], 4096 * 4096 * 4.5 / 8)

    def test_no_law_rows_say_why(self):
        gen = gm.generation_of(gm.load_generations(HEADER), 70)
        self.assertIn("bits per weight", gm.law_linear(self.board, gen, None, "a16", 8, 8, 1)["why"])
        self.assertIn("a4", gm.law_linear(self.board, gen, 4.5, "a4", 8, 8, 1)["why"])
        # Volta has no int8 tensor rate: DRAM law only, with a note, never a made-up rate.
        law = gm.law_linear(self.board, gen, 4.5, "a8", 4096, 4096, 1)
        self.assertIsNone(law["tensor_us"])
        self.assertIn("tensor law missing", law["note"])

    def test_bands(self):
        self.assertEqual([gm.band_of(t) for t in (1, 3, 8, 16, 17, 256, 257)],
                         ["T1", "T2-4", "T5-8", "T9-16", "T17-64", "T65-256", "T257+"])


class Rows(unittest.TestCase):
    def test_ambiguous_board_is_no_law(self):
        self.assertIsNone(gm.board_from_facts(FACTS, "tw", 20, None))
        self.assertEqual(len(gm.board_candidates(FACTS, "tw", 20, None)), 2)
        rows = gm.rows_from_route_table(table(tuned_on="tw:20sm"), "t", FACTS, self.gens, self.bits)
        self.assertTrue(all(r["gap"] is None and "ambiguous" in r["why"] for r in rows))
        # the same chip and SM count as the table's own board: the table's board stands
        doc = table(tuned_on="ad999:100sm")
        rows = gm.rows_from_route_table(doc, "t", FACTS, self.gens, self.bits)
        self.assertTrue(all(r["gap"] is not None for r in rows))

    def test_swiglu_and_add_through_route_rows(self):
        base = gm.rows_from_route_table(table(), "a", FACTS, self.gens, self.bits)[0]
        sw = gm.rows_from_route_table(table(epilogue="swiglu"), "b", FACTS, self.gens, self.bits)[0]
        ad = gm.rows_from_route_table(table(epilogue="add"), "c", FACTS, self.gens, self.bits)[0]
        self.assertAlmostEqual(sw["bytes"] - base["bytes"], 4096 * 4096 * 4.5 / 8)
        self.assertAlmostEqual(ad["bytes"] - base["bytes"], 1 * 4096 * 2)

    def test_l2_flag(self):
        facts = {"gpus": [dict(FACTS["gpus"][0], l2_mib=1000)]}
        r = gm.rows_from_route_table(table(), "t", facts, self.gens, self.bits)[0]
        self.assertTrue(r["l2_resident"])
        facts = {"gpus": [dict(FACTS["gpus"][0], l2_mib=1)]}
        self.assertFalse(gm.rows_from_route_table(table(), "t", facts, self.gens, self.bits)[0]["l2_resident"])

    def test_below_law_rows_are_flagged_and_left_out(self):
        rows = [dict(card="C", kind="linear", format="q4k", band="T1", gap=0.6, law_us=50.0,
                     measured_us=30.0, binds="dram", why=None),
                dict(card="C", kind="linear", format="q4k", band="T1", gap=2.0, law_us=50.0,
                     measured_us=100.0, binds="dram", why=None)]
        self.assertEqual(len(gm.below_law(rows)), 1)
        (g,) = gm.aggregate(rows, 5.0)
        self.assertEqual((g["scored"], g["below_law"]), (1, 1))
        self.assertEqual(len(gm.top_gaps(rows, 5, 5.0)["linear"]), 1)

    def test_fitted_rates_give_no_tensor_law(self):
        gens = gm.load_generations(HEADER + '    {7, 5, 9, "turing", 1536.0, 2048.0, false, 1.0, 1.11, true},')
        volta, turing = gm.generation_of(gens, 70), gm.generation_of(gens, 75)
        board = {"sms": 10, "boost_mhz": 1000.0, "dram_gbs": 100.0}
        self.assertIn("fitted", gm.tensor_rate(volta, board, "a16")[1])
        self.assertIn("fitted", gm.tensor_rate(turing, board, "a16")[1])
        self.assertIsNotNone(gm.tensor_rate(turing, board, "a8")[0])
        doc = self.decode_doc(active_params=1e9, input_table_params=0)
        doc["cards"]["C"].update(major=7, minor=0)
        doc["rows"] = [{"card": "C", "model": "M", "quant": "Q", "prompt": 1000, "prefill": 1000.0}]
        r = gm.rows_from_prefill(doc, gens)[0]
        self.assertIsNone(r["gap"])
        self.assertIn("fitted rate", r["why"])

    def setUp(self):
        self.gens = gm.load_generations(HEADER)
        self.bits = gm.load_format_bits(DOC)

    def test_route_table_uses_shipped_column(self):
        rows = gm.rows_from_route_table(table(), "t", FACTS, self.gens, self.bits)
        self.assertEqual([r["measured_us"] for r in rows], [50.0, 80.0])
        self.assertTrue(all(r["gap"] == r["measured_us"] / r["law_us"] for r in rows))

    def test_borrowed_evidence_goes_to_the_board_it_was_measured_on(self):
        rows = gm.rows_from_route_table(table(tuned_on="gx:10sm"), "t", FACTS, self.gens, self.bits)
        self.assertEqual({r["card"] for r in rows}, {"Volta Test"})
        rows = gm.rows_from_route_table(table(tuned_on="nope:5sm"), "t", FACTS, self.gens, self.bits)
        self.assertTrue(all(r["gap"] is None and "no gpu_facts" in r["why"] for r in rows))

    def test_identical_evidence_counted_once(self):
        seen = set()
        a = gm.rows_from_route_table(table(), "a", FACTS, self.gens, self.bits, seen)
        b = gm.rows_from_route_table(table(), "b", FACTS, self.gens, self.bits, seen)
        self.assertEqual((len(a), len(b)), (2, 0))

    def test_moe_and_unknown_format_have_no_law(self):
        doc = table(format="mystery")
        doc["moe"] = [{"format": "mxfp4"}]
        rows = gm.rows_from_route_table(doc, "t", FACTS, self.gens, self.bits)
        self.assertEqual(sum(r["gap"] is None for r in rows), 3)
        self.assertEqual(rows[-1]["format"], "moe")

    def decode_doc(self, **model):
        m = {"experts": 0, "sliding_attention_width": 0.0, "recurrent": False,
             "kv_bytes_per_token": {"int8": 1000}, "state_bytes": {},
             "quants": {"Q": {"weight_bytes": 10_000_000, "input_table_bytes": 2_000_000}}}
        m.update(model)
        return {"models": {"M": m},
                "cards": {"C": {"major": 8, "minor": 9, "sms": 10, "clock_hz": 2e9,
                                "dram_bytes_per_s": 8e9}},
                "rows": [{"card": "C", "model": "M", "quant": "Q", "kv": "int8", "state": "f16", "ring": "bf16",
                          "concurrency": 2, "context": 100, "decode": 100.0}]}

    def test_decode_step_bytes(self):
        r = gm.rows_from_decode(self.decode_doc(), self.gens)[0]
        nbytes = 8_000_000 + 2 * 1000 * 100
        self.assertAlmostEqual(r["law_us"], nbytes / 8e9 * 1e6)
        self.assertAlmostEqual(r["measured_us"], 1e4)
        self.assertEqual(r["binds"], "dram")

    def test_decode_leaves_out_speculation_and_counts_the_ring(self):
        doc = self.decode_doc(sliding_attention_width=512.0, window=64,
                              ring_bytes_per_token={"bf16": 10})
        doc["models"]["M"]["quants"]["Q"]["speculation_bytes"] = 500_000
        doc["rows"][0].update(ring="bf16", context=5000)
        r = gm.rows_from_decode(doc, self.gens)[0]
        nbytes = (10_000_000 - 2_000_000 - 500_000) + 2 * (1000 * 5000 + 10 * 64)
        self.assertAlmostEqual(r["bytes"], nbytes)

    def test_decode_no_law_cases(self):
        self.assertIsNone(gm.rows_from_decode(self.decode_doc(experts=8), self.gens)[0]["gap"])
        doc = self.decode_doc(sliding_attention_width=512.0)
        doc["rows"][0].update(context=5000, ring="bf16")
        self.assertIsNone(gm.rows_from_decode(doc, self.gens)[0]["gap"])  # ring bytes not in the fixture

    def test_prefill_uses_int8_rate_where_the_generation_has_one(self):
        doc = self.decode_doc(active_params=1e9, input_table_params=1e8)
        doc["rows"] = [{"card": "C", "model": "M", "quant": "Q", "prompt": 1000, "prefill": 1000.0}]
        r = gm.rows_from_prefill(doc, self.gens)[0]
        self.assertAlmostEqual(r["tensor_us"], 2 * 1e9 * 1000 / (10 * 2e9 * 2048.0) * 1e6)
        self.assertEqual(r["binds"], "tensor")


class Output(unittest.TestCase):
    def test_floor_and_aggregation(self):
        rows = [dict(card="C", kind="linear", format="q4k", band="T1", gap=2.0, law_us=10.0,
                     measured_us=20.0, binds="dram", why=None),
                dict(card="C", kind="linear", format="q4k", band="T1", gap=9.0, law_us=1.0,
                     measured_us=9.0, binds="dram", why=None),
                dict(card="C", kind="linear", format="q4k", band="T1", gap=None, law_us=None,
                     measured_us=5.0, binds="no law", why="x")]
        (g,) = gm.aggregate(rows, 5.0)
        self.assertEqual((g["scored"], g["below_floor"], g["no_law"], g["median_gap"]), (1, 1, 1, 2.0))
        self.assertEqual(gm.top_gaps(rows, 5, 5.0)["linear"][0]["gap"], 2.0)

    def test_cli_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src/core/data").mkdir(parents=True)
            (root / "docs/maintainer").mkdir(parents=True)
            (root / "routes").mkdir()
            (root / "src/core/data/gpu_facts.json").write_text(json.dumps(FACTS), encoding="utf-8")
            (root / "src/core/generation.h").write_text(HEADER, encoding="utf-8")
            (root / "docs/maintainer/tensor-formats.md").write_text(DOC, encoding="utf-8")
            (root / "routes/t.json").write_text(json.dumps(table(n=8192, k=8192)), encoding="utf-8")
            self.assertEqual(gm.main(["--repo", str(root), "--md", str(root / "o.md"),
                                      "--json", str(root / "o.json")]), 0)
            rep = json.loads((root / "o.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["schema"], gm.SCHEMA)
            self.assertEqual(len(rep["rows"]), 2)
            self.assertIn("## Largest gaps", (root / "o.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
