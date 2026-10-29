"""tools/spec/registry_add.py: the "*" fallback it writes and the load table warnings (no GPU).

    python3 tests/test_registry_add.py
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "spec"))
import registry_add  # noqa: E402
import round_facts  # noqa: E402


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

    def test_local_paths_are_dropped(self):
        tune = json.loads(json.dumps(self.tune))
        tune["artifact"] = "/root/x.ninferno"
        tune["drafters"] = [{"name": "a", "artifact": "/root/a.ninferno"}]
        pair = registry_add.pair_of(tune, self.manifest, "org/m@abc", "Q4_K_M", ["a" * 64])
        self.assertNotIn("artifact", pair["tune"])
        self.assertNotIn("artifact", pair["tune"]["drafters"][0])

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


class RoundFactsTest(unittest.TestCase):
    """tools/spec/round_facts.py: what each pass of a round reads, from a tensor table."""

    @staticmethod
    def table(names: dict, meta: dict | None = None) -> dict:
        return {"meta": meta or {}, "tensors": {n: b for n, (b, e) in names.items()},
                "elems": {n: e for n, (b, e) in names.items()}}

    def pair(self, configs: list[str], drafters: dict | None = None, kinds: tuple = ()) -> dict:
        return {"target_repo": "org/t@1 Q4_K_M", "drafters": drafters or {},
                "tune": {"key": {"gpu": "ad104:60sm"}, "drafters": [{"kind": k} for k in kinds],
                         "measurements": [{"config": c} for c in configs + ["plain"]]}}

    def test_mtp_layers_and_dflash(self):
        meta = {"general.architecture": "qwen35", "qwen35.block_count": 3, "qwen35.nextn_predict_layers": 1}
        target = self.table({"token_embd.weight": (100, 400), "output.weight": (100, 400),
                             "blk.0.attn_q.weight": (50, 200), "blk.1.ffn_up.weight": (50, 200),
                             "blk.2.nextn.eh_proj.weight": (10, 40), "v.blk.0.w": (999, 999)}, meta)
        drafter = self.table({"model.layers.0.mlp.up_proj.weight": (30, 15)}, {"block_size": 16})
        tables = {"org/t@1 Q4_K_M": target, "drafter:z/d@2": drafter}
        facts = round_facts.round_facts(self.pair(["mtp:3", "dflash:7"], {"dflash": "z/d@2"}), tables, {})
        self.assertEqual(facts["target"], {"bytes": 200, "params": 800})
        self.assertEqual(facts["drafters"]["mtp"], {"bytes": 10, "params": 40, "rows": 1, "per_draft": True,
                                                    "head": {"bytes": 100, "params": 400}})
        self.assertEqual(facts["drafters"]["dflash"],
                         {"bytes": 30, "params": 15, "rows": 16, "per_draft": False,
                          "head": {"bytes": 100, "params": 400}})
        drafter["meta"] = {}
        facts = round_facts.round_facts(self.pair(["dflash:7"], {"dflash": "z/d@2"}), tables, {})
        self.assertIsNone(facts)  # no block_size: no DFlash facts

    def test_nemotron_sidecar_counts_its_nextn_layer(self):
        meta = {"general.architecture": "nemotron_h_moe", "nemotron_h_moe.expert_count": 8,
                "nemotron_h_moe.expert_used_count": 2}
        target = self.table({"output.weight": (10, 40), "blk.0.ffn_up_exps.weight": (80, 320),
                             "blk.0.attn_q.weight": (5, 20)}, meta)
        side = self.table({"output.weight": (10, 40), "blk.0.attn_q.weight": (3, 12),
                           "blk.0.ffn_up_exps.weight": (16, 64)},
                          dict(meta, **{"nemotron_h_moe.block_count": 1,
                                        "nemotron_h_moe.nextn_predict_layers": 1}))
        registry = {"models": [{"drafters": [{"kind": "nemotron_h_mtp", "repo": "b/g", "gguf": "mtp-Q4_0"}]}]}
        tables = {"org/t@1 Q4_K_M": target, "drafter:b/g@3:mtp-Q4_0.gguf": side}
        facts = round_facts.round_facts(self.pair(["mtp:2"], {"nemotron_h_mtp": "b/g@3"}, ("nemotron_h_mtp",)),
                                        tables, registry)
        self.assertEqual(facts["drafters"]["mtp"], {"bytes": 13, "params": 52, "expert_bytes": 16,
                                                    "expert_params": 64, "experts": 8, "experts_used": 2,
                                                    "rows": 1, "per_draft": True})

    def test_packed_safetensors_count_logical_parameters(self):
        header = {"a.qweight": {"dtype": "I32", "shape": [4, 2], "data_offsets": [0, 32]},
                  "a.scales": {"dtype": "F16", "shape": [2], "data_offsets": [32, 36]},
                  "b.weight_packed": {"dtype": "U8", "shape": [4], "data_offsets": [36, 40]},
                  "b.weight_scale": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [40, 41]},
                  "c.weight": {"dtype": "BF16", "shape": [3], "data_offsets": [41, 47]}}
        tensors, elems, shapes = round_facts.parse_safetensors_header(header, 4)
        self.assertEqual(elems, {"a.qweight": 64, "a.scales": 0, "b.weight_packed": 8, "b.weight_scale": 0,
                                 "c.weight": 3})
        self.assertEqual(tensors["a.scales"], 4)
        self.assertEqual(shapes["a.qweight"], ["int4", [4, 2]])
        self.assertEqual(shapes["b.weight_packed"], ["nvfp4", [4]])
        self.assertEqual(shapes["b.weight_scale"], ["fp8", [1]])
        self.assertEqual(shapes["c.weight"], ["bf16", [3]])
        # compressed-tensors MXFP4: the same uint8 weight_packed, its block scales E8M0 (uint8).
        mx = {"d.weight_packed": {"dtype": "U8", "shape": [512, 128], "data_offsets": [0, 65536]},
              "d.weight_scale": {"dtype": "U8", "shape": [512, 8], "data_offsets": [65536, 69632]}}
        _, elems, shapes = round_facts.parse_safetensors_header(mx, 4)
        self.assertEqual(shapes["d.weight_packed"], ["mxfp4", [512, 128]])
        self.assertEqual(round_facts.linear_shape("d.weight_packed", "mxfp4", [512, 128],
                                                  elems["d.weight_packed"], "compressed-tensors"), (512, 256, 1))

    def test_gguf_sizes_from_offsets(self):
        def s(text):
            raw = text.encode()
            return struct.pack("<Q", len(raw)) + raw
        buf = b"GGUF" + struct.pack("<IQQ", 3, 2, 0)
        buf += s("x") + struct.pack("<I", 1) + struct.pack("<Q", 64) + struct.pack("<IQ", 99, 0)
        buf += s("y") + struct.pack("<I", 1) + struct.pack("<Q", 32) + struct.pack("<IQ", 1, 96)
        _, tensors, elems, shapes = round_facts.parse_gguf(buf)
        self.assertEqual(tensors, {"x": 96, "y": 64})  # x (an unknown type) by the next offset
        self.assertEqual(elems, {"x": 64, "y": 32})
        self.assertEqual(shapes, {"x": ["gguf:type99", [64]], "y": ["gguf:F16", [32]]})

    def test_nested_quantization_and_newer_ggml_types(self):
        meta, bits = round_facts._config_meta({"text_config": {"quantization_config": {"quant_method": "gptq",
                                                                                       "bits": 4}}})
        self.assertEqual((meta["quant_method"], bits), ("gptq", 4))
        self.assertEqual(round_facts.GGML_NAMES[40], "NVFP4")
        self.assertEqual(round_facts.GGML_BLOCKS[40], (64, 36))

    def test_norms_by_role_and_int8(self):
        # Nemotron-H's grouped SSM norm is 2-D, no linear; compressed-tensors W8A16 reads int8.
        self.assertIsNone(round_facts.linear_shape("blk.0.ssm_norm.weight", "gguf:F32", [512, 8], 4096, ""))
        self.assertIsNone(round_facts.linear_shape("model.layers.0.mixer.norm.weight", "bf16", [8, 512], 4096, ""))
        self.assertEqual(round_facts.safetensors_format("a.weight", "I8", 8), "int8")
        self.assertEqual(round_facts.linear_shape("a.weight", "int8", [512, 256], 131072, "compressed-tensors"),
                         (512, 256, 1))

    @staticmethod
    def shaped(shapes: dict, meta: dict | None = None, bytes_per: float = 1.0) -> dict:
        """A table from {name: [format, dims, parameters]}: `bytes_per` bytes a parameter."""
        return {"meta": meta or {}, "tensors": {n: int(e * bytes_per) for n, (f, d, e) in shapes.items()},
                "elems": {n: e for n, (f, d, e) in shapes.items()},
                "shapes": {n: [f, d] for n, (f, d, e) in shapes.items()}}

    def test_gguf_linears_tied_head_experts_and_mtp(self):
        meta = {"general.architecture": "qwen35moe", "qwen35moe.block_count": 2,
                "qwen35moe.nextn_predict_layers": 1}
        target = self.shaped({
            "token_embd.weight": ["gguf:Q6_K", [256, 1000], 256000],
            "blk.0.attn_q.weight": ["gguf:Q4_K", [256, 512], 131072],
            "blk.0.attn_k.weight": ["gguf:Q4_K", [256, 64], 16384],
            "blk.0.ssm_conv1d.weight": ["gguf:F32", [4, 512], 2048],  # a convolution: no linear
            "blk.0.attn_norm.weight": ["gguf:F32", [256], 256],
            "blk.0.ffn_up_exps.weight": ["gguf:Q4_K", [256, 128, 8], 262144],
            "blk.1.nextn.eh_proj.weight": ["gguf:Q8_0", [512, 256], 131072],
            "blk.1.ffn_up_exps.weight": ["gguf:Q4_K", [256, 128, 8], 262144]}, meta)
        facts = round_facts.round_facts(self.pair(["mtp:3"]), {"org/t@1 Q4_K_M": target}, {})
        head = {"format": "gguf:Q6_K", "n": 1000, "k": 256, "count": 1, "head": True, "expert": False}
        self.assertEqual(facts["target"]["linears"], [
            {"format": "gguf:Q4_K", "n": 64, "k": 256, "count": 1, "head": False, "expert": False},
            {"format": "gguf:Q4_K", "n": 512, "k": 256, "count": 1, "head": False, "expert": False},
            {"format": "gguf:Q4_K", "n": 128, "k": 256, "count": 8, "head": False, "expert": True}, head])
        self.assertEqual(facts["drafters"]["mtp"]["linears"], [
            {"format": "gguf:Q8_0", "n": 256, "k": 512, "count": 1, "head": False, "expert": False},
            {"format": "gguf:Q4_K", "n": 128, "k": 256, "count": 8, "head": False, "expert": True}])
        self.assertEqual(facts["drafters"]["mtp"]["head"]["linears"], [head])  # per draft, apart

    def test_checkpoint_linears_as_published_and_converted(self):
        target = self.shaped({
            "model.embed_tokens.weight": ["bf16", [1000, 256], 256000],
            "lm_head.weight": ["nvfp4", [1000, 128], 256000],
            "model.layers.0.self_attn.q_proj.weight": ["nvfp4", [512, 128], 131072],
            "model.layers.0.self_attn.q_proj.weight_scale": ["fp8", [512, 16], 0],
            "model.layers.0.mlp.experts.gate_up_blocks": ["mxfp4", [8, 128, 8, 16], 262144],
            "model.layers.0.mlp.experts.down_proj": ["bf16", [8, 256, 128], 262144],
            "model.layers.0.linear_attn.conv1d.weight": ["bf16", [512, 1, 4], 2048],
            "model.layers.0.mlp.gate.weight": ["bf16", [8, 256], 2048]},
            {"expert_count": 8, "expert_used_count": 2}, 0.5)
        drafter = self.shaped({"layers.0.mlp.up_proj.weight": ["bf16", [512, 256], 131072],
                               "layers.0.conv.base_kernel": ["bf16", [2, 2, 256], 1024]},
                              {"block_size": 8}, 2)
        registry = {"kinds": {"dflash": {"quant": "q8"}}}
        tables = {"org/t@1 Q4_K_M": target, "drafter:z/d@2": drafter}
        facts = round_facts.round_facts(self.pair(["dflash:7"], {"dflash": "z/d@2"}), tables, registry)
        head = {"format": "nvfp4", "n": 1000, "k": 256, "count": 1, "head": True, "expert": False}
        self.assertEqual(facts["target"]["linears"], [
            {"format": "bf16", "n": 8, "k": 256, "count": 1, "head": False, "expert": False},
            {"format": "nvfp4", "n": 512, "k": 256, "count": 1, "head": False, "expert": False},
            {"format": "bf16", "n": 256, "k": 128, "count": 8, "head": False, "expert": True},
            {"format": "mxfp4", "n": 128, "k": 256, "count": 8, "head": False, "expert": True}, head])
        dflash = facts["drafters"]["dflash"]
        self.assertEqual(dflash["linears"],
                         [{"format": "conv:q8", "n": 512, "k": 256, "count": 1, "head": False, "expert": False}])
        self.assertEqual(dflash["head"]["linears"], [head])

    def test_awq_and_gptq_qweight_orientation(self):
        # AWQ: qweight [k, n / 8]; GPTQ: [k / 8, n]; compressed-tensors weight_packed: [n, k / 8].
        self.assertEqual(round_facts.linear_shape("a.qweight", "int4", [256, 64], 256 * 512, "awq"), (512, 256, 1))
        self.assertEqual(round_facts.linear_shape("a.qweight", "int4", [32, 512], 256 * 512, "gptq"), (512, 256, 1))
        self.assertEqual(round_facts.linear_shape("a.weight_packed", "int4", [512, 32], 256 * 512,
                                                  "compressed-tensors"), (512, 256, 1))
        # A qweight of a method whose orientation is not known: no linear rather than a guess.
        self.assertIsNone(round_facts.linear_shape("a.qweight", "int4", [256, 64], 256 * 512, "auto-round"))
        # A dense MXFP4 weight's blocks [n, k / 32, 16].
        self.assertEqual(round_facts.linear_shape("a.weight_blocks", "mxfp4", [512, 8, 16], 512 * 256, ""),
                         (512, 256, 1))

    def test_row_tables_are_in_no_pass(self):
        # DFlash2's selector codebooks are read a few rows by id: neither its bytes nor a linear.
        target = self.table({"output.weight": (10, 40), "blk.0.attn_q.weight": (5, 20)})
        drafter = self.shaped({"layers.0.mlp.up_proj.weight": ["bf16", [512, 256], 131072],
                               "candidate_selector.predecessor_codebook": ["bf16", [1000, 256], 256000],
                               "candidate_selector.successor_codebook": ["bf16", [1000, 256], 256000]},
                              {"block_size": 8}, 2)
        facts = round_facts.round_facts(self.pair(["dflash2:7"], {"dflash2": "z/d@2"}),
                                        {"org/t@1 Q4_K_M": target, "drafter:z/d@2": drafter}, {})
        dflash2 = facts["drafters"]["dflash2"]
        self.assertEqual((dflash2["bytes"], dflash2["params"]), (262144, 131072))
        self.assertEqual(len(dflash2["linears"]), 1)

    def test_a_sidecar_without_a_head_runs_the_targets(self):
        meta = {"general.architecture": "nemotron_h_moe", "nemotron_h_moe.block_count": 1,
                "nemotron_h_moe.nextn_predict_layers": 1}
        target = self.shaped({"output.weight": ["gguf:Q6_K", [256, 1000], 256000],
                              "blk.0.attn_q.weight": ["gguf:Q4_0", [256, 256], 65536]})
        side = self.shaped({"blk.0.attn_q.weight": ["gguf:Q4_0", [256, 512], 131072]}, meta)
        registry = {"models": [{"drafters": [{"kind": "nemotron_h_mtp", "repo": "b/g", "gguf": "m"}]}]}
        facts = round_facts.round_facts(self.pair(["mtp:2"], {"nemotron_h_mtp": "b/g@3"}, ("nemotron_h_mtp",)),
                                        {"org/t@1 Q4_K_M": target, "drafter:b/g@3:m.gguf": side}, registry)
        self.assertEqual(facts["drafters"]["mtp"]["linears"], [
            {"format": "gguf:Q4_0", "n": 512, "k": 256, "count": 1, "head": False, "expert": False},
            {"format": "gguf:Q6_K", "n": 1000, "k": 256, "count": 1, "head": True, "expert": False}])

    def test_embeddings_by_role_and_the_sidecar_first(self):
        meta = {"expert_count": 8, "expert_used_count": 2}
        target = self.table({"backbone.embeddings.weight": (700, 350), "lm_head.weight": (10, 40),
                             "backbone.layers.0.mixer.q_proj.weight": (5, 20),
                             "mtp.layers.0.q_proj.weight": (90, 90)}, meta)
        side = self.table({"blk.0.attn_q.weight": (3, 12)})
        registry = {"models": [{"drafters": [{"kind": "nemotron_h_mtp", "repo": "b/g", "gguf": "m"}]}]}
        tables = {"org/t@1 Q4_K_M": target, "drafter:b/g@3:m.gguf": side}
        facts = round_facts.round_facts(self.pair(["mtp:2"], {"nemotron_h_mtp": "b/g@3"}, ("nemotron_h_mtp",)),
                                        tables, registry)
        self.assertEqual(facts["target"], {"bytes": 15, "params": 60})  # no embedding table read
        self.assertEqual(facts["drafters"]["mtp"]["bytes"], 13)  # the sidecar, not mtp.* (90)

    def test_a_drafter_as_its_kind_attaches_it(self):
        assistant = self.table({"model.layers.0.mlp.up_proj.weight": (2 * 65536, 65536),
                                "model.layers.0.input_layernorm.weight": (64, 32),
                                "model.embed_tokens.weight": (2 * 400 * 256, 400 * 256)}, {"vocab_size": 400})
        registry = {"kinds": {"gemma4_assistant": {"quant": "q8", "proposal_rows": 100}},
                    "models": [{"drafters": [{"kind": "gemma4_assistant", "repo": "g/a"}]}]}
        target = self.table({"output.weight": (10, 40), "blk.0.attn_q.weight": (5, 20)})
        facts = round_facts.round_facts(self.pair(["mtp:3"], {"gemma4_assistant": "g/a@1"}, ("gemma4_assistant",)),
                                        {"org/t@1 Q4_K_M": target, "drafter:g/a@1": assistant}, registry)
        # q8 projection (65536 x 34/32), the norm kept (64), the head's 100 of 400 rows in BF16.
        self.assertEqual(facts["drafters"]["mtp"], {"bytes": 69632 + 64 + 2 * 100 * 256,
                                                    "params": 65536 + 32 + 100 * 256, "rows": 1, "per_draft": True})

    def test_assistant_linears_converted_and_its_head_over_proposal_rows(self):
        assistant = self.shaped({"model.layers.0.mlp.up_proj.weight": ["bf16", [256, 256], 65536],
                                 "model.layers.0.input_layernorm.weight": ["bf16", [256], 256],
                                 "model.embed_tokens.weight": ["bf16", [400, 256], 102400]},
                                {"vocab_size": 400}, 2)
        registry = {"kinds": {"gemma4_assistant": {"quant": "q8", "proposal_rows": 100}},
                    "models": [{"drafters": [{"kind": "gemma4_assistant", "repo": "g/a"}]}]}
        target = self.table({"output.weight": (10, 40), "blk.0.attn_q.weight": (5, 20)})
        facts = round_facts.round_facts(self.pair(["mtp:3"], {"gemma4_assistant": "g/a@1"}, ("gemma4_assistant",)),
                                        {"org/t@1 Q4_K_M": target, "drafter:g/a@1": assistant}, registry)
        self.assertEqual(facts["drafters"]["mtp"]["linears"], [
            {"format": "conv:q8", "n": 256, "k": 256, "count": 1, "head": False, "expert": False},
            {"format": "bf16", "n": 100, "k": 256, "count": 1, "head": True, "expert": False}])
        self.assertNotIn("linears", facts["target"])  # a table without shapes lists none

    def test_stale_facts_are_not_kept(self):
        old = {"target": "w", "drafters": {"dflash": "z/d@1"}, "generation": "sm89",
               "tune": {"key": {"gpu": "ad104:60sm"}}, "round_facts": {"drafters": {"dflash": {}}}}
        new = {"target": "w", "drafters": {"dflash": "z/d@1"}, "generation": "sm89", "target_repo": "t",
               "tune": {"key": {"gpu": "ad104:60sm"}, "measurements": [{"config": "mtp:3"}, {"config": "plain"}]}}
        with contextlib.redirect_stderr(io.StringIO()):
            out = registry_add.put_pair([old], new)
        self.assertNotIn("round_facts", out[0])

    def test_a_new_tune_keeps_the_facts_and_board(self):
        old = {"target": "w", "drafters": {"dflash": "z/d@1"}, "generation": "sm89",
               "tune": {"key": {"gpu": "ad104:60sm"}}, "round_facts": {"x": 1}, "lab_board": "0x2782"}
        new = {"target": "w", "drafters": {"dflash": "z/d@1"}, "generation": "sm89",
               "tune": {"key": {"gpu": "ad104:60sm"}}}
        out = registry_add.put_pair([old], new)
        self.assertEqual(out, [dict(new, round_facts={"x": 1}, lab_board="0x2782")])

if __name__ == "__main__":
    unittest.main()
