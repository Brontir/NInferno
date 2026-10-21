"""Convert small synthetic Qwen3.5-shaped checkpoints with both converters and compare bytes.

    python synthetic_qwen3_5.py NINFERNO_CONVERT [REPOSITORY]

The checkpoints cover every official Qwen3.5 recipe and component: dense and MoE text (GDN and
gated attention blocks, per-head query/gate row selections, the transposed convolution, fused
expert tensors read at offsets), Vision (5-D patch embedding, qkv row and bias splits,
preprocessor resources), MTP, DFlash and DFlash2 companions (whose context key/value weights are
shared aliases), the indexed proposal head, and the NVFP4 recipes' quantized sources: a
compressed-tensors NVFP4 checkpoint with text-only names (so every name is remapped), a mixed
NVFP4/FP8-row checkpoint (FP8 imports, the FP8 embedding requantized by fp8_row_maxabs), and
per-expert NVFP4 MoE banks. Each artifact must match the Python converter's byte for byte after
the 32-byte header, and the reports must match apart from timing, ids, paths and torch.

Exits 77 (skipped) when the interpreter lacks torch, which the Python converter needs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import torch
except ImportError:
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_gemma4 import VOCAB, Writer, compare, resources  # noqa: E402

H, HEADS, KV, D = 256, 4, 2, 64
LK, LKD, LV, LVD, TAPS = 2, 64, 4, 64, 4
LAYERS = 4
INTER = 384
EXPERTS, MOE_INTER, SHARED = 4, 128, 128
# Two heads of 64: the narrowest Vision head the runtime runs (Qwen3.5 0.8B to 9B).
VISION = {"depth": 2, "hidden_size": 128, "intermediate_size": 256, "num_heads": 2,
          "patch_size": 4, "temporal_patch_size": 2, "spatial_merge_size": 2,
          "num_position_embeddings": 16}


def text_fields(moe: bool, tied: bool) -> dict:
    fields = {
        "hidden_act": "silu", "attention_bias": False, "attn_output_gate": True,
        "hidden_size": H, "vocab_size": VOCAB, "num_hidden_layers": LAYERS,
        "max_position_embeddings": 4096, "tie_word_embeddings": tied, "rms_norm_eps": 1e-06,
        "full_attention_interval": 4, "num_attention_heads": HEADS, "num_key_value_heads": KV,
        "head_dim": D, "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_type": "default", "mrope_interleaved": True,
                            "mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000},
        "mamba_ssm_dtype": "float32", "linear_num_key_heads": LK, "linear_key_head_dim": LKD,
        "linear_num_value_heads": LV, "linear_value_head_dim": LVD,
        "linear_conv_kernel_dim": TAPS, "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    }
    if moe:
        fields.update({"num_experts": EXPERTS, "num_experts_per_tok": 2,
                       "moe_intermediate_size": MOE_INTER, "shared_expert_intermediate_size": SHARED})
    else:
        fields["intermediate_size"] = INTER
    return fields


def config(moe: bool, *, nested: bool, tied: bool = False, vision: bool = False) -> dict:
    arch = "Qwen3_5Moe" if moe else "Qwen3_5"
    if not nested:
        return {"architectures": [arch + "ForCausalLM"], **text_fields(moe, tied)}
    out = {"architectures": [arch + "ForConditionalGeneration"], "model_type": "qwen3_5",
           "tie_word_embeddings": tied, "text_config": text_fields(moe, tied)}
    if vision:
        out["vision_config"] = {**VISION, "in_channels": 3, "hidden_act": "gelu_pytorch_tanh",
                                "deepstack_visual_indexes": [], "out_hidden_size": H}
    return out


class Q(Writer):
    """Writer with the encodings the Qwen sources read."""

    def fp8(self, prefix, rows, k):
        codes = torch.randint(0, 0x7F, (rows, k), generator=self.generator, dtype=torch.int64)
        signs = torch.randint(0, 2, (rows, k), generator=self.generator, dtype=torch.int64) * 0x80
        self.raw(prefix + ".weight", "F8_E4M3", (rows, k), (codes | signs).to(torch.uint8).numpy().tobytes())
        scales = torch.rand((rows, 1), generator=self.generator) * 0.01 + 0.001
        self.raw(prefix + ".weight_scale", "BF16", (rows, 1),
                 scales.to(torch.bfloat16).view(torch.uint8).numpy().tobytes())

    def matrix(self, kind, prefix, rows, k, divisor=0.5, input_divisor=0.8):
        if kind == "bf16":
            self.bf16(prefix + ".weight", (rows, k))
        elif kind == "fp8":
            self.fp8(prefix, rows, k)
        else:
            self.nvfp4(prefix, rows, k, divisor, input_divisor)


def attention(w: Q, p: str, kind: str = "bf16", layer: int = 0) -> None:
    for index, (role, rows, k) in enumerate((("q_proj", 2 * HEADS * D, H), ("k_proj", KV * D, H),
                                             ("v_proj", KV * D, H), ("o_proj", H, HEADS * D))):
        w.matrix(kind, p + "self_attn." + role, rows, k, 0.4 + 0.05 * index * layer, 0.9)
    w.bf16(p + "self_attn.q_norm.weight", (D,))
    w.bf16(p + "self_attn.k_norm.weight", (D,))


def gdn(w: Q, p: str, kind: str = "bf16", layer: int = 0) -> None:
    kg, vg = LK * LKD, LV * LVD
    channels = 2 * kg + vg
    p += "linear_attn."
    w.raw(p + "A_log", "F32", (LV,), torch.randn(LV, generator=w.generator).numpy().tobytes())
    w.bf16(p + "dt_bias", (LV,))
    w.bf16(p + "conv1d.weight", (channels, 1, TAPS))
    w.bf16(p + "in_proj_a.weight", (LV, H))
    w.bf16(p + "in_proj_b.weight", (LV, H))
    w.matrix(kind, p + "in_proj_qkv", channels, H, 0.3 + 0.1 * layer, 0.7)
    w.matrix(kind, p + "in_proj_z", vg, H, 0.3 + 0.1 * layer, 0.75)
    w.bf16(p + "norm.weight", (LVD,))
    w.matrix(kind, p + "out_proj", H, vg, 0.6, 0.65)


def mlp(w: Q, p: str, kind: str = "bf16", layer: int = 0) -> None:
    # Layer 0 shares its gate/up divisor (one packed NVFP4 parent), the others do not.
    w.matrix(kind, p + "mlp.gate_proj", INTER, H, 0.35, 0.8)
    w.matrix(kind, p + "mlp.up_proj", INTER, H, 0.35 if layer == 0 else 0.45, 0.85)
    w.matrix(kind, p + "mlp.down_proj", H, INTER, 0.5, 0.6)


def moe(w: Q, p: str) -> None:
    p += "mlp."
    w.bf16(p + "gate.weight", (EXPERTS, H))
    w.bf16(p + "shared_expert_gate.weight", (1, H))
    w.bf16(p + "experts.gate_up_proj", (EXPERTS, 2 * MOE_INTER, H))
    w.bf16(p + "experts.down_proj", (EXPERTS, H, MOE_INTER))
    for role in ("gate_proj", "up_proj"):
        w.bf16(p + "shared_expert." + role + ".weight", (SHARED, H))
    w.bf16(p + "shared_expert.down_proj.weight", (H, SHARED))


def block(w: Q, p: str, mixer: str, is_moe: bool) -> None:
    w.bf16(p + "input_layernorm.weight", (H,))
    w.bf16(p + "post_attention_layernorm.weight", (H,))
    (attention if mixer == "full_attention" else gdn)(w, p)
    (moe if is_moe else mlp)(w, p)


def mixers() -> list[str]:
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(LAYERS)]


def base_checkpoint(root: Path, *, moe_model: bool, nested: bool, tied: bool, vision: bool, seed: int) -> None:
    prefix = "model.language_model." if nested else "model."
    w = Q(seed)
    w.bf16(prefix + "embed_tokens.weight", (VOCAB, H))
    if not tied:
        w.bf16("lm_head.weight", (VOCAB, H))
    w.bf16(prefix + "norm.weight", (H,))
    for layer, mixer in enumerate(mixers()):
        block(w, f"{prefix}layers.{layer}.", mixer, moe_model)
    w.bf16("mtp.fc.weight", (H, 2 * H))
    for field in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        w.bf16("mtp." + field + ".weight", (H,))
    block(w, "mtp.layers.0.", "full_attention", moe_model)
    if vision:
        h, inter = VISION["hidden_size"], VISION["intermediate_size"]
        ps, pt = VISION["patch_size"], VISION["temporal_patch_size"]
        v = "model.visual."
        w.bf16(v + "patch_embed.proj.weight", (h, 3, pt, ps, ps))
        w.bf16(v + "patch_embed.proj.bias", (h,))
        w.bf16(v + "pos_embed.weight", (VISION["num_position_embeddings"], h))
        for i in range(VISION["depth"]):
            b = f"{v}blocks.{i}."
            for norm in ("norm1", "norm2"):
                w.bf16(b + norm + ".weight", (h,))
                w.bf16(b + norm + ".bias", (h,))
            w.bf16(b + "attn.qkv.weight", (3 * h, h))
            w.bf16(b + "attn.qkv.bias", (3 * h,))
            w.bf16(b + "attn.proj.weight", (h, h))
            w.bf16(b + "attn.proj.bias", (h,))
            w.bf16(b + "mlp.linear_fc1.weight", (inter, h))
            w.bf16(b + "mlp.linear_fc1.bias", (inter,))
            w.bf16(b + "mlp.linear_fc2.weight", (h, inter))
            w.bf16(b + "mlp.linear_fc2.bias", (h,))
        merger = VISION["spatial_merge_size"] ** 2 * h
        w.bf16(v + "merger.norm.weight", (h,))
        w.bf16(v + "merger.norm.bias", (h,))
        w.bf16(v + "merger.linear_fc1.weight", (merger, merger))
        w.bf16(v + "merger.linear_fc1.bias", (merger,))
        w.bf16(v + "merger.linear_fc2.weight", (H, merger))
        w.bf16(v + "merger.linear_fc2.bias", (H,))
    w.save(root, shards=2)
    (root / "config.json").write_text(json.dumps(config(moe_model, nested=nested, tied=tied, vision=vision)), encoding="utf-8")
    resources(root)
    if vision:
        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            (root / name).write_text(json.dumps({
                "patch_size": VISION["patch_size"], "temporal_patch_size": VISION["temporal_patch_size"],
                "merge_size": VISION["spatial_merge_size"], "image_mean": [0.5, 0.5, 0.5]}), encoding="utf-8")
    counts = torch.randint(0, 1000, (VOCAB,), generator=torch.Generator().manual_seed(seed))
    (root / "ranking.i64").write_bytes(counts.to(torch.int64).numpy().tobytes())


def quantized_dense(root: Path, *, fp8: bool, seed: int) -> None:
    """Text-only names (model.layers.*): every site the NVFP4 recipes read, encoded."""
    w = Q(seed)
    w.bf16("model.embed_tokens.weight", (VOCAB, H))
    other = "fp8" if fp8 else "nvfp4"
    if fp8:
        w.fp8("lm_head", VOCAB, H)
    else:
        w.bf16("lm_head.weight", (VOCAB, H))
    w.bf16("model.norm.weight", (H,))
    for layer, mixer in enumerate(mixers()):
        p = f"model.layers.{layer}."
        w.bf16(p + "input_layernorm.weight", (H,))
        w.bf16(p + "post_attention_layernorm.weight", (H,))
        if mixer == "full_attention":
            attention(w, p, other, layer)
        else:
            gdn(w, p, other, layer)
        mlp(w, p, "nvfp4", layer)
    w.save(root)
    (root / "config.json").write_text(json.dumps(config(False, nested=False)), encoding="utf-8")


def quantized_moe(root: Path, seed: int) -> None:
    """Per-expert compressed-tensors NVFP4 banks and shared experts, each with its divisor."""
    w = Q(seed)
    for layer in range(LAYERS):
        p = f"model.language_model.layers.{layer}.mlp."
        for expert in range(EXPERTS):
            e = f"{p}experts.{expert}."
            w.nvfp4(e + "gate_proj", MOE_INTER, H, 0.2 + 0.01 * expert, 0.5 + 0.05 * expert)
            w.nvfp4(e + "up_proj", MOE_INTER, H, 0.25 + 0.01 * expert, 0.55)
            w.nvfp4(e + "down_proj", H, MOE_INTER, 0.3 + 0.02 * expert, 0.4 + 0.03 * expert)
        for role in ("gate_proj", "up_proj"):
            w.nvfp4(p + "shared_expert." + role, SHARED, H, 0.33, 0.6)
        w.nvfp4(p + "shared_expert.down_proj", H, SHARED, 0.37, 0.61)
    w.save(root)
    (root / "config.json").write_text(json.dumps({"quantization_config": {"format": "nvfp4-pack-quantized"}}), encoding="utf-8")


def drafter(root: Path, backend: str, seed: int) -> None:
    dflash = {"target_layer_ids": [1, 3], "mask_token_id": 4}
    if backend == "dflash2":
        dflash.update({"conv_kernel_size": 3, "conv_group_size": 64, "selector_rank": 32,
                       "selector_top_k": 8})
    cfg = {
        "architectures": ["DFlash2DraftModel" if backend == "dflash2" else "DFlashDraftModel"],
        "model_type": "qwen3", "hidden_act": "silu", "attention_bias": False, "is_causal": False,
        "hidden_size": H, "vocab_size": VOCAB, "num_target_layers": LAYERS,
        "intermediate_size": INTER, "num_attention_heads": HEADS, "num_key_value_heads": KV,
        "head_dim": D, "num_hidden_layers": 2, "max_position_embeddings": 4096,
        "rope_theta": 1000000, "use_sliding_window": True, "max_window_layers": 1,
        "sliding_window": 128, "dflash_config": dflash,
    }
    w = Q(seed)
    w.bf16("fc.weight", (H, 2 * H))
    w.bf16("hidden_norm.weight", (H,))
    w.bf16("norm.weight", (H,))
    for layer in range(2):
        p = f"layers.{layer}."
        w.bf16(p + "input_layernorm.weight", (H,))
        w.bf16(p + "post_attention_layernorm.weight", (H,))
        for role, rows, k in (("q_proj", HEADS * D, H), ("k_proj", KV * D, H), ("v_proj", KV * D, H),
                              ("o_proj", H, HEADS * D)):
            w.bf16(p + "self_attn." + role + ".weight", (rows, k))
        w.bf16(p + "self_attn.q_norm.weight", (D,))
        w.bf16(p + "self_attn.k_norm.weight", (D,))
        mlp(w, p)
        if backend == "dflash2":
            for branch in ("attention_conv", "mlp_conv"):
                w.bf16(p + branch + ".base_kernel", (2, 3, H))
                w.bf16(p + branch + ".kernel_projection.weight", (2 * 3 * (H // 64), H))
    if backend == "dflash2":
        w.bf16("candidate_selector.hidden_projection.weight", (32, H))
        for role in ("predecessor_codebook", "successor_codebook"):
            w.bf16("candidate_selector." + role, (VOCAB, 32))
    w.save(root)
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-qwen-") as tmp:
        work = Path(tmp)
        dense, moe_dir = work / "dense", work / "moe"
        base_checkpoint(dense, moe_model=False, nested=True, tied=False, vision=True, seed=21)
        base_checkpoint(moe_dir, moe_model=True, nested=False, tied=True, vision=False, seed=22)
        # The dense checkpoint with a YaRN rope_scaling, as Qwen documents it for long contexts:
        # both converters carry it as yarn_factor / yarn_original.
        yarn_dir = work / "dense-yarn"
        shutil.copytree(dense, yarn_dir)
        yarn_config = json.loads((yarn_dir / "config.json").read_text(encoding="utf-8"))
        yarn_config["text_config"]["rope_parameters"].update(
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096})
        (yarn_dir / "config.json").write_text(json.dumps(yarn_config), encoding="utf-8")
        quantized_dense(work / "q-nvfp4", fp8=False, seed=23)
        quantized_dense(work / "q-fp8", fp8=True, seed=24)
        quantized_moe(work / "q-moe", seed=25)
        drafter(work / "dflash", "dflash", 26)
        drafter(work / "dflash2", "dflash2", 27)
        full = ["--components=text,vision,mtp,dflash,dflash2", f"--source=dflash={work / 'dflash'}",
                f"--source=dflash2={work / 'dflash2'}", "--proposal", "--proposal-rows=100",
                f"--ranking={dense / 'ranking.i64'}"]
        cases = {
            "dense-q6": [f"--model={dense}", "--recipe=qwen3_6_27b", *full, "--name=synthetic qwen × dense"],
            "dense-q8": [f"--model={dense}", "--recipe=qwen3_8_27b", "--components=text,dflash",
                         f"--source=dflash={work / 'dflash'}", f"--source=unused={work / 'q-fp8'}"],
            "dense-nvfp4": [f"--model={dense}", "--recipe=qwen3_6_27b_nvfp4", *full,
                            f"--source=quantized={work / 'q-nvfp4'}"],
            "dense-fp8": [f"--model={dense}", "--recipe=qwen3_8_27b_nvfp4", "--components=text,mtp",
                          f"--source=quantized={work / 'q-fp8'}", "--proposal", "--proposal-rows=64",
                          f"--ranking={dense / 'ranking.i64'}"],
            "dense-yarn": [f"--model={yarn_dir}", "--recipe=qwen3_8_27b", "--components=text"],
            "moe": [f"--model={moe_dir}", "--recipe=qwen3_6_35b_a3b", "--components=text,mtp",
                    "--proposal", "--proposal-rows=100", f"--ranking={moe_dir / 'ranking.i64'}"],
            "moe-nvfp4": [f"--model={moe_dir}", "--recipe=qwen3_6_35b_a3b_nvfp4", "--components=text,mtp",
                          f"--source=quantized={work / 'q-moe'}", "--max-file-bytes=2000000"],
        }
        # What each case must exercise, read back from its report: formats, methods, and
        # whether the DFlash context weights are bound as aliases of the query-side ones.
        expect = {
            "dense-q6": ({"q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16", "fp32"},
                         {"grouped_absmax", "cast_direct"}, True),
            "dense-q8": ({"q4_g64_fp16", "q5_g64_fp16", "q8_g32_fp16"}, {"grouped_absmax"}, True),
            "dense-nvfp4": ({"nvfp4", "q8_g32_fp16"}, {"import_encoded"}, True),
            "dense-fp8": ({"nvfp4", "fp8_e4m3fn_row_bf16", "int32"},
                          {"import_encoded", "fp8_row_maxabs"}, False),
            "dense-yarn": ({"q4_g64_fp16", "q5_g64_fp16", "q8_g32_fp16"}, {"grouped_absmax"}, False),
            "moe": ({"q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16"}, {"grouped_absmax"}, False),
            "moe-nvfp4": ({"nvfp4"}, {"import_encoded"}, False),
        }
        for name, args in cases.items():
            (work / "py").mkdir(exist_ok=True)
            (work / "native").mkdir(exist_ok=True)
            python_out, native_out = work / "py" / f"{name}.ninferno", work / "native" / f"{name}.ninferno"
            failed = False
            for command in (
                [sys.executable, "-m", "tools.convert", *args, "--device=cpu", f"--out={python_out}"],
                # The parity is the recipes' arithmetic; most quantize the BF16 checkpoint.
                [str(native), *args, "--maintainer-quantize", "--quiet", f"--out={native_out}"],
            ):
                result = subprocess.run(command, cwd=repository, capture_output=True, text=True)
                if result.returncode:
                    failed = True
                    errors.append(f"{name}: {command[0]} failed\n{result.stderr[-2000:]}")
            found = [] if failed else compare(name, python_out, native_out)
            if not failed:
                report = json.loads(Path(str(native_out) + ".conversion.json").read_text(encoding="utf-8"))
                formats, methods, aliases = expect[name]
                missing = (formats - set(report["formats"])) | (
                    methods - {method["method"] for method in report["methods"]})
                bound = {p for method in report["methods"] for p in method["parameters"]}
                shared = ("dflash/layers/0/attention/key" in bound
                          and "dflash/layers/0/attention/context_key" not in bound)
                if missing or shared != aliases:
                    found.append(f"{name}: expected {sorted(missing)} and aliases={aliases}")
            errors += found
            print(f"{name}: {'ok' if not failed and not found else 'FAILED'}", flush=True)
        # --drafter-quant on a target quantizes its companion drafter only (native only: the
        # Python converter has no such option). The text keeps the recipe's methods.
        methods = {}
        for tag, extra in (("plain", []), ("q8", ["--drafter-quant=q8"])):
            out = work / "native" / f"companion-{tag}.ninferno"
            args = [f"--model={dense}", "--recipe=qwen3_8_27b", "--components=text,dflash2",
                    f"--source=dflash2={work / 'dflash2'}", *extra]
            result = subprocess.run([str(native), *args, "--maintainer-quantize", "--quiet", f"--out={out}"],
                                    cwd=repository, capture_output=True, text=True)
            if result.returncode:
                errors.append(f"companion-{tag}: failed\n{result.stderr[-2000:]}")
                continue
            report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
            methods[tag] = ({p: m["method"] for m in report["methods"] for p in m["parameters"]},
                            report["provenance"].get("drafter_quantization"))
        if len(methods) == 2:
            (plain, plain_quant), (q8, q8_quant) = methods["plain"], methods["q8"]
            drafted = [p for p in q8 if p.startswith("dflash2/") and q8[p] == "grouped_absmax"]
            text_same = all(q8.get(p) == m for p, m in plain.items() if p.startswith("text/"))
            changed = [p for p in drafted if plain.get(p) != "grouped_absmax"]
            ok = plain_quant is None and q8_quant == "q8" and changed and text_same
            if not ok:
                errors.append(f"companion drafter quant: provenance {plain_quant}/{q8_quant}, "
                              f"{len(changed)} dflash2 projections quantized, text same={text_same}")
            print(f"companion drafter quant: {'ok' if ok else 'FAILED'}", flush=True)
        # qwen3_5_bf16 (the dense Qwen3.5 GGUF default) left a DFlash companion BF16, which the
        # drafter's fused Q/K/V refused at load (attn_input_proj: invalid query/key/value
        # weight). Its projections take Q8, as every other recipe gives them; the text stays
        # BF16.
        for tag, extra, want, quant in (("bf16-default", [], "q8_g32_fp16", None),):
            out = work / "native" / f"companion-{tag}.ninferno"
            args = [f"--model={dense}", "--recipe=qwen3_5_bf16", "--components=text,dflash",
                    f"--source=dflash={work / 'dflash'}", *extra]
            result = subprocess.run([str(native), *args, "--quiet", f"--out={out}"],
                                    cwd=repository, capture_output=True, text=True)
            if result.returncode:
                errors.append(f"companion-{tag}: failed\n{result.stderr[-2000:]}")
                continue
            report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
            formats = {p: m["format"] for m in report["methods"] for p in m["parameters"]}
            drafted = {p: f for p, f in formats.items() if p.startswith("dflash/layers/")
                       and p.rsplit("/", 1)[-1] in ("query", "key", "value", "output", "gate_up",
                                                    "gate", "up", "down")}
            attention = [f"dflash/layers/0/attention/{r}" for r in ("query", "key", "value")]
            text_bf16 = all(f == "bf16" for p, f in formats.items()
                            if p.startswith("text/layers/") and p.endswith("/attention/query"))
            ok = (all(formats.get(p) == want for p in attention) and drafted
                  and all(f == want for f in drafted.values()) and text_bf16
                  and report["provenance"].get("drafter_quantization") == quant)
            if not ok:
                errors.append(f"companion-{tag}: drafter {sorted(set(drafted.values()))}, "
                              f"attention {[formats.get(p) for p in attention]}, text bf16="
                              f"{text_bf16}, provenance "
                              f"{report['provenance'].get('drafter_quantization')}")
            print(f"companion-{tag}: {'ok' if ok else 'FAILED'}", flush=True)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
