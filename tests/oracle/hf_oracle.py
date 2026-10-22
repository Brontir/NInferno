"""Diff oracle: synthetic mini models through ninferno-convert and the engine, against transformers.

    python hf_oracle.py NINFERNO_CONVERT ORACLE_RUNNER [REPOSITORY] [--only CASE,...] [--keep DIR]
                        [--no-sabotage] [--moe]

Two mini architectures with random weights, four layers holding every layer type: Qwen3.5 dense
(three GDN layers and a gated full-attention layer, tied head) and Gemma 4 dense (sliding and
global layers with attention_k_eq_v, a sliding window of 64). Head geometries are the production
ones (the attention kernels are compiled per head count); widths and vocabulary are small. Each is
written as a Hugging Face checkpoint that transformers loads, plus a compressed-tensors source
quantized from the same weights (NVFP4 with fused groups sharing their global scale and calibrated
input scales, FP8 attention for Gemma), and converted with ninferno-convert by the recipes that apply:
Qwen qwen3_5_bf16, qwen3_5_q8, qwen3_6_27b (Q4/Q5/Q6) and qwen3_6_27b_nvfp4; Gemma
gemma4_31b_nvfp4 from the BF16 checkpoint (Q8/Q6) and from the NVFP4/FP8 source. The engine scores
three sequences of 150, 257 and 300 tokens teacher-forced (Gemma also through decode steps and with
48-token prefill chunks) and its logits are compared with transformers in FP32 on the CPU.

The reference gets the weights the artifact actually holds: every quantized projection is decoded
from the artifact and written back over the checkpoint rows it came from (found by matching rows,
which also checks that the converter kept every row whole and in its place). What remains between
the two sides is arithmetic. Limits come from the arithmetic floor, the same reference run in BF16
against itself in FP32 (floor_limits): KL(reference || engine) maximum and 99th percentile within
twice the floor (never below 1e-4), and top-1 agreement wherever the reference's top-2 gap is not a
near-tie. Q4-Q6 weights prefill with int8 activations: that case bounds their encoding error, and
the same artifact is checked tightly under --activations exact. W4A4 cases encode the projection
inputs in the reference as the engine's routes do; on
these models that encoding is so sensitive to its input that the floor is as large as the encoding
error itself, so those cases only bound it, and the same artifact is checked tightly again under
A16 (Gemma's ActivationProfile::A16Full; for Qwen the Uses rewritten to A16Only).

Sabotage arms change the reference the way a real bug would change the engine (a permutation of
attention heads, the other RMSNorm convention, a doubled attention scale, a doubled weight scale)
and must each fail the same criteria, or the oracle is mute (W4A4 cases report them). Gemma 4 is
also compared layer by layer at lengths window-1, window, window+1, 2*window+17 and
4*window, with a sabotage arm that windows the global layers. On the same target, one MTP
assistant draft step is compared with transformers' Gemma4AssistantForCausalLM: the
input the drafter reads and its logits (cosine above 0.999); a reference with another RoPE must
disagree, and a drafter with another RoPE must be refused at load.

--moe adds the mini MoE models: the engine's MoE ops take only the production geometries today
(Qwen SparseMoe 256 experts of 512 at hidden 2048, Gemma4Moe 128 experts of 704 at hidden 2816),
so they fail at load and are not part of the default run.

Exits 77 (skipped) without torch or transformers, or without a CUDA device.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

try:
    import torch
    import transformers  # noqa: F401
    from safetensors.torch import save_file
except ImportError:
    sys.exit(77)

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled  # noqa: E402
from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales  # noqa: E402
from tools.artifact.codecs.row_split import dequantize_row_split  # noqa: E402
from tools.artifact.formats import QUANT_FORMATS  # noqa: E402
from tools.artifact.framing import HEADER  # noqa: E402
from tools.artifact.layouts import block_scale_geometry  # noqa: E402
from tools.artifact.reader import Artifact  # noqa: E402

VOCAB = 512
SEEDS = {"qwen-dense": 11, "qwen-moe": 12, "gemma-moe": 13, "gemma-dense": 14}
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# A reference top-2 gap below this is a near-tie: BF16 logits alone move by up to 1/64 at |z| < 4.
NEAR_TIE = 0.1
# The BF16 target: KL(FP32 reference || engine) below 1e-4 where the model's own BF16 floor allows.
KL_LIMIT = 1e-4
FLOOR_MARGIN = 2.0
# Per-layer relative error of the attention context and the layer output.
LAYER_LIMIT = 0.05


# ---------------------------------------------------------------------------------------------
# Architectures


@dataclass
class Arch:
    name: str
    family: str  # qwen3_5 | gemma4
    moe: bool
    config: dict
    window: int = 0

    @property
    def text(self) -> dict:
        return self.config


def qwen_arch(moe: bool) -> Arch:
    layers = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    config = {
        "architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "hidden_act": "silu", "attention_bias": False, "attn_output_gate": True,
        "hidden_size": 256, "vocab_size": VOCAB, "num_hidden_layers": len(layers),
        "max_position_embeddings": 4096, "tie_word_embeddings": not moe, "rms_norm_eps": 1e-06,
        "full_attention_interval": 4, "layer_types": layers,
        # Qwen3.5 0.8B's attention heads: the causal-cache kernels are compiled per (q, kv) pair.
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256, "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_type": "default", "mrope_interleaved": True,
                            "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000},
        "mamba_ssm_dtype": "float32", "linear_num_key_heads": 2, "linear_key_head_dim": 128,
        "linear_num_value_heads": 4, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
    }
    if moe:
        # The engine's SparseMoe implements top-8 routing only.
        config.update({"num_experts": 16, "num_experts_per_tok": 8, "moe_intermediate_size": 128,
                       "shared_expert_intermediate_size": 128})
    else:
        config["intermediate_size"] = 512
    return Arch("qwen-moe" if moe else "qwen-dense", "qwen3_5", moe, config)


def gemma_arch(moe: bool, window: int = 64) -> Arch:
    layers = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    config = {
        "architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text",
        "hidden_size": 256, "vocab_size": VOCAB, "num_hidden_layers": len(layers),
        "max_position_embeddings": 4096, "tie_word_embeddings": True, "rms_norm_eps": 1e-06,
        "layer_types": layers, "num_attention_heads": 2, "attention_k_eq_v": not moe,
        "sliding_window": window, "head_dim": 256, "num_key_value_heads": 1,
        "global_head_dim": 512, "num_global_key_value_heads": 1,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
        },
        "final_logit_softcapping": 30.0, "intermediate_size": 256 if moe else 512,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0, "attention_bias": False, "attention_dropout": 0.0,
        "enable_moe_block": moe, "use_double_wide_mlp": False,
        "pad_token_id": 0, "eos_token_id": 1, "bos_token_id": 2,
    }
    if moe:
        config.update({"num_experts": 8, "top_k_experts": 2, "moe_intermediate_size": 128})
    return Arch("gemma-moe" if moe else "gemma-dense", "gemma4", moe, config, window)


def hf_model(arch: Arch, config: dict | None = None):
    """The transformers model of `arch` in FP32 with eager attention."""
    fields = dict(config or arch.config)
    for key in ("architectures", "model_type"):
        fields.pop(key, None)
    if arch.family == "gemma4":
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        cfg, cls = Gemma4TextConfig(**fields), Gemma4ForCausalLM
    elif arch.moe:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

        cfg, cls = Qwen3_5MoeTextConfig(**fields), Qwen3_5MoeForCausalLM
    else:
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        cfg, cls = Qwen3_5TextConfig(**fields), Qwen3_5ForCausalLM
    cfg._attn_implementation = "eager"
    return cls(cfg).float().eval()


# ---------------------------------------------------------------------------------------------
# Random weights


def init_tensor(arch: Arch, name: str, shape, gen: torch.Generator) -> torch.Tensor:
    def randn(scale=1.0):
        return torch.randn(shape, generator=gen) * scale

    fan_in = shape[-1] if len(shape) > 1 else 1
    if name.endswith("A_log"):
        return torch.log(torch.rand(shape, generator=gen) * 15 + 1)
    if name.endswith("dt_bias"):
        return randn(0.5)
    if "conv1d" in name:
        return randn(0.5)
    if name.endswith("layer_scalar"):
        return torch.rand(shape, generator=gen) * 0.4 + 0.6
    if name.endswith(("router.scale", "router.per_expert_scale")):
        return 1 + randn(0.1)
    if len(shape) == 1:  # a norm weight
        if arch.family == "gemma4" and ("post_attention_layernorm" in name
                                        or "post_feedforward_layernorm" in name):
            # Gemma normalizes every block's output before the residual add; at unit weight the
            # blocks barely move the scaled embedding and the model hides what they compute.
            return 3 * (1 + randn(0.1))
        if arch.family == "gemma4" and ("q_norm" in name or "k_norm" in name):
            # Gemma attends at scale 1 over normalized queries and keys: unit weights give scores
            # with a spread of sqrt(head_dim) = 16, a softmax so sharp that BF16 rounding picks
            # other keys. A quarter keeps the spread near one, as trained weights do.
            return 0.25 * (1 + randn(0.1))
        plain = arch.family == "gemma4" or "linear_attn.norm" in name
        return 1 + randn(0.1) if plain else randn(0.1)
    if "embed_tokens" in name:
        # Qwen ties its head to this table (logit spread ~2.5); Gemma scales it by sqrt(H).
        return randn(0.16)
    if "lm_head" in name:
        return randn(2.5 / math.sqrt(fan_in))
    if name.endswith(("mlp.gate.weight", "router.proj.weight")):
        # Routers are wide so that routing near-ties are rare: a flipped expert is not rounding.
        return randn((8.0 if arch.family == "qwen3_5" else 64.0) / math.sqrt(fan_in))
    if arch.family == "qwen3_5" and name.endswith(
            ("o_proj.weight", "out_proj.weight", "down_proj.weight", "experts.down_proj")):
        # Qwen's blocks have no post-norms: at full scale every block's output dwarfs the
        # residual it adds to, and the mini model turns chaotic (0.5 % weight noise -> KL 1e-2),
        # far from a trained model. A third of the scale keeps it smooth (KL 5e-4).
        return randn(0.3 / math.sqrt(fan_in))
    return randn(1.0 / math.sqrt(fan_in))


def random_weights(arch: Arch, seed: int) -> dict[str, torch.Tensor]:
    """Checkpoint tensors under transformers' names (plus Qwen's MTP layer), BF16 except A_log."""
    gen = torch.Generator().manual_seed(seed)
    model = hf_model(arch)
    shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    if arch.config.get("tie_word_embeddings"):
        shapes.pop("lm_head.weight", None)
    if arch.family == "qwen3_5" and arch.moe:
        h = arch.config["hidden_size"]
        shapes["mtp.fc.weight"] = (h, 2 * h)
        for norm in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            shapes[f"mtp.{norm}.weight"] = (h,)
        full = arch.config["layer_types"].index("full_attention")
        for name, shape in list(shapes.items()):
            if name.startswith(f"model.layers.{full}."):
                shapes["mtp.layers.0." + name[len(f"model.layers.{full}."):]] = shape
    out = {}
    for name in sorted(shapes):
        value = init_tensor(arch, name, shapes[name], gen)
        out[name] = value.float() if name.endswith("A_log") else value.to(torch.bfloat16)
    return out


# ---------------------------------------------------------------------------------------------
# Checkpoints


def bytes_to_unicode() -> dict[int, str]:
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping, extra = {}, 0
    for byte in range(256):
        if byte in visible:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(256 + extra)
            extra += 1
    return mapping


SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]


def resources(root: Path) -> None:
    """A byte-level BPE tokenizer whose public domain is the whole vocabulary."""
    mapping = bytes_to_unicode()
    vocab = {mapping[b]: b for b in range(256)}
    names = SPECIALS + [f"<|x{i}|>" for i in range(256 - len(SPECIALS))]
    added = [{"id": 256 + i, "content": n, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": n not in ("<think>", "</think>")}
             for i, n in enumerate(names)]
    (root / "tokenizer.json").write_text(json.dumps(
        {"version": "1.0", "model": {"type": "BPE", "vocab": vocab, "merges": []},
         "added_tokens": added}), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps(
        {"add_bos_token": False, "add_prefix_space": False, "eos_token": "<|im_end|>",
         "pad_token": "<|endoftext|>",
         "added_tokens_decoder": {str(a["id"]): a for a in added}}), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [256, 258]}), encoding="utf-8")
    (root / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")


def expert_views(arch: Arch, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Every fused expert bank's matrices under their per-expert names, as views of the bank."""
    out = {}
    if not arch.moe:
        return out
    inter = arch.config["moe_intermediate_size"]
    for name, value in weights.items():
        if name.endswith("experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            for e in range(value.shape[0]):
                out[f"{base}{e}.gate_proj.weight"] = value[e, :inter]
                out[f"{base}{e}.up_proj.weight"] = value[e, inter:]
        elif name.endswith("experts.down_proj"):
            base = name[: -len("down_proj")]
            for e in range(value.shape[0]):
                out[f"{base}{e}.down_proj.weight"] = value[e]
    return out


def converter_names(arch: Arch, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Gemma 4 experts under their per-expert names too (the layout the converter reads)."""
    out = dict(weights)
    if arch.family == "gemma4":
        out.update({k: v.clone() for k, v in expert_views(arch, weights).items()})
    return out


def write_checkpoint(root: Path, arch: Arch, tensors: dict[str, torch.Tensor],
                     config: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(root / "model.safetensors"),
              metadata={"format": "pt"})
    (root / "config.json").write_text(json.dumps(config or arch.config, indent=1), encoding="utf-8")
    resources(root)


def nvfp4_encode(weight: torch.Tensor, input_amax: float, amax: float | None = None):
    """compressed-tensors NVFP4 words for one [N,K] matrix, and its exact dequantization.
    `amax` sets the global divisor when a fused group shares one."""
    w = weight.float()
    n, k = w.shape
    divisor = 448.0 * 6.0 / (amax or float(w.abs().max()))
    blocks = w.reshape(n, k // 16, 16)
    scale = (blocks.abs().amax(-1) / 6.0 * divisor).clamp(min=2.0**-9, max=448.0)
    scale = scale.to(torch.float8_e4m3fn)
    scaled = blocks * divisor / scale.float().unsqueeze(-1)
    magnitude = (scaled.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    codes = magnitude | ((scaled < 0) & (magnitude > 0)).to(torch.int64) << 3
    codes = codes.reshape(n, k).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    dequant = (E2M1[magnitude] * torch.where(scaled < 0, -1.0, 1.0) * scale.float().unsqueeze(-1)
               / divisor).reshape(n, k)
    return {
        "weight_packed": packed.contiguous(),
        "weight_scale": scale.contiguous(),
        "weight_global_scale": torch.tensor([divisor], dtype=torch.float32),
        "input_global_scale": torch.tensor([activation_divisor(input_amax)], dtype=torch.float32),
    }, dequant


def fp8_encode(weight: torch.Tensor):
    w = weight.float()
    scale = (w.abs().amax(1, keepdim=True) / 448.0).to(torch.bfloat16)
    codes = (w / scale.float()).to(torch.float8_e4m3fn)
    return {"weight": codes.contiguous(), "weight_scale": scale.contiguous()}


def fused_group(name: str, names) -> str:
    """The fused group whose matrices share one NVFP4 global scale, as llm-compressor and Model
    Optimizer write them (the published Gemma 4 and Qwen checkpoints do): gate with up, and q, k
    with v where the layer has its own values. A full Gemma layer with attention_k_eq_v keeps
    separate q and k scales, as RedHat's Gemma 4 31B does."""
    for role, group in (("gate_proj", "gate_up"), ("up_proj", "gate_up"), ("q_proj", "qkv"),
                        ("k_proj", "qkv"), ("v_proj", "qkv")):
        suffix = f"{role}.weight"
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if group == "qkv" and stem + "v_proj.weight" not in names:
                return name
            return stem + group
    return name


def input_amax(arch: Arch, weights: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per projection, the largest |input| over calibration text: what llm-compressor measures to
    set input_global_scale (a static divisor too small saturates the W4A4 activation scales)."""
    model = hf_model(arch)
    model.load_state_dict({k: v.float() for k, v in weights.items() if k in model.state_dict()},
                          strict=False)
    if arch.config.get("tie_word_embeddings"):
        model.tie_weights()
    amax: dict[str, float] = {}

    def record(name):
        return lambda module, args: amax.__setitem__(
            name, max(amax.get(name, 0.0), float(args[0].abs().max())))
    hooks = [m.register_forward_pre_hook(record(n)) for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for ids in sequences((256, 256), 3):
            model(input_ids=torch.tensor([ids]), use_cache=False)
    for hook in hooks:
        hook.remove()
    return amax


def activation_divisor(amax: float) -> float:
    return 448.0 * 6.0 / amax


def quantized_checkpoint(root: Path, arch: Arch, weights: dict[str, torch.Tensor], plan,
                         inputs: dict[str, float]) -> None:
    """A compressed-tensors checkpoint: `plan(name)` says nvfp4, fp8 or None (kept) per matrix;
    `inputs` holds the calibrated input amax per projection."""
    out = {}
    tensors = converter_names(arch, weights)
    amax: dict[str, float] = {}
    for name, value in tensors.items():
        if plan(name, value) == "nvfp4":
            group = fused_group(name, tensors)
            amax[group] = max(amax.get(group, 0.0), float(value.float().abs().max()))
    for name, value in tensors.items():
        kind = plan(name, value)
        if kind is None:
            out[name] = value
            continue
        prefix = name[: -len(".weight")]
        encoded = fp8_encode(value) if kind == "fp8" else nvfp4_encode(
            value, inputs.get(prefix, 4.0), amax[fused_group(name, tensors)])[0]
        for part, tensor in encoded.items():
            out[f"{prefix}.{part}"] = tensor
    config = dict(arch.config)
    config["quantization_config"] = {"format": "nvfp4-pack-quantized", "quant_method": "compressed-tensors"}
    write_checkpoint(root, arch, out, config)


# ---------------------------------------------------------------------------------------------
# Cases: one artifact each


@dataclass
class Case:
    name: str
    arch: Arch
    recipe: str
    model: str  # checkpoint directory key: base, nvfp4, fp8...
    sources: dict = field(default_factory=dict)
    components: str = "text"
    profile: str = "artifact"
    # Extra allowance for routes that encode activations (max KL, p99 KL, near-tie margin);
    # BF16-activation cases take their limits from the arithmetic floor alone.
    limits: tuple = (0.0, 0.0)
    margin: float = 0.0
    options: dict = field(default_factory=dict)  # runner options: mode, chunk
    lengths: tuple = (150, 257, 300)
    layers: bool = False  # also compare every layer around the sliding window
    # Under W4A4 the activation encoding's own noise hides the subtler sabotage arms on these
    # models (see a4_floor in the report); the A16 case of the same artifact is the tight check.
    strict_sabotage: bool = True
    drafter: bool = False  # also check one MTP assistant draft step on this target

    @property
    def artifact_key(self) -> str:
        return "-".join([self.arch.name, self.recipe, self.model, self.components,
                         *sorted(f"{k}={v}" for k, v in self.sources.items())])


def cases(archs: dict[str, Arch]) -> list[Case]:
    qd, gd = archs["qwen-dense"], archs["gemma-dense"]
    moe = []
    if "qwen-moe" in archs:  # --moe: the mini MoE models, which today's MoE ops refuse
        moe = [Case("qwen-moe-q456", archs["qwen-moe"], "qwen3_6_35b_a3b", "base",
                    components="text,mtp"),
               Case("gemma-moe-q8", archs["gemma-moe"], "gemma4_26b_a4b_nvfp4", "base")]
    return moe + [
        Case("qwen-dense-bf16", qd, "qwen3_5_bf16", "base"),
        Case("qwen-dense-q8", qd, "qwen3_5_q8", "base"),
        # Calibrated on this model: the int8 activations measure kl_max 4.0e-2, kl_p99 5.6e-3;
        # under --activations exact 4.0e-3 and 1.7e-3, within the floor.
        Case("qwen-dense-q456", qd, "qwen3_6_27b", "base", limits=(8e-2, 1.2e-2)),
        Case("qwen-dense-q456-exact", qd, "qwen3_6_27b", "base",
             options={"activations": "16"}),
        Case("qwen-dense-nvfp4", qd, "qwen3_6_27b_nvfp4", "base", {"quantized": "nvfp4"},
             strict_sabotage=False),
        Case("qwen-dense-nvfp4-a16", qd, "qwen3_6_27b_nvfp4", "base", {"quantized": "nvfp4"},
             profile="a16"),
        Case("gemma-dense-q8", gd, "gemma4_31b_nvfp4", "base", layers=True, drafter=True),
        Case("gemma-dense-q8-decode", gd, "gemma4_31b_nvfp4", "base", options={"mode": "decode"}),
        Case("gemma-dense-q8-chunk48", gd, "gemma4_31b_nvfp4", "base", options={"chunk": 48}),
        Case("gemma-dense-nvfp4", gd, "gemma4_31b_nvfp4", "nvfp4", strict_sabotage=False),
        Case("gemma-dense-nvfp4-a16", gd, "gemma4_31b_nvfp4", "nvfp4", profile="a16"),
    ]


def checkpoints(work: Path, arch: Arch, weights) -> dict:
    """Every checkpoint directory an architecture's cases read."""
    root = work / arch.name
    dirs = {"base": root / "base"}
    write_checkpoint(dirs["base"], arch, converter_names(arch, weights))
    matrices = lambda n, v: (v.dim() == 2 and n.endswith(".weight") and  # noqa: E731
                             "embed_tokens" not in n and n != "lm_head.weight" and
                             v.shape[0] % 128 == 0 and v.shape[1] % 64 == 0)
    inputs = input_amax(arch, weights)
    dirs["nvfp4"] = root / "nvfp4"
    if arch.family == "qwen3_5":
        quantized_checkpoint(dirs["nvfp4"], arch, weights,
                             lambda n, v: "nvfp4" if matrices(n, v) else None, inputs)
    else:
        # Gemma dense: FP8 attention and NVFP4 MLP, the mixture NVIDIA's Gemma 4 31B publishes.
        quantized_checkpoint(dirs["nvfp4"], arch, weights, lambda n, v: (
            None if not matrices(n, v) else "fp8" if ".self_attn." in n else "nvfp4"), inputs)
    dirs["inputs"] = inputs
    return dirs


def convert(native: Path, case: Case, dirs: dict[str, Path], out: Path) -> dict:
    args = [str(native), f"--model={dirs[case.model]}", f"--recipe={case.recipe}", "--quiet",
            "--maintainer-quantize", f"--out={out}"]
    if case.arch.family == "qwen3_5":
        args.append(f"--components={case.components}")
    for role, key in case.sources.items():
        args.append(f"--source={role}={dirs[key]}")
    for stale in out.parent.glob(out.name + "*"):
        stale.unlink()
    result = subprocess.run(args, cwd=REPOSITORY, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{case.name}: ninferno-convert failed\n{result.stderr[:1500]}")
    return json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# The weights an artifact holds, in checkpoint coordinates


def decode_object(artifact: Artifact, obj) -> torch.Tensor:
    payload = artifact.read_object(obj.id)
    n, k = obj.shape
    if obj.format in QUANT_FORMATS:
        return dequantize_row_split(payload, obj.format, obj.shape, dtype=torch.float32)
    if obj.format == "fp8_e4m3fn_row_bf16":
        return dequantize_fp8_row_scaled(payload, obj.shape)
    if obj.format == "nvfp4":
        g = block_scale_geometry("nvfp4", obj.shape, obj.divisors)
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        codes = raw[: g.code_plane_bytes].reshape(n, k // 2).to(torch.int64)
        words = torch.stack([codes & 0xF, codes >> 4], -1).reshape(n, k)
        values = E2M1[words & 7] * torch.where(words >= 8, -1.0, 1.0)
        scales = unswizzle_nvfp4_scales(
            raw[g.scale_plane_offset: g.scale_plane_offset + g.scale_plane_bytes], (n, k))
        scales = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
        divisors = torch.frombuffer(bytearray(payload[g.weight_divisor_offset:]),
                                    dtype=torch.float32)
        rows = divisors.repeat_interleave(n // obj.divisors).unsqueeze(1)
        return values * scales / rows
    raise ValueError(f"{obj.id}: no decoder for {obj.format}")


SOURCE = re.compile(r":((?:model|lm_head|mtp|pre_projection|post_projection)[\w.]*)")


def artifact_weights(path: Path, report: dict, weights: dict[str, torch.Tensor], arch: Arch):
    """The checkpoint tensors with every quantized row replaced by the artifact's own value.

    Returns (tensors, config, stats, encoded): `encoded` maps every checkpoint projection whose
    rows the artifact holds as W4A4-capable NVFP4 (Use policy AllowA4) to True. Rows are matched to their checkpoint rows by cosine; a row
    that matches nothing well, or two rows that match one, is a converter layout error."""
    out = {k: v.float().clone() for k, v in weights.items()}
    out.update(expert_views(arch, out))  # writes to an expert's rows land in its bank
    weights = {**weights, **expert_views(arch, weights)}
    config = dict(arch.config)
    stats: dict[str, list[float]] = {}
    encoded: dict[str, bool] = {}
    with Artifact(path) as artifact:
        bindings = artifact.directory.bindings
        a4 = {use["parameter"] for use in artifact.directory.uses
              if use.get("activation_policy") == "AllowA4"}
        for method in report["methods"]:
            if method["format"] in ("bf16", "fp32", "int32"):
                continue
            obj = artifact.object(method["object"])
            dense = decode_object(artifact, obj)
            k = obj.shape[1]
            for parameter, source in zip(method["parameters"], method["sources"]):
                names = []
                for found in SOURCE.findall(source):
                    for candidate in (found, found + ".weight",
                                      found.replace("model.language_model.", "model.")):
                        if candidate in weights and candidate not in names:
                            names.append(candidate)
                            break
                expert = re.fullmatch(r"text/layers/(\d+)/moe/experts/(\d+)/(gate_up|down)", parameter)
                if not names and expert:  # a stacked bank member names no tensor in its source
                    layer, e, role = expert.groups()
                    roles = ("gate_proj", "up_proj") if role == "gate_up" else ("down_proj",)
                    names = [f"model.layers.{layer}.experts.{e}.{r}.weight" for r in roles]
                if not names:
                    raise RuntimeError(f"{parameter}: no checkpoint tensor in {source!r}")
                parts = bindings[parameter].get("parts") or [
                    {"object": bindings[parameter]["object"], "range": [0, obj.shape[0] * k]}]
                rows = torch.cat([dense.reshape(-1)[a:b] for p in parts
                                  for a, b in [p["range"]] if p["object"] == obj.id]).reshape(-1, k)
                if parameter == "text/output_head" and arch.config.get("tie_word_embeddings"):
                    # The engine's head is its own object; the reference unties it.
                    config["tie_word_embeddings"] = False
                    names = ["model.embed_tokens.weight"]
                    target = "lm_head.weight"
                    if target not in out:
                        out[target] = weights["model.embed_tokens.weight"].float().clone()
                else:
                    target = None
                sources = torch.cat([weights[n].float().reshape(-1, k) for n in names])
                cos = torch.nn.functional.normalize(rows, dim=1) @ torch.nn.functional.normalize(
                    sources, dim=1).T
                best, index = cos.abs().max(1)
                if bool((best < 0.9).any()) or index.unique().numel() != index.numel():
                    raise RuntimeError(f"{parameter}: artifact rows do not map onto {names} "
                                       f"(min cos {float(best.min()):.3f})")
                if parameter == "text/token_embedding" and arch.family == "gemma4":
                    rows = rows / math.sqrt(arch.config["hidden_size"])
                offsets = [0]
                for n in names:
                    offsets.append(offsets[-1] + weights[n].numel() // k)
                for row, source_row in zip(rows, index.tolist()):
                    i = next(j for j in range(len(names)) if source_row < offsets[j + 1])
                    dest = out[target] if target else out[names[i]]
                    dest.view(-1, k)[source_row - offsets[i]] = row
                error = (rows - sources[index]).norm(dim=1) / sources[index].norm(dim=1)
                stats.setdefault(method["format"], []).append(float(error.max()))
                if method["format"] == "nvfp4" and parameter in a4:
                    encoded.update({n: True for n in names})
    return out, config, {f: max(v) for f, v in stats.items()}, encoded


# ---------------------------------------------------------------------------------------------
# Both sides


def sequences(lengths, seed: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (n,), generator=gen).tolist() for n in lengths]


def a16_variant(artifact: Path) -> Path:
    """The same artifact with every AllowA4 Use recorded as A16Only (the directory JSON keeps its
    length, so no offset moves): W4A16, for an engine without an activation-profile switch."""
    out = artifact.with_name(artifact.stem + "-a16.ninferno")
    data = bytearray(artifact.read_bytes())
    _, json_bytes, _ = HEADER.unpack(bytes(data[: HEADER.size]))
    directory = bytes(data[HEADER.size: HEADER.size + json_bytes])
    data[HEADER.size: HEADER.size + json_bytes] = directory.replace(b'"AllowA4"', b'"A16Only"')
    out.write_bytes(bytes(data))
    return out


def run_engine(runner: Path, case: Case, artifact: Path, seqs, work: Path, env=None,
               **options) -> list:
    tokens = work / f"{case.name}.tokens"
    words = [len(seqs)]
    for s in seqs:
        words += [len(s), *s]
    tokens.write_bytes(struct.pack(f"<{len(words)}i", *words))
    out = work / f"{case.name}.logits"
    args = [str(runner), case.arch.family, str(artifact), str(tokens), str(out)]
    args += [f"{k}={v}" for k, v in {"profile": case.profile, **options}.items()]
    result = subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if result.returncode:
        raise RuntimeError(f"{case.name}: runner failed\n{result.stderr[-3000:]}")
    if options.get("mode") == "draft":  # the drafter's dumps are the output
        return []
    flat = torch.frombuffer(bytearray(out.read_bytes()), dtype=torch.float32)
    total = sum(len(s) - 1 for s in seqs)
    logits = flat.reshape(total, -1)
    split, begin = [], 0
    for s in seqs:
        split.append(logits[begin: begin + len(s) - 1])
        begin += len(s) - 1
    return split


def fake_nvfp4(x: torch.Tensor, divisor: float) -> torch.Tensor:
    """NVFP4 activation encoding as W4A4 routes apply it: per 16 inputs an E4M3 scale of
    amax / 6 under the static global divisor, E2M1 values rounded to nearest, saturated."""
    shape, k = x.shape, x.shape[-1]
    blocks = x.float().reshape(-1, k // 16, 16)
    scale = (blocks.abs().amax(-1) / 6.0 * divisor).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    safe = torch.where(scale > 0, scale, 1.0).unsqueeze(-1)
    scaled = (blocks * divisor / safe).clamp(-6.0, 6.0)
    magnitude = E2M1[(scaled.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)]
    values = torch.copysign(magnitude, scaled) * safe / divisor
    return torch.where(scale.unsqueeze(-1) > 0, values, 0.0).reshape(shape).to(x.dtype)


def reference(arch: Arch, tensors: dict, config: dict, seqs, sabotage=None,
              dtype=torch.float32, activations: dict | None = None) -> list:
    """Teacher-forced logits of transformers. `activations` maps a projection's module name to the
    divisor with which its input is NVFP4-encoded first (the engine's W4A4 routes)."""
    model = hf_model(arch, config)
    wanted = model.state_dict().keys()
    missing, _ = model.load_state_dict({k: v for k, v in tensors.items() if k in wanted},
                                       strict=False)
    missing = [m for m in missing if not (m == "lm_head.weight" and config.get("tie_word_embeddings"))]
    if missing:
        raise RuntimeError(f"reference misses {missing[:5]}")
    if config.get("tie_word_embeddings"):
        model.tie_weights()
    model.to(dtype)
    hooks = [model.get_submodule(name).register_forward_pre_hook(
        lambda m, a, d=divisor: (fake_nvfp4(a[0], d),) + a[1:])
        for name, divisor in (activations or {}).items()]
    restore = sabotage(model, arch) if sabotage else None
    try:
        with torch.no_grad():
            return [model(input_ids=torch.tensor([s]), use_cache=False).logits[0, :-1].float()
                    for s in seqs]
    finally:
        if restore:
            restore()
        for hook in hooks:
            hook.remove()


def compare(ref: list, eng: list, margin: float = NEAR_TIE) -> dict:
    """KL(reference || engine) per position, top-1 agreement, and top-1 agreement where the
    reference's top-2 gap is at least `margin` (a smaller gap is a near-tie)."""
    kls, dzs, agree, strict, strict_total, ties = [], [], 0, 0, 0, 0
    for r, e in zip(ref, eng):
        r, e = r.double(), e.double()[:, : r.shape[1]]
        lp, lq = torch.log_softmax(r, -1), torch.log_softmax(e, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        kls.append(kl)
        # The largest log-probability error over tokens the reference gives at least 1e-3.
        dzs.append(torch.where(lp.exp() > 1e-3, (lp - lq).abs(), 0.0).amax(-1))
        top2 = r.topk(2, -1).values
        tie = (top2[:, 0] - top2[:, 1]) < margin
        same = r.argmax(-1) == e.argmax(-1)
        agree += int(same.sum())
        strict += int(same[~tie].sum())
        strict_total += int((~tie).sum())
        ties += int(tie.sum())
    kl = torch.cat(kls)
    return {
        "positions": int(kl.numel()), "kl_max": float(kl.max()), "kl_p99": float(kl.quantile(0.99)),
        "kl_mean": float(kl.mean()), "top1": agree / kl.numel(),
        "top1_strict": strict / max(strict_total, 1), "near_ties": ties, "margin": margin,
        "dz_max": float(torch.cat(dzs).max()), "dz_p99": float(torch.cat(dzs).quantile(0.99)),
        "logit_std": float(torch.cat(ref).std()),
    }


@dataclass
class Limits:
    kl_max: float
    kl_p99: float
    margin: float  # near-tie margin of the top-1 criterion

    def passes(self, m: dict) -> bool:
        return m["top1_strict"] == 1.0 and m["kl_max"] < self.kl_max and m["kl_p99"] < self.kl_p99


def floor_limits(floor: dict, case: "Case") -> Limits:
    """Limits from the arithmetic floor: the same reference run in BF16 against itself in FP32.
    The engine computes in BF16 too, so it may sit as far from FP32 as that run does; FLOOR_MARGIN
    times the floor (never below 1e-4) is its allowance, and a reference top-2 gap below twice the
    floor's 99th-percentile log-probability error is a near-tie. A case whose routes encode activations
    (W4A4, W8A8) adds that encoding error, calibrated on these models (case.limits)."""
    kl_max = max(KL_LIMIT, FLOOR_MARGIN * floor["kl_max"], case.limits[0])
    kl_p99 = max(KL_LIMIT, FLOOR_MARGIN * floor["kl_p99"], case.limits[1])
    return Limits(kl_max, kl_p99, max(NEAR_TIE, 2 * floor["dz_p99"], case.margin))


# ---------------------------------------------------------------------------------------------
# Sabotage arms: the reference changed the way a real bug would change the engine


def permute_heads(model, arch):
    layer = arch.config["layer_types"].index("full_attention")
    proj = model.model.layers[layer].self_attn.q_proj
    heads = arch.config["num_attention_heads"]
    old = proj.weight.data.clone()
    proj.weight.data = torch.cat(list(reversed(old.chunk(heads, 0))))

    def undo():
        proj.weight.data = old
    return undo


def norm_convention(model, arch):
    """(1 + w) where the model means w, or w where it means 1 + w."""
    norm_cls = type(model.model.norm)
    original = norm_cls.forward

    def flipped(self, x):
        if not getattr(self, "with_scale", True):
            return original(self, x)
        w = self.weight.float()
        return (self._norm(x.float()) * (1 + w if arch.family == "gemma4" else w)).type_as(x)
    norm_cls.forward = flipped

    def undo():
        norm_cls.forward = original
    return undo


def doubled_attention_scale(model, arch):
    attentions = [m for m in model.modules() if hasattr(m, "scaling") and hasattr(m, "o_proj")]
    old = [a.scaling for a in attentions]
    for a in attentions:
        a.scaling = a.scaling * 2

    def undo():
        for a, s in zip(attentions, old):
            a.scaling = s
    return undo


def doubled_weight_scale(model, arch):
    """One projection's dequantization scale doubled (layer 0's MLP gate projection: Gemma
    normalizes most projections' outputs, which would hide a doubled down or up projection)."""
    layer = model.model.layers[0]
    mlp = getattr(layer, "mlp")
    target = mlp.shared_expert.gate_proj if hasattr(mlp, "shared_expert") else mlp.gate_proj
    old = target.weight.data.clone()
    target.weight.data = old * 2

    def undo():
        target.weight.data = old
    return undo


SABOTAGE = {"permuted-heads": permute_heads, "norm-convention": norm_convention,
            "attention-scale-x2": doubled_attention_scale, "weight-scale-x2": doubled_weight_scale}


# ---------------------------------------------------------------------------------------------
# Gemma 4 layer by layer: sliding and global attention around the window


def masks(length: int, window: int, global_window: int | None) -> dict:
    """Additive masks per layer type: sliding layers see the last `window` positions (their
    own included, as transformers' sliding mask does); global layers see everything, or only
    `global_window` positions when that sabotage is asked for."""
    i = torch.arange(length)
    distance = i[:, None] - i[None, :]

    def mask(w):
        keep = (distance >= 0) & ((distance < w) if w else True)
        return torch.where(keep, 0.0, torch.finfo(torch.float32).min)[None, None]
    return {"sliding_attention": mask(window), "full_attention": mask(global_window)}


def layer_reference(model, arch: Arch, ids: list[int], global_window=None):
    """Per layer: the attention context (o_proj's input) and the layer's output."""
    contexts, outputs, hooks = {}, {}, []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, i=i: contexts.__setitem__(i, a[0][0].detach().float())))
        hooks.append(layer.register_forward_hook(
            lambda m, a, o, i=i: outputs.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0]
                                                     .detach().float())))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False,
                  attention_mask=masks(len(ids), arch.window, global_window))
    finally:
        for hook in hooks:
            hook.remove()
    return contexts, outputs


def read_dump(path: Path, total: int) -> torch.Tensor:
    """A NINFERNO_GEMMA4_DUMP file: BF16 columns, one per position, appended in pass order."""
    raw = path.read_bytes()
    words = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.int32) & 0xFFFF
    return (words << 16).view(torch.float32).reshape(total, -1)


def relative(e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return (e - r).norm(dim=-1) / r.norm(dim=-1).clamp(min=1e-12)


def layer_check(runner: Path, case: Case, artifact: Path, tensors: dict, config: dict,
                work: Path) -> dict:
    """Every layer's attention context and output at lengths around the sliding window."""
    arch, window = case.arch, case.arch.window
    lengths = [window - 1, window, window + 1, 2 * window + 17, 4 * window]
    gen = torch.Generator().manual_seed(29)
    seqs = [torch.randint(0, VOCAB, (n,), generator=gen).tolist() for n in lengths]
    dump = work / f"{case.name}.dump"
    shutil.rmtree(dump, ignore_errors=True)
    dump.mkdir()
    run_engine(runner, case, artifact, seqs, work, env={"NINFERNO_GEMMA4_DUMP": str(dump)})
    total = sum(lengths)
    model = hf_model(arch, config)
    model.load_state_dict({k: v for k, v in tensors.items() if k in model.state_dict()},
                          strict=False)
    if config.get("tie_word_embeddings"):
        model.tie_weights()
    kinds = arch.config["layer_types"]
    result = {}
    for li, kind in enumerate(kinds):
        engine = {name: read_dump(dump / f"L{li:02d}.{name}.bin", total).split(lengths)
                  for name in ("attn", "x")}
        worst = {"attn": 0.0, "x": 0.0, "attn_beyond_window": 0.0, "windowed_global": 0.0,
                 "unwindowed_sliding": 0.0}
        for s, ids in enumerate(seqs):
            context, output = layer_reference(model, arch, ids)
            attn = relative(engine["attn"][s], context[li])
            worst["attn"] = max(worst["attn"], float(attn.max()))
            worst["x"] = max(worst["x"], float(relative(engine["x"][s], output[li]).max()))
            if len(ids) > window:
                worst["attn_beyond_window"] = max(worst["attn_beyond_window"],
                                                  float(attn[window:].max()))
                # Discrimination: the other window rule, applied to this layer's kind only.
                other = layer_reference(model, arch, ids, global_window=window)[0] \
                    if kind == "full_attention" else None
                if other is not None:
                    worst["windowed_global"] = max(worst["windowed_global"], float(
                        relative(engine["attn"][s], other[li])[window:].max()))
        result[f"L{li}:{kind}"] = worst
    return result


def layer_verdict(result: dict, limit: float) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for layer, w in result.items():
        good = w["attn"] < limit and w["x"] < limit
        caught = "full" not in layer or w["windowed_global"] > 10 * limit
        ok = ok and good and caught
        lines.append(f"  {layer}: attn_rel_max={w['attn']:.2e} (beyond window "
                     f"{w['attn_beyond_window']:.2e}) out_rel_max={w['x']:.2e}"
                     + (f" windowed-global sabotage={w['windowed_global']:.2e}"
                        f" {'caught' if caught else 'MISSED'}" if "full" in layer else ""))
    return ok, lines


# ---------------------------------------------------------------------------------------------
# Gemma 4 MTP assistant, one draft step

# The engine pairs a drafter only with a target of the same RoPE (load_mtp.cpp). A drafter
# whose RoPE differs must be refused at load, and the reference built with it must disagree with
# the engine's draft (the RoPE arm: the draft depends on the RoPE it is given).
OTHER_ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 40000.0},
    "full_attention": {"rope_type": "proportional", "rope_theta": 4000000.0,
                       "partial_rotary_factor": 0.25},
}
DRAFT_COSINE = 0.999


def assistant_config(target: Arch, rope: dict) -> dict:
    t = target.config
    return {
        "architectures": ["Gemma4AssistantForCausalLM"], "model_type": "gemma4_assistant",
        "backbone_hidden_size": t["hidden_size"], "use_ordered_embeddings": False,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text", "hidden_size": 256, "vocab_size": VOCAB,
            "num_hidden_layers": 2, "num_kv_shared_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_attention_heads": t["num_attention_heads"], "head_dim": t["head_dim"],
            "num_key_value_heads": t["num_key_value_heads"],
            "global_head_dim": t["global_head_dim"],
            "num_global_key_value_heads": t["num_global_key_value_heads"],
            "max_position_embeddings": 4096, "intermediate_size": 256,
            "sliding_window": t["sliding_window"], "rms_norm_eps": 1e-06, "rope_parameters": rope,
            "hidden_activation": "gelu_pytorch_tanh", "enable_moe_block": False,
            "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0,
            "attention_bias": False, "attention_dropout": 0.0, "use_double_wide_mlp": False,
        },
    }


def assistant_model(config: dict):
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM

    fields = {k: v for k, v in config.items() if k not in ("architectures", "model_type")}
    cfg = Gemma4AssistantConfig(**fields)
    cfg.get_text_config()._attn_implementation = "eager"
    cfg._attn_implementation = "eager"
    return Gemma4AssistantForCausalLM(cfg).float().eval()


def target_state(model, ids: list[int]):
    """The target's final-normed hidden state at the last position, its argmax, and the keys and
    values of its last sliding and last full layer (what the assistant shares)."""
    shared: dict = {}
    hooks = [layer.self_attn.register_forward_pre_hook(
        lambda m, a, kw: (shared.setdefault("dict", kw["shared_kv_states"]), None)[1],
        with_kwargs=True)
        for layer in model.model.layers]
    try:
        with torch.no_grad():
            out = model.model(input_ids=torch.tensor([ids]), use_cache=False)
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
    finally:
        for hook in hooks:
            hook.remove()
    kv = {k: (v[0].detach().float(), v[1].detach().float()) for k, v in shared["dict"].items()}
    return out.last_hidden_state[0, -1].float(), int(logits.argmax()), kv


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(),
                                                       dim=0))


def convert_assistant(native: Path, arch: Arch, weights: dict, acfg: dict, work: Path,
                      name: str) -> Path:
    root = work / name
    write_checkpoint(root, arch, weights, acfg)
    drafter = work / f"{name}.ninferno"
    for stale in work.glob(f"{name}.ninferno*"):
        stale.unlink()
    result = subprocess.run([str(native), f"--model={root}", "--recipe=gemma4_assistant_q8",
                             "--quiet", f"--out={drafter}"], cwd=REPOSITORY, capture_output=True,
                            text=True)
    if result.returncode:
        raise RuntimeError(f"{name}: ninferno-convert failed\n{result.stderr[:1500]}")
    return drafter


def drafter_check(native: Path, runner: Path, case: Case, target_artifact: Path, tensors: dict,
                  config: dict, work: Path) -> tuple[bool, list[str]]:
    """One MTP draft step on the engine against transformers' Gemma4AssistantForCausalLM: the
    input the drafter reads ([target embedding of the pending token; the target's final-normed
    state]) and the drafter's logits, cos > DRAFT_COSINE."""
    arch = case.arch
    acfg = assistant_config(arch, arch.config["rope_parameters"])
    model = assistant_model(acfg)
    gen = torch.Generator().manual_seed(SEEDS["gemma-dense"] + 100)
    shapes = {n: tuple(t.shape) for n, t in model.state_dict().items() if n != "lm_head.weight"}
    weights = {n: init_tensor(arch, n, shapes[n], gen).to(torch.bfloat16) for n in sorted(shapes)}
    drafter = convert_assistant(native, arch, weights, acfg, work, "assistant")
    report = json.loads(Path(str(drafter) + ".conversion.json").read_text(encoding="utf-8"))
    atensors, aconfig, _, _ = artifact_weights(drafter, report, weights,
                                               Arch("gemma-assistant", "gemma4", False, acfg))

    lengths = [40, arch.window + 37, 3 * arch.window + 5]
    seqs = sequences(lengths, 31)
    dump = work / "assistant.dump"
    shutil.rmtree(dump, ignore_errors=True)
    dump.mkdir()
    run_engine(runner, case, target_artifact, seqs, work, env={"NINFERNO_GEMMA4_DUMP": str(dump)},
               mode="draft", drafter=str(drafter))
    b = arch.config["hidden_size"]
    inputs = read_dump(dump / "D0.input.bin", len(seqs))
    engine_logits = read_dump(dump / "D0.logits.bin", len(seqs))

    target = hf_model(arch, config)
    target.load_state_dict({k: v for k, v in tensors.items() if k in target.state_dict()},
                           strict=False)
    if config.get("tie_word_embeddings"):
        target.tie_weights()
    references = {}
    for label, rope in (("own", arch.config["rope_parameters"]), ("other-rope", OTHER_ROPE)):
        assistant = assistant_model({**aconfig, "text_config": {**acfg["text_config"],
                                                                "rope_parameters": rope}})
        assistant.load_state_dict({k: v for k, v in atensors.items()
                                   if k in assistant.state_dict()}, strict=False)
        references[label] = assistant
    lines, ok = [], True
    for s, ids in enumerate(seqs):
        hidden, token, kv = target_state(target, ids)
        with torch.no_grad():
            table = target.model.embed_tokens(torch.arange(VOCAB)[None])[0].float()
        # The pending token is the engine's own greedy draw; at a near-tie it may differ from the
        # reference's argmax, and the drafter is checked on what the engine actually drafts from.
        drawn = inputs[s][:b]
        pending = int(torch.nn.functional.cosine_similarity(table, drawn[None], dim=-1).argmax())
        embedded = table[pending]
        cos_embed, cos_hidden = cosine(drawn, embedded), cosine(inputs[s][b:], hidden)
        row = {}
        for label, assistant in references.items():
            with torch.no_grad():
                out = assistant(inputs_embeds=torch.cat([embedded, hidden])[None, None],
                                position_ids=torch.tensor([[len(ids)]]), shared_kv_states=kv)
            row[label] = out.logits[0, -1].float()
        cos_own, cos_rope = cosine(engine_logits[s], row["own"]), cosine(engine_logits[s],
                                                                         row["other-rope"])
        same = int(engine_logits[s].argmax()) == int(row["own"].argmax())
        good = min(cos_embed, cos_hidden, cos_own) > DRAFT_COSINE and same
        caught = cos_rope < DRAFT_COSINE
        ok = ok and good and caught
        lines.append(f"  prompt {len(ids)}: pending token {pending} (reference argmax {token}); "
                     f"input cos embedding={cos_embed:.6f} "
                     f"hidden={cos_hidden:.6f}; drafter logits cos={cos_own:.6f} top1 "
                     f"{'same' if same else 'DIFFERS'}; reference with another RoPE "
                     f"cos={cos_rope:.6f} {'caught' if caught else 'MISSED'}")
    # A drafter whose RoPE is not the target's never drafts: the pairing is refused at load.
    other = convert_assistant(native, arch, weights, assistant_config(arch, OTHER_ROPE), work,
                              "assistant-other-rope")
    try:
        run_engine(runner, case, target_artifact, seqs[:1], work, mode="draft", drafter=str(other))
        refused = "loaded"
    except RuntimeError as error:
        refused = "refused" if "rope theta" in str(error) and "differs from the target" in str(error) else f"failed: {error}"
    ok = ok and refused == "refused"
    lines.append(f"  a drafter with another RoPE: {refused}")
    return ok, lines


# ---------------------------------------------------------------------------------------------


def cuda_available() -> bool:
    try:
        return torch.cuda.is_available() or subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True).returncode == 0
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("native")
    parser.add_argument("runner")
    parser.add_argument("repository", nargs="?")
    parser.add_argument("--only", default="")
    parser.add_argument("--keep", default="")
    parser.add_argument("--no-sabotage", action="store_true")
    parser.add_argument("--moe", action="store_true",
                        help="add the mini MoE models (the engine's MoE ops take only the "
                             "production geometries today, so these fail at load)")
    args = parser.parse_args()
    if not cuda_available():
        print("skip: no CUDA device")
        return 77
    native, runner = Path(args.native).resolve(), Path(args.runner).resolve()
    only = {c for c in args.only.split(",") if c}
    archs = {a.name: a for a in (qwen_arch(False), gemma_arch(False))}
    if args.moe:
        archs.update({a.name: a for a in (qwen_arch(True), gemma_arch(True))})
    todo = [c for c in cases(archs) if not only or c.name in only or c.arch.name in only]
    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="ninferno-hf-oracle-"))
    work.mkdir(parents=True, exist_ok=True)
    failures, summary = [], {}
    try:
        weights, dirs = {}, {}
        for arch in {c.arch.name: c.arch for c in todo}.values():
            weights[arch.name] = random_weights(arch, seed=SEEDS[arch.name])
            dirs[arch.name] = checkpoints(work, arch, weights[arch.name])
        converted = {}
        for case in todo:
            try:
                start = time.time()
                arch = case.arch
                if case.artifact_key not in converted:
                    out = work / f"{case.name}.ninferno"
                    report = convert(native, case, dirs[arch.name], out)
                    converted[case.artifact_key] = (out, *artifact_weights(
                        out, report, weights[arch.name], arch))
                out, tensors, config, row_errors, encoded = converted[case.artifact_key]
                seqs = sequences(case.lengths, 5)
                if case.profile == "a16" and arch.family == "qwen3_5":
                    out = a16_variant(out)
                engine = run_engine(runner, case, out, seqs, work, **case.options)
                # Under the artifact's profile the W4A4 routes encode their inputs: the reference
                # applies the same encoding with the checkpoint's calibrated divisors.
                activations = {} if case.profile == "a16" else {
                    name[: -len(".weight")]: activation_divisor(
                        dirs[arch.name]["inputs"][name[: -len(".weight")]]) for name in encoded}
                ref = reference(arch, tensors, config, seqs, activations=activations)
                floor = compare(ref, reference(arch, tensors, config, seqs, dtype=torch.bfloat16,
                                               activations=activations))
                limits = floor_limits(floor, case)
                m = compare(ref, engine, limits.margin)
                m["row_error"], m["floor"], m["limits"] = row_errors, floor, limits.__dict__
                ok = limits.passes(m)
                print(f"{case.name}: {'ok' if ok else 'FAILED'} positions={m['positions']} "
                      f"top1={m['top1']:.4f} top1_strict={m['top1_strict']:.4f} ties={m['near_ties']} "
                      f"kl_max={m['kl_max']:.3e} kl_p99={m['kl_p99']:.3e} kl_mean={m['kl_mean']:.3e} "
                      f"dz_max={m['dz_max']:.3f} logit_std={m['logit_std']:.2f} "
                      f"({time.time() - start:.1f}s)\n"
                      f"  bf16 floor: kl_max={floor['kl_max']:.3e} kl_p99={floor['kl_p99']:.3e} "
                      f"kl_mean={floor['kl_mean']:.3e} dz_max={floor['dz_max']:.3f} -> limits "
                      f"kl_max<{limits.kl_max:.2e} kl_p99<{limits.kl_p99:.2e} "
                      f"near-tie<{limits.margin:.3f}\n"
                      f"  artifact rows vs checkpoint (max rel): {json.dumps(row_errors)}"
                      + (f"\n  W4A4 inputs encoded in the reference: {len(activations)} projections; "
                         f"without that encoding kl_max="
                         f"{compare(reference(arch, tensors, config, seqs), engine)['kl_max']:.3e}"
                         if activations else ""), flush=True)
                if not ok:
                    failures.append(case.name)
                m["sabotage"] = {}
                if not args.no_sabotage:
                    for arm, fn in SABOTAGE.items():
                        s = compare(reference(arch, tensors, config, seqs, fn,
                                              activations=activations), engine, limits.margin)
                        caught = not limits.passes(s)
                        m["sabotage"][arm] = s
                        verdict = "caught" if caught else (
                            "MISSED" if case.strict_sabotage else "within the W4A4 noise")
                        print(f"  sabotage {arm}: {verdict} "
                              f"top1_strict={s['top1_strict']:.4f} kl_max={s['kl_max']:.3e} "
                              f"kl_p99={s['kl_p99']:.3e} kl_mean={s['kl_mean']:.3e}", flush=True)
                        if not caught and case.strict_sabotage:
                            failures.append(f"{case.name}/{arm}")
                if case.layers:
                    m["layers"] = layer_check(runner, case, out, tensors, config, work)
                    good, lines = layer_verdict(m["layers"], LAYER_LIMIT)
                    print(f"  layers around the window ({arch.window}): {'ok' if good else 'FAILED'}")
                    print("\n".join(lines), flush=True)
                    if not good:
                        failures.append(f"{case.name}/layers")
                if case.drafter:
                    good, lines = drafter_check(native, runner, case, out, tensors, config, work)
                    print(f"  MTP assistant, one draft step: {'ok' if good else 'FAILED'}")
                    print("\n".join(lines), flush=True)
                    if not good:
                        failures.append(f"{case.name}/drafter")
                summary[case.name] = m
            except Exception as error:  # one broken case must not hide the others
                print(f"{case.name}: FAILED with {type(error).__name__}: {error}", flush=True)
                failures.append(case.name)
        (work / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
