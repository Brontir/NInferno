"""Gemma 4 architecture adapter: mathematical config, source axes, and packing groups.

Dense (31B) and MoE (26B-A4B) text share this explicit mapping. Official checkpoint names
and quantization assignments live in official_recipes.

Three Gemma 4 facts decide the shape of everything below and are stated once here.

  * Layer types are `sliding_attention` (window 1024) and `full_attention`, and they carry
    *different* head geometry: `head_dim` for sliding layers, `global_head_dim` for full ones,
    with `num_key_value_heads` and `num_global_key_value_heads` to match.
  * `attention_k_eq_v` makes full-attention layers reuse the key projection as the value
    projection, so those layers carry no `v_proj` tensor at all. Keys and values still differ
    after their norms (K takes `k_norm` and RoPE, V takes an unweighted `v_norm` and no RoPE),
    so this saves a weight matrix, never a cache plane.
  * A MoE layer runs its dense MLP and its expert bank *in parallel* from one input and adds
    the two results. It is not a shared expert; the dense branch has its own width and its own
    pair of norms.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Mapping

import torch

from .config_checks import check_source_fields as _check_source_fields
from .config_checks import f32 as _f32
from .config_checks import fixed as _fixed
from .config_checks import has_model_config as _has_model_config
from .config_checks import positive as _positive
from .model import Model, Parameter
from .resources import load_resources
from .sources.logical import LogicalSource, select_rows, transpose_source
from .sources.safetensors import SafetensorsSource, tensor_source
from .sources.compressed_tensors import (
    divisor_scaled_source,
    matrix_source,
    nvfp4_layout,
    stacked_matrix_source,
)


def _divisor(word: bytes) -> float:
    return struct.unpack("<f", word)[0]


def _concat_values(parts, shape, label):
    """Row-concatenate value sources when there is no encoded representation to preserve."""
    k = shape[1]

    def read(begin: int, end: int) -> torch.Tensor:
        pieces, cursor = [], 0
        for part in parts:
            count = part.shape[0] * k
            low, high = max(begin, cursor), min(end, cursor + count)
            if low < high:
                pieces.append(part.values(low - cursor, high - cursor))
            cursor += count
        return torch.cat(pieces) if len(pieces) != 1 else pieces[0]

    return LogicalSource(shape, f"concat({label}, {len(parts)} matrices)", read)

_TEXT_ARCHITECTURES = (
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
)
# Gemma 4 12B ships as the "unified" family (text, vision and audio towers in one checkpoint); its
# text decoder is the Gemma 4 text model under another model_type.
TEXT_MODEL_TYPES = ("gemma4_text", "gemma4_unified_text")


def text_model_type(raw: dict, label: str) -> None:
    if raw.get("model_type") not in TEXT_MODEL_TYPES:
        raise ValueError(
            f"{label}.model_type: expected gemma4_text or gemma4_unified_text, got "
            f"{raw.get('model_type')!r}"
        )

# Both layer types Gemma 4 text uses. `linear_attention` belongs to Qwen and never appears here.
_LAYER_TYPES = ("sliding_attention", "full_attention")


def _rope_group(raw: dict, layer_type: str, head_dim: int, label: str) -> dict:
    """One RoPE description per layer type, resolved against that type's head width.

    Gemma 4 publishes `rope_parameters` keyed by layer type. `full_attention` uses
    `rope_type: proportional`, whose rotary width is `int(partial_rotary_factor*head_dim//2)`
    *pairs* out of `head_dim//2`, and whose pairs are `(i, i+head_dim/2)` — the `rotate_half`
    convention, not a contiguous leading `rotary_dim`. The remaining pairs carry a zero inverse
    frequency, which is the identity. Recording `rotary_pairs` explicitly keeps that decision in
    the artifact instead of in a reader's assumption.
    """
    group = raw.get(layer_type)
    if not isinstance(group, dict):
        raise ValueError(f"{label}: rope_parameters must describe {layer_type}")
    rope_type = group.get("rope_type", "default")
    if rope_type not in ("default", "proportional"):
        raise ValueError(f"{label}.{layer_type}: unsupported rope_type {rope_type!r}")
    if _f32(group.get("factor", 1.0), f"{label}.{layer_type}.factor") != 1.0:
        raise ValueError(f"{label}.{layer_type}: scaled RoPE is not implemented")
    theta = _f32(group.get("rope_theta"), f"{label}.{layer_type}.rope_theta")
    factor = group.get("partial_rotary_factor", 1.0)
    if rope_type == "default" and "partial_rotary_factor" in group:
        raise ValueError(f"{label}.{layer_type}: default RoPE takes no partial factor")
    proportion = _f32(factor, f"{label}.{layer_type}.partial_rotary_factor")
    if proportion > 1.0:
        raise ValueError(f"{label}.{layer_type}: partial_rotary_factor exceeds one")
    if head_dim % 2:
        raise ValueError(f"{label}.{layer_type}: head_dim must be even")
    pairs = int(proportion * head_dim // 2)
    if pairs <= 0:
        raise ValueError(f"{label}.{layer_type}: rotary width is empty")
    return {
        "rope_type": rope_type,
        "rope_theta": theta,
        "partial_rotary_factor": proportion,
        "rotary_pairs": pairs,
        "head_dim": head_dim,
    }


def text_config(source: dict) -> dict:
    architectures = source.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in _TEXT_ARCHITECTURES
    ):
        raise ValueError(f"unsupported Gemma text architecture {architectures!r}")
    raw = source.get("text_config", source)
    text_model_type(raw, "text")
    _fixed(raw, "hidden_activation", "gelu_pytorch_tanh", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "attention_dropout", 0.0, "text")
    # The engine implements one resident text stream. Per-layer embeddings (PLE) and KV sharing
    # are separate mathematics; refuse them here rather than convert something that cannot run.
    _fixed(raw, "hidden_size_per_layer_input", 0, "text")
    _fixed(raw, "num_kv_shared_layers", 0, "text")
    _fixed(raw, "use_double_wide_mlp", False, "text")
    if raw.get("use_bidirectional_attention") not in (None, "vision"):
        raise ValueError("text.use_bidirectional_attention: only 'vision' is implemented")

    result = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
    }
    for key in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings"):
        result[key] = _positive(raw.get(key), "text." + key)

    tied = raw.get("tie_word_embeddings", source.get("tie_word_embeddings", False))
    if type(tied) is not bool:
        raise ValueError("text.tie_word_embeddings must be boolean")
    result["tie_word_embeddings"] = tied
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")

    layers = raw.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(kind not in _LAYER_TYPES for kind in layers)
    ):
        raise ValueError("text.layer_types must cover every text block")
    # Gemma 4 requires the last block to be global. The reference implementation rewrites a
    # trailing sliding layer to full_attention and only warns; converting the config as written
    # would build an artifact whose topology differs from the model everyone else runs.
    if layers[-1] != "full_attention":
        raise ValueError("text.layer_types: the last Gemma 4 block must be full_attention")
    result["layer_types"] = list(layers)

    heads = _positive(raw.get("num_attention_heads"), "text.num_attention_heads")
    result["num_attention_heads"] = heads
    k_eq_v = raw.get("attention_k_eq_v", False)
    if type(k_eq_v) is not bool:
        raise ValueError("text.attention_k_eq_v must be boolean")
    result["attention_k_eq_v"] = k_eq_v

    geometry = {}
    if "sliding_attention" in layers:
        result["sliding_window"] = _positive(raw.get("sliding_window"), "text.sliding_window")
        geometry["sliding_attention"] = (
            _positive(raw.get("head_dim"), "text.head_dim"),
            _positive(raw.get("num_key_value_heads"), "text.num_key_value_heads"),
        )
    if "full_attention" in layers:
        geometry["full_attention"] = (
            _positive(raw.get("global_head_dim"), "text.global_head_dim"),
            _positive(
                raw.get("num_global_key_value_heads")
                if k_eq_v
                else raw.get("num_key_value_heads"),
                "text.num_global_key_value_heads",
            ),
        )
    rope_source = raw.get("rope_parameters")
    if not isinstance(rope_source, dict):
        raise ValueError("text.rope_parameters must be an object keyed by layer type")
    result["attention_geometry"] = {}
    result["rope_parameters"] = {}
    for layer_type, (head_dim, kv_heads) in geometry.items():
        if heads % kv_heads:
            raise ValueError(
                f"text.{layer_type}: attention query heads must be divisible by KV heads"
            )
        result["attention_geometry"][layer_type] = {
            "head_dim": head_dim,
            "num_key_value_heads": kv_heads,
        }
        result["rope_parameters"][layer_type] = _rope_group(
            rope_source, layer_type, head_dim, "text.rope_parameters"
        )

    # Gemma 4 folds the attention scale into the learned query norm and leaves the softmax
    # scale at one. Reading the checkpoint as if it were 1/sqrt(head_dim) is a silent error.
    result["attention_scale"] = 1.0

    softcap = raw.get("final_logit_softcapping")
    if softcap is not None:
        result["final_logit_softcapping"] = _f32(softcap, "text.final_logit_softcapping")

    result["intermediate_size"] = _positive(raw.get("intermediate_size"), "text.intermediate_size")
    moe = raw.get("enable_moe_block", False)
    if type(moe) is not bool:
        raise ValueError("text.enable_moe_block must be boolean")
    if moe:
        result["num_experts"] = _positive(raw.get("num_experts"), "text.num_experts")
        result["num_experts_per_tok"] = _positive(raw.get("top_k_experts"), "text.top_k_experts")
        result["moe_intermediate_size"] = _positive(
            raw.get("moe_intermediate_size"), "text.moe_intermediate_size"
        )
        if result["num_experts_per_tok"] > result["num_experts"]:
            raise ValueError("selected experts exceed expert count")
    elif raw.get("num_experts"):
        raise ValueError("text: Dense config carries MoE fields")

    # Applied to the embedding table at conversion time, never at runtime. The output head reads
    # the same source tensor through its own logical parameter and is deliberately left unscaled.
    # Gemma's reference normalizer is sqrt(hidden_size) in the embedding's BF16 compute dtype
    # (transformers: embed_scale.to(weight.dtype); 53.0 for H=2816, not 53.066).
    root = torch.tensor(float(result["hidden_size"]) ** 0.5, dtype=torch.float32)
    result["embedding_scale"] = _f32(
        root.to(torch.bfloat16).float().item(), "text.embedding_scale"
    )
    return result


def _scaled_source(source: LogicalSource, factor: float) -> LogicalSource:
    """Multiply a logical source by a constant, keeping the read lazy and range-addressed."""
    if factor == 1.0:
        return source
    return LogicalSource(
        source.shape,
        f"scale({source.label},{factor:.9g})",
        lambda begin, end: source.values(begin, end).to(torch.float32) * factor,
    )


class _Builder:
    def __init__(self, model: Model):
        self.model = model

    def validate_source(self, selected, original, name):
        """An additional source must describe the same mathematics at the site it supplies."""
        if selected is original or not _has_model_config(selected):
            return
        actual, expected = text_config(selected.config), self.model.config
        fields = {"hidden_size", "num_hidden_layers", "attention_k_eq_v"}
        if name.endswith(("token_embedding", "output_head")):
            fields.add("vocab_size")
        if name == "text/output_head":
            fields.add("tie_word_embeddings")
        if "/mlp/" in name:
            fields.add("intermediate_size")
        if "/moe/experts/" in name:
            fields.update(("num_experts", "moe_intermediate_size"))
        if name.endswith("/moe/router"):
            fields.add("num_experts")
        _check_source_fields(name, actual, expected, fields)
        if "/layers/" in name:
            index = int(name.split("/layers/", 1)[1].split("/", 1)[0])
            if index >= len(actual["layer_types"]) or (
                actual["layer_types"][index] != expected["layer_types"][index]
            ):
                raise ValueError(
                    f"{name}: additional source layer topology differs; "
                    "provide an explicit logical source"
                )
            if "/attention/" in name:
                kind = expected["layer_types"][index]
                if actual["attention_geometry"].get(kind) != expected["attention_geometry"][kind]:
                    raise ValueError(
                        f"{name}: additional source attention geometry differs; "
                        "provide an explicit logical source"
                    )

    def add(
        self,
        name,
        store,
        source_name,
        shape,
        *,
        inputs=(),
        direct="bf16",
        source_shape=None,
        offset=0,
        rows=None,
        transpose=None,
        scale=None,
        stack=None,
        compensate=None,
    ):
        shape = tuple(shape)
        original = shape if source_shape is None else tuple(source_shape)

        def factory(selected, format=None):
            self.validate_source(selected, store, name)
            selected_name = source_name
            if name.startswith("text/") and _has_model_config(selected):
                if "text_config" not in selected.config:
                    selected_name = selected_name.replace("model.language_model.", "model.", 1)
                elif "text_config" not in store.config and selected_name.startswith("model."):
                    selected_name = selected_name.replace(
                        "model.", "model.language_model.", 1
                    )
            if compensate is not None:
                # This expert's down projection, read against its divisor times d_up / d_gate:
                # the stacked gate_up plane kept gate's divisor for both halves, so its up rows
                # come out multiplied by that ratio, and down is where the product ends.
                ref = matrix_source(selected, selected_name, original, format)
                gate_name, up_name, pair_shape = compensate
                if selected_name.startswith("model.") and "language_model" not in selected_name:
                    gate_name = gate_name.replace("model.language_model.", "model.", 1)
                    up_name = up_name.replace("model.language_model.", "model.", 1)
                encoded = all(
                    nvfp4_layout(selected, n.removesuffix(".weight")) is not None
                    for n in (selected_name, gate_name, up_name)
                )
                if not encoded:
                    return ref
                gate = matrix_source(selected, gate_name, pair_shape, "nvfp4")
                up = matrix_source(selected, up_name, pair_shape, "nvfp4")

                def ratio() -> float:
                    return _divisor(up.weight_divisor()) / _divisor(gate.weight_divisor())

                return divisor_scaled_source(ref, ratio, name)
            if transpose is not None:
                if format is not None:
                    raise ValueError(f"{name}: transpose source requires value access")
                ref = tensor_source(selected, selected_name, original)
                return transpose_source(ref, transpose, shape)
            if stack is not None:
                members = tuple(
                    part.replace("model.language_model.", "model.", 1)
                    if selected_name.startswith("model.") and "language_model" not in selected_name
                    else part
                    for part in stack
                )
                rows_each = shape[0] // len(members)
                parts = [
                    matrix_source(selected, part, (rows_each, shape[1]), format)
                    for part in members
                ]
                if format is None and not all(
                    nvfp4_layout(selected, part.removesuffix(".weight")) is not None
                    for part in members
                ):
                    # Nothing encoded to preserve: read the pair as plain values and concatenate.
                    return _concat_values(parts, shape, name)
                return stacked_matrix_source(parts, name, adopt_first=True)
            if scale is not None:
                if format is not None:
                    raise ValueError(f"{name}: scaled source requires value access")
                return _scaled_source(tensor_source(selected, selected_name, shape), scale)
            if len(original) == 2 and offset == 0:
                ref = matrix_source(selected, selected_name, original, format)
                return select_rows(ref, rows) if rows is not None else ref
            if format is not None:
                raise ValueError(f"{name}: provide an explicit encoded source mapping")
            return tensor_source(
                selected, selected_name, shape, offset=offset, source_shape=original
            )

        self.model.add(
            Parameter(
                name,
                shape,
                factory(store),
                factory,
                tuple(inputs),
                direct,
                residency=name.split("/", 1)[0],
            )
        )

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def attention(self, prefix, source_prefix, store, config, layer_type):
        h = config["hidden_size"]
        geometry = config["attention_geometry"][layer_type]
        d, kv = geometry["head_dim"], geometry["num_key_value_heads"]
        q = config["num_attention_heads"] * d
        k = kv * d
        # A full-attention layer under attention_k_eq_v publishes no v_proj: the value stream is
        # the key projection read through an unweighted v_norm. Emitting a value parameter here
        # would invent a tensor the checkpoint does not contain.
        shares_value = config["attention_k_eq_v"] and layer_type == "full_attention"
        roles = [("query", "q_proj", q), ("key", "k_proj", k)]
        if not shares_value:
            roles.append(("value", "v_proj", k))
        for role, field, rows in roles:
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn." + field + ".weight",
                (rows, h),
                inputs=(prefix + "mixer_input",),
            )
        for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn." + field + ".weight",
                (d,),
            )
        self.add(
            prefix + "attention/output",
            store,
            source_prefix + "self_attn.o_proj.weight",
            (h, q),
            inputs=(prefix + "attention/context",),
        )
        packed = [prefix + "attention/" + role for role, _, _ in roles]
        self.group(*packed)
        if len(packed) == 3:
            self.group(*packed[:2])
            self.group(*packed[1:])

    def dense(self, prefix, source_prefix, store, h, intermediate):
        for role in ("gate", "up"):
            self.add(
                prefix + "mlp/" + role,
                store,
                source_prefix + "mlp." + role + "_proj.weight",
                (intermediate, h),
                inputs=(prefix + "ffn_input",),
            )
        self.add(
            prefix + "mlp/down",
            store,
            source_prefix + "mlp.down_proj.weight",
            (h, intermediate),
            inputs=(prefix + "mlp/product",),
        )
        self.group(prefix + "mlp/gate", prefix + "mlp/up")

    def moe(self, prefix, source_prefix, store, config):
        h, e = config["hidden_size"], config["num_experts"]
        ir = config["moe_intermediate_size"]
        p = prefix + "moe/"
        self.add(
            p + "router",
            store,
            source_prefix + "router.proj.weight",
            (e, h),
            inputs=(p + "router_input",),
        )
        # Two scales the Qwen router has no counterpart for: a per-width vector applied before
        # the projection and a per-expert vector applied to the normalized top-k weights.
        self.add(p + "router_scale", store, source_prefix + "router.scale", (h,), direct="fp32")
        self.add(
            p + "expert_scale",
            store,
            source_prefix + "router.per_expert_scale",
            (e,),
            direct="fp32",
        )
        gate_up, downs = [], []
        for expert in range(e):
            ep = p + f"experts/{expert}/"
            # Gemma 4 publishes gate and up as separate matrices with separate NVFP4 divisors,
            # but one expert is one logical matrix here: the reference implementation's own
            # parameter is a fused `gate_up_proj [E, 2I, H]`, and the Gemma 4 MoE kernels take
            # one divisor per expert. The pair usually shares one divisor (RedHat: every
            # expert; NVIDIA: 3834 of 3840), and where it does not — ratios up to 1.46 — a fold
            # onto one divisor would either overflow E4M3 or round every block scale. So the
            # plane keeps gate's divisor and every code and scale untouched; its up rows come out
            # scaled by d_up/d_gate, which GeGLU carries linearly into down, and this expert's
            # down divides it back out exactly (divisor_scaled_source).
            self.add(
                ep + "gate_up",
                store,
                source_prefix + f"experts.{expert}.gate_proj.weight",
                (2 * ir, h),
                inputs=(p + "expert_input",),
                stack=(
                    source_prefix + f"experts.{expert}.gate_proj.weight",
                    source_prefix + f"experts.{expert}.up_proj.weight",
                ),
            )
            gate_up.append(ep + "gate_up")
            self.add(
                ep + "down",
                store,
                source_prefix + f"experts.{expert}.down_proj.weight",
                (h, ir),
                inputs=(ep + "product",),
                compensate=(
                    source_prefix + f"experts.{expert}.gate_proj.weight",
                    source_prefix + f"experts.{expert}.up_proj.weight",
                    (ir, h),
                ),
            )
            downs.append(ep + "down")
        self.group(*gate_up)
        self.group(*downs)

    def block(self, prefix, source_prefix, store, config, layer_type):
        h = config["hidden_size"]
        moe = "num_experts" in config
        self.add(prefix + "input_norm", store, source_prefix + "input_layernorm.weight", (h,))
        self.attention(prefix, source_prefix, store, config, layer_type)
        self.add(
            prefix + "post_attention_norm",
            store,
            source_prefix + "post_attention_layernorm.weight",
            (h,),
        )
        self.add(
            prefix + "pre_ffn_norm",
            store,
            source_prefix + "pre_feedforward_layernorm.weight",
            (h,),
        )
        self.dense(prefix, source_prefix, store, h, config["intermediate_size"])
        self.add(
            prefix + "post_ffn_norm",
            store,
            source_prefix + "post_feedforward_layernorm.weight",
            (h,),
        )
        if moe:
            for role, field in (
                ("pre_moe_norm", "pre_feedforward_layernorm_2"),
                ("post_mlp_norm", "post_feedforward_layernorm_1"),
                ("post_moe_norm", "post_feedforward_layernorm_2"),
            ):
                self.add(prefix + role, store, source_prefix + field + ".weight", (h,))
            self.moe(prefix, source_prefix, store, config)
        # Applied to the whole layer output, residual included, so it folds into no weight.
        self.add(prefix + "output_scale", store, source_prefix + "layer_scalar", (1,), direct="fp32")


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if selected != {"text"}:
        raise ValueError("Gemma 4 onboarding implements the text component only")
    config = text_config(base.config)
    records = {"text": {"config": config}}
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        vision_config=None,
        overrides=resource_overrides,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Builder(model)
    h, r = config["hidden_size"], config["vocab_size"]
    text_prefix = "model.language_model." if "text_config" in base.config else "model."

    # Gemma scales the embedding lookup by sqrt(hidden_size). Folding it into the table keeps the
    # runtime free of a whole-hidden-state pass; the tied output head reads the same bytes through
    # its own parameter and must stay unscaled, which is why these are two `add` calls and not one.
    builder.add(
        "text/token_embedding",
        base,
        text_prefix + "embed_tokens.weight",
        (r, h),
        scale=config["embedding_scale"],
    )
    head_source = (
        text_prefix + "embed_tokens.weight"
        if config["tie_word_embeddings"]
        else "lm_head.weight"
    )
    builder.add(
        "text/output_head", base, head_source, (r, h), inputs=("text/final_hidden",)
    )
    builder.add("text/final_norm", base, text_prefix + "norm.weight", (h,))
    for i, kind in enumerate(config["layer_types"]):
        builder.block(f"text/layers/{i}/", text_prefix + f"layers.{i}.", base, config, kind)
    return model


_PROJECTION_LEAF = {
    "attention/query": "self_attn.q_proj",
    "attention/key": "self_attn.k_proj",
    "attention/value": "self_attn.v_proj",
    "attention/output": "self_attn.o_proj",
    "mlp/gate": "mlp.gate_proj",
    "mlp/up": "mlp.up_proj",
    "mlp/down": "mlp.down_proj",
    "moe/router": "router.proj",
}


def checkpoint_prefix(parameter: str) -> str | None:
    """Checkpoint prefix of a text-layer projection, without the leading `model.` root.

    Stated once here rather than carried alongside the model, because the artifact's component
    schema accepts exactly `config`, `target`, `resources` and `proposal` and refuses anything
    else — a mapping smuggled in beside them fails at the writer, two hours in.
    """
    if not parameter.startswith("text/layers/"):
        return None
    index, _, role = parameter[len("text/layers/") :].partition("/")
    if role in _PROJECTION_LEAF:
        return f"layers.{index}.{_PROJECTION_LEAF[role]}"
    if role.startswith("moe/experts/"):
        expert, _, leaf = role[len("moe/experts/") :].partition("/")
        # A fused gate_up parameter is probed through its gate half: the two are published as
        # separate matrices and always carry the same representation.
        leaf = "gate_proj" if leaf == "gate_up" else f"{leaf}_proj"
        return f"layers.{index}.experts.{expert}.{leaf}"
    return None


def encoded_sites(model: Model, store: SafetensorsSource) -> dict[str, str]:
    """Report, per logical parameter, which encoded representation the checkpoint carries.

    Returns only the sites that are actually encoded in this file. The published Gemma 4 NVFP4
    checkpoints disagree about coverage — one publisher quantizes the expert bank alone, another
    the attention projections as well — so a recipe that decides from the model's name rather
    than from the file will silently dequantize half of it. Both NVFP4 field layouts count:
    compressed-tensors (RedHat, Unsloth) and NVIDIA Model Optimizer.
    """
    found: dict[str, str] = {}
    for parameter, item in model.parameters.items():
        if not item.projection:
            continue
        leaf = checkpoint_prefix(parameter)
        if leaf is None:
            continue
        for root in ("model.language_model.", "model."):
            prefix = root + leaf
            if nvfp4_layout(store, prefix) is not None:
                found[parameter] = "nvfp4"
                break
            if store.has(prefix + ".weight") and store.describe(prefix + ".weight").dtype == "F8_E4M3":
                found[parameter] = "fp8_e4m3fn_row_bf16"
                break
    return found
