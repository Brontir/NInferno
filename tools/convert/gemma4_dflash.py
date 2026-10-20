"""Gemma 4 DFlash drafter adapter: z-lab's block-diffusion drafter as its own artifact.

z-lab publishes one drafter per Gemma 4 model (`z-lab/gemma-4-*-it-DFlash`, `DFlashDraftModel`,
declared as a Qwen3 stack). It owns no embedding and no head: it embeds with the target's
(scaled) table and scores with the target's tied head. One round:

    context  = context_norm(feature_projection([h_l for l in target_layer_ids]))   per token
    K/V      = per layer: key/value(context), key_norm on K, RoPE at the token's position
    x        = target_embedding([anchor, mask x (block-1)])   at positions P..P+block-1
    x        = layers(x)   queries over [cached context K/V ; the block's own K/V]
    drafts   = argmax target_head(final_norm(x[:, 1:]))

where h_l is the target's residual stream after decoder layer l. Sliding layers are causal
inside the block and see a window of keys; the full layer sees the whole block. The artifact is
a separate file; the engine binds it next to its target and checks the pairing.
"""

from __future__ import annotations

from .config_checks import f32 as _f32, fixed as _fixed, positive as _positive
from .gemma4 import _Builder
from .model import Model
from .resources import load_resources
from .sources.safetensors import SafetensorsSource


def dflash_config(source: dict) -> dict:
    if source.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("expected a DFlashDraftModel checkpoint")
    _fixed(source, "model_type", "qwen3", "dflash")
    _fixed(source, "hidden_act", "silu", "dflash")
    _fixed(source, "attention_bias", False, "dflash")
    # tie_word_embeddings differs between the published drafters (False for 26B, True for 31B) and
    # means nothing here: neither ships an embedding or a head; build_model checks that.
    if source.get("final_logit_softcapping") is None:
        # The Gemma drafters score with the target's soft-capped head; a Qwen drafter has none.
        raise ValueError("dflash: final_logit_softcapping is missing; not a Gemma 4 drafter")
    for key in ("sample_from_anchor", "input_embedding_scale", "output_multiplier"):
        if key in source or key in source.get("dflash_config", {}):
            raise ValueError(f"dflash: {key} is a DFlash2 field this adapter does not implement")
    layers = _positive(source.get("num_hidden_layers"), "dflash.num_hidden_layers")
    kinds = source.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != layers:
        raise ValueError("dflash: layer_types must cover every layer")
    if any(kind not in ("sliding_attention", "full_attention") for kind in kinds):
        raise ValueError("dflash: unknown layer type")
    if source.get("rope_scaling") not in (None, {}):
        raise ValueError("dflash: rope_scaling is not implemented")
    heads = _positive(source.get("num_attention_heads"), "dflash.num_attention_heads")
    kv_heads = _positive(source.get("num_key_value_heads"), "dflash.num_key_value_heads")
    if heads % kv_heads:
        raise ValueError("dflash: query heads must divide by key/value heads")
    draft = source.get("dflash_config", {})
    taps = draft.get("target_layer_ids")
    target_layers = _positive(source.get("num_target_layers"), "dflash.num_target_layers")
    if (
        not isinstance(taps, list)
        or not taps
        or any(type(i) is not int or not 0 <= i < target_layers for i in taps)
        or len(set(taps)) != len(taps)
    ):
        raise ValueError("dflash: invalid target_layer_ids")
    vocab = _positive(source.get("vocab_size"), "dflash.vocab_size")
    mask = draft.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < vocab:
        raise ValueError("dflash: mask token outside the vocabulary")
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "gemma4_dflash",
        "hidden_size": _positive(source.get("hidden_size"), "dflash.hidden_size"),
        "vocab_size": vocab,
        "num_hidden_layers": layers,
        "num_target_layers": target_layers,
        "intermediate_size": _positive(source.get("intermediate_size"), "dflash.intermediate_size"),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": _positive(source.get("head_dim"), "dflash.head_dim"),
        "layer_types": list(kinds),
        "sliding_window": _positive(source.get("sliding_window"), "dflash.sliding_window"),
        "rope_theta": _f32(_rope_theta(source), "dflash.rope_theta"),
        "rms_norm_eps": _f32(source.get("rms_norm_eps", 1e-6), "dflash.rms_norm_eps"),
        "block_size": _positive(source.get("block_size"), "dflash.block_size"),
        "final_logit_softcapping": _f32(
            source["final_logit_softcapping"], "dflash.final_logit_softcapping"
        ),
        "dflash_config": {"target_layer_ids": list(taps), "mask_token_id": mask},
    }


def _rope_theta(source: dict):
    """The rotary base: top-level rope_theta, or rope_parameters.rope_theta in newer configs."""
    theta = source.get("rope_theta")
    rope = source.get("rope_parameters")
    if theta is None and isinstance(rope, dict):
        if rope.get("rope_type", "default") != "default":
            raise ValueError("dflash.rope_parameters.rope_type must be default")
        theta = rope.get("rope_theta")
    return theta


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions=None,
    resource_overrides=None,
) -> Model:
    if set(components) != {"text"}:
        raise ValueError("a Gemma 4 DFlash artifact holds the drafter alone")
    config = dflash_config(base.config)
    for name in ("embed_tokens.weight", "lm_head.weight", "model.embed_tokens.weight"):
        if base.has(name):
            raise ValueError(f"dflash: {name} present; this adapter uses the target's table")
    records = {"text": {"config": config}}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], vision_config=None, overrides=resource_overrides
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Builder(model)
    h, d = config["hidden_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    inter = config["intermediate_size"]
    taps = len(config["dflash_config"]["target_layer_ids"])

    builder.add("text/feature_projection", base, "fc.weight", (h, taps * h),
                inputs=("text/target_features",))
    builder.add("text/context_norm", base, "hidden_norm.weight", (h,))
    builder.add("text/final_norm", base, "norm.weight", (h,))
    context = []
    for i in range(config["num_hidden_layers"]):
        p, s = f"text/layers/{i}/", f"layers.{i}."
        builder.add(p + "input_norm", base, s + "input_layernorm.weight", (h,))
        builder.add(p + "post_attention_norm", base, s + "post_attention_layernorm.weight", (h,))
        for role, field, rows in (("query", "q_proj", q), ("key", "k_proj", kv),
                                  ("value", "v_proj", kv)):
            builder.add(p + "attention/" + role, base, s + f"self_attn.{field}.weight",
                        (rows, h), inputs=(p + "mixer_input",))
        builder.group(*(p + "attention/" + r for r in ("query", "key", "value")))
        # The same key/value weights also read the context: a second logical use with its own
        # input, so every layer's context projection packs into one plane over one input.
        for role, field in (("key", "k_proj"), ("value", "v_proj")):
            name = p + "attention/context_" + role
            builder.add(name, base, s + f"self_attn.{field}.weight", (kv, h),
                        inputs=("text/context_input",))
            context.append(name)
        builder.add(p + "attention/query_norm", base, s + "self_attn.q_norm.weight", (d,))
        builder.add(p + "attention/key_norm", base, s + "self_attn.k_norm.weight", (d,))
        builder.add(p + "attention/output", base, s + "self_attn.o_proj.weight", (h, q),
                    inputs=(p + "attention/context",))
        for role in ("gate", "up"):
            builder.add(p + "mlp/" + role, base, s + f"mlp.{role}_proj.weight", (inter, h),
                        inputs=(p + "ffn_input",))
        builder.group(p + "mlp/gate", p + "mlp/up")
        builder.add(p + "mlp/down", base, s + "mlp.down_proj.weight", (h, inter),
                    inputs=(p + "mlp/product",))
    builder.group(*context)
    return model
