"""Gemma 4 MTP drafter ("assistant") adapter: its own small model over a target's state.

Google publishes one drafter per Gemma 4 model (`google/gemma-4-*-it-assistant`,
`Gemma4AssistantForCausalLM`). It is not a head on the target: it is a four-layer Gemma 4 text
stack of its own width that owns no key/value projections at all. Every one of its layers reads
the target's cached keys and values — the last sliding layer's for its sliding layers, the last
full layer's for its full layer — so it adds no cache and recomputes no context. One draft step:

    x      = pre_projection([target_embedding(token); hidden])      2B -> h
    x      = layers(x)          queries only, over the target's cache
    x      = final_norm(x)
    logits = head(x)            the drafter's own tied table, no soft cap
    hidden = post_projection(x) h -> B, the next step's `hidden`

where B is the target width and `hidden` starts as the target's final-normed state. The
artifact is a separate file; the engine binds it next to its target and checks the pairing.
"""

from __future__ import annotations

from pathlib import Path

from .config_checks import f32 as _f32, fixed as _fixed, positive as _positive
from .gemma4 import _Builder, _rope_group, text_model_type
from .model import Model, Parameter
from .proposal import shortlist_ids
from .resources import load_resources
from .sources.logical import array_source, gather_source
from .sources.safetensors import SafetensorsSource


def assistant_config(source: dict) -> dict:
    if source.get("architectures") not in (
        ["Gemma4AssistantForCausalLM"],
        ["Gemma4UnifiedAssistantForCausalLM"],
    ):
        raise ValueError("expected a Gemma4AssistantForCausalLM checkpoint")
    if source.get("use_ordered_embeddings", False):
        # The centroid-ordered head needs a token ordering table these checkpoints do not ship.
        raise ValueError("assistant: the ordered-embedding head is not implemented")
    raw = source.get("text_config")
    if not isinstance(raw, dict):
        raise ValueError("assistant: text_config is missing")
    text_model_type(raw, "assistant.text")
    _fixed(raw, "hidden_activation", "gelu_pytorch_tanh", "assistant.text")
    _fixed(raw, "enable_moe_block", False, "assistant.text")
    _fixed(raw, "hidden_size_per_layer_input", 0, "assistant.text")
    layers = _positive(raw.get("num_hidden_layers"), "assistant.num_hidden_layers")
    if raw.get("num_kv_shared_layers") != layers:
        raise ValueError("assistant: every layer must read the target's key/value cache")
    kinds = raw.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != layers:
        raise ValueError("assistant: layer_types must cover every layer")
    if any(kind not in ("sliding_attention", "full_attention") for kind in kinds):
        raise ValueError("assistant: unknown layer type")
    heads = _positive(raw.get("num_attention_heads"), "assistant.num_attention_heads")
    geometry = {
        "sliding_attention": (
            _positive(raw.get("head_dim"), "assistant.head_dim"),
            _positive(raw.get("num_key_value_heads"), "assistant.num_key_value_heads"),
        ),
        "full_attention": (
            _positive(raw.get("global_head_dim"), "assistant.global_head_dim"),
            _positive(raw.get("num_global_key_value_heads"), "assistant.num_global_key_value_heads"),
        ),
    }
    result = {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "hidden_size": _positive(raw.get("hidden_size"), "assistant.hidden_size"),
        "backbone_hidden_size": _positive(
            source.get("backbone_hidden_size"), "assistant.backbone_hidden_size"
        ),
        "vocab_size": _positive(raw.get("vocab_size"), "assistant.vocab_size"),
        "num_hidden_layers": layers,
        "max_position_embeddings": _positive(
            raw.get("max_position_embeddings"), "assistant.max_position_embeddings"
        ),
        "intermediate_size": _positive(raw.get("intermediate_size"), "assistant.intermediate_size"),
        "num_attention_heads": heads,
        "layer_types": list(kinds),
        "sliding_window": _positive(raw.get("sliding_window"), "assistant.sliding_window"),
        "rms_norm_eps": _f32(raw.get("rms_norm_eps", 1e-6), "assistant.rms_norm_eps"),
        "attention_scale": 1.0,
        "tie_word_embeddings": True,
        "attention_geometry": {},
        "rope_parameters": {},
    }
    for kind, (head_dim, kv_heads) in geometry.items():
        if heads % kv_heads:
            raise ValueError(f"assistant.{kind}: query heads must divide by the target's KV heads")
        result["attention_geometry"][kind] = {"head_dim": head_dim, "num_key_value_heads": kv_heads}
        result["rope_parameters"][kind] = _rope_group(
            raw.get("rope_parameters", {}), kind, head_dim, "assistant.rope_parameters"
        )
    return result


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions=None,
    resource_overrides=None,
) -> Model:
    if set(components) != {"text"}:
        raise ValueError("a Gemma 4 assistant artifact holds the drafter alone")
    config = assistant_config(base.config)
    records = {"text": {"config": config}}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], vision_config=None, overrides=resource_overrides
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Builder(model)
    h, b, v = config["hidden_size"], config["backbone_hidden_size"], config["vocab_size"]
    heads, inter = config["num_attention_heads"], config["intermediate_size"]

    builder.add("text/pre_projection", base, "pre_projection.weight", (h, 2 * b),
                inputs=("text/draft_input",))
    builder.add("text/post_projection", base, "post_projection.weight", (b, h),
                inputs=("text/final_hidden",))
    # The head is the drafter's own tied table, read unscaled; the drafter never embeds tokens
    # itself, its token input comes from the target's (scaled) embedding.
    builder.add("text/output_head", base, "model.embed_tokens.weight", (v, h),
                inputs=("text/final_hidden",))
    builder.add("text/final_norm", base, "model.norm.weight", (h,))
    for i, kind in enumerate(config["layer_types"]):
        p, s = f"text/layers/{i}/", f"model.layers.{i}."
        d = config["attention_geometry"][kind]["head_dim"]
        for role, field in (
            ("input_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("pre_ffn_norm", "pre_feedforward_layernorm"),
            ("post_ffn_norm", "post_feedforward_layernorm"),
        ):
            builder.add(p + role, base, s + field + ".weight", (h,))
        builder.add(p + "attention/query", base, s + "self_attn.q_proj.weight", (heads * d, h),
                    inputs=(p + "mixer_input",))
        builder.add(p + "attention/query_norm", base, s + "self_attn.q_norm.weight", (d,))
        builder.add(p + "attention/output", base, s + "self_attn.o_proj.weight", (h, heads * d),
                    inputs=(p + "attention/context",))
        for role in ("gate", "up"):
            builder.add(p + "mlp/" + role, base, s + f"mlp.{role}_proj.weight", (inter, h),
                        inputs=(p + "ffn_input",))
        builder.add(p + "mlp/down", base, s + "mlp.down_proj.weight", (h, inter),
                    inputs=(p + "mlp/product",))
        builder.group(p + "mlp/gate", p + "mlp/up")
        builder.add(p + "output_scale", base, s + "layer_scalar", (1,), direct="fp32")
    return model


def restrict_head(model: Model, *, ranking: str | Path, rows: int) -> None:
    """Keep only the `rows` most frequent tokens in the drafter's head.

    A draft step reads the whole head, and the full table is half of the drafter's bytes (256 MiB
    of the 26B drafter's 478 MiB in Q8). A token outside the shortlist can never be drafted, so
    the shortlist caps acceptance at its coverage of the target's output. The engine takes the
    argmax over the kept rows and maps it back through `text/head_token_ids`. Call this before
    the recipe is built, so the recipe sees the restricted head as an ordinary projection.
    """
    head = model.parameters["text/output_head"]
    ids = shortlist_ids(model, ranking, rows)
    source = gather_source(head.source, ids)
    model.parameters["text/output_head"] = Parameter(
        head.name, source.shape, source, inputs=head.inputs
    )
    model.add(
        Parameter(
            "text/head_token_ids",
            (rows,),
            array_source(ids, f"shortlist({ranking})"),
            direct_format="int32",
        )
    )
    model.config["head_rows"] = rows
