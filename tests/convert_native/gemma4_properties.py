"""Gemma 4 model properties, one assertion each, against the reference implementation.

    python gemma4_properties.py NINFERNO_CONVERT [REPOSITORY]

Each property is checked twice: what Hugging Face transformers' Gemma 4 does (when the installed
transformers has the model), and what the native converter writes for a small dense Gemma 4
checkpoint whose config exercises the property. Every checker also runs once on sabotaged data
(the mistake the property guards against) and must fail there, so no check can pass vacuously.

  attention_scale   Q.K is scaled by 1.0 on sliding and full layers; a config carrying a
                    Gemma-3 style query_pre_attn_scalar changes nothing (the reference ignores it).
  gelu_tanh         the MLP activation is the tanh approximation, distinguishable from exact GELU
                    on [-10, 10]; any other hidden_activation is refused by name.
  rmsnorm_plain_w   Gemma 4 RMSNorm multiplies by w, not (1 + w): norm vectors are stored as the
                    checkpoint's words, and layer_scalar stays a separate FP32 value (no fold).
  k_eq_v            full-attention layers under attention_k_eq_v have no value projection: the
                    value stream is the key projection (weights and scales), so no value binding.
  kv_sharing        num_kv_shared_layers: each shared layer reads the keys transformers gives it
                    (the last earlier layer of its type), has no key/value/key-norm binding, and
                    use_double_wide_mlp doubles its MLP; a shared type with no source is refused.
  ple               per-layer embeddings: table and norm stored as published, and the BF16
                    sqrt(P), binary32 H^-1/2 and 2^-1/2 factors of the reference.
  embedding_scale   the embedding is multiplied by sqrt(hidden_size) rounded to the embedding's
                    BF16 compute dtype (Gemma's reference normalizer), not by the FP32 root.
  final_softcap     final_logit_softcapping is carried through exactly; absent means absent.

Exits 77 (skipped) when the interpreter lacks torch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

try:
    import torch
except ImportError:
    sys.exit(77)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthetic_gemma4 as base  # noqa: E402

VOCAB, INTER = base.VOCAB, 384
# sqrt(288) = 16.9706 has no BF16 representation (BF16 gives 17.0): the embedding property is
# only observable at a width whose root is not a BF16 number.
HIDDEN = 288


def bf16(value: float) -> float:
    return torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float().item()


def text_config(**overrides) -> dict:
    config = {
        "architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text",
        "hidden_size": HIDDEN, "vocab_size": VOCAB, "num_hidden_layers": 2,
        "max_position_embeddings": 4096, "tie_word_embeddings": True, "rms_norm_eps": 1e-06,
        "layer_types": ["sliding_attention", "full_attention"], "num_attention_heads": 4,
        "attention_k_eq_v": True, "sliding_window": 128, "head_dim": 64,
        "num_key_value_heads": 2, "global_head_dim": 128, "num_global_key_value_heads": 1,
        "rope_parameters": base.rope(), "final_logit_softcapping": 30.0,
        "intermediate_size": INTER, "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "attention_bias": False,
        "enable_moe_block": False,
    }
    for key, value in overrides.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    return config


def mlp_widths(config: dict) -> list[int]:
    """transformers' MLP width per layer: twice intermediate_size on KV-shared layers under
    use_double_wide_mlp."""
    layers = len(config["layer_types"])
    first = layers - (config.get("num_kv_shared_layers") or 0)
    wide = config.get("use_double_wide_mlp") and 0 < first
    return [INTER * (2 if wide and i >= first else 1) for i in range(layers)]


def checkpoint(root: Path, config: dict, seed: int) -> dict[str, bytes]:
    """A dense BF16 Gemma 4 checkpoint; returns the raw bytes of every tensor by name. KV-shared
    layers keep their unused key/value tensors, as Google's checkpoints do."""
    w = base.Writer(seed)
    h = config["hidden_size"]
    w.bf16("model.embed_tokens.weight", (VOCAB, h), scale=1.0)
    w.bf16("model.norm.weight", (h,), scale=1.0)
    ple = config.get("hidden_size_per_layer_input") or 0
    if ple:
        rows = ple * len(config["layer_types"])
        w.bf16("model.embed_tokens_per_layer.weight", (VOCAB, rows), scale=1.0)
        w.bf16("model.per_layer_model_projection.weight", (rows, h))
        w.bf16("model.per_layer_projection_norm.weight", (ple,), scale=1.0)
    widths = mlp_widths(config)
    for layer, kind in enumerate(config["layer_types"]):
        if ple:
            p = f"model.layers.{layer}."
            w.bf16(p + "per_layer_input_gate.weight", (ple, h))
            w.bf16(p + "per_layer_projection.weight", (h, ple))
            w.bf16(p + "post_per_layer_input_norm.weight", (h,), scale=1.0)
        p = f"model.layers.{layer}."
        d, kv = (64, 2) if kind == "sliding_attention" else (128, 1)
        w.bf16(p + "self_attn.q_proj.weight", (4 * d, h))
        w.bf16(p + "self_attn.k_proj.weight", (kv * d, h))
        if kind == "sliding_attention" or not config.get("attention_k_eq_v"):
            w.bf16(p + "self_attn.v_proj.weight", (kv * d, h))
        w.bf16(p + "self_attn.o_proj.weight", (h, 4 * d))
        w.bf16(p + "self_attn.q_norm.weight", (d,), scale=1.0)
        w.bf16(p + "self_attn.k_norm.weight", (d,), scale=1.0)
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            w.bf16(p + norm + ".weight", (h,), scale=1.0)
        w.bf16(p + "mlp.gate_proj.weight", (widths[layer], h))
        w.bf16(p + "mlp.up_proj.weight", (widths[layer], h))
        w.bf16(p + "mlp.down_proj.weight", (h, widths[layer]))
        w.f32(p + "layer_scalar", [1], 0.625 + 0.125 * layer)
    w.save(root)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    base.resources(root)
    return {name: data for name, (_, _, data) in w.tensors.items()}


def convert(native: Path, repository: Path, model: Path, out: Path,
            published: bool = False) -> subprocess.CompletedProcess:
    # A BF16 checkpoint under the Q8 table: NInferno's own quantization, forced; or as published.
    recipe = ["--recipe=gemma4_published"] if published else [
        "--recipe=gemma4_31b_nvfp4", "--maintainer-quantize"]
    return subprocess.run(
        [str(native), f"--model={model}", *recipe, "--quiet", f"--out={out}"],
        cwd=repository, capture_output=True, text=True,
    )


class Artifact:
    def __init__(self, path: Path) -> None:
        data = path.read_bytes()
        length = struct.unpack("<Q", data[8:16])[0]
        self.directory = json.loads(data[32:32 + length].decode())
        self.payload = data[-(-(32 + length) // 4096) * 4096:]
        self.objects = {o["id"]: o for o in self.directory["objects"]}
        self.config = self.directory["components"]["text"]["config"]
        self.bindings = self.directory["bindings"]

    def object_bytes(self, parameter: str) -> tuple[dict, bytes]:
        o = self.objects[self.bindings[parameter]["object"]]
        return o, self.payload[o["offset"]:o["offset"] + o["bytes"]]


# ------------------------------------------------------------------------------------------------
# Checkers: each returns a list of failures for the data it is given.


def check_attention_scale(config: dict) -> list[str]:
    value = config.get("attention_scale")
    return [] if value == 1.0 else [f"attention_scale is {value}, the reference uses 1.0"]


def check_norm_words(artifact_words: dict[str, bytes], source_words: dict[str, bytes]) -> list[str]:
    return [f"{name}: stored words differ from the checkpoint's plain w"
            for name, words in artifact_words.items() if words != source_words[name]]


def check_layer_scalar(stored: dict[str, float], source: dict[str, float]) -> list[str]:
    return [f"{name}: layer_scalar {stored[name]} != source {value}"
            for name, value in source.items() if stored.get(name) != value]


def check_value_bindings(bindings: dict, layer_types: list[str]) -> list[str]:
    errors = []
    for layer, kind in enumerate(layer_types):
        present = f"text/layers/{layer}/attention/value" in bindings
        if kind == "full_attention" and present:
            errors.append(f"layer {layer}: k_eq_v full layer has its own value binding")
        if kind == "sliding_attention" and not present:
            errors.append(f"layer {layer}: sliding layer lost its value binding")
        if f"text/layers/{layer}/attention/key" not in bindings:
            errors.append(f"layer {layer}: no key binding")
    return errors


def check_kv_sharing(config: dict, bindings: dict, sources: list[int], widths: list[int]) -> list[str]:
    """Each layer reads the keys of `sources[layer]`; a sharing layer has no key, value or key
    norm of its own, and every layer's MLP has the reference width."""
    errors = []
    if config.get("kv_source_layers") != sources:
        errors.append(f"kv_source_layers {config.get('kv_source_layers')} != reference {sources}")
    for layer, source in enumerate(sources):
        p = f"text/layers/{layer}/"
        own = [role for role in ("attention/key", "attention/value", "attention/key_norm") if p + role in bindings]
        if source != layer and own:
            errors.append(f"layer {layer} shares layer {source}'s keys but binds {own}")
        if source == layer and p + "attention/key" not in bindings:
            errors.append(f"layer {layer} owns its keys but has no key binding")
        if p + "attention/query" not in bindings:
            errors.append(f"layer {layer}: no query binding")
    if config.get("layer_intermediate_sizes", [config.get("intermediate_size")] * len(sources)) != widths:
        errors.append(f"layer_intermediate_sizes {config.get('layer_intermediate_sizes')} != reference {widths}")
    return errors


def check_ple(config: dict, stored: dict[str, bytes], source: dict[str, bytes], width: int, hidden: int) -> list[str]:
    """Per-layer embeddings as the reference applies them: the table and its norm as published,
    sqrt(P) in BF16 on the rows, H^-1/2 (binary32) on the projection, 2^-1/2 on the sum."""
    errors = [f"{name}: stored words differ from the checkpoint" for name, words in stored.items()
              if words != source[name]]
    expected = {"hidden_size_per_layer_input": width, "vocab_size_per_layer_input": VOCAB,
                "per_layer_embedding_scale": bf16(math.sqrt(width)),
                "per_layer_projection_scale": float(torch.tensor(hidden ** -0.5, dtype=torch.float32)),
                "per_layer_input_scale": float(torch.tensor(2.0 ** -0.5, dtype=torch.float32))}
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"{key} {config.get(key)} != reference {value}")
    return errors


def check_embedding_scale(estimate: float, expected: float, wrong: float) -> list[str]:
    # The table is Q8 (absmax per 32): a least-squares scale over the whole table is exact to
    # ~1e-5; the BF16 and FP32 roots of 288 differ by 1.7e-3.
    tolerance = abs(expected - wrong) / expected / 8
    if abs(estimate / expected - 1) > tolerance:
        return [f"table scaled by {estimate:.6f}, expected {expected} (the FP32 root is {wrong:.6f})"]
    return []


def check_softcap(config: dict, expected) -> list[str]:
    if expected is None:
        return [] if "final_logit_softcapping" not in config else [
            f"softcap {config['final_logit_softcapping']} invented for a config without one"]
    value = config.get("final_logit_softcapping")
    return [] if value == expected else [f"final_logit_softcapping {value} != source {expected}"]


def refused(result: subprocess.CompletedProcess, field: str) -> list[str]:
    if result.returncode == 0:
        return [f"a config with an unsupported {field} converted"]
    if field not in result.stderr:
        return [f"the refusal does not name {field}: {result.stderr[-400:]}"]
    return []


def least_squares_scale(table: torch.Tensor, source: torch.Tensor) -> float:
    return float((table.double() * source.double()).sum() / (source.double() ** 2).sum())


def decode_q8(obj: dict, words: bytes) -> torch.Tensor:
    from tools.artifact.codecs.row_split import dequantize_row_split

    return dequantize_row_split(words, obj["format"], obj["shape"], dtype=torch.float32)


# ------------------------------------------------------------------------------------------------


class Results:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def property(self, name: str, errors: list[str]) -> None:
        print(f"{name}: {'ok' if not errors else 'FAILED'}", flush=True)
        self.failed += [f"{name}: {e}" for e in errors]

    def control(self, name: str, errors: list[str]) -> None:
        # A sabotaged input must be caught.
        print(f"{name} (sabotage arm): {'caught' if errors else 'NOT CAUGHT'}", flush=True)
        if not errors:
            self.failed.append(f"{name}: the checker passes the sabotaged input")


def hf_reference(results: Results) -> None:
    try:
        from transformers.activations import ACT2FN
        from transformers.models.gemma4 import modeling_gemma4
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    except ImportError:
        print("hf: transformers without Gemma 4; reference halves skipped", flush=True)
        return
    raw = text_config(query_pre_attn_scalar=256)
    fields = {k: v for k, v in raw.items() if k not in ("architectures", "model_type")}
    torch.manual_seed(0)
    model = modeling_gemma4.Gemma4ForCausalLM(Gemma4TextConfig(**fields)).to(torch.bfloat16).eval()
    layers = model.model.layers
    scales = {layer.self_attn.layer_type: layer.self_attn.scaling for layer in layers}
    results.property("hf.attention_scale", [
        f"{kind}: HF scaling {value} (query_pre_attn_scalar=256 would give 1/16)"
        for kind, value in scales.items() if value != 1.0])

    x = torch.linspace(-10, 10, 20001)
    act = layers[0].mlp.act_fn
    gap = (act(x) - torch.nn.functional.gelu(x)).abs().max().item()
    tanh_gap = (act(x) - ACT2FN["gelu_pytorch_tanh"](x)).abs().max().item()
    print(f"hf.gelu_tanh: |tanh - exact| max {gap:.3e} on [-10, 10]", flush=True)
    results.property("hf.gelu_tanh", ([] if tanh_gap == 0 else ["HF MLP activation is not gelu_pytorch_tanh"])
                     + ([] if gap > 1e-4 else ["tanh and exact GELU are indistinguishable on [-10, 10]"]))

    norm = layers[0].input_layernorm
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HIDDEN) * 0.5)
        h = torch.randn(3, HIDDEN)
        y = norm(h.to(torch.bfloat16)).float()
        hn = h.to(torch.bfloat16).float()
        plain = hn * torch.pow(hn.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5) * norm.weight.float()
        shifted = plain / norm.weight.float() * (1 + norm.weight.float())

    def check_norm(reference):
        rel = ((y - reference).norm() / reference.norm()).item()
        return [] if rel < 1e-2 else [f"HF RMSNorm differs from the formula by rel {rel:.3e}"]

    results.property("hf.rmsnorm_plain_w", check_norm(plain))
    results.control("hf.rmsnorm_plain_w", check_norm(shifted))

    full = [layer.self_attn for layer in layers if layer.self_attn.layer_type == "full_attention"]
    sliding = [layer.self_attn for layer in layers if layer.self_attn.layer_type == "sliding_attention"]
    results.property("hf.k_eq_v", [
        "a k_eq_v full layer has a v_proj" for a in full if a.v_proj is not None] + [
        "a sliding layer lost its v_proj" for a in sliding if a.v_proj is None])

    embed = model.model.embed_tokens
    effective = embed.embed_scale.to(embed.weight.dtype).float().item()
    results.property("hf.embedding_scale", [] if effective == bf16(math.sqrt(HIDDEN)) else [
        f"HF normalizer {effective}, expected BF16 root {bf16(math.sqrt(HIDDEN))}"])
    print(f"hf.embedding_scale: BF16 root {effective} vs FP32 root {math.sqrt(HIDDEN):.6f}; published "
          + ", ".join(f"H={h}: {100 * (bf16(math.sqrt(h)) / math.sqrt(h) - 1):+.3f}%"
                      for h in (2816, 3840, 5376)), flush=True)
    results.property("hf.final_softcap", [] if model.config.final_logit_softcapping == 30.0 else [
        "HF dropped final_logit_softcapping"])


def native_checks(results: Results, native: Path, repository: Path, work: Path) -> None:
    # One checkpoint carries every property's input: query_pre_attn_scalar, k_eq_v, softcap 30.
    config = text_config(query_pre_attn_scalar=256)
    words = checkpoint(work / "gemma", config, seed=5)
    out = work / "gemma.ninferno"
    result = convert(native, repository, work / "gemma", out)
    if result.returncode:
        results.property("convert", [f"conversion failed: {result.stderr[-2000:]}"])
        return
    artifact = Artifact(out)

    results.property("attention_scale", check_attention_scale(artifact.config))
    results.control("attention_scale", check_attention_scale({**artifact.config, "attention_scale": 1 / 16}))

    for activation in ("gelu", "gelu_new", "silu"):
        results.property(f"gelu_tanh[{activation}]", refused(
            convert(native, repository, rewrite(work, config, "act_" + activation, hidden_activation=activation),
                    work / f"act_{activation}.ninferno"), "hidden_activation"))
    results.control("gelu_tanh", refused(result, "hidden_activation"))

    norms, source = {}, {}
    names = {"text/final_norm": "model.norm.weight"}
    for layer in range(2):
        for role, field in (("input_norm", "input_layernorm"), ("post_attention_norm", "post_attention_layernorm"),
                            ("pre_ffn_norm", "pre_feedforward_layernorm"), ("post_ffn_norm", "post_feedforward_layernorm"),
                            ("attention/query_norm", "self_attn.q_norm"), ("attention/key_norm", "self_attn.k_norm")):
            names[f"text/layers/{layer}/{role}"] = f"model.layers.{layer}.{field}.weight"
    for parameter, tensor in names.items():
        obj, stored = artifact.object_bytes(parameter)
        norms[parameter] = stored if obj["format"] == "bf16" else b"not bf16"
        source[parameter] = words[tensor]
    results.property("rmsnorm_plain_w", check_norm_words(norms, source))
    shifted = {}
    for parameter, stored in norms.items():
        values = torch.frombuffer(bytearray(stored), dtype=torch.bfloat16).float() + 1
        shifted[parameter] = values.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    results.control("rmsnorm_plain_w", check_norm_words(shifted, source))
    stored_scalars, source_scalars = {}, {}
    for layer in range(2):
        obj, raw = artifact.object_bytes(f"text/layers/{layer}/output_scale")
        stored_scalars[layer] = struct.unpack("<f", raw[:4])[0] if obj["format"] == "fp32" else None
        source_scalars[layer] = 0.625 + 0.125 * layer
    results.property("rmsnorm_plain_w[layer_scalar]", check_layer_scalar(stored_scalars, source_scalars))
    results.control("rmsnorm_plain_w[layer_scalar]",
                    check_layer_scalar({k: v * 1.5 for k, v in stored_scalars.items()}, source_scalars))

    results.property("k_eq_v", check_value_bindings(artifact.bindings, config["layer_types"]))
    results.control("k_eq_v", check_value_bindings(
        {**artifact.bindings, "text/layers/1/attention/value": artifact.bindings["text/layers/1/attention/key"]},
        config["layer_types"]))

    # A shared full layer with no earlier full layer to read is refused by name.
    results.property("kv_sharing[no source]", refused(
        convert(native, repository, rewrite(work, config, "shared", num_kv_shared_layers=1),
                work / "shared.ninferno"), "num_kv_shared_layers"))
    results.control("kv_sharing[no source]", refused(result, "num_kv_shared_layers"))

    shared_checks(results, native, repository, work)

    obj, stored = artifact.object_bytes("text/token_embedding")
    table = decode_q8(obj, stored)
    embedding = torch.frombuffer(bytearray(words["model.embed_tokens.weight"]), dtype=torch.bfloat16)
    estimate = least_squares_scale(table, embedding.float().reshape(VOCAB, HIDDEN))
    expected, wrong = bf16(math.sqrt(HIDDEN)), math.sqrt(HIDDEN)
    print(f"embedding_scale: table scaled by {estimate:.6f} (config {artifact.config.get('embedding_scale')}); "
          f"BF16 root {expected}, FP32 root {wrong:.6f}", flush=True)
    results.property("embedding_scale", check_embedding_scale(estimate, expected, wrong)
                     + check_embedding_scale(artifact.config.get("embedding_scale", 0.0), expected, wrong))
    results.control("embedding_scale", check_embedding_scale(estimate * wrong / expected, expected, wrong))

    final_softcap_checks(results, native, repository, work, config, artifact)


# E2B's shape in small: the last two of four layers read the keys of the last earlier layer of
# their type, with double-wide MLPs, and per-layer embeddings of an even width whose root is
# not a BF16 number (sqrt(18) = 4.2426 -> 4.25).
PLE = 18
SHARED_PLE = {"num_hidden_layers": 4, "num_kv_shared_layers": 2, "use_double_wide_mlp": True,
              "layer_types": ["sliding_attention", "full_attention"] * 2,
              "hidden_size_per_layer_input": PLE, "vocab_size_per_layer_input": VOCAB}


def hf_kv_sources(config: dict):
    """transformers' reading of KV sharing: the layer each layer's keys come from (the earlier
    layer of its type that stores full-length keys) and each layer's MLP width; None without
    Gemma 4 in transformers."""
    try:
        from transformers.models.gemma4 import modeling_gemma4
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    except ImportError:
        return None
    fields = {k: v for k, v in config.items() if k not in ("architectures", "model_type")}
    model = modeling_gemma4.Gemma4ForCausalLM(Gemma4TextConfig(**fields)).eval()
    layers = model.model.layers
    stores = {layer.self_attn.layer_type: i for i, layer in enumerate(layers)
              if layer.self_attn.store_full_length_kv}
    sources = [stores[layer.self_attn.layer_type] if layer.self_attn.is_kv_shared_layer else i
               for i, layer in enumerate(layers)]
    return sources, [layer.mlp.intermediate_size for layer in layers]


def shared_checks(results: Results, native: Path, repository: Path, work: Path) -> None:
    shared = text_config(**SHARED_PLE)
    shared_words = checkpoint(work / "shared_ple", shared, seed=7)
    converted = convert(native, repository, work / "shared_ple", work / "shared_ple.ninferno", published=True)
    if converted.returncode:
        results.property("kv_sharing", [f"conversion failed: {converted.stderr[-2000:]}"])
        return
    shared_artifact = Artifact(work / "shared_ple.ninferno")
    sources, widths = hf_kv_sources(shared) or ([0, 1, 0, 1], mlp_widths(shared))
    results.property("kv_sharing", check_kv_sharing(shared_artifact.config, shared_artifact.bindings,
                                                    sources, widths))
    results.control("kv_sharing", check_kv_sharing(
        {**shared_artifact.config, "kv_source_layers": list(range(4))}, shared_artifact.bindings,
        sources, widths))
    results.control("kv_sharing[mlp]", check_kv_sharing(
        {**shared_artifact.config, "layer_intermediate_sizes": [INTER] * 4}, shared_artifact.bindings,
        sources, widths))
    ple_names = {"text/ple_table": "model.embed_tokens_per_layer.weight",
                 "text/ple_norm": "model.per_layer_projection_norm.weight"}
    stored = {name: shared_artifact.object_bytes(name)[1] for name in ple_names}
    source = {name: shared_words[tensor] for name, tensor in ple_names.items()}
    results.property("ple", check_ple(shared_artifact.config, stored, source, PLE, HIDDEN))
    results.control("ple", check_ple({**shared_artifact.config, "per_layer_embedding_scale": math.sqrt(PLE + 1)},
                                     stored, source, PLE, HIDDEN))


def final_softcap_checks(results: Results, native: Path, repository: Path, work: Path, config: dict,
                         artifact: Artifact) -> None:
    results.property("final_softcap", check_softcap(artifact.config, 30.0))
    results.control("final_softcap", check_softcap({**artifact.config, "final_logit_softcapping": 20.0}, 30.0))
    for value in (None, 12.5):
        variant = rewrite(work, config, f"softcap_{value}", final_logit_softcapping=value)
        out_variant = work / f"softcap_{value}.ninferno"
        converted = convert(native, repository, variant, out_variant)
        if converted.returncode:
            results.property(f"final_softcap[{value}]", [f"conversion failed: {converted.stderr[-400:]}"])
            continue
        results.property(f"final_softcap[{value}]", check_softcap(Artifact(out_variant).config, value))
    results.control("final_softcap[None]", check_softcap({**artifact.config}, None))


def python_parity(results: Results, repository: Path, work: Path, native_out: Path) -> None:
    """tools/convert writes the same embedding table (both converters implement the property)."""
    out = work / "gemma.python.ninferno"
    run = subprocess.run([sys.executable, "-m", "tools.convert", f"--model={work / 'gemma'}",
                          "--recipe=gemma4_31b_nvfp4", "--device=cpu", f"--out={out}"],
                         cwd=repository, capture_output=True, text=True)
    if run.returncode:
        results.property("embedding_scale[tools.convert]", [f"python conversion failed: {run.stderr[-1500:]}"])
        return
    a, b = Artifact(native_out), Artifact(out)
    same = a.object_bytes("text/token_embedding")[1] == b.object_bytes("text/token_embedding")[1]
    results.property("embedding_scale[tools.convert]", ([] if same else ["embedding tables differ"])
                     + ([] if a.config == b.config else ["configs differ"]))


def rewrite(work: Path, config: dict, name: str, **overrides) -> Path:
    """A copy of the base checkpoint with config fields changed (None removes one)."""
    root = work / name
    root.mkdir()
    for item in (work / "gemma").iterdir():
        (root / item.name).symlink_to(item)
    (root / "config.json").unlink()
    changed = dict(config)
    for key, value in overrides.items():
        if value is None:
            changed.pop(key, None)
        else:
            changed[key] = value
    (root / "config.json").write_text(json.dumps(changed, indent=2), encoding="utf-8")
    return root


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else HERE.parents[1]).resolve()
    sys.path.insert(0, str(repository))
    results = Results()
    hf_reference(results)
    with tempfile.TemporaryDirectory(prefix="ninferno-gemma4-properties-") as tmp:
        work = Path(tmp)
        native_checks(results, native, repository, work)
        if (work / "gemma.ninferno").exists():
            python_parity(results, repository, work, work / "gemma.ninferno")
    for failure in results.failed:
        print(failure, file=sys.stderr)
    return 1 if results.failed else 0


if __name__ == "__main__":
    sys.exit(main())
