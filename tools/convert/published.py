"""The activation widths a publisher validated its quantized matrices with (provenance).

Mirrors the native converter (src/convert/convert.cpp, published_activation_summary): the same
provenance["published_activations"] keys and counts for the same recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

from .methods import import_encoded


def _ascii_lower(text: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)


def activation_bits(algorithm: str) -> int:
    """ModelOpt's quant_algo as activation bits: W4A16_NVFP4 16, FP8 8, NVFP4 4; 0 unknown."""
    name = _ascii_lower(algorithm)
    # Weight-only schemes (W4A16_NVFP4, FP8_PB_WO, INT8_WO, INT4_AWQ) keep 16-bit activations.
    if "a16" in name or name.endswith("_wo"):
        return 16
    if "a8" in name or name.startswith(("fp8", "int8", "mxfp8")):
        return 8
    if name.startswith("int4"):
        return 16
    # ModelOpt's MXFP4 quantizes activations as the weights (W4A4), as NVFP4 does.
    if name.startswith(("nvfp4", "mxfp4")) or "a4" in name:
        return 4
    return 0


def published_activations(store) -> tuple[str, int, dict[str, int]]:
    """(source, bits of every layer or 0, bits per named module) of a checkpoint.

    ModelOpt's hf_quant_config.json (one quant_algo, or one per module under MIXED_PRECISION), else
    config.json's quantization_config: ModelOpt's fields, or compressed-tensors groups that agree on
    input_activations.num_bits (none given: 16). source "": the checkpoint says nothing.
    """
    source, every, layers = "", 0, {}

    def modelopt(body, name: str) -> None:
        nonlocal source, every
        # A description of the KV cache alone says nothing of the weights' activations.
        if not isinstance(body, dict) or (
            "quant_algo" not in body and "quantized_layers" not in body
        ):
            return
        source = name
        if isinstance(body.get("quant_algo"), str):
            every = activation_bits(body["quant_algo"])
        if isinstance(body.get("quantized_layers"), dict):
            for module, layer in body["quantized_layers"].items():
                if isinstance(layer, dict) and isinstance(layer.get("quant_algo"), str):
                    layers[module] = activation_bits(layer["quant_algo"])

    # The checkpoint's directory: the path may name one .safetensors file or its index.
    described = Path(getattr(store, "root", store.path)) / "hf_quant_config.json"
    if described.is_file():
        try:
            data = json.loads(described.read_text(encoding="utf-8"))
            body = data["quantization"] if isinstance(data, dict) and "quantization" in data else data
            modelopt(body, "hf_quant_config.json")
        except (OSError, ValueError):
            source, every, layers = "", 0, {}
        # An unreadable or silent file leaves config.json's quantization_config to answer.
        if source:
            return source, every, layers
    quantization = store.config.get("quantization_config") if isinstance(store.config, dict) else None
    if isinstance(quantization, dict) and "quant_algo" in quantization:
        modelopt(quantization, "quantization_config")
        return source, every, layers
    if not isinstance(quantization, dict) or not isinstance(quantization.get("config_groups"), dict):
        return source, every, layers
    widths = set()
    for group in quantization["config_groups"].values():
        activations = group.get("input_activations") if isinstance(group, dict) else None
        bits = activations.get("num_bits") if isinstance(activations, dict) else None
        widths.add(bits if type(bits) is int else 16)
    if len(widths) == 1:
        source, every = "compressed-tensors", next(iter(widths))
    return source, every, layers


def _module_bits(leaf: str, every: int, layers: dict[str, int]) -> int | None:
    """The bits of the longest named module holding `leaf`, under either text root."""
    names = [leaf]
    for old, new in (("model.language_model.", "model."), ("model.", "model.language_model.")):
        if leaf.startswith(old):
            names.append(new + leaf[len(old):])
        elif not leaf.startswith("model."):
            names.append(old + leaf)
    for name in names:
        while name:
            if name in layers:
                return layers[name]
            name = name.rpartition(".")[0]
    return None


def published_activation_summary(model, recipe) -> dict | None:
    """provenance["published_activations"]: the published widths of the matrices kept as published.

    Each matrix answers to the description of the checkpoint it is imported from (an NVFP4
    recipe's quantized source, not the unquantized --model); one whose checkpoint describes nothing
    is not counted. below: matrices the recorded policy lets run below that width (fast mode);
    below_int8: the same under --activations int8, which runs every use as A8 but an A16Only
    head's or embedding's.
    """
    width = {"AllowA4": 4, "AllowA8": 8}
    described: dict[int, tuple[str, int, dict[str, int]]] = {}
    named: list[str] = []
    counts: dict[int, int] = {}
    below = below_int8 = 0
    for name, parameter in model.parameters.items():
        selections = recipe.selections[name]
        if not parameter.inputs or not all(s.method is import_encoded for s in selections):
            continue
        origin = [pair for selection in selections for pair in selection.source.origin]
        if not origin:
            continue
        store = origin[0][0]
        if id(store) not in described:
            described[id(store)] = published_activations(store)
        source, every, layers = described[id(store)]
        if not source:
            continue
        if source not in named:
            named.append(source)
        bits = every
        for leaf in dict.fromkeys(pair[1] for pair in origin):
            found = _module_bits(leaf, every, layers)
            if found is not None:
                bits = found
                break
        counts[bits] = counts.get(bits, 0) + 1  # 0: the description names no width for it
        policy = recipe.policies[(name, parameter.inputs[0])]
        fixed = "head" in name or "embed" in name
        int8 = policy if policy == "A16Only" and fixed else "AllowA8"
        if bits > 0:
            below += width.get(policy, 16) < bits
            below_int8 += width.get(int8, 16) < bits
    if not counts:
        return None
    out = {"source": ", ".join(named)}
    for bits in sorted(counts):
        out["unknown" if bits == 0 else str(bits)] = counts[bits]
    out["below"] = below
    out["below_int8"] = below_int8
    return out
