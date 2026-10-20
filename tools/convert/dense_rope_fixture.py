"""dense_rope_fixture.py <fixture.json> <name> <config.json>: adds (or replaces) one entry of
tests/convert_native/fixtures/dense_rope.json: the checkpoint's config.json and, per layer type,
transformers' own rotary frequencies as binary32 bit patterns (ROPE_INIT_FUNCTIONS of the model's
rope type) with the attention factor. A LongRoPE rope also records its long-context frequencies
(transformers switches to them past original_max_position_embeddings). Needs transformers >= 5.
"""
import importlib
import json
import struct
import sys

import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def bits(tensor):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in tensor.float().tolist()]


def main():
    fixture_path, name, config_path = sys.argv[1:4]
    raw = json.load(open(config_path, encoding="utf-8"))
    config = AutoConfig.for_model(**raw)
    config = getattr(config, "text_config", None) or config
    params = config.rope_parameters
    layer_types = sorted(set(getattr(config, "layer_types", None) or ["full_attention"]))
    rope = {}
    for kind in layer_types:
        p = params.get(kind, params) if isinstance(params, dict) else params
        rope_type = p.get("rope_type", "default")
        init = ROPE_INIT_FUNCTIONS[rope_type] if rope_type != "default" else None
        kwargs = {"layer_type": kind} if kind in params else {}
        if init is None:
            # the model's own rotary class (its default reads the layer type's parameters)
            modeling = importlib.import_module(type(config).__module__.replace(".configuration_", ".modeling_"))
            rotary = next(getattr(modeling, n) for n in dir(modeling) if n.endswith("RotaryEmbedding"))
            inv, scale = rotary.compute_default_rope_parameters(config, torch.device("cpu"), **kwargs)
        else:
            inv, scale = init(config, torch.device("cpu"), **kwargs)
        entry = {"rope_type": rope_type, "attention_scaling": float(scale), "inv_freq_bits": bits(inv)}
        if rope_type == "longrope":
            original = p.get("original_max_position_embeddings") or config.original_max_position_embeddings
            long_inv, _ = init(config, torch.device("cpu"), seq_len=original + 1, **kwargs)
            entry["long_inv_freq_bits"] = bits(long_inv)
            entry["long_after"] = int(original)
        rope[kind] = entry
    try:
        fixtures = json.load(open(fixture_path, encoding="utf-8"))
    except FileNotFoundError:
        fixtures = {}
    fixtures[name] = {"config": raw, "rope": rope}
    with open(fixture_path, "w", encoding="utf-8", newline="\n") as out:
        json.dump(fixtures, out)  # one line, as the file is kept
    print(name, {k: (v["rope_type"], v["attention_scaling"], len(v["inv_freq_bits"])) for k, v in rope.items()})


if __name__ == "__main__":
    main()
