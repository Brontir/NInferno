"""Published NVFP4 with and without calibrated activations, through dense_published.

    python synthetic_w4a16.py NINFERNO_CONVERT [REPOSITORY]

A two-layer Qwen3-shaped checkpoint holds every projection in Model Optimizer NVFP4 (U8 codes,
E4M3 block scales, F32 weight_scale_2), written twice from the same seed:

  w4a4   each projection also has input_scale (0.02): every use keeps AllowA4 and its
         activation_input_divisor auxiliary holds 1 / input_scale = 50 exactly;
  w4a16  no input_scale anywhere (the MLPs of a ModelOpt MIXED checkpoint such as
         nvidia/Muse-Glimmer-30B-NVFP4): the checkpoint converts, the fused q/k/v and
         gate/up planes included, every use is A16Only with no activation auxiliary, and
         every weight object is byte-identical to the w4a4 artifact's (activation
         calibration never changes the kept weights).

Before the fix a fused plane of W4A16 matrices asked each source for its activation divisor and
the conversion failed on the first missing input_scale. Exits 77 (skipped) without numpy/torch.
"""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import torch  # noqa: F401  (synthetic_formats needs it)
except ImportError:
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_gemma4 as base  # noqa: E402
import synthetic_formats as sf  # noqa: E402

HIDDEN, INTER, HEAD = 256, 384, 64
INPUT_SCALE = 0.02
PROJECTIONS = {"self_attn.q_proj": (4 * HEAD, HIDDEN), "self_attn.k_proj": (2 * HEAD, HIDDEN),
               "self_attn.v_proj": (2 * HEAD, HIDDEN), "self_attn.o_proj": (HIDDEN, 4 * HEAD),
               "mlp.gate_proj": (INTER, HIDDEN), "mlp.up_proj": (INTER, HIDDEN),
               "mlp.down_proj": (HIDDEN, INTER)}


def checkpoint(root: Path, calibrated: bool) -> None:
    rng = np.random.default_rng(11)
    config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": HIDDEN,
              "intermediate_size": INTER, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "head_dim": HEAD, "vocab_size": base.VOCAB,
              "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "rope_theta": 1000000.0,
              "tie_word_embeddings": True, "hidden_act": "silu", "attention_bias": False,
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    tensors = {}

    def plain(name, shape):
        values = rng.standard_normal(shape).astype(np.float32) * 0.05
        tensors[name] = ("BF16", list(shape), sf.bf16_bytes(values))

    plain("model.embed_tokens.weight", (base.VOCAB, HIDDEN))
    plain("model.norm.weight", (HIDDEN,))
    for layer in range(2):
        p = f"model.layers.{layer}."
        for role, (n, k) in PROJECTIONS.items():
            tensors[p + role + ".weight"] = sf.Matrix.tensor("U8", rng.integers(0, 256, (n, k // 2)).astype(np.uint8))
            scales = rng.integers(0x28, 0x40, (n, k // 16)).astype(np.uint8)
            tensors[p + role + ".weight_scale"] = ("F8_E4M3", [n, k // 16], scales.tobytes())
            tensors[p + role + ".weight_scale_2"] = sf.Matrix.tensor("F32", sf.f32([0.01]))
            if calibrated:
                tensors[p + role + ".input_scale"] = sf.Matrix.tensor("F32", sf.f32([INPUT_SCALE]))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            plain(p + norm + ".weight", (HIDDEN,))
        plain(p + "self_attn.q_norm.weight", (HEAD,))
        plain(p + "self_attn.k_norm.weight", (HEAD,))
    sf.write_checkpoint(root, tensors, config)


def projection_uses(value):
    return [u for u in value["uses"] if "/attention/" in u["parameter"] or "/mlp/" in u["parameter"]]


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []
    artifacts = {}
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-w4a16-") as tmp:
        for name, calibrated in (("w4a4", True), ("w4a16", False)):
            root = Path(tmp) / name
            checkpoint(root, calibrated)
            out = Path(tmp) / f"{name}.ninferno"
            result = subprocess.run([str(native), f"--model={root}", "--recipe=dense_published", "--quiet",
                                     f"--out={out}"], cwd=repository, capture_output=True, text=True)
            if result.returncode:
                errors.append(f"{name}: conversion failed: {result.stderr.strip().splitlines()[0][:400]}")
                continue
            artifacts[name] = sf.directory(out)
            value, objects, data, offset = artifacts[name]
            uses = projection_uses(value)
            if len(uses) != 2 * len(PROJECTIONS):
                errors.append(f"{name}: {len(uses)} projection uses, expected {2 * len(PROJECTIONS)}")
            for use in uses:
                policy, aux = use.get("activation_policy"), use.get("auxiliaries", {})
                if calibrated:
                    ref = aux.get("activation_input_divisor", {}).get("object")
                    if policy != "AllowA4" or ref is None:
                        errors.append(f"{name}: {use['parameter']}: {policy}, auxiliaries {aux}")
                        continue
                    o = objects[ref]
                    word = struct.unpack("<f", data[offset + o["offset"]:offset + o["offset"] + 4])[0]
                    if word != np.float32(1.0 / np.float32(INPUT_SCALE)):
                        errors.append(f"{name}: {use['parameter']}: activation divisor {word}, expected 1/input_scale")
                elif policy != "A16Only" or aux:
                    errors.append(f"{name}: {use['parameter']}: {policy}, auxiliaries {aux}")
            print(f"{name}: converted, {len(uses)} projection uses checked", flush=True)
        if len(artifacts) == 2:
            (_, a_obj, a_data, a_off), (_, b_obj, b_data, b_off) = artifacts["w4a4"], artifacts["w4a16"]
            weights = sorted(i for i in a_obj if i.startswith("weight/"))
            if weights != sorted(i for i in b_obj if i.startswith("weight/")):
                errors.append("the two artifacts hold different weight objects")
            for i in weights:
                a, b = a_obj[i], b_obj.get(i, {})
                if (a["format"], a["shape"], a["bytes"]) != (b.get("format"), b.get("shape"), b.get("bytes")) or \
                        a_data[a_off + a["offset"]:a_off + a["offset"] + a["bytes"]] != \
                        b_data[b_off + b["offset"]:b_off + b["offset"] + b["bytes"]]:
                    errors.append(f"{i}: differs between the calibrated and the W4A16 artifact")
            print(f"weights: {len(weights)} objects compared", flush=True)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
