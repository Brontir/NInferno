"""A tied output head is one copy of the token embedding in the artifact.

    python synthetic_tied.py NINFERNO_CONVERT [REPOSITORY]

Two-layer dense checkpoints (Qwen3 and Granite shapes) written from one seed, converted with the
dense recipes:

  qwen3 tied, dense_bf16 and dense_q8   text/output_head is bound to the very object that
        text/token_embedding is bound to (one weight object fewer than the untied twin, and the
        payload smaller by exactly that object's bytes); the head keeps its own Use record on
        text/final_hidden;
  qwen3 untied, lm_head.weight holding the same values   two objects, byte-identical to each
        other and to the tied artifact's shared one (sharing changes no weight byte);
  granite tied, embedding_multiplier 12   the embedding table is scaled at conversion and the
        head is not: the encodings differ, so both are kept;
  qwen3_5 (synthetic_qwen3_5's dense checkpoint), qwen3_5_bf16: tied shares, untied does not.

Before the change every tied checkpoint wrote the table twice (Qwen3.5-4B BF16: 1.18 GiB more).
Exits 77 (skipped) without numpy/torch.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import torch  # noqa: F401  (synthetic_formats needs it)
except ImportError:
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_formats as sf  # noqa: E402
import synthetic_qwen3_5 as sq  # noqa: E402

VOCAB, HIDDEN, INTER, HEAD = 512, 256, 384, 64
PROJECTIONS = (("self_attn.q_proj", (4 * HEAD, HIDDEN)), ("self_attn.k_proj", (2 * HEAD, HIDDEN)),
               ("self_attn.v_proj", (2 * HEAD, HIDDEN)), ("self_attn.o_proj", (HIDDEN, 4 * HEAD)),
               ("mlp.gate_proj", (INTER, HIDDEN)), ("mlp.up_proj", (INTER, HIDDEN)),
               ("mlp.down_proj", (HIDDEN, INTER)))


def checkpoint(root: Path, *, model_type: str, tied: bool) -> None:
    rng = np.random.default_rng(5)
    arch = "Qwen3ForCausalLM" if model_type == "qwen3" else "GraniteForCausalLM"
    config = {"architectures": [arch], "model_type": model_type, "hidden_size": HIDDEN,
              "intermediate_size": INTER, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "head_dim": HEAD, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
              "max_position_embeddings": 4096, "rope_theta": 1000000.0, "tie_word_embeddings": tied,
              "hidden_act": "silu", "attention_bias": False}
    if model_type == "granite":
        config.update({"embedding_multiplier": 12.0, "logits_scaling": 8.0,
                       "residual_multiplier": 0.22, "attention_multiplier": 0.015625})
    tensors = {}

    def plain(name, shape, values=None):
        if values is None:
            values = rng.standard_normal(shape).astype(np.float32) * 0.05
        tensors[name] = ("BF16", list(shape), sf.bf16_bytes(values))
        return values

    table = plain("model.embed_tokens.weight", (VOCAB, HIDDEN))
    if not tied:
        plain("lm_head.weight", (VOCAB, HIDDEN), table)
    plain("model.norm.weight", (HIDDEN,))
    for layer in range(2):
        p = f"model.layers.{layer}."
        for role, shape in PROJECTIONS:
            plain(p + role + ".weight", shape)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            plain(p + norm + ".weight", (HIDDEN,))
        if model_type == "qwen3":
            plain(p + "self_attn.q_norm.weight", (HEAD,))
            plain(p + "self_attn.k_norm.weight", (HEAD,))
    sf.write_checkpoint(root, tensors, config)


def object_of(value, name):
    binding = value["bindings"][name]
    if "object" not in binding:
        raise ValueError(f"{name}: bound in parts {binding}")
    return binding["object"]


def payload(artifact, ident):
    _, objects, data, offset = artifact
    o = objects[ident]
    return data[offset + o["offset"]:offset + o["offset"] + o["bytes"]]


def weights(artifact):
    return {i: o for i, o in artifact[1].items() if i.startswith("weight/")}


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-tied-") as tmp:
        def convert(name, model_type, tied, recipe):
            root = Path(tmp) / f"{name}-src"
            if not root.exists():
                checkpoint(root, model_type=model_type, tied=tied)
            out = Path(tmp) / f"{name}-{recipe}.ninferno"
            # dense_q8 quantizes an unquantized checkpoint, which the converter runs only for a
            # maintainer (the default refuses it); the tied-head rule is the same for both recipes.
            forced = ["--maintainer-quantize"] if recipe == "dense_q8" else []
            result = subprocess.run([str(native), f"--model={root}", f"--recipe={recipe}", "--quiet",
                                     f"--out={out}", *forced], cwd=repository, capture_output=True,
                                    text=True)
            if result.returncode:
                errors.append(f"{name} {recipe}: conversion failed: {result.stderr.strip()[:400]}")
                return None
            return sf.directory(out)

        for recipe in ("dense_bf16", "dense_q8"):
            tied = convert("tied", "qwen3", True, recipe)
            untied = convert("untied", "qwen3", False, recipe)
            if tied is None or untied is None:
                continue
            tv, uv = tied[0], untied[0]
            shared = object_of(tv, "text/token_embedding")
            head = object_of(tv, "text/output_head")
            if head != shared:
                errors.append(f"{recipe}: tied head bound to {head}, embedding to {shared}")
            head_uses = [u["input"] for u in tv["uses"] if u["parameter"] == "text/output_head"]
            if head_uses != ["text/final_hidden"]:
                errors.append(f"{recipe}: tied head uses {head_uses}")
            u_table = object_of(uv, "text/token_embedding")
            u_head = object_of(uv, "text/output_head")
            if u_table == u_head:
                errors.append(f"{recipe}: an untied head was shared")
            if not payload(tied, shared) == payload(untied, u_table) == payload(untied, u_head):
                errors.append(f"{recipe}: the shared object differs from the untied table or head")
            tw, uw = weights(tied), weights(untied)
            saved = sum(o["bytes"] for o in uw.values()) - sum(o["bytes"] for o in tw.values())
            if len(uw) - len(tw) != 1 or saved != tw[shared]["bytes"]:
                errors.append(f"{recipe}: {len(tw)} tied / {len(uw)} untied weight objects, "
                              f"{saved} bytes saved, table {tw[shared]['bytes']}")
            print(f"{recipe}: tied head shares {shared} ({tw[shared]['format']}, {saved} bytes saved), "
                  f"{len(tw)} weight objects against {len(uw)}", flush=True)
        # Qwen3.5 (GDN + gated attention, the Engine path): the same rule through its own adapter.
        for tied in (True, False):
            root = Path(tmp) / f"qwen3_5-{tied}"
            sq.base_checkpoint(root, moe_model=False, nested=True, tied=tied, vision=False, seed=31)
            out = Path(tmp) / f"qwen3_5-{tied}.ninferno"
            result = subprocess.run([str(native), f"--model={root}", "--recipe=qwen3_5_bf16", "--quiet",
                                     f"--out={out}"], cwd=repository, capture_output=True, text=True)
            if result.returncode:
                errors.append(f"qwen3_5 tied={tied}: conversion failed: {result.stderr.strip()[:400]}")
                continue
            value = sf.directory(out)[0]
            same = object_of(value, "text/output_head") == object_of(value, "text/token_embedding")
            if same != tied:
                errors.append(f"qwen3_5 tied={tied}: head and table share an object: {same}")
            else:
                print(f"qwen3_5 tied={tied}: {'one object' if same else 'two objects'}", flush=True)
        granite = convert("granite", "granite", True, "dense_bf16")
        if granite is not None:
            table = object_of(granite[0], "text/token_embedding")
            head = object_of(granite[0], "text/output_head")
            if table == head:
                errors.append("granite: the scaled embedding table was shared with the unscaled head")
            elif payload(granite, table) == payload(granite, head):
                errors.append("granite: embedding_multiplier did not reach the table")
            else:
                print("granite: scaled table and head kept apart", flush=True)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
