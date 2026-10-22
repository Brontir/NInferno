"""Convert small synthetic Gemma-4-shaped checkpoints with both converters and compare bytes.

    python synthetic_gemma4.py NINFERNO_CONVERT [REPOSITORY]

Five checkpoints cover every mechanism the official Gemma 4 recipes use: compressed-tensors
and Model Optimizer NVFP4 (direct import, the per-expert divisor stack and its compensated
down projection, qkv packing on equal divisors, Q8 from dequantized NVFP4 where rows are not
128-aligned), BF16 projections to Q8, the scaled Q8 embedding and Q6 head, untied heads,
sharded indexes, the assistant's shortlisted head with K padding, the DFlash drafter with
resource overrides, and split output files. Each artifact must match the Python converter's
byte for byte after the 32-byte header (whose last 16 bytes are the random artifact id), and
the reports must match apart from timing, ids, paths and the torch environment.

Exits 77 (skipped) when the interpreter lacks torch, which the Python converter needs.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

try:
    import torch
except ImportError:
    sys.exit(77)

VOCAB = 512


def rope():
    return {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
        },
    }


class Writer:
    """Tensors for one checkpoint, written as one or more safetensors shards."""

    def __init__(self, seed: int) -> None:
        self.tensors: dict[str, tuple[str, list[int], bytes]] = {}
        self.generator = torch.Generator().manual_seed(seed)

    def raw(self, name, dtype, shape, data):
        self.tensors[name] = (dtype, list(shape), data)

    def bf16(self, name, shape, scale=0.05):
        values = torch.randn(shape, generator=self.generator) * scale
        self.raw(name, "BF16", shape, values.to(torch.bfloat16).view(torch.uint8).numpy().tobytes())

    def f32(self, name, shape, value):
        self.raw(name, "F32", shape, struct.pack("<f", value))

    def nvfp4(self, prefix, rows, k, divisor, input_divisor, layout="ct"):
        codes = torch.randint(0, 256, (rows, k // 2), generator=self.generator, dtype=torch.int64)
        scales = torch.randint(0, 0x7F, (rows, k // 16), generator=self.generator, dtype=torch.int64)
        packed = codes.to(torch.uint8).numpy().tobytes()
        scale_bytes = scales.to(torch.uint8).numpy().tobytes()
        if layout == "ct":
            self.raw(prefix + ".weight_packed", "U8", (rows, k // 2), packed)
            self.raw(prefix + ".weight_scale", "F8_E4M3", (rows, k // 16), scale_bytes)
            self.f32(prefix + ".weight_global_scale", [1], divisor)
            self.f32(prefix + ".input_global_scale", [1], input_divisor)
        else:  # Model Optimizer: multipliers, scalar shapes
            self.raw(prefix + ".weight", "U8", (rows, k // 2), packed)
            self.raw(prefix + ".weight_scale", "F8_E4M3", (rows, k // 16), scale_bytes)
            self.f32(prefix + ".weight_scale_2", [], 1.0 / divisor)
            self.f32(prefix + ".input_scale", [], 1.0 / input_divisor)

    def save(self, root: Path, shards: int = 1) -> None:
        root.mkdir(parents=True, exist_ok=True)
        names = sorted(self.tensors)
        groups = [names[i::shards] for i in range(shards)]
        weight_map = {}
        for index, group in enumerate(groups):
            file = "model.safetensors" if shards == 1 else f"model-{index+1:05d}-of-{shards:05d}.safetensors"
            header, blobs, offset = {}, [], 0
            for name in group:
                dtype, shape, data = self.tensors[name]
                header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
                blobs.append(data)
                offset += len(data)
                weight_map[name] = file
            encoded = json.dumps(header).encode()
            encoded += b" " * (-len(encoded) % 8)
            with open(root / file, "wb") as stream:
                stream.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
        if shards > 1:
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
            )


def resources(root: Path, *, template=True, tokenizer=True) -> None:
    if tokenizer:
        vocab = {f"tok{i}": i for i in range(VOCAB)}
        added = [{"id": i, "content": f"tok{i}", "special": True} for i in (0, 1, 2, 3)]
        (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": added}), encoding="utf-8")
        (root / "tokenizer_config.json").write_text(
            json.dumps({"added_tokens_decoder": {"4": {"content": "tok4", "special": True}}}), encoding="utf-8"
        )
        (root / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")
    if template:
        (root / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}×{% endfor %}\n", encoding="utf-8")


def text_checkpoint(root: Path, *, moe: bool, layout: str, nested: bool, tied: bool, seed: int) -> None:
    hidden, inter = 256, (192 if moe else 384)
    text = {
        "model_type": "gemma4_text", "hidden_size": hidden, "vocab_size": VOCAB,
        "num_hidden_layers": 2, "max_position_embeddings": 4096, "tie_word_embeddings": tied,
        "rms_norm_eps": 1e-06, "layer_types": ["sliding_attention", "full_attention"],
        "num_attention_heads": 4, "attention_k_eq_v": True, "sliding_window": 128,
        "head_dim": 64, "num_key_value_heads": 2, "global_head_dim": 128,
        "num_global_key_value_heads": 1, "rope_parameters": rope(),
        "final_logit_softcapping": 30.0, "intermediate_size": inter,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0, "attention_bias": False, "enable_moe_block": moe,
    }
    if moe:
        text.update({"num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 64})
    if nested:
        config = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
                  "tie_word_embeddings": tied, "text_config": text}
        prefix = "model.language_model."
    else:
        config = {"architectures": ["Gemma4ForCausalLM"], **text}
        prefix = "model."
    w = Writer(seed)
    w.bf16(prefix + "embed_tokens.weight", (VOCAB, hidden))
    if not tied:
        w.bf16("lm_head.weight", (VOCAB, hidden))
    w.bf16(prefix + "norm.weight", (hidden,))
    encoded_attention = layout == "ct"
    for layer, kind in enumerate(text["layer_types"]):
        p = f"{prefix}layers.{layer}."
        d = 64 if kind == "sliding_attention" else 128
        kv = 2 if kind == "sliding_attention" else 1
        roles = [("q_proj", 4 * d), ("k_proj", kv * d)] + ([("v_proj", kv * d)] if kind == "sliding_attention" else [])
        for index, (role, rows) in enumerate(roles):
            if encoded_attention:
                # Layer 0 shares one divisor across q/k/v (one packed parent); layer 1 does not.
                divisor = 0.5 if layer == 0 else 0.5 + 0.25 * index
                w.nvfp4(p + "self_attn." + role, rows, hidden, divisor, 0.7 + 0.1 * index, layout)
            else:
                w.bf16(p + "self_attn." + role + ".weight", (rows, hidden))
        if encoded_attention:
            w.nvfp4(p + "self_attn.o_proj", hidden, 4 * d, 0.9, 1.1, layout)
        else:
            w.bf16(p + "self_attn.o_proj.weight", (hidden, 4 * d))
        w.bf16(p + "self_attn.q_norm.weight", (d,))
        w.bf16(p + "self_attn.k_norm.weight", (d,))
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            w.bf16(p + norm + ".weight", (hidden,))
        w.bf16(p + "layer_scalar", (1,))
        if layout == "ct":
            # One divisor for gate and up, as the fused-layer publishers write them: the MoE's
            # 192-row halves are kept only packed into one 384-row parent (whole scale tiles).
            for role in ("gate_proj", "up_proj"):
                w.nvfp4(p + "mlp." + role, inter, hidden, 0.3, 0.8, layout)
            w.nvfp4(p + "mlp.down_proj", hidden, inter, 0.45, 0.6, layout)
        else:
            for role in ("gate_proj", "up_proj"):
                w.bf16(p + "mlp." + role + ".weight", (inter, hidden))
            w.bf16(p + "mlp.down_proj.weight", (hidden, inter))
        if moe:
            for norm in ("pre_feedforward_layernorm_2", "post_feedforward_layernorm_1",
                         "post_feedforward_layernorm_2"):
                w.bf16(p + norm + ".weight", (hidden,))
            w.bf16(p + "router.proj.weight", (4, hidden))
            w.bf16(p + "router.scale", (hidden,))
            w.bf16(p + "router.per_expert_scale", (4,))
            for expert in range(4):
                e = f"{p}experts.{expert}."
                gate = 0.2 + 0.01 * expert
                # Expert 0 shares gate/up divisors; the others take d_up/d_gate != 1.
                up = gate if expert == 0 else gate * (1.0 + 0.1 * expert)
                w.nvfp4(e + "gate_proj", 64, hidden, gate, 0.5 + 0.05 * expert, layout)
                w.nvfp4(e + "up_proj", 64, hidden, up, 0.55, layout)
                w.nvfp4(e + "down_proj", hidden, 64, 0.25 + 0.02 * expert, 0.4 + 0.03 * expert, layout)
    w.save(root, shards=2 if layout == "modelopt" else 1)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    resources(root)


def assistant_checkpoint(root: Path) -> None:
    h, b, inter = 128, 256, 96
    config = {
        "architectures": ["Gemma4AssistantForCausalLM"], "backbone_hidden_size": b,
        "text_config": {
            "model_type": "gemma4_text", "hidden_size": h, "vocab_size": VOCAB,
            "num_hidden_layers": 2, "num_kv_shared_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"], "num_attention_heads": 4,
            "head_dim": 64, "num_key_value_heads": 2, "global_head_dim": 128,
            "num_global_key_value_heads": 1, "max_position_embeddings": 4096,
            "intermediate_size": inter, "sliding_window": 128, "rms_norm_eps": 1e-06,
            "rope_parameters": rope(), "hidden_activation": "gelu_pytorch_tanh",
            "enable_moe_block": False, "hidden_size_per_layer_input": 0,
        },
    }
    w = Writer(11)
    w.bf16("pre_projection.weight", (h, 2 * b))
    w.bf16("post_projection.weight", (b, h))
    w.bf16("model.embed_tokens.weight", (VOCAB, h))
    w.bf16("model.norm.weight", (h,))
    for layer, d in enumerate((64, 128)):
        p = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            w.bf16(p + norm + ".weight", (h,))
        w.bf16(p + "self_attn.q_proj.weight", (4 * d, h))
        w.bf16(p + "self_attn.q_norm.weight", (d,))
        w.bf16(p + "self_attn.o_proj.weight", (h, 4 * d))
        w.bf16(p + "mlp.gate_proj.weight", (inter, h))
        w.bf16(p + "mlp.up_proj.weight", (inter, h))
        w.bf16(p + "mlp.down_proj.weight", (h, inter))
        w.bf16(p + "layer_scalar", (1,))
    w.save(root)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    resources(root, template=False)
    counts = torch.randint(0, 1000, (VOCAB,), generator=torch.Generator().manual_seed(3))
    counts[7] = counts[9]  # a tie, broken by id order
    (root / "ranking.i64").write_bytes(counts.to(torch.int64).numpy().tobytes())


def dflash_checkpoint(root: Path) -> None:
    h, d, inter = 256, 64, 192
    config = {
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3", "hidden_act": "silu",
        "attention_bias": False, "final_logit_softcapping": 30.0, "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"], "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": d, "num_target_layers": 4,
        "dflash_config": {"target_layer_ids": [1, 3], "mask_token_id": 4}, "vocab_size": VOCAB,
        "hidden_size": h, "intermediate_size": inter, "sliding_window": 128,
        "rope_theta": 1000000, "rms_norm_eps": 1e-06, "block_size": 16, "rope_scaling": None,
    }
    w = Writer(13)
    w.bf16("fc.weight", (h, 2 * h))
    w.bf16("hidden_norm.weight", (h,))
    w.bf16("norm.weight", (h,))
    for layer in range(2):
        p = f"layers.{layer}."
        w.bf16(p + "input_layernorm.weight", (h,))
        w.bf16(p + "post_attention_layernorm.weight", (h,))
        w.bf16(p + "self_attn.q_proj.weight", (4 * d, h))
        w.bf16(p + "self_attn.k_proj.weight", (2 * d, h))
        w.bf16(p + "self_attn.v_proj.weight", (2 * d, h))
        w.bf16(p + "self_attn.q_norm.weight", (d,))
        w.bf16(p + "self_attn.k_norm.weight", (d,))
        w.bf16(p + "self_attn.o_proj.weight", (h, 4 * d))
        w.bf16(p + "mlp.gate_proj.weight", (inter, h))
        w.bf16(p + "mlp.up_proj.weight", (inter, h))
        w.bf16(p + "mlp.down_proj.weight", (h, inter))
    w.save(root)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def normalized_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    for key in ("preparation_seconds", "seconds", "artifact_id", "output", "torch_version", "cuda_version"):
        report.pop(key)
    for file in report["files"]:
        file.pop("path")
    return report


def compare(name: str, python_out: Path, native_out: Path) -> list[str]:
    errors = []
    py_report = json.loads(Path(str(python_out) + ".conversion.json").read_text(encoding="utf-8"))
    for index, record in enumerate(py_report["files"]):
        py_file = python_out if index == 0 else python_out.parent / record["path"].rsplit("/", 1)[-1]
        native_file = native_out if index == 0 else native_out.parent / py_file.name.replace(python_out.name, native_out.name)
        a, b = py_file.read_bytes(), native_file.read_bytes()
        if len(a) != len(b) or a[:16] != b[:16] or a[32:] != b[32:]:
            errors.append(f"{name}: {py_file.name} differs from {native_file.name}")
    if normalized_report(Path(str(python_out) + ".conversion.json")) != normalized_report(
        Path(str(native_out) + ".conversion.json")
    ):
        errors.append(f"{name}: conversion reports differ")
    # The source identity's parts (src/artifact/source_identity.h): the inputs read and a sealed
    # content digest per object, the same from both converters (the bytes above are).
    from artifact_directory import directory
    sealed = []
    for out in (python_out, native_out):
        value, objects, _, _ = directory(out)
        provenance = value.get("provenance", {})
        digests = provenance.get("content_sha256", {})
        if not provenance.get("inputs") or not all(
            f.get("sha256") for i in provenance["inputs"] for f in i.get("files", [])
        ):
            errors.append(f"{name}: {out.name} records no inputs")
        if set(digests) != set(objects) or any(d == "0" * 64 for d in digests.values()):
            errors.append(f"{name}: {out.name} has no sealed digest for every object")
        sealed.append((provenance.get("inputs"), digests))
    if sealed[0] != sealed[1]:
        errors.append(f"{name}: the converters record other inputs or digests")
    return errors


def consume_case(work: Path, native: Path, repository: Path) -> list[str]:
    """--consume-source (mapped output, journal, punched source) writes what the default run
    (direct output) writes, and leaves no shard, journal or partial file behind."""
    errors = []
    checkpoint = work / "moe-consumed"
    text_checkpoint(checkpoint, moe=True, layout="modelopt", nested=True, tied=True, seed=4)
    reference, consumed = work / "native" / "reference.ninferno", work / "native" / "consumed.ninferno"
    common = [str(native), f"--model={checkpoint}", "--recipe=gemma4_26b_a4b_nvfp4", "--quiet"]
    for command in ([*common, f"--out={reference}"], [*common, "--consume-source", f"--out={consumed}"]):
        result = subprocess.run(command, cwd=repository, capture_output=True, text=True)
        if result.returncode:
            errors.append(f"consume: {' '.join(command[-2:])} failed\n{result.stderr[-2000:]}")
            return errors
    a, b = reference.read_bytes(), consumed.read_bytes()
    if len(a) != len(b) or a[:16] != b[:16] or a[32:] != b[32:]:
        errors.append("consume: the consumed conversion differs from the default one")
    if normalized_report(Path(str(reference) + ".conversion.json")) != normalized_report(
        Path(str(consumed) + ".conversion.json")
    ):
        errors.append("consume: conversion reports differ")
    leftovers = [p.name for p in checkpoint.glob("*.safetensors")] + [
        p.name for p in consumed.parent.glob("consumed.ninferno.*") if not p.name.endswith(".json")
    ]
    if leftovers:
        errors.append(f"consume: left behind {leftovers}")
    print(f"consume: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-synthetic-") as tmp:
        work = Path(tmp)
        target = work / "moe-ct"
        text_checkpoint(target, moe=True, layout="ct", nested=True, tied=True, seed=1)
        text_checkpoint(work / "moe-modelopt", moe=True, layout="modelopt", nested=True, tied=True, seed=2)
        text_checkpoint(work / "dense-ct", moe=False, layout="ct", nested=False, tied=False, seed=3)
        assistant_checkpoint(work / "assistant")
        dflash_checkpoint(work / "dflash")
        overrides = [f"--resource={role}={target / role}" for role in
                     ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")]
        cases = {
            "moe-ct": [f"--model={target}", "--recipe=gemma4_26b_a4b_nvfp4", "--name=synthetic moe × ct"],
            "moe-modelopt": [f"--model={work / 'moe-modelopt'}", "--recipe=gemma4_26b_a4b_nvfp4"],
            "dense-ct": [f"--model={work / 'dense-ct'}/", "--recipe=gemma4_31b_nvfp4"],
            "assistant": [f"--model={work / 'assistant'}", "--recipe=gemma4_assistant_q8", "--proposal",
                          "--proposal-rows=100", f"--ranking={work / 'assistant' / 'ranking.i64'}",
                          f"--resource=chat_template.jinja={target / 'chat_template.jinja'}"],
            "assistant-split": [f"--model={work / 'assistant'}", "--recipe=gemma4_assistant_q8",
                                "--max-file-bytes=300000",
                                f"--resource=chat_template.jinja={target / 'chat_template.jinja'}"],
            "dflash": [f"--model={work / 'dflash'}", "--recipe=gemma4_dflash_q8", *overrides],
        }
        for name, args in cases.items():
            (work / "py").mkdir(exist_ok=True)
            (work / "native").mkdir(exist_ok=True)
            python_out, native_out = work / "py" / f"{name}.ninferno", work / "native" / f"{name}.ninferno"
            for command in (
                [sys.executable, "-m", "tools.convert", *args, "--device=cpu", f"--out={python_out}"],
                [str(native), *args, "--quiet", f"--out={native_out}"],
            ):
                result = subprocess.run(command, cwd=repository, capture_output=True, text=True)
                if result.returncode:
                    errors.append(f"{name}: {command[0]} failed\n{result.stderr[-2000:]}")
            if not errors:
                errors += compare(name, python_out, native_out)
            print(f"{name}: {'ok' if not errors else 'FAILED'}", flush=True)
        errors += consume_case(work, native, repository)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
