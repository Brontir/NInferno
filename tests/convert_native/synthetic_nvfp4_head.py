"""Published NVFP4 matrices whose rows end inside a 128-row scale tile, through the recipes.

    python synthetic_nvfp4_head.py NINFERNO_CONVERT [REPOSITORY]

A two-layer Qwen3-shaped checkpoint quantized as Model Optimizer's mixed export of
nvidia/Muse-Glimmer-30B-NVFP4: FP8 attention in layer 0 (per-tensor weight and input scales),
weight-only NVFP4 MLPs of 320 rows (no input_scale) and an untied weight-only NVFP4 lm_head of
448 rows (three whole 128-row scale tiles and one of 64 rows, as the 202048-row Muse vocabulary
ends); layer 1's attention is calibrated NVFP4 whose q, k and v (256, 64 and 64 rows) and whose
MLP gate and up were quantized against divisors of their own; the embedding and norms BF16.

  default           no --recipe: dense_encoded converts; text/output_head is one nvfp4 object
                    [448, 256] of the geometry's size, its codes, block scale words and divisor
                    (1 / weight_scale_2) the checkpoint's bit for bit, the scale words of the last
                    tile's rows past 448 zero, its use A16Only without an activation auxiliary;
                    every NVFP4 layer matrix decodes to the checkpoint's values with its block
                    scales: layer 0's gate/up one packed [640, 256] parent, layer 1's gate, up,
                    key and value parents of their own (320 and 64 rows), the key and value AllowA4
                    with their activation divisor; layer 0's attention FP8, the embedding BF16.
  vocab_q8          --vocab q8 quantizes the BF16 embedding to q8_g32_fp16 and keeps the head's
                    words (the same object bytes as default).
  published         --recipe dense_published keeps the head and the layers the same way.
  calibrated        lm_head also has input_scale: the head still keeps BF16 activations (A16Only,
                    no activation auxiliary), the same weight bytes.
  moe-shared        a Qwen3-MoE checkpoint whose NVFP4 experts are 192 rows (moe_intermediate_size
                    not a multiple of 128): with one weight divisor over the gate/up bank it is one
                    [1536, 256] plane of one divisor, the down bank (whole tiles) one [1024, 192]
                    plane of a divisor per expert, every expert its checkpoint's values.
  moe-split         the same with a divisor per expert matrix: no plane holds that bank losslessly,
                    so the conversion is refused naming the bank and the reason.
  qwen3_5           a Qwen3.5 checkpoint whose MLPs (text and MTP) are compressed-tensors NVFP4
                    of 320 rows, layer 0's gate and up of one divisor, the others' of two:
                    qwen3_5_published keeps every one as NVFP4 (one [640, 256] parent, or parents
                    of their own) with the checkpoint's values, where it once widened them to BF16.
  qwen-moe-planner  qwen3_6_35b_a3b_nvfp4 stacks routed expert banks without the recipe-level
                    check: over 192-row experts of a divisor each the planner refuses the bank's
                    stacked parent by name, where a scale tile would span two divisors.
  publisher         hf_quant_config.json (W4A16_NVFP4, layer 1's key NVFP4): provenance counts the
                    kept matrices per published activation width and those fast mode runs below it.
  python-tail       the Python writer streams a 200-row NVFP4 matrix in 128-row blocks
                    (TensorOutput.write_codes, the last block ragged): the object is the codec's
                    encoding of the whole matrix and decodes back to its words.

Exits 77 (skipped) without torch.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthetic_gemma4 as base  # noqa: E402
import synthetic_scales as ss  # noqa: E402

HIDDEN, INTER, HEAD, VOCAB = 256, 320, 64, 448
HEADS, KV_HEADS = 4, 1
WEIGHT_SCALE_2 = 2.0**-7  # divisor 128, exact
INPUT_SCALE = 0.0625  # activation divisor 16, exact
ATTENTION = {"q_proj": (HEADS * HEAD, HIDDEN), "k_proj": (KV_HEADS * HEAD, HIDDEN),
             "v_proj": (KV_HEADS * HEAD, HIDDEN), "o_proj": (HIDDEN, HEADS * HEAD)}
MLP = {"gate_proj": (INTER, HIDDEN), "up_proj": (INTER, HIDDEN), "down_proj": (HIDDEN, INTER)}
# Layer 1 quantizes q, k, v and gate, up against divisors of their own: no packed parent.
LAYER1_SCALES = {"q_proj": 2.0**-7, "k_proj": 2.0**-6, "v_proj": 2.0**-5, "o_proj": 2.0**-7,
                 "gate_proj": 2.0**-7, "up_proj": 2.0**-8, "down_proj": 2.0**-7}
PARAMETERS = {"q_proj": "attention/query", "k_proj": "attention/key", "v_proj": "attention/value",
              "o_proj": "attention/output", "gate_proj": "mlp/gate", "up_proj": "mlp/up",
              "down_proj": "mlp/down"}
EXPERTS, MOE_INTER = 4, 192


class Writer(base.Writer):
    """Model Optimizer's NVFP4 (weight_scale_2, optional input_scale) and per-tensor FP8."""

    def nvfp4_modelopt(self, prefix: str, n: int, k: int, input_scale: float | None,
                       weight_scale_2: float = WEIGHT_SCALE_2) -> None:
        codes = torch.randint(0, 256, (n, k // 2), generator=self.generator).to(torch.uint8)
        scales = torch.randint(0x20, 0x48, (n, k // 16), generator=self.generator).to(torch.uint8)
        self.raw(prefix + ".weight", "U8", (n, k // 2), codes.numpy().tobytes())
        self.raw(prefix + ".weight_scale", "F8_E4M3", (n, k // 16), scales.numpy().tobytes())
        self.f32(prefix + ".weight_scale_2", [], weight_scale_2)
        if input_scale is not None:
            self.f32(prefix + ".input_scale", [], input_scale)

    def fp8(self, prefix: str, n: int, k: int) -> None:
        values = torch.randn((n, k), generator=self.generator) * 2
        self.raw(prefix + ".weight", "F8_E4M3", (n, k),
                 values.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes())
        self.f32(prefix + ".weight_scale", [], 0.0123456789)
        self.f32(prefix + ".input_scale", [], 0.05)


def resources(root: Path, config: dict) -> None:
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    vocab = {f"tok{i}": i for i in range(VOCAB)}
    added = [{"id": i, "content": f"tok{i}", "special": True} for i in (0, 1, 2, 3)]
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab},
                                                     "added_tokens": added}), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({}), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")
    (root / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m['content'] }}{% endfor %}\n", encoding="utf-8")


def text_config(architecture: str, model_type: str, layers: dict, **extra) -> dict:
    return {"architectures": [architecture], "model_type": model_type, "hidden_size": HIDDEN,
            "intermediate_size": INTER, "num_hidden_layers": 2, "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS, "head_dim": HEAD, "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "rope_theta": 1000000.0,
            "tie_word_embeddings": False, "hidden_act": "silu", "attention_bias": False,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "producer": {"name": "modelopt"},
                                    "quantized_layers": layers},
            **extra}


def layer_norms(w: Writer, p: str) -> None:
    for norm in ("input_layernorm", "post_attention_layernorm"):
        w.bf16(p + norm + ".weight", (HIDDEN,))
    w.bf16(p + "self_attn.q_norm.weight", (HEAD,))
    w.bf16(p + "self_attn.k_norm.weight", (HEAD,))


def checkpoint(root: Path, calibrated_head: bool) -> dict:
    w = Writer(23)
    w.bf16("model.embed_tokens.weight", (VOCAB, HIDDEN))
    w.bf16("model.norm.weight", (HIDDEN,))
    w.nvfp4_modelopt("lm_head", VOCAB, HIDDEN, INPUT_SCALE if calibrated_head else None)
    layers = {}
    for layer in range(2):
        p = f"model.layers.{layer}."
        for role, (n, k) in ATTENTION.items():
            if layer == 0:
                w.fp8(p + "self_attn." + role, n, k)
                layers[p + "self_attn." + role] = {"quant_algo": "FP8"}
            else:
                w.nvfp4_modelopt(p + "self_attn." + role, n, k, INPUT_SCALE, LAYER1_SCALES[role])
                layers[p + "self_attn." + role] = {"quant_algo": "NVFP4", "group_size": 16}
        for role, (n, k) in MLP.items():
            scale = WEIGHT_SCALE_2 if layer == 0 else LAYER1_SCALES[role]
            w.nvfp4_modelopt(p + "mlp." + role, n, k, None, scale)
            layers[p + "mlp." + role] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        layer_norms(w, p)
    layers["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    w.save(root)
    resources(root, text_config("Qwen3ForCausalLM", "qwen3", layers))
    return ss.read_checkpoint(root)


def moe_checkpoint(root: Path, split_divisors: bool) -> dict:
    """Qwen3-MoE, BF16 but for NVFP4 experts of 192 rows; the down bank a divisor per expert."""
    w = Writer(29)
    w.bf16("model.embed_tokens.weight", (VOCAB, HIDDEN))
    w.bf16("model.norm.weight", (HIDDEN,))
    w.bf16("lm_head.weight", (VOCAB, HIDDEN))
    layers = {}
    for layer in range(2):
        p = f"model.layers.{layer}."
        for role, (n, k) in ATTENTION.items():
            w.bf16(p + "self_attn." + role + ".weight", (n, k))
        w.bf16(p + "mlp.gate.weight", (EXPERTS, HIDDEN))
        for e in range(EXPERTS):
            x = f"{p}mlp.experts.{e}."
            gate_up = 2.0**-(6 + e) if split_divisors else WEIGHT_SCALE_2
            w.nvfp4_modelopt(x + "gate_proj", MOE_INTER, HIDDEN, None, gate_up)
            w.nvfp4_modelopt(x + "up_proj", MOE_INTER, HIDDEN, None, gate_up)
            w.nvfp4_modelopt(x + "down_proj", HIDDEN, MOE_INTER, None, 2.0**-(5 + e))
            for role in ("gate_proj", "up_proj", "down_proj"):
                layers[x + role] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        layer_norms(w, p)
    w.save(root)
    resources(root, text_config("Qwen3MoeForCausalLM", "qwen3_moe", layers,
                                num_experts=EXPERTS, num_experts_per_tok=2,
                                moe_intermediate_size=MOE_INTER, norm_topk_prob=True,
                                decoder_sparse_step=1, mlp_only_layers=[]))
    return ss.read_checkpoint(root)


def convert(native: Path, repository: Path, model: Path, out: Path, *extra: str):
    return subprocess.run([str(native), f"--model={model}", f"--out={out}", "--quiet", *extra],
                          cwd=repository, capture_output=True, text=True)


def object_of(artifact: ss.Artifact, parameter: str) -> dict:
    """The object a parameter is bound to (the first part's of a packed one)."""
    binding = artifact.bindings[parameter]
    return artifact.objects[binding["object"] if "object" in binding
                            else binding["parts"][0]["object"]]


def head_checks(name: str, artifact: ss.Artifact, tensors: dict) -> list[str]:
    from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
    from tools.artifact.layouts import block_scale_geometry

    errors = []
    head = object_of(artifact, "text/output_head")
    if (head["format"], head["shape"]) != ("nvfp4", [VOCAB, HIDDEN]):
        return [f"{name}: text/output_head is {head['format']} {head['shape']}, not nvfp4 "
                f"[{VOCAB}, {HIDDEN}]"]
    g = block_scale_geometry("nvfp4", (VOCAB, HIDDEN))
    if head["bytes"] != g.payload_bytes or g.scale_rows != 512:
        errors.append(f"{name}: head object of {head['bytes']} bytes, geometry {g.payload_bytes} "
                      f"over {g.scale_rows} scale rows")
    # Decoding checks the scale words of rows 448..511 are zero.
    codes, scales, divisor = decode_nvfp4_words(bytes(artifact.words(head["id"])),
                                                (VOCAB, HIDDEN))
    source_codes = torch.frombuffer(bytearray(tensors["lm_head.weight"][2]), dtype=torch.uint8)
    source_scales = torch.frombuffer(bytearray(tensors["lm_head.weight_scale"][2]),
                                     dtype=torch.uint8)
    if not torch.equal(codes.reshape(-1), source_codes):
        errors.append(f"{name}: head codes differ from the checkpoint's")
    if not torch.equal(scales.reshape(-1), source_scales):
        errors.append(f"{name}: head block scale words differ from the checkpoint's")
    if divisor.item() != 1.0 / WEIGHT_SCALE_2:
        errors.append(f"{name}: head divisor {divisor.item()}, expected {1.0 / WEIGHT_SCALE_2}")
    errors += ss.same_values(artifact, tensors, "modelopt", {"text/output_head": "lm_head"}, "")
    # A calibrated head too: its logits keep BF16 activations.
    use = artifact.uses["text/output_head"]
    if use.get("activation_policy") != "A16Only" or use.get("auxiliaries"):
        errors.append(f"{name}: head use {use.get('activation_policy')}, auxiliaries "
                      f"{use.get('auxiliaries')}")
    return errors


def layer_checks(name: str, artifact: ss.Artifact, tensors: dict) -> list[str]:
    errors = []
    pairs = {}
    for layer in range(2):
        roles = MLP if layer == 0 else {**ATTENTION, **MLP}
        for role in roles:
            parameter = f"text/layers/{layer}/{PARAMETERS[role]}"
            section = "self_attn." if role in ATTENTION else "mlp."
            if object_of(artifact, parameter)["format"] != "nvfp4":
                errors.append(f"{name}: {parameter} is {object_of(artifact, parameter)['format']}")
                continue
            pairs[parameter] = f"model.layers.{layer}.{section}{role}"
    errors += ss.same_values(artifact, tensors, "modelopt", pairs, "")
    # One divisor: layer 0's gate/up pack into one parent; divisors of their own keep layer 1's
    # gate, up, key and value apart, each ending inside a scale tile.
    want = {"text/layers/0/mlp/gate": [2 * INTER, HIDDEN],
            "text/layers/1/mlp/gate": [INTER, HIDDEN], "text/layers/1/mlp/up": [INTER, HIDDEN],
            "text/layers/1/attention/key": [KV_HEADS * HEAD, HIDDEN],
            "text/layers/1/attention/value": [KV_HEADS * HEAD, HIDDEN]}
    for parameter, shape in want.items():
        if object_of(artifact, parameter)["shape"] != shape:
            errors.append(f"{name}: {parameter} object {object_of(artifact, parameter)['shape']}, "
                          f"expected {shape}")
    for role in ("key", "value"):
        parameter = f"text/layers/1/attention/{role}"
        policy = artifact.uses[parameter].get("activation_policy")
        divisor = artifact.activation_divisor(parameter)
        if policy != "AllowA4" or divisor != 1.0 / INPUT_SCALE:
            errors.append(f"{name}: {parameter} use {policy}, activation divisor {divisor}")
    query_format = object_of(artifact, "text/layers/0/attention/query")["format"]
    if not query_format.startswith("fp8"):
        errors.append(f"{name}: layer 0 attention query is {query_format}, not FP8")
    return errors


def dense_cases(work: Path, native: Path, repository: Path) -> list[str]:
    errors: list[str] = []
    head_bytes: dict[str, bytes] = {}
    tensors = checkpoint(work / "w4a16", calibrated_head=False)
    calibrated_tensors = checkpoint(work / "calibrated", calibrated_head=True)
    cases = [("default", work / "w4a16", (), tensors),
             ("vocab_q8", work / "w4a16", ("--vocab=q8",), tensors),
             ("published", work / "w4a16", ("--recipe=dense_published",), tensors),
             ("calibrated", work / "calibrated", (), calibrated_tensors)]
    for name, model, extra, source in cases:
        out = work / f"{name}.ninferno"
        result = convert(native, repository, model, out, *extra)
        if result.returncode:
            errors.append(f"{name}: conversion failed ({result.returncode}): "
                          f"{result.stderr.strip()[:600]}")
            continue
        artifact = ss.Artifact(out)
        case_errors = head_checks(name, artifact, source) + layer_checks(name, artifact, source)
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        recipe = report["provenance"]["recipe"]
        want = "dense_published" if name == "published" else "dense_encoded"
        if recipe != want:
            case_errors.append(f"{name}: recipe {recipe}, expected {want}")
        embedding = object_of(artifact, "text/token_embedding")["format"]
        if embedding != ("q8_g32_fp16" if name == "vocab_q8" else "bf16"):
            case_errors.append(f"{name}: token embedding is {embedding}")
        head = object_of(artifact, "text/output_head")
        head_bytes[name] = bytes(artifact.words(head["id"]))
        errors += case_errors
        print(f"{name}: {'ok' if not case_errors else 'FAILED'} (head {head['format']} "
              f"{head['shape']}, {head['bytes']} bytes)", flush=True)
    if len(set(head_bytes.values())) > 1:
        errors.append("the head's words differ between the recipes, --vocab and calibration")
    errors += publisher_case(work, native, repository)
    return errors


def publisher_case(work: Path, native: Path, repository: Path) -> list[str]:
    """ModelOpt's hf_quant_config.json says which activation width the publisher validated: W4A16
    everywhere but layer 1's key (NVFP4, 4-bit). provenance.published_activations counts the kept
    matrices per width and those the recorded policy lets run below it (layer 1's calibrated value
    and query, A4 under W4A16); a checkpoint without the file records nothing."""
    import shutil
    errors: list[str] = []
    root = work / "publisher"
    shutil.copytree(work / "w4a16", root)
    (root / "hf_quant_config.json").write_text(json.dumps({"quantization": {
        "quant_algo": "W4A16_NVFP4",
        "quantized_layers": {"model.layers.1.self_attn.k_proj": {"quant_algo": "NVFP4"}}}}),
        encoding="utf-8")
    out = work / "publisher.ninferno"
    result = convert(native, repository, root, out)
    if result.returncode:
        return [f"publisher: conversion failed ({result.returncode}): {result.stderr.strip()[:600]}"]
    published = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))[
        "provenance"].get("published_activations")
    if (not published or published.get("source") != "hf_quant_config.json"
            or published.get("4") != 1 or not published.get("16") or not published.get("below")):
        errors.append(f"publisher: published_activations {published}")
    plain = json.loads(Path(str(work / "default.ninferno") + ".conversion.json").read_text(
        encoding="utf-8"))["provenance"]
    if "published_activations" in plain:
        errors.append(f"publisher: a checkpoint without hf_quant_config.json recorded "
                      f"{plain['published_activations']}")
    print(f"publisher: {'ok' if not errors else 'FAILED'} {published}", flush=True)
    return errors


def moe_cases(work: Path, native: Path, repository: Path) -> list[str]:
    errors: list[str] = []
    tensors = moe_checkpoint(work / "moe-shared", split_divisors=False)
    out = work / "moe-shared.ninferno"
    result = convert(native, repository, work / "moe-shared", out)
    if result.returncode:
        errors.append(f"moe-shared: conversion failed ({result.returncode}): "
                      f"{result.stderr.strip()[:600]}")
    else:
        artifact = ss.Artifact(out)
        pairs = {}
        for layer in range(2):
            for e in range(EXPERTS):
                for role in ("gate", "up", "down"):
                    pairs[f"text/layers/{layer}/mlp/experts/{e}/{role}"] = \
                        f"model.layers.{layer}.mlp.experts.{e}.{role}_proj"
            banks = {"gate": ([EXPERTS * 2 * MOE_INTER, HIDDEN], 1),
                     "down": ([EXPERTS * HIDDEN, MOE_INTER], EXPERTS)}
            for role, (shape, divisors) in banks.items():
                bank = object_of(artifact, f"text/layers/{layer}/mlp/experts/0/{role}")
                got = (bank["format"], bank["shape"], bank.get("divisors", 1))
                if got != ("nvfp4", shape, divisors):
                    errors.append(f"moe-shared: layer {layer} {role} bank is {got}, expected "
                                  f"nvfp4 {shape} of {divisors} divisors")
        errors += ss.same_values(artifact, tensors, "modelopt", pairs, "")
    print(f"moe-shared: {'ok' if not errors else 'FAILED'}", flush=True)

    moe_checkpoint(work / "moe-split", split_divisors=True)
    split = []
    for extra in ((), ("--maintainer-quantize",)):
        refused = convert(native, repository, work / "moe-split", work / "moe-split.ninferno", *extra)
        if (refused.returncode == 0 or "text/layers/0/mlp/experts/0/gate" not in refused.stderr
                or "not whole 128-row scale tiles" not in refused.stderr
                or "different weight divisors" not in refused.stderr):
            split.append(f"moe-split {' '.join(extra)}: expected a refusal naming the bank, got "
                         f"rc {refused.returncode}: {refused.stderr.strip()[-600:]}")
    print(f"moe-split: {'ok' if not split else 'FAILED'}", flush=True)
    return errors + split


def qwen_case(work: Path, native: Path, repository: Path) -> list[str]:
    """qwen3_5_published over NVFP4 MLPs whose rows end inside a scale tile."""
    import synthetic_qwen3_5 as qwen

    root = work / "qwen3_5"
    default_inter, qwen.INTER = qwen.INTER, INTER
    try:
        qwen.base_checkpoint(root, moe_model=False, nested=True, tied=False, vision=False,
                             seed=41)
    finally:
        qwen.INTER = default_inter
    tensors = ss.read_checkpoint(root)
    w = qwen.Q(43)
    stems = {f"model.language_model.layers.{i}.mlp.": f"text/layers/{i}/mlp/"
             for i in range(qwen.LAYERS)}
    stems["mtp.layers.0.mlp."] = "mtp/layers/0/mlp/"
    pairs, parent_rows = {}, {}
    for index, (stem, parameter) in enumerate(stems.items()):
        # Layer 0's gate and up share a divisor: one packed parent.
        packed = 2 * INTER if index == 0 else INTER
        for role, (n, k), divisor, rows in (
                ("gate", (INTER, qwen.H), 0.35, packed),
                ("up", (INTER, qwen.H), 0.35 if index == 0 else 0.45, packed),
                ("down", (qwen.H, INTER), 0.5, qwen.H)):
            del tensors[f"{stem}{role}_proj.weight"]
            w.nvfp4(f"{stem}{role}_proj", n, k, divisor, 0.8)
            pairs[parameter + role] = f"{stem}{role}_proj"
            parent_rows[parameter + role] = rows
    for name, (dtype, shape, data) in tensors.items():
        w.raw(name, dtype, shape, data)
    for file in root.glob("*.safetensors"):
        file.unlink()
    (root / "model.safetensors.index.json").unlink(missing_ok=True)
    w.save(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {"format": "nvfp4-pack-quantized"}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    out = work / "qwen3_5.ninferno"
    result = convert(native, repository, root, out, "--recipe=qwen3_5_published",
                     "--components=text,mtp", "--no-drafters")
    errors = []
    if result.returncode:
        errors.append(f"qwen3_5: conversion failed ({result.returncode}): "
                      f"{result.stderr.strip()[:600]}")
    else:
        artifact = ss.Artifact(out)
        for parameter, rows in parent_rows.items():
            got = object_of(artifact, parameter)
            if (got["format"], got["shape"][0]) != ("nvfp4", rows):
                errors.append(f"qwen3_5: {parameter} is {got['format']} {got['shape']}, "
                              f"expected nvfp4 of {rows} rows")
        errors += ss.same_values(artifact, ss.read_checkpoint(root), "ct", pairs, "")
    print(f"qwen3_5: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def qwen_moe_planner_case(work: Path, native: Path, repository: Path) -> list[str]:
    """The planner's own guard: a stacked NVFP4 parent of several divisors over sources that
    end inside a scale tile."""
    import synthetic_qwen3_5 as qwen

    model, quantized = work / "qwen-moe", work / "qwen-moe-nvfp4"
    default_moe, qwen.MOE_INTER = qwen.MOE_INTER, MOE_INTER
    try:
        qwen.base_checkpoint(model, moe_model=True, nested=False, tied=True, vision=False,
                             seed=47)
        qwen.quantized_moe(quantized, seed=48)
    finally:
        qwen.MOE_INTER = default_moe
    refused = convert(native, repository, model, work / "qwen-moe.ninferno",
                      "--recipe=qwen3_6_35b_a3b_nvfp4", "--components=text",
                      f"--source=quantized={quantized}")
    errors = []
    if (refused.returncode == 0 or "text/layers/0/moe/experts/0/" not in refused.stderr
            or "whole 128-row scale tiles" not in refused.stderr
            or f"not {MOE_INTER} rows" not in refused.stderr):
        errors.append(f"qwen-moe-planner: expected the planner to refuse the bank, got rc "
                      f"{refused.returncode}: {refused.stderr.strip()[-600:]}")
    print(f"qwen-moe-planner: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def python_tail(work: Path) -> list[str]:
    """TensorOutput.write_codes of a matrix whose last 128-row block is ragged."""
    from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
    from tools.artifact.reader import Artifact
    from tools.artifact.tensor_output import TensorOutput
    from tools.artifact.writer import ArtifactWriter
    from tools.convert.methods import import_encoded
    from tools.convert.model import Model, Parameter
    from tools.convert.recipe import Recipe
    from tools.convert.sources.logical import EncodedRows, LogicalSource

    rows, k = 200, 128
    generator = torch.Generator().manual_seed(5)
    codes = torch.randint(0, 256, (rows, k // 2), generator=generator).to(torch.uint8)
    scales = torch.randint(1, 0x7F, (rows, k // 16), generator=generator).to(torch.uint8)
    word = struct.pack("<f", 128.0)

    def no_values(begin, end):
        raise AssertionError("an encoded import reads no decoded values")

    source = LogicalSource((rows, k), "tail", no_values,
                           lambda begin, end: EncodedRows("nvfp4", codes[begin:end],
                                                          scales[begin:end], word),
                           lambda: word, None)
    model = Model({"text": {"config": {}}})
    model.add(Parameter("tail", (rows, k), source))
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    path = work / "python-tail.ninferno"
    with ArtifactWriter(path, [job.spec for job in prepared.weights],
                        components=model.components, bindings=prepared.bindings,
                        uses=prepared.uses) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
    errors = []
    with Artifact(path) as artifact:
        payload = artifact.read_object(prepared.weights[0].spec.id)
    if payload != encode_nvfp4(codes, scales, word, (rows, k)):
        errors.append("python-tail: the streamed object differs from the codec's encoding")
    got_codes, got_scales, divisor = decode_nvfp4_words(payload, (rows, k))
    if (not torch.equal(got_codes, codes) or not torch.equal(got_scales, scales)
            or divisor.item() != 128.0):
        errors.append("python-tail: the object does not decode to the streamed words")
    print(f"python-tail: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else HERE.parents[1]).resolve()
    sys.path.insert(0, str(repository))
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-nvfp4-head-") as tmp:
        work = Path(tmp)
        errors = dense_cases(work, native, repository)
        errors += moe_cases(work, native, repository)
        errors += qwen_case(work, native, repository)
        errors += qwen_moe_planner_case(work, native, repository)
        errors += python_tail(work)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
