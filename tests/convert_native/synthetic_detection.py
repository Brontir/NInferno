"""Quantized or not is decided by the tensors, not by a config string or a name list;
a model runs as published.

    python synthetic_detection.py NINFERNO_CONVERT [REPOSITORY]

Every conversion runs without --recipe, as a user would:

  qwen-nvfp4-no-config   a Qwen3.5 MoE checkpoint whose experts and shared expert are NVFP4 and
                         whose config has no quantization_config: the NVFP4 recipe is chosen,
                         every expert matrix kept as NVFP4, and the recipe note names the
                         disagreement;
  qwen-nvfp4-config      the same with quantization_config (and an ignore list that names only
                         lm_head while mlp.gate, the shared-expert gate and mtp.* stay BF16): the
                         NVFP4 recipe, no note (the control);
  qwen-bf16-claims       a BF16 MoE checkpoint whose config claims NVFP4: the plain recipe with a
                         note, instead of a refusal for missing NVFP4 tensors; the MoE has no
                         unquantized recipe, so the conversion is then refused as NInferno's own
                         quantization (and runs with --maintainer-quantize);
  qwen-bf16-shared       NVFP4 routed experts with a BF16 shared expert: qwen3_6_35b_a3b_nvfp4 has
                         no NVFP4 source for the shared expert, so qwen3_5_published runs it as
                         published (routed NVFP4, shared BF16: the generic MoE op), with a note;
  qwen-cached-text       NVFP4 experts with FP8 MTP attention: --components=text plans the text
                         alone and keeps qwen3_6_35b_a3b_nvfp4 (converted into the checkpoint's
                         cache); a default conversion keeps the MTP layers, which that text-only
                         artifact proves nothing about, so the planner runs and chooses
                         qwen3_5_published; with mtp recorded in the cached artifact's report,
                         its recipe answers without planning;
  gemma-bf16-layer       a Gemma 4 MoE checkpoint whose layer 1 is BF16 throughout (its experts
                         too) while layer 0 is NVFP4 and quantization_config ignores nothing: layer
                         0 imported as NVFP4, layer 1 quantized from BF16, per site;
  gemma-mlp-packed       Gemma 4 MoE whose shared MLP gate/up (192 rows each, not whole 128-row
                         scale tiles) share one divisor: kept as NVFP4 in one packed 384-row
                         parent; with different divisors they cannot be packed and each is kept
                         as an NVFP4 parent of its own, whose last scale tile the layout completes;
  dense-bf16             a Llama checkpoint in BF16: dense_bf16, every projection kept BF16;
                         --recipe dense_q8 refused naming dense_bf16, and forced with
                         --maintainer-quantize;
  dense-<encoding>       a Llama checkpoint whose projections carry a published quantization and
                         whose config says nothing about it: dense_encoded (chosen by the tensors),
                         and each encoding either kept as published or the conversion refused
                         naming it, never requantized. For AWQ int4 and FP8 the vocabulary is
                         kept native (the checkpoint's BF16 words, byte for byte), and --vocab q8
                         changes exactly the embedding and head objects to q8_g32_fp16.

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
    import numpy as np
    import torch  # noqa: F401
except ImportError:
    sys.exit(77)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthetic_formats as fmt  # noqa: E402
import synthetic_gemma4 as gemma  # noqa: E402
import synthetic_qwen3_5 as qwen  # noqa: E402
from synthetic_scales import read_checkpoint, rewrite_checkpoint  # noqa: E402


def qwen_moe(root: Path, *, nvfp4: bool, quantization_config, bf16_shared=False) -> None:
    qwen.base_checkpoint(root, moe_model=True, nested=True, tied=True, vision=False, seed=22)
    if nvfp4:
        quantized = root.parent / (root.name + "-quantized")
        qwen.quantized_moe(quantized, seed=25)
        encoded = read_checkpoint(quantized)

        def change(t):
            # The MTP layer stays BF16 (as published NVFP4 checkpoints keep it): only the model's
            # layers take the encoded banks.
            for name in [n for n in t if (".mlp.experts" in n or ".mlp.shared_expert." in n)
                         and not n.startswith("mtp.")]:
                if not (bf16_shared and ".shared_expert." in name):
                    del t[name]
            for name, value in encoded.items():
                if not (bf16_shared and ".shared_expert." in name):
                    t[name] = value
        rewrite_checkpoint(root, change)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def fp8_mtp_attention(root: Path) -> None:
    """The MTP layer's attention projections published as FP8 (channel scales)."""
    rng = np.random.default_rng(27)
    encode = fmt.enc_fp8("channel", "BF16")

    def change(t):
        for name in [n for n in t if n.startswith("mtp.layers.0.self_attn.")
                     and n.endswith("_proj.weight")]:
            n, k = t[name][1]
            del t[name]
            for suffix, value in encode(n, k, rng).q.items():
                t[name[:-len(".weight")] + suffix] = value
    rewrite_checkpoint(root, change)


def gemma_mixed(root: Path) -> None:
    gemma.text_checkpoint(root, moe=True, layout="ct", nested=True, tied=True, seed=9)
    bf16 = root.parent / (root.name + "-bf16")
    gemma.text_checkpoint(bf16, moe=True, layout="modelopt", nested=True, tied=True, seed=10)
    plain = read_checkpoint(bf16)
    # layout="modelopt" encodes the experts; make them BF16 in layer 1 by hand.
    def change(t):
        for name in [n for n in t if ".layers.1." in n]:
            del t[name]
        for name, value in plain.items():
            if ".layers.1." in name and not name.endswith((".weight_scale", ".weight_scale_2",
                                                           ".input_scale")):
                t[name] = value
        g = torch.Generator().manual_seed(3)
        for expert in range(4):
            p = f"model.language_model.layers.1.experts.{expert}."
            for role, shape in (("gate_proj", (64, 256)), ("up_proj", (64, 256)), ("down_proj", (256, 64))):
                t.pop(p + role + ".weight", None)
                values = (torch.randn(shape, generator=g) * 0.05).to(torch.bfloat16)
                t[p + role + ".weight"] = ("BF16", list(shape), values.view(torch.uint8).numpy().tobytes())
    rewrite_checkpoint(root, change)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {"format": "nvfp4-pack-quantized", "ignore": []}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def convert(native: Path, repository: Path, model: Path, out: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run([str(native), f"--model={model}", f"--out={out}", *extra], cwd=repository,
                          capture_output=True, text=True)


def dense_checkpoint(root: Path, encode, seed: int) -> None:
    """A two-layer Llama (GQA 4/2, head 64) whose seven projections per layer come from
    `encode` (a synthetic_formats encoder); embeddings, norms and the head in BF16."""
    rng = np.random.default_rng(seed)
    hidden, inter, heads, kv, d = 256, 512, 4, 2, 64
    config = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": hidden,
              "intermediate_size": inter, "num_hidden_layers": 2, "num_attention_heads": heads,
              "num_key_value_heads": kv, "head_dim": d, "vocab_size": gemma.VOCAB,
              "max_position_embeddings": 4096, "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
              "tie_word_embeddings": False, "hidden_act": "silu"}
    tensors = {}

    def plain(name, shape):
        tensors[name] = ("BF16", list(shape),
                         fmt.bf16_bytes(rng.standard_normal(shape).astype(np.float32) * 0.05))

    plain("model.embed_tokens.weight", (gemma.VOCAB, hidden))
    plain("lm_head.weight", (gemma.VOCAB, hidden))
    plain("model.norm.weight", (hidden,))
    for layer in range(2):
        p = f"model.layers.{layer}."
        for role, (n, k) in {"self_attn.q_proj": (heads * d, hidden), "self_attn.k_proj": (kv * d, hidden),
                             "self_attn.v_proj": (kv * d, hidden), "self_attn.o_proj": (hidden, heads * d),
                             "mlp.gate_proj": (inter, hidden), "mlp.up_proj": (inter, hidden),
                             "mlp.down_proj": (hidden, inter)}.items():
            tensors.update({p + role + s: v for s, v in encode(n, k, rng).q.items()})
        plain(p + "input_layernorm.weight", (hidden,))
        plain(p + "post_attention_layernorm.weight", (hidden,))
    fmt.write_checkpoint(root, tensors, config)


# Published encodings of a dense checkpoint without --recipe: what each must become.
#   name -> (encoder, None: refused naming `encoding`, else the format every projection keeps)
DENSE_ENCODINGS = {
    "fp8-channel-bf16": (fmt.enc_fp8("channel", "BF16"), "fp8_e4m3fn_row_bf16", "fp8"),
    "fp8-tensor-f32": (fmt.enc_fp8("tensor", "F32"), "fp8_e4m3fn_row_f32", "fp8"),
    "fp8-block128": (fmt.enc_fp8_block, "fp8_e4m3fn_b128_f32", "fp8"),
    "awq-int4-g128": (fmt.enc_awq(128), "u4_g128_zp_fp16", "awq int4"),
    "gptq-int4-g128": (fmt.enc_gptq(128, act_order=False), "u4_g128_zp_fp16", "gptq int4"),
    "gptq-int4-g128-actorder": (fmt.enc_gptq(128), "u4_g128_zp_fp16", "gptq int4"),
    "gptq-int8-g128-actorder": (fmt.enc_gptq(128, bits=8, sym=True), None, "gptq int8"),
    "ct-int8-channel": (fmt.enc_ct_int8("F32"), None, "compressed-tensors int8"),
    "ct-int8-channel-bf16": (fmt.enc_ct_int8("BF16"), "q8_g32_fp16", "compressed-tensors int8"),
    "gptq-int4-g32": (fmt.enc_gptq(32, act_order=False), "u4_g32_zp_fp16", "gptq int4"),
    "ct-int4-g64-sym": (fmt.enc_ct_int(4, 64, False, False), "u4_g64_zp_fp16", "int4"),
    "gptq-int8-g128": (fmt.enc_gptq(128, act_order=False, bits=8, sym=True), "q8_g32_fp16", "gptq int8"),
}


def dense_cases(work: Path, native: Path, repository: Path, case) -> None:
    root = work / "dense-bf16"
    dense_checkpoint(root, fmt.enc_plain("BF16"), 31)
    out = work / "dense-bf16.ninferno"
    run = convert(native, repository, root, out)
    errors = []
    if run.returncode:
        errors.append(f"conversion failed: {run.stderr.strip()[-400:]}")
    else:
        if "recipe dense_bf16 (chosen" not in run.stdout:
            errors.append(f"expected dense_bf16: {[l for l in run.stdout.splitlines() if 'recipe' in l]}")
        body = {(m["format"], m["method"]) for m in methods(out)
                if all(p.startswith("text/layers/") for p in m["parameters"])}
        if body != {("bf16", "cast_direct")}:
            errors.append(f"body not kept as published BF16: {sorted(body)}")
    forced_out = work / "dense-bf16-q8.ninferno"
    refused = convert(native, repository, root, work / "dense-bf16-refused.ninferno", "--recipe=dense_q8")
    if (refused.returncode == 0 or "--recipe dense_bf16" not in refused.stderr
            or "--maintainer-quantize" not in refused.stderr):
        errors.append(f"dense_q8 of a BF16 checkpoint was not refused naming dense_bf16: {refused.stderr[-500:]}")
    forced = convert(native, repository, root, forced_out, "--recipe=dense_q8", "--maintainer-quantize")
    if forced.returncode or "--maintainer-quantize: this artifact is NInferno's own quantization" not in forced.stdout:
        errors.append(f"dense_q8 --maintainer-quantize: rc {forced.returncode} {forced.stderr[-400:]}")
    case("dense-bf16", errors)

    for index, (name, (encode, kept, encoding)) in enumerate(DENSE_ENCODINGS.items()):
        root = work / f"dense-{name}"
        dense_checkpoint(root, encode, 40 + index)
        out = work / f"dense-{name}.ninferno"
        run = convert(native, repository, root, out)
        errors = []
        if "recipe dense_encoded (chosen" not in run.stdout:
            errors.append(f"expected dense_encoded: {[l for l in run.stdout.splitlines() if 'recipe' in l]}")
        if kept is None:
            if run.returncode == 0 or "would not run" not in run.stderr or encoding not in run.stderr:
                errors.append(f"expected a refusal naming {encoding!r}, got rc {run.returncode}: {run.stderr[-500:]}")
        elif run.returncode:
            errors.append(f"conversion failed: {run.stderr.strip()[-400:]}")
        else:
            body = {(m["format"], m["method"]) for m in methods(out)
                    if all(p.startswith("text/layers/") for p in m["parameters"])
                    and not m["format"] in ("bf16", "fp32")}
            if body != {(kept, "import_encoded")}:
                errors.append(f"projections not kept as {kept}: {sorted(body)}")
            if name in VOCAB_CASES:
                errors += vocab_check(native, repository, root, out, work / f"dense-{name}-q8.ninferno")
        case(f"dense-{name}", errors)


# Published quantizations whose vocabulary --vocab is checked on: int4 AWQ and row-scaled FP8.
VOCAB_CASES = ("awq-int4-g128", "fp8-channel-bf16")
VOCAB = {"text/token_embedding": "model.embed_tokens.weight", "text/output_head": "lm_head.weight"}


def vocab_check(native: Path, repository: Path, root: Path, out: Path, q8_out: Path) -> list[str]:
    """The default conversion `out` keeps the vocabulary native: provenance says so and the two
    objects are the checkpoint's BF16 words byte for byte. --vocab q8 of the same checkpoint differs
    from it only in those two objects, now q8_g32_fp16."""
    errors = []
    run = convert(native, repository, root, q8_out, "--vocab=q8")
    if run.returncode:
        return [f"--vocab q8 failed: {run.stderr.strip()[-400:]}"]
    source = read_checkpoint(root)
    arms = {}
    for label, path in (("native", out), ("q8", q8_out)):
        value, objects, data, offset = fmt.directory(path)
        if value["provenance"].get("vocabulary") != label:
            errors.append(f"{label}: provenance vocabulary {value['provenance'].get('vocabulary')!r}")
        vocab = {}
        for m in methods(path):
            for parameter in m["parameters"]:
                if parameter in VOCAB:
                    vocab[parameter] = (m["object"], m["format"], m["method"])
        payload = {oid: data[offset + o["offset"]:offset + o["offset"] + o["bytes"]]
                   for oid, o in objects.items()}
        arms[label] = (vocab, objects, payload)
    vocab, objects, payload = arms["native"]
    for parameter, checkpoint_name in VOCAB.items():
        if parameter not in vocab:
            errors.append(f"native: no method for {parameter}")
            continue
        oid, format_, method = vocab[parameter]
        if (format_, method) != ("bf16", "cast_direct") or payload[oid] != source[checkpoint_name][2]:
            errors.append(f"native: {parameter} ({format_}, {method}) is not the checkpoint's BF16 words")
    q8_vocab, q8_objects, q8_payload = arms["q8"]
    if {f for _, f, _ in q8_vocab.values()} != {"q8_g32_fp16"} or len(q8_vocab) != len(VOCAB):
        errors.append(f"q8: vocabulary not q8_g32_fp16: {q8_vocab}")
    vocab_objects = {v[0] for v in vocab.values()} | {v[0] for v in q8_vocab.values()}
    if set(objects) != set(q8_objects):
        errors.append(f"q8: object sets differ: {sorted(set(objects) ^ set(q8_objects))}")
    differing = sorted(oid for oid in set(objects) & set(q8_objects)
                       if payload[oid] != q8_payload[oid] or objects[oid].get("format") != q8_objects[oid].get("format"))
    if not differing or set(differing) - vocab_objects:
        errors.append(f"native and q8 differ in {differing}, expected exactly the vocabulary {sorted(vocab_objects)}")
    return errors


def methods(out: Path) -> list[dict]:
    return json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))["methods"]


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else HERE.parents[1]).resolve()
    failures: list[str] = []

    def case(name: str, errors: list[str]) -> None:
        print(f"{name}: {'ok' if not errors else 'FAILED'}", flush=True)
        failures.extend(f"{name}: {e}" for e in errors)

    with tempfile.TemporaryDirectory(prefix="ninferno-convert-detection-") as tmp:
        work = Path(tmp)
        ignore_only_head = {"format": "nvfp4-pack-quantized", "ignore": ["lm_head"]}
        for name, nvfp4, config, want_recipe, want_note in (
            ("qwen-nvfp4-no-config", True, None, "qwen3_6_35b_a3b_nvfp4", "matrices are NVFP4"),
            ("qwen-nvfp4-config", True, ignore_only_head, "qwen3_6_35b_a3b_nvfp4", None),
            ("qwen-bf16-claims", False, ignore_only_head, "qwen3_6_35b_a3b", "no matrix"),
        ):
            root = work / name
            qwen_moe(root, nvfp4=nvfp4, quantization_config=config)
            out = work / (name + ".ninferno")
            run = convert(native, repository, root, out)
            log = run.stdout + run.stderr
            errors = []
            if not nvfp4:
                # The Qwen3.5 MoE has no unquantized recipe: its Q4/Q5 table on a BF16 checkpoint
                # is NInferno's own quantization, refused unless forced.
                if run.returncode == 0 or "no unquantized recipe" not in run.stderr:
                    errors.append(f"expected a refusal of NInferno's own quantization: {run.stderr[-400:]}")
                run = convert(native, repository, root, out, "--maintainer-quantize")
                log = run.stdout + run.stderr
            if run.returncode:
                errors.append(f"conversion failed: {log.strip().splitlines()[0][:300]}")
            if f"recipe {want_recipe} (chosen" not in log:
                errors.append(f"expected recipe {want_recipe}: {[l for l in log.splitlines() if 'recipe' in l]}")
            notes = [l for l in log.splitlines() if l.startswith("recipe note:")]
            if want_note is None and notes:
                errors.append(f"unexpected note {notes}")
            if want_note is not None and not any(want_note in l for l in notes):
                errors.append(f"no recipe note naming the disagreement: {notes}")
            if not run.returncode and nvfp4:
                # The model's expert banks; the MTP layer's stay BF16 in the checkpoint.
                experts = [m for m in methods(out) if any("/moe/" in p for p in m["parameters"])
                           and not any("mtp" in p for p in m["parameters"])
                           and not any(p.endswith(("/router", "/shared_score")) for p in m["parameters"])]
                if not experts or any(m["format"] != "nvfp4" or m["method"] != "import_encoded" for m in experts):
                    errors.append(f"expert banks not kept as NVFP4: {[(m['format'], m['method']) for m in experts][:4]}")
            case(name, errors)

        root = work / "qwen-bf16-shared"
        qwen_moe(root, nvfp4=True, quantization_config=ignore_only_head, bf16_shared=True)
        out = work / "qwen-bf16-shared.ninferno"
        run = convert(native, repository, root, out)
        log = run.stdout + run.stderr
        errors = []
        if run.returncode:
            errors.append(f"conversion failed: {log.strip().splitlines()[-1][:300]}")
        else:
            if "recipe qwen3_5_published (chosen" not in log or "does not fit this checkpoint" not in log:
                errors.append(f"expected qwen3_5_published with a note: {[l for l in log.splitlines() if 'recipe' in l]}")
            banks = {"routed": set(), "shared": set()}
            for m in methods(out):
                names = m["parameters"]
                if (any("/moe/" in p for p in names) and not any("mtp" in p for p in names)
                        and not any(p.endswith(("/router", "/shared_score")) for p in names)):
                    kind = "shared" if any("/moe/shared/" in p for p in names) else "routed"
                    banks[kind].add((m["format"], m["method"]))
            if banks["routed"] != {("nvfp4", "import_encoded")} or not banks["shared"] or \
                    any(format != "bf16" for format, _ in banks["shared"]):
                errors.append(f"expected every routed bank NVFP4 kept and the shared expert BF16: {banks}")
        case("qwen-bf16-shared", errors)

        root = work / "qwen-cached-text"
        qwen_moe(root, nvfp4=True, quantization_config=ignore_only_head)
        fp8_mtp_attention(root)
        cached = root / ".ninferno-cache" / "qwen3_6_35b_a3b_nvfp4.ninferno"
        cached.parent.mkdir()
        errors = []

        def chose(run, want, label, converts=True):
            log = run.stdout + run.stderr
            if f"recipe {want} (chosen" not in log:
                errors.append(f"{label}: expected {want}: {[l for l in log.splitlines() if 'recipe' in l]}")
            elif converts and run.returncode:
                errors.append(f"{label}: conversion failed: {log.strip().splitlines()[-1][:300]}")

        chose(convert(native, repository, root, cached, "--components=text"),
              "qwen3_6_35b_a3b_nvfp4", "--components=text")
        report_path = Path(str(cached) + ".conversion.json")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if sorted(report["components"]) != ["text"]:
                errors.append(f"text-only report components: {sorted(report['components'])}")
            chose(convert(native, repository, root, work / "qwen-cached-text.ninferno"),
                  "qwen3_5_published", "default after a text-only cache")
            report["components"]["mtp"] = {}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            chose(convert(native, repository, root, work / "qwen-cached-mtp.ninferno"),
                  "qwen3_6_35b_a3b_nvfp4", "default with mtp cached", converts=False)
        case("qwen-cached-text", errors)

        root = work / "gemma-bf16-layer"
        gemma_mixed(root)
        out = work / "gemma-bf16-layer.ninferno"
        run = convert(native, repository, root, out)
        errors = []
        if run.returncode:
            errors.append(f"conversion failed: {(run.stdout + run.stderr).strip().splitlines()[-1][:300]}")
        else:
            by_layer = {0: set(), 1: set()}
            for m in methods(out):
                for p in m["parameters"]:
                    if "/moe/experts/" in p:
                        by_layer[int(p.split("/")[2])].add((m["format"], m["method"]))
            if by_layer[0] != {("nvfp4", "import_encoded")}:
                errors.append(f"layer 0 experts {by_layer[0]}")
            if not by_layer[1] or any(f == "nvfp4" for f, _ in by_layer[1]):
                errors.append(f"layer 1 (BF16) experts {by_layer[1]}")
            print(f"   gemma layer 0 experts {sorted(by_layer[0])}, layer 1 experts {sorted(by_layer[1])}")
        case("gemma-bf16-layer", errors)

        root = work / "gemma-mlp-packed"
        gemma.text_checkpoint(root, moe=True, layout="ct", nested=True, tied=True, seed=11)
        out = work / "gemma-mlp-packed.ninferno"
        run = convert(native, repository, root, out, "--recipe=gemma4_26b_a4b_nvfp4")
        errors = []
        if run.returncode:
            errors.append(f"conversion failed: {run.stderr.strip()[-400:]}")
        else:
            shared = [m for m in methods(out) if m["parameters"] == ["text/layers/0/mlp/gate", "text/layers/0/mlp/up"]]
            if not shared or (shared[0]["format"], shared[0]["method"], shared[0]["shape"]) != (
                    "nvfp4", "import_encoded", [384, 256]):
                errors.append(f"shared MLP gate/up not kept as one NVFP4 parent: {shared}")
        rewrite_checkpoint(root, lambda t: t.__setitem__(
            "model.language_model.layers.0.mlp.up_proj.weight_global_scale",
            ("F32", [1], struct.pack("<f", 0.35))))
        split_out = work / "gemma-mlp-split.ninferno"
        split = convert(native, repository, root, split_out, "--recipe=gemma4_26b_a4b_nvfp4")
        if split.returncode:
            errors.append(f"gate/up of different divisors failed: {split.stderr[-500:]}")
        else:
            for role in ("gate", "up"):
                own = [m for m in methods(split_out)
                       if m["parameters"] == [f"text/layers/0/mlp/{role}"]]
                if not own or (own[0]["format"], own[0]["method"], own[0]["shape"]) != (
                        "nvfp4", "import_encoded", [192, 256]):
                    errors.append(f"shared MLP {role} of its own divisor not kept as NVFP4: {own}")
        case("gemma-mlp-packed", errors)

        dense_cases(work, native, repository, case)

    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
