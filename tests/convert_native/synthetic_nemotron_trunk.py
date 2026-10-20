"""A NemotronH MoE trunk GGUF that also carries its MTP (NextN) block, as unsloth publishes
NVIDIA-Nemotron-3.5-Lightning-30B-A3B (block_count 53, the attention+MoE MTP block at blk.52,
nextn_predict_layers 1), through the native converter.

    python synthetic_nemotron_trunk.py NINFERNO_CONVERT [REPOSITORY]

The synthetic trunk has four layers (Mamba2, MoE, attention, MoE) and a fifth block, the MTP
block, whose per-layer attention.head_count_kv and feed_forward_length are both nonzero and which
carries the nextn.* tensors. llama.cpp's graph stops before it, so the converted target must have
the four trunk layers only: num_hidden_layers 4, layer_types of those four, and no binding of a
fifth layer. The same trunk without the MTP block (nextn_predict_layers absent, block_count 4)
converts to the same text config and the same bindings. Both record the mapping's version
(provenance gguf_mapper nemotron_h_moe/2). A trunk that declares nextn_predict_layers without the
MTP block (no blk.<trunk>.nextn.eh_proj.weight) is refused rather than cut short. A trunk whose routed up
bank is zero-padded past the hidden width (nemotron_h_moe.moe_expert_embedding_length_padded, as
the padded IQ3_XXS GGUF of Lightning) converts with moe.expert_input_size, and the padded key
absent the text config has none. The trunk's MTP block converts alone as the drafter
(--components mtp): a nemotron_h_mtp text component, with moe.expert_input_size when the trunk
pads its up banks; a trunk without the block is refused naming the NextN block. The converted
target takes drafters (--attach-drafters exits 0, no drafter planned); the same artifact without
its gguf_mapper, as converted before the versions, at a depth no registry model has, is refused
with a hint to reconvert it (exit 1), with no --model-id; with a source directory in place of the
.gguf file as well (a Hugging Face checkpoint's), the artifact is not a GGUF conversion
(--plan-drafters: gguf_source false) and takes drafters.
Exits 77 (skipped) without numpy or gguf.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from artifact_directory import directory

try:
    import numpy as np
    import gguf
    from gguf import GGMLQuantizationType as Q
except ImportError:
    sys.exit(77)

H, HEADS, KV, D = 192, 4, 2, 48
EXPERTS, FF, SHARED = 4, 64, 128
M_HEADS, M_DIM, GROUPS, STATE = 8, 16, 2, 16
INNER = M_HEADS * M_DIM
PADDED = H + 64  # a padded up bank's K
CONV = INNER + 2 * GROUPS * STATE
VOCAB = 96
KINDS = ["mamba", "moe", "attention", "moe"]
LAYERS = len(KINDS)


def write_trunk(path: Path, with_mtp: bool, stray_nextn: bool = False,
                padded: bool = False, vocabulary: bool = False) -> None:
    rng = np.random.default_rng(7)  # the same trunk tensors with or without the MTP block
    blocks = LAYERS + (1 if with_mtp else 0)
    w = gguf.GGUFWriter(path, "nemotron_h_moe")
    w.add_block_count(blocks)
    w.add_context_length(4096)
    w.add_embedding_length(H)
    w.add_head_count(HEADS)
    kv = [KV if k == "attention" else 0 for k in KINDS]
    # A padded trunk widens the expert banks past the per-layer width it keeps (pirola: 1856 in
    # feed_forward_length, 2048 in expert_feed_forward_length).
    layer_ff = FF - 8 if padded else FF
    ff = [layer_ff if k == "moe" else 0 for k in KINDS]
    if with_mtp:  # the MTP block: attention and MoE both
        kv.append(KV)
        ff.append(layer_ff)
    w.add_head_count_kv(kv)
    w.add_feed_forward_length(ff)
    w.add_key_length(D)
    w.add_value_length(D)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_ssm_conv_kernel(4)
    w.add_ssm_state_size(STATE)
    w.add_ssm_group_count(GROUPS)
    w.add_ssm_time_step_rank(M_HEADS)
    w.add_ssm_inner_size(INNER)
    w.add_expert_count(EXPERTS)
    w.add_expert_used_count(2)
    w.add_expert_feed_forward_length(FF)
    w.add_expert_shared_feed_forward_length(SHARED)
    w.add_expert_shared_count(1)
    w.add_expert_group_count(1)
    w.add_expert_group_used_count(1)
    w.add_expert_weights_norm(True)
    w.add_float32("nemotron_h_moe.expert_weights_scale", 2.5)
    if with_mtp or stray_nextn:
        w.add_nextn_predict_layers(1)
    w.add_vocab_size(VOCAB)
    if padded:
        w.add_uint32("nemotron_h_moe.moe_expert_embedding_length_padded", PADDED)
    if vocabulary:
        # As unsloth's NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF writes it: byte-level BPE under
        # llama.cpp's pre "pixtral" (nvidia's tokenizer.json: the tekken Split regex,
        # ignore_merges), add_bos_token false; the repository has no tokenizer files.
        tokens = [f"tok{i}" for i in range(VOCAB)]
        tokens[6], tokens[7], tokens[8] = "x", "9", "8"
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("pixtral")
        w.add_token_list(tokens)
        w.add_token_types([3, 3, 3, 3] + [1] * (VOCAB - 4))
        w.add_token_merges(["x 9", "9 8", "x 8"])
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)
        w.add_add_bos_token(False)
        w.add_chat_template("{% for m in messages %}{{ m['content'] }}{% endfor %}\n")

    def q4(name, rows, k):
        n = rows * k // 32
        raw = rng.integers(0, 256, (n, 18), dtype=np.uint8)
        raw[:, 0:2] = rng.uniform(2e-4, 2e-3, n).astype(np.float16).view(np.uint8).reshape(n, 2)
        w.add_tensor(name, raw.reshape(rows, k // 32 * 18), raw_dtype=Q.Q4_0)

    def f32(name, shape, negative=False):
        values = rng.standard_normal(shape).astype(np.float32) * 0.05
        w.add_tensor(name, -np.abs(values) - 0.01 if negative else values)

    q4("token_embd.weight", VOCAB, H)
    q4("output.weight", VOCAB, H)
    f32("output_norm.weight", H)

    def attention(s):
        q4(s + "attn_q.weight", HEADS * D, H)
        q4(s + "attn_k.weight", KV * D, H)
        q4(s + "attn_v.weight", KV * D, H)
        q4(s + "attn_output.weight", H, HEADS * D)

    def moe(s):
        f32(s + "ffn_gate_inp.weight", (EXPERTS, H))
        f32(s + "exp_probs_b.bias", EXPERTS)
        q4(s + "ffn_up_exps.weight", EXPERTS * FF, PADDED if padded else H)
        q4(s + "ffn_down_exps.weight", EXPERTS * H, FF)
        q4(s + "ffn_up_shexp.weight", SHARED, H)
        q4(s + "ffn_down_shexp.weight", H, SHARED)

    for i, kind in enumerate(KINDS):
        s = f"blk.{i}."
        f32(s + "attn_norm.weight", H)
        if kind == "mamba":
            q4(s + "ssm_in.weight", INNER + CONV + M_HEADS, H)
            f32(s + "ssm_conv1d.weight", (CONV, 4))
            f32(s + "ssm_conv1d.bias", CONV)
            f32(s + "ssm_a", (M_HEADS, 1), negative=True)
            f32(s + "ssm_d", (M_HEADS, 1))
            f32(s + "ssm_dt.bias", M_HEADS)
            f32(s + "ssm_norm.weight", (GROUPS, INNER // GROUPS))
            q4(s + "ssm_out.weight", H, INNER)
        elif kind == "attention":
            attention(s)
        else:
            moe(s)
    if with_mtp:
        s = f"blk.{LAYERS}."
        f32(s + "nextn.enorm.weight", H)
        f32(s + "nextn.hnorm.weight", H)
        q4(s + "nextn.eh_proj.weight", H, 2 * H)
        f32(s + "attn_norm.weight", H)
        attention(s)
        f32(s + "post_attention_norm.weight", H)
        moe(s)
        f32(s + "nextn.shared_head_norm.weight", H)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ninferno-nemotron-trunk-") as tmp:
        work = Path(tmp)
        converted = {}
        for label, with_mtp in (("mtp", True), ("plain", False), ("stray", False),
                                ("padded", False), ("padded_mtp", True)):
            gdir = work / label
            gdir.mkdir()
            path = gdir / "trunk.gguf"
            write_trunk(path, with_mtp, stray_nextn=label == "stray",
                        padded=label.startswith("padded"))
            vocab = {f"tok{i}": i for i in range(VOCAB)}
            added = [{"id": i, "content": f"tok{i}", "special": True} for i in (0, 1, 2, 3)]
            (gdir / "tokenizer.json").write_text(
                json.dumps({"model": {"vocab": vocab}, "added_tokens": added}), encoding="utf-8")
            (gdir / "tokenizer_config.json").write_text(
                json.dumps({"added_tokens_decoder": {"4": {"content": "tok4", "special": True}}}), encoding="utf-8")
            (gdir / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")
            (gdir / "chat_template.jinja").write_text(
                "{% for m in messages %}{{ m['content'] }}{% endfor %}\n", encoding="utf-8")
            out = work / f"{label}.ninferno"
            result = subprocess.run(
                [str(native), f"--model={path}", f"--out={out}", "--quiet"],
                cwd=repository, capture_output=True, text=True)
            if label == "stray":
                said = result.stdout + result.stderr
                if result.returncode == 0 or "nextn.eh_proj.weight" not in said:
                    errors.append(f"a stray nextn_predict_layers: rc {result.returncode}, expected a "
                                  f"refusal naming the absent NextN tensor: {said[-400:]}")
                continue
            if result.returncode:
                print(result.stdout[-3000:])
                print(result.stderr[-3000:], file=sys.stderr)
                return 1
            converted[label] = directory(out)[0]
            if label in ("mtp", "plain", "padded_mtp"):
                drafter = work / f"{label}_drafter.ninferno"
                result = subprocess.run(
                    [str(native), f"--model={path}", f"--out={drafter}", "--components=mtp",
                     "--quiet"], cwd=repository, capture_output=True, text=True)
                said = result.stdout + result.stderr
                if label == "plain":
                    if result.returncode == 0 or "NextN" not in said:
                        errors.append(f"--components mtp of a trunk without the MTP block: rc "
                                      f"{result.returncode}, expected a refusal naming the NextN "
                                      f"block: {said[-400:]}")
                    continue
                if result.returncode:
                    errors.append(f"{label}: --components mtp rc {result.returncode}: {said[-600:]}")
                    continue
                text = directory(drafter)[0]["components"]["text"]["config"]
                width = text.get("moe", {}).get("expert_input_size")
                want = PADDED if label == "padded_mtp" else None
                if text.get("model_type") != "nemotron_h_mtp" or width != want:
                    errors.append(f"{label}: the drafter from the trunk is {text.get('model_type')!r} "
                                  f"with moe.expert_input_size {width!r}, expected nemotron_h_mtp "
                                  f"with {want!r}")

        value = converted["mtp"]
        config = value["components"]["text"]["config"]
        if config.get("num_hidden_layers") != LAYERS:
            errors.append(f"num_hidden_layers {config.get('num_hidden_layers')}, expected {LAYERS}: "
                          "the MTP block counted as a trunk layer")
        if config.get("layer_types") != KINDS:
            errors.append(f"layer_types {config.get('layer_types')}, expected {KINDS}")
        extra = sorted(n for n in value["bindings"] if n.startswith(f"text/layers/{LAYERS}/"))
        if extra:
            errors.append(f"bindings of a layer past the trunk: {extra[:6]}")
        for label, converted_value in converted.items():
            mapper = converted_value.get("provenance", {}).get("gguf_mapper")
            if mapper != "nemotron_h_moe/2":
                errors.append(f"{label}: provenance gguf_mapper {mapper!r}, expected 'nemotron_h_moe/2'")
        plain = converted["plain"]
        padded_moe = converted["padded"]["components"]["text"]["config"]["moe"]
        if padded_moe.get("expert_input_size") != PADDED:
            errors.append(f"padded up bank: moe.expert_input_size "
                          f"{padded_moe.get('expert_input_size')!r}, expected {PADDED}")
        if "expert_input_size" in config["moe"]:
            errors.append("an unpadded trunk's text config has moe.expert_input_size")
        if plain["components"]["text"]["config"] != config:
            errors.append("the text config differs from the same trunk without the MTP block")
        if sorted(plain["bindings"]) != sorted(value["bindings"]):
            only = sorted(set(value["bindings"]) ^ set(plain["bindings"]))
            errors.append(f"the bindings differ from the same trunk without the MTP block: {only[:6]}")

        # --attach-drafters without --model-id: the versioned conversion is fine, the same artifact
        # without its gguf_mapper (the key renamed in place, the directory's length unchanged) is
        # stale.
        current = work / "mtp.ninferno"
        stale = work / "stale.ninferno"
        data = bytearray(current.read_bytes())
        at = data.find(b'"gguf_mapper"', 32, 32 + struct.unpack("<Q", data[8:16])[0])
        data[at:at + 13] = b'"gguf_mappeX"'
        stale.write_bytes(bytes(data))
        # A Hugging Face-sourced artifact records its checkpoint directory: the base path's
        # "trunk.gguf" renamed in place to the directory "trunk_gguf".
        checkpoint = work / "checkpoint.ninferno"
        end = 32 + struct.unpack("<Q", data[8:16])[0]
        data[32:end] = bytes(data[32:end]).replace(b'trunk.gguf"', b'trunk_gguf"')
        checkpoint.write_bytes(bytes(data))
        for path, source in ((current, True), (stale, True), (checkpoint, False)):
            result = subprocess.run([str(native), "--plan-drafters", f"--target-artifact={path}"],
                                    cwd=repository, capture_output=True, text=True)
            try:
                target = json.loads(result.stdout)["target"]
            except (ValueError, KeyError):
                target = {}
            if result.returncode or target.get("gguf_source") is not source:
                errors.append(f"--plan-drafters {path.name}: rc {result.returncode}, gguf_source "
                              f"{target.get('gguf_source')!r}, expected {source}: "
                              f"{(result.stdout + result.stderr)[-600:]}")
        for path, want in ((current, 0), (stale, 1), (checkpoint, 0)):
            result = subprocess.run([str(native), "--attach-drafters", f"--target-artifact={path}"],
                                    cwd=repository, capture_output=True, text=True)
            said = result.stdout + result.stderr
            if result.returncode != want or (want and "reconvert" not in said):
                errors.append(f"--attach-drafters {path.name}: rc {result.returncode}, expected "
                              f"{want}{' and a hint to reconvert' if want else ''}: {said[-600:]}")
        # No tokenizer files beside the GGUF (a hub snapshot of unsloth's repository): they are
        # rebuilt from tokenizer.ggml. Before "pixtral" was a known pre-tokenizer the conversion
        # stopped on "cannot read <dir>/tokenizer.json".
        gdir = work / "rebuilt"
        gdir.mkdir()
        write_trunk(gdir / "trunk.gguf", True, vocabulary=True)
        out = work / "rebuilt.ninferno"
        result = subprocess.run([str(native), f"--model={gdir / 'trunk.gguf'}", f"--out={out}", "--quiet"],
                                cwd=repository, capture_output=True, text=True)
        said = result.stdout + result.stderr
        if result.returncode:
            errors.append(f"rebuilt: conversion without tokenizer files failed: {said[-800:]}")
        else:
            _, objects, data, offset = directory(out)

            def resource(role):
                o = objects.get(f"resource/text/{role}")
                return None if o is None else json.loads(
                    data[offset + o["offset"]:offset + o["offset"] + o["bytes"]])

            tokenizer, tokenizer_config = resource("tokenizer.json"), resource("tokenizer_config.json")
            steps = (tokenizer or {}).get("pre_tokenizer", {}).get("pretokenizers", [])
            regex = steps[0].get("pattern", {}).get("Regex") if steps else None
            if not tokenizer or not regex or r"[\r\n/]*" not in regex or \
                    tokenizer["model"].get("ignore_merges") is not True:
                errors.append(f"rebuilt: tokenizer.json is not the tekken one: {str(tokenizer)[:300]}")
            if (tokenizer_config or {}).get("add_bos_token") is not False:
                errors.append(f"rebuilt: tokenizer_config.json {str(tokenizer_config)[:300]}, "
                              "expected add_bos_token false")
            # The drafter path: the trunk's own MTP block, its frontend files rebuilt as well.
            result = subprocess.run([str(native), f"--model={gdir / 'trunk.gguf'}",
                                     f"--out={work / 'rebuilt_drafter.ninferno'}", "--components=mtp",
                                     "--quiet"], cwd=repository, capture_output=True, text=True)
            if result.returncode:
                errors.append(f"rebuilt: --components mtp without tokenizer files: "
                              f"{(result.stdout + result.stderr)[-600:]}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"nemotron-trunk: {'ok' if not errors else 'FAILED'}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
