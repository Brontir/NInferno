"""The NemotronH MTP drafter sidecar as llama.cpp publishes it (a nemotron_h_moe GGUF that carries
only the folded NextN block), through the native converter.

    python synthetic_nemotron_mtp.py NINFERNO_CONVERT [REPOSITORY]

The synthetic sidecar is what llama.cpp's conversion/nemotron.py writes from the HF mtp.* tensors:
general.architecture nemotron_h_moe (the trunk's own string), block_count one past the trunk with
the MTP block folded at the last index, the 53-entry per-layer attention.head_count_kv and
feed_forward_length arrays placing attention (kv = 2) and MoE (ff) on that block and nothing on
the phantom trunk, nextn_predict_layers = 1, the block's tensors (nextn.enorm / hnorm / eh_proj,
attn_*, post_attention_norm, the MoE bank) in Q4_0 with an F32 exp_probs_b.bias and a BF16 [E, H]
router, and the target's token_embd / output / output_norm re-shipped beside them.

Checks against the converted artifact's directory: the text config (model_type nemotron_h_mtp,
the geometry, both epsilons and the MoE routing), every binding's shape and format (Q4_0 kept
word for word, the router BF16 [E, H], the selection bias FP32, the norms BF16), the element range
each of q / k / v covers in its stacked parent and each expert in its bank, the input each
projection reads (the attention's and the MoE's apart), the payload bytes of every Q4_0 binding
and of the router against the source GGUF's, and that no binding of the re-shipped globals
exists. A sidecar without the target's tokenizer files beside it is refused naming --resource.
Exits 77 (skipped) without numpy or gguf.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
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

# A small but faithful geometry: the 30B-A3B shape divided down (2688 -> 192, 128 experts -> 4).
H, HEADS, KV, D = 192, 4, 2, 48
LAYERS, EXPERTS, FF, SHARED = 6, 4, 64, 128
VOCAB = 96
BLOCK = LAYERS + 1  # the trunk's layers + the folded MTP block
# Distinct epsilons, so a swapped pair shows.
RMS_EPS, LAYER_NORM_EPS = 1e-5, 2e-6


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    rng = np.random.default_rng(42)
    errors: list[str] = []
    sources: dict[str, np.ndarray] = {}  # GGUF tensor name -> its stored bytes, row by row
    with tempfile.TemporaryDirectory(prefix="ninferno-nemotron-mtp-") as tmp:
        work = Path(tmp)
        gdir = work / "gguf"
        gdir.mkdir()
        path = gdir / "mtp.gguf"
        w = gguf.GGUFWriter(path, "nemotron_h_moe")
        w.add_block_count(BLOCK)
        w.add_context_length(4096)
        w.add_embedding_length(H)
        w.add_head_count(HEADS)
        # Per-layer arrays: no kv heads and no ff on the phantom trunk (all zero), attention on
        # the MTP block; feed_forward_length is nonzero only there.
        w.add_head_count_kv([2 if i == BLOCK - 1 else 0 for i in range(BLOCK)])
        w.add_feed_forward_length([FF if i == BLOCK - 1 else 0 for i in range(BLOCK)])
        w.add_key_length(D)
        w.add_value_length(D)
        w.add_layer_norm_rms_eps(RMS_EPS)
        w.add_float32("nemotron_h_moe.attention.layer_norm_epsilon", LAYER_NORM_EPS)
        w.add_ssm_conv_kernel(4)
        w.add_ssm_state_size(64)
        w.add_ssm_group_count(2)
        w.add_ssm_time_step_rank(24)
        w.add_ssm_inner_size(24 * 8)
        w.add_expert_count(EXPERTS)
        w.add_expert_used_count(2)
        w.add_expert_feed_forward_length(FF)
        w.add_expert_shared_feed_forward_length(SHARED)
        w.add_expert_shared_count(1)
        w.add_expert_group_count(1)
        w.add_expert_group_used_count(1)
        w.add_expert_weights_norm(True)
        w.add_float32("nemotron_h_moe.expert_weights_scale", 2.5)
        w.add_nextn_predict_layers(1)
        w.add_rope_freq_base(10000.0)
        w.add_rope_dimension_count(H // HEADS // 4)
        w.add_vocab_size(VOCAB)

        def q4(name, rows, k):
            """Random Q4_0 blocks with moderate binary16 scales, as a projection bank."""
            blocks = rows * k // 32
            raw = rng.integers(0, 256, (blocks, 18), dtype=np.uint8)
            raw[:, 0:2] = rng.uniform(2e-4, 2e-3, blocks).astype(np.float16).view(np.uint8).reshape(blocks, 2)
            sources[name] = raw.reshape(rows, k // 32, 18)
            w.add_tensor(name, raw.reshape(rows, k // 32 * 18), raw_dtype=Q.Q4_0)

        def f32(name, n):
            w.add_tensor(name, (rng.standard_normal(n).astype(np.float32) * 0.05))

        def bf16(name, rows, k):
            """A [rows, k] bfloat16 matrix: float32 rounded to nearest even on its upper 16 bits."""
            bits = np.ascontiguousarray(rng.standard_normal(rows * k).astype(np.float32)).view(np.uint32)
            words = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            sources[name] = words.reshape(rows, k)
            w.add_tensor(name, words.view(np.uint8).reshape(rows, 2 * k), raw_dtype=Q.BF16)

        s = f"blk.{BLOCK - 1}."
        # The re-shipped globals of the target.
        q4("token_embd.weight", VOCAB, H)
        q4("output.weight", VOCAB, H)
        f32("output_norm.weight", H)
        f32(s + "nextn.enorm.weight", H)
        f32(s + "nextn.hnorm.weight", H)
        q4(s + "nextn.eh_proj.weight", H, 2 * H)
        f32(s + "attn_norm.weight", H)
        q4(s + "attn_q.weight", HEADS * D, H)
        q4(s + "attn_k.weight", KV * D, H)
        q4(s + "attn_v.weight", KV * D, H)
        q4(s + "attn_output.weight", H, HEADS * D)
        f32(s + "post_attention_norm.weight", H)
        bf16(s + "ffn_gate_inp.weight", EXPERTS, H)  # the router: [E, H] rows of BF16
        f32(s + "exp_probs_b.bias", EXPERTS)
        q4(s + "ffn_up_exps.weight", EXPERTS * FF, H)
        q4(s + "ffn_down_exps.weight", EXPERTS * H, FF)
        q4(s + "ffn_up_shexp.weight", SHARED, H)
        q4(s + "ffn_down_shexp.weight", H, SHARED)
        f32(s + "nextn.shared_head_norm.weight", H)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        # Without the target's tokenizer files beside it the sidecar is refused, naming --resource.
        bare = work / "bare"
        bare.mkdir()
        shutil.copy(path, bare / "mtp.gguf")
        result = subprocess.run(
            [str(native), f"--model={bare / 'mtp.gguf'}", f"--out={work / 'bare.ninferno'}", "--quiet"],
            cwd=repository, capture_output=True, text=True)
        if result.returncode == 0 or "--resource" not in result.stderr + result.stdout:
            errors.append(f"a sidecar without tokenizer files: rc {result.returncode}, expected a "
                          f"refusal naming --resource: {(result.stderr + result.stdout)[-600:]}")

        # The sidecar ships beside no tokenizer files; the converter takes them from the
        # directory, as the drafter's own copy of the target's.
        vocab = {f"tok{i}": i for i in range(VOCAB)}
        added = [{"id": i, "content": f"tok{i}", "special": True} for i in (0, 1, 2, 3)]
        (gdir / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": vocab}, "added_tokens": added}), encoding="utf-8")
        (gdir / "tokenizer_config.json").write_text(
            json.dumps({"added_tokens_decoder": {"4": {"content": "tok4", "special": True}}}), encoding="utf-8")
        (gdir / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")
        (gdir / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}\n", encoding="utf-8")

        out = work / "drafter.ninferno"
        result = subprocess.run(
            [str(native), f"--model={path}", f"--out={out}", "--quiet"],
            cwd=repository, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:], file=sys.stderr)
            return 1
        value, objects, data, payload = directory(out)

        config = value["components"]["text"]["config"]
        want = {
            "model_type": "nemotron_h_mtp",
            "architectures": ["NemotronHMtpForCausalLM"],
            "hidden_size": H, "vocab_size": VOCAB, "num_hidden_layers": 1,
            "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": D,
            # The converter keeps each epsilon as its binary32 value.
            "rms_norm_eps": float(np.float32(RMS_EPS)),
            "layer_norm_eps": float(np.float32(LAYER_NORM_EPS)),
        }
        for key, expected in want.items():
            if config.get(key) != expected:
                errors.append(f"config {key} = {config.get(key)}, expected {expected}")
        moe = {"num_experts": EXPERTS, "num_experts_per_tok": 2,
               "moe_intermediate_size": FF, "shared_intermediate_size": SHARED,
               "scoring": "sigmoid", "renormalize": True, "routed_scaling_factor": 2.5,
               "activation": "relu2"}
        for key, expected in moe.items():
            if config.get("moe", {}).get(key) != expected:
                errors.append(f"config moe.{key} = {config.get('moe', {}).get(key)}, "
                              f"expected {expected}")

        def parts(name):
            """(object, [first, last) element range) of each part of a binding."""
            b = value["bindings"].get(name)
            if b is None:
                errors.append(f"binding {name} is absent")
                return []
            if "object" in b:
                o = objects[b["object"]]
                return [(o, (0, int(np.prod(o["shape"]))))]
            return [(objects[p["object"]], tuple(p["range"])) for p in b["parts"]]

        def binding(name):
            got = parts(name)
            if not got:
                return None, None
            return {o["format"] for o, _ in got}, (got[0][0]["shape"] if len(got) == 1 else None)

        # Every binding, its format and the shape of the object it lies in. The attention's q, k
        # and v are one stacked q/k/v parent, so each lies in the parent's whole shape; the row
        # ranges are checked below.
        qkv = [(HEADS + 2 * KV) * D, H]
        expect = {
            "mtp/embedding_norm": ("bf16", [H]), "mtp/hidden_norm": ("bf16", [H]),
            "mtp/attention_norm": ("bf16", [H]), "mtp/post_attention_norm": ("bf16", [H]),
            "mtp/final_norm": ("bf16", [H]),
            "mtp/eh_proj": ("ggml_q4_0", [H, 2 * H]),
            "mtp/attention/query": ("ggml_q4_0", qkv),
            "mtp/attention/key": ("ggml_q4_0", qkv),
            "mtp/attention/value": ("ggml_q4_0", qkv),
            "mtp/attention/output": ("ggml_q4_0", [H, HEADS * D]),
            "mtp/mlp/router": ("bf16", [EXPERTS, H]),
            "mtp/mlp/selection_bias": ("fp32", [EXPERTS]),
            "mtp/mlp/shared_up": ("ggml_q4_0", [SHARED, H]),
            "mtp/mlp/shared_down": ("ggml_q4_0", [H, SHARED]),
        }
        for name, (fmt, shape) in expect.items():
            got_fmt, got_shape = binding(name)
            if got_fmt != {fmt}:
                errors.append(f"{name} is {got_fmt}, expected {fmt}")
            if got_shape != shape:
                errors.append(f"{name} lies in {got_shape}, expected {shape}")

        def expect_range(name, first_row, rows, k):
            """The binding is one part covering rows [first_row, first_row + rows) of K = k."""
            got = parts(name)
            want_range = (first_row * k, (first_row + rows) * k)
            if len(got) != 1 or got[0][1] != want_range:
                errors.append(f"{name} covers {[r for _, r in got]}, expected [{list(want_range)}] "
                              f"(rows [{first_row}, {first_row + rows}) of K {k})")

        # q, k and v: consecutive row ranges of their parent, in that order.
        expect_range("mtp/attention/query", 0, HEADS * D, H)
        expect_range("mtp/attention/key", HEADS * D, KV * D, H)
        expect_range("mtp/attention/value", (HEADS + KV) * D, KV * D, H)
        expect_range("mtp/mlp/router", 0, EXPERTS, H)
        # The expert banks: every expert one row range of one stacked parent, in expert order.
        for bank, rows, k in (("up", FF, H), ("down", H, FF)):
            ids = set()
            for e in range(EXPERTS):
                name = f"mtp/mlp/experts/{e}/{bank}"
                expect_range(name, e * rows, rows, k)
                ids.update(o["id"] for o, _ in parts(name))
            if len(ids) != 1:
                errors.append(f"the {bank} bank is not one parent: {ids}")
            else:
                parent = objects[ids.pop()]
                if parent["format"] != "ggml_q4_0" or parent["shape"] != [EXPERTS * rows, k]:
                    errors.append(f"the {bank} bank parent is {parent['format']} "
                                  f"{parent['shape']}, expected ggml_q4_0 [{EXPERTS * rows}, {k}]")

        def payload_rows(o, first, last, row_bytes, plane_offset=0):
            start = payload + o["offset"] + plane_offset
            return data[start + first * row_bytes:start + last * row_bytes]

        def expect_q4_bytes(name, source, first_source_row=0):
            """Q4_0 kept word for word: ggml_planes_v1 splits each 18-byte block, bytes unchanged,
            into its 16 code bytes ([N, K/32*16]) and, at the next 256-byte boundary, its binary16
            scale ([N, K/32*2])."""
            for o, (first, last) in parts(name):
                if o["format"] != "ggml_q4_0" or o.get("layout") != "ggml_planes_v1":
                    errors.append(f"{name}: {o['format']} / {o.get('layout')}, expected "
                                  f"ggml_q4_0 / ggml_planes_v1")
                    continue
                n, k = o["shape"]
                r0, r1 = first // k, last // k
                src = sources[source][first_source_row:first_source_row + (r1 - r0)]
                codes = -(-(n * (k // 32) * 16) // 256) * 256
                got_codes = payload_rows(o, r0, r1, k // 32 * 16)
                got_scales = payload_rows(o, r0, r1, k // 32 * 2, codes)
                if got_codes != src[:, :, 2:].tobytes() or got_scales != src[:, :, 0:2].tobytes():
                    errors.append(f"{name}: payload rows [{r0}, {r1}) are not the bytes of "
                                  f"{source} rows [{first_source_row}, "
                                  f"{first_source_row + r1 - r0})")

        expect_q4_bytes("mtp/eh_proj", s + "nextn.eh_proj.weight")
        expect_q4_bytes("mtp/attention/query", s + "attn_q.weight")
        expect_q4_bytes("mtp/attention/key", s + "attn_k.weight")
        expect_q4_bytes("mtp/attention/value", s + "attn_v.weight")
        expect_q4_bytes("mtp/attention/output", s + "attn_output.weight")
        expect_q4_bytes("mtp/mlp/shared_up", s + "ffn_up_shexp.weight")
        expect_q4_bytes("mtp/mlp/shared_down", s + "ffn_down_shexp.weight")
        for e in range(EXPERTS):
            expect_q4_bytes(f"mtp/mlp/experts/{e}/up", s + "ffn_up_exps.weight", e * FF)
            expect_q4_bytes(f"mtp/mlp/experts/{e}/down", s + "ffn_down_exps.weight", e * H)
        # The router's BF16 words, kept as stored.
        for o, (first, last) in parts("mtp/mlp/router"):
            start = payload + o["offset"] + 2 * first
            if data[start:start + 2 * (last - first)] != sources[s + "ffn_gate_inp.weight"].tobytes():
                errors.append("mtp/mlp/router: payload is not the source GGUF's BF16 words")

        # The input each projection reads: the attention sub-layer's and the MoE sub-layer's
        # normed inputs are different activations.
        uses = {u["parameter"]: u["input"] for u in value["uses"]}
        inputs = {"mtp/eh_proj": "mtp/fusion_input",
                  "mtp/attention/query": "mtp/attention_input",
                  "mtp/attention/key": "mtp/attention_input",
                  "mtp/attention/value": "mtp/attention_input",
                  "mtp/attention/output": "mtp/attention/context",
                  "mtp/mlp/router": "mtp/ffn_input",
                  "mtp/mlp/shared_up": "mtp/ffn_input",
                  "mtp/mlp/shared_down": "mtp/mlp/shared_product"}
        for e in range(EXPERTS):
            inputs[f"mtp/mlp/experts/{e}/up"] = "mtp/ffn_input"
            inputs[f"mtp/mlp/experts/{e}/down"] = "mtp/mlp/expert_product"
        for name, expected in inputs.items():
            if uses.get(name) != expected:
                errors.append(f"{name} reads {uses.get(name)}, expected {expected}")

        # The re-shipped globals of the target map to nothing.
        for name in value["bindings"]:
            if not name.startswith("mtp/"):
                errors.append(f"unexpected binding {name} of the re-shipped globals")
        # The selection bias is the whole exp_probs_b.bias object.
        if value["bindings"]["mtp/mlp/selection_bias"].get("object") is None:
            errors.append("the selection bias is not one whole object")
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        notes = report.get("native_notes") or []
        if not all(n["lossless"] for n in notes if n["encoding"].startswith("gguf")):
            errors.append(f"expected every kept gguf note lossless, got {notes}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"nemotron-mtp: {'ok' if not errors else 'FAILED'}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
