"""llama.cpp's dflash GGUF with DFlash2's convolutions and candidate selector, converted natively
as a Qwen3.5 target's dflash2 component, against its Hugging Face twin.

    python synthetic_dflash2_gguf.py NINFERNO_CONVERT [REPOSITORY]

The drafter is written the way llama.cpp's DFlashModel writes z-lab's DFlash2DraftModel
(gguf-py's writer): target_layers one past the transformers target_layer_ids, the mask token in
the tokenizer metadata, block/convolution/selector sizes, the sliding-window flags, degenerate
M-RoPE sections, norms and convolution base kernels in F32, and the matrices (codebooks among
them) in BF16 or Q4_0 blocks. The twin is the DFlash2DraftModel checkpoint those tensors come
from. Cases:
  - dflash2-bf16: target GGUF + BF16 drafter GGUF against target twin + drafter twin, every
    artifact object, component, binding and use equal;
  - dflash2-q4_0: every Q4_0 matrix (the codebooks too) kept as ggml_q4_0 whose rows decode to
    the GGUF's values, the K/V context projections bound to the key/value objects.

Exits 77 (skipped) without numpy, torch or gguf.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import numpy as np
    from gguf import GGMLQuantizationType as Q
except ImportError:
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_server  # noqa: E402
import synthetic_gguf as sg  # noqa: E402
from synthetic_gguf import G, Hf, VOCAB, compare, kept_kquant_check, qwen_case, tokenizer_metadata  # noqa: E402

LAYERS, HEADS, KV, D, INTER = 2, 4, 2, 64, 384
KERNEL, GROUP, RANK, TOP_K, BLOCK, WINDOW = 2, 16, 64, 16, 8, 1024
TAPS = [0, 2]  # target layers recorded (the synthetic target has sg.LAYERS = 4)


def drafter_case(work: Path, qtype, seed: int):
    """The dflash GGUF (G) and its DFlash2DraftModel twin (T)."""
    rng = np.random.default_rng(seed)
    h = sg.H
    gdir = work / "drafter_gguf"
    gdir.mkdir(parents=True)
    g = G(gdir / "model.gguf", "dflash", rng)
    t = Hf()
    w = g.writer
    w.add_block_count(LAYERS)
    w.add_context_length(4096)
    w.add_embedding_length(h)
    w.add_feed_forward_length(INTER)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV)
    w.add_key_length(D)
    w.add_value_length(D)
    w.add_rope_freq_base(10000000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_causal_attention(False)
    w.add_uint32("dflash.block_size", BLOCK)
    w.add_uint32("dflash.conv_kernel_size", KERNEL)
    w.add_uint32("dflash.conv_group_size", GROUP)
    w.add_uint32("dflash.selector_rank", RANK)
    w.add_uint32("dflash.selector_top_k", TOP_K)
    w.add_array("dflash.target_layers", [tap + 1 for tap in TAPS])
    w.add_sliding_window(WINDOW)
    w.add_array("dflash.attention.sliding_window_pattern", [True] * LAYERS)
    w.add_rope_dimension_sections([D // 2, 0, 0, 0])
    tokenizer_metadata(g, "qwen35")
    w.add_uint32("tokenizer.ggml.mask_token_id", VOCAB - 1)

    def randn(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def matrix(gname, hname, rows, k):
        if qtype is None:
            g.bf16(gname, randn(rows, k))
        else:
            g.quant(gname, (rows, k), qtype)
        sg.put_twin(t, hname, g, gname)

    def vector(gname, hname, shape, scale=0.2):
        # llama.cpp writes the drafter's BF16 norms and base kernels as F32: exact BF16 values.
        values = sg.bf16_round(1.0 + randn(*shape, scale=scale) if len(shape) == 1 else randn(*shape))
        g.f32(gname, values)
        t.bf16(hname, values)

    matrix("fc.weight", "fc.weight", h, len(TAPS) * h)
    vector("enc.output_norm.weight", "hidden_norm.weight", (h,))
    vector("output_norm.weight", "norm.weight", (h,))
    matrix("selector_hidden.weight", "candidate_selector.hidden_projection.weight", RANK, h)
    matrix("selector_predecessor.weight", "candidate_selector.predecessor_codebook", VOCAB, RANK)
    matrix("selector_successor.weight", "candidate_selector.successor_codebook", VOCAB, RANK)
    conv_rows = 2 * KERNEL * (h // GROUP)
    for layer in range(LAYERS):
        s, p = f"blk.{layer}.", f"layers.{layer}."
        vector(s + "attn_norm.weight", p + "input_layernorm.weight", (h,))
        vector(s + "ffn_norm.weight", p + "post_attention_layernorm.weight", (h,))
        vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", (D,))
        vector(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", (D,))
        matrix(s + "attn_q.weight", p + "self_attn.q_proj.weight", HEADS * D, h)
        matrix(s + "attn_k.weight", p + "self_attn.k_proj.weight", KV * D, h)
        matrix(s + "attn_v.weight", p + "self_attn.v_proj.weight", KV * D, h)
        matrix(s + "attn_output.weight", p + "self_attn.o_proj.weight", h, HEADS * D)
        matrix(s + "ffn_gate.weight", p + "mlp.gate_proj.weight", INTER, h)
        matrix(s + "ffn_up.weight", p + "mlp.up_proj.weight", INTER, h)
        matrix(s + "ffn_down.weight", p + "mlp.down_proj.weight", h, INTER)
        for gname, hname in (("attn_conv", "attention_conv"), ("ffn_conv", "mlp_conv")):
            # transformers' base_kernel [2, kernel, hidden]; ggml [hidden, kernel, 2].
            vector(s + gname + "_base", p + hname + ".base_kernel", (2, KERNEL, h))
            matrix(s + gname + "_proj.weight", p + hname + ".kernel_projection.weight", conv_rows, h)
    g.save(gdir, files=False)
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "attention_bias": False,
        "hidden_act": "silu", "hidden_size": h, "intermediate_size": INTER, "head_dim": D,
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "num_hidden_layers": LAYERS,
        "num_target_layers": sg.LAYERS, "vocab_size": VOCAB, "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-06, "is_causal": False, "use_sliding_window": True,
        "sliding_window": WINDOW, "max_window_layers": LAYERS,
        "layer_types": ["sliding_attention"] * LAYERS,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dflash_config": {"block_size": BLOCK, "conv_group_size": GROUP, "conv_kernel_size": KERNEL,
                          "mask_token_id": VOCAB - 1, "selector_rank": RANK, "selector_top_k": TOP_K,
                          "target_layer_ids": TAPS},
    }
    t.save(work / "drafter_twin", config)
    return gdir / "model.gguf", work / "drafter_twin"


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []

    def run(model, drafter, out, *extra):
        return subprocess.run([str(native), f"--model={model}", "--recipe=qwen3_5_bf16",
                               "--components=text,dflash2", f"--source=dflash2={drafter}",
                               f"--out={out}", *extra],
                              cwd=repository, capture_output=True, text=True)

    with tempfile.TemporaryDirectory(prefix="ninferno-convert-dflash2-") as tmp:
        root = Path(tmp)
        # BF16: the GGUF pair converts to the twin pair's artifact exactly.
        work = root / "dflash2-bf16"
        target, target_twin, _ = qwen_case(work / "target", False, None, 71)
        drafter, drafter_twin = drafter_case(work, None, 72)
        outs = {}
        case = []
        for source, model, draft in (("gguf", target, drafter), ("twin", target_twin, drafter_twin)):
            out = work / f"{source}.ninferno"
            result = run(model, draft, out, "--maintainer-quantize")
            if result.returncode:
                case.append(f"dflash2-bf16: converting the {source} failed\n{result.stderr[-3000:]}")
            outs[source] = out
        if not case:
            case += compare("dflash2-bf16", outs["gguf"], outs["twin"])
            value, *_ = sg.directory(outs["gguf"])
            draft = value["components"]["dflash2"]["config"]
            if draft["dflash_config"]["target_layer_ids"] != TAPS or \
                    draft["dflash_config"]["mask_token_id"] != VOCAB - 1:
                case.append(f"dflash2-bf16: drafter config {draft['dflash_config']}")
        print(f"dflash2-bf16: {'ok' if not case else 'FAILED'}", flush=True)
        errors += case

        # Q4_0 (the published file's form): kept, codebooks included, decoding to the GGUF's values.
        work = root / "dflash2-q4_0"
        target, _, _ = qwen_case(work / "target", False, None, 73)
        drafter, _ = drafter_case(work, Q.Q4_0, 74)
        # fc, the selector's projection and two codebooks; per layer q k v o gate up down and the
        # two kernel projections.
        matrices = 4 + LAYERS * 9
        case = kept_kquant_check("dflash2-q4_0", work, target, ["ggml_q4_0"],
                                 lambda model, out, *extra: run(model, drafter, out, *extra),
                                 matrices)
        value, objects, *_ = sg.directory(work / "kept.ninferno")
        for name in ("dflash2/candidate_selector/predecessor_codebook",
                     "dflash2/candidate_selector/successor_codebook",
                     "dflash2/layers/0/attention_conv/kernel_projection",
                     "dflash2/feature_projection"):
            fmt = {objects[oid]["format"] for oid, *_ in sg.binding_parts(value, name)}
            if fmt != {"ggml_q4_0"}:
                case.append(f"dflash2-q4_0: {name} stored as {fmt}")
        for layer in range(LAYERS):
            p = f"dflash2/layers/{layer}/attention/"
            for role in ("key", "value"):
                if sg.binding_parts(value, p + "context_" + role) != sg.binding_parts(value, p + role):
                    case.append(f"dflash2-q4_0: {p}context_{role} is not bound to {role}'s rows")
        print(f"dflash2-q4_0: {'ok' if not case else 'FAILED'}", flush=True)
        errors += case

        # A registry companion declared "quant": "q8" and published in Q4_0 (as the Qwen3.8 GGUF's
        # DFlash2 is), fetched with its GGUF target: the declared mode cannot apply (a published
        # encoding is kept) and that is expected, so no warning; --drafter-quant q8 given on the
        # command line says it changed nothing.
        work = root / "registry-q4_0"
        # The target carries its tokenizer as GGUF metadata, as the drafter does (one fingerprint).
        target, target_twin, _ = qwen_case(work / "target", False, None, 75, rebuilt=True)
        twin_config = json.loads((target_twin / "config.json").read_text(encoding="utf-8"))
        twin_config = twin_config.get("text_config", twin_config)
        # A GGUF drafter records no target depth: the entry's geometry is what it is checked by.
        geometry = {key: twin_config[key] for key in ("hidden_size", "num_hidden_layers", "vocab_size")}
        drafter, _ = drafter_case(work, Q.Q4_0, 76)
        commit = "c0ffee0123456789abcdef0123456789abcdef01"
        hub = fetch_server.Hub()
        hub.add_repo("org/qwen-tiny-GGUF", {"qwen-tiny-BF16.gguf": target.read_bytes()})
        hub.add_repo("lab/qwen-tiny-DFlash2-GGUF", {"dflash2-qwen-tiny-Q4_0.gguf": drafter.read_bytes()})
        registry = {
            "schema": 1, "published": "test",
            "kinds": {"dflash2": {"engines": ["qwen3_5"], "architectures": ["DFlash2DraftModel"]},
                      "mtp": {"engines": ["qwen3_5"]}},
            "models": [{"id": "qwen-tiny", "engine": "qwen3_5",
                        "match": {"repos": ["org/qwen-tiny-GGUF"], "geometry": geometry},
                        "drafters": [{"name": "dflash2", "kind": "dflash2",
                                      "repo": "lab/qwen-tiny-DFlash2-GGUF", "revision": commit,
                                      "gguf": "dflash2-qwen-tiny-Q4_0", "component": "dflash2",
                                      "quant": "q8"}],
                        "profiles": [{"gpu_class": "*",
                                      "default": {"config": "dflash2:7", "drafter": "dflash2"}}]}],
        }
        registry_path = work / "registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        server, base = fetch_server.start(hub)
        environment = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        environment.update({"HF_ENDPOINT": base, "NINFERNO_DRAFTER_REGISTRY": str(registry_path)})
        case = []
        try:
            for tag, extra, warned in (("declared", [], False), ("flag", ["--drafter-quant", "q8"], True)):
                result = subprocess.run([str(native), "--fetch", "org/qwen-tiny-GGUF", "--quant", "BF16",
                                         "--cache", str(work / f"cache-{tag}"), "--with-drafters", *extra],
                                        cwd=repository, capture_output=True, text=True, env=environment)
                log = result.stdout + result.stderr
                if result.returncode:
                    case.append(f"registry-q4_0 {tag}: failed\n{log[-3000:]}")
                    continue
                if "with its companion drafters" not in log:
                    case.append(f"registry-q4_0 {tag}: the companion was not converted in\n{log[-3000:]}")
                if ("changed nothing" in log) != warned:
                    lines = [line for line in log.splitlines() if "drafter" in line]
                    case.append(f"registry-q4_0 {tag}: warning {'missing' if warned else 'shown'}: {lines}")
                artifacts = list((work / f"cache-{tag}").rglob("*.ninferno"))
                reports = [json.loads(Path(str(a) + ".conversion.json").read_text(encoding="utf-8"))
                           for a in artifacts if Path(str(a) + ".conversion.json").exists()]
                formats = {m["format"] for r in reports for m in r["methods"]
                           for p in m["parameters"]
                           if p in {f"dflash2/layers/0/attention/{r}" for r in ("query", "key", "value")}}
                if formats != {"ggml_q4_0"} or any("drafter_quantization" in r["provenance"] for r in reports):
                    case.append(f"registry-q4_0 {tag}: drafter formats {formats}, provenance "
                                f"{[r['provenance'].get('drafter_quantization') for r in reports]}")
        finally:
            server.shutdown()
        print(f"registry-q4_0: {'ok' if not case else 'FAILED'}", flush=True)
        errors += case
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
