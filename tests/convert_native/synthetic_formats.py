"""Every checkpoint encoding the native converter reads, checked against exact expectations.

    python synthetic_formats.py NINFERNO_CONVERT [REPOSITORY]

For each encoding a small Gemma-4-shaped checkpoint Q holds its projections in that encoding,
and a twin checkpoint T holds what the encoding must become, in a form tools/convert reads:

  * a lossy site (dequantized, then quantized again): T holds the dequantized values as F32,
    computed here with numpy from the published packing rules, independently of the C++;
  * a kept FP8 site: T holds the same codes with the BF16 row scale the engine stores; the
    artifact's code and scale planes are also checked against Q's words directly (a BF16 scale is
    stored as published; an F32 scale is stored as BF16 only when the BF16 word is the same
    binary32 value, bit for bit);
  * an FP8 site kept with binary32 scales (per-tensor and per-channel F32 that BF16 cannot hold,
    128x128 blocks), which tools/convert cannot write: the artifact's code and scale planes are
    checked against Q's words directly;
  * an int4 site with 32-, 64- or 128-value groups or per-channel scales (AWQ, GPTQ without
    act-order, compressed-tensors), kept as u4_g{32,64,128}_zp_fp16 (a per-channel scale repeated
    per 128 values), which tools/convert cannot write either: the artifact's code, zero point and
    scale planes are checked against the unsigned words the encoder published;
  * an int8 site (GPTQ, compressed-tensors) with neutral zero points and binary16-exact scales,
    kept as q8_g32_fp16: the code plane must be code - zero and the scale plane the published
    scale repeated per 32 values;
  * a kept MXFP4 site: T holds the same codes as compressed-tensors NVFP4 with the E4M3 scales
    and power-of-two divisor that reproduce every value.

Without --maintainer-quantize the native converter must refuse every lossy site (naming the
encoding) and the unquantized checkpoints (the recipe's table would be NInferno's own quantization
of a checkpoint that publishes none), and convert the others. With it, the native converter
converts Q, tools/convert converts T, and every weight object must match byte for byte (for
MXFP4, whose NVFP4 twin must also carry activation divisors, the weights only). The report must flag each encoding as the test expects (lossless or lossy). Encodings
tools/convert reads itself (BF16, F16, per-channel BF16 FP8) are converted by both from Q and
must match entirely. A MoE checkpoint with FP8 experts checks the row-scaled FP8 expert stack,
which tools/convert cannot build, against expected bytes directly.

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
    import torch
except ImportError:
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_gemma4 as base  # noqa: E402
from artifact_directory import directory  # noqa: E402

HIDDEN, INTER, VOCAB = 256, 384, base.VOCAB
E4M3 = torch.arange(256, dtype=torch.int64).to(torch.uint8).view(torch.float8_e4m3fn).float().numpy()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def bf16_bytes(values):
    return torch.from_numpy(np.array(values, dtype=np.float32)).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()


def bf16_round(values):
    return torch.from_numpy(np.array(values, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def random_fp8(rng, n, k):
    values = torch.from_numpy(rng.standard_normal((n, k)).astype(np.float32) * 2)
    codes = values.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    return codes.astype(np.uint8)


class Matrix:
    """One projection: the tensors Q stores and the tensors T stores instead."""

    def __init__(self):
        self.q = {}  # suffix -> (dtype, shape, bytes)
        self.t = {}
        # A kept site: (engine format, arrays). u4_g<G>_zp_fp16: (unsigned codes [n, k], zero
        # points [n, k/G], F16 scales [n, k/G]); q8_g32_fp16: (int8 codes [n, k], F16 scales
        # [n, k/32]).
        self.u4 = None

    @staticmethod
    def tensor(dtype, array):
        array = np.ascontiguousarray(array)
        return (dtype, list(array.shape), array.tobytes())


def u4_target(k, group):
    """The group size an int4 site of K columns and `group` (K: per-channel) is kept at."""
    return next((g for g in (128, 64, 32) if k % g == 0 and group % g == 0), None)


def kept_u4(codes, zeros, scales, g_idx, act_order):
    """A u4 record: (codes [n, k], zeros, scales, column order or None). Act-order codes are kept
    with their columns sorted by group (stable), the order stored with the plane."""
    if not act_order:
        return (codes, zeros, scales, None)
    order = np.argsort(g_idx, kind="stable")
    return (codes[:, order], zeros, scales, order.astype(np.int32))


def expand(per_group, group, target, k):
    """[n, K/group] -> [n, K/target]: each target group takes its published group's value."""
    return per_group[:, (np.arange(k // target) * target) // group]


def dequantized_twin(values):
    return {".weight": Matrix.tensor("F32", f32(values))}


def pack_bits(values, bits, axis):
    """compressed-tensors pack_to_int32 of unsigned values along `axis` (0 or 1)."""
    values = np.asarray(values, dtype=np.uint32)
    if 32 % bits:  # one little-endian bit stream per row (3-bit: 32 codes in three words)
        assert axis == 1
        rows, cols = values.shape
        bits_le = ((values[:, :, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
        stream = np.zeros((rows, -(-cols * bits // 32) * 32), np.uint8)
        stream[:, :cols * bits] = bits_le.reshape(rows, -1)
        return np.packbits(stream, axis=1, bitorder="little").view("<i4")
    if axis == 0:
        values = values.T
    per = 32 // bits
    rows, cols = values.shape
    padded = np.zeros((rows, -(-cols // per) * per), np.uint32)
    padded[:, :cols] = values
    grouped = padded.reshape(rows, -1, per)
    packed = np.zeros(grouped.shape[:2], np.uint32)
    for i in range(per):
        packed |= grouped[:, :, i] << np.uint32(i * bits)
    if axis == 0:
        packed = packed.T
    return packed.view(np.int32)


# ------------------------------------------------------------------------------------------------
# Encoders: (n, k, rng) -> Matrix, with T per the module docstring.


def enc_fp8(scale_kind, scale_dtype):
    def encode(n, k, rng):
        m = Matrix()
        codes = random_fp8(rng, n, k)
        if scale_kind == "channel":
            raw = rng.uniform(0.001, 0.02, n).astype(np.float32)
            if scale_dtype == "BF16" or scale_dtype == "F32-exact":
                raw = bf16_round(raw)
            scale = raw
            shape = (n, 1)
        else:  # tensor, inexact in BF16
            scale = f32([0.0123456789])
            shape = (1,)
        m.q[".weight"] = ("F8_E4M3", [n, k], codes.tobytes())
        if scale_dtype == "BF16":
            m.q[".weight_scale"] = ("BF16", list(shape), bf16_bytes(scale))
        else:
            m.q[".weight_scale"] = Matrix.tensor("F32", scale.reshape(shape))
        rows = np.broadcast_to(scale.reshape(-1), (n,)) if scale.size == 1 else scale
        m.t[".weight"] = ("F8_E4M3", [n, k], codes.tobytes())
        m.t[".weight_scale"] = ("BF16", [n, 1], bf16_bytes(rows))
        return m

    return encode


def enc_fp8_block(n, k, rng):
    m = Matrix()
    codes = random_fp8(rng, n, k)
    blocks = rng.uniform(0.001, 0.02, (-(-n // 128), -(-k // 128))).astype(np.float32)
    m.q[".weight"] = ("F8_E4M3", [n, k], codes.tobytes())
    m.q[".weight_scale_inv"] = Matrix.tensor("F32", blocks)
    scale = np.repeat(np.repeat(blocks, 128, 0), 128, 1)[:n, :k]
    m.t = dequantized_twin(f32(E4M3[codes]) * scale)
    return m


def enc_ct_int(bits, group, zero_point, act_order, scales=(0.001, 0.05)):
    """group None: per-channel (one scale per row)."""
    def encode(n, k, rng):
        m = Matrix()
        g = group or k
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        q = rng.integers(lo, hi + 1, (n, k))
        groups = k // g
        scale = bf16_round(rng.uniform(*scales, (n, groups)))
        z = rng.integers(lo, hi + 1, (n, groups)) if zero_point else np.zeros((n, groups), int)
        g_idx = rng.permutation(np.repeat(np.arange(groups), g)) if act_order else np.arange(k) // g
        m.q[".weight_packed"] = Matrix.tensor("I32", pack_bits(q + (1 << (bits - 1)), bits, 1))
        m.q[".weight_scale"] = ("BF16", [n, groups], bf16_bytes(scale))
        m.q[".weight_shape"] = Matrix.tensor("I64", np.array([n, k], np.int64))
        if zero_point:
            m.q[".weight_zero_point"] = Matrix.tensor("I32", pack_bits(z + (1 << (bits - 1)), bits, 0))
        if act_order:
            m.q[".weight_g_idx"] = Matrix.tensor("I32", g_idx.astype(np.int32))
        rows = np.arange(n)[:, None]
        m.t = dequantized_twin(f32(q - z[rows, g_idx[None, :]]) * scale[rows, g_idx[None, :]])
        exact = np.all(scale.astype(np.float16).astype(np.float32) == scale)
        if bits == 4 and exact:
            t = u4_target(k, g)
            m.u4 = (f"u4_g{t}_zp_fp16",
                    kept_u4(q + 8, expand(z + 8, g, t, k), expand(scale, g, t, k).astype(np.float16),
                            g_idx, act_order))
        if bits == 8 and not act_order and not zero_point and exact:
            m.u4 = ("q8_g32_fp16", (q.astype(np.int8), expand(scale, g, 32, k).astype(np.float16)))
        if bits == 3 and not act_order and not zero_point and exact:
            t = next(t for t in (128, 64) if k % t == 0 and g % t == 0)
            m.u4 = (f"q3_g{t}_fp16", (q, expand(scale, g, t, k).astype(np.float16)))
        return m

    return encode


def enc_ct_int8(scale_dtype):
    """compressed-tensors int-quantized: I8 codes, one scale per row."""
    def encode(n, k, rng):
        m = Matrix()
        q = rng.integers(-127, 128, (n, k))
        scale = rng.uniform(0.0001, 0.01, (n, 1)).astype(np.float32)
        m.q[".weight"] = Matrix.tensor("I8", q.astype(np.int8))
        if scale_dtype == "BF16":
            scale = bf16_round(scale)
            m.q[".weight_scale"] = ("BF16", [n, 1], bf16_bytes(scale))
            m.u4 = ("q8_g32_fp16", (q.astype(np.int8), np.repeat(scale, k // 32, 1).astype(np.float16)))
        else:
            m.q[".weight_scale"] = Matrix.tensor("F32", scale)
        m.t = dequantized_twin(f32(q) * scale)
        return m

    return encode


AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def awq_pack(values):  # [rows, n] unsigned nibbles -> [rows, n/8] int32
    rows, n = values.shape
    packed = np.zeros((rows, n // 8), np.uint32)
    for i, column in enumerate(AWQ_ORDER):
        packed |= values[:, column::8].astype(np.uint32) << np.uint32(4 * i)
    return packed.view(np.int32)


def enc_awq(group):
    def encode(n, k, rng):
        m = Matrix()
        q = rng.integers(0, 16, (k, n))  # [in, out]
        groups = k // group
        z = rng.integers(0, 16, (groups, n))
        scale = rng.uniform(0.001, 0.05, (groups, n)).astype(np.float16)
        m.q[".qweight"] = Matrix.tensor("I32", awq_pack(q))
        m.q[".qzeros"] = Matrix.tensor("I32", awq_pack(z))
        m.q[".scales"] = Matrix.tensor("F16", scale)
        g = np.arange(k) // group
        values = f32(q - z[g, :]) * scale.astype(np.float32)[g, :]
        m.t = dequantized_twin(values.T)
        m.u4 = (f"u4_g{group}_zp_fp16", (q.T, z.T, scale.T, None))
        return m

    return encode


def enc_gptq(group, act_order=True, bits=4, sym=False, v2=False, zero_shift=0):
    """GPTQ words: codes packed along K, zero points along N (v1 stores zero - 1, v2 the zero),
    F16 scales. group None: per-channel (group_size -1). sym: every zero point 2^(bits-1), stored
    with `zero_shift` added (a checkpoint whose words contradict its config)."""
    def encode(n, k, rng):
        m = Matrix()
        g = group or k
        top = 1 << bits
        q = rng.integers(0, top, (k, n))
        groups = k // g
        if sym:
            z = np.full((groups, n), top // 2)
        elif v2:
            z = rng.integers(0, top, (groups, n))
        else:
            z = rng.integers(1, top + 1, (groups, n))  # v1: stored as z - 1 (so z may be 2^bits)
        scale = rng.uniform(0.001, 0.05, (groups, n)).astype(np.float16)
        g_idx = (rng.permutation(np.repeat(np.arange(groups), g)) if act_order
                 else np.arange(k) // g)
        m.q[".qweight"] = Matrix.tensor("I32", pack_bits(q, bits, 0))
        m.q[".qzeros"] = Matrix.tensor("I32", pack_bits(z - (0 if v2 else 1) + zero_shift, bits, 1))
        m.q[".scales"] = Matrix.tensor("F16", scale)
        m.q[".g_idx"] = Matrix.tensor("I32", g_idx.astype(np.int32))
        values = f32(q - z[g_idx, :]) * scale.astype(np.float32)[g_idx, :]
        m.t = dequantized_twin(values.T)
        if bits == 4:
            t = u4_target(k, g)
            m.u4 = (f"u4_g{t}_zp_fp16",
                    kept_u4(q.T, expand(z.T, g, t, k), expand(scale.T, g, t, k), g_idx, act_order))
        if not act_order and bits == 8 and sym:
            m.u4 = ("q8_g32_fp16", ((q.T - z.T[:, g_idx]).astype(np.int8), expand(scale.T, g, 32, k)))
        return m

    return encode


def enc_mxfp4(span):
    """span <= 17: exponents fit one NVFP4 divisor (kept); larger: dequantized."""

    def encode(n, k, rng):
        m = Matrix()
        codes = rng.integers(0, 256, (n, k // 2)).astype(np.uint8)
        emax = 120
        exps = rng.integers(emax - span, emax + 1, (n, k // 32)).astype(np.uint8)
        exps[0, 0] = emax
        m.q[".weight_packed"] = Matrix.tensor("U8", codes)
        m.q[".weight_scale"] = Matrix.tensor("U8", exps)
        nibbles = np.stack((codes & 15, codes >> 4), -1).reshape(n, k)
        values = np.ldexp(E2M1[nibbles], np.repeat(exps.astype(np.int32), 32, 1) - 127).astype(np.float32)
        if span > 17:
            m.t = dequantized_twin(values)
            return m
        t = 135 - emax
        powers = np.repeat(exps.astype(np.int32), 2, 1) - emax + 8
        e4m3 = np.where(powers >= -6, (powers + 7) << 3, 1 << np.maximum(powers + 9, 0)).astype(np.uint8)
        m.t[".weight_packed"] = Matrix.tensor("U8", codes)
        m.t[".weight_scale"] = ("F8_E4M3", [n, k // 16], e4m3.tobytes())
        m.t[".weight_global_scale"] = Matrix.tensor("F32", f32([2.0**t]))
        m.t[".input_global_scale"] = Matrix.tensor("F32", f32([1.0]))
        return m

    return encode


def enc_gptoss(span):
    """gpt-oss MXFP4: <name>_blocks [2, n/2, k/32, 16] (element 2i in the low nibble, the two
    leading dimensions flattened into rows) and <name>_scales [2, n/2, k/32]; T as enc_mxfp4."""

    def encode(n, k, rng):
        m = enc_mxfp4(span)(n, k, rng)
        codes = np.frombuffer(m.q.pop(".weight_packed")[2], np.uint8).reshape(n, k // 2)
        exps = np.frombuffer(m.q.pop(".weight_scale")[2], np.uint8).reshape(n, k // 32)
        m.q["_blocks"] = Matrix.tensor("U8", codes.reshape(2, n // 2, k // 32, 16))
        m.q["_scales"] = Matrix.tensor("U8", exps.reshape(2, n // 2, k // 32))
        return m

    return encode


def enc_plain(dtype):
    def encode(n, k, rng):
        m = Matrix()
        values = rng.standard_normal((n, k)).astype(np.float32) * 0.05
        if dtype == "BF16":
            m.q[".weight"] = ("BF16", [n, k], bf16_bytes(values))
        else:
            m.q[".weight"] = Matrix.tensor("F16", values.astype(np.float16))
        m.t = dict(m.q)
        return m

    return encode


# ------------------------------------------------------------------------------------------------


def write_checkpoint(root: Path, tensors: dict, config: dict) -> None:
    root.mkdir(parents=True)
    w = base.Writer(0)
    for name, (dtype, shape, data) in tensors.items():
        w.raw(name, dtype, shape, data)
    w.save(root)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    base.resources(root)


def dense_model(encode, quantization_config, seed, kept=None):
    """Q and T tensors of a two-layer dense Gemma 4 text model plus its config; `kept` (a dict)
    receives each projection's published int4 g128 words by checkpoint prefix."""
    rng = np.random.default_rng(seed)
    text = {
        "model_type": "gemma4_text", "hidden_size": HIDDEN, "vocab_size": VOCAB,
        "num_hidden_layers": 2, "max_position_embeddings": 4096, "tie_word_embeddings": True,
        "rms_norm_eps": 1e-06, "layer_types": ["sliding_attention", "full_attention"],
        "num_attention_heads": 4, "attention_k_eq_v": True, "sliding_window": 128,
        "head_dim": 64, "num_key_value_heads": 2, "global_head_dim": 128,
        "num_global_key_value_heads": 1, "rope_parameters": base.rope(),
        "final_logit_softcapping": 30.0, "intermediate_size": INTER,
        "hidden_activation": "gelu_pytorch_tanh", "enable_moe_block": False,
    }
    config = {"architectures": ["Gemma4ForCausalLM"], **text}
    if quantization_config:
        config["quantization_config"] = quantization_config
    q, t = {}, {}

    def plain(name, shape):
        data = bf16_bytes(rng.standard_normal(shape).astype(np.float32) * 0.05)
        q[name] = t[name] = ("BF16", list(shape), data)

    plain("model.embed_tokens.weight", (VOCAB, HIDDEN))
    plain("model.norm.weight", (HIDDEN,))
    for layer, (d, kv) in enumerate(((64, 2), (128, 1))):
        p = f"model.layers.{layer}."
        shapes = {"self_attn.q_proj": (4 * d, HIDDEN), "self_attn.k_proj": (kv * d, HIDDEN),
                  "self_attn.o_proj": (HIDDEN, 4 * d), "mlp.gate_proj": (INTER, HIDDEN),
                  "mlp.up_proj": (INTER, HIDDEN), "mlp.down_proj": (HIDDEN, INTER)}
        if layer == 0:
            shapes["self_attn.v_proj"] = (kv * d, HIDDEN)
        for role, (n, k) in shapes.items():
            m = encode(n, k, rng)
            if kept is not None and m.u4 is not None:
                kept[p + role] = m.u4
            q.update({p + role + s: v for s, v in m.q.items()})
            t.update({p + role + s: v for s, v in m.t.items()})
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            plain(p + norm + ".weight", (HIDDEN,))
        plain(p + "self_attn.q_norm.weight", (d,))
        plain(p + "self_attn.k_norm.weight", (d,))
        plain(p + "layer_scalar", (1,))
    return q, t, config


def run(command, repository):
    result = subprocess.run(command, cwd=repository, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed:\n{result.stderr[-3000:]}")
    return result.stdout


def compare(name, native_out, python_out, weights_only):
    errors = []
    nv, nobj, ndata, noff = directory(native_out)
    pv, pobj, pdata, poff = directory(python_out)
    for oid, o in pobj.items():
        if weights_only and not oid.startswith("weight/"):
            continue
        other = nobj.get(oid)
        if other is None or (o.get("shape"), o.get("format"), o["bytes"]) != (
            other.get("shape"), other.get("format"), other["bytes"]
        ):
            errors.append(f"{name}: {oid} differs in shape/format/size")
            continue
        a = ndata[noff + other["offset"]:noff + other["offset"] + other["bytes"]]
        b = pdata[poff + o["offset"]:poff + o["offset"] + o["bytes"]]
        if a != b:
            errors.append(f"{name}: {oid} ({o.get('format')}) bytes differ")
    if not weights_only and nv["bindings"] != pv["bindings"]:
        errors.append(f"{name}: bindings differ")
    if not weights_only and nv["uses"] != pv["uses"]:
        errors.append(f"{name}: uses differ")
    return errors


# name -> (encoder, quantization_config, expected note: None = Python-readable, else lossless?)
CASES = {
    "bf16": (enc_plain("BF16"), None, None),
    "f16": (enc_plain("F16"), None, None),
    "fp8-channel-bf16": (enc_fp8("channel", "BF16"), None, None),
    "fp8-channel-f32-exact": (enc_fp8("channel", "F32-exact"), None, True),
    # Per-channel F32 scales that BF16 cannot hold (a compressed-tensors FP8-dynamic re-save with
    # binary32 scales): kept as binary32, never rounded.
    "fp8-channel-f32": (enc_fp8("channel", "F32"), None, True),
    "fp8-tensor-bf16": (enc_fp8("tensor", "BF16"), None, True),
    "fp8-tensor-f32": (enc_fp8("tensor", "F32"), None, True),
    "fp8-block128": (enc_fp8_block, {"quant_method": "fp8", "weight_block_size": [128, 128]}, True),
    "ct-int4-g64-sym": (enc_ct_int(4, 64, False, False), None, True),
    "ct-int4-g32-sym": (enc_ct_int(4, 32, False, False), None, True),
    "ct-int4-g32-asym": (enc_ct_int(4, 32, True, False), None, True),
    "ct-int4-channel-sym": (enc_ct_int(4, None, False, False), None, True),
    "ct-int4-channel-asym": (enc_ct_int(4, None, True, False), None, True),
    # Act-order: kept with the columns sorted by group and that order stored with the plane.
    "ct-int4-g64-asym-actorder": (enc_ct_int(4, 64, True, True), None, True),
    "ct-int8-g128": (enc_ct_int(8, 128, False, False), None, True),
    # 3-bit symmetric (ISTA-DASLab GSQ): kept as q3_g128_fp16 / q3_g64_fp16 codes.
    "ct-int3-g128-sym": (enc_ct_int(3, 128, False, False), None, True),
    "ct-int3-g64-sym": (enc_ct_int(3, 64, False, False), None, True),
    "ct-int8-g32": (enc_ct_int(8, 32, False, False), None, True),
    "ct-int8-packed-channel": (enc_ct_int(8, None, False, False), None, True),
    # Signed int8 codes minus a zero point leave int8: not kept.
    "ct-int8-g128-asym": (enc_ct_int(8, 128, True, False), None, False),
    "ct-int8-channel": (enc_ct_int8("F32"), None, False),
    "ct-int8-channel-bf16": (enc_ct_int8("BF16"), None, True),
    "awq-int4-g64": (enc_awq(64), {"quant_method": "awq", "bits": 4, "group_size": 64}, True),
    "awq-int4-g32": (enc_awq(32), {"quant_method": "awq", "bits": 4, "group_size": 32}, True),
    "gptq-int4-g64-actorder": (enc_gptq(64), {"quant_method": "gptq", "bits": 4, "group_size": 64}, True),
    "gptq-int4-g32-actorder": (enc_gptq(32), {"quant_method": "gptq", "bits": 4, "group_size": 32}, True),
    # q8_g32_fp16 stores no column order: act-order int8 is dequantized, so refused.
    "gptq-int8-g128-actorder": (enc_gptq(128, bits=8, sym=True),
                                {"quant_method": "gptq", "bits": 8, "group_size": 128, "sym": True},
                                False),
    "gptq-int4-g32": (enc_gptq(32, act_order=False),
                      {"quant_method": "gptq", "bits": 4, "group_size": 32, "desc_act": False}, True),
    "gptq-int4-channel": (enc_gptq(None, act_order=False),
                          {"quant_method": "gptq", "bits": 4, "group_size": -1, "desc_act": False}, True),
    "gptq-v2-int4-g64": (enc_gptq(64, act_order=False, v2=True),
                         {"quant_method": "gptq", "bits": 4, "group_size": 64,
                          "checkpoint_format": "gptq_v2"}, True),
    "gptq-int8-g128": (enc_gptq(128, act_order=False, bits=8, sym=True),
                       {"quant_method": "gptq", "bits": 8, "group_size": 128, "sym": True}, True),
    "gptq-int8-channel": (enc_gptq(None, act_order=False, bits=8, sym=True),
                          {"quant_method": "gptq", "bits": 8, "group_size": -1, "sym": True}, True),
    # GPTQ 2-bit words have no engine format: dequantized, so refused.
    "gptq-int2-g64": (enc_gptq(64, act_order=False, bits=2),
                      {"quant_method": "gptq", "bits": 2, "group_size": 64}, False),
    # AutoRound's GPTQ packing (zero - 1) under its own quant_method.
    "autoround-gptq-int4-g128": (enc_gptq(128, act_order=False, sym=True),
                                 {"quant_method": "auto-round", "bits": 4, "group_size": 128,
                                  "sym": True, "packing_format": "auto_round:auto_gptq"}, True),
    "awq-int4-g128": (enc_awq(128), {"quant_method": "awq", "bits": 4, "group_size": 128}, True),
    "gptq-int4-g128": (enc_gptq(128, act_order=False),
                       {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False}, True),
    "gptq-int4-g128-actorder": (enc_gptq(128), {"quant_method": "gptq", "bits": 4, "group_size": 128}, True),
    "ct-int4-g128-sym": (enc_ct_int(4, 128, False, False), None, True),
    "ct-int4-g128-asym": (enc_ct_int(4, 128, True, False), None, True),
    # BF16 scales below binary16's normal range: not held exactly, so not kept.
    "ct-int4-g128-tiny-scales": (enc_ct_int(4, 128, False, False, scales=(1e-7, 1e-6)), None, False),
    "mxfp4-exact": (enc_mxfp4(12), None, True),
    "mxfp4-wide": (enc_mxfp4(30), None, False),
    "gptoss-mxfp4-exact": (enc_gptoss(12), None, True),
    "gptoss-mxfp4-wide": (enc_gptoss(30), None, False),
}


# Cases kept in an engine format with binary32 scales: the format every body projection must take.
F32_KEPT = {"fp8-tensor-f32": "fp8_e4m3fn_row_f32", "fp8-channel-f32": "fp8_e4m3fn_row_f32",
            "fp8-block128": "fp8_e4m3fn_b128_f32"}
# Cases kept as fp8_e4m3fn_row_bf16: their scale words are checked against the checkpoint's.
BF16_KEPT = ("fp8-channel-bf16", "fp8-channel-f32-exact", "fp8-tensor-bf16")
# Cases kept in a u4 zero-point format or as q8_g32_fp16 (the encoder records which).
U4_KEPT = tuple(name for name, (_, _, lossless) in CASES.items()
                if lossless and name.split("-")[0] in ("awq", "gptq", "ct", "autoround"))


def u4_kept_check(name, out, kept):
    """Every projection object of `out` is the kept format its encoder recorded, holding the
    published words of its sources in order: u4 unsigned codes (element 2i in the low nibble),
    zero points and binary16 scales per group; q8 int8 code - zero and binary16 scales per 32."""
    errors = []
    value, objects, data, offset = directory(out)
    methods = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))["methods"]
    body = [m for m in methods if all(p.startswith("text/layers/") for p in m["parameters"])
            and m["format"] not in ("bf16", "fp32")]
    formats = {f for f, _ in kept.values()}
    if len(formats) != 1:
        return [f"{name}: the encoder kept its sites in {sorted(formats)}"]
    fmt = formats.pop()
    if not body or any((m["format"], m["method"]) != (fmt, "import_encoded") for m in body):
        return [f"{name}: projections not kept as {fmt}: "
                f"{sorted({(m['format'], m['method']) for m in body})}"]
    for m in body:
        obj = objects[m["object"]]
        n, k = m["shape"]
        payload = data[offset + obj["offset"]:offset + obj["offset"] + obj["bytes"]]
        parts = []
        for source in m["sources"]:
            prefix = source.split(":", 1)[1]
            prefix = prefix[: -len(".weight")] if prefix.endswith(".weight") else prefix
            parts.append(kept[prefix][1])
        if fmt.startswith("q3_"):
            # row_split_k128_v1: per row K*3/8 bytes of 3-bit two's-complement codes (code i in
            # bits 3i..3i+2, little-endian), then binary16 scales per group.
            group = int(fmt.split("_")[1][1:])
            q = np.concatenate([c for c, _ in parts]).astype(np.int64) & 7
            bits = ((q[:, :, None] >> np.arange(3)) & 1).astype(np.uint8).reshape(n, -1)
            codes = np.packbits(bits, axis=1, bitorder="little").tobytes()
            scales = np.concatenate([s for _, s in parts]).astype(np.float16).tobytes()
            scale_at = -(-len(codes) // 256) * 256
            if payload[:len(codes)] != codes:
                errors.append(f"{name}: {m['object']} 3-bit code plane differs")
            if payload[scale_at:] != scales or len(scales) != n * (k // group) * 2:
                errors.append(f"{name}: {m['object']} scale plane differs")
            continue
        if fmt == "q8_g32_fp16":
            # row_split_k128_v1 (K here is whole 128-value spans): codes [n, k], scales [n, k/32].
            codes = np.concatenate([c for c, _ in parts]).astype(np.int8).tobytes()
            scales = np.concatenate([s for _, s in parts]).astype(np.float16).tobytes()
            scale_at = -(-n * k // 256) * 256
            if payload[:n * k] != codes:
                errors.append(f"{name}: {m['object']} code plane differs from code - zero")
            if payload[scale_at:scale_at + len(scales)] != scales or len(payload) != scale_at + len(scales):
                errors.append(f"{name}: {m['object']} scale plane differs")
            continue
        group = int(fmt.split("_")[1][1:])
        codes = np.concatenate([(u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8) for u, _, _, _ in parts]).tobytes()
        zeros = np.concatenate([z.astype(np.uint8) for _, z, _, _ in parts]).tobytes()
        scales = np.concatenate([s.astype(np.float16) for _, _, s, _ in parts]).tobytes()
        high = -(-n * k // 2 // 256) * 256
        scale_at = high + -(-n * (k // group) // 256) * 256
        scale_end = scale_at + n * (k // group) * 2
        orders = [o for _, _, _, o in parts]
        ordered = orders[0] is not None
        # One plane holds one column order: act-order members with different orders stay apart.
        if any((o is None) != (not ordered) or (ordered and not np.array_equal(o, orders[0]))
               for o in orders):
            errors.append(f"{name}: {m['object']} packs sources of different column orders")
            continue
        want_layout = "row_group_zp_ordered_v1" if ordered else "row_group_zp_v1"
        if obj["layout"] != want_layout:
            errors.append(f"{name}: {m['object']} layout {obj['layout']}, not {want_layout}")
        if payload[:n * k // 2] != codes:
            errors.append(f"{name}: {m['object']} code plane differs from the published words")
        if payload[high:high + n * (k // group)] != zeros:
            errors.append(f"{name}: {m['object']} zero point plane differs")
        if payload[scale_at:scale_end] != scales:
            errors.append(f"{name}: {m['object']} scale plane differs")
        if ordered:
            order_at = -(-scale_end // 256) * 256
            if payload[order_at:] != orders[0].astype("<i4").tobytes():
                errors.append(f"{name}: {m['object']} column order plane differs")
        elif len(payload) != scale_end:
            errors.append(f"{name}: {m['object']} has bytes after its scale plane")
    return errors


# Checkpoints refused before any dequantization, and what the refusal must say.
def refused_cases():
    transform = {"quant_method": "compressed-tensors", "format": "pack-quantized",
                 "transform_config": {"config_groups": {"u": {"type": "hadamard", "apply": []}}}}
    sparse = {"quant_method": "compressed-tensors", "format": "pack-quantized",
              "sparsity_config": {"format": "sparse-24-bitmask", "sparsity_structure": "2:4"}}
    return {
        "ct-transform-hadamard": (enc_ct_int(4, 128, False, False), transform, "transform_config"),
        "ct-sparsity-24": (enc_ct_int(4, 128, False, False), sparse, "sparsity_config"),
        # The config says sym, the words (read as GPTQ v1) do not: the zero convention is wrong.
        "gptq-sym-contradicted": (enc_gptq(128, act_order=False, sym=True, zero_shift=1),
                                  {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True},
                                  "declares symmetric quantization"),
    }


def refusal_case(work, native, repository, name, encode, quantization, expected, seed):
    q, _, config = dense_model(encode, quantization, seed=seed)
    write_checkpoint(work / name / "q", q, config)
    errors = []
    for forced in ([], ["--maintainer-quantize"]):
        result = subprocess.run([str(native), f"--model={work / name / 'q'}", "--recipe=gemma4_31b_nvfp4",
                                 "--quiet", *forced, f"--out={work / 'native' / (name + str(len(forced)) + '.ninferno')}"],
                                cwd=repository, capture_output=True, text=True)
        if result.returncode == 0 or expected not in result.stderr:
            errors.append(f"{name}: expected a refusal naming {expected!r}{' when forced' if forced else ''}, "
                          f"got rc {result.returncode}: {result.stderr[-600:]}")
    print(f"{name}: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def f32_kept_check(name, out, q):
    """Every projection object of `out` holds Q's FP8 code words in source order and Q's scales as
    binary32 words: one per row (a per-tensor scale repeated), or every 128x128 block's."""
    errors = []
    value, objects, data, offset = directory(out)
    methods = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))["methods"]
    body = [m for m in methods if all(p.startswith("text/layers/") for p in m["parameters"])
            and m["format"] not in ("bf16", "fp32")]
    if not body or any((m["format"], m["method"]) != (F32_KEPT[name], "import_encoded") for m in body):
        return [f"{name}: projections not kept as {F32_KEPT[name]}: "
                f"{sorted({(m['format'], m['method']) for m in body})}"]
    for m in body:
        obj = objects[m["object"]]
        n, k = m["shape"]
        payload = data[offset + obj["offset"]:offset + obj["offset"] + obj["bytes"]]
        codes, scales = b"", b""
        for source in m["sources"]:
            prefix = source.split(":", 1)[1][: -len(".weight")]
            dtype, shape, raw = q[prefix + ".weight"]
            codes += raw
            if F32_KEPT[name] == "fp8_e4m3fn_row_f32":
                _, sshape, sraw = q[prefix + ".weight_scale"]
                words = np.frombuffer(sraw, np.float32)
                scales += (np.broadcast_to(words, (shape[0],)) if words.size == 1 else words).astype(np.float32).tobytes()
            else:
                scales += q[prefix + ".weight_scale_inv"][2]
        scale_offset = -(-n * k // 256) * 256
        if payload[:n * k] != codes:
            errors.append(f"{name}: {m['object']} code plane differs from the checkpoint's words")
        if payload[scale_offset:scale_offset + len(scales)] != scales or len(payload) != scale_offset + len(scales):
            errors.append(f"{name}: {m['object']} scale plane differs from the checkpoint's scales")
    return errors


def bf16_kept_check(name, out, q):
    """Every projection object of `out` is fp8_e4m3fn_row_bf16 holding Q's FP8 code words in source
    order and, per row, the checkpoint's scale bit for bit: a BF16 scale word as published, an F32
    scale as the BF16 word whose binary32 value has the published bits (a per-tensor scale
    repeated on every row)."""
    errors = []
    value, objects, data, offset = directory(out)
    methods = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))["methods"]
    body = [m for m in methods if all(p.startswith("text/layers/") for p in m["parameters"])
            and m["format"] not in ("bf16", "fp32")]
    if not body or any((m["format"], m["method"]) != ("fp8_e4m3fn_row_bf16", "import_encoded") for m in body):
        return [f"{name}: projections not kept as fp8_e4m3fn_row_bf16: "
                f"{sorted({(m['format'], m['method']) for m in body})}"]
    for m in body:
        obj = objects[m["object"]]
        n, k = m["shape"]
        payload = data[offset + obj["offset"]:offset + obj["offset"] + obj["bytes"]]
        codes, words = b"", []
        for source in m["sources"]:
            prefix = source.split(":", 1)[1][: -len(".weight")]
            _, shape, raw = q[prefix + ".weight"]
            codes += raw
            sdtype, _, sraw = q[prefix + ".weight_scale"]
            if sdtype == "BF16":
                published = np.frombuffer(sraw, np.uint16)
            else:
                bits = np.frombuffer(sraw, np.uint32)
                if np.any(bits & 0xFFFF):
                    errors.append(f"{name}: an F32 scale BF16 cannot hold went to fp8_e4m3fn_row_bf16")
                published = (bits >> 16).astype(np.uint16)
            words.append(np.broadcast_to(published, (shape[0],)) if published.size == 1 else published)
        scales = np.concatenate(words).astype(np.uint16).tobytes()
        scale_offset = -(-n * k // 256) * 256
        if payload[:n * k] != codes:
            errors.append(f"{name}: {m['object']} code plane differs from the checkpoint's words")
        if payload[scale_offset:scale_offset + len(scales)] != scales or len(payload) != scale_offset + len(scales):
            errors.append(f"{name}: {m['object']} scale plane differs from the checkpoint's scales")
    return errors


def fp8_moe_case(work, native, repository):
    """FP8 experts: gate/up and down banks stay row-scaled FP8, one plane per bank."""
    errors = []
    root = work / "fp8-moe"
    base.text_checkpoint(root, moe=True, layout="ct", nested=True, tied=True, seed=21)
    # Replace every expert matrix by FP8 with a BF16 per-channel scale.
    files = sorted(root.glob("*.safetensors"))
    header_len = struct.unpack("<Q", files[0].read_bytes()[:8])[0]
    header = json.loads(files[0].read_bytes()[8:8 + header_len])
    data = files[0].read_bytes()[8 + header_len:]
    rng = np.random.default_rng(5)
    tensors, expected = {}, {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = (item["dtype"], item["shape"], data[item["data_offsets"][0]:item["data_offsets"][1]])
    for name in list(tensors):
        if ".experts." in name and name.endswith(".weight_packed"):
            prefix = name[: -len(".weight_packed")]
            n, k = tensors[name][1][0], tensors[name][1][1] * 2
            for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale"):
                tensors.pop(prefix + suffix)
            codes = random_fp8(rng, n, k)
            scale = bf16_round(rng.uniform(0.001, 0.02, n))
            tensors[prefix + ".weight"] = ("F8_E4M3", [n, k], codes.tobytes())
            tensors[prefix + ".weight_scale"] = ("BF16", [n, 1], bf16_bytes(scale))
            expected[prefix] = (codes, scale)
    for f in files:
        f.unlink()
    w = base.Writer(0)
    for name, (dtype, shape, raw) in tensors.items():
        w.raw(name, dtype, shape, raw)
    w.save(root)
    out = work / "native" / "fp8-moe.ninferno"
    log = run([str(native), f"--model={root}", "--recipe=gemma4_26b_a4b_nvfp4", f"--out={out}"], repository)
    report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
    value, objects, raw, offset = directory(out)
    for layer in range(2):
        for bank, halves in (("gate_up", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
            names = [f"text/layers/{layer}/moe/experts/{e}/{bank}" for e in range(4)]
            method = next(m for m in report["methods"] if m["parameters"] == names)
            if method["format"] != "fp8_e4m3fn_row_bf16" or method["method"] != "import_encoded":
                errors.append(f"fp8-moe: layer {layer} {bank} bank is {method['format']}")
                continue
            codes = np.concatenate([expected[f"model.language_model.layers.{layer}.experts.{e}.{h}"][0]
                                    for e in range(4) for h in halves])
            scales = np.concatenate([expected[f"model.language_model.layers.{layer}.experts.{e}.{h}"][1]
                                     for e in range(4) for h in halves])
            o = objects[method["object"]]
            n, k = o["shape"]
            body = raw[offset + o["offset"]:offset + o["offset"] + o["bytes"]]
            plane = -(-n * k // 256) * 256
            if body[: n * k] != codes.tobytes() or body[plane:plane + 2 * n] != bf16_bytes(scales):
                errors.append(f"fp8-moe: layer {layer} {bank} bank bytes differ")
    # Per-channel BF16 FP8 is an encoding tools/convert reads, so the report stays the Python
    # one; the log states the import.
    if "kept as fp8_e4m3fn_row_bf16, lossless" not in log:
        errors.append("fp8-moe: the log does not state the lossless FP8 import")
    print(f"fp8-moe: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def read_tensors(root: Path) -> dict:
    """name -> (dtype, shape, bytes) over every shard of a checkpoint."""
    tensors = {}
    for f in sorted(root.glob("*.safetensors")):
        raw = f.read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + header_len])
        data = raw[8 + header_len:]
        for name, item in header.items():
            if name != "__metadata__":
                begin, end = item["data_offsets"]
                tensors[name] = (item["dtype"], item["shape"], data[begin:end])
    return tensors


def qwen_int4_moe_case(work, native, repository):
    """A Qwen3.5 MoE checkpoint as GPTQ / AutoRound publish one: every routed expert (text layers
    and the MTP layer) its own int4 g128 matrix with zero points (experts.<e>.{gate,up,down}_proj
    .qweight/.qzeros/.scales), everything else BF16. qwen3_5_published must keep every expert as
    u4_g128_zp_fp16, each routed gate/up and down bank one object holding the published words of
    its experts in bank order, and quantize nothing."""
    import synthetic_qwen3_5 as qwen  # noqa: E402

    errors = []
    root = work / "qwen-int4-moe"
    qwen.base_checkpoint(root, moe_model=True, nested=True, tied=False, vision=False, seed=31)
    tensors = read_tensors(root)
    rng = np.random.default_rng(17)
    encode = enc_gptq(128, act_order=False)
    kept = {}
    prefixes = [f"model.language_model.layers.{i}.mlp." for i in range(qwen.LAYERS)] + ["mtp.layers.0.mlp."]
    for p in prefixes:
        tensors.pop(p + "experts.gate_up_proj")
        tensors.pop(p + "experts.down_proj")
        for e in range(qwen.EXPERTS):
            for role, (n, k) in (("gate_proj", (qwen.MOE_INTER, qwen.H)), ("up_proj", (qwen.MOE_INTER, qwen.H)),
                                 ("down_proj", (qwen.H, qwen.MOE_INTER))):
                m = encode(n, k, rng)
                name = f"{p}experts.{e}.{role}"
                kept[name] = m.u4
                tensors.update({name + suffix: value for suffix, value in m.q.items() if suffix != ".g_idx"})
    for f in root.glob("*.safetensors"):
        f.unlink()
    (root / "model.safetensors.index.json").unlink(missing_ok=True)
    w = base.Writer(0)
    for name, (dtype, shape, raw) in tensors.items():
        w.raw(name, dtype, shape, raw)
    w.save(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {"quant_method": "gptq", "bits": 4, "group_size": 128,
                                     "desc_act": False, "sym": False, "checkpoint_format": "gptq"}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    out = work / "native" / "qwen-int4-moe.ninferno"
    try:
        run([str(native), f"--model={root}", "--recipe=qwen3_5_published", "--components=text,mtp",
             "--no-drafters", "--quiet", f"--out={out}"], repository)
    except RuntimeError as error:
        print("qwen-int4-moe: FAILED", flush=True)
        return [f"qwen-int4-moe: {error}"]
    errors += u4_kept_check("qwen-int4-moe", out, kept)
    methods = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))["methods"]
    for layer in [f"text/layers/{i}" for i in range(qwen.LAYERS)] + ["mtp/layers/0"]:
        for bank, roles in (("gate_up", ("gate", "up")), ("down", ("down",))):
            names = [f"{layer}/moe/experts/{e}/{r}" for e in range(qwen.EXPERTS) for r in roles]
            if not any(m["parameters"] == names and m["format"] == "u4_g128_zp_fp16" for m in methods):
                errors.append(f"qwen-int4-moe: {layer} {bank} bank is not one u4 object in bank order")
    report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
    notes = report.get("native_notes") or []
    if not notes or any(n["lossless"] is not True for n in notes):
        errors.append(f"qwen-int4-moe: expected every note lossless, got {notes}")
    print(f"qwen-int4-moe: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-formats-") as tmp:
        work = Path(tmp)
        (work / "native").mkdir()
        (work / "python").mkdir()
        for index, (name, (encode, quantization, lossless)) in enumerate(CASES.items()):
            case_errors = []
            kept = {}
            q, t, config = dense_model(encode, quantization, seed=100 + index, kept=kept)
            write_checkpoint(work / name / "q", q, config)
            twin_config = {k: v for k, v in config.items() if k != "quantization_config"}
            write_checkpoint(work / name / "t", t, twin_config)
            native_out = work / "native" / f"{name}.ninferno"
            python_out = work / "python" / f"{name}.ninferno"
            common = ["--recipe=gemma4_31b_nvfp4"]
            # Lossy sites are dequantized, and an unquantized checkpoint under the Q8 table is
            # NInferno's own quantization: both refused unless forced.
            refused = lossless is False or name in ("bf16", "f16")
            plain = subprocess.run([str(native), f"--model={work / name / 'q'}", *common, "--quiet",
                                    f"--out={work / 'native' / (name + '-unforced.ninferno')}"],
                                   cwd=repository, capture_output=True, text=True)
            if refused != (plain.returncode != 0):
                case_errors.append(f"{name}: expected {'a refusal' if refused else 'a conversion'} without "
                                   f"--maintainer-quantize, got rc {plain.returncode}: {plain.stderr[-600:]}")
            elif refused and ("would not run" not in plain.stderr or "--maintainer-quantize" not in plain.stderr):
                case_errors.append(f"{name}: the refusal does not name the way out: {plain.stderr[-600:]}")
            elif name == "ct-int4-g128-tiny-scales" and "scales not exact in binary16" not in plain.stderr:
                case_errors.append(f"{name}: the refusal does not name why the int4 site is not kept: "
                                   f"{plain.stderr[-600:]}")
            if name in F32_KEPT or name in U4_KEPT:
                if plain.returncode == 0:
                    unforced = work / "native" / (name + "-unforced.ninferno")
                    case_errors += (f32_kept_check(name, unforced, q) if name in F32_KEPT
                                    else u4_kept_check(name, unforced, kept))
                    report = json.loads(Path(str(work / "native" / (name + "-unforced.ninferno")) +
                                             ".conversion.json").read_text(encoding="utf-8"))
                    notes = report.get("native_notes") or []
                    if not notes or any(n["lossless"] is not True for n in notes):
                        case_errors.append(f"{name}: expected every note lossless, got {notes}")
                summary = ", ".join(sorted({f"{n['encoding']}: {n['handling']}" for n in notes})) if plain.returncode == 0 else ""
                print(f"{name}: {'ok' if not case_errors else 'FAILED'}  {summary}", flush=True)
                errors += case_errors
                continue
            try:
                run([str(native), f"--model={work / name / 'q'}", *common, "--maintainer-quantize", "--quiet",
                     f"--out={native_out}"], repository)
                source = "q" if lossless is None else "t"
                run([sys.executable, "-m", "tools.convert", f"--model={work / name / source}", *common,
                     "--device=cpu", f"--out={python_out}"], repository)
            except RuntimeError as error:
                errors.append(f"{name}: {error}")
                print(f"{name}: FAILED", flush=True)
                continue
            case_errors += compare(name, native_out, python_out, weights_only=name.endswith("mxfp4-exact"))
            if name in BF16_KEPT:
                case_errors += bf16_kept_check(name, native_out, q)
            report = json.loads(Path(str(native_out) + ".conversion.json").read_text(encoding="utf-8"))
            notes = report.get("native_notes")
            if lossless is None:
                if notes is not None:
                    case_errors.append(f"{name}: a Python-readable checkpoint got native notes")
            elif not notes or any(n["lossless"] != lossless for n in notes):
                case_errors.append(f"{name}: expected every note lossless={lossless}, got {notes}")
            summary = ", ".join(sorted({f"{n['encoding']}: {n['handling']}" for n in notes or []}))
            print(f"{name}: {'ok' if not case_errors else 'FAILED'}  {summary}", flush=True)
            errors += case_errors
        for index, (name, (encode, quantization, expected)) in enumerate(refused_cases().items()):
            errors += refusal_case(work, native, repository, name, encode, quantization, expected,
                                   seed=900 + index)
        errors += fp8_moe_case(work, native, repository)
        errors += qwen_int4_moe_case(work, native, repository)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
