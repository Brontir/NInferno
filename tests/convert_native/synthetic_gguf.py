"""GGUF checkpoints converted natively, against their Hugging Face twins, byte for byte.

    python synthetic_gguf.py NINFERNO_CONVERT [REPOSITORY]

Each case starts from a small synthetic Hugging Face checkpoint O (Qwen3.5 dense and MoE with
MTP, Gemma 4 dense and MoE, Llama / Qwen2 / Qwen3 / Granite / OLMo 2-3 / SmolLM3 / Ministral 3 / Muse Glimmer dense
decoders, gpt-oss, llama.cpp's DFlash drafter) and writes the GGUF file G that llama.cpp's converter would write
from it, with llama.cpp's own gguf-py writer: GGUF names, 1 + w RMSNorm weights and -exp(A_log)
(Qwen3.5), value heads re-tiled (Qwen3.5 GDN), split expert banks, NVFP4 blocks with their
global and input scales (folded per-expert scales for Gemma 4 MoE), MXFP4 blocks, or random
ggml quant blocks (Q8_0, Q6_K, Q4_K, Q5_0, Q4_0, IQ4_XS) in place of the projections.

The twin T is built independently in numpy from G's contents: gguf-py dequantizes the blocks,
the llama.cpp transforms are inverted, NVFP4 and MXFP4 become compressed-tensors / Model Optimizer
tensors with the same codes. The native converter converts T (the safetensors path, proven
against tools/convert by the parity tests) and G (the GGUF path) with the same recipe: every
artifact object, the components (configs and resources), bindings and uses must match. The GGUF
report must flag quantized sources as lossy and kept NVFP4 as lossless. Where the inversion is
not exact (1 + w and -exp in binary32), T follows G, and the count of values that differ from O
is printed.

Exits 77 (skipped) without numpy, torch or gguf.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import torch
    import gguf
    from gguf import GGMLQuantizationType as Q
    from gguf.quants import dequantize
except ImportError:
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_gemma4 import VOCAB, resources, rope  # noqa: E402
from artifact_directory import directory  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Tensors


def bf16_round(values):
    return torch.from_numpy(np.ascontiguousarray(values, np.float32)).to(torch.bfloat16).float().numpy()


def bf16_words(values):
    return torch.from_numpy(np.ascontiguousarray(values, np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def words_to_f32(words):
    return (np.asarray(words, np.uint16).astype(np.uint32) << 16).view(np.float32)


class Hf:
    """A one-shard safetensors checkpoint."""

    def __init__(self):
        self.tensors = {}

    def put(self, name, dtype, array):
        array = np.ascontiguousarray(array)
        self.tensors[name] = (dtype, list(array.shape), array.tobytes())

    def bf16(self, name, values):
        self.put(name, "BF16", bf16_words(values))

    def f32(self, name, values):
        self.put(name, "F32", np.asarray(values, np.float32))

    def save(self, root: Path, config: dict) -> None:
        root.mkdir(parents=True)
        header, blobs, offset = {}, [], 0
        for name in sorted(self.tensors):
            dtype, shape, data = self.tensors[name]
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
            blobs.append(data)
            offset += len(data)
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
        (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        resources(root)


SCALE_FIELDS = {  # byte offsets of each block's binary16 scales
    Q.Q8_0: [0], Q.Q4_0: [0], Q.Q5_0: [0], Q.Q4_K: [0, 2], Q.Q5_K: [0, 2], Q.Q6_K: [208],
    Q.Q3_K: [108], Q.Q2_K: [80, 82],
    Q.IQ4_XS: [0], Q.Q4_1: [0, 2], Q.Q5_1: [0, 2], Q.IQ4_NL: [0], Q.IQ3_XXS: [0], Q.IQ3_S: [0],
    Q.IQ2_XXS: [0], Q.IQ2_XS: [0], Q.IQ2_S: [0], Q.IQ1_S: [0], Q.IQ1_M: [], Q.TQ1_0: [52],
    Q.TQ2_0: [64], Q.MXFP4: [],
}


class G:
    """A GGUF file, and the values its tensors decode to (gguf-py's dequantizers)."""

    def __init__(self, path: Path, arch: str, rng):
        self.writer = gguf.GGUFWriter(path, arch)
        self.arch = arch
        self.rng = rng
        self.values = {}  # GGUF name -> float32 values in numpy (row-major) shape
        # Q8_0 blocks as ggml's quantizer writes them (codes -127..127, one at +-127 per block),
        # instead of random bytes (which hold -128, a code no engine format keeps).
        self.q8_ggml = False
        self.words = {}   # GGUF name -> BF16 words when stored as BF16

    def f32(self, name, values):
        values = np.ascontiguousarray(values, np.float32)
        self.writer.add_tensor(name, values)
        self.values[name] = values

    def bf16(self, name, values):
        words = bf16_words(values)
        self.writer.add_tensor(name, words.view(np.uint8).reshape(*words.shape[:-1], -1), raw_dtype=Q.BF16)
        self.words[name] = words
        self.values[name] = words_to_f32(words).reshape(words.shape)

    def quant(self, name, shape, qtype):
        """Random blocks of `qtype` with moderate binary16 scales."""
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        rows = int(np.prod(shape[:-1]))
        blocks = rows * shape[-1] // block
        raw = self.rng.integers(0, 256, (blocks, size), dtype=np.uint8)
        for offset in SCALE_FIELDS[qtype]:
            scale = self.rng.uniform(2e-4, 2e-3, blocks).astype(np.float16)
            raw[:, offset:offset + 2] = scale.view(np.uint8).reshape(blocks, 2)
        if qtype == Q.IQ1_M:  # d's four nibbles top the four 16-bit scale words (bytes 48..55)
            bits = self.rng.uniform(2e-4, 2e-3, blocks).astype(np.float16).view(np.uint16)
            for i in range(4):
                nibble = ((bits >> (4 * i)) & 0xF).astype(np.uint8)
                raw[:, 49 + 2 * i] = (raw[:, 49 + 2 * i] & 0x0F) | (nibble << 4)
        if qtype == Q.MXFP4:  # E8M0 exponents wider than one NVFP4 divisor spans (2^-9..2^8),
            raw[:, 0] = self.rng.integers(105, 135, blocks, dtype=np.uint8)  # as real MXFP4_MOE
        if qtype == Q.Q8_0 and self.q8_ggml:
            codes = self.rng.integers(-127, 128, (blocks, 32)).astype(np.int8)
            codes[np.arange(blocks), self.rng.integers(0, 32, blocks)] = \
                np.where(self.rng.integers(0, 2, blocks) == 1, 127, -127).astype(np.int8)
            raw[:, 2:] = codes.view(np.uint8)
        data = raw.reshape(*shape[:-1], shape[-1] // block * size)
        self.writer.add_tensor(name, data, raw_dtype=qtype)
        self.values[name] = dequantize(data, qtype).astype(np.float32).reshape(shape)

    def nvfp4(self, name, codes, scales):
        """codes [rows, k/2] with element 2i in the low nibble, E4M3 scales [rows, k/16]."""
        rows, k = codes.shape[0], codes.shape[1] * 2
        nibbles = np.stack((codes & 15, codes >> 4), -1).reshape(rows, k // 16, 16)
        qs = (nibbles[:, :, :8] | (nibbles[:, :, 8:] << 4)).astype(np.uint8)  # [rows, k/16, 8]
        d = (scales & 0x7F).reshape(rows, k // 64, 4)
        raw = np.concatenate([d, qs.reshape(rows, k // 64, 32)], -1).reshape(rows, k // 64 * 36)
        self.writer.add_tensor(name, raw, raw_dtype=Q.NVFP4)
        self.values[name] = dequantize(raw, Q.NVFP4).astype(np.float32).reshape(rows, k)

    def mxfp4(self, name, codes, exps):
        """codes [rows, k/2] natural nibble order, E8M0 exponents [rows, k/32]."""
        rows, k = codes.shape[0], codes.shape[1] * 2
        nibbles = np.stack((codes & 15, codes >> 4), -1).reshape(rows, k // 32, 32)
        qs = (nibbles[:, :, :16] | (nibbles[:, :, 16:] << 4)).astype(np.uint8)
        raw = np.concatenate([exps.reshape(rows, k // 32, 1), qs], -1).reshape(rows, k // 32 * 17)
        self.writer.add_tensor(name, raw, raw_dtype=Q.MXFP4)
        self.values[name] = dequantize(raw, Q.MXFP4).astype(np.float32).reshape(rows, k)

    def save(self, root: Path, files: bool = True) -> Path:
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()
        if files:
            resources(root)
        return root


# ------------------------------------------------------------------------------------------------
# Tokenizer metadata, and the Hugging Face files the converter must rebuild from it.

TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['content'] }}<|im_end|>{% endfor %}\n"
# Gemma 4's turn markup: the converter takes the stops it adds from the template's conventions.
GEMMA4_TEMPLATE = ("{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n"
                   "{% endfor %}<|turn>model\n")
# gpt-oss's harmony messages: <|end|> closes one, the reply ends on <|return|> or <|call|>.
HARMONY_TEMPLATE = ("{% for m in messages %}<|start|>{{ m['role'] }}<|channel|>final<|message|>"
                    "{{ m['content'] }}<|end|>{% endfor %}<|start|>assistant")


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tokenizer_metadata(g: G, family: str, qwen35: bool = False) -> dict:
    """Write tokenizer.ggml.* for VOCAB tokens; return the files the converter must rebuild.
    `qwen35`: a Qwen3.5 model (its tokenizer_config fix-ups whatever the pre-tokenizer)."""
    if family == "spm":
        return spm_metadata(g)
    tokens = [f"t{i}" for i in range(VOCAB)]
    types = [3, 3, 3, 3, 4] + [1] * (VOCAB - 5)  # CONTROL, USER_DEFINED, NORMAL
    if family == "gemma4":
        tokens[5] = "a b"  # a token holding a space: its merge part is written as U+0120
        tokens[6], tokens[7] = "0", "x9"
        # Gemma 4's turn end and the tool's turn, CONTROL: stops beside the named eos and eot
        # (google/gemma-4-*-it's generation_config.json ends on <eos>, <turn|>, <|tool_response>).
        tokens[10], tokens[11] = "<turn|>", "<|tool_response>"
        types[10] = types[11] = 3
        # <eos>, CONTROL, while the file names another eos (unsloth's E4B names <turn|>): a stop.
        tokens[12], types[12] = "<eos>", 3
        merges_hf = [["t1", "0"], ["a b", "x9"], ["t20", "t30"]]
        merges = [" ".join(part.replace(" ", "Ġ") for part in pair) for pair in merges_hf]
    else:
        # (merges must not build an added token's text: such a piece is a plain vocabulary entry)
        tokens[6], tokens[7], tokens[8] = "x", "9", "8"
        merges_hf = merges = ["x 9", "9 8", "x 8"]
    plain = set()
    if family == "refact":
        # llama.cpp types IBM Granite 3.x's U+FFFD piece CONTROL; merges join its byte-level text,
        # so the rebuilt tokenizer has it as a plain vocabulary entry, not an added token.
        tokens[3] = "�"
        merges_hf = merges = merges + [FFFD_PIECE + " 9"]
        plain = {3}
    gpt2 = family in GPT2_PRES
    g.writer.add_tokenizer_model("gpt2" if gpt2 else "gemma4")
    if gpt2:
        g.writer.add_tokenizer_pre(family)
    adds_bos = gpt2 and GPT2_PRES[family][3]  # llama.cpp's default: no add_bos_token key
    g.writer.add_token_list(tokens)
    g.writer.add_token_types(types)
    g.writer.add_token_merges(merges)
    g.writer.add_bos_token_id(2)
    g.writer.add_eos_token_id(1)
    g.writer.add_eot_token_id(3)
    g.writer.add_pad_token_id(0)
    g.writer.add_unk_token_id(9)
    template = GEMMA4_TEMPLATE if family == "gemma4" else TEMPLATE
    g.writer.add_chat_template(template)
    added = [{"id": i, "content": tokens[i], "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": types[i] == 3} for i in range(VOCAB) if types[i] in (3, 4) and i not in plain]
    if gpt2:
        split, nfc, ignore_merges, _ = GPT2_PRES[family]
        byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
        post = byte_level
        if adds_bos:
            s = {"SpecialToken": {"id": tokens[2], "type_id": 0}}
            a = {"Sequence": {"id": "A", "type_id": 0}}
            b = {"Sequence": {"id": "B", "type_id": 1}}
            post = {"type": "Sequence", "processors": [byte_level, {
                "type": "TemplateProcessing", "single": [s, a], "pair": [s, a, s, b],
                "special_tokens": {tokens[2]: {"id": tokens[2], "ids": [2], "tokens": [tokens[2]]}}}]}
        head = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
                "normalizer": {"type": "NFC"} if nfc else None,
                "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": split}, "behavior": "Isolated", "invert": False},
                    byte_level] if split not in (None, "digits") else [
                    *([{"type": "Digits", "individual_digits": True}] if split == "digits" else []),
                    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}]},
                "post_processor": post, "decoder": byte_level}
        model = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                 "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False,
                 "ignore_merges": ignore_merges,
                 "vocab": {(FFFD_PIECE if i in plain else tokens[i]): i for i in range(VOCAB)},
                 "merges": merges_hf}
    else:
        head = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
                "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
                "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious",
                                  "invert": False},
                "post_processor": {"type": "TemplateProcessing",
                                   "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                                   "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                                            {"Sequence": {"id": "B", "type_id": 1}}],
                                   "special_tokens": {}},
                "decoder": {"type": "Sequence", "decoders": [
                    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                    {"type": "ByteFallback"}, {"type": "Fuse"}]}}
        model = {"type": "BPE", "dropout": None, "unk_token": tokens[9], "continuing_subword_prefix": None,
                 "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                 "vocab": {(FFFD_PIECE if i in plain else tokens[i]): i for i in range(VOCAB)},
                 "merges": merges_hf}
    decoder = {str(a["id"]): {"content": a["content"], "lstrip": False, "normalized": False, "rstrip": False,
                              "single_word": False, "special": a["special"]} for a in added}
    return {
        "tokenizer.json": compact({**head, "model": model}),
        "tokenizer_config.json": compact({"added_tokens_decoder": decoder, "bos_token": tokens[2],
                                          "eos_token": tokens[1], "pad_token": tokens[0], "unk_token": tokens[9],
                                          **({"add_bos_token": True} if adds_bos
                                             else {"add_bos_token": False} if family == "qwen35" or qwen35 else {}),
                                          **({"add_prefix_space": False} if gpt2 else {})}),
        "generation_config.json": compact({"bos_token_id": 2,
                                           "eos_token_id": [1, 3, 10, 11, 12] if family == "gemma4" else [1, 3],
                                           "pad_token_id": 0}),
        "chat_template.jinja": template,
    }


QWEN35_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+"
                r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
FFFD_PIECE = "ï¿½"  # GPT-2 byte-level text of U+FFFD (EF BF BD)
LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
TEKKEN_SPLIT = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?"
                r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
GPT4O_SPLIT = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
               r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
               r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# llama.cpp tokenizer.ggml.pre -> (the Hugging Face Split regex, or None for the GPT-2 regex of
# ByteLevel ("digits": after a Digits step); NFC normalizer; ignore_merges; BOS prepended when
# the GGUF does not say).
GPT2_PRES = {
    "qwen35": (QWEN35_SPLIT, True, False, False),
    "qwen2": (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+", True, False, False),
    "llama-bpe": (LLAMA3_SPLIT, False, True, True),
    "dbrx": (LLAMA3_SPLIT, False, False, False),
    "smaug-bpe": (LLAMA3_SPLIT, False, True, False),
    "tekken": (TEKKEN_SPLIT, False, True, True),
    "pixtral": (LLAMA3_SPLIT, False, True, True),  # pixtral-12b; a nemotron_h file takes tekken's
    "gpt-4o": (GPT4O_SPLIT, False, False, False),
    "llama4": (GPT4O_SPLIT, False, True, True),
    "smollm": ("digits", False, False, False),
    "refact": (None, False, False, False),
}


def spm_merges(tokens, scores):
    """transformers' SentencePieceExtractor.extract with scores: a piece's splits into two
    vocabulary pieces (code points), ordered by the halves' ids; all merges by descending score."""
    vocab = {piece: i for i, piece in enumerate(tokens)}
    merges = []
    for piece, score in zip(tokens, scores):
        local = [(piece[:k], piece[k:], score) for k in range(1, len(piece))
                 if piece[:k] in vocab and piece[k:] in vocab]
        merges += sorted(local, key=lambda m: (vocab[m[0]], vocab[m[1]]))
    merges = sorted(merges, key=lambda m: m[2], reverse=True)
    return [[a, b] for a, b, _ in merges]


def spm_metadata(g: G) -> dict:
    """A SentencePiece vocabulary (tokenizer.ggml.model llama) with scores, and the tokenizer.json
    transformers' LlamaConverter writes for it (byte fallback, U+2581 prefix, BOS template)."""
    sp = "▁"
    letters = ["a", "b", "c", "é", "ß"]
    pieces = [sp] + letters
    for x in [sp] + letters:
        for y in letters:
            pieces.append(x + y)
    for x in list(pieces[len(letters) + 1:]):
        for y in letters:
            if len(pieces) < VOCAB - 259:
                pieces.append(x + y)
    pieces = list(dict.fromkeys(pieces))
    pieces += [sp * n for n in (2, 4)]  # whitespace runs: SentencePiece scores them -1e9
    pieces = pieces[:VOCAB - 259]
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)] + pieces
    types = [2, 3, 3] + [6] * 256 + [1] * len(pieces)
    scores = [0.0] * 259 + [-1e9 if set(p) == {sp} and len(p) > 1 else -float(i) for i, p in enumerate(pieces)]
    scores[259 + 7] = scores[259 + 8]  # a tie: the stable sort keeps the collection order
    w = g.writer
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_chat_template(TEMPLATE)
    f32 = [float(np.float32(s)) for s in scores]
    added = [{"id": i, "content": tokens[i], "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for i in range(3)]
    a = {"Sequence": {"id": "A", "type_id": 0}}
    b = {"Sequence": {"id": "B", "type_id": 1}}
    head = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": sp},
                {"type": "Replace", "pattern": {"String": " "}, "content": sp}]},
            "pre_tokenizer": None,
            "post_processor": {"type": "TemplateProcessing",
                               "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, a],
                               "pair": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, a,
                                        {"SpecialToken": {"id": "<s>", "type_id": 1}}, b],
                               "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}},
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": sp}, "content": " "},
                {"type": "ByteFallback"}, {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}}
    model = {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
             "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
             "vocab": {tok: i for i, tok in enumerate(tokens)}, "merges": spm_merges(tokens, f32)}
    decoder = {str(x["id"]): {"content": x["content"], "lstrip": False, "normalized": False, "rstrip": False,
                              "single_word": False, "special": True} for x in added}
    return {
        "tokenizer.json": compact({**head, "model": model}),
        "tokenizer_config.json": compact({"added_tokens_decoder": decoder, "bos_token": "<s>", "eos_token": "</s>",
                                          "pad_token": None, "unk_token": "<unk>", "add_bos_token": True}),
        "generation_config.json": compact({"bos_token_id": 1, "eos_token_id": 2}),
        "chat_template.jinja": TEMPLATE,
    }


def twin_value(g: G, name: str):
    """The twin's copy of a GGUF tensor: BF16 words when stored as BF16, else float32."""
    if name in g.words:
        return "BF16", g.words[name]
    return "F32", g.values[name]


def put_twin(t: Hf, hf_name: str, g: G, gguf_name: str, transform=None):
    dtype, data = twin_value(g, gguf_name)
    if transform is not None:
        data = transform(g.values[gguf_name] if dtype == "BF16" else data)
        dtype = "F32"
    t.put(hf_name, dtype, data)


# ------------------------------------------------------------------------------------------------
# Qwen3.5 (qwen35 / qwen35moe)

H, HEADS, KV, D = 256, 4, 2, 64
LK, LKD, LV, LVD, TAPS = 2, 64, 4, 64, 4
LAYERS, INTER = 4, 384
EXPERTS, MOE_INTER, SHARED = 4, 256, 256
PER = LV // LK


KQUANTS = (Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K)
# Formats whose 256-value superblocks span value heads (the converter keeps such an out_proj tiled).
SUPERBLOCKS = KQUANTS + (Q.IQ4_XS, Q.IQ3_XXS, Q.IQ3_S, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S, Q.IQ1_S, Q.IQ1_M,
                         Q.TQ1_0, Q.TQ2_0)


@contextlib.contextmanager
def moe_geometry(inter: int):
    """Build Qwen3.5 MoE cases with routed experts `inter` wide (the down banks' K)."""
    global MOE_INTER
    saved = MOE_INTER
    MOE_INTER = inter
    try:
        yield
    finally:
        MOE_INTER = saved


@contextlib.contextmanager
def qwen_geometry(lk: int, lv: int):
    """Build Qwen3.5 cases with `lk` key heads and `lv` value heads (PER = lv // lk)."""
    global LK, LV, PER
    saved = LK, LV, PER
    LK, LV, PER = lk, lv, lv // lk
    try:
        yield
    finally:
        LK, LV, PER = saved


def tiled(values, dim, head_dim):
    """llama.cpp's _reorder_v_heads: grouped-by-key-head value heads to tiled, along `dim`."""
    shape = list(values.shape)
    new = shape[:dim] + [LK, PER, head_dim] + shape[dim + 1:]
    out = values.reshape(new)
    out = np.swapaxes(out, dim, dim + 1)
    return np.ascontiguousarray(out).reshape(shape)


def grouped(values, dim, head_dim):
    """The inverse: tiled value heads back to grouped order."""
    shape = list(values.shape)
    new = shape[:dim] + [PER, LK, head_dim] + shape[dim + 1:]
    out = values.reshape(new)
    out = np.swapaxes(out, dim, dim + 1)
    return np.ascontiguousarray(out).reshape(shape)


def qwen_case(work: Path, moe: bool, quant: dict | None, seed: int, rebuilt: bool = False,
              base_model: str | None = None, files: bool = True, q8_ggml: bool = False,
              dt_bias: int | None = None, pre: str = "qwen35"):
    """O -> G (llama.cpp's transforms) -> T (inverse, from G); returns (G dir, T dir, stats).
    With `rebuilt`, G carries tokenizer metadata instead of files beside it; `base_model` names
    a repository URL as general.base_model.0.repo_url; `files` writes resources beside G;
    `dt_bias` stores blk.N.ssm_dt.bias (a tensor without .weight) in that ggml block type;
    `pre` is the rebuilt tokenizer's tokenizer.ggml.pre."""
    rng = np.random.default_rng(seed)
    arch = "qwen35moe" if moe else "qwen35"
    gdir = work / "gguf"
    gdir.mkdir(parents=True)
    g = G(gdir / "model.gguf", arch, rng)
    g.q8_ggml = q8_ggml
    t = Hf()
    stats = {"norm values": 0, "norm values inexact": 0, "A_log values": 0, "A_log values inexact": 0}
    # Metadata through gguf-py's own writers, so every key is llama.cpp's spelling.
    w = g.writer
    w.add_block_count(LAYERS + 1)
    w.add_nextn_predict_layers(1)
    w.add_context_length(4096)
    w.add_embedding_length(H)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV)
    w.add_key_length(D)
    w.add_value_length(D)
    w.add_rope_freq_base(10000000.0)
    w.add_rope_dimension_count(D // 4)
    w.add_rope_dimension_sections([3, 3, 2, 0])
    w.add_layer_norm_rms_eps(1e-6)
    w.add_ssm_conv_kernel(TAPS)
    w.add_ssm_state_size(LKD)
    w.add_ssm_group_count(LK)
    w.add_ssm_time_step_rank(LV)
    w.add_ssm_inner_size(LV * LVD)
    w.add_full_attention_interval(4)
    if moe:
        w.add_expert_count(EXPERTS)
        w.add_expert_used_count(2)
        w.add_expert_feed_forward_length(MOE_INTER)
        w.add_expert_shared_feed_forward_length(SHARED)
    else:
        w.add_feed_forward_length(INTER)

    def weight(o_values):
        return bf16_round(o_values)

    def randn(*shape, scale=0.05):
        return weight(rng.standard_normal(shape).astype(np.float32) * scale)

    def matrix(gguf_name, hf_name, values, to_gguf=None, from_gguf=None, qtype=None, columns=False):
        """A projection: stored BF16 or as random `qtype` blocks (then O is what they decode to).
        NVFP4: the twin holds the same codes as compressed-tensors NVFP4 (`columns`: from_gguf
        reorders column heads, applied to code bytes and scale words at their own widths)."""
        stored = values if to_gguf is None else to_gguf(values)
        if qtype == Q.NVFP4 and stored.shape[-1] % 64 == 0 and stored.ndim == 2:
            rows, k = stored.shape
            codes = rng.integers(0, 256, (rows, k // 2), dtype=np.uint8)
            scales = rng.integers(0, 0x7F, (rows, k // 16), dtype=np.uint8)
            scale2 = np.float32(rng.uniform(0.3, 0.9))
            input_scale = np.float32(rng.uniform(0.8, 1.6))
            g.nvfp4(gguf_name, codes, scales)
            g.f32(gguf_name[: -len(".weight")] + ".scale", np.array([scale2], np.float32))
            g.f32(gguf_name[: -len(".weight")] + ".input_scale", np.array([input_scale], np.float32))
            if from_gguf is not None and columns:
                codes, scales = grouped(codes, 1, LVD // 2), grouped(scales, 1, LVD // 16)
            elif from_gguf is not None:
                codes, scales = from_gguf(codes), from_gguf(scales)
            prefix = hf_name[: -len(".weight")]
            t.put(prefix + ".weight_packed", "U8", codes)
            t.put(prefix + ".weight_scale", "F8_E4M3", scales)
            t.f32(prefix + ".weight_global_scale", np.array([np.float32(1.0) / scale2], np.float32))
            t.f32(prefix + ".input_global_scale", np.array([np.float32(1.0) / input_scale], np.float32))
            return
        elif qtype is not None and qtype != Q.NVFP4 and stored.shape[-1] % gguf.GGML_QUANT_SIZES[qtype][0] == 0:
            g.quant(gguf_name, stored.shape, qtype)
        else:
            g.bf16(gguf_name, stored)
        dtype, data = twin_value(g, gguf_name)
        if from_gguf is not None:
            data = from_gguf(data if dtype == "F32" else data)
        t.put(hf_name, dtype, data)

    def norm(gguf_name, hf_name, n):
        o = randn(n, scale=0.2)
        g.f32(gguf_name, o + np.float32(1.0))  # binary32 1 + w
        back = (g.values[gguf_name] - np.float32(1.0)).astype(np.float32)
        t.f32(hf_name, back)
        stats["norm values"] += n
        stats["norm values inexact"] += int(np.count_nonzero(bf16_round(back) != o))

    def q_of(role):
        return None if quant is None else quant.get(role)

    # A K-quant out_proj whose superblocks span value heads stays in llama.cpp's tiled column
    # order (the converter cannot move it losslessly); the twin says the same in its config.
    tiled_out = q_of("gdn") in SUPERBLOCKS and PER > 1 and LVD % 256 != 0

    def block(s, p, linear):
        norm(s + "attn_norm.weight", p + "input_layernorm.weight", H)
        norm(s + "post_attention_norm.weight", p + "post_attention_layernorm.weight", H)
        if not linear:
            for gname, role, rows, k in (("attn_q", "q_proj", 2 * HEADS * D, H), ("attn_k", "k_proj", KV * D, H),
                                         ("attn_v", "v_proj", KV * D, H), ("attn_output", "o_proj", H, HEADS * D)):
                matrix(s + gname + ".weight", p + "self_attn." + role + ".weight", randn(rows, k), qtype=q_of("attention"))
            norm(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", D)
            norm(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", D)
        else:
            l = p + "linear_attn."
            kg, vg = LK * LKD, LV * LVD
            channels = 2 * kg + vg

            def qkv_to(v):
                return np.concatenate([v[:2 * kg], tiled(v[2 * kg:], 0, LVD)])

            def qkv_from(v):
                return np.concatenate([v[:2 * kg], grouped(v[2 * kg:], 0, LVD)])

            matrix(s + "attn_qkv.weight", l + "in_proj_qkv.weight", randn(channels, H),
                   lambda v: qkv_to(v), lambda v: qkv_from(v), q_of("gdn"))
            matrix(s + "attn_gate.weight", l + "in_proj_z.weight", randn(vg, H),
                   lambda v: tiled(v, 0, LVD), lambda v: grouped(v, 0, LVD), q_of("gdn"))
            for gname, role in (("ssm_alpha", "in_proj_a"), ("ssm_beta", "in_proj_b")):
                matrix(s + gname + ".weight", l + role + ".weight", randn(LV, H),
                       lambda v: tiled(v, 0, 1), lambda v: grouped(v, 0, 1), q_of("control"))
            matrix(s + "ssm_out.weight", l + "out_proj.weight", randn(H, vg),
                   lambda v: tiled(v, 1, LVD), None if tiled_out else (lambda v: grouped(v, 1, LVD)),
                   q_of("gdn"), columns=True)
            g.f32(s + "ssm_norm.weight", randn(LVD))
            put_twin(t, l + "norm.weight", g, s + "ssm_norm.weight")
            conv = randn(channels, 1, TAPS)
            stored = np.concatenate([conv[:2 * kg, 0], tiled(conv[2 * kg:, 0], 0, LVD)])
            g.f32(s + "ssm_conv1d.weight", stored)
            back = np.concatenate([stored[:2 * kg], grouped(stored[2 * kg:], 0, LVD)])
            t.f32(l + "conv1d.weight", back.reshape(channels, 1, TAPS))
            a_log = rng.standard_normal(LV).astype(np.float32)
            neg = (-torch.exp(torch.from_numpy(a_log))).numpy()
            g.f32(s + "ssm_a", tiled(neg, 0, 1))
            recovered = np.log(-grouped(g.values[s + "ssm_a"], 0, 1).astype(np.float64)).astype(np.float32)
            t.f32(l + "A_log", recovered)
            stats["A_log values"] += LV
            stats["A_log values inexact"] += int(np.count_nonzero(recovered != a_log))
            if dt_bias is None:
                g.f32(s + "ssm_dt.bias", tiled(randn(LV), 0, 1))
            else:
                g.quant(s + "ssm_dt.bias", (LV,), dt_bias)
            t.f32(l + "dt_bias", grouped(g.values[s + "ssm_dt.bias"], 0, 1))
        if not moe:
            for gname, role, rows, k in (("ffn_gate", "gate_proj", INTER, H), ("ffn_up", "up_proj", INTER, H),
                                         ("ffn_down", "down_proj", H, INTER)):
                matrix(s + gname + ".weight", p + "mlp." + role + ".weight", randn(rows, k), qtype=q_of("mlp"))
            return
        m = p + "mlp."
        matrix(s + "ffn_gate_inp.weight", m + "gate.weight", randn(EXPERTS, H))
        g.f32(s + "ffn_gate_inp_shexp.weight", randn(1, H))
        put_twin(t, m + "shared_expert_gate.weight", g, s + "ffn_gate_inp_shexp.weight")
        for gname, role, rows, k in (("ffn_gate_shexp", "gate_proj", SHARED, H), ("ffn_up_shexp", "up_proj", SHARED, H),
                                     ("ffn_down_shexp", "down_proj", H, SHARED)):
            matrix(s + gname + ".weight", m + "shared_expert." + role + ".weight", randn(rows, k), qtype=q_of("experts"))
        gate_up = randn(EXPERTS, 2 * MOE_INTER, H)
        # The routed banks: "experts_gate_up" / "experts_down" override "experts" (which the
        # shared expert takes).
        routed_gate_up = q_of("experts_gate_up") or q_of("experts")
        for half, gname in enumerate(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
            part = gate_up[:, half * MOE_INTER:(half + 1) * MOE_INTER]
            if routed_gate_up:
                g.quant(s + gname, part.shape, routed_gate_up)
            else:
                g.bf16(s + gname, part)
        halves = [twin_value(g, s + name) for name in ("ffn_gate_exps.weight", "ffn_up_exps.weight")]
        dtype = halves[0][0]
        t.put(m + "experts.gate_up_proj", dtype, np.concatenate([halves[0][1], halves[1][1]], axis=1))
        matrix(s + "ffn_down_exps.weight", m + "experts.down_proj", randn(EXPERTS, H, MOE_INTER),
               qtype=q_of("experts_down") or q_of("experts"))

    matrix("token_embd.weight", "model.embed_tokens.weight", randn(VOCAB, H), qtype=q_of("embedding"))
    matrix("output.weight", "lm_head.weight", randn(VOCAB, H), qtype=q_of("embedding"))
    norm("output_norm.weight", "model.norm.weight", H)
    for layer in range(LAYERS):
        block(f"blk.{layer}.", f"model.layers.{layer}.", (layer + 1) % 4 != 0)
    mtp = f"blk.{LAYERS}."
    block(mtp, "mtp.layers.0.", False)
    matrix(mtp + "nextn.eh_proj.weight", "mtp.fc.weight", randn(H, 2 * H))
    for gname, hf in (("nextn.enorm", "pre_fc_norm_embedding"), ("nextn.hnorm", "pre_fc_norm_hidden"),
                      ("nextn.shared_head_norm", "norm")):
        norm(mtp + gname + ".weight", "mtp." + hf + ".weight", H)
    expected = tokenizer_metadata(g, pre, qwen35=True) if rebuilt else None
    if base_model:
        g.writer.add_string("general.base_model.0.repo_url", base_model)
    g.save(gdir, files=files and not rebuilt)

    fields = {
        "hidden_act": "silu", "attention_bias": False, "attn_output_gate": True,
        "hidden_size": H, "vocab_size": VOCAB, "num_hidden_layers": LAYERS,
        "max_position_embeddings": 4096, "tie_word_embeddings": False, "rms_norm_eps": 1e-06,
        "full_attention_interval": 4, "num_attention_heads": HEADS, "num_key_value_heads": KV,
        "head_dim": D, "rope_parameters": {"rope_type": "default", "mrope_interleaved": True,
                                           "mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25,
                                           "rope_theta": 10000000},
        "mamba_ssm_dtype": "float32", "linear_num_key_heads": LK, "linear_key_head_dim": LKD,
        "linear_num_value_heads": LV, "linear_value_head_dim": LVD,
        "linear_conv_kernel_dim": TAPS, "mtp_num_hidden_layers": 1,
    }
    if tiled_out:
        fields["linear_output_head_order"] = "tiled"
    if moe:
        fields.update({"num_experts": EXPERTS, "num_experts_per_tok": 2,
                       "moe_intermediate_size": MOE_INTER, "shared_expert_intermediate_size": SHARED})
    else:
        fields["intermediate_size"] = INTER
    arch_name = "Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"
    t.save(work / "twin", {"architectures": [arch_name], **fields})
    for role, text in (expected or {}).items():
        (work / "twin" / role).write_text(text, encoding="utf-8")
    return gdir / "model.gguf", work / "twin", stats


# ------------------------------------------------------------------------------------------------
# Dense decoders (llama / qwen2 / qwen3 / granite / olmo2 / smollm3 / mistral3)

DH, DHEADS, DKV, DD, DLAYERS, DINTER = 256, 4, 2, 64, 2, 512
DENSE = {  # case -> (GGUF arch, architecture, model_type, tokenizer pre, q/k/v biases, q/k norm,
    #                  Q/K rows permuted (LlamaModel.permute), OLMo post-norms)
    "llama": ("llama", "LlamaForCausalLM", "llama", "llama-bpe", False, None, True, False),
    "tinyllama": ("llama", "LlamaForCausalLM", "llama", "spm", False, None, True, False),
    "smollm2": ("llama", "LlamaForCausalLM", "llama", "smollm", False, None, True, False),
    "qwen2": ("qwen2", "Qwen2ForCausalLM", "qwen2", "qwen2", True, None, False, False),
    "qwen3": ("qwen3", "Qwen3ForCausalLM", "qwen3", "qwen2", False, "head", False, False),
    "granite": ("granite", "GraniteForCausalLM", "granite", "refact", False, None, True, False),
    "olmo2": ("olmo2", "Olmo2ForCausalLM", "olmo2", "dbrx", False, "projection", False, True),
    "olmo3": ("olmo2", "Olmo3ForCausalLM", "olmo3", "dbrx", False, "projection", False, True),
    "smollm3": ("smollm3", "SmolLM3ForCausalLM", "smollm3", "smaug-bpe", False, None, True, False),
    "mistral3": ("mistral3", "Ministral3ForCausalLM", "ministral3", "tekken", False, None, True, False),
    "phi3": ("phi3", "Phi3ForCausalLM", "phi3", "gpt-4o", False, None, False, False),
    "muse": ("muse-glimmer", "MuseGlimmerForConditionalGeneration", "muse_glimmer_text", "llama4", False,
             None, True, False),
}


def llama_permute(values, heads):
    """llama.cpp's LlamaModel.permute: each head's rows interleaved for ggml's NORM rope."""
    return np.ascontiguousarray(values.reshape(heads, 2, values.shape[0] // heads // 2, *values.shape[1:])
                                .swapaxes(1, 2)).reshape(values.shape)


def llama_unpermute(values, heads):
    return np.ascontiguousarray(values.reshape(heads, values.shape[0] // heads // 2, 2, *values.shape[1:])
                                .swapaxes(1, 2)).reshape(values.shape)


def llama3_factors(theta, dim, factor=32.0, low=1.0, high=4.0, old=8192):
    """rope_freqs as llama.cpp's LlamaModel.generate_extra_tensors writes them."""
    import math
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    out = []
    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq
        if wavelen < old / high:
            out.append(1)
        elif wavelen > old / low:
            out.append(factor)
        else:
            smooth = (old / wavelen - low) / (high - low)
            out.append(1 / ((1 - smooth) / factor + smooth))
    return np.array(out, np.float32)


def llama3_inv_freq(theta, dim, factor=32.0, low=1.0, high=4.0, old=8192):
    """transformers' _compute_llama3_parameters, in binary32."""
    import math
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    wavelen = 2 * math.pi / inv
    scaled = torch.where(wavelen > old / low, inv / factor, inv)
    smooth = (old / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < old / high) * ~(wavelen > old / low)
    return torch.where(medium, smoothed, scaled).numpy()


def dense_case(work: Path, arch: str, quant: dict | None, seed: int, rebuilt: bool = False,
               pre_tokenizer: str | None = None,
               tied: bool = True, llama3: bool = False):
    """O -> G (llama.cpp's LlamaModel / Qwen2Model / Qwen3Model / GraniteModel / Olmo2Model /
    SmolLM3Model / Mistral3Model) -> T (the inverse, from G)."""
    gguf_arch, architecture, model_type, pre, qkv_bias, qk_norm, permuted, post_norms = DENSE[arch]
    pre = pre_tokenizer or pre
    layers = 4 if arch in ("smollm3", "olmo3", "muse") else DLAYERS  # a NoPE / full-attention 4th layer
    rng = np.random.default_rng(seed)
    gdir = work / "gguf"
    gdir.mkdir(parents=True)
    g = G(gdir / "model.gguf", gguf_arch, rng)
    g.q8_ggml = True
    t = Hf()
    theta = {"llama": 500000.0, "tinyllama": 10000.0, "granite": 10000000.0, "olmo2": 500000.0,
             "olmo3": 500000.0, "smollm3": 5000000.0, "muse": 500000.0}.get(arch, 1000000.0)
    w = g.writer
    w.add_block_count(layers)
    w.add_context_length(4096)
    w.add_embedding_length(DH)
    w.add_feed_forward_length(DINTER)
    w.add_head_count(DHEADS)
    w.add_head_count_kv(DKV)
    w.add_key_length(DD)
    w.add_value_length(DD)
    w.add_rope_freq_base(theta)
    rot = DD * 3 // 4 if arch == "phi3" else DD  # Phi-4-mini rotates 3/4 of each head
    w.add_rope_dimension_count(rot)
    w.add_layer_norm_rms_eps(1e-6)
    # T follows G's binary32 metadata as the GGUF reader gives it back: the shortest decimal that
    # round-trips the float (0.22, not 0.2199999988079071).
    f32 = lambda x: float(str(np.float32(x)))  # noqa: E731
    extra = {}
    rope = {"rope_type": "default", "rope_theta": theta}
    if arch == "granite":
        w.add_attention_scale(0.015625)
        w.add_embedding_scale(12.0)
        w.add_residual_scale(0.22)
        w.add_logit_scale(8.0)
        extra = {"attention_multiplier": 0.015625, "embedding_multiplier": 12.0,
                 "residual_multiplier": f32(0.22), "logits_scaling": 8.0}
    elif arch == "smollm3":
        extra = {"no_rope_layers": [int((i + 1) % 4 != 0) for i in range(layers)], "no_rope_layer_interval": 4}
    elif arch == "olmo3":
        w.add_sliding_window(64)
        w.add_sliding_window_pattern([True, True, True, False])
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(8.0)
        w.add_rope_scaling_orig_ctx_len(1024)
        w.add_rope_scaling_attn_factors(1.2079441541679836)  # the key OLMo 3 GGUFs carry
        w.add_rope_scaling_yarn_beta_fast(32.0)
        w.add_rope_scaling_yarn_beta_slow(1.0)
        extra = {"sliding_window": 64, "layer_types": ["sliding_attention"] * 3 + ["full_attention"]}
        rope = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 1024,
                "attention_factor": f32(1.2079441541679836), "beta_fast": 32.0, "beta_slow": 1.0}
        extra["rope_scaling"] = rope
        extra["rope_theta"] = theta
        rope = None  # OLMo 3's legacy keys: scaling on full-attention layers only
    elif arch == "mistral3":
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(16.0)
        w.add_rope_scaling_orig_ctx_len(1024)
        w.add_rope_scaling_yarn_beta_fast(32.0)
        w.add_rope_scaling_yarn_beta_slow(1.0)
        w.add_rope_scaling_yarn_log_mul(1.0)
        w.add_attn_temperature_scale(0.1)
        rope = {"rope_type": "yarn", "rope_theta": theta, "factor": 16.0, "original_max_position_embeddings": 1024,
                "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": f32(0.1)}
    elif arch == "muse":
        # MuseGlimmerModel: the window and the period of its layer pattern (as unsloth's files
        # carry it; the official ones a flag per layer), output_multiplier as logit_scale, the
        # soft cap; T's layer_rope_theta has 0 on the full (NoPE) layer.
        w.add_sliding_window(64)
        w.add_uint32(gguf.Keys.Attention.SLIDING_WINDOW_PATTERN.format(arch=gguf_arch), 4)
        w.add_logit_scale(0.19611613513818404)
        w.add_final_logit_softcapping(20.0)
        extra = {"sliding_window": 64, "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
                 "layer_rope_theta": [theta] * 3 + [0.0], "output_multiplier": f32(0.19611613513818404),
                 "final_logit_softcapping": 20.0, "post_norm_eps": 1e-08, "qk_scale_factor": f32(3.87)}
    elif arch == "phi3":
        # Phi3MiniModel: partial rotary, LongRoPE factors as tensors (added below), the attention
        # factor it computes, and Phi-4's sliding_window past every position.
        import math
        w.add_rope_scaling_orig_ctx_len(1024)
        w.add_rope_scaling_attn_factors(math.sqrt(1 + math.log(4096 / 1024) / math.log(1024)))
        w.add_sliding_window(262144)
        rope = {"rope_type": "longrope", "rope_theta": theta, "original_max_position_embeddings": 1024}
        extra = {"partial_rotary_factor": 0.75, "sliding_window": 262144}

    def randn(*shape, scale=0.05):
        return bf16_round(rng.standard_normal(shape).astype(np.float32) * scale)

    def q_of(role):
        return None if quant is None else quant.get(role)

    def matrix(gname, hname, rows, k, qtype=None, heads=None):
        values = randn(rows, k)
        stored = values if heads is None else llama_permute(values, heads)
        if qtype is not None and k % gguf.GGML_QUANT_SIZES[qtype][0] == 0:
            g.quant(gname, stored.shape, qtype)
        else:
            g.bf16(gname, stored)
        dtype, data = twin_value(g, gname)
        t.put(hname, dtype, data if heads is None else llama_unpermute(data, heads))

    def vector(gname, hname, n, scale=0.2):
        g.f32(gname, randn(n, scale=scale) + np.float32(1.0))
        put_twin(t, hname, g, gname)

    matrix("token_embd.weight", "model.embed_tokens.weight", VOCAB, DH, q_of("embedding"))
    if not tied:
        matrix("output.weight", "lm_head.weight", VOCAB, DH, q_of("embedding"))
    vector("output_norm.weight", "model.norm.weight", DH)
    for layer in range(layers):
        s, p = f"blk.{layer}.", f"model.layers.{layer}."
        if arch == "muse":  # the sandwich, each gain stored as binary32 1 + w
            for gname, hname in (("attn_norm", "input_layernorm"), ("post_attention_norm", "post_attention_layernorm"),
                                 ("ffn_norm", "pre_feedforward_layernorm"),
                                 ("post_ffw_norm", "post_feedforward_layernorm")):
                g.f32(s + gname + ".weight", randn(DH, scale=0.2) + np.float32(1.0))
                t.f32(p + hname + ".weight", (g.values[s + gname + ".weight"] - np.float32(1.0)).astype(np.float32))
            matrix(s + "attn_gate.weight", p + "self_attn.gate_proj.weight", DHEADS * DD, DH, q_of("attention"))
            g.f32(s + "attn_q_norm.weight", np.full(DD, 3.87, np.float32))
            g.f32(s + "attn_k_norm.weight", np.ones(DD, np.float32))
        elif post_norms:
            vector(s + "post_attention_norm.weight", p + "post_attention_layernorm.weight", DH)
            vector(s + "post_ffw_norm.weight", p + "post_feedforward_layernorm.weight", DH)
        else:
            vector(s + "attn_norm.weight", p + "input_layernorm.weight", DH)
            vector(s + "ffn_norm.weight", p + "post_attention_layernorm.weight", DH)
        if arch == "phi3":  # fused as published: qkv_proj (q, k, v rows), gate_up_proj (gate, up rows)
            matrix(s + "attn_qkv.weight", p + "self_attn.qkv_proj.weight", (DHEADS + 2 * DKV) * DD, DH,
                   q_of("attention"))
        for gname, role, heads in () if arch == "phi3" else (
                ("attn_q", "q_proj", DHEADS), ("attn_k", "k_proj", DKV), ("attn_v", "v_proj", DKV)):
            interleaved = heads if permuted and role != "v_proj" else None
            matrix(s + gname + ".weight", p + "self_attn." + role + ".weight", heads * DD, DH,
                   q_of("attention"), interleaved)
            if qkv_bias:
                g.f32(s + gname + ".bias", randn(heads * DD))
                put_twin(t, p + "self_attn." + role + ".bias", g, s + gname + ".bias")
        matrix(s + "attn_output.weight", p + "self_attn.o_proj.weight", DH, DHEADS * DD, q_of("attention"))
        if qk_norm == "head":
            vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", DD)
            vector(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", DD)
        elif qk_norm == "projection":
            vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", DHEADS * DD)
            vector(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", DKV * DD)
        mlp = (("ffn_up", "gate_up_proj", 2 * DINTER, DH),) if arch == "phi3" else (
            ("ffn_gate", "gate_proj", DINTER, DH), ("ffn_up", "up_proj", DINTER, DH))
        for gname, role, rows, k in mlp + (("ffn_down", "down_proj", DH, DINTER),):
            matrix(s + gname + ".weight", p + "mlp." + role + ".weight", rows, k, q_of("mlp"))
    if arch == "phi3":
        for name, key in (("rope_factors_long.weight", "long_factor"), ("rope_factors_short.weight", "short_factor")):
            g.f32(name, rng.uniform(1.0, 8.0, rot // 2).astype(np.float32))
            rope[key] = [float(f) for f in g.values[name]]
    if llama3:
        g.f32("rope_freqs.weight", llama3_factors(theta, DD))
        rope = {"rope_type": "frequency_factors", "rope_theta": theta,
                "factors": [float(f) for f in g.values["rope_freqs.weight"]]}
    expected = tokenizer_metadata(g, pre) if rebuilt else None
    g.save(gdir, files=not rebuilt)
    t.save(work / "twin", {
        "architectures": [architecture], "model_type": model_type, "hidden_size": DH, "vocab_size": VOCAB,
        "num_hidden_layers": layers, "max_position_embeddings": 4096, "num_attention_heads": DHEADS,
        "num_key_value_heads": DKV, "head_dim": DD, "intermediate_size": DINTER, "rms_norm_eps": 1e-06,
        "hidden_act": "silu", "tie_word_embeddings": tied, "attention_bias": qkv_bias,
        **({"rope_parameters": rope} if rope is not None else {}), **extra,
    })
    for role, text in (expected or {}).items():
        (work / "twin" / role).write_text(text, encoding="utf-8")
    return gdir / "model.gguf", work / "twin", {}


# ------------------------------------------------------------------------------------------------
# gpt-oss

OH, OHEADS, OKV, OD, OLAYERS, OEXPERTS, OUSED, OFF = 256, 4, 2, 64, 2, 4, 2, 256


def mxfp4_raw(codes, exps):
    """ggml MXFP4 blocks of codes [..., k/2] (natural nibble order) and exponents [..., k/32]."""
    lead, k = codes.shape[:-1], codes.shape[-1] * 2
    rows = int(np.prod(lead))
    c = codes.reshape(rows, k // 2)
    nibbles = np.stack((c & 15, c >> 4), -1).reshape(rows, k // 32, 32)
    qs = (nibbles[:, :, :16] | (nibbles[:, :, 16:] << 4)).astype(np.uint8)
    raw = np.concatenate([exps.reshape(rows, k // 32, 1), qs], -1)
    return raw.reshape(*lead, k // 32 * 17)


def gpt_oss_case(work: Path, seed: int, q8_vocabulary: bool = True, harmony: bool = False):
    """O -> G (llama.cpp's GptOssModel: gate/up banks split from the interleaved gate_up_proj,
    MXFP4 blocks repacked, attention and vocabulary in Q8_0 as ggml-org publishes it, F32 router,
    sinks and biases) -> T (transformers' gpt-oss layout: <name>_blocks / _scales with gate and up
    rows interleaved). Without q8_vocabulary the embedding and output head are BF16 (a twin
    comparison under --maintainer-quantize decodes a published Q8_0 table to BF16 but quantizes
    the twin's plain one). With harmony the GGUF carries a harmony-style tokenizer.ggml vocabulary
    (<|endoftext|> 0, <|return|> 1 as the named eos, <|call|> 3, <|end|> 4, all CONTROL) and no
    tokenizer files, so the converter rebuilds them."""
    rng = np.random.default_rng(seed)
    gdir = work / "gguf"
    gdir.mkdir(parents=True)
    g = G(gdir / "model.gguf", "gpt-oss", rng)
    g.q8_ggml = True
    t = Hf()
    w = g.writer
    w.add_block_count(OLAYERS)
    w.add_context_length(4096)
    w.add_embedding_length(OH)
    w.add_feed_forward_length(OFF)
    w.add_head_count(OHEADS)
    w.add_head_count_kv(OKV)
    w.add_key_length(OD)
    w.add_value_length(OD)
    w.add_rope_freq_base(150000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_expert_count(OEXPERTS)
    w.add_expert_used_count(OUSED)
    w.add_expert_feed_forward_length(OFF)
    w.add_sliding_window(128)
    w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
    w.add_rope_scaling_factor(32.0)
    w.add_rope_scaling_orig_ctx_len(1024)
    w.add_rope_scaling_yarn_beta_fast(32.0)
    w.add_rope_scaling_yarn_beta_slow(1.0)
    f32 = lambda x: float(str(np.float32(x)))  # noqa: E731

    def randn(*shape, scale=0.05):
        return bf16_round(rng.standard_normal(shape).astype(np.float32) * scale)

    def q8(gname, hname, rows, k):
        g.quant(gname, (rows, k), Q.Q8_0)
        put_twin(t, hname, g, gname)

    def vector(gname, hname, values):
        g.f32(gname, values)
        put_twin(t, hname, g, gname)

    def codes_exps(*lead, k):
        codes = rng.integers(0, 256, (*lead, k // 2), dtype=np.uint8)
        exps = rng.integers(118, 127, (*lead, k // 32), dtype=np.uint8)
        return codes, exps

    for gname, hname in (("token_embd.weight", "model.embed_tokens.weight"), ("output.weight", "lm_head.weight")):
        if q8_vocabulary:
            q8(gname, hname, VOCAB, OH)
        else:
            g.bf16(gname, randn(VOCAB, OH))
            put_twin(t, hname, g, gname)
    vector("output_norm.weight", "model.norm.weight", randn(OH, scale=0.2) + np.float32(1.0))
    for layer in range(OLAYERS):
        s, p = f"blk.{layer}.", f"model.layers.{layer}."
        vector(s + "attn_norm.weight", p + "input_layernorm.weight", randn(OH, scale=0.2) + np.float32(1.0))
        vector(s + "post_attention_norm.weight", p + "post_attention_layernorm.weight",
               randn(OH, scale=0.2) + np.float32(1.0))
        for gname, role, heads in (("attn_q", "q_proj", OHEADS), ("attn_k", "k_proj", OKV), ("attn_v", "v_proj", OKV)):
            q8(s + gname + ".weight", p + "self_attn." + role + ".weight", heads * OD, OH)
            vector(s + gname + ".bias", p + "self_attn." + role + ".bias", randn(heads * OD))
        q8(s + "attn_output.weight", p + "self_attn.o_proj.weight", OH, OHEADS * OD)
        vector(s + "attn_output.bias", p + "self_attn.o_proj.bias", randn(OH))
        vector(s + "attn_sinks.weight", p + "self_attn.sinks", randn(OHEADS, scale=1.0))
        vector(s + "ffn_gate_inp.weight", p + "mlp.router.weight", randn(OEXPERTS, OH))
        vector(s + "ffn_gate_inp.bias", p + "mlp.router.bias", randn(OEXPERTS))
        gate_codes, gate_exps = codes_exps(OEXPERTS, OFF, k=OH)
        up_codes, up_exps = codes_exps(OEXPERTS, OFF, k=OH)
        down_codes, down_exps = codes_exps(OEXPERTS, OH, k=OFF)
        for gname, codes, exps in (("ffn_gate_exps", gate_codes, gate_exps), ("ffn_up_exps", up_codes, up_exps),
                                   ("ffn_down_exps", down_codes, down_exps)):
            w.add_tensor(s + gname + ".weight", mxfp4_raw(codes, exps), raw_dtype=Q.MXFP4)
        gate_bias, up_bias, down_bias = randn(OEXPERTS, OFF), randn(OEXPERTS, OFF), randn(OEXPERTS, OH)
        g.f32(s + "ffn_gate_exps.bias", gate_bias)
        g.f32(s + "ffn_up_exps.bias", up_bias)
        g.f32(s + "ffn_down_exps.bias", down_bias)
        # transformers: gate on the even rows of gate_up_proj, up on the odd ones.
        gu_codes = np.stack((gate_codes, up_codes), 2).reshape(OEXPERTS, 2 * OFF, OH // 2)
        gu_exps = np.stack((gate_exps, up_exps), 2).reshape(OEXPERTS, 2 * OFF, OH // 32)
        t.put(p + "mlp.experts.gate_up_proj_blocks", "U8", gu_codes.reshape(OEXPERTS, 2 * OFF, OH // 32, 16))
        t.put(p + "mlp.experts.gate_up_proj_scales", "U8", gu_exps)
        t.f32(p + "mlp.experts.gate_up_proj_bias", np.stack((gate_bias, up_bias), 2).reshape(OEXPERTS, 2 * OFF))
        t.put(p + "mlp.experts.down_proj_blocks", "U8", down_codes.reshape(OEXPERTS, OH, OFF // 32, 16))
        t.put(p + "mlp.experts.down_proj_scales", "U8", down_exps)
        t.f32(p + "mlp.experts.down_proj_bias", down_bias)
    if harmony:
        tokens = [f"t{i}" for i in range(VOCAB)]
        tokens[0], tokens[1], tokens[2], tokens[3], tokens[4] = (
            "<|endoftext|>", "<|return|>", "<|startoftext|>", "<|call|>", "<|end|>")
        tokens[6], tokens[7], tokens[8] = "x", "9", "8"
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("gpt-4o")
        w.add_token_list(tokens)
        w.add_token_types([3] * 5 + [1] * (VOCAB - 5))
        w.add_token_merges(["x 9", "9 8", "x 8"])
        w.add_bos_token_id(2)
        w.add_eos_token_id(1)
        w.add_pad_token_id(0)
        w.add_chat_template(HARMONY_TEMPLATE)
    g.save(gdir, files=not harmony)
    t.save(work / "twin", {
        "architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "hidden_size": OH, "vocab_size": VOCAB,
        "num_hidden_layers": OLAYERS, "max_position_embeddings": 4096, "num_attention_heads": OHEADS,
        "num_key_value_heads": OKV, "head_dim": OD, "intermediate_size": OFF, "num_local_experts": OEXPERTS,
        "num_experts_per_tok": OUSED, "rms_norm_eps": f32(1e-5), "hidden_act": "silu", "attention_bias": True,
        "tie_word_embeddings": False, "swiglu_limit": 7.0, "sliding_window": 128,
        "layer_types": ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(OLAYERS)],
        "rope_parameters": {"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0,
                            "original_max_position_embeddings": 1024, "beta_fast": 32.0, "beta_slow": 1.0,
                            "truncate": False},
    })
    return gdir / "model.gguf", work / "twin", {}


def gpt_oss_cases(root: Path, native: Path, repository: Path) -> list[str]:
    """A gpt-oss GGUF as ggml-org publishes it runs as published under the default recipe
    (moe_published): the MXFP4 experts as mxfp4 and the Q8_0 attention, embedding and output head
    as q8_g32_fp16, word for word, with nothing dequantized."""
    errors = []
    name = "gpt-oss-published"
    work = root / name
    gguf_file, _, _ = gpt_oss_case(work, 60)
    out = work / "published.ninferno"
    result = subprocess.run([str(native), f"--model={gguf_file}", f"--out={out}", "--components=text"],
                            cwd=repository, capture_output=True, text=True)
    if result.returncode:
        errors.append(f"{name}: conversion failed\n{result.stderr[-2000:]}")
    else:
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        formats = {}
        for m in report["methods"]:
            for parameter in m["parameters"]:
                kind = ("experts" if "/mlp/experts/" in parameter and not parameter.endswith("_bias") else
                        "attention" if "/attention/" in parameter and parameter.split("/")[-1] in
                        ("query", "key", "value", "output") else
                        parameter if parameter in ("text/token_embedding", "text/output_head") else None)
                if kind:
                    formats.setdefault(kind, set()).add((m["format"], m["method"]))
        want = {"experts": {("mxfp4", "import_encoded")},
                "attention": {("q8_g32_fp16", "import_encoded")},
                "text/token_embedding": {("q8_g32_fp16", "import_encoded")},
                "text/output_head": {("q8_g32_fp16", "import_encoded")}}
        if formats != want:
            errors.append(f"{name}: expected {want}, got {formats}")
        lossy = [n for n in report.get("native_notes", []) if not n["lossless"]]
        if lossy:
            errors.append(f"{name}: lossy handling {lossy}")
    print(f"{name}: {'ok' if not errors else 'FAILED'}", flush=True)

    # The stops llama.cpp's vocabulary loader adds to the named eos (special_eog_ids): a
    # harmony vocabulary ends on <|return|>, <|endoftext|> and <|call|> (gpt-oss's own
    # generation_config.json), not on the <|end|> between its messages.
    name = "gpt-oss-harmony-stops"
    work = root / name
    gguf_file, _, _ = gpt_oss_case(work, 61, harmony=True)
    out = work / "stops.ninferno"
    result = subprocess.run([str(native), f"--model={gguf_file}", f"--out={out}", "--components=text"],
                            cwd=repository, capture_output=True, text=True)
    failed = []
    if result.returncode:
        failed.append(f"{name}: conversion failed\n{result.stderr[-2000:]}")
    else:
        _, objects, data, offset = directory(out)
        resource = objects.get("resource/text/generation_config.json")
        body = json.loads(data[offset + resource["offset"]:offset + resource["offset"] + resource["bytes"]]) \
            if resource else {}
        if body.get("eos_token_id") != [1, 0, 3]:
            failed.append(f"{name}: expected eos_token_id [1, 0, 3] (<|return|>, <|endoftext|>, <|call|>), "
                          f"got {body}")
    print(f"{name}: {'ok' if not failed else 'FAILED'}", flush=True)
    return errors + failed




# DFlash drafter (llama.cpp's LLM_ARCH_DFLASH, as its MuseGlimmerAssistantModel writes it)

def dflash_case(work: Path, quant: dict | None, seed: int):
    """O -> G (Hugging Face layout kept: plain norms, no Q/K permutation; target_layers one past
    target_layer_ids; tokenizer.ggml.mask_token_id) -> T (a DFlashDraftModel config with
    model_type dflash and the block mask, the tensors under their transformers names)."""
    rng = np.random.default_rng(seed)
    gdir = work / "gguf"
    gdir.mkdir(parents=True)
    g = G(gdir / "model.gguf", "dflash", rng)
    g.q8_ggml = True
    t = Hf()
    layers, taps, window = 2, [1, 3, 5], 64
    w = g.writer
    w.add_block_count(layers)
    w.add_context_length(4096)
    w.add_embedding_length(DH)
    w.add_feed_forward_length(DINTER)
    w.add_head_count(DHEADS)
    w.add_head_count_kv(DKV)
    w.add_key_length(DD)
    w.add_value_length(DD)
    w.add_rope_freq_base(500000.0)
    w.add_layer_norm_rms_eps(1e-5)
    # DFlash keys by name: released gguf-py versions have no add_block_size / add_target_layers.
    w.add_uint32("dflash.block_size", 16)
    w.add_array("dflash.target_layers", [tap + 1 for tap in taps])
    w.add_sliding_window(window)
    w.add_sliding_window_pattern([True, False])
    w.add_uint32("tokenizer.ggml.mask_token_id", VOCAB - 1)

    def randn(*shape, scale=0.05):
        return bf16_round(rng.standard_normal(shape).astype(np.float32) * scale)

    def matrix(gname, hname, rows, k, qtype=None):
        if qtype is not None and k % gguf.GGML_QUANT_SIZES[qtype][0] == 0:
            g.quant(gname, (rows, k), qtype)
        else:
            g.bf16(gname, randn(rows, k))
        put_twin(t, hname, g, gname)

    def vector(gname, hname, n):
        g.f32(gname, randn(n, scale=0.2) + np.float32(1.0))
        put_twin(t, hname, g, gname)

    q_of = (lambda role: None) if quant is None else quant.get
    matrix("fc.weight", "encoder.fc.weight", DH, len(taps) * DH, q_of("attention"))
    vector("enc.output_norm.weight", "encoder.output_norm_enc.weight", DH)
    vector("output_norm.weight", "norm.weight", DH)
    for layer in range(layers):
        s, p = f"blk.{layer}.", f"layers.{layer}."
        vector(s + "attn_norm.weight", p + "input_layernorm.weight", DH)
        vector(s + "ffn_norm.weight", p + "post_attention_layernorm.weight", DH)
        vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", DD)
        vector(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", DD)
        # llama.cpp's Muse Glimmer GGUF mixes formats inside one layer (q/k Q4_K, v Q6_K).
        for gname, role, heads, qtype in (("attn_q", "q_proj", DHEADS, q_of("attention")),
                                          ("attn_k", "k_proj", DKV, q_of("attention")),
                                          ("attn_v", "v_proj", DKV, q_of("value"))):
            matrix(s + gname + ".weight", p + "self_attn." + role + ".weight", heads * DD, DH, qtype)
        matrix(s + "attn_output.weight", p + "self_attn.o_proj.weight", DH, DHEADS * DD, q_of("attention"))
        for gname, role, rows, k in (("ffn_gate", "gate_proj", DINTER, DH), ("ffn_up", "up_proj", DINTER, DH),
                                     ("ffn_down", "down_proj", DH, DINTER)):
            matrix(s + gname + ".weight", p + "mlp." + role + ".weight", rows, k,
                   q_of("down") if gname == "ffn_down" else q_of("mlp"))
    expected = tokenizer_metadata(g, "llama4")
    g.save(gdir, files=False)
    t.save(work / "twin", {
        "architectures": ["DFlashDraftModel"], "model_type": "dflash", "hidden_size": DH, "vocab_size": VOCAB,
        "num_hidden_layers": layers, "num_attention_heads": DHEADS, "num_key_value_heads": DKV, "head_dim": DD,
        "intermediate_size": DINTER, "rms_norm_eps": float(str(np.float32(1e-5))), "hidden_act": "silu",
        "attention_bias": False, "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": window, "rope_theta": 500000.0, "block_size": 16,
        "dflash_config": {"target_layer_ids": taps, "mask_token_id": VOCAB - 1},
    })
    for role, text in expected.items():
        (work / "twin" / role).write_text(text, encoding="utf-8")
    return gdir / "model.gguf", work / "twin", {}


# ------------------------------------------------------------------------------------------------
# Gemma 4

GH, GINTER, GEXPERTS, GMOE, GPLE = 256, 256, 4, 128, 32


def gemma_case(work: Path, moe: bool, mode: str, seed: int, rebuilt: bool = False, small: bool = False):
    """mode: bf16, nvfp4 (compressed-tensors origin; experts also for MoE), mxfp4, iq4; tables
    (small only: BF16 projections, a Q6_K token table and a Q8_0 per-layer table).

    small: Gemma 4 E2B/E4B as llama.cpp writes them - four layers, the last two reading the keys
    and values of the last earlier layer of their kind (attention.shared_kv_layers; their k/v are
    absent, as in Google's QAT GGUFs), twice as wide an MLP there (per-layer feed_forward_length),
    and per-layer embeddings (PLE) of width GPLE."""
    rng = np.random.default_rng(seed)
    gdir = work / "gguf"
    gdir.mkdir(parents=True)
    g = G(gdir / "model.gguf", "gemma4", rng)
    t = Hf()
    g.q8_ggml = True  # Q8_0 blocks as llama.cpp writes them (no code -128)
    stats = {"divisor words": 0, "divisor words changed": 0}
    layers = 4 if small else 2
    shared = 2 if small else 0
    ple = GPLE if small else 0
    widths = [GINTER] * (layers - shared) + [2 * GINTER if small else GINTER] * shared
    w = g.writer
    w.add_block_count(layers)
    w.add_context_length(4096)
    w.add_embedding_length(GH)
    w.add_feed_forward_length(widths if small else GINTER)
    w.add_head_count(4)
    w.add_head_count_kv([2, 1] * (layers // 2))
    w.add_key_length(128)
    w.add_value_length(128)
    w.add_key_length_swa(64)
    w.add_value_length_swa(64)
    w.add_sliding_window(128)
    w.add_sliding_window_pattern([True, False] * (layers // 2))
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0)
    w.add_rope_freq_base_swa(10000.0)
    w.add_rope_dimension_count(128)
    w.add_rope_dimension_count_swa(64)
    w.add_final_logit_softcapping(30.0)
    w.add_embedding_length_per_layer_input(ple)
    w.add_shared_kv_layers(shared)
    if moe:
        w.add_expert_count(GEXPERTS)
        w.add_expert_used_count(2)
        w.add_expert_feed_forward_length(GMOE)
    g.f32("rope_freqs.weight", np.array([1.0] * 16 + [1e30] * 48, np.float32))

    def randn(*shape, scale=0.05):
        return bf16_round(rng.standard_normal(shape).astype(np.float32) * scale)

    def vector(gname, hf, n, scale=0.05):
        g.f32(gname, randn(n, scale=scale))
        put_twin(t, hf, g, gname)

    def nvfp4(gname, hf, rows, k, divisor, input_divisor):
        """compressed-tensors origin: GGUF scale = 1/global (binary32), twin divisor = 1/scale."""
        codes = rng.integers(0, 256, (rows, k // 2), dtype=np.uint8)
        scales = rng.integers(0, 0x7F, (rows, k // 16), dtype=np.uint8)
        g.nvfp4(gname, codes, scales)
        stem = gname[: -len(".weight")]
        scale2 = np.float32(1.0) / np.float32(divisor)
        input_scale = np.float32(1.0) / np.float32(input_divisor)
        g.f32(stem + ".scale", np.array([scale2], np.float32))
        g.f32(stem + ".input_scale", np.array([input_scale], np.float32))
        prefix = hf[: -len(".weight")]
        t.put(prefix + ".weight_packed", "U8", codes)
        t.put(prefix + ".weight_scale", "F8_E4M3", scales)
        back = np.float32(1.0) / scale2
        t.f32(prefix + ".weight_global_scale", np.array([back], np.float32))
        t.f32(prefix + ".input_global_scale", np.array([np.float32(1.0) / input_scale], np.float32))
        stats["divisor words"] += 1
        stats["divisor words changed"] += int(back != np.float32(divisor))

    def mxfp4(gname, hf, rows, k):
        codes = rng.integers(0, 256, (rows, k // 2), dtype=np.uint8)
        exps = rng.integers(108, 121, (rows, k // 32), dtype=np.uint8)
        exps[0, 0] = 120
        g.mxfp4(gname, codes, exps)
        prefix = hf[: -len(".weight")]
        t.put(prefix + ".weight_packed", "U8", codes)
        t.put(prefix + ".weight_scale", "U8", exps)

    def matrix(gname, hf, rows, k, divisor=0.5, input_divisor=0.8):
        if mode == "nvfp4":
            nvfp4(gname, hf, rows, k, divisor, input_divisor)
        elif mode == "mxfp4":
            mxfp4(gname, hf, rows, k)
        elif mode == "iq4" and k % 256 == 0:
            g.quant(gname, (rows, k), Q.IQ4_XS)
            put_twin(t, hf, g, gname)
        elif mode == "iq4":
            g.quant(gname, (rows, k), Q.Q5_0)
            put_twin(t, hf, g, gname)
        else:
            g.bf16(gname, randn(rows, k))
            put_twin(t, hf, g, gname)

    if mode in ("iq4", "tables"):
        g.quant("token_embd.weight", (VOCAB, GH), Q.Q6_K)
    else:
        g.bf16("token_embd.weight", randn(VOCAB, GH))
    put_twin(t, "model.embed_tokens.weight", g, "token_embd.weight")
    vector("output_norm.weight", "model.norm.weight", GH)
    if small:
        if mode == "tables":
            g.quant("per_layer_token_embd.weight", (VOCAB, layers * ple), Q.Q8_0)
        else:
            g.bf16("per_layer_token_embd.weight", randn(VOCAB, layers * ple))
        put_twin(t, "model.embed_tokens_per_layer.weight", g, "per_layer_token_embd.weight")
        g.bf16("per_layer_model_proj.weight", randn(layers * ple, GH))
        put_twin(t, "model.per_layer_model_projection.weight", g, "per_layer_model_proj.weight")
        vector("per_layer_proj_norm.weight", "model.per_layer_projection_norm.weight", ple)
    for layer, (kind, d, kvh) in enumerate((("sliding", 64, 2), ("full", 128, 1)) * (layers // 2)):
        s, p = f"blk.{layer}.", f"model.layers.{layer}."
        owns = layer < layers - shared
        roles = [("attn_q", "q_proj", 4 * d)] + ([("attn_k", "k_proj", kvh * d)] if owns else [])
        if kind == "sliding" and owns:
            roles.append(("attn_v", "v_proj", kvh * d))
        for index, (gname, role, rows) in enumerate(roles):
            matrix(s + gname + ".weight", p + "self_attn." + role + ".weight", rows, GH,
                   0.5 if layer == 0 else 0.5 + 0.25 * index, 0.7 + 0.1 * index)
        matrix(s + "attn_output.weight", p + "self_attn.o_proj.weight", GH, 4 * d, 0.9, 1.1)
        vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", d)
        if owns:
            vector(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", d)
        for gname, hf in (("attn_norm", "input_layernorm"), ("post_attention_norm", "post_attention_layernorm"),
                          ("ffn_norm", "pre_feedforward_layernorm"), ("post_ffw_norm", "post_feedforward_layernorm")):
            vector(s + gname + ".weight", p + hf + ".weight", GH)
        vector(s + "layer_output_scale.weight", p + "layer_scalar", 1)
        ff = widths[layer]
        matrix(s + "ffn_gate.weight", p + "mlp.gate_proj.weight", ff, GH, 0.3, 0.8)
        matrix(s + "ffn_up.weight", p + "mlp.up_proj.weight", ff, GH, 0.35, 0.8)
        matrix(s + "ffn_down.weight", p + "mlp.down_proj.weight", GH, ff, 0.45, 0.6)
        if small:
            matrix(s + "inp_gate.weight", p + "per_layer_input_gate.weight", ple, GH)
            matrix(s + "proj.weight", p + "per_layer_projection.weight", GH, ple)
            vector(s + "post_norm.weight", p + "post_per_layer_input_norm.weight", GH)
        if not moe:
            continue
        for gname, hf in (("pre_ffw_norm_2", "pre_feedforward_layernorm_2"),
                          ("post_ffw_norm_1", "post_feedforward_layernorm_1"),
                          ("post_ffw_norm_2", "post_feedforward_layernorm_2")):
            vector(s + gname + ".weight", p + hf + ".weight", GH)
        g.bf16(s + "ffn_gate_inp.weight", randn(GEXPERTS, GH))
        put_twin(t, p + "router.proj.weight", g, s + "ffn_gate_inp.weight")
        vector(s + "ffn_gate_inp.scale", p + "router.scale", GH)
        per_expert = bf16_round(rng.uniform(0.5, 1.5, GEXPERTS).astype(np.float32))
        # Experts: NVFP4 (Model Optimizer origin) with per-expert global scales in [E] vectors,
        # per_expert_scale folded into the down scales, or random quant / BF16 banks.
        banks = {}
        for bank, (rows, k) in (("gate", (GMOE, GH)), ("up", (GMOE, GH)), ("down", (GH, GMOE))):
            if mode == "nvfp4":
                codes = rng.integers(0, 256, (GEXPERTS, rows, k // 2), dtype=np.uint8)
                scales = rng.integers(0, 0x7F, (GEXPERTS, rows, k // 16), dtype=np.uint8)
                scale2 = rng.uniform(0.001, 0.01, GEXPERTS).astype(np.float32)
                banks[bank] = (codes, scales, scale2, rng.uniform(0.5, 2.0, GEXPERTS).astype(np.float32))
            elif mode == "iq4" and k % 256 == 0:
                banks[bank] = ("quant", Q.IQ4_XS, (GEXPERTS, rows, k))
            elif mode == "iq4":
                banks[bank] = ("quant", Q.Q4_0, (GEXPERTS, rows, k))
            else:
                banks[bank] = ("bf16", randn(GEXPERTS, rows, k))
        for bank in ("gate", "up", "down"):
            gname = s + f"ffn_{bank}_exps.weight"
            entry = banks[bank]
            if mode == "nvfp4":
                codes, scales, scale2, input_scale = entry
                if bank == "down":
                    scale2 = (scale2 * per_expert).astype(np.float32)  # llama.cpp's fold
                rows = codes.shape[1]
                # One stacked [E, rows, k] bank of ggml NVFP4 blocks.
                nibbles = np.stack((codes & 15, codes >> 4), -1).reshape(GEXPERTS, rows, -1, 16)
                qs = (nibbles[..., :8] | (nibbles[..., 8:] << 4)).astype(np.uint8)
                d = (scales & 0x7F).reshape(GEXPERTS, rows, -1, 4)
                raw = np.concatenate([d, qs.reshape(GEXPERTS, rows, -1, 32)], -1).reshape(GEXPERTS, rows, -1)
                g.writer.add_tensor(gname, raw, raw_dtype=Q.NVFP4)
                g.f32(gname[: -len(".weight")] + ".scale", scale2)
                g.f32(gname[: -len(".weight")] + ".input_scale", input_scale)
                for e in range(GEXPERTS):
                    prefix = f"{p}experts.{e}.{bank}_proj"
                    t.put(prefix + ".weight", "U8", codes[e])
                    t.put(prefix + ".weight_scale", "F8_E4M3", scales[e])
                    t.f32(prefix + ".weight_scale_2", np.array(scale2[e], np.float32))
                    t.f32(prefix + ".input_scale", np.array(input_scale[e], np.float32))
            elif entry[0] == "quant":
                g.quant(gname, entry[2], entry[1])
                for e in range(GEXPERTS):
                    t.f32(f"{p}experts.{e}.{bank}_proj.weight", g.values[gname][e])
            else:
                g.bf16(gname, entry[1])
                for e in range(GEXPERTS):
                    t.put(f"{p}experts.{e}.{bank}_proj.weight", "BF16", g.words[gname][e])
        if mode == "nvfp4":
            t.f32(p + "router.per_expert_scale", np.ones(GEXPERTS, np.float32))
        else:
            g.f32(s + "ffn_down_exps.scale", per_expert)
            t.f32(p + "router.per_expert_scale", per_expert)
    expected = tokenizer_metadata(g, "gemma4") if rebuilt else None
    g.save(gdir, files=not rebuilt)
    text = {
        "model_type": "gemma4_text", "hidden_size": GH, "vocab_size": VOCAB,
        "num_hidden_layers": layers, "max_position_embeddings": 4096, "tie_word_embeddings": True,
        "rms_norm_eps": 1e-06, "layer_types": ["sliding_attention", "full_attention"] * (layers // 2),
        "num_attention_heads": 4, "attention_k_eq_v": True, "sliding_window": 128,
        "head_dim": 64, "num_key_value_heads": 2, "global_head_dim": 128,
        "num_global_key_value_heads": 1, "rope_parameters": rope(),
        "final_logit_softcapping": 30.0, "intermediate_size": GINTER,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size_per_layer_input": ple,
        "num_kv_shared_layers": shared, "attention_bias": False, "enable_moe_block": moe,
    }
    if small:
        text.update({"use_double_wide_mlp": True, "vocab_size_per_layer_input": VOCAB})
    if moe:
        text.update({"num_experts": GEXPERTS, "top_k_experts": 2, "moe_intermediate_size": GMOE})
    t.save(work / "twin", {"architectures": ["Gemma4ForCausalLM"], **text})
    for role, text_value in (expected or {}).items():
        (work / "twin" / role).write_text(text_value, encoding="utf-8")
    return gdir / "model.gguf", work / "twin", stats


def gemma_assistant_case(work: Path, seed: int):
    """Gemma 4's MTP drafter as llama.cpp writes it (gemma4-assistant, e.g. unsloth's
    mtp-gemma-4-31B-it.gguf): Q8_0 projections, F32 norms, four query-only layers reading the
    target's keys and values (three sliding, one full with proportional RoPE), the projections
    to and from the target's width (nextn.pre_projection / nextn.post_projection), its own token
    table as the tied head. The twin is transformers' Gemma4AssistantForCausalLM."""
    rng = np.random.default_rng(seed)
    gdir = work / "gguf"
    gdir.mkdir(parents=True)
    arch = "gemma4-assistant"
    g = G(gdir / "model.gguf", arch, rng)
    g.q8_ggml = True
    t = Hf()
    layers, heads, h, backbone, inter = 4, 4, 128, GH, 256
    kinds = [("sliding", 64, 2), ("sliding", 64, 2), ("sliding", 64, 2), ("full", 128, 1)]
    w = g.writer
    w.add_block_count(layers)
    w.add_context_length(4096)
    w.add_embedding_length(h)
    w.add_uint32(f"{arch}.embedding_length_out", backbone)
    w.add_feed_forward_length(inter)
    w.add_head_count(heads)
    w.add_head_count_kv([kv for _, _, kv in kinds])
    w.add_key_length(128)
    w.add_value_length(128)
    w.add_key_length_swa(64)
    w.add_value_length_swa(64)
    w.add_sliding_window(128)
    w.add_sliding_window_pattern([kind == "sliding" for kind, _, _ in kinds])
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0)
    w.add_rope_freq_base_swa(10000.0)
    w.add_rope_dimension_count(128)
    w.add_rope_dimension_count_swa(64)
    w.add_embedding_length_per_layer_input(0)
    w.add_shared_kv_layers(layers)
    w.add_uint32(f"{arch}.nextn_predict_layers", layers)
    g.f32("rope_freqs.weight", np.array([1.0] * 16 + [1e30] * 48, np.float32))

    def vector(gname, hf, n):
        g.f32(gname, bf16_round(rng.standard_normal(n).astype(np.float32) * 0.05))
        put_twin(t, hf, g, gname)

    def matrix(gname, hf, rows, k):
        g.quant(gname, (rows, k), Q.Q8_0)
        put_twin(t, hf, g, gname)

    matrix("token_embd.weight", "model.embed_tokens.weight", VOCAB, h)
    vector("output_norm.weight", "model.norm.weight", h)
    matrix("nextn.pre_projection.weight", "pre_projection.weight", h, 2 * backbone)
    matrix("nextn.post_projection.weight", "post_projection.weight", backbone, h)
    for layer, (kind, d, _) in enumerate(kinds):
        s, p = f"blk.{layer}.", f"model.layers.{layer}."
        matrix(s + "attn_q.weight", p + "self_attn.q_proj.weight", heads * d, h)
        matrix(s + "attn_output.weight", p + "self_attn.o_proj.weight", h, heads * d)
        vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", d)
        for gname, hf in (("attn_norm", "input_layernorm"), ("post_attention_norm", "post_attention_layernorm"),
                          ("ffn_norm", "pre_feedforward_layernorm"), ("post_ffw_norm", "post_feedforward_layernorm")):
            vector(s + gname + ".weight", p + hf + ".weight", h)
        vector(s + "layer_output_scale.weight", p + "layer_scalar", 1)
        matrix(s + "ffn_gate.weight", p + "mlp.gate_proj.weight", inter, h)
        matrix(s + "ffn_up.weight", p + "mlp.up_proj.weight", inter, h)
        matrix(s + "ffn_down.weight", p + "mlp.down_proj.weight", h, inter)
    g.save(gdir)
    text = {
        "model_type": "gemma4_text", "hidden_size": h, "vocab_size": VOCAB, "num_hidden_layers": layers,
        "max_position_embeddings": 4096, "tie_word_embeddings": True, "rms_norm_eps": 1e-06,
        "layer_types": [kind + "_attention" for kind, _, _ in kinds], "num_attention_heads": heads,
        "sliding_window": 128, "head_dim": 64, "num_key_value_heads": 2, "global_head_dim": 128,
        "num_global_key_value_heads": 1, "rope_parameters": rope(), "intermediate_size": inter,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": layers, "enable_moe_block": False,
    }
    t.save(work / "twin", {"architectures": ["Gemma4AssistantForCausalLM"], "model_type": "gemma4_assistant",
                           "backbone_hidden_size": backbone, "use_ordered_embeddings": False,
                           "tie_word_embeddings": True, "text_config": text})
    return gdir / "model.gguf", work / "twin", {}


# ------------------------------------------------------------------------------------------------


def compare(name, gguf_out: Path, twin_out: Path, kept_a8: bool = False):
    """With `kept_a8`, the GGUF artifact's kept blocks allow int8 activations (llama.cpp runs them
    with Q8_1 activations) where the recipe's twin says A16Only; the uses must match otherwise."""
    errors = []
    gv, gobj, gdata, goff = directory(gguf_out)
    tv, tobj, tdata, toff = directory(twin_out)
    if set(gobj) != set(tobj):
        errors.append(f"{name}: object sets differ: {sorted(set(gobj) ^ set(tobj))[:5]}")
    for oid, o in tobj.items():
        other = gobj.get(oid)
        if other is None:
            continue
        if (o.get("shape"), o.get("format"), o["bytes"]) != (other.get("shape"), other.get("format"), other["bytes"]):
            errors.append(f"{name}: {oid} differs in shape/format/size: {o.get('format')} vs {other.get('format')}")
            continue
        a = gdata[goff + other["offset"]:goff + other["offset"] + other["bytes"]]
        b = tdata[toff + o["offset"]:toff + o["offset"] + o["bytes"]]
        if a != b:
            errors.append(f"{name}: {oid} ({o.get('format')}) bytes differ")
    if kept_a8:
        uses = json.dumps(gv["uses"])
        if '"AllowA8"' not in uses:
            errors.append(f"{name}: no kept matrix allows int8 activations")
        gv = {**gv, "uses": json.loads(uses.replace('"AllowA8"', '"A16Only"'))}
    for key in ("components", "bindings", "uses"):
        if gv[key] != tv[key]:
            errors.append(f"{name}: {key} differ")
    return errors


def muse_assistant_case(root: Path, native: Path, repository: Path) -> list[str]:
    """Meta's Muse Glimmer assistant (MuseGlimmerAssistantModel) as published: the DFlash drafter
    of dflash_case (BF16) with its config in transformers' form - no vocab_size and no target depth
    (it owns no table), target_layer_ids / mask_token_id / block_size at the top level,
    rope_parameters - and no tokenizer beside it (the target's, by --resource). Its artifact must
    be the DFlashDraftModel twin's: the same objects, bindings, uses and resources; the text
    config differs only in the architecture and the vocabulary it does not name."""
    work = root / "muse-assistant"
    _, twin, _ = dflash_case(work, None, 60)
    muse = work / "muse"
    shutil.copytree(twin, muse)
    config = json.loads((twin / "config.json").read_text(encoding="utf-8"))
    draft = config.pop("dflash_config")
    for key in ("vocab_size", "rope_theta"):
        config.pop(key)
    config.update({"architectures": ["MuseGlimmerAssistantModel"], "model_type": "muse_glimmer_assistant",
                   "target_layer_ids": draft["target_layer_ids"], "mask_token_id": draft["mask_token_id"],
                   "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}})
    (muse / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    roles = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")
    for role in roles:
        (muse / role).unlink(missing_ok=True)
    outs, errors = {}, []
    for source, model, extra in (("twin", twin, []), ("muse", muse, [f"--resource={r}={twin / r}" for r in roles
                                                                        if (twin / r).exists()])):
        out = work / f"{source}.ninferno"
        result = subprocess.run([str(native), f"--model={model}", "--recipe=dflash_published", f"--out={out}", *extra],
                                cwd=repository, capture_output=True, text=True)
        if result.returncode:
            errors.append(f"muse-assistant: converting the {source} failed\n{result.stderr[-2000:]}")
        outs[source] = out
    if not errors:
        mv, mobj, mdata, moff = directory(outs["muse"])
        tv, tobj, tdata, toff = directory(outs["twin"])
        if set(mobj) != set(tobj):
            errors.append("muse-assistant: object sets differ")
        for oid, o in tobj.items():
            other = mobj.get(oid)
            if other is None or (o.get("format"), o["bytes"]) != (other.get("format"), other["bytes"]) or \
                    tdata[toff + o["offset"]:toff + o["offset"] + o["bytes"]] != \
                    mdata[moff + other["offset"]:moff + other["offset"] + other["bytes"]]:
                errors.append(f"muse-assistant: {oid} differs")
        for key in ("bindings", "uses"):
            if mv[key] != tv[key]:
                errors.append(f"muse-assistant: {key} differ")
        mc, tc = mv["components"], tv["components"]
        mt, tt = mc["text"].pop("config"), tc["text"].pop("config")
        if mc != tc:
            errors.append("muse-assistant: components (resources) differ")
        if mt.get("architectures") != ["MuseGlimmerAssistantModel"] or "vocab_size" in mt:
            errors.append(f"muse-assistant: unexpected config {mt}")
        for key in ("architectures", "vocab_size"):
            mt.pop(key, None)
            tt.pop(key, None)
        if mt != tt:
            errors.append(f"muse-assistant: configs differ beyond the architecture and vocabulary: {mt} vs {tt}")
    print(f"muse-assistant: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def yarn_reference(base: float, dim: int, factor: float, original: int, fast: float, slow: float):
    """transformers' _compute_yarn_parameters (truncate false) in float64: (inv_freq, attention
    scaling)."""
    half = np.arange(0, dim, 2, dtype=np.float64) / dim
    pos = base ** half
    extrapolation, interpolation = 1.0 / pos, 1.0 / (factor * pos)

    def correction(rotations):
        return dim * np.log(original / (rotations * 2 * np.pi)) / (2 * np.log(base))

    low, high = max(correction(fast), 0.0), min(correction(slow), dim - 1.0)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    keep = 1.0 - ramp
    return interpolation * (1.0 - keep) + extrapolation * keep, 0.1 * np.log(factor) + 1.0


def zlab_dflash_case(root: Path, native: Path, repository: Path) -> list[str]:
    """z-lab's gpt-oss DFlash drafter as published (DFlashDraftModel, model_type qwen3): full
    layers only (sliding_window null), q/k/v/o biases, YaRN rope, no soft cap. dflash_published
    keeps every weight as published (the biases as attention/{query,key,value,output}_bias, BF16
    word for word) and resolves the rope into the frequency table transformers computes; the config
    records attention_bias and no sliding window. A Gemma 4 drafter (with its soft cap) keeps the
    Gemma 4 recipe."""
    work = root / "zlab-dflash"
    work.mkdir(parents=True)
    rng = np.random.default_rng(71)
    h, heads, kv, d, inter, layers, taps = 256, 4, 2, 64, 384, 2, [1, 3, 5]
    t = Hf()
    words = {}

    def put(name, *shape, scale=0.05, offset=0.0):
        values = rng.standard_normal(shape).astype(np.float32) * np.float32(scale) + np.float32(offset)
        t.bf16(name, values)
        words[name] = bf16_words(values).tobytes()

    put("fc.weight", h, len(taps) * h)
    put("hidden_norm.weight", h, scale=0.2, offset=1.0)
    put("norm.weight", h, scale=0.2, offset=1.0)
    for layer in range(layers):
        q = f"layers.{layer}."
        put(q + "input_layernorm.weight", h, scale=0.2, offset=1.0)
        put(q + "post_attention_layernorm.weight", h, scale=0.2, offset=1.0)
        for role, rows in (("q_proj", heads * d), ("k_proj", kv * d), ("v_proj", kv * d)):
            put(q + f"self_attn.{role}.weight", rows, h)
            put(q + f"self_attn.{role}.bias", rows, scale=0.5)
        put(q + "self_attn.o_proj.weight", h, heads * d)
        put(q + "self_attn.o_proj.bias", h, scale=0.5)
        put(q + "self_attn.q_norm.weight", d, scale=0.2, offset=1.0)
        put(q + "self_attn.k_norm.weight", d, scale=0.2, offset=1.0)
        put(q + "mlp.gate_proj.weight", inter, h)
        put(q + "mlp.up_proj.weight", inter, h)
        put(q + "mlp.down_proj.weight", h, inter)
    yarn = {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0, "original_max_position_embeddings": 4096,
            "rope_type": "yarn", "truncate": False}
    config = {
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3", "attention_bias": True,
        "attention_dropout": 0.0, "block_size": 8, "hidden_act": "silu", "hidden_size": h, "head_dim": d,
        "intermediate_size": inter, "layer_types": ["full_attention"] * layers,
        "max_position_embeddings": 131072, "max_window_layers": layers, "num_attention_heads": heads,
        "num_hidden_layers": layers, "num_key_value_heads": kv, "num_target_layers": 8,
        "rms_norm_eps": 1e-05, "rope_scaling": yarn, "rope_theta": 150000, "sliding_window": None,
        "use_sliding_window": False, "tie_word_embeddings": False, "vocab_size": VOCAB,
        "dflash_config": {"mask_token_id": VOCAB - 2, "target_layer_ids": taps},
    }
    source = work / "drafter"
    t.save(source, config)
    out = work / "zlab.ninferno"
    result = subprocess.run([str(native), f"--model={source}", f"--out={out}"], cwd=repository,
                            capture_output=True, text=True)
    if result.returncode:
        return [f"zlab-dflash: conversion failed\n{result.stderr[-2000:]}"]
    errors = []
    value, objects, data, offset = directory(out)
    report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
    if "dflash_published" not in json.dumps(report):
        errors.append("zlab-dflash: the default recipe is not dflash_published")
    text = value["components"]["text"]["config"]
    if text.get("model_type") != "dflash" or text.get("attention_bias") is not True or "sliding_window" in text:
        errors.append(f"zlab-dflash: config {dict((k, text.get(k)) for k in ('model_type', 'attention_bias', 'sliding_window'))}")
    freq, scaling = yarn_reference(150000.0, d, 32.0, 4096, 32.0, 1.0)
    rope = text.get("rope") or {}
    got = np.asarray(rope.get("inv_freq", []), np.float64)
    if got.shape != freq.shape or not np.allclose(got, freq, rtol=2e-6, atol=0.0):
        errors.append(f"zlab-dflash: inv_freq {got[:4]} vs transformers' {freq[:4]}")
    if abs(rope.get("attention_scaling", 0.0) - scaling) > 1e-6 or rope.get("rotary_dim") != d:
        errors.append(f"zlab-dflash: rope {dict((k, v) for k, v in rope.items() if k != 'inv_freq')}, "
                      f"expected attention_scaling {scaling}")
    # Object ids are opaque (weight/NNNNNN): the report's methods name the parameters each one holds.
    holder = {p: m["object"] for m in report["methods"] for p in m["parameters"]}
    for layer in range(layers):
        for field, role in (("q", "query"), ("k", "key"), ("v", "value"), ("o", "output")):
            source_name = f"layers.{layer}.self_attn.{field}_proj.bias"
            found = [objects[oid] for p, oid in holder.items() if p.endswith(f"layers/{layer}/attention/{role}_bias")]
            if len(found) != 1:
                errors.append(f"zlab-dflash: {len(found)} objects for layer {layer} {role}_bias")
                continue
            o = found[0]
            if data[offset + o["offset"]:offset + o["offset"] + o["bytes"]] != words[source_name]:
                errors.append(f"zlab-dflash: layer {layer} {role}_bias is not the published BF16 words")
    print(f"zlab-dflash: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def generation_fallback_case(root: Path, native: Path, repository: Path) -> list[str]:
    """A safetensors checkpoint published without generation_config.json (Qwen3.5-0.8B is one):
    its end-of-sequence ids come from config.json and tokenizer_config.json's eos_token."""
    work = root / "generation-fallback"
    _, twin, _ = qwen_case(work, False, None, 13)
    (twin / "generation_config.json").unlink()
    config = json.loads((twin / "config.json").read_text(encoding="utf-8"))
    config["eos_token_id"] = 3
    (twin / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokenizer = json.loads((twin / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer["eos_token"] = "tok4"
    (twin / "tokenizer_config.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    out = work / "fallback.ninferno"
    result = subprocess.run([str(native), f"--model={twin}", "--recipe=qwen3_6_27b", "--maintainer-quantize",
                             f"--out={out}"],
                            cwd=repository, capture_output=True, text=True)
    if result.returncode:
        return [f"generation-fallback: conversion failed\n{result.stderr[-2000:]}"]
    value, objects, data, offset = directory(out)
    resource = objects.get("resource/text/generation_config.json")
    if resource is None:
        return ["generation-fallback: no generation_config.json resource"]
    body = json.loads(data[offset + resource["offset"]:offset + resource["offset"] + resource["bytes"]])
    report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
    errors = []
    if body != {"eos_token_id": [3, 4]}:
        errors.append(f"generation-fallback: derived {body}, expected eos [3, 4]")
    if not any("generation_config.json" in n for n in report.get("resource_notes", [])):
        errors.append("generation-fallback: the report does not note the derived resource")
    print(f"generation-fallback: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def plain_decode_cases(root: Path, native: Path, repository: Path) -> list[str]:
    """A tensor stored in a quantized ggml type and read as values is never decoded silently,
    whatever its name: blk.N.ssm_dt.bias (no .weight suffix) as Q8_0 is refused without
    --maintainer-quantize, naming the decode; with it the report lists the decode. A decode a
    source's own transform needs (Gemma's embedding scale folded into a Q6_K table) converts and
    the report says why."""
    errors = []

    def convert(model, out, *extra):
        return subprocess.run([str(native), f"--model={model}", f"--out={out}", "--components=text",
                               *extra], cwd=repository, capture_output=True, text=True)

    work = root / "plain-decode-dt-bias"
    with qwen_geometry(2, 32):
        gguf_file, _, _ = qwen_case(work, False, None, 60, dt_bias=Q.Q8_0)
    refused = convert(gguf_file, work / "refused.ninferno", "--recipe=qwen3_6_27b")
    if refused.returncode == 0 or "gguf q8_0" not in refused.stderr or "dequantized" not in refused.stderr:
        errors.append("plain-decode: a Q8_0 ssm_dt.bias was decoded without --maintainer-quantize\n"
                      + refused.stderr[-1500:])
    out = work / "forced.ninferno"
    forced = convert(gguf_file, out, "--recipe=qwen3_6_27b", "--maintainer-quantize")
    if forced.returncode:
        errors.append(f"plain-decode: the forced conversion failed\n{forced.stderr[-1500:]}")
    else:
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        notes = [n for n in report.get("native_notes") or [] if n["encoding"].startswith("gguf q8_0")]
        objects = {o for n in notes for o in n["objects"]}
        dt = [m["object"] for m in report["methods"] if any("dt_bias" in s for s in m["sources"])]
        if not notes or not dt or not set(dt) <= objects:
            errors.append(f"plain-decode: the report does not list the decoded ssm_dt.bias: {notes}")

    work = root / "plain-decode-embedding-scale"
    gguf_file, _, _ = gemma_case(work, False, "iq4", 61)
    out = work / "scaled.ninferno"
    scaled = convert(gguf_file, out, "--recipe=gemma4_31b_nvfp4", "--maintainer-quantize")
    if scaled.returncode:
        errors.append(f"plain-decode: the Gemma conversion failed\n{scaled.stderr[-1500:]}")
    else:
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        notes = [n for n in report.get("native_notes") or []
                 if n["encoding"].startswith("gguf q6_k") and "to fold the constant" in n["handling"]]
        if not notes or "decoded on purpose: gguf q6_k" not in scaled.stdout + scaled.stderr:
            errors.append("plain-decode: the Q6_K embedding decoded to fold the embedding scale is not "
                          f"reported: {report.get('native_notes')}")
    print(f"plain-decode: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def selection_cases(root: Path, native: Path, repository: Path) -> list[str]:
    """Without --recipe the converter picks the official recipe for the checkpoint: the same
    artifact as naming it; an unquantized Qwen3.5 GGUF runs as BF16 (qwen3_5_bf16); a Qwen3.5
    NVFP4 GGUF takes the _nvfp4 recipe with itself as the quantized source; a GGUF whose types
    have no native execution is refused (never requantized); an architecture without a recipe
    is refused naming the supported ones."""
    errors = []

    def convert(model, out, *extra):
        return subprocess.run([str(native), f"--model={model}", f"--out={out}", *extra],
                              cwd=repository, capture_output=True, text=True)

    work = root / "auto-recipe"
    gguf_file, _, _ = qwen_case(work, False, None, 14)
    auto, named = work / "auto.ninferno", work / "named.ninferno"
    first, second = convert(gguf_file, auto), convert(gguf_file, named, "--recipe=qwen3_5_bf16")
    if first.returncode or second.returncode:
        errors.append(f"auto-recipe: conversion failed\n{first.stderr[-1500:]}{second.stderr[-1500:]}")
    else:
        if "recipe qwen3_5_bf16 (chosen for the checkpoint)" not in first.stdout:
            errors.append("auto-recipe: the chosen recipe is not qwen3_5_bf16")
        errors += compare("auto-recipe", auto, named)
    refused = convert(gguf_file, work / "q8.ninferno", "--recipe=qwen3_8_27b")
    if refused.returncode == 0 or "--maintainer-quantize" not in refused.stderr:
        errors.append("auto-recipe: qwen3_8_27b quantized an unquantized GGUF without --maintainer-quantize")

    # The official NVFP4 layout (full attention and MTP published unquantized, GDN and MLP in
    # NVFP4) converts as published. NVFP4 where the recipe has no NVFP4 site (the full attention
    # here) would be dequantized: refused.
    work = root / "auto-nvfp4-attention"
    gguf_file, _, _ = qwen_case(work, False, {"attention": Q.NVFP4, "gdn": Q.NVFP4, "mlp": Q.NVFP4}, 15)
    refused = convert(gguf_file, work / "refused.ninferno", "--components=text")
    if (refused.returncode == 0 or "gguf nvfp4: dequantized to bf16" not in refused.stderr
            or "--maintainer-quantize" not in refused.stderr):
        errors.append(f"auto-nvfp4-attention: NVFP4 attention was not refused: {refused.stderr[-600:]}")
    work = root / "auto-nvfp4"
    gguf_file, _, _ = qwen_case(work, False, {"gdn": Q.NVFP4, "mlp": Q.NVFP4}, 15)
    out = work / "auto.ninferno"
    result = convert(gguf_file, out, "--components=text")
    if result.returncode:
        errors.append(f"auto-nvfp4: conversion failed\n{result.stderr[-2000:]}")
    else:
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        kept = [n for n in report.get("native_notes", []) if n["handling"] == "kept as nvfp4"]
        if "recipe qwen3_6_27b_nvfp4 (chosen" not in result.stdout or not kept or not all(n["lossless"] for n in kept):
            errors.append(f"auto-nvfp4: expected qwen3_6_27b_nvfp4 keeping NVFP4, got {result.stdout[:300]}")

    # NVFP4 mixed with other types (as imatrix NVFP4 GGUFs are): the NVFP4 recipe cannot plan,
    # so the plain recipe is chosen, saying why; it would dequantize, so the conversion is refused
    # naming the encodings; --maintainer-quantize gives what naming the recipe gives.
    work = root / "auto-mixed-nvfp4"
    gguf_file, _, _ = qwen_case(work, False, {"attention": Q.NVFP4, "gdn": Q.Q8_0, "mlp": Q.NVFP4}, 17)
    auto, named = work / "auto.ninferno", work / "named.ninferno"
    refused = convert(gguf_file, work / "refused.ninferno")
    first = convert(gguf_file, auto, "--maintainer-quantize")
    second = convert(gguf_file, named, "--recipe=qwen3_5_bf16", "--maintainer-quantize")
    if (refused.returncode == 0 or "recipe note: qwen3_6_27b_nvfp4 does not fit this GGUF" not in refused.stdout
            or "would not run" not in refused.stderr or "gguf" not in refused.stderr):
        errors.append(f"auto-mixed-nvfp4: expected a refusal naming the GGUF encodings, got "
                      f"{refused.stdout[-400:]}{refused.stderr[-600:]}")
    if first.returncode or second.returncode:
        errors.append(f"auto-mixed-nvfp4: conversion failed\n{first.stderr[-1500:]}{second.stderr[-1500:]}")
    else:
        if "recipe qwen3_5_bf16 (chosen for the checkpoint)" not in first.stdout:
            errors.append(f"auto-mixed-nvfp4: expected qwen3_5_bf16, got {first.stdout[:400]}")
        errors += compare("auto-mixed-nvfp4", auto, named)

    # A MoE GGUF with MXFP4 routed experts (llama.cpp's MXFP4_MOE) runs as published: the
    # experts keep their MXFP4 words (their exponents span more than NVFP4's one divisor can
    # hold), the Q8_0 tensors beside them word for word.
    work = root / "auto-moe-mxfp4"
    gguf_file, _, _ = qwen_case(work, True, {"attention": Q.Q8_0, "gdn": Q.Q8_0, "embedding": Q.Q8_0,
                                             "experts": Q.Q8_0, "experts_gate_up": Q.MXFP4,
                                             "experts_down": Q.MXFP4}, 19, q8_ggml=True)
    out = work / "auto.ninferno"
    result = convert(gguf_file, out, "--components=text")
    if result.returncode:
        errors.append(f"auto-moe-mxfp4: conversion failed\n{result.stdout[-800:]}{result.stderr[-2000:]}")
    else:
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        notes = report.get("native_notes", [])
        kept = [n for n in notes if n["handling"] == "kept as mxfp4"]
        if ("recipe qwen3_5_published (chosen" not in result.stdout or not kept
                or not all(n["lossless"] for n in kept) or any("nvfp4" in n["handling"] for n in notes)):
            errors.append(f"auto-moe-mxfp4: expected qwen3_5_published keeping the MXFP4 experts as mxfp4, "
                          f"got {result.stdout[:400]} {report.get('native_notes')}")

    work = root / "unsupported"
    work.mkdir()
    writer = gguf.GGUFWriter(work / "falcon.gguf", "falcon")
    writer.add_tensor("token_embd.weight", np.zeros((4, 32), np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    result = convert(work / "falcon.gguf", work / "x.ninferno")
    if result.returncode == 0 or "supported architectures" not in result.stderr or "qwen35" not in result.stderr:
        errors.append(f"unsupported: expected a refusal naming the supported architectures, got {result.stderr[-500:]}")
    print(f"recipe selection: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def gemma4_small_tables_case(root: Path, native: Path, repository: Path) -> list[str]:
    """Gemma 4 E2B/E4B tables as Google's QAT GGUFs publish them (a Q6_K token table, a quantized
    per-layer table) under gemma4_published: both kept word for word - the token table without
    sqrt(H) folded in (embedding_scale_at_run_time: the engine scales the gathered rows), since
    folding would decode and quantize it again, the tied head bound to the same object - and the
    shared-KV layers bind no key or value."""
    errors = []
    work = root / "gemma4small-tables"
    gguf_file, _, _ = gemma_case(work, False, "tables", 19, small=True)
    out = work / "published.ninferno"
    result = subprocess.run([str(native), f"--model={gguf_file}", "--recipe=gemma4_published",
                             "--components=text", f"--out={out}"],
                            cwd=repository, capture_output=True, text=True)
    if result.returncode:
        errors.append(f"gemma4small-tables: conversion failed\n{result.stderr[-2000:]}")
    else:
        value, objects, _, _ = directory(out)

        def formats(name):
            binding = value["bindings"].get(name)
            if binding is None:
                return None
            ids = [binding["object"]] if "object" in binding else [p["object"] for p in binding["parts"]]
            return {objects[i]["format"] for i in ids}

        config = value["components"]["text"]["config"]
        want = {"text/token_embedding": {"ggml_q6_k"}, "text/ple_table": {"q8_g32_fp16"},
                "text/ple_projection": {"bf16"}, "text/layers/3/ple/gate": {"bf16"}}
        for name, formats_wanted in want.items():
            if formats(name) != formats_wanted:
                errors.append(f"gemma4small-tables: {name} is {formats(name)}, expected {formats_wanted}")
        table, head = value["bindings"]["text/token_embedding"], value["bindings"]["text/output_head"]
        if table.get("object") is None or table.get("object") != head.get("object"):
            errors.append(f"gemma4small-tables: the tied head is not the table's object: {table} {head}")
        if config.get("embedding_scale_at_run_time") is not True:
            errors.append("gemma4small-tables: the kept token table is not marked for a run-time scale")
        if config.get("kv_source_layers") != [0, 1, 0, 1]:
            errors.append(f"gemma4small-tables: kv_source_layers {config.get('kv_source_layers')}")
        if config.get("layer_intermediate_sizes") != [GINTER, GINTER, 2 * GINTER, 2 * GINTER]:
            errors.append(f"gemma4small-tables: widths {config.get('layer_intermediate_sizes')}")
        shared = [n for n in value["bindings"] if n.startswith(("text/layers/2/attention/k",
                                                                 "text/layers/3/attention/k",
                                                                 "text/layers/2/attention/v"))]
        if shared:
            errors.append(f"gemma4small-tables: shared-KV layers bind {shared}")
    print(f"gemma4small-tables: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def gemma4_assistant_kept_case(root: Path, native: Path, repository: Path) -> list[str]:
    """The gemma4-assistant GGUF under its chosen recipe (no --maintainer-quantize): every Q8_0
    projection kept word for word as q8_g32_fp16 (lossless), byte-equal to the twin converted with
    the named gemma4_assistant_q8 (whose absmax Q8 of the same values gives the same codes and scales;
    the kept blocks allow int8 activations, the twin's say A16Only)."""
    errors = []
    work = root / "gemma4-assistant-kept"
    gguf_file, twin, _ = gemma_assistant_case(work, 21)
    outs = {}
    twin_q8 = ["--recipe=gemma4_assistant_q8", "--maintainer-quantize"]
    for source, model, extra in (("gguf", gguf_file, []), ("twin", twin, twin_q8)):
        out = work / f"{source}.ninferno"
        result = subprocess.run([str(native), f"--model={model}", "--components=text", f"--out={out}", *extra],
                                cwd=repository, capture_output=True, text=True)
        if result.returncode:
            errors.append(f"gemma4-assistant-kept: converting the {source} failed\n{result.stderr[-2000:]}")
        elif source == "gguf" and ("recipe gemma4_assistant_published (chosen" not in result.stdout
                                   or "kept 23 published groupwise Q8 matrices" not in result.stdout):
            errors.append(f"gemma4-assistant-kept: not the chosen recipe with 23 kept Q8 matrices\n{result.stdout[-2000:]}")
        outs[source] = out
    if not errors:
        errors += compare("gemma4-assistant-kept", outs["gguf"], outs["twin"], kept_a8=True)
        notes = json.loads(Path(str(outs["gguf"]) + ".conversion.json").read_text(encoding="utf-8")).get("native_notes") or []
        if {n["lossless"] for n in notes if n["encoding"].startswith("gguf")} != {True}:
            errors.append(f"gemma4-assistant-kept: expected lossless gguf notes, got {notes}")
    print(f"gemma4-assistant-kept: {'ok' if not errors else 'FAILED'}", flush=True)
    return errors


def q8_cases(root: Path, native: Path, repository: Path) -> list[str]:
    """GGUF Q8_0 runs as published: its blocks are q8_g32_fp16's groups (32 int8 codes, one
    binary16 scale), so the chosen recipe keeps them word for word - bit-exact against the twin's
    values quantized to Q8 by grouped_absmax (each block holds a code of magnitude 127, so the
    absmax scale is the block's own). Control projections (in_proj_a/b) published in Q8_0 are
    kept too (the engine runs them through the generic path). A block with code -128 cannot be
    kept and stops the conversion."""
    errors = []

    def convert(model, out, *extra):
        return subprocess.run([str(native), f"--model={model}", f"--out={out}", "--components=text", *extra],
                              cwd=repository, capture_output=True, text=True)

    quant = {"attention": Q.Q8_0, "gdn": Q.Q8_0, "mlp": Q.Q8_0, "embedding": Q.Q8_0}
    work = root / "q8-kept"
    gguf_file, twin, _ = qwen_case(work, False, quant, 21, q8_ggml=True)
    kept, requantized = work / "kept.ninferno", work / "twin.ninferno"
    first = convert(gguf_file, kept)
    second = convert(twin, requantized, "--recipe=qwen3_5_q8", "--maintainer-quantize")
    if first.returncode or second.returncode:
        errors.append(f"q8-kept: conversion failed\n{first.stderr[-1500:]}{second.stderr[-1500:]}")
    else:
        if "published groupwise Q8 matrices as q8_g32_fp16" not in first.stdout:
            errors.append(f"q8-kept: the log does not say the Q8 matrices were kept: {first.stdout[-600:]}")
        report = json.loads(Path(str(kept) + ".conversion.json").read_text(encoding="utf-8"))
        notes = [(n["encoding"], n["handling"], n["lossless"]) for n in report.get("native_notes", [])]
        if notes != [("gguf q8_0", "kept as q8_g32_fp16", True)]:
            errors.append(f"q8-kept: expected every Q8_0 matrix kept losslessly, got {notes}")
        errors += compare("q8-kept", kept, requantized, kept_a8=True)

    work = root / "q8-control"
    gguf_file, _, _ = qwen_case(work, False, {**quant, "control": Q.Q8_0}, 22, q8_ggml=True)
    out = work / "kept.ninferno"
    result = convert(gguf_file, out)
    if result.returncode:
        errors.append(f"q8-control: conversion failed\n{result.stderr[-1500:]}")
    else:
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        control = {(m["format"], m["method"]) for m in report["methods"]
                   if any(p.endswith(("/gdn/a_projection", "/gdn/b_projection")) for p in m["parameters"])}
        if control != {("q8_g32_fp16", "import_encoded")}:
            errors.append(f"q8-control: control projections not kept as Q8: {control}")

    work = root / "q8-minus128"
    gguf_file, _, _ = qwen_case(work, False, quant, 23)
    result = convert(gguf_file, work / "refused.ninferno")
    if result.returncode == 0 or "code -128, which q8_g32_fp16 cannot keep" not in result.stderr:
        errors.append(f"q8-minus128: expected a stop naming code -128, got rc {result.returncode}: "
                      f"{result.stderr[-600:]}")
    for name in ("q8-kept", "q8-control", "q8-minus128"):
        failed = any(e.startswith(name + ":") for e in errors)
        print(f"{name}: {'FAILED' if failed else 'ok'}", flush=True)
    return errors


# ggml_planes_v1: format -> (qtype, values, code, high, scale bytes per block, the planes (code 0,
# high 1, scale 2) in GGUF block order).
GGML_PLANES = {
    "ggml_q6_k": (Q.Q6_K, 256, 192, 16, 2, (0, 1, 2)), "ggml_q3_k": (Q.Q3_K, 256, 96, 12, 2, (0, 1, 2)),
    "ggml_q2_k": (Q.Q2_K, 256, 64, 16, 4, (1, 0, 2)),
    "ggml_iq4_xs": (Q.IQ4_XS, 256, 128, 4, 4, (2, 1, 0)),
    "ggml_iq3_xxs": (Q.IQ3_XXS, 256, 64, 32, 2, (2, 0, 1)),
    "ggml_iq3_s": (Q.IQ3_S, 256, 104, 4, 2, (2, 0, 1)),
    "ggml_q4_0": (Q.Q4_0, 32, 16, 0, 2, (2, 1, 0)), "ggml_q4_1": (Q.Q4_1, 32, 16, 0, 4, (2, 1, 0)),
    "ggml_q5_0": (Q.Q5_0, 32, 16, 4, 2, (2, 1, 0)), "ggml_q5_1": (Q.Q5_1, 32, 16, 4, 4, (2, 1, 0)),
    "ggml_iq4_nl": (Q.IQ4_NL, 32, 16, 0, 2, (2, 1, 0)),
    "ggml_iq2_xxs": (Q.IQ2_XXS, 256, 64, 0, 2, (2, 0, 1)),
    "ggml_iq2_xs": (Q.IQ2_XS, 256, 64, 8, 2, (2, 0, 1)),
    "ggml_iq2_s": (Q.IQ2_S, 256, 72, 8, 2, (2, 0, 1)),
    "ggml_iq1_s": (Q.IQ1_S, 256, 32, 16, 2, (2, 0, 1)),
    "ggml_iq1_m": (Q.IQ1_M, 256, 32, 16, 8, (0, 1, 2)),
    "ggml_tq1_0": (Q.TQ1_0, 256, 52, 0, 2, (0, 1, 2)),
    "ggml_tq2_0": (Q.TQ2_0, 256, 64, 0, 2, (0, 1, 2)),
}


def published_codes(fmt: str, codes: np.ndarray) -> np.ndarray:
    """ggml_frag_v1 code bytes [n, blocks, code] back in the published order, by the maps in
    docs/maintainer/storage-layouts.md (Section 7), written here independently of core/ggml_frag."""
    n, blocks, code = codes.shape
    out = codes.copy()
    if fmt == "ggml_q4_0":  # each whole four-block span: a 4 x 4 transpose of 32-bit words
        spans = blocks // 4
        rows = out.reshape(n, blocks * 16)
        head = rows[:, :spans * 64].reshape(n, spans, 4, 4, 4)
        rows[:, :spans * 64] = head.transpose(0, 1, 3, 2, 4).reshape(n, spans * 64)
        return rows.reshape(n, blocks, code)
    if fmt == "ggml_q4_k":  # the 16-byte head, then each 64-byte half word-transposed
        for c in (16, 80):
            out[:, :, c:c + 64] = codes[:, :, c:c + 64].reshape(n, blocks, 4, 4, 4) \
                .transpose(0, 1, 3, 2, 4).reshape(n, blocks, 64)
        return out
    if fmt == "ggml_iq3_xxs":  # frag[32h + 8q + j] = pub[32h + 4j + q]
        return codes.reshape(n, blocks, 2, 4, 8).transpose(0, 1, 2, 4, 3).reshape(n, blocks, 64)
    if fmt == "ggml_iq2_s":  # per half h: qs even | signs even | qs odd | signs odd; qh kept
        f = codes[:, :, :64].reshape(n, blocks, 2, 2, 2, 8)  # h, b, (qs | signs), j
        qs = f[:, :, :, :, 0, :].transpose(0, 1, 2, 4, 3).reshape(n, blocks, 32)  # 16h + 2j + b
        signs = f[:, :, :, :, 1, :].transpose(0, 1, 2, 4, 3).reshape(n, blocks, 32)
        return np.concatenate([qs, signs, codes[:, :, 64:]], -1)
    raise ValueError(f"no ggml_frag_v1 map for {fmt}")


def ggml_blocks(o: dict, payload: bytes) -> tuple[np.ndarray, object]:
    """A kept ggml block-format object's rows as the GGUF's blocks [n, row bytes] (the planes put
    back together, ggml_frag_v1 code bytes put back in the published order), and its type."""
    n, k = o["shape"]
    blocks = k // (GGML_PLANES[o["format"]][1] if o["format"] in GGML_PLANES else 256)
    raw = np.frombuffer(payload, np.uint8)
    frag = o.get("layout") == "ggml_frag_v1"
    if o["format"] in GGML_PLANES:
        qtype, _, code, high_size, scale, order = GGML_PLANES[o["format"]]
        codes = raw[:n * blocks * code].reshape(n, blocks, code)
        if frag:
            codes = published_codes(o["format"], codes)
        high = -(-n * blocks * code // 256) * 256
        sc = raw[high:high + n * blocks * high_size].reshape(n, blocks, high_size)
        at = high + -(-n * blocks * high_size // 256) * 256
        d = raw[at:at + n * blocks * scale].reshape(n, blocks, scale)
        planes = (codes, sc, d)
        size = code + high_size + scale
        data = np.concatenate([planes[i] for i in order], -1).reshape(n, blocks * size)
    else:
        size = 144 if o["format"] == "ggml_q4_k" else 176
        data = raw[:n * blocks * size].reshape(n, blocks, size)
        if frag:
            data = published_codes(o["format"], data)
        data = data.reshape(n, blocks * size)
        qtype = Q.Q4_K if o["format"] == "ggml_q4_k" else Q.Q5_K
    return data, qtype


def kquant_rows(o: dict, payload: bytes, rows: slice) -> np.ndarray:
    """Rows of a kept ggml block-format object (K-quants, i-quants, 32-value blocks; published or
    ggml_frag_v1), the GGUF blocks put back together from the planes and decoded by gguf-py."""
    data, qtype = ggml_blocks(o, payload)
    return dequantize(np.ascontiguousarray(data[rows]), qtype).astype(np.float32).reshape(-1, o["shape"][1])


def binding_parts(value: dict, parameter: str) -> list[tuple[str, int, int]]:
    b = value["bindings"][parameter]
    parts = b["parts"] if "parts" in b else [{"object": b["object"], "range": b.get("range")}]
    return [(p["object"], *(p["range"] or (None, None))) for p in parts]


def companion_q8_check(name, parameter, values, reference, objects, data, offset) -> list[str]:
    """`values` [rows, K] (the kept blocks decoded) against a q8_g32_fp16 reference: every value
    within one Q8 step of its group (absmax / 127; the maintainer quantizes the BF16-rounded
    decode) plus the FP16 scale's rounding."""
    from tools.artifact.codecs.row_split import dequantize_row_split
    want = []
    for oid, begin, end in reference:
        o = objects[oid]
        words = data[offset + o["offset"]:offset + o["offset"] + o["bytes"]]
        full = dequantize_row_split(words, o["format"], o["shape"]).float().numpy().reshape(-1)
        begin, end = (0, full.size) if begin is None else (begin, end)
        want.append(full[begin:end])
    want = np.concatenate(want).reshape(values.shape)
    k = values.shape[1]
    groups = np.abs(values.reshape(values.shape[0], k // 32, 32)).max(axis=2, keepdims=True)
    bound = (groups * (1.0 / 127 + 2.0 ** -8)).repeat(32, axis=2).reshape(values.shape)
    gap = np.abs(values.astype(np.float64) - want.astype(np.float64))
    if (gap > bound + 1e-12).any():
        return [f"{name}: {parameter} Q8 reference off by {gap.max()} (bound {bound.max()})"]
    return []


def kept_kquant_check(name, work, gguf_file, formats, convert, minimum) -> list[str]:
    """Convert a GGUF with the chosen recipe (K-quants kept) and with --maintainer-quantize (BF16
    words): every kept parameter's rows, decoded by gguf-py and rounded to BF16, equal the BF16
    words, and every GGUF encoding is noted as kept losslessly. A DFlash / DFlash2 companion's
    projection is Q8 in the maintainer artifact (the recipe's table gives the drafter Q8): its
    rows equal the Q8 dequantization within Q8's rounding of each 32-value group."""
    errors = []
    kept, decoded = work / "kept.ninferno", work / "bf16.ninferno"
    first = convert(gguf_file, kept)
    second = convert(gguf_file, decoded, "--maintainer-quantize")
    if first.returncode or second.returncode:
        return [f"{name}: conversion failed\n{first.stderr[-1500:]}{second.stderr[-1500:]}"]
    for fmt in formats:
        if f"published matrices as {fmt}" not in first.stdout:
            errors.append(f"{name}: the log does not say {fmt} matrices were kept: {first.stdout[-800:]}")
    report = json.loads(Path(str(kept) + ".conversion.json").read_text(encoding="utf-8"))
    notes = report.get("native_notes", [])
    if not notes or any(not n["lossless"] or not n["handling"].startswith("kept as") for n in notes):
        errors.append(f"{name}: expected every GGUF encoding kept losslessly, got {notes}")
    kv, kobj, kdata, koff = directory(kept)
    dv, dobj, ddata, doff = directory(decoded)
    checked = 0
    for parameter in kv["bindings"]:
        parts = binding_parts(kv, parameter)
        if not any(kobj[oid]["format"].startswith("ggml_") for oid, *_ in parts):
            continue
        got = []
        for oid, begin, end in parts:
            o = kobj[oid]
            k = o["shape"][1]
            begin, end = (0, o["shape"][0] * k) if begin is None else (begin, end)
            payload = kdata[koff + o["offset"]:koff + o["offset"] + o["bytes"]]
            got.append(kquant_rows(o, payload, slice(begin // k, end // k)))
        reference = binding_parts(dv, parameter)
        if any(dobj[oid]["format"] == "q8_g32_fp16" for oid, *_ in reference):
            if not parameter.startswith(("dflash/", "dflash2/")):
                errors.append(f"{name}: {parameter} is Q8 in the maintainer artifact")
            else:
                errors += companion_q8_check(name, parameter, np.concatenate(got), reference,
                                             dobj, ddata, doff)
            checked += 1
            continue
        want_words = b""
        for oid, begin, end in reference:
            o = dobj[oid]
            begin, end = (0, o["shape"][0] * o["shape"][1]) if begin is None else (begin, end)
            want_words += ddata[doff + o["offset"] + 2 * begin:doff + o["offset"] + 2 * end]
        if bf16_words(np.concatenate(got)).tobytes() != want_words:
            errors.append(f"{name}: {parameter} decodes to other values than the GGUF's")
        checked += 1
    if checked < minimum:
        errors.append(f"{name}: only {checked} kept K-quant parameters")
    return errors


def dense_cases(root: Path, native: Path, repository: Path) -> list[str]:
    """Dense GGUFs (llama / qwen2 / qwen3) run their K-quants as published: the auto recipe keeps
    them (llama's Q/K rows de-interleaved whole, the words unchanged); Llama 3.x's rope_freqs give
    the rotary frequencies transformers computes from its llama3 rope parameters."""
    errors = []

    def convert(model, out, *extra):
        return subprocess.run([str(native), f"--model={model}", f"--out={out}", *extra],
                              cwd=repository, capture_output=True, text=True)

    configs = [
        ("dense-llama-kept", "llama", {"attention": Q.Q4_K, "mlp": Q.Q6_K, "embedding": Q.Q5_K},
         ("ggml_q4_k", "ggml_q6_k", "ggml_q5_k")),
        ("dense-qwen3-kept", "qwen3", {"attention": Q.Q6_K, "mlp": Q.Q4_K, "embedding": Q.Q8_0},
         ("ggml_q6_k", "ggml_q4_k")),
    ]
    for index, (name, arch, quant, formats) in enumerate(configs):
        work = root / name
        gguf_file, _, _ = dense_case(work, arch, quant, 40 + index, rebuilt=True)
        errors += kept_kquant_check(name, work, gguf_file, formats, convert, 8)

    name = "dense-llama3-rope"
    work = root / name
    gguf_file, _, _ = dense_case(work, "llama", None, 43, rebuilt=True, llama3=True)
    result = convert(gguf_file, work / "rope.ninferno")
    if result.returncode:
        errors.append(f"{name}: conversion failed\n{result.stderr[-1500:]}")
    else:
        value, *_ = directory(work / "rope.ninferno")
        groups = value["components"]["text"]["config"]["rope_groups"]
        got = np.array([f for group in groups.values() for f in group["inv_freq"]], np.float32)
        want = np.tile(llama3_inv_freq(500000.0, DD), len(groups)).astype(np.float32)
        worst = float(np.max(np.abs(got / want - 1))) if got.shape == want.shape else float("inf")
        if not worst <= 1e-6:
            errors.append(f"{name}: rotary frequencies from rope_freqs differ from transformers' llama3 "
                          f"ones by {worst:.2e} (relative)")
    for n in [c[0] for c in configs] + [name]:
        failed = any(e.startswith(n + ":") for e in errors)
        print(f"{n}: {'FAILED' if failed else 'ok'}", flush=True)
    return errors


def kquant_cases(root: Path, native: Path, repository: Path) -> list[str]:
    """GGUF Q2_K .. Q6_K run as published: the chosen (BF16) recipe keeps them as ggml_q4_k /
    ggml_q5_k (the blocks unchanged) and ggml_q6_k / ggml_q3_k / ggml_q2_k (code, scale and d
    planes), losslessly. Every kept parameter's rows, decoded by gguf-py and rounded to BF16, equal the
    BF16 words the same GGUF converts to under --maintainer-quantize (which dequantizes): the
    words, the row order and the plane layout all check. A GDN output projection whose columns
    llama.cpp reorders by value head (64-value chunks, here LV > LK) cannot move whole 256-value
    superblocks: it is kept byte for byte in the tiled order, which the config records."""
    errors = []

    def convert(model, out, *extra):
        return subprocess.run([str(native), f"--model={model}", f"--out={out}", "--components=text", *extra],
                              cwd=repository, capture_output=True, text=True)

    configs = [
        ("kquant-q4-q6", {"attention": Q.Q4_K, "embedding": Q.Q6_K}, ("ggml_q4_k", "ggml_q6_k")),
        ("kquant-q5-q4", {"attention": Q.Q5_K, "embedding": Q.Q4_K}, ("ggml_q5_k", "ggml_q4_k")),
        ("kquant-q6-q5", {"attention": Q.Q6_K, "embedding": Q.Q5_K}, ("ggml_q6_k", "ggml_q5_k")),
        ("kquant-q3-q2", {"attention": Q.Q3_K, "embedding": Q.Q2_K}, ("ggml_q3_k", "ggml_q2_k")),
        ("kquant-q2-q3", {"attention": Q.Q2_K, "embedding": Q.Q3_K}, ("ggml_q2_k", "ggml_q3_k")),
        # 32-value blocks and i-quants: the same planes, the same lossless keep.
        ("legacy-q40-q51", {"attention": Q.Q4_0, "embedding": Q.Q5_1}, ("ggml_q4_0", "ggml_q5_1")),
        ("legacy-q41-q50", {"attention": Q.Q4_1, "embedding": Q.Q5_0}, ("ggml_q4_1", "ggml_q5_0")),
        ("legacy-iq4nl-q40", {"attention": Q.IQ4_NL, "embedding": Q.Q4_0}, ("ggml_iq4_nl", "ggml_q4_0")),
        ("iquant-iq4xs-iq3s", {"attention": Q.IQ4_XS, "embedding": Q.IQ3_S}, ("ggml_iq4_xs", "ggml_iq3_s")),
        ("iquant-iq3xxs-iq4xs", {"attention": Q.IQ3_XXS, "embedding": Q.IQ4_XS},
         ("ggml_iq3_xxs", "ggml_iq4_xs")),
        # 2- and 1-bit i-quants and the ternary formats (Q1_0 / Q2_0: gguf-py cannot decode them;
        # ninferno_ggml_decode_test holds them bit-exact with ggml).
        ("iquant-iq2xxs-iq2s", {"attention": Q.IQ2_XXS, "embedding": Q.IQ2_S},
         ("ggml_iq2_xxs", "ggml_iq2_s")),
        ("iquant-iq2xs-iq1s", {"attention": Q.IQ2_XS, "embedding": Q.IQ1_S}, ("ggml_iq2_xs", "ggml_iq1_s")),
        ("iquant-iq1m-tq20", {"attention": Q.IQ1_M, "embedding": Q.TQ2_0}, ("ggml_iq1_m", "ggml_tq2_0")),
        ("ternary-tq10-iq2xxs", {"attention": Q.TQ1_0, "embedding": Q.IQ2_XXS},
         ("ggml_tq1_0", "ggml_iq2_xxs")),
    ]
    for index, (name, quant, formats) in enumerate(configs):
        work = root / name
        gguf_file, _, _ = qwen_case(work, False, {"gdn": Q.Q8_0, "mlp": Q.Q8_0, **quant}, 24 + index,
                                    q8_ggml=True)
        errors += kept_kquant_check(name, work, gguf_file, formats, convert, 5)

    # The GDN output projection kept as published in llama.cpp's tiled column order, with the
    # config saying so; its rows decode to exactly the words --maintainer-quantize gives. Q4_K and
    # Q5_K keep whole blocks, so their bytes are also compared with blk.N.ssm_out.weight. PER = 2
    # is Qwen3.5-4B / Qwen3.6-35B-A3B (16 key heads, 32 value heads); PER = 3 is Qwen3.8-27B (16 / 48).
    tiled_cases = [("kquant-gdn-tiled", Q.Q4_K, 2, 4, 27), ("kquant-gdn-tiled-q6", Q.Q6_K, 2, 4, 28),
                   ("kquant-gdn-tiled-per3", Q.Q5_K, 4, 12, 29), ("iquant-gdn-tiled-per3", Q.IQ4_XS, 4, 12, 30)]
    for name, qtype, lk, lv, seed in tiled_cases:
        work = root / name
        fmt = "ggml_" + qtype.name.lower()
        with qwen_geometry(lk, lv):
            gguf_file, twin, _ = qwen_case(work, False, {"gdn": qtype, "mlp": Q.Q8_0}, seed, q8_ggml=True)
        errors += kept_kquant_check(name, work, gguf_file, (fmt,), convert, 5)
        if not (work / "kept.ninferno").exists():
            continue
        kv, kobj, kdata, koff = directory(work / "kept.ninferno")
        order = kv["components"]["text"]["config"].get("linear_output_head_order")
        if order != "tiled":
            errors.append(f"{name}: expected linear_output_head_order tiled, got {order}")
        reader = gguf.GGUFReader(gguf_file)
        stored = {t.name: t for t in reader.tensors}
        outputs = sorted(p for p in kv["bindings"] if p.endswith("/gdn/output"))
        if not outputs:
            errors.append(f"{name}: no gdn/output parameters")
        for parameter in outputs:
            layer = int(parameter.split("/")[-3])
            tensor = stored[f"blk.{layer}.ssm_out.weight"]
            want = np.ascontiguousarray(tensor.data).tobytes()
            got = b""
            for oid, begin, end in binding_parts(kv, parameter):
                o = kobj[oid]
                if o["format"] != fmt:
                    errors.append(f"{name}: {parameter} kept as {o['format']}, not {fmt}")
                    break
                if qtype not in (Q.Q4_K, Q.Q5_K):
                    continue  # planes, not blocks: the decode check above covers it
                block, size = gguf.GGML_QUANT_SIZES[qtype]
                row_bytes = o["shape"][1] // block * size
                first, last = (0, o["shape"][0]) if begin is None else (begin // o["shape"][1], end // o["shape"][1])
                got += kdata[koff + o["offset"] + first * row_bytes:koff + o["offset"] + last * row_bytes]
            if qtype in (Q.Q4_K, Q.Q5_K) and got != want:
                errors.append(f"{name}: {parameter} bytes differ from blk.{layer}.ssm_out.weight")
    for name in [c[0] for c in configs] + [c[0] for c in tiled_cases]:
        failed = any(e.startswith(name + ":") for e in errors)
        print(f"{name}: {'FAILED' if failed else 'ok'}", flush=True)
    return errors


def frag_cases(root: Path, native: Path, repository: Path) -> list[str]:
    """Routed-expert banks of a Qwen3.5 MoE GGUF under the converter's default --layout auto:
    Q4_K and IQ2_S gate/up banks and Q4_0 downs at K 1024 in ggml_frag_v1, a Q4_0 down at K 256,
    the shared expert and every other object in their published layouts. The frag banks' bytes,
    put back in the published order, equal --layout published's, and every bank's rows are the
    GGUF's blocks (as a set per layer: the order is the published conversion's own)."""
    errors = []

    def convert(model, out, *extra):
        return subprocess.run([str(native), f"--model={model}", f"--out={out}", "--components=text", *extra],
                              cwd=repository, capture_output=True, text=True)

    configs = [
        # name, quantization, routed width, the formats expected in ggml_frag_v1
        ("frag-q4k-q40", {"experts": Q.Q4_K, "experts_down": Q.Q4_0}, 1024, {"ggml_q4_k", "ggml_q4_0"}),
        ("frag-iq2s-q40-narrow", {"experts": Q.Q4_K, "experts_gate_up": Q.IQ2_S, "experts_down": Q.Q4_0}, 256,
         {"ggml_iq2_s"}),
        ("frag-q4k-iq3xxs", {"experts": Q.Q4_K, "experts_down": Q.IQ3_XXS}, 256, {"ggml_q4_k", "ggml_iq3_xxs"}),
    ]
    for index, (name, quant, inter, frag_formats) in enumerate(configs):
        work = root / name
        with moe_geometry(inter):
            gguf_file, _, _ = qwen_case(work, True, {"attention": Q.Q8_0, "gdn": Q.Q8_0, "embedding": Q.Q8_0,
                                                     **quant}, 70 + index, q8_ggml=True)
        auto_out, published_out = work / "auto.ninferno", work / "published.ninferno"
        first, second = convert(gguf_file, auto_out), convert(gguf_file, published_out, "--layout=published")
        if first.returncode or second.returncode:
            errors.append(f"{name}: conversion failed\n{first.stderr[-1500:]}{second.stderr[-1500:]}")
            continue
        if "ggml_frag_v1" not in first.stdout:
            errors.append(f"{name}: the log does not report ggml_frag_v1 banks: {first.stdout[-600:]}")
        av, aobj, adata, aoff = directory(auto_out)
        pv, pobj, pdata, poff = directory(published_out)
        banks = {}  # object id -> layer of the routed bank it holds
        for parameter in av["bindings"]:
            m = re.search(r"(?:^|/)layers/(\d+)/(?:.*/)?experts/", parameter)
            if m:
                for oid, *_ in binding_parts(av, parameter):
                    banks[oid] = int(m.group(1))
        seen = set()
        for oid, o in aobj.items():
            p = pobj.get(oid)
            if p is None or p.get("layout") == "ggml_frag_v1":
                errors.append(f"{name}: {oid} missing from, or ggml_frag_v1 in, the published artifact")
                continue
            fmt = o.get("format", "")
            want = "ggml_frag_v1" if oid in banks and fmt in frag_formats and not (
                fmt == "ggml_q4_0" and o["shape"][1] < 1024) else p.get("layout")
            if o.get("layout") != want:
                errors.append(f"{name}: {oid} ({fmt}, K {o.get('shape', [0, 0])[-1]}) stored as "
                              f"{o.get('layout')}, expected {want}")
            abytes = adata[aoff + o["offset"]:aoff + o["offset"] + o["bytes"]]
            pbytes = pdata[poff + p["offset"]:poff + p["offset"] + p["bytes"]]
            if o.get("layout") != "ggml_frag_v1":
                if abytes != pbytes:
                    errors.append(f"{name}: {oid} differs from the published conversion")
                continue
            seen.add(fmt)
            blocks, _ = ggml_blocks(o, abytes)
            published, _ = ggml_blocks(p, pbytes)
            if not np.array_equal(blocks, published):
                errors.append(f"{name}: {oid} put back in the published order differs from --layout published")
            # The bank's rows against the GGUF's expert tensors of its layer.
            layer = banks[oid]
            reader = gguf.GGUFReader(gguf_file)
            stored = {t.name: t for t in reader.tensors}
            names = [f"blk.{layer}.ffn_{w}_exps.weight" for w in ("gate", "up", "down")]
            rows = []
            for tname in names:
                t = stored.get(tname)
                if t is not None and t.tensor_type == blocks_type(o["format"]) and \
                        t.shape[0] == o["shape"][1]:
                    data = np.ascontiguousarray(t.data)
                    rows += [bytes(r) for r in data.reshape(-1, data.shape[-1])]
            got = sorted(bytes(r) for r in blocks)
            if not rows or not set(got) <= set(rows) or len(got) != len(set(got)):
                errors.append(f"{name}: {oid}'s rows are not the GGUF's blocks of layer {layer}")
        if seen != frag_formats:
            errors.append(f"{name}: ggml_frag_v1 banks of {sorted(seen)}, expected {sorted(frag_formats)}")
        print(f"{name}: {'FAILED' if any(e.startswith(name + ':') for e in errors) else 'ok'}", flush=True)
    return errors


def blocks_type(fmt: str):
    return GGML_PLANES[fmt][0] if fmt in GGML_PLANES else (Q.Q4_K if fmt == "ggml_q4_k" else Q.Q5_K)


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []
    cases = {
        # name: (builder, recipe, components, expected note lossless flags)
        "qwen35-bf16": (lambda w: qwen_case(w, False, None, 1), "qwen3_6_27b", "text,mtp", set()),
        "qwen35-kquants": (lambda w: qwen_case(w, False, {"attention": Q.Q6_K, "gdn": Q.Q4_K, "mlp": Q.Q8_0,
                                                          "embedding": Q.Q6_K}, 2),
                           "qwen3_6_27b", "text,mtp", {False}),
        "qwen35moe-bf16": (lambda w: qwen_case(w, True, None, 3), "qwen3_6_35b_a3b", "text,mtp", set()),
        "qwen35moe-q8": (lambda w: qwen_case(w, True, {"experts": Q.Q8_0}, 4), "qwen3_6_35b_a3b", "text,mtp", {False}),
        "gemma4-bf16": (lambda w: gemma_case(w, False, "bf16", 5), "gemma4_31b_nvfp4", "text", set()),
        "gemma4-nvfp4": (lambda w: gemma_case(w, False, "nvfp4", 6), "gemma4_31b_nvfp4", "text", {True}),
        "gemma4-mxfp4": (lambda w: gemma_case(w, False, "mxfp4", 7), "gemma4_31b_nvfp4", "text", {True}),
        "gemma4-iq4": (lambda w: gemma_case(w, False, "iq4", 8), "gemma4_31b_nvfp4", "text", {False}),
        "gemma4moe-nvfp4": (lambda w: gemma_case(w, True, "nvfp4", 9), "gemma4_26b_a4b_nvfp4", "text", {True}),
        "gemma4moe-iq4": (lambda w: gemma_case(w, True, "iq4", 10), "gemma4_26b_a4b_nvfp4", "text", {False}),
        # Gemma 4 E2B/E4B: per-layer embeddings, shared-KV layers, double-wide MLP, as published.
        "gemma4small-bf16": (lambda w: gemma_case(w, False, "bf16", 18, small=True), "gemma4_published",
                             "text", set()),
        # Gemma 4's MTP drafter as a GGUF (gemma4-assistant); kept as published: gemma4_assistant_kept_case.
        "gemma4-assistant-q8": (lambda w: gemma_assistant_case(w, 20), "gemma4_assistant_q8", "text", {False}),
        # NVFP4 projections kept (the _nvfp4 recipe reads the checkpoint itself as its quantized
        # source), including value-head-reordered GDN rows and out_proj columns; the attention of
        # the first layers is read as values (CastDirect) through the NVFP4 decode and divisor.
        "qwen35-nvfp4": (lambda w: qwen_case(w, False, {"attention": Q.NVFP4, "gdn": Q.NVFP4, "mlp": Q.NVFP4}, 16),
                         "qwen3_6_27b_nvfp4", "text,mtp", {True, False}),
        # No tokenizer files beside the GGUF: rebuilt from its tokenizer.ggml metadata.
        "qwen35-rebuilt-resources": (lambda w: qwen_case(w, False, None, 11, rebuilt=True),
                                     "qwen3_6_27b", "text,mtp", set()),
        # A Qwen3.5 GGUF naming the "qwen2" pre-tokenizer (a transformers 5 re-save converted):
        # the Qwen2 regex, and Qwen3.5's add_bos_token false all the same.
        "qwen35-rebuilt-qwen2-pre": (lambda w: qwen_case(w, False, None, 19, rebuilt=True, pre="qwen2"),
                                     "qwen3_6_27b", "text,mtp", set()),
        "gemma4-rebuilt-resources": (lambda w: gemma_case(w, False, "bf16", 12, rebuilt=True),
                                     "gemma4_31b_nvfp4", "text", set()),
        # Dense decoders: llama's permuted Q/K rows, qwen2's biases, qwen3's q/k norms; tokenizers
        # rebuilt for llama.cpp's llama-bpe (BOS prepended) and qwen2 pre-tokenizers.
        "llama-bf16": (lambda w: dense_case(w, "llama", None, 33, rebuilt=True, llama3=True),
                       "dense_bf16", "text", set()),
        "llama-q8-untied": (lambda w: dense_case(w, "llama", {"attention": Q.Q8_0, "mlp": Q.Q4_K}, 34, tied=False),
                            "dense_bf16", "text", {False}),
        "qwen2-q8": (lambda w: dense_case(w, "qwen2", {"attention": Q.Q8_0, "mlp": Q.Q8_0}, 35, rebuilt=True),
                     "dense_bf16", "text", {False}),
        "qwen3-kquants": (lambda w: dense_case(w, "qwen3", {"attention": Q.Q4_K, "mlp": Q.Q6_K,
                                                            "embedding": Q.Q5_K}, 36, rebuilt=True),
                          "dense_bf16", "text", {False}),
        # Round 3: granite's muP multipliers (refact tokenizer), OLMo 2's post-norms and projection
        # q/k norms (dbrx), OLMo 3's sliding layers with yarn on the full ones, SmolLM3's NoPE
        # layers (smaug-bpe), Ministral 3's yarn and attention temperature (tekken), and a
        # SentencePiece vocabulary with scores (TinyLlama-style llama).
        "granite-kquants": (lambda w: dense_case(w, "granite", {"attention": Q.Q4_K, "mlp": Q.Q6_K,
                                                                "embedding": Q.Q6_K}, 50, rebuilt=True),
                            "dense_bf16", "text", {False}),
        "olmo2-untied": (lambda w: dense_case(w, "olmo2", {"attention": Q.Q8_0, "mlp": Q.Q4_K}, 51,
                                              rebuilt=True, tied=False),
                         "dense_bf16", "text", {False}),
        "olmo3-bf16": (lambda w: dense_case(w, "olmo3", None, 52, rebuilt=True, tied=False),
                       "dense_bf16", "text", set()),
        "smollm3-kquants": (lambda w: dense_case(w, "smollm3", {"attention": Q.Q4_K, "mlp": Q.Q5_K}, 53,
                                                 rebuilt=True),
                            "dense_bf16", "text", {False}),
        "mistral3-kquants": (lambda w: dense_case(w, "mistral3", {"attention": Q.Q6_K, "mlp": Q.Q4_K,
                                                                  "embedding": Q.Q8_0}, 54, rebuilt=True),
                             "dense_bf16", "text", {False}),
        # llama.cpp's "pixtral" hash outside Nemotron (mistral-community/pixtral-12b: the Llama 3
        # regex), no tokenizer files beside the GGUF.
        "mistral3-pixtral-rebuilt": (lambda w: dense_case(w, "mistral3", None, 60, rebuilt=True,
                                                          pre_tokenizer="pixtral"),
                                     "dense_bf16", "text", set()),
        "tinyllama-spm": (lambda w: dense_case(w, "tinyllama", {"mlp": Q.Q4_K}, 55, rebuilt=True),
                          "dense_bf16", "text", {False}),
        "smollm2-q8": (lambda w: dense_case(w, "smollm2", {"attention": Q.Q8_0, "mlp": Q.Q8_0}, 56, rebuilt=True),
                       "dense_bf16", "text", {False}),
        # Phi-3 (Phi-4-mini): fused attn_qkv / ffn_up split by rows against the Hugging Face
        # qkv_proj / gate_up_proj, partial rotary, LongRoPE factors (gpt-4o tokenizer).
        "phi3-kquants": (lambda w: dense_case(w, "phi3", {"attention": Q.Q5_K, "mlp": Q.Q4_K,
                                                          "embedding": Q.Q6_K}, 57, rebuilt=True),
                         "dense_bf16", "text", {False}),
        # gpt-oss (ggml-org's layout): MXFP4 expert banks split into gate / up, Q8_0 attention and
        # vocabulary, F32 router, sinks and biases, against transformers' interleaved blocks.
        # Under --maintainer-quantize the Q8_0 attention is decoded and quantized again (the
        # twin's values give the same words); the MXFP4 experts are kept either way.
        "gpt-oss-q8": (lambda w: gpt_oss_case(w, 58, q8_vocabulary=False), "moe_q8", "text", {True, False}),
        # Muse Glimmer: sandwich norms stored as 1 + w, permuted Q/K, the attention output gate,
        # qk_scale_factor from the constant attn_q_norm, NoPE full layers from a pattern period,
        # logit_scale / soft cap (llama4 pre-tokenizer: the o200k regex, BOS prepended).
        "muse-kquants": (lambda w: dense_case(w, "muse", {"attention": Q.Q3_K, "mlp": Q.Q4_K,
                                                          "embedding": Q.Q6_K}, 58, rebuilt=True,
                                              tied=False),
                         "dense_bf16", "text", {False}),
        # llama.cpp's DFlash drafter of Muse Glimmer: no table, no head, target_layers one past,
        # the mask token from the tokenizer metadata, K-quant formats mixed inside a layer.
        "dflash-kquants": (lambda w: dflash_case(w, {"attention": Q.Q4_K, "value": Q.Q6_K, "mlp": Q.Q4_K,
                                                     "down": Q.Q6_K}, 59),
                           "dflash_published", "text", {False}),
    }
    # NINFERNO_GGUF_CASES=frag,kquant,...: only those groups of the cases after the table (a quick
    # run of one area); unset, everything runs.
    only = {c for c in os.environ.get("NINFERNO_GGUF_CASES", "").split(",") if c}
    if only:
        cases = {}
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-gguf-") as tmp:
        root = Path(tmp)
        for name, (build, recipe, components, flags) in cases.items():
            work = root / name
            case_errors = []
            try:
                gguf_file, twin, stats = build(work)
            except Exception as error:  # noqa: BLE001
                errors.append(f"{name}: building the case failed: {error!r}")
                print(f"{name}: FAILED", flush=True)
                continue
            outs = {}
            for source, model in (("gguf", gguf_file), ("twin", twin)):
                out = work / f"{source}.ninferno"
                extra = [f"--source=quantized={model}"] if recipe.endswith("_nvfp4") and recipe.startswith("qwen") else []
                result = subprocess.run([str(native), f"--model={model}", f"--recipe={recipe}",
                                         f"--components={components}", "--maintainer-quantize",
                                         f"--out={out}", *extra],
                                        cwd=repository, capture_output=True, text=True)
                if result.returncode:
                    case_errors.append(f"{name}: converting the {source} failed\n{result.stderr[-3000:]}")
                outs[source] = out
            if not case_errors:
                case_errors += compare(name, outs["gguf"], outs["twin"])
                report = json.loads(Path(str(outs["gguf"]) + ".conversion.json").read_text(encoding="utf-8"))
                notes = report.get("native_notes") or []
                got = {n["lossless"] for n in notes if n["encoding"].startswith("gguf")}
                if got != flags:
                    case_errors.append(f"{name}: expected gguf notes lossless {flags}, got {notes}")
                if "gguf" not in report:
                    case_errors.append(f"{name}: the report has no gguf section")
            summary = ", ".join(f"{k} {v}" for k, v in stats.items())
            print(f"{name}: {'ok' if not case_errors else 'FAILED'}  ({summary})", flush=True)
            errors += case_errors
        groups = {
            "generation_fallback": generation_fallback_case, "muse_assistant": muse_assistant_case,
            "selection": selection_cases, "plain_decode": plain_decode_cases, "q8": q8_cases,
            "gemma4_small_tables": gemma4_small_tables_case,
            "gemma4_assistant_kept": gemma4_assistant_kept_case, "kquant": kquant_cases,
            "frag": frag_cases, "dense": dense_cases, "gpt_oss": gpt_oss_cases,
            "zlab_dflash": zlab_dflash_case,
        }
        for group, run in groups.items():
            if not only or group in only:
                errors += run(root, native, repository)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
