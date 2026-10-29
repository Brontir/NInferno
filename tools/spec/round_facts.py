#!/usr/bin/env python3
"""Writes each registry pair's round facts: the weight bytes and parameters a speculation round's
passes read (src/spec/round_cost.h), from the tensor tables of the target's and its drafters' files,
and the lab board that measured it.

    round_facts.py [--registry R] [--tables T.json] [--fetch]

A tensor table is {"meta": {...}, "tensors": {name: bytes}, "elems": {name: parameters},
"shapes": {name: [format, dims]}}: a GGUF
header's (its metadata; each tensor's bytes from the offsets of the next one, the last by its
type) or a safetensors header's with its config.json (expert counts as expert_count /
expert_used_count, vocab_size, a DFlash drafter's block_size). Parameters are the logical ones: a
packed tensor counts the values it holds (AWQ's qweight and compressed-tensors' weight_packed int32:
32 / bits each; a uint8 4-bit weight or MXFP4 block: two a byte); scales, zero points and group
indices count bytes, no parameters. --tables reads the tables from a JSON file keyed by a pair's
target_repo ("org/name@commit QUANT") and by "drafter:org/name@revision[:file]"; --fetch reads
the missing ones (and ones without shapes) from the hub with HTTP range reads (headers only: a target's files are the ones
whose LFS SHA-256 the pair names; a GGUF drafter's the file the registry's model entry names) and
adds each to the --tables file, the cache of later runs (the hub's rate limit, HTTP 429, is
waited out). A pair whose tables are missing keeps no round facts: its rounds stay
priced by the measured ratios (spec/pairs.h).

Each pass lists its linears too (round_cost.h LinearGroup), from the shapes: the matrices it
computes, grouped as {format, n output rows, k input columns, count, head, expert}. A format is the
published one ("gguf:Q4_K", a checkpoint's "bf16", "nvfp4", "mxfp4", "int4", "fp8"), or "conv:q8"
where the converter re-encodes a drafter's projections; an expert bank counts each expert; the head
is the output head (a tied one: the input table's; Gemma 4's assistant's over its proposal_rows).
They are the shapes a verify's routes are asked for (ops/generic/route_query.h).

What a pass reads (round_cost.h PassFacts / DraftFacts):
  target  its layers and output head (the input table when tied), its routed experts apart;
          vision / audio towers, the token embedding and MTP layers excluded.
  mtp     the target's own MTP layers and, apart ("head", per draft), its output head (per draft,
          one row: a proposal head reads its rows' share of it); a separate MTP
          drafter (Nemotron's GGUF sidecar) its layers (NextN ones included), its experts and its
          head (the target's when it has none); Gemma 4's assistant its layers and its head over
          its kind's proposal_rows (98 304; BF16, gathered).
  A drafter is priced as the converter attaches it: its kind's rule ("quant": q8 re-encodes the
  plain BF16 projections of Gemma 4 assistants and DFlash drafters; a published encoding is kept).
  dflash  the drafter's layers over its block (config block_size) and, once a round, the target's
          output head over the k drafted rows (models/speculation/dflash.cpp: the head runs on the
          drafted rows only); DFlash2's selector codebooks are row tables (ROW_TABLES), no pass's.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry_add import LAB_BOARDS, write_registry  # noqa: E402

REGISTRY = Path(__file__).resolve().parents[2] / "src/spec/data/drafter_registry.json"
NOT_TEXT = re.compile(r"(^v\.|^mm\.|visual|vision|audio|^a\.|embed_vision|embed_audio)")
NOT_PARAMS = re.compile(r"(scale|zero|g_idx|_scales$)")
# Row tables a round reads by id, a few rows of: DFlash2's selector codebooks (vocabulary x rank; the
# candidate path reads a drafted position's top_k successor rows and one predecessor row by token id,
# ops/candidate_selector/bf16/candidate_selector_path.cu: < 0.1 MB a round against the drafter's GBs).
# Resident, never a pass.
ROW_TABLES = re.compile(r"(_codebook$|^selector_(predecessor|successor)\.)")


def is_gguf(table: dict) -> bool:
    return any(n.startswith("blk.") for n in table["tensors"])


def tensor_class(name: str, gguf: bool, nextn: int, blocks: int) -> str | None:
    """A tensor's class: body, head, embed, experts, mtp, mtp_experts or rows (a table read by id,
    ROW_TABLES); None outside the text model."""
    if NOT_TEXT.search(name):
        return None
    if ROW_TABLES.search(name):
        return "rows"
    if gguf:
        m = re.match(r"blk\.(\d+)\.", name)
        layer = int(m.group(1)) if m else -1
        if name.startswith("token_embd") or "per_layer_token_embd" in name:
            return "embed"
        if name.startswith("output."):
            return "head"
        if nextn and layer >= blocks - nextn:
            return "mtp_experts" if "_exps" in name else "mtp"
        return "experts" if "_exps" in name else "body"
    # An input embedding table by its role: outside the layers, named for embedding (embed_tokens,
    # Nemotron's backbone.embeddings, wte, word_embeddings, ...): a token reads one row of it, no
    # pass reads it whole.
    in_layer = re.search(r"\.(layers|h|blocks)\.\d+\.", name) is not None
    if name.startswith("mtp."):
        return "mtp_experts" if ".experts" in name else "mtp"
    if ("embed" in name and not in_layer) or name.split(".")[-2:-1] in (["wte"], ["tok_embeddings"]):
        return "embed"
    if name.startswith("lm_head"):
        return "head"
    return "experts" if ".experts" in name else "body"


def _layout(table: dict) -> tuple[bool, int, int, int, int]:
    """GGUF or not, NextN layers, blocks, experts, experts a token uses."""
    meta = table.get("meta", {})
    arch = meta.get("general.architecture", "")
    return (is_gguf(table), int(meta.get(f"{arch}.nextn_predict_layers", 0) or 0),
            int(meta.get(f"{arch}.block_count", 0) or 0),
            int(meta.get(f"{arch}.expert_count", meta.get("expert_count", 0)) or 0),
            int(meta.get(f"{arch}.expert_used_count", meta.get("expert_used_count", 0)) or 0))


def classes(table: dict) -> tuple[dict, dict, int, int]:
    """Bytes and parameters per class (body, head, embed, experts, mtp, mtp_experts), expert counts."""
    tensors, elems = table["tensors"], table["elems"]
    gguf, nextn, blocks, experts, used = _layout(table)
    b: dict[str, float] = {}
    n: dict[str, float] = {}
    for name, size in tensors.items():
        c = tensor_class(name, gguf, nextn, blocks)
        if c is None:
            continue
        b[c] = b.get(c, 0) + size
        n[c] = n.get(c, 0) + elems[name]
    if not b.get("head"):  # tied: the head reads the input table
        b["head"], n["head"] = b.get("embed", 0), n.get("embed", 0)
    return b, n, experts, used


# ---- The linears a pass computes, from each tensor's published format and dims.
GGML_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
              23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
              30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
NORM = re.compile(r"norm[^.]*\.weight$")  # a normalization's weights by role (Nemotron-H's grouped ssm_norm [512, 8])
WEIGHT_SUFFIX = (".weight", "qweight", "weight_packed", "_blocks")
FUSED_EXPERTS = re.compile(r"\.experts\.\w+_proj$")  # a checkpoint's experts as one [E, n, k] tensor


def safetensors_format(name: str, dtype: str, bits: int) -> str:
    """A safetensors weight's published format: nvfp4 (ModelOpt's uint8 weight, compressed-tensors'
    weight_packed; mxfp4 where its weight_scale is E8M0, parse_safetensors_header), mxfp4 (_blocks), int<bits> (AWQ / GPTQ qweight, compressed-tensors' int32
    weight_packed), fp8, int8 (compressed-tensors int-quantized), else its dtype (bf16, f16, f32)."""
    if dtype == "U8" and "_blocks" in name:
        return "mxfp4"
    if dtype == "U8" and name.endswith((".weight", "weight_packed")):
        return "nvfp4"
    if dtype in ("I32", "U32") and name.endswith(("qweight", "weight_packed")):
        return f"int{bits}"
    if dtype.startswith("F8"):
        return "fp8"
    if dtype == "I8":
        return "int8"  # compressed-tensors int-quantized W8A16 (the converter's name)
    return dtype.lower()


def linear_shape(name: str, fmt: str, dims: list[int], elems: int, method: str) -> tuple[int, int, int] | None:
    """(n output rows, k input columns, count) of a weight tensor computed as a linear; None for
    anything else (norms, convolutions, scales, and row tables read by id: DFlash2's selector
    codebooks, models/qwen3_5/execution/parameters.cpp). A GGUF tensor is [k, n(, experts)]; a
    checkpoint's [(experts,) n, k packed]: MXFP4 blocks [(experts,) n, k/32, 16], AWQ's qweight
    [k, n packed], GPTQ's [k packed, n] (a qweight of another method: None). A fused expert bank
    (no .weight) is read as HF Qwen3-MoE stores it, [E, n, k]; banks stored [E, k, n] (gpt-oss's
    BF16 gate_up_proj, Llama 4) would read transposed: none in the registry, add a rule with one."""
    if not elems or not dims or NORM.search(name):
        return None
    if fmt.startswith("gguf:"):
        if not name.endswith(".weight") or len(dims) < 2:
            return None
        k, n = dims[0], dims[1]
        count = elems // max(1, n * k)
    else:
        fused = FUSED_EXPERTS.search(name) is not None
        if not (name.endswith(WEIGHT_SUFFIX) or fused):
            return None
        blocks = name.endswith("_blocks")  # gpt-oss MXFP4: [(experts,) n, k/32, 16]
        rank = len(dims) - (1 if blocks else 0)
        if rank != (3 if fused or (blocks and ".experts" in name) else 2):
            return None
        count = dims[0] if rank == 3 else 1
        if name.endswith("qweight"):
            if method not in ("awq", "gptq"):
                return None
            n = dims[1] if method == "gptq" else elems // max(1, dims[0])
        else:
            n = dims[-3] if blocks else dims[-2]
        k = elems // max(1, n * count)
    if k < 32 or n < 1 or n * k * count != elems:
        return None
    return n, k, count


def linears(table: dict, wanted: tuple[str, ...], quant: str = "", head_rows: int = 0) -> list[dict]:
    """The linears of a table's classes in `wanted` grouped as {format, n, k, count, head, expert}:
    a pass's matrices as published (a tied head: the input table's), "conv:q8" where the converter
    re-encodes them (as_converted), a head over at most `head_rows` rows. Empty without shapes."""
    shapes = table.get("shapes")
    if not shapes:
        return []
    gguf, nextn, blocks, _, _ = _layout(table)
    method = str(table.get("meta", {}).get("quant_method", ""))
    convert = quant == "q8" and not gguf
    has_head = any(tensor_class(t, gguf, nextn, blocks) == "head" for t in shapes)
    groups: dict[tuple, int] = {}
    for name, (fmt, dims) in shapes.items():
        c = tensor_class(name, gguf, nextn, blocks)
        if c == "embed" and "head" in wanted and not has_head and "per_layer" not in name:
            c = "head"  # tied: the head reads the input table
        if c not in wanted:
            continue
        e = table["elems"].get(name, 0)
        shape = linear_shape(name, fmt, dims, e, method)
        if shape is None:
            continue
        n, k, count = shape
        if convert and e >= 65536 and table["tensors"][name] >= 1.9 * e:
            fmt = "conv:q8"
        if c == "head" and head_rows:
            n = min(n, head_rows)
        key = (fmt, n, k, c == "head", c in ("experts", "mtp_experts"))
        groups[key] = groups.get(key, 0) + count
    return [dict(format=f, n=n, k=k, count=count, head=h, expert=x)
            for (f, n, k, h, x), count in sorted(groups.items(), key=lambda g: (g[0][3], g[0][4], g[0]))]


def pass_of(b, n, dense, routed, experts, used) -> dict:
    out = {"bytes": int(sum(b.get(c, 0) for c in dense)), "params": int(sum(n.get(c, 0) for c in dense))}
    eb = int(sum(b.get(c, 0) for c in routed))
    if eb and experts and used:
        out.update(expert_bytes=eb, expert_params=int(sum(n.get(c, 0) for c in routed)),
                   experts=experts, experts_used=used)
    return out


def drafter_key(registry: dict, pair: dict, kind: str) -> str:
    """The tables' key of the pair's drafter of `kind`: "drafter:repo@revision", with ":file" for a
    GGUF sidecar (the registry model entry's "gguf")."""
    at = (pair.get("drafters") or {}).get(kind, "")
    repo = at.split("@")[0]
    for m in registry.get("models", []):
        for d in m.get("drafters", []):
            if d.get("kind") == kind and d.get("repo") == repo and d.get("gguf"):
                return f"drafter:{at}:{d['gguf']}.gguf"
    return "drafter:" + at


def drafter_rule(registry: dict, pair: dict, kind: str) -> tuple[str, int]:
    """The quantization and head rows a drafter of `kind` is attached with (src/spec/registry.h
    apply_kind_rule): its registry entry's own, else its kind's rule."""
    rule = (registry.get("kinds") or {}).get(kind, {})
    repo = (pair.get("drafters") or {}).get(kind, "").split("@")[0]
    for m in registry.get("models", []):
        for d in m.get("drafters", []):
            if d.get("kind") == kind and d.get("repo") == repo:
                return (d.get("quant") or rule.get("quant", ""),
                        int(d.get("proposal_rows") or rule.get("proposal_rows", 0) or 0))
    return rule.get("quant", ""), int(rule.get("proposal_rows", 0) or 0)


def as_converted(table: dict, quant: str) -> dict:
    """The table as the converter attaches the drafter: q8 re-encodes its plain BF16 / F16
    projections (Q8_0, 34 bytes a 32 values); a published encoding (a GGUF, a quantized tensor)
    and small vectors (norms, biases) are kept."""
    if quant != "q8" or any(n.startswith("blk.") for n in table["tensors"]):
        return table
    out = dict(table, tensors=dict(table["tensors"]))
    for name, size in table["tensors"].items():
        e = table["elems"][name]
        if e >= 65536 and size >= 1.9 * e:
            out["tensors"][name] = e * 34 // 32
    return out


def with_linears(d: dict, groups: list[dict]) -> dict:
    """The pass with its linears, when its tables carry shapes."""
    if groups:
        d["linears"] = groups
    return d


def round_facts(pair: dict, tables: dict, registry: dict) -> dict | None:
    target = tables.get(pair["target_repo"])
    if target is None:
        return None
    b, n, experts, used = classes(target)
    out = {"target": with_linears(pass_of(b, n, ["body", "head"], ["experts"], experts, used),
                                  linears(target, ("body", "head", "experts"))), "drafters": {}}
    target_head = linears(target, ("head",))
    modes = {m["config"].split(":")[0] for m in pair["tune"]["measurements"] if m["config"] != "plain"}
    kinds = {d.get("kind") for d in pair["tune"].get("drafters", [])}
    for mode in sorted(modes):
        if mode == "mtp":
            # A separate MTP drafter the pair was measured with takes precedence over MTP layers the
            # checkpoint carries (Nemotron NVFP4 keeps bf16 mtp.* tensors; the lab ran the sidecar).
            if "nemotron_h_mtp" in kinds or "gemma4_assistant" in kinds:
                kind = "nemotron_h_mtp" if "nemotron_h_mtp" in kinds else "gemma4_assistant"
                table = tables.get(drafter_key(registry, pair, kind))
                if table is None:
                    continue
                quant, head_rows = drafter_rule(registry, pair, kind)
                qb, qn, qe, qu = classes(as_converted(table, quant))
                if kind == "nemotron_h_mtp":
                    if not qb.get("head"):
                        qb["head"], qn["head"] = b["head"], n["head"]
                    # A sidecar's layers are its NextN layers: all of them, and their experts.
                    d = pass_of(qb, qn, ["body", "mtp", "head"], ["experts", "mtp_experts"],
                                qe or experts, qu or used)
                    own = linears(table, ("body", "mtp", "head", "experts", "mtp_experts"), quant)
                    if own and not any(g["head"] for g in own):
                        own += target_head
                    with_linears(d, own)
                else:
                    # Its own head over the shortlist's rows, gathered as published (BF16): the
                    # kind's proposal_rows (the full vocabulary when none).
                    vocab = max(1, int(table.get("meta", {}).get("vocab_size", 262144)))
                    rows = min(head_rows or vocab, vocab)
                    head_params = qn.get("head", 0) * rows / vocab
                    d = with_linears({"bytes": int(qb.get("body", 0) + 2 * head_params),
                                      "params": int(qn.get("body", 0) + head_params)},
                                     linears(table, ("body",), quant) + linears(table, ("head",), "", rows))
            elif b.get("mtp") or b.get("mtp_experts"):
                # Its layers per draft, and the target's output head per draft apart: a proposal
                # head (spec/round_cost.h ProposalRows) reads a restricted share of its rows.
                d = with_linears(pass_of(b, n, ["mtp"], ["mtp_experts"], experts, used),
                                 linears(target, ("mtp", "mtp_experts")))
                d["head"] = with_linears({"bytes": int(b["head"]), "params": int(n["head"])}, target_head)
            else:
                continue
            d.update(rows=1, per_draft=True)
        else:
            table = tables.get(drafter_key(registry, pair, mode))
            if table is None:
                continue
            meta = table.get("meta", {})
            arch = meta.get("general.architecture", "")
            block = int(meta.get("block_size", 0) or meta.get(f"{arch}.block_size", 0) or 0)
            if not 1 <= block <= 16:
                print(f"{pair['target_repo']}: {mode} drafter without a block_size", file=sys.stderr)
                continue
            quant = drafter_rule(registry, pair, mode)[0]
            qb, qn, _, _ = classes(as_converted(table, quant))
            d = with_linears({"bytes": int(qb.get("body", 0)), "params": int(qn.get("body", 0)),
                              "rows": block, "per_draft": False,
                              "head": with_linears({"bytes": int(b["head"]), "params": int(n["head"])},
                                                   target_head)},
                             linears(table, ("body",), quant))
        if d["bytes"] > 0 and d["params"] > 0:
            out["drafters"][mode] = d
    return out if out["drafters"] and out["target"]["bytes"] > 0 else None


# ---- The hub: tensor tables by HTTP range reads of the headers.
# ggml types: (elements a block, bytes a block); the last tensor of a file is sized by it.
GGML_BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
               9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
               14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
               19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1),
               25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
               35: (256, 66), 36: (256, 66), 37: (256, 66), 38: (256, 66), 39: (32, 17), 40: (64, 36),
               41: (128, 18), 42: (64, 18)}  # convert/ggml_quants.cpp traits


def _open(req, tries: int = 8) -> bytes:
    """The response body; the hub's rate limit (HTTP 429) is waited out, longer each time."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or i + 1 == tries:
                raise
            after = (error.headers or {}).get("Retry-After") or "0"
            error.close()
            time.sleep(max(int(after) if after.isdigit() else 0, 30 * (i + 1)))
    raise AssertionError("unreachable")


def _get(url: str) -> bytes:
    return _open(url)


def _range(url: str, a: int, b: int) -> bytes:
    return _open(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}))


def _gguf(url: str) -> tuple[dict, dict, dict, dict]:
    size = 8 << 20
    while True:
        buf = _range(url, 0, size - 1)
        try:
            return parse_gguf(buf)
        except (IndexError, struct.error):
            if len(buf) < size:
                raise
            size *= 4


def parse_gguf(buf: bytes) -> tuple[dict, dict, dict, dict]:
    """Metadata, and bytes, parameters and [format, dims] per tensor of a GGUF header (`buf` its
    first bytes; a format is "gguf:" and the ggml type's name)."""
    at = [4]

    def rd(fmt):
        v = struct.unpack_from(fmt, buf, at[0])[0]
        at[0] += struct.calcsize(fmt)
        return v

    def text():
        k = rd("<Q")
        if at[0] + k > len(buf):
            raise IndexError
        v = buf[at[0]:at[0] + k].decode("utf8", "replace")
        at[0] += k
        return v

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
              11: "<q", 12: "<d"}

    def value(t):
        if t == 8:
            return text()
        if t == 9:
            inner, count = rd("<I"), rd("<Q")
            return [value(inner) for _ in range(count)]
        return rd(scalar[t])

    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    rd("<I")
    count, keys = rd("<Q"), rd("<Q")
    meta = {}
    for _ in range(keys):
        k = text()
        v = value(rd("<I"))
        if not isinstance(v, list):
            meta[k] = v
    infos, shapes = [], {}
    for _ in range(count):
        name = text()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        kind, offset = rd("<I"), rd("<Q")
        e = 1
        for d in dims:
            e *= d
        infos.append((offset, name, kind, e))
        shapes[name] = ["gguf:" + GGML_NAMES.get(kind, f"type{kind}"), dims]
    infos.sort()
    tensors, elems = {}, {}
    for i, (offset, name, kind, e) in enumerate(infos):
        if i + 1 < len(infos):
            tensors[name] = infos[i + 1][0] - offset  # exact, its alignment padding with it
        elif kind in GGML_BLOCKS:
            per, size = GGML_BLOCKS[kind]
            tensors[name] = e * size // per
        else:
            raise ValueError(f"ggml type {kind} of the last tensor {name}: no size known")
        if kind not in GGML_BLOCKS:
            print(f"note: ggml type {kind} ({name}) sized by its offsets", file=sys.stderr)
        elems[name] = e
    return meta, tensors, elems, shapes


def parse_safetensors_header(header: dict, bits: int) -> tuple[dict, dict, dict]:
    """Bytes, logical parameters (packed values counted; scales and zero points none) and
    [format, dims] (safetensors_format) per tensor."""
    tensors, elems, shapes = {}, {}, {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        e = 1
        for d in v["shape"]:
            e *= d
        if NOT_PARAMS.search(k.split(".")[-1]) or k.endswith(("_scales", ".qzeros", ".g_idx")):
            e = 0
        elif v["dtype"] in ("I32", "U32") and k.endswith(("qweight", "weight_packed")):
            e *= 32 // bits
        elif v["dtype"] == "U8" and (k.endswith((".weight", "weight_packed")) or "_blocks" in k):
            e *= 2  # two 4-bit values a byte (ModelOpt NVFP4, compressed-tensors NVFP4, MXFP4)
        tensors[k], elems[k] = b - a, e
        fmt = safetensors_format(k, v["dtype"], bits)
        if fmt == "nvfp4" and (header.get(k.rsplit(".", 1)[0] + ".weight_scale") or {}).get("dtype") == "U8":
            fmt = "mxfp4"  # compressed-tensors MXFP4: the same packed codes, E8M0 (uint8) block scales
        shapes[k] = [fmt, list(v["shape"])]
    return tensors, elems, shapes


def _config_meta(cfg: dict) -> tuple[dict, int]:
    """The config's facts a table keeps, and the packed weights' bits (4 when it names none)."""
    meta, text = {}, cfg.get("text_config", cfg)
    for k, out in (("num_experts", "expert_count"), ("num_local_experts", "expert_count"),
                   ("n_routed_experts", "expert_count"), ("num_experts_per_tok", "expert_used_count"),
                   ("top_k_experts", "expert_used_count")):
        if text.get(k):
            meta[out] = text[k]
    if text.get("vocab_size"):
        meta["vocab_size"] = text["vocab_size"]
    block = cfg.get("block_size") or (cfg.get("dflash_config") or {}).get("block_size")
    if block:
        meta["block_size"] = block
    q = cfg.get("quantization_config") or text.get("quantization_config") or {}
    if q.get("quant_method"):
        meta["quant_method"] = q["quant_method"]
    bits = q.get("bits") or q.get("w_bit")
    for group in (q.get("config_groups") or {}).values():
        bits = bits or (group.get("weights") or {}).get("num_bits")
    return meta, int(bits or 4)


def fetch(repo_at: str, files: list[str] | None = None, shas: list[str] | None = None) -> dict:
    """The tensor table of org/name@revision: `files`, else the files whose LFS SHA-256 is one of
    `shas`, else its safetensors; with config.json's facts."""
    repo, _, rev = repo_at.partition("@")
    rev = rev or "main"
    listing = json.loads(_get(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"))
    siblings = listing.get("siblings", [])
    names = [s["rfilename"] for s in siblings]
    if not files and shas:
        files = [s["rfilename"] for s in siblings if (s.get("lfs") or {}).get("sha256") in shas]
    if not files:
        files = [f for f in names if f.endswith(".safetensors")]
    base = f"https://huggingface.co/{repo}/resolve/{rev}/"
    meta, bits = ({}, 4)
    if "config.json" in names:
        meta, bits = _config_meta(json.loads(_get(base + "config.json")))
    tensors, elems, shapes = {}, {}, {}
    for f in files:
        if f.endswith(".gguf"):
            m, t, e, s = _gguf(base + f)
            meta.update(m)
        else:
            n = struct.unpack("<Q", _range(base + f, 0, 7))[0]
            t, e, s = parse_safetensors_header(json.loads(_range(base + f, 8, 8 + n - 1)), bits)
        tensors.update(t)
        elems.update(e)
        shapes.update(s)
    if not tensors:
        raise ValueError(f"no weight files found at {repo_at}")
    return {"meta": meta, "tensors": tensors, "elems": elems, "shapes": shapes}


def wanted_tables(registry: dict, pair: dict) -> list[tuple]:
    """(key, org/name@revision, files, LFS SHA-256s) of the tables a pair's facts read."""
    out = [(pair["target_repo"], pair["target_repo"].split(" ")[0], None, pair["target"].split(","))]
    for kind in (pair.get("drafters") or {}):
        key = drafter_key(registry, pair, kind)
        parts = key[len("drafter:"):].split(":")
        out.append((key, parts[0], parts[1:] or None, None))
    return out


def refresh_tables(registry: dict, tables: dict, read=None, save=None) -> set[str]:
    """Reads the pairs' missing tables (and ones without shapes) into `tables` with `read` (fetch),
    calling `save` after each; returns the keys still unread at the end (their pairs keep the facts
    they have). A table one pair could not read and a later pair read is no longer unread."""
    read = read or fetch
    failed: set[str] = set()
    for p in registry.get("pairs", []):
        for key, at, files, shas in wanted_tables(registry, p):
            if key in tables and "shapes" in tables[key]:
                continue  # the cache: read before (a table without shapes is read again)
            try:
                tables[key] = read(at, files, shas)
                print(f"{key}: {len(tables[key]['tensors'])} tensors", file=sys.stderr)
            except Exception as error:  # noqa: BLE001 - one model's failure skips its pairs
                print(f"{key}: {error}", file=sys.stderr)
                failed.add(key)
                continue
            failed.discard(key)
            if save is not None:
                save(tables)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    before = copy.deepcopy(registry)
    tables = json.loads(args.tables.read_text(encoding="utf-8")) if args.tables and args.tables.exists() else {}
    failed: set[str] = set()
    if args.fetch:
        def save(t: dict) -> None:
            if args.tables:  # whole or not at all: a crash mid-write keeps the cache
                part = args.tables.with_name(args.tables.name + ".part")
                part.write_text(json.dumps(t), encoding="utf-8")
                os.replace(part, args.tables)
        failed = refresh_tables(registry, tables, fetch, save)
    done = 0
    for p in registry.get("pairs", []):
        board = LAB_BOARDS.get(p["tune"]["key"]["gpu"])
        if board:
            p["lab_board"] = board
        missing = failed & {key for key, *_ in wanted_tables(registry, p)}
        if missing:
            print(f"{p['target_repo']}: {sorted(missing)} not read, its facts kept as they are", file=sys.stderr)
            done += "round_facts" in p
            continue
        facts = round_facts(p, tables, registry)
        if facts:
            p["round_facts"] = facts
            done += 1
        else:
            p.pop("round_facts", None)
            print(f"{p['target_repo']}: no tables, no round facts", file=sys.stderr)
    write_registry(registry, before, args.registry)
    print(f"{args.registry}: round facts for {done} of {len(registry.get('pairs', []))} pair(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
