#!/usr/bin/env python3
"""Writes each registry pair's round facts: the weight bytes and parameters a speculation round's
passes read (src/spec/round_cost.h), from the tensor tables of the target's and its drafters' files,
and the lab board that measured it.

    round_facts.py [--registry R] [--tables T.json] [--fetch]

A tensor table is {"meta": {...}, "tensors": {name: bytes}, "elems": {name: parameters}}: a GGUF
header's (its metadata; each tensor's bytes from the offsets of the next one, the last by its
type) or a safetensors header's with its config.json (expert counts as expert_count /
expert_used_count, vocab_size, a DFlash drafter's block_size). Parameters are the logical ones: a
packed tensor counts the values it holds (AWQ's qweight and compressed-tensors' weight_packed int32:
32 / bits each; a uint8 4-bit weight or MXFP4 block: two a byte); scales, zero points and group
indices count bytes, no parameters. --tables reads the tables from a JSON file keyed by a pair's
target_repo ("org/name@commit QUANT") and by "drafter:org/name@revision[:file]"; --fetch reads
the missing ones from the hub with HTTP range reads (headers only: a target's files are the ones
whose LFS SHA-256 the pair names; a GGUF drafter's the file the registry's model entry names) and
adds them to the --tables file. A pair whose tables are missing keeps no round facts: its rounds stay
priced by the measured ratios (spec/pairs.h).

What a pass reads (round_cost.h PassFacts / DraftFacts):
  target  its layers and output head (the input table when tied), its routed experts apart;
          vision / audio towers, the token embedding and MTP layers excluded.
  mtp     the target's own MTP layers and its output head (per draft, one row); a separate MTP
          drafter (Nemotron's GGUF sidecar) its layers (NextN ones included), its experts and its
          head (the target's when it has none); Gemma 4's assistant its layers and its head over
          its first 98 304 vocabulary rows (the proposal rows the engine scores).
  dflash  the drafter's layers over its block (config block_size) and, once a round, the target's
          output head over the k drafted rows (models/speculation/dflash.cpp: the head runs on the
          drafted rows only).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry_add import LAB_BOARDS, write_registry  # noqa: E402

REGISTRY = Path(__file__).resolve().parents[2] / "src/spec/data/drafter_registry.json"
GEMMA_ASSISTANT_HEAD_ROWS = 98304
NOT_TEXT = re.compile(r"(^v\.|^mm\.|visual|vision|audio|^a\.|embed_vision|embed_audio)")
NOT_PARAMS = re.compile(r"(scale|zero|g_idx|_scales$)")


def classes(table: dict) -> tuple[dict, dict, int, int]:
    """Bytes and parameters per class (body, head, embed, experts, mtp, mtp_experts), expert counts."""
    meta, tensors, elems = table.get("meta", {}), table["tensors"], table["elems"]
    gguf = any(n.startswith("blk.") for n in tensors)
    arch = meta.get("general.architecture", "")
    nextn = int(meta.get(f"{arch}.nextn_predict_layers", 0) or 0)
    blocks = int(meta.get(f"{arch}.block_count", 0) or 0)
    experts = int(meta.get(f"{arch}.expert_count", meta.get("expert_count", 0)) or 0)
    used = int(meta.get(f"{arch}.expert_used_count", meta.get("expert_used_count", 0)) or 0)
    b: dict[str, float] = {}
    n: dict[str, float] = {}
    for name, size in tensors.items():
        if NOT_TEXT.search(name):
            continue
        m = re.match(r"blk\.(\d+)\.", name)
        layer = int(m.group(1)) if m else -1
        if gguf:
            if name.startswith("token_embd") or "per_layer_token_embd" in name:
                c = "embed"
            elif name.startswith("output."):
                c = "head"
            elif nextn and layer >= blocks - nextn:
                c = "mtp_experts" if "_exps" in name else "mtp"
            elif "_exps" in name:
                c = "experts"
            else:
                c = "body"
        else:
            # An input embedding table by its role: outside the layers, named for embedding
            # (embed_tokens, Nemotron's backbone.embeddings, wte, word_embeddings, ...): a token
            # reads one row of it, no pass reads it whole.
            in_layer = re.search(r"\.(layers|h|blocks)\.\d+\.", name) is not None
            if name.startswith("mtp."):
                c = "mtp_experts" if ".experts" in name else "mtp"
            elif ("embed" in name and not in_layer) or name.split(".")[-2:-1] in (["wte"], ["tok_embeddings"]):
                c = "embed"
            elif name.startswith("lm_head"):
                c = "head"
            elif ".experts" in name:
                c = "experts"
            else:
                c = "body"
        b[c] = b.get(c, 0) + size
        n[c] = n.get(c, 0) + elems[name]
    if not b.get("head"):  # tied: the head reads the input table
        b["head"], n["head"] = b.get("embed", 0), n.get("embed", 0)
    return b, n, experts, used


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


def round_facts(pair: dict, tables: dict, registry: dict) -> dict | None:
    target = tables.get(pair["target_repo"])
    if target is None:
        return None
    b, n, experts, used = classes(target)
    out = {"target": pass_of(b, n, ["body", "head"], ["experts"], experts, used), "drafters": {}}
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
                qb, qn, qe, qu = classes(table)
                if kind == "nemotron_h_mtp":
                    if not qb.get("head"):
                        qb["head"], qn["head"] = b["head"], n["head"]
                    # A sidecar's layers are its NextN layers: all of them, and their experts.
                    d = pass_of(qb, qn, ["body", "mtp", "head"], ["experts", "mtp_experts"],
                                qe or experts, qu or used)
                else:
                    share = GEMMA_ASSISTANT_HEAD_ROWS / max(1, int(table.get("meta", {}).get(
                        "vocab_size", 262144)))
                    d = {"bytes": int(qb.get("body", 0) + qb.get("head", 0) * share),
                         "params": int(qn.get("body", 0) + qn.get("head", 0) * share)}
            elif b.get("mtp") or b.get("mtp_experts"):
                d = pass_of(b, n, ["mtp", "head"], ["mtp_experts"], experts, used)
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
            qb, qn, _, _ = classes(table)
            d = {"bytes": int(qb.get("body", 0)), "params": int(qn.get("body", 0)), "rows": block,
                 "per_draft": False, "head": {"bytes": int(b["head"]), "params": int(n["head"])}}
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
               35: (256, 66), 36: (256, 66), 37: (256, 66), 38: (256, 66), 39: (32, 17)}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _range(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _gguf(url: str) -> tuple[dict, dict, dict]:
    size = 8 << 20
    while True:
        buf = _range(url, 0, size - 1)
        try:
            return parse_gguf(buf)
        except (IndexError, struct.error):
            if len(buf) < size:
                raise
            size *= 4


def parse_gguf(buf: bytes) -> tuple[dict, dict, dict]:
    """Metadata, bytes and parameters per tensor of a GGUF header (`buf` its first bytes)."""
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
    infos = []
    for _ in range(count):
        name = text()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        kind, offset = rd("<I"), rd("<Q")
        e = 1
        for d in dims:
            e *= d
        infos.append((offset, name, kind, e))
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
    return meta, tensors, elems


def parse_safetensors_header(header: dict, bits: int) -> tuple[dict, dict]:
    """Bytes and logical parameters per tensor (packed values counted; scales and zero points none)."""
    tensors, elems = {}, {}
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
    return tensors, elems


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
    q = cfg.get("quantization_config") or {}
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
    tensors, elems = {}, {}
    for f in files:
        if f.endswith(".gguf"):
            m, t, e = _gguf(base + f)
            meta.update(m)
        else:
            n = struct.unpack("<Q", _range(base + f, 0, 7))[0]
            t, e = parse_safetensors_header(json.loads(_range(base + f, 8, 8 + n - 1)), bits)
        tensors.update(t)
        elems.update(e)
    if not tensors:
        raise ValueError(f"no weight files found at {repo_at}")
    return {"meta": meta, "tensors": tensors, "elems": elems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    before = copy.deepcopy(registry)
    tables = json.loads(args.tables.read_text(encoding="utf-8")) if args.tables and args.tables.exists() else {}
    if args.fetch:
        for p in registry.get("pairs", []):
            wanted = [(p["target_repo"], p["target_repo"].split(" ")[0], None, p["target"].split(","))]
            for kind in (p.get("drafters") or {}):
                key = drafter_key(registry, p, kind)
                parts = key[len("drafter:"):].split(":")
                wanted.append((key, parts[0], parts[1:] or None, None))
            for key, at, files, shas in wanted:
                if key in tables:
                    continue
                try:
                    tables[key] = fetch(at, files, shas)
                    print(f"{key}: {len(tables[key]['tensors'])} tensors", file=sys.stderr)
                except Exception as error:  # noqa: BLE001 - one model's failure skips it
                    print(f"{key}: {error}", file=sys.stderr)
            if args.tables:
                args.tables.write_text(json.dumps(tables), encoding="utf-8")
    done = 0
    for p in registry.get("pairs", []):
        board = LAB_BOARDS.get(p["tune"]["key"]["gpu"])
        if board:
            p["lab_board"] = board
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
