#!/usr/bin/env python3
"""Writes each registry pair's round facts: the weight bytes and parameters a speculation round's
passes read (src/spec/round_cost.h), from the tensor tables of the target's and its drafters' files,
and the lab board that measured it.

    round_facts.py [--registry R] [--tables T.json] [--fetch] [--alias KEY=OTHER ...]

A tensor table is {"meta": {...}, "tensors": {name: bytes}, "elems": {name: elements}}: a GGUF
header's (its metadata and tensor infos) or a safetensors header's (its "meta" the config.json's
expert counts as expert_count / expert_used_count). --tables reads them from a JSON file keyed by a
pair's target_repo ("org/name@commit QUANT") and by "drafter:org/name" for a separate drafter;
--fetch reads the missing ones from the hub with HTTP range reads (the header only) and adds them
to the --tables file. A pair whose tables are missing keeps no round facts: its rounds stay priced
by the measured ratios (spec/pairs.h).

What a pass reads (round_cost.h PassFacts):
  target  its layers and output head (the input table when tied), its routed experts apart;
          vision / audio towers, the token embedding and MTP layers excluded.
  mtp     the target's own MTP layers and its output head (per draft, one row); a separate MTP
          drafter (Nemotron's) its layers, experts and head (the target's when it has none);
          Gemma 4's assistant its layers and its head over its first 98 304 vocabulary rows
          (the proposal rows the engine scores: choice-notes DECISIONS 53).
  dflash  the drafter's layers and the target's output head, one pass over its block of 16 rows.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import copy
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry_add import write_registry  # noqa: E402

REGISTRY = Path(__file__).resolve().parents[2] / "src/spec/data/drafter_registry.json"
# The lab's boards by GPU class (the tunes' key.gpu): their PCI product ids in
# src/core/data/gpu_facts.json.
LAB_BOARDS = {
    "ad104:60sm": "0x2782",   # GeForce RTX 4070 Ti
    "ad102:128sm": "0x2684",  # GeForce RTX 4090
    "ad106:34sm": "0x2805",   # GeForce RTX 4060 Ti 16GB
    "gb203:70sm": "0x2C05",   # GeForce RTX 5070 Ti
    "gb202:170sm": "0x2B85",  # GeForce RTX 5090
    "ga106:28sm": "0x2503",   # GeForce RTX 3060
    "tu102:68sm": "0x1E04",   # GeForce RTX 2080 Ti
    "gv100:80sm": "0x1DBA",   # Quadro GV100
}
DFLASH_BLOCK = 16
GEMMA_ASSISTANT_HEAD_ROWS = 98304
NOT_TEXT = re.compile(r"(^v\.|^mm\.|visual|vision|audio|^a\.|embed_vision|embed_audio)")


def classes(table: dict) -> tuple[dict, dict, int, int]:
    """Bytes and elements per class (body, head, embed, experts, mtp, mtp_experts), expert counts."""
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
            if "embed_tokens" in name or ("per_layer" in name and "embed" in name):
                c = "embed"
            elif name.startswith("lm_head"):
                c = "head"
            elif name.startswith("mtp."):
                c = "mtp_experts" if ".experts" in name else "mtp"
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
    if eb and experts:
        out.update(expert_bytes=eb, expert_params=int(sum(n.get(c, 0) for c in routed)),
                   experts=experts, experts_used=used)
    return out


def drafter_repo(pair: dict, kind: str) -> str:
    return (pair.get("drafters") or {}).get(kind, "").split("@")[0]


def round_facts(pair: dict, tables: dict, alias: dict) -> dict | None:
    get = lambda key: tables.get(alias.get(key, key))
    target = get(pair["target_repo"])
    if target is None:
        return None
    b, n, experts, used = classes(target)
    out = {"target": pass_of(b, n, ["body", "head"], ["experts"], experts, used), "drafters": {}}
    modes = {m["config"].split(":")[0] for m in pair["tune"]["measurements"] if m["config"] != "plain"}
    kinds = {d.get("kind") for d in pair["tune"].get("drafters", [])}
    for mode in sorted(modes):
        if mode == "mtp":
            if b.get("mtp") or b.get("mtp_experts"):
                d = pass_of(b, n, ["mtp", "head"], ["mtp_experts"], experts, used)
            elif "nemotron_h_mtp" in kinds or "gemma4_assistant" in kinds:
                kind = "nemotron_h_mtp" if "nemotron_h_mtp" in kinds else "gemma4_assistant"
                table = get("drafter:" + drafter_repo(pair, kind))
                if table is None:
                    continue
                qb, qn, _, _ = classes(table)
                if kind == "nemotron_h_mtp":
                    if not qb.get("head") or qb.get("head") == qb.get("embed") and not qb.get("embed"):
                        qb["head"], qn["head"] = b["head"], n["head"]
                    d = pass_of(qb, qn, ["body", "head"], ["experts"], experts, used)
                else:
                    share = GEMMA_ASSISTANT_HEAD_ROWS / max(1, int(table.get("meta", {}).get(
                        "vocab_size", 262144)))
                    d = {"bytes": int(qb.get("body", 0) + qb.get("head", 0) * share),
                         "params": int(qn.get("body", 0) + qn.get("head", 0) * share)}
            else:
                continue
            d.update(rows=1, per_draft=True)
        else:
            table = get("drafter:" + drafter_repo(pair, mode))
            if table is None:
                continue
            qb, qn, _, _ = classes(table)
            d = {"bytes": int(qb.get("body", 0) + b["head"]), "params": int(qn.get("body", 0) + n["head"]),
                 "rows": DFLASH_BLOCK, "per_draft": False}
        if d["bytes"] > 0 and d["params"] > 0:
            out["drafters"][mode] = d
    return out if out["drafters"] and out["target"]["bytes"] > 0 else None


# ---- The hub: tensor tables by HTTP range reads of the headers.
GGML_BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
               9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
               14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
               19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1),
               25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
               35: (256, 66), 36: (256, 66), 37: (256, 66), 38: (256, 66), 39: (32, 17)}


def _range(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _gguf(url: str) -> tuple[dict, dict, dict]:
    size = 8 << 20
    while True:
        buf = _range(url, 0, size - 1)
        try:
            return _parse_gguf(buf)
        except (IndexError, struct.error):
            size *= 4


def _parse_gguf(buf: bytes) -> tuple[dict, dict, dict]:
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
    tensors, elems = {}, {}
    for _ in range(count):
        name = text()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        kind = rd("<I")
        rd("<Q")
        e = 1
        for d in dims:
            e *= d
        per, size = GGML_BLOCKS.get(kind, (1, 2))
        tensors[name], elems[name] = e * size // per, e
    return meta, tensors, elems


def _safetensors(url: str) -> tuple[dict, dict, dict]:
    n = struct.unpack("<Q", _range(url, 0, 7))[0]
    header = json.loads(_range(url, 8, 8 + n - 1))
    tensors, elems = {}, {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        e = 1
        for d in v["shape"]:
            e *= d
        if v["dtype"] == "U8" and k.endswith(".weight"):  # packed 4-bit
            e *= 2
        tensors[k], elems[k] = b - a, e
    return {}, tensors, elems


def fetch(repo_at: str, quant: str) -> dict:
    """The tensor table of org/name@revision, its files of quantization `quant` (GGUF) or its
    safetensors, with config.json's expert counts."""
    repo, _, rev = repo_at.partition("@")
    rev = rev or "main"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}", timeout=60) as r:
        files = [s["rfilename"] for s in json.load(r).get("siblings", [])]
    base = f"https://huggingface.co/{repo}/resolve/{rev}/"
    ggufs = [f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()
             and (not quant or quant.lower() in f.lower())]
    meta, tensors, elems = {}, {}, {}
    for f in ggufs or [f for f in files if f.endswith(".safetensors")]:
        m, t, e = _gguf(base + f) if f.endswith(".gguf") else _safetensors(base + f)
        meta.update(m)
        tensors.update(t)
        elems.update(e)
    if not ggufs and "config.json" in files:
        with urllib.request.urlopen(base + "config.json", timeout=60) as r:
            cfg = json.load(r)
        cfg = cfg.get("text_config", cfg)
        for k, out in (("num_experts", "expert_count"), ("num_local_experts", "expert_count"),
                       ("n_routed_experts", "expert_count"), ("num_experts_per_tok", "expert_used_count")):
            if cfg.get(k):
                meta[out] = cfg[k]
        if cfg.get("vocab_size"):
            meta["vocab_size"] = cfg["vocab_size"]
    return {"meta": meta, "tensors": tensors, "elems": elems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--alias", action="append", default=[])
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    before = copy.deepcopy(registry)
    tables = json.loads(args.tables.read_text(encoding="utf-8")) if args.tables and args.tables.exists() else {}
    alias = dict(a.split("=", 1) for a in args.alias)
    if args.fetch:
        for p in registry.get("pairs", []):
            repo_at, _, quant = p["target_repo"].partition(" ")
            wanted = [(p["target_repo"], repo_at, quant)]
            for kind, rev in (p.get("drafters") or {}).items():
                wanted.append(("drafter:" + rev.split("@")[0], rev, ""))
            for key, at, q in wanted:
                if alias.get(key, key) not in tables:
                    try:
                        tables[key] = fetch(at, q)
                    except Exception as error:  # noqa: BLE001 - one model's failure skips it
                        print(f"{key}: {error}", file=sys.stderr)
        if args.tables:
            args.tables.write_text(json.dumps(tables), encoding="utf-8")
    done = 0
    for p in registry.get("pairs", []):
        board = LAB_BOARDS.get(p["tune"]["key"]["gpu"])
        if board:
            p["lab_board"] = board
        facts = round_facts(p, tables, alias)
        if facts:
            p["round_facts"] = facts
            done += 1
        else:
            p.pop("round_facts", None)
            print(f"{p['target_repo']}: no tables, no round facts", file=sys.stderr)
    args.registry.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.registry}: round facts for {done} of {len(registry.get('pairs', []))} pair(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
