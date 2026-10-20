"""Maintainer tool: propose drafter candidates for a registry entry, for human review.

    python tools/spec/propose_candidates.py REPO [--endpoint URL] [--limit N] [--json]

Never part of the user path: the product attaches only drafters a model bundle declares or the
official registry names (src/convert/drafters.h). This script helps the maintainers who write
that registry. It reads the Hugging Face API only (metadata, config.json, tokenizer.json;
nothing is downloaded to disk):

  1. the target's base_model chain (model card metadata);
  2. searches named by convention over the target and every base: <name>, <name>-assistant,
     -DFlash, -speculator, -EAGLE3, -DSpark (GGUF/MLX/AWQ/GPTQ repackagings are skipped);
  3. for every candidate: its architecture and drafter kind, whether an NInferno engine runs that
     kind, how it links to the target (declared base_model or name only), licence, gating, size,
     and the static checks against the target (vocabulary fingerprint when it ships
     tokenizer.json, else vocab size; hidden size; target layers and taps; block size).

A candidate that passes is still only a candidate: it enters the registry after spec-tune has
measured it on the maintainers' GPUs (acceptance by task family, speed against plain decode).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from vocab_fingerprint import vocab_fingerprint

MARKERS = ("assistant", "dflash", "eagle", "speculator", "dspark", "medusa", "draft")
SKIPPED = ("gguf", "mlx", "awq", "gptq", "exl2", "exl3", "onnx", "openvino")
KINDS = {
    "Gemma4AssistantForCausalLM": ("gemma4_assistant", ["gemma4"]),
    "DFlashDraftModel": ("dflash", ["gemma4", "qwen3_5"]),
    "DFlash2DraftModel": ("dflash2", ["qwen3_5"]),
    "Eagle3DraftModel": ("eagle3", []),
    "LlamaForCausalLMEagle3": ("eagle3", []),
    "DSparkDraftModel": ("dspark", []),
}


def get(endpoint: str, path: str):
    try:
        with urllib.request.urlopen(endpoint + path, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def get_json(endpoint, path):
    body = get(endpoint, path)
    return None if body is None else json.loads(body)


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def text_scope(config: dict) -> dict:
    return config.get("text_config") if isinstance(config.get("text_config"), dict) else config


def engine_of(architecture: str) -> str:
    if architecture.startswith("Gemma4") and "Assistant" not in architecture:
        return "gemma4"
    if architecture.startswith("Qwen3_5"):
        return "qwen3_5"
    return ""


def checks(target: dict, config: dict, drafter_fp: str) -> list[str]:
    out = []
    arch = (config.get("architectures") or [""])[0]
    kind, engines = KINDS.get(arch, ("", []))
    out.append(("ok " if kind else "FAIL ") + f"architecture {arch or '?'} ({kind or 'unknown'})")
    if kind and target["engine"] not in engines:
        out.append(f"FAIL the {target['engine'] or 'target'} engine runs no {kind} drafter")
    scope = text_scope(config)
    vocab = config.get("vocab_size") or scope.get("vocab_size")
    if drafter_fp and target["fingerprint"]:
        out.append(("ok " if drafter_fp == target["fingerprint"] else "FAIL ") + f"vocabulary {drafter_fp}")
    elif vocab:
        out.append(("ok " if vocab == target["vocab"] else "FAIL ") + f"vocab size {vocab} (no fingerprint)")
    if kind == "gemma4_assistant":
        out.append(("ok " if config.get("backbone_hidden_size") == target["hidden"] else "FAIL ")
                   + f"backbone_hidden_size {config.get('backbone_hidden_size')}")
    if kind in ("dflash", "dflash2"):
        out.append(("ok " if config.get("hidden_size") == target["hidden"] else "FAIL ")
                   + f"hidden_size {config.get('hidden_size')}")
        out.append(("ok " if config.get("num_target_layers") == target["layers"] else "FAIL ")
                   + f"num_target_layers {config.get('num_target_layers')}")
        taps = (config.get("dflash_config") or {}).get("target_layer_ids") or []
        out.append(("ok " if taps and all(0 <= t < target["layers"] for t in taps) else "FAIL ")
                   + f"target_layer_ids {taps}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ep = args.endpoint.rstrip("/")

    config = json.loads(get(ep, f"/{args.repo}/resolve/main/config.json") or b"{}")
    scope = text_scope(config)
    tokenizer = get(ep, f"/{args.repo}/resolve/main/tokenizer.json")
    arch = (config.get("architectures") or [""])[0]
    target = {"repo": args.repo, "engine": engine_of(arch), "hidden": scope.get("hidden_size"),
              "layers": scope.get("num_hidden_layers"), "vocab": scope.get("vocab_size"),
              "fingerprint": vocab_fingerprint(json.loads(tokenizer))[0] if tokenizer else ""}
    chain = [args.repo]
    while len(chain) < 5:
        meta = get_json(ep, f"/api/models/{chain[-1]}") or {}
        base = (meta.get("cardData") or {}).get("base_model")
        base = base[0] if isinstance(base, list) and base else base
        if not isinstance(base, str) or base.lower() in (c.lower() for c in chain):
            break
        chain.append(base)

    seen = {c.lower() for c in chain}
    found = []
    for name in chain:
        stem = name.split("/", 1)[-1]
        for query in [stem] + [stem + s for s in ("-assistant", "-DFlash", "-speculator", "-EAGLE3", "-DSpark")]:
            items = get_json(ep, "/api/models?" + urllib.parse.urlencode({"search": query, "limit": 100})) or []
            for item in items:
                repo = item.get("id") or item.get("modelId") or ""
                n = normalized(repo.split("/", 1)[-1])
                if (not repo or repo.lower() in seen or normalized(stem) not in n
                        or not any(m in n for m in MARKERS) or any(s in n for s in SKIPPED)):
                    continue
                seen.add(repo.lower())
                found.append(repo)
    candidates = []
    for repo in found[: args.limit]:
        meta = get_json(ep, f"/api/models/{repo}?blobs=true") or {}
        card = meta.get("cardData") or {}
        bases = card.get("base_model") or []
        bases = [bases] if isinstance(bases, str) else bases
        files = {s.get("rfilename"): s.get("size") or 0 for s in meta.get("siblings", [])}
        cfg = json.loads(get(ep, f"/{repo}/resolve/main/config.json") or b"{}")
        fp = ""
        if "tokenizer.json" in files:
            fp = vocab_fingerprint(json.loads(get(ep, f"/{repo}/resolve/main/tokenizer.json")))[0]
        result = checks(target, cfg, fp)
        candidates.append({
            "repo": repo, "revision": meta.get("sha"),
            "link": "base_model" if any(b.lower() in (c.lower() for c in chain) for b in bases) else "name",
            "license": card.get("license"), "gated": meta.get("gated"),
            "bytes": sum(files.values()), "checks": result,
            "passes": not any(c.startswith("FAIL") for c in result)})
    report = {"target": target, "lineage": chain, "candidates": candidates,
              "note": "candidates for human review; measure with spec-tune before adding to the registry"}
    if args.json:
        print(json.dumps(report, indent=1))
        return 0
    print(f"target {args.repo} ({target['engine'] or 'no engine'}), lineage {' <- '.join(chain)}")
    for c in candidates:
        mark = "PASS" if c["passes"] else "fail"
        print(f"  {mark} {c['repo']:<52} link={c['link']:<10} {c['bytes'] / 1e9:5.2f} GB "
              f"license={c['license']} rev={str(c['revision'])[:12]}")
        for line in c["checks"]:
            print(f"        {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
