"""ninferno-convert attaching speculative drafters, against a local hub (fetch_server.py).

    python drafters_test.py NINFERNO_CONVERT [REPOSITORY]

Small synthetic Gemma-4-shaped checkpoints stand in for a target, its MTP assistant and its
DFlash drafter; a registry file ($NINFERNO_DRAFTER_REGISTRY) names them. Checks that --fetch
attaches, deterministically and without any search:

  * the registry's drafters for the target's repository id (fetched at the pinned revision,
    converted, listed with the published profiles in <artifact>.speculation.json);
  * the same drafters for a fine-tune whose model card declares the target as its base model;
  * the drafters a checkpoint declares itself in ninferno_speculation.json (the bundle wins);
  * nothing for a model neither declares nor the registry knows (the manifest says so, with the
    prompt-lookup fallback), and nothing at all with --no-drafters;
  * a declared drafter whose vocabulary or geometry does not fit the target is rejected before
    its weights are downloaded;
  * --plan-drafters prints the plan without touching the network;
  * --attach-drafters adds drafters to an existing artifact, from given drafter artifacts;
  * a drafter declaring "variants" is also converted in NInferno's own quantizations (--drafter-quant)
    from the same download, each attached as "<name>.<variant>"; --drafter-quant on a target is
    refused.

Exits 77 (skipped) when the interpreter lacks torch (the synthetic checkpoints need it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_server  # noqa: E402

try:
    import synthetic_gemma4  # noqa: E402  (exits 77 without torch)
except SystemExit:
    sys.exit(77)

COMMIT = "c0ffee0123456789abcdef0123456789abcdef01"


def files_of(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors: list[str] = []

    def case(name, ok, detail=""):
        print(f"{name}: {'ok' if ok else 'FAILED'}", flush=True)
        if not ok:
            errors.append(f"{name} {detail}")

    with tempfile.TemporaryDirectory(prefix="ninferno-drafters-") as tmp:
        work = Path(tmp)
        target, assistant, dflash = work / "target", work / "assistant", work / "dflash"
        synthetic_gemma4.text_checkpoint(target, moe=True, layout="ct", nested=True, tied=True, seed=1)
        synthetic_gemma4.assistant_checkpoint(assistant)
        (assistant / "ranking.i64").unlink()
        synthetic_gemma4.dflash_checkpoint(dflash)
        # The synthetic drafter taps a 4-layer target; this target has 2 layers.
        config = json.loads((dflash / "config.json").read_text(encoding="utf-8"))
        config["num_target_layers"] = 2
        config["dflash_config"]["target_layer_ids"] = [0, 1]
        (dflash / "config.json").write_text(json.dumps(config), encoding="utf-8")
        # A drafter of another vocabulary (one more token) and one of another depth.
        foreign = work / "foreign"
        synthetic_gemma4.assistant_checkpoint(foreign)
        (foreign / "ranking.i64").unlink()
        tokenizer = json.loads((foreign / "tokenizer.json").read_text(encoding="utf-8"))
        tokenizer["added_tokens"].append({"id": 511, "content": "<other>", "special": True})
        (foreign / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
        narrow = work / "narrow"
        synthetic_gemma4.dflash_checkpoint(narrow)
        config = json.loads((narrow / "config.json").read_text(encoding="utf-8"))
        config["num_target_layers"] = 3
        config["dflash_config"]["target_layer_ids"] = [0, 1]
        (narrow / "config.json").write_text(json.dumps(config), encoding="utf-8")

        hub = fetch_server.Hub()
        hub.add_repo("org/gemma-tiny", files_of(target))
        finetune = files_of(target)
        finetune["README.md"] = b"---\nlicense: apache-2.0\nbase_model:\n- org/gemma-tiny\n---\n# a fine-tune\n"
        hub.add_repo("user/gemma-tiny-finetune", finetune)
        hub.add_repo("org/gemma-tiny-assistant", files_of(assistant))
        hub.add_repo("lab/gemma-tiny-DFlash", files_of(dflash))
        hub.add_repo("evil/gemma-tiny-assistant", files_of(foreign))
        hub.add_repo("evil/gemma-tiny-DFlash", files_of(narrow))
        declared = files_of(target)
        declared["ninferno_speculation.json"] = json.dumps({
            "drafters": [{"name": "dflash", "kind": "dflash", "repo": "lab/gemma-tiny-DFlash",
                          "revision": COMMIT, "recipe": "gemma4_dflash_q8"}],
            "profiles": [{"gpu_class": "*", "default": {"config": "dflash:7", "drafter": "dflash"}}],
        }).encode()
        hub.add_repo("vendor/gemma-tiny-bundle", declared)
        hub.add_repo("someone/unknown-gemma", files_of(target))
        hub.add_repo("org/gemma-mismatch", files_of(target))
        hub.add_repo("org/gemma-tiny-variants", files_of(target))

        profile = {"gpu_class": "*", "applies_to": [],
                   "default": {"config": "dflash:7", "drafter": "dflash"},
                   "non_thinking": {"config": "mtp:1", "drafter": "assistant"},
                   "thinking": {"config": "dflash:11", "drafter": "dflash"}}
        registry = {
            "schema": 1, "published": "test",
            "kinds": {"gemma4_assistant": {"engines": ["gemma4"], "architectures": ["Gemma4AssistantForCausalLM"],
                                           "recipe": "gemma4_assistant_q8"},
                      "dflash": {"engines": ["gemma4", "qwen3_5"], "architectures": ["DFlashDraftModel"],
                                 "recipe": "gemma4_dflash_q8"},
                      "mtp": {"engines": ["qwen3_5"]}},
            "models": [
                {"id": "gemma-tiny", "engine": "gemma4",
                 "match": {"repos": ["org/gemma-tiny"], "lineage": ["org/gemma-tiny"]},
                 "drafters": [
                     {"name": "assistant", "kind": "gemma4_assistant", "repo": "org/gemma-tiny-assistant",
                      "revision": COMMIT, "recipe": "gemma4_assistant_q8"},
                     {"name": "dflash", "kind": "dflash", "repo": "lab/gemma-tiny-DFlash",
                      "revision": COMMIT, "recipe": "gemma4_dflash_q8"}],
                 "profiles": [profile]},
                {"id": "gemma-mismatch", "engine": "gemma4",
                 "match": {"repos": ["org/gemma-mismatch"]},
                 "drafters": [
                     {"name": "assistant", "kind": "gemma4_assistant", "repo": "evil/gemma-tiny-assistant",
                      "revision": COMMIT, "recipe": "gemma4_assistant_q8"},
                     {"name": "dflash", "kind": "dflash", "repo": "evil/gemma-tiny-DFlash",
                      "revision": COMMIT, "recipe": "gemma4_dflash_q8"}],
                 "profiles": [profile]},
                {"id": "gemma-variants", "engine": "gemma4",
                 "match": {"repos": ["org/gemma-tiny-variants"]},
                 "drafters": [
                     {"name": "assistant", "kind": "gemma4_assistant", "repo": "org/gemma-tiny-assistant",
                      "revision": COMMIT, "recipe": "gemma4_assistant_published", "variants": ["q8", "nvfp4"]}],
                 "profiles": [{"gpu_class": "*", "default": {"config": "mtp:2", "drafter": "assistant"}},
                              {"gpu_class": "*", "requires": ["assistant.q8"],
                               "default": {"config": "mtp:4", "drafter": "assistant.q8"}}]},
            ],
        }
        registry_path = work / "registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")

        server, base = fetch_server.start(hub)
        environment = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        environment.update({"HF_ENDPOINT": base, "NINFERNO_DRAFTER_REGISTRY": str(registry_path)})

        def served(repo):
            stats = json.loads(urllib.request.urlopen(base + "/stats").read())
            return [r[0] for r in stats["requests"] if r[0].startswith(f"/{repo}/resolve/")]

        def reset():
            urllib.request.urlopen(base + "/reset").read()

        def convert(*args, check=True):
            result = subprocess.run([str(native), *args, "--recipe", "gemma4_26b_a4b_nvfp4"], cwd=repository,
                                    capture_output=True, text=True, env=environment)
            if check and result.returncode:
                errors.append(f"{args}: failed\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
            return result

        def artifact_of(cache, repo):
            return (work / cache / f"models--{repo.replace('/', '--')}" / "ninferno" / COMMIT /
                    "model.gemma4_26b_a4b_nvfp4.ninferno")

        def manifest_of(cache, repo):
            path = Path(str(artifact_of(cache, repo)) + ".speculation.json")
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        # 1. The registry's drafters for the target's repository id.
        convert("--fetch", "org/gemma-tiny", "--cache", str(work / "c1"))
        m = manifest_of("c1", "org/gemma-tiny")
        drafters = {d["name"]: d for d in (m or {}).get("drafters", [])}
        case("registry by repository id",
             m is not None and m["source"] == "registry" and m["matched_by"] == "repo"
             and m["entry"] == "gemma-tiny" and m["fallback"] == "ngram"
             and drafters.get("assistant", {}).get("status") == "converted"
             and drafters.get("dflash", {}).get("status") == "converted"
             and all(Path(d["artifact"]).exists() for d in drafters.values())
             and m["profiles"][0]["default"]["config"] == "dflash:7", json.dumps(m)[:800])
        # The assistant ships a tokenizer (checked before the fetch); the DFlash drafter does not
        # (vocab size first, its converted artifact's vocabulary after).
        case("drafters carry the target's vocabulary",
             drafters and all(any(c.startswith("ok converted drafter's vocabulary fingerprint")
                                  for c in d["checks"]) for d in drafters.values())
             and any(c.startswith("ok vocabulary fingerprint") for c in drafters["assistant"]["checks"]),
             json.dumps([d["checks"] for d in drafters.values()]))

        # 2. A fine-tune declaring the target as its base model.
        convert("--fetch", "user/gemma-tiny-finetune", "--cache", str(work / "c2"))
        m = manifest_of("c2", "user/gemma-tiny-finetune")
        case("registry by declared base model",
             m is not None and m["source"] == "registry" and m["matched_by"] == "lineage"
             and all(d.get("status") == "converted" for d in m["drafters"]), json.dumps(m)[:800])

        # 3. The bundle's own declaration wins.
        convert("--fetch", "vendor/gemma-tiny-bundle", "--cache", str(work / "c3"))
        m = manifest_of("c3", "vendor/gemma-tiny-bundle")
        case("bundle declaration",
             m is not None and m["source"] == "bundle" and [d["name"] for d in m["drafters"]] == ["dflash"]
             and m["drafters"][0]["status"] == "converted", json.dumps(m)[:800])

        # 4. Unknown to the registry: no drafter, the fallback named; --no-drafters: no manifest.
        convert("--fetch", "someone/unknown-gemma", "--cache", str(work / "c4"))
        m = manifest_of("c4", "someone/unknown-gemma")
        case("unknown model", m is not None and m["source"] == "none" and m["drafters"] == []
             and m["fallback"] == "ngram", json.dumps(m)[:500])
        convert("--fetch", "org/gemma-tiny", "--cache", str(work / "c5"), "--no-drafters")
        case("--no-drafters", artifact_of("c5", "org/gemma-tiny").exists()
             and manifest_of("c5", "org/gemma-tiny") is None)

        # 5. Drafters that do not fit are rejected before their weights are fetched.
        reset()
        result = convert("--fetch", "org/gemma-mismatch", "--cache", str(work / "c6"), check=False)
        m = manifest_of("c6", "org/gemma-mismatch")
        weights = [p for repo in ("evil/gemma-tiny-assistant", "evil/gemma-tiny-DFlash")
                   for p in served(repo) if p.endswith(".safetensors")]
        status = {d["name"]: d.get("status", "") for d in (m or {}).get("drafters", [])}
        case("incompatible drafters rejected",
             result.returncode == 1 and m is not None
             and status.get("assistant", "").startswith("incompatible: vocabulary fingerprint")
             and status.get("dflash", "").startswith("incompatible: num_target_layers")
             and not weights, f"{status} {weights} rc={result.returncode}")

        # 6. The plan, offline: an unreachable hub is never contacted.
        offline = {**environment, "HF_ENDPOINT": "http://127.0.0.1:9"}
        result = subprocess.run([str(native), "--plan-drafters", "--model", str(target), "--model-id",
                                 "org/gemma-tiny"], cwd=repository, capture_output=True, text=True, env=offline)
        plan = json.loads(result.stdout) if result.returncode == 0 else {}
        case("plan without network",
             plan.get("source") == "registry" and [d["name"] for d in plan.get("drafters", [])] == ["assistant", "dflash"],
             result.stderr[-500:])

        # 7. Attach to an existing artifact from given drafter artifacts (no fetch at all).
        existing = artifact_of("c5", "org/gemma-tiny")
        given = {d["name"]: d["artifact"] for d in manifest_of("c1", "org/gemma-tiny")["drafters"]}
        result = subprocess.run([str(native), "--attach-drafters", "--target-artifact", str(existing),
                                 "--model-id", "org/gemma-tiny",
                                 "--drafter-artifact", f"assistant={given['assistant']}",
                                 "--drafter-artifact", f"dflash={given['dflash']}"],
                                cwd=repository, capture_output=True, text=True, env=offline)
        m = manifest_of("c5", "org/gemma-tiny")
        case("attach given drafters",
             result.returncode == 0 and m is not None
             and all(d.get("status") == "attached" for d in m["drafters"]), result.stdout[-800:] + result.stderr[-500:])

        # 8. Drafter variants: NInferno's own quantizations of the assistant, converted from its one
        #    download beside the published one and attached under their own names.
        reset()
        convert("--fetch", "org/gemma-tiny-variants", "--cache", str(work / "c7"))
        m = manifest_of("c7", "org/gemma-tiny-variants")
        drafters = {d["name"]: d for d in (m or {}).get("drafters", [])}
        weights = [p for p in served("org/gemma-tiny-assistant") if p.endswith(".safetensors")]
        reports = {name: json.loads(Path(d["artifact"] + ".conversion.json").read_text(encoding="utf-8"))
                   for name, d in drafters.items() if d.get("artifact")}
        quant = {name: r["provenance"].get("drafter_quantization") for name, r in reports.items()}
        formats = {name: set(r.get("formats", {})) for name, r in reports.items()}
        case("drafter variants built from one download",
             m is not None and list(drafters) == ["assistant", "assistant.q8", "assistant.nvfp4"]
             and all(d.get("status") == "converted" for d in drafters.values())
             and drafters["assistant.q8"].get("quant") == "q8"
             and len({d["artifact"] for d in drafters.values()}) == 3
             and quant == {"assistant": None, "assistant.q8": "q8", "assistant.nvfp4": "nvfp4"}
             and "q8_g32_fp16" in formats["assistant.q8"] and "q8_g32_fp16" not in formats["assistant"]
             and len(weights) == 1,
             f"{json.dumps(m)[:900]} {quant} {formats} weights={weights}")
        result = subprocess.run([str(native), "--model", str(target), "--recipe", "gemma4_26b_a4b_nvfp4",
                                 "--drafter-quant", "q8", "--out", str(work / "refused.ninferno")],
                                cwd=repository, capture_output=True, text=True, env=environment)
        case("--drafter-quant refused on a target",
             result.returncode != 0 and "--drafter-quant applies to drafters" in result.stderr
             and not (work / "refused.ninferno").exists(), result.stderr[-500:])

        server.shutdown()
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
