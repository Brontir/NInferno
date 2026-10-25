"""ninferno-convert --fetch against a local hub (fetch_server.py).

    python fetch_test.py NINFERNO_CONVERT [REPOSITORY]

Checks the Hugging Face flow (listing, 302 + X-Linked-Etag, cache layout and reuse, file
choice), parallel range transfers, resume from another downloader's prefix and from this
downloader's own progress record, stall abort and resume, a digest mismatch (nothing published),
an expired signed URL, a server without ranges, gated repositories ($HF_TOKEN sent to the hub
only and never printed), GGUF quantization choice, ModelScope, plain URLs, a fetched GGUF
converted without --out, its tokenizer files taken from the base model it names, an unsupported
architecture refused from config.json before any weight moves, and $HF_HUB_OFFLINE.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_server  # noqa: E402

MIB = 1 << 20


def blob(size, seed):
    return random.Random(seed).randbytes(size)


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    hub = fetch_server.Hub()
    weights = blob(20 * MIB + 12345, 1)
    mtp = blob(MIB + 77, 21)
    tokenizer = json.dumps({"model": {"vocab": {f"t{i}": i for i in range(8)}}, "added_tokens": []}).encode()
    hub.add_repo("org/tiny", {
        "config.json": b'{"architectures": ["Qwen3_5ForCausalLM"]}',
        "model.safetensors": weights,
        # A part the index names in its own file (Qwen3.5's MTP layers in RedHatAI checkpoints).
        "model_mtp.safetensors": mtp,
        "tokenizer.json": tokenizer,
        "tokenizer_config.json": b"{}",
        "README.md": b"# tiny",
        "original/consolidated.safetensors": blob(2 * MIB, 2),
        # Mistral repositories ship a second, Mistral-format copy of the weights at top level.
        "consolidated.safetensors": blob(2 * MIB, 9),
    }, xet_small=True)
    stall = blob(6 * MIB, 3)
    corrupt = blob(3 * MIB, 4)
    expire = blob(3 * MIB, 5)
    norange = blob(3 * MIB, 6)
    for name, data, behavior in (("org/stall", stall, {"stall_once"}), ("org/corrupt", corrupt, {"corrupt"}),
                                 ("org/expire", expire, {"expire_once"}), ("org/norange", norange, {"no_range"})):
        hub.add_repo(name, {"config.json": b"{}", "model.safetensors": data})
        hub.behavior[hub.sha(data)] = behavior
    hub.add_repo("org/gated", {"config.json": b"{}", "model.safetensors": blob(2 * MIB, 7)}, gated=True)
    unsupported = blob(4 * MIB, 40)
    hub.add_repo("org/unsupported", {"config.json": b'{"architectures": ["Lfm2ForCausalLM"]}',
                                     "model.safetensors": unsupported})
    shared = blob(8 * MIB, 60)
    hub.add_repo("org/shared", {"config.json": b"{}", "model.safetensors": shared})
    hub.behavior[hub.sha(shared)] = {"slow"}
    dropped = blob(12 * MIB, 8)
    hub.add_repo("org/drop", {"config.json": b"{}", "model.safetensors": dropped})
    hub.behavior[hub.sha(dropped)] = {"drop_once"}
    hub.add_repo("org/busy", {"config.json": b"{}", "model.safetensors": blob(2 * MIB, 61)})
    hub.repos["org/busy"]["busy_once"] = True
    ggufs = {name: blob(1536 + i, 10 + i) for i, name in enumerate(
        ["m-Q4_K_M.gguf", "m-Q8_0.gguf", "m-UD-Q4_K_XL.gguf", "mmproj-F16.gguf",
         "m-Q6_K-00001-of-00002.gguf", "m-Q6_K-00002-of-00002.gguf", "m-MTP-ONLY-Q8_0.gguf",
         "mtp-m-Q4_0.gguf"])}
    hub.add_repo("org/gguf", ggufs)
    hub.add_repo("org/gguf-twice", {"m-Q8_0.gguf": blob(1600, 30), "mtp-m-Q4_0.gguf": blob(1601, 31),
                                     "old/mtp-m-Q4_0.gguf": blob(1602, 32)})
    ggufhead = {"m-Q4_K_M.gguf": blob(3 * MIB + 5, 50), "m-Q8_0.gguf": blob(2 * MIB, 51)}
    hub.add_repo("org/ggufhead", ggufhead)
    hub.repos["org/ggufhead"]["no_sizes"] = True
    # A small vision projector without a quantization in its name, beside a model.
    hub.add_repo("org/smallproj", {"m-Q4_0.gguf": blob(MIB, 52), "mmproj-model.gguf": blob(MIB, 53)})
    hub.repos["files"] = {"files": {"plain.bin": blob(5 * MIB, 20)}, "lfs": set(), "commit": "", "gated": False}

    server, base = fetch_server.start(hub)
    errors = []
    with tempfile.TemporaryDirectory(prefix="ninferno-fetch-") as tmp:
        work = Path(tmp)

        def stats():
            return json.loads(urllib.request.urlopen(base + "/stats").read())

        def reset():
            urllib.request.urlopen(base + "/reset").read()

        def fetch(spec, *extra, cache="cache", env=None, check=True):
            environment = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
            environment.update({"HF_ENDPOINT": base, "MODELSCOPE_ENDPOINT": base, **(env or {})})
            command = [str(native), "--fetch", spec, "--fetch-only", "--cache", str(work / cache),
                       "--chunk-bytes", str(MIB), *extra]
            result = subprocess.run(command, cwd=repository, capture_output=True, text=True, env=environment)
            if check and result.returncode:
                errors.append(f"{spec} {extra}: failed\n{result.stderr[-2000:]}")
            return result

        def case(name, ok):
            print(f"{name}: {'ok' if ok else 'FAILED'}", flush=True)
            if not ok:
                errors.append(name)

        repo_dir = work / "cache" / "models--org--tiny"
        snapshot = repo_dir / "snapshots" / "c0ffee0123456789abcdef0123456789abcdef01"

        # 1. Listing, file choice, parallel ranges, digests, cache layout.
        reset()
        result = fetch("org/tiny", "--streams", "4")
        chosen = sorted(p.name for p in snapshot.iterdir()) if snapshot.exists() else []
        sha = hub.sha(weights)
        case("layout and choice",
             chosen == ["config.json", "model.safetensors", "model_mtp.safetensors", "tokenizer.json",
                        "tokenizer_config.json"]
             and (snapshot / "model_mtp.safetensors").read_bytes() == mtp
             and (snapshot / "model.safetensors").is_symlink()
             and (snapshot / "model.safetensors").read_bytes() == weights
             and (repo_dir / "blobs" / sha).exists()
             and (repo_dir / "refs" / "main").read_text(encoding="utf-8") == "c0ffee0123456789abcdef0123456789abcdef01"
             and "verified" in result.stdout)
        ranges = [r for r in stats()["requests"] if r[0].startswith("/cdn/")]
        case("parallel ranges", len(ranges) >= 21 and all(r[1] and r[1].startswith("bytes=") for r in ranges))

        # 2. Reuse: nothing transferred again.
        reset()
        fetch("org/tiny")
        case("reuse", not any(r[0].startswith("/cdn/") for r in stats()["requests"]))

        # 2a. A hub that answers 503 once per file is asked again.
        case("busy hub retried", fetch("org/busy", check=False).returncode == 0)

        # 2b. Two processes fetching one repository into one cache at once take turns on the
        # file's lock: both succeed and the blob is transferred once.
        reset()
        environment = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        environment.update({"HF_ENDPOINT": base, "MODELSCOPE_ENDPOINT": base})
        both = [subprocess.Popen([str(native), "--fetch", "org/shared", "--fetch-only", "--cache",
                                  str(work / "cache-shared"), "--chunk-bytes", str(MIB), "--streams", "2"],
                                 cwd=repository, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                 env=environment) for _ in range(2)]
        outputs = [p.communicate(timeout=300) for p in both]
        shared_blob = work / "cache-shared" / "models--org--shared" / "blobs" / hub.sha(shared)
        served = stats().get("served", {}).get(hub.sha(shared), 0)
        # One of them waited on the other's lock (the transfer is slow enough that they overlap),
        # under huggingface_hub's lock path, so Python downloaders are excluded as well.
        waited = any("another process is fetching it; waiting" in o[1] + o[0] for o in outputs)
        hf_lock = work / "cache-shared" / ".locks" / "models--org--shared" / (hub.sha(shared) + ".lock")
        case("two processes, one cache",
             all(p.returncode == 0 for p in both) and shared_blob.exists()
             and shared_blob.read_bytes() == shared and served <= len(shared) + MIB
             and waited and hf_lock.exists())
        if not all(p.returncode == 0 for p in both):
            errors.append("concurrent fetch: " + " | ".join(o[1][-500:] for o in outputs))

        # 3. Resume from another downloader's prefix (a single-stream .incomplete).
        (repo_dir / "blobs" / sha).unlink()
        (snapshot / "model.safetensors").unlink()
        (repo_dir / "blobs" / (sha + ".incomplete")).write_bytes(weights[: 7 * MIB])
        reset()
        result = fetch("org/tiny", "--streams", "2")
        served = stats()["served"].get(sha, 0)
        case("resume from a prefix",
             (snapshot / "model.safetensors").read_bytes() == weights
             and served == len(weights) - 7 * MIB and "resuming" in result.stdout)

        # 3b. Resume from this downloader's own record: a dropped connection with no retries
        # fails the run, which leaves the partial file and its per-chunk progress; the next run
        # transfers only what is missing.
        drop_sha = hub.sha(dropped)
        first = fetch("org/drop", "--streams", "1", "--retries", "0", "--chunk-bytes", str(3 * MIB),
                      check=False)
        blobs = work / "cache" / "models--org--drop" / "blobs"
        left = (blobs / (drop_sha + ".incomplete.progress")).exists()
        reset()
        second = fetch("org/drop", "--streams", "2", "--chunk-bytes", str(3 * MIB))
        served = stats()["served"].get(drop_sha, 0)
        case("resume from the progress record",
             first.returncode != 0 and left and second.returncode == 0
             and (blobs / drop_sha).read_bytes() == dropped and 0 < served <= len(dropped) - MIB)

        # 4. Stall: one stream stops sending; it is aborted by the speed floor and resumed.
        reset()
        begin = time.monotonic()
        result = fetch("org/stall", "--streams", "2", "--stall-seconds", "2", "--min-speed", "100000",
                       "--chunk-bytes", str(3 * MIB))
        stalled = work / "cache" / "models--org--stall" / "snapshots" / "c0ffee0123456789abcdef0123456789abcdef01"
        case("stall abort and resume",
             (stalled / "model.safetensors").exists() and (stalled / "model.safetensors").read_bytes() == stall
             and "interrupted" in result.stdout and time.monotonic() - begin < 25)

        # 5. A digest mismatch publishes nothing.
        result = fetch("org/corrupt", check=False)
        corrupt_dir = work / "cache" / "models--org--corrupt"
        case("sha256 mismatch",
             result.returncode != 0 and "sha256 mismatch" in result.stderr
             and not (corrupt_dir / "blobs" / hub.sha(corrupt)).exists()
             and not (corrupt_dir / "blobs" / (hub.sha(corrupt) + ".incomplete")).exists())

        # 6. An expired signed URL is resolved again.
        result = fetch("org/expire", "--streams", "1")
        case("expired URL renewed", result.returncode == 0 and "renewed" in result.stdout)

        # 7. A server without ranges: one stream.
        result = fetch("org/norange", "--streams", "4")
        norange_file = work / "cache" / "models--org--norange" / "snapshots" / "c0ffee0123456789abcdef0123456789abcdef01" / "model.safetensors"
        case("no ranges", result.returncode == 0 and norange_file.exists() and norange_file.read_bytes() == norange)

        # 8. Gated: refused without a token; with one, the hub gets it, the CDN never, logs never.
        result = fetch("org/gated", check=False)
        refused = result.returncode != 0 and "HF_TOKEN" in result.stderr
        reset()
        result = fetch("org/gated", env={"HF_TOKEN": fetch_server.TOKEN})
        requests = stats()["requests"]
        case("gated repository",
             refused and result.returncode == 0
             and fetch_server.TOKEN not in result.stdout + result.stderr
             and not any(r[0] == "CREDENTIALS SENT TO CDN" for r in requests)
             and any(r[2] for r in requests if r[0].startswith("/api/")))

        # 9. GGUF choice: preference (Q8_0), a named quantization with its split parts, ambiguity,
        # a width.
        # An MTP-only sidecar (m-MTP-ONLY-Q8_0.gguf, first by name among the Q8_0 files) is never
        # the model.
        gguf_snapshot = work / "cache" / "models--org--gguf" / "snapshots" / "c0ffee0123456789abcdef0123456789abcdef01"
        fetch("org/gguf")
        default_pick = sorted(p.name for p in gguf_snapshot.iterdir())
        fetch("org/gguf", "--quant", "q6_k", cache="cache-q6")
        q6 = sorted(p.name for p in (work / "cache-q6" / "models--org--gguf" / "snapshots" /
                                      "c0ffee0123456789abcdef0123456789abcdef01").iterdir())
        fetch("org/gguf", "--quant", "Q8_0", cache="cache-q8")
        q8 = sorted(p.name for p in (work / "cache-q8" / "models--org--gguf" / "snapshots" /
                                      "c0ffee0123456789abcdef0123456789abcdef01").iterdir())
        ambiguous = fetch("org/gguf", "--quant", "Q4_K", cache="cache-q4", check=False)
        # A sidecar named whole (a drafter's "gguf" in the registry: mtp-<model>-Q4_0) is that file.
        fetch("org/gguf", "--quant", "mtp-m-Q4_0", cache="cache-mtp")
        sidecar = sorted(p.name for p in (work / "cache-mtp" / "models--org--gguf" / "snapshots" /
                                          "c0ffee0123456789abcdef0123456789abcdef01").iterdir())
        # A projector named whole is still not a model (it comes with the model, as its projector).
        projector = fetch("org/gguf", "--quant", "mmproj-F16", cache="cache-mmproj", check=False)
        # The same sidecar name in two directories is ambiguous, not the first listed.
        twice = fetch("org/gguf-twice", "--quant", "mtp-m-Q4_0", cache="cache-twice", check=False)
        # A width: the largest model of that many bits (UD-Q4_K_XL over Q4_K_M), never the
        # MTP-only sidecar (8-bit and larger than Q8_0).
        picked = {}
        for width in ("4bit", "8-bit"):
            fetch("org/gguf", "--quant", width, cache=f"cache-{width}")
            picked[width] = sorted(p.name for p in (work / f"cache-{width}" / "models--org--gguf" / "snapshots" /
                                                     "c0ffee0123456789abcdef0123456789abcdef01").iterdir())
        case("gguf quantization choice",
             default_pick == ["m-Q8_0.gguf"]
             and q6 == ["m-Q6_K-00001-of-00002.gguf", "m-Q6_K-00002-of-00002.gguf"]
             and q8 == ["m-Q8_0.gguf"]
             and sidecar == ["mtp-m-Q4_0.gguf"] and projector.returncode != 0
             and twice.returncode != 0 and "old/mtp-m-Q4_0.gguf" in twice.stderr
             and picked == {"4bit": ["m-UD-Q4_K_XL.gguf"], "8-bit": ["m-Q8_0.gguf"]}
             and ambiguous.returncode != 0 and "m-UD-Q4_K_XL.gguf" in ambiguous.stderr
             and "MTP-ONLY" not in ambiguous.stderr)

        # 10. ModelScope.
        result = fetch("org/tiny", "--hub", "modelscope", "--streams", "3", cache="cache-ms")
        ms = work / "cache-ms" / "modelscope" / "models--org--tiny" / "snapshots" / "master" / "model.safetensors"
        case("modelscope", result.returncode == 0 and ms.exists() and ms.read_bytes() == weights)
        # Pinned to a commit: ModelScope lists each file's own last revision, not the resolved
        # commit, so the pin is not checked against it (Hugging Face resolves it, and is checked).
        pinned = "0123456789abcdef0123456789abcdef01234567"
        hub.repos["org/tiny"]["file_revision"] = "f" * 40
        result = fetch(f"org/tiny@{pinned}", "--hub", "modelscope", cache="cache-ms-pinned", check=False)
        hub.repos["org/tiny"].pop("file_revision")
        case("modelscope pinned commit", result.returncode == 0)
        result = fetch(f"org/tiny@{pinned}", cache="cache-hf-pinned", check=False)
        case("hugging face pinned commit checked", result.returncode != 0 and "resolved the pinned commit" in result.stderr)

        # A GGUF file's first bytes (fetch_head_probe) from a hub that does not say how large the
        # file is: the size from the answer's Content-Range; a file shorter than asked for, whole.
        if len(sys.argv) > 3:
            probe = Path(sys.argv[3]).resolve()

            def head(quant, count):
                environment = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
                environment["HF_ENDPOINT"] = base
                result = subprocess.run([str(probe), "org/ggufhead", quant, str(count), str(work / "cache-head")],
                                        capture_output=True, text=True, env=environment)
                return json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr}

            reset()
            first = head("Q4_K_M", MIB)
            ranges = [r[1] for r in stats()["requests"] if r[0].startswith("/cdn/")]
            case("gguf head", first.get("path") == "m-Q4_K_M.gguf" and first.get("size") == len(ggufhead["m-Q4_K_M.gguf"])
                 and first.get("bytes") == MIB and ranges == [f"bytes=0-{MIB - 1}"])
            whole = head("Q8_0", 8 * MIB)
            case("gguf head of a short file", whole.get("size") == 2 * MIB and whole.get("bytes") == 2 * MIB)

            # The listing a plan reads counts what the fetch takes: the Transformers-named weights with
            # their MTP part, not a second copy beside them (consolidated.safetensors); a small
            # projector is listed.
            def listed(spec):
                environment = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
                environment["HF_ENDPOINT"] = base
                result = subprocess.run([str(probe), spec, "--list", "0", str(work / "cache-list")],
                                        capture_output=True, text=True, env=environment)
                return json.loads(result.stdout) if result.returncode == 0 else []

            rows = listed("org/tiny")
            weights_row = [r for r in rows if r["format"] == "safetensors"]
            case("listed weights are the fetched ones",
                 len(weights_row) == 1 and "consolidated.safetensors" not in weights_row[0]["files"]
                 and sorted(weights_row[0]["files"]) == ["model.safetensors", "model_mtp.safetensors"]
                 and weights_row[0]["mtp_bytes"] == len(mtp))
            case("a small projector is listed",
                 any("mmproj-model.gguf" in r["files"] for r in listed("org/smallproj")))

        # 11. A plain URL, then its cached copy.
        url = base + "/files/plain.bin"
        first = fetch(url)
        reset()
        second = fetch(url)
        case("plain URL", first.returncode == 0 and second.returncode == 0
             and "/files/plain.bin" not in [r[0] for r in stats()["requests"] if r[1]])

        # 11b. An architecture no recipe converts: refused from config.json, the weights untouched.
        reset()
        environment = {**os.environ, "HF_ENDPOINT": base}
        result = subprocess.run([str(native), "--fetch", "org/unsupported", "--cache", str(work / "cache-u")],
                                cwd=repository, capture_output=True, text=True, env=environment)
        case("unsupported architecture refused before the download",
             result.returncode != 0 and "cannot be converted" in result.stderr
             and "Lfm2ForCausalLM" in result.stderr
             and "supported architectures" in result.stderr
             and stats()["served"].get(hub.sha(unsupported), 0) == 0)

        # 11c. $HF_HUB_OFFLINE: a cached snapshot serves without asking the hub; an uncached one
        # is refused naming the cache.
        reset()
        result = fetch("org/tiny", env={"HF_HUB_OFFLINE": "1"})
        offline_ok = (result.returncode == 0 and "using the cached snapshot" in result.stdout
                      and not [r for r in stats()["requests"] if r[0] != "/stats"])
        result = fetch("org/norange", cache="cache-empty", env={"HF_HUB_OFFLINE": "1"}, check=False)
        case("HF_HUB_OFFLINE", offline_ok and result.returncode != 0
             and "has no snapshot in the cache" in result.stderr)

        # 11d. A hub that cannot be reached: a ref's cached snapshot is refused (the ref may have
        # moved), naming both ways out; the commit it was cached at serves.
        dead = {"HF_ENDPOINT": "http://127.0.0.1:9"}
        result = fetch("org/tiny", env=dead, check=False)
        commit = (work / "cache" / "models--org--tiny" / "refs" / "main").read_text().strip()
        pinned_result = fetch(f"org/tiny@{commit}", env=dead, check=False)
        case("unreachable hub: a ref is refused, its commit serves",
             result.returncode != 0 and "HF_HUB_OFFLINE=1" in result.stderr
             and f"org/tiny@{commit}" in result.stderr
             and pinned_result.returncode == 0 and "using the cached snapshot" in pinned_result.stdout)

        # 12. Fetch a GGUF model and convert it without --out: the recipe is chosen, and the
        # tokenizer files come from the base model the GGUF names (none travel with it).
        try:
            import synthetic_gguf
        except SystemExit:
            synthetic_gguf = None
        if synthetic_gguf is None:
            print("SKIP fetch and convert: no torch or gguf", flush=True)
        else:
            gguf_file, _, _ = synthetic_gguf.qwen_case(work / "qwen-src", False, None, 30,
                                                       base_model="https://huggingface.co/org/qwen-base",
                                                       files=False)
            hub.add_repo("org/qwen-gguf", {"model-Q8_0.gguf": gguf_file.read_bytes(), "README.md": b""})
            # 12a. Whether the engine runs a model, from what the hub shows before a download: a GGUF
            # of another architecture (its header, by a range request), MLX weights (the card's
            # library). Both refused, nothing in the cache.
            other_arch = gguf_file.read_bytes().replace(b"\x06\x00\x00\x00\x00\x00\x00\x00qwen35",
                                                        b"\x06\x00\x00\x00\x00\x00\x00\x00lfm2zz", 1)
            hub.add_repo("org/lfm-gguf", {"model-Q8_0.gguf": other_arch})
            hub.add_repo("org/mlx-model", {"config.json": (work / "qwen-src" / "twin" / "config.json").read_bytes()
                                           if (work / "qwen-src" / "twin" / "config.json").exists()
                                           else b'{"architectures": ["Qwen3ForCausalLM"]}',
                                           "model.safetensors": blob(MIB, 71)})
            hub.repos["org/mlx-model"]["card"] = {"library_name": "mlx", "pipeline_tag": "text-generation"}
            # A repository of supported safetensors and a GGUF of another architecture: --include
            # naming the GGUF is what the fetch takes, so its header decides.
            hub.add_repo("org/mixed-include", {"config.json": b'{"architectures": ["Qwen3ForCausalLM"]}',
                                               "model.safetensors": blob(MIB, 72),
                                               "other-Q8_0.gguf": other_arch})
            for spec, says, extra in (("org/lfm-gguf", "GGUF architecture 'lfm2zz'", []),
                                      ("org/mlx-model", "MLX", []),
                                      ("org/mixed-include", "GGUF architecture 'lfm2zz'", ["--include", "*.gguf"])):
                refused = subprocess.run([str(native), "--fetch", spec, *extra, "--cache", str(work / "cache-support")],
                                         cwd=repository, capture_output=True, text=True, env=environment)
                cached = list((work / "cache-support").glob(f"models--{spec.replace('/', '--')}/snapshots/*/*"))
                ok = (refused.returncode != 0 and "nothing was downloaded" in refused.stderr
                      and says in refused.stderr and not cached)
                if not ok:
                    errors.append(refused.stderr[-600:])
                case(f"{spec} refused before the download: {says}", ok)
            # Without --include the fetch takes its safetensors: the hub's summary of the GGUF
            # beside them is not what decides.
            hub.repos["org/mixed-include"]["card"] = {"gguf": {"architecture": "lfm2zz"}}
            mixed = subprocess.run([str(native), "--fetch", "org/mixed-include", "--cache", str(work / "cache-support")],
                                   cwd=repository, capture_output=True, text=True, env=environment)
            case("a mixed repository's safetensors judged by their config, not the GGUF beside them",
                 "GGUF architecture" not in mixed.stderr)
            synthetic_gguf.resources(work / "qwen-src" / "twin")
            base_files = {name: (work / "qwen-src" / "twin" / name).read_bytes()
                          for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                       "generation_config.json", "config.json")}
            hub.add_repo("org/qwen-base", {**base_files, "model.safetensors": blob(2 * MIB, 31)})
            environment = {**os.environ, "HF_ENDPOINT": base}
            result = subprocess.run([str(native), "--fetch", "org/qwen-gguf", "--cache", str(work / "cache-e2e")],
                                    cwd=repository, capture_output=True, text=True, env=environment)
            out = (work / "cache-e2e" / "models--org--qwen-gguf" / "ninferno" /
                   "c0ffee0123456789abcdef0123456789abcdef01" / "model-Q8_0.qwen3_5_bf16.ninferno")
            report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8")) if out.exists() else {}
            case("fetch and convert",
                 result.returncode == 0 and out.exists() and "tokenizer files from the base model" in result.stdout
                 and "resource_notes" not in report and report.get("gguf", {}).get("architecture") == "qwen35")
            if result.returncode:
                errors.append(result.stderr[-2000:])

            # 12b. Its download removed (as ninferno-ui does) and its branch moved on the hub: a
            # rerun keeps the converted weights with a notice; --accept-converted records them
            # (no notice after); a hub that cannot be reached leaves them; --refetch converts the
            # new commit and removes the old conversion.
            repo_dir = work / "cache-e2e" / "models--org--qwen-gguf"
            for sub in ("blobs", "snapshots"):
                shutil.rmtree(repo_dir / sub, ignore_errors=True)
            old, new = "c0ffee0123456789abcdef0123456789abcdef01", "d" * 40
            hub.repos["org/qwen-gguf"]["commit"] = new

            def again(*extra, env=environment):
                run = subprocess.run([str(native), "--fetch", "org/qwen-gguf", "--cache", str(work / "cache-e2e"),
                                      *extra], cwd=repository, capture_output=True, text=True, env=env)
                if run.returncode:
                    errors.append(f"--fetch org/qwen-gguf {extra}:\n{run.stderr[-1500:]}")
                return run.returncode, run.stdout + run.stderr

            notice = again()
            accepted = again("--accept-converted")
            after = again()
            unreachable = again(env={**environment, "HF_ENDPOINT": "http://127.0.0.1:9"})
            case("a moved branch keeps the converted weights, with a notice",
                 notice[0] == 0 and "org/qwen-gguf@main is now dddddddddddd" in notice[1]
                 and "--refetch downloads the new commit" in notice[1] and "converted earlier" in notice[1])
            case("--accept-converted keeps them without the notice",
                 accepted[0] == 0 and (repo_dir / "ninferno" / old / "accepted.json").exists()
                 and after[0] == 0 and "as accepted earlier" in after[1] and "warning:" not in after[1])
            dead_refetch = subprocess.run([str(native), "--fetch", "org/qwen-gguf", "--cache", str(work / "cache-e2e"),
                                           "--refetch"], cwd=repository, capture_output=True, text=True,
                                          env={**environment, "HF_ENDPOINT": "http://127.0.0.1:9"})
            case("an unreachable hub leaves the converted weights, said; --refetch then fails",
                 unreachable[0] == 0 and "could not ask the hub whether org/qwen-gguf@main moved" in unreachable[1]
                 and dead_refetch.returncode != 0 and "--refetch: cannot ask the hub" in dead_refetch.stderr)
            refetched = again("--refetch")
            case("--refetch converts the new commit and removes the old conversion",
                 refetched[0] == 0 and (repo_dir / "ninferno" / new / "model-Q8_0.qwen3_5_bf16.ninferno").exists()
                 and not (repo_dir / "ninferno" / old).exists() and "removed the old conversion" in refetched[1])
            # With tokenizer metadata in the GGUF, its own chat template and end-of-sequence ids
            # are kept; the base model supplies only the Hugging Face tokenizer files.
            gguf_file, _, _ = synthetic_gguf.qwen_case(work / "qwen-src2", False, None, 32, rebuilt=True,
                                                       base_model="https://huggingface.co/org/qwen-base")
            hub.add_repo("org/qwen-gguf2", {"model-Q8_0.gguf": gguf_file.read_bytes()})
            result = subprocess.run([str(native), "--fetch", "org/qwen-gguf2", "--cache", str(work / "cache-e2e")],
                                    cwd=repository, capture_output=True, text=True, env=environment)
            out = (work / "cache-e2e" / "models--org--qwen-gguf2" / "ninferno" /
                   "c0ffee0123456789abcdef0123456789abcdef01" / "model-Q8_0.qwen3_5_bf16.ninferno")
            report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8")) if out.exists() else {}
            notes = " ".join(report.get("resource_notes", []))
            # A GGUF repository with a config.json (unsloth's): further quantizations fetched into
            # the snapshot the first ones share are converted too. The preflight read the whole
            # directory and refused the third ("holds several GGUF models").
            twin_config = (work / "qwen-src" / "twin" / "config.json").read_bytes()
            hub.add_repo("org/qwen-gguf3", {"config.json": twin_config, "m-Q8_0.gguf": gguf_file.read_bytes(),
                                             "m-Q4_0.gguf": gguf_file.read_bytes(),
                                             "m-Q6_K.gguf": gguf_file.read_bytes()})
            third = [subprocess.run([str(native), "--fetch", "org/qwen-gguf3", "--quant", quant, "--cache",
                                     str(work / "cache-e2e")], cwd=repository, capture_output=True, text=True,
                                    env=environment) for quant in ("Q8_0", "Q4_0", "Q6_K")]
            base3 = work / "cache-e2e" / "models--org--qwen-gguf3" / "ninferno" / "c0ffee0123456789abcdef0123456789abcdef01"
            case("a third quantization of a cached GGUF repository",
                 all(r.returncode == 0 for r in third) and len(list(base3.glob("m-*.ninferno"))) == 3)
            for r in third:
                if r.returncode:
                    errors.append(r.stderr[-1500:])
            case("fetch and convert, the GGUF's template kept",
                 result.returncode == 0
                 and "base model org/qwen-base: tokenizer.json, tokenizer_config.json" in result.stdout
                 and "chat_template.jinja" in notes and "generation_config.json" in notes
                 and "tokenizer.json" not in notes)
            if result.returncode:
                errors.append(result.stderr[-2000:])

    server.shutdown()
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
