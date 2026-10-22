#!/usr/bin/env python3
"""Measure one server setting the way docs/performance/ does, against a running ninferno-serve
(or, with --engine llama.cpp, a llama.cpp server on the same GGUF file: a reference row).

    python3 tools/bench/perf_matrix.py --port 8080 [--jsonl rows.jsonl --model ORG/NAME ...]

Two workloads, both greedy, each after one untimed warm-up request
(docs/performance/methodology.md):

- long24k_fr2026_6p: six ~24K-token slices of the Federal Register text (September-October 2026,
  new to every model measured) in examples/cli/messages/long_niah_256k.json, each followed by a
  request to summarize it. Prefill = prompt tokens / wall time of a max_tokens=1
  request; decode = completion tokens / wall time of the same prompt with 512 tokens (its prefix is
  then cached). The medians are reported, with min..max.
- short_chat_20req: twenty fixed short prompts with up to 512 tokens each, sent by C concurrent
  workers (worker k takes prompts k, k+C, ...); aggregate = all completion tokens / wave wall time.

With --jsonl, one row per metric is appended in the schema of docs/performance/data/bench.jsonl.
"""
import argparse
import datetime
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HAYSTACK = ROOT / "examples/cli/messages/long_niah_256k.json"
QUESTION = "\n\nSummarize the text above in detail, section by section."
CHAT_PROMPTS = [
    "Explain the difference between optimistic and pessimistic locking, with a short code example in SQL.",
    "Write a Python function that merges two sorted lists into one sorted list without using sort().",
    "A train leaves at 14:35 travelling 84 km/h; another at 15:10 at 96 km/h same direction from 40 km "
    "behind. When does it catch up?",
    "Summarize the trade-offs between REST and gRPC for internal microservices.",
    "Write a bash one-liner that finds the 10 largest files under /var excluding symlinks.",
    "Explain why attention scales quadratically with sequence length and what KV caching changes.",
    "Draft a polite email declining a meeting proposal but offering two alternative slots.",
    "What are the main causes of TCP retransmission storms and how do you diagnose them?",
    "Write a TypeScript type for a recursive directory tree with files and folders.",
    "Give a recipe for a quick tomato-basil pasta for two, with gram amounts.",
    "Explain the CAP theorem with a concrete example of choosing two properties.",
    "Write a Python script that watches a log file and prints new ERROR lines with timestamps.",
    "How does CUDA warp divergence reduce throughput? Give one mitigation.",
    "Translate to French: 'The server room must stay below 22 degrees at all times.'",
    "What is the difference between a mutex and a semaphore? One paragraph.",
    "Write a SQL query returning the top 5 customers by revenue in the last 90 days.",
    "Explain MoE routing: what is top-k routing and why do experts specialize?",
    "Write a JavaScript debounce function with cancel() and immediate options.",
    "A disk shows 98% util but low throughput. What are likely causes and checks?",
    "Explain what speculative decoding is and when it does not help.",
]


def is_commit(text):
    """True for a git commit id (7 to 40 hex digits), not a release version such as 0.1.0."""
    return bool(text) and 7 <= len(text) <= 40 and all(c in "0123456789abcdef" for c in text.lower())


class Server:
    def __init__(self, host, port):
        self.base = f"http://{host}:{port}"
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]
        self.model = model["id"]
        # The decision fingerprint the server started with (absent on other engines).
        meta = model.get("meta") or {}
        self.fingerprint = {key: meta[key] for key in ("decision_fingerprint", "decision_manifest")
                            if key in meta}
        # The speculation the server runs since startup ("plain" or a drafter and depth, how it
        # was selected and any fallback), whatever --spec labels the row: tools/spec_gate.py
        # judges what was served.
        served = meta.get("speculation") or {}
        self.served_speculation = {key: served[key] for key in ("config", "selection", "fallback")
                                   if key in served}

    def chat(self, prompt, max_tokens, top_p=None):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
        if top_p is not None:
            body["top_p"] = top_p
        request = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        start = time.time()
        with urllib.request.urlopen(request, timeout=3600) as r:
            usage = json.load(r).get("usage") or {}
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), time.time() - start


def long_context(server, chars, prompts, out_tokens):
    user = json.loads(HAYSTACK.read_text(encoding="utf-8"))[-1]["content"]
    blob = user.split("<document>\n", 1)[1].split("\n</document>", 1)[0]
    server.chat("Say hello in one short sentence.", 16)  # warm-up
    prefill, decode, lengths = [], [], []
    for i in range(prompts):
        offset = (i * 7919 * 13) % max(1, len(blob) - chars)
        prompt = blob[offset:offset + chars] + QUESTION
        prompt_tokens, _, first = server.chat(prompt, 1)
        _, completion, wall = server.chat(prompt, out_tokens)
        prefill.append(prompt_tokens / first)
        decode.append(completion / wall)
        lengths.append(prompt_tokens)
    return {"prompt_len": int(statistics.median(lengths)),
            "prefill": (statistics.median(prefill), [min(prefill), max(prefill)]),
            "decode": (statistics.median(decode), [min(decode), max(decode)])}


def short_chat(server, concurrency, out_tokens):
    server.chat("Say hello.", 16)  # warm-up
    done, lock = [], threading.Lock()

    def worker(prompts):
        for prompt in prompts:
            _, completion, _ = server.chat(prompt, out_tokens, top_p=1.0)
            with lock:
                done.append(completion)

    threads = [threading.Thread(target=worker, args=(CHAT_PROMPTS[k::concurrency],))
               for k in range(concurrency)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sum(done) / (time.time() - start)


def machine(device):
    query = "name,compute_cap,memory.total,power.limit,driver_version"
    line = subprocess.run(["nvidia-smi", f"--id={device}", f"--query-gpu={query}",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True,
                          check=True).stdout.strip()
    name, cap, mem, power, driver = [x.strip() for x in line.split(",")]
    cuda = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cuda = cuda.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in cuda else None
    return {"gpu_name": name, "sm": cap.replace(".", ""), "vram_gb": round(float(mem) / 1024, 1),
            "power_limit_w": float(power), "driver": driver, "cuda": cuda}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--chars", type=int, default=95940, help="characters per long prompt (~24K tokens)")
    ap.add_argument("--prompts", type=int, default=6, help="long prompts")
    ap.add_argument("--out-tokens", type=int, default=512)
    ap.add_argument("--concurrency", default="1,4,8",
                    help="comma list for the short chat wave; the published rows run each "
                         "value on its own server with --max-concurrency equal to it")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--skip-chat", action="store_true")
    rows = ap.add_argument_group("bench.jsonl rows (with --jsonl)")
    rows.add_argument("--jsonl", type=Path, help="append one row per metric to this file")
    rows.add_argument("--model", help="the Hugging Face checkpoint the setting was converted from")
    rows.add_argument("--recipe", help="the exact source: repository and --quant label or recipe")
    rows.add_argument("--spec", default="none", help="none, mtp, dflash or dflash2")
    rows.add_argument("--draft-tokens", type=int)
    rows.add_argument("--kv", default="auto", help="the server's --kv-dtype")
    rows.add_argument("--activations", default="fast", help="the server's --activations mode")
    rows.add_argument("--flags", default="", help="the server flags, as on the pair's page")
    rows.add_argument("--version", help="the NInferno release or public commit (with --engine, that engine's build)")
    rows.add_argument("--commit", help="the NInferno git commit the server was built from (default: "
                      "--version when it is a commit id); tools/spec_gate.py compares it with the "
                      "speculation-decision code")
    rows.add_argument("--engine", default="ninferno",
                      help="the server measured: ninferno, or llama.cpp for a reference row "
                           "(llama-server with -np equal to the concurrency)")
    rows.add_argument("--device", type=int, default=0, help="CUDA device for the machine record")
    args = ap.parse_args()
    if args.jsonl and not (args.model and args.recipe):
        ap.error("--jsonl needs --model and --recipe")

    server = Server(args.host, args.port)
    results = []
    if not args.skip_long:
        r = long_context(server, args.chars, args.prompts, args.out_tokens)
        print(f"long24k_fr2026_6p prompt_tokens~{r['prompt_len']} prefill={r['prefill'][0]:.0f} "
              f"({r['prefill'][1][0]:.0f}..{r['prefill'][1][1]:.0f}) decode={r['decode'][0]:.2f} "
              f"({r['decode'][1][0]:.2f}..{r['decode'][1][1]:.2f})", flush=True)
        for metric in ("prefill", "decode"):
            results.append({"metric": metric, "prompt_len": r["prompt_len"], "prompt_set": "long24k_fr2026_6p",
                            "concurrency": 1, "repeats": args.prompts, "value": round(r[metric][0], 2),
                            "spread": [round(x, 2) for x in r[metric][1]]})
    if not args.skip_chat:
        for c in (int(x) for x in args.concurrency.split(",")):
            value = short_chat(server, c, args.out_tokens)
            print(f"short_chat_20req C={c} aggregate={value:.1f}", flush=True)
            results.append({"metric": "decode_aggregate", "prompt_len": None,
                            "prompt_set": "short_chat_20req", "concurrency": c, "repeats": 1,
                            "value": round(value, 2), "spread": None})
    if args.jsonl:
        common = {"machine": machine(args.device), "model": args.model, "recipe": args.recipe,
                  "spec": args.spec, "draft_tokens": args.draft_tokens, "kv": args.kv,
                  "activations": args.activations,
                  "sampling": "greedy", "warmup": 1, "unit": "tok/s", "flags": args.flags,
                  "out_tokens": args.out_tokens,
                  "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if args.version:
            common["version"] = args.version
        # The NInferno build only: another engine's --version is its own (llama.cpp's is hex too).
        commit = args.commit or (args.version if is_commit(args.version) else None)
        if commit and args.engine == "ninferno":
            common["commit"] = commit
        if args.engine != "ninferno":
            common["engine"] = args.engine
        common.update(server.fingerprint)
        if server.served_speculation:
            common["served_speculation"] = server.served_speculation
        with args.jsonl.open("a", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps({**common, **row}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
