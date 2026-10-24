#!/usr/bin/env python3
"""Check the configuration choice's estimates (docs/maintainer/configuration-choice.md) against a real
start: a mixed workload of sampled example tasks, timed, against what the engine itself estimates.

    python3 tools/bench/choice_check.py --serve build/apps/ninferno-serve MODEL [--concurrency C]
        [--seed S] [--count N] [--max-context N] [extra ninferno-serve flags]

1. `ninferno-serve MODEL --plan --no-load --json` (with the same flags): the choice's estimates for this
   GPU (`choice.chosen`: decode tokens a second per stream and prefill tokens a second).
2. A start of MODEL with those flags (--spec none: the estimates are without speculation), the choice
   applied as a start applies it, and its `choice |` line kept.
3. N tasks sampled with seed S from examples/cli/messages (code, story, translation, CSV and SQL,
   multi-turn chat, reasoning, needles in 8K and about 22K tokens), each capped at 96-512 answer tokens,
   greedy; C workers each run the whole list, shifted.
4. The estimate of the wall time: (sum of prompt tokens / prefill + sum of answer tokens / decode) / C
   (with several streams a prompt's chunks ride in the other streams' decode steps), against the
   measured one.

Nothing here feeds a decision: it reports how far the estimates the plan shows are from this machine.
"""
import argparse
import http.client
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MESSAGES = ROOT / "examples/cli/messages"
# file, answer cap, characters of the last user message kept (0: all)
TASKS = [
    ("scenario_code_python.json", 512, 0), ("scenario_story_en_mystery.json", 384, 0),
    ("scenario_translation_en_zh.json", 256, 0), ("scenario_structured_csv.json", 192, 0),
    ("text_chat_history.json", 128, 0), ("reasoning_jacobian_counterexample_3d.json", 384, 0),
    ("long_niah_8k.json", 96, 0), ("long_niah_64k.json", 128, 90000), ("text_code_review.json", 256, 0),
    ("scenario_structured_sql.json", 256, 0), ("scenario_translation_markdown.json", 320, 0),
    ("long_decode_math_02.json", 448, 0),
]


def messages_of(name, cut):
    msgs = json.loads((MESSAGES / name).read_text(encoding="utf-8"))
    if cut and isinstance(msgs[-1].get("content"), str):
        msgs[-1]["content"] = msgs[-1]["content"][:cut]
    return msgs


def ask(port, msgs, max_tokens):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    body = {"model": "m", "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    started = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    reply = json.loads(response.read())
    seconds = time.perf_counter() - started
    # A refused request (a prompt past the served context, ...) is an error, not an empty answer.
    if response.status != 200 or "usage" not in reply:
        raise RuntimeError(f"request refused ({response.status}): {json.dumps(reply)[:300]}")
    usage = reply["usage"]
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), seconds


def healthy(port, process, timeout=900):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            return False
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return True
        except OSError:
            pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serve", required=True, help="the ninferno-serve binary")
    ap.add_argument("model", help="an artifact, GGUF or hub id, as ninferno-serve takes it")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--port", type=int, default=18979)
    a, flags = ap.parse_known_args()  # the rest goes to ninferno-serve as given
    if not 1 <= a.count <= len(TASKS):
        ap.error(f"--count takes 1..{len(TASKS)} (the task mix's size)")
    extra = [f for f in flags if f != "--"]
    # Plain decoding unless the caller names a speculation: the plan and the server see the same
    # flags (a drafter's weights and KV change the plan, its speed the time).
    spec = [] if any(f == "--spec" or f.startswith("--spec=") for f in extra) else ["--spec", "none"]
    common = [a.model, "--max-context", str(a.max_context), "--max-concurrency", str(a.concurrency), *extra, *spec]

    plan = json.loads(subprocess.run([a.serve, *common, "--plan", "--no-load", "--json"], check=True,
                                     capture_output=True, text=True).stdout)
    chosen = (plan.get("choice") or {}).get("chosen")
    if not chosen:
        sys.exit("the plan has no configuration choice: " + json.dumps(plan.get("choice")))
    decode, prefill = chosen["decode_tok_s"], chosen["prefill_tok_s"]

    log = Path(f"choice_check_{a.port}.log")
    with log.open("w") as out:
        server = subprocess.Popen([a.serve, *common, "--port", str(a.port)], stdout=out,
                                  stderr=subprocess.STDOUT)
    try:
        if not healthy(a.port, server):
            sys.exit(f"ninferno-serve did not start (see {log})")
        tasks = random.Random(a.seed).sample(TASKS, a.count)
        ask(a.port, [{"role": "user", "content": "Hi"}], 8)  # untimed warm-up
        rows, lock = [], threading.Lock()

        def worker(w):
            for i in range(len(tasks)):
                name, cap, cut = tasks[(i + w) % len(tasks)]
                p, g, s = ask(a.port, messages_of(name, cut), cap)
                with lock:
                    rows.append((name, p, g, s))

        started = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(a.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - started
    finally:
        server.terminate()
        server.wait()
    line = next((l.strip() for l in log.read_text(errors="replace").splitlines() if "choice |" in l), "")
    prompt = sum(r[1] for r in rows)
    answer = sum(r[2] for r in rows)
    estimate = (prompt / prefill + answer / decode) / a.concurrency
    print(f"model {a.model}, {a.concurrency} stream(s), seed {a.seed}, {len(rows)} requests")
    print(line)
    print(f"estimates: decode {decode:.1f} tok/s per stream, prefill {prefill:.0f} tok/s")
    print(f"prompt tokens {prompt}, answer tokens {answer}")
    print(f"wall {wall:.2f} s, estimated {estimate:.2f} s: {100 * (estimate / wall - 1):+.1f} % "
          f"({answer / wall:.1f} tok/s measured, {answer / estimate:.1f} estimated)")


if __name__ == "__main__":
    main()
