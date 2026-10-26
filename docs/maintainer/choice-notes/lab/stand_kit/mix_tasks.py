#!/usr/bin/env python3
"""A small mixed workload for checking the choice model's predictions: a fixed random sample of the repository's
example tasks (code, story, translation, structured output, multi-turn chat, reasoning, a needle in an 8K and a
cut 24K haystack) with output caps from 96 to 512 tokens, greedy. Each request's prompt and completion tokens and
wall time go to stdout as JSON lines; --concurrency N sends N at once (each worker its own slice, then repeats).
usage: mix_tasks.py --port P --messages DIR [--concurrency N] [--seed S]"""
import argparse
import http.client
import json
import os
import random
import threading
import time

TASKS = [  # file, max_tokens, cut (characters of the last user message kept; 0: all)
    ("scenario_code_python.json", 512, 0), ("scenario_story_en_mystery.json", 384, 0),
    ("scenario_translation_en_zh.json", 256, 0), ("scenario_structured_csv.json", 192, 0),
    ("text_chat_history.json", 128, 0), ("reasoning_jacobian_counterexample_3d.json", 384, 0),
    ("long_niah_8k.json", 96, 0), ("long_niah_64k.json", 128, 90000), ("text_code_review.json", 256, 0),
    ("scenario_structured_sql.json", 256, 0), ("scenario_translation_markdown.json", 320, 0), ("long_decode_math_02.json", 448, 0),
]


def load(d, f, cut):
    msgs = json.load(open(os.path.join(d, f), encoding="utf-8"))
    if cut:
        last = msgs[-1]
        if isinstance(last.get("content"), str):
            last["content"] = last["content"][:cut]
    return msgs


def ask(port, msgs, max_tokens):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    body = {"model": "m", "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read())
    dt = time.perf_counter() - t0
    u = r.get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--messages", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--count", type=int, default=8)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    tasks = rnd.sample(TASKS, a.count)
    ask(a.port, [{"role": "user", "content": "Hi"}], 8)  # untimed warm-up
    out, lock = [], threading.Lock()

    def worker(w):
        for i, (f, mt, cut) in enumerate(tasks):
            if i % a.concurrency != w and a.concurrency > 1:
                pass
            f2, mt2, cut2 = tasks[(i + w) % len(tasks)]  # every worker runs the whole list, shifted
            p, g, dt = ask(a.port, load(a.messages, f2, cut2), mt2)
            with lock:
                rec = {"task": f2, "worker": w, "P": p, "G": g, "s": round(dt, 4), "t": round(time.perf_counter() - T0, 3)}
                out.append(rec)
                print(json.dumps(rec), flush=True)

    T0 = time.perf_counter()
    th = [threading.Thread(target=worker, args=(w,)) for w in range(a.concurrency)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - T0
    toks = sum(r["G"] for r in out)
    print(json.dumps({"summary": True, "C": a.concurrency, "requests": len(out), "wall_s": round(wall, 3),
                      "completion_tokens": toks, "prompt_tokens": sum(r["P"] for r in out),
                      "tok_s_aggregate": round(toks / wall, 2)}), flush=True)


if __name__ == "__main__":
    main()
