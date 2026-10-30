#!/usr/bin/env python3
# sm_bench.py PORT MODE OUT.json : speed-model verification points against a running ninferno-serve.
#   MODE decode: plain decode at 1/2/4/8 concurrent streams (short prompts, 256 tokens each)
#   MODE prefill: one request per prompt length 1K..16K (max_tokens 1), distinct text each, twice
#   MODE spec: one stream, 4 prompts x 256 tokens (a speculating server)
import json, sys, threading, urllib.request, statistics, time

port, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
HAY = json.load(open(sys.argv[4] if len(sys.argv) > 4 else "/root/claude/sm/long_niah_256k.json"))[1]["content"]
TOPICS = ["the history of the steam engine", "how bridges are designed", "the life cycle of stars",
          "the invention of the printing press", "how vaccines train the immune system",
          "the economics of shipping containers", "the geology of volcanoes", "how compilers optimize loops"]


def ask(content, max_tokens):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "seed": 1, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(r, timeout=1800))
    t = o.get("timings", {})
    return {k: t.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                                  "predicted_per_second", "draft_n", "draft_n_accepted", "cache_n")}


def essay(i):
    return f"Write a long, detailed essay about {TOPICS[i % len(TOPICS)]}. Use several sections and examples."


res = {"mode": mode, "rows": []}
ask(essay(7), 32)  # warm-up
if mode == "decode":
    for c in (1, 2, 4, 8):
        for rep in range(2):
            got = [None] * c
            def run(i):
                got[i] = ask(essay(i + rep), 256)
            th = [threading.Thread(target=run, args=(i,)) for i in range(c)]
            [t.start() for t in th]; [t.join() for t in th]
            res["rows"].append({"streams": c, "rep": rep, "per_stream": [g["predicted_per_second"] for g in got],
                                "n": [g["predicted_n"] for g in got], "prompt_n": [g["prompt_n"] for g in got]})
            print(c, rep, round(statistics.median(g["predicted_per_second"] for g in got), 2), flush=True)
elif mode == "prefill":
    off = 0
    for n in (1024, 2048, 4096, 8192, 16384):
        for rep in range(2):
            chars = int(n * 3.6)
            text = HAY[off:off + chars]; off += chars
            g = ask(text + "\n\nIn one word: what is the text about?", 1)
            res["rows"].append({"target": n, "rep": rep, **g})
            print(n, rep, g["prompt_n"], round(g["prompt_per_second"], 1), flush=True)
elif mode == "spec":
    for i in range(4):
        g = ask(essay(i), 256)
        res["rows"].append(g)
        print(i, g["predicted_n"], round(g["predicted_per_second"], 2), g.get("draft_n"), g.get("draft_n_accepted"), flush=True)
json.dump(res, open(out, "w"), indent=1)
