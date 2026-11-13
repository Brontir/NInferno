#!/usr/bin/env python3
"""The whole-step A/B of #850 (kernel proposal P, acceptance item 4): requests and summary.

    python3 tools/bench/step_ab.py prompt [--chars N] > prompt.txt
    python3 tools/bench/step_ab.py request --port P --arm A --round I --prompt prompt.txt \
        --out rounds.jsonl [--requests N] [--out-tokens N]
    python3 tools/bench/step_ab.py summarize OUT

tools/bench/run_850_step_ab.sh drives it; it starts and stops the servers, this file only talks to
one and reads what the rounds wrote.

- prompt: the first slice of the long-context prompt set docs/performance measures
  (long24k_fr2026_6p, tools/bench/perf_matrix.py): CHARS characters of the Federal Register text,
  then the request to summarize it. 95940 characters are about 22.2K Qwen3.5 tokens.
- request: one short warm-up, then REQUESTS times the fixed prompt, greedy, `ignore_eos` so every
  request decodes OUT_TOKENS tokens. One JSON line per timed request: the server's own decode rate
  (`timings.predicted_per_second`: predicted_n - 1 token intervals over predicted_ms, so the
  prompt's cost never enters it), the token counts and a hash of the output. A request that drafted
  (`draft_n` in its timings), stopped short or had a short prompt is refused: the A/B is plain
  decode at 22K.
- summarize: OUT/rounds.jsonl, OUT/meta.txt and OUT/clocks_<round>.csv into OUT/summary.md. A
  round's value is the median of its requests; an arm's, the median of its rounds; B/A is the ratio
  of the two. The noise is half the larger of the two arms' spreads (max - min over the median of
  its rounds: A against A, B against B); the verdict "no slower" needs B/A >= 1 - noise.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_matrix  # noqa: E402  (the long-context prompt set: its text and question)

# The frozen workload (kernel proposal P): the default CHARS (95940) makes a 22,184-token prompt.
# A prompt outside ±3 % of it is another context and cache regime, and the A/B refuses it.
FROZEN_PROMPT_TOKENS = 22184
FROZEN_ROUNDS = 6  # repeats a build, kernel proposal P (#850)
PROMPT_TOKEN_SLACK   = 0.03


def prompt_text(chars):
    user = json.loads(perf_matrix.HAYSTACK.read_text(encoding="utf-8"))[-1]["content"]
    blob = user.split("<document>\n", 1)[1].split("\n</document>", 1)[0]
    if len(blob) < chars:
        sys.exit(f"the haystack has {len(blob)} characters, fewer than {chars}")
    return blob[:chars] + perf_matrix.QUESTION


def post(base, body, timeout=3600):
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as error:
        sys.exit(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:2000]}")


def decode_rate(timings):
    """The server's decode tok/s; else (predicted_n - 1) / predicted_ms, the same quantity."""
    rate = timings.get("predicted_per_second")
    if rate:
        return float(rate)
    n, ms = timings.get("predicted_n") or 0, timings.get("predicted_ms") or 0
    return (n - 1) / (ms / 1000.0) if n > 1 and ms > 0 else 0.0


def request(args):
    base = f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    common = {"model": model, "temperature": 0.0, "stream": False}
    post(base, {**common, "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 16})
    with open(args.out, "a", encoding="utf-8") as out:
        for i in range(args.requests):
            started = time.time()
            body = {**common, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.out_tokens, "ignore_eos": True}
            response = post(base, body)
            elapsed = time.time() - started
            timings = response.get("timings") or {}
            usage = response.get("usage") or {}
            message = (response.get("choices") or [{}])[0].get("message") or {}
            text = (message.get("reasoning_content") or "") + "\x00" + (message.get("content") or "")
            row = {
                "arm": args.arm, "round": args.round, "request": i + 1,
                "decode_tok_s": round(decode_rate(timings), 3),
                "predicted_n": timings.get("predicted_n"), "predicted_ms": timings.get("predicted_ms"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_ms": timings.get("prompt_ms"), "wall_s": round(elapsed, 3),
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            }
            # `speculation` describes how the server runs (also "plain"); only drafted tokens mean
            # the request ran speculatively (openai_chat_response.cpp: draft_n only when nonzero).
            row["speculation"] = timings.get("speculation")
            if (timings.get("draft_n") or 0) > 0:
                sys.exit(f"the request drafted ({timings}): the A/B is plain decode (--spec none)")
            tokens = row["prompt_tokens"] or 0
            if abs(tokens - FROZEN_PROMPT_TOKENS) > PROMPT_TOKEN_SLACK * FROZEN_PROMPT_TOKENS:
                sys.exit(f"prompt of {tokens} tokens: the frozen A/B decodes at "
                         f"{FROZEN_PROMPT_TOKENS} (±{PROMPT_TOKEN_SLACK:.0%}); keep CHARS at its "
                         "default")
            if (row["completion_tokens"] or 0) < args.out_tokens or row["decode_tok_s"] <= 0:
                sys.exit(f"the request decoded {row['completion_tokens']} of {args.out_tokens} "
                         f"tokens (rate {row['decode_tok_s']}): no full decode to time")
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{args.arm} round {args.round} request {i + 1}: {row['decode_tok_s']:.2f} tok/s "
                  f"(prompt {row['prompt_tokens']}, {row['completion_tokens']} tokens)", flush=True)


def read_clocks(path):
    """(min, median, max SM MHz, max temperature) of one round's nvidia-smi samples, or None."""
    sm, temp = [], []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for fields in csv.reader(f):
                try:
                    sm.append(float(fields[0]))
                    temp.append(float(fields[2]))
                except (IndexError, ValueError):
                    continue
    except OSError:
        return None
    if not sm:
        return None
    return min(sm), statistics.median(sm), max(sm), max(temp) if temp else None


def summarize(args):
    out = Path(args.out)
    lines_in = (out / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines_in if line.strip()]
    meta = (out / "meta.txt").read_text(encoding="utf-8") if (out / "meta.txt").exists() else ""
    rounds = {}
    for row in rows:
        rounds.setdefault((row["round"], row["arm"]), []).append(row)
    order = sorted(rounds)
    arms = {"A": [], "B": []}
    lines = ["| round | arm | decode tok/s | requests | SM MHz min / med / max | max temp C |",
             "|---|---|---|---|---|---|"]
    for key in order:
        rnd, arm = key
        values = [r["decode_tok_s"] for r in rounds[key]]
        value = statistics.median(values)
        arms[arm].append(value)
        clocks = read_clocks(out / f"clocks_{rnd}.csv")
        clock_text = "-" if clocks is None else f"{clocks[0]:.0f} / {clocks[1]:.0f} / {clocks[2]:.0f}"
        temp_text = "-" if clocks is None or clocks[3] is None else f"{clocks[3]:.0f}"
        lines.append(f"| {rnd} | {arm} | {value:.2f} | {', '.join(f'{v:.2f}' for v in values)} | "
                     f"{clock_text} | {temp_text} |")
    if not arms["A"] or not arms["B"]:
        sys.exit(f"{out}: rounds of both arms are needed")
    # Proposal P's acceptance reads six repeats a build (kernel-proposals.md, frozen): no verdict
    # from fewer or more.
    if len(arms["A"]) != FROZEN_ROUNDS or len(arms["B"]) != FROZEN_ROUNDS:
        sys.exit(f"{out}: {len(arms['A'])} A and {len(arms['B'])} B rounds, "
                 f"{FROZEN_ROUNDS} each needed (frozen)")
    expected = getattr(args, "rounds", None)
    if expected and (len(arms["A"]) != expected or len(arms["B"]) != expected):
        sys.exit(f"{out}: {len(arms['A'])} A and {len(arms['B'])} B rounds, {expected} each needed")

    med = {arm: statistics.median(v) for arm, v in arms.items()}
    spread = {arm: (max(v) - min(v)) / med[arm] for arm, v in arms.items()}
    noise = max(spread.values()) / 2
    ratio = med["B"] / med["A"]
    # The ABBA blocks: B over A within each block of four rounds, the drift cancelled.
    blocks = []
    for start in range(0, len(order), 4):
        block = order[start:start + 4]
        a = [statistics.median(r["decode_tok_s"] for r in rounds[k]) for k in block if k[1] == "A"]
        b = [statistics.median(r["decode_tok_s"] for r in rounds[k]) for k in block if k[1] == "B"]
        if a and b:
            blocks.append(statistics.mean(b) / statistics.mean(a))
    if ratio > 1 + noise:
        verdict = "faster (B/A above 1 + noise)"
    elif ratio >= 1 - noise:
        verdict = "no slower (B/A within the noise)"
    else:
        verdict = "SLOWER (B/A below 1 - noise): the card's entries fail acceptance item 4"
    hashes = {arm: {r["output_sha256"] for r in rows if r["arm"] == arm} for arm in "AB"}
    same = hashes["A"] == hashes["B"] and len(hashes["A"]) == 1
    prompt_tokens = sorted({r["prompt_tokens"] for r in rows})

    text = ["# #850 whole-step A/B (kernel proposal P, acceptance item 4)", "",
            "Plain decode, C = 1, `--spec none`, greedy, ~22K-token prompt, ABBA rounds; "
            "A = master, B = the PR.", "", "```", meta.strip(), "```", "",
            f"Prompt tokens: {', '.join(str(t) for t in prompt_tokens)}; output tokens a request: "
            f"{rows[0]['completion_tokens']}.", "", *lines, "",
            "| arm | rounds | median tok/s | min | max | spread (max - min) / median |",
            "|---|---|---|---|---|---|"]
    for arm in "AB":
        v = arms[arm]
        text.append(f"| {arm} | {len(v)} | {med[arm]:.2f} | {min(v):.2f} | {max(v):.2f} | "
                    f"{100 * spread[arm]:.2f} % |")
    text += ["",
             f"**B/A = {ratio:.4f}** ({100 * (ratio - 1):+.2f} %); ABBA blocks: "
             f"{', '.join(f'{b:.4f}' for b in blocks)}.",
             f"Noise (half the larger arm spread): {100 * noise:.2f} %, so \"no slower\" needs "
             f"B/A >= {1 - noise:.4f}.",
             f"**Verdict: {verdict}.**",
             f"Outputs: {'identical in every request of both arms' if same else 'NOT identical'} "
             f"({len(hashes['A'])} distinct in A, {len(hashes['B'])} in B; the entries are "
             "bit-identical schedules, so greedy outputs should match).", ""]
    report = "\n".join(text)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompt")
    p.add_argument("--chars", type=int, default=95940)
    r = sub.add_parser("request")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--arm", choices=("A", "B"), required=True)
    r.add_argument("--round", type=int, required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--requests", type=int, default=2)
    r.add_argument("--out-tokens", type=int, default=512)
    s = sub.add_parser("summarize")
    s.add_argument("out")
    s.add_argument("--rounds", type=int, help="rounds each arm must have (the run's ROUNDS)")
    args = parser.parse_args()
    if args.command == "prompt":
        sys.stdout.write(prompt_text(args.chars))
    elif args.command == "request":
        request(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
