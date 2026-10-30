#!/usr/bin/env python3
"""hard_set.py: one serving configuration through the hard question set, for quality and speed.

    tools/bench/hard_set.py --url http://127.0.0.1:18300/v1 --set eval/corpora/hard50/hard50.json \
        --tag 9b_q4km_mtp3 --out runs/9b_q4km_mtp3.json

The set (eval/corpora/hard50/hard50.json) holds questions the lab's earlier EvalScope runs answered
right in some configurations and wrong in others (GSM8K, MATH-500, MMLU-Pro), with the prompt as
those runs sent it and the reference answer. Each question is asked once, greedy, one request at a
time, thinking off; the server's own timings give the decode time and, under speculation, the
drafts it verified and accepted. Per question the output keeps whether the answer matched the
reference, the answer, the decode tokens and milliseconds, and the drafts; the summary gives the
share answered right, the decode rate and, under speculation, tokens per round: the server's
own count of the request's rounds (timings.speculation.rounds), whatever each drafted (a fixed
depth, a stepped-down window or a prompt-lookup copy round, copy_rounds, whose accepted tokens are
among the request's); a server that reports none gives tokens - 1 - accepted - fallback_steps -
plain_steps (types.h: a request generates 1 + rounds + accepted + fallback_steps + plain_steps).
Tokens per round = (accepted + rounds) / rounds (1 for plain decode). A failed request counts as a wrong answer and
is kept with its error; the output is rewritten after every question.

Grading: GSM8K and MATH-500 compare the last \\boxed{...} with the reference after a light
normalization (spaces, $, %, \\text{}, \\left/\\right, a trailing ".0", thousands commas, a
fraction as a/b, a trailing unit, case), GSM8K as numbers, a GSM8K answer without a box by its last
number (as EvalScope does); MMLU-Pro compares the letter of the last "ANSWER: X" (or "the answer is
X"). A configuration's score is meant to be compared with another's on the same set; an answer the
normalization misses counts as wrong in both, so the difference stays honest.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request


def last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def normalize(answer):
    a = answer.strip().strip("$").strip()
    a = re.sub(r"\\text(?:bf)?\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\(left|right)", "", a)
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = a.replace("^{\\circ}", "").replace("^\\circ", "")
    a = a.replace("\\$", "").replace("\\%", "").replace("%", "").replace("{,}", ",")
    a = a.replace("\\!", "").replace("\\,", "").replace(" ", "")
    a = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", a)
    a = re.sub(r"\\frac\{([^{}]*)\}(\w)", r"\1/\2", a)
    a = re.sub(r"\\frac(\w)(\w)", r"\1/\2", a)
    a = re.sub(r"(?<=\d)(degrees?|units?|dollars|cents|cm|inches|feet|meters|miles|hours?|"
               r"minutes?)$", "", a)
    a = re.sub(r"(?<=\d),(?=\d{3}\b)", "", a)
    a = re.sub(r"^(-?\d+)\.0+$", r"\1", a)
    return a.rstrip(".").lower()


def as_number(answer):
    """A GSM8K answer as a number ("$1,200." -> 1200.0), else None."""
    a = normalize(answer)
    if not re.fullmatch(r"-?\d*\.?\d+", a):
        return None
    return float(a)


def grade(item, answer):
    if item["dataset"] == "mmlu_pro":
        found = (re.findall(r"ANSWER:\s*(?:\$?\\boxed\{)?\(?([A-J])\b", answer) or
                 re.findall(r"(?i:answer) is:?\s*\(?([A-J])\b", answer))
        return bool(found) and found[-1] == item["target"].strip()
    boxed = last_boxed(answer)
    if boxed is None and item["dataset"] == "gsm8k":
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer)
        boxed = numbers[-1] if numbers else None
    if boxed is None:
        return False
    if item["dataset"] == "gsm8k":
        got, want = as_number(boxed), as_number(item["target"])
        if got is not None and want is not None:
            return got == want
    return normalize(boxed) == normalize(item["target"])


def ask(url, item, timeout):
    body = {"model": "m", "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": item["max_tokens"], "temperature": 0.0, "seed": 42, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    wall = time.time() - t0
    timings = out.get("timings") or {}
    spec = timings.get("speculation") or {}
    return {"answer": out["choices"][0]["message"].get("content") or "",
            "config": spec.get("config", ""),
            "tokens": (timings.get("predicted_n") or
                       (out.get("usage") or {}).get("completion_tokens") or 0),
            "decode_ms": timings.get("predicted_ms") or 0.0,
            "accepted": timings.get("draft_n_accepted") or 0,
            "drafted": timings.get("draft_n") or 0,
            "rounds": spec.get("rounds"),
            "fallback_steps": spec.get("fallback_steps", 0),
            "plain_steps": spec.get("plain_steps", 0),
            "step_downs": spec.get("step_downs", 0),
            "capped_rounds": spec.get("capped_rounds", 0),
            "copy_rounds": spec.get("copy_rounds", 0),
            "watchdog_tripped": spec.get("watchdog_tripped", False),
            "wall_s": wall}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:18300/v1")
    ap.add_argument("--set", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    items = json.load(open(args.set, encoding="utf-8"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for item in items:
        try:
            r = ask(args.url, item, args.timeout)
        except Exception as e:  # one failed request is one wrong answer, not a lost run
            r = {"answer": "", "config": "", "tokens": 0, "decode_ms": 0.0, "accepted": 0,
                 "drafted": 0, "error": str(e)}
        r.update(id=item["id"], dataset=item["dataset"], correct=grade(item, r["answer"]))
        rows.append(r)
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)
        print(f"{item['id']:28s} {'ok ' if r['correct'] else 'BAD'} {r['tokens']:5d} tok "
              f"{r['decode_ms']:8.0f} ms drafts {r['drafted']} {r.get('error', '')}",
              file=sys.stderr)
    tokens = sum(r["tokens"] for r in rows)
    # The server times decode from the first token to the last: the first is not in decode_ms.
    timed = sum(max(r["tokens"] - 1, 0) for r in rows)
    decode_ms = sum(r["decode_ms"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    rounds = sum(r["rounds"] if r.get("rounds") is not None else
                 max(r["tokens"] - 1 - r["accepted"] - r.get("fallback_steps", 0) -
                     r.get("plain_steps", 0), 0)
                 for r in rows)
    summary = {"tag": args.tag, "questions": len(rows),
               "correct": sum(r["correct"] for r in rows),
               "by_dataset": {d: sum(r["correct"] for r in rows if r["dataset"] == d)
                              for d in sorted({r["dataset"] for r in rows})},
               "tokens": tokens, "decode_ms": decode_ms,
               "tok_s": 1000.0 * timed / decode_ms if decode_ms else 0.0,
               "rounds": rounds,
               "accepted": accepted,
               "capped_rounds": sum(r.get("capped_rounds", 0) for r in rows),
               "copy_rounds": sum(r.get("copy_rounds", 0) for r in rows),
               "errors": sum("error" in r for r in rows),
               "tokens_per_round": (accepted + rounds) / rounds if rounds else None}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
