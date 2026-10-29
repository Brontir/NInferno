#!/usr/bin/env python3
"""Live repro of the "schema-order" bug class against a running ninferno-serve.

A schema with five optional fields (schema order name, email, city, company, phone) and a prompt
that asks for them in the order city, name, phone, email, company (C, A, E, B, D). Each mode runs
N times; per mode it reports the mean number of fields written, the runs that lost a field, the
fields lost per run, the share of responses whose keys are exactly in schema order, the share in
the prompt's order, and near-empty objects (at most one field) that ended with finish_reason stop.
A grammar that holds optional keys to schema order loses name, email and company (A, B, D) only
when constrained.

usage: python3 -m tools.smoke.serve_key_order --model M --out result.json [--base-url URL]
       [--runs 20] [--modes a,b] [--think]
"""
import argparse
import json
import re
import sys
import time
import urllib.request

FIELDS = ["name", "email", "city", "company", "phone"]  # schema order A..E
PROMPT_ORDER = ["city", "name", "phone", "email", "company"]  # C, A, E, B, D
SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in FIELDS},
    "additionalProperties": False,
}
CONTACT = ("Maria Lopez works at Acme Robotics in Lisbon. Her phone is +351 912 345 678 and her "
           "email is maria.lopez@acme.example.")
ORDER_TEXT = ", ".join(PROMPT_ORDER)
TOOL_PROMPT = (f"Save this contact with the save_contact tool. Write the arguments in exactly this "
               f"order: {ORDER_TEXT}. Fill every field.\n\n{CONTACT}")
JSON_PROMPT = (f"Return this contact as one JSON object with the keys in exactly this order: "
               f"{ORDER_TEXT}. Fill every field. Reply with the JSON object only.\n\n{CONTACT}")


def tool(strict):
    return {"type": "function", "function": {
        "name": "save_contact", "description": "Save a contact record.",
        "parameters": SCHEMA, "strict": strict}}


MODES = {
    # tool calls
    "tool_free": dict(prompt=TOOL_PROMPT, tools=[tool(False)], tool_choice="auto"),
    "tool_nonstrict_required": dict(prompt=TOOL_PROMPT, tools=[tool(False)],
                                    tool_choice="required"),
    "tool_strict_auto": dict(prompt=TOOL_PROMPT, tools=[tool(True)], tool_choice="auto"),
    "tool_strict_required": dict(prompt=TOOL_PROMPT, tools=[tool(True)], tool_choice="required"),
    # JSON answers
    "json_free": dict(prompt=JSON_PROMPT),
    "json_schema": dict(prompt=JSON_PROMPT, response_format={
        "type": "json_schema", "json_schema": {"name": "contact", "schema": SCHEMA,
                                               "strict": True}}),
}


def post(url, body, timeout=600):
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ordered_keys(text):
    """The object's keys in the order written (json.loads keeps order; duplicates noted)."""
    pairs_seen = []

    def hook(pairs):
        pairs_seen.append([k for k, _ in pairs])
        return dict(pairs)

    value = json.loads(text, object_pairs_hook=hook)
    return value, pairs_seen[-1] if pairs_seen else []


def extract_json(content):
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", content, re.S)
    if fence:
        content = fence.group(1).strip()
    start = content.find("{")
    end = content.rfind("}")
    return content[start:end + 1] if start >= 0 and end > start else content


def run(url, mode, spec, runs, think, model):
    rows = []
    for index in range(runs):
        body = {"model": model, "messages": [{"role": "user", "content": spec["prompt"]}],
                "temperature": 0.8, "top_p": 0.95, "seed": 1000 + index, "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": think}}
        for key in ("tools", "tool_choice", "response_format"):
            if key in spec:
                body[key] = spec[key]
        started = time.time()
        row = {"run": index}
        try:
            answer = post(url, body)
        except Exception as error:  # noqa: BLE001 - recorded per run
            row["error"] = str(error)[:300]
            rows.append(row)
            continue
        choice = answer["choices"][0]
        message = choice["message"]
        row["finish_reason"] = choice.get("finish_reason")
        row["completion_tokens"] = answer.get("usage", {}).get("completion_tokens")
        row["seconds"] = round(time.time() - started, 2)
        calls = message.get("tool_calls") or []
        try:
            if calls:
                text = calls[0]["function"]["arguments"]
                row["source"] = "tool_call"
            elif "tools" in spec:
                text = None
                row["source"] = "no_call"
            else:
                text = extract_json(message.get("content") or "")
                row["source"] = "content"
            if text is not None:
                value, keys = ordered_keys(text)
                row["keys"] = keys
                row["filled"] = [k for k in FIELDS if isinstance(value.get(k), str) and value[k]]
        except Exception as error:  # noqa: BLE001
            row["parse_error"] = str(error)[:200]
            row["raw"] = (message.get("content") or "")[:300]
        rows.append(row)
    return rows


def summarize(rows):
    answered = [r for r in rows if "keys" in r]
    n = len(answered)
    if n == 0:
        return {"runs": len(rows), "answered": 0}

    def share(pred):
        return round(sum(1 for r in answered if pred(r)) / n, 3)

    lost = {f: sum(1 for r in answered if f not in r["filled"]) for f in FIELDS}
    return {
        "runs": len(rows),
        "answered": n,
        "errors": sum(1 for r in rows if "error" in r or "parse_error" in r),
        "no_call": sum(1 for r in rows if r.get("source") == "no_call"),
        "mean_fields": round(sum(len(r["filled"]) for r in answered) / n, 2),
        "runs_losing_a_field": sum(1 for r in answered if len(r["filled"]) < len(FIELDS)),
        "lost_per_field": lost,
        "schema_order_share": share(lambda r: len(r["keys"]) > 1 and r["keys"] == [
            f for f in FIELDS if f in r["keys"]]),
        "prompt_order_share": share(lambda r: r["keys"] == PROMPT_ORDER),
        "near_empty_stop": sum(1 for r in answered if len(r["keys"]) <= 1 and
                               r.get("finish_reason") in ("stop", "tool_calls")),
        "mean_completion_tokens": round(sum(r.get("completion_tokens") or 0 for r in answered) / n,
                                        1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = {}
    for mode in args.modes.split(","):
        rows = run(args.base_url, mode, MODES[mode], args.runs, args.think, args.model)
        result[mode] = {"summary": summarize(rows), "rows": rows}
        print(mode, json.dumps(result[mode]["summary"]), flush=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    sys.exit(main())
