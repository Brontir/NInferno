"""API conformance of a running ninferno-serve.

Every check is a pair or a table, never a single sample:
- stream == non-stream: content, reasoning, tool calls, finish reason and usage, for plain,
  thinking, stop-string, length-in-a-code-point, tool_choice none/auto/required/named, a length cut
  inside a call, and structured output; OpenAI chat and Anthropic messages;
- every SSE event is checked against the dialect's shape: OpenAI one role event, exactly one
  finish_reason and last, usage chunk with choices == [], a single [DONE]; Anthropic
  message_start (type, role, stop_sequence present), monotonic content-block indices with
  matching start/delta/stop, message_delta with stop_reason and stop_sequence, message_stop last;
- tool calls: OpenAI arguments is a JSON string, no empty tool_calls array, ids unique;
  Anthropic input is an object; history accepts arguments as a string;
- unsupported or invalid requests are 4xx in the dialect's error shape, never 5xx, and a
  streaming request is rejected with a JSON error before any SSE byte (n != 1, logprobs,
  top_logprobs, nonzero logit_bias, invalid UTF-8, wrong types, out-of-range values);
- response_format json_schema output parses and satisfies the schema; a seeded sampled request
  repeats byte for byte.

Standard library only. Exit status 1 when any check fails; the JSON report goes to stdout.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Http:
    status: int
    content_type: str
    body: bytes
    first_byte_s: float = 0.0


@dataclass
class Report:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, case: str, check: str, ok: bool, detail: Any = None) -> bool:
        entry: dict[str, Any] = {"case": case, "check": check, "ok": bool(ok)}
        if detail is not None:
            entry["detail"] = detail
        self.checks.append(entry)
        if not ok:
            print(f"FAIL [{case}] {check}: {detail}", file=sys.stderr)
        return bool(ok)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if not c["ok"])


class Client:
    def __init__(self, base_url: str) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("", "http"):
            raise SystemExit(f"{base_url}: the server speaks plain HTTP (http://host:port)")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80

    def call(self, method: str, path: str, payload: Any = None, raw: bytes | None = None,
             timeout: float = 600.0) -> Http:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        body = raw if raw is not None else (
            None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        headers = {"Content-Type": "application/json"} if body is not None else {}
        start = time.monotonic()
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        first = response.read(1)
        first_byte = time.monotonic() - start
        rest = response.read()
        conn.close()
        return Http(response.status, response.getheader("Content-Type", "").split(";")[0].strip(),
                    first + rest, first_byte)


# ---------------------------------------------------------------- OpenAI chat

def openai_nonstream(http: Http) -> dict[str, Any]:
    if http.status != 200:
        raise AssertionError(f"HTTP {http.status}: {http.body[:300]!r}")
    value = json.loads(http.body)
    choice = value["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls")
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in calls or []],
        "tool_calls_raw": calls,
        "finish": choice.get("finish_reason"),
        "usage": value.get("usage"),
        "choices": len(value["choices"]),
    }


def openai_stream(http: Http, errors: list[str]) -> dict[str, Any]:
    """Parses and validates an OpenAI chat SSE stream; shape violations go to errors."""
    if http.status != 200 or http.content_type != "text/event-stream":
        raise AssertionError(f"stream HTTP {http.status} {http.content_type}: {http.body[:300]!r}")
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage = None
    roles = 0
    done = 0
    ids: set[str] = set()
    for block in http.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        # An SSE comment (the server opens a stream with ": stream") carries no event.
        if not block.strip() or all(line.startswith(":") for line in block.splitlines()):
            continue
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(data) != 1:
            errors.append(f"malformed SSE block {block[:80]!r}")
            continue
        if data[0] == "[DONE]":
            done += 1
            continue
        if done:
            errors.append("data after [DONE]")
        event = json.loads(data[0])
        ids.add(event.get("id"))
        if event.get("object") != "chat.completion.chunk":
            errors.append(f"object {event.get('object')!r}")
        choices = event.get("choices")
        if event.get("usage") is not None:
            if choices != []:
                errors.append("usage chunk carries choices")
            if usage is not None:
                errors.append("usage twice")
            if finish is None:
                errors.append("usage before finish")
            usage = event["usage"]
            continue
        if usage is not None:
            errors.append("event after usage")
        if not isinstance(choices, list) or len(choices) != 1:
            errors.append("chunk without exactly one choice")
            continue
        if finish is not None:
            errors.append("event after finish_reason")
        choice = choices[0]
        if choice.get("index") != 0:
            errors.append("choice index != 0")
        delta = choice.get("delta", {})
        if "role" in delta:
            roles += 1
        if isinstance(delta.get("content"), str):
            content.append(delta["content"])
        if isinstance(delta.get("reasoning_content"), str):
            reasoning.append(delta["reasoning_content"])
        for call in delta.get("tool_calls") or []:
            entry = calls.setdefault(call["index"], {"name": "", "arguments": "", "id": None})
            if call.get("id"):
                entry["id"] = call["id"]
            fn = call.get("function") or {}
            if "arguments" in fn and not isinstance(fn["arguments"], str):
                errors.append("streamed arguments is not a string")
            entry["name"] += fn.get("name") or ""
            entry["arguments"] += fn.get("arguments") or ""
        if "tool_calls" in delta and not delta["tool_calls"]:
            errors.append("empty tool_calls array in a delta")
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
    if roles != 1:
        errors.append(f"{roles} role events")
    if done != 1:
        errors.append(f"{done} [DONE] markers")
    if len(ids) != 1:
        errors.append(f"{len(ids)} distinct chunk ids")
    if finish is None:
        errors.append("no finish_reason")
    call_ids = [c["id"] for c in calls.values()]
    if len(set(call_ids)) != len(call_ids) or any(i is None for i in call_ids):
        errors.append("tool call ids missing or repeated")
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": [(calls[i]["name"], calls[i]["arguments"]) for i in sorted(calls)],
        "finish": finish,
        "usage": usage,
    }


# ---------------------------------------------------------------- Anthropic messages

def anthropic_nonstream(http: Http) -> dict[str, Any]:
    if http.status != 200:
        raise AssertionError(f"HTTP {http.status}: {http.body[:300]!r}")
    value = json.loads(http.body)
    return {"blocks": anthropic_blocks(value["content"]), "stop_reason": value.get("stop_reason"),
            "stop_sequence": value.get("stop_sequence", "<absent>"), "usage": value.get("usage"),
            "type": value.get("type"), "role": value.get("role")}


def anthropic_blocks(content: list[dict[str, Any]]) -> list[tuple[str, str]]:
    blocks = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            blocks.append(("text", block.get("text", "")))
        elif kind == "thinking":
            blocks.append(("thinking", block.get("thinking", "")))
        elif kind == "tool_use":
            blocks.append(("tool_use", block.get("name", "") + " " +
                           json.dumps(block.get("input"), sort_keys=True)))
        else:
            blocks.append((kind or "?", json.dumps(block, sort_keys=True)))
    return blocks


def anthropic_stream(http: Http, errors: list[str]) -> dict[str, Any]:
    if http.status != 200 or http.content_type != "text/event-stream":
        raise AssertionError(f"stream HTTP {http.status} {http.content_type}: {http.body[:300]!r}")
    events = []
    for block in http.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        # An SSE comment (the server opens a stream with ": stream") carries no event.
        if not block.strip() or all(line.startswith(":") for line in block.splitlines()):
            continue
        name = [line[7:] for line in block.splitlines() if line.startswith("event: ")]
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(name) != 1 or len(data) != 1:
            errors.append(f"malformed Anthropic SSE block {block[:80]!r}")
            continue
        payload = json.loads(data[0])
        if payload.get("type") != name[0]:
            errors.append(f"event name {name[0]} != data.type {payload.get('type')}")
        events.append(payload)
    types = [e.get("type") for e in events if e.get("type") != "ping"]
    if not types or types[0] != "message_start" or types[-1] != "message_stop":
        errors.append(f"event order {types[:3]}...{types[-2:]}")
    start = next((e for e in events if e.get("type") == "message_start"), {})
    message = start.get("message", {})
    for key in ("id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"):
        if key not in message:
            errors.append(f"message_start.message lacks {key}")
    if message.get("type") != "message" or message.get("role") != "assistant":
        errors.append("message_start type/role")
    blocks: dict[int, dict[str, Any]] = {}
    open_index = None
    next_index = 0
    stop_reason = None
    stop_sequence = "<absent>"
    usage = None
    for e in events:
        kind = e.get("type")
        if kind == "content_block_start":
            if open_index is not None:
                errors.append("block started while another is open")
            if e.get("index") != next_index:
                errors.append(f"block index {e.get('index')} != {next_index}")
            open_index = e.get("index")
            next_index += 1
            cb = e.get("content_block", {})
            blocks[open_index] = {"type": cb.get("type"), "text": "", "name": cb.get("name", ""),
                                  "json": ""}
            if cb.get("type") == "tool_use" and not isinstance(cb.get("input"), dict):
                errors.append("tool_use start input is not an object")
        elif kind == "content_block_delta":
            if e.get("index") != open_index:
                errors.append("delta for a block that is not open")
                continue
            delta = e.get("delta", {})
            block = blocks[open_index]
            dt = delta.get("type")
            if dt == "text_delta" and block["type"] == "text":
                block["text"] += delta.get("text", "")
            elif dt == "thinking_delta" and block["type"] == "thinking":
                block["text"] += delta.get("thinking", "")
            elif dt == "input_json_delta" and block["type"] == "tool_use":
                block["json"] += delta.get("partial_json", "")
            elif dt == "signature_delta" and block["type"] == "thinking":
                pass
            else:
                errors.append(f"delta {dt} inside a {block['type']} block")
        elif kind == "content_block_stop":
            if e.get("index") != open_index:
                errors.append("stop for a block that is not open")
            open_index = None
        elif kind == "message_delta":
            delta = e.get("delta", {})
            if "stop_reason" not in delta or "stop_sequence" not in delta:
                errors.append("message_delta lacks stop_reason or stop_sequence")
            stop_reason = delta.get("stop_reason")
            stop_sequence = delta.get("stop_sequence", "<absent>")
            usage = e.get("usage")
    if open_index is not None:
        errors.append("a block was never stopped")
    out = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "tool_use":
            try:
                parsed = json.loads(b["json"] or "{}")
            except json.JSONDecodeError:
                errors.append("tool_use partial_json does not assemble to JSON")
                parsed = None
            if not isinstance(parsed, dict):
                errors.append("tool_use input is not an object")
            out.append(("tool_use", b["name"] + " " + json.dumps(parsed, sort_keys=True)))
        else:
            out.append((b["type"], b["text"]))
    return {"blocks": out, "stop_reason": stop_reason, "stop_sequence": stop_sequence,
            "usage": usage}


# ---------------------------------------------------------------- the checks

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}
TIME_TOOL = {"type": "function", "function": {
    "name": "get_time", "description": "Current local time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


EMPTY_OPENAI = {"content": "", "reasoning": "", "tool_calls": [], "tool_calls_raw": None,
                "finish": None, "usage": {}, "choices": 0}
EMPTY_ANTHROPIC = {"blocks": [], "stop_reason": None, "stop_sequence": None, "usage": {},
                   "type": None, "role": None}


def pair_openai(report: Report, client: Client, case: str, payload: dict[str, Any]) -> tuple:
    try:
        return _pair_openai(report, client, case, payload)
    except (AssertionError, KeyError, IndexError, TypeError, ValueError, OSError) as error:
        report.add(case, "request completed", False, f"{type(error).__name__}: {error}"[:500])
        return dict(EMPTY_OPENAI), dict(EMPTY_OPENAI)


def pair_anthropic(report: Report, client: Client, case: str, payload: dict[str, Any]) -> tuple:
    try:
        return _pair_anthropic(report, client, case, payload)
    except (AssertionError, KeyError, IndexError, TypeError, ValueError, OSError) as error:
        report.add(case, "request completed", False, f"{type(error).__name__}: {error}"[:500])
        return dict(EMPTY_ANTHROPIC), dict(EMPTY_ANTHROPIC)


def _pair_openai(report: Report, client: Client, case: str, payload: dict[str, Any]) -> tuple:
    """Runs payload non-stream and stream (with usage) and compares them."""
    ns = openai_nonstream(client.call("POST", "/v1/chat/completions", payload))
    errors: list[str] = []
    streamed = dict(payload, stream=True, stream_options={"include_usage": True})
    st = openai_stream(client.call("POST", "/v1/chat/completions", streamed), errors)
    report.add(case, "SSE shape", not errors, errors or None)
    for key in ("content", "reasoning", "tool_calls", "finish"):
        report.add(case, f"stream == non-stream: {key}", ns[key] == st[key],
                   None if ns[key] == st[key] else {"non_stream": ns[key], "stream": st[key]})
    # cached_tokens reports prefix-cache state, which the first of the pair changes for the second.
    def comparable(usage):
        usage = json.loads(json.dumps(usage or {}))
        usage.get("prompt_tokens_details", {}).pop("cached_tokens", None)
        return usage
    same_usage = comparable(ns["usage"]) == comparable(st["usage"])
    report.add(case, "stream usage == non-stream usage", same_usage,
               None if same_usage else {"non_stream": ns["usage"], "stream": st["usage"]})
    usage = ns["usage"] or {}
    report.add(case, "total = prompt + completion",
               usage.get("total_tokens") == usage.get("prompt_tokens", 0) +
               usage.get("completion_tokens", 0), usage)
    limit = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if limit is not None:
        report.add(case, "completion_tokens <= limit", usage.get("completion_tokens", 0) <= limit,
                   usage)
    if ns["tool_calls_raw"] is not None:
        report.add(case, "no empty tool_calls array", len(ns["tool_calls_raw"]) > 0)
        for call in ns["tool_calls_raw"]:
            args = call["function"]["arguments"]
            ok = isinstance(args, str)
            try:
                ok = ok and isinstance(json.loads(args), dict)
            except (TypeError, json.JSONDecodeError):
                ok = False
            report.add(case, "arguments is a JSON-object string", ok, args)
    return ns, st


def _pair_anthropic(report: Report, client: Client, case: str, payload: dict[str, Any]) -> tuple:
    ns = anthropic_nonstream(client.call("POST", "/v1/messages", payload))
    report.add(case, "message envelope", ns["type"] == "message" and ns["role"] == "assistant" and
               ns["stop_sequence"] != "<absent>", {k: ns[k] for k in ("type", "role",
                                                                      "stop_sequence")})
    errors: list[str] = []
    st = anthropic_stream(client.call("POST", "/v1/messages", dict(payload, stream=True)), errors)
    report.add(case, "SSE shape", not errors, errors or None)
    for key in ("blocks", "stop_reason", "stop_sequence"):
        report.add(case, f"stream == non-stream: {key}", ns[key] == st[key],
                   None if ns[key] == st[key] else {"non_stream": ns[key], "stream": st[key]})
    out_ns = (ns["usage"] or {}).get("output_tokens")
    out_st = (st["usage"] or {}).get("output_tokens")
    report.add(case, "stream output_tokens == non-stream", out_ns == out_st,
               {"non_stream": out_ns, "stream": out_st})
    return ns, st


def expect_error(report: Report, client: Client, case: str, path: str, payload: Any = None,
                 raw: bytes | None = None, anthropic: bool = False) -> None:
    """A rejected request: 4xx, the dialect's error shape, and for a stream no SSE at all."""
    for stream in (False, True):
        body = raw
        data = payload
        if raw is None and isinstance(payload, dict):
            data = dict(payload, stream=True) if stream else payload
        elif raw is not None and stream:
            body = raw.replace(b'"stream": false', b'"stream": true')
        http = client.call("POST", path, data, raw=body)
        label = f"{case} ({'stream' if stream else 'non-stream'})"
        ok = 400 <= http.status < 500 and http.content_type == "application/json"
        detail: Any = {"status": http.status, "content_type": http.content_type}
        try:
            value = json.loads(http.body)
            if anthropic:
                ok = ok and value.get("type") == "error" and isinstance(
                    value.get("error", {}).get("message"), str) and isinstance(
                    value["error"].get("type"), str)
            else:
                err = value.get("error", {})
                ok = ok and isinstance(err.get("message"), str) and isinstance(err.get("type"), str)
                ok = ok and "param" in err and "code" in err
            detail["error"] = value.get("error")
        except (json.JSONDecodeError, AttributeError):
            ok = False
            detail["body"] = http.body[:200].decode("utf-8", "replace")
        report.add(label, "4xx in the dialect error shape, no SSE", ok, None if ok else detail)


def check_rejections(report: Report, client: Client, model: str) -> None:
    base = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    table = {
        "n=2": {"n": 2},
        "logprobs=true": {"logprobs": True},
        "top_logprobs=2": {"top_logprobs": 2},
        "nonzero logit_bias": {"logit_bias": {"1": 5}},
        "temperature as string": {"temperature": "hot"},
        "temperature out of range": {"temperature": 7},
        "top_p out of range": {"top_p": 1.5},
        "top_k out of range": {"top_k": 99},
        "max_tokens as string": {"max_tokens": "8"},
        "seventeen stop strings": {"stop": [f"s{i}" for i in range(17)]},
        "stop not a string": {"stop": 5},
        "messages not an array": {"messages": "hi"},
        "unknown role": {"messages": [{"role": "wizard", "content": "hi"}]},
        "tool_choice names a missing tool": {"tools": [WEATHER_TOOL], "tool_choice": {
            "type": "function", "function": {"name": "nope"}}},
        "response_format unknown type": {"response_format": {"type": "yaml"}},
        "seed as string": {"seed": "x"},
        "presence_penalty out of range": {"presence_penalty": 3},
        "tools not an array": {"tools": {"a": 1}},
    }
    for name, patch in table.items():
        expect_error(report, client, f"reject {name}", "/v1/chat/completions", dict(base, **patch))
    missing_model = {k: v for k, v in base.items() if k != "model"}
    expect_error(report, client, "reject missing model", "/v1/chat/completions", missing_model)
    # Invalid UTF-8 inside a JSON string: raw bytes 0xff 0xfe.
    raw = (b'{"model": "' + model.encode() + b'", "stream": false, "max_tokens": 4, '
           b'"messages": [{"role": "user", "content": "bad \xff\xfe bytes"}]}')
    expect_error(report, client, "reject invalid UTF-8", "/v1/chat/completions", raw=raw)
    expect_error(report, client, "reject malformed JSON", "/v1/chat/completions",
                 raw=b'{"model": "x", "stream": false, "messages": [')
    anthropic_base = {"model": model, "max_tokens": 8, "messages": [{"role": "user",
                                                                     "content": "hi"}]}
    for name, patch in {"temperature out of range": {"temperature": 3},
                        "messages not an array": {"messages": 1},
                        "top_k as string": {"top_k": "a"}}.items():
        expect_error(report, client, f"anthropic reject {name}", "/v1/messages",
                     dict(anthropic_base, **patch), anthropic=True)
    # Anthropic declares max_tokens required; NInferno applies its default instead (documented
    # leniency). Either answer is acceptable, a 5xx is not.
    lenient = client.call("POST", "/v1/messages",
                          {k: v for k, v in anthropic_base.items() if k != "max_tokens"})
    report.add("anthropic max_tokens missing", "200 with a default or 4xx, never 5xx",
               lenient.status < 500, {"status": lenient.status, "note": "lenient default"
                                      if lenient.status == 200 else "rejected"})
    raw = (b'{"model": "' + model.encode() + b'", "stream": false, "max_tokens": 4, '
           b'"messages": [{"role": "user", "content": "bad \xc3\x28 bytes"}]}')
    expect_error(report, client, "anthropic reject invalid UTF-8", "/v1/messages", raw=raw,
                 anthropic=True)


def validate_schema(value: Any, schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        if any(k not in value for k in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
                k not in schema.get("properties", {}) for k in value):
            return False
        return all(validate_schema(value[k], s) for k, s in schema.get("properties", {}).items()
                   if k in value)
    if kind == "array":
        return isinstance(value, list) and all(validate_schema(v, schema.get("items", {}))
                                               for v in value)
    if kind == "string":
        return isinstance(value, str) and ("enum" not in schema or value in schema["enum"])
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return True


def exercise(client: Client, model: str) -> Report:
    report = Report()
    greedy = {"model": model, "temperature": 0}

    def chat(content: str, **extra: Any) -> dict[str, Any]:
        return dict(greedy, messages=[{"role": "user", "content": content}], **extra)

    # stream == non-stream on the plain paths.
    ns, _ = pair_openai(report, client, "plain", chat(
        "Write two sentences about lighthouses.", max_tokens=64,
        chat_template_kwargs={"enable_thinking": False}))
    pair_openai(report, client, "thinking", chat(
        "What is 17 * 23? Think briefly.", max_tokens=160,
        chat_template_kwargs={"enable_thinking": True}))
    # A stop string taken from the model's own output, then stream == non-stream.
    words = ns["content"].split()
    if len(words) > 6:
        stop = words[5]
        ns_stop, _ = pair_openai(report, client, "stop string", chat(
            "Write two sentences about lighthouses.", max_tokens=64, stop=[stop],
            chat_template_kwargs={"enable_thinking": False}))
        report.add("stop string", "output ends before the stop and finish is stop",
                   stop not in ns_stop["content"] and ns_stop["finish"] == "stop",
                   {"stop": stop, "content": ns_stop["content"], "finish": ns_stop["finish"]})
    # A length cut inside multi-byte text at every small budget.
    for limit in (1, 2, 3, 5, 8):
        ns_len, _ = pair_openai(report, client, f"length {limit} in CJK/emoji", chat(
            "Reply only with: 北京欢迎你 🌏🚀 東京 😀", max_tokens=limit,
            chat_template_kwargs={"enable_thinking": False}))
        report.add(f"length {limit} in CJK/emoji", "finish is length",
                   ns_len["finish"] == "length", ns_len["finish"])
    # tool_choice none/auto/required/named, stream == non-stream.
    tool_question = "What is the weather in Oslo and the time in Tokyo? Use the tools."
    for choice in ("none", "auto", "required", {"type": "function",
                                                "function": {"name": "get_time"}}):
        label = f"tool_choice {choice if isinstance(choice, str) else 'named'}"
        ns_tool, _ = pair_openai(report, client, label, chat(
            tool_question, max_tokens=256, tools=[WEATHER_TOOL, TIME_TOOL], tool_choice=choice,
            chat_template_kwargs={"enable_thinking": False}))
        if choice == "none":
            report.add(label, "no tool calls", not ns_tool["tool_calls"], ns_tool["tool_calls"])
        if choice == "required":
            report.add(label, "at least one call, finish tool_calls",
                       len(ns_tool["tool_calls"]) >= 1 and ns_tool["finish"] == "tool_calls",
                       ns_tool["finish"])
        if isinstance(choice, dict):
            report.add(label, "only the named tool",
                       bool(ns_tool["tool_calls"]) and
                       all(name == "get_time" for name, _ in ns_tool["tool_calls"]),
                       ns_tool["tool_calls"])
    # A length cut inside a call: the partial markup is never a structured call.
    ns_cut, _ = pair_openai(report, client, "length inside a call", chat(
        tool_question, max_tokens=12, tools=[WEATHER_TOOL, TIME_TOOL], tool_choice="auto",
        chat_template_kwargs={"enable_thinking": False}))
    report.add("length inside a call", "finish length, no structured call",
               ns_cut["finish"] == "length" and not ns_cut["tool_calls"], ns_cut["finish"])
    # History with a previous call: arguments as a JSON string is the OpenAI form.
    history = [{"role": "user", "content": "Weather in Oslo?"},
               {"role": "assistant", "content": None, "tool_calls": [{
                   "id": "call_1", "type": "function", "function": {
                       "name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 7}"}]
    pair_openai(report, client, "tool result in history", dict(
        greedy, messages=history, tools=[WEATHER_TOOL], max_tokens=64,
        chat_template_kwargs={"enable_thinking": False}))
    # response_format json_schema is enforced; the output satisfies the schema.
    schema = {"type": "object", "properties": {
        "city": {"type": "string"}, "population": {"type": "integer"},
        "landmarks": {"type": "array", "items": {"type": "string"}}},
        "required": ["city", "population", "landmarks"], "additionalProperties": False}
    ns_json, _ = pair_openai(report, client, "json_schema", chat(
        "Describe Paris.", max_tokens=200, chat_template_kwargs={"enable_thinking": False},
        response_format={"type": "json_schema", "json_schema": {
            "name": "city", "schema": schema, "strict": True}}))
    try:
        parsed = json.loads(ns_json["content"])
        ok = validate_schema(parsed, schema)
    except json.JSONDecodeError:
        ok = ns_json["finish"] == "length"  # a length cut may end the JSON early
        parsed = ns_json["content"]
    report.add("json_schema", "output parses and satisfies the schema", ok, parsed)
    # Constrained decoding: the same json_schema request four times, alternating
    # non-stream and stream, must give one content and one completion-token count.
    runs = []
    json_payload = chat("Describe Paris.", max_tokens=200,
                        chat_template_kwargs={"enable_thinking": False},
                        response_format={"type": "json_schema", "json_schema": {
                            "name": "city", "schema": schema, "strict": True}})
    for i in range(4):
        try:
            if i % 2 == 0:
                got = openai_nonstream(client.call("POST", "/v1/chat/completions", json_payload))
            else:
                got = openai_stream(client.call("POST", "/v1/chat/completions", dict(
                    json_payload, stream=True, stream_options={"include_usage": True})), [])
            runs.append((got["content"], (got["usage"] or {}).get("completion_tokens")))
        except (AssertionError, KeyError, ValueError, OSError) as error:
            runs.append((f"ERROR {error}", None))
    report.add("json_schema repeat", "4 alternating runs agree on content and token count",
               len(set(runs)) == 1, [{"completion_tokens": n, "content": c} for c, n in runs])
    # A seeded sampled request repeats byte for byte; another seed is independent.
    sampled = chat("Invent a name for a new color and describe it.", max_tokens=48, temperature=0.9,
                   top_p=0.95, seed=4242, chat_template_kwargs={"enable_thinking": False})
    first = openai_nonstream(client.call("POST", "/v1/chat/completions", sampled))
    second = openai_nonstream(client.call("POST", "/v1/chat/completions", sampled))
    report.add("seeded sampling", "same seed repeats byte for byte",
               first["content"] == second["content"], [first["content"], second["content"]])
    pair_openai(report, client, "seeded sampling stream", sampled)
    # Anthropic dialect: plain, thinking, tools, stop sequence.
    abase = {"model": model, "temperature": 0}
    pair_anthropic(report, client, "anthropic plain", dict(
        abase, max_tokens=64, messages=[{"role": "user", "content": "Name three rivers."}]))
    pair_anthropic(report, client, "anthropic thinking", dict(
        abase, max_tokens=1200, thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{"role": "user", "content": "What is 19 * 21?"}]))
    anthropic_tools = [{"name": "get_weather", "description": "Current weather for a city.",
                        "input_schema": WEATHER_TOOL["function"]["parameters"]}]
    ns_at, _ = pair_anthropic(report, client, "anthropic tools", dict(
        abase, max_tokens=256, tools=anthropic_tools, tool_choice={"type": "any"},
        messages=[{"role": "user", "content": "Weather in Oslo?"}]))
    report.add("anthropic tools", "stop_reason tool_use with a tool_use block",
               ns_at["stop_reason"] == "tool_use" and any(b[0] == "tool_use"
                                                           for b in ns_at["blocks"]),
               ns_at["stop_reason"])
    ns_as, _ = pair_anthropic(report, client, "anthropic stop sequence", dict(
        abase, max_tokens=64, stop_sequences=["3"],
        messages=[{"role": "user", "content": "Count from 1 to 5, separated by spaces."}]))
    report.add("anthropic stop sequence", "stop_reason stop_sequence names the sequence",
               ns_as["stop_reason"] != "stop_sequence" or ns_as["stop_sequence"] == "3",
               {"stop_reason": ns_as["stop_reason"], "stop_sequence": ns_as["stop_sequence"]})
    # max_tokens = 0 is accepted by the parser; it must be a clean 200 or a clean 4xx.
    zero = client.call("POST", "/v1/chat/completions", chat("hi", max_tokens=0))
    report.add("max_tokens=0", "clean 200 or 4xx, never 5xx", zero.status < 500,
               {"status": zero.status, "body": zero.body[:300].decode("utf-8", "replace")})
    check_rejections(report, client, model)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    report = exercise(client, args.model)
    print(json.dumps({"format": "ninferno_serve_conformance_v1", "checks": len(report.checks),
                      "failures": report.failures, "results": report.checks},
                     ensure_ascii=False, indent=1))
    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
