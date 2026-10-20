"""Official SDK clients against a running ninferno-serve.

The OpenAI and Anthropic Python SDKs are what most clients actually use; they validate response
shapes, accumulate streams and parse structured output themselves, so a response that a raw-HTTP
check (serve_conformance.py) accepts can still break them. Every case goes through the SDK's own
public API, greedy, and is a pair where one is possible:

OpenAI (base_url .../v1):
- models.list names the served model;
- chat.completions.create plain: non-stream == raw stream (stream=True, include_usage) ==
  the chat.completions.stream() helper's final completion, content, finish reason and usage;
- the async client streams the same bytes as the sync one;
- tools: tool_choice required and named, parsed as ChatCompletionMessageToolCall, arguments a
  JSON string; the stream helper accumulates the same calls; a tool result round trip answers;
- structured output: chat.completions.parse(response_format=<pydantic model>) and a
  pydantic_function_tool(strict) call parse into the model; both also streamed;
- Responses API: responses.create == responses.stream() final response; a function_call item and
  a function_call_output round trip; responses.parse(text_format=<model>);
  responses.input_tokens.count agrees with the usage of the same request;
- typed errors: an unsupported request raises openai.BadRequestError, not a generic APIError.

Anthropic (base_url is the server root):
- messages.create == messages.stream() (text_stream joined, get_final_message());
- thinking enabled: thinking and text blocks agree between stream and non-stream;
- tools: tool_choice tool / any give a tool_use block with a dict input, stream == non-stream;
  a tool_result round trip answers;
- structured output: messages.parse(output_format=<pydantic model>);
- messages.count_tokens agrees with usage.input_tokens of the same request;
- typed errors: anthropic.BadRequestError.

Agent-client shapes (the "trace replay" half, synthetic: no captured trace):
- Codex-like Responses turn pair: developer message, strict=false tool, reasoning effort,
  store=false, streamed; the first response's items are sent back with the tool output.
  reasoning.summary and include ["reasoning.encrypted_content"] are refused by server policy
  (typed 400s, checked): a client configured to send them is refused on its first request.
  The developer role renders as system (owner decision 2026-09-25: Qwen 3.5/3.6 templates raise
  "Unexpected message role" on developer, so the Codex turn was refused before; it passes since);
- Claude Code-like Messages turn pair: cached system blocks, metadata, anthropic-beta headers,
  $schema in tool schemas, thinking, streamed; thinking + tool_use sent back with tool_result.

The SDK versions are pinned (PINNED below, and tools/smoke/sdk_requirements.txt holds the full
resolved set); a different installed version fails the run unless --any-version is given, since
this file records what was verified. Install:
    python3 -m venv .venv-sdk
    .venv-sdk/bin/pip install -r tools/smoke/sdk_requirements.txt
Retries are off (max_retries=0) so a server error is never hidden by a retry.

Exit status 1 when any check fails; the JSON report goes to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

PINNED = {
    "python": "3.14.4",
    "openai": "3.19.2",
    "anthropic": "1.8.0",
    "httpx2": "2.13.1",
    "httpcore2": "2.13.1",
    "pydantic": "2.13.5",
}

import anthropic  # noqa: E402
import openai  # noqa: E402
import pydantic  # noqa: E402


@dataclass
class Report:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, case: str, check: str, ok: bool, detail: Any = None) -> bool:
        entry: dict[str, Any] = {"case": case, "check": check, "ok": bool(ok)}
        if detail is not None:
            entry["detail"] = detail
        self.checks.append(entry)
        return bool(ok)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if not c["ok"])

    def run(self, case: str, body: Callable[[], None]) -> None:
        """One case; an exception from the SDK is a failed check with the traceback tail."""
        try:
            body()
        except Exception as error:  # noqa: BLE001 - every SDK error is a finding here
            self.add(case, "completes without an exception", False,
                     {"type": type(error).__name__, "error": str(error)[:600],
                      "trace": traceback.format_exc().splitlines()[-4:]})


class City(pydantic.BaseModel):
    city: str
    population: int
    landmarks: list[str]


class WeatherArgs(pydantic.BaseModel):
    city: str


WEATHER_PARAMETERS = {"type": "object", "properties": {"city": {"type": "string"}},
                      "required": ["city"], "additionalProperties": False}
TIME_PARAMETERS = {"type": "object", "properties": {"timezone": {"type": "string"}},
                   "required": ["timezone"], "additionalProperties": False}
OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                      "parameters": WEATHER_PARAMETERS}},
    {"type": "function", "function": {"name": "get_time", "description": "Current time in a timezone.",
                                      "parameters": TIME_PARAMETERS}},
]
ANTHROPIC_TOOLS = [
    {"name": "get_weather", "description": "Current weather for a city.",
     "input_schema": WEATHER_PARAMETERS},
    {"name": "get_time", "description": "Current time in a timezone.", "input_schema": TIME_PARAMETERS},
]
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def versions() -> dict[str, str]:
    got = {"python": ".".join(str(v) for v in sys.version_info[:3])}
    for name in PINNED:
        if name == "python":
            continue
        try:
            got[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            got[name] = "missing"
    return got


def usage_tuple(usage: Any) -> tuple | None:
    if usage is None:
        return None
    return (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)


def calls_of(message: Any) -> list[tuple[str, Any]]:
    out = []
    for call in message.tool_calls or []:
        out.append((call.function.name, json.loads(call.function.arguments)))
    return out


def raw_stream(stream: Any) -> dict[str, Any]:
    """Accumulate a raw chat.completions stream the way a plain SDK user does."""
    content, reasoning, finish, usage = [], [], None, None
    calls: dict[int, dict[str, Any]] = {}
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            delta = choice.delta
            if delta.content:
                content.append(delta.content)
            extra = getattr(delta, "model_extra", None) or {}
            if extra.get("reasoning_content"):
                reasoning.append(extra["reasoning_content"])
            for call in delta.tool_calls or []:
                slot = calls.setdefault(call.index, {"name": "", "arguments": ""})
                if call.function and call.function.name:
                    slot["name"] += call.function.name
                if call.function and call.function.arguments:
                    slot["arguments"] += call.function.arguments
            if choice.finish_reason:
                finish = choice.finish_reason
    return {"content": "".join(content), "reasoning": "".join(reasoning), "finish": finish,
            "usage": usage_tuple(usage),
            "tool_calls": [(c["name"], json.loads(c["arguments"])) for _, c in sorted(calls.items())]}


def exercise_openai(report: Report, base_url: str, model: str) -> None:
    client = openai.OpenAI(base_url=base_url + "/v1", api_key="ninferno", max_retries=0, timeout=600)
    greedy = {"model": model, "temperature": 0}

    def models() -> None:
        ids = [m.id for m in client.models.list()]
        report.add("openai models.list", "lists the served model", model in ids, ids)
    report.run("openai models.list", models)

    messages = [{"role": "user", "content": "Write two sentences about lighthouses."}]

    def plain() -> None:
        ns = client.chat.completions.create(messages=messages, max_completion_tokens=64,
                                            extra_body=NO_THINK, **greedy)
        msg = ns.choices[0].message
        report.add("openai plain", "typed completion: str content, finish, usage",
                   isinstance(msg.content, str) and bool(msg.content) and
                   ns.choices[0].finish_reason in ("stop", "length") and ns.usage is not None,
                   {"finish": ns.choices[0].finish_reason, "content": msg.content})
        st = raw_stream(client.chat.completions.create(
            messages=messages, max_completion_tokens=64, stream=True,
            stream_options={"include_usage": True}, extra_body=NO_THINK, **greedy))
        report.add("openai plain", "raw stream == non-stream (content, finish, usage)",
                   (st["content"], st["finish"], st["usage"]) ==
                   (msg.content, ns.choices[0].finish_reason, usage_tuple(ns.usage)),
                   {"stream": st, "nonstream_usage": usage_tuple(ns.usage)})
        with client.chat.completions.stream(messages=messages, max_completion_tokens=64,
                                            stream_options={"include_usage": True},
                                            extra_body=NO_THINK, **greedy) as helper:
            deltas = [e.delta for e in helper if e.type == "content.delta"]
            final = helper.get_final_completion()
        report.add("openai plain", "stream() helper: deltas and final completion == non-stream",
                   "".join(deltas) == msg.content and final.choices[0].message.content == msg.content
                   and final.choices[0].finish_reason == ns.choices[0].finish_reason,
                   {"helper": final.choices[0].message.content})

        async def run_async() -> str:
            aclient = openai.AsyncOpenAI(base_url=base_url + "/v1", api_key="ninferno", max_retries=0,
                                         timeout=600)
            parts = []
            stream = await aclient.chat.completions.create(
                messages=messages, max_completion_tokens=64, stream=True, extra_body=NO_THINK, **greedy)
            async for chunk in stream:
                for choice in chunk.choices:
                    if choice.delta.content:
                        parts.append(choice.delta.content)
            await aclient.close()
            return "".join(parts)
        got = asyncio.run(run_async())
        report.add("openai plain", "async client stream == sync non-stream", got == msg.content, got)
    report.run("openai plain", plain)

    def thinking() -> None:
        q = [{"role": "user", "content": "What is 17 * 23? Think briefly."}]
        think = {"chat_template_kwargs": {"enable_thinking": True}}
        ns = client.chat.completions.create(messages=q, max_completion_tokens=400, extra_body=think,
                                            **greedy)
        msg = ns.choices[0].message
        reasoning = (msg.model_extra or {}).get("reasoning_content") or ""
        st = raw_stream(client.chat.completions.create(messages=q, max_completion_tokens=400,
                                                       stream=True, extra_body=think, **greedy))
        report.add("openai thinking", "reasoning_content present and stream == non-stream",
                   bool(reasoning) and (st["reasoning"], st["content"]) == (reasoning, msg.content or ""),
                   {"reasoning_chars": len(reasoning), "stream_reasoning_chars": len(st["reasoning"]),
                    "content": msg.content, "stream_content": st["content"]})
    report.run("openai thinking", thinking)

    question = [{"role": "user", "content": "What is the weather in Oslo and the time in Tokyo? Use the tools."}]

    def tools() -> None:
        for choice in ("required", {"type": "function", "function": {"name": "get_time"}}):
            label = "openai tools " + (choice if isinstance(choice, str) else "named")
            ns = client.chat.completions.create(messages=question, tools=OPENAI_TOOLS, tool_choice=choice,
                                                max_completion_tokens=256, extra_body=NO_THINK, **greedy)
            c = ns.choices[0]
            got = calls_of(c.message)
            ids = [t.id for t in c.message.tool_calls or []]
            ok = bool(got) and c.finish_reason == "tool_calls" and len(set(ids)) == len(ids)
            if isinstance(choice, dict):
                ok = ok and all(name == "get_time" for name, _ in got)
            report.add(label, "typed tool calls, JSON-string arguments, finish tool_calls, unique ids",
                       ok, {"finish": c.finish_reason, "calls": got})
            st = raw_stream(client.chat.completions.create(
                messages=question, tools=OPENAI_TOOLS, tool_choice=choice, max_completion_tokens=256,
                stream=True, extra_body=NO_THINK, **greedy))
            report.add(label, "raw stream accumulates the same calls",
                       st["tool_calls"] == got and st["finish"] == c.finish_reason, st["tool_calls"])
            with client.chat.completions.stream(messages=question, tools=OPENAI_TOOLS, tool_choice=choice,
                                                max_completion_tokens=256, extra_body=NO_THINK,
                                                **greedy) as helper:
                for _ in helper:
                    pass
                final = helper.get_final_completion()
            report.add(label, "stream() helper final completion has the same calls",
                       calls_of(final.choices[0].message) == got, calls_of(final.choices[0].message))
        first = client.chat.completions.create(messages=question[:1], tools=OPENAI_TOOLS[:1],
                                               tool_choice="required", max_completion_tokens=128,
                                               extra_body=NO_THINK, **greedy).choices[0].message
        # A model may call the tool more than once in a turn; a client answers every call.
        history = question[:1] + [first.model_dump(exclude_none=True)] + [
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp_c": 7, "sky": "rain"})}
            for call in first.tool_calls]
        answer = client.chat.completions.create(messages=history, tools=OPENAI_TOOLS[:1],
                                                max_completion_tokens=96, extra_body=NO_THINK, **greedy)
        report.add("openai tool round trip", "the SDK's own assistant message is accepted back; answer is text",
                   bool(answer.choices[0].message.content) and not answer.choices[0].message.tool_calls,
                   answer.choices[0].message.content)
    report.run("openai tools", tools)

    def structured() -> None:
        q = [{"role": "user", "content": "Describe Paris as JSON."}]
        parsed = client.chat.completions.parse(messages=q, response_format=City, max_completion_tokens=300,
                                               extra_body=NO_THINK, **greedy)
        value = parsed.choices[0].message.parsed
        report.add("openai parse(response_format=model)", "parsed into the pydantic model",
                   isinstance(value, City), value.model_dump() if value else parsed.choices[0].message.content)
        with client.chat.completions.stream(messages=q, response_format=City, max_completion_tokens=300,
                                            extra_body=NO_THINK, **greedy) as helper:
            for _ in helper:
                pass
            final = helper.get_final_completion()
        report.add("openai parse(response_format=model)", "stream() helper parses the same value",
                   final.choices[0].message.parsed == value,
                   final.choices[0].message.parsed.model_dump() if final.choices[0].message.parsed else None)
        tool = openai.pydantic_function_tool(WeatherArgs, name="get_weather",
                                             description="Current weather for a city.")
        parsed = client.chat.completions.parse(messages=[{"role": "user", "content": "Weather in Oslo?"}],
                                               tools=[tool], tool_choice="required", max_completion_tokens=128,
                                               extra_body=NO_THINK, **greedy)
        calls = parsed.choices[0].message.tool_calls or []
        report.add("openai pydantic_function_tool(strict)", "parsed_arguments is the pydantic model",
                   bool(calls) and isinstance(calls[0].function.parsed_arguments, WeatherArgs),
                   [c.function.arguments for c in calls])
    report.run("openai structured", structured)

    def responses() -> None:
        base = {"model": model, "temperature": 0, "max_output_tokens": 96, "extra_body": NO_THINK}
        ns = client.responses.create(input="Write two sentences about lighthouses.", **base)
        report.add("openai responses", "typed response with output_text and usage",
                   bool(ns.output_text) and ns.status in ("completed", "incomplete") and ns.usage is not None,
                   {"status": ns.status, "text": ns.output_text})
        with client.responses.stream(input="Write two sentences about lighthouses.", **base) as stream:
            deltas = [e.delta for e in stream if e.type == "response.output_text.delta"]
            final = stream.get_final_response()
        report.add("openai responses", "stream() deltas and final response == non-stream",
                   "".join(deltas) == ns.output_text and final.output_text == ns.output_text and
                   final.usage.output_tokens == ns.usage.output_tokens,
                   {"stream": final.output_text})
        count = client.responses.input_tokens.count(model=model, input="Write two sentences about lighthouses.",
                                                    extra_body=NO_THINK)
        report.add("openai responses", "input_tokens.count == usage.input_tokens",
                   count.input_tokens == ns.usage.input_tokens,
                   {"count": count.input_tokens, "usage": ns.usage.input_tokens})
        tool = {"type": "function", "name": "get_weather", "description": "Current weather for a city.",
                "parameters": WEATHER_PARAMETERS, "strict": True}
        first = client.responses.create(input="Weather in Oslo? Use the tool.", tools=[tool],
                                        tool_choice="required", **dict(base, max_output_tokens=128))
        calls = [item for item in first.output if item.type == "function_call"]
        report.add("openai responses tools", "a typed function_call item with JSON arguments",
                   bool(calls) and calls[0].name == "get_weather" and isinstance(json.loads(calls[0].arguments), dict),
                   [item.type for item in first.output])
        if calls:
            follow = client.responses.create(
                input=[{"role": "user", "content": "Weather in Oslo? Use the tool."},
                       calls[0].model_dump(exclude_none=True),
                       {"type": "function_call_output", "call_id": calls[0].call_id,
                        "output": json.dumps({"temp_c": 7, "sky": "rain"})}],
                tools=[tool], **base)
            report.add("openai responses tools", "function_call_output round trip answers in text",
                       bool(follow.output_text), follow.output_text)
        parsed = client.responses.parse(input="Describe Paris as JSON.", text_format=City,
                                        **dict(base, max_output_tokens=300))
        report.add("openai responses.parse(text_format=model)", "output_parsed is the pydantic model",
                   isinstance(parsed.output_parsed, City),
                   parsed.output_parsed.model_dump() if parsed.output_parsed else parsed.output_text)
    report.run("openai responses", responses)

    def errors() -> None:
        try:
            client.chat.completions.create(messages=messages, n=2, max_completion_tokens=8, **greedy)
            report.add("openai errors", "n=2 raises BadRequestError", False, "no exception")
        except openai.BadRequestError as error:
            report.add("openai errors", "n=2 raises BadRequestError with a message",
                       error.status_code == 400 and bool(error.message), error.message)
        try:
            client.chat.completions.create(messages=messages, n=2, max_completion_tokens=8, stream=True,
                                           **greedy)
            report.add("openai errors", "streamed n=2 raises BadRequestError before any event", False,
                       "no exception")
        except openai.BadRequestError as error:
            report.add("openai errors", "streamed n=2 raises BadRequestError before any event",
                       error.status_code == 400, error.message)
    report.run("openai errors", errors)


def blocks(message: Any) -> list[tuple[str, Any]]:
    out = []
    for block in message.content:
        if block.type == "text":
            out.append(("text", block.text))
        elif block.type == "thinking":
            out.append(("thinking", block.thinking))
        elif block.type == "tool_use":
            out.append(("tool_use", (block.name, block.input)))
        else:
            out.append((block.type, None))
    return out


def exercise_anthropic(report: Report, base_url: str, model: str) -> None:
    client = anthropic.Anthropic(base_url=base_url, api_key="ninferno", max_retries=0, timeout=600)
    greedy = {"model": model, "extra_body": {"temperature": 0}}
    rivers = [{"role": "user", "content": "Name three rivers."}]

    def plain() -> None:
        ns = client.messages.create(messages=rivers, max_tokens=64, thinking={"type": "disabled"}, **greedy)
        report.add("anthropic plain", "typed message: text block, stop_reason, usage",
                   ns.content and ns.content[0].type == "text" and ns.stop_reason in ("end_turn", "max_tokens")
                   and ns.usage.input_tokens > 0, {"stop": ns.stop_reason, "blocks": blocks(ns)})
        with client.messages.stream(messages=rivers, max_tokens=64, thinking={"type": "disabled"},
                                    **greedy) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        report.add("anthropic plain", "stream() text_stream and final message == non-stream",
                   text == ns.content[0].text and blocks(final) == blocks(ns) and
                   final.stop_reason == ns.stop_reason and final.usage.output_tokens == ns.usage.output_tokens,
                   {"stream": blocks(final), "usage": [final.usage.output_tokens, ns.usage.output_tokens]})
        count = client.messages.count_tokens(model=model, messages=rivers, thinking={"type": "disabled"})
        report.add("anthropic plain", "count_tokens == usage.input_tokens",
                   count.input_tokens == ns.usage.input_tokens,
                   {"count": count.input_tokens, "usage": ns.usage.input_tokens})

        async def run_async() -> list:
            aclient = anthropic.AsyncAnthropic(base_url=base_url, api_key="ninferno", max_retries=0, timeout=600)
            async with aclient.messages.stream(messages=rivers, max_tokens=64, thinking={"type": "disabled"},
                                               **greedy) as stream:
                final = await stream.get_final_message()
            await aclient.close()
            return blocks(final)
        got = asyncio.run(run_async())
        report.add("anthropic plain", "async client stream == sync non-stream", got == blocks(ns), got)
    report.run("anthropic plain", plain)

    def thinking() -> None:
        q = [{"role": "user", "content": "What is 19 * 21?"}]
        args = dict(messages=q, max_tokens=1200, thinking={"type": "enabled", "budget_tokens": 1024}, **greedy)
        ns = client.messages.create(**args)
        with client.messages.stream(**args) as stream:
            final = stream.get_final_message()
        kinds = [k for k, _ in blocks(ns)]
        report.add("anthropic thinking", "a thinking block then text; stream == non-stream",
                   "thinking" in kinds and blocks(final) == blocks(ns), {"kinds": kinds})
    report.run("anthropic thinking", thinking)

    def tools() -> None:
        q = [{"role": "user", "content": "What is the weather in Oslo and the time in Tokyo? Use the tools."}]
        for choice in ({"type": "any"}, {"type": "tool", "name": "get_time"}):
            label = "anthropic tools " + choice["type"]
            args = dict(messages=q, tools=ANTHROPIC_TOOLS, tool_choice=choice, max_tokens=256,
                        thinking={"type": "disabled"}, **greedy)
            ns = client.messages.create(**args)
            uses = [b for b in ns.content if b.type == "tool_use"]
            ok = ns.stop_reason == "tool_use" and bool(uses) and all(isinstance(b.input, dict) for b in uses)
            if choice["type"] == "tool":
                ok = ok and all(b.name == "get_time" for b in uses)
            report.add(label, "stop_reason tool_use, typed tool_use blocks with dict input", ok,
                       {"stop": ns.stop_reason, "blocks": blocks(ns)})
            with client.messages.stream(**args) as stream:
                final = stream.get_final_message()
            report.add(label, "stream() final message == non-stream", blocks(final) == blocks(ns), blocks(final))
        first = client.messages.create(messages=q[:1], tools=ANTHROPIC_TOOLS[:1], tool_choice={"type": "any"},
                                       max_tokens=128, thinking={"type": "disabled"}, **greedy)
        # Every tool_use block of the turn needs its tool_result (the server refuses a missing one).
        results = [{"type": "tool_result", "tool_use_id": b.id, "content": json.dumps({"temp_c": 7})}
                   for b in first.content if b.type == "tool_use"]
        follow = client.messages.create(
            messages=q[:1] + [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in first.content]},
                              {"role": "user", "content": results}],
            tools=ANTHROPIC_TOOLS[:1], max_tokens=96, thinking={"type": "disabled"}, **greedy)
        report.add("anthropic tool round trip", "the SDK's own content blocks are accepted back; answer is text",
                   follow.stop_reason in ("end_turn", "max_tokens") and any(b.type == "text" for b in follow.content),
                   blocks(follow))
    report.run("anthropic tools", tools)

    def structured() -> None:
        parsed = client.messages.parse(messages=[{"role": "user", "content": "Describe Paris as JSON."}],
                                       output_format=City, max_tokens=300, thinking={"type": "disabled"}, **greedy)
        value = getattr(parsed, "parsed_output", None)
        report.add("anthropic parse(output_format=model)", "parsed_output is the pydantic model",
                   isinstance(value, City), value.model_dump() if value else blocks(parsed))
    report.run("anthropic structured", structured)

    def errors() -> None:
        try:
            client.messages.create(messages=[], max_tokens=16, **greedy)
            report.add("anthropic errors", "empty messages raises BadRequestError", False, "no exception")
        except anthropic.BadRequestError as error:
            report.add("anthropic errors", "empty messages raises BadRequestError with a message",
                       error.status_code == 400 and bool(error.message), error.message)
    report.run("anthropic errors", errors)


SHELL_PARAMETERS = {"type": "object", "properties": {
    "command": {"type": "array", "items": {"type": "string"}, "description": "The command and its arguments."},
    "workdir": {"type": "string", "description": "The working directory."},
    "timeout_ms": {"type": "number"}}, "required": ["command"], "additionalProperties": False}


def replay_codex(report: Report, base_url: str, model: str) -> None:
    """A Codex-shaped agent turn through the Responses API: developer message, a strict=false function
    tool, reasoning with an encrypted-content include, store=false, a prompt_cache_key, streamed;
    then the second turn sends the first response's output items back (as a store=false client
    must) with a function_call_output. Shape only: no captured trace is replayed byte for byte."""
    client = openai.OpenAI(base_url=base_url + "/v1", api_key="ninferno", max_retries=0, timeout=600)
    tools = [{"type": "function", "name": "shell", "description": "Runs a shell command and returns its output.",
              "strict": False, "parameters": SHELL_PARAMETERS}]
    history: list[Any] = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text":
            "<permissions instructions>Commands run in a sandbox; the workspace is writable.</permissions instructions>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text":
            "<environment_context><cwd>/work/app</cwd><shell>bash</shell></environment_context>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text":
            "List the files in the current directory, then tell me how many there are."}]}]
    # reasoning.summary is refused by policy (400 reasoning_option_not_supported: the server does not
    # summarize reasoning and will not drop the request silently). A Codex configured with reasoning
    # summaries on (summary "auto") is therefore refused on its first request; the typed refusal is
    # checked here, and the replay runs with effort only.
    try:
        client.responses.create(model=model, input="hi", max_output_tokens=8,
                                reasoning={"effort": "low", "summary": "auto"})
        report.add("replay codex", "reasoning.summary is refused with a typed 400", False, "accepted")
    except openai.BadRequestError as error:
        report.add("replay codex", "reasoning.summary is refused with a typed 400",
                   error.status_code == 400 and error.code == "reasoning_option_not_supported",
                   {"code": error.code, "message": error.message})
    # include ["reasoning.encrypted_content"] (what a store=false client sends to carry reasoning
    # between turns) is refused the same way: 400 include_not_supported, no encrypted reasoning.
    try:
        client.responses.create(model=model, input="hi", max_output_tokens=8, store=False,
                                include=["reasoning.encrypted_content"])
        report.add("replay codex", "include reasoning.encrypted_content is refused with a typed 400",
                   False, "accepted")
    except openai.BadRequestError as error:
        report.add("replay codex", "include reasoning.encrypted_content is refused with a typed 400",
                   error.status_code == 400 and error.code == "include_not_supported",
                   {"code": error.code, "message": error.message})
    common = dict(model=model, instructions="You are a coding agent running in a terminal. Use the shell tool.",
                  tools=tools, tool_choice="auto", parallel_tool_calls=False,
                  reasoning={"effort": "low"}, store=False, prompt_cache_key="qa2-codex-replay",
                  max_output_tokens=1024)

    def turn(items: list[Any]) -> tuple[Any, list[str]]:
        kinds: list[str] = []
        final = None
        for event in client.responses.create(input=items, stream=True, **common):
            kinds.append(event.type)
            if event.type in ("response.completed", "response.incomplete", "response.failed"):
                final = event.response
        return final, kinds

    first, kinds = turn(history)
    report.add("replay codex turn 1", "stream ends in response.completed with output items",
               first is not None and kinds[-1] == "response.completed" and bool(first.output),
               {"last_events": kinds[-3:], "items": [i.type for i in (first.output if first else [])]})
    calls = [i for i in first.output if i.type == "function_call"] if first else []
    report.add("replay codex turn 1", "the agent calls the shell tool with a command array",
               bool(calls) and isinstance(json.loads(calls[0].arguments).get("command"), list),
               [c.arguments for c in calls])
    if not calls:
        return
    followup = history + [item.model_dump(exclude_none=True) for item in first.output] + [
        {"type": "function_call_output", "call_id": calls[0].call_id,
         "output": json.dumps({"output": "README.md\nmain.py\nsetup.cfg\n", "metadata": {"exit_code": 0}})}]
    second, kinds = turn(followup)
    report.add("replay codex turn 2", "the server's own items (reasoning included) are accepted back",
               second is not None and kinds[-1] == "response.completed" and
               (bool(second.output_text) or any(i.type == "function_call" for i in second.output)),
               {"last_events": kinds[-3:], "text": second.output_text if second else None})


def replay_claude_code(report: Report, base_url: str, model: str) -> None:
    """A Claude Code-shaped agent turn through the Messages API: system blocks with cache_control,
    metadata.user_id, anthropic-beta headers, tools whose input_schema carries $schema, interleaved
    thinking, a <system-reminder> text block, streamed; then the second turn sends the first
    message's content (thinking with its signature, tool_use) back with a tool_result. Shape only:
    no captured trace is replayed byte for byte."""
    client = anthropic.Anthropic(base_url=base_url, api_key="ninferno", max_retries=0, timeout=600)
    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
              "properties": {"command": {"type": "string", "description": "The command to execute"},
                             "description": {"type": "string"}},
              "required": ["command"], "additionalProperties": False}
    tools = [{"name": "Bash", "description": "Executes a bash command.", "input_schema": schema},
             {"name": "Read", "description": "Reads a file.", "input_schema": {
                 "$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                 "properties": {"file_path": {"type": "string"}}, "required": ["file_path"],
                 "additionalProperties": False}}]
    system = [{"type": "text", "text": "You are an interactive CLI coding agent.",
               "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "Working directory: /work/app. Use the tools to answer.",
               "cache_control": {"type": "ephemeral"}}]
    messages: list[Any] = [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>The task list is empty.</system-reminder>"},
        {"type": "text", "text": "How many files are in the working directory?",
         "cache_control": {"type": "ephemeral"}}]}]
    common = dict(model=model, system=system, tools=tools, max_tokens=4096,
                  thinking={"type": "enabled", "budget_tokens": 1024},
                  metadata={"user_id": "user_qa2_account__session_replay"},
                  extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14,"
                                                   "fine-grained-tool-streaming-2025-05-14"},
                  extra_body={"temperature": 1})

    def turn(history: list[Any]) -> Any:
        with client.messages.stream(messages=history, **common) as stream:
            for _ in stream:
                pass
            return stream.get_final_message()

    first = turn(messages)
    uses = [b for b in first.content if b.type == "tool_use"]
    report.add("replay claude code turn 1", "stop_reason tool_use with a Bash/Read call",
               first.stop_reason == "tool_use" and bool(uses) and uses[0].name in ("Bash", "Read"),
               {"stop": first.stop_reason, "kinds": [b.type for b in first.content]})
    if not uses:
        return
    followup = messages + [
        {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in first.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": u.id,
                                      "content": "README.md\nmain.py\nsetup.cfg"} for u in uses]}]
    second = turn(followup)
    report.add("replay claude code turn 2", "the server's own thinking and tool_use blocks are accepted back",
               second.stop_reason in ("end_turn", "tool_use", "max_tokens") and bool(second.content),
               {"stop": second.stop_reason, "kinds": [b.type for b in second.content]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080", help="server root, no /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--any-version", action="store_true",
                        help="run with SDK versions other than the pinned ones (recorded, not failed)")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    report = Report()
    installed = versions()
    mismatch = {k: [PINNED[k], v] for k, v in installed.items() if v != PINNED[k]}
    report.add("sdk versions", "installed versions match the pin", not mismatch or args.any_version,
               {"installed": installed, "mismatch": mismatch})
    exercise_openai(report, base_url, args.model)
    exercise_anthropic(report, base_url, args.model)
    report.run("replay codex", lambda: replay_codex(report, base_url, args.model))
    report.run("replay claude code", lambda: replay_claude_code(report, base_url, args.model))
    print(json.dumps({"format": "ninferno_serve_sdk_conformance_v1", "sdk": installed, "checks": len(report.checks),
                      "failures": report.failures, "results": report.checks},
                     ensure_ascii=False, indent=1, default=str))
    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
