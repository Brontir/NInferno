"""The transformers side of ninferno_qwen3_5_chat_template_parity_test.

    python tests/models/qwen3_5/hf_chat_template_parity.py TOKENIZER_DIR OUT_DIR

TOKENIZER_DIR holds a checkpoint's frontend files (tokenizer.json, tokenizer_config.json,
generation_config.json, chat_template.jinja, optionally preprocessor_config.json). OUT_DIR receives
copies of them and cases.jsonl: a matrix of OpenAI Chat Completions requests, each with the prompt
transformers renders for it (apply_chat_template, tokenize=False) and that prompt's ids.

The reference is what an OpenAI-compatible server built on transformers hands the template: the
request's messages as sent, except that assistant tool-call arguments (a JSON string on the wire)
are decoded to an object, the request's tools as sent, add_generation_prompt unless the request
continues its final assistant message, and chat_template_kwargs plus a top-level reasoning_effort
as template variables. The ids are the rendered text encoded by the checkpoint's own tokenizer.json
through `tokenizers` (special tokens parsed, none added), which is what tokenize=True does without
transformers 5's rebuilt pre-tokenizer (see hf_tokenizer_parity.py). An image is one
<|vision_start|><|image_pad|><|vision_end|> in the text; the C++ side collapses the expanded pad
run before comparing.

The matrix: dialogs (one user turn; system and user; multi-turn; developer turn; tool call with
its result; CJK and emoji tool arguments; reasoning in an earlier assistant turn; an image part;
content as a list of text parts) x tools {none, two tools whose keys and properties are not in
alphabetical order} x enable_thinking {unset, true, false} x reasoning_effort {unset, "low"} x
preserve_thinking {unset, true}; plus assistant-prefix continuations with thinking off.
"""

from __future__ import annotations

import base64
import copy
import itertools
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def png(width, height, rgb):
    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


IMAGE_URL = "data:image/png;base64," + base64.b64encode(png(64, 64, (200, 30, 30))).decode()

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {
            "zeta": {"type": "string", "description": "город"},
            "alpha": {"type": "integer"},
            "mid": {"type": "object", "properties": {"z": {"type": "boolean"}, "a": {"type": "number"}}}},
            "required": ["zeta"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Web search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}}},
]

CALL_ARGS = '{"zeta":"北京","alpha":3,"mid":{"z":true,"a":1.5}}'
EMOJI_ARGS = '{"q":"天気 😀 \\"quoted\\" }","n":2}'

DIALOGS = {
    "user": [{"role": "user", "content": "What is 2+2?"}],
    "system_user": [{"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Name a colour."}],
    "multi_turn": [{"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": "Hello! How can I help?"},
                   {"role": "user", "content": "Tell me a joke."}],
    "developer": [{"role": "developer", "content": "Answer in French."},
                  {"role": "user", "content": "Good morning"}],
    "tool_result": [{"role": "user", "content": "Weather in Beijing?"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "call_1", "type": "function",
                                     "function": {"name": "get_weather", "arguments": CALL_ARGS}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "晴，25°C"},
                    {"role": "user", "content": "And tomorrow?"}],
    "two_calls_cjk": [{"role": "user", "content": "搜索一下"},
                      {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "c1", "type": "function",
                                       "function": {"name": "search", "arguments": EMOJI_ARGS}},
                                      {"id": "c2", "type": "function",
                                       "function": {"name": "get_weather", "arguments": CALL_ARGS}}]},
                      {"role": "tool", "tool_call_id": "c1", "content": "result one"},
                      {"role": "tool", "tool_call_id": "c2", "content": "result two"}],
    "past_reasoning": [{"role": "user", "content": "First question"},
                       {"role": "assistant", "content": "First answer",
                        "reasoning_content": "I think about the first question."},
                       {"role": "user", "content": "Second question"}],
    "image": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE_URL}},
                                          {"type": "text", "text": "What colour is this?"}]}],
    "text_parts": [{"role": "user", "content": [{"type": "text", "text": "Part one. "},
                                               {"type": "text", "text": "Part two."}]}],
}

CONTINUATIONS = {
    "prefix_plain": "The answer is",
    "prefix_think": "<think>\nquoted",
    "prefix_cjk": "答案是",
}


def hf_messages(messages):
    """The messages a transformers-based server renders: tool-call arguments decoded to objects;
    list content kept as parts (text parts carry "text", image parts become {"type": "image"})."""
    out = copy.deepcopy(messages)
    for message in out:
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
        if isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if part["type"] == "image_url":
                    parts.append({"type": "image"})
                else:
                    parts.append(part)
            message["content"] = parts
    return out


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja", "preprocessor_config.json",
                 "video_preprocessor_config.json"):
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    auto = AutoTokenizer.from_pretrained(src)
    raw = Tokenizer.from_file(str(src / "tokenizer.json"))
    vision = ((src / "preprocessor_config.json").exists()
              and (src / "video_preprocessor_config.json").exists())

    cases = []

    def add(case_id, request, messages, add_generation_prompt, continue_final, kwargs, tools):
        try:
            text = auto.apply_chat_template(
                hf_messages(messages), tools=tools, add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final, tokenize=False, **kwargs)
        except Exception as error:  # the template's own refusal (raise_exception)
            cases.append({"id": case_id, "request": request, "refused": str(error), "ids": []})
            return
        ids = raw.encode(text, add_special_tokens=False).ids
        cases.append({"id": case_id, "request": request, "text": text, "ids": ids})

    for (dialog, messages), tools, thinking, effort, preserve in itertools.product(
            DIALOGS.items(), (None, TOOLS), (None, True, False), (None, "low"), (None, True)):
        if dialog == "image" and not vision:
            continue
        request = {"model": "m", "messages": messages, "max_tokens": 16}
        kwargs = {}
        template_kwargs = {}
        if tools:
            request["tools"] = tools
        if thinking is not None:
            template_kwargs["enable_thinking"] = thinking
        if preserve is not None:
            template_kwargs["preserve_thinking"] = preserve
        if template_kwargs:
            request["chat_template_kwargs"] = template_kwargs
            kwargs.update(template_kwargs)
        if effort is not None:
            request["reasoning_effort"] = effort
            kwargs["reasoning_effort"] = effort
        case_id = f"{dialog} tools={'2' if tools else '0'} thinking={thinking} effort={effort} preserve={preserve}"
        add(case_id, request, messages, True, False, kwargs, tools)
        if thinking is False and effort is not None:
            # NInferno policy: a reasoning effort with thinking disabled is a conflicting request
            # (400), where transformers renders whichever the template reads.
            cases[-1]["expect_ninferno_refusal"] = "reasoning effort conflicts with enable_thinking"

    # Assistant prefill: NInferno continues a final assistant message through the Anthropic Messages
    # dialect (with thinking disabled); the reference is continue_final_message=True.
    for name, prefix in CONTINUATIONS.items():
        messages = [{"role": "user", "content": "Finish the sentence."},
                    {"role": "assistant", "content": prefix}]
        request = {"model": "m", "messages": messages, "max_tokens": 16,
                   "thinking": {"type": "disabled"}}
        add(f"continuation {name}", request, messages, False, True, {"enable_thinking": False}, None)
        cases[-1]["dialect"] = "anthropic"
        if "<think>" in prefix:
            # NInferno policy: control-token text inside message content stays text (it never becomes
            # the control token), where tokenizing the rendered string turns it into one. The
            # rendered text must still agree.
            cases[-1]["literal_controls"] = True

    with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    refused = sum(1 for case in cases if "refused" in case)
    print(f"wrote {len(cases)} cases ({refused} refused by the template; vision "
          f"{'on' if vision else 'off'}) to {out / 'cases.jsonl'}")


if __name__ == "__main__":
    main()
