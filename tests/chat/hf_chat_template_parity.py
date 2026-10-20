"""The transformers side of ninferno_chat_template_parity_test: every chat template NInferno serves, the
prompt ids of ninferno-serve against transformers' apply_chat_template(..., tokenize=True).

    python tests/chat/hf_chat_template_parity.py OUT_DIR [TAG ...]

For each model below (all by default, or the TAGs named), OUT_DIR/<tag>/ receives the frontend
files ninferno-serve reads from an artifact -- tokenizer.json, tokenizer_config.json,
chat_template.jinja, generation_config.json, derived from the checkpoint as the converter derives
them (src/convert/checks.cpp resource_fallbacks: a template kept in tokenizer_config.json, the
end-of-sequence ids when generation_config.json is missing) -- plus frontend.json (the serving
family, whose frontend the test builds as that family does) and cases.jsonl: OpenAI Chat
Completions requests with the ids transformers gives for the same messages. OUT_DIR/required.json
names every turn markup, reasoning markup and tool-call format of src/chat/data/conventions.json:
the test fails unless the templates below resolve to each of them at least once.

Only tokenizer files are fetched (no weights), at pinned revisions. The models: the current
generation of every model family NInferno serves, sizes >= 1B (a
template shared by several sizes is fetched once per template variant), and, marked "coverage",
a few older checkpoints kept only because their template is the one a convention entry describes
(Llama 3 turns and JSON tool calls, Granite 3.x tool calls).

The reference: transformers' apply_chat_template with tokenize=True on the request's messages as a
transformers-based server hands them over -- assistant tool-call arguments (a JSON string on the
wire) decoded to an object, the request's tools, add_generation_prompt, chat_template_kwargs as
template variables, a top-level reasoning_effort as one -- where NInferno's request contract does not
decide otherwise (hf_messages: a developer message is a system message, a null content is empty;
a request with tools for a template without tool-call syntax, or with a reasoning effort and
thinking disabled, is refused; a continuation needs the template's turn markup). When the checkpoint's tokenizer.json run
directly by `tokenizers` gives other ids for the rendered text (transformers 5 rebuilds some
pre-tokenizers from their class), the case keeps them as `tokenizers_ids`; the test accepts ids
equal to either and counts the cases where NInferno agrees with the checkpoint's tokenizer.json but
not with transformers (a transformers deviation from the shipped tokenizer, as
tests/models/qwen3_5/hf_tokenizer_parity.py records for Qwen 3.5).

The matrix per template: dialogs (one user turn; system and user; multi-turn; developer turn;
a tool call with its result; two parallel calls with CJK and emoji arguments; reasoning in an
earlier assistant turn) x tools {none, two whose keys are not in alphabetical order} x
enable_thinking {unset, true, false} when the template reads it x reasoning_effort {unset, "low"}
when the template reads it; plus an assistant-prefix continuation (Anthropic prefill).
"""

from __future__ import annotations

import copy
import itertools
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]

# tag, repo, revision, serving family (runtime/engine), why it is here.
MODELS = [
    ("qwen3_5_2b", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "qwen3_5", "lineup"),
    ("qwen3_5_9b", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "qwen3_5", "lineup"),
    ("qwen3_5_35b_a3b", "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754", "qwen3_5", "lineup"),
    ("qwen3_6_27b", "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "qwen3_5", "lineup"),
    ("qwen3_8_27b", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "qwen3_5", "lineup"),
    ("gemma4_e2b", "google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7", "hf_text", "lineup"),
    ("gemma4_12b", "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "hf_text", "lineup"),
    ("gemma4_26b_a4b", "google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", "hf_text", "lineup"),
    ("gemma4_31b", "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", "hf_text", "lineup"),
    ("muse_glimmer_30b", "meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68", "hf_text", "lineup"),
    ("ministral3_3b", "mistralai/Ministral-3-3B-Instruct-2512", "b35d4dfe56c142746f54dbd64f579faab2744308", "hf_text", "lineup"),
    ("ministral3_14b", "mistralai/Ministral-3-14B-Instruct-2512", "29439f81c2be264d8d393273f99e7db9c0961120", "hf_text", "lineup"),
    ("phi4_mini", "microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", "hf_text", "lineup"),
    ("phi4", "microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25", "hf_text", "lineup"),
    ("phi4_mini_reasoning", "microsoft/Phi-4-mini-reasoning", "0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a", "hf_text", "lineup"),
    ("granite4_2_3b", "ibm-granite/granite-4.2-3b", "e459acceac81e5fe67c07d9cfc72329a332e7eb1", "hf_text", "lineup"),
    ("granite4_2_8b", "ibm-granite/granite-4.2-8b", "f8de16cdcdbc6c779ca517604e050d82cc119e44", "hf_text", "lineup"),
    ("olmo3_7b_instruct", "allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc", "hf_text", "lineup"),
    ("olmo3_7b_think", "allenai/Olmo-3-7B-Think", "d97e442d7cc678210054dbcc9b440894d62c89a4", "hf_text", "lineup"),
    ("olmo2_1b", "allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", "hf_text", "lineup"),
    ("smollm3_3b", "HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", "hf_text", "lineup"),
    ("smollm2_1_7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", "hf_text", "lineup"),
    ("gpt_oss_20b", "openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", "hf_text", "lineup"),
    ("deepseek_r1_qwen3_8b", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "6e8885a6ff5c1dc5201574c8fd700323f23c25fa", "hf_text", "lineup"),
    ("deepseek_r1_qwen_1_5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562", "hf_text", "lineup"),
    ("gemma3_1b", "unsloth/gemma-3-1b-it", "5b11413a10db4e486ef16a20101fd028f8f2499c", "hf_text", "lineup"),
    ("llama3_2_1b", "unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c", "hf_text", "coverage"),
    ("granite3_3_2b", "ibm-granite/granite-3.3-2b-instruct", "707f574c62054322f6b5b04b6d075f0a8f05e0f0", "hf_text", "coverage"),
]

FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
         "config.json", "special_tokens_map.json")

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
}


def fetch(repo, revision, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        try:
            path = hf_hub_download(repo, name, revision=revision)
        except Exception:
            continue
        shutil.copyfile(path, dest / name)


def derive_resources(src):
    """The four frontend files as the converter derives them (resource_fallbacks)."""
    tokenizer_config = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not (src / "chat_template.jinja").exists():
        chat = tokenizer_config.get("chat_template")
        if isinstance(chat, list) and chat:
            chosen = chat[0]
            for item in chat:
                if item.get("name") == "default":
                    chosen = item
            chat = chosen.get("template")
        if not isinstance(chat, str):
            raise RuntimeError(f"{src}: no chat template")
        (src / "chat_template.jinja").write_text(chat, encoding="utf-8")
    if not (src / "generation_config.json").exists():
        config = json.loads((src / "config.json").read_text(encoding="utf-8"))
        stops = []

        def add(value):
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, int) and item not in stops:
                    stops.append(item)
        add(config.get("eos_token_id"))
        add((config.get("text_config") or {}).get("eos_token_id"))
        added = {}
        for key, token in (tokenizer_config.get("added_tokens_decoder") or {}).items():
            added[token.get("content")] = int(key)
        if not added:
            raw = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
            added = {token["content"]: token["id"] for token in raw.get("added_tokens", [])}

        def lookup(key):
            name = tokenizer_config.get(key)
            if isinstance(name, dict):
                name = name.get("content")
            return added.get(name) if isinstance(name, str) else None
        eos = lookup("eos_token")
        if eos is not None:
            add(eos)
        if not stops:
            raise RuntimeError(f"{src}: no end-of-sequence id")
        generation = {}
        if lookup("bos_token") is not None:
            generation["bos_token_id"] = lookup("bos_token")
        generation["eos_token_id"] = stops[0] if len(stops) == 1 else stops
        if lookup("pad_token") is not None:
            generation["pad_token_id"] = lookup("pad_token")
        (src / "generation_config.json").write_text(json.dumps(generation, indent=2), encoding="utf-8")


def hf_messages(messages):
    """The messages as the template receives them. As a transformers-based server hands them over:
    tool-call arguments (a JSON string on the wire) decoded to objects. As NInferno hands them over,
    by its request contract, not transformers': a developer message renders as a system message
    (src/models/qwen3_5/frontend/chat_template.cpp role_name: templates
    know system, and the official Qwen templates raise on any other role), and a null content is
    the empty string (the request parser keeps content as text)."""
    out = copy.deepcopy(messages)
    for message in out:
        if message["role"] == "developer":
            message["role"] = "system"
        if message.get("content") is None:
            message["content"] = ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    return out


def first_match(conventions, category, template):
    """The convention entry NInferno resolves a template to (src/chat/conventions.cpp first_match)."""
    for entry in conventions[category]:
        if all(marker in template for marker in entry["detect"]):
            return entry
    return None


def ids_of(result):
    if isinstance(result, dict) or hasattr(result, "keys"):
        result = result["input_ids"]
    if result and isinstance(result[0], list):
        result = result[0]
    return [int(x) for x in result]


def build(tag, repo, revision, family, why, out, conventions):
    d = out / tag
    if d.exists():
        shutil.rmtree(d)
    fetch(repo, revision, d)
    derive_resources(d)
    auto = AutoTokenizer.from_pretrained(d)
    raw = Tokenizer.from_file(str(d / "tokenizer.json"))
    template = (d / "chat_template.jinja").read_text(encoding="utf-8")
    # transformers renders the template the tokenizer holds; the converter's derivation must
    # agree with it (a list of named templates is where they could differ).
    auto.chat_template = template
    reads_thinking = "enable_thinking" in template
    reads_effort = "reasoning_effort" in template
    # NInferno policy: tools for a template that declares no tool-call syntax are refused (the
    # server names --chat-template), where transformers renders the template, which drops them.
    takes_tools = first_match(conventions, "tools", template) is not None
    cases = []
    counts = {"cases": 0, "refused_by_template": 0, "tokenizers_differ": 0}

    def add(case_id, request, messages, add_generation_prompt, continue_final, kwargs, tools):
        counts["cases"] += 1
        try:
            text = auto.apply_chat_template(
                hf_messages(messages), tools=tools, add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final, tokenize=False, **kwargs)
            ids = ids_of(auto.apply_chat_template(
                hf_messages(messages), tools=tools, add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final, tokenize=True, return_dict=False, **kwargs))
        except Exception as error:  # the template's own refusal (raise_exception)
            counts["refused_by_template"] += 1
            cases.append({"id": case_id, "request": request, "refused": str(error), "ids": []})
            return
        case = {"id": case_id, "request": request, "text": text, "ids": ids}
        direct = raw.encode(text, add_special_tokens=False).ids
        if direct != ids:
            counts["tokenizers_differ"] += 1
            case["tokenizers_ids"] = direct
        cases.append(case)

    for (dialog, messages), tools, thinking, effort in itertools.product(
            DIALOGS.items(), (None, TOOLS), (None, True, False) if reads_thinking else (None,),
            (None, "low") if reads_effort else (None,)):
        request = {"model": "m", "messages": messages, "max_tokens": 16}
        kwargs = {}
        if tools:
            request["tools"] = tools
        if thinking is not None:
            request["chat_template_kwargs"] = {"enable_thinking": thinking}
            kwargs["enable_thinking"] = thinking
        if effort is not None:
            request["reasoning_effort"] = effort
            kwargs["reasoning_effort"] = effort
        add(f"{dialog} tools={'2' if tools else '0'} thinking={thinking} effort={effort}",
            request, messages, True, False, kwargs, tools)
        if tools and not takes_tools:
            cases[-1]["expect_ninferno_refusal"] = "declares no tool-call syntax"
        elif thinking is False and effort is not None:
            # NInferno policy: a reasoning effort with thinking disabled is a conflicting request
            # (400), where transformers renders whichever the template reads.
            cases[-1]["expect_ninferno_refusal"] = "reasoning effort conflicts with enable_thinking"

    # Assistant prefill: NInferno continues a final assistant message through the Anthropic Messages
    # dialect; the reference is continue_final_message=True.
    messages = [{"role": "user", "content": "Finish the sentence."},
                {"role": "assistant", "content": "The answer is"}]
    request = {"model": "m", "messages": messages, "max_tokens": 16}
    kwargs = {}
    if reads_thinking:
        request["thinking"] = {"type": "disabled"}
        kwargs["enable_thinking"] = False
    add("continuation", request, messages, False, True, kwargs, None)
    cases[-1]["dialect"] = "anthropic"
    if first_match(conventions, "turns", template) is None:
        # NInferno continues a final assistant message only where it can locate that message's
        # content in the rendered prompt: through the template's turn markup (conventions.json).
        cases[-1]["expect_ninferno_refusal"] = "does not expose the final assistant content"

    with open(d / "cases.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    (d / "frontend.json").write_text(json.dumps(
        {"family": family, "repo": repo, "revision": revision, "why": why}), encoding="utf-8")
    print(f"{tag}: {repo}@{(revision or 'main')[:8]} family {family} ({why}): {counts}")


def main():
    out = Path(sys.argv[1])
    wanted = set(sys.argv[2:])
    out.mkdir(parents=True, exist_ok=True)
    conventions = json.loads((ROOT / "src/chat/data/conventions.json").read_text(encoding="utf-8"))
    (out / "required.json").write_text(json.dumps({
        "turns": [entry["name"] for entry in conventions["turns"]],
        "reasoning": [entry["name"] for entry in conventions["reasoning"]],
        "tools": [entry["format"] for entry in conventions["tools"]],
    }), encoding="utf-8")
    for tag, repo, revision, family, why in MODELS:
        if wanted and tag not in wanted:
            continue
        build(tag, repo, revision, family, why, out, conventions)


if __name__ == "__main__":
    main()
