#!/usr/bin/env python3
"""Reference fixtures for ninferno::text::Tokenizer (tests/text/test_tokenizer.cpp).

Every model directory under --root holds tokenizer.json (plus tokenizer_config.json and chat
template files when the repository has them). For each directory this writes fixture.json:
texts with the ids and decoded strings produced by the Hugging Face `tokenizers` library
(Tokenizer.from_file), which the C++ tokenizer must reproduce exactly.

    python tokenizer_fixtures.py --root /path/to/fixtures \
        --download granite=ibm-granite/granite-4.2-8b --synthetic

--download NAME=REPO fetches the tokenizer files of REPO into ROOT/NAME first (public repos;
HF_TOKEN is used when set). --synthetic also derives tokenizer.json variants that exercise
pipeline options no downloaded model uses (Digits, Punctuation, Metaspace always/split,
NFKC/NFD/NFKD, Lowercase, Replace regex, Split behaviours, added-token lstrip/rstrip/
single_word/normalized); it needs the mistral7b and granite directories.
"""

import argparse
import copy
import json
import os
import random
import sys
import urllib.error
import urllib.request

from tokenizers import Tokenizer

CORPORA = ("wikitext", "pg19", "zhwiki", "ninferno")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json",
                   "chat_template.jinja", "chat_template.json", "special_tokens_map.json")

HARD_CASES = [
    ("empty", ""),
    ("space", " "),
    ("contractions", "I'M sure DON'T you'd they'LL WE'VE it's He'S she'd 'Re"),
    ("contractions_fold", "IT'ſ it'ſt O'ﬀ 'K 'KELVIN"),
    ("digits", "1234567 12 123 1234 3.14159 1,000,000 ١٢٣٤ ①②③ 2024-09-25"),
    ("emoji", "Hi 👋🏽! Family: 👨‍👩‍👧‍👦, flag 🇺🇦🇯🇵, heart ❤️, keycap 1️⃣ #️⃣ 🏳️‍🌈"),
    ("combining", "e\u0301 cafe\u0301 Å A\u030a n\u0303 \u1100\u1161\u11a8 x\u0336y\u0336 Z\u0351\u0327"),
    ("cjk", "日本語のテキスト、中文文本。한국어 텍스트입니다. 漢字かな交じり文"),
    ("arabic", "مرحبا بالعالم، هذا نص عربي. שלום עולם"),
    ("devanagari", "नमस्ते दुनिया, यह एक परीक्षण है। ภาษาไทย"),
    ("cyrillic", "Привет, мир! ЁЖИК ёжик — Україна, Ґанок"),
    ("ws_leading", "   leading spaces"),
    ("ws_trailing", "trailing spaces   "),
    ("ws_newlines", "a\n\nb\n\n\nc\n"),
    ("ws_crlf", "line one\r\nline two\r\n\r\nend"),
    ("ws_tabs", "\tindent\t\ttab\t \t mixed"),
    ("ws_multi_before_word", "word      word2     \n   x"),
    ("ws_only", " \n \t  \r\n  "),
    ("ws_unicode", "a\u00a0b\u2003c\u3000d\u2028e\u0085f\u200bg\u180eh"),
    ("code", "def f(x):\n    return x**2  # square\n\nif __name__ == '__main__':\n\tprint(f(3))\n"),
    ("json", '{"key": [1, 2.5, null, true], "nested": {"a": "b\\n"}}'),
    ("urls", "https://example.com/path?q=1&r=%20x#frag user@mail.org C:\\Windows\\System32"),
    ("punct", "Hello!!! What?! ... --- ((())) [[x]] <<>> ~~~ ``` '' \"\" «» „“ — – …"),
    ("math", "∑_{i=1}^{n} x_i² ≤ ∫₀^∞ e^{-x} dx ≈ 1.0 ± 0.1 → ∞ ≠ √2"),
    ("mixed_scripts", "Tokyo東京Москва123abcDEF_ghi-jkl.mno"),
    ("long_word", "Pneumonoultramicroscopicsilicovolcanoconiosis" * 3),
    ("long_digits", "31415926535897932384626433832795028841971693993751"),
    ("long_spaces", "a" + " " * 64 + "b" + "\n" * 20 + "c"),
    ("replacement_char", "bad \ufffd char and \U0010fffd private"),
    ("bmp_edges", "\u0000\u0001\u001f\u007f\u0080\u00ff\uffff"),
]

SPECIAL_STRINGS = ["<|endoftext|>", "<s>", "</s>", "[INST]", "[/INST]", "<start_of_turn>",
                   "<end_of_turn>", "<|im_start|>", "<|im_end|>", "<unk>", "<pad>",
                   "<|begin_of_text|>", "<|eot_id|>", "<think>", "</think>", "<bos>", "<eos>"]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
    {"role": "user", "content": "Hi! Can you translate 'good morning' into Japanese and Ukrainian?"},
    {"role": "assistant", "content": "Sure: おはようございます (ohayō gozaimasu), and «Доброго ранку»."},
    {"role": "user", "content": "Thanks 🙏  Now count:\n1 2 3\n\tdone"},
]


def download(root, name, repo):
    target = os.path.join(root, name)
    os.makedirs(target, exist_ok=True)
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    for filename in TOKENIZER_FILES:
        path = os.path.join(target, filename)
        if os.path.exists(path):
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r:
                data = r.read()
        except urllib.error.HTTPError as ex:
            if filename == "tokenizer.json":
                raise SystemExit(f"{repo}: cannot fetch tokenizer.json ({ex.code})")
            continue
        with open(path, "wb") as f:
            f.write(data)
    with open(os.path.join(target, "REPO"), "w", encoding="utf-8", newline="\n") as f:
        f.write(repo + "\n")


def random_texts(seed, count):
    """Printable strings over mixed scripts, emoji, marks and whitespace (valid UTF-8 only)."""
    rng = random.Random(seed)
    pools = [
        [chr(c) for c in range(0x20, 0x7F)],
        [chr(c) for c in range(0xA1, 0x250)],
        [chr(c) for c in range(0x400, 0x4FF)],
        [chr(c) for c in range(0x300, 0x36F)],
        [chr(c) for c in range(0x4E00, 0x4E00 + 400)],
        [chr(c) for c in range(0x3040, 0x30FF)],
        [chr(c) for c in range(0x1F300, 0x1F650)],
        [chr(c) for c in range(0x600, 0x6FF)],
        [" ", " ", "  ", "\n", "\t", "\r\n", "\u00a0", "\u200d"],
        list("0123456789"),
        ["'s", "'t", "'re", "'VE", "'M", "'ll", "'D"],
    ]
    texts = []
    for i in range(count):
        length = rng.choice([1, 2, 5, 20, 80, 300])
        pieces = []
        for _ in range(length):
            pool = rng.choice(pools)
            pieces.append(rng.choice(pool))
        texts.append((f"random_{i}", "".join(pieces)))
    return texts


def special_texts(tokenizer):
    decoder = tokenizer.get_added_tokens_decoder()
    added = [decoder[i].content for i in sorted(decoder)]
    rng = random.Random(7)
    picks = sorted(set(rng.sample(added, min(len(added), 12)) + [a for a in added if len(a) < 16][:8]
                       + added[-5:]))
    texts = [("special_strings", " ".join(SPECIAL_STRINGS) + " " + "".join(SPECIAL_STRINGS))]
    for i, token in enumerate(picks):
        texts.append((f"added_{i}", f"Hello{token}world {token} x {token}{token}\n{token} end "))
    texts.append(("added_all", "".join(added[:200])))
    return texts


def chat_text(model_dir):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as ex:  # noqa: BLE001 - any loader failure only skips this case
        print(f"  chat template: AutoTokenizer failed: {ex}", file=sys.stderr)
        return None
    if not getattr(tok, "chat_template", None):
        return None
    for messages in (MESSAGES, MESSAGES[1:]):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as ex:  # noqa: BLE001 - templates may reject a system turn
            print(f"  chat template: {ex}", file=sys.stderr)
    return None


def build_fixture(model_dir, corpus_dir):
    tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    # ninferno::text::Tokenizer reads truncation and padding past (as transformers does by default);
    # the reference must not cut or pad either (Gemma 4 sets truncation to 8192 tokens).
    tokenizer.no_truncation()
    tokenizer.no_padding()
    texts = []
    for name in CORPORA:
        path = os.path.join(corpus_dir, name, "00.txt")
        with open(path, encoding="utf-8") as f:
            texts.append((f"corpus:{name}", f.read()))
    texts += HARD_CASES
    texts += special_texts(tokenizer)
    texts += random_texts(1234, 24)
    chat = chat_text(model_dir)
    if chat is not None:
        texts.append(("chat_template", chat))
    else:
        print(f"  {model_dir}: no chat template case", file=sys.stderr)

    cases = []
    for name, text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        ids_special = tokenizer.encode(text, add_special_tokens=True).ids
        cases.append({
            "name": name,
            "text": text,
            "ids": ids,
            "ids_special": ids_special,
            "decoded": tokenizer.decode(ids_special, skip_special_tokens=False),
            "decoded_skip": tokenizer.decode(ids_special, skip_special_tokens=True),
            "decoded_plain": tokenizer.decode(ids, skip_special_tokens=False),
        })

    # Decode-only cases: random id sequences reach byte-fallback and lossy UTF-8 paths.
    rng = random.Random(99)
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    added_ids = sorted(tokenizer.get_added_tokens_decoder().keys())
    decode_cases = []
    for i in range(24):
        ids = [rng.randrange(size + 8) for _ in range(rng.choice([1, 3, 16, 64]))]
        if added_ids and i % 3 == 0:
            ids.insert(rng.randrange(len(ids) + 1), rng.choice(added_ids))
        decode_cases.append({
            "name": f"random_ids_{i}",
            "ids": ids,
            "decoded": tokenizer.decode(ids, skip_special_tokens=False),
            "decoded_skip": tokenizer.decode(ids, skip_special_tokens=True),
        })
    byte_tokens = [tokenizer.token_to_id(f"<0x{b:02X}>") for b in range(256)]
    if all(t is not None for t in byte_tokens):
        for i, seq in enumerate([[0xE2, 0x82, 0xAC], [0xE2, 0x82], [0xFF, 0x41, 0xC3], [0xF0, 0x9F, 0x98, 0x80, 0x80]]):
            ids = [byte_tokens[b] for b in seq]
            decode_cases.append({
                "name": f"byte_fallback_{i}",
                "ids": ids,
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
                "decoded_skip": tokenizer.decode(ids, skip_special_tokens=True),
            })

    repo_file = os.path.join(model_dir, "REPO")
    if os.path.exists(repo_file):
        with open(repo_file, encoding="utf-8") as f:
            repo = f.read().strip()
    else:
        repo = os.path.basename(model_dir)
    fixture = {"repo": repo, "cases": cases, "decode_cases": decode_cases}
    with open(os.path.join(model_dir, "fixture.json"), "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False)
    print(f"{os.path.basename(model_dir)}: {len(cases)} texts, {len(decode_cases)} decode cases")


def write_variant(root, name, base, mutate, repo_note):
    with open(os.path.join(root, base, "tokenizer.json"), encoding="utf-8") as f:
        data = json.load(f)
    data = mutate(copy.deepcopy(data))
    target = os.path.join(root, name)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    with open(os.path.join(target, "REPO"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"synthetic: {repo_note}\n")
    Tokenizer.from_file(os.path.join(target, "tokenizer.json"))  # must load in the library


def synthetic_variants(root):
    def metaspace_variant(data):
        data["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "Lowercase"}]}
        data["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {"type": "Punctuation", "behavior": "Isolated"},
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
        ]}
        data["decoder"] = {"type": "Sequence", "decoders": [
            {"type": "ByteFallback"},
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
            {"type": "Fuse"},
        ]}
        next_id = len(data["model"]["vocab"])
        for content, flags in [("<lstrip>", {"lstrip": True}), ("<rstrip>", {"rstrip": True}),
                               ("<both>", {"lstrip": True, "rstrip": True}),
                               ("Wordy", {"single_word": True, "special": False}),
                               ("<NORM>", {"normalized": True, "special": False})]:
            token = {"id": next_id, "content": content, "single_word": False, "lstrip": False,
                     "rstrip": False, "normalized": False, "special": True}
            token.update(flags)
            data["added_tokens"].append(token)
            next_id += 1
        return data

    def split_variant_a(data):
        data["normalizer"] = {"type": "Sequence", "normalizers": [
            {"type": "NFD"}, {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]}
        data["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "[,.;:]"}, "behavior": "MergedWithPrevious", "invert": False},
            {"type": "Split", "pattern": {"String": "the"}, "behavior": "MergedWithNext", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        ]}
        return data

    def split_variant_b(data):
        data["normalizer"] = {"type": "NFKD"}
        data["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Removed", "invert": False},
            {"type": "Split", "pattern": {"Regex": "[aeiouAEIOU]+"}, "behavior": "Contiguous", "invert": True},
            {"type": "Digits", "individual_digits": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ]}
        first = min((t for t in data["added_tokens"] if t["special"]), key=lambda t: t["id"])
        marker = first["content"]
        data["post_processor"] = {"type": "TemplateProcessing",
                                  "single": [{"SpecialToken": {"id": marker, "type_id": 0}},
                                             {"Sequence": {"id": "A", "type_id": 0}},
                                             {"SpecialToken": {"id": marker, "type_id": 0}}],
                                  "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                                  "special_tokens": {marker: {"id": marker, "ids": [first["id"]], "tokens": [marker]}}}
        return data

    write_variant(root, "synth_metaspace", "mistral7b", metaspace_variant,
                  "mistral7b + NFKC/Lowercase, Digits, Punctuation, Metaspace always/split, added-token flags")
    write_variant(root, "synth_split_a", "granite", split_variant_a,
                  "granite + NFD, Replace regex, Split MergedWithPrevious/MergedWithNext, ByteLevel add_prefix_space")
    write_variant(root, "synth_split_b", "granite", split_variant_b,
                  "granite + NFKD, Split Removed/Contiguous invert, Digits contiguous, template with two specials")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory of model directories")
    parser.add_argument("--corpus-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                             "eval", "corpora", "perplexity-1m", "data"),
                        help="the perplexity corpus (default: the repository's eval/corpora)")
    parser.add_argument("--download", action="append", default=[], metavar="NAME=REPO")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="limit to these model directories")
    args = parser.parse_args()

    for spec in args.download:
        name, repo = spec.split("=", 1)
        download(args.root, name, repo)
    if args.synthetic:
        synthetic_variants(args.root)
    for name in sorted(os.listdir(args.root)):
        model_dir = os.path.join(args.root, name)
        if args.only and name not in args.only:
            continue
        if os.path.isfile(os.path.join(model_dir, "tokenizer.json")):
            build_fixture(model_dir, args.corpus_dir)


if __name__ == "__main__":
    main()
