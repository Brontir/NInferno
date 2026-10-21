r"""Writes tokenizer_split_variants_fixture.h for ninferno_qwen3_5_tokenizer_split_variants_test.

    python tests/models/qwen3_5/hf_tokenizer_split_variants.py QWEN35_DIR QWEN2_SPLIT_DIR OUT_HEADER

QWEN35_DIR holds Qwen/Qwen3.8-27B's tokenizer.json (Split regex with \p{M} in the letter runs);
QWEN2_SPLIT_DIR a re-save of the same vocabulary whose Split is the Qwen2 regex (marks outside
letter runs), as huginnfork/Qwen3.8-27B-NVFP4A16 ships it. Both must share vocabulary and merges.

For every text of TEXTS the reference ids are each tokenizer.json run by the `tokenizers`
library. The header embeds only what those encodings touch: the 256 byte symbols, the merges BPE
applies (in rank order) and their results, with the checkpoint's ids. BPE with a subset of the
merges that contains every merge applied picks the same merge at every step, so the reduced
vocabulary gives the same ids; the script checks that with `tokenizers` too.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer
import tokenizers

TEXTS = [
    "Hello world",
    "It's 'S they'RE we've I'm you'll he'd 'LL",
    "abc123 4567 x9",
    "Hello, world!!! (ok)... --- ;;",
    "a  b   c \t d\n\ne\r\n f  \n",
    "   ",
    "end  ",
    "\n\n\n",
    "नमस्ते दुनिया",
    "สวัสดีชาวโลก",
    "é",
    "á̂b",
    "x́",
    " ́abc",
    "́",
    "!́?",
    "Tiếng Việt có dấu: Xin chào thế giới",
    "Ngự̀i",
    "emoji \U0001F600\U0001F44D\U0001F3FD❤️ ok",
    "क्षत्रिय 12 मार्ग।",
    "ไทย123ไทย",
    "क्\nกิ กิ",
    "café",
    "ẓ̈x",
    "́́\n",
    "She said: \"नमस्ते!\" 'll",
    "<|im_start|>user\nनमस्ते<|im_end|>\n",
]

ADDED = [(248044, "<|endoftext|>"), (248045, "<|im_start|>"), (248046, "<|im_end|>")]


def bpe(word: str, ranks: dict[tuple[str, str], int], used: set[tuple[str, str]]) -> list[str]:
    symbols = list(word)
    while len(symbols) > 1:
        best = None
        for i in range(len(symbols) - 1):
            rank = ranks.get((symbols[i], symbols[i + 1]))
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, i)
        if best is None:
            break
        i = best[1]
        used.add((symbols[i], symbols[i + 1]))
        symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
    return symbols


def c_string(text: str) -> str:
    out = []
    for byte in text.encode("utf-8"):
        ch = chr(byte)
        if ch == "\\" or ch == '"':
            out.append("\\" + ch)
        elif 0x20 <= byte < 0x7F:
            out.append(ch)
        else:
            out.append("\\%03o" % byte)
    return '"' + "".join(out) + '"'


def split_regex(root: dict) -> str:
    return root["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]


def main() -> int:
    new_dir, old_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    new_root = json.loads((new_dir / "tokenizer.json").read_text(encoding="utf-8"))
    old_root = json.loads((old_dir / "tokenizer.json").read_text(encoding="utf-8"))
    pair = lambda m: tuple(m) if isinstance(m, list) else tuple(m.split(" "))
    merges = [pair(m) for m in new_root["model"]["merges"]]
    assert merges == [pair(m) for m in old_root["model"]["merges"]], "merges differ"
    assert new_root["model"]["vocab"] == old_root["model"]["vocab"], "vocabularies differ"
    vocab = new_root["model"]["vocab"]
    ranks = {m: r for r, m in enumerate(merges)}
    byte_symbols = [s for s, _ in sorted(vocab.items(), key=lambda kv: kv[1]) if len(s) == 1]
    added = {content: id for id, content in ADDED}

    used: set[tuple[str, str]] = set()
    expected = {}
    for name, d in (("qwen35", new_dir), ("qwen2", old_dir)):
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        expected[name] = []
        for text in TEXTS:
            ref = tok.encode(text, add_special_tokens=False).ids
            # The same ids by hand: added tokens, then NFC + Split + ByteLevel + BPE.
            mine = []
            rest = text
            while rest:
                cut = min((rest.find(c), c) for c in added if rest.find(c) >= 0) if any(
                    c in rest for c in added) else (len(rest), None)
                piece, rest = rest[:cut[0]], rest[cut[0]:]
                if piece:
                    normalized = unicodedata.normalize("NFC", piece)
                    for word, _ in tok.pre_tokenizer.pre_tokenize_str(normalized):
                        mine += [vocab[s] for s in bpe(word, ranks, used)]
                if cut[1] is not None:
                    mine.append(added[cut[1]])
                    rest = rest[len(cut[1]):]
            assert mine == ref, (name, text, mine, ref)
            expected[name].append(ref)

    kept = sorted(used, key=lambda m: ranks[m])
    symbols = set(byte_symbols) | {a + b for a, b in kept}
    small_vocab = {s: vocab[s] for s in symbols}
    # The reduced vocabulary gives the same ids under both regexes.
    for name, root in (("qwen35", new_root), ("qwen2", old_root)):
        small = dict(root)
        small["model"] = dict(root["model"], vocab=small_vocab, merges=[list(m) for m in kept])
        small["added_tokens"] = [t for t in root["added_tokens"] if t["content"] in added]
        small["truncation"] = None
        tok = Tokenizer.from_str(json.dumps(small))
        got = [tok.encode(t, add_special_tokens=False).ids for t in TEXTS]
        for i, text in enumerate(TEXTS):
            # (tokenizers maps added tokens of a sparse vocabulary to other ids; the hand encoding
            # above checked those texts against the full files.)
            if any(c in text for c in added):
                continue
            assert got[i] == expected[name][i], (name, text, got[i], expected[name][i])

    decoder = Tokenizer.from_file(str(new_dir / "tokenizer.json"))
    lines = [
        "// Generated by tests/models/qwen3_5/hf_tokenizer_split_variants.py (tokenizers "
        + tokenizers.__version__ + ") from the",
        "// tokenizer.json of Qwen/Qwen3.8-27B (Qwen3.5 Split regex) and of",
        "// huginnfork/Qwen3.8-27B-NVFP4A16 (the Qwen2 Split regex, same vocabulary): the merges the",
        "// texts below use, their symbols with the checkpoint's ids, and each file's ids. Do not edit.",
        "#pragma once",
        "",
        "#include <string_view>",
        "#include <utility>",
        "#include <vector>",
        "",
        "namespace split_variants_fixture {",
        "",
        "inline constexpr std::string_view kQwen35Split = " + c_string(split_regex(new_root)) + ";",
        "inline constexpr std::string_view kQwen2Split  = " + c_string(split_regex(old_root)) + ";",
        "",
        "struct Symbol {",
        "    std::string_view text;",
        "    int id;",
        "};",
        "",
        "inline const std::vector<Symbol> kVocab = {",
    ]
    for s in sorted(symbols, key=lambda s: vocab[s]):
        lines.append("    {" + c_string(s) + ", " + str(vocab[s]) + "},")
    lines += ["};", "", "inline const std::vector<std::pair<std::string_view, std::string_view>> kMerges = {"]
    for a, b in kept:
        lines.append("    {" + c_string(a) + ", " + c_string(b) + "},")
    lines += ["};", "", "inline const std::vector<Symbol> kAdded = {"]
    for id, content in ADDED:
        lines.append("    {" + c_string(content) + ", " + str(id) + "},")
    lines += ["};", "", "struct Case {", "    std::string_view text;", "    std::vector<int> qwen35;",
              "    std::vector<int> qwen2;", "    std::string_view decoded;", "};", "",
              "inline const std::vector<Case> kCases = {"]
    for i, text in enumerate(TEXTS):
        a, b = expected["qwen35"][i], expected["qwen2"][i]
        dec = decoder.decode(a, skip_special_tokens=False)
        assert dec == decoder.decode(b, skip_special_tokens=False)
        lines.append("    {" + c_string(text) + ",")
        lines.append("     {" + ", ".join(map(str, a)) + "},")
        lines.append("     {" + ", ".join(map(str, b)) + "},")
        lines.append("     " + c_string(dec) + "},")
    lines += ["};", "", "} // namespace split_variants_fixture", ""]
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    differ = sum(expected["qwen35"][i] != expected["qwen2"][i] for i in range(len(TEXTS)))
    print(json.dumps({"texts": len(TEXTS), "differ": differ, "merges": len(kept),
                      "symbols": len(symbols), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
