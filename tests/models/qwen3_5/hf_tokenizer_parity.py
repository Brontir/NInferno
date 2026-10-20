r"""The transformers side of ninferno_qwen3_5_tokenizer_parity_test.

    python tests/models/qwen3_5/hf_tokenizer_parity.py TOKENIZER_DIR OUT_DIR

TOKENIZER_DIR holds a checkpoint's tokenizer files (tokenizer.json, tokenizer_config.json,
generation_config.json, chat_template.jinja; a transformers-4 or a transformers-5 layout). OUT_DIR
receives copies of the first three and expected.jsonl: for every text of the corpus below, the
reference ids and the text transformers decodes them to (special tokens kept, no clean-up).

The reference ids are the checkpoint's own tokenizer.json run by the `tokenizers` library, inside
the special tokens transformers adds (add_special_tokens as the checkpoint declares). transformers
5.x's Qwen2Tokenizer rebuilds the pre-tokenizer from its class and drops the combining-mark (\p{M})
classes that Qwen3.5 checkpoints declare, so it splits Indic and Thai words differently from the
shipped definition. Texts where transformers deviates keep its ids as `transformers_ids`, and the
summary counts them.

The corpus: every paragraph of the repository's Markdown files, the lines of a sample of its
C++ and Python sources, the built-in speculation prompts raw and rendered through the chat
template (thinking on and off, so control tokens appear inside text), multilingual sentences, and
whitespace, Unicode-normalisation and special-token edge cases.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

EDGE_CASES = [
    "", " ", "  ", "\n", "\n\n\n", "\t\tindented\n", "trailing spaces   ", "   leading",
    "a\r\nb\r\n", "tabs\tand\tspaces  mixed \t\n", "x" * 3000, "ab" * 1500,
    "1234567890 3.14159 -2.5e-10 0x1F 1,000,000", "https://example.com/a/b?c=d&e=f#g",
    "emoji: \U0001F600\U0001F680\U0001F1FA\U0001F1E6 ❤️", "café café",
    "Å Å Å", "ﬁ ＡＢ", "zero​width‍joiner",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "<|endoftext|><|im_end|>",
    "<think>\nreasoning\n</think>\n\nanswer", "<|vision_start|><|image_pad|><|vision_end|>",
    "<|im_start|", "<|im_start|><|im_start|>", "text<|im_end|>more text<tool_call>",
    "don't won't it's we're they've I'm you'll he'd", "DON'T WON'T IT'S",
]

MULTILINGUAL = [
    "今天天气很好，我们去公园散步吧。",
    "機械学習のモデルを高速化するには、メモリ帯域が重要です。",
    "한국어 문장도 제대로 토큰화되어야 합니다.",
    "Привет, как дела? Сегодня мы тестируем токенизатор.",
    "Українська мова має літери ї, є та ґ.",
    "مرحبا بالعالم، هذا اختبار للنص العربي.",
    "שלום עולם, זהו מבחן.",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "สวัสดีครับ นี่คือการทดสอบ",
    "Γειά σου κόσμε, αυτό είναι ένα τεστ.",
    "Tiếng Việt có nhiều dấu thanh: à á ả ã ạ.",
    "Français : l'été dernier, nous avons visité Montréal et Québec.",
    "Deutsch: Größenordnung, Straße, Übermäßig, Äpfel und Öl.",
    "Español: ¿Dónde está la biblioteca? ¡Qué día tan bonito!",
    "Português: ação, coração, pão e informação.",
    "Polski: zażółć gęślą jaźń.",
    "Türkçe: İstanbul'da ılık bir öğleden sonra.",
]


def paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [p for p in text.split("\n\n") if p.strip()]


def corpus(tok) -> list[str]:
    texts = list(EDGE_CASES) + list(MULTILINGUAL)
    for md in sorted(ROOT.glob("**/*.md")):
        if "third_party" in md.parts or "build" in md.parts:
            continue
        texts += paragraphs(md)
    sources = sorted(p for p in (ROOT / "src").rglob("*") if p.suffix in (".cpp", ".h", ".cu"))
    sources += sorted((ROOT / "tools").rglob("*.py"))
    for path in sources[::7]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        texts += ["".join(lines[i:i + 20]) for i in range(0, len(lines), 20)]
    spec = json.loads((ROOT / "src/spec/data/spec_corpus.json").read_text(encoding="utf-8"))
    for prompt in spec["prompts"]:
        texts.append(prompt["text"])
        for thinking in (False, True):
            texts.append(tok.apply_chat_template([{"role": "user", "content": prompt["text"]}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=thinking))
    return texts


def main() -> int:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(source))
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copyfile(source / name, out / name)
    from tokenizers import Tokenizer

    shipped = Tokenizer.from_file(str(source / "tokenizer.json"))
    texts = corpus(tok)
    tokens = 0
    deviations = 0
    with (out / "expected.jsonl").open("w", encoding="utf-8") as stream:
        for text in texts:
            with_specials = tok(text).input_ids
            plain = tok(text, add_special_tokens=False).input_ids
            start = next(i for i in range(len(with_specials) + 1)
                         if with_specials[i:i + len(plain)] == plain)
            reference = shipped.encode(text, add_special_tokens=False).ids
            ids = with_specials[:start] + reference + with_specials[start + len(plain):]
            record = {"text": text, "ids": ids}
            if reference != plain:
                deviations += 1
                record["transformers_ids"] = with_specials
            record["decoded"] = tok.decode(ids, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
            tokens += len(ids)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"transformers": transformers.__version__, "tokenizer": type(tok).__name__,
                      "texts": len(texts), "tokens": tokens,
                      "transformers_deviations_from_tokenizer_json": deviations,
                      "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
