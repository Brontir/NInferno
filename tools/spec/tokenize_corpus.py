"""Add a model family's rendered prompt ids to the built-in speculation corpus.

    python tools/spec/tokenize_corpus.py --tokenizer CHECKPOINT_DIR [--corpus PATH]
        [--stops 1,106,50] [--label TEXT] [--check]

Engines that carry no tokenizer of their own (apps/gemma4) read the corpus as token ids, keyed by
the vocabulary fingerprint of the model (src/spec/fingerprint.h). This renders every prompt with
the checkpoint's chat template (thinking on or off as the prompt says), tokenizes it, and stores
the ids under the tokenizer's fingerprint together with the stop ids of the family. --check only
prints the fingerprint. Needs transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from vocab_fingerprint import vocab_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src/spec/data/spec_corpus.json")
    parser.add_argument("--stops", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    fingerprint, count = vocab_fingerprint(
        json.loads((args.tokenizer / "tokenizer.json").read_text(encoding="utf-8")))
    print(f"{fingerprint} ({count} tokens)")
    if args.check:
        return 0

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    prompts = {}
    for prompt in corpus["prompts"]:
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt["text"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=prompt["thinking"])
        ids = tok(rendered, add_special_tokens=False).input_ids
        # A model with a BOS token starts every sequence with it (as its own generate() does).
        if tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
            ids = [tok.bos_token_id] + ids
        prompts[prompt["id"]] = ids
    stops = [int(s) for s in args.stops.split(",") if s]
    corpus.setdefault("tokens", {})[fingerprint] = {
        "label": args.label or args.tokenizer.name,
        "stops": stops,
        "prompts": prompts,
    }
    # Integer lists on one line keep the file small and reviewable.
    text = json.dumps(corpus, indent=2, ensure_ascii=False)
    # (Only the arrays json.dumps spread over lines: never a bracket inside a prompt's text.)
    text = re.sub(r"\[\n\s*(-?\d+(?:,\n\s*-?\d+)*)\n\s*\]",
                  lambda m: "[" + ",".join(v.strip() for v in m.group(1).split(",")) + "]", text)
    args.corpus.write_text(text + "\n", encoding="utf-8")
    lengths = [len(v) for v in prompts.values()]
    print(f"wrote {len(prompts)} prompts ({min(lengths)}..{max(lengths)} ids) to {args.corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
