"""A tokenizer's vocabulary fingerprint: the same value as ninferno::spec::vocab_fingerprint
(src/spec/fingerprint.h), which keys the speculation corpus and checks a drafter against its target."""

from __future__ import annotations

import hashlib


def vocab_fingerprint(tokenizer: dict) -> tuple[str, int]:
    """The fingerprint of a parsed tokenizer.json and the number of tokens it covers."""
    vocab = tokenizer["model"]["vocab"]
    text: dict[int, str] = {}
    if isinstance(vocab, dict):
        for token, index in vocab.items():
            if isinstance(index, int) and index >= 0:
                text[index] = token
    else:
        for index, entry in enumerate(vocab):
            if isinstance(entry, list) and entry and isinstance(entry[0], str):
                text[index] = entry[0]
    for entry in tokenizer.get("added_tokens") or []:
        if isinstance(entry.get("id"), int) and isinstance(entry.get("content"), str):
            text[entry["id"]] = entry["content"]
    lines = "".join(f"{i}\t{text[i]}\n" for i in sorted(text))
    return "v1:" + hashlib.sha256(lines.encode("utf-8")).hexdigest()[:16], len(text)
