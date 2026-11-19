#pragma once

// A tiny Hugging Face tokenizer.json for tests that need a real HF codec without a model's file: a
// BPE whose "a" + "b" merge into "ab", one special added token "<T>" (id 9), and byte-fallback
// decoding. The chat codec test and the Qwen3.5 frontend's shared-tokenizer test read this one copy
// (#1021), so the two cannot drift apart.

#include <string_view>

inline constexpr std::string_view kTinyHfTokenizer = R"({
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, " ": 3, "<": 4, "T": 5, ">": 6,
            "<0x41>": 7, "<0xE2>": 8}, "merges": ["a b"], "byte_fallback": true},
  "added_tokens": [{"id": 9, "content": "<T>", "special": true}],
  "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]}
})";
