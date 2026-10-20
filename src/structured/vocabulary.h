#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ninferno::structured {

// Model-independent description of the output token domain as the output presentation sees it.
// A target Frontend builds it once from its tokenizer; the constraint compiler derives its grammar
// vocabularies from it.
struct GrammarVocabulary {
    // Exact byte string of every token id in [0, token_domain). Ids the tokenizer does not define
    // are empty.
    std::vector<std::string> token_bytes;
    // True for control tokens whose bytes the output presentation drops unless special tokens are
    // preserved (OutputOptions::preserve_special_tokens or raw output).
    std::vector<bool> special;
    // Model-default end-of-output token ids. A constraint admits them only where its grammar can
    // end, and a completed grammar admits nothing else.
    std::vector<std::int32_t> stop_token_ids;
    // Byte sequence that closes a reasoning section in the output stream ("</think>" for Qwen).
    // Empty when the model has no reasoning section.
    std::string reasoning_close;
    // Byte sequence with which the model opens a reasoning section itself ("<think>",
    // "<|channel>thought", "<|channel|>analysis<|message|>"). Empty when it has none.
    std::string reasoning_open;
    // Sampled token domain; token_bytes and special have exactly this many entries.
    std::uint32_t token_domain = 0;
};

// Number of 32-bit words of one allowed-token bitmask row, padded to 32 bytes.
[[nodiscard]] constexpr std::uint32_t mask_words_for(std::uint32_t token_domain) noexcept {
    return (((token_domain + 31U) / 32U) + 7U) & ~7U;
}

} // namespace ninferno::structured
