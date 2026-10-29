#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

namespace ninferno::runtime {

// The whole-model calls a prefill runs after a prompt's own chunks: the generation prompt's
// execution boundaries (the assistant turn's begin and its body's, models/qwen3_5/frontend
// prompt_layout.cpp) split the prefill so that a restored checkpoint and a cold prefill sum the
// recurrence alike, and each piece after the last message is its own call, a decode step's read
// of every weight (DECISIONS 62). Read from the artifact's own frontend (its chat template and
// tokenizer, a boundary inside a token dropped): the frontiers of a one-message prompt that keep
// their distance from its end whatever the message's length. nullopt where the artifact has no
// frontend this build reads.
[[nodiscard]] std::optional<std::uint32_t> prompt_tail_calls(const std::filesystem::path& artifact);

} // namespace ninferno::runtime
