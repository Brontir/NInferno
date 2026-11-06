#pragma once

#include "runtime/engine/hf_text_frontend.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace ninferno::runtime {

// The whole-model calls a prefill runs after a prompt's own chunks: the generation prompt's
// execution boundaries (the assistant turn's begin and its body's, models/qwen3_5/frontend
// prompt_layout.cpp) split the prefill so that a restored checkpoint and a cold prefill sum the
// recurrence alike, and each piece after the last message is its own call, a decode step's read
// of every weight (DECISIONS 62). Read from the artifact's own frontend (its chat template and
// tokenizer, a boundary inside a token dropped): the frontiers of a one-message prompt that keep
// their distance from its end whatever the message's length, and one more where the automatic
// shared prefix at the message's end is captured below it, on the 64-token grid (the remainder's
// call, prompt_tail.cpp), where the cache captures at all (`prefix_reuse`); `chat_template` the
// served --chat-template, empty for the artifact's own. Throws where the frontend cannot be read.
[[nodiscard]] std::uint32_t prompt_tail_calls(const std::filesystem::path& artifact,
                                              const std::filesystem::path& chat_template = {},
                                              bool prefix_reuse = true);

// The same from a frontend's own files where no artifact holds them yet: a GGUF's, rebuilt from its
// metadata (convert::gguf_frontend_resources), as its conversion will write them.
[[nodiscard]] std::uint32_t prompt_tail_calls_of_files(const HfTextFiles& files,
                                                       const std::filesystem::path& chat_template = {},
                                                       bool prefix_reuse = true);

// The same for the product's plan (product::set_prompt_tail_calls_source): a frontend that cannot
// be read is said on stderr, once per artifact, and prices no tail call.
[[nodiscard]] std::optional<std::uint32_t>
prompt_tail_calls_or_warn(const std::filesystem::path& artifact,
                          const std::filesystem::path& chat_template = {}, bool prefix_reuse = true);
// prompt_tail_calls_of_files for the plan (product::set_prompt_tail_calls_files_source); `what`
// names the files' source in the warning.
[[nodiscard]] std::optional<std::uint32_t>
prompt_tail_calls_of_files_or_warn(const HfTextFiles& files, std::string_view what,
                                   const std::filesystem::path& chat_template = {}, bool prefix_reuse = true);

} // namespace ninferno::runtime
