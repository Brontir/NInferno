#pragma once

// ninferno-convert --relayout: an existing artifact's routed-expert banks moved between their
// published ggml layout and ggml_frag_v1 (core/ggml_frag.h) without the checkpoint. Only the order
// of each bank row's code bytes and the objects' layout names change; every other byte, the
// geometry and the artifact id stay.
//
// In place (no `out`), the payload is rewritten chunk by chunk under an undo journal
// (<artifact>.relayout-journal, at most one chunk and the directory): each chunk's original bytes
// are made durable there before the chunk is overwritten, and the directory is rewritten last,
// padded to its reserved length. A run that stops (crash, power loss, kill) leaves the journal;
// rerunning the same command restores the interrupted chunk and finishes. While the journal exists
// artifact::Reader refuses the artifact (the engines do not load it). An exclusive flock on the
// entry file keeps two relayouts apart; nothing keeps a server that has the artifact open from
// reading bytes being rewritten, so a relayout in place must not run while one does.
//
// With `out`, a new artifact (new id) is written and the source is only read; the source's
// <artifact>.speculation.json and <artifact>.spec_profile.json are carried over with the new id.
// Either way <artifact>.conversion.json, when present, records the relayout ("relayouts").

#include "convert/expert_layout.h"
#include "convert/files.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace ninferno::convert {

struct RelayoutOptions {
    std::filesystem::path artifact;                // --target-artifact
    std::optional<std::filesystem::path> out;      // --out: a second artifact; else in place
    ExpertLayout layout          = ExpertLayout::Auto;
    std::uint64_t max_file_bytes = kDefaultMaxFileBytes; // --max-file-bytes (with --out)
    // Bytes read, permuted and written at a time: the most an undo journal holds.
    std::uint64_t chunk_bytes = 64ULL << 20;
    std::function<void(const std::string&)> log; // progress lines; empty for silence
    // Tests: in place, stop with an exception once this many permuted bytes are written, in the
    // middle of a chunk, leaving the artifact and the journal as a crash there would.
    std::optional<std::uint64_t> interrupt_after;
    // Tests: in place, stop with an exception after half the new directory is written.
    bool interrupt_directory = false;
};

struct RelayoutResult {
    std::uint64_t objects    = 0; // banks whose layout changed
    std::uint64_t code_bytes = 0; // code-plane bytes permuted
    bool resumed             = false;
    double seconds           = 0.0;
};

[[nodiscard]] RelayoutResult relayout(const RelayoutOptions& options);

// The journal an in-place relayout keeps next to `artifact` while it runs.
[[nodiscard]] std::filesystem::path relayout_journal(const std::filesystem::path& artifact);

} // namespace ninferno::convert
