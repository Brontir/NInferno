#pragma once

// The file-reading half of the decision fingerprint (runtime/engine/decision_fingerprint.h is the
// pure half): how the Engine names a file a decision read. None of these throws: a file that
// cannot be read, or a name the platform cannot convert, is named "unreadable".

#include <filesystem>
#include <string>

namespace ninferno::runtime {

// A separate drafter's source identity (artifact/source_identity.h, 64 lowercase hex digits), read
// as the main artifact's is; empty for an empty path (no separate drafter), "none" for a drafter
// converted before the converter recorded its inputs, "unreadable" when its directory cannot be
// read.
[[nodiscard]] std::string drafter_identity(const std::filesystem::path& drafter) noexcept;

// Its artifact id (32 lowercase hex digits; observed, not a decision input): empty for an empty
// path, "unreadable" when its directory cannot be read.
[[nodiscard]] std::string drafter_artifact_id(const std::filesystem::path& drafter) noexcept;

// A --context-cost-presets file: "<file name> sha256:<hex of its bytes>", "none" for an empty
// path, "unreadable" when it cannot be read.
[[nodiscard]] std::string
context_cost_presets_source(const std::filesystem::path& presets) noexcept;

} // namespace ninferno::runtime
