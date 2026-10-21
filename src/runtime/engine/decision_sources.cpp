#include "runtime/engine/decision_sources.h"

#include "artifact/reader.h"
#include "artifact/source_identity.h"
#include "core/hex.h"
#include "core/sha256.h"

#include <exception>

namespace ninferno::runtime {

std::string drafter_identity(const std::filesystem::path& drafter) noexcept {
    try {
        if (drafter.empty()) { return {}; }
        return artifact::source_identity(drafter).value_or("none");
    } catch (...) { return "unreadable"; }
}

std::string drafter_artifact_id(const std::filesystem::path& drafter) noexcept {
    try {
        if (drafter.empty()) { return {}; }
        const artifact::Reader reader(drafter);
        return hex_encode(reader.artifact_id().data(), reader.artifact_id().size());
    } catch (...) { return "unreadable"; }
}

std::string context_cost_presets_source(const std::filesystem::path& presets) noexcept {
    try {
        if (presets.empty()) { return "none"; }
        return named_file_sha256(presets).value_or("unreadable");
    } catch (...) { return "unreadable"; }
}

} // namespace ninferno::runtime
