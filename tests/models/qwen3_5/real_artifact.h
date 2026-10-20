#pragma once

// What a real-test artifact carries. The Qwen real tests run on whatever NINFERNO_TEST_ARTIFACT
// names, published artifacts included: a text-only artifact (no Vision), one without MTP or a
// masked drafter, one without a proposal head. A case needing an absent component is skipped with
// a line saying so; nothing is narrowed silently.

#include "artifact/reader.h"
#include "ninferno/types.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::test {

struct RealArtifact {
    bool vision   = false;
    bool mtp      = false;
    bool dflash   = false;
    bool dflash2  = false;
    bool proposal = false;
    bool moe      = false;

    [[nodiscard]] bool has(SpeculativeBackend backend) const noexcept {
        switch (backend) {
        case SpeculativeBackend::None:
            return true;
        case SpeculativeBackend::Mtp:
            return mtp;
        case SpeculativeBackend::DFlash:
            return dflash;
        case SpeculativeBackend::DFlash2:
            return dflash2;
        }
        return false;
    }
};

// The components of `path`, read once per process (a test binary runs on one artifact).
inline const RealArtifact& real_artifact(const char* path) {
    static const std::string first = path;
    static const RealArtifact out  = [&] {
        const artifact::Reader reader(first);
        const auto& components = reader.directory().components;
        RealArtifact result;
        result.vision   = components.contains("vision");
        result.mtp      = components.contains("mtp");
        result.dflash   = components.contains("dflash");
        result.dflash2  = components.contains("dflash2");
        result.proposal = reader.directory().component("text").proposal.has_value();
        result.moe      = reader.directory().component("text").config.contains("num_experts");
        return result;
    }();
    if (first != path) { throw std::logic_error("real tests inspect one artifact per process"); }
    return out;
}

inline void skip_case(std::string_view what, std::string_view why) {
    std::cout << "skip " << what << ": " << why << '\n';
}

// The optimized proposal head when the artifact has one; otherwise the full head, said once.
inline ProposalHead proposal_head(const char* path, ProposalHead wanted) {
    if (wanted != ProposalHead::Optimized || real_artifact(path).proposal) { return wanted; }
    static const bool said = [] {
        skip_case("optimized proposal head", "the artifact has no proposal head; the full head drafts");
        return true;
    }();
    (void)said;
    return ProposalHead::Full;
}

// Vision as requested when the artifact has it; a text-only artifact runs without it, said once.
inline bool vision(const char* path, bool wanted) {
    if (!wanted || real_artifact(path).vision) { return wanted; }
    static const bool said = [] {
        skip_case("Vision cases", "no vision component (text-only artifact)");
        return true;
    }();
    (void)said;
    return false;
}

} // namespace ninferno::test
