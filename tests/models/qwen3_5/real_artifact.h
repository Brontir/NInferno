#pragma once

// What a real-test artifact carries. The Qwen real tests run on whatever NINFERNO_TEST_ARTIFACT
// names, published artifacts included: a text-only artifact (no Vision), one without MTP or a
// masked drafter, one without a proposal head. A case needing an absent component is skipped with
// a line saying so; nothing is narrowed silently.

#include "artifact/reader.h"
#include "ninferno/types.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <set>
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
    // The drafters the proposal head serves (a proposal/head Use of their final_hidden): the
    // converter's default head serves MTP and DFlash, never DFlash2.
    std::set<std::string> head_drafters;

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
        for (const char* drafter : {"mtp", "dflash", "dflash2"}) {
            if (reader.directory().uses.contains(
                    {"proposal/head", std::string(drafter) + "/final_hidden"})) {
                result.head_drafters.insert(drafter);
            }
        }
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

// The optimized proposal head when the artifact's head serves `backend`'s drafter; otherwise the
// full head, said once (#343: an MTP + DFlash2 artifact's default head serves MTP alone, and
// DFlash2 under it has no proposal/head Use).
inline ProposalHead proposal_head(const char* path, ProposalHead wanted,
                                  SpeculativeBackend backend) {
    wanted = proposal_head(path, wanted);
    const char* drafter = backend == SpeculativeBackend::Mtp       ? "mtp"
                          : backend == SpeculativeBackend::DFlash  ? "dflash"
                          : backend == SpeculativeBackend::DFlash2 ? "dflash2"
                                                                   : nullptr;
    if (wanted != ProposalHead::Optimized || drafter == nullptr ||
        real_artifact(path).head_drafters.contains(drafter)) {
        return wanted;
    }
    static std::set<std::string> said; // once per drafter
    if (said.insert(drafter).second) {
        skip_case(std::string("optimized proposal head for ") + drafter,
                  "the artifact's proposal head does not serve it; the full head drafts");
    }
    return ProposalHead::Full;
}

// A case whose engine the plan refuses (CapacityError) on a card below kSmallCardBytes (an RTX 4060
// Ti's 16 GiB under a 35B artifact: three lanes of 4K beside 14.9 GiB of weights) is skipped, said
// once per case; on a larger card the refusal is a failure (a planner regression must not pass as
// a skip). True: skipped.
inline constexpr std::size_t kSmallCardBytes = std::size_t{24} << 30;
inline bool small_card_skips(const CapacityError& error, std::string_view what) {
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || total_bytes >= kSmallCardBytes) {
        return false;
    }
    skip_case(what, std::string("does not fit this card: ") + error.what());
    return true;
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
