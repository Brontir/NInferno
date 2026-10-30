#pragma once

#include "ninferno/types.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "none" || value == "off") { return SpeculativeBackend::None; }
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend (none, off, mtp, dflash, dflash2): " +
                                std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// A speculation as the products' logs name it: "<backend> K=<depth>".
[[nodiscard]] inline std::string speculation_label(const SpeculativeOptions& speculative) {
    return std::string(speculative_backend_name(speculative.backend)) +
           " K=" + std::to_string(speculative.draft_tokens);
}

// Why a speculation gave way to a fallback: the Engine refused it (`why`) at this context.
[[nodiscard]] inline std::string fit_refusal(const std::string& label, std::uint32_t max_context,
                                             std::string_view why, bool automatic = false) {
    // An automatic context (under --max-context-cap: max_context is its ceiling).
    const std::string context =
        !automatic         ? "max_context " + std::to_string(max_context)
        : max_context != 0 ? "an automatic context (at most " + std::to_string(max_context) + ")"
                           : std::string("an automatic context");
    return label + " does not fit " + context + " on this device (" + std::string(why) + ")";
}

// --draft-width: the DFlash drafter's width in every round (SpeculativeOptions::draft_width).
[[nodiscard]] inline DFlashDraftWidth parse_draft_width(std::string_view value) {
    // Case-sensitive, as every other flag value and --spec-load's width (spec::draft_width_from).
    if (value == "block") { return DFlashDraftWidth::Block; }
    if (value == "drafts") { return DFlashDraftWidth::Drafts; }
    throw std::invalid_argument("invalid draft width (block or drafts): " + std::string(value));
}

// --draft-width names a DFlash drafter's width: under an explicit --spec dflash|dflash2, or under
// --spec auto, which drops it with a note when the served speculation does not run it
// (product::fit_draft_widths).
inline void validate_draft_width(const SpeculativeOptions& options, bool from_profile) {
    if (options.draft_width != DFlashDraftWidth::Default && !from_profile &&
        options.backend != SpeculativeBackend::DFlash &&
        options.backend != SpeculativeBackend::DFlash2) {
        throw std::invalid_argument("--draft-width requires --spec dflash, dflash2 or auto");
    }
}

// SpeculativeOptions::load as text: "2:7,3:3,5:0" (lanes:depth), "none" when empty.
[[nodiscard]] inline std::string speculation_load_label(const std::vector<SpeculationLoad>& load) {
    if (load.empty()) { return "none"; }
    std::string out;
    for (const SpeculationLoad& bound : load) {
        if (!out.empty()) { out += ','; }
        out += std::to_string(bound.lanes) + ':' + std::to_string(bound.draft_tokens);
        if (bound.draft_width == DFlashDraftWidth::Block) { out += ":block"; }
        if (bound.draft_width == DFlashDraftWidth::Drafts) { out += ":drafts"; }
    }
    return out;
}

// The draft depth an explicit --spec runs without --draft-tokens when the model's profile for this
// GPU has none for that backend: the depths the documentation's examples use.
[[nodiscard]] inline std::uint32_t default_draft_tokens(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::Mtp ? 3U : backend == SpeculativeBackend::None ? 0U : 7U;
}

// `depth_from_model`: an explicit --spec may omit --draft-tokens (0), and the server takes the
// depth from the model (ninferno-serve); otherwise the depth is required.
inline void validate_speculative_cli_options(const SpeculativeOptions& options,
                                             bool depth_from_model = false) {
    const bool omitted = depth_from_model && options.draft_tokens == 0;
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full ||
            !options.drafter.empty()) {
            throw std::invalid_argument(
                "--draft-tokens, --drafter and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        // The widest MTP round any family runs (the Program's round domain); the model's own
        // maximum is the Engine's check (Qwen 3.5's MTP layer drafts up to 5, Gemma 4's
        // assistant up to 15).
        if (!omitted && (options.draft_tokens == 0 || options.draft_tokens > 15)) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (!omitted && (options.draft_tokens == 0 || options.draft_tokens > 15)) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (!omitted && (options.draft_tokens == 0 || options.draft_tokens > 15)) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninferno::product
