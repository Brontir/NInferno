#pragma once

// What the serving Engine (ninferno::Engine: ninferno-serve, the ninferno CLI) runs of a model
// type's speculation. A profile is tuned where the most configurations run (ninferno-gemma4's
// Session also runs DFlash and adaptive depths for Gemma 4); a server can run only what its Engine
// family implements, so spec-tune also records the best of those (Profile::served) and the server's
// menu takes its choices from there. The Engine families check their own limits against this table
// (static_assert beside their constants), so it cannot drift from them.

#include "spec/config.h"

#include <cstdint>
#include <string_view>

namespace ninferno::spec {

struct EngineSupport {
    std::int32_t mtp_depth    = 0; // deepest MTP window (0: no MTP)
    std::int32_t dflash_depth = 0; // deepest DFlash / DFlash2 window (0: none)
    bool dflash2              = false;
    bool proposal_head        = false; // the optimized proposal head (+head)
    // A DFlash round drafts its drafter's whole block (DFlashDraftWidth::Block) as well as the
    // anchor and its K drafts only; without it, only the latter.
    bool dflash_block_width = false;

    // Adaptive depths (mtp:K:K0) and prompt lookup run in no Engine family.
    [[nodiscard]] constexpr bool runs(const SpecConfig& c) const noexcept {
        if (!c.enabled()) { return true; }
        if (c.adaptive() || c.k < 1 || (c.proposal_head && !proposal_head)) { return false; }
        switch (c.mode) {
        case Mode::Mtp:
            return c.k <= mtp_depth;
        case Mode::DFlash:
            return c.k <= dflash_depth;
        case Mode::DFlash2:
            return dflash2 && c.k <= dflash_depth;
        default:
            return false;
        }
    }
    [[nodiscard]] constexpr bool any() const noexcept { return mtp_depth > 0 || dflash_depth > 0; }
};

// The widest DFlash block the serving Program's round holds (its anchor and the drafts behind it);
// a round verifies at most one block's drafts.
inline constexpr std::int32_t kDFlashBlockColumns = 16;

// By the model_type an artifact's text component declares (runtime/engine/family_registry.cpp).
[[nodiscard]] constexpr EngineSupport engine_support(std::string_view model_type) noexcept {
    if (model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text") {
        return {.mtp_depth = 5, .dflash_depth = 15, .dflash2 = true, .proposal_head = true};
    }
    if (model_type == "gemma4_text") { return {.mtp_depth = 15}; }
    if (model_type == "nemotron_h") { return {.mtp_depth = 7}; }
    // A dense or MoE decoder (gpt-oss) runs a DFlash drafter paired at load: at most a block's
    // drafts (the drafter's own block may be narrower, gpt-oss's holds 8; the Engine refuses a
    // deeper window). Both run DenseExecution's round, block width included.
    if (model_type == "dense_decoder" || model_type == "moe_decoder") {
        return {.dflash_depth = kDFlashBlockColumns - 1, .dflash_block_width = true};
    }
    return {};
}

// A registry engine family (spec/registry.h ModelEntry::engine) as the serving Engine runs it: its
// model types' support (the dense family's MoE decoder runs as its dense one).
[[nodiscard]] constexpr EngineSupport engine_family_support(std::string_view engine) noexcept {
    if (engine == "qwen3_5") { return engine_support("qwen3_5_text"); }
    if (engine == "gemma4") { return engine_support("gemma4_text"); }
    if (engine == "nemotron_h") { return engine_support("nemotron_h"); }
    if (engine == "dense") { return engine_support("dense_decoder"); }
    return {};
}

// Whether an engine family serves a drafter of `kind` (the registry's kinds): a family's own
// format (Qwen's MTP layers, Gemma 4's assistant, NemotronH's MTP block) on that family, a DFlash
// drafter (it reads any target's tapped layers) on any, and either only where the family's Engine
// runs its mode. The one answer to what a drafter serves with: the drafter plan fetches by it, the
// registry's entries and profiles are held to it, and the Engine families check this table
// against their own limits (static_assert beside their constants).
[[nodiscard]] constexpr bool engine_runs_kind(std::string_view engine,
                                              std::string_view kind) noexcept {
    const EngineSupport s = engine_family_support(engine);
    if (kind == "mtp") { return engine == "qwen3_5" && s.mtp_depth > 0; }
    if (kind == "gemma4_assistant") { return engine == "gemma4" && s.mtp_depth > 0; }
    if (kind == "nemotron_h_mtp") { return engine == "nemotron_h" && s.mtp_depth > 0; }
    if (kind == "dflash") { return s.dflash_depth > 0; }
    if (kind == "dflash2") { return s.dflash2 && s.dflash_depth > 0; }
    return false;
}

} // namespace ninferno::spec
