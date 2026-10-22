#pragma once

// The Engine's load bounds (SpeculativeOptions::load) as the planner and the decode step read
// them. Header-only and free of device types, so CPU tests check them directly.

#include "ninferno/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace ninferno::models::qwen3_5::detail {

// Whether the load bounds (SpeculativeOptions::load) have strictly ascending lanes from 2 (one
// request is never bounded), depths no deeper than the engine's window (the largest of
// draft_tokens and request_draft_tokens: a bound never widens a launch), and no speculating bound
// after a bound of 0 (a row turned plain cannot resume speculating).
[[nodiscard]] inline bool speculation_load_valid(const SpeculativeOptions& options) {
    std::uint32_t window = options.draft_tokens;
    for (const std::uint32_t k : options.request_draft_tokens) { window = std::max(window, k); }
    std::uint32_t lanes = 1;
    bool plain          = false;
    for (const SpeculationLoad& bound : options.load) {
        if (bound.lanes <= lanes || bound.draft_tokens > window ||
            (plain && bound.draft_tokens != 0)) {
            return false;
        }
        lanes = bound.lanes;
        plain = bound.draft_tokens == 0;
    }
    return true;
}

// The context-class bounds (SpeculativeOptions::class_depth): non-increasing by class, each
// kNoClassBound or at most the engine's window (a bound never widens a launch).
[[nodiscard]] inline bool speculation_class_depth_valid(const SpeculativeOptions& options) {
    std::uint32_t window = options.draft_tokens;
    for (const std::uint32_t k : options.request_draft_tokens) { window = std::max(window, k); }
    std::uint32_t previous = SpeculativeOptions::kNoClassBound;
    for (const std::uint32_t k : options.class_depth) {
        if (k > previous || (k != SpeculativeOptions::kNoClassBound && k > window)) {
            return false;
        }
        previous = k;
    }
    return true;
}

// The load bound of a step of `rows` rows: the entry with the largest lanes at most `rows`, or
// the largest uint32 (no bound) below the first entry.
[[nodiscard]] inline std::uint32_t speculation_load_bound(const std::vector<SpeculationLoad>& load,
                                                          std::size_t rows) noexcept {
    std::uint32_t bound = std::numeric_limits<std::uint32_t>::max();
    for (const SpeculationLoad& entry : load) {
        if (entry.lanes > rows) { break; }
        bound = entry.draft_tokens;
    }
    return bound;
}

// The DFlash drafter's width in a round of `rows` rows: `forced` when it names one
// (SpeculativeOptions::draft_width), else the width of the load bound that applies to the rows
// when it names one, else the engine's choice: the whole block for one row, K + 1 columns from two
// rows on. One row's block is cheap beside its verify, and the block's full attention proposes
// better drafts; from two rows on the drafter's columns grow with every row and the narrow pass
// pays (RTX 5090, Muse Glimmer 30B NVFP4 + DFlash, served tok/s, block -> K + 1: C=2 K3 270.7 ->
// 280.9, C=4 K3 380.9 -> 401.9; C=1 K7 164.8 -> 160.7).
[[nodiscard]] inline DFlashDraftWidth dflash_draft_width(DFlashDraftWidth forced,
                                                         const std::vector<SpeculationLoad>& load,
                                                         std::size_t rows) noexcept {
    if (forced != DFlashDraftWidth::Default) { return forced; }
    DFlashDraftWidth bound = DFlashDraftWidth::Default;
    for (const SpeculationLoad& entry : load) {
        if (entry.lanes > rows) { break; }
        bound = entry.draft_width;
    }
    if (bound != DFlashDraftWidth::Default) { return bound; }
    return rows <= 1 ? DFlashDraftWidth::Block : DFlashDraftWidth::Drafts;
}

// Whether the options name a DFlash draft width anywhere (SpeculativeOptions::draft_width or a
// load bound's).
[[nodiscard]] inline bool names_draft_width(const SpeculativeOptions& options) noexcept {
    return options.draft_width != DFlashDraftWidth::Default ||
           std::any_of(options.load.begin(), options.load.end(), [](const SpeculationLoad& bound) {
               return bound.draft_width != DFlashDraftWidth::Default;
           });
}

// The widest speculative launch under `load` for up to `max_concurrency` rows: below the lanes of
// its bound of 0, if any.
[[nodiscard]] inline std::uint32_t speculative_batch_limit(const std::vector<SpeculationLoad>& load,
                                                           std::uint32_t max_concurrency) noexcept {
    for (const SpeculationLoad& entry : load) {
        if (entry.draft_tokens == 0) { return std::min(max_concurrency, entry.lanes - 1U); }
    }
    return max_concurrency;
}

// A launch window's shape as the planner lays it out (ProgramImpl's LaunchWindow has the same
// members): its draft depth and the smallest speculative batch its graphs cover.
struct LaunchWindowShape {
    std::uint32_t drafts    = 0;
    std::uint32_t min_batch = 1;
};

// The launch windows an engine captures for `speculative` at `max_concurrency`, ascending: every
// menu depth (draft_tokens and request_draft_tokens) from one row, and every depth only a load
// bound asks from the fewest lanes of the bounds that ask it (a smaller speculative batch takes
// the next wider window; a bound of more lanes than the engine admits never applies), then the
// copy window. None for plain decode.
[[nodiscard]] inline std::vector<LaunchWindowShape>
launch_window_shapes(const SpeculativeOptions& speculative, std::uint32_t max_concurrency) {
    std::vector<LaunchWindowShape> out;
    if (speculative.backend == SpeculativeBackend::None) { return out; }
    std::vector<std::uint32_t> menu = speculative.request_draft_tokens;
    menu.push_back(speculative.draft_tokens);
    std::vector<std::uint32_t> depths = menu;
    for (const SpeculationLoad& bound : speculative.load) {
        if (bound.draft_tokens != 0 && bound.lanes <= max_concurrency) {
            depths.push_back(bound.draft_tokens);
        }
    }
    std::sort(depths.begin(), depths.end());
    depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
    for (const std::uint32_t depth : depths) {
        std::uint32_t first = 1;
        if (std::find(menu.begin(), menu.end(), depth) == menu.end()) {
            first = max_concurrency;
            for (const SpeculationLoad& bound : speculative.load) {
                if (bound.draft_tokens == depth && bound.lanes <= max_concurrency) {
                    first = std::min(first, bound.lanes);
                }
            }
        }
        out.push_back({.drafts = depth, .min_batch = first});
    }
    // The copy window (SpeculativeOptions::ngram_draft_tokens): a round that verifies a request's
    // copy proposal, deeper than every depth, from one row on. The widest window.
    if (speculative.backend == SpeculativeBackend::Mtp &&
        speculative.ngram_draft_tokens != SpeculativeOptions::kCopyWindowAuto &&
        speculative.ngram_draft_tokens > depths.back()) {
        out.push_back({.drafts = speculative.ngram_draft_tokens, .min_batch = 1});
    }
    return out;
}

// The window a speculative launch of `batch` rows takes when its deepest row verifies `drafts`:
// the narrowest of `windows` (ascending by drafts, with drafts and min_batch members) that covers
// `drafts` and whose graphs cover `batch` rows (a depth only a load bound asks is captured from
// the bound's lanes on). None when no window does. ProgramImpl::launch_window_for, the masks of
// launch_window_batches and the riding budget (stopped_rows_ride) all take this one.
template <class Window>
[[nodiscard]] std::optional<std::size_t>
launch_window_of(const std::vector<Window>& windows, std::uint32_t drafts,
                 std::uint32_t batch) noexcept {
    for (std::size_t i = 0; i < windows.size(); ++i) {
        if (windows[i].drafts >= drafts && windows[i].min_batch <= batch) { return i; }
    }
    return std::nullopt;
}

// Whether `stopped` rows (their drafting stopped by the watchdog) ride the speculative launch in
// window `window` of `windows`, verifying for nothing: their columns, `stopped` times that
// window's width (drafts + 1), stay within `ride_columns`.
template <class Window>
[[nodiscard]] bool stopped_rows_ride(const std::vector<Window>& windows, std::size_t window,
                                     std::uint32_t stopped, std::uint32_t ride_columns) {
    return stopped * (windows.at(window).drafts + 1U) <= ride_columns;
}

// The speculative batches each launch window runs under `load`, as bit (batch - 1) of the window's
// mask: `windows` ascending (the last the engine's window), each captured from `min_batch` rows
// on. A launch of `batch` rows in a step of `rows` >= batch rows (the rest decode plainly) verifies
// at most the step's bound per row (speculation_load_bound; 0 launches nothing), and takes the
// window launch_window_of picks for its deepest row. Batches up to `batch_limit` (at most 32).
[[nodiscard]] inline std::vector<std::uint32_t>
launch_window_batches(const std::vector<std::uint32_t>& windows,
                      const std::vector<std::uint32_t>& min_batch,
                      const std::vector<SpeculationLoad>& load, std::uint32_t max_concurrency,
                      std::uint32_t batch_limit) {
    std::vector<std::uint32_t> out(windows.size(), 0U);
    if (windows.empty() || min_batch.size() != windows.size()) { return out; }
    std::vector<LaunchWindowShape> shapes;
    for (std::size_t w = 0; w < windows.size(); ++w) {
        shapes.push_back({.drafts = windows[w], .min_batch = min_batch[w]});
    }
    for (std::uint32_t batch = 1; batch <= std::min(batch_limit, 32U); ++batch) {
        std::uint32_t deepest = 0;
        for (std::uint32_t rows = batch; rows <= max_concurrency; ++rows) {
            deepest =
                std::max(deepest, std::min(speculation_load_bound(load, rows), windows.back()));
        }
        for (std::uint32_t depth = 1; depth <= deepest; ++depth) {
            if (const auto w = launch_window_of(shapes, depth, batch)) {
                out[*w] |= 1U << (batch - 1U);
            }
        }
    }
    return out;
}

// Whether `mask` (launch_window_batches) holds a batch of `batch` rows.
[[nodiscard]] inline bool window_runs_batch(std::uint32_t mask, std::uint32_t batch) noexcept {
    return batch >= 1 && batch <= 32 && ((mask >> (batch - 1U)) & 1U) != 0;
}

} // namespace ninferno::models::qwen3_5::detail
