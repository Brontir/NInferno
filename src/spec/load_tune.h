#pragma once

// spec-tune --load's plan and bookkeeping, apart from the engines it drives
// (apps/cli/spec_tune.cpp): the engines one bound needs, the cells they measure, the file that
// keeps them between runs and the choice per concurrency. Pure functions, so the tune's decisions
// are tested without a GPU.

#include "spec/profile.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::spec {

// The load table of one cell engine, measuring `bound` drafts under load (0: plain decode): every
// step of two rows or more bounded there, none at `window` (the engine's deepest depth: the engine
// as it runs without a table). With only the served menu's depths and its own bound as launch
// windows, the engine captures what a server bounded there does, not every candidate bound's
// windows at every batch: Gemma 4 31B at C=8 on an RTX 4090, and at C=4 on an RTX 5080, did not fit
// an engine capturing them all.
[[nodiscard]] std::vector<LoadBound> load_cell_table(std::int32_t bound, std::int32_t window);

// The engines spec-tune --load builds for one bound, widest first. `measure(widest, wanted)` builds
// an engine for up to `widest` rows, measures the concurrencies of `wanted` (ascending) its KV pool
// holds, and returns those it measured (empty: the engine does not fit, or its pool holds none of
// them). A concurrency left over is tried again on an engine sized for it (fewer rows capture fewer
// graphs, leaving more of the memory to the pool), the widest remaining first; one no engine holds
// is returned, ascending. At most one engine per concurrency.
[[nodiscard]] std::vector<std::uint32_t> measure_load_cells(
    std::vector<std::uint32_t> wanted,
    const std::function<std::vector<std::uint32_t>(std::uint32_t,
                                                   const std::vector<std::uint32_t>&)>& measure);

// One cell: `concurrency` requests in flight over the corpus, every request capped at `bound`
// drafts (0: plain decode) on an engine bounded there.
struct LoadCell {
    std::uint32_t concurrency = 0;
    std::uint32_t bound       = 0;
    std::int64_t tokens       = 0;
    double seconds            = 0;
    std::uint64_t rounds = 0, drafted = 0, accepted = 0, plain_steps = 0, trips = 0;

    [[nodiscard]] double tok_s() const noexcept { return seconds > 0 ? tokens / seconds : 0.0; }
};

// "plain" (0), "none" (the window: no bound), else the depth.
[[nodiscard]] std::string load_bound_name(std::uint32_t bound, std::uint32_t window);
// The inverse; throws std::invalid_argument naming the text for anything else, or a depth beyond
// the window.
[[nodiscard]] std::uint32_t parse_load_bound_name(std::string_view name, std::uint32_t window);

// What spec-tune --load --out keeps, rewritten after every cell. `settings` fingerprints what the
// cells were measured with (the GPU, the artifact and its drafters, the sampling, the corpus, the
// context, the KV codec, the prefill chunk, the candidate bounds); --resume takes the cells only
// when every field of it is equal. `prefill_chunk` is the chunk every engine of the tune runs.
struct LoadCells {
    nlohmann::json settings = nlohmann::json::object();
    std::uint32_t window        = 0;
    std::uint32_t prefill_chunk = 0; // 0: not pinned yet (no engine built)
    std::vector<LoadCell> cells;
    std::vector<LoadCell> unheld; // concurrency and bound only: no engine held them
};

// `load` and `measured_on` are the result once decided (empty while measuring).
[[nodiscard]] nlohmann::json load_cells_json(const LoadCells& cells, const std::string& load,
                                             const std::string& measured_on);
// Throws std::runtime_error when the settings differ from `expected` (naming the first field that
// does), a bound name is unknown, or a (concurrency, bound) cell appears twice.
[[nodiscard]] LoadCells load_cells_from_json(const nlohmann::json& doc,
                                             const nlohmann::json& expected);

// The choice per concurrency over the cells measured there: the shallowest bound within `tolerance`
// of the fastest, speculation only when it beats plain decode by `min_gain`. A concurrency without
// plain decode, or without any speculative bound, measured is undecided: nothing to compare.
// Throws std::invalid_argument unless 0 <= tolerance < 1 and min_gain >= 0.
struct LoadDecision {
    std::vector<LoadChoice> choices; // ascending concurrency
    std::vector<std::uint32_t> undecided;
};
[[nodiscard]] LoadDecision decide_load(const std::vector<LoadCell>& cells,
                                       const std::vector<std::uint32_t>& concurrency,
                                       const std::vector<std::uint32_t>& bounds, double tolerance,
                                       double min_gain);

// The published table of a decision (load_table). An undecided concurrency between two decided ones
// takes the next decided one's bound (load_table's lanes rule: a step never runs deeper than
// measured above it). Above the widest decided concurrency, when a wider one was asked for and not
// decided, the bound is capped at the shallowest one decided: the load it would have been measured
// at is heavier than any measured.
[[nodiscard]] std::vector<LoadBound> decided_load_table(const LoadDecision& decision,
                                                        std::int32_t window);

} // namespace ninferno::spec
