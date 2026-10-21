#pragma once

#include "ninferno/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::runtime {

// Throws the free-memory refusal again with `advice` (what fits instead) after its message, still
// as a CapacityError: callers recover from a shortage by that type (ninferno-serve's fit and plain
// fallbacks, the load bounds moved to captured depths), not by the message.
[[noreturn]] void throw_capacity_shortage(const CapacityError& refusal, const std::string& advice);

// What a capacity refusal of construct_model says fits instead (appended to the free-memory
// refusal by throw_capacity_shortage).
struct CapacityAdvice {
    std::vector<ConcurrencyFit> fits;  // what fits instead, the asked concurrency first
    std::vector<std::uint32_t> chunks; // the prefill chunks the fits may take, narrowest last
    // An explicit --prefill-chunk that fits no concurrency while a narrower one fits (0: none).
    std::uint32_t narrowed_from   = 0;
    bool prefill_chunk_automatic  = false;
    bool kv_capacity_explicit     = false; // the fits take an automatic KV capacity in its place
    bool context_automatic        = false;
    std::uint32_t ceiling         = 0; // the asked max_context
    std::uint32_t max_concurrency = 1;
    // Every sizing option explicit: the plan reserves an allowance for its CUDA graphs.
    bool graph_allowance = false;
};

[[nodiscard]] std::string capacity_advice(const CapacityAdvice& advice);

// The plan ladder's contexts (plan_ladder.h) are whole multiples of this grain, one paged-KV page
// (core/paged_kv_cache.h, checked in kv_capacity.cpp): a context between two pages takes the next
// page's memory.
inline constexpr std::uint32_t kAutomaticContextGrain = 64;
// The shortest survival context of the ladder (only its last-resort rung, without speculation, is
// shorter).
inline constexpr std::uint32_t kAutomaticContextMinimum = 1024;
// The ladder's context steps end here (or at the model's own, when smaller); below it the rungs
// are survival rungs, served with a warning.
inline constexpr std::uint32_t kAutomaticContextFloor = 8192;

// The automatic checkpoint pool (H, --device-state-slots omitted) holds up to two cached Device
// states per active lane beside the lane's own: three per request, so an agent's tool loop keeps
// its turn closure and its endpoint while other requests run (SGLang reserves three state slots
// per running request). The ladder's top rung takes it; its states rungs step it down.
inline constexpr std::uint32_t kAutomaticCachedStatesPerLane = 2;

// The automatic checkpoint pool for `concurrency` active lanes.
[[nodiscard]] constexpr std::uint32_t
automatic_cached_device_states(std::uint32_t concurrency) noexcept {
    return kAutomaticCachedStatesPerLane * concurrency;
}

// The narrowest prefill chunk the ladder's chunk rungs go to (the smallest the Program accepts: a
// Gemma 4 sliding ring holds window + chunk - 1 rows per slot), and the multiple every chunk is.
inline constexpr std::uint32_t kNarrowestAutomaticPrefillChunk = 128;
inline constexpr std::uint32_t kPrefillChunkQuantum            = 128;

} // namespace ninferno::runtime
