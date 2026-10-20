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

// The automatic KV headroom on a GPU shared through WDDM (WSL 2's /dev/dxg): Windows' own use of
// the card (the desktop compositor, other applications: 1.2-1.9 GiB on an RTX 5090 driving a
// display) moves while ninferno runs, and WDDM pages Device memory out to Host memory when the card
// fills, without an error. A pool that left the driver's 32 MiB prefilled 4x slower (Qwen3.8-27B
// NVFP4, C=8 x 65,536 on an RTX 5090: 2.26k against 9.34k tok/s at 24K tokens; decode unchanged);
// 0.45 GiB below it already ran at full speed. One 32nd of the card, at most 1 GiB, is kept back
// (an RTX 5090: 1 GiB, about 35,000 of 285,000 KV tokens there) as far as the free memory allows:
// it is best-effort headroom, so a plan that fitted with the driver's 32 MiB still fits (a small
// card keeps less). An explicit capacity is kept.
//
// A card that also drives the display (`drives_display`: Windows' kernel-run watchdog is on, as on
// every GPU with a monitor) keeps an 8th, at most 4 GiB: the desktop's own use grows by gigabytes
// (a browser, a video) and WDDM then pages the pool out while the compositor waits. An RTX 5090
// whose pool left 0.55 GiB (Gemma 4 26B, 1.06M KV tokens) took 1.8 s, steps of 244 ms, to give
// another process 3 GiB, the desktop frozen meanwhile; with 5.5 GiB left, 51 ms, as with no pool.
[[nodiscard]] KvCapacityPolicy with_paravirtual_headroom(KvCapacityPolicy policy, bool paravirtual,
                                                         bool drives_display,
                                                         std::size_t device_total_bytes);

// A GPU that shares the host's memory (cudaDeviceProp::integrated: DGX Spark's GB10, Jetson)
// reports that memory as its own, so an automatic pool would leave the operating system, its page
// cache and every other process the driver's 32 MiB. It keeps an 8th, at least 4 GiB (the share
// pinned host memory leaves the host, core/host_memory.cpp), best-effort like the above.
[[nodiscard]] KvCapacityPolicy with_unified_memory_headroom(KvCapacityPolicy policy,
                                                            bool integrated,
                                                            std::size_t device_total_bytes);

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// The cached Device StateImages of the context cache. A configured count is kept, and refused with
// its numbers when it does not fit. An unconfigured one is automatic, a count in
// [automatic_minimum, automatic_default]. With an explicit KV capacity it is the largest count
// whose Program reservation fits: the capacity asked for comes first, the cached states next.
// With an automatic KV capacity the pool comes first, one whole slot context (the curve's minimum
// pages) at a time: the largest count whose pool holds as many whole contexts as the smallest
// count's, the largest pool; with room for every slot's context (the curve's maximum) that is the
// largest count beside it. States first, a plain Gemma 4 31B at C=4 x 4096 on an RTX 5090 kept 4
// cached states beside a pool of 4,608 of the 16,384 tokens. This sizes the pool at a given
// context (--max-context without --kv-capacity); an automatic max_context is chosen before it
// with one cached state per slot, fewer only below kAutomaticContextFloor (construct_model).
// The minimum is 1, or 0 when the Host tier keeps
// StateImages (a checkpoint is then captured to and restored from Host memory). When not even the
// smallest count fits, the refusal carries the numbers of that smallest choice. The choice depends
// only on the policy, the curves and the free bytes.
struct CachedStatePolicy {
    std::optional<std::uint32_t> configured;
    std::uint32_t automatic_default = 0;
    std::uint32_t automatic_minimum = 1;
};

// curve_for(H) is the Program's capacity curve with H cached Device states.
[[nodiscard]] KvCapacityResolution
resolve_runtime_capacity(const KvCapacityPolicy& policy, const CachedStatePolicy& states,
                         const std::function<SequenceCapacityCurve(std::uint32_t)>& curve_for,
                         std::size_t available_runtime_bytes);

// resolve_runtime_capacity without the refusal: nullopt when the free bytes are short. Invalid
// policies and curves still throw.
[[nodiscard]] std::optional<KvCapacityResolution>
try_resolve_runtime_capacity(const KvCapacityPolicy& policy, const CachedStatePolicy& states,
                             const std::function<SequenceCapacityCurve(std::uint32_t)>& curve_for,
                             std::size_t available_runtime_bytes);

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

// An automatic max_context (EngineOptions::max_context_automatic) considers the model's own
// position capacity and, below it, the multiples of this grain from kAutomaticContextMinimum up.
// The grain is one paged-KV page (core/paged_kv_cache.h, checked in kv_capacity.cpp): a context
// between two pages takes the next page's memory, so no finer grain fits more, and a coarser one
// leaves up to a grain of KV unused (31B NVFP4 at C=1: 80 KiB per token, 80 MiB per 1024 tokens).
inline constexpr std::uint32_t kAutomaticContextGrain = 64;
// The shortest automatic max_context below the model's own: a plan that fits less is refused.
inline constexpr std::uint32_t kAutomaticContextMinimum = 1024;
// Below this context (or the model's own, when smaller) an automatic max_context gives up cached
// Device states of the context cache for a longer context, then (EngineOptions::
// prefill_chunk_automatic) prefill width (construct_model).
inline constexpr std::uint32_t kAutomaticContextFloor = 8192;

// The automatic checkpoint pool (H, --device-state-slots omitted) holds up to two cached Device
// states per active lane beside the lane's own: three per request, so an agent's tool loop keeps
// its turn closure and its endpoint while other requests run (SGLang reserves three state slots
// per running request). The automatic context is still chosen with one per lane.
inline constexpr std::uint32_t kAutomaticCachedStatesPerLane = 2;

// The automatic checkpoint pool for `concurrency` active lanes.
[[nodiscard]] constexpr std::uint32_t
automatic_cached_device_states(std::uint32_t concurrency) noexcept {
    return kAutomaticCachedStatesPerLane * concurrency;
}

// The narrowest prefill chunk an automatic choice goes to (the smallest the Program accepts: a
// Gemma 4 sliding ring holds window + chunk - 1 rows per slot), and the multiple every chunk is.
inline constexpr std::uint32_t kNarrowestAutomaticPrefillChunk = 128;
inline constexpr std::uint32_t kPrefillChunkQuantum            = 128;

// The largest candidate context in [1, ceiling] that fits: `ceiling` itself when fits(ceiling),
// else the largest multiple of kAutomaticContextGrain in [kAutomaticContextMinimum, ceiling) that
// fits, found by bisection (fits must not accept a context above one it refuses: a longer context
// never needs less memory); nullopt when not even the smallest candidate,
// min(ceiling, kAutomaticContextMinimum), fits. The choice depends only on the ceiling and the
// predicate.
[[nodiscard]] std::optional<std::uint32_t>
largest_fitting_context(std::uint32_t ceiling, const std::function<bool(std::uint32_t)>& fits);

} // namespace ninferno::runtime
