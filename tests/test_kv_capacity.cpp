#include "runtime/engine/kv_capacity.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninferno::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic = ninferno::runtime::resolve_kv_capacity(
        ninferno::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    // A GPU shared through WDDM (WSL 2) keeps a 32nd of the card back from an automatic pool, at
    // most 1 GiB; a native one and an explicit capacity are left as asked.
    {
        using ninferno::KvCapacityPolicy;
        using ninferno::runtime::with_paravirtual_headroom;
        constexpr std::size_t kGiB = 1ULL << 30;
        const auto card32 =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), true, false, 32 * kGiB);
        const auto card16 =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), true, false, 16 * kGiB);
        const auto card48 =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), true, false, 48 * kGiB);
        const auto native =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), false, false, 32 * kGiB);
        const auto asked  = with_paravirtual_headroom(KvCapacityPolicy::explicit_capacity(4096),
                                                      true, true, 32 * kGiB);
        const auto larger = with_paravirtual_headroom(KvCapacityPolicy::automatic(2 * kGiB), true,
                                                      false, 32 * kGiB);
        constexpr std::size_t kDriver = ninferno::kDefaultKvCapacityHeadroomBytes;
        failures += check(card32.automatic_headroom_bytes == kDriver &&
                              card32.best_effort_headroom_bytes == kGiB - kDriver &&
                              card16.best_effort_headroom_bytes == kGiB / 2 - kDriver &&
                              card48.best_effort_headroom_bytes == kGiB - kDriver &&
                              native.best_effort_headroom_bytes == 0 &&
                              asked.mode == ninferno::KvCapacityMode::Explicit &&
                              asked.automatic_headroom_bytes == 0 &&
                              asked.best_effort_headroom_bytes == 0 &&
                              larger.automatic_headroom_bytes == 2 * kGiB &&
                              larger.best_effort_headroom_bytes == 0,
                          "the WDDM headroom of an automatic KV pool is not a 32nd of the card up to 1 GiB");
        // A card driving the display keeps an 8th, at most 4 GiB; a native one still nothing.
        const auto display32 =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), true, true, 32 * kGiB);
        const auto display16 =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), true, true, 16 * kGiB);
        const auto display48 =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), true, true, 48 * kGiB);
        const auto native_display =
            with_paravirtual_headroom(KvCapacityPolicy::automatic(), false, true, 32 * kGiB);
        failures += check(display32.best_effort_headroom_bytes == 4 * kGiB - kDriver &&
                              display16.best_effort_headroom_bytes == 2 * kGiB - kDriver &&
                              display48.best_effort_headroom_bytes == 4 * kGiB - kDriver &&
                              native_display.best_effort_headroom_bytes == 0,
                          "a WDDM card driving the display does not keep an 8th, up to 4 GiB");
        // A GPU sharing the host's memory keeps an 8th, at least 4 GiB; a discrete one nothing.
        using ninferno::runtime::with_unified_memory_headroom;
        const auto spark   = with_unified_memory_headroom(KvCapacityPolicy::automatic(), true,
                                                          128 * kGiB);
        const auto jetson  = with_unified_memory_headroom(KvCapacityPolicy::automatic(), true,
                                                          16 * kGiB);
        const auto discrete = with_unified_memory_headroom(KvCapacityPolicy::automatic(), false,
                                                           128 * kGiB);
        const auto explicit_spark = with_unified_memory_headroom(
            KvCapacityPolicy::explicit_capacity(4096), true, 128 * kGiB);
        failures += check(spark.best_effort_headroom_bytes == 16 * kGiB - kDriver &&
                              jetson.best_effort_headroom_bytes == 4 * kGiB - kDriver &&
                              discrete.best_effort_headroom_bytes == 0 &&
                              explicit_spark.best_effort_headroom_bytes == 0,
                          "a unified-memory GPU does not keep an 8th, at least 4 GiB");
        // The best-effort part comes out of the pages above the minimum reservation: all of it
        // with room, what there is without, and never a refusal the driver's headroom passes.
        auto shared                       = KvCapacityPolicy::automatic(50);
        shared.best_effort_headroom_bytes = 300;
        const auto roomy  = ninferno::runtime::resolve_kv_capacity(shared, curve, 1360 + 256);
        const auto tight  = ninferno::runtime::resolve_kv_capacity(shared, curve, 1100);
        failures += check(roomy.main_page_groups == 4 && roomy.automatic_headroom_bytes == 350 &&
                              tight.main_page_groups == 2 && tight.automatic_headroom_bytes == 100,
                          "the best-effort headroom is not taken from the pages above the minimum");
    }

    const auto capped = ninferno::runtime::resolve_kv_capacity(
        ninferno::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninferno::runtime::resolve_kv_capacity(
        ninferno::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninferno::runtime::resolve_kv_capacity(ninferno::KvCapacityPolicy::automatic(50),
                                                     curve, 1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // Cached Device states resolved with the KV capacity: each state costs 100 bytes.
    const auto curve_for = [&](std::uint32_t states) {
        auto c                             = curve;
        c.minimum_device_reservation_bytes = 1000 + 100ULL * states;
        return c;
    };
    const ninferno::runtime::CachedStatePolicy automatic_states{.automatic_default = 4};
    const auto all_states = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::explicit_capacity(128), automatic_states, curve_for, 1400);
    failures += check(all_states.cached_device_states == 4 &&
                          all_states.cached_device_states_default == 4 &&
                          all_states.cached_device_states_automatic &&
                          all_states.runtime_reservation_bytes == 1400,
                      "automatic cached states did not keep the default when it fits");
    const auto fewer_states = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::explicit_capacity(129), automatic_states, curve_for, 1399);
    failures += check(fewer_states.cached_device_states == 2 &&
                          fewer_states.main_page_groups == 3 &&
                          fewer_states.runtime_reservation_bytes == 1328,
                      "automatic cached states did not yield to the explicit KV capacity");
    // An automatic KV capacity comes before the cached states, one whole slot context (2 pages)
    // at a time: the largest count whose pool holds as many whole contexts as the fewest states'.
    // 1310 bytes after the headroom: one state leaves 3 pages (one context), 3 states 2 (one
    // context), 4 do not fit: 3 states.
    const auto automatic_kv = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50), automatic_states, curve_for, 1360);
    failures += check(automatic_kv.cached_device_states == 3 &&
                          automatic_kv.main_page_groups == 2 &&
                          automatic_kv.cached_device_states_automatic,
                      "automatic KV gave cached states up for less than a whole context");
    // On a GPU Windows shares the best-effort headroom comes before the states as well: 1400
    // bytes keep 250 of it beside one state; 3 states would hold the same context and keep 50.
    auto shared_kv                       = ninferno::KvCapacityPolicy::automatic(50);
    shared_kv.best_effort_headroom_bytes = 300;
    const auto headroom_first =
        ninferno::runtime::resolve_runtime_capacity(shared_kv, automatic_states, curve_for, 1400);
    failures += check(headroom_first.cached_device_states == 1 &&
                          headroom_first.automatic_headroom_bytes == 300,
                      "automatic cached states took the best-effort headroom");
    // States that cost a small part of a context (10 pages of 10 bytes; a state 5 bytes; three
    // slots). 1160 bytes: one state leaves 25 pages, three 24: both two whole contexts, so the
    // three are kept (the rule before gave them all up, no count holding the full 30).
    ninferno::runtime::SequenceCapacityCurve fine = curve;

    fine.minimum_main_page_groups             = 10;
    fine.maximum_main_page_groups             = 30;
    fine.bytes_per_additional_main_page_group = 10;
    const auto fine_for = [&](std::uint32_t states) {
        auto c                             = fine;
        c.minimum_device_reservation_bytes = 1000 + 5ULL * states;
        return c;
    };
    const ninferno::runtime::CachedStatePolicy three_states{.automatic_default = 3};
    const auto cheap_states = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50), three_states, fine_for, 1210);
    failures += check(cheap_states.cached_device_states == 3 &&
                          cheap_states.main_page_groups == 24,
                      "automatic KV gave cheap cached states up for 4% of the pool");
    // 1114 bytes: one state leaves 20 pages (two contexts), two 20, three 19 (one): two states.
    const auto one_step = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50), three_states, fine_for, 1164);
    failures += check(one_step.cached_device_states == 2 && one_step.main_page_groups == 20,
                      "automatic KV did not give up only the state that costs a whole context");
    // The jump at a whole context (kv_capacity.cpp): 1150 bytes keep 3 states beside 18 pages
    // (one context at every count), 1155 bytes 1 state beside 20 (the fewest reach two).
    const auto below_jump = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50), three_states, fine_for, 1150);
    const auto above_jump = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50), three_states, fine_for, 1155);
    failures += check(below_jump.cached_device_states == 3 && below_jump.main_page_groups == 18 &&
                          above_jump.cached_device_states == 1 &&
                          above_jump.main_page_groups == 20,
                      "automatic KV did not choose by whole contexts at the boundary");
    // 1750 bytes: 4 states leave 4 pages, 3 leave 5, 2 hold the full 6.
    const auto full_pool = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50), automatic_states, curve_for, 1800);
    failures += check(full_pool.cached_device_states == 2 && full_pool.main_page_groups == 6,
                      "automatic KV did not keep the most cached states beside the full pool");
    const auto default_full = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50), automatic_states, curve_for, 10000);
    failures += check(default_full.cached_device_states == 4 && default_full.main_page_groups == 6,
                      "automatic KV gave cached states up although the full pool fits beside them");
    const auto host_backed_kv = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::automatic(50),
        ninferno::runtime::CachedStatePolicy{.automatic_default = 4, .automatic_minimum = 0},
        curve_for, 1360);
    failures += check(host_backed_kv.cached_device_states == 0 &&
                          host_backed_kv.main_page_groups == 4,
                      "automatic KV with a Host tier did not go down to no cached Device state");
    bool automatic_kv_refused = false;
    try {
        (void)ninferno::runtime::resolve_runtime_capacity(
            ninferno::KvCapacityPolicy::automatic(50), automatic_states, curve_for, 1100);
    } catch (const std::invalid_argument& error) {
        const std::string what = error.what();
        automatic_kv_refused   = what.find("at least one cached Device state") != what.npos &&
                               what.find("1100 bytes") != what.npos;
    }
    failures += check(automatic_kv_refused,
                      "automatic KV without room for one cached state was not refused with "
                      "numbers");
    bool no_state_refused = false;
    try {
        (void)ninferno::runtime::resolve_runtime_capacity(
            ninferno::KvCapacityPolicy::explicit_capacity(128), automatic_states, curve_for, 1099);
    } catch (const std::invalid_argument& error) {
        no_state_refused = std::string(error.what()).find("1100 bytes") != std::string::npos;
    }
    failures += check(no_state_refused,
                      "a cache without room for one cached state was not refused with numbers");
    // With the Host tier keeping StateImages the automatic count may go down to none.
    const ninferno::runtime::CachedStatePolicy host_backed_states{.automatic_default = 4,
                                                                .automatic_minimum = 0};
    const auto no_state = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::explicit_capacity(128), host_backed_states, curve_for, 1099);
    failures += check(no_state.cached_device_states == 0 &&
                          no_state.cached_device_states_automatic &&
                          no_state.runtime_reservation_bytes == 1000,
                      "a Host-backed cache did not go down to no cached Device state");
    const auto host_backed_default = ninferno::runtime::resolve_runtime_capacity(
        ninferno::KvCapacityPolicy::explicit_capacity(128), host_backed_states, curve_for, 1400);
    failures += check(host_backed_default.cached_device_states == 4,
                      "a Host-backed cache gave up cached Device states that fit");
    bool nothing_refused = false;
    try {
        (void)ninferno::runtime::resolve_runtime_capacity(
            ninferno::KvCapacityPolicy::explicit_capacity(128), host_backed_states, curve_for, 999);
    } catch (const std::invalid_argument& error) {
        nothing_refused = std::string(error.what()).find("no cached Device state") !=
                              std::string::npos &&
                          std::string(error.what()).find("1000 bytes") != std::string::npos;
    }
    failures += check(nothing_refused,
                      "a Host-backed cache without room for the reservation was not refused with "
                      "numbers");
    bool configured_refused = false;
    try {
        (void)ninferno::runtime::resolve_runtime_capacity(
            ninferno::KvCapacityPolicy::explicit_capacity(128), {.configured = 4U}, curve_for,
            1399);
    } catch (const std::invalid_argument&) { configured_refused = true; }
    failures += check(configured_refused, "a configured cached state count was lowered");

    // The non-throwing resolution: nullopt exactly where resolve_runtime_capacity refuses.
    failures += check(!ninferno::runtime::try_resolve_runtime_capacity(
                           ninferno::KvCapacityPolicy::explicit_capacity(128), automatic_states,
                           curve_for, 1099)
                           .has_value() &&
                          ninferno::runtime::try_resolve_runtime_capacity(
                              ninferno::KvCapacityPolicy::explicit_capacity(129), automatic_states,
                              curve_for, 1399)
                                  ->cached_device_states == 2,
                      "try_resolve_runtime_capacity does not match resolve_runtime_capacity");

    // An automatic max_context: the ceiling when it fits, else the largest multiple of the grain
    // (one KV page) from the minimum up that fits, by bisection over a monotone predicate; nullopt
    // when nothing fits.
    using ninferno::runtime::kAutomaticContextGrain;
    using ninferno::runtime::kAutomaticContextMinimum;
    using ninferno::runtime::largest_fitting_context;
    std::uint32_t probes = 0;
    const auto up_to     = [&](std::uint32_t limit) {
        return [&probes, limit](std::uint32_t context) {
            ++probes;
            return context <= limit;
        };
    };
    failures += check(largest_fitting_context(262144, up_to(262144)) == 262144U && probes == 1,
                      "a fitting native context was not kept with one probe");
    probes = 0;
    failures += check(kAutomaticContextGrain == 64U && kAutomaticContextMinimum == 1024U,
                      "the automatic context grain is one KV page from 1024 tokens");
    // 100000 = 1562 pages + 32 tokens: the page below it, not the 1024-token multiple (99328).
    failures += check(largest_fitting_context(262144, up_to(100000)) ==
                              1562U * kAutomaticContextGrain &&
                          probes <= 14,
                      "an automatic context is not the largest fitting page");
    failures += check(largest_fitting_context(40960, up_to(40959)) == 40896U,
                      "an automatic context did not step one page below a native that does not fit");
    // 31B NVFP4 at C=1: 126,976 tokens with 1024-token steps left 52 MiB of 80 KiB per
    // token unused; the pages that fit in it are taken now.
    failures += check(largest_fitting_context(262144, up_to(127656)) == 127616U,
                      "the pages between two 1024-token steps are not taken");
    failures += check(largest_fitting_context(4096, up_to(1024)) == 1024U &&
                          largest_fitting_context(4096, up_to(1087)) == 1024U &&
                          largest_fitting_context(4096, up_to(1088)) == 1088U,
                      "the shortest automatic context was not chosen when only it fits");
    failures += check(!largest_fitting_context(4096, up_to(1023)).has_value() &&
                          !largest_fitting_context(4096, up_to(960)).has_value(),
                      "an automatic context below the minimum was accepted");
    failures += check(largest_fitting_context(1000, up_to(1000)) == 1000U &&
                          !largest_fitting_context(1000, up_to(999)).has_value(),
                      "a native context below the minimum was not kept or refused as a whole");
    failures += check(largest_fitting_context(1030, up_to(1029)) == 1024U &&
                          !largest_fitting_context(1024, up_to(1023)).has_value(),
                      "a native context just above the minimum did not fall back to it");

    // A free-memory shortage is a CapacityError from the resolution through the Engine's fit
    // advice: ninferno-serve's fit and plain fallbacks catch that type, so an invalid_argument in
    // its place would keep a server from starting that fits with less.
    bool shortage_typed = false;
    try {
        (void)ninferno::runtime::resolve_runtime_capacity(
            ninferno::KvCapacityPolicy::explicit_capacity(128), automatic_states, curve_for, 1099);
    } catch (const ninferno::CapacityError&) {
        shortage_typed = true;
    } catch (const std::invalid_argument&) {}
    failures += check(shortage_typed, "a KV shortage was not refused as a CapacityError");
    bool advice_typed = false;
    try {
        ninferno::runtime::throw_capacity_shortage(ninferno::CapacityError("short by 64 bytes"),
                                                 " What fits instead: --max-concurrency 1.");
    } catch (const ninferno::CapacityError& error) {
        advice_typed = std::string(error.what()) ==
                       "short by 64 bytes What fits instead: --max-concurrency 1.";
    } catch (const std::invalid_argument&) {}
    failures += check(advice_typed,
                      "the fit advice did not keep the shortage a CapacityError with its message");
    bool invalid_untyped = false;
    try {
        (void)ninferno::runtime::resolve_kv_capacity(
            ninferno::KvCapacityPolicy::explicit_capacity(0), curve, 10000);
    } catch (const ninferno::CapacityError&) {
    } catch (const std::invalid_argument&) { invalid_untyped = true; }
    failures += check(invalid_untyped, "an invalid KV policy was refused as a CapacityError");

    // What a capacity refusal says fits instead.
    {
        using ninferno::runtime::CapacityAdvice;
        using ninferno::runtime::capacity_advice;
        // Automatic chunks: what fits at the asked concurrency (its asked context) and below.
        CapacityAdvice fits;
        fits.fits                    = {{.max_concurrency = 4, .max_context = 8192},
                                        {.max_concurrency = 2, .max_context = 16384}};
        fits.chunks                  = {1024, 512, 256, 128};
        fits.ceiling                 = 8192;
        fits.max_concurrency         = 4;
        fits.prefill_chunk_automatic = true;
        failures += check(capacity_advice(fits) ==
                              " What fits instead (prefill chunk down to 128): --max-concurrency 4 "
                              "with the asked 8192 tokens of context; --max-concurrency 2 with up "
                              "to 16384 tokens of context.",
                          "the advice lists what fits at each concurrency");
        // An explicit --prefill-chunk 1024 that fits nothing, 512 fits, and an explicit KV.
        CapacityAdvice narrowed;
        narrowed.fits                 = {{.max_concurrency = 4, .max_context = 4096}};
        narrowed.chunks               = {512};
        narrowed.narrowed_from        = 1024;
        narrowed.kv_capacity_explicit = true;
        narrowed.ceiling              = 8192;
        narrowed.max_concurrency      = 4;
        failures += check(capacity_advice(narrowed) ==
                              " --prefill-chunk 1024 fits no concurrency on this device; a "
                              "narrower chunk takes less workspace and StateImage. What fits "
                              "instead (--prefill-chunk 512, an automatic KV capacity in place of "
                              "--kv-capacity): --max-concurrency 4 with up to 4096 tokens of "
                              "context.",
                          "the advice names the narrower chunk that fits");
        // Nothing fits: with an explicit chunk, at any chunk down to the narrowest; the graph
        // allowance of an all-explicit plan is said either way.
        CapacityAdvice nothing;
        nothing.chunks          = {1024};
        nothing.ceiling         = 8192;
        nothing.max_concurrency = 4;
        nothing.graph_allowance = true;
        failures += check(capacity_advice(nothing) ==
                              " No lower --max-concurrency fits either, at any prefill chunk down "
                              "to 128. This plan reserves an allowance for its CUDA graphs "
                              "because every sizing option is explicit; with --prefill-chunk "
                              "left automatic the kernel warm-up measures them instead.",
                          "the advice when nothing fits at any chunk");
        nothing.prefill_chunk_automatic = true;
        nothing.graph_allowance         = false;
        failures += check(capacity_advice(nothing) == " No lower --max-concurrency fits either.",
                          "the advice when nothing fits with an automatic chunk");
        nothing.max_concurrency = 1;
        failures +=
            check(capacity_advice(nothing).empty(), "advice for one slot that fits nothing");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
