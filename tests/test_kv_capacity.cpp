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

    // The curve: a reservation affine in the pages, refused outside its bounds.
    failures += check(curve.reservation_bytes(2) == 1000 && curve.reservation_bytes(6) == 1512 &&
                          curve.resolved_tokens(4) == 256,
                      "the capacity curve is not affine in its pages");
    bool outside = false;
    try {
        (void)curve.reservation_bytes(7);
    } catch (const std::invalid_argument&) { outside = true; }
    failures += check(outside, "a page count above the curve was accepted");
    using ninferno::runtime::kAutomaticContextGrain;
    using ninferno::runtime::kAutomaticContextMinimum;
    failures += check(kAutomaticContextGrain == 64U && kAutomaticContextMinimum == 1024U,
                      "the ladder's grain is one KV page, its survival minimum 1024 tokens");

    // A free-memory shortage stays a CapacityError with the advice after its message:
    // ninferno-serve's fit and plain fallbacks catch that type.
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
