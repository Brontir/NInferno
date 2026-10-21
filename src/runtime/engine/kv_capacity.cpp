#include "runtime/engine/kv_capacity.h"

#include "core/checked.h"
#include "core/paged_kv_cache.h"
#include "core/units.h"
#include "ninferno/types.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>

namespace ninferno::runtime {

static_assert(kAutomaticContextGrain == static_cast<std::uint32_t>(kPagedKVPageSize),
              "an automatic context steps by one paged-KV page");
static_assert(kAutomaticContextMinimum % kAutomaticContextGrain == 0,
              "the shortest automatic context is a whole number of pages");

namespace {

void validate_curve(const SequenceCapacityCurve& curve) {
    if (curve.main_page_tokens == 0 || curve.minimum_main_page_groups == 0 ||
        curve.minimum_main_page_groups > curve.maximum_main_page_groups ||
        curve.minimum_device_reservation_bytes == 0) {
        throw std::invalid_argument("sequence capacity curve is invalid");
    }
    if (curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable sequence capacity curve has zero byte stride");
    }
}

} // namespace

std::size_t SequenceCapacityCurve::reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(minimum_device_reservation_bytes,
                       checked_mul(additional, bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::uint32_t SequenceCapacityCurve::resolved_tokens(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::uint64_t tokens = static_cast<std::uint64_t>(main_page_groups) * main_page_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("resolved KV token capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

std::string capacity_advice(const CapacityAdvice& advice) {
    std::string basis;
    if (!advice.chunks.empty()) {
        basis = advice.chunks.size() > 1
                    ? "prefill chunk down to " + std::to_string(advice.chunks.back())
                    : "--prefill-chunk " + std::to_string(advice.chunks.back());
    }
    if (advice.kv_capacity_explicit) {
        basis += ", an automatic KV capacity in place of --kv-capacity";
    }
    std::string instead;
    for (const ConcurrencyFit& fit : advice.fits) {
        instead += (instead.empty() ? " What fits instead (" + basis + "): " : std::string("; ")) +
                   "--max-concurrency " + std::to_string(fit.max_concurrency) +
                   (fit.max_context == advice.ceiling && !advice.context_automatic
                        ? " with the asked "
                        : " with up to ") +
                   std::to_string(fit.max_context) + " tokens of context";
    }
    if (!instead.empty()) {
        const std::string narrowed =
            advice.narrowed_from == 0
                ? std::string()
                : " --prefill-chunk " + std::to_string(advice.narrowed_from) +
                      " fits no concurrency on this device; a narrower chunk takes less "
                      "workspace and StateImage.";
        instead = narrowed + instead + ".";
    }
    if (advice.fits.empty() && advice.max_concurrency > 1) {
        instead = advice.prefill_chunk_automatic
                      ? " No lower --max-concurrency fits either."
                      : " No lower --max-concurrency fits either, at any prefill chunk down to " +
                            std::to_string(kNarrowestAutomaticPrefillChunk) + ".";
    }
    if (advice.graph_allowance) {
        instead += " This plan reserves an allowance for its CUDA graphs because every sizing "
                   "option is explicit; with --prefill-chunk left automatic the kernel warm-up "
                   "measures them instead.";
    }
    return instead;
}

void throw_capacity_shortage(const CapacityError& refusal, const std::string& advice) {
    throw CapacityError(refusal.what() + advice);
}

} // namespace ninferno::runtime
