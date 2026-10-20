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

KvCapacityPolicy with_paravirtual_headroom(KvCapacityPolicy policy, bool paravirtual,
                                           bool drives_display, std::size_t device_total_bytes) {
    if (policy.mode != KvCapacityMode::Automatic || !paravirtual) { return policy; }
    const std::size_t share = drives_display
                                  ? std::min<std::size_t>(device_total_bytes / 8, 4 * kGiB)
                                  : std::min<std::size_t>(device_total_bytes / 32, kGiB);
    if (share > policy.automatic_headroom_bytes) {
        policy.best_effort_headroom_bytes =
            std::max(policy.best_effort_headroom_bytes, share - policy.automatic_headroom_bytes);
    }
    return policy;
}

KvCapacityPolicy with_unified_memory_headroom(KvCapacityPolicy policy, bool integrated,
                                              std::size_t device_total_bytes) {
    if (policy.mode != KvCapacityMode::Automatic || !integrated) { return policy; }
    const std::size_t share = std::max<std::size_t>(device_total_bytes / 8, 4 * kGiB);
    if (share > policy.automatic_headroom_bytes) {
        policy.best_effort_headroom_bytes =
            std::max(policy.best_effort_headroom_bytes, share - policy.automatic_headroom_bytes);
    }
    return policy;
}

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

std::uint32_t explicit_page_groups(const KvCapacityPolicy& policy,
                                   const SequenceCapacityCurve& curve) {
    if (policy.explicit_tokens == 0) {
        throw std::invalid_argument("explicit KV capacity must be positive");
    }
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) / curve.main_page_tokens;
    if (pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages);
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

namespace {

// The resolution, or the message of a memory shortage. Invalid policies and curves still throw.
std::variant<KvCapacityResolution, std::string>
try_resolve_kv_capacity(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                        std::size_t available_runtime_bytes) {
    validate_curve(curve);

    std::uint32_t pages         = curve.minimum_main_page_groups;
    std::size_t capacity_budget = available_runtime_bytes;
    std::size_t best_effort     = 0;
    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.automatic_headroom_bytes != 0 || policy.best_effort_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
        pages = explicit_page_groups(policy, curve);
        break;
    case KvCapacityMode::Automatic:
        if (available_runtime_bytes < policy.automatic_headroom_bytes) {
            return "automatic KV headroom requires " +
                   std::to_string(policy.automatic_headroom_bytes) + " bytes, but only " +
                   std::to_string(available_runtime_bytes) + " bytes are available after weights";
        }
        capacity_budget -= policy.automatic_headroom_bytes;
        if (capacity_budget < curve.minimum_device_reservation_bytes) {
            return std::string(
                "minimum Engine runtime reservation requires ") +
                std::to_string(curve.minimum_device_reservation_bytes) + " bytes in addition to " +
                std::to_string(policy.automatic_headroom_bytes) +
                " bytes of automatic headroom, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights";
        }
        // The best-effort headroom from the pages above the minimum reservation only.
        best_effort = std::min(policy.best_effort_headroom_bytes,
                               capacity_budget - curve.minimum_device_reservation_bytes);
        capacity_budget -= best_effort;
        if (curve.minimum_main_page_groups < curve.maximum_main_page_groups) {
            const std::size_t additional =
                (capacity_budget - curve.minimum_device_reservation_bytes) /
                curve.bytes_per_additional_main_page_group;
            const std::uint64_t candidate =
                static_cast<std::uint64_t>(curve.minimum_main_page_groups) + additional;
            pages = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(candidate, curve.maximum_main_page_groups));
        }
        break;
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    const std::size_t reservation = curve.reservation_bytes(pages);
    if (reservation > capacity_budget) {
        return "requested Engine runtime reservation requires " + std::to_string(reservation) +
               " bytes, but only " + std::to_string(capacity_budget) +
               " bytes are available for runtime capacity";
    }

    return KvCapacityResolution{
        .mode                                 = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                      = curve.resolved_tokens(pages),
        .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes            = reservation,
        .available_after_weights_bytes        = available_runtime_bytes,
        .automatic_headroom_bytes             = policy.automatic_headroom_bytes + best_effort,
        .planned_slack_bytes                  = available_runtime_bytes - reservation,
    };
}

} // namespace

KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                         const SequenceCapacityCurve& curve,
                                         std::size_t available_runtime_bytes) {
    auto result = try_resolve_kv_capacity(policy, curve, available_runtime_bytes);
    if (auto* shortage = std::get_if<std::string>(&result)) {
        throw CapacityError(*shortage);
    }
    return std::get<KvCapacityResolution>(std::move(result));
}

namespace {

// The resolution, or the refusal's message when the free bytes are short.
std::variant<KvCapacityResolution, std::string>
resolve_runtime_capacity_or_shortage(
    const KvCapacityPolicy& policy, const CachedStatePolicy& states,
    const std::function<SequenceCapacityCurve(std::uint32_t)>& curve_for,
    std::size_t available_runtime_bytes) {
    const auto with_states = [&](KvCapacityResolution resolution, std::uint32_t count) {
        resolution.cached_device_states           = count;
        resolution.cached_device_states_default   = states.configured.value_or(
            states.automatic_default);
        resolution.cached_device_states_automatic = !states.configured.has_value();
        return resolution;
    };
    if (states.configured) {
        auto result =
            try_resolve_kv_capacity(policy, curve_for(*states.configured), available_runtime_bytes);
        if (auto* resolution = std::get_if<KvCapacityResolution>(&result)) {
            return with_states(std::move(*resolution), *states.configured);
        }
        return result;
    }
    if (states.automatic_default == 0) {
        throw std::invalid_argument("automatic cached Device state count has no default");
    }
    if (states.automatic_minimum > 1 || states.automatic_minimum > states.automatic_default) {
        throw std::invalid_argument("automatic cached Device state minimum must be 0 or 1");
    }
    std::string smallest_shortage;
    if (policy.mode == KvCapacityMode::Automatic) {
        // The pool comes before the cached states, one whole slot context at a time: the largest
        // count whose pool holds as many whole contexts (the curve's minimum: one slot's context)
        // as the fewest states' pool, the largest. States are kept while the pool keeps its whole
        // contexts (C-1 states beside 99% of the largest pool keep their prefix reuse); the pool
        // that holds every slot's context (the curve's maximum) is one such count. Near a whole
        // context the choice jumps: when the fewest states' pool just reaches one more context,
        // even a small state would cost it, so a little more free memory can mean fewer states
        // (10-page contexts, 5-byte states, 10-byte pages: 1150 bytes keep 3 states beside 18
        // pages, 1155 bytes 1 state beside 20). Fewer states never hold less KV, so the smallest
        // count decides whether any fits.
        const auto whole_contexts = [](const KvCapacityResolution& resolution,
                                       const SequenceCapacityCurve& curve) {
            return resolution.main_page_groups / curve.minimum_main_page_groups;
        };
        const SequenceCapacityCurve fewest_curve = curve_for(states.automatic_minimum);

        auto fewest = try_resolve_kv_capacity(policy, fewest_curve, available_runtime_bytes);
        if (auto* resolution = std::get_if<KvCapacityResolution>(&fewest)) {
            const std::uint32_t wanted = whole_contexts(*resolution, fewest_curve);
            for (std::uint32_t count = states.automatic_default; count > states.automatic_minimum;
                 --count) {
                const SequenceCapacityCurve curve = curve_for(count);

                auto result = try_resolve_kv_capacity(policy, curve, available_runtime_bytes);
                auto* more  = std::get_if<KvCapacityResolution>(&result);
                // The best-effort headroom comes before the states too: a count whose states
                // take it back is not kept (a GPU Windows shares filled to 194 MiB with 4 states).
                if (more != nullptr && whole_contexts(*more, curve) >= wanted &&
                    more->automatic_headroom_bytes >= resolution->automatic_headroom_bytes) {
                    return with_states(std::move(*more), count);
                }
            }
            return with_states(std::move(*resolution), states.automatic_minimum);
        }
        smallest_shortage = std::get<std::string>(std::move(fewest));
    } else {
        for (std::uint32_t count = states.automatic_default;; --count) {
            auto result =
                try_resolve_kv_capacity(policy, curve_for(count), available_runtime_bytes);
            if (auto* resolution = std::get_if<KvCapacityResolution>(&result)) {
                return with_states(std::move(*resolution), count);
            }
            smallest_shortage = std::get<std::string>(std::move(result));
            if (count == states.automatic_minimum) { break; }
        }
    }
    if (states.automatic_minimum == 0) {
        return "with no cached Device state (the smallest automatic choice: the Host tier keeps "
               "the context cache's checkpoints; default " +
               std::to_string(states.automatic_default) + "), " + smallest_shortage +
               ". A smaller max_context needs less.";
    }
    return "the context cache needs at least one cached Device state beside the active ones; with "
           "one (the smallest automatic choice, default " +
           std::to_string(states.automatic_default) + "), " + smallest_shortage +
           ". A smaller max_context, a Host tier (then no cached Device state is needed) or a "
           "disabled context cache needs less.";
}

} // namespace

KvCapacityResolution
resolve_runtime_capacity(const KvCapacityPolicy& policy, const CachedStatePolicy& states,
                         const std::function<SequenceCapacityCurve(std::uint32_t)>& curve_for,
                         std::size_t available_runtime_bytes) {
    auto result =
        resolve_runtime_capacity_or_shortage(policy, states, curve_for, available_runtime_bytes);
    if (auto* shortage = std::get_if<std::string>(&result)) {
        throw CapacityError(*shortage);
    }
    return std::get<KvCapacityResolution>(std::move(result));
}

std::optional<KvCapacityResolution>
try_resolve_runtime_capacity(const KvCapacityPolicy& policy, const CachedStatePolicy& states,
                             const std::function<SequenceCapacityCurve(std::uint32_t)>& curve_for,
                             std::size_t available_runtime_bytes) {
    auto result =
        resolve_runtime_capacity_or_shortage(policy, states, curve_for, available_runtime_bytes);
    if (auto* resolution = std::get_if<KvCapacityResolution>(&result)) {
        return std::move(*resolution);
    }
    return std::nullopt;
}

std::optional<std::uint32_t>
largest_fitting_context(std::uint32_t ceiling, const std::function<bool(std::uint32_t)>& fits) {
    if (ceiling == 0) { throw std::invalid_argument("automatic max_context has no ceiling"); }
    if (fits(ceiling)) { return ceiling; }
    // Candidates below the ceiling: k * grain for k in [minimum / grain, (ceiling - 1) / grain].
    constexpr std::uint32_t grain = kAutomaticContextGrain;
    const std::uint32_t lowest    = kAutomaticContextMinimum / grain;
    const std::uint32_t highest   = (ceiling - 1U) / grain;
    if (highest < lowest) { return std::nullopt; }
    if (!fits(lowest * grain)) { return std::nullopt; }
    std::uint32_t fitting = lowest;      // fits(fitting * grain)
    std::uint32_t refused = highest + 1; // refused * grain >= ceiling, which does not fit
    while (refused - fitting > 1) {
        const std::uint32_t middle = fitting + (refused - fitting) / 2U;
        if (fits(middle * grain)) {
            fitting = middle;
        } else {
            refused = middle;
        }
    }
    return fitting * grain;
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
