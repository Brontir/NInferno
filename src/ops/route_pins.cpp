#include "ops/route_pins.h"

#include <atomic>
#include <stdexcept>

namespace ninferno::ops {
namespace {

// Written by set_route_pins before a model loads, read on every route resolution: relaxed
// atomics, one byte each.
std::atomic<std::uint8_t> g_prefill{static_cast<std::uint8_t>(PrefillRoute::Auto)};
std::atomic<std::uint8_t> g_decode{static_cast<std::uint8_t>(DecodeRoute::Auto)};
std::atomic<std::uint8_t> g_moe{static_cast<std::uint8_t>(MoePath::Auto)};

struct AtomicBand {
    std::atomic<bool> seen{false};
    std::atomic<bool> took{false};
    std::atomic<std::uint32_t> could{0};

    void clear() noexcept {
        seen.store(false, std::memory_order_relaxed);
        took.store(false, std::memory_order_relaxed);
        could.store(0, std::memory_order_relaxed);
    }

    void note(bool t, std::uint32_t c) noexcept {
        seen.store(true, std::memory_order_relaxed);
        if (t) { took.store(true, std::memory_order_relaxed); }
        could.fetch_or(c, std::memory_order_relaxed);
    }

    [[nodiscard]] PinBand load() const noexcept {
        return {seen.load(std::memory_order_relaxed), took.load(std::memory_order_relaxed),
                could.load(std::memory_order_relaxed)};
    }
};

AtomicBand g_prefill_band;
AtomicBand g_decode_band;
std::atomic<std::uint32_t> g_moe_choices{0};
std::atomic<std::uint32_t> g_moe_fused_ready{0};

template <class Family, std::size_t N>
std::string families(std::uint32_t mask, const Family (&all)[N], const char* (*name)(Family)) {
    std::string out;
    for (const Family f : all) {
        if ((mask & (1U << static_cast<unsigned>(f))) == 0) { continue; }
        if (!out.empty()) { out += ", "; }
        out += name(f);
    }
    return out.empty() ? std::string("none") : out;
}

const char* prefill_name(PrefillRoute f) { return prefill_route_name(f); }

const char* decode_name(DecodeRoute f) { return decode_route_name(f); }

constexpr PrefillRoute kPrefillFamilies[] = {
    PrefillRoute::A16, PrefillRoute::W4a8, PrefillRoute::I8, PrefillRoute::Mmq, PrefillRoute::W4a4};
constexpr DecodeRoute kDecodeFamilies[] = {DecodeRoute::A16, DecodeRoute::R8, DecodeRoute::Vec,
                                           DecodeRoute::Mmq};

} // namespace

void set_route_pins(const RoutePins& pins) noexcept {
    g_prefill.store(static_cast<std::uint8_t>(pins.prefill), std::memory_order_relaxed);
    g_decode.store(static_cast<std::uint8_t>(pins.decode), std::memory_order_relaxed);
    g_moe.store(static_cast<std::uint8_t>(pins.moe), std::memory_order_relaxed);
    g_prefill_band.clear();
    g_decode_band.clear();
    g_moe_choices.store(0, std::memory_order_relaxed);
    g_moe_fused_ready.store(0, std::memory_order_relaxed);
}

RoutePins route_pins() noexcept {
    return {static_cast<PrefillRoute>(g_prefill.load(std::memory_order_relaxed)),
            static_cast<DecodeRoute>(g_decode.load(std::memory_order_relaxed)),
            static_cast<MoePath>(g_moe.load(std::memory_order_relaxed))};
}

void note_prefill_pin(bool took, std::uint32_t could) noexcept { g_prefill_band.note(took, could); }

void note_decode_pin(bool took, std::uint32_t could) noexcept { g_decode_band.note(took, could); }

void note_moe_choice(bool fused_ready) noexcept {
    g_moe_choices.fetch_add(1, std::memory_order_relaxed);
    if (fused_ready) { g_moe_fused_ready.fetch_add(1, std::memory_order_relaxed); }
}

RoutePinLedger route_pin_ledger() noexcept {
    return {g_prefill_band.load(), g_decode_band.load(),
            g_moe_choices.load(std::memory_order_relaxed),
            g_moe_fused_ready.load(std::memory_order_relaxed)};
}

bool moe_takes_fused(bool fused_ready, const std::string& why_not) {
    note_moe_choice(fused_ready);
    switch (route_pins().moe) {
    case MoePath::Auto:
        return fused_ready;
    case MoePath::Generic:
        return false;
    case MoePath::Fused:
        if (!fused_ready) {
            throw std::invalid_argument("--moe-path fused: the fused MoE kernels cannot run this "
                                        "layer (" +
                                        why_not +
                                        "); what runs it is the generic MoE (--moe-path generic "
                                        "or auto)");
        }
        return true;
    }
    return fused_ready;
}

std::optional<bool> pinned_shape_quantized(std::int32_t tokens,
                                           std::optional<PrefillRoute> quantized) noexcept {
    const RoutePins pins = route_pins();
    if (tokens >= kPrefillPinMinTokens) {
        if (pins.prefill == PrefillRoute::A16) { return false; }
        if (quantized && pins.prefill == *quantized) { return true; }
        return std::nullopt;
    }
    if (tokens >= 1 && pins.decode == DecodeRoute::A16) { return false; }
    return std::nullopt;
}

void note_shape_activations(std::int32_t tokens, std::optional<PrefillRoute> family,
                            bool quantized) noexcept {
    const RoutePins pins = route_pins();
    if (tokens >= kPrefillPinMinTokens) {
        if (pins.prefill == PrefillRoute::Auto) { return; }
        const std::uint32_t could =
            family_bit(PrefillRoute::A16) | (family ? family_bit(*family) : 0U);
        const bool took =
            quantized ? family && pins.prefill == *family : pins.prefill == PrefillRoute::A16;
        note_prefill_pin(took, could);
    } else if (tokens >= 1 && pins.decode != DecodeRoute::Auto) {
        note_decode_pin(!quantized && pins.decode == DecodeRoute::A16,
                        family_bit(DecodeRoute::A16));
    }
}

std::string prefill_families(std::uint32_t mask) {
    return families(mask, kPrefillFamilies, &prefill_name);
}

std::string decode_families(std::uint32_t mask) {
    return families(mask, kDecodeFamilies, &decode_name);
}

std::vector<std::string> check_route_pin_coverage(const RoutePins& pins) {
    const RoutePinLedger ledger = route_pin_ledger();
    std::vector<std::string> report;
    const auto band = [&](const char* option, const char* value, const char* widths,
                          const PinBand& b, const std::string& could) {
        if (!b.seen) {
            report.push_back(std::string(option) + " " + value + ": no " + widths +
                             " projection call was resolved at startup (none checked yet)");
            return;
        }
        if (!b.took) {
            throw std::invalid_argument(
                std::string(option) + " " + value + ": the " + widths +
                " projections of this model on this GPU cannot take it (format, K, alignment, "
                "width or --activations); what can run them: " +
                could);
        }
        report.push_back(std::string(option) + " " + value + ": taken where it runs; the " +
                         widths + " calls met can take " + could +
                         " (a call it cannot run keeps the automatic route)");
    };
    if (pins.prefill != PrefillRoute::Auto) {
        band("--prefill-route", prefill_route_name(pins.prefill), "prefill-width (17+ tokens)",
             ledger.prefill, prefill_families(ledger.prefill.could));
    }
    if (pins.decode != DecodeRoute::Auto) {
        band("--decode-route", decode_route_name(pins.decode), "decode-width (1..16 tokens)",
             ledger.decode, decode_families(ledger.decode.could));
    }
    if (pins.moe != MoePath::Auto) {
        if (ledger.moe_choices == 0) {
            throw std::invalid_argument(
                std::string("--moe-path ") + moe_path_name(pins.moe) +
                ": this model has no MoE layer that chooses between fused kernels and the generic "
                "MoE (no routed experts, or its family has only the generic MoE Op): nothing to "
                "pin");
        }
        report.push_back(std::string("--moe-path ") + moe_path_name(pins.moe) + ": " +
                         std::to_string(ledger.moe_choices) + " MoE layers (" +
                         std::to_string(ledger.moe_fused_ready) + " could take the fused kernels)");
    }
    return report;
}

} // namespace ninferno::ops
