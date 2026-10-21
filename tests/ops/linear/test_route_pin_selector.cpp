// The generic path under a maintainer's route pin (--prefill-route, --decode-route;
// ops/route_pins.h), on substituted boards as the selector golden test runs them:
//   - a pin takes every call of its band that its family can compute (pinned_family_route) and
//     leaves every other call on the automatic route, which stays what it was without the pin;
//   - the route a pin gives is of the pinned family;
//   - representative calls take each family (MMQ / W4A8 / I8 / A16 / W4A4 prefill, R8 / vec / MMQ
//     / A16 decode), and the registered shapes' hand-off gates follow the pin (the int8 prefill
//     tiers, the Q4/Q5/Q6 G64 int8 promotion, R8 decode);
//   - setting the pins back to auto restores every automatic route (no state left behind).
// Byte-identical automatic selection is ninferno_selector_golden_test's: it runs with no pin.
//
// The route table and several heuristics cache what they read of the device on first use, so one
// process runs one board: without arguments the test runs itself once per board.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"
#include "ops/route_pins.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using generic::Family;
using ops::LinearPolicy;

struct PinBoard {
    const char* name;
    std::uint32_t pci_device_id;
    int major, minor, sms;
    std::size_t l2_mib, memory_gib;
    int sm_clock_mhz, memory_clock_mhz, bus_bits;
};

constexpr PinBoard kBoards[] = {
    {"rtx5090", 0x2B85, 12, 0, 170, 96, 32, 2407, 14001, 512},
    {"rtx4090", 0x2684, 8, 9, 128, 72, 24, 2520, 10501, 384},
    {"a100", 0x20B0, 8, 0, 108, 40, 40, 1410, 1215, 5120},
    {"rtx2080ti", 0x1E07, 7, 5, 68, 5, 11, 1545, 7000, 352},
    {"v100", 0x1DB5, 7, 0, 80, 6, 32, 1530, 877, 4096},
};

constexpr QType kFormats[] = {QType::BF16,      QType::Q4_G64_FP16, QType::Q8_G32_FP16,
                              QType::NVFP4,     QType::MXFP4,       QType::GGML_Q4_K,
                              QType::GGML_Q6_K, QType::GGML_IQ3_S,  QType::GGML_Q4_0};

struct Shape {
    std::int32_t n, k;
};

constexpr Shape kShapes[]        = {{2048, 1024}, {4096, 4096}, {17408, 5120}};
constexpr std::int32_t kTokens[] = {1, 2, 4, 8, 12, 16, 17, 64, 512, 2048};

constexpr PrefillRoute kPrefill[] = {PrefillRoute::A16, PrefillRoute::W4a8, PrefillRoute::I8,
                                     PrefillRoute::Mmq, PrefillRoute::W4a4};
constexpr DecodeRoute kDecode[]   = {DecodeRoute::A16, DecodeRoute::R8, DecodeRoute::Vec,
                                     DecodeRoute::Mmq};

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    if (g_failures <= 40) std::cerr << "FAIL: " << what << '\n';
}

bool a16_family(Family f) {
    return f == Family::RegisterMma || f == Family::TiledMma || f == Family::StagedMma ||
           f == Family::Nvfp4TiledMma || f == Family::Scalar;
}

bool of_family(PrefillRoute pin, Family f) {
    switch (pin) {
    case PrefillRoute::A16:
        return a16_family(f);
    case PrefillRoute::W4a8:
        return f == Family::W4a8Mma;
    case PrefillRoute::I8:
        return f == Family::I8Mma;
    case PrefillRoute::Mmq:
        return f == Family::GgmlMmq;
    case PrefillRoute::W4a4:
        return f == Family::W4a4Mma;
    case PrefillRoute::Auto:
        return false;
    }
    return false;
}

bool of_family(DecodeRoute pin, Family f) {
    switch (pin) {
    case DecodeRoute::A16:
        return a16_family(f);
    case DecodeRoute::R8:
        return f == Family::R8Mma;
    case DecodeRoute::Vec:
        return f == Family::GgmlVec;
    case DecodeRoute::Mmq:
        return f == Family::GgmlMmq;
    case DecodeRoute::Auto:
        return false;
    }
    return false;
}

struct Case {
    QType qtype;
    LinearPolicy policy;
    generic::Epilogue epilogue;
    Shape shape;
    bool prefill;
    std::int32_t tokens;
};

std::string name_of(const Case& c) {
    std::ostringstream out;
    out << generic::qtype_name(c.qtype) << " p" << static_cast<int>(c.policy) << " "
        << generic::epilogue_name(c.epilogue) << " " << c.shape.n << "x" << c.shape.k
        << (c.prefill ? " prefill" : " speed") << " T=" << c.tokens;
    return out.str();
}

// What `pin` covered on this board: (format, family) pairs that took at least one call.
std::map<std::pair<QType, int>, bool> g_prefill_took, g_decode_took;

int run_board(const PinBoard& b) {
    DeviceClass d      = synthetic_device_class(b.pci_device_id, b.major, b.minor, b.sms,
                                                b.l2_mib * kMiB, b.memory_gib * 1024 * kMiB);
    d.sm_clock_khz     = b.sm_clock_mhz * 1000;
    d.memory_clock_khz = b.memory_clock_mhz * 1000;
    d.memory_bus_bits  = b.bus_bits;
    d.facts            = machine_facts_of(d);
    const ScopedDeviceClass scoped(d);
    ops::set_route_pins({});
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);

    // The automatic routes of every case, before any pin was set.
    struct Item {
        Case c;
        Weight w;
        generic::Problem problem;
        generic::Route automatic;
    };

    std::vector<quantized_weight::PackedWeight> hosts;
    hosts.reserve(std::size(kFormats) * std::size(kShapes));
    std::vector<Item> items;
    for (const QType qtype : kFormats) {
        for (const Shape shape : kShapes) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            try {
                hosts.push_back(
                    quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options));
            } catch (const std::invalid_argument&) { continue; }
            const Weight w = hosts.back().device_weight(payload);
            if (!generic::supports(w)) continue;
            for (const LinearPolicy policy :
                 {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
                if (policy == LinearPolicy::AllowA4 && qtype != QType::NVFP4) continue;
                for (const generic::Epilogue epilogue :
                     {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                    const std::int32_t rows =
                        epilogue == generic::Epilogue::SwiGlu ? shape.n / 2 : shape.n;
                    const generic::Problem problem{qtype, policy, rows, shape.k, epilogue};
                    for (const bool prefill : {false, true}) {
                        for (const std::int32_t t : kTokens) {
                            if (prefill && t < 17) continue;
                            const Case c{qtype, policy, epilogue, shape, prefill, t};
                            items.push_back(
                                {c, w, problem, generic::select_route(problem, w, t, prefill)});
                        }
                    }
                }
            }
        }
    }

    // Each pin: the pinned family where pinned_family_route gives one, else the automatic route.
    const auto sweep = [&](auto pin, auto set, std::map<std::pair<QType, int>, bool>& took) {
        set(pin);
        for (const Item& item : items) {
            const Case& c    = item.c;
            const bool band  = std::is_same_v<decltype(pin), PrefillRoute>
                                   ? c.tokens >= ops::kPrefillPinMinTokens
                                   : c.tokens <= ops::kDecodePinMaxTokens;
            const bool volta = generic::volta_tiled_route(item.w, c.policy, c.tokens).has_value();
            const auto expected =
                band && !volta
                    ? generic::pinned_family_route(pin, item.problem, item.w, c.tokens, c.prefill)
                    : std::nullopt;
            const generic::Route r =
                generic::select_route(item.problem, item.w, c.tokens, c.prefill);
            const std::string what = std::string(b.name) + " " + name_of(c) + " pin ";
            if (expected) {
                check(r == *expected, what + "takes " + generic::route_name(r) + ", not " +
                                          generic::route_name(*expected));
                check(of_family(pin, expected->family),
                      what + "gives " + generic::route_name(*expected) + ", another family");
                took[{c.qtype, static_cast<int>(pin)}] = true;
                // A pin never gives an A16Only weight quantized activations.
                check(c.policy != LinearPolicy::A16Only || a16_family(expected->family),
                      what + "gave an A16Only weight " + generic::route_name(*expected));
            } else {
                check(r == item.automatic,
                      what + "moved a call it cannot take: " + generic::route_name(r) + " for " +
                          generic::route_name(item.automatic));
            }
        }
    };
    for (const PrefillRoute pin : kPrefill) {
        sweep(
            pin, [](PrefillRoute p) { ops::set_route_pins({p, DecodeRoute::Auto, MoePath::Auto}); },
            g_prefill_took);
    }
    for (const DecodeRoute pin : kDecode) {
        sweep(
            pin, [](DecodeRoute p) { ops::set_route_pins({PrefillRoute::Auto, p, MoePath::Auto}); },
            g_decode_took);
    }

    // Back to auto: every automatic route as before.
    ops::set_route_pins({});
    for (const Item& item : items) {
        check(generic::select_route(item.problem, item.w, item.c.tokens, item.c.prefill) ==
                  item.automatic,
              std::string(b.name) + " " + name_of(item.c) + " differs after the pins were cleared");
    }

    // Representative calls take each family where the board runs it.
    const bool tensor = b.major * 10 + b.minor >= 75;
    const auto took_p = [&](QType q, PrefillRoute f) {
        return g_prefill_took.contains({q, static_cast<int>(f)});
    };
    const auto took_d = [&](QType q, DecodeRoute f) {
        return g_decode_took.contains({q, static_cast<int>(f)});
    };
    // Without tensor mma.sync the wide calls of a tiled-codec format keep the tiled route.
    check(!tensor || took_p(QType::Q8_G32_FP16, PrefillRoute::A16),
          std::string(b.name) + ": Q8 prefill a16");
    check(took_d(QType::GGML_Q4_K, DecodeRoute::A16), std::string(b.name) + ": Q4_K decode a16");
    check(took_d(QType::GGML_Q4_K, DecodeRoute::R8), std::string(b.name) + ": Q4_K decode r8");
    check(took_d(QType::GGML_Q4_K, DecodeRoute::Vec), std::string(b.name) + ": Q4_K decode vec");
    if (tensor) {
        check(took_p(QType::GGML_Q4_K, PrefillRoute::Mmq),
              std::string(b.name) + ": Q4_K prefill mmq");
        check(took_p(QType::GGML_Q4_K, PrefillRoute::W4a8),
              std::string(b.name) + ": Q4_K prefill w4a8");
        check(took_p(QType::GGML_Q4_K, PrefillRoute::A16),
              std::string(b.name) + ": Q4_K prefill a16");
        if (b.major >= 8) {
            check(took_p(QType::Q4_G64_FP16, PrefillRoute::I8),
                  std::string(b.name) + ": Q4 G64 prefill i8");
        }
        check(took_d(QType::GGML_IQ3_S, DecodeRoute::Mmq),
              std::string(b.name) + ": IQ3_S verify mmq");
    }
    if (generic::w4a4_schedule_count() > 0 && b.major == 12) {
        check(took_p(QType::NVFP4, PrefillRoute::W4a4),
              std::string(b.name) + ": NVFP4 prefill w4a4");
    }
    // A format a family has no codec for: never taken.

    // The registered shapes' hand-off gates (the wrappers ask them before their own tables).
    {
        quantized_weight::PatternedWeightOptions options;
        const auto q8_host =
            quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 4096, 4096, 7U, options);
        const auto q4_host =
            quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 4096, 4096, 7U, options);
        const Weight q8 = q8_host.device_weight(payload);
        const Weight q4 = q4_host.device_weight(payload);
        const ops::ScopedPrefillRoutes prefill_profile;
        const auto with = [](PrefillRoute p, DecodeRoute dr = DecodeRoute::Auto) {
            ops::set_route_pins({p, dr, MoePath::Auto});
        };
        with(PrefillRoute::Auto);
        const bool q8_auto         = generic::q8_prefill_a8(q8, LinearPolicy::AllowA8, 512);
        const LinearPolicy q4_auto = generic::prefill_a8_policy(q4.qtype, LinearPolicy::A16Only);
        const bool r8_auto         = generic::owned_decode_r8(q8, LinearPolicy::AllowA8, 1);
        with(PrefillRoute::A16);
        check(!generic::q8_prefill_a8(q8, LinearPolicy::AllowA8, 512),
              std::string(b.name) + ": --prefill-route a16 keeps Q8 on its table");
        check(generic::prefill_a8_policy(q4.qtype, LinearPolicy::A16Only) == LinearPolicy::A16Only,
              std::string(b.name) + ": --prefill-route a16 promotes no Q4 G64 weight to int8");
        if (tensor) {
            with(PrefillRoute::W4a8);
            check(generic::prefill_a8_policy(q4.qtype, LinearPolicy::A16Only) ==
                      LinearPolicy::AllowA8,
                  std::string(b.name) + ": --prefill-route w4a8 promotes Q4 G64 on every card");
            check(generic::q8_prefill_a8(q8, LinearPolicy::AllowA8, 512),
                  std::string(b.name) + ": --prefill-route w4a8 takes the Q8 int8 tier");
            with(PrefillRoute::I8);
            check(!generic::q8_prefill_a8(q8, LinearPolicy::AllowA8, 512),
                  std::string(b.name) + ": --prefill-route i8 has no Q8 tier (I8 reads G64)");
            check(generic::q8_prefill_a8(q4, LinearPolicy::AllowA8, 512),
                  std::string(b.name) + ": --prefill-route i8 takes the Q4 G64 tier");
            check(!generic::q8_prefill_a8(q8, LinearPolicy::A16Only, 512),
                  std::string(b.name) +
                      ": a pin never gives an A16Only Q8 weight int8 activations");
        }
        with(PrefillRoute::Auto, DecodeRoute::A16);
        check(!generic::owned_decode_r8(q8, LinearPolicy::AllowA8, 1),
              std::string(b.name) + ": --decode-route a16 keeps a decode call on its table");
        with(PrefillRoute::Auto, DecodeRoute::R8);
        check(generic::owned_decode_r8(q8, LinearPolicy::AllowA8, 1) == r8_auto,
              std::string(b.name) + ": --decode-route r8 hands decode as the rule does");
        with(PrefillRoute::Auto);
        check(generic::q8_prefill_a8(q8, LinearPolicy::AllowA8, 512) == q8_auto &&
                  generic::prefill_a8_policy(q4.qtype, LinearPolicy::A16Only) == q4_auto &&
                  generic::owned_decode_r8(q8, LinearPolicy::AllowA8, 1) == r8_auto,
              std::string(b.name) + ": the gates are the rule's again without a pin");
    }

    // The ledger: a pinned call notes its band, what took it and the families that could.
    ops::set_route_pins({PrefillRoute::Mmq, DecodeRoute::Auto, MoePath::Auto});
    for (const Item& item : items) {
        if (item.c.policy == LinearPolicy::A16Only && item.c.tokens >= 17) {
            (void)generic::select_route(item.problem, item.w, item.c.tokens, item.c.prefill);
        }
    }
    const ops::RoutePinLedger a16_only = ops::route_pin_ledger();
    check(a16_only.prefill.seen && !a16_only.prefill.took &&
              (a16_only.prefill.could & ops::family_bit(PrefillRoute::A16)) != 0 &&
              (a16_only.prefill.could & ops::family_bit(PrefillRoute::Mmq)) == 0,
          std::string(b.name) + ": an A16Only model's ledger under --prefill-route mmq");
    bool refused = false;
    try {
        (void)ops::check_route_pin_coverage(ops::route_pins());
    } catch (const std::invalid_argument& error) {
        refused = std::string_view(error.what()).find("a16") != std::string_view::npos;
    }
    check(refused, std::string(b.name) +
                       ": --prefill-route mmq on A16Only weights alone is refused naming a16");
    ops::set_route_pins({});

    if (g_failures != 0) {
        std::cerr << b.name << ": " << g_failures << " failures\n";
        return 1;
    }
    std::cout << b.name << ": " << items.size() << " cases, every pin as specified\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string board;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--board" && i + 1 < argc) board = argv[++i];
        }
        if (!board.empty()) {
            for (const PinBoard& b : kBoards) {
                if (board == b.name) return run_board(b);
            }
            std::cerr << "unknown board " << board << '\n';
            return 2;
        }
        int failed = 0;
        for (const PinBoard& b : kBoards) {
            const std::string command = std::string(argv[0]) + " --board " + b.name;
            if (std::system(command.c_str()) != 0) ++failed;
        }
        if (failed != 0) {
            std::cerr << "route pin selector: " << failed << " boards failed\n";
            return 1;
        }
        std::cout << "route pin selector: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "route pin selector: " << error.what() << '\n';
        return 1;
    }
}
