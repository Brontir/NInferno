// Route selection on GPU classes this machine does not have (code review 30.09, item 5). For each
// board below a substituted DeviceClass (core/device_class.h) stands in for the device while the
// generic path chooses routes for every format it reads, each policy, three epilogues, three shapes
// and decode, verify and prefill widths. Every choice must be one route_admits accepts and must
// need no feature the board lacks (route_requires); a route whose kernels this build lacks
// (W4A4 and the NVFP4 tiled routes outside 120a) is never chosen, whatever the board; and the
// route table in effect is the nearest measured board's of its instruction set (board_distance,
// route_table.cpp), the board's own where routes/ has it. The register-MMA family's coverage
// (#307): route_admits takes a schedule exactly where rmma_refusal names no reason, a cluster
// schedule is not compiled outside the full build and needs clusters on the board, the 32-value
// decoders' large tiles are not compiled, and the family's own pick, where it does not run, says
// why (counted, not failed: the selection then takes another tile). The R8 family's too: each
// schedule's refusal as route_admits decides it, an A16 call refused for its policy, and why each
// call does or does not take R8 (counted).
//
// The route table and several heuristics cache what they read of the device on first use, so one
// process runs one board: without arguments the test runs itself once per board.
#include "ops/direct_bf16_weight.h"
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/route_table.h"
#include "ops/common/native_nvfp4.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

struct TestBoard {
    const char* name;
    std::uint32_t pci_device_id; // 0: unknown to the id table, the chip follows from the SM count
    int major, minor, sms;
    std::size_t l2_mib, memory_gib;
    const char* table; // the class of the nearest route table in routes/ ("" when none is near)
};

constexpr TestBoard kBoards[] = {
    {"rtx5090", 0x2B85, 12, 0, 170, 96, 32, "gb202"},
    {"rtx5080", 0x2C02, 12, 0, 84, 64, 16, "gb203"},
    {"rtx4090", 0x2684, 8, 9, 128, 72, 24, "ad102"},
    {"rtx4070", 0x2786, 8, 9, 46, 36, 12, "ad106"}, // no ad104 table: the RTX 4060 Ti's, 0.5 away
    {"rtx3090", 0x2204, 8, 6, 82, 6, 24, "ga102"},
    {"rtx2080ti", 0x1E07, 7, 5, 68, 5, 11, "tu102"},
    {"unknown-sm120-48sm", 0, 12, 0, 48, 32, 12, "gb206"}, // the RTX 5060 Ti's, the nearest
};

constexpr QType kFormats[] = {
    QType::Q4_G64_FP16,
    QType::Q5_G64_FP16,
    QType::Q6_G64_FP16,
    QType::Q8_G32_FP16,
    QType::FP8_E4M3FN_ROW_BF16,
    QType::FP8_E4M3FN_ROW_F32,
    QType::FP8_E4M3FN_B128_F32,
    QType::NVFP4,
    QType::MXFP4,
    QType::U4_G128_ZP_FP16,
    QType::GGML_Q4_K,
    QType::GGML_Q5_K,
    QType::GGML_Q6_K,
    QType::GGML_Q3_K,
    QType::GGML_Q2_K,
    QType::GGML_IQ4_XS,
    QType::GGML_IQ3_XXS,
    QType::GGML_IQ3_S,
    QType::GGML_IQ2_XXS,
    QType::GGML_IQ2_XS,
    QType::GGML_IQ2_S,
    QType::GGML_Q4_0,
    QType::GGML_Q5_0,
    QType::GGML_IQ4_NL,
};

struct Shape {
    std::int32_t n, k;
};

bool aligned(const void* p, std::uintptr_t bytes) {
    return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % bytes) == 0;
}

// W4A8's admission as route_admits decided it before w4a8_refusal, written apart from it: the
// generic path reads the weight, the board has the tile's instructions, a producer-warp tile reads
// a ggml high plane on 16 bytes, the tile's shared memory fits, A8 allowed, a W4A8 codec, a K of
// whole W4A8 groups, 16-byte codes, NVFP4 under its prefill A8 policy, and a producer-warp tile
// built for the format (the list below) with an IQ3_XXS sign plane on 8 bytes.
bool w4a8_old_rule(int index, const Weight& w, LinearPolicy policy,
                   const DeviceFeatures& features) {
    const generic::Route r{generic::Family::W4a8Mma, static_cast<std::uint8_t>(index)};
    const bool producer = generic::w4a8_schedule_info(index).producers > 0;
    const QType q       = w.qtype;
    const bool ws_built = q == QType::GGML_IQ3_XXS || q == QType::GGML_IQ2_S ||
                          q == QType::GGML_Q4_K || q == QType::GGML_Q6_K ||
                          q == QType::GGML_Q4_0 || q == QType::GGML_IQ4_XS ||
                          q == QType::GGML_IQ2_XS || q == QType::GGML_IQ2_XXS ||
                          q == QType::GGML_IQ3_S;
    const bool high_off =
        w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned(w.qhigh, 16);
    return generic::supports(w) && has_features(features, generic::route_requires(r)) &&
           !(producer && high_off) && generic::w4a8_fits(index, q) && allows_a8(policy) &&
           generic::w4a8_supports(q) && generic::w4a8_k_admits(q, w.k) && aligned(w.qdata, 16) &&
           (q != QType::NVFP4 || generic::nvfp4_prefill_a8_policy(q, policy)) &&
           (!producer || (ws_built && (q != QType::GGML_IQ3_XXS || aligned(w.qhigh, 8))));
}

// The tiled family's admission as route_admits decided it before tiled_refusal, written apart from
// it: the generic path reads the weight, the board has the tile's instructions, a ggml high plane
// on 16 bytes; a CUTLASS-only format (no tiled tile for its codec, on Volta) as schedule 0 with K
// in 64-value steps; any other a codec the tiled kernel takes (Q8 too on Volta), the tile's shared
// memory, K in 64-value steps and 16-byte codes.
bool tiled_old_rule(int index, const Weight& w, const DeviceFeatures& features) {
    const generic::Route r{generic::Family::TiledMma, static_cast<std::uint8_t>(index)};
    if (!generic::supports(w) || !has_features(features, generic::route_requires(r))) return false;
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned(w.qhigh, 16)) {
        return false;
    }
    const auto codec = generic::codec_of(w);
    const bool takes = generic::tiled_supports(codec) ||
                       (codec == generic::WeightCodec::Q8 && !features.mma_tensor);
    if (!takes && !features.mma_tensor) return index == 0 && (w.k % 64) == 0;
    return index < generic::tiled_schedule_count() && takes && generic::tiled_fits(index, codec) &&
           (w.k % 64) == 0 && aligned(w.qdata, 16);
}

// The tiled family's coverage of `w` (#307): each schedule's refusal as route_admits decides it
// and as the old rule did; the counts; a ggml high plane 4 bytes off refuses every schedule that
// admitted the weight for alignment. Counts the refusals in `counts`.
template <class Fail>
void check_tiled(const Weight& w, const DeviceFeatures& features, const std::string& where,
                 std::map<std::string, int>& counts, Fail&& fail) {
    const auto c = generic::tiled_coverage(w);
    int legal    = 0;
    for (const auto& s : c.schedules) {
        const generic::Route r{generic::Family::TiledMma, static_cast<std::uint8_t>(s.schedule)};
        const bool admits = generic::route_admits(r, w, LinearPolicy::A16Only);
        ++counts[std::string(generic::schedule_refusal_name(s.refusal))];
        legal += s.refusal == generic::ScheduleRefusal::None;
        if (admits != (s.refusal == generic::ScheduleRefusal::None) ||
            admits != tiled_old_rule(s.schedule, w, features)) {
            fail(where + " " + generic::route_name(r) + ": " +
                 std::string(generic::schedule_refusal_name(s.refusal)) +
                 ", which route_admits or the old rule decides otherwise");
        }
    }
    if (c.declared != generic::tiled_schedule_count() || c.legal != legal ||
        (c.compiled != 0 && c.compiled != 1 && c.compiled != c.declared) ||
        (c.heuristic >= 0 && c.heuristic_refusal != c.schedules[c.heuristic].refusal)) {
        fail(where + ": tiled counts declared " + std::to_string(c.declared) + ", compiled " +
             std::to_string(c.compiled) + ", legal " + std::to_string(c.legal));
    }
    if (c.heuristic >= 0 && c.heuristic_refusal != generic::ScheduleRefusal::None) {
        ++counts["pick " + std::string(generic::schedule_refusal_name(c.heuristic_refusal))];
    }
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && legal > 0) {
        Weight off = w;
        off.qhigh  = static_cast<const char*>(w.qhigh) + 4;
        const auto o = generic::tiled_coverage(off);
        for (std::size_t i = 0; i < c.schedules.size(); ++i) {
            if (c.schedules[i].refusal == generic::ScheduleRefusal::None &&
                o.schedules[i].refusal != generic::ScheduleRefusal::Alignment) {
                fail(where + ": tiled with the high plane 4 bytes off");
            }
        }
        if (o.compiled != c.compiled) fail(where + ": tiled compiled count moved with the plane");
    }
    // A layout the generic path does not read: every schedule refused for format, and what the
    // build holds (compiled, the pick) unchanged.
    Weight bad = w;
    bad.layout =
        w.layout == QuantLayout::GgmlPlanes ? QuantLayout::RowSplit : QuantLayout::GgmlPlanes;
    if (!generic::supports(bad)) {
        const auto b          = generic::tiled_coverage(bad);
        const bool all_format = std::all_of(
            b.schedules.begin(), b.schedules.end(),
            [](const auto& s) { return s.refusal == generic::ScheduleRefusal::Format; });
        if (!all_format || b.compiled != c.compiled || b.heuristic != c.heuristic) {
            fail(where + ": tiled in a layout the generic path does not read");
        }
    }
}

// The staged family's admission as route_admits decided it before staged_refusal, written apart
// from it: the generic path reads the weight, the board has the tile's instructions, a ggml high
// plane on 16 bytes, the tile within the build's and the device's shared memory, a BF16 weight, K
// in 64-value steps and 16-byte codes.
bool staged_old_rule(int index, const Weight& w, const DeviceFeatures& features) {
    const generic::Route r{generic::Family::StagedMma, static_cast<std::uint8_t>(index)};
    if (!generic::supports(w) || !has_features(features, generic::route_requires(r))) return false;
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned(w.qhigh, 16)) {
        return false;
    }
    return index < generic::staged_schedule_count() && generic::staged_fits(index) &&
           generic::staged_device_fits(index) &&
           generic::codec_of(w) == generic::WeightCodec::Bf16 && (w.k % 64) == 0 &&
           aligned(w.qdata, 16);
}

// The staged family's coverage of `w` (#307): each schedule's refusal as route_admits decides it
// and as the old rule did; the counts; codes 8 bytes off refuse every schedule that admitted the
// weight for alignment.
template <class Fail>
void check_staged(const Weight& w, const DeviceFeatures& features, const std::string& where,
                  std::map<std::string, int>& counts, Fail&& fail) {
    const auto c = generic::staged_coverage(w);
    int legal    = 0;
    for (const auto& s : c.schedules) {
        const generic::Route r{generic::Family::StagedMma, static_cast<std::uint8_t>(s.schedule)};
        const bool admits = generic::route_admits(r, w, LinearPolicy::A16Only);
        ++counts[std::string(generic::schedule_refusal_name(s.refusal))];
        legal += s.refusal == generic::ScheduleRefusal::None;
        if (admits != (s.refusal == generic::ScheduleRefusal::None) ||
            admits != staged_old_rule(s.schedule, w, features)) {
            fail(where + " " + generic::route_name(r) + ": " +
                 std::string(generic::schedule_refusal_name(s.refusal)) +
                 ", which route_admits or the old rule decides otherwise");
        }
    }
    if (c.declared != generic::staged_schedule_count() || c.legal != legal ||
        (c.compiled != 0 && c.compiled != c.declared) ||
        (c.heuristic >= 0 && c.heuristic_refusal != c.schedules[c.heuristic].refusal)) {
        fail(where + ": staged counts declared " + std::to_string(c.declared) + ", compiled " +
             std::to_string(c.compiled) + ", legal " + std::to_string(c.legal));
    }
    if (c.heuristic >= 0 && c.heuristic_refusal != generic::ScheduleRefusal::None) {
        ++counts["pick " + std::string(generic::schedule_refusal_name(c.heuristic_refusal))];
    }
    if (legal > 0) {
        Weight off = w;
        off.qdata  = static_cast<const char*>(w.qdata) + 8;
        const auto o = generic::staged_coverage(off);
        for (std::size_t i = 0; i < c.schedules.size(); ++i) {
            if (c.schedules[i].refusal == generic::ScheduleRefusal::None &&
                o.schedules[i].refusal != generic::ScheduleRefusal::Alignment) {
                fail(where + ": staged with the codes 8 bytes off");
            }
        }
    }
}

// The I8 family's admission as route_admits decided it before i8_refusal, written apart from it:
// the generic path reads the weight, the board has the tile's instructions, a ggml high plane on
// 16 bytes, A8 allowed, Q4 / Q5 / Q6 G64, K and the padded K in whole 512-value runs, 16-byte
// codes and scales, a Q5 high plane on 8 bytes and a Q6 one on 16, and the tile's stages fit.
bool i8_old_rule(int index, const Weight& w, LinearPolicy policy, const DeviceFeatures& features) {
    const generic::Route r{generic::Family::I8Mma, static_cast<std::uint8_t>(index)};
    if (!generic::supports(w) || !has_features(features, generic::route_requires(r))) return false;
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned(w.qhigh, 16)) {
        return false;
    }
    const bool q4 = w.qtype == QType::Q4_G64_FP16, q5 = w.qtype == QType::Q5_G64_FP16,
               q6 = w.qtype == QType::Q6_G64_FP16;
    return index < generic::i8_schedule_count() && allows_a8(policy) && (q4 || q5 || q6) &&
           (w.k % 512) == 0 && (w.padded_shape[1] % 512) == 0 && aligned(w.qdata, 16) &&
           aligned(w.scales, 16) && (q4 || aligned(w.qhigh, q5 ? 8 : 16)) &&
           generic::i8_fits(index, w.qtype);
}

// The I8 family's coverage of `w` under `policy` (#307): each schedule's refusal as route_admits
// decides it and as the old rule did; the counts; i8_route takes the family's pick where some
// schedule admits the weight and none where none does.
template <class Fail>
void check_i8(const Weight& w, LinearPolicy policy, const DeviceFeatures& features,
              const std::string& where, std::map<std::string, int>& counts, Fail&& fail) {
    for (const std::int32_t tokens : {17, 512, 2048}) {
        const generic::Problem p{w.qtype, policy, w.n, w.k, generic::Epilogue::Store};
        const auto c = generic::i8_coverage(p, w, tokens);
        int legal    = 0;
        for (const auto& s : c.schedules) {
            const generic::Route r{generic::Family::I8Mma, static_cast<std::uint8_t>(s.schedule)};
            const bool admits = generic::route_admits(r, w, policy);
            if (tokens == 512) ++counts[std::string(generic::schedule_refusal_name(s.refusal))];
            legal += s.refusal == generic::ScheduleRefusal::None;
            if (admits != (s.refusal == generic::ScheduleRefusal::None) ||
                admits != i8_old_rule(s.schedule, w, policy, features)) {
                fail(where + " " + generic::route_name(r) + ": " +
                     std::string(generic::schedule_refusal_name(s.refusal)) +
                     ", which route_admits or the old rule decides otherwise");
            }
        }
        const auto route = generic::i8_route(p, w, tokens);
        if (c.declared != generic::i8_schedule_count() || c.legal != legal ||
            (c.compiled != 0 && c.compiled != c.declared) || route.has_value() != (legal > 0) ||
            (route && route->schedule != c.heuristic)) {
            fail(where + " T=" + std::to_string(tokens) + ": I8 counts declared " +
                 std::to_string(c.declared) + ", compiled " + std::to_string(c.compiled) +
                 ", legal " + std::to_string(c.legal) + ", pick " + std::to_string(c.heuristic));
        }
    }
}

// The remaining families' admission as route_admits decided it before their refusals, written
// apart from them: the generic checks (the generic path reads the weight, the tile's
// instructions, a ggml high plane on 16 bytes), then W4A4: a compiled schedule, NVFP4, A4 allowed
// and a finite positive input scale divisor; NVFP4 tiled: the 120a kernels, NVFP4, 8-byte codes;
// ggml's kernels: schedule 0 and the kernel's own admission (ggml_mmq_admits / ggml_vec_admits).
bool other_old_rule(generic::Family family, int index, const Weight& w, LinearPolicy policy,
                    const DeviceFeatures& features) {
    const generic::Route r{family, static_cast<std::uint8_t>(index)};
    if (!generic::supports(w) || !has_features(features, generic::route_requires(r))) return false;
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned(w.qhigh, 16)) {
        return false;
    }
    switch (family) {
    case generic::Family::W4a4Mma:
        return index < generic::w4a4_schedule_count() && w.qtype == QType::NVFP4 &&
               allows_a4(policy) && std::isfinite(w.input_scale_divisor) &&
               w.input_scale_divisor > 0.0F;
    case generic::Family::Nvfp4TiledMma:
        return ops::detail::kNativeNvfp4 && index < generic::nvfp4_tiled_schedule_count() &&
               w.qtype == QType::NVFP4 && aligned(w.qdata, 8);
    case generic::Family::GgmlMmq:
        return index == 0 && generic::ggml_mmq_admits(w, policy);
    case generic::Family::GgmlVec:
        return index == 0 && generic::ggml_vec_admits(w, policy);
    default:
        return false;
    }
}

// The remaining families' coverage of `w` under `policy` (#307): each declared schedule's refusal
// as route_admits decides it and as the old rule did, and the counts.
template <class Fail>
void check_others(const Weight& w, LinearPolicy policy, const DeviceFeatures& features,
                  const std::string& where, std::map<std::string, int>& counts, Fail&& fail) {
    for (const auto family : {generic::Family::W4a4Mma, generic::Family::Nvfp4TiledMma,
                              generic::Family::GgmlMmq, generic::Family::GgmlVec}) {
        const auto c = generic::other_family_coverage(family, w, policy);
        int legal    = 0;
        for (const auto& s : c.schedules) {
            const generic::Route r{family, static_cast<std::uint8_t>(s.schedule)};
            const bool admits = generic::route_admits(r, w, policy);
            const std::string name =
                family == generic::Family::GgmlMmq ? "mmq"
                : family == generic::Family::GgmlVec ? "vec"
                : family == generic::Family::W4a4Mma ? "w4a4"
                                                     : "nvfp4t";
            ++counts[name + " " + std::string(generic::schedule_refusal_name(s.refusal))];
            legal += s.refusal == generic::ScheduleRefusal::None;
            if (admits != (s.refusal == generic::ScheduleRefusal::None) ||
                admits != other_old_rule(family, s.schedule, w, policy, features)) {
                fail(where + " policy=" + std::to_string(static_cast<int>(policy)) + " " +
                     generic::route_name(r) + ": " +
                     std::string(generic::schedule_refusal_name(s.refusal)) +
                     ", which route_admits or the old rule decides otherwise");
            }
        }
        if (c.legal != legal || c.compiled > c.declared || c.declared == 0) {
            fail(where + ": counts of family " + std::to_string(static_cast<int>(family)));
        }
    }
}

constexpr Shape kShapes[]        = {{1024, 2048}, {5120, 5120}, {1024, 12288}};
constexpr std::int32_t kTokens[] = {1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 64, 128, 512, 2048};

// Volta (Quadro GV100, sm_70: no tensor mma.sync), where a format without a tiled tile takes the
// tiled route's CUTLASS path as schedule 0 and Q8 takes the tiles: the tiled family's coverage of
// every format and shape.
int run_volta_tiled() {
    const ScopedDeviceClass scoped(
        synthetic_device_class(0x1DBA, 7, 0, 80, 6 * kMiB, std::size_t{32} * 1024 * kMiB));
    const DeviceClass& d = device_class();
    int failures         = 0;
    std::map<std::string, int> counts;
    const auto fail = [&](const std::string& what) {
        if (++failures <= 20) std::cerr << "gv100 (" << d.id() << "): " << what << '\n';
    };
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        for (const Shape shape : kShapes) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            const auto host =
                quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options);
            const Weight w = host.device_weight(payload);
            if (!generic::supports(w)) continue;
            check_tiled(w, d.features,
                        std::string(generic::qtype_name(qtype)) + " " + std::to_string(shape.n) +
                            "x" + std::to_string(shape.k),
                        counts, fail);
        }
    }
    if (counts["codec_tile"] == 0 || counts["none"] == 0) {
        fail("no CUTLASS-only format or no admitted tile on Volta");
    }
    std::cout << "gv100 (" << d.id() << ", " << d.arch() << "): " << failures
              << " failures; tiled schedules:";
    for (const auto& [why, count] : counts) std::cout << " " << why << " " << count;
    std::cout << '\n';
    return failures == 0 ? 0 : 1;
}

int run_board(const TestBoard& b) {
    const ScopedDeviceClass scoped(synthetic_device_class(
        b.pci_device_id, b.major, b.minor, b.sms, b.l2_mib * kMiB, b.memory_gib * 1024 * kMiB));
    const DeviceClass& d = device_class();
    int failures = 0, checked = 0;
    int coverage_checked = 0;
    std::map<std::string, int> heuristic_refused; // the family's pick, refused: why, how often
    std::map<std::string, int> r8_calls;          // why a call does or does not take R8
    std::map<std::string, int> w4a8_refused;      // W4A8's schedules, per weight and policy: why
    std::map<std::string, int> tiled_refused;     // the tiled family's schedules, per weight: why
    std::map<std::string, int> staged_refused;    // the staged family's schedules, per weight: why
    std::map<std::string, int> i8_refused;        // the I8 family's schedules, per weight: why
    std::map<std::string, int> others_refused;    // W4A4, NVFP4 tiled, ggml: family and why
    const auto fail = [&](const std::string& what) {
        if (++failures <= 20) std::cerr << b.name << " (" << d.id() << "): " << what << '\n';
    };
    // Weights are never read here: an aligned address stands in for the payload.
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        for (const Shape shape : kShapes) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            const auto host =
                quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options);
            const Weight w = host.device_weight(payload);
            if (!generic::supports(w)) continue;
            {
                const generic::Problem problem{qtype, LinearPolicy::A16Only, shape.n, shape.k,
                                               generic::Epilogue::Store};
                const auto c = generic::rmma_coverage(problem, w, 1, false);
                const std::string where = std::string(generic::qtype_name(qtype)) + " " +
                                          std::to_string(shape.n) + "x" + std::to_string(shape.k);
                int compiled = 0, legal = 0;
                for (const auto& s : c.schedules) {
                    const auto info = generic::rmma_schedule_info(s.schedule);
                    const generic::Route r{generic::Family::RegisterMma,
                                           static_cast<std::uint8_t>(s.schedule)};
                    const std::string at = where + " " + generic::route_name(r) + ": " +
                                           std::string(generic::schedule_refusal_name(s.refusal));
                    if (generic::route_admits(r, w, LinearPolicy::AllowA8) !=
                        (s.refusal == generic::ScheduleRefusal::None)) {
                        fail(at + ", which route_admits decides otherwise");
                    }
                    if (info.cluster > 1 && !ops::detail::kNativeNvfp4 &&
                        s.refusal != generic::ScheduleRefusal::NotCompiled) {
                        fail(at + ", a cluster schedule this build does not compile");
                    }
                    if (info.cluster > 1 && ops::detail::kNativeNvfp4 && !d.features.clusters &&
                        s.refusal != generic::ScheduleRefusal::Instruction) {
                        fail(at + ", a cluster schedule on a board without clusters");
                    }
                    const bool quad = qtype == QType::GGML_Q4_K || qtype == QType::GGML_Q6_K;
                    if (quad && info.cluster == 1 && info.m_tiles * info.n_tiles >= 16 &&
                        s.refusal != generic::ScheduleRefusal::CodecTile) {
                        fail(at + ", a large tile the ggml decoder is not compiled for");
                    }
                    compiled += s.refusal != generic::ScheduleRefusal::NotCompiled &&
                                s.refusal != generic::ScheduleRefusal::CodecTile;
                    legal += s.refusal == generic::ScheduleRefusal::None;
                }
                if (c.declared != generic::rmma_schedule_count() ||
                    static_cast<int>(c.schedules.size()) != c.declared || c.compiled != compiled ||
                    c.legal != legal || c.legal > c.compiled || c.compiled > c.declared) {
                    fail(where + ": coverage counts declared " + std::to_string(c.declared) +
                         ", compiled " + std::to_string(c.compiled) + ", legal " +
                         std::to_string(c.legal));
                }
                // A ggml high plane 4 bytes off its 16-byte boundary (a view into a packed buffer):
                // every schedule the build holds for the codec refuses it for alignment, and the
                // tiles it does not hold still count as not compiled.
                if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr) {
                    Weight off = w;
                    off.qhigh  = static_cast<const char*>(w.qhigh) + 4;
                    const auto o = generic::rmma_coverage(problem, off, 1, false);
                    if (o.legal != 0 || o.compiled != c.compiled) {
                        fail(where + " qhigh +4: legal " + std::to_string(o.legal) + ", compiled " +
                             std::to_string(o.compiled) + " against " +
                             std::to_string(c.compiled));
                    }
                    for (std::size_t i = 0; i < o.schedules.size(); ++i) {
                        const auto aligned = c.schedules[i].refusal;
                        const auto refused = o.schedules[i].refusal;
                        const bool want =
                            aligned == generic::ScheduleRefusal::None
                                ? refused == generic::ScheduleRefusal::Alignment
                                : refused == aligned ||
                                      (aligned == generic::ScheduleRefusal::Extent &&
                                       refused == generic::ScheduleRefusal::Alignment);
                        const generic::Route r{generic::Family::RegisterMma,
                                               static_cast<std::uint8_t>(i)};
                        if (!want || generic::route_admits(r, off, LinearPolicy::AllowA8)) {
                            fail(where + " qhigh +4 " + generic::route_name(r) + ": " +
                                 std::string(generic::schedule_refusal_name(refused)));
                        }
                    }
                }
                // A weight of the same codec in a layout the generic path does not read: every
                // schedule refuses its format, and the compiled count stays the codec's.
                {
                    Weight bad = w;
                    bad.layout = w.layout == QuantLayout::GgmlPlanes ? QuantLayout::RowSplit
                                                                     : QuantLayout::GgmlPlanes;
                    if (!generic::supports(bad)) {
                        const auto b = generic::rmma_coverage(problem, bad, 1, false);
                        const bool all_format = std::all_of(
                            b.schedules.begin(), b.schedules.end(), [](const auto& s) {
                                return s.refusal == generic::ScheduleRefusal::Format ||
                                       s.refusal == generic::ScheduleRefusal::NotCompiled;
                            });
                        if (!all_format || b.legal != 0 || b.compiled != c.compiled) {
                            fail(where + " wrong layout: compiled " + std::to_string(b.compiled) +
                                 " against " + std::to_string(c.compiled));
                        }
                    }
                }
                // R8, per weight (#307): the schedule rule against the old one written apart
                // (the generic path reads the weight, the call's admission to R8, the ggml high
                // plane on 16 bytes); a high plane 4 bytes off refuses every schedule for
                // alignment; another layout refuses for format and keeps the compiled count.
                {
                    const generic::Problem a8{qtype, LinearPolicy::AllowA8, shape.n, shape.k,
                                              generic::Epilogue::Store};
                    const auto r8 = generic::r8_coverage(a8, w, 1);
                    const bool admitted = r8.call != generic::R8CallRefusal::NotAdmitted;
                    if (r8.family.declared != generic::r8_schedule_count() ||
                        (r8.family.compiled != 0 && r8.family.compiled != r8.family.declared) ||
                        r8.family.legal != (admitted ? r8.family.declared : 0)) {
                        fail(where + ": R8 counts declared " + std::to_string(r8.family.declared) +
                             ", compiled " + std::to_string(r8.family.compiled) + ", legal " +
                             std::to_string(r8.family.legal));
                    }
                    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && admitted) {
                        Weight off = w;
                        off.qhigh  = static_cast<const char*>(w.qhigh) + 4;
                        const auto o = generic::r8_coverage(a8, off, 1);
                        const bool all_alignment = std::all_of(
                            o.family.schedules.begin(), o.family.schedules.end(), [](const auto& s) {
                                return s.refusal == generic::ScheduleRefusal::Alignment;
                            });
                        if (!all_alignment || o.family.compiled != r8.family.compiled) {
                            fail(where + ": R8 with the high plane 4 bytes off");
                        }
                    }
                    Weight bad = w;
                    bad.layout = w.layout == QuantLayout::GgmlPlanes ? QuantLayout::RowSplit
                                                                     : QuantLayout::GgmlPlanes;
                    if (!generic::supports(bad)) {
                        const auto b = generic::r8_coverage(a8, bad, 1);
                        const bool all_format = std::all_of(
                            b.family.schedules.begin(), b.family.schedules.end(), [](const auto& s) {
                                return s.refusal == generic::ScheduleRefusal::Format;
                            });
                        if (!all_format || b.family.compiled != r8.family.compiled) {
                            fail(where + ": R8 in a layout the generic path does not read");
                        }
                    }
                }
                // W4A8, per weight and policy (#307): route_admits decides each schedule by
                // w4a8_refusal, and agrees with the old rule written apart; an A16 call is refused
                // for its policy (or before it); the counts; a high plane 4 bytes off refuses the
                // producer-warp tiles for alignment and leaves the classic ones; another layout
                // refuses for format and keeps the compiled count.
                for (const LinearPolicy policy :
                     {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
                    const generic::Problem p{qtype, policy, shape.n, shape.k,
                                             generic::Epilogue::Store};
                    const auto c   = generic::w4a8_coverage(p, w, 512);
                    const auto pol = " policy=" + std::to_string(static_cast<int>(policy));
                    int legal      = 0;
                    for (const auto& s : c.schedules) {
                        const generic::Route r{generic::Family::W4a8Mma,
                                               static_cast<std::uint8_t>(s.schedule)};
                        const bool admits = generic::route_admits(r, w, policy);
                        if (s.schedule == c.heuristic &&
                            s.refusal != generic::ScheduleRefusal::None) {
                            ++w4a8_refused["pick " + std::string(generic::schedule_refusal_name(
                                                         s.refusal))];
                        }
                        ++w4a8_refused[std::string(generic::schedule_refusal_name(s.refusal))];
                        legal += s.refusal == generic::ScheduleRefusal::None;
                        if (admits != (s.refusal == generic::ScheduleRefusal::None) ||
                            admits != w4a8_old_rule(s.schedule, w, policy, d.features)) {
                            fail(where + pol + " " + generic::route_name(r) + ": " +
                                 std::string(generic::schedule_refusal_name(s.refusal)) +
                                 ", which route_admits or the old rule decides otherwise");
                        }
                        if (policy == LinearPolicy::A16Only &&
                            s.refusal != generic::ScheduleRefusal::Policy &&
                            s.refusal != generic::ScheduleRefusal::Format &&
                            s.refusal != generic::ScheduleRefusal::NotCompiled &&
                            s.refusal != generic::ScheduleRefusal::Resource) {
                            fail(where + pol + " " + generic::route_name(r) +
                                 ": an A16 call's W4A8 refusal is not its policy");
                        }
                    }
                    if (c.declared != generic::w4a8_schedule_count() || c.legal != legal ||
                        c.heuristic_refusal == generic::ScheduleRefusal::NotDeclared) {
                        fail(where + pol + ": W4A8 counts declared " + std::to_string(c.declared) +
                             ", compiled " + std::to_string(c.compiled) + ", legal " +
                             std::to_string(c.legal));
                    }
                    if (policy != LinearPolicy::AllowA8) continue;
                    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr) {
                        Weight off = w;
                        off.qhigh  = static_cast<const char*>(w.qhigh) + 4;
                        const auto o = generic::w4a8_coverage(p, off, 512);
                        for (std::size_t i = 0; i < c.schedules.size(); ++i) {
                            const bool producer =
                                generic::w4a8_schedule_info(c.schedules[i].schedule).producers > 0;
                            const auto before = c.schedules[i].refusal;
                            const auto after  = o.schedules[i].refusal;
                            const bool right =
                                producer && before == generic::ScheduleRefusal::None
                                    ? after == generic::ScheduleRefusal::Alignment
                                    : (producer || after == before);
                            if (!right) fail(where + ": W4A8 with the high plane 4 bytes off");
                        }
                        if (o.compiled != c.compiled) {
                            fail(where + ": W4A8's compiled count moved with the high plane");
                        }
                    }
                    Weight bad = w;
                    bad.layout = w.layout == QuantLayout::GgmlPlanes ? QuantLayout::RowSplit
                                                                     : QuantLayout::GgmlPlanes;
                    if (!generic::supports(bad)) {
                        const auto b = generic::w4a8_coverage(p, bad, 512);
                        const bool all_format =
                            std::all_of(b.schedules.begin(), b.schedules.end(), [](const auto& s) {
                                return s.refusal == generic::ScheduleRefusal::Format;
                            });
                        if (!all_format || b.compiled != c.compiled) {
                            fail(where + ": W4A8 in a layout the generic path does not read");
                        }
                    }
                }
                check_tiled(w, d.features, where, tiled_refused, fail);
                check_staged(w, d.features, where, staged_refused, fail);
                for (const LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
                    check_i8(w, policy, d.features, where, i8_refused, fail);
                }
                for (const LinearPolicy policy :
                     {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
                    check_others(w, policy, d.features, where, others_refused, fail);
                    // A ggml high plane 4 bytes off: ggml's kernels refuse it for alignment.
                    if (w.layout != QuantLayout::GgmlPlanes || w.qhigh == nullptr) continue;
                    Weight off = w;
                    off.qhigh  = static_cast<const char*>(w.qhigh) + 4;
                    for (const auto family : {generic::Family::GgmlMmq, generic::Family::GgmlVec}) {
                        if (generic::ggml_refusal(family, 0, w, policy) ==
                                generic::ScheduleRefusal::None &&
                            generic::ggml_refusal(family, 0, off, policy) !=
                                generic::ScheduleRefusal::Alignment) {
                            fail(where + ": ggml with the high plane 4 bytes off");
                        }
                    }
                }
                ++coverage_checked;
            }
            for (const LinearPolicy policy :
                 {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
                if (policy == LinearPolicy::AllowA4 && qtype != QType::NVFP4) continue;
                for (const generic::Epilogue epilogue :
                     {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                      generic::Epilogue::SwiGlu}) {
                    const std::int32_t rows =
                        epilogue == generic::Epilogue::SwiGlu ? shape.n / 2 : shape.n;
                    const generic::Problem problem{qtype, policy, rows, shape.k, epilogue};
                    for (const std::int32_t t : kTokens) {
                        for (const bool prefill : {false, true}) {
                            if (prefill && t < 17) continue;
                            const std::string where =
                                std::string(generic::qtype_name(qtype)) + " " +
                                std::to_string(rows) + "x" + std::to_string(shape.k) +
                                " T=" + std::to_string(t) + (prefill ? " prefill" : "") +
                                " policy=" + std::to_string(static_cast<int>(policy)) +
                                " epilogue=" + std::to_string(static_cast<int>(epilogue));
                            generic::Route route;
                            try {
                                route = generic::select_route(problem, w, t, prefill);
                            } catch (const std::exception& error) {
                                fail(where + ": select_route threw: " + error.what());
                                continue;
                            }
                            ++checked;
                            const std::string name = generic::route_name(route);
                            const auto coverage =
                                generic::rmma_coverage(problem, w, t, prefill);
                            if (coverage.heuristic_refusal ==
                                generic::ScheduleRefusal::NotDeclared) {
                                fail(where + ": the register-MMA pick is no declared schedule");
                            } else if (coverage.heuristic_refusal !=
                                       generic::ScheduleRefusal::None) {
                                ++heuristic_refused[std::string(generic::schedule_refusal_name(
                                    coverage.heuristic_refusal))];
                            }
                            // R8's coverage (#307): route_admits decides each schedule by
                            // r8_refusal; a policy without A8 refuses every schedule of a format R8
                            // reads as Policy; a call that takes R8 has a schedule to run it on.
                            const auto r8 = generic::r8_coverage(problem, w, t);
                            ++r8_calls[std::string(generic::r8_call_refusal_name(r8.call))];
                            for (const auto& s : r8.family.schedules) {
                                const generic::Route r{generic::Family::R8Mma,
                                                       static_cast<std::uint8_t>(s.schedule)};
                                // The old rule, written apart from r8_refusal: the generic path
                                // reads the weight (filtered above) and the call's weight is
                                // admitted to R8 (allows_r8: r8_call_refusal's first test).
                                const bool old_rule =
                                    r8.call != generic::R8CallRefusal::NotAdmitted;
                                if (generic::route_admits(r, w, policy) != old_rule ||
                                    generic::route_admits(r, w, policy) !=
                                        (s.refusal == generic::ScheduleRefusal::None)) {
                                    fail(where + " " + generic::route_name(r) + ": " +
                                         std::string(generic::schedule_refusal_name(s.refusal)) +
                                         ", which route_admits decides otherwise");
                                }
                                if (policy == LinearPolicy::A16Only && r8.family.compiled > 0 &&
                                    s.refusal != generic::ScheduleRefusal::Policy &&
                                    s.refusal != generic::ScheduleRefusal::Format) {
                                    fail(where + " " + generic::route_name(r) +
                                         ": an A16 call's R8 refusal is not its policy");
                                }
                            }
                            if (r8.call == generic::R8CallRefusal::None && r8.family.legal == 0) {
                                fail(where + ": the call takes R8 and no R8 schedule admits it");
                            }
                            if (route.family == generic::Family::R8Mma && r8.family.legal == 0) {
                                fail(where + ": chose " + name + ", which no R8 schedule admits");
                            }
                            if (route.family == generic::Family::W4a8Mma &&
                                generic::w4a8_coverage(problem, w, t).legal == 0) {
                                fail(where + ": chose " + name + ", which no W4A8 schedule admits");
                            }
                            if (!generic::route_admits(route, w, policy)) {
                                fail(where + ": chose " + name + ", which route_admits refuses");
                            }
                            if (!has_features(d.features, generic::route_requires(route))) {
                                fail(where + ": chose " + name + ", which needs a feature " +
                                     "the board lacks");
                            }
                            if (!ops::detail::kNativeNvfp4 &&
                                (route.family == generic::Family::W4a4Mma ||
                                 route.family == generic::Family::Nvfp4TiledMma)) {
                                fail(where + ": chose " + name + ", absent from this build");
                            }
                        }
                    }
                }
            }
        }
    }
    // The staged family reads BF16 weights only, which the formats above leave out: every shape,
    // and a K off the 64-value steps (the generic path reads it, the staged tiles do not). Where
    // the BF16 MMA is native some schedule admits them.
    for (const Shape shape : {Shape{1024, 2048}, Shape{5120, 5120}, Shape{1024, 2080}}) {
        const auto host = direct_bf16_weight::make_patterned(shape.n, shape.k, 7U);
        const Weight w  = host.device_weight(payload);
        const std::string where =
            "BF16 " + std::to_string(shape.n) + "x" + std::to_string(shape.k);
        check_staged(w, d.features, where, staged_refused, fail);
        check_tiled(w, d.features, where, tiled_refused, fail);
    }
    // (The matrix kernel asks the stand's own device, ggml_mmq::device_supported: a Volta stand
    // runs these boards without it.)
    if (others_refused["vec none"] == 0) fail("no ggml weight admitted to ggml's vector kernel");
    if (d.features.mma_bf16_native && i8_refused["none"] == 0) {
        fail("no G64 weight admitted to an I8 tile");
    }
    if (d.features.mma_bf16_native &&
        (staged_refused["none"] == 0 || staged_refused["extent"] == 0)) {
        fail("no BF16 weight admitted to a staged tile, or none refused for its K");
    }
    const auto status     = ops::route_table_status();
    const bool want_table = std::string_view(b.table) != "";
    if (status.loaded != want_table) {
        fail(std::string("route table ") + (status.loaded ? "loaded" : "not loaded") + " (" +
             status.message + "), expected " + (want_table ? b.table : "none"));
    }
    if (status.loaded && status.gpu_class != b.table) {
        fail("route table class " + status.gpu_class + ", expected the nearest board's " + b.table);
    }
    if (checked < 1000) fail("only " + std::to_string(checked) + " choices checked");
    std::cout << b.name << " (" << d.id() << ", " << d.arch() << "): " << checked
              << " route choices, " << coverage_checked << " register-MMA coverages, "
              << failures << " failures; the register-MMA pick refused:";
    for (const auto& [why, count] : heuristic_refused) std::cout << " " << why << " " << count;
    std::cout << "; R8 calls:";
    for (const auto& [why, count] : r8_calls) std::cout << " " << why << " " << count;
    std::cout << "; W4A8 schedules:";
    for (const auto& [why, count] : w4a8_refused) std::cout << " " << why << " " << count;
    std::cout << "; tiled schedules:";
    for (const auto& [why, count] : tiled_refused) std::cout << " " << why << " " << count;
    std::cout << "; staged schedules:";
    for (const auto& [why, count] : staged_refused) std::cout << " " << why << " " << count;
    std::cout << "; I8 schedules:";
    for (const auto& [why, count] : i8_refused) std::cout << " " << why << " " << count;
    std::cout << "; others:";
    for (const auto& [why, count] : others_refused) std::cout << " " << why << " " << count;
    std::cout << '\n';
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--board") {
            for (const TestBoard& b : kBoards) {
                if (std::string_view(argv[2]) == b.name) return run_board(b);
            }
            std::cerr << "unknown board " << argv[2] << '\n';
            return 2;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--volta-tiled") return run_volta_tiled();
        int failed = 0;
        for (const TestBoard& b : kBoards) {
            const std::string command = std::string(argv[0]) + " --board " + b.name;
            if (std::system(command.c_str()) != 0) ++failed;
        }
        if (std::system((std::string(argv[0]) + " --volta-tiled").c_str()) != 0) ++failed;
        if (failed != 0) {
            std::cerr << "route classes: " << failed << " boards failed\n";
            return 1;
        }
        std::cout << "route classes: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "route classes: " << error.what() << '\n';
        return 1;
    }
}
