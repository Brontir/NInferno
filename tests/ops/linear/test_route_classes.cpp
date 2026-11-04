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
#include "ops/generic/build_manifest.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"

#include <cuda_runtime.h>

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

// The compiled-variant manifest (#307) against the families' refusals for a weight: a schedule
// the manifest says this build holds no kernel of for the weight's format is never admitted, and
// the refusal says so (not compiled, the codec's tile, a tile past the build's shared memory, or a
// format the family does not read); a schedule refused as not compiled, for its codec's tile
// (register MMA) or for the build's shared memory (tiled, W4A8, I8, staged past staged_fits) is
// one the manifest says is absent. A route whose instructions the build itself lacks
// (route_requires against build_features: I8 on sm_75) is not checked: a synthetic board above
// the build is offered routes whose kernels it compiled empty, which no real device of that build
// takes (its features are the build's in common).
generic::ScheduleRefusal family_refusal(generic::Route r, const Weight& w, LinearPolicy policy) {
    using generic::Family;
    switch (r.family) {
    case Family::RegisterMma: return generic::rmma_refusal(r.schedule, w);
    case Family::W4a4Mma: return generic::w4a4_refusal(r.schedule, w, policy);
    case Family::TiledMma: return generic::tiled_refusal(r.schedule, w);
    case Family::W4a8Mma: return generic::w4a8_refusal(r.schedule, w, policy);
    case Family::StagedMma: return generic::staged_refusal(r.schedule, w);
    case Family::Nvfp4TiledMma: return generic::nvfp4_tiled_refusal(r.schedule, w);
    case Family::R8Mma: return generic::r8_refusal(r.schedule, w, policy);
    case Family::GgmlMmq:
    case Family::GgmlVec: return generic::ggml_refusal(r.family, r.schedule, w, policy);
    case Family::I8Mma: return generic::i8_refusal(r.schedule, w, policy);
    case Family::Scalar: break;
    }
    return generic::ScheduleRefusal::None;
}

template <class Fail>
void check_manifest(const generic::BuildManifest& manifest, const Weight& w, LinearPolicy policy,
                    const std::string& where, std::map<std::string, int>& counts, Fail&& fail) {
    using generic::ScheduleRefusal;
    for (const auto& s : manifest.schedules) {
        if (s.route.family == generic::Family::Scalar ||
            !has_features(build_features(), generic::route_requires(s.route))) {
            continue;
        }
        const bool listed = s.only_formats.empty() ||
                            std::find(s.only_formats.begin(), s.only_formats.end(), w.qtype) !=
                                s.only_formats.end();
        const bool omitted =
            std::any_of(s.omitted_formats.begin(), s.omitted_formats.end(), [&](const auto& f) {
                return f.qtype == w.qtype && (f.k_multiple == 0 || w.k % f.k_multiple == 0);
            });
        const bool absent = !s.compiled || !listed || omitted;
        const auto why    = family_refusal(s.route, w, policy);
        using generic::Family;
        const Family family = s.route.family;
        const bool build_resource =
            why == ScheduleRefusal::Resource &&
            (family == Family::TiledMma || family == Family::W4a8Mma || family == Family::I8Mma ||
             (family == Family::StagedMma && !generic::staged_fits(s.route.schedule)));
        const bool unbuilt_refusal =
            why == ScheduleRefusal::NotCompiled || build_resource ||
            (why == ScheduleRefusal::CodecTile && family == Family::RegisterMma);
        const bool absent_refusal = unbuilt_refusal || why == ScheduleRefusal::Format ||
                                    why == ScheduleRefusal::CodecTile ||
                                    why == ScheduleRefusal::Resource;
        ++counts[absent ? "absent" : "held"];
        if ((unbuilt_refusal && !absent) || (absent && !absent_refusal) ||
            (absent && generic::route_admits(s.route, w, policy))) {
            fail(where + " policy=" + std::to_string(static_cast<int>(policy)) + " " +
                 generic::route_name(s.route) + ": the manifest says " +
                 (absent ? "absent" : "held") + ", the refusal " +
                 std::string(generic::schedule_refusal_name(why)));
        }
    }
}

// The manifest's schedules are the families' declared lists, in order, each once.
template <class Fail>
void check_manifest_lists(const generic::BuildManifest& manifest, Fail&& fail) {
    using generic::Family;
    std::map<Family, int> seen;
    for (const auto& s : manifest.schedules) {
        if (s.route.schedule != seen[s.route.family]++) {
            fail("manifest: " + generic::route_name(s.route) + " out of its family's order");
        }
    }
    const std::pair<Family, int> declared[] = {
        {Family::Scalar, 1},
        {Family::RegisterMma, generic::rmma_schedule_count()},
        {Family::W4a4Mma, generic::w4a4_schedule_names()},
        {Family::TiledMma, generic::tiled_schedule_count()},
        {Family::W4a8Mma, generic::w4a8_schedule_count()},
        {Family::StagedMma, generic::staged_schedule_count()},
        {Family::Nvfp4TiledMma, generic::nvfp4_tiled_schedule_count()},
        {Family::R8Mma, generic::r8_schedule_count()},
        {Family::GgmlMmq, 1},
        {Family::GgmlVec, 1},
        {Family::I8Mma, generic::i8_schedule_count()},
    };
    for (const auto& [family, count] : declared) {
        if (seen[family] != count) {
            fail("manifest: family " + std::to_string(static_cast<int>(family)) + " lists " +
                 std::to_string(seen[family]) + " schedules, " + std::to_string(count) +
                 " declared");
        }
    }
    const int compiled_w4a4 =
        static_cast<int>(std::count_if(manifest.schedules.begin(), manifest.schedules.end(),
                                       [](const auto& s) {
                                           return s.route.family == Family::W4a4Mma && s.compiled;
                                       }));
    if (compiled_w4a4 != generic::w4a4_schedule_count() ||
        manifest.native_nvfp4 != (generic::w4a4_schedule_count() > 0)) {
        fail("manifest: the W4A4 kernels held disagree with the build");
    }
    const auto json = generic::build_manifest_json(manifest, {{"version", "test"}});
    if (json.find("\"build\": {\n    \"version\": \"test\"") == std::string::npos ||
        json.find("\"rmma\"") == std::string::npos) {
        fail("manifest: the document lacks its fields");
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

// #298: the register-MMA kernels' compiled resources on this device (rmma_resources, the launch
// path described instead of run): a description exactly where the schedule is compiled for the
// weight's codec (rmma_refusal neither not compiled nor the codec's tile nor a format the generic
// path does not read, and K on the codec's step), with the schedule's threads, its grid over the
// call's rows and tokens, and registers the SM holds. Prints each codec's most registers and spill
// bytes over the schedules and widths a call takes.
int run_resources(bool detail) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "resources: CUDA unavailable; skipped\n";
        return 0;
    }
    int failures = 0, described = 0;
    const auto fail = [&](const std::string& what) {
        if (++failures <= 20) std::cerr << "resources: " << what << '\n';
    };
    int registers_per_sm = 65536;
    bool tensor_mma      = true;
    try {
        registers_per_sm = device_class().registers_per_sm > 0 ? device_class().registers_per_sm
                                                               : registers_per_sm;
        tensor_mma       = device_class().features.mma_tensor;
    } catch (const std::exception&) {}
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    struct Most {
        int registers = 0;
        std::size_t local = 0;
        std::string where;
    };
    std::map<std::string, Most> most;
    for (const QType qtype : kFormats) {
        for (const Shape shape : {Shape{4096, 4096}, Shape{1024, 12288}}) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            const auto host =
                quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options);
            const Weight w = host.device_weight(payload);
            if (!generic::supports(w)) continue;
            for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                const auto why  = generic::rmma_refusal(i, w);
                const auto info = generic::rmma_schedule_info(i);
                const bool held = why != generic::ScheduleRefusal::NotCompiled &&
                                  why != generic::ScheduleRefusal::CodecTile &&
                                  why != generic::ScheduleRefusal::Format &&
                                  why != generic::ScheduleRefusal::Extent;
                for (const std::int32_t t : {1, 4, 8}) {
                    const auto r = generic::rmma_resources(i, w, generic::Epilogue::Store, t);
                    const generic::Route route{generic::Family::RegisterMma,
                                               static_cast<std::uint8_t>(i)};
                    const std::string at = std::string(generic::qtype_name(qtype)) + " " +
                                           std::to_string(shape.n) + "x" +
                                           std::to_string(shape.k) + " " +
                                           generic::route_name(route) + " T=" + std::to_string(t);
                    // Without tensor mma.sync Volta's HMMA and GEMV kernels take some held calls.
                    if (r.has_value() != held && (r || tensor_mma)) {
                        fail(at + ": described " + (r ? "yes" : "no") + ", refusal " +
                             std::string(generic::schedule_refusal_name(why)));
                        continue;
                    }
                    if (!r) continue;
                    ++described;
                    const int threads = 32 * info.warps_m * info.warps_k;
                    if (r->threads != threads || r->registers <= 0 || r->grid_x == 0 ||
                        r->grid_y == 0 || r->registers * r->threads > registers_per_sm ||
                        r->max_threads_per_block < r->threads) {
                        fail(at + ": threads " + std::to_string(r->threads) + " (schedule " +
                             std::to_string(threads) + "), registers " +
                             std::to_string(r->registers) + ", grid " + std::to_string(r->grid_x) +
                             "x" + std::to_string(r->grid_y));
                    }
                    if (detail) {
                        std::cout << at << ": " << r->registers << " registers, " << r->local_bytes
                                  << " B local, " << r->static_smem << " B smem, grid " << r->grid_x
                                  << "x" << r->grid_y << "x" << r->grid_z
                                  << (r->latency_variant ? ", latency" : "") << '\n';
                    }
                    Most& m = most[generic::qtype_name(qtype)];
                    if (r->registers > m.registers ||
                        (r->registers == m.registers && r->local_bytes > m.local)) {
                        m = {r->registers, r->local_bytes, at};
                    }
                    if (r->local_bytes > m.local) m.local = r->local_bytes;
                }
            }
        }
    }
    if (described == 0) fail("no register-MMA kernel described");
    // R8 (r8_resources): nothing where route_admits refuses R8 under AllowA8; where described, the
    // schedule's threads, one CTA per warps x rows of output rows, a one-dimensional grid, no
    // pre-quantized form for one token, and registers the SM holds.
    int r8_described = 0, r8_pre = 0, r8_admitted = 0;
    for (const QType qtype : kFormats) {
        const Shape shape{4096, 4096};
        quantized_weight::PatternedWeightOptions options;
        if (qtype == QType::NVFP4) options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
        const auto host = quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options);
        const Weight w  = host.device_weight(payload);
        if (!generic::supports(w)) continue;
        for (int i = 0; i < generic::r8_schedule_count(); ++i) {
            const generic::Route route{generic::Family::R8Mma, static_cast<std::uint8_t>(i)};
            const bool admitted = generic::route_admits(route, w, LinearPolicy::AllowA8);
            const auto info     = generic::r8_schedule_info(i);
            r8_admitted += admitted ? 1 : 0;
            for (const std::int32_t t : {1, 2, 4, 8}) {
                const auto r = generic::r8_resources(i, w, generic::Epilogue::Store, t);
                const std::string at = std::string(generic::qtype_name(qtype)) + " " +
                                       generic::route_name(route) + " T=" + std::to_string(t);
                if (r && !admitted) {
                    fail(at + ": described although route_admits refuses it");
                    continue;
                }
                if (!r) continue;
                ++r8_described;
                r8_pre += r->prequantized ? 1 : 0;
                const auto& k = r->kernel;
                const unsigned ctas = static_cast<unsigned>(
                    (shape.n + info.warps * info.rows - 1) / (info.warps * info.rows));
                if (k.threads != 32 * info.warps || k.grid_x != ctas || k.grid_y != 1 ||
                    k.grid_z != 1 || k.registers <= 0 ||
                    k.registers * k.threads > registers_per_sm || (t == 1 && r->prequantized)) {
                    fail(at + ": threads " + std::to_string(k.threads) + ", grid " +
                         std::to_string(k.grid_x) + "x" + std::to_string(k.grid_y) + "x" +
                         std::to_string(k.grid_z) + " (" + std::to_string(ctas) + " CTAs), " +
                         std::to_string(k.registers) + " registers" +
                         (r->prequantized ? ", pre-quantized" : ""));
                }
                if (detail) {
                    std::cout << at << ": " << k.registers << " registers, "
                              << k.local_bytes << " B local, " << k.dynamic_smem
                              << " B dynamic smem" << (r->prequantized ? ", pre-quantized" : "")
                              << '\n';
                }
            }
        }
    }
    // Where route_admits holds R8 for some format and schedule, some call takes it.
    if (r8_described == 0 && r8_admitted > 0) fail("no R8 kernel described");
    // The verify widths run the pre-quantized form (T = 2..kR8VerifyTokens) where a format has it.
    if (r8_admitted > 0 && r8_pre == 0) fail("no pre-quantized R8 kernel described");
    std::cout << "resources R8: " << r8_described << " described (" << r8_pre
              << " pre-quantized)\n";
    std::cout << "resources (" << device_class().id() << "): " << described << " described, "
              << failures << " failures; per format the most registers and spill bytes:";
    for (const auto& [format, m] : most) {
        std::cout << " " << format << " " << m.registers << "r/" << m.local << "B";
    }
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
    std::map<std::string, int> manifest_checked;  // schedules the manifest holds or omits
    const auto fail = [&](const std::string& what) {
        if (++failures <= 20) std::cerr << b.name << " (" << d.id() << "): " << what << '\n';
    };
    const generic::BuildManifest manifest = generic::build_manifest();
    check_manifest_lists(manifest, fail);
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
                    check_manifest(manifest, w, policy, where, manifest_checked, fail);
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
        check_manifest(manifest, w, LinearPolicy::A16Only, where, manifest_checked, fail);
    }
    // ggml_mmq::device_supported reads the board (device_class), so a board with tensor mma.sync
    // admits some weight to the matrix kernel too.
    if (others_refused["vec none"] == 0 ||
        (d.features.mma_tensor && others_refused["mmq none"] == 0)) {
        fail("no ggml weight admitted to ggml's vector or matrix kernel");
    }
    if (manifest_checked["absent"] == 0 || manifest_checked["held"] == 0) {
        fail("the manifest held or omitted no schedule for any weight");
    }
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
    std::cout << "; manifest:";
    for (const auto& [what, count] : manifest_checked) std::cout << " " << what << " " << count;
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
        if (argc == 2 && std::string_view(argv[1]) == "--resources") return run_resources(false);
        if (argc == 2 && std::string_view(argv[1]) == "--resources-detail") {
            return run_resources(true);
        }
        int failed = 0;
        for (const TestBoard& b : kBoards) {
            const std::string command = std::string(argv[0]) + " --board " + b.name;
            if (std::system(command.c_str()) != 0) ++failed;
        }
        if (std::system((std::string(argv[0]) + " --volta-tiled").c_str()) != 0) ++failed;
        if (std::system((std::string(argv[0]) + " --resources").c_str()) != 0) ++failed;
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
