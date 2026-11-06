// The generic path's family coverage (rmma_coverage .. w4a8_coverage, build_manifest): which
// schedules a build declares, compiles and refuses for a call, read from the same rules the
// selection runs.

#include "core/device_class.h"
#include "core/tensor.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/native_nvfp4.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/route_table.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::ops::detail::generic {

FamilyCoverage rmma_coverage(const Problem& problem, const Weight& w, std::int32_t tokens,
                             bool prefill_band) {
    FamilyCoverage c;
    c.declared = rmma_schedule_count();
    // Compiled is the build's for the weight's codec, whatever else refuses the weight (a
    // malformed layout refuses every schedule as Format): none where the format has no codec.
    std::optional<WeightCodec> codec;
    try {
        codec = codec_of(w);
    } catch (const std::exception&) {
    }
    for (int i = 0; i < c.declared; ++i) {
        const ScheduleRefusal why = rmma_refusal(i, w);
        c.schedules.push_back({i, why});
        const auto s = rmma_schedule_info(i);
        c.compiled += rmma_schedule_launchable(i) && codec.has_value() &&
                      rmma_tile_compiled(*codec, s.m_tiles, s.n_tiles, w.k);
        c.legal += why == ScheduleRefusal::None;
    }
    c.heuristic         = heuristic_rmma(problem, tokens, prefill_band);
    c.heuristic_refusal = rmma_refusal(c.heuristic, w);
    return c;
}

FamilyCoverage tiled_coverage(const Weight& w) {
    FamilyCoverage c;
    c.declared = tiled_schedule_count();
    // What the build holds follows from the format's codec alone, whether or not this weight is
    // one the generic path reads (its schedules then refuse for format).
    bool cutlass_only = false, takes = false;
    WeightCodec codec{};
    try {
        codec        = codec_of(w);
        cutlass_only = volta_cutlass_only(codec);
        takes        = tiled_takes(codec);
    } catch (const std::exception&) {} // a format with no codec: nothing compiled
    for (int i = 0; i < c.declared; ++i) {
        const ScheduleRefusal why = tiled_refusal(i, w);
        c.schedules.push_back({i, why});
        c.compiled += takes || (cutlass_only && i == 0);
        c.legal += why == ScheduleRefusal::None;
        if (c.heuristic < 0 && (cutlass_only ? i == 0 : takes && tiled_fits(i, codec))) {
            c.heuristic         = i;
            c.heuristic_refusal = why;
        }
    }
    return c;
}

FamilyCoverage other_family_coverage(Family family, const Weight& w, LinearPolicy policy) {
    FamilyCoverage c;
    const bool nvfp4 = w.qtype == QType::NVFP4;
    switch (family) {
    case Family::W4a4Mma:
        c.declared = w4a4_schedule_names();
        for (int i = 0; i < c.declared; ++i) {
            const ScheduleRefusal why = w4a4_refusal(i, w, policy);
            c.schedules.push_back({i, why});
            c.compiled += nvfp4 && i < w4a4_schedule_count();
            c.legal += why == ScheduleRefusal::None;
        }
        return c;
    case Family::Nvfp4TiledMma:
        c.declared = nvfp4_tiled_schedule_count();
        for (int i = 0; i < c.declared; ++i) {
            const ScheduleRefusal why = nvfp4_tiled_refusal(i, w);
            c.schedules.push_back({i, why});
            c.compiled += nvfp4 && kNativeNvfp4;
            c.legal += why == ScheduleRefusal::None;
        }
        return c;
    case Family::GgmlMmq:
    case Family::GgmlVec: {
        c.declared               = 1;
        const ScheduleRefusal why = ggml_refusal(family, 0, w, policy);
        c.schedules.push_back({0, why});
        const auto type = mmq_type(w);
        c.compiled = type && (family == Family::GgmlMmq ? ggml_mmq::matrix_supported(*type)
                                                        : ggml_mmq::vec_supported(*type));
        c.legal    = why == ScheduleRefusal::None;
        return c;
    }
    default:
        return c;
    }
}

FamilyCoverage i8_coverage(const Problem& problem, const Weight& w, std::int32_t tokens) {
    FamilyCoverage c;
    c.declared       = i8_schedule_count();
    const bool reads = w.qtype == QType::Q4_G64_FP16 || w.qtype == QType::Q5_G64_FP16 ||
                       w.qtype == QType::Q6_G64_FP16;
    for (int i = 0; i < c.declared; ++i) {
        const ScheduleRefusal why = i8_refusal(i, w, problem.policy);
        c.schedules.push_back({i, why});
        c.compiled += reads;
        c.legal += why == ScheduleRefusal::None;
    }
    if (reads) {
        c.heuristic = i8_pick(problem, w.qtype, tokens);
        if (c.heuristic >= 0) c.heuristic_refusal = i8_refusal(c.heuristic, w, problem.policy);
    }
    return c;
}

BuildManifest build_manifest() {
    BuildManifest m;
#ifdef NINFERNO_BUILD_ARCH
    m.arch = NINFERNO_BUILD_ARCH;
#endif
#ifdef NINFERNO_BUILD_ARCH_SPECIFIC
    m.arch_specific = true;
#endif
    m.native_nvfp4 = kNativeNvfp4;
    // The formats the generic path has a codec for: every QType number route_table names.
    std::vector<std::pair<QType, WeightCodec>> formats;
    for (int n = 0; n <= 0xFF; ++n) {
        Weight w{};
        w.qtype = static_cast<QType>(n);
        if (std::string_view(qtype_name(w.qtype)) == "unknown") continue;
        try {
            formats.emplace_back(w.qtype, codec_of(w));
        } catch (const std::exception&) {}
    }
    const auto add = [&](Family family, int count, auto&& fill) {
        for (int i = 0; i < count; ++i) {
            ManifestSchedule s;
            s.route    = Route{family, static_cast<std::uint8_t>(i)};
            s.compiled = true;
            fill(i, s);
            m.schedules.push_back(std::move(s));
        }
    };
    const auto none = [](int, ManifestSchedule&) {};
    add(Family::Scalar, 1, none);
    add(Family::RegisterMma, rmma_schedule_count(), [&](int i, ManifestSchedule& s) {
        s.compiled         = rmma_schedule_launchable(i);
        const auto info    = rmma_schedule_info(i);
        for (const auto& [qtype, codec] : formats) {
            if (!rmma_tile_compiled(codec, info.m_tiles, info.n_tiles, 0)) {
                s.omitted_formats.push_back({qtype, 0});
            } else if (!rmma_tile_compiled(codec, info.m_tiles, info.n_tiles, 128)) {
                s.omitted_formats.push_back({qtype, 128});
            }
        }
    });
    add(Family::W4a4Mma, w4a4_schedule_names(),
        [](int i, ManifestSchedule& s) { s.compiled = i < w4a4_schedule_count(); });
    // A tile past this build's shared-memory ceiling (kMaxSharedBytesPerBlock) compiles to a stub
    // (generic_tiled.cu, generic_w4a8.cuh, generic_staged.cu, generic_i8.cu): the fits checks the
    // refusals answer Resource by. Staged's device check (staged_device_fits) is the board's.
    // The tiles are compiled for tiled_supports' codecs and Q8 (the Volta tiles, volta_tiled_route).
    // On a build without tensor mma.sync, schedule 0 alone names the CUTLASS copy of every other
    // format (volta_cutlass_only; tiled_refusal: CodecTile past it): those formats are omitted from
    // the later schedules, and from schedule 0 too when the build has no CUTLASS.
    m.volta_cutlass         = volta_cutlass_compiled();
    const bool cutlass_tier = !build_features().mma_tensor;
    add(Family::TiledMma, tiled_schedule_count(), [&](int i, ManifestSchedule& s) {
        for (const auto& [qtype, codec] : formats) {
            const bool tile = tiled_supports(codec) || codec == WeightCodec::Q8;
            if ((tile && !tiled_fits(i, codec)) ||
                (!tile && cutlass_tier && (i != 0 || !m.volta_cutlass))) {
                s.omitted_formats.push_back({qtype, 0});
            }
        }
    });
    add(Family::W4a8Mma, w4a8_schedule_count(), [&](int i, ManifestSchedule& s) {
        const bool producer = w4a8_schedule_info(i).producers > 0;
        int read            = 0;
        for (const auto& f : formats) {
            if (!w4a8_supports(f.first)) continue;
            ++read;
            if (producer && w4a8_ws_supports(f.first)) s.only_formats.push_back(f.first);
            if (!w4a8_fits(i, f.first)) s.omitted_formats.push_back({f.first, 0});
        }
        // Omitted for every format it reads (the producer tiles below 99 KiB): no kernel at all.
        s.compiled = static_cast<int>(s.omitted_formats.size()) < read;
    });
    add(Family::StagedMma, staged_schedule_count(),
        [](int i, ManifestSchedule& s) { s.compiled = staged_fits(i); });
    add(Family::Nvfp4TiledMma, nvfp4_tiled_schedule_count(),
        [](int, ManifestSchedule& s) { s.compiled = kNativeNvfp4; });
    add(Family::R8Mma, r8_schedule_count(), none);
    for (const Family family : {Family::GgmlMmq, Family::GgmlVec}) {
        add(family, 1, [&](int, ManifestSchedule& s) {
            for (const auto& f : formats) {
                Weight w{};
                w.qtype         = f.first;
                const auto type = mmq_type(w);
                if (type && (family == Family::GgmlMmq ? ggml_mmq::matrix_supported(*type)
                                                       : ggml_mmq::vec_supported(*type))) {
                    s.only_formats.push_back(f.first);
                }
            }
        });
    }
    // generic_i8.cu compiles its kernels from sm_80 on (NINFERNO_I8_KERNELS); below, empty bodies.
    add(Family::I8Mma, i8_schedule_count(), [&](int i, ManifestSchedule& s) {
        s.compiled = m.arch == 0 || m.arch >= 80;
        for (const QType q : {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16}) {
            if (!i8_fits(i, q)) s.omitted_formats.push_back({q, 0});
        }
    });
    for (const auto& f : formats) {
        if (r8_supports(f.first) && r8_verify_compiled(f.first)) {
            m.r8_verify_formats.push_back(f.first);
        }
    }
    return m;
}

FamilyCoverage staged_coverage(const Weight& w) {
    FamilyCoverage c;
    c.declared       = staged_schedule_count();
    const bool reads = w.qtype == QType::BF16; // the format alone, whatever the weight
    for (int i = 0; i < c.declared; ++i) {
        const ScheduleRefusal why = staged_refusal(i, w);
        c.schedules.push_back({i, why});
        c.compiled += reads;
        c.legal += why == ScheduleRefusal::None;
        if (c.heuristic < 0 && reads && staged_fits(i) && staged_device_fits(i)) {
            c.heuristic         = i;
            c.heuristic_refusal = why;
        }
    }
    return c;
}

FamilyCoverage w4a8_coverage(const Problem& problem, const Weight& w, std::int32_t tokens) {
    FamilyCoverage c;
    c.declared       = w4a8_schedule_count();
    const bool reads = w4a8_supports(w.qtype);
    for (int i = 0; i < c.declared; ++i) {
        const ScheduleRefusal why = w4a8_refusal(i, w, problem.policy);
        c.schedules.push_back({i, why});
        c.compiled += reads && (w4a8_schedule_info(i).producers == 0 || w4a8_ws_supports(w.qtype));
        c.legal += why == ScheduleRefusal::None;
    }
    c.heuristic         = heuristic_w4a8(problem, w, tokens);
    c.heuristic_refusal = w4a8_refusal(c.heuristic, w, problem.policy);
    return c;
}

} // namespace ninferno::ops::detail::generic
