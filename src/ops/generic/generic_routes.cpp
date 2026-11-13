// The host part of the generic path's selector (#620 step 11): schedule tables' accessors, the
// route names and their parser, the register-MMA variant rules and the format admission tests that
// read no device. Moved verbatim from generic_gemm.cpp and, for the functions that sat in the .cu
// files (tiled_supports, volta_hmma_takes, volta_gemv_takes and their kin: tiled_fits and
// staged_fits read the shared-byte sizes of generic_shared_bytes.h), from those, so it builds into
// ninferno_ops_host with no CUDA runtime.

#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_shared_bytes.h"
#include "ops/common/arch_limits.h"
#include "ops/common/native_nvfp4.h"
#include "ops/generic/codec_traits.h"
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/format_traits.h"
#include "ops/generic/generic_i8_schedules.h"
#include "ops/generic/generic_schedule_tables.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_variant_rules.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::ops::detail::generic {

bool rmma_variant_occupancy_rule(std::int64_t grid_ctas, int sms, int variant_ctas,
                                 int plain_ctas) {
    if (variant_ctas <= 0 || plain_ctas <= 0 || sms <= 0) return false;
    if (variant_ctas >= plain_ctas) return true;
    const auto waves = [&](int resident) {
        const std::int64_t wave = static_cast<std::int64_t>(resident) * sms;
        return (grid_ctas + wave - 1) / wave;
    };
    return waves(variant_ctas) <= waves(plain_ctas);
}

bool rmma_wide_rule(double bytes_per_sm_cycle, int compute_capability, bool occupancy,
                    std::size_t wide_local_bytes, std::size_t plain_local_bytes) {
    return bytes_per_sm_cycle > 0.0 && bytes_per_sm_cycle < kRmmaWideBytes &&
           compute_capability >= 75 && occupancy &&
           wide_local_bytes <= plain_local_bytes + kRmmaWideSpillSlack;
}

bool rmma_plane_rule(std::size_t plane_local_bytes, std::size_t twin_local_bytes, int plane_ctas,
                     int twin_ctas) noexcept {
    return plane_ctas > 0 && twin_ctas > 0 && plane_ctas >= twin_ctas &&
           plane_local_bytes <= twin_local_bytes;
}

bool operator==(Route a, Route b) { return a.family == b.family && a.schedule == b.schedule; }

int rmma_schedule_count() { return static_cast<int>(kRmmaSchedules.size()); }

RmmaScheduleInfo rmma_schedule_info(int index) {
    return kRmmaSchedules.at(static_cast<std::size_t>(index));
}

bool rmma_schedule_launchable(int index) {
#ifdef NINFERNO_WITHOUT_NVFP4
    return rmma_schedule_info(index).cluster == 1;
#else
    return index >= 0 && index < rmma_schedule_count();
#endif
}

int w4a4_schedule_count() {
#ifdef NINFERNO_WITHOUT_NVFP4
    // Below sm_120a the W4A4 kernel TU is not compiled (kind::mxf4nvf4 is Blackwell-only), so no
    // W4A4 route is offered, admitted or run; NVFP4 weights take the register-MMA A16 path. The
    // names still parse (parse_route), so one route table serves every build.
    return 0;
#else
    return static_cast<int>(kW4a4Schedules.size());
#endif
}

int w4a4_schedule_names() { return static_cast<int>(kW4a4Schedules.size()); }

W4a4ScheduleInfo w4a4_schedule_info(int index) {
    return kW4a4Schedules.at(static_cast<std::size_t>(index));
}

int w4a8_schedule_count() { return static_cast<int>(kW4a8Schedules.size()); }

W4a8ScheduleInfo w4a8_schedule_info(int index) {
    return kW4a8Schedules.at(static_cast<std::size_t>(index));
}

bool w4a8_fits(int schedule, QType qtype) {
    const auto s = w4a8_schedule_info(schedule);
    // Two buffers of int8 codes and two floats per 32-value block for block_m + block_n rows, and
    // a half-min codec's (Q2_K, codec_traits.h) two mins per block and weight row
    // (W4a8BufferOf::kSmem); the producer ring is sized for 99 KiB.
    if (s.producers > 0) return kMaxSharedBytesPerBlock >= 99 * 1024;
    const int half_min = codec_traits(qtype).half_min ? s.block_n * 32 : 0;
    return 2 * ((s.block_m + s.block_n) * (128 + 32) + half_min) <=
           static_cast<int>(kMaxSharedBytesPerBlock);
}

int r8_schedule_count() { return static_cast<int>(kR8Schedules.size()); }

R8ScheduleInfo r8_schedule_info(int index) {
    return kR8Schedules.at(static_cast<std::size_t>(index));
}

bool r8_supports(QType qtype) { return compiled(Variant::R8, qtype); }

std::int32_t r8_k_values(QType qtype) { return format_traits(qtype).r8_k_values; }

bool r8_verify_supports(QType qtype) { return format_traits(qtype).r8_verify; }

std::int32_t r8_verify_widest(QType qtype) {
    return qtype == QType::Q8_G32_FP16 ? kR8VerifyTokens : 4;
}

bool r8_verify_compiled(QType qtype) { return compiled(Variant::R8Prequant, qtype); }

// K for the W4A8 tiles: whole 128-value tiles, or for a format of 32-value blocks any whole number
// of blocks, the last tile's blocks past K zero (weights not loaded, activation quantized as
// zeros). Gemma 4 26B's shared MLP down (K 2112) ran the BF16 register MMA before: RTX 5090 nsys,
// chunk 1024, 114 us a call against 102 us for the twice-as-large gate/up on W4A8.
bool w4a8_k_admits(QType qtype, std::int32_t k) {
    if (k <= 0) return false;
    if (k % 128 == 0) return true;
    const std::int32_t tail = format_traits(qtype).w4a8_tail_k;
    return tail != 0 && k % tail == 0;
}

bool w4a8_supports(QType qtype) { return compiled(Variant::W4a8, qtype); }

int nvfp4_tiled_schedule_count() { return static_cast<int>(kNvfp4TiledSchedules.size()); }

Nvfp4TiledScheduleInfo nvfp4_tiled_schedule_info(int index) {
    return kNvfp4TiledSchedules.at(static_cast<std::size_t>(index));
}

int tiled_schedule_count() { return static_cast<int>(kTiledSchedules.size()); }

TiledScheduleInfo tiled_schedule_info(int index) {
    return kTiledSchedules.at(static_cast<std::size_t>(index));
}

int staged_schedule_count() { return static_cast<int>(kStagedSchedules.size()); }

StagedScheduleInfo staged_schedule_info(int index) {
    return kStagedSchedules.at(static_cast<std::size_t>(index));
}

std::string route_name(Route route) {
    switch (route.family) {
    case Family::Scalar:
        return "scalar";
    case Family::RegisterMma: {
        const auto s = rmma_schedule_info(route.schedule);
        return "rmma.m" + std::to_string(s.m_tiles) + ".n" + std::to_string(s.n_tiles) + ".w" +
               std::to_string(s.warps_m) + "x" + std::to_string(s.warps_k) +
               (s.cluster > 1 ? ".c" + std::to_string(s.cluster) : std::string());
    }
    case Family::W4a4Mma: {
        const auto s = w4a4_schedule_info(route.schedule);
        return "w4a4." + std::to_string(s.block_m) + "x" + std::to_string(s.block_n);
    }
    case Family::TiledMma: {
        const auto s = tiled_schedule_info(route.schedule);
        return "tiled." + std::to_string(s.block_m) + "x" + std::to_string(s.block_n);
    }
    case Family::W4a8Mma: {
        const auto s = w4a8_schedule_info(route.schedule);
        return (s.producers > 0 ? "w4a8ws." : "w4a8.") + std::to_string(s.block_m) + "x" +
               std::to_string(s.block_n) +
               (s.producers > 0 ? ".p" + std::to_string(s.producers) : std::string());
    }
    case Family::StagedMma: {
        const auto s = staged_schedule_info(route.schedule);
        return "staged." + std::to_string(s.rows) + "x" + std::to_string(s.tokens) + ".w" +
               std::to_string(s.warps_m) + "x" + std::to_string(s.warps_n) + ".s" +
               std::to_string(s.stages);
    }
    case Family::Nvfp4TiledMma: {
        const auto s = nvfp4_tiled_schedule_info(route.schedule);
        return "nvfp4t.c" + std::to_string(s.classes) + "." + std::to_string(s.block_m) + "x" +
               std::to_string(s.block_n);
    }
    case Family::R8Mma: {
        const auto s = r8_schedule_info(route.schedule);
        return "r8.w" + std::to_string(s.warps) + "r" + std::to_string(s.rows);
    }
    case Family::GgmlMmq:
        return "ggml_mmq";
    case Family::GgmlVec:
        return "ggml_vec";
    case Family::I8Mma: {
        const auto s = i8_schedule_info(route.schedule);
        return "i8." + std::to_string(s.block_m) + "x" + std::to_string(s.block_n);
    }
    }
    return "unknown";
}

std::string_view family_name(Family family) noexcept {
    switch (family) {
    case Family::RegisterMma:
        return "rmma";
    case Family::Scalar:
        return "scalar";
    case Family::W4a4Mma:
        return "w4a4";
    case Family::TiledMma:
        return "tiled";
    case Family::W4a8Mma:
        return "w4a8";
    case Family::StagedMma:
        return "staged";
    case Family::Nvfp4TiledMma:
        return "nvfp4t";
    case Family::R8Mma:
        return "r8";
    case Family::GgmlMmq:
        return "ggml_mmq";
    case Family::GgmlVec:
        return "ggml_vec";
    case Family::I8Mma:
        return "i8";
    }
    return "unknown";
}

std::optional<Route> parse_route(std::string_view name) {
    if (name == "scalar") return Route{Family::Scalar, 0};
    if (name == "ggml_mmq") return Route{Family::GgmlMmq, 0};
    if (name == "ggml_vec") return Route{Family::GgmlVec, 0};
    for (int i = 0; i < rmma_schedule_count(); ++i) {
        const Route r{Family::RegisterMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    // Every W4A4 name, compiled or not: a shipped Blackwell table names them, and route_admits
    // refuses them on a build without the kernels.
    for (int i = 0; i < w4a4_schedule_names(); ++i) {
        const Route r{Family::W4a4Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < w4a8_schedule_count(); ++i) {
        const Route r{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < staged_schedule_count(); ++i) {
        const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < nvfp4_tiled_schedule_count(); ++i) {
        const Route r{Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < r8_schedule_count(); ++i) {
        const Route r{Family::R8Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < i8_schedule_count(); ++i) {
        const Route r{Family::I8Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    return std::nullopt;
}

int i8_schedule_count() { return static_cast<int>(std::size(kI8Schedules)); }

int i8_bits(QType q) { return q == QType::Q5_G64_FP16 ? 5 : q == QType::Q6_G64_FP16 ? 6 : 4; }

bool i8_fits(int schedule, QType qtype) {
    // The high plane takes 0 / 8 / 16 bytes a group and weight row in every stage; a tile runs
    // three stages where four do not fit (generic_i8.cu kI8Stages, generic_i8_schedules.h).
    const auto s = i8_schedule_info(schedule);
    return i8_smem_bytes(s.block_m, s.block_n, i8_bits(qtype), 3) <= kMaxSharedBytesPerBlock;
}

I8ScheduleInfo i8_schedule_info(int index) {
    if (index < 0 || index >= i8_schedule_count()) {
        throw std::out_of_range("generic I8: schedule index");
    }
    return kI8Schedules[index];
}

bool tiled_supports(WeightCodec codec) { return compiled(Variant::Tiled, codec); }

bool volta_hmma_codec(WeightCodec codec) { return compiled(Variant::VoltaHmma, codec); }

int volta_hmma_k_multiple(WeightCodec codec) { return codec == WeightCodec::Q8 ? 64 : 256; }

bool tiled_fits(int schedule, WeightCodec codec) {
    const int c = tiled_codec_index(codec);
    if (c < 0 || schedule < 0 || schedule >= static_cast<int>(kTiledSharedBytes[0].size())) {
        return false;
    }
    return static_cast<std::size_t>(tiled_shared_table(c, schedule)) <= kMaxSharedBytesPerBlock;
}

bool staged_fits(int schedule) {
    if (schedule < 0 || schedule >= static_cast<int>(kStagedSharedBytes.size())) { return false; }
    return static_cast<std::size_t>(kStagedSharedBytes[static_cast<std::size_t>(schedule)]) <=
           kMaxSharedBytesPerBlock;
}

bool volta_gemv_takes(WeightCodec codec, const GemmArgs& args) {
    if (args.lane_local == 0 || args.tokens < 1 || args.tokens > 4 || args.join_previous != 0) {
        return false;
    }
    // The kernel reads each token's x in 16-byte words.
    if ((args.ldx % 8) != 0 || (reinterpret_cast<std::uintptr_t>(args.x) % 16) != 0) return false;
    if (!compiled(Variant::VoltaGemv, codec)) return false;
    // BF16 rows in 16-byte words; the quantized formats in whole 256-value superblocks.
    return codec == WeightCodec::Bf16 ? (args.k % 8) == 0 : (args.k % 256) == 0;
}

bool volta_hmma_takes(WeightCodec codec, const GemmArgs& args) {
    // A join (Call::join_previous) needs programmatic dependent launch, which sm_70 does not have;
    // the segmented chain asks for one only after a call that released, so a later part of a
    // segmented projection takes this kernel (and the lane-local GEMV, which refuses a join) as a
    // separate call does.
    if (args.lane_local == 0 || args.tokens < kVoltaHmmaMinTokens ||
        args.tokens > kVoltaHmmaMaxTokens) {
        return false;
    }
    // Whole superblocks, or for Q8 G32 whole pairs of groups (gpt-oss's K = 2880: its LM head ran
    // the register-MMA tile on FP32 FMAs, 1.9 ms a token on the GV100); the staging reads four
    // bf16 values at a time.
    return volta_hmma_codec(codec) && (args.k % volta_hmma_k_multiple(codec)) == 0 &&
           (args.ldx % 4) == 0 && (reinterpret_cast<std::uintptr_t>(args.x) % 8) == 0;
}

Route r8_rules_route(const Problem& problem, std::int32_t tokens) {
    return Route{Family::R8Mma, static_cast<std::uint8_t>(rules_r8(problem, tokens))};
}

std::optional<RouteTraffic> route_traffic(Route route, const Weight& weight, const Problem& problem,
                                          std::int32_t tokens) {
    const auto tile = route_tile(route, problem.epilogue, tokens);
    if (!tile || problem.n <= 0 || problem.k <= 0 || tokens <= 0) return std::nullopt;
    const double n = problem.n, k = problem.k, t = tokens;
    const auto cdiv = [](double a, double b) { return std::ceil(a / b); };
    // The activation in the route's representation: bytes a value, the K and the tokens it holds.
    // BF16 for the A16 tiles; W4A8 one byte a value and an FP16 scale a 32 (and the block sum,
    // 40 bytes a 32, for a format with a min), K in whole 128-value tiles; I8 an FP32 scale and an
    // int32 sum a 64-value group, tokens padded to whole 256-token tiles.
    double x_bytes = 2.0, x_k = k, x_t = t;
    const QType q      = problem.qtype;
    const bool has_min = q == QType::GGML_Q4_1 || q == QType::GGML_Q5_1 || q == QType::GGML_Q4_K ||
                         q == QType::GGML_Q5_K;
    if (route.family == Family::W4a8Mma) {
        x_bytes = (has_min ? 40.0 : 36.0) / 32.0;
        x_k     = cdiv(k, 128.0) * 128.0;
    } else if (route.family == Family::I8Mma) {
        x_bytes = 1.0 + 8.0 / 64.0;
        x_t     = i8_padded_tokens(tokens);
    }
    const double weight_reads = cdiv(t, tile->tokens);
    // The register MMA holds its operands in registers: each of a CTA's row warps loads the
    // activation slice itself (the shared-memory tiles stage it once for the CTA).
    const double x_readers =
        route.family == Family::RegisterMma ? rmma_schedule_info(route.schedule).warps_m : 1.0;
    const double bpw = bytes_per_weight(weight, problem);
    return RouteTraffic{n * k * bpw * weight_reads,
                        x_t * x_k * x_bytes * cdiv(n, tile->out_rows) * x_readers, n * t * 2.0};
}

std::optional<PreparedRule> prepared_rule(const ResolvedRoute& route, const Weight& weight) {
    using R = PreparedRepresentation;
    switch (route.executed.family) {
    case Family::GgmlVec:
        return PreparedRule{R::GgmlQ8_1};
    case Family::W4a8Mma:
        return PreparedRule{R::W4a8Blocks, w4a8_ld(weight.k)};
    case Family::R8Mma:
        if (!route.r8_prequantized) { return std::nullopt; } // quantized inside the kernel
        return PreparedRule{R::W4a8Blocks, weight.k};
    case Family::I8Mma:
        return PreparedRule{R::I8Blocks64};
    case Family::GgmlMmq:
        return PreparedRule{R::GgmlQ8_1Mmq, 0, static_cast<std::int32_t>(weight.qtype)};
    case Family::W4a4Mma:
        return PreparedRule{R::Nvfp4, 0, 0, true};
    case Family::RegisterMma:
    case Family::Scalar:
    case Family::TiledMma:
    case Family::StagedMma:
    case Family::Nvfp4TiledMma:
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<double> route_cost_seconds(const Problem& problem, const Weight& w, Route route,
                                         std::int32_t tokens) {
    std::optional<double> out;
    price_routes(problem, w, tokens, [&](Route r, double time) {
        if (r.family == route.family && r.schedule == route.schedule) out = time;
    });
    return out;
}

} // namespace ninferno::ops::detail::generic
