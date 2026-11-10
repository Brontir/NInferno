// The host part of the generic path's selector (#620 step 11): schedule tables' accessors, the
// route names and their parser, the register-MMA variant rules and the format admission tests that
// read no device. Moved verbatim from generic_gemm.cpp, so it builds into ninferno_ops_host with no
// CUDA runtime.

#include "ops/generic/generic_gemm.h"
#include "ops/common/arch_limits.h"
#include "ops/common/native_nvfp4.h"
#include "ops/generic/format_traits.h"
#include "ops/generic/generic_i8_schedules.h"
#include "ops/generic/generic_schedule_tables.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_variant_rules.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <array>
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
    // Q2_K's two mins per block and weight row (W4a8BufferOf::kSmem); the producer ring is sized
    // for 99 KiB.
    if (s.producers > 0) return kMaxSharedBytesPerBlock >= 99 * 1024;
    const int half_min = qtype == QType::GGML_Q2_K ? s.block_n * 32 : 0;
    return 2 * ((s.block_m + s.block_n) * (128 + 32) + half_min) <=
           static_cast<int>(kMaxSharedBytesPerBlock);
}

int r8_schedule_count() { return static_cast<int>(kR8Schedules.size()); }

R8ScheduleInfo r8_schedule_info(int index) {
    return kR8Schedules.at(static_cast<std::size_t>(index));
}

bool r8_supports(QType qtype) { return admits(format_traits(qtype).r8, kNativeNvfp4); }

std::int32_t r8_k_values(QType qtype) { return format_traits(qtype).r8_k_values; }

bool r8_verify_supports(QType qtype) { return format_traits(qtype).r8_verify; }

std::int32_t r8_verify_widest(QType qtype) {
    return qtype == QType::Q8_G32_FP16 ? kR8VerifyTokens : 4;
}

bool r8_verify_compiled(QType qtype) {
    return r8_supports(qtype) && format_traits(qtype).r8_prequant;
}

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

bool w4a8_supports(QType qtype) { return admits(format_traits(qtype).w4a8, kNativeNvfp4); }

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
    // three stages where four do not fit (generic_i8.cu kI8Stages).
    const auto s     = i8_schedule_info(schedule);
    const int high   = i8_bits(qtype) == 4 ? 0 : i8_bits(qtype) == 5 ? 8 : 16;
    const auto ring  = static_cast<std::size_t>(2 * s.block_n * 16);
    const auto stage = (s.smem_bytes - ring) / static_cast<std::size_t>(s.stages) +
                       static_cast<std::size_t>(s.block_n) * high;
    return 3 * stage + ring <= kMaxSharedBytesPerBlock;
}

I8ScheduleInfo i8_schedule_info(int index) {
    if (index < 0 || index >= i8_schedule_count()) {
        throw std::out_of_range("generic I8: schedule index");
    }
    return kI8Schedules[index];
}

} // namespace ninferno::ops::detail::generic
