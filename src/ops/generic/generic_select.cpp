#include "ops/generic/generic_gemm.h"
#include "ops/generic/format_traits.h"
#include "core/per_device.h"
#include "ops/common/arch_limits.h"
#include "ops/common/native_nvfp4.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/measured_boards.h"
#include "core/tensor.h"
#include "core/layout.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/route_probe.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_i8_schedules.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/generic_schedule_tables.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/generic/route_table.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/route_pins.h"


#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::ops::detail::generic {

namespace {

int run_values(WeightCodec codec) {
    switch (codec) {
    case WeightCodec::Bf16:
    case WeightCodec::Q8:
    case WeightCodec::U4ZpG32:
        return 8;
    case WeightCodec::Fp8:
    case WeightCodec::Fp8RowF32:
    case WeightCodec::Fp8B128:
    case WeightCodec::Nvfp4:
    case WeightCodec::Q4:
    case WeightCodec::Q5:
    case WeightCodec::Q6:
    case WeightCodec::U4ZpG64:
    case WeightCodec::Q3G128:
    case WeightCodec::Q3G64:
        return 16;
    case WeightCodec::Mxfp4:
    case WeightCodec::U4Zp:
    case WeightCodec::GgmlQ4K:
    case WeightCodec::GgmlQ5K:
    case WeightCodec::GgmlQ6K:
    case WeightCodec::GgmlQ3K:
    case WeightCodec::GgmlQ2K:
    case WeightCodec::GgmlIq4Xs:
    case WeightCodec::GgmlIq3Xxs:
    case WeightCodec::GgmlIq3S:
    case WeightCodec::GgmlIq2Xxs:
    case WeightCodec::GgmlIq2Xs:
    case WeightCodec::GgmlIq2S:
    case WeightCodec::GgmlIq1S:
    case WeightCodec::GgmlIq1M:
    case WeightCodec::GgmlTq10:
    case WeightCodec::GgmlTq20:
        return 32;
    // A K step is four 32-value blocks, each lane holding a slice of all four: K must be whole
    // steps for the register MMA (the scalar path takes any K of whole blocks).
    case WeightCodec::GgmlQ40:
    case WeightCodec::GgmlQ41:
    case WeightCodec::GgmlQ50:
    case WeightCodec::GgmlQ51:
    case WeightCodec::GgmlIq4Nl:
    case WeightCodec::GgmlQ10:
    case WeightCodec::GgmlQ20:
        return 128;
    }
    return 16;
}


} // namespace

bool allows_w4a4(const Weight& w, LinearPolicy policy) {
    return w.qtype == QType::NVFP4 && allows_a4(policy) && std::isfinite(w.input_scale_divisor) &&
           w.input_scale_divisor > 0.0F;
}

namespace {

bool w4a8_policy_allows(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) &&
           (w.qtype != QType::NVFP4 || nvfp4_prefill_a8_policy(w.qtype, policy));
}


} // namespace

bool allows_w4a8(const Weight& w, LinearPolicy policy) {
    return w4a8_policy_allows(w, policy) && w4a8_supports(w.qtype) && w4a8_k_admits(w.qtype, w.k) &&
           aligned_to(w.qdata, 16);
}

bool i8_format(QType qtype) { return format_traits(qtype).qx_g64; }

bool i8_planes_aligned(const Weight& w) {
    const bool q4 = w.qtype == QType::Q4_G64_FP16, q5 = w.qtype == QType::Q5_G64_FP16;
    return aligned_to(w.qdata, 16) && aligned_to(w.scales, 16) &&
           (q4 || aligned_to(w.qhigh, q5 ? 8 : 16));
}

bool i8_extent(const Weight& w) { return (w.k % 512) == 0 && (w.padded_shape[1] % 512) == 0; }

bool w4a8_ws_admits(const Weight& w) {
    return w4a8_ws_supports(w.qtype) && (w.qtype != QType::GGML_IQ3_XXS || aligned_to(w.qhigh, 8));
}

bool allows_r8(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) && r8_supports(w.qtype) && (w.k % r8_k_values(w.qtype)) == 0 &&
           aligned_to(w.qdata, 16);
}

bool r8_call(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return r8_call_refusal(w, policy, tokens) == R8CallRefusal::None;
}

MmqPlaneBytes mmq_plane_bytes(QType q) {
    if (q == QType::GGML_Q4_K) return {144, 0, 0};
    if (q == QType::GGML_Q5_K) return {176, 0, 0};
    const auto p = ggml_plane_bytes(q);
    return {static_cast<int>(p.code), static_cast<int>(p.high), static_cast<int>(p.scale)};
}

namespace {

bool ggml_layout_admits(const Weight& w) {
    const bool planes = w.qtype != QType::GGML_Q4_K && w.qtype != QType::GGML_Q5_K;
    if (planes != (w.layout == QuantLayout::GgmlPlanes)) return false;
    const auto bytes = mmq_plane_bytes(w.qtype);
    if ((bytes.high > 0 && w.qhigh == nullptr) || (bytes.scales > 0 && w.scales == nullptr)) {
        return false;
    }
    // ggml's loaders read every plane in vector words: a sign or scale plane off a 16-byte
    // boundary (a view into a packed buffer) takes another route.
    return aligned_to(w.qdata, 16) && (bytes.high == 0 || aligned_to(w.qhigh, 16)) &&
           (bytes.scales == 0 || aligned_to(w.scales, 16));
}

bool mmq_prefill_device() { return measured::mmq_prefill_device_pays(); }


} // namespace

std::int32_t vec_max_tokens() {
    try {
        const auto& f = device_class().features;
        return f.mma_tensor && !f.mma_slow ? ggml_mmq::kVecMaxColumns
                                           : 2 * ggml_mmq::kVecMaxColumns;
    } catch (const std::exception&) { return ggml_mmq::kVecMaxColumns; }
}

namespace {

constexpr std::int32_t kVecRuleMaxTokens = 4;

std::int32_t vec_rule_max_tokens() {
    // vec_max_tokens() reaches past eight columns only without tensor mma.sync.
    const std::int32_t widest = vec_max_tokens();
    return widest > ggml_mmq::kVecMaxColumns ? widest : kVecRuleMaxTokens;
}


} // namespace

bool vec_rule(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 2 && tokens <= vec_rule_max_tokens() && vec_call(w, policy, tokens);
}

namespace {

bool mmq_verify_format(QType q) { return format_traits(q).mmq_verify; }

constexpr std::int32_t kMmqVerifyMinTokens = 9;

bool mmq_verify_rule(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    // The measured band is 9..16 tokens of several sequences (a C4 verify). One sequence keeps its
    // register MMA at every width (ops::round_lanes): a C1 MTP round of 5..8 tokens and a C1 copy
    // round of 16 lost on mmq (RTX 5070 Ti C1 117.0 -> 115.2 tok/s, all of it in the copy rounds:
    // with them off, 116.5 both ways), while the C4 verify gains (233.5 -> 249.3).
    // Where the BF16 MMA is emulated (Turing) a verify of 5..8 tokens keeps mmq for the codebook
    // formats only: the affine ones (the K-quants, Q4_0) dequant cheaply on the register MMA (RTX
    // 2080 Ti, one call each, T 5..8: Q3_K-Q6_K 10-58 % faster off mmq; Gemma 4 12B Q4_K_M MTP5
    // 116.7 -> 136.1 tok/s), while the codebook IQ4_XS lost 23-38 % there.
    if (!bf16_mma_native())
        return mmq_call(w, policy, rows, tokens) &&
               (tokens >= kMmqVerifyMinTokens || mmq_verify_format(w.qtype));
    return mmq_call(w, policy, rows, tokens) && tokens >= kMmqVerifyMinTokens &&
           ops::round_lanes() != 1 && mmq_verify_format(w.qtype);
}

int sm_count() {
    try {
        return device_class().sms;
    } catch (const std::exception&) {
        return 1;
    }
}


} // namespace

int rmma_index(int warps_m, int warps_k, int m_tiles, int n_tiles, int cluster) {
#ifdef NINFERNO_WITHOUT_NVFP4
    if (cluster > 1) {
        // Thread-block clusters are sm_90+; the Ada build compiles C == 1 schedules only and
        // the selector must never offer a cluster split there.
        cluster = 1;
    }
#endif
    for (std::size_t i = 0; i < kRmmaSchedules.size(); ++i) {
        const auto& s = kRmmaSchedules[i];
        if (s.warps_m == warps_m && s.warps_k == warps_k && s.m_tiles == m_tiles &&
            s.n_tiles == n_tiles && s.cluster == cluster) {
            return static_cast<int>(i);
        }
    }
    return 0;
}

namespace {

bool light_ggml_step(QType q) { return format_traits(q).light_step; }

int one_wave_split([[maybe_unused]] std::int64_t ctas, [[maybe_unused]] std::int64_t sms) {
#ifdef NINFERNO_WITHOUT_NVFP4
    return rmma_index(1, 16, 1, 1);
#else
    return 4 * ctas <= sms ? rmma_index(1, 8, 1, 1, 8) : rmma_index(1, 8, 1, 1, 2);
#endif
}

double speed_waves(std::int64_t n, std::int32_t tokens, int rows, int cols) {
    const std::int64_t tiles = ((n + rows - 1) / rows) * ((tokens + cols - 1) / cols);
    return static_cast<double>(tiles) / static_cast<double>(sm_count());
}

bool heavy_decode(QType q) { return q != QType::BF16; }

bool tall_tile_format(QType q) { return format_traits(q).tall_tile; }

inline constexpr double kRmmaQ6kFourWarpBytes = 4.0;

bool ring_swiglu_four_warps(QType q) {
    // Q4_K and Q5_K: the formats measured (Q3_K and Q6_K SwiGLU keep eight warps until they are).
    if (q != QType::GGML_Q4_K && q != QType::GGML_Q5_K) return false;
    try {
        const auto& d = device_class();
        return turing_cc(d.major, d.minor) && dram_bytes_per_sm_cycle() >= kRmmaLatencySwiGluBytes;
    } catch (const std::exception&) {
        return false;
    }
}

std::int64_t rmma_k_step(QType qtype, std::int32_t k) {
    Weight weight;
    weight.qtype = qtype;
    WeightCodec codec{};
    try {
        codec = codec_of(weight);
    } catch (const std::invalid_argument&) {
        return 128;
    }
    const bool quad_u4 = (codec == WeightCodec::U4ZpG32 || codec == WeightCodec::U4ZpG64) &&
                         k > 0 && (k % 128) == 0;
    const int run = quad_u4 ? 32 : run_values(codec);
    return run >= 128 ? run : 4 * run;
}

int rmma_k_split_fit(int index, const Problem& problem, std::int64_t ctas, std::int64_t sms) {
    if (problem.epilogue == Epilogue::SwiGlu || sms <= 0 || ctas <= sms) return index;
    const RmmaScheduleInfo info = rmma_schedule_info(index);
    if (info.warps_m != 1 || info.m_tiles != 1 || info.n_tiles != 1 || info.cluster != 1) {
        return index;
    }
    const std::int64_t step  = rmma_k_step(problem.qtype, problem.k);
    const std::int64_t steps = (static_cast<std::int64_t>(problem.k) + step - 1) / step;
    const auto refills = [&](int warps_k) { return (steps + warps_k - 1) / warps_k > 2; };
    // The warps a SM holds at 128 registers a thread (kRmmaMinBlocks).
    constexpr std::int64_t kSlotWarps = 65536 / (32 * 128);
    if (info.warps_k <= 4 || refills(info.warps_k) || !refills(4) || ctas * 4 < kSlotWarps * sms) {
        return index;
    }
    return rmma_index(1, 4, 1, 1);
}

bool tiled_speed_pays(std::int32_t n, std::int32_t tokens) {
    return tokens >= kTiledSpeedMinTokens && bf16_mma_native() &&
           speed_waves(n, tokens, 128, 128) >= kSpeedFill;
}

int invariant_rmma(std::int32_t k, std::int32_t tokens) {
    const int n_tiles = tokens <= 8 ? 1 : 4;
    if (k >= 1024) return rmma_index(1, 8, 1, n_tiles);
    if (k >= 512) return rmma_index(1, 4, 1, n_tiles);
    return tokens <= 8 ? rmma_index(4, 1, 1, 1) : rmma_index(4, 1, 1, 8);
}

int heuristic_w4a4(std::int32_t rows, std::int32_t tokens, bool prefill_band) {
    if (prefill_band) return w4a4_schedule_count() - 1;
    const int sms   = sm_count();
    const auto grid = [&](int block_m, int block_n) {
        return ((rows + block_n - 1) / block_n) * ((tokens + block_m - 1) / block_m);
    };
    if (grid(128, 128) >= sms) return 3;
    if (grid(64, 128) >= sms) return 2;
    if (grid(32, 128) >= sms) return 1;
    return 0;
}


} // namespace

bool w4a8_ws_extra(QType q) {
    if (!format_traits(q).w4a8_ws_tuned) return false;
    static PerDevice<bool> on;
    return on.get([](int) {
        try {
            const auto& d = device_class();
            return gb20x_cc(d.major, d.minor) || ada_cc(d.major, d.minor);
        } catch (const std::exception&) {
            return false;
        }
    });
}

namespace {

bool w4a8_ws_wins(const Problem& p, const Weight& w) {
    if (!w4a8_ws_admits(w)) return false;
    bool blackwell = false, ada = false, ampere = false;
    try {
        const auto& d = device_class();
        blackwell     = gb20x_cc(d.major, d.minor);
        ada           = ada_cc(d.major, d.minor);
        ampere        = ampere_cc(d.major, d.minor);
    } catch (const std::exception&) {
        return false;
    }
    // Ampere, sm_80 and sm_86 alike (measured on an RTX 3090, ga102: the tile forced on every
    // format it reads, one 16K prompt, D WS
    // WS D): Muse 30B Q4_K_M prefill 1278 / 1273 -> 1398 / 1399 t/s, Qwen3.8 27B UD-Q4_K_XL 1290 /
    // 1289 -> 1314 / 1314, Gemma 4 31B 1071 -> 1081 / 1081; no format lost.
    if (ampere) return true;
    switch (p.qtype) {
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ2_S:
        return blackwell || ada;
    // Q6_K on sm_89 lost once the 128-token tiles run token-major there (kW4a8TokenMajor): RTX
    // 4090, Muse 30B Q4_K_M (Q6_K Store/AddResidual is its only producer-warp format on sm_89),
    // 16K prompt: 2935 / 2931 t/s with the tile against 2997 / 3001 without (+2.2 %).
    case QType::GGML_Q6_K:
        return blackwell;
    case QType::GGML_Q4_K:
        return blackwell;
    case QType::GGML_Q4_0:
        return ada && p.n >= 5376 && p.k >= 5376;
    // ISTA GSQ IQ3_S mix, 24K prefill against ninfer-all's MMQ (RTX 5060 Ti nsys, same
    // tokens): the classic-tile formats carried the whole gap - IQ4_XS 16.1 s against 10.99,
    // Q2_K 3.10 / 2.51, IQ2_XS 2.81 / 2.32, IQ2_XXS 1.00 / 0.79 - while the producer-warp formats
    // were even (IQ3_XXS 12.0 / 11.8, IQ2_S 3.96 / 3.67). In the product (24K prompt, A B B A,
    // greedy byte-identical): RTX 5060 Ti 981 -> 1011 t/s with IQ4_XS / IQ2_XS / IQ2_XXS;
    // RTX 4060 Ti 900 -> 901 with those, 926 with IQ3_S on its shared-table codec too.
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ3_S:
        return w4a8_ws_extra(p.qtype);
    default:
        return false;
    }
}

bool w4a8_wide_q4k(const Problem& p, std::int32_t tokens) {
    if (p.qtype != QType::GGML_Q4_K) return false;
    try {
        const auto& d = device_class();
        if (!ada_cc(d.major, d.minor)) return false;
        const std::int64_t rows_per_block = p.epilogue == Epilogue::SwiGlu ? 64 : 128;
        const std::int64_t ctas =
            ((p.n + rows_per_block - 1) / rows_per_block) * ((tokens + 127) / 128);
        return 4 * ctas >= 3 * static_cast<std::int64_t>(d.sms);
    } catch (const std::exception&) {
        return false;
    }
}

int r8_index(int warps, int rows) {
    for (std::size_t i = 0; i < kR8Schedules.size(); ++i) {
        if (kR8Schedules[i].warps == warps && kR8Schedules[i].rows == rows) {
            return static_cast<int>(i);
        }
    }
    return 0;
}

constexpr bool r8_one_slot_build() {
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
    return true;
#else
    return false;
#endif
}


} // namespace

int rules_r8(const Problem& p, std::int32_t tokens) {
    if (p.qtype == QType::Q8_G32_FP16) return r8_index(4, 1);
    // IQ1_S / IQ1_M: grid-lookup decoders of IQ2's weight, scheduled with them (sched_iq2).
    const bool iq2 = format_traits(p.qtype).sched_iq2;
    if (tokens >= 2) return iq2 && (tokens >= 4 || p.n >= 34816) ? r8_index(4, 2) : r8_index(4, 1);
    // Turing (the RTX 2080 Ti, tu102:68sm, ninferno_r8_schedule_bench on the IQ3 16 GB
    // artifacts' 63 problems: picks +6.25 % above the best schedules in sum) decodes the codebook
    // formats and Q3_K slower per word, and from K = 5120 eight warps of two rows win at every N
    // and epilogue (IQ3_XXS 17408x5120 w4r1 110.2 -> w8r2 84.5 us, IQ3_S 16384x5120 131.3 -> 114.2,
    // Q3_K 5120x17408 add 112.9 -> 89.0); Q6_K gains two rows per warp from N = 8192 or K = 4096
    // (the 248320x2048 lm_head 807 -> 723 us, 2816x8192 w8r2 42.3 -> w4r2 40.3).
    if (!device_class().features.mma_bf16_native) {
        const bool decode_heavy = format_traits(p.qtype).decode_heavy;
        if (decode_heavy && p.k >= 5120) return r8_index(8, 2);
        // Volta (an sm_70 build, whose R8 kernel is the one-slot form: generic_r8.cuh
        // NINFERNO_R8_ONE_SLOT; a GPU without mma.sync) keeps one row per warp for Q6_K at every N
        // and K, where the two-slot form took two. With one slot a second row per warp raises the
        // registers from 137 to 213 (3 CTAs per SM to 2) and the code a cold SM fetches first from
        // 960 to 1480 instructions. GV100 with ten other kernels between launches, as in a model
        // step: 4096x12288 add w4r1 88.8 against w4r2 94.7 us, 8192x4096 store 57.9 against 63.3.
        if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu && r8_one_slot_build() &&
            !device_class().features.mma_tensor)
            return r8_index(4, 1);
        if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu &&
            (p.n >= 8192 || p.k >= 4096))
            return r8_index(4, 2);
    }
    if (p.qtype == QType::GGML_Q3_K && p.k >= 8192) return r8_index(4, 1);
    if (p.k >= 8192 || (iq2 && p.n >= 12288)) return r8_index(8, 2);
    if (p.epilogue != Epilogue::SwiGlu && p.k >= 5120 &&
        consumer_blackwell_major(device_class().major)) {
        if (iq2 && p.n >= 5120) return r8_index(8, 2);
        // An SM-count rule, its edge between the measured boards: at 96 SMs or fewer (the
        // 5080's 84) IQ3 takes the wider warps; at 170 (the 5090) it keeps four single-row
        // warps: 10240x5120 IQ3_S w4r1 18.4 us against w8r2 20.3, IQ3_XXS 16.0 against w4r2
        // 18.0; IQ2 still gains (IQ2_S 10240x5120 w4r2 29.9 -> w8r2 20.1 us).
        if (device_class().sms <= 96) {
            if (p.qtype == QType::GGML_IQ3_S && p.n >= 5120) return r8_index(8, 2);
            if (p.qtype == QType::GGML_IQ3_XXS && p.n >= 5120) return r8_index(4, 2);
        }
    }
    if (iq2 && p.n >= 4096) return r8_index(4, 2);
    return r8_index(4, 1);
}

namespace {

constexpr double kR8CarryDistance = 0.5;

const R8DecodeTable* r8_decode_table() {
    static PerDevice<const R8DecodeTable*> table;
    try {
        return table.get([](int) {
            return nearest_known_measured(std::span(kR8DecodeTables), board_of(device_class()),
                                          kR8CarryDistance);
        });
    } catch (const std::exception&) {
        return nullptr;
    }
}

std::optional<int> tuned_r8(const Problem& p) {
    const R8DecodeTable* table = r8_decode_table();
    if (table == nullptr) return std::nullopt;
    const Epilogue epilogue = p.epilogue == Epilogue::AddResidual ? Epilogue::Store : p.epilogue;
    for (const R8DecodeEntry& e : table->entries) {
        if (e.qtype == p.qtype && e.epilogue == epilogue && e.n == p.n && e.k == p.k) {
            return r8_index(e.warps, e.rows);
        }
    }
    return std::nullopt;
}


} // namespace

bool w4a8_ws_supports(QType q) { return format_traits(q).w4a8_ws; }

int heuristic_rmma(const Problem& problem, std::int32_t tokens, bool prefill_band) {
    const std::int32_t k = problem.k;
    const int split      = k >= 1024 ? 8 : k >= 512 ? 4 : 1;
    const auto narrow    = [&] {
        if (split == 8) {
            if (tokens <= 8) {
                const std::int64_t ctas = (static_cast<std::int64_t>(problem.n) + 15) / 16;
                const std::int64_t sms  = sm_count();
                if (ctas <= sms) return one_wave_split(ctas, sms);
                if (ctas > 2 * sms && light_ggml_step(problem.qtype) &&
                    problem.epilogue != Epilogue::SwiGlu) {
                    return rmma_k_split_fit(rmma_index(1, 4, 1, 1), problem, ctas, sms);
                }
                if (ctas > 2 * sms && problem.epilogue == Epilogue::SwiGlu &&
                    ring_swiglu_four_warps(problem.qtype)) {
                    return rmma_index(1, 4, 1, 1);
                }
                if (ctas <= 2 * sms && problem.qtype == QType::GGML_Q6_K &&
                    problem.epilogue != Epilogue::SwiGlu) {
                    double bytes = 0.0; // a device whose class cannot be read: eight warps
                    try {
                        bytes = dram_bytes_per_sm_cycle();
                    } catch (const std::exception&) {
                    }
                    if (bytes > 0.0 && bytes < kRmmaQ6kFourWarpBytes) {
                        return rmma_k_split_fit(rmma_index(1, 4, 1, 1), problem, ctas, sms);
                    }
                }
                return rmma_k_split_fit(rmma_index(1, 8, 1, 1), problem, ctas, sms);
            } else if (tokens <= 16) {
                // 9..16 tokens (verify of an 8- or 16-token draft): one 8-token tile per CTA would
                // put two or three CTA columns on the token axis, each streaming and decoding the
                // whole weight. Four K-split warps with two tiles cover 16 tokens per column: 1.00-1.12
                // of the best of seven schedules against 1.24-1.58 for w1x8 n1 (geomean per format,
                // RTX 5080, Q8/Q4_0/IQ4_XS/Q4_K/Q5_K/Q6_K/IQ3_XXS at 27B/31B/35B
                // shapes; at T = 24 Q8 1.35 against 1.68).
                // A call of at most one CTA per SM through T = 16 takes the decode split of K
                // above (two CTA columns): the four-warp tiles leave SMs idle on that grid. The
                // dense tuner's timings of nine cards (sm_70 to sm_120) had the rule above lose
                // such calls (n = 48..2048 rows: K/V, gates, small outputs) by x1.4..2.1 (median
                // per card) to the sixteen-warp split; on the RTX 5090 the CTA pair or cluster of
                // eight was 0..17 % faster still (n = 48 and 1024, T = 12 and 16: 8..20 us against
                // 16..41 for the rule).
                const std::int64_t ctas = (static_cast<std::int64_t>(problem.n) + 15) / 16;
                const std::int64_t sms  = sm_count();
                if (ctas <= sms) return one_wave_split(ctas, sms);
                return rmma_index(1, 4, 1, 2);
            }
            return rmma_index(1, 8, 1, 1);
        }
        if (split == 4) return rmma_index(1, 4, 1, 1);
        return rmma_index(4, 1, 1, 1);
    };
    if (prefill_band) {
        // 64 tokens per CTA column. A narrower band (a chunk's tail) pads that column: through 32
        // tokens 32 per column cover it with the same grid and half the MMA work, and through 64
        // two columns of 32 while one column leaves SMs idle (fewer than two CTAs per SM). RTX 4080
        // Super (ad103:80sm, the dense tuner's prefill bands, A16 IQ1_S, IQ1_M and Q2_K at 5120 to
        // 17408 rows, time of m1.n8 over the best): T = 17 and 32 1.48-3.21 on every problem; T = 64
        // 1.20-1.58 below 160 CTAs, 1.00 at 272. RTX 5090 (gb202:170sm), Q6_K A16 output head
        // (248320 rows): m1.n4 the best at T = 17 (1.22) and 32 (1.54), m1.n8 at 64. RTX 5060 Ti
        // (gb206:36sm, the same artifacts as the 4080 SUPER): T = 17 and 32 1.60-2.14 on every
        // problem. In the product (ninferno-perplexity at context 32, tuner off on both arms): 4080
        // SUPER Qwen3.8-27B IQ3 246.6 -> 252.9 tok/s, the 5090's Q4_K_XL even.
        const std::int64_t ctas = (static_cast<std::int64_t>(problem.n) + 63) / 64;
        const bool narrow_band = tokens <= 32 || (tokens <= 64 && ctas < 2 * sm_count());
        return rmma_index(4, 1, 1, narrow_band ? 4 : 8);
    }
    if (tokens <= 16 || (split != 8 && tokens < kRmmaMidMinTokens)) return narrow();
    // From 17 tokens outside the prefill profile (a batched speculative verify: C4 x MTP3, C8 x
    // DFlash; a DFlash drafter's context) one rule for every board, from the call's shape, the
    // SMs and whether the BF16 MMA is native: the largest tile whose grid still fills
    // kSpeedFill waves of the SMs (speed_waves), else a narrower token tile, else K split over
    // warps. ninferno_route_tune's timings of seven boards (RTX 5090, 4090, 3090, 2080 Ti, A10,
    // L4, A100: Qwen3.5/3.8 and ISTA shapes in seven formats, n 48..248320, T 17..2048), the rule
    // fitted on six and scored on the seventh: 1.116 of the best (geomean) against 1.219 for the
    // bands by T and CTAs per SM it replaced; a separate rule per board gained 1.1 % at most, so
    // a board nobody measured takes the same tiles. What moved the best tile besides the waves
    // was the format's kernel, not the board (heavy_decode, tall_tile_format).
    const bool emulated  = !bf16_mma_native();
    const bool heavy     = heavy_decode(problem.qtype) || emulated;
    const std::int64_t n = problem.n;
    const auto waves     = [&](int rows, int cols) { return speed_waves(n, tokens, rows, cols); };
    // Q4_K: 128 rows by 32 tokens, half the activation loads per MMA of the 64x64 tile, 10..20 %
    // faster on every board where its tiles fill the SMs.
    if (!emulated && tokens >= kRmmaMidMinTokens && tall_tile_format(problem.qtype) &&
        waves(128, 32) >= kSpeedFill) {
        return rmma_index(4, 1, 2, 4);
    }
    // Four row warps of 64 rows and the widest token tile of under twice the call's tokens that
    // fills the SMs. Where the BF16 MMA is emulated the token tiles stop at 32, and that path
    // takes this regime only past the stream band.
    const int nt_cap = emulated ? 4 : 8;
    if (tokens > kStreamMaxTokens ||
        (!emulated && tokens >= kRmmaMidMinTokens && waves(64, 8 * nt_cap) >= kSpeedFill)) {
        for (int nt = nt_cap; nt > 1; nt /= 2) {
            if (8 * nt < 2 * std::max(8, static_cast<int>(tokens)) &&
                waves(64, 8 * nt) >= kSpeedFill) {
                return rmma_index(4, 1, 1, nt);
            }
        }
        return rmma_index(4, 1, 1, 1);
    }
    // The stream regime: one 16-row M tile per CTA with K over the warps, as in decode, and as
    // many 8-token tiles (4, 2, 1) as the call's tokens use while the grid fills the SMs. With one
    // token tile sixteen K warps where the call has at most one CTA per SM or a long K (from 72
    // K steps), else eight.
    const int pad = static_cast<int>(std::bit_ceil(static_cast<unsigned>((tokens + 7) / 8)));
    if (pad >= 4 && waves(16, 32) >= kSpeedFill) {
        return heavy ? rmma_index(1, 4, 1, 4) : rmma_index(1, 8, 1, 4);
    }
    if (pad >= 2 && waves(16, 16) >= kSpeedFill) return rmma_index(1, 4, 1, 2);
    const std::int64_t step  = heavy ? 128 : 32;
    const std::int64_t steps = (k + step - 1) / step;
    const std::int64_t ctas  = (n + 15) / 16;
    return ctas <= sm_count() || 2 * steps >= 9 * 16 ? rmma_index(1, 16, 1, 1)
                                                     : rmma_index(1, 8, 1, 1);
}

int heuristic_w4a8(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    constexpr int k32x128 = 0, k128x128 = 2, k128x64 = 3, k64x64 = 4, k128x64ws = 5;
    // 17..32 tokens (verify of a 16-token draft, short prompts): 32 tokens by 128 rows wastes less of
    // the tile than 64x64 and reads each weight tile once per 128 rows. (RTX 5080,
    // 7 GGUF formats at 27B/31B/35B shapes, time over the best of seven routes, geomean per format)
    // at T = 17: 32x128 1.05-1.45 against 64x64 1.14-1.57, better for every format at 17, 20, 24, 32.
    if (tokens <= 32) return k32x128;
    if (tokens <= 64) return k64x64;
    // From 65 tokens a 128-token tile reads each weight tile once where 64x64 reads it twice, but
    // it halves the grid: it pays while that grid still gives half the SMs a CTA (rows split as
    // the kernel does, SwiGLU's gate and up pairs halving a tile's output rows). The dense tuner's
    // prefill bands (perplexity at context 2048 on Qwen3.8-27B shapes, time of the old choice
    // over the best), 64x64 against 128x64 below that grid: RTX 4080 SUPER (ad103:80sm) at 1024
    // rows (16 CTAs at T = 128, 32 at 256) 1.33-1.37 for Q4_K, Q5_K and IQ4_XS; RTX 5090
    // (gb202:170sm) at 5120 rows and T = 128 (80 CTAs) 1.15-1.36 for Q4_K, Q5_K and IQ4_XS, at 1024
    // rows and T = 128..512 (16..64 CTAs) 1.14-1.35 for Q4_K and Q5_K, 1.07 for Q6_K; RTX 5060 Ti
    // (gb206:36sm) at 1024 rows and T = 128 1.19-1.50 for Q4_K, Q5_K and IQ4_XS.
    const std::int64_t rows_per_cta = problem.epilogue == Epilogue::SwiGlu ? 32 : 64;
    const std::int64_t wide_ctas =
        ((problem.n + rows_per_cta - 1) / rows_per_cta) * ((tokens + 127) / 128);
    if (2 * wide_ctas < sm_count()) return k64x64;
    bool wide_format = false;
    switch (problem.qtype) {
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ2_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
        wide_format = true;
        break;
    default:
        break;
    }
    // 65..127 tokens: the formats that take a 128-token tile from 128 tokens take 128x64 already,
    // and so do IQ2_XXS and Q3_K, whose wider tiles were not measured in the product. The same
    // 4080 SUPER: 128x64 against 64x64 1.10-1.56 on IQ2_*, IQ3_*, IQ4_XS from 5120 rows (80+ CTAs),
    // Q3_K 1.09, IQ2_XXS 1.10-1.14, Q4_K 1.09 at 17408 rows (at 5120 rows 64x64 was as fast). The
    // 5090: Q4_K 1.05-1.09, Q5_K 1.07-1.17, Q6_K 1.17-1.20 from 6144 rows (96+ CTAs; 5120 rows
    // stay under its grid bound and on 64x64, the tuner's choice there too). IQ4_XS takes 128x64
    // too although the 5090's tuner kept 64x64 for it (the 4080 SUPER's gain is the measured one;
    // the tuner corrects the other). The 5060 Ti: IQ2_S, IQ2_XS, IQ3_S, IQ3_XXS 1.08-1.29 at every
    // row count. Q4_0 and the rest (64x64 from 128 tokens too) keep 64x64. In the product
    // (ninferno-perplexity at context 100, tuner off on both arms, A B A B A B, perplexity
    // bit-identical): 4080 SUPER Qwen3.8-27B IQ3 412.9 -> 472.4 tok/s, 5090 Q4_K_XL 738 -> 757.
    if (tokens < kW4a8WideMinTokens) {
        return wide_format || problem.qtype == QType::GGML_IQ2_XXS ||
                       problem.qtype == QType::GGML_Q3_K
                   ? k128x64
                   : k64x64;
    }
    if (w4a8_ws_wins(problem, weight)) return k128x64ws;
    if (w4a8_wide_q4k(problem, tokens)) return k128x128;
    return wide_format ? k128x64 : k64x64;
}

int heuristic_r8(const Problem& p, std::int32_t tokens) {
    if (tokens == 1) {
        if (const auto tuned = tuned_r8(p)) return *tuned;
    }
    return rules_r8(p, tokens);
}

int k_multiple(WeightCodec codec) {
    switch (codec) {
    case WeightCodec::GgmlQ40:
    case WeightCodec::GgmlQ41:
    case WeightCodec::GgmlQ50:
    case WeightCodec::GgmlQ51:
    case WeightCodec::GgmlIq4Nl:
        return 32;
    default:
        return run_values(codec);
    }
}

std::optional<ggml_mmq::Type> mmq_type(const Weight& w) {
    switch (w.qtype) {
    case QType::GGML_Q4_K:
        return ggml_mmq::Type::Q4_K;
    case QType::GGML_Q5_K:
        return ggml_mmq::Type::Q5_K;
    case QType::GGML_Q6_K:
        return ggml_mmq::Type::Q6_K;
    case QType::GGML_IQ4_XS:
        return ggml_mmq::Type::IQ4_XS;
    case QType::GGML_IQ3_XXS:
        return ggml_mmq::Type::IQ3_XXS;
    case QType::GGML_IQ3_S:
        return ggml_mmq::Type::IQ3_S;
    case QType::GGML_Q4_0:
        return ggml_mmq::Type::Q4_0;
    case QType::GGML_Q4_1:
        return ggml_mmq::Type::Q4_1;
    case QType::GGML_Q5_0:
        return ggml_mmq::Type::Q5_0;
    case QType::GGML_Q5_1:
        return ggml_mmq::Type::Q5_1;
    case QType::GGML_IQ4_NL:
        return ggml_mmq::Type::IQ4_NL;
    case QType::GGML_Q2_K:
        return ggml_mmq::Type::Q2_K;
    case QType::GGML_Q3_K:
        return ggml_mmq::Type::Q3_K;
    case QType::GGML_IQ2_XXS:
        return ggml_mmq::Type::IQ2_XXS;
    case QType::GGML_IQ2_XS:
        return ggml_mmq::Type::IQ2_XS;
    case QType::GGML_IQ2_S:
        return ggml_mmq::Type::IQ2_S;
    case QType::GGML_IQ1_S:
        return ggml_mmq::Type::IQ1_S;
    case QType::GGML_IQ1_M:
        return ggml_mmq::Type::IQ1_M;
    default:
        return std::nullopt;
    }
}

bool mmq_admits(const Weight& w, LinearPolicy policy) {
    const auto type = mmq_type(w);
    if (!type || !ggml_mmq::matrix_supported(*type) || !allows_a8(policy) ||
        (w.k % ggml_mmq::kKMultiple) != 0) {
        return false;
    }
    return ggml_layout_admits(w) && ggml_mmq::device_supported();
}

bool mmq_prefill_wins(const Problem& p, const Weight& w) {
    return mmq_prefill_device() && ggml_mmq_format(p.qtype) &&
           (p.n % ggml_mmq::kRowMultiple) == 0 && mmq_admits(w, p.policy);
}

bool vec_admits(const Weight& w, LinearPolicy policy) {
    const auto type = mmq_type(w);
    if (!type || !ggml_mmq::vec_supported(*type) || !allows_a8(policy) ||
        (w.k % static_cast<std::int32_t>(ggml_block_values(w.qtype))) != 0) {
        return false;
    }
    return ggml_layout_admits(w);
}

bool vec_call(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 1 && tokens <= vec_max_tokens() && vec_admits(w, policy);
}

bool mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    return tokens >= kMmqMinTokens && tokens <= kMmqMaxTokens &&
           (rows % ggml_mmq::kRowMultiple) == 0 && mmq_admits(w, policy);
}

double dram_bytes_per_sm_cycle() {
    static PerDevice<double> bytes;
    // A device whose class cannot be read throws, and PerDevice retries on the next call (the
    // launch takes the plain kernel meanwhile): caching 0 would turn the variant off for good.
    return bytes.get([](int) {
        const auto& d = device_class();
        const auto& f = d.facts;
        if (!f.has_roofline() || d.sms <= 0) return 0.0;
        return f.dram_bytes_per_s() / (d.sms * f.reference_clock_hz());
    });
}

bool r8_stage_grid() {
    try {
        return !device_class().features.mma_bf16_native;
    } catch (const std::exception&) { return false; }
}

bool r8_fits(QType qtype, std::int32_t k, std::int32_t tokens) {
    if (tokens <= 0 || tokens > kR8MaxTokens || k <= 0) return false;
    std::int64_t slots = 1;
    while (slots < tokens) slots *= 2;
    // A codec with a min also holds each block's sum (r8_smem_bytes).
    const bool sums = qtype == QType::GGML_Q4_1 || qtype == QType::GGML_Q5_1 ||
                      qtype == QType::GGML_Q4_K || qtype == QType::GGML_Q5_K;
    // A codebook format's grid follows it where r8_kernel stages it (r8_stage_grid).
    std::int64_t grid = 0;
    if (r8_stage_grid()) switch (qtype) {
        case QType::GGML_IQ3_XXS:
            grid = 1024;
            break;
        case QType::GGML_IQ3_S:
        case QType::GGML_IQ2_XXS:
            grid = 2048;
            break;
        case QType::GGML_IQ2_XS:
            grid = 4096;
            break;
        case QType::GGML_IQ2_S:
        case QType::GGML_IQ1_S:
        case QType::GGML_IQ1_M:
            grid = 8192;
            break;
        default:
            break;
        }
    if (grid > 0) grid += 4; // r8_grid_pad (generic_r8.cuh): the grid starts 8-byte aligned
    return slots * (static_cast<std::int64_t>(k) / 32) * (sums ? 40 : 36) + grid <=
           kStaticSharedBytesPerBlock;
}

inline constexpr std::int64_t kR8VerifyQ40MinValues = 24ll << 20;

bool r8_verify_takes_shape(QType qtype, std::int64_t n, std::int64_t k) {
    if (qtype != QType::GGML_Q4_0 || n * k >= kR8VerifyQ40MinValues) {
        return r8_verify_takes(qtype);
    }
    // A small Q4_0 matrix keeps the A16 register MMA where it is native (Turing: R8, as for
    // every compiled format).
    try {
        return !device_class().features.mma_bf16_native && r8_verify_compiled(qtype);
    } catch (const std::exception&) { return false; }
}

bool bf16_mma_native() {
    try {
        return device_class().features.mma_bf16_native;
    } catch (const std::exception&) { return true; }
}

bool r8_verify_takes(QType qtype) {
    if (r8_verify_supports(qtype)) return true;
    try {
        return !device_class().features.mma_bf16_native && r8_verify_compiled(qtype);
    } catch (const std::exception&) { return false; }
}

bool staged_device_fits(int schedule) {
    const auto s     = kStagedSchedules.at(static_cast<std::size_t>(schedule));
    const auto bytes = static_cast<std::size_t>(s.rows + s.tokens) * 64 * 2 * s.stages;
    std::size_t limit = 0;
    try {
        limit = device_class().smem_per_block_optin;
    } catch (const std::exception&) {
        return true; // no device (a CPU-only tool): staged_fits holds the build ceiling
    }
    return limit == 0 || bytes <= limit;
}

WeightCodec codec_of(const Weight& weight) {
    switch (weight.qtype) {
    case QType::BF16:
        return WeightCodec::Bf16;
    case QType::FP8_E4M3FN_ROW_BF16:
        return WeightCodec::Fp8;
    case QType::NVFP4:
        return WeightCodec::Nvfp4;
    case QType::Q4_G64_FP16:
        return WeightCodec::Q4;
    case QType::Q5_G64_FP16:
        return WeightCodec::Q5;
    case QType::Q6_G64_FP16:
        return WeightCodec::Q6;
    case QType::Q8_G32_FP16:
        return WeightCodec::Q8;
    case QType::MXFP4:
        return WeightCodec::Mxfp4;
    case QType::FP8_E4M3FN_ROW_F32:
        return WeightCodec::Fp8RowF32;
    case QType::FP8_E4M3FN_B128_F32:
        return WeightCodec::Fp8B128;
    case QType::U4_G128_ZP_FP16:
        return WeightCodec::U4Zp;
    case QType::U4_G32_ZP_FP16:
        return WeightCodec::U4ZpG32;
    case QType::U4_G64_ZP_FP16:
        return WeightCodec::U4ZpG64;
    case QType::Q3_G128_FP16:
        return WeightCodec::Q3G128;
    case QType::Q3_G64_FP16:
        return WeightCodec::Q3G64;
    case QType::GGML_Q4_K:
        return WeightCodec::GgmlQ4K;
    case QType::GGML_Q5_K:
        return WeightCodec::GgmlQ5K;
    case QType::GGML_Q6_K:
        return WeightCodec::GgmlQ6K;
    case QType::GGML_Q3_K:
        return WeightCodec::GgmlQ3K;
    case QType::GGML_Q2_K:
        return WeightCodec::GgmlQ2K;
    case QType::GGML_IQ4_XS:
        return WeightCodec::GgmlIq4Xs;
    case QType::GGML_IQ3_XXS:
        return WeightCodec::GgmlIq3Xxs;
    case QType::GGML_IQ3_S:
        return WeightCodec::GgmlIq3S;
    case QType::GGML_Q4_0:
        return WeightCodec::GgmlQ40;
    case QType::GGML_Q4_1:
        return WeightCodec::GgmlQ41;
    case QType::GGML_Q5_0:
        return WeightCodec::GgmlQ50;
    case QType::GGML_Q5_1:
        return WeightCodec::GgmlQ51;
    case QType::GGML_IQ4_NL:
        return WeightCodec::GgmlIq4Nl;
    case QType::GGML_IQ2_XXS:
        return WeightCodec::GgmlIq2Xxs;
    case QType::GGML_IQ2_XS:
        return WeightCodec::GgmlIq2Xs;
    case QType::GGML_IQ2_S:
        return WeightCodec::GgmlIq2S;
    case QType::GGML_IQ1_S:
        return WeightCodec::GgmlIq1S;
    case QType::GGML_IQ1_M:
        return WeightCodec::GgmlIq1M;
    case QType::GGML_TQ1_0:
        return WeightCodec::GgmlTq10;
    case QType::GGML_TQ2_0:
        return WeightCodec::GgmlTq20;
    case QType::GGML_Q1_0:
        return WeightCodec::GgmlQ10;
    case QType::GGML_Q2_0:
        return WeightCodec::GgmlQ20;
    default:
        throw std::invalid_argument("generic linear: unsupported weight format");
    }
}

bool nvfp4_prefill_a8_policy(QType qtype, LinearPolicy policy) {
    return qtype == QType::NVFP4 && allows_a8(policy) &&
           !(w4a4_schedule_count() > 0 && allows_a4(policy));
}

bool supports(const Weight& w) {
    if (w.n <= 0 || w.k <= 0 || w.qdata == nullptr) return false;
    switch (w.qtype) {
    case QType::BF16:
        return w.layout == QuantLayout::Contiguous;
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32:
        return w.layout == QuantLayout::RowScale && w.scales != nullptr;
    case QType::FP8_E4M3FN_B128_F32:
        return w.layout == QuantLayout::BlockScale128F32 && w.scales != nullptr;
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        return (w.layout == QuantLayout::RowGroupZp ||
                (w.layout == QuantLayout::RowGroupZpOrdered && w.k_order != nullptr)) &&
               w.scales != nullptr && w.qhigh != nullptr &&
               (w.k % static_cast<int>(u4_zp_group(w.qtype))) == 0;
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return w.layout == QuantLayout::GgmlBlocks && (w.k % 256) == 0;
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_TQ1_0:
    case QType::GGML_TQ2_0:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
    {
        const auto plane = ggml_plane_bytes(w.qtype);
        return w.layout == QuantLayout::GgmlPlanes && w.scales != nullptr &&
               (w.qhigh != nullptr || plane.high == 0) &&
               (w.k % static_cast<std::int32_t>(plane.values)) == 0;
    }
    case QType::NVFP4:
        // Any N: every generic route guards its rows, and the scale plane holds whole 128-row
        // tiles (a last tile N does not fill included).
        return w.layout == QuantLayout::BlockScaleK16M128x4 && w.scales != nullptr &&
               (w.k % 64) == 0 && std::isfinite(w.weight_scale_divisor) &&
               w.weight_scale_divisor > 0.0F;
    case QType::MXFP4:
        return w.layout == QuantLayout::BlockScaleK32E8M0 && w.scales != nullptr &&
               (w.k % 32) == 0;
    case QType::Q4_G64_FP16:
    case QType::Q8_G32_FP16:
        return w.layout == QuantLayout::RowSplit && w.scales != nullptr &&
               w.padded_shape[1] >= w.k && (w.padded_shape[1] % 128) == 0;
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        return w.layout == QuantLayout::RowSplit && w.scales != nullptr && w.qhigh != nullptr &&
               w.padded_shape[1] >= w.k && (w.padded_shape[1] % 128) == 0;
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
        return w.layout == QuantLayout::RowSplit && w.scales != nullptr &&
               w.padded_shape[1] >= w.k && (w.padded_shape[1] % 128) == 0;
    default:
        return false;
    }
}

bool rmma_tile_compiled(WeightCodec codec, int m_tiles, int n_tiles, std::int32_t k) {
    // run_values: 32 for a decoder of 32-value blocks, 128 for the ggml 32-blocks that step four
    // blocks at a time; both read 32 values per decode (kRun == 32). So does the u4 G32/G64
    // quad-run decoder that launch_rmma_u4zp_g32/_g64 (and the MoE launches) take at K % 128 == 0.
    const bool quad_u4 = (codec == WeightCodec::U4ZpG32 || codec == WeightCodec::U4ZpG64) &&
                         k > 0 && (k % 128) == 0;
    return !((run_values(codec) >= 32 || quad_u4) && m_tiles * n_tiles >= 16);
}

DeviceFeatures route_requires(Route route) {
    DeviceFeatures f;
    switch (route.family) {
    case Family::Scalar:
        return f;
    case Family::RegisterMma:
        f.mma_bf16 = true;
        f.clusters = route.schedule < rmma_schedule_count() &&
                     rmma_schedule_info(route.schedule).cluster > 1;
        return f;
    case Family::W4a4Mma:
        f.mma_bf16     = true;
        f.mma_mxf4nvf4 = true;
        return f;
    case Family::Nvfp4TiledMma:
        // The E2M1 converter of the sm_120 family decodes its tiles.
        f.mma_bf16     = true;
        f.mma_mxf4nvf4 = true;
        return f;
    case Family::TiledMma:
    case Family::W4a8Mma: // int8 m16n8k32: part of mma_bf16 (core/device_class.h)
    case Family::StagedMma:
    case Family::R8Mma:
        f.mma_bf16 = true;
        return f;
    case Family::GgmlMmq: // sm_75 and up, held by mmq_admits
    case Family::GgmlVec: // dp4a
        return f;
    case Family::I8Mma: // m16n8k32 s8 x u8, ldmatrix, cp.async: sm_80 on
        f.mma_bf16        = true;
        f.mma_bf16_native = true;
        return f;
    }
    return f;
}

DeviceFeatures current_features() {
    try {
        return device_class().features;
    } catch (const std::exception&) {
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .mma_f8f6f4      = true,
                .mma_mxf4nvf4    = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    }
}

bool volta_cutlass_takes(std::int32_t tokens);

bool volta_cutlass_runs(bool only, std::int32_t tokens);

bool tiled_takes(WeightCodec codec) {
    return tiled_supports(codec) || (codec == WeightCodec::Q8 && !current_features().mma_tensor);
}

bool volta_cutlass_only(WeightCodec codec) {
    return !tiled_takes(codec) && !current_features().mma_tensor;
}

R8CallRefusal r8_call_refusal(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    if (!allows_r8(w, policy)) return R8CallRefusal::NotAdmitted;
    if (tokens <= kR8DecodeTokens) {
        return r8_fits(w.qtype, w.k, tokens) ? R8CallRefusal::None : R8CallRefusal::DecodeFit;
    }
    if (tokens > r8_verify_widest(w.qtype)) return R8CallRefusal::VerifyWidth;
    return r8_verify_takes_shape(w.qtype, w.n, w.k) ? R8CallRefusal::None
                                                     : R8CallRefusal::VerifyShape;
}

ScheduleRefusal r8_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= r8_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || !r8_supports(w.qtype)) return ScheduleRefusal::Format;
    if (!allows_a8(policy)) return ScheduleRefusal::Policy;
    const Route route{Family::R8Mma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off a 16-byte boundary (generic_refusal), or the codes.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        !aligned_to(w.qdata, 16)) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % r8_k_values(w.qtype)) != 0) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal w4a8_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= w4a8_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || !w4a8_supports(w.qtype)) return ScheduleRefusal::Format;
    const bool producer = w4a8_schedule_info(index).producers > 0;
    if (producer && !w4a8_ws_supports(w.qtype)) return ScheduleRefusal::NotCompiled;
    if (!w4a8_fits(index, w.qtype)) return ScheduleRefusal::Resource;
    if (!w4a8_policy_allows(w, policy)) return ScheduleRefusal::Policy;
    const Route route{Family::W4a8Mma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // The classic tiles read a ggml high plane a byte at a time; the producer tiles in vector
    // words (generic_refusal), and IQ3_XXS's sign plane 8 bytes at a time (w4a8_ws_admits).
    const bool high_off = w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr &&
                          !aligned_to(w.qhigh, 16);
    if (!aligned_to(w.qdata, 16) || (producer && high_off) ||
        (producer && w.qtype == QType::GGML_IQ3_XXS && !aligned_to(w.qhigh, 8))) {
        return ScheduleRefusal::Alignment;
    }
    if (!w4a8_k_admits(w.qtype, w.k)) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal rmma_refusal(int index, const Weight& w) {
    if (index < 0 || index >= rmma_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!rmma_schedule_launchable(index)) return ScheduleRefusal::NotCompiled;
    if (!supports(w)) return ScheduleRefusal::Format;
    // Whether the build holds the tile for the codec, ahead of the weight's own checks, so an
    // alignment or extent refusal does not hide it (rmma_coverage counts compiled tiles by it).
    const WeightCodec codec = codec_of(w);
    const auto s            = rmma_schedule_info(index);
    if (!rmma_tile_compiled(codec, s.m_tiles, s.n_tiles, w.k)) return ScheduleRefusal::CodecTile;
    const Route route{Family::RegisterMma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off a 16-byte boundary (generic_refusal).
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % k_multiple(codec)) != 0) return ScheduleRefusal::Extent;
    if (codec == WeightCodec::Fp8 || codec == WeightCodec::Fp8RowF32 ||
        codec == WeightCodec::Fp8B128 || codec == WeightCodec::Bf16 ||
        codec == WeightCodec::Mxfp4 || codec == WeightCodec::U4Zp ||
        codec == WeightCodec::U4ZpG64 ||
        codec == WeightCodec::GgmlQ4K || codec == WeightCodec::GgmlQ5K ||
        codec == WeightCodec::GgmlQ6K || codec == WeightCodec::GgmlQ3K ||
        codec == WeightCodec::GgmlQ2K || codec == WeightCodec::GgmlIq4Xs ||
        codec == WeightCodec::GgmlIq3Xxs || codec == WeightCodec::GgmlIq3S ||
        codec == WeightCodec::GgmlQ40 || codec == WeightCodec::GgmlQ41 ||
        codec == WeightCodec::GgmlQ50 || codec == WeightCodec::GgmlQ51 ||
        codec == WeightCodec::GgmlIq4Nl || is_ggml_block(w.qtype)) {
        return aligned_to(w.qdata, 16) ? ScheduleRefusal::None : ScheduleRefusal::Alignment;
    }
    return aligned_to(w.qdata, 8) ? ScheduleRefusal::None : ScheduleRefusal::Alignment;
}

ScheduleRefusal tiled_refusal(int index, const Weight& w) {
    if (index < 0 || index >= tiled_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w)) return ScheduleRefusal::Format;
    const WeightCodec codec = codec_of(w);
    const bool cutlass_only = volta_cutlass_only(codec);
    if (!cutlass_only && !tiled_takes(codec)) return ScheduleRefusal::Format;
    // A CUTLASS-only format (volta_cutlass_only) runs no tile of its own: schedule 0 names it.
    if (cutlass_only && index != 0) return ScheduleRefusal::CodecTile;
    if (!cutlass_only && !tiled_fits(index, codec)) return ScheduleRefusal::Resource;
    const Route route{Family::TiledMma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off a 16-byte boundary (generic_refusal), or the codes where
    // the tiles read them.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        (!cutlass_only && !aligned_to(w.qdata, 16))) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % 64) != 0) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal staged_refusal(int index, const Weight& w) {
    if (index < 0 || index >= staged_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || codec_of(w) != WeightCodec::Bf16) return ScheduleRefusal::Format;
    if (!staged_fits(index) || !staged_device_fits(index)) return ScheduleRefusal::Resource;
    const Route route{Family::StagedMma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off 16 bytes (generic_refusal; a BF16 weight has none), or
    // the codes.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        !aligned_to(w.qdata, 16)) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % 64) != 0) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal i8_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= i8_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || !i8_format(w.qtype)) return ScheduleRefusal::Format;
    if (!i8_fits(index, w.qtype)) return ScheduleRefusal::Resource;
    if (!allows_a8(policy)) return ScheduleRefusal::Policy;
    const Route route{Family::I8Mma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off 16 bytes (generic_refusal; a G64 weight has none), the
    // codes or scales off 16 (one 16-byte copy per row), a Q5 / Q6 high plane off 8 / 16.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        !i8_planes_aligned(w)) {
        return ScheduleRefusal::Alignment;
    }
    if (!i8_extent(w)) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

namespace {

ScheduleRefusal generic_refusal(Route route, const Weight& w) {
    if (!supports(w)) return ScheduleRefusal::Format;
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) {
        return ScheduleRefusal::Alignment;
    }
    return ScheduleRefusal::NotDeclared;
}


} // namespace

ScheduleRefusal w4a4_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= w4a4_schedule_names()) return ScheduleRefusal::NotDeclared;
    if (index >= w4a4_schedule_count()) return ScheduleRefusal::NotCompiled;
    if (!supports(w) || w.qtype != QType::NVFP4 || !std::isfinite(w.input_scale_divisor) ||
        w.input_scale_divisor <= 0.0F) {
        return ScheduleRefusal::Format;
    }
    if (!allows_a4(policy)) return ScheduleRefusal::Policy;
    const auto common =
        generic_refusal(Route{Family::W4a4Mma, static_cast<std::uint8_t>(index)}, w);
    return common == ScheduleRefusal::NotDeclared ? ScheduleRefusal::None : common;
}

ScheduleRefusal nvfp4_tiled_refusal(int index, const Weight& w) {
    if (index < 0 || index >= nvfp4_tiled_schedule_count()) return ScheduleRefusal::NotDeclared;
    // The schedules parse on every build, but their kernels are 120a-only.
    if (!kNativeNvfp4) return ScheduleRefusal::NotCompiled;
    if (!supports(w) || w.qtype != QType::NVFP4) return ScheduleRefusal::Format;
    const auto common =
        generic_refusal(Route{Family::Nvfp4TiledMma, static_cast<std::uint8_t>(index)}, w);
    if (common != ScheduleRefusal::NotDeclared) return common;
    // supports() holds K to whole 64-value steps; a lane reads eight code bytes at once.
    return aligned_to(w.qdata, 8) ? ScheduleRefusal::None : ScheduleRefusal::Alignment;
}

ScheduleRefusal ggml_refusal(Family family, int index, const Weight& w, LinearPolicy policy) {
    const bool matrix = family == Family::GgmlMmq;
    if (index != 0 || (!matrix && family != Family::GgmlVec)) return ScheduleRefusal::NotDeclared;
    if (!supports(w)) return ScheduleRefusal::Format;
    const auto type = mmq_type(w);
    if (!type || !(matrix ? ggml_mmq::matrix_supported(*type) : ggml_mmq::vec_supported(*type))) {
        return ScheduleRefusal::Format;
    }
    // ggml_layout_admits: Q4_K / Q5_K as blocks, the others as planes, each plane it reads there.
    const bool planes = w.qtype != QType::GGML_Q4_K && w.qtype != QType::GGML_Q5_K;
    const auto bytes  = mmq_plane_bytes(w.qtype);
    if (planes != (w.layout == QuantLayout::GgmlPlanes) || (bytes.high > 0 && w.qhigh == nullptr) ||
        (bytes.scales > 0 && w.scales == nullptr)) {
        return ScheduleRefusal::Format;
    }
    if (!allows_a8(policy)) return ScheduleRefusal::Policy;
    const auto common = generic_refusal(Route{family, 0}, w);
    if (common != ScheduleRefusal::NotDeclared) return common;
    if (matrix && !ggml_mmq::device_supported()) return ScheduleRefusal::Instruction;
    // ggml's loaders read every plane in vector words.
    if (!aligned_to(w.qdata, 16) || (bytes.high > 0 && !aligned_to(w.qhigh, 16)) ||
        (bytes.scales > 0 && !aligned_to(w.scales, 16))) {
        return ScheduleRefusal::Alignment;
    }
    const auto multiple = matrix ? static_cast<std::int32_t>(ggml_mmq::kKMultiple)
                                 : static_cast<std::int32_t>(ggml_block_values(w.qtype));
    return (w.k % multiple) == 0 ? ScheduleRefusal::None : ScheduleRefusal::Extent;
}

bool route_admits(Route route, const Weight& w, LinearPolicy policy) {
    if (route.family == Family::RegisterMma) {
        return rmma_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::R8Mma) {
        return r8_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::W4a8Mma) {
        return w4a8_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::TiledMma) {
        return tiled_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::StagedMma) {
        return staged_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::I8Mma) {
        return i8_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::W4a4Mma) {
        return w4a4_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::Nvfp4TiledMma) {
        return nvfp4_tiled_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::GgmlMmq || route.family == Family::GgmlVec) {
        return ggml_refusal(route.family, route.schedule, w, policy) == ScheduleRefusal::None;
    }
    // The scalar path: every other family returned its refusal above (generic_refusal: a ggml high
    // plane off 16 bytes among them). The scalar path reads that plane byte by byte. A family with
    // no refusal of its own is refused here.
    return route.family == Family::Scalar && supports(w) &&
           has_features(current_features(), route_requires(route));
}

namespace {

bool rmma_sums_lane_local(int schedule) {
    return !current_features().mma_tensor && !invariant_routes_active() &&
           rmma_schedule_info(schedule).n_tiles == 1;
}


} // namespace

bool route_is_token_invariant(Route route) {
    switch (route.family) {
    case Family::RegisterMma:
        return rmma_schedule_info(route.schedule).warps_k == 1 &&
               !rmma_sums_lane_local(route.schedule);
    case Family::W4a4Mma:
    case Family::TiledMma:
    case Family::W4a8Mma:
    case Family::StagedMma:
    case Family::R8Mma: // a token's sums depend on its own activation only
        return true;
    case Family::Nvfp4TiledMma:
        return nvfp4_tiled_schedule_info(route.schedule).classes == 1; // rmma WarpsK == 1
    case Family::Scalar:
        return true; // one lane order per element, independent of T
    case Family::GgmlMmq: // stream-k splits K by the call's tile count
        return false;
    case Family::GgmlVec: // a lane's slices and one butterfly per token, whatever T
        return true;
    case Family::I8Mma: // K groups ascending, never split: a token's sums are its own
        return true;
    }
    return false;
}

SummationOrder summation_order(Route route) {
    // The orders of the prefill and invariant profiles, where its users act (the prefill route's
    // bands, in the route tables, route-tune and auto_route): there no register-MMA tile adds its
    // min-term sums in FP32 (rmma_add_sums), so the WarpsK == 1 tiles of every width and the staged
    // MMA sum alike. Outside them the tiles of four or more token tiles do not.
    if ((route.family == Family::RegisterMma && route.schedule < rmma_schedule_count() &&
         rmma_schedule_info(route.schedule).warps_k == 1 &&
         !rmma_sums_lane_local(route.schedule)) ||
        (route.family == Family::StagedMma && route.schedule < staged_schedule_count())) {
        return {SummationOrder::Kind::Register, Route{}};
    }
    if (route.family == Family::I8Mma) return {SummationOrder::Kind::I8, Route{}};
    return {SummationOrder::Kind::Own, route};
}

bool same_summation(Route a, Route b) { return summation_order(a) == summation_order(b); }

namespace {

std::optional<Route> nvfp4_tiled_route(const Problem& problem, const Weight& weight,
                                       [[maybe_unused]] int classes,
                                       [[maybe_unused]] std::int32_t tokens) {
    if (problem.qtype != QType::NVFP4) return std::nullopt;
#ifdef NINFERNO_WITHOUT_NVFP4
    return std::nullopt;
#endif
    for (int i = 0; i < nvfp4_tiled_schedule_count(); ++i) {
        const auto s = nvfp4_tiled_schedule_info(i);
        if (s.classes != classes) continue;
        const Route r{Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)};
        if (!route_admits(r, weight, problem.policy)) return std::nullopt;
        const std::int64_t rows_per_cta =
            problem.epilogue == Epilogue::SwiGlu ? s.block_n / 2 : s.block_n;
        const std::int64_t ctas = ((problem.n + rows_per_cta - 1) / rows_per_cta) *
                                  ((tokens + s.block_m - 1) / s.block_m);
        if (tokens < 2 * s.block_m || ctas < sm_count()) return std::nullopt;
        return r;
    }
    return std::nullopt;
}


} // namespace

int i8_pick(const Problem& problem, QType qtype, std::int32_t tokens) {
    const double sets = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    const double sms  = std::max(sm_count(), 1);
    int best          = -1;
    double best_cost  = 0;
    for (int i = 0; i < i8_schedule_count(); ++i) {
        if (!i8_fits(i, qtype)) continue;
        const auto s       = kI8Schedules[i];
        const double ctas  = std::ceil(problem.n * sets / s.block_n) * std::ceil(tokens / double(s.block_m));
        const double waves = std::ceil(ctas / sms);
        const double cost  = waves * s.block_m * s.block_n / kI8Eff[i];
        if (best < 0 || cost < best_cost) {
            best      = i;
            best_cost = cost;
        }
    }
    return best;
}

std::optional<Route> i8_route(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    const int pick = i8_pick(problem, weight.qtype, tokens);
    if (pick < 0) return std::nullopt;
    const Route r{Family::I8Mma, static_cast<std::uint8_t>(pick)};
    if (!route_admits(r, weight, problem.policy)) return std::nullopt;
    return r;
}

std::optional<Route> w4a8_route(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    const Route tuned{Family::W4a8Mma,
                      static_cast<std::uint8_t>(heuristic_w4a8(problem, weight, tokens))};
    if (route_admits(tuned, weight, problem.policy)) return tuned;
    for (int i = 0; i < w4a8_schedule_count(); ++i) {
        const Route r{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
        if (route_admits(r, weight, problem.policy)) return r;
    }
    return std::nullopt;
}

std::optional<Route> staged_speed_route(const Problem& problem, const Weight& weight,
                                        std::int32_t tokens) {
    if (tokens < kRmmaMidMinTokens) return std::nullopt;
    struct Pick {
        int rows, cols, stages;
        double waves;
    };
    for (const Pick p : {Pick{256, 128, 2, 2.0}, Pick{128, 64, 2, 1.0}, Pick{64, 64, 3, 0.5}}) {
        if (speed_waves(problem.n, tokens, p.rows, p.cols) < p.waves) continue;
        for (int i = 0; i < staged_schedule_count(); ++i) {
            const auto st = staged_schedule_info(i);
            if (st.rows != p.rows || st.tokens != p.cols || st.stages != p.stages) continue;
            const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
            if (route_admits(r, weight, problem.policy)) return r;
        }
    }
    return std::nullopt;
}

Route heuristic_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                      bool prefill) {
    // Under the prefill profile every call of 17 tokens or more takes one route; outside it, routes
    // by T band. A4 only where the policy allows it, and never at T <= 16, which is where decode
    // and speculative verify run.
    const bool prefill_band = prefill && tokens >= 17;
    if (tokens >= 17 && allows_w4a4(weight, problem.policy)) {
        const Route r{Family::W4a4Mma,
                      static_cast<std::uint8_t>(heuristic_w4a4(problem.n, tokens, prefill_band))};
        if (route_admits(r, weight, problem.policy)) return r;
    }
    // A8 where the policy allows it: ggml's vector kernel for speculative verify (2..8 tokens) of
    // the ggml block formats it decodes (vec_rule), R8 for decode and for the verify of the other
    // formats with an R8 codec, W4A8 from 17 tokens; the rest stays A16. The vector kernel streams
    // each weight row once for every token of the call and runs the epilogue in place: on the RTX
    // 5090 ISTA IQ3_S MTP3 it beat the R8 and A16 schedules the dense tuner picked there (C1 166 ->
    // 174 tok/s), whose L2-cold timings of one call at a time miss the stream of a verify step. The
    // tuner still replaces it where a route beats it by its margin (Muse Glimmer 30B Q4_K_M: the
    // rule alone lost C1 by 19 %).
    if (vec_rule(weight, problem.policy, tokens)) return Route{Family::GgmlVec, 0};
    // From 17 tokens outside the prefill profile the cost model's route of either activation class
    // (cost_route, a8_candidates); the rules below where the device reports no clocks.
    if (!prefill_band && tokens >= 17 && problem.qtype != QType::NVFP4) {
        if (const auto r = cost_route(problem, weight, tokens)) return *r;
    }
    if (r8_call(weight, problem.policy, tokens)) {
        const Route r{Family::R8Mma, static_cast<std::uint8_t>(heuristic_r8(problem, tokens))};
        if (route_admits(r, weight, problem.policy)) return r;
    }
    if (tokens >= 17 && mmq_prefill_wins(problem, weight)) return Route{Family::GgmlMmq, 0};
    if (tokens >= 17) {
        if (const auto r = i8_route(problem, weight, tokens)) return *r;
    }
    if (tokens >= 17 && allows_w4a8(weight, problem.policy)) {
        if (const auto r = w4a8_route(problem, weight, tokens)) return *r;
    }
    // Where the BF16 MMA is emulated (Turing), the band between R8 / the vector kernel and W4A8
    // takes ggml's int8 matrix kernel for the formats it reads: the register MMA there runs on
    // the emulation, about three times its native cost.
    if (mmq_verify_rule(weight, problem.policy, problem.n, tokens)) return Route{Family::GgmlMmq, 0};
    // NVFP4 A16 outside the invariant profile stays on the register MMA: against its prefill
    // schedule the tiled route loses on the Muse gate/up and synthetic shapes (+11..+20%)
    // and wins on the down projection only from T = 2048. A route table may still name it.
    // A format with a tiled codec takes it for every prefill-profile call of 17 tokens or more (one
    // summation order for all of them) and, outside the profile, where its tiles fill the SMs
    // (tiled_speed_pays).
    // The first schedule whose tiles fit the build: the tuned 128x128 everywhere but Turing.
    if (prefill_band || tiled_speed_pays(problem.n, tokens)) {
        for (int i = 0; i < tiled_schedule_count(); ++i) {
            const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
            if (route_admits(r, weight, problem.policy)) return r;
        }
    }
    // BF16 weights take the staged register MMA (same bytes, operands through shared memory):
    // under the prefill profile the first schedule that fits the build, and outside it once its
    // 64x64 tiles cover half the SMs, with the largest tile whose grid still fills them
    // (staged_speed_route). Only on the native BF16 MMA: where it is emulated its tiles lost to
    // the register MMA, x2.5..4 on Turing (RTX 2080 Ti, BF16 N 1536 K 2048, T 17..1024: 142..1005
    // us against 35..402) and x6..8 on Volta's FP32 FMAs (V100, T 17..128: 942 us against 110).
    if (bf16_mma_native()) {
        if (prefill_band) {
            for (int i = 0; i < staged_schedule_count(); ++i) {
                const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
                if (route_admits(r, weight, problem.policy)) return r;
            }
        } else if (const auto r = staged_speed_route(problem, weight, tokens)) {
            return *r;
        }
    }
    const Route r{Family::RegisterMma,
                  static_cast<std::uint8_t>(heuristic_rmma(problem, tokens, prefill_band))};
    if (route_admits(r, weight, problem.policy)) return r;
    // A register-MMA tile the build does not hold for the call: the narrower tiles the dense
    // tuner picks where the BF16 MMA is emulated (m1.n2 and m1.n1 of four row warps: V100 and RTX
    // 2080 Ti, BF16 N 1536 K 2048, T 17..2048), then the staged MMA, before the scalar route (V100,
    // gpt-oss BF16 attention at T = 79: the scalar route took 7.3 ms a call).
    for (const int n_tiles : {2, 1}) {
        const int index = rmma_index(4, 1, 1, n_tiles);
        const Route narrow{Family::RegisterMma, static_cast<std::uint8_t>(index)};
        if (route_admits(narrow, weight, problem.policy)) return narrow;
    }
    if (prefill_band || tokens >= kStagedSpeedMinTokens) {
        for (int i = 0; i < staged_schedule_count(); ++i) {
            const Route staged{Family::StagedMma, static_cast<std::uint8_t>(i)};
            if (route_admits(staged, weight, problem.policy)) return staged;
        }
    }
    return Route{Family::Scalar, 0};
}

std::optional<Route> volta_tiled_route(const Weight& weight, LinearPolicy policy,
                                       std::int32_t tokens) {
    // Without tensor mma.sync (Volta) every register-MMA, W4A8 and W4A4 tile runs on FP32 FMAs or
    // dp4a over warp shuffles (ops/common/mma.cuh), while the tiled kernel issues the tensor cores'
    // own mma.m8n8k4 (generic_tiled.cuh): a call of 17 tokens or more whose format has a tiled
    // codec takes it whatever the policy allows (f16 activations, not int8 ones). Decode and verify
    // (T <= 16) keep their routes, so the switch is at T = 17.
    if (tokens < 17 || current_features().mma_tensor || !supports(weight)) return std::nullopt;
    if (volta_cutlass_only(codec_of(weight))) {
        // From 17 tokens, as the tiled formats: a prefill token's bytes do not depend on its call's
        // width (a narrower call on another route summed in another order).
        const Route r{Family::TiledMma, 0};
        if (volta_cutlass_runs(true, tokens) && route_admits(r, weight, policy)) return r;
        return std::nullopt;
    }
    if (!tiled_takes(codec_of(weight))) return std::nullopt;
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (route_admits(r, weight, policy)) return r;
    }
    return std::nullopt;
}

namespace {

Route auto_route(const Problem& problem, const Weight& weight, std::int32_t tokens, bool prefill);

std::optional<Route> pinned_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                                  bool prefill);


} // namespace

std::optional<Route> volta_hmma_route(const Problem& problem, const Weight& weight,
                                      std::int32_t tokens) {
    if (tokens < kVoltaHmmaMinTokens || tokens > kVoltaHmmaMaxTokens ||
        current_features().mma_tensor || !supports(weight) ||
        !volta_hmma_codec(codec_of(weight)) ||
        (weight.k % volta_hmma_k_multiple(codec_of(weight))) != 0) {
        return std::nullopt;
    }
    if (tokens <= kVecRuleMaxTokens &&
        (vec_call(weight, problem.policy, tokens) || r8_call(weight, problem.policy, tokens))) {
        return std::nullopt;
    }
    const Route r{Family::RegisterMma,
                  static_cast<std::uint8_t>(heuristic_rmma(problem, tokens, false))};
    if (route_admits(r, weight, problem.policy)) return r;
    return std::nullopt;
}

Route select_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                   bool prefill) {
    // The invariant profile: one summation order per problem at every width, BF16 activations
    // whatever the policy allows. The scalar route (one lane order per element) takes what the
    // register MMA cannot read.
    if (invariant_routes_active()) {
        const Route r{Family::RegisterMma,
                      static_cast<std::uint8_t>(invariant_rmma(problem.k, tokens))};
        if (!route_admits(r, weight, problem.policy)) return Route{Family::Scalar, 0};
        // NVFP4 wide calls: the tiled kernel sums in the register MMA's order (the same bytes).
        if (const auto t =
                nvfp4_tiled_route(problem, weight, rmma_schedule_info(r.schedule).warps_k, tokens)) {
            return *t;
        }
        return r;
    }
    // A route probe forcing one activation route for every width (route_probe.h).
    if (const std::int8_t forced = route_probe_state.quantized_act; forced >= 0) {
        const bool prefill_band = prefill && tokens >= 17;
        if (forced == 1 && allows_w4a4(weight, problem.policy)) {
            const Route r{Family::W4a4Mma,
                          static_cast<std::uint8_t>(heuristic_w4a4(problem.n, tokens, prefill_band))};
            if (route_admits(r, weight, problem.policy)) return r;
        }
        if (forced == 1 && r8_call(weight, problem.policy, tokens)) {
            const Route r{Family::R8Mma, static_cast<std::uint8_t>(heuristic_r8(problem, tokens))};
            if (route_admits(r, weight, problem.policy)) return r;
        }
        if (forced == 1 && allows_w4a8(weight, problem.policy)) {
            if (const auto r = w4a8_route(problem, weight, tokens)) return *r;
        }
        const Route r{Family::RegisterMma,
                      static_cast<std::uint8_t>(heuristic_rmma(problem, tokens, prefill_band))};
        if (route_admits(r, weight, problem.policy)) return r;
        return Route{Family::Scalar, 0};
    }
    if (const auto r = volta_tiled_route(weight, problem.policy, tokens)) {
        // Volta keeps its tiled route under a prefill pin (f16 activations: the a16 family's
        // there), and the ledger notes it, so a pin no prefill call can take is refused.
        if (tokens >= kPrefillPinMinTokens && route_pins().prefill != PrefillRoute::Auto) {
            note_prefill_pin(route_pins().prefill == PrefillRoute::A16,
                             family_bit(PrefillRoute::A16));
        }
        return *r;
    }
    // A maintainer's route pin (--prefill-route, --decode-route; ops/route_pins.h): the pinned
    // family where it can compute the call, else the automatic route below. No pin: nothing here.
    // It acts before Volta's HMMA verify route (volta_fixed_route's second half), the automatic
    // choice of that band, so an A/B can pin vec or R8 against it.
    if (const auto pinned = pinned_route(problem, weight, tokens, prefill)) return *pinned;
    if (const auto r = volta_hmma_route(problem, weight, tokens)) return *r;
    return auto_route(problem, weight, tokens, prefill);
}

namespace {

Route auto_route(const Problem& problem, const Weight& weight, std::int32_t tokens, bool prefill) {
    // NVFP4 keeps the width invariance its own kernels give where a device has them (the shape
    // tables of sm_100+; ops/linear/nvfp4): on sm_89/86 the same projections are the generic
    // path's, whose speed bands split K by width (heuristic_rmma: 4, 8, 2 or 1 K-warps from 17
    // tokens), so a prefill token's bytes followed its chunk's size (Gemma 4 --a16 on an RTX 4090,
    // test_linear_gemma4_token_invariance). An A16 call of 17 tokens or more takes one summation
    // order at every width; decode and verify (T <= 16) keep their routes. W4A4 and W4A8 sum each
    // token over its own activation whatever the tile, so A4/A8 calls keep their speed tiles.
    if (problem.qtype == QType::NVFP4 && tokens >= 17 && !allows_w4a4(weight, problem.policy) &&
        !allows_w4a8(weight, problem.policy)) {
        // A weight that fits half the L2 takes the invariant profile's register MMA (the decode
        // split of K, 16 rows per CTA): its 32-token columns re-read the weight from L2, where the
        // 128x128 prefill tile left a 1024-row projection eight CTAs (RTX 4090, Gemma 4 26B
        // shapes 1024..8192 x 2112..8192 at T 17..128: 2-4x slower than the speed bands).
        // A larger weight takes the prefill profile's route (31B 16384x5376, 43008x5376,
        // 5376x21504: 0.3-0.8x the speed bands from 32 tokens).
        const auto weight_bytes =
            static_cast<std::size_t>(problem.n) * static_cast<std::size_t>(problem.k) * 9 / 16;
        if (2 * weight_bytes <= device_class().l2_bytes) {
            const Route r{Family::RegisterMma,
                          static_cast<std::uint8_t>(invariant_rmma(problem.k, tokens))};
            if (route_admits(r, weight, problem.policy)) return r;
        }
        prefill = true;
    }
    // A W4A8 problem of 17 tokens or more takes the W4A8 heuristic: route tables hold A16 schedules
    // only (compute_class keys these formats as a16 whatever the policy), which apply below that.
    // An R8 call likewise.
    // ggml's matrix kernel stands in for the heuristic only: a route table's or the dense tuner's
    // A16 schedule keeps its shape.
    const bool a8_wide = tokens >= 17 && allows_w4a8(weight, problem.policy);
    const bool vec     = vec_rule(weight, problem.policy, tokens);
    const bool a8      = a8_wide || vec || r8_call(weight, problem.policy, tokens) ||
                    mmq_verify_rule(weight, problem.policy, problem.n, tokens);
    const auto tuned   = a8 ? std::optional<Route>{} : table_route(problem, tokens, prefill);
    if (tuned) {
        // Under the prefill profile a format with a tiled codec keeps its tiled route: the tiled
        // and register-MMA families sum in different orders.
        const bool prefill_band = prefill && tokens >= 17;
        if (route_admits(*tuned, weight, problem.policy) &&
            (!prefill_band || route_is_token_invariant(*tuned)) &&
            !(prefill_band && supports(weight) && tiled_supports(codec_of(weight)) &&
              tuned->family != Family::TiledMma)) {
            return *tuned;
        }
    }
    // The dense tuner's choice at decode widths (generic_tune.cpp): an A16 schedule for an A16 call,
    // an R8 schedule for an R8 decode call; never a W4A8 call.
    if (!a8_wide) {
        // Under the vector kernel's rule any timed route that beat it may stand in.
        const bool r8_ok = r8_call(weight, problem.policy, tokens) ||
                           r8_verify_tunable(weight, problem.policy, tokens);
        if (const auto dense = tuned_dense_route(problem, tokens);
            dense && route_admits(*dense, weight, problem.policy) &&
            (dense->family == Family::R8Mma ? r8_ok
                                            : vec || !r8_call(weight, problem.policy, tokens))) {
            return *dense;
        }
    }
    const Route heuristic = heuristic_route(problem, weight, tokens, prefill);
    // Its prefill bands: a W4A8 schedule for the W4A8 heuristic (the same bytes at every tile), a
    // route of the prefill route's summation order under the prefill profile.
    if (const auto band = tuned_prefill_route(problem, tokens, heuristic);
        band && route_admits(*band, weight, problem.policy) &&
        ((band->family == Family::W4a8Mma && heuristic.family == Family::W4a8Mma) ||
         (prefill && band->family != Family::W4a8Mma && same_summation(*band, heuristic)))) {
        return *band;
    }
    return heuristic;
}


} // namespace

std::optional<Route> prefill_family_route(PrefillRoute f, const Problem& p, const Weight& w,
                                          std::int32_t tokens, bool prefill) {
    switch (f) {
    case PrefillRoute::Auto:
        return std::nullopt;
    case PrefillRoute::A16: {
        // The automatic selection with BF16 activations: what an A16Only weight takes.
        Problem a16 = p;
        a16.policy  = LinearPolicy::A16Only;
        return auto_route(a16, w, tokens, prefill);
    }
    case PrefillRoute::W4a8: {
        if (!allows_w4a8(w, p.policy)) return std::nullopt;
        const auto r = w4a8_route(p, w, tokens);
        if (!r) return std::nullopt;
        // Its tuned prefill band, as the automatic W4A8 call takes it (the same bytes).
        if (const auto band = tuned_prefill_route(p, tokens, *r);
            band && band->family == Family::W4a8Mma && route_admits(*band, w, p.policy)) {
            return *band;
        }
        return r;
    }
    case PrefillRoute::I8:
        return i8_route(p, w, tokens);
    case PrefillRoute::Mmq:
        if (ggml_mmq_format(p.qtype) && (p.n % ggml_mmq::kRowMultiple) == 0 &&
            mmq_admits(w, p.policy)) {
            return Route{Family::GgmlMmq, 0};
        }
        return std::nullopt;
    case PrefillRoute::W4a4: {
        if (!allows_w4a4(w, p.policy)) return std::nullopt;
        const Route r{Family::W4a4Mma, static_cast<std::uint8_t>(
                                           heuristic_w4a4(p.n, tokens, prefill && tokens >= 17))};
        if (route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    }
    return std::nullopt;
}

std::optional<Route> decode_family_route(DecodeRoute f, const Problem& p, const Weight& w,
                                         std::int32_t tokens, bool prefill) {
    switch (f) {
    case DecodeRoute::Auto:
        return std::nullopt;
    case DecodeRoute::A16: {
        Problem a16 = p;
        a16.policy  = LinearPolicy::A16Only;
        return auto_route(a16, w, tokens, prefill);
    }
    case DecodeRoute::R8: {
        // Decode where the activation fits a CTA; the verify widths on the pre-quantized form of
        // every format that has it (the dense tuner's R8 candidates, r8_verify_tunable).
        const bool runs = tokens <= kR8DecodeTokens ? r8_call(w, p.policy, tokens)
                                                    : r8_verify_tunable(w, p.policy, tokens);
        if (!runs) return std::nullopt;
        const Route r{Family::R8Mma, static_cast<std::uint8_t>(heuristic_r8(p, tokens))};
        if (route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    case DecodeRoute::Vec: {
        const Route r{Family::GgmlVec, 0};
        if (vec_call(w, p.policy, tokens) && route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    case DecodeRoute::Mmq: {
        const Route r{Family::GgmlMmq, 0};
        if (mmq_call(w, p.policy, p.n, tokens) && route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    }
    return std::nullopt;
}

namespace {

constexpr PrefillRoute kPinnedPrefillFamilies[] = {
    PrefillRoute::A16, PrefillRoute::W4a8, PrefillRoute::I8, PrefillRoute::Mmq, PrefillRoute::W4a4};

constexpr DecodeRoute kPinnedDecodeFamilies[] = {DecodeRoute::A16, DecodeRoute::R8,
                                                 DecodeRoute::Vec, DecodeRoute::Mmq};

std::optional<Route> pinned_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                                  bool prefill) {
    const RoutePins pins = route_pins();
    if (tokens >= kPrefillPinMinTokens && pins.prefill != PrefillRoute::Auto) {
        std::optional<Route> mine;
        std::uint32_t could = 0;
        for (const PrefillRoute f : kPinnedPrefillFamilies) {
            const auto r = prefill_family_route(f, problem, weight, tokens, prefill);
            if (!r) continue;
            could |= family_bit(f);
            if (f == pins.prefill) mine = r;
        }
        note_prefill_pin(mine.has_value(), could);
        return mine;
    }
    if (tokens >= 1 && tokens <= kDecodePinMaxTokens && pins.decode != DecodeRoute::Auto) {
        std::optional<Route> mine;
        std::uint32_t could = 0;
        for (const DecodeRoute f : kPinnedDecodeFamilies) {
            const auto r = decode_family_route(f, problem, weight, tokens, prefill);
            if (!r) continue;
            could |= family_bit(f);
            if (f == pins.decode) mine = r;
        }
        note_decode_pin(mine.has_value(), could);
        return mine;
    }
    return std::nullopt;
}


} // namespace

bool r8_verify_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 2 && tokens <= kR8VerifyTokens && allows_r8(w, policy) &&
           r8_verify_compiled(w.qtype);
}

constexpr std::int32_t kVoltaCutlassMinTokens = 17;

constexpr std::int32_t kVoltaTileMaxTokens    = 256;

constexpr double kVoltaCopyMinDramGbps        = 870.0 / 1.13;

bool volta_cutlass_takes(std::int32_t tokens) {
    static PerDevice<bool> enabled;
    return tokens >= kVoltaCutlassMinTokens && enabled.get([](int) {
        if (!volta_cutlass_compiled()) return false;
        try {
            return !device_class().features.mma_tensor;
        } catch (const std::exception&) { return false; }
    });
}

bool volta_copy_pays(std::int32_t tokens) {
    static PerDevice<bool> fast_dram;
    return tokens >= kVoltaTileMaxTokens || fast_dram.get([](int) {
        try {
            return device_class().dram_gbps() >= kVoltaCopyMinDramGbps;
        } catch (const std::exception&) { return false; }
    });
}

bool volta_cutlass_runs(bool only, std::int32_t tokens) {
    return volta_cutlass_takes(tokens) && (only || volta_copy_pays(tokens));
}

} // namespace ninferno::ops::detail::generic
