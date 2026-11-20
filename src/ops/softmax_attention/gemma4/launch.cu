#include "ops/softmax_attention/gemma4/launch.h"
#include "core/per_device.h"

#include "core/device.h"
#include "core/generation.h"
#include "core/pdl.cuh"
#include "core/kernel_occupancy.h"
#include "ops/common/arch_limits.h"
#include "ops/common/math.h"
#include "ops/generic/kernel_facts.h"
#include "ops/softmax_attention/gemma4/kernel.cuh"
#include "ops/softmax_attention/gemma4/volta_decode.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>

namespace ninferno::ops::detail {
namespace {

// Prefill tiles: 32 packed rows x 32 keys at D=256 and D=512. D=256 runs on 2x2 warps in 40 KiB, so
// two CTAs share an SM; D=512 runs on 2x4 warps in 80 KiB, the head split into four 128-column warp
// slices so no warp holds more than 64 FP32 accumulators and Q never occupies shared memory.
// Measured interleaved against 64x64 and 64x32 at D=256 (32x32 faster at 11 of 12 extents of
// T=512..4096, base 0 and 3000, typically by 7-19%) and against 32x16 and 16x32 at D=512 (32x32
// faster everywhere).
template <int D>
struct PrefillConfig;

template <>
struct PrefillConfig<256> {
    static constexpr int WR = 2;
    static constexpr int Bc = 32;
};

template <>
struct PrefillConfig<512> {
    static constexpr int WR = 2;
    static constexpr int Bc = 32;
};

// The wide D=512 prefill tile: 64 rows on 4 x 2 warps of 256 columns (96 KiB). Per warp the same
// softmax and score exchange as the 32-row tile over twice the MMAs, and half the K/V reads per
// row. The same bytes: 8-token row blocks of group 8 never cross a 32-key tile, and the exchange
// sums the four 128-column score partials in the same order. RTX 4090 (sm_89), interleaved bench
// (ga_04/ga_06, time against the 32-row tile): 31B full layer (4 KV heads) T 1024 0.94 at base 0,
// 0.90 at 8192 and 20480; T 2048 0.92 / 0.90 / 0.90 at 0 / 4096 / 16384; 26B full layer (2 KV
// heads) T 1024 0.92 / 0.90 / 0.90, T 2048 0.92 / 0.90 / 0.90. Below 1024 tokens it loses from
// base 0 (31B T 768 1.04, T 512 1.26; 26B T 512 1.58) though it wins from base 4096 (0.90-0.96):
// the host does not know a call's base, so it takes 1024 tokens and more. Measured for group 8
// (Gemma 4 31B, 26B) on BF16 planes only; a 64-row block of group 1 or of a runtime group would
// fold other key tiles than the 32-row one, and those keep it.
using WideGemma4Geometry                       = Gemma4Geometry<512, 8, 256>;
constexpr int kWidePrefillWR                   = 4;
constexpr std::int32_t kWidePrefillMinTokens   = 1024;
std::atomic<Gemma4PrefillSlices> g_prefill_slices{Gemma4PrefillSlices::Auto};
std::atomic<Gemma4PrefillChain> g_prefill_chain{Gemma4PrefillChain::Auto};

// D=64 (gpt-oss) is one 64-wide warp column; four warp rows as at D=128, 8 KiB.
template <>
struct PrefillConfig<64> {
    static constexpr int WR = 4;
    static constexpr int Bc = 32;
    // Full layers (no window) take 64-key tiles: half the tiles' barriers and softmax rescales per
    // key. RTX 5090, oss-full T=2048 at 32K: BF16 6.39 -> 6.20 ms, FP8 6.89 -> 6.61; T=512 at 32K
    // BF16 2.01 -> 1.95, FP8 2.26 -> 2.14. The 128-key sliding window loses 11-15 % with them.
    static constexpr int FullBc = 64;
};

// The prefill key tile of a layer: PrefillConfig::FullBc where a geometry has one and the layer
// has no window, else PrefillConfig::Bc.
template <int D>
constexpr int prefill_full_bc() {
    if constexpr (requires { PrefillConfig<D>::FullBc; }) {
        return PrefillConfig<D>::FullBc;
    } else {
        return PrefillConfig<D>::Bc;
    }
}

// D=128 (DFlash drafter) is one warp column; four warp rows give 64-row CTAs of 128 threads in
// 16 KiB (on Volta the HMMA warp scratch adds to that, and prefill_bc may halve the key tile).
// The drafter never runs T > 16, so this route is correct but untuned at this width.
template <>
struct PrefillConfig<128> {
    static constexpr int WR = 4;
    static constexpr int Bc = 32;
};

// Decode key tile. At D >= 256 one CTA holds all rows of a KV group up to 32 rows (1 or 2 warp
// rows). A 128-wide head is a single warp column, so a CTA holds up to four warp rows (64 rows,
// 128 threads, 16 KiB): the 26B drafter's whole T=16 GQA block, or half of the 31B one's. At
// T=16, 32k keys, four warp rows measured 2-5% faster than two on the 64-head drafter and equal
// on the 32-head one; eight (one 256-thread CTA per SM) were no faster than four; raising the
// occupancy bound to 3 or 4 CTAs per SM was flat or slower.
constexpr int kDecodeBc              = 32;
constexpr int kDecodeMaxWarpRows     = 2;
constexpr int kDecodeMaxWarpRowsD128 = 4;

// NINFERNO-SPECIAL(arch:sm75): a 32-key tile of D=512 takes 80 KiB, over Turing's 64 KiB; half the
// keys per tile (40 KiB) keep every row, split and combine the same. Every other build keeps the
// measured 32. Sized with the FP8 scales, so BF16 and FP8 planes of a geometry share one tile; a
// BF16-only instance (the wide D=512 prefill tile) is sized without them.
// (FP8 tiles of codes alone, D = 512, are smaller than the BF16 tile and are sized by it.)
// The build's own instance of the tile's architecture facts is the build's compile-time limits
// (#912): the kernels and the plan of this build's boards read one shape.
static_assert(gemma4_arch_facts(kGemma4BuildArch).max_shared_per_block ==
                      static_cast<int>(kMaxSharedBytesPerBlock) &&
                  gemma4_arch_facts(kGemma4BuildArch).shared_per_sm ==
                      static_cast<int>(kSharedBytesPerSm) &&
                  gemma4_arch_facts(kGemma4BuildArch).reserved_per_block ==
                      kSharedBytesReservedPerBlock,
              "Gemma4ArchFacts of the build differ from ops/common/arch_limits.h");

// Arch: the architecture whose build compiles the tile (#912; the kernels: this build's).
template <class Geo, int WR, int Bc, bool WithFp8 = true, int Arch = kGemma4BuildArch>
inline constexpr int kFittingBc =
    std::max(Gemma4Tile<Geo, WR, Bc, false, Arch>::SmemBytes,
             WithFp8 ? Gemma4Tile<Geo, WR, Bc, true, Arch>::SmemBytes : 0) <=
            gemma4_arch_facts(Arch).max_shared_per_block
        ? Bc
        : Bc / 2;

// The prefill key tile: kFittingBc, halved while the SM's shared memory holds fewer than the eight
// warps the measured tiles run (PrefillConfig: D=256 on 2x2 warps, two CTAs an SM; D=512 on 2x4,
// one). NINFERNO-SPECIAL(arch:sm75): Turing's 64 KiB a SM holds one 40 KiB D=256 tile: four warps
// stalled on their own ldmatrix and MMA latencies (RTX 2080 Ti, Gemma 4 12B 2K prefill, ncu:
// issue slots 18 % busy, occupancy 12.5 %, 1.36 ms a sliding layer). Its 16-key tile (20 KiB)
// fits three CTAs by shared memory, and the registers hold two (ncu's block limit). Every other
// build holds the measured tiles at 32 keys (Ampere, Ada and Blackwell 100 KiB a SM, Volta 96).
// Arch: the architecture whose build compiles the tile (#912, #1112; the kernels: this build's).
template <class Geo, int WR, int Bc, bool WithFp8 = true, int Arch = kGemma4BuildArch>
constexpr int prefill_bc() {
    constexpr Gemma4ArchFacts facts = gemma4_arch_facts(Arch);
    constexpr int fitting           = kFittingBc<Geo, WR, Bc, WithFp8, Arch>;
    constexpr int bytes =
        std::max(Gemma4Tile<Geo, WR, fitting, false, Arch>::SmemBytes,
                 WithFp8 ? Gemma4Tile<Geo, WR, fitting, true, Arch>::SmemBytes : 0);
    constexpr int ctas  = facts.shared_per_sm / (bytes + facts.reserved_per_block);
    constexpr int warps = Gemma4Tile<Geo, WR, fitting, false, Arch>::Warps;
    if constexpr (ctas * warps < 8 && fitting > 16) {
        return prefill_bc<Geo, WR, fitting / 2, WithFp8, Arch>();
    } else {
        return fitting;
    }
}
template <class Geo, int WR, int Bc, bool WithFp8 = true, int Arch = kGemma4BuildArch>
inline constexpr int kPrefillBc = prefill_bc<Geo, WR, Bc, WithFp8, Arch>();

// The wide tile only where it keeps the 32-row tile's keys per tile: a halved key tile would fold
// the online softmax over other key tiles and change the bytes (V100: 96 KiB per block, which the
// wide tile's BF16 planes and score exchange fill exactly; Turing's 64 KiB holds neither at 32).
// Nor where the narrow tile runs Volta's m8n8k4 (Gemma4Tile::Volta) and the wide one cannot (its
// 256-column accumulators do not fit the planes for the output conversion): the two would sum in
// other orders, and a call's bytes would follow its width.
// Of the tile `Arch`'s build compiles (#1112: a plan prices a substituted board's own).
template <int Arch>
inline constexpr bool kWidePrefillFitsOf =
    Gemma4Tile<WideGemma4Geometry, kWidePrefillWR, PrefillConfig<512>::Bc, false, Arch>::Volta ==
        Gemma4Tile<Gemma4Geometry<512, 8>, PrefillConfig<512>::WR, PrefillConfig<512>::Bc, false,
                   Arch>::Volta &&
    kFittingBc<WideGemma4Geometry, kWidePrefillWR, PrefillConfig<512>::Bc, false, Arch> ==
        PrefillConfig<512>::Bc &&
    kFittingBc<Gemma4Geometry<512, 8>, PrefillConfig<512>::WR, PrefillConfig<512>::Bc, true,
               Arch> == PrefillConfig<512>::Bc;
inline constexpr bool kWidePrefillFits = kWidePrefillFitsOf<kGemma4BuildArch>;

template <int Arch = kGemma4BuildArch>
bool wide_prefill(std::int32_t tokens, bool fp8) {
    constexpr bool kFp8Hardware = gemma4_arch_facts(Arch).fp8_hardware;
    if (!kWidePrefillFitsOf<Arch>) return false;
    switch (g_prefill_slices.load(std::memory_order_relaxed)) {
    case Gemma4PrefillSlices::Narrow:
        return false;
    case Gemma4PrefillSlices::Wide:
        return !fp8 || kFp8Hardware;
    case Gemma4PrefillSlices::Auto:
        break;
    }
    // Every GPU whose shared memory holds the tile whole (kWidePrefillFits): the RTX 5090 (gb202,
    // Gemma 4 31B UD-Q4_K_XL prefill of a 16K prompt) gains 1.5-1.7 % as the 4090 did,
    // and 5.0 % with the token-major W4A8 grid beside it (2946 -> 3093 t/s).
    // FP8 rows take it too where their D = 512 tile holds the codes alone (64 KiB wide).
    return (!fp8 || kFp8Hardware) && tokens >= kWidePrefillMinTokens;
}


// The decode key tile of a geometry, at its widest decode CTA.
template <class Geo, int Arch = kGemma4BuildArch>
inline constexpr int kGeoDecodeBc =
    kFittingBc<Geo, (Geo::D <= 128 ? kDecodeMaxWarpRowsD128 : kDecodeMaxWarpRows), kDecodeBc, true,
               Arch>;

// The architecture whose tiles a board's plan prices (#912): this build's where the board runs
// this build's kernels (or no snapshot names its kernels), else the release build of its
// architecture (kernel_facts::sass_arch), whose snapshot holds those kernels' facts. `body` takes
// the architecture as a template argument and its snapshot label.
// `body` on the tiles of the build a snapshot label names (with_tile_arch's list).
template <class Body>
decltype(auto) with_label_tile_arch(std::string_view arch, Body&& body) {
    if (arch == "70") return body.template operator()<70>(std::string_view("70"));
    if (arch == "75") return body.template operator()<75>(std::string_view("75"));
    if (arch == "80") return body.template operator()<80>(std::string_view("80"));
    if (arch == "86") return body.template operator()<86>(std::string_view("86"));
    if (arch == "89") return body.template operator()<89>(std::string_view("89"));
    if (arch == "90") return body.template operator()<90>(std::string_view("90"));
    if (arch == "100") return body.template operator()<100>(std::string_view("100"));
    if (arch == "120a") return body.template operator()<120>(std::string_view("120a"));
    // A snapshot label this list lacks: no tile of its build here, so nothing of the board's
    // (never this build's tile in its place).
    throw std::logic_error("gemma4_attention: no tile of the sm_" + std::string(arch) +
                           " build here (with_tile_arch)");
}

template <class Body>
decltype(auto) with_tile_arch(Body&& body) {
    namespace kf                          = generic::kernel_facts;
    const std::optional<std::string> arch = kf::sass_arch();
    if (arch && *arch != kf::build_arch()) {
        return with_label_tile_arch(*arch, std::forward<Body>(body));
    }
    return body.template operator()<kGemma4BuildArch>(kf::build_arch());
}

// D<=128 splits: at least 64 keys each, and at most 64 per row group, under the resident wave
// (gemma4_attention_decode_plan). Every split costs T * Hq * D * 4 bytes of FP32 partials each
// way, so few-token targets gain from more splits over long keys and wide drafter blocks less.
// 64 per row group measured on full D<=128 layers of targets (gpt-oss-20b plain decode at 69K on
// the RTX 5090 177.6 -> 221.6 tok/s against 16; +8% long on the RTX 4090 and 5080). The 16 before
// it came from the Gemma 4 DFlash drafters at T=16 (16 splits 10-17% faster than the full wave
// at 2k-8k keys on the 32-head one), which ninferno-serve does not run: it speculates Gemma 4 with
// the MTP assistant only. A drafter served here would want its own row-group bound.
// Both are counts, the same on every board: the 64 splits a row group measured on three boards
// (RTX 5090, 4090, 5080), the 64-key floor on the RTX 5090.
constexpr int kDecodeMinSplitKeysD128   = 64;
constexpr int kDecodeSplitsPerGroupD128 = 64;

template <int D, class Body>
void with_group(std::int32_t group, Body&& body) {
    switch (group) {
    case 1:
        body.template operator()<Gemma4Geometry<D, 1>>();
        return;
    case 2:
        body.template operator()<Gemma4Geometry<D, 2>>();
        return;
    case 4:
        body.template operator()<Gemma4Geometry<D, 4>>();
        return;
    case 8:
        body.template operator()<Gemma4Geometry<D, 8>>();
        return;
    case 16:
        body.template operator()<Gemma4Geometry<D, 16>>();
        return;
    default:
        // Every other group up to 16 shares one instance that reads the group at run time.
        if (group < 1 || group > kGemma4MaxGroup) {
            throw std::invalid_argument("gemma4_attention: GQA group must be 1..16");
        }
        body.template operator()<Gemma4Geometry<D, 0>>();
        return;
    }
}

// Head widths 64, 128, 256 and 512 (the kernels' 128-column warp slices), any head counts whose
// group is 1..16.
template <class Body>
void with_geometry(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                   Body&& body) {
    if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("gemma4_attention: query heads must be a multiple of KV heads");
    }
    const std::int32_t group = q_heads / kv_heads;
    switch (head_dim) {
    case 64:
        with_group<64>(group, body);
        return;
    case 128:
        with_group<128>(group, body);
        return;
    case 256:
        with_group<256>(group, body);
        return;
    case 512:
        with_group<512>(group, body);
        return;
    default:
        throw std::invalid_argument("gemma4_attention: head_dim must be 64, 128, 256 or 512");
    }
}

// Decode instances exist for WR in {1, 2} at every width and WR = 4 at D=128 only.
template <class Geo, class Body>
void with_warp_rows(int warp_rows, Body&& body) {
    if (warp_rows == 1) {
        body.template operator()<1>();
    } else if (warp_rows == 2) {
        body.template operator()<2>();
    } else if constexpr (Geo::D <= 128) {
        if (warp_rows != 4) {
            throw std::invalid_argument("gemma4_attention: inconsistent decode plan");
        }
        body.template operator()<4>();
    } else {
        throw std::invalid_argument("gemma4_attention: inconsistent decode plan");
    }
}

// Smallest power-of-two warp-row count covering `rows`, capped per head width.
int decode_warp_rows(std::int32_t head_dim, std::int32_t rows) {
    const int cap = head_dim <= 128 ? kDecodeMaxWarpRowsD128 : kDecodeMaxWarpRows;
    int warp_rows = 1;
    while (warp_rows < cap && 16 * warp_rows < rows) { warp_rows *= 2; }
    return warp_rows;
}

// Every instance exceeds the default 48 KiB dynamic shared-memory ceiling.
template <class Geo, int WR, int Bc, bool Partial, bool Fp8 = false, bool Chain = false>
auto prepared_kernel() {
    using Tile        = Gemma4Tile<Geo, WR, Bc, Fp8>;
    const auto kernel = gemma4_attention_kernel<Geo, WR, Bc, Partial, Fp8, Chain>;
    static ::ninferno::PerDevice<bool> attrib;
    ::ninferno::allow_dynamic_smem(attrib, kernel, Tile::SmemBytes);
    return kernel;
}

// Bf16Only: an instance compiled for BF16 planes alone (the caller never passes FP8 ones).
template <class Geo, int WR, int Bc, bool Partial, bool Bf16Only = false, bool Chain = false>
void launch_attention(const Gemma4AttentionArgs& args, dim3 grid, cudaStream_t stream) {
    const auto body = [&]<bool Fp8>() {
        using Tile        = Gemma4Tile<Geo, WR, Bc, Fp8>;
        const auto kernel = prepared_kernel<Geo, WR, Bc, Partial, Fp8, Chain>();
        kernel<<<grid, Tile::Threads, Tile::SmemBytes, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    };
    if constexpr (Bf16Only) {
        if (args.k_scale != nullptr) {
            throw std::logic_error("gemma4_attention: FP8 planes on a BF16-only instance");
        }
        body.template operator()<false>();
    } else if (args.k_scale != nullptr) {
        body.template operator()<true>();
    } else {
        body.template operator()<false>();
    }
}

// The kernel-facts key of gemma4_attention_kernel<Geo, WR, Bc, Partial, Fp8, Chain> (#830).
template <class Geo, int WR, int Bc, bool Partial, bool Fp8, bool Chain>
generic::kernel_facts::Gemma4KernelKey attention_key() {
    generic::kernel_facts::Gemma4KernelKey key;
    key.kernel  = "attention";
    key.D       = Geo::D;
    key.G       = Geo::G;
    key.slice   = Geo::Width;
    key.rows    = WR;
    key.Bc      = Bc;
    key.partial = Partial;
    key.fp8     = Fp8;
    key.chain   = Chain;
    return key;
}

// The CTAs of gemma4_attention_kernel<Geo, WR, Bc, Partial, Fp8, Chain> one SM holds, from its
// facts (gemma4_resident_ctas_per_sm); the live fallback prepares the kernel (its shared-memory
// opt-in) only where it asks the device.
// Arch / tile_arch: the architecture the tile is sized for and its snapshot label (with_tile_arch);
// only this build's tile has a kernel here to ask live.
template <class Geo, int WR, int Bc, bool Partial, bool Fp8 = false, bool Chain = false,
          int Arch = kGemma4BuildArch>
int attention_resident_per_sm(std::string_view tile_arch = generic::kernel_facts::build_arch()) {
    using Tile                = Gemma4Tile<Geo, WR, Bc, Fp8, Arch>;
    std::function<int()> live = []() -> int {
        throw std::logic_error("gemma4_attention: another architecture's tile has no kernel here");
    };
    if constexpr (Arch == kGemma4BuildArch) {
        live = [] {
            const auto kernel = prepared_kernel<Geo, WR, Bc, Partial, Fp8, Chain>();
            return kernel_resident_ctas_per_sm(reinterpret_cast<const void*>(kernel), Tile::Threads,
                                               Tile::SmemBytes);
        };
    }
    return gemma4_resident_ctas_per_sm(attention_key<Geo, WR, Bc, Partial, Fp8, Chain>(),
                                       Tile::Threads, Tile::SmemBytes, live, tile_arch);
}

// CTAs of one decode instance the whole device holds at once.
template <class Geo, int WR>
int decode_resident_ctas() {
    // The board's own tile (#912): its key tile, threads and shared bytes, as its release build
    // compiles them.
    return with_tile_arch([&]<int Arch>(std::string_view label) {
               return attention_resident_per_sm<Geo, WR, kGeoDecodeBc<Geo, Arch>, true, false,
                                                false, Arch>(label);
           }) *
           device_class().sms;
}

// The layer's KV heads: a page pool keeps them on its fourth axis.
std::int32_t kv_heads_of(const Gemma4AttentionLaunch& launch) {
    return launch.address.tables != nullptr ? launch.cache_k.ne[3] : launch.cache_k.ne[2];
}

Gemma4AttentionArgs make_args(const Gemma4AttentionLaunch& launch) {
    return {
        .q       = static_cast<const __nv_bfloat16*>(launch.q.data),
        .cache_k = launch.cache_k.data,
        .cache_v = launch.cache_v.data,
        .k_scale =
            launch.k_scale != nullptr ? static_cast<const float*>(launch.k_scale->data) : nullptr,
        .v_scale =
            launch.v_scale != nullptr ? static_cast<const float*>(launch.v_scale->data) : nullptr,
        .positions   = static_cast<const std::int32_t*>(launch.positions.data),
        .out         = static_cast<__nv_bfloat16*>(launch.out.data),
        .partial_acc = nullptr,
        .partial_m   = nullptr,
        .partial_l   = nullptr,
        .tokens      = launch.q.ne[2],
        .capacity    = launch.address.capacity,
        .window      = launch.window,
        .block       = launch.block ? 1 : 0,
        .splits      = 1,
        .scale_log2  = launch.scale * 1.4426950408889634074f,
        .q_heads     = launch.q.ne[1],
        .kv_heads    = kv_heads_of(launch),
        .kv          = launch.address,
        .batch_row   = 0,
        .sinks = launch.sinks != nullptr ? static_cast<const float*>(launch.sinks->data) : nullptr,
        .chain_ticket = nullptr,
        .chain_flags  = nullptr,
        .chain_state  = nullptr,
        .chain_blocks = 0,
        .chain_spans  = 0,
    };
}

// Batch row b of a batched call: its slice of q, positions and out, and its binding.
Gemma4AttentionArgs row_args(Gemma4AttentionArgs args, std::int32_t b, std::int32_t head_dim) {
    const std::int64_t columns = static_cast<std::int64_t>(b) * args.tokens;
    const std::int64_t q_row   = columns * args.q_heads * head_dim;
    args.q                     = args.q + q_row;
    args.out                   = args.out + q_row;
    args.positions             = args.positions + columns;
    args.batch_row             = b;
    return args;
}

} // namespace

Gemma4DecodePlan gemma4_attention_decode_plan(std::int32_t head_dim, std::int32_t q_heads,
                                              std::int32_t kv_heads, std::int32_t window,
                                              std::int32_t tokens) {
    if (head_dim <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || tokens < 1 ||
        tokens > kGemma4DecodeMaxTokens || window < 0) {
        throw std::invalid_argument("gemma4_attention decode plan: invalid inputs");
    }
    const std::int32_t rows       = tokens * (q_heads / kv_heads);
    const std::int32_t warp_rows  = decode_warp_rows(head_dim, rows);
    const std::int32_t row_groups = div_up(rows, 16 * warp_rows);
    // The most splits worth running. The device cuts a call's own key range into them (key tiles
    // of Bc, splits past the range empty), so a call's bytes follow from its positions and this
    // count. The count takes nothing from the envelope: a session's or a batch's key capacity
    // must not change the bytes of the keys a call actually has (a DFlash session near the end of
    // a tight context drafted other blocks than one with room). A sliding layer bounds the keys
    // by window + T - 1: one Bc tile per split, or at D=128 at least kDecodeMinSplitKeysD128 keys
    // and at most kDecodeSplitsPerGroupD128 splits per row group. A full layer: at D<=128 those
    // splits per row group, at D >= 256 the resident wave below.
    std::int64_t tiles = std::numeric_limits<std::int32_t>::max();
    int decode_bc      = kDecodeBc;
    with_geometry(head_dim, q_heads, kv_heads, [&]<class Geo>() {
        decode_bc =
            with_tile_arch([]<int Arch>(std::string_view) { return kGeoDecodeBc<Geo, Arch>; });
    });
    if (window > 0) {
        const std::int64_t keys = static_cast<std::int64_t>(window) + tokens - 1;
        std::int64_t split_keys = decode_bc;
        if (head_dim <= 128) {
            // Rounded up: at most kDecodeSplitsPerGroupD128 splits per row group, the full
            // layer's count, which sizes the window-free workspace (2049 keys once made 17).
            split_keys = std::max<std::int64_t>(
                kDecodeMinSplitKeysD128,
                div_up<std::int64_t>(keys, static_cast<std::int64_t>(kDecodeSplitsPerGroupD128) *
                                               row_groups));
        }
        tiles = std::max<std::int64_t>(1, div_up<std::int64_t>(keys, split_keys));
    } else if (head_dim <= 128) {
        tiles = static_cast<std::int64_t>(kDecodeSplitsPerGroupD128) * row_groups;
    }
    // One resident wave: every split of every row group and KV head runs concurrently. A second
    // wave measured no faster and doubles the partial traffic the combine has to read.
    // The SIMT route (no native BF16 MMA) launches its own kernel, splits x KV heads CTAs.
    int resident = 0;
    if (gemma4_volta_decode_takes(head_dim, rows)) {
        resident = gemma4_volta_decode_resident_ctas(head_dim, rows);
    } else {
        with_geometry(head_dim, q_heads, kv_heads, [&]<class Geo>() {
            with_warp_rows<Geo>(warp_rows,
                                [&]<int WR>() { resident = decode_resident_ctas<Geo, WR>(); });
        });
    }
    const std::int32_t wanted = resident / (kv_heads * row_groups);
    return {
        .rows       = rows,
        .warp_rows  = warp_rows,
        .row_groups = row_groups,
        .splits     = static_cast<std::int32_t>(std::clamp<std::int64_t>(wanted, 1, tiles)),
    };
}

void gemma4_attention_set_prefill_slices(Gemma4PrefillSlices slices) noexcept {
    g_prefill_slices.store(slices, std::memory_order_relaxed);
}

void gemma4_attention_set_prefill_chain(Gemma4PrefillChain chain) noexcept {
    g_prefill_chain.store(chain, std::memory_order_relaxed);
}

namespace {

// The prefill tile of a call: Body<Geo, TileGeo, WR, Bf16Only>; Arch: the build whose tile it is
// (this build's for a launch; a substituted board's own for the plan's bytes, #1112).
template <int Arch = kGemma4BuildArch, class Body>
void with_prefill_tile(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                       std::int32_t tokens, bool fp8, Body&& body) {
    with_geometry(head_dim, q_heads, kv_heads, [&]<class Geo>() {
        if constexpr (std::is_same_v<Geo, Gemma4Geometry<512, 8>>) {
            if (wide_prefill<Arch>(tokens, fp8)) {
                if constexpr (gemma4_arch_facts(Arch).fp8_hardware) {
                    if (fp8) {
                        body.template operator()<Geo, WideGemma4Geometry, kWidePrefillWR, false>();
                        return;
                    }
                }
                body.template operator()<Geo, WideGemma4Geometry, kWidePrefillWR, true>();
                return;
            }
        }
        body.template operator()<Geo, Geo, PrefillConfig<Geo::D>::WR, false>();
    });
}

bool chain_enabled() {
    switch (g_prefill_chain.load(std::memory_order_relaxed)) {
    case Gemma4PrefillChain::Off:
        return false;
    case Gemma4PrefillChain::On:
        return true;
    case Gemma4PrefillChain::Auto:
        break;
    }
    return true;
}

// CTAs of the chained kernel the device holds at once (its persistent grid).
template <class TileGeo, int WR, int Bc>
int chain_resident() {
    return attention_resident_per_sm<TileGeo, WR, Bc, false, false, true>() * device_class().sms;
}

// Whether a call chains (same bytes either way, so the device may decide): only where the plain
// launch's last wave is mostly idle. Its CTAs (row blocks x KV heads) carry equal work once the
// keys are long, so the plain launch reaches ctas / (waves * resident) of the full-wave speed; a
// chained unit pays its prologue (Q fragments, the first key tile, the state hand-off), which
// lost 2-3 % where the plain waves were already full (RTX 5090: 31B T=4096, 24.1 waves).
//
// Only where the chained tile measured near the plain one's speed in full waves: the generation
// row's attention_chain (core/generation.h: gb20x, RTX 5090 31B T=4096 -2.2 %; RTX 5080 26B
// T=1024 at 16K keys +5.5 %). On Ada the chained wide tile runs 25-35 % slower even in whole
// waves (RTX 4090: 26B 96 -> 72 TFLOP/s at 16K-60K keys, same registers and no spills as the
// plain tile), so the tail it fills never pays (RTX 4080 SUPER: Gemma 4 26B 16K prefill 6566 ->
// 6455 t/s). Every other generation keeps the plain launch (Ada measured, the rest unmeasured),
// and so does one newer than the table: its lent row's measured preference is not its own
// (attention_chain_measured, #300).
bool chain_measured_fast() {
    try {
        const DeviceClass& d = device_class();
        return attention_chain_measured(d.major, d.minor);
    } catch (const std::exception&) { return false; }
}

template <class TileGeo, int WR, int Bc>
bool chain_pays(std::int32_t ctas) {
    if (g_prefill_chain.load(std::memory_order_relaxed) == Gemma4PrefillChain::On) return true;
    if (!chain_measured_fast()) return false;
    const std::int64_t resident = chain_resident<TileGeo, WR, Bc>();
    const std::int64_t waves    = div_up<std::int64_t>(ctas, resident);
    return static_cast<double>(ctas) < 0.9 * static_cast<double>(waves * resident);
}

// Facts of a chained prefill launch of one batch row.
struct ChainPlan {
    std::int32_t blocks = 0; // row blocks per KV head
    std::size_t flags   = 0; // bytes of the ticket and the flags
    std::size_t bytes   = 0; // all of it, 0: the call runs plain
};

template <class Geo, class TileGeo, int WR, bool Bf16Only, int Arch = kGemma4BuildArch>
ChainPlan chain_plan(std::int32_t group, std::int32_t kv_heads, std::int32_t window, bool block,
                     std::int32_t tokens) {
    constexpr int Bc = kPrefillBc<TileGeo, WR, PrefillConfig<Geo::D>::Bc, !Bf16Only, Arch>;
    using Tile       = Gemma4Tile<TileGeo, WR, Bc, false, Arch>;
    // A row block inside one key tile (no span blocks: the last row's keys bound every block's)
    // of a full causal layer, wider than a decode call. D >= 256: the layers whose CTAs are few
    // and long (Gemma 4 full layers, one CTA per SM).
    // The chained kernel also holds its ticket in static shared memory: on V100 the wide tile's
    // planes and score exchange fill the 96 KiB a block may take exactly, and the ticket on top
    // made the shared-memory attribute fail (cudaErrorInvalidValue under a forced chain).
    constexpr bool fits = TileGeo::D >= 256 && TileGeo::Pow2 &&
                          (Tile::Br >> TileGeo::GLog2) <= Bc &&
                          Tile::SmemBytes + static_cast<int>(sizeof(int)) <=
                              gemma4_arch_facts(Arch).max_shared_per_block;
    if (!fits || window != 0 || block || tokens <= kGemma4DecodeMaxTokens || !chain_enabled()) {
        return {};
    }
    const std::int32_t blocks = div_up(tokens * group, (16 * WR / group) * group) + 1;
    const std::int64_t units  = static_cast<std::int64_t>(blocks) * kv_heads;
    ChainPlan plan;
    plan.blocks = blocks;
    plan.flags  = round_up<std::size_t>(sizeof(std::int32_t) * (1 + units), 256);
    plan.bytes  = plan.flags + static_cast<std::size_t>(units) * (TileGeo::Slice / 8 + 1) *
                                  Tile::Threads * sizeof(float4);
    return plan;
}

} // namespace

std::size_t gemma4_attention_chain_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                         std::int32_t kv_heads, std::int32_t window, bool block,
                                         bool fp8, std::int32_t tokens) {
    // The board's own tile (#1112): a substituted board of another architecture holds the chain
    // its release build's tile fits, never this build's.
    std::size_t bytes = 0;
    with_tile_arch([&]<int Arch>(std::string_view) {
        with_prefill_tile<Arch>(head_dim, q_heads, kv_heads, tokens, fp8,
                                [&]<class Geo, class TileGeo, int WR, bool Bf16Only>() {
                                    bytes = chain_plan<Geo, TileGeo, WR, Bf16Only, Arch>(
                                                q_heads / kv_heads, kv_heads, window, block, tokens)
                                                .bytes;
                                });
    });
    return bytes;
}

bool gemma4_tile_arch_known(std::string_view label) {
    try {
        with_label_tile_arch(label, []<int Arch>(std::string_view) { return Arch; });
        return true;
    } catch (const std::logic_error&) { return false; }
}

std::size_t gemma4_attention_chain_bound(std::int32_t head_dim, std::int32_t q_heads,
                                         std::int32_t kv_heads, std::int32_t window,
                                         std::int32_t max_tokens) {
    // Both grow with T: the narrow tile's widest call is just under the wide one's threshold.
    std::size_t bytes = 0;
    for (const std::int32_t tokens :
         {max_tokens, std::min(max_tokens, kWidePrefillMinTokens - 1)}) {
        bytes = std::max(bytes, gemma4_attention_chain_bytes(head_dim, q_heads, kv_heads, window,
                                                             false, false, tokens));
    }
    return bytes;
}

void gemma4_attention_prefill_launch(const Gemma4AttentionLaunch& launch, std::int32_t batch,
                                     void* chain, cudaStream_t stream) {
    const Gemma4AttentionArgs args = make_args(launch);
    const bool fp8                 = args.k_scale != nullptr;
    with_prefill_tile(
        launch.q.ne[0], launch.q.ne[1], args.kv_heads, args.tokens, fp8,
        [&]<class Geo, class TileGeo, int WR, bool Bf16Only>() {
            constexpr int Bc = kPrefillBc<TileGeo, WR, PrefillConfig<Geo::D>::Bc, !Bf16Only>;
            // One block more than the rows need: row blocks sit on the absolute-position grid, and
            // a call that starts off it spans one block more (see the kernel). A block holds Br / G
            // whole tokens (Br rows for a power-of-two group).
            const std::int32_t group      = args.q_heads / args.kv_heads;
            const std::int32_t block_rows = (16 * WR / group) * group;
            const std::int32_t blocks     = div_up(args.tokens * group, block_rows) + 1;
            // A full layer of a geometry with its own full-layer key tile (no chain below D=256).
            constexpr int FullBc = kPrefillBc<TileGeo, WR, prefill_full_bc<Geo::D>(), !Bf16Only>;
            if constexpr (FullBc != Bc) {
                if (args.window == 0) {
                    const dim3 grid(static_cast<unsigned>(blocks),
                                    static_cast<unsigned>(args.kv_heads), 1u);
                    for (std::int32_t b = 0; b < batch; ++b) {
                        launch_attention<TileGeo, WR, FullBc, false, Bf16Only>(
                            row_args(args, b, Geo::D), grid, stream);
                    }
                    return;
                }
            }
            const ChainPlan plan          = chain_plan<Geo, TileGeo, WR, Bf16Only>(
                group, args.kv_heads, args.window, args.block != 0, args.tokens);
            if (chain != nullptr && plan.bytes > 0 &&
                chain_pays<TileGeo, WR, Bc>(blocks * args.kv_heads)) {
                if (plan.blocks != blocks) {
                    throw std::logic_error(
                        "gemma4_attention: chain plan disagrees with the launch");
                }
                // A persistent grid of the resident CTAs; spans wanted: sixteen units per CTA.
                const int resident = chain_resident<TileGeo, WR, Bc>();
                for (std::int32_t b = 0; b < batch; ++b) {
                    Gemma4AttentionArgs row = row_args(args, b, Geo::D);
                    auto* base              = static_cast<unsigned char*>(chain);
                    row.chain_ticket        = reinterpret_cast<std::int32_t*>(base);
                    row.chain_flags         = row.chain_ticket + 1;
                    row.chain_state         = reinterpret_cast<float4*>(base + plan.flags);
                    row.chain_blocks        = blocks;
                    row.chain_spans         = std::max<std::int32_t>(
                        1, div_up<std::int32_t>(16 * resident, blocks * args.kv_heads));
                    CUDA_CHECK(cudaMemsetAsync(chain, 0, plan.flags, stream));
                    launch_attention<TileGeo, WR, Bc, false, Bf16Only, true>(
                        row, dim3(static_cast<unsigned>(resident)), stream);
                }
                return;
            }
            const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(args.kv_heads),
                            1u);
            for (std::int32_t b = 0; b < batch; ++b) {
                launch_attention<TileGeo, WR, Bc, false, Bf16Only>(row_args(args, b, Geo::D), grid,
                                                                   stream);
            }
        });
}

void gemma4_attention_decode_launch(const Gemma4AttentionLaunch& launch,
                                    const Gemma4DecodePlan& plan, std::int32_t batch,
                                    std::int32_t rows_per_launch, Tensor& partial_acc,
                                    Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    Gemma4AttentionArgs args = make_args(launch);
    args.partial_acc         = static_cast<float*>(partial_acc.data);
    args.partial_m           = static_cast<float*>(partial_m.data);
    args.partial_l           = static_cast<float*>(partial_l.data);
    args.splits              = plan.splits;
    with_geometry(launch.q.ne[0], launch.q.ne[1], args.kv_heads, [&]<class Geo>() {
        if (plan.rows != args.tokens * (args.q_heads / args.kv_heads) || plan.splits < 1 ||
            plan.warp_rows < 1 || plan.row_groups != div_up(plan.rows, 16 * plan.warp_rows) ||
            rows_per_launch < 1) {
            throw std::invalid_argument("gemma4_attention: inconsistent decode plan");
        }
        // One launch per rows_per_launch batch rows (RTX 5090, Gemma 4 26B C=4: the
        // per-row attention and combine launches were 15 % of the decode step).
        for (std::int32_t b = 0; b < batch; b += rows_per_launch) {
            const auto rows = static_cast<unsigned>(std::min(rows_per_launch, batch - b));
            const dim3 grid(static_cast<unsigned>(plan.row_groups),
                            static_cast<unsigned>(plan.splits),
                            static_cast<unsigned>(args.kv_heads) * rows);
            const dim3 combine_grid(static_cast<unsigned>((Geo::D + 127) / 128),
                                    static_cast<unsigned>(args.q_heads),
                                    static_cast<unsigned>(args.tokens) * rows);
            const Gemma4AttentionArgs row = row_args(args, b, Geo::D);
            const bool volta = gemma4_volta_decode_takes(Geo::D, plan.rows);
            if (volta) {
                gemma4_volta_decode_launch(row, Geo::D, kGemma4VoltaDecodeBc,
                                           dim3(grid.y, grid.z), stream);
            } else {
                with_warp_rows<Geo>(plan.warp_rows, [&]<int WR>() {
                    launch_attention<Geo, WR, kGeoDecodeBc<Geo>, true>(row, grid, stream);
                });
            }
            // The combine launches under the attention grid's tail (programmatic dependent): its
            // launch and prologue leave the chain.
            // The split boundaries of the route that wrote the partials.
            const auto combine = volta ? gemma4_attention_combine_kernel<Geo, kGemma4VoltaDecodeBc>
                                       : gemma4_attention_combine_kernel<Geo, kGeoDecodeBc<Geo>>;
            CUDA_CHECK(pdl::launch({combine_grid, dim3(kGemma4CombineWarps * 32), 0, stream},
                                   device_class().features.pdl, false, combine, row));
        }
    });
}

void gemma4_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                             const Gemma4KvLayer& cache, const Gemma4KvBinding& binding,
                             cudaStream_t stream) {
    const std::int32_t head_dim = k.ne[0];
    const std::int32_t kv_heads = k.ne[1];
    const std::int32_t tokens   = k.ne[2];
    const std::int64_t columns  = static_cast<std::int64_t>(tokens) * binding.batch;
    const std::int64_t chunks   = columns * kv_heads * (head_dim / 8);
    const auto blocks           = static_cast<unsigned>(div_up<std::int64_t>(chunks, 256));
    const auto body             = [&]<int D>() {
        if (binding.fp8) {
            // One warp per row.
            const std::int64_t rows = columns * kv_heads;
            gemma4_kv_append_fp8_kernel<D>
                <<<static_cast<unsigned>(div_up<std::int64_t>(rows, 8)), 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<std::uint8_t*>(cache.k.data),
                    static_cast<std::uint8_t*>(cache.v.data),
                    static_cast<float*>(cache.k_scale.data), static_cast<float*>(cache.v_scale.data),
                    static_cast<std::int32_t>(columns), kv_heads, tokens, binding.address);
        } else {
            gemma4_kv_append_kernel<D><<<blocks, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data),
                static_cast<__nv_bfloat16*>(cache.k.data),
                static_cast<__nv_bfloat16*>(cache.v.data), static_cast<std::int32_t>(columns),
                kv_heads, tokens, binding.address);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    if (head_dim == 64) {
        body.template operator()<64>();
    } else if (head_dim == 128) {
        body.template operator()<128>();
    } else if (head_dim == 256) {
        body.template operator()<256>();
    } else if (head_dim == 512) {
        body.template operator()<512>();
    } else {
        throw std::invalid_argument("gemma4_kv_append: unregistered head width");
    }
}

int gemma4_resident_ctas_per_sm(const generic::kernel_facts::Gemma4KernelKey& key, int threads,
                                std::size_t dynamic_shared_bytes, const std::function<int()>& live,
                                std::optional<std::string_view> tile_arch) {
    namespace kf                          = generic::kernel_facts;
    const std::optional<std::string> arch = kf::sass_arch();
    // The key, threads and shared bytes are the tile of `tile_arch`'s build (Gemma4Tile of its
    // Gemma4ArchFacts, #912): they are the board's where it runs that architecture's kernels;
    // nullopt: a key of no tile (the Volta decode kernel's), the board's on every architecture.
    const bool own_tile = arch && (!tile_arch || *arch == *tile_arch);
    const std::optional<kf::KernelFacts> facts =
        own_tile ? kf::gemma4_facts(*arch, key) : std::nullopt;
    if (facts) {
        const int per_sm = resident_ctas_per_sm(
            device_class(),
            KernelShape{facts->registers, facts->static_shared_bytes, threads, dynamic_shared_bytes,
                        /*dynamic_shared_opt_in=*/dynamic_shared_bytes > 0});
        if (per_sm < 1) {
            throw std::runtime_error(
                "gemma4_attention: the " + key.kernel +
                " kernel does not fit one SM by its facts (sm_" + *arch + ": D " +
                std::to_string(key.D) + " G " + std::to_string(key.G) + " slice " +
                std::to_string(key.slice) + " rows " + std::to_string(key.rows) + " Bc " +
                std::to_string(key.Bc) + ", " + std::to_string(facts->registers) + " registers, " +
                std::to_string(facts->static_shared_bytes) + " B static and " +
                std::to_string(dynamic_shared_bytes) + " B dynamic shared, " +
                std::to_string(threads) + " threads)");
        }
        return per_sm;
    }
    // No snapshot holds the kernel: a substituted board has no answer of its own (the host GPU's
    // kernels are not the board's), a real device its own kernel's.
    if (substituted_device_class() != nullptr) {
        throw std::runtime_error(
            "gemma4_attention: no kernel facts of the " + key.kernel +
            " kernel for the substituted board" + (arch ? " (sm_" + *arch + ")" : std::string()) +
            (arch && !own_tile
                 ? ": the tile priced is sm_" + std::string(tile_arch.value_or("?")) + "'s (#912)"
                 : std::string()));
    }
    return std::max(1, live());
}

} // namespace ninferno::ops::detail
