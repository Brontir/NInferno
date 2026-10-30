#pragma once

// Gemma 4 text attention over head-major slot storage; the contract is
// include/ninferno/ops/gemma4_attention.h.
//
// One kernel body serves both routes. A CTA owns one KV head and a block of rows, where the query
// heads of the GQA group are packed token-major into the MMA M dimension (row = token * G + head
// in group). Every K/V tile staged in shared memory therefore feeds every query head of the group.
//
// Warps form a WR x WC grid. Warp row wr owns 16 rows; warp column wc owns a slice of the head
// dimension (128 columns, or 256 on the wide D=512 prefill tile). Q stays in registers as the A
// fragments of the warp's slice. S = Q K^T is computed per 128-column part, the D / 128 partial
// score tiles are summed through shared memory in one fixed order (the same at every slice
// width), and every warp of a row group then runs the same online softmax on the same scores and
// multiplies its own BF16 P fragments (straight from registers) by the V slice it owns. A 512-wide
// head therefore never needs a full-width accumulator per warp or a resident Q tile.
//
// K and V are single-buffered: V(kb) streams in during QK/softmax of tile kb, K(kb+1) during
// softmax/PV of tile kb. Deeper K+V rings (2-4 stages of 16- or 32-key tiles) measured no faster on
// the decode route, which is bounded by one resident wave of CTAs rather than by the pipeline.
// Rows of a tile outside the CTA's key range are zero-filled instead of loaded, so a ring slot
// holding a dead or never-written row can never feed a NaN to the MMA.
//
// Prefill (Partial=false): grid (row blocks, Hkv). The key range is the union of the visible keys
// of the block's rows, so a sliding layer never stages a key outside its window. Row blocks are
// aligned to absolute positions, so a token's result does not depend on where its call began.
// Decode  (Partial=true):  grid (row groups, splits, Hkv * B). The union key range of the call is
// cut into equal Bc-aligned chunks derived on the device from positions; each live split writes
// FP32 unnormalized partials that gemma4_attention_combine_kernel merges. The B batch rows of one
// launch (gemma4_batch_row) each do exactly the single-row call's work on their own slice
// of q, positions and partials, so a row's bytes do not depend on how many rows share the launch.
//
// The Block mask (DFlash drafter full layer) only moves every row's last visible key to the call's
// last position, positions[T-1]. The union key range of a call, and therefore the decode split
// geometry and the combine, are the same as under the causal mask.
//
// FP8 planes (Fp8): a tile's codes stream in (cp.async) with its row scales beside the score
// exchange, and every code reaches an MMA widened to BF16 as scale * code. At D >= 256 the K codes
// go straight from shared memory into the score MMA's fragments (ldmatrix, widened in registers;
// Q takes the matching contraction order), and at D = 512 the V codes into the value MMA's
// (ldmatrix.trans and a byte permute; output n-tile pairs hold even and odd dims), so the D = 512
// tile holds the codes alone. Elsewhere the codes land in the upper half of the plane's BF16 tile
// and, after a barrier, are written back widened into the swizzled tile the BF16 MMA loop reads.

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/gemma4/fp8_row.cuh"
#include "ops/softmax_attention/gemma4/kv_address.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninferno::ops {

inline constexpr int kGemma4Slice = 128; // head-dim columns owned by one warp column

// The compile-time part of a layer's geometry: the head width and the GQA group size. The head
// counts themselves are runtime (Gemma4AttentionArgs), so every model whose group is a power of
// two up to 16 runs on the same instances. Group 0 is the instance for every other group size
// (3, 5, 6, 7, 9..15): the group is then read from the head counts at run time. It enters only
// the row <-> (token, head) mapping and the prefill row-block grid, never the inner loops. A head
// narrower than a slice (D = 64) is one warp column of its own width.
//
// SliceWidth: head-dim columns per warp column, a multiple of kGemma4Slice. A wider slice keeps the
// score partials of its 128-column parts apart, so the exchange sums the same D / 128 partials in
// the same order at every slice width (the bytes do not depend on it).
template <int HeadDim, int Group, int SliceWidth = kGemma4Slice>
struct Gemma4Geometry {
    static constexpr int D     = HeadDim;
    static constexpr int G     = Group; // 0: runtime group
    static constexpr int Slice = HeadDim < SliceWidth ? HeadDim : SliceWidth;
    static constexpr int WC    = HeadDim / Slice;
    // 128-column score partials: per warp column, and of the head (summed in index order).
    static constexpr int Sub   = Slice > kGemma4Slice ? Slice / kGemma4Slice : 1;
    static constexpr int Parts = WC * Sub;
    static constexpr bool Pow2 = G > 0 && (G & (G - 1)) == 0;
    static constexpr int GLog2 = G == 1 ? 0 : G == 2 ? 1 : G == 4 ? 2 : G == 8 ? 3 : 4;
    static_assert(G == 0 || (1 << GLog2) == G);
    static_assert(HeadDim % Slice == 0 && Slice % 64 == 0);
    static_assert(Sub == 1 || Slice % kGemma4Slice == 0);
};

// The GQA group of one call: a packed row's token (floor division, since a prefill row block that
// starts off the position grid begins below row 0) and its head within the group. A power-of-two
// group keeps its shifts and masks; the runtime group divides.
template <class Geo>
struct Gemma4Group {
    int runtime; // q_heads / kv_heads, used when Geo::G == 0

    __device__ __forceinline__ int size() const {
        if constexpr (Geo::G > 0) {
            return Geo::G;
        } else {
            return runtime;
        }
    }
    __device__ __forceinline__ int token(int row) const {
        if constexpr (Geo::Pow2) {
            return row >> Geo::GLog2;
        } else {
            const int g = size();
            return (row >= 0 ? row : row - (g - 1)) / g;
        }
    }
    __device__ __forceinline__ int member(int row) const {
        if constexpr (Geo::Pow2) {
            return row & (Geo::G - 1);
        } else {
            return row - token(row) * size();
        }
    }
};

// E4M3 converts in hardware from sm_89 (one cvt per code pair); below it cuda_fp8.h widens each
// code in software. The direct FP8 paths widen a tile once per warp row, so they pay only where the
// conversion is cheap: on sm_86 they halved the FP8 prefill (RTX 3090, 26B full at 32K 8.3 -> 4.4
// TFLOP/s), and that build keeps the shared-memory widening. One tree compiles one architecture
// (NINFERNO_BUILD_ARCH), so host and device code agree.
#if defined(NINFERNO_BUILD_ARCH)
inline constexpr bool kGemma4Fp8Hardware = NINFERNO_BUILD_ARCH >= 89;
#else
inline constexpr bool kGemma4Fp8Hardware = true;
#endif

// NINFERNO-SPECIAL(arch:sm70): Volta has none of the m16n8k16 MMAs the kernel is written in, and
// their FP32-FMA form (ops/common/mma.cuh) held a Gemma 4 12B prefill's attention at 3.7 TFLOP/s
// on the Quadro GV100 (383 ms of a 1913-token prefill's 1277). Its own mma.m8n8k4 takes f16
// operands with FP32 accumulation: K and V are rounded to f16 once as they are staged (bf16's
// 8-bit mantissa fits f16's 10; magnitudes saturate at 65504, as on Turing), Q as it is loaded and
// P as it is made, and each quad-pair computes one 8 x 8 block of a 16 x 16 output pair over the
// whole contraction (four m8n8k4 per 16-wide step), so no partial sums cross quad-pairs. The
// accumulators keep Volta's layout through a tile: the scores reach the softmax through the score
// exchange (scattered to the m16n8 lane layout the exchange holds), P and the rescale factors
// through a per-warp scratch, and the output converts once, through the freed K/V planes. Only the
// geometries with an exchange (D >= 256, Gemma 4's) take it.
#if defined(NINFERNO_BUILD_ARCH)
inline constexpr bool kGemma4VoltaHmma = NINFERNO_BUILD_ARCH >= 70 && NINFERNO_BUILD_ARCH < 75;
#else
inline constexpr bool kGemma4VoltaHmma = false;
#endif

template <class Geo, int WR, int Bc, bool Fp8 = false>
struct Gemma4Tile {
    static constexpr int Warps      = WR * Geo::WC;
    static constexpr int Threads    = Warps * 32;
    static constexpr int Br         = 16 * WR;
    // FP8 rows at D = 512 reach both MMAs straight from their codes (the kernel's KDirect and
    // VDirect): the tile holds the codes alone, half the bytes of a BF16 tile.
    static constexpr bool CodesOnly = Fp8 && kGemma4Fp8Hardware && Geo::D >= 512;
    static constexpr int PlaneBytes = Bc * Geo::D * (CodesOnly ? 1 : 2); // one K or V tile
    static constexpr int CodeOffset = CodesOnly ? 0 : PlaneBytes / 2;    // FP8 codes in a plane
    // Slice-partial score exchange; a 128-wide head (one warp column) needs none.
    static constexpr int RedBytes   = Geo::Parts > 1 ? WR * Geo::Parts * (Bc / 8) * 32 * 16 : 0;
    static constexpr int ScaleBytes = Fp8 ? 2 * Bc * 4 : 0; // FP32 row scales of K and V
    // Volta's m8n8k4 route where the two planes hold every warp's 16 x Slice FP32 accumulators
    // for the output conversion (not the wide D = 512 tile, which keeps the FMA form).
    static constexpr bool Volta =
        kGemma4VoltaHmma && Geo::Parts > 1 && Warps * 16 * Geo::Slice * 4 <= 2 * PlaneBytes;
    // Volta's per-warp scratch: P of its 16 rows (f16, Bc keys) and their 16 rescale factors.
    static constexpr int VoltaWarpBytes = Volta ? 16 * Bc * 2 + 16 * 4 : 0;
    static constexpr int SmemBytes  = 2 * PlaneBytes + RedBytes + ScaleBytes + Warps * VoltaWarpBytes;
    static_assert(Bc % 16 == 0 && (Bc * Geo::D / 8) % Threads == 0);
    static_assert(!Fp8 || (Bc * Geo::D / 16) % Threads == 0);
};

struct Gemma4AttentionArgs {
    const __nv_bfloat16* q;
    const void* cache_k; // BF16, or FP8_E4M3FN codes with k_scale / v_scale
    const void* cache_v;
    const float* k_scale;
    const float* v_scale;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    float* partial_acc;
    float* partial_m;
    float* partial_l;
    std::int32_t tokens;
    std::int32_t capacity;
    std::int32_t window;
    std::int32_t block;  // nonzero: Block mask, every row sees keys up to positions[T-1]
    std::int32_t splits; // decode: grid capacity of key chunks per KV head
    float scale_log2;
    std::int32_t q_heads;
    std::int32_t kv_heads;
    detail::Gemma4KvAddress kv; // where the binding of batch_row keeps its rows
    std::int32_t batch_row;
    // Optional per-query-head attention sink logits (FP32 [Hq], gpt-oss): exp(sink) joins every
    // row's softmax denominator and contributes no value.
    const float* sinks;
    // Chained prefill (gemma4_attention_kernel<..., Chain>), null otherwise; zeroed per launch.
    std::int32_t* chain_ticket; // units handed out so far
    std::int32_t* chain_flags;  // [Hkv * chain_blocks]: spans of the row block folded so far
    float4* chain_state;        // [Hkv * chain_blocks][(PVN + 1) * Threads]: the folded state
    std::int32_t chain_blocks;  // row blocks per KV head (the plain launch's grid.x)
    std::int32_t chain_spans;   // spans wanted per row block (a call with fewer keys takes fewer)
};

// Batch row b of a decode launch: its tokens' q, positions and out, its binding, and its own
// block of partials (rows_total rows of every split and KV head). Kept beside the arguments,
// never written into them: a kernel that assigns to its by-value argument struct copies it to
// the stack, and every read in the key loop then goes through local memory (184-byte frame,
// Gemma 4 31B decode C=4 several times slower).
struct Gemma4BatchRow {
    const __nv_bfloat16* q;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    std::int32_t batch_row;
    float* partial_acc;
    float* partial_m;
    float* partial_l;
};

__device__ __forceinline__ Gemma4BatchRow gemma4_batch_row(const Gemma4AttentionArgs& args, int b,
                                                           int D) {
    const std::int64_t columns = static_cast<std::int64_t>(b) * args.tokens;
    const std::int64_t q_row   = columns * args.q_heads * D;
    const std::int64_t partial =
        static_cast<std::int64_t>(b) * args.splits * args.q_heads * args.tokens;
    return {args.q + q_row,
            args.positions + columns,
            args.out + q_row,
            args.batch_row + b,
            args.partial_acc + (args.partial_acc != nullptr ? partial * D : 0),
            args.partial_m + (args.partial_m != nullptr ? partial : 0),
            args.partial_l + (args.partial_l != nullptr ? partial : 0)};
}

// The union key range [lo, hi) of a call, cut into `splits` chunks of equal Bc-aligned length
// (a floor above one tile was slower: RTX 4090, Gemma 4 26B -5..-16 %). Chunks past
// `active` are empty. Decode partials and the combine derive
// it identically.
__device__ __forceinline__ void gemma4_split_range(int lo, int hi, int splits, int bc, int split,
                                                   int& begin, int& end, int& active) {
    const int len   = hi - lo;
    const int chunk = round_up(max(div_up(len, splits), 1), bc);
    active          = div_up(len, chunk);
    begin           = lo + split * chunk;
    end             = min(hi, begin + chunk);
}

// One past the call's last position, positions[T-1] + 1. A verify call pads a row with fewer
// drafts than its width by repeating its last real position, so the call ends there and not at
// positions[0] + T: at the end of the context that is past the binding's capacity, where a paged
// binding has no table entry (Gemma 4 26B MTP read a garbage page there).
__device__ __forceinline__ int gemma4_call_end(const std::int32_t* positions, int tokens) {
    return positions[tokens - 1] + 1;
}

__device__ __forceinline__ void gemma4_union_range(const std::int32_t* positions, int tokens,
                                                   int window, int& lo, int& hi) {
    const int p0 = positions[0];
    lo           = window > 0 ? max(0, p0 - window + 1) : 0;
    hi           = gemma4_call_end(positions, tokens);
}

__device__ __forceinline__ int gemma4_load_acquire(const std::int32_t* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void gemma4_store_release(std::int32_t* p, int v) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Chain (prefill of a full causal layer): a persistent grid takes units from a ticket counter. A
// unit is one span of chain_tiles key tiles of one row block. Units are handed out span by span, so
// a unit's predecessor (its row block's previous span) holds a smaller ticket and has started
// before it: waiting for it cannot deadlock. A span continues the online softmax from the state
// its predecessor left in chain_state, and folds its tiles in the order and arithmetic of one CTA
// sweeping them all, so the output bytes are the plain launch's. Only the unit of work changes: one
// CTA per row block left most of a second wave idle (Gemma 4 26B, 2 KV heads, T=1024: 256 CTAs of
// equal work on 170 SMs).
// Fewest key tiles a span folds: a unit's prologue (Q fragments, the first key tile not overlapped,
// the state hand-off) costs about two tiles, and 16-tile spans lost 13-16 % at base 0 (RTX
// 5090, 26B/31B T=1024..2048, where the plain launch's triangle balanced itself).
inline constexpr int kGemma4ChainMinTiles = 64;

template <class Geo, int WR, int Bc, bool Partial, bool Fp8 = false, bool Chain = false>
__launch_bounds__(Gemma4Tile<Geo, WR, Bc, Fp8>::Threads, 1) __global__
    void gemma4_attention_kernel(const Gemma4AttentionArgs args) {
    static_assert(!(Chain && Partial), "a chained launch is a prefill launch");
    const Gemma4BatchRow row_ =
        gemma4_batch_row(args, Partial ? static_cast<int>(blockIdx.z) / args.kv_heads : 0, Geo::D);
    using Tile               = Gemma4Tile<Geo, WR, Bc, Fp8>;
    constexpr int D          = Geo::D;
    constexpr int WC         = Geo::WC;
    constexpr int Threads    = Tile::Threads;
    constexpr int Br         = Tile::Br;
    constexpr int KS         = Geo::Slice / 16; // QK contraction steps over the slice
    constexpr int QKN        = Bc / 8;            // score n-tiles
    constexpr int PVK        = Bc / 16;           // PV contraction steps over keys
    constexpr int PVN        = Geo::Slice / 8;  // output n-tiles of the slice
    constexpr int RowBytes   = D * 2;
    constexpr int RowChunks  = D / 8;
    constexpr int StageIters = Bc * RowChunks / Threads;
    constexpr int CodeChunks = D / 16; // 16-byte code chunks of an FP8 row
    constexpr int CodeIters  = Fp8 ? Bc * CodeChunks / Threads : 1;
    // FP8 K reaches the score MMA straight from its codes (score_tile) at D >= 256: every row warp
    // widens its own copy there, which the wide rows pay for (RTX 5090, T=2048 at 32K: 26B full
    // 21.2 -> 19.5 ms, 31B full 39.9 -> 37.2, sliding 272 -> 262 us), while D = 64 (gpt-oss)
    // lost 6 % and keeps the shared-memory widening.
    constexpr bool KDirect = Fp8 && kGemma4Fp8Hardware && D >= 256;
    // V likewise (value_tile) at D = 512: each pair of output n-tiles takes the even and the odd
    // head dims of a 16-dim chunk, so the epilogue writes four adjacent dims per lane (26B full
    // 19.7 -> 19.0 ms, 31B full 37.2 -> 36.2); the D = 256 sliding layers lost 6 % and keep it.
    constexpr bool VDirect = Fp8 && kGemma4Fp8Hardware && D >= 512;
    static_assert(Tile::CodesOnly == (KDirect && VDirect),
                  "a codes-only tile is read only directly");
    constexpr bool Volta = Tile::Volta;
    static_assert(!Volta || (!KDirect && !VDirect));
#if defined(__CUDA_ARCH__)
    static_assert(!Volta || NINFERNO_HMMA884, "the Volta route compiles only where m8n8k4 does");
#endif
    const float NegInf       = -CUDART_INF_F;

    extern __shared__ __align__(128) unsigned char gemma4_smem[];
    float4* red = reinterpret_cast<float4*>(gemma4_smem + 2 * Tile::PlaneBytes);
    float* row_scales =
        reinterpret_cast<float*>(gemma4_smem + 2 * Tile::PlaneBytes + Tile::RedBytes);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // Volta (Tile::Volta): the lane's quad-pair (one 8 x 8 block of a 16 x 16 output pair: rows
    // 8 * (vq & 1), columns 8 * (vq >> 1)), its row of A (vj, as the m8n8k4 fragments index), and
    // the warp's scratch: P [16][Bc] f16, then 16 rescale factors.
    [[maybe_unused]] const int vq = (lane >> 2) & 3;
    [[maybe_unused]] const int vj = (lane & 3) + 4 * (lane >> 4);
    [[maybe_unused]] __half* const vp = reinterpret_cast<__half*>(
        gemma4_smem + 2 * Tile::PlaneBytes + Tile::RedBytes + Tile::ScaleBytes +
        warp * Tile::VoltaWarpBytes);
    [[maybe_unused]] float* const valpha = reinterpret_cast<float*>(vp + 16 * Bc);
    // A decode call's combine is its programmatic dependent: it may launch once every CTA here
    // has started (one row's grid is one resident wave, so its CTAs take only the slots left
    // over; a launch of several rows starts its last wave first) and waits for this grid before
    // it reads the partials.
    if constexpr (Partial) {
        if (tid == 0) pdl::trigger_dependents();
    }
    const int wr   = warp / WC;
    const int wc   = warp - wr * WC;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;

    const Gemma4Group<Geo> group{args.q_heads / args.kv_heads};
    const int G          = group.size();
    const int tokens     = args.tokens;
    const int rows_total = tokens * G;
    const int window     = args.window;
    const int capacity   = args.capacity;
    // Prefill row blocks hold whole tokens: Br / G of them, Br rows for a power-of-two group, and
    // the rows past block_rows of the CTA (fewer than G) are idle for any other group. Decode row
    // groups are plain Br-row slices of the call's rows.
    const int block_rows = Geo::Pow2 || Partial ? Br : (Br / G) * G;

    [[maybe_unused]] __shared__ int chain_unit;
    for (;;) {
        int kv_head  = 0;
        int row_base = 0;
        int row_end  = rows_total; // rows of this CTA are [row_base, min(row_end, row_base + Br))
        int split    = 0;
        int span     = 0;    // Chain: the unit's span of its row block
        int chain_block = 0; // Chain: KV head * chain_blocks + row block
        int chain_tiles = 0; // Chain: key tiles per span (on the device: it follows the positions)
        if constexpr (Chain) {
            if (tid == 0) { chain_unit = atomicAdd(args.chain_ticket, 1); }
            __syncthreads();
            const int unit = chain_unit;
            __syncthreads(); // read by every thread before thread 0 takes the next ticket
            // Spans that hold a key tile of some row block (the call's last row sees the most
            // keys).
            const int last_tiles = div_up(gemma4_call_end(row_.positions, tokens), Bc);
            const int per_span   = args.chain_blocks * args.kv_heads;
            chain_tiles          = max(kGemma4ChainMinTiles, div_up(last_tiles, args.chain_spans));
            span                 = unit / per_span;
            if (span >= div_up(last_tiles, chain_tiles)) { break; }
            chain_block = unit - span * per_span;
        }
        if constexpr (Partial) {
            row_base = static_cast<int>(blockIdx.x) * Br;
            split    = static_cast<int>(blockIdx.y);
            kv_head  = static_cast<int>(blockIdx.z) % args.kv_heads;
        } else {
            // Row blocks sit on a grid of absolute positions (Br / G tokens each), not on the
            // call's first token: a block's key range and the order its tiles are folded in then
            // depend only on where its tokens are in the sequence, so a token's output is the same
            // whichever call it arrives in. A call that starts off the grid covers a partial first
            // block; the launch carries one block more than the rows need for it (empty when the
            // call is on the grid).
            int shift = 0;
            if constexpr (Geo::Pow2) {
                shift = (row_.positions[0] & ((Br >> Geo::GLog2) - 1)) << Geo::GLog2;
            } else {
                shift = (row_.positions[0] % (block_rows / G)) * G;
            }
            // Later row blocks see more keys; launch them first.
            int bx     = static_cast<int>(blockIdx.x);
            int blocks = static_cast<int>(gridDim.x);
            kv_head    = static_cast<int>(blockIdx.y);
            if constexpr (Chain) {
                blocks  = args.chain_blocks;
                kv_head = chain_block / blocks;
                bx      = chain_block - kv_head * blocks;
            }
            row_base = (blocks - 1 - bx) * block_rows - shift;
            if constexpr (!Geo::Pow2) { row_end = min(rows_total, row_base + block_rows); }
            if (row_base >= rows_total || row_base + block_rows <= 0) {
                if constexpr (Chain) { continue; }
                return;
            }
        }

        // Q A fragments of this warp's 16 rows over its head-dim slice, as the MMA reads them
        // (mma_a_bf16). They do not depend on the positions, so their loads overlap the positions
        // load.
        unsigned qf[KS][4];
        // Volta: the lane's row of A (8 * (vq & 1) + vj) over each 16-dim step, four m8n8k4
        // fragments of 4 dims (two f16x2 each).
        [[maybe_unused]] unsigned qv[KS][4][2];
        if constexpr (Volta) {
            const int row    = row_base + wr * 16 + 8 * (vq & 1) + vj;
            const bool valid = row >= 0 && row < row_end;
            const __nv_bfloat16* src =
                row_.q +
                (static_cast<std::int64_t>(group.token(row)) * args.q_heads + kv_head * G +
                 group.member(row)) * D +
                wc * Geo::Slice;
#pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint2 v = valid ? __ldg(reinterpret_cast<const uint2*>(src + ks * 16 + kk * 4))
                                          : make_uint2(0u, 0u);
                    qv[ks][kk][0] = bf16x2_to_f16x2(v.x);
                    qv[ks][kk][1] = bf16x2_to_f16x2(v.y);
                }
            }
        }
#pragma unroll
    for (int r = 0; r < 2 && !Volta; ++r) {
        const int row            = row_base + wr * 16 + gid + 8 * r;
        const bool valid         = row >= 0 && row < row_end;
        const int token          = group.token(row);
        const int head           = kv_head * G + group.member(row);
        // FP8 K tiles reach the MMA as each lane's four consecutive head dims 4 * tig .. + 3 of a
        // 16-dim step (one 32-bit word of codes), so the contraction runs over the step's dims in
        // the order (4t, 4t+1 | 4t+2, 4t+3) instead of (2t, 2t+1 | 2t+8, 2t+9): Q takes the same
        // order, and every product of the dot product is the same.
        const __nv_bfloat16* src = row_.q +
                                   (static_cast<std::int64_t>(token) * args.q_heads + head) * D +
                                   wc * Geo::Slice + (KDirect ? 4 : 2) * tig;
        constexpr int kHi = KDirect ? 2 : 8;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            qf[ks][r] =
                mma_a_bf16(valid ? __ldg(reinterpret_cast<const unsigned*>(src + ks * 16)) : 0u);
            qf[ks][2 + r] = mma_a_bf16(
                valid ? __ldg(reinterpret_cast<const unsigned*>(src + ks * 16 + kHi)) : 0u);
        }
    }

    // Under the Block mask every row's last visible key is the call's last position (its oldest
    // is still its own window's); otherwise it is the row's own position.
    const bool block  = args.block != 0;
    // A padding column (positions repeating the last real one) is the last real column again.
    const int p0      = row_.positions[0];
    const int p_call  = gemma4_call_end(row_.positions, tokens) - 1;
    const int t_first = group.token(row_base);
    const int t_last  = min(tokens - 1, group.token(row_base + block_rows - 1));
    // A row slice of padding columns only starts at p_call, not past it.
    int range_lo = window > 0 ? max(0, min(p0 + t_first, p_call) - window + 1) : 0;
    int range_hi = block ? p_call + 1 : min(p0 + t_last, p_call) + 1;
    if constexpr (Partial) {
        int lo = 0, hi = 0, begin = 0, end = 0, active = 0;
        gemma4_union_range(row_.positions, tokens, window, lo, hi);
        gemma4_split_range(lo, hi, args.splits, Bc, split, begin, end, active);
        if (split >= active) { return; }
        range_lo = max(range_lo, begin);
        range_hi = min(range_hi, end);
    }
    // Key tiles are folded from range_lo in steps of Bc, and folding a tile in which a row sees no
    // key is not an exact no-op (its rescale factor exp2(m*s - round(m*s)) may round above 1). A
    // row block of Br / G tokens that is not a power of two, or wider than a tile (group 1 at
    // D <= 128), can cross a tile boundary, so a call ending inside the block would fold fewer
    // tiles than one running through it. Such blocks span their tiles to the block's last token
    // whatever the call; the keys past the call are zero-filled (range_hi) and masked. On a sliding
    // layer range_lo is B - window + 1 for a block starting at B, so a tile boundary falls between
    // the block's first and second token whatever the block's width: a call ending on a block's
    // first token (a decode step, or a prefill call ending one token into the block) folded one
    // tile fewer than a call running through it. Sliding blocks span their tiles too.
    constexpr bool kWideBlock = Geo::Pow2 && (Br >> Geo::GLog2) > Bc;
    constexpr bool kSpanBlock = !Partial && (!Geo::Pow2 || kWideBlock);
    int tile_hi               = range_hi;
    if (!Partial && !block && (kSpanBlock || window > 0)) {
        tile_hi = p0 + group.token(row_base + block_rows - 1) + 1;
    }
    const int n_tiles = tile_hi > range_lo ? div_up(tile_hi - range_lo, Bc) : 0;
    int t0            = 0; // the key tiles this unit folds
    int t1            = n_tiles;
    if constexpr (Chain) {
        t0 = span * chain_tiles;
        t1 = min(n_tiles, t0 + chain_tiles);
        if (t0 >= t1) { continue; } // past this row block's keys
    }
    // A partial first block (a call that starts off the row grid) reaches back past the oldest key
    // the call may see. Those rows are masked for every live row and may be dead in the ring, so
    // they are zero-filled like the rows past range_hi. They are fewer than Br / G, which is at most
    // Bc except for a wide block (group 1 at D <= 128), whose second tile may hold some too.
    const int first_dead =
        Partial || window == 0 ? 0 : max(0, max(0, p0 - window + 1) - range_lo);

    // Each row's visible keys, intersected with the CTA's key range.
    int vis_lo[2];
    int vis_hi[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = row_base + wr * 16 + gid + 8 * r;
        const int p     = min(p0 + group.token(row), p_call);
        const bool live = row >= 0 && row < row_end;
        vis_lo[r]       = live ? max(range_lo, window > 0 ? p - window + 1 : 0) : 1;
        vis_hi[r]       = live ? min(range_hi - 1, block ? p_call : p) : 0;
    }

    const int binding        = detail::gemma4_kv_binding(args.kv, row_.batch_row);
    const std::int64_t plane = detail::gemma4_kv_plane_offset(args.kv, binding) +
                               static_cast<std::int64_t>(kv_head) * capacity * D;
    const std::int32_t* table = detail::gemma4_kv_table(args.kv, binding);

    // Stage key rows [k0, k0 + Bc) of one plane. Slot storage reads slot (position % capacity): a
    // tile never exceeds the capacity, so the ring wraps at most once inside it. A paged binding
    // reads each row through its block table. Rows below dead_rows and from range_hi on are
    // zero-filled. The tile's first slot (k0 % capacity) is taken once per stage: computed per row
    // under the row's condition, the compiler kept one integer division per staged chunk (+35 %
    // instructions in the prefill kernel, Gemma 4 26B prefill -6..-10 %, gate round 4, 905).
    // The 16-byte chunk a K code chunk lands in (KDirect): the ldmatrix that reads eight keys'
    // chunk c reads eight rows D bytes apart, so the chunk is XORed with the row.
    [[maybe_unused]] const auto code_swizzle = [](int row, int col) { return col ^ (row & 7); };
    const auto stage = [&](bool value_plane, int k0, int dead_rows) {
        const int valid_rows = min(Bc, range_hi - k0);
        const int slot0      = k0 % capacity;
        const auto slot_of   = [&](int row) {
            if (table != nullptr) { return detail::gemma4_kv_page_slot(table, k0 + row); }
            const int slot = slot0 + row;
            return slot >= capacity ? slot - capacity : slot;
        };
        if constexpr (!Fp8) {
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(
                gemma4_smem + (value_plane ? Tile::PlaneBytes : 0));
            const __nv_bfloat16* head =
                static_cast<const __nv_bfloat16*>(value_plane ? args.cache_v : args.cache_k) +
                plane;
#pragma unroll
            for (int i = 0; i < StageIters; ++i) {
                const int chunk  = tid + i * Threads;
                const int row    = chunk / RowChunks;
                const int col    = chunk - row * RowChunks;
                __nv_bfloat16* d = dst + row * D + ((col ^ (row & 7)) << 3);
                if (row < valid_rows && row >= dead_rows) {
                    const int slot = slot_of(row);
                    if constexpr (Volta) {
                        // Rounded to f16 once here, for every warp's m8n8k4 that reads it.
                        const uint4 v = __ldcg(reinterpret_cast<const uint4*>(
                            head + static_cast<std::int64_t>(slot) * D + col * 8));
                        store_vec(d, make_uint4(bf16x2_to_f16x2(v.x), bf16x2_to_f16x2(v.y),
                                                bf16x2_to_f16x2(v.z), bf16x2_to_f16x2(v.w)));
                    } else {
                        cp_async<16, Cache::cg>(
                            d, head + static_cast<std::int64_t>(slot) * D + col * 8);
                    }
                } else {
                    store_vec(d, make_int4(0, 0, 0, 0));
                }
            }
        } else {
            // Codes into the upper half of the plane's tile, row r at r * D; scales beside.
            unsigned char* dst =
                gemma4_smem + (value_plane ? Tile::PlaneBytes : 0) + Tile::CodeOffset;
            const auto* head =
                static_cast<const std::uint8_t*>(value_plane ? args.cache_v : args.cache_k) + plane;
            const float* scales = (value_plane ? args.v_scale : args.k_scale) + plane / D;
            float* scale_dst    = row_scales + (value_plane ? Bc : 0);
#pragma unroll
            for (int i = 0; i < CodeIters; ++i) {
                const int chunk  = tid + i * Threads;
                const int row    = chunk / CodeChunks;
                const int col    = chunk - row * CodeChunks;
                unsigned char* d =
                    dst + row * D +
                    ((value_plane ? VDirect : KDirect) ? code_swizzle(row, col) : col) * 16;
                if (row < valid_rows && row >= dead_rows) {
                    const int slot = slot_of(row);
                    cp_async<16, Cache::cg>(d, head + static_cast<std::int64_t>(slot) * D + col * 16);
                } else {
                    store_vec(d, make_int4(0, 0, 0, 0));
                }
            }
            for (int row = tid; row < Bc; row += Threads) {
                if (row < valid_rows && row >= dead_rows) {
                    cp_async<4>(scale_dst + row, scales + slot_of(row));
                } else {
                    scale_dst[row] = 0.0f;
                }
            }
        }
    };

    // FP8: the landed codes of one plane's tile, widened in place to the BF16 tile (scale * code).
    // Every thread first reads its chunks, so no write can overtake another thread's read.
    const auto widen = [&](bool value_plane) {
        if constexpr (Fp8) {
            unsigned char* base      = gemma4_smem + (value_plane ? Tile::PlaneBytes : 0);
            const unsigned char* src = base + Tile::CodeOffset;
            const float* scale_src   = row_scales + (value_plane ? Bc : 0);
            uint4 codes[CodeIters];
#pragma unroll
            for (int i = 0; i < CodeIters; ++i) {
                const int chunk = tid + i * Threads;
                const int row   = chunk / CodeChunks;
                const int col   = chunk - row * CodeChunks;
                codes[i]        = *reinterpret_cast<const uint4*>(src + row * D + col * 16);
            }
            __syncthreads();
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(base);
#pragma unroll
            for (int i = 0; i < CodeIters; ++i) {
                const int chunk   = tid + i * Threads;
                const int row     = chunk / CodeChunks;
                const int col     = chunk - row * CodeChunks;
                const float scale = scale_src[row];
                uint2 a           = detail::gemma4_fp8x4_to_bf16(codes[i].x, scale);
                uint2 b           = detail::gemma4_fp8x4_to_bf16(codes[i].y, scale);
                uint2 c           = detail::gemma4_fp8x4_to_bf16(codes[i].z, scale);
                uint2 e           = detail::gemma4_fp8x4_to_bf16(codes[i].w, scale);
                if constexpr (Volta) { // the BF16 widening, then f16 as a BF16 row is staged
                    const auto f16 = [](uint2 w) {
                        return make_uint2(bf16x2_to_f16x2(w.x), bf16x2_to_f16x2(w.y));
                    };
                    a = f16(a);
                    b = f16(b);
                    c = f16(c);
                    e = f16(e);
                }
                store_vec(dst + row * D + (((2 * col) ^ (row & 7)) << 3),
                          make_uint4(a.x, a.y, b.x, b.y));
                store_vec(dst + row * D + (((2 * col + 1) ^ (row & 7)) << 3),
                          make_uint4(c.x, c.y, e.x, e.y));
            }
            __syncthreads();
        }
    };

    // Per-lane ldmatrix addresses. K is the non-transposed B operand of
    // S = Q K^T (x4 covers two key n-tiles of one 16-wide contraction step); V is the transposed B
    // operand of O = P V (x4 covers two output n-tiles of one 16-key step). Both tiles use the
    // row-XOR 16-byte swizzle.
    const int swz            = lane & 7;
    const unsigned smem_base = smem_addr(gemma4_smem);
    const unsigned k_lane    = smem_base +
                            static_cast<unsigned>((((lane >> 4) & 1) * 8 + swz) * RowBytes) +
                            static_cast<unsigned>(wc * Geo::Slice * 2);
    const int k_hi        = (lane >> 3) & 1;
    const unsigned v_lane = smem_base + static_cast<unsigned>(Tile::PlaneBytes) +
                            static_cast<unsigned>((((lane >> 3) & 1) * 8 + swz) * RowBytes) +
                            static_cast<unsigned>(wc * Geo::Slice * 2);
    const int v_hi         = (lane >> 4) & 1;
    const float scale_log2 = args.scale_log2;

    float acc[PVN][4];
#pragma unroll
    for (int n = 0; n < PVN; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f; }
    float m_run[2] = {NegInf, NegInf};
    float l_run[2] = {0.0f, 0.0f};
    [[maybe_unused]] float4* const state =
        Chain ? args.chain_state + static_cast<std::int64_t>(chain_block) * (PVN + 1) * Threads
              : nullptr;
    if constexpr (Chain) {
        if (span > 0) {
            if (tid == 0) {
                while (gemma4_load_acquire(args.chain_flags + chain_block) < span) {
                    __nanosleep(256);
                }
            }
            __syncthreads();
            // L2 reads: this SM may hold an older span's state of the block in L1.
#pragma unroll
            for (int n = 0; n < PVN; ++n) {
                const float4 a = __ldcg(state + n * Threads + tid);
                acc[n][0]      = a.x;
                acc[n][1]      = a.y;
                acc[n][2]      = a.z;
                acc[n][3]      = a.w;
            }
            const float4 ml = __ldcg(state + PVN * Threads + tid);
            m_run[0]        = ml.x;
            m_run[1]        = ml.y;
            l_run[0]        = ml.z;
            l_run[1]        = ml.w;
        }
    }

    // S = Q K^T over this warp's slice, one 128-column part at a time; with more than one part
    // per head each part's tile is published to `red` (part wr * Parts + wc * Sub + j).
    constexpr int Parts    = Geo::Parts;
    constexpr int PartKS   = KS / Geo::Sub; // contraction steps of one part
    const auto score_tile = [&](float (&s)[QKN][4]) {
#if NINFERNO_HMMA884
        if constexpr (Volta) {
            // Each quad-pair: rows 8 * (vq & 1) of the warp's 16 by keys 8 * (vq >> 1) of a 16-key
            // pair; B (K^T, col-major) is the lane's key's four dims, one 8-byte read of the f16
            // tile. The 8 x 8 blocks scatter into the exchange as the m16n8 lanes hold them.
            const __half* kt = reinterpret_cast<const __half*>(gemma4_smem);
            float* redf      = reinterpret_cast<float*>(red);
#pragma unroll
            for (int j = 0; j < Geo::Sub; ++j) {
                const int part = wr * Parts + wc * Geo::Sub + j;
#pragma unroll
                for (int nt2 = 0; nt2 < QKN / 2; ++nt2) {
                    float c[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    const int key   = nt2 * 16 + 8 * (vq >> 1) + vj;
                    const __half* k = kt + key * D;
#pragma unroll
                    for (int ks = j * PartKS; ks < (j + 1) * PartKS; ++ks) {
#pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            const int dim = wc * Geo::Slice + ks * 16 + kk * 4;
                            const uint2 b = *reinterpret_cast<const uint2*>(
                                k + ((((dim >> 3) ^ (key & 7))) << 3) + (dim & 7));
                            mma_m8n8k4_f16(c, qv[ks][kk][0], qv[ks][kk][1], b.x, b.y);
                        }
                    }
                    const int nt = 2 * nt2 + (vq >> 1);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int r  = 8 * (vq & 1) + (lane & 1) + (i & 2) + 4 * (lane >> 4);
                        const int cc = (i & 4) + (lane & 2) + (i & 1);
                        redf[((part * QKN + nt) * 32 + 4 * (r & 7) + (cc >> 1)) * 4 + (cc & 1) +
                             2 * (r >> 3)] = c[i];
                    }
                }
            }
            return;
        }
#endif
        const unsigned base = k_lane;
        // FP8: the K codes stay codes in shared memory. One ldmatrix.x4 reads, as 8x8 b16
        // matrices, keys (n-tiles 2*nt2, 2*nt2+1) x (k-steps ks, ks+1); a lane's word of each is
        // four codes of key gid, widened in registers exactly as the shared-memory widening did
        // (scale * code, rounded to BF16).
        [[maybe_unused]] float kscale[QKN];
        [[maybe_unused]] unsigned code_lane = 0;
        if constexpr (KDirect) {
#pragma unroll
            for (int nt = 0; nt < QKN; ++nt) { kscale[nt] = row_scales[nt * 8 + gid]; }
            code_lane = smem_base + static_cast<unsigned>(Tile::CodeOffset);
        }
#pragma unroll
        for (int j = 0; j < Geo::Sub; ++j) {
#pragma unroll
            for (int nt = 0; nt < QKN; ++nt) {
                s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            }
            if constexpr (KDirect) {
                static_assert(PartKS % 2 == 0 && QKN % 2 == 0);
                const int mrow = (lane & 7) + ((lane >> 3) & 1) * 8;
                const int mks  = (lane >> 4) & 1;
#pragma unroll
                for (int ks = j * PartKS; ks < (j + 1) * PartKS; ks += 2) {
                    const int col = wc * (Geo::Slice / 16) + ks + mks;
#pragma unroll
                    for (int nt2 = 0; nt2 < QKN / 2; ++nt2) {
                        const int key = nt2 * 16 + mrow;
                        unsigned w0, w1, w2, w3;
                        ldmatrix_x4(w0, w1, w2, w3,
                                    code_lane + static_cast<unsigned>(key * D +
                                                                      code_swizzle(key, col) * 16));
                        const uint2 a = detail::gemma4_fp8x4_to_bf16(w0, kscale[2 * nt2]);
                        const uint2 b = detail::gemma4_fp8x4_to_bf16(w1, kscale[2 * nt2 + 1]);
                        const uint2 c = detail::gemma4_fp8x4_to_bf16(w2, kscale[2 * nt2]);
                        const uint2 e = detail::gemma4_fp8x4_to_bf16(w3, kscale[2 * nt2 + 1]);
                        mma_bf16_prepared(s[2 * nt2][0], s[2 * nt2][1], s[2 * nt2][2],
                                          s[2 * nt2][3], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3],
                                          a.x, a.y);
                        mma_bf16_prepared(s[2 * nt2 + 1][0], s[2 * nt2 + 1][1], s[2 * nt2 + 1][2],
                                          s[2 * nt2 + 1][3], qf[ks][0], qf[ks][1], qf[ks][2],
                                          qf[ks][3], b.x, b.y);
                        mma_bf16_prepared(s[2 * nt2][0], s[2 * nt2][1], s[2 * nt2][2],
                                          s[2 * nt2][3], qf[ks + 1][0], qf[ks + 1][1],
                                          qf[ks + 1][2], qf[ks + 1][3], c.x, c.y);
                        mma_bf16_prepared(s[2 * nt2 + 1][0], s[2 * nt2 + 1][1], s[2 * nt2 + 1][2],
                                          s[2 * nt2 + 1][3], qf[ks + 1][0], qf[ks + 1][1],
                                          qf[ks + 1][2], qf[ks + 1][3], e.x, e.y);
                    }
                }
            }
#pragma unroll
            for (int ks = j * PartKS; ks < (j + 1) * PartKS && !KDirect; ++ks) {
                const unsigned col = static_cast<unsigned>(((2 * ks + k_hi) ^ swz) << 4);
#pragma unroll
                for (int nt2 = 0; nt2 < QKN / 2; ++nt2) {
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4(b0, b1, b2, b3,
                                base + static_cast<unsigned>(nt2 * 16 * RowBytes) + col);
                    mma_bf16_prepared(s[2 * nt2][0], s[2 * nt2][1], s[2 * nt2][2], s[2 * nt2][3],
                                      qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
                    mma_bf16_prepared(s[2 * nt2 + 1][0], s[2 * nt2 + 1][1], s[2 * nt2 + 1][2],
                                      s[2 * nt2 + 1][3], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3],
                                      b2, b3);
                }
            }
            if constexpr (Parts > 1) {
                const int part = wr * Parts + wc * Geo::Sub + j;
#pragma unroll
                for (int nt = 0; nt < QKN; ++nt) {
                    red[(part * QKN + nt) * 32 + lane] =
                        make_float4(s[nt][0], s[nt][1], s[nt][2], s[nt][3]);
                }
            }
        }
    };

    // Sum the published slice partials, mask, advance the online softmax and return BF16 P.
    const auto softmax_tile = [&](int k0, float (&s)[QKN][4], unsigned (&pa)[PVK][4]) {
        if constexpr (Parts > 1) {
            // One fixed summation order, so every warp of the row group derives bit-identical
            // scores, maxima and normalizers.
#pragma unroll
            for (int nt = 0; nt < QKN; ++nt) {
                float4 total = red[((wr * Parts) * QKN + nt) * 32 + lane];
#pragma unroll
                for (int c = 1; c < Parts; ++c) {
                    const float4 part = red[((wr * Parts + c) * QKN + nt) * 32 + lane];
                    total.x += part.x;
                    total.y += part.y;
                    total.z += part.z;
                    total.w += part.w;
                }
                s[nt][0] = total.x;
                s[nt][1] = total.y;
                s[nt][2] = total.z;
                s[nt][3] = total.w;
            }
        }
        float bm0 = NegInf;
        float bm1 = NegInf;
#pragma unroll
        for (int nt = 0; nt < QKN; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int key = k0 + nt * 8 + 2 * tig + j;
                s[nt][j]      = (key >= vis_lo[0] && key <= vis_hi[0]) ? s[nt][j] : NegInf;
                s[nt][2 + j]  = (key >= vis_lo[1] && key <= vis_hi[1]) ? s[nt][2 + j] : NegInf;
            }
            bm0 = fmaxf(bm0, fmaxf(s[nt][0], s[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(s[nt][2], s[nt][3]));
        }
        bm0 = warp_max<4>(bm0);
        bm1 = warp_max<4>(bm1);

        // A row with no visible key yet keeps m = -inf; its exponent base is pinned to 0 so that
        // exp2(-inf - base) is an exact zero instead of NaN.
        const float nm0    = fmaxf(m_run[0], bm0);
        const float nm1    = fmaxf(m_run[1], bm1);
        const float base0  = nm0 == NegInf ? 0.0f : nm0 * scale_log2;
        const float base1  = nm1 == NegInf ? 0.0f : nm1 * scale_log2;
        const float alpha0 = exp2_approx(__fmaf_rn(m_run[0], scale_log2, -base0));
        const float alpha1 = exp2_approx(__fmaf_rn(m_run[1], scale_log2, -base1));
        m_run[0]           = nm0;
        m_run[1]           = nm1;

        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKN; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(s[nt][0], scale_log2, -base0));
            const float p01 = exp2_approx(__fmaf_rn(s[nt][1], scale_log2, -base0));
            const float p10 = exp2_approx(__fmaf_rn(s[nt][2], scale_log2, -base1));
            const float p11 = exp2_approx(__fmaf_rn(s[nt][3], scale_log2, -base1));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            if constexpr (Volta) {
                // P rows gid, gid + 8 at keys nt * 8 + 2 tig, + 1, as value_tile's A reads them.
                *reinterpret_cast<unsigned*>(vp + gid * Bc + nt * 8 + 2 * tig)       = pack_f16x2(p00, p01);
                *reinterpret_cast<unsigned*>(vp + (gid + 8) * Bc + nt * 8 + 2 * tig) = pack_f16x2(p10, p11);
            } else {
                pa[nt >> 1][(nt & 1) * 2]     = mma_a_bf16(pack_bf16x2(p00, p01));
                pa[nt >> 1][(nt & 1) * 2 + 1] = mma_a_bf16(pack_bf16x2(p10, p11));
            }
        }
        l_run[0] = __fmaf_rn(l_run[0], alpha0, bl0);
        l_run[1] = __fmaf_rn(l_run[1], alpha1, bl1);
        if constexpr (Volta) { // value_tile rescales its own layout's rows
            if (tig == 0) {
                valpha[gid]     = alpha0;
                valpha[gid + 8] = alpha1;
            }
            __syncwarp();
            return;
        }
#pragma unroll
        for (int n = 0; n < PVN; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
    };

    // O += P V over this warp's slice.
    const auto value_tile = [&](const unsigned (&pa)[PVK][4]) {
#if NINFERNO_HMMA884
        if constexpr (Volta) {
            // acc[2 n2 + h][e] holds the 8 x 8 block's c[4 h + e] of dims pair n2: rows
            // 8 * (vq & 1) + (lane & 1) + (c & 2) + 4 * (lane >> 4), dims 16 n2 + 8 * (vq >> 1) +
            // (c & 4) + (lane & 2) + (c & 1). A is the lane's P row over four keys; B (V,
            // row-major) is key lane % 4 of four, four dims.
            const __half* vt = reinterpret_cast<const __half*>(gemma4_smem + Tile::PlaneBytes);
            const int r0     = 8 * (vq & 1) + (lane & 1) + 4 * (lane >> 4);
            const float al0  = valpha[r0];
            const float al1  = valpha[r0 + 2];
            unsigned pv[PVK][4][2];
#pragma unroll
            for (int pk = 0; pk < PVK; ++pk) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint2 a = *reinterpret_cast<const uint2*>(
                        vp + (8 * (vq & 1) + vj) * Bc + pk * 16 + kk * 4);
                    pv[pk][kk][0] = a.x;
                    pv[pk][kk][1] = a.y;
                }
            }
#pragma unroll
            for (int n2 = 0; n2 < PVN / 2; ++n2) {
                float c[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    c[i] = acc[2 * n2 + (i >> 2)][i & 3] * ((i & 2) ? al1 : al0);
                }
                const int dim = wc * Geo::Slice + n2 * 16 + 8 * (vq >> 1) + 4 * (lane >> 4);
#pragma unroll
                for (int pk = 0; pk < PVK; ++pk) {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const int key = pk * 16 + kk * 4 + (lane & 3);
                        const uint2 b = *reinterpret_cast<const uint2*>(
                            vt + key * D + ((((dim >> 3) ^ (key & 7))) << 3) + (dim & 7));
                        mma_m8n8k4_f16_row_b(c, pv[pk][kk][0], pv[pk][kk][1], b.x, b.y);
                    }
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[2 * n2 + (i >> 2)][i & 3] = c[i]; }
            }
            return;
        }
#endif
        const unsigned base = v_lane;
        if constexpr (VDirect) {
            // FP8: V stays codes in shared memory. One ldmatrix.x4.trans reads, as 8x8 b16
            // matrices of (key, dim pair), keys (lo 8, hi 8) of the step x dim chunks (c, c+1); a
            // lane's word of each is keys 2t, 2t+1 of dim pair gid, so a byte permute gives the
            // even dim's and the odd dim's codes of keys 2t, 2t+1, 2t+8, 2t+9, widened with each
            // key's scale exactly as the shared-memory widening did.
            static_assert(PVN % 4 == 0);
            const unsigned code_lane =
                smem_base + static_cast<unsigned>(Tile::PlaneBytes + Tile::CodeOffset);
            const float* vscale = row_scales + Bc;
            const int mkey      = (lane & 7) + ((lane >> 3) & 1) * 8;
            const int mchunk    = (lane >> 4) & 1;
#pragma unroll
            for (int pk = 0; pk < PVK; ++pk) {
                const float2 s_lo = *reinterpret_cast<const float2*>(vscale + pk * 16 + 2 * tig);
                const float2 s_hi =
                    *reinterpret_cast<const float2*>(vscale + pk * 16 + 8 + 2 * tig);
                const int key = pk * 16 + mkey;
#pragma unroll
                for (int n4 = 0; n4 < PVN / 4; ++n4) {
                    const int col = wc * (Geo::Slice / 16) + 2 * n4 + mchunk;
                    unsigned lo0, hi0, lo1, hi1;
                    ldmatrix_x4_t(lo0, hi0, lo1, hi1,
                                  code_lane +
                                      static_cast<unsigned>(key * D + code_swizzle(key, col) * 16));
                    const uint2 e0 = detail::gemma4_fp8x4_to_bf16_keys(
                        __byte_perm(lo0, hi0, 0x6420), s_lo, s_hi);
                    const uint2 o0 = detail::gemma4_fp8x4_to_bf16_keys(
                        __byte_perm(lo0, hi0, 0x7531), s_lo, s_hi);
                    const uint2 e1 = detail::gemma4_fp8x4_to_bf16_keys(
                        __byte_perm(lo1, hi1, 0x6420), s_lo, s_hi);
                    const uint2 o1 = detail::gemma4_fp8x4_to_bf16_keys(
                        __byte_perm(lo1, hi1, 0x7531), s_lo, s_hi);
                    float (&a0)[4] = acc[4 * n4];
                    float (&a1)[4] = acc[4 * n4 + 1];
                    float (&a2)[4] = acc[4 * n4 + 2];
                    float (&a3)[4] = acc[4 * n4 + 3];
                    mma_bf16_prepared(a0[0], a0[1], a0[2], a0[3], pa[pk][0], pa[pk][1], pa[pk][2],
                                      pa[pk][3], e0.x, e0.y);
                    mma_bf16_prepared(a1[0], a1[1], a1[2], a1[3], pa[pk][0], pa[pk][1], pa[pk][2],
                                      pa[pk][3], o0.x, o0.y);
                    mma_bf16_prepared(a2[0], a2[1], a2[2], a2[3], pa[pk][0], pa[pk][1], pa[pk][2],
                                      pa[pk][3], e1.x, e1.y);
                    mma_bf16_prepared(a3[0], a3[1], a3[2], a3[3], pa[pk][0], pa[pk][1], pa[pk][2],
                                      pa[pk][3], o1.x, o1.y);
                }
            }
            return;
        }
#pragma unroll
        for (int pk = 0; pk < PVK; ++pk) {
#pragma unroll
            for (int n2 = 0; n2 < PVN / 2; ++n2) {
                unsigned b0, b1, b2, b3;
                ldmatrix_x4_t(b0, b1, b2, b3,
                              base + static_cast<unsigned>(pk * 16 * RowBytes) +
                                  static_cast<unsigned>(((2 * n2 + v_hi) ^ swz) << 4));
                mma_bf16_prepared(acc[2 * n2][0], acc[2 * n2][1], acc[2 * n2][2], acc[2 * n2][3],
                                  pa[pk][0], pa[pk][1], pa[pk][2], pa[pk][3], b0, b1);
                mma_bf16_prepared(acc[2 * n2 + 1][0], acc[2 * n2 + 1][1], acc[2 * n2 + 1][2],
                                  acc[2 * n2 + 1][3], pa[pk][0], pa[pk][1], pa[pk][2], pa[pk][3],
                                  b2, b3);
            }
        }
    };

    if (t1 > t0) {
        stage(false, range_lo + t0 * Bc, first_dead);
        cp_commit();
        stage(true, range_lo + t0 * Bc, first_dead);
        cp_commit();
    }
    for (int kb = t0; kb < t1; ++kb) {
        const int k0 = range_lo + kb * Bc;
        cp_wait<1>(); // K(kb) landed; V(kb) may still be in flight
        __syncthreads();
        if constexpr (!KDirect) { widen(false); }
        float s[QKN][4];
        score_tile(s);
        __syncthreads(); // partial scores published; every warp is done reading K(kb)
        if (kb + 1 < t1) {
            stage(false, k0 + Bc, kWideBlock ? max(0, first_dead - (k0 + Bc - range_lo)) : 0);
        }
        cp_commit();
        unsigned pa[PVK][4];
        softmax_tile(k0, s, pa);
        cp_wait<1>(); // V(kb) landed; K(kb+1) may still be in flight
        __syncthreads();
        if constexpr (!VDirect) { widen(true); }
        value_tile(pa);
        __syncthreads(); // every warp is done reading V(kb)
        if (kb + 1 < t1) {
            stage(true, k0 + Bc, kWideBlock ? max(0, first_dead - (k0 + Bc - range_lo)) : 0);
        }
        cp_commit();
    }

    if constexpr (Chain) {
        if (t1 < n_tiles) {
#pragma unroll
            for (int n = 0; n < PVN; ++n) {
                __stcg(state + n * Threads + tid,
                       make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]));
            }
            __stcg(state + PVN * Threads + tid,
                   make_float4(m_run[0], m_run[1], l_run[0], l_run[1]));
            __threadfence();
            __syncthreads();
            if (tid == 0) { gemma4_store_release(args.chain_flags + chain_block, span + 1); }
            continue;
        }
    }

    if constexpr (Volta) {
        // The accumulators to the m16n8 lane layout the stores below read, through this warp's
        // 16 x Slice of the freed K/V planes.
        __syncthreads();
        float* o = reinterpret_cast<float*>(gemma4_smem) + warp * 16 * Geo::Slice;
#pragma unroll
        for (int n2 = 0; n2 < PVN / 2; ++n2) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r   = 8 * (vq & 1) + (lane & 1) + (i & 2) + 4 * (lane >> 4);
                const int col = n2 * 16 + 8 * (vq >> 1) + (i & 4) + (lane & 2) + (i & 1);
                o[r * Geo::Slice + col] = acc[2 * n2 + (i >> 2)][i & 3];
            }
        }
        __syncwarp();
#pragma unroll
        for (int n = 0; n < PVN; ++n) {
            const float2 top = *reinterpret_cast<const float2*>(o + gid * Geo::Slice + n * 8 + 2 * tig);
            const float2 bot =
                *reinterpret_cast<const float2*>(o + (gid + 8) * Geo::Slice + n * 8 + 2 * tig);
            acc[n][0] = top.x;
            acc[n][1] = top.y;
            acc[n][2] = bot.x;
            acc[n][3] = bot.y;
        }
    }

    l_run[0] = warp_sum<4>(l_run[0]);
    l_run[1] = warp_sum<4>(l_run[1]);

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = row_base + wr * 16 + gid + 8 * r;
        if (row < 0 || row >= row_end) { continue; }
        if constexpr (!Partial) {
            const int token    = group.token(row);
            const int head     = kv_head * G + group.member(row);
            float total = l_run[r];
            if (args.sinks != nullptr) {
                // The sink's term against the row's exponent base (0 for a row that saw no key).
                const float base = m_run[r] == NegInf ? 0.0f : m_run[r] * scale_log2;
                total += exp2f(args.sinks[head] * 1.4426950408889634074f - base);
            }
            const float inv    = 1.0f / total;
            __nv_bfloat16* dst = row_.out +
                                 (static_cast<std::int64_t>(token) * args.q_heads + head) * D +
                                 wc * Geo::Slice + (VDirect ? 4 : 2) * tig;
            if constexpr (VDirect) {
#pragma unroll
                for (int c = 0; c < PVN / 2; ++c) {
                    store_vec(dst + c * 16,
                              make_uint2(
                                  pack_bf16x2(acc[2 * c][2 * r] * inv, acc[2 * c + 1][2 * r] * inv),
                                  pack_bf16x2(acc[2 * c][2 * r + 1] * inv,
                                              acc[2 * c + 1][2 * r + 1] * inv)));
                }
            } else {
#pragma unroll
                for (int n = 0; n < PVN; ++n) {
                    store_vec(dst + n * 8,
                              pack_bf16x2(acc[n][2 * r] * inv, acc[n][2 * r + 1] * inv));
                }
            }
        } else {
            const std::int64_t index =
                static_cast<std::int64_t>(split * args.kv_heads + kv_head) * rows_total + row;
            float* dst = row_.partial_acc + index * D + wc * Geo::Slice + (VDirect ? 4 : 2) * tig;
            if constexpr (VDirect) {
#pragma unroll
                for (int c = 0; c < PVN / 2; ++c) {
                    store_vec(dst + c * 16,
                              make_float4(acc[2 * c][2 * r], acc[2 * c + 1][2 * r],
                                          acc[2 * c][2 * r + 1], acc[2 * c + 1][2 * r + 1]));
                }
            } else {
#pragma unroll
                for (int n = 0; n < PVN; ++n) {
                    store_vec(dst + n * 8, make_float2(acc[n][2 * r], acc[n][2 * r + 1]));
                }
            }
            if (wc == 0 && tig == 0) {
                row_.partial_m[index] = m_run[r] == NegInf ? NegInf : m_run[r] * scale_log2;
                row_.partial_l[index] = l_run[r];
            }
        }
    }
    if constexpr (!Chain) { break; }
    } // per unit
}

inline constexpr int kGemma4CombineWarps = 8;
// Splits one combine warp loads in a single round trip (6 registers each). Warps x Batch = 192
// covers the split counts the decode plan makes for the Gemma 4 layers on RTX 50 and RTX PRO 6000
// cards (at most one D=512 CTA per SM, so at most the SM count); more take extra round trips.
inline constexpr int kGemma4CombineBatch = 24;

// Merge the live splits of one (token, query head) row over a 128-wide feature chunk. Grid is
// (D / 128, Hq, T * B): the head, the token and the batch row come from the grid, and the head
// counts enter only as the stride between splits, so no head count is divided at run time.
//
// The kernel is latency-bound (a few dozen CTAs of a few KiB each), so its time is the number of
// dependent memory round trips. Warp w owns splits w, w + 8, ...; its first Batch of them are
// loaded in one round trip, before the running maximum is known, and the call's maximum is taken
// from those same values instead of a separate pass over partial_m. The partials are then folded
// in ascending split order per warp and the warps in ascending order: the same arithmetic in the
// same order as a split-at-a-time loop, so the output is independent of the batch size.
template <class Geo, int Bc>
__launch_bounds__(kGemma4CombineWarps * 32) __global__
    void gemma4_attention_combine_kernel(const Gemma4AttentionArgs args) {
    // A programmatic dependent (a decode projection of the output) may launch now: it waits for
    // this kernel before it reads, and streams its weights meanwhile.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    // Launched as the attention grid's dependent (where the device has PDL): the partials are
    // its output.
    pdl::wait_for_dependencies();
    constexpr int D     = Geo::D;
    constexpr int Warps = kGemma4CombineWarps;
    constexpr int Batch = kGemma4CombineBatch;
    const float NegInf  = -CUDART_INF_F;

    __shared__ float max_s[Warps];
    __shared__ float4 acc_s[Warps][32];
    __shared__ float l_s[Warps];

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int head       = static_cast<int>(blockIdx.y);
    const int tokens     = args.tokens;
    const int batch_row  = static_cast<int>(blockIdx.z) / tokens;
    const int token      = static_cast<int>(blockIdx.z) - batch_row * tokens;
    const Gemma4BatchRow row_ = gemma4_batch_row(args, batch_row, Geo::D);
    const Gemma4Group<Geo> group{args.q_heads / args.kv_heads};
    const int G          = group.size();
    const int kv_head    = group.token(head);
    const int rows_total = tokens * G;
    const int row        = token * G + group.member(head);
    const int feature    = static_cast<int>(blockIdx.x) * 128 + lane * 4;
    const bool in_head   = feature < D; // a 64-wide head leaves half the lanes idle

    // Partial row of split s: (s * Hkv + kv_head) * rows_total + row.
    const std::int64_t split_stride = static_cast<std::int64_t>(args.kv_heads) * rows_total;
    const std::int64_t first_row    = static_cast<std::int64_t>(kv_head) * rows_total + row;

    int lo = 0, hi = 0, begin = 0, end = 0, active = 0;
    gemma4_union_range(row_.positions, tokens, args.window, lo, hi);
    gemma4_split_range(lo, hi, args.splits, Bc, 0, begin, end, active);

    // This warp's first Batch splits: every load in flight together.
    float m_b[Batch];
    float l_b[Batch];
    float4 x_b[Batch];
#pragma unroll
    for (int i = 0; i < Batch; ++i) {
        const int split = warp + i * Warps;
        if (split >= active) { break; }
        const std::int64_t index = first_row + split * split_stride;
        m_b[i]                   = row_.partial_m[index];
        l_b[i]                   = row_.partial_l[index];
        x_b[i] = in_head ? *reinterpret_cast<const float4*>(row_.partial_acc + index * D + feature)
                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float m_max = NegInf;
#pragma unroll
    for (int i = 0; i < Batch; ++i) {
        if (warp + i * Warps >= active) { break; }
        m_max = fmaxf(m_max, m_b[i]);
    }
    // Splits past the batch, if any: the block's threads stride over them.
    if (active > Warps * Batch) {
        for (int split = Warps * Batch + tid; split < active; split += Warps * 32) {
            m_max = fmaxf(m_max, row_.partial_m[first_row + split * split_stride]);
        }
        m_max = warp_max(m_max);
    }
    if (lane == 0) { max_s[warp] = m_max; }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < Warps; ++w) { m_max = fmaxf(m_max, max_s[w]); }

    float4 acc  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float l_sum = 0.0f;
    const auto fold = [&](float m, float l, const float4& x) {
        const float w = m == NegInf ? 0.0f : exp2_approx(m - m_max);
        l_sum         = __fmaf_rn(w, l, l_sum);
        acc.x         = __fmaf_rn(w, x.x, acc.x);
        acc.y         = __fmaf_rn(w, x.y, acc.y);
        acc.z         = __fmaf_rn(w, x.z, acc.z);
        acc.w         = __fmaf_rn(w, x.w, acc.w);
    };
#pragma unroll
    for (int i = 0; i < Batch; ++i) {
        if (warp + i * Warps >= active) { break; }
        fold(m_b[i], l_b[i], x_b[i]);
    }
#pragma unroll 4
    for (int split = warp + Warps * Batch; split < active; split += Warps) {
        const std::int64_t index = first_row + split * split_stride;
        fold(row_.partial_m[index], row_.partial_l[index],
             in_head ? *reinterpret_cast<const float4*>(row_.partial_acc + index * D + feature)
                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
    acc_s[warp][lane] = acc;
    if (lane == 0) { l_s[warp] = l_sum; }
    __syncthreads();
    if (warp != 0) { return; }
    float total_l = 0.0f;
#pragma unroll
    for (int w = 0; w < Warps; ++w) {
        const float4 x = acc_s[w][lane];
        if (w != 0) {
            acc.x += x.x;
            acc.y += x.y;
            acc.z += x.z;
            acc.w += x.w;
        }
        total_l += l_s[w];
    }
    if (args.sinks != nullptr) {
        total_l += exp2_approx(args.sinks[head] * 1.4426950408889634074f - m_max);
    }
    const float inv = 1.0f / total_l;
    if (!in_head) { return; }
    __nv_bfloat16* dst =
        row_.out + (static_cast<std::int64_t>(token) * args.q_heads + head) * D + feature;
    store_vec(dst, make_uint2(pack_bf16x2(acc.x * inv, acc.y * inv),
                              pack_bf16x2(acc.z * inv, acc.w * inv)));
}

// Bit-exact row copy into a layer's storage: one thread per 16-byte chunk of K and of V. Column c
// of the call belongs to batch row c / row_tokens; a column past the row's stored count writes
// nothing.
template <int D>
__launch_bounds__(256) __global__
    void gemma4_kv_append_kernel(const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v,
                                 const std::int32_t* __restrict__ positions,
                                 __nv_bfloat16* __restrict__ cache_k,
                                 __nv_bfloat16* __restrict__ cache_v, std::int32_t tokens,
                                 std::int32_t kv_heads, std::int32_t row_tokens,
                                 detail::Gemma4KvAddress kv) {
    constexpr int RowChunks = D / 8;
    const std::int64_t chunk =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
    if (chunk >= static_cast<std::int64_t>(tokens) * kv_heads * RowChunks) { return; }
    const std::int64_t row = chunk / RowChunks; // token * Hkv + head
    const int col          = static_cast<int>(chunk - row * RowChunks);
    const int token        = static_cast<int>(row / kv_heads);
    const int head         = static_cast<int>(row - static_cast<std::int64_t>(token) * kv_heads);
    const int batch_row    = token / row_tokens;
    if (!detail::gemma4_kv_stores(kv, batch_row, token - batch_row * row_tokens)) { return; }
    const int binding      = detail::gemma4_kv_binding(kv, batch_row);
    const int slot         = detail::gemma4_kv_slot(kv, binding, positions[token]);
    const std::int64_t dst = detail::gemma4_kv_plane_offset(kv, binding) +
                             (static_cast<std::int64_t>(head) * kv.capacity + slot) * D + col * 8;
    store_vec(cache_k + dst, load_vec<int4>(k + chunk * 8));
    store_vec(cache_v + dst, load_vec<int4>(v + chunk * 8));
}

// FP8 planes: one warp per row (one column of one KV head), which quantizes it with the row codec
// of fp8_row.cuh. Column c of the call belongs to batch row c / row_tokens; a column past the
// row's stored count writes nothing.
template <int D>
__launch_bounds__(256) __global__
    void gemma4_kv_append_fp8_kernel(const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v,
                                     const std::int32_t* __restrict__ positions,
                                     std::uint8_t* __restrict__ cache_k,
                                     std::uint8_t* __restrict__ cache_v,
                                     float* __restrict__ k_scale, float* __restrict__ v_scale,
                                     std::int32_t tokens, std::int32_t kv_heads,
                                     std::int32_t row_tokens, detail::Gemma4KvAddress kv) {
    constexpr int E = D / 32;
    const std::int64_t row =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5; // token*Hkv+head
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (row >= static_cast<std::int64_t>(tokens) * kv_heads) { return; } // warp-uniform
    const int token        = static_cast<int>(row / kv_heads);
    const int head         = static_cast<int>(row - static_cast<std::int64_t>(token) * kv_heads);
    const int batch_row    = token / row_tokens;
    if (!detail::gemma4_kv_stores(kv, batch_row, token - batch_row * row_tokens)) { return; }
    const int binding      = detail::gemma4_kv_binding(kv, batch_row);
    const int slot         = detail::gemma4_kv_slot(kv, binding, positions[token]);
    const std::int64_t dst = detail::gemma4_kv_plane_offset(kv, binding) / D +
                             static_cast<std::int64_t>(head) * kv.capacity + slot; // row index
#pragma unroll
    for (int plane = 0; plane < 2; ++plane) {
        const __nv_bfloat16* src = (plane == 0 ? k : v) + row * D + lane * E;
        float x[E];
#pragma unroll
        for (int j = 0; j < E; j += 2) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(src + j));
            x[j]           = f.x;
            x[j + 1]       = f.y;
        }
        detail::gemma4_fp8_store_row<E>((plane == 0 ? cache_k : cache_v) + dst * D,
                                        (plane == 0 ? k_scale : v_scale) + dst, x, lane);
    }
}

} // namespace ninferno::ops
