#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops {

// The SIMT linear tiles' partial-tile columns (q*_a16_simt.cuh): on Volta computed on the last live
// column without a per-column branch, elsewhere skipped under the branch.
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
inline constexpr bool kSimtIdleColumnsOnLastLive = true;
#else
inline constexpr bool kSimtIdleColumnsOnLastLive = false;
#endif

enum class Cache { ca, cg };

// NINFERNO-SPECIAL(arch:sm75): Turing has no cp.async. Its build copies synchronously through
// registers: the copy is complete when it returns, so every commit and wait is empty and a
// pipeline's __syncthreads still orders the copy before the reads. The staging depth a kernel
// chose for latency hiding buys nothing there; correctness is unchanged.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#    define NINFERNO_HAS_CP_ASYNC 0
#else
#    define NINFERNO_HAS_CP_ASYNC 1
#endif

template <int Bytes>
__device__ __forceinline__ void copy_sync(void* smem_dst, const void* gmem_src) {
    if constexpr (Bytes == 16) {
        *static_cast<uint4*>(smem_dst) = *static_cast<const uint4*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *static_cast<uint2*>(smem_dst) = *static_cast<const uint2*>(gmem_src);
    } else {
        *static_cast<unsigned*>(smem_dst) = *static_cast<const unsigned*>(gmem_src);
    }
}

// The first src_bytes of the chunk from global memory, the rest zero (cp.async's zfill).
template <int Bytes>
__device__ __forceinline__ void copy_sync_zfill(void* smem_dst, const void* gmem_src,
                                                int src_bytes) {
    if (src_bytes >= Bytes) {
        copy_sync<Bytes>(smem_dst, gmem_src);
        return;
    }
    auto* dst       = static_cast<unsigned char*>(smem_dst);
    const auto* src = static_cast<const unsigned char*>(gmem_src);
#pragma unroll
    for (int i = 0; i < Bytes; ++i) dst[i] = i < src_bytes ? src[i] : 0;
}

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

// The 128-byte XOR swizzle of a shared-memory tile of 64 BF16 columns: the eight 16-byte chunks
// of a row are permuted by the row's low three bits, so the rows of an 8x8 fragment fall in
// different banks.
__device__ __forceinline__ int swizzle_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

// Values consecutive BF16 values (a multiple of 8, from a 16-byte aligned address) as BF16x2
// words: one vector load per eight values. ReadOnly loads through the read-only data cache.
template <int Values>
struct Bf16Words {
    static_assert(Values % 8 == 0);
    std::uint32_t words[Values / 2];
};

template <int Values, bool ReadOnly = false, class T>
__device__ __forceinline__ Bf16Words<Values> load_bf16_words(const T* src) {
    static_assert(sizeof(T) == 2);
    Bf16Words<Values> result;
#pragma unroll
    for (int chunk = 0; chunk < Values / 8; ++chunk) {
        const uint4 packed =
            ReadOnly ? load_ldg<uint4>(src + chunk * 8) : load_vec<uint4>(src + chunk * 8);
        result.words[chunk * 4]     = packed.x;
        result.words[chunk * 4 + 1] = packed.y;
        result.words[chunk * 4 + 2] = packed.z;
        result.words[chunk * 4 + 3] = packed.w;
    }
    return result;
}

// The depth of a load ring that keeps about `budget` bytes per lane in flight, `step` bytes per
// step, over `steps` steps: at least one slot, never more slots than steps.
__host__ __device__ constexpr int ring_depth(int budget, int step, int steps) {
    const int depth = budget / step < 1 ? 1 : budget / step;
    return depth > steps ? steps : depth;
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg)
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
#if !NINFERNO_HAS_CP_ASYNC
    copy_sync<Bytes>(smem_dst, gmem_src);
#else
    if constexpr (Policy == Cache::cg) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg)
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
#if !NINFERNO_HAS_CP_ASYNC
    copy_sync_zfill<Bytes>(smem_dst, gmem_src, src_bytes);
#else
    if constexpr (Policy == Cache::cg) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
#endif
}

__device__ __forceinline__ void cp_commit() {
#if NINFERNO_HAS_CP_ASYNC
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if NINFERNO_HAS_CP_ASYNC
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninferno::ops
