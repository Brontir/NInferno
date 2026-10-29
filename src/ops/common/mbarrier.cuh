#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

namespace ninferno::ops {

// NINFERNO-SPECIAL(arch:sm89): transaction barriers serve the TMA routes, which are sm_90. Below
// it the host runs the cp.async tile instead (launch_*_tma_mma), so these bodies trap unreached.

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
#endif
}

// A consumer's generic-proxy reads of a stage (ld.shared, ldmatrix) ordered before the async-proxy
// writes of its refill (cp.async.bulk.tensor): the mbarrier orders generic-proxy accesses only
// (PTX ISA, memory consistency model: proxies), so every reading thread runs this after its last
// read of a stage and before its warp arrives on the stage's empty barrier. Without it a refill
// may overwrite a stage still being read (Wallawalla47/Infernix 4b5eae7a8c: wrong outputs on the
// RTX 5090 while another shared-memory-heavy kernel shared the SMs, never in an isolated test).
__device__ __forceinline__ void cta_fence_reads_before_async_refill() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#endif
}

} // namespace ninferno::ops
