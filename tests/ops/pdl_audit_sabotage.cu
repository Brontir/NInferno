// Sabotage fixture for tools/pdl_sass_audit.py (ctest ninferno_pdl_sass_audit_sabotage_test): two
// PDL consumers of the same shape. The waiting one reads the producer's output after
// griddepcontrol.wait; the hoisted one reads it through `const float* __restrict__` before the
// wait - a defect that once shipped, placed in source so the SASS carries it whatever the
// scheduler does. The audit must flag the hoisted kernel's parameter 0 (the producer's output)
// and nothing else: the weights (parameter 8) are allowed before the wait in both.
#include "core/pdl.cuh"

namespace ninferno::test::pdl_audit {

__global__ void pdl_audit_hoisted_consumer(const float* __restrict__ produced,
                                           const float* __restrict__ weights, float* out) {
    const int i   = static_cast<int>(threadIdx.x);
    const float w = weights[i];
    // A load pinned before the wait. A plain C++ read of `const __restrict__` data is free to
    // move either way, and in this small kernel nvcc 13.1 sinks it below the wait (so does ptxas
    // with an explicit ld.global.nc); a volatile load keeps its place, so the fixture states the
    // hoisted order itself.
    float v;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(v) : "l"(produced + i) : "memory");
    pdl::wait_for_dependencies();
    out[i] = v * w;
}

__global__ void pdl_audit_waiting_consumer(const float* __restrict__ produced,
                                           const float* __restrict__ weights, float* out) {
    const int i   = static_cast<int>(threadIdx.x);
    const float w = weights[i];
    pdl::wait_for_dependencies();
    out[i] = produced[i] * w;
}

} // namespace ninferno::test::pdl_audit
