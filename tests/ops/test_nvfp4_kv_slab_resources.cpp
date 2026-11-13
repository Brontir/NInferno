// The NVFP4 KV slab kernel's instances (nvfp4/slab_mma.cuh, #849) as the running GPU holds them:
// no local memory (the law's instruction count has no spill in it, and a spill is a refusal), at
// most 255 registers, the launch's shared memory within the build's opt-in, and at least the CTAs
// per SM the plan's wave partition assumes (slab_rule.h). Each instance's numbers are printed.
// Skips (77) without a GPU, where the GPU does not run this build's SASS, or where the build
// launches no slab instance (sm_70 and sm_75 keep the grouped kernel).

#include "ops/common/arch_limits.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"

#include <cuda_runtime.h>

#include <iostream>

namespace {

constexpr int kSkip = 77;

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return kSkip;
    }
    const auto instances = ninferno::ops::detail::nvfp4_kv_slab_kernel_resources();
    if (instances.empty()) {
        std::cout << "SKIP: this build launches no slab instance\n";
        return kSkip;
    }
    int failures = 0, held = 0;
    for (const auto& r : instances) {
        if (r.status == cudaErrorNoKernelImageForDevice ||
            r.status == cudaErrorInvalidDeviceFunction) {
            std::cout << r.name << ": not this GPU's image (" << cudaGetErrorString(r.status)
                      << ")\n";
            continue;
        }
        ++held;
        const std::size_t shared = r.static_shared_bytes + std::size_t(r.dynamic_shared_bytes);
        std::cout << r.name << ": " << r.registers << " registers, " << r.local_bytes
                  << " B local, " << r.static_shared_bytes << " + " << r.dynamic_shared_bytes
                  << " B shared, " << r.blocks_per_sm << " CTAs/SM of " << r.threads
                  << " threads\n";
        const bool fails = r.status != cudaSuccess || r.local_bytes != 0 || r.registers > 255 ||
                           shared > ninferno::ops::kMaxSharedBytesPerBlock ||
                           r.blocks_per_sm < r.min_blocks;
        if (fails) {
            std::cerr << r.name << ": FAIL (" << cudaGetErrorString(r.status) << "; local "
                      << r.local_bytes << ", registers " << r.registers << ", shared " << shared
                      << " of " << ninferno::ops::kMaxSharedBytesPerBlock << ", CTAs/SM "
                      << r.blocks_per_sm << " of " << r.min_blocks << " assumed)\n";
            ++failures;
        }
    }
    if (held == 0) {
        std::cout << "SKIP: the GPU runs none of this build's slab kernels\n";
        return kSkip;
    }
    if (failures != 0) {
        std::cerr << "FAIL: " << failures << " of " << held << " slab instances\n";
        return 1;
    }
    std::cout << "PASS: " << held << " slab instances\n";
    return 0;
}
