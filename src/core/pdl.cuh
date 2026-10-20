#pragma once

#include "core/device_class.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninferno::pdl {

// griddepcontrol needs sm_90+. On sm_89 the device intrinsics must not even compile (ptxas
// rejects the instruction for the target), so every use compiles under NINFERNO_HAS_PDL and the
// portable fallbacks below run instead: trigger is a no-op and the dependent launch becomes a
// plain launch — the consumer just waits for stream order, which is what PDL overlaps away, so
// the kernels stay correct and only lose the overlap.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define NINFERNO_HAS_PDL 1
#else
#define NINFERNO_HAS_PDL 0
#endif

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// The current device's greatest stream priority (the range is the same on every device served).
[[nodiscard]] inline int urgent_priority() {
    static const int priority = [] {
        int least = 0, greatest = 0;
        return cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess ? greatest : 0;
    }();
    return priority;
}

// Whether the current device runs programmatic dependent launches (DeviceFeatures::pdl, sm_90+).
// Below it the kernels' griddepcontrol compiles out, yet a launch that carries the programmatic
// serialization attribute still costs: a captured node takes 1.3-1.5 us on an RTX 3090 against
// 0.84-0.96 us without it, and on an RTX 4090 the norms launched with it alone took the decode of
// Qwen 3.6 35B from 252 to 245 tok/s. So the attribute is set only where it overlaps anything, and
// the launch is otherwise an ordinary stream-ordered one.
[[nodiscard]] inline bool supported() { return device_class().features.pdl; }

// Launches a kernel, as a programmatic dependent of the immediately preceding producer kernel in
// the same stream when `dependent` (every consumer control path that reads producer output must
// first call wait_for_dependencies()), at the device's greatest priority when `urgent`: its CTAs
// are dispatched ahead of the pending CTAs of normal-priority grids running beside it (a branch
// stream's), in a graph too (decode graphs instantiate with the nodes' priorities).
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t launch(const LaunchConfig& launch, bool dependent, bool urgent,
                                        void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attributes[2]{};
    unsigned count = 0;
    if (dependent && supported()) {
        attributes[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[count].val.programmaticStreamSerializationAllowed = 1;
        ++count;
    }
    if (urgent) {
        attributes[count].id           = cudaLaunchAttributePriority;
        attributes[count].val.priority = urgent_priority();
        ++count;
    }

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = count > 0 ? attributes : nullptr;
    config.numAttrs         = count;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    return pdl::launch(launch, true, false, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
__device__ __forceinline__ void trigger_dependents() {
#if NINFERNO_HAS_PDL
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() {
#if NINFERNO_HAS_PDL
    cudaGridDependencySynchronize();
#endif
}

} // namespace ninferno::pdl
