#pragma once
#include "core/per_device.h"
#include "core/device.h"
#include "ops/common/arch_limits.h"

#include <stdexcept>

namespace ninferno::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* q8_shared_storage() {
    // No upper bound here: a tile over the build's shared memory (Turing's 64 KiB) compiles, and
    // its launch is refused on the host (*_prepare_shared). NINFERNO-SPECIAL(arch:sm75)
    if constexpr (Bytes > kStaticSharedBytesPerBlock) {
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        return dynamic_shared;
    } else {
        __shared__ __align__(16) unsigned char storage[Bytes];
        return storage;
    }
}

template <int Bytes, auto Kernel>
int q8_prepare_shared() {
    if constexpr (Bytes > kMaxSharedBytesPerBlock) {
        // Routes take a fitting schedule (kFitsSharedLimit); a direct call is refused.
#ifdef NINFERNO_SHARED_AUDIT
        shared_limit_audit<Bytes, Kernel>();
#endif
        throw std::invalid_argument("Q8: schedule exceeds this build's shared memory");
    } else if constexpr (Bytes > kStaticSharedBytesPerBlock) {
        static ::ninferno::PerDevice<bool> attribute;
        ::ninferno::allow_dynamic_smem(attribute, Kernel, Bytes);
        return Bytes;
    } else
        return 0;
}
} // namespace ninferno::ops::detail
