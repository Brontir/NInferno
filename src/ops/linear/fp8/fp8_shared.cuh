#pragma once
#include "core/per_device.h"
#include "ops/common/arch_limits.h"
#include "core/device.h"

#include <stdexcept>

namespace ninferno::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* fp8_shared_storage() {
    // No upper bound here: a tile over the build's shared memory (Turing's 64 KiB) compiles, and
    // its launch is refused on the host (*_prepare_shared). NINFERNO-SPECIAL(arch:sm75)
    static_assert(Bytes > 0);
    if constexpr (Bytes > kStaticSharedBytesPerBlock) {
        extern __shared__ __align__(16) unsigned char shared_dynamic[];
        return shared_dynamic;
    } else {
        __shared__ __align__(16) unsigned char shared_static[Bytes];
        return shared_static;
    }
}

template <int Bytes, auto Kernel, bool Dynamic = false>
int fp8_prepare_shared() {
    if constexpr (Bytes > kMaxSharedBytesPerBlock) {
        // Routes take a fitting schedule (kFitsSharedLimit); a direct call is refused.
#ifdef NINFERNO_SHARED_AUDIT
        shared_limit_audit<Bytes, Kernel>();
#endif
        throw std::invalid_argument("FP8: schedule exceeds this build's shared memory");
    } else if constexpr (Bytes > kStaticSharedBytesPerBlock) {
        static ::ninferno::PerDevice<bool> status;
        ::ninferno::allow_dynamic_smem(status, Kernel, Bytes);
        return Bytes;
    } else {
        return Dynamic ? Bytes : 0;
    }
}

template <class Schedule, class Epilogue>
inline constexpr int fp8_mma_shared_bytes = [] {
    constexpr int extra = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return Epilogue::template kSharedBytes<Schedule>;
        else
            return 0;
    }();
    return Schedule::kSharedBytes > extra ? Schedule::kSharedBytes : extra;
}();
} // namespace ninferno::ops::detail
