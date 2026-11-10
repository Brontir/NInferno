#include "core/kernel_occupancy.h"

#include <cuda_occupancy.h>

namespace ninferno {

int resident_ctas_per_sm(const DeviceClass& d, const KernelShape& k) {
    if (d.major <= 0 || d.sms <= 0 || d.max_threads_per_sm <= 0 || d.registers_per_sm <= 0 ||
        d.smem_per_sm == 0 || k.registers <= 0 || k.threads <= 0) {
        return 0;
    }
    // The device as the calculator describes it: the limits DeviceClass carries (the driver's
    // on a real device, the architecture table's on a synthetic one), the per-block constants of
    // every architecture from Volta on (1024 threads and 64 K registers per block, 48 KiB of
    // shared memory without the opt-in).
    cudaOccDeviceProp p;
    p.computeMajor                = d.major;
    p.computeMinor                = d.minor;
    p.maxThreadsPerBlock          = 1024;
    p.maxThreadsPerMultiprocessor = d.max_threads_per_sm;
    p.regsPerBlock                = 64 * 1024;
    p.regsPerMultiprocessor       = d.registers_per_sm;
    p.warpSize                    = 32;
    p.sharedMemPerBlock           = 48 * 1024;
    p.sharedMemPerMultiprocessor  = d.smem_per_sm;
    p.numSms                      = d.sms;
    p.sharedMemPerBlockOptin      = d.smem_per_block_optin;
    p.reservedSharedMemPerBlock   = reserved_shared_bytes_per_cta(d.major);
    // The kernel as its attributes describe it, without a dynamic shared-memory opt-in (the
    // register-MMA launches stay within the 48 KiB; a launch past it is refused here as it would
    // be by the driver without the attribute).
    cudaOccFuncAttributes a;
    a.maxThreadsPerBlock        = k.threads;
    a.numRegs                   = k.registers;
    a.sharedSizeBytes           = k.static_shared_bytes;
    a.partitionedGCConfig       = PARTITIONED_GC_OFF;
    a.shmemLimitConfig          = FUNC_SHMEM_LIMIT_DEFAULT;
    a.maxDynamicSharedSizeBytes = 0;
    a.numBlockBarriers          = 1;
    const cudaOccDeviceState state; // the default carveout: the whole shared memory of the SM
    cudaOccResult result;
    if (cudaOccMaxActiveBlocksPerMultiprocessor(&result, &p, &a, &state, k.threads,
                                                k.dynamic_shared_bytes) != CUDA_OCC_SUCCESS) {
        return 0;
    }
    return result.activeBlocksPerMultiprocessor;
}

} // namespace ninferno
