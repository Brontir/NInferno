#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops {

/**
 * Op: finite typed scalar state transitions
 *
 * Math / indexing:
 *   set_i32_scalar:       destination' = value
 *   assign_i32_scalar:    destination' = source
 *   add_i32_scalars:      destination' = lhs + rhs
 *   increment_i32_scalar: scalar' = scalar + 1
 *   increment_i64_scalar: scalar' = scalar + 1
 *   fill_i32:             destination'[i] = value, 0 <= i < N
 *
 * Logical shapes:
 *   Every tensor argument is one contiguous scalar element of the named dtype, except fill_i32's
 *   destination: a contiguous I32 vector [N], N >= 1.
 *
 * Numeric:
 *   Add/increment callers keep the exact result within the corresponding signed integer range.
 *
 * Effects:
 *   Each call writes only its destination scalar (fill_i32: the N destination elements, whatever
 *   they held). assign_i32_scalar and add_i32_scalars require
 *   destination storage distinct from their inputs; the other calls update their single
 *   destination in place.
 *
 * Workspace:
 *   None. There is no state side effect beyond the stated destination transition.
 */
void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream);
void assign_i32_scalar(const Tensor& source, Tensor& destination, cudaStream_t stream);
void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     cudaStream_t stream);
void increment_i32_scalar(Tensor& scalar, cudaStream_t stream);
void increment_i64_scalar(Tensor& scalar, cudaStream_t stream);
void fill_i32(Tensor& destination, std::int32_t value, cudaStream_t stream);

/**
 * Op: device bytes into pinned host memory, written by a kernel
 *
 *   copy_to_pinned_host: host'[0, bytes) = device[0, bytes)
 *
 * host is page-locked host memory (cudaMallocHost / cudaHostAlloc); the kernel stores into it
 * through its device mapping, so the call queues no copy-engine work. A decode graph ends with
 * this copy of its egress: as a memcpy node it waits on the copy engine for the round's sampler,
 * and a host-to-device copy queued behind it on the same engine (the token mask release the
 * sampler spins on) then waits for the sampler in turn until its release wait expires.
 */
void copy_to_pinned_host(void* host, const void* device, std::size_t bytes, cudaStream_t stream);

} // namespace ninferno::ops
