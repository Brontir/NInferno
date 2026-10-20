#pragma once

// Host-visible limits shared by the rope wrapper and its CUDA kernels. Kept out of rope.cuh so
// the wrapper does not have to be compiled by nvcc to learn one bound.

namespace ninferno::ops {

// Coefficient cache width of the generic kernel, in rotated pairs (half the rotated width).
inline constexpr int kRopeMaxHalf = 128;

} // namespace ninferno::ops
