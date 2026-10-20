#pragma once

// ninferno::ops::detail - private launch prototype for sigmoid_mul. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void sigmoid_gate_mul_launch(const Tensor& gate, Tensor& x, cudaStream_t stream);

// A gate whose columns sit gate.nb[1] bytes apart (rows of a wider plane); x contiguous.
// Needs 16-byte alignment, ne[0] % 8 == 0 and nb[1] % 16 == 0 (throws otherwise).
void sigmoid_gate_mul_rows_launch(const Tensor& gate, Tensor& x, cudaStream_t stream);

// Fixed-route launch control used by production and qualification benchmarks.
void sigmoid_gate_mul_bf16x8_launch(const Tensor& gate, Tensor& x, int block, cudaStream_t stream);

} // namespace ninferno::ops::detail
