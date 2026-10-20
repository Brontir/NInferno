#pragma once

// Whether a CUDA call failed because this machine has no usable GPU: the tests then skip.

#include <cuda_runtime.h>

inline bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}
