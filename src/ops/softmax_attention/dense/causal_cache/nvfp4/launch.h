#pragma once

#include "ninferno/ops/softmax_attention.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ninferno::ops::detail {

void nvfp4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution);

void nvfp4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution);

// One slab kernel instance this build launches (slab_mma.cuh, #849) as the running GPU holds it:
// cudaFuncGetAttributes and the driver's occupancy at the launch's threads and dynamic shared
// memory. The resource test reads them; no decision does.
struct Nvfp4KvSlabKernelResources {
    std::string name; // geometry, tokens and the variant flags
    int registers                   = 0;
    std::size_t local_bytes         = 0;
    std::size_t static_shared_bytes = 0;
    int dynamic_shared_bytes        = 0;
    int threads                     = 0;
    int min_blocks                  = 0; // the schedule's, which the plan's wave partition assumes
    int blocks_per_sm               = 0;
    cudaError_t status              = cudaSuccess; // why the GPU does not hold the kernel, if not
};

std::vector<Nvfp4KvSlabKernelResources> nvfp4_kv_slab_kernel_resources();

} // namespace ninferno::ops::detail
