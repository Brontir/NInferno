#include "models/qwen3_5/program/numeric_status.h"

#include "core/device.h" // CUDA_CHECK
#include "ninferno/ops/finite_check.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace ninferno::models::qwen3_5::detail {
namespace {

constexpr std::size_t kWords = 3U * kMaximumConcurrency;

} // namespace

NumericStatus::NumericStatus() {
    void* host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, kWords * sizeof(std::uint32_t),
                             cudaHostAllocMapped | cudaHostAllocPortable));
    void* mapped             = nullptr;
    const cudaError_t status = cudaHostGetDevicePointer(&mapped, host, 0);
    if (status != cudaSuccess) {
        cudaFreeHost(host);
        CUDA_CHECK(status);
    }
    host_   = static_cast<std::uint32_t*>(host);
    device_ = static_cast<std::uint32_t*>(mapped);
    clear();
    if (const char* file = std::getenv("NINFERNO_DEBUG_NAN_FILE"); file != nullptr) {
        inject_file_ = file;
    }
}

NumericStatus::~NumericStatus() {
    if (host_ != nullptr) { cudaFreeHost(host_); }
}

void NumericStatus::clear() noexcept { std::fill_n(host_, kWords, 0U); }

void NumericStatus::check(Launch launch, const Tensor& values, const std::int32_t* valid_columns,
                          cudaStream_t stream) {
    std::error_code ignored;
    if (!inject_file_.empty() && std::filesystem::remove(inject_file_, ignored)) {
        CUDA_CHECK(cudaMemsetAsync(values.data, 0xFF, dtype_size(values.dtype), stream));
    }
    ::ninferno::ops::finite_check(values, valid_columns, device_ + offset(launch), stream);
}

bool NumericStatus::faulted(Launch launch, std::size_t row) const noexcept {
    return static_cast<const volatile std::uint32_t*>(host_)[offset(launch) + row] != 0U;
}

} // namespace ninferno::models::qwen3_5::detail
