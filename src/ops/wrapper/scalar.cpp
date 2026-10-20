#include "ninferno/ops/scalar.h"

#include "core/cuda_error.h"
#include "ops/launcher/scalar.h"

#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require_scalar(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != 1 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be a non-null contiguous scalar");
    }
}

} // namespace

void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream) {
    require_scalar(destination, DType::I32, "set_i32_scalar destination");
    detail::set_i32_scalar_launch(destination, value, stream);
}

void assign_i32_scalar(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    require_scalar(source, DType::I32, "assign_i32_scalar source");
    require_scalar(destination, DType::I32, "assign_i32_scalar destination");
    if (source.data == destination.data) {
        throw std::invalid_argument("assign_i32_scalar: source and destination must not alias");
    }
    detail::assign_i32_scalar_launch(source, destination, stream);
}

void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     cudaStream_t stream) {
    require_scalar(lhs, DType::I32, "add_i32_scalars lhs");
    require_scalar(rhs, DType::I32, "add_i32_scalars rhs");
    require_scalar(destination, DType::I32, "add_i32_scalars destination");
    if (destination.data == lhs.data || destination.data == rhs.data) {
        throw std::invalid_argument("add_i32_scalars: inputs must not alias destination");
    }
    detail::add_i32_scalars_launch(lhs, rhs, destination, stream);
}

void increment_i32_scalar(Tensor& scalar, cudaStream_t stream) {
    require_scalar(scalar, DType::I32, "increment_i32_scalar scalar");
    detail::increment_i32_scalar_launch(scalar, stream);
}

void increment_i64_scalar(Tensor& scalar, cudaStream_t stream) {
    require_scalar(scalar, DType::I64, "increment_i64_scalar scalar");
    detail::increment_i64_scalar_launch(scalar, stream);
}

void fill_i32(Tensor& destination, std::int32_t value, cudaStream_t stream) {
    if (destination.dtype != DType::I32 || destination.ne[0] < 1 || destination.ne[1] != 1 ||
        destination.ne[2] != 1 || destination.ne[3] != 1 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument(
            "fill_i32 destination must be a non-null contiguous I32 vector");
    }
    detail::fill_i32_launch(destination, value, stream);
}

void copy_to_pinned_host(void* host, const void* device, std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) { return; }
    if (host == nullptr || device == nullptr) {
        throw std::invalid_argument("copy_to_pinned_host: null pointer");
    }
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, host);
    // The query also returns an earlier launch's asynchronous failure: that is the device's
    // (CudaError, HTTP 503), not a wrong argument. An unregistered pointer is
    // cudaErrorInvalidValue.
    if (status != cudaSuccess && status != cudaErrorInvalidValue) {
        throw CudaError(status, cuda_error_message("copy_to_pinned_host", status));
    }
    if (status != cudaSuccess || attributes.type != cudaMemoryTypeHost ||
        attributes.devicePointer == nullptr) {
        (void)cudaGetLastError();
        throw std::invalid_argument("copy_to_pinned_host: host is not mapped page-locked memory");
    }
    detail::copy_to_pinned_host_launch(attributes.devicePointer, device, bytes, stream);
}

} // namespace ninferno::ops
