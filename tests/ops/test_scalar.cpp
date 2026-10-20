// Bit-exact public-contract qualification for the finite typed scalar state
// transitions.  Expected values are computed directly from the logical input
// state; no launcher or kernel implementation is used as an oracle.
#include "ninferno/ops/scalar.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

template <typename T>
void store(GuardedDeviceBuffer& buffer, T value) {
    buffer.copy_from_host(&value, sizeof(value));
}

template <typename T>
T load(const GuardedDeviceBuffer& buffer) {
    T value{};
    buffer.copy_to_host(&value, sizeof(value));
    return value;
}

int set_and_increment_i32_contract(cudaStream_t stream) {
    GuardedDeviceBuffer state(sizeof(std::int32_t));
    store<std::int32_t>(state, -918273);
    Tensor scalar(state.data(), DType::I32, {1});

    constexpr std::int32_t assigned = 123456789;
    ops::set_i32_scalar(scalar, assigned, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("set_i32_scalar transition",
                             std::vector<std::int32_t>{load<std::int32_t>(state)},
                             std::vector<std::int32_t>{assigned});
    failures += state.verify_guards("set_i32_scalar destination");

    const std::int32_t expected_increment = assigned + 1;
    ops::increment_i32_scalar(scalar, stream);
    cuda_synchronize(stream);
    failures += verify_exact("increment_i32_scalar transition",
                             std::vector<std::int32_t>{load<std::int32_t>(state)},
                             std::vector<std::int32_t>{expected_increment});
    failures += state.verify_guards("increment_i32_scalar destination");
    return failures;
}

int assign_i32_contract(cudaStream_t stream) {
    GuardedDeviceBuffer source(sizeof(std::int32_t));
    GuardedDeviceBuffer destination(sizeof(std::int32_t));
    constexpr std::int32_t source_value      = -19088743;
    constexpr std::int32_t destination_value = 1985229328;
    store<std::int32_t>(source, source_value);
    store<std::int32_t>(destination, destination_value);
    Tensor source_tensor(source.data(), DType::I32, {1});
    Tensor destination_tensor(destination.data(), DType::I32, {1});

    ops::assign_i32_scalar(source_tensor, destination_tensor, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("assign_i32_scalar destination transition",
                             std::vector<std::int32_t>{load<std::int32_t>(destination)},
                             std::vector<std::int32_t>{source_value});
    failures += verify_exact("assign_i32_scalar source remains unchanged",
                             std::vector<std::int32_t>{load<std::int32_t>(source)},
                             std::vector<std::int32_t>{source_value});
    failures += source.verify_guards("assign_i32_scalar source");
    failures += destination.verify_guards("assign_i32_scalar destination");
    return failures;
}

int add_i32_contract(cudaStream_t stream) {
    GuardedDeviceBuffer lhs(sizeof(std::int32_t));
    GuardedDeviceBuffer rhs(sizeof(std::int32_t));
    GuardedDeviceBuffer destination(sizeof(std::int32_t));
    constexpr std::int32_t lhs_value = 23;
    constexpr std::int32_t rhs_value = 41;
    store<std::int32_t>(lhs, lhs_value);
    store<std::int32_t>(rhs, rhs_value);
    store<std::int32_t>(destination, -1);
    Tensor lhs_tensor(lhs.data(), DType::I32, {1});
    Tensor rhs_tensor(rhs.data(), DType::I32, {1});
    Tensor destination_tensor(destination.data(), DType::I32, {1});

    ops::add_i32_scalars(lhs_tensor, rhs_tensor, destination_tensor, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("add_i32_scalars destination transition",
                             std::vector<std::int32_t>{load<std::int32_t>(destination)},
                             std::vector<std::int32_t>{lhs_value + rhs_value});
    failures += verify_exact("add_i32_scalars lhs remains unchanged",
                             std::vector<std::int32_t>{load<std::int32_t>(lhs)},
                             std::vector<std::int32_t>{lhs_value});
    failures += verify_exact("add_i32_scalars rhs remains unchanged",
                             std::vector<std::int32_t>{load<std::int32_t>(rhs)},
                             std::vector<std::int32_t>{rhs_value});
    failures += lhs.verify_guards("add_i32_scalars lhs");
    failures += rhs.verify_guards("add_i32_scalars rhs");
    failures += destination.verify_guards("add_i32_scalars destination");
    return failures;
}

int increment_i64_contract(cudaStream_t stream) {
    GuardedDeviceBuffer state(sizeof(std::int64_t));
    constexpr std::int64_t initial = (std::int64_t{1} << 45) + 987654321;
    store<std::int64_t>(state, initial);
    Tensor scalar(state.data(), DType::I64, {1});

    ops::increment_i64_scalar(scalar, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("increment_i64_scalar transition",
                             std::vector<std::int64_t>{load<std::int64_t>(state)},
                             std::vector<std::int64_t>{initial + 1});
    failures += state.verify_guards("increment_i64_scalar destination");
    return failures;
}

int fill_i32_contract(cudaStream_t stream) {
    int failures = 0;
    // One element, a DFlash block's rows, and more than one launch grid's stride.
    for (const std::int32_t count : {1, 8, 262147}) {
        GuardedDeviceBuffer state(sizeof(std::int32_t) * static_cast<std::size_t>(count));
        std::vector<std::int32_t> initial(static_cast<std::size_t>(count));
        for (std::int32_t i = 0; i < count; ++i) { initial[static_cast<std::size_t>(i)] = -i - 7; }
        state.copy_from_host(initial.data(), initial.size() * sizeof(std::int32_t));
        Tensor vector(state.data(), DType::I32, {count});
        constexpr std::int32_t value = 16;
        ops::fill_i32(vector, value, stream);
        cuda_synchronize(stream);
        std::vector<std::int32_t> got(static_cast<std::size_t>(count));
        state.copy_to_host(got.data(), got.size() * sizeof(std::int32_t));
        const std::string label = "fill_i32 transition N=" + std::to_string(count);
        failures += verify_exact(label.c_str(), got,
                                 std::vector<std::int32_t>(static_cast<std::size_t>(count), value));
        failures += state.verify_guards("fill_i32 destination");
    }
    return failures;
}

// Byte counts around the 16-byte path and offsets that force the byte path; a pageable host
// pointer is refused. Bytes past the copy keep their sentinel.
int copy_to_pinned_host_contract(cudaStream_t stream) {
    int failures        = 0;
    unsigned char* host = nullptr;
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&host), 4096), "cudaMallocHost");
    for (const std::size_t bytes :
         {std::size_t{1}, std::size_t{15}, std::size_t{16}, std::size_t{1000}, std::size_t{4000}}) {
        for (const std::size_t offset : {std::size_t{0}, std::size_t{3}}) {
            std::vector<unsigned char> source(bytes);
            for (std::size_t i = 0; i < bytes; ++i) {
                source[i] = static_cast<unsigned char>((i * 37U + bytes + offset) & 0xFFU);
            }
            GuardedDeviceBuffer device(bytes + offset);
            device.copy_from_host(std::vector<unsigned char>(bytes + offset, 0).data(),
                                  bytes + offset);
            cuda_check(cudaMemcpyAsync(static_cast<unsigned char*>(device.data()) + offset,
                                       source.data(), bytes, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");
            std::fill_n(host, 4096, static_cast<unsigned char>(0xA5));
            ops::copy_to_pinned_host(
                host + offset, static_cast<unsigned char*>(device.data()) + offset, bytes, stream);
            cuda_synchronize(stream);
            std::vector<unsigned char> got(host + offset, host + offset + bytes);
            std::vector<unsigned char> tail(host + offset + bytes, host + offset + bytes + 16);
            const std::string label = "copy_to_pinned_host bytes=" + std::to_string(bytes) +
                                      " offset=" + std::to_string(offset);
            failures += verify_exact(label.c_str(), got, source);
            failures +=
                verify_exact((label + " tail").c_str(), tail, std::vector<unsigned char>(16, 0xA5));
        }
    }
    cuda_check(cudaFreeHost(host), "cudaFreeHost");

    std::vector<unsigned char> pageable(64);
    GuardedDeviceBuffer device(64);
    bool refused = false;
    try {
        ops::copy_to_pinned_host(pageable.data(), device.data(), 64, stream);
    } catch (const std::invalid_argument&) { refused = true; }
    if (!refused) {
        std::cout << "FAIL copy_to_pinned_host accepted pageable host memory" << std::endl;
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    int failures = 0;
    failures += set_and_increment_i32_contract(stream);
    failures += assign_i32_contract(stream);
    failures += add_i32_contract(stream);
    failures += increment_i64_contract(stream);
    failures += fill_i32_contract(stream);
    failures += copy_to_pinned_host_contract(stream);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " scalar public contract\n";
    return failures == 0 ? 0 : 1;
}
