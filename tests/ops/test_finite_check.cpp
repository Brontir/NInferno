#include "ninferno/ops/finite_check.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

// Values [H,C,R] of finite numbers; `poison` lists (column,row) cells whose middle element is set
// to `bits` (a NaN or an infinity). A row is flagged when a poisoned cell lies inside its valid
// columns.
int run_case(DType dtype, std::int32_t hidden, std::int32_t columns, std::int32_t rows,
             const std::vector<std::int32_t>& valid, std::uint32_t bits,
             const std::vector<std::pair<std::int32_t, std::int32_t>>& poison) {
    const std::size_t element = dtype == DType::FP32 ? 4U : 2U;
    const std::size_t count   = static_cast<std::size_t>(hidden) * columns * rows;
    std::vector<unsigned char> host(count * element);
    for (std::size_t i = 0; i < count; ++i) {
        const float value = -3.0f + static_cast<float>(i % 97U) * 0.0625f;
        if (dtype == DType::FP32) {
            std::memcpy(host.data() + i * 4U, &value, 4U);
        } else {
            // F16: finite bit patterns around 0.5..0.6 (exponent 0b01110).
            const std::uint16_t half = dtype == DType::BF16
                                           ? f32_to_bf16(value)
                                           : static_cast<std::uint16_t>(0x3800U + i % 97U);
            std::memcpy(host.data() + i * 2U, &half, 2U);
        }
    }
    std::vector<std::uint32_t> expected(static_cast<std::size_t>(rows), 0U);
    const std::uint32_t exponent = dtype == DType::FP32   ? 0x7F800000U
                                   : dtype == DType::BF16 ? 0x7F80U
                                                          : 0x7C00U;
    const bool flags             = (bits & exponent) == exponent;
    for (const auto& [column, row] : poison) {
        const std::size_t index =
            (static_cast<std::size_t>(row) * columns + column) * hidden + hidden / 2;
        if (dtype == DType::FP32) {
            std::memcpy(host.data() + index * 4U, &bits, 4U);
        } else {
            const auto half = static_cast<std::uint16_t>(bits);
            std::memcpy(host.data() + index * 2U, &half, 2U);
        }
        if (flags && (valid.empty() || column < valid[static_cast<std::size_t>(row)])) {
            expected[static_cast<std::size_t>(row)] = 1U;
        }
    }

    GuardedDeviceBuffer device_values(host.size());
    GuardedDeviceBuffer device_status(static_cast<std::size_t>(rows) * sizeof(std::uint32_t));
    GuardedDeviceBuffer device_valid(static_cast<std::size_t>(rows) * sizeof(std::int32_t));
    device_values.copy_from_host(host.data(), host.size());
    device_status.fill(0);
    if (!valid.empty()) {
        device_valid.copy_from_host(valid.data(), valid.size() * sizeof(std::int32_t));
    }
    Tensor values(device_values.data(), dtype, {hidden, columns, rows});
    ops::finite_check(
        values, valid.empty() ? nullptr : static_cast<const std::int32_t*>(device_valid.data()),
        static_cast<std::uint32_t*>(device_status.data()), nullptr);
    cuda_synchronize();

    const std::string label = std::string("finite_check ") +
                              (dtype == DType::FP32   ? "F32"
                               : dtype == DType::BF16 ? "BF16"
                                                      : "F16") +
                              " H=" + std::to_string(hidden) + " C=" + std::to_string(columns) +
                              " R=" + std::to_string(rows) + " bits=" + std::to_string(bits);
    int failures = verify_exact(
        label.c_str(),
        from_device<std::uint32_t>(device_status.data(), static_cast<std::size_t>(rows)), expected);
    failures += device_values.verify_guards((label + " values").c_str());
    failures += device_status.verify_guards((label + " status").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // An ordinary decode's hidden [H,1,B]: a NaN in row 2 flags it alone.
    failures += run_case(DType::BF16, 5120, 1, 4, {}, 0x7FC0U, {{0, 2}});
    failures += run_case(DType::BF16, 5120, 1, 4, {}, 0xFF80U, {{0, 0}, {0, 3}}); // -inf
    failures += run_case(DType::BF16, 2048, 1, 8, {}, 0x7FC0U, {});
    // A verify launch [H,width,B]: a NaN past a row's verified columns does not count.
    failures += run_case(DType::BF16, 4096, 5, 3, {2, 5, 1}, 0x7FC0U, {{3, 0}, {4, 1}, {0, 2}});
    failures += run_case(DType::FP16, 1000, 3, 2, {3, 3}, 0x7C00U, {{2, 1}});
    failures += run_case(DType::FP32, 777, 2, 2, {}, 0x7FC00000U, {{1, 0}});
    // The largest finite magnitudes are not flagged.
    failures += run_case(DType::BF16, 64, 1, 2, {}, 0x7F7FU, {{0, 1}});
    failures += run_case(DType::FP16, 64, 1, 2, {}, 0xFBFFU, {{0, 0}});
    std::cout << (failures ? "FAIL" : "OK") << " finite_check\n";
    return failures ? 1 : 0;
}
