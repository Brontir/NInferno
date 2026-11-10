// The selector on the real device, with no substituted board (#620): every per-device cache
// (PerDevice) reads the ordinal and the class of the GPU the process runs on. The golden test runs
// the selector under ScopedDeviceClass only, so it never exercises the real-device reads; this one
// calls select_route and w4a8_ws_extra bare over four formats, three policies and five widths, and
// requires r8_decode_table to be non-null on a board that has a measured table of its own. The
// device reads catch their own exceptions (sm_count falls back to 1, bf16_mma_native to false,
// r8_decode_table to nullptr), so "does not throw" proves little by itself; the table check is the
// one that sees a silent fallback. Skips (77) without a GPU.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/r8_decode_table.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

int check() {
    int failures          = 0;
    const char* names[]   = {"Q4_G64_FP16", "GGML_Q4_K", "GGML_IQ2_S", "BF16"};
    const QType formats[] = {QType::Q4_G64_FP16, QType::GGML_Q4_K, QType::GGML_IQ2_S, QType::BF16};
    for (std::size_t i = 0; i < 4; ++i) {
        try {
            (void)generic::w4a8_ws_extra(formats[i]);
        } catch (const std::exception& e) {
            std::cerr << "w4a8_ws_extra(" << names[i] << ") threw: " << e.what() << "\n";
            ++failures;
        }
        Weight w;
        w.qtype = formats[i];
        w.n     = 4096;
        w.k     = 4096;
        w.group = 64;
        for (const LinearPolicy policy :
             {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
            const generic::Problem problem{formats[i], policy, w.n, w.k, generic::Epilogue::Store};
            for (const std::int32_t t : {1, 4, 16, 64, 512}) {
                try {
                    (void)generic::select_route(problem, w, t, t > 64);
                } catch (const std::exception& e) {
                    std::cerr << "select_route(" << names[i] << ", T=" << t
                              << ") threw: " << e.what() << "\n";
                    ++failures;
                }
            }
        }
    }

    // A board with a measured R8 decode table of its own gets it: nullptr would mean the tuned
    // schedules are never used.
    const std::string id = device_class().id();
    bool has_table       = false;
    for (const auto& t : generic::kR8DecodeTables) { has_table = has_table || t.gpu == id; }
    try {
        const bool present = generic::r8_decode_table() != nullptr;
        if (has_table && !present) {
            std::cerr << "r8_decode_table is null on " << id << ", which has a measured table\n";
            ++failures;
        }
        std::cout << id << ": r8_decode_table " << (present ? "present" : "absent")
                  << (has_table ? " (measured board)" : " (no table of its own)") << "\n";
    } catch (const std::exception& e) {
        std::cerr << "r8_decode_table threw: " << e.what() << "\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "CUDA unavailable; skipping the real-device selector test\n";
        return 77;
    }
    try {
        return check() == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "device_class threw: " << e.what() << "\n";
        return 1;
    }
}
