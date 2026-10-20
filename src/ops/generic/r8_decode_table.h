#pragma once

// R8 decode (T = 1) schedules for exact problems (codebook i-quants, K-quants), per GPU
// (class id: chip and SM count): the fastest compiled schedule where it beats the rules
// (r8_rules_route) by at least 0.5 % in the median of repeated ninferno_r8_schedule_bench
// sweeps and in every sweep. Every R8 schedule sums a row in one order, so an entry changes
// no output byte. Written by tools/bench/r8_decode_table.py; do not edit by hand. An add
// call takes the store entry (the same kernel up to the final read-modify-write).

#include "core/device_class.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace ninferno::ops::detail::generic {

struct R8DecodeEntry {
    QType qtype;
    Epilogue epilogue; // Store or SwiGlu
    std::int32_t n;    // the op's output rows (SwiGLU: the gate rows)
    std::int32_t k;
    std::int8_t warps;
    std::int8_t rows;
};

struct R8DecodeTable {
    std::string_view gpu; // DeviceClass::id() of the board, "gb202:170sm"
    Board board;          // the board the sweeps ran on: the device takes the nearest
    std::span<const R8DecodeEntry> entries;
};

// ad102:128sm: 106 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeAd102_128sm[] = {
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 2048, 4, 1},     // w4r2 5.89 -> 5.73
    {QType::GGML_IQ2_S, Epilogue::Store, 8192, 2048, 4, 1},     // w4r2 9.76 -> 9.44
    {QType::GGML_IQ2_S, Epilogue::Store, 12288, 2048, 4, 1},    // w8r2 12.87 -> 12.34
    {QType::GGML_IQ2_S, Epilogue::Store, 6656, 4096, 4, 1},     // w4r2 14.93 -> 13.38
    {QType::GGML_IQ2_S, Epilogue::Store, 6144, 5120, 4, 1},     // w4r2 18.50 -> 16.28
    {QType::GGML_IQ2_S, Epilogue::Store, 12288, 5120, 4, 1},    // w8r2 26.95 -> 25.65
    {QType::GGML_IQ2_S, Epilogue::Store, 14336, 5120, 4, 1},    // w8r2 32.80 -> 29.51
    {QType::GGML_IQ2_S, Epilogue::Store, 16384, 5120, 4, 1},    // w8r2 34.87 -> 32.75
    {QType::GGML_IQ2_S, Epilogue::Store, 34816, 5120, 4, 1},    // w8r2 74.10 -> 65.01
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 5376, 8, 2},     // w4r2 11.25 -> 10.97
    {QType::GGML_IQ2_S, Epilogue::Store, 8192, 5376, 4, 1},     // w4r2 21.18 -> 19.81
    {QType::GGML_IQ2_S, Epilogue::Store, 16384, 5376, 4, 1},    // w8r2 38.25 -> 36.23
    {QType::GGML_IQ2_S, Epilogue::Store, 18432, 5376, 4, 1},    // w8r2 44.70 -> 40.64
    {QType::GGML_IQ2_S, Epilogue::Store, 43008, 5376, 4, 1},    // w8r2 100.20 -> 89.36
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 6656, 8, 2},     // w4r2 12.97 -> 12.82
    {QType::GGML_IQ2_S, Epilogue::Store, 19968, 6656, 4, 2},    // w8r2 54.46 -> 53.11
    {QType::GGML_IQ2_S, Epilogue::Store, 39936, 6656, 4, 2},    // w8r2 107.45 -> 102.06
    {QType::GGML_IQ2_S, Epilogue::Store, 5376, 8192, 4, 1},     // w8r2 24.85 -> 23.64
    {QType::GGML_IQ2_S, Epilogue::Store, 5376, 21504, 4, 1},    // w8r2 61.20 -> 60.76
    {QType::GGML_IQ2_S, Epilogue::SwiGlu, 17408, 5120, 4, 1},   // w8r2 78.06 -> 65.02
    {QType::GGML_IQ2_S, Epilogue::SwiGlu, 19968, 6656, 4, 1},   // w8r2 107.96 -> 97.02
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 2048, 4, 1},    // w4r2 5.27 -> 4.78
    {QType::GGML_IQ2_XS, Epilogue::Store, 8192, 2048, 4, 1},    // w4r2 9.45 -> 7.69
    {QType::GGML_IQ2_XS, Epilogue::Store, 9216, 2048, 4, 1},    // w4r2 9.37 -> 8.19
    {QType::GGML_IQ2_XS, Epilogue::Store, 12288, 2048, 4, 1},   // w8r2 11.66 -> 10.03
    {QType::GGML_IQ2_XS, Epilogue::Store, 6144, 5120, 8, 2},    // w4r2 13.16 -> 12.83
    {QType::GGML_IQ2_XS, Epilogue::Store, 12288, 5120, 4, 1},   // w8r2 23.48 -> 22.29
    {QType::GGML_IQ2_XS, Epilogue::Store, 14336, 5120, 4, 1},   // w8r2 29.56 -> 25.51
    {QType::GGML_IQ2_XS, Epilogue::Store, 16384, 5120, 4, 1},   // w8r2 31.06 -> 28.60
    {QType::GGML_IQ2_XS, Epilogue::Store, 34816, 5120, 4, 1},   // w8r2 62.94 -> 57.36
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 5376, 8, 2},    // w4r2 9.93 -> 9.64
    {QType::GGML_IQ2_XS, Epilogue::Store, 8192, 5376, 4, 1},    // w4r2 21.21 -> 17.32
    {QType::GGML_IQ2_XS, Epilogue::Store, 16384, 5376, 4, 1},   // w8r2 33.72 -> 30.82
    {QType::GGML_IQ2_XS, Epilogue::Store, 18432, 5376, 4, 1},   // w8r2 37.67 -> 34.78
    {QType::GGML_IQ2_XS, Epilogue::Store, 43008, 5376, 4, 1},   // w8r2 83.59 -> 76.29
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 6656, 8, 2},    // w4r2 11.81 -> 11.42
    {QType::GGML_IQ2_XS, Epilogue::Store, 8704, 6656, 8, 2},    // w4r2 25.09 -> 23.06
    {QType::GGML_IQ2_XS, Epilogue::Store, 19968, 6656, 4, 1},   // w8r2 50.33 -> 46.13
    {QType::GGML_IQ2_XS, Epilogue::Store, 39936, 6656, 4, 2},   // w8r2 92.50 -> 90.44
    {QType::GGML_IQ2_XS, Epilogue::Store, 6656, 19968, 4, 2},   // w8r2 64.82 -> 58.13
    {QType::GGML_IQ2_XS, Epilogue::SwiGlu, 17408, 5120, 4, 1},  // w8r2 69.26 -> 58.50
    {QType::GGML_IQ2_XS, Epilogue::SwiGlu, 19968, 6656, 4, 1},  // w8r2 95.83 -> 88.85
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4096, 2048, 4, 1},   // w4r2 4.69 -> 4.40
    {QType::GGML_IQ2_XXS, Epilogue::Store, 8192, 2048, 4, 1},   // w4r2 8.06 -> 7.00
    {QType::GGML_IQ2_XXS, Epilogue::Store, 9216, 2048, 4, 1},   // w4r2 8.11 -> 7.47
    {QType::GGML_IQ2_XXS, Epilogue::Store, 12288, 2048, 4, 1},  // w8r2 9.93 -> 9.01
    {QType::GGML_IQ2_XXS, Epilogue::Store, 6656, 4096, 4, 1},   // w4r2 10.05 -> 9.80
    {QType::GGML_IQ2_XXS, Epilogue::Store, 6144, 5120, 8, 2},   // w4r2 11.66 -> 11.59
    {QType::GGML_IQ2_XXS, Epilogue::Store, 12288, 5120, 4, 1},  // w8r2 20.67 -> 19.95
    {QType::GGML_IQ2_XXS, Epilogue::Store, 14336, 5120, 4, 1},  // w8r2 25.17 -> 22.87
    {QType::GGML_IQ2_XXS, Epilogue::Store, 16384, 5120, 4, 1},  // w8r2 26.49 -> 25.58
    {QType::GGML_IQ2_XXS, Epilogue::Store, 34816, 5120, 4, 1},  // w8r2 52.55 -> 51.32
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4096, 5376, 8, 2},   // w4r2 9.06 -> 8.67
    {QType::GGML_IQ2_XXS, Epilogue::Store, 8192, 5376, 4, 1},   // w4r2 17.63 -> 15.21
    {QType::GGML_IQ2_XXS, Epilogue::Store, 16384, 5376, 4, 1},  // w8r2 29.92 -> 28.04
    {QType::GGML_IQ2_XXS, Epilogue::Store, 18432, 5376, 4, 1},  // w8r2 33.13 -> 31.13
    {QType::GGML_IQ2_XXS, Epilogue::Store, 43008, 5376, 4, 1},  // w8r2 72.69 -> 68.76
    {QType::GGML_IQ2_XXS, Epilogue::Store, 5120, 6144, 8, 2},   // w4r2 11.59 -> 11.42
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4096, 6656, 8, 2},   // w4r2 10.84 -> 10.39
    {QType::GGML_IQ2_XXS, Epilogue::Store, 8704, 6656, 4, 1},   // w4r2 21.59 -> 20.50
    {QType::GGML_IQ2_XXS, Epilogue::Store, 19968, 6656, 4, 1},  // w8r2 44.13 -> 40.70
    {QType::GGML_IQ2_XXS, Epilogue::Store, 39936, 6656, 4, 2},  // w8r2 81.13 -> 79.33
    {QType::GGML_IQ2_XXS, Epilogue::Store, 6656, 19968, 4, 2},  // w8r2 55.04 -> 48.06
    {QType::GGML_IQ2_XXS, Epilogue::SwiGlu, 17408, 5120, 4, 1}, // w8r2 55.09 -> 51.34
    {QType::GGML_IQ2_XXS, Epilogue::SwiGlu, 19968, 6656, 4, 1}, // w8r2 81.41 -> 77.98
    {QType::GGML_IQ3_S, Epilogue::Store, 4096, 5376, 8, 2},     // w4r1 13.76 -> 13.47
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 6144, 4, 2},     // w4r1 19.32 -> 17.89
    {QType::GGML_IQ3_S, Epilogue::Store, 4096, 6656, 8, 2},     // w4r1 15.97 -> 15.89
    {QType::GGML_IQ3_S, Epilogue::Store, 4608, 6656, 4, 2},     // w4r1 20.39 -> 18.26
    {QType::GGML_IQ3_S, Epilogue::Store, 5376, 8192, 4, 1},     // w8r2 27.77 -> 25.92
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 17408, 8, 1},    // w8r2 54.03 -> 50.71
    {QType::GGML_IQ3_S, Epilogue::Store, 5376, 21504, 4, 1},    // w8r2 67.75 -> 66.95
    {QType::GGML_IQ3_XXS, Epilogue::Store, 4096, 6656, 8, 2},   // w4r1 13.55 -> 13.29
    {QType::GGML_IQ3_XXS, Epilogue::Store, 4608, 6656, 4, 2},   // w4r1 16.72 -> 15.29
    {QType::GGML_IQ3_XXS, Epilogue::Store, 6656, 19968, 4, 2},  // w8r2 71.01 -> 64.05
    {QType::GGML_IQ4_NL, Epilogue::Store, 4096, 5120, 8, 1},    // w4r1 14.17 -> 14.05
    {QType::GGML_IQ4_XS, Epilogue::Store, 4096, 5120, 8, 2},    // w4r1 13.77 -> 13.35
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 6144, 8, 1},    // w4r1 18.77 -> 18.52
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 17408, 8, 1},   // w8r2 59.68 -> 53.32
    {QType::GGML_Q4_0, Epilogue::Store, 8192, 1024, 8, 1},      // w4r1 6.47 -> 6.31
    {QType::GGML_Q4_0, Epilogue::Store, 16384, 1024, 8, 1},     // w4r1 11.38 -> 11.20
    {QType::GGML_Q4_0, Epilogue::Store, 3712, 2688, 8, 1},      // w4r1 6.96 -> 6.91
    {QType::GGML_Q4_0, Epilogue::Store, 4224, 2816, 4, 2},      // w4r1 8.74 -> 8.67
    {QType::GGML_Q4_0, Epilogue::Store, 2688, 4096, 8, 1},      // w4r1 7.13 -> 7.03
    {QType::GGML_Q4_0, Epilogue::Store, 256, 5120, 8, 1},       // w4r1 2.76 -> 2.47
    {QType::GGML_Q4_0, Epilogue::Store, 1024, 5120, 8, 1},      // w4r1 5.10 -> 4.76
    {QType::GGML_Q4_0, Epilogue::Store, 1280, 5120, 8, 1},      // w4r1 5.82 -> 5.41
    {QType::GGML_Q4_0, Epilogue::Store, 1024, 5632, 8, 1},      // w4r1 5.56 -> 5.21
    {QType::GGML_Q4_0, Epilogue::Store, 1024, 8192, 8, 1},      // w8r2 7.12 -> 6.76
    {QType::GGML_Q4_0, Epilogue::Store, 2816, 8192, 8, 1},      // w8r2 15.33 -> 14.90
    {QType::GGML_Q4_0, Epilogue::Store, 5376, 8192, 8, 1},      // w8r2 27.68 -> 27.27
    {QType::GGML_Q4_0, Epilogue::Store, 1024, 10752, 8, 1},     // w8r2 8.89 -> 8.23
    {QType::GGML_Q4_0, Epilogue::Store, 1024, 16384, 8, 1},     // w8r2 13.16 -> 12.40
    {QType::GGML_Q4_K, Epilogue::Store, 256, 6656, 8, 1},       // w4r1 3.07 -> 2.88
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 6656, 8, 1},      // w4r1 5.87 -> 5.73
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 17408, 8, 1},     // w8r2 56.79 -> 56.37
    {QType::GGML_Q4_K, Epilogue::Store, 6656, 19968, 8, 1},     // w8r2 82.36 -> 81.72
    {QType::GGML_Q5_K, Epilogue::Store, 5120, 6144, 8, 1},      // w4r1 24.83 -> 24.59
    {QType::GGML_Q5_K, Epilogue::Store, 256, 6656, 8, 1},       // w4r1 4.10 -> 4.04
    {QType::GGML_Q5_K, Epilogue::Store, 5120, 17408, 8, 1},     // w8r2 68.78 -> 67.79
    {QType::GGML_Q6_K, Epilogue::Store, 1024, 5120, 8, 1},      // w4r1 6.91 -> 6.86
    {QType::GGML_Q6_K, Epilogue::Store, 1024, 6656, 8, 1},      // w4r1 8.19 -> 7.95
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 17408, 8, 1},     // w8r2 82.11 -> 80.95
    {QType::Q8_G32_FP16, Epilogue::Store, 3712, 2688, 8, 1},    // w4r1 12.63 -> 12.31
    {QType::Q8_G32_FP16, Epilogue::Store, 48, 5120, 8, 1},      // w4r1 2.49 -> 2.37
    {QType::Q8_G32_FP16, Epilogue::Store, 1024, 5120, 8, 1},    // w4r1 7.00 -> 6.88
};

// ad103:80sm: 52 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeAd103_80sm[] = {
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 2048, 4, 1}, // w4r2 8.41 -> 7.59
    {QType::GGML_IQ2_S, Epilogue::Store, 8192, 2048, 4, 1}, // w4r2 12.58 -> 12.19
    {QType::GGML_IQ2_S, Epilogue::Store, 12288, 2048, 4, 1}, // w8r2 17.92 -> 16.66
    {QType::GGML_IQ2_S, Epilogue::Store, 6656, 4096, 4, 1}, // w4r2 19.02 -> 16.78
    {QType::GGML_IQ2_S, Epilogue::Store, 12288, 5120, 4, 1}, // w8r2 38.08 -> 35.55
    {QType::GGML_IQ2_S, Epilogue::Store, 14336, 5120, 4, 1}, // w8r2 44.97 -> 41.23
    {QType::GGML_IQ2_S, Epilogue::Store, 16384, 5120, 4, 1}, // w8r2 51.56 -> 46.03
    {QType::GGML_IQ2_S, Epilogue::Store, 34816, 5120, 4, 1}, // w8r2 102.74 -> 94.40
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 5376, 4, 1}, // w4r2 18.76 -> 16.11
    {QType::GGML_IQ2_S, Epilogue::Store, 8192, 5376, 4, 1}, // w4r2 29.00 -> 28.26
    {QType::GGML_IQ2_S, Epilogue::Store, 16384, 5376, 4, 1}, // w8r2 57.34 -> 51.30
    {QType::GGML_IQ2_S, Epilogue::Store, 18432, 5376, 4, 1}, // w8r2 63.80 -> 57.68
    {QType::GGML_IQ2_S, Epilogue::Store, 43008, 5376, 4, 1}, // w8r2 140.13 -> 128.94
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 6656, 8, 2}, // w4r2 21.59 -> 19.97
    {QType::GGML_IQ2_S, Epilogue::Store, 19968, 6656, 4, 2}, // w8r2 78.11 -> 76.80
    {QType::GGML_IQ2_S, Epilogue::Store, 39936, 6656, 4, 2}, // w8r2 153.36 -> 144.67
    {QType::GGML_IQ2_S, Epilogue::Store, 5376, 8192, 4, 2}, // w8r2 32.31 -> 27.80
    {QType::GGML_IQ2_S, Epilogue::SwiGlu, 17408, 5120, 4, 1}, // w8r2 103.94 -> 88.71
    {QType::GGML_IQ2_S, Epilogue::SwiGlu, 19968, 6656, 4, 1}, // w8r2 153.24 -> 132.51
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 2048, 4, 1}, // w4r2 6.31 -> 6.24
    {QType::GGML_IQ2_XS, Epilogue::Store, 8192, 2048, 4, 1}, // w4r2 10.02 -> 9.80
    {QType::GGML_IQ2_XS, Epilogue::Store, 9216, 2048, 4, 1}, // w4r2 11.06 -> 10.66
    {QType::GGML_IQ2_XS, Epilogue::Store, 12288, 2048, 4, 1}, // w8r2 14.06 -> 13.67
    {QType::GGML_IQ2_XS, Epilogue::Store, 6144, 5120, 4, 1}, // w4r2 18.11 -> 16.32
    {QType::GGML_IQ2_XS, Epilogue::Store, 12288, 5120, 4, 1}, // w8r2 33.77 -> 29.51
    {QType::GGML_IQ2_XS, Epilogue::Store, 14336, 5120, 4, 1}, // w8r2 37.56 -> 34.81
    {QType::GGML_IQ2_XS, Epilogue::Store, 16384, 5120, 4, 1}, // w8r2 43.88 -> 39.44
    {QType::GGML_IQ2_XS, Epilogue::Store, 34816, 5120, 4, 1}, // w8r2 87.94 -> 82.33
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 5376, 4, 1}, // w4r2 14.17 -> 13.57
    {QType::GGML_IQ2_XS, Epilogue::Store, 8192, 5376, 4, 1}, // w4r2 24.68 -> 24.06
    {QType::GGML_IQ2_XS, Epilogue::Store, 16384, 5376, 4, 1}, // w8r2 47.92 -> 45.33
    {QType::GGML_IQ2_XS, Epilogue::Store, 18432, 5376, 4, 1}, // w8r2 51.99 -> 50.35
    {QType::GGML_IQ2_XS, Epilogue::Store, 43008, 5376, 4, 2}, // w8r2 115.88 -> 114.52
    {QType::GGML_IQ2_XS, Epilogue::Store, 19968, 6656, 4, 2}, // w8r2 66.59 -> 65.62
    {QType::GGML_IQ2_XS, Epilogue::SwiGlu, 17408, 5120, 4, 1}, // w8r2 81.94 -> 80.86
    {QType::GGML_IQ2_XS, Epilogue::SwiGlu, 19968, 6656, 4, 1}, // w8r2 125.15 -> 123.51
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4096, 2048, 4, 1}, // w4r2 5.65 -> 5.43
    {QType::GGML_IQ2_XXS, Epilogue::Store, 8192, 2048, 4, 1}, // w4r2 9.02 -> 8.89
    {QType::GGML_IQ2_XXS, Epilogue::Store, 9216, 2048, 4, 1}, // w4r2 10.24 -> 9.95
    {QType::GGML_IQ2_XXS, Epilogue::Store, 12288, 2048, 4, 2}, // w8r2 12.71 -> 12.34
    {QType::GGML_IQ2_XXS, Epilogue::Store, 6656, 4096, 4, 1}, // w4r2 13.58 -> 12.39
    {QType::GGML_IQ2_XXS, Epilogue::Store, 6144, 5120, 4, 1}, // w4r2 15.53 -> 14.57
    {QType::GGML_IQ2_XXS, Epilogue::Store, 12288, 5120, 4, 1}, // w8r2 28.31 -> 26.64
    {QType::GGML_IQ2_XXS, Epilogue::Store, 16384, 5120, 4, 2}, // w8r2 37.10 -> 35.51
    {QType::GGML_IQ2_XXS, Epilogue::Store, 34816, 5120, 4, 2}, // w8r2 73.06 -> 71.68
    {QType::GGML_IQ2_XXS, Epilogue::Store, 16384, 5376, 4, 2}, // w8r2 41.95 -> 40.77
    {QType::GGML_IQ2_XXS, Epilogue::Store, 5120, 6144, 4, 1}, // w4r2 16.62 -> 14.54
    {QType::GGML_IQ2_XXS, Epilogue::Store, 19968, 6656, 4, 2}, // w8r2 58.50 -> 57.22
    {QType::GGML_IQ2_XXS, Epilogue::Store, 39936, 6656, 4, 2}, // w8r2 111.34 -> 109.82
    {QType::GGML_IQ2_XXS, Epilogue::SwiGlu, 17408, 5120, 4, 1}, // w8r2 73.02 -> 70.50
    {QType::GGML_IQ2_XXS, Epilogue::SwiGlu, 19968, 6656, 4, 1}, // w8r2 111.84 -> 107.43
    {QType::GGML_IQ3_S, Epilogue::Store, 5376, 8192, 4, 2}, // w8r2 35.75 -> 32.22
};

// ad106:34sm: 26 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeAd106_34sm[] = {
    {QType::GGML_IQ2_S, Epilogue::Store, 6144, 5120, 4, 1}, // w4r2 45.47 -> 40.47
    {QType::GGML_IQ2_S, Epilogue::Store, 10240, 5120, 4, 1}, // w4r2 72.70 -> 64.34
    {QType::GGML_IQ2_S, Epilogue::Store, 17408, 5120, 4, 1}, // w8r2 121.44 -> 106.79
    {QType::GGML_IQ2_S, Epilogue::Store, 5120, 6144, 8, 1}, // w4r2 42.92 -> 40.77
    {QType::GGML_IQ2_S, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 119.21 -> 116.72
    {QType::GGML_IQ2_S, Epilogue::SwiGlu, 17408, 5120, 4, 1}, // w8r2 244.32 -> 208.96
    {QType::GGML_IQ2_XS, Epilogue::Store, 6144, 5120, 8, 1}, // w4r2 39.56 -> 34.90
    {QType::GGML_IQ2_XS, Epilogue::Store, 17408, 5120, 8, 1}, // w8r2 105.47 -> 93.93
    {QType::GGML_IQ2_XS, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 108.05 -> 101.00
    {QType::GGML_IQ2_XXS, Epilogue::Store, 6144, 5120, 8, 1}, // w4r2 32.69 -> 31.13
    {QType::GGML_IQ2_XXS, Epilogue::Store, 10240, 5120, 8, 1}, // w4r2 51.98 -> 50.33
    {QType::GGML_IQ2_XXS, Epilogue::Store, 17408, 5120, 8, 1}, // w8r2 85.69 -> 83.86
    {QType::GGML_IQ2_XXS, Epilogue::Store, 5120, 6144, 8, 1}, // w4r2 31.86 -> 31.25
    {QType::GGML_IQ2_XXS, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 92.70 -> 89.74
    {QType::GGML_IQ2_XXS, Epilogue::SwiGlu, 17408, 5120, 4, 1}, // w8r2 168.87 -> 166.74
    {QType::GGML_IQ3_S, Epilogue::Store, 1024, 5120, 8, 1}, // w4r1 10.02 -> 9.68
    {QType::GGML_IQ3_S, Epilogue::Store, 6144, 5120, 8, 1}, // w4r1 50.26 -> 49.94
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 6144, 8, 1}, // w4r1 50.41 -> 49.88
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 147.28 -> 143.02
    {QType::GGML_IQ3_XXS, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 130.94 -> 125.91
    {QType::GGML_IQ4_XS, Epilogue::Store, 1024, 5120, 8, 1}, // w4r1 10.78 -> 10.70
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 173.59 -> 171.52
    {QType::GGML_Q3_K, Epilogue::Store, 5120, 17408, 8, 1}, // w4r1 141.59 -> 140.73
    {QType::GGML_Q6_K, Epilogue::Store, 2048, 4096, 8, 1}, // w4r1 26.54 -> 26.37
    {QType::GGML_Q6_K, Epilogue::Store, 2816, 8192, 8, 1}, // w8r2 70.46 -> 69.78
    {QType::Q8_G32_FP16, Epilogue::Store, 48, 5120, 8, 1}, // w4r1 2.37 -> 2.18
};

// ga100:108sm: 53 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeGa100_108sm[] = {
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 17408, 8, 1},   // w8r2 81.94 -> 65.47
    {QType::GGML_IQ4_NL, Epilogue::Store, 10240, 5120, 8, 1},  // w4r1 36.63 -> 32.89
    {QType::GGML_IQ4_NL, Epilogue::Store, 17408, 5120, 8, 1},  // w4r1 60.54 -> 54.22
    {QType::GGML_IQ4_XS, Epilogue::Store, 6144, 5120, 8, 1},   // w4r1 24.80 -> 23.55
    {QType::GGML_IQ4_XS, Epilogue::Store, 10240, 5120, 8, 1},  // w4r1 39.88 -> 36.63
    {QType::GGML_IQ4_XS, Epilogue::Store, 17408, 5120, 8, 1},  // w4r1 65.28 -> 61.17
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 6144, 8, 1},   // w4r1 26.64 -> 23.69
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 17408, 8, 1},  // w8r2 70.52 -> 59.87
    {QType::GGML_IQ4_XS, Epilogue::SwiGlu, 17408, 5120, 4, 2}, // w4r1 114.11 -> 104.41
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 5120, 8, 1},     // w4r1 7.53 -> 7.15
    {QType::GGML_Q4_K, Epilogue::Store, 6144, 5120, 8, 1},     // w4r1 21.47 -> 19.76
    {QType::GGML_Q4_K, Epilogue::Store, 10240, 5120, 8, 1},    // w4r1 33.86 -> 29.75
    {QType::GGML_Q4_K, Epilogue::Store, 16384, 5120, 8, 1},    // w4r1 53.13 -> 46.68
    {QType::GGML_Q4_K, Epilogue::Store, 17408, 5120, 8, 1},    // w4r1 55.91 -> 48.66
    {QType::GGML_Q4_K, Epilogue::Store, 12288, 5376, 8, 1}, // w4r1 45.60 -> 39.22
    {QType::GGML_Q4_K, Epilogue::Store, 18432, 5376, 8, 1}, // w4r1 62.00 -> 53.01
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 6144, 8, 1},     // w4r1 21.74 -> 19.18
    {QType::GGML_Q4_K, Epilogue::Store, 5376, 8192, 8, 1}, // w8r2 27.44 -> 25.45
    {QType::GGML_Q4_K, Epilogue::Store, 5376, 16384, 8, 1}, // w8r2 50.81 -> 49.95
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 17408, 8, 1},    // w8r2 57.02 -> 50.35
    {QType::GGML_Q4_K, Epilogue::Store, 5376, 21504, 8, 1}, // w8r2 67.28 -> 66.58
    {QType::GGML_Q5_K, Epilogue::Store, 1024, 5120, 8, 1},     // w4r1 9.13 -> 8.87
    {QType::GGML_Q5_K, Epilogue::Store, 6144, 5120, 8, 2},     // w4r1 28.07 -> 25.67
    {QType::GGML_Q5_K, Epilogue::Store, 7168, 5120, 8, 2},     // w4r1 32.27 -> 31.42
    {QType::GGML_Q5_K, Epilogue::Store, 10240, 5120, 8, 2},    // w4r1 43.79 -> 39.34
    {QType::GGML_Q5_K, Epilogue::Store, 16384, 5120, 8, 2},    // w4r1 67.38 -> 60.36
    {QType::GGML_Q5_K, Epilogue::Store, 17408, 5120, 8, 2},    // w4r1 71.68 -> 65.35
    {QType::GGML_Q5_K, Epilogue::Store, 12288, 5376, 8, 2}, // w4r1 53.15 -> 47.63
    {QType::GGML_Q5_K, Epilogue::Store, 18432, 5376, 8, 2}, // w4r1 77.76 -> 69.10
    {QType::GGML_Q5_K, Epilogue::Store, 262144, 5376, 8, 2}, // w4r1 1063.37 -> 889.84
    {QType::GGML_Q5_K, Epilogue::Store, 5120, 6144, 8, 2},     // w4r1 29.42 -> 25.87
    {QType::GGML_Q5_K, Epilogue::SwiGlu, 17408, 5120, 8, 1},   // w4r1 116.50 -> 114.74
    {QType::GGML_Q5_K, Epilogue::SwiGlu, 21504, 5376, 8, 1}, // w4r1 145.43 -> 144.06
    {QType::GGML_Q6_K, Epilogue::Store, 6144, 5120, 8, 1},     // w4r1 33.57 -> 33.08
    {QType::GGML_Q6_K, Epilogue::Store, 10240, 5120, 8, 1},    // w4r1 51.69 -> 50.88
    {QType::GGML_Q6_K, Epilogue::Store, 17408, 5120, 8, 1},    // w4r1 83.20 -> 82.21
    {QType::GGML_Q6_K, Epilogue::Store, 248320, 5120, 8, 1},   // w4r1 1123.26 -> 1021.15
    {QType::GGML_Q6_K, Epilogue::Store, 4096, 5376, 8, 1}, // w4r1 25.12 -> 24.56
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 6144, 8, 1},     // w4r1 33.13 -> 32.22
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 17408, 8, 1},    // w8r2 82.72 -> 77.74
    {QType::GGML_Q6_K, Epilogue::Store, 5376, 21504, 8, 1}, // w8r2 114.41 -> 103.83
    {QType::Q8_G32_FP16, Epilogue::Store, 2048, 512, 8, 1},    // w4r1 4.22 -> 4.11
    {QType::Q8_G32_FP16, Epilogue::Store, 9216, 2048, 8, 1},   // w4r1 18.48 -> 18.11
    {QType::Q8_G32_FP16, Epilogue::Store, 12288, 2048, 8, 1},  // w4r1 23.48 -> 23.11
    {QType::Q8_G32_FP16, Epilogue::Store, 8192, 2816, 8, 2},   // w4r1 22.39 -> 22.25
    {QType::Q8_G32_FP16, Epilogue::Store, 9216, 2816, 8, 1},   // w4r1 24.78 -> 24.54
    {QType::Q8_G32_FP16, Epilogue::Store, 262144, 2816, 4, 2}, // w4r1 550.23 -> 547.05
    {QType::Q8_G32_FP16, Epilogue::Store, 2048, 4096, 8, 1},   // w4r1 11.21 -> 10.62
    {QType::Q8_G32_FP16, Epilogue::Store, 48, 5120, 8, 1},     // w4r1 4.49 -> 4.01
    {QType::Q8_G32_FP16, Epilogue::Store, 1024, 5120, 8, 1},   // w4r1 8.50 -> 7.90
    {QType::Q8_G32_FP16, Epilogue::Store, 5120, 6144, 8, 2},   // w4r1 31.93 -> 29.46
    {QType::Q8_G32_FP16, Epilogue::Store, 2816, 8192, 8, 2},   // w4r1 23.79 -> 22.78
    {QType::Q8_G32_FP16, Epilogue::SwiGlu, 2112, 2816, 8, 2},  // w4r1 14.13 -> 13.69
};

// ga102:72sm: 21 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeGa102_72sm[] = {
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 124.31 -> 112.93
    {QType::GGML_IQ4_NL, Epilogue::Store, 10240, 5120, 8, 1}, // w4r1 64.05 -> 63.11
    {QType::GGML_IQ4_NL, Epilogue::Store, 17408, 5120, 8, 1}, // w4r1 105.49 -> 104.87
    {QType::GGML_IQ4_NL, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 117.57 -> 111.16
    {QType::GGML_IQ4_XS, Epilogue::Store, 6144, 5120, 8, 1}, // w4r1 38.84 -> 38.11
    {QType::GGML_IQ4_XS, Epilogue::Store, 10240, 5120, 8, 1}, // w4r1 62.07 -> 61.61
    {QType::GGML_IQ4_XS, Epilogue::Store, 17408, 5120, 8, 1}, // w4r1 104.47 -> 100.98
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 6144, 8, 1}, // w4r1 39.93 -> 39.25
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 125.27 -> 108.46
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 118.10 -> 109.29
    {QType::GGML_Q5_K, Epilogue::Store, 1024, 5120, 8, 1}, // w4r1 10.53 -> 10.41
    {QType::GGML_Q5_K, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 143.60 -> 141.29
    {QType::GGML_Q6_K, Epilogue::Store, 248320, 2048, 8, 1}, // w4r1 881.20 -> 818.77
    {QType::GGML_Q6_K, Epilogue::Store, 248320, 5120, 4, 2}, // w4r1 2262.78 -> 2079.98
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 180.67 -> 162.58
    {QType::Q8_G32_FP16, Epilogue::Store, 2048, 512, 8, 1}, // w4r1 4.93 -> 4.89
    {QType::Q8_G32_FP16, Epilogue::Store, 12288, 2048, 8, 1}, // w4r1 53.15 -> 52.63
    {QType::Q8_G32_FP16, Epilogue::Store, 2048, 4096, 8, 1}, // w4r1 20.74 -> 19.83
    {QType::Q8_G32_FP16, Epilogue::Store, 48, 5120, 8, 1}, // w4r1 4.69 -> 4.47
    {QType::Q8_G32_FP16, Epilogue::Store, 1024, 5120, 8, 1}, // w4r1 14.61 -> 14.08
    {QType::Q8_G32_FP16, Epilogue::Store, 5120, 6144, 8, 2}, // w4r1 70.40 -> 68.64
};

// ga102:82sm: 34 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeGa102_82sm[] = {
    {QType::GGML_IQ4_NL, Epilogue::Store, 4096, 5120, 8, 1},  // w4r1 16.71 -> 15.94
    {QType::GGML_IQ4_NL, Epilogue::Store, 6144, 5120, 8, 1},  // w4r1 23.64 -> 23.01
    {QType::GGML_IQ4_NL, Epilogue::Store, 10240, 5120, 8, 1}, // w4r1 36.95 -> 36.63
    {QType::GGML_IQ4_NL, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 70.45 -> 63.74
    {QType::GGML_IQ4_XS, Epilogue::Store, 6144, 5120, 8, 1},  // w4r1 23.48 -> 22.87
    {QType::GGML_IQ4_XS, Epilogue::Store, 10240, 5120, 8, 1}, // w4r1 37.10 -> 35.75
    {QType::GGML_IQ4_XS, Epilogue::Store, 17408, 5120, 8, 1}, // w4r1 60.79 -> 57.98
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 6144, 8, 1},  // w4r1 24.41 -> 23.48
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 17408, 8, 1}, // w8r2 65.19 -> 63.93
    {QType::GGML_Q4_0, Epilogue::Store, 5376, 8192, 8, 1},    // w8r2 33.31 -> 31.80
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 5120, 8, 1},    // w4r1 6.35 -> 6.11
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 6144, 8, 1},    // w4r1 23.45 -> 23.26
    {QType::GGML_Q4_K, Epilogue::Store, 256, 6656, 8, 1},     // w4r1 4.81 -> 4.47
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 6656, 8, 1},    // w4r1 7.97 -> 7.85
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 17408, 8, 1},   // w8r2 65.66 -> 64.96
    {QType::GGML_Q4_K, Epilogue::Store, 6656, 19968, 8, 1},   // w8r2 94.62 -> 92.33
    {QType::GGML_Q5_K, Epilogue::Store, 1024, 5120, 8, 1},    // w4r1 7.66 -> 7.37
    {QType::GGML_Q5_K, Epilogue::Store, 7168, 5120, 8, 2},    // w4r1 35.07 -> 34.24
    {QType::GGML_Q5_K, Epilogue::Store, 5120, 6144, 8, 1},    // w4r1 29.71 -> 28.84
    {QType::GGML_Q6_K, Epilogue::Store, 1024, 5120, 8, 1},    // w4r1 9.30 -> 8.94
    {QType::GGML_Q6_K, Epilogue::Store, 6144, 5120, 8, 1},    // w4r1 36.57 -> 36.35
    {QType::GGML_Q6_K, Epilogue::Store, 10240, 5120, 8, 1},   // w4r1 56.88 -> 56.23
    {QType::GGML_Q6_K, Epilogue::Store, 17408, 5120, 8, 1},   // w4r1 91.70 -> 90.59
    {QType::GGML_Q6_K, Epilogue::Store, 248320, 5120, 8, 1},  // w4r1 1253.65 -> 1179.26
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 6144, 8, 1},    // w4r1 35.14 -> 34.54
    {QType::GGML_Q6_K, Epilogue::Store, 1024, 6656, 8, 1},    // w4r1 11.04 -> 10.02
    {QType::GGML_Q6_K, Epilogue::Store, 4096, 6656, 8, 1},    // w4r1 31.03 -> 30.02
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 17408, 8, 1},   // w8r2 100.04 -> 91.94
    {QType::GGML_Q6_K, Epilogue::Store, 6656, 19968, 8, 1},   // w8r2 149.71 -> 135.20
    {QType::Q8_G32_FP16, Epilogue::Store, 2048, 512, 8, 1},   // w4r1 3.38 -> 3.36
    {QType::Q8_G32_FP16, Epilogue::Store, 9216, 2048, 8, 1},  // w4r1 24.39 -> 24.23
    {QType::Q8_G32_FP16, Epilogue::Store, 48, 5120, 8, 1},    // w4r1 3.93 -> 3.75
    {QType::Q8_G32_FP16, Epilogue::Store, 1024, 5120, 8, 1},  // w4r1 9.42 -> 9.13
    {QType::Q8_G32_FP16, Epilogue::Store, 5120, 6144, 8, 1},  // w4r1 42.34 -> 40.82
};

// gb202:170sm: 116 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeGb202_170sm[] = {
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 2048, 4, 1},     // w4r2 5.47 -> 5.43
    {QType::GGML_IQ2_S, Epilogue::Store, 12288, 2048, 4, 1},    // w8r2 10.41 -> 9.92
    {QType::GGML_IQ2_S, Epilogue::Store, 6656, 4096, 8, 2},     // w4r2 10.89 -> 10.32
    {QType::GGML_IQ2_S, Epilogue::Store, 6144, 5120, 8, 2},     // w4r2 12.69 -> 11.86
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 5376, 8, 2},     // w4r2 10.96 -> 10.25
    {QType::GGML_IQ2_S, Epilogue::Store, 8192, 5376, 8, 2},     // w4r2 17.70 -> 17.34
    {QType::GGML_IQ2_S, Epilogue::Store, 5120, 6144, 8, 2},     // w4r2 12.68 -> 11.06
    {QType::GGML_IQ2_S, Epilogue::Store, 4096, 6656, 8, 2},     // w4r2 12.57 -> 11.65
    {QType::GGML_IQ2_S, Epilogue::Store, 4608, 6656, 8, 2},     // w4r2 13.67 -> 11.79
    {QType::GGML_IQ2_S, Epilogue::Store, 8704, 6656, 8, 2},     // w4r2 22.29 -> 21.71
    {QType::GGML_IQ2_XS, Epilogue::Store, 2048, 512, 8, 2},     // w4r1 3.38 -> 3.24
    {QType::GGML_IQ2_XS, Epilogue::Store, 8192, 2048, 8, 2},    // w4r2 6.90 -> 6.80
    {QType::GGML_IQ2_XS, Epilogue::Store, 6144, 5120, 8, 2},    // w4r2 10.81 -> 10.63
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 5376, 8, 2},    // w4r2 9.79 -> 9.49
    {QType::GGML_IQ2_XS, Epilogue::Store, 5120, 6144, 8, 2},    // w4r2 10.79 -> 9.96
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 6656, 4, 1},    // w4r2 10.80 -> 10.29
    {QType::GGML_IQ2_XS, Epilogue::Store, 4608, 6656, 8, 2},    // w4r2 11.76 -> 10.70
    {QType::GGML_IQ2_XXS, Epilogue::Store, 2048, 512, 8, 2},    // w4r1 2.75 -> 2.70
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4096, 2048, 4, 1},   // w4r2 4.25 -> 4.07
    {QType::GGML_IQ2_XXS, Epilogue::Store, 12288, 2048, 4, 2},  // w8r2 7.82 -> 7.56
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4096, 5376, 4, 1},   // w4r2 8.17 -> 7.80
    {QType::GGML_IQ2_XXS, Epilogue::Store, 8192, 5376, 8, 2},   // w4r2 12.33 -> 11.99
    {QType::GGML_IQ2_XXS, Epilogue::Store, 5120, 6144, 8, 2},   // w4r2 8.79 -> 8.22
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4096, 6656, 4, 1},   // w4r2 9.14 -> 8.75
    {QType::GGML_IQ2_XXS, Epilogue::Store, 4608, 6656, 8, 2},   // w4r2 9.39 -> 9.08
    {QType::GGML_IQ2_XXS, Epilogue::SwiGlu, 17408, 5120, 4, 1}, // w8r2 35.35 -> 34.46
    {QType::GGML_IQ3_S, Epilogue::Store, 8192, 2048, 4, 2},     // w4r1 7.61 -> 7.54
    {QType::GGML_IQ3_S, Epilogue::Store, 6656, 4096, 4, 2},     // w4r1 12.40 -> 10.44
    {QType::GGML_IQ3_S, Epilogue::Store, 6144, 5120, 8, 2},     // w4r1 14.84 -> 12.09
    {QType::GGML_IQ3_S, Epilogue::Store, 12288, 5120, 8, 2},    // w4r1 22.69 -> 21.41
    {QType::GGML_IQ3_S, Epilogue::Store, 14336, 5120, 8, 2},    // w4r1 24.51 -> 23.56
    {QType::GGML_IQ3_S, Epilogue::Store, 16384, 5120, 4, 2},    // w4r1 27.76 -> 27.13
    {QType::GGML_IQ3_S, Epilogue::Store, 34816, 5120, 4, 2},    // w4r1 55.77 -> 52.88
    {QType::GGML_IQ3_S, Epilogue::Store, 8192, 5376, 4, 2},     // w4r1 18.10 -> 17.17
    {QType::GGML_IQ3_S, Epilogue::Store, 16384, 5376, 4, 2},    // w4r1 31.26 -> 30.47
    {QType::GGML_IQ3_S, Epilogue::Store, 18432, 5376, 4, 2},    // w4r1 34.09 -> 33.15
    {QType::GGML_IQ3_S, Epilogue::Store, 43008, 5376, 8, 2},    // w4r1 78.56 -> 71.29
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 6144, 8, 2},     // w4r1 12.36 -> 11.54
    {QType::GGML_IQ3_S, Epilogue::Store, 4608, 6656, 8, 2},     // w4r1 12.37 -> 12.20
    {QType::GGML_IQ3_S, Epilogue::Store, 8704, 6656, 4, 2},     // w4r1 21.54 -> 20.97
    {QType::GGML_IQ3_S, Epilogue::Store, 19968, 6656, 8, 2},    // w4r1 42.90 -> 42.08
    {QType::GGML_IQ3_S, Epilogue::Store, 39936, 6656, 8, 2},    // w4r1 84.37 -> 76.89
    {QType::GGML_IQ3_S, Epilogue::Store, 5120, 17408, 8, 1},    // w8r2 28.82 -> 28.38
    {QType::GGML_IQ3_S, Epilogue::SwiGlu, 19968, 6656, 4, 2},   // w4r1 84.82 -> 80.40
    {QType::GGML_IQ3_XXS, Epilogue::Store, 12288, 2048, 4, 2},  // w4r1 8.74 -> 8.58
    {QType::GGML_IQ3_XXS, Epilogue::Store, 6656, 4096, 4, 2},   // w4r1 9.77 -> 9.46
    {QType::GGML_IQ3_XXS, Epilogue::Store, 6144, 5120, 4, 2},   // w4r1 12.09 -> 10.96
    {QType::GGML_IQ3_XXS, Epilogue::Store, 12288, 5120, 8, 2},  // w4r1 19.27 -> 18.97
    {QType::GGML_IQ3_XXS, Epilogue::Store, 14336, 5120, 4, 2},  // w4r1 21.34 -> 20.96
    {QType::GGML_IQ3_XXS, Epilogue::Store, 16384, 5120, 4, 2},  // w4r1 24.00 -> 23.52
    {QType::GGML_IQ3_XXS, Epilogue::Store, 34816, 5120, 4, 2},  // w4r1 47.03 -> 44.90
    {QType::GGML_IQ3_XXS, Epilogue::Store, 8192, 5376, 4, 2},   // w4r1 15.62 -> 15.37
    {QType::GGML_IQ3_XXS, Epilogue::Store, 16384, 5376, 4, 2},  // w4r1 28.25 -> 26.89
    {QType::GGML_IQ3_XXS, Epilogue::Store, 18432, 5376, 4, 2},  // w4r1 30.10 -> 28.59
    {QType::GGML_IQ3_XXS, Epilogue::Store, 43008, 5376, 4, 2},  // w4r1 66.55 -> 60.65
    {QType::GGML_IQ3_XXS, Epilogue::Store, 5120, 6144, 8, 2},   // w4r1 10.61 -> 10.50
    {QType::GGML_IQ3_XXS, Epilogue::Store, 8704, 6656, 4, 2},   // w4r1 18.62 -> 18.38
    {QType::GGML_IQ3_XXS, Epilogue::Store, 19968, 6656, 8, 2},  // w4r1 37.76 -> 37.50
    {QType::GGML_IQ3_XXS, Epilogue::Store, 39936, 6656, 4, 2},  // w4r1 75.29 -> 69.96
    {QType::GGML_IQ3_XXS, Epilogue::SwiGlu, 19968, 6656, 4, 2}, // w4r1 69.20 -> 67.03
    {QType::GGML_IQ4_NL, Epilogue::Store, 6144, 5120, 8, 2},    // w4r1 13.78 -> 13.31
    {QType::GGML_IQ4_NL, Epilogue::Store, 10240, 5120, 8, 1},   // w4r1 19.98 -> 19.59
    {QType::GGML_IQ4_NL, Epilogue::Store, 5120, 17408, 8, 1},   // w8r2 34.05 -> 33.46
    {QType::GGML_IQ4_XS, Epilogue::Store, 5120, 6144, 8, 2},    // w4r1 14.63 -> 12.86
    {QType::GGML_Q3_K, Epilogue::Store, 17408, 5120, 8, 1},     // w4r1 31.39 -> 30.69
    {QType::GGML_Q4_0, Epilogue::Store, 2816, 4096, 8, 1},      // w4r1 5.58 -> 5.35
    {QType::GGML_Q4_0, Epilogue::SwiGlu, 2112, 2816, 8, 1},     // w4r1 5.65 -> 5.47
    {QType::GGML_Q4_K, Epilogue::Store, 6656, 4096, 8, 1},      // w4r1 11.22 -> 10.91
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 5120, 8, 1},      // w4r1 4.66 -> 4.29
    {QType::GGML_Q4_K, Epilogue::Store, 4096, 5120, 8, 2},      // w4r1 10.55 -> 9.74
    {QType::GGML_Q4_K, Epilogue::Store, 6144, 5120, 8, 1},      // w4r1 14.15 -> 13.33
    {QType::GGML_Q4_K, Epilogue::Store, 10240, 5120, 8, 1},     // w4r1 20.86 -> 20.23
    {QType::GGML_Q4_K, Epilogue::Store, 16384, 5120, 8, 1},     // w4r1 31.50 -> 30.85
    {QType::GGML_Q4_K, Epilogue::Store, 17408, 5120, 8, 1},     // w4r1 33.41 -> 32.75
    {QType::GGML_Q4_K, Epilogue::Store, 12288, 5376, 8, 1},     // w4r1 25.10 -> 24.28
    {QType::GGML_Q4_K, Epilogue::Store, 18432, 5376, 8, 1},     // w4r1 35.92 -> 35.25
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 6144, 8, 1},      // w4r1 14.49 -> 13.80
    {QType::GGML_Q4_K, Epilogue::Store, 256, 6656, 8, 1},       // w4r1 3.01 -> 2.61
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 6656, 8, 1},      // w4r1 5.45 -> 5.00
    {QType::GGML_Q4_K, Epilogue::Store, 4096, 6656, 8, 2},      // w4r1 12.65 -> 11.59
    {QType::GGML_Q4_K, Epilogue::Store, 4608, 6656, 8, 2},      // w4r1 14.31 -> 12.89
    {QType::GGML_Q4_K, Epilogue::Store, 6656, 6656, 8, 1},      // w4r1 18.08 -> 17.54
    {QType::GGML_Q4_K, Epilogue::Store, 39936, 6656, 8, 1},     // w4r1 91.39 -> 90.87
    {QType::GGML_Q4_K, Epilogue::Store, 5376, 8192, 8, 1},      // w8r2 17.98 -> 17.85
    {QType::GGML_Q4_K, Epilogue::Store, 6656, 19968, 8, 1},     // w8r2 52.17 -> 50.52
    {QType::GGML_Q5_K, Epilogue::Store, 1024, 5120, 8, 1},      // w4r1 5.05 -> 4.80
    {QType::GGML_Q5_K, Epilogue::Store, 4096, 5120, 8, 2},      // w4r1 12.19 -> 11.15
    {QType::GGML_Q5_K, Epilogue::Store, 6144, 5120, 8, 1},      // w4r1 16.06 -> 15.65
    {QType::GGML_Q5_K, Epilogue::Store, 7168, 5120, 8, 1},      // w4r1 18.73 -> 17.77
    {QType::GGML_Q5_K, Epilogue::Store, 10240, 5120, 8, 1},     // w4r1 24.87 -> 24.19
    {QType::GGML_Q5_K, Epilogue::Store, 16384, 5120, 8, 1},     // w4r1 37.47 -> 36.89
    {QType::GGML_Q5_K, Epilogue::Store, 17408, 5120, 8, 1},     // w4r1 39.58 -> 39.17
    {QType::GGML_Q5_K, Epilogue::Store, 4096, 5376, 8, 1},      // w4r1 12.76 -> 11.46
    {QType::GGML_Q5_K, Epilogue::Store, 12288, 5376, 8, 1},     // w4r1 30.10 -> 29.75
    {QType::GGML_Q5_K, Epilogue::Store, 18432, 5376, 8, 1},     // w4r1 43.43 -> 43.01
    {QType::GGML_Q5_K, Epilogue::Store, 5120, 6144, 8, 2},      // w4r1 16.03 -> 15.38
    {QType::GGML_Q5_K, Epilogue::Store, 256, 6656, 8, 1},       // w4r1 3.17 -> 2.93
    {QType::GGML_Q5_K, Epilogue::Store, 5120, 17408, 8, 1},     // w8r2 41.97 -> 41.25
    {QType::GGML_Q5_K, Epilogue::Store, 5376, 21504, 8, 1},     // w8r2 53.61 -> 52.48
    {QType::GGML_Q6_K, Epilogue::Store, 1024, 5120, 8, 1},      // w4r1 5.89 -> 5.48
    {QType::GGML_Q6_K, Epilogue::Store, 6144, 5120, 8, 1},      // w4r1 19.91 -> 19.30
    {QType::GGML_Q6_K, Epilogue::Store, 10240, 5120, 8, 1},     // w4r1 29.73 -> 29.21
    {QType::GGML_Q6_K, Epilogue::Store, 17408, 5120, 8, 1},     // w4r1 48.62 -> 47.06
    {QType::GGML_Q6_K, Epilogue::Store, 248320, 5120, 8, 1},    // w4r1 627.34 -> 615.50
    {QType::GGML_Q6_K, Epilogue::Store, 4096, 5376, 8, 1},      // w4r1 14.66 -> 13.85
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 6144, 8, 1},      // w4r1 19.30 -> 18.81
    {QType::GGML_Q6_K, Epilogue::Store, 256, 6656, 8, 1},       // w4r1 3.25 -> 3.13
    {QType::GGML_Q6_K, Epilogue::Store, 202048, 6656, 8, 1},    // w4r1 680.27 -> 672.92
    {QType::GGML_Q6_K, Epilogue::Store, 5120, 17408, 8, 1},     // w8r2 50.89 -> 49.47
    {QType::GGML_Q6_K, Epilogue::SwiGlu, 17408, 5120, 8, 1},    // w4r1 91.21 -> 90.73
    {QType::Q8_G32_FP16, Epilogue::Store, 2048, 512, 8, 2},     // w4r1 2.08 -> 2.02
    {QType::Q8_G32_FP16, Epilogue::Store, 2048, 4096, 8, 1},    // w4r1 7.33 -> 7.13
    {QType::Q8_G32_FP16, Epilogue::Store, 2880, 4096, 8, 1},    // w4r1 9.48 -> 9.40
    {QType::Q8_G32_FP16, Epilogue::Store, 48, 5120, 8, 1},      // w4r1 2.52 -> 2.32
    {QType::Q8_G32_FP16, Epilogue::Store, 1024, 5120, 8, 1},    // w4r1 5.89 -> 5.56
    {QType::Q8_G32_FP16, Epilogue::Store, 5120, 6144, 8, 2},    // w4r1 23.03 -> 22.18
};

// gb206:36sm: 28 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeGb206_36sm[] = {
    {QType::GGML_IQ2_S, Epilogue::Store, 6144, 5120, 8, 1},   // w8r2 34.98 -> 34.58
    {QType::GGML_IQ2_S, Epilogue::Store, 10240, 5120, 8, 1},  // w8r2 56.38 -> 55.09
    {QType::GGML_IQ2_S, Epilogue::Store, 17408, 5120, 8, 1},  // w8r2 93.21 -> 91.45
    {QType::GGML_IQ2_S, Epilogue::Store, 5120, 6144, 8, 1},   // w8r2 34.38 -> 34.00
    {QType::GGML_IQ2_S, Epilogue::SwiGlu, 17408, 5120, 8, 1}, // w8r2 192.91 -> 171.46
    {QType::GGML_IQ2_XXS, Epilogue::Store, 5120, 6144, 8, 1}, // w8r2 25.61 -> 25.30
    {QType::GGML_IQ3_S, Epilogue::Store, 1024, 5120, 8, 1},   // w4r1 8.61 -> 7.85
    {QType::GGML_IQ3_S, Epilogue::Store, 6144, 5120, 8, 1},   // w8r2 37.88 -> 36.86
    {QType::GGML_IQ3_S, Epilogue::Store, 10240, 5120, 8, 1},  // w8r2 59.86 -> 57.40
    {QType::GGML_IQ3_S, Epilogue::Store, 13312, 5120, 8, 1},  // w8r2 75.68 -> 74.01
    {QType::GGML_IQ3_S, Epilogue::Store, 16384, 5120, 8, 1},  // w8r2 93.70 -> 89.51
    {QType::GGML_IQ3_S, Epilogue::Store, 17408, 5120, 8, 1},  // w8r2 99.34 -> 94.59
    {QType::GGML_IQ3_XXS, Epilogue::Store, 6144, 5120, 8, 2}, // w4r2 33.67 -> 32.81
    {QType::GGML_IQ3_XXS, Epilogue::Store, 5120, 6144, 8, 2}, // w4r2 33.15 -> 32.64
    {QType::GGML_IQ4_XS, Epilogue::Store, 1024, 5120, 4, 2},  // w4r1 9.84 -> 9.17
    {QType::GGML_Q3_K, Epilogue::Store, 5120, 17408, 8, 1},   // w4r1 104.56 -> 99.57
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 5120, 8, 2},    // w4r1 10.37 -> 9.49
    {QType::GGML_Q4_K, Epilogue::Store, 17408, 5120, 8, 1},   // w4r1 120.41 -> 119.77
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 6144, 8, 1},    // w4r1 44.32 -> 43.71
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 17408, 8, 1},   // w8r2 126.69 -> 123.04
    {QType::GGML_Q5_K, Epilogue::Store, 1024, 5120, 8, 1},    // w4r1 11.81 -> 10.93
    {QType::GGML_Q6_K, Epilogue::Store, 8192, 2816, 8, 1},    // w4r1 48.11 -> 47.61
    {QType::GGML_Q6_K, Epilogue::Store, 9216, 2816, 8, 1},    // w4r1 53.96 -> 53.51
    {QType::GGML_Q6_K, Epilogue::Store, 2048, 4096, 8, 1},    // w4r1 21.50 -> 20.28
    {QType::GGML_Q6_K, Epilogue::Store, 2816, 4096, 4, 2},    // w4r1 27.50 -> 26.40
    {QType::GGML_Q6_K, Epilogue::Store, 2816, 8192, 4, 1},    // w8r2 48.92 -> 48.16
    {QType::Q8_G32_FP16, Epilogue::Store, 48, 5120, 8, 1},    // w4r1 2.57 -> 2.39
    {QType::Q8_G32_FP16, Epilogue::Store, 2816, 8192, 8, 2},  // w4r1 59.62 -> 59.15
};

// tu102:68sm: 12 entries (median us: rules -> entry).
inline constexpr R8DecodeEntry kR8DecodeTu102_68sm[] = {
    {QType::GGML_IQ2_S, Epilogue::Store, 5120, 6144, 8, 1}, // w8r2 37.22 -> 35.00
    {QType::GGML_IQ2_XS, Epilogue::Store, 4096, 5120, 8, 1}, // w8r2 22.53 -> 22.10
    {QType::GGML_IQ3_S, Epilogue::Store, 1024, 5120, 8, 1}, // w8r2 10.27 -> 9.37
    {QType::GGML_IQ3_XXS, Epilogue::Store, 1024, 5120, 8, 1}, // w8r2 8.43 -> 8.06
    {QType::GGML_IQ4_XS, Epilogue::Store, 1024, 5120, 8, 1}, // w4r1 9.73 -> 8.94
    {QType::GGML_IQ4_XS, Epilogue::Store, 17408, 5120, 8, 1}, // w4r1 87.28 -> 86.26
    {QType::GGML_Q4_K, Epilogue::Store, 5120, 2560, 8, 1}, // w4r1 17.87 -> 17.55
    {QType::GGML_Q4_K, Epilogue::Store, 2560, 4096, 8, 1}, // w4r1 14.86 -> 13.95
    {QType::GGML_Q4_K, Epilogue::Store, 1024, 5120, 8, 1}, // w4r1 9.44 -> 8.66
    {QType::GGML_Q4_K, Epilogue::Store, 2560, 9216, 4, 2}, // w8r2 32.33 -> 29.63
    {QType::GGML_Q6_K, Epilogue::Store, 1024, 2560, 4, 2}, // w4r1 8.63 -> 8.40
    {QType::GGML_Q6_K, Epilogue::Store, 2560, 9216, 4, 1}, // w4r2 47.41 -> 45.42
};

inline constexpr std::array<R8DecodeTable, 9> kR8DecodeTables{{
    {"ad102:128sm", {8, 9, 128, 72, 1008}, kR8DecodeAd102_128sm},
    {"ad103:80sm", {8, 9, 80, 64, 736}, kR8DecodeAd103_80sm},
    {"ad106:34sm", {8, 9, 34, 32, 288}, kR8DecodeAd106_34sm},
    {"ga100:108sm", {8, 0, 108, 40, 1555}, kR8DecodeGa100_108sm},
    {"ga102:72sm", {8, 6, 72, 6, 600}, kR8DecodeGa102_72sm},
    {"ga102:82sm", {8, 6, 82, 6, 936}, kR8DecodeGa102_82sm},
    {"gb202:170sm", {12, 0, 170, 96, 1792}, kR8DecodeGb202_170sm},
    {"gb206:36sm", {12, 0, 36, 32, 448}, kR8DecodeGb206_36sm},
    {"tu102:68sm", {7, 5, 68, 5.5, 616}, kR8DecodeTu102_68sm},
}};

} // namespace ninferno::ops::detail::generic
