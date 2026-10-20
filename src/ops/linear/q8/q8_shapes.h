#pragma once

#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_launch.h"

namespace ninferno::ops::detail {

using Q8N1024K2048   = Q8LinearGeometry<1024, 2048>;
using Q8N1024K5120   = Q8LinearGeometry<1024, 5120>;
using Q8N2048K4096   = Q8LinearGeometry<2048, 4096>;
using Q8N2048K4608   = Q8LinearGeometry<2048, 4608>;
using Q8N2048K16384  = Q8LinearGeometry<2048, 16384>;
using Q8N4608K4608   = Q8LinearGeometry<4608, 4608>;
using Q8N5120K4608   = Q8LinearGeometry<5120, 4608>;
using Q8N5120K6144   = Q8LinearGeometry<5120, 6144>;
using Q8N5120K10240  = Q8LinearGeometry<5120, 10240>;
using Q8N5120K17408  = Q8LinearGeometry<5120, 17408>;
using Q8N5120K25600  = Q8LinearGeometry<5120, 25600>;
using Q8N6144K5120   = Q8LinearGeometry<6144, 5120>;
using Q8N9216K2048   = Q8LinearGeometry<9216, 2048>;
using Q8N12288K2048  = Q8LinearGeometry<12288, 2048>;
using Q8N14336K5120  = Q8LinearGeometry<14336, 5120>;
using Q8N34816K5120  = Q8LinearGeometry<34816, 5120>;
using Q8N248320K5120 = Q8LinearGeometry<248320, 5120>;
// Gemma 4 26B-A4B and 31B.
using Q8N4224K2816  = Q8LinearGeometry<4224, 2816>;
using Q8N8192K2816  = Q8LinearGeometry<8192, 2816>;
using Q8N9216K2816  = Q8LinearGeometry<9216, 2816>;
using Q8N2816K4096  = Q8LinearGeometry<2816, 4096>;
using Q8N2816K8192  = Q8LinearGeometry<2816, 8192>;
using Q8N2816K2112  = Q8LinearGeometry<2816, 2112>;
using Q8N16384K5376 = Q8LinearGeometry<16384, 5376>;
using Q8N18432K5376 = Q8LinearGeometry<18432, 5376>;
using Q8N5376K8192  = Q8LinearGeometry<5376, 8192>;
using Q8N5376K16384 = Q8LinearGeometry<5376, 16384>;
// Gemma 4 MTP drafters (26B-A4B and 31B) and the shortlisted drafter heads.
using Q8N1024K5632   = Q8LinearGeometry<1024, 5632>;
using Q8N4096K1024   = Q8LinearGeometry<4096, 1024>;
using Q8N8192K1024   = Q8LinearGeometry<8192, 1024>;
using Q8N1024K4096   = Q8LinearGeometry<1024, 4096>;
using Q8N1024K8192   = Q8LinearGeometry<1024, 8192>;
using Q8N16384K1024  = Q8LinearGeometry<16384, 1024>;
using Q8N2816K1024   = Q8LinearGeometry<2816, 1024>;
using Q8N262144K1024 = Q8LinearGeometry<262144, 1024>;
using Q8N1024K10752  = Q8LinearGeometry<1024, 10752>;
using Q8N1024K16384  = Q8LinearGeometry<1024, 16384>;
using Q8N5376K1024   = Q8LinearGeometry<5376, 1024>;
using Q8N32768K1024  = Q8LinearGeometry<32768, 1024>;
using Q8N65536K1024  = Q8LinearGeometry<65536, 1024>;
using Q8N98304K1024  = Q8LinearGeometry<98304, 1024>;
using Q8N131072K1024 = Q8LinearGeometry<131072, 1024>;
// z-lab DFlash drafters for Gemma 4 26B-A4B and 31B.
using Q8N5376K10752 = Q8LinearGeometry<5376, 10752>;
using Q8N21504K5376 = Q8LinearGeometry<21504, 5376>;
using Q8N10240K5376 = Q8LinearGeometry<10240, 5376>;
using Q8N5376K32256 = Q8LinearGeometry<5376, 32256>;
using Q8N2816K5632  = Q8LinearGeometry<2816, 5632>;
using Q8N11264K2816 = Q8LinearGeometry<11264, 2816>;
using Q8N6144K2816  = Q8LinearGeometry<6144, 2816>;
using Q8N10240K2816 = Q8LinearGeometry<10240, 2816>;
using Q8N2816K16896 = Q8LinearGeometry<2816, 16896>;

// The dispatcher admits N/K and positive T; each shape owns its complete T selection.
[[nodiscard]] Q8Launch select_q8_n1024_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k4096(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k16384(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n4608_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k10240(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k25600(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n9216_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n12288_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n4224_k2816(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n8192_k2816(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n9216_k2816(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2816_k4096(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2816_k8192(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2816_k2112(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n16384_k5376(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n18432_k5376(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5376_k8192(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5376_k16384(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k5632(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n4096_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n8192_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k4096(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k8192(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n16384_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2816_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n262144_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k10752(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k16384(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5376_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n32768_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n65536_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n98304_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n131072_k1024(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5376_k10752(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n21504_k5376(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n10240_k5376(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5376_k32256(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2816_k5632(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n11264_k2816(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n6144_k2816(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n10240_k2816(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2816_k16896(std::int32_t tokens);

} // namespace ninferno::ops::detail
