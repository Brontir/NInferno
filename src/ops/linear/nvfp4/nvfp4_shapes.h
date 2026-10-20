#pragma once
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace ninferno::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
};

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
// Gemma 4 26B-A4B.
extern const Nvfp4LinearShape kNvfp4N8192K2816;
extern const Nvfp4LinearShape kNvfp4N1024K2816;
extern const Nvfp4LinearShape kNvfp4N2816K4096;
extern const Nvfp4LinearShape kNvfp4N2816K8192;
extern const Nvfp4LinearShape kNvfp4N2816K2112;
// Gemma 4 31B.
extern const Nvfp4LinearShape kNvfp4N16384K5376;
extern const Nvfp4LinearShape kNvfp4N18432K5376;
extern const Nvfp4LinearShape kNvfp4N2048K5376;
extern const Nvfp4LinearShape kNvfp4N43008K5376;
extern const Nvfp4LinearShape kNvfp4N5376K8192;
extern const Nvfp4LinearShape kNvfp4N5376K16384;
extern const Nvfp4LinearShape kNvfp4N5376K21504;
} // namespace ninferno::ops::detail
