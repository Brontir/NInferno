#pragma once
#include "ops/linear/common/output.cuh"

namespace ninferno::ops::detail {
using Nvfp4GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;
} // namespace ninferno::ops::detail
