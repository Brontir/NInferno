#pragma once
#include "ops/softmax_attention/common/causal_operands.h"

namespace ninferno::ops::detail {
template <bool Writable>
using Fp8KvCacheView = QuantizedCausalCacheView<std::uint8_t, __half, __half, Writable>;
using Fp8KvReadView  = Fp8KvCacheView<false>;

} // namespace ninferno::ops::detail
