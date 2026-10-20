#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_prefill_small_t.cuh"

namespace ninferno::ops::detail {
namespace {

using Output   = LinearBf16SegmentedOutput<8192, 4096>;
using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;

} // namespace

void q8_gdn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                stream);
}

void q8_gdn_input_mma_prefill_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                             Tensor& z, cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    // Swept over 17..128 columns: the 32-row tile leads through 96, where the wide route takes
    // over.
    q8_prefill_small_t_dispatch(Q8PrefillSmallTile::R32C32, [&]<class Small>() {
        static_assert((8192 % Small::kBlockRows) == 0 && (4096 % Small::kBlockRows) == 0);
        launch_q8_a16_mma<Small>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                 stream);
    });
}
} // namespace ninferno::ops::detail
