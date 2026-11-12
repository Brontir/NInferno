// One block format's forced launches for tools/bench/mmq_launch_microbench.cu, in its own unit so
// the heavy template builds in parallel. BENCH ONLY.
#include "mmq_launch_microbench_launch.cuh"

namespace ninferno::bench::mmq_launch {
const FormatOps kQ5K = impl::ops<GGML_TYPE_Q5_K>("q5_K", ggml_mmq::Type::Q5_K);
} // namespace ninferno::bench::mmq_launch
