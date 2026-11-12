// One block format's forced launches for tools/bench/mmq_launch_microbench.cu, in its own unit so
// the heavy template builds in parallel. BENCH ONLY.
#include "mmq_launch_microbench_launch.cuh"

namespace ninferno::bench::mmq_launch {
const FormatOps kQ4K = impl::ops<GGML_TYPE_Q4_K>("q4_K", ggml_mmq::Type::Q4_K);
} // namespace ninferno::bench::mmq_launch
