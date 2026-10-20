// A kernel compiled for sm_90 SASS only (CUDA_ARCHITECTURES 90-real in
// tests/cmake/CoreTests.cmake), standing for a library of the build compiled for another
// architecture.
#include <cuda_runtime.h>

namespace ninferno::test {
namespace {
__global__ void sm90_only_probe_kernel() {}
} // namespace

const void* sm90_only_kernel() { return reinterpret_cast<const void*>(sm90_only_probe_kernel); }

} // namespace ninferno::test
