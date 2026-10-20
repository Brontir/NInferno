#include "core/tune_timer.h"

#include "core/device.h"
#include "core/device_class.h"

#include <algorithm>
#include <cstdint>
#include <exception>

namespace ninferno {
namespace {

__global__ void read_flush_kernel(const uint4* p, std::size_t n, unsigned* never) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // The buffer is zeros: the store never happens, but the loads cannot be dropped.
    if (acc == 0x9e3779b9U && never != nullptr) { *never = acc; }
}

} // namespace

TuneTimer::TuneTimer(cudaStream_t stream, int warm, int reps)
    : stream_(stream), warm_(warm), reps_(reps), start_(static_cast<std::size_t>(warm + reps)),
      stop_(static_cast<std::size_t>(warm + reps)) {
    try {
        const std::size_t want = std::min<std::size_t>(2 * device_class().l2_bytes, 256u << 20);
        if (want > 0) {
            flush_ = DeviceBuffer(want);
            flush_.fill(0);
        }
    } catch (const std::exception&) {
        flush_ = DeviceBuffer();
        (void)cudaGetLastError();
    }
    for (std::size_t r = 0; r < start_.size(); ++r) {
        CUDA_CHECK(cudaEventCreate(&start_[r]));
        CUDA_CHECK(cudaEventCreate(&stop_[r]));
    }
}

TuneTimer::~TuneTimer() {
    for (std::size_t r = 0; r < start_.size(); ++r) {
        if (start_[r] != nullptr) { (void)cudaEventDestroy(start_[r]); }
        if (stop_[r] != nullptr) { (void)cudaEventDestroy(stop_[r]); }
    }
}

void TuneTimer::flush() {
    if (flush_.bytes == 0) { return; }
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    read_flush_kernel<<<sms * 4, 512, 0, stream_>>>(static_cast<const uint4*>(flush_.p),
                                                    flush_.bytes / 16, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void TuneTimer::record(cudaEvent_t event) { CUDA_CHECK(cudaEventRecord(event, stream_)); }

float TuneTimer::median() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::vector<float> us(static_cast<std::size_t>(reps_));
    for (int r = 0; r < reps_; ++r) {
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_[static_cast<std::size_t>(warm_ + r)],
                                        stop_[static_cast<std::size_t>(warm_ + r)]));
        us[static_cast<std::size_t>(r)] = ms * 1000.0F;
    }
    std::nth_element(us.begin(), us.begin() + reps_ / 2, us.end());
    return us[static_cast<std::size_t>(reps_ / 2)];
}

} // namespace ninferno
