// DeviceBuffer::fill is complete when it returns.
//
// Every NInferno stream is cudaStreamNonBlocking, so it does not order against the legacy default
// stream that cudaMemset uses, and cudaMemset of device memory may return before the fill runs.
// A buffer filled and then written by the next operation on a non-blocking stream could have the
// fill land on top of that write. This happened: a replay stress test intermittently read an
// eager Op output still holding its poison fill (1 of 8 processes locally, 3 of 4 before).
//
// The case: fill a 512 MiB buffer, then at once overwrite its last 4 KiB on a non-blocking stream,
// 40 times; the tail must always hold the second write. A control arm issues the same raw
// cudaMemset without settling the default stream and only reports how often its fill won the
// race on this device (it shows the case can see the defect; it is not asserted, since a race
// cannot be forced).

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kBytes     = std::size_t{512} << 20;
constexpr std::size_t kTailBytes = 4096;
constexpr int kRounds            = 40;

bool check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

// Rounds whose tail did not end as the second write (0x55).
int lost_writes(ninferno::DeviceBuffer& buffer, cudaStream_t stream, bool raw_fill) {
    auto* tail = static_cast<std::uint8_t*>(buffer.p) + kBytes - kTailBytes;
    std::vector<std::uint8_t> host(kTailBytes);
    int lost = 0;
    for (int round = 0; round < kRounds; ++round) {
        if (raw_fill) {
            if (!check(cudaMemset(buffer.p, 0xAA, kBytes), "raw cudaMemset")) { return -1; }
        } else {
            buffer.fill(0xAA);
        }
        if (!check(cudaMemsetAsync(tail, 0x55, kTailBytes, stream), "tail write") ||
            !check(cudaStreamSynchronize(stream), "tail sync") ||
            !check(cudaDeviceSynchronize(), "device sync") ||
            !check(cudaMemcpy(host.data(), tail, kTailBytes, cudaMemcpyDeviceToHost), "read tail")) {
            return -1;
        }
        for (const std::uint8_t byte : host) {
            if (byte != 0x55) {
                ++lost;
                break;
            }
        }
    }
    return lost;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "skip: no CUDA device\n";
        return 77;
    }
    ninferno::DeviceBuffer buffer(kBytes);
    cudaStream_t stream = nullptr;
    if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream")) { return 1; }
    const int control = lost_writes(buffer, stream, true);
    const int settled = lost_writes(buffer, stream, false);
    (void)cudaStreamDestroy(stream);
    std::cout << "control (raw legacy cudaMemset, not settled): the fill landed after the "
                 "next write in "
              << control << " of " << kRounds << " rounds (informational)\n";
    std::cout << "DeviceBuffer::fill: the fill landed after the next write in " << settled
              << " of " << kRounds << " rounds (must be 0)\n";
    if (control < 0 || settled != 0) {
        std::cerr << "FAIL: DeviceBuffer::fill is not complete when it returns\n";
        return 1;
    }
    std::cout << "device buffer fill order holds\n";
    return 0;
}
