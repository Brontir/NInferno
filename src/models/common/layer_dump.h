#pragma once

// The model targets' debug layer dump: <VARIABLE>=<dir> turns it on, <VARIABLE>_ROWS=lo:hi keeps
// the token columns at absolute positions in [lo, hi) (default all). Writes synchronize the device:
// for localizing numerics between two paths, never for timing.

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models {

struct LayerDump {
    std::string dir;
    long long lo = 0;
    long long hi = std::numeric_limits<long long>::max();
};

// The dump `variable` asks for; nullopt when it is unset or empty.
[[nodiscard]] inline std::optional<LayerDump> read_layer_dump(const std::string& variable) {
    const char* dir = std::getenv(variable.c_str());
    if (dir == nullptr || *dir == 0) { return std::nullopt; }
    LayerDump out{.dir = dir};
    const std::string rows_variable = variable + "_ROWS";
    if (const char* rows = std::getenv(rows_variable.c_str())) {
        if (std::sscanf(rows, "%lld:%lld", &out.lo, &out.hi) != 2) {
            throw std::invalid_argument(rows_variable + " must be lo:hi");
        }
    }
    return out;
}

// A pass dumps when the dump is on and its stream is not being captured into a graph.
[[nodiscard]] inline bool dumps_on(const LayerDump* dump, cudaStream_t stream) {
    if (dump == nullptr) { return false; }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status == cudaStreamCaptureStatusNone;
}

// `bytes` of device memory written to `path` (appended with `append`), after the device finishes.
inline void write_device_bytes(const void* device, std::size_t bytes, const std::string& path,
                               bool append) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<char> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    std::FILE* file = std::fopen(path.c_str(), append ? "ab" : "wb");
    if (file == nullptr) { throw std::runtime_error("cannot write " + path); }
    std::fwrite(host.data(), 1, host.size(), file);
    std::fclose(file);
}

// The columns of a token-major device buffer (`column_bytes` per token, `t` tokens from absolute
// position p0) that lie in the dump's rows, written to `path`; nothing when none does.
inline void write_device_rows(const LayerDump& dump, const void* device, std::size_t column_bytes,
                              std::int32_t t, long long p0, const std::string& path, bool append) {
    const long long lo = std::max(dump.lo, p0);
    const long long hi = std::min(dump.hi, p0 + t);
    if (lo >= hi) { return; }
    write_device_bytes(static_cast<const char*>(device) + (lo - p0) * column_bytes,
                       static_cast<std::size_t>(hi - lo) * column_bytes, path, append);
}

} // namespace ninferno::models
