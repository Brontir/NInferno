#pragma once

// The enumerations of the Engine options that the device, op and tuner layers read too
// (ninferno/types.h: DecodePacingOptions, MaintainerOptions), with their option spellings. Kept
// apart so those layers, CUDA sources among them, need not include the whole Engine API.

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno {

// How the host waits for a decode round's end (--decode-wait): Auto spins, and blocks while the
// host stalls where a probe found that faster; Spin never blocks; Block blocks on every round.
enum class DecodeWait : std::uint8_t { Auto, Spin, Block };

// The CUDA synchronization schedule of the device context (--cuda-sync, cudaSetDeviceFlags).
enum class CudaSyncSchedule : std::uint8_t { Spin, Blocking, Yield, Auto };

// A lab tuner's mode (--dense-tune, --moe-tune): Off reads the release's measurements and times
// nothing; On times what they do not cover; Retune times everything again.
enum class TuneMode : std::uint8_t { Off, On, Retune };

// How artifact payloads (the weights) are read at load (--artifact-io): Auto reads local disks with
// O_DIRECT (a 20+ GB payload kept out of the page cache) and network, FUSE and host-shared file
// systems through the page cache; Direct and Buffered force one. The bytes read are the same.
enum class ArtifactIo : std::uint8_t { Auto, Direct, Buffered };

// The most ordinary decode steps one round runs back to back (--decode-steps).
inline constexpr std::uint32_t kMaximumDecodeSteps = 32;

[[nodiscard]] constexpr const char* decode_wait_name(DecodeWait wait) noexcept {
    switch (wait) {
    case DecodeWait::Auto:
        return "auto";
    case DecodeWait::Spin:
        return "spin";
    case DecodeWait::Block:
        return "block";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char* cuda_sync_schedule_name(CudaSyncSchedule schedule) noexcept {
    switch (schedule) {
    case CudaSyncSchedule::Spin:
        return "spin";
    case CudaSyncSchedule::Blocking:
        return "blocking";
    case CudaSyncSchedule::Yield:
        return "yield";
    case CudaSyncSchedule::Auto:
        return "auto";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char* tune_mode_name(TuneMode mode) noexcept {
    switch (mode) {
    case TuneMode::Off:
        return "off";
    case TuneMode::On:
        return "on";
    case TuneMode::Retune:
        return "retune";
    }
    return "unknown";
}

// --decode-wait auto|spin|block; anything else is refused naming the option.
[[nodiscard]] inline DecodeWait parse_decode_wait(std::string_view text) {
    for (const DecodeWait wait : {DecodeWait::Auto, DecodeWait::Spin, DecodeWait::Block}) {
        if (text == decode_wait_name(wait)) { return wait; }
    }
    throw std::invalid_argument("--decode-wait must be auto, spin or block, not '" +
                                std::string(text) + "'");
}

// --decode-steps auto|N: 0 for auto, else N in [1, kMaximumDecodeSteps].
[[nodiscard]] inline std::uint32_t parse_decode_steps(std::string_view text) {
    if (text == "auto") { return 0; }
    std::uint32_t value = 0;
    bool digits         = !text.empty() && text.size() <= 2;
    for (const char c : text) {
        if (c < '0' || c > '9') {
            digits = false;
            break;
        }
        value = value * 10 + static_cast<std::uint32_t>(c - '0');
    }
    if (!digits || value == 0 || value > kMaximumDecodeSteps) {
        throw std::invalid_argument("--decode-steps must be auto or an integer in [1, " +
                                    std::to_string(kMaximumDecodeSteps) + "], not '" +
                                    std::string(text) + "'");
    }
    return value;
}

// --cuda-sync spin|blocking|yield|auto.
[[nodiscard]] inline CudaSyncSchedule parse_cuda_sync_schedule(std::string_view text) {
    for (const CudaSyncSchedule schedule : {CudaSyncSchedule::Spin, CudaSyncSchedule::Blocking,
                                            CudaSyncSchedule::Yield, CudaSyncSchedule::Auto}) {
        if (text == cuda_sync_schedule_name(schedule)) { return schedule; }
    }
    throw std::invalid_argument("--cuda-sync must be spin, blocking, yield or auto, not '" +
                                std::string(text) + "'");
}

[[nodiscard]] constexpr const char* artifact_io_name(ArtifactIo io) noexcept {
    switch (io) {
    case ArtifactIo::Auto:
        return "auto";
    case ArtifactIo::Direct:
        return "direct";
    case ArtifactIo::Buffered:
        return "buffered";
    }
    return "auto";
}

// --artifact-io auto|direct|buffered.
[[nodiscard]] inline ArtifactIo parse_artifact_io(std::string_view text) {
    for (const ArtifactIo io : {ArtifactIo::Auto, ArtifactIo::Direct, ArtifactIo::Buffered}) {
        if (text == artifact_io_name(io)) { return io; }
    }
    throw std::invalid_argument("--artifact-io must be auto, direct or buffered, not '" +
                                std::string(text) + "'");
}

// --dense-tune / --moe-tune off|on|retune (`flag` names the option in the refusal).
[[nodiscard]] inline TuneMode parse_tune_mode(std::string_view text, std::string_view flag) {
    for (const TuneMode mode : {TuneMode::Off, TuneMode::On, TuneMode::Retune}) {
        if (text == tune_mode_name(mode)) { return mode; }
    }
    throw std::invalid_argument(std::string(flag) + " must be off, on or retune, not '" +
                                std::string(text) + "'");
}

} // namespace ninferno
