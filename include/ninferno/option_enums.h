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

// A maintainer's pin of the kernel family that computes the projections of prefill width (17
// tokens or more, --prefill-route) or of decode and verify width (1..16 tokens, --decode-route),
// for A/B runs that check the automatic choice: Auto is the release's selection, any other value
// takes that family for every call it can compute (docs/maintainer/environment.md, Route pins).
// A16: BF16 activations (the register MMA, tiled or staged tiles); W4a8: the int8-activation
// W4A8 tiles; I8: the Q4/Q5/Q6 G64 int8 tiles; Mmq: ggml's int8 matrix kernel; W4a4: NVFP4
// weights on NVFP4 activations; R8: the int8 decode kernel (1..4 tokens); Vec: ggml's dp4a vector
// kernel (1..8 tokens, 16 without tensor mma.sync).
enum class PrefillRoute : std::uint8_t { Auto, A16, W4a8, I8, Mmq, W4a4 };
enum class DecodeRoute : std::uint8_t { Auto, A16, R8, Vec, Mmq };

// A maintainer's pin of the routed experts' kernels (--moe-path): Auto is the release's choice,
// Fused the family's fused MoE kernels (Qwen 3.5/3.6 SparseMoe, Gemma 4 NVFP4 experts), Generic
// the generic MoE Op (ops::moe) that reads every expert format.
enum class MoePath : std::uint8_t { Auto, Fused, Generic };

// How artifact payloads (the weights) are read at load (--artifact-io): Auto reads local disks with
// O_DIRECT (a 20+ GB payload kept out of the page cache) and network, FUSE and host-shared file
// systems through the page cache; Direct and Buffered force one. The bytes read are the same.
enum class ArtifactIo : std::uint8_t { Auto, Direct, Buffered };

// The most ordinary decode steps one round runs back to back (--decode-steps).
inline constexpr std::uint32_t kMaximumDecodeSteps = 32;

[[nodiscard]] constexpr const char* prefill_route_name(PrefillRoute route) noexcept {
    switch (route) {
    case PrefillRoute::Auto:
        return "auto";
    case PrefillRoute::A16:
        return "a16";
    case PrefillRoute::W4a8:
        return "w4a8";
    case PrefillRoute::I8:
        return "i8";
    case PrefillRoute::Mmq:
        return "mmq";
    case PrefillRoute::W4a4:
        return "w4a4";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char* decode_route_name(DecodeRoute route) noexcept {
    switch (route) {
    case DecodeRoute::Auto:
        return "auto";
    case DecodeRoute::A16:
        return "a16";
    case DecodeRoute::R8:
        return "r8";
    case DecodeRoute::Vec:
        return "vec";
    case DecodeRoute::Mmq:
        return "mmq";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char* moe_path_name(MoePath path) noexcept {
    switch (path) {
    case MoePath::Auto:
        return "auto";
    case MoePath::Fused:
        return "fused";
    case MoePath::Generic:
        return "generic";
    }
    return "unknown";
}

// --prefill-route auto|a16|w4a8|i8|mmq|w4a4.
[[nodiscard]] inline PrefillRoute parse_prefill_route(std::string_view text) {
    for (const PrefillRoute route : {PrefillRoute::Auto, PrefillRoute::A16, PrefillRoute::W4a8,
                                     PrefillRoute::I8, PrefillRoute::Mmq, PrefillRoute::W4a4}) {
        if (text == prefill_route_name(route)) { return route; }
    }
    throw std::invalid_argument("--prefill-route must be auto, a16, w4a8, i8, mmq or w4a4, not '" +
                                std::string(text) + "'");
}

// --decode-route auto|a16|r8|vec|mmq.
[[nodiscard]] inline DecodeRoute parse_decode_route(std::string_view text) {
    for (const DecodeRoute route : {DecodeRoute::Auto, DecodeRoute::A16, DecodeRoute::R8,
                                    DecodeRoute::Vec, DecodeRoute::Mmq}) {
        if (text == decode_route_name(route)) { return route; }
    }
    throw std::invalid_argument("--decode-route must be auto, a16, r8, vec or mmq, not '" +
                                std::string(text) + "'");
}

// --moe-path auto|fused|generic.
[[nodiscard]] inline MoePath parse_moe_path(std::string_view text) {
    for (const MoePath path : {MoePath::Auto, MoePath::Fused, MoePath::Generic}) {
        if (text == moe_path_name(path)) { return path; }
    }
    throw std::invalid_argument("--moe-path must be auto, fused or generic, not '" +
                                std::string(text) + "'");
}

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
