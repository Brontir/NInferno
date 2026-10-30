#pragma once

// ninferno::ops - compile-time limits of the architecture this build compiles kernels for.
//
// A kernel's static schedule is checked against these when it is compiled. Which device runs it
// is a runtime question (core/device_class.h), and cudaFuncSetAttribute refuses a launch the
// running device cannot hold.

#include <cstddef>
#include <type_traits>

namespace ninferno::ops {

// The dynamic shared memory one block may opt into, by the build's architecture
// (NINFERNO_BUILD_ARCH, CMakeLists.txt): Volta (7.0, 7.2) 96 KiB, Turing (7.5) 64 KiB, GA100 and
// Orin (8.0, 8.7) 163 KiB, everything else the 99 KiB of GA10x Ampere, Ada and consumer Blackwell
// (101376 B opt-in). Most tiles are tuned within 99 KiB; the ones above it (deeper pipelines for a
// tensor core that outruns its 99 KiB stages) are admitted only where the block holds them. One tree
// compiles one architecture, so the value holds for host and device code alike.
inline constexpr std::size_t shared_bytes_per_block_for_arch(int arch) {
    if (arch == 75) { return 64 * 1024; }
    if (arch >= 70 && arch < 75) { return 96 * 1024; }
    if (arch == 80 || arch == 87) { return 163 * 1024; }
    return 99 * 1024;
}
#if defined(NINFERNO_BUILD_ARCH)
inline constexpr std::size_t kMaxSharedBytesPerBlock =
    shared_bytes_per_block_for_arch(NINFERNO_BUILD_ARCH);
#else
inline constexpr std::size_t kMaxSharedBytesPerBlock = 99 * 1024;
#endif

// The shared memory of one SM, which its resident blocks share (CUDA programming guide, compute
// capabilities table): Volta 96 KiB, Turing 64 KiB, GA100 and Orin 164 KiB, GA10x Ampere, Ada and
// consumer Blackwell 100 KiB. From sm_80 the driver reserves 1 KiB of it per resident block.
inline constexpr std::size_t shared_bytes_per_sm_for_arch(int arch) {
    if (arch == 75) { return 64 * 1024; }
    if (arch >= 70 && arch < 75) { return 96 * 1024; }
    if (arch == 80 || arch == 87) { return 164 * 1024; }
    return 100 * 1024;
}
inline constexpr int shared_bytes_reserved_per_block_for_arch(int arch) { return arch >= 80 ? 1024 : 0; }
#if defined(NINFERNO_BUILD_ARCH)
inline constexpr std::size_t kSharedBytesPerSm = shared_bytes_per_sm_for_arch(NINFERNO_BUILD_ARCH);
inline constexpr int kSharedBytesReservedPerBlock =
    shared_bytes_reserved_per_block_for_arch(NINFERNO_BUILD_ARCH);
#else
inline constexpr std::size_t kSharedBytesPerSm     = 100 * 1024;
inline constexpr int kSharedBytesReservedPerBlock = 1024;
#endif

// The shared memory a block gets without opting in (cudaFuncAttributeMaxDynamicSharedMemorySize):
// static arrays above it do not compile, and a launch that needs more must raise the attribute.
inline constexpr int kStaticSharedBytesPerBlock = 48 * 1024;

// A grid's y and z extents stop here on every architecture; longer token runs stride or split.
inline constexpr int kMaxGridYZ = 65535;

// A schedule states whether its shared memory fits one block of this build (kFitsSharedLimit); its
// kernel asserts it. A route names the schedule it was tuned with and the one it falls back to, and
// takes the first that fits: Turing's 64 KiB and Volta's 96 KiB turn the tiles above them away,
// every other build keeps its measured schedule.
template <class Preferred, class Fallback>
using fitting_schedule = std::conditional_t<Preferred::kFitsSharedLimit, Preferred, Fallback>;

// The launch of a tile over the build's block limit is refused on the host (*_prepare_shared). A
// build with NINFERNO_SHARED_AUDIT defined names every such tile a route can reach at compile time:
// the refusal instantiates this deprecated marker, and the compiler's warning carries the kernel.
template <int Bytes, auto Kernel>
[[deprecated("NINFERNO_SHARED_AUDIT: a reachable tile exceeds this build's shared memory")]]
constexpr void shared_limit_audit() {}

} // namespace ninferno::ops
