#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

// Debug-only poisoning of pooled device state (MaintainerOptions::debug_poison, the engine's
// --debug-poison). While set, every pooled buffer returned to its free list is overwritten with
// one byte value, so a later owner that reads it before initializing it sees poison instead of
// the previous owner's plausible data: 0xFF is NaN in BF16, FP16, FP32 and FP8, and -1 in integer
// indices. Off by default, at no cost beyond one check per release.
[[nodiscard]] std::optional<unsigned char> debug_poison_byte();

// --debug-poison 0|1|nan|BYTE: 0 is off, 1 and nan are 0xFF, else that byte ("0x.." or decimal,
// 1..255). Anything else is refused naming the option.
[[nodiscard]] std::optional<std::uint8_t> parse_debug_poison(std::string_view text);

// Sets the process's poison byte and the sites it skips (kv, ingress, state, weights; an unknown
// site is refused). The engine sets both at construction, before it allocates; a test that
// materializes without an engine sets them itself.
void set_debug_poison(std::optional<std::uint8_t> byte, const std::vector<std::string>& skip);

// debug_poison_byte() for one poisoning site, for bisecting which poisoned memory changed a
// result: the sites set_debug_poison skips are left unpoisoned. Sites: "kv" (device KV pool
// pages), "ingress" (per-round host ingress rows past the batch), "state" (released StateImage
// slots), "weights" (the weight arena before upload).
[[nodiscard]] std::optional<unsigned char> debug_poison_byte_for(const char* site);

// Debug-only executor freeze. While the file named by NINFERNO_DEBUG_FREEZE_FILE
// exists, debug_freeze_point() blocks, polling every 10 ms, as a hung device wait blocks the
// executor. Unset (the default), it returns after one cached check.
void debug_freeze_point() noexcept;

// Debug-only device fault. The first call after the file named by
// NINFERNO_DEBUG_TRAP_FILE appears launches a kernel that executes a device-side trap, which
// poisons the CUDA context exactly as a failed device-side assert does. Unset, one cached check.
void debug_device_fault_point() noexcept;

// Debug-only transient CUDA error. When the file named by NINFERNO_DEBUG_CUDA_ERROR_FILE exists,
// debug_transient_cuda_error_point() removes it and throws a CudaError the device survives
// (cudaErrorMemoryAllocation), as a failed allocation or launch configuration would. Each new
// file is one error. Unset, one cached check.
void debug_transient_cuda_error_point();

} // namespace ninferno
