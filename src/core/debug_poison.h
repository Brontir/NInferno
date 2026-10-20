#pragma once

#include <optional>

namespace ninferno {

// Debug-only poisoning of pooled device state. With NINFERNO_DEBUG_POISON set, every
// pooled buffer returned to its free list is overwritten with one byte value, so a later owner
// that reads it before initializing it sees poison instead of the previous owner's plausible data:
// 0xFF is NaN in BF16, FP16, FP32 and FP8, and -1 in integer indices. Values:
//   unset, empty or "0"  off (the default; no cost beyond one cached check per release);
//   "1" or "nan"         0xFF;
//   "0x.." or a decimal  that byte (1..255).
// Any other value is rejected at first use.
[[nodiscard]] std::optional<unsigned char> debug_poison_byte();

// debug_poison_byte() for one poisoning site, for bisecting which poisoned memory changed a
// result: NINFERNO_DEBUG_POISON_SKIP is a comma-separated list of sites left unpoisoned. Sites:
// "kv" (device KV pool pages), "ingress" (per-round host ingress rows past the batch), "state"
// (released StateImage slots), "weights" (the weight arena before upload).
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
