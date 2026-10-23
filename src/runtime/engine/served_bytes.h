#pragma once

// What an Engine of given options runs, resolved the way the Engine resolves it, for products
// that key data by it before the Engine is built (the context-class factors' byte class,
// spec/context_class.h served_byte_class): the KV storage and the activation bits of its quantized
// linears.

#include "ninferno/types.h"

#include <string>
#include <string_view>

namespace ninferno::runtime {

// The KV storage an Engine of `options` runs: options.kv_cache, or for kv_cache_automatic the
// family's (automatic_kv_cache) on the device's compute capability; a host without the device
// keeps the family's storage. The Engine resolves its own options with this.
[[nodiscard]] KvCacheStorage resolved_kv_cache(const EngineOptions& options);

// The activation bits as the byte class names them: "exact" under ActivationMode::Exact;
// "fast-a4" when a recorded use's effective policy (models::effective_policy) allows four bits
// and the W4A4 route can run (route_admits' weight-free half: the use's activation divisor, the
// device's features, this build's W4A4 kernels); else "fast-a8". Whether a call of a given width
// then takes W4A4 is the route's own choice: "fast-a4" says four bits are admitted. Pure:
// `allow_a4_recorded`: some use of the artifact records AllowA4 with its divisor.
[[nodiscard]] std::string_view activation_bits(ActivationMode mode, bool allow_a4_recorded,
                                               bool w4a4_device) noexcept;
// The same for the artifact and device of `options` (an unreadable artifact or device: no A4).
[[nodiscard]] std::string served_activations(const EngineOptions& options);

} // namespace ninferno::runtime
