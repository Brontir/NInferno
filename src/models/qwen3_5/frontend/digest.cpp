#include "models/qwen3_5/frontend/digest.h"
#include "core/sha256.h"

namespace ninferno::models::qwen3_5::frontend {

// Forwarding wrappers: the implementation is core/sha256.cpp.

Sha256Digest sha256(std::span<const std::uint8_t> input) { return ninferno::sha256(input); }

Sha256Digest sha256(std::span<const std::uint8_t> input, const std::function<void()>& checkpoint) {
    return ninferno::sha256(input, checkpoint);
}

Sha256Digest sha256(std::string_view input) { return ninferno::sha256(input); }

std::string sha256_hex(const Sha256Digest& digest) { return ninferno::sha256_hex(digest); }

} // namespace ninferno::models::qwen3_5::frontend
