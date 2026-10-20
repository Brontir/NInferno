#pragma once

#include "product/media_acquire/source.h"

#include <cstddef>
#include <chrono>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct sockaddr;

namespace ninferno::product::media_acquire {

enum class ErrorKind {
    BudgetExceeded,
    // The URL itself is wrong: its host does not exist, its server answers 4xx, or its redirects
    // lead nowhere. The request's input, not a server fault.
    RemoteRejected,
    RemoteUnavailable,
    RemoteTimeout,
    DeadlineExceeded,
    Cancelled,
};

class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

struct Policy {
    std::size_t max_bytes      = 256ULL << 20;
    int connect_timeout_ms     = 5'000;
    int timeout_ms             = 60'000;
    int max_redirects          = 3;
    bool allow_remote          = true;
    bool allow_private_network = false;
    std::filesystem::path media_root;
    std::chrono::steady_clock::time_point deadline;
    std::function<bool()> is_cancelled;
};

std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy = {});

// Whether a media URL may reach the resolved address only with allow_private_network: private,
// loopback, link-local and special-purpose IPv4 ranges, site-local and unique-local IPv6, and the
// IPv6 forms that carry such an IPv4 address (mapped, compatible, NAT64, 6to4) or tunnel to one
// of their own choosing (Teredo).
[[nodiscard]] bool private_network_address(const sockaddr* address) noexcept;

} // namespace ninferno::product::media_acquire
