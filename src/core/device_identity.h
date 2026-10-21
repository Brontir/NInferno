#pragma once

// Which physical GPU a CUDA ordinal names. An ordinal is only a position in this process's
// enumeration: CUDA_VISIBLE_DEVICES reorders and filters it, and CUDA_DEVICE_ORDER defaults to
// FASTEST_FIRST, CUDA's own speed guess, so the same cards get other ordinals on another run or
// machine. The identity of a board is its PCI address (domain:bus:device.function) and its UUID;
// the UUID tells apart MIG instances, which share their parent's PCI address.
//
// Within one process an ordinal stays bound to its board, so in-process caches may be indexed by
// it (core/per_device.h). Anything that outlives the process or chooses among GPUs reads the
// identity instead: a choice among GPUs depends only on the set of identities, sorted
// canonically (canonical_device_order), never on the enumeration order. An ordinal the user
// passes (--device N) is an explicit option and is taken as given.
//
// The identity names a board; it is never a decision input beyond that ordering, and it is not
// part of the decision fingerprint: two boards of one chip type make the same decisions. The
// fingerprint prints it beside the hash, to say which board the facts were read from.
//
// No CUDA here: the dispatcher, which links no CUDA library, and CPU tests use the same ordering.
// The query from the runtime is device_identity() in core/device_class.h.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

// The device option's default: no ordinal given, so the first GPU in canonical order runs.
inline constexpr int kDefaultDevice = -1;

struct DeviceIdentity {
    std::uint32_t pci_domain   = 0;
    std::uint32_t pci_bus      = 0;
    std::uint32_t pci_device   = 0;
    std::uint32_t pci_function = 0;
    std::array<std::uint8_t, 16> uuid{}; // cudaDeviceProp::uuid; all zero when unknown

    // The canonical order: by PCI address, then by UUID.
    friend auto operator<=>(const DeviceIdentity&, const DeviceIdentity&) = default;
};

// "dddd:bb:dd.f" in hex as cudaDeviceGetPCIBusId and NVML write it (NVML's domain has 8 digits,
// CUDA's 4); nullopt for anything else.
[[nodiscard]] inline std::optional<DeviceIdentity> parse_pci_bus_id(std::string_view text) {
    DeviceIdentity identity;
    const auto field = [&text](char end, std::size_t max_digits, std::uint32_t& value) -> bool {
        const std::size_t stop = end == '\0' ? text.size() : text.find(end);
        if (stop == std::string_view::npos || stop == 0 || stop > max_digits) { return false; }
        const auto [ptr, error] = std::from_chars(text.data(), text.data() + stop, value, 16);
        if (error != std::errc{} || ptr != text.data() + stop) { return false; }
        text.remove_prefix(end == '\0' ? stop : stop + 1);
        return true;
    };
    if (!field(':', 8, identity.pci_domain) || !field(':', 2, identity.pci_bus) ||
        !field('.', 2, identity.pci_device) || !field('\0', 1, identity.pci_function)) {
        return std::nullopt;
    }
    if (identity.pci_device > 0x1F || identity.pci_function > 7) { return std::nullopt; }
    return identity;
}

// "0000:01:00.0".
[[nodiscard]] inline std::string pci_bus_id_text(const DeviceIdentity& identity) {
    char text[32];
    std::snprintf(text, sizeof(text), "%04x:%02x:%02x.%x", identity.pci_domain, identity.pci_bus,
                  identity.pci_device, identity.pci_function);
    return text;
}

// "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", as nvidia-smi and CUDA_VISIBLE_DEVICES spell it.
[[nodiscard]] inline std::string uuid_text(const DeviceIdentity& identity) {
    std::string text      = "GPU-";
    constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < identity.uuid.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { text += '-'; }
        text += kHex[identity.uuid[i] >> 4];
        text += kHex[identity.uuid[i] & 0xF];
    }
    return text;
}

// "0000:01:00.0 GPU-...": how logs and the fingerprint name a board.
[[nodiscard]] inline std::string identity_text(const DeviceIdentity& identity) {
    return pci_bus_id_text(identity) + ' ' + uuid_text(identity);
}

// The ordinals of `enumerated` (entry i is ordinal i's identity) in canonical order. The same set
// of boards gives the same sequence of identities however the enumeration ordered them.
[[nodiscard]] inline std::vector<int>
canonical_device_order(std::span<const DeviceIdentity> enumerated) {
    std::vector<int> order(enumerated.size());
    for (std::size_t i = 0; i < order.size(); ++i) { order[i] = static_cast<int>(i); }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return enumerated[static_cast<std::size_t>(a)] < enumerated[static_cast<std::size_t>(b)];
    });
    return order;
}

// The device an option names: `requested` as given when it is an ordinal, else (kDefaultDevice)
// the first board in canonical order; kDefaultDevice when nothing is enumerated.
[[nodiscard]] inline int resolve_device(int requested, std::span<const DeviceIdentity> enumerated) {
    if (requested != kDefaultDevice) { return requested; }
    const std::vector<int> order = canonical_device_order(enumerated);
    return order.empty() ? kDefaultDevice : order.front();
}

} // namespace ninferno
