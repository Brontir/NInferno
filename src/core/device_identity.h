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
// canonically (canonical_device_order), never on the enumeration order. The user names a GPU by
// --device (DeviceSelector): N counts the visible GPUs in that order, as nvidia-smi numbers them,
// or a PCI address or UUID names one board.
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
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

// EngineOptions::device's default: no ordinal given, so the first GPU in canonical order runs.
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

// The domain of a board CUDA could not report a PCI address for: sorted last, never named.
inline constexpr std::uint32_t kUnknownPciDomain = 0xFFFFFFFFu;

// Whether the identity's PCI address and UUID were reported (an unknown one is never printed as
// if it named a board).
[[nodiscard]] inline bool pci_known(const DeviceIdentity& identity) noexcept {
    return identity.pci_domain != kUnknownPciDomain;
}
[[nodiscard]] inline bool uuid_known(const DeviceIdentity& identity) noexcept {
    for (const std::uint8_t byte : identity.uuid) {
        if (byte != 0) { return true; }
    }
    return false;
}

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

// "0000:01:00.0 GPU-...": how logs name a board, each part only where the driver reported it
// ("PCI address unknown" for a missing address, no UUID for a missing one).
[[nodiscard]] inline std::string identity_text(const DeviceIdentity& identity) {
    std::string text = pci_known(identity) ? pci_bus_id_text(identity) : "PCI address unknown";
    if (uuid_known(identity)) { text += ' ' + uuid_text(identity); }
    return text;
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

// The ordinal EngineOptions::device names: as given, or for kDefaultDevice the first board in
// canonical order; kDefaultDevice when nothing is enumerated.
[[nodiscard]] inline int resolve_device(int requested, std::span<const DeviceIdentity> enumerated) {
    if (requested != kDefaultDevice) { return requested; }
    const std::vector<int> order = canonical_device_order(enumerated);
    return order.empty() ? kDefaultDevice : order.front();
}

// What --device names. Every form reads the boards, never the enumeration order:
//   absent              the first visible GPU in canonical order (Default);
//   N                   the N-th visible GPU in canonical order, counted from 0: nvidia-smi's
//                       numbering when CUDA_VISIBLE_DEVICES hides none (Index);
//   pci:0000:01:00.0    the board at that PCI address; of MIG instances there, the first (Pci);
//   GPU-<uuid>          the board with that UUID, as nvidia-smi -L prints it (Uuid).
// MIG-<uuid> is not a form: CUDA gives a MIG instance's UUID in the same bytes as a GPU's, and
// only NVML tells them apart, untested here. CUDA_VISIBLE_DEVICES=MIG-<uuid> chooses an instance,
// which is then the only visible device (deterministic-decisions.md, Device identity).
struct DeviceSelector {
    enum class Kind { Default, Index, Pci, Uuid };
    Kind kind = Kind::Default;
    int index = 0;          // Index
    DeviceIdentity address; // Pci: the PCI fields; Uuid: the uuid
    friend bool operator==(const DeviceSelector&, const DeviceSelector&) = default;
};

// "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" in either case; nullopt for anything else.
[[nodiscard]] inline std::optional<std::array<std::uint8_t, 16>>
parse_uuid_text(std::string_view text) {
    if (text.size() != 40 || (text.substr(0, 4) != "GPU-" && text.substr(0, 4) != "gpu-")) {
        return std::nullopt;
    }
    std::array<std::uint8_t, 16> uuid{};
    std::size_t at = 4;
    for (std::size_t i = 0; i < uuid.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            if (text[at++] != '-') { return std::nullopt; }
        }
        unsigned byte           = 0;
        const char* first       = text.data() + at;
        const auto [ptr, error] = std::from_chars(first, first + 2, byte, 16);
        if (error != std::errc{} || ptr != first + 2) { return std::nullopt; }
        uuid[i] = static_cast<std::uint8_t>(byte);
        at += 2;
    }
    return uuid;
}

// The selector `text` spells (see DeviceSelector); nullopt for anything else.
[[nodiscard]] inline std::optional<DeviceSelector> parse_device_selector(std::string_view text) {
    DeviceSelector selector;
    if (text.starts_with("pci:")) {
        const auto address = parse_pci_bus_id(text.substr(4));
        if (!address) { return std::nullopt; }
        selector.kind    = DeviceSelector::Kind::Pci;
        selector.address = *address;
        return selector;
    }
    if (const auto uuid = parse_uuid_text(text)) {
        selector.kind         = DeviceSelector::Kind::Uuid;
        selector.address.uuid = *uuid;
        return selector;
    }
    int index               = 0;
    const auto [ptr, error] = std::from_chars(text.data(), text.data() + text.size(), index);
    if (text.empty() || error != std::errc{} || ptr != text.data() + text.size() || index < 0) {
        return std::nullopt;
    }
    selector.kind  = DeviceSelector::Kind::Index;
    selector.index = index;
    return selector;
}

// parse_device_selector for an option parser: throws std::invalid_argument naming the forms.
[[nodiscard]] inline DeviceSelector parse_device_option(std::string_view text) {
    if (const auto selector = parse_device_selector(text)) { return *selector; }
    throw std::invalid_argument("--device must be N (counting GPUs in PCI bus order, as "
                                "nvidia-smi does), pci:DDDD:BB:DD.F or GPU-<uuid>, not \"" +
                                std::string(text) + "\"");
}

// How the selector is spelled ("" for the default).
[[nodiscard]] inline std::string device_selector_text(const DeviceSelector& selector) {
    switch (selector.kind) {
    case DeviceSelector::Kind::Default:
        return "";
    case DeviceSelector::Kind::Index:
        return std::to_string(selector.index);
    case DeviceSelector::Kind::Pci:
        return "pci:" + pci_bus_id_text(selector.address);
    case DeviceSelector::Kind::Uuid:
        return uuid_text(selector.address);
    }
    return "";
}

// The ordinal of the board `selector` names among `enumerated` (entry i is ordinal i's identity);
// nullopt when none does.
[[nodiscard]] inline std::optional<int> select_device(const DeviceSelector& selector,
                                                      std::span<const DeviceIdentity> enumerated) {
    const std::vector<int> order = canonical_device_order(enumerated);
    const auto at                = [&](int ordinal) -> const DeviceIdentity& {
        return enumerated[static_cast<std::size_t>(ordinal)];
    };
    switch (selector.kind) {
    case DeviceSelector::Kind::Default:
        if (!order.empty()) { return order.front(); }
        break;
    case DeviceSelector::Kind::Index:
        if (static_cast<std::size_t>(selector.index) < order.size()) {
            return order[static_cast<std::size_t>(selector.index)];
        }
        break;
    case DeviceSelector::Kind::Pci:
        for (const int ordinal : order) {
            const DeviceIdentity& d = at(ordinal);
            if (pci_known(d) && d.pci_domain == selector.address.pci_domain &&
                d.pci_bus == selector.address.pci_bus &&
                d.pci_device == selector.address.pci_device &&
                d.pci_function == selector.address.pci_function) {
                return ordinal;
            }
        }
        break;
    case DeviceSelector::Kind::Uuid:
        for (const int ordinal : order) {
            if (uuid_known(at(ordinal)) && at(ordinal).uuid == selector.address.uuid) { return ordinal; }
        }
        break;
    }
    return std::nullopt;
}

} // namespace ninferno
