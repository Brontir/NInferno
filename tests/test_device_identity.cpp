// The device identity (core/device_identity.h): PCI bus ids parse and print as CUDA and NVML write
// them, and the GPU a default chooses depends on the set of boards, never on the order CUDA
// enumerates them in (CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER). No GPU.

#include "core/device_identity.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninferno::DeviceIdentity;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

DeviceIdentity board(std::uint32_t domain, std::uint32_t bus, std::uint32_t device,
                     std::uint32_t function, std::uint8_t uuid_tag) {
    DeviceIdentity identity{domain, bus, device, function, {}};
    identity.uuid.fill(uuid_tag);
    return identity;
}

void test_pci_bus_id_text() {
    const auto cuda = ninferno::parse_pci_bus_id("0000:41:00.0");
    expect(cuda && cuda->pci_domain == 0 && cuda->pci_bus == 0x41 && cuda->pci_device == 0 &&
               cuda->pci_function == 0,
           "CUDA's bus id must parse");
    const auto nvml = ninferno::parse_pci_bus_id("00000001:C1:1F.7");
    expect(nvml && nvml->pci_domain == 1 && nvml->pci_bus == 0xC1 && nvml->pci_device == 0x1F &&
               nvml->pci_function == 7,
           "NVML's eight-digit domain and upper-case hex must parse");
    expect(cuda && ninferno::pci_bus_id_text(*cuda) == "0000:41:00.0", "bus id must round-trip");
    for (const char* bad : {"", "0000:41:00", "0000:41:00.8", "0000:41:20.0", "0000:141:00.0",
                            "000000000:41:00.0", "0000:41:00.0x", "zz:41:00.0", "0000-41-00.0"}) {
        expect(!ninferno::parse_pci_bus_id(bad), std::string("must refuse bus id \"") + bad + '"');
    }

    DeviceIdentity identity{};
    for (std::size_t i = 0; i < identity.uuid.size(); ++i) {
        identity.uuid[i] = static_cast<std::uint8_t>(i * 17);
    }
    expect(ninferno::uuid_text(identity) == "GPU-00112233-4455-6677-8899-aabbccddeeff",
           "UUID must print as nvidia-smi prints it, got " + ninferno::uuid_text(identity));
}

// Every enumeration order of one set of boards: the canonical sequence of identities and the
// default board are the same.
void test_permutations_choose_the_same_board() {
    const std::vector<DeviceIdentity> boards = {
        board(0, 0x01, 0, 0, 0x10), // the lowest PCI address
        board(0, 0x41, 0, 0, 0x20), board(0, 0x81, 0, 0, 0x30),
        board(1, 0x01, 0, 0, 0x05), // a second PCI domain sorts after the first
        board(0, 0x41, 0, 0, 0x08), // a MIG instance: the parent's PCI address, its own UUID
    };
    std::vector<DeviceIdentity> canonical = boards;
    std::sort(canonical.begin(), canonical.end());
    expect(canonical.front() == boards[0], "the lowest PCI address must sort first");
    expect(canonical[1] == boards[4] && canonical[2] == boards[1],
           "boards at one PCI address must sort by UUID");

    std::vector<int> permutation(boards.size());
    for (std::size_t i = 0; i < permutation.size(); ++i) { permutation[i] = static_cast<int>(i); }
    int seen = 0;
    do {
        // CUDA's enumeration under this permutation: ordinal i is boards[permutation[i]].
        std::vector<DeviceIdentity> enumerated;
        for (const int p : permutation) {
            enumerated.push_back(boards[static_cast<std::size_t>(p)]);
        }

        std::vector<DeviceIdentity> ordered;
        for (const int ordinal : ninferno::canonical_device_order(enumerated)) {
            ordered.push_back(enumerated[static_cast<std::size_t>(ordinal)]);
        }
        expect(ordered == canonical, "canonical order must not depend on the enumeration");

        const int chosen = ninferno::resolve_device(ninferno::kDefaultDevice, enumerated);
        expect(chosen >= 0 && enumerated[static_cast<std::size_t>(chosen)] == boards[0],
               "the default must be the same board under every enumeration");
        ++seen;
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    expect(seen == 120, "every permutation of five boards must be tried");
}

// An explicit ordinal is the user's option and is taken as given, whatever the boards.
void test_explicit_ordinal_is_kept() {
    const std::vector<DeviceIdentity> enumerated = {board(0, 0x81, 0, 0, 1),
                                                    board(0, 0x01, 0, 0, 2)};
    expect(ninferno::resolve_device(0, enumerated) == 0, "--device 0 must stay ordinal 0");
    expect(ninferno::resolve_device(1, enumerated) == 1, "--device 1 must stay ordinal 1");
    expect(ninferno::resolve_device(ninferno::kDefaultDevice, enumerated) == 1,
           "the default must be the board at the lowest PCI address");
    expect(ninferno::resolve_device(ninferno::kDefaultDevice, {}) == ninferno::kDefaultDevice,
           "no board must resolve to no device");
}

// A board the driver could not report (domain 0xFFFFFFFF) is never the default.
void test_unreadable_board_sorts_last() {
    const std::vector<DeviceIdentity> enumerated = {board(0xFFFFFFFFu, 0, 0, 0, 0),
                                                    board(0, 0x02, 0, 0, 3)};
    expect(ninferno::resolve_device(ninferno::kDefaultDevice, enumerated) == 1,
           "an unreadable board must not be the default");
}

} // namespace

int main() {
    test_pci_bus_id_text();
    test_permutations_choose_the_same_board();
    test_explicit_ordinal_is_kept();
    test_unreadable_board_sorts_last();
    if (failures != 0) {
        std::cerr << failures << " device identity check(s) failed\n";
        return 1;
    }
    std::cout << "device identity: ok\n";
    return 0;
}
