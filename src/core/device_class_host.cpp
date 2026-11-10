#include "core/device_class.h"
#include "core/generation.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// The pure part of the device class (#620 step 11): feature sets, chip and class names, board
// distance and the machine-facts glue. No CUDA runtime, so it builds into ninferno_core_host. The
// probe of a real device stays in device_class.cpp.
namespace ninferno {

namespace {

// Full-chip SM counts per architecture, ascending.
struct FullChip {
    int major, minor, sms;
    const char* chip;
};

constexpr FullChip kFullChips[] = {
    {12, 0, 20, "gb207"},  {12, 0, 36, "gb206"}, {12, 0, 50, "gb205"}, {12, 0, 84, "gb203"},
    {12, 0, 192, "gb202"}, {8, 9, 24, "ad107"},  {8, 9, 36, "ad106"},  {8, 9, 60, "ad104"},
    {8, 9, 80, "ad103"},   {8, 9, 144, "ad102"}, {8, 6, 20, "ga107"},  {8, 6, 30, "ga106"},
    {8, 6, 48, "ga104"},   {8, 6, 60, "ga103"},  {8, 6, 84, "ga102"},  {7, 5, 16, "tu117"},
    {7, 5, 24, "tu116"},   {7, 5, 36, "tu106"},  {7, 5, 48, "tu104"},  {7, 5, 72, "tu102"},
    {8, 0, 128, "ga100"},  {7, 0, 80, "gv100"},
};

} // namespace

bool chip_mma_slow(int major, int minor, std::string_view chip) noexcept {
    return major == 7 && minor == 5 && (chip == "tu116" || chip == "tu117");
}

bool board_mma_slow(const MachineFacts& facts, int major, int minor,
                    std::string_view chip) noexcept {
    return !facts.sku.empty() ? facts.mma_slow : chip_mma_slow(major, minor, chip);
}

DeviceFeatures device_features_for(int major, int minor) noexcept {
    switch (major) {
    case 7: // Turing (7.5): the sm_75 build composes the m16n8k16 / m16n8k32 shapes from its f16
            // m16n8k8 and s8 m8n8k16 and rounds BF16 operands to f16. Volta (7.0, 7.2) has
            // neither: its build computes the same fragments in FP32 FMAs and dp4a over warp
            // shuffles (ops/common/mma.cuh), so every MMA path runs, exactly.
        return {.mma_bf16 = true, .mma_tensor = minor >= 5};
    case 8: // Ampere (8.0, 8.6, 8.7) and Ada (8.9): the E4M3 mma.sync arrived with Ada.
        return {
            .mma_bf16 = true, .mma_bf16_native = true, .mma_tensor = true, .mma_fp8 = minor >= 9};
    case 9: // Hopper
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    case 10: // datacenter Blackwell: the block-scaled FP8/FP4 kinds go through tcgen05, not
    case 11: // mma.sync; the un-prefixed E4M3 mma.sync is PTX every target from sm_89 carries
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    case 12: // consumer Blackwell (sm_120, sm_121)
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .mma_f8f6f4      = true,
                .mma_mxf4nvf4    = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    default:
        // A later generation than this table knows (major > 12) keeps every portable family:
        // mma.sync BF16 and E4M3, cp.async.bulk and griddepcontrol are PTX the newer targets
        // carry; the architecture-specific block-scaled kinds are not assumed. Older unknowns
        // have none.
        if (major > 12) {
            return {.mma_bf16        = true,
                    .mma_bf16_native = true,
                    .mma_tensor      = true,
                    .mma_fp8         = true,
                    .tma             = true,
                    .pdl             = true,
                    .clusters        = true};
        }
        return {};
    }
}

bool has_features(const DeviceFeatures& available, const DeviceFeatures& required) noexcept {
    return (!required.mma_bf16 || available.mma_bf16) &&
           (!required.mma_bf16_native || available.mma_bf16_native) &&
           (!required.mma_tensor || available.mma_tensor) &&
           (!required.mma_fp8 || available.mma_fp8) &&
           (!required.mma_f8f6f4 || available.mma_f8f6f4) &&
           (!required.mma_mxf4nvf4 || available.mma_mxf4nvf4) && (!required.tma || available.tma) &&
           (!required.pdl || available.pdl) && (!required.clusters || available.clusters);
}

DeviceFeatures common_features(const DeviceFeatures& a, const DeviceFeatures& b) noexcept {
    return {.mma_bf16        = a.mma_bf16 && b.mma_bf16,
            .mma_bf16_native = a.mma_bf16_native && b.mma_bf16_native,
            .mma_tensor      = a.mma_tensor && b.mma_tensor,
            .mma_fp8         = a.mma_fp8 && b.mma_fp8,
            .mma_f8f6f4      = a.mma_f8f6f4 && b.mma_f8f6f4,
            .mma_mxf4nvf4    = a.mma_mxf4nvf4 && b.mma_mxf4nvf4,
            .tma             = a.tma && b.tma,
            .pdl             = a.pdl && b.pdl,
            .clusters        = a.clusters && b.clusters,
            .mma_slow        = a.mma_slow || b.mma_slow};
}

std::string gpu_arch_id(int major, int minor) {
    return "sm" + std::to_string(major) + std::to_string(minor);
}

namespace {

// The chip whose PCI id block holds pci_device_id, for a GPU of that chip's architecture; null
// for an unknown id.
const char* pci_block_chip(std::uint32_t pci_device_id, int major, int minor) {
    // NVIDIA gives each chip blocks of PCI product ids. Verified on hardware: 0x2B85 (RTX 5090,
    // gb202) and 0x2C02 (RTX 5080, gb203); the other rows follow the published id lists.
    struct IdBlock {
        std::uint32_t first, last;
        const char* chip;
    };

    static constexpr IdBlock kIds[] = {
        {0x2B80, 0x2BFF, "gb202"}, {0x2C00, 0x2C7F, "gb203"}, {0x2F00, 0x2F7F, "gb205"},
        {0x2D00, 0x2D7F, "gb206"}, {0x2D80, 0x2DFF, "gb207"}, {0x2680, 0x26FF, "ad102"},
        {0x2700, 0x277F, "ad103"}, {0x2780, 0x27FF, "ad104"}, {0x2800, 0x287F, "ad106"},
        {0x2880, 0x28FF, "ad107"}, {0x2200, 0x227F, "ga102"}, {0x2400, 0x247F, "ga103"},
        {0x2480, 0x24FF, "ga104"}, {0x2500, 0x257F, "ga106"}, {0x2580, 0x25FF, "ga107"},
        {0x1E00, 0x1E7F, "tu102"}, {0x1E80, 0x1EFF, "tu104"}, {0x1F00, 0x1F7F, "tu106"},
        {0x1F80, 0x1FFF, "tu117"}, {0x2180, 0x21FF, "tu116"}, {0x1D80, 0x1DFF, "gv100"},
        {0x2080, 0x20FF, "ga100"},
    };
    // A block names its chip only for a GPU of that chip's architecture: Jetson Thor (sm_110)
    // reports 0x2B00, inside GB202's block, and must not take an sm_120 chip's data.
    for (const auto& block : kIds) {
        if (pci_device_id < block.first || pci_device_id > block.last) continue;
        for (const auto& c : kFullChips) {
            if (c.chip == std::string_view(block.chip) && c.major == major && c.minor == minor) {
                return block.chip;
            }
        }
    }
    return nullptr;
}

} // namespace

bool gpu_chip_by_pci(std::uint32_t pci_device_id, int major, int minor) {
    return pci_block_chip(pci_device_id, major, minor) != nullptr;
}

std::vector<std::string> device_fact_notes(const DeviceClass& d) {
    std::vector<std::string> notes;
    if (!d.chip_from_pci) {
        char id[16];
        std::snprintf(id, sizeof(id), "0x%04X", static_cast<unsigned>(d.pci_device_id));
        const std::string why  = d.pci_device_id == 0
                                     ? std::string("PCI id unreadable")
                                     : std::string("PCI id ") + id + " in no chip block";
        const std::string arch = gpu_arch_id(d.major, d.minor);
        // The PCI-ID reference table may still know the board (H100: its chip has no block): its
        // facts are the table's, only the per-chip keys are guessed.
        const GpuFactsRow* row = d.facts.sku.empty() ? nullptr : find_gpu_facts(d.pci_device_id);
        const std::string board =
            row != nullptr ? "; the board's clocks and bandwidth are the reference table's " +
                                 row->name + " (" + row->chip + ")"
                           : std::string();
        notes.push_back(d.chip == arch ? "chip unknown (" + why + "; no chip of " + arch +
                                             " in the chip blocks): per-chip data is keyed by the "
                                             "architecture " +
                                             arch + board
                                       : "chip " + d.chip + " from the SM count (" + why +
                                             "): the smallest chip of " + arch + " holding " +
                                             std::to_string(d.sms) + " SMs" + board);
    }
    std::string missing;
    const auto add = [&](bool absent, const char* what) {
        if (absent) missing += (missing.empty() ? "" : ", ") + std::string(what);
    };
    add(d.sm_clock_khz <= 0, "SM clock");
    add(d.memory_clock_khz <= 0, "memory clock");
    add(d.memory_bus_bits <= 0, "memory bus width");
    const MachineFacts& f = d.facts;
    if (!missing.empty() && f.has_roofline()) {
        // A board the PCI-ID table knows: its row stands in, and no decision reads the driver's.
        notes.push_back(missing +
                        " unreported (a virtualized or MIG GPU, for example): the "
                        "PCI-ID reference table's facts stand in (" +
                        describe(f) + ")");
    } else if (!missing.empty()) {
        notes.push_back(missing +
                        " unreported (a virtualized or MIG GPU, for example): the route cost model "
                        "and the speculative load model are off, routes come from the fixed rules, "
                        "and measured tables are matched to this board without its bandwidth");
    }
    // Facts that only approximate this device (MachineFacts::source extrapolated), with the
    // conservative choice taken.
    if (f.source == FactSource::Extrapolated) {
        std::string why;
        if (f.sku.empty()) {
            why = "PCI id not in the reference table: the driver's clocks on the 2^(1/16) grid";
        } else if (f.form == FormFactor::Laptop) {
            why = f.sku + ", a laptop: priced at the bottom of its published boost range";
        } else if (f.share < 1.0) {
            why = f.sku + ", a slice of " + std::to_string(d.sms) +
                  " SMs (MIG): its share of the board's bandwidth";
        } else if (const GpuFactsRow* row = find_gpu_facts(d.pci_device_id);
                   row != nullptr && row->sms == d.sms) {
            why = f.sku + " on another memory than its row's: the driver's bus width and memory "
                          "clock price it";
        } else {
            why = f.sku + " reporting " + std::to_string(d.sms) + " SMs, not the row's";
        }
        notes.push_back("device facts extrapolated (" + why + "): " + describe(f));
    }
    return notes;
}

std::string gpu_chip(std::uint32_t pci_device_id, int major, int minor, int sms) {
    if (const char* chip = pci_block_chip(pci_device_id, major, minor)) return chip;
    // Unknown id (or unreadable): the smallest chip of the architecture whose full SM count holds
    // the board's. Full-chip SM counts per architecture, ascending.
    for (const auto& c : kFullChips) {
        if (c.major == major && c.minor == minor && sms <= c.sms) { return c.chip; }
    }
    return gpu_arch_id(major, minor);
}

DriverReport driver_report_of(const DeviceClass& d) noexcept {
    return {.pci_device_id    = d.pci_device_id,
            .major            = d.major,
            .minor            = d.minor,
            .sms              = d.sms,
            .sm_clock_khz     = d.sm_clock_khz,
            .memory_clock_khz = d.memory_clock_khz,
            .memory_bus_bits  = d.memory_bus_bits};
}

MachineFacts machine_facts_of(const DeviceClass& d) { return machine_facts(driver_report_of(d)); }

Board board_of(const DeviceClass& d) noexcept {
    return {d.major,       d.minor,
            d.sms,         static_cast<double>(d.l2_bytes) / (1 << 20),
            d.dram_gbps(), d.features.mma_slow};
}

namespace {

double board_distance_of(const Board& a, const Board& b, bool hmma_rate) noexcept {
    const DeviceFeatures fa = device_features_for(a.major, a.minor);
    const DeviceFeatures fb = device_features_for(b.major, b.minor);
    // One compute capability is one kind; across two, a generation the table does not know has
    // no kind to compare (#300).
    const bool same_cc = a.major == b.major && a.minor == b.minor;
    const auto kind_a  = fp32acc_full_rate_of(a.major, a.minor);
    if (!has_features(fa, fb) || !has_features(fb, fa) || a.mma_slow != b.mma_slow || a.sms <= 0 ||
        b.sms <= 0 ||
        (hmma_rate && !same_cc && (!kind_a || kind_a != fp32acc_full_rate_of(b.major, b.minor)))) {
        return std::numeric_limits<double>::infinity();
    }
    const auto apart = [](double x, double y) {
        return x > 0 && y > 0 ? std::abs(std::log2(x) - std::log2(y)) : 0.0; // symmetric to the bit
    };
    return apart(a.sms, b.sms) + apart(a.dram_gbps, b.dram_gbps) + 0.5 * apart(a.l2_mib, b.l2_mib);
}

} // namespace

double board_distance(const Board& a, const Board& b) noexcept {
    return board_distance_of(a, b, true);
}

double board_memory_distance(const Board& a, const Board& b) noexcept {
    return board_distance_of(a, b, false);
}

BoardUnknowns board_unknowns(const Board& a, const Board& b) noexcept {
    return {.l2 = !(a.l2_mib > 0 && b.l2_mib > 0), .dram = !(a.dram_gbps > 0 && b.dram_gbps > 0)};
}

std::string gpu_chip_arch(std::string_view gpu_class) {
    const auto chip = gpu_class.substr(0, gpu_class.find(':'));
    if (chip.size() > 2 && chip.starts_with("sm") &&
        chip.find_first_not_of("0123456789", 2) == std::string_view::npos) {
        if (chip.size() != gpu_class.size()) { return {}; }
        for (const auto& full : kFullChips) {
            if (gpu_arch_id(full.major, full.minor) == chip) { return std::string(chip); }
        }
        return {};
    }
    for (const auto& full : kFullChips) {
        if (chip == full.chip) { return gpu_arch_id(full.major, full.minor); }
    }
    return {};
}

int gpu_chip_full_sms(std::string_view gpu_class) {
    const auto chip = gpu_class.substr(0, gpu_class.find(':'));
    for (const auto& full : kFullChips) {
        if (chip == full.chip) { return full.sms; }
    }
    return 0;
}

std::string gpu_class_id(std::string_view chip, int sms) {
    return std::string(chip) + ":" + std::to_string(sms) + "sm";
}

std::vector<std::string> gpu_similarity_chain(std::string_view chip, int sms, int major, int minor,
                                              std::optional<bool> mma_slow) {
    std::vector<std::string> chain = gpu_class_chain(chip, sms, gpu_arch_id(major, minor));
    chain.pop_back(); // "*" goes after the architecture's own chips
    const auto add = [&](std::string key) {
        if (std::find(chain.begin(), chain.end(), key) == chain.end()) {
            chain.push_back(std::move(key));
        }
    };
    for (auto& similar : gpu_similar_chips(major, minor, sms, chip, mma_slow)) {
        add(std::move(similar));
    }
    add("*");
    // The other architectures, nearest compute capability first.
    std::vector<std::pair<int, int>> archs;
    for (const auto& c : kFullChips) {
        const std::pair<int, int> arch{c.major, c.minor};
        if (arch != std::pair{major, minor} &&
            std::find(archs.begin(), archs.end(), arch) == archs.end()) {
            archs.push_back(arch);
        }
    }
    const int cc = major * 10 + minor;
    std::stable_sort(archs.begin(), archs.end(), [cc](const auto& a, const auto& b) {
        return std::abs(a.first * 10 + a.second - cc) < std::abs(b.first * 10 + b.second - cc);
    });
    for (const auto& [other_major, other_minor] : archs) {
        for (auto& similar : gpu_similar_chips(other_major, other_minor, sms, "")) {
            add(std::move(similar));
        }
    }
    return chain;
}

std::vector<std::string> gpu_similar_chips(int major, int minor, int sms, std::string_view own_chip,
                                           std::optional<bool> mma_slow) {
    // A board without tensor cores (mma_slow) and one with them are other kinds: measured data of
    // one does not carry to the other. The board's own fact, where given: a GTX board on TU106
    // has none, though its chip does.
    const bool slow = mma_slow.value_or(chip_mma_slow(major, minor, own_chip));
    std::vector<const FullChip*> chips;
    for (const auto& c : kFullChips) {
        if (c.major == major && c.minor == minor && own_chip != c.chip &&
            chip_mma_slow(c.major, c.minor, c.chip) == slow) {
            chips.push_back(&c);
        }
    }
    const double board  = std::log2(static_cast<double>(std::max(sms, 1)));
    const auto distance = [board](const FullChip* c) {
        return std::abs(std::log2(static_cast<double>(c->sms)) - board);
    };
    const auto nearest_first = [&](const FullChip* a, const FullChip* b) {
        const double da = distance(a);
        const double db = distance(b);
        return da != db ? da < db : a->sms > b->sms;
    };
    std::stable_sort(chips.begin(), chips.end(), nearest_first);
    // Then the chips of a sibling architecture: the same major with the same feature row (sm_80
    // and sm_86, sm_121 and sm_120), where the measured data of the other minor is the nearest
    // there is; an architecture with no chip of its own (A100, A30, GB10) otherwise borrows none.
    const DeviceFeatures own = device_features_for(major, minor);
    const auto first_sibling = chips.size();
    for (const auto& c : kFullChips) {
        const DeviceFeatures other = device_features_for(c.major, c.minor);
        if (c.major == major && c.minor != minor && own_chip != c.chip &&
            has_features(own, other) && has_features(other, own)) {
            chips.push_back(&c);
        }
    }
    std::stable_sort(chips.begin() + static_cast<std::ptrdiff_t>(first_sibling), chips.end(),
                     nearest_first);
    std::vector<std::string> out;
    out.reserve(chips.size());
    for (const FullChip* c : chips) { out.emplace_back(c->chip); }
    return out;
}

std::vector<std::string> gpu_class_chain(std::string_view chip, int sms, std::string_view arch) {
    std::vector<std::string> chain;
    for (std::string key :
         {gpu_class_id(chip, sms), std::string(chip), std::string(arch), std::string("*")}) {
        if (!key.empty() && std::find(chain.begin(), chain.end(), key) == chain.end()) {
            chain.push_back(std::move(key));
        }
    }
    return chain;
}

std::string DeviceClass::id() const { return gpu_class_id(chip, sms); }

std::string DeviceClass::arch() const { return gpu_arch_id(major, minor); }

std::vector<std::string> DeviceClass::class_chain() const {
    return gpu_class_chain(chip, sms, arch());
}

// The board's own mma_slow is features.mma_slow alone: the query and synthetic_device_class set
// it from board_mma_slow, its one source.
std::vector<std::string> DeviceClass::similar_chips() const {
    return gpu_similar_chips(major, minor, sms, chip, features.mma_slow);
}

std::vector<std::string> DeviceClass::similarity_chain() const {
    return gpu_similarity_chain(chip, sms, major, minor, features.mma_slow);
}

} // namespace ninferno
