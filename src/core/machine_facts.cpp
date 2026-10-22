#include "core/machine_facts.h"

#include "core/embedded_gpu_facts.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>

namespace ninferno {

std::string_view to_string(FactSource source) noexcept {
    switch (source) {
    case FactSource::Verified:
        return "verified";
    case FactSource::Extrapolated:
        return "extrapolated";
    case FactSource::None:
        break;
    }
    return "none";
}

std::string_view to_string(Segment segment) noexcept {
    switch (segment) {
    case Segment::GeForce:
        return "geforce";
    case Segment::Pro:
        return "pro";
    case Segment::Datacenter:
        return "datacenter";
    case Segment::Unknown:
        break;
    }
    return "unknown";
}

std::string_view to_string(FormFactor form) noexcept {
    switch (form) {
    case FormFactor::Desktop:
        return "desktop";
    case FormFactor::Laptop:
        return "laptop";
    case FormFactor::Server:
        return "server";
    case FormFactor::Unknown:
        break;
    }
    return "unknown";
}

namespace {

using Json = nlohmann::json;

Segment segment_of(const std::string& s) {
    if (s == "geforce") return Segment::GeForce;
    if (s == "pro") return Segment::Pro;
    if (s == "datacenter") return Segment::Datacenter;
    throw std::runtime_error("segment \"" + s + "\" is not geforce, pro or datacenter");
}

FormFactor form_of(const std::string& s) {
    if (s == "desktop") return FormFactor::Desktop;
    if (s == "laptop") return FormFactor::Laptop;
    if (s == "server") return FormFactor::Server;
    throw std::runtime_error("form \"" + s + "\" is not desktop, laptop or server");
}

std::uint32_t pci_id_of(const std::string& s) {
    if (s.size() != 6 || s[0] != '0' || s[1] != 'x') {
        throw std::runtime_error("id \"" + s + "\" is not 0x and four hex digits");
    }
    std::size_t used      = 0;
    const unsigned long v = std::stoul(s.substr(2), &used, 16);
    if (used != 4) throw std::runtime_error("id \"" + s + "\" is not 0x and four hex digits");
    return static_cast<std::uint32_t>(v);
}

GpuFactsRow parse_row(const Json& j) {
    GpuFactsRow r;
    r.pci_device_id = pci_id_of(j.at("id").get<std::string>());
    r.name          = j.at("name").get<std::string>();
    r.chip          = j.at("chip").get<std::string>();
    const int cc    = j.at("cc").get<int>();
    r.major         = cc / 10;
    r.minor         = cc % 10;
    r.segment       = segment_of(j.at("segment").get<std::string>());
    r.form          = form_of(j.at("form").get<std::string>());
    r.sms           = j.at("sms").get<int>();
    r.boost_mhz     = j.at("boost_mhz").get<int>();
    r.boost_min_mhz = j.value("boost_min_mhz", 0);
    r.bus_bits      = j.at("bus_bits").get<int>();
    r.gbps          = j.at("gbps").get<double>();
    r.tdp_w         = j.at("tdp_w").get<int>();
    r.vram_gib      = j.at("vram_gib").get<double>();
    r.pcie_gen      = j.at("pcie").get<int>();
    r.mma_slow      = j.value("mma_slow", false);
    r.target        = j.value("target", true);
    r.sources       = j.at("sources").get<std::vector<std::string>>();
    r.note          = j.value("note", std::string());
    if (cc < 70 || r.sms <= 0 || r.boost_mhz <= 0 || r.bus_bits <= 0 || r.gbps <= 0 ||
        r.tdp_w <= 0 || r.vram_gib <= 0 || r.pcie_gen < 3 || r.name.empty() || r.chip.empty()) {
        throw std::runtime_error("a quantity is missing or out of range");
    }
    if (r.boost_min_mhz < 0 || r.boost_min_mhz > r.boost_mhz ||
        (r.boost_min_mhz > 0 && r.form != FormFactor::Laptop)) {
        throw std::runtime_error("boost_min_mhz is not a laptop's bottom of its boost range");
    }
    if (r.sources.empty()) throw std::runtime_error("no source");
    for (const auto& s : r.sources) {
        if (s.rfind("https://", 0) != 0)
            throw std::runtime_error("source \"" + s + "\" is not https");
    }
    return r;
}

int memory_clock_khz_of(double gbps) { return static_cast<int>(std::llround(gbps * 1e6 / 2.0)); }

} // namespace

std::vector<GpuFactsRow> parse_gpu_facts(std::string_view json_text) {
    const Json root = Json::parse(json_text);
    if (root.at("schema").get<int>() != 1) throw std::runtime_error("gpu facts: schema is not 1");
    std::vector<GpuFactsRow> rows;
    std::unordered_set<std::uint32_t> seen;
    for (const Json& j : root.at("gpus")) {
        try {
            rows.push_back(parse_row(j));
        } catch (const std::exception& e) {
            throw std::runtime_error("gpu facts: row " + j.value("id", std::string("?")) + ": " +
                                     e.what());
        }
        if (!seen.insert(rows.back().pci_device_id).second) {
            throw std::runtime_error("gpu facts: id " + j.at("id").get<std::string>() +
                                     " appears twice");
        }
    }
    return rows;
}

std::span<const GpuFactsRow> gpu_facts_table() {
    static const std::vector<GpuFactsRow> table = parse_gpu_facts(embedded::kGpuFactsJson);
    return table;
}

const GpuFactsRow* find_gpu_facts(std::uint32_t pci_device_id) {
    for (const GpuFactsRow& r : gpu_facts_table()) {
        if (r.pci_device_id == pci_device_id) return &r;
    }
    return nullptr;
}

// The decision is k = round(16 log2 x); the rest is the exact power 2^(k/16) of a binary k/16.
// Different libm implementations of log2 and exp2 (glibc, MSVC, a CUDA host compiler's) differ by
// at most a few ulp, and that cannot move the snap of an integer-kHz input, which is every input
// machine_facts passes (a driver clock as int, 1 <= x < 2^31):
//  - log2: the snap turns at the midpoints 2^((k+1/2)/16), which are irrational, so no integer
//    lies on one. Over all integers below 2^31 the closest one, 1767116489 next to 2^(983/32),
//    gives |16 log2 x - (k+1/2)| = 3.8e-10 (exact arithmetic, 60 digits), while a few ulp of
//    16 log2 x <= 496 is about 1e-13: more than three orders of magnitude to spare.
//  - exp2: the caller rounds 2^(k/16) to an integer kHz (llround), and for 0 <= k < 496 no grid
//    point lies within 1.1e-3 of a half-integer (closest: k = 86, 41.49887), against an error of
//    a few ulp: about 2e-6 at most, at the 2^31 end.
// A non-integer input could sit on a midpoint, and then the libm decides; no decision passes one.
double snap_to_fact_grid(double x) noexcept {
    if (!(x > 0)) return 0;
    const double k = std::floor(16.0 * std::log2(x) + 0.5);
    return std::exp2(k / 16.0);
}

MachineFacts machine_facts(const DriverReport& driver) {
    return machine_facts(driver, gpu_facts_table());
}

MachineFacts machine_facts(const DriverReport& driver, std::span<const GpuFactsRow> table) {
    const GpuFactsRow* row = nullptr;
    if (driver.pci_device_id != 0) {
        for (const GpuFactsRow& r : table) {
            if (r.pci_device_id == driver.pci_device_id) row = &r;
        }
    }
    // A row names the device only for its architecture (the driver's compute capability, where
    // reported): an id reused by another architecture is unknown.
    if (row != nullptr && driver.major != 0 &&
        (row->major != driver.major || row->minor != driver.minor)) {
        row = nullptr;
    }
    MachineFacts f;
    if (row != nullptr) {
        f.source              = FactSource::Verified;
        f.sku                 = row->name;
        f.segment             = row->segment;
        f.form                = row->form;
        f.reference_clock_khz = row->boost_mhz * 1000;
        f.memory_bus_bits     = row->bus_bits;
        f.memory_clock_khz    = memory_clock_khz_of(row->gbps);
        f.tdp_w               = row->tdp_w;
        f.vram_gib            = row->vram_gib;
        f.pcie_gen            = row->pcie_gen;
        f.mma_slow            = row->mma_slow;
        // A laptop's clocks follow the OEM's power limit (80 to 150 W for one name): it is priced
        // at the bottom of the published boost range (principle 7, the unknown is conservative),
        // or, where the table lacks the bottom, at the lower of the row's top and the driver's
        // clock on the grid. A device reporting other SMs than the row (a MIG slice) is a part of
        // the board, and sees that part of its bandwidth.
        if (row->form == FormFactor::Laptop) {
            f.source = FactSource::Extrapolated;
            if (row->boost_min_mhz > 0) {
                f.reference_clock_khz = row->boost_min_mhz * 1000;
            } else {
                f.driver_read = true;
                if (driver.sm_clock_khz > 0) {
                    const int snapped =
                        static_cast<int>(std::llround(snap_to_fact_grid(driver.sm_clock_khz)));
                    f.reference_clock_khz = std::min(f.reference_clock_khz, snapped);
                }
            }
        }
        if (driver.sms > 0 && driver.sms != row->sms) {
            f.source = FactSource::Extrapolated;
            if (driver.sms < row->sms) f.share = static_cast<double>(driver.sms) / row->sms;
        }
        // A board sharing its id with a variant on another memory bus or a slower memory (an RTX
        // 4070 with GDDR6 under the GDDR6X board's id, an 8 GB RTX 3060 on a 128-bit bus) prices
        // its memory at the driver's: the bus width it reports, and a memory clock more than half
        // a grid step (2.2 %) below the row's on the 2^(1/16) grid. A faster clock than the row's
        // (an overclock) and a driver's rounding (14001 MHz against 14000) are not read, nor a
        // value no memory has (a bus width that is not a whole number of 32-bit channels, a clock
        // under 100 MHz): it is as good as unreported. Only a whole board off the laptop rows is
        // read so: a slice's memory values are its part of the board's, which the share above
        // already prices, and a laptop is priced from its published range whatever its driver
        // reports (two laptops of one name fingerprint alike).
        const bool whole_board =
            row->form != FormFactor::Laptop && (driver.sms <= 0 || driver.sms == row->sms);
        if (whole_board && driver.memory_bus_bits >= 32 && driver.memory_bus_bits % 32 == 0 &&
            driver.memory_bus_bits != row->bus_bits) {
            f.source          = FactSource::Extrapolated;
            f.driver_read     = true;
            f.memory_bus_bits = driver.memory_bus_bits;
        }
        if (whole_board && driver.memory_clock_khz >= 100000 &&
            std::int64_t{driver.memory_clock_khz} * 1000 < std::int64_t{f.memory_clock_khz} * 978) {
            f.source      = FactSource::Extrapolated;
            f.driver_read = true;
            f.memory_clock_khz =
                static_cast<int>(std::llround(snap_to_fact_grid(driver.memory_clock_khz)));
        }
        return f;
    }
    // Unknown: the driver's values on the 2^(1/16) grid, so an overclock or a driver's rounding
    // lands on the same point as the stock board. The bus width is discrete and kept.
    f.reference_clock_khz =
        static_cast<int>(std::llround(snap_to_fact_grid(std::max(driver.sm_clock_khz, 0))));
    f.memory_clock_khz =
        static_cast<int>(std::llround(snap_to_fact_grid(std::max(driver.memory_clock_khz, 0))));
    f.memory_bus_bits = std::max(driver.memory_bus_bits, 0);
    f.driver_read     = true;
    f.source = f.reference_clock_khz > 0 || f.memory_clock_khz > 0 ? FactSource::Extrapolated
                                                                   : FactSource::None;
    return f;
}

std::string describe(const MachineFacts& f) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s%s%s ref %d MHz %.0f GB/s%s",
                  std::string(to_string(f.source)).c_str(), f.sku.empty() ? "" : " ", f.sku.c_str(),
                  f.reference_clock_khz / 1000, f.dram_gbps(), f.mma_slow ? " mma_slow" : "");
    return buf;
}

} // namespace ninferno
