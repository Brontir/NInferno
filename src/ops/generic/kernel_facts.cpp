#include "ops/generic/kernel_facts.h"

#include "core/device_class.h"
#include "ops/generic/embedded_kernel_facts.h" // generated: the committed snapshots' text

#include <cctype>
#include <charconv>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninferno::ops::detail::generic::kernel_facts {

namespace {

struct Snapshot {
    SnapshotHeader header;
    std::unordered_map<std::string, KernelFacts> rows; // by row_key
};

// The columns of a schema 2 snapshot (tools/kernel_facts/kernel_facts.py COLUMNS), the order
// parse() reads the loop's counts in.
constexpr std::string_view kColumns =
    "family decoder warps_m warps_k m_tiles n_tiles cluster epilogue latency add_sums registers "
    "stack local shared loops k_iter mma mma_ones mma_shape ldg ldg_bytes ld ld_bytes ldgsts "
    "ldgsts_bytes lds lds_bytes sts sts_bytes local local_bytes shfl int fp uniform other total";
// The length of the resources-only (schema 1) columns line: kColumns up to "shared".
constexpr std::size_t kResourceColumnsLength = kColumns.find(" loops ");

// One string per key: the row's spelling without the family column.
std::string row_key(const RmmaKernelKey& k) {
    std::string s = k.decoder;
    for (const int v : {k.warps_m, k.warps_k, k.m_tiles, k.n_tiles, k.cluster, k.epilogue,
                        k.latency ? 1 : 0, k.add_sums ? 1 : 0}) {
        s += ' ';
        s += std::to_string(v);
    }
    return s;
}

// A snapshot's text (tools/kernel_facts/kernel_facts.py: header lines `name value`, rows
// `rmma <decoder> <5 schedule ints> <epilogue> <latency> <add_sums> <registers> <stack> <local>
// <shared>`). A malformed line is a build error of the data, not a runtime condition: it throws.
Snapshot parse(std::string_view text) {
    Snapshot s;
    std::optional<bool> with_loops; // what the columns line, read before any row, promises
    std::istringstream in{std::string(text)};
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string first;
        fields >> first;
        if (first == "rmma") {
            if (!with_loops) {
                throw std::runtime_error("kernel facts: a row before the columns line: " + line);
            }
            RmmaKernelKey k;
            int latency = 0, add_sums = 0;
            KernelFacts f;
            fields >> k.decoder >> k.warps_m >> k.warps_k >> k.m_tiles >> k.n_tiles >> k.cluster >>
                k.epilogue >> latency >> add_sums >> f.registers >> f.stack_bytes >>
                f.local_bytes >> f.static_shared_bytes;
            if (!fields) throw std::runtime_error("kernel facts: malformed row: " + line);
            // Schema 2: the K loop's counts follow (tools/kernel_facts/sass_counts.py COLUMNS),
            // on every row of a file whose columns line names them and on none of another's.
            if (*with_loops) {
                LoopCounts l;
                fields >> l.loops >> l.k_iter >> l.mma >> l.mma_ones >> l.mma_shape >> l.ldg >>
                    l.ldg_bytes >> l.ld >> l.ld_bytes >> l.ldgsts >> l.ldgsts_bytes >> l.lds >>
                    l.lds_bytes >> l.sts >> l.sts_bytes >> l.local >> l.local_bytes >> l.shfl >>
                    l.int_alu >> l.fp_alu >> l.uniform >> l.other >> l.total;
                if (!fields) throw std::runtime_error("kernel facts: malformed loop: " + line);
                f.loop = std::move(l);
            }
            if (fields >> std::ws; !fields.eof()) {
                throw std::runtime_error("kernel facts: fields past the columns': " + line);
            }
            k.latency  = latency != 0;
            k.add_sums = add_sums != 0;
            s.rows.emplace(row_key(k), f);
        } else {
            std::string rest;
            std::getline(fields, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            if (first == "arch")
                s.header.arch = rest;
            else if (first == "nvcc")
                s.header.nvcc = rest;
            else if (first == "flags")
                s.header.flags = rest;
            else if (first == "source")
                s.header.source = rest;
            else if (first == "kernels")
                s.header.kernels = std::stoul(rest);
            else if (first == "columns") {
                // The columns are read by position: the file spells either list exactly.
                if (rest == kColumns)
                    with_loops = true;
                else if (rest == kColumns.substr(0, kResourceColumnsLength))
                    with_loops = false;
                else
                    throw std::runtime_error("kernel facts: unexpected columns: " + rest);
            }
        }
    }
    if (s.header.arch.empty() || s.header.kernels != s.rows.size()) {
        throw std::runtime_error("kernel facts: a snapshot header disagrees with its rows");
    }
    return s;
}

// Every committed snapshot, parsed once.
const std::vector<std::unique_ptr<Snapshot>>& snapshots() {
    static const std::vector<std::unique_ptr<Snapshot>> all = [] {
        std::vector<std::unique_ptr<Snapshot>> v;
        for (const auto& e : embedded::kKernelFactsSnapshots) {
            v.push_back(std::make_unique<Snapshot>(parse(e.text)));
            if (v.back()->header.arch != e.arch) {
                throw std::runtime_error(std::string("kernel facts: sm_") + e.arch +
                                         " names arch " + v.back()->header.arch);
            }
        }
        return v;
    }();
    return all;
}

const Snapshot* find(std::string_view arch) {
    for (const auto& s : snapshots()) {
        if (s->header.arch == arch) return s.get();
    }
    return nullptr;
}

// A label's number and kind: "120a" -> 120, 'a'; "89" -> 89, 0.
std::pair<int, char> label_parts(std::string_view label) {
    int number      = 0;
    const auto end  = std::from_chars(label.data(), label.data() + label.size(), number).ptr;
    const char kind = end != label.data() + label.size() ? *end : '\0';
    return {number, kind};
}

// Whether the SASS of a build architecture label runs on a device of this capability: an
// architecture-specific target on its exact capability, a family target on its major, a plain
// number on its major from its minor up (cubin compatibility).
bool sass_runs(std::string_view label, int major, int minor) {
    const auto [number, kind] = label_parts(label);
    if (kind == 'a') return number == major * 10 + minor;
    if (kind == 'f') return number / 10 == major;
    return number / 10 == major && number % 10 <= minor;
}

} // namespace

std::vector<std::string_view> snapshot_archs() {
    std::vector<std::string_view> v;
    for (const auto& s : snapshots()) v.push_back(s->header.arch);
    return v;
}

std::optional<SnapshotHeader> snapshot_header(std::string_view arch) {
    const Snapshot* s = find(arch);
    if (s == nullptr) return std::nullopt;
    return s->header;
}

std::optional<KernelFacts> rmma_facts(std::string_view arch, const RmmaKernelKey& key) {
    const Snapshot* s = find(arch);
    if (s == nullptr) return std::nullopt;
    const auto it = s->rows.find(row_key(key));
    if (it == s->rows.end()) return std::nullopt;
    return it->second;
}

std::string release_arch(int major, int minor) {
    const int cc = major * 10 + minor;
    std::string best;
    int best_number = -1;
    for (const auto& s : snapshots()) {
        const auto [number, kind] = label_parts(s->header.arch);
        if (kind == 'a' && number == cc) return s->header.arch;
    }
    for (const auto& s : snapshots()) {
        const auto [number, kind] = label_parts(s->header.arch);
        if (kind == 'f' && number / 10 == major) return s->header.arch;
    }
    for (const auto& s : snapshots()) {
        const auto [number, kind] = label_parts(s->header.arch);
        if (kind == '\0' && number / 10 == major && number <= cc && number > best_number) {
            best        = s->header.arch;
            best_number = number;
        }
    }
    return best;
}

std::string_view build_arch() {
#ifdef NINFERNO_BUILD_ARCH_LABEL
    return NINFERNO_BUILD_ARCH_LABEL;
#else
    static const std::string label = std::to_string(NINFERNO_BUILD_ARCH);
    return label;
#endif
}

std::optional<std::string> sass_arch() {
    if (const DeviceClass* s = substituted_device_class()) {
        std::string arch = release_arch(s->major, s->minor);
        if (arch.empty()) return std::nullopt;
        return arch;
    }
    try {
        const DeviceClass& d = device_class();
        if (sass_runs(build_arch(), d.major, d.minor)) return std::string(build_arch());
    } catch (const std::exception&) {} // no current device: no kernels run
    return std::nullopt;
}

namespace {
thread_local LiveCheckSink* t_live = nullptr;
}

ScopedLiveCheck::ScopedLiveCheck(LiveCheckSink& sink) noexcept : previous_(t_live) {
    t_live = &sink;
}

ScopedLiveCheck::~ScopedLiveCheck() { t_live = previous_; }

LiveCheckSink* live_check_target() noexcept { return t_live; }

std::string describe(const LiveRead& r) {
    std::ostringstream out;
    out << "sm_" << (r.arch.empty() ? "?" : r.arch) << ' ' << row_key(r.key) << ": snapshot ";
    if (r.snapshot) {
        out << r.snapshot->registers << " registers, " << r.snapshot->stack_bytes << " B stack, "
            << r.snapshot->static_shared_bytes << " B shared, " << r.snapshot_ctas << " CTAs/SM";
    } else {
        out << "none";
    }
    out << "; live " << r.live.registers << " registers, " << r.live.stack_bytes << " B stack, "
        << r.live.static_shared_bytes << " B shared, " << r.live_ctas << " CTAs/SM (" << r.threads
        << " threads, " << r.dynamic_shared_bytes << " B dynamic)";
    return out.str();
}

std::string canonical_type_name(std::string_view raw) {
    std::string s(raw);
    for (const std::string_view drop : {"ninferno::ops::detail::generic::", "struct ", "class ",
                                        "enum ", "(int)", "(bool)", "(unsigned int)", "(long)"}) {
        for (std::size_t at; (at = s.find(drop)) != std::string::npos;) s.erase(at, drop.size());
    }
    std::string out;
    for (const char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c))) out += c;
    }
    return out;
}

} // namespace ninferno::ops::detail::generic::kernel_facts
