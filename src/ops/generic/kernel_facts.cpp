#include "ops/generic/kernel_facts.h"

#include "core/device_class.h"
#include "ops/generic/embedded_kernel_facts.h" // generated: the committed snapshots' text

#include <cctype>
#include <charconv>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninferno::ops::detail::generic::kernel_facts {

namespace {

// One section family's rows (#785): its header lines and its rows by their key's spelling.
struct Section {
    SectionHeader header;
    bool has_nvcc = false, has_kernels = false;
    std::map<std::string, KernelFacts> rows; // by the key as the file spells it
};

struct Snapshot {
    SnapshotHeader header;
    std::unordered_map<std::string, KernelFacts> rows;    // by row_key
    std::map<std::string, Section, std::less<>> sections; // by family
};

// The columns of a schema 4 snapshot (tools/kernel_facts/kernel_facts.py COLUMNS), the order
// parse() reads the loop's counts and the static schedule in.
constexpr std::string_view kColumns =
    "family decoder warps_m warps_k m_tiles n_tiles cluster epilogue latency sums registers "
    "stack local shared loops k_iter mma mma_ones mma_shape ldg ldg_bytes ld ld_bytes ldgsts "
    "ldgsts_bytes lds lds_bytes sts sts_bytes local local_bytes shfl int fp uniform other total "
    "stall_sum sb_waits sb_waits_local outside_total outside_stall outside_waits outside_bar";
// The length of the one shorter columns line kColumns begins with: the resources alone (schema 1,
// up to "shared"). The older loop-without-schedule columns (schema 2) are refused.
constexpr std::size_t kResourceColumnsLength = kColumns.find(" loops ");
// The loop's columns of every schema 4 row (from "loops" on), and the pipe counts a section's
// rows carry after them (kernel_facts.py Family.schema_columns, sass_counts.py PIPE_FIELDS).
constexpr std::string_view kLoopColumns = kColumns.substr(kResourceColumnsLength);
constexpr std::string_view kPipeColumns = " i2f i2fp ffma fmul fadd imad bar";

// A section family's columns up to the resources (kernel_facts.py FAMILIES: "family", the key,
// the resources), and how many key fields lead as text (the codec) before the integers.
struct SectionFamily {
    std::string_view name;
    std::string_view head_columns;
    int text_keys = 0;
    int int_keys  = 0;
};

constexpr SectionFamily kSections[] = {
    {"mmq", "family type J fallback prec registers stack local shared", 0, 4},
    {"w4a8",
     "family codec block_m block_n warps_m warps_n epilogue flavor stages producers registers "
     "stack local shared",
     1, 8},
    {"gemma4", "family kernel D G slice rows Bc partial fp8 chain registers stack local shared", 1,
     8},
};
static_assert(std::size(kSections) == std::size(kSectionFamilies));

const SectionFamily* section_family(std::string_view name) {
    for (const auto& f : kSections) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

// What a snapshot's rows carry past the resources, by its columns line.
enum class Columns { Resources, Schedule };

// The K loop's counts and the static schedule (schema 4) of a row, from `fields` at the loop's
// first column; the pipe counts after them where `pipes`.
LoopCounts read_loop(std::istringstream& fields, bool pipes, const std::string& line) {
    LoopCounts l;
    fields >> l.loops >> l.k_iter >> l.mma >> l.mma_ones >> l.mma_shape >> l.ldg >> l.ldg_bytes >>
        l.ld >> l.ld_bytes >> l.ldgsts >> l.ldgsts_bytes >> l.lds >> l.lds_bytes >> l.sts >>
        l.sts_bytes >> l.local >> l.local_bytes >> l.shfl >> l.int_alu >> l.fp_alu >> l.uniform >>
        l.other >> l.total;
    if (!fields) throw std::runtime_error("kernel facts: malformed loop: " + line);
    StaticSchedule d;
    fields >> d.stall_sum >> d.sb_waits >> d.sb_waits_local >> d.outside_total >> d.outside_stall >>
        d.outside_waits >> d.outside_bar;
    if (!fields) throw std::runtime_error("kernel facts: malformed schedule: " + line);
    l.schedule = d;
    if (pipes) {
        PipeCounts p;
        fields >> p.i2f >> p.i2fp >> p.ffma >> p.fmul >> p.fadd >> p.imad >> p.bar;
        if (!fields) throw std::runtime_error("kernel facts: malformed pipe counts: " + line);
        l.pipes = p;
    }
    return l;
}

// One row of a section family under its columns: the key's fields (spelled back with single
// spaces as the map's key), the resources, the loop where the columns name it.
void read_section_row(Section& s, const SectionFamily& f, Columns columns,
                      std::istringstream& fields, const std::string& line) {
    std::string key;
    for (int i = 0; i < f.text_keys + f.int_keys; ++i) {
        std::string v;
        fields >> v;
        if (!fields) throw std::runtime_error("kernel facts: malformed row: " + line);
        if (i >= f.text_keys) {
            int n         = 0;
            const auto r  = std::from_chars(v.data(), v.data() + v.size(), n);
            const bool ok = r.ec == std::errc{} && r.ptr == v.data() + v.size();
            if (!ok) throw std::runtime_error("kernel facts: malformed key: " + line);
        }
        if (i != 0) key += ' ';
        key += v;
    }
    KernelFacts k;
    fields >> k.registers >> k.stack_bytes >> k.local_bytes >> k.static_shared_bytes;
    if (!fields) throw std::runtime_error("kernel facts: malformed row: " + line);
    if (columns != Columns::Resources) k.loop = read_loop(fields, true, line);
    if (fields >> std::ws; !fields.eof()) {
        throw std::runtime_error("kernel facts: fields past the columns': " + line);
    }
    if (!s.rows.emplace(std::move(key), k).second) {
        throw std::runtime_error("kernel facts: a repeated row: " + line);
    }
}

// One string per key: the row's spelling without the family column.
std::string row_key(const RmmaKernelKey& k) {
    std::string s = k.decoder;
    for (const int v : {k.warps_m, k.warps_k, k.m_tiles, k.n_tiles, k.cluster, k.epilogue,
                        k.latency ? 1 : 0, k.sums}) {
        s += ' ';
        s += std::to_string(v);
    }
    return s;
}

// A snapshot's text (tools/kernel_facts/kernel_facts.py: header lines `name value`, rows
// `rmma <decoder> <5 schedule ints> <epilogue> <latency> <sums> <registers> <stack> <local>
// <shared>`). A malformed line is a build error of the data, not a runtime condition: it throws.
// After the register-MMA rows, the sections (#785): a `family <name>` line, then its
// family_nvcc, family_kernels and family_columns lines and rows of that family alone.
Snapshot parse(std::string_view text) {
    Snapshot s;
    std::optional<Columns> columns; // what the columns line, read before any row, promises
    Section* section            = nullptr; // the section being read, after its `family` line
    const SectionFamily* family = nullptr;
    std::optional<Columns> section_columns;
    std::istringstream in{std::string(text)};
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string first;
        fields >> first;
        std::string rest;
        if (first == "family") {
            fields >> rest;
            family = section_family(rest);
            if (family == nullptr || s.sections.count(rest) != 0 ||
                (fields >> std::ws, !fields.eof())) {
                throw std::runtime_error("kernel facts: unexpected or repeated section: " + line);
            }
            section = &s.sections[rest];
            section_columns.reset();
            continue;
        }
        if (section != nullptr) {
            if (first == family->name) {
                if (!section_columns) {
                    throw std::runtime_error("kernel facts: a row before its columns: " + line);
                }
                read_section_row(*section, *family, *section_columns, fields, line);
                continue;
            }
            std::getline(fields, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            if (first == "family_nvcc") {
                section->header.nvcc = rest;
                section->has_nvcc    = !rest.empty();
            } else if (first == "family_kernels") {
                section->header.kernels = std::stoul(rest);
                section->has_kernels    = true;
            } else if (first == "family_columns") {
                const std::string loop = std::string(family->head_columns) +
                                         std::string(kLoopColumns) + std::string(kPipeColumns);
                if (rest == loop)
                    section_columns = Columns::Schedule;
                else if (rest == family->head_columns)
                    section_columns = Columns::Resources;
                else
                    throw std::runtime_error("kernel facts: unexpected family_columns: " + rest);
            } else {
                throw std::runtime_error("kernel facts: unexpected line in a section: " + line);
            }
            continue;
        }
        if (first == "rmma") {
            if (!columns) {
                throw std::runtime_error("kernel facts: a row before the columns line: " + line);
            }
            RmmaKernelKey k;
            int latency = 0;
            KernelFacts f;
            fields >> k.decoder >> k.warps_m >> k.warps_k >> k.m_tiles >> k.n_tiles >> k.cluster >>
                k.epilogue >> latency >> k.sums >> f.registers >> f.stack_bytes >> f.local_bytes >>
                f.static_shared_bytes;
            if (!fields) throw std::runtime_error("kernel facts: malformed row: " + line);
            // Schema 4: the K loop's counts follow (tools/kernel_facts/sass_counts.py COLUMNS),
            // then the static schedule, on every row of a file whose columns line names them and
            // on none of a resources-only one's.
            if (*columns != Columns::Resources) f.loop = read_loop(fields, false, line);
            if (fields >> std::ws; !fields.eof()) {
                throw std::runtime_error("kernel facts: fields past the columns': " + line);
            }
            k.latency = latency != 0;
            s.rows.emplace(row_key(k), f);
        } else {
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
                // The columns are read by position: the file spells one of the lists exactly.
                if (rest == kColumns)
                    columns = Columns::Schedule;
                else if (rest == kColumns.substr(0, kResourceColumnsLength))
                    columns = Columns::Resources;
                else
                    throw std::runtime_error("kernel facts: unexpected columns: " + rest);
            }
        }
    }
    if (s.header.arch.empty() || s.header.kernels != s.rows.size()) {
        throw std::runtime_error("kernel facts: a snapshot header disagrees with its rows");
    }
    for (const auto& [name, sec] : s.sections) {
        if (!sec.has_nvcc || !sec.has_kernels || sec.header.kernels != sec.rows.size()) {
            throw std::runtime_error("kernel facts: the " + name +
                                     " section's header disagrees with its rows");
        }
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

const Section* find_section(std::string_view arch, std::string_view family) {
    const Snapshot* s = find(arch);
    if (s == nullptr) return nullptr;
    const auto it = s->sections.find(family);
    return it == s->sections.end() ? nullptr : &it->second;
}

std::optional<KernelFacts> section_facts(std::string_view arch, std::string_view family,
                                         const std::string& key) {
    const Section* s = find_section(arch, family);
    if (s == nullptr) return std::nullopt;
    const auto it = s->rows.find(key);
    if (it == s->rows.end()) return std::nullopt;
    return it->second;
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

bool snapshot_text_accepted(std::string_view text) {
    try {
        (void)parse(text);
        return true;
    } catch (const std::runtime_error&) { return false; }
}

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

std::optional<SectionHeader> section_header(std::string_view arch, std::string_view family) {
    const Section* s = find_section(arch, family);
    if (s == nullptr) return std::nullopt;
    return s->header;
}

std::vector<std::pair<std::string, KernelFacts>> section_rows(std::string_view arch,
                                                              std::string_view family) {
    std::vector<std::pair<std::string, KernelFacts>> v;
    if (const Section* s = find_section(arch, family)) v.assign(s->rows.begin(), s->rows.end());
    return v;
}

std::optional<KernelFacts> mmq_facts(std::string_view arch, const MmqKernelKey& key) {
    return section_facts(arch, "mmq",
                         std::to_string(key.type) + ' ' + std::to_string(key.J) + ' ' +
                             (key.fallback ? "1" : "0") + ' ' + std::to_string(key.prec));
}

std::optional<KernelFacts> w4a8_facts(std::string_view arch, const W4a8KernelKey& key) {
    std::string k = key.codec;
    for (const int v : {key.block_m, key.block_n, key.warps_m, key.warps_n, key.epilogue,
                        key.flavor, key.stages, key.producers}) {
        k += ' ';
        k += std::to_string(v);
    }
    return section_facts(arch, "w4a8", k);
}

std::optional<KernelFacts> gemma4_facts(std::string_view arch, const Gemma4KernelKey& key) {
    std::string k = key.kernel;
    for (const int v : {key.D, key.G, key.slice, key.rows, key.Bc, key.partial ? 1 : 0,
                        key.fp8 ? 1 : 0, key.chain ? 1 : 0}) {
        k += ' ';
        k += std::to_string(v);
    }
    return section_facts(arch, "gemma4", k);
}

std::optional<KernelFacts> rmma_facts(std::string_view arch, const RmmaKernelKey& key) {
    const Snapshot* s = find(arch);
    if (s == nullptr) return std::nullopt;
    const auto it = s->rows.find(row_key(key));
    if (it == s->rows.end()) return std::nullopt;
    return it->second;
}

std::vector<KernelFacts> rmma_rows(std::string_view arch) {
    std::vector<KernelFacts> v;
    if (const Snapshot* s = find(arch)) {
        v.reserve(s->rows.size());
        for (const auto& [key, facts] : s->rows) v.push_back(facts);
    }
    return v;
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

std::optional<std::string> sass_arch_of(const DeviceClass& d, bool substituted) {
    if (substituted) {
        std::string arch = release_arch(d.major, d.minor);
        if (arch.empty()) return std::nullopt;
        return arch;
    }
    if (sass_runs(build_arch(), d.major, d.minor)) return std::string(build_arch());
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
