// The one loader of the per-GPU-class route table (include/ninferno/ops/route_table.h). It parses a
// ninferno-route-table/1 file once and serves both consumers from the same entries: the generic
// projection path's "bands" / "prefill" (ops/generic/route_table.h) and the shape-owned Linear
// tables' "route_as" (ops/common/shape_routes.h).
#include "core/per_device.h"
#include "ninferno/ops/route_table.h"
#include "core/sha256.h"
#include "core/sys.h"

#include "core/device_class.h"
#include "ops/common/shape_routes.h"
#include "ops/generic/route_table.h"
#include "ops/moe/moe_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <array>
#include <mutex>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef NINFERNO_DEFAULT_ROUTE_DIR
#define NINFERNO_DEFAULT_ROUTE_DIR ""
#endif

namespace ninferno::ops {
namespace {

using detail::RouteCompute;
using detail::generic::Epilogue;
using detail::generic::Route;
using detail::generic::RouteBand;
using detail::generic::RouteEntry;
using Json = nlohmann::json;

// The first call width the prefill route profile makes token-invariant.
constexpr std::int32_t kPrefillMinTokens = 17;

std::uint64_t key_of(QType q, RouteCompute c, Epilogue e, std::int32_t n, std::int32_t k) {
    return (static_cast<std::uint64_t>(q) << 56) | (static_cast<std::uint64_t>(c) << 52) |
           (static_cast<std::uint64_t>(e) << 48) |
           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(n) & 0xffffffU) << 24) |
           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(k) & 0xffffffU));
}

// "For a call of t_min..t_max tokens, ask the shape's table at `as` tokens."
struct RouteAsBand {
    std::int32_t t_min;
    std::int32_t t_max;
    std::int32_t as;
};

// Both formats and the whole geometry, compared exactly (no packing to truncate).
using MoeKey = std::tuple<int, int, std::int32_t, std::int32_t, std::int32_t, std::int32_t>;
MoeKey moe_key_of(const detail::moe::MoeRouteKey& k) {
    return {static_cast<int>(k.qtype), static_cast<int>(k.down_qtype), k.experts, k.top_k,
            k.hidden, k.inter};
}

struct Table {
    std::map<MoeKey, detail::moe::MoeRouteEntry> moe;                     // generic MoE layers
    std::unordered_set<std::uint64_t> problems;                           // every entry's key
    std::unordered_map<std::uint64_t, RouteEntry> generic;                // bands / prefill
    std::unordered_map<std::uint64_t, std::vector<RouteAsBand>> route_as; // shape-owned tables
    RouteTableStatus status;
    std::optional<Board> board; // the board it was measured on (gpu.board), when recorded
};

std::optional<QType> parse_qtype(const std::string& s) {
    if (s == "bf16") return QType::BF16;
    if (s == "fp8") return QType::FP8_E4M3FN_ROW_BF16;
    if (s == "nvfp4") return QType::NVFP4;
    if (s == "q4") return QType::Q4_G64_FP16;
    if (s == "q5") return QType::Q5_G64_FP16;
    if (s == "q6") return QType::Q6_G64_FP16;
    if (s == "q8") return QType::Q8_G32_FP16;
    if (s == "mxfp4") return QType::MXFP4;
    if (s == "fp8r32") return QType::FP8_E4M3FN_ROW_F32;
    if (s == "fp8b128") return QType::FP8_E4M3FN_B128_F32;
    if (s == "u4zp") return QType::U4_G128_ZP_FP16;
    if (s == "u4zp32") return QType::U4_G32_ZP_FP16;
    if (s == "u4zp64") return QType::U4_G64_ZP_FP16;
    if (s == "q3g128") return QType::Q3_G128_FP16;
    if (s == "q3g64") return QType::Q3_G64_FP16;
    if (s == "q4k") return QType::GGML_Q4_K;
    if (s == "q5k") return QType::GGML_Q5_K;
    if (s == "q6k") return QType::GGML_Q6_K;
    if (s == "q3k") return QType::GGML_Q3_K;
    if (s == "q2k") return QType::GGML_Q2_K;
    if (s == "iq4xs") return QType::GGML_IQ4_XS;
    if (s == "iq3xxs") return QType::GGML_IQ3_XXS;
    if (s == "iq3s") return QType::GGML_IQ3_S;
    if (s == "q40") return QType::GGML_Q4_0;
    if (s == "q41") return QType::GGML_Q4_1;
    if (s == "q50") return QType::GGML_Q5_0;
    if (s == "q51") return QType::GGML_Q5_1;
    if (s == "iq4nl") return QType::GGML_IQ4_NL;
    if (s == "iq2xxs") return QType::GGML_IQ2_XXS;
    if (s == "iq2xs") return QType::GGML_IQ2_XS;
    if (s == "iq2s") return QType::GGML_IQ2_S;
    if (s == "iq1s") return QType::GGML_IQ1_S;
    if (s == "iq1m") return QType::GGML_IQ1_M;
    if (s == "tq10") return QType::GGML_TQ1_0;
    if (s == "tq20") return QType::GGML_TQ2_0;
    if (s == "q10") return QType::GGML_Q1_0;
    if (s == "q20") return QType::GGML_Q2_0;
    return std::nullopt;
}

std::optional<RouteCompute> parse_compute(const std::string& s) {
    if (s == "a16") return RouteCompute::A16;
    if (s == "a4") return RouteCompute::A4;
    if (s == "a8") return RouteCompute::A8;
    return std::nullopt;
}

std::optional<Epilogue> parse_epilogue(const std::string& s) {
    if (s == "store") return Epilogue::Store;
    if (s == "add") return Epilogue::AddResidual;
    if (s == "swiglu") return Epilogue::SwiGlu;
    return std::nullopt;
}

[[noreturn]] void refuse(const std::string& what, const std::string& where) {
    throw std::runtime_error("route table: " + what + " in " + where);
}

Route require_route(const Json& value, const std::string& where) {
    const auto name = value.get<std::string>();
    const auto r    = detail::generic::parse_route(name);
    if (!r) refuse("unknown route '" + name + "'", where);
    return *r;
}

// Sorts [t_min, t_max, ...] bands and refuses empty or overlapping ones.
template <class Band>
void check_bands(std::vector<Band>& bands, const std::string& where) {
    for (const Band& b : bands) {
        if (b.t_min < 1 || b.t_max < b.t_min) refuse("an empty band or one below T = 1", where);
    }
    std::sort(bands.begin(), bands.end(),
              [](const Band& a, const Band& b) { return a.t_min < b.t_min; });
    for (std::size_t i = 1; i < bands.size(); ++i) {
        if (bands[i].t_min <= bands[i - 1].t_max) refuse("overlapping bands", where);
    }
}

void parse_entry(const Json& e, const std::string& path, Table& table) {
    const std::string where = path + " entry " + e.dump();
    if (!e.is_object()) refuse("an entry that is not an object", where);
    const auto q       = parse_qtype(e.at("format").get<std::string>());
    const auto compute = parse_compute(e.at("compute").get<std::string>());
    const auto epi     = parse_epilogue(e.value("epilogue", std::string("store")));
    const auto n       = e.at("n").get<std::int32_t>();
    const auto k       = e.at("k").get<std::int32_t>();
    if (!q || !compute || !epi || n < 1 || k < 1) refuse("a bad key", where);
    // A compute class the weight format does not have would never be asked.
    if ((*compute == RouteCompute::A4 && *q != QType::NVFP4) ||
        (*compute == RouteCompute::A8 && *q != QType::FP8_E4M3FN_ROW_BF16)) {
        refuse("a compute class the weight format does not have", where);
    }
    const auto key = key_of(*q, *compute, *epi, n, k);
    if (!table.problems.insert(key).second) refuse("a duplicate problem", where);
    ++table.status.entries;

    // The generic projection path: bands per token range, one prefill route.
    if (e.contains("bands") || e.contains("prefill")) {
        if (*compute == RouteCompute::A8) refuse("bands on a8 (the generic path has no A8 schedule)", where);
        RouteEntry entry{*q, *compute, *epi, n, k, {}, std::nullopt, {}, false};
        for (const Json& b : e.value("bands", Json::array())) {
            entry.bands.push_back(
                {b.at(0).get<std::int32_t>(), b.at(1).get<std::int32_t>(), require_route(b.at(2), where)});
        }
        check_bands(entry.bands, where);
        if (e.contains("prefill")) {
            // Chunk invariance: under the prefill profile every call of 17 tokens or more takes
            // this route, so it must sum K in the one order every T shares.
            const Route r = require_route(e.at("prefill"), where);
            if (!detail::generic::route_is_token_invariant(r)) {
                refuse("a prefill route that is not token-invariant (it would make a prefill "
                       "token's bytes depend on its call's width)", where);
            }
            entry.prefill = r;
        }
        // Prefill bands: a width-dependent choice among routes that sum as the prefill route
        // does (the same bytes whichever one a call takes).
        if (e.contains("prefill_bands")) {
            if (!entry.prefill) refuse("prefill_bands without a prefill route", where);
            for (const Json& b : e.at("prefill_bands")) {
                entry.prefill_bands.push_back({b.at(0).get<std::int32_t>(), b.at(1).get<std::int32_t>(),
                                               require_route(b.at(2), where)});
            }
            check_bands(entry.prefill_bands, where);
            for (const auto& b : entry.prefill_bands) {
                if (b.t_min < kPrefillMinTokens) {
                    refuse("a prefill band below T = " + std::to_string(kPrefillMinTokens), where);
                }
                if (!detail::generic::same_summation(b.route, *entry.prefill)) {
                    refuse("a prefill band whose route sums in another order than the prefill "
                           "route (a prefill token's bytes would depend on its call's width)",
                           where);
                }
            }
        }
        entry.claim = e.value("claim", false);
        if (entry.claim && *epi != Epilogue::Store) refuse("claim on a fused epilogue", where);
        table.generic.emplace(key, std::move(entry));
        ++table.status.generic_entries;
    }

    // The shape-owned Linear tables: re-ask the shape's table at a wider width.
    if (e.contains("route_as")) {
        if (*epi != Epilogue::Store) refuse("route_as on a fused epilogue", where);
        std::vector<RouteAsBand> bands;
        for (const Json& b : e.at("route_as")) {
            bands.push_back({b.at(0).get<std::int32_t>(), b.at(1).get<std::int32_t>(),
                             b.at(2).get<std::int32_t>()});
        }
        check_bands(bands, where);
        for (const RouteAsBand& b : bands) {
            // Never below the band's own widths: a call is not asked about fewer tokens than it
            // has, so a prefill width (T >= 17) is only re-asked at another prefill width.
            if (b.as < b.t_max) refuse("a route_as width below its band's t_max", where);
        }
        if (!bands.empty()) {
            table.route_as.emplace(key, std::move(bands));
            ++table.status.route_as_entries;
        }
    }
}

// A generic-MoE entry: {"format", ["down_format",] "experts", "top_k", "hidden", "inter",
// "bands": [[t_min, t_max, "<gate/up>|<down>"], ...], "prefill": "<gate/up>|<down>"};
// "down_format" (default: "format") names the down bank's format of a mixed layer.
void parse_moe_entry(const Json& e, const std::string& path, Table& table) {
    const std::string where = path + " moe entry " + e.dump();
    if (!e.is_object()) refuse("a moe entry that is not an object", where);
    const auto q  = parse_qtype(e.at("format").get<std::string>());
    const auto qd = e.contains("down_format")
                        ? parse_qtype(e.at("down_format").get<std::string>())
                        : q;
    detail::moe::MoeRouteEntry entry;
    entry.key = {q.value_or(QType::BF16),
                 qd.value_or(QType::BF16),
                 e.at("experts").get<std::int32_t>(),
                 e.at("top_k").get<std::int32_t>(),
                 e.at("hidden").get<std::int32_t>(),
                 e.at("inter").get<std::int32_t>()};
    if (!q || !qd || entry.key.experts < 1 || entry.key.top_k < 1 || entry.key.hidden < 1 ||
        entry.key.inter < 1) {
        refuse("a bad moe key", where);
    }
    const auto plan = [&](const Json& v) {
        const auto name = v.get<std::string>();
        const auto p    = detail::moe::parse_plan(name, entry.key.hidden);
        if (!p) refuse("unknown moe plan '" + name + "'", where);
        return *p;
    };
    for (const Json& b : e.value("bands", Json::array())) {
        entry.bands.push_back({b.at(0).get<std::int32_t>(), b.at(1).get<std::int32_t>(), plan(b.at(2))});
    }
    check_bands(entry.bands, where);
    // An R8 stage computes A8 at decode widths only, and only on a format with its kernel: each
    // stage's own bank's format.
    for (const auto& band : entry.bands) {
        if (band.plan.gate_up_r8 < 0 && band.plan.down_r8 < 0) continue;
        if ((band.plan.gate_up_r8 >= 0 && !detail::moe::r8_expert_format(entry.key.qtype)) ||
            (band.plan.down_r8 >= 0 && !detail::moe::r8_expert_format(entry.key.down_qtype))) {
            refuse("an R8 moe band on a format without the R8 kernel", where);
        }
        if (band.t_max > detail::moe::kR8Tokens) {
            refuse("an R8 moe band beyond T = " + std::to_string(detail::moe::kR8Tokens), where);
        }
    }
    if (e.contains("prefill")) {
        const auto p = plan(e.at("prefill"));
        if (!detail::moe::plan_is_token_invariant(p)) {
            refuse("a moe prefill plan that is not token-invariant", where);
        }
        entry.prefill = p;
    }
    if (!table.moe.emplace(moe_key_of(entry.key), std::move(entry)).second) {
        refuse("a duplicate moe layer", where);
    }
    ++table.status.entries;
}

std::shared_ptr<Table> parse_table(const std::string& path) {
    std::ifstream in(path_from_utf8(path), std::ios::binary);
    if (!in) throw std::runtime_error("route table: cannot open " + path);
    // The bytes are hashed as read, so the status names exactly what was parsed.
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw std::runtime_error("route table: cannot read " + path);
    Json root;
    try {
        root = Json::parse(bytes);
    } catch (const std::exception& e) {
        throw std::runtime_error("route table: " + path + ": " + e.what());
    }
    if (!root.is_object() || root.value("schema", std::string{}) != "ninferno-route-table/1") {
        throw std::runtime_error("route table: " + path + " is not ninferno-route-table/1");
    }
    auto table = std::make_shared<Table>();
    try {
        table->status.path           = path;
        table->status.content_sha256 = sha256_hex(std::string_view(bytes));
        table->status.gpu_class = root.at("gpu").at("class").get<std::string>();
        // "board": {"cc": 89, "sms": 58, "l2_mib": 48, "dram_gbps": 300}: the card it was
        // measured on, which the default lookup ranks tables by (board_distance). "mma_slow"
        // (written only when set) defaults to the class's chip fact, so a table without it keyed
        // tu116 / tu117 still counts as one of those chips.
        if (const Json& gpu = root.at("gpu"); gpu.contains("board")) {
            const Json& b    = gpu.at("board");
            const int cc     = b.at("cc").get<int>();
            const auto& cls  = table->status.gpu_class;
            const auto chip  = std::string_view(cls).substr(0, cls.find(':'));
            table->board     = Board{cc / 10, cc % 10, b.at("sms").get<int>(),
                                     b.value("l2_mib", 0.0), b.value("dram_gbps", 0.0),
                                     b.value("mma_slow", chip_mma_slow(cc / 10, cc % 10, chip))};
        }
        for (const Json& e : root.at("entries")) parse_entry(e, path, *table);
        for (const Json& e : root.value("moe", Json::array())) parse_moe_entry(e, path, *table);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("route table: " + path + ": " + e.what());
    }
    return table;
}

std::string summary(const RouteTableStatus& s) {
    return std::to_string(s.entries) + " problems (" + std::to_string(s.generic_entries) +
           " generic-path, " + std::to_string(s.route_as_entries) + " route_as) from " + s.path;
}

// A class key as a file name: ':' cannot appear in a Windows file name, so the variant
// "gb202:170sm" is stored as "gb202-170sm.json"; the key inside the file keeps the colon.
std::string route_table_file_name(std::string key) {
    std::replace(key.begin(), key.end(), ':', '-');
    return key + ".json";
}

// The keys an explicit table is accepted by: the device's own class chain, then the chips of its
// architecture (and of a sibling minor with the same features) nearest in SM count
// (DeviceClass::similarity_chain up to "*"). Routes are kernels of one architecture, so the chain
// never crosses into another.
std::vector<std::string> class_chain() {
    try {
        auto chain = device_class().similarity_chain();
        chain.erase(std::find(chain.begin(), chain.end(), std::string("*")), chain.end());
        chain.push_back("*");
        return chain;
    } catch (const std::exception&) {
        return {"*"};
    }
}

// The device's own class keys (id, chip, arch, "*"); none when the device is unreadable.
std::vector<std::string> own_keys() {
    try {
        return device_class().class_chain();
    } catch (const std::exception&) {
        return {};
    }
}

// How far a table's data carries to this device: the distance of the board it was measured on
// (board_distance). A table that records no board counts as measured on this device when it is
// keyed by one of the device's own classes, and as past every recorded board when keyed by a
// similar chip; another architecture's is infinitely far.
double table_distance(const Table& table) {
    try {
        const auto& device = device_class();
        if (table.board) return board_distance(board_of(device), *table.board);
        const auto own   = device.class_chain();
        const auto chain = class_chain();
        const auto& key  = table.status.gpu_class;
        if (std::find(own.begin(), own.end(), key) != own.end()) return 0.0;
        const auto at = std::find(chain.begin(), chain.end(), key);
        if (at != chain.end()) return 1e6 + static_cast<double>(at - chain.begin());
    } catch (const std::exception&) {
    }
    return std::numeric_limits<double>::infinity();
}

// The default lookup's tables, nearest measured board first: every table of `dir` whose routes
// this device runs (table_distance finite), ties in file-name order. A table named after one of
// the device's own classes that fails to parse is the caller's error to report; any other that
// fails lends nothing.
std::vector<std::shared_ptr<Table>> tables_by_distance(const std::string& dir) {
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() == ".json" && it->is_regular_file(ec)) files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    const auto own = own_keys();
    std::vector<std::pair<double, std::shared_ptr<Table>>> ranked;
    for (const auto& file : files) {
        std::shared_ptr<Table> table;
        try {
            table = parse_table(utf8_path(file));
        } catch (const std::exception&) {
            for (const auto& key : own) {
                if (route_table_file_name(key) == file.filename().string()) throw;
            }
            continue;
        }
        const double d = table_distance(*table);
        if (std::isfinite(d)) ranked.emplace_back(d, std::move(table));
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::shared_ptr<Table>> out;
    out.reserve(ranked.size());
    for (auto& entry : ranked) out.push_back(std::move(entry.second));
    return out;
}

// A problem the nearest table lacks takes the exact entry (same format, compute, epilogue, N, K)
// of the next nearest table that has one, before the heuristic. The default lookup ran on the
// heuristic wherever a chip's table missed a model's shapes, and the startup tuner that used to
// fill those gaps is off: the RTX 5070 Ti (gb203:70sm, whose table the RTX 5080 measured on other
// models) ran ISTA Qwen3.8 27B IQ3_S's projections untuned and lost C4 by 10 % to ninfer-all,
// while gb206 and gb202 had measured the same shapes. Only the generic entries borrow: a nearest
// shape of the own table was no better than the heuristic (r84), and MoE and route_as entries
// stay the nearest table's. A borrowed entry never claims a problem.
void fill_from(Table& table, const std::vector<std::shared_ptr<Table>>& lenders) {
    std::size_t added = 0;
    std::string from;
    std::string hashes;
    for (const auto& other : lenders) {
        std::size_t lent = 0;
        for (auto& [key, entry] : other->generic) {
            if (table.problems.count(key) != 0) continue;
            entry.claim = false;
            table.generic.emplace(key, std::move(entry));
            table.problems.insert(key);
            ++lent;
        }
        if (lent == 0) continue;
        added += lent;
        from += (from.empty() ? "" : ", ") + other->status.gpu_class + " " + std::to_string(lent);
        hashes += (hashes.empty() ? "" : ", ") + other->status.gpu_class + " " +
                  other->status.content_sha256;
    }
    if (added == 0) return;
    table.status.generic_entries += added;
    table.status.borrowed_entries = added;
    table.status.borrowed_from    = from;
    table.status.lender_sha256    = hashes;
}

// The table state of one device: each GPU of a mixed host reads its own class's table.
struct DeviceTables {
    std::mutex mutex;
    std::shared_ptr<const Table> table;
    bool default_attempted = false;
    // The active table was put in place explicitly (load_route_table, clear_route_table), not
    // found by the default lookup: set_route_table_dir drops it.
    bool explicit_table = false;
    // Why the default lookup installed nothing (no file for the class chain, a file for another
    // class, or a file refused and why); what route_table_status() reports while no table is
    // active.
    RouteTableStatus default_status;
    // Lock-free fast path for the hot dispatch sites: once the table is settled, a consumer whose
    // part of it is empty (route_as on every gb202 run) answers without taking the mutex.
    std::atomic<bool> settled{false};
    std::atomic<bool> has_route_as{false};
    std::atomic<bool> has_generic{false};
    std::atomic<bool> has_moe{false};
};

std::array<DeviceTables, kMaxDevices> g_devices;

// The calling thread's device, or device 0 where none is current (a table inspected or loaded
// before any device is set).
DeviceTables& here() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) device = 0;
    return g_devices[static_cast<std::size_t>(device)];
}

void publish_locked(DeviceTables& d, std::shared_ptr<const Table> table) {
    d.table = std::move(table);
    d.has_route_as.store(d.table && !d.table->route_as.empty(), std::memory_order_release);
    d.has_generic.store(d.table && !d.table->generic.empty(), std::memory_order_release);
    d.has_moe.store(d.table && !d.table->moe.empty(), std::memory_order_release);
    d.settled.store(true, std::memory_order_release);
}

RouteTableStatus publish_loaded_locked(DeviceTables& d, std::shared_ptr<Table> table) {
    const auto own       = own_keys();
    const auto& key      = table->status.gpu_class;
    const bool elsewhere = std::find(own.begin(), own.end(), key) == own.end();
    table->status.loaded  = true;
    table->status.message = "loaded " + summary(table->status) +
                            (elsewhere ? " (measured on " + key + ", the nearest measured board; " +
                                             "this device is " + route_gpu_class() + ")"
                                       : "") +
                            (table->status.borrowed_entries == 0
                                 ? ""
                                 : "; " + std::to_string(table->status.borrowed_entries) +
                                       " problems it lacks from the next nearest (" +
                                       table->status.borrowed_from + ")");
    const RouteTableStatus status = table->status;
    publish_locked(d, std::move(table));
    return status;
}

// An explicit table (load_route_table: the engine's --route-table) stands alone.
RouteTableStatus install_locked(DeviceTables& d, const std::string& path) {
    auto table       = parse_table(path);
    const auto chain = class_chain();
    const auto& key  = table->status.gpu_class;
    if (std::find(chain.begin(), chain.end(), key) == chain.end()) {
        RouteTableStatus s = table->status;
        s.loaded           = false;
        s.message          = "route table " + path + " is for " + key + ", this device is " +
                    route_gpu_class() + "; not applied";
        return s;
    }
    return publish_loaded_locked(d, std::move(table));
}

void ensure_default_locked(DeviceTables& d) {
    if (d.default_attempted) return;
    d.default_attempted = true;
    const std::string dir    = route_table_dir();
    d.default_status         = {};
    d.default_status.message = "no route table for " + route_gpu_class() +
                               (dir.empty() ? "" : " in " + dir) + "; routes are untuned";
    std::string attempted;
    try {
        if (!dir.empty()) {
            // The nearest measured board's table, its gaps filled from the next nearest.
            attempted   = dir;
            auto tables = tables_by_distance(dir);
            if (!tables.empty()) {
                auto nearest = std::move(tables.front());
                tables.erase(tables.begin());
                fill_from(*nearest, tables);
                (void)publish_loaded_locked(d, std::move(nearest));
            }
        }
    } catch (const std::exception& error) {
        // A broken default table must not stop the engine: every consumer answers as untuned.
        // It must not go unnoticed either: the reason is the table status (the startup log
        // prints it) and a warning now.
        d.table.reset();
        d.default_status      = {};
        d.default_status.path = attempted;
        d.default_status.message =
            "route table " + attempted + " refused (" + error.what() + "); routes are untuned";
        std::fprintf(stderr, "ninferno: warning: %s\n", d.default_status.message.c_str());
    }
    publish_locked(d, d.table);
}

std::shared_ptr<const Table> active_table(DeviceTables& d) {
    std::lock_guard lock(d.mutex);
    ensure_default_locked(d);
    return d.table;
}

std::shared_ptr<const Table> active_table() { return active_table(here()); }

} // namespace

std::string route_gpu_class() {
    try {
        return device_class().id();
    } catch (const std::exception&) {
        return "unknown";
    }
}

std::string route_dir_beside(const std::string& exe) {
    const std::filesystem::path bin = std::filesystem::path(exe).parent_path();
    for (const auto& dir : {bin / "routes", bin.parent_path() / "share" / "ninferno" / "routes"}) {
        // A routes/ directory holds the per-GPU-class tables (<class>.json).
        std::error_code ec;
        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() == ".json" && it->is_regular_file(ec)) return dir.string();
        }
    }
    return {};
}

namespace {
std::mutex g_dir_mutex;
std::filesystem::path g_dir; // set_route_table_dir; empty: the shipped directory
} // namespace

std::string route_table_dir() {
    {
        std::lock_guard lock(g_dir_mutex);
        if (!g_dir.empty()) return g_dir.string();
    }
    // A binary copied off the machine that built it (a release, a copy on another machine) reads the
    // routes shipped beside it, not whatever the build's source tree holds now or nothing at all.
    static const std::string beside = [] {
        const auto exe = sys::self_exe();
        return exe.empty() ? std::string() : route_dir_beside(exe.string());
    }();
    if (!beside.empty()) return beside;
    return NINFERNO_DEFAULT_ROUTE_DIR;
}

void set_route_table_dir(const std::filesystem::path& dir) {
    bool changed = false;
    {
        std::lock_guard lock(g_dir_mutex);
        changed = g_dir != dir;
        g_dir   = dir;
    }
    for (DeviceTables& d : g_devices) {
        std::lock_guard lock(d.mutex);
        if (!changed && !d.explicit_table) continue;
        d.explicit_table    = false;
        d.default_attempted = false;
        d.default_status    = {};
        d.table.reset();
        d.settled.store(false, std::memory_order_release);
    }
}

RouteTableStatus load_route_table(const std::string& path) {
    DeviceTables& d = here();
    std::lock_guard lock(d.mutex);
    d.default_attempted = true;
    d.explicit_table    = true;
    // Settled whatever the outcome (a refused or unreadable file leaves the active table), but only
    // once the outcome is known: settled before the file is parsed, a lock-free reader would answer
    // untuned from the old table's empty parts meanwhile.
    try {
        auto status = install_locked(d, path);
        publish_locked(d, d.table);
        return status;
    } catch (...) {
        publish_locked(d, d.table);
        throw;
    }
}

RouteTableStatus inspect_route_table(const std::string& path) {
    auto status    = parse_table(path)->status;
    status.message = "valid: " + summary(status);
    return status;
}

std::vector<std::int32_t> route_table_breakpoints(std::int32_t from) {
    std::vector<std::int32_t> out;
    // In 64 bits: a band's t_max + 1 is past int32 when the band runs to INT32_MAX, and no width
    // lies past that.
    const auto add = [&](std::int64_t width) {
        if (width >= from && width <= std::numeric_limits<std::int32_t>::max()) {
            out.push_back(static_cast<std::int32_t>(width));
        }
    };
    const auto table = active_table();
    if (!table) return out;
    for (const auto& entry : table->generic | std::views::values) {
        for (const RouteBand& band : entry.bands) {
            add(band.t_min);
            add(static_cast<std::int64_t>(band.t_max) + 1);
        }
        for (const RouteBand& band : entry.prefill_bands) {
            add(band.t_min);
            add(static_cast<std::int64_t>(band.t_max) + 1);
        }
    }
    for (const auto& bands : table->route_as | std::views::values) {
        for (const RouteAsBand& band : bands) {
            add(band.t_min);
            add(static_cast<std::int64_t>(band.t_max) + 1);
            // A band holds a call of t_min..t_max and re-asks the shape's own C++ ladder at
            // `as`, so the width that ladder turns at inside the band matters too.
            add(band.as);
        }
    }
    for (const auto& entry : table->moe | std::views::values) {
        for (const auto& band : entry.bands) {
            add(band.t_min);
            add(static_cast<std::int64_t>(band.t_max) + 1);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void clear_route_table() {
    DeviceTables& d = here();
    std::lock_guard lock(d.mutex);
    d.default_attempted      = true;
    d.explicit_table         = true;
    d.default_status         = {};
    d.default_status.message = "route table cleared; routes are untuned";
    publish_locked(d, nullptr);
}

RouteTableStatus route_table_status() {
    DeviceTables& d = here();
    std::lock_guard lock(d.mutex);
    ensure_default_locked(d);
    if (d.table) return d.table->status;
    return d.default_status;
}

namespace detail {

std::int32_t route_tokens(const ShapeRouteKey& key, std::int32_t tokens) {
    // A live route probe measures the shape table's own routes: the class overrides stand aside.
    const RouteProbeState& probe = route_probe_state;
    if (probe.tokens_as != 0 || probe.quantized_act >= 0) return route_tokens(tokens);
    DeviceTables& d = here();
    if (d.settled.load(std::memory_order_acquire) && !d.has_route_as.load(std::memory_order_acquire)) {
        return tokens;
    }
    const auto table = active_table(d);
    if (!table) return tokens;
    const auto it = table->route_as.find(key_of(key.qtype, key.compute, Epilogue::Store, key.n, key.k));
    if (it == table->route_as.end()) return tokens;
    for (const RouteAsBand& band : it->second) {
        if (tokens >= band.t_min && tokens <= band.t_max) return std::max(tokens, band.as);
    }
    return tokens;
}

namespace moe {

std::optional<MoePlan> table_plan(const MoeRouteKey& key, std::int32_t tokens, bool prefill) {
    DeviceTables& d = here();
    if (d.settled.load(std::memory_order_acquire) && !d.has_moe.load(std::memory_order_acquire)) {
        return std::nullopt;
    }
    const auto table = active_table(d);
    if (!table) return std::nullopt;
    const auto it = table->moe.find(moe_key_of(key));
    if (it == table->moe.end()) return std::nullopt;
    const MoeRouteEntry& e = it->second;
    if (prefill && tokens >= kPrefillMinTokens) return e.prefill;
    for (const auto& band : e.bands) {
        if (tokens >= band.t_min && tokens <= band.t_max) return band.plan;
    }
    return std::nullopt;
}

} // namespace moe

namespace generic {

ComputeClass compute_class(QType qtype, LinearPolicy policy) {
    return qtype == QType::NVFP4 && allows_a4(policy) ? ComputeClass::A4 : ComputeClass::A16;
}

const char* compute_class_name(ComputeClass c) { return route_compute_name(c); }

const char* epilogue_name(Epilogue e) {
    switch (e) {
    case Epilogue::Store:
        return "store";
    case Epilogue::AddResidual:
        return "add";
    case Epilogue::SwiGlu:
        return "swiglu";
    }
    return "store";
}

const char* qtype_name(QType q) {
    switch (q) {
    case QType::BF16:
        return "bf16";
    case QType::FP8_E4M3FN_ROW_BF16:
        return "fp8";
    case QType::NVFP4:
        return "nvfp4";
    case QType::Q4_G64_FP16:
        return "q4";
    case QType::Q5_G64_FP16:
        return "q5";
    case QType::Q6_G64_FP16:
        return "q6";
    case QType::Q8_G32_FP16:
        return "q8";
    case QType::MXFP4:
        return "mxfp4";
    case QType::FP8_E4M3FN_ROW_F32:
        return "fp8r32";
    case QType::FP8_E4M3FN_B128_F32:
        return "fp8b128";
    case QType::U4_G128_ZP_FP16:
        return "u4zp";
    case QType::U4_G32_ZP_FP16:
        return "u4zp32";
    case QType::U4_G64_ZP_FP16:
        return "u4zp64";
    case QType::Q3_G128_FP16:
        return "q3g128";
    case QType::Q3_G64_FP16:
        return "q3g64";
    case QType::GGML_Q4_K:
        return "q4k";
    case QType::GGML_Q5_K:
        return "q5k";
    case QType::GGML_Q6_K:
        return "q6k";
    case QType::GGML_Q3_K:
        return "q3k";
    case QType::GGML_Q2_K:
        return "q2k";
    case QType::GGML_IQ4_XS:
        return "iq4xs";
    case QType::GGML_IQ3_XXS:
        return "iq3xxs";
    case QType::GGML_IQ3_S:
        return "iq3s";
    case QType::GGML_Q4_0:
        return "q40";
    case QType::GGML_Q4_1:
        return "q41";
    case QType::GGML_Q5_0:
        return "q50";
    case QType::GGML_Q5_1:
        return "q51";
    case QType::GGML_IQ4_NL:
        return "iq4nl";
    case QType::GGML_IQ2_XXS:
        return "iq2xxs";
    case QType::GGML_IQ2_XS:
        return "iq2xs";
    case QType::GGML_IQ2_S:
        return "iq2s";
    case QType::GGML_IQ1_S:
        return "iq1s";
    case QType::GGML_IQ1_M:
        return "iq1m";
    case QType::GGML_TQ1_0:
        return "tq10";
    case QType::GGML_TQ2_0:
        return "tq20";
    case QType::GGML_Q1_0:
        return "q10";
    case QType::GGML_Q2_0:
        return "q20";
    default:
        return "unknown";
    }
}

std::string format_label(QType q) {
    return std::string(qtype_name(q)) + " (format " + std::to_string(static_cast<unsigned>(q)) +
           ")";
}

std::optional<Route> table_route(const Problem& problem, std::int32_t tokens, bool prefill) {
    DeviceTables& d = here();
    if (d.settled.load(std::memory_order_acquire) && !d.has_generic.load(std::memory_order_acquire)) {
        return std::nullopt;
    }
    const auto table = active_table(d);
    if (!table) return std::nullopt;
    const auto compute = compute_class(problem.qtype, problem.policy);
    auto it =
        table->generic.find(key_of(problem.qtype, compute, problem.epilogue, problem.n, problem.k));
    // fp8_e4m3fn_row_f32 (an FP8 checkpoint's binary32 row scales, kept as published) runs the
    // register MMA of fp8_e4m3fn_row_bf16 but for the multiplier's width, read once per row after
    // the K loop: without an entry of its own it takes the fp8 (row BF16) entry of its shape. Its
    // compute class is A16, so only an A16 entry is taken.
    if (it == table->generic.end() && problem.qtype == QType::FP8_E4M3FN_ROW_F32) {
        it = table->generic.find(key_of(QType::FP8_E4M3FN_ROW_BF16, compute, problem.epilogue,
                                        problem.n, problem.k));
    }
    if (it == table->generic.end()) return std::nullopt;
    const RouteEntry& e = it->second;
    if (prefill && tokens >= kPrefillMinTokens) {
        for (const auto& band : e.prefill_bands) {
            if (tokens >= band.t_min && tokens <= band.t_max) return band.route;
        }
        return e.prefill;
    }
    for (const auto& band : e.bands) {
        if (tokens >= band.t_min && tokens <= band.t_max) return band.route;
    }
    return std::nullopt;
}

bool table_claims(const Problem& problem, std::int32_t tokens, bool prefill) {
    // A policy that allows quantized activations keeps the shape's own table: its A8/A4 kernels
    // are what that policy is for, and the generic entries hold A16 (NVFP4: A4) routes.
    if (problem.policy != LinearPolicy::A16Only) return false;
    DeviceTables& d = here();
    if (d.settled.load(std::memory_order_acquire) && !d.has_generic.load(std::memory_order_acquire)) {
        return false;
    }
    const auto table = active_table(d);
    if (!table) return false;
    const auto compute = compute_class(problem.qtype, problem.policy);
    const auto it =
        table->generic.find(key_of(problem.qtype, compute, problem.epilogue, problem.n, problem.k));
    if (it == table->generic.end() || !it->second.claim) return false;
    return table_route(problem, tokens, prefill).has_value();
}

std::string write_route_table(const std::string& gpu_class, const std::string& gpu_name,
                              const std::vector<RouteEntry>& entries) {
    Json root;
    root["schema"] = "ninferno-route-table/1";
    root["gpu"]    = {{"class", gpu_class}, {"name", gpu_name}};
    // The board it is measured on, which the default lookup ranks tables by.
    try {
        const Board b        = board_of(device_class());
        root["gpu"]["board"] = {{"cc", b.major * 10 + b.minor}, {"sms", b.sms},
                                {"l2_mib", b.l2_mib}, {"dram_gbps", b.dram_gbps}};
        if (b.mma_slow) root["gpu"]["board"]["mma_slow"] = true;
    } catch (const std::exception&) {
    }
    Json list      = Json::array();
    for (const auto& e : entries) {
        Json j;
        j["format"]   = qtype_name(e.qtype);
        j["compute"]  = compute_class_name(e.compute);
        j["epilogue"] = epilogue_name(e.epilogue);
        j["n"]        = e.n;
        j["k"]        = e.k;
        Json bands    = Json::array();
        for (const auto& b : e.bands) bands.push_back({b.t_min, b.t_max, route_name(b.route)});
        j["bands"] = bands;
        if (e.prefill) j["prefill"] = route_name(*e.prefill);
        if (!e.prefill_bands.empty()) {
            Json pb = Json::array();
            for (const auto& b : e.prefill_bands) pb.push_back({b.t_min, b.t_max, route_name(b.route)});
            j["prefill_bands"] = pb;
        }
        if (e.claim) j["claim"] = true;
        list.push_back(j);
    }
    root["entries"] = list;
    return root.dump(1) + "\n";
}

} // namespace generic
} // namespace detail
} // namespace ninferno::ops
