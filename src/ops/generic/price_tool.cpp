#include "ops/generic/price_tool.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/machine_facts.h"
#include "core/sm_rate_facts.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <system_error>
#include <vector>

namespace ninferno::tools {

namespace {

namespace generic = ninferno::ops::detail::generic;
namespace kr      = generic::kernel_roofline;
using generic::Epilogue;
using generic::Family;
using generic::Route;
using ops::LinearPolicy;

// Every format `--format` takes, by the route table's name (qtype_name): the one list the parser
// matches against and the help prints, so they cannot drift apart.
const std::vector<std::pair<QType, std::string>>& known_formats() {
    static const std::vector<std::pair<QType, std::string>> formats = [] {
        std::vector<std::pair<QType, std::string>> out;
        for (int v = 0; v < 256; ++v) {
            const auto q           = static_cast<QType>(v);
            const std::string name = generic::qtype_name(q);
            if (name == "unknown") continue;
            bool seen = false;
            for (const auto& have : out) seen = seen || have.second == name;
            if (!seen) out.emplace_back(q, name);
        }
        return out;
    }();
    return formats;
}

std::string usage_text() {
    std::string formats;
    for (const auto& format : known_formats()) {
        if (!formats.empty()) formats += ", ";
        formats += format.second;
    }
    return "usage: ninferno-serve price --board BOARD --format FORMAT --n N --k K --t T\n"
           "                            [--epilogue store|add|swiglu] [--policy a16|a8]\n"
           "                            [--registers R] [--local-bytes B] [--route FAMILY:INDEX]\n"
           "       ninferno-serve price --list-boards\n"
           "BOARD: a facts-table name without spaces (rtx4090, l4, a100pcie40gb; see "
           "--list-boards) or\n"
           "its PCI id (0x2684).\n"
           "FORMAT: a route-table format name, case and underscores ignored: " +
           formats +
           ".\n"
           "--registers / --local-bytes: the register-MMA kernels' compiled facts, applied to "
           "every\n"
           "schedule of the heavy form (sub-block-scale formats where the board's terms are "
           "measured);\n"
           "absent, it is priced by threads alone, as a process without them is.\n";
}

// "GeForce RTX 4090" -> "rtx4090": lower case, the vendor words and the separators dropped.
std::string board_key(std::string_view name) {
    std::string key;
    for (const char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    for (const std::string_view prefix : {"nvidia", "geforce"}) {
        if (key.rfind(prefix, 0) == 0) key.erase(0, prefix.size());
    }
    return key;
}

const GpuFactsRow* find_board(std::string_view arg) {
    const auto table = gpu_facts_table();
    if (arg.rfind("0x", 0) == 0 || arg.rfind("0X", 0) == 0) {
        std::uint32_t id     = 0;
        const char* digits   = arg.data() + 2;
        const auto [end, ec] = std::from_chars(digits, arg.data() + arg.size(), id, 16);
        if (digits == arg.data() + arg.size() || ec != std::errc() ||
            end != arg.data() + arg.size()) {
            return nullptr;
        }
        for (const auto& row : table) {
            if (row.pci_device_id == id) return &row;
        }
        return nullptr;
    }
    const std::string key = board_key(arg);
    for (const auto& row : table) {
        if (board_key(row.name) == key) return &row;
    }
    return nullptr;
}

std::optional<QType> find_format(std::string_view arg) {
    std::string want;
    for (const char c : arg) {
        if (c != '_') want += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (const auto& format : known_formats()) {
        std::string have;
        for (const char c : format.second) {
            if (c != '_') have += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (have == want) return format.first;
    }
    return std::nullopt;
}

struct Options {
    const GpuFactsRow* board = nullptr;
    std::optional<QType> format;
    std::int32_t n = 0, k = 0, t = 0;
    Epilogue epilogue   = Epilogue::Store;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<int> registers;
    double local_bytes = 0;
    std::optional<Route> only;
    bool list_boards = false;
};

std::optional<Route> parse_route(std::string_view s) {
    const auto colon = s.find(':');
    if (colon == std::string_view::npos) return std::nullopt;
    const std::string_view family = s.substr(0, colon);
    const std::string digits(s.substr(colon + 1));
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos ||
        digits.size() > 3)
        return std::nullopt;
    const int index = std::stoi(digits);
    if (index > 255) return std::nullopt;
    Family f;
    if (family == "rmma") {
        f = Family::RegisterMma;
    } else if (family == "tiled") {
        f = Family::TiledMma;
    } else if (family == "staged") {
        f = Family::StagedMma;
    } else {
        return std::nullopt;
    }
    return Route{f, static_cast<std::uint8_t>(index)};
}

// A whole decimal integer in [lo, hi]: trailing text, an empty value or one out of range is an
// error, not the prefix std::atoi would take.
std::int32_t whole_number(std::string_view option, const std::string& v, long long lo,
                          long long hi) {
    long long value      = 0;
    const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), value);
    if (v.empty() || ec != std::errc() || end != v.data() + v.size() || value < lo || value > hi) {
        throw std::invalid_argument("bad value " + v + " for " + std::string(option));
    }
    return static_cast<std::int32_t>(value);
}

// A whole finite non-negative decimal number.
double finite_number(std::string_view option, const std::string& v) {
    char* end          = nullptr;
    const double value = v.empty() ? 0.0 : std::strtod(v.c_str(), &end);
    if (v.empty() || end != v.c_str() + v.size() || !std::isfinite(value) || value < 0) {
        throw std::invalid_argument("bad value " + v + " for " + std::string(option));
    }
    return value;
}

// Throws std::invalid_argument on a malformed command line.
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--list-boards") {
            o.list_boards = true;
            continue;
        }
        if (a == "--help" || a == "-h") throw std::invalid_argument("");
        if (i + 1 >= argc) throw std::invalid_argument("missing value after " + std::string(a));
        const std::string v = argv[++i];
        if (a == "--board") {
            o.board = find_board(v);
            if (!o.board) throw std::invalid_argument("unknown board " + v + " (--list-boards)");
        } else if (a == "--format") {
            o.format = find_format(v);
            if (!o.format) throw std::invalid_argument("unknown format " + v);
        } else if (a == "--n") {
            o.n = whole_number(a, v, 1, 1 << 30);
        } else if (a == "--k") {
            o.k = whole_number(a, v, 1, 1 << 30);
        } else if (a == "--t") {
            o.t = whole_number(a, v, 1, 1 << 20);
        } else if (a == "--epilogue") {
            if (v == "store") {
                o.epilogue = Epilogue::Store;
            } else if (v == "add" || v == "residual") {
                o.epilogue = Epilogue::AddResidual;
            } else if (v == "swiglu") {
                o.epilogue = Epilogue::SwiGlu;
            } else {
                throw std::invalid_argument("unknown epilogue " + v);
            }
        } else if (a == "--policy") {
            if (v == "a16") {
                o.policy = LinearPolicy::A16Only;
            } else if (v == "a8") {
                o.policy = LinearPolicy::AllowA8;
            } else {
                throw std::invalid_argument("unknown policy " + v);
            }
        } else if (a == "--registers") {
            o.registers = whole_number(a, v, 1, 255);
        } else if (a == "--local-bytes") {
            o.local_bytes = finite_number(a, v);
        } else if (a == "--route") {
            o.only = parse_route(v);
            if (!o.only)
                throw std::invalid_argument("unknown route " + v + " (rmma:N, tiled:N, staged:N)");
        } else {
            throw std::invalid_argument("unknown option " + std::string(a));
        }
    }
    if (o.list_boards) return o;
    if (!o.board || !o.format || o.n <= 0 || o.k <= 0 || o.t <= 0) {
        throw std::invalid_argument("--board, --format, --n, --k and --t are required");
    }
    return o;
}

double us(double seconds) { return seconds * 1e6; }

// The selector's time of a route among the priced ones, if it offered it.
std::optional<double> selector_time(const std::vector<generic::PricedRouteQuery>& priced, Route r) {
    for (const auto& p : priced) {
        if (p.route == r) return p.seconds;
    }
    return std::nullopt;
}

// "=" where the replica's total is the selector's number bit for bit, "!=" where the selector
// priced the route otherwise (it read compiled facts this replica was not given), "-" where the
// selector did not offer it (not admitted on the board, or not compiled for the weight).
const char* agreement(std::optional<double> selector, double replica) {
    if (!selector) return "-";
    return *selector == replica ? "=" : "!=";
}

void print_grid(const kr::Grid& g) {
    std::printf("  grid: ctas %.0f, waves %.0f, occupancy %.0f CTA/SM, rounds %.0f, warps %.0f, "
                "fill %.3f\n",
                g.ctas, g.waves, g.occupancy, g.rounds, g.warps, g.fill);
}

// The DRAM law of #690 (kernel_roofline.h dram_terms): P, b_w, the busiest SM's first-round rate
// R_sm(W) = min(P, W x b_w / L0), and which bound held.
void print_dram_law(const kr::DramTerms& d) {
    std::printf("  dram law: P %.1f GB/s an SM (%s carve-out), b_w %.0f B a warp (counted %.0f), "
                "R_sm %.1f GB/s; bus %.2f us, busiest SM %.2f us: the %s bound\n",
                d.peak / 1e9, d.shared_carved ? "shared-largest" : "L1-largest", d.warp_bytes,
                d.counted, d.round_rate / 1e9, us(d.bus), us(d.sm), d.bus_bound ? "bus" : "per-SM");
}

// The price the selector offers for an A16 tile's terms: total x the A16 scale, plus the
// completeness terms where they are on (generic_cost.cpp price_routes, the same order).
template <class Terms>
double offered(const Terms& t, double a16, bool complete) {
    const double scaled = t.total * a16;
    return complete ? scaled + (t.output + t.residual + t.quant + t.launch) : scaled;
}

// The completeness terms of a tile (kernel_roofline.h Completeness, #613 PR-8a).
template <class Terms>
void print_completeness(const Terms& t, bool complete) {
    std::printf(
        "  completeness terms (%s): output %.2f + residual %.2f + quant %.2f + launch "
        "%.2f%s us; complete %.2f us\n",
        complete ? "on, offered" : "off, computed only", us(t.output), us(t.residual), us(t.quant),
        us(t.launch),
        t.launch_bound && t.launch_extrapolated
            ? " (no launch facts: the slowest measured architecture's, its ramp extrapolated)"
        : t.launch_bound        ? " (no launch facts: the slowest measured architecture's)"
        : t.launch_extrapolated ? " (ramp extrapolated past the largest measured grid)"
                                : "",
        us(t.complete));
}

void print_register_mma(const kr::RegisterMmaTerms& t, bool heavy, double a16, double l2_fitted,
                        bool complete, std::optional<double> sel) {
    print_grid(t.grid);
    if (!t.legacy) print_dram_law(t.dram_law);
    if (!heavy) {
        std::printf("  bytes (plain form): mem %.2f us\n", us(t.mem));
    } else {
        std::printf("  bytes (heavy form): dram %.2f + l2 weight %.2f + l2 activation %.2f + spill "
                    "%.2f = %.2f us\n",
                    us(t.dram), us(t.l2_weight), us(t.l2_activation), us(t.spill), us(t.bytes));
    }
    std::printf("  mma %.2f us\n", us(t.mma));
    std::printf("  reduce: cluster %.2f + warps %.2f = %.2f us\n", us(t.reduce_cluster),
                us(t.reduce_warps), us(t.reduce));
    std::printf("  chain %.2f us\n", us(t.chain));
    // The L2 rate Little's law gives the busiest SM's warps at the counted bytes in flight, beside
    // the fitted rate the heavy form prices by (printed, not priced: PR-7b's).
    if (t.l2_little_bytes_per_clock) {
        std::printf(
            "  l2 by Little's law: %.0f warps x b_w / the ISA row's L2 latency = %.1f B/SM/clk "
            "(no port cap), the fitted rate %.0f\n",
            t.grid.warps, *t.l2_little_bytes_per_clock, l2_fitted);
    }
    print_completeness(t, complete);
    const double scaled = offered(t, a16, complete);
    const char* law     = t.legacy ? "(bytes + mma + reduce) / fill + chain (legacy DRAM law)"
                          : heavy  ? "dram + (l2 + spill + mma) / fill + reduce + chain"
                                   : "mem + mma / fill + reduce + chain";
    std::printf("  total %s = %.2f us, x a16 scale %.2f%s = %.2f us; selector %s%s\n", law,
                us(t.total), a16, complete ? " + the completeness terms" : "", us(scaled),
                sel ? (std::to_string(us(*sel)) + " us ").c_str() : "", agreement(sel, scaled));
}

void print_pipelined(const kr::PipelinedTerms& t, double a16, bool complete,
                     std::optional<double> sel) {
    print_grid(t.grid);
    if (!t.legacy) print_dram_law(t.dram_law);
    std::printf("  mem %.2f us, mma %.2f us, chain %.2f us\n", us(t.mem), us(t.mma), us(t.chain));
    print_completeness(t, complete);
    const double scaled = offered(t, a16, complete);
    std::printf("  total %s = %.2f us, x a16 scale %.2f%s = %.2f us; selector %s%s\n",
                t.legacy ? "max(mem, mma) / fill + chain (legacy DRAM law)"
                         : "max(mem, mma / fill) + chain",
                us(t.total), a16, complete ? " + the completeness terms" : "", us(scaled),
                sel ? (std::to_string(us(*sel)) + " us ").c_str() : "", agreement(sel, scaled));
}

int run(const Options& o) {
    if (o.list_boards) {
        for (const auto& row : gpu_facts_table()) {
            std::printf("%-44s 0x%04X %-6s sm_%d%d %4d SMs %5d MHz %7.0f GB/s%s\n",
                        board_key(row.name).c_str(), row.pci_device_id, row.chip.c_str(), row.major,
                        row.minor, row.sms, row.boost_mhz, row.bus_bits * row.gbps / 8.0,
                        row.target ? "" : " (not a target)");
        }
        return 0;
    }
    const GpuFactsRow& row = *o.board;
    const QType qtype      = *o.format;
    const DeviceClass d    = price_board_class(row);
    if (d.sms <= 0 || d.registers_per_sm <= 0 || !d.facts.has_roofline()) {
        std::fprintf(stderr, "price: %s has no roofline in its facts\n", row.name.c_str());
        return 1;
    }
    const kr::ChipFacts chip = kr::chip_facts_of(d);
    // The selector's switch (generic_cost.cpp price_routes: completeness_terms()).
    const kr::Calibration cal = [&] {
        kr::Calibration c    = kr::calibration_of(d.major, d.minor);
        c.completeness_terms = generic::completeness_terms();
        return c;
    }();
    const bool complete = cal.completeness_terms;
    const bool emulated = !d.features.mma_bf16_native;
    const double n = o.n, k = o.k;
    const std::uint64_t payload = generic::described_weight_bytes(qtype, o.epilogue, o.n, o.k);
    const kr::Call call{n,
                        k,
                        static_cast<double>(o.t),
                        o.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0,
                        static_cast<double>(payload) / (n * k),
                        o.epilogue == Epilogue::AddResidual};
    const bool heavy = has_subblock_scales(qtype) && rmma_terms_measured(d.major, d.minor);
    const double a16 = fitted_route_scales(d.major, d.minor).value_or(FittedRouteScales{}).a16_cost;
    const auto priced = generic::priced_routes_query(d, qtype, o.policy, o.epilogue, o.n, o.k, o.t);

    std::printf(
        "board %s (0x%04X, %s, sm_%d%d): %.0f SMs, %.0f MHz, %.0f GB/s, bf16 rate %.0f/SM/clk, "
        "%.0f threads, %.0f CTAs, %.0f registers, %.0f KiB shared per SM%s\n",
        row.name.c_str(), row.pci_device_id, row.chip.c_str(), d.major, d.minor, chip.sms,
        chip.clock_hz / 1e6, chip.dram_bytes_per_s / 1e9, chip.tensor_rate, chip.max_threads_per_sm,
        chip.max_ctas_per_sm, chip.registers_per_sm, chip.smem_per_sm / 1024.0,
        emulated ? " (BF16 MMA emulated: no pipelined tile)" : "");
    if (kr::legacy_dram(chip)) {
        const auto row = sm_rate_row(d.major, d.minor);
        std::printf("sm rate facts: %s sm_%d%d (core/sm_rate_facts.h): the legacy DRAM law\n",
                    row ? "another card refutes the row of" : "no row for", d.major, d.minor);
    } else {
        // b_w_max is a cap only where the probe saw a warp saturate; elsewhere the counted bytes
        // stand (no cap).
        std::printf("sm rate facts (core/sm_rate_facts.h): L0 %.1f ns, P %.2f B/clk free, %.2f "
                    "B/clk placed, ",
                    *chip.dram_latency_s * 1e9, *chip.sm_dram_bytes_per_clock_free,
                    *chip.sm_dram_bytes_per_clock_placed);
        if (chip.sm_max_warp_bytes) {
            std::printf("b_w_max %.0f B\n", *chip.sm_max_warp_bytes);
        } else {
            std::printf("no b_w cap (no warp saturated in the probe)\n");
        }
    }
    std::printf("call %s %dx%d T=%d %s: payload %llu bytes, %.4f bytes/value, sets %.0f; "
                "register-MMA terms: %s; a16 scale %.2f\n",
                generic::qtype_name(qtype), o.n, o.k, o.t, generic::epilogue_name(o.epilogue),
                static_cast<unsigned long long>(payload), call.bpw, call.sets,
                heavy ? "heavy form (sub-block scales, rmma_terms_measured)" : "plain form", a16);
    std::printf("calibration (route_cost_constants.h): rmma step %.0f cycles / %.0f K, saturating "
                "warps %.0f (heavy %.0f), L2 %.0f B/SM/clk, L1 %.0f B/SM/clk, spill accesses %.0f, "
                "reduce cluster %.0f / warp %.2f cycles, pipelined stage %.0f cycles / %.0f K, "
                "saturating warps %.0f, min fill %.2f\n",
                cal.rmma_step_cycles, cal.rmma_step_k, cal.rmma_saturating_warps,
                cal.rmma_heavy_saturating_warps, cal.l2_bytes_per_sm_clock,
                cal.l1_bytes_per_sm_clock, cal.spill_accesses_per_step, cal.cluster_reduce_cycles,
                cal.warp_reduce_cycles, cal.pipelined_stage_cycles, cal.pipelined_stage_k,
                cal.pipelined_saturating_warps, cal.min_warp_fill);
    // The completeness terms (#613 PR-8a): their facts and counts, and whether the selector
    // offers them.
    if (chip.launch_node_s && chip.launch_ramp) {
        std::printf("launch facts (core/launch_facts.h): graph node %.3f us, dispatch ramp",
                    *chip.launch_node_s * 1e6);
        const RampPoints& p = *chip.launch_ramp;
        for (int i = 1; i < 5; ++i) {
            if (p.seconds[0] && p.seconds[i]) {
                std::printf(" %.0f CTAs %.3f us", p.ctas[i], (*p.seconds[i] - *p.seconds[0]) * 1e6);
            }
        }
        std::printf(" (piecewise linear between them); ");
    } else {
        std::printf("launch facts: none for sm_%d%d (the launch term is unknown, priced at the "
                    "slowest measured architecture's); ",
                    d.major, d.minor);
    }
    std::printf(
        "completeness terms %s (route_cost::kCompletenessTerms); output %.0f bytes = %.2f us"
        "%s; quantizer bytes a value counted W4A8 %.3f (padding %.3f), I8 %.3f (padding "
        "%.3f), MMQ %.3f, fitted %.1f with a %.1f us launch while off\n",
        complete ? "on" : "off", n * o.t * 2.0, us(n * o.t * 2.0 / chip.dram_bytes_per_s),
        call.residual ? ", the residual read the same" : "",
        generic::route_cost::kQuantW4a8BytesPerValue.value(),
        generic::route_cost::kQuantW4a8PadBytesPerValue.value(),
        generic::route_cost::kQuantI8BytesPerValue.value(),
        generic::route_cost::kQuantI8PadBytesPerValue.value(),
        generic::route_cost::kQuantMmqBytesPerValue.value(),
        generic::route_cost::kQuantBytesPerValue.value(),
        us(generic::route_cost::kQuantLaunchSeconds.value()));
    if ((o.registers || o.local_bytes > 0) && !heavy) {
        std::printf("kernel facts given but not read: the plain form prices by threads alone\n");
    } else if (o.registers || o.local_bytes > 0) {
        std::printf("kernel facts given: %s registers, %.0f local bytes per thread\n",
                    o.registers ? std::to_string(*o.registers).c_str() : "unset", o.local_bytes);
    } else if (heavy) {
        std::printf("kernel facts: none given (priced by threads alone; the selector may have read "
                    "its own)\n");
    }
    // select_route prices routes (cost_route) only from 17 tokens outside the prefill band and
    // never for NVFP4; elsewhere it picks by rule, and these are the formula's figures alone.
    // route_authorities.cpp cost_model's condition (outside the prefill profile, which this tool
    // does not model).
    const bool in_domain = o.t >= generic::kPrefillBandMinTokens && qtype != QType::NVFP4;
    if (in_domain) {
        std::printf("selector offered %zu routes; terms in microseconds\n\n", priced.size());
    } else {
        std::printf("outside the selector's pricing domain (it prices from 17 tokens, never "
                    "NVFP4, and picks by rule here): %zu routes, the formula's figures only; terms "
                    "in microseconds\n\n",
                    priced.size());
    }

    const bool one = o.only.has_value();
    if (!one) {
        std::printf("%-10s %-9s %6s %5s %4s %6s %5s | %9s %9s %9s %9s | %9s %9s %-2s | %9s %9s\n",
                    "route", "tile", "ctas", "waves", "occ", "rounds", "fill", "bytes", "mma",
                    "reduce", "chain", "total", in_domain ? "selector" : "formula", "", "launch",
                    "if on");
    }
    // The completeness columns of a table row: the tile's launch ("*" after it where its grid's
    // ramp is extrapolated past the largest measured grid, "?" where the architecture has no
    // launch facts and the slowest measured one's stands in) and the price the selector offers
    // with the completeness terms on (total x a16 + the terms).
    const auto completeness_columns = [&](const auto& t) {
        std::printf(" | %8.2f%s %9.2f\n", us(t.launch),
                    t.launch_bound          ? "?"
                    : t.launch_extrapolated ? "*"
                                            : " ",
                    us(offered(t, a16, true)));
    };
    const auto wants = [&](Route r) { return !one || *o.only == r; };
    // The selector's two rules, read from the selector: a route it does not admit for this call
    // (route_admitted_query: route_admits, the family's whole rule on this build and board, for
    // this format and shape) or does not price on this board (a16_route_priced: price_routes'
    // pricing rule where the BF16 MMA is emulated) gets no price, since the figures would read as
    // a route it could take. `blocked` prints the row's reason and gives the exit of a --route of
    // it (2); nullopt for a route that is priced.
    const auto blocked = [&](Route r) -> std::optional<int> {
        const char* why =
            !generic::route_admitted_query(d, r, qtype, o.policy, o.epilogue, o.n, o.k)
                ? "not admitted on this build or board for this format"
            : !generic::a16_route_priced(r, !emulated) ? "not priced where the BF16 MMA is emulated"
                                                       : nullptr;
        if (why == nullptr) return std::nullopt;
        const std::string family(generic::family_name(r.family));
        if (one) {
            std::fprintf(stderr, "price: %s:%d %s\n", family.c_str(), static_cast<int>(r.schedule),
                         why);
        } else {
            std::printf("%s:%-*d %s\n", family.c_str(),
                        r.family == Family::RegisterMma ? 5
                        : r.family == Family::TiledMma  ? 4
                                                        : 3,
                        static_cast<int>(r.schedule), why);
        }
        return 2;
    };
    if (one) { // a schedule past the family's count names no route
        const Family f  = o.only->family;
        const int count = f == Family::RegisterMma ? generic::rmma_schedule_count()
                          : f == Family::TiledMma  ? generic::tiled_schedule_count()
                                                   : generic::staged_schedule_count();
        if (o.only->schedule >= count) {
            std::fprintf(stderr, "price: the family has %d schedules; --route names %d\n", count,
                         static_cast<int>(o.only->schedule));
            return 2;
        }
    }
    for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
        const Route r{Family::RegisterMma, static_cast<std::uint8_t>(i)};
        if (!wants(r)) continue;
        const auto s = generic::rmma_schedule_info(i);
        if (const auto exit = blocked(r)) {
            if (one) return *exit;
            continue;
        }
        const auto tile = *generic::route_tile(r, o.epilogue, o.t);
        kr::RegisterMmaKernel kernel;
        kernel.rows    = tile.out_rows;
        kernel.cols    = tile.tokens;
        kernel.cluster = tile.k_split;
        kernel.threads = 32.0 * s.warps_m * s.warps_k;
        kernel.warps_k = s.warps_k;
        kernel.m_tiles = s.m_tiles;
        kernel.k_step  = static_cast<double>(generic::rmma_step_values(qtype, o.k));
        kernel.heavy   = heavy;
        // The instance's static shared memory by the geometry rule; the selector reads the
        // compiled whole where it reads the kernel's facts (the heavy form), which this replica is
        // not given.
        kernel.shared_bytes = kr::register_mma_static_shared_bytes(
            generic::rmma_geometry_decoder(qtype), s.warps_m, s.warps_k, s.m_tiles, s.n_tiles,
            s.cluster, o.epilogue == Epilogue::SwiGlu, d.major * 10 + d.minor);
        if (heavy) { // the price reads compiled facts only in the heavy form, as price_routes does
            kernel.registers   = o.registers;
            kernel.local_bytes = o.local_bytes;
        }
        const auto t                = kr::price_register_mma(chip, cal, call, kernel);
        const auto sel              = selector_time(priced, r);
        const std::string tile_text = std::to_string(tile.out_rows) + "x" +
                                      std::to_string(tile.tokens) +
                                      (tile.k_split > 1 ? "/" + std::to_string(tile.k_split) : "");
        if (one) {
            std::printf("rmma:%d tile %s, %.0f threads, warps_k %.0f\n", i, tile_text.c_str(),
                        kernel.threads, kernel.warps_k);
            print_register_mma(t, heavy, a16, cal.l2_bytes_per_sm_clock, complete, sel);
            continue;
        }
        const double scaled = offered(t, a16, complete);
        std::printf(
            "rmma:%-5d %-9s %6.0f %5.0f %4.0f %6.0f %5.3f | %9.2f %9.2f %9.2f %9.2f | %9.2f ", i,
            tile_text.c_str(), t.grid.ctas, t.grid.waves, t.grid.occupancy, t.grid.rounds,
            t.grid.fill, us(t.bytes), us(t.mma), us(t.reduce), us(t.chain), us(scaled));
        if (sel) {
            std::printf("%9.2f %-2s", us(*sel), agreement(sel, scaled));
        } else {
            std::printf("%9s %-2s", "", agreement(sel, scaled));
        }
        completeness_columns(t);
    }
    const auto pipelined = [&](Route r, const kr::PipelinedKernel& kernel) {
        const auto t                = kr::price_pipelined(chip, cal, call, kernel);
        const auto sel              = selector_time(priced, r);
        const std::string tile_text = std::to_string(static_cast<int>(kernel.rows)) + "x" +
                                      std::to_string(static_cast<int>(kernel.cols));
        const std::string family(generic::family_name(r.family));
        if (one) {
            std::printf("%s:%d tile %s, %.0f threads, %.0f stages of %.0fx%.0f\n", family.c_str(),
                        r.schedule, tile_text.c_str(), kernel.threads, kernel.stages,
                        kernel.stage_rows, kernel.stage_cols);
            print_pipelined(t, a16, complete, sel);
            return;
        }
        const double scaled = offered(t, a16, complete);
        std::printf("%s:%-*d %-9s %6.0f %5.0f %4.0f %6.0f %5.3f | %9.2f %9.2f %9s %9.2f | %9.2f ",
                    family.c_str(), r.family == Family::TiledMma ? 4 : 3, r.schedule,
                    tile_text.c_str(), t.grid.ctas, t.grid.waves, t.grid.occupancy, t.grid.rounds,
                    t.grid.fill, us(t.mem), us(t.mma), "", us(t.chain), us(scaled));
        if (sel) {
            std::printf("%9.2f %-2s", us(*sel), agreement(sel, scaled));
        } else {
            std::printf("%9s %-2s", "", agreement(sel, scaled));
        }
        completeness_columns(t);
    };
    for (int i = 0; i < generic::tiled_schedule_count(); ++i) {
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (!wants(r)) continue;
        if (const auto exit = blocked(r)) {
            if (one) return *exit;
            continue;
        }
        const auto s    = generic::tiled_schedule_info(i);
        const auto tile = *generic::route_tile(r, o.epilogue, o.t);
        kr::PipelinedKernel kernel;
        kernel.rows            = tile.out_rows;
        kernel.cols            = tile.tokens;
        kernel.threads         = 256;
        kernel.stages          = 3;
        kernel.stage_rows      = s.block_n;
        kernel.stage_cols      = s.block_m;
        kernel.tensor_fraction = generic::route_cost::kTiledTensorFraction.value();
        pipelined(r, kernel);
    }
    for (int i = 0; i < generic::staged_schedule_count(); ++i) {
        const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
        if (!wants(r)) continue;
        if (const auto exit = blocked(r)) {
            if (one) return *exit;
            continue;
        }
        const auto s    = generic::staged_schedule_info(i);
        const auto tile = *generic::route_tile(r, o.epilogue, o.t);
        kr::PipelinedKernel kernel;
        kernel.rows            = tile.out_rows;
        kernel.cols            = tile.tokens;
        kernel.threads         = 32.0 * s.warps_m * s.warps_n;
        kernel.stages          = s.stages;
        kernel.stage_rows      = s.rows;
        kernel.stage_cols      = s.tokens;
        kernel.tensor_fraction = generic::route_cost::kStagedTensorFraction.value();
        pipelined(r, kernel);
    }
    // The selector's other priced routes (under --policy a8: the W4A8, I8 and MMQ tiles of
    // a8_candidates), so the table holds every route it offered and its cheapest. Their terms are
    // not in kernel_roofline.h yet (#613 PR-5 / PR-6): the row is the selector's price alone.
    if (!one) {
        for (const auto& p : priced) {
            if (p.route.family == Family::RegisterMma || p.route.family == Family::TiledMma ||
                p.route.family == Family::StagedMma) {
                continue;
            }
            std::printf("%-10s %-9s %6s %5s %4s %6s %5s | %9s %9s %9s %9s | %9s %9.2f (a8 "
                        "candidate: the selector's price, no roofline terms)\n",
                        generic::route_name(p.route).c_str(), "", "", "", "", "", "", "", "", "",
                        "", "", us(p.seconds));
        }
    }
    return 0;
}

} // namespace

std::vector<std::string> price_format_names() {
    std::vector<std::string> names;
    for (const auto& format : known_formats()) names.push_back(format.second);
    return names;
}

std::optional<QType> price_find_format(std::string_view name) { return find_format(name); }

std::string price_usage_text() { return usage_text(); }

DeviceClass price_board_class(const GpuFactsRow& row) {
    return board_class({.pci_device_id = row.pci_device_id,
                        .major         = row.major,
                        .minor         = row.minor,
                        .sms           = row.sms,
                        .global_bytes  = static_cast<std::size_t>(row.vram_gib * 1024 * kMiB)});
}

int price_main(int argc, char** argv) {
    Options options;
    try {
        options = parse(argc, argv);
    } catch (const std::invalid_argument& error) {
        if (*error.what() != '\0') std::fprintf(stderr, "price: %s\n", error.what());
        std::fputs(usage_text().c_str(), stderr);
        return 2;
    }
    try {
        return run(options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "price: %s\n", error.what());
        return 1;
    }
}

} // namespace ninferno::tools
