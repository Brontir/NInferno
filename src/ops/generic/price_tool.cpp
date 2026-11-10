#include "ops/generic/price_tool.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/machine_facts.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::tools {

namespace {

namespace generic = ninferno::ops::detail::generic;
namespace kr      = generic::kernel_roofline;
using generic::Epilogue;
using generic::Family;
using generic::Route;
using ops::LinearPolicy;

constexpr const char* kUsage =
    "usage: ninferno-serve price --board BOARD --format FORMAT --n N --k K --t T\n"
    "                            [--epilogue store|add|swiglu] [--policy a16|a8]\n"
    "                            [--registers R] [--local-bytes B] [--route FAMILY:INDEX]\n"
    "       ninferno-serve price --list-boards\n"
    "BOARD: a facts-table name without spaces (rtx4090, l4, a100) or its PCI id (0x2684).\n"
    "FORMAT: a route-table format name (q4, q4_k, iq3_s, bf16).\n"
    "--registers / --local-bytes: the register-MMA kernels' compiled facts, applied to every\n"
    "schedule (absent: priced by threads alone, as a process without them is).\n";

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
        const auto id =
            static_cast<std::uint32_t>(std::strtoul(std::string(arg).c_str(), nullptr, 16));
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

// The board as a substituted class: its row's facts, the architecture's limits (as the selector
// golden boards are built, selector_facts.cpp).
DeviceClass board_class(const GpuFactsRow& row) {
    DeviceClass d      = synthetic_device_class(row.pci_device_id, row.major, row.minor, row.sms,
                                                static_cast<std::size_t>(row.l2_mib * kMiB),
                                                static_cast<std::size_t>(row.vram_gib * 1024 * kMiB));
    d.sm_clock_khz     = row.boost_mhz * 1000;
    d.memory_clock_khz = static_cast<int>(row.gbps * 1e6 / 2.0); // half the data rate, kHz
    d.memory_bus_bits  = row.bus_bits;
    d.facts            = machine_facts_of(d);
    return d;
}

std::optional<QType> find_format(std::string_view arg) {
    std::string want;
    for (const char c : arg) {
        if (c != '_') want += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (int v = 0; v < 256; ++v) {
        const auto q                = static_cast<QType>(v);
        const std::string_view name = generic::qtype_name(q);
        if (name == "unknown") continue;
        std::string have;
        for (const char c : name) {
            if (c != '_') have += c;
        }
        if (have == want) return q;
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
    const int index               = std::atoi(std::string(s.substr(colon + 1)).c_str());
    if (index < 0 || index > 255) return std::nullopt;
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
            o.n = std::atoi(v.c_str());
        } else if (a == "--k") {
            o.k = std::atoi(v.c_str());
        } else if (a == "--t") {
            o.t = std::atoi(v.c_str());
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
            o.registers = std::atoi(v.c_str());
        } else if (a == "--local-bytes") {
            o.local_bytes = std::atof(v.c_str());
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

const char* family_name(Family f) {
    return f == Family::RegisterMma ? "rmma" : f == Family::TiledMma ? "tiled" : "staged";
}

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

void print_register_mma(const kr::RegisterMmaTerms& t, bool heavy, double a16,
                        std::optional<double> sel) {
    print_grid(t.grid);
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
    const double scaled = t.total * a16;
    std::printf("  total (bytes + mma + reduce) / fill + chain = %.2f us, x a16 scale %.2f = %.2f "
                "us; selector %s%s\n",
                us(t.total), a16, us(scaled),
                sel ? (std::to_string(us(*sel)) + " us ").c_str() : "", agreement(sel, scaled));
}

void print_pipelined(const kr::PipelinedTerms& t, double a16, std::optional<double> sel) {
    print_grid(t.grid);
    std::printf("  mem %.2f us, mma %.2f us, chain %.2f us\n", us(t.mem), us(t.mma), us(t.chain));
    const double scaled = t.total * a16;
    std::printf("  total max(mem, mma) / fill + chain = %.2f us, x a16 scale %.2f = %.2f us; "
                "selector %s%s\n",
                us(t.total), a16, us(scaled),
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
    const DeviceClass d    = board_class(row);
    if (d.sms <= 0 || d.registers_per_sm <= 0 || !d.facts.has_roofline()) {
        std::fprintf(stderr, "price: %s has no roofline in its facts\n", row.name.c_str());
        return 1;
    }
    const kr::ChipFacts chip  = kr::chip_facts_of(d);
    const kr::Calibration cal = kr::calibration_of(d.major, d.minor);
    const bool emulated       = !d.features.mma_bf16_native;
    const double n = o.n, k = o.k;
    const std::uint64_t payload = generic::described_weight_bytes(qtype, o.epilogue, o.n, o.k);
    const kr::Call call{n, k, static_cast<double>(o.t), o.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0,
                        static_cast<double>(payload) / (n * k)};
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
    if (o.registers) {
        std::printf("kernel facts given: %d registers, %.0f local bytes per thread\n", *o.registers,
                    o.local_bytes);
    } else if (heavy) {
        std::printf("kernel facts: none given (priced by threads alone; the selector may have read "
                    "its own)\n");
    }
    std::printf("selector offered %zu routes; terms in microseconds\n\n", priced.size());

    const bool one = o.only.has_value();
    if (!one) {
        std::printf("%-10s %-9s %6s %5s %4s %6s %5s | %9s %9s %9s %9s | %9s %9s %s\n", "route",
                    "tile", "ctas", "waves", "occ", "rounds", "fill", "bytes", "mma", "reduce",
                    "chain", "total", "selector", "");
    }
    const auto wants = [&](Route r) { return !one || *o.only == r; };
    for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
        const Route r{Family::RegisterMma, static_cast<std::uint8_t>(i)};
        if (!wants(r)) continue;
        const auto s = generic::rmma_schedule_info(i);
        if (!generic::rmma_schedule_launchable(i) || (emulated && s.n_tiles == 8)) {
            if (one) std::printf("rmma:%d is not priced on this build or board\n", i);
            continue;
        }
        const auto tile = *generic::route_tile(r, o.epilogue, o.t);
        kr::RegisterMmaKernel kernel;
        kernel.rows                 = tile.out_rows;
        kernel.cols                 = tile.tokens;
        kernel.cluster              = tile.k_split;
        kernel.threads              = 32.0 * s.warps_m * s.warps_k;
        kernel.warps_k              = s.warps_k;
        kernel.heavy                = heavy;
        kernel.registers            = o.registers;
        kernel.local_bytes          = o.local_bytes;
        const auto t                = kr::price_register_mma(chip, cal, call, kernel);
        const auto sel              = selector_time(priced, r);
        const std::string tile_text = std::to_string(tile.out_rows) + "x" +
                                      std::to_string(tile.tokens) +
                                      (tile.k_split > 1 ? "/" + std::to_string(tile.k_split) : "");
        if (one) {
            std::printf("rmma:%d tile %s, %.0f threads, warps_k %.0f\n", i, tile_text.c_str(),
                        kernel.threads, kernel.warps_k);
            print_register_mma(t, heavy, a16, sel);
            continue;
        }
        const double scaled = t.total * a16;
        std::printf(
            "rmma:%-5d %-9s %6.0f %5.0f %4.0f %6.0f %5.3f | %9.2f %9.2f %9.2f %9.2f | %9.2f ", i,
            tile_text.c_str(), t.grid.ctas, t.grid.waves, t.grid.occupancy, t.grid.rounds,
            t.grid.fill, us(t.bytes), us(t.mma), us(t.reduce), us(t.chain), us(scaled));
        if (sel) {
            std::printf("%9.2f %s\n", us(*sel), agreement(sel, scaled));
        } else {
            std::printf("%9s %s\n", "", agreement(sel, scaled));
        }
    }
    const auto pipelined = [&](Route r, const kr::PipelinedKernel& kernel) {
        const auto t                = kr::price_pipelined(chip, cal, call, kernel);
        const auto sel              = selector_time(priced, r);
        const std::string tile_text = std::to_string(static_cast<int>(kernel.rows)) + "x" +
                                      std::to_string(static_cast<int>(kernel.cols));
        if (one) {
            std::printf("%s:%d tile %s, %.0f threads, %.0f stages of %.0fx%.0f\n",
                        family_name(r.family), r.schedule, tile_text.c_str(), kernel.threads,
                        kernel.stages, kernel.stage_rows, kernel.stage_cols);
            print_pipelined(t, a16, sel);
            return;
        }
        const double scaled = t.total * a16;
        std::printf("%s:%-*d %-9s %6.0f %5.0f %4.0f %6.0f %5.3f | %9.2f %9.2f %9s %9.2f | %9.2f ",
                    family_name(r.family), r.family == Family::TiledMma ? 4 : 3, r.schedule,
                    tile_text.c_str(), t.grid.ctas, t.grid.waves, t.grid.occupancy, t.grid.rounds,
                    t.grid.fill, us(t.mem), us(t.mma), "", us(t.chain), us(scaled));
        if (sel) {
            std::printf("%9.2f %s\n", us(*sel), agreement(sel, scaled));
        } else {
            std::printf("%9s %s\n", "", agreement(sel, scaled));
        }
    };
    if (emulated) return 0;
    for (int i = 0; i < generic::tiled_schedule_count(); ++i) {
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (!wants(r)) continue;
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
    return 0;
}

} // namespace

int price_main(int argc, char** argv) {
    Options options;
    try {
        options = parse(argc, argv);
    } catch (const std::invalid_argument& error) {
        if (*error.what() != '\0') std::fprintf(stderr, "price: %s\n", error.what());
        std::fputs(kUsage, stderr);
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
