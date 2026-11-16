// ninferno_r8_law_predict: the R8 decode schedule law's prediction (ops/generic/r8_law.h, #613
// PR-10) for every problem of a shapes file on one board, from the committed kernel-facts
// snapshots alone (no GPU). Its output is committed before the lab's grids are scored, so the
// prediction is frozen by its commit (tests/fixtures/r8_law/predicted_<board>.csv).
//
//   ninferno_r8_law_predict --board <name> --pci 0x2684 --shapes FILE
//
// FILE is the r8 grid's "<format> <compute> <epilogue> <n> <k>" per line (format q4k, q6k, q3k,
// iq2xxs, iq2xs, iq2s, iq3xxs, iq3s; epilogue store or swiglu), decode at T = 1. Output CSV:
//   format,epilogue,n,k,schedule,occupancy,rounds,passes,dram_us,issue_us,chain_us,total_us,pick
// one row per schedule the snapshot holds (pick 1 on the law's choice), or one row with schedule
// "none" where the law has no answer (a fact missing).
#include "core/device_class.h"
#include "ops/generic/codec_traits.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/r8_law.h"
#include "ops/generic/selector_facts.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace g  = ninferno::ops::detail::generic;
namespace kf = g::kernel_facts;
namespace kr = g::kernel_roofline;
namespace rl = g::r8_law;

// A format of the grid: the snapshot's codec key, the weight's DRAM bytes per value (the ggml
// block over its values), whether its blocks carry a min, its staged grid in 32-bit words.
struct Format {
    const char* name;
    const char* codec;
    double bytes_per_value;
    bool sums;
    int grid_words;
};

constexpr Format kFormats[] = {
    {"q4k", "W4a8Q45K<4>", 144.0 / 256, true, 0},
    {"q6k", "W4a8Q6K", 210.0 / 256, false, 0},
    {"q3k", "W4a8Q3K", 110.0 / 256, false, 0},
    {"iq2xxs", "W4a8Iq2Xxs", 66.0 / 256, false, 512},
    {"iq2xs", "W4a8Iq2Xs", 74.0 / 256, false, 1024},
    {"iq2s", "W4a8Iq2S", 82.0 / 256, false, 2048},
    {"iq3xxs", "W4a8Iq3Xxs", 98.0 / 256, false, 256},
    {"iq3s", "W4a8Iq3S", 110.0 / 256, false, 512},
};

const Format* format_of(const std::string& name) {
    for (const auto& f : kFormats) {
        if (name == f.name) return &f;
    }
    return nullptr;
}

// The snapshot label of a device: "120a" for consumer Blackwell, major and minor otherwise.
std::string arch_label(const ninferno::DeviceClass& d) {
    if (d.major == 12) return "120a";
    return std::to_string(d.major * 10 + d.minor);
}

} // namespace

int main(int argc, char** argv) {
    std::string board, shapes;
    std::uint32_t pci = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--board")
            board = argv[i + 1];
        else if (a == "--pci")
            pci = static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 0));
        else if (a == "--shapes")
            shapes = argv[i + 1];
    }
    if (board.empty() || pci == 0 || shapes.empty()) {
        std::cerr << "usage: ninferno_r8_law_predict --board <name> --pci 0x.... --shapes FILE\n";
        return 2;
    }
    const g::FactsBoard facts_board    = g::registry_facts_board(board.c_str(), pci);
    const ninferno::DeviceClass device = g::board_device_class(facts_board);
    const kr::ChipFacts chip           = kr::chip_facts_of(device);
    const std::string arch             = arch_label(device);
    std::ifstream in(shapes);
    if (!in) {
        std::cerr << "cannot read " << shapes << '\n';
        return 2;
    }
    std::printf("format,epilogue,n,k,schedule,occupancy,rounds,passes,dram_us,issue_us,chain_us,"
                "total_us,pick\n");
    for (std::string line; std::getline(in, line);) {
        std::istringstream fields(line);
        std::string format, compute, epilogue;
        std::int64_t n = 0, k = 0;
        if (!(fields >> format >> compute >> epilogue >> n >> k)) continue;
        const Format* f = format_of(format);
        if (f == nullptr) continue;
        const bool swiglu = epilogue == "swiglu";
        rl::Call call;
        call.n                      = n;
        call.k                      = k;
        call.nt                     = 1;
        call.swiglu                 = swiglu;
        call.sums                   = f->sums;
        call.weight_bytes_per_value = f->bytes_per_value;
        // Below sm_80 a codebook grid is staged in shared memory after the activation.
        call.grid_bytes = device.major < 8 && f->grid_words > 0
                              ? static_cast<std::int64_t>(f->grid_words) * 4 +
                                    static_cast<std::int64_t>(g::r8_grid_pad)
                              : 0;

        struct Row {
            rl::Schedule s;
            rl::Terms t;
        };

        std::vector<Row> rows;
        for (const rl::Schedule s : {rl::Schedule{4, 1}, rl::Schedule{4, 2}, rl::Schedule{8, 1},
                                     rl::Schedule{8, 2}, rl::Schedule{2, 1}}) {
            const auto facts =
                kf::r8_facts(arch, {f->codec, s.warps, s.rows, 1, swiglu ? 2 : 0, false});
            if (!facts) continue;
            if (const auto t = rl::terms(chip, device, *facts, call, s)) rows.push_back({s, *t});
        }
        if (rows.empty()) {
            std::printf("%s,%s,%lld,%lld,none,,,,,,,,0\n", format.c_str(), epilogue.c_str(),
                        static_cast<long long>(n), static_cast<long long>(k));
            continue;
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].t.total < rows[best].t.total) best = i;
        }
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            std::printf("%s,%s,%lld,%lld,w%dr%d,%g,%g,%g,%.3f,%.3f,%.3f,%.3f,%d\n", format.c_str(),
                        epilogue.c_str(), static_cast<long long>(n), static_cast<long long>(k),
                        r.s.warps, r.s.rows, r.t.occupancy, r.t.rounds, r.t.passes, r.t.dram * 1e6,
                        r.t.issue * 1e6, r.t.chain * 1e6, r.t.total * 1e6, i == best ? 1 : 0);
        }
    }
    return 0;
}
