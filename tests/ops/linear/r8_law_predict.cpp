// ninferno_r8_law_predict: the R8 decode schedule law's prediction (ops/generic/r8_law.h, #613
// PR-10) for every problem of a shapes file on one board, from the committed kernel-facts
// snapshots alone (no GPU). Its output is committed before the lab's grids are scored, so the
// prediction is frozen by its commit (tests/fixtures/r8_law/predicted_<board>.csv).
//
//   ninferno_r8_law_predict --board <name> --pci 0x2684 --shapes FILE [--nt 1]
//
// FILE is the r8 grid's "<format> <compute> <epilogue> <n> <k>" per line (format q4k, q6k, q3k,
// iq2xxs, iq2xs, iq2s, iq3xxs, iq3s; epilogue store or swiglu); --nt the kernel's token slots
// (1, 2, 4, 8; default 1, decode at T = 1). Output CSV:
//   format,epilogue,n,k,schedule,occupancy,rounds,passes,bus_us,dram_us,issue_us,chain_us,
//   rank_us,total_us,pick
// one row per schedule the snapshot holds, or one row with schedule "none" where the law has no
// answer (a fact missing). The schedules are compared on rank (the busiest SM's terms; the bus is
// the same for all): pick 1 on the law's choice, 2 on a schedule whose rank equals it (a tie the
// law does not break; the choice is then the first in the snapshot's order).
#include "core/device_class.h"
#include "core/format_facts.h"
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

// A format of the grid: its snapshot codec key and its QType. The weight's DRAM bytes per value
// (ggml_bits_per_value), its min (codec_traits: the activation's block sums travel with it) and
// its staged grid (codec_traits grid_words) are read from the format's own facts.
struct Format {
    const char* name;
    const char* codec;
    ninferno::QType q;
};

constexpr Format kFormats[] = {
    {"q4k", "W4a8Q45K<4>", ninferno::QType::GGML_Q4_K},
    {"q6k", "W4a8Q6K", ninferno::QType::GGML_Q6_K},
    {"q3k", "W4a8Q3K", ninferno::QType::GGML_Q3_K},
    {"iq2xxs", "W4a8Iq2Xxs", ninferno::QType::GGML_IQ2_XXS},
    {"iq2xs", "W4a8Iq2Xs", ninferno::QType::GGML_IQ2_XS},
    {"iq2s", "W4a8Iq2S", ninferno::QType::GGML_IQ2_S},
    {"iq3xxs", "W4a8Iq3Xxs", ninferno::QType::GGML_IQ3_XXS},
    {"iq3s", "W4a8Iq3S", ninferno::QType::GGML_IQ3_S},
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
    int nt            = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--board")
            board = argv[i + 1];
        else if (a == "--pci")
            pci = static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 0));
        else if (a == "--shapes")
            shapes = argv[i + 1];
        else if (a == "--nt")
            nt = std::atoi(argv[i + 1]);
    }
    if (board.empty() || pci == 0 || shapes.empty() || (nt != 1 && nt != 2 && nt != 4 && nt != 8)) {
        std::cerr << "usage: ninferno_r8_law_predict --board <name> --pci 0x.... --shapes FILE "
                     "[--nt 1|2|4|8]\n";
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
    std::printf("format,epilogue,n,k,schedule,occupancy,rounds,passes,bus_us,dram_us,issue_us,"
                "chain_us,rank_us,total_us,pick\n");
    for (std::string line; std::getline(in, line);) {
        std::istringstream fields(line);
        std::string format, compute, epilogue;
        std::int64_t n = 0, k = 0;
        if (!(fields >> format >> compute >> epilogue >> n >> k)) continue;
        const Format* f = format_of(format);
        if (f == nullptr) continue;
        const bool swiglu           = epilogue == "swiglu";
        const g::CodecTraits traits = g::codec_traits(f->q);
        rl::Call call;
        call.n                      = n;
        call.k                      = k;
        call.nt                     = nt;
        call.slots                  = rl::slots_of(arch);
        call.swiglu                 = swiglu;
        call.sums                   = traits.min;
        call.weight_bytes_per_value = ninferno::ggml_bits_per_value(f->q) / 8.0;
        // Below sm_80 a codebook grid is staged in shared memory after the activation.
        call.grid_bytes = device.major < 8 && traits.grid_words > 0
                              ? static_cast<std::int64_t>(traits.grid_words) * 4 +
                                    static_cast<std::int64_t>(g::r8_grid_pad)
                              : 0;

        struct Row {
            rl::Schedule s;
            rl::Terms t;
        };

        std::vector<Row> rows;
        for (const rl::Schedule s : {rl::Schedule{4, 1}, rl::Schedule{4, 2}, rl::Schedule{8, 1},
                                     rl::Schedule{8, 2}, rl::Schedule{2, 1}}) {
            // As the launch takes it (r8_launch_schedule): 2..8 tokens with a workspace run the
            // pre-quantized verify form where the codec has it (its facts row), else the in-CTA
            // quantizer up to 4 tokens; 8 tokens have no other form.
            const auto pre_facts =
                nt > 1 ? kf::r8_facts(arch, {f->codec, s.warps, s.rows, nt, swiglu ? 2 : 0, true})
                       : std::nullopt;
            const auto facts =
                pre_facts
                    ? pre_facts
                    : kf::r8_facts(arch, {f->codec, s.warps, s.rows, nt, swiglu ? 2 : 0, false});
            if (!facts) continue;
            rl::Call c = call;
            c.pre      = pre_facts.has_value();
            if (const auto t = rl::terms(chip, device, *facts, c, s)) rows.push_back({s, *t});
        }
        if (rows.empty()) {
            std::printf("%s,%s,%lld,%lld,none,,,,,,,,,,0\n", format.c_str(), epilogue.c_str(),
                        static_cast<long long>(n), static_cast<long long>(k));
            continue;
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].t.rank < rows[best].t.rank) best = i;
        }
        const double best_rank = rows[best].t.rank;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& r  = rows[i];
            const int pick = i == best ? 1 : r.t.rank == best_rank ? 2 : 0;
            std::printf("%s,%s,%lld,%lld,w%dr%d,%g,%g,%g,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
                        format.c_str(), epilogue.c_str(), static_cast<long long>(n),
                        static_cast<long long>(k), r.s.warps, r.s.rows, r.t.occupancy, r.t.rounds,
                        r.t.passes, r.t.bus * 1e6, r.t.dram * 1e6, r.t.issue * 1e6, r.t.chain * 1e6,
                        r.t.rank * 1e6, r.t.total * 1e6, pick);
        }
    }
    return 0;
}
