// mmqgrid BOARD LAUNCHES K_OFFSET: #613 PR-5's prediction of the #607 forced-width grid. For each
// prompt size N of the grid and each forced column tile J, the counted MMQ price of one prefill's
// mul_mat_q launches (the selector's counted slice: mmq_unit::launch_seconds plus the kernel's
// graph node, generic_cost.cpp), columns = N - K_OFFSET. LAUNCHES: lines "COUNT FORMAT ROWS K"
// (FORMAT q4_K / q5_K / q6_K). Prints "N,columns,J,ms". No GPU.
#include "core/format_facts.h"
#include "core/isa_facts.h"
#include "core/launch_facts.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/mmq_unit.h"
#include "ops/generic/price_tool.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace ninferno;
    if (argc != 4) {
        std::fprintf(stderr, "usage: mmqgrid BOARD LAUNCHES K_OFFSET\n");
        return 2;
    }
    const GpuFactsRow* row = tools::price_find_board(argv[1]);
    if (row == nullptr) {
        std::fprintf(stderr, "no board %s\n", argv[1]);
        return 1;
    }
    const DeviceClass d    = tools::price_board_class(*row);
    const std::string arch = ops::detail::generic::kernel_facts::release_arch(d.major, d.minor);
    const auto isa         = isa_row(d.major, d.minor);
    if (arch.empty() || !isa) {
        std::fprintf(stderr, "no kernel facts or isa row\n");
        return 1;
    }
    const double node = launch_seconds(d.major, d.minor, LaunchKind::GraphNode).value_or(0.0);
    const double clk  = d.facts.reference_clock_hz();
    const double bw   = d.facts.dram_bytes_per_s();
    const std::map<std::string, QType> formats{
        {"q4_K", QType::GGML_Q4_K}, {"q5_K", QType::GGML_Q5_K}, {"q6_K", QType::GGML_Q6_K}};

    struct Launch {
        int count;
        QType q;
        int rows, k;
    };

    std::vector<Launch> launches;
    std::ifstream in(argv[2]);
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Launch l{};
        std::string f;
        ss >> l.count >> f >> l.rows >> l.k;
        l.q = formats.at(f);
        launches.push_back(l);
    }
    const int k_offset = std::stoi(argv[3]);
    std::fprintf(
        stderr,
        "board %s arch %s sms %d clock %.0f MHz dram %.1f GB/s node %.2f us, %zu launch kinds\n",
        argv[1], arch.c_str(), d.sms, clk / 1e6, bw / 1e9, node * 1e6, launches.size());
    const int sizes[] = {32,  40,  48,  64,  80,  96,  112, 128, 142, 144, 160, 176,
                         192, 208, 224, 240, 256, 270, 288, 320, 384, 448, 512};
    std::printf("N,columns,J,ms\n");
    for (const int n : sizes) {
        const int columns = n - k_offset;
        for (int j = 16; j <= 128; j += 16) {
            double s = 0;
            bool ok  = true;
            for (const Launch& l : launches) {
                const auto t = mmq_unit::launch_seconds(arch, l.q, *isa, l.rows, l.k, columns, j,
                                                        d.sms, clk, bw, node);
                if (!t) {
                    ok = false;
                    break;
                }
                s += l.count * (*t + node);
            }
            if (ok) {
                std::printf("%d,%d,%d,%.4f\n", n, columns, j, s * 1e3);
            } else {
                std::printf("%d,%d,%d,\n", n, columns, j);
            }
        }
    }
    return 0;
}
