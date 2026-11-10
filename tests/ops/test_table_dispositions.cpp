// What the measured value tables' answers rest on (#616 step 3), host only, no GPU: the generic MoE
// op's class A8 route (moe::a8_route_answer) and the sparse MoE's Q4/Q5 groupwise prefill frontier
// (sparse_moe_q4_groupwise_frontier_answer) for every target SKU of the reference table
// (core/data/gpu_facts.json): Own, Carried (donor, distance) or Fallback (the table's declared
// literal). Compared with table_dispositions/dispositions.txt; --update rewrites it. Also checks
// that the disposition's value is the one the production lookup returns on the same board (the
// disposition never disagrees with the answer), that every measured board answers Own for itself,
// and that a Fallback has no donor and the declared literal.
#include "core/device_class.h"
#include "core/gate_atlas.h"
#include "core/machine_facts.h"
#include "core/measured_boards.h"
#include "core/weight.h"
#include "ops/generic/selector_facts.h"
#include "ops/moe/moe_internal.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace {

using namespace ninferno;
namespace moe     = ninferno::ops::detail::moe;
namespace detail  = ninferno::ops::detail;
namespace generic = ninferno::ops::detail::generic;
using measured::TableDisposition;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok && ++failures <= 20) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

std::string board_text(const std::optional<Board>& b) {
    if (!b) return "-";
    char buf[96];
    std::snprintf(buf, sizeof buf, "sm%d%d/%dsm/%gMiB/%gGBps", b->major, b->minor, b->sms,
                  static_cast<double>(b->l2_mib), b->dram_gbps);
    return buf;
}

std::string route_text(const moe::A8Route& r) {
    std::ostringstream out;
    out << "down=" << r.down << " 2bit=";
    for (const auto& row : r.gate_up_2bit) {
        for (const int s : row) out << s;
        out << '/';
    }
    return out.str();
}

bool same_route(const moe::A8Route& a, const moe::A8Route& b) {
    return a.down == b.down && a.gate_up_2bit == b.gate_up_2bit;
}

std::string dump() {
    std::ostringstream out;
    out << "# SKU | a8 route: disposition donor distance route | q4 frontier: disposition donor "
           "distance min_tokens\n";
    for (const GpuFactsRow& row : gpu_facts_table()) {
        if (!row.target) continue;
        const Board board   = measured::table_board(row);
        const auto a8       = moe::a8_route_answer(board);
        const auto frontier = detail::sparse_moe_q4_groupwise_frontier_answer(board);
        char buf[64];
        out << row.name << " | a8 " << measured::table_disposition_name(a8.disposition) << ' '
            << board_text(a8.donor) << ' ';
        std::snprintf(buf, sizeof buf, "%.2f", a8.distance);
        out << buf << ' ' << route_text(a8.route) << " | q4 "
            << measured::table_disposition_name(frontier.disposition) << ' '
            << board_text(frontier.donor) << ' ';
        std::snprintf(buf, sizeof buf, "%.2f", frontier.distance);
        out << buf << ' ' << frontier.min_tokens << '\n';
    }
    return out.str();
}

void check_against_production() {
    // The answer's value is the production lookup's on the same board (a substituted class).
    for (const generic::FactsBoard& fb : generic::kFactsBoards) {
        const DeviceClass d = generic::board_device_class(fb);
        const ScopedDeviceClass scoped(d);
        const Board board = board_of(device_class());
        const auto a8     = moe::a8_route_answer(board);
        const auto prod   = moe::a8_default_route();
        expect(prod && same_route(*prod, a8.route),
               std::string("a8 route answer differs from a8_default_route on ") + fb.name);
        const auto frontier = detail::sparse_moe_q4_groupwise_frontier_answer(board);
        expect(frontier.min_tokens == detail::sparse_moe_prefill_min_tokens(
                                          QType::Q4_G64_FP16, QType::Q5_G64_FP16, false),
               std::string("frontier answer differs from the prefill minimum on ") + fb.name);
    }
}

void check_measured_boards_and_fallback() {
    for (const Board& b : {measured::kRtx4090, measured::kRtx5090, measured::kRtx5080}) {
        const auto a8 = moe::a8_route_answer(b);
        expect(a8.disposition == TableDisposition::Own && a8.distance == 0 && a8.donor,
               "a measured board's A8 route is Own");
    }
    for (const Board& b : {measured::kRtx5090, measured::kRtx4090, measured::kRtx3090}) {
        const auto f = detail::sparse_moe_q4_groupwise_frontier_answer(b);
        expect(f.disposition == TableDisposition::Own && f.distance == 0 && f.donor,
               "a measured board's frontier is Own");
    }
    // The V100 is far from every measured board of both tables: the declared literals.
    const auto a8 = moe::a8_route_answer(measured::kV100Sxm2);
    expect(a8.disposition == TableDisposition::Fallback && !a8.donor && a8.distance == 0 &&
               a8.route.down == -1,
           "the V100 takes the A8 fallback literal");
    const auto f = detail::sparse_moe_q4_groupwise_frontier_answer(measured::kV100Sxm2);
    expect(f.disposition == TableDisposition::Fallback && !f.donor &&
               f.min_tokens == detail::kSparseMoePrefillQ4Q5Min,
           "the V100 takes the frontier fallback literal");
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool update = false;
        for (int i = 1; i < argc; ++i) update = update || std::string_view(argv[i]) == "--update";
        check_against_production();
        check_measured_boards_and_fallback();
        const std::string now = dump();
        const std::filesystem::path path =
            std::filesystem::path(NINFERNO_TABLE_DISPOSITIONS_DIR) / "dispositions.txt";
        if (update) {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream file(path, std::ios::binary);
            file << now;
            std::cout << "wrote " << path.string() << '\n';
        } else {
            std::ifstream file(path, std::ios::binary);
            const std::string was((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
            if (was != now) {
                std::cerr << "table dispositions: " << path.string()
                          << (was.empty() ? " is missing" : " differs from this build's answers")
                          << " (run with --update after a change that means to move them)\n";
                ++failures;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "table dispositions: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
