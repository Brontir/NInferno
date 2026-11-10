// What the measured value tables' answers rest on (#616 step 3), host only, no GPU: the generic MoE
// op's class A8 route (moe::a8_route_answer) and the sparse MoE's Q4/Q5 groupwise prefill frontier
// (sparse_moe_q4_groupwise_frontier_answer) for every target SKU of the reference table
// (core/data/gpu_facts.json): Measured, Carried (donor, distance), Incomplete (the nearest entry
// has a fact unknown) or None (the table's declared literal), in the vocabulary of the measured
// gates. Compared with table_dispositions/dispositions.txt (keyed by PCI id; --update rewrites it,
// from the code under test, so it only pins what the code does). Also checks that the disposition's
// value is the one the production lookup returns on the same board (the disposition never disagrees
// with the answer), that the route's evidence kind follows the answer's disposition, that every
// measured board answers Measured for itself, that a literal has no donor, the counts per
// disposition that docs/maintainer/table-dispositions.md quotes, and a few SKUs whose answer is
// written by hand from the measurement comments beside the tables (the independent check).
#include "core/device_class.h"
#include "core/gate_atlas.h"
#include "core/machine_facts.h"
#include "core/measured_boards.h"
#include "core/weight.h"
#include "ops/generic/selector_facts.h"
#include "ops/moe/moe_internal.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include <cmath>
#include <cstdint>
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
using measured::Disposition;

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

// Rows per disposition, over the target SKUs; the counts docs/maintainer/table-dispositions.md
// quotes (a change of either moves both, or this test fails).
struct Counts {
    int measured = 0, carried = 0, incomplete = 0, none = 0;

    void add(Disposition d) {
        switch (d) {
        case Disposition::Measured:
            ++measured;
            break;
        case Disposition::Carried:
            ++carried;
            break;
        case Disposition::Incomplete:
            ++incomplete;
            break;
        default:
            ++none;
            break;
        }
    }
};

std::string dump(Counts& a8_counts, Counts& q4_counts) {
    std::ostringstream out;
    out << "# PCI id SKU | a8 route: disposition donor distance route | q4 frontier: disposition "
           "donor distance min_tokens\n";
    for (const GpuFactsRow& row : gpu_facts_table()) {
        if (!row.target) continue;
        const Board board   = measured::table_board(row);
        const auto a8       = moe::a8_route_answer(board);
        const auto frontier = detail::sparse_moe_q4_groupwise_frontier_answer(board);
        a8_counts.add(a8.disposition);
        q4_counts.add(frontier.disposition);
        char buf[64];
        std::snprintf(buf, sizeof buf, "0x%04X ", static_cast<unsigned>(row.pci_device_id));
        out << buf << row.name << " | a8 " << measured::disposition_name(a8.disposition) << ' '
            << board_text(a8.donor) << ' ';
        std::snprintf(buf, sizeof buf, "%.2f", a8.distance);
        out << buf << ' ' << route_text(a8.route) << " | q4 "
            << measured::disposition_name(frontier.disposition) << ' ' << board_text(frontier.donor)
            << ' ';
        std::snprintf(buf, sizeof buf, "%.2f", frontier.distance);
        out << buf << ' ' << frontier.min_tokens << '\n';
    }
    return out.str();
}

void check_against_production() {
    // The answer's value is the production lookup's on the same board (a substituted class), and
    // the route's evidence kind is the answer's disposition, from the one answer.
    for (const generic::FactsBoard& fb : generic::kFactsBoards) {
        const DeviceClass d = generic::board_device_class(fb);
        const ScopedDeviceClass scoped(d);
        const Board board = board_of(device_class());
        const auto a8     = moe::a8_route_answer(board);
        const auto prod   = moe::a8_default_route();
        expect(prod && same_route(*prod, a8.route),
               std::string("a8 route answer differs from a8_default_route on ") + fb.name);
        const moe::Evidence ev = moe::class_a8_route_evidence();
        const bool donor_kind =
            a8.disposition == Disposition::Measured || a8.disposition == Disposition::Carried;
        expect(donor_kind == (ev.kind == moe::Evidence::Kind::Measured ||
                              ev.kind == moe::Evidence::Kind::Carried),
               std::string("a8 evidence kind does not follow the answer's disposition on ") +
                   fb.name);
        if (donor_kind) {
            expect((ev.kind == moe::Evidence::Kind::Measured) ==
                           (a8.disposition == Disposition::Measured) &&
                       ev.distance == a8.distance && a8.donor && !ev.donor.empty(),
                   std::string("a8 evidence differs from the answer on ") + fb.name);
        } else {
            expect(ev.donor == "fallback-a8-route" && !a8.donor,
                   std::string("a8 fallback evidence names a donor on ") + fb.name);
        }
        const auto frontier = detail::sparse_moe_q4_groupwise_frontier_answer(board);
        expect(frontier.min_tokens == detail::sparse_moe_prefill_min_tokens(
                                          QType::Q4_G64_FP16, QType::Q5_G64_FP16, false),
               std::string("frontier answer differs from the prefill minimum on ") + fb.name);
    }
}

void check_measured_boards_and_fallback() {
    for (const Board& b : {measured::kRtx4090, measured::kRtx5090, measured::kRtx5080}) {
        const auto a8 = moe::a8_route_answer(b);
        expect(a8.disposition == Disposition::Measured && a8.distance == 0 && a8.donor,
               "a measured board's A8 route is Measured");
    }
    for (const Board& b : {measured::kRtx5090, measured::kRtx4090, measured::kRtx3090}) {
        const auto f = detail::sparse_moe_q4_groupwise_frontier_answer(b);
        expect(f.disposition == Disposition::Measured && f.distance == 0 && f.donor,
               "a measured board's frontier is Measured");
    }
    // The V100 is far from every measured board of both tables: the declared literals.
    const auto a8 = moe::a8_route_answer(measured::kV100Sxm2);
    expect(a8.disposition == Disposition::None && !a8.donor && a8.distance == 0 &&
               a8.route.down == -1,
           "the V100 takes the A8 fallback literal");
    const auto f = detail::sparse_moe_q4_groupwise_frontier_answer(measured::kV100Sxm2);
    expect(f.disposition == Disposition::None && !f.donor &&
               f.min_tokens == detail::kSparseMoePrefillQ4Q5Min,
           "the V100 takes the frontier fallback literal");
    // A fact unknown against the nearest entry is Incomplete, not None, and still takes the
    // literal: the RTX 3090's frontier row with its L2 unknown, and the RTX 5090's A8 row.
    Board no_l2_3090  = measured::kRtx3090;
    no_l2_3090.l2_mib = 0;
    const auto fi     = detail::sparse_moe_q4_groupwise_frontier_answer(no_l2_3090);
    expect(fi.disposition == Disposition::Incomplete && !fi.donor &&
               fi.min_tokens == detail::kSparseMoePrefillQ4Q5Min,
           "a 3090 without its L2 is Incomplete, with the literal");
    Board no_l2_5090  = measured::kRtx5090;
    no_l2_5090.l2_mib = 0;
    const auto ai     = moe::a8_route_answer(no_l2_5090);
    expect(ai.disposition == Disposition::Incomplete && !ai.donor && ai.route.down == -1,
           "a 5090 without its L2 is Incomplete, with the literal");
}

// One SKU's answers written by hand, each from the measurement or the distance recorded in the
// source beside its table (not read off the code under test): the A8 disposition and its route's
// down schedule, the frontier's disposition and first T, and the distance where the source names
// one.
struct Expected {
    std::uint32_t id;
    const char* why;
    Disposition a8;
    int a8_down;
    Disposition q4;
    int q4_min;
    double a8_distance = -1; // < 0: not checked
    double q4_distance = -1;
};

void check_hand_written() {
    constexpr Disposition M = Disposition::Measured, C = Disposition::Carried,
                          I = Disposition::Incomplete, N = Disposition::None;
    const Expected rows[] = {
        // moe.cpp, kA8Routes: gb202's down is the streaming register MMA (-1); the frontier
        // table (sparse_moe_prefill_plan.cpp) has the row, 47.
        {0x2B85, "RTX 5090: a row of both tables", M, -1, M, 47},
        // moe.cpp: ad102's down is 2 x 2 (schedule 2); the frontier table has the row, 47.
        {0x2684, "RTX 4090: a row of both tables", M, 2, M, 47},
        // measured_boards.h (kCarryDistance): the 4090 D is 0.17 from the 4090.
        {0x2685, "RTX 4090 D: a variant of the 4090", C, 2, C, 47, 0.17, 0.17},
        // moe.cpp: gb203's down is 4 x 1 (schedule 3); the frontier table has no 5080 row and the
        // 5090's is beyond reach, so its literal, 47.
        {0x2C02, "RTX 5080: an A8 row, no frontier row", M, 3, N, 47},
        // measured_boards.h: the 5070 Ti is 0.57 from the 5080.
        {0x2C05, "RTX 5070 Ti: near the 5080", C, 3, N, 47, 0.57},
        // sparse_moe_prefill_plan.cpp: the 3090 measured 17; moe.cpp lists no Ampere row.
        {0x2204, "RTX 3090: the Ampere frontier row", N, -1, M, 17},
        // measured_boards.h: the 3090 Ti is 0.14 from the 3090.
        {0x2203, "RTX 3090 Ti: a variant of the 3090", N, -1, C, 17, -1, 0.14},
        // measured_boards.h: the 4090's entries on the 4080 SUPER (1.2) are out of reach.
        {0x2702, "RTX 4080 SUPER: beyond kCarryDistance of the 4090", N, -1, N, 47},
        // moe.cpp: "ad106 (RTX 4060 Ti, no entry)": the unmeasured default.
        {0x2803, "RTX 4060 Ti: no entry", N, -1, N, 47},
        // gpu_facts.json gives no l2_mib for these three SKUs, so the nearest measured entry has a
        // fact unknown (a device reports its L2 itself, and may be Carried); Incomplete keeps the
        // literal. The A10G and the 3080 12GB are Ampere (no A8 row), near the 3090.
        {0x2237, "A10G: no L2 in the facts table, near the 3090", N, -1, I, 47},
        {0x220A, "RTX 3080 12GB: no L2 in the facts table, near the 3090", N, -1, I, 47},
        {0x2BB5, "RTX PRO 6000 Server: no L2 in the facts table, near the 5090", I, -1, I, 47},
    };
    for (const Expected& e : rows) {
        const GpuFactsRow* row = find_gpu_facts(e.id);
        if (row == nullptr) {
            expect(false, std::string("no facts row for ") + e.why);
            continue;
        }
        const Board board      = measured::table_board(*row);
        const auto a8          = moe::a8_route_answer(board);
        const auto frontier    = detail::sparse_moe_q4_groupwise_frontier_answer(board);
        const std::string what = std::string(" for ") + e.why;
        expect(a8.disposition == e.a8, "a8 disposition" + what);
        expect(a8.route.down == e.a8_down, "a8 down schedule" + what);
        expect(frontier.disposition == e.q4, "frontier disposition" + what);
        expect(frontier.min_tokens == e.q4_min, "frontier min tokens" + what);
        if (e.a8_distance >= 0) {
            expect(std::abs(a8.distance - e.a8_distance) < 0.005, "a8 distance" + what);
        }
        if (e.q4_distance >= 0) {
            expect(std::abs(frontier.distance - e.q4_distance) < 0.005, "frontier distance" + what);
        }
        if (e.a8 == I || e.q4 == I) {
            expect(row->l2_mib == 0, "an Incomplete row lacks its L2" + what);
        }
    }
    // gb202's IQ2_XXS goes w4r2 from T = 4, w4r1 before; gb203's IQ2_S takes w2r1 at T = 4..7 and
    // w4r1 at 8 (moe.cpp, the 2026-11-03 sweep).
    const auto r5090 = moe::a8_route_answer(measured::kRtx5090).route;
    expect(r5090.gate_up_2bit[0][2] == 0 && r5090.gate_up_2bit[0][3] == 1,
           "5090 IQ2_XXS T = 3 / 4");
    const auto r5080 = moe::a8_route_answer(measured::kRtx5080).route;
    expect(r5080.gate_up_2bit[2][3] == 3 && r5080.gate_up_2bit[2][6] == 3 &&
               r5080.gate_up_2bit[2][7] == 0,
           "5080 IQ2_S T = 4..7 w2r1, 8 w4r1");
}

// The counts docs/maintainer/table-dispositions.md quotes over the 105 target SKUs
// (measured / carried / incomplete / none).
void check_counts(const Counts& a8, const Counts& q4) {
    const auto same = [](const Counts& c, int m, int ca, int i, int n) {
        return c.measured == m && c.carried == ca && c.incomplete == i && c.none == n;
    };
    expect(same(a8, 0, 0, 0, 0), "A8 route counts differ from the document");
    expect(same(q4, 0, 0, 0, 0), "frontier counts differ from the document");
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool update = false;
        for (int i = 1; i < argc; ++i) update = update || std::string_view(argv[i]) == "--update";
        check_against_production();
        check_measured_boards_and_fallback();
        Counts a8_counts, q4_counts;
        const std::string now = dump(a8_counts, q4_counts);
        check_hand_written();
        check_counts(a8_counts, q4_counts);
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
