// The rows a restricted draft head keeps (spec/draft_head.h): the bytes/acceptance bound on the
// repository's Qwen token ranking, and its limits on synthetic counts. No GPU.

#include "spec/draft_head.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace sp = ninferno::spec;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<std::int64_t> qwen_ranking() {
    // Row 0 of the ranking (total counts over the Qwen 248,320-id vocabulary).
    std::ifstream in(std::string(NINFERNO_SOURCE_DIR) +
                         "/tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64",
                     std::ios::binary);
    std::vector<std::int64_t> counts(248320);
    in.read(reinterpret_cast<char*>(counts.data()),
            static_cast<std::streamsize>(counts.size() * sizeof(std::int64_t)));
    if (!in) { counts.clear(); }
    return counts;
}

} // namespace

int main() {
    const auto counts = qwen_ranking();
    expect(!counts.empty(), "the repository's Qwen ranking reads");
    if (!counts.empty()) {
        // Qwen3.5-9B Q4_K_M with z-lab's DFlash drafter (the registry's round facts): head
        // 834,355,200 B of a round's 5,117,970,432 + 1,372,703,744 + 834,355,200.
        const double share = 834355200.0 / (5117970432.0 + 1372703744.0 + 834355200.0);
        const auto r7      = sp::draft_head_rows(counts, share, 7);
        expect(r7.rows == 147456, "9B dflash:7 keeps 147456 rows, got " + std::to_string(r7.rows));
        expect(std::abs(r7.gain - 0.0361) < 0.001, "9B dflash:7 bound +3.6 %");
        expect(r7.coverage > 0.996, "9B dflash:7 coverage above 99.6 %");
        // Deeper rounds lose more to a missing id: more rows; shallower ones fewer.
        const auto r3  = sp::draft_head_rows(counts, share, 3);
        const auto r15 = sp::draft_head_rows(counts, share, 15);
        expect(r3.rows == 106496, "9B depth 3 keeps 106496 rows, got " + std::to_string(r3.rows));
        expect(r15.rows == 172032, "9B depth 15 keeps 172032 rows, got " + std::to_string(r15.rows));
        // MTP reads the head once a draft: 9B mtp:3 (MTP pass 1,002,113,024 B with its head).
        const double mtp_share =
            3 * 834355200.0 / (5117970432.0 + 3 * 1002113024.0);
        const auto mtp = sp::draft_head_rows(counts, mtp_share, sp::kDraftHeadMtpDepth);
        expect(mtp.rows == 57344 && mtp.gain > 0.2,
               "9B mtp:3 keeps 57344 rows, got " + std::to_string(mtp.rows));
        // A head dearer in the round (a small active MoE): fewer rows, more gain.
        const auto heavy = sp::draft_head_rows(counts, 0.3, 7);
        expect(heavy.rows == 98304 && heavy.gain > 0.15, "share 0.3 keeps 98304 rows");
    }
    // Every id equally frequent: no restriction pays (each removed row costs its share of drafts).
    const std::vector<std::int64_t> flat(65536, 1);
    expect(sp::draft_head_rows(flat, 0.1, 7).rows == 0, "a flat ranking keeps the full head");
    // Degenerate facts keep the full head.
    expect(sp::draft_head_rows(counts, 0.0, 7).rows == 0, "no head share keeps the full head");
    expect(sp::draft_head_rows(counts, 0.1, 0).rows == 0, "no drafts keep the full head");
    expect(sp::draft_head_rows(std::vector<std::int64_t>(100, 5), 0.1, 7).rows == 0,
           "a vocabulary under one step keeps the full head");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
