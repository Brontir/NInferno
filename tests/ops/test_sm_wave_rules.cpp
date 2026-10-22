// The SM-count launch rules (ops/common/sm_wave_rules.h) against the literals they replaced: on the
// RTX 5090's 170 SMs every decision is the old one, at every width; on other SM counts the test
// prints what the property rule decides instead and pins it. The Gemma 4 Q8 5376x16384 tile is no
// SM-count rule: it keeps master's literal (q8/shapes/n5376_k16384.cu), the same on 170 SMs. No GPU.
#include "ops/common/sm_wave_rules.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using namespace ninferno::ops::detail;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "sm wave rules: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kTunedSms = 170; // the literal the rules replaced

// The FP8 TMA final-wave split before: the wave was the device's SMs capped at the schedule's
// WaveCtas (170), and 170 where the device had no SM count.
sm_rules::SplitTail old_fp8_split(int tiles, int k, int sms, int block_k, int max_parts) {
    const int wave = sms > 0 ? std::min(sms, kTunedSms) : kTunedSms;
    sm_rules::SplitTail plan{tiles, 0, 0};
    const int tail = tiles % wave;
    if (tail && tail <= wave / 2) {
        const int parts = std::min(max_parts, k / block_k);
        if (parts >= 2) plan = {tiles - tail, tail, std::min(wave, tail * parts)};
    }
    return plan;
}

bool same(const sm_rules::SplitTail& a, const sm_rules::SplitTail& b) {
    return a.full_tiles == b.full_tiles && a.tail_tiles == b.tail_tiles &&
           a.split_ctas == b.split_ctas;
}

// Gemma 4 NVFP4 2048x5376 A4 past T = 640 before: the 128-token MMA tile through 1280, TMA after.
bool old_nvfp4_tma(std::int32_t tokens) { return tokens > 1280; }

bool new_nvfp4_tma(std::int32_t tokens, int sms) {
    return sm_rules::grid_passes_wave(tokens, sms, 2048 / 128, 128);
}

constexpr std::int32_t kMaxTokens = 65536;
constexpr int kBlockK             = 128;
constexpr int kMaxParts           = 4;
// The K of every shape that runs an FP8 split schedule.
constexpr int kFp8Ks[] = {5120, 6144, 17408};

void check_tuned_board_unchanged() {
    for (int k : kFp8Ks) {
        for (int tiles = 1; tiles <= 16384; ++tiles) {
            expect(same(sm_rules::fp8_split_tail(tiles, k, kTunedSms, kBlockK, kMaxParts),
                        old_fp8_split(tiles, k, kTunedSms, kBlockK, kMaxParts)),
                   "fp8 split differs on 170 SMs at tiles " + std::to_string(tiles) + ", K " +
                       std::to_string(k));
        }
    }
    for (std::int32_t t = 641; t <= kMaxTokens; ++t) {
        expect(new_nvfp4_tma(t, kTunedSms) == old_nvfp4_tma(t),
               "nvfp4 2048x5376 route differs on 170 SMs at T " + std::to_string(t));
    }
}

// The FP8 split differs from the old one only where the old cap bit: more than 170 SMs.
void report_fp8(int sms) {
    int changed = 0;
    for (int k : kFp8Ks) {
        for (int tiles = 1; tiles <= 16384; ++tiles) {
            if (!same(sm_rules::fp8_split_tail(tiles, k, sms, kBlockK, kMaxParts),
                      old_fp8_split(tiles, k, sms, kBlockK, kMaxParts))) {
                ++changed;
            }
        }
    }
    expect((changed == 0) == (sms <= kTunedSms), "fp8 split on " + std::to_string(sms) +
                                                     " SMs: " + std::to_string(changed) +
                                                     " changed (tiles, K) pairs");
    std::printf("  fp8 TMA final-wave split: %d of %d (tiles, K) plans changed\n", changed,
                16384 * 3);
}

void report_nvfp4(int sms) {
    std::int32_t first_tma = 0;
    for (std::int32_t t = 641; t <= kMaxTokens && first_tma == 0; ++t) {
        if (new_nvfp4_tma(t, sms)) first_tma = t;
    }
    std::printf("  nvfp4 2048x5376 A4 TMA route: from T = %d (old: from 1281)\n", first_tma);
}

} // namespace

int main() {
    check_tuned_board_unchanged();
    std::printf("170 SMs: every decision unchanged (fp8 split, tiles 1..16384 x K %d/%d/%d; "
                "nvfp4 route to T %d)\n",
                kFp8Ks[0], kFp8Ks[1], kFp8Ks[2], kMaxTokens);

    // Boards of the lab and the PCI-ID table: 84 RTX 5080, 76 RTX 4080, 58 L4, 46 RTX 4070,
    // 34 RTX 4060 Ti, 108 A100, 80 V100 / RTX 4080 SUPER, 68 RTX 2080 Ti, 188 RTX PRO
    // 6000, 128 RTX 4090, 82 RTX 3090, 70 RTX 5070 Ti, 48 RTX 5070, 36 RTX 5060 Ti.
    for (int sms : {84, 76, 58, 46, 34, 108, 80, 68, 188, 128, 82, 70, 48, 36}) {
        std::printf("%d SMs:\n", sms);
        report_fp8(sms);
        report_nvfp4(sms);
    }

    expect(new_nvfp4_tma(641, 84) && !new_nvfp4_tma(640, 84), "84 SMs: TMA from 641");
    expect(!new_nvfp4_tma(1408, 188) && new_nvfp4_tma(1409, 188), "188 SMs: TMA from 1409");
    // No SM count: no split, no TMA (no device runs then).
    expect(same(sm_rules::fp8_split_tail(171, 5120, 0, kBlockK, kMaxParts), {171, 0, 0}),
           "0 SMs: no split");
    expect(!new_nvfp4_tma(4096, 0), "0 SMs: no wave rule fires");
    // Resident CTAs an SM widen the wave: 85 SMs at two CTAs an SM is the 5090's 170-slot wave.
    expect(!sm_rules::grid_passes_wave(1280, 85, 16, 128, 2) &&
               sm_rules::grid_passes_wave(1281, 85, 16, 128, 2),
           "85 SMs x 2 CTAs: TMA from 1281");

    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("sm wave rules: all checks passed\n");
    return 0;
}
