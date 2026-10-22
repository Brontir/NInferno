// The position-aligned KV splits of the invariant route profile (CausalKvPartition, --activations
// exact), no GPU:
//   - a host replica of the aligned BF16 grouped kernel's split walk (bf16/grouped_mma.cuh: one
//     CTA per split of its launch window, rows whose own splits are finer close them at their
//     boundaries, a store at the end) gives every token exactly the partials of a decode of that
//     token alone, whatever cuts its prompt into calls and 64-column launches: one call, random
//     cuts, decode steps one by one, across the 32768- and 65536-key shift boundaries and for the
//     query-head groups of every causal geometry;
//   - every launch's CTA splits fit the capacity its envelope plans;
//   - the split plan (keys per split, splits, CTAs of a 64-column prefill launch and of a decode
//     step, per SM) at 4K..32K keys on 108/82/170/84/80/46 SMs, printed. The partition follows
//     the position only, so it is the same on every card; the CTAs per SM show the fill.

#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {
using ninferno::ops::detail::Bf16KvPartition;
using ninferno::ops::detail::CausalAlignedRow;
using ninferno::ops::detail::CausalKvPartition;

constexpr int kQueryRows   = 32; // the aligned BF16 instance (Grouped32)
constexpr int kKeyRows     = 32;
constexpr int kLaunchWidth = 64; // kCausalInvariantChunkColumns

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok && ++failures <= 12) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

struct Segment {
    int lo = 0, hi = 0; // the keys a partial sums: [lo, hi)
    int stores = 0;
};

// Per row: the partials it stored, by its own split index. A token's first and last query head
// are kept (the rows a 32-row tile can part from the token's other heads): index
// ((token * 2 + last head) * kAlignedSplits + split).
using Partials        = std::vector<Segment>;
constexpr int kSplits = CausalKvPartition::kAlignedSplits;

// The aligned BF16 grouped kernel's walk for one launch of `positions` (consecutive columns of one
// request) at query-head group `group`: per CTA split of the launch window and per 32-row tile,
// which keys each row sums into which of its own splits.
void replay_launch(const std::vector<int>& positions, int group, int capacity, Partials& out) {
    const int live       = static_cast<int>(positions.size());
    const int window     = positions[live - 1] + 1;
    const auto work      = Bf16KvPartition::aligned_live(window);
    const int tile_shift = CausalKvPartition::aligned_key_shift(window);
    expect(work.keys_per_split == (1 << tile_shift), "live keys follow the window's shift");
    expect(work.splits <= capacity, "a launch of " + std::to_string(work.splits) +
                                        " splits fits its capacity " + std::to_string(capacity));
    const int rows = live * group;
    for (int row_begin = 0; row_begin < rows; row_begin += kQueryRows) {
        const int last_token = std::min(live, (row_begin + kQueryRows + group - 1) / group) - 1;
        for (int split = 0; split < work.splits; ++split) {
            const int start = split * work.keys_per_split;
            const int stop  = std::min(window, start + work.keys_per_split);
            const int end   = std::min(stop, positions[last_token] + 1);
            if (start >= end) continue; // aligned: no row reaches these keys, no store
            const int last_k0 = start + (end - start - 1) / kKeyRows * kKeyRows;
            for (int row = row_begin; row < std::min(rows, row_begin + kQueryRows); ++row) {
                const int token = row / group, head = row % group;
                if (head != 0 && head != group - 1) continue;
                const int p                = positions[token];
                const std::size_t slot     = std::size_t(p) * 2 + (head == 0 ? 0 : 1);
                const CausalAlignedRow own = CausalKvPartition::aligned_row(p);
                // The keys this row sums since its last reset: [seg_lo, k) clipped to <= p.
                int seg_lo       = start;
                const auto store = [&](int split_index, int lo, int hi) {
                    if (!own.stores(split_index)) return;
                    auto& seg = out[slot * kSplits + split_index];
                    seg.lo    = lo;
                    seg.hi    = std::min(hi, p + 1);
                    ++seg.stores;
                };
                // The kernel checks closes() at each key tile past the first on the minimum split
                // grid; a row closes only at its own split boundaries.
                const int grid = 1 << CausalKvPartition::kAlignedMinKeyShift;
                for (int k0 = (start / grid + 1) * grid; k0 < end; k0 += grid) {
                    if (k0 <= start || k0 % kKeyRows != 0) continue;
                    if (!own.closes(k0, tile_shift)) continue;
                    store(own.split_of(k0) - 1, seg_lo, k0);
                    seg_lo = k0;
                }
                store(own.split_of(last_k0), seg_lo, end);
            }
        }
    }
}

// The decode of token p alone: split j of its own shift holds [j << s, min((j + 1) << s, p + 1)).
void check_against_decode(const Partials& got, int length, const std::string& label) {
    int bad = 0;
    for (int p = 0; p < length && bad < 4; ++p) {
        const CausalAlignedRow own = CausalKvPartition::aligned_row(p);
        const auto merged          = Bf16KvPartition::aligned_live(p + 1);
        bool ok                    = own.splits <= kSplits && merged.splits == own.splits &&
                  merged.keys_per_split == (1 << own.shift);
        int split = 0;
        for (; ok && split < 2 * kSplits; ++split) {
            const auto& seg = got[std::size_t(p) * 2 * kSplits + split];
            const int j     = split % kSplits;
            if (j >= own.splits) {
                ok = seg.stores == 0;
                continue;
            }
            const int lo = j << own.shift, hi = std::min((j + 1) << own.shift, p + 1);
            ok = seg.stores == 1 && seg.lo == lo && seg.hi == hi;
        }
        if (!ok) {
            ++bad;
            expect(false, label + ": token " + std::to_string(p) + " head " +
                              (split <= kSplits ? "first" : "last") + " split " +
                              std::to_string(std::max(split - 1, 0) % kSplits) +
                              " differs from its decode's (" + std::to_string(own.splits) +
                              " splits of " + std::to_string(1 << own.shift) + " keys)");
        }
    }
}

// A prompt of `length` tokens cut into calls at `cuts` (call starts past 0), each call run as
// launches of up to 64 columns over an envelope of the call's last key.
Partials replay_prompt(int length, const std::vector<int>& cuts, int group) {
    Partials out(std::size_t(length) * 2 * kSplits);
    std::vector<int> starts{0};
    for (const int c : cuts)
        if (c > starts.back() && c < length) starts.push_back(c);
    starts.push_back(length);
    for (std::size_t c = 0; c + 1 < starts.size(); ++c) {
        const int capacity = CausalKvPartition::aligned_capacity(starts[c + 1]);
        for (int first = starts[c]; first < starts[c + 1]; first += kLaunchWidth) {
            std::vector<int> positions;
            for (int p = first; p < std::min(first + kLaunchWidth, starts[c + 1]); ++p)
                positions.push_back(p);
            replay_launch(positions, group, capacity, out);
        }
    }
    return out;
}

void check_chunkings() {
    constexpr int length = 70000; // past the 32768- and 65536-key shift boundaries
    std::mt19937 rng(20261021u);
    for (const int group : {4, 6, 8}) { // 16/4 and 8/2, 24/4, 16/2
        const std::string g = "group " + std::to_string(group) + " ";
        check_against_decode(replay_prompt(length, {}, group), length, g + "one call");
        for (int trial = 0; trial < 3; ++trial) {
            std::vector<int> cuts;
            for (int at = 0; at < length;) {
                at += 1 + static_cast<int>(rng() % (trial == 0 ? 97u : trial == 1 ? 2049u : 9000u));
                cuts.push_back(at);
            }
            // Calls that start just below a shift boundary (their launches straddle it, so the
            // rows below it close their finer splits inside a CTA), or on it. The random cuts
            // between keep the next call short of the following one.
            const std::vector<int> boundary = trial == 0   ? std::vector<int>{4095, 32767, 65535}
                                              : trial == 1 ? std::vector<int>{32705, 65473}
                                                           : std::vector<int>{32768, 65536};
            cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                                      [&](int c) {
                                          for (const int at : boundary)
                                              if (c > at && c <= at + 64) return true;
                                          return false;
                                      }),
                       cuts.end());
            cuts.insert(cuts.end(), boundary.begin(), boundary.end());
            std::sort(cuts.begin(), cuts.end());
            check_against_decode(replay_prompt(length, cuts, group), length,
                                 g + "random cuts " + std::to_string(trial));
        }
        std::vector<int> every;
        for (int p = 1; p < length; ++p) every.push_back(p);
        check_against_decode(replay_prompt(length, every, group), length, g + "decode steps");
    }
}

// The rule this replaced (64-key minimum), for the plan table only.
int previous_shift(int visible) {
    int shift = 6;
    while ((CausalKvPartition::kAlignedSplits << shift) < visible) ++shift;
    return shift;
}

void print_plan() {
    // Qwen3.5-9B / Qwen3.8 attention: 16 query heads over 4 KV heads; a 64-column prefill launch
    // runs 4 x 64 * 4 / 32 = 32 row tiles, a decode step 4.
    constexpr int kv_heads = 4, group = 4;
    constexpr int prefill_tiles = kv_heads * kLaunchWidth * group / kQueryRows;
    constexpr int decode_tiles  = kv_heads;
    const int sms[]             = {108, 82, 170, 84, 80, 46};
    std::printf("aligned split plan (16/4 heads, %d prefill tiles, %d decode tiles), old = 64-key "
                "minimum, new = %d-key minimum; per SM: prefill CTAs (decode CTAs) per SM\n",
                prefill_tiles, decode_tiles, 1 << CausalKvPartition::kAlignedMinKeyShift);
    std::printf("%6s %11s %9s %13s", "keys", "keys/split", "splits", "prefill CTAs");
    for (const int s : sms) std::printf(" %10d", s);
    std::printf("\n");
    for (int v = 4096; v <= 32768; v += 4096) {
        const int old_shift  = previous_shift(v);
        const int old_splits = (v + (1 << old_shift) - 1) >> old_shift;
        const auto live      = Bf16KvPartition::aligned_live(v);
        std::printf("%6d %5d->%-5d %4d->%-4d %6d->%-6d", v, 1 << old_shift, live.keys_per_split,
                    old_splits, live.splits, prefill_tiles * old_splits,
                    prefill_tiles * live.splits);
        for (const int s : sms)
            std::printf(" %5.1f(%3.1f)", double(prefill_tiles * live.splits) / s,
                        double(decode_tiles * live.splits) / s);
        std::printf("\n");
    }
}
} // namespace

int main() {
    expect(CausalKvPartition::aligned_key_shift(1) == CausalKvPartition::kAlignedMinKeyShift,
           "the shortest call splits at the minimum");
    expect(CausalKvPartition::aligned_key_shift(32768) == 9, "512 keys up to 32768");
    expect(CausalKvPartition::aligned_key_shift(32769) == 10, "1024 keys past 32768");
    expect(CausalKvPartition::aligned_key_shift(1 << 20) == 14, "16384 keys at 1M");
    expect(CausalKvPartition::aligned_capacity(32768) == 64 &&
               CausalKvPartition::aligned_capacity(1 << 20) == 64 &&
               CausalKvPartition::aligned_capacity(512) == 1 &&
               CausalKvPartition::aligned_capacity(513) == 2,
           "capacity: ceil(keys / 512), at most 64");
    check_chunkings();
    print_plan();
    if (failures)
        std::fprintf(stderr, "%d failure(s)\n", failures);
    else
        std::printf("position partition: every chunking gives each token its decode's splits\n");
    return failures ? 1 : 0;
}
