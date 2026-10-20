// Source-consuming conversion bookkeeping: a checkpoint block is punched only after every object
// reading it has been written, synced and journaled (in that order), never when pinned, never
// twice; every block that may be punched eventually is; and resumed runs keep the same rule.

#include "convert/consume.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ninferno::convert;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok && failures++ < 20) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}

constexpr std::uint64_t kBlock = 4096;

struct Layout {
    std::vector<std::uint64_t> shards;
    std::vector<SourceSegment> segments;
    std::size_t objects = 0;
};

// Segments cover every shard exactly: a pinned header, then tensors of mixed sizes (many far
// smaller than a block, so blocks are shared), some unread, some pinned, some read by several
// objects.
Layout random_layout(std::mt19937_64& rng) {
    Layout out;
    out.objects       = 1 + rng() % 12;
    const auto shards = 1 + rng() % 3;
    for (std::size_t s = 0; s < shards; ++s) {
        std::uint64_t cursor = 100 + rng() % 9000;
        out.segments.push_back({static_cast<int>(s), 0, cursor, {}, true});
        const auto tensors = 1 + rng() % 40;
        for (std::size_t t = 0; t < tensors; ++t) {
            const auto kind  = rng() % 4;
            const auto bytes = kind == 0 ? 4 : kind == 1 ? 1 + rng() % 3000 : 1 + rng() % 50000;
            SourceSegment segment{static_cast<int>(s), cursor, cursor + bytes, {}, false};
            const auto r = rng() % 10;
            if (r == 0) {
                segment.pinned = true; // a divisor scalar the plan reads
            } else if (r != 1) {       // r == 1: unread
                const auto readers = 1 + rng() % 3;
                for (std::size_t i = 0; i < readers; ++i) {
                    segment.readers.push_back(static_cast<std::uint32_t>(rng() % out.objects));
                }
            }
            out.segments.push_back(segment);
            cursor += bytes;
        }
        out.shards.push_back(cursor);
    }
    return out;
}

// The only I/O the committer performs, checked against the invariant as it happens.
struct CheckingIo final : CommitIo {
    const Layout& layout;
    std::set<std::uint32_t> written;                      // objects whose bytes are complete
    std::set<std::uint32_t> synced;                       // written as of the last output sync
    std::set<std::uint32_t> journaled;                    // durable
    std::map<std::pair<int, std::uint64_t>, int> punched; // block -> times
    std::string context;

    explicit CheckingIo(const Layout& l) : layout(l) {}

    void sync_output() override { synced = written; }

    void journal(std::span<const std::uint32_t> ids) override {
        for (const auto id : ids) {
            check(synced.contains(id), context + ": object journaled before its data was synced");
            journaled.insert(id);
        }
    }

    void punch(const Punch& punch) override {
        check(punch.offset % kBlock == 0, context + ": unaligned punch");
        const auto shard_end = layout.shards[static_cast<std::size_t>(punch.shard)];
        check(punch.bytes > 0 &&
                  (punch.bytes % kBlock == 0 || punch.offset + punch.bytes == shard_end),
              context + ": partial block punched inside the shard");
        for (auto b = punch.offset / kBlock; b * kBlock < punch.offset + punch.bytes; ++b) {
            check(++punched[{punch.shard, b}] == 1, context + ": block punched twice");
            for (const auto& segment : layout.segments) {
                if (segment.shard != punch.shard || segment.end <= b * kBlock ||
                    segment.begin >= (b + 1) * kBlock) {
                    continue;
                }
                check(!segment.pinned, context + ": pinned bytes punched");
                for (const auto reader : segment.readers) {
                    check(journaled.contains(reader),
                          context + ": block punched before its reader was durable");
                }
            }
        }
    }
};

// Blocks that may be punched once `durable` objects are all durable.
std::set<std::pair<int, std::uint64_t>> punchable(const Layout& layout,
                                                  const std::set<std::uint32_t>& durable) {
    std::map<std::pair<int, std::uint64_t>, bool> blocks;
    for (const auto& segment : layout.segments) {
        if (segment.begin >= segment.end) { continue; }
        bool free = !segment.pinned;
        for (const auto reader : segment.readers) { free = free && durable.contains(reader); }
        for (auto b = segment.begin / kBlock; b <= (segment.end - 1) / kBlock; ++b) {
            auto [it, inserted] = blocks.try_emplace({segment.shard, b}, free);
            if (!inserted) { it->second = it->second && free; }
        }
    }
    std::set<std::pair<int, std::uint64_t>> out;
    for (const auto& [block, free] : blocks) {
        if (free) { out.insert(block); }
    }
    return out;
}

void run_case(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto layout  = random_layout(rng);
    const auto context = "seed " + std::to_string(seed);
    std::vector<std::uint64_t> output(layout.objects);
    for (auto& bytes : output) { bytes = 1 + rng() % 100000; }

    // A first run that stops after a random number of commits (a crash), then a resumed run.
    std::set<std::uint32_t> durable;
    for (int run = 0; run < 2; ++run) {
        ConsumptionLedger ledger(layout.shards, kBlock, layout.segments, layout.objects);
        auto planning = ledger;
        const std::vector<std::uint32_t> already(durable.begin(), durable.end());
        (void)planning.durable(already);
        const auto schedule = schedule_consumption(planning, output, 1000, 150000);
        auto sorted         = schedule.order;
        sorted.insert(sorted.end(), already.begin(), already.end());
        std::sort(sorted.begin(), sorted.end());
        check(sorted.size() == layout.objects &&
                  std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end() &&
                  (sorted.empty() || sorted.back() == layout.objects - 1),
              context + ": schedule is not a permutation of the remaining objects");

        CheckingIo io(layout);
        io.context   = context + (run ? " (resumed)" : "");
        io.journaled = durable;
        io.written   = durable;
        io.synced    = durable;
        Committer committer(ledger, io, schedule.order, already);

        // Complete objects in a random order, committing at random moments.
        auto pending = schedule.order;
        std::shuffle(pending.begin(), pending.end(), rng);
        const auto stop       = run == 0 ? rng() % (pending.size() + 1) : pending.size();
        std::size_t completed = 0;
        while (completed < stop) {
            const auto object = pending[completed++];
            io.written.insert(object);
            committer.complete(object);
            if (rng() % 3 == 0) { committer.commit(); }
        }
        if (run == 1) {
            while (committer.commit()) {}
            check(committer.done(), context + ": resumed run did not commit everything");
        } else {
            committer.commit();
        }
        durable = io.journaled;
        // Liveness: everything punchable for the durable set was punched (across both runs the
        // ledger re-issues punches of the first run, which a real punch absorbs).
        for (const auto& block : punchable(layout, durable)) {
            check(io.punched.contains(block), context + ": punchable block left in place");
        }
    }
}

// The scheduler puts an object that frees what it writes before one that only writes.
void schedule_prefers_freeing() {
    const std::vector<std::uint64_t> shards = {10 * kBlock};
    std::vector<SourceSegment> segments     = {
        {0, 0, kBlock, {}, true},
        {0, kBlock, 5 * kBlock, {0}, false},         // object 0 frees 4 blocks
        {0, 5 * kBlock, 10 * kBlock, {1, 2}, false}, // shared by 1 and 2
    };
    ConsumptionLedger ledger(shards, kBlock, segments, 3);
    const std::vector<std::uint64_t> output = {2 * kBlock, 3 * kBlock, 3 * kBlock};
    const auto schedule                     = schedule_consumption(ledger, output, 0, 0);
    check(schedule.order.size() == 3 && schedule.order[0] == 0, "freeing object first");
    // Window 0: one object in flight. Peak: object 0 writes 2 (0 freed yet) -> +2, then frees 4
    // (net -2); object 1 writes 3 -> +1; object 2 writes 3 -> +4, then frees 5 -> -1.
    check(schedule.peak_extra == 4 * kBlock,
          "peak bound " + std::to_string(schedule.peak_extra / kBlock) + " blocks, expected 4");
}

} // namespace

int main() {
    for (std::uint64_t seed = 1; seed <= 3000; ++seed) { run_case(seed); }
    schedule_prefers_freeing();
    if (failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("consume ledger: all checks passed\n");
    return 0;
}
