#pragma once

// The merge table of a BPE encoder: (left id, right id) -> (rank, result), open addressed with
// linear probing.
//
// Every adjacent symbol pair of every word is looked up here, and the lookup dominates encoding,
// so the table is stored flat: one power-of-two array of 16-byte entries instead of a bucket array
// plus one heap node per rule. A hit reads a single cache line rather than following a pointer
// chase, and the whole structure is smaller than the equivalent std::unordered_map even at a load
// factor below one half.
//
// An empty slot is recognisable by result < 0. Token ids are never negative, so a stored rule
// always has result >= 0.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninferno::text {

class BpeMergeTable {
public:
    struct Entry {
        std::uint64_t key = 0;
        int rank          = 0;
        int result        = -1; // < 0 marks an empty slot
    };

    // Sizes the table for `rules` rules: the smallest power of two (at least 16) above twice that
    // count, so at least half of the slots stay empty and every probe chain ends.
    void reset(std::size_t rules) {
        std::size_t capacity = 16;
        while (capacity < rules * 2U) { capacity <<= 1U; }
        slots_.assign(capacity, Entry{});
        mask_ = capacity - 1U;
    }

    // Adds a rule; false (and the table unchanged) when the pair is already present.
    bool insert(int left, int right, int rank, int result) {
        Entry& slot = probe(pack(left, right));
        if (slot.result >= 0) { return false; }
        slot = Entry{pack(left, right), rank, result};
        return true;
    }

    // Adds or replaces a rule: a repeated pair takes the later rank, as collecting into Rust's
    // HashMap does.
    void assign(int left, int right, int rank, int result) {
        probe(pack(left, right)) = Entry{pack(left, right), rank, result};
    }

    [[nodiscard]] const Entry* find(int left, int right) const noexcept {
        const std::uint64_t key = pack(left, right);
        std::size_t index       = mix(key) & mask_;
        while (true) {
            const Entry& entry = slots_[index];
            if (entry.result < 0) { return nullptr; }
            if (entry.key == key) { return &entry; }
            index = (index + 1U) & mask_;
        }
    }

private:
    static std::uint64_t pack(int left, int right) noexcept {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) |
               static_cast<std::uint32_t>(right);
    }

    // splitmix64 finalizer. A key is two token ids packed into one word, so its low bits alone
    // would cluster badly across a power-of-two slot count.
    static std::uint64_t mix(std::uint64_t key) noexcept {
        key += 0x9e3779b97f4a7c15ULL;
        key = (key ^ (key >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        key = (key ^ (key >> 27U)) * 0x94d049bb133111ebULL;
        return key ^ (key >> 31U);
    }

    // The slot holding `key`, or the empty slot that ends its probe chain.
    Entry& probe(std::uint64_t key) {
        std::size_t index = mix(key) & mask_;
        while (slots_[index].result >= 0 && slots_[index].key != key) {
            index = (index + 1U) & mask_;
        }
        return slots_[index];
    }

    // One empty slot by default, so find() on a default-constructed table terminates.
    std::vector<Entry> slots_ = std::vector<Entry>(1);
    std::size_t mask_         = 0;
};

} // namespace ninferno::text
