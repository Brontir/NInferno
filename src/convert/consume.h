#pragma once

// Source-consuming conversion (--consume-source): the checkpoint's disk space is released while
// the artifact is written, so a conversion fits on a disk that cannot hold both.
//
// The rule that makes this safe: a checkpoint byte is punched out only after every artifact
// object that reads it is durable -- written, fdatasync'd, and recorded in a fsync'd journal.
// A crash at any point leaves a journal whose completed objects are durable and whose punched
// bytes belong only to those objects, so a restart re-plans, skips them and redoes the rest.
//
// Everything here is bookkeeping without I/O; the driver supplies the I/O through CommitIo.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninferno::convert {

// A byte range of one checkpoint shard: one tensor, or a gap between tensors.
struct SourceSegment {
    int shard           = 0;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
    std::vector<std::uint32_t> readers; // artifact objects (weight jobs) that read it
    bool pinned = false; // read while planning (headers, divisor scalars): never punched
};

struct Punch {
    int shard            = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

// Which checkpoint blocks may be punched, given which objects are durable. A block is punched
// once every segment overlapping it is free: unpinned and either unread or read only by durable
// objects. Punches are whole blocks, so a partial block shared with a live segment is kept.
class ConsumptionLedger {
public:
    ConsumptionLedger(std::vector<std::uint64_t> shard_bytes, std::uint64_t block,
                      std::vector<SourceSegment> segments, std::size_t objects);

    // Blocks free before any object is written: unread, unpinned segments.
    [[nodiscard]] std::vector<Punch> initial();
    // Mark objects durable; returns the blocks this frees. Each block is returned once.
    [[nodiscard]] std::vector<Punch> durable(std::span<const std::uint32_t> objects);

    [[nodiscard]] const std::vector<SourceSegment>& segments() const noexcept { return segments_; }

    // Bytes a segment frees when it becomes free (its block-interior share), for scheduling.
    [[nodiscard]] std::uint64_t segment_bytes(std::size_t segment) const noexcept;

    [[nodiscard]] bool is_durable(std::uint32_t object) const noexcept { return durable_[object]; }

    [[nodiscard]] std::uint64_t block() const noexcept { return block_; }

private:
    void release(std::size_t segment, std::vector<Punch>& out);

    std::vector<std::uint64_t> shard_bytes_;
    std::uint64_t block_ = 4096;
    std::vector<SourceSegment> segments_;
    std::vector<std::uint32_t> remaining_;          // live readers per segment
    std::vector<bool> freed_;                       // segment released
    std::vector<std::vector<std::uint32_t>> reads_; // segments per object
    std::vector<bool> durable_;
    std::vector<std::vector<std::uint16_t>> live_; // per shard, per block: live segments
};

struct Schedule {
    std::vector<std::uint32_t> order; // objects in production order
    // Upper bound of disk bytes above the starting checkpoint at any moment, when at most
    // `window` output bytes of uncommitted objects are in flight (one object may exceed it).
    std::uint64_t peak_extra = 0;
    std::uint64_t window     = 0;
};

// Order objects so sources are finished early: objects whose completion frees at least what
// they write go first (smallest first), then the rest by most bytes freed. `prologue` is written
// before any object (header, resources, auxiliaries).
[[nodiscard]] Schedule schedule_consumption(const ConsumptionLedger& ledger,
                                            const std::vector<std::uint64_t>& output_bytes,
                                            std::uint64_t prologue, std::uint64_t window);

// The peak bound of a given order (as schedule_consumption computes it).
[[nodiscard]] std::uint64_t consumption_peak(const ConsumptionLedger& ledger,
                                             const std::vector<std::uint64_t>& output_bytes,
                                             const std::vector<std::uint32_t>& order,
                                             std::uint64_t prologue, std::uint64_t window);

// Append-only journal of durable objects: a header naming the plan it belongs to, then one
// 8-byte record per object. A torn trailing record is ignored.
class Journal {
public:
    // Opens `path` if it exists (throws when its plan fingerprint differs), else creates it.
    Journal(std::filesystem::path path, std::uint64_t fingerprint,
            const std::array<std::byte, 16>& artifact_id, std::uint64_t objects);
    ~Journal();
    Journal(const Journal&)            = delete;
    Journal& operator=(const Journal&) = delete;

    [[nodiscard]] bool resumed() const noexcept { return resumed_; }

    [[nodiscard]] const std::array<std::byte, 16>& artifact_id() const noexcept { return id_; }

    [[nodiscard]] const std::vector<std::uint32_t>& completed() const noexcept {
        return completed_;
    }

    // Append and fsync.
    void append(std::span<const std::uint32_t> objects);
    void remove() noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    // Read an existing journal's artifact id without opening it for writing.
    [[nodiscard]] static std::optional<std::array<std::byte, 16>>
    existing_id(const std::filesystem::path& path);

private:
    std::filesystem::path path_;
    int fd_       = -1;
    bool resumed_ = false;
    std::array<std::byte, 16> id_{};
    std::vector<std::uint32_t> completed_;
};

// The durability pipeline, in its only allowed order: output data synced, journal record
// synced, then source blocks punched.
struct CommitIo {
    virtual ~CommitIo()                                      = default;
    virtual void sync_output()                               = 0;
    virtual void journal(std::span<const std::uint32_t> ids) = 0; // append + fsync
    virtual void punch(const Punch& punch)                   = 0;
};

// Commits objects in schedule order as they complete. complete() may be called from any thread
// in any order; commit() (one thread) makes the longest completed prefix durable, journals it
// and punches what it frees. Objects already in the journal count as committed from the start.
class Committer {
public:
    Committer(ConsumptionLedger& ledger, CommitIo& io, std::vector<std::uint32_t> order,
              std::span<const std::uint32_t> already_durable);

    void complete(std::uint32_t object);
    // Returns the number of objects committed by this call.
    std::size_t commit();
    [[nodiscard]] bool done() const;
    [[nodiscard]] std::size_t committed() const;

    [[nodiscard]] std::uint64_t punched_bytes() const noexcept { return punched_; }

private:
    ConsumptionLedger& ledger_;
    CommitIo& io_;
    std::vector<std::uint32_t> order_;
    std::vector<bool> complete_;
    std::size_t next_      = 0; // first uncommitted position in order_
    std::uint64_t punched_ = 0;
    mutable std::mutex mutex_;
};

} // namespace ninferno::convert
