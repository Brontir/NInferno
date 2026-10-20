#pragma once

// tools/artifact/writer.py in C++: the fixed directory reserve, file segmentation, temporary
// files published by link once every byte is written, and bounded dirty output.
//
// Payload bytes go out one of two ways. By default (WriterOptions::direct) with O_DIRECT from
// aligned staging segments, each written by the worker whose store completes it: page-cache
// writeback is a single flusher per device, direct writes from every worker run in parallel,
// and the artifact never enters the page cache. A resumable (source-consuming) conversion
// stores through a shared writable mapping instead, whose pages fdatasync can make durable at
// any point: buffered pwrite() from many threads serializes on the inode lock, faults into a
// mapping do not. Mapped pages are unmapped at once and writeback is started every few hundred
// MiB, so neither the process nor the page cache holds the artifact.

#include "artifact/framing.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ninferno::convert {

struct OutputFile {
    std::optional<std::string> path; // null for the entry file, else the sibling part name
    std::uint64_t payload_bytes = 0;
};

// layout_directory: the directory text (padded to its reserve) and the file segmentation.
struct ArtifactLayout {
    std::vector<OutputFile> files;
    std::string directory;             // padded with spaces to the reserve
    std::uint64_t directory_bytes = 0; // padded length, as the header records it
    std::uint64_t payload_offset  = 0; // entry file offset of the payload
    std::uint64_t payload_bytes   = 0;
    // Bytes of every output file together (headers, directory and payload).
    [[nodiscard]] std::uint64_t file_bytes() const noexcept;
};

// `description` is the compact directory JSON without "files" (an object); "files" is added as
// its last member exactly as layout_directory does.
[[nodiscard]] ArtifactLayout layout_artifact(const std::string& entry_name,
                                             const std::string& description,
                                             std::uint64_t payload_bytes,
                                             std::uint64_t max_file_bytes);

// A payload byte range the conversion writes. Everything else in the payload stays zero.
struct PayloadRange {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct WriterOptions {
    // Temporary files named "<file>.partial" instead of a random name, so a later run can find
    // them (source-consuming conversions resume).
    bool named_partial = false;
    // Reopen existing partial files written with this artifact id instead of creating them.
    std::optional<artifact::ArtifactId> resume;
    // Write with O_DIRECT: stores are staged in aligned segments, and the store that completes a
    // segment (every byte of `written` inside it has arrived) writes it. Page-cache writeback is
    // one flusher thread per device; direct writes from every worker run in parallel. Requires
    // `written` to list every range the conversion will store; not for resumed files.
    bool direct = false;
    std::vector<PayloadRange> written;
};

class ArtifactWriter {
public:
    ArtifactWriter(std::filesystem::path path, ArtifactLayout layout, WriterOptions options = {});
    ~ArtifactWriter();
    ArtifactWriter(const ArtifactWriter&)            = delete;
    ArtifactWriter& operator=(const ArtifactWriter&) = delete;

    // Thread-safe for disjoint ranges.
    void write(std::uint64_t payload_offset, const void* data, std::size_t bytes);
    // Make everything written so far durable (fdatasync of every output file).
    void sync();
    void finish();
    void abort() noexcept;

    // Leave the partial files in place on destruction (a resumable conversion that failed).
    void keep_partial() noexcept { keep_partial_ = true; }

    [[nodiscard]] const artifact::ArtifactId& artifact_id() const noexcept { return id_; }

    [[nodiscard]] const std::vector<OutputFile>& files() const noexcept { return layout_.files; }

    [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return layout_.payload_bytes; }

    [[nodiscard]] std::filesystem::path file_path(std::size_t index) const;
    [[nodiscard]] static std::filesystem::path partial_path(const std::filesystem::path& target);

private:
    struct Mapping {
        std::byte* base     = nullptr;
        std::uint64_t bytes = 0;
    };

    // O_DIRECT staging of one output file.
    struct Segment {
        std::byte* data      = nullptr; // aligned, zero-filled
        std::uint64_t filled = 0;
        bool flushed         = false;
    };

    struct DirectFile {
        int fd = -1;                         // opened with O_DIRECT
        std::vector<std::uint64_t> expected; // bytes written per segment
        std::vector<Segment> segments;
        std::uint64_t length = 0;
    };

    void create(std::size_t index);
    void reopen(std::size_t index);
    void prepare_direct();
    void stage(std::size_t index, std::uint64_t offset, const std::byte* data, std::size_t bytes);
    void flush_segment(DirectFile& file, std::size_t segment, std::byte* data);
    void enqueue(std::size_t file, std::size_t segment, std::byte* data);
    [[nodiscard]] std::byte* acquire_segment();
    void release_segment(std::byte* data) noexcept;
    void drain();
    void stop_writers() noexcept;
    void finish_direct();
    void write_file(std::size_t index, std::uint64_t offset, const void* data, std::size_t bytes);
    void store(std::size_t index, std::uint64_t offset, const std::byte* data, std::size_t bytes);
    void written(std::size_t bytes);
    void unmap() noexcept;
    [[nodiscard]] std::uint64_t file_length(std::size_t index) const noexcept;

    std::filesystem::path path_;
    ArtifactLayout layout_;
    WriterOptions options_;
    artifact::ArtifactId id_{};
    std::vector<std::uint64_t> prefixes_;
    std::vector<int> fds_;
    std::vector<Mapping> mappings_;
    std::vector<std::filesystem::path> temporary_;
    std::vector<std::filesystem::path> published_;
    std::atomic<std::uint64_t> dirty_{0};
    std::mutex writeback_;
    std::vector<DirectFile> direct_;
    std::array<std::mutex, 256> segment_locks_;

    struct Pending {
        std::size_t file    = 0;
        std::size_t segment = 0;
        std::byte* data     = nullptr;
    };

    std::mutex queue_mutex_;
    std::condition_variable queue_changed_;
    std::deque<Pending> queue_;
    std::uint64_t queued_bytes_ = 0;
    bool closing_               = false;
    std::exception_ptr writer_error_;
    std::vector<std::thread> writers_;
    std::mutex spare_mutex_;
    std::vector<std::byte*> spare_; // staging buffers ready for reuse
    bool finished_     = false;
    bool keep_partial_ = false;
};

} // namespace ninferno::convert
