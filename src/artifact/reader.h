#pragma once

#include "artifact/framing.h"
#include "artifact/schema.h"
#include "core/weight_view.h"

#include <filesystem>
#include <memory>
#include <span>

namespace ninferno::artifact {

struct ReadSegment {
    std::size_t file_index           = 0;
    std::uint64_t file_offset        = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes              = 0;
};

// <entry>.relayout-journal: an in-place `ninferno-convert --relayout` of the artifact that has not
// finished. Until it does, part of the payload is in one layout while the directory names the
// other, so the artifact must not be read.
[[nodiscard]] std::filesystem::path relayout_journal_path(const std::filesystem::path& entry);

// Whether a Reader opens an artifact with a pending relayout: only the relayout itself does.
enum class PendingRelayout { Refuse, Allow };

// Owns the cold directory and lazily opened files. Runtime views borrow materialized storage,
// never this Reader. Encodings are interpreted only for requested objects.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path,
                    PendingRelayout pending = PendingRelayout::Refuse);
    ~Reader();
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    [[nodiscard]] const Directory& directory() const noexcept;
    [[nodiscard]] const ArtifactId& artifact_id() const noexcept;
    [[nodiscard]] std::uint64_t file_bytes() const noexcept;
    [[nodiscard]] ObjectHandle find(std::string_view id) const;
    [[nodiscard]] const WeightGeometry& geometry(ObjectHandle handle) const;
    void validate_object(ObjectHandle handle) const;

    [[nodiscard]] std::vector<ReadSegment> segments(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    void read_into(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::vector<std::byte> read_range(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    [[nodiscard]] std::vector<std::byte> read_object(ObjectHandle handle) const;
    [[nodiscard]] std::size_t read_direct(std::size_t file_index, std::uint64_t file_offset,
                                          std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninferno::artifact
