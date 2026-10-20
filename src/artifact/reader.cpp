#include "artifact/reader.h"

#include "artifact/file_io.h"
#include "artifact/formats.h"
#include "artifact/layouts.h"

#include <algorithm>
#include <array>
#include <limits>

namespace ninferno::artifact {

// A directory longer than this is a damaged or forged header, refused before it is read into
// memory (the converter's largest directories are a few MiB).
constexpr std::uint64_t kMaximumDirectoryBytes = 256ULL << 20;

struct Reader::Impl {
    std::filesystem::path entry;
    Directory directory;
    ArtifactId id{};
    std::uint64_t entry_payload_start = 0;
    std::uint64_t file_bytes          = 0;
    mutable std::vector<std::unique_ptr<InputFile>> files;
    mutable std::vector<std::optional<WeightGeometry>> geometries;

    Impl(const std::filesystem::path& path, PendingRelayout pending) : entry(path) {
        std::error_code error;
        if (pending == PendingRelayout::Refuse &&
            std::filesystem::exists(relayout_journal_path(entry), error)) {
            throw ArtifactError(entry.string() +
                                ": an in-place ninferno-convert --relayout of this "
                                "artifact did not finish (" +
                                relayout_journal_path(entry).string() +
                                "); rerun the same --relayout command to finish it (it says "
                                "when the journal is not this artifact's and can be deleted)");
        }
        auto file = std::make_unique<InputFile>(entry);
        std::array<std::byte, kHeaderBytes> header{};
        file->read_exact(0, header);
        if (!has_magic(kEntryMagic, header.data())) {
            throw ArtifactError(entry.string() + ": not a NInferno artifact (no NInferno entry magic)");
        }
        const auto json_bytes = read_u64_le(header.data() + 8);
        if (!json_bytes || json_bytes > file->bytes() - kHeaderBytes ||
            json_bytes > kMaximumDirectoryBytes) {
            throw ArtifactError(entry.string() + ": invalid directory length " +
                                std::to_string(json_bytes) + " (file " +
                                std::to_string(file->bytes()) + " bytes, at most " +
                                std::to_string(kMaximumDirectoryBytes) + ")");
        }
        std::copy_n(header.begin() + 16, id.size(), id.begin());
        std::string text(static_cast<std::size_t>(json_bytes), '\0');
        file->read_exact(kHeaderBytes, std::as_writable_bytes(std::span(text.data(), text.size())));
        directory =
            parse_directory(parse_json(text, "artifact directory"), entry.filename().string());
        entry_payload_start = align_up(checked_add(kHeaderBytes, json_bytes, "metadata end"),
                                       kPayloadAlignment, "entry payload start");
        const auto expected =
            checked_add(entry_payload_start, directory.files[0].payload_bytes, "entry length");
        if (expected != file->bytes()) {
            throw ArtifactError(entry.string() + ": entry length differs from directory");
        }
        file_bytes = checked_add(entry_payload_start, directory.payload_bytes, "file bytes");
        file_bytes = checked_add(
            file_bytes,
            checked_mul(directory.files.size() - 1, kPayloadAlignment, "continuation headers"),
            "file bytes");
        files.resize(directory.files.size());
        files[0] = std::move(file);
        geometries.resize(directory.objects.size());
    }

    InputFile& file(std::size_t index) const {
        if (index >= files.size()) { throw ArtifactError("invalid continuation index"); }
        if (!files[index]) {
            const auto& record = directory.files[index];
            auto input         = std::make_unique<InputFile>(entry.parent_path() / *record.path);
            std::array<std::byte, kHeaderBytes> header{};
            input->read_exact(0, header);
            if (!has_magic(kPartMagic, header.data()) ||
                read_u64_le(header.data() + 8) != index ||
                !std::equal(id.begin(), id.end(), header.begin() + 16)) {
                throw ArtifactError(*record.path +
                                    ": continuation header does not belong to this entry");
            }
            if (input->bytes() !=
                checked_add(kPayloadAlignment, record.payload_bytes, "part length")) {
                throw ArtifactError(*record.path + ": continuation length differs from directory");
            }
            files[index] = std::move(input);
        }
        return *files[index];
    }
};

std::filesystem::path relayout_journal_path(const std::filesystem::path& entry) {
    return std::filesystem::path(entry.string() + ".relayout-journal");
}

Reader::Reader(const std::filesystem::path& path, PendingRelayout pending)
    : impl_(std::make_unique<Impl>(path, pending)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const Directory& Reader::directory() const noexcept { return impl_->directory; }

const ArtifactId& Reader::artifact_id() const noexcept { return impl_->id; }

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file_bytes; }

ObjectHandle Reader::find(std::string_view id) const {
    const auto found = directory().object_index.find(id);
    if (found == directory().object_index.end()) {
        throw ArtifactError("missing object " + std::string(id));
    }
    return found->second;
}

const WeightGeometry& Reader::geometry(ObjectHandle handle) const {
    const auto& tensor = directory().tensor(handle);
    auto& result       = impl_->geometries[handle.index];
    if (!result) { result = describe_tensor(tensor); }
    return *result;
}

void Reader::validate_object(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    if (std::holds_alternative<TensorObject>(object)) {
        (void)geometry(handle);
    } else if (std::get<ResourceObject>(object).encoding != kRawBytesEncoding) {
        throw ArtifactError(object_id(object) + ": unsupported resource encoding");
    }
}

std::vector<ReadSegment> Reader::segments(std::uint64_t offset, std::uint64_t bytes) const {
    const auto& data = directory();
    if (offset > data.payload_bytes || bytes > data.payload_bytes - offset) {
        throw ArtifactError("read range exceeds logical payload");
    }
    std::vector<ReadSegment> result;
    if (!bytes) { return result; }
    const auto found     = std::upper_bound(data.files.begin(), data.files.end(), offset,
                                            [](std::uint64_t position, const FileRecord& file) {
                                            return position < file.logical_begin;
                                        });
    std::size_t index    = static_cast<std::size_t>(found - data.files.begin() - 1);
    std::uint64_t copied = 0;
    while (copied < bytes) {
        const auto& file = data.files[index];
        const auto local = offset + copied - file.logical_begin;
        const auto count = std::min(bytes - copied, file.payload_bytes - local);
        const auto start = index == 0 ? impl_->entry_payload_start : kPayloadAlignment;
        result.push_back({index, checked_add(start, local, "file offset"), copied, count});
        copied += count;
        ++index;
    }
    return result;
}

void Reader::read_into(std::uint64_t offset, std::span<std::byte> destination) const {
    for (const auto& segment : segments(offset, destination.size())) {
        impl_->file(segment.file_index)
            .read_exact(segment.file_offset,
                        destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                            static_cast<std::size_t>(segment.bytes)));
    }
}

std::vector<std::byte> Reader::read_range(std::uint64_t offset, std::uint64_t bytes) const {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host read size exceeds size_t");
    }
    (void)segments(offset, bytes);
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    read_into(offset, result);
    return result;
}

std::vector<std::byte> Reader::read_object(ObjectHandle handle) const {
    validate_object(handle);
    const auto& object = directory().object(handle);
    return read_range(object_offset(object), object_bytes(object));
}

std::size_t Reader::read_direct(std::size_t file_index, std::uint64_t file_offset,
                                std::span<std::byte> destination) const {
    return impl_->file(file_index).read_direct(file_offset, destination);
}

} // namespace ninferno::artifact
