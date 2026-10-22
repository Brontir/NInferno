#include "convert/writer.h"

#include "artifact/reader.h"
#include "convert/pyjson.h"
#include "core/checked.h"
#include "core/file_io.h"

#include "core/sys.h"

#include <algorithm>
#include <condition_variable>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <openssl/evp.h>

namespace ninferno::convert {
namespace {

constexpr std::uint64_t kAlignment      = artifact::kPayloadAlignment;
constexpr std::uint64_t kHeaderBytes    = artifact::kHeaderBytes;
constexpr std::uint64_t kWritebackBytes = 256ULL << 20;
// O_DIRECT output: staging segments, writer threads, and the bytes of complete segments that may
// wait for a writer before producers pause.
constexpr std::uint64_t kSegmentBytes  = 8ULL << 20;
constexpr std::uint64_t kDirectWriters = 2;
constexpr std::uint64_t kQueueBytes    = 256ULL << 20;

constexpr std::uint64_t kDirectBlock = 4096;

std::string files_json(const std::vector<OutputFile>& files) {
    OJson list = OJson::array();
    for (const auto& file : files) {
        OJson item            = OJson::object();
        item["path"]          = file.path ? OJson(*file.path) : OJson(nullptr);
        item["payload_bytes"] = file.payload_bytes;
        list.push_back(std::move(item));
    }
    return py_dumps_compact(list);
}

void put_u64(std::byte* out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) { out[i] = std::byte((value >> (8 * i)) & 0xFF); }
}

} // namespace

std::uint64_t ArtifactLayout::file_bytes() const noexcept {
    std::uint64_t out = payload_offset + (files.empty() ? 0 : files.front().payload_bytes);
    for (std::size_t i = 1; i < files.size(); ++i) { out += kAlignment + files[i].payload_bytes; }
    return out;
}

ArtifactLayout layout_artifact(const std::string& entry_name, const std::string& description,
                               std::uint64_t payload_bytes, std::uint64_t max_file_bytes) {
    if (!payload_bytes) { throw std::runtime_error("payload bytes: expected a positive integer"); }
    if (!max_file_bytes) { throw std::runtime_error("--max-file-bytes must be positive"); }
    if (description.size() < 2 || description.back() != '}') {
        throw std::logic_error("directory description must be a JSON object");
    }
    const auto head = description.substr(0, description.size() - 1) +
                      (description.size() > 2 ? "," : "") + "\"files\":";
    const auto encode = [&](const std::vector<OutputFile>& files) {
        return head + files_json(files) + "}";
    };
    // layout_directory: grow the reserve until the directory, files list included, fits.
    std::vector<OutputFile> files = {{std::nullopt, payload_bytes}};
    std::uint64_t reserve =
        align_up(kHeaderBytes + encode(files).size(), kAlignment, "artifact header") - kHeaderBytes;
    std::string encoded;
    while (true) {
        const auto entry_start = kHeaderBytes + reserve;
        if (entry_start >= max_file_bytes) {
            throw std::runtime_error(
                "maximum file bytes cannot contain the entry metadata and payload");
        }
        files.clear();
        if (entry_start + payload_bytes <= max_file_bytes) {
            files.push_back({std::nullopt, payload_bytes});
        } else {
            const auto entry_capacity = (max_file_bytes - entry_start) / kAlignment * kAlignment;
            const auto part_capacity  = (max_file_bytes - kAlignment) / kAlignment * kAlignment;
            if (std::min(entry_capacity, part_capacity) == 0 || max_file_bytes <= kAlignment) {
                throw std::runtime_error(
                    "maximum file bytes cannot contain a positive aligned segment");
            }
            files.push_back({std::nullopt, entry_capacity});
            auto remaining = payload_bytes - entry_capacity;
            while (remaining) {
                const auto count = std::min(remaining, part_capacity);
                char suffix[32];
                std::snprintf(suffix, sizeof(suffix), ".part-%04zu", files.size());
                files.push_back({entry_name + suffix, count});
                remaining -= count;
            }
        }
        encoded = encode(files);
        if (encoded.size() <= reserve) { break; }
        reserve =
            align_up(kHeaderBytes + encoded.size(), kAlignment, "artifact header") - kHeaderBytes;
    }
    ArtifactLayout layout;
    // writer.py records the padded length: the header counts the reserve, spaces included.
    encoded.append(reserve - encoded.size(), ' ');
    layout.directory_bytes = encoded.size();
    layout.directory       = std::move(encoded);
    layout.files           = std::move(files);
    layout.payload_offset  = kHeaderBytes + reserve;
    layout.payload_bytes   = payload_bytes;
    return layout;
}

std::filesystem::path ArtifactWriter::partial_path(const std::filesystem::path& target) {
    return target.string() + ".partial";
}

ArtifactWriter::ArtifactWriter(std::filesystem::path path, ArtifactLayout layout,
                               WriterOptions options)
    : path_(std::move(path)), layout_(std::move(layout)), options_(std::move(options)) {
    prefixes_ = {0};
    for (const auto& file : layout_.files) {
        prefixes_.push_back(prefixes_.back() + file.payload_bytes);
    }
    if (options_.resume) {
        id_ = *options_.resume;
    } else {
        if (!sys::random_bytes(id_.data(), id_.size())) {
            throw std::runtime_error("cannot draw a random artifact id");
        }
        id_[6] = (id_[6] & std::byte{0x0F}) | std::byte{0x40}; // uuid4 version
        id_[8] = (id_[8] & std::byte{0x3F}) | std::byte{0x80}; // RFC 4122 variant
    }
    for (std::size_t i = 0; i < layout_.files.size(); ++i) {
        if (std::filesystem::exists(file_path(i))) {
            throw std::runtime_error("conversion output already exists: " + file_path(i).string());
        }
    }
    create_parent_directories(path_);
    if (options_.direct && options_.resume) {
        throw std::logic_error("direct output cannot resume a partial file");
    }
    try {
        if (options_.direct) { prepare_direct(); }
        for (std::size_t i = 0; i < layout_.files.size(); ++i) {
            if (options_.resume) {
                reopen(i);
            } else {
                create(i);
            }
        }
        for (std::size_t i = 0; i < layout_.files.size() && !options_.direct; ++i) {
            const auto bytes = file_length(i);
            void* base       = sys::map(fds_[i], bytes, sys::Map::ReadWriteShared);
            if (base == nullptr) { throw std::runtime_error("cannot map the output file"); }
            mappings_.push_back({static_cast<std::byte*>(base), bytes});
        }
    } catch (...) {
        if (options_.resume) { keep_partial_ = true; }
        abort();
        throw;
    }
}

ArtifactWriter::~ArtifactWriter() {
    if (!finished_) { abort(); }
    for (auto* data : spare_) { sys::aligned_free(data); }
}

std::filesystem::path ArtifactWriter::file_path(std::size_t index) const {
    return index == 0 ? path_ : path_.parent_path() / *layout_.files[index].path;
}

std::uint64_t ArtifactWriter::file_length(std::size_t index) const noexcept {
    return (index == 0 ? layout_.payload_offset : kAlignment) + layout_.files[index].payload_bytes;
}

void ArtifactWriter::create(std::size_t index) {
    const auto target = file_path(index);
    int fd            = -1;
    std::string name;
    if (options_.named_partial) {
        name = partial_path(target).string();
        fd   = sys::open(name, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    } else {
        name = (target.parent_path() / ("." + target.filename().string() + ".XXXXXX.tmp")).string();
        fd   = sys::create_unique(name, 4);
    }
    if (fd < 0) { throw std::runtime_error("cannot create " + name + ": " + sys::error_text(errno)); }
    fds_.push_back(fd);
    temporary_.emplace_back(name);
    if (sys::truncate(fd, static_cast<std::int64_t>(file_length(index))) != 0) {
        throw std::runtime_error("cannot size " + name);
    }
    std::byte header[kHeaderBytes];
    const auto& magic = index == 0 ? artifact::kEntryMagic : artifact::kPartMagic;
    std::memcpy(header, magic.data(), 8);
    put_u64(header + 8, index == 0 ? layout_.directory_bytes : index);
    std::memcpy(header + 16, id_.data(), 16);
    if (options_.direct) {
        direct_[index].fd = sys::open(name, O_WRONLY | O_DIRECT | O_CLOEXEC);
        if (direct_[index].fd < 0) {
            throw std::runtime_error("cannot open " + name +
                                     " for direct writes: " + sys::error_text(errno));
        }
        stage(index, 0, header, sizeof(header));
        if (index == 0) {
            stage(0, kHeaderBytes, reinterpret_cast<const std::byte*>(layout_.directory.data()),
                  layout_.directory.size());
        }
        return;
    }
    write_file(index, 0, header, sizeof(header));
    if (index == 0) {
        write_file(0, kHeaderBytes, layout_.directory.data(), layout_.directory.size());
    }
}

void ArtifactWriter::prepare_direct() {
    for (std::uint64_t i = 0; i < kDirectWriters; ++i) {
        writers_.emplace_back([this] { drain(); });
    }
    direct_.resize(layout_.files.size());
    for (std::size_t i = 0; i < layout_.files.size(); ++i) {
        auto& file          = direct_[i];
        file.length         = file_length(i);
        const auto segments = (file.length + kSegmentBytes - 1) / kSegmentBytes;
        file.expected.assign(segments, 0);
        file.segments.resize(segments);
    }
    const auto add = [&](std::size_t i, std::uint64_t begin, std::uint64_t end) {
        auto& expected = direct_[i].expected;
        while (begin < end) {
            const auto segment = begin / kSegmentBytes;
            const auto stop    = std::min(end, (segment + 1) * kSegmentBytes);
            expected[segment] += stop - begin;
            begin = stop;
        }
    };
    add(0, 0, layout_.payload_offset); // header and the padded directory
    for (std::size_t i = 1; i < layout_.files.size(); ++i) { add(i, 0, kHeaderBytes); }
    for (const auto& range : options_.written) {
        for (std::size_t i = 0; i < layout_.files.size(); ++i) {
            const auto begin = std::max(range.begin, prefixes_[i]);
            const auto end   = std::min(range.end, prefixes_[i + 1]);
            if (begin >= end) { continue; }
            const auto start = i == 0 ? layout_.payload_offset : kAlignment;
            add(i, start + begin - prefixes_[i], start + end - prefixes_[i]);
        }
    }
}

void ArtifactWriter::stage(std::size_t index, std::uint64_t offset, const std::byte* data,
                           std::size_t bytes) {
    auto& file = direct_[index];
    while (bytes) {
        const auto segment  = static_cast<std::size_t>(offset / kSegmentBytes);
        const auto within   = offset - segment * kSegmentBytes;
        const auto count    = std::min<std::uint64_t>(bytes, kSegmentBytes - within);
        std::byte* complete = nullptr;
        {
            std::lock_guard lock(segment_locks_[(segment * 31 + index) % segment_locks_.size()]);
            auto& s = file.segments[segment];
            if (s.flushed) {
                throw std::logic_error("output store into a segment already written");
            }
            if (!s.data) {
                s.data = acquire_segment();
                std::memset(s.data, 0, kSegmentBytes);
            }
            std::memcpy(s.data + within, data, count);
            s.filled += count;
            if (s.filled > file.expected[segment]) {
                throw std::logic_error("output store outside the planned ranges");
            }
            if (s.filled == file.expected[segment]) {
                complete  = s.data;
                s.data    = nullptr;
                s.flushed = true;
            }
        }
        if (complete) { enqueue(index, segment, complete); }
        offset += count;
        data += count;
        bytes -= count;
    }
}

// Staging buffers come from a pool the writer owns: a buffer is filled by a producer thread and
// written and returned by a writer thread, and freeing it there would leave it resident in the
// producer's malloc arena. The pool holds at most the peak number of buffers in use at once.
std::byte* ArtifactWriter::acquire_segment() {
    {
        std::lock_guard lock(spare_mutex_);
        if (!spare_.empty()) {
            auto* data = spare_.back();
            spare_.pop_back();
            return data;
        }
    }
    auto* data = static_cast<std::byte*>(sys::aligned_alloc(kDirectBlock, kSegmentBytes));
    if (!data) { throw std::bad_alloc(); }
    return data;
}

void ArtifactWriter::release_segment(std::byte* data) noexcept {
    if (!data) { return; }
    std::lock_guard lock(spare_mutex_);
    try {
        spare_.push_back(data);
    } catch (...) { sys::aligned_free(data); }
}

void ArtifactWriter::flush_segment(DirectFile& file, std::size_t segment, std::byte* data) {
    const auto offset  = segment * kSegmentBytes;
    auto length        = std::min(kSegmentBytes, file.length - offset);
    length             = align_up(length, kDirectBlock, "artifact segment");
    std::uint64_t done = 0;
    while (done < length) {
        const auto count =
            sys::pwrite(file.fd, data + done, length - done, static_cast<std::int64_t>(offset + done));
        if (count < 0 && errno == EINTR) { continue; }
        if (count <= 0) {
            throw std::runtime_error("direct write at file offset " + std::to_string(offset) +
                                     " failed: " + sys::error_text(errno));
        }
        done += static_cast<std::uint64_t>(count);
    }
}

// Complete segments queue for the writer threads; producers wait while the queue holds more
// than its byte bound, so memory stays bounded when the device is the bottleneck.
void ArtifactWriter::enqueue(std::size_t file, std::size_t segment, std::byte* data) {
    std::unique_lock lock(queue_mutex_);
    queue_changed_.wait(
        lock, [&] { return queued_bytes_ < kQueueBytes || writer_error_ || writers_.empty(); });
    if (writer_error_) {
        release_segment(data);
        std::rethrow_exception(writer_error_);
    }
    queue_.push_back({file, segment, data});
    queued_bytes_ += kSegmentBytes;
    queue_changed_.notify_all();
}

void ArtifactWriter::drain() {
    while (true) {
        Pending item;
        {
            std::unique_lock lock(queue_mutex_);
            queue_changed_.wait(lock, [&] { return !queue_.empty() || closing_; });
            if (queue_.empty()) { return; }
            item = queue_.front();
            queue_.pop_front();
        }
        try {
            flush_segment(direct_[item.file], item.segment, item.data);
        } catch (...) {
            std::lock_guard lock(queue_mutex_);
            if (!writer_error_) { writer_error_ = std::current_exception(); }
        }
        release_segment(item.data);
        std::lock_guard lock(queue_mutex_);
        queued_bytes_ -= kSegmentBytes;
        queue_changed_.notify_all();
    }
}

void ArtifactWriter::stop_writers() noexcept {
    {
        std::lock_guard lock(queue_mutex_);
        closing_ = true;
    }
    queue_changed_.notify_all();
    for (auto& writer : writers_) {
        if (writer.joinable()) { writer.join(); }
    }
    writers_.clear();
    for (auto& item : queue_) { release_segment(item.data); }
    queue_.clear();
}

// Every planned byte must have arrived (the writer's coverage check); the file then gets its
// exact length back (a direct write rounds the tail up to a block) and is made durable.
void ArtifactWriter::finish_direct() {
    stop_writers();
    if (writer_error_) { std::rethrow_exception(writer_error_); }
    for (std::size_t i = 0; i < direct_.size(); ++i) {
        auto& file = direct_[i];
        for (std::size_t s = 0; s < file.segments.size(); ++s) {
            if (file.expected[s] && !file.segments[s].flushed) {
                throw std::runtime_error("output coverage is incomplete in " +
                                         temporary_[i].string() + " at byte " +
                                         std::to_string(s * kSegmentBytes));
            }
        }
        if (sys::truncate(fds_[i], static_cast<std::int64_t>(file.length)) != 0) {
            throw std::runtime_error("cannot size " + temporary_[i].string());
        }
    }
}

void ArtifactWriter::reopen(std::size_t index) {
    const auto name = partial_path(file_path(index));
    const int fd    = sys::open(name, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("cannot resume: partial output " + name.string() + " is missing");
    }
    fds_.push_back(fd);
    temporary_.push_back(name);
    if (static_cast<std::uint64_t>(sys::file_size(fd)) != file_length(index)) {
        throw std::runtime_error("cannot resume: " + name.string() + " has the wrong length");
    }
    // The partial file must belong to this plan: same header, same directory, same id.
    const auto prefix = index == 0 ? kHeaderBytes + layout_.directory.size() : kHeaderBytes;
    std::vector<std::byte> actual(prefix);
    if (sys::pread(fd, actual.data(), prefix, 0) != static_cast<std::int64_t>(prefix)) {
        throw std::runtime_error("cannot read " + name.string());
    }
    std::vector<std::byte> expected(prefix);
    const auto& magic = index == 0 ? artifact::kEntryMagic : artifact::kPartMagic;
    std::memcpy(expected.data(), magic.data(), 8);
    put_u64(expected.data() + 8, index == 0 ? layout_.directory_bytes : index);
    std::memcpy(expected.data() + 16, id_.data(), 16);
    if (index == 0) {
        std::memcpy(expected.data() + kHeaderBytes, layout_.directory.data(),
                    layout_.directory.size());
    }
    if (actual != expected) {
        throw std::runtime_error("cannot resume: " + name.string() +
                                 " was written for a different plan or artifact id");
    }
}

void ArtifactWriter::write_file(std::size_t index, std::uint64_t offset, const void* data,
                                std::size_t bytes) {
    const auto* cursor = static_cast<const char*>(data);
    while (bytes) {
        const auto count = sys::pwrite(fds_[index], cursor, bytes, static_cast<std::int64_t>(offset));
        if (count < 0 && errno == EINTR) { continue; }
        if (count <= 0) {
            throw std::runtime_error("short write at file offset " + std::to_string(offset) + ": " +
                                     sys::error_text(errno));
        }
        cursor += count;
        offset += static_cast<std::uint64_t>(count);
        bytes -= static_cast<std::size_t>(count);
        written(static_cast<std::size_t>(count));
    }
}

void ArtifactWriter::written(std::size_t bytes) {
    if (dirty_.fetch_add(bytes) + bytes < kWritebackBytes) { return; }
    std::unique_lock lock(writeback_, std::try_to_lock);
    if (!lock.owns_lock()) { return; }
    dirty_.store(0);
    // Start writeback of everything dirty and drop what is already clean: the page cache holds
    // a bounded window of the output instead of the whole artifact.
    // Pages shared by two stores stay mapped after each store; drop them all here, so the
    // process holds at most one writeback interval of the output.
    for (const auto& mapping : mappings_) {
        sys::advise(mapping.base, mapping.bytes, sys::Advice::DontNeed);
    }
    for (const int fd : fds_) {
        sys::start_writeback(fd);
        sys::advise_file(fd, 0, 0, sys::Advice::DontNeed);
    }
}

void ArtifactWriter::write(std::uint64_t payload_offset, const void* data, std::size_t bytes) {
    if (payload_offset + bytes > layout_.payload_bytes) {
        throw std::runtime_error("output range exceeds the payload");
    }
    const auto* cursor = static_cast<const std::byte*>(data);
    while (bytes) {
        const auto index = static_cast<std::size_t>(
            std::upper_bound(prefixes_.begin(), prefixes_.end(), payload_offset) -
            prefixes_.begin() - 1);
        const auto count = std::min<std::uint64_t>(bytes, prefixes_[index + 1] - payload_offset);
        const auto start = index == 0 ? layout_.payload_offset : kAlignment;
        store(index, start + payload_offset - prefixes_[index], cursor, count);
        cursor += count;
        payload_offset += count;
        bytes -= count;
    }
}

void ArtifactWriter::store(std::size_t index, std::uint64_t offset, const std::byte* data,
                           std::size_t bytes) {
    if (options_.direct) {
        stage(index, offset, data, bytes);
        return;
    }
    static const auto page = static_cast<std::uint64_t>(sys::page_size());
    auto* base             = mappings_[index].base;
    std::memcpy(base + offset, data, bytes);
    // Unmap the pages this store completed; they stay dirty in the page cache.
    const auto first = (offset + page - 1) / page * page;
    const auto last  = (offset + bytes) / page * page;
    if (last > first) { sys::advise(base + first, last - first, sys::Advice::DontNeed); }
    written(bytes);
}

void ArtifactWriter::unmap() noexcept {
    for (const auto& mapping : mappings_) { sys::unmap(mapping.base, mapping.bytes); }
    mappings_.clear();
}

void ArtifactWriter::sync() {
    for (const int fd : fds_) {
        if (sys::fdatasync(fd) != 0) { throw std::runtime_error("fdatasync of the output failed"); }
    }
}

void ArtifactWriter::finish() { finish({}); }

void ArtifactWriter::read(std::uint64_t payload_offset, void* out, std::size_t bytes) const {
    if (payload_offset + bytes > layout_.payload_bytes) {
        throw std::runtime_error("read range exceeds the payload");
    }
    auto* cursor = static_cast<char*>(out);
    while (bytes) {
        const auto index = static_cast<std::size_t>(
            std::upper_bound(prefixes_.begin(), prefixes_.end(), payload_offset) -
            prefixes_.begin() - 1);
        const auto count = std::min<std::uint64_t>(bytes, prefixes_[index + 1] - payload_offset);
        const auto start = index == 0 ? layout_.payload_offset : kAlignment;
        auto offset      = start + payload_offset - prefixes_[index];
        auto left        = static_cast<std::size_t>(count);
        while (left) {
            const auto got =
                sys::pread(fds_[index], cursor, left, static_cast<std::int64_t>(offset));
            if (got < 0 && errno == EINTR) { continue; }
            if (got <= 0) {
                throw std::runtime_error("cannot read back " + temporary_[index].string() + ": " +
                                         sys::error_text(errno));
            }
            cursor += got;
            offset += static_cast<std::uint64_t>(got);
            left -= static_cast<std::size_t>(got);
        }
        payload_offset += count;
        bytes -= static_cast<std::size_t>(count);
    }
}

OJson content_digest_placeholders(const OJson& objects) {
    OJson out = OJson::object();
    for (const auto& object : objects) {
        out[object.at("id").get<std::string>()] = std::string(64, '0');
    }
    return out;
}

OJson content_digests(const ArtifactWriter& writer, const OJson& objects, unsigned threads) {
    const std::size_t count = objects.size();
    std::vector<std::string> digests(count);
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    const auto work = [&] {
        std::vector<char> buffer(16 << 20);
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                                    EVP_MD_CTX_free);
        try {
            for (std::size_t i = next++; i < count; i = next++) {
                auto offset = objects[i].at("offset").get<std::uint64_t>();
                auto left   = objects[i].at("bytes").get<std::uint64_t>();
                EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr);
                while (left) {
                    const auto n =
                        static_cast<std::size_t>(std::min<std::uint64_t>(left, buffer.size()));
                    writer.read(offset, buffer.data(), n);
                    EVP_DigestUpdate(ctx.get(), buffer.data(), n);
                    offset += n;
                    left -= n;
                }
                unsigned char digest[EVP_MAX_MD_SIZE];
                unsigned size = 0;
                EVP_DigestFinal_ex(ctx.get(), digest, &size);
                static const char* hex = "0123456789abcdef";
                std::string text(2 * size, '0');
                for (unsigned b = 0; b < size; ++b) {
                    text[2 * b]     = hex[digest[b] >> 4];
                    text[2 * b + 1] = hex[digest[b] & 15];
                }
                digests[i] = std::move(text);
            }
        } catch (...) {
            const std::lock_guard lock(error_mutex);
            if (!error) { error = std::current_exception(); }
            next = count;
        }
    };
    if (threads == 0) { threads = std::max(1U, std::thread::hardware_concurrency()); }
    threads =
        static_cast<unsigned>(std::min<std::size_t>(threads, std::max<std::size_t>(count, 1)));
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) { pool.emplace_back(work); }
    work();
    for (auto& thread : pool) { thread.join(); }
    if (error) { std::rethrow_exception(error); }
    OJson out = OJson::object();
    for (std::size_t i = 0; i < count; ++i) {
        out[objects[i].at("id").get<std::string>()] = digests[i];
    }
    return out;
}

void ArtifactWriter::finish(const std::function<std::string()>& seal) {
    if (finished_) { return; }
    try {
        unmap();
        if (options_.direct) { finish_direct(); }
        if (seal) {
            const auto text = seal();
            if (text.size() != layout_.directory.size()) {
                throw std::logic_error("the sealed directory changed its length");
            }
            write_file(0, kHeaderBytes, text.data(), text.size());
            layout_.directory = text;
        }
        for (const int fd : fds_) {
            if (sys::fdatasync(fd) != 0) { throw std::runtime_error("fdatasync failed"); }
            sys::advise_file(fd, 0, 0, sys::Advice::DontNeed);
        }
        for (const int fd : fds_) { sys::close(fd); }
        fds_.clear();
        for (auto& file : direct_) {
            if (file.fd >= 0) { sys::close(file.fd); }
            file.fd = -1;
        }
        std::vector<std::size_t> order;
        for (std::size_t i = 1; i < temporary_.size(); ++i) { order.push_back(i); }
        order.push_back(0);
        // A relayout journal left at this path belongs to an artifact this one replaces.
        std::filesystem::remove(artifact::relayout_journal_path(file_path(0)));
        std::filesystem::remove(artifact::relayout_journal_path(file_path(0)).string() + ".tmp");
        for (const auto i : order) {
            const auto target = file_path(i);
            std::error_code linked;
            std::filesystem::create_hard_link(temporary_[i], target, linked);
            // A file system without hard links (exFAT, FAT32, many network and cloud-synced
            // folders): the temporary file is renamed in instead, once the target is known absent.
            if (linked && linked != std::errc::file_exists && !std::filesystem::exists(target)) {
                std::error_code renamed;
                std::filesystem::rename(temporary_[i], target, renamed);
                if (!renamed) { linked.clear(); }
            }
            if (linked) {
                throw std::runtime_error("cannot publish " + target.string() + ": " + linked.message());
            }
            published_.push_back(target);
        }
        for (const auto& temporary : temporary_) { std::filesystem::remove(temporary); }
        temporary_.clear();
        finished_ = true;
    } catch (...) {
        abort();
        throw;
    }
}

void ArtifactWriter::abort() noexcept {
    stop_writers();
    unmap();
    for (auto& file : direct_) {
        if (file.fd >= 0) { sys::close(file.fd); }
        file.fd = -1;
        for (auto& segment : file.segments) {
            release_segment(segment.data);
            segment.data = nullptr;
        }
    }
    for (auto* data : spare_) { sys::aligned_free(data); }
    spare_.clear();
    for (const int fd : fds_) { sys::close(fd); }
    fds_.clear();
    if (!finished_) {
        std::error_code ignored;
        if (!keep_partial_) {
            for (const auto& path : temporary_) { std::filesystem::remove(path, ignored); }
        }
        for (const auto& path : published_) { std::filesystem::remove(path, ignored); }
    }
    temporary_.clear();
    published_.clear();
}

} // namespace ninferno::convert
