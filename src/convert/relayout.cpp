#include "convert/relayout.h"
#include "convert/files.h"
#include "core/cache_file.h"
#include "core/hex.h"

#include "artifact/expert_groups.h"
#include "artifact/formats.h"
#include "artifact/framing.h"
#include "artifact/reader.h"
#include "convert/pyjson.h"
#include "convert/writer.h"
#include "core/ggml_frag.h"
#include "core/host_memory.h"
#include "spec/manifest.h"
#include "spec/profile.h"

#include "core/sys.h"
#include "core/fnv.h"
#include "core/units.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninferno::convert {
namespace {

using Clock = std::chrono::steady_clock;

// A bank whose layout changes: its code plane is permuted row by row.
struct Bank {
    std::string id;
    QType format            = QType::BF16;
    std::uint64_t k         = 0;
    std::uint64_t plane     = 0; // payload offset of the code plane
    std::uint64_t rows      = 0;
    std::uint64_t row_bytes = 0;
    QuantLayout target      = QuantLayout::GgmlPlanes;
};

// A payload range processed at once: whole rows of one bank, or (with `out`) bytes copied as
// they are.
struct Chunk {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
    const Bank* bank     = nullptr;
};

// A flock on the entry file for the relayout's duration: exclusive in place, shared when only read
// (--out), so a relayout in place never runs beside another relayout of the same artifact.
class EntryLock {
public:
    EntryLock(const std::filesystem::path& entry, bool exclusive) {
        fd_ = sys::open(entry, O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { throw std::runtime_error("cannot open " + entry.string()); }
        if (sys::lock(fd_, false, !exclusive) != 0) {
            sys::close(fd_);
            throw std::runtime_error("another ninferno-convert --relayout of " + entry.string() +
                                     " is running");
        }
    }
    ~EntryLock() { sys::close(fd_); }
    EntryLock(const EntryLock&)            = delete;
    EntryLock& operator=(const EntryLock&) = delete;

private:
    int fd_ = -1;
};

// The entry's directory as stored: its reserved length, the artifact id and (parsed) the JSON.
struct RawDirectory {
    std::uint64_t json_bytes = 0;
    artifact::ArtifactId id{};
    OJson root;
};

RawDirectory read_directory(const std::filesystem::path& entry, bool parse) {
    std::ifstream in(entry, std::ios::binary);
    std::array<std::byte, artifact::kHeaderBytes> header{};
    if (!in.read(reinterpret_cast<char*>(header.data()), header.size())) {
        throw std::runtime_error(entry.string() + ": cannot read the artifact header");
    }
    if (!artifact::has_magic(artifact::kEntryMagic, header.data())) {
        throw std::runtime_error(entry.string() + ": not a .ninferno v3 entry file");
    }
    RawDirectory out;
    out.json_bytes = artifact::read_u64_le(header.data() + 8);
    std::copy(header.begin() + 16, header.end(), out.id.begin());
    std::string text(out.json_bytes, ' ');
    if (!in.read(text.data(), static_cast<std::streamsize>(text.size()))) {
        throw std::runtime_error(entry.string() + ": cannot read the artifact directory");
    }
    if (parse) { out.root = OJson::parse(text); }
    return out;
}

// The banks whose layout `choice` changes: routed-expert objects of a format with a map.
std::vector<Bank> plan_banks(const artifact::Reader& reader, ExpertLayout choice) {
    const auto& directory = reader.directory();
    const auto layers     = artifact::routed_expert_object_layers(directory);
    std::vector<Bank> out;
    for (std::size_t i = 0; i < directory.objects.size(); ++i) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&directory.objects[i]);
        if (tensor == nullptr || !layers[i]) { continue; }
        const auto format = artifact::parse_format(tensor->format);
        const auto& g     = reader.geometry(artifact::ObjectHandle{i});
        if (!is_ggml_block(format) || !ggml_frag_format(format) || g.shape.size() != 2) {
            continue;
        }
        const auto published = ggml_published_layout(format);
        const auto target    = expert_bank_layout(format, g.shape[1], choice, published);
        if (target == g.layout) { continue; }
        Bank bank;
        bank.id        = tensor->id;
        bank.format    = format;
        bank.k         = g.shape[1];
        bank.plane     = tensor->offset;
        bank.rows      = g.shape[0];
        bank.row_bytes = g.code_bytes_per_row;
        bank.target    = target;
        out.push_back(std::move(bank));
    }
    std::sort(out.begin(), out.end(),
              [](const Bank& a, const Bank& b) { return a.plane < b.plane; });
    return out;
}

void add_rows(std::vector<Chunk>& out, const Bank& bank, std::uint64_t chunk) {
    const auto rows = std::max<std::uint64_t>(1, chunk / bank.row_bytes);
    for (std::uint64_t r = 0; r < bank.rows; r += rows) {
        const auto count = std::min(rows, bank.rows - r);
        out.push_back({bank.plane + r * bank.row_bytes, count * bank.row_bytes, &bank});
    }
}

void add_copy(std::vector<Chunk>& out, std::uint64_t begin, std::uint64_t end,
              std::uint64_t chunk) {
    for (auto at = begin; at < end; at += chunk) {
        out.push_back({at, std::min(chunk, end - at), nullptr});
    }
}

// Every bank's code plane, in payload order; with `payload_bytes`, the bytes between them too.
std::vector<Chunk> plan_chunks(const std::vector<Bank>& banks, std::uint64_t chunk,
                               std::optional<std::uint64_t> payload_bytes) {
    if (chunk == 0) { throw std::runtime_error("relayout: chunk bytes must be positive"); }
    std::vector<Chunk> out;
    std::uint64_t at = 0;
    for (const auto& bank : banks) {
        if (payload_bytes) { add_copy(out, at, bank.plane, chunk); }
        add_rows(out, bank, chunk);
        at = bank.plane + bank.rows * bank.row_bytes;
    }
    if (payload_bytes) { add_copy(out, at, *payload_bytes, chunk); }
    return out;
}

void permute(const Chunk& chunk, const std::uint8_t* in, std::uint8_t* out) {
    const auto& bank = *chunk.bank;
    const bool to    = bank.target == QuantLayout::GgmlFrag;
    for (std::uint64_t r = 0; r < chunk.bytes / bank.row_bytes; ++r) {
        ggml_frag_row(bank.format, bank.k, in + r * bank.row_bytes, out + r * bank.row_bytes, to);
    }
}

// The directory with every bank's new layout name.
OJson relabeled(OJson root, const std::vector<Bank>& banks) {
    std::map<std::string, QuantLayout> target;
    for (const auto& bank : banks) { target[bank.id] = bank.target; }
    for (auto& object : root.at("objects")) {
        const auto it = target.find(object.at("id").get<std::string>());
        if (it != target.end()) {
            object["layout"] = std::string(artifact::layout_name(it->second));
        }
    }
    return root;
}

void write_all(int fd, const void* data, std::uint64_t bytes, std::uint64_t offset,
               const std::string& what) {
    const auto* p = static_cast<const char*>(data);
    while (bytes) {
        const auto n = sys::pwrite(fd, p, std::min<std::uint64_t>(bytes, kGiB),
                                   static_cast<std::int64_t>(offset));
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) {
            throw std::runtime_error("cannot write " + what + ": " + sys::error_text(errno));
        }
        p += n;
        bytes -= static_cast<std::uint64_t>(n);
        offset += static_cast<std::uint64_t>(n);
    }
}

// The directory text (its reserved length) written over the entry's and made durable.
void write_directory(const std::filesystem::path& entry, const std::vector<std::uint8_t>& text) {
    const int fd = sys::open(entry, O_RDWR | O_CLOEXEC);
    if (fd < 0) { throw std::runtime_error("cannot open " + entry.string() + " for writing"); }
    try {
        write_all(fd, text.data(), text.size(), artifact::kHeaderBytes, entry.string());
        if (sys::fdatasync(fd) != 0) { throw std::runtime_error("cannot sync " + entry.string()); }
    } catch (...) {
        sys::close(fd);
        throw;
    }
    sys::close(fd);
}

// The artifact's files opened for writing, addressed by payload offset.
class Files {
public:
    Files(const std::filesystem::path& entry, const artifact::Reader& reader) : reader_(reader) {
        const auto& files = reader.directory().files;
        for (std::size_t i = 0; i < files.size(); ++i) {
            paths_.push_back(i == 0 ? entry : entry.parent_path() / *files[i].path);
            const int fd = sys::open(paths_.back(), O_RDWR | O_CLOEXEC);
            if (fd < 0) {
                close_all();
                throw std::runtime_error("cannot open " + paths_.back().string() + " for writing");
            }
            fds_.push_back(fd);
        }
    }
    ~Files() { close_all(); }
    Files(const Files&)            = delete;
    Files& operator=(const Files&) = delete;

    void write(std::uint64_t offset, const std::uint8_t* data, std::uint64_t bytes) {
        for (const auto& s : reader_.segments(offset, bytes)) {
            write_all(fds_.at(s.file_index), data + s.destination_offset, s.bytes, s.file_offset,
                      paths_.at(s.file_index).string());
        }
    }

    void sync() {
        for (std::size_t i = 0; i < fds_.size(); ++i) {
            if (sys::fdatasync(fds_[i]) != 0) {
                throw std::runtime_error("cannot sync " + paths_[i].string());
            }
        }
    }

private:
    void close_all() noexcept {
        for (const int fd : fds_) { sys::close(fd); }
        fds_.clear();
    }

    const artifact::Reader& reader_;
    std::vector<std::filesystem::path> paths_;
    std::vector<int> fds_;
};

// ---------------------------------------------------------------------------------------------
// The undo journal: magic, then little-endian u64 fields (phase, chunk position, payload offset,
// data bytes, layout choice, check offset, check bytes, check hash, checksum), the artifact id,
// then the data: the chunk's bytes as they were before it was overwritten (phase 1), or the new
// directory text (phase 2). The check names the last chunk written in full and the FNV-1a hash of
// its new bytes: a resume verifies them, so an artifact put back from a copy made before the
// relayout (same id, every chunk as it was) is refused instead of finished half relaid. It is
// written to a temporary name, synced and renamed, so it is always one complete record.

constexpr std::array<char, 8> kJournalMagic = {'N', 'I', 'N', 'F', 'R', 'L', 'Y', '2'};
constexpr std::uint64_t kPhaseChunk         = 1;
constexpr std::uint64_t kPhaseDirectory     = 2;
constexpr std::size_t kJournalHeader        = 8 + 9 * 8 + 16;

struct Journal {
    std::uint64_t phase = 0, position = 0, offset = 0, layout = 0;
    std::uint64_t check_offset = 0, check_bytes = 0, check_hash = 0;
    artifact::ArtifactId id{};
    std::vector<std::uint8_t> data;
};

constexpr std::uint64_t kFnvBasis = fnv::kOffset;

std::uint64_t fnv(const std::uint8_t* data, std::uint64_t bytes, std::uint64_t h = kFnvBasis) {
    for (std::uint64_t i = 0; i < bytes; ++i) { h = fnv::mix(h, data[i]); }
    return h;
}

std::uint64_t checksum(const Journal& j) {
    std::uint64_t h    = kFnvBasis;
    const auto mix     = [&](std::uint8_t byte) { h = fnv::mix(h, byte); };
    const auto mix_u64 = [&](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) { mix(static_cast<std::uint8_t>(v >> (8 * i))); }
    };
    mix_u64(j.phase);
    mix_u64(j.position);
    mix_u64(j.offset);
    mix_u64(j.data.size());
    mix_u64(j.layout);
    mix_u64(j.check_offset);
    mix_u64(j.check_bytes);
    mix_u64(j.check_hash);
    for (const auto b : j.id) { mix(std::to_integer<std::uint8_t>(b)); }
    for (const auto b : j.data) { mix(b); }
    return h;
}

void put(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) { out.push_back(static_cast<std::uint8_t>(v >> (8 * i))); }
}

std::uint64_t get(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) { v |= static_cast<std::uint64_t>(p[i]) << (8 * i); }
    return v;
}

void write_journal(const std::filesystem::path& path, const Journal& j) {
    std::vector<std::uint8_t> head(kJournalMagic.begin(), kJournalMagic.end());
    put(head, j.phase);
    put(head, j.position);
    put(head, j.offset);
    put(head, j.data.size());
    put(head, j.layout);
    put(head, j.check_offset);
    put(head, j.check_bytes);
    put(head, j.check_hash);
    put(head, checksum(j));
    for (const auto b : j.id) { head.push_back(std::to_integer<std::uint8_t>(b)); }
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    const int fd = sys::open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { throw std::runtime_error("cannot create " + temporary.string()); }
    try {
        write_all(fd, head.data(), head.size(), 0, temporary.string());
        write_all(fd, j.data.data(), j.data.size(), head.size(), temporary.string());
        if (sys::fdatasync(fd) != 0) { throw std::runtime_error("cannot sync " + temporary.string()); }
    } catch (...) {
        sys::close(fd);
        throw;
    }
    sys::close(fd);
    std::filesystem::rename(temporary, path);
    sync_directory_of(path);
}

std::optional<Journal> read_journal(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return std::nullopt; }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    if (bytes.size() < kJournalHeader ||
        !std::equal(kJournalMagic.begin(), kJournalMagic.end(), bytes.begin())) {
        throw std::runtime_error(path.string() + " is not a relayout journal");
    }
    const auto* p = bytes.data() + 8;
    Journal j;
    j.phase                = get(p);
    j.position             = get(p + 8);
    j.offset               = get(p + 16);
    const auto size        = get(p + 24);
    j.layout               = get(p + 32);
    j.check_offset         = get(p + 40);
    j.check_bytes          = get(p + 48);
    j.check_hash           = get(p + 56);
    const auto stored_hash = get(p + 64);
    for (std::size_t i = 0; i < 16; ++i) { j.id[i] = std::byte{bytes[8 + 72 + i]}; }
    if (bytes.size() != kJournalHeader + size) {
        throw std::runtime_error(path.string() + ": relayout journal of the wrong length");
    }
    j.data.assign(bytes.begin() + kJournalHeader, bytes.end());
    if (checksum(j) != stored_hash) {
        throw std::runtime_error(path.string() + ": relayout journal checksum differs");
    }
    return j;
}

void log_line(const RelayoutOptions& options, const std::string& line) {
    if (options.log) { options.log(line); }
}

// Payload bytes read from the files the directory `root` (reserved length `json_bytes`) lists,
// without a Reader: a resume inside the directory write has no readable directory in the entry.
std::vector<std::uint8_t> read_payload_raw(const std::filesystem::path& entry, const OJson& root,
                                           std::uint64_t json_bytes, std::uint64_t offset,
                                           std::uint64_t bytes) {
    std::vector<std::uint8_t> out(bytes);
    std::uint64_t file_begin = 0, done = 0;
    const auto& files        = root.at("files");
    for (std::size_t i = 0; i < files.size() && done < bytes; ++i) {
        const auto count = files[i].at("payload_bytes").get<std::uint64_t>();
        const auto at    = offset + done;
        if (at < file_begin + count) {
            const auto take = std::min(bytes - done, file_begin + count - at);
            const auto path = i == 0 ? entry
                                     : entry.parent_path() / files[i].at("path").get<std::string>();
            const auto start = i == 0 ? (artifact::kHeaderBytes + json_bytes + 4095) / 4096 * 4096
                                      : std::uint64_t{4096};
            std::ifstream in(path, std::ios::binary);
            in.seekg(static_cast<std::streamoff>(start + at - file_begin));
            if (!in.read(reinterpret_cast<char*>(out.data() + done),
                         static_cast<std::streamsize>(take))) {
                throw std::runtime_error("cannot read " + path.string());
            }
            done += take;
        }
        file_begin += count;
    }
    if (done != bytes) {
        throw std::runtime_error(entry.string() + ": payload range past the end");
    }
    return out;
}

// A resume's check: the journal's last full chunk holds the bytes the relayout wrote there.
void verify_resume(const std::filesystem::path& entry, const std::filesystem::path& journal,
                   const Journal& j, const OJson& root, std::uint64_t json_bytes) {
    if (j.check_bytes == 0) { return; }
    const auto now = read_payload_raw(entry, root, json_bytes, j.check_offset, j.check_bytes);
    if (fnv(now.data(), now.size()) != j.check_hash) {
        throw std::runtime_error(
            entry.string() + " no longer holds what the interrupted relayout wrote (payload " +
            "bytes " + std::to_string(j.check_offset) + " + " + std::to_string(j.check_bytes) +
            " differ; was it restored from a copy?). If it is a complete copy from before the "
            "relayout, delete " + journal.string() + " to use it as it is; otherwise convert it "
            "again. Resuming would leave banks in the other order than the directory says.");
    }
}

// The new directory text, padded to the reserved length the header records.
std::string padded_directory(const OJson& root, std::uint64_t json_bytes) {
    auto text = py_dumps_compact(root);
    if (text.size() > json_bytes) {
        throw std::runtime_error("the relabeled directory (" + std::to_string(text.size()) +
                                 " bytes) exceeds its reserved " + std::to_string(json_bytes) +
                                 " bytes; relayout to a new artifact with --out");
    }
    text.append(json_bytes - text.size(), ' ');
    return text;
}

// The objects `root` stores in ggml_frag_v1.
OJson frag_objects(const OJson& root) {
    auto out = OJson::array();
    for (const auto& object : root.at("objects")) {
        if (object.value("layout", std::string()) == "ggml_frag_v1") {
            out.push_back(object.at("id"));
        }
    }
    return out;
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    write_file_atomic(path, text);
}

// <artifact>.conversion.json (from `source`, written to `target`'s) with a "relayouts" record:
// the choice, the banks it changed and every object then in ggml_frag_v1; with `new_id`, the
// report's artifact id and output are the new artifact's.
void note_relayout(const std::filesystem::path& source, const std::filesystem::path& target,
                   const OJson& root, const RelayoutOptions& options, std::uint64_t changed,
                   const std::optional<std::string>& new_id) {
    const auto from = std::filesystem::path(source.string() + ".conversion.json");
    std::ifstream in(from);
    if (!in) { return; }
    auto report = OJson::parse(in, nullptr, false);
    if (report.is_discarded() || !report.is_object()) { return; }
    if (new_id) {
        report["artifact_id"] = *new_id;
        report["output"]      = target.string();
    }
    auto& relayouts = report["relayouts"];
    if (!relayouts.is_array()) { relayouts = OJson::array(); }
    OJson record = OJson::object();
    record["layout"]       = std::string(expert_layout_name(options.layout));
    record["from"]         = source.string();
    record["changed"]      = changed;
    record["ggml_frag_v1"] = frag_objects(root);
    relayouts.push_back(std::move(record));
    write_text(target.string() + ".conversion.json", py_dumps_indent2(report) + "\n");
}

// The source's speculation manifest and spec-tune profile for the new artifact: its target id is
// the new artifact's, and a drafter artifact named by a relative path is named absolutely.
void carry_speculation(const std::filesystem::path& source, const std::filesystem::path& out,
                       const std::string& new_id, const RelayoutOptions& options) {
    const auto dir = std::filesystem::absolute(source).parent_path();
    const auto carry = [&](const std::filesystem::path& from, const std::filesystem::path& to,
                           const auto& rewrite) {
        std::ifstream in(from);
        if (!in) { return; }
        auto doc = nlohmann::json::parse(in, nullptr, false);
        if (doc.is_discarded() || !doc.is_object()) {
            log_line(options, "relayout: " + from.string() + " does not parse; not carried over");
            return;
        }
        rewrite(doc);
        write_text(to, doc.dump(2) + "\n");
        log_line(options, "relayout: " + to.string() + " carried over for the new artifact id");
    };
    const auto absolute_drafters = [&](nlohmann::json& list) {
        if (!list.is_array()) { return; }
        for (auto& d : list) {
            if (d.is_object() && d.contains("artifact") && d["artifact"].is_string()) {
                const std::filesystem::path path = d["artifact"].get<std::string>();
                if (!path.empty() && path.is_relative()) { d["artifact"] = (dir / path).string(); }
            }
        }
    };
    carry(spec::manifest_path(source), spec::manifest_path(out), [&](nlohmann::json& doc) {
        doc["target"]["artifact_id"] = new_id;
        if (doc.contains("drafters")) { absolute_drafters(doc["drafters"]); }
    });
    carry(spec::default_profile_path(source), spec::default_profile_path(out),
          [&](nlohmann::json& doc) {
              doc["key"]["artifact_id"] = new_id;
              if (doc.contains("drafters")) { absolute_drafters(doc["drafters"]); }
          });
}

RelayoutResult in_place(const RelayoutOptions& options, RelayoutResult result,
                        const Clock::time_point begin) {
    const auto& entry   = options.artifact;
    const EntryLock lock(entry, true);
    const auto journal  = relayout_journal(entry);
    const auto pending  = read_journal(journal);
    const auto choice   = static_cast<std::uint64_t>(options.layout);
    if (pending && pending->layout != choice) {
        throw std::runtime_error(
            "an interrupted --relayout " +
            std::string(expert_layout_name(static_cast<ExpertLayout>(pending->layout))) +
            " of this artifact is pending (" + journal.string() + "); rerun that command");
    }
    if (pending && read_directory(entry, false).id != pending->id) {
        throw std::runtime_error(journal.string() + " belongs to another artifact, not to " +
                                 entry.string() + "; delete it (rm " + journal.string() +
                                 ") to use this artifact");
    }
    if (pending && pending->phase == kPhaseDirectory) {
        // Every chunk is done; the directory write (which may have been cut short) is redone.
        const std::string text(pending->data.begin(), pending->data.end());
        verify_resume(entry, journal, *pending, OJson::parse(text), pending->data.size());
        write_directory(entry, pending->data);
        std::filesystem::remove(journal);
        sync_directory_of(journal);
        log_line(options, "relayout: resumed; the directory is written");
        result.resumed = true;
        result.seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        return result;
    }
    const RawDirectory dir = read_directory(entry, true);
    const artifact::Reader reader(entry, artifact::PendingRelayout::Allow);
    const auto banks  = plan_banks(reader, options.layout);
    const auto chunks = plan_chunks(banks, options.chunk_bytes, std::nullopt);
    if (banks.empty()) {
        if (pending) { throw std::runtime_error(journal.string() + ": no bank left to finish"); }
        log_line(options, "relayout: every routed-expert bank is already in its " +
                              std::string(expert_layout_name(options.layout)) + " layout");
        return result;
    }
    const auto new_root = relabeled(dir.root, banks);
    const auto text     = padded_directory(new_root, dir.json_bytes);
    std::uint64_t largest = 0;
    for (const auto& chunk : chunks) { largest = std::max(largest, chunk.bytes); }
    if (!pending && free_bytes_beside(entry) < 2 * largest + text.size()) {
        throw std::runtime_error("relayout in place needs " +
                                 format_gib(2 * largest + text.size()) +
                                 " free next to the artifact for its journal; " +
                                 format_gib(free_bytes_beside(entry)) + " free");
    }
    Files files(entry, reader);
    std::size_t start = 0;
    if (pending) {
        if (pending->position >= chunks.size() ||
            chunks[pending->position].offset != pending->offset ||
            chunks[pending->position].bytes != pending->data.size()) {
            throw std::runtime_error(journal.string() + " does not match this artifact's banks");
        }
        verify_resume(entry, journal, *pending, dir.root, dir.json_bytes);
        files.write(pending->offset, pending->data.data(), pending->data.size());
        files.sync();
        start          = static_cast<std::size_t>(pending->position);
        result.resumed = true;
        log_line(options, "relayout: resumed at chunk " + std::to_string(start) + " of " +
                              std::to_string(chunks.size()));
    }
    Journal record;
    record.phase  = kPhaseChunk;
    record.layout = choice;
    record.id     = dir.id;
    if (pending) { // the chunk before the interrupted one, verified above
        record.check_offset = pending->check_offset;
        record.check_bytes  = pending->check_bytes;
        record.check_hash   = pending->check_hash;
    }
    std::vector<std::uint8_t> permuted;
    for (std::size_t i = start; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        record.position   = i;
        record.offset     = chunk.offset;
        record.data.resize(chunk.bytes);
        reader.read_into(chunk.offset,
                         std::span(reinterpret_cast<std::byte*>(record.data.data()), chunk.bytes));
        write_journal(journal, record);
        permuted.resize(chunk.bytes);
        permute(chunk, record.data.data(), permuted.data());
        if (options.interrupt_after && result.code_bytes + chunk.bytes > *options.interrupt_after) {
            const auto partial = *options.interrupt_after - result.code_bytes;
            files.write(chunk.offset, permuted.data(), partial);
            files.sync();
            throw std::runtime_error("relayout interrupted (test)");
        }
        files.write(chunk.offset, permuted.data(), chunk.bytes);
        files.sync();
        result.code_bytes += chunk.bytes;
        record.check_offset = chunk.offset;
        record.check_bytes  = chunk.bytes;
        record.check_hash   = fnv(permuted.data(), permuted.size());
    }
    record.phase    = kPhaseDirectory;
    record.position = chunks.size();
    record.offset   = 0;
    record.data.assign(text.begin(), text.end());
    write_journal(journal, record);
    if (options.interrupt_directory) {
        const std::vector<std::uint8_t> half(record.data.begin(),
                                             record.data.begin() + record.data.size() / 2);
        write_directory(entry, half);
        throw std::runtime_error("relayout interrupted (test)");
    }
    write_directory(entry, record.data);
    std::filesystem::remove(journal);
    sync_directory_of(journal);
    note_relayout(entry, entry, new_root, options, banks.size(), std::nullopt);
    result.objects = banks.size();
    result.seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    return result;
}

RelayoutResult to_new(const RelayoutOptions& options, RelayoutResult result,
                      const Clock::time_point begin) {
    const auto& out = *options.out;
    if (std::filesystem::exists(out)) {
        throw std::runtime_error("relayout output already exists: " + out.string());
    }
    if (std::filesystem::exists(relayout_journal(options.artifact))) {
        throw std::runtime_error(options.artifact.string() +
                                 " has an unfinished in-place --relayout (" +
                                 relayout_journal(options.artifact).string() +
                                 "); finish it first by rerunning that command");
    }
    const EntryLock lock(options.artifact, false);
    const artifact::Reader reader(options.artifact);
    const RawDirectory dir = read_directory(options.artifact, true);
    const auto banks    = plan_banks(reader, options.layout);
    const auto payload  = reader.directory().payload_bytes;
    const auto chunks   = plan_chunks(banks, options.chunk_bytes, payload);
    OJson description   = relabeled(dir.root, banks);
    const OJson new_root = description;
    description.erase("files");
    auto layout = layout_artifact(out.filename().string(), py_dumps_compact(description), payload,
                                  options.max_file_bytes);
    create_parent_directories(out);
    const auto needed = layout.file_bytes();
    if (free_bytes_beside(out) < needed + (64ULL << 20)) {
        throw std::runtime_error("relayout to " + out.string() + " needs " + format_gib(needed) +
                                 "; " + format_gib(free_bytes_beside(out)) +
                                 " free (relayout in place without --out)");
    }
    ArtifactWriter writer(out, std::move(layout));
    try {
        std::vector<std::uint8_t> in, permuted;
        for (const auto& chunk : chunks) {
            in.resize(chunk.bytes);
            reader.read_into(chunk.offset,
                             std::span(reinterpret_cast<std::byte*>(in.data()), chunk.bytes));
            if (chunk.bank == nullptr) {
                writer.write(chunk.offset, in.data(), chunk.bytes);
                continue;
            }
            permuted.resize(chunk.bytes);
            permute(chunk, in.data(), permuted.data());
            writer.write(chunk.offset, permuted.data(), chunk.bytes);
            result.code_bytes += chunk.bytes;
        }
        writer.finish();
    } catch (...) {
        writer.abort();
        throw;
    }
    const auto new_id = hex_encode(writer.artifact_id().data(), writer.artifact_id().size());
    carry_speculation(options.artifact, out, new_id, options);
    note_relayout(options.artifact, out, new_root, options, banks.size(), new_id);
    result.objects = banks.size();
    result.seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    return result;
}

} // namespace

std::filesystem::path relayout_journal(const std::filesystem::path& artifact) {
    return artifact::relayout_journal_path(artifact);
}

RelayoutResult relayout(const RelayoutOptions& options) {
    const auto begin = Clock::now();
    RelayoutResult result;
    result = options.out ? to_new(options, result, begin) : in_place(options, result, begin);
    if (result.objects) {
        log_line(options, "relayout: " + std::to_string(result.objects) +
                              " routed-expert banks changed layout (" +
                              format_gib(result.code_bytes) + " of code bytes permuted)");
    }
    return result;
}

} // namespace ninferno::convert
