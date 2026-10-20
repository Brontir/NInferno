#include "convert/consume.h"
#include "convert/files.h"

#include "core/sys.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninferno::convert {
namespace {

constexpr char kJournalMagic[8]      = {'N', 'I', 'N', 'J', 'R', 'N', 'L', '1'};
constexpr std::size_t kJournalHeader = 40;
constexpr std::uint32_t kRecordCheck = 0x9E3779B9U;

void put_u64(std::byte* out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) { out[i] = std::byte((value >> (8 * i)) & 0xFF); }
}

std::uint64_t get_u64(const std::byte* in) {
    std::uint64_t out = 0;
    for (int i = 0; i < 8; ++i) {
        out |= std::uint64_t(std::to_integer<unsigned>(in[i])) << (8 * i);
    }
    return out;
}

void write_all(int fd, const void* data, std::size_t bytes, const std::string& label) {
    const auto* cursor = static_cast<const char*>(data);
    while (bytes) {
        const auto count = sys::write(fd, cursor, bytes);
        if (count <= 0) { throw std::runtime_error("cannot write " + label); }
        cursor += count;
        bytes -= static_cast<std::size_t>(count);
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------

ConsumptionLedger::ConsumptionLedger(std::vector<std::uint64_t> shard_bytes, std::uint64_t block,
                                     std::vector<SourceSegment> segments, std::size_t objects)
    : shard_bytes_(std::move(shard_bytes)), block_(block), segments_(std::move(segments)),
      remaining_(segments_.size()), freed_(segments_.size(), false), reads_(objects),
      durable_(objects, false) {
    if (!block_) { throw std::invalid_argument("block size must be positive"); }
    live_.resize(shard_bytes_.size());
    for (std::size_t s = 0; s < shard_bytes_.size(); ++s) {
        live_[s].assign((shard_bytes_[s] + block_ - 1) / block_, 0);
    }
    for (std::size_t i = 0; i < segments_.size(); ++i) {
        auto& segment = segments_[i];
        std::sort(segment.readers.begin(), segment.readers.end());
        segment.readers.erase(std::unique(segment.readers.begin(), segment.readers.end()),
                              segment.readers.end());
        remaining_[i] = static_cast<std::uint32_t>(segment.readers.size());
        for (const auto object : segment.readers) {
            if (object >= objects) { throw std::invalid_argument("segment reader out of range"); }
            reads_[object].push_back(static_cast<std::uint32_t>(i));
        }
        if (segment.begin >= segment.end) { continue; }
        auto& counts = live_.at(static_cast<std::size_t>(segment.shard));
        for (auto b = segment.begin / block_; b <= (segment.end - 1) / block_; ++b) {
            if (counts[b] == std::numeric_limits<std::uint16_t>::max()) {
                throw std::runtime_error("too many checkpoint tensors share one block");
            }
            ++counts[b];
        }
    }
}

std::uint64_t ConsumptionLedger::segment_bytes(std::size_t segment) const noexcept {
    const auto& s = segments_[segment];
    return s.end - s.begin;
}

void ConsumptionLedger::release(std::size_t index, std::vector<Punch>& out) {
    auto& segment = segments_[index];
    if (freed_[index] || segment.pinned) { return; }
    freed_[index] = true;
    if (segment.begin >= segment.end) { return; }
    auto& counts = live_[static_cast<std::size_t>(segment.shard)];
    for (auto b = segment.begin / block_; b <= (segment.end - 1) / block_; ++b) {
        if (--counts[b] != 0) { continue; }
        const auto begin = b * block_;
        const auto end =
            std::min(begin + block_, shard_bytes_[static_cast<std::size_t>(segment.shard)]);
        if (!out.empty() && out.back().shard == segment.shard &&
            out.back().offset + out.back().bytes == begin) {
            out.back().bytes += end - begin;
        } else {
            out.push_back({segment.shard, begin, end - begin});
        }
    }
}

std::vector<Punch> ConsumptionLedger::initial() {
    std::vector<Punch> out;
    for (std::size_t i = 0; i < segments_.size(); ++i) {
        if (segments_[i].readers.empty()) { release(i, out); }
    }
    return out;
}

std::vector<Punch> ConsumptionLedger::durable(std::span<const std::uint32_t> objects) {
    std::vector<Punch> out;
    for (const auto object : objects) {
        if (durable_.at(object)) { continue; }
        durable_[object] = true;
        for (const auto segment : reads_[object]) {
            if (--remaining_[segment] == 0) { release(segment, out); }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------

std::uint64_t consumption_peak(const ConsumptionLedger& ledger,
                               const std::vector<std::uint64_t>& output_bytes,
                               const std::vector<std::uint32_t>& order, std::uint64_t prologue,
                               std::uint64_t window) {
    auto simulation = ledger;
    const auto sum  = [](const std::vector<Punch>& punches) {
        std::uint64_t total = 0;
        for (const auto& punch : punches) { total += punch.bytes; }
        return total;
    };
    // Signed: freeing can take the checkpoint below its starting size.
    long double net =
        static_cast<long double>(prologue) - static_cast<long double>(sum(simulation.initial()));
    long double peak = std::max<long double>(net, 0);
    for (std::size_t p = 0; p < order.size(); ++p) {
        // Objects in flight while the prefix [0, p) is committed: in order, up to the window,
        // and always at least the next one.
        long double inflight = static_cast<long double>(output_bytes[order[p]]);
        for (auto q = p + 1; q < order.size(); ++q) {
            const auto next = inflight + static_cast<long double>(output_bytes[order[q]]);
            if (next > static_cast<long double>(window)) { break; }
            inflight = next;
        }
        peak                          = std::max(peak, net + inflight);
        const std::uint32_t object[1] = {order[p]};
        net += static_cast<long double>(output_bytes[order[p]]) -
               static_cast<long double>(sum(simulation.durable(object)));
    }
    peak = std::max(peak, net);
    return static_cast<std::uint64_t>(std::max<long double>(peak, 0));
}

Schedule schedule_consumption(const ConsumptionLedger& ledger,
                              const std::vector<std::uint64_t>& output_bytes,
                              std::uint64_t prologue, std::uint64_t window) {
    const auto objects   = output_bytes.size();
    const auto& segments = ledger.segments();
    std::vector<std::uint32_t> remaining(segments.size());
    std::vector<std::vector<std::uint32_t>> reads(objects);
    for (std::size_t i = 0; i < segments.size(); ++i) {
        remaining[i] = static_cast<std::uint32_t>(segments[i].readers.size());
        for (const auto object : segments[i].readers) {
            reads[object].push_back(static_cast<std::uint32_t>(i));
        }
    }
    std::vector<bool> done(objects, false);
    Schedule out;
    out.window = window;
    // Objects already durable (a resumed run) are not scheduled; they still release readers.
    for (std::uint32_t o = 0; o < objects; ++o) {
        if (!ledger.is_durable(o)) { continue; }
        done[o] = true;
        for (const auto s : reads[o]) { --remaining[s]; }
    }
    while (true) {
        std::optional<std::uint32_t> best_freeing;
        std::optional<std::uint32_t> best_other;
        std::uint64_t best_freeing_out = 0, best_other_freed = 0, best_other_out = 0;
        for (std::uint32_t o = 0; o < objects; ++o) {
            if (done[o]) { continue; }
            std::uint64_t freed = 0;
            for (const auto s : reads[o]) {
                if (remaining[s] == 1 && !segments[s].pinned) { freed += ledger.segment_bytes(s); }
            }
            const auto written = output_bytes[o];
            if (freed >= written) {
                if (!best_freeing || written < best_freeing_out) {
                    best_freeing     = o;
                    best_freeing_out = written;
                }
            } else if (!best_other || freed > best_other_freed ||
                       (freed == best_other_freed && written < best_other_out)) {
                best_other       = o;
                best_other_freed = freed;
                best_other_out   = written;
            }
        }
        const auto pick = best_freeing ? best_freeing : best_other;
        if (!pick) { break; }
        done[*pick] = true;
        for (const auto s : reads[*pick]) { --remaining[s]; }
        out.order.push_back(*pick);
    }
    out.peak_extra = consumption_peak(ledger, output_bytes, out.order, prologue, window);
    return out;
}

// ---------------------------------------------------------------------------------------------

Journal::Journal(std::filesystem::path path, std::uint64_t fingerprint,
                 const std::array<std::byte, 16>& artifact_id, std::uint64_t objects)
    : path_(std::move(path)), id_(artifact_id) {
    if (std::filesystem::exists(path_)) {
        fd_ = sys::open(path_, O_RDWR | O_CLOEXEC);
        if (fd_ < 0) { throw std::runtime_error("cannot open journal " + path_.string()); }
        const auto size = static_cast<std::size_t>(sys::seek(fd_, 0, SEEK_END));
        std::vector<std::byte> data(size);
        if (sys::pread(fd_, data.data(), size, 0) != static_cast<std::int64_t>(size) ||
            size < kJournalHeader || std::memcmp(data.data(), kJournalMagic, 8) != 0) {
            throw std::runtime_error("journal " + path_.string() + " is damaged");
        }
        if (get_u64(data.data() + 8) != fingerprint || get_u64(data.data() + 32) != objects) {
            throw std::runtime_error("journal " + path_.string() +
                                     " belongs to a different conversion plan; remove it and the "
                                     "partial output only if the checkpoint was not consumed");
        }
        std::memcpy(id_.data(), data.data() + 16, 16);
        std::size_t valid = kJournalHeader;
        for (auto at = kJournalHeader; at + 8 <= size; at += 8) {
            std::uint32_t index = 0, check = 0;
            std::memcpy(&index, data.data() + at, 4);
            std::memcpy(&check, data.data() + at + 4, 4);
            if ((index ^ kRecordCheck) != check || index >= objects) { break; }
            completed_.push_back(index);
            valid = at + 8;
        }
        if (valid != size && sys::truncate(fd_, static_cast<std::int64_t>(valid)) != 0) {
            throw std::runtime_error("cannot trim journal " + path_.string());
        }
        sys::seek(fd_, static_cast<std::int64_t>(valid), SEEK_SET);
        resumed_ = true;
        return;
    }
    fd_ = sys::open(path_, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd_ < 0) { throw std::runtime_error("cannot create journal " + path_.string()); }
    std::byte header[kJournalHeader];
    std::memcpy(header, kJournalMagic, 8);
    put_u64(header + 8, fingerprint);
    std::memcpy(header + 16, id_.data(), 16);
    put_u64(header + 32, objects);
    write_all(fd_, header, sizeof(header), path_.string());
    if (sys::fsync(fd_) != 0) { throw std::runtime_error("cannot fsync " + path_.string()); }
    sync_directory_of(path_);
}

Journal::~Journal() {
    if (fd_ >= 0) { sys::close(fd_); }
}

void Journal::append(std::span<const std::uint32_t> objects) {
    std::vector<std::byte> records(objects.size() * 8);
    for (std::size_t i = 0; i < objects.size(); ++i) {
        const std::uint32_t check = objects[i] ^ kRecordCheck;
        std::memcpy(records.data() + 8 * i, &objects[i], 4);
        std::memcpy(records.data() + 8 * i + 4, &check, 4);
    }
    write_all(fd_, records.data(), records.size(), path_.string());
    if (sys::fdatasync(fd_) != 0) { throw std::runtime_error("cannot fsync " + path_.string()); }
    completed_.insert(completed_.end(), objects.begin(), objects.end());
}

void Journal::remove() noexcept {
    if (fd_ >= 0) {
        sys::close(fd_);
        fd_ = -1;
    }
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    sync_directory_of(path_);
}

std::optional<std::array<std::byte, 16>> Journal::existing_id(const std::filesystem::path& path) {
    const int fd = sys::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return std::nullopt; }
    std::byte header[kJournalHeader];
    const auto count = sys::pread(fd, header, sizeof(header), 0);
    sys::close(fd);
    if (count != static_cast<std::int64_t>(sizeof(header)) ||
        std::memcmp(header, kJournalMagic, 8) != 0) {
        return std::nullopt;
    }
    std::array<std::byte, 16> id{};
    std::memcpy(id.data(), header + 16, 16);
    return id;
}

// ---------------------------------------------------------------------------------------------

Committer::Committer(ConsumptionLedger& ledger, CommitIo& io, std::vector<std::uint32_t> order,
                     std::span<const std::uint32_t> already_durable)
    : ledger_(ledger), io_(io), order_(std::move(order)) {
    std::size_t objects = 0;
    for (const auto o : order_) { objects = std::max<std::size_t>(objects, o + 1); }
    for (const auto o : already_durable) { objects = std::max<std::size_t>(objects, o + 1); }
    complete_.assign(objects, false);
    // Durable before this run: their journal records are already synced, so their blocks (and
    // every unread block) may be punched now, before anything else happens.
    auto punches       = ledger_.initial();
    const auto resumed = ledger_.durable(already_durable);
    punches.insert(punches.end(), resumed.begin(), resumed.end());
    for (const auto& punch : punches) {
        io_.punch(punch);
        punched_ += punch.bytes;
    }
}

void Committer::complete(std::uint32_t object) {
    std::lock_guard lock(mutex_);
    complete_.at(object) = true;
}

std::size_t Committer::commit() {
    std::vector<std::uint32_t> batch;
    {
        std::lock_guard lock(mutex_);
        for (auto i = next_; i < order_.size() && complete_[order_[i]]; ++i) {
            batch.push_back(order_[i]);
        }
    }
    if (batch.empty()) { return 0; }
    io_.sync_output();
    io_.journal(batch);
    for (const auto& punch : ledger_.durable(batch)) {
        io_.punch(punch);
        punched_ += punch.bytes;
    }
    std::lock_guard lock(mutex_);
    next_ += batch.size();
    return batch.size();
}

bool Committer::done() const {
    std::lock_guard lock(mutex_);
    return next_ == order_.size();
}

std::size_t Committer::committed() const {
    std::lock_guard lock(mutex_);
    return next_;
}

} // namespace ninferno::convert
