#include "convert/fetch.h"

#include "core/ascii.h"
#include "core/hex.h"
#include "core/sha256.h"

#include "convert/gguf.h"

#include "core/sys.h"
#include "core/units.h"

#include <curl/curl.h>
#include <openssl/evp.h>
#ifndef _WIN32
#include <fnmatch.h>
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninferno::convert {
namespace {

namespace fs = std::filesystem;
using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;

// fs::rename that waits out a brief hold on the file: on Windows an antivirus scanner or the
// search indexer opens a freshly written multi-GB download for a moment, and the rename fails with
// a sharing violation meanwhile. Up to five seconds, then the error.
void rename_settled(const fs::path& from, const fs::path& to) {
    std::error_code error;
    for (int attempt = 0; attempt < 50; ++attempt) {
        fs::rename(from, to, error);
        if (!error || error == std::errc::no_such_file_or_directory) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (error) throw fs::filesystem_error("rename", from, to, error);
}

void log_line(const FetchOptions& options, const std::string& line) {
    if (options.log) { options.log(line); }
}

std::string trim_etag(std::string etag) {
    if (etag.starts_with("W/")) { etag = etag.substr(2); }
    if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"') {
        etag = etag.substr(1, etag.size() - 2);
    }
    return etag;
}

std::string human(double bytes) {
    char text[64];
    if (bytes >= 1e9) {
        std::snprintf(text, sizeof(text), "%.2f GB", bytes / 1e9);
    } else {
        std::snprintf(text, sizeof(text), "%.1f MB", bytes / 1e6);
    }
    return text;
}

std::string curl_text(CURLcode code, long status) {
    return code == CURLE_OK ? "HTTP " + std::to_string(status) : curl_easy_strerror(code);
}

void curl_init() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

struct Easy {
    CURL* handle        = nullptr;
    curl_slist* headers = nullptr;

    Easy() {
        curl_init();
        handle = curl_easy_init();
        if (!handle) { throw std::runtime_error("curl_easy_init failed"); }
        curl_easy_setopt(handle, CURLOPT_USERAGENT, "ninferno-convert/1");
        curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(handle, CURLOPT_TCP_KEEPALIVE, 1L);
    }

    ~Easy() {
        if (headers) { curl_slist_free_all(headers); }
        if (handle) { curl_easy_cleanup(handle); }
    }

    Easy(const Easy&)            = delete;
    Easy& operator=(const Easy&) = delete;

    void header(const std::string& line) { headers = curl_slist_append(headers, line.c_str()); }
};

struct Response {
    long status = 0;
    std::map<std::string, std::string> headers; // lowercase names, last response of a chain
    std::string body;
    std::string redirect; // absolute Location of a 3xx
    std::string effective;
    CURLcode code = CURLE_OK;
};

std::size_t header_callback(char* data, std::size_t size, std::size_t count, void* user) {
    auto& response   = *static_cast<Response*>(user);
    const auto bytes = size * count;
    std::string line(data, bytes);
    if (line.starts_with("HTTP/")) { response.headers.clear(); }
    const auto colon = line.find(':');
    if (colon != std::string::npos) {
        auto value = line.substr(colon + 1);
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
            value.erase(0, 1);
        }
        while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) {
            value.pop_back();
        }
        response.headers[ascii_lower(line.substr(0, colon))] = value;
    }
    return bytes;
}

std::size_t body_callback(char* data, std::size_t size, std::size_t count, void* user) {
    static_cast<Response*>(user)->body.append(data, size * count);
    return size * count;
}

// One request: GET (or HEAD), following redirects or not.
Response request_once(const std::string& url, const std::vector<std::string>& headers, bool head,
                      bool follow) {
    Easy easy;
    Response response;
    for (const auto& line : headers) { easy.header(line); }
    curl_easy_setopt(easy.handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(easy.handle, CURLOPT_HTTPHEADER, easy.headers);
    curl_easy_setopt(easy.handle, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(easy.handle, CURLOPT_HEADERDATA, &response);
    curl_easy_setopt(easy.handle, CURLOPT_WRITEFUNCTION, body_callback);
    curl_easy_setopt(easy.handle, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(easy.handle, CURLOPT_FOLLOWLOCATION, follow ? 1L : 0L);
    curl_easy_setopt(easy.handle, CURLOPT_TIMEOUT, 120L);
    if (head) { curl_easy_setopt(easy.handle, CURLOPT_NOBODY, 1L); }
    response.code = curl_easy_perform(easy.handle);
    curl_easy_getinfo(easy.handle, CURLINFO_RESPONSE_CODE, &response.status);
    char* location = nullptr;
    curl_easy_getinfo(easy.handle, CURLINFO_REDIRECT_URL, &location);
    if (location) { response.redirect = location; }
    char* effective = nullptr;
    curl_easy_getinfo(easy.handle, CURLINFO_EFFECTIVE_URL, &effective);
    if (effective) { response.effective = effective; }
    return response;
}

// A failure a second try may not meet: a connection or TLS handshake cut short, an empty
// reply, a busy or failing server. An unreachable host (offline) and a timeout are not retried.
bool transient(const Response& response) {
    switch (response.code) {
    case CURLE_SSL_CONNECT_ERROR:
    case CURLE_SEND_ERROR:
    case CURLE_RECV_ERROR:
    case CURLE_GOT_NOTHING:
    case CURLE_PARTIAL_FILE:
    case CURLE_HTTP2:
    case CURLE_HTTP2_STREAM:
        return true;
    case CURLE_OK:
        return response.status == 429 || response.status >= 500;
    default:
        return false;
    }
}

// request_once, tried again after 1, 2 and 4 s while it fails transiently.
Response request(const std::string& url, const std::vector<std::string>& headers, bool head,
                 bool follow) {
    auto response = request_once(url, headers, head, follow);
    for (int retry = 0; retry < 3 && transient(response); ++retry) {
        std::this_thread::sleep_for(std::chrono::seconds(1 << retry));
        response = request_once(url, headers, head, follow);
    }
    return response;
}

std::string url_path(const std::string& path) {
    std::string out;
    for (const unsigned char c : path) {
        if (std::isalnum(c) || c == '/' || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            char code[4];
            std::snprintf(code, sizeof(code), "%%%02X", c);
            out += code;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Digests.

// sha256 of a file, or git's blob sha1 (sha1 of "blob <size>\0" + content).
std::string file_digest(const fs::path& path, bool git_sha1) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    EVP_DigestInit_ex(ctx.get(), git_sha1 ? EVP_sha1() : EVP_sha256(), nullptr);
    if (git_sha1) {
        const auto prefix = "blob " + std::to_string(fs::file_size(path));
        EVP_DigestUpdate(ctx.get(), prefix.data(), prefix.size() + 1); // with its NUL
    }
    std::ifstream stream(path, std::ios::binary);
    std::vector<char> buffer(16 << 20);
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(stream.gcount()));
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned size = 0;
    EVP_DigestFinal_ex(ctx.get(), digest, &size);
    return hex_encode(digest, size);
}

} // namespace

std::string file_sha256(const fs::path& path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) {
        throw std::runtime_error("cannot read " + path.string() + " to take its sha256");
    }
    return file_digest(path, false);
}

namespace {

// ---------------------------------------------------------------------------------------------
// Chunked, resumable, parallel transfer of one file.

class NoRanges : public std::runtime_error {
public:
    NoRanges() : std::runtime_error("the server ignores HTTP ranges") {}
};

class Transfer {
public:
    Transfer(const DownloadRequest& request, const FetchOptions& options)
        : request_(request), options_(options), url_(request.url),
          chunk_bytes_(std::max<std::uint64_t>(options.chunk_bytes, kMiB)) {
        name_     = request.label.empty() ? request.target.filename().string() : request.label;
        partial_  = request.target.string() + ".incomplete";
        progress_ = partial_ + ".progress";
    }

    FetchedFile run() {
        FetchedFile out;
        out.path = name_;
        fs::create_directories(request_.target.parent_path());
        const auto begin = Clock::now();
        if (request_.size == 0) {
            single_stream();
            out.streams = 1;
        } else {
            plan_chunks();
            try {
                out.streams = parallel();
            } catch (const NoRanges&) {
                log_line(options_, "  " + out.path + ": the server ignores ranges; one stream");
                std::error_code ignored;
                fs::remove(partial_, ignored);
                fs::remove(progress_, ignored);
                single_stream();
                out.streams = 1;
            }
        }
        out.seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        out.bytes   = fs::file_size(partial_);
        if (request_.size && out.bytes != request_.size) {
            throw std::runtime_error(out.path + ": received " + std::to_string(out.bytes) +
                                     " bytes, expected " + std::to_string(request_.size));
        }
        if (options_.verify && (!request_.sha256.empty() || !request_.git_sha1.empty())) {
            const bool git  = request_.sha256.empty();
            const auto want = ascii_lower(git ? request_.git_sha1 : request_.sha256);
            const auto got  = file_digest(partial_, git);
            if (got != want) {
                std::error_code ignored;
                fs::remove(partial_, ignored);
                fs::remove(progress_, ignored);
                throw std::runtime_error(out.path + ": " + (git ? "git sha1" : "sha256") +
                                         " mismatch: expected " + want + ", got " + got +
                                         " (the download was removed; fetch again)");
            }
            out.verified = true;
        }
        // Writable: FlushFileBuffers (Windows) flushes nothing through a read-only handle.
        if (const int fd = sys::open(partial_, O_RDWR); fd >= 0) {
            sys::fsync(fd);
            sys::close(fd);
        }
        rename_settled(partial_, request_.target);
        std::error_code ignored;
        fs::remove(progress_, ignored);
        return out;
    }

private:
    struct Chunk {
        std::uint64_t start = 0;
        std::uint64_t end   = 0;
        std::atomic<std::uint64_t> done{0};
        std::atomic<int> state{0}; // 0 pending, 1 active, 2 complete
        unsigned failures = 0;

        [[nodiscard]] std::uint64_t size() const noexcept { return end - start; }
    };

    struct Sink {
        Transfer* transfer = nullptr;
        Chunk* chunk       = nullptr;
        CURL* easy         = nullptr;
        int fd             = -1;
        bool accepted      = false;
        bool refused       = false; // a 200 for a range that does not start the file
    };

    void plan_chunks() {
        const auto size  = request_.size;
        const auto count = (size + chunk_bytes_ - 1) / chunk_bytes_;
        chunks_          = std::deque<Chunk>(count);
        for (std::uint64_t i = 0; i < count; ++i) {
            chunks_[i].start = i * chunk_bytes_;
            chunks_[i].end   = std::min(size, (i + 1) * chunk_bytes_);
        }
        // Resume: this downloader's per-chunk record, or the contiguous prefix another
        // downloader (a single-stream huggingface_hub) left behind.
        std::uint64_t resumed = 0;
        std::ifstream record(progress_);
        std::uint64_t record_size = 0, record_chunk = 0;
        if (fs::exists(partial_) && record && (record >> record_size >> record_chunk) &&
            record_size == size && record_chunk == chunk_bytes_) {
            for (auto& c : chunks_) {
                std::uint64_t done = 0;
                if (!(record >> done)) { break; }
                c.done = std::min(done, c.size());
            }
        } else if (fs::exists(partial_)) {
            const auto prefix = std::min<std::uint64_t>(fs::file_size(partial_), size);
            for (auto& c : chunks_) {
                c.done = prefix > c.start ? std::min(prefix, c.end) - c.start : 0;
            }
        }
        for (auto& c : chunks_) {
            resumed += c.done;
            if (c.done == c.size()) { c.state = 2; }
        }
        if (resumed) {
            log_line(options_, "  resuming " + name_ + " at " +
                                   human(static_cast<double>(resumed)) + " of " +
                                   human(static_cast<double>(size)));
        }
    }

    void save_progress() {
        std::lock_guard lock(progress_mutex_);
        const auto temporary = progress_ + ".tmp";
        {
            std::ofstream record(temporary);
            record << request_.size << ' ' << chunk_bytes_ << '\n';
            for (const auto& c : chunks_) { record << c.done.load() << '\n'; }
        }
        rename_settled(temporary, progress_);
    }

    static std::size_t write_range(char* data, std::size_t size, std::size_t count, void* user) {
        auto& sink       = *static_cast<Sink*>(user);
        const auto bytes = size * count;
        auto& chunk      = *sink.chunk;
        if (!sink.accepted) {
            long status = 0;
            curl_easy_getinfo(sink.easy, CURLINFO_RESPONSE_CODE, &status);
            const bool whole = status == 200 && chunk.start + chunk.done == 0;
            if (status != 206 && !whole) {
                sink.refused = status == 200;
                return 0;
            }
            sink.accepted = true;
        }
        const auto offset = chunk.start + chunk.done;
        const auto take   = std::min<std::uint64_t>(bytes, chunk.size() - chunk.done);
        std::size_t written = 0;
        while (written < take) {
            const auto n = sys::pwrite(sink.fd, data + written, take - written,
                                       static_cast<std::int64_t>(offset + written));
            if (n <= 0) { return 0; }
            written += static_cast<std::size_t>(n);
        }
        chunk.done += take;
        sink.transfer->received_ += take;
        return bytes; // the tail of a whole-file 200 past this chunk is dropped
    }

    std::string current_url() {
        std::lock_guard lock(url_mutex_);
        return url_;
    }

    void refresh_url(const std::string& stale) {
        std::lock_guard lock(url_mutex_);
        if (url_ != stale || !request_.refresh) { return; }
        url_ = request_.refresh();
        log_line(options_, "  " + name_ + ": download URL renewed");
    }

    // The rest of one chunk; returns when it completes, throws after too many failures.
    void fetch_chunk(Chunk& chunk, int fd) {
        while (chunk.done < chunk.size() && !failed_) {
            const auto url = current_url();
            Easy easy;
            Sink sink{this, &chunk, easy.handle, fd};
            const auto range =
                std::to_string(chunk.start + chunk.done) + "-" + std::to_string(chunk.end - 1);
            // Credentials go to the hub URL itself, never to a CDN it redirected to.
            if (url == request_.url) {
                for (const auto& line : request_.headers) { easy.header(line); }
            }
            curl_easy_setopt(easy.handle, CURLOPT_URL, url.c_str());
            curl_easy_setopt(easy.handle, CURLOPT_HTTPHEADER, easy.headers);
            curl_easy_setopt(easy.handle, CURLOPT_RANGE, range.c_str());
            curl_easy_setopt(easy.handle, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(easy.handle, CURLOPT_WRITEFUNCTION, write_range);
            curl_easy_setopt(easy.handle, CURLOPT_WRITEDATA, &sink);
            curl_easy_setopt(easy.handle, CURLOPT_LOW_SPEED_LIMIT,
                             static_cast<long>(options_.min_speed));
            curl_easy_setopt(easy.handle, CURLOPT_LOW_SPEED_TIME,
                             static_cast<long>(options_.stall_seconds));
            const auto code = curl_easy_perform(easy.handle);
            long status     = 0;
            curl_easy_getinfo(easy.handle, CURLINFO_RESPONSE_CODE, &status);
            if (chunk.done == chunk.size()) { return; }
            if (sink.refused) { throw NoRanges(); }
            if (status == 401 || status == 403 || status == 410) { refresh_url(url); }
            if (code == CURLE_OPERATION_TIMEDOUT) { ++stalls_; }
            if (++chunk.failures > options_.retries) {
                throw std::runtime_error(name_ + ": bytes " +
                                         range + " failed " + std::to_string(chunk.failures) +
                                         " times (last: " + curl_text(code, status) + ")");
            }
            log_line(options_, "  " + name_ + ": bytes " + range +
                                   " interrupted (" + curl_text(code, status) + "), resuming");
            std::this_thread::sleep_for(std::chrono::milliseconds(200 * chunk.failures));
        }
    }

    Chunk* claim() {
        std::lock_guard lock(claim_mutex_);
        for (auto& c : chunks_) {
            int pending = 0;
            if (c.state.compare_exchange_strong(pending, 1)) { return &c; }
        }
        return nullptr;
    }

    bool all_complete() const {
        return std::all_of(chunks_.begin(), chunks_.end(),
                           [](const Chunk& c) { return c.state == 2; });
    }

    unsigned parallel() {
        const int fd = sys::open(partial_, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) { throw std::runtime_error("cannot open " + partial_); }
        if (sys::truncate(fd, static_cast<std::int64_t>(request_.size)) != 0) {
            sys::close(fd);
            throw std::runtime_error("cannot size " + partial_);
        }
        const bool adaptive = options_.streams == 0;
        const unsigned most = adaptive ? std::max(1U, options_.max_streams) : options_.streams;
        const auto pending  = static_cast<unsigned>(std::count_if(
            chunks_.begin(), chunks_.end(), [](const Chunk& c) { return c.state != 2; }));
        const unsigned workers = std::max(1U, std::min(most, pending));
        target_                = adaptive ? 1U : workers;
        std::vector<std::thread> threads;
        std::exception_ptr error;
        std::mutex error_mutex;
        for (unsigned id = 0; id < workers; ++id) {
            threads.emplace_back([&, id] {
                try {
                    while (!failed_) {
                        if (id >= target_) {
                            if (all_complete()) { return; }
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                            continue;
                        }
                        Chunk* chunk = claim();
                        if (!chunk) { return; }
                        fetch_chunk(*chunk, fd);
                        if (chunk->done == chunk->size()) { chunk->state = 2; }
                    }
                } catch (...) {
                    std::lock_guard lock(error_mutex);
                    if (!error) { error = std::current_exception(); }
                    failed_ = true;
                }
            });
        }
        // Throughput-driven ramp: double the streams while each doubling gains 15 %.
        const auto window = std::chrono::milliseconds(2500);
        double best_rate  = 0.0;
        unsigned best     = target_;
        bool ramping      = adaptive && workers > 1;
        auto mark         = Clock::now();
        auto mark_bytes   = received_.load();
        auto last_save    = Clock::now();
        while (!failed_ && !all_complete()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const auto now = Clock::now();
            if (now - last_save > std::chrono::seconds(2)) {
                save_progress();
                last_save = now;
            }
            if (!ramping || now - mark < window) { continue; }
            const double rate = static_cast<double>(received_.load() - mark_bytes) /
                                std::chrono::duration<double>(now - mark).count();
            log_line(options_, "  " + name_ + ": " +
                                   std::to_string(target_.load()) + " stream(s) " + human(rate) +
                                   "/s");
            if (rate > best_rate * 1.15 && target_ < workers) {
                best_rate = rate;
                best      = target_;
                target_   = std::min(workers, target_ * 2);
            } else {
                if (rate > best_rate) { best = target_; }
                target_ = best;
                ramping = false;
            }
            mark       = Clock::now();
            mark_bytes = received_.load();
        }
        for (auto& thread : threads) { thread.join(); }
        sys::close(fd);
        if (error) {
            save_progress();
            std::rethrow_exception(error);
        }
        return target_;
    }

    // Unknown size (or a server without ranges): one stream, appended to what a previous
    // attempt wrote when the server honours an open range.
    struct StreamSink {
        CURL* easy = nullptr;
        std::FILE* file = nullptr;
        const std::string* path = nullptr;
        bool opened = false;
        bool append = false;
    };

    static std::size_t write_stream(char* data, std::size_t size, std::size_t count, void* user) {
        auto& sink = *static_cast<StreamSink*>(user);
        if (!sink.opened) {
            long status = 0;
            curl_easy_getinfo(sink.easy, CURLINFO_RESPONSE_CODE, &status);
            if (status != 200 && status != 206) { return 0; }
            // A 200 restarts the file; a 206 continues it.
            sink.file   = std::fopen(sink.path->c_str(), status == 206 && sink.append ? "ab" : "wb");
            sink.opened = true;
            if (!sink.file) { return 0; }
        }
        return std::fwrite(data, 1, size * count, sink.file) == size * count ? size * count : 0;
    }

    void single_stream() {
        const auto url = current_url();
        for (unsigned attempt = 0;; ++attempt) {
            const auto have = fs::exists(partial_) ? fs::file_size(partial_) : 0;
            Easy easy;
            StreamSink sink{easy.handle, nullptr, &partial_, false, have > 0};
            if (url == request_.url) {
                for (const auto& line : request_.headers) { easy.header(line); }
            }
            const auto range = std::to_string(have) + "-";
            curl_easy_setopt(easy.handle, CURLOPT_URL, url.c_str());
            curl_easy_setopt(easy.handle, CURLOPT_HTTPHEADER, easy.headers);
            if (have) { curl_easy_setopt(easy.handle, CURLOPT_RANGE, range.c_str()); }
            curl_easy_setopt(easy.handle, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(easy.handle, CURLOPT_WRITEFUNCTION, write_stream);
            curl_easy_setopt(easy.handle, CURLOPT_WRITEDATA, &sink);
            curl_easy_setopt(easy.handle, CURLOPT_LOW_SPEED_LIMIT,
                             static_cast<long>(options_.min_speed));
            curl_easy_setopt(easy.handle, CURLOPT_LOW_SPEED_TIME,
                             static_cast<long>(options_.stall_seconds));
            const auto code = curl_easy_perform(easy.handle);
            long status     = 0;
            curl_easy_getinfo(easy.handle, CURLINFO_RESPONSE_CODE, &status);
            if (sink.file) { std::fclose(sink.file); }
            if (code == CURLE_OK && (status == 200 || status == 206)) { return; }
            if (attempt >= options_.retries) {
                throw std::runtime_error(name_ +
                                         ": download failed (" + curl_text(code, status) + ")");
            }
            log_line(options_, "  " + name_ + ": interrupted (" +
                                   curl_text(code, status) + "), resuming");
            std::this_thread::sleep_for(std::chrono::milliseconds(200 * (attempt + 1)));
        }
    }

    const DownloadRequest& request_;
    const FetchOptions& options_;
    std::string url_;
    std::uint64_t chunk_bytes_;
    std::mutex url_mutex_, claim_mutex_, progress_mutex_;
    std::string name_, partial_, progress_;
    std::deque<Chunk> chunks_;
    std::atomic<unsigned> target_{1};
    std::atomic<std::uint64_t> received_{0};
    std::atomic<bool> failed_{false};
    std::atomic<unsigned> stalls_{0};
};

// ---------------------------------------------------------------------------------------------
// Hubs.

struct RemoteFile {
    std::string path;
    std::uint64_t size = 0;
    std::string sha256;   // LFS files
    std::string git_sha1; // small files kept in git
};

struct Listing {
    std::string commit;
    std::vector<RemoteFile> files;
};

// Where to download one file from, and what to check it against.
struct Resolved {
    std::string url;
    std::uint64_t size = 0;
    std::string sha256;
    std::string git_sha1;
    bool authorized = false; // the URL takes the hub's credentials
};

class Hub {
public:
    virtual ~Hub()                                                            = default;
    [[nodiscard]] virtual Listing list(const std::string& repo, const std::string& revision) = 0;
    [[nodiscard]] virtual Resolved resolve(const std::string& repo, const std::string& commit,
                                           const RemoteFile& file)                           = 0;
    [[nodiscard]] virtual std::vector<std::string> credentials() const                       = 0;
    [[nodiscard]] virtual fs::path cache_dir(const fs::path& root, const std::string& repo) const = 0;
};

std::runtime_error hub_error(const std::string& what, const Response& response,
                             const std::string& hint) {
    return std::runtime_error(what + ": " + curl_text(response.code, response.status) +
                              (hint.empty() ? "" : " (" + hint + ")"));
}

class HuggingFace final : public Hub {
public:
    explicit HuggingFace(std::string endpoint) : endpoint_(std::move(endpoint)) {
        token_ = sys::env("HF_TOKEN");
        if (token_.empty()) { token_ = sys::env("HUGGING_FACE_HUB_TOKEN"); }
    }

    Listing list(const std::string& repo, const std::string& revision) override {
        const auto url = endpoint_ + "/api/models/" + repo + "/revision/" + url_path(revision) +
                         "?blobs=true";
        const auto response = request(url, credentials(), false, true);
        if (response.code != CURLE_OK || response.status != 200) {
            throw hub_error(
                "listing " + repo + "@" + revision, response,
                response.status == 401 || response.status == 403 ? listing_hint(repo)
                : response.status == 404
                    ? "no such repository or revision (without a token a private repository "
                      "answers 404 too: set HF_TOKEN)"
                    : "");
        }
        const auto body = Json::parse(response.body);
        Listing out;
        out.commit = body.value("sha", revision);
        for (const auto& sibling : body.value("siblings", Json::array())) {
            RemoteFile file;
            file.path = sibling.value("rfilename", std::string());
            file.size = sibling.value("size", std::uint64_t{0});
            if (const auto it = sibling.find("lfs"); it != sibling.end() && it->is_object()) {
                file.sha256 = it->value("sha256", it->value("oid", std::string()));
                file.size   = it->value("size", file.size);
            } else {
                file.git_sha1 = sibling.value("blobId", std::string());
            }
            if (!file.path.empty()) { out.files.push_back(std::move(file)); }
        }
        return out;
    }

    Resolved resolve(const std::string& repo, const std::string& commit,
                     const RemoteFile& file) override {
        Resolved out;
        out.url      = endpoint_ + "/" + repo + "/resolve/" + commit + "/" + url_path(file.path);
        out.size     = file.size;
        out.sha256   = file.sha256;
        out.git_sha1 = file.git_sha1;
        // The hub answers an LFS file with a redirect to its CDN, naming the content sha256 in
        // X-Linked-Etag (the CDN's own ETag is something else); a small file comes directly.
        for (int hop = 0; hop < 5; ++hop) {
            const auto head = request(out.url, credentials(), true, false);
            if (head.code != CURLE_OK) { throw hub_error("resolving " + file.path, head, ""); }
            if (head.status >= 300 && head.status < 400 && !head.redirect.empty()) {
                if (const auto it = head.headers.find("x-linked-etag"); it != head.headers.end()) {
                    // 64 hex digits: an LFS file's sha256; 40: a git-stored file's blob sha1
                    // (Xet-backed repositories redirect those too).
                    const auto linked = ascii_lower(trim_etag(it->second));
                    auto& expected    = linked.size() == 40 ? out.git_sha1 : out.sha256;
                    if (!expected.empty() && linked != ascii_lower(expected)) {
                        throw std::runtime_error(file.path + ": X-Linked-Etag " + linked +
                                                 " disagrees with the listing's " +
                                                 (linked.size() == 40 ? "blob id " : "sha256 ") +
                                                 expected);
                    }
                    expected = linked;
                    if (const auto size = head.headers.find("x-linked-size");
                        size != head.headers.end()) {
                        out.size = std::stoull(size->second);
                    }
                    out.url        = head.redirect;
                    out.authorized = false;
                    return out;
                }
                out.url = head.redirect; // a renamed repository: follow on the hub
                continue;
            }
            if (head.status == 200) {
                out.authorized = true;
                if (!out.size) {
                    if (const auto it = head.headers.find("content-length");
                        it != head.headers.end()) {
                        out.size = std::stoull(it->second);
                    }
                }
                return out;
            }
            throw hub_error("resolving " + file.path, head,
                            head.status == 401 || head.status == 403 ? access_hint(repo) : "");
        }
        throw std::runtime_error(file.path + ": too many redirects");
    }

    std::vector<std::string> credentials() const override {
        if (token_.empty()) { return {}; }
        return {"Authorization: Bearer " + token_};
    }

    fs::path cache_dir(const fs::path& root, const std::string& repo) const override {
        return root / hub_repo_directory(repo);
    }

private:
    // The hub answers a listing 401 for a repository that does not exist and for a private one
    // alike (a gated repository lists; its files answer 401).
    std::string listing_hint(const std::string& repo) const {
        return token_.empty()
                   ? "no such repository, or a private one: check the id at " + endpoint_ + "/" +
                         repo + "; a private one needs HF_TOKEN"
                   : "no such repository, or one the token in HF_TOKEN cannot see: check the id "
                     "at " + endpoint_ + "/" + repo;
    }

    // A gated repository answers 401/403 for its files until the account behind the token has
    // accepted its terms on the model page.
    std::string access_hint(const std::string& repo) const {
        return token_.empty()
                   ? "gated or private: accept its terms at " + endpoint_ + "/" + repo +
                         " and set HF_TOKEN to a read token of that account"
                   : "the token in HF_TOKEN has no access: accept the terms at " + endpoint_ +
                         "/" + repo + " with the token's account, or use a token that has access";
    }

    std::string endpoint_;
    std::string token_;
};

class ModelScope final : public Hub {
public:
    explicit ModelScope(std::string endpoint) : endpoint_(std::move(endpoint)) {
        token_ = sys::env("MODELSCOPE_API_TOKEN");
    }

    Listing list(const std::string& repo, const std::string& revision) override {
        const auto url = endpoint_ + "/api/v1/models/" + repo + "/repo/files?Revision=" +
                         url_path(revision) + "&Recursive=true";
        const auto response = request(url, credentials(), false, true);
        if (response.code != CURLE_OK || response.status != 200) {
            throw hub_error("listing " + repo + "@" + revision + " on ModelScope", response,
                            response.status == 404 ? "no such repository or revision" : "");
        }
        const auto body = Json::parse(response.body);
        Listing out;
        out.commit = revision;
        const auto& files = body.at("Data").at("Files");
        for (const auto& item : files) {
            if (item.value("Type", std::string("blob")) != "blob") { continue; }
            RemoteFile file;
            file.path   = item.value("Path", item.value("Name", std::string()));
            file.size   = item.value("Size", std::uint64_t{0});
            file.sha256 = item.value("Sha256", std::string());
            if (item.contains("Revision") && item["Revision"].is_string()) {
                out.commit = item["Revision"].get<std::string>();
            }
            if (!file.path.empty()) { out.files.push_back(std::move(file)); }
        }
        return out;
    }

    Resolved resolve(const std::string& repo, const std::string& commit,
                     const RemoteFile& file) override {
        Resolved out;
        out.url        = endpoint_ + "/models/" + repo + "/resolve/" + url_path(commit) + "/" +
                         url_path(file.path);
        out.size       = file.size;
        out.sha256     = file.sha256;
        out.authorized = true;
        return out;
    }

    std::vector<std::string> credentials() const override {
        if (token_.empty()) { return {}; }
        return {"Authorization: Bearer " + token_};
    }

    fs::path cache_dir(const fs::path& root, const std::string& repo) const override {
        return root / "modelscope" / hub_repo_directory(repo);
    }

private:
    std::string endpoint_;
    std::string token_;
};

} // namespace

std::string hub_repo_directory(std::string_view repo) {
    std::string name = "models--" + std::string(repo);
    if (const auto slash = name.find('/'); slash != std::string::npos) {
        name.replace(slash, 1, "--");
    }
    return name;
}

std::string hub_repo_of_directory(std::string_view directory) {
    if (!directory.starts_with("models--")) { return {}; }
    std::string repo(directory.substr(8));
    if (const auto separator = repo.find("--"); separator != std::string::npos) {
        repo.replace(separator, 2, "/");
    }
    return repo;
}

fs::path default_hub_cache() {
    if (const auto hub = sys::env("HF_HUB_CACHE"); !hub.empty()) { return hub; }
    if (const auto home = sys::env("HF_HOME"); !home.empty()) { return fs::path(home) / "hub"; }
    if (const auto xdg = sys::env("XDG_CACHE_HOME"); !xdg.empty()) {
        return fs::path(xdg) / "huggingface" / "hub";
    }
    const auto home = sys::home_dir();
    return (home.empty() ? fs::temp_directory_path() : home) / ".cache" / "huggingface" / "hub";
}

namespace {

bool hub_offline() {
    const auto value = ascii_lower(sys::env("HF_HUB_OFFLINE"));
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

// ---------------------------------------------------------------------------------------------
// What to download.

#ifdef _WIN32
// fnmatch(pattern, path, 0): * and ? match any characters ('/' too), [...] a set ([!...] or
// [^...] its complement, a-z ranges), and a backslash quotes the next character.
bool fnmatch_like(const char* p, const char* s) {
    for (; *p; ++p, ++s) {
        if (*p == '*') {
            while (p[1] == '*') { ++p; }
            for (const char* rest = s;; ++rest) {
                if (fnmatch_like(p + 1, rest)) { return true; }
                if (!*rest) { return false; }
            }
        }
        if (!*s) { return false; }
        if (*p == '?') { continue; }
        if (*p == '[') {
            const char* q     = p + 1;
            const bool invert = *q == '!' || *q == '^';
            if (invert) { ++q; }
            bool hit = false;
            for (bool first = true; *q && (first || *q != ']'); first = false, ++q) {
                if (q[1] == '-' && q[2] && q[2] != ']') {
                    hit = hit || (*s >= q[0] && *s <= q[2]);
                    q += 2;
                } else {
                    hit = hit || *s == *q;
                }
            }
            if (*q != ']') { // no closing bracket: '[' is itself
                if (*s != '[') { return false; }
                continue;
            }
            if (hit == invert) { return false; }
            p = q;
            continue;
        }
        if (*p == '\\' && p[1]) { ++p; }
        if (*p != *s) { return false; }
    }
    return !*s;
}
#endif

bool glob(const std::string& pattern, const std::string& path) {
#ifdef _WIN32
    return fnmatch_like(pattern.c_str(), path.c_str());
#else
    return fnmatch(pattern.c_str(), path.c_str(), 0) == 0;
#endif
}

// The split stem of a GGUF file name ("x-00001-of-00003.gguf" -> "x"), or the name.
std::string gguf_group(const std::string& path) {
    static const std::regex split(R"((.*)-\d{5}-of-\d{5}\.gguf)", std::regex::icase);
    std::smatch match;
    return std::regex_match(path, match, split) ? match[1].str() : path;
}

// Quantization preference when the user names none and no GPU picks one (ninferno-serve takes
// the largest that fits its GPU before it fetches: serve/plan_offline.h; --prepare-only and
// ninferno-convert come here): lossless into the engine first (NVFP4, MXFP4), then the
// near-lossless and larger types the engine serves (8 bits or fewer); BF16 and F16, which it does
// not, last.
int gguf_rank(const std::string& name) {
    static const std::vector<std::string> order = {
        "nvfp4",  "mxfp4", "q8_0", "q6_k",   "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "iq4_xs",
        "iq4_nl", "q4_0",  "q4_1", "q3_k_m", "q3_k_l", "q3_k_s", "q2_k",   "bf16",   "f16"};
    const auto text = ascii_lower(name);
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (text.find(order[i]) != std::string::npos) { return static_cast<int>(i); }
    }
    return static_cast<int>(order.size());
}

constexpr const char* kFrontend[] = {"config.json",           "tokenizer.json",
                                     "tokenizer_config.json", "chat_template.jinja",
                                     "generation_config.json", "preprocessor_config.json",
                                     "video_preprocessor_config.json"};

// Transformers names its weights model.safetensors or model-NNNNN-of-NNNNN.safetensors, and a
// checkpoint may keep a part in its own model_<part>.safetensors that the index names (Qwen3.5's
// MTP layers: model_mtp.safetensors); when they are there, other top-level files are a second copy
// in another layout (Mistral repositories also ship consolidated.safetensors, as large as the
// model). The fetch takes those files (choose_files) and the listing counts them (list_quants).
bool transformers_weight_file(const std::string& path) {
    return path == "model.safetensors" ||
           ((path.starts_with("model-") || path.starts_with("model_")) &&
            path.ends_with(".safetensors"));
}

std::vector<RemoteFile> choose_files(const Listing& listing, const FetchOptions& options,
                                     const std::string& repo) {
    std::vector<RemoteFile> out;
    if (!options.include.empty()) {
        for (const auto& file : listing.files) {
            for (const auto& pattern : options.include) {
                if (glob(pattern, file.path)) {
                    out.push_back(file);
                    break;
                }
            }
        }
        if (out.empty()) {
            std::string patterns;
            for (const auto& pattern : options.include) {
                patterns += (patterns.empty() ? "" : ", ") + pattern;
            }
            throw std::runtime_error(repo + ": no file matches --include " + patterns + " (of " +
                                     std::to_string(listing.files.size()) + " files)");
        }
        return out;
    }
    std::vector<const RemoteFile*> safetensors, ggufs;
    for (const auto& file : listing.files) {
        const auto name = ascii_lower(file.path);
        if (name.ends_with(".safetensors")) { safetensors.push_back(&file); }
        if (name.ends_with(".gguf") && !gguf_auxiliary_file(name)) { ggufs.push_back(&file); }
    }
    const auto add_frontend = [&] {
        for (const auto& file : listing.files) {
            if (std::find(std::begin(kFrontend), std::end(kFrontend), file.path) !=
                std::end(kFrontend)) {
                out.push_back(file);
            }
        }
    };
    if (!safetensors.empty() && options.quant.empty()) {
        add_frontend();
        // The top-level weights (subdirectories hold originals and other formats).
        for (const auto& file : listing.files) {
            if (file.path == "model.safetensors.index.json") { out.push_back(file); }
        }
        // Transformers-named weights, when there are any (transformers_weight_file).
        const bool named = std::any_of(safetensors.begin(), safetensors.end(), [&](const auto* f) {
            return transformers_weight_file(f->path);
        });
        for (const auto* file : safetensors) {
            if (file->path.find('/') == std::string::npos &&
                (!named || transformers_weight_file(file->path))) {
                out.push_back(*file);
            }
        }
        return out;
    }
    // A --quant that is an MTP sidecar's whole name (a drafter: mtp-<model>-Q4_0) takes that file;
    // a sidecar is never the model otherwise, and a projector or an importance matrix never is.
    if (!options.quant.empty()) {
        const auto want = ascii_lower(options.quant) + ".gguf";
        std::vector<const RemoteFile*> named;
        for (const auto& file : listing.files) {
            const auto name = ascii_lower(std::filesystem::path(file.path).filename().string());
            if (name == want && gguf_auxiliary_file(name) && name.find("mmproj") == std::string::npos &&
                name.find("imatrix") == std::string::npos) {
                named.push_back(&file);
            }
        }
        if (named.size() > 1) {
            std::string paths;
            for (const auto* file : named) { paths += "\n  " + file->path; }
            throw std::runtime_error(repo + ": --quant " + options.quant + " names " +
                                     std::to_string(named.size()) + " files:" + paths);
        }
        if (named.size() == 1) {
            out.push_back(*named.front());
            return out;
        }
    }
    if (ggufs.empty()) {
        throw std::runtime_error(repo + " holds neither safetensors weights nor a GGUF model" +
                                 (options.quant.empty() ? "" : " (for --quant " + options.quant + ")"));
    }
    std::map<std::string, std::vector<const RemoteFile*>> groups;
    for (const auto* file : ggufs) { groups[gguf_group(file->path)].push_back(file); }
    std::string chosen;
    if (const int width = quant_width_tag(options.quant); width > 0) {
        // A width ("4bit"): the largest GGUF model of that width, the most faithful of them.
        std::uint64_t largest = 0;
        for (const auto& [group, files] : groups) {
            const auto name = std::filesystem::path(files.front()->path).filename().string();
            if (quant_label_bits(gguf_quant_label(name)) != width) { continue; }
            std::uint64_t bytes = 0;
            for (const auto* file : files) { bytes += file->size; }
            if (bytes > largest) {
                largest = bytes;
                chosen  = group;
            }
        }
        if (chosen.empty()) {
            std::string names;
            for (const auto& [group, files] : groups) { names += "\n  " + group; }
            throw std::runtime_error(repo + ": --quant " + options.quant + " matches no " +
                                     std::to_string(width) + "-bit GGUF model; available:" + names);
        }
    } else if (!options.quant.empty()) {
        std::vector<std::string> matches;
        const auto want = ascii_lower(options.quant);
        for (const auto& [group, files] : groups) {
            if (ascii_lower(group).find(want) != std::string::npos) { matches.push_back(group); }
        }
        // "Q4_K" also matches "Q4_K_M" and "UD-Q4_K_XL"; an exact label wins.
        if (matches.size() > 1) {
            std::vector<std::string> exact;
            for (const auto& group : matches) {
                const auto text = ascii_lower(group);
                const auto at   = text.find(want);
                const auto next = at + want.size();
                if (next == text.size() || text.compare(next, 5, ".gguf") == 0) {
                    exact.push_back(group);
                }
            }
            if (exact.size() == 1) { matches = exact; }
        }
        if (matches.size() != 1) {
            std::string names;
            for (const auto& [group, files] : groups) { names += "\n  " + group; }
            throw std::runtime_error(repo + ": --quant " + options.quant + " matches " +
                                     std::to_string(matches.size()) + " GGUF models; available:" +
                                     names);
        }
        chosen = matches.front();
    } else {
        int best = 1 << 30;
        for (const auto& [group, files] : groups) {
            if (const auto rank = gguf_rank(group); rank < best) {
                best   = rank;
                chosen = group;
            }
        }
    }
    add_frontend();
    for (const auto* file : groups[chosen]) { out.push_back(*file); }
    return out;
}

// A relative symlink from snapshots/<commit>/<path> to blobs/<digest>.
void link_snapshot(const fs::path& repo_dir, const std::string& commit, const std::string& path,
                   const std::string& blob) {
    const auto link = repo_dir / "snapshots" / commit / path;
    fs::create_directories(link.parent_path());
    std::string up = "../..";
    for (const auto c : path) {
        if (c == '/') { up += "/.."; }
    }
    const auto target = fs::path(up) / "blobs" / blob;
    std::error_code error;
    if (!fs::is_symlink(link, error) && fs::exists(link, error)) { return; }
    // Made beside it and renamed over it: another process linking the same file at once
    // (two quantizations of one repository share its small files) removed and recreated it
    // between the two calls here ("cannot create symlink: File exists"). The staged name is
    // random as well: processes in containers sharing a cache often have the same pid.
    const fs::path staged = link.string() + ".link-" + std::to_string(sys::pid()) + "-" +
                            std::to_string(std::random_device{}());
    try {
#ifdef _WIN32
        // Symbolic links need Developer Mode or administrator rights on Windows: a hard link to
        // the blob reads the same (huggingface_hub copies there instead).
        fs::create_symlink(target, staged, error);
        if (error) { fs::create_hard_link((link.parent_path() / target).lexically_normal(), staged); }
#else
        fs::create_symlink(target, staged);
#endif
        rename_settled(staged, link);
    } catch (...) {
        fs::remove(staged, error);
        throw;
    }
}

// The lock a download takes: huggingface_hub's for a blob of its cache layout
// (<root>/<repo folder>/blobs/<etag> -> <root>/.locks/<repo folder>/<etag>.lock), so ninferno and
// a Python downloader (huggingface_hub, vLLM, transformers) sharing one cache take turns too;
// both flock the file. Any other target is locked beside itself.
fs::path download_lock_path(const fs::path& target) {
    const fs::path blobs = target.parent_path();
    if (blobs.filename() == "blobs" && blobs.parent_path().has_parent_path()) {
        const fs::path repo = blobs.parent_path();
        return repo.parent_path() / ".locks" / repo.filename() /
               (target.filename().string() + ".lock");
    }
    return target.string() + ".lock";
}

} // namespace

FetchedFile download_file(const DownloadRequest& request, const FetchOptions& options) {
    // Two processes fetching one file into a shared cache (two quantizations of a repository
    // share its small files; ninferno-ui beside the CLI) take turns on a lock beside it, as
    // huggingface_hub does: without it the second renamed or removed the first one's .incomplete
    // under it ("cannot get file size: No such file or directory [...incomplete]"). The second
    // then finds the file done. A file system without flock fetches unlocked, as before.
    fs::create_directories(request.target.parent_path());
    const fs::path lock_path = download_lock_path(request.target);
    std::error_code lock_dir_error;
    fs::create_directories(lock_path.parent_path(), lock_dir_error);
    const int lock = sys::open(lock_path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (lock >= 0 && sys::lock(lock, false) != 0 && errno == EWOULDBLOCK) {
        const std::string name =
            request.label.empty() ? request.target.filename().string() : request.label;
        log_line(options, "  " + name + ": another process is fetching it; waiting");
        while (sys::lock(lock, true) != 0 && errno == EINTR) {}
    }
    struct Unlock {
        int fd;
        ~Unlock() {
            if (fd >= 0) { sys::close(fd); } // closing releases the lock
        }
    } unlock{lock};
    std::error_code error;
    if (fs::exists(request.target, error) &&
        (request.size == 0 || fs::file_size(request.target, error) == request.size)) {
        return {request.label.empty() ? request.target.filename().string() : request.label,
                fs::file_size(request.target, error), 0.0, 0, true, false};
    }
    Transfer transfer(request, options);
    return transfer.run();
}

std::optional<std::string> gguf_base_repository(const fs::path& gguf) {
    const int fd = sys::open(gguf, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return std::nullopt; }
    const auto size = static_cast<std::size_t>(std::max<std::int64_t>(sys::file_size(fd), 0));
    void* base      = size ? sys::map(fd, size, sys::Map::ReadPrivate) : nullptr;
    sys::close(fd);
    if (base == nullptr) { return std::nullopt; }
    std::optional<std::string> out;
    try {
        const auto file = parse_gguf(static_cast<const std::byte*>(base), size, gguf.string());
        for (const char* key : {"general.base_model.0.repo_url", "general.source.url",
                                "general.source.huggingface.repository"}) {
            const auto* value = file.find(key);
            if (!value || !value->is_string()) { continue; }
            auto text = value->get<std::string>();
            for (const char* prefix : {"https://huggingface.co/", "http://huggingface.co/"}) {
                if (text.starts_with(prefix)) { text = text.substr(std::strlen(prefix)); }
            }
            if (std::count(text.begin(), text.end(), '/') == 1 &&
                text.find(':') == std::string::npos) {
                out = text;
                break;
            }
        }
    } catch (const std::exception&) {}
    sys::unmap(base, size);
    return out;
}

namespace {

// "org/name[@revision]": the repository and the revision (the options' when the spec names none).
} // namespace

std::pair<std::string, std::string> split_spec(const std::string& spec,
                                               const FetchOptions& options) {
    auto repo     = spec;
    auto revision = options.revision;
    if (const auto at = spec.find('@'); at != std::string::npos) {
        repo     = spec.substr(0, at);
        revision = spec.substr(at + 1);
    }
    if (std::count(repo.begin(), repo.end(), '/') != 1) {
        throw std::runtime_error("expected org/name[@revision] or a URL, got '" + spec + "'");
    }
    return {repo, revision};
}

namespace {

// The options' hub; an empty `revision` becomes the hub's default branch.
std::unique_ptr<Hub> make_hub(const FetchOptions& options, std::string& revision) {
    if (is_hugging_face_hub(options.hub)) {
        auto endpoint = options.endpoint.empty() ? sys::env("HF_ENDPOINT") : options.endpoint;
        if (revision.empty()) { revision = "main"; }
        return std::make_unique<HuggingFace>(endpoint.empty() ? "https://huggingface.co"
                                                              : endpoint);
    }
    if (is_modelscope_hub(options.hub)) {
        auto endpoint =
            options.endpoint.empty() ? sys::env("MODELSCOPE_ENDPOINT") : options.endpoint;
        if (revision.empty()) { revision = "master"; }
        return std::make_unique<ModelScope>(endpoint.empty() ? "https://modelscope.cn" : endpoint);
    }
    throw std::runtime_error("unknown hub '" + options.hub + "' (huggingface or modelscope)");
}

// Offline: the commit a snapshot of this revision already resolved to (a commit id names its
// snapshot directly). Throws, with why the listing failed, when the cache holds none.
std::string cached_commit(const fs::path& repo_dir, const std::string& repo,
                          const std::string& revision, const fs::path& root,
                          const std::exception& error) {
    const auto ref = repo_dir / "refs" / revision;
    std::string commit;
    if (fs::exists(ref)) {
        std::ifstream stream(ref);
        stream >> commit;
    } else if (revision.size() == 40 && fs::is_directory(repo_dir / "snapshots" / revision)) {
        commit = revision;
    } else {
        throw std::runtime_error(std::string(error.what()) + "; and " + repo + "@" + revision +
                                 " has no snapshot in the cache " + root.string());
    }
    // A ref (a branch, a tag) may have moved since it was cached: its snapshot is served only when
    // the user asked for the cache ($HF_HUB_OFFLINE), never because the hub could not be reached
    // (the same command would serve another model by the state of the network). A commit is its
    // own snapshot.
    if (!hub_offline() && revision != commit) {
        throw std::runtime_error(std::string(error.what()) + "; " + repo + "@" + revision +
                                 " was cached at " + commit +
                                 ", which the ref may have left: set HF_HUB_OFFLINE=1 to serve "
                                 "that snapshot, or name the commit (" + repo + "@" + commit + ")");
    }
    return commit;
}

// The GGUF file of `files` to convert: the first part of a split model; none: `snapshot`.
fs::path model_of(const fs::path& snapshot, const std::vector<RemoteFile>& files) {
    fs::path model = snapshot;
    for (const auto& file : files) {
        if (ascii_lower(file.path).ends_with(".gguf")) {
            const auto candidate = snapshot / file.path;
            if (model == snapshot || candidate < model) { model = candidate; }
        }
    }
    return model;
}

// What a cached snapshot holds for `options`, chosen from its complete files as a listing's are:
// --quant selects offline as it does online, and a snapshot without the file it names is refused.
fs::path cached_model(const fs::path& snapshot, const FetchOptions& options,
                      const std::string& repo) {
    Listing listing;
    for (const auto& entry : fs::recursive_directory_iterator(snapshot)) {
        // By the link's own name (fs::relative would resolve it into blobs/).
        if (entry.is_regular_file()) { // a link to a blob not yet downloaded is not
            listing.files.push_back(
                RemoteFile{.path = entry.path().lexically_relative(snapshot).generic_string()});
        }
    }
    try {
        return model_of(snapshot, choose_files(listing, options, repo));
    } catch (const std::exception& error) {
        // Without a selector, what the snapshot holds (small files the caller reads, or weights
        // not yet downloaded: the caller says so).
        if (options.quant.empty() || !options.include.empty()) { return snapshot; }
        throw std::runtime_error("the hub is unreachable and the cached snapshot " +
                                 snapshot.string() + " has no model for " + options.quant + ": " +
                                 error.what());
    }
}

// A revision pinned to a commit resolves to that commit. Hugging Face only: its listing names the
// commit the revision resolved to; a ModelScope listing's is the last file's own revision.
void check_pinned(const FetchOptions& options, const std::string& repo,
                  const std::string& revision, const Listing& listing) {
    if (!is_hugging_face_hub(options.hub)) { return; }
    const bool commit = revision.size() == 40 &&
                        std::all_of(revision.begin(), revision.end(),
                                    [](unsigned char c) { return std::isxdigit(c) != 0; });
    if (commit && ascii_lower(listing.commit) != ascii_lower(revision)) {
        throw std::runtime_error(repo + "@" + revision +
                                 ": the hub resolved the pinned commit to " + listing.commit);
    }
}

// The first `bytes` of `url` by an HTTP range request. `headers` (credentials) go to `url` only,
// never to where it redirects (a CDN); a server that ignores the range is cut off at `bytes`.
// `total`: the whole file's bytes as the answer gives them (Content-Range, or the Content-Length
// of a 200), 0 when it does not; a file shorter than `bytes` is returned whole.
std::string range_prefix(const std::string& url, const std::vector<std::string>& headers,
                         std::uint64_t bytes, std::uint64_t& total) {
    struct Sink {
        std::string data;
        std::uint64_t limit = 0;
    };
    const auto get = [&](const std::string& from, const std::vector<std::string>& lines,
                         bool follow, Response& response) {
        Easy easy;
        Sink sink{.limit = bytes};
        for (const auto& line : lines) { easy.header(line); }
        easy.header("Range: bytes=0-" + std::to_string(bytes - 1));
        curl_easy_setopt(easy.handle, CURLOPT_URL, from.c_str());
        curl_easy_setopt(easy.handle, CURLOPT_HTTPHEADER, easy.headers);
        curl_easy_setopt(easy.handle, CURLOPT_HEADERFUNCTION, header_callback);
        curl_easy_setopt(easy.handle, CURLOPT_HEADERDATA, &response);
        const auto write = +[](char* data, std::size_t size, std::size_t count,
                               void* user) -> std::size_t {
            auto& into       = *static_cast<Sink*>(user);
            const auto taken = std::min<std::uint64_t>(into.limit - into.data.size(), size * count);
            into.data.append(data, taken);
            return into.data.size() < into.limit ? size * count : 0; // full: stop the transfer
        };
        curl_easy_setopt(easy.handle, CURLOPT_WRITEFUNCTION, write);
        curl_easy_setopt(easy.handle, CURLOPT_WRITEDATA, &sink);
        curl_easy_setopt(easy.handle, CURLOPT_FOLLOWLOCATION, follow ? 1L : 0L);
        curl_easy_setopt(easy.handle, CURLOPT_TIMEOUT, 300L);
        response.code = curl_easy_perform(easy.handle);
        curl_easy_getinfo(easy.handle, CURLINFO_RESPONSE_CODE, &response.status);
        char* location = nullptr;
        curl_easy_getinfo(easy.handle, CURLINFO_REDIRECT_URL, &location);
        if (location) { response.redirect = location; }
        return sink;
    };
    Response response;
    auto sink = get(url, headers, false, response);
    if (response.status / 100 == 3 && !response.redirect.empty()) {
        const auto location = response.redirect;
        response            = Response{};
        sink                = get(location, {}, true, response);
    }
    total = 0;
    const auto header = [&](const char* name) {
        const auto it = response.headers.find(name);
        return it == response.headers.end() ? std::string() : it->second;
    };
    const auto number = [](const std::string& text) -> std::uint64_t {
        std::uint64_t value = 0;
        const auto* end     = text.data() + text.size();
        return std::from_chars(text.data(), end, value).ec == std::errc() ? value : 0;
    };
    if (const auto range = header("content-range"); range.find('/') != std::string::npos) {
        total = number(range.substr(range.find('/') + 1));
    } else if (response.status == 200) {
        total = number(header("content-length"));
    }
    // Enough bytes, or the whole of a shorter file.
    const bool full = sink.data.size() >= bytes || (total != 0 && sink.data.size() == total);
    if ((response.code != CURLE_OK && !(full && response.code == CURLE_WRITE_ERROR)) ||
        (response.status != 200 && response.status != 206) || !full) {
        throw hub_error("reading the first " + std::to_string(bytes) + " bytes of " + url,
                        response, full ? "" : "short response");
    }
    return sink.data;
}

} // namespace

bool snapshot_holds_model(const fs::path& snapshot, const FetchOptions& options,
                          const std::string& repo) {
    std::error_code ec;
    if (!fs::is_directory(snapshot, ec)) { return false; }
    Listing listing;
    for (const auto& entry : fs::recursive_directory_iterator(snapshot, ec)) {
        if (entry.is_regular_file(ec)) { // a link to a blob not yet downloaded is not
            listing.files.push_back(
                RemoteFile{.path = entry.path().lexically_relative(snapshot).generic_string()});
        }
    }
    std::vector<RemoteFile> chosen;
    try {
        chosen = choose_files(listing, options, repo);
    } catch (const std::exception&) {
        return false; // no model there for options.quant
    }
    bool weights = false;
    for (const auto& file : chosen) {
        const auto name = ascii_lower(file.path);
        if (name.ends_with(".gguf") || name.ends_with(".safetensors")) { weights = true; }
        // Every part of a split GGUF ("x-00001-of-00003.gguf"): a download stopped after one
        // part leaves the others unlinked, and the listing above sees only what is there.
        static const std::regex split(R"((.*)-\d{5}-of-(\d{5})\.gguf)", std::regex::icase);
        std::smatch match;
        if (std::regex_match(file.path, match, split)) {
            const int parts = std::stoi(match[2].str());
            for (int part = 1; part <= parts; ++part) {
                char suffix[32];
                std::snprintf(suffix, sizeof suffix, "-%05d-of-%05d.gguf", part, parts);
                if (!fs::is_regular_file(snapshot / (match[1].str() + suffix), ec)) { return false; }
            }
        }
        // Every shard a safetensors index names.
        if (name.ends_with("model.safetensors.index.json")) {
            try {
                std::ifstream in(snapshot / file.path);
                const auto index = Json::parse(in);
                for (const auto& [tensor, shard] : index.at("weight_map").items()) {
                    if (!fs::is_regular_file(snapshot / shard.get<std::string>(), ec)) { return false; }
                }
            } catch (const std::exception&) {
                return false;
            }
        }
    }
    return weights;
}


std::optional<std::string> current_commit(const std::string& spec, const FetchOptions& options) {
    if (hub_offline()) { return std::nullopt; }
    auto [repo, revision] = split_spec(spec, options);
    const auto hub        = make_hub(options, revision);
    return hub->list(repo, revision).commit;
}

FetchResult fetch_model(const std::string& spec, const FetchOptions& options) {
    const auto root = options.cache.empty() ? default_hub_cache() : options.cache;
    FetchResult out;
    if (spec.starts_with("http://") || spec.starts_with("https://")) {
        // One file by URL: cached under urls/<sha256 of the URL>/<file name>.
        auto name = spec.substr(spec.rfind('/') + 1);
        name      = name.substr(0, name.find('?'));
        const auto dir    = root / "urls" / ninferno::sha256_hex(spec).substr(0, 16);
        const auto target = dir / name;
        out.repo          = spec;
        out.snapshot      = dir;
        out.model         = target;
        if (fs::exists(target)) {
            out.files.push_back({name, fs::file_size(target), 0.0, 0, true, false});
            return out;
        }
        const auto head = request(spec, {}, true, true);
        if (head.code != CURLE_OK || head.status >= 400) {
            throw hub_error("resolving " + spec, head, "");
        }
        DownloadRequest file;
        file.url    = head.effective.empty() ? spec : head.effective;
        file.target = target;
        if (const auto it = head.headers.find("content-length"); it != head.headers.end()) {
            file.size = std::stoull(it->second);
        }
        log_line(options, "fetching " + name + " (" + human(static_cast<double>(file.size)) + ")");
        out.files.push_back(download_file(file, options));
        return out;
    }
    auto [repo, revision] = split_spec(spec, options);
    const auto hub        = make_hub(options, revision);
    const auto repo_dir   = hub->cache_dir(root, repo);
    Listing listing;
    try {
        if (hub_offline()) { throw std::runtime_error("HF_HUB_OFFLINE is set"); }
        listing = hub->list(repo, revision);
    } catch (const std::exception& error) {
        const auto commit = cached_commit(repo_dir, repo, revision, root, error);
        log_line(options, std::string("listing failed (") + error.what() +
                              "); using the cached snapshot " + commit);
        out.repo     = repo;
        out.commit   = commit;
        out.offline  = true;
        out.snapshot = repo_dir / "snapshots" / commit;
        out.model    = cached_model(out.snapshot, options, repo);
        return out;
    }
    check_pinned(options, repo, revision, listing);
    const auto files = choose_files(listing, options, repo);
    out.repo         = repo;
    out.commit       = listing.commit;
    out.snapshot     = repo_dir / "snapshots" / listing.commit;
    std::uint64_t total = 0;
    for (const auto& file : files) { total += file.size; }
    log_line(options, "fetching " + repo + "@" + listing.commit.substr(0, 12) + ": " +
                          std::to_string(files.size()) + " file(s), " +
                          human(static_cast<double>(total)));
    if (options.before_download) {
        FetchPlan plan{.repo = repo, .commit = listing.commit, .root = root, .total_bytes = total};
        for (const auto& file : files) {
            plan.files.push_back(file.path);
            const auto link = out.snapshot / file.path;
            if (!(fs::exists(link) && (!file.size || fs::file_size(link) == file.size))) {
                plan.missing_bytes += file.size;
            }
        }
        options.before_download(plan);
    }
    for (const auto& file : files) {
        const auto link = out.snapshot / file.path;
        if (fs::exists(link) && (!file.size || fs::file_size(link) == file.size)) {
            out.files.push_back({file.path, fs::file_size(link), 0.0, 0, true, false});
            continue;
        }
        auto resolved = hub->resolve(repo, listing.commit, file);
        const auto blob =
            !resolved.sha256.empty() ? resolved.sha256
            : !resolved.git_sha1.empty()
                ? resolved.git_sha1
                : file.path.substr(file.path.rfind('/') + 1) + "." + listing.commit;
        const auto blob_path = repo_dir / "blobs" / blob;
        FetchedFile fetched;
        if (fs::exists(blob_path) && (!resolved.size || fs::file_size(blob_path) == resolved.size)) {
            fetched = {file.path, fs::file_size(blob_path), 0.0, 0, true, false};
        } else {
            log_line(options, "  " + file.path + " (" + human(static_cast<double>(resolved.size)) +
                                  ")");
            DownloadRequest download;
            download.label    = file.path;
            download.url      = resolved.url;
            download.target   = blob_path;
            download.size     = resolved.size;
            download.sha256   = resolved.sha256;
            download.git_sha1 = resolved.sha256.empty() ? resolved.git_sha1 : std::string();
            if (resolved.authorized) { download.headers = hub->credentials(); }
            auto* hub_ptr    = hub.get();
            download.refresh = [hub_ptr, repo, commit = listing.commit, file] {
                return hub_ptr->resolve(repo, commit, file).url;
            };
            fetched      = download_file(download, options);
            fetched.path = file.path;
            // fetched.cached: another process fetched it while this one waited on its lock.
            if (!fetched.cached) {
                log_line(options, "  " + file.path + ": " +
                                      human(static_cast<double>(fetched.bytes)) + " in " +
                                      std::to_string(fetched.seconds).substr(0, 5) + " s (" +
                                      human(static_cast<double>(fetched.bytes) /
                                            std::max(fetched.seconds, 1e-6)) +
                                      "/s, " + std::to_string(fetched.streams) + " stream(s)" +
                                      (fetched.verified ? ", verified" : "") + ")");
            }
        }
        link_snapshot(repo_dir, listing.commit, file.path, blob);
        out.files.push_back(fetched);
    }
    if (revision != listing.commit) {
        fs::create_directories(repo_dir / "refs");
        std::ofstream(repo_dir / "refs" / revision) << listing.commit;
    }
    out.model = model_of(out.snapshot, files);
    return out;
}

GgufHead fetch_gguf_head(const std::string& spec, const FetchOptions& options,
                         std::uint64_t bytes) {
    if (bytes == 0) { throw std::invalid_argument("fetch_gguf_head: no bytes asked for"); }
    const auto root       = options.cache.empty() ? default_hub_cache() : options.cache;
    auto [repo, revision] = split_spec(spec, options);
    const auto hub        = make_hub(options, revision);
    const auto repo_dir   = hub->cache_dir(root, repo);
    GgufHead out;
    out.repo              = repo;
    const auto read_local = [&](const fs::path& file) {
        out.size = fs::file_size(file);
        out.bytes.resize(static_cast<std::size_t>(std::min(bytes, out.size)));
        std::ifstream stream(file, std::ios::binary);
        stream.read(out.bytes.data(), static_cast<std::streamsize>(out.bytes.size()));
        if (stream.gcount() != static_cast<std::streamsize>(out.bytes.size())) {
            throw std::runtime_error("cannot read " + file.string());
        }
        out.cached = true;
    };
    const auto no_gguf = [&](const std::string& where) {
        return std::runtime_error(repo + "@" + revision + ": no GGUF model" +
                                  (options.quant.empty() ? "" : " for " + options.quant) + where);
    };
    Listing listing;
    try {
        if (hub_offline()) { throw std::runtime_error("HF_HUB_OFFLINE is set"); }
        listing = hub->list(repo, revision);
    } catch (const std::exception& error) {
        out.commit          = cached_commit(repo_dir, repo, revision, root, error);
        out.offline         = true;
        const auto snapshot = repo_dir / "snapshots" / out.commit;
        const auto model    = cached_model(snapshot, options, repo);
        if (model == snapshot) { throw no_gguf(" in the cached snapshot " + snapshot.string()); }
        out.path = model.lexically_relative(snapshot).generic_string();
        read_local(model);
        return out;
    }
    check_pinned(options, repo, revision, listing);
    auto files = choose_files(listing, options, repo);
    std::erase_if(files,
                  [](const RemoteFile& f) { return !ascii_lower(f.path).ends_with(".gguf"); });
    if (files.empty()) { throw no_gguf(""); }
    const auto first = std::min_element(
        files.begin(), files.end(), [](const auto& a, const auto& b) { return a.path < b.path; });
    out.commit      = listing.commit;
    out.path        = first->path;
    const auto link = repo_dir / "snapshots" / listing.commit / first->path;
    if (fs::exists(link) && (!first->size || fs::file_size(link) == first->size)) {
        read_local(link);
        return out;
    }
    const auto resolved = hub->resolve(repo, listing.commit, *first);
    out.size            = resolved.size ? resolved.size : first->size;
    const auto want     = out.size ? std::min(bytes, out.size) : bytes;
    std::uint64_t total = 0;
    out.bytes = range_prefix(resolved.url,
                             resolved.authorized ? hub->credentials() : std::vector<std::string>{},
                             want, total);
    // A listing without sizes: the answer says how large the file is.
    if (out.size == 0) { out.size = total; }
    if (out.size == 0 || out.bytes.size() > out.size) {
        throw std::runtime_error(repo + "/" + out.path + ": the hub does not say how large it is");
    }
    log_line(options, "read " + human(static_cast<double>(out.bytes.size())) + " of " + repo + "/" +
                          out.path + " (" + human(static_cast<double>(out.size)) + ")");
    return out;
}


// ---- Hub listings for ninferno-ui: quantizations of a repository, search, small files.

namespace {

std::string native_quant_label(const std::string& path) {
    static const std::regex format(R"([-_.](nvfp4|mxfp4|fp8|int8|int4|q8|q6|q5|q4)(?=[-_.]|$))");
    const auto stem = ascii_lower(fs::path(path).stem().string());
    std::smatch match;
    if (!std::regex_search(stem, match, format)) { return "native"; }
    return ascii_upper(match[1].str());
}

} // namespace

std::string gguf_quant_label(const std::string& file_name) {
    static const std::regex label(
        R"(.*?[-_.]((?:UD-)?(?:I?Q[0-9][A-Z0-9_]*|BF16|F16|F32|MXFP4[A-Z0-9_]*|NVFP4[A-Z0-9_]*))(?:[-_.]MTP)?)",
        std::regex::icase);
    const auto stem  = gguf_group(fs::path(file_name).filename().string());
    std::string base = stem;
    if (ascii_lower(base).ends_with(".gguf")) { base.resize(base.size() - 5); }
    std::smatch match;
    return std::regex_match(base, match, label) ? match[1].str() : std::string();
}

namespace {

std::vector<HubModel> search_modelscope(const std::string& term, const FetchOptions& options,
                                        int limit, const std::string& sort) {
    auto endpoint = options.endpoint.empty() ? sys::env("MODELSCOPE_ENDPOINT") : options.endpoint;
    if (endpoint.empty()) { endpoint = "https://modelscope.cn"; }
    const auto url = endpoint + "/openapi/v1/models?search=" + term +
                     "&page_size=" + std::to_string(std::clamp(limit, 1, 100));
    const auto response = request(url, {}, false, true);
    if (response.code != CURLE_OK || response.status != 200) {
        throw hub_error("searching ModelScope for " + term, response, "");
    }
    const auto body = Json::parse(response.body);
    std::vector<HubModel> out;
    for (const auto& item : body.value("data", Json::object()).value("models", Json::array())) {
        HubModel model;
        model.id        = item.value("id", std::string());
        model.downloads = item.value("downloads", std::uint64_t{0});
        model.likes     = item.value("likes", std::uint64_t{0});
        model.updated =
            item.value(sort == "createdAt" ? "created_at" : "last_modified", std::string());
        const auto tasks = item.value("tasks", Json::array());
        if (!tasks.empty() && tasks[0].is_string()) {
            model.pipeline = tasks[0].get<std::string>();
        }
        for (const auto& tag : item.value("tags", Json::array())) {
            if (!tag.is_string()) { continue; }
            const auto text = tag.get<std::string>();
            if (text == "library:gguf") { model.gguf = true; }
            if (text.starts_with("model_type:")) { model.architecture = text.substr(11); }
            if (text == "library:mlx") { model.library = "mlx"; }
        }
        if (!model.id.empty()) { out.push_back(std::move(model)); }
    }
    std::stable_sort(out.begin(), out.end(), [&](const HubModel& a, const HubModel& b) {
        if (sort == "likes") { return a.likes > b.likes; }
        if (sort == "lastModified" || sort == "createdAt") { return a.updated > b.updated; }
        return a.downloads > b.downloads;
    });
    return out;
}

} // namespace

// The quantization a config.json declares when the converter has no reader for it (formats.cpp:
// quant_config and the matrix sources it picks), "" otherwise.
static std::string unread_quantization(const Json& config) {
    auto q = config.value("quantization_config", Json::object());
    if (q.empty()) {
        q = config.value("text_config", Json::object())
                .value("quantization_config", Json::object());
    }
    if (!q.is_object() || q.empty()) { return {}; }
    const auto method                       = ascii_lower(q.value("quant_method", std::string()));
    static const std::set<std::string> read = {
        "",         "fp8",        "fbgemm_fp8",       "awq",  "gptq", "compressed-tensors",
        "modelopt", "auto-round", "intel/auto-round", "mxfp4"};
    if (!read.contains(method)) { return method; }
    if (const auto t = q.find("transform_config"); t != q.end() && t->is_object() && !t->empty()) {
        return method + " (transforms)";
    }
    if (const auto s = q.find("sparsity_config"); s != q.end() && s->is_object() && !s->empty() &&
                                                  s->value("format", std::string()) != "dense") {
        return method + " (sparse)";
    }
    return {};
}

std::vector<HubModel> search_models(const std::string& query, const FetchOptions& options,
                                    int limit, const std::string& sort) {
    const bool modelscope = is_modelscope_hub(options.hub);
    if (!modelscope && !is_hugging_face_hub(options.hub)) {
        throw std::runtime_error("unknown hub '" + options.hub + "' (huggingface or modelscope)");
    }
    static const std::set<std::string> orders = {"downloads", "likes", "lastModified", "createdAt",
                                                 "trendingScore"};
    if (!orders.contains(sort)) { throw std::invalid_argument("unknown search order: " + sort); }
    auto endpoint = options.endpoint.empty() ? sys::env("HF_ENDPOINT") : options.endpoint;
    if (endpoint.empty()) { endpoint = "https://huggingface.co"; }
    std::string term;
    for (const unsigned char c : query.substr(0, 120)) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '/') {
            term += static_cast<char>(c);
        } else {
            char code[4];
            std::snprintf(code, sizeof(code), "%%%02X", c);
            term += code;
        }
    }
    if (modelscope) { return search_modelscope(term, options, limit, sort); }
    // expand[] names every field the answer carries: the listing's own and what the hub read of
    // the weights (config.json, the GGUF header) for the architecture.
    std::string url = endpoint + "/api/models?search=" + term +
                      (term.empty() ? "&filter=gguf" : "") + "&sort=" + sort +
                      "&direction=-1&limit=" + std::to_string(std::clamp(limit, 1, 100));
    for (const char* field : {"downloads", "likes", "lastModified", "pipeline_tag", "tags",
                              "config", "gguf", "library_name"}) {
        url += std::string("&expand%5B%5D=") + field;
    }
    const HuggingFace hub(endpoint);
    const auto response = request(url, hub.credentials(), false, true);
    if (response.code != CURLE_OK || response.status != 200) {
        throw hub_error("searching " + query, response, "");
    }
    std::vector<HubModel> out;
    for (const auto& item : Json::parse(response.body)) {
        HubModel model;
        model.id        = item.value("id", item.value("modelId", std::string()));
        model.downloads = item.value("downloads", std::uint64_t{0});
        model.likes     = item.value("likes", std::uint64_t{0});
        model.updated   = item.value("lastModified", std::string());
        model.pipeline  = item.value("pipeline_tag", std::string());
        for (const auto& tag : item.value("tags", Json::array())) {
            if (tag.is_string() && tag.get<std::string>() == "gguf") { model.gguf = true; }
            if (tag.is_string() && tag.get<std::string>() == "mlx") { model.library = "mlx"; }
        }
        if (model.library.empty()) { model.library = item.value("library_name", std::string()); }
        if (const auto gguf = item.find("gguf"); gguf != item.end() && gguf->is_object()) {
            model.architecture = gguf->value("architecture", std::string());
            if (const auto n = gguf->find("context_length"); n != gguf->end() && n->is_number()) {
                model.context = n->get<std::int64_t>();
            }
        }
        if (const auto config = item.find("config"); config != item.end() && config->is_object()) {
            model.unread_quantization = unread_quantization(*config);
        }
        if (const auto config = item.find("config");
            model.architecture.empty() && config != item.end() && config->is_object()) {
            const auto archs   = config->value("architectures", Json::array());
            model.architecture = !archs.empty() && archs[0].is_string()
                                     ? archs[0].get<std::string>()
                                     : config->value("model_type", std::string());
        }
        if (!model.id.empty()) { out.push_back(std::move(model)); }
    }
    return out;
}

std::string fetch_text(const std::string& url, std::size_t max_bytes) {
    if (!url.starts_with("https://") && !url.starts_with("http://")) {
        throw std::invalid_argument("not an http(s) URL: " + url);
    }
    // A private Hugging Face repository (the catalog's dataset) answers 401 without the token,
    // which goes to the hub's own host only.
    std::vector<std::string> headers;
    auto endpoint = sys::env("HF_ENDPOINT");
    if (endpoint.empty()) { endpoint = "https://huggingface.co"; }
    auto token = sys::env("HF_TOKEN");
    if (token.empty()) { token = sys::env("HUGGING_FACE_HUB_TOKEN"); }
    if (!token.empty() && url.starts_with(endpoint + "/")) {
        headers.push_back("Authorization: Bearer " + token);
    }
    const auto response = request(url, headers, false, true);
    if (response.code != CURLE_OK || response.status != 200) {
        throw hub_error("reading " + url, response,
                        response.status == 401 && headers.empty() && url.starts_with(endpoint + "/")
                            ? "a private repository needs HF_TOKEN"
                            : "");
    }
    if (response.body.size() > max_bytes) {
        throw std::runtime_error(url + " is larger than " + std::to_string(max_bytes) + " bytes");
    }
    return response.body;
}

Json fetch_config(const std::string& spec, const FetchOptions& options) {
    return fetch_repo_json(spec, options, "config.json");
}

Json fetch_repo_json(const std::string& spec, const FetchOptions& options,
                     const std::string& file) {
    if (!is_hugging_face_hub(options.hub)) {
        throw std::runtime_error(file + " is read from Hugging Face only");
    }
    auto [repo, revision] = split_spec(spec, options);
    if (revision.empty()) { revision = "main"; }
    auto endpoint = options.endpoint.empty() ? sys::env("HF_ENDPOINT") : options.endpoint;
    if (endpoint.empty()) { endpoint = "https://huggingface.co"; }
    const HuggingFace hub(endpoint);
    const auto response = request(endpoint + "/" + repo + "/resolve/" + revision + "/" + file,
                                  hub.credentials(), false, true);
    if (response.code != CURLE_OK || response.status != 200) {
        throw hub_error("reading " + repo + "'s " + file, response, "");
    }
    auto parsed = Json::parse(response.body, nullptr, false);
    if (!parsed.is_object()) {
        throw std::runtime_error(repo + "'s " + file + " is not a JSON object");
    }
    return parsed;
}

int quant_width_tag(const std::string& quant) {
    static const std::regex tag(R"(([0-9]{1,2})[-_]?bits?)", std::regex::icase);
    std::smatch match;
    if (!std::regex_match(quant, match, tag)) { return 0; }
    const int bits = std::stoi(match[1].str());
    return bits >= 1 && bits <= 16 ? bits : 0;
}

int quant_label_bits(const std::string& label) {
    const auto text = ascii_lower(label);
    if (text.find("bf16") != std::string::npos || text.find("f16") != std::string::npos) {
        return 16;
    }
    if (text.find("f32") != std::string::npos) { return 32; }
    if (text.find("fp4") != std::string::npos) { return 4; }
    static const std::regex bits(R"((?:^|[-_])i?q([0-9]))");
    std::smatch match;
    return std::regex_search(text, match, bits) ? std::stoi(match[1].str()) : 0;
}

std::vector<RepoQuant> list_quants(const std::string& spec, const FetchOptions& options) {
    auto [repo, revision] = split_spec(spec, options);
    const auto hub        = make_hub(options, revision);
    const auto listing    = hub->list(repo, revision);
    std::vector<RepoQuant> out;
    std::map<std::string, std::size_t> index; // group key -> out position
    RepoQuant weights;
    weights.format = "safetensors";
    std::vector<const RemoteFile*> top_level; // the top-level safetensors weights
    for (const auto& file : listing.files) {
        const auto name = ascii_lower(fs::path(file.path).filename().string());
        // An importance matrix (imatrix*.gguf) or another file too small to be a model.
        if (name.find("imatrix") != std::string::npos) { continue; }
        // A vision projector is listed whatever its size (best_projector chooses among them).
        if (name.ends_with(".gguf") && gguf_quant_label(file.path).empty() &&
            file.size < (256ULL << 20) && !name.starts_with("mmproj")) {
            continue;
        }
        if (name.ends_with(".gguf")) {
            const bool sidecar = gguf_auxiliary_file(name) ||
                                 name.find("dflash") != std::string::npos ||
                                 name.find("eagle") != std::string::npos ||
                                 name.find("assistant") != std::string::npos;
            auto label       = gguf_quant_label(file.path);
            const auto group = gguf_group(fs::path(file.path).filename().string());
            const auto key =
                sidecar || label.empty() ? "file:" + file.path : "quant:" + ascii_lower(label);
            if (label.empty() || sidecar) {
                label = sidecar ? group : fs::path(file.path).stem().string();
            }
            auto [it, fresh] = index.emplace(key, out.size());
            if (fresh) {
                RepoQuant quant;
                quant.quant   = label;
                quant.format  = "gguf";
                quant.bits    = quant_label_bits(label);
                quant.drafter = sidecar;
                out.push_back(std::move(quant));
            }
            out[it->second].bytes += file.size;
            out[it->second].files.push_back(file.path);
        } else if (name.ends_with(".ninferno") && file.path.find('/') == std::string::npos) {
            RepoQuant quant;
            quant.quant  = native_quant_label(file.path);
            quant.format = "ninferno";
            quant.bits   = quant_label_bits(quant.quant);
            quant.bytes  = file.size;
            quant.files.push_back(file.path);
            out.push_back(std::move(quant));
        } else if (name.ends_with(".safetensors") && name.find("visual") == std::string::npos &&
                   file.path.find('/') == std::string::npos) {
            // The top-level weights, as the fetch takes them (subdirectories hold originals and
            // other formats: gpt-oss's original/model.safetensors).
            top_level.push_back(&file);
        }
    }
    // Of them, the Transformers-named ones when there are any, as choose_files takes them.
    const bool named = std::any_of(top_level.begin(), top_level.end(), [](const auto* f) {
        return transformers_weight_file(f->path);
    });
    for (const auto* file : top_level) {
        if (!named || transformers_weight_file(file->path)) {
            weights.bytes += file->size;
            weights.files.push_back(file->path);
            // A part of its own for the MTP layers (model_mtp.safetensors): fetched and converted
            // with the model, loaded only under MTP.
            if (ascii_lower(file->path).find("mtp") != std::string::npos) {
                weights.mtp_bytes += file->size;
            }
        }
    }
    if (weights.bytes > 0) { out.push_back(std::move(weights)); }
    return out;
}

} // namespace ninferno::convert
