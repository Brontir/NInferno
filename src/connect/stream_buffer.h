#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace ninferno::connect {

// The bytes of one stream's answer that the engine has produced and the router has not yet taken.
// The producer appends and the consumer takes from the front. Consumed bytes are dropped once they
// are at least as many as the unread ones, so the storage stays within about twice the unread bytes
// (plus the last append) however long the stream runs, and each byte is moved at most once for the
// bytes taken after it. Not thread-safe: the stream's mutex guards it.
class StreamBuffer {
public:
    void append(const char* data, std::size_t size) { bytes_.append(data, size); }
    // Replaces the content with `bytes`, all of it unread.
    void assign(std::string bytes) {
        bytes_ = std::move(bytes);
        taken_ = 0;
    }

    [[nodiscard]] std::size_t unread() const noexcept { return bytes_.size() - taken_; }
    // Bytes held, read or not: what the buffer costs in memory (apart from spare capacity).
    [[nodiscard]] std::size_t stored() const noexcept { return bytes_.size(); }

    // Copies up to `length` unread bytes to `out`, removes them and returns their count.
    std::size_t take(char* out, std::size_t length) {
        const std::size_t n = std::min(length, unread());
        std::memcpy(out, bytes_.data() + taken_, n);
        taken_ += n;
        if (taken_ == bytes_.size()) {
            bytes_.clear();
            taken_ = 0;
        } else if (taken_ >= unread()) {
            // A producer that keeps the buffer nonempty never lets it drain, so the consumed
            // prefix is reclaimed here.
            bytes_.erase(0, taken_);
            taken_ = 0;
        }
        return n;
    }

private:
    std::string bytes_;
    std::size_t taken_ = 0;
};

} // namespace ninferno::connect
