#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::media::pdf {

// A PDF read page by page with the pdfium library loaded at run time: libpdfium.so (pdfium.dll on
// Windows) beside the executable, in the release's lib/ directory, the path in NINFERNO_PDFIUM, or
// the system's library path. Without it, PDF documents are refused and nothing else changes.

enum class ErrorKind {
    Unavailable, // no pdfium library on this machine
    Malformed,   // not a PDF, or one pdfium cannot open
    Encrypted,   // password protected
    TooLarge,    // more pages than the limit
};

class Error : public std::runtime_error {
public:
    Error(ErrorKind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

struct Limits {
    std::size_t max_pages = 1000;
    // A page without a text layer (a scan, a figure) rendered as an image for a vision model:
    // at most this many pages, each at most max_render_dpi and max_render_pixels.
    bool render_textless          = false;
    std::size_t max_rendered      = 100;
    double max_render_dpi         = 150.0;
    std::uint64_t max_render_pixels = 1920ULL * 1080ULL;
};

struct Page {
    std::string text; // UTF-8; lines end in '\n'; empty for a page without a text layer
    // The page as a PNG when it has no text layer and Limits::render_textless asked for it.
    std::vector<std::uint8_t> image_png;
};

struct Document {
    std::vector<Page> pages;
    std::size_t textless_not_rendered = 0; // textless pages past Limits::max_rendered
};

// True when the pdfium library loads; the reason it does not, in `why`, otherwise.
bool available(std::string* why = nullptr);

// Every page's text layer, and images of the textless ones when asked. Throws Error.
Document read(std::span<const std::uint8_t> bytes, const Limits& limits = {});

// UTF-16 code units (pdfium's text) as UTF-8, CRLF and lone CR as '\n'; an unpaired surrogate
// becomes U+FFFD. Exposed for tests.
std::string utf16_to_utf8(std::span<const std::uint16_t> units);

// An 8-bit RGB image (rows of width * 3 bytes) as a PNG with stored (uncompressed) deflate
// blocks: any decoder reads it, and it never leaves the process. Exposed for tests.
std::vector<std::uint8_t> encode_png_rgb(int width, int height, std::span<const std::uint8_t> rgb);

} // namespace ninferno::media::pdf
