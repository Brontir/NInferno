#include "media/pdf/pdf_text.h"

#include "core/sys.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>

namespace ninferno::media::pdf {
namespace {

// pdfium's C API (public/fpdfview.h, fpdf_text.h) as far as NInferno uses it. The handles are
// opaque pointers; on 64-bit targets the calling convention is the platform's one.
using Handle = void*;
struct Api {
    void (*init)()                                                         = nullptr;
    Handle (*load_mem_document)(const void*, int, const char*)             = nullptr;
    unsigned long (*last_error)()                                          = nullptr;
    int (*page_count)(Handle)                                              = nullptr;
    Handle (*load_page)(Handle, int)                                       = nullptr;
    void (*close_page)(Handle)                                             = nullptr;
    void (*close_document)(Handle)                                         = nullptr;
    float (*page_width)(Handle)                                            = nullptr;
    float (*page_height)(Handle)                                           = nullptr;
    Handle (*text_load_page)(Handle)                                       = nullptr;
    void (*text_close_page)(Handle)                                        = nullptr;
    int (*text_count_chars)(Handle)                                        = nullptr;
    int (*text_get_text)(Handle, int, int, unsigned short*)                = nullptr;
    Handle (*bitmap_create)(int, int, int)                                 = nullptr;
    int (*bitmap_fill_rect)(Handle, int, int, int, int, unsigned long)     = nullptr;
    void (*render_page_bitmap)(Handle, Handle, int, int, int, int, int, int) = nullptr;
    void* (*bitmap_buffer)(Handle)                                         = nullptr;
    int (*bitmap_stride)(Handle)                                           = nullptr;
    void (*bitmap_destroy)(Handle)                                         = nullptr;
};

constexpr unsigned long kErrorPassword = 4; // FPDF_ERR_PASSWORD
constexpr int kRenderAnnotations       = 0x01; // FPDF_ANNOT

struct Library {
    Api api;
    std::string why; // empty when loaded
};

template <typename F> bool bind(void* library, const char* name, F& slot) {
    slot = reinterpret_cast<F>(sys::symbol(library, name));
    return slot != nullptr;
}

void* open_library() {
    std::vector<std::filesystem::path> candidates;
    if (const char* env = std::getenv("NINFERNO_PDFIUM"); env != nullptr && *env) {
        candidates.emplace_back(env);
    }
#ifdef _WIN32
    constexpr const char* kName = "pdfium.dll";
#else
    constexpr const char* kName = "libpdfium.so";
#endif
    if (const auto exe = sys::self_exe(); !exe.empty()) {
        candidates.push_back(exe.parent_path() / kName);
        candidates.push_back(exe.parent_path().parent_path() / "lib" / kName);
    }
    for (const auto& path : candidates) {
        const std::string name = path.string();
        if (void* library = sys::load_library(name.c_str(), name.c_str())) return library;
    }
    return sys::load_library(kName, kName); // the system's library path
}

const Library& library() {
    static const Library loaded = [] {
        Library result;
        void* handle = open_library();
        if (handle == nullptr) {
            result.why = "PDF documents need the pdfium library (libpdfium.so, or pdfium.dll on "
                         "Windows) beside the server or in NINFERNO_PDFIUM; this installation has none";
            return result;
        }
        Api& a        = result.api;
        const bool ok = bind(handle, "FPDF_InitLibrary", a.init) &&
                        bind(handle, "FPDF_LoadMemDocument", a.load_mem_document) &&
                        bind(handle, "FPDF_GetLastError", a.last_error) &&
                        bind(handle, "FPDF_GetPageCount", a.page_count) &&
                        bind(handle, "FPDF_LoadPage", a.load_page) &&
                        bind(handle, "FPDF_ClosePage", a.close_page) &&
                        bind(handle, "FPDF_CloseDocument", a.close_document) &&
                        bind(handle, "FPDF_GetPageWidthF", a.page_width) &&
                        bind(handle, "FPDF_GetPageHeightF", a.page_height) &&
                        bind(handle, "FPDFText_LoadPage", a.text_load_page) &&
                        bind(handle, "FPDFText_ClosePage", a.text_close_page) &&
                        bind(handle, "FPDFText_CountChars", a.text_count_chars) &&
                        bind(handle, "FPDFText_GetText", a.text_get_text) &&
                        bind(handle, "FPDFBitmap_Create", a.bitmap_create) &&
                        bind(handle, "FPDFBitmap_FillRect", a.bitmap_fill_rect) &&
                        bind(handle, "FPDF_RenderPageBitmap", a.render_page_bitmap) &&
                        bind(handle, "FPDFBitmap_GetBuffer", a.bitmap_buffer) &&
                        bind(handle, "FPDFBitmap_GetStride", a.bitmap_stride) &&
                        bind(handle, "FPDFBitmap_Destroy", a.bitmap_destroy);
        if (!ok) {
            result.why = "the pdfium library found lacks functions NInferno uses";
            return result; // the handle stays loaded: unloading a library mid-process is not worth it
        }
        a.init();
        return result;
    }();
    return loaded;
}

// pdfium is not thread-safe: one document at a time in the whole process.
std::mutex& pdfium_mutex() {
    static std::mutex mutex;
    return mutex;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

bool has_text(const std::string& text) {
    return text.find_first_not_of(" \t\n") != std::string::npos;
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc = 0) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320U ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<std::uint8_t>(v >> shift));
}

void chunk(std::vector<std::uint8_t>& out, const char type[4], const std::vector<std::uint8_t>& data) {
    put32(out, static_cast<std::uint32_t>(data.size()));
    const std::size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    put32(out, crc32(out.data() + start, out.size() - start));
}

// One page as RGB at a scale within the dpi and pixel limits, white behind transparent content.
std::vector<std::uint8_t> render_png(const Api& a, Handle page, const Limits& limits) {
    const double page_width  = static_cast<double>(a.page_width(page));
    const double page_height = static_cast<double>(a.page_height(page));
    if (!std::isfinite(page_width) || !std::isfinite(page_height)) return {};
    const double width_pt  = std::max(1.0, page_width);
    const double height_pt = std::max(1.0, page_height);
    const auto budget      = static_cast<double>(limits.max_render_pixels);
    double scale           = limits.max_render_dpi / 72.0;
    const double pixels    = width_pt * height_pt * scale * scale;
    if (pixels > budget) { scale *= std::sqrt(budget / pixels); }
    // A sliver page rounds its short side up to one pixel: its long side then takes what is left
    // of the budget, so the bitmap never exceeds it.
    double width  = std::max(1.0, std::floor(width_pt * scale));
    double height = std::max(1.0, std::floor(height_pt * scale));
    if (width * height > budget) {
        if (width > height) {
            width = std::max(1.0, std::floor(budget / height));
        } else {
            height = std::max(1.0, std::floor(budget / width));
        }
    }
    if (width * height > budget) return {};
    const int w   = static_cast<int>(width);
    const int h   = static_cast<int>(height);
    Handle bitmap = a.bitmap_create(w, h, 0); // BGRx
    if (bitmap == nullptr) return {};
    const std::unique_ptr<void, void (*)(Handle)> guard(bitmap, a.bitmap_destroy);
    a.bitmap_fill_rect(bitmap, 0, 0, w, h, 0xFFFFFFFFUL);
    a.render_page_bitmap(bitmap, page, 0, 0, w, h, 0, kRenderAnnotations);
    const auto* buffer = static_cast<const std::uint8_t*>(a.bitmap_buffer(bitmap));
    const int stride   = a.bitmap_stride(bitmap);
    if (buffer == nullptr || stride < w * 4) return {};
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* row = buffer + static_cast<std::size_t>(y) * stride;
        std::uint8_t* to        = rgb.data() + static_cast<std::size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            to[3 * x]     = row[4 * x + 2];
            to[3 * x + 1] = row[4 * x + 1];
            to[3 * x + 2] = row[4 * x];
        }
    }
    return encode_png_rgb(w, h, rgb);
}

} // namespace

std::vector<std::uint8_t> encode_png_rgb(int width, int height, std::span<const std::uint8_t> rgb) {
    const std::size_t row = static_cast<std::size_t>(width) * 3;
    if (width <= 0 || height <= 0 || rgb.size() != row * static_cast<std::size_t>(height)) {
        throw std::invalid_argument("encode_png_rgb: the buffer does not hold width x height RGB");
    }
    std::vector<std::uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::vector<std::uint8_t> header;
    put32(header, static_cast<std::uint32_t>(width));
    put32(header, static_cast<std::uint32_t>(height));
    header.insert(header.end(), {8, 2, 0, 0, 0}); // 8-bit RGB, deflate, no filter set, no interlace
    chunk(out, "IHDR", header);

    // zlib stream of stored blocks over the filtered rows (filter byte 0 before each row).
    std::vector<std::uint8_t> raw;
    raw.reserve((row + 1) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + static_cast<std::ptrdiff_t>(y * row),
                   rgb.begin() + static_cast<std::ptrdiff_t>((y + 1) * row));
    }
    std::vector<std::uint8_t> z = {0x78, 0x01};
    z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    std::size_t at = 0;
    do {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - at);
        z.push_back(at + n == raw.size() ? 1 : 0);
        z.push_back(static_cast<std::uint8_t>(n));
        z.push_back(static_cast<std::uint8_t>(n >> 8));
        z.push_back(static_cast<std::uint8_t>(~n));
        z.push_back(static_cast<std::uint8_t>(~n >> 8));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(at),
                 raw.begin() + static_cast<std::ptrdiff_t>(at + n));
        at += n;
    } while (at < raw.size());
    std::uint32_t s1 = 1, s2 = 0;
    for (const std::uint8_t b : raw) {
        s1 = (s1 + b) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    put32(z, (s2 << 16) | s1);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
    return out;
}

std::string utf16_to_utf8(std::span<const std::uint16_t> units) {
    std::string out;
    out.reserve(units.size());
    for (std::size_t i = 0; i < units.size(); ++i) {
        std::uint32_t u = units[i];
        if (u == 0) break; // pdfium's terminator
        if (u == '\r') {
            out += '\n';
            if (i + 1 < units.size() && units[i + 1] == '\n') ++i;
            continue;
        }
        if (u >= 0xD800 && u <= 0xDBFF && i + 1 < units.size() && units[i + 1] >= 0xDC00 &&
            units[i + 1] <= 0xDFFF) {
            u = 0x10000 + ((u - 0xD800) << 10) + (units[i + 1] - 0xDC00);
            ++i;
        } else if (u >= 0xD800 && u <= 0xDFFF) {
            u = 0xFFFD;
        }
        append_utf8(out, u);
    }
    return out;
}

bool available(std::string* why) {
    const Library& lib = library();
    if (why != nullptr) *why = lib.why;
    return lib.why.empty();
}

Document read(std::span<const std::uint8_t> bytes, const Limits& limits) {
    const Library& lib = library();
    if (!lib.why.empty()) throw Error(ErrorKind::Unavailable, lib.why);
    if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw Error(ErrorKind::TooLarge, "PDF document exceeds 2 GiB");
    }
    const Api& a = lib.api;
    std::lock_guard lock(pdfium_mutex());
    Handle doc = a.load_mem_document(bytes.data(), static_cast<int>(bytes.size()), nullptr);
    if (doc == nullptr) {
        if (a.last_error() == kErrorPassword) {
            throw Error(ErrorKind::Encrypted, "the PDF document is password protected");
        }
        throw Error(ErrorKind::Malformed, "the document is not a PDF pdfium can open");
    }
    const std::unique_ptr<void, void (*)(Handle)> doc_guard(doc, a.close_document);
    const int pages = a.page_count(doc);
    if (pages < 0) throw Error(ErrorKind::Malformed, "the PDF document has no page tree");
    if (static_cast<std::size_t>(pages) > limits.max_pages) {
        throw Error(ErrorKind::TooLarge, "the PDF document has " + std::to_string(pages) +
                                             " pages; at most " + std::to_string(limits.max_pages) +
                                             " are read");
    }
    Document result;
    result.pages.resize(static_cast<std::size_t>(pages));
    std::vector<std::uint16_t> units;
    std::size_t rendered = 0;
    for (int p = 0; p < pages; ++p) {
        Handle page = a.load_page(doc, p);
        if (page == nullptr) continue; // a broken page reads as empty; the rest still count
        const std::unique_ptr<void, void (*)(Handle)> page_guard(page, a.close_page);
        Page& out = result.pages[static_cast<std::size_t>(p)];
        if (Handle text = a.text_load_page(page)) {
            const std::unique_ptr<void, void (*)(Handle)> text_guard(text, a.text_close_page);
            const int chars = a.text_count_chars(text);
            if (chars > 0) {
                units.assign(static_cast<std::size_t>(chars) + 1, 0);
                const int written =
                    a.text_get_text(text, 0, chars, reinterpret_cast<unsigned short*>(units.data()));
                if (written > 0) {
                    out.text = utf16_to_utf8(
                        std::span<const std::uint16_t>(units.data(), static_cast<std::size_t>(written)));
                }
            }
        }
        if (limits.render_textless && !has_text(out.text)) {
            if (rendered < limits.max_rendered) {
                out.image_png = render_png(a, page, limits);
                rendered += out.image_png.empty() ? 0 : 1;
            } else {
                ++result.textless_not_rendered;
            }
        }
    }
    return result;
}

} // namespace ninferno::media::pdf
