// PDF reading through pdfium. Without the library on the machine, reading must refuse with
// Unavailable; with it, PDFs built here must read back their text, and a page without text must
// render as a PNG when asked.

#include "media/pdf/pdf_text.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace pdf = ninferno::media::pdf;

namespace {

int check(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what);
    return ok ? 0 : 1;
}

// A minimal PDF with one page per content stream, Helvetica as F1, xref offsets computed.
std::string pdf_of(const std::vector<std::string>& streams) {
    std::vector<std::string> objects = {"<< /Type /Catalog /Pages 2 0 R >>", ""};
    std::string kids;
    for (std::size_t i = 0; i < streams.size(); ++i) {
        const std::size_t page = 4 + 2 * i, content = page + 1;
        kids += std::to_string(page) + " 0 R ";
        (void)content;
    }
    objects[1] = "<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(streams.size()) + " >>";
    objects.push_back("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"); // object 3
    for (std::size_t i = 0; i < streams.size(); ++i) {
        const std::size_t content = 5 + 2 * i;
        objects.push_back("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents " +
                          std::to_string(content) + " 0 R /Resources << /Font << /F1 3 0 R >> >> >>");
        objects.push_back("<< /Length " + std::to_string(streams[i].size()) + " >>\nstream\n" +
                          streams[i] + "\nendstream");
    }
    std::string out = "%PDF-1.4\n";
    std::vector<std::size_t> offsets;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        offsets.push_back(out.size());
        out += std::to_string(i + 1) + " 0 obj\n" + objects[i] + "\nendobj\n";
    }
    const std::size_t xref = out.size();
    out += "xref\n0 " + std::to_string(objects.size() + 1) + "\n0000000000 65535 f \n";
    for (const std::size_t offset : offsets) {
        char line[32];
        std::snprintf(line, sizeof line, "%010zu 00000 n \n", offset);
        out += line;
    }
    out += "trailer\n<< /Size " + std::to_string(objects.size() + 1) +
           " /Root 1 0 R >>\nstartxref\n" + std::to_string(xref) + "\n%%EOF\n";
    return out;
}

std::string text_stream(const std::string& text) {
    return "BT /F1 24 Tf 72 720 Td (" + text + ") Tj ET";
}

std::vector<std::uint8_t> bytes_of(const std::string& s) { return {s.begin(), s.end()}; }

bool is_png(const std::vector<std::uint8_t>& b) {
    return b.size() > 8 && b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G';
}

} // namespace

int main() {
    int failures = 0;

    const std::vector<std::uint16_t> units = {'a', '\r', '\n', 'b', '\r', 0x00E9, 0x20AC,
                                              0xD83D, 0xDE00, 0xD800, 'c', 0, 'z'};
    failures += check(pdf::utf16_to_utf8(units) == "a\nb\n\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80\xEF\xBF\xBD" "c",
                      "UTF-16 to UTF-8: CRLF, BMP, surrogate pair, lone surrogate, terminator");

    // 2x70000 pixels: more than one stored deflate block.
    std::vector<std::uint8_t> rgb(2 * 70000 * 3, 0x7F);
    const auto png = pdf::encode_png_rgb(2, 70000, rgb);
    failures += check(is_png(png) && png.size() > rgb.size(), "PNG encoding of a tall image");
    if (const char* out = std::getenv("NINFERNO_PDF_TEST_PNG"); out != nullptr && *out) {
        std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(png.data()),
                                                   static_cast<std::streamsize>(png.size()));
    }

    std::string why;
    if (!pdf::available(&why)) {
        failures += check(!why.empty(), "an unavailable library names why");
        bool refused = false;
        try {
            (void)pdf::read(bytes_of(pdf_of({text_stream("x")})));
        } catch (const pdf::Error& error) { refused = error.kind() == pdf::ErrorKind::Unavailable; }
        failures += check(refused, "reading without pdfium refuses as Unavailable");
        std::printf("pdfium not found (%s): reading checked as refused\n", why.c_str());
        return failures;
    }

    const auto doc = pdf::read(bytes_of(pdf_of({text_stream("Hello PDF 42"), text_stream("Second")})));
    failures += check(doc.pages.size() == 2, "two pages read");
    failures += check(doc.pages.size() == 2 && doc.pages[0].text.find("Hello PDF 42") != std::string::npos &&
                          doc.pages[1].text.find("Second") != std::string::npos,
                      "page text read back");
    failures += check(doc.pages.size() == 2 && doc.pages[0].image_png.empty(),
                      "a text page rendered without being asked");

    // A page with a filled rectangle and no text: an image only when asked.
    const auto scan = bytes_of(pdf_of({"0 0 1 rg 100 100 300 400 re f", text_stream("words")}));
    const auto plain = pdf::read(scan);
    failures += check(plain.pages.size() == 2 && plain.pages[0].text.empty() &&
                          plain.pages[0].image_png.empty(),
                      "a textless page rendered without render_textless");
    pdf::Limits render;
    render.render_textless   = true;
    render.max_render_pixels = 200 * 260;
    const auto rendered = pdf::read(scan, render);
    failures += check(rendered.pages.size() == 2 && is_png(rendered.pages[0].image_png) &&
                          rendered.pages[1].image_png.empty(),
                      "the textless page rendered as a PNG, the text page not");
    if (const char* out = std::getenv("NINFERNO_PDF_TEST_PAGE"); out != nullptr && *out &&
                                                                 !rendered.pages.empty()) {
        const auto& page = rendered.pages[0].image_png;
        std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(page.data()),
                                                   static_cast<std::streamsize>(page.size()));
    }
    render.max_rendered = 0;
    failures += check(pdf::read(scan, render).textless_not_rendered == 1,
                      "a textless page past max_rendered is counted");

    bool malformed = false;
    try {
        (void)pdf::read(bytes_of("not a pdf at all"));
    } catch (const pdf::Error& error) { malformed = error.kind() == pdf::ErrorKind::Malformed; }
    failures += check(malformed, "junk refuses as Malformed");

    bool too_large = false;
    try {
        pdf::Limits none;
        none.max_pages = 0;
        (void)pdf::read(bytes_of(pdf_of({text_stream("x")})), none);
    } catch (const pdf::Error& error) { too_large = error.kind() == pdf::ErrorKind::TooLarge; }
    failures += check(too_large, "page limit refuses as TooLarge");
    std::printf("pdfium: text and page rendering checked\n");
    return failures;
}
