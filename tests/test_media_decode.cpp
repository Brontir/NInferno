#include "media/decode/decode.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

// This 300x200 yuvj420p JPEG selected an swscale SIMD path that overwrote a
// width-tight RGB24 destination. Keep the exact public reproducer so the test does not depend on
// an encoder or external files.
constexpr std::string_view issue_20_jpeg_base64 = R"jpeg(
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAx
NDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy
MjIyMjIyMjL/wAARCADIASwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF
BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW
V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi
4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC
AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm
Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq
8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDFooor7E+RCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA7TQtC0290W3uLi23yvu3NvYZwxHY+1aP/AAjGj/8APn/5
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
z/nP/IyxH+Of/pTP0rAf7rS/wx/JBRRRXmnWFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXDeM/+QxD/
ANe6/wDoTV3NcN4z/wCQxD/17r/6E1fXcE/8jZf4ZHicQf7k/VHOUUUV+yHwoUUUUAFFFFABRRRQAUUUUAFFFFAHeeHdQsoNBto5
ry3jcbsq8oBHzHsTWr/aunf8/wDa/wDf5f8AGvLqK+HxfA+HxOIqV3Vac23surufQUeIatKlGmoLRJfceo/2rp3/AD/2v/f5f8aP
7V07/n/tf+/y/wCNeXUVz/8AEP8ADf8AP6X3I1/1lq/8+197PUf7V07/AJ/7X/v8v+NH9q6d/wA/9r/3+X/GvLqKP+If4b/n9L7k
H+stX/n2vvZ6j/aunf8AP/a/9/l/xo/tXTv+f+1/7/L/AI15dRR/xD/Df8/pfcg/1lq/8+197PUf7V07/n/tf+/y/wCNH9q6d/z/
ANr/AN/l/wAa8uoo/wCIf4b/AJ/S+5B/rLV/59r72eo/2rp3/P8A2v8A3+X/ABo/tXTv+f8Atf8Av8v+NeXUUf8AEP8ADf8AP6X3
IP8AWWr/AM+197PUf7V07/n/ALX/AL/L/jR/aunf8/8Aa/8Af5f8a8uoo/4h/hv+f0vuQf6y1f8An2vvZ6j/AGrp3/P/AGv/AH+X
/Gj+1dO/5/7X/v8AL/jXl1FH/EP8N/z+l9yD/WWr/wA+197PUf7V07/n/tf+/wAv+NH9q6d/z/2v/f5f8a8uoo/4h/hv+f0vuQf6
y1f+fa+9nqP9q6d/z/2v/f5f8aP7V07/AJ/7X/v8v+NeXUUf8Q/w3/P6X3IP9Zav/Ptfez1H+1dO/wCf+1/7/L/jR/aunf8AP/a/
9/l/xry6ij/iH+G/5/S+5B/rLV/59r72eo/2rp3/AD/2v/f5f8aP7V07/n/tf+/y/wCNeXUUf8Q/w3/P6X3IP9Zav/Ptfez1H+1d
O/5/7X/v8v8AjXGeLbiG51WJ4Jo5VEABaNgwzubjisGivUyfhOjlmJWJhUcnZqzS6nHjs6qYuj7KUUgooor6w8YKKKKACiiigAoo
ooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
gD//2Q==
)jpeg";

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
    std::array<int, 256> values;
    values.fill(-1);
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        values[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
    }

    std::vector<std::uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    unsigned accumulator = 0;
    int bits             = 0;
    for (const unsigned char byte : encoded) {
        if (std::isspace(byte)) { continue; }
        if (byte == '=') { break; }
        const int value = values[byte];
        if (value < 0) { throw std::runtime_error("invalid base64 fixture"); }
        accumulator = (accumulator << 6U) | static_cast<unsigned>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(accumulator >> bits));
            accumulator &= (1U << bits) - 1U;
        }
    }
    return out;
}

void expect_pixel(const ninferno::media::decode::Image& image, int x, int y,
                  std::array<int, 3> expected, int tolerance) {
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 3;
    for (int channel = 0; channel < 3; ++channel) {
        const int actual = image.rgb[offset + static_cast<std::size_t>(channel)];
        if (actual < expected[static_cast<std::size_t>(channel)] - tolerance ||
            actual > expected[static_cast<std::size_t>(channel)] + tolerance) {
            throw std::runtime_error("decoded JPEG pixel mismatch");
        }
    }
}

void test_issue_20_unaligned_jpeg() {
    const std::vector<std::uint8_t> encoded     = decode_base64(issue_20_jpeg_base64);
    const ninferno::media::decode::ImageInfo info =
        ninferno::media::decode::inspect_image(encoded, {});
    const ninferno::media::decode::Image image = ninferno::media::decode::decode_image(encoded, {});
    if (info.width != image.width || info.height != image.height || image.width != 300 ||
        image.height != 200 || image.rgb.size() != 300U * 200U * 3U) {
        throw std::runtime_error("decoded JPEG dimensions mismatch");
    }

    constexpr std::array<int, 3> blue   = {29, 120, 199};
    constexpr std::array<int, 3> yellow = {255, 200, 1};
    expect_pixel(image, 0, 0, blue, 3);
    expect_pixel(image, 299, 0, blue, 3);
    expect_pixel(image, 0, 199, blue, 3);
    expect_pixel(image, 299, 199, blue, 3);
    expect_pixel(image, 150, 100, yellow, 4);

    const ninferno::media::decode::VideoInfo video_info =
        ninferno::media::decode::inspect_video(encoded, {}, 2.0, 4, 16);
    const ninferno::media::decode::Video video =
        ninferno::media::decode::decode_video(encoded, {}, 2.0, 4, 16);
    if (video_info.width != video.width || video_info.height != video.height ||
        video_info.sampled_frames != static_cast<int>(video.frames.size()) ||
        video_info.indices != video.indices) {
        throw std::runtime_error("inspected video geometry differs from decoded video");
    }
}

std::size_t resident_bytes() {
    std::ifstream statm("/proc/self/statm");
    std::size_t pages    = 0;
    std::size_t resident = 0;
    statm >> pages >> resident;
    return resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
}

// Every Decoder opens its input through a custom AVIOContext. Probing replaces the context's
// buffer, so the buffer must be freed from the context and not from the pointer handed to
// avio_alloc_context. Found by LeakSanitizer: each image or video request leaked one probe buffer
// of about 32 KiB, for the whole server lifetime. 2000 inspect+decode rounds leaked ~128 MiB;
// the bound leaves room for allocator noise only.
std::size_t peak_resident_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) { return std::stoull(line.substr(6)) * 1024U; }
    }
    return 0;
}

// A grayscale PNG of side x side zero pixels whose IDAT is one fixed-Huffman deflate block of
// back-references (258 bytes per 13 bits): 16384 x 16384 is 268 MB decoded and 2.5 MB encoded.
std::vector<std::uint8_t> zero_png(std::uint32_t side) {
    std::vector<std::uint8_t> zlib = {0x78, 0x01};
    std::uint64_t bits             = 0;
    int count                      = 0;
    const auto put                 = [&](std::uint32_t value, int length) {
        bits |= static_cast<std::uint64_t>(value) << count;
        count += length;
        while (count >= 8) {
            zlib.push_back(static_cast<std::uint8_t>(bits & 0xffU));
            bits >>= 8U;
            count -= 8;
        }
    };
    const auto huffman = [&](std::uint32_t code, int length) {
        std::uint32_t reversed = 0;
        for (int i = 0; i < length; ++i) { reversed |= ((code >> i) & 1U) << (length - 1 - i); }
        put(reversed, length);
    };
    const std::uint64_t raw = static_cast<std::uint64_t>(side) * (side + 1U); // filter byte + row
    put(1, 1);                                                                // final block
    put(1, 2);                                                                // fixed Huffman
    huffman(0x30, 8);                                                         // literal 0
    std::uint64_t left = raw - 1;
    for (; left >= 258; left -= 258) {
        huffman(0xc5, 8); // length 258
        huffman(0, 5);    // distance 1
    }
    for (; left > 0; --left) { huffman(0x30, 8); }
    huffman(0, 7); // end of block
    if (count > 0) { zlib.push_back(static_cast<std::uint8_t>(bits & 0xffU)); }
    const std::uint32_t adler = static_cast<std::uint32_t>(raw % 65521U) << 16U | 1U;
    for (int shift = 24; shift >= 0; shift -= 8) {
        zlib.push_back(static_cast<std::uint8_t>(adler >> shift));
    }

    std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    const auto chunk              = [&](const char* type, const std::vector<std::uint8_t>& data) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            png.push_back(static_cast<std::uint8_t>(data.size() >> shift));
        }
        const std::size_t begin = png.size();
        png.insert(png.end(), type, type + 4);
        png.insert(png.end(), data.begin(), data.end());
        std::uint32_t crc = 0xffffffffU;
        for (std::size_t i = begin; i < png.size(); ++i) {
            crc ^= png[i];
            for (int k = 0; k < 8; ++k) { crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U))); }
        }
        crc ^= 0xffffffffU;
        for (int shift = 24; shift >= 0; shift -= 8) {
            png.push_back(static_cast<std::uint8_t>(crc >> shift));
        }
    };
    std::vector<std::uint8_t> header;
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            header.push_back(static_cast<std::uint8_t>(side >> shift));
        }
    }
    header.insert(header.end(), {8, 0, 0, 0, 0}); // 8-bit grayscale, deflate, no interlace
    chunk("IHDR", header);
    chunk("IDAT", zlib);
    chunk("IEND", {});
    return png;
}

// A decompression bomb is refused before FFmpeg allocates its frame: 268 MB of declared pixels
// past the 64 Mpx budget grow the peak resident set by far less than the frame.
void test_pixel_bomb_refused_before_allocation() {
    const std::vector<std::uint8_t> bomb = zero_png(16384);
    ninferno::media::decode::Policy small;
    small.max_decoded_pixels = 4096U * 4096U;
    bool small_refused       = false;
    try {
        (void)ninferno::media::decode::decode_image(zero_png(4097), small);
    } catch (const ninferno::media::decode::Error&) { small_refused = true; }
    if (!small_refused) { throw std::runtime_error("a 4097^2 PNG passed a 4096^2 budget"); }
    for (const bool inspect : {true, false}) {
        const std::size_t before = peak_resident_bytes();
        bool refused             = false;
        try {
            if (inspect) {
                (void)ninferno::media::decode::inspect_image(bomb, {});
            } else {
                (void)ninferno::media::decode::decode_image(bomb, {});
            }
        } catch (const ninferno::media::decode::Error&) { refused = true; }
        const std::size_t grown = peak_resident_bytes() - before;
        if (!refused) { throw std::runtime_error("a 16384^2 PNG passed the 64 Mpx budget"); }
        if (grown > (64U << 20U)) {
            throw std::runtime_error("refusing a 16384^2 PNG grew the peak resident set by " +
                                     std::to_string(grown >> 20U) + " MiB");
        }
    }
}

void test_decoder_releases_probe_buffers() {
    const std::vector<std::uint8_t> encoded = decode_base64(issue_20_jpeg_base64);
    const auto round                        = [&] {
        (void)ninferno::media::decode::inspect_image(encoded, {});
        (void)ninferno::media::decode::decode_image(encoded, {});
    };
    for (int i = 0; i < 200; ++i) { round(); } // warm the allocator and codec tables
    const std::size_t before = resident_bytes();
    for (int i = 0; i < 2000; ++i) { round(); }
    const std::size_t after = resident_bytes();
    const std::size_t grown = after > before ? after - before : 0;
    if (grown > (16U << 20U)) {
        throw std::runtime_error("resident memory grew by " + std::to_string(grown >> 10U) +
                                 " KiB over 4000 decoder lifetimes");
    }
}

} // namespace

int main() {
    try {
        test_issue_20_unaligned_jpeg();
        test_pixel_bomb_refused_before_allocation();
        test_decoder_releases_probe_buffers();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
