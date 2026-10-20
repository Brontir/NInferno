// ggml_frag_v1 (core/ggml_frag.h): each format's map is a bijection of its unit; a row goes to
// fragment order and back unchanged (random rows, Q4_0 rows with a natural tail, and, given GGUF
// paths as arguments, the routed-expert rows of real checkpoints); the geometry is the published
// layout's; the auto rule picks the measured shapes; and ninferno-convert --relayout moves a
// hand-built multi-file artifact's banks to ggml_frag_v1 and back byte for byte, in place (also
// across an interruption inside a chunk and inside the directory write, with the artifact refused
// meanwhile; a foreign or damaged journal refused; a concurrent relayout refused; a directory
// without room to grow refused untouched) and into a new artifact (its speculation files and
// report carried over under the new id).
//
//   ninferno_convert_ggml_frag_test [model.gguf ...]

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "convert/expert_layout.h"
#include "convert/gguf.h"
#include "convert/relayout.h"
#include "core/ggml_frag.h"
#include "core/weight_view.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace ninferno;
namespace cv = ninferno::convert;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

bool throws(const std::function<void()>& body, const std::string& fragment) {
    try {
        body();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(fragment) != std::string::npos;
    }
    return false;
}

constexpr QType kFormats[] = {QType::GGML_Q4_0, QType::GGML_Q4_K, QType::GGML_IQ3_XXS,
                              QType::GGML_IQ2_S};

const char* name(QType format) {
    switch (format) {
    case QType::GGML_Q4_0: return "q4_0";
    case QType::GGML_Q4_K: return "q4_k";
    case QType::GGML_IQ3_XXS: return "iq3_xxs";
    case QType::GGML_IQ2_S: return "iq2_s";
    default: return "?";
    }
}

std::uint64_t code_row_bytes(QType format, std::uint64_t k) {
    const std::uint64_t shape[] = {1, k};
    return weight_geometry(format, ggml_published_layout(format), shape).code_bytes_per_row;
}

std::vector<std::uint8_t> random_bytes(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> out(n);
    for (auto& b : out) { b = static_cast<std::uint8_t>(rng()); }
    return out;
}

// Every unit byte has exactly one source; the bytes the format documents as unchanged (the Q4_K
// head, the IQ2_S qh bytes) stay.
void test_maps() {
    for (const auto format : kFormats) {
        const auto unit = ggml_frag_unit_bytes(format);
        std::vector<int> hits(unit, 0);
        for (std::uint32_t i = 0; i < unit; ++i) {
            const auto s = ggml_frag_source(format, i);
            check(s < unit, std::string(name(format)) + ": source out of the unit");
            if (s < unit) { ++hits[s]; }
        }
        check(std::all_of(hits.begin(), hits.end(), [](int h) { return h == 1; }),
              std::string(name(format)) + ": the map is not a bijection");
    }
    for (std::uint32_t i = 0; i < 16; ++i) {
        check(ggml_frag_source(QType::GGML_Q4_K, i) == i, "the Q4_K head moved");
    }
    for (std::uint32_t i = 64; i < 72; ++i) {
        check(ggml_frag_source(QType::GGML_IQ2_S, i) == i, "an IQ2_S qh byte moved");
    }
    check(!ggml_frag_format(QType::GGML_Q6_K) && !ggml_frag_format(QType::GGML_IQ4_XS) &&
              !ggml_frag_format(QType::NVFP4),
          "a format without a fragment-order kernel has a map");
    std::uint8_t in[256] = {}, out[256] = {};
    check(throws([&] { ggml_frag_row(QType::GGML_Q6_K, 256, in, out, true); }, "no map"),
          "Q6_K rows are refused");
    check(throws([&] { ggml_frag_row(QType::GGML_Q4_K, 128, in, out, true); }, "no map"),
          "a K that is not whole superblocks is refused");
}

// One row to fragment order and back; each unit's bytes where the map puts them; the Q4_0 tail
// (K/32 % 4 blocks) unchanged. Returns whether the row changed.
bool round_trip(QType format, std::uint64_t k, const std::uint8_t* row, const std::string& what) {
    const auto bytes = code_row_bytes(format, k);
    std::vector<std::uint8_t> frag(bytes, 0xA5), back(bytes, 0x5A);
    ggml_frag_row(format, k, row, frag.data(), true);
    ggml_frag_row(format, k, frag.data(), back.data(), false);
    check(std::memcmp(back.data(), row, bytes) == 0, what + ": the inverse differs");
    const auto unit   = ggml_frag_unit_bytes(format);
    const auto blocks = k / ggml_block_values(format);
    const auto units  = format == QType::GGML_Q4_0 ? blocks / 4 : blocks;
    bool mapped       = true;
    for (std::uint64_t u = 0; u < units; ++u) {
        for (std::uint32_t i = 0; i < unit; ++i) {
            mapped &= frag[u * unit + i] == row[u * unit + ggml_frag_source(format, i)];
        }
    }
    check(mapped, what + ": a byte is not where the map puts it");
    check(std::memcmp(frag.data() + units * unit, row + units * unit, bytes - units * unit) == 0,
          what + ": the row tail moved");
    return std::memcmp(frag.data(), row, bytes) != 0;
}

void test_random_rows() {
    std::uint32_t seed = 1;
    for (const auto format : kFormats) {
        const auto values = ggml_block_values(format);
        std::vector<std::uint64_t> ks = {values, 2 * values, 7 * values, 11 * values};
        if (format == QType::GGML_Q4_0) { ks = {32, 64, 96, 128, 160, 704, 1856, 2688, 2816}; }
        for (const auto k : ks) {
            const auto row  = random_bytes(code_row_bytes(format, k), ++seed);
            const auto what = std::string(name(format)) + " K" + std::to_string(k);
            const bool changed = round_trip(format, k, row.data(), what);
            // A Q4_0 row of fewer than four blocks is all tail.
            check(changed == (format != QType::GGML_Q4_0 || k >= 128), what + ": changed");
        }
    }
}

// The layout's geometry is the published one's under another name.
void test_geometry() {
    for (const auto format : kFormats) {
        const std::uint64_t shape[] = {24, format == QType::GGML_Q4_0 ? 704ULL : 2048ULL};
        const auto published = weight_geometry(format, ggml_published_layout(format), shape);
        const auto frag      = weight_geometry(format, QuantLayout::GgmlFrag, shape);
        check(frag.layout == QuantLayout::GgmlFrag, std::string(name(format)) + ": layout");
        check(frag.bytes == published.bytes && frag.code_bytes == published.code_bytes &&
                  frag.code_bytes_per_row == published.code_bytes_per_row &&
                  frag.high_offset == published.high_offset &&
                  frag.high_bytes == published.high_bytes &&
                  frag.scale_offset == published.scale_offset &&
                  frag.scale_bytes == published.scale_bytes &&
                  frag.alignment == published.alignment,
              std::string(name(format)) + ": the frag geometry is not the published one");
    }
    const std::uint64_t shape[] = {8, 512};
    check(throws([&] { (void)weight_geometry(QType::GGML_Q6_K, QuantLayout::GgmlFrag, shape); },
                 "ggml_frag_v1"),
          "a Q6_K frag geometry is refused");
    check(artifact::layout_name(QuantLayout::GgmlFrag) == "ggml_frag_v1" &&
              artifact::parse_layout("ggml_frag_v1") == QuantLayout::GgmlFrag,
          "the layout name registry");
}

void test_rule() {
    using cv::ExpertLayout;
    const auto layout = [](QType f, std::uint64_t k, ExpertLayout c) {
        return cv::expert_bank_layout(f, k, c, ggml_published_layout(f));
    };
    const auto frag = QuantLayout::GgmlFrag;
    // The shipped banks: Nemotron Q4_0 2688 / 1856, Gemma 4 26B Q4_0 2816 (gate/up) and 704
    // (down, measured slower: published), Qwen3.6 Q4_K 2048, IQ2_S 2048, IQ3_XXS 512; Q6_K and
    // Q5_K downs have no map.
    check(layout(QType::GGML_Q4_0, 2688, ExpertLayout::Auto) == frag, "q4_0 2688 auto");
    check(layout(QType::GGML_Q4_0, 1856, ExpertLayout::Auto) == frag, "q4_0 1856 auto");
    check(layout(QType::GGML_Q4_0, 2816, ExpertLayout::Auto) == frag, "q4_0 2816 auto");
    check(layout(QType::GGML_Q4_0, 704, ExpertLayout::Auto) == QuantLayout::GgmlPlanes,
          "q4_0 704 auto stays published");
    check(layout(QType::GGML_Q4_K, 2048, ExpertLayout::Auto) == frag, "q4_k auto");
    check(layout(QType::GGML_IQ2_S, 2048, ExpertLayout::Auto) == frag, "iq2_s auto");
    check(layout(QType::GGML_IQ3_XXS, 512, ExpertLayout::Auto) == frag, "iq3_xxs auto");
    check(layout(QType::GGML_Q6_K, 512, ExpertLayout::Auto) == QuantLayout::GgmlPlanes, "q6_k");
    check(layout(QType::GGML_Q5_K, 512, ExpertLayout::Frag) == QuantLayout::GgmlBlocks, "q5_k");
    check(layout(QType::GGML_Q4_0, 704, ExpertLayout::Frag) == frag, "q4_0 704 frag");
    check(layout(QType::GGML_Q4_K, 2048, ExpertLayout::Published) == QuantLayout::GgmlBlocks,
          "q4_k published");
    check(cv::expert_bank_layout(QType::MXFP4, 2048, ExpertLayout::Frag,
                                 QuantLayout::RowSplit) == QuantLayout::RowSplit,
          "a non-ggml format keeps its layout");
    check(cv::parse_expert_layout("published") == ExpertLayout::Published &&
              cv::parse_expert_layout("frag") == ExpertLayout::Frag &&
              cv::parse_expert_layout("auto") == ExpertLayout::Auto,
          "layout names");
    check(throws([] { (void)cv::parse_expert_layout("fragment"); }, "fragment"),
          "an unknown layout name is refused");
}

// ---------------------------------------------------------------------------------------------
// The routed-expert rows of real GGUF checkpoints (argv): every row of the adopted formats'
// expert tensors (sampled on large ones) goes to fragment order and back unchanged.

struct Mapped {
    const std::byte* data = nullptr;
    std::size_t size      = 0;
    int fd                = -1;
    explicit Mapped(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { throw std::runtime_error("cannot open " + path); }
        struct stat st{};
        fstat(fd, &st);
        size = static_cast<std::size_t>(st.st_size);
        auto* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) { throw std::runtime_error("cannot map " + path); }
        data = static_cast<const std::byte*>(p);
    }
    ~Mapped() {
        if (data) { munmap(const_cast<std::byte*>(data), size); }
        if (fd >= 0) { close(fd); }
    }
    Mapped(const Mapped&)            = delete;
    Mapped& operator=(const Mapped&) = delete;
};

// A ggml block's code bytes (the code plane of its published layout) within the block.
struct BlockCodes {
    QType format;
    std::uint64_t block, begin, end;
};

std::optional<BlockCodes> block_codes(std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 2: return BlockCodes{QType::GGML_Q4_0, 18, 2, 18};     // d qs[16]
    case 12: return BlockCodes{QType::GGML_Q4_K, 144, 0, 144};  // the whole superblock
    case 18: return BlockCodes{QType::GGML_IQ3_XXS, 98, 2, 66}; // d qs[64] scales/signs[32]
    case 22: return BlockCodes{QType::GGML_IQ2_S, 82, 2, 74};   // d qs[32] signs[32] qh[8]
    default: return std::nullopt;
    }
}

void test_gguf(const std::string& path) {
    const Mapped file(path);
    const auto gguf = cv::parse_gguf(file.data, file.size, path);
    std::map<std::string, std::uint64_t> rows_by_format;
    std::uint64_t changed = 0, total = 0;
    std::mt19937_64 rng(7);
    for (const auto& t : gguf.tensors) {
        const auto codes = block_codes(t.type);
        if (!codes || t.name.find("_exps") == std::string::npos) { continue; }
        const auto k         = t.ne.at(0);
        const auto values    = ggml_block_values(codes->format);
        const auto blocks    = k / values;
        const auto row_bytes = blocks * codes->block;
        const auto rows      = t.bytes / row_bytes;
        const auto* base =
            reinterpret_cast<const std::uint8_t*>(file.data) + gguf.data_offset + t.offset;
        std::vector<std::uint64_t> sample;
        for (std::uint64_t r = 0; r < std::min<std::uint64_t>(rows, 512); ++r) {
            sample.push_back(r);
        }
        for (int i = 0; i < 1536 && rows > 512; ++i) { sample.push_back(rng() % rows); }
        std::vector<std::uint8_t> plane(code_row_bytes(codes->format, k));
        const auto code = codes->end - codes->begin;
        for (const auto r : sample) {
            const auto* row = base + r * row_bytes;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                std::memcpy(plane.data() + b * code, row + b * codes->block + codes->begin, code);
            }
            const auto what = t.name + " row " + std::to_string(r);
            changed += round_trip(codes->format, k, plane.data(), what);
            ++total;
        }
        rows_by_format[std::string(name(codes->format)) + " K" + std::to_string(k)] +=
            sample.size();
    }
    std::string summary;
    for (const auto& [key, n] : rows_by_format) {
        summary += " " + key + ":" + std::to_string(n);
    }
    std::printf("%s: %llu expert rows round-tripped (%llu changed by the map);%s\n",
                path.c_str(), static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(changed), summary.c_str());
    check(total > 0, path + ": no routed-expert rows of an adopted format");
    check(changed * 10 >= total * 9, path + ": the map left most rows unchanged");
}

// ---------------------------------------------------------------------------------------------
// --relayout on a hand-built artifact: an entry file and a second payload file split inside a
// bank; routed banks of every adopted format (Q4_0 at K 2816 and 704), a dense Q4_0 projection
// and a resource that must not move.

using Json = nlohmann::json;

struct Object {
    std::string id, binding;
    QType format;
    std::uint64_t n, k;
    std::uint64_t offset = 0, bytes = 0, code_row = 0;
};

void put(std::vector<std::uint8_t>& out, std::size_t at, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) { out[at + i] = static_cast<std::uint8_t>(v >> (8 * i)); }
}

struct TestArtifact {
    std::filesystem::path dir, entry;
    std::vector<Object> objects;
    std::vector<std::uint8_t> payload;
    std::uint8_t id_byte;

    explicit TestArtifact(std::uint8_t id = 0x71) : id_byte(id) {
        auto pattern = (std::filesystem::temp_directory_path() / "ninferno-frag-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* made = ::mkdtemp(buffer.data());
        if (!made) { throw std::runtime_error("cannot create the test directory"); }
        dir   = made;
        entry = dir / "model.ninferno";
        objects = {
            {"l0_gate_up", "layers/0/mlp/experts/gate_up", QType::GGML_Q4_0, 16, 2816},
            {"l0_down", "layers/0/mlp/experts/down", QType::GGML_Q4_0, 16, 704},
            {"l0_q", "layers/0/attention/q", QType::GGML_Q4_0, 8, 2816},
            {"l1_gate_up", "layers/1/mlp/experts/gate_up", QType::GGML_Q4_K, 8, 2048},
            {"l1_down", "layers/1/mlp/experts/down", QType::GGML_IQ3_XXS, 8, 512},
            {"l2_gate_up", "layers/2/mlp/experts/gate_up", QType::GGML_IQ2_S, 8, 2048},
        };
        std::uint64_t at = 4096; // the resource sits at 0
        for (auto& o : objects) {
            const std::uint64_t shape[] = {o.n, o.k};
            const auto g = weight_geometry(o.format, ggml_published_layout(o.format), shape);
            o.offset     = at;
            o.bytes      = g.bytes;
            o.code_row   = g.code_bytes_per_row;
            at           = (at + g.bytes + 4095) / 4096 * 4096;
        }
        payload = random_bytes(at, 99);
    }
    ~TestArtifact() {
        std::error_code error;
        std::filesystem::remove_all(dir, error);
    }
    TestArtifact(const TestArtifact&)            = delete;
    TestArtifact& operator=(const TestArtifact&) = delete;

    // The directory reserve as the converter's writer leaves it (the entry payload starts at the
    // next 4096-byte boundary), or with `tight` none at all; the `frag` objects in ggml_frag_v1.
    void write(bool tight = false, const std::vector<std::string>& frag = {}) const {
        const auto bytes = expected(frag);
        const Json text_component = {{"config", Json::object()},
                                     {"resources", {{"notes", "notes"}}}};
        Json root = {{"components", {{"text", text_component}}},
                     {"objects", Json::array({{{"id", "notes"},
                                               {"kind", "resource"},
                                               {"encoding", "raw_bytes_v1"},
                                               {"offset", 0},
                                               {"bytes", 5}}})},
                     {"bindings", Json::object()},
                     {"uses", Json::array()},
                     {"metadata", {{"name", "frag-relayout-test"}}}};
        for (const auto& o : objects) {
            root["objects"].push_back(
                {{"id", o.id},
                 {"kind", "tensor"},
                 {"shape", {o.n, o.k}},
                 {"format", std::string(artifact::format_name(o.format))},
                 {"layout", std::string(artifact::layout_name(
                                std::find(frag.begin(), frag.end(), o.id) != frag.end()
                                    ? QuantLayout::GgmlFrag
                                    : ggml_published_layout(o.format)))},
                 {"offset", o.offset},
                 {"bytes", o.bytes}});
            root["bindings"][o.binding] = {{"object", o.id}};
        }
        // The second file starts inside the IQ3_XXS bank's code plane.
        const auto split = objects[4].offset + 3 * objects[4].code_row + 17;
        root["files"]    = Json::array({{{"path", nullptr}, {"payload_bytes", split}},
                                        {{"path", "weights-2.bin"},
                                         {"payload_bytes", payload.size() - split}}});
        auto text = root.dump();
        if (!tight) {
            text.append((32 + text.size() + 4095) / 4096 * 4096 - 32 - text.size(), ' ');
        }
        for (int i = 0; i < 2; ++i) {
            std::vector<std::uint8_t> header(32, 0);
            const char* magic = i == 0 ? "NInferno" : "NInfPart";
            std::memcpy(header.data(), magic, 8);
            put(header, 8, i == 0 ? text.size() : 1);
            header[16] = id_byte;
            std::ofstream out(i == 0 ? entry : dir / "weights-2.bin",
                              std::ios::binary | std::ios::trunc);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out.write(reinterpret_cast<const char*>(header.data()), 32);
            std::size_t position = 32;
            if (i == 0) {
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                position += text.size();
            }
            const std::string padding((position + 4095) / 4096 * 4096 - position, '\0');
            out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
            const auto begin = i == 0 ? 0 : split;
            const auto end   = i == 0 ? split : payload.size();
            out.write(reinterpret_cast<const char*>(bytes.data() + begin),
                      static_cast<std::streamsize>(end - begin));
        }
    }

    // The payload with the named objects' code rows in fragment order.
    std::vector<std::uint8_t> expected(const std::vector<std::string>& frag) const {
        auto out = payload;
        for (const auto& o : objects) {
            if (std::find(frag.begin(), frag.end(), o.id) == frag.end()) { continue; }
            for (std::uint64_t r = 0; r < o.n; ++r) {
                const auto at = o.offset + r * o.code_row;
                ggml_frag_row(o.format, o.k, payload.data() + at, out.data() + at, true);
            }
        }
        return out;
    }
};

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> read_payload(const std::filesystem::path& entry) {
    const artifact::Reader reader(entry);
    const auto bytes = reader.read_range(0, reader.directory().payload_bytes);
    std::vector<std::uint8_t> out(bytes.size());
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

std::map<std::string, std::string> layouts(const std::filesystem::path& entry) {
    const artifact::Reader reader(entry);
    std::map<std::string, std::string> out;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        const auto* t = std::get_if<artifact::TensorObject>(&reader.directory().objects[i]);
        if (t) {
            out[t->id] = t->layout;
            (void)reader.geometry(artifact::ObjectHandle{i}); // the layout is readable
        }
    }
    return out;
}

cv::RelayoutResult run(const std::filesystem::path& entry, cv::ExpertLayout layout,
                       std::uint64_t chunk = 64ULL << 20,
                       std::optional<std::filesystem::path> out = std::nullopt) {
    cv::RelayoutOptions options;
    options.artifact    = entry;
    options.layout      = layout;
    options.chunk_bytes = chunk;
    options.out         = std::move(out);
    return cv::relayout(options);
}

std::string hex_id(const std::filesystem::path& entry) {
    const auto id = artifact::Reader(entry).artifact_id();
    std::string out;
    char digits[3];
    for (const auto b : id) {
        std::snprintf(digits, sizeof digits, "%02x", std::to_integer<unsigned>(b));
        out += digits;
    }
    return out;
}

Json read_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    return Json::parse(in);
}

void write_json(const std::filesystem::path& path, const Json& value) {
    std::ofstream(path) << value.dump(2);
}

// Interrupted in the middle of a chunk (4 KiB chunks): the journal is left and the artifact is
// refused until the same command finishes.
void interrupt(const std::filesystem::path& entry, cv::ExpertLayout layout) {
    cv::RelayoutOptions options;
    options.artifact        = entry;
    options.layout          = layout;
    options.chunk_bytes     = 4096;
    options.interrupt_after = 20000;
    check(throws([&] { (void)cv::relayout(options); }, "interrupted"), "the interruption");
    check(std::filesystem::exists(cv::relayout_journal(entry)), "the journal is left");
}

void test_relayout() {
    using cv::ExpertLayout;
    const TestArtifact a;
    a.write();
    const std::vector<std::string> automatic = {"l0_gate_up", "l1_gate_up", "l1_down",
                                                "l2_gate_up"};
    auto every = automatic;
    every.push_back("l0_down");
    const auto published = a.expected({});
    check(read_payload(a.entry) == published, "the test artifact reads back");

    // In place, auto: four banks; the K 704 down, the dense projection and the resource stay.
    auto done = run(a.entry, ExpertLayout::Auto);
    check(done.objects == 4 && !done.resumed, "auto changed " + std::to_string(done.objects));
    check(read_payload(a.entry) == a.expected(automatic), "auto: the payload");
    auto names = layouts(a.entry);
    check(names["l0_gate_up"] == "ggml_frag_v1" && names["l1_gate_up"] == "ggml_frag_v1" &&
              names["l1_down"] == "ggml_frag_v1" && names["l2_gate_up"] == "ggml_frag_v1" &&
              names["l0_down"] == "ggml_planes_v1" && names["l0_q"] == "ggml_planes_v1",
          "auto: the layout names");
    check(!std::filesystem::exists(cv::relayout_journal(a.entry)), "auto: the journal is gone");
    check(run(a.entry, ExpertLayout::Auto).objects == 0, "auto twice changes nothing");

    // Back to published: the original bytes and names.
    done = run(a.entry, ExpertLayout::Published);
    check(done.objects == 4, "published changed " + std::to_string(done.objects));
    check(read_payload(a.entry) == published, "published: the original payload");
    names = layouts(a.entry);
    check(names["l1_gate_up"] == "ggml_blocks_v1" && names["l0_gate_up"] == "ggml_planes_v1",
          "published: the layout names");

    // frag, interrupted in the middle of a chunk, then resumed. Meanwhile the engines' reader, a
    // relayout to --out and a relayout to another layout refuse the artifact.
    interrupt(a.entry, ExpertLayout::Frag);
    check(throws([&] { (void)artifact::Reader(a.entry); }, "--relayout"),
          "the reader refuses an artifact with a pending relayout");
    check(throws([&] { (void)run(a.entry, ExpertLayout::Auto, 4096, a.dir / "x.ninferno"); },
                 "unfinished"),
          "--out refuses a source with a pending relayout");
    {
        const auto payload = artifact::Reader(a.entry, artifact::PendingRelayout::Allow)
                                 .read_range(0, a.payload.size());
        check(std::memcmp(payload.data(), published.data(), published.size()) != 0,
              "the interruption wrote part of a chunk");
    }
    check(throws([&] { (void)run(a.entry, ExpertLayout::Published, 4096); }, "pending"),
          "another layout is refused while a relayout is pending");
    // A damaged journal is refused, and so is another artifact's.
    {
        const auto journal = cv::relayout_journal(a.entry);
        auto bytes         = read_file(journal);
        bytes.back() ^= 0x40;
        write_file(journal, bytes);
        check(throws([&] { (void)run(a.entry, ExpertLayout::Frag, 4096); }, "checksum"),
              "a damaged journal is refused");
        bytes.back() ^= 0x40;
        write_file(journal, bytes);
        const TestArtifact other(0x72);
        other.write();
        interrupt(other.entry, ExpertLayout::Frag);
        const auto mine = read_file(journal);
        write_file(journal, read_file(cv::relayout_journal(other.entry)));
        check(throws([&] { (void)run(a.entry, ExpertLayout::Frag, 4096); }, "delete it"),
              "another artifact's journal is refused, naming its deletion");
        write_file(journal, mine);
    }
    // A second relayout while one holds the artifact.
    {
        const int fd = open(a.entry.c_str(), O_RDONLY | O_CLOEXEC);
        check(fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0, "the test takes the entry's lock");
        check(throws([&] { (void)run(a.entry, ExpertLayout::Frag, 4096); }, "is running"),
              "a relayout beside another is refused");
        if (fd >= 0) { close(fd); }
    }
    done = run(a.entry, ExpertLayout::Frag, 4096);
    check(done.resumed && done.objects == 5, "frag resumed");
    check(read_payload(a.entry) == a.expected(every), "frag after the resume: the payload");
    names = layouts(a.entry);
    check(names["l0_down"] == "ggml_frag_v1" && names["l0_q"] == "ggml_planes_v1",
          "frag: the layout names");
    check(!std::filesystem::exists(cv::relayout_journal(a.entry)), "frag: the journal is gone");

    // Interrupted inside the directory write: every chunk is done, the directory half written.
    {
        cv::RelayoutOptions options;
        options.artifact            = a.entry;
        options.layout              = ExpertLayout::Published;
        options.interrupt_directory = true;
        check(throws([&] { (void)cv::relayout(options); }, "interrupted"),
              "the directory interruption");
        check(throws([&] { (void)artifact::Reader(a.entry); }, "--relayout"),
              "the reader refuses a half-written directory");
        done = run(a.entry, ExpertLayout::Published);
        check(done.resumed, "the directory write resumed");
        check(read_payload(a.entry) == published, "published after the directory resume");
        check(layouts(a.entry)["l0_down"] == "ggml_planes_v1", "the resumed directory's names");
        check(run(a.entry, ExpertLayout::Frag).objects == 5, "back to frag");
    }

    // Into a new artifact: auto from all-frag moves the K 704 down back; the source stays. The
    // speculation manifest, the spec-tune profile and the report follow with the new id.
    const auto old_id = hex_id(a.entry);
    write_json(a.entry.string() + ".speculation.json",
               {{"schema", 1},
                {"target", {{"artifact_id", old_id}, {"repo", "test/model"}}},
                {"drafters",
                 Json::array({{{"name", "dflash"}, {"artifact", "drafter.ninferno"}},
                              {{"name", "mtp"}, {"component", "mtp"}}})},
                {"profiles", Json::array()}});
    write_json(a.entry.string() + ".spec_profile.json",
               {{"key", {{"artifact_id", old_id}, {"gpu", "ad102:128sm"}}}});
    write_json(a.entry.string() + ".conversion.json",
               {{"artifact_id", old_id}, {"output", a.entry.string()}});
    const auto copy = a.dir / "copy" / "model.ninferno";
    done            = run(a.entry, ExpertLayout::Auto, 64ULL << 20, copy);
    check(done.objects == 1, "--out auto changed " + std::to_string(done.objects));
    check(read_payload(copy) == a.expected(automatic), "--out: the new artifact's payload");
    check(read_payload(a.entry) == a.expected(every), "--out: the source is unchanged");
    const auto new_id = hex_id(copy);
    check(new_id != old_id, "--out: a new artifact id");
    {
        const auto manifest = read_json(copy.string() + ".speculation.json");
        check(manifest["target"]["artifact_id"] == new_id && manifest["target"]["repo"] ==
                                                                 "test/model",
              "--out: the manifest names the new id");
        check(manifest["drafters"][0]["artifact"] == (a.dir / "drafter.ninferno").string() &&
                  manifest["drafters"][1]["component"] == "mtp",
              "--out: a relative drafter path is made absolute");
        check(read_json(copy.string() + ".spec_profile.json")["key"]["artifact_id"] == new_id,
              "--out: the spec profile names the new id");
        const auto report = read_json(copy.string() + ".conversion.json");
        check(report["artifact_id"] == new_id && report["output"] == copy.string() &&
                  report["relayouts"].size() == 1 &&
                  report["relayouts"][0]["layout"] == "auto" &&
                  report["relayouts"][0]["changed"] == 1 &&
                  report["relayouts"][0]["ggml_frag_v1"].size() == 4,
              "--out: the report records the relayout");
        check(read_json(a.entry.string() + ".speculation.json")["target"]["artifact_id"] ==
                  old_id,
              "--out: the source's manifest is unchanged");
    }
    check(throws([&] { (void)run(a.entry, ExpertLayout::Auto, 64ULL << 20, copy); },
                 "already exists"),
          "--out refuses an existing path");
    done = run(copy, ExpertLayout::Published);
    check(done.objects == 4 && read_payload(copy) == published, "the copy back to published");
    {
        const auto report = read_json(copy.string() + ".conversion.json");
        check(report["relayouts"].size() == 2 && report["relayouts"][1]["layout"] == "published" &&
                  report["relayouts"][1]["ggml_frag_v1"].empty(),
              "in place: the report records the relayout");
    }

    // No room for the longer published names: refused before any byte is written, --out named.
    {
        const TestArtifact tight(0x73);
        tight.write(true, automatic);
        check(read_payload(tight.entry) == tight.expected(automatic), "tight: written in frag");
        const auto before = read_file(tight.entry);
        check(throws([&] { (void)run(tight.entry, ExpertLayout::Published); }, "--out"),
              "tight: published does not fit and names --out");
        check(read_file(tight.entry) == before &&
                  !std::filesystem::exists(cv::relayout_journal(tight.entry)),
              "tight: the refused relayout left the artifact as it was");
        check(read_payload(tight.entry) == tight.expected(automatic), "tight: still readable");
    }
}

// An artifact put back from a copy made before an interrupted relayout (same id, every chunk as
// it was) with the journal left beside it: the resume must refuse, in the chunk phase and in the
// directory phase, and the copy is usable once the journal is deleted.
void test_restored_copy() {
    using cv::ExpertLayout;
    const TestArtifact a(0x74);
    a.write();
    const auto entry = read_file(a.entry), part = read_file(a.dir / "weights-2.bin");
    const auto restore = [&] {
        write_file(a.entry, entry);
        write_file(a.dir / "weights-2.bin", part);
    };
    interrupt(a.entry, ExpertLayout::Auto); // several chunks written, then one in part
    restore();
    check(throws([&] { (void)run(a.entry, ExpertLayout::Auto, 4096); }, "no longer holds"),
          "a copy restored in the chunk phase is refused");
    {
        cv::RelayoutOptions options;
        options.artifact            = a.entry;
        options.layout              = ExpertLayout::Auto;
        options.interrupt_directory = true;
        std::filesystem::remove(cv::relayout_journal(a.entry));
        check(throws([&] { (void)cv::relayout(options); }, "interrupted"),
              "the directory interruption");
    }
    restore();
    check(throws([&] { (void)run(a.entry, ExpertLayout::Auto); }, "no longer holds"),
          "a copy restored in the directory phase is refused");
    std::filesystem::remove(cv::relayout_journal(a.entry));
    check(read_payload(a.entry) == a.expected({}), "the restored copy reads as published");
    // A new artifact written where a stale journal lies removes it.
    const auto copy = a.dir / "fresh" / "model.ninferno";
    std::filesystem::create_directories(copy.parent_path());
    write_file(cv::relayout_journal(copy), {1, 2, 3});
    (void)run(a.entry, ExpertLayout::Auto, 64ULL << 20, copy);
    check(!std::filesystem::exists(cv::relayout_journal(copy)),
          "the writer removes a stale journal at its path");
    check(read_payload(copy).size() == a.payload.size(), "the new artifact opens");
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_maps();
        test_random_rows();
        test_geometry();
        test_rule();
        test_relayout();
        test_restored_copy();
        for (int i = 1; i < argc; ++i) { test_gguf(argv[i]); }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", error.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "ggml_frag: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("ggml_frag: all checks passed\n");
    return 0;
}
