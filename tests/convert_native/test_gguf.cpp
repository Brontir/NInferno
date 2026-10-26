// GGUF container parsing: metadata of every value type, tensor infos, alignment, and the
// structural errors a damaged file must raise.

#include "convert/checks.h"
#include "convert/ggml_quants.h"
#include "convert/gguf.h"
#include "gguf_writer.h"
#include "support/temp_dir.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <vector>

using namespace ninferno::convert;
using ninferno::convert::testing::GgufWriter;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

GgufFile parse(const std::vector<std::uint8_t>& bytes) {
    return parse_gguf(reinterpret_cast<const std::byte*>(bytes.data()), bytes.size(), "test.gguf");
}

bool throws(const std::function<void()>& body, const std::string& fragment) {
    try {
        body();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(fragment) != std::string::npos;
    }
    return false;
}

std::vector<std::uint8_t> q8_block(bool unit_scale) {
    std::vector<std::uint8_t> b(34, 0);
    const std::uint16_t d = unit_scale ? 0x3C00 : 0x3800;
    std::memcpy(b.data(), &d, 2);
    for (int j = 0; j < 32; ++j) { b[2 + j] = static_cast<std::uint8_t>(j); }
    return b;
}

void test_metadata_and_tensors() {
    GgufWriter w(64);
    w.kv_str("general.architecture", "qwen35");
    w.kv_u32("general.alignment", 64);
    w.kv_u32("qwen35.block_count", 3);
    w.kv_i32("test.negative", -5);
    w.kv_u64("test.big", 1ULL << 40);
    w.kv_f32("qwen35.attention.layer_norm_rms_eps", 1e-6F);
    w.kv_bool("test.flag", true);
    w.kv_arr_str("tokenizer.ggml.tokens", {"a", "b", "\xc3\xa9"});
    w.kv_arr_i32("tokenizer.ggml.token_type", {1, 3, 4});
    w.kv_arr_bool("test.pattern", {true, false});
    w.kv_arr_f32("test.floats", {0.1F, 2.5F});
    std::vector<std::uint8_t> f32(4 * 6);
    for (int i = 0; i < 6; ++i) {
        const float v = static_cast<float>(i) - 2.5F;
        std::memcpy(f32.data() + 4 * i, &v, 4);
    }
    w.tensor("blk.0.attn_norm.weight", {6}, ggml::F32, f32);
    auto q8 = q8_block(true);
    auto q8b = q8_block(false);
    q8.insert(q8.end(), q8b.begin(), q8b.end());
    w.tensor("blk.0.ffn_up.weight", {32, 2}, ggml::Q8_0, q8);
    const auto bytes = w.finish();
    const auto file  = parse(bytes);
    check(file.version == 3, "version");
    check(file.alignment == 64 && file.data_offset % 64 == 0, "general.alignment is honoured");
    check(file.metadata.at("general.architecture") == "qwen35", "string value");
    check(file.metadata.at("qwen35.block_count") == 3, "u32 value");
    check(file.metadata.at("test.negative") == -5, "i32 value");
    check(file.metadata.at("test.big") == (1ULL << 40), "u64 value");
    check(file.metadata.at("qwen35.attention.layer_norm_rms_eps").get<double>() == 1e-06,
          "an F32 value reads as its shortest decimal (1e-06)");
    check(file.metadata.at("test.flag") == true, "bool value");
    check(file.metadata.at("tokenizer.ggml.tokens").size() == 3 &&
              file.metadata.at("tokenizer.ggml.tokens")[2] == "\xc3\xa9",
          "string array");
    check(file.metadata.at("test.pattern") == nlohmann::json::array({true, false}), "bool array");
    check(file.metadata.at("test.floats")[0].get<double>() == 0.1, "f32 array, shortest decimal");
    check(file.types.at("tokenizer.ggml.token_type").first == GgufValueType::I32 &&
              file.types.at("tokenizer.ggml.token_type").second,
          "array element types are recorded");
    check(file.tensors.size() == 2, "tensor count");
    const auto& up = file.tensors[1];
    check(up.name == "blk.0.ffn_up.weight" && up.type == ggml::Q8_0 && up.bytes == 68 &&
              up.shape() == std::vector<std::uint64_t>{2, 32},
          "tensor info: ne reversed into a row-major shape, bytes from blocks");
    std::vector<float> values(64);
    ggml::dequantize(up.type,
                     reinterpret_cast<const std::byte*>(bytes.data()) + file.data_offset + up.offset,
                     64, values.data());
    check(values[5] == 5.0F && values[32 + 5] == 2.5F, "tensor data is found at its offset");
}

void test_errors() {
    GgufWriter good;
    good.kv_str("general.architecture", "x");
    good.tensor("t", {32}, ggml::Q8_0, q8_block(true));
    const auto bytes = good.finish();
    check(throws([&] { parse(std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + 40)); },
                 "truncated"),
          "a truncated file is rejected");
    auto magic = bytes;
    magic[0]   = 'X';
    check(throws([&] { parse(magic); }, "not a GGUF"), "a wrong magic is rejected");
    check(throws([&] { parse(good.finish(1)); }, "unsupported GGUF version 1"),
          "version 1 is rejected");
    auto big = bytes;
    std::uint32_t be = 0x03000000;
    std::memcpy(big.data() + 4, &be, 4);
    check(throws([&] { parse(big); }, "big-endian"), "a big-endian file is named as such");
    GgufWriter unknown;
    unknown.tensor("t", {32}, 4, std::vector<std::uint8_t>(18));
    check(throws([&] { parse(unknown.finish()); }, "unknown ggml type 4"),
          "an unknown tensor type is rejected");
    GgufWriter partial;
    partial.tensor("t", {48}, ggml::Q8_0, std::vector<std::uint8_t>(51));
    check(throws([&] { parse(partial.finish()); }, "whole q8_0 blocks"),
          "a row that does not fill whole blocks is rejected");
    auto cut = bytes;
    cut.resize(cut.size() - 1);
    check(throws([&] { parse(cut); }, "beyond the end"), "tensor data past the file end");
    // Four dimensions of 2^20 each pass the per-dimension bound; their product wraps to 0.
    GgufWriter wrap;
    wrap.tensor("t", {1ULL << 20, 1ULL << 20, 1ULL << 20, 1ULL << 20}, ggml::F32,
                std::vector<std::uint8_t>(4));
    check(throws([&] { parse(wrap.finish()); }, "shape overflows"),
          "a shape whose product wraps is rejected");
    GgufWriter dup;
    dup.kv_u32("a", 1);
    dup.kv_u32("a", 2);
    check(throws([&] { parse(dup.finish()); }, "duplicate metadata key"), "duplicate keys");
    // gguf-split --no-tensor-first-split writes a first shard of metadata alone, ending right
    // after it with no padding to the alignment: it holds no tensor data to place.
    GgufWriter head_only;
    head_only.kv_u32("split.count", 2);
    head_only.kv_str("general.architecture", "qwen35");
    auto metadata_only = head_only.finish();
    while (!metadata_only.empty() && metadata_only.back() == 0) { metadata_only.pop_back(); }
    check(metadata_only.size() % 32 != 0 && parse(metadata_only).tensors.empty() &&
              parse(metadata_only).metadata.at("split.count") == 2,
          "a metadata-only split shard is rejected");
}

// The metadata from the file's first bytes: a head that holds the tensor infos parses, its
// tensors checked against the whole file's size; a shorter one throws GgufTruncated.
void test_head() {
    GgufWriter w(64);
    w.kv_str("general.architecture", "dflash");
    w.kv_arr_str("tokenizer.ggml.tokens", {"a", "b", "c"});
    w.tensor("t", {64}, ggml::Q8_0, [] {
        auto two     = q8_block(true);
        const auto b = q8_block(false);
        two.insert(two.end(), b.begin(), b.end());
        return two;
    }());
    const auto bytes = w.finish();
    const auto head  = [&](std::size_t size, std::uint64_t file_size) {
        return parse_gguf_head(reinterpret_cast<const std::byte*>(bytes.data()), size, file_size,
                               "head.gguf");
    };
    const auto whole = parse(bytes);
    const auto file  = head(static_cast<std::size_t>(whole.data_offset), bytes.size());
    check(file.metadata == whole.metadata && file.tensors.size() == 1 &&
              file.tensors[0].bytes == whole.tensors[0].bytes &&
              file.data_offset == whole.data_offset,
          "a head through the tensor infos reads what the whole file does");
    bool truncated = false;
    try {
        (void)head(40, bytes.size());
    } catch (const GgufTruncated&) {
        truncated = true;
    }
    check(truncated, "a head ending inside the metadata throws GgufTruncated");
    check(throws([&] { (void)head(static_cast<std::size_t>(whole.data_offset), bytes.size() - 1); },
                 "beyond the end"),
          "tensors are checked against the whole file's size");

    // A head that ends inside a long array of strings, and an array whose count the bytes held
    // cannot contain (a damaged count is refused before anything is reserved for it).
    GgufWriter strings;
    strings.kv_arr_str("tokenizer.ggml.tokens", std::vector<std::string>(4096, "token"));
    const auto long_array = strings.finish();
    const auto truncates  = [](const std::function<void()>& body) {
        try {
            body();
        } catch (const GgufTruncated&) {
            return true;
        } catch (const std::exception&) {}
        return false;
    };
    check(truncates([&] {
              (void)parse_gguf_head(reinterpret_cast<const std::byte*>(long_array.data()), 4096,
                                    long_array.size(), "strings.gguf");
          }),
          "a head ending inside a string array throws GgufTruncated");
    GgufWriter counted;
    counted.kv_arr_i32("test.oversized", {1, 2, 3});
    auto oversized    = counted.finish();
    const auto key    = std::string("test.oversized");
    const auto at     = std::search(oversized.begin(), oversized.end(), key.begin(), key.end());
    const auto offset = static_cast<std::size_t>(at - oversized.begin()) + key.size() + 8;

    const std::uint64_t huge = 1ULL << 40;
    std::memcpy(oversized.data() + offset, &huge, sizeof(huge));
    check(truncates([&] {
              (void)parse_gguf_head(reinterpret_cast<const std::byte*>(oversized.data()),
                                    oversized.size(), 1ULL << 50, "oversized.gguf");
          }),
          "an array count past the bytes held throws GgufTruncated");
}

void test_v2() {
    GgufWriter w;
    w.kv_str("general.architecture", "gemma4");
    const auto file = parse(w.finish(2));
    check(file.version == 2 && file.tensors.empty(), "version 2 files parse");
}

// A GGUF's rebuilt tokenizer files stand in for the directory's unless those are this model's:
// another model's tokenizer.json beside the file is not read.
void test_resources_beside() {
    namespace fs  = std::filesystem;
    const TempDir scratch("ninferno_test_gguf_beside");
    const auto& dir  = scratch.path();
    const auto write = [&](const char* name, const std::string& text) {
        std::ofstream(dir / name, std::ios::binary) << text;
    };
    const std::string own =
        R"({"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2},"merges":[]}})";
    const ResourceFallbacks fallbacks = {{"tokenizer.json", own},
                                         {"tokenizer_config.json", "{}"},
                                         {"chat_template.jinja", "{{ messages }}"},
                                         {"generation_config.json", R"({"eos_token_id":2})"}};
    const auto tokenizer_bytes = [](const Model& model) {
        for (const auto& r : model.resources) {
            if (r.id == "resource/text/tokenizer.json") { return r.bytes; }
        }
        return std::string();
    };
    write("tokenizer.json",
          R"({"added_tokens":[],"model":{"type":"BPE","vocab":{"x":0,"y":1,"z":2},"merges":[]}})");
    write("tokenizer_config.json", R"({"model_max_length":8})");
    Model foreign;
    load_resources(foreign, dir, 3, {}, nullptr, fallbacks);
    check(tokenizer_bytes(foreign) == own, "another model's tokenizer.json beside the GGUF is not read");
    check(std::any_of(foreign.resource_notes.begin(), foreign.resource_notes.end(),
                      [](const std::string& note) {
                          return note.find("another model's") != std::string::npos;
                      }),
          "the skipped files are noted");
    const std::string same =
        R"({"added_tokens": [], "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []}})";
    write("tokenizer.json", same);
    Model matching;
    load_resources(matching, dir, 3, {}, nullptr, fallbacks);
    check(tokenizer_bytes(matching) == same, "this model's tokenizer.json beside the GGUF is read");
}

} // namespace

// Which GGUF files beside a model are not the model (listings on the hub).
void test_auxiliary_files() {
    for (const char* aux : {"mmproj-Qwen3.8-27B-F16.gguf", "imatrix.gguf", "imatrix_unsloth.gguf",
                            "NVIDIA-Nemotron-imatrix.gguf", "mtp-NVIDIA-Nemotron-Q4_0.gguf",
                            "Qwen3.6-35B-MTP-ONLY-Q8_0.gguf", "dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
                            "sub/DFlash2-Qwen3.8-27B-Q4_0.gguf", "eagle3-gpt-oss-20b-Q8_0.gguf"}) {
        check(gguf_auxiliary_file(aux), std::string("auxiliary: ") + aux);
    }
    for (const char* model : {"Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
                              "Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf", "gemma-4-12b-it-Q4_K_M.gguf"}) {
        check(!gguf_auxiliary_file(model), std::string("a model: ") + model);
    }
}

// The repository a GGUF's metadata names as its source: the first key that names one, org/name.
void test_base_repository() {
    using J = nlohmann::json;
    const auto of = [](const J& md) { return gguf_metadata_base_repository(md); };
    check(of(J{{"general.base_model.0.repo_url", "https://huggingface.co/Qwen/Qwen3.5-9B"}}) ==
              std::optional<std::string>("Qwen/Qwen3.5-9B"),
          "a base model's address");
    check(of(J{{"general.base_model.0.repo_url", "https://hf.co/org/m/tree/main/"}}) ==
              std::optional<std::string>("org/m"),
          "a /tree tail is not the repository");
    check(of(J{{"general.base_model.0.repo_url", "https://github.com/org/m"},
               {"general.source.huggingface.repository", "org/src"}}) ==
              std::optional<std::string>("org/src"),
          "another site's address names none: the next key");
    check(of(J{{"general.source.url", "https://huggingface.co/org/s"}}) ==
              std::optional<std::string>("org/s"),
          "the source address");
    check(!of(J{{"general.base_model.0.repo_url", "org"}}) && !of(J{{"general.base_model.0.repo_url", 3}}) &&
              !of(J::object()),
          "none named");
}

int main() {
    test_base_repository();
    test_auxiliary_files();
    test_resources_beside();
    test_metadata_and_tensors();
    test_errors();
    test_v2();
    test_head();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("gguf reader: ok\n");
    return 0;
}
