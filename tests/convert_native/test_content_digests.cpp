// The content digests convert() seals into an artifact's directory (src/convert/writer.h,
// artifact/source_identity.h) and the source identity they key. CPU.
//
// - convert() on a synthetic dense checkpoint, with direct output (the default) and with
//   --consume-source (mapped output): it records the inputs (a local file by its content) and a
//   digest per object, the SHA-256 of the object's bytes; a second conversion (another artifact
//   id) keeps the identity, one changed weight byte moves it. A direct-write failure fails.
// - A partial entry the writer sealed before a crash (sealed digests, not yet published) still
//   resumes, and is sealed again; any other change to its directory refuses the resume.
// - A relayout to a new artifact seals the digests again; one in place drops them (no identity).
//
//   test_content_digests                         the self test
//   test_content_digests --rewrite IN OUT MAX    writes IN's description and payload again with the
//                                                C++ writer (for content_digests_interop.py)
//   test_content_digests --identity PATH         prints PATH's source identity, or "none"

#include "artifact/reader.h"
#include "artifact/source_identity.h"
#include "convert/convert.h"
#include "convert/pyjson.h"
#include "convert/relayout.h"
#include "convert/writer.h"
#include "core/sha256.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using ninferno::convert::OJson;
namespace fs  = std::filesystem;
namespace art = ninferno::artifact;
namespace cv  = ninferno::convert;

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAILED: " << what << '\n'; }
    return ok ? 0 : 1;
}

// Writes `description` (no "files"; its provenance["content_sha256"] set to placeholders here)
// and `payload` with the C++ writer, sealing the digests at finish.
void write_artifact(const fs::path& path, OJson description, const std::vector<char>& payload,
                    std::uint64_t max_file_bytes) {
    description["provenance"]["content_sha256"] =
        cv::content_digest_placeholders(description["objects"]);
    auto layout = cv::layout_artifact(path.filename().string(), cv::py_dumps_compact(description),
                                      payload.size(), max_file_bytes);
    cv::ArtifactWriter writer(path, std::move(layout));
    for (const auto& object : description["objects"]) {
        const auto offset = object["offset"].get<std::uint64_t>();
        writer.write(offset, payload.data() + offset, object["bytes"].get<std::size_t>());
    }
    writer.finish([&] {
        description["provenance"]["content_sha256"] =
            cv::content_digests(writer, description["objects"], 2);
        return cv::layout_artifact(path.filename().string(), cv::py_dumps_compact(description),
                                   payload.size(), max_file_bytes)
            .directory;
    });
}

OJson raw_directory(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    char header[32];
    in.read(header, sizeof header);
    std::uint64_t length = 0;
    for (int i = 0; i < 8; ++i) {
        length |= std::uint64_t(static_cast<unsigned char>(header[8 + i])) << (8 * i);
    }
    std::string text(length, ' ');
    in.read(text.data(), static_cast<std::streamsize>(length));
    while (!text.empty() && text.back() == ' ') { text.pop_back(); }
    return OJson::parse(text);
}

// Whether every object's sealed digest is the SHA-256 of its bytes.
bool digests_hold(const fs::path& path) {
    const art::Reader reader(path);
    const auto& digests = reader.directory().provenance.at("content_sha256");
    for (const auto& object : reader.directory().objects) {
        const auto bytes = reader.read_range(art::object_offset(object), art::object_bytes(object));
        const auto want  = ninferno::sha256_hex(
            std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
        if (digests.at(art::object_id(object)) != want) { return false; }
    }
    return digests.size() == reader.directory().objects.size();
}

// --- a synthetic dense checkpoint (LlamaForCausalLM, BF16) -------------------------------------

void write_safetensors(const fs::path& path,
                       const std::vector<std::pair<std::string, std::vector<std::uint64_t>>>& specs,
                       unsigned seed) {
    std::string header   = "{";
    std::uint64_t offset = 0;
    std::vector<std::uint64_t> sizes;
    for (const auto& [name, shape] : specs) {
        std::uint64_t count = 1;
        std::string dims;
        for (const auto d : shape) {
            count *= d;
            dims += (dims.empty() ? "" : ",") + std::to_string(d);
        }
        if (header.size() > 1) { header += ","; }
        header += "\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[" + dims +
                  "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(offset + 2 * count) + "]}";
        offset += 2 * count;
        sizes.push_back(count);
    }
    header += "}";
    while (header.size() % 8) { header += " "; }
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::uint32_t state = seed;
    for (const auto count : sizes) {
        for (std::uint64_t i = 0; i < count; ++i) {
            state = state * 1664525u + 1013904223u;
            // A small BF16 value: sign, exponent 120..127, a random mantissa.
            const std::uint16_t bits = static_cast<std::uint16_t>(
                ((state >> 31) << 15) | ((120u + ((state >> 24) & 7u)) << 7) |
                ((state >> 8) & 0x7Fu));
            out.write(reinterpret_cast<const char*>(&bits), 2);
        }
    }
}

void write_checkpoint(const fs::path& dir, unsigned seed) {
    fs::create_directories(dir);
    const std::uint64_t h = 64, inter = 128, vocab = 256, heads = 2, kv = 1, head_dim = 32;
    std::ofstream(dir / "config.json") << R"({"architectures": ["LlamaForCausalLM"],
        "model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 32, "vocab_size": 256, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-05, "rope_theta": 10000.0, "tie_word_embeddings": false,
        "hidden_act": "silu", "torch_dtype": "bfloat16", "bos_token_id": 1, "eos_token_id": 2})";
    std::string vocab_json = "{";
    for (std::uint64_t i = 0; i < vocab; ++i) {
        vocab_json +=
            (i ? "," : "") + std::string("\"t") + std::to_string(i) + "\":" + std::to_string(i);
    }
    vocab_json += "}";
    std::ofstream(dir / "tokenizer.json")
        << R"({"version": "1.0", "added_tokens": [], "model": {"type": "BPE", "vocab": )"
        << vocab_json << R"(, "merges": []}})";
    std::ofstream(dir / "tokenizer_config.json") << R"({"bos_token": "t1", "eos_token": "t2"})";
    std::ofstream(dir / "chat_template.jinja")
        << "{% for message in messages %}{{ message.content }}{% endfor %}";
    std::ofstream(dir / "generation_config.json") << R"({"bos_token_id": 1, "eos_token_id": 2})";
    const std::string p = "model.layers.0.";
    write_safetensors(dir / "model.safetensors",
                      {{"model.embed_tokens.weight", {vocab, h}},
                       {p + "input_layernorm.weight", {h}},
                       {p + "self_attn.q_proj.weight", {heads * head_dim, h}},
                       {p + "self_attn.k_proj.weight", {kv * head_dim, h}},
                       {p + "self_attn.v_proj.weight", {kv * head_dim, h}},
                       {p + "self_attn.o_proj.weight", {h, heads * head_dim}},
                       {p + "post_attention_layernorm.weight", {h}},
                       {p + "mlp.gate_proj.weight", {inter, h}},
                       {p + "mlp.up_proj.weight", {inter, h}},
                       {p + "mlp.down_proj.weight", {h, inter}},
                       {"model.norm.weight", {h}},
                       {"lm_head.weight", {vocab, h}}},
                      seed);
}

cv::ConvertResult convert(const fs::path& model, const fs::path& out, bool consume) {
    cv::ConvertOptions options;
    options.model          = model;
    options.recipe         = "dense_bf16";
    options.out            = out;
    options.consume_source = consume;
    options.threads        = 2;
    return cv::convert(options);
}

std::string artifact_id(const fs::path& path) {
    const art::Reader reader(path);
    std::string out;
    for (const auto b : reader.artifact_id()) {
        char hex[3];
        std::snprintf(hex, sizeof hex, "%02x", static_cast<unsigned>(b));
        out += hex;
    }
    return out;
}

int convert_test(const fs::path& dir) {
    int failures = 0;
    write_checkpoint(dir / "ckpt", 7);
    const auto a = dir / "a.ninferno";
    (void)convert(dir / "ckpt", a, false); // direct output; a direct-write failure throws
    const auto id = art::source_identity(a);
    failures +=
        check(id && art::is_source_identity(*id), "convert(): a conversion has a source identity");
    failures += check(digests_hold(a), "convert(): every sealed digest is its object's SHA-256");
    {
        const art::Reader reader(a);
        const auto& inputs = reader.directory().provenance.at("inputs");
        const auto weights = ninferno::file_sha256_hex(dir / "ckpt" / "model.safetensors");
        bool listed        = false;
        for (const auto& file : inputs.at(0).at("files")) {
            listed = listed || (file.size() == 1 && file.at("sha256") == *weights);
        }
        failures += check(inputs.at(0).at("role") == "base" && listed,
                          "convert(): the inputs list the local weights by their content alone");
    }
    // Again, from a copy elsewhere: another artifact id, the same bytes, the same identity.
    fs::copy(dir / "ckpt", dir / "copy", fs::copy_options::recursive);
    const auto b = dir / "b.ninferno";
    (void)convert(dir / "copy", b, false);
    failures += check(artifact_id(a) != artifact_id(b) && art::source_identity(b) == id,
                      "a reconversion (another id, another path) keeps the identity");
    // --consume-source writes through a mapping: the same bytes, digests and identity.
    fs::copy(dir / "ckpt", dir / "consumed", fs::copy_options::recursive);
    const auto c = dir / "c.ninferno";
    (void)convert(dir / "consumed", c, true);
    failures += check(digests_hold(c) && art::source_identity(c) == id,
                      "--consume-source: the same digests and identity");
    // One weight byte changed (every shape kept): another identity.
    write_checkpoint(dir / "changed", 7);
    {
        std::fstream f(dir / "changed" / "model.safetensors",
                       std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(0, std::ios::end);
        const auto size = static_cast<std::streamoff>(f.tellg());
        f.seekp(size - 7);
        f.put('\x3c');
    }
    const auto d = dir / "d.ninferno";
    (void)convert(dir / "changed", d, false);
    failures += check(art::source_identity(d) && art::source_identity(d) != id,
                      "one changed weight byte moves the identity");
    return failures;
}

// --- a partial entry sealed before a crash ------------------------------------------------------

const char* kDescription = R"({"components":{"text":{"config":{}}},"objects":[
  {"id":"matrix","shape":[130,130],"format":"bf16","layout":"contiguous_le_v1","offset":0,
   "bytes":33800,"kind":"tensor"},
  {"id":"tail","shape":[64],"format":"bf16","layout":"contiguous_le_v1","offset":36864,
   "bytes":128,"kind":"tensor"}],
  "bindings":{"whole":{"object":"matrix"},"tail":{"object":"tail"}},"uses":[],"metadata":{},
  "provenance":{"converter":"ninferno-v3","recipe":"synthetic","inputs":[{"role":"base",
  "files":[{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}]}})";

int resume_test(const fs::path& dir) {
    int failures = 0;
    std::vector<char> payload(36864 + 128);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<char>((i * 37 + 11) % 251);
    }
    const auto path  = dir / "r.ninferno";
    auto description = OJson::parse(kDescription);
    description["provenance"]["content_sha256"] =
        cv::content_digest_placeholders(description["objects"]);
    const auto text  = cv::py_dumps_compact(description);
    const auto write = [&](std::optional<art::ArtifactId> resume) {
        cv::WriterOptions options;
        options.named_partial = true;
        options.resume        = resume;
        auto writer           = std::make_unique<cv::ArtifactWriter>(
            path, cv::layout_artifact(path.filename().string(), text, payload.size(), 1'000'000),
            options);
        writer->keep_partial();
        for (const auto& object : description["objects"]) {
            const auto offset = object["offset"].get<std::uint64_t>();
            writer->write(offset, payload.data() + offset, object["bytes"].get<std::size_t>());
        }
        return writer;
    };
    // A run that sealed its directory and stopped before publishing (the crash window).
    auto first                             = write(std::nullopt);
    const auto id                          = first->artifact_id();
    auto sealed                            = description;
    sealed["provenance"]["content_sha256"] = cv::content_digests(*first, description["objects"], 1);
    first->abort(); // keep_partial: the partial file stays, as a crash leaves it
    first.reset();
    const auto partial = cv::ArtifactWriter::partial_path(path);
    const auto sealed_text =
        cv::layout_artifact(path.filename().string(), cv::py_dumps_compact(sealed), payload.size(),
                            1'000'000)
            .directory;
    {
        const int fd = ::open(partial.c_str(), O_WRONLY);
        failures += check(fd >= 0 && ::pwrite(fd, sealed_text.data(), sealed_text.size(), 32) ==
                                         static_cast<ssize_t>(sealed_text.size()),
                          "the sealed directory is written into the partial entry");
        ::close(fd);
    }
    // The resume takes it, and seals again.
    try {
        auto again = write(id);
        again->finish([&] {
            auto d                            = description;
            d["provenance"]["content_sha256"] = cv::content_digests(*again, d["objects"], 1);
            return cv::layout_artifact(path.filename().string(), cv::py_dumps_compact(d),
                                       payload.size(), 1'000'000)
                .directory;
        });
        failures += check(digests_hold(path), "a resumed sealed partial publishes its digests");
    } catch (const std::exception& e) {
        failures += check(false, std::string("a sealed partial entry resumes: ") + e.what());
    }
    // Any other change to the partial directory refuses the resume.
    fs::remove(path);
    auto third     = write(std::nullopt);
    const auto id3 = third->artifact_id();
    third->abort();
    third.reset();
    {
        const int fd = ::open(partial.c_str(), O_WRONLY);
        const char x = 'X';
        (void)::pwrite(fd, &x, 1, 40); // inside the directory, before the digests
        ::close(fd);
    }
    bool refused = false;
    try {
        (void)write(id3);
    } catch (const std::exception&) { refused = true; }
    failures += check(refused, "a partial entry whose directory differs otherwise is refused");
    std::error_code ignored;
    fs::remove(partial, ignored);
    return failures;
}

// --- relayout ------------------------------------------------------------------------------------

const char* kBank = R"({"components":{"text":{"config":{}}},"objects":[
  {"id":"text/layers/0/experts/gate","shape":[4,256],"format":"ggml_q4_k",
   "layout":"ggml_blocks_v1","offset":0,"bytes":576,"kind":"tensor"}],
  "bindings":{"text/layers/0/experts/gate":{"object":"text/layers/0/experts/gate"}},"uses":[],
  "metadata":{},"provenance":{"converter":"ninferno-v3","recipe":"synthetic","inputs":[
  {"role":"base","files":[{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}]}})";

int relayout_test(const fs::path& dir) {
    int failures = 0;
    std::vector<char> payload(576);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<char>((i * 53 + 3) % 241);
    }
    const auto source = dir / "bank.ninferno";
    write_artifact(source, OJson::parse(kBank), payload, 1'000'000);
    failures +=
        check(art::source_identity(source).has_value(), "the bank artifact has an identity");
    cv::RelayoutOptions to_new;
    to_new.artifact  = source;
    to_new.out       = dir / "bank.frag.ninferno";
    to_new.layout    = cv::ExpertLayout::Frag;
    const auto moved = cv::relayout(to_new);
    failures += check(moved.objects == 1, "relayout to a new artifact moves the bank");
    failures += check(digests_hold(*to_new.out) && art::source_identity(*to_new.out) &&
                          art::source_identity(*to_new.out) != art::source_identity(source),
                      "relayout to a new artifact seals the digests again: another identity");
    cv::RelayoutOptions in_place;
    in_place.artifact = source;
    in_place.layout   = cv::ExpertLayout::Frag;
    (void)cv::relayout(in_place);
    failures += check(!art::source_identity(source),
                      "relayout in place drops the digests: no identity until converted again");
    return failures;
}

int self_test() {
    const auto dir = fs::temp_directory_path() / ("ninferno-digests-" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    int failures = 0;
    for (const auto& [name, test] :
         {std::pair{"convert", &convert_test}, std::pair{"resume", &resume_test},
          std::pair{"relayout", &relayout_test}}) {
        try {
            failures += test(dir);
        } catch (const std::exception& e) {
            failures += check(false, std::string(name) + ": " + e.what());
        }
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::strcmp(argv[1], "--rewrite") == 0) {
            const fs::path in = argv[2];
            auto description  = raw_directory(in);
            description.erase("files");
            const art::Reader reader(in);
            std::vector<char> payload(reader.directory().payload_bytes);
            const auto bytes = reader.read_range(0, payload.size());
            std::memcpy(payload.data(), bytes.data(), payload.size());
            write_artifact(argv[3], description, payload, std::stoull(argv[4]));
            return 0;
        }
        if (argc == 3 && std::strcmp(argv[1], "--identity") == 0) {
            std::cout << art::source_identity(fs::path(argv[2])).value_or("none") << '\n';
            return 0;
        }
        const int failures = self_test();
        if (failures == 0) { std::cout << "content digests: all checks passed\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << '\n';
        return 1;
    }
}
