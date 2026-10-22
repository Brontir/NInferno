// The content digests the converter seals into an artifact's directory (src/convert/writer.h,
// artifact/source_identity.h): the SHA-256 of each object's bytes, read back before publishing,
// in direct and buffered writes, single and split files; the source identity follows the bytes,
// not the artifact id. CPU.
//
//   test_content_digests                         the self test
//   test_content_digests --rewrite IN OUT MAX    writes IN's description and payload again with the
//                                                C++ writer (for content_digests_interop.py)
//   test_content_digests --identity PATH         prints PATH's source identity, or "none"

#include "artifact/reader.h"
#include "artifact/source_identity.h"
#include "convert/pyjson.h"
#include "convert/writer.h"
#include "core/sha256.h"

#include <cstdio>
#include <optional>
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninferno::convert::OJson;
namespace fs = std::filesystem;

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAILED: " << what << '\n'; }
    return ok ? 0 : 1;
}

// Writes `description` (no "files"; provenance["content_sha256"] is set to placeholders here) and
// `payload` to `path`, sealing the digests at finish.
void write_artifact(const fs::path& path, OJson description, const std::vector<char>& payload,
                    std::uint64_t max_file_bytes, bool direct) {
    using namespace ninferno::convert;
    description["provenance"]["content_sha256"] =
        content_digest_placeholders(description["objects"]);
    auto layout = layout_artifact(path.filename().string(), py_dumps_compact(description),
                                  payload.size(), max_file_bytes);
    WriterOptions options;
    options.direct = direct;
    for (const auto& object : description["objects"]) {
        const auto offset = object["offset"].get<std::uint64_t>();
        options.written.push_back({offset, offset + object["bytes"].get<std::uint64_t>()});
    }
    ArtifactWriter writer(path, std::move(layout), options);
    for (const auto& object : description["objects"]) {
        const auto offset = object["offset"].get<std::uint64_t>();
        writer.write(offset, payload.data() + offset, object["bytes"].get<std::size_t>());
    }
    writer.finish([&] {
        description["provenance"]["content_sha256"] =
            content_digests(writer, description["objects"], 2);
        return layout_artifact(path.filename().string(), py_dumps_compact(description),
                               payload.size(), max_file_bytes)
            .directory;
    });
}

void remove_artifact(const fs::path& path) {
    std::error_code ignored;
    for (const auto& entry : fs::directory_iterator(path.parent_path(), ignored)) {
        if (entry.path().filename().string().starts_with(path.filename().string())) {
            fs::remove(entry.path(), ignored);
        }
    }
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

const char* kDescription = R"({"components":{"text":{"config":{}}},"objects":[
  {"id":"matrix","shape":[130,130],"format":"bf16","layout":"contiguous_le_v1","offset":0,
   "bytes":33800,"kind":"tensor"},
  {"id":"tail","shape":[64],"format":"bf16","layout":"contiguous_le_v1","offset":36864,
   "bytes":128,"kind":"tensor"}],
  "bindings":{"whole":{"object":"matrix"},"tail":{"object":"tail"}},"uses":[],"metadata":{},
  "provenance":{"converter":"ninferno-v3","recipe":"synthetic","inputs":[{"role":"base",
  "files":[{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}]}})";

int self_test() {
    int failures   = 0;
    const auto dir = fs::temp_directory_path() / ("ninferno-digests-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::vector<char> payload(36864 + 128);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<char>((i * 37 + 11) % 251);
    }
    const auto description = OJson::parse(kDescription);
    const auto expected    = [&](std::uint64_t offset, std::uint64_t bytes) {
        return ninferno::sha256_hex(std::string_view(payload.data() + offset, bytes));
    };
    std::optional<std::string> first;
    for (const bool direct : {false, true}) {
        for (const std::uint64_t limit : {std::uint64_t{1'000'000}, std::uint64_t{24576}}) {
            const auto label = std::string(direct ? "direct" : "buffered") +
                               (limit < 100'000 ? ", split" : ", one file");
            const auto path = dir / "a.ninferno";
            remove_artifact(path);
            try {
                write_artifact(path, description, payload, limit, direct);
            } catch (const std::exception& e) {
                if (direct && std::string(e.what()).find("direct") != std::string::npos) {
                    std::cerr << "note: " << label << " skipped (" << e.what() << ")\n";
                    continue;
                }
                throw;
            }
            const ninferno::artifact::Reader reader(path);
            const auto& digests = reader.directory().provenance.at("content_sha256");
            failures += check(digests.at("matrix") == expected(0, 33800) &&
                                  digests.at("tail") == expected(36864, 128),
                              label + ": the sealed digests are the SHA-256 of the bytes");
            const auto id = ninferno::artifact::source_identity(reader.directory());
            failures += check(id && ninferno::artifact::is_source_identity(*id),
                              label + ": an artifact with digests and inputs has an identity");
            if (!first) { first = id; }
            failures += check(id == first, label + ": the same bytes, another artifact id and "
                                                   "another file split: the same identity");
        }
    }
    // One byte more or less: another identity.
    auto changed = payload;
    changed[100] ^= 1;
    const auto path = dir / "b.ninferno";
    remove_artifact(path);
    write_artifact(path, description, changed, 1'000'000, false);
    failures += check(ninferno::artifact::source_identity(path) != first,
                      "a changed byte that keeps every shape moves the identity");
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
            const ninferno::artifact::Reader reader(in);
            std::vector<char> payload(reader.directory().payload_bytes);
            const auto bytes = reader.read_range(0, payload.size());
            std::memcpy(payload.data(), bytes.data(), payload.size());
            write_artifact(argv[3], description, payload, std::stoull(argv[4]), false);
            return 0;
        }
        if (argc == 3 && std::strcmp(argv[1], "--identity") == 0) {
            std::cout << ninferno::artifact::source_identity(fs::path(argv[2])).value_or("none")
                      << '\n';
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
