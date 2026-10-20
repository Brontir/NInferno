// A sharded safetensors checkpoint whose model.safetensors.index.json files tensors under the
// wrong shard of the same checkpoint (as ibm-granite/granite-4.2-3b publishes it): every tensor
// the index names is read from the shard that holds it, a tensor the shards hold but the index
// omits stays invisible, and a name the index lists that no shard holds fails on use. A header
// whose shape product wraps 64 bits (and so would match an empty byte range) is refused.

#include "convert/safetensors.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include "support/posix.h"

namespace fs = std::filesystem;
using ninferno::convert::Checkpoint;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// One F32 tensor of `count` copies of `value` per name.
void write_shard(const fs::path& path, const std::map<std::string, float>& tensors, int count) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (const auto& [name, value] : tensors) {
        (void)value;
        if (header.size() > 1) { header += ","; }
        const auto bytes = static_cast<std::uint64_t>(count) * 4;
        header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(count) +
                  "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(offset + bytes) + "]}";
        offset += bytes;
    }
    header += "}";
    while (header.size() % 8) { header += " "; }
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& [name, value] : tensors) {
        (void)name;
        for (int i = 0; i < count; ++i) { out.write(reinterpret_cast<const char*>(&value), 4); }
    }
}

float first_value(const Checkpoint& checkpoint, const std::string& name) {
    const auto& info = checkpoint.describe(name);
    float value      = 0;
    std::memcpy(&value, info.data, 4);
    return value;
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("ninferno_index_" + std::to_string(getpid()));
    fs::create_directories(dir);
    write_shard(dir / "model-00001-of-00002.safetensors", {{"a.weight", 1.0F}, {"b.weight", 2.0F}},
                16);
    write_shard(dir / "model-00002-of-00002.safetensors",
                {{"c.weight", 3.0F}, {"d.weight", 4.0F}, {"hidden.weight", 5.0F}}, 16);
    // b and c are filed under each other's shard; hidden is not listed; ghost is in no shard.
    std::ofstream(dir / "model.safetensors.index.json")
        << R"({"metadata":{},"weight_map":{)"
        << R"("a.weight":"model-00001-of-00002.safetensors",)"
        << R"("b.weight":"model-00002-of-00002.safetensors",)"
        << R"("c.weight":"model-00001-of-00002.safetensors",)"
        << R"("d.weight":"model-00002-of-00002.safetensors",)"
        << R"("ghost.weight":"model-00002-of-00002.safetensors"}})";
    try {
        const Checkpoint checkpoint(dir);
        for (const auto& [name, value] :
             std::map<std::string, float>{{"a.weight", 1}, {"b.weight", 2}, {"c.weight", 3},
                                          {"d.weight", 4}}) {
            check(checkpoint.has(name), name + " is visible");
            check(first_value(checkpoint, name) == value, name + " is read from its own shard");
        }
        check(!checkpoint.has("hidden.weight"), "a tensor the index omits is not visible");
        check(checkpoint.has("ghost.weight"), "a name the index lists is known");
        bool threw = false;
        try {
            (void)checkpoint.describe("ghost.weight");
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "a listed name no shard holds fails on use");
    } catch (const std::exception& error) {
        check(false, std::string("opening the checkpoint: ") + error.what());
    }
    fs::remove_all(dir);
    const fs::path wrap = fs::temp_directory_path() / ("ninferno_wrap_" + std::to_string(getpid()));
    fs::create_directories(wrap);
    {
        std::string header =
            R"({"t":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}})";
        while (header.size() % 8) { header += " "; }
        std::ofstream out(wrap / "model.safetensors", std::ios::binary);
        const std::uint64_t length = header.size();
        out.write(reinterpret_cast<const char*>(&length), 8);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    bool refused = false;
    try {
        const Checkpoint checkpoint(wrap);
        (void)checkpoint.readable("t");
    } catch (const std::runtime_error& error) {
        refused = std::string(error.what()).find("shape overflows") != std::string::npos;
    }
    check(refused, "a shape whose product wraps 64 bits is refused");
    fs::remove_all(wrap);
    if (failures) { return 1; }
    std::printf("checkpoint index: a misfiled index entry reads the shard that holds the tensor\n");
    return 0;
}
