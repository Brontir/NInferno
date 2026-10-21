#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/views.h"
#include "core/debug_poison.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>

namespace ninferno::test {
void materialization_cuda_errors(DeviceContext& device);
}

namespace {

using namespace ninferno;
using namespace ninferno::artifact;
using namespace ninferno::test::artifact_fixture;

void materialization(DeviceContext& device) {
    Fixture fixture;
    fixture.write(true);
    std::optional<MaterializedArtifact> backing;
    ParameterReference row;
    ObjectHandle quantized;
    ObjectHandle divisors;
    ObjectHandle resource;
    const std::byte* original_host = nullptr;
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        row       = binder.parameter("row", {1, 130});
        quantized = reader.find("q5");
        divisors  = reader.find("divisors");
        (void)binder.parameter("values", {2}, Residency::Host, QType::FP32);
        binder.require_device(divisors);
        resource      = binder.resource("text", "tokenizer.json");
        original_host = binder.host_object(divisors).data();
        backing.emplace(materialize(reader, std::move(binder).finish(), device));
    }
    const auto& stats = backing->stats();
    require(stats.device_object_count == 2 && stats.h2d_bytes == 536 &&
                stats.device_capacity_bytes == 776 && stats.retained_host_bytes == 13,
            "resident parents were duplicated or file gaps were allocated");
    require(backing->host_bytes(divisors).data() == original_host,
            "retained Host bytes moved after creating borrowed resource views");
    require(backing->host_bytes(resource).size() == 5,
            "Host resource was lost after Reader destruction");
    const auto view   = bind_view(row, *backing);
    const auto native = native_weight(view);
    require(native.n == 1 && native.k == 130 &&
                native.payload == backing->device_parent(quantized).data,
            "row view no longer refers to its owning parent");
    std::vector<std::byte> downloaded(528);
    CUDA_CHECK(cudaMemcpy(downloaded.data(), backing->device_parent(quantized).data,
                          downloaded.size(), cudaMemcpyDeviceToHost));
    require(std::equal(downloaded.begin(), downloaded.end(), fixture.payload.begin() + 256),
            "cross-file upload changed encoded parent bytes");
    std::array<std::byte, 8> values{};
    CUDA_CHECK(cudaMemcpy(values.data(), backing->device_parent(divisors).data, values.size(),
                          cudaMemcpyDeviceToHost));
    require(std::equal(values.begin(), values.end(), backing->host_bytes(divisors).begin()),
            "Host/device demand did not retain identical bytes");
    std::array<std::byte, 1> code{};
    CUDA_CHECK(cudaMemcpy(code.data(), native.qdata, 1, cudaMemcpyDeviceToHost));
    require(code[0] == std::byte{0x52}, "native row pointer addressed a different row");
}

// Residency::Mapped: the cross-file parent is read into pinned host memory and bound at a device
// address, without a device arena or an upload; one object is never both copied and mapped.
void mapped_residency(DeviceContext& device) {
    Fixture fixture;
    fixture.write(true);
    std::optional<MaterializedArtifact> backing;
    ParameterReference row;
    ObjectHandle quantized;
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        row       = binder.parameter("row", {1, 130}, Residency::Mapped);
        quantized = reader.find("q5");
        backing.emplace(materialize(reader, std::move(binder).finish(), device));
    }
    const auto& stats = backing->stats();
    require(stats.device_capacity_bytes == 0 && stats.h2d_bytes == 0 &&
                stats.device_object_count == 0 && stats.mapped_object_count == 1 &&
                stats.mapped_host_bytes == 528,
            "a mapped parent was placed in the device arena");
    const auto native = native_weight(bind_view(row, *backing));
    require(backing->has_device(quantized) &&
                native.payload == backing->device_parent(quantized).data,
            "the mapped row does not address its parent");
    std::vector<std::byte> read(528);
    CUDA_CHECK(cudaMemcpy(read.data(), backing->device_parent(quantized).data, read.size(),
                          cudaMemcpyDefault));
    require(std::equal(read.begin(), read.end(), fixture.payload.begin() + 256),
            "mapped bytes differ from the cross-file parent");
    std::array<std::byte, 1> code{};
    CUDA_CHECK(cudaMemcpy(code.data(), native.qdata, 1, cudaMemcpyDefault));
    require(code[0] == std::byte{0x52}, "mapped row pointer addressed a different row");

    Reader reader(fixture.entry);
    Binder binder(reader);
    (void)binder.parameter("matrix", {2, 130});
    (void)binder.parameter("row", {1, 130}, Residency::Mapped);
    rejects([&] { (void)std::move(binder).finish(); },
            "one object was demanded both in the arena and mapped");
}

void failure_and_host_only(DeviceContext& device) {
    Fixture fixture;
    fixture.write();
    {
        std::fstream part(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        part.seekp(16);
        part.put(0);
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        rejects([&] { (void)materialize(reader, std::move(binder).finish(), device); },
                "invalid required continuation was uploaded");
    }
    fixture.write();
    Reader reader(fixture.entry);
    Binder binder(reader);
    const auto resource = binder.resource("text", "tokenizer.json");
    auto backing        = materialize(reader, std::move(binder).finish(), device);
    require(backing.stats().device_capacity_bytes == 0 && backing.stats().h2d_bytes == 0 &&
                backing.host_bytes(resource).size() == 5,
            "Host-only demand allocated device weights");
    materialization(device);
}

void file_roundtrip(DeviceContext& device, const std::filesystem::path& path, bool writer_fixture) {
    std::optional<MaterializedArtifact> storage;
    std::map<std::size_t, std::vector<std::byte>> expected;
    std::vector<ParameterReference> parameters;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> placements; // arena offset, bytes
    {
        Reader reader(path);
        require(reader.file_bytes() < 32ULL * 1024 * 1024,
                "this complete-byte fixture check is limited to small artifacts");
        Binder binder(reader);
        for (const auto& [name, binding] : reader.directory().bindings) {
            auto shape = binding.whole_object
                             ? reader.directory().tensor(binding.parts.front().object).shape
                             : Shape{binding.elements};
            parameters.push_back(binder.parameter(name, std::move(shape)));
        }
        for (const auto& [key, use] : reader.directory().uses) {
            for (const auto& [name, auxiliary] : use.auxiliaries) {
                (void)binder.values(auxiliary);
            }
        }
        for (const auto& [name, component] : reader.directory().components) {
            for (const auto& [role, object] : component.resources) {
                (void)binder.resource(name, role);
            }
        }
        auto plan = std::move(binder).finish();
        placements.clear();
        for (const auto& placement : plan.device_objects) {
            placements.emplace_back(placement.offset, placement.bytes);
        }
        for (const auto& placement : plan.device_objects) {
            auto bytes = reader.read_object(placement.object);
            if (writer_fixture && reader.directory().tensor(placement.object).id == "matrix") {
                require(bytes.size() == 130 * 130 * 2, "Python writer changed tensor dimensions");
                for (std::size_t i = 0; i < bytes.size(); ++i) {
                    require(bytes[i] == std::byte((i * 37 + 11) % 251),
                            "C++ reading differs from known Python writer input bytes");
                }
            }
            expected.emplace(placement.object.index, std::move(bytes));
        }
        storage.emplace(materialize(reader, std::move(plan), device));
    }
    for (const auto& [index, bytes] : expected) {
        std::vector<std::byte> actual(bytes.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), storage->device_parent({index}).data, actual.size(),
                              cudaMemcpyDeviceToHost));
        require(actual == bytes, "small artifact upload differs from source bytes");
    }
    for (const auto& parameter : parameters) {
        const auto view        = bind_view(parameter, *storage);
        std::uint64_t elements = 0;
        for (const auto& part : view.parts) {
            elements += part.end - part.begin;
            require(part.parent->data != nullptr, "resolved view lost its parent");
        }
        require(elements == weight_element_count(view.shape),
                "resolved view lost logical coverage");
    }
    // Under --debug-poison the arena starts as poison: every byte outside the objects (the
    // alignment gaps) must still hold it, which shows the poison was there for the upload to
    // overwrite - so the byte-exact comparison above also proves no object byte was left unwritten
    // (no NaN in parameters after load, whatever the buffer held).
    if (const auto poison = debug_poison_byte(); poison && !expected.empty()) {
        const auto first             = expected.begin()->first;
        const std::byte* object_base = storage->device_parent({first}).data;
        std::uint64_t first_offset   = 0;
        std::uint64_t extent         = 0;
        for (std::size_t i = 0; i < placements.size(); ++i) {
            extent = std::max(extent, placements[i].first + placements[i].second);
        }
        std::size_t index = 0;
        for (const auto& [index_key, bytes] : expected) {
            if (index_key == first) { first_offset = placements[index].first; }
            ++index;
        }
        std::vector<std::byte> arena(extent);
        CUDA_CHECK(cudaMemcpy(arena.data(), object_base - first_offset, arena.size(),
                              cudaMemcpyDeviceToHost));
        std::vector<bool> covered(extent, false);
        for (const auto& [offset, bytes] : placements) {
            std::fill(covered.begin() + static_cast<std::ptrdiff_t>(offset),
                      covered.begin() + static_cast<std::ptrdiff_t>(offset + bytes), true);
        }
        std::uint64_t gaps = 0, unpoisoned = 0;
        for (std::uint64_t i = 0; i < extent; ++i) {
            if (covered[i]) { continue; }
            ++gaps;
            unpoisoned += arena[i] != std::byte(*poison) ? 1 : 0;
        }
        std::cout << path.filename().string() << ": poisoned arena, " << placements.size()
                  << " objects, " << gaps << " gap bytes, " << unpoisoned
                  << " of them not poison\n";
        require(gaps > 0 && unpoisoned == 0, "the weight arena was not poisoned before the upload");
    }
    std::cout << path.filename().string() << ": all bound parent bytes and logical views passed\n";
}

void staging_reuse(DeviceContext& device) {
    // More than one full staging ring, with distinct pages and a partial final block.
    constexpr std::size_t bytes = 5ULL * 64 * 1024 * 1024 + 1024;
    Fixture fixture;
    fixture.payload.clear();
    fixture.root    = {{"components", {{"text", {{"config", Json::object()}}}}},
                       {"objects", Json::array({{{"id", "large"},
                                                 {"kind", "tensor"},
                                                 {"shape", {bytes / 2}},
                                                 {"format", "bf16"},
                                                 {"layout", "contiguous_le_v1"},
                                                 {"offset", 0},
                                                 {"bytes", bytes}}})},
                       {"bindings", {{"large", {{"object", "large"}}}}},
                       {"uses", Json::array()},
                       {"files", Json::array({{{"path", nullptr}, {"payload_bytes", bytes}}})}};
    const auto text = fixture.root.dump();
    std::array<std::byte, 4096> header{};
    const std::array<unsigned char, 8> magic{'N', 'I', 'n', 'f', 'e', 'r', 'n', 'o'};
    for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
    put_word(header, 8, text.size(), 8);
    std::memcpy(header.data() + 32, text.data(), text.size());
    std::ofstream file(fixture.entry, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    std::array<std::byte, 4096> page{};
    for (std::size_t offset = 0; offset < bytes; offset += page.size()) {
        for (std::size_t i = 0; i < page.size(); ++i) {
            page[i] = std::byte(((offset + i) * 17 + (offset / 4096) * 13) % 251);
        }
        file.write(reinterpret_cast<const char*>(page.data()),
                   std::min(page.size(), bytes - offset));
    }
    file.close();
    Reader reader(fixture.entry);
    Binder binder(reader);
    (void)binder.parameter("large", {bytes / 2});
    auto backing     = materialize(reader, std::move(binder).finish(), device);
    const auto* base = backing.device_parent(reader.find("large")).data;
    std::vector<std::byte> chunk(1024 * 1024);
    for (std::size_t offset = 0; offset < bytes; offset += chunk.size()) {
        const auto count = std::min(chunk.size(), bytes - offset);
        CUDA_CHECK(cudaMemcpy(chunk.data(), base + offset, count, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < count; ++i) {
            const auto position = offset + i;
            require(chunk[i] == std::byte((position * 17 + (position / 4096) * 13) % 251),
                    "staging slot reuse overwrote an in-flight or later block");
        }
    }
}

// An act-order u4 plane (row_group_zp_ordered_v1) binds only when its column order is a
// permutation of K: a kernel gathers input column order[j] for every j.
void column_order(DeviceContext& device) {
    constexpr std::size_t bytes = 1024; // u4_g32 [1, 64]: codes, zeros @256, scales @512, order @768
    const auto artifact = [&](Fixture& fixture, bool valid) {
        fixture.payload.clear();
        fixture.root = {{"components", {{"text", {{"config", Json::object()}}}}},
                        {"objects", Json::array({{{"id", "ordered"},
                                                  {"kind", "tensor"},
                                                  {"shape", {1, 64}},
                                                  {"format", "u4_g32_zp_fp16"},
                                                  {"layout", "row_group_zp_ordered_v1"},
                                                  {"offset", 0},
                                                  {"bytes", bytes}}})},
                        {"bindings", {{"ordered", {{"object", "ordered"}}}}},
                        {"uses", Json::array()},
                        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", bytes}}})}};
        const auto text = fixture.root.dump();
        std::array<std::byte, 4096> header{};
        const std::array<unsigned char, 8> magic{'N', 'I', 'n', 'f', 'e', 'r', 'n', 'o'};
        for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
        put_word(header, 8, text.size(), 8);
        std::memcpy(header.data() + 32, text.data(), text.size());
        std::vector<std::byte> payload(bytes);
        for (std::size_t j = 0; j < 64; ++j) {
            put_word(payload, 768 + 4 * j, valid ? 63 - j : j / 2, 4); // reversed, or repeats
        }
        std::ofstream file(fixture.entry, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.write(reinterpret_cast<const char*>(header.data()), header.size());
        file.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    };
    {
        Fixture fixture;
        artifact(fixture, true);
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("ordered", {1, 64});
        auto backing        = materialize(reader, std::move(binder).finish(), device);
        const auto& parent  = backing.device_parent(reader.find("ordered"));
        std::array<std::uint32_t, 64> order{};
        CUDA_CHECK(cudaMemcpy(order.data(), parent.data + parent.geometry.order_offset,
                              sizeof(order), cudaMemcpyDeviceToHost));
        require(parent.geometry.order_offset == 768 && order[0] == 63 && order[63] == 0,
                "the column order plane did not reach the device");
    }
    Fixture fixture;
    artifact(fixture, false);
    rejects(
        [&] {
            Reader reader(fixture.entry);
            Binder binder(reader);
            (void)binder.parameter("ordered", {1, 64});
            (void)materialize(reader, std::move(binder).finish(), device);
        },
        "a column order that is not a permutation of K was bound");
}

} // namespace

int main(int argc, char** argv) {
    try {
        int count         = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && count == 0)) {
            return 77;
        }
        CUDA_CHECK(result);
        DeviceContext device;
        // --debug-poison BYTE before the file arguments: the weight arena starts as that poison.
        if (argc >= 3 && std::string_view(argv[1]) == "--debug-poison") {
            ninferno::set_debug_poison(ninferno::parse_debug_poison(argv[2]), {});
            argc -= 2;
            argv += 2;
        }
        if (argc == 3 && (std::string_view(argv[1]) == "--artifact" ||
                          std::string_view(argv[1]) == "--writer-fixture")) {
            file_roundtrip(device, argv[2], std::string_view(argv[1]) == "--writer-fixture");
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "expected [--debug-poison BYTE] [--artifact|--writer-fixture PATH]");
        }
        materialization(device);
        failure_and_host_only(device);
        mapped_residency(device);
        ninferno::test::materialization_cuda_errors(device);
        staging_reuse(device);
        column_order(device);
        std::cout << "artifact materialization checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
