#include "artifact/source_identity.h"

#include "artifact/reader.h"
#include "core/sha256.h"

#include <algorithm>
#include <vector>

namespace ninferno::artifact {
namespace {

bool hex_digits(std::string_view text, std::size_t count) noexcept {
    return text.size() == count && std::all_of(text.begin(), text.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

std::string text_of(const Json& object, const char* key) {
    const auto it = object.find(key);
    return it != object.end() && it->is_string() ? it->get<std::string>() : std::string();
}

} // namespace

std::string tensor_index_sha256(const Directory& directory) {
    std::vector<const Object*> objects;
    objects.reserve(directory.objects.size());
    for (const auto& object : directory.objects) { objects.push_back(&object); }
    std::sort(objects.begin(), objects.end(),
              [](const Object* a, const Object* b) { return object_id(*a) < object_id(*b); });
    std::string lines;
    for (const auto* object : objects) {
        if (const auto* tensor = std::get_if<TensorObject>(object)) {
            std::string shape;
            for (const auto dimension : tensor->shape) {
                shape += (shape.empty() ? "" : ",") + std::to_string(dimension);
            }
            lines += "tensor\t" + tensor->id + "\t" + tensor->format + "\t" + tensor->layout +
                     "\t" + shape + "\t" + std::to_string(tensor->bytes) + "\t" +
                     std::to_string(tensor->divisors) + "\n";
        } else {
            const auto& resource = std::get<ResourceObject>(*object);
            lines += "resource\t" + resource.id + "\t" + resource.encoding + "\t" +
                     std::to_string(resource.bytes) + "\n";
        }
    }
    return sha256_hex(lines);
}

std::optional<Json> source_record(const Directory& directory) {
    const Json& provenance = directory.provenance;
    const auto inputs      = provenance.find("inputs");
    if (inputs == provenance.end() || !inputs->is_array() || inputs->empty()) {
        return std::nullopt;
    }
    const auto converter = text_of(provenance, "converter");
    const auto recipe    = text_of(provenance, "recipe");
    if (converter.empty() || recipe.empty()) { return std::nullopt; }
    Json sorted = Json::array();
    for (auto input : *inputs) {
        if (!input.is_object() || !input.contains("files") || !input["files"].is_array()) {
            return std::nullopt;
        }
        auto& files = input["files"];
        std::vector<Json> list(files.begin(), files.end());
        std::sort(list.begin(), list.end(),
                  [](const Json& a, const Json& b) { return a.dump() < b.dump(); });
        files = Json(list);
        sorted.push_back(std::move(input));
    }
    Json choices = Json::object();
    for (const char* key : {"drafter_quantization", "vocabulary", "gguf_mapper", "ranking"}) {
        if (const auto it = provenance.find(key); it != provenance.end()) { choices[key] = *it; }
    }
    return Json{{"schema", std::string(kSourceIdentitySchema)},
                {"inputs", std::move(sorted)},
                {"quant", text_of(provenance, "quant")},
                {"converter", converter},
                {"recipe", recipe},
                {"choices", std::move(choices)},
                {"tensor_index", tensor_index_sha256(directory)}};
}

std::optional<std::string> source_identity(const Directory& directory) {
    const auto record = source_record(directory);
    if (!record) { return std::nullopt; }
    return sha256_hex(record->dump());
}

std::optional<std::string> source_identity(const std::filesystem::path& artifact) {
    const Reader reader(artifact);
    return source_identity(reader.directory());
}

bool is_source_identity(std::string_view text) noexcept { return hex_digits(text, 64); }

std::optional<HubFile> hub_file(const std::filesystem::path& file) {
    std::vector<std::string> parts;
    for (const auto& part : file.lexically_normal()) { parts.push_back(utf8_file_name(part)); }
    for (std::size_t i = 0; i + 3 < parts.size(); ++i) {
        const auto& repo = parts[i];
        if (!repo.starts_with("models--") || parts[i + 1] != "snapshots" ||
            !hex_digits(parts[i + 2], 40)) {
            continue;
        }
        const auto name = repo.substr(8);
        const auto cut  = name.find("--");
        if (cut == std::string::npos || cut == 0 || cut + 2 >= name.size()) { return std::nullopt; }
        HubFile out;
        out.repo   = name.substr(0, cut) + "/" + name.substr(cut + 2);
        out.commit = parts[i + 2];
        for (std::size_t j = i + 3; j < parts.size(); ++j) {
            out.name += (out.name.empty() ? "" : "/") + parts[j];
        }
        return out;
    }
    return std::nullopt;
}

Json input_file(const std::filesystem::path& file, const std::string& sha256) {
    if (const auto hub = hub_file(file)) {
        return Json{
            {"repo", hub->repo}, {"commit", hub->commit}, {"name", hub->name}, {"sha256", sha256}};
    }
    return Json{{"sha256", sha256}};
}

} // namespace ninferno::artifact
