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

std::string structure_sha256(const Directory& directory) {
    const auto id_of = [&](ObjectHandle handle) {
        return handle.index < directory.objects.size() ? object_id(directory.objects[handle.index])
                                                       : std::string("?");
    };
    const auto binding_json = [&](const Binding& binding) {
        Json parts = Json::array();
        for (const auto& part : binding.parts) {
            parts.push_back(
                Json{{"object", id_of(part.object)}, {"begin", part.begin}, {"end", part.end}});
        }
        return Json{{"whole_object", binding.whole_object},
                    {"parts", std::move(parts)},
                    {"elements", binding.elements}};
    };
    Json components = Json::object();
    for (const auto& [name, component] : directory.components) {
        Json resources = Json::object();
        for (const auto& [role, handle] : component.resources) { resources[role] = id_of(handle); }
        Json item = {{"config", component.config}, {"resources", std::move(resources)}};
        if (component.target) { item["target"] = *component.target; }
        if (component.proposal) {
            item["proposal"] =
                Json{{"indexed", component.proposal->indexed}, {"rows", component.proposal->rows}};
        }
        components[name] = std::move(item);
    }
    Json bindings = Json::object();
    for (const auto& [name, binding] : directory.bindings) {
        bindings[name] = binding_json(binding);
    }
    Json uses = Json::array();
    for (const auto& [key, use] : directory.uses) {
        Json auxiliaries = Json::object();
        for (const auto& [name, binding] : use.auxiliaries) {
            auxiliaries[name] = binding_json(binding);
        }
        uses.push_back(Json{{"parameter", use.parameter},
                            {"input", use.input},
                            {"activation_policy",
                             use.activation_policy ? Json(static_cast<int>(*use.activation_policy))
                                                   : Json(nullptr)},
                            {"auxiliaries", std::move(auxiliaries)}});
    }
    // The public instance name (--name) is a label, not bytes: the same conversion under another
    // name keeps its identity.
    Json metadata = directory.metadata.is_object() ? directory.metadata : Json::object();
    metadata.erase("name");
    const Json all = {{"components", std::move(components)},
                      {"bindings", std::move(bindings)},
                      {"uses", std::move(uses)},
                      {"metadata", metadata}};
    return sha256_hex(all.dump());
}

std::optional<std::string> tensor_index_sha256(const Directory& directory) {
    const auto digests = directory.provenance.find("content_sha256");
    if (digests == directory.provenance.end() || !digests->is_object()) { return std::nullopt; }
    const auto digest_of = [&](const std::string& id) -> std::optional<std::string> {
        const auto it = digests->find(id);
        if (it == digests->end() || !it->is_string()) { return std::nullopt; }
        const auto text = it->get<std::string>();
        if (!hex_digits(text, 64) || text == std::string(64, '0')) { return std::nullopt; }
        return text;
    };
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
            const auto digest = digest_of(tensor->id);
            if (!digest) { return std::nullopt; }
            lines += "tensor\t" + tensor->id + "\t" + tensor->format + "\t" + tensor->layout +
                     "\t" + shape + "\t" + std::to_string(tensor->bytes) + "\t" +
                     std::to_string(tensor->divisors) + "\t" + *digest + "\n";
        } else {
            const auto& resource = std::get<ResourceObject>(*object);
            const auto digest    = digest_of(resource.id);
            if (!digest) { return std::nullopt; }
            lines += "resource\t" + resource.id + "\t" + resource.encoding + "\t" +
                     std::to_string(resource.bytes) + "\t" + *digest + "\n";
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
    const auto index = tensor_index_sha256(directory);
    if (!index) { return std::nullopt; }
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
    // By role: the order the converter opened them in does not enter.
    std::stable_sort(sorted.begin(), sorted.end(), [](const Json& a, const Json& b) {
        return text_of(a, "role") < text_of(b, "role");
    });
    Json choices = Json::object();
    // Never a path: a Python --override and a custom recipe enter by their file's content and the
    // function they name.
    for (const char* key :
         {"drafter_quantization", "vocabulary", "gguf_mapper", "ranking_sha256",
          "round_tiny_scales", "maintainer_quantize", "override_sha256", "override_function"}) {
        if (const auto it = provenance.find(key); it != provenance.end()) { choices[key] = *it; }
    }
    return Json{{"schema", std::string(kSourceIdentitySchema)},
                {"inputs", std::move(sorted)},
                {"quant", text_of(provenance, "quant")},
                {"converter", converter},
                {"recipe", provenance.contains("recipe_sha256")
                               ? Json{{"sha256", text_of(provenance, "recipe_sha256")},
                                      {"function", text_of(provenance, "recipe_function")}}
                               : Json(recipe)},
                {"choices", std::move(choices)},
                {"structure", structure_sha256(directory)},
                {"tensor_index", *index}};
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

} // namespace ninferno::artifact
