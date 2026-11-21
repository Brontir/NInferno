#include "runtime/engine/artifact_model_type.h"

#include "artifact/reader.h"

#include <stdexcept>

namespace ninferno::runtime {

std::string artifact_model_type(const std::filesystem::path& path) {
    const artifact::Reader reader(path);
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) {
        throw std::invalid_argument("artifact " + path.string() + " has no text component");
    }
    const auto& config = text->second.config;
    if (!config.is_object() || !config.contains("model_type") ||
        !config.at("model_type").is_string()) {
        throw std::invalid_argument("artifact " + path.string() +
                                    " text component declares no model_type");
    }
    return config.at("model_type").get<std::string>();
}

} // namespace ninferno::runtime
