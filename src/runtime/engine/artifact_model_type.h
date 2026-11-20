#pragma once

#include <filesystem>
#include <string>

namespace ninferno::runtime {

// The text component's model_type of the artifact at `path`, read from its directory only. Its own
// translation unit, apart from the family table (family_registry.cpp), so that a host program that
// links no family (ninferno-ui's plan, prompt_tail.cpp) reads it without the loaders (#793).
[[nodiscard]] std::string artifact_model_type(const std::filesystem::path& path);

} // namespace ninferno::runtime
