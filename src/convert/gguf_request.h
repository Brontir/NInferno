#pragma once

#include <string>

namespace ninferno::convert {

// What a conversion asks of a GGUF checkpoint. Mtp: the MTP head alone (--components mtp); an
// architecture whose MTP head is a drafter model of its own (nemotron_h_moe) then reads the file's
// NextN block as that drafter. Safetensors checkpoints and the other GGUF architectures read the
// same either way.
enum class GgufRequest { Model, Mtp };

[[nodiscard]] inline GgufRequest gguf_request_for(const std::string& components) {
    return components == "mtp" ? GgufRequest::Mtp : GgufRequest::Model;
}

} // namespace ninferno::convert
