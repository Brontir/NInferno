#include "models/dense/frontend.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "text/jinja.h"

#include <algorithm>
#include <stdexcept>

namespace ninferno::models::dense {
namespace {

// A special token of tokenizer_config.json: a string, or an AddedToken object with "content".
std::string token_text(const nlohmann::json& config, const char* key) {
    if (!config.contains(key) || config.at(key).is_null()) { return {}; }
    const auto& value = config.at(key);
    if (value.is_string()) { return value.get<std::string>(); }
    if (value.is_object() && value.contains("content")) {
        return value.at("content").get<std::string>();
    }
    return {};
}

} // namespace

Frontend::~Frontend()                              = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;

Frontend Frontend::load(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    artifact::Binder binder(reader);
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    const auto tokenizer_json  = resource("tokenizer.json");
    const auto tokenizer_cfg   = resource("tokenizer_config.json");
    const auto chat_template   = resource("chat_template.jinja");
    const auto generation_json = resource("generation_config.json");
    Frontend out;
    out.tokenizer_ = std::make_unique<text::Tokenizer>(tokenizer_json, tokenizer_cfg);
    out.template_  = std::make_unique<text::JinjaTemplate>(chat_template, "chat_template.jinja");
    const auto config = nlohmann::json::parse(tokenizer_cfg);
    out.bos_token_    = token_text(config, "bos_token");
    out.eos_token_    = token_text(config, "eos_token");
    out.stops_ = end_of_generation_ids(nlohmann::json::parse(generation_json),
                                       out.tokenizer_->eos_token_id());
    out.conventions_ = chat::resolve_conventions(chat_template);
    return out;
}

std::vector<std::int32_t> end_of_generation_ids(const nlohmann::json& generation,
                                                std::optional<std::int32_t> tokenizer_eos) {
    std::vector<std::int32_t> stops;
    const auto add = [&](std::int32_t id) {
        if (std::find(stops.begin(), stops.end(), id) == stops.end()) { stops.push_back(id); }
    };
    if (generation.contains("eos_token_id")) {
        const auto& eos = generation.at("eos_token_id");
        if (eos.is_array()) {
            for (const auto& id : eos) { add(id.get<std::int32_t>()); }
        } else if (eos.is_number_integer()) {
            add(eos.get<std::int32_t>());
        }
    }
    if (tokenizer_eos) { add(*tokenizer_eos); }
    if (stops.empty()) {
        throw std::invalid_argument("dense frontend: no end-of-generation token id");
    }
    return stops;
}

std::string Frontend::render_chat_text(const nlohmann::ordered_json& messages) const {
    // The variables transformers' apply_chat_template always defines: tools and documents are
    // None when not given (OLMo 3's template tests `tools is none`, which an undefined name fails).
    nlohmann::ordered_json context;
    context["messages"]              = messages;
    context["tools"]                 = nullptr;
    context["documents"]             = nullptr;
    context["add_generation_prompt"] = true;
    context["bos_token"]             = bos_token_;
    context["eos_token"]             = eos_token_;
    return template_->render(context).text;
}

std::vector<std::int32_t> Frontend::render_chat(const nlohmann::ordered_json& messages) const {
    return encode(render_chat_text(messages), false);
}

std::vector<std::int32_t> Frontend::encode(std::string_view text, bool add_special_tokens) const {
    const auto ids = tokenizer_->encode(text, add_special_tokens);
    return {ids.begin(), ids.end()};
}

std::string Frontend::decode(std::span<const std::int32_t> ids, bool skip_special_tokens) const {
    const std::vector<int> words(ids.begin(), ids.end());
    return tokenizer_->decode(words, skip_special_tokens);
}

} // namespace ninferno::models::dense
