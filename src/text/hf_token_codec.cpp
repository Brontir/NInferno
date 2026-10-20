#include "text/hf_token_codec.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninferno::text {
namespace {

std::vector<int> stop_ids(std::string_view generation_config_json, const Tokenizer& tokenizer) {
    std::vector<int> ids;
    if (!generation_config_json.empty()) {
        const auto config = nlohmann::json::parse(generation_config_json, nullptr, false);
        if (config.is_discarded() || !config.is_object()) {
            throw std::invalid_argument("generation_config.json must contain a JSON object");
        }
        if (const auto it = config.find("eos_token_id"); it != config.end()) {
            const auto add = [&](const nlohmann::json& value) {
                if (!value.is_number_integer()) {
                    throw std::invalid_argument(
                        "generation_config.json eos_token_id must be an id or a list of ids");
                }
                const int id = value.get<int>();
                if (std::find(ids.begin(), ids.end(), id) == ids.end()) { ids.push_back(id); }
            };
            if (it->is_array()) {
                for (const auto& value : *it) { add(value); }
            } else if (!it->is_null()) {
                add(*it);
            }
        }
    }
    if (ids.empty()) {
        if (const auto eos = tokenizer.eos_token_id()) { ids.push_back(*eos); }
    }
    for (const int id : ids) {
        if (!tokenizer.is_valid_token(id)) {
            throw std::invalid_argument("stop token " + std::to_string(id) +
                                        " is outside the tokenizer vocabulary");
        }
    }
    return ids;
}

} // namespace

HfTokenCodec::HfTokenCodec(std::string_view tokenizer_json, std::string_view tokenizer_config_json,
                           std::string_view generation_config_json)
    : tokenizer_(tokenizer_json, tokenizer_config_json),
      stops_(stop_ids(generation_config_json, tokenizer_)) {}

BoundaryEncodedText HfTokenCodec::encode_with_boundaries(std::string_view text,
                                                         std::span<const std::size_t> boundaries,
                                                         EncodeOptions options,
                                                         std::span<const ByteSpan> literal) const {
    return tokenizer_.encode_with_boundaries(text, boundaries, options, literal);
}

std::string HfTokenCodec::decode(std::span<const int> ids, DecodeOptions options) const {
    if (!ids.empty() && std::find(options.stop_token_ids.begin(), options.stop_token_ids.end(),
                                  ids.back()) != options.stop_token_ids.end()) {
        ids = ids.first(ids.size() - 1);
    }
    return tokenizer_.decode(ids, options.skip_special_tokens);
}

DecodedTokenView HfTokenCodec::decoded_token(int id) const {
    if (!tokenizer_.is_valid_token(id)) {
        throw std::out_of_range("token id " + std::to_string(id) + " is outside the vocabulary");
    }
    return DecodedTokenView{.bytes = tokenizer_.token_bytes(id),
                            .special = tokenizer_.is_special_token(id)};
}

bool HfTokenCodec::has_exact_token_domain(std::size_t size) const noexcept {
    if (size != tokenizer_.vocab_size()) { return false; }
    for (std::size_t id = 0; id < size; ++id) {
        if (!tokenizer_.is_valid_token(static_cast<int>(id))) { return false; }
    }
    return true;
}

} // namespace ninferno::text
