#include "spec/fingerprint.h"

#include "core/file_io.h"
#include "core/hex.h"

#include "artifact/reader.h"

#include <openssl/evp.h>

#include <map>
#include <memory>
#include <stdexcept>

namespace ninferno::spec {
std::string sha256_hex(std::string_view bytes) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned length = 0;
    EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx.get(), bytes.data(), bytes.size());
    EVP_DigestFinal_ex(ctx.get(), digest, &length);
    return hex_encode(digest, length);
}

VocabFingerprint vocab_fingerprint(const nlohmann::json& tokenizer) {
    const auto model = tokenizer.find("model");
    if (model == tokenizer.end() || !model->is_object()) {
        throw std::runtime_error("tokenizer.json has no model section");
    }
    const auto vocab = model->find("vocab");
    if (vocab == model->end()) { throw std::runtime_error("tokenizer.json has no model.vocab"); }
    std::map<std::uint64_t, std::string> text;
    if (vocab->is_object()) {
        for (const auto& [token, id] : vocab->items()) {
            if (id.is_number_integer() && id.get<std::int64_t>() >= 0) {
                text[id.get<std::uint64_t>()] = token;
            }
        }
    } else if (vocab->is_array()) { // Unigram: [[token, score], ...] indexed by id
        for (std::size_t i = 0; i < vocab->size(); ++i) {
            const auto& entry = (*vocab)[i];
            if (entry.is_array() && !entry.empty() && entry[0].is_string()) {
                text[i] = entry[0].get<std::string>();
            }
        }
    } else {
        throw std::runtime_error("tokenizer.json model.vocab is neither a map nor a list");
    }
    if (const auto added = tokenizer.find("added_tokens");
        added != tokenizer.end() && added->is_array()) {
        for (const auto& entry : *added) {
            if (entry.contains("id") && entry.contains("content") &&
                entry["id"].is_number_integer() && entry["content"].is_string()) {
                text[entry["id"].get<std::uint64_t>()] = entry["content"].get<std::string>();
            }
        }
    }
    std::string lines;
    for (const auto& [id, token] : text) {
        lines += std::to_string(id);
        lines += '\t';
        lines += token;
        lines += '\n';
    }
    VocabFingerprint out;
    out.value  = "v1:" + sha256_hex(lines).substr(0, 16);
    out.tokens = text.size();
    out.max_id = text.empty() ? 0 : text.rbegin()->first + 1;
    return out;
}

VocabFingerprint vocab_fingerprint_text(std::string_view tokenizer_json) {
    return vocab_fingerprint(nlohmann::json::parse(tokenizer_json));
}

VocabFingerprint vocab_fingerprint_file(const std::filesystem::path& tokenizer_json) {
    return vocab_fingerprint_text(read_file(tokenizer_json));
}

std::optional<nlohmann::json> artifact_tokenizer_json(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    const auto& components = reader.directory().components;
    std::vector<const artifact::Component*> order;
    if (const auto text = components.find("text"); text != components.end()) {
        order.push_back(&text->second);
    }
    for (const auto& [name, component] : components) {
        if (name != "text") { order.push_back(&component); }
    }
    for (const auto* component : order) {
        const auto it = component->resources.find("tokenizer.json");
        if (it == component->resources.end()) { continue; }
        const auto bytes = reader.read_object(it->second);
        return nlohmann::json::parse(
            std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }
    return std::nullopt;
}

std::optional<VocabFingerprint> artifact_vocab_fingerprint(const std::filesystem::path& artifact) {
    const auto tokenizer = artifact_tokenizer_json(artifact);
    if (!tokenizer) { return std::nullopt; }
    return vocab_fingerprint(*tokenizer);
}

std::string artifact_id_hex(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    const auto& id = reader.artifact_id();
    return hex_encode(id.data(), id.size());
}

} // namespace ninferno::spec
