#include "spec/corpus.h"

#include "core/file_io.h"
#include "spec/embedded_data.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <stdexcept>

namespace ninferno::spec {

const CorpusTokens* Corpus::tokens_for(const std::string& fingerprint) const {
    const auto it = tokens.find(fingerprint);
    return it == tokens.end() ? nullptr : &it->second;
}

Corpus parse_corpus(const std::string& json_text) {
    const auto doc = nlohmann::json::parse(json_text);
    if (doc.value("schema", 0) != 1) { throw std::runtime_error("spec corpus: unknown schema"); }
    Corpus out;
    for (const auto& item : doc.at("prompts")) {
        CorpusPrompt prompt;
        prompt.id         = item.at("id").get<std::string>();
        prompt.family     = item.at("family").get<std::string>();
        prompt.thinking   = item.value("thinking", false);
        prompt.max_tokens = item.value("max_tokens", 160);
        prompt.text       = item.at("text").get<std::string>();
        out.prompts.push_back(std::move(prompt));
    }
    if (const auto tokens = doc.find("tokens"); tokens != doc.end()) {
        for (const auto& [fingerprint, entry] : tokens->items()) {
            CorpusTokens set;
            set.label = entry.value("label", std::string());
            set.stops = entry.value("stops", std::vector<std::int32_t>{});
            for (const auto& [id, ids] : entry.at("prompts").items()) {
                set.prompts[id] = ids.get<std::vector<std::int32_t>>();
            }
            for (const auto& prompt : out.prompts) {
                if (!set.prompts.contains(prompt.id)) {
                    throw std::runtime_error("spec corpus: " + fingerprint + " lacks prompt " +
                                             prompt.id);
                }
            }
            out.tokens[fingerprint] = std::move(set);
        }
    }
    return out;
}

Corpus load_corpus(const std::filesystem::path& path) {
    return parse_corpus(read_file(path, "spec corpus"));
}

const Corpus& builtin_corpus() {
    static const Corpus corpus = parse_corpus(std::string(embedded::kSpecCorpusJson));
    return corpus;
}

} // namespace ninferno::spec
