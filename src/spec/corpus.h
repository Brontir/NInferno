#pragma once

// The built-in speculation-tuning corpus (src/spec/data/spec_corpus.json, compiled in): a few
// prompts per task family, as text for engines that render prompts themselves, and as rendered
// ids per vocabulary fingerprint for engines that carry no tokenizer.

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::spec {

inline constexpr const char* kFamilies[] = {"math", "code", "prose", "thinking"};

struct CorpusPrompt {
    std::string id;
    std::string family;
    bool thinking          = false;
    std::int32_t max_tokens = 0;
    std::string text;
};

struct CorpusTokens {
    std::string label;
    std::vector<std::int32_t> stops;
    std::map<std::string, std::vector<std::int32_t>> prompts; // by prompt id
};

struct Corpus {
    std::vector<CorpusPrompt> prompts;
    std::map<std::string, CorpusTokens> tokens; // by vocabulary fingerprint

    // The rendered ids for a vocabulary, if the corpus carries them.
    [[nodiscard]] const CorpusTokens* tokens_for(const std::string& fingerprint) const;
};

// The compiled-in corpus, or the file at $NINFERNO_SPEC_CORPUS when that is set.
[[nodiscard]] const Corpus& builtin_corpus();
[[nodiscard]] Corpus load_corpus(const std::filesystem::path& path);
[[nodiscard]] Corpus parse_corpus(const std::string& json_text);

} // namespace ninferno::spec
