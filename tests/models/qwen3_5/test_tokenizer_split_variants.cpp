// The Qwen tokenizer under both Split regexes a Qwen3.5+ tokenizer.json declares: Qwen3.5's
// (combining marks in letter runs, Qwen/Qwen3.8-27B) and Qwen2's (marks outside them, the
// transformers 5 re-saves the community NVFP4A16 builds ship), against the ids the `tokenizers`
// library gives for each checkpoint's own file (tokenizer_split_variants_fixture.h, written by
// hf_tokenizer_split_variants.py from the real files and embedded, so no Python at test time).

#include "models/qwen3_5/frontend/tokenizer.h"
#include "tokenizer_split_variants_fixture.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fi  = ninferno::models::qwen3_5::frontend;
namespace fix = split_variants_fixture;

std::string tokenizer_json(std::string_view regex) {
    nlohmann::json vocab = nlohmann::json::object();
    for (const auto& symbol : fix::kVocab) { vocab[std::string(symbol.text)] = symbol.id; }
    nlohmann::json merges = nlohmann::json::array();
    for (const auto& [left, right] : fix::kMerges) {
        merges.push_back({std::string(left), std::string(right)});
    }
    nlohmann::json added = nlohmann::json::array();
    for (const auto& token : fix::kAdded) {
        added.push_back({{"id", token.id},
                         {"content", std::string(token.text)},
                         {"single_word", false},
                         {"lstrip", false},
                         {"rstrip", false},
                         {"normalized", false},
                         {"special", true}});
    }
    const nlohmann::json byte_level = {{"type", "ByteLevel"},
                                       {"add_prefix_space", false},
                                       {"trim_offsets", false},
                                       {"use_regex", false}};
    return nlohmann::json{{"version", "1.0"},
                          {"added_tokens", std::move(added)},
                          {"normalizer", {{"type", "NFC"}}},
                          {"pre_tokenizer",
                           {{"type", "Sequence"},
                            {"pretokenizers",
                             {{{"type", "Split"},
                               {"pattern", {{"Regex", std::string(regex)}}},
                               {"behavior", "Isolated"},
                               {"invert", false}},
                              byte_level}}}},
                          {"post_processor", byte_level},
                          {"decoder", byte_level},
                          {"model",
                           {{"type", "BPE"},
                            {"dropout", nullptr},
                            {"continuing_subword_prefix", ""},
                            {"end_of_word_suffix", ""},
                            {"ignore_merges", false},
                            {"vocab", std::move(vocab)},
                            {"merges", std::move(merges)}}}}
        .dump();
}

std::string ids_text(const std::vector<int>& ids) {
    std::string text;
    for (const int id : ids) { text += (text.empty() ? "" : " ") + std::to_string(id); }
    return text;
}

int check_variant(const char* name, std::string_view regex,
                  std::vector<int> fix::Case::* expected) {
    const std::string json = tokenizer_json(regex);
    const fi::Tokenizer tokenizer({.tokenizer_json         = json,
                                   .tokenizer_config_json  = "{}",
                                   .generation_config_json = R"({"eos_token_id":[248046]})"});
    int failures = 0;
    for (std::size_t index = 0; index < fix::kCases.size(); ++index) {
        const fix::Case& item      = fix::kCases[index];
        const std::vector<int> got = tokenizer.encode(item.text);
        if (got != item.*expected) {
            std::fprintf(stderr, "%s split, case %zu: ids [%s], tokenizer.json gives [%s]\n", name,
                         index, ids_text(got).c_str(), ids_text(item.*expected).c_str());
            ++failures;
        }
        const std::string decoded = tokenizer.decode(item.*expected);
        if (decoded != item.decoded) {
            std::fprintf(stderr, "%s split, case %zu: decode differs\n", name, index);
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        int failures = check_variant("Qwen3.5", fix::kQwen35Split, &fix::Case::qwen35);
        failures += check_variant("Qwen2", fix::kQwen2Split, &fix::Case::qwen2);
        // The fixture exercises the difference: some texts split apart under the two regexes.
        std::size_t differ = 0;
        for (const auto& item : fix::kCases) { differ += item.qwen35 != item.qwen2 ? 1 : 0; }
        if (differ == 0) {
            std::fprintf(stderr, "no fixture text tells the two Split regexes apart\n");
            ++failures;
        }
        // Any other Split regex is still refused, not tokenized as either variant.
        std::string other(fix::kQwen2Split);
        other.replace(other.find("\\p{L}+"), 6, "\\p{L}*");
        try {
            const std::string json = tokenizer_json(other);
            (void)fi::Tokenizer({json, "{}", R"({"eos_token_id":[248046]})"});
            std::fprintf(stderr, "an unknown Split regex was accepted\n");
            ++failures;
        } catch (const std::invalid_argument&) {}
        std::printf(
            "tokenizer split variants: %zu texts, %zu tell the regexes apart, %d failures\n",
            fix::kCases.size(), differ, failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "tokenizer split variants: %s\n", ex.what());
        return 1;
    }
}
