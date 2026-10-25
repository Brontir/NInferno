// Checks ninferno::text::Tokenizer against fixtures written by tools/text/tokenizer_fixtures.py.
//
//     test_tokenizer <model dir> [<model dir> ...]
//
// Each model directory holds tokenizer.json, optionally tokenizer_config.json, and fixture.json
// with reference ids and decoded strings from the Hugging Face `tokenizers` library. Every text
// must encode to exactly the reference ids (with and without special tokens) and every
// reference id sequence must decode to exactly the reference string. Exit status is non-zero
// on any mismatch. Without arguments only the built-in refusal and pattern checks run.

#include "text/pattern.h"
#include "text/tokenizer.h"
#include "support/files.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json  = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
using ninferno::text::Pattern;
using ninferno::text::EncodeOptions;
using ninferno::text::Tokenizer;

constexpr int kReportedMismatches = 4;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string escaped(const std::string& text, std::size_t limit = 60) {
    const std::string dumped = Json(text).dump(-1, ' ', false, Json::error_handler_t::replace);
    return dumped.size() > limit ? dumped.substr(0, limit) + "...\"" : dumped;
}

std::string describe(const Tokenizer& tokenizer, const std::vector<int>& ids, std::size_t at) {
    std::string out;
    const std::size_t begin = at >= 3 ? at - 3 : 0;
    const std::size_t end   = std::min(ids.size(), at + 4);
    for (std::size_t i = begin; i < end; ++i) {
        const auto token = tokenizer.id_to_token(ids[i]);
        out += (i == at ? " >" : " ") + std::to_string(ids[i]) + ":" +
               escaped(token.value_or("?"), 24);
    }
    return out.empty() ? " (end)" : out;
}

struct ModelResult {
    int texts          = 0;
    int checks         = 0;
    int mismatches     = 0;
    std::size_t bytes  = 0;
    double encode_secs = 0;
};

class ModelCheck {
public:
    ModelCheck(const Tokenizer& tokenizer, std::string model) : tokenizer_(tokenizer), model_(std::move(model)) {}

    void ids(const std::string& what, const std::vector<int>& got, const Json& expected) {
        ++result.checks;
        const auto want = expected.get<std::vector<int>>();
        if (got == want) { return; }
        std::size_t at = 0;
        while (at < got.size() && at < want.size() && got[at] == want[at]) { ++at; }
        report(what + ": ids differ at index " + std::to_string(at) + " (got " +
               std::to_string(got.size()) + ", want " + std::to_string(want.size()) + ")\n      want" +
               describe(tokenizer_, want, at) + "\n      got " + describe(tokenizer_, got, at));
    }

    void text(const std::string& what, const std::string& got, const Json& expected) {
        ++result.checks;
        const auto want = expected.get<std::string>();
        if (got == want) { return; }
        std::size_t at = 0;
        while (at < got.size() && at < want.size() && got[at] == want[at]) { ++at; }
        const std::size_t from = at >= 16 ? at - 16 : 0;
        report(what + ": decoded text differs at byte " + std::to_string(at) + "\n      want " +
               escaped(want.substr(from, 48)) + "\n      got  " + escaped(got.substr(from, 48)));
    }

    void failure(const std::string& what, const std::exception& ex) {
        ++result.checks;
        report(what + ": threw " + ex.what());
    }

    ModelResult result;

private:
    void report(const std::string& message) {
        if (result.mismatches++ < kReportedMismatches) {
            std::cout << "  MISMATCH " << model_ << " " << message << "\n";
        }
    }

    const Tokenizer& tokenizer_;
    std::string model_;
};

bool check_model(const fs::path& dir) {
    const std::string name = dir.filename().string();
    const std::string config =
        fs::exists(dir / "tokenizer_config.json") ? read_file(dir / "tokenizer_config.json") : "";
    const auto load_start = Clock::now();
    std::optional<Tokenizer> loaded;
    try {
        loaded.emplace(read_file(dir / "tokenizer.json"), config);
    } catch (const std::exception& ex) {
        std::cout << name << ": REFUSED " << ex.what() << "\n";
        return false;
    }
    const Tokenizer& tokenizer = *loaded;
    const double load_secs     = seconds_since(load_start);
    const Json fixture         = Json::parse(read_file(dir / "fixture.json"));

    ModelCheck check(tokenizer, name);
    for (const Json& item : fixture.at("cases")) {
        const std::string label = item.at("name").get<std::string>();
        const std::string text  = item.at("text").get<std::string>();
        ++check.result.texts;
        try {
            const auto start     = Clock::now();
            const auto ids       = tokenizer.encode(text, false);
            const double elapsed = seconds_since(start);
            if (label.starts_with("corpus:")) {
                // Best of three runs, so a scheduler hiccup does not decide the figure.
                double best = elapsed;
                for (int run = 0; run < 2; ++run) {
                    const auto again = Clock::now();
                    (void)tokenizer.encode(text, false);
                    best = std::min(best, seconds_since(again));
                }
                check.result.bytes += text.size();
                check.result.encode_secs += best;
            }
            check.ids(label + " encode", ids, item.at("ids"));
            check.ids(label + " encode+special", tokenizer.encode(text, true), item.at("ids_special"));
            const auto special = item.at("ids_special").get<std::vector<int>>();
            const auto plain   = item.at("ids").get<std::vector<int>>();
            check.text(label + " decode", tokenizer.decode(special, false), item.at("decoded"));
            check.text(label + " decode skip", tokenizer.decode(special, true), item.at("decoded_skip"));
            check.text(label + " decode plain", tokenizer.decode(plain, false), item.at("decoded_plain"));
        } catch (const std::exception& ex) {
            check.failure(label, ex);
        }
    }
    for (const Json& item : fixture.value("decode_cases", Json::array())) {
        const std::string label = item.at("name").get<std::string>();
        const auto ids          = item.at("ids").get<std::vector<int>>();
        try {
            check.text(label, tokenizer.decode(ids, false), item.at("decoded"));
            check.text(label + " skip", tokenizer.decode(ids, true), item.at("decoded_skip"));
        } catch (const std::exception& ex) {
            check.failure(label, ex);
        }
    }

    const ModelResult& r = check.result;
    const double mbps    = r.encode_secs > 0 ? static_cast<double>(r.bytes) / 1e6 / r.encode_secs : 0;
    std::printf("%-16s %-44s texts %3d  checks %4d  mismatches %d  encode %.1f MB/s (%.2f MB)  load %.2f s\n",
                name.c_str(), fixture.value("repo", std::string{}).c_str(), r.texts, r.checks,
                r.mismatches, mbps, static_cast<double>(r.bytes) / 1e6, load_secs);
    return r.mismatches == 0;
}

// An overflow check (EncodeOptions::overflow_check_only) stops once the pieces' bytes prove the
// text needs more ids than max_tokens; the count it answers is the bounded encoding's, a text
// within the limit encodes exactly as without it, and a vocabulary that can drop a character
// gets no bound.
bool check_overflow_check() {
    // A byte fallback over every byte: each byte of a piece has a token, so the bound holds.
    Json vocab = {{"a", 0}, {"b", 1}, {"ab", 2}};
    for (int byte = 0; byte < 256; ++byte) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", byte);
        vocab[name] = 3 + byte;
    }
    const Tokenizer tokenizer(
        Json{{"model",
              {{"type", "BPE"}, {"byte_fallback", true}, {"vocab", vocab}, {"merges", {"a b"}}}}}
            .dump());
    const auto encode = [&](const Tokenizer& t, const std::string& text, std::size_t limit,
                            bool check) {
        return t
            .encode_with_boundaries(
                text, {}, EncodeOptions{.max_tokens = limit, .overflow_check_only = check})
            .input_ids;
    };
    const std::size_t over = encode(tokenizer, std::string(1U << 20, 'a'), 1001, true).size();
    const std::string text = "abababa";
    const bool exact =
        encode(tokenizer, text, 100, true) == tokenizer.encode_with_boundaries(text, {}).input_ids;
    const std::size_t bounded = encode(tokenizer, std::string(500, 'b'), 100, false).size();
    const std::size_t early   = encode(tokenizer, std::string(500, 'b'), 100, true).size();
    // Without a token for every byte a character can vanish (no unk token): no bound, no early
    // stop.
    const Tokenizer dropping(
        R"({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}})");
    const std::size_t dropped = encode(dropping, std::string(5000, 'c'), 10, true).size();
    const bool ok = over == 1001 && exact && bounded == 100 && early == 100 && dropped == 0;
    if (!ok) {
        std::cout << "  OVERFLOW CHECK over=" << over << " bounded=" << bounded
                  << " early=" << early << " exact=" << exact << " dropped=" << dropped << "\n";
    }
    return ok;
}

// Constructs that must be refused rather than tokenized wrongly, and settings read past.
bool check_refusals() {
    const std::string model =
        R"("model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]})";
    const std::vector<std::pair<std::string, std::string>> cases = {
        {R"({"model": {"type": "WordPiece", "vocab": {}}})", "WordPiece"},
        {R"({"normalizer": {"type": "BertNormalizer"}, )" + model + "}", "BertNormalizer"},
        {R"({"pre_tokenizer": {"type": "Whitespace"}, )" + model + "}", "Whitespace"},
        {R"({"decoder": {"type": "CTC"}, )" + model + "}", "CTC"},
        {R"({"post_processor": {"type": "RobertaProcessing"}, )" + model + "}", "RobertaProcessing"},
        {R"({"pre_tokenizer": {"type": "Split", "pattern": {"Regex": "(a)\\1"}, "behavior": "Isolated"}, )" + model + "}", "escape"},
        {R"({"pre_tokenizer": {"type": "Split", "pattern": {"Regex": "(?<=a)b"}, "behavior": "Isolated"}, )" + model + "}", "lookbehind"},
        {R"({"pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\p{Han}+"}, "behavior": "Isolated"}, )" + model + "}", "Han"},
        {R"({"pre_tokenizer": {"type": "Split", "pattern": {"Regex": "a*"}, "behavior": "Isolated"}, )" + model + "}", "empty string"},
        {R"({"model": {"type": "BPE", "dropout": 0.1, "vocab": {"a": 0}, "merges": []}})", "dropout"},
        {R"({"added_tokens": [{"id": 7, "content": "<x>", "special": true}], )" + model + "}", "assigns 3"},
        {R"({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 15999999}, "merges": ["a b"]}})",
         "far beyond"},
    };
    bool ok = true;
    for (const auto& [json, needle] : cases) {
        try {
            Tokenizer tokenizer(json);
            std::cout << "  REFUSAL MISSING for " << needle << "\n";
            ok = false;
        } catch (const std::invalid_argument& ex) {
            if (std::string(ex.what()).find(needle) == std::string::npos) {
                std::cout << "  REFUSAL message lacks \"" << needle << "\": " << ex.what() << "\n";
                ok = false;
            }
        }
    }
    // truncation and padding are read past, as transformers does by default: the encoding is
    // neither cut to max_length nor padded to a fixed length.
    try {
        const Tokenizer tokenizer(
            R"({"truncation": {"direction": "Right", "max_length": 1, "strategy": "LongestFirst", "stride": 0}, )"
            R"("padding": {"strategy": {"Fixed": 8}, "direction": "Right", "pad_to_multiple_of": null, "pad_id": 0, "pad_type_id": 0, "pad_token": "a"}, )" +
            model + "}");
        const std::vector<int> ids = tokenizer.encode("ababab", true);
        if (ids != std::vector<int>{2, 2, 2}) {
            std::cout << "  truncation/padding changed the encoding:";
            for (int id : ids) { std::cout << " " << id; }
            std::cout << "\n";
            ok = false;
        }
    } catch (const std::exception& ex) {
        std::cout << "  truncation/padding refused: " << ex.what() << "\n";
        ok = false;
    }
    try {
        Tokenizer tokenizer("{" + model + "}");
        (void)tokenizer.encode("a\xFF", false);
        std::cout << "  REFUSAL MISSING for invalid UTF-8 input\n";
        ok = false;
    } catch (const std::invalid_argument&) {
    }
    return ok;
}

bool check_patterns() {
    struct Case {
        const char* regex;
        const char* text;
        std::vector<std::string> matches;
    };
    const std::vector<Case> cases = {
        {R"(\s+(?!\S)|\s+)", "a   b", {"  ", " "}},
        {R"((?i:'s|'t)|\p{L}+)", "IT'S it'ſ", {"IT", "'S", "it", "'ſ"}},
        {R"(\p{N}{1,3})", "1234567", {"123", "456", "7"}},
        {R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+)", "HelloWORLDx", {"Hello", "WORLDx"}},
        {R"(a+?b|a++)", "aaab aaa", {"aaab", "aaa"}},
        {R"(\x{1F600}|[A-C]{2,})", "ABCD😀", {"ABC", "😀"}},
    };
    bool ok = true;
    for (const Case& c : cases) {
        const Pattern pattern(c.regex);
        std::vector<std::string> got;
        const std::string text = c.text;
        for (const auto& span : pattern.find_all(text)) { got.push_back(text.substr(span.begin, span.end - span.begin)); }
        if (got != c.matches) {
            std::cout << "  PATTERN " << c.regex << " on " << escaped(text) << " gave";
            for (const auto& m : got) { std::cout << " " << escaped(m); }
            std::cout << "\n";
            ok = false;
        }
    }
    return ok;
}

} // namespace

// The special ids a TemplateProcessing puts before a document (what a causal score starts every
// window from): <s> for "<s> $A", nothing for "$A </s>" or without a template.
bool check_template_prefix() {
    const std::string model =
        R"("added_tokens": [{"id": 3, "content": "<s>", "special": true}, {"id": 4, "content": "</s>", "special": true}], )"
        R"("model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]})";
    const auto with = [&](const std::string& single) {
        return Tokenizer(R"({"post_processor": {"type": "TemplateProcessing", "single": )" + single +
                         R"(, "special_tokens": {"<s>": {"id": "<s>", "ids": [3], "tokens": ["<s>"]}, )"
                         R"("</s>": {"id": "</s>", "ids": [4], "tokens": ["</s>"]}}}, )" + model + "}")
            .template_prefix();
    };
    const auto front = with(R"([{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}])");
    const auto back  = with(R"([{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}])");
    const auto none  = Tokenizer("{" + model + "}").template_prefix();
    const bool ok    = front == std::vector<int>{3} && back.empty() && none.empty();
    if (!ok) { std::cout << "  TEMPLATE PREFIX wrong\n"; }
    return ok;
}

int main(int argc, char** argv) {
    bool ok = true;
    ok &= check_patterns();
    ok &= check_template_prefix();
    ok &= check_refusals();
    ok &= check_overflow_check();
    std::cout << "pattern and refusal checks: " << (ok ? "ok" : "FAILED") << "\n";
    for (int i = 1; i < argc; ++i) {
        try {
            ok &= check_model(argv[i]);
        } catch (const std::exception& ex) {
            std::cout << argv[i] << ": ERROR " << ex.what() << "\n";
            ok = false;
        }
    }
    std::cout << (ok ? "ALL PASSED" : "FAILURES") << "\n";
    return ok ? 0 : 1;
}
