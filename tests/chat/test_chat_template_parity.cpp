// Chat-template parity of ninferno-serve with transformers, for every template NInferno serves.
//
//   python tests/chat/hf_chat_template_parity.py <root>
//   NINFERNO_CHAT_TEMPLATE_PARITY_ROOT=<root> ninferno_chat_template_parity_test
//
// The script writes one directory per model (the frontend files ninferno-serve reads from an
// artifact, frontend.json naming the serving family, and cases.jsonl: OpenAI Chat Completions
// requests with the ids transformers' apply_chat_template(..., tokenize=True) gives for their
// messages) and required.json (every convention of src/chat/data/conventions.json). The test runs
// each request the way ninferno-serve does -- the dialect's request parser, prompt semantics,
// to_prompt_input, and the frontend built as the model's family builds it (runtime/engine: the
// Qwen 3.5 tokenizer, or attach_hf_text_tokenizer / apply_hf_text_options for the dense decoders
// and Gemma 4) -- and requires the same prompt ids, token for token. A second render must give
// the same ids (prefix-cache stability). The templates must resolve, between them, to every turn
// markup, reasoning markup and tool-call format of the conventions. Without the variable the test
// is skipped (77): the files are the checkpoints', not the repository's.

#include "chat/conventions.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "serve/anthropic_messages.h"
#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/translate.h"
#include "support/files.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using ninferno::TokenId;
namespace qwen = ninferno::models::qwen3_5;

// The item is parsed order-preserving, as ninferno-serve parses a request body.
ninferno::PromptInput product_prompt(const nlohmann::ordered_json& item) {
    using namespace ninferno::serve;
    const RequestJson& body = item.at("request");
    RequestLimits limits;
    limits.default_max_tokens = 64;
    const bool anthropic      = item.value("dialect", "openai") == "anthropic";
    const GenerationRequest request =
        anthropic ? parse_anthropic_messages_request(body, limits).generation
                  : parse_chat_completion_request(body, limits).generation;
    const ServeOptions server;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, server);
    return to_prompt_input(request, semantics, [](const ContentPart&) -> ninferno::OwnedMedia {
        throw std::invalid_argument("the parity cases carry no media");
    });
}

std::vector<TokenId> prompt_ids(const qwen::Frontend& frontend,
                                const nlohmann::ordered_json& item) {
    const qwen::PreparedPrompt prepared = frontend.prepare(product_prompt(item));
    return qwen::FrontendTestAccess::inspect(prepared).token_ids;
}

struct Totals {
    std::size_t cases = 0, compared_tokens = 0, mismatches = 0, unstable = 0, refused = 0,
                refused_by_both = 0, policy_refusals = 0, tokenizers_differ = 0,
                shipped_tokenizer = 0;
};

// One model's directory; its counts are added to `all`. Returns the conventions it resolved to.
ninferno::chat::ChatConventions run_model(const std::filesystem::path& dir, Totals& all) {
    const auto meta           = nlohmann::json::parse(read_file(dir / "frontend.json"));
    const std::string family  = meta.at("family").get<std::string>();
    const std::string tokenizer_json    = read_file(dir / "tokenizer.json");
    const std::string tokenizer_config  = read_file(dir / "tokenizer_config.json");
    const std::string chat_template     = read_file(dir / "chat_template.jinja");
    const std::string generation_config = read_file(dir / "generation_config.json");
    qwen::FrontendResources resources{.tokenizer_json         = tokenizer_json,
                                      .tokenizer_config_json  = tokenizer_config,
                                      .chat_template_jinja    = chat_template,
                                      .generation_config_json = generation_config};
    qwen::FrontendOptions options;
    options.vision_enabled = false;
    if (family == "qwen3_5") {
        qwen::attach_qwen3_5_tokenizer(resources);
    } else if (family == "hf_text") {
        qwen::attach_hf_text_tokenizer(resources);
        qwen::apply_hf_text_options(options);
    } else {
        throw std::runtime_error(dir.string() + ": unknown family " + family);
    }
    options.max_context = std::numeric_limits<std::uint32_t>::max();
    const qwen::Frontend frontend = qwen::make_frontend(resources, options);
    const auto conventions        = ninferno::chat::resolve_conventions(chat_template);

    Totals model;
    std::ifstream cases(dir / "cases.jsonl");
    if (!cases) { throw std::runtime_error("cannot read cases.jsonl in " + dir.string()); }
    const std::string name = dir.filename().string();
    std::string line;
    std::size_t reported = 0;
    while (std::getline(cases, line)) {
        if (line.empty()) { continue; }
        const auto item      = nlohmann::ordered_json::parse(line);
        const std::string id = name + " | " + item.at("id").get<std::string>();
        ++model.cases;
        if (item.contains("tokenizers_ids")) { ++model.tokenizers_differ; }
        if (item.contains("refused")) {
            // The template itself refuses (raise_exception): the product must refuse too.
            try {
                (void)prompt_ids(frontend, item);
                ++model.mismatches;
                if (++reported <= 8) { std::cerr << "[" << id << "] rendered what the template refuses\n"; }
            } catch (const std::exception&) { ++model.refused_by_both; }
            continue;
        }
        const auto want = item.at("ids").get<std::vector<TokenId>>();
        std::vector<TokenId> got, again;
        try {
            got   = prompt_ids(frontend, item);
            again = prompt_ids(frontend, item);
        } catch (const std::exception& error) {
            const std::string expected = item.value("expect_ninferno_refusal", "");
            if (!expected.empty() && std::string(error.what()).find(expected) != std::string::npos) {
                ++model.policy_refusals;
                continue;
            }
            ++model.refused;
            if (++reported <= 8) { std::cerr << "[" << id << "] refused: " << error.what() << "\n"; }
            continue;
        }
        if (item.contains("expect_ninferno_refusal")) {
            ++model.refused;
            if (++reported <= 8) { std::cerr << "[" << id << "] rendered a request the policy refuses\n"; }
        }
        model.compared_tokens += want.size();
        if (got != again) {
            ++model.unstable;
            if (++reported <= 8) { std::cerr << "[" << id << "] a second render gave different ids\n"; }
        }
        if (got == want) { continue; }
        if (item.contains("tokenizers_ids") &&
            got == item.at("tokenizers_ids").get<std::vector<TokenId>>()) {
            // transformers deviates from the checkpoint's tokenizer.json on this text; NInferno
            // runs tokenizer.json as shipped.
            ++model.shipped_tokenizer;
            continue;
        }
        ++model.mismatches;
        if (++reported > 8) { continue; }
        std::size_t at = 0;
        while (at < got.size() && at < want.size() && got[at] == want[at]) { ++at; }
        const auto window = [&](const std::vector<TokenId>& ids) {
            const std::size_t begin = at > 8 ? at - 8 : 0;
            const std::size_t end   = std::min(ids.size(), at + 16);
            std::string out;
            for (std::size_t i = begin; i < end; ++i) {
                out += (i == at ? " >" : " ") + std::to_string(ids[i]) + "=" +
                       nlohmann::json(resources.tokenizer->decode(std::vector<TokenId>{ids[i]}))
                           .dump();
            }
            return out;
        };
        std::cerr << "[" << id << "] ids differ at " << at << " of " << want.size() << " (ninferno "
                  << got.size() << ")\n  transformers:" << window(want)
                  << "\n  ninferno:      " << window(got) << "\n";
    }
    std::cout << name << " (" << family << ", " << meta.value("repo", "") << "): turns "
              << (conventions.turns ? conventions.turns->name : "none") << ", reasoning "
              << (conventions.reasoning ? conventions.reasoning->name : "none") << ", tools "
              << ninferno::chat::tool_call_format_name(conventions.tools.format) << "; "
              << model.cases << " cases, " << model.compared_tokens << " tokens, mismatches "
              << model.mismatches << ", unstable " << model.unstable << ", refused "
              << model.refused << ", refused by both " << model.refused_by_both
              << ", policy refusals " << model.policy_refusals
              << ", tokenizers-library ids differ from transformers' " << model.tokenizers_differ
              << " (NInferno equal to tokenizer.json there: " << model.shipped_tokenizer << ")\n";
    all.cases += model.cases;
    all.compared_tokens += model.compared_tokens;
    all.mismatches += model.mismatches;
    all.unstable += model.unstable;
    all.refused += model.refused;
    all.refused_by_both += model.refused_by_both;
    all.policy_refusals += model.policy_refusals;
    all.tokenizers_differ += model.tokenizers_differ;
    all.shipped_tokenizer += model.shipped_tokenizer;
    return conventions;
}

} // namespace

int main() {
    const char* root_text = std::getenv("NINFERNO_CHAT_TEMPLATE_PARITY_ROOT");
    if (root_text == nullptr || *root_text == '\0') {
        std::cout << "chat template parity: NINFERNO_CHAT_TEMPLATE_PARITY_ROOT not set, skipped\n";
        return 77;
    }
    try {
        const std::filesystem::path root(root_text);
        std::vector<std::filesystem::path> dirs;
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_directory() && std::filesystem::exists(entry.path() / "frontend.json")) {
                dirs.push_back(entry.path());
            }
        }
        std::sort(dirs.begin(), dirs.end());
        if (dirs.empty()) { throw std::runtime_error("no model directories in " + root.string()); }
        Totals all;
        std::map<std::string, std::set<std::string>> covered;
        std::size_t failed_models = 0;
        for (const auto& dir : dirs) {
            try {
                const auto conventions = run_model(dir, all);
                if (conventions.turns) { covered["turns"].insert(conventions.turns->name); }
                if (conventions.reasoning) { covered["reasoning"].insert(conventions.reasoning->name); }
                covered["tools"].insert(
                    std::string(ninferno::chat::tool_call_format_name(conventions.tools.format)));
            } catch (const std::exception& error) {
                ++failed_models;
                std::cerr << dir.filename().string() << ": " << error.what() << "\n";
            }
        }
        std::size_t uncovered = 0;
        const auto required = nlohmann::json::parse(read_file(root / "required.json"));
        for (const auto& [kind, names] : required.items()) {
            for (const auto& name : names) {
                if (!covered[kind].contains(name.get<std::string>())) {
                    ++uncovered;
                    std::cerr << "no template resolves to the " << kind << " convention "
                              << name.get<std::string>() << "\n";
                }
            }
        }
        std::cout << "chat template parity: " << dirs.size() << " models, " << all.cases
                  << " requests, " << all.compared_tokens << " reference tokens, mismatches="
                  << all.mismatches << " unstable=" << all.unstable << " refused=" << all.refused
                  << " refused_by_both=" << all.refused_by_both
                  << " policy_refusals=" << all.policy_refusals << " models_failed=" << failed_models
                  << " uncovered_conventions=" << uncovered
                  << " tokenizers_library_differs=" << all.tokenizers_differ
                  << " equal_to_shipped_tokenizer_not_transformers=" << all.shipped_tokenizer
                  << "\n";
        return all.mismatches == 0 && all.unstable == 0 && all.refused == 0 &&
                       failed_models == 0 && uncovered == 0
                   ? 0
                   : 1;
    } catch (const std::exception& error) {
        std::cerr << "chat template parity error: " << error.what() << "\n";
        return 1;
    }
}
