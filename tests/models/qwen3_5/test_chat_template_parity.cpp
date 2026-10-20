// Chat-template parity with Hugging Face transformers through the product path.
//
//   python tests/models/qwen3_5/hf_chat_template_parity.py <tokenizer dir> <out dir>
//
// writes the checkpoint's frontend files and cases.jsonl: OpenAI Chat Completions requests (and
// Anthropic Messages requests for assistant prefill) with the prompt transformers'
// apply_chat_template renders for each and that prompt's ids. Then
//
//   NINFERNO_CHAT_TEMPLATE_PARITY_DIR=<out dir> ninferno_qwen3_5_chat_template_parity_test
//
// runs every request the way ninferno-serve does -- the dialect's request parser, prompt semantics,
// to_prompt_input, the Qwen frontend -- and compares the prompt ids with the reference (an
// expanded image-pad run counts as the one pad the template writes). Each prompt is also
// prepared twice: a second render must give the same ids (prefix-cache stability). Without the
// variable the test is skipped (77): the files are the checkpoint's, not the repository's.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "product/media_acquire/acquire.h"
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
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fi = ninferno::models::qwen3_5::frontend;
using ninferno::TokenId;

constexpr TokenId kImagePad = 248056; // <|image_pad|>, checked against the tokenizer below

std::vector<TokenId> collapse_image_pads(const std::vector<TokenId>& ids) {
    std::vector<TokenId> out;
    for (const TokenId id : ids) {
        if (id == kImagePad && !out.empty() && out.back() == kImagePad) { continue; }
        out.push_back(id);
    }
    return out;
}

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
    return to_prompt_input(request, semantics, [](const ContentPart& part) {
        ninferno::OwnedMedia media;
        media.kind  = part.kind == ContentKind::Image ? ninferno::MediaKind::Image
                                                      : ninferno::MediaKind::Video;
        media.bytes = ninferno::product::media_acquire::acquire_bytes(part.source);
        media.media_type          = part.source.media_type;
        media.image_resize_policy = part.image_resize_policy;
        return media;
    });
}

// An optional fixture file: empty when the checkpoint has none.
std::string read_optional(const std::filesystem::path& path) {
    return std::filesystem::exists(path) ? read_file(path) : std::string();
}

} // namespace

int main() {
    const char* dir_text = std::getenv("NINFERNO_CHAT_TEMPLATE_PARITY_DIR");
    if (dir_text == nullptr || *dir_text == '\0') {
        std::cout << "chat template parity: NINFERNO_CHAT_TEMPLATE_PARITY_DIR not set, skipped\n";
        return 77;
    }
    try {
        const std::filesystem::path dir(dir_text);
        const std::string tokenizer_json    = read_file(dir / "tokenizer.json");
        const std::string tokenizer_config  = read_file(dir / "tokenizer_config.json");
        const std::string generation_config = read_file(dir / "generation_config.json");
        const std::string chat_template     = read_file(dir / "chat_template.jinja");
        const std::string preprocessor      = read_optional(dir / "preprocessor_config.json");
        const std::string video             = read_optional(dir / "video_preprocessor_config.json");

        ninferno::models::qwen3_5::FrontendResources resources{
            tokenizer_json, tokenizer_config, chat_template, generation_config, preprocessor,
            video};
        resources.tokenizer = std::make_shared<const fi::Tokenizer>(
            fi::TokenizerResources{tokenizer_json, tokenizer_config, generation_config});
        resources.public_token_count =
            static_cast<std::uint32_t>(resources.tokenizer->vocab_size());
        if (resources.tokenizer->encode("<|image_pad|>") != std::vector<TokenId>{kImagePad}) {
            throw std::runtime_error("unexpected <|image_pad|> id in this tokenizer");
        }
        ninferno::models::qwen3_5::FrontendOptions options;
        options.vision_enabled = !preprocessor.empty() && !video.empty();
        options.max_context    = std::numeric_limits<std::uint32_t>::max();
        const auto frontend    = ninferno::models::qwen3_5::make_frontend(resources, options);

        std::ifstream cases(dir / "cases.jsonl");
        if (!cases) { throw std::runtime_error("cannot read cases.jsonl in " + dir.string()); }
        std::size_t total = 0, mismatches = 0, unstable = 0, errors = 0, compared_tokens = 0,
                    agreed_refusals = 0, policy_refusals = 0, literal_controls = 0;
        std::string line;
        while (std::getline(cases, line)) {
            if (line.empty()) { continue; }
            const auto item = nlohmann::ordered_json::parse(line);
            const auto want = item.at("ids").get<std::vector<TokenId>>();
            const std::string id = item.at("id").get<std::string>();
            const bool refused   = item.contains("refused");
            ++total;
            std::vector<TokenId> got;
            std::vector<TokenId> again;
            if (refused) {
                // The template itself refuses (raise_exception): the product must refuse too.
                try {
                    (void)frontend.prepare(product_prompt(item));
                    if (++mismatches <= 12) {
                        std::cerr << "[" << id << "] rendered what the template refuses\n";
                    }
                } catch (const std::exception&) { ++agreed_refusals; }
                continue;
            }
            try {
                got = collapse_image_pads(ninferno::models::qwen3_5::FrontendTestAccess::inspect(
                                              frontend.prepare(product_prompt(item)))
                                              .token_ids);
                again = collapse_image_pads(ninferno::models::qwen3_5::FrontendTestAccess::inspect(
                                                frontend.prepare(product_prompt(item)))
                                                .token_ids);
            } catch (const std::exception& error) {
                const std::string expected = item.value("expect_ninferno_refusal", "");
                if (!expected.empty() && std::string(error.what()).find(expected) != std::string::npos) {
                    ++policy_refusals;
                    continue;
                }
                if (++errors <= 400) { std::cerr << "[" << id << "] refused: " << error.what() << "\n"; }
                continue;
            }
            if (item.contains("expect_ninferno_refusal") && ++errors <= 400) {
                std::cerr << "[" << id << "] rendered a request the policy refuses\n";
            }
            if (item.value("literal_controls", false) && got != want) {
                // Control-token text in message content stays text: compare the rendered text.
                if (resources.tokenizer->decode(got) == item.at("text").get<std::string>()) {
                    ++literal_controls;
                    continue;
                }
            }
            compared_tokens += want.size();
            if (got != again && ++unstable <= 5) {
                std::cerr << "[" << id << "] a second render gave different ids\n";
            }
            if (got == want) { continue; }
            if (++mismatches > 400) { continue; }
            std::size_t at = 0;
            while (at < got.size() && at < want.size() && got[at] == want[at]) { ++at; }
            const auto window = [&](const std::vector<TokenId>& ids) {
                const std::size_t begin = at > 12 ? at - 12 : 0;
                const std::size_t end   = std::min(ids.size(), at + 24);
                return nlohmann::json(resources.tokenizer->decode(std::vector<TokenId>(
                                          ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                          ids.begin() + static_cast<std::ptrdiff_t>(end))))
                    .dump();
            };
            std::cerr << "[" << id << "] ids differ at " << at << " of " << want.size()
                      << " (ninferno " << got.size() << ")\n  transformers: " << window(want)
                      << "\n  ninferno:       " << window(got) << "\n";
        }
        std::cout << "chat template parity: " << total << " requests, " << compared_tokens
                  << " reference tokens, mismatches=" << mismatches << " unstable=" << unstable
                  << " refused=" << errors << " refused_by_both=" << agreed_refusals
                  << " policy_refusals=" << policy_refusals
                  << " literal_controls_text_equal=" << literal_controls << "\n";
        return mismatches == 0 && unstable == 0 && errors == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "chat template parity error: " << error.what() << "\n";
        return 1;
    }
}
