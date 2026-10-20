#include "structured/constraint.h"

#include "structured/schema_support.h"

#include <log_hook.h>
#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <future>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ninferno::structured {

struct SharedVocabulary {
    GrammarVocabulary vocabulary;
    std::uint32_t mask_words = 0;
    std::vector<std::uint32_t> no_stop_mask;
    std::vector<std::uint32_t> stop_only_mask;
    std::vector<TokenId> open_candidates;
    std::vector<TokenId> blank_tokens;
};

class CompiledConstraint {
public:
    CompiledConstraint(xgrammar::CompiledGrammar compiled,
                       std::shared_ptr<const SharedVocabulary> shared, bool visible)
        : grammar(std::move(compiled)), vocabulary(std::move(shared)),
          special_tokens_visible(visible), memory_bytes(grammar.MemorySizeBytes()) {}

    xgrammar::CompiledGrammar grammar;
    std::shared_ptr<const SharedVocabulary> vocabulary;
    bool special_tokens_visible = false;
    std::size_t memory_bytes    = 0;
};

namespace {

void set_bit(std::vector<std::uint32_t>& words, std::uint32_t token, bool value) {
    if (value) {
        words[token >> 5U] |= 1U << (token & 31U);
    } else {
        words[token >> 5U] &= ~(1U << (token & 31U));
    }
}

std::shared_ptr<const SharedVocabulary> make_shared_vocabulary(GrammarVocabulary vocabulary) {
    if (vocabulary.token_domain == 0 || vocabulary.token_bytes.size() != vocabulary.token_domain ||
        vocabulary.special.size() != vocabulary.token_domain) {
        throw std::invalid_argument("grammar vocabulary does not cover its token domain");
    }
    for (const std::int32_t stop : vocabulary.stop_token_ids) {
        if (stop < 0 || static_cast<std::uint32_t>(stop) >= vocabulary.token_domain) {
            throw std::invalid_argument("grammar vocabulary stop token is outside the domain");
        }
    }
    auto shared        = std::make_shared<SharedVocabulary>();
    shared->mask_words = mask_words_for(vocabulary.token_domain);
    shared->no_stop_mask.assign(shared->mask_words, 0U);
    shared->stop_only_mask.assign(shared->mask_words, 0U);
    for (std::uint32_t token = 0; token < vocabulary.token_domain; ++token) {
        set_bit(shared->no_stop_mask, token, true);
    }
    for (const std::int32_t stop : vocabulary.stop_token_ids) {
        set_bit(shared->no_stop_mask, static_cast<std::uint32_t>(stop), false);
        set_bit(shared->stop_only_mask, static_cast<std::uint32_t>(stop), true);
    }
    // Presented in both variants (a marker's special tokens are structural, not special).
    const std::string& open = vocabulary.reasoning_open;
    for (std::uint32_t token = 0; token < vocabulary.token_domain; ++token) {
        const std::string& bytes = vocabulary.token_bytes[token];
        if (vocabulary.special[token] || bytes.empty()) { continue; }
        if (bytes.size() <= 4 && bytes.find_first_not_of(" \t\n\r") == std::string::npos) {
            shared->blank_tokens.push_back(static_cast<TokenId>(token));
        }
        if (open.empty() || open.find(bytes.front()) == std::string::npos) { continue; }
        for (std::size_t at = 0; at < open.size(); ++at) {
            const std::size_t take = std::min(open.size() - at, bytes.size());
            if (open.compare(at, take, bytes, 0, take) == 0) {
                shared->open_candidates.push_back(static_cast<TokenId>(token));
                break;
            }
        }
    }
    shared->vocabulary = std::move(vocabulary);
    return shared;
}

std::string escape_ebnf_literal(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (byte < 0x20U || byte == 0x7fU) {
            char buffer[8];
            std::snprintf(buffer, sizeof(buffer), "\\x%02x", byte);
            out += buffer;
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

const char* unsupported_code(OutputConstraintKind kind) {
    switch (kind) {
    case OutputConstraintKind::JsonObject:
    case OutputConstraintKind::JsonSchema:
        return "unsupported_schema";
    case OutputConstraintKind::Regex:
        return "unsupported_regex";
    case OutputConstraintKind::Choice:
        return "invalid_choice";
    case OutputConstraintKind::Grammar:
        return "unsupported_grammar";
    case OutputConstraintKind::StructuralTag:
    case OutputConstraintKind::ToolCalls:
        return "unsupported_structural_tag";
    }
    return "unsupported_schema";
}

const char* invalid_code(OutputConstraintKind kind) {
    switch (kind) {
    case OutputConstraintKind::JsonObject:
    case OutputConstraintKind::JsonSchema:
        return "invalid_schema";
    case OutputConstraintKind::Regex:
        return "invalid_regex";
    case OutputConstraintKind::Choice:
        return "invalid_choice";
    case OutputConstraintKind::Grammar:
        return "invalid_grammar";
    case OutputConstraintKind::StructuralTag:
    case OutputConstraintKind::ToolCalls:
        return "invalid_structural_tag";
    }
    return "invalid_schema";
}

std::string cache_key(const OutputConstraintSpec& spec) {
    std::string key;
    key.reserve(spec.definition.size() + 32);
    key.push_back(static_cast<char>('0' + static_cast<int>(spec.kind)));
    key.push_back(spec.compact_json ? 'c' : 'w');
    key.push_back(spec.follows_reasoning ? 'r' : 'n');
    key.push_back(spec.special_tokens_visible ? 'v' : 'h');
    key.push_back('\0');
    key += spec.definition;
    if (!spec.answer_schema.empty()) {
        key.push_back('\0');
        key += "answer:";
        key += spec.answer_schema;
    }
    for (const std::string& choice : spec.choices) {
        key.push_back('\0');
        key += std::to_string(choice.size());
        key.push_back(':');
        key += choice;
    }
    return key;
}

// JSON inside a structural tag (tool-call arguments, a client's tag) keeps the whitespace bound of
// a JSON response format: every json_schema node without its own max_whitespace_cnt gets `run`.
// xgrammar leaves those runs unbounded by default, so a model could write whitespace inside the
// arguments until max_tokens. A definition that is not JSON is left for xgrammar to reject.
std::string bound_structural_tag_whitespace(const std::string& definition, int run) {
    nlohmann::ordered_json tag = nlohmann::ordered_json::parse(definition, nullptr, false);
    if (tag.is_discarded()) { return definition; }
    const auto bound = [run](const auto& self, nlohmann::ordered_json& node) -> void {
        if (node.is_array()) {
            for (nlohmann::ordered_json& value : node) { self(self, value); }
            return;
        }
        if (!node.is_object()) { return; }
        const auto type          = node.find("type");
        const bool schema_format = type != node.end() && *type == "json_schema";
        if (schema_format && !node.contains("max_whitespace_cnt")) {
            node["max_whitespace_cnt"] = run;
        }
        for (auto& entry : node.items()) {
            // The schema itself is the client's JSON Schema, not a format node.
            if (!(schema_format && entry.key() == "json_schema")) { self(self, entry.value()); }
        }
    };
    bound(bound, tag);
    return tag.dump();
}

struct Variant {
    xgrammar::TokenizerInfo info;
    std::unique_ptr<xgrammar::GrammarCompiler> compiler;
};

} // namespace

std::string_view ConstraintVocabularyView::presented(TokenId token) const noexcept {
    if (vocabulary == nullptr || token < 0 ||
        static_cast<std::uint32_t>(token) >= vocabulary->token_domain) {
        return {};
    }
    const auto index = static_cast<std::size_t>(token);
    if (vocabulary->special[index] && !special_tokens_visible) { return {}; }
    return vocabulary->token_bytes[index];
}

ConstraintVocabularyView vocabulary_view(const CompiledConstraint& compiled) noexcept {
    const SharedVocabulary& shared = *compiled.vocabulary;
    return ConstraintVocabularyView{
        .vocabulary             = &shared.vocabulary,
        .special_tokens_visible = compiled.special_tokens_visible,
        .mask_words             = shared.mask_words,
        .no_stop_mask           = shared.no_stop_mask.data(),
        .stop_only_mask         = shared.stop_only_mask.data(),
        .open_candidates        = &shared.open_candidates,
        .blank_tokens           = &shared.blank_tokens,
    };
}

struct ConstraintCompiler::Impl : std::enable_shared_from_this<ConstraintCompiler::Impl> {
    std::shared_ptr<const SharedVocabulary> shared;
    CompilerOptions options;

    std::mutex variant_mutex;
    std::unique_ptr<Variant> variants[2];

    std::mutex cache_mutex;
    using CacheEntry = std::pair<std::string, std::shared_ptr<const CompiledConstraint>>;
    std::list<CacheEntry> lru;
    std::unordered_map<std::string, std::list<CacheEntry>::iterator> index;
    std::size_t cached_bytes = 0;

    std::mutex slot_mutex;
    std::condition_variable slot_cv;
    std::uint32_t active = 0;

    // One compilation per key at a time: a request for a key being compiled waits on that
    // compilation, and one that outlives its first caller's timeout still lands in the cache.
    using Compiled = std::shared_future<std::shared_ptr<const CompiledConstraint>>;
    std::mutex inflight_mutex;
    std::unordered_map<std::string, Compiled> inflight;

    void finish_inflight(const std::string& key) {
        std::lock_guard lock(inflight_mutex);
        inflight.erase(key);
    }

    Variant& variant(bool visible) {
        std::lock_guard lock(variant_mutex);
        std::unique_ptr<Variant>& slot = variants[visible ? 1 : 0];
        if (!slot) {
            const GrammarVocabulary& vocabulary = shared->vocabulary;
            std::vector<std::string> encoded(vocabulary.token_domain);
            for (std::uint32_t token = 0; token < vocabulary.token_domain; ++token) {
                // An empty string is XGrammar's "never admitted" (special) token.
                if (!vocabulary.special[token] || visible) {
                    encoded[token] = vocabulary.token_bytes[token];
                }
            }
            std::vector<std::int32_t> stops(vocabulary.stop_token_ids.begin(),
                                            vocabulary.stop_token_ids.end());
            auto made  = std::make_unique<Variant>(Variant{
                xgrammar::TokenizerInfo(encoded, xgrammar::VocabType::RAW,
                                        static_cast<int>(vocabulary.token_domain), stops, false),
                nullptr});
            made->compiler = std::make_unique<xgrammar::GrammarCompiler>(
                made->info, options.compile_threads, false);
            slot = std::move(made);
        }
        return *slot;
    }

    std::shared_ptr<const CompiledConstraint> cached(const std::string& key) {
        std::lock_guard lock(cache_mutex);
        const auto it = index.find(key);
        if (it == index.end()) { return nullptr; }
        lru.splice(lru.begin(), lru, it->second);
        return it->second->second;
    }

    void insert(std::string key, std::shared_ptr<const CompiledConstraint> compiled) {
        std::lock_guard lock(cache_mutex);
        if (index.contains(key) || compiled->memory_bytes > options.cache_bytes) { return; }
        cached_bytes += compiled->memory_bytes;
        lru.emplace_front(std::move(key), std::move(compiled));
        index.emplace(lru.front().first, lru.begin());
        while (cached_bytes > options.cache_bytes && !lru.empty()) {
            cached_bytes -= lru.back().second->memory_bytes;
            index.erase(lru.back().first);
            lru.pop_back();
        }
    }

    xgrammar::Grammar build(const OutputConstraintSpec& spec, Variant& target) const {
        const bool whitespace = !spec.compact_json;
        const std::optional<std::pair<std::string, std::string>> separators =
            whitespace ? std::nullopt
                       : std::optional<std::pair<std::string, std::string>>({",", ":"});
        const std::optional<int> whitespace_run =
            whitespace ? std::optional<int>(options.json_whitespace_run) : std::nullopt;
        std::optional<xgrammar::Grammar> grammar;
        switch (spec.kind) {
        case OutputConstraintKind::JsonObject:
            grammar = xgrammar::Grammar::FromJSONSchema(R"({"type":"object"})", whitespace,
                                                        std::nullopt, separators, false,
                                                        whitespace_run);
            break;
        case OutputConstraintKind::JsonSchema: {
            const nlohmann::ordered_json parsed =
                nlohmann::ordered_json::parse(spec.definition, nullptr, false);
            if (parsed.is_discarded()) {
                throw OutputConstraintError("invalid_schema", "JSON Schema is not valid JSON");
            }
            const nlohmann::ordered_json normalized = normalize_supported_json_schema(parsed);
            grammar = xgrammar::Grammar::FromJSONSchema(normalized.dump(), whitespace,
                                                        std::nullopt, separators, true,
                                                        whitespace_run);
            break;
        }
        case OutputConstraintKind::Regex:
            grammar = xgrammar::Grammar::FromRegex(spec.definition);
            break;
        case OutputConstraintKind::Choice: {
            std::string ebnf = "root ::= ";
            for (std::size_t index = 0; index < spec.choices.size(); ++index) {
                if (index != 0) { ebnf += " | "; }
                ebnf += escape_ebnf_literal(spec.choices[index]);
            }
            grammar = xgrammar::Grammar::FromEBNF(ebnf, "root");
            break;
        }
        case OutputConstraintKind::Grammar:
            grammar = xgrammar::Grammar::FromEBNF(spec.definition, "root");
            break;
        case OutputConstraintKind::StructuralTag: {
            auto result = xgrammar::Grammar::FromStructuralTag(
                bound_structural_tag_whitespace(spec.definition, options.json_whitespace_run),
                target.info);
            if (const auto* error = std::get_if<xgrammar::StructuralTagError>(&result)) {
                throw OutputConstraintError(
                    "invalid_structural_tag",
                    std::visit([](const auto& e) { return std::string(e.what()); }, *error));
            }
            grammar = std::get<xgrammar::Grammar>(std::move(result));
            if (!spec.answer_schema.empty()) {
                // response_format beside tools: the calls, or an answer in the format.
                const nlohmann::ordered_json answer =
                    nlohmann::ordered_json::parse(spec.answer_schema, nullptr, false);
                if (answer.is_discarded()) {
                    throw OutputConstraintError("invalid_schema", "JSON Schema is not valid JSON");
                }
                grammar = xgrammar::Grammar::Union(
                    {*grammar, xgrammar::Grammar::FromJSONSchema(
                                   normalize_supported_json_schema(answer).dump(), whitespace,
                                   std::nullopt, separators, true, whitespace_run)});
            }
            break;
        }
        case OutputConstraintKind::ToolCalls:
            throw OutputConstraintError("invalid_tool_constraint",
                                        "tool-call constraints are rendered by the Engine");
        }
        if (spec.follows_reasoning) {
            const std::string prefix =
                "root ::= [ \\t\\n\\r]{0," + std::to_string(options.reasoning_whitespace_run) + "}";
            grammar = xgrammar::Grammar::Concat({xgrammar::Grammar::FromEBNF(prefix), *grammar});
        }
        return std::move(*grammar);
    }

    std::shared_ptr<const CompiledConstraint> build_and_compile(const OutputConstraintSpec& spec) {
        Variant& target = variant(spec.special_tokens_visible);
        ninferno_xgrammar::WarningCapture capture;
        xgrammar::Grammar grammar = [&] {
            try {
                return build(spec, target);
            } catch (const OutputConstraintError&) { throw; } catch (const std::exception& error) {
                throw OutputConstraintError(invalid_code(spec.kind),
                                            ninferno_xgrammar::plain_error_message(error));
            }
        }();
        const auto reject_warnings = [&] {
            if (!capture.warnings().empty()) {
                throw OutputConstraintError(unsupported_code(spec.kind),
                                            "the grammar compiler would not enforce part of the "
                                            "constraint: " +
                                                capture.warnings().front());
            }
        };
        reject_warnings();
        std::optional<xgrammar::CompiledGrammar> compiled;
        try {
            compiled.emplace(target.compiler->CompileGrammar(grammar));
        } catch (const std::exception& error) {
            throw OutputConstraintError(invalid_code(spec.kind),
                                        ninferno_xgrammar::plain_error_message(error));
        }
        // Regex-to-FSM lowering reports its own degradations (for example an ignored lookahead)
        // during compilation.
        reject_warnings();
        return std::make_shared<const CompiledConstraint>(std::move(*compiled), shared,
                                                          spec.special_tokens_visible);
    }
};

ConstraintCompiler::ConstraintCompiler(GrammarVocabulary vocabulary, CompilerOptions options)
    : impl_(std::make_shared<Impl>()) {
    impl_->shared  = make_shared_vocabulary(std::move(vocabulary));
    impl_->options = options;
}

ConstraintCompiler::~ConstraintCompiler() = default;

const GrammarVocabulary& ConstraintCompiler::vocabulary() const noexcept {
    return impl_->shared->vocabulary;
}

std::shared_ptr<const CompiledConstraint>
ConstraintCompiler::compile(const OutputConstraintSpec& spec) const {
    const CompilerOptions& options = impl_->options;
    std::size_t bytes              = spec.definition.size();
    for (const std::string& choice : spec.choices) { bytes += choice.size(); }
    if (bytes > options.max_definition_bytes) {
        throw OutputConstraintError("constraint_too_large",
                                    "constraint definition exceeds " +
                                        std::to_string(options.max_definition_bytes) + " bytes");
    }
    if (spec.kind == OutputConstraintKind::Choice) {
        if (spec.choices.empty()) {
            throw OutputConstraintError("invalid_choice", "choice requires at least one string");
        }
        if (spec.choices.size() > options.max_choices) {
            throw OutputConstraintError("constraint_too_large",
                                        "choice accepts at most " +
                                            std::to_string(options.max_choices) + " strings");
        }
    } else if (!spec.choices.empty()) {
        throw OutputConstraintError("invalid_choice", "choices are only valid for a choice");
    }
    if (spec.kind == OutputConstraintKind::ToolCalls || !spec.tools.empty()) {
        throw OutputConstraintError("invalid_tool_constraint",
                                    "tool-call constraints are rendered by the Engine");
    }
    if (!spec.answer_schema.empty() && spec.kind != OutputConstraintKind::StructuralTag) {
        throw OutputConstraintError("invalid_tool_constraint",
                                    "an answer schema is only an alternative to tool-call markup");
    }
    if (spec.definition.empty() && (spec.kind == OutputConstraintKind::JsonSchema ||
                                    spec.kind == OutputConstraintKind::Regex ||
                                    spec.kind == OutputConstraintKind::Grammar ||
                                    spec.kind == OutputConstraintKind::StructuralTag)) {
        throw OutputConstraintError(invalid_code(spec.kind), "constraint definition is empty");
    }

    std::string key = cache_key(spec);
    if (auto hit = impl_->cached(key)) { return hit; }

    const auto deadline = std::chrono::steady_clock::now() + options.compile_timeout;
    const auto wait_for = [&](const Impl::Compiled& compiled) {
        if (compiled.wait_until(deadline) != std::future_status::ready) {
            throw OutputConstraintError(
                "constraint_compile_timeout",
                "grammar compilation did not finish within " +
                    std::to_string(options.compile_timeout.count()) + " ms");
        }
        return compiled.get();
    };

    auto promise = std::make_shared<std::promise<std::shared_ptr<const CompiledConstraint>>>();
    std::optional<Impl::Compiled> running;
    Impl::Compiled compiled;
    {
        std::lock_guard lock(impl_->inflight_mutex);
        if (const auto it = impl_->inflight.find(key); it != impl_->inflight.end()) {
            running = it->second;
        } else {
            compiled = promise->get_future().share();
            impl_->inflight.emplace(key, compiled);
        }
    }
    if (running) { return wait_for(*running); }
    // This call compiles the key; every way out before the thread owns it answers the waiters.
    const auto abandon = [&](std::exception_ptr error) {
        try {
            promise->set_exception(error);
        } catch (...) {}
        impl_->finish_inflight(key);
    };
    {
        std::unique_lock lock(impl_->slot_mutex);
        if (!impl_->slot_cv.wait_until(lock, deadline,
                                       [&] { return impl_->active < options.max_concurrent; })) {
            const OutputConstraintError busy("constraint_compiler_busy",
                                             "no grammar compiler slot became free in time");
            lock.unlock();
            abandon(std::make_exception_ptr(busy));
            throw busy;
        }
        ++impl_->active;
    }

    try {
        std::thread([impl = impl_, spec, key, promise]() mutable {
            try {
                auto built = impl->build_and_compile(spec);
                impl->insert(key, built);
                promise->set_value(std::move(built));
            } catch (...) {
                try {
                    promise->set_exception(std::current_exception());
                } catch (...) {}
            }
            impl->finish_inflight(key);
            {
                std::lock_guard lock(impl->slot_mutex);
                --impl->active;
            }
            impl->slot_cv.notify_one();
        }).detach();
    } catch (...) {
        {
            std::lock_guard lock(impl_->slot_mutex);
            --impl_->active;
        }
        impl_->slot_cv.notify_one();
        abandon(std::current_exception());
        throw;
    }
    return wait_for(compiled);
}

struct ConstraintMatcher::State {
    explicit State(const xgrammar::CompiledGrammar& grammar) : matcher(grammar) {}
    xgrammar::GrammarMatcher matcher;
};

ConstraintMatcher::ConstraintMatcher(std::shared_ptr<const CompiledConstraint> compiled)
    : compiled_(std::move(compiled)) {
    if (!compiled_) { throw std::invalid_argument("constraint matcher requires a constraint"); }
    state_ = std::make_unique<State>(compiled_->grammar);
}

ConstraintMatcher::~ConstraintMatcher()                                       = default;
ConstraintMatcher::ConstraintMatcher(ConstraintMatcher&&) noexcept            = default;
ConstraintMatcher& ConstraintMatcher::operator=(ConstraintMatcher&&) noexcept = default;

bool ConstraintMatcher::accept_token(TokenId token) {
    return state_->matcher.AcceptToken(static_cast<std::int32_t>(token));
}

bool ConstraintMatcher::accept_bytes(std::string_view bytes) {
    return state_->matcher.AcceptString(std::string(bytes));
}

void ConstraintMatcher::rollback(std::uint32_t steps) {
    if (steps != 0) { state_->matcher.Rollback(static_cast<int>(steps)); }
}

void ConstraintMatcher::fill_mask(std::uint32_t* words) {
    const std::uint32_t domain = compiled_->vocabulary->vocabulary.token_domain;
    const std::uint32_t total  = compiled_->vocabulary->mask_words;
    std::int64_t shape         = xgrammar::GetBitmaskSize(static_cast<int>(domain));
    DLTensor tensor{};
    tensor.data   = words;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim   = 1;
    tensor.dtype  = xgrammar::GetBitmaskDLType();
    tensor.shape  = &shape;
    state_->matcher.FillNextTokenBitmask(&tensor);
    std::fill(words + shape, words + total, 0U);
    // The last real word may carry bits past the domain; the samplers never read them, but keep
    // the row exact.
    if ((domain & 31U) != 0U) { words[domain >> 5U] &= (1U << (domain & 31U)) - 1U; }
}

bool ConstraintMatcher::terminated() const { return state_->matcher.IsTerminated(); }

bool ConstraintMatcher::completed() const {
    return state_->matcher.IsTerminated() || state_->matcher.IsCompleted();
}

} // namespace ninferno::structured
