#pragma once

#include "ninferno/types.h"
#include "structured/vocabulary.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace ninferno::structured {

struct CompilerOptions {
    // Largest accepted schema, regex, grammar or structural-tag text, and choice list.
    std::size_t max_definition_bytes = 256U * 1024U;
    std::size_t max_choices          = 4096;
    // A compile that does not finish in time returns "constraint_compile_timeout"; its worker
    // finishes in the background and its result is discarded. At most max_concurrent compiles run
    // at once; a request that cannot start one within the timeout gets "constraint_compiler_busy".
    std::chrono::milliseconds compile_timeout{10000};
    std::uint32_t max_concurrent = 4;
    // Compiled constraints are cached by their exact spec up to this many bytes.
    std::size_t cache_bytes = 512ULL << 20U;
    int compile_threads     = 8;
    // Longest insignificant whitespace run between JSON tokens (unless compact_json), and longest
    // leading whitespace run after a reasoning section (follows_reasoning).
    int json_whitespace_run      = 24;
    int reasoning_whitespace_run = 4;
};

class CompiledConstraint;

// Per-vocabulary grammar compiler. Thread-safe; compile() is called from request threads.
class ConstraintCompiler {
public:
    ConstraintCompiler(GrammarVocabulary vocabulary, CompilerOptions options = {});
    ~ConstraintCompiler();
    ConstraintCompiler(const ConstraintCompiler&)            = delete;
    ConstraintCompiler& operator=(const ConstraintCompiler&) = delete;

    // Throws OutputConstraintError; never returns null.
    [[nodiscard]] std::shared_ptr<const CompiledConstraint>
    compile(const OutputConstraintSpec& spec) const;

    [[nodiscard]] const GrammarVocabulary& vocabulary() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// One request's position in a compiled constraint. Not thread-safe.
class ConstraintMatcher {
public:
    explicit ConstraintMatcher(std::shared_ptr<const CompiledConstraint> compiled);
    ~ConstraintMatcher();
    ConstraintMatcher(ConstraintMatcher&&) noexcept;
    ConstraintMatcher& operator=(ConstraintMatcher&&) noexcept;

    // Each successful accept is one rollback step.
    [[nodiscard]] bool accept_token(TokenId token);
    [[nodiscard]] bool accept_bytes(std::string_view bytes);
    void rollback(std::uint32_t steps);
    // Writes mask_words() words: the tokens admitted next. Padding words are zero.
    void fill_mask(std::uint32_t* words);
    // The grammar accepted a stop token.
    [[nodiscard]] bool terminated() const;
    // The text accepted so far is a complete sentence of the grammar.
    [[nodiscard]] bool completed() const;

    [[nodiscard]] const CompiledConstraint& compiled() const noexcept { return *compiled_; }

private:
    struct State;
    std::shared_ptr<const CompiledConstraint> compiled_;
    std::unique_ptr<State> state_;
};

// Vocabulary-derived facts a request needs besides the matcher.
struct ConstraintVocabularyView {
    const GrammarVocabulary* vocabulary = nullptr;
    bool special_tokens_visible         = false;
    std::uint32_t mask_words            = 0;
    // Every token except the stop tokens: the mask while the constrained text has not started.
    const std::uint32_t* no_stop_mask = nullptr;
    // Only the stop tokens: the mask that ends a request whose constraint failed.
    const std::uint32_t* stop_only_mask = nullptr;
    // Tokens that spell part of the reasoning open marker (at any offset), and whitespace-only
    // tokens of up to four bytes: what a request may sample beside its grammar before it.
    const std::vector<TokenId>* open_candidates = nullptr;
    const std::vector<TokenId>* blank_tokens    = nullptr;

    // The bytes the output presentation shows for a token.
    [[nodiscard]] std::string_view presented(TokenId token) const noexcept;
};

[[nodiscard]] ConstraintVocabularyView vocabulary_view(const CompiledConstraint& compiled) noexcept;

// Whether a constraint holds a client's free grammar, an EBNF definition: context-free, as nested
// and ambiguous as the client wrote it, so the engine declares no bound on the host time of its
// mask rows (runtime::MaskRowClass::FreeGrammar). That is a grammar spec, or a client's structural
// tag with a format of type "grammar" anywhere in it. Every other constraint is a grammar the
// engine shapes (a JSON Schema or object, a choice list, a structural tag of other formats, the
// tool-call markup the Engine renders, also Gemma 4's with its own argument grammar,
// OutputConstraintSpec::engine_rendered) or a regular language (a regex). A pure function of the
// spec, whose every input the compile cache's key holds: nothing is measured.
[[nodiscard]] bool declares_free_grammar(const OutputConstraintSpec& spec);
[[nodiscard]] bool declares_free_grammar(const CompiledConstraint& compiled) noexcept;

} // namespace ninferno::structured
