#pragma once

#include "ninferno/types.h"
#include "structured/constraint.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ninferno::structured {

// Byte-level detector for the reasoning-section terminator in the presented output stream. It
// mirrors the output presentation, which splits reasoning from content at the same bytes.
class ReasoningGate {
public:
    ReasoningGate() = default;
    ReasoningGate(std::string marker, bool open) : marker_(std::move(marker)), open_(open) {}

    [[nodiscard]] bool open() const noexcept { return open_; }

    // Feeds one token's presented bytes. Returns the byte offset just past the marker when this
    // token completes it (the gate is then open), or npos.
    std::size_t feed(std::string_view bytes) noexcept;

private:
    std::string marker_;
    std::size_t matched_ = 0;
    bool open_           = true;
};

// Where a constrained output stands relative to its constrained text.
struct ConstraintGate {
    enum class Mode : std::uint8_t {
        // Before the first token of a model that opens its reasoning itself (Gemma 4's
        // <|channel>thought, gpt-oss's analysis channel, Qwen3's <think>): a token continues
        // either the reasoning open marker or the constrained text, which ever it can.
        Pending,
        // Inside a reasoning section: unconstrained, stop tokens withheld, until its terminator.
        Reasoning,
        // After a reasoning section the model opened: a short whitespace run before the text.
        Gap,
        // The constrained text: every token masked by the grammar.
        Open,
    };
    Mode mode = Mode::Open;
    // Pending: bytes of the open marker matched so far, and whether the grammar admitted every
    // token too.
    std::size_t open_matched = 0;
    bool grammar_alive       = true;
    // Reasoning: the model opened it (its terminator leads to the Gap, not straight to the text).
    bool self_opened = false;
    // Gap: whitespace bytes taken.
    std::size_t gap_bytes = 0;
    // Reasoning: the terminator detector.
    ReasoningGate close;
};

// One request's constrained-decoding state. The Engine worker owns it; nothing here is shared.
//
// Positions sampled while the reasoning section is open are unconstrained except that stop
// tokens are withheld, so a request cannot finish before its constrained content. From the token
// after the reasoning terminator (or from the first token when there is no reasoning), every
// position is masked by the grammar, and stop tokens are admitted only where the grammar accepts
// end of output. A model that opens its reasoning itself may do so at the first token: until a
// token departs from the open marker, rows admit both the marker's continuation and the grammar.
class RequestConstraint {
public:
    // may_open_reasoning: the model may open a reasoning section itself at the first token (the
    // prompt allows it and the output is presented, not raw), as the output session splits it.
    RequestConstraint(std::shared_ptr<const CompiledConstraint> compiled, bool starts_in_reasoning,
                      bool may_open_reasoning = false);

    // Every round of a constrained request is masked, including rounds inside an open reasoning
    // section (stop tokens only). Writes drafts.size()+1 mask rows of stride_words each. Row j
    // admits the tokens that may follow the committed output plus drafts[0..j). Rows after a draft
    // the grammar rejects, or after an accepted stop token, are not consumable by acceptance and
    // repeat the last valid row. The matcher is restored before returning. On an internal failure
    // the constraint is marked violated and every row admits only stop tokens, which ends the
    // request. A non-null progress counter is advanced after every row computed.
    void fill_round(std::span<const TokenId> drafts, std::uint32_t* masks, std::size_t stride_words,
                    std::uint32_t* progress = nullptr) noexcept;

    // Advances with tokens the Engine committed to the sequence (model-sampled or injected).
    void commit(std::span<const TokenId> tokens) noexcept;

    // The device reported that a mask could not be applied in time for this request.
    void record_device_fault() noexcept;

    [[nodiscard]] bool violated() const noexcept { return violated_; }

    // The slowest mask row fill_round has computed, in nanoseconds (a round's time over its rows).
    [[nodiscard]] std::uint64_t slowest_row_ns() const noexcept { return slowest_row_ns_; }
    [[nodiscard]] OutputConstraintStats stats() const;
    [[nodiscard]] std::uint32_t mask_words() const noexcept { return view_.mask_words; }

private:
    // The matcher a token advances and the steps it took there.
    struct Track {
        ConstraintMatcher* matcher = nullptr;
        // Successful accepts on `matcher` (a speculative round rolls them back).
        std::int64_t steps = 0;
        bool simulate      = false;
        // A speculative round moved to the scratch matcher; main_steps are the committed one's.
        bool scratch            = false;
        std::int64_t main_steps = 0;
    };

    // Moves `gate` and the track's matcher past one token. False when the token is not
    // admitted (a committed one is then a violation).
    bool advance(ConstraintGate& gate, TokenId token, Track& track, std::string& why);
    // Bytes after a self-opened reasoning's terminator: the Gap, then the grammar.
    bool after_reasoning(ConstraintGate& gate, std::string_view tail, Track& track,
                         std::string& why);
    // The grammar restarts after a reasoning section the model opened.
    void reopen(Track& track);
    void violate(std::string reason) noexcept;
    void write_row(std::uint32_t* row, const ConstraintGate& gate, ConstraintMatcher& matcher);

    std::shared_ptr<const CompiledConstraint> compiled_;
    ConstraintMatcher matcher_;
    std::optional<ConstraintMatcher> scratch_;
    ConstraintVocabularyView view_;
    ConstraintGate gate_;
    bool violated_ = false;
    std::string violation_;
    std::uint64_t masked_positions_ = 0;
    std::uint64_t mask_host_ns_     = 0;
    std::uint64_t slowest_row_ns_   = 0;
};

} // namespace ninferno::structured
