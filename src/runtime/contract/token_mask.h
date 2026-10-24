#pragma once

#include "ninferno/types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::runtime {

// What a constrained row's constraint declares about the host time of its next mask rows. Nothing
// is measured (docs/maintainer/deterministic-decisions.md): the class follows from what the
// constraint was compiled from (structured::declares_free_grammar) and where its committed tokens
// left it (structured::RequestConstraint::free_grammar_rows), so the decisions below are a pure
// function of the step's constraints, their committed tokens and the draft shape.
enum class MaskRowClass : std::uint8_t {
    // A grammar the engine shapes (a JSON Schema or object, a choice list, the tool-call markup, a
    // structural tag without an EBNF format) or a regular language (a regex). Also a free
    // grammar's rows inside a reasoning section, which copy a fixed mask and match nothing.
    Shaped,
    // A client's EBNF grammar (also as a format of a client's structural tag) outside a reasoning
    // section: context-free, nested and ambiguous as the client wrote it.
    FreeGrammar,
};

// The class of a constrained row from its constraint's free_grammar_rows().
[[nodiscard]] constexpr MaskRowClass mask_row_class(bool free_grammar_rows) noexcept {
    return free_grammar_rows ? MaskRowClass::FreeGrammar : MaskRowClass::Shaped;
}

// The declared host time of one mask row of each class. These are estimates, not measurements;
// they await calibration against the host time the responses report (constraint_mask_ms over
// constraint_masked_n) on lab hosts.
//
// A step computes its masks while its kernels run, and the samplers wait for them only up to a
// bound (ops::kTokenMaskReleaseWaitNs, 1.5 s without progress). A step whose declared mask time
// passes an eighth of that bound (kLaunchMaskOverlapNs) fills and releases its masks before its
// launch instead, so no sampler waits on them, and its constrained DFlash rows verify without
// drafts (their drafts reach the host only mid-round). The measured rule this replaces compared
// the same eighth with the slowest row its requests had computed.
//
// A shaped row: xgrammar takes it from the token-mask cache it built at compile time and checks
// only the tokens that cache left uncertain. At a declared 1 ms a row, the widest step the
// Program builds (kMaximumConcurrency rows of 16 verify columns, 128 ms) still overlaps its masks,
// so JSON Schema, choice, regex and tool-call requests keep their DFlash drafts at every shape:
// what the measured rule decided for every row it timed below 1.46 ms.
inline constexpr std::uint64_t kShapedMaskRowNs = 1'000'000;
// A free grammar's row has no time the engine can declare (a deeply nested grammar takes hundreds
// of milliseconds a row). It is priced at the samplers' whole wait bound, so a step that holds one
// always fills its masks before its launch, and no sampler waits on a free grammar. Inside a
// reasoning section the row is shaped: a speculative round whose drafts cross the terminator
// computes up to its draft count (at most 15) of grammar rows while its kernels run, each row
// renewing the samplers' wait, as the measured rule's first slow round did; the step after the
// terminator is committed fills its masks first.
inline constexpr std::uint64_t kFreeGrammarMaskRowNs = 1'500'000'000;
// An eighth of ops::kTokenMaskReleaseWaitNs, restated so that this contract includes no CUDA
// header (decode.cpp asserts that the two agree).
inline constexpr std::uint64_t kLaunchMaskOverlapNs = 187'500'000;

[[nodiscard]] constexpr std::uint64_t declared_mask_row_ns(MaskRowClass row_class) noexcept {
    return row_class == MaskRowClass::FreeGrammar ? kFreeGrammarMaskRowNs : kShapedMaskRowNs;
}

// Whether a step whose masks are declared to take `declared_ns` (the sum of each constrained row's
// declared_mask_row_ns times the mask rows it fills) computes them while its kernels run, rather
// than before its launch.
[[nodiscard]] constexpr bool masks_overlap_launch(std::uint64_t declared_ns) noexcept {
    return declared_ns <= kLaunchMaskOverlapNs;
}

static_assert(!masks_overlap_launch(kFreeGrammarMaskRowNs),
              "one free grammar row must fill its step's masks before the launch");

// One constrained row of a decode step.
struct StepMaskRow {
    MaskRowClass row_class = MaskRowClass::Shaped;
    // The row speculates this step (MTP or DFlash); a plain row samples one column.
    bool speculative = false;
};

// How a decode step computes its constrained rows' masks (plan_step_masks).
struct StepMaskPlan {
    // The declared host time of the masks as launched: an MTP row masks its draft_window + 1
    // columns, a DFlash or plain row one.
    std::uint64_t mask_ns = 0;
    // The same with every speculative row masked over its draft_window + 1 columns.
    std::uint64_t drafted_ns = 0;
    // The step fills and releases its masks before its launch (mask_ns does not overlap).
    bool masks_first = false;
    // The step's constrained DFlash rows draft: their masks follow the drafts mid-round
    // (drafted_ns overlaps). Otherwise they verify with a zero draft extent.
    bool dflash_masks_follow_drafts = false;
};

// The mask plan of a decode step from its constrained rows alone (`constrained`, any order), its
// backend and its widest draft window; `any_speculative`: the step holds a speculative row,
// constrained or not. A pure function: nothing is timed.
[[nodiscard]] constexpr StepMaskPlan plan_step_masks(std::span<const StepMaskRow> constrained,
                                                     bool mtp, bool any_speculative,
                                                     std::uint32_t draft_window) noexcept {
    const std::uint64_t drafted_columns = std::uint64_t{draft_window} + 1U;
    StepMaskPlan plan;
    for (const StepMaskRow& row : constrained) {
        const std::uint64_t row_ns = declared_mask_row_ns(row.row_class);
        plan.mask_ns += row_ns * (row.speculative && mtp ? drafted_columns : 1U);
        plan.drafted_ns += row_ns * (row.speculative ? drafted_columns : 1U);
    }
    plan.masks_first = !masks_overlap_launch(plan.mask_ns);
    plan.dflash_masks_follow_drafts =
        !mtp && any_speculative && !constrained.empty() && masks_overlap_launch(plan.drafted_ns);
    return plan;
}

// Engine side of the constrained-decoding mask protocol for one Program execution unit. Rows are
// aligned with the unit's membership order (a prefill unit has one row).
//
// The Program asks constrained() before it prepares the unit, points each constrained row's
// sampling configuration at a fixed mask buffer, launches the unit, and only then calls fill() for
// every constrained row, so mask computation on the host overlaps the device forward pass (a unit
// whose masks are declared slow, masks_overlap_launch(), fills and releases them before its
// launch). The sampling and acceptance kernels wait for the Program's release of the unit's mask
// epoch before reading a mask; the Program releases on every path, including exceptions.
class TokenMaskSource {
public:
    virtual ~TokenMaskSource() = default;

    [[nodiscard]] virtual bool constrained(std::size_t row) const noexcept = 0;

    // Writes drafts.size()+1 mask rows of stride_words 32-bit words each. Row j constrains the
    // token sampled after the committed output plus drafts[0..j). Must not throw: a failure is
    // recorded by the source and expressed as a mask that ends the request. The source advances
    // *progress after every row it computes, so the waiting kernels know the host is working.
    virtual void fill(std::size_t row, std::span<const TokenId> drafts, std::uint32_t* masks,
                      std::size_t stride_words, std::uint32_t* progress) noexcept = 0;

    // What a constrained row's constraint declares about the host time of its mask rows (read for
    // a constrained row only).
    [[nodiscard]] virtual MaskRowClass row_class(std::size_t row) const noexcept = 0;

    // The device did not observe this row's mask release within the kernels' wait bound; the
    // unit's tokens for the row were sampled without the mask.
    virtual void fault(std::size_t row) noexcept = 0;
};

} // namespace ninferno::runtime
