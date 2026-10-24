#include "structured/request_constraint.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

namespace ninferno::structured {

namespace {

// Longest whitespace run between a reasoning section the model opened itself and the constrained
// text, as CompilerOptions::reasoning_whitespace_run allows after a prompt's reasoning section.
constexpr std::size_t kGapBytes = 4;

bool blank(std::string_view bytes) noexcept {
    return !bytes.empty() && std::all_of(bytes.begin(), bytes.end(), [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    });
}

// How `bytes` continues `marker` after `matched` bytes: 0 when it does not, else the bytes of
// the token that lie on the marker (all of them, or fewer when the token completes it and goes
// on).
std::size_t on_marker(std::string_view marker, std::size_t matched,
                      std::string_view bytes) noexcept {
    if (bytes.empty() || matched >= marker.size()) { return 0; }
    const std::size_t rest = marker.size() - matched;
    const std::size_t take = std::min(rest, bytes.size());
    return marker.compare(matched, take, bytes.substr(0, take)) == 0 ? take : 0;
}

} // namespace

std::size_t ReasoningGate::feed(std::string_view bytes) noexcept {
    if (open_) { return std::string_view::npos; }
    for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
        const char byte = bytes[offset];
        // The marker has no repeated prefix for the models NInferno serves ("</think>"), but a
        // general failure function keeps the detector exact for any marker.
        while (matched_ != 0 && byte != marker_[matched_]) {
            std::size_t fallback = 0;
            for (std::size_t length = matched_ - 1; length != 0; --length) {
                if (marker_.compare(0, length, marker_, matched_ - length, length) == 0) {
                    fallback = length;
                    break;
                }
            }
            matched_ = fallback;
        }
        if (byte == marker_[matched_]) { ++matched_; }
        if (matched_ == marker_.size()) {
            open_    = true;
            matched_ = 0;
            return offset + 1;
        }
    }
    return std::string_view::npos;
}

RequestConstraint::RequestConstraint(std::shared_ptr<const CompiledConstraint> compiled,
                                     bool starts_in_reasoning, bool may_open_reasoning)
    : compiled_(compiled), matcher_(std::move(compiled)),
      view_(vocabulary_view(matcher_.compiled())) {
    const std::string& close = view_.vocabulary->reasoning_close;
    gate_.close              = ReasoningGate(close, true);
    if (close.empty()) {
        gate_.mode = ConstraintGate::Mode::Open;
    } else if (starts_in_reasoning) {
        gate_.mode  = ConstraintGate::Mode::Reasoning;
        gate_.close = ReasoningGate(close, false);
    } else if (may_open_reasoning && !view_.vocabulary->reasoning_open.empty()) {
        gate_.mode = ConstraintGate::Mode::Pending;
    }
}

void RequestConstraint::violate(std::string reason) noexcept {
    if (violated_) { return; }
    violated_ = true;
    try {
        violation_ = std::move(reason);
    } catch (...) {}
}

void RequestConstraint::write_row(std::uint32_t* row, const ConstraintGate& gate,
                                  ConstraintMatcher& matcher) {
    const std::size_t words = view_.mask_words;
    const auto admit        = [&](TokenId token) {
        row[static_cast<std::uint32_t>(token) >> 5U] |=
            1U << (static_cast<std::uint32_t>(token) & 31U);
    };
    switch (gate.mode) {
    case ConstraintGate::Mode::Reasoning:
        std::memcpy(row, view_.no_stop_mask, words * sizeof(std::uint32_t));
        return;
    case ConstraintGate::Mode::Open:
        matcher.fill_mask(row);
        return;
    case ConstraintGate::Mode::Gap:
        matcher.fill_mask(row);
        for (const TokenId token : *view_.blank_tokens) {
            if (gate.gap_bytes + view_.presented(token).size() <= kGapBytes) { admit(token); }
        }
        return;
    case ConstraintGate::Mode::Pending: {
        if (gate.grammar_alive) {
            matcher.fill_mask(row);
        } else {
            std::memset(row, 0, words * sizeof(std::uint32_t));
        }
        const std::string_view marker = view_.vocabulary->reasoning_open;
        for (const TokenId token : *view_.open_candidates) {
            if (on_marker(marker, gate.open_matched, view_.presented(token)) != 0) { admit(token); }
        }
        return;
    }
    }
}

void RequestConstraint::reopen(Track& track) {
    // Before the open marker the grammar was at its start (Pending holds only while every token
    // lies on the marker), so the text after the reasoning starts a fresh matcher. A speculative
    // round takes a scratch one and leaves the committed matcher to its own rollback.
    if (track.simulate) {
        if (!track.scratch) { track.main_steps = track.steps; }
        scratch_.emplace(compiled_);
        track.matcher = &*scratch_;
        track.steps   = 0;
        track.scratch = true;
    } else {
        matcher_ = ConstraintMatcher(compiled_);
    }
}

bool RequestConstraint::after_reasoning(ConstraintGate& gate, std::string_view tail, Track& track,
                                        std::string& why) {
    // A whitespace run the gap admits, then the grammar's text.
    std::size_t at = 0;
    while (at < tail.size() && gate.gap_bytes < kGapBytes && blank(tail.substr(at, 1))) {
        ++at;
        ++gate.gap_bytes;
    }
    if (at == tail.size()) { return true; }
    gate.mode = ConstraintGate::Mode::Open;
    if (!track.matcher->accept_bytes(tail.substr(at))) {
        why = "content following the reasoning terminator is not admitted by the constraint";
        return false;
    }
    ++track.steps;
    return true;
}

bool RequestConstraint::advance(ConstraintGate& gate, TokenId token, Track& track,
                                std::string& why) {
    ConstraintMatcher& matcher = *track.matcher;
    switch (gate.mode) {
    case ConstraintGate::Mode::Open:
        if (matcher.terminated()) {
            why = "a token was committed after the constrained output ended";
            return false;
        }
        if (!matcher.accept_token(token)) {
            why = "committed token " + std::to_string(token) + " is not admitted by the constraint";
            return false;
        }
        ++track.steps;
        return true;
    case ConstraintGate::Mode::Gap: {
        const std::string_view bytes = view_.presented(token);
        if (blank(bytes) && gate.gap_bytes + bytes.size() <= kGapBytes) {
            gate.gap_bytes += bytes.size();
            return true;
        }
        gate.mode = ConstraintGate::Mode::Open;
        if (!advance(gate, token, track, why)) {
            gate.mode = ConstraintGate::Mode::Gap;
            return false;
        }
        return true;
    }
    case ConstraintGate::Mode::Reasoning: {
        const std::string_view bytes = view_.presented(token);
        const std::size_t tail       = gate.close.feed(bytes);
        if (tail == std::string_view::npos) { return true; }
        if (!gate.self_opened) {
            // A reasoning section the prompt opened goes straight to the grammar, which carries
            // its own whitespace prefix (follows_reasoning).
            gate.mode = ConstraintGate::Mode::Open;
            if (tail < bytes.size()) {
                if (!matcher.accept_bytes(bytes.substr(tail))) {
                    why = "content following the reasoning terminator in token " +
                          std::to_string(token) + " is not admitted by the constraint";
                    return false;
                }
                ++track.steps;
            }
            return true;
        }
        gate.mode = ConstraintGate::Mode::Gap;
        return after_reasoning(gate, bytes.substr(tail), track, why);
    }
    case ConstraintGate::Mode::Pending: {
        const std::string_view bytes  = view_.presented(token);
        const std::string_view marker = view_.vocabulary->reasoning_open;
        const std::size_t on          = on_marker(marker, gate.open_matched, bytes);
        bool grammar                  = false;
        if (gate.grammar_alive && !matcher.terminated() && matcher.accept_token(token)) {
            grammar = true;
            ++track.steps;
        }
        if (on != 0 && gate.open_matched + on == marker.size()) {
            // The model opened its reasoning: the grammar waits for its terminator.
            gate.mode        = ConstraintGate::Mode::Reasoning;
            gate.close       = ReasoningGate(view_.vocabulary->reasoning_close, false);
            gate.self_opened = true;
            reopen(track);
            const std::string_view rest = bytes.substr(on);
            if (!rest.empty()) {
                const std::size_t tail = gate.close.feed(rest);
                if (tail != std::string_view::npos) {
                    gate.mode = ConstraintGate::Mode::Gap;
                    return after_reasoning(gate, rest.substr(tail), track, why);
                }
            }
            return true;
        }
        if (on != 0) {
            gate.open_matched += on;
            if (!grammar) { gate.grammar_alive = false; }
            return true;
        }
        if (grammar) {
            // The text started: what it took while it could still have been the marker stays.
            gate.mode = ConstraintGate::Mode::Open;
            return true;
        }
        why = "committed token " + std::to_string(token) +
              " neither opens the reasoning nor is admitted by the constraint";
        return false;
    }
    }
    return false;
}

void RequestConstraint::fill_round(std::span<const TokenId> drafts, std::uint32_t* masks,
                                   std::size_t stride_words, std::uint32_t* progress) noexcept {
    const auto started      = std::chrono::steady_clock::now();
    const std::size_t words = view_.mask_words;
    const std::size_t rows  = drafts.size() + 1;
    const auto row_at       = [&](std::size_t index) { return masks + index * stride_words; };
    // A large grammar can take longer over a wide verify window than the kernels wait for a
    // silent host; each row computed tells them the host is still working.
    const auto advanced = [&] {
        if (progress != nullptr) {
            std::atomic_ref<std::uint32_t>(*progress).fetch_add(1U, std::memory_order_relaxed);
        }
    };
    const auto stop_only    = [&] {
        for (std::size_t index = 0; index < rows; ++index) {
            std::memcpy(row_at(index), view_.stop_only_mask, words * sizeof(std::uint32_t));
        }
    };
    if (violated_) {
        stop_only();
        return;
    }

    Track track{.matcher = &matcher_, .simulate = true};
    const auto restore = [&] {
        matcher_.rollback(
            static_cast<std::uint32_t>(track.scratch ? track.main_steps : track.steps));
        scratch_.reset();
    };
    try {
        ConstraintGate gate = gate_;
        write_row(row_at(0), gate, *track.matcher);
        advanced();
        bool consumable = true;
        std::string why;
        for (std::size_t index = 0; index < drafts.size(); ++index) {
            // A draft the row admits is always admitted here; one it does not ends what
            // acceptance can consume (the masked choice at this column differs from it).
            if (consumable && !advance(gate, drafts[index], track, why)) { consumable = false; }
            if (consumable && !track.matcher->terminated()) {
                write_row(row_at(index + 1), gate, *track.matcher);
                advanced();
            } else {
                // Not consumable by acceptance: its draft was rejected or the output already
                // ended. Any non-empty row is sound; repeat the previous one.
                consumable = false;
                std::memcpy(row_at(index + 1), row_at(index), words * sizeof(std::uint32_t));
            }
        }
        restore();
        masked_positions_ += rows;
    } catch (const std::exception& error) {
        try {
            restore();
        } catch (...) {}
        violate(std::string("grammar matcher failed while computing a mask: ") + error.what());
        stop_only();
    } catch (...) {
        violate("grammar matcher failed while computing a mask");
        stop_only();
    }
    const auto elapsed =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - started)
                                       .count());
    mask_host_ns_ += elapsed; // telemetry only (stats().mask_host_seconds)
}

void RequestConstraint::commit(std::span<const TokenId> tokens) noexcept {
    try {
        std::string why;
        for (const TokenId token : tokens) {
            if (violated_) { return; }
            Track track{.matcher = &matcher_, .simulate = false};
            if (!advance(gate_, token, track, why)) { violate(why); }
        }
    } catch (const std::exception& error) {
        violate(std::string("grammar matcher failed on a committed token: ") + error.what());
    } catch (...) {
        violate("grammar matcher failed on a committed token");
    }
}

void RequestConstraint::record_device_fault() noexcept {
    violate("the device did not observe the token mask release in time; the round was sampled "
            "without its mask");
}

OutputConstraintStats RequestConstraint::stats() const {
    OutputConstraintStats out;
    out.active    = true;
    out.violated  = violated_;
    out.violation = violation_;
    out.complete  = !violated_ && gate_.mode == ConstraintGate::Mode::Open && matcher_.completed();
    out.masked_positions  = masked_positions_;
    out.mask_host_seconds = static_cast<double>(mask_host_ns_) * 1.0e-9;
    return out;
}

} // namespace ninferno::structured
