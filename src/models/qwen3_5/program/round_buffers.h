#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninferno::models::qwen3_5 {

// The Program's MTP round domain: transfer formats, envelopes and round state hold up to the widest
// verify it runs (15 drafts, 16 columns: DFlash's as well). A family's own MTP may draft fewer
// (validate_program_options: Qwen 3.5's MTP layer stops at kMaximumMtpDraftTokens).
inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 15;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

struct RoundStateSpec {
    std::int32_t hidden          = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    SpeculativeBackend backend   = SpeculativeBackend::None;
    bool causal_scoring          = false;
};

// The most ordinary decode steps one round runs back to back (ProgramImpl::launch_ordinary).
inline constexpr std::uint32_t kMaximumOrdinaryDecodeSteps = 32;

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per step; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

// sampled_tokens holds the last step's tokens. A round of several steps records step
// j < steps - 1 in step_tokens[j * kMaximumConcurrency + row] as the next step takes it as input.
struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
    std::array<TokenId, kMaximumConcurrency*(kMaximumOrdinaryDecodeSteps - 1)> step_tokens{};
};

// The ingress prefix each later step of a multi-step round takes from the device: its tokens and
// cache and RoPE positions, contiguous at the start of the ingress.
inline constexpr std::size_t kOrdinaryDecodeFeedBytes =
    offsetof(OrdinaryDecodeIngress, text_kv_table_rows);
static_assert(offsetof(OrdinaryDecodeIngress, tokens) == 0 &&
              offsetof(OrdinaryDecodeIngress, cache_positions) ==
                  sizeof(OrdinaryDecodeIngress::tokens) &&
              offsetof(OrdinaryDecodeIngress, rope_positions) +
                      sizeof(OrdinaryDecodeIngress::rope_positions) ==
                  kOrdinaryDecodeFeedBytes);

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; RoundState binds only the configured [K,C] and [K+1,C] prefixes.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> base_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> remaining_budgets{};
    std::array<std::int32_t, kMaximumConcurrency> current_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> next_extents{};
};

// Prefill binds the current chunk independently of the compact decode batch.
struct DFlashPrefillIngress {
    std::int32_t append_count           = 0;
    std::int32_t state_destination_slot = 0;
    std::int32_t full_kv_table_row      = 0;
};

// Stable pinned/device transfer formats for one exact-B DFlash transaction. The proposal is
// produced and verified in the same round, so no draft state crosses the round boundary.
struct DFlashDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> execution_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> context_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_valid_columns{};
    // DFlash uses logical positions for its own attention. Target verification carries a separate
    // continuation RoPE position so multimodal rows retain their per-sequence rope_delta.
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth>
        target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> dflash_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> active_lanes{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
    // Stamped on the round's drafts when they reach the host (DFlashDecodeEgress::drafts).
    std::uint32_t round_epoch = 0;
};

struct DFlashDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kDFlashDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Mid-round, before the target verifies: the drafts (row r's k drafts at r * k), then the
    // ingress round_epoch, so a constrained row's verify masks can follow them along its grammar.
    // Not part of the copy at the round's end (kDFlashDecodeRoundEgressBytes).
    std::array<TokenId, kMaximumConcurrency * kDFlashDecodeMaximumDrafts> drafts{};
    std::uint32_t drafts_epoch = 0;
};

inline constexpr std::size_t kDFlashDecodeRoundEgressBytes = offsetof(DFlashDecodeEgress, drafts);

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
    TensorRegion draft_tokens;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
};

struct DFlashPrefillStateLayout {
    LayoutRegion ingress;
    TensorRegion local_append_count;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    TensorRegion proposal_logits;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct DFlashDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion verify_positions;
    std::optional<TensorRegion> candidate_ids;
    std::optional<TensorRegion> proposal_q;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion draft_tokens;
    TensorRegion verify_ids;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<DFlashPrefillStateLayout> dflash_prefill;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    std::optional<DFlashDecodeStateLayout> dflash_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor step_tokens; // I32 [kMaximumConcurrency * (kMaximumOrdinaryDecodeSteps - 1)]
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after scalar logits.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;
    Tensor draft_tokens;
    Tensor target_input_ids;
    Tensor target_positions;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct DFlashPrefillState {
    DeviceSpan ingress;
    Tensor append_count;
    Tensor state_destination_slot;
    Tensor full_kv_table_row;
    // Oversized chunks retain only their tail locally; full KV still uses append_count.
    Tensor local_append_count;

    DFlashPrefillState() = default;
    DFlashPrefillState(DeviceSpan backing, const DFlashPrefillStateLayout& layout);
};

struct MtpDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor current_drafts;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor proposal_logits;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t draft_window);

    // The same frame for a narrower draft window (at most this one's): every draft- or
    // column-shaped tensor re-viewed as a contiguous prefix of its storage, so a round launched at
    // that window reads and writes a dense [window + 1, batch] layout. Host ingress rows and egress
    // rows are strided by the narrower window as well.
    [[nodiscard]] MtpDecodeState narrowed(std::uint32_t draft_window) const;

    // The same frame for its first `rows` batch rows: every batch-shaped tensor sliced along its
    // batch axis (what one decode body of that batch size reads and writes).
    [[nodiscard]] MtpDecodeState rows(std::int32_t rows) const;
};

struct DFlashDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor execution_frontiers;
    Tensor context_frontiers;
    Tensor proposal_extents;
    Tensor target_valid_columns;
    Tensor proposal_valid_columns;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor dflash_kv_table_rows;
    Tensor active_lanes;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor proposal_ids;
    Tensor proposal_positions;
    Tensor verify_positions;
    Tensor candidate_ids;
    Tensor proposal_q;
    Tensor append_positions;
    Tensor append_counts;
    Tensor draft_tokens;
    Tensor verify_ids;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    // The drafter's per-lane pending target features ([features, window + 1, lanes], a view of
    // DFlashPersistentState::pending_features at this frame's width); unset: the whole buffer.
    Tensor pending_features;

    DFlashDecodeState() = default;
    DFlashDecodeState(DeviceSpan backing, const DFlashDecodeStateLayout& layout,
                      std::uint32_t batch_capacity, std::uint32_t draft_window);

    // As MtpDecodeState::narrowed().
    [[nodiscard]] DFlashDecodeState narrowed(std::uint32_t draft_window) const;
};

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    std::optional<MtpPrefillState> mtp;
    std::optional<DFlashPrefillState> dflash_prefill;
    std::optional<MtpDecodeState> mtp_decode;
    std::optional<DFlashDecodeState> dflash_decode;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninferno::models::qwen3_5
