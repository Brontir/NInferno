#include "core/layout.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/speculative/mtp_alignment.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/program/vision_control.h"

#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/planning/neutral_frontier.h"
#include "models/qwen3_5/program/planning/rebuild_work.h"
#include "models/qwen3_5/program/planning/prefill_unit.h"
#include "models/qwen3_5/execution/prefill_calls.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ops/softmax_attention/sliding_window/launch.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninferno::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::DecoderStateSpec decoder_spec(ninferno::KvCacheStorage storage, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 256,
        .kv_storage                = storage,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
    };
}

void test_decoder_layout() {
    ninferno::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 = q36::plan_decoder_state(
        bf16_builder, decoder_spec(ninferno::KvCacheStorage::BFloat16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pages.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pages.spec.page_group_count == 5 &&
               bf16.text_kv.execution_tables.spec.logical_page_capacity == 3 &&
               bf16.text_kv.execution_tables.spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(bf16.text_kv.pages.planes[0].geometry.dtype == ninferno::DType::BF16 &&
               bf16.text_kv.pages.planes[1].geometry.dtype == ninferno::DType::FP16 &&
               bf16.text_kv.pages.planes[2].geometry.dtype == ninferno::DType::BF16 &&
               bf16.text_kv.pages.planes[3].geometry.dtype == ninferno::DType::FP16,
           "BF16 KV has BF16 K and FP16 V without scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninferno::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 = q36::plan_decoder_state(
        int8_builder, decoder_spec(ninferno::KvCacheStorage::Int8Group64, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pages.planes.size() == 8 &&
               int8.text_kv.pages.planes[2].geometry.dtype == ninferno::DType::FP16 &&
               int8.text_kv.pages.planes[3].geometry.dtype == ninferno::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pages.planes.size() == 4 &&
               int8.mtp_kv->pages.spec.page_group_count == 4 &&
               int8.mtp_kv->execution_tables.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pages.planes[2].geometry.dtype == ninferno::DType::FP16 &&
               int8.mtp_kv->pages.planes[3].geometry.dtype == ninferno::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");

    q36::DecoderStateSpec fp8_spec = decoder_spec(ninferno::KvCacheStorage::Fp8E4M3Row256, true);
    ninferno::LayoutBuilder fp8_builder;
    const q36::DecoderStateLayout fp8 = q36::plan_decoder_state(fp8_builder, fp8_spec);
    (void)fp8_builder.finish(256);
    expect(fp8.text_kv.pages.planes.size() == 8 &&
               fp8.text_kv.pages.planes[0].geometry.dtype == ninferno::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[2].geometry.dtype == ninferno::DType::FP16 &&
               fp8.text_kv.pages.planes[2].geometry.leading_extent == 1,
           "FP8 Text KV has row-scaled code and scale planes per layer");
    expect(fp8.mtp_kv && fp8.mtp_kv->pages.planes.size() == 4 &&
               fp8.mtp_kv->pages.planes[0].geometry.dtype == ninferno::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.leading_extent == 1,
           "FP8 MTP KV has row-scaled code and scale planes");
    expect(fp8.kv_payload_bytes() == fp8.text_kv.payload_bytes() + fp8.mtp_kv->payload_bytes(),
           "FP8 Text/MTP KV payload accounting");

    ninferno::LayoutBuilder nvfp4_builder;
    const q36::DecoderStateLayout nvfp4 = q36::plan_decoder_state(
        nvfp4_builder, decoder_spec(ninferno::KvCacheStorage::Nvfp4Group16, true));
    (void)nvfp4_builder.finish(256);
    expect(nvfp4.text_kv.pages.planes.size() == 8 &&
               nvfp4.text_kv.pages.planes[0].geometry.dtype == ninferno::DType::U8 &&
               nvfp4.text_kv.pages.planes[0].geometry.leading_extent == 128 &&
               nvfp4.text_kv.pages.planes[2].geometry.dtype == ninferno::DType::U8 &&
               nvfp4.text_kv.pages.planes[2].geometry.leading_extent == 16,
           "NVFP4 Text KV has packed E2M1 code and E4M3 scale planes");
    constexpr std::size_t nvfp4_vector_bytes = 128 + 16;
    constexpr std::size_t text_vectors       = 2ULL * 5ULL * 64ULL * 2ULL;
    constexpr std::size_t mtp_vectors        = 1ULL * 4ULL * 64ULL * 2ULL;
    expect(nvfp4.text_kv.payload_bytes() == 2ULL * nvfp4_vector_bytes * text_vectors &&
               nvfp4.mtp_kv &&
               nvfp4.mtp_kv->payload_bytes() == 2ULL * nvfp4_vector_bytes * mtp_vectors,
           "NVFP4 Text/MTP physical payload bytes");

    ninferno::LayoutBuilder k8v4_builder;
    const q36::DecoderStateLayout k8v4 = q36::plan_decoder_state(
        k8v4_builder, decoder_spec(ninferno::KvCacheStorage::Fp8KeyNvfp4Value, true));
    (void)k8v4_builder.finish(256);
    expect(k8v4.text_kv.pages.planes.size() == 8 &&
               k8v4.text_kv.pages.planes[0].geometry.dtype == ninferno::DType::FP8_E4M3FN &&
               k8v4.text_kv.pages.planes[0].geometry.leading_extent == 256 &&
               k8v4.text_kv.pages.planes[1].geometry.dtype == ninferno::DType::U8 &&
               k8v4.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               k8v4.text_kv.pages.planes[2].geometry.dtype == ninferno::DType::FP16 &&
               k8v4.text_kv.pages.planes[2].geometry.leading_extent == 1 &&
               k8v4.text_kv.pages.planes[3].geometry.dtype == ninferno::DType::U8 &&
               k8v4.text_kv.pages.planes[3].geometry.leading_extent == 16,
           "K8V4 Text KV preserves independent key/value code and scale geometry");
    constexpr std::size_t k8_vector_bytes = 256 + 2;
    constexpr std::size_t v4_vector_bytes = 128 + 16;
    expect(k8v4.text_kv.payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * text_vectors &&
               k8v4.mtp_kv &&
               k8v4.mtp_kv->payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * mtp_vectors,
           "K8V4 Text/MTP asymmetric physical payload bytes");
}

void test_round_layout() {
    ninferno::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{.hidden       = 32,
                                     .output_rows  = 128,
                                     .draft_window = 5,
                                     .backend      = ninferno::SpeculativeBackend::Mtp});
    const ninferno::TensorRegion exact_prefill =
        builder.add_tensor(ninferno::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.logits.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.mtp->draft_tokens.region.offset,
           "exact prefill extension retains established round-region order");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    ninferno::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder, q36::RoundStateSpec{.hidden       = 32,
                                                 .output_rows  = 128,
                                                 .draft_window = 15,
                                                 .backend = ninferno::SpeculativeBackend::DFlash});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->local_append_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");

    ninferno::LayoutBuilder scoring_builder;
    auto scoring = q36::begin_round_state_layout(
        scoring_builder, {.hidden = 32, .output_rows = 128, .causal_scoring = true});
    q36::complete_round_state_layout(scoring_builder, scoring);
    expect(scoring.rope_delta.region.bytes != 0 && scoring.text_kv_table_row.region.bytes != 0,
           "scoring keeps its Text prefill controls");
    expect(!scoring.ordinary && !scoring.mtp_decode && !scoring.dflash_decode &&
               scoring.token.region.bytes == 0 && scoring.logits.region.bytes == 0,
           "scoring does not reserve generation frames or sampled output");
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControlPlan plan =
        q36::plan_vision_control(prompt, {.spatial_merge_size = 2, .position_grid_side = 48});
    const q36::VisionControl control = q36::build_vision_control(prompt, plan, 0);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");

    const q36::VisionControl suffix = q36::build_vision_control(prompt, plan, 1);
    expect(suffix.prepared_item_begin == 1 && suffix.items.size() == 1 &&
               suffix.items[0].patch_begin == control.items[1].patch_begin &&
               suffix.items[0].scatter_indices == control.items[1].scatter_indices &&
               suffix.items[0].position_ids == control.items[1].position_ids,
           "Vision suffix control contents");
}

q36::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                          static_cast<std::uint8_t>(q36::PromptModality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    q36::VisionItem item{.modality    = q36::PromptModality::Image,
                         .grid        = {.temporal = 1, .height = 2, .width = 4},
                         .patch_begin = 0,
                         .patch_count = 8,
                         .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(q36::PreparedPromptData& prompt, ninferno::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    q36::PreparedPromptData original    = identity_prompt();
    std::vector<ninferno::TokenId> ledger = original.token_ids;
    q36::detail::ResidentPrefixIdentity resident;
    q36::detail::PrefixShortlistDigests digests;
    resident.reserve(16);
    resident.assign(original);
    digests.reserve(16);
    digests.assign(original);

    expect(q36::detail::prefix_matches(original, ledger, resident, original.token_ids.size()),
           "identical multimodal prefix identity");

    q36::PreparedPromptData changed_media = identity_prompt(2);
    expect(!q36::detail::prefix_matches(changed_media, ledger, resident,
                                        changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(q36::detail::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!q36::detail::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!q36::detail::prefix_matches(changed_position, ledger, resident,
                                        changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    q36::PreparedPromptData changed_decomposition              = identity_prompt();
    changed_decomposition.identity.rewrite_execution_frontiers = {1};
    expect(!q36::detail::prefix_matches(changed_decomposition, ledger, resident,
                                        changed_decomposition.token_ids.size()),
           "different GDN execution decomposition must not reuse resident state");
    changed_decomposition.identity.rewrite_execution_frontiers = {4};
    expect(q36::detail::prefix_matches(changed_decomposition, ledger, resident, 3),
           "execution decomposition wholly after the frontier changed prefix identity");

    q36::PreparedPromptData resident_future              = identity_prompt();
    resident_future.identity.rewrite_execution_frontiers = {1, 4};
    q36::detail::ResidentPrefixIdentity resident_with_future;
    resident_with_future.assign(resident_future);
    q36::PreparedPromptData incoming_future              = identity_prompt();
    incoming_future.identity.rewrite_execution_frontiers = {1, 3};
    expect(q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 1),
           "resident execution decomposition after the frontier changed prefix identity");
    expect(!q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 3),
           "different execution decomposition inside the frontier reused resident state");

    q36::detail::PrefixShortlistDigests future_digest;
    future_digest.assign(resident_future);
    q36::detail::PrefixShortlistDigests incoming_digest;
    incoming_digest.assign(incoming_future);
    expect(future_digest.at(1) == incoming_digest.at(1),
           "future execution boundaries changed an earlier content shortlist");
    expect(future_digest.at(3) != incoming_digest.at(3),
           "different in-prefix execution boundaries shared a shortlist digest");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    const std::array<ninferno::TokenId, 1> generated{12};
    digests.append_generated(generated, original.rope_delta);
    append_text_token(original, 12, 4);
    q36::detail::PrefixShortlistDigests rebuilt;
    rebuilt.assign(original);
    expect(digests.at(ledger.size()) == rebuilt.at(ledger.size()),
           "incremental generated-token shortlist diverged from a full rebuild");
    expect(q36::detail::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    q36::PreparedPromptData accepted_rebuild = identity_prompt();
    const std::array<ninferno::TokenId, 3> proposed{20, 21, 22};
    const std::span<const ninferno::TokenId> accepted(proposed.data(), 2);
    std::vector<ninferno::TokenId> accepted_ledger = accepted_rebuild.token_ids;
    accepted_ledger.insert(accepted_ledger.end(), accepted.begin(), accepted.end());
    q36::detail::ResidentPrefixIdentity accepted_resident;
    q36::detail::PrefixShortlistDigests accepted_digests;
    accepted_resident.assign(accepted_rebuild);
    accepted_digests.assign(accepted_rebuild);
    accepted_resident.append_generated(accepted.size(), accepted_rebuild.rope_delta, 1);
    accepted_digests.append_generated(accepted, accepted_rebuild.rope_delta, 1);
    append_text_token(accepted_rebuild, accepted[0], 4);
    append_text_token(accepted_rebuild, accepted[1], 5);
    accepted_rebuild.identity.rewrite_execution_frontiers = {5};
    q36::detail::PrefixShortlistDigests rebuilt_accepted_digests;
    rebuilt_accepted_digests.assign(accepted_rebuild);
    expect(q36::detail::prefix_matches(accepted_rebuild, accepted_ledger, accepted_resident,
                                       accepted_ledger.size()) &&
               accepted_digests.at(accepted_ledger.size()) ==
                   rebuilt_accepted_digests.at(accepted_ledger.size()),
           "accepted generated prefix and rebuilt history formed different cache identities");

    const q36::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    digests.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    q36::detail::PrefixShortlistDigests prompt_digest;
    prompt_digest.assign(prompt_only);
    expect(digests.at(ledger.size()) == prompt_digest.at(ledger.size()),
           "truncated shortlist did not restore the original frontier digest");
    expect(q36::detail::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}

void test_rebuild_work_prompt_frontier_boundary() {
    constexpr std::uint32_t prompt_tokens = 100;
    constexpr std::uint32_t prefill_chunk = 2048;
    std::uint32_t tail_begin              = 0;
    q36::runtime_support::include_rebuild_boundary(tail_begin, prompt_tokens, prompt_tokens);
    expect(tail_begin == prompt_tokens,
           "prompt-frontier rebuild boundary was not retained for continuation growth");

    ninferno::runtime::PrefillWork work =
        ninferno::runtime::make_prefill_work(0, prompt_tokens, 0, 0, prefill_chunk);
    q36::runtime_support::advance_segmented_rebuild_work(work, tail_begin, prompt_tokens,
                                                         prompt_tokens + 1, prefill_chunk);
    const ninferno::runtime::PrefillWork exact =
        ninferno::runtime::make_prefill_work(0, prompt_tokens + 1, 0, 0, prefill_chunk);
    expect(work.chunks == 2 && work.tokens == exact.tokens &&
               work.attention_pairs == exact.attention_pairs,
           "continuation growth did not preserve the prompt-frontier rebuild split");
}

// A staged prefill unit sizes its calls toward the next capture frontier inside the prompt: each
// segment up to a frontier then takes ceil(segment / chunk) units, as the admission plan projects,
// and the last one ends on the frontier.
void test_staged_prefill_unit_end() {
    namespace ex = q36::execution;
    namespace rs = q36::runtime_support;
    expect(rs::staged_prefill_unit_end(10, 100, std::nullopt) == 100 &&
               rs::staged_prefill_unit_end(10, 100, 100) == 100 &&
               rs::staged_prefill_unit_end(10, 100, 10) == 100 &&
               rs::staged_prefill_unit_end(10, 100, 60) == 60,
           "staged prefill unit end is not the next capture frontier inside the prompt");
    int mismatches = 0;
    for (const std::uint32_t chunk : {128U, 256U, 512U, 2048U}) {
        for (std::uint32_t cursor = 0; cursor < 300; cursor += 7) {
            for (std::uint32_t frontier = cursor + 1; frontier < cursor + 600; ++frontier) {
                const std::uint32_t prompt = frontier + 1 + frontier % 37;
                const std::uint32_t end    = rs::staged_prefill_unit_end(cursor, prompt, frontier);
                std::uint32_t at           = cursor;
                std::uint64_t units        = 0;
                while (at < end) {
                    at += ex::prefill_call_tokens(chunk, at, end);
                    ++units;
                }
                if (at != frontier || units != 1ULL + (frontier - cursor - 1ULL) / chunk) {
                    if (mismatches++ < 4) {
                        std::cerr << "chunk " << chunk << " cursor " << cursor << " frontier "
                                  << frontier << ": " << units << " units end at " << at << '\n';
                    }
                }
            }
        }
    }
    expect(mismatches == 0, "staged prefill units do not match the projected service work");
}

// Every call boundary of a segment lies a multiple of the scan's 16-token chunk after the
// segment's start, for every --prefill-chunk (a multiple of 128) and for a start on the 64-token
// grid or off it (a turn's rewrite frontier), so the Gated DeltaNet scan folds the same chunks
// whatever the chunk size. A segment that starts on the grid stops early where it always did: on
// the absolute grid.
void test_prefill_calls_on_segment_grid() {
    namespace ex       = q36::execution;
    int off_chunk_grid = 0;
    int moved          = 0;
    for (const std::uint32_t start : {0U, 64U, 1000U, 4099U}) {
        for (const std::uint32_t chunk : {128U, 256U, 512U, 1024U, 2048U}) {
            for (std::uint32_t end = start + 1; end < start + 3000; ++end) {
                std::uint32_t at = start;
                while (at < end) {
                    const std::uint32_t next = ex::prefill_call_end(chunk, at, end);
                    if (next != end && (next - start) % 16 != 0) { ++off_chunk_grid; }
                    if (start % ex::kPrefillGridTokens == 0 && next != end && next != at + chunk &&
                        next != (end - ex::kPrefillMinTokens) / ex::kPrefillGridTokens *
                                    ex::kPrefillGridTokens) {
                        ++moved;
                    }
                    at = next;
                }
            }
        }
    }
    expect(off_chunk_grid == 0, "a prefill call boundary divides a scan chunk of its segment");
    expect(moved == 0, "an early stop of a segment on the grid left the absolute grid");
}

} // namespace

// Shared and long-anchor capture and restore frontiers: only where the prefill decomposition of a
// cold run already splits, or where a grid-aligned split leaves every call on the grid and at
// least 17 tokens long.
void test_neutral_prefill_frontier() {
    namespace rs = q36::runtime_support;
    const std::vector<std::uint32_t> none;
    // One static segment [0, 300): grid points up to 300 - 17.
    expect(rs::neutral_prefill_frontier(200, 300, none) == std::optional<std::uint32_t>(192),
           "an off-grid frontier moves down to the grid");
    expect(rs::neutral_prefill_frontier(283, 300, none) == std::optional<std::uint32_t>(256),
           "a grid frontier 17 tokens before the end is neutral");
    expect(rs::neutral_prefill_frontier(290, 300, none) == std::optional<std::uint32_t>(256),
           "a frontier inside the last 16 tokens moves below them");
    expect(rs::neutral_prefill_frontier(270, 280, none) == std::optional<std::uint32_t>(256),
           "the grid point 24 tokens before the end is neutral");
    expect(!rs::neutral_prefill_frontier(6, 15, none),
           "a short prompt has no neutral interior frontier");
    expect(!rs::neutral_prefill_frontier(63, 300, none), "no grid point lies above zero");
    expect(rs::neutral_prefill_frontier(300, 300, none) == std::optional<std::uint32_t>(300),
           "the prompt end is neutral");
    expect(!rs::neutral_prefill_frontier(0, 300, none) && !rs::neutral_prefill_frontier(301, 300, none),
           "frontiers outside the prompt are rejected");
    // Static frontiers 13 and 14 (a turn closure and its assistant header), prompt 20.
    const std::vector<std::uint32_t> turn{13, 14, 18};
    expect(rs::neutral_prefill_frontier(13, 20, turn) == std::optional<std::uint32_t>(13),
           "a static frontier is neutral");
    expect(rs::neutral_prefill_frontier(6, 20, turn) == std::nullopt,
           "a frontier before the first static one has no grid point");
    expect(rs::neutral_prefill_frontier(16, 20, turn) == std::optional<std::uint32_t>(14),
           "a frontier inside an off-grid segment moves to its static start");
    // A long prompt: a segment that starts off the grid admits only its start; one on the grid
    // admits grid points.
    const std::vector<std::uint32_t> history{1000, 1003, 2048, 2051};
    expect(rs::neutral_prefill_frontier(1900, 4000, history) == std::optional<std::uint32_t>(1003),
           "an off-grid segment start is the only neutral point of its segment");
    expect(rs::neutral_prefill_frontier(900, 4000, history) == std::optional<std::uint32_t>(896),
           "the first segment admits grid points");
    expect(rs::neutral_prefill_frontier(990, 4000, history) == std::optional<std::uint32_t>(960),
           "a grid point stays 17 tokens before the next static frontier");
    expect(rs::is_neutral_prefill_frontier(896, 4000, history) &&
               !rs::is_neutral_prefill_frontier(900, 4000, history) &&
               !rs::is_neutral_prefill_frontier(3968, 3980, history) &&
               rs::is_neutral_prefill_frontier(2051, 4000, history),
           "restore frontiers are neutral exactly where a capture could lie");
}

// A plan's cache opportunities after the neutral rule: frontiers move down, one without a neutral
// point is dropped, and two of a kind that land together merge (the resource manager looks a
// shared candidate up by the frontier its capture uses).
void test_neutral_cache_opportunities() {
    namespace rs = q36::runtime_support;
    struct Opportunity {
        int kind                  = 0;
        std::uint32_t evidence    = 0;
        std::uint32_t frontier    = 0;
        std::uint32_t input_order = 0;
    };
    const std::vector<std::uint32_t> statics{284, 287};
    const std::vector<Opportunity> input{
        {.kind = 0, .evidence = 1, .frontier = 92, .input_order = 3},
        {.kind = 0, .evidence = 2, .frontier = 70, .input_order = 1},
        {.kind = 1, .evidence = 0, .frontier = 92, .input_order = 4},
        {.kind = 0, .evidence = 4, .frontier = 284, .input_order = 5},
        {.kind = 0, .evidence = 8, .frontier = 12, .input_order = 0},
    };
    const std::vector<Opportunity> out = rs::neutral_cache_opportunities(
        std::span<const Opportunity>(input), 289, statics);
    expect(out.size() == 3, "an opportunity without a neutral point is dropped and duplicates merge");
    expect(out.size() == 3 && out[0].kind == 0 && out[0].frontier == 64 && out[0].evidence == 3 &&
               out[0].input_order == 1,
           "two shared opportunities on one neutral frontier merge their evidence and order");
    expect(out.size() == 3 && out[1].kind == 1 && out[1].frontier == 64,
           "opportunities of different kinds stay apart");
    expect(out.size() == 3 && out[2].frontier == 284 && out[2].evidence == 4,
           "a static frontier stays where it is");
    for (const Opportunity& opportunity : out) {
        expect(rs::is_neutral_prefill_frontier(opportunity.frontier, 289, statics),
               "every normalized opportunity lies on a neutral frontier");
    }
}

// DFlash graph profiles of one topology class share one executable, switched by
// cudaGraphExecUpdate: the drafter's sliding attention must take the same route (direct, or split
// with its reduce kernel) at every profile of a class. The Qwen3.6 drafter queries k + 1 columns,
// the dense drafter its whole 16-column block; its local envelope is [0, the profile's max].
void test_dflash_graph_classes_follow_sliding_routes() {
    namespace ops = ninferno::ops::detail;
    for (const std::uint32_t capacity : {2048U, 8192U}) {
        for (std::uint32_t batch = 1; batch <= 8; ++batch) {
            for (const std::uint32_t k : {1U, 3U, 7U, 8U, 15U}) {
                const auto profiles = q36::detail::dflash_graph_profiles(
                    ninferno::SpeculativeBackend::DFlash, capacity, k, batch);
                for (const std::uint32_t window : {2048U, 4096U}) {
                    for (const std::int32_t tokens : {static_cast<std::int32_t>(k + 1), 16}) {
                        const auto route = [&](const auto& profile) {
                            return ops::sliding_window_attention_resolve_plan(
                                       window, tokens, static_cast<std::int32_t>(batch),
                                       {0, profile.max})
                                .route;
                        };
                        for (std::size_t i = 0; i < profiles.size(); ++i) {
                            for (std::size_t j = i + 1; j < profiles.size(); ++j) {
                                if (profiles[i].topology_class != profiles[j].topology_class) {
                                    continue;
                                }
                                expect(route(profiles[i]) == route(profiles[j]),
                                       "DFlash capacity " + std::to_string(capacity) + " batch " +
                                           std::to_string(batch) + " K=" + std::to_string(k) +
                                           " T=" + std::to_string(tokens) + " window " +
                                           std::to_string(window) + ": profiles ending " +
                                           std::to_string(profiles[i].max) + " and " +
                                           std::to_string(profiles[j].max) +
                                           " share a topology class across sliding routes");
                            }
                        }
                    }
                }
            }
        }
    }
}

int main() {
    test_dflash_graph_classes_follow_sliding_routes();
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_prefix_identity();
    test_rebuild_work_prompt_frontier_boundary();
    test_staged_prefill_unit_end();
    test_prefill_calls_on_segment_grid();
    test_neutral_prefill_frontier();
    test_neutral_cache_opportunities();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}
