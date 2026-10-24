#include "models/qwen3_5/program/planning/sequence_bytes.h"

#include "core/checked.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/units.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ninferno/ops/sampling.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::models::qwen3_5::detail {
namespace {

constexpr std::size_t kArenaAlign = 256ULL;

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

// A model dimension in the Tensor integer domain (execution::dimension's check).
std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

ProgramBufferShape program_buffers_of(const StateShape& shape) {
    return {.hidden        = shape.hidden,
            .output_rows   = shape.output_rows,
            .public_tokens = shape.public_tokens};
}

const DrafterStateShape& drafter_of(const StateShape& shape) {
    if (!shape.drafter) {
        throw std::logic_error("a masked-draft plan needs its drafter's state shape");
    }
    return *shape.drafter;
}

// The Main KV planes of one layer of a drafter's paged pool, in PagedKVCache::layer_view order (K,
// V, then the scales), `layers` times.
KVPageGeometry drafter_page_geometry(const PagedKVStorageLayout& storage, std::int32_t kv_heads,
                                     std::uint32_t layers) {
    KVPageGeometry geometry{.page_tokens        = kPagedKVPageSize,
                            .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                            .planes             = {}};
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {storage.key.data_dtype, storage.key.data_leading_extent, kv_heads, 256});
        geometry.planes.push_back(
            {storage.value.data_dtype, storage.value.data_leading_extent, kv_heads, 256});
        if (storage.key.has_scale()) {
            geometry.planes.push_back(
                {storage.key.scale_dtype, storage.key.scale_leading_extent, kv_heads, 256});
        }
        if (storage.value.has_scale()) {
            geometry.planes.push_back(
                {storage.value.scale_dtype, storage.value.scale_leading_extent, kv_heads, 256});
        }
    }
    return geometry;
}

// A DFlash drafter's paged pool beside the target's (its full layers), its prefill and pending
// target features: the Program's DFlash layout.
DFlashPersistentLayout plan_drafter_state(LayoutBuilder& builder, const DrafterStateShape& draft,
                                          const SequencePlanImpl& plan, KvCacheStorage storage) {
    const auto chunk = static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    DFlashPersistentLayout out;
    if (draft.full_layers != 0) {
        const PagedKVStorageLayout layer_storage = paged_kv_storage_layout(storage, draft.head_dim);
        out.full                                 = qwen3_5::PagedKVCacheLayout{
                                            .pages = plan_device_kv_page_pool(
                builder,
                DeviceKVPagePoolSpec{.page_group_count = plan.main_page_groups,
                                                                     .geometry         = drafter_page_geometry(
                                         layer_storage, draft.kv_heads, draft.full_layers)}),
                                            .execution_tables = plan_kv_execution_tables(
                builder,
                KVExecutionTableSpec{
                                                    .logical_page_capacity = paged_kv_capacity_pages(plan.capacity),
                                                    .table_rows            = static_cast<std::int32_t>(plan.max_concurrency)}),
                                            .layers        = draft.full_layers,
                                            .max_context   = plan.capacity,
                                            .kv_heads      = draft.kv_heads,
                                            .layer_storage = layer_storage,
        };
    }
    out.prefill_features = add_tensor(builder, DType::BF16, {draft.prefill_feature_rows, chunk},
                                      "DFlash prefill target features");
    out.prefill_positions =
        add_tensor(builder, DType::I32, {chunk}, "DFlash prefill target positions");
    out.pending_features =
        add_tensor(builder, DType::BF16,
                   {draft.feature_rows, static_cast<std::int32_t>(plan.draft_window + 1U),
                    static_cast<std::int32_t>(plan.max_concurrency)},
                   "DFlash pending target features");
    return out;
}

// The paged pool of a ring decoder, Gemma 4 and NemotronH: head-major planes the gemma4 attention
// reads, BF16 or FP8 (E4M3 codes and an FP32 scale per row).
struct HeadMajorPool {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    PagedKVStorageLayout layer_storage;
};

HeadMajorPool head_major_pool(KvCacheStorage requested, std::int32_t head_dim) {
    const bool fp8 = requested == KvCacheStorage::Fp8E4M3Row256;
    const PagedKVVectorLayout vector =
        fp8 ? PagedKVVectorLayout{DType::FP8_E4M3FN, head_dim, DType::FP32, 1}
            : PagedKVVectorLayout{DType::BF16, head_dim, DType::U8, 0};
    const KvCacheStorage storage = fp8 ? KvCacheStorage::Fp8E4M3Row256 : KvCacheStorage::BFloat16;
    return {.storage       = storage,
            .layer_storage = PagedKVStorageLayout{storage, head_dim, vector, vector}};
}

std::int32_t state_image_slots(const SequencePlanImpl& plan, const char* not_normalized,
                               const char* overflow) {
    if (!plan.context_cache.device_state_slots) { throw std::logic_error(not_normalized); }
    const std::uint64_t slots =
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots;
    if (slots > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(overflow);
    }
    return static_cast<std::int32_t>(slots);
}

// A DFlash drafter's sliding-layer ring keeps Int8Group64 K/V at D128 (about half the bytes of
// every StateImage slot's ring): on an RTX 5090, Qwen3.6-35B-A3B with DFlash decodes 0.7-2.4 %
// faster at 1-8 concurrent requests with 0.6 GiB less runtime. DFlash2 keeps BF16: an int8 ring
// takes it off the fused materialization onto the per-layer path (Qwen3.8-27B: +2.8 % at one
// request, -2.7 % at four, -0.6 % at eight with 31K contexts).
bool dflash_local_int8(std::int64_t head_dim, bool dflash2) { return head_dim == 128 && !dflash2; }

// ---- Qwen 3.5 / 3.6: the paged KV of the full layers and the MTP layer, GatedDeltaNet states and
// their replay records, a DFlash drafter's ring and pool.
PersistentLayout qwen3_5_layout(const StateShape& shape, const SequencePlanImpl& plan) {
    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen3.5 StateImage slot count exceeds int32");
    const std::uint32_t physical_pages     = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages    = plan.features.mtp()
                                                 ? static_cast<std::uint64_t>(plan.max_concurrency) *
                                                    paged_kv_pages(plan.draft_window - 1U)
                                                 : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    // The GatedDeltaNet geometry (zero without one: the family fills it from its config's gdn).
    const bool gdn = shape.value_heads != 0 || shape.key_heads != 0 || shape.conv_kernel != 0;
    const std::int32_t conv_channels =
        gdn ? dimension(checked_add(
                  checked_mul(2,
                              checked_mul(static_cast<std::uint64_t>(shape.key_heads),
                                          static_cast<std::uint64_t>(shape.key_dim),
                                          "GDN key width"),
                              "GDN key width"),
                  checked_mul(static_cast<std::uint64_t>(shape.value_heads),
                              static_cast<std::uint64_t>(shape.value_dim), "GDN value width"),
                  "GDN convolution channels"))
            : 0;
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = shape.full_layers,
                     .mtp_layers                = 1,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = shape.kv_heads,
                     .attention_head_dim        = shape.head_dim,
                     .kv_storage                = plan.kv_storage,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                 });
    qwen3_5::StateImageSpec state_image_spec{
        .linear =
            {
                .layers        = shape.recurrent_layers,
                .conv_channels = conv_channels,
                .conv_width =
                    gdn ? dimension(static_cast<std::uint32_t>(shape.conv_kernel - 1)) : 0,
                .value_heads     = shape.value_heads,
                .value_head_dim  = shape.value_dim,
                .key_head_dim    = shape.key_dim,
                .slot_count      = state_image_slots,
                .conv_dtype      = DType::BF16,
                .recurrent_dtype = plan.recurrent_state_dtype,
            },
        .hidden = shape.hidden,
    };
    if (plan.features.masked_draft()) {
        const DrafterStateShape& draft = drafter_of(shape);
        state_image_spec.dflash_local  = qwen3_5::DFlashLocalStateSpec{
             .layers   = draft.local_layers,
             .capacity = draft.window,
             .kv_heads = draft.kv_heads,
             .head_dim = draft.head_dim,
             .int8     = dflash_local_int8(draft.head_dim, draft.dflash2),
        };
    }
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder, GdnReplayRecordSpec{
                         .layers          = dimension(shape.recurrent_layers),
                         .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                         .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                         .conv_channels   = conv_channels,
                         .qk_heads        = shape.key_heads,
                         .value_heads     = shape.value_heads,
                         .key_dim         = shape.key_dim,
                         .value_dim       = shape.value_dim,
                     });
    }
    if (plan.features.masked_draft()) {
        const DrafterStateShape& draft = drafter_of(shape);
        // A DFlash drafter's full layers keep per-token G64 int8 K/V at D128 (half the bytes of
        // BF16 K + FP16 V plus 1/32 for scales), like its sliding ring: with both, Qwen3.6-35B-A3B
        // on an RTX 4090 with four 40K requests holds ~129K tokens instead of 90K and its ~30K
        // rounds stop collapsing (516 / 112 -> 655 tok/s), while an RTX 5090 is flat at 1-8
        // requests. DFlash2 keeps BF16 (the int8 full-layer cache was not evaluated for it).
        const bool drafter_int8 = !draft.dflash2 && draft.head_dim == 128;
        out.dflash              = plan_drafter_state(builder, draft, plan,
                                        drafter_int8 ? KvCacheStorage::Int8Group64
                                                                  : KvCacheStorage::BFloat16);
    }
    plan_program_buffers(builder, plan, program_buffers_of(shape), out);
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

// ---- The dense decoder family: a head-major pool, one ring group of its sliding layers, a DFlash
// drafter's ring (its sliding layers) and BF16 pool (its full layers).
PersistentLayout dense_layout(const StateShape& shape, const SequencePlanImpl& plan) {
    const std::int32_t slots =
        state_image_slots(plan, "dense decoder context cache options are not normalized",
                          "dense decoder state image slot count exceeds int32");
    // --kv-dtype fp8: the full layers' pool holds E4M3 codes and an FP32 scale per row (the
    // Gemma4KvLayer FP8 storage the attention reads); the rings follow --sliding-kv-dtype.
    const HeadMajorPool pool = head_major_pool(plan.kv_storage, shape.head_dim);

    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = shape.full_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = shape.kv_heads,
                     .attention_head_dim        = shape.head_dim,
                     .kv_storage                = pool.storage,
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = plan.main_page_groups,
                     // The attention reads keys and values from a head-major pool.
                     .layer_storage = pool.layer_storage,
                     .plane_order   = PagedKVPlaneOrder::HeadMajor,
                 });
    qwen3_5::StateImageSpec images{
        .linear = {.layers = 0, .slot_count = slots},
        .hidden = shape.hidden,
    };
    if (shape.sliding_layers != 0) {
        images.rings.push_back({.layers   = shape.sliding_layers,
                                .capacity = shape.window + ring_decoder_widest_call(plan) - 1U,
                                .kv_heads = shape.sliding_kv_heads,
                                .head_dim = shape.sliding_head_dim,
                                .fp8      = plan.ring_storage == KvCacheStorage::Fp8E4M3Row256});
    }
    if (plan.features.masked_draft()) {
        const DrafterStateShape& draft = drafter_of(shape);
        if (draft.local_layers != 0) {
            images.dflash_local = qwen3_5::DFlashLocalStateSpec{.layers   = draft.local_layers,
                                                                .capacity = draft.window,
                                                                .kv_heads = draft.kv_heads,
                                                                .head_dim = draft.head_dim};
        }
    }
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, images);
    if (plan.features.masked_draft()) {
        // The drafter's full layers keep their keys and values in a paged pool of their own, at
        // the target's page geometry with head-major BF16 planes (the Program's DFlash layout).
        out.dflash = plan_drafter_state(builder, drafter_of(shape), plan, KvCacheStorage::BFloat16);
    }
    plan_program_buffers(builder, plan, program_buffers_of(shape), out);
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

// ---- Gemma 4: the full layers' head-major pool and the sliding layers' rings (a layer that reads
// another's keys and values adds none).
PersistentLayout gemma4_layout(const StateShape& shape, const SequencePlanImpl& plan) {
    const std::int32_t slots =
        state_image_slots(plan, "Gemma 4 context cache options are not normalized",
                          "Gemma 4 state image slot count exceeds int32");
    // --kv-dtype fp8: the full layers' pool holds E4M3 codes and an FP32 scale per row
    // (Gemma4KvLayer's FP8 storage, any head_dim); the sliding layers' rings follow
    // --sliding-kv-dtype, the same storage in the images layout.
    const HeadMajorPool pool = head_major_pool(plan.kv_storage, shape.head_dim);

    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = shape.full_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = shape.kv_heads,
                     .attention_head_dim        = shape.head_dim,
                     .kv_storage                = pool.storage,
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = plan.main_page_groups,
                     // The Gemma 4 attention reads keys and values from a head-major pool.
                     .layer_storage = pool.layer_storage,
                     .plane_order   = PagedKVPlaneOrder::HeadMajor,
                 });
    out.state_images = qwen3_5::plan_state_image_device_pool(
        builder, qwen3_5::StateImageSpec{
                     .linear = {.layers = 0, .slot_count = slots},
                     .hidden = shape.hidden,
                     .rings  = {{.layers   = shape.sliding_layers,
                                 .capacity = shape.window + ring_decoder_widest_call(plan) - 1U,
                                 .kv_heads = shape.sliding_kv_heads,
                                 .head_dim = shape.sliding_head_dim,
                                 .fp8      = plan.ring_storage == KvCacheStorage::Fp8E4M3Row256}},
                 });
    plan_program_buffers(builder, plan, program_buffers_of(shape), out);
    out.kv_payload_bytes = out.decoder.kv_payload_bytes();
    return out;
}

// ---- NemotronH: the attention layers' head-major pool and the MTP head's, Mamba2 states and their
// replay records.
PersistentLayout nemotron_h_layout(const StateShape& shape, const SequencePlanImpl& plan) {
    const std::int32_t slots =
        state_image_slots(plan, "nemotron_h context cache options are not normalized",
                          "nemotron_h state image slot count exceeds int32");
    // --kv-dtype fp8: the pools (the MTP head's too) hold E4M3 codes and an FP32 scale per row.
    const HeadMajorPool pool = head_major_pool(plan.kv_storage, shape.head_dim);
    // The MTP pool: every main page group, and per row the pages the draft chain writes past
    // its frontier.
    const std::uint64_t mtp_extra_pages = plan.features.mtp()
                                              ? static_cast<std::uint64_t>(plan.max_concurrency) *
                                                    paged_kv_pages(plan.draft_window - 1U)
                                              : 0ULL;
    const std::uint64_t mtp_pages       = plan.main_page_groups + mtp_extra_pages;
    if (mtp_pages > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("nemotron_h MTP Paged KV physical pages exceed int32");
    }
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = shape.full_layers,
                     .mtp_layers                = 1,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = shape.kv_heads,
                     .attention_head_dim        = shape.head_dim,
                     .kv_storage                = pool.storage,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = plan.main_page_groups,
                     .mtp_physical_page_groups  = static_cast<std::uint32_t>(mtp_pages),
                     .layer_storage             = pool.layer_storage,
                     .plane_order               = PagedKVPlaneOrder::HeadMajor,
                 });
    // Mamba2: heads x head_dim x state per layer; the convolution window over x, B and C (inner +
    // 2 * groups * state channels), three rows.
    const std::int32_t conv_channels =
        shape.value_heads * shape.value_dim + 2 * shape.key_heads * shape.key_dim;
    const qwen3_5::StateImageSpec images{
        .linear = {.layers         = shape.recurrent_layers,
                   .conv_channels  = conv_channels,
                   .conv_width     = 3,
                   .value_heads    = shape.value_heads,
                   .value_head_dim = shape.value_dim,
                   .key_head_dim   = shape.key_dim,
                   .slot_count     = slots,
                   .conv_dtype     = DType::BF16},
        .hidden = shape.hidden,
    };
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, images);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        // The replay records of a Mamba2 stack, in the Program's record planes: conv = the conv
        // inputs (xBC before the convolution) [C, width, rows], value = the conv outputs
        // [C, 1, width, rows], key = dt [H, 1, width, rows] (the gate plane is unused).
        out.replay_records = plan_gdn_replay_records(
            builder,
            GdnReplayRecordSpec{.layers = static_cast<std::int32_t>(shape.recurrent_layers),
                                .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                                .width         = static_cast<std::int32_t>(plan.draft_window + 1U),
                                .conv_channels = conv_channels,
                                .qk_heads      = 1,
                                .value_heads   = 1,
                                .key_dim       = shape.value_heads,
                                .value_dim     = conv_channels});
    }
    plan_program_buffers(builder, plan, program_buffers_of(shape), out);
    out.kv_payload_bytes = out.decoder.kv_payload_bytes();
    return out;
}

} // namespace

void plan_program_buffers(LayoutBuilder& builder, const SequencePlanImpl& plan,
                          const ProgramBufferShape& traits, PersistentLayout& out) {
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    out.round = qwen3_5::begin_round_state_layout(
        builder, qwen3_5::RoundStateSpec{.hidden         = traits.hidden,
                                         .output_rows    = traits.output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .backend        = plan.speculative_backend,
                                         .causal_scoring = plan.causal_scoring});
    out.prefill_hidden = add_tensor(builder, DType::BF16, {traits.hidden, effective_prefill_chunk},
                                    "step prefill hidden");
    if (plan.causal_scoring) {
        out.score_hidden = add_tensor(builder, DType::BF16,
                                      {traits.hidden, static_cast<std::int32_t>(kCausalScoreTile)},
                                      "causal score hidden staging");
    }
    qwen3_5::complete_round_state_layout(builder, out.round);
    if (!plan.causal_scoring) {
        out.token_counts =
            add_tensor(builder, DType::I32,
                       {traits.public_tokens, static_cast<std::int32_t>(plan.max_concurrency)},
                       "sampling token counts");
        const auto config_words = static_cast<std::int32_t>(
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
        out.sampling_config = add_tensor(
            builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
            "sampling config");
    }
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
}

PersistentLayout plan_persistent_layout(const StateShape& shape, const SequencePlanImpl& plan) {
    switch (shape.family) {
    case StateFamily::Qwen35:
        return qwen3_5_layout(shape, plan);
    case StateFamily::DenseDecoder:
        return dense_layout(shape, plan);
    case StateFamily::Gemma4:
        return gemma4_layout(shape, plan);
    case StateFamily::NemotronH:
        return nemotron_h_layout(shape, plan);
    }
    throw std::logic_error("unknown state family");
}

void validate_ring_decoder_storage(const EngineOptions& options, std::string_view family) {
    const auto bf16_or_fp8 = [](KvCacheStorage storage) {
        return storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Fp8E4M3Row256;
    };
    if (!bf16_or_fp8(options.kv_cache)) {
        throw std::invalid_argument(std::string(family) +
                                    " keeps keys and values in BF16 or FP8 (--kv-dtype bf16|fp8)");
    }
    if (!bf16_or_fp8(options.sliding_kv_cache)) {
        throw std::invalid_argument(std::string(family) + " keeps its sliding rings in BF16 or FP8 "
                                                          "(--sliding-kv-dtype bf16|fp8)");
    }
}

void validate_state_storage(const StateShape& shape, const EngineOptions& options) {
    switch (shape.family) {
    case StateFamily::Qwen35:
        if (options.sliding_kv_cache != KvCacheStorage::BFloat16) {
            throw std::invalid_argument(
                "Qwen 3.5 has no sliding-window layers (--sliding-kv-dtype is for their rings)");
        }
        return;
    case StateFamily::DenseDecoder:
        validate_ring_decoder_storage(options, "the dense decoder");
        // FP8 rings on a model with no ring would only be logged: refused, as Qwen 3.5 and
        // Nemotron-H refuse it.
        if (shape.sliding_layers == 0 && options.sliding_kv_cache != KvCacheStorage::BFloat16) {
            throw std::invalid_argument("this dense decoder has no sliding-window layers "
                                        "(--sliding-kv-dtype is for their rings)");
        }
        return;
    case StateFamily::Gemma4:
        validate_ring_decoder_storage(options, "Gemma 4");
        return;
    case StateFamily::NemotronH:
        // Its attention layers read a head-major pool through gemma4_attention, which takes BF16
        // or E4M3 codes with an FP32 scale per row; it has no sliding-window layers.
        if (options.kv_cache != KvCacheStorage::BFloat16 &&
            options.kv_cache != KvCacheStorage::Fp8E4M3Row256) {
            throw std::invalid_argument(
                "Nemotron-H keeps keys and values in BF16 or FP8 (--kv-dtype bf16|fp8)");
        }
        if (options.sliding_kv_cache != KvCacheStorage::BFloat16) {
            throw std::invalid_argument(
                "Nemotron-H has no sliding-window layers (--sliding-kv-dtype is for their rings)");
        }
        return;
    }
}

DType recurrent_state_dtype(RecurrentStateStorage storage) noexcept {
    switch (storage) {
    case RecurrentStateStorage::BF16:
        return DType::BF16;
    case RecurrentStateStorage::FP16:
        return DType::FP16;
    case RecurrentStateStorage::FP32:
        break;
    }
    return DType::FP32;
}

SpeculativeOptions resolve_copy_window(SpeculativeOptions speculative, bool copy_rounds) {
    if (speculative.ngram_draft_tokens != SpeculativeOptions::kCopyWindowAuto) {
        return speculative;
    }
    std::uint32_t deepest = speculative.draft_tokens;
    for (const std::uint32_t k : speculative.request_draft_tokens) {
        deepest = std::max(deepest, k);
    }
    const bool runs = speculative.backend == SpeculativeBackend::Mtp && copy_rounds &&
                      deepest < kMtpDecodeMaximumDrafts && speculative.ngram_min_match != 0;
    speculative.ngram_draft_tokens = runs ? kMtpDecodeMaximumDrafts : 0U;
    return speculative;
}

SequencePlanningInputs sequence_planning_inputs(const EngineOptions& options, bool copy_rounds,
                                                std::int32_t multiprocessor_count,
                                                bool graph_memory_held) {
    const SpeculativeOptions speculative = resolve_copy_window(options.speculative, copy_rounds);
    std::vector<std::uint32_t> launch_windows;
    std::vector<std::uint32_t> launch_window_min_batch;
    for (const LaunchWindowShape& window :
         launch_window_shapes(speculative, options.max_concurrency)) {
        launch_windows.push_back(window.drafts);
        launch_window_min_batch.push_back(window.min_batch);
    }
    const std::uint32_t batch_limit =
        launch_windows.empty() ? 0U
                               : speculative_batch_limit(speculative.load, options.max_concurrency);
    return SequencePlanningInputs{
        .model                   = nullptr,
        .capacity                = options.max_context,
        .max_concurrency         = options.max_concurrency,
        .prefill_chunk           = std::min(options.prefill_chunk, options.max_context),
        .draft_window            = launch_windows.empty() ? 0U : launch_windows.back(),
        .launch_windows          = launch_windows,
        .launch_window_min_batch = launch_window_min_batch,
        .launch_window_batches =
            launch_window_batches(launch_windows, launch_window_min_batch, speculative.load,
                                  options.max_concurrency, batch_limit),
        .speculative_batch_limit = batch_limit,
        .default_draft_depth     = speculative.draft_tokens,
        .speculative_backend     = speculative.backend,
        .kv_storage              = options.kv_cache,
        .ring_storage            = options.sliding_kv_cache,
        .recurrent_state_dtype   = recurrent_state_dtype(options.recurrent_state),
        .proposal_head           = speculative.proposal_head,
        .draft_width             = speculative.draft_width,
        .acceptance_floor        = speculative.acceptance_floor,
        .step_down               = speculative.step_down,
        .load                    = speculative.load,
        .context_limit           = speculative.context_limit,
        .watchdog_window         = speculative.watchdog_window,
        .watchdog_min_rounds     = speculative.watchdog_min_rounds,
        .shared_trip_floor       = speculative.shared_trip_floor,
        .ngram_draft_tokens      = speculative.ngram_draft_tokens,
        .ngram_min_match         = speculative.ngram_min_match,
        .kv_lease_window         = options.kv_lease_window,
        .decode_pacing           = options.pacing,
        .features                = models::load_options(options),
        .use_cuda_graph          = options.use_cuda_graph,
        .causal_scoring          = options.purpose == EnginePurpose::CausalScoring,
        .device                  = options.device,
        .multiprocessor_count    = multiprocessor_count,
        .context_cache           = options.context_cache,
        .graph_memory_held       = graph_memory_held,
    };
}

SequencePageRange sequence_page_range(std::uint32_t capacity, std::uint32_t max_concurrency) {
    const std::uint32_t logical_pages = paged_kv_capacity_pages(capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    return {.minimum = minimum_pages, .maximum = static_cast<std::uint32_t>(maximum_pages64)};
}

std::unique_ptr<SequencePlanImpl> sequence_plan_shell(const SequencePlanningInputs& inputs,
                                                      std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                     = std::make_unique<SequencePlanImpl>();
    impl->model                   = inputs.model;
    impl->capacity                = inputs.capacity;
    impl->main_page_groups        = main_page_groups;
    impl->kv_capacity             = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency         = inputs.max_concurrency;
    impl->prefill_chunk           = inputs.prefill_chunk;
    impl->draft_window            = inputs.draft_window;
    impl->launch_windows          = inputs.launch_windows;
    impl->launch_window_min_batch = inputs.launch_window_min_batch;
    impl->launch_window_batches   = inputs.launch_window_batches;
    impl->speculative_batch_limit = inputs.speculative_batch_limit;
    impl->default_draft_depth     = inputs.default_draft_depth;
    impl->speculative_backend     = inputs.speculative_backend;
    impl->proposal_head           = inputs.proposal_head;
    impl->draft_width             = inputs.draft_width;
    impl->acceptance_floor        = inputs.acceptance_floor;
    impl->step_down               = inputs.step_down;
    impl->load                    = inputs.load;
    impl->context_limit           = inputs.context_limit;
    impl->watchdog_window         = inputs.watchdog_window;
    impl->watchdog_min_rounds     = inputs.watchdog_min_rounds;
    impl->shared_trip_floor       = inputs.shared_trip_floor;
    impl->ngram_draft_tokens      = inputs.ngram_draft_tokens;
    impl->ngram_min_match         = inputs.ngram_min_match;
    impl->kv_lease_window         = inputs.kv_lease_window;
    impl->decode_pacing           = inputs.decode_pacing;
    impl->features                = inputs.features;
    impl->use_cuda_graph          = inputs.use_cuda_graph;
    impl->causal_scoring          = inputs.causal_scoring;
    impl->device                  = inputs.device;
    impl->multiprocessor_count    = inputs.multiprocessor_count;
    impl->context_cache           = inputs.context_cache;
    impl->kv_storage              = inputs.kv_storage;
    impl->ring_storage            = inputs.ring_storage;
    impl->recurrent_state_dtype   = inputs.recurrent_state_dtype;
    return impl;
}

void plan_graph_allowance(SequencePlanImpl& plan, bool graph_memory_held) {
    plan.graph_allowance_bytes = 0;
    plan.launch_window_graph_bytes.clear();
    if (!plan.use_cuda_graph || graph_memory_held) { return; }
    // Definitions remain per execution profile, but only one executable is instantiated for
    // each reachable node-topology class: the allowance is a bound per instantiated class.
    // Measured (ninferno-serve's startup ledger: allowance - (free after startup - planned
    // slack); every graph is captured at startup, memory does not grow while serving): an
    // ordinary class takes 2-9 MiB, an MTP or DFlash class 3-7 MiB, whatever the visible
    // length (Qwen 3.6 35B MTP C=8 at 262,144 tokens: 34 MiB in all; Qwen 3.8 27B DFlash2
    // depths 5/11/15 at C=8: 370 MiB for 64 classes; Gemma 4 26B MTP C=8: 103 MiB). The bounds
    // keep about twice the largest. The driver and module state every engine materializes
    // besides (7-31 MiB) is the automatic KV headroom's (kDefaultKvCapacityHeadroomBytes).
    // Every generation engine captures the plain decode graphs (a speculative engine runs a
    // request whose acceptance watchdog tripped through them).
    constexpr std::uint64_t kOrdinaryClassBytes    = 12ULL * kMiB;
    constexpr std::uint64_t kSpeculativeClassBytes = 16ULL * kMiB;
    plan.graph_allowance_bytes =
        checked_mul(kOrdinaryClassBytes, plan.max_concurrency, "ordinary exact-b graph allowance");
    // One set of speculative graphs per launch window, for the speculative batches it runs
    // under the load bounds (launch_window_batches: at most the widest speculative launch,
    // below a bound of 0; a depth only a bound asks from that bound's lanes; a window deeper
    // than every bound from a step's rows on, not at that many rows).
    plan.launch_window_graph_bytes.assign(plan.launch_windows.size(), 0);
    for (std::size_t w = 0; w < plan.launch_windows.size(); ++w) {
        const std::uint32_t window = plan.launch_windows[w];
        const std::uint32_t mask   = plan.launch_window_batches.at(w);
        const std::uint32_t last   = plan.speculative_batch_limit;
        std::size_t& window_bytes  = plan.launch_window_graph_bytes[w];
        if (plan.speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles                   = mtp_graph_profiles(plan.capacity, window);
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles, [](GraphExecutionProfile) { return kSpeculativeClassBytes; },
                "MTP graph allowance");
            std::uint32_t batches = 0;
            for (std::uint32_t batch_size = 1; batch_size <= last; ++batch_size) {
                batches += window_runs_batch(mask, batch_size) ? 1U : 0U;
            }
            window_bytes = checked_mul(per_batch_allowance, batches, "MTP exact-b graph allowance");
        } else {
            for (std::uint32_t batch_size = 1; batch_size <= last; ++batch_size) {
                if (!window_runs_batch(mask, batch_size)) { continue; }
                const auto profiles = dflash_graph_profiles(plan.speculative_backend, plan.capacity,
                                                            window, batch_size);
                window_bytes        = checked_add(
                    window_bytes,
                    graph_topology_allowance(
                        profiles, [](GraphExecutionProfile) { return kSpeculativeClassBytes; },
                        "DFlash graph allowance"),
                    "DFlash exact-b graph allowance");
            }
        }
        plan.graph_allowance_bytes =
            checked_add(plan.graph_allowance_bytes, window_bytes, "speculative graphs");
    }
}

std::uint32_t ring_decoder_widest_call(const SequencePlanImpl& plan) noexcept {
    // A prefill chunk, or a speculative verify of the draft window.
    return std::max(std::min(plan.prefill_chunk, plan.capacity), plan.draft_window + 1U);
}

} // namespace ninferno::models::qwen3_5::detail
