#include "ninferno/ops/softmax_attention.h"

#include "core/arena.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

constexpr int kPage = 64;

// The registered profile under test: [128,32,8] (group 4), then [64,64,8] (group 8, gpt-oss's
// DFlash drafter). Every case below reads it.
struct Profile {
    int d;
    int q_heads;
    int kv_heads;
};
int kD       = 128;
int kQHeads  = 32;
int kKVHeads = 8;
float kScale = 0.08838834764831844055f;
ops::AttentionHeadGeometry kGeometry{128, 32, 8};

void select_profile(Profile selected) {
    kD        = selected.d;
    kQHeads   = selected.q_heads;
    kKVHeads  = selected.kv_heads;
    kScale    = 1.0F / std::sqrt(static_cast<float>(selected.d));
    kGeometry = {selected.d, selected.q_heads, selected.kv_heads};
}

constexpr ReductionCriterion kContextQueryBf16Criterion{
    .relative_l2                     = 2.95e-3,
    .gross_absolute                  = 3e-4,
    .gross_relative_to_max_reference = 5.7e-3,
};

std::size_t q_index(int d, int q_head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(q_head) +
                static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token));
}

std::size_t query_kv_index(int d, int kv_head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(kv_head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t context_index(int d, int kv_head, int position, int padded_context) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(kv_head));
}

enum class MappingPattern {
    Identity,
    Offset,
    Fragmented,
};

std::vector<std::int32_t> page_mapping(int logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) {
        switch (pattern) {
        case MappingPattern::Identity:
            mapping[static_cast<std::size_t>(page)] = page;
            break;
        case MappingPattern::Offset:
            mapping[static_cast<std::size_t>(page)] = logical_pages + page;
            break;
        case MappingPattern::Fragmented:
            mapping[static_cast<std::size_t>(page)] = 2 * page + 1;
            break;
        }
    }
    return mapping;
}

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::size_t physical_context_index(int d, int kv_head, int position, int physical_page,
                                   int physical_pages) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(position % kPage) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(physical_page) +
                     static_cast<std::size_t>(physical_pages) * static_cast<std::size_t>(kv_head)));
}

void scatter_context_mapping(std::vector<float>& physical, const std::vector<float>& logical,
                             int logical_stride, int storage_capacity,
                             const std::vector<std::int32_t>& mapping, int physical_pages) {
    for (int position = 0; position < storage_capacity; ++position) {
        const int physical_page = mapping[static_cast<std::size_t>(position / kPage)];
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) {
                physical[physical_context_index(d, kv_head, position, physical_page,
                                                physical_pages)] =
                    logical[context_index(d, kv_head, position, logical_stride)];
            }
        }
    }
}

std::vector<float> scatter_context(const std::vector<float>& logical, int logical_stride,
                                   int storage_capacity, int physical_pages,
                                   const std::vector<std::int32_t>& mapping, std::uint32_t seed) {
    std::vector<float> physical(static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages);
    fill_uniform(physical, seed, -0.9f, 0.9f);
    round_to_bf16(physical);
    scatter_context_mapping(physical, logical, logical_stride, storage_capacity, mapping,
                            physical_pages);
    return physical;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<std::uint16_t> fp16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const __half value = __float2half_rn(values[i]);
        std::memcpy(&bits[i], &value, sizeof(value));
    }
    return bits;
}

void context_attention_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                              const std::vector<float>& query_v,
                              const std::vector<float>& context_k,
                              const std::vector<float>& context_v, int tokens, int context_length,
                              int valid_columns, int padded_context, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    const int key_count = context_length + valid_columns;
    naive_dense_softmax_attention(
        kGeometry, valid_columns, key_count, static_cast<double>(kScale),
        [&](int d, int head, int token) { return static_cast<double>(q[q_index(d, head, token)]); },
        [&](int d, int head, int key) {
            return key < context_length
                       ? static_cast<double>(context_k[context_index(d, head, key, padded_context)])
                       : static_cast<double>(
                             query_k[query_kv_index(d, head, key - context_length)]);
        },
        [&](int d, int head, int key) {
            return key < context_length
                       ? static_cast<double>(context_v[context_index(d, head, key, padded_context)])
                       : static_cast<double>(
                             query_v[query_kv_index(d, head, key - context_length)]);
        },
        [](int, int) { return true; },
        [&](int d, int head, int token, double value) { out[q_index(d, head, token)] = value; });
}

PagedKVBatchLayerView make_context_view(DeviceBuffer& k, DeviceBuffer& v,
                                        DeviceBuffer& block_tables, int logical_pages,
                                        int physical_pages, int table_rows = 1) {
    return {
        .k_pages      = Tensor(k.p, DType::BF16, {kD, kPage, physical_pages, kKVHeads}),
        .v_pages      = Tensor(v.p, DType::FP16, {kD, kPage, physical_pages, kKVHeads}),
        .block_tables = Tensor(block_tables.p, DType::I32, {logical_pages, table_rows}),
        .head_dim     = kD,
        .num_kv_heads = kKVHeads,
        .storage      = KvCacheStorage::BFloat16,
    };
}

enum class InputProfile {
    Random,
    QueryVisibility,
};

int run_case(int tokens, int context_length, InputProfile profile = InputProfile::Random,
             int envelope_max = -1, MappingPattern mapping_pattern = MappingPattern::Identity,
             int valid_columns = -1, bool sparse_envelope_storage = false) {
    if (envelope_max < 0) envelope_max = context_length;
    if (valid_columns < 0) valid_columns = tokens;
    const int max_context      = std::max({context_length, envelope_max, 1});
    const int populated_rows   = std::max(context_length, 1);
    const int populated_pages  = (populated_rows + kPage - 1) / kPage;
    const int logical_pages    = (max_context + kPage - 1) / kPage;
    const int logical_capacity = logical_pages * kPage;
    const int logical_stride =
        ((sparse_envelope_storage ? populated_rows : max_context) + 127) / 128 * 128;
    const int physical_pages  = 2 * (sparse_envelope_storage ? populated_pages : logical_pages) + 1;
    const std::size_t q_count = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_context_count =
        static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    const std::size_t physical_context_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_kv_count);
    std::vector<float> query_v(query_kv_count);
    std::vector<float> context_k(logical_context_count);
    std::vector<float> context_v(logical_context_count);

    const auto seed = static_cast<unsigned>(tokens * 131 + context_length * 17);
    fill_uniform(q, 1001u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 2003u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 3001u + seed, -0.8f, 0.8f);
    fill_uniform(context_k, 4001u + seed, -0.4f, 0.4f);
    fill_uniform(context_v, 5003u + seed, -0.8f, 0.8f);

    if (profile == InputProfile::QueryVisibility) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(query_k.begin(), query_k.end(), 0.0f);
        std::fill(query_v.begin(), query_v.end(), 0.0f);
        std::fill(context_k.begin(), context_k.end(), 0.0f);
        std::fill(context_v.begin(), context_v.end(), 0.0f);
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) { query_v[query_kv_index(d, kv_head, tokens - 1)] = 1.0f; }
        }
    }

    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    std::vector<double> reference;
    context_attention_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             valid_columns, logical_stride, reference);

    std::vector<std::int32_t> mapping = page_mapping(logical_pages, mapping_pattern);
    if (sparse_envelope_storage) { std::fill(mapping.begin() + populated_pages, mapping.end(), 0); }
    if (context_length == 0 && envelope_max == 0) {
        mapping[0] = std::numeric_limits<std::int32_t>::max();
    }
    const int scattered_rows = sparse_envelope_storage ? populated_rows : logical_capacity;
    const std::vector<float> physical_k =
        context_length == 0 && envelope_max == 0
            ? std::vector<float>(static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages,
                                 0.25f)
            : scatter_context(context_k, logical_stride, scattered_rows, physical_pages, mapping,
                              6007u + seed);
    const std::vector<float> physical_v =
        context_length == 0 && envelope_max == 0
            ? std::vector<float>(static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages,
                                 -0.5f)
            : scatter_context(context_v, logical_stride, scattered_rows, physical_pages, mapping,
                              7001u + seed);

    const auto q_expected         = bf16_bits(q);
    const auto query_k_expected   = bf16_bits(query_k);
    const auto query_v_expected   = bf16_bits(query_v);
    const auto context_k_expected = bf16_bits(physical_k);
    const auto context_v_expected = fp16_bits(physical_v);
    const std::vector<int> length_expected{context_length};
    const std::vector<int> valid_expected{valid_columns};
    const std::vector<int> table_row_expected{0};

    DeviceBuffer d_q         = to_device(q_expected);
    DeviceBuffer d_query_k   = to_device(query_k_expected);
    DeviceBuffer d_query_v   = to_device(query_v_expected);
    DeviceBuffer d_context_k = to_device(context_k_expected);
    DeviceBuffer d_context_v = to_device(context_v_expected);
    DeviceBuffer d_table     = to_device(mapping);
    DeviceBuffer d_length    = to_device_i32(length_expected);
    DeviceBuffer d_valid     = to_device_i32(valid_expected);
    DeviceBuffer d_table_row = to_device_i32(table_row_expected);
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor length_tensor(d_length.p, DType::I32, {1});
    Tensor valid_tensor(d_valid.p, DType::I32, {1});
    Tensor table_row_tensor(d_table_row.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    PagedKVBatchLayerView context =
        make_context_view(d_context_k, d_context_v, d_table, logical_pages, physical_pages);
    const ops::ContextAttentionExecutionEnvelope envelope{0,
                                                          static_cast<std::uint32_t>(envelope_max)};
    const std::size_t workspace_bytes = ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, envelope, tokens, tokens, 1);
    DeviceArena workspace(workspace_bytes);

    ops::context_softmax_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                   valid_tensor, table_row_tensor, kGeometry, kScale, context,
                                   envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();

    std::string label = "context_softmax_attention T=" + std::to_string(tokens) +
                        " V=" + std::to_string(valid_columns) +
                        " L=" + std::to_string(context_length) +
                        " mapping=" + mapping_name(mapping_pattern);
    if (envelope_max != context_length) {
        label += " envelope=[0," + std::to_string(envelope_max) + "]";
    }
    if (sparse_envelope_storage) label += " sparse-envelope-storage";
    if (profile == InputProfile::QueryVisibility) label += " query-visibility";

    int failures =
        valid_columns == 0
            ? verify_exact(label.c_str(), from_device<std::uint16_t>(d_out.data(), q_count),
                           std::vector<std::uint16_t>(q_count, 0))
            : verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count), reference,
                               kContextQueryBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(d_q, q_count), q_expected);
    failures +=
        verify_exact((label + " query k unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_k, query_kv_count), query_k_expected);
    failures +=
        verify_exact((label + " query v unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_v, query_kv_count), query_v_expected);
    failures += verify_exact((label + " context k unchanged").c_str(),
                             from_device<std::uint16_t>(d_context_k, physical_context_count),
                             context_k_expected);
    failures += verify_exact((label + " context v unchanged").c_str(),
                             from_device<std::uint16_t>(d_context_v, physical_context_count),
                             context_v_expected);
    failures += verify_exact((label + " block table unchanged").c_str(),
                             from_device<std::int32_t>(d_table, mapping.size()), mapping);
    failures += verify_exact((label + " context length unchanged").c_str(),
                             from_device<int>(d_length, 1), length_expected);
    failures += verify_exact((label + " valid columns unchanged").c_str(),
                             from_device<int>(d_valid, 1), valid_expected);
    failures += verify_exact((label + " table row unchanged").c_str(),
                             from_device<int>(d_table_row, 1), table_row_expected);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

// context_window_attention: the plain op over the last `window` context rows. The oracle is the
// plain one with the rows below the window masked.
int run_window_case(int tokens, int context_length, int window,
                    MappingPattern mapping_pattern = MappingPattern::Fragmented,
                    int valid_columns              = -1) {
    if (valid_columns < 0) valid_columns = tokens;
    const int max_context            = std::max(context_length, 1);
    const int logical_pages          = (max_context + kPage - 1) / kPage;
    const int logical_capacity       = logical_pages * kPage;
    const int logical_stride         = (max_context + 127) / 128 * 128;
    const int physical_pages         = 2 * logical_pages + 1;
    const std::size_t q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_context_count =
        static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    std::vector<float> q(q_count), query_k(query_kv_count), query_v(query_kv_count);
    std::vector<float> context_k(logical_context_count), context_v(logical_context_count);
    const auto seed = static_cast<unsigned>(tokens * 131 + context_length * 17 + window);
    fill_uniform(q, 1001u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 2003u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 3001u + seed, -0.8f, 0.8f);
    fill_uniform(context_k, 4001u + seed, -0.4f, 0.4f);
    fill_uniform(context_v, 5003u + seed, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    const int first = std::max(context_length - window, 0);
    std::vector<double> reference(q_count, 0.0);
    naive_dense_softmax_attention(
        kGeometry, valid_columns, context_length + valid_columns, static_cast<double>(kScale),
        [&](int d, int head, int token) { return static_cast<double>(q[q_index(d, head, token)]); },
        [&](int d, int head, int key) {
            return key < context_length
                       ? static_cast<double>(context_k[context_index(d, head, key, logical_stride)])
                       : static_cast<double>(
                             query_k[query_kv_index(d, head, key - context_length)]);
        },
        [&](int d, int head, int key) {
            return key < context_length
                       ? static_cast<double>(context_v[context_index(d, head, key, logical_stride)])
                       : static_cast<double>(
                             query_v[query_kv_index(d, head, key - context_length)]);
        },
        [&](int, int key) { return key >= first; },
        [&](int d, int head, int token, double value) {
            reference[q_index(d, head, token)] = value;
        });

    const std::vector<std::int32_t> mapping = page_mapping(logical_pages, mapping_pattern);
    const std::vector<float> physical_k     = scatter_context(
        context_k, logical_stride, logical_capacity, physical_pages, mapping, 6007u + seed);
    const std::vector<float> physical_v = scatter_context(
        context_v, logical_stride, logical_capacity, physical_pages, mapping, 7001u + seed);
    DeviceBuffer d_q         = to_device(bf16_bits(q));
    DeviceBuffer d_query_k   = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v   = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k = to_device(bf16_bits(physical_k));
    DeviceBuffer d_context_v = to_device(fp16_bits(physical_v));
    DeviceBuffer d_table     = to_device(mapping);
    DeviceBuffer d_length    = to_device_i32(std::vector<int>{context_length});
    DeviceBuffer d_valid     = to_device_i32(std::vector<int>{valid_columns});
    DeviceBuffer d_table_row = to_device_i32(std::vector<int>{0});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor length_tensor(d_length.p, DType::I32, {1});
    Tensor valid_tensor(d_valid.p, DType::I32, {1});
    Tensor table_row_tensor(d_table_row.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    PagedKVBatchLayerView context =
        make_context_view(d_context_k, d_context_v, d_table, logical_pages, physical_pages);
    const ops::ContextAttentionExecutionEnvelope envelope{0,
                                                          static_cast<std::uint32_t>(max_context)};
    const std::size_t workspace_bytes = ops::context_window_attention_workspace_capacity_bytes(
        kGeometry, envelope, window, tokens, tokens, 1);
    DeviceArena workspace(workspace_bytes);
    ops::context_window_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                  valid_tensor, table_row_tensor, kGeometry, kScale, window,
                                  context, envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();
    const std::string label = "context_window_attention T=" + std::to_string(tokens) +
                              " V=" + std::to_string(valid_columns) +
                              " L=" + std::to_string(context_length) +
                              " W=" + std::to_string(window);
    int failures = verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                    reference, kContextQueryBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_window_cases() {
    int failures = 0;
    failures += run_window_case(1, 50, 64); // L <= W: the plain op
    failures += run_window_case(4, 64, 64);
    failures += run_window_case(1, 300, 64);
    failures += run_window_case(4, 300, 100, MappingPattern::Offset); // W not a tile multiple
    failures += run_window_case(8, 1000, 128);
    failures += run_window_case(8, 1000, 130, MappingPattern::Fragmented, 5);
    failures += run_window_case(16, 700, 256);
    failures += run_window_case(2, 5000, 2048);
    failures += run_window_case(8, 4097, 1);
    return failures;
}

int graph_mapping_replay_case() {
    constexpr int tokens            = 1;
    constexpr int context_length    = 65;
    constexpr int logical_pages     = 2;
    constexpr int logical_capacity  = logical_pages * kPage;
    constexpr int physical_pages    = 6;
    constexpr int logical_stride    = 128;
    const std::size_t q_count       = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_count   = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t context_count = static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    const std::size_t physical_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_count);
    std::vector<float> query_v(query_count);
    std::vector<float> context_k(context_count);
    std::vector<float> context_v(context_count);
    fill_uniform(q, 11003u, -0.35f, 0.35f);
    fill_uniform(query_k, 12007u, -0.4f, 0.4f);
    fill_uniform(query_v, 13001u, -0.8f, 0.8f);
    fill_uniform(context_k, 14009u, -0.4f, 0.4f);
    fill_uniform(context_v, 15013u, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    const std::array<std::array<std::int32_t, logical_pages>, 3> replay_mappings{
        std::array<std::int32_t, logical_pages>{0, 1},
        std::array<std::int32_t, logical_pages>{2, 3},
        std::array<std::int32_t, logical_pages>{5, 4},
    };
    std::vector<float> physical_k(physical_count);
    std::vector<float> physical_v(physical_count);
    for (const auto& replay_mapping : replay_mappings) {
        const std::vector<std::int32_t> mapping(replay_mapping.begin(), replay_mapping.end());
        scatter_context_mapping(physical_k, context_k, logical_stride, logical_capacity, mapping,
                                physical_pages);
        scatter_context_mapping(physical_v, context_v, logical_stride, logical_capacity, mapping,
                                physical_pages);
    }

    std::vector<double> reference;
    context_attention_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             tokens, logical_stride, reference);

    DeviceBuffer d_q         = to_device(bf16_bits(q));
    DeviceBuffer d_query_k   = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v   = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k = to_device(bf16_bits(physical_k));
    DeviceBuffer d_context_v = to_device(fp16_bits(physical_v));
    DeviceBuffer d_table     = to_device<std::int32_t>({0, 1});
    DeviceBuffer d_length    = to_device_i32({context_length});
    DeviceBuffer d_valid     = to_device_i32({tokens});
    DeviceBuffer d_table_row = to_device_i32({0});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor length_tensor(d_length.p, DType::I32, {1});
    Tensor valid_tensor(d_valid.p, DType::I32, {1});
    Tensor table_row_tensor(d_table_row.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    PagedKVBatchLayerView context =
        make_context_view(d_context_k, d_context_v, d_table, logical_pages, physical_pages);
    constexpr ops::ContextAttentionExecutionEnvelope envelope{0, context_length};
    DeviceArena workspace(ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, envelope, tokens, tokens, 1));

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create context-attention stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin context-attention capture");
    ops::context_softmax_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                   valid_tensor, table_row_tensor, kGeometry, kScale, context,
                                   envelope, workspace, out_tensor, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end context-attention capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate context-attention graph");

    int failures = 0;
    for (const auto& replay_mapping : replay_mappings) {
        d_table.copy_from_host(replay_mapping.data(), sizeof(replay_mapping));
        cuda_check(cudaGraphLaunch(executable, stream), "launch context-attention graph");
        cuda_synchronize(stream);
        const std::vector<std::int32_t> mapping(replay_mapping.begin(), replay_mapping.end());
        const std::string label =
            "context_softmax_attention graph mapping=" + std::to_string(mapping[0]) + "," +
            std::to_string(mapping[1]);
        failures += verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                     reference, kContextQueryBf16Criterion);
        failures += verify_exact((label + " block table unchanged").c_str(),
                                 from_device<std::int32_t>(d_table, mapping.size()), mapping);
    }

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += d_out.verify_guards("context-attention graph output guards");
    return failures;
}

int batch_table_case() {
    constexpr int tokens            = 2;
    constexpr int batch             = 2;
    constexpr int logical_pages     = 2;
    constexpr int logical_capacity  = logical_pages * kPage;
    constexpr int physical_pages    = 6;
    const std::size_t row_q_count   = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t row_kv_count  = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_count = static_cast<std::size_t>(kD) * logical_capacity * kKVHeads;
    const std::size_t physical_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(row_q_count * batch);
    std::vector<float> query_k(row_kv_count * batch);
    std::vector<float> query_v(row_kv_count * batch);
    std::vector<float> logical_k(logical_count * batch);
    std::vector<float> logical_v(logical_count * batch);
    std::vector<float> physical_k(physical_count);
    std::vector<float> physical_v(physical_count);
    fill_uniform(q, 16001u, -0.35f, 0.35f);
    fill_uniform(query_k, 17011u, -0.4f, 0.4f);
    fill_uniform(query_v, 18013u, -0.8f, 0.8f);
    fill_uniform(logical_k, 19001u, -0.4f, 0.4f);
    fill_uniform(logical_v, 20011u, -0.8f, 0.8f);
    fill_uniform(physical_k, 21001u, -0.9f, 0.9f);
    fill_uniform(physical_v, 22003u, -0.9f, 0.9f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(logical_k);
    round_to_bf16(logical_v);
    round_to_bf16(physical_k);
    round_to_bf16(physical_v);

    const std::vector<std::int32_t> block_tables{0, 2, 5, 3};
    for (int row = 0; row < batch; ++row) {
        const std::vector<float> row_k(
            logical_k.begin() + static_cast<std::ptrdiff_t>(row * logical_count),
            logical_k.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_count));
        const std::vector<float> row_v(
            logical_v.begin() + static_cast<std::ptrdiff_t>(row * logical_count),
            logical_v.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_count));
        const std::vector<std::int32_t> mapping(
            block_tables.begin() + static_cast<std::ptrdiff_t>(row * logical_pages),
            block_tables.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_pages));
        scatter_context_mapping(physical_k, row_k, logical_capacity, logical_capacity, mapping,
                                physical_pages);
        scatter_context_mapping(physical_v, row_v, logical_capacity, logical_capacity, mapping,
                                physical_pages);
    }

    const std::vector<std::int32_t> lengths{65, 1};
    const std::vector<std::int32_t> valid{2, 1};
    const std::vector<std::int32_t> table_rows{1, 0};
    DeviceBuffer d_q          = to_device(bf16_bits(q));
    DeviceBuffer d_query_k    = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v    = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k  = to_device(bf16_bits(physical_k));
    DeviceBuffer d_context_v  = to_device(fp16_bits(physical_v));
    DeviceBuffer d_tables     = to_device(block_tables);
    DeviceBuffer d_lengths    = to_device(lengths);
    DeviceBuffer d_valid      = to_device(valid);
    DeviceBuffer d_table_rows = to_device(table_rows);
    GuardedDeviceBuffer d_out(row_q_count * batch * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, batch});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor length_tensor(d_lengths.p, DType::I32, {batch});
    Tensor valid_tensor(d_valid.p, DType::I32, {batch});
    Tensor table_row_tensor(d_table_rows.p, DType::I32, {batch});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, batch});
    auto context =
        make_context_view(d_context_k, d_context_v, d_tables, logical_pages, physical_pages, batch);
    constexpr ops::ContextAttentionExecutionEnvelope envelope{0, 65};

    std::vector<double> expected(row_q_count * batch);
    for (int b = 0; b < batch; ++b) {
        const auto q_begin     = q.begin() + static_cast<std::ptrdiff_t>(b * row_q_count);
        const auto kv_begin    = query_k.begin() + static_cast<std::ptrdiff_t>(b * row_kv_count);
        const int selected_row = table_rows[static_cast<std::size_t>(b)];
        const auto context_begin =
            logical_k.begin() + static_cast<std::ptrdiff_t>(selected_row * logical_count);
        const std::vector<float> row_q(q_begin, q_begin + row_q_count);
        const std::vector<float> row_query_k(kv_begin, kv_begin + row_kv_count);
        const auto query_v_begin = query_v.begin() + static_cast<std::ptrdiff_t>(b * row_kv_count);
        const std::vector<float> row_query_v(query_v_begin, query_v_begin + row_kv_count);
        const std::vector<float> row_context_k(context_begin, context_begin + logical_count);
        const auto context_v_begin =
            logical_v.begin() + static_cast<std::ptrdiff_t>(selected_row * logical_count);
        const std::vector<float> row_context_v(context_v_begin, context_v_begin + logical_count);
        std::vector<double> row_expected;
        context_attention_oracle(row_q, row_query_k, row_query_v, row_context_k, row_context_v,
                                 tokens, lengths[static_cast<std::size_t>(b)],
                                 valid[static_cast<std::size_t>(b)], logical_capacity,
                                 row_expected);
        std::copy(row_expected.begin(), row_expected.end(),
                  expected.begin() + static_cast<std::ptrdiff_t>(b * row_q_count));
    }

    DeviceArena workspace(ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, envelope, tokens, tokens, batch));
    ops::context_softmax_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                   valid_tensor, table_row_tensor, kGeometry, kScale, context,
                                   envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction("context_softmax_attention B=2 mixed lengths and table rows",
                                    from_device_bf16(d_out.data(), row_q_count * batch), expected,
                                    kContextQueryBf16Criterion);
    failures += d_out.verify_guards("context-attention B=2 output guards");
    return failures;
}

int run_profile_cases() {
    int failures = 0;
    constexpr ops::ContextAttentionExecutionEnvelope capacity_envelope{0, 196609};
    const std::size_t interval = ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, capacity_envelope, 1, 16, 1);
    const std::size_t witness = std::max(ops::context_softmax_attention_workspace_capacity_bytes(
                                             kGeometry, capacity_envelope, 8, 8, 1),
                                         ops::context_softmax_attention_workspace_capacity_bytes(
                                             kGeometry, capacity_envelope, 16, 16, 1));
    if (interval != witness) {
        std::cerr << "context-attention interval capacity missed a token-band endpoint\n";
        ++failures;
    }
    try {
        (void)ops::context_softmax_attention_workspace_capacity_bytes(kGeometry, capacity_envelope,
                                                                      9, 8, 1);
        std::cerr << "context-attention accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += run_case(1, 0);
    failures += run_case(2, 1);
    failures += run_case(1, 63);
    failures += run_case(2, 64, InputProfile::Random, -1, MappingPattern::Offset);
    failures += run_case(4, 65, InputProfile::Random, -1, MappingPattern::Fragmented);
    failures += run_case(8, 95, InputProfile::Random, 4096, MappingPattern::Fragmented);
    failures += run_case(16, 65, InputProfile::Random, 65537, MappingPattern::Fragmented, 16, true);
    failures += run_case(16, 257);
    failures += run_case(7, 130, InputProfile::Random, -1, MappingPattern::Fragmented);
    failures += run_case(9, 200, InputProfile::Random, -1, MappingPattern::Offset);
    failures += run_case(12, 70, InputProfile::Random, 70000, MappingPattern::Fragmented, 10, true);
    failures += run_case(1, 4096, InputProfile::Random, -1, MappingPattern::Fragmented);
    failures += run_case(4, 0, InputProfile::QueryVisibility);
    failures += run_case(8, 0, InputProfile::Random, 0, MappingPattern::Identity, 0);
    failures += run_case(8, 65, InputProfile::Random, 65, MappingPattern::Fragmented, 0);
    failures += graph_mapping_replay_case();
    failures += batch_table_case();
    failures += run_window_cases();
    return failures;
}

// Int8Group64 context at D128 (the DFlash drafter's int8 pool): per-token codes with an FP16
// scale per 64 dimensions, FP16-RNE(absmax / 127), the codes from the reciprocal of that value as
// kv_cache_append writes them. The oracle reads the dequantized context the kernel stages: K
// rounded to BF16, V to FP16.
struct Int8Context {
    std::vector<std::int8_t> codes;
    std::vector<__half> scales;
    std::vector<float> dequant;
};

Int8Context quantize_context_int8(const std::vector<float>& physical, int physical_pages,
                                  bool value_plane) {
    const int groups = kD / 64;
    Int8Context out;
    out.codes.resize(physical.size());
    out.dequant.resize(physical.size());
    out.scales.resize(static_cast<std::size_t>(groups) * kPage * physical_pages * kKVHeads);
    for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (int page = 0; page < physical_pages; ++page) {
            for (int row = 0; row < kPage; ++row) {
                for (int g = 0; g < groups; ++g) {
                    const std::size_t base =
                        physical_context_index(g * 64, kv_head, row, page, physical_pages);
                    float m = 0.0f;
                    for (int i = 0; i < 64; ++i) m = std::max(m, std::fabs(physical[base + i]));
                    const __half scale  = __float2half_rn(m > 0.0f ? m / 127.0f : 0.0f);
                    const float s       = __half2float(scale);
                    const float inverse = s > 0.0f ? 1.0f / s : 0.0f;
                    for (int i = 0; i < 64; ++i) {
                        const int q = std::clamp(
                            static_cast<int>(std::nearbyint(physical[base + i] * inverse)), -127,
                            127);
                        out.codes[base + i]   = static_cast<std::int8_t>(q);
                        const float x         = static_cast<float>(q) * s;
                        out.dequant[base + i] = value_plane ? __half2float(__float2half_rn(x))
                                                            : bf16_to_f32(f32_to_bf16(x));
                    }
                    out.scales[static_cast<std::size_t>(g) +
                               static_cast<std::size_t>(groups) *
                                   (static_cast<std::size_t>(row) +
                                    static_cast<std::size_t>(kPage) *
                                        (static_cast<std::size_t>(page) +
                                         static_cast<std::size_t>(physical_pages) *
                                             static_cast<std::size_t>(kv_head)))] = scale;
                }
            }
        }
    }
    return out;
}

int run_int8_case(int tokens, int context_length, MappingPattern mapping_pattern) {
    const int logical_pages          = std::max((context_length + kPage - 1) / kPage, 1);
    const int logical_stride         = (logical_pages * kPage + 127) / 128 * 128;
    const int physical_pages         = 2 * logical_pages + 1;
    const std::size_t q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_count  = static_cast<std::size_t>(kD) * logical_stride * kKVHeads;

    std::vector<float> q(q_count), query_k(query_kv_count), query_v(query_kv_count);
    std::vector<float> context_k(logical_count), context_v(logical_count);
    const auto seed = static_cast<unsigned>(tokens * 977 + context_length * 13);
    fill_uniform(q, 11u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 23u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 37u + seed, -0.8f, 0.8f);
    fill_uniform(context_k, 41u + seed, -0.4f, 0.4f);
    fill_uniform(context_v, 53u + seed, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);

    const std::vector<std::int32_t> mapping = page_mapping(logical_pages, mapping_pattern);
    const int rows                          = logical_pages * kPage;
    const Int8Context k8                    = quantize_context_int8(
        scatter_context(context_k, logical_stride, rows, physical_pages, mapping, 61u + seed),
        physical_pages, false);
    const Int8Context v8 = quantize_context_int8(
        scatter_context(context_v, logical_stride, rows, physical_pages, mapping, 67u + seed),
        physical_pages, true);
    // The oracle's logical context: the dequantized values at each position's physical page.
    for (int position = 0; position < context_length; ++position) {
        const int physical_page = mapping[static_cast<std::size_t>(position / kPage)];
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) {
                const std::size_t from =
                    physical_context_index(d, kv_head, position, physical_page, physical_pages);
                context_k[context_index(d, kv_head, position, logical_stride)] = k8.dequant[from];
                context_v[context_index(d, kv_head, position, logical_stride)] = v8.dequant[from];
            }
        }
    }
    std::vector<double> reference;
    context_attention_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             tokens, logical_stride, reference);

    DeviceBuffer d_q       = to_device(bf16_bits(q));
    DeviceBuffer d_query_k = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v = to_device(bf16_bits(query_v));
    DeviceBuffer d_k       = to_device(k8.codes);
    DeviceBuffer d_v       = to_device(v8.codes);
    DeviceBuffer d_k_scale = to_device(k8.scales);
    DeviceBuffer d_v_scale = to_device(v8.scales);
    DeviceBuffer d_table   = to_device(mapping);
    DeviceBuffer d_length  = to_device_i32(std::vector<int>{context_length});
    DeviceBuffer d_valid   = to_device_i32(std::vector<int>{tokens});
    DeviceBuffer d_row     = to_device_i32(std::vector<int>{0});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    const int groups = kD / 64;
    PagedKVBatchLayerView context{
        .k_pages = Tensor(d_k.p, DType::I8, {kD, kPage, physical_pages, kKVHeads}),
        .v_pages = Tensor(d_v.p, DType::I8, {kD, kPage, physical_pages, kKVHeads}),
        .k_scale_pages =
            Tensor(d_k_scale.p, DType::FP16, {groups, kPage, physical_pages, kKVHeads}),
        .v_scale_pages =
            Tensor(d_v_scale.p, DType::FP16, {groups, kPage, physical_pages, kKVHeads}),
        .block_tables = Tensor(d_table.p, DType::I32, {logical_pages, 1}),
        .head_dim     = kD,
        .num_kv_heads = kKVHeads,
        .storage      = KvCacheStorage::Int8Group64,
    };
    const ops::ContextAttentionExecutionEnvelope envelope{
        0, static_cast<std::uint32_t>(context_length)};
    DeviceArena workspace(ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, envelope, tokens, tokens, 1));
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    ops::context_softmax_attention(Tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1}),
                                   Tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1}),
                                   Tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1}),
                                   Tensor(d_length.p, DType::I32, {1}),
                                   Tensor(d_valid.p, DType::I32, {1}),
                                   Tensor(d_row.p, DType::I32, {1}), kGeometry, kScale, context,
                                   envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "context_softmax_attention int8 T=" + std::to_string(tokens) +
                              " L=" + std::to_string(context_length) +
                              " mapping=" + mapping_name(mapping_pattern);
    int failures = verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                    reference, kContextQueryBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    return failures;
}

int run_int8_cases() {
    int failures = 0;
    for (const MappingPattern pattern : {MappingPattern::Identity, MappingPattern::Fragmented}) {
        for (const auto [tokens, length] :
             {std::array<int, 2>{1, 1}, std::array<int, 2>{1, 300}, std::array<int, 2>{4, 1000},
              std::array<int, 2>{8, 129}, std::array<int, 2>{16, 4100}}) {
            failures += run_int8_case(tokens, length, pattern);
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// FOLD (context_fold_attention): the context cache holds UNROTATED keys; the query rows and the
// query block's own keys arrive rotated at their positions. Window branch: context rows
// [L-W, L), each key rotated to its position j (NeoX pairs (i, i + 64), angle j f_i), plus the V
// query keys as given; ordinary scores. Far branch: rows [0, L-W) with keys as stored and the
// query rotated by -shared f_i, shared = L - W - V, scores alpha * scale * q.k; Z_N = sum exp,
// Z_F = (sum exp(s_F))^(1/alpha); the output merges o_N and o_F by Z_N and Z_F. The oracle is
// double precision on the stored (BF16) values; the kernel rounds rotated operands to BF16, hence
// the looser criterion.
// ---------------------------------------------------------------------------------------------

double fold_inv_frequency(int pair) { return std::pow(1.0e7, -2.0 * pair / 128.0); }

// NeoX rotate-half of one D128 vector by angle position * f_i, in place (double).
void fold_rotate(double* x, double position) {
    const int half = kD / 2;
    for (int i = 0; i < half; ++i) {
        const double angle = position * fold_inv_frequency(i);
        const double c     = std::cos(angle);
        const double s     = std::sin(angle);
        const double a     = x[i];
        const double b     = x[i + half];
        x[i]               = a * c - b * s;
        x[i + half]        = b * c + a * s;
    }
}

constexpr ReductionCriterion kFoldBf16Criterion{
    .relative_l2                     = 6e-3,
    .gross_absolute                  = 8e-4,
    .gross_relative_to_max_reference = 1.2e-2,
};

struct FoldRow {
    int length;
    int valid;
};

// One batch row's reference, written to out[0 .. D*Hq*tokens). Tokens at or past `valid` stay 0.
void fold_oracle_row(const std::vector<float>& q, const std::vector<float>& query_k,
                     const std::vector<float>& query_v, const std::vector<float>& context_k,
                     const std::vector<float>& context_v, int tokens, int length, int valid,
                     int stride, int window, double alpha, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    const int group        = kQHeads / kKVHeads;
    const int window_start = std::max(0, length - window);
    const int shared       = length - window - valid;
    for (int token = 0; token < valid; ++token) {
        for (int head = 0; head < kQHeads; ++head) {
            const int kv = head / group;
            auto qd      = [&](int d) { return static_cast<double>(q[q_index(d, head, token)]); };
            // Window branch: context rows [window_start, length) and the `valid` query rows.
            std::vector<double> sw;
            for (int key = window_start; key < length + valid; ++key) {
                std::vector<double> k(static_cast<std::size_t>(kD));
                for (int d = 0; d < kD; ++d) {
                    k[static_cast<std::size_t>(d)] =
                        key < length
                            ? static_cast<double>(context_k[context_index(d, kv, key, stride)])
                            : static_cast<double>(query_k[query_kv_index(d, kv, key - length)]);
                }
                if (key < length) fold_rotate(k.data(), static_cast<double>(key));
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) dot += qd(d) * k[static_cast<std::size_t>(d)];
                sw.push_back(dot * static_cast<double>(kScale));
            }
            double mw = -std::numeric_limits<double>::infinity();
            for (double s : sw) mw = std::max(mw, s);
            double zw = 0.0;
            std::vector<double> on(static_cast<std::size_t>(kD), 0.0);
            for (std::size_t n = 0; n < sw.size(); ++n) {
                const double p = std::exp(sw[n] - mw);
                zw += p;
                const int key = window_start + static_cast<int>(n);
                for (int d = 0; d < kD; ++d) {
                    const double v =
                        key < length
                            ? static_cast<double>(context_v[context_index(d, kv, key, stride)])
                            : static_cast<double>(query_v[query_kv_index(d, kv, key - length)]);
                    on[static_cast<std::size_t>(d)] += p * v;
                }
            }
            const double log_zn = mw + std::log(zw);
            std::vector<double> result(static_cast<std::size_t>(kD));
            if (window_start == 0) {
                for (int d = 0; d < kD; ++d) {
                    result[static_cast<std::size_t>(d)] = on[static_cast<std::size_t>(d)] / zw;
                }
            } else {
                // Far branch.
                std::vector<double> qf(static_cast<std::size_t>(kD));
                for (int d = 0; d < kD; ++d) qf[static_cast<std::size_t>(d)] = qd(d);
                fold_rotate(qf.data(), -static_cast<double>(shared));
                std::vector<double> sf(static_cast<std::size_t>(window_start));
                double mf = -std::numeric_limits<double>::infinity();
                for (int j = 0; j < window_start; ++j) {
                    double dot = 0.0;
                    for (int d = 0; d < kD; ++d) {
                        dot += qf[static_cast<std::size_t>(d)] *
                               static_cast<double>(context_k[context_index(d, kv, j, stride)]);
                    }
                    sf[static_cast<std::size_t>(j)] = alpha * dot * static_cast<double>(kScale);
                    mf = std::max(mf, sf[static_cast<std::size_t>(j)]);
                }
                double zf = 0.0;
                std::vector<double> of(static_cast<std::size_t>(kD), 0.0);
                for (int j = 0; j < window_start; ++j) {
                    const double p = std::exp(sf[static_cast<std::size_t>(j)] - mf);
                    zf += p;
                    for (int d = 0; d < kD; ++d) {
                        of[static_cast<std::size_t>(d)] +=
                            p * static_cast<double>(context_v[context_index(d, kv, j, stride)]);
                    }
                }
                const double log_zf = (mf + std::log(zf)) / alpha; // log of (sum exp)^(1/alpha)
                const double top    = std::max(log_zn, log_zf);
                const double wn     = std::exp(log_zn - top);
                const double wf     = std::exp(log_zf - top);
                for (int d = 0; d < kD; ++d) {
                    const auto u = static_cast<std::size_t>(d);
                    result[u]    = (wn * on[u] / zw + wf * of[u] / zf) / (wn + wf);
                }
            }
            for (int d = 0; d < kD; ++d) {
                out[q_index(d, head, token)] = result[static_cast<std::size_t>(d)];
            }
        }
    }
}

// `int8`: the context in the drafter's Int8Group64 pool (the oracle reads its dequantized values).
int run_fold_case(int tokens, const std::vector<FoldRow>& rows, int window, float alpha,
                  bool compare_plain, bool int8 = false) {
    const int batch = static_cast<int>(rows.size());
    int max_length  = 1;
    for (const FoldRow& r : rows) max_length = std::max(max_length, r.length);
    const int logical_pages   = (max_length + kPage - 1) / kPage;
    const int logical_stride  = (logical_pages * kPage + 127) / 128 * 128;
    const int physical_pages  = (batch + 1) * logical_pages + 1; // pages never adjacent
    const std::size_t row_q   = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t row_kv  = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t row_ctx = static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    const std::size_t physical_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(row_q * batch), query_k(row_kv * batch), query_v(row_kv * batch);
    std::vector<float> ctx_k(row_ctx * batch), ctx_v(row_ctx * batch);
    const auto seed = static_cast<unsigned>(tokens * 71 + max_length * 13 + window * 7 + batch);
    fill_uniform(q, 31001u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 32003u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 33001u + seed, -0.8f, 0.8f);
    fill_uniform(ctx_k, 34001u + seed, -0.4f, 0.4f);
    fill_uniform(ctx_v, 35003u + seed, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(ctx_k);
    round_to_bf16(ctx_v);

    std::vector<float> physical_k(physical_count), physical_v(physical_count);
    fill_uniform(physical_k, 36001u + seed, -0.9f, 0.9f);
    fill_uniform(physical_v, 37001u + seed, -0.9f, 0.9f);
    round_to_bf16(physical_k);
    round_to_bf16(physical_v);
    std::vector<std::int32_t> tables(static_cast<std::size_t>(logical_pages) * batch);
    std::vector<double> expected(row_q * batch);
    const auto row_mapping = [&](int b) {
        std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
        for (int p = 0; p < logical_pages; ++p) {
            mapping[static_cast<std::size_t>(p)] = p * (batch + 1) + b + 1;
        }
        return mapping;
    };
    for (int b = 0; b < batch; ++b) {
        const std::vector<std::int32_t> mapping = row_mapping(b);
        std::copy(mapping.begin(), mapping.end(),
                  tables.begin() + static_cast<std::ptrdiff_t>(b * logical_pages));
        const auto k0 = ctx_k.begin() + static_cast<std::ptrdiff_t>(b * row_ctx);
        const auto v0 = ctx_v.begin() + static_cast<std::ptrdiff_t>(b * row_ctx);
        scatter_context_mapping(physical_k, std::vector<float>(k0, k0 + static_cast<std::ptrdiff_t>(row_ctx)),
                                logical_stride, logical_pages * kPage, mapping, physical_pages);
        scatter_context_mapping(physical_v, std::vector<float>(v0, v0 + static_cast<std::ptrdiff_t>(row_ctx)),
                                logical_stride, logical_pages * kPage, mapping, physical_pages);
    }
    Int8Context k8, v8;
    if (int8) {
        k8 = quantize_context_int8(physical_k, physical_pages, false);
        v8 = quantize_context_int8(physical_v, physical_pages, true);
        // Each row's logical context: the dequantized values at its positions' physical pages.
        for (int b = 0; b < batch; ++b) {
            const std::vector<std::int32_t> mapping = row_mapping(b);
            for (int position = 0; position < logical_pages * kPage; ++position) {
                const int page = mapping[static_cast<std::size_t>(position / kPage)];
                for (int kv = 0; kv < kKVHeads; ++kv) {
                    for (int d = 0; d < kD; ++d) {
                        const std::size_t from =
                            physical_context_index(d, kv, position, page, physical_pages);
                        const std::size_t to = b * row_ctx + context_index(d, kv, position, logical_stride);
                        ctx_k[to] = k8.dequant[from];
                        ctx_v[to] = v8.dequant[from];
                    }
                }
            }
        }
    }
    for (int b = 0; b < batch; ++b) {
        const auto k0 = ctx_k.begin() + static_cast<std::ptrdiff_t>(b * row_ctx);
        const auto v0 = ctx_v.begin() + static_cast<std::ptrdiff_t>(b * row_ctx);
        const std::vector<float> row_k(k0, k0 + static_cast<std::ptrdiff_t>(row_ctx));
        const std::vector<float> row_v(v0, v0 + static_cast<std::ptrdiff_t>(row_ctx));
        const auto sub = [&](const std::vector<float>& all, std::size_t count) {
            const auto begin = all.begin() + static_cast<std::ptrdiff_t>(b * count);
            return std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(count));
        };
        std::vector<double> row_expected;
        fold_oracle_row(sub(q, row_q), sub(query_k, row_kv), sub(query_v, row_kv), row_k, row_v,
                        tokens, rows[static_cast<std::size_t>(b)].length,
                        rows[static_cast<std::size_t>(b)].valid, logical_stride, window,
                        static_cast<double>(alpha), row_expected);
        std::copy(row_expected.begin(), row_expected.end(),
                  expected.begin() + static_cast<std::ptrdiff_t>(b * row_q));
    }

    std::vector<std::int32_t> lengths, valid, table_rows;
    for (int b = 0; b < batch; ++b) {
        lengths.push_back(rows[static_cast<std::size_t>(b)].length);
        valid.push_back(rows[static_cast<std::size_t>(b)].valid);
        table_rows.push_back(b);
    }
    DeviceBuffer d_q          = to_device(bf16_bits(q));
    DeviceBuffer d_query_k    = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v    = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k  = int8 ? to_device(k8.codes) : to_device(bf16_bits(physical_k));
    DeviceBuffer d_context_v  = int8 ? to_device(v8.codes) : to_device(fp16_bits(physical_v));
    DeviceBuffer d_k_scale    = to_device(int8 ? k8.scales : std::vector<__half>(1));
    DeviceBuffer d_v_scale    = to_device(int8 ? v8.scales : std::vector<__half>(1));
    DeviceBuffer d_tables     = to_device(tables);
    DeviceBuffer d_lengths    = to_device(lengths);
    DeviceBuffer d_valid      = to_device(valid);
    DeviceBuffer d_table_rows = to_device(table_rows);
    GuardedDeviceBuffer d_out(row_q * batch * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, batch});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor length_tensor(d_lengths.p, DType::I32, {batch});
    Tensor valid_tensor(d_valid.p, DType::I32, {batch});
    Tensor table_row_tensor(d_table_rows.p, DType::I32, {batch});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, batch});
    PagedKVBatchLayerView context =
        make_context_view(d_context_k, d_context_v, d_tables, logical_pages, physical_pages, batch);
    if (int8) {
        const int groups      = kD / 64;
        context.k_pages       = Tensor(d_context_k.p, DType::I8, {kD, kPage, physical_pages, kKVHeads});
        context.v_pages       = Tensor(d_context_v.p, DType::I8, {kD, kPage, physical_pages, kKVHeads});
        context.k_scale_pages = Tensor(d_k_scale.p, DType::FP16, {groups, kPage, physical_pages, kKVHeads});
        context.v_scale_pages = Tensor(d_v_scale.p, DType::FP16, {groups, kPage, physical_pages, kKVHeads});
        context.storage       = KvCacheStorage::Int8Group64;
    }
    const ops::ContextAttentionExecutionEnvelope envelope{
        0, static_cast<std::uint32_t>(max_length)};
    const std::size_t workspace_bytes = ops::context_fold_attention_workspace_capacity_bytes(
        kGeometry, envelope, window, tokens, tokens, batch);
    DeviceArena workspace(workspace_bytes);
    ops::context_fold_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                valid_tensor, table_row_tensor, kGeometry, kScale,
                                ops::ContextFold{window, alpha}, context, envelope, workspace,
                                out_tensor, nullptr);
    cuda_synchronize();

    std::string label = std::string(int8 ? "context_fold_attention int8 T=" : "context_fold_attention T=") + std::to_string(tokens) +
                        " B=" + std::to_string(batch) + " W=" + std::to_string(window) +
                        " alpha=" + std::to_string(alpha) + " L=";
    for (int b = 0; b < batch; ++b) {
        label += std::string(b ? "," : "") +
                 std::to_string(rows[static_cast<std::size_t>(b)].length) + "/V" +
                 std::to_string(rows[static_cast<std::size_t>(b)].valid);
    }
    int failures = verify_reduction(label.c_str(), from_device_bf16(d_out.data(), row_q * batch),
                                    expected, kFoldBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace high-water exceeds the queried capacity\n";
        ++failures;
    }

    if (compare_plain && !int8) {
        // L <= W for every row: FOLD is context_softmax_attention on a context whose keys are
        // pre-rotated at their positions (FOLD itself reads the unrotated copy above).
        std::vector<float> rotated_ctx_k = ctx_k;
        for (int b = 0; b < batch; ++b) {
            for (int kv = 0; kv < kKVHeads; ++kv) {
                for (int j = 0; j < logical_pages * kPage; ++j) {
                    std::vector<double> key(static_cast<std::size_t>(kD));
                    for (int d = 0; d < kD; ++d) {
                        key[static_cast<std::size_t>(d)] =
                            ctx_k[b * row_ctx + context_index(d, kv, j, logical_stride)];
                    }
                    fold_rotate(key.data(), static_cast<double>(j));
                    for (int d = 0; d < kD; ++d) {
                        rotated_ctx_k[b * row_ctx + context_index(d, kv, j, logical_stride)] =
                            static_cast<float>(key[static_cast<std::size_t>(d)]);
                    }
                }
            }
        }
        round_to_bf16(rotated_ctx_k);
        std::vector<float> rotated_physical_k = physical_k;
        for (int b = 0; b < batch; ++b) {
            std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
            for (int p = 0; p < logical_pages; ++p) {
                mapping[static_cast<std::size_t>(p)] = p * (batch + 1) + b + 1;
            }
            const auto k0 = rotated_ctx_k.begin() + static_cast<std::ptrdiff_t>(b * row_ctx);
            const std::vector<float> row_k(k0, k0 + static_cast<std::ptrdiff_t>(row_ctx));
            scatter_context_mapping(rotated_physical_k, row_k, logical_stride,
                                    logical_pages * kPage, mapping, physical_pages);
        }
        DeviceBuffer d_rotated_k = to_device(bf16_bits(rotated_physical_k));
        PagedKVBatchLayerView plain_context = make_context_view(
            d_rotated_k, d_context_v, d_tables, logical_pages, physical_pages, batch);
        GuardedDeviceBuffer d_plain(row_q * batch * sizeof(std::uint16_t));
        d_plain.fill(0x7f);
        Tensor plain_tensor(d_plain.data(), DType::BF16, {kD, kQHeads, tokens, batch});
        DeviceArena plain_workspace(ops::context_softmax_attention_workspace_capacity_bytes(
            kGeometry, envelope, tokens, tokens, batch));
        ops::context_softmax_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                       valid_tensor, table_row_tensor, kGeometry, kScale,
                                       plain_context, envelope, plain_workspace, plain_tensor, nullptr);
        cuda_synchronize();
        failures += verify_reduction((label + " vs context_softmax_attention").c_str(),
                                     from_device_bf16(d_out.data(), row_q * batch),
                                     from_device_bf16(d_plain.data(), row_q * batch),
                                     kContextQueryBf16Criterion);
    }
    return failures;
}

int run_fold_cases() {
    int failures = 0;
    // FOLD reduces to plain attention while the context fits the window.
    failures += run_fold_case(1, {{50, 1}}, 64, 2.0F, true);
    failures += run_fold_case(4, {{64, 4}}, 64, 2.0F, true);
    failures += run_fold_case(8, {{100, 8}, {37, 8}}, 128, 2.0F, true);
    failures += run_fold_case(8, {{200, 5}, {128, 8}}, 256, 2.0F, true);
    // Far memory live: L > W.
    for (const int tokens : {1, 4, 8}) {
        failures += run_fold_case(tokens, {{300, tokens}}, 64, 2.0F, false);
        failures += run_fold_case(tokens, {{400, tokens}, {250, tokens}}, 128, 2.0F, false);
    }
    failures += run_fold_case(8, {{333, 8}, {129, 3}}, 64, 2.0F, false); // inert columns, mixed L
    failures += run_fold_case(4, {{500, 4}, {70, 4}}, 64, 2.0F, false);  // one row far, one barely
    failures += run_fold_case(8, {{1000, 8}}, 128, 2.0F, false);
    failures += run_fold_case(4, {{300, 4}, {300, 4}}, 64, 2.0F, false); // two equal rows
    failures += run_fold_case(4, {{300, 4}, {40, 4}}, 64, 2.0F, false);  // a far row beside a window row
    failures += run_fold_case(4, {{40, 4}, {300, 4}}, 64, 2.0F, false);
    failures += run_fold_case(1, {{300, 1}, {300, 1}}, 64, 2.0F, false);
    failures += run_fold_case(1, {{300, 1}, {300, 1}, {300, 1}}, 64, 2.0F, false);
    failures += run_fold_case(1, {{100, 1}, {100, 1}}, 64, 2.0F, false);
    // The drafter's int8 pool: both branches dequantize before the keys turn.
    failures += run_fold_case(1, {{50, 1}}, 64, 2.0F, false, true);
    failures += run_fold_case(4, {{300, 4}, {40, 4}}, 64, 2.0F, false, true);
    failures += run_fold_case(8, {{1000, 8}, {333, 3}}, 128, 2.0F, false, true);
    // alpha = 1 sanity.
    failures += run_fold_case(4, {{300, 4}}, 64, 1.0F, false);
    failures += run_fold_case(8, {{400, 8}, {250, 8}}, 128, 1.0F, false);
    return failures;
}

} // namespace

int run_softmax_attention_context_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    int failures = 0;
    for (const Profile selected : {Profile{128, 32, 8}, Profile{64, 64, 8}}) {
        select_profile(selected);
        std::cout << "context_softmax_attention profile [" << selected.d << ","
                  << selected.q_heads << "," << selected.kv_heads << "]\n";
        failures += run_profile_cases();
    }
    select_profile({128, 32, 8});
    std::cout << "context_softmax_attention int8 G64 [128,32,8]\n";
    failures += run_int8_cases();
    std::cout << "context_fold_attention [128,32,8]\n";
    failures += run_fold_cases();
    // An unregistered geometry is refused.
    try {
        (void)ops::context_softmax_attention_workspace_capacity_bytes(
            {64, 32, 8}, ops::ContextAttentionExecutionEnvelope{0, 64}, 1, 1, 1);
        std::cerr << "context-attention accepted an unregistered geometry\n";
        ++failures;
    } catch (const std::invalid_argument&) {}

    if (failures != 0) {
        std::cerr << "context_softmax_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "context_softmax_attention: PASS\n";
    return 0;
}
