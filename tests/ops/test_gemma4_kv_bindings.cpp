// Gemma 4 KV bindings: a batched call over ring images or a paged pool computes, for every batch
// row, the same bits as the single-sequence call over that row's own contiguous slot storage.
// The single-sequence Ops are checked against the FP64 oracle in test_gemma4_attention; this test
// only proves the layouts are one computation: attention (decode and prefill routes), the K/V
// append and the fused Q/K/V preparation write and read the same rows whichever layout holds them,
// and write none past a batch row's stored count; and a verify row padded at the end of the
// context reads nothing past it.

#include "core/device.h"
#include "ninferno/ops/gemma4_attention.h"
#include "ninferno/ops/gemma4_fused.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

constexpr int kPage = 64;

struct Geometry {
    const char* name;
    int D, Hq, Hkv, window;
    bool sinks = false; // a learned sink logit per query head (gpt-oss)
};

// Points of the shipped profiles: the 26B-A4B full and sliding layers, the 31B full layer, a
// DFlash drafter full layer (also the Llama 3.1 / Qwen3 8B geometry), and gpt-oss-20b's full and
// sliding layers with their attention sinks.
constexpr Geometry kGeometries[] = {
    {"full 512/16/2", 512, 16, 2, 0},
    {"sliding 256/16/8/1024", 256, 16, 8, 1024},
    {"full 512/32/4", 512, 32, 4, 0},
    {"draft full 128/32/8", 128, 32, 8, 0},
    {"gpt-oss full 64/64/8 sinks", 64, 64, 8, 0, true},
    {"gpt-oss sliding 64/64/8/128 sinks", 64, 64, 8, 128, true},
};

std::uint16_t random_bf16(std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0F, 1.0F);
    const float value = dist(rng);
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}

std::vector<std::uint16_t> random_bits(std::size_t count, std::mt19937& rng) {
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = random_bf16(rng); }
    return out;
}

template <class T>
GuardedDeviceBuffer upload(const std::vector<T>& host) {
    GuardedDeviceBuffer buffer(std::max<std::size_t>(host.size() * sizeof(T), 4));
    buffer.copy_from_host(host.data(), host.size() * sizeof(T));
    return buffer;
}

template <class T>
std::vector<T> download(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> out(count);
    buffer.copy_to_host(out.data(), count * sizeof(T));
    return out;
}

// One batch row's history: every K/V row of positions [0, context + T) of every KV head, and
// the T queries at positions [context, context + T).
struct Row {
    int context = 0;
    std::vector<std::uint16_t> k, v; // [positions][Hkv][D]
    std::vector<std::uint16_t> q;    // [T][Hq][D]
};

// Appends positions [begin, end) of `row` to `layer` in calls of at most `chunk` tokens.
void fill(const Geometry& g, const Row& row, int begin, int end, int chunk,
          const ops::Gemma4KvLayer& layer) {
    for (int p = begin; p < end; p += chunk) {
        const int t          = std::min(chunk, end - p);
        const std::size_t at = static_cast<std::size_t>(p) * g.Hkv * g.D;
        const std::size_t n  = static_cast<std::size_t>(t) * g.Hkv * g.D;
        std::vector<std::uint16_t> k(row.k.begin() + at, row.k.begin() + at + n);
        std::vector<std::uint16_t> v(row.v.begin() + at, row.v.begin() + at + n);
        std::vector<int> positions(t);
        std::iota(positions.begin(), positions.end(), p);
        auto dk = upload(k), dv = upload(v), dp = upload(positions);
        ops::gemma4_kv_append(Tensor(dk.data(), DType::BF16, {g.D, g.Hkv, t}),
                              Tensor(dv.data(), DType::BF16, {g.D, g.Hkv, t}),
                              Tensor(dp.data(), DType::I32, {t}), layer, nullptr);
    }
}

int compare(const std::string& label, const std::vector<std::uint16_t>& got,
            const std::vector<std::uint16_t>& want) {
    if (got.size() != want.size()) {
        std::cerr << label << ": size " << got.size() << " vs " << want.size() << "\n";
        return 1;
    }
    std::size_t diff = 0;
    for (std::size_t i = 0; i < got.size(); ++i) { diff += got[i] != want[i]; }
    if (diff != 0) {
        std::cerr << label << ": " << diff << " of " << got.size() << " values differ\n";
        return 1;
    }
    return 0;
}

// Batched attention over images and over a paged pool against per-row single-sequence calls.
int run_attention(const Geometry& g, int tokens, const std::vector<int>& contexts,
                  std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int batch         = static_cast<int>(contexts.size());
    const int max_positions = *std::max_element(contexts.begin(), contexts.end()) + tokens;
    const int chunk         = 256; // history appends; also the ring's call width
    const int max_tokens    = std::max(tokens, chunk);
    // A ring holds window + widest call - 1 rows; a full layer every position.
    const int capacity = g.window > 0 ? g.window + max_tokens - 1 : max_positions;
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_positions), tokens};
    const std::size_t head_rows = static_cast<std::size_t>(g.Hkv) * g.D;

    std::vector<Row> rows(batch);
    for (int b = 0; b < batch; ++b) {
        rows[b].context = contexts[b];
        rows[b].k = random_bits(static_cast<std::size_t>(contexts[b] + tokens) * head_rows, rng);
        rows[b].v = random_bits(static_cast<std::size_t>(contexts[b] + tokens) * head_rows, rng);
        rows[b].q = random_bits(static_cast<std::size_t>(tokens) * g.Hq * g.D, rng);
    }
    std::vector<float> sink_values(g.sinks ? g.Hq : 0);
    std::normal_distribution<float> sink_dist(0.0F, 2.0F);
    for (float& value : sink_values) { value = sink_dist(rng); }
    GuardedDeviceBuffer sink_buffer = upload(sink_values);
    const Tensor sink_tensor(sink_buffer.data(), DType::FP32, {g.Hq});
    const Tensor* sinks = g.sinks ? &sink_tensor : nullptr;
    // Room for every row's decode partials: the batched call then runs its rows in one launch
    // (the per-row reference calls in one each).
    WorkspaceArena workspace(std::max<std::size_t>(
        1, ops::gemma4_attention_workspace_bytes(g.D, g.Hq, g.Hkv, envelope, g.window) *
                   static_cast<std::size_t>(batch) +
               4096));
    const std::size_t out_count = static_cast<std::size_t>(tokens) * g.Hq * g.D;

    // Reference: one sequence at a time in its own contiguous slot storage.
    std::vector<std::uint16_t> reference;
    for (int b = 0; b < batch; ++b) {
        GuardedDeviceBuffer k(static_cast<std::size_t>(capacity) * head_rows * 2);
        GuardedDeviceBuffer v(static_cast<std::size_t>(capacity) * head_rows * 2);
        const ops::Gemma4KvLayer layer{
            .k      = Tensor(k.data(), DType::BF16, {g.D, capacity, g.Hkv}),
            .v      = Tensor(v.data(), DType::BF16, {g.D, capacity, g.Hkv}),
            .window = g.window,
        };
        fill(g, rows[b], 0, contexts[b] + tokens, chunk, layer);
        std::vector<int> positions(tokens);
        std::iota(positions.begin(), positions.end(), contexts[b]);
        auto dq = upload(rows[b].q), dp = upload(positions);
        GuardedDeviceBuffer out(out_count * 2);
        Tensor o(out.data(), DType::BF16, {g.D, g.Hq, tokens});
        ops::gemma4_attention(Tensor(dq.data(), DType::BF16, {g.D, g.Hq, tokens}),
                              Tensor(dp.data(), DType::I32, {tokens}), layer, 1.0F, envelope,
                              workspace, o, nullptr,
                              ops::Gemma4AttentionMask::Causal, sinks);
        const auto got = download<std::uint16_t>(out, out_count);
        reference.insert(reference.end(), got.begin(), got.end());
    }

    std::vector<std::uint16_t> q_all;
    std::vector<int> positions_all;
    for (int b = 0; b < batch; ++b) {
        q_all.insert(q_all.end(), rows[b].q.begin(), rows[b].q.end());
        for (int t = 0; t < tokens; ++t) { positions_all.push_back(contexts[b] + t); }
    }
    auto dq = upload(q_all), dp = upload(positions_all);
    const Tensor q(dq.data(), DType::BF16, {g.D, g.Hq, tokens, batch});
    const Tensor positions(dp.data(), DType::I32, {tokens, batch});
    int failures = 0;

    // Ring images, bound in an order that is not the batch order and with one spare image.
    {
        const int images = batch + 1;
        std::vector<int> bindings(batch);
        // A rotation, so no two rows share an image (b * 2 + 1 collides for an even count).
        for (int b = 0; b < batch; ++b) { bindings[b] = (b + 2) % images; }
        GuardedDeviceBuffer k(static_cast<std::size_t>(capacity) * head_rows * images * 2);
        GuardedDeviceBuffer v(static_cast<std::size_t>(capacity) * head_rows * images * 2);
        k.fill(0xff);
        v.fill(0xff);
        const Tensor pk(k.data(), DType::BF16, {g.D, capacity, g.Hkv, images});
        const Tensor pv(v.data(), DType::BF16, {g.D, capacity, g.Hkv, images});
        for (int b = 0; b < batch; ++b) {
            auto one = upload(std::vector<int>{bindings[b]});
            fill(g, rows[b], 0, contexts[b], chunk,
                 {.k      = pk,
                  .v      = pv,
                  .window = g.window,
                  .rows   = Tensor(one.data(), DType::I32, {1})});
        }
        // The call's own rows through one batched append, then one batched attention.
        std::vector<std::uint16_t> k_new, v_new;
        for (int b = 0; b < batch; ++b) {
            const std::size_t at = static_cast<std::size_t>(contexts[b]) * head_rows;
            k_new.insert(k_new.end(), rows[b].k.begin() + at,
                         rows[b].k.begin() + at + tokens * head_rows);
            v_new.insert(v_new.end(), rows[b].v.begin() + at,
                         rows[b].v.begin() + at + tokens * head_rows);
        }
        auto dk = upload(k_new), dv = upload(v_new), drows = upload(bindings);
        const ops::Gemma4KvLayer layer{.k      = pk,
                                       .v      = pv,
                                       .window = g.window,
                                       .rows   = Tensor(drows.data(), DType::I32, {batch})};
        if (tokens <= capacity) {
            ops::gemma4_kv_append(Tensor(dk.data(), DType::BF16, {g.D, g.Hkv, tokens, batch}),
                                  Tensor(dv.data(), DType::BF16, {g.D, g.Hkv, tokens, batch}),
                                  positions, layer, nullptr);
        }
        GuardedDeviceBuffer out(out_count * batch * 2);
        Tensor o(out.data(), DType::BF16, {g.D, g.Hq, tokens, batch});
        ops::gemma4_attention(q, positions, layer, 1.0F, envelope, workspace, o, nullptr,
                              ops::Gemma4AttentionMask::Causal, sinks);
        failures += compare(std::string(g.name) + " images T=" + std::to_string(tokens),
                            download<std::uint16_t>(out, out_count * batch), reference);
        failures += out.verify_guards("images out") + k.verify_guards("images k");
    }

    // A paged pool: every row's logical pages scattered over the pool in a shuffled order, the
    // rows bound to table rows out of batch order.
    {
        const int pages_per_row = (max_positions + kPage - 1) / kPage;
        const int table_rows    = batch + 1;
        const int pages         = pages_per_row * table_rows + 3;
        std::vector<int> physical(pages);
        std::iota(physical.begin(), physical.end(), 0);
        std::shuffle(physical.begin(), physical.end(), rng);
        std::vector<int> tables(static_cast<std::size_t>(pages_per_row) * table_rows);
        for (std::size_t i = 0; i < tables.size(); ++i) { tables[i] = physical[i]; }
        std::vector<int> bindings(batch);
        for (int b = 0; b < batch; ++b) { bindings[b] = table_rows - 1 - b; }
        GuardedDeviceBuffer k(static_cast<std::size_t>(pages) * kPage * head_rows * 2);
        GuardedDeviceBuffer v(static_cast<std::size_t>(pages) * kPage * head_rows * 2);
        k.fill(0xff);
        v.fill(0xff);
        auto dtables = upload(tables);
        const Tensor pk(k.data(), DType::BF16, {g.D, kPage, pages, g.Hkv});
        const Tensor pv(v.data(), DType::BF16, {g.D, kPage, pages, g.Hkv});
        const Tensor table_tensor(dtables.data(), DType::I32, {pages_per_row, table_rows});
        for (int b = 0; b < batch; ++b) {
            auto one = upload(std::vector<int>{bindings[b]});
            fill(g, rows[b], 0, contexts[b] + tokens, chunk,
                 {.k            = pk,
                  .v            = pv,
                  .window       = g.window,
                  .block_tables = table_tensor,
                  .rows         = Tensor(one.data(), DType::I32, {1})});
        }
        auto drows = upload(bindings);
        const ops::Gemma4KvLayer layer{.k            = pk,
                                       .v            = pv,
                                       .window       = g.window,
                                       .block_tables = table_tensor,
                                       .rows         = Tensor(drows.data(), DType::I32, {batch})};
        GuardedDeviceBuffer out(out_count * batch * 2);
        Tensor o(out.data(), DType::BF16, {g.D, g.Hq, tokens, batch});
        ops::gemma4_attention(q, positions, layer, 1.0F, envelope, workspace, o, nullptr,
                              ops::Gemma4AttentionMask::Causal, sinks);
        failures += compare(std::string(g.name) + " paged T=" + std::to_string(tokens),
                            download<std::uint16_t>(out, out_count * batch), reference);
        failures += out.verify_guards("paged out") + k.verify_guards("paged k");
    }
    return failures;
}

// The fused preparation writes each column's K/V where gemma4_kv_append would: into its batch
// row's page through the table.
int run_prepare(std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int D = 512, Hq = 16, Hkv = 2, batch = 3, tokens = 2;
    const int pages_per_row = 4, table_rows = 3, pages = 14;
    std::vector<int> physical(pages);
    std::iota(physical.begin(), physical.end(), 0);
    std::shuffle(physical.begin(), physical.end(), rng);
    std::vector<int> tables(physical.begin(), physical.begin() + pages_per_row * table_rows);
    const std::vector<int> bindings  = {2, 0, 1};
    const std::vector<int> positions = {70, 71, 3, 4, 200, 201}; // [T, B]
    const int columns                = tokens * batch;
    auto q_src                       = random_bits(static_cast<std::size_t>(Hq) * D * columns, rng);
    auto k_src = random_bits(static_cast<std::size_t>(Hkv) * D * columns, rng);
    auto v_src = random_bits(static_cast<std::size_t>(Hkv) * D * columns, rng);
    auto q_w = random_bits(D, rng), k_w = random_bits(D, rng);
    auto dq = upload(q_src), dk = upload(k_src), dv = upload(v_src), dqw = upload(q_w),
         dkw = upload(k_w), dp = upload(positions), dt = upload(tables), dr = upload(bindings);
    const Tensor tq(dq.data(), DType::BF16, {Hq * D, columns});
    const Tensor tk(dk.data(), DType::BF16, {Hkv * D, columns});
    const Tensor tv(dv.data(), DType::BF16, {Hkv * D, columns});
    const Tensor tqw(dqw.data(), DType::BF16, {D});
    const Tensor tkw(dkw.data(), DType::BF16, {D});
    const Tensor tp(dp.data(), DType::I32, {columns});

    // Reference: the unfused outputs [D, heads, columns].
    GuardedDeviceBuffer q_ref(static_cast<std::size_t>(Hq) * D * columns * 2);
    GuardedDeviceBuffer k_ref(static_cast<std::size_t>(Hkv) * D * columns * 2);
    GuardedDeviceBuffer v_ref(static_cast<std::size_t>(Hkv) * D * columns * 2);
    Tensor q_out(q_ref.data(), DType::BF16, {D, Hq, columns});
    Tensor k_out(k_ref.data(), DType::BF16, {D, Hkv, columns});
    Tensor v_out(v_ref.data(), DType::BF16, {D, Hkv, columns});
    ops::gemma4_qkv_prepare(tq, tk, tv, D, tqw, tkw, 1e-6F, tp, 1e6F, D / 8, q_out, k_out, v_out,
                            nullptr);

    GuardedDeviceBuffer pool_k(static_cast<std::size_t>(pages) * kPage * Hkv * D * 2);
    GuardedDeviceBuffer pool_v(static_cast<std::size_t>(pages) * kPage * Hkv * D * 2);
    pool_k.fill(0);
    pool_v.fill(0);
    GuardedDeviceBuffer q_fused(static_cast<std::size_t>(Hq) * D * columns * 2);
    Tensor q_fused_out(q_fused.data(), DType::BF16, {D, Hq, columns});
    const ops::Gemma4KvLayer layer{
        .k            = Tensor(pool_k.data(), DType::BF16, {D, kPage, pages, Hkv}),
        .v            = Tensor(pool_v.data(), DType::BF16, {D, kPage, pages, Hkv}),
        .window       = 0,
        .block_tables = Tensor(dt.data(), DType::I32, {pages_per_row, table_rows}),
        .rows         = Tensor(dr.data(), DType::I32, {batch}),
    };
    ops::gemma4_qkv_prepare(tq, tk, tv, D, tqw, tkw, 1e-6F, tp, 1e6F, D / 8, q_fused_out, layer,
                            nullptr);

    int failures  = compare("prepare q", download<std::uint16_t>(q_fused, q_ref.bytes() / 2),
                            download<std::uint16_t>(q_ref, q_ref.bytes() / 2));
    const auto kr = download<std::uint16_t>(k_ref, k_ref.bytes() / 2);
    const auto vr = download<std::uint16_t>(v_ref, v_ref.bytes() / 2);
    const auto pk = download<std::uint16_t>(pool_k, pool_k.bytes() / 2);
    const auto pv = download<std::uint16_t>(pool_v, pool_v.bytes() / 2);
    std::vector<std::uint16_t> got_k, got_v, want_k, want_v;
    const std::size_t head_plane = static_cast<std::size_t>(pages) * kPage * D;
    for (int c = 0; c < columns; ++c) {
        const int b = c / tokens;
        const int p = positions[c];
        const int slot =
            tables[static_cast<std::size_t>(bindings[b]) * pages_per_row + p / kPage] * kPage +
            p % kPage;
        for (int h = 0; h < Hkv; ++h) {
            const std::size_t at   = h * head_plane + static_cast<std::size_t>(slot) * D;
            const std::size_t from = (static_cast<std::size_t>(c) * Hkv + h) * D;
            got_k.insert(got_k.end(), pk.begin() + at, pk.begin() + at + D);
            got_v.insert(got_v.end(), pv.begin() + at, pv.begin() + at + D);
            want_k.insert(want_k.end(), kr.begin() + from, kr.begin() + from + D);
            want_v.insert(want_v.end(), vr.begin() + from, vr.begin() + from + D);
        }
    }
    failures += compare("prepare paged k", got_k, want_k);
    failures += compare("prepare paged v", got_v, want_v);
    // Nothing else in the pool was written.
    std::size_t written = 0;
    for (const auto x : pk) { written += x != 0; }
    std::size_t expected = 0;
    for (const auto x : want_k) { expected += x != 0; }
    if (written != expected) {
        std::cerr << "prepare paged: " << written << " nonzero pool values, expected " << expected
                  << "\n";
        ++failures;
    }
    return failures;
}

// A verify call pads a row with fewer drafts than the launch's width by repeating its last
// position. With Gemma4KvLayer::stored, only a row's first stored[b] tokens write: the fused
// preparation and the append leave the padding's bytes out of the cache, both where the padding
// repeats the last real position (row 0: it would race that token for its slot) and where it
// does not (row 1: its own slots stay zero).
int run_stored(std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int D = 512, Hq = 16, Hkv = 2, batch = 2, tokens = 4;
    const int pages_per_row = 4, table_rows = 2, pages = 9;
    std::vector<int> physical(pages);
    std::iota(physical.begin(), physical.end(), 0);
    std::shuffle(physical.begin(), physical.end(), rng);
    std::vector<int> tables(physical.begin(), physical.begin() + pages_per_row * table_rows);
    const std::vector<int> bindings  = {1, 0};
    const std::vector<int> positions = {70, 71, 71, 71, 3, 4, 5, 6}; // row-major [B][T]
    const std::vector<int> stored    = {2, 1};
    const int columns                = tokens * batch;
    auto q_src                       = random_bits(static_cast<std::size_t>(Hq) * D * columns, rng);
    auto k_src = random_bits(static_cast<std::size_t>(Hkv) * D * columns, rng);
    auto v_src = random_bits(static_cast<std::size_t>(Hkv) * D * columns, rng);
    auto q_w = random_bits(D, rng), k_w = random_bits(D, rng);
    auto dq = upload(q_src), dk = upload(k_src), dv = upload(v_src), dqw = upload(q_w),
         dkw = upload(k_w), dp = upload(positions), dt = upload(tables), dr = upload(bindings),
         ds = upload(stored);
    const Tensor tq(dq.data(), DType::BF16, {Hq * D, columns});
    const Tensor tk(dk.data(), DType::BF16, {Hkv * D, columns});
    const Tensor tv(dv.data(), DType::BF16, {Hkv * D, columns});
    const Tensor tqw(dqw.data(), DType::BF16, {D});
    const Tensor tkw(dkw.data(), DType::BF16, {D});
    const Tensor tp(dp.data(), DType::I32, {columns});

    GuardedDeviceBuffer q_ref(static_cast<std::size_t>(Hq) * D * columns * 2);
    GuardedDeviceBuffer k_ref(static_cast<std::size_t>(Hkv) * D * columns * 2);
    GuardedDeviceBuffer v_ref(static_cast<std::size_t>(Hkv) * D * columns * 2);
    Tensor q_out(q_ref.data(), DType::BF16, {D, Hq, columns});
    Tensor k_out(k_ref.data(), DType::BF16, {D, Hkv, columns});
    Tensor v_out(v_ref.data(), DType::BF16, {D, Hkv, columns});
    ops::gemma4_qkv_prepare(tq, tk, tv, D, tqw, tkw, 1e-6F, tp, 1e6F, D / 8, q_out, k_out, v_out,
                            nullptr);
    const auto kr = download<std::uint16_t>(k_ref, k_ref.bytes() / 2);
    const auto vr = download<std::uint16_t>(v_ref, v_ref.bytes() / 2);

    int failures = 0;
    for (const bool fused : {true, false}) {
        const std::string label = fused ? "stored prepare" : "stored append";
        GuardedDeviceBuffer pool_k(static_cast<std::size_t>(pages) * kPage * Hkv * D * 2);
        GuardedDeviceBuffer pool_v(static_cast<std::size_t>(pages) * kPage * Hkv * D * 2);
        pool_k.fill(0);
        pool_v.fill(0);
        const ops::Gemma4KvLayer layer{
            .k            = Tensor(pool_k.data(), DType::BF16, {D, kPage, pages, Hkv}),
            .v            = Tensor(pool_v.data(), DType::BF16, {D, kPage, pages, Hkv}),
            .window       = 0,
            .block_tables = Tensor(dt.data(), DType::I32, {pages_per_row, table_rows}),
            .rows         = Tensor(dr.data(), DType::I32, {batch}),
            .stored       = Tensor(ds.data(), DType::I32, {batch}),
        };
        if (fused) {
            GuardedDeviceBuffer q_fused(static_cast<std::size_t>(Hq) * D * columns * 2);
            Tensor q_fused_out(q_fused.data(), DType::BF16, {D, Hq, columns});
            ops::gemma4_qkv_prepare(tq, tk, tv, D, tqw, tkw, 1e-6F, tp, 1e6F, D / 8, q_fused_out,
                                    layer, nullptr);
            // Every column's queries are prepared; only the cache writes follow `stored`.
            failures +=
                compare(label + " q", download<std::uint16_t>(q_fused, q_ref.bytes() / 2),
                        download<std::uint16_t>(q_ref, q_ref.bytes() / 2));
        } else {
            ops::gemma4_kv_append(Tensor(k_ref.data(), DType::BF16, {D, Hkv, tokens, batch}),
                                  Tensor(v_ref.data(), DType::BF16, {D, Hkv, tokens, batch}),
                                  Tensor(dp.data(), DType::I32, {tokens, batch}), layer, nullptr);
        }
        const auto pk = download<std::uint16_t>(pool_k, pool_k.bytes() / 2);
        const auto pv = download<std::uint16_t>(pool_v, pool_v.bytes() / 2);
        std::vector<std::uint16_t> got_k, got_v, want_k, want_v;
        const std::size_t head_plane = static_cast<std::size_t>(pages) * kPage * D;
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < stored[b]; ++j) {
                const int c = b * tokens + j;
                const int p = positions[c];
                const int slot =
                    tables[static_cast<std::size_t>(bindings[b]) * pages_per_row + p / kPage] *
                        kPage +
                    p % kPage;
                for (int h = 0; h < Hkv; ++h) {
                    const std::size_t at   = h * head_plane + static_cast<std::size_t>(slot) * D;
                    const std::size_t from = (static_cast<std::size_t>(c) * Hkv + h) * D;
                    got_k.insert(got_k.end(), pk.begin() + at, pk.begin() + at + D);
                    got_v.insert(got_v.end(), pv.begin() + at, pv.begin() + at + D);
                    want_k.insert(want_k.end(), kr.begin() + from, kr.begin() + from + D);
                    want_v.insert(want_v.end(), vr.begin() + from, vr.begin() + from + D);
                }
            }
        }
        failures += compare(label + " k", got_k, want_k);
        failures += compare(label + " v", got_v, want_v);
        std::size_t written = 0, expected = 0;
        for (const auto x : pk) { written += x != 0; }
        for (const auto x : want_k) { expected += x != 0; }
        if (written != expected) {
            std::cerr << label << ": " << written << " nonzero pool values, expected " << expected
                      << "\n";
            ++failures;
        }
        failures += pool_k.verify_guards(label + " pool k") + pool_v.verify_guards(label + " pool v");
    }
    return failures;
}

// A verify call at the end of the context (Gemma 4 26B MTP read a garbage page there): a row with
// fewer drafts than the call's width repeats its last real position, the binding's last slot. The
// table row after the sequence's holds page numbers far outside the pool, so a read past the
// capacity faults. The real columns must be the bits the same call computes with a page of
// headroom, and a padding column is the last real column again: here its query is a copy of that
// column's (a verify pads with the anchor's), so the two must agree bitwise. Under the invariant
// route profile (every call on the prefill kernel) a padding row folds its tiles on its own row
// block, so it is only finite there.
int run_padded_tail(const Geometry& g, int tokens, int real, bool invariant, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int capacity          = 2048;
    const int first             = capacity - real; // the real columns end on the last slot
    const std::size_t head_rows = static_cast<std::size_t>(g.Hkv) * g.D;
    const std::size_t column_q  = static_cast<std::size_t>(g.Hq) * g.D;
    Row row;
    row.context = first;
    row.k       = random_bits(static_cast<std::size_t>(capacity) * head_rows, rng);
    row.v       = random_bits(static_cast<std::size_t>(capacity) * head_rows, rng);
    row.q       = random_bits(static_cast<std::size_t>(real) * column_q, rng);
    for (int t = real; t < tokens; ++t) {
        row.q.insert(row.q.end(), row.q.begin() + (real - 1) * column_q,
                     row.q.begin() + real * column_q);
    }
    std::vector<int> positions(tokens);
    for (int t = 0; t < tokens; ++t) { positions[t] = first + std::min(t, real - 1); }
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(capacity), tokens};
    const std::size_t out_count = static_cast<std::size_t>(tokens) * column_q;
    auto dq = upload(row.q), dp = upload(positions), drow = upload(std::vector<int>{0});
    std::optional<ops::ScopedInvariantRoutes> profile;
    if (invariant) { profile.emplace(); }

    // headroom 0: the sequence fills its table row; 1: one more page, never written (NaN).
    std::vector<std::uint16_t> outputs[2];
    for (const int headroom : {0, 1}) {
        const int pages_per_row = capacity / kPage + headroom;
        const int pages         = pages_per_row + 1;
        // The next row's entries: page numbers far outside the pool, so a read through them faults.
        std::vector<int> tables(static_cast<std::size_t>(pages_per_row) * 2, 1 << 24);
        std::iota(tables.begin(), tables.begin() + pages_per_row, 0);
        std::shuffle(tables.begin(), tables.begin() + pages_per_row, rng);
        GuardedDeviceBuffer k(static_cast<std::size_t>(pages) * kPage * head_rows * 2);
        GuardedDeviceBuffer v(static_cast<std::size_t>(pages) * kPage * head_rows * 2);
        k.fill(0xff);
        v.fill(0xff);
        auto dtables = upload(tables);
        const ops::Gemma4KvLayer layer{
            .k            = Tensor(k.data(), DType::BF16, {g.D, kPage, pages, g.Hkv}),
            .v            = Tensor(v.data(), DType::BF16, {g.D, kPage, pages, g.Hkv}),
            .window       = g.window,
            .block_tables = Tensor(dtables.data(), DType::I32, {pages_per_row, 2}),
            .rows         = Tensor(drow.data(), DType::I32, {1}),
        };
        fill(g, row, 0, capacity, 256, layer);
        WorkspaceArena workspace(std::max<std::size_t>(
            1, ops::gemma4_attention_workspace_bytes(g.D, g.Hq, g.Hkv, envelope, g.window)));
        GuardedDeviceBuffer out(out_count * 2);
        Tensor o(out.data(), DType::BF16, {g.D, g.Hq, tokens});
        ops::gemma4_attention(Tensor(dq.data(), DType::BF16, {g.D, g.Hq, tokens}),
                              Tensor(dp.data(), DType::I32, {tokens}), layer, 1.0F, envelope,
                              workspace, o, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        outputs[headroom] = download<std::uint16_t>(out, out_count);
    }

    const std::string label = std::string(g.name) + " padded tail T=" + std::to_string(tokens) +
                              " real=" + std::to_string(real) + (invariant ? " invariant" : "");
    int failures      = compare(label + " vs headroom", outputs[0], outputs[1]);
    const auto column = [&](int t) {
        return std::vector<std::uint16_t>(outputs[0].begin() + t * column_q,
                                          outputs[0].begin() + (t + 1) * column_q);
    };
    for (int t = real; t < tokens; ++t) {
        if (!invariant) {
            failures +=
                compare(label + " padding " + std::to_string(t), column(t), column(real - 1));
            continue;
        }
        // The prefill kernel folds a padding row's tiles on its own row block, so only finite.
        const auto bits       = column(t);
        const auto non_finite = std::count_if(
            bits.begin(), bits.end(), [](std::uint16_t x) { return (x & 0x7f80) == 0x7f80; });
        if (non_finite != 0) {
            std::cerr << label << " padding " << t << ": " << non_finite << " non-finite values\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        std::uint32_t seed = 11;
        for (const Geometry& g : kGeometries) {
            // Decode (1), verify (16) and prefill (17, 100) widths, contexts that start on and off
            // page boundaries; a sliding layer's longest context wraps its ring.
            for (const int tokens : {1, 16, 17, 100}) {
                failures += run_attention(g, tokens, {700, 64, 1500 + tokens % 7}, seed++);
            }
        }
        failures += run_prepare(seed++);
        failures += run_stored(seed++);
        // MTP verify widths at the end of the context: depth 4 with one draft, depth 1 with none.
        for (const Geometry& g : kGeometries) {
            if (g.window != 0) { continue; } // a sliding layer's ring has no table to run off
            for (const bool invariant : {false, true}) {
                failures += run_padded_tail(g, 5, 2, invariant, seed++);
                failures += run_padded_tail(g, 2, 1, invariant, seed++);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " gemma4 kv bindings (" << failures
              << " failures)\n";
    return failures == 0 ? 0 : 1;
}
