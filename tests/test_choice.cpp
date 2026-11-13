// The configuration choice (src/product/choice/choice.h):
//   - the speed model reproduces every stand measurement its constants were fitted on
//     (tests/fixtures/choice/speed_measured.json) within the fit's error;
//   - the choice keeps the quant rule (the largest that fits, the floor bits), keeps the pins, picks
//     the more accurate of two configurations that cost about the same (INT8 over FP8 KV), the much
//     faster one where the accurate one is much slower (fast over exact), and the cheaper KV read
//     where a codec reads slowly (INT8 over BF16 KV on Turing);
//   - nothing fitting gives no choice, and explain names every alternative with why it lost.
// CPU.

#include "product/choice/choice.h"
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/measured_boards.h"
#include "core/generation.h"
#include "core/mmq_columns.h"
#include "product/choice/choice_replay.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#ifndef NINFERNO_SOURCE_DIR
#    error "NINFERNO_SOURCE_DIR must name the source tree"
#endif

namespace {

// The kernel's column tile picks (core/mmq_columns.h): GA10x, Ada and GB20x take the least work of
// their widths, Turing the cheapest launch of the same widths (its launch law, DECISIONS 78), every
// other card the fewest tiles of 16, 32, 64, 128. Never wider than the fixup plane's bound
// (ggml_mmq fixup_bound: the column count rounded up to a power of two from 16, at most 128).
namespace pins {
using ninferno::mmq_columns::pick;
// A 4096 x 4096 product on the RTX 2080 Ti's facts (68 SMs, the 1545 MHz reference clock, 352 bits
// at 14 Gbps): what the launch law reads; the tile law's generations do not read it.
constexpr ninferno::mmq_columns::LaunchFacts k4096{4096, 4096, 68, 1545e6, 616e9};
constexpr ninferno::mmq_columns::LaunchFacts k12288{12288, 4096, 68, 1545e6, 616e9};
static_assert(pick(144, 8, 6, true, 128, k4096) == 80 && pick(160, 8, 6, true, 128, k4096) == 80 &&
              pick(192, 8, 6, true, 128, k4096) == 96 && pick(224, 8, 6, true, 128, k4096) == 128 &&
              pick(256, 8, 6, true, 128, k4096) == 128 && pick(16, 8, 6, true, 128, k4096) == 16 &&
              pick(8, 8, 6, true, 128, k4096) == 16 && pick(1, 8, 6, true, 128, k4096) == 16 &&
              pick(320, 8, 6, true, 128, k4096) == 80);
static_assert(pick(144, 12, 0, true, 128, k4096) == 80 &&
              pick(192, 12, 0, true, 128, k4096) == 96 &&
              pick(224, 12, 0, true, 128, k4096) == 80 &&
              pick(256, 12, 0, true, 128, k4096) == 128 && pick(16, 12, 0, true, 128, k4096) == 16);
// Ada takes the least work too (f 0.20, as GA10x).
static_assert(pick(144, 8, 9, true, 128, k4096) == 80 && pick(192, 8, 9, true, 128, k4096) == 96 &&
              pick(224, 8, 9, true, 128, k4096) == 128 && pick(16, 8, 9, true, 128, k4096) == 16);
// Turing takes the cheapest launch (DECISIONS 78): at 105 and 121 columns two 64-column tiles (64
// CTAs, a tile a CTA on 68 SMs) where one 128-column tile runs stream-k with a fixup; then 80, 96
// and 128 as their two tiles cover the columns, 80 and 96 again at four tiles of 313 and 377
// columns (128 CTAs, 94 % of two waves, where 128's three tiles, 96 CTAs, run stream-k).
static_assert(pick(105, 7, 5, true, 128, k4096) == 64 && pick(121, 7, 5, true, 128, k4096) == 64 &&
              pick(137, 7, 5, true, 128, k4096) == 80 && pick(153, 7, 5, true, 128, k4096) == 80 &&
              pick(169, 7, 5, true, 128, k4096) == 96 && pick(185, 7, 5, true, 128, k4096) == 96 &&
              pick(201, 7, 5, true, 128, k4096) == 128 &&
              pick(249, 7, 5, true, 128, k4096) == 128 && pick(313, 7, 5, true, 128, k4096) == 80 &&
              pick(377, 7, 5, true, 128, k4096) == 96);
// The 12288-row linears the same at 121 and 201 columns (192 CTAs, 94 % of three waves, against 96
// stream-k ones); at 441 columns 64 (seven tiles, 672 CTAs in ten waves, 99 %).
static_assert(pick(121, 7, 5, true, 128, k12288) == 64 &&
              pick(201, 7, 5, true, 128, k12288) == 128 &&
              pick(441, 7, 5, true, 128, k12288) == 64);
static_assert(pick(121, 7, 5, false, 128, k4096) == 128);
// A Turing board whose facts give no roofline keeps the fewest tiles of 16, 32, 64, 128.
static_assert(pick(70, 7, 5, true, 128, {}) == 128 && pick(121, 7, 5, true, 128, {}) == 128 &&
              pick(150, 7, 5, true, 128, {}) == 128 && pick(40, 7, 5, true, 128, {}) == 64);
// A format the grids did not run keeps the fewest tiles on every card
// (mmq_columns::work_tiles_format).
static_assert(pick(144, 8, 6, false, 128, k4096) == 128 &&
              pick(192, 12, 0, false, 128, k4096) == 128 &&
              pick(16, 8, 6, false, 128, k4096) == 16);
static_assert(ninferno::mmq_columns::work_tiles_format(ninferno::QType::GGML_Q4_K) &&
              ninferno::mmq_columns::work_tiles_format(ninferno::QType::GGML_Q6_K) &&
              !ninferno::mmq_columns::work_tiles_format(ninferno::QType::GGML_Q4_0) &&
              !ninferno::mmq_columns::work_tiles_format(ninferno::QType::GGML_IQ4_XS));

// A tile's shared memory (ggml's mmq_get_nbytes_shared): Q2_K's 128-column tile is 70 144 bytes,
// over Turing's 64 KiB (its widest there is 64) and under Volta's 96 KiB and Ampere's; Q4_K's
// (stride 76 words) is 57 856 and fits.
static_assert(ninferno::mmq_columns::tile_shared_bytes(ninferno::mmq_columns::SramLayout::Q2_K,
                                                       128) == 70144);
static_assert(ninferno::mmq_columns::tile_shared_bytes(ninferno::mmq_columns::SramLayout::Q8_1,
                                                       128) == 128 * 76 * 4 + 512 + 18432);
static_assert(ninferno::mmq_columns::widest_tile(ninferno::QType::GGML_Q2_K, 7, 5) == 64);
static_assert(ninferno::mmq_columns::widest_tile(ninferno::QType::GGML_Q4_K, 7, 5) == 128);
static_assert(ninferno::mmq_columns::widest_tile(ninferno::QType::GGML_Q2_K, 8, 6) == 128);
static_assert(ninferno::mmq_columns::widest_tile(ninferno::QType::GGML_Q2_K, 7, 0) == 128);
// cost_route's MMQ column count (min(128, widest, 8-column steps)) for Q2_K at 100 tokens on sm_75
// is 64: two 64-column tiles, as the kernel runs.
static_assert(std::min(128, ninferno::mmq_columns::widest_tile(ninferno::QType::GGML_Q2_K, 7, 5)) ==
              64);
// A Turing Q2_K product of 65..128 columns takes two 64-column tiles, not one of 128.
static_assert(pick(100, 7, 5, false,
                   ninferno::mmq_columns::widest_tile(ninferno::QType::GGML_Q2_K, 7, 5),
                   k4096) == 64);

constexpr bool within_fixup_bound() {
    for (int c = 1; c <= 512; ++c) {
        int bound = 16;
        while (bound < c && bound < 128) bound *= 2;
        for (const int cc : {60, 70, 75, 80, 86, 89, 90, 120}) {
            if (pick(c, cc / 10, cc % 10, true, 128, k4096) > bound) return false;
        }
    }
    return true;
}

static_assert(within_fixup_bound());
} // namespace pins

namespace ch = ninferno::product::choice;
using ninferno::ActivationMode;
using ninferno::KvCacheStorage;
using ninferno::RecurrentStateStorage;
using nlohmann::json;

// A fixture card by its PCI id (#612 step 9): the compute capability, SMs and DRAM rate the
// measured data keyed by the board reads (core/measured_boards.h board_of_pci: the registry row)
// and the row's reference clock (its published boost), the fields these fixtures price with. The L2
// and FP4 stay unset, as the fixtures always had them.
ch::CardFacts board_card(std::uint32_t pci_device_id) {
    const ninferno::Board b                       = ninferno::measured::board_of_pci(pci_device_id);
    const ninferno::board_registry::BoardRow* row = ninferno::board_registry::find(pci_device_id);
    return {.major            = b.major,
            .minor            = b.minor,
            .sms              = b.sms,
            .clock_hz         = row->boost_mhz * 1e6,
            .dram_bytes_per_s = b.dram_gbps * 1e9};
}

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

KvCacheStorage kv_of(const std::string& name) {
    return *ninferno::parse_kv_cache_storage(name);
}

ch::ModelFacts model_of(const json& m) {
    ch::ModelFacts out;
    out.family               = m.at("family").get<std::string>();
    out.params               = m.at("params").get<double>();
    out.active_params        = m.at("active_params").get<double>();
    out.experts              = m.at("experts").get<int>();
    out.experts_per_token    = m.at("experts_per_token").get<int>();
    out.routed_expert_params = m.at("routed_expert_params").get<double>();
    out.input_table_params   = m.at("input_table_params").get<double>();
    const json kv            = m.at("kv_bytes_per_token");
    out.kv_bytes_per_token   = [kv](KvCacheStorage s) {
        return kv.at(std::string(ninferno::kv_cache_storage_name(s))).get<std::uint64_t>();
    };
    const json st   = m.at("state_bytes");
    out.state_bytes = [st](RecurrentStateStorage s) {
        return st.value(std::string(ninferno::recurrent_state_storage_name(s)),
                        st.at("f16").get<std::uint64_t>());
    };
    out.recurrent               = m.value("recurrent", true);
    out.full_attention_width    = m.value("full_attention_width", 0.0);
    out.sliding_attention_width = m.value("sliding_attention_width", 0.0);
    out.layers                  = m.value("layers", 0U);
    out.hidden                  = m.value("hidden", 0.0);
    out.full_layers             = m.value("full_layers", 0U);
    out.full_head_dim           = m.value("full_head_dim", 0.0);
    out.full_kv_heads           = m.value("full_kv_heads", 0.0);
    out.sliding_layers          = m.value("sliding_layers", 0U);
    out.sliding_head_dim        = m.value("sliding_head_dim", 0.0);
    out.sliding_kv_heads        = m.value("sliding_kv_heads", 0.0);
    out.recurrent_state_elements = m.value("recurrent_state_elements", 0.0);
    if (m.contains("linears")) {
        for (const json& l : m.at("linears")) {
            out.linears.push_back({l.at(0).get<double>(), l.at(1).get<double>(), l.at(2).get<double>(), false,
                                   l.size() > 3 && l.at(3).get<int>() != 0});
        }
    }
    if (m.contains("ring_bytes_per_token")) {
        const json ring         = m.at("ring_bytes_per_token");
        out.ring_bytes_per_token = [ring](KvCacheStorage s) {
            return ring.at(std::string(ninferno::kv_cache_storage_name(s))).get<std::uint64_t>();
        };
        out.window = m.at("window").get<std::uint32_t>();
    }
    if (!out.recurrent) { out.state_bytes = nullptr; }
    return out;
}

// The model at one quantization: its input table and speculation-only weights as stored.
ch::ModelFacts model_at(const json& m, const std::string& quant) {
    ch::ModelFacts out      = model_of(m);
    const json& q           = m.at("quants").at(quant);
    out.input_table_bytes   = q.value("input_table_bytes", 0.0);
    // Each linear a format by its role's shares, as plan_facts.cpp splits a GGUF's.
    if (q.contains("role_formats") && m.contains("linear_roles")) {
        std::vector<ch::LinearShape> split;
        for (std::size_t i = 0; i < out.linears.size(); ++i) {
            const json& shares = q.at("role_formats").at(m.at("linear_roles").at(i).get<std::string>());
            for (const auto& [format, share] : shares.items()) {
                ch::LinearShape part = out.linears[i];
                part.per_token *= share.get<double>();
                part.format = format;
                split.push_back(part);
            }
        }
        out.linears = std::move(split);
    }
    return out;
}

ch::CardFacts card_of(const json& c) {
    ch::CardFacts out;
    out.major            = c.at("major").get<int>();
    out.minor            = c.at("minor").get<int>();
    out.sms              = c.at("sms").get<int>();
    out.clock_hz         = c.at("clock_hz").get<double>();
    out.dram_bytes_per_s = c.at("dram_bytes_per_s").get<double>();
    out.fp4              = c.at("fp4").get<bool>();
    return out;
}

ch::QuantOption quant_of(const std::string& label, const json& q) {
    ch::QuantOption out;
    out.label           = label;
    out.bits_per_weight = q.at("bits_per_weight").get<double>();
    // What a decode step reads (product::quant_option_of): the file less its MTP layers and drafter.
    out.weight_bytes    = q.at("weight_bytes").get<double>() - q.value("speculation_bytes", 0.0);
    out.weights         = ch::label_class(label);
    // An MoE's routed banks as its GGUF's tensor table stores them (ffn_*_exps).
    out.routed_expert_bytes = q.value("routed_expert_bytes", 0.0);
    return out;
}

// The 2026-10-23 points: an MoE's decode from facts at one stream and on the 2026-10-30 step rows
// (DECISIONS 65); a dense model's one-stream decode from facts (its several-stream rows are wave
// aggregates, not a step's rate: decodes_from_facts measures those) and its steps at C rows (mbatch.sh); prefill (the 22K points, at their prompt,
// attention over it) within the refit's (tools/fits/prefill/fit_prefill.py).
void reproduces_the_measurements(const json& fixture) {
    double decode_sq = 0, prefill_sq = 0, decode_worst = 0, prefill_worst = 0, moe_sq = 0, moe_worst = 0;
    int decodes = 0, prefills = 0, moes = 0;
    double dense_sq = 0, dense_worst = 0;
    int denses = 0;
    double big_sq = 0, big_worst = 0;
    int bigs = 0;
    for (const json& row : fixture.at("rows")) {
        const json& m           = fixture.at("models").at(row.at("model").get<std::string>());
        const std::string q     = row.at("quant").get<std::string>();
        const ch::ModelFacts mf = model_at(m, q);
        const ch::CardFacts cf  = card_of(fixture.at("cards").at(row.at("card").get<std::string>()));
        const ch::QuantOption qo = quant_of(q, m.at("quants").at(q));
        ch::Candidate c;
        c.kv          = kv_of(row.at("kv").get<std::string>());
        c.ring        = kv_of(row.value("ring", std::string("bf16")));
        c.state       = row.at("state").get<std::string>() == "f32" ? RecurrentStateStorage::FP32
                                                                    : RecurrentStateStorage::FP16;
        c.activations = row.at("activations").get<std::string>() == "exact" ? ActivationMode::Exact
                                                                           : ActivationMode::Fast;
        const ch::Speed s = ch::speed_of(mf, cf, qo, c, row.at("concurrency").get<std::uint32_t>(),
                                         row.at("context").get<double>(), row.value("prompt", 0U));
        const double d    = s.decode_per_stream / row.at("decode").get<double>() - 1;
        // A step's rate: one stream, or a row measured as the step itself ("step": the decode of C
        // concurrent requests apart from their prompts); the other several-stream rows are wave
        // aggregates.
        const bool step = row.at("concurrency").get<std::uint32_t>() == 1 || row.value("step", false);
        if (row.value("long", false)) {
            // The big models' steps at an ~18K-token context (mbatchl.sh, 2026-10-31): their own guard.
            big_sq += d * d, big_worst = std::max(big_worst, std::abs(d)), ++bigs;
            std::cout << "  long step " << row.at("card").get<std::string>() << ' ' << row.at("model").get<std::string>()
                      << ' ' << q << " x" << row.at("concurrency").get<int>() << ": " << std::lround(d * 100) << " %\n";
        } else if (mf.moe() && step) {
            moe_sq += d * d;
            moe_worst = std::max(moe_worst, std::abs(d));
            ++moes;
        } else if (!mf.moe() && row.value("step", false)) {
            // A dense model's step at C rows (mbatch.sh, 2026-10-31: Qwen3.5 9B's and Gemma 4 12B's
            // quantizations on seven cards).
            dense_sq += d * d;
            dense_worst = std::max(dense_worst, std::abs(d));
            ++denses;
            std::cout << "  step " << row.at("card").get<std::string>() << ' ' << row.at("model").get<std::string>() << ' '
                      << q << " x" << row.at("concurrency").get<int>() << ": " << std::lround(d * 100) << " %\n";
        } else if (!mf.moe() && row.at("concurrency").get<std::uint32_t>() == 1) {
            decode_sq += d * d;
            decode_worst = std::max(decode_worst, std::abs(d));
            ++decodes;
        }
        if (!row.at("prefill").is_null() && row.at("kv").get<std::string>() == "int8") {
            const double p = s.prefill / row.at("prefill").get<double>() - 1;
            prefill_sq += p * p;
            prefill_worst = std::max(prefill_worst, std::abs(p));
            ++prefills;
        }
    }
    const double decode_rms  = std::sqrt(decode_sq / decodes);
    const double prefill_rms = std::sqrt(prefill_sq / prefills);
    std::cout << "decode " << decodes << " points: rms " << decode_rms * 100 << " %, worst "
              << decode_worst * 100 << " %; prefill " << prefills << ": rms " << prefill_rms * 100
              << " %, worst " << prefill_worst * 100 << " %\n";
    const double moe_rms = std::sqrt(moe_sq / moes);
    std::cout << "MoE decode " << moes << " points: rms " << moe_rms * 100 << " %, worst " << moe_worst * 100 << " %\n";
    std::cout << "dense step " << denses << " points: rms " << std::sqrt(dense_sq / std::max(denses, 1)) * 100
              << " %, worst " << dense_worst * 100 << " %\n";
    // 200 steps (2026-10-31, predicted before they were measured): rms 6.0 %, 1 row 3.0 %, 8 rows -6 %
    // on average (the estimate slower); the worst the RTX 2080 Ti's Q6_K at 2 and 4 rows (+25, +20 %;
    // +27, +22 % under the decode rows of master f68eca34, open).
    check(denses == 200 && std::sqrt(dense_sq / denses) < 0.07 && dense_worst < 0.28,
          "a dense model's step at C rows within the model's error");
    std::cout << "big models' long steps " << bigs << " points: rms " << std::sqrt(big_sq / std::max(bigs, 1)) * 100
              << " %, worst " << big_worst * 100 << " %\n";
    // The worst: Qwen3.6 35B-A3B's UD-Q4_K_M at 4 and 8 rows on the RTX 5090 (-11, -13 %; open).
    check(bigs == 22 && std::sqrt(big_sq / bigs) < 0.06 && big_worst < 0.14,
          "the big models' steps at a long context within the model's error");
    check(decodes == 26 && prefills == 15, "the fixture's decode points and 15 prefill points");
    check(decode_rms < 0.10 && decode_worst < 0.17, "a dense model's one-stream decode within the model's error");
    // 33 points (2026-10-30): rms 4.4 %, worst 13 % (UD-IQ3_XXS at 4-6 rows on the RTX 5070 Ti).
    check(moes == 33 && moe_rms < 0.055 && moe_worst < 0.16, "an MoE's decode from facts within the model's error");
    check(prefill_rms < 0.10 && prefill_worst < 0.17, "prefill within the fit's error");
}

// Plain decode from facts (choice_constants.h): every point of the 2026-10-28 campaign
// (tests/fixtures/choice/decode_measured.json: 1/2/4/8 streams, Qwen3.5-9B Q4_K_M and Q8_0, Gemma 4
// 12B Q4_K_M, seven cards of four generations) within the model's spread; the generations that decide
// at the error kDecodePredictionError states (the worth's tie band: derived here, not tuned).
void decodes_from_facts(const json& fixture) {
    double sq = 0, worst = 0;
    int n = 0;
    std::map<std::string, std::pair<double, int>> by_generation; // sum of ln(predicted / measured)^2
    double own_worst = 0; // the worst on the generations that decide
    double long_sq = 0, long_worst = 0;
    int longs = 0, long_open = 0; // Volta's long rows: open
    double steps_sq = 0, steps_worst = 0;
    int long_steps = 0; // several rows at an ~18K-token context (mbatchl.sh)
    double volta_sq = 0, volta_worst = 0;
    int volta_rows = 0; // the Quadro GV100's: its own rows (DECISIONS 72), one card
    for (const json& row : fixture.at("rows")) {
        const json& m            = fixture.at("models").at(row.at("model").get<std::string>());
        const std::string q      = row.at("quant").get<std::string>();
        const ch::ModelFacts mf  = model_at(m, q);
        const ch::CardFacts cf   = card_of(fixture.at("cards").at(row.at("card").get<std::string>()));
        const ch::QuantOption qo = quant_of(q, m.at("quants").at(q));
        ch::Candidate c;
        c.kv    = kv_of(row.at("kv").get<std::string>());
        c.ring  = kv_of(row.at("ring").get<std::string>());
        c.state = row.at("state").get<std::string>() == "f32" ? RecurrentStateStorage::FP32 : RecurrentStateStorage::FP16;
        const ch::Speed s = ch::speed_of(mf, cf, qo, c, row.at("concurrency").get<std::uint32_t>(),
                                         row.at("context").get<double>());
        const double d    = s.decode_per_stream / row.at("decode").get<double>() - 1;
        std::cout << "  " << row.at("card").get<std::string>() << ' ' << row.at("model").get<std::string>() << ' ' << q
                  << " x" << row.at("concurrency").get<int>() << ": " << s.decode_per_stream << " against "
                  << row.at("decode").get<double>() << " (" << d * 100 << " %)\n";
        if (row.value("long", false)) { // 7K .. 82K tokens of context (2026-10-31): their own guard
            const std::string gen = std::string(ch::generation_of(cf));
            if (row.at("concurrency").get<int>() > 1 || row.at("source").get<std::string>().starts_with("mbatchl")) {
                steps_sq += d * d, steps_worst = std::max(steps_worst, std::abs(d)), ++long_steps;
            } else if (gen != "volta") {
                long_sq += d * d, long_worst = std::max(long_worst, std::abs(d)), ++longs;
            } else {
                ++long_open;
            }
            continue;
        }
        const std::string gen = std::string(ch::generation_of(cf));
        if (gen == "volta") { // the Quadro GV100's cool points (thchk.sh): their own guard
            volta_sq += d * d, volta_worst = std::max(volta_worst, std::abs(d)), ++volta_rows;
            continue;
        }
        sq += d * d;
        worst = std::max(worst, std::abs(d));
        ++n;
        auto& g = by_generation[gen];
        g.first += std::log1p(d) * std::log1p(d);
        ++g.second;
        if (gen == "ada" || gen == "gb20x") { own_worst = std::max(own_worst, std::abs(std::log1p(d))); }
    }
    const double rms = std::sqrt(sq / n);
    std::cout << "decode from facts, " << n << " points: rms " << rms * 100 << " %, worst " << worst * 100 << " %\n";
    check(n == 100 && rms < 0.04, "decode from facts within the model's spread");
    // Volta (DECISIONS 72): its K-quants at the SMs' bytes a clock, its Q8_0 at its HBM2's share.
    const double volta_rms = std::sqrt(volta_sq / volta_rows);
    std::cout << "Volta decode, " << volta_rows << " points: rms " << volta_rms * 100 << " %, worst "
              << volta_worst * 100 << " %\n";
    check(volta_rows == 12 && volta_rms < 0.04 && volta_worst < 0.07, "Volta decode from its own rows");
    // The generations whose rows rest on their own census: within 10 %; Turing (one card) within its
    // stated rows' 6 % rms; Ampere (one card, its own census) within 3 %.
    const auto gen_rms = [&](const char* g) {
        const auto& v = by_generation[g];
        return v.second > 0 ? std::sqrt(v.first / v.second) : 1.0;
    };
    check(by_generation["turing"].second == 12 && gen_rms("turing") < 0.06, "Turing's estimate within its rows' error");
    check(by_generation["ga10x"].second == 12 && gen_rms("ga10x") < 0.03, "Ampere's estimate within its rows' error");
    for (const ch::PredictionError& e : ch::kDecodePredictionError) {
        const auto& g       = by_generation[std::string(e.generation)];
        const double g_rms  = g.second > 0 ? std::sqrt(g.first / g.second) : 0;
        std::cout << "  " << e.generation << ": " << g.second << " points, rms " << g_rms << " (stated " << e.rms << ")\n";
        check(g.second >= 24 && std::abs(g_rms - e.rms) < 0.0015, "the stated decode error is the measured one");
    }
    check(own_worst < 0.10, "the deciding generations' worst point within 10 %");
    std::cout << "long-context decode, " << longs << " points: rms " << std::sqrt(long_sq / std::max(longs, 1)) * 100
              << " %, worst " << long_worst * 100 << " % (" << long_open << " open)\n";
    // The worst: the RTX 4070 Ti at 82K (+6 .. +15 %, measured twice; open).
    check(longs == 63 && long_open == 9 && std::sqrt(long_sq / longs) < 0.04 && long_worst < 0.16,
          "long-context decode on Turing, Ada, Blackwell and Ampere within 4 %");
    std::cout << "long-context steps, " << long_steps << " points: rms "
              << std::sqrt(steps_sq / std::max(long_steps, 1)) * 100 << " %, worst " << steps_worst * 100 << " %\n";
    // The worst: the RTX 3060's Qwen Q4_K_M at 8 rows (+16 %, its 4 rows +1 %; open).
    check(long_steps == 72 && std::sqrt(steps_sq / long_steps) < 0.04 && steps_worst < 0.17,
          "steps at 1 .. 8 rows at a long context within 4 %");
    {
        // kKvSmRate holds for the geometry it was read on: Gemma 4 12B's one KV head; two (the 26B's
        // group 8) keep the codec's factor.
        const json& g            = fixture.at("models").at("gemma-4-12b");
        const ch::CardFacts tu   = card_of(fixture.at("cards").at("RTX 2080 Ti"));
        const ch::QuantOption qo = quant_of("Q4_K_M", g.at("quants").at("Q4_K_M"));
        ch::Candidate c;
        c.kv = c.ring            = KvCacheStorage::BFloat16;
        const ch::ModelFacts own = model_at(g, "Q4_K_M");
        ch::ModelFacts other     = own;
        other.full_kv_heads      = 2;
        const double a           = ch::speed_of(own, tu, qo, c, 1, 82000).decode_per_stream;
        const double b           = ch::speed_of(other, tu, qo, c, 1, 82000).decode_per_stream;
        check(a > b * 1.1, "the SM rate holds only for its own head geometry");
    }
}

// The kernels a step launches are the graph's (kDecodeLaunches against the nsys census,
// tests/fixtures/choice/decode_census.json): every card's weight-reading kernels, and its small ones at
// one row.
void counts_the_kernels(const json& census) {
    int n = 0;
    for (const json& row : census.at("rows")) {
        const std::string model = row.at("model").get<std::string>();
        const std::string family = model.rfind("gemma", 0) == 0 ? "gemma4_text" : "qwen3_5_text";
        const ch::WeightClass w  = ch::label_class(row.at("quant").get<std::string>());
        const auto* l = std::find_if(std::begin(ch::kDecodeLaunches), std::end(ch::kDecodeLaunches),
                                     [&](const ch::DecodeLaunches& r) { return !r.moe && r.family == family && r.weights == w; });
        if (l == std::end(ch::kDecodeLaunches)) {
            check(false, "a census row has its family's launches");
            continue;
        }
        check(row.at("gemv_kernels").get<double>() == l->gemv, "the weight-reading kernels a step: the census's");
        if (row.at("streams").get<int>() == 1) {
            // The Quadro GV100 appends Qwen3.5's int8 KV in a kernel of its own a full-attention layer
            // (kv_cache_append_full_i8, 8; fused into the norm and RoPE from Turing on): 8 more.
            const bool own_append = row.at("card").get<std::string>() == "Quadro GV100" && family == "qwen3_5_text";
            check(row.at("small_kernels").get<double>() == l->small[0] + (own_append ? 8 : 0),
                  "the small kernels at one row: the census's");
        }
        ++n;
    }
    check(n >= 68, "the census rows");
    // The weight class from formats (their scheduling class), else the label (label_formats).
    using ninferno::QType;
    const auto of = [](std::initializer_list<QType> formats) {
        return ch::weight_class_of(std::span<const QType>(formats.begin(), formats.size()));
    };
    check(of({QType::GGML_Q4_K, QType::GGML_Q6_K, QType::Q8_G32_FP16, QType::BF16}) ==
              ch::WeightClass::KQuant,
          "a K-quant mix with Q8_0 and 16-bit tensors");
    check(of({QType::Q8_G32_FP16, QType::BF16}) == ch::WeightClass::Q8, "Q8_0 alone");
    check(of({QType::GGML_IQ4_NL, QType::Q8_G32_FP16, QType::GGML_Q5_K}) == ch::WeightClass::IQ4,
          "IQ4_NL with K-quant tensors: IQ4");
    check(of({QType::GGML_Q4_K, QType::GGML_Q3_K}) == ch::WeightClass::Other &&
              of({QType::BF16, QType::FP32}) == ch::WeightClass::Other &&
              of({}) == ch::WeightClass::Other,
          "a format no census measured, plain words alone and no format: Other");
    for (const char* q : {"IQ4_XS", "IQ4_NL", "UD-IQ4_XS"}) {
        check(ch::label_class(q) == ch::WeightClass::IQ4, "an IQ4 label");
    }
    for (const char* k : {"Q4_K_M", "UD-Q4_K_XL", "Q5_K_M", "UD-Q5_K_XL", "Q6_K", "q6_k"}) {
        check(ch::label_class(k) == ch::WeightClass::KQuant, "a K-quant label");
    }
    for (const char* q : {"Q8_0", "Q8", "q8_0", "UD-Q8_K_XL"}) {
        check(ch::label_class(q) == ch::WeightClass::Q8, "a Q8 label");
    }
    for (const char* o : {"UD-IQ3_XXS", "Q4_0", "Q3_K_M", "MXFP4", "NVFP4", "BF16", "Q8_KV", "Q8_K_R8", "Q8_0_R8", "Q4_K_R4", "IQ4_XS_R8", "Q4_1", "Q4_0"}) {
        check(ch::label_class(o) == ch::WeightClass::Other, "an unmeasured label");
    }
    // The formats a label names: its own, not a repacked layout's; none for a label of no format.
    check(ch::label_formats("UD-Q4_K_XL") == std::set<QType>{QType::GGML_Q4_K} &&
              ch::label_formats("Q8_0") == std::set<QType>{QType::Q8_G32_FP16} &&
              ch::label_formats("UD-Q8_K_XL") == std::set<QType>{QType::Q8_G32_FP16, QType::BF16} &&
              ch::label_formats("IQ4_XS") == std::set<QType>{QType::GGML_IQ4_XS} &&
              ch::label_formats("Q4_1") == std::set<QType>{QType::GGML_Q4_1} &&
              ch::label_formats("TQ1_0") == std::set<QType>{QType::GGML_TQ1_0} &&
              ch::label_formats("NVFP4") == std::set<QType>{QType::NVFP4},
          "a label names its formats");
    for (const char* none :
         {"Q8_KV", "Q4_K_R4", "Q8_0_R8", "IQ4_XS_R8", "the model's weights", ""}) {
        check(!ch::label_formats(none), "a label of no format (or a repacked layout's) names none");
    }
}

// Every label of the decode census (tests/fixtures/choice/decode_census.json) names formats whose
// class is the one its rows were measured as (#615 F7: the label's parse, label_formats, gives the
// class the string compare gave), and a family's launches are there for it.
void census_labels_keep_their_class(const json& census) {
    using ninferno::QType;
    const std::map<std::string, ch::WeightClass> measured = {{"Q4_K_M", ch::WeightClass::KQuant},
                                                             {"Q8_0", ch::WeightClass::Q8},
                                                             {"IQ4_XS", ch::WeightClass::IQ4},
                                                             {"IQ4_NL", ch::WeightClass::IQ4}};
    std::set<std::string> labels;
    for (const json& row : census.at("rows")) { labels.insert(row.at("quant").get<std::string>()); }
    check(labels.size() == measured.size(), "the census's labels are the ones listed");
    for (const std::string& label : labels) {
        const auto want                              = measured.find(label);
        const std::optional<std::set<QType>> formats = ch::label_formats(label);
        if (want == measured.end() || !formats) {
            check(false, "a census label names its formats");
            continue;
        }
        const std::vector<QType> read(formats->begin(), formats->end());
        check(ch::weight_class_of(read) == want->second && ch::label_class(label) == want->second,
              "a census label's formats keep its measured class");
    }
}

// A prefill's tail calls (DECISIONS 62): each a plain decode step at the prompt's depth, in the
// prefill rate and in a request's time; none where the frontend counted none.
void prefills_pay_tail_calls(const json& fixture) {
    const json& m          = fixture.at("models").at("Qwen3.5-9B-MTP");
    ch::ModelFacts none    = model_at(m, "Q4_K_M");
    none.drafters          = {{"mtp:4", "mtp", 1.9, 2, "measured"}};
    ch::ModelFacts two     = none;
    two.prompt_tail_calls  = 2;
    const ch::CardFacts cf = card_of(fixture.at("cards").at("RTX 4070 Ti"));
    const ch::QuantOption q = quant_of("Q4_K_M", m.at("quants").at("Q4_K_M"));
    ch::Candidate c;
    c.kv                       = KvCacheStorage::Int8Group64;
    const std::uint32_t prompt = 2048;
    const double step = 1 / ch::speed_of(none, cf, q, c, 1, prompt, 0).decode_per_stream;
    const double t0   = prompt / ch::speed_of(none, cf, q, c, 1, prompt, prompt).prefill;
    const double t2   = prompt / ch::speed_of(two, cf, q, c, 1, prompt, prompt).prefill;
    check(std::abs((t2 - t0) - 2 * step) < 1e-9 * t2, "two tail calls add two decode steps to a prefill");
    ch::Task task;
    task.prompt     = prompt;
    task.answer     = 1;
    const double r0 = ch::single_request_seconds(none, cf, q, c, task, task.prompt);
    const double r2 = ch::single_request_seconds(two, cf, q, c, task, task.prompt);
    check(std::abs((r2 - r0) - 2 * step) < 1e-9 * r2, "and to a request's time");
    ch::Candidate drafted = c;
    drafted.drafter       = 1;
    const double d0       = ch::single_request_seconds(none, cf, q, drafted, task, task.prompt);
    const double d2       = ch::single_request_seconds(two, cf, q, drafted, task, task.prompt);
    check(std::abs((d2 - d0) - 2 * step) < 1e-9 * d2, "a drafted candidate's tail calls are plain decode steps");
    std::cout << "tail calls: a decode step " << step * 1e3 << " ms, 2K prefill " << prompt / t0 << " -> "
              << prompt / t2 << " tok/s\n";
}

// Prefill by format (choice_constants.h kPrefillTilesEta, kMmqTileCost): every 3K-22K-token prompt (the 41K
// and 82K ones apart, their own guard)
// of tests/fixtures/choice/prefill_measured.json (Qwen3.5 9B on seven cards, Gemma 4 12B on seven, Qwen3.5 4B on eight; Q8_0
// re-measured after PR #249, master f8d4399d; Qwen3.5 9B's Q6_K, Q5_K_M and Q4_K_S at 7K and 22K),
// Q8_0 on the int8 tiles, the K-quants on MMQ's tiles (DECISIONS 69) where the card takes MMQ.
// The Q8 rows are the tile's clock count, not a fit of these points (DECISIONS 67); Turing's is its
// measured row.
void prefills_by_format(const json& fixture) {
    double sq[2] = {0, 0}, worst[2] = {0, 0}, worst_before[2] = {0, 0}, sq_before[2] = {0, 0};
    int before[2] = {0, 0};
    // The K-quant rows before DECISIONS 69's check (Q4_K_M) and that check's (Qwen3.5-9B's other mixes).
    double sq_first = 0, sq_mixes = 0, worst_mixes = 0, sq_gemma = 0, worst_gemma = 0;
    int first = 0, mixes = 0, gemma = 0, volta_rows = 0;
    double sq_volta = 0, worst_volta = 0, sq_q3 = 0, worst_q3 = 0;
    double sq_long = 0, worst_long = 0;
    int long_rows = 0;
    int q3 = 0, iq4 = 0;
    double sq_iq4 = 0, worst_iq4 = 0, sq_xl = 0, worst_xl = 0;
    int xl_rows = 0;
    std::map<std::string, std::vector<double>> by_card;
    int n[2] = {0, 0};

    struct Stated {
        double sq = 0, worst = 0;
        int n = 0;
        std::set<std::string> cards;
    };

    std::map<std::string, Stated> stated; // by generation
    for (const json& row : fixture.at("rows")) {
        const json& m           = fixture.at("models").at(row.at("model").get<std::string>());
        const std::string q     = row.at("quant").get<std::string>();
        const ch::ModelFacts mf = model_at(m, q);
        const ch::CardFacts cf  = card_of(fixture.at("cards").at(row.at("card").get<std::string>()));
        const ch::QuantOption qo = quant_of(q, m.at("quants").at(q));
        ch::Candidate c;
        c.kv                       = KvCacheStorage::Int8Group64;
        const std::uint32_t prompt = row.at("prompt").get<std::uint32_t>();
        const double d = ch::speed_of(mf, cf, qo, c, 1, prompt, prompt).prefill / row.at("prefill").get<double>() - 1;
        if (prompt > 30000) { // the long prompts (41K, 82K): their own guard
            std::cout << "  long " << row.at("card").get<std::string>() << ' ' << row.at("model").get<std::string>() << ' '
                      << q << ' ' << prompt << ": " << std::lround(d * 100) << " %\n";
            sq_long += d * d, worst_long = std::max(worst_long, std::abs(d)), ++long_rows;
            continue;
        }
        if (cf.major == 7 && cf.minor == 0) { // Volta's rows (DECISIONS 70): their own guard
            std::cout << "  Volta " << q << " " << row.at("model").get<std::string>() << " " << prompt << ": "
                      << std::lround(d * 100) << " %\n";
            sq_volta += d * d, worst_volta = std::max(worst_volta, std::abs(d)), ++volta_rows;
            continue;
        }
        // kPrefillPredictionError's points: a dense model's Own estimate on the fast route.
        const ch::Speed fast = ch::speed_of(mf, cf, qo, c, 1, prompt, prompt);
        if (!mf.moe() && fast.basis == ch::SpeedBasis::Own &&
            ch::route_of(qo, c.activations, cf) == ch::Route::Int8 &&
            prompt >= ch::kPrefillErrorFromPrompt && prompt <= ch::kPrefillErrorToPrompt) {
            auto& g = stated[std::string(ch::generation_of(cf))];
            g.sq += d * d, g.worst = std::max(g.worst, std::abs(d)), ++g.n;
            g.cards.insert(row.at("card").get<std::string>());
        }
        const int i    = qo.weights == ch::WeightClass::Q8 ? 1 : 0;
        by_card[q + " " + row.at("model").get<std::string>() + " " +
                row.at("card").get<std::string>()]
            .push_back(d);
        sq[i] += d * d;
        worst[i] = std::max(worst[i], std::abs(d));
        if (i == 0 && q == "UD-Q4_K_XL") {
            sq_xl += d * d, worst_xl = std::max(worst_xl, std::abs(d)), ++xl_rows;
        } else if (i == 0 && q == "IQ4_XS") {
            sq_iq4 += d * d, worst_iq4 = std::max(worst_iq4, std::abs(d)), ++iq4;
        } else if (i == 0 && q.rfind("Q3_K", 0) == 0) { // MMQ's tiles; the RTX 5070 Ti's W4A8 ones
            sq_q3 += d * d, worst_q3 = std::max(worst_q3, std::abs(d)), ++q3;
        } else if (i == 0 && row.at("model").get<std::string>() == "Qwen3.5-9B") {
            sq_mixes += d * d, worst_mixes = std::max(worst_mixes, std::abs(d)), ++mixes;
        } else if (i == 0 && q != "Q4_K_M") {
            sq_gemma += d * d, worst_gemma = std::max(worst_gemma, std::abs(d)), ++gemma;
        } else if (i == 0 && row.at("card").get<std::string>() != "RTX 5090") {
            sq_first += d * d, ++first;
        }
        if (!(row.at("model").get<std::string>() == "gemma-4-12b" && row.at("card").get<std::string>() == "RTX 5090")) {
            worst_before[i] = std::max(worst_before[i], std::abs(d));
            sq_before[i] += d * d;
            ++before[i];
        }
        ++n[i];
    }
    const double mmq = std::sqrt(sq[0] / n[0]), tiles = std::sqrt(sq[1] / n[1]);
    // The prefill error a caller shows (kPrefillPredictionError, #618 step 11): derived here, not
    // tuned, as kDecodePredictionError's; a row for each generation of two or more lab cards, none
    // for one of a single card.
    for (const auto& [gen, g] : stated) {
        const double rms = std::sqrt(g.sq / g.n);
        std::cout << "prefill error " << gen << ": " << g.n << " points on " << g.cards.size()
                  << " cards, rms " << rms << ", worst " << g.worst << "\n";
        const auto it = std::find_if(
            std::begin(ch::kPrefillPredictionError), std::end(ch::kPrefillPredictionError),
            [&](const ch::PrefillPredictionError& e) { return e.generation == gen; });
        if (g.cards.size() < 2) {
            check(it == std::end(ch::kPrefillPredictionError),
                  "prefill error: " + gen + " (one card) states none");
            continue;
        }
        check(it != std::end(ch::kPrefillPredictionError) &&
                  it->cards == static_cast<int>(g.cards.size()) &&
                  std::abs(it->rms - rms) < 0.0015 && std::abs(it->worst - g.worst) < 0.0015,
              "prefill error: " + gen + "'s stated row is the measured one");
    }
    for (const ch::PrefillPredictionError& e : ch::kPrefillPredictionError) {
        check(stated.count(std::string(e.generation)) == 1,
              "prefill error: every stated row is measured here");
    }
    for (const auto& [card, ds] : by_card) {
        std::cout << "  " << card << ":";
        for (const double d : ds) { std::cout << " " << std::lround(d * 100) << " %"; }
        std::cout << "\n";
    }
    std::cout << "prefill by format: K-quants " << n[0] << " points rms " << mmq * 100 << " % (worst "
              << worst[0] * 100 << " %), Q8_0 " << n[1] << " rms " << tiles * 100 << " % (worst " << worst[1] * 100
              << " %)\n";
    // DECISIONS 66: the tiles by linear in rounds of SMs x resident CTAs and the small kernels, no
    // fit. Its worst points are Gemma 4 12B's on the RTX 5090 and 4090 (+17 .. +26 %, its Q4_K_M
    // likewise): Gemma's attention kernel runs 1.9x the attention term there (the census at 1714
    // tokens: 14.2 against 7.7 ms on the 5090, 18.3 against 9.7 on the 4090). The points before
    // the RTX 5090's Gemma keep their bounds.
    check(n[1] == 66 && tiles < 0.10 && worst[1] < 0.27, "Q8_0 prefill on the int8 tiles within the model's spread");
    check(before[1] == 60 && std::sqrt(sq_before[1] / before[1]) < 0.08 && worst_before[1] < 0.20,
          "Q8_0 prefill but the RTX 5090's Gemma: rms within 8 %, every point within 20 %");
    // The K-quants (Q4_K_M, and Qwen3.5 9B's Q6_K, Q5_K_M and Q4_K_S on seven cards at 7K and 22K: DECISIONS 69),
    // with Q3_K_M and IQ4_XS.
    check(before[0] == 197 && std::sqrt(sq_before[0] / before[0]) < 0.10 && worst_before[0] < 0.28,
          "K-quant prefill but the RTX 5090's Gemma as before");
    check(n[0] == 207 && mmq < 0.10 && worst[0] < 0.28, "K-quant prefill on MMQ's tiles within the model's spread");
    // The Q4_K_M points alone keep their guard (all but the RTX 5090's: rms within 10 %), and the
    // other mixes have their own: rms within 8 %, every point within 20 %.
    check(first == 84 && std::sqrt(sq_first / first) < 0.10, "Q4_K_M prefill as before");
    check(mixes == 42 && std::sqrt(sq_mixes / mixes) < 0.08 && worst_mixes < 0.20,
          "Qwen3.5 9B's Q6_K, Q5_K_M and Q4_K_S within 8 % rms, every point within 20 %");
    // Gemma 4 12B's Q5_K_M and Q6_K, predicted before they were measured: within 8 % rms; the RTX
    // 2080 Ti's Q6_K at 7K the worst (+27 %, its Q5_K_M +11 %: open, DECISIONS 69), the rest within 11 %.
    check(gemma == 27 && std::sqrt(sq_gemma / gemma) < 0.08 && worst_gemma < 0.30,
          "Gemma 4 12B's Q5_K_M and Q6_K within 8 % rms");
    std::cout << "Q3_K_M prefill: " << q3 << " points rms " << std::sqrt(sq_q3 / q3) * 100 << " % (worst "
              << worst_q3 * 100 << " %)\n";
    // Qwen3.5 9B's Q3_K_M on its tiles (MMQ; the RTX 5070 Ti's W4A8 64 x 64), seven cards at 7K and 22K.
    check(q3 == 14 && std::sqrt(sq_q3 / q3) < 0.07 && worst_q3 < 0.10, "Q3_K_M prefill on its tiles within 10 %");
    // Qwen3.5 9B's IQ4_XS on its tiles (MMQ; the RTX 5070 Ti's producer-warp W4A8 128 x 64), six cards.
    std::cout << "IQ4_XS prefill: " << iq4 << " points rms " << std::sqrt(sq_iq4 / iq4) * 100 << " % (worst "
              << worst_iq4 * 100 << " %)\n";
    check(iq4 == 12 && std::sqrt(sq_iq4 / iq4) < 0.07 && worst_iq4 < 0.10, "IQ4_XS prefill on its tiles within 10 %");
    std::cout << "UD-Q4_K_XL prefill: " << xl_rows << " points rms " << std::sqrt(sq_xl / xl_rows) * 100 << " % (worst "
              << worst_xl * 100 << " %)\n";
    // Unsloth's UD-Q4_K_XL (Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 linears) by its linears' formats.
    check(xl_rows == 14 && std::sqrt(sq_xl / xl_rows) < 0.07 && worst_xl < 0.12, "UD-Q4_K_XL prefill within 12 %");
    {
        // What the IQ4 law leaves to the share: an IQ4_NL label, a K-quant label in the IQ4 class (a
        // Q4_K_M file with an iq4 tensor), a file of no linear read; an iq4_nl linear among read ones
        // takes the share alone, beside the others' tiles.
        const json& qm             = fixture.at("models").at("Qwen3.5-9B");
        const ch::ModelFacts named = model_at(qm, "IQ4_XS");
        const ch::CardFacts ada    = card_of(fixture.at("cards").at("RTX 4060 Ti"));
        ch::QuantOption xs         = quant_of("IQ4_XS", qm.at("quants").at("IQ4_XS"));
        xs.weights                 = ch::WeightClass::IQ4;
        ch::QuantOption nl = xs, km = xs;
        nl.label           = "IQ4_NL";
        km.label           = "Q4_K_M";
        ch::ModelFacts with_nl = named;
        with_nl.linears.front().format = "ggml_iq4_nl";
        // A K label whose named linears hold iq4_xs ones (UD-Q4_K_XL: the IQ4 class) takes its linears'.
        ch::QuantOption xl = quant_of("UD-Q4_K_XL", qm.at("quants").at("UD-Q4_K_XL"));
        xl.weights         = ch::WeightClass::IQ4;
        const ch::ModelFacts xl_named = model_at(qm, "UD-Q4_K_XL");
        check(ch::prefill_linears_seconds(xl_named, ada, xl, 1024) > 0, "UD-Q4_K_XL in the IQ4 class on its linears");
        ch::ModelFacts unnamed = named;
        for (ch::LinearShape& l : unnamed.linears) { l.format.clear(); }
        ch::ModelFacts all_nl = named;
        for (ch::LinearShape& l : all_nl.linears) { l.format = "ggml_iq4_nl"; }
        check(ch::prefill_linears_seconds(named, ada, xs, 1024) > 0 && ch::prefill_linears_seconds(named, ada, xs, 16) == 0 &&
                  ch::prefill_linears_seconds(unnamed, ada, km, 1024) == 0 &&
                  ch::prefill_linears_seconds(all_nl, ada, xs, 1024) == 0,
              "IQ4_NL, an unnamed K label in the IQ4 class and a file of no linear read keep the share");
        bool nl_transferred = false;
        const double nl_seconds = ch::prefill_linears_seconds(with_nl, ada, xs, 1024, &nl_transferred);
        ch::ModelFacts rest = with_nl;
        rest.linears.erase(rest.linears.begin());
        check(nl_seconds > ch::prefill_linears_seconds(rest, ada, xs, 1024) && nl_transferred,
              "an iq4_nl linear among read ones: the share alone, on top of the others' tiles, marked transferred");
        // Q4_1 (WeightClass::Other) on its census's step where it was read (Ada, Ampere), one card each:
        // near Q4_K's tiles; Turing has none read (the share).
        ch::ModelFacts q41_model = named, q4k_model = named;
        for (ch::LinearShape& l : q41_model.linears) { l.format = "ggml_q4_1"; }
        for (ch::LinearShape& l : q4k_model.linears) { l.format = "ggml_q4_k"; }
        ch::QuantOption q41 = xs;
        q41.label           = "Q4_1";
        q41.weights         = ch::WeightClass::Other;
        bool q41_transferred = false;
        const double q41_seconds = ch::prefill_linears_seconds(q41_model, ada, q41, 1024, &q41_transferred);
        const double q4k_seconds = ch::prefill_linears_seconds(q4k_model, ada, km, 1024);
        // GA10x and Ada (#524, DECISIONS 74, f 0.20 on both): a slice of 144 tokens takes two
        // 80-column tiles (1.57 of a 128-column tile's steps), so it costs less than a slice of 256
        // (two 128-column tiles). Turing keeps the fewest tiles.
        const ch::CardFacts rtx3060 = card_of(fixture.at("cards").at("RTX 3060"));
        const double at_144         = ch::prefill_linears_seconds(q4k_model, rtx3060, km, 144);
        const double at_256         = ch::prefill_linears_seconds(q4k_model, rtx3060, km, 256);
        check(at_144 > 0 && at_144 < 0.9 * at_256,
              "GA10x: 144 tokens take narrower column tiles than 256 (two 80-column tiles against "
              "two 128)");
        const double ada_144 = ch::prefill_linears_seconds(q4k_model, ada, km, 144);
        check(ada_144 > 0 && ada_144 < 0.9 * ch::prefill_linears_seconds(q4k_model, ada, km, 256),
              "Ada: 144 tokens take narrower column tiles than 256 (two 80-column tiles against "
              "two 128)");
        const ch::CardFacts turing = card_of(fixture.at("cards").at("RTX 2080 Ti"));
        check(q41_seconds > 0 && q41_transferred && std::abs(q41_seconds / q4k_seconds - 1) < 0.05 &&
                  ch::prefill_linears_seconds(q41_model, turing, q41, 1024) == 0,
              "Q4_1 on its read step (Ada, near Q4_K's), marked transferred; Turing: the share");
        // A renamed file's label names no format: its named linears' format stands for it.
        ch::QuantOption renamed = q41;
        renamed.label           = "the model's weights";
        check(ch::prefill_linears_seconds(q41_model, ada, renamed, 1024) == q41_seconds,
              "a label of no format over named Q4_1 linears prices them as the Q4_1 label");
        // The codebook IQ formats on their census's steps where read (Ampere, Ada: one card each): a file
        // of IQ3_XXS linears only (no K format among them) above IQ4_XS's tiles by its step; Turing has
        // none read (the share); an IQ1_M linear (its W4A8 route, not read) the share on top of the
        // others' tiles.
        const ch::CardFacts ampere = card_of(fixture.at("cards").at("RTX 3060"));
        ch::ModelFacts iq3_model = named, iq4_model = named;
        for (ch::LinearShape& l : iq3_model.linears) { l.format = "ggml_iq3_xxs"; }
        for (ch::LinearShape& l : iq4_model.linears) { l.format = "ggml_iq4_xs"; }
        ch::QuantOption iq3 = xs;
        iq3.label           = "UD-IQ3_XXS";
        iq3.weights         = ch::WeightClass::Other;
        bool iq3_transferred = false;
        const double iq3_seconds = ch::prefill_linears_seconds(iq3_model, ampere, iq3, 1024, &iq3_transferred);
        const double iq4_seconds = ch::prefill_linears_seconds(iq4_model, ampere, xs, 1024);
        check(iq3_seconds > iq4_seconds && iq3_seconds < iq4_seconds * 40.7 / 36.6 * 1.02 && iq3_transferred &&
                  ch::prefill_linears_seconds(iq3_model, turing, iq3, 1024) == 0,
              "IQ3_XXS linears on their read step (Ampere, Ada), marked transferred; Turing: the share");
        // Ada's MMQ row reads two cards: the IQ step alone (one) marks its estimate transferred.
        bool ada_transferred = false;
        check(ch::prefill_linears_seconds(iq3_model, ada, iq3, 1024, &ada_transferred) > 0 && ada_transferred,
              "IQ3_XXS linears on Ada's read step, marked transferred by the IQ step's one card");
        ch::ModelFacts with_iq1m = iq3_model;
        with_iq1m.linears.front().format = "ggml_iq1_m";
        ch::ModelFacts iq3_rest = iq3_model;
        iq3_rest.linears.erase(iq3_rest.linears.begin());
        const double with_iq1m_seconds = ch::prefill_linears_seconds(with_iq1m, ampere, iq3, 1024);
        check(with_iq1m_seconds > ch::prefill_linears_seconds(iq3_rest, ampere, iq3, 1024) &&
                  with_iq1m_seconds != iq3_seconds,
              "an IQ1_M linear among IQ3_XXS ones: the share on top of their tiles, not a tile step");
    }
    std::cout << "long prompts: " << long_rows << " points rms " << std::sqrt(sq_long / std::max(long_rows, 1)) * 100
              << " % (worst " << worst_long * 100 << " %)\n";
    // 41K and 82K prompts (2026-10-31, predicted before the measurement): Qwen3.5 9B's Q4_K_M and Q8_0 and
    // Gemma 4 12B's Q4_K_M on seven cards, rms 5.1 %; the worst the Quadro GV100's Qwen (-5 .. -14 %).
    check(long_rows == 42 && std::sqrt(sq_long / long_rows) < 0.07 && worst_long < 0.16,
          "long-prompt prefill within the model's error");
    std::cout << "Volta prefill: " << volta_rows << " points rms " << std::sqrt(sq_volta / volta_rows) * 100 << " % (worst "
              << worst_volta * 100 << " %)\n";
    // The Quadro GV100 end to end (DECISIONS 70's linears, Gemma's Volta row): Qwen3.5 9B's Q6_K,
    // Q5_K_M, Q4_K_S and Gemma 4 12B's Q4_K_M .. Q6_K at 7K and 22K; the worst Qwen's Q4_K_S and
    // Q5_K_M at 7K (+11, +13 %: open), one card.
    check(volta_rows == 14 && std::sqrt(sq_volta / volta_rows) < 0.08 && worst_volta < 0.15,
          "Volta's prefill within 8 % rms, every point within 15 %");

    // What the tiles row changes, and what it leaves. The model's Q8_0 and Q4_K_M at one size
    // (the rate is per parameter), so the weight class alone differs.
    const json& m             = fixture.at("models").at("Qwen3.5-9B-MTP");
    const ch::ModelFacts mf   = model_at(m, "Q8_0");
    const ch::QuantOption q8  = quant_of("Q8_0", m.at("quants").at("Q8_0"));
    ch::QuantOption kq        = q8;
    kq.weights                = ch::WeightClass::KQuant;
    ch::Candidate c;
    c.kv = KvCacheStorage::Int8Group64;
    const auto at = [&](const char* card, const ch::QuantOption& q, const ch::ModelFacts& model,
                        std::uint32_t rows) {
        return ch::speed_of(model, card_of(fixture.at("cards").at(card)), q, c, rows, 4096, 2048);
    };
    const ch::Speed ada_q8 = at("RTX 4090", q8, mf, 1), ada_kq = at("RTX 4090", kq, mf, 1);
    // Without attention the prefill is the linears alone: their rates are the tile's clock count at
    // the model's hidden width (Ada c0 3352, c1 12.15 clocks against 2048 s8 operations a SM-clock:
    // 0.308 at 4096) and MMQ's int8 row (0.306), exactly; a narrower model pays c0 over fewer K steps.
    ch::ModelFacts linears_only          = mf;
    linears_only.full_attention_width    = 0;
    linears_only.sliding_attention_width = 0;
    linears_only.linears.clear(); // the shares: without the shapes the tiles by linear have nothing to price
    const auto tiles_over_mmq            = [&](double hidden) {
        linears_only.hidden = hidden;
        return at("RTX 4090", q8, linears_only, 1).prefill / at("RTX 4090", kq, linears_only, 1).prefill;
    };
    const auto law = [](double k) { return 2.0 * 64 * 64 / ((3352 / k + 12.15) * 2048) / 0.306; };
    check(std::abs(tiles_over_mmq(4096) - law(4096)) < 1e-9 && std::abs(law(4096) * 0.306 - 0.308) < 5e-4,
          "Ada: a Q8 prefill takes the tile law at K 4096 (0.308 against 0.306)");
    check(std::abs(tiles_over_mmq(2560) - law(2560)) < 1e-9 && tiles_over_mmq(2560) < tiles_over_mmq(5120),
          "Ada: the tile law at the model's own width (2560 below 5120)");
    check(!ada_q8.transferred, "Ada: the tiles row rests on three cards");
    // One card's rows price only their own boards: marked transferred.
    check(at("RTX 2080 Ti", q8, mf, 1).transferred && at("RTX 2080 Ti", kq, mf, 1).transferred &&
              at("RTX 3060", q8, mf, 1).transferred,
          "Turing / Ampere prefill rows of one card are transferred");
    // Volta prefills Q8_0 on its CUTLASS route (DECISIONS 70: one card, transferred); a K-quant without
    // its format in the label keeps the share, transferred too.
    ch::CardFacts volta = card_of(fixture.at("cards").at("RTX 2080 Ti"));
    volta.major = 7;
    volta.minor = 0;
    const ch::Speed v8 = ch::speed_of(mf, volta, q8, c, 1, 4096, 2048);
    const ch::Speed vk = ch::speed_of(mf, volta, kq, c, 1, 4096, 2048);
    check(v8.transferred && vk.transferred && v8.prefill > 0 && vk.prefill > 0 && v8.prefill != vk.prefill,
          "Volta: Q8_0 on its CUTLASS route, the label-less K-quant on the share, both transferred");
    // An MoE keeps its MoE row whatever its weights.
    ch::ModelFacts moe        = mf;
    moe.experts               = 128;
    moe.experts_per_token     = 8;
    moe.routed_expert_params  = mf.params * 0.8;
    const ch::Speed m8 = at("RTX 4090", q8, moe, 1), mk = at("RTX 4090", kq, moe, 1);
    check(std::abs(m8.prefill / mk.prefill - 1) < 1e-9, "an MoE's Q8 prefill on its MoE row");
}

struct Bench {
    ch::ModelFacts model;
    ch::CardFacts card;
    std::vector<ch::QuantOption> quants;
};

Bench nine_b(const json& fixture, const std::string& card) {
    const json& m = fixture.at("models").at("Qwen3.5-9B");
    Bench b{model_of(m), card_of(fixture.at("cards").at(card)), {}};
    b.quants.push_back(quant_of("Q4_K_M", m.at("quants").at("Q4_K_M")));
    ch::QuantOption q8 = b.quants[0];
    q8.label           = "Q8_0";
    q8.bits_per_weight = 8.5;
    q8.weight_bytes    = b.quants[0].weight_bytes * 8.5 / b.quants[0].bits_per_weight;
    b.quants.push_back(q8);
    ch::QuantOption q3 = b.quants[0];
    q3.label           = "IQ3_XXS";
    q3.bits_per_weight = 3.1;
    q3.weight_bytes    = b.quants[0].weight_bytes * 3.1 / b.quants[0].bits_per_weight;
    b.quants.push_back(q3);
    return b;
}

// A memory oracle: the quantization's weights plus KV at the context must stay under `budget`
// bytes.
ch::OwnContextFn oracle(const Bench& b, double budget, std::size_t quant = 0) {
    return [&b, budget, quant](const ch::Candidate& c) -> std::uint32_t {
        const double room = budget - b.quants[quant].weight_bytes;
        if (room <= 0) { return 0; }
        return static_cast<std::uint32_t>(room / b.model.kv_bytes_per_token(c.kv));
    };
}

const ch::Scored& chosen(const ch::Choice& c) { return c.scored[*c.chosen]; }

// Attention over the prompt: a full layer's grows with the prompt, a sliding one's stops at its window;
// Volta's prefill takes its own rows.
void prices_attention(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4090");
    ch::Candidate c;
    const auto per_token = [&](const ch::ModelFacts& m, std::uint32_t n) {
        return 1 / ch::speed_of(m, b.card, b.quants[0], c, 1, n, n).prefill;
    };
    ch::ModelFacts none = b.model, full = b.model, sliding = b.model;
    // The share's attention (no head geometry: no prompt kernel's tiles, DECISIONS 68).
    none.full_head_dim = full.full_head_dim = sliding.full_head_dim = 0;
    none.full_attention_width    = 0;
    full.full_attention_width    = 65536;
    sliding.full_attention_width = 0;
    sliding.sliding_attention_width = 65536;
    sliding.window                  = 1024;
    const double base = per_token(none, 4096);
    check(per_token(full, 8192) - base > 1.9 * (per_token(full, 4096) - base),
          "a full layer's attention a token doubles with the prompt");
    check(std::abs((per_token(sliding, 65536) - base) / (per_token(sliding, 32768) - base) - 1) < 0.02,
          "a sliding layer's attention a token stops growing past its window");
    ch::CardFacts volta = b.card;
    volta.major = 7;
    volta.minor = 0;
    ch::CardFacts turing = volta;
    turing.minor         = 5;
    // Linears only: Volta's int8 route runs at its BF16 rate (no int8 MMA), 1536 against Turing's
    // 2048, at its own row (the GV100 census: dequantized FP16 on the HMMA tiles), one card.
    const ch::Speed v = ch::speed_of(none, volta, b.quants[0], c, 1, 4096, 4096);
    const ch::Speed t = ch::speed_of(none, turing, b.quants[0], c, 1, 4096, 4096);
    check(v.transferred && std::abs(v.prefill / t.prefill - (0.248 * 1536.0) / (0.185 * 2048.0)) < 1e-9,
          "Volta's prefill takes its own row, transferred (one card)");
    // Its exact route: its own int8 row at Turing's exact-to-int8 ratio.
    ch::Candidate exact;
    exact.activations  = ActivationMode::Exact;
    const ch::Speed ve = ch::speed_of(none, volta, b.quants[0], exact, 1, 4096, 4096);
    const ch::Speed te = ch::speed_of(none, turing, b.quants[0], exact, 1, 4096, 4096);
    check(ve.transferred && std::abs(ve.prefill / te.prefill - (0.248 * 1536.0) / (0.185 * 1536.0)) < 1e-9,
          "Volta's exact prefill scales its own int8 row by Turing's ratio");
    // An MoE on Volta keeps the dense model's exact-to-fast ratio.
    ch::ModelFacts moe          = none;
    moe.experts                 = 128;
    moe.experts_per_token       = 8;
    moe.routed_expert_params    = moe.params * 0.8;
    const ch::Speed mf_fast     = ch::speed_of(moe, volta, b.quants[0], c, 1, 4096, 4096);
    const ch::Speed mf_exact    = ch::speed_of(moe, volta, b.quants[0], exact, 1, 4096, 4096);
    check(std::abs((mf_exact.prefill / mf_fast.prefill) / (ve.prefill / v.prefill) - 1) < 1e-9,
          "an MoE's exact-to-fast prefill ratio on Volta is the dense model's");
}

void chooses(const json& fixture) {
    ch::Task task;
    task.contexts = {32768};
    {
        // At Q4_K_M: fast beats exact (+26 % time for 0.0044 KLD); INT8 KV beats FP8 (same cost,
        // more accurate) and, by a hair, BF16 (on top of fast, INT8 KV adds about nothing: measured
        // 0.00388 -> 0.00403 with the FP16 state).
        const Bench b = nine_b(fixture, "RTX 4070 Ti");
        const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                        {}, task, oracle(b, 9.6e9));
        check(c.chosen.has_value(), "4070 Ti: a choice");
        if (c.chosen) {
            const ch::Scored& s = chosen(c);
            check(s.candidate.activations == ActivationMode::Fast, "4070 Ti: fast over exact");
            check(s.candidate.kv == KvCacheStorage::Int8Group64, "4070 Ti: INT8 KV");
            check(s.candidate.state == RecurrentStateStorage::FP16, "4070 Ti: FP16 state under fast");
            std::cout << c.why << '\n';
        }
        const auto alts = ch::explain(c);
        int named = 0;
        for (const ch::Alternative& a : alts) {
            if (a.why_not.empty()) { ++named; }
            check(a.scored != nullptr, "every alternative scored");
        }
        check(named == 3, "the chosen value once per axis (kv, activations, state)");
        bool exact = false;
        for (const ch::Alternative& a : alts) {
            if (a.axis == ch::Axis::Activations && a.value == "exact") { exact = true; }
        }
        check(exact, "exact is an alternative (its FP32 state follows it)");
    }
    {
        // RTX 2080 Ti: BF16 and FP8 KV read slowly on Turing; INT8 stays.
        const Bench b = nine_b(fixture, "RTX 2080 Ti");
        const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                        {}, task, oracle(b, 9.0e9));
        check(c.chosen && chosen(c).candidate.kv == KvCacheStorage::Int8Group64, "2080 Ti: INT8 KV");
    }
    {
        // Pins stay: exact asked, BF16 KV asked.
        const Bench b = nine_b(fixture, "RTX 4070 Ti");
        ch::Pins pins;
        pins.activations = ActivationMode::Exact;
        pins.kv          = KvCacheStorage::BFloat16;
        const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                        pins, task, oracle(b, 9.6e9));
        check(c.chosen && chosen(c).candidate.activations == ActivationMode::Exact &&
                  chosen(c).candidate.kv == KvCacheStorage::BFloat16 &&
                  chosen(c).candidate.state == RecurrentStateStorage::FP32 &&
                  !c.activations_by_worth,
              "pins kept; exact couples the FP32 state; pinned activations are not weighed");
    }
    {
        // The state one step finer than the KV cache: BF16 KV under fast takes the FP32 state; a
        // ring follows the KV codec where the family's rings take FP8.
        const Bench b = nine_b(fixture, "RTX 4070 Ti");
        ch::Pins pins;
        pins.kv = KvCacheStorage::BFloat16;
        const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                        pins, task, oracle(b, 9.6e9));
        check(c.chosen && chosen(c).candidate.state == RecurrentStateStorage::FP32, "BF16 KV: the FP32 state");
        ch::ModelFacts sliding = b.model;
        sliding.window   = 1024;
        sliding.fp8_ring = true;
        ch::Pins none;
        check(ch::coupled_storages(sliding, KvCacheStorage::Fp8E4M3Row256, ActivationMode::Fast, none).ring ==
                  KvCacheStorage::Fp8E4M3Row256 &&
              ch::coupled_storages(sliding, KvCacheStorage::BFloat16, ActivationMode::Fast, none).ring ==
                  KvCacheStorage::BFloat16,
              "the ring follows the KV codec");
        sliding.fp8_ring = false;
        check(ch::coupled_storages(sliding, KvCacheStorage::Fp8E4M3Row256, ActivationMode::Fast, none).ring ==
                  KvCacheStorage::BFloat16,
              "a family whose rings take no FP8 keeps BF16 rings");
    }
    {
        // Nothing holds the context: no choice, and why.
        const Bench b = nine_b(fixture, "RTX 4070 Ti");
        const ch::Choice none = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                           {}, task, oracle(b, 1e9));
        check(!none.chosen && none.why.find("no candidate holds") != std::string::npos, "nothing fits: no choice");
    }
}

// What a decision may read: every board of a generation prices on its reference card, so the
// 4070 Ti and the 4090 decide the same; the family's defaults stand where the constants rest on one
// lab card or the generation has no reference card. Volta (no int8 MMA) still prices finitely.
void decides_by_generation(const json& fixture) {
    const Bench a = nine_b(fixture, "RTX 4070 Ti");
    const Bench b = nine_b(fixture, "RTX 4090");
    const auto ra = ch::reference_of(a.card), rb = ch::reference_of(b.card);
    check(ra && rb, "Ada has a reference card");
    if (ra && rb) {
        ch::Task task;
        ch::Candidate c;
        c.kv = KvCacheStorage::Int8Group64;
        c.state = RecurrentStateStorage::FP16;
        const double sa = ch::request_seconds(a.model, ra->card, a.quants[0], c, task);
        const double sb = ch::request_seconds(b.model, rb->card, b.quants[0], c, task);
        check(sa == sb && sa > 0, "the 4070 Ti and the 4090 price a decision alike");
        check(ra->decides && ch::accuracy_decides("ada", "qwen3_5_text") &&
                  ch::decides_switches(a.model, a.card, {true, true}) &&
                  !ch::decides_switches(a.model, a.card, {false, true}),
              "Ada decides once the accuracy terms rest on two cards");
    }
    // Blackwell gb20x (#618 step 1): its KV factors rest on two cards (the RTX 5070 Ti and 5090),
    // its exact-route factor on none: the worth decides the KV codec, the activations keep fast.
    // Qwen3.5-9B's BF16, FP8 and K8V4 have two-card rows of their own there; its NVFP4 SM rate is
    // one card's (the two 6.4 % apart per reference SM-clock), and the MoE family's NVFP4, of
    // another geometry, carries Ada's factor: the worth takes neither (kv_decides), while on Ada
    // every codec decides.
    const Bench g = nine_b(fixture, "RTX 5070 Ti");
    const Bench h = nine_b(fixture, "RTX 5090");
    const auto rg = ch::reference_of(g.card), rh = ch::reference_of(h.card);
    check(rg && rh && rg->decides && !rg->activations_decide && ra && ra->activations_decide,
          "gb20x decides the KV codec, not the activations; Ada both");
    if (rg && rh) {
        ch::Task task;
        ch::Candidate c;
        c.kv            = KvCacheStorage::Int8Group64;
        c.state         = RecurrentStateStorage::FP16;
        const double sg = ch::request_seconds(g.model, rg->card, g.quants[0], c, task);
        const double sh = ch::request_seconds(h.model, rh->card, h.quants[0], c, task);
        check(sg == sh && sg > 0, "the 5070 Ti and the 5090 price a decision alike");
        check(ch::decides_switches(g.model, g.card, {true, true}), "gb20x's worth decides");
    }
    for (const KvCacheStorage kv :
         {KvCacheStorage::Int8Group64, KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Nvfp4Group16}) {
        check(ch::kv_decides(g.model, "gb20x", kv) == (kv != KvCacheStorage::Nvfp4Group16) &&
                  ch::kv_decides(a.model, "ada", kv),
              std::string("kv_decides ") + std::string(ninferno::kv_cache_storage_name(kv)));
    }
    {
        const ch::ModelFacts moe = model_of(fixture.at("models").at("Qwen3.6-35B-A3B"));
        check(!ch::kv_decides(moe, "gb20x", KvCacheStorage::Nvfp4Group16) &&
                  ch::kv_decides(moe, "gb20x", KvCacheStorage::Fp8KeyNvfp4Value) &&
                  ch::kv_decides(moe, "ada", KvCacheStorage::Nvfp4Group16),
              "the MoE family's NVFP4 does not decide on gb20x (Ada's factor carried)");
    }
    {
        ch::Task task;
        task.contexts           = {32768};
        const ch::Choice choice = ch::choose(g.model, g.card, g.quants[0],
                                             ch::allowed_axes(g.model, KvCacheStorage::Int8Group64),
                                             {}, task, oracle(g, 15e9));
        check(choice.by_worth && !choice.activations_by_worth && choice.chosen &&
                  chosen(choice).candidate.activations == ActivationMode::Fast &&
                  ch::kv_decides(g.model, "gb20x", chosen(choice).candidate.kv),
              "gb20x: the KV codec by worth among its own rows, the activations fast");
        bool activations_default = false;
        for (const ch::Alternative& alt : ch::explain(choice)) {
            if (alt.axis == ch::Axis::Activations && alt.value == "exact") {
                activations_default = alt.why_code == "default";
            }
        }
        bool nvfp4_default = false;
        for (const ch::Alternative& alt : ch::explain(choice)) {
            if (alt.axis == ch::Axis::Kv && alt.value == "nvfp4") {
                nvfp4_default = alt.why_code == "default";
            }
        }
        check(activations_default && nvfp4_default &&
                  choice.kv_not_decided ==
                      std::vector<KvCacheStorage>{KvCacheStorage::Nvfp4Group16},
              "gb20x: exact and NVFP4 are the family's default's alternatives, not weighed");
        if (choice.chosen) { std::cout << choice.why << '\n'; }
    }
    const Bench t = nine_b(fixture, "RTX 2080 Ti");
    check(!ch::reference_of(t.card), "Turing has no reference card");
    ch::CardFacts volta = board_card(0x1DB5);
    ch::Candidate c;
    const ch::Speed v = ch::speed_of(t.model, volta, t.quants[0], c, 1, 2048);
    check(v.decode_per_stream > 0 && std::isfinite(v.decode_per_stream) && v.prefill > 0,
          "Volta prices finitely");
    // An MoE's exact prefill: its int8 efficiency times the dense rows' BF16-to-int8 ratio, so the
    // exact route costs an MoE what it costs a dense model (about -72 % on Blackwell, not -35 %).
    const json& m35       = fixture.at("models").at("Qwen3.6-35B-A3B");
    const ch::ModelFacts moe = model_of(m35);
    const ch::CardFacts bw = card_of(fixture.at("cards").at("RTX 5070 Ti"));
    const ch::QuantOption q = quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS"));
    ch::Candidate fast, exact;
    exact.activations = ActivationMode::Exact;
    const double ratio = ch::speed_of(moe, bw, q, exact, 1, 2048).prefill / ch::speed_of(moe, bw, q, fast, 1, 2048).prefill;
    check(ratio > 0.25 && ratio < 0.32, "an MoE's exact prefill at the dense routes' ratio");
    // An MoE's prefill grows with the tokens each expert takes from a chunk: Qwen3.6-35B-A3B (256
    // experts, 8 a token) takes 32 a chunk of 1024 and 64 of 2048 (shares 0.813, 0.952).
    ch::Candidate wide;
    wide.chunk = 2048;
    const double grows = ch::speed_of(moe, bw, q, wide, 1, 2048).prefill / ch::speed_of(moe, bw, q, fast, 1, 2048).prefill;
    check(std::abs(grows - 0.952 / 0.813) < 1e-6, "an MoE's prefill at chunk 2048");
    const ch::Speed dense_wide = ch::speed_of(t.model, volta, t.quants[0], wide, 1, 2048);
    check(dense_wide.prefill == v.prefill, "a dense model's prefill does not take the chunk");
    // A 256-token prompt fills 256 of the chunk: 8 tokens per expert (the linears: attention apart).
    ch::ModelFacts linears        = moe;
    linears.full_attention_width  = 0;
    const double short_prompt = ch::speed_of(linears, bw, q, fast, 1, 2048, 256).prefill / ch::speed_of(linears, bw, q, fast, 1, 2048).prefill;
    check(std::abs(short_prompt - 0.409 / 0.813) < 1e-6, "a short prompt fills only its own tokens of the chunk");
    // A 1280-token prompt at chunk 1024: one whole chunk and a 256-token call, each at its rate.
    ch::Task t1280;
    t1280.prompt       = 1280;
    t1280.answer       = 1;
    const double whole = ch::speed_of(linears, bw, q, fast, 1, 1280, 1024).prefill;
    const double rest  = ch::speed_of(linears, bw, q, fast, 1, 1280, 256).prefill;
    const double dec   = ch::speed_of(linears, bw, q, fast, 1, 1280.5, 1280).decode_per_stream;
    check(std::abs(ch::single_request_seconds(linears, bw, q, fast, t1280, t1280.prompt) -
                   (1024 / whole + 256 / rest + 1 / dec)) < 1e-9,
          "the final partial chunk at its own rate");
}

// The quantization curve: a measured label takes its point; other bits per weight interpolate in
// log KLD; at or above Q8_0's bits it is the reference. The worth weighs the switches at a
// quantization where its family's terms rest on two cards (which quantization a listing serves is
// plan_quants', test_plan_dry_run).
void weighs_the_quantizations(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4090");
    ch::Candidate c;
    c.kv    = KvCacheStorage::Int8Group64;
    c.state = RecurrentStateStorage::FP16;
    const auto quant_term = [&](const ch::QuantOption& q) {
        for (const ch::Term& t : ch::accuracy_of(b.model, q, c, b.card).terms) {
            if (t.axis == ch::Axis::Quant) { return t; }
        }
        return ch::Term{};
    };
    ch::QuantOption q = b.quants[0];
    q.label           = "Q4_K_M";
    // #299: the measured point is the files measured; the same label at the same bits on files of
    // another (or unknown) identity takes the same value as a proxy, never as measured.
    const std::string measured_q4 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8";
    q.files = measured_q4;
    check(std::abs(quant_term(q).kld - 0.01925) < 1e-9 && quant_term(q).measured &&
              quant_term(q).evidence == ch::Evidence::Measured,
          "Q4_K_M, the files measured: its measured point");
    q.files.clear();
    check(std::abs(quant_term(q).kld - 0.01925) < 1e-9 && !quant_term(q).measured &&
              quant_term(q).evidence == ch::Evidence::Proxy &&
              quant_term(q).source.find("identity is unknown") != std::string::npos,
          "Q4_K_M, files unknown: the same value, a proxy");
    q.files = std::string(64, 'a'); // another repository's Q4_K_M at the same bits
    check(std::abs(quant_term(q).kld - 0.01925) < 1e-9 && quant_term(q).evidence == ch::Evidence::Proxy &&
              quant_term(q).source.find("not these files") != std::string::npos,
          "Q4_K_M, other files: the same value, a proxy");
    q.files = measured_q4;
    q.label   = "renamed"; // the files measured under another label: still their measurement
    check(std::abs(quant_term(q).kld - 0.01925) < 1e-9 && quant_term(q).measured, "the files decide, not the label");
    // #368: only the files' own measurement may decide between quantizations; a proxy of the same
    // value, another family and unknown bits are estimates to show.
    {
        using A = ch::QualityAuthority;
        ch::QuantOption a = b.quants[0];
        a.label           = "Q4_K_M";
        a.files           = measured_q4;
        check(ch::quant_authority(b.model, a) == A::Direct, "the files measured: direct");
        a.files.clear();
        check(ch::quant_authority(b.model, a) == A::Estimate, "files unknown: an estimate");
        a.files = std::string(64, 'a');
        check(ch::quant_authority(b.model, a) == A::Estimate, "other files, the same value: an estimate");
        a.files = measured_q4;
        a.bits_per_weight += 0.3;
        check(ch::quant_authority(b.model, a) == A::Estimate, "the files at other bits: an estimate");
        a.bits_per_weight = 0;
        check(ch::quant_authority(b.model, a) == A::Unknown, "unknown bits: unknown");
        a.bits_per_weight = b.quants[0].bits_per_weight;
        ch::ModelFacts other = b.model;
        other.family         = "gemma4_text";
        check(ch::quant_authority(other, a) == A::Estimate, "another family's curve: an estimate");
    }
    q.bits_per_weight = 5.2; // the same files counted at other bits: the curve at those bits, as before
    check(!quant_term(q).measured && quant_term(q).evidence == ch::Evidence::Interpolated &&
              quant_term(q).kld < 0.01925 && quant_term(q).kld > 0.01828,
          "the measured files at other bits: interpolated");
    q.bits_per_weight = b.quants[0].bits_per_weight;
    q.label = "Q4_K_M";
    q.files.clear();
    q.bits_per_weight = 4.9; // another model's Q4_K_M: interpolated, not the 9B's point
    check(!quant_term(q).measured && quant_term(q).kld > 0.01925 &&
              quant_term(q).evidence == ch::Evidence::Interpolated,
          "a label at other bits: interpolated");
    q.label           = "custom";
    q.bits_per_weight = 5.5;
    const double mid  = std::exp(std::log(0.01828) + (5.5 - 5.321) / (5.674 - 5.321) * (std::log(0.01103) - std::log(0.01828)));
    check(std::abs(quant_term(q).kld - mid) < 1e-9 && !quant_term(q).measured, "5.5 bits: interpolated");
    q.bits_per_weight = 9.0;
    check(quant_term(q).kld == 0, "above Q8_0's bits: the reference");
    q.bits_per_weight = (6.665 + 8.516) / 2; // above Q6_K: a straight line to 0 at Q8_0's bits
    check(std::abs(quant_term(q).kld - 0.00629 / 2) < 1e-9, "between Q6_K and Q8_0: halfway to 0");
    q.label           = "Q8_0";
    q.bits_per_weight = 8.3; // another model's Q8_0 (other bits): its own reference
    check(quant_term(q).kld == 0 && !quant_term(q).measured, "any model's Q8_0: the reference");
    q.bits_per_weight = 8.516;
    q.files         = "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4";
    check(quant_term(q).kld == 0 && quant_term(q).measured, "the reference's own files: measured (0 against itself)");
    q.files.clear();
    check(quant_term(q).kld == 0 && quant_term(q).evidence == ch::Evidence::Reference, "another Q8_0 at its bits: the reference");
    q.bits_per_weight = 8.3;
    check(quant_term(q).kld == 0 && quant_term(q).evidence == ch::Evidence::Reference &&
              quant_term(q).source.find("the reference") == 0,
          "another model's Q8_0: the reference, and its source says so");
    ch::Task task;
    const ch::Gates open{true, true};
    const ch::Choice choice = ch::choose(b.model, b.card, b.quants[0],
                                         ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task,
                                         oracle(b, 20e9), open);
    check(choice.chosen && choice.by_worth && choice.why.find("best worth on the ada reference card") != std::string::npos,
          "Ada: the switches by worth");
    bool fp8 = false;
    for (const ch::Alternative& a : ch::explain(choice)) {
        if (a.axis == ch::Axis::Kv && a.value == "fp8") {
            check(a.why_code == "lower_worth", "FP8 KV: the same cost, less accurate");
            fp8 = true;
        }
    }
    check(fp8, "FP8 KV explained");
    // The constants' own gates: the worth decides once the accuracy terms rest on two cards.
    const ch::Choice gated = ch::choose(b.model, b.card, b.quants[0],
                                        ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task,
                                        oracle(b, 20e9));
    check(gated.by_worth == ch::accuracy_decides("ada", "qwen3_5_text"),
          "the worth decides only with the accuracy terms on two cards");
    check(ch::weighs_quantizations(b.model, b.card, open) &&
              !ch::weighs_quantizations(nine_b(fixture, "RTX 2080 Ti").model, nine_b(fixture, "RTX 2080 Ti").card, open),
          "Ada weighs the quantizations (Qwen 3.5 dense's curve); Turing has no reference card");
    // A family the terms were not measured on keeps its defaults.
    Bench gemma = b;
    gemma.model.family = "gemma4_text";
    const ch::Choice other = ch::choose(gemma.model, gemma.card, gemma.quants[0],
                                        ch::allowed_axes(gemma.model, KvCacheStorage::Fp8E4M3Row256), {}, task,
                                        oracle(gemma, 20e9), open);
    check(!other.by_worth && other.why.find("no accuracy terms were measured on gemma4_text") != std::string::npos,
          "another family: its defaults");
    // A dense decoder (its own terms, two cards): the worth decides its KV codec and activations,
    // while a listing's quantization stays the rule's (the quantization curve is Qwen 3.5 dense's).
    Bench dense = b;
    dense.model.family = "dense_decoder";
    const ch::Choice dense_choice = ch::choose(dense.model, dense.card, dense.quants[0],
                                               ch::allowed_axes(dense.model, KvCacheStorage::Fp8E4M3Row256), {},
                                               task, oracle(dense, 20e9), open);
    check(ch::accuracy_decides("ada", "dense_decoder") && dense_choice.by_worth,
          "a dense decoder decides by worth on its own terms");
    check(!ch::weighs_quantizations(dense.model, dense.card, open),
          "a dense decoder keeps the quant rule's largest (no curve of its own)");
}

// The chunk of an MoE: on Ada a 2048-token prompt fills a chunk of 2048 (4096 adds nothing for it,
// and the tie goes to the narrower: more context), by speed alone (a chunk has no accuracy term, so
// the accuracy gate does not hold it); on Blackwell (no reference card) the default, with the chunk
// the carried curve estimates; on a generation without such rows the default.
void chooses_the_chunk(const json& fixture) {
    const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
    Bench b{model_of(m35), card_of(fixture.at("cards").at("RTX 4090")), {quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS"))}};
    b.model.recurrent = true;
    ch::Task task;
    task.contexts  = {2560}; // an ask whose request mix is the 2048-token prompt alone
    const auto own = [&b](const ch::Candidate& c) -> std::uint32_t {
        // The workspace grows with the chunk: a wider one holds a little less context.
        const double room = 20e9 - b.quants[0].weight_bytes - 4e5 * c.chunk;
        return room > 0 ? static_cast<std::uint32_t>(room / b.model.kv_bytes_per_token(c.kv)) : 0;
    };
    const ch::Choice ada = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task, own);
    // A chunk has no accuracy term: the resolution rule never moves it (#301, DECISIONS 75).
    check(ada.chosen && chosen(ada).candidate.chunk == 2048 && ada.chunk_by_speed &&
              !ada.resolved_within_resolution,
          "Ada, an MoE: chunk 2048 for a 2048-token prompt");
    int chunks = 0;
    for (const ch::Alternative& a : ch::explain(ada)) {
        if (a.axis == ch::Axis::Chunk && a.value == "4096") {
            check(a.why_code == "tie_context", "chunk 4096: as fast for this prompt, less context");
            ++chunks;
        }
        if (a.axis == ch::Axis::Chunk && a.value == "1024") {
            // Slower only: a loss outright where it is slower by more than the resolution, else
            // indistinguishable within it (#301).
            const double slower = std::log(a.scored->decision_seconds / chosen(ada).decision_seconds);
            check(a.why_code == (slower > ada.resolution ? "lower_worth" : "within_resolution"),
                  "chunk 1024: slower");
            ++chunks;
        }
    }
    check(chunks == 2, "both other chunks explained");
    ch::Pins wide;
    wide.chunk = 4096;
    const ch::Choice pinned = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), wide, task, own);
    check(pinned.chosen && chosen(pinned).candidate.chunk == 4096 && !pinned.chunk_by_speed, "a pinned chunk stays");
    b.card = card_of(fixture.at("cards").at("RTX 5070 Ti"));
    const ch::Choice bw = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task, own);
    // Blackwell has no measured chunk sweep: its own MoE prefill rate (two lab cards) with the
    // curve's slope carried estimates the chunk, and the pick stays the default.
    check(bw.chosen && !bw.chunk_by_speed && bw.chunk_basis == ch::ChunkBasis::CarriedCurve &&
              chosen(bw).candidate.chunk == 1024 && bw.chunk_estimate == 2048,
          "Blackwell: the default chunk, 2048 estimated for a 2048-token prompt");
    check(ada.chunk_basis == ch::ChunkBasis::Own && ada.chunk_estimate == 0,
          "Ada: the chunk rests on its own rows");
    // A generation without an MoE prefill row of two cards keeps the default chunk.
    b.card = card_of(fixture.at("cards").at("RTX 2080 Ti"));
    const ch::Choice tu =
        ch::choose(b.model, b.card, b.quants[0],
                   ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task, own);
    check(tu.chosen && !tu.chunk_by_speed && tu.chunk_basis == ch::ChunkBasis::Transferred &&
              chosen(tu).candidate.chunk == 1024,
          "Turing: the default chunk (transferred)");
}

// The worth prices the benchmark's request mix (#507, DECISIONS 73): a 32K ask puts a 24K prompt
// into it, where INT8 KV and a half-precision state win on the Ada reference card (their bytes
// cost in proportion to the depth); an ask of 2560 tokens has no long request, and BF16 / FP32
// stays.
void prices_the_request_mix(const json& fixture) {
    const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
    Bench b{model_of(m35),
            card_of(fixture.at("cards").at("RTX 4090")),
            {quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS"))}};
    b.model.recurrent = true;
    const auto own    = [&b](const ch::Candidate& c) -> std::uint32_t {
        const double room = 20e9 - b.quants[0].weight_bytes - 4e5 * c.chunk;
        return room > 0 ? static_cast<std::uint32_t>(room / b.model.kv_bytes_per_token(c.kv)) : 0;
    };
    ch::Task asked;
    asked.contexts = {32768};
    ch::Task middle;
    middle.contexts = {16384};
    check(middle.long_prompt() == 16384 - ch::kRequestAnswer,
          "a 16K ask: the long request's prompt is the context less the answer");
    check(asked.long_prompt() == ch::kLongPrompt,
          "a 32K ask: the long request stops at the benchmark's 24K prompt");
    ch::Task small;
    small.contexts = {2560};
    check(small.long_prompt() == 0, "a 2560-token ask has no long request");
    const auto allowed   = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
    const ch::Choice big = ch::choose(b.model, b.card, b.quants[0], allowed, {}, asked, own);
    check(big.chosen && chosen(big).candidate.kv == KvCacheStorage::Int8Group64 &&
              chosen(big).candidate.state == RecurrentStateStorage::FP16,
          "a 32K ask: KV int8, state f16");
    check(big.chosen && chosen(big).candidate.chunk == 4096,
          "a 32K ask: the 24K prompt fills the 4096 chunk");
    const ch::Choice little = ch::choose(b.model, b.card, b.quants[0], allowed, {}, small, own);
    check(little.chosen && chosen(little).candidate.kv == KvCacheStorage::BFloat16 &&
              chosen(little).candidate.state == RecurrentStateStorage::FP32,
          "a 2560-token ask: KV bf16, state f32");
    // The mix is the geometric mean of the two requests, weighted by the benchmark's counts.
    const ch::Candidate c = chosen(big).candidate;
    const double shorter =
        ch::single_request_seconds(b.model, b.card, b.quants[0], c, asked, asked.prompt);
    const double longer =
        ch::single_request_seconds(b.model, b.card, b.quants[0], c, asked, asked.long_prompt());
    const double expected = std::exp(
        (ch::kMixShortRequests * std::log(shorter) + ch::kMixLongRequests * std::log(longer)) /
        (ch::kMixShortRequests + ch::kMixLongRequests));
    check(std::abs(ch::request_seconds(b.model, b.card, b.quants[0], c, asked) - expected) <
              1e-12 * expected,
          "the request mix is the weighted geometric mean of the short and the long request");
    check(big.why.find("request mix: 20 of 2048+512, 6 of 24576+512 tokens") != std::string::npos,
          "the reason names the mix");
    // Auto's steps: where only 8K fits, the long request is priced at 8192 less the answer, the
    // step being tried, never at the 24K of the largest step.
    ch::Task automatic;
    automatic.contexts     = {32768, 16384, 8192};
    const auto little_room = [](const ch::Candidate&) -> std::uint32_t {
        return 8192;
    }; // every candidate holds 8K only
    const ch::Choice steps =
        ch::choose(b.model, b.card, b.quants[0], allowed, {}, automatic, little_room);
    ch::Task pinned_8k;
    pinned_8k.contexts = {8192};
    const ch::Choice eight =
        ch::choose(b.model, b.card, b.quants[0], allowed, {}, pinned_8k, little_room);
    check(steps.chosen && steps.context == 8192, "automatic steps settle on the 8K that fits");
    check(steps.chosen && eight.chosen &&
              chosen(steps).decision_seconds == chosen(eight).decision_seconds &&
              chosen(steps).candidate.kv == chosen(eight).candidate.kv,
          "the 8K step prices the mix as an 8192 ask does");
    check(steps.why.find("6 of 7680+512 tokens") != std::string::npos &&
              steps.why.find("24576") == std::string::npos,
          "the reason prices the long request at 8192 less the answer");
}

// A coarse quantization does not make a switch's error free: the quantization's term adds on top
// of the switches' root sum square, so on Ada a ~3-bit model still keeps INT8 KV over K8V4.
void coarse_quant_keeps_kv(const json& fixture) {
    Bench b = nine_b(fixture, "RTX 4090");
    b.quants = {b.quants[2]}; // IQ3_XXS-like, 3.1 bits: a large interpolated quantization term
    ch::Task task;
    task.contexts = {8192};
    const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    {}, task, oracle(b, 20e9));
    check(c.chosen && chosen(c).candidate.kv == KvCacheStorage::Int8Group64, "a coarse quantization keeps INT8 KV");
}

// Auto's steps at a quantization: the first own-context step a candidate holds.
void steps_before_the_floor(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4070 Ti");
    ch::Task task;
    task.contexts = {32768, 16384, 8192};
    // Q4_K_M leaves room for 20K tokens of INT8 KV: 16K.
    const double budget = b.quants[0].weight_bytes + 20000.0 * b.model.kv_bytes_per_token(KvCacheStorage::Int8Group64);
    // KV pinned to INT8: where the worth decides, a narrower codec would itself reach 32K at
    // Q4_K_M (the KV codec is a memory lever too).
    ch::Pins pins;
    pins.kv = KvCacheStorage::Int8Group64;
    const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    pins, task, oracle(b, budget));
    check(c.chosen && c.context == 16384 && c.why.find("every stream holds 16384 tokens") != std::string::npos,
          "16K at Q4_K_M: Auto's second step");
}

// The int8 activations' term is a model constant (since ggml's MMQ prefill sums its blocks from
// the codes): the same on every generation. With it Ada decides, and fast's 0.004 KLD is worth less
// than exact's +34 % time: Qwen3.5-9B Q4_K_M keeps fast.
void activations_by_generation(const json& fixture) {
    ch::Candidate c;
    c.kv    = KvCacheStorage::Int8Group64;
    c.state = RecurrentStateStorage::FP16;
    const auto act = [&](const char* card) {
        const Bench b = nine_b(fixture, card);
        for (const ch::Term& t : ch::accuracy_of(b.model, b.quants[0], c, b.card).terms) {
            if (t.axis == ch::Axis::Activations) { return t; }
        }
        return ch::Term{};
    };
    for (const char* card : {"RTX 4090", "RTX 5070 Ti", "RTX 2080 Ti"}) {
        check(std::abs(act(card).kld - 0.00394) < 1e-9 && act(card).measured, std::string(card) + ": the model constant");
    }
    check(ch::accuracy_decides("ada", "qwen3_5_text"), "Ada: every term the worth decides by rests on two cards");
    const Bench b = nine_b(fixture, "RTX 4090");
    ch::Task task;
    const ch::Choice choice = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                         {}, task, oracle(b, 9.6e9));
    check(choice.by_worth && choice.chosen && chosen(choice).candidate.activations == ActivationMode::Fast,
          "Ada decides: fast over exact");
    if (choice.chosen) { std::cout << choice.why << '\n'; }
}

// Qwen 3.5 MoE has switch terms of its own (INT8 KV 0.0033, 3.4 times dense's) on two cards and no
// quantization curve: on Ada the worth decides its KV codec and activations, the quant rule its
// quantization (weighs_quantizations false).
void moe_family(const json& fixture) {
    const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
    Bench b{model_of(m35), card_of(fixture.at("cards").at("RTX 4090")), {}};
    b.quants.push_back(quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS")));
    ch::QuantOption larger = b.quants[0];
    larger.label           = "Q4_K_M";
    larger.bits_per_weight = 4.9;
    larger.weight_bytes    = b.quants[0].weight_bytes * 4.9 / b.quants[0].bits_per_weight;
    b.quants.push_back(larger);
    ch::Candidate c;
    c.kv = KvCacheStorage::Int8Group64;
    double kv_term = -1;
    bool transferred = true;
    for (const ch::Term& t : ch::accuracy_of(b.model, b.quants[0], c, b.card).terms) {
        if (t.axis == ch::Axis::Kv) {
            kv_term     = t.kld;
            transferred = t.transferred;
        }
    }
    check(std::abs(kv_term - 0.00329) < 1e-9 && !transferred, "MoE: its own INT8 KV term");
    check(ch::accuracy_decides("ada", "qwen3_5_moe_text"), "MoE: its switch terms on two cards");
    ch::Task task;
    const ch::Choice choice = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                         {}, task, oracle(b, 40e9));
    check(choice.by_worth && choice.chosen && !ch::weighs_quantizations(b.model, b.card),
          "MoE on Ada: the switches by worth, the quantization by the rule");
    if (choice.chosen) { std::cout << choice.why << '\n'; }
}

// The MoE step reads the op's route (#616 step 7, ModelFacts::moe_routes): the pair its
// kMoeRoutedEta rows were measured on (R8 gate/up over the register-MMA down) prices and types the
// step as not asking does; another pair (the register MMA on both stages, as the op runs gpt-oss's
// MXFP4 banks from five rows) has no measured rows: the same price, transferred past one row; a
// share of the bytes on an unknown pair marks it too. One row reads at kDramEta whatever the pair.
void moe_step_reads_the_route(const json& fixture) {
    const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
    int lifts       = 0; // the steps an unmeasured pair turned from Own to Transferred
    for (const char* card_name : {"RTX 4090", "RTX 5090"}) {
        const ch::CardFacts card = card_of(fixture.at("cards").at(card_name));
        const ch::QuantOption q  = quant_of("UD-Q3_K_S", m35.at("quants").at("UD-Q3_K_S"));
        ch::Candidate c;
        c.kv = KvCacheStorage::Int8Group64;
        // The fixture's model without its layers is a transferred estimate whatever the route: the
        // trunk's 40 layers make it the measured generations' own.
        ch::ModelFacts plain = model_at(m35, "UD-Q3_K_S");
        plain.layers         = 40;
        const auto with      = [&](std::vector<ch::MoeStepRoute> routes) {
            ch::ModelFacts m = plain;
            m.moe_routes     = [routes](std::uint32_t, ActivationMode) { return routes; };
            return m;
        };
        const ch::ModelFacts measured =
            with({{ch::ExpertKernel::R8, ch::ExpertKernel::RegisterMma, 1}});
        const ch::ModelFacts rmma =
            with({{ch::ExpertKernel::RegisterMma, ch::ExpertKernel::RegisterMma, 1}});
        const ch::ModelFacts mixed =
            with({{ch::ExpertKernel::R8, ch::ExpertKernel::RegisterMma, 3},
                  {ch::ExpertKernel::Unknown, ch::ExpertKernel::Unknown, 1}});
        // No route query to ask (ninferno-ui, #766): the measured pair's price, never Own past
        // one row.
        ch::ModelFacts unqueried      = plain;
        unqueried.moe_route_unqueried = true;
        for (const std::uint32_t rows : {1U, 2U, 4U, 8U}) {
            const ch::Speed a    = ch::speed_of(plain, card, q, c, rows, 1536, 0);
            const ch::Speed b    = ch::speed_of(measured, card, q, c, rows, 1536, 0);
            const ch::Speed r    = ch::speed_of(rmma, card, q, c, rows, 1536, 0);
            const ch::Speed x    = ch::speed_of(mixed, card, q, c, rows, 1536, 0);
            const ch::Speed u    = ch::speed_of(unqueried, card, q, c, rows, 1536, 0);
            const std::string at = std::string(card_name) + " x" + std::to_string(rows);
            check(b.decode_per_stream == a.decode_per_stream && b.basis == a.basis && b.why.empty(),
                  "MoE step: the measured pair is the price of not asking (" + at + ")");
            check(u.decode_per_stream == a.decode_per_stream,
                  "MoE step: unqueried, the measured pair's share (" + at + ")");
            check(rows > 1
                      ? u.basis == ch::SpeedBasis::Transferred && u.why == "MoE route not queried"
                      : u.basis == a.basis && u.why.empty(),
                  "MoE step: unqueried, transferred past one row and says why (" + at + ")");
            check(rows == 1 || x.why == "MoE route not answered",
                  "MoE step: a kind with no answer says why (" + at + ")");
            check(std::abs(r.decode_per_stream / a.decode_per_stream - 1) < 1e-12 &&
                      std::abs(x.decode_per_stream / a.decode_per_stream - 1) < 1e-12,
                  "MoE step: an unmeasured pair keeps the measured pair's share (" + at + ")");
            const bool lifted = rows > 1 && a.basis == ch::SpeedBasis::Own;
            lifts += lifted ? 1 : 0;
            check(!lifted || (r.basis == ch::SpeedBasis::Transferred &&
                              x.basis == ch::SpeedBasis::Transferred),
                  "MoE step: an unmeasured pair past one row is transferred (" + at + ")");
            check(rows > 1 || (r.basis == a.basis && x.basis == a.basis),
                  "MoE step: one row reads at kDramEta on every pair (" + at + ")");
        }
    }
    check(lifts >= 3, "MoE step: the Own steps of several rows are the measured pair's alone");
}

// Qwen3.6-35B-A3B's linears a prefill token runs (the routed experts' marked), as the census reads
// them.
std::vector<ch::LinearShape> qwen36_moe_linears() {
    std::vector<ch::LinearShape> out;
    for (int i = 0; i < 10; ++i) {
        out.push_back({9216, 2048, 1, false});
        out.push_back({2048, 4096, 1, false});
    }
    out.push_back({12288, 2048, 30, false});
    out.push_back({2048, 4096, 30, false});
    out.push_back({1024, 2048, 40, false});
    out.push_back({2048, 512, 40, false});
    out.push_back({1024, 2048, 320, true});
    out.push_back({2048, 512, 320, true});
    return out;
}

// Contradictory pins say so instead of "nothing fits".
// An MoE's prefill linears by their tiles (DECISIONS 66): Qwen3.6-35B-A3B UD-Q4_K_M on the RTX 4090,
// one 1819-token chunk; the census (nsys, no PDL) put the experts at 50.5 ms, the int8 calls beside
// them at 30.0 and the routing's gather and scatter at 5.9: 86.4 ms.
void prices_moe_prefill_tiles() {
    ch::ModelFacts m;
    m.family               = "qwen3_5_moe_text";
    m.experts              = 256;
    m.experts_per_token    = 8;
    m.routed_expert_params = 34e9;
    m.linears              = qwen36_moe_linears();
    ch::CardFacts rtx4090 = board_card(0x2684);
    ch::QuantOption q;
    q.expert_banks = ch::ExpertBanks::Int8;
    q.rest_weights = ch::WeightClass::Q8; // UD-Q4_K_M's linears beside the experts: Q8_0
    const double t = ch::prefill_linears_seconds(m, rtx4090, q, 1819);
    std::cout << "MoE prefill tiles, Qwen3.6 UD-Q4_K_M RTX 4090 1819 tokens: " << t * 1e3 << " ms against 86.4\n";
    check(std::abs(t / 0.0864 - 1) < 0.12, "an MoE's prefill linears by their tiles within 12 % of the census");
    ch::CardFacts v100 = board_card(0x1DB5);
    check(ch::prefill_linears_seconds(m, v100, q, 1819) == 0, "no int8 tile row on Volta: unknown");
    // A route on a kept format the file does not tell (Gemma 4's GGUF MXFP4 experts, #769): the
    // same price, transferred.
    {
        ch::QuantOption assumed      = q;
        assumed.expert_banks_assumed = true;
        bool was = false, is = false;
        const double known = ch::prefill_linears_seconds(m, rtx4090, q, 1819, &was);
        const double guess = ch::prefill_linears_seconds(m, rtx4090, assumed, 1819, &is);
        check(guess == known && !was && is, "assumed banks: the same price, transferred");
    }
    q.expert_banks = ch::ExpertBanks::Unknown;
    check(ch::prefill_linears_seconds(m, rtx4090, q, 1819) == 0, "unknown banks: unknown");
}

// The banks' GEMM follows the activations a candidate runs (#769): under --activations int8 the
// start runs every routed bank at AllowA8 (models::effective_policy), so the prefill prices the
// banks by their formats (QuantOption::expert_banks_int8); fast activations keep the recorded
// policy's route; with no int8 route given, both read expert_banks.
void moe_prefill_follows_the_activations(const json& fixture) {
    const json& m35          = fixture.at("models").at("Qwen3.6-35B-A3B");
    const ch::CardFacts card = card_of(fixture.at("cards").at("RTX 4090"));
    ch::ModelFacts model     = model_at(m35, "UD-Q3_K_S");
    // The banks' route reaches the prefill through its tiles (prefill_linears_seconds): the
    // linears by shape, and the trunk's layers the small kernels count.
    model.linears           = qwen36_moe_linears();
    model.layers            = 40;
    model.hidden            = 2048;
    const ch::QuantOption q = quant_of("UD-Q3_K_S", m35.at("quants").at("UD-Q3_K_S"));
    ch::Candidate fast;
    fast.kv            = KvCacheStorage::Int8Group64;
    fast.chunk         = 2048; // the chunk the tiles price
    ch::Candidate int8 = fast;
    int8.activations   = ActivationMode::Int8;
    const auto prefill = [&](ch::ExpertBanks banks, std::optional<ch::ExpertBanks> under_int8,
                             const ch::Candidate& c) {
        ch::QuantOption o = q;
        o.rest_weights    = ch::WeightClass::Q8; // the linears beside the experts on the int8 tiles
        o.expert_banks    = banks;
        o.expert_banks_int8 = under_int8;
        return ch::speed_of(model, card, o, c, 1, 1536, 1536).prefill;
    };
    const double int8_banks = prefill(ch::ExpertBanks::Int8, std::nullopt, fast);
    const double a16_banks  = prefill(ch::ExpertBanks::A16, std::nullopt, fast);
    check(int8_banks > 0 && a16_banks > 0 && int8_banks != a16_banks,
          "MoE prefill: the banks' route prices the prefill");
    check(prefill(ch::ExpertBanks::A16, ch::ExpertBanks::Int8, int8) ==
              prefill(ch::ExpertBanks::Int8, std::nullopt, int8),
          "MoE prefill under int8: the formats' route");
    check(prefill(ch::ExpertBanks::A16, ch::ExpertBanks::Int8, fast) == a16_banks,
          "MoE prefill under fast activations: the recorded policy's route");
    check(prefill(ch::ExpertBanks::A16, std::nullopt, int8) !=
              prefill(ch::ExpertBanks::Int8, std::nullopt, int8),
          "MoE prefill: no int8 route given, expert_banks under int8 too");
    // An assumed route (Gemma 4's GGUF MXFP4 experts) where the tiles do not price it: linears
    // beside the banks not Q8, so prefill_linears_seconds gives 0 and the share stands. The
    // estimate is Transferred there too, where the same banks not assumed keep their basis (codex
    // on #792).
    {
        ch::QuantOption k       = q;
        k.rest_weights          = ch::WeightClass::KQuant;
        k.expert_banks          = ch::ExpertBanks::Int8;
        ch::QuantOption kk      = k;
        kk.expert_banks_assumed = true;
        const ch::Speed known   = ch::speed_of(model, card, k, fast, 1, 1536, 1536);
        const ch::Speed assumed = ch::speed_of(model, card, kk, fast, 1, 1536, 1536);
        check(ch::prefill_linears_seconds(model, card, kk, 2048) == 0 &&
                  known.basis == ch::SpeedBasis::Own,
              "MoE prefill, share path: K-quant linears, no tile price, the basis own");
        check(assumed.basis == ch::SpeedBasis::Transferred && assumed.prefill == known.prefill,
              "MoE prefill, share path: an assumed route Transferred at the same price");
    }
}

// A dense model's Q8 linears by the same tiles: Qwen3.5 9B Q8_0, one 1819-token prompt in chunks
// of 1024 and 795 (the token-major grid, PR #249); the census's int8 kernel time (nsys) on the
// RTX 3060, 4060 Ti, 4070 Ti, 4090 and 5070 Ti.
void prices_dense_prefill_tiles() {
    ch::ModelFacts m;
    m.family = "qwen3_5_text";
    for (int i = 0; i < 8; ++i) {
        m.linears.push_back({10240, 4096, 1, false});
        m.linears.push_back({4096, 4096, 1, false});
    }
    m.linears.push_back({12288, 4096, 24, false});
    m.linears.push_back({4096, 4096, 24, false});
    m.linears.push_back({24576, 4096, 32, false});
    m.linears.push_back({4096, 12288, 32, false});
    ch::QuantOption q;
    q.weights = ch::WeightClass::Q8;
    struct Census {
        ch::CardFacts card;
        double ms;
    };

    const Census rows[] = {
        {board_card(0x2504), 767.4}, {board_card(0x2805), 440.4}, {board_card(0x2782), 245.9},
        {board_card(0x2684), 127.4}, {board_card(0x2C05), 212.0},
    };
    double worst = 0;
    for (const Census& r : rows) {
        const double t = ch::prefill_linears_seconds(m, r.card, q, 1024) + ch::prefill_linears_seconds(m, r.card, q, 795);
        worst          = std::max(worst, std::abs(t * 1e3 / r.ms - 1));
        std::cout << "dense Q8 tiles, " << r.card.sms << " SMs: " << t * 1e3 << " ms against " << r.ms << "\n";
    }
    check(worst < 0.10, "a dense model's Q8 linears by their tiles within 10 % of the census");
    // A grid of no more CTAs than SMs runs one a SM alone (kLoneTile): Qwen3.5 9B's b/a projection
    // (N 64, K 4096) over a 1024-token chunk, 16 CTAs, against the census on six cards.
    ch::ModelFacts one;
    one.linears.push_back({64, 4096, 1, false});
    struct Lone {
        ch::CardFacts card;
        double us;
    };

    const Lone lone[] = {
        {rows[0].card, 39.3}, {rows[1].card, 28.4}, {rows[2].card, 26.1},
        {rows[3].card, 27.5}, {rows[4].card, 25.5}, {board_card(0x2B85), 27.1},
    };
    double lone_worst = 0;
    for (const Lone& l : lone) {
        const double t = ch::prefill_linears_seconds(one, l.card, q, 1024);
        lone_worst     = std::max(lone_worst, std::abs(t * 1e6 / l.us - 1));
    }
    check(lone_worst < 0.06, "an underfilled grid: a CTA alone on its SM, within 6 % on six cards");
    q.weights = ch::WeightClass::KQuant;
    check(ch::prefill_linears_seconds(m, rows[3].card, q, 1024) == 0, "a K-quant without a K format in its label: unknown");
    // ggml's matrix kernel by its tiles (DECISIONS 69): Qwen3.5 9B Q6_K's 512-token calls on the RTX
    // 4060 Ti, 12288 x 4096 (a tile a CTA in 12 rounds of its 34 SMs: the census's 998.6 us) and
    // 4096 x 12288 (four rounds: 994.8 us, from the census's mixture with the K-4096 calls of that
    // grid), the epilogue apart (the DRAM rate set out of reach); a gated pair as two grids.
    {
        ch::QuantOption q6 = q;
        q6.label           = "Q6_K";
        ch::CardFacts ti    = board_card(0x2805);
        ti.dram_bytes_per_s = 288e15; // out of reach: the tiles alone
        const auto call = [&](double n, double k, bool gated) {
            ch::ModelFacts one = m;
            one.linears        = {{n, k, 1, false, gated}};
            return ch::prefill_linears_seconds(one, ti, q6, 512) * 1e6;
        };
        const double e1 = call(12288, 4096, false) / 998.6 - 1, e2 = call(4096, 12288, false) / 994.8 - 1;
        std::cout << "MMQ tiles, RTX 4060 Ti: " << e1 * 100 << " %, " << e2 * 100 << " %\n";
        check(std::abs(e1) < 0.05 && std::abs(e2) < 0.05, "MMQ's tiles in rounds within 5 % of the census's calls");
        // Volta (DECISIONS 70): the Quadro GV100's CUTLASS route, a 7K prompt's linears in 1024-token
        // chunks against the census's GEMM, decode, cast and epilogue (Q8_0 1842 ms, Q6_K 1882).
        {
            const ch::CardFacts gv100 = board_card(0x1DBA);
            ch::ModelFacts qwen = m;
            qwen.hidden         = 4096;
            qwen.linears        = {{10240, 4096, 8}, {4096, 4096, 8}, {12288, 4096, 24}, {4096, 4096, 24},
                                   {24576, 4096, 32, false, true}, {4096, 12288, 32}};
            const auto prompt = [&](const ch::QuantOption& quant) {
                double s = 0;
                for (double done = 0; done < 7238; done += 1024) {
                    s += ch::prefill_linears_seconds(qwen, gv100, quant, std::min(1024.0, 7238 - done));
                }
                return s * 1e3;
            };
            ch::QuantOption q8v = q6;
            q8v.weights         = ch::WeightClass::Q8;
            q8v.label           = "Q8_0";
            const double v8 = prompt(q8v) / 1842 - 1, v6 = prompt(q6) / 1882 - 1;
            std::cout << "Volta linears: Q8_0 " << v8 * 100 << " %, Q6_K " << v6 * 100 << " %\n";
            check(std::abs(v8) < 0.06 && std::abs(v6) < 0.06, "Volta's CUTLASS route within 6 % of the census");
            // Off the route: sm_72 (no CUTLASS in its tree), and a tiled format's short call on slow DRAM.
            ch::CardFacts xavier = gv100, titan_v = gv100;
            xavier.minor             = 2;
            titan_v.dram_bytes_per_s = 653e9;
            check(ch::prefill_linears_seconds(qwen, xavier, q6, 1024) == 0 &&
                      ch::prefill_linears_seconds(qwen, titan_v, q8v, 128) == 0 &&
                      ch::prefill_linears_seconds(qwen, gv100, q6, 16) == 0 &&
                      ch::prefill_linears_seconds(qwen, titan_v, q8v, 256) > 0 &&
                      ch::prefill_linears_seconds(qwen, titan_v, q6, 128) > 0,
                  "Volta's estimate keeps CUTLASS where the runtime takes it");
        }
        // Where MMQ does not pay: the RTX 5070 Ti's producer-warp Q6_K tiles, the gate and up's 1024-token
        // call (384 x 8 tiles in 44 rounds of its 70 SMs: the census's 2131.1 us); a call under 128
        // tokens keeps the share.
        const ch::CardFacts r5070 = board_card(0x2C05);
        ch::ModelFacts gate_up = m;
        gate_up.linears        = {{24576, 4096, 1, false, true}};
        const double ws        = ch::prefill_linears_seconds(gate_up, r5070, q6, 1024) * 1e6 / 2131.1 - 1;
        std::cout << "W4A8 tiles, RTX 5070 Ti: " << ws * 100 << " %\n";
        check(std::abs(ws) < 0.05 && ch::prefill_linears_seconds(gate_up, r5070, q6, 100) == 0,
              "the W4A8 tiles where MMQ does not pay, from 128 tokens");
        check(std::abs(call(24576, 4096, true) / (2 * call(12288, 4096, false)) - 1) < 1e-3,
              "a gated pair: two grids of half its rows");
    }
}

// The small kernels beside a prefill's linears (DECISIONS 66): Qwen3.5 9B Q8_0's 1819-token prompt,
// the census outside the linears, attention and the tail calls, seven cards (the scan's SM-clocks
// and the rest's bytes are read from these points, a generation's mean: within a quarter of each).
void prices_prefill_small_kernels() {
    ch::ModelFacts m;
    m.family                   = "qwen3_5_text";
    m.layers                   = 32;
    m.hidden                   = 4096;
    m.recurrent_state_elements = 24.0 * 32 * 128 * 128;
    struct Census {
        ch::CardFacts card;
        double ms;
    };

    const Census rows[] = {
        {board_card(0x2504), 63.8}, {board_card(0x2805), 35.3}, {board_card(0x2782), 21.6},
        {board_card(0x2684), 11.5}, {board_card(0x2C05), 14.1}, {board_card(0x2B85), 7.9},
        {board_card(0x1E07), 76.9},
    };
    double worst = 0;
    for (const Census& r : rows) {
        const double t = ch::prefill_small_seconds(m, r.card, 1819);
        worst          = std::max(worst, std::abs(t * 1e3 / r.ms - 1));
        std::cout << "prefill small kernels, " << r.card.sms << " SMs: " << t * 1e3 << " ms against " << r.ms << "\n";
    }
    check(worst < 0.25, "a prefill's small kernels by the scan's clocks and the rest's bytes within a quarter");
    const ch::CardFacts v100 = board_card(0x1DB5);
    check(ch::prefill_small_seconds(m, v100, 1819) > 0, "Volta's small kernels: the GV100's census (DECISIONS 70)");
}

// An MoE's whole prefill through the tiles (DECISIONS 66): the model's facts (speed_measured.json)
// with its linears and recurrent state, against the prefill rows of master f8d4399d (fit_prefill.py's
// method: 2K / 7K / 22K-token prompts, the median of three; the server's chunk). Printed, and the
// estimate within 15 % of each (the router, the plan and the selection not yet priced).
void prefills_moe_by_tiles(const json& fixture) {
    struct Point {
        const char* model;
        const char* card;
        std::uint32_t chunk, prompt;
        double measured;
    };
    const Point points[] = {
        {"Qwen3.6-35B-A3B", "RTX 4090", 2048, 1785, 17375.5}, {"Qwen3.6-35B-A3B", "RTX 4090", 2048, 7406, 17842.8},
        {"Qwen3.6-35B-A3B", "RTX 4090", 2048, 21928, 16292.6}, {"Qwen3.6-35B-A3B", "RTX 5090", 1024, 1785, 20170.4},
        {"Qwen3.6-35B-A3B", "RTX 5090", 1024, 7406, 20817.6}, {"Qwen3.6-35B-A3B", "RTX 5090", 1024, 21928, 19535.4},
        {"gemma-4-26B-A4B", "RTX 4090", 2048, 1752, 15719.1}, {"gemma-4-26B-A4B", "RTX 4090", 2048, 7347, 16157.3},
        {"gemma-4-26B-A4B", "RTX 4090", 2048, 21721, 14127.9},
    };
    double worst = 0;
    for (const Point& p : points) {
        const json& m    = fixture.at("models").at(p.model);
        ch::ModelFacts mf = model_at(m, "UD-Q4_K_M");
        const bool qwen  = std::string(p.model).rfind("Qwen", 0) == 0;
        if (qwen) {
            mf.layers = 40, mf.hidden = 2048, mf.recurrent_state_elements = 30.0 * 32 * 128 * 128;
            mf.prompt_tail_calls = 2;
            mf.full_layers = 10, mf.full_head_dim = 256;
            mf.linears = {{9216, 2048, 10, false}, {2048, 4096, 10, false}, {12288, 2048, 30, false},
                          {2048, 4096, 30, false}, {1024, 2048, 40, false}, {2048, 512, 40, false},
                          {1024, 2048, 320, true}, {2048, 512, 320, true}};
        } else {
            mf.layers = 30, mf.hidden = 2816;
            mf.prompt_tail_calls = 2; // the census (RTX 4090): two decode steps' kernels after the chunk, 13 ms
            mf.linears = {{8192, 2816, 25, false}, {2816, 4096, 25, false}, {9216, 2816, 5, false},
                          {2816, 8192, 5, false},  {4224, 2816, 30, false}, {2816, 2112, 30, false},
                          {1408, 2816, 240, true}, {2816, 704, 240, true}};
        }
        ch::QuantOption q  = quant_of("UD-Q4_K_M", m.at("quants").at("UD-Q4_K_M"));
        q.expert_banks     = ch::ExpertBanks::Int8;
        q.rest_weights     = ch::WeightClass::Q8; // UD-Q4_K_M's linears beside the experts
        const ch::CardFacts cf = card_of(fixture.at("cards").at(p.card));
        ch::Candidate c;
        c.kv    = KvCacheStorage::Int8Group64;
        c.chunk = p.chunk;
        const double est = ch::speed_of(mf, cf, q, c, 1, p.prompt, p.prompt).prefill;
        worst            = std::max(worst, std::abs(est / p.measured - 1));
        std::cout << "MoE prefill by tiles, " << p.model << " UD-Q4_K_M " << p.card << " " << p.prompt << ": " << est
                  << " against " << p.measured << " (" << (est / p.measured - 1) * 100 << " %)\n";
    }
    check(worst < 0.15, "an MoE's prefill through the tiles within 15 % of the rows");
}

// A prompt's full attention by its CTAs (DECISIONS 68): Qwen3.5 9B's eight full layers (16 heads of
// 256) over INT8 keys, the census's per-chunk calls (a layer's median) against the grid dealt in
// the kernel's layout and longest-first issue order: a 7K prompt's 1024-token chunks on the RTX
// 5090 (no PDL), 5070 Ti, 4060 Ti and 3060, the 4070 Ti's the 21K prompt's first seven (the costs
// were read on the 21K prompts).
void prices_attention_tiles() {
    ch::ModelFacts m;
    m.family               = "qwen3_5_text";
    m.full_layers          = 1;
    m.full_head_dim        = 256;
    m.full_attention_width = 16 * 2 * 256;
    struct Series {
        ch::CardFacts card;
        std::vector<double> us; // chunk i of 1024 tokens after i x 1024
    };

    const Series series[] = {
        {board_card(0x2B85), {89.3, 154.7, 221.4, 285.5, 350.9, 414.2, 480.3}},
        {board_card(0x2805), {272.4, 580.9, 890.2, 1231.7, 1528.9, 1821.7, 2152.4}},
        {board_card(0x2504), {566.2, 1110.8, 1646.7, 2190.3, 2734.7, 3275.4, 3833.7}},
        {board_card(0x2C05), {109.5, 236.5, 362.2, 486.4, 611.5, 735.3, 860.2}},
        // The RTX 4070 Ti: 60 SMs, its first two 64-row slots as half-height CTAs.
        {board_card(0x2782), {174.1, 368.9, 560.4, 751.4, 946.8, 1133.5, 1354.6}},
    };
    double worst = 0;
    for (const Series& s : series) {
        double before = 0;
        for (std::size_t i = 0; i < s.us.size(); ++i) {
            const double through = ch::prefill_attention_tiled_seconds(m, s.card, KvCacheStorage::Int8Group64,
                                                                       1024.0 * (i + 1), 1024);
            const double call = (through - before) * 1e6;
            before            = through;
            worst             = std::max(worst, std::abs(call / s.us[i] - 1));
            std::cout << "attention tiles, " << s.card.sms << " SMs, chunk " << i << ": " << call << " us against "
                      << s.us[i] << "\n";
        }
    }
    check(worst < 0.10, "a prompt's attention by its CTAs within 10 % of the census's calls");
    // FP8 keys: the FP8 prompt kernel (128 rows x a split of the keys a CTA) and its merge, a 7K
    // prompt's chunks (the kernel and the merge summed), the cards' L2 beside. The worst is the RTX
    // 4060 Ti's first chunk (+13 %): its merge's 40 MiB outgrow the 32 MiB L2, which still holds a
    // part of them.
    struct Fp8Series {
        ch::CardFacts card;
        double l2_mib;
        std::vector<double> us;
    };
    const Fp8Series fp8_series[] = {
        {series[0].card, 96, {80.3, 149.2, 219.8, 267.8, 321.9, 369.9, 416.5}},
        {series[1].card, 32, {405.4, 976.7, 1376.5, 1687.2, 1955.6, 2232.3, 2532.3}},
        {series[2].card, 2.25, {998.8, 2258.7, 3543.2, 4554.1, 5265.6, 6024.8, 6814.7}},
        {series[3].card, 48, {140.1, 348.8, 548.3, 653.0, 758.3, 861.5, 961.4}},
        {series[4].card, 48, {187.1, 536.4, 853.1, 1078.8, 1231.3, 1378.5, 1567.4}},
    };
    double fp8_worst = 0;
    for (const Fp8Series& s : fp8_series) {
        ch::CardFacts card = s.card;
        card.l2_bytes      = s.l2_mib * (1 << 20);
        double before      = 0;
        for (std::size_t i = 0; i < s.us.size(); ++i) {
            const double through = ch::prefill_attention_tiled_seconds(m, card, KvCacheStorage::Fp8E4M3Row256,
                                                                       1024.0 * (i + 1), 1024);
            fp8_worst = std::max(fp8_worst, std::abs((through - before) * 1e6 / s.us[i] - 1));
            before    = through;
        }
    }
    check(fp8_worst < 0.14, "FP8 keys: the FP8 prompt kernel and its merge within 14 % of the census's chunks");
    check(ch::prefill_attention_tiled_seconds(m, series[0].card, KvCacheStorage::Fp8E4M3Row256, 80, 1024) == 0,
          "FP8 keys: a call of 80 tokens takes the grouped kernels' share");
    // BF16 keys: the BF16 prompt kernel (64 rows a CTA in plain order), the RTX 5090 read, the
    // RTX 5070 Ti checked, a 7K prompt's chunks.
    const Series bf16_series[] = {
        {series[0].card, {87.5, 189.7, 298.8, 401.8, 504.9, 603.0, 706.2}},
        {series[3].card, {154.0, 364.5, 564.3, 763.7, 962.9, 1161.6, 1361.9}},
        {series[1].card, {312.6, 781.5, 1241.5, 1744.9, 2175.1, 2608.4, 3099.0}},
        {series[2].card, {525.4, 1317.8, 2108.4, 2907.3, 3735.2, 4535.9, 5351.4}},
    };
    double bf16_worst = 0;
    for (const Series& s : bf16_series) {
        double before = 0;
        for (std::size_t i = 0; i < s.us.size(); ++i) {
            const double through = ch::prefill_attention_tiled_seconds(m, s.card, KvCacheStorage::BFloat16,
                                                                       1024.0 * (i + 1), 1024);
            bf16_worst = std::max(bf16_worst, std::abs((through - before) * 1e6 / s.us[i] - 1));
            before     = through;
        }
    }
    check(bf16_worst < 0.04, "BF16 keys: the BF16 prompt kernel within 4 % of the census's chunks");
    // Its group-8 geometry (16 heads over 2 KV heads) takes 128-row tiles past an 8K prefix: the share
    // there, the same as with no prompt kernel at all.
    ch::ModelFacts g8  = m;
    g8.full_kv_heads   = 2;
    const double at_8k = ch::prefill_attention_tiled_seconds(g8, series[0].card, KvCacheStorage::BFloat16, 8192, 1024);
    const double at_9k = ch::prefill_attention_tiled_seconds(g8, series[0].card, KvCacheStorage::BFloat16, 9216, 1024);
    const double pairs = 2 * g8.full_attention_width * (9216.0 * 9216 / 2 - 8192.0 * 8192 / 2);
    check(std::abs((at_9k - at_8k) * 0.740 * 512 * 170 * 2.407e9 / pairs - 1) < 1e-9,
          "BF16 keys, 16 heads over 2, past an 8K prefix: the share");
    // A call of 256 tokens or fewer takes the grouped kernels: a 256-token prompt keeps the share, and
    // a 100-token tail after 32K costs its flops at the share, not 16 CTAs over the whole prefix.
    check(ch::prefill_attention_tiled_seconds(m, series[0].card, KvCacheStorage::Int8Group64, 256, 1024) == 0,
          "a short prompt: the grouped kernels' share");
    const double whole = ch::prefill_attention_tiled_seconds(m, series[0].card, KvCacheStorage::Int8Group64, 32768, 1024);
    const double tail  = ch::prefill_attention_tiled_seconds(m, series[0].card, KvCacheStorage::Int8Group64, 32868, 1024);
    // Exact activations run every route invariant: their attention keeps the share.
    {
        ch::ModelFacts mm = m;
        mm.full_layers = 8;
        mm.full_attention_width = 8 * 16 * 2 * 256;
        mm.params = mm.active_params = 9e9;
        mm.kv_bytes_per_token = [](KvCacheStorage) { return std::uint64_t{32768}; };
        ch::QuantOption q8;
        q8.weights = ch::WeightClass::Q8;
        q8.weight_bytes = 9.5e9;
        ch::Candidate fast, exact;
        fast.kv = exact.kv = KvCacheStorage::Int8Group64;
        fast.activations  = ActivationMode::Fast;
        exact.activations = ActivationMode::Exact;
        ch::ModelFacts shares = mm;
        shares.full_head_dim  = 0; // no prompt kernel: the share
        const double e = ch::speed_of(mm, series[0].card, q8, exact, 1, 16384, 16384).prefill;
        const double s = ch::speed_of(shares, series[0].card, q8, exact, 1, 16384, 16384).prefill;
        const double f = ch::speed_of(mm, series[0].card, q8, fast, 1, 16384, 16384).prefill;
        const double g = ch::speed_of(shares, series[0].card, q8, fast, 1, 16384, 16384).prefill;
        check(e > 0 && e == s && f != g, "exact activations: attention off the prompt kernel's law");
    }
    check(tail > whole && tail - whole < 4e-3, "a short tail chunk at the share (its flops, 2.8 ms), not 16 CTAs over the prefix (17 ms)");
}

// Gemma 4 12B's attention by its CTAs (DECISIONS 68): its 40 sliding layers (16 heads of 256 over
// 8 KV heads, a 1024 window) and 8 full ones (16 heads of 512 over 1), a 7K prompt's 1024-token
// chunks, the census's per-layer medians summed over the layers, on the RTX 4060 Ti, 5090 and 2080 Ti.
void prices_gemma_attention_tiles(const json& fixture) {
    const ch::ModelFacts m = model_of(fixture.at("models").at("gemma-4-12b"));
    struct Series {
        ch::CardFacts card;
        std::vector<double> sliding, full; // a layer's us, chunk i after i x 1024
    };

    const Series series[] = {
        {board_card(0x2805),
         {300.3, 610.7, 620.2, 609.8, 611.2, 610.0},
         {812.9, 2403.9, 4062.3, 5596.3, 7259.1, 8854.9}},
        {board_card(0x2B85),
         {83.0, 141.7, 142.6, 143.4, 143.3, 143.9},
         {154.0, 492.4, 735.8, 1052.7, 1282.6, 1605.3}},
        // The RTX 2080 Ti (Turing's row, read on it).
        {board_card(0x1E07),
         {745.3, 1143.0, 1142.5, 1137.9, 1135.9, 1134.9},
         {1387.0, 4448.4, 7658.4, 10789.3, 13900.7, 17006.3}},
        // The Quadro GV100 (Volta's row, read on it).
        {board_card(0x1DBA),
         {974.1, 1568.3, 1525.7, 1522.6, 1508.6, 1524.7},
         {1666.7, 5241.9, 8712.2, 12284.3, 15746.2, 19391.7}},
    };
    double worst = 0, first = 0;
    for (const Series& s : series) {
        double before = 0;
        for (std::size_t i = 0; i < s.sliding.size(); ++i) {
            const double through = ch::prefill_attention_tiled_seconds(m, s.card, KvCacheStorage::BFloat16,
                                                                       1024.0 * (i + 1), 1024);
            const double call     = (through - before) * 1e6;
            const double measured = m.sliding_layers * s.sliding[i] + m.full_layers * s.full[i];
            before                = through;
            // The first chunk (keys from 16 to 1024 a CTA) runs slower than its keys say on Turing and
            // Volta (-9 %, -12 %: open, DECISIONS 68); the rest within 12 %.
            const bool open_first = i == 0 && s.card.major == 7; // Turing and Volta
            (open_first ? first : worst) = std::max(open_first ? first : worst, std::abs(call / measured - 1));
            std::cout << "Gemma attention tiles, " << s.card.sms << " SMs, chunk " << i << ": " << call
                      << " us against " << measured << "\n";
        }
    }
    check(worst < 0.12 && first < 0.13, "Gemma 4's attention by its CTAs within 12 % of the census's chunks");
    // A KV-shared model (Gemma 4 E4B-like: 35 sliding layers attend, 20 hold their own KV, 8 heads over
    // 2): its heads and layers from the attending ones, 35 / 20 of the same layers held alone.
    ch::ModelFacts shared;
    shared.family                   = "gemma4_text";
    shared.window                   = 512;
    shared.sliding_layers           = 20;
    shared.sliding_attending_layers = 35;
    shared.sliding_head_dim         = 256;
    shared.sliding_kv_heads         = 2;
    shared.sliding_attention_width  = 35 * 8 * 512;
    ch::ModelFacts alone           = shared;
    alone.sliding_attending_layers = 20;
    alone.sliding_attention_width  = 20 * 8 * 512;
    const double a = ch::prefill_attention_tiled_seconds(shared, series[0].card, KvCacheStorage::BFloat16, 4096, 1024);
    const double b = ch::prefill_attention_tiled_seconds(alone, series[0].card, KvCacheStorage::BFloat16, 4096, 1024);
    check(b > 0 && std::abs(a / b - 35.0 / 20) < 1e-9, "a KV-shared Gemma 4: its attending layers' heads and count");
}

void contradictory_pins(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4070 Ti");
    ch::Pins pins;
    pins.activations = ActivationMode::Exact;
    pins.state       = RecurrentStateStorage::FP16;
    const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    pins, ch::Task{}, oracle(b, 20e9));
    check(!c.chosen && c.why.find("refuses") != std::string::npos, "exact with a pinned FP16 state: refused");
}

// The coupled storages are explained: the recurrent state's other values beside the chosen.
void explains_the_state(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4070 Ti");
    ch::Task task;
    const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    {}, task, oracle(b, 9.6e9));
    int states = 0, coupled = 0;
    for (const ch::Alternative& a : ch::explain(c)) {
        if (a.axis == ch::Axis::State) {
            ++states;
            if (a.why_code == "coupled") { ++coupled; }
        }
    }
    check(states == 3 && coupled == 2, "the state: the chosen f16 and its two other values");
}

} // namespace

// The drafter is one more axis: its memory in the plan, its gain in the
// decode. The fastest that holds the asked context wins where the worth decides; one the context
// leaves out is explained with its gain; past its load table's streams it gains nothing; a pin
// keeps it; where the worth does not decide, the resolved profile's stands.
void weighs_the_drafters(const json& fixture) {
    Bench b = nine_b(fixture, "RTX 4090");
    b.model.drafters = {{"mtp:4", "mtp", 1.9, 2, "measured"}, {"dflash:7", "dflash", 2.5, 2, "measured"}};
    b.model.default_drafter = 1; // the resolved profile's: MTP
    // A drafter's weights take memory from the context: MTP 0.2 GB, DFlash 2.4 GB.
    const auto oracle_with = [&b](double budget, std::size_t quant = 0) -> ch::OwnContextFn {
        return [&b, budget, quant](const ch::Candidate& c) -> std::uint32_t {
            const double drafter = c.drafter == 1 ? 0.2e9 : c.drafter == 2 ? 2.4e9 : 0.0;
            const double room    = budget - b.quants[quant].weight_bytes - drafter;
            if (room <= 0) { return 0; }
            return static_cast<std::uint32_t>(room / b.model.kv_bytes_per_token(c.kv));
        };
    };
    const ch::Gates open{true, true};
    const auto allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
    ch::Task task;
    task.contexts = {32768};
    const ch::Choice roomy = ch::choose(b.model, b.card, b.quants[0], allowed, {}, task, oracle_with(40e9), open);
    check(roomy.chosen && chosen(roomy).candidate.drafter == 2 && chosen(roomy).drafter == "dflash:7" &&
              roomy.why.find("speculation dflash:7") != std::string::npos,
          "room for every drafter: the fastest, DFlash: " + roomy.why);
    int lost = 0;
    for (const ch::Alternative& a : ch::explain(roomy)) {
        if (a.axis == ch::Axis::Drafter && (a.value == "plain" || a.value == "mtp:4")) {
            check(a.why_code == "lower_worth", a.value + " fits and loses on worth");
            ++lost;
        }
    }
    check(lost == 2, "plain and MTP explained beside DFlash");
    // The largest quantization's weights leave room for MTP's 0.2 GB but not DFlash's 2.4 GB at 32K:
    // MTP, and DFlash explained with its gain.
    std::size_t largest = 0;
    for (std::size_t i = 1; i < b.quants.size(); ++i) {
        if (b.quants[i].weight_bytes > b.quants[largest].weight_bytes) { largest = i; }
    }
    const double per_token = b.model.kv_bytes_per_token(KvCacheStorage::Int8Group64);
    const double tight     = b.quants[largest].weight_bytes + 0.2e9 + 40000 * per_token;
    ch::Pins int8;
    int8.kv = KvCacheStorage::Int8Group64;
    const ch::Choice tight_choice = ch::choose(b.model, b.card, b.quants[largest], allowed, int8, task,
                                               oracle_with(tight, largest), open);
    check(tight_choice.chosen && chosen(tight_choice).candidate.drafter == 1,
          "DFlash does not hold 32K beside the largest quantization: MTP");
    bool said = false;
    for (const ch::Alternative& a : ch::explain(tight_choice)) {
        if (a.axis == ch::Axis::Drafter && a.value == "dflash:7") {
            said = a.why_code == "no_fit" && a.why_not.find("decodes x") != std::string::npos;
        }
    }
    check(said, "the faster drafter the context leaves out is said with its gain");
    // Two streams: past the load tables' bound neither drafter gains; plain keeps the most context.
    ch::Task two = task;
    two.concurrency = 2;
    const ch::Choice pair = ch::choose(b.model, b.card, b.quants[0], allowed, {}, two, oracle_with(40e9), open);
    check(pair.chosen && chosen(pair).candidate.drafter == 0, "two streams: plain decode");
    // A pin keeps the drafter; closed gates keep the resolved profile's.
    ch::Pins plain;
    plain.drafter = 0;
    const ch::Choice pinned = ch::choose(b.model, b.card, b.quants[0], allowed, plain, task, oracle_with(40e9), open);
    check(pinned.chosen && chosen(pinned).candidate.drafter == 0, "--spec none: plain decode");
    // A drafter published without its speeds (the "*" profile) weighs a gain of 1: it is not
    // weighed, and the profile's choice stands (never plain decode for a gain nobody measured).
    {
        Bench star = b;
        star.model.drafters = {{"mtp:5+head", "mtp", 1.0, 0, "generic", false}};
        star.model.default_drafter = 1;
        const ch::Choice kept = ch::choose(star.model, star.card, star.quants[0], allowed, {}, task,
                                           oracle_with(40e9), open);
        check(kept.chosen && kept.by_worth && !kept.drafter_by_worth &&
                  chosen(kept).candidate.drafter == 1,
              "an unmeasured drafter: the profile's choice stands");
        bool said = false;
        for (const ch::Alternative& a : ch::explain(kept)) {
            if (a.axis == ch::Axis::Drafter && a.value == "plain") {
                said = a.why_code == "unmeasured" && a.why_not.find("not measured") != std::string::npos;
            }
        }
        check(said, "plain decode explained: the drafter's gain is not measured here");
        // Its gain, given or not, is not counted in the decode estimate (the page shows its own).
        star.model.drafters[0].speedup = 3.0;
        const ch::Candidate at{KvCacheStorage::Int8Group64, KvCacheStorage::BFloat16,
                               RecurrentStateStorage::FP16, ActivationMode::Fast, ch::kDefaultChunk, 1};
        ch::Candidate without = at;
        without.drafter       = 0;
        check(ch::speed_of(star.model, star.card, star.quants[0], at, 1, 4096).decode_per_stream ==
                  ch::speed_of(star.model, star.card, star.quants[0], without, 1, 4096).decode_per_stream,
              "an unmeasured gain is not counted in the decode estimate");
    }
    const ch::Choice closed = ch::choose(b.model, b.card, b.quants[0], allowed, {}, task, oracle_with(40e9));
    check(closed.chosen && (closed.by_worth || chosen(closed).candidate.drafter == 1),
          "the worth not deciding: the profile's drafter");
}

// A load table that bounds the drafter's depth (#303): from its bound on, the gain counted is
// the bounded depth's, one request measured; an unmeasured depth counts none, and from the plain
// bound on there is none.
void load_bounds_the_gain(const json& fixture) {
    Bench b = nine_b(fixture, "RTX 4090");
    ch::DrafterOption d{"mtp:4", "mtp", 2.0, 8, "measured"};
    d.load = {{.streams = 2, .speedup = 1.5, .depth = 2, .bounded = true},
              {.streams = 4, .speedup = 1.0, .depth = 1, .bounded = true, .measured = false}};
    check(d.speedup_at(1) == 2.0 && d.speedup_at(2) == 1.5 && d.speedup_at(3) == 1.5 &&
              d.speedup_at(4) == 1.0 && d.speedup_at(7) == 1.0 && d.speedup_at(8) == 1.0,
          "the gain by streams: 2, then the bounds', then plain");
    // A bound whose depth was not measured (a published profile lists only its choice's): its
    // gain is not measured there, so the drafter is not weighed at that load.
    check(d.measured_at(1) && d.measured_at(3) && !d.measured_at(4) && !d.measured_at(7) &&
              d.measured_at(8),
          "measured by streams: the bound's own depth, plain decode's from its bound");
    b.model.drafters        = {d};
    b.model.default_drafter = 1;
    const ch::Candidate at{KvCacheStorage::Int8Group64, KvCacheStorage::BFloat16,
                           RecurrentStateStorage::FP16, ActivationMode::Fast, ch::kDefaultChunk, 1};
    ch::Candidate without = at;
    without.drafter       = 0;
    for (const auto& [streams, gain] : std::vector<std::pair<std::uint32_t, double>>{{1, 2.0}, {2, 1.5}, {3, 1.5}, {4, 1.0}}) {
        const double with  = ch::speed_of(b.model, b.card, b.quants[0], at, streams, 4096).decode_per_stream;
        const double plain = ch::speed_of(b.model, b.card, b.quants[0], without, streams, 4096).decode_per_stream;
        check(plain > 0 && std::abs(with / plain - gain) < 1e-9,
              "decode at " + std::to_string(streams) + " streams: x" + std::to_string(gain));
    }
}

// The joint verify's price scales the gain counted at the streams it is priced for (#303): the
// bound's one-request gain times R_1 / R_C, below 1 where the round no longer pays (plain decode
// then wins the drafter axis); streams not priced keep the one-request ceiling, plain decode's 1
// from its bound on is not scaled.
void joint_price_scales_the_gain(const json& fixture) {
    Bench b = nine_b(fixture, "RTX 4090");
    ch::DrafterOption d{"mtp:4", "mtp", 2.0, 5, "measured"};
    d.load  = {{.streams = 2, .speedup = 1.5, .depth = 2, .bounded = true}};
    d.joint = {
        {.streams = 2, .scale = 0.8}, {.streams = 3, .scale = 0.5}, {.streams = 5, .scale = 0.1}};
    check(d.speedup_at(1) == 2.0 && std::abs(d.speedup_at(2) - 1.2) < 1e-12 &&
              std::abs(d.speedup_at(3) - 0.75) < 1e-12 &&
              std::abs(d.speedup_at(4) - 0.75) < 1e-12 && d.speedup_at(5) == 1.0,
          "gain by streams: bound x joint scale where priced, the least priced below where not, "
          "plain from "
          "its bound");
    check(!d.joint_priced_at(1) && d.joint_priced_at(2) && d.joint_priced_at(4) &&
              !d.joint_priced_at(5),
          "priced at the listed streams below the plain bound");
    // An unmeasured bound's placeholder gain (1) priced at 2 streams does not hold the measured
    // bound at 4 streams below its own price.
    ch::DrafterOption u{"mtp:4", "mtp", 2.0, 8, "measured"};
    u.load  = {{.streams = 2, .speedup = 1.0, .depth = 1, .bounded = true, .measured = false},
               {.streams = 4, .speedup = 1.6, .depth = 3, .bounded = true}};
    u.joint = {{.streams = 2, .scale = 0.9}, {.streams = 4, .scale = 0.8}};
    check(!u.measured_at(2) && u.measured_at(4) && std::abs(u.speedup_at(4) - 1.28) < 1e-12,
          "an unmeasured bound's placeholder does not bound a measured gain: " +
              std::to_string(u.speedup_at(4)));
    // Nor does the placeholder's own count count its price: the estimate multiplies by 1 there
    // (measured_at is false), and speedup_at says the same.
    check(u.speedup_at(2) == 1.0 && !u.joint_priced_at(2) && u.speedup_at(3) == 1.0 &&
              !u.joint_priced_at(3),
          "a price at an unmeasured bound's own count is not counted: " +
              std::to_string(u.speedup_at(2)));
    // An option nobody measured is not priced at all.
    ch::DrafterOption nobody{"mtp:4", "mtp", 1.5, 0, "measured"};
    nobody.measured = false;
    nobody.joint    = {{.streams = 2, .scale = 0.45}};
    check(nobody.speedup_at(2) == 1.5 && !nobody.joint_priced_at(2),
          "an unmeasured option's gain is not priced");
    // A price above 1 is clamped away: it supplies nothing.
    ch::DrafterOption over{"mtp:4", "mtp", 2.0, 0, "measured"};
    over.joint = {{.streams = 2, .scale = 1.05}};
    check(over.speedup_at(2) == 2.0 && !over.joint_priced_at(2),
          "a scale above 1 does not raise the gain nor count as a price");
    // Nor is it held to a larger count: where a bound lifts the gain above one request's, the
    // scale's product at two streams (1.5 x 1.2) is below the gain at three (2.0), and would be
    // held there as a price.
    ch::DrafterOption lifted{"mtp:4", "mtp", 1.5, 0, "measured"};
    lifted.load  = {{.streams = 3, .speedup = 2.0, .depth = 2, .bounded = true}};
    lifted.joint = {{.streams = 2, .scale = 1.2}};
    check(lifted.speedup_at(3) == 2.0 && !lifted.joint_priced_at(3) &&
              lifted.price_source_at(3).streams == 0,
          "a scale above 1 is not held to a larger count: " + std::to_string(lifted.speedup_at(3)));
    // A count whose own bound is unmeasured takes no price from a smaller count: the estimate
    // counts x1 there (measured_at is false), and a price held from two streams (1.1 x 0.8) would
    // say x0.88.
    ch::DrafterOption gap{"mtp:4", "mtp", 1.1, 0, "measured"};
    gap.load  = {{.streams = 3, .speedup = 1.0, .depth = 1, .bounded = true, .measured = false}};
    gap.joint = {{.streams = 2, .scale = 0.8}};
    check(!gap.measured_at(3) && gap.speedup_at(3) == 1.0 && !gap.joint_priced_at(3) &&
              std::abs(gap.speedup_at(2) - 0.88) < 1e-12 && gap.joint_priced_at(2),
          "a count whose bound is unmeasured takes no price from a smaller count: " +
              std::to_string(gap.speedup_at(3)));
    // The count that supplies the gain, with its own scale (not R_1 / R_C of no count).
    ch::DrafterOption held{"mtp:4", "mtp", 2.0, 0, "measured"};
    held.load         = {{.streams = 2, .speedup = 1.5, .depth = 2, .bounded = true},
                         {.streams = 4, .speedup = 1.2, .depth = 1, .bounded = true}};
    held.joint        = {{.streams = 2, .scale = 0.5}};
    const auto source = held.price_source_at(4);
    check(source.streams == 2 && source.scale == 0.5 && std::abs(source.gain - 0.75) < 1e-12,
          "the source of a held gain is the count and the scale it is held from");
    b.model.drafters        = {d};
    b.model.default_drafter = 1;
    ch::Candidate with{KvCacheStorage::Int8Group64, KvCacheStorage::BFloat16,
                       RecurrentStateStorage::FP16, ActivationMode::Fast,
                       ch::kDefaultChunk,           1};
    ch::Candidate without = with;
    without.drafter       = 0;
    for (const auto& [streams, gain] :
         std::vector<std::pair<std::uint32_t, double>>{{1, 2.0}, {2, 1.2}, {3, 0.75}, {4, 0.75}}) {
        const double a =
            ch::speed_of(b.model, b.card, b.quants[0], with, streams, 4096).decode_per_stream;
        const double p =
            ch::speed_of(b.model, b.card, b.quants[0], without, streams, 4096).decode_per_stream;
        check(p > 0 && std::abs(a / p - gain) < 1e-9,
              "decode at " + std::to_string(streams) + " streams: x" + std::to_string(gain));
    }
}

// The gain is monotone in the streams once the joint verify is priced (#303): mtp:3 priced at two
// streams (4 rows each, 8 verify rows) below 1, three streams (12 rows) not priced: its gain is
// no more than two streams', never the one-request 2.0 again.
void joint_price_keeps_the_gain_monotone() {
    ch::DrafterOption d{"mtp:3", "mtp", 2.0, 0, "measured"};
    d.joint = {{.streams = 2, .scale = 0.45}};
    check(d.speedup_at(1) == 2.0 && std::abs(d.speedup_at(2) - 0.9) < 1e-12 &&
              d.speedup_at(3) <= d.speedup_at(2) && d.speedup_at(8) <= d.speedup_at(2),
          "an unpriced count of streams does not climb back above the priced one below it");
    ch::DrafterOption unpriced{"mtp:3", "mtp", 2.0, 0, "measured"};
    check(unpriced.speedup_at(3) == 2.0 && !unpriced.joint_priced_at(3),
          "before any count is priced the one-request gain stands");
    // A count priced above the least price below it is held there too: 2.0 x 0.5 at two streams,
    // 2.0 x 0.9 at three, so three streams gain no more than two's 1.0.
    ch::DrafterOption climbing{"mtp:3", "mtp", 2.0, 0, "measured"};
    climbing.joint = {{.streams = 2, .scale = 0.5}, {.streams = 3, .scale = 0.9}};
    check(std::abs(climbing.speedup_at(2) - 1.0) < 1e-12 &&
              std::abs(climbing.speedup_at(3) - 1.0) < 1e-12 &&
              std::abs(climbing.speedup_at(4) - 1.0) < 1e-12,
          "a priced count of streams does not climb above the least price below it");
}

// The joint verify's scales go through the replay record (#303): written as text and read back, a
// choice at three streams replays to the same, and the record holds them.
void joint_price_replays(const json& fixture) {
    Bench b = nine_b(fixture, "RTX 4090");
    ch::DrafterOption d{"mtp:3", "mtp", 2.0, 0, "measured"};
    d.joint                   = {{.streams = 2, .scale = 0.45}};
    b.model.drafters          = {d};
    b.model.default_drafter   = 1;
    const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
    ch::Task task;
    task.concurrency              = 3;
    const ch::ChoiceRecord record = ch::record_choice(b.model, b.card, b.quants[0], allowed, {},
                                                      task, oracle(b, 40e9), ch::Gates{true, true});
    const std::string text        = record.inputs.dump();
    check(text.find("\"joint\"") != std::string::npos && text.find("0.45") != std::string::npos,
          "the record holds the joint scales");
    const ch::Choice again = ch::replay_choice(nlohmann::ordered_json::parse(text));
    check(ch::choice_summary(again) == ch::choice_summary(record.choice),
          "a choice with joint scales replays to the same");
    // A record whose joint scales are not ascending by streams, or not positive, is rejected on
    // read (speedup_at reads them in order).
    const auto reads = [&](const nlohmann::ordered_json& joint) {
        nlohmann::ordered_json bad           = nlohmann::ordered_json::parse(text);
        bad["model"]["drafters"][0]["joint"] = joint;
        try {
            (void)ch::replay_choice(bad);
            return true;
        } catch (const std::invalid_argument&) { return false; }
    };
    check(reads(nlohmann::ordered_json::array(
              {{{"streams", 2}, {"scale", 0.5}}, {{"streams", 3}, {"scale", 0.6}}})),
          "ascending joint scales are read");
    check(!reads(nlohmann::ordered_json::array(
              {{{"streams", 3}, {"scale", 0.5}}, {{"streams", 2}, {"scale", 0.6}}})),
          "joint scales not ascending by streams are rejected");
    check(!reads(nlohmann::ordered_json::array({{{"streams", 2}, {"scale", 0.0}}})),
          "a joint scale that is not positive is rejected");
    // One request's verify is the base the scale is taken against: an entry below two streams is
    // not a joint price.
    check(!reads(nlohmann::ordered_json::array({{{"streams", 1}, {"scale", 0.5}}})) &&
              !reads(nlohmann::ordered_json::array({{{"streams", 0}, {"scale", 0.5}}})),
          "a joint scale below two streams is rejected");
}

// The order a manifest lists its drafters in does not decide: two drafters of equal worth tie
// and the same one wins whichever comes first (#304).
void drafter_order_does_not_decide(const json& fixture) {
    Bench b = nine_b(fixture, "RTX 4090");
    const ch::Gates open{true, true};
    const auto allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
    ch::Task task;
    task.contexts = {32768};
    const ch::OwnContextFn room = [&b](const ch::Candidate& c) -> std::uint32_t {
        const double left = 40e9 - b.quants[0].weight_bytes - (c.drafter != 0 ? 0.2e9 : 0.0);
        return left <= 0 ? 0 : static_cast<std::uint32_t>(left / b.model.kv_bytes_per_token(c.kv));
    };
    const ch::DrafterOption head{"mtp:4+head", "mtp", 2.0, 2, "measured"};
    const ch::DrafterOption plain_head{"mtp:4", "mtp", 2.0, 2, "measured"};
    std::vector<std::string> winners;
    for (const auto& order : {std::vector<ch::DrafterOption>{head, plain_head},
                              std::vector<ch::DrafterOption>{plain_head, head}}) {
        b.model.drafters        = order;
        b.model.default_drafter = 1;
        const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], allowed, {}, task, room, open);
        check(c.chosen.has_value(), "two tied drafters: a choice");
        if (c.chosen) { winners.push_back(chosen(c).drafter); }
    }
    check(winners.size() == 2 && winners[0] == winners[1] && winners[0] == "mtp:4",
          "the tied drafter that wins does not follow the manifest's order");
}

// #301: the order a caller lists an axis's values in never changes the choice (the enumeration
// order breaks exact ties: choose() enumerates each axis in one canonical order). Every ordering of
// the KV codecs and activations, on every fixture card, at a long and a short context.
void axis_order_does_not_decide(const json& fixture) {
    for (const auto& [name, unused] : fixture.at("cards").items()) {
        (void)unused;
        const Bench b = nine_b(fixture, name);
        const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
        for (const std::uint32_t context : {4096U, 131072U}) {
            ch::Task task;
            task.contexts = {context};
            const auto describe = [](const ch::Choice& c) {
                if (!c.chosen) { return std::string("none"); }
                const ch::Candidate& k = c.scored[*c.chosen].candidate;
                return std::string(kv_cache_storage_name(k.kv)) + "/" + std::to_string(static_cast<int>(k.activations)) +
                       "/" + std::to_string(k.chunk) + "/" + std::to_string(k.drafter) + "|" + c.why;
            };
            const std::string first =
                describe(ch::choose(b.model, b.card, b.quants[0], allowed, {}, task, oracle(b, 20e9), ch::Gates{true, true}));
            ch::Allowed shuffled = allowed;
            std::sort(shuffled.kv.begin(), shuffled.kv.end());
            bool same = true;
            int orders = 0;
            do {
                std::sort(shuffled.activations.begin(), shuffled.activations.end());
                do {
                    ++orders;
                    same = same && describe(ch::choose(b.model, b.card, b.quants[0], shuffled, {}, task,
                                                       oracle(b, 20e9), ch::Gates{true, true})) == first;
                } while (std::next_permutation(shuffled.activations.begin(), shuffled.activations.end()));
            } while (std::next_permutation(shuffled.kv.begin(), shuffled.kv.end()));
            check(same && orders == 720, name + ", " + std::to_string(context) + " tokens: no axis order changes the choice");
        }
    }
}

// The worth's resolution (#301): where the worth decides it is the reference generation's decode
// error, and an alternative is within_resolution exactly when it loses by more than a tie and no
// more than the smaller of that and the two's speed difference; a larger gap stays lower_worth.
void resolution_labels(const json& fixture) {
    int within = 0, lower = 0;
    for (const auto& [name, unused] : fixture.at("cards").items()) {
        (void)unused;
        Bench b = nine_b(fixture, name);
        const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
        for (const std::uint32_t context : {4096U, 32768U, 131072U}) {
            ch::Task task;
            task.contexts = {context};
            const ch::Choice c = ch::choose(b.model, b.card, b.quants[0], allowed, {}, task,
                                            oracle(b, 40e9), ch::Gates{true, true});
            if (!c.chosen) { continue; }
            check(c.by_worth ? c.resolution > 0 && c.resolution < 0.1 : true,
                  name + ": a resolution where the worth decides");
            const ch::Scored& best = chosen(c);
            if (c.resolved_within_resolution) {
                const auto alts = ch::explain(c);
                const auto anchors =
                    std::count_if(alts.begin(), alts.end(), [&](const ch::Alternative& a) {
                        return a.scored == &c.scored[c.anchor];
                    });
                check(anchors == 1, name + ": the passed-over best worth is listed exactly once");
            }
            for (const ch::Alternative& a : ch::explain(c)) {
                if (a.scored == nullptr ||
                    (a.why_code != "within_resolution" && a.why_code != "lower_worth" &&
                     a.why_code != "outside_resolution")) {
                    continue;
                }
                const double gap   = best.worth - a.scored->worth;
                if (gap < -1e-9) {
                    // Above the pick only under a pick the resolution rule moved: the best worth
                    // and the others inside the band of their comparison with it (#618: the
                    // wider basis of the two, at its resolution_width) are within_resolution, the
                    // ones slower than it by more than that are outside_resolution; never
                    // lower_worth.
                    const ch::Scored& anchor = c.scored[c.anchor];
                    const double width       = ch::resolution_width(
                        ch::resolution_for(c.resolutions, ch::widest(anchor.decision_basis,
                                                                           a.scored->decision_basis)));
                    const bool inside =
                        a.scored == &anchor || std::log(a.scored->decision_seconds /
                                                        anchor.decision_seconds) <= width + 1e-9;
                    check(c.resolved_within_resolution &&
                              a.why_code == (inside ? "within_resolution" : "outside_resolution"),
                          name + " " + a.value + ": a higher worth is " +
                              (inside ? "within_resolution" : "outside_resolution"));
                    continue;
                }
                const double apart = std::abs(std::log(a.scored->decision_seconds / best.decision_seconds));
                // The band of this alternative's own comparison with the chosen (#618).
                const double width = ch::resolution_width(ch::resolution_for(
                    c.resolutions, ch::widest(best.decision_basis, a.scored->decision_basis)));
                const bool inside  = gap > 1e-9 && gap <= std::min(width, apart) + 1e-9;
                check(inside == (a.why_code == "within_resolution"),
                      name + " " + a.value + ": within_resolution exactly inside the band");
                (inside ? within : lower) += 1;
            }
        }
    }
    std::cout << "resolution labels: " << within << " within, " << lower << " lower\n";
    check(lower > 0, "some alternatives lose outright");
}

// #304: a choice made on the generation's reference card reads none of the board's continuous
// facts: another clock, memory bandwidth, SM count or L2 of the same generation moves the speeds
// shown, never what is chosen. Every fixture card, a short and a long context, the dense 9B and
// the MoE 35B with two measured drafters (its chunk by speed and the drafter by worth open).
void board_facts_do_not_decide(const json& fixture) {
    std::vector<std::pair<std::string, Bench>> benches;
    for (const auto& [name, unused] : fixture.at("cards").items()) {
        (void)unused;
        benches.emplace_back(name + ", Qwen3.5-9B", nine_b(fixture, name));
        const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
        Bench moe{model_of(m35), card_of(fixture.at("cards").at(name)), {}};
        moe.quants.push_back(quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS")));
        moe.model.drafters        = {{"dflash:7", "dflash", 2.5, 2, "measured"}, {"mtp:4", "mtp", 1.9, 2, "measured"}};
        moe.model.default_drafter = 2;
        benches.emplace_back(name + ", Qwen3.6-35B-A3B", moe);
    }
    for (const auto& [label, b] : benches) {
        const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
        for (const std::uint32_t context : {4096U, 131072U}) {
            ch::Task task;
            task.contexts = {context};
            const auto decided = [&](const ch::CardFacts& card) {
                const ch::Choice c = ch::choose(b.model, card, b.quants[0], allowed, {}, task,
                                                oracle(b, 40e9), ch::Gates{true, true});
                if (!c.chosen) { return std::string("none"); }
                const ch::Candidate& k = chosen(c).candidate;
                return std::string(kv_cache_storage_name(k.kv)) + "/" +
                       std::to_string(static_cast<int>(k.activations)) + "/" + std::to_string(k.chunk) +
                       "/" + std::to_string(k.drafter) + (c.by_worth ? "/worth" : "/defaults") +
                       (c.chunk_by_speed ? "/chunk" : "") + (c.drafter_by_worth ? "/drafter" : "");
            };
            const std::string base = decided(b.card);
            bool same = true;
            for (const double f : {0.7, 1.3}) {
                ch::CardFacts clock = b.card, dram = b.card, sms = b.card, l2 = b.card;
                clock.clock_hz *= f;
                dram.dram_bytes_per_s *= f;
                sms.sms = static_cast<int>(sms.sms * f);
                l2.l2_bytes = (l2.l2_bytes > 0 ? l2.l2_bytes : 32e6) * f;
                for (const ch::CardFacts& other : {clock, dram, sms, l2}) { same = same && decided(other) == base; }
            }
            check(same, label + ", " + std::to_string(context) + " tokens: the board's own rates do not decide (" + base + ")");
        }
    }
}

// #300: a compute capability the generation table has no row of (a newer major) estimates on the
// nearest row (gb20x), says so, and its reference decides nothing, whatever reference cards the
// lent generation gains later; a known generation keeps its own evidence. Today no lent row has a
// reference card, so the decision is blocked twice; the reason names the missing row.
void lent_generation_does_not_decide(const json& fixture) {
    const Bench known = nine_b(fixture, "RTX 5090");
    Bench lent        = known;
    lent.card.major   = 13;
    lent.card.minor   = 0;
    ch::Candidate c;
    c.kv    = KvCacheStorage::Int8Group64;
    c.state = RecurrentStateStorage::FP16;
    const ch::Speed own   = ch::speed_of(known.model, known.card, known.quants[0], c, 1, 4096);
    const ch::Speed other = ch::speed_of(lent.model, lent.card, lent.quants[0], c, 1, 4096);
    check(own.basis != ch::SpeedBasis::LentGeneration, "the RTX 5090: its own generation's rows");
    check(other.basis == ch::SpeedBasis::LentGeneration && other.transferred &&
              std::abs(other.decode_per_stream - own.decode_per_stream) < 1e-9 * own.decode_per_stream,
          "compute capability 13.0: gb20x's decode estimate, marked lent");
    const ch::Gates open{true, true};
    const auto ref = ch::reference_of(lent.card, open);
    check(ref && !ref->decides && !ref->chunk_decides &&
              ref->why.find("no generation row of its own") != std::string::npos,
          "compute capability 13.0: a reference that never decides, and why");
    ch::Task task;
    const auto allowed      = ch::allowed_axes(lent.model, KvCacheStorage::Int8Group64);
    const ch::Choice theirs = ch::choose(lent.model, lent.card, lent.quants[0], allowed, {}, task, oracle(lent, 20e9), open);
    check(!theirs.by_worth && theirs.chosen && theirs.why.find("no generation row of its own") != std::string::npos,
          "compute capability 13.0: the family's defaults, and why");
    check(!ch::weighs_quantizations(lent.model, lent.card, open), "compute capability 13.0: the quantizations by the rule");
    // A known generation with a reference card still decides (the RTX 4090: Ada).
    const Bench ada = nine_b(fixture, "RTX 4090");
    check(ch::choose(ada.model, ada.card, ada.quants[0], allowed, {}, task, oracle(ada, 20e9), open).by_worth,
          "the RTX 4090: the switches by worth");
}

// #304: a choice made again from its record alone. On every fixture card, for the dense 9B and the
// MoE 35B with its two measured drafters, at a short and a long context, with and without a pin:
// the record (written as text and read back) replays to the same choice, every scored candidate
// alike; a record lacking a value it is asked is incomplete, never a guess; a record of another
// schema is refused.
void choices_replay(const json& fixture) {
    std::vector<std::pair<std::string, Bench>> benches;
    for (const auto& [name, unused] : fixture.at("cards").items()) {
        (void)unused;
        benches.emplace_back(name + ", Qwen3.5-9B", nine_b(fixture, name));
        const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
        Bench moe{model_of(m35), card_of(fixture.at("cards").at(name)), {}};
        moe.quants.push_back(quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS")));
        moe.model.drafters        = {{"dflash:7", "dflash", 2.5, 2, "measured"}, {"mtp:4", "mtp", 1.9, 2, "measured"}};
        moe.model.default_drafter = 2;
        // A load table too: its bounds go through the record.
        moe.model.drafters[0].load = {{4, 1.6, 3, true, true}, {16, 1.2, 1, true, false}};
        benches.emplace_back(name + ", Qwen3.6-35B-A3B", moe);
    }
    int replays = 0;
    for (const auto& [label, b] : benches) {
        const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
        for (const std::uint32_t context : {4096U, 131072U}) {
            for (const bool pinned : {false, true}) {
                ch::Task task;
                task.contexts = {context, context / 2};
                ch::Pins pins;
                if (pinned) {
                    pins.kv          = KvCacheStorage::BFloat16;
                    pins.activations = ActivationMode::Fast;
                }
                const ch::ChoiceRecord record = ch::record_choice(
                    b.model, b.card, b.quants[0], allowed, pins, task, oracle(b, 40e9), ch::Gates{true, true});
                const ch::Choice direct = ch::choose(b.model, b.card, b.quants[0], allowed, pins, task,
                                                     oracle(b, 40e9), ch::Gates{true, true});
                const auto read_back = nlohmann::ordered_json::parse(record.inputs.dump());
                const ch::Choice again = ch::replay_choice(read_back);
                ++replays;
                check(ch::choice_summary(record.choice) == ch::choice_summary(direct) &&
                          ch::choice_summary(again) == ch::choice_summary(direct),
                      label + ", " + std::to_string(context) + (pinned ? " pinned" : "") +
                          ": the record replays to the same choice");
            }
        }
    }
    check(replays == static_cast<int>(benches.size()) * 4, "every bench replayed");
    // Incomplete: a candidate's own context the record lacks.
    const Bench b = nine_b(fixture, "RTX 4090");
    const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
    ch::ChoiceRecord record = ch::record_choice(b.model, b.card, b.quants[0], allowed, {}, {}, oracle(b, 40e9),
                                                ch::Gates{true, true});
    auto lacking = record.inputs;
    lacking["own_context"].erase(lacking["own_context"].begin());
    bool incomplete = false;
    try {
        (void)ch::replay_choice(lacking);
    } catch (const ch::ReplayIncomplete&) { incomplete = true; }
    check(incomplete, "a record lacking a candidate's own context is incomplete");
    auto other = record.inputs;
    other["schema"] = "ninferno-choice-replay/0";
    bool refused = false;
    try {
        (void)ch::replay_choice(other);
    } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "a record of another schema is refused");
    // A record of the schema before typed_resolution (/1) is refused, naming the version it
    // carries; no default stands in for the gate it lacks.
    auto old_record      = record.inputs;
    old_record["schema"] = "ninferno-choice-replay/1";
    old_record["gates"].erase("typed_resolution");
    std::string old_message;
    try {
        (void)ch::replay_choice(old_record);
    } catch (const std::invalid_argument& e) { old_message = e.what(); }
    check(old_message.find("replay/1") != std::string::npos &&
              old_message.find("replay/3") != std::string::npos,
          "a schema-1 record is refused with both versions named");
    // A record of the schema before the banks' int8 route (/2) is refused too: its quantization
    // has no expert_banks_int8 or expert_banks_assumed to replay.
    {
        auto two      = record.inputs;
        two["schema"] = "ninferno-choice-replay/2";
        two["quant"].erase("expert_banks_int8");
        two["quant"].erase("expert_banks_assumed");
        bool two_refused = false;
        try {
            (void)ch::replay_choice(two);
        } catch (const std::invalid_argument&) { two_refused = true; }
        check(two_refused, "a schema-2 record is refused");
    }
    // The banks' int8 route and the assumed flag go through the record (#769): an MoE whose banks
    // are A16 under fast activations, Int8 under int8, on an assumed format, int8 pinned. The
    // record replays to the same choice; without the two fields it would not.
    {
        const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
        Bench moe{model_of(m35), card_of(fixture.at("cards").at("RTX 4090")), {}};
        moe.model.linears      = qwen36_moe_linears();
        moe.model.layers       = 40;
        moe.model.hidden       = 2048;
        ch::QuantOption q      = quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS"));
        q.rest_weights         = ch::WeightClass::Q8;
        q.expert_banks         = ch::ExpertBanks::A16;
        q.expert_banks_int8    = ch::ExpertBanks::Int8;
        q.expert_banks_assumed = true;
        moe.quants.push_back(q); // the oracle's weights
        ch::Pins pins;
        pins.activations              = ActivationMode::Int8;
        const ch::Allowed moe_allowed = ch::allowed_axes(moe.model, KvCacheStorage::Int8Group64);
        ch::Task task;
        task.contexts = {4096, 2048};
        const ch::ChoiceRecord rec =
            ch::record_choice(moe.model, moe.card, q, moe_allowed, pins, task, oracle(moe, 40e9),
                              ch::Gates{true, true});
        const auto read_back = nlohmann::ordered_json::parse(rec.inputs.dump());
        check(read_back.at("quant").at("expert_banks_int8") ==
                      static_cast<int>(ch::ExpertBanks::Int8) &&
                  read_back.at("quant").at("expert_banks_assumed") == true,
              "replay: the banks' int8 route and the assumed flag are recorded");
        check(ch::choice_summary(ch::replay_choice(read_back)) == ch::choice_summary(rec.choice),
              "replay: an int8-pinned MoE on assumed banks replays to the same choice");
        auto dropped                             = read_back;
        dropped["quant"]["expert_banks_int8"]    = nullptr;
        dropped["quant"]["expert_banks_assumed"] = false;
        check(ch::choice_summary(ch::replay_choice(dropped)) != ch::choice_summary(rec.choice),
              "replay: without the two fields the choice differs");
        auto none                          = read_back;
        none["quant"]["expert_banks_int8"] = nullptr;
        bool none_read                     = true;
        try {
            (void)ch::replay_choice(none);
        } catch (const std::exception&) { none_read = false; }
        check(none_read, "replay: a null int8 route is read (none given)");
    }
    auto no_gate = record.inputs;
    no_gate["gates"].erase("typed_resolution");
    bool gate_refused = false;
    try {
        (void)ch::replay_choice(no_gate);
    } catch (const std::invalid_argument&) { gate_refused = true; }
    check(gate_refused, "a current-schema record without typed_resolution is refused");
    // The gate round-trips at both values, and the replay answers what was recorded.
    for (const bool typed : {false, true}) {
        ch::Gates gates{true, true};
        gates.typed_resolution     = typed;
        const ch::ChoiceRecord rec = ch::record_choice(b.model, b.card, b.quants[0], allowed, {},
                                                       {}, oracle(b, 40e9), gates);
        const auto back            = nlohmann::ordered_json::parse(rec.inputs.dump());
        check(back.at("gates").at("typed_resolution").get<bool>() == typed,
              "typed_resolution is written to the record");
        check(ch::choice_summary(ch::replay_choice(back)) == ch::choice_summary(rec.choice),
              "a record with typed_resolution replays to the choice it recorded");
    }
    // Malformed, not wrapped or thrown past the contract: a negative own context, a schema that
    // is no string.
    for (const auto& corrupt : {[&] {
                                    auto r = record.inputs;
                                    r["own_context"].begin().value() = -1;
                                    return r;
                                }(),
                                [&] {
                                    auto r      = record.inputs;
                                    r["schema"] = 1;
                                    return r;
                                }(),
                                [&] {
                                    auto r                        = record.inputs;
                                    r["model"]["default_drafter"] = -1;
                                    return r;
                                }(),
                                [&] {
                                    auto r = record.inputs;
                                    r["model"]["kv_bytes_per_token"]["0junk"] = 1;
                                    return r;
                                }(),
                                [&] {
                                    auto r = record.inputs;
                                    r["model"]["state_bytes"]["99"] = 1;
                                    return r;
                                }(),
                                [&] {
                                    auto r                   = record.inputs;
                                    r["task"]["contexts"]    = nullptr;
                                    return r;
                                }(),
                                [&] {
                                    auto r                     = record.inputs;
                                    r["model"]["linears"]      = 7;
                                    return r;
                                }(),
                                [&] {
                                    auto r                       = record.inputs;
                                    r["allowed"]["automatic_kv"] = 4294967296LL;
                                    return r;
                                }(),
                                [&] {
                                    auto r                = record.inputs;
                                    r["card"]["fp4"]      = 1;
                                    return r;
                                }()}) {
        bool malformed = false;
        try {
            (void)ch::replay_choice(corrupt);
        } catch (const std::invalid_argument&) { malformed = true; }
        check(malformed, "a malformed record is refused as invalid_argument");
    }
}

// A one-row step's named low-bit linear on R8 (kR8DecodeIssue): its read takes the longer of its DRAM
// time and its warp instructions at the generation's issue rate. Against route-tune's fastest T = 1 R8
// time of a 16384 x 5120 IQ2_XXS store (master 75b7209e, 2026-11-06), less the launch (kKernelCost):
// Ampere within 10 % on both its cards; Ada, Blackwell and Turing (no rate: kR8IssueRate) and a 4-bit
// format at the DRAM rate as before; two rows and the exact route (no R8) unchanged.
void prices_r8_issue() {
    const double n = 16384, k = 5120, bytes = n * k * 2.0625 / 8;
    const auto step = [&](const ch::CardFacts& card, const std::string& format, std::uint32_t rows,
                          ActivationMode activations, bool* transferred = nullptr) {
        ch::ModelFacts m;
        m.family        = "qwen3_5_text";
        m.params        = n * k;
        m.active_params = n * k;
        m.linears.push_back({n, k, 1, false, false, format});
        ch::QuantOption q;
        q.weights      = ch::WeightClass::Other;
        q.weight_bytes = bytes;
        ch::Candidate c;
        c.activations     = activations;
        const ch::Speed s = ch::speed_of(m, card, q, c, rows, 1);
        if (transferred != nullptr) { *transferred = s.transferred; }
        return 1 / s.decode_per_stream;
    };
    struct Census {
        ch::CardFacts card;
        double us; // route-tune's fastest R8 schedule
        double within;
    };

    const Census rows[] = {
        {board_card(0x2204), 37.9, 0.10},
        {board_card(0x2504), 100.9, 0.10},
    };
    for (const Census& r : rows) {
        const std::string_view gen = ch::generation_of(r.card);
        double launch              = 0;
        for (const ch::KernelCost& c : ch::kKernelCost) {
            if (c.generation == gen) { launch = c.gemv; }
        }
        const double excess = step(r.card, "ggml_iq2_xxs", 1, ActivationMode::Fast) - step(r.card, "", 1, ActivationMode::Fast);
        const double kernel = bytes / (ch::kDramEta * r.card.dram_bytes_per_s) + excess + launch;
        const double d      = kernel * 1e6 / r.us - 1;
        std::cout << "R8 IQ2_XXS T=1, " << gen << ' ' << r.card.sms << " SMs: " << kernel * 1e6 << " us against " << r.us
                  << " (" << d * 100 << " %)\n";
        check(excess > 0 && std::abs(d) < r.within, "a low-bit linear's R8 read at its instructions' rate");
        check(step(r.card, "ggml_q4_k", 1, ActivationMode::Fast) == step(r.card, "", 1, ActivationMode::Fast),
              "a 4-bit linear's R8 read at the DRAM rate");
        check(step(r.card, "ggml_iq2_xxs", 2, ActivationMode::Fast) == step(r.card, "", 2, ActivationMode::Fast),
              "two rows: no R8 issue term");
        check(step(r.card, "ggml_iq2_xxs", 1, ActivationMode::Exact) == step(r.card, "", 1, ActivationMode::Exact),
              "the exact route reads no R8");
    }
    // Every row's generation and format: a generation with an issue rate, a format R8 decodes (a
    // codec of r8_supports: the IQ1_S..IQ3_S codebooks, Q2_K, Q3_K).
    for (const ch::R8DecodeIssue& f : ch::kR8DecodeIssue) {
        const bool rated = std::any_of(std::begin(ch::kR8IssueRate), std::end(ch::kR8IssueRate),
                                       [&](const ch::R8IssueRate& r) { return r.generation == f.generation; });
        check(rated && f.bits > 1 && f.bits < 4 && f.warp_inst_per_kib > 0, "an R8 issue row of a rated generation");
    }
    const ch::CardFacts unrated[] = {
        board_card(0x2684),
        board_card(0x2C05),
        board_card(0x1E07),
    };
    for (const ch::CardFacts& card : unrated) {
        check(step(card, "ggml_iq2_xxs", 1, ActivationMode::Fast) == step(card, "", 1, ActivationMode::Fast),
              "a generation without an issue rate reads at the DRAM rate");
    }
}

// The tie rule inside the resolution (#301, DECISIONS 75), on a constructed case. A is the best
// worth (KLD 0.02, 1000 tokens). B: 1.5 % slower, KLD 0.01. C: as B but KLD 0.005 and slower by
// 1.6 %. D: 5 % slower, KLD 0.001. E: as B, listed after it. F: as A in KLD and context, listed
// before it. At a resolution of 2 % the lowest KLD inside it wins (C), not D (slower than the
// resolution), and context decides nothing.
void resolution_rule() {
    const auto make = [](double seconds, double kld, std::uint32_t context) {
        ch::Scored s;
        s.decision_seconds = seconds;
        s.accuracy.kld     = kld;
        s.own_context      = context;
        s.worth            = -std::log(seconds) - kld;
        return s;
    };
    const std::vector<ch::Scored> scored{make(1.0, 0.02, 1000),   make(1.015, 0.01, 1000),
                                         make(1.016, 0.005, 500), make(1.05, 0.001, 9000),
                                         make(1.015, 0.01, 900),  make(1.0, 0.02, 1000)};
    const std::vector<std::size_t> all{0, 1, 2, 3, 4, 5};
    check(ch::resolve_within_resolution(scored, all, 0, 0.02) == 2,
          "inside the resolution: the lowest KLD");
    check(ch::resolve_within_resolution(scored, all, 0, 0) == 0, "no resolution: the best worth");
    check(ch::resolve_within_resolution(scored, all, 0, 0.01) == 0,
          "outside the resolution: the best worth");
    check(ch::resolve_within_resolution(scored, all, 0, 0.06) == 3,
          "a wider resolution reaches the 5 % one");
    check(ch::resolve_within_resolution(scored, {0, 4, 1}, 0, 0.02) == 1,
          "equal KLD: the lower index");
    check(ch::resolve_within_resolution(scored, {4, 1}, 4, 0.02) == 4,
          "equal KLD to the anchor: it stays");
    // A lower index of equal KLD and equal context must not move the pick (#507: BF16 over INT8).
    check(ch::resolve_within_resolution(scored, {5, 0}, 0, 0.02) == 0,
          "equal KLD and context: the anchor stays");
    // More context alone never moves it.
    std::vector<ch::Scored> roomy = scored;
    roomy[1].accuracy.kld         = 0.02;
    roomy[1].own_context          = 90000;
    check(ch::resolve_within_resolution(roomy, {0, 1}, 0, 0.02) == 0,
          "more context is headroom, not a reason");
    // A NaN KLD never wins.
    roomy[1].accuracy.kld = std::nan("");
    check(ch::resolve_within_resolution(roomy, {0, 1}, 0, 0.02) == 0, "NaN KLD: the anchor stays");
    std::vector<ch::Scored> none = scored;
    none[0].decision_seconds     = 0;
    check(ch::resolve_within_resolution(none, all, 0, 0.02) == 0,
          "no reference estimate: unchanged");
}

// #618 step 8: properties of the choice over every board of src/core/data/gpu_facts.json. Host
// only, no behaviour change. A property that fails is a finding: it is printed as "FINDING" with
// the offending case; any finding fails the test. What is not an invariant of the rules (a
// snapshot of today's picks) is printed as "NOTE" and fails nothing.
struct PropertyBoard {
    std::string name;
    std::string published; // the row's facts without its name and PCI id
    int boost_min_mhz = 0; // a laptop row's bottom of the published boost range (0: none)
    ch::CardFacts card;
};

// Every board as the plan prices it: product::gpu_key_of's synthetic class of the row's PCI id,
// compute capability and SMs with the table's facts, through choice::card_facts_of, the function
// gpu_key_of takes its rates from (a laptop at the bottom of its published boost range, NVFP4 from
// the architecture's features).
std::vector<PropertyBoard> gpu_fact_boards() {
    std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/src/core/data/gpu_facts.json");
    const json facts = json::parse(in);
    std::vector<PropertyBoard> out;
    for (const json& g : facts.at("gpus")) {
        PropertyBoard b;
        b.name        = g.at("name").get<std::string>();
        const int cc  = g.at("cc").get<int>();
        const int sms = g.at("sms").get<int>();
        const auto id =
            static_cast<std::uint32_t>(std::stoul(g.at("id").get<std::string>(), nullptr, 16));
        ninferno::DeviceClass d = ninferno::synthetic_device_class(id, cc / 10, cc % 10, sms, 0, 0);
        d.facts                 = ninferno::machine_facts_of(d);
        b.card                  = ch::card_facts_of(d);
        b.boost_min_mhz         = g.value("boost_min_mhz", 0);
        b.published             = std::to_string(cc) + "/" + std::to_string(sms) + "/" +
                      g.at("form").get<std::string>() + "/" + g.at("boost_mhz").dump() + "/" +
                      g.value("boost_min_mhz", json(0)).dump() + "/" + g.at("bus_bits").dump() +
                      "/" + g.at("gbps").dump() + "/" + g.value("l2_mib", json(0)).dump();
        out.push_back(std::move(b));
    }
    return out;
}

std::string pick_of(const ch::Choice& c) {
    if (!c.chosen) { return "none"; }
    const ch::Scored& s    = chosen(c);
    const ch::Candidate& k = s.candidate;
    return std::string(kv_cache_storage_name(k.kv)) + "/act" +
           std::to_string(static_cast<int>(k.activations)) + "/chunk" + std::to_string(k.chunk) +
           "/drafter" + std::to_string(k.drafter) + "/state" +
           std::to_string(static_cast<int>(k.state)) + "/ring" +
           std::to_string(static_cast<int>(k.ring));
}

void finding(std::vector<std::string>& findings, const std::string& text) {
    findings.push_back(text);
    std::cout << "  FINDING: " << text << '\n';
}

void choice_properties(const json& fixture) {
    const std::vector<PropertyBoard> boards = gpu_fact_boards();
    check(boards.size() >= 100, "gpu_facts.json lists the boards");
    // The boards are priced as the plan prices them: a laptop row at the bottom of its published
    // boost range (machine_facts), not at its top.
    int laptops = 0;
    for (const PropertyBoard& pb : boards) {
        if (pb.boost_min_mhz == 0) { continue; }
        ++laptops;
        check(pb.card.clock_hz == pb.boost_min_mhz * 1e6,
              "a laptop board is priced at its boost_min_mhz: " + pb.name);
    }
    check(laptops > 0, "gpu_facts.json has laptop rows with a published boost range");
    std::vector<std::string> findings;
    const json& m35     = fixture.at("models").at("Qwen3.6-35B-A3B");
    const auto bench_on = [&](const ch::CardFacts& card, bool moe) {
        Bench b = nine_b(fixture, "RTX 4090");
        if (moe) {
            b = Bench{
                model_of(m35), card, {quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS"))}};
            b.model.drafters        = {{"dflash:7", "dflash", 2.5, 2, "measured"},
                                       {"mtp:4", "mtp", 1.9, 2, "measured"}};
            b.model.default_drafter = 2;
        }
        b.card = card;
        return b;
    };
    const auto key_of = [](const ch::CardFacts& c) {
        return std::to_string(c.major) + "." + std::to_string(c.minor) + "/" +
               std::to_string(c.sms) + "/" + std::to_string(c.clock_hz) + "/" +
               std::to_string(c.dram_bytes_per_s) + "/" + std::to_string(c.l2_bytes) + "/" +
               (c.fp4 ? "fp4" : "");
    };
    ch::Task task;
    task.contexts = {32768};

    // (a) Determinism and equal facts. (a1) Rows of gpu_facts.json that publish the same facts
    // under another name or PCI id (an RTX 5090 D, a laptop board's second id) are priced alike
    // and pick alike: the production derivation reads the PCI id to find the row (machine_facts),
    // so a name or id leaking into the card facts would show here. (a2) Boards of one generation
    // picking different candidates is printed, not asserted: choose() reads each board's own SMs,
    // clock and DRAM rate (e.g. a DRAM rate under kVoltaCopyMinDramBps), so it is a snapshot of
    // today's rules, not an invariant; boards whose facts are equal are (a1). (a3) The same call
    // twice gives the same pick (no hidden mutable state).
    std::map<std::string, std::tuple<std::string, std::string, std::string>> by_row;
    std::map<std::string, std::pair<std::string, std::string>> by_generation;
    std::set<std::string> generation_noted;
    int equal_fact_rows = 0, repeats = 0, generation_notes = 0;
    for (const PropertyBoard& pb : boards) {
        for (const bool moe : {false, true}) {
            const Bench b             = bench_on(pb.card, moe);
            const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
            const ch::Choice first =
                ch::choose(b.model, b.card, b.quants[0], allowed, {}, task, oracle(b, 20e9));
            const std::string pick = pick_of(first);
            ++repeats;
            if (pick != pick_of(ch::choose(b.model, b.card, b.quants[0], allowed, {}, task,
                                           oracle(b, 20e9)))) {
                finding(findings, "(a3) the same call twice picks differently: " + pb.name +
                                      (moe ? " MoE" : " dense"));
            }
            const std::string kind = moe ? " MoE" : " dense";
            auto [row, fresh_row] =
                by_row.try_emplace(pb.published + kind, key_of(pb.card), pick, pb.name);
            if (!fresh_row) {
                const auto& [card, row_pick, row_name] = row->second;
                ++equal_fact_rows;
                if (card != key_of(pb.card)) {
                    finding(findings, "(a1) equal published facts, other card facts: " + pb.name +
                                          " " + key_of(pb.card) + ", " + row_name + " " + card);
                } else if (row_pick != pick) {
                    finding(findings, "(a1) equal facts, different pick: " + pb.name + " picks " +
                                          pick + ", " + row_name + " picks " + row_pick);
                }
            }
            const std::string key = std::string(ch::generation_of(pb.card)) + kind;
            auto [git, gfresh]    = by_generation.try_emplace(key, pick, pb.name);
            if (!gfresh && git->second.first != pick && generation_noted.insert(key).second) {
                ++generation_notes;
                std::cout << "  NOTE (a2) generation " << key << ": " << pb.name << " picks "
                          << pick << ", " << git->second.second << " picks " << git->second.first
                          << '\n';
            }
        }
    }

    // (b) Monotonicity, one board of each generation, read from choose()'s output. More weight on
    // accuracy (lambda) never raises the chosen KLD (only where the worth decides: a board whose
    // generation has a deciding reference card); a longer asked context never raises the most
    // streams that hold it (the memory law: a candidate's room per stream does not depend on the
    // asked context, so what holds a longer one holds a shorter one); more free memory never
    // lowers the stage (the context the choice settles on, Auto's steps). The streams run to
    // kMaximumConcurrency, the product's limit. The KV bytes a token the choice settles on as the
    // context grows are printed, not asserted: a codec whose dequantisation costs more time than
    // its bytes save can legitimately go wider at a longer context, so the cost law holds no
    // monotone order there.
    std::set<std::string> seen;
    int lambda_boards = 0, boards_checked = 0, kv_notes = 0;
    for (const PropertyBoard& pb : boards) {
        const std::string gen(ch::generation_of(pb.card));
        if (!seen.insert(gen).second) { continue; }
        ++boards_checked;
        const Bench b             = bench_on(pb.card, false);
        const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
        double last_kld           = INFINITY;
        bool decides              = false;
        for (const double lambda : {0.0, 5.0, 25.0, 100.0, 400.0}) {
            ch::Task t        = task;
            t.accuracy_weight = lambda;
            const ch::Choice c =
                ch::choose(b.model, b.card, b.quants[0], allowed, {}, t, oracle(b, 20e9));
            if (!c.chosen || !c.by_worth) { break; }
            decides          = true;
            const double kld = chosen(c).accuracy.kld;
            if (kld > last_kld + 1e-12) {
                finding(findings, "(b) " + gen + " (" + pb.name + "): lambda " +
                                      std::to_string(lambda) + " raises the chosen KLD " +
                                      std::to_string(last_kld) + " -> " + std::to_string(kld));
            }
            last_kld = kld;
        }
        lambda_boards += decides ? 1 : 0;
        int last_streams = 1 << 30;
        double last_kv   = INFINITY;
        for (const std::uint32_t context : {4096U, 8192U, 16384U, 32768U, 65536U, 131072U}) {
            int most = 0;
            for (int streams = 1; streams <= static_cast<int>(ninferno::kMaximumConcurrency);
                 ++streams) {
                ch::Task t        = task;
                t.contexts        = {context};
                t.concurrency     = static_cast<std::uint32_t>(streams);
                const auto shared = [&](const ch::Candidate& cand) -> std::uint32_t {
                    const double room = 20e9 - b.quants[0].weight_bytes;
                    return room <= 0 ? 0
                                     : static_cast<std::uint32_t>(
                                           room / streams / b.model.kv_bytes_per_token(cand.kv));
                };
                if (ch::choose(b.model, b.card, b.quants[0], allowed, {}, t, shared).chosen) {
                    most = streams;
                }
            }
            if (most > last_streams) {
                finding(findings, "(b) " + gen + " (" + pb.name + "): context " +
                                      std::to_string(context) + " holds " + std::to_string(most) +
                                      " streams, a shorter one " + std::to_string(last_streams));
            }
            last_streams = most;
            ch::Task t   = task;
            t.contexts   = {context};
            const ch::Choice c =
                ch::choose(b.model, b.card, b.quants[0], allowed, {}, t, oracle(b, 40e9));
            if (c.chosen) {
                const double kv = b.model.kv_bytes_per_token(chosen(c).candidate.kv);
                if (kv > last_kv + 1e-9) {
                    ++kv_notes;
                    std::cout << "  NOTE (b) " << gen << " (" << pb.name << "): context " << context
                              << " settles on wider KV, " << last_kv << " -> " << kv
                              << " bytes a token\n";
                }
                last_kv = kv;
            }
        }
        std::uint32_t last_stage = 0;
        for (const double budget : {8e9, 10e9, 12e9, 16e9, 24e9, 40e9}) {
            ch::Task t = task;
            t.contexts = {131072, 65536, 32768, 16384, 8192};
            const ch::Choice c =
                ch::choose(b.model, b.card, b.quants[0], allowed, {}, t, oracle(b, budget));
            const std::uint32_t stage = c.chosen ? c.context : 0;
            if (stage < last_stage) {
                finding(findings, "(b) " + gen + " (" + pb.name +
                                      "): " + std::to_string(budget / 1e9) + " GB settles on " +
                                      std::to_string(stage) + ", less memory on " +
                                      std::to_string(last_stage));
            }
            last_stage = stage;
        }
    }

    // (c) Listing permutation: the order the axes' values and the drafters are listed in does not
    // change the pick (three fixed shuffles of each). True by construction (choose() puts each axis
    // in its canonical order, #301) and kept as a regression guard for that; the quantization rows,
    // which choose() does not sort, are permuted in test_plan_dry_run.cpp through plan_quants.
    int permuted = 0;
    for (std::size_t i = 0; i < boards.size(); i += 4) {
        const PropertyBoard& pb   = boards[i];
        const Bench b             = bench_on(pb.card, true);
        const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
        const auto labelled       = [](const Bench& x, const ch::Choice& c) {
            const ch::Candidate& k = chosen(c).candidate;
            return std::string(kv_cache_storage_name(k.kv)) + "/" +
                   std::to_string(static_cast<int>(k.activations)) + "/" + std::to_string(k.chunk) +
                   "/" + (k.drafter == 0 ? "plain" : x.model.drafters[k.drafter - 1].label);
        };
        const ch::Choice base =
            ch::choose(b.model, b.card, b.quants[0], allowed, {}, task, oracle(b, 20e9));
        if (!base.chosen) { continue; }
        for (unsigned seed = 1; seed <= 3; ++seed) {
            Bench shuffled    = b;
            ch::Allowed order = allowed;
            std::mt19937 rng(seed * 7919U + static_cast<unsigned>(i));
            std::shuffle(order.kv.begin(), order.kv.end(), rng);
            std::shuffle(order.activations.begin(), order.activations.end(), rng);
            std::shuffle(shuffled.model.drafters.begin(), shuffled.model.drafters.end(), rng);
            // default_drafter indexes the listing: follow the drafter it names.
            for (std::size_t d = 0; d < shuffled.model.drafters.size(); ++d) {
                if (shuffled.model.drafters[d].label ==
                    b.model.drafters[b.model.default_drafter - 1].label) {
                    shuffled.model.default_drafter = d + 1;
                }
            }
            ++permuted;
            const ch::Choice again = ch::choose(shuffled.model, shuffled.card, shuffled.quants[0],
                                                order, {}, task, oracle(shuffled, 20e9));
            if (!again.chosen || labelled(b, base) != labelled(shuffled, again)) {
                finding(findings,
                        "(c) " + pb.name + " seed " + std::to_string(seed) + ": " +
                            labelled(b, base) + " against " +
                            (again.chosen ? labelled(shuffled, again) : std::string("none")));
            }
        }
    }

    // (d) The plan's choice line equals what the start picks: both reach product::choose_at_weights
    // (serve/plan_offline.cpp:597 and :879) but from different inputs (the plan's `inputs_for` of
    // the chosen listing row, the start's `planning.inputs`); a CPU test needs an artifact and a
    // device, so it is not part of this host test (listed in the PR).
    std::cout << "choice properties: " << boards.size() << " boards, " << repeats
              << " repeated calls, " << equal_fact_rows
              << " rows sharing another row's published facts (dense and MoE), " << boards_checked
              << " generations swept (lambda on " << lambda_boards << " where the worth decides), "
              << permuted << " permutations, " << findings.size() << " findings, "
              << generation_notes << " generation notes, " << kv_notes << " KV notes\n";
    check(equal_fact_rows > 0, "gpu_facts.json has rows with equal published facts to compare");
    check(lambda_boards > 0, "the lambda sweep covers a board where the worth decides");
    check(findings.empty(), "no property finding (the FINDING lines above)");
}

// The resolution of a Transferred estimate (#618 step 3), derived from decode_measured.json per
// target generation: the rms of ln(predicted / measured) when the rows of the lender
// (kTransferGeneration) are carried onto the cards of a generation that states no error of its own
// (kDecodePredictionError: Ampere, Turing, Volta). Each card is priced as a card of the lender's
// compute capability with its own bytes rate, SMs and clock; every short-context row (not marked
// long) counts, the long rows have their own guards in decodes_from_facts. The rows are not pooled.
// Volta's row also carries the lender's route and ISA pricing (no int8 MMA, HBM2). The constants
// are checked against it, not fitted.
void transferred_resolution(const json& fixture) {
    // The lender's compute capability, from the generation table.
    int lender_major = 0, lender_minor = 0;
    for (const ninferno::GenerationRow& row : ninferno::kGenerations) {
        if (std::string_view(row.name) == ch::kTransferGeneration) {
            lender_major = row.major, lender_minor = row.minor_lo;
        }
    }
    check(lender_major > 0, "the lender is a generation of the table");
    std::map<std::string, std::pair<double, int>> by_target;
    std::map<std::string, std::set<std::string>> cards_of; // the lab cards behind each target
    for (const json& row : fixture.at("rows")) {
        if (row.value("long", false)) { continue; }
        const json& m           = fixture.at("models").at(row.at("model").get<std::string>());
        const std::string q     = row.at("quant").get<std::string>();
        const ch::ModelFacts mf = model_at(m, q);
        ch::CardFacts cf      = card_of(fixture.at("cards").at(row.at("card").get<std::string>()));
        const std::string gen = std::string(ch::generation_of(cf));
        if (std::any_of(std::begin(ch::kDecodePredictionError),
                        std::end(ch::kDecodePredictionError),
                        [&](const ch::PredictionError& e) { return e.generation == gen; })) {
            continue; // its own rows state its error
        }
        cf.major                 = lender_major,
        cf.minor                 = lender_minor; // carried: the lender's rows, this board's rates
        const ch::QuantOption qo = quant_of(q, m.at("quants").at(q));
        ch::Candidate c;
        c.kv    = kv_of(row.at("kv").get<std::string>());
        c.ring  = kv_of(row.at("ring").get<std::string>());
        c.state = row.at("state").get<std::string>() == "f32" ? RecurrentStateStorage::FP32
                                                              : RecurrentStateStorage::FP16;
        const ch::Speed s = ch::speed_of(mf, cf, qo, c, row.at("concurrency").get<std::uint32_t>(),
                                         row.at("context").get<double>());
        const double e    = std::log(s.decode_per_stream / row.at("decode").get<double>());
        auto& g           = by_target[gen];
        g.first += e * e, ++g.second;
        cards_of[gen].insert(row.at("card").get<std::string>());
    }
    double worst = 0;
    for (const ch::TransferredError& t : ch::kTransferredDecodeError) {
        const auto it = by_target.find(std::string(t.target));
        check(it != by_target.end() && it->second.second == 12,
              std::string(t.target) + ": 12 carried rows");
        if (it == by_target.end()) { continue; }
        const double rms = std::sqrt(it->second.first / it->second.second);
        std::cout << "  carried " << ch::kTransferGeneration << " -> " << t.target << ": "
                  << it->second.second << " points, rms " << rms << " (stated " << t.rms << ")\n";
        check(std::abs(rms - t.rms) < 0.0015,
              std::string(t.target) +
                  ": the stated transferred resolution is the carried rows' rms");
        // Its label: the lab cards it rests on, by count and name (one each today: it never
        // decides, Resolution::decides). Adding a card's rows to the fixture must relabel the
        // constant (`cards`, `card`) and recompute its rms; at two cards the row decides.
        const std::set<std::string>& cards = cards_of[std::string(t.target)];
        check(t.cards == static_cast<int>(cards.size()) && t.cards == 1 &&
                  cards.count(std::string(t.card)) == 1,
              std::string(t.target) + ": the carry's lab cards are labelled");
        worst = std::max(worst, rms);
    }
    check(by_target.size() == std::size(ch::kTransferredDecodeError),
          "every carried target has a stated row");
    check(std::abs(worst - ch::kUnmeasuredCarryDecodeError) < 0.0015 &&
              ch::kUnmeasuredCarryDecodeError == ch::largest_transferred_decode_error(),
          "the unmeasured carry is the largest measured transfer");
    // The Own rows' labels: two or more lab cards each (they may decide), named.
    for (const ch::PredictionError& e : ch::kDecodePredictionError) {
        int named = 0;
        for (const auto& [name, unused] : fixture.at("cards").items()) {
            (void)unused;
            const bool of_gen =
                ch::generation_of(card_of(fixture.at("cards").at(name))) == e.generation;
            named += of_gen && e.card_names.find(name) != std::string_view::npos;
        }
        check(e.cards >= 2 && named == e.cards,
              std::string(e.generation) + ": the own error's lab cards are labelled");
    }
    // The basis by board and by the estimates compared (resolution_of, #618): Own estimates on Ada
    // and Blackwell state their own rows' error and decide; on the other known generations their
    // carried row (one card: never decides); a Transferred estimate on a generation with its own
    // error has no measured carry: the assumed worst case; lent estimates the unmeasured carry.
    const auto carry_of = [](std::string_view gen) {
        for (const ch::TransferredError& t : ch::kTransferredDecodeError) {
            if (t.target == gen) { return t.rms; }
        }
        return 0.0;
    };
    using B = ch::ResolutionBasis;
    using S = ch::SpeedBasis;

    struct Want {
        const char* card;
        S estimates;
        B basis;
        bool own_number; // the generation's own error, else its carry or the assumed worst case
        bool decides;
    };

    for (const Want& w :
         std::vector<Want>{{"RTX 4090", S::Own, B::Own, true, true},
                           {"RTX 5090", S::Own, B::Own, true, true},
                           {"RTX 4090", S::Transferred, B::Transferred, false, false},
                           {"RTX 4090", S::LentGeneration, B::Unmeasured, false, false},
                           {"RTX 3060", S::Own, B::Transferred, false, false},
                           {"RTX 3060", S::Transferred, B::Transferred, false, false},
                           {"RTX 2080 Ti", S::Own, B::Transferred, false, false},
                           {"Quadro GV100", S::Transferred, B::Transferred, false, false}}) {
        if (!fixture.at("cards").contains(w.card)) { continue; }
        const ch::CardFacts card   = card_of(fixture.at("cards").at(w.card));
        const ch::Resolution r     = ch::resolution_of(card, w.estimates);
        const std::string_view gen = ch::generation_of(card);
        const double want          = w.own_number               ? ch::worth_band(card)
                                     : w.basis == B::Unmeasured ? ch::kUnmeasuredCarryDecodeError
                                     : carry_of(gen) > 0        ? carry_of(gen)
                                                                : ch::kUnmeasuredCarryDecodeError;
        check(r.basis == w.basis && r.rms == want && want > 0 && r.decides == w.decides,
              std::string(w.card) + " " + std::string(ch::speed_basis_name(w.estimates)) +
                  ": the resolution's basis, width and whether it decides");
    }
    ch::CardFacts lent           = card_of(fixture.at("cards").at("RTX 5090"));
    lent.major                   = 13;
    lent.minor                   = 0;
    const ch::Resolution carried = ch::resolution_of(lent, S::LentGeneration);
    check(carried.basis == B::Unmeasured && carried.rms == ch::kUnmeasuredCarryDecodeError &&
              !carried.decides,
          "a compute capability with no generation row: the unmeasured carry, not deciding");
    // Its floor is the lent generation's own band (generation_of: its rows price the board), not
    // the lender's: cc 13.0 is priced from Blackwell's rows, floor 0.0170, Ada's is 0.0226.
    check(carried.floor == ch::worth_band(lent) && carried.floor > 0 &&
              ch::resolution_of(lent, S::Transferred).floor == carried.floor &&
              ch::resolution_of(lent, S::Own).floor == carried.floor,
          "a compute capability with no generation row: the lent generation's floor");
    {
        // The review's case at cc 13.0: B 2 % slower (ln 0.0198), above the 0.0170 floor and
        // below Ada's 0.0226: the best worth A stays, as on master.
        const auto make = [](double seconds, double kld, S basis) {
            ch::Scored s;
            s.decision_seconds = seconds;
            s.accuracy.kld     = kld;
            s.own_context      = 1000;
            s.decision_basis   = basis;
            s.worth            = -std::log(seconds) - 0.3 * kld;
            return s;
        };
        const ch::Resolutions bands{ch::resolution_of(lent, S::Own),
                                    ch::resolution_of(lent, S::Transferred),
                                    ch::resolution_of(lent, S::LentGeneration)};
        const std::vector<ch::Scored> ab{make(1.00, 0.020, S::LentGeneration),
                                         make(1.020, 0.005, S::LentGeneration)};
        check(ch::resolve_within_resolution(ab, {0, 1}, 0, bands) == 0,
              "cc 13.0: a candidate beyond the lent generation's floor does not displace the best");
    }
    // A one-card row of kDecodePredictionError would not zero the floor that decides: it falls
    // back to the lender's row of two or more cards (resolution_of; no such row exists today, the
    // table is constexpr, so this is held by the code and the check below that every row has two).
    for (const ch::PredictionError& e : ch::kDecodePredictionError) {
        check(e.cards >= 2, std::string(e.generation) + ": its own error rests on two cards");
    }
    // A one-card or assumed band only widens: on every generation and every basis of the estimates
    // compared, never narrower than the Own band they rest on or are carried from (the
    // generation's own error, else the lender's), and only an Own band of two cards decides. A
    // comparison rests on its widest estimate (mixed Own and Transferred: the Transferred band;
    // mixed with a lent estimate: the unmeasured carry).
    double lender_band = 0;
    for (const ch::PredictionError& e : ch::kDecodePredictionError) {
        if (e.generation == ch::kTransferGeneration) { lender_band = e.rms; }
    }
    check(lender_band > 0, "the lender states its own error");
    const auto same = [](const ch::Resolution& a, const ch::Resolution& b) {
        return a.basis == b.basis && a.rms == b.rms && a.decides == b.decides;
    };
    for (const ninferno::GenerationRow& row : ninferno::kGenerations) {
        ch::CardFacts card = card_of(fixture.at("cards").at("RTX 4090"));
        card.major = row.major, card.minor = row.minor_lo;
        const double own           = ch::worth_band(card);
        const ch::Resolution own_r = ch::resolution_of(card, S::Own);
        for (const S s : {S::Own, S::Transferred, S::LentGeneration}) {
            const ch::Resolution r = ch::resolution_of(card, s);
            check(r.rms >= (own > 0 ? own : lender_band) && r.rms >= own_r.rms,
                  std::string(row.name) + " " + std::string(ch::speed_basis_name(s)) +
                      ": the band is never narrower than the Own band");
            check(!r.decides || (s == S::Own && r.basis == B::Own),
                  std::string(row.name) + ": only an Own band decides");
            // Its floor: the Own band it widens (the generation's own, else the lender's, both
            // of two or more cards), at which a comparison resting on it is resolved, never
            // beyond; an Own band's floor is itself.
            check(r.floor == (own > 0 ? own : lender_band) &&
                      ch::resolution_width(r) == (r.decides ? r.rms : r.floor) &&
                      ch::resolution_width(r) <= r.rms,
                  std::string(row.name) + " " + std::string(ch::speed_basis_name(s)) +
                      ": resolved at its Own floor where it does not decide");
        }
        check(same(ch::resolution_of(card, ch::widest(S::Own, S::Transferred)),
                   ch::resolution_of(card, S::Transferred)) &&
                  same(ch::resolution_of(card, ch::widest(S::Transferred, S::LentGeneration)),
                       ch::resolution_of(card, S::LentGeneration)),
              std::string(row.name) + ": a mixed comparison takes its widest estimate's band");
    }
    for (const ch::TransferredError& t : ch::kTransferredDecodeError) {
        check(t.cards < 2, std::string(t.target) + ": a one-card carry (it does not decide)");
    }
}

// The resolution rule weighs a comparison at the band of the two estimates it compares (#618), not
// at the board's generation's alone: the wider of their bases (resolution_of), at its
// resolution_width: the full band where it decides, else its Own floor (a one-card or assumed band
// never widens the decision). Ada's bands: Own 2.26 % deciding; Transferred and lent the assumed
// carry (54.5 %), not deciding, floor 2.26 %.
void resolution_by_pair() {
    const auto make = [](double seconds, double kld, ch::SpeedBasis basis) {
        ch::Scored s;
        s.decision_seconds = seconds;
        s.accuracy.kld     = kld;
        s.own_context      = 1000;
        s.decision_basis   = basis;
        s.worth            = -std::log(seconds) - 0.3 * kld;
        return s;
    };
    using S  = ch::SpeedBasis;
    using RB = ch::ResolutionBasis;
    const ch::Resolution own{RB::Own, 0.0226, true, 0.0226};
    const ch::Resolution assumed{RB::Transferred, ch::kUnmeasuredCarryDecodeError, false, 0.0226};
    const ch::Resolution lent{RB::Unmeasured, ch::kUnmeasuredCarryDecodeError, false, 0.0226};
    const ch::Resolutions ada{own, assumed, lent};
    // A Transferred candidate inside the Own floor is weighed there, as by the one number: the
    // estimates are indistinguishable at least within it.
    const std::vector<ch::Scored> floor{make(1.0, 0.02, S::Own), make(1.02, 0.005, S::Transferred),
                                        make(1.01, 0.01, S::Own), make(1.0, 0.02, S::Transferred),
                                        make(1.10, 0.001, S::Transferred)};
    check(ch::resolve_within_resolution(floor, {0, 1, 2}, 0, 0.0226) == 1 &&
              ch::resolve_within_resolution(floor, {0, 1, 2}, 0, ada) == 1,
          "a Transferred candidate inside the Own floor: weighed there, as by the one number");
    check(ch::resolve_within_resolution(floor, {0, 2, 4}, 0, ada) == 2,
          "a Transferred candidate beyond the floor, inside its own 54 % band: never taken");
    check(ch::resolve_within_resolution(floor, {3, 1, 2}, 3, ada) == 1,
          "a Transferred best worth: its comparisons resolved at the floor too");
    // The review's case (lambda 0.3): A the best worth, C Transferred 0.5 % slower and the most
    // accurate, P Own 2 % slower and less accurate than C. The pick is C, never the dominated P.
    const std::vector<ch::Scored> review{
        make(1.00, 0.020, S::Own), make(1.005, 0.004, S::Transferred), make(1.02, 0.010, S::Own)};
    check(ch::resolve_within_resolution(review, {0, 1, 2}, 0, ada) == 1,
          "the faster and more accurate C, not the dominated P");
    // Comparisons of unequal widths (a hypothetical deciding Transferred band wider than the Own
    // one): P is inside its wider band, C (Own, faster and more accurate) outside the Own one; the
    // dominated P is never the pick.
    const ch::Resolutions wider{own, {RB::Transferred, 0.05, true, 0.0226}, lent};
    const std::vector<ch::Scored> unequal{make(1.00, 0.020, S::Own), make(1.03, 0.004, S::Own),
                                          make(1.04, 0.010, S::Transferred)};
    check(ch::resolve_within_resolution(unequal, {0, 1, 2}, 0, wider) == 1,
          "unequal widths: a dominated pick gives way to the candidate dominating it");
    // Among several dominators the lowest KLD, then the lowest index (choice.h, DECISIONS 75).
    const std::vector<ch::Scored> several{make(1.00, 0.020, S::Own), make(1.035, 0.006, S::Own),
                                          make(1.035, 0.004, S::Own), make(1.03, 0.004, S::Own),
                                          make(1.04, 0.010, S::Transferred)};
    check(ch::resolve_within_resolution(several, {0, 1, 2, 3, 4}, 0, wider) == 2,
          "several dominators: the lowest KLD, then the lowest index");
    // A deciding Transferred band: inside it the more accurate, outside it the best worth.
    const ch::Resolutions deciding{own, {RB::Transferred, 0.0786, true, 0.0226}, lent};
    const std::vector<ch::Scored> carry{make(1.0, 0.02, S::Own), make(1.05, 0.005, S::Transferred)};
    check(ch::resolve_within_resolution(carry, {0, 1}, 0, deciding) == 1,
          "a deciding Transferred band: inside it, the more accurate");
    check(ch::resolve_within_resolution(carry, {0, 1}, 0, ada) == 0,
          "a band that does not decide: beyond its floor the best worth stands");
    // The property: over random sets, the pick is never dominated on both axes (no slower and
    // strictly more accurate) by another eligible candidate, whatever the bases and bands.
    std::uint64_t seed = 12345;
    const auto next    = [&] {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(seed >> 11) / static_cast<double>(1ULL << 53);
    };
    int dominated = 0, moved = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<ch::Scored> set;
        const int n = 2 + static_cast<int>(next() * 6);
        for (int i = 0; i < n; ++i) {
            const double u = next();
            set.push_back(make(1 + 0.06 * next(), 0.02 * next(),
                               u < 0.5    ? S::Own
                               : u < 0.85 ? S::Transferred
                                          : S::LentGeneration));
        }
        std::vector<std::size_t> all(set.size());
        std::size_t best = 0;
        for (std::size_t i = 0; i < set.size(); ++i) {
            all[i] = i;
            if (set[i].worth > set[best].worth) { best = i; }
        }
        for (const ch::Resolutions& bands : {ada, wider, deciding}) {
            const std::size_t pick = ch::resolve_within_resolution(set, all, best, bands);
            moved += pick != best;
            for (const ch::Scored& other : set) {
                dominated += other.decision_seconds <= set[pick].decision_seconds &&
                             other.accuracy.kld + ch::kWorthTie < set[pick].accuracy.kld;
            }
        }
    }
    check(dominated == 0 && moved > 0, "the pick is never dominated on both axes");
}

// The resolution a decision uses is the one of the estimates it compares (#618), not the board's
// generation's alone. On an Ada board (its own error, a reference card): Q4_K_M at 32K compares
// Own estimates and Transferred ones (a candidate that reads a row of one Ada card: the exact
// route's prefill, the BF16-KV prompt kernel); a quantization of WeightClass::Other (an IQ3 mix:
// its routes not measured) makes every estimate Transferred. A comparison with a Transferred
// estimate rests on the assumed carry (no carry onto those rows was measured), which does not
// decide: it is resolved at the Own floor, 2.26 %, as an Own one is, never at 54.5 %. So on Ada
// the typed rule picks as the one number did; the choice reports the best worth's band and floor.
// The choice summary (the plan's and the replay's fields) and the replay report the band.
void resolution_of_compared(const json& fixture) {
    using S = ch::SpeedBasis;
    Bench b = nine_b(fixture, "RTX 4090");
    // The fixture's IQ3_XXS row is Q4_K_M's tables at other bits (a K-quant class); an IQ3 mix
    // of its own class (WeightClass::Other) as the last.
    ch::QuantOption other = b.quants[2];
    other.label           = "IQ3_XXS (Other)";
    other.weights         = ch::WeightClass::Other;
    b.quants.push_back(other);
    const std::size_t iq3         = b.quants.size() - 1;
    const ch::Allowed allowed     = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
    const ch::Resolution own_band = ch::resolution_of(b.card, S::Own);
    const ch::Resolution carried  = ch::resolution_of(b.card, S::Transferred);
    check(own_band.decides && own_band.rms == ch::worth_band(b.card) && !carried.decides &&
              carried.rms > own_band.rms && carried.floor == own_band.rms,
          "Ada: the Own band decides, the Transferred one is wider, does not, floors at Own");
    int own_pairs = 0, carried_pairs = 0, carried_best = 0;
    for (const std::uint32_t context : {4096U, 32768U, 131072U}) {
        ch::Task task;
        task.contexts = {context};
        for (std::size_t qi = 0; qi < b.quants.size(); ++qi) {
            const ch::Choice c = ch::choose(b.model, b.card, b.quants[qi], allowed, {}, task,
                                            oracle(b, 40e9, qi), ch::Gates{true, true});
            if (!c.chosen) { continue; }
            const ch::Scored& anchor = c.scored[c.anchor];
            std::cout << "  resolution of the compared: " << b.quants[qi].label << ' ' << context
                      << ": best worth " << ch::speed_basis_name(anchor.decision_basis) << ", band "
                      << ch::resolution_basis_name(c.resolution_basis) << ' ' << c.resolution
                      << (c.resolution_decides ? "" : " (not deciding)") << ", floor "
                      << c.resolution_floor << ", held";
            for (const ch::Scored& s : c.scored) {
                if (s.variant || !s.fits) { continue; }
                std::cout << ' ' << kv_cache_storage_name(s.candidate.kv) << '/'
                          << static_cast<int>(s.candidate.activations) << '='
                          << ch::speed_basis_name(s.decision_basis);
                // Every comparison weighed at the Own band: its own where both are Own, the
                // floor (the same 2.26 %) where either is Transferred.
                const bool pair_own = anchor.decision_basis == S::Own && s.decision_basis == S::Own;
                check(s.weighed_at == own_band.rms,
                      b.quants[qi].label + ": every comparison weighed at the Own band or floor");
                (pair_own ? own_pairs : carried_pairs) += 1;
            }
            std::cout << '\n';
            // The band reported: the best worth's, with its floor.
            const ch::Resolution want = ch::resolution_of(b.card, anchor.decision_basis);
            check(c.resolution_basis == want.basis && c.resolution == want.rms &&
                      c.resolution_decides == want.decides && c.resolution_floor == want.floor &&
                      c.by_worth,
                  b.quants[qi].label +
                      ": the band of the best worth's estimate, the worth decides");
            check(!c.resolved_within_resolution ||
                      std::log(chosen(c).decision_seconds / anchor.decision_seconds) <=
                          chosen(c).weighed_at + 1e-9,
                  b.quants[qi].label + ": a moved pick lies inside its comparison's width");
            // The one number (Gates::typed_resolution closed) picks the same on Ada.
            ch::Gates old{true, true};
            old.typed_resolution    = false;
            const ch::Choice before = ch::choose(b.model, b.card, b.quants[qi], allowed, {}, task,
                                                 oracle(b, 40e9, qi), old);
            check(before.chosen && *before.chosen == *c.chosen,
                  b.quants[qi].label + ": on Ada the typed rule picks as the one number");
            if (!c.resolution_decides) {
                carried_best += 1;
                check(c.why.find("at its own floor") != std::string::npos,
                      b.quants[qi].label + ": a Transferred best worth, and why");
            }
        }
    }
    check(own_pairs > 0 && carried_pairs > 0 && carried_best > 0,
          "Ada: Own and Transferred comparisons both met, and Transferred best worths");
    // An IQ3 mix: every estimate Transferred, the band not deciding, its floor the Own band.
    ch::Task task;
    task.contexts       = {32768};
    const ch::Choice q3 = ch::choose(b.model, b.card, b.quants[iq3], allowed, {}, task,
                                     oracle(b, 40e9, iq3), ch::Gates{true, true});
    bool all_carried    = q3.chosen.has_value();
    for (const ch::Scored& s : q3.scored) {
        if (!s.variant && s.fits) {
            all_carried = all_carried && s.decision_basis == S::Transferred;
        }
    }
    check(all_carried && q3.resolution_basis == ch::ResolutionBasis::Transferred &&
              q3.resolution == ch::kUnmeasuredCarryDecodeError && !q3.resolution_decides &&
              q3.resolution_floor == own_band.rms,
          "Ada IQ3 (Other): every estimate Transferred, the assumed band, not deciding, Own floor");
    // A pinned KV codec stands whatever the bases (--kv-dtype: Pins::kv).
    ch::Pins pinned;
    pinned.kv             = KvCacheStorage::BFloat16;
    const ch::Choice kept = ch::choose(b.model, b.card, b.quants[0], allowed, pinned, task,
                                       oracle(b, 40e9, 0), ch::Gates{true, true});
    check(kept.chosen && chosen(kept).candidate.kv == KvCacheStorage::BFloat16,
          "a pinned KV codec stands");
    // The summary (the plan's and the replay's fields) reports the band, whether it decided and
    // its floor, and the replay of the record answers it.
    for (const std::size_t qi : {std::size_t{0}, iq3}) {
        const ch::ChoiceRecord rec =
            ch::record_choice(b.model, b.card, b.quants[qi], allowed, {}, task, oracle(b, 40e9, qi),
                              ch::Gates{true, true});
        const auto summary = ch::choice_summary(rec.choice);
        check(summary.at("resolution_basis") ==
                      std::string(ch::resolution_basis_name(rec.choice.resolution_basis)) &&
                  summary.at("resolution_decides").get<bool>() == rec.choice.resolution_decides &&
                  summary.at("resolution_floor").get<double>() == rec.choice.resolution_floor &&
                  summary.at("scored").at(rec.choice.anchor).at("decision_basis") ==
                      std::string(ch::speed_basis_name(
                          rec.choice.scored[rec.choice.anchor].decision_basis)),
              b.quants[qi].label + ": the summary reports the band, whether it decided, its floor");
        if (qi == iq3) {
            check(summary.at("resolution_basis") == "transferred" &&
                      !summary.at("resolution_decides").get<bool>(),
                  "IQ3 (Other): the summary says transferred, not deciding");
        }
        const auto back = nlohmann::ordered_json::parse(rec.inputs.dump());
        check(ch::choice_summary(ch::replay_choice(back)) == summary,
              b.quants[qi].label + ": the replay answers the recorded band");
    }
}

// What typing the resolution by the estimate's basis changes (#618 step 3, a behaviour change made
// visible): for every fixture card and bench, the old (the reference generation's one number,
// Gates::typed_resolution closed) and the new resolution, the candidates each chooses, and whether
// the choice moves. The worth and its best are unchanged. A comparison resting on a band that does
// not decide is resolved at its Own floor, so on the fixture no choice moves (asserted), and no
// pick is dominated on both axes by a candidate it was weighed against; the table is the record.
void resolution_by_basis(const json& fixture) {
    // No reference, no estimate: the basis is None, not Own.
    const Bench none = nine_b(fixture, "RTX 3060");
    const ch::Choice undecided =
        ch::choose(none.model, none.card, none.quants[0],
                   ch::allowed_axes(none.model, KvCacheStorage::Int8Group64), {}, ch::Task{},
                   oracle(none, 20e9));
    check(undecided.resolution_basis == ch::ResolutionBasis::None && undecided.resolution == 0,
          "a board with no reference card: basis none");
    int total = 0, moved = 0, typed = 0, declined = 0, dominated = 0;
    std::vector<std::pair<std::string, Bench>> benches;
    for (const auto& [name, unused] : fixture.at("cards").items()) {
        (void)unused;
        benches.emplace_back(name + " Qwen3.5-9B", nine_b(fixture, name));
        Bench lent_nine =
            nine_b(fixture, name); // the same board of a compute capability with no row
        lent_nine.card.major = 13, lent_nine.card.minor = 0;
        benches.emplace_back(name + " (cc 13.0) Qwen3.5-9B", lent_nine);
        const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
        Bench moe{model_of(m35), card_of(fixture.at("cards").at(name)), {}};
        for (const auto& [qname, qrow] : m35.at("quants").items()) {
            moe.quants.push_back(quant_of(qname, qrow));
        }
        moe.model.drafters        = {{"dflash:7", "dflash", 2.5, 2, "measured"},
                                     {"mtp:4", "mtp", 1.9, 2, "measured"}};
        moe.model.default_drafter = 2;
        benches.emplace_back(name + " Qwen3.6-35B-A3B", moe);
    }
    const auto describe = [](const ch::Choice& c) {
        if (!c.chosen) { return std::string("none"); }
        const ch::Candidate& k = chosen(c).candidate;
        return std::string(kv_cache_storage_name(k.kv)) + "/ring" +
               std::string(kv_cache_storage_name(k.ring)) + "/state" +
               std::to_string(static_cast<int>(k.state)) + "/act" +
               std::to_string(static_cast<int>(k.activations)) + "/chunk" +
               std::to_string(k.chunk) + "/drafter" + std::to_string(k.drafter) + "/context" +
               std::to_string(c.context);
    };
    for (const auto& [label, b] : benches) {
        const ch::Allowed allowed = ch::allowed_axes(b.model, KvCacheStorage::Int8Group64);
        for (std::size_t qi = 0; qi < b.quants.size(); ++qi) {
            for (const std::uint32_t context : {4096U, 32768U, 131072U}) {
                ch::Task task;
                task.contexts = {context};
                ch::Gates old_gates{true, true};
                old_gates.typed_resolution = false;
                const ch::Choice before    = ch::choose(b.model, b.card, b.quants[qi], allowed, {},
                                                        task, oracle(b, 40e9, qi), old_gates);
                const ch::Choice after =
                    ch::choose(b.model, b.card, b.quants[qi], allowed, {}, task,
                               oracle(b, 40e9, qi), ch::Gates{true, true});
                if (!before.chosen || !after.chosen) { continue; }
                const bool differs = describe(before) != describe(after);
                const bool retyped = before.resolution != after.resolution ||
                                     before.resolution_basis != after.resolution_basis;
                ++total, moved += differs, typed += retyped;
                declined += before.by_worth && !after.by_worth;
                if (differs || retyped) {
                    std::cout << "  resolution by basis: " << label << ' ' << b.quants[qi].label
                              << ' ' << context << ": old " << before.resolution << ", new "
                              << after.resolution << " ("
                              << ch::resolution_basis_name(after.resolution_basis)
                              << (after.resolution_decides ? "" : ", not deciding") << "), "
                              << (differs ? "CHANGED " + describe(before) + " -> " + describe(after)
                                          : "choice unchanged: " + describe(after))
                              << '\n';
                }
                check(after.resolution_basis != ch::ResolutionBasis::Own ||
                          after.resolution_decides,
                      label + ": an Own band decides");
                check(before.by_worth == after.by_worth && before.anchor == after.anchor,
                      label + ": the worth and its best are as before (only the rule is typed)");
                // The pick is never dominated on both axes (no slower on the reference card and
                // strictly more accurate) by a candidate the rule weighed.
                for (const ch::Scored& s : after.scored) {
                    dominated += s.weighed_at > 0 &&
                                 s.decision_seconds <= chosen(after).decision_seconds &&
                                 s.accuracy.kld + ch::kWorthTie < chosen(after).accuracy.kld;
                }
            }
        }
    }
    std::cout << "resolution by basis: " << total << " choices, " << typed
              << " with a retyped resolution, " << declined << " no longer by worth, " << moved
              << " changed, " << dominated << " dominated\n";
    check(typed > 0 && declined == 0, "some resolutions are retyped, the worth decides as before");
    check(moved == 0, "resolved at the Own floor, the typing moves no fixture choice");
    check(dominated == 0, "no pick is dominated by a candidate it was weighed against");
}

// The fixtures' cards are the registry rows of the boards they were measured on (board_card, by
// PCI id): a card's facts in the JSON (which the fit scripts read too) may not drift from the row.
void fixture_cards_are_registry_rows(const json& doc, const std::string& what) {
    const std::map<std::string, std::uint32_t> pci = {
        {"RTX 4060 Ti", 0x2805}, {"RTX 4070 Ti", 0x2782}, {"RTX 4090", 0x2684},
        {"RTX 3060", 0x2504},    {"RTX 2080 Ti", 0x1E07}, {"RTX 5070 Ti", 0x2C05},
        {"RTX 5090", 0x2B85},    {"Quadro GV100", 0x1DBA}};
    const auto same = [](double a, double b) { return std::abs(a - b) <= 1e-9 * std::abs(b); };
    for (const auto& [name, card] : doc.at("cards").items()) {
        const auto id = pci.find(name);
        check(id != pci.end(), what + ": card " + name + " has a PCI id here");
        if (id == pci.end()) { continue; }
        const ch::CardFacts c = card_of(card), row = board_card(id->second);
        check(c.major == row.major && c.minor == row.minor && c.sms == row.sms &&
                  same(c.clock_hz, row.clock_hz) && same(c.dram_bytes_per_s, row.dram_bytes_per_s),
              what + ": card " + name + " is its registry row");
    }
}

// What a caller shows beside the choice (#618 step 11): the gate as a code with the measurement
// that would open it, the reference card it decided on, the bands its estimates carry, the answer
// scale where calibrated. None of it decides.
void reports_gates_and_bands(const json& fixture) {
    ch::Task task;
    task.contexts      = {32768};
    const Bench a      = nine_b(fixture, "RTX 4070 Ti");
    const auto allowed = ch::allowed_axes(a.model, KvCacheStorage::Int8Group64);
    const ch::Choice c =
        ch::choose(a.model, a.card, a.quants[0], allowed, {}, task, oracle(a, 9.6e9));
    check(c.chosen && c.by_worth && c.why_code == "by_worth" && c.gate_code == "open" &&
              c.gates_closed.empty(),
          "gates: Ada, Qwen 3.5: every gate open");
    check(c.reference && c.reference->generation == "ada" &&
              c.reference->cards == "RTX 4070 Ti, RTX 4090" && !c.reference->lent &&
              c.reference->decides && c.reference->card.sms == 94,
          "gates: decided on Ada's reference card, the mean of the 4070 Ti and the 4090");
    // The board's bands by basis (resolution_of on the board): an Own estimate carries the
    // generation's measured decode error, a lent one the assumed carry.
    const ch::Resolution own = ch::resolution_for(c.board_resolutions, ch::SpeedBasis::Own);
    check(own.basis == ch::ResolutionBasis::Own && own.rms == 0.0226 && own.decides,
          "bands: an Own estimate on Ada, the three cards' 2.26 %");
    check(ch::resolution_for(c.board_resolutions, ch::SpeedBasis::LentGeneration).basis ==
              ch::ResolutionBasis::Unmeasured,
          "bands: a lent estimate, the assumed carry");
    // The prefill band of each estimate: at the short request's 2K prompt, below the measured
    // range, no error is stated; the generation's row beside it.
    const ch::PrefillBand& cb = chosen(c).prefill_band;
    check(!cb.holds && cb.why_code == "short_prompt" && cb.cards == 3 && cb.stated_rms > 0 &&
              !cb.moe && cb.prompt == task.prompt,
          "bands: a 2K prompt claims no prefill error, Ada's stated row beside it");
    ch::Candidate fast;
    const auto band_at = [&](const ch::ModelFacts& m, const ch::CardFacts& card,
                             const ch::Candidate& cand, ch::SpeedBasis basis,
                             std::uint32_t prompt) {
        return ch::prefill_band_of(m, card, a.quants[0], cand, basis, prompt);
    };
    const ch::PrefillBand at3k =
        band_at(a.model, a.card, fast, ch::SpeedBasis::Own, ch::kPrefillErrorFromPrompt);
    check(at3k.holds && at3k.why_code.empty() &&
              at3k.stated_rms == ch::kPrefillPredictionError[0].rms,
          "bands: inside the measured range, the generation's row holds");
    check(band_at(a.model, a.card, fast, ch::SpeedBasis::Own, 41221).why_code == "long_prompt",
          "bands: beyond the measured range, none");
    check(band_at(a.model, a.card, fast, ch::SpeedBasis::Transferred, 7406).why_code ==
              "transferred",
          "bands: a Transferred estimate, none");
    ch::Candidate exact;
    exact.activations = ActivationMode::Exact;
    check(band_at(a.model, a.card, exact, ch::SpeedBasis::Own, 7406).why_code == "not_fast_route",
          "bands: the exact route was not measured, none");
    check(band_at(a.model, board_card(0x1DBA), fast, ch::SpeedBasis::Own, 7406).why_code ==
                  "generation_unchecked" &&
              band_at(a.model, card_of(fixture.at("cards").at("RTX 2080 Ti")), fast,
                      ch::SpeedBasis::Own, 7406)
                      .why_code == "generation_unchecked",
          "bands: Volta and Turing (one card each) state none");
    ch::ModelFacts moe       = a.model;
    moe.experts              = 256;
    moe.experts_per_token    = 8;
    moe.routed_expert_params = moe.params * 0.8;
    const ch::PrefillBand m  = band_at(moe, a.card, fast, ch::SpeedBasis::Own, 7406);
    check(m.moe && !m.holds && m.why_code == "moe_unchecked",
          "bands: an MoE's prefill states none");
    // Turing: no reference card; what opens it.
    const Bench t = nine_b(fixture, "RTX 2080 Ti");
    const ch::Choice ct =
        ch::choose(t.model, t.card, t.quants[0], allowed, {}, task, oracle(t, 9.0e9));
    check(ct.chosen && !ct.by_worth && ct.why_code == "defaults" &&
              ct.gate_code == "no_reference_card" && !ct.reference && !ct.gates_closed.empty() &&
              ct.gates_closed[0].opens_with == "reference_factors" &&
              ct.gates_closed[0].generation == "turing" && ct.gates_closed[0].cards == 0,
          "gates: Turing, no reference card, opened by its KV and exact factors on two cards");
    check(ct.gates_closed.back().axis == "quant" &&
              ct.gates_closed.back().code == "quant_switches_closed" &&
              ct.gates_closed.back().opens_with.empty(),
          "gates: Turing's quantizations wait on the switches' gate");
    // Another family on Ada: neither its switches' terms nor its curve were measured.
    ch::ModelFacts gemma = a.model;
    gemma.family         = "gemma4_text";
    const auto g         = ch::closed_gates(gemma, a.card);
    check(g.size() == 2 && g[0].code == "family_unmeasured" &&
              g[0].opens_with == "family_kld_terms" && g[0].family == "gemma4_text" &&
              g[1].axis == "quant" && g[1].code == "quant_curve_unmeasured" &&
              g[1].opens_with == "quant_curve",
          "gates: Gemma 4 on Ada, its accuracy terms and its curve");
    // A family with switch terms and no curve: the switches decide, the quantizations by the rule.
    ch::ModelFacts qmoe = moe;
    qmoe.family         = "qwen3_5_moe_text";
    const auto q        = ch::closed_gates(qmoe, a.card);
    check(q.size() == 1 && q[0].axis == "quant" && q[0].code == "quant_curve_unmeasured" &&
              ch::decides_switches(qmoe, a.card) && !ch::weighs_quantizations(qmoe, a.card),
          "gates: Qwen 3.5 MoE on Ada, only the curve closed");
    // A test's closed gate, and a compute capability with no generation row.
    const auto closed = ch::closed_gates(a.model, a.card, {false, true});
    check(closed.size() == 2 && closed[0].code == "accuracy_gate_closed" &&
              closed[0].opens_with.empty() && closed[1].code == "quant_switches_closed",
          "gates: a closed accuracy gate has nothing to measure, and closes the trade");
    ch::CardFacts lent = card_of(fixture.at("cards").at("RTX 5090"));
    lent.major         = 13;
    lent.minor         = 0;
    const auto lg      = ch::closed_gates(a.model, lent);
    const auto lr      = ch::reference_of(lent);
    check(!lg.empty() && lg[0].code == "lent_generation" && lg[0].opens_with == "generation_row" &&
              lr && lr->lent && lr->cards.empty() && !lr->decides,
          "gates: a lent generation, opened by a generation row");
    // The answer scale: only the calibrated family.
    const ch::AnswerScale* scale = ch::answer_scale_of("qwen3_5_text");
    check(scale && scale->points_per_kld == 40.0 && scale->benchmark == "mmlu_pro" &&
              scale->model == "Qwen3.5-9B" && !ch::answer_scale_of("gemma4_text") &&
              !ch::answer_scale_of("qwen3_5_moe_text"),
          "answers: Qwen 3.5 dense only");
    // Only the quantization's own term against Q8_0, on the calibrated model's measured files.
    ch::Accuracy measured;
    measured.kld   = 0.03;
    measured.terms = {{ch::Axis::Quant, "Q4_K_M", 0.02, true, false, "", ch::Evidence::Measured},
                      {ch::Axis::Kv, "int8", 0.01, true, false, "", ch::Evidence::Measured}};
    std::string_view why = "x";
    const auto pts       = ch::answer_points(measured, scale, &why);
    check(pts && std::abs(*pts - 0.8) < 1e-12 && why.empty(),
          "answers: 40 points a KLD of the quantization's term alone, not the switches'");
    ch::Accuracy proxy      = measured;
    proxy.terms[0].evidence = ch::Evidence::Proxy; // the same label on another model's files
    check(!ch::answer_points(proxy, scale, &why) && why == "model_not_calibrated",
          "answers: another model of the family (the 27B) is not calibrated");
    // Q8_0 is the curve's zero on any model (Evidence::Reference): that is not the 9B's identity.
    // A 27B of the family at Q8_0 gets no points; the 9B's own measured Q8_0 file gets 0.
    {
        ch::Candidate c;
        c.kv            = KvCacheStorage::BFloat16;
        const json& m27 = fixture.at("models").at("Qwen3.8-27B");
        const Bench big{model_of(m27), a.card, {}};
        ch::QuantOption q8        = quant_of("UD-Q4_K_XL", m27.at("quants").at("UD-Q4_K_XL"));
        q8.label                  = "Q8_0";
        q8.bits_per_weight        = 8.5;
        q8.files                  = "27b-q8_0-files"; // not the 9B's measured file
        const ch::Accuracy big_q8 = ch::accuracy_of(big.model, q8, c, big.card);
        const ch::Term* q8_term   = nullptr;
        for (const ch::Term& t : big_q8.terms) {
            if (t.axis == ch::Axis::Quant) { q8_term = &t; }
        }
        why = "x";
        check(big.model.family == "qwen3_5_text" && q8_term && q8_term->kld == 0 &&
                  q8_term->evidence == ch::Evidence::Reference &&
                  !ch::answer_points(big_q8, ch::answer_scale_of(big.model.family), &why) &&
                  why == "model_not_calibrated",
              "answers: a 27B's Q8_0 (the curve's zero, not the 9B's file) is not calibrated");
        q8.files = "";
        why      = "x";
        check(!ch::answer_points(ch::accuracy_of(big.model, q8, c, big.card), scale, &why) &&
                  why == "model_not_calibrated",
              "answers: a Q8_0 of unknown files is not calibrated");
        ch::QuantOption own = q8;
        own.files           = std::string(ch::kQuantKld[0].file); // the 9B's measured Q8_0
        own.bits_per_weight = ch::kQuantKld[0].bits_per_weight;
        why                 = "x";
        const auto zero = ch::answer_points(ch::accuracy_of(a.model, own, c, a.card), scale, &why);
        check(ch::kQuantKld[0].label == "Q8_0" && zero && *zero == 0 && why.empty(),
              "answers: the 9B's own Q8_0 file loses 0 points");
    }
    check(!ch::answer_points(measured, nullptr, &why) && why == "family_not_calibrated",
          "answers: another family is not calibrated");
    // The gate code agrees with the worth on every fixture board, and with decides_switches.
    for (const auto& [name, unused] : fixture.at("cards").items()) {
        const Bench b = nine_b(fixture, name);
        const ch::Choice x =
            ch::choose(b.model, b.card, b.quants[0], allowed, {}, task, oracle(b, 12e9));
        check(x.chosen && (x.gate_code == "open") == x.by_worth &&
                  x.by_worth == ch::decides_switches(b.model, b.card) &&
                  (x.reference.has_value() == ch::reference_of(b.card).has_value()),
              "gates: " + name + ": the gate code is the worth's");
    }
}

int main() {
    for (const char* name : {"speed_measured", "decode_measured", "prefill_measured"}) {
        std::ifstream cards(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/" + name +
                            ".json");
        fixture_cards_are_registry_rows(json::parse(cards), name);
    }
    std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/speed_measured.json");
    const json fixture = json::parse(in);
    reproduces_the_measurements(fixture);
    {
        std::ifstream campaign(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/decode_measured.json");
        const json decode_rows = json::parse(campaign);
        decodes_from_facts(decode_rows);
        transferred_resolution(decode_rows);
        std::ifstream prefills(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/prefill_measured.json");
        const json prefill_rows = json::parse(prefills);
        prefills_by_format(prefill_rows);
        prices_gemma_attention_tiles(prefill_rows);
        prefills_pay_tail_calls(prefill_rows);
        std::ifstream census(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/decode_census.json");
        const json census_rows = json::parse(census);
        counts_the_kernels(census_rows);
        census_labels_keep_their_class(census_rows);
    }
    prefills_moe_by_tiles(fixture);
    prices_attention(fixture);
    chooses(fixture);
    decides_by_generation(fixture);
    lent_generation_does_not_decide(fixture);
    steps_before_the_floor(fixture);
    weighs_the_quantizations(fixture);
    chooses_the_chunk(fixture);
    prices_the_request_mix(fixture);
    coarse_quant_keeps_kv(fixture);
    explains_the_state(fixture);
    contradictory_pins(fixture);
    activations_by_generation(fixture);
    moe_family(fixture);
    moe_step_reads_the_route(fixture);
    moe_prefill_follows_the_activations(fixture);
    prices_moe_prefill_tiles();
    prices_dense_prefill_tiles();
    prices_prefill_small_kernels();
    prices_attention_tiles();
    prices_r8_issue();
    weighs_the_drafters(fixture);
    drafter_order_does_not_decide(fixture);
    load_bounds_the_gain(fixture);
    joint_price_scales_the_gain(fixture);
    joint_price_keeps_the_gain_monotone();
    joint_price_replays(fixture);
    axis_order_does_not_decide(fixture);
    board_facts_do_not_decide(fixture);
    choices_replay(fixture);
    resolution_labels(fixture);
    resolution_rule();
    choice_properties(fixture);
    resolution_by_pair();
    resolution_of_compared(fixture);
    resolution_by_basis(fixture);
    reports_gates_and_bands(fixture);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "choice: ok\n";
    return 0;
}
