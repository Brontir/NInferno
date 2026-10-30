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

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifndef NINFERNO_SOURCE_DIR
#    error "NINFERNO_SOURCE_DIR must name the source tree"
#endif

namespace {

namespace ch = ninferno::product::choice;
using ninferno::ActivationMode;
using ninferno::KvCacheStorage;
using ninferno::RecurrentStateStorage;
using nlohmann::json;

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
    out.weights         = ch::weight_class_of({}, label);
    // An MoE's routed banks as its GGUF's tensor table stores them (ffn_*_exps).
    out.routed_expert_bytes = q.value("routed_expert_bytes", 0.0);
    return out;
}

// The 2026-10-23 points: an MoE's decode from facts at one stream and on the 2026-10-30 step rows
// (DECISIONS 65); a dense model's one-stream decode from facts (its several-stream rows are wave
// aggregates, not a step's rate: decodes_from_facts measures those); prefill (the 22K points, at their prompt,
// attention over it) within the refit's (choice-notes/lab/scripts/fit_prefill.py).
void reproduces_the_measurements(const json& fixture) {
    double decode_sq = 0, prefill_sq = 0, decode_worst = 0, prefill_worst = 0, moe_sq = 0, moe_worst = 0;
    int decodes = 0, prefills = 0, moes = 0;
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
        if (mf.moe() && step) {
            moe_sq += d * d;
            moe_worst = std::max(moe_worst, std::abs(d));
            ++moes;
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
        sq += d * d;
        worst = std::max(worst, std::abs(d));
        ++n;
        const std::string gen = std::string(ch::generation_of(cf));
        auto& g = by_generation[gen];
        g.first += std::log1p(d) * std::log1p(d);
        ++g.second;
        if (gen == "ada" || gen == "gb20x") { own_worst = std::max(own_worst, std::abs(std::log1p(d))); }
    }
    const double rms = std::sqrt(sq / n);
    std::cout << "decode from facts, " << n << " points: rms " << rms * 100 << " %, worst " << worst * 100 << " %\n";
    check(n == 100 && rms < 0.04, "decode from facts within the model's spread");
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
}

// The kernels a step launches are the graph's (kDecodeLaunches against the nsys census,
// tests/fixtures/choice/decode_census.json): every card's weight-reading kernels, and its small ones at
// one row.
void counts_the_kernels(const json& census) {
    int n = 0;
    for (const json& row : census.at("rows")) {
        const std::string model = row.at("model").get<std::string>();
        const std::string family = model.rfind("gemma", 0) == 0 ? "gemma4_text" : "qwen3_5_text";
        const ch::WeightClass w  = ch::weight_class_of({}, row.at("quant").get<std::string>());
        const auto* l = std::find_if(std::begin(ch::kDecodeLaunches), std::end(ch::kDecodeLaunches),
                                     [&](const ch::DecodeLaunches& r) { return !r.moe && r.family == family && r.weights == w; });
        if (l == std::end(ch::kDecodeLaunches)) {
            check(false, "a census row has its family's launches");
            continue;
        }
        check(row.at("gemv_kernels").get<double>() == l->gemv, "the weight-reading kernels a step: the census's");
        if (row.at("streams").get<int>() == 1) {
            check(row.at("small_kernels").get<double>() == l->small[0], "the small kernels at one row: the census's");
        }
        ++n;
    }
    check(n >= 68, "the census rows");
    // The weight class from formats, else the label.
    check(ch::weight_class_of({"ggml_q4_k", "ggml_q6_k", "q8_g32_fp16", "bf16"}, "") == ch::WeightClass::KQuant,
          "a K-quant mix with Q8_0 and 16-bit tensors");
    check(ch::weight_class_of({"q8_g32_fp16", "bf16"}, "") == ch::WeightClass::Q8, "Q8_0 alone");
    check(ch::weight_class_of({"ggml_iq4_nl", "q8_g32_fp16", "ggml_q5_k"}, "Q4_K_M") == ch::WeightClass::IQ4,
          "IQ4_NL: the formats decide over the label");
    for (const char* q : {"IQ4_XS", "IQ4_NL", "UD-IQ4_XS"}) { check(ch::weight_class_of({}, q) == ch::WeightClass::IQ4, "an IQ4 label"); }
    for (const char* k : {"Q4_K_M", "UD-Q4_K_XL", "Q5_K_M", "UD-Q5_K_XL", "Q6_K", "q6_k"}) {
        check(ch::weight_class_of({}, k) == ch::WeightClass::KQuant, "a K-quant label");
    }
    for (const char* q : {"Q8_0", "Q8", "q8_0", "UD-Q8_K_XL"}) { check(ch::weight_class_of({}, q) == ch::WeightClass::Q8, "a Q8 label"); }
    for (const char* o : {"UD-IQ3_XXS", "Q4_0", "Q3_K_M", "MXFP4", "NVFP4", "BF16", "Q8_KV", "Q8_K_R8", "Q8_0_R8", "Q4_K_R4", "IQ4_XS_R8", "Q4_1", "Q4_0"}) {
        check(ch::weight_class_of({}, o) == ch::WeightClass::Other, "an unmeasured label");
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
    task.prompt = prompt;
    task.answer = 1;
    const double r0 = ch::request_seconds(none, cf, q, c, task);
    const double r2 = ch::request_seconds(two, cf, q, c, task);
    check(std::abs((r2 - r0) - 2 * step) < 1e-9 * r2, "and to a request's time");
    ch::Candidate drafted = c;
    drafted.drafter       = 1;
    const double d0 = ch::request_seconds(none, cf, q, drafted, task);
    const double d2 = ch::request_seconds(two, cf, q, drafted, task);
    check(std::abs((d2 - d0) - 2 * step) < 1e-9 * d2, "a drafted candidate's tail calls are plain decode steps");
    std::cout << "tail calls: a decode step " << step * 1e3 << " ms, 2K prefill " << prompt / t0 << " -> "
              << prompt / t2 << " tok/s\n";
}

// Prefill by format (choice_constants.h kPrefillTilesEta): every 3K-16K-token prompt of the
// 2026-10-28 campaign (tests/fixtures/choice/prefill_measured.json), Q8_0 on the int8 tiles, Q4_K_M on
// MMQ's int8 rows; Q8_0 had taken the MMQ rows (+35-61 % at 14K before). The Q8 rows are the mean of
// these same points (Ada, Blackwell: each card within 7 % of its generation's mean; Ampere and Turing
// one card each, so in sample).
void prefills_by_format(const json& fixture) {
    double sq[2] = {0, 0}, worst[2] = {0, 0};
    int n[2] = {0, 0};
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
        const int i    = qo.weights == ch::WeightClass::Q8 ? 1 : 0;
        sq[i] += d * d;
        worst[i] = std::max(worst[i], std::abs(d));
        ++n[i];
    }
    const double mmq = std::sqrt(sq[0] / n[0]), tiles = std::sqrt(sq[1] / n[1]);
    std::cout << "prefill by format: Q4_K_M " << n[0] << " points rms " << mmq * 100 << " % (worst "
              << worst[0] * 100 << " %), Q8_0 " << n[1] << " rms " << tiles * 100 << " % (worst " << worst[1] * 100
              << " %)\n";
    check(n[1] == 42 && tiles < 0.08 && worst[1] < 0.15, "Q8_0 prefill on the int8 tiles within the model's spread");
    check(n[0] == 42 && mmq < 0.10 && worst[0] < 0.28, "Q4_K_M prefill on MMQ's rows as before");

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
    check(ada_q8.prefill < ada_kq.prefill * 0.8, "Ada: a Q8 prefill takes the tiles row (0.212 against 0.306)");
    check(!ada_q8.transferred, "Ada: the tiles row rests on three cards");
    // One card's rows price only their own boards: marked transferred.
    check(at("RTX 2080 Ti", q8, mf, 1).transferred && at("RTX 2080 Ti", kq, mf, 1).transferred &&
              at("RTX 3060", q8, mf, 1).transferred,
          "Turing / Ampere prefill rows of one card are transferred");
    // A generation without a tiles row keeps its int8 row, transferred (Volta: its own).
    ch::CardFacts volta = card_of(fixture.at("cards").at("RTX 2080 Ti"));
    volta.major = 7;
    volta.minor = 0;
    const ch::Speed v8 = ch::speed_of(mf, volta, q8, c, 1, 4096, 2048);
    const ch::Speed vk = ch::speed_of(mf, volta, kq, c, 1, 4096, 2048);
    check(v8.transferred && std::abs(v8.prefill / vk.prefill - 1) < 1e-9,
          "no tiles row: the int8 row, transferred");
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
                  chosen(c).candidate.state == RecurrentStateStorage::FP32,
              "pins kept; exact couples the FP32 state");
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
    const Bench t = nine_b(fixture, "RTX 2080 Ti");
    check(!ch::reference_of(t.card), "Turing has no reference card");
    ch::CardFacts volta{7, 0, 80, 1.53e9, 900e9, false};
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
    t1280.prompt = 1280;
    t1280.answer = 1;
    const double whole = ch::speed_of(linears, bw, q, fast, 1, 1280, 1024).prefill;
    const double rest  = ch::speed_of(linears, bw, q, fast, 1, 1280, 256).prefill;
    const double dec   = ch::speed_of(linears, bw, q, fast, 1, 1280.5, 1280).decode_per_stream;
    check(std::abs(ch::request_seconds(linears, bw, q, fast, t1280) - (1024 / whole + 256 / rest + 1 / dec)) < 1e-9,
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
    check(std::abs(quant_term(q).kld - 0.01925) < 1e-9 && quant_term(q).measured, "Q4_K_M: its measured point");
    q.bits_per_weight = 4.9; // another model's Q4_K_M: interpolated, not the 9B's point
    check(!quant_term(q).measured && quant_term(q).kld > 0.01925, "a label at other bits: interpolated");
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
// the accuracy gate does not hold it); on Blackwell (no reference card) the default.
void chooses_the_chunk(const json& fixture) {
    const json& m35 = fixture.at("models").at("Qwen3.6-35B-A3B");
    Bench b{model_of(m35), card_of(fixture.at("cards").at("RTX 4090")), {quant_of("UD-IQ3_XXS", m35.at("quants").at("UD-IQ3_XXS"))}};
    b.model.recurrent = true;
    ch::Task task;
    const auto own = [&b](const ch::Candidate& c) -> std::uint32_t {
        // The workspace grows with the chunk: a wider one holds a little less context.
        const double room = 20e9 - b.quants[0].weight_bytes - 4e5 * c.chunk;
        return room > 0 ? static_cast<std::uint32_t>(room / b.model.kv_bytes_per_token(c.kv)) : 0;
    };
    const ch::Choice ada = ch::choose(b.model, b.card, b.quants[0], ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task, own);
    check(ada.chosen && chosen(ada).candidate.chunk == 2048 && ada.chunk_by_speed,
          "Ada, an MoE: chunk 2048 for a 2048-token prompt");
    int chunks = 0;
    for (const ch::Alternative& a : ch::explain(ada)) {
        if (a.axis == ch::Axis::Chunk && a.value == "4096") {
            check(a.why_code == "tie_context", "chunk 4096: as fast for this prompt, less context");
            ++chunks;
        }
        if (a.axis == ch::Axis::Chunk && a.value == "1024") {
            check(a.why_code == "lower_worth", "chunk 1024: slower");
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
    check(bw.chosen && chosen(bw).candidate.chunk == 1024, "Blackwell: the default chunk");
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

// Contradictory pins say so instead of "nothing fits".
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

// The drafter is one more axis (choice/SPECULATION.md S3): its memory in the plan, its gain in the
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

int main() {
    std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/speed_measured.json");
    const json fixture = json::parse(in);
    reproduces_the_measurements(fixture);
    {
        std::ifstream campaign(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/decode_measured.json");
        decodes_from_facts(json::parse(campaign));
        std::ifstream prefills(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/prefill_measured.json");
        const json prefill_rows = json::parse(prefills);
        prefills_by_format(prefill_rows);
        prefills_pay_tail_calls(prefill_rows);
        std::ifstream census(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/decode_census.json");
        counts_the_kernels(json::parse(census));
    }
    prices_attention(fixture);
    chooses(fixture);
    decides_by_generation(fixture);
    steps_before_the_floor(fixture);
    weighs_the_quantizations(fixture);
    chooses_the_chunk(fixture);
    coarse_quant_keeps_kv(fixture);
    explains_the_state(fixture);
    contradictory_pins(fixture);
    activations_by_generation(fixture);
    moe_family(fixture);
    weighs_the_drafters(fixture);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "choice: ok\n";
    return 0;
}
