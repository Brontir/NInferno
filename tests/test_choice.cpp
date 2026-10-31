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
    out.weights         = ch::weight_class_of({}, label);
    // An MoE's routed banks as its GGUF's tensor table stores them (ffn_*_exps).
    out.routed_expert_bytes = q.value("routed_expert_bytes", 0.0);
    return out;
}

// The 2026-10-23 points: an MoE's decode from facts at one stream and on the 2026-10-30 step rows
// (DECISIONS 65); a dense model's one-stream decode from facts (its several-stream rows are wave
// aggregates, not a step's rate: decodes_from_facts measures those) and its steps at C rows (mbatch.sh); prefill (the 22K points, at their prompt,
// attention over it) within the refit's (choice-notes/lab/scripts/fit_prefill.py).
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
    check(bigs == 26 && std::sqrt(big_sq / bigs) < 0.06 && big_worst < 0.14,
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
    int volta_open = 0;
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
        if (gen == "volta") { // the Quadro GV100's points (2026-10-31): open, no census of its step yet
            ++volta_open;
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
    check(n == 100 && volta_open == 12 && rms < 0.04, "decode from facts within the model's spread");
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
        // What the IQ4 law leaves to the share: an IQ4_NL label, an iq4_nl linear, a K-quant label in
        // the IQ4 class (a Q4_K_M file with an iq4 tensor).
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
        check(ch::prefill_linears_seconds(named, ada, xs, 1024) > 0 && ch::prefill_linears_seconds(named, ada, xs, 16) == 0 &&
                  ch::prefill_linears_seconds(unnamed, ada, km, 1024) == 0 &&
                  ch::prefill_linears_seconds(with_nl, ada, xs, 1024) == 0,
              "IQ4_NL, an iq4_nl linear and an unnamed K label in the IQ4 class keep the share");
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
// An MoE's prefill linears by their tiles (DECISIONS 66): Qwen3.6-35B-A3B UD-Q4_K_M on the RTX 4090,
// one 1819-token chunk; the census (nsys, no PDL) put the experts at 50.5 ms, the int8 calls beside
// them at 30.0 and the routing's gather and scatter at 5.9: 86.4 ms.
void prices_moe_prefill_tiles() {
    ch::ModelFacts m;
    m.family               = "qwen3_5_moe_text";
    m.experts              = 256;
    m.experts_per_token    = 8;
    m.routed_expert_params = 34e9;
    for (int i = 0; i < 10; ++i) {
        m.linears.push_back({9216, 2048, 1, false});
        m.linears.push_back({2048, 4096, 1, false});
    }
    m.linears.push_back({12288, 2048, 30, false});
    m.linears.push_back({2048, 4096, 30, false});
    m.linears.push_back({1024, 2048, 40, false});
    m.linears.push_back({2048, 512, 40, false});
    m.linears.push_back({1024, 2048, 320, true});
    m.linears.push_back({2048, 512, 320, true});
    ch::CardFacts rtx4090{.major = 8, .minor = 9, .sms = 128, .clock_hz = 2.52e9, .dram_bytes_per_s = 1008e9};
    ch::QuantOption q;
    q.expert_banks = ch::ExpertBanks::Int8;
    q.rest_weights = ch::WeightClass::Q8; // UD-Q4_K_M's linears beside the experts: Q8_0
    const double t = ch::prefill_linears_seconds(m, rtx4090, q, 1819);
    std::cout << "MoE prefill tiles, Qwen3.6 UD-Q4_K_M RTX 4090 1819 tokens: " << t * 1e3 << " ms against 86.4\n";
    check(std::abs(t / 0.0864 - 1) < 0.12, "an MoE's prefill linears by their tiles within 12 % of the census");
    ch::CardFacts v100{.major = 7, .minor = 0, .sms = 84, .clock_hz = 1.62e9, .dram_bytes_per_s = 870e9};
    check(ch::prefill_linears_seconds(m, v100, q, 1819) == 0, "no int8 tile row on Volta: unknown");
    q.expert_banks = ch::ExpertBanks::Unknown;
    check(ch::prefill_linears_seconds(m, rtx4090, q, 1819) == 0, "unknown banks: unknown");
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
        {{.major = 8, .minor = 6, .sms = 28, .clock_hz = 1.777e9, .dram_bytes_per_s = 360e9}, 767.4},
        {{.major = 8, .minor = 9, .sms = 34, .clock_hz = 2.535e9, .dram_bytes_per_s = 288e9}, 440.4},
        {{.major = 8, .minor = 9, .sms = 60, .clock_hz = 2.61e9, .dram_bytes_per_s = 504e9}, 245.9},
        {{.major = 8, .minor = 9, .sms = 128, .clock_hz = 2.52e9, .dram_bytes_per_s = 1008e9}, 127.4},
        {{.major = 12, .minor = 0, .sms = 70, .clock_hz = 2.452e9, .dram_bytes_per_s = 896e9}, 212.0},
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
        {rows[0].card, 39.3}, {rows[1].card, 28.4}, {rows[2].card, 26.1}, {rows[3].card, 27.5}, {rows[4].card, 25.5},
        {{.major = 12, .minor = 0, .sms = 170, .clock_hz = 2.407e9, .dram_bytes_per_s = 1792e9}, 27.1},
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
        const ch::CardFacts ti{.major = 8, .minor = 9, .sms = 34, .clock_hz = 2.535e9, .dram_bytes_per_s = 288e15};
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
            const ch::CardFacts gv100{.major = 7, .minor = 0, .sms = 80, .clock_hz = 1.627e9, .dram_bytes_per_s = 870e9};
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
        const ch::CardFacts r5070{.major = 12, .minor = 0, .sms = 70, .clock_hz = 2.452e9, .dram_bytes_per_s = 896e9};
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
        {{.major = 8, .minor = 6, .sms = 28, .clock_hz = 1.777e9, .dram_bytes_per_s = 360e9}, 63.8},
        {{.major = 8, .minor = 9, .sms = 34, .clock_hz = 2.535e9, .dram_bytes_per_s = 288e9}, 35.3},
        {{.major = 8, .minor = 9, .sms = 60, .clock_hz = 2.61e9, .dram_bytes_per_s = 504e9}, 21.6},
        {{.major = 8, .minor = 9, .sms = 128, .clock_hz = 2.52e9, .dram_bytes_per_s = 1008e9}, 11.5},
        {{.major = 12, .minor = 0, .sms = 70, .clock_hz = 2.452e9, .dram_bytes_per_s = 896e9}, 14.1},
        {{.major = 12, .minor = 0, .sms = 170, .clock_hz = 2.407e9, .dram_bytes_per_s = 1792e9}, 7.9},
        {{.major = 7, .minor = 5, .sms = 68, .clock_hz = 1.545e9, .dram_bytes_per_s = 616e9}, 76.9},
    };
    double worst = 0;
    for (const Census& r : rows) {
        const double t = ch::prefill_small_seconds(m, r.card, 1819);
        worst          = std::max(worst, std::abs(t * 1e3 / r.ms - 1));
        std::cout << "prefill small kernels, " << r.card.sms << " SMs: " << t * 1e3 << " ms against " << r.ms << "\n";
    }
    check(worst < 0.25, "a prefill's small kernels by the scan's clocks and the rest's bytes within a quarter");
    const ch::CardFacts v100{.major = 7, .minor = 0, .sms = 80, .clock_hz = 1.53e9, .dram_bytes_per_s = 900e9};
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
        {{.major = 12, .minor = 0, .sms = 170, .clock_hz = 2.407e9, .dram_bytes_per_s = 1792e9},
         {89.3, 154.7, 221.4, 285.5, 350.9, 414.2, 480.3}},
        {{.major = 8, .minor = 9, .sms = 34, .clock_hz = 2.535e9, .dram_bytes_per_s = 288e9},
         {272.4, 580.9, 890.2, 1231.7, 1528.9, 1821.7, 2152.4}},
        {{.major = 8, .minor = 6, .sms = 28, .clock_hz = 1.777e9, .dram_bytes_per_s = 360e9},
         {566.2, 1110.8, 1646.7, 2190.3, 2734.7, 3275.4, 3833.7}},
        {{.major = 12, .minor = 0, .sms = 70, .clock_hz = 2.452e9, .dram_bytes_per_s = 896e9},
         {109.5, 236.5, 362.2, 486.4, 611.5, 735.3, 860.2}},
        // The RTX 4070 Ti: 60 SMs, its first two 64-row slots as half-height CTAs.
        {{.major = 8, .minor = 9, .sms = 60, .clock_hz = 2.61e9, .dram_bytes_per_s = 504e9},
         {174.1, 368.9, 560.4, 751.4, 946.8, 1133.5, 1354.6}},
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
        {{.major = 8, .minor = 9, .sms = 34, .clock_hz = 2.535e9, .dram_bytes_per_s = 288e9},
         {300.3, 610.7, 620.2, 609.8, 611.2, 610.0},
         {812.9, 2403.9, 4062.3, 5596.3, 7259.1, 8854.9}},
        {{.major = 12, .minor = 0, .sms = 170, .clock_hz = 2.407e9, .dram_bytes_per_s = 1792e9},
         {83.0, 141.7, 142.6, 143.4, 143.3, 143.9},
         {154.0, 492.4, 735.8, 1052.7, 1282.6, 1605.3}},
        // The RTX 2080 Ti (Turing's row, read on it).
        {{.major = 7, .minor = 5, .sms = 68, .clock_hz = 1.545e9, .dram_bytes_per_s = 616e9},
         {745.3, 1143.0, 1142.5, 1137.9, 1135.9, 1134.9},
         {1387.0, 4448.4, 7658.4, 10789.3, 13900.7, 17006.3}},
        // The Quadro GV100 (Volta's row, read on it).
        {{.major = 7, .minor = 0, .sms = 80, .clock_hz = 1.627e9, .dram_bytes_per_s = 870e9},
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
        prices_gemma_attention_tiles(prefill_rows);
        prefills_pay_tail_calls(prefill_rows);
        std::ifstream census(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/decode_census.json");
        counts_the_kernels(json::parse(census));
    }
    prefills_moe_by_tiles(fixture);
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
    prices_moe_prefill_tiles();
    prices_dense_prefill_tiles();
    prices_prefill_small_kernels();
    prices_attention_tiles();
    weighs_the_drafters(fixture);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "choice: ok\n";
    return 0;
}
