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
    out.recurrent = true;
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
    out.weight_bytes    = q.at("weight_bytes").get<double>();
    return out;
}

// Every fitted point within the fit's error: decode 5.2 % rms (worst 15.5 %), prefill 2.0 % (5.4 %).
void reproduces_the_measurements(const json& fixture) {
    double decode_sq = 0, prefill_sq = 0, decode_worst = 0, prefill_worst = 0;
    int decodes = 0, prefills = 0;
    for (const json& row : fixture.at("rows")) {
        const json& m           = fixture.at("models").at(row.at("model").get<std::string>());
        const ch::ModelFacts mf = model_of(m);
        const ch::CardFacts cf  = card_of(fixture.at("cards").at(row.at("card").get<std::string>()));
        const std::string q     = row.at("quant").get<std::string>();
        const ch::QuantOption qo = quant_of(q, m.at("quants").at(q));
        ch::Candidate c;
        c.kv          = kv_of(row.at("kv").get<std::string>());
        c.state       = row.at("state").get<std::string>() == "f32" ? RecurrentStateStorage::FP32
                                                                    : RecurrentStateStorage::FP16;
        c.activations = row.at("activations").get<std::string>() == "exact" ? ActivationMode::Exact
                                                                           : ActivationMode::Fast;
        const ch::Speed s = ch::speed_of(mf, cf, qo, c, row.at("concurrency").get<std::uint32_t>(),
                                         row.at("context").get<double>());
        const double d    = s.decode_per_stream / row.at("decode").get<double>() - 1;
        decode_sq += d * d;
        decode_worst = std::max(decode_worst, std::abs(d));
        ++decodes;
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
    check(decodes == 49 && prefills == 15, "the fixture's 49 decode points and 15 prefill points");
    check(decode_rms < 0.055 && decode_worst < 0.16, "decode within the fit's error");
    check(prefill_rms < 0.025 && prefill_worst < 0.06, "prefill within the fit's error");
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

// A memory oracle: weights plus KV at the context must stay under `budget` bytes.
ch::OwnContextFn oracle(const Bench& b, double budget) {
    return [&b, budget](const ch::Candidate& c) -> std::uint32_t {
        const double room = budget - b.quants[c.quant].weight_bytes;
        if (room <= 0) { return 0; }
        return static_cast<std::uint32_t>(room / b.model.kv_bytes_per_token(c.kv));
    };
}

const ch::Scored& chosen(const ch::Choice& c) { return c.scored[*c.chosen]; }

void chooses(const json& fixture) {
    ch::Task task;
    task.contexts = {32768};
    {
        // Q8_0 does not leave 32K here; Q4_K_M is the largest that fits. Fast beats exact (+26 %
        // time for 0.0044 KLD); INT8 KV beats FP8 (same cost, more accurate) and, by a hair, BF16
        // (on top of fast, INT8 KV adds about nothing: measured 0.00388 -> 0.00403 with the FP16
        // state).
        const Bench b = nine_b(fixture, "RTX 4070 Ti");
        const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                        {}, task, oracle(b, 9.6e9));
        check(c.chosen.has_value(), "4070 Ti: a choice");
        if (c.chosen) {
            const ch::Scored& s = chosen(c);
            check(b.quants[s.candidate.quant].label == "Q4_K_M", "4070 Ti: Q4_K_M, the largest that fits");
            check(s.candidate.activations == ActivationMode::Fast, "4070 Ti: fast over exact");
            check(s.candidate.kv == KvCacheStorage::Int8Group64, "4070 Ti: INT8 KV");
            check(s.candidate.state == RecurrentStateStorage::FP16, "4070 Ti: FP16 state under fast");
            std::cout << c.why << '\n';
        }
        const auto alts = ch::explain(c, b.quants);
        int named = 0;
        for (const ch::Alternative& a : alts) {
            if (a.why_not.empty()) { ++named; }
            check(a.scored != nullptr, "every alternative scored");
        }
        check(named == 4, "the chosen value once per axis (quant, kv, activations, state)");
        bool exact = false;
        for (const ch::Alternative& a : alts) {
            if (a.axis == ch::Axis::Activations && a.value == "exact") { exact = true; }
        }
        check(exact, "exact is an alternative (its FP32 state follows it)");
    }
    {
        // RTX 2080 Ti: BF16 and FP8 KV read slowly on Turing; INT8 stays.
        const Bench b = nine_b(fixture, "RTX 2080 Ti");
        const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                        {}, task, oracle(b, 9.0e9));
        check(c.chosen && chosen(c).candidate.kv == KvCacheStorage::Int8Group64, "2080 Ti: INT8 KV");
    }
    {
        // A big card: Q8_0 fits, so the rule takes it.
        const Bench b = nine_b(fixture, "RTX 5070 Ti");
        const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                        {}, task, oracle(b, 14e9));
        check(c.chosen && b.quants[chosen(c).candidate.quant].label == "Q8_0", "5070 Ti: Q8_0 fits");
    }
    {
        // Pins stay: exact asked, BF16 KV asked.
        const Bench b = nine_b(fixture, "RTX 4070 Ti");
        ch::Pins pins;
        pins.activations = ActivationMode::Exact;
        pins.kv          = KvCacheStorage::BFloat16;
        const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
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
        const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
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
        // Below the floor only when nothing above it fits; nothing fits: no choice.
        const Bench b = nine_b(fixture, "RTX 4070 Ti");
        const ch::Choice low = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                          {}, task, oracle(b, 4.4e9));
        check(low.chosen && b.quants[chosen(low).candidate.quant].label == "IQ3_XXS",
              "below the floor when nothing above fits");
        // Unknown bits rank at or above the floor, as in the quant rule: unlabelled weights that
        // fit come before known ones below the floor (the IQ3_XXS above), whatever their size.
        Bench unknown           = b;
        ch::QuantOption unnamed = b.quants[2]; // IQ3_XXS's bytes, less a little
        unnamed.label           = "unnamed";
        unnamed.bits_per_weight = 0;
        unnamed.weight_bytes *= 0.98;
        unknown.quants.push_back(unnamed);
        const ch::Choice unlabelled =
            ch::choose(unknown.model, unknown.card, unknown.quants,
                       ch::allowed_axes(unknown.model, KvCacheStorage::Int8Group64), {}, task,
                       oracle(unknown, 4.4e9));
        check(unlabelled.chosen &&
                  unknown.quants[chosen(unlabelled).candidate.quant].label == "unnamed",
              "unknown bits rank at or above the floor");
        // Unknown bits have no place on the quantization curve: the rule picks the quantization.
        check(!unlabelled.quant_by_worth && low.quant_by_worth,
              "unknown bits: the quantization is the rule's, not the worth's");
        const ch::Choice none = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                           {}, task, oracle(b, 1e9));
        check(!none.chosen, "nothing fits: no choice");
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
    // A 256-token prompt fills 256 of the chunk: 8 tokens per expert.
    const double short_prompt = ch::speed_of(moe, bw, q, fast, 1, 2048, 256).prefill / ch::speed_of(moe, bw, q, fast, 1, 2048).prefill;
    check(std::abs(short_prompt - 0.409 / 0.813) < 1e-6, "a short prompt fills only its own tokens of the chunk");
    // A 1280-token prompt at chunk 1024: one whole chunk and a 256-token call, each at its rate.
    ch::Task t1280;
    t1280.prompt = 1280;
    t1280.answer = 1;
    const double whole = ch::speed_of(moe, bw, q, fast, 1, 1280, 1024).prefill;
    const double rest  = ch::speed_of(moe, bw, q, fast, 1, 1280, 256).prefill;
    const double dec   = ch::speed_of(moe, bw, q, fast, 1, 1280.5, 1280).decode_per_stream;
    check(std::abs(ch::request_seconds(moe, bw, q, fast, t1280) - (1024 / whole + 256 / rest + 1 / dec)) < 1e-9,
          "the final partial chunk at its own rate");
}

// Without decision constants the quantization is ranked among the family's defaults only: Q8_0
// fitting only with NVFP4 KV does not displace Q4_K_M at the default INT8 KV (the 2080 Ti: no
// reference card).
void ranks_eligible_only(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 2080 Ti");
    ch::Task task;
    // Q8_0 + 32K of NVFP4 KV fits, Q8_0 + 32K of INT8 KV does not.
    const double budget = b.quants[1].weight_bytes + 33000.0 * b.model.kv_bytes_per_token(KvCacheStorage::Nvfp4Group16);
    const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    {}, task, oracle(b, budget));
    check(c.chosen && b.quants[chosen(c).candidate.quant].label == "Q4_K_M" &&
              chosen(c).candidate.kv == KvCacheStorage::Int8Group64 && c.context == 32768,
          "defaults only: Q4_K_M at INT8, not Q8_0 at NVFP4");
}

// The quantization curve: a measured label takes its point; other bits per weight interpolate in
// log KLD; at or above Q8_0's bits it is the reference. On Ada (a deciding generation) the worth
// weighs the quantizations: Q6_K, not the largest (Q8_0: +26 % time for -0.003 KLD) nor the
// fastest above the floor (Q4_K_M: -20 % time for +0.012 KLD).
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
    const double mid  = std::exp(std::log(0.01925) + (5.5 - 5.16) / (5.97 - 5.16) * (std::log(0.01061) - std::log(0.01925)));
    check(std::abs(quant_term(q).kld - mid) < 1e-9 && !quant_term(q).measured, "5.5 bits: interpolated");
    q.bits_per_weight = 9.0;
    check(quant_term(q).kld == 0, "above Q8_0's bits: the reference");
    q.bits_per_weight = (6.77 + 8.65) / 2; // above Q6_K: a straight line to 0 at Q8_0's bits
    check(std::abs(quant_term(q).kld - 0.00629 / 2) < 1e-9, "between Q6_K and Q8_0: halfway to 0");
    q.label           = "Q8_0";
    q.bits_per_weight = 8.5; // another model's Q8_0: its own reference
    check(quant_term(q).kld == 0 && !quant_term(q).measured, "any model's Q8_0: the reference");
    Bench with6 = b;
    ch::QuantOption q6 = b.quants[0];
    q6.label           = "Q6_K";
    q6.bits_per_weight = 6.77;
    q6.weight_bytes    = b.quants[0].weight_bytes * 6.77 / b.quants[0].bits_per_weight;
    with6.quants.push_back(q6);
    ch::Task task;
    const ch::Gates open{true, true};
    const ch::Choice choice = ch::choose(with6.model, with6.card, with6.quants,
                                         ch::allowed_axes(with6.model, KvCacheStorage::Int8Group64), {}, task,
                                         oracle(with6, 20e9), open);
    check(choice.chosen && with6.quants[chosen(choice).candidate.quant].label == "Q6_K" &&
              choice.why.find("best worth among the quantizations") != std::string::npos,
          "Ada: the quantizations by worth (Q6_K)");
    int codes = 0;
    for (const ch::Alternative& a : ch::explain(choice, with6.quants)) {
        if (a.axis == ch::Axis::Quant && a.value == "Q8_0") {
            check(a.why_code == "lower_worth", "Q8_0 fits and lost on worth, not on the rule");
            ++codes;
        }
        if (a.axis == ch::Axis::Quant && a.value == "IQ3_XXS") {
            check(a.why_code == "quant_rule", "a quantization below the floor: the rule's tiers, not the worth");
            ++codes;
        }
        if (a.axis == ch::Axis::Kv && a.value == "fp8") {
            check(a.why_code == "lower_worth", "FP8 KV: the same cost, less accurate");
            ++codes;
        }
    }
    check(codes == 3, "Q8_0, IQ3_XXS and FP8 KV explained");
    // The constants' own gates: the worth decides once the accuracy terms rest on two cards.
    const ch::Choice gated = ch::choose(with6.model, with6.card, with6.quants,
                                        ch::allowed_axes(with6.model, KvCacheStorage::Int8Group64), {}, task,
                                        oracle(with6, 20e9));
    check(gated.by_worth == ch::accuracy_decides("ada", "qwen3_5_text"),
          "the worth decides only with the accuracy terms on two cards");
    // A family the terms were not measured on keeps the rule and its defaults.
    Bench gemma = with6;
    gemma.model.family = "gemma4_text";
    const ch::Choice other = ch::choose(gemma.model, gemma.card, gemma.quants,
                                        ch::allowed_axes(gemma.model, KvCacheStorage::Fp8E4M3Row256), {}, task,
                                        oracle(gemma, 20e9), open);
    check(!other.by_worth && other.why.find("no accuracy terms were measured on gemma4_text") != std::string::npos,
          "another family: the rule's largest and its defaults");
    // Two quantizations alike in every respect: a tie, broken by the smaller label.
    Bench twins           = b;
    twins.quants          = {b.quants[0], b.quants[0]};
    twins.quants[1].label = "Q4_K_M copy";
    const ch::Choice tie = ch::choose(twins.model, twins.card, twins.quants,
                                      ch::allowed_axes(twins.model, KvCacheStorage::Int8Group64), {}, task,
                                      oracle(twins, 20e9), open);
    int ties = 0;
    for (const ch::Alternative& a : ch::explain(tie, twins.quants)) {
        if (a.axis == ch::Axis::Quant && a.value == "Q4_K_M copy") {
            check(a.why_code == "tie_order" && a.why_not.find("int8") == std::string::npos,
                  "an equal quantization: the smaller label");
            ++ties;
        }
    }
    check(ties == 1, "the equal quantization explained");
    // Listed the other way round, the tie still goes to the smaller label, as the quant rule's.
    Bench swapped           = twins;
    swapped.quants          = {twins.quants[1], twins.quants[0]};
    for (const ch::Gates& g : {open, ch::Gates{false, true}}) {
        const ch::Choice c = ch::choose(swapped.model, swapped.card, swapped.quants,
                                        ch::allowed_axes(swapped.model, KvCacheStorage::Int8Group64), {}, task,
                                        oracle(swapped, 20e9), g);
        check(c.chosen && swapped.quants[chosen(c).candidate.quant].label == "Q4_K_M",
              "an equal quantization: the smaller label wherever it is listed");
        for (const ch::Alternative& a : ch::explain(c, swapped.quants)) {
            if (a.axis == ch::Axis::Quant && a.value == "Q4_K_M copy") {
                check(a.why_code == "tie_order", "an equal quantization explained as the label tie");
            }
        }
    }
    // Without decision constants (Turing) the rule's largest stands.
    const Bench t = nine_b(fixture, "RTX 2080 Ti");
    Bench t6      = t;
    t6.quants.push_back(q6);
    const ch::Choice rule = ch::choose(t6.model, t6.card, t6.quants, ch::allowed_axes(t6.model, KvCacheStorage::Int8Group64),
                                       {}, task, oracle(t6, 20e9));
    check(rule.chosen && t6.quants[chosen(rule).candidate.quant].label == "Q8_0", "Turing: the rule's largest (Q8_0)");
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
    const ch::Choice ada = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task, own);
    check(ada.chosen && chosen(ada).candidate.chunk == 2048 && ada.chunk_by_speed,
          "Ada, an MoE: chunk 2048 for a 2048-token prompt");
    int chunks = 0;
    for (const ch::Alternative& a : ch::explain(ada, b.quants)) {
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
    const ch::Choice pinned = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), wide, task, own);
    check(pinned.chosen && chosen(pinned).candidate.chunk == 4096 && !pinned.chunk_by_speed, "a pinned chunk stays");
    b.card = card_of(fixture.at("cards").at("RTX 5070 Ti"));
    const ch::Choice bw = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64), {}, task, own);
    check(bw.chosen && chosen(bw).candidate.chunk == 1024, "Blackwell: the default chunk");
}

// A coarse quantization does not make a switch's error free: the quantization's term adds on top
// of the switches' root sum square, so on Ada a ~3-bit model still keeps INT8 KV over K8V4.
void coarse_quant_keeps_kv(const json& fixture) {
    Bench b = nine_b(fixture, "RTX 4090");
    b.quants = {b.quants[2]}; // IQ3_XXS-like, 3.1 bits: a large interpolated quantization term
    ch::Task task;
    task.contexts = {8192};
    const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    {}, task, oracle(b, 20e9));
    check(c.chosen && chosen(c).candidate.kv == KvCacheStorage::Int8Group64, "a coarse quantization keeps INT8 KV");
}

// Auto's steps: every own-context step above the floor bits before any quantization below them.
void steps_before_the_floor(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4070 Ti");
    ch::Task task;
    task.contexts = {32768, 16384, 8192};
    // Q4_K_M leaves room for 20K tokens of INT8 KV, IQ3_XXS for far more: 16K at Q4_K_M wins.
    const double budget = b.quants[0].weight_bytes + 20000.0 * b.model.kv_bytes_per_token(KvCacheStorage::Int8Group64);
    // KV pinned to INT8: where the worth decides, a narrower codec would itself reach 32K at
    // Q4_K_M (the KV codec is a memory lever too).
    ch::Pins pins;
    pins.kv = KvCacheStorage::Int8Group64;
    const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    pins, task, oracle(b, budget));
    check(c.chosen && b.quants[chosen(c).candidate.quant].label == "Q4_K_M" && c.context == 16384,
          "16K at Q4_K_M before 32K below the floor");
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
    const ch::Choice choice = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                         {}, task, oracle(b, 9.6e9));
    check(choice.by_worth && choice.chosen && chosen(choice).candidate.activations == ActivationMode::Fast,
          "Ada decides: fast over exact");
    if (choice.chosen) { std::cout << choice.why << '\n'; }
}

// Qwen 3.5 MoE has switch terms of its own (INT8 KV 0.0033, 3.4 times dense's) on two cards and no
// quantization curve: on Ada the worth decides its KV codec and activations, the quant rule its
// quantization.
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
    const ch::Choice choice = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                         {}, task, oracle(b, 40e9));
    check(choice.by_worth && !choice.quant_by_worth && choice.chosen &&
              b.quants[chosen(choice).candidate.quant].label == "Q4_K_M",
          "MoE on Ada: the switches by worth, the quantization by the rule (the largest)");
    if (choice.chosen) { std::cout << choice.why << '\n'; }
}

// Contradictory pins say so instead of "nothing fits".
void contradictory_pins(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4070 Ti");
    ch::Pins pins;
    pins.activations = ActivationMode::Exact;
    pins.state       = RecurrentStateStorage::FP16;
    const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    pins, ch::Task{}, oracle(b, 20e9));
    check(!c.chosen && c.why.find("refuses") != std::string::npos, "exact with a pinned FP16 state: refused");
    pins       = {};
    pins.quant = 7;
    const ch::Choice out = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                      pins, ch::Task{}, oracle(b, 20e9));
    check(!out.chosen && out.why.find("not among") != std::string::npos, "a pinned quantization out of range");
}

// The coupled storages are explained: the recurrent state's other values beside the chosen.
void explains_the_state(const json& fixture) {
    const Bench b = nine_b(fixture, "RTX 4070 Ti");
    ch::Task task;
    const ch::Choice c = ch::choose(b.model, b.card, b.quants, ch::allowed_axes(b.model, KvCacheStorage::Int8Group64),
                                    {}, task, oracle(b, 9.6e9));
    int states = 0, coupled = 0;
    for (const ch::Alternative& a : ch::explain(c, b.quants)) {
        if (a.axis == ch::Axis::State) {
            ++states;
            if (a.why_code == "coupled") { ++coupled; }
        }
    }
    check(states == 3 && coupled == 2, "the state: the chosen f16 and its two other values");
}

} // namespace

int main() {
    std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/choice/speed_measured.json");
    const json fixture = json::parse(in);
    reproduces_the_measurements(fixture);
    chooses(fixture);
    decides_by_generation(fixture);
    steps_before_the_floor(fixture);
    ranks_eligible_only(fixture);
    weighs_the_quantizations(fixture);
    chooses_the_chunk(fixture);
    coarse_quant_keeps_kv(fixture);
    explains_the_state(fixture);
    contradictory_pins(fixture);
    activations_by_generation(fixture);
    moe_family(fixture);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "choice: ok\n";
    return 0;
}
