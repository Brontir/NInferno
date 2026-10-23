#include "product/plan/plan_choice.h"

#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/quant_rule.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace ninferno::product {

namespace {

using Json = nlohmann::json;

std::string activation_name(ActivationMode mode) {
    return mode == ActivationMode::Exact ? "exact" : mode == ActivationMode::Int8 ? "int8" : "fast";
}

double round_to(double value, int digits) {
    const double scale = std::pow(10.0, digits);
    return std::round(value * scale) / scale;
}

Json estimates_json(const choice::Scored& s) {
    Json terms = Json::array();
    for (const choice::Term& t : s.accuracy.terms) {
        terms.push_back({{"axis", std::string(choice::axis_name(t.axis))},
                         {"value", t.value},
                         {"kld", round_to(t.kld, 5)},
                         {"measured", t.measured},
                         {"transferred", t.transferred},
                         {"source", t.source}});
    }
    return {{"own_context", s.own_context},
            {"fits", s.fits},
            {"decode_tok_s", round_to(s.speed.decode_per_stream, 1)},
            {"prefill_tok_s", round_to(s.speed.prefill, 0)},
            {"speed_transferred", s.speed.transferred},
            {"request_seconds", round_to(s.seconds, 3)},
            {"kld", round_to(s.accuracy.kld, 5)},
            {"kld_terms", terms},
            {"worth", round_to(s.worth, 4)}};
}

Json candidate_json(const choice::Candidate& c) {
    return {{"kv", std::string(kv_cache_storage_name(c.kv))},
            {"ring", std::string(kv_cache_storage_name(c.ring))},
            {"state", std::string(recurrent_state_storage_name(c.state))},
            {"activations", activation_name(c.activations)}};
}

} // namespace

choice::ModelFacts choice_model_of(const PlanSourceFacts& facts) {
    const runtime::PlanModelFacts m = facts.model;
    const PlanTargetShape& s        = facts.shape;
    choice::ModelFacts out;
    out.family               = m.model_type;
    out.params               = m.params;
    out.active_params        = s.active_params > 0 ? s.active_params : m.params;
    out.experts              = s.experts;
    out.experts_per_token    = s.experts_per_token;
    out.routed_expert_params = s.routed_expert_params;
    out.input_table_params =
        s.tied_embeddings ? 0.0 : static_cast<double>(s.vocab_size) * static_cast<double>(s.hidden_size);
    out.kv_bytes_per_token = [m](KvCacheStorage storage) {
        return runtime::plan_kv_bytes_per_token(m, storage);
    };
    if (m.sliding_layers != 0) {
        out.ring_bytes_per_token = [m](KvCacheStorage storage) {
            return runtime::plan_ring_bytes_per_token(m, storage);
        };
        out.window = m.window;
    }
    if (m.recurrent_layers != 0) {
        out.state_bytes = [m](RecurrentStateStorage storage) {
            return runtime::plan_state_bytes(m, storage);
        };
    }
    // The Qwen 3.5 families' Gated DeltaNet state takes --state-dtype; Nemotron-H keeps FP32.
    out.recurrent = m.recurrent_layers != 0 && m.model_type.rfind("qwen3_5", 0) == 0;
    return out;
}

choice::CardFacts choice_card_of(const spec::GpuKey& gpu) {
    choice::CardFacts out;
    out.major            = gpu.compute_capability / 10;
    out.minor            = gpu.compute_capability % 10;
    out.sms              = gpu.sms;
    out.clock_hz         = gpu.sm_clock_khz * 1e3;
    out.dram_bytes_per_s = gpu.dram_gbps * 1e9;
    out.fp4              = gpu.nvfp4;
    return out;
}

DryRunInputs inputs_with(const DryRunInputs& inputs, const choice::Candidate& c) {
    DryRunInputs out                        = inputs;
    out.options.kv_cache                    = c.kv;
    out.options.kv_cache_automatic          = false;
    out.options.sliding_kv_cache            = c.ring;
    out.options.recurrent_state             = c.state;
    out.options.recurrent_state_automatic   = false;
    out.options.activations                 = c.activations;
    return out;
}

std::optional<ServedChoice> choose_at_weights(const PlanSourceFacts& facts,
                                              const DryRunInputs& inputs, const ChoicePins& pins) {
    if (!facts.model.known()) { return std::nullopt; }
    ServedChoice out;
    const choice::ModelFacts model = choice_model_of(facts);
    const choice::CardFacts card   = choice_card_of(inputs.gpu);
    choice::QuantOption quant;
    quant.label           = "own weights";
    quant.weight_bytes    = static_cast<double>(facts.model.weight_bytes);
    quant.bits_per_weight = facts.model.params > 0 ? quant.weight_bytes * 8 / facts.model.params : 0;
    const auto& formats   = facts.shape.weight_formats;
    quant.nvfp4      = std::find(formats.begin(), formats.end(), "nvfp4") != formats.end();
    quant.int8_route = formats.empty() || facts.shape.kquant ||
                       std::any_of(formats.begin(), formats.end(), [](const std::string& f) {
                           return f != "bf16" && f != "fp32" && f != "int32" && f != "f16";
                       });
    out.quants = {quant};

    choice::Pins p;
    p.quant = 0;
    const EngineOptions& o = inputs.options;
    if (pins.kv) { p.kv = o.kv_cache; }
    if (pins.activations) { p.activations = o.activations; }
    if (pins.state) { p.state = o.recurrent_state; }
    if (pins.ring) { p.ring = o.sliding_kv_cache; }

    out.task.concurrency = std::max<std::uint32_t>(o.max_concurrency, 1);
    const runtime::QuantAsk ask = quant_ask_of(inputs, facts.model.native_context);
    const auto own = [&](const choice::Candidate& c) -> std::uint32_t {
        const DryRunInputs pinned        = inputs_with(inputs, c);
        const PlannedSpeculation planned = plan_model(facts, pinned);
        return own_context_of(facts, pinned, planned);
    };
    // Own contexts cached by what changes the memory (the KV codec, the ring and the state).
    std::vector<std::pair<choice::Candidate, std::uint32_t>> seen;
    const choice::OwnContextFn cached = [&](const choice::Candidate& c) {
        for (const auto& [k, v] : seen) {
            if (k.kv == c.kv && k.ring == c.ring && k.state == c.state) { return v; }
        }
        const std::uint32_t v = own(c);
        seen.emplace_back(c, v);
        return v;
    };
    const choice::Allowed allowed = choice::allowed_axes(model, card);
    for (const std::uint32_t target : runtime::quant_own_targets(ask)) {
        out.task.context   = target;
        out.target_context = target;
        out.choice = choice::choose(model, card, out.quants, allowed, p, out.task, cached);
        if (out.choice.chosen) { break; }
    }
    return out;
}

std::string choice_line(const ServedChoice& served) {
    if (!served.choice.chosen) { return "choice | " + served.choice.why; }
    const choice::Scored& s = served.choice.scored[*served.choice.chosen];
    std::ostringstream out;
    out << "choice | KV " << kv_cache_storage_name(s.candidate.kv) << ", activations "
        << activation_name(s.candidate.activations);
    if (s.accuracy.terms.size() > 0) {
        for (const choice::Term& t : s.accuracy.terms) {
            if (t.axis == choice::Axis::State) { out << ", state " << t.value; }
        }
    }
    out << std::fixed << std::setprecision(1) << " | " << s.speed.decode_per_stream
        << " tok/s decode, " << std::setprecision(0) << s.speed.prefill << " prefill, KLD "
        << std::setprecision(4) << s.accuracy.kld << " (estimates) | " << served.choice.why;
    return out.str();
}

Json choice_json(const ServedChoice& served) {
    Json out = {{"task",
                 {{"context", served.task.context},
                  {"concurrency", served.task.concurrency},
                  {"prompt", served.task.prompt},
                  {"answer", served.task.answer},
                  {"accuracy_weight", served.task.accuracy_weight}}},
                {"why", served.choice.why}};
    if (!served.choice.chosen) {
        out["chosen"] = nullptr;
        return out;
    }
    const choice::Scored& s = served.choice.scored[*served.choice.chosen];
    Json chosen             = candidate_json(s.candidate);
    chosen.update(estimates_json(s));
    out["chosen"]    = chosen;
    Json alternatives = Json::array();
    for (const choice::Alternative& a : choice::explain(served.choice, served.quants)) {
        if (a.why_not.empty()) { continue; }
        Json row = {{"axis", std::string(choice::axis_name(a.axis))}, {"value", a.value}, {"why_not", a.why_not}};
        if (a.scored != nullptr) {
            row.update(estimates_json(*a.scored));
            row["worth_delta"] = round_to(a.scored->worth - s.worth, 4);
        }
        alternatives.push_back(row);
    }
    out["alternatives"] = alternatives;
    return out;
}

} // namespace ninferno::product
