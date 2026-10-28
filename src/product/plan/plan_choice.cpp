#include "product/plan/plan_choice.h"

#include "runtime/engine/plan_dry_run.h"
#include "product/speculation/speculation.h"
#include "spec/gpu_identity.h"
#include "runtime/engine/quant_rule.h"

#include <algorithm>
#include <limits>
#include <set>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
    return {{"chunk", c.chunk},
            {"kv", std::string(kv_cache_storage_name(c.kv))},
            {"ring", std::string(kv_cache_storage_name(c.ring))},
            {"state", std::string(recurrent_state_storage_name(c.state))},
            {"activations", activation_name(c.activations)}};
}

} // namespace

DrafterAxis drafter_axis(const SpeculationSubject& subject, const SpeculationAsk& ask) {
    DrafterAxis out;
    if (!subject.resolved || subject.resolved->profile.disabled) { return out; }
    const bool per_request       = per_request_of(ask, &subject);
    const auto base              = served_speculation(subject, ask.thinking, per_request);
    const spec::Profile& profile = subject.resolved->profile;
    std::set<spec::Mode> modes;
    for (const spec::Measurement& m : profile.measurements) {
        if (m.config.enabled()) { modes.insert(m.config.mode); }
    }
    // Every drafter the target carries, the profile's or not (each kind resolved alone).
    for (const spec::ResolvedSpeculation& other : subject.resolved->other_kinds) {
        const auto drafts = spec::drafting_modes(other.profile);
        modes.insert(drafts.begin(), drafts.end());
    }
    if (profile.measurements.empty() && base && base->engine.backend != SpeculativeBackend::None) {
        modes.insert(base->menu.mode);
        // A published profile's fit fallback drafts with another drafter where the default does
        // not fit (the 27B on GB203: DFlash2, MTP to fit): an option of its own.
        if (profile.fit_fallback && profile.fit_fallback->config.enabled()) {
            modes.insert(profile.fit_fallback->config.mode);
        }
    }
    for (const spec::Mode mode : modes) {
        const SpeculationSubject one = subject_of_mode(subject, mode);
        const spec::Profile& p       = one.resolved->profile;
        const auto served            = served_speculation(one, ask.thinking, per_request);
        if (!served || served->engine.backend == SpeculativeBackend::None ||
            served->menu.mode != mode) {
            continue;
        }
        // The gain of the default thinking flag's choice. Per request with that choice plain
        // and the other flag's drafting (served_speculation seeds the menu with it), the gain
        // is only the other flag's requests': not weighed, so the profile's choice stands.
        const spec::Choice& own   = spec::pick(p, ask.thinking);
        const spec::Choice& other = spec::pick(p, !ask.thinking);
        const bool split = per_request && !own.config.enabled() && other.config.enabled();
        const spec::Choice& c = split ? other : own;
        choice::DrafterOption option;
        option.label   = served->menu.primary.config.label();
        option.drafter = c.drafter;
        option.speedup = std::max(1.0, c.speedup());
        option.measured = !split && c.tok_s > 0 && c.plain_tok_s > 0;
        option.basis   = one.resolved->basis;
        for (const spec::LoadBound& bound : p.load) {
            if (bound.depth == 0) {
                option.plain_from_streams = static_cast<std::uint32_t>(bound.lanes);
                break;
            }
        }
        out.options.push_back(std::move(option));
        out.served.push_back(*served);
        if (base && base->engine.backend != SpeculativeBackend::None && base->menu.mode == mode) {
            out.default_drafter = out.options.size();
        }
    }
    return out;
}

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
    out.full_attention_width    = s.full_attention_width;
    out.sliding_attention_width = s.sliding_attention_width;
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
    // FP8 sliding rings: Gemma 4 and the dense decoders (types.h sliding_kv_cache).
    out.fp8_ring = m.model_type == "gemma4_text" || m.model_type == "dense_decoder";
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
    out.options.prefill_chunk               = c.chunk; // the widest: the ladder may narrow it
    return out;
}

std::optional<ServedChoice> choose_at_weights(const PlanSourceFacts& facts,
                                              const DryRunInputs& inputs, const ChoicePins& pins,
                                              const std::string& label, OwnContext memory) {
    if (!facts.model.known()) { return std::nullopt; }
    // The KV codec, activations and state change the output bytes: the choice reads an idle card of
    // the device's class, never the free memory now (another process's share); the start's ladder
    // still takes the rung the free memory allows (docs/maintainer/deterministic-decisions.md).
    DryRunInputs idle  = inputs;
    idle.device.free_bytes.reset();
    ServedChoice out;
    choice::ModelFacts model       = choice_model_of(facts);
    const choice::CardFacts card   = choice_card_of(inputs.gpu);
    choice::QuantOption quant;
    quant.label           = label.empty() ? "the model's weights" : label;
    quant.weight_bytes    = static_cast<double>(facts.model.weight_bytes);
    quant.bits_per_weight = facts.model.bits_per_weight(quant.weight_bytes);
    const auto& formats   = facts.shape.weight_formats;
    quant.nvfp4      = std::find(formats.begin(), formats.end(), "nvfp4") != formats.end();
    // An int8 route (AllowA8): any quantized format but the ternary and two-bit GGML blocks
    // (formats.cpp kept_block_policy: TQ1_0, TQ2_0, Q1_0 and Q2_0 keep BF16 activations).
    quant.int8_route = formats.empty() ||
                       std::any_of(formats.begin(), formats.end(), [](const std::string& f) {
                           return f != "bf16" && f != "fp32" && f != "int32" && f != "f16" &&
                                  f != "ggml_tq1_0" && f != "ggml_tq2_0" && f != "ggml_q1_0" &&
                                  f != "ggml_q2_0";
                       });
    // A quantization named by its listing's label (a hub row: the facts' header may be another
    // row's) takes its route from the label, as the quant table's estimate does.
    if (!label.empty()) {
        quant.nvfp4      = label.find("NVFP4") != std::string::npos || label.find("nvfp4") != std::string::npos;
        quant.int8_route = true;
        for (const char* a16 : {"TQ1_0", "TQ2_0", "Q1_0", "Q2_0"}) {
            if (label.find(a16) != std::string::npos) { quant.int8_route = false; }
        }
    }
    out.quant = quant;

    choice::Pins p;
    // The drafter: an axis of the choice where the subject's profile offers one; --spec given pins
    // it (none: plain decode; a backend the axis offers: that option).
    DrafterAxis axis;
    if (inputs.subject != nullptr) { axis = drafter_axis(*inputs.subject, inputs.ask); }
    if (!axis.options.empty() && !inputs.ask.automatic) {
        std::optional<std::size_t> pinned;
        if (inputs.ask.given.backend == SpeculativeBackend::None) { pinned = 0; }
        for (std::size_t i = 0; i < axis.served.size(); ++i) {
            if (axis.served[i].engine.backend == inputs.ask.given.backend) { pinned = i + 1; }
        }
        if (pinned) {
            p.drafter = pinned;
        } else {
            axis = {}; // a speculation the axis does not offer: planned as given, not chosen
        }
    }
    model.drafters       = axis.options;
    model.default_drafter = axis.default_drafter;
    const EngineOptions& o = inputs.options;
    if (pins.kv) { p.kv = o.kv_cache; }
    if (pins.activations) { p.activations = o.activations; }
    if (pins.state) { p.state = o.recurrent_state; }
    if (pins.ring) { p.ring = o.sliding_kv_cache; }
    if (pins.chunk) { p.chunk = o.prefill_chunk; }

    out.task.concurrency = std::max<std::uint32_t>(o.max_concurrency, 1);
    const runtime::QuantAsk ask = quant_ask_of(inputs, facts.model.native_context);
    const std::vector<std::uint32_t> steps = runtime::quant_own_targets(ask);
    const auto own = [&](const choice::Candidate& c) -> std::uint32_t {
        if (memory == OwnContext::Unbounded) { return std::numeric_limits<std::uint32_t>::max(); }
        DryRunInputs pinned = inputs_with(idle, c);
        // The candidate's drafter planned as given: its weights and state in the memory. An
        // explicit --spec pins its option and is planned with its own settings (depth, head).
        if (!axis.options.empty() && inputs.ask.automatic) {
            pinned.ask.automatic          = false;
            pinned.ask.depth_from_profile = false;
            pinned.ask.given = c.drafter > 0 ? axis.served[c.drafter - 1].engine : SpeculativeOptions{};
            pinned.options.speculative = pinned.ask.given;
        }
        if (memory == OwnContext::Steps) {
            // The first of Auto's steps it holds of its own (steps are tried longest first).
            for (const std::uint32_t step : steps) {
                if (holds_own_context(facts, pinned, step)) { return step; }
            }
            return 0;
        }
        const PlannedSpeculation planned = plan_model(facts, pinned);
        return own_context_of(facts, pinned, planned);
    };
    // Own contexts cached by what changes the memory (the KV codec, ring, state, chunk, drafter).
    std::vector<std::pair<choice::Candidate, std::uint32_t>> seen;
    const choice::OwnContextFn cached = [&](const choice::Candidate& c) {
        for (const auto& [k, v] : seen) {
            if (k.kv == c.kv && k.ring == c.ring && k.state == c.state && k.chunk == c.chunk &&
                k.drafter == c.drafter) {
                return v;
            }
        }
        const std::uint32_t v = own(c);
        seen.emplace_back(c, v);
        return v;
    };
    // The family's automatic KV codec on this card (the default where the generation's constants
    // do not decide).
    const auto storage =
        runtime::plan_family_storage(facts.model.model_type, inputs.device.compute_capability);
    const choice::Allowed allowed =
        choice::allowed_axes(model, storage ? storage->kv : KvCacheStorage::BFloat16);
    out.task.contexts  = steps;
    out.allowed_activations = allowed.activations;
    // A mode the family does not run (int8 outside Qwen): the start refuses it, so there is no
    // choice to make at it.
    if (p.activations && std::find(allowed.activations.begin(), allowed.activations.end(),
                                   *p.activations) == allowed.activations.end()) {
        out.choice.why = activation_refusal(facts, *p.activations);
        return out;
    }
    out.choice         = choice::choose(model, card, out.quant, allowed, p, out.task, cached);
    out.drafters       = std::move(axis.served);
    out.drafter_options = axis.options;
    const std::uint32_t asked = ask.context != 0 ? ask.context : runtime::quant_own_targets(ask).front();
    out.depth                 = std::max<std::uint32_t>(512, asked / 4);
    if (out.choice.chosen) {
        out.decode_at_depth = choice::speed_of(model, card, quant, out.choice.scored[*out.choice.chosen].candidate,
                                               out.task.concurrency, out.depth)
                                  .decode_per_stream;
    }
    return out;
}

std::string activation_refusal(const PlanSourceFacts& facts, ActivationMode mode) {
    if (!facts.model.known()) { return ""; }
    const choice::Allowed allowed = choice::allowed_axes(choice_model_of(facts), KvCacheStorage::BFloat16);
    if (std::find(allowed.activations.begin(), allowed.activations.end(), mode) !=
        allowed.activations.end()) {
        return "";
    }
    return std::string("--activations ") + activation_name(mode) +
           " is for Qwen 3.5/3.6 models; this model (" + facts.model.model_type +
           ") runs fast or exact";
}

void refuse_unrun_activations_at_start(const std::filesystem::path& artifact, ActivationMode mode) {
    const std::string why = activation_refusal(plan_facts_of_artifact(artifact), mode);
    if (!why.empty()) { throw std::invalid_argument(why); }
}

void refuse_unrun_activations(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                              PlannedSpeculation& planned) {
    if (!inputs.pins.activations) { return; }
    const std::string why = activation_refusal(facts, inputs.options.activations);
    if (why.empty()) { return; }
    planned.plan.fits         = false;
    planned.plan.rung         = 0;
    planned.plan.reason       = why;
    planned.plan.refused_code = "activations_not_run";
    planned.plan.fits_instead.clear();
}

DryRunInputs inputs_at_choice(const DryRunInputs& inputs, const ServedChoice& served,
                              const ChoicePins& pins) {
    const StartAxes axes = start_axes(served, pins);
    DryRunInputs out     = inputs;
    apply_start_axes(out.options, StartAxes{.kv          = axes.kv,
                                            .ring        = axes.ring,
                                            .state       = axes.state,
                                            .activations = axes.activations,
                                            .chunk       = axes.chunk});
    return out;
}

PlannedSpeculation plan_at_choice(const PlanSourceFacts& facts, const DryRunInputs& asked,
                                  const ServedChoice& served, const ChoicePins& pins,
                                  PlannedSpeculation planned) {
    if (!served.choice.chosen) { return planned; }
    const StartAxes axes = start_axes(served, pins);
    // The storages a start runs (apply_start_axes: the KV codec, the state, the rings and the
    // activations the choice took for the axes left automatic): the plan shown is of them, its
    // memory and its context, not of the family's automatic codec the choice did not take.
    const DryRunInputs inputs = inputs_at_choice(asked, served, pins);
    const EngineOptions& was  = asked.options;
    const bool storages_differ =
        (axes.kv && *axes.kv != planned.plan.kv_cache) ||
        (axes.state && *axes.state != planned.plan.recurrent) ||
        (axes.ring && *axes.ring != was.sliding_kv_cache) ||
        (axes.activations && *axes.activations != was.activations) ||
        (axes.chunk && *axes.chunk != was.prefill_chunk);
    if (!inputs.ask.automatic || inputs.subject == nullptr) {
        return storages_differ ? plan_model(facts, inputs) : planned;
    }
    const auto drafter = axes.drafter;
    const SpeculativeBackend backend = planned.speculation.engine.backend;
    const spec::Mode serving = backend == SpeculativeBackend::Mtp       ? spec::Mode::Mtp
                               : backend == SpeculativeBackend::DFlash  ? spec::Mode::DFlash
                               : backend == SpeculativeBackend::DFlash2 ? spec::Mode::DFlash2
                                                                        : spec::Mode::None;
    // The plan serves the choice's configuration already: its drafter at the depth the choice
    // weighed (a fit fallback or the model's head of the same mode is another configuration, and
    // the choice found its own fits: planned again at it).
    const choice::Scored& chosen_scored = served.choice.scored[*served.choice.chosen];
    const std::size_t option    = chosen_scored.candidate.drafter;
    const bool same_config =
        option == 0 || option > served.drafters.size() ||
        (served.drafters[option - 1].engine.backend == backend &&
         served.drafters[option - 1].engine.draft_tokens == planned.speculation.engine.draft_tokens);
    if (!drafter) { return storages_differ ? plan_model(facts, inputs) : planned; }
    if (*drafter == serving && same_config && !storages_differ) { return planned; }
    // Planned again at the choice's drafter (its mode pinned: the profile's own pick at other
    // storages may be another) and the choice's storages.
    DryRunInputs again = inputs;
    std::optional<SpeculationSubject> chosen;
    if (*drafter == spec::Mode::None) {
        again.ask.automatic = false;
        again.ask.given     = {};
    } else {
        chosen        = subject_of_mode(*inputs.subject, *drafter);
        again.subject = &*chosen;
    }
    planned = plan_model(facts, again);
    if (*drafter == spec::Mode::None) {
        planned.speculation.ask   = "auto";
        planned.speculation.basis = "choice";
        planned.speculation.notes.push_back(
            "the configuration choice decodes plainly: no drafter it weighed is worth more at the "
            "asked context and streams");
    }
    return planned;
}

StartAxes start_axes(const ServedChoice& served, const ChoicePins& pins) {
    StartAxes out;
    if (!served.choice.chosen) { return out; }
    const choice::Scored& s = served.choice.scored[*served.choice.chosen];
    const auto has          = [&](choice::Axis axis) {
        return std::any_of(s.accuracy.terms.begin(), s.accuracy.terms.end(),
                                    [axis](const choice::Term& t) { return t.axis == axis; });
    };
    if (!pins.kv) { out.kv = s.candidate.kv; }
    if (!pins.activations) { out.activations = s.candidate.activations; }
    if (!pins.state && has(choice::Axis::State)) { out.state = s.candidate.state; }
    if (!pins.ring && has(choice::Axis::Ring)) { out.ring = s.candidate.ring; }
    if (!pins.chunk) { out.chunk = s.candidate.chunk; }
    if (!served.drafters.empty()) {
        out.drafter = s.candidate.drafter == 0 ? spec::Mode::None
                                               : served.drafters[s.candidate.drafter - 1].menu.mode;
    }
    return out;
}

void apply_start_axes(EngineOptions& options, const StartAxes& axes) {
    if (axes.kv) {
        options.kv_cache           = *axes.kv;
        options.kv_cache_automatic = false;
    }
    if (axes.activations) { options.activations = *axes.activations; }
    if (axes.state) {
        options.recurrent_state           = *axes.state;
        options.recurrent_state_automatic = false;
    }
    if (axes.ring) { options.sliding_kv_cache = *axes.ring; }
    if (axes.chunk) { options.prefill_chunk = *axes.chunk; } // the widest; still automatic
}

std::optional<ServedChoice>
choose_for_artifact(const std::filesystem::path& artifact, int device, const EngineOptions& options,
                    const SpeculationAsk& ask, const std::string& spec_profile,
                    const ChoicePins& pins, std::string* why) {
    try {
        DryRunInputs inputs;
        inputs.device  = plan_device_of_cuda(device, true, runtime::plan_automatic_fit(options));
        inputs.gpu     = spec::gpu_identity(device).key;
        inputs.options = options;
        inputs.ask     = ask;
        const PlanSourceFacts facts       = plan_facts_of_artifact(artifact);
        const SpeculationSubject subject = speculation_subject(artifact, inputs.gpu, spec_profile);
        inputs.subject                    = &subject;
        return choose_at_weights(facts, inputs, pins);
    } catch (const std::exception& e) {
        if (why != nullptr) { *why = e.what(); }
        return std::nullopt;
    }
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
    if (s.candidate.chunk != choice::kDefaultChunk) { out << ", chunk " << s.candidate.chunk; }
    out << std::fixed << std::setprecision(1) << " | " << s.speed.decode_per_stream
        << " tok/s decode, " << std::setprecision(0) << s.speed.prefill << " prefill, KLD "
        << std::setprecision(4) << s.accuracy.kld << " (estimates) | " << served.choice.why;
    return out.str();
}

Json choice_json(const ServedChoice& served) {
    Json out = {{"task",
                 {{"context", served.choice.context},
                  {"contexts", served.task.contexts},
                  {"concurrency", served.task.concurrency},
                  {"prompt", served.task.prompt},
                  {"answer", served.task.answer},
                  {"accuracy_weight", served.task.accuracy_weight}}},
                {"why", served.choice.why},
                {"by_worth", served.choice.by_worth},
                {"chunk_by_speed", served.choice.chunk_by_speed}};
    // The activation modes the model runs (a choice the user may pin: --activations), whatever
    // is pinned now: what a selector offers.
    Json modes = Json::array();
    for (const ActivationMode m : served.allowed_activations) { modes.push_back(activation_name(m)); }
    out["activations_allowed"] = modes;
    if (!served.choice.chosen) {
        out["chosen"] = nullptr;
        return out;
    }
    const choice::Scored& s = served.choice.scored[*served.choice.chosen];
    Json chosen             = candidate_json(s.candidate);
    chosen["quant"]         = served.quant.label;
    chosen["drafter"]       = s.drafter;
    // Whether the decode estimates count the drafter's gain (a measured one).
    chosen["drafter_counted"] = s.candidate.drafter > 0 &&
                                s.candidate.drafter <= served.drafter_options.size() &&
                                served.drafter_options[s.candidate.drafter - 1].measured;
    chosen["depth"]         = served.depth;
    chosen["decode_tok_s_at_depth"] = round_to(served.decode_at_depth, 1);
    chosen.update(estimates_json(s));
    out["chosen"]    = chosen;
    Json alternatives = Json::array();
    for (const choice::Alternative& a : choice::explain(served.choice)) {
        if (a.why_not.empty()) { continue; }
        Json row = {{"axis", std::string(choice::axis_name(a.axis))},
                    {"value", a.value},
                    {"why_not", a.why_not},
                    {"why_code", a.why_code}};
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
