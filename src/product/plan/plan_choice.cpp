#include "product/plan/plan_choice.h"

#include "runtime/engine/plan_dry_run.h"
#include "convert/formats.h"
#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/paged_kv_storage.h"
#include "product/speculation/speculation.h"
#include "spec/gpu_identity.h"
#include "runtime/engine/quant_rule.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

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

// A band an estimate carries (choice::Resolution): its basis, its rms of ln(predicted / measured),
// whether it would decide at its full width, and the Own floor a decision on it is held to.
Json band_json(const choice::Resolution& r) {
    return {{"basis", std::string(choice::resolution_basis_name(r.basis))},
            {"rms", round_to(r.rms, 4)},
            {"decides", r.decides},
            {"floor", round_to(r.floor, 4)}};
}

// The prefill estimate's measured error (choice::prefill_band_of): `rms` and `worst` only where it
// holds for this estimate (a dense model, the fast route, an Own estimate, a prompt inside the
// measured range, a generation with a row of two lab cards), else null and `why_code`. The
// generation's stated row beside it (`stated_rms`, `stated_worst`, `cards`; null without one).
Json prefill_band_json(const choice::PrefillBand& b) {
    const bool row = b.cards > 0;
    return {{"kind", b.moe ? "moe" : "dense"},
            {"prompt", b.prompt},
            {"from_prompt", b.from_prompt},
            {"to_prompt", b.to_prompt},
            {"rms", b.holds ? Json(round_to(b.stated_rms, 4)) : Json(nullptr)},
            {"worst", b.holds ? Json(round_to(b.stated_worst, 4)) : Json(nullptr)},
            {"why_code", std::string(b.why_code)},
            {"stated_rms", row ? Json(round_to(b.stated_rms, 4)) : Json(nullptr)},
            {"stated_worst", row ? Json(round_to(b.stated_worst, 4)) : Json(nullptr)},
            {"cards", b.cards}};
}

// What the estimates of one choice carry beside their numbers (#618 step 11).
struct EstimateContext {
    const choice::Resolutions* board   = nullptr; // the board's bands by basis
    const choice::AnswerScale* answers = nullptr; // nullptr: the family is not calibrated
};

Json estimates_json(const choice::Scored& s, const EstimateContext& ctx) {
    Json terms = Json::array();
    for (const choice::Term& t : s.accuracy.terms) {
        terms.push_back({{"axis", std::string(choice::axis_name(t.axis))},
                         {"value", t.value},
                         {"kld", round_to(t.kld, 5)},
                         {"measured", t.measured},
                         {"transferred", t.transferred},
                         {"source", t.source},
                         {"evidence", std::string(choice::evidence_name(t.evidence))}});
    }
    Json out = {{"own_context", s.own_context},
                {"fits", s.fits},
                {"decode_tok_s", round_to(s.speed.decode_per_stream, 1)},
                {"prefill_tok_s", round_to(s.speed.prefill, 0)},
                {"speed_transferred", s.speed.transferred},
                {"speed_basis", std::string(choice::speed_basis_name(s.speed.basis))},
                {"request_seconds", round_to(s.seconds, 3)},
                {"kld", round_to(s.accuracy.kld, 5)},
                {"kld_terms", terms},
                {"worth", round_to(s.worth, 4)}};
    // Why the speed is Transferred where a rule names it ("MoE route not queried", #766; "MoE route
    // not answered"; "MoE banks assumed", #784).
    if (!s.speed.why.empty()) { out["speed_why"] = std::string(s.speed.why); }
    // The decision's own price: the request mix on the reference card (null without one), what it
    // rests on, and the width the resolution rule weighed it at (0: not weighed).
    const bool decided      = s.decision_seconds > 0;
    out["decision_seconds"] = decided ? Json(round_to(s.decision_seconds, 3)) : Json(nullptr);
    out["decision_basis"] =
        decided ? Json(std::string(choice::speed_basis_name(s.decision_basis))) : Json(nullptr);
    out["weighed_at"] = round_to(s.weighed_at, 4);
    // The bands of the numbers shown: decode by the board's generation and the estimate's basis,
    // prefill by its generation's measured error where that holds for it.
    Json band = Json::object();
    if (ctx.board != nullptr) {
        band["decode"] = band_json(choice::resolution_for(*ctx.board, s.speed.basis));
    }
    band["prefill"] = prefill_band_json(s.prefill_band);
    out["band"]     = band;
    // The MMLU-Pro points the quantization's KLD against Q8_0 stands for, lost against Q8_0 (the
    // calibration's quantity: not the switches' terms), on the calibrated model's own measured
    // files only; else null and why.
    std::string_view why_not;
    const auto points             = choice::answer_points(s.accuracy, ctx.answers, &why_not);
    out["answer_points_lost"]     = points ? Json(round_to(*points, 2)) : Json(nullptr);
    out["answer_points_why_code"] = std::string(why_not);
    return out;
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
    // Every drafter the target carries, the profile's or not (each kind resolved alone); a kind
    // that only prices the MTP head beside a published profile is no option (pricing_only, #777).
    for (const spec::ResolvedSpeculation& other : subject.resolved->other_kinds) {
        if (other.pricing_only) { continue; }
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
        // The choice is the one the Engine runs (served_choices, as served_speculation picks it):
        // where the profile's default is one the Engine cannot run, the served choice's speeds,
        // depth and joint price describe the option, never the rejected default's; a served
        // choice published without its speeds is not measured.
        const spec::Profile runs  = served_choices(p);
        const spec::Choice& own   = spec::pick(runs, ask.thinking);
        const spec::Choice& other = spec::pick(runs, !ask.thinking);
        const bool split = per_request && !own.config.enabled() && other.config.enabled();
        const spec::Choice& c = split ? other : own;
        choice::DrafterOption option;
        option.label   = served->menu.primary.config.label();
        option.drafter = c.drafter;
        option.speedup = std::max(1.0, c.speedup());
        option.measured = !split && c.tok_s > 0 && c.plain_tok_s > 0;
        option.basis   = one.resolved->basis;
        const bool thinking = split ? !ask.thinking : ask.thinking;
        for (const spec::LoadBound& bound : p.load) {
            if (bound.depth == 0) {
                option.plain_from_streams = static_cast<std::uint32_t>(bound.lanes);
                break;
            }
            // A bound below the configuration's depth runs that depth (#303): its own measured
            // gain, not the configuration's; a bound at or above it, the configuration's.
            const std::int32_t depth = std::min(bound.depth, c.config.k);
            spec::SpecConfig at      = c.config;
            at.k                     = depth;
            const auto gain          = depth == c.config.k ? std::optional<double>(option.speedup)
                                                           : spec::measured_gain(p, at, thinking);
            // A DFlash bound's draft width is not a configuration of its own: the depth's gain at
            // one request's width stands for it (a ceiling, as the load's verify is).
            option.load.push_back({.streams  = static_cast<std::uint32_t>(bound.lanes),
                                   .speedup  = gain ? std::max(1.0, *gain) : 1.0,
                                   .depth    = static_cast<std::uint32_t>(depth),
                                   .bounded  = depth < c.config.k,
                                   .measured = gain.has_value()});
        }
        // The joint verify's price (#303): at C streams the profile runs the depth its load table
        // bounds (the configuration's below the first bound), and a round of C x (depth + 1) rows
        // costs more than one request's, R_1 / R_C of it, where the round cost prices it. A
        // DFlash bound's draft width (LoadBound::width) is not read: the facts price the drafter's
        // whole block (load_gain refuses it past the priced rows), the widest it can run.
        // The joint verify is priced at C = 2..kPricedStreams streams (Profile::joint_scale:
        // a round of C x (k + 1) rows up to kVerifyColumnRows, at its tiles' rows). A gain
        // nobody measured (option.measured false) is not priced: the estimate does not count it.
        for (std::int32_t streams = 2; option.measured && streams <= spec::kPricedStreams;
             ++streams) {
            if (option.plain_from_streams != 0 &&
                static_cast<std::uint32_t>(streams) >= option.plain_from_streams) {
                break;
            }
            // The last bound at or below these streams, as the load table above reads it.
            std::int32_t depth = c.config.k;
            for (const spec::LoadBound& bound : p.load) {
                if (bound.lanes > streams) { break; }
                depth = std::min(bound.depth, c.config.k);
            }
            spec::SpecConfig at = c.config;
            at.k                = depth;
            const auto priced   = p.joint_scale.find(at.label());
            if (depth < 1 || priced == p.joint_scale.end()) { continue; }
            const auto scale = priced->second.find(streams);
            if (scale != priced->second.end()) {
                option.joint.push_back(
                    {.streams = static_cast<std::uint32_t>(streams), .scale = scale->second});
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
    out.full_layers             = m.full_layers;
    out.full_head_dim           = m.full_key_dim;
    out.full_kv_heads    = m.full_layers > 0 ? static_cast<double>(m.full_kv_heads_total) / m.full_layers : 0;
    out.sliding_layers   = m.sliding_layers;
    out.full_attending_layers    = m.full_attending_layers;
    out.sliding_attending_layers = m.sliding_attending_layers;
    out.sliding_head_dim = m.sliding_key_dim;
    out.sliding_kv_heads =
        m.sliding_layers > 0 ? static_cast<double>(m.sliding_kv_heads_total) / m.sliding_layers : 0;
    out.sliding_attention_width = s.sliding_attention_width;
    out.input_table_params =
        s.tied_embeddings ? 0.0 : static_cast<double>(s.vocab_size) * static_cast<double>(s.hidden_size);
    // As stored apart from the head (a tied model's converter may keep a table of its own: Gemma 4's
    // artifact holds a Q8 input table beside its Q6 head); 0 where one object serves both.
    out.input_table_bytes = static_cast<double>(s.input_table_bytes);
    out.layers            = static_cast<std::uint32_t>(s.num_hidden_layers);
    out.hidden            = static_cast<double>(s.hidden_size);
    out.prompt_tail_calls = s.prompt_tail_calls;
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
    for (const runtime::PlanLinear& l : m.linears) {
        out.linears.push_back({l.n, l.k, l.per_token, l.expert, l.gated, l.format});
    }
    out.recurrent_state_elements = static_cast<double>(m.recurrent_state_elements);
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
    out.l2_bytes         = gpu.l2_bytes;
    out.fp4              = gpu.nvfp4;
    return out;
}

namespace {

// The op's stage kernel (ops/moe MoeStageKernel, as the registered query hands it on) as the
// choice's table names it.
choice::ExpertKernel expert_kernel_of(runtime::PlanExpertKernel kernel) noexcept {
    switch (kernel) {
    case runtime::PlanExpertKernel::Unknown:
        return choice::ExpertKernel::Unknown;
    case runtime::PlanExpertKernel::R8:
        return choice::ExpertKernel::R8;
    case runtime::PlanExpertKernel::RegisterMma:
        return choice::ExpertKernel::RegisterMma;
    case runtime::PlanExpertKernel::W4a8:
        return choice::ExpertKernel::W4a8;
    case runtime::PlanExpertKernel::VoltaHmma:
        return choice::ExpertKernel::VoltaHmma;
    }
    return choice::ExpertKernel::Unknown;
}

} // namespace

void with_moe_routes(choice::ModelFacts& model, const PlanSourceFacts& facts,
                     const runtime::PlanDeviceFacts& device) {
    const runtime::PlanMoeRouteQuery& query = runtime::plan_moe_route_query();
    // No query registered (ninferno-ui, #766): the MoE step's estimate says it was not asked.
    model.moe_route_unqueried = !query && model.moe();
    if (!query || !model.moe() || facts.model.expert_layers.empty()) { return; }
    // What the query reads of the model: its routed layers and their geometry.
    runtime::PlanModelFacts layers;
    layers.experts       = facts.model.experts;
    layers.experts_used  = facts.model.experts_used;
    layers.hidden        = facts.model.hidden;
    layers.expert_layers = facts.model.expert_layers;

    struct Answers {
        std::mutex mutex;
        std::map<std::pair<std::uint32_t, ActivationMode>, std::vector<choice::MoeStepRoute>> at;
    };

    auto answers     = std::make_shared<Answers>();
    model.moe_routes = [query, layers = std::move(layers), device,
                        answers](std::uint32_t rows, ActivationMode activations) {
        const std::lock_guard<std::mutex> lock(answers->mutex);
        if (const auto it = answers->at.find({rows, activations}); it != answers->at.end()) {
            return it->second;
        }
        std::vector<choice::MoeStepRoute> out;
        for (const runtime::PlanExpertLayers& kind : layers.expert_layers) {
            choice::MoeStepRoute step;
            step.share = static_cast<double>(kind.bytes);
            if (const auto route = query(device, layers, kind, rows, activations)) {
                step.gate_up = expert_kernel_of(route->gate_up);
                step.down    = expert_kernel_of(route->down);
            }
            const auto same =
                std::find_if(out.begin(), out.end(), [&](const choice::MoeStepRoute& s) {
                    return s.gate_up == step.gate_up && s.down == step.down;
                });
            if (same != out.end()) {
                same->share += step.share;
            } else {
                out.push_back(step);
            }
        }
        answers->at.emplace(std::pair{rows, activations}, out);
        return out;
    };
}

namespace {

choice::PrefillKernel prefill_kernel_of(runtime::PlanPrefillKernel kernel) {
    switch (kernel) {
    case runtime::PlanPrefillKernel::Unknown:
        return choice::PrefillKernel::Unknown;
    case runtime::PlanPrefillKernel::Mmq:
        return choice::PrefillKernel::Mmq;
    case runtime::PlanPrefillKernel::W4a8:
        return choice::PrefillKernel::W4a8;
    case runtime::PlanPrefillKernel::OtherA8:
        return choice::PrefillKernel::OtherA8;
    case runtime::PlanPrefillKernel::A16:
        return choice::PrefillKernel::A16;
    }
    return choice::PrefillKernel::Unknown;
}

} // namespace

void with_prefill_routes(choice::ModelFacts& model, const runtime::PlanDeviceFacts& device) {
    const runtime::PlanPrefillRouteQuery& query = runtime::plan_prefill_route_query();
    if (!query || device.compute_capability == 0) { return; }
    using Key = std::tuple<std::string, double, double, bool, std::uint32_t, ActivationMode>;

    struct Answers {
        std::mutex mutex;
        std::map<Key, choice::PrefillKernel> at;
    };

    auto answers         = std::make_shared<Answers>();
    model.prefill_routes = [query, device, answers](const choice::LinearShape& l,
                                                    std::uint32_t tokens,
                                                    ActivationMode activations) {
        const Key key{l.format, l.n, l.k, l.gated, tokens, activations};
        const std::lock_guard<std::mutex> lock(answers->mutex);
        if (const auto it = answers->at.find(key); it != answers->at.end()) { return it->second; }
        const choice::PrefillKernel kernel =
            prefill_kernel_of(query(device, l.format, l.n, l.k, l.gated, tokens, activations));
        answers->at.emplace(key, kernel);
        return kernel;
    };
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

namespace {

// The weight class of formats as the facts spell them (artifact::format_name; a spelling the table
// does not name, a GGUF type the converter does not keep, is Other), else, where the facts name
// none (a hub row), the label's (choice::label_formats).
choice::WeightClass weight_class_of(const std::vector<std::string>& formats,
                                    const std::string& label) {
    if (formats.empty()) { return choice::label_class(label); }
    std::vector<QType> read;
    for (const std::string& f : formats) {
        const std::optional<QType> q = find_format(f);
        if (!q) { return choice::WeightClass::Other; }
        read.push_back(*q);
    }
    return choice::weight_class_of(read);
}

// A site of the format may take 8-bit activations: its codes widen to int8 (int8_input: every
// GGUF block format but the ternary TQ1_0 / TQ2_0 and the two-bit Q1_0 / Q2_0, which keep BF16
// activations), or it is an encoded FP8, MXFP4 or NVFP4 site (convert::encoded_matrix_format; an
// NVFP4 file routes apart, QuantOption::nvfp4).
bool a8_input(QType q) noexcept {
    return format_facts(q).int8_input ||
           convert::encoded_matrix_format(q) != convert::MatrixFormat::Auto;
}

} // namespace

choice::QuantOption quant_option_of(const PlanSourceFacts& facts, const std::string& asked) {
    const std::string label = asked.empty() ? facts.shape.quant_label : asked;
    choice::QuantOption quant;
    quant.label = label.empty() ? "the model's weights" : label;
    // The files the facts were read at (a hub row's own: row_facts): a label's accuracy point is
    // their measurement only if they are the files measured (#299).
    quant.files = facts.shape.weights;
    // The target's weights: the model's own DFlash drafter (an artifact's dflash component) is
    // the drafter's, priced by its round, and the proposal head (proposal/*) the drafters' (read in
    // their place of the full head, never by a decode step); the bits over the elements the same
    // bytes hold (a GGUF's tensor table, an artifact's tensors: one measure on both paths).
    const std::uint64_t aside = facts.model.own_drafter_bytes + facts.model.proposal_bytes;
    const std::uint64_t target =
        facts.model.weight_bytes > aside ? facts.model.weight_bytes - aside : facts.model.weight_bytes;
    quant.bits_per_weight = facts.model.bits_per_weight(static_cast<double>(target));
    // What one token's decode step reads: the trunk, not the MTP layers (read by an MTP round
    // only, which the drafter's gain prices).
    quant.weight_bytes = static_cast<double>(target > facts.model.mtp_bytes ? target - facts.model.mtp_bytes
                                                                            : target);
    quant.routed_expert_bytes = static_cast<double>(facts.model.routed_expert_bytes);
    static_assert(static_cast<int>(runtime::ExpertPrefill::Mxfp4) == static_cast<int>(choice::ExpertBanks::Mxfp4));
    quant.expert_banks        = static_cast<choice::ExpertBanks>(facts.model.expert_prefill);
    quant.expert_banks_int8    = static_cast<choice::ExpertBanks>(facts.model.expert_prefill_int8);
    quant.expert_banks_assumed = facts.model.expert_prefill_assumed;
    quant.rest_weights         = weight_class_of(facts.shape.rest_layer_formats, label);
    // The weight class: the file's own layers' formats where known (a hub row's from its own
    // table only: row_facts; the vocabulary's apart), else the label (a row not read).
    quant.weights = weight_class_of(facts.shape.layer_formats, label);
    // The routes of the encodings the file holds (its weight_formats): NVFP4's where it holds
    // NVFP4, an int8 route where any format may take 8-bit activations (a8_input). A spelling the
    // table does not name (a GGUF type the converter does not keep) is not known to keep BF16 ones.
    const auto& formats = facts.shape.weight_formats;
    bool a8             = false;
    quant.nvfp4         = false;
    for (const std::string& f : formats) {
        const std::optional<QType> q = find_format(f);
        quant.nvfp4                  = quant.nvfp4 || q == QType::NVFP4;
        a8                           = a8 || !q || a8_input(*q);
    }
    quant.int8_route = formats.empty() || a8;
    // A quantization named by its listing's label whose formats were not read (a hub row without
    // its own table: row_facts gives it none of another file's) takes its routes from the formats
    // the label names (choice::label_formats; none named: an int8 route); one whose header was
    // read, and an artifact's recorded source, keep the routes of the encodings they hold (#811).
    if (!asked.empty() && formats.empty()) {
        const std::optional<std::set<QType>> named = choice::label_formats(label);
        quant.nvfp4                                = named && named->count(QType::NVFP4) > 0;
        quant.int8_route = !named || std::any_of(named->begin(), named->end(), a8_input);
    }
    return quant;
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
    with_moe_routes(model, facts, inputs.device);
    with_prefill_routes(model, inputs.device);
    const choice::CardFacts card   = choice_card_of(inputs.gpu);
    const choice::QuantOption quant = quant_option_of(facts, label);
    out.quant = quant;
    out.family                      = model.family;
    out.quant_authority             = choice::quant_authority(model, quant);

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
    out.kv_options          = kv_options(facts, inputs.device.compute_capability);
    // A mode the family does not run (int8 outside Qwen): the start refuses it, so there is no
    // choice to make at it.
    if (p.activations && std::find(allowed.activations.begin(), allowed.activations.end(),
                                   *p.activations) == allowed.activations.end()) {
        out.choice.why      = activation_refusal(facts, *p.activations);
        out.choice.why_code = "activations_not_run";
        return out;
    }
    // A KV codec pinned that the model does not take: the start refuses it, likewise.
    if (p.kv) {
        if (std::string why = kv_refusal(facts, *p.kv); !why.empty()) {
            out.choice.why      = std::move(why);
            out.choice.why_code = "kv_not_run";
            return out;
        }
    }
    out.choice         = choice::choose(model, card, out.quant, allowed, p, out.task, cached);
    out.drafters       = std::move(axis.served);
    out.drafter_options = axis.options;
    const std::uint32_t asked = ask.context != 0 ? ask.context : runtime::quant_own_targets(ask).front();
    out.depth                 = std::max<std::uint32_t>(512, asked / 4);
    out.prompt_tail_calls     = model.prompt_tail_calls;
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

std::vector<KvOption> kv_options(const PlanSourceFacts& facts, int compute_capability) {
    const KvCacheStorage codecs[] = {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                     KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                     KvCacheStorage::Fp8KeyNvfp4Value};
    const bool e4m3 =
        compute_capability <= 0 ||
        device_features_for(compute_capability / 10, compute_capability % 10).mma_fp8;
    const choice::ModelFacts model = choice_model_of(facts);
    const choice::Allowed allowed  = choice::allowed_axes(model, KvCacheStorage::BFloat16);
    // The Qwen families keep the paged pool, whose target append and attention are 256-wide only
    // (ops kv_cache_append, causal_softmax_attention: D256): its compact codecs need 256-wide
    // heads (paged_kv_storage_layout's 128-wide INT8 is the DFlash drafter pool's, not the target's).
    const bool paged = choice::is_qwen_family(model);
    std::vector<KvOption> out;
    for (const KvCacheStorage kv : codecs) {
        KvOption o;
        o.codec        = kv;
        o.slow_on_card = !e4m3 && (kv == KvCacheStorage::Fp8E4M3Row256 ||
                                   kv == KvCacheStorage::Fp8KeyNvfp4Value);
        const std::string name(kv_cache_storage_name(kv));
        if (kv == KvCacheStorage::BFloat16) {
            out.push_back(o);
            continue;
        }
        if (!facts.model.known()) {
            o.allowed  = false;
            o.why_code = "shape_unknown";
            o.why      = "the model's shape is not read: --kv-dtype " + name + " cannot be checked";
        } else if (std::find(allowed.kv.begin(), allowed.kv.end(), kv) == allowed.kv.end()) {
            o.allowed  = false;
            o.why_code = "family_codec";
            o.why      = "this model (" + facts.model.model_type +
                    ") keeps keys and values in BF16 or FP8 (--kv-dtype bf16|fp8), not " + name;
        } else if (paged && facts.model.full_key_dim != kD256KVCacheHeadDim) {
            o.allowed  = false;
            o.why_code = "head_width";
            o.why      = "this model's heads are " + std::to_string(facts.model.full_key_dim) +
                    " wide: --kv-dtype " + name + " takes 256-wide heads";
        }
        out.push_back(o);
    }
    return out;
}

std::string kv_refusal(const PlanSourceFacts& facts, KvCacheStorage kv) {
    for (const KvOption& o : kv_options(facts, 0)) {
        if (o.codec == kv) { return o.allowed ? std::string() : o.why; }
    }
    return "";
}

void refuse_unrun_kv_at_start(const std::filesystem::path& artifact, KvCacheStorage kv) {
    // A shape the facts do not name (no context length or geometry in the config) is the
    // engine's to judge at load: refused here only by the family's codecs and the head width.
    const PlanSourceFacts facts = plan_facts_of_artifact(artifact);
    if (!facts.model.known()) { return; }
    const std::string why = kv_refusal(facts, kv);
    if (!why.empty()) { throw std::invalid_argument(why); }
}

PinRefusal pin_refusal(const PlanSourceFacts& facts, const DryRunInputs& inputs) {
    if (inputs.pins.activations) {
        if (std::string why = activation_refusal(facts, inputs.options.activations); !why.empty()) {
            return {"activations_not_run", std::move(why)};
        }
    }
    if (inputs.pins.kv && !inputs.options.kv_cache_automatic) {
        if (std::string why = kv_refusal(facts, inputs.options.kv_cache); !why.empty()) {
            return {"kv_not_run", std::move(why)};
        }
    }
    return {};
}

void refuse_unrun_pins(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                       PlannedSpeculation& planned) {
    const PinRefusal refusal = pin_refusal(facts, inputs);
    if (refusal.code.empty()) { return; }
    planned.plan.fits         = false;
    planned.plan.rung         = 0;
    planned.plan.reason       = refusal.why;
    planned.plan.refused_code = refusal.code;
    planned.plan.fits_instead.clear();
}

std::uint64_t plan_free_now_of_nvml(std::uint64_t nvml_free_bytes) noexcept {
    return nvml_free_bytes > runtime::kPlanIdleContextBytes
               ? nvml_free_bytes - runtime::kPlanIdleContextBytes
               : 0;
}

PlannedSpeculation plan_now(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                          const ServedChoice* served, const ChoicePins& pins,
                          std::uint64_t free_now_bytes) {
    DryRunInputs now      = served != nullptr ? inputs_at_choice(inputs, *served, pins) : inputs;
    now.device.free_bytes = free_now_bytes;
    PlannedSpeculation planned = plan_model(facts, now);
    if (served != nullptr) { planned = plan_at_choice(facts, now, *served, pins, std::move(planned)); }
    refuse_unrun_pins(facts, now, planned);
    return planned;
}

nlohmann::json plan_now_json(const PlannedSpeculation& now, std::uint64_t free_now_bytes) {
    const runtime::DryRunPlan& p      = now.plan;
    const runtime::DryRunRung* chosen = p.chosen();
    return {{"free_now_bytes", free_now_bytes},
            {"fits_now", p.fits},
            {"context_now", chosen != nullptr ? chosen->rung.max_context : 0U},
            {"rung_now", chosen != nullptr ? p.rung + 1 : 0U},
            {"rungs", p.ladder.size()},
            {"private_contexts_now", chosen != nullptr && p.private_contexts()},
            {"survival_now", chosen != nullptr && chosen->rung.survival},
            {"reason_now", p.reason},
            {"reason_code_now", plan_reason_code(p).code},
            {"reason_params_now", plan_reason_code(p).params}};
}

ArtifactPlan plan_artifact(const std::filesystem::path& artifact, DryRunInputs inputs,
                           const std::filesystem::path& spec_profile) {
    ArtifactPlan out;
    out.facts         = plan_facts_of_artifact(artifact);
    out.subject       = std::make_unique<SpeculationSubject>(
        speculation_subject(artifact, inputs.gpu, spec_profile));
    inputs.subject    = out.subject.get();
    inputs.subject_of = {};
    out.inputs        = inputs;
    out.planned       = plan_model(out.facts, out.inputs);
    refuse_unrun_pins(out.facts, out.inputs, out.planned);
    out.served = choose_at_weights(out.facts, out.inputs, out.inputs.pins);
    // A start serves the choice's storages and drafter: the plan shown is of them.
    if (out.served && out.planned.plan.refused_code.empty()) {
        out.planned = plan_at_choice(out.facts, out.inputs, *out.served, out.inputs.pins,
                                     std::move(out.planned));
    }
    return out;
}

nlohmann::json artifact_plan_document(const ArtifactPlan& plan, const std::string& source) {
    nlohmann::json doc =
        dry_plan_document(plan.facts, source, "artifact", plan.inputs, plan.planned, nullptr);
    if (plan.served) { doc["choice"] = choice_json(*plan.served); }
    return doc;
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
        << " tok/s decode, " << std::setprecision(0) << s.speed.prefill
        << " prefill at the short request (" << served.prompt_tail_calls << " tail calls), KLD "
        << std::setprecision(4) << s.accuracy.kld << " (estimates) | " << served.choice.why;
    return out.str();
}

Json choice_json(const ServedChoice& served) {
    // The mix the chosen request time was priced on: an automatic context that settled below its
    // first target re-scored at the settled one (choose()).
    choice::Task priced = served.task;
    if (served.choice.context != 0) { priced.contexts = {served.choice.context}; }
    Json out = {{"task",
                 {{"context", served.choice.context},
                  {"contexts", served.task.contexts},
                  {"concurrency", served.task.concurrency},
                  {"prompt", served.task.prompt},
                  {"answer", served.task.answer},
                  {"long_prompt", priced.long_prompt()},
                  {"mix_short_requests", choice::kMixShortRequests},
                  {"mix_long_requests", choice::kMixLongRequests},
                  {"accuracy_weight", served.task.accuracy_weight}}},
                {"why", served.choice.why},
                {"by_worth", served.choice.by_worth},
                {"activations_by_worth", served.choice.activations_by_worth},
                {"resolution", round_to(served.choice.resolution, 4)},
                {"resolution_basis",
                 std::string(choice::resolution_basis_name(served.choice.resolution_basis))},
                {"resolution_decides", served.choice.resolution_decides},
                {"resolution_floor", round_to(served.choice.resolution_floor, 4)},
                {"chunk_by_speed", served.choice.chunk_by_speed},
                {"chunk_basis", std::string(choice::chunk_basis_name(served.choice.chunk_basis))}};
    // The activation modes the model runs (a choice the user may pin: --activations), whatever
    // is pinned now: what a selector offers.
    Json modes = Json::array();
    for (const ActivationMode m : served.allowed_activations) { modes.push_back(activation_name(m)); }
    // Every KV codec a start takes or refuses here, and why (what a selector offers).
    Json kv = Json::array();
    for (const KvOption& o : served.kv_options) {
        kv.push_back({{"codec", std::string(kv_cache_storage_name(o.codec))},
                      {"allowed", o.allowed},
                      {"why_code", o.why_code},
                      {"why", o.why},
                      {"slow_on_card", o.slow_on_card}});
    }
    out["kv_options"] = kv;
    out["activations_allowed"] = modes;
    // The why and the gates as codes a caller words (#618 step 11): what kind of why, the first
    // closed gate of the switches ("open" where the worth decides them), every closed gate with
    // the measurement that would open it.
    out["why_code"]  = served.choice.why_code;
    out["gate_code"] = served.choice.gate_code;
    Json gates       = Json::array();
    for (const choice::ClosedGate& g : served.choice.gates_closed) {
        gates.push_back({{"axis", std::string(g.axis)},
                         {"code", std::string(g.code)},
                         {"opens_with", std::string(g.opens_with)},
                         {"generation", g.generation},
                         {"family", g.family},
                         {"cards", g.cards}});
    }
    out["gates_closed"] = gates;
    // The reference card the decision priced on, beside the board the estimates are for (null:
    // none, so nothing was decided by worth).
    if (const auto& ref = served.choice.reference) {
        out["decided_on"] = {{"generation", std::string(ref->generation)},
                             {"cards", std::string(ref->cards)},
                             {"lent", ref->lent},
                             {"decides", ref->decides},
                             {"activations_decide", ref->activations_decide},
                             {"sms", ref->card.sms},
                             {"clock_mhz", round_to(ref->card.clock_hz / 1e6, 0)},
                             {"dram_gb_s", round_to(ref->card.dram_bytes_per_s / 1e9, 0)}};
    } else {
        out["decided_on"] = nullptr;
    }
    // The KLD's answer scale (choice_constants.h kAnswerScales): calibrated only for the one model
    // it was measured on, on its own measured files (the chosen quantization's term says), and
    // only for the quantization's KLD against Q8_0; elsewhere not calibrated and why. Only a
    // chosen candidate's measured quantization clears it: without one (no fit, a refused pin)
    // nothing says the files are the calibrated model's (no_choice).
    const choice::AnswerScale* answers = choice::answer_scale_of(served.family);
    std::string_view uncalibrated      = answers == nullptr ? "family_not_calibrated" : "no_choice";
    if (answers != nullptr && served.choice.chosen) {
        (void)choice::answer_points(served.choice.scored[*served.choice.chosen].accuracy, answers,
                                    &uncalibrated);
    }
    Json scale = {{"calibrated", uncalibrated.empty()}, {"family", served.family}};
    if (answers != nullptr) {
        scale["model"]          = std::string(answers->model);
        scale["benchmark"]      = std::string(answers->benchmark);
        scale["points_per_kld"] = answers->points_per_kld;
        scale["applies_to"]     = "quant_kld_vs_q8_0";
        scale["basis"]          = {{"models", 1},
                                   {"quantizations", answers->quantizations},
                                   {"noise_points", answers->noise_points}};
    }
    if (!uncalibrated.empty()) { scale["why_code"] = std::string(uncalibrated); }
    out["answer_scale"] = scale;
    const EstimateContext ctx{&served.choice.board_resolutions, answers};
    // The chunk a carried curve predicts (an estimate for an A/B; the pick is the default).
    if (served.choice.chunk_estimate != 0) { out["chunk_estimate"] = served.choice.chunk_estimate; }
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
                                served.drafter_options[s.candidate.drafter - 1].measured_at(
                                    served.task.concurrency);
    // The drafter's gain one request measured and the gain counted at the task's concurrency (the
    // depth its load table bounds there, scaled by the joint verify's price where the round cost
    // prices it; #303). Otherwise a ceiling under load.
    if (s.candidate.drafter > 0 && s.candidate.drafter <= served.drafter_options.size()) {
        const choice::DrafterOption& d = served.drafter_options[s.candidate.drafter - 1];
        const std::uint32_t streams    = served.task.concurrency;
        Json gain = {{"one_request", round_to(d.speedup, 3)},
                     {"at_concurrency", round_to(d.speedup_at(streams), 3)}};
        const choice::DrafterOption::LoadGain* in_force = nullptr;
        for (const choice::DrafterOption::LoadGain& g : d.load) {
            if (g.streams > streams) { break; }
            in_force = &g;
        }
        const bool plain = d.plain_from_streams != 0 && streams >= d.plain_from_streams;
        if (plain) {
            gain["plain_from_streams"] = d.plain_from_streams;
        } else if (in_force != nullptr && in_force->bounded) {
            gain["bounded_depth"]  = in_force->depth;
            gain["depth_measured"] = in_force->measured;
        }
        // Where a joint price supplies the gain the estimate counts
        // (DrafterOption::price_source_at, the one hold rule): none where the drafter is not
        // counted at these streams.
        const choice::DrafterOption::PriceSource source = d.price_source_at(streams);
        if (!plain && streams > 1 && d.measured_at(streams) && source.streams != 0) {
            // The one-request gain at the depth this load drafts times R_1 / R_C, the round cost's
            // price of the streams' rows verified together (spec/round_cost.h), at the count that
            // supplies the gain: this count's own price, or a smaller count's it is held to.
            gain["joint_verify_scale"] = round_to(source.scale, 3);
            const bool priced_here     = std::any_of(
                d.joint.begin(), d.joint.end(),
                [&](const choice::DrafterOption::JointScale& j) { return j.streams == streams; });
            if (source.streams == streams) {
                gain["note"] = "the one-request gain at the depth this load drafts, priced for the "
                               "streams' rows verified together (the round cost's R_1 / R_C); the "
                               "requests' attention and state bytes are not in it";
            } else {
                gain["joint_priced_at_streams"] = source.streams;
                gain["note"] = "the one-request gain at the depth this load drafts, held from the "
                               "price at " +
                               std::to_string(source.streams) +
                               " streams (the round cost's R_1 / R_C there; the joint verify at "
                               "this count is " +
                               (priced_here ? "priced no lower" : "not priced") +
                               "), the gain not climbing with load; the requests' attention and "
                               "state bytes are not in it";
            }
        } else if (!plain && streams > 1) {
            gain["note"] = "a ceiling: one request's gain at the depth this load drafts; the verify "
                           "of the streams' rows together is not priced";
        }
        chosen["drafter_gain"] = gain;
    }
    chosen["depth"]         = served.depth;
    chosen["decode_tok_s_at_depth"] = round_to(served.decode_at_depth, 1);
    // Whether the quantization's accuracy term may decide between quantizations.
    chosen["quality_authority"] =
        std::string(choice::quality_authority_name(served.quant_authority));
    chosen.update(estimates_json(s, ctx));
    out["chosen"]    = chosen;
    Json alternatives = Json::array();
    for (const choice::Alternative& a : choice::explain(served.choice)) {
        if (a.why_not.empty()) { continue; }
        Json row = {{"axis", std::string(choice::axis_name(a.axis))},
                    {"value", a.value},
                    {"why_not", a.why_not},
                    {"why_code", a.why_code}};
        if (a.scored != nullptr) {
            row.update(estimates_json(*a.scored, ctx));
            row["worth_delta"] = round_to(a.scored->worth - s.worth, 4);
        }
        alternatives.push_back(row);
    }
    out["alternatives"] = alternatives;
    return out;
}

} // namespace ninferno::product
