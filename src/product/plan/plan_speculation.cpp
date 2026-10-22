#include "product/plan/plan_speculation.h"

#include "convert/drafters.h"
#include "core/device_class.h"
#include "product/speculative_options.h"
#include "runtime/engine/kv_capacity.h"
#include "spec/load_model.h"
#include "spec/manifest.h"
#include "spec/registry.h"

#include <algorithm>
#include <exception>
#include <set>
#include <utility>

namespace ninferno::product {

namespace {

using Json = nlohmann::json;

// The registry's engine of a family (the drafter plan's DrafterTarget::engine,
// TargetFacts::family).
std::string engine_of(std::string_view model_type) {
    if (model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text") { return "qwen3_5"; }
    if (model_type == "gemma4_text") { return "gemma4"; }
    if (model_type == "nemotron_h") { return "nemotron_h"; }
    if (model_type == "dense_decoder" || model_type == "moe_decoder") { return "dense"; }
    return {};
}

// The registry's drafter kind a mode drafts with in a family.
std::string drafter_kind_of(spec::Mode mode, std::string_view engine) {
    switch (mode) {
    case spec::Mode::Mtp:
        return engine == "gemma4"       ? "gemma4_assistant"
               : engine == "nemotron_h" ? "nemotron_h_mtp"
                                        : "mtp";
    case spec::Mode::DFlash:
        return "dflash";
    case spec::Mode::DFlash2:
        return "dflash2";
    default:
        return {};
    }
}

// The decay of acceptance along a block (spec/load_model.h: RTX 4090 grid, DFlash2 0.97, MTP
// 0.91): a lab constant per drafter kind.
double decay_of(spec::Mode mode) noexcept { return mode == spec::Mode::Mtp ? 0.91 : 0.97; }

spec::TargetFacts target_facts_of(const PlanSourceFacts& facts) {
    const runtime::PlanModelFacts& m = facts.model;
    const PlanTargetShape& s         = facts.shape;
    spec::TargetFacts out;
    out.active_params        = s.active_params;
    out.bytes_per_weight     = m.params > 0 ? static_cast<double>(m.weight_bytes) / m.params : 0.0;
    out.kquant               = s.kquant;
    out.hidden_size          = s.hidden_size;
    out.num_hidden_layers    = s.num_hidden_layers;
    out.vocab_size           = s.vocab_size;
    out.experts              = s.experts;
    out.experts_per_token    = s.experts_per_token;
    out.routed_expert_params = s.routed_expert_params;
    out.family               = engine_of(m.model_type);
    out.weight_formats       = s.weight_formats;
    return out;
}

// The bounds of a profile's table the Engine took (SpeculativeOptions::load), as the registry's.
std::vector<spec::LoadBound> bounds_of(const std::vector<SpeculationLoad>& load) {
    std::vector<spec::LoadBound> out;
    for (const SpeculationLoad& bound : load) {
        out.push_back(
            {.lanes = static_cast<std::int32_t>(bound.lanes),
             .depth = static_cast<std::int32_t>(bound.draft_tokens),
             .width = bound.draft_width == DFlashDraftWidth::Block    ? spec::DraftWidth::Block
                      : bound.draft_width == DFlashDraftWidth::Drafts ? spec::DraftWidth::Drafts
                                                                      : spec::DraftWidth::Default});
    }
    return out;
}

// A step of `lanes` rows under the load bounds (Engine: the last bound at or below the lanes).
std::int32_t depth_at(const SpeculativeOptions& engine, std::uint32_t lanes) {
    if (engine.backend == SpeculativeBackend::None) { return 0; }
    std::int32_t depth = static_cast<std::int32_t>(engine.draft_tokens);
    for (const SpeculationLoad& bound : engine.load) {
        if (lanes >= bound.lanes) {
            depth = std::min(depth, static_cast<std::int32_t>(bound.draft_tokens));
        }
    }
    return depth;
}

// Whole contexts an automatic pool holds at an explicit context (GenerationService's
// pool_contexts: 0 when the pool or the context is automatic in the other sense).
std::uint32_t pool_contexts_of(const runtime::DryRunPlan& plan, const EngineOptions& options) {
    const runtime::DryRunRung* chosen = plan.chosen();
    if (chosen == nullptr || options.kv_capacity.mode != KvCapacityMode::Automatic ||
        options.max_context_automatic || chosen->rung.max_context == 0) {
        return 0;
    }
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(chosen->rung.kv_pages) *
                                      runtime::kAutomaticContextGrain / chosen->rung.max_context);
}

std::string label_of(const SpeculativeOptions& speculative) {
    if (speculative.backend == SpeculativeBackend::None) { return "plain"; }
    return std::string(speculative_backend_name(speculative.backend)) + ":" +
           std::to_string(speculative.draft_tokens);
}

// Where the served profile came from (ServedSpeculation::basis, set where it was resolved).
std::string basis_of(const ServedSpeculation& served) {
    return served.basis.empty() ? std::string("own") : served.basis;
}

} // namespace

SpeculationSubject planned_subject(const PlannedTarget& target, const spec::GpuKey& gpu) {
    const PlanSourceFacts& facts = target.facts;
    SpeculationSubject out;
    out.components = {"text"};
    if (facts.shape.builtin_mtp) { out.components.insert("mtp"); }
    out.support = spec::engine_support(facts.model.model_type);
    // A GGUF conversion carries no proposal head; a checkpoint's recipe may (Qwen's +head): the
    // conservative side, the full head.
    out.proposal_head = false;

    convert::DrafterTarget t;
    t.repo    = target.repo;
    t.lineage = target.lineage;
    if (!facts.shape.base_model.empty() &&
        std::find(t.lineage.begin(), t.lineage.end(), facts.shape.base_model) == t.lineage.end()) {
        t.lineage.push_back(facts.shape.base_model);
    }
    t.engine       = engine_of(facts.model.model_type);
    t.architecture = facts.model.architecture;
    t.geometry     = spec::Geometry{facts.shape.hidden_size, facts.shape.num_hidden_layers,
                                facts.shape.vocab_size};
    t.builtin_mtp  = facts.shape.builtin_mtp;
    t.components.assign(out.components.begin(), out.components.end());
    t.gguf_source = target.gguf;
    // A conversion by this build records its mapping's version (never a stale target).
    t.gguf_mapper = target.gguf ? "planned" : "";

    std::optional<spec::SpeculationManifest> manifest;
    const spec::ModelEntry* entry = nullptr;
    const std::string label       = target.repo.empty() ? facts.model.architecture : target.repo;
    try {
        if (!target.gguf || target.with_drafters) {
            // The manifest the converter writes (convert/acquire.cpp: a checkpoint's drafters by
            // default, a GGUF's with --with-drafters): each drafter the target's engine runs is
            // attached, a component one embedded in the artifact.
            convert::DrafterPlan plan = convert::plan_drafters(t);
            Json doc                  = plan.to_json();
            for (std::size_t i = 0; i < plan.drafters.size() && i < doc["drafters"].size(); ++i) {
                const convert::PlannedDrafter& d = plan.drafters[i];
                Json& item                       = doc["drafters"][i];
                if (!d.supported || !d.reason.empty()) { continue; }
                if (d.decl.kind == "mtp") {
                    if (facts.shape.builtin_mtp) { item["status"] = "embedded"; }
                } else if (!d.decl.component.empty()) {
                    item["status"] = "embedded";
                    out.components.insert(d.decl.component);
                } else {
                    item["status"] = "attached";
                    out.attached_drafters.push_back(d.decl);
                }
            }
            manifest = spec::manifest_from_json(doc);
        } else {
            // No manifest (a GGUF hub model without --with-drafters): the registry entry of its
            // shape (spec::resolve_speculation's third source), its vocabulary unread.
            entry = spec::Registry::builtin().nearest(t.engine, t.geometry, "", t.builtin_mtp);
            if (entry != nullptr &&
                (t.geometry.hidden_size == 0 || t.geometry.num_hidden_layers == 0 ||
                 spec::shape_distance(entry->geometry, t.geometry) != 0.0)) {
                entry = nullptr;
            }
        }
        out.resolved =
            spec::resolve_planned_speculation(manifest, entry, out.components, gpu, label);
    } catch (const std::exception& error) {
        out.resolved = spec::ResolvedSpeculation{};
        out.resolved->notes.push_back(std::string("drafters not planned: ") + error.what());
    }
    apply_builtin_default(out);
    return out;
}

spec::GpuKey gpu_key_of(const runtime::PlanDeviceFacts& device) {
    DeviceClass d =
        synthetic_device_class(device.pci_device_id, device.compute_capability / 10,
                               device.compute_capability % 10, device.sms, 0, device.total_bytes);
    d.facts = machine_facts_of(d);
    return spec::GpuKey{d.similarity_chain(),
                        d.global_bytes,
                        d.features.mma_mxf4nvf4,
                        d.class_chain().size() - 1,
                        d.sms,
                        d.compute_capability(),
                        d.facts.reference_clock_khz,
                        d.dram_gbps(),
                        d.features.mma_slow,
                        d.facts.source != FactSource::Verified || d.facts.driver_read};
}

PlannedSpeculation plan_with_speculation(const PlanSourceFacts& facts,
                                         const runtime::PlanDeviceFacts& device,
                                         const EngineOptions& options,
                                         const SpeculationSubject* subject,
                                         const SpeculationAsk& ask, const spec::GpuKey& gpu,
                                         std::uint64_t drafter_bytes) {
    PlannedSpeculation out;
    SpeculationReport& r            = out.speculation;
    const std::uint32_t concurrency = std::max(1U, options.max_concurrency);
    const auto plan_of              = [&](const SpeculativeOptions& speculative) {
        EngineOptions planned     = options;
        planned.speculative       = speculative;
        runtime::PlanModelFacts m = facts.model;
        // A separate drafter (an artifact of its own: Gemma 4's assistant, gpt-oss's DFlash) is
        // uploaded beside the target.
        if (speculative.backend != SpeculativeBackend::None && !speculative.drafter.empty()) {
            m.drafter_bytes = drafter_bytes;
        }
        return runtime::dry_run_plan(m, device, planned);
    };
    const auto step = [&](std::string name, const SpeculativeOptions& speculative,
                          const runtime::DryRunPlan& plan, std::string note = {}) {
        r.fallbacks.push_back(SpeculationFallbackStep{.step  = std::move(name),
                                                      .label = label_of(speculative),
                                                      .tried = true,
                                                      .fits  = plan.fits,
                                                      .note  = std::move(note)});
    };
    const auto serve = [&](const SpeculativeOptions& speculative, runtime::DryRunPlan plan) {
        r.engine = speculative;
        r.label  = label_of(speculative);
        if (!r.fallbacks.empty()) { r.fallbacks.back().served = true; }
        out.plan = std::move(plan);
    };

    std::optional<ServedSpeculation> served;
    if (!ask.automatic) {
        SpeculationAsk explicit_ask = ask;
        SpeculativeOptions& given   = explicit_ask.given;
        if (ask.depth_from_profile && given.backend != SpeculativeBackend::None) {
            // GenerationService: the profile's depth for this backend, else the default.
            given.draft_tokens = default_draft_tokens(given.backend);
            if (subject != nullptr && subject->resolved && subject->resolved->found()) {
                const spec::Choice& choice = spec::pick(subject->resolved->profile, ask.thinking);
                if (backend_of(choice.config.mode) == given.backend && choice.config.k > 0) {
                    given.draft_tokens = static_cast<std::uint32_t>(choice.config.k);
                }
            }
        }
        const bool drafts        = given.backend != SpeculativeBackend::None;
        r.ask                    = drafts ? label_of(given) : "none";
        r.basis                  = drafts ? "explicit" : "none";
        runtime::DryRunPlan plan = plan_of(given);
        step(drafts ? "explicit" : "plain", given, plan,
             drafts ? "--spec keeps its drafter: a plan that does not fit is refused" : "");
        serve(given, std::move(plan));
        if (drafts) {
            r.depths = {static_cast<std::int32_t>(given.draft_tokens)};
            for (const std::uint32_t k : given.request_draft_tokens) {
                r.depths.push_back(static_cast<std::int32_t>(k));
            }
        }
    } else {
        r.ask = "auto";
        if (subject != nullptr) {
            served = served_speculation(*subject, ask.thinking, ask.per_request);
            if (subject->resolved) {
                for (const std::string& note : subject->resolved->notes) {
                    r.notes.push_back(note);
                }
            }
        }
        const auto with_given = [&](const ServedSpeculation& s) {
            return with_given_settings(s.engine, ask.given);
        };
        if (!served || served->menu.mode == spec::Mode::None) {
            const SpeculativeOptions plain;
            runtime::DryRunPlan plan = plan_of(plain);
            if (served) {
                // A profile whose choice is plain decode: measured faster without speculation.
                r.source = served->source;
                r.basis  = "measured_plain";
                step("plain", plain, plan, "measured faster without speculation here");
            } else {
                step("plain", plain, plan, "no speculation profile for this model on this GPU");
            }
            serve(plain, std::move(plan));
        } else {
            r.source = served->source;
            r.basis  = basis_of(*served);
            for (const std::string& note : served->notes) {
                if (std::find(r.notes.begin(), r.notes.end(), note) == r.notes.end()) {
                    r.notes.push_back(note);
                }
            }
            SpeculativeOptions picked = with_given(*served);
            runtime::DryRunPlan plan  = plan_of(picked);
            step("profile", picked, plan, served->source);
            bool built = plan.fits;
            if (built) { serve(picked, std::move(plan)); }
            // The profile's fit fallback.
            if (!built) {
                if (auto fallback = served_fit_fallback(*subject, ask.per_request)) {
                    const SpeculativeOptions sp = with_given(*fallback);
                    runtime::DryRunPlan p       = plan_of(sp);
                    step("fit_fallback", sp, p, fallback->source);
                    if (p.fits) {
                        served = std::move(fallback);
                        serve(sp, std::move(p));
                        built = true;
                    }
                }
            }
            // Shallower drafts with the same drafter: the deepest that fits.
            if (!built) {
                std::optional<std::pair<SpeculativeOptions, runtime::DryRunPlan>> last;
                const auto deepest = deepest_fitting_shallower(
                    served->menu.primary,
                    [&](const spec::SpecConfig& config, double floor) {
                        return served_shallower(*subject, ask.per_request, config, floor);
                    },
                    [&](const ServedSpeculation& candidate) {
                        const SpeculativeOptions sp = with_given(candidate);
                        runtime::DryRunPlan p       = plan_of(sp);
                        step("shallower", sp, p);
                        const bool fits = p.fits;
                        if (fits) { last.emplace(sp, std::move(p)); }
                        return fits;
                    });
                if (deepest && last) {
                    // The step the search ended on is the one served.
                    for (auto& s : r.fallbacks) {
                        s.served = false;
                        if (s.step == "shallower" && s.label == label_of(last->first) && s.fits) {
                            s.served = true;
                        }
                    }
                    served   = *deepest;
                    r.engine = last->first;
                    r.label  = label_of(last->first);
                    out.plan = std::move(last->second);
                    built    = true;
                }
            }
            // The model's own MTP head, before plain decode.
            if (!built) {
                if (auto head = served_head_fallback(*subject, ask.per_request)) {
                    const SpeculativeOptions sp = with_given(*head);
                    runtime::DryRunPlan p       = plan_of(sp);
                    step("head", sp, p, head->source);
                    if (p.fits) {
                        served = std::move(head);
                        serve(sp, std::move(p));
                        built = true;
                    }
                }
            }
            if (!built) {
                served.reset();
                const SpeculativeOptions plain;
                runtime::DryRunPlan p = plan_of(plain);
                step("plain", plain, p, "nothing speculative fits: --spec auto decodes plainly");
                serve(plain, std::move(p));
            }
            // The pool rule: a drafter that leaves the automatic pool fewer whole contexts than C
            // while plain decode holds more gives way (to the MTP head when it keeps four fifths of
            // plain decode's contexts).
            if (served && built) {
                const std::uint32_t with_drafter = pool_contexts_of(out.plan, options);
                if (with_drafter > 0 && with_drafter < concurrency) {
                    const SpeculativeOptions plain;
                    runtime::DryRunPlan plain_plan = plan_of(plain);
                    const std::uint32_t plain_pool = pool_contexts_of(plain_plan, options);
                    bool head_kept                 = false;
                    if (plain_pool > with_drafter) {
                        if (auto head = served_head_fallback(*subject, ask.per_request)) {
                            const SpeculativeOptions sp = with_given(*head);
                            runtime::DryRunPlan p       = plan_of(sp);
                            head_kept = p.fits && 5 * pool_contexts_of(p, options) >=
                                                      4 * std::min(plain_pool, concurrency);
                            step("pool_head", sp, p,
                                 "the drafter leaves " + std::to_string(with_drafter) + " of " +
                                     std::to_string(concurrency) + " whole contexts");
                            if (head_kept) {
                                for (auto& s : r.fallbacks) { s.served = false; }
                                served = std::move(head);
                                serve(sp, std::move(p));
                            }
                        }
                        if (!head_kept) {
                            for (auto& s : r.fallbacks) { s.served = false; }
                            step("pool_plain", plain, plain_plan,
                                 "plain decode holds " + std::to_string(plain_pool) +
                                     " whole contexts against " + std::to_string(with_drafter));
                            served.reset();
                            serve(plain, std::move(plain_plan));
                        }
                    }
                }
            }
        }
    }

    // The served choice's facts: drafter, depths, load, the model's load and class depths.
    const std::string engine_name = engine_of(facts.model.model_type);
    if (r.engine.backend != SpeculativeBackend::None) {
        spec::Mode mode = r.engine.backend == SpeculativeBackend::Mtp      ? spec::Mode::Mtp
                          : r.engine.backend == SpeculativeBackend::DFlash ? spec::Mode::DFlash
                                                                           : spec::Mode::DFlash2;
        r.drafter_kind  = drafter_kind_of(mode, engine_name);
        r.drafter =
            r.engine.drafter.empty() ? r.drafter_kind : r.engine.drafter.filename().string();
        if (served) {
            r.depths = served->menu.depths();
            r.source = served->source;
        }
        r.load                 = bounds_of(r.engine.load);
        r.load_basis           = r.load.empty() ? "none" : "measured";
        r.depth_at_concurrency = depth_at(r.engine, concurrency);
        // The round cost of this card for this target and drafter, from facts (spec/load_model.h).
        const spec::DrafterFacts drafter{.kind = r.drafter_kind,
                                         .proposal_head =
                                             r.engine.proposal_head == ProposalHead::Optimized};
        const std::optional<spec::RoundCost> cost =
            spec::round_cost_of(gpu, target_facts_of(facts), drafter);
        double tokens               = 0;
        std::int32_t measured_depth = static_cast<std::int32_t>(r.engine.draft_tokens);
        if (served) { // the profile's tokens per round at its depth
            const spec::MenuChoice& primary = served->menu.primary;
            measured_depth                  = primary.config.k;
            if (subject != nullptr && subject->resolved) {
                const spec::Choice& choice = spec::pick(subject->resolved->profile, ask.thinking);
                if (choice.config.mode == mode && choice.config.k == measured_depth) {
                    tokens = choice.tokens_per_round;
                }
            }
        }
        const double d = decay_of(mode);
        const double a = tokens > 1.0 && measured_depth > 0
                             ? spec::first_draft_acceptance(tokens, measured_depth, d)
                             : 0.0;
        if (!cost) {
            r.model_note = "no cost model for this card, target or drafter";
        } else if (!cost->priced || !cost->fit.reliable) {
            r.model_note = cost->fit.reason;
        }
        if (cost && cost->priced && a > 0 && !r.depths.empty()) {
            r.modeled_load = spec::modeled_load(*cost, a, d, r.depths,
                                                static_cast<std::int32_t>(kMaximumConcurrency));
        } else if (a <= 0 && r.model_note.empty()) {
            r.model_note = "no measured tokens per round for the chosen depth";
        }
        // The class factors of the registry for this model pair (spec/context_class.h).
        spec::AcceptanceKey key;
        if (const spec::ModelEntry* entry = spec::Registry::builtin().by_repo(ask.model_repo)) {
            key.model = entry->id;
        }
        key.target_artifact = ask.artifact_id;
        key.drafter         = r.drafter_kind;
        key.depth           = measured_depth;
        key.lanes           = static_cast<std::int32_t>(concurrency);
        std::array<spec::ClassFactor, spec::kContextClasses> factors{};
        for (std::size_t i = 0; i < spec::kContextClasses; ++i) {
            factors[i] = spec::factor(spec::Registry::builtin().context_classes(), key,
                                      static_cast<spec::ContextClass>(i));
        }
        const std::vector<std::int32_t> captured =
            r.depths.empty() ? std::vector<std::int32_t>{measured_depth} : r.depths;
        const auto depths =
            spec::class_depths(cost && a > 0 ? cost : std::optional<spec::RoundCost>{}, a, d,
                               captured, factors, measured_depth);
        for (std::size_t i = 0; i < spec::kContextClasses; ++i) {
            SpeculationClassRow& row = r.classes[i];
            row.context_class        = static_cast<spec::ContextClass>(i);
            row.below  = i < spec::kContextBounds.size() ? spec::kContextBounds[i] : 0;
            row.depth  = depths[i].depth;
            row.basis  = depths[i].estimate ? "modeled" : "no estimate";
            row.reason = depths[i].reason;
        }
    } else {
        for (std::size_t i = 0; i < spec::kContextClasses; ++i) {
            SpeculationClassRow& row = r.classes[i];
            row.context_class        = static_cast<spec::ContextClass>(i);
            row.below  = i < spec::kContextBounds.size() ? spec::kContextBounds[i] : 0;
            row.basis  = "no estimate";
            row.reason = "plain decode: nothing speculates";
        }
    }
    return out;
}

Json to_json(const SpeculationReport& r) {
    const auto bounds = [](const std::vector<spec::LoadBound>& load) {
        Json out = Json::array();
        for (const spec::LoadBound& b : load) {
            Json item = {{"lanes", b.lanes}, {"depth", b.depth}};
            if (b.width != spec::DraftWidth::Default) {
                item["width"] = std::string(spec::draft_width_name(b.width));
            }
            out.push_back(std::move(item));
        }
        return out;
    };
    Json classes = Json::array();
    for (const SpeculationClassRow& row : r.classes) {
        classes.push_back({{"class", spec::class_name(row.context_class)},
                           {"below_tokens", row.below == 0 ? Json(nullptr) : Json(row.below)},
                           {"depth", row.depth},
                           {"basis", row.basis},
                           {"reason", row.reason}});
    }
    Json fallbacks = Json::array();
    for (const SpeculationFallbackStep& s : r.fallbacks) {
        fallbacks.push_back({{"step", s.step},
                             {"label", s.label},
                             {"fits", s.fits},
                             {"served", s.served},
                             {"note", s.note}});
    }
    const bool drafts = r.engine.backend != SpeculativeBackend::None;
    return {{"ask", r.ask},
            {"served", r.label},
            {"backend", speculative_backend_name(r.engine.backend)},
            {"depth", drafts ? Json(r.engine.draft_tokens) : Json(0)},
            {"drafter", r.drafter},
            {"drafter_kind", r.drafter_kind},
            {"basis", r.basis},
            {"source", r.source},
            {"depths", r.depths},
            {"floor", drafts ? Json(r.engine.acceptance_floor) : Json(nullptr)},
            {"load", bounds(r.load)},
            {"load_label", spec::load_label(r.load)},
            {"load_basis", r.load_basis.empty() ? std::string("none") : r.load_basis},
            {"depth_at_concurrency", r.depth_at_concurrency},
            {"modeled_load", bounds(r.modeled_load)},
            {"model_note", r.model_note},
            {"classes", std::move(classes)},
            {"fallbacks", std::move(fallbacks)},
            {"notes", r.notes}};
}

} // namespace ninferno::product
