#include "product/plan/plan_speculation.h"

#include "convert/drafters.h"
#include "core/device_class.h"
#include "product/plan/plan_quants.h"
#include "product/speculative_options.h"
#include "runtime/engine/kv_capacity.h"
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
// the choice's family).
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

// The components a planned target's artifact carries before its drafters: the text, and its own MTP
// layers where it has them.
std::set<std::string> own_components(const PlanSourceFacts& facts) {
    std::set<std::string> out{"text"};
    if (facts.shape.builtin_mtp) { out.insert("mtp"); }
    return out;
}

// The target as the converter's drafter plan sees it (convert::describe_checkpoint and
// describe_gguf_artifact): repository, lineage (the target's, then the base models its facts
// name), engine, shape, built-in MTP and components.
convert::DrafterTarget drafter_target(const PlannedTarget& target) {
    const PlanSourceFacts& facts = target.facts;
    convert::DrafterTarget t;
    t.repo    = target.repo;
    t.lineage = target.lineage;
    for (const auto& base : facts.shape.base_models) {
        if (std::find(t.lineage.begin(), t.lineage.end(), base) == t.lineage.end()) {
            t.lineage.push_back(base);
        }
    }
    t.engine       = engine_of(facts.model.model_type);
    t.architecture = facts.model.architecture;
    t.geometry     = spec::Geometry{facts.shape.hidden_size, facts.shape.num_hidden_layers,
                                facts.shape.vocab_size};
    t.builtin_mtp  = facts.shape.builtin_mtp;
    const auto components = own_components(facts);
    t.components.assign(components.begin(), components.end());
    t.gguf_source = target.gguf;
    // A conversion by this build records its mapping's version (never a stale target).
    t.gguf_mapper = target.gguf ? "planned" : "";
    return t;
}

// The drafters the converter declares for a target (convert/acquire.cpp: a checkpoint's by default,
// a GGUF's with --with-drafters): the manifest it writes, each drafter the target's engine runs
// attached, a component one embedded in the artifact.
struct DeclaredDrafters {
    spec::SpeculationManifest manifest;
    std::set<std::string> components; // embedded beside the text (a DFlash component)
    // The drafters whose weights are not the target's listing: attached ones, and embedded ones
    // from a repository of their own (a DFlash drafter of Qwen 3.5), counted as attached ones are.
    std::vector<spec::DrafterDecl> attached;
};

DeclaredDrafters declared_drafters(const PlannedTarget& target) {
    const convert::DrafterPlan plan = convert::plan_drafters(drafter_target(target));
    Json doc                        = plan.to_json();
    DeclaredDrafters out;
    for (std::size_t i = 0; i < plan.drafters.size() && i < doc["drafters"].size(); ++i) {
        const convert::PlannedDrafter& d = plan.drafters[i];
        Json& item                       = doc["drafters"][i];
        if (!d.supported || !d.reason.empty()) { continue; }
        if (d.decl.kind == "mtp") {
            if (target.facts.shape.builtin_mtp) { item["status"] = "embedded"; }
        } else if (!d.decl.component.empty()) {
            item["status"] = "embedded";
            out.components.insert(d.decl.component);
            if (!d.decl.repo.empty()) { out.attached.push_back(d.decl); }
        } else {
            item["status"] = "attached";
            out.attached.push_back(d.decl);
        }
    }
    out.manifest = spec::manifest_from_json(doc);
    return out;
}

} // namespace

SpeculationSubject planned_subject(const PlannedTarget& target, const spec::GpuKey& gpu) {
    const PlanSourceFacts& facts = target.facts;
    SpeculationSubject out;
    out.components = own_components(facts);
    out.support    = spec::engine_support(facts.model.model_type);
    // A GGUF conversion carries no proposal head; a checkpoint's recipe may (Qwen's +head): the
    // conservative side, the full head.
    out.proposal_head = false;

    const std::string engine = engine_of(facts.model.model_type);
    std::optional<spec::SpeculationManifest> manifest;
    const spec::ModelEntry* entry = nullptr;
    const std::string label       = target.repo.empty() ? facts.model.architecture : target.repo;
    try {
        if (!target.gguf || target.with_drafters) {
            DeclaredDrafters declared = declared_drafters(target);
            out.components.insert(declared.components.begin(), declared.components.end());
            out.attached_drafters = std::move(declared.attached);
            manifest              = std::move(declared.manifest);
        } else {
            // No manifest (a GGUF hub model without --with-drafters): the registry entry of its
            // shape (spec::resolve_speculation's third source), its vocabulary unread.
            const spec::Geometry geometry{facts.shape.hidden_size, facts.shape.num_hidden_layers,
                                          facts.shape.vocab_size};
            entry = spec::Registry::builtin().nearest(engine, geometry, "", facts.shape.builtin_mtp);
            if (entry != nullptr &&
                (geometry.hidden_size == 0 || geometry.num_hidden_layers == 0 ||
                 spec::shape_distance(entry->geometry, geometry) != 0.0)) {
                entry = nullptr;
            }
        }
        out.resolved = spec::resolve_planned_speculation(manifest, entry, out.components, gpu,
                                                         label, target.weights, target.quant,
                                                         engine,
                                                         target.repo.empty() || target.commit.empty()
                                                             ? std::string()
                                                             : target.repo + "@" + target.commit);
    } catch (const std::exception& error) {
        out.resolved = spec::ResolvedSpeculation{};
        out.resolved->notes.push_back(std::string("drafters not planned: ") + error.what());
    }
    apply_builtin_default(out);
    return out;
}

SpeculationSubject hub_quant_subject(const PlannedTarget& target, const spec::GpuKey& gpu,
                                     const std::optional<std::filesystem::path>& cached,
                                     const std::filesystem::path& spec_profile) {
    if (!cached) { return planned_subject(target, gpu); }
    SpeculationSubject out = speculation_subject(*cached, gpu, spec_profile);
    try {
        out.attached_drafters = declared_drafters(target).attached;
    } catch (const std::exception&) {} // none planned: none counted, as planned_subject
    return out;
}

PlannedSubjects::PlannedSubjects(PlannedTarget model, spec::GpuKey gpu, ConversionOf conversion_of,
                         std::filesystem::path spec_profile)
    : model_(std::move(model)),
      gpu_(std::move(gpu)),
      conversion_of_(std::move(conversion_of)),
      spec_profile_(std::move(spec_profile)) {
    model_.with_drafters = !model_.repo.empty(); // a hub model's start attaches them, a local one's not
}

const SpeculationSubject& PlannedSubjects::of(const std::string& quant, const std::string& weights) {
    if (const auto it = made_.find(quant); it != made_.end()) { return it->second; }
    PlannedTarget target = model_;
    target.weights       = weights;
    target.quant         = quant;
    const std::optional<std::filesystem::path> converted =
        conversion_of_ && !quant.empty() ? conversion_of_(quant) : std::nullopt;
    return made_.emplace(quant, hub_quant_subject(target, gpu_, converted, spec_profile_))
        .first->second;
}

std::function<const SpeculationSubject*(const QuantListing&)> PlannedSubjects::lookup() {
    return [this](const QuantListing& q) -> const SpeculationSubject* {
        return q.drafter ? nullptr : &of(q.quant, q.weights);
    };
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
                        d.dram_gbps()};
}

bool per_request_of(const SpeculationAsk& ask, const SpeculationSubject* subject) {
    if (ask.automatic && ask.per_request_from_profile) {
        return subject != nullptr && per_request_pays(*subject, ask.thinking);
    }
    return ask.per_request;
}

namespace {

// The backend of a configuration label ("dflash:7", "mtp:5+head"), None for "plain".
SpeculativeBackend backend_of_label(std::string_view label) {
    const std::string_view mode = label.substr(0, label.find(':'));
    return mode == "plain" ? SpeculativeBackend::None : parse_speculative_backend(mode);
}

} // namespace

bool uses_attached_drafter(const SpeculationSubject& subject, const SpeculationReport& report) {
    // A drafter file given (--drafter) drafts in their place.
    if (!report.engine.drafter.empty()) { return false; }
    std::set<SpeculativeBackend> served = {report.engine.backend};
    if (report.per_request) {
        for (const auto* variant : {&report.with_thinking, &report.without_thinking}) {
            if (*variant) { served.insert(backend_of_label((*variant)->label)); }
        }
    }
    served.erase(SpeculativeBackend::None);
    return std::any_of(subject.attached_drafters.begin(), subject.attached_drafters.end(),
                       [&](const spec::DrafterDecl& d) {
                           return served.contains(backend_of_drafter_kind(d.kind));
                       });
}

PlannedSpeculation plan_with_speculation(const PlanSourceFacts& facts,
                                         const runtime::PlanDeviceFacts& device,
                                         const EngineOptions& options,
                                         const SpeculationSubject* subject,
                                         const SpeculationAsk& ask,
                                         std::uint64_t drafter_bytes,
                                         const std::optional<models::DrafterStateShape>& drafter_state) {
    PlannedSpeculation out;
    SpeculationReport& r            = out.speculation;
    const std::uint32_t concurrency = std::max(1U, options.max_concurrency);
    const bool per_request          = per_request_of(ask, subject);
    const auto plan_of              = [&](const SpeculativeOptions& speculative) {
        EngineOptions planned     = options;
        planned.speculative       = speculative;
        runtime::PlanModelFacts m = facts.model;
        // A separate drafter (an artifact of its own: Gemma 4's assistant, gpt-oss's DFlash) is
        // uploaded beside the target: one named by the options, or, before a conversion names its
        // path (an explicit --spec with --with-drafters), one the drafter plan takes from its own
        // repository for this backend (attached, or embedded as a component).
        const bool attached =
            subject != nullptr &&
            std::any_of(subject->attached_drafters.begin(), subject->attached_drafters.end(),
                        [&](const spec::DrafterDecl& d) {
                            return backend_of_drafter_kind(d.kind) == speculative.backend;
                        });
        if (speculative.backend != SpeculativeBackend::None &&
            (!speculative.drafter.empty() || attached)) {
            m.drafter_bytes = drafter_bytes;
            // Its state from its config: the family planner sizes its KV and ring.
            if (!m.state.drafter && drafter_state &&
                (speculative.backend == SpeculativeBackend::DFlash ||
                 speculative.backend == SpeculativeBackend::DFlash2)) {
                m.state.drafter = drafter_state;
            }
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
            // The profile's depth for this backend, else the default (as GenerationService).
            const auto depth   = subject != nullptr ? profile_depth(*subject, given.backend, ask.thinking)
                                                    : std::nullopt;
            given.draft_tokens = depth.value_or(default_draft_tokens(given.backend));
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
            served = served_speculation(*subject, ask.thinking, per_request);
            if (subject->resolved) {
                for (const std::string& note : subject->resolved->notes) {
                    r.notes.push_back(note);
                }
            }
        }
        const auto with_given = [&](const ServedSpeculation& s) {
            return with_given_settings(s.engine, ask.given);
        };
        if (!served || !served->drafts()) {
            const SpeculativeOptions plain;
            runtime::DryRunPlan plan = plan_of(plain);
            // The profile's own choice, when it speculates: its drafter is absent here, or the
            // Engine does not run it (never "measured faster without speculation").
            const spec::Choice* wanted = nullptr;
            if (served && subject != nullptr && subject->resolved) {
                const spec::Choice& c = spec::pick(subject->resolved->profile, ask.thinking);
                if (c.config.enabled()) { wanted = &c; }
            }
            if (wanted != nullptr) {
                const std::string name =
                    wanted->drafter.empty() ? spec::mode_name(wanted->config.mode) : wanted->drafter;
                const bool absent = !subject->resolved->drafters.contains(name);
                r.source = served->source;
                r.basis  = absent ? "drafter_missing" : "engine_refuses";
                step("plain", plain, plan,
                     absent ? "the profile's " + wanted->config.label() + " needs the drafter '" + name +
                                  "', which is not attached (a GGUF model takes its separate "
                                  "drafters with --with-drafters)"
                            : "the profile's " + wanted->config.label() +
                                  " is not one this model's Engine runs");
            } else if (served && served->basis == "preliminary") {
                // A group's choice for a target nobody measured is plain decode: a fine-tune's
                // acceptance may fall short of the group's (spec::profile_for_group), not a
                // measurement of this target.
                r.source = served->source;
                r.basis  = "preliminary";
                step("plain", plain, plan,
                     "the group's choice for an unmeasured target: plain decode (a fine-tune may "
                     "accept less than the measured pairs)");
            } else if (served) {
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
            // The start's order when the choice does not fit (fit_fallbacks): the profile's fit
            // fallback, the deepest shallower draft, the model's own MTP head; each step planned
            // and recorded, the one the order accepts last served.
            if (!built) {
                std::optional<runtime::DryRunPlan> accepted;
                auto fallback = fit_fallbacks(
                    *subject, *served, per_request,
                    [&](const ServedSpeculation& candidate, FallbackStep at) {
                        const SpeculativeOptions sp = with_given(candidate);
                        runtime::DryRunPlan p       = plan_of(sp);
                        step(fallback_step_name(at), sp, p,
                             at == FallbackStep::Shallower ? std::string() : candidate.source);
                        const bool fits = p.fits;
                        if (fits) { accepted.emplace(std::move(p)); }
                        return fits;
                    });
                if (fallback && accepted) {
                    served = std::move(fallback);
                    serve(with_given(*served), std::move(*accepted));
                    built = true;
                }
            }
            if (!built) {
                served.reset();
                const SpeculativeOptions plain;
                runtime::DryRunPlan p = plan_of(plain);
                step("plain", plain, p, "nothing speculative fits: --spec auto decodes plainly");
                serve(plain, std::move(p));
            }
            // The pool rule (pool_rule): a drafter that leaves the automatic pool fewer whole
            // contexts than C while plain decode holds more gives way (to the MTP head when it
            // keeps four fifths of plain decode's contexts).
            if (served && built) {
                const std::uint32_t with_drafter = pool_contexts_of(out.plan, options);
                std::optional<runtime::DryRunPlan> plain_plan, head_plan;
                const PoolDecision pool = pool_rule(
                    *subject, per_request, with_drafter, concurrency,
                    [&] {
                        plain_plan = plan_of(SpeculativeOptions{});
                        return pool_contexts_of(*plain_plan, options);
                    },
                    [&](const ServedSpeculation& head) -> std::optional<std::uint32_t> {
                        const SpeculativeOptions sp = with_given(head);
                        head_plan                   = plan_of(sp);
                        step("pool_head", sp, *head_plan,
                             "the drafter leaves " + std::to_string(with_drafter) + " of " +
                                 std::to_string(concurrency) + " whole contexts");
                        if (!head_plan->fits) { return std::nullopt; }
                        return pool_contexts_of(*head_plan, options);
                    });
                if (pool.keep == PoolKeep::Head) {
                    for (auto& s : r.fallbacks) { s.served = false; }
                    served = pool.head;
                    serve(with_given(*served), std::move(*head_plan));
                } else if (pool.keep == PoolKeep::Plain) {
                    for (auto& s : r.fallbacks) { s.served = false; }
                    const SpeculativeOptions plain;
                    step("pool_plain", plain, *plain_plan,
                         "plain decode holds " + std::to_string(pool.plain) +
                             " whole contexts against " + std::to_string(with_drafter));
                    served.reset();
                    serve(plain, std::move(*plain_plan));
                }
            }
        }
    }

    // The profile's choices by the thinking flag, as this start serves them: only where the plan
    // serves the profile's own choice (no fit fallback, no pool step, not plain for an absent or
    // refused drafter), each flag's entry of the menu that step serves (one drafter loaded: an
    // alternate the drafter cannot run takes the primary, build_menu) and the gain the profile
    // measured for that very configuration (1 for a plain decode it chose).
    const bool serves_profile =
        std::any_of(r.fallbacks.begin(), r.fallbacks.end(), [](const SpeculationFallbackStep& s) {
            return s.served && s.step == "profile";
        });
    if (ask.automatic && serves_profile && served && subject != nullptr && subject->resolved &&
        subject->resolved->found()) {
        const spec::Profile profile = served_choices(subject->resolved->profile);
        const auto variant = [&](bool thinking) {
            const spec::MenuChoice& entry = served->menu.pick(thinking);
            const spec::Choice& c         = spec::pick(profile, thinking);
            SpeculationReport::ThinkingVariant v{
                .label = entry.config.enabled() ? entry.config.label() : std::string("plain")};
            if (c.config == entry.config && c.tok_s > 0 && c.plain_tok_s > 0) { v.speedup = c.speedup(); }
            return v;
        };
        r.with_thinking    = variant(true);
        r.without_thinking = variant(false);
        r.per_request      = per_request;
    }
    // The served choice's facts: drafter, depths, load.
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
            {"fallbacks", std::move(fallbacks)},
            {"notes", r.notes},
            {"by_thinking", r.with_thinking && r.without_thinking
                                ? Json{{"thinking", {{"config", r.with_thinking->label},
                                                     {"speedup", r.with_thinking->speedup}}},
                                       {"non_thinking", {{"config", r.without_thinking->label},
                                                         {"speedup", r.without_thinking->speedup}}},
                                       {"per_request", r.per_request}}
                                : Json(nullptr)}};
}

} // namespace ninferno::product
