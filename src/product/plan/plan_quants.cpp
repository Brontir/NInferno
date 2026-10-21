#include "product/plan/plan_quants.h"

#include "product/numeric_options.h"
#include "product/speculative_options.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace ninferno::product {

namespace {

using Json            = nlohmann::json;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

std::string grouped(std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return digits;
}

std::string gib(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof text, "%.2f GiB", static_cast<double>(bytes) / kGiB);
    return text;
}

double round_to(double x, double step) { return std::round(x / step) * step; }

std::string context_text(const EngineOptions& options) {
    return options.max_context_automatic ? std::string("auto") : grouped(options.max_context);
}

std::string class_text(const runtime::PlanDeviceFacts& device) {
    const std::uint64_t vram_class =
        device.total_bytes / runtime::kPlanVramClassGrainBytes * runtime::kPlanVramClassGrainBytes;
    char text[64];
    std::snprintf(text, sizeof text, "%.2f GiB class", static_cast<double>(vram_class) / kGiB);
    return text;
}

// Why a listed quantization is not planned ("" when it is).
std::string unsupported(const QuantListing& q, double bits_per_weight) {
    if (q.drafter) { return "a drafter or sidecar file, not a model"; }
    if (q.format == "safetensors") {
        return q.bits > 8 || q.bits == 0
                   ? "converted at start by its recipe: its size after conversion is not known "
                     "before"
                   : "";
    }
    if (q.bits > 8 || (q.bits == 0 && bits_per_weight > 9.0)) {
        return "above 8 bits per weight: the engine serves 8 bits or fewer";
    }
    return "";
}

std::string reserve_name(runtime::PlanReserve reserve) {
    return std::string(runtime::plan_reserve_name(reserve));
}

} // namespace

LaunchPlanOptions launch_options(const LaunchAsk& launch) {
    LaunchPlanOptions out;
    EngineOptions& o          = out.options;
    o.max_context             = launch.max_context;
    o.max_context_automatic   = launch.max_context == 0;
    o.max_concurrency         = launch.max_concurrency == 0 ? 1U : launch.max_concurrency;
    o.max_pending_requests    = 16;
    o.pending_timeout_ms      = 600000;
    o.prefill_chunk           = 1024;
    o.prefill_chunk_automatic = true;
    o.kv_lease_window         = 4096;
    // ninferno-ui's launch: exact takes a BF16 KV cache unless one is named (make_launch).
    const std::string kv        = launch.kv_dtype == "auto" ? std::string() : launch.kv_dtype;
    o.kv_cache_automatic        = kv.empty() && !launch.exact;
    o.kv_cache                  = o.kv_cache_automatic ? KvCacheStorage::BFloat16
                                  : kv.empty()         ? KvCacheStorage::BFloat16
                                                       : parse_kv_dtype(kv);
    o.recurrent_state           = RecurrentStateStorage::FP32;
    o.recurrent_state_automatic = true;
    o.activations               = launch.exact ? ActivationMode::Exact : kDefaultActivationMode;
    // ninferno-serve: an automatic context or several slots take an automatic pool, one slot at
    // an explicit context exactly its tokens.
    o.kv_capacity                       = o.max_context_automatic || o.max_concurrency > 1
                                              ? KvCapacityPolicy::automatic()
                                              : KvCapacityPolicy::explicit_capacity(o.max_context);
    o.context_cache                     = ContextCacheOptions{};
    o.context_cache.host_tier_automatic = true;
    SpeculationAsk& ask                 = out.ask;
    if (launch.spec.empty() || launch.spec == "auto") {
        ask.automatic = true;
    } else {
        ask.automatic          = false;
        ask.given.backend      = parse_speculative_backend(launch.spec);
        ask.depth_from_profile = ask.given.backend != SpeculativeBackend::None;
    }
    o.speculative = ask.given;
    return out;
}

PlannedSpeculation plan_model(const PlanSourceFacts& facts, const DryRunInputs& inputs) {
    return plan_with_speculation(facts, inputs.device, inputs.options, inputs.subject, inputs.ask,
                                 inputs.gpu, inputs.drafter_bytes);
}

QuantPlan plan_quants(const PlanSourceFacts& shape, const std::vector<QuantListing>& listing,
                      const DryRunInputs& inputs) {
    QuantPlan out;
    std::vector<runtime::QuantCandidate> candidates;
    std::vector<runtime::QuantFit> fits;
    for (const QuantListing& q : listing) {
        if (q.drafter) { continue; }
        QuantRow row;
        row.listing = q;
        row.basis   = q.format == "ninferno" ? runtime::WeightBasis::Artifact
                                             : runtime::WeightBasis::FileSize;
        // The shape's own file keeps the header's tensor bytes.
        if (shape.model.weight_bytes == q.bytes &&
            shape.model.weight_basis != runtime::WeightBasis::Unknown) {
            row.basis = shape.model.weight_basis;
        }
        row.candidate.label      = q.quant;
        row.candidate.bytes      = q.bytes;
        row.candidate.label_bits = q.bits;
        row.candidate.bits_per_weight =
            shape.model.params > 0 ? static_cast<double>(q.bytes) * 8.0 / shape.model.params : 0.0;
        row.why_unsupported     = unsupported(q, row.candidate.bits_per_weight);
        row.candidate.supported = row.why_unsupported.empty();
        if (row.candidate.supported) {
            const PlanSourceFacts facts = with_weights(shape, q.bytes, row.basis);
            row.planned                 = plan_model(facts, inputs);
            row.fit                     = runtime::quant_fit_of(row.planned.plan);
        }
        candidates.push_back(row.candidate);
        fits.push_back(row.fit);
        out.rows.push_back(std::move(row));
    }
    out.chosen = runtime::largest_fitting_quant(candidates, fits);
    if (out.chosen) { out.rows[*out.chosen].largest = true; }
    out.rule = quant_rule_line(out, inputs);
    return out;
}

std::string quant_rule_line(const QuantPlan& plan, const DryRunInputs& inputs) {
    const std::string ask =
        context_text(inputs.options) + " x " + std::to_string(inputs.options.max_concurrency);
    const std::string where = inputs.device.name + " (" + class_text(inputs.device) + ")";
    if (plan.chosen) {
        const QuantRow& row = plan.rows[*plan.chosen];
        std::string line    = row.listing.quant + ": the largest that fits " + ask + " on " + where;
        if (row.fit.shared) {
            line += ", one pool the " + std::to_string(inputs.options.max_concurrency) +
                    " streams share (no quantization holds a whole context for each)";
        } else if (inputs.options.max_context_automatic) {
            line += " (context " + grouped(row.fit.context) + ")";
        }
        return line;
    }
    // Nothing fits: the smallest supported quantization's reason and what fits instead.
    const QuantRow* smallest = nullptr;
    for (const QuantRow& row : plan.rows) {
        if (row.candidate.supported &&
            (smallest == nullptr || row.candidate.bytes < smallest->candidate.bytes)) {
            smallest = &row;
        }
    }
    if (smallest == nullptr) {
        return "no quantization of this repository is served (" + ask + ")";
    }
    std::string line = "no quantization fits " + ask + " on " + where + "; the smallest, " +
                       smallest->listing.quant + ": ";
    const runtime::DryRunPlan& p = smallest->planned.plan;
    if (p.fits) {
        line += p.private_contexts() ? "fits only below the floor quality (a survival rung)"
                                     : "fits only as a shared pool";
    } else {
        line += p.reason;
    }
    for (const ConcurrencyFit& fit : p.fits_instead) {
        line += "; --max-concurrency " + std::to_string(fit.max_concurrency) + " fits " +
                grouped(fit.max_context) + " tokens";
    }
    return line;
}

Json plan_json(const runtime::DryRunPlan& plan, bool ladder) {
    const auto bytes_json = [](const runtime::PlanRungBytes& b) {
        return Json{{"kv", b.kv},         {"states", b.states},
                    {"rings", b.rings},   {"workspace", b.workspace},
                    {"graphs", b.graphs}, {"fixed", b.fixed},
                    {"total", b.total()}};
    };
    const auto rung_json = [&](const runtime::DryRunRung& r) {
        return Json{{"context", r.rung.max_context},
                    {"kv_tokens",
                     static_cast<std::uint64_t>(r.rung.kv_pages) * runtime::kAutomaticContextGrain},
                    {"cached_states", r.rung.cached_states},
                    {"prefill_chunk", r.rung.prefill_chunk},
                    {"step", std::string(runtime::plan_step_name(r.rung.step))},
                    {"survival", r.rung.survival},
                    {"bytes", bytes_json(r.bytes)},
                    {"needed_bytes", r.needed}};
    };
    Json out     = {{"planned", plan.planned},
                    {"fits", plan.fits},
                    {"estimate", true},
                    {"kv_dtype", std::string(kv_cache_storage_name(plan.kv_cache))},
                    {"state_dtype", plan.recurrent == RecurrentStateStorage::FP32   ? "fp32"
                                    : plan.recurrent == RecurrentStateStorage::BF16 ? "bf16"
                                                                                    : "f16"},
                    {"rungs", plan.ladder.size()},
                    {"weights_bytes", plan.weight_bytes},
                    {"free_bytes", plan.free_bytes},
                    {"free_after_weights_bytes", plan.free_after_weights},
                    {"class_reserve_bytes", plan.axes.class_reserve},
                    {"base_margin_bytes", plan.margins.base},
                    {"allowance_bytes", plan.margins.allowance},
                    {"context_ceiling", plan.axes.context_ceiling},
                    {"reason", plan.reason}};
    Json instead = Json::array();
    for (const ConcurrencyFit& fit : plan.fits_instead) {
        instead.push_back({{"concurrency", fit.max_concurrency}, {"context", fit.max_context}});
    }
    out["fits_instead"] = std::move(instead);
    if (const runtime::DryRunRung* chosen = plan.chosen()) {
        out["rung"]             = plan.rung + 1;
        out["reserve"]          = reserve_name(plan.reserve);
        out["reserve_kept"]     = plan.reserve == runtime::PlanReserve::Kept;
        out["chosen"]           = rung_json(*chosen);
        out["context"]          = chosen->rung.max_context;
        out["pool_contexts"]    = round_to(plan.pool_contexts(), 0.1);
        out["private_contexts"] = plan.private_contexts();
        out["description"]      = runtime::describe_plan_rung(chosen->rung, plan.axes.concurrency);
    } else {
        out["rung"]             = 0;
        out["reserve"]          = "";
        out["reserve_kept"]     = false;
        out["chosen"]           = nullptr;
        out["context"]          = 0;
        out["pool_contexts"]    = 0;
        out["private_contexts"] = false;
        out["description"]      = "";
    }
    if (ladder) {
        Json rungs = Json::array();
        for (const runtime::DryRunRung& r : plan.ladder) { rungs.push_back(rung_json(r)); }
        out["ladder"] = std::move(rungs);
    }
    return out;
}

Json planned_json(const PlannedSpeculation& planned, bool ladder) {
    Json out           = plan_json(planned.plan, ladder);
    out["speculation"] = to_json(planned.speculation);
    return out;
}

Json device_json(const runtime::PlanDeviceFacts& device, const runtime::DryRunPlan& plan) {
    return {{"name", device.name},
            {"source", device.source},
            {"total_bytes", device.total_bytes},
            {"class_bytes", device.total_bytes / runtime::kPlanVramClassGrainBytes *
                                runtime::kPlanVramClassGrainBytes},
            {"compute_capability", device.compute_capability},
            {"pci_device_id", device.pci_device_id},
            {"wddm", device.paravirtual},
            {"display", device.drives_display},
            {"integrated", device.integrated},
            {"free_basis", device.free_bytes ? "measured_now" : "idle_card"},
            {"free_bytes", plan.free_bytes},
            {"class_reserve_bytes",
             runtime::plan_class_reserve(device.total_bytes, device.paravirtual,
                                         device.drives_display, device.integrated)}};
}

Json model_json(const PlanSourceFacts& facts, const std::string& source, const std::string& kind) {
    const runtime::PlanModelFacts& m = facts.model;
    return {{"source", source},
            {"kind", kind},
            {"architecture", m.architecture},
            {"family", m.model_type},
            {"native_context", m.native_context},
            {"params", m.params},
            {"shape_known", m.known()},
            {"weights_bytes", m.weight_bytes},
            {"weights_basis", std::string(runtime::weight_basis_name(m.weight_basis))},
            {"kv_bytes_per_token",
             {{"bf16", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::BFloat16)},
              {"int8", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Int8Group64)},
              {"fp8", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Fp8E4M3Row256)},
              {"nvfp4", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Nvfp4Group16)},
              {"k8v4", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Fp8KeyNvfp4Value)}}},
            {"window", m.window},
            {"builtin_mtp", facts.shape.builtin_mtp}};
}

Json ask_json(const EngineOptions& options, const SpeculationAsk& ask, bool kv_automatic) {
    return {{"context", options.max_context_automatic ? Json("auto") : Json(options.max_context)},
            {"concurrency", options.max_concurrency},
            {"kv_dtype", kv_automatic ? std::string("auto")
                                      : std::string(kv_cache_storage_name(options.kv_cache))},
            {"spec", ask.automatic ? std::string("auto")
                     : ask.given.backend == SpeculativeBackend::None
                         ? std::string("none")
                         : std::string(speculative_backend_name(ask.given.backend)) + ":" +
                               std::to_string(ask.given.draft_tokens)}};
}

Json quant_plan_json(const QuantPlan& plan) {
    Json rows = Json::array();
    for (const QuantRow& row : plan.rows) {
        Json item    = {{"quant", row.listing.quant},
                        {"format", row.listing.format},
                        {"bits", row.listing.bits},
                        {"bits_per_weight", round_to(row.candidate.bits_per_weight, 0.01)},
                        {"weights_bytes", row.listing.bytes},
                        {"weights_basis", std::string(runtime::weight_basis_name(row.basis))},
                        {"supported", row.candidate.supported},
                        {"why_unsupported", row.why_unsupported},
                        {"fits", row.fit.fits},
                        {"fits_shared", row.fit.shared},
                        {"largest", row.largest}};
        item["plan"] = row.candidate.supported ? planned_json(row.planned, false) : Json(nullptr);
        rows.push_back(std::move(item));
    }
    return {{"quants", std::move(rows)},
            {"chosen", plan.chosen ? Json(plan.rows[*plan.chosen].listing.quant) : Json(nullptr)},
            {"rule", plan.rule}};
}

Json dry_plan_document(const PlanSourceFacts& facts, const std::string& source,
                       const std::string& kind, const DryRunInputs& inputs,
                       const PlannedSpeculation& planned, const QuantPlan* quants) {
    Json out = {{"schema", "ninferno-dry-plan/1"},
                {"estimate", true},
                {"model", model_json(facts, source, kind)},
                {"device", device_json(inputs.device, planned.plan)},
                {"ask", ask_json(inputs.options, inputs.ask, inputs.options.kv_cache_automatic)},
                {"plan", planned_json(planned, true)}};
    if (quants != nullptr) {
        Json q        = quant_plan_json(*quants);
        out["quants"] = std::move(q["quants"]);
        out["chosen"] = std::move(q["chosen"]);
        out["rule"]   = std::move(q["rule"]);
    } else {
        out["quants"] = Json::array();
        out["chosen"] = nullptr;
        out["rule"]   = "";
    }
    return out;
}

std::string render_dry_plan_text(const PlanSourceFacts& facts, const std::string& source,
                                 const DryRunInputs& inputs, const PlannedSpeculation& planned,
                                 const QuantPlan* quants) {
    std::ostringstream out;
    const runtime::DryRunPlan& plan        = planned.plan;
    const runtime::PlanDeviceFacts& device = inputs.device;
    out << "plan without the weights (an estimate; the start is the authority)\n";
    out << "model  | " << source << " (" << facts.model.architecture << ", family "
        << (facts.model.model_type.empty() ? std::string("none") : facts.model.model_type)
        << ", own context " << grouped(facts.model.native_context) << ")\n";
    out << "device | " << device.name << " (" << device.source << ", " << class_text(device)
        << (device.paravirtual ? ", WDDM" : "") << (device.drives_display ? ", display" : "")
        << "), " << (device.free_bytes ? "free now " : "idle card ") << gib(plan.free_bytes)
        << "\n";
    out << "ask    | context " << context_text(inputs.options) << " x "
        << inputs.options.max_concurrency << " streams, KV "
        << (inputs.options.kv_cache_automatic ? "auto"
                                              : kv_cache_storage_name(inputs.options.kv_cache))
        << " (" << kv_cache_storage_name(plan.kv_cache) << "), spec "
        << ask_json(inputs.options, inputs.ask, false)["spec"].get<std::string>() << "\n";
    if (quants != nullptr) {
        out << "quants | (largest that fits first; * the rule's pick)\n";
        std::vector<const QuantRow*> order;
        for (const QuantRow& row : quants->rows) { order.push_back(&row); }
        std::stable_sort(order.begin(), order.end(), [](const QuantRow* a, const QuantRow* b) {
            return runtime::quant_ranks_above(a->candidate, b->candidate);
        });
        for (const QuantRow* row : order) {
            char line[256];
            std::snprintf(line, sizeof line, "  %c %-14s %5.2f bpw %9s  ", row->largest ? '*' : ' ',
                          row->listing.quant.c_str(), row->candidate.bits_per_weight,
                          gib(row->listing.bytes).c_str());
            out << line;
            if (!row->candidate.supported) {
                out << "not served: " << row->why_unsupported;
            } else if (row->fit.fits) {
                out << "fits, context " << grouped(row->fit.context);
            } else if (row->fit.shared) {
                out << "fits as a shared pool, context " << grouped(row->fit.context);
            } else if (row->planned.plan.fits) {
                out << "survival rung only";
            } else {
                out << "does not fit";
            }
            out << " (" << runtime::weight_basis_name(row->basis) << ")\n";
        }
        out << "quant  | " << quants->rule << "\n";
    }
    if (!plan.planned) {
        out << "plan   | " << plan.reason << "\n";
        return out.str();
    }
    for (std::size_t i = 0; i < plan.ladder.size(); ++i) {
        const runtime::DryRunRung& r = plan.ladder[i];
        out << (plan.fits && i == plan.rung ? "  > " : "    ") << "rung " << (i + 1) << " "
            << runtime::plan_step_name(r.rung.step) << (r.rung.survival ? " (survival)" : "")
            << ": " << runtime::describe_plan_rung(r.rung, plan.axes.concurrency) << "; "
            << gib(r.bytes.total()) << ", needs " << gib(r.needed) << "\n";
    }
    if (plan.fits) {
        out << "plan   | rung " << (plan.rung + 1) << " of " << plan.ladder.size() << " ("
            << reserve_name(plan.reserve) << "): "
            << runtime::describe_plan_rung(plan.ladder[plan.rung].rung, plan.axes.concurrency)
            << (plan.private_contexts() ? "" : "; the streams share one pool") << "\n";
    } else {
        out << "plan   | does not fit: " << plan.reason << "\n";
        for (const ConcurrencyFit& fit : plan.fits_instead) {
            out << "         --max-concurrency " << fit.max_concurrency << " fits "
                << grouped(fit.max_context) << " tokens\n";
        }
    }
    const SpeculationReport& s = planned.speculation;
    out << "spec   | " << s.ask << " -> " << s.label;
    if (!s.source.empty()) { out << " (" << s.basis << ": " << s.source << ")"; }
    out << "\n";
    if (s.engine.backend != SpeculativeBackend::None) {
        out << "         load " << spec::load_label(s.load) << ", at "
            << inputs.options.max_concurrency << " streams depth " << s.depth_at_concurrency
            << "; modeled load " << spec::load_label(s.modeled_load)
            << (s.model_note.empty() ? "" : " (" + s.model_note + ")") << "\n";
        for (const SpeculationClassRow& row : s.classes) {
            out << "         " << spec::class_name(row.context_class) << ": depth " << row.depth
                << " (" << row.basis << ": " << row.reason << ")\n";
        }
    }
    for (const SpeculationFallbackStep& f : s.fallbacks) {
        out << "         " << (f.served ? "> " : "  ") << f.step << " " << f.label << ": "
            << (f.fits ? "fits" : "does not fit") << (f.note.empty() ? "" : " (" + f.note + ")")
            << "\n";
    }
    return out.str();
}

} // namespace ninferno::product
