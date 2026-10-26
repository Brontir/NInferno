#include "serve/plan_offline.h"

#include "convert/acquire.h"
#include "convert/cache.h"
#include "convert/drafters.h"
#include "convert/fetch.h"
#include "convert/gguf.h"
#include "convert/support.h"
#include "product/plan/plan_choice.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "serve/generation_service.h"
#include "spec/fingerprint.h"
#include "spec/pairs.h"
#include "spec/gpu_identity.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>

namespace ninferno::serve {

namespace {

namespace fs = std::filesystem;
using Json   = nlohmann::json;

// What a model is before its weights: its facts at one quantization (the shape every one shares)
// and, for a hub repository, every quantization's listing.
struct ModelRead {
    product::PlanSourceFacts facts;
    std::vector<product::QuantListing> listing; // empty for a single file or artifact
    std::string source, kind;                   // kind: hub | gguf | checkpoint | artifact
    std::string repo;                           // a hub id
    bool gguf = false;
    fs::path artifact; // a .ninferno file
    std::string quant; // the quantization the facts were read at ("" for a single model)
    // The weights the facts were read at (spec::weights_identity of their files' digests): the key
    // of the lab's speculation pairs. "" when the listing gives no digest for one of them.
    std::string weights;
    std::string commit; // the commit the hub listing was read at ("" unknown)
};

std::string weights_of(const convert::RepoQuant& q) { return spec::weights_identity(q.sha256); }

std::string ascii_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// A GGUF header from the first bytes of a file, read longer until its metadata fits.
convert::GgufFile read_local_head(const fs::path& path) {
    const std::uint64_t size = fs::file_size(path);
    for (std::uint64_t want : {8ULL << 20, 64ULL << 20, 512ULL << 20}) {
        const std::uint64_t n = std::min(want, size);
        std::string bytes(static_cast<std::size_t>(n), '\0');
        std::ifstream in(path, std::ios::binary);
        in.read(bytes.data(), static_cast<std::streamsize>(n));
        try {
            return convert::parse_gguf_head(reinterpret_cast<const std::byte*>(bytes.data()),
                                            bytes.size(), size, path.string());
        } catch (const convert::GgufTruncated&) {
            if (n == size) { throw; }
        }
    }
    throw std::runtime_error(path.string() + ": the GGUF metadata runs past 512 MiB");
}

// Every part of a split GGUF beside `path` ("x-00001-of-00003.gguf"), or the file alone.
std::uint64_t local_gguf_bytes(const fs::path& path) {
    static const std::regex split(R"((.*)-\d{5}-of-(\d{5})\.gguf)", std::regex::icase);
    std::smatch match;
    const std::string name = path.filename().string();
    if (!std::regex_match(name, match, split)) { return fs::file_size(path); }
    std::uint64_t total = 0;
    for (const auto& entry : fs::directory_iterator(path.parent_path())) {
        std::smatch other;
        const std::string n = entry.path().filename().string();
        if (std::regex_match(n, other, split) && other[1] == match[1]) {
            total += entry.file_size();
        }
    }
    return total;
}

std::uint64_t safetensors_bytes(const fs::path& dir) {
    std::uint64_t total = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (ascii_lower(entry.path().extension().string()) == ".safetensors" &&
            entry.path().filename().string().find('/') == std::string::npos) {
            total += entry.file_size();
        }
    }
    return total;
}

convert::FetchOptions fetch_options() {
    convert::FetchOptions fetch;
    return fetch;
}

// The GGUF quantizations of a hub repository already in the local hub cache, each with its file:
// a downloaded GGUF (the first part of a split one), or the artifact converted from one
// (<snapshot>/.ninferno-cache/<gguf stem>.<recipe>.ninferno) where the download was removed.
std::map<std::string, fs::path> cached_quants(const std::string& repo,
                                              const std::string& revision) {
    std::map<std::string, fs::path> out;
    std::error_code error;
    // The requested revision's snapshot alone (a commit, or the commit its ref names: "main" when
    // none was asked), never another revision's files.
    const fs::path repo_dir = convert::default_hub_cache() / convert::hub_repo_directory(repo);
    const std::string rev   = revision.empty() ? std::string("main") : revision;
    std::string commit;
    const bool hex = rev.size() == 40 && std::all_of(rev.begin(), rev.end(), [](char c) {
                         return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                     });
    if (hex) {
        commit = rev;
    } else {
        std::ifstream ref(repo_dir / "refs" / rev);
        std::getline(ref, commit);
    }
    if (commit.empty()) { return out; }
    const fs::path root = repo_dir / "snapshots" / commit;
    if (!fs::is_directory(root, error)) { return out; }
    for (auto it = fs::recursive_directory_iterator(root, error);
         !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
        const std::string name  = it->path().filename().string();
        const std::string lower = ascii_lower(name);
        std::string gguf;
        bool artifact = false;
        if (lower.ends_with(".gguf")) {
            static const std::regex later(
                R"(.*-0000[2-9]-of-\d{5}\.gguf|.*-000[1-9]\d-of-\d{5}\.gguf)", std::regex::icase);
            if (std::regex_match(name, later)) { continue; } // a later part of a split file
            gguf = name;
        } else if (lower.ends_with(".ninferno") &&
                   it->path().parent_path().filename() == ".ninferno-cache") {
            // <stem>.<recipe>.ninferno: the stem is the GGUF's.
            const std::string stem = it->path().stem().string(); // <stem>.<recipe>
            const auto dot         = stem.rfind('.');
            if (dot == std::string::npos) { continue; }
            gguf     = stem.substr(0, dot) + ".gguf";
            artifact = true;
        } else {
            continue;
        }
        if (ascii_lower(gguf).starts_with("mmproj") || ascii_lower(gguf).find("mtp") == 0) {
            continue;
        }
        if (const std::string label = convert::gguf_quant_label(gguf); !label.empty()) {
            // A downloaded file before its conversion: the header is the GGUF's.
            if (!artifact || !out.contains(label)) { out[label] = it->path(); }
        }
    }
    return out;
}

// The bytes of the vision projector the start fetches beside a GGUF model under --vision (the same
// choice: convert::best_projector, every part of a split one); 0 when there is none.
std::uint64_t listed_projector_bytes(const std::vector<convert::RepoQuant>& quants) {
    std::vector<std::string> names;
    for (const auto& q : quants) { names.insert(names.end(), q.files.begin(), q.files.end()); }
    const auto parts = convert::best_projector(names);
    std::uint64_t bytes = 0;
    for (const auto& q : quants) {
        // A projector file is a listing entry of its own (a sidecar), its bytes its own.
        if (q.files.size() == 1 &&
            std::find(parts.begin(), parts.end(), q.files.front()) != parts.end()) {
            bytes += q.bytes;
        }
    }
    return bytes;
}

// A hub repository: its listing and the facts at `quant` (the smallest GGUF when none is named).
// With `vision`, a GGUF model's facts count the projector the start fetches beside it.
ModelRead read_hub(const std::string& spec, const std::string& quant, bool vision,
                   bool template_given = false) {
    ModelRead out;
    out.kind                          = "hub";
    out.source                        = spec;
    convert::FetchOptions fetch       = fetch_options();
    // A model the engine does not run is not planned: the converter's own rules, from what the
    // hub shows before a download (convert/support.h). A hub not asked or not answering leaves
    // the listing below to fail or plan.
    {
        auto ask  = fetch;
        ask.quant = quant;
        std::optional<convert::SupportFacts> facts;
        try {
            facts = convert::hub_support_facts(spec, ask);
        } catch (const std::exception&) {}
        if (facts) {
            if (template_given) { facts->chat_template = true; } // --chat-template
            if (const auto refusal = convert::serve_refusal(convert::model_support(*facts));
                !refusal.empty()) {
                throw std::runtime_error(spec + " cannot be run: " + refusal);
            }
        }
    }
    const auto quants                 = convert::list_quants(spec, fetch);
    out.repo                          = convert::split_spec(spec, fetch).first;
    out.commit                        = quants.empty() ? std::string() : quants.front().commit;
    std::optional<convert::RepoQuant> named; // the one --quant names (listed_quant_for)
    const convert::RepoQuant* header  = nullptr;
    const convert::RepoQuant* weights = nullptr;
    const convert::RepoQuant* native  = nullptr;
    for (const auto& q : quants) {
        out.listing.push_back({.quant   = q.quant,
                               .format  = q.format,
                               .bytes   = q.bytes,
                               .bits    = q.bits,
                               .drafter = q.drafter,
                               .weights = weights_of(q)});
        if (q.drafter) { continue; }
        if (q.format == "gguf") {
            if (quant.empty() && (header == nullptr || q.bytes < header->bytes)) { header = &q; }
        } else if (q.format == "safetensors") {
            weights = &q;
        } else if (q.format == "ninferno") {
            native = &q;
        }
    }
    // --quant: the GGUF model the fetch takes for it (convert::listed_quant_for: "Q4_K_M" takes
    // UD-Q4_K_M where that is the one, "4bit" the largest 4-bit model), throwing as the fetch does.
    if (!quant.empty() &&
        std::any_of(quants.begin(), quants.end(),
                    [](const auto& q) { return !q.drafter && q.format == "gguf"; })) {
        named  = convert::listed_quant_for(quants, quant, out.repo);
        header = &*named;
    }
    // No --quant: a revision with top-level safetensors weights is fetched as them, whatever GGUF
    // files it also holds (convert/fetch.cpp choose_files), so the plan is theirs.
    if (quant.empty() && weights != nullptr) {
        header = nullptr;
        native = nullptr; // and not as a published .ninferno beside them
    }
    if (header != nullptr) {
        out.gguf    = true;
        out.quant   = header->quant;
        out.weights = weights_of(*header);
        fetch.quant = header->quant;
        for (const std::uint64_t bytes : {8ULL << 20, 32ULL << 20, 128ULL << 20}) {
            try {
                const auto head = convert::fetch_gguf_head(spec, fetch, bytes);
                const auto file =
                    convert::parse_gguf_head(reinterpret_cast<const std::byte*>(head.bytes.data()),
                                             head.bytes.size(), head.size, head.path);
                out.facts = product::plan_facts_of_gguf(file, header->bytes);
                if (vision) { out.facts.model.vision_bytes = listed_projector_bytes(quants); }
                return out;
            } catch (const convert::GgufTruncated&) { continue; }
        }
        throw std::runtime_error(spec + ": the GGUF metadata of " + header->quant +
                                 " runs past 128 MiB");
    }
    if (!quant.empty()) {
        throw std::runtime_error(spec + ": no GGUF quantization " + quant + " in the listing");
    }
    if (native != nullptr) {
        // The engine's own artifact: its manifest names the base model's config.
        const Json manifest    = convert::fetch_repo_json(spec, fetch, "artifact-manifest.json");
        const std::string base = manifest.value("source", Json::object())
                                     .value("base", Json::object())
                                     .value("repo_id", out.repo);
        out.facts = product::plan_facts_of_config(convert::fetch_config(base, fetch), native->bytes,
                                                  runtime::WeightBasis::FileSize);
        out.quant   = native->quant;
        out.weights = weights_of(*native);
        out.listing.clear();
        return out;
    }
    if (weights != nullptr) {
        const Json config = convert::fetch_config(spec, fetch);
        out.facts         = product::plan_facts_of_config(config, weights->bytes,
                                                          runtime::WeightBasis::FileSize);
        out.facts.model.mtp_bytes = weights->mtp_bytes;
        out.facts.shape.base_models = convert::hub_base_models(spec, fetch);
        out.weights               = weights_of(*weights);
        // MTP layers are what the weights hold, declared or not: a part of their own
        // (model_mtp.safetensors), else what the weight index lists, as the converter finds them
        // (convert::sharded_checkpoint_mtp) whatever the number of weight files; no index to
        // read: the declaration.
        if (weights->mtp_bytes > 0) {
            out.facts.shape.builtin_mtp = true;
        } else {
            Json index;
            try {
                index = convert::fetch_repo_json(spec, fetch, "model.safetensors.index.json");
            } catch (const std::exception&) {}
            if (index.is_object()) {
                out.facts.shape.builtin_mtp = convert::sharded_checkpoint_mtp(config, index);
            }
        }
        out.listing.clear();
        return out;
    }
    throw std::runtime_error(spec + " holds neither GGUF models nor safetensors weights");
}

ModelRead read_model(const ServeOptions& options) {
    const std::string& name = options.model_source;
    std::error_code error;
    ModelRead out;
    out.source = name;
    if (fs::exists(name, error)) {
        const fs::path path(name);
        // A checkpoint or GGUF file the engine does not run is not planned (convert/support.h).
        if (convert::is_convertible(name)) {
            auto facts = convert::local_support_facts(path);
            if (!options.chat_template_path.empty()) { facts.chat_template = true; } // --chat-template
            if (const auto refusal = convert::serve_refusal(convert::model_support(facts));
                !refusal.empty()) {
                throw std::runtime_error(name + " cannot be run: " + refusal);
            }
        }
        if (fs::is_directory(path)) {
            if (!fs::exists(path / "config.json")) {
                throw std::runtime_error(name + " has no config.json to plan from");
            }
            std::ifstream in(path / "config.json");
            const Json config = Json::parse(in);
            out.kind          = "checkpoint";
            out.facts         = product::plan_facts_of_config(config, safetensors_bytes(path),
                                                              runtime::WeightBasis::FileSize);
            // What the converter finds (convert::checkpoint_mtp): the weights' own MTP part, the
            // index, or the one weight file's header; else the declaration.
            out.facts.shape.builtin_mtp = convert::checkpoint_mtp(path, config);
            out.facts.shape.base_models = convert::readme_base_models(path / "README.md");
            return out;
        }
        if (ascii_lower(path.extension().string()) == ".gguf") {
            out.kind  = "gguf";
            out.gguf  = true;
            out.facts = product::plan_facts_of_gguf(read_local_head(path), local_gguf_bytes(path));
            return out;
        }
        out.kind     = "artifact";
        out.artifact = path;
        out.facts    = product::plan_facts_of_artifact(path);
        return out;
    }
    if (!convert::is_repository_spec(name)) {
        throw std::runtime_error("'" + name + "' is neither a file or directory nor a hub id");
    }
    return read_hub(name, options.quant, options.enable_vision,
                    !options.chat_template_path.empty());
}

// The quantization label a plan reads a bit class from (spec::label_bits): a hub listing's, or a
// local GGUF's file name ("Qwen3.5-9B-UD-IQ2_M.gguf"), which names it the same way.
std::string quant_label_of(const ModelRead& model) {
    if (!model.quant.empty() || model.kind != "gguf") { return model.quant; }
    return fs::path(model.source).stem().string();
}

// The axes the user pinned with flags (the configuration choice picks the others).
product::ChoicePins pins_of(const ServeOptions& options) {
    return {.kv          = options.kv_cache.has_value(),
            .activations = options.activations_given,
            .state       = options.recurrent_state.has_value(),
            .ring        = options.sliding_kv_given,
            .chunk       = options.prefill_chunk_explicit};
}

struct Planning {
    product::DryRunInputs inputs;
    std::optional<product::SpeculationSubject> subject;
    // Each quantization's own subject (inputs.subject_of), made when first asked.
    std::shared_ptr<std::map<std::string, product::SpeculationSubject>> per_quant;
};

// The inputs a server's options give for a model read without its weights.
Planning planning_of(const ServeOptions& options, const ModelRead& model,
                     const runtime::PlanDeviceFacts& device, const spec::GpuKey& gpu) {
    Planning out;
    out.inputs.device  = device;
    out.inputs.gpu     = gpu;
    out.inputs.options = engine_options_of(options);
    out.inputs.pins    = pins_of(options);
    out.inputs.ask     = product::SpeculationAsk{.automatic   = options.speculation_from_profile,
                                                 .given       = options.speculative,
                                                 .thinking    = options.enable_thinking.value_or(true),
                                                 .per_request = options.per_request_speculation,
                                                 .per_request_from_profile =
                                                     options.speculation_from_profile &&
                                                     !options.per_request_given,
                                                 .depth_from_profile = options.draft_tokens_from_model};
    if (!model.artifact.empty()) {
        out.subject = product::speculation_subject(model.artifact, gpu, options.spec_profile);
    } else {
        out.subject =
            product::planned_subject(product::PlannedTarget{.repo          = model.repo,
                                                            .facts         = model.facts,
                                                            .gguf          = model.gguf,
                                                            .with_drafters = options.with_drafters,
                                                            .weights       = model.weights,
                                                            .quant         = quant_label_of(model),
                                                            .commit        = model.commit},
                                     gpu);
        // A separate drafter attached with --with-drafters is uploaded beside the target.
        const convert::FetchOptions fetch = fetch_options();
        const product::AttachedDrafterMemory drafter = product::attached_drafter_memory(
            *out.subject, model.listing, model.facts.model.model_type, model.facts.model.hidden, &fetch);
        out.inputs.drafter_bytes = drafter.bytes;
        out.inputs.drafter_state = drafter.state;
        // A repository's quantizations each have their weights, and the lab's pairs are of
        // weights: every row of the table resolves its own (the subject above is the one read).
        if (model.listing.size() > 1) {
            out.per_quant = std::make_shared<std::map<std::string, product::SpeculationSubject>>();
            out.inputs.subject_of = [per_quant = out.per_quant, repo = model.repo,
                                     commit = model.commit, facts = model.facts, gguf = model.gguf,
                                     with_drafters = options.with_drafters,
                                     gpu](const product::QuantListing& q)
                -> const product::SpeculationSubject* {
                if (q.drafter) { return nullptr; }
                auto it = per_quant->find(q.quant);
                if (it == per_quant->end()) {
                    it = per_quant
                             ->emplace(q.quant,
                                       product::planned_subject(
                                           product::PlannedTarget{.repo          = repo,
                                                                  .facts         = facts,
                                                                  .gguf          = gguf,
                                                                  .with_drafters = with_drafters,
                                                                  .weights       = q.weights,
                                                                  .quant         = q.quant,
                                                                  .commit        = commit},
                                           gpu))
                             .first;
                }
                return &it->second;
            };
        }
    }
    return out;
}

} // namespace

OfflinePlan plan_without_load(const ServeOptions& options) {
    runtime::PlanDeviceFacts device;
    spec::GpuKey gpu;
    if (!options.plan_device.empty()) {
        std::string error;
        const auto board = product::plan_device_of_board(options.plan_device, &error);
        if (!board) { throw std::runtime_error("--plan-device: " + error); }
        device = *board;
        gpu    = product::gpu_key_of(device);
    } else {
        try {
            device = product::plan_device_of_cuda(
                options.device, true, runtime::plan_automatic_fit(engine_options_of(options)));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("no GPU to plan for (") + e.what() +
                                     "): name a board with --plan-device");
        }
        gpu = spec::gpu_identity(options.device).key;
    }
    const ModelRead model = read_model(options);
    // A quantization is chosen on an idle card of this GPU (as the fetch without --quant does:
    // the same card type, the same flags, the same file); the free memory now is reported beside.
    const bool table = model.kind == "hub" && model.gguf && options.quant.empty();
    const std::optional<std::uint64_t> free_now = device.free_bytes;
    if (table) { device.free_bytes.reset(); }
    Planning planning       = planning_of(options, model, device, gpu);
    planning.inputs.subject = planning.subject ? &*planning.subject : nullptr;
    OfflinePlan out;
    std::optional<product::QuantPlan> quants;
    product::PlannedSpeculation planned;
    product::PlanSourceFacts facts = model.facts;
    // A GGUF repository without --quant: every quantization and the largest that fits.
    if (model.kind == "hub" && model.gguf && options.quant.empty()) {
        quants = product::plan_quants(model.facts, model.listing, planning.inputs);
        const product::QuantRow* chosen = quants->chosen ? &quants->rows[*quants->chosen] : nullptr;
        if (chosen == nullptr) {
            // Nothing fits: the smallest supported one's plan says why.
            for (const auto& row : quants->rows) {
                if (row.candidate.supported &&
                    (chosen == nullptr || row.candidate.bytes < chosen->candidate.bytes)) {
                    chosen = &row;
                }
            }
        }
        if (chosen != nullptr) {
            planned = chosen->planned;
            facts   = product::with_weights(model.facts, chosen->weight_bytes, chosen->basis);
        } else {
            planned = product::plan_model(model.facts, planning.inputs);
        }
    } else {
        planned = product::plan_model(model.facts, planning.inputs);
    }
    // The configuration choice at the chosen weights: what the start would pick for the axes left
    // automatic, and what every other value would change.
    std::string label = model.quant;
    if (quants && quants->chosen) { label = quants->rows[*quants->chosen].listing.quant; }
    const product::DryRunInputs at_chosen =
        quants && quants->chosen
            ? product::inputs_for(planning.inputs, quants->rows[*quants->chosen].listing)
            : planning.inputs;
    const auto served = product::choose_at_weights(facts, at_chosen, pins_of(options), label);
    // A start serves the choice's drafter (product::StartAxes::drafter): the plan shown is of it.
    if (served) { planned = product::plan_at_choice(facts, at_chosen, *served, pins_of(options), planned); }
    const std::string source =
        model.source + (model.quant.empty() || quants ? std::string() : " " + model.quant);
    out.json = product::dry_plan_document(facts, source, model.kind, planning.inputs, planned,
                                          quants ? &*quants : nullptr);
    out.text = product::render_dry_plan_text(facts, source, planning.inputs, planned,
                                             quants ? &*quants : nullptr);
    out.fits = planned.plan.fits;
    if (!out.fits) { out.reason = planned.plan.reason; }
    if (served) {
        out.json["choice"] = product::choice_json(*served);
        out.text += product::choice_line(*served) + "\n";
    }
    // The free memory now (another process may hold some): would the pick start right now?
    if (table && free_now && quants && quants->pick) {
        const product::QuantRow& row             = quants->rows[quants->pick->index];
        product::DryRunInputs now                = product::inputs_for(planning.inputs, row.listing);
        now.device.free_bytes                    = free_now;
        const product::PlanSourceFacts at_weights =
            product::with_weights(model.facts, row.weight_bytes, row.basis);
        product::PlannedSpeculation at_now = product::plan_model(at_weights, now);
        // The choice is made at the chosen row: there, the drafter it serves.
        if (served && quants->chosen && *quants->chosen == quants->pick->index) {
            at_now = product::plan_at_choice(at_weights, now, *served, pins_of(options),
                                             std::move(at_now));
        }
        out.json["device"]["free_now_bytes"] = *free_now;
        out.json["device"]["fits_now"]       = at_now.plan.fits;
        if (!at_now.plan.fits) {
            out.text += "now    | free now " +
                        std::to_string(*free_now / (std::uint64_t{1} << 20)) +
                        " MiB: " + row.listing.quant + " would not start right now (" +
                        at_now.plan.reason + ")\n";
        }
    }
    return out;
}

namespace {

// What the fetch is told and the launch pins (HubQuantChoice), with the rule's log lines.
HubQuantChoice choice_of(const product::QuantRow& row, const product::QuantPlan& plan,
                         const ServeOptions& options,
                         const std::function<void(const std::string&)>& log) {
    HubQuantChoice out{.quant = row.listing.quant};
    // An automatic context with several streams: the launch keeps the own context it was chosen
    // for (as ninferno-ui's does), never a pool the streams silently share.
    if (options.max_context == 0 && options.max_concurrency > 1 && row.launch_context != 0) {
        out.launch_context = row.launch_context;
        if (log) {
            log("quant | --max-context-cap " + std::to_string(row.launch_context) +
                ": each of the " + std::to_string(options.max_concurrency) + " streams keeps " +
                std::to_string(row.launch_context) +
                " tokens of its own (an automatic context would let them share one pool; a busy "
                "card still steps down)");
        }
    }
    if (log && plan.pick && plan.pick->tier == runtime::QuantPickTier::Desktop &&
        &row == &plan.rows[plan.pick->index]) {
        log("quant | WARNING: " + row.listing.quant +
            " fits only by giving up the memory kept for the desktop: the screen may stutter");
    }
    return out;
}

} // namespace

std::optional<HubQuantChoice>
largest_fitting_hub_quant(const ServeOptions& options,
                          const std::function<void(const std::string&)>& log) {
    if (!options.quant.empty() || !convert::is_repository_spec(options.model_source)) {
        return std::nullopt;
    }
    const std::string repo = convert::split_spec(options.model_source, fetch_options()).first;
    const std::map<std::string, fs::path> cached =
        cached_quants(repo, convert::split_spec(options.model_source, fetch_options()).second);
    runtime::PlanDeviceFacts device;
    std::optional<spec::GpuKey> gpu;
    std::string no_gpu;
    try {
        // An idle card of this GPU's class: the same card type gets the same quantization.
        device = product::plan_device_of_cuda(options.device, false);
        gpu    = spec::gpu_identity(options.device).key;
    } catch (const std::exception& e) { no_gpu = e.what(); }

    ModelRead model;
    try {
        model = read_hub(options.model_source, "", options.enable_vision,
                         !options.chat_template_path.empty());
    } catch (const std::exception& e) {
        // The hub cannot be reached: a quantization already here that fits is served (each
        // cached file planned from its own header); otherwise the choice is the user's, never a
        // default that ignores the card.
        if (!cached.empty() && gpu) {
            ModelRead local;
            local.kind  = "hub";
            local.repo  = repo;
            local.gguf  = true;
            bool shaped = false;
            for (const auto& [label, path] : cached) {
                std::error_code error;
                std::uint64_t bytes = 0;
                try {
                    if (ascii_lower(path.extension().string()) == ".gguf") {
                        bytes = local_gguf_bytes(path);
                        if (!shaped) {
                            local.facts = product::plan_facts_of_gguf(read_local_head(path), bytes);
                            shaped      = local.facts.model.known();
                        }
                    } else {
                        bytes = fs::file_size(path, error);
                        if (!shaped) {
                            local.facts = product::plan_facts_of_artifact(path);
                            shaped      = local.facts.model.known();
                        }
                    }
                } catch (const std::exception&) { continue; }
                local.listing.push_back({.quant  = label,
                                         .format = "gguf",
                                         .bytes  = bytes,
                                         .bits   = convert::quant_label_bits(label)});
            }
            if (shaped) {
                Planning planning       = planning_of(options, local, device, *gpu);
                planning.inputs.subject = planning.subject ? &*planning.subject : nullptr;
                const product::QuantPlan plan =
                    product::plan_quants(local.facts, local.listing, planning.inputs);
                if (plan.pick) {
                    const product::QuantRow& row = plan.rows[plan.pick->index];
                    if (log) {
                        log("quant | can't reach the hub to choose a quant (" +
                            std::string(e.what()) + "); the cached " + row.listing.quant +
                            " fits: " + plan.rule + " (--quant LABEL takes another)");
                    }
                    return choice_of(row, plan, options, log);
                }
                throw std::runtime_error(repo + ": can't reach the hub to choose a quant (" +
                                         e.what() + "), and no cached one fits: " + plan.rule +
                                         "; pass --quant LABEL");
            }
        }
        if (log) { log(std::string("quant | the largest that fits is not planned: ") + e.what()); }
        return std::nullopt;
    }
    if (!model.gguf) { return std::nullopt; }
    // Safetensors weights beside the GGUF files: the fetch takes those (convert/fetch.cpp).
    for (const auto& q : model.listing) {
        if (q.format == "safetensors") { return std::nullopt; }
    }
    if (!gpu) {
        if (log) { log("quant | no GPU to plan for: " + no_gpu); }
        return std::nullopt;
    }
    // The shape could not be read: no plan before the download (the fetch's default).
    if (!model.facts.model.known()) {
        if (log) {
            log("quant | can't plan before download: the model's shape is not in its header "
                "(the fetch's default; --quant LABEL takes another)");
        }
        return std::nullopt;
    }
    Planning planning       = planning_of(options, model, device, *gpu);
    planning.inputs.subject = planning.subject ? &*planning.subject : nullptr;
    const product::QuantPlan plan =
        product::plan_quants(model.facts, model.listing, planning.inputs);
    if (!plan.pick) {
        throw std::runtime_error(model.repo + ": " + plan.rule +
                                 " (nothing was downloaded; --quant LABEL takes one anyway, "
                                 "ninferno-serve " +
                                 model.repo + " --plan --no-load lists every quantization)");
    }
    const product::QuantRow& row = plan.rows[plan.pick->index];
    // A quantization already here that runs at the ask is kept: other flags never download
    // another file silently.
    const product::QuantRow* kept = nullptr;
    for (const product::QuantRow& r : plan.rows) {
        if (!cached.contains(r.listing.quant) || !r.candidate.supported) { continue; }
        if (r.fit.state != runtime::QuantFitState::Own &&
            r.fit.state != runtime::QuantFitState::Shared) {
            continue;
        }
        if (kept == nullptr || runtime::quant_ranks_above(r.candidate, kept->candidate)) {
            kept = &r;
        }
    }
    if (kept != nullptr && kept != &row) {
        if (log) {
            if (runtime::quant_ranks_above(kept->candidate, row.candidate)) {
                // Larger than the pick (the pick is in an earlier tier: each stream its own
                // context, where the cached one only fits as a shared pool, or the reverse).
                log("quant | using the cached " + kept->listing.quant + " (" +
                    std::string(runtime::quant_fit_state_name(kept->fit.state)) +
                    " here); the rule's pick at this ask is " + row.listing.quant + " (--quant " +
                    row.listing.quant + " downloads it)");
            } else {
                log("quant | using the cached " + kept->listing.quant + "; " + row.listing.quant +
                    " would be larger (--quant " + row.listing.quant + " downloads it)");
            }
        }
        return choice_of(*kept, plan, options, log);
    }
    if (log) { log("quant | " + plan.rule + " (an estimate; --quant LABEL takes another)"); }
    return choice_of(row, plan, options, log);
}

std::optional<product::StartAxes>
choose_served_axes(const ServeOptions& options, const std::function<void(const std::string&)>& log) {
    const product::ChoicePins pins = pins_of(options);
    // Every axis pinned and the drafter given (--spec): nothing left to choose.
    if (pins.kv && pins.activations && pins.state && pins.ring && pins.chunk &&
        !options.speculation_from_profile) {
        return std::nullopt;
    }
    runtime::PlanDeviceFacts device;
    spec::GpuKey gpu;
    try {
        device = product::plan_device_of_cuda(
            options.device, true, runtime::plan_automatic_fit(engine_options_of(options)));
        gpu = spec::gpu_identity(options.device).key;
    } catch (const std::exception& e) {
        if (log) { log(std::string("not planned (no GPU facts: ") + e.what() + ")"); }
        return std::nullopt;
    }
    ServeOptions at_artifact = options;
    at_artifact.model_source = options.artifact_path;
    std::optional<product::ServedChoice> served;
    try {
        const ModelRead model   = read_model(at_artifact);
        Planning planning       = planning_of(at_artifact, model, device, gpu);
        planning.inputs.subject = planning.subject ? &*planning.subject : nullptr;
        served                  = product::choose_at_weights(model.facts, planning.inputs, pins);
    } catch (const std::exception& e) {
        if (log) { log(std::string("not planned: ") + e.what()); }
        return std::nullopt;
    }
    if (!served) { return std::nullopt; }
    if (log) { log(product::choice_line(*served).substr(9)); }
    if (!served->choice.chosen) { return std::nullopt; }
    return product::start_axes(*served, pins);
}

} // namespace ninferno::serve
