#include "serve/plan_offline.h"

#include "convert/acquire.h"
#include "convert/fetch.h"
#include "convert/gguf.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "serve/generation_service.h"
#include "spec/fingerprint.h"
#include "spec/gpu_identity.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
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
};

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

// A hub repository: its listing and the facts at `quant` (the smallest GGUF when none is named).
ModelRead read_hub(const std::string& spec, const std::string& quant) {
    ModelRead out;
    out.kind                          = "hub";
    out.source                        = spec;
    convert::FetchOptions fetch       = fetch_options();
    const auto quants                 = convert::list_quants(spec, fetch);
    out.repo                          = convert::split_spec(spec, fetch).first;
    const convert::RepoQuant* header  = nullptr;
    const convert::RepoQuant* weights = nullptr;
    const convert::RepoQuant* native  = nullptr;
    for (const auto& q : quants) {
        out.listing.push_back({.quant   = q.quant,
                               .format  = q.format,
                               .bytes   = q.bytes,
                               .bits    = q.bits,
                               .drafter = q.drafter});
        if (q.drafter) { continue; }
        if (q.format == "gguf") {
            const bool named = !quant.empty() && ascii_lower(q.quant) == ascii_lower(quant);
            if (named || (quant.empty() && (header == nullptr || q.bytes < header->bytes))) {
                header = &q;
            }
        } else if (q.format == "safetensors") {
            weights = &q;
        } else if (q.format == "ninferno") {
            native = &q;
        }
    }
    if (header != nullptr) {
        out.gguf    = true;
        out.quant   = header->quant;
        fetch.quant = header->quant;
        for (const std::uint64_t bytes : {8ULL << 20, 32ULL << 20, 128ULL << 20}) {
            try {
                const auto head = convert::fetch_gguf_head(spec, fetch, bytes);
                const auto file =
                    convert::parse_gguf_head(reinterpret_cast<const std::byte*>(head.bytes.data()),
                                             head.bytes.size(), head.size, head.path);
                out.facts = product::plan_facts_of_gguf(file, header->bytes);
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
        out.quant = native->quant;
        out.listing.clear();
        return out;
    }
    if (weights != nullptr) {
        out.facts = product::plan_facts_of_config(convert::fetch_config(spec, fetch),
                                                  weights->bytes, runtime::WeightBasis::FileSize);
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
        if (fs::is_directory(path)) {
            if (!fs::exists(path / "config.json")) {
                throw std::runtime_error(name + " has no config.json to plan from");
            }
            std::ifstream in(path / "config.json");
            const Json config = Json::parse(in);
            out.kind          = "checkpoint";
            out.facts         = product::plan_facts_of_config(config, safetensors_bytes(path),
                                                              runtime::WeightBasis::FileSize);
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
    return read_hub(name, options.quant);
}

struct Planning {
    product::DryRunInputs inputs;
    std::optional<product::SpeculationSubject> subject;
};

// The inputs a server's options give for a model read without its weights.
Planning planning_of(const ServeOptions& options, const ModelRead& model,
                     const runtime::PlanDeviceFacts& device, const spec::GpuKey& gpu) {
    Planning out;
    out.inputs.device  = device;
    out.inputs.gpu     = gpu;
    out.inputs.options = engine_options_of(options);
    out.inputs.ask     = product::SpeculationAsk{.automatic   = options.speculation_from_profile,
                                                 .given       = options.speculative,
                                                 .thinking    = options.enable_thinking.value_or(true),
                                                 .per_request = options.per_request_speculation,
                                                 .depth_from_profile = options.draft_tokens_from_model,
                                                 .model_repo         = model.repo};
    if (!model.artifact.empty()) {
        out.subject = product::speculation_subject(model.artifact, gpu, options.spec_profile);
        try {
            out.inputs.ask.artifact_id = spec::artifact_id_hex(model.artifact);
        } catch (const std::exception&) {}
    } else {
        out.subject =
            product::planned_subject(product::PlannedTarget{.repo          = model.repo,
                                                            .facts         = model.facts,
                                                            .gguf          = model.gguf,
                                                            .with_drafters = options.with_drafters},
                                     gpu);
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
            device = product::plan_device_of_cuda(options.device, true);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("no GPU to plan for (") + e.what() +
                                     "): name a board with --plan-device");
        }
        gpu = spec::gpu_identity(options.device).key;
    }
    const ModelRead model   = read_model(options);
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
            facts   = product::with_weights(model.facts, chosen->listing.bytes, chosen->basis);
        } else {
            planned = product::plan_model(model.facts, planning.inputs);
        }
    } else {
        planned = product::plan_model(model.facts, planning.inputs);
    }
    const std::string source =
        model.source + (model.quant.empty() || quants ? std::string() : " " + model.quant);
    out.json = product::dry_plan_document(facts, source, model.kind, planning.inputs, planned,
                                          quants ? &*quants : nullptr);
    out.text = product::render_dry_plan_text(facts, source, planning.inputs, planned,
                                             quants ? &*quants : nullptr);
    out.fits = planned.plan.fits;
    return out;
}

std::optional<std::string>
largest_fitting_hub_quant(const ServeOptions& options,
                          const std::function<void(const std::string&)>& log) {
    if (!options.quant.empty() || !convert::is_repository_spec(options.model_source)) {
        return std::nullopt;
    }
    ModelRead model;
    try {
        model = read_hub(options.model_source, "");
    } catch (const std::exception& e) {
        if (log) { log(std::string("quant | the largest that fits is not planned: ") + e.what()); }
        return std::nullopt;
    }
    if (!model.gguf) { return std::nullopt; }
    // Safetensors weights beside the GGUF files: the fetch takes those (convert/fetch.cpp).
    for (const auto& q : model.listing) {
        if (q.format == "safetensors") { return std::nullopt; }
    }
    runtime::PlanDeviceFacts device;
    try {
        // An idle card of this GPU's class: the same card type gets the same quantization.
        device = product::plan_device_of_cuda(options.device, false);
    } catch (const std::exception& e) {
        if (log) { log(std::string("quant | no GPU to plan for: ") + e.what()); }
        return std::nullopt;
    }
    const spec::GpuKey gpu  = spec::gpu_identity(options.device).key;
    Planning planning       = planning_of(options, model, device, gpu);
    planning.inputs.subject = planning.subject ? &*planning.subject : nullptr;
    const product::QuantPlan plan =
        product::plan_quants(model.facts, model.listing, planning.inputs);
    if (!plan.chosen) {
        throw std::runtime_error(model.repo + ": " + plan.rule +
                                 " (nothing was downloaded; --quant LABEL takes one anyway, "
                                 "ninferno-serve " +
                                 model.repo + " --plan --no-load lists every quantization)");
    }
    if (log) { log("quant | " + plan.rule + " (an estimate; --quant LABEL takes another)"); }
    return plan.rows[*plan.chosen].listing.quant;
}

} // namespace ninferno::serve
