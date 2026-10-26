#pragma once

// What ninferno-ui offers to run, and how well it would run on a given GPU.
//
// The catalog (src/ui/data/catalog.json, compiled in) names the recommended models, where to get
// each (hub repository and quantization, several sources per model) and what the stands measured.
// For one GPU every source gets:
//
//   fit        fits | tight | no: its weights against the card's memory, with the reserve the
//              engine needs for activations and a small context; sizes come from the hub listing
//              (unknown until it arrives: then params x bits);
//   speed      measured on this GPU model when the stands have it, else the nearest measured card
//              scaled by memory bandwidth (decode is bandwidth-bound), else bandwidth / active
//              bytes per token at the efficiency the measured cards show; each says which;
//   context    the context the stands ran on this card, else "auto" (the engine sizes it at start);
//   preferred  the source the UI selects: NVFP4 where the card has FP4 tensor cores, else GGUF,
//              among those that fit.

#include "ui/gpu_stats.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::product {
struct PlanSourceFacts;
}

namespace ninferno::ui {

struct CatalogSource {
    std::string repo, quant, format; // format: nvfp4 | gguf | safetensors
    int bits = 4;
};

struct CatalogMeasure {
    std::string gpu, context, quant;
    double tps = 0, llama_cpp = 0;
};

struct CatalogModel {
    std::string id, name, kind, use, spec;
    double params_b = 0, active_b = 0;
    std::vector<CatalogSource> sources;
    std::vector<CatalogMeasure> measured;
    // Decode of one request at 1, 2, 4... parallel requests, without and with speculation, as
    // the stands measured it ([{gpu, quant, context, run, per_request: {none: {"1": tok/s, ...},
    // auto: {...}}}]); the page interpolates between the counts.
    nlohmann::json load = nlohmann::json::array();
    // Every drafter that exists for the model (kind, repository, whether the engine runs it, how it
    // is converted, what others measured) and the stands' speculation suite (tok/s per drafter
    // configuration and task family): shown as they are, read by the page.
    nlohmann::json drafters   = nlohmann::json::array();
    nlohmann::json spec_suite = nlohmann::json::array();
};

struct CatalogGpu {
    std::string key;
    std::vector<std::string> match;
    double memory_gib = 0, bandwidth_gbs = 0;
    bool fp4 = false;
};

class Catalog {
public:
    [[nodiscard]] static Catalog parse(const nlohmann::json& document);
    [[nodiscard]] static const Catalog& builtin();

    [[nodiscard]] const std::vector<CatalogModel>& models() const noexcept { return models_; }

    [[nodiscard]] const std::vector<CatalogGpu>& gpus() const noexcept { return gpus_; }

    // The catalog GPU an NVML name is (substring match), nullptr when none.
    [[nodiscard]] const CatalogGpu* gpu_for(const std::string& nvml_name) const;
    [[nodiscard]] const CatalogGpu* gpu_by_key(const std::string& key) const;

private:
    std::vector<CatalogModel> models_;
    std::vector<CatalogGpu> gpus_;
};

// Sizes of hub files, as the listings report them: "repo|quant" -> bytes (quant "" for a
// safetensors repository's weights).
using SizeTable = std::map<std::string, std::uint64_t>;
[[nodiscard]] std::string size_key(const std::string& repo, const std::string& quant);

// The GPU the estimate is for: what NVML says (memory, bandwidth, name, FP4 by compute capability).
struct TargetGpu {
    std::string name;
    double memory_gib    = 0;
    double bandwidth_gbs = 0;
    std::string chip;                       // "ad102" (core/device_class.h gpu_chip)
    std::vector<std::string> profile_chain; // gpu_similarity_chain: own class, chip, arch, ...
    bool fp4             = false;
};

[[nodiscard]] TargetGpu target_of(const GpuStat& stat);

// Which speculation the model would start with on that GPU, from the drafter registry: its
// drafters, and the profile the class chain reaches first (own chip, a similar chip, the "*"
// fallback, or none: no speculation profile, so prompt lookup or plain decode).
struct SpeculationPlan {
    std::vector<std::string> drafters; // kinds: mtp, dflash, dflash2, gemma4_assistant, ...
    std::string profile;               // own | similar | generic | none
    std::string profile_class;         // the class key it came from ("ga102", "ad102", "*")
};

[[nodiscard]] SpeculationPlan speculation_plan(const std::string& registry_id,
                                               const TargetGpu& gpu);
void attach_speculation(nlohmann::json& model_json, const SpeculationPlan& plan);

struct SourceEstimate {
    std::size_t source = 0;
    double size_gib    = 0;
    bool size_known    = false;
    std::string fit; // fits | tight | no
    double tps = 0;
    std::string tps_basis; // measured | scaled from <gpu> | bandwidth
    double llama_cpp = 0;  // measured on this GPU model, 0 otherwise
    std::string context;   // measured context, or "auto"
};

struct ModelEstimate {
    const CatalogModel* model = nullptr;
    std::vector<SourceEstimate> sources;
    std::optional<std::size_t> preferred; // index into sources
    bool recommended = false;             // a preferred source fits
};

// Engine reserve beside the weights: activations, workspaces, CUDA context and a small context.
inline constexpr double kReserveGib = 2.0;

// When none of the model's sources fits the GPU, the model with one more source: the largest
// quantization of 2 to 4 bits from its GGUF repositories' listings (repo -> [{quant, bytes, bits,
// drafter}], as App keeps them) that fits with the reserve. A 16 GB card then gets the 3-bit
// quantization of a 30B model instead of "does not fit". Unchanged when a source fits, or when no
// listing has a quantization that fits.
//
// `fits(repo, quant)` is the plan without the weights (App::plan_json: an idle card of this GPU,
// automatic context, one stream, the server's own ladder); nullopt where it cannot say (the
// repository's shape not read yet), where the weights against the card with kReserveGib stand in.
using QuantFits =
    std::function<std::optional<bool>(const std::string& repo, const nlohmann::json& quant)>;
// `pick(repo)`: the rule's pick for the repository at the user's context x streams (nullopt: not
// planned); when it has one, that is the quant added, else the listing's largest 2-4 bit quant
// that fits.
using QuantPickOf = std::function<std::optional<std::string>(const std::string& repo)>;
[[nodiscard]] CatalogModel with_fitting_quant(const CatalogModel& model, const TargetGpu& gpu,
                                              const std::map<std::string, nlohmann::json>& listings,
                                              const QuantFits& fits   = {},
                                              const QuantPickOf& pick = {});

// What the page reads of a model's KV, from the plan's facts (product::PlanSourceFacts: the one
// reading of a GGUF header or a config.json): its architecture, whether its KV is known (layers
// that keep KV: not latent attention, not an unread shape), the full layers' key size (the int8,
// fp8 and nvfp4 codecs need 256) and its trained context (the context slider's end). Unknown
// without facts.
[[nodiscard]] nlohmann::json kv_summary(const product::PlanSourceFacts* facts);

[[nodiscard]] ModelEstimate estimate(const Catalog& catalog, const CatalogModel& model,
                                     const TargetGpu& gpu, const SizeTable& sizes);
[[nodiscard]] nlohmann::json to_json(const ModelEstimate& estimate);

} // namespace ninferno::ui
