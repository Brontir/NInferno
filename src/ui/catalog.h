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
#include <map>
#include <optional>
#include <string>
#include <vector>

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
    // The lab's measurement of each --activations mode ([{gpu, quant, mode: auto|16|8|4, kld,
    // reference, top1, perplexity, decode, prefill, run, ...}]): the model's own "precision" list,
    // else the build's rows for its id (data/precision.json, tools/docs/gen_ui_precision.py). The
    // page shows a mode's speed and quality only from a row of the user's card and quantization.
    nlohmann::json precision = nlohmann::json::array();
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
[[nodiscard]] CatalogModel
with_fitting_quant(const CatalogModel& model, const TargetGpu& gpu,
                   const std::map<std::string, nlohmann::json>& listings);

// The KV cache a model keeps per token of a request, from its GGUF metadata (general.architecture
// and its keys): full-attention layers grow with the context, sliding-window layers stop at the
// window, recurrent layers (GDN, Mamba) keep none. INT8 KV (the default storage): one byte per
// element. known is false when the metadata lacks the attention geometry.
//
// The same metadata gives the parameter counts the speed estimates use (0: unknown): all of them,
// those one token reads with one request (the routed experts it takes, the shared ones, attention,
// the output head; not the embedding table), and the routed experts' share of all with the
// fraction a token takes (more requests at once touch more experts: 1 - (1 - used)^requests).
// Counted from the shapes (attention, feed-forward, experts, vocabulary), not from the file:
// recurrent layers count as attention ones, so a hybrid's figures are approximate.
//
// Memory as ninferno-serve plans it (runtime/engine/kv_capacity.h), checked against its capacity
// lines on the 4090 (Qwen3.6 35B at 1 x 8K: 160 MiB KV + 2 x 60 MiB states + 79 MiB workspace
// = 357 MiB runtime, as logged; at 4 x 8K 1.12 GiB + 85 MiB = 1.18 GiB):
//   weights + KV pool (requests x context x full_bytes_per_token)
//   + per sequence, each active request and each cached device state of the context cache:
//     state_bytes and a sliding ring of window + prefill chunk rows of window_bytes_per_token
//   + workspace_bytes + kRuntimeOverheadGib (CUDA context, kernels, graphs).
// KV is bf16 here (the page scales it by the --kv-dtype it sends). The recurrent state follows
// --activations as ninferno-serve's automatic state does (runtime::normalize_engine_options):
// fp32 under 16, under auto, 8 and 4 the family's own (f16 for Qwen 3.5's Gated DeltaNet, fp32
// for the others); its convolution is bf16.
inline constexpr double kKvElementBytes     = 2;
inline constexpr long kPrefillChunk         = 1024; // serve's default prefill chunk
inline constexpr double kRuntimeOverheadGib = 0.75;
struct KvModel {
    std::string architecture; // general.architecture, or config.json's architectures[0]
    bool known                    = false;
    double full_bytes_per_token   = 0; // all full-attention layers together
    double window_bytes_per_token = 0; // all sliding-window layers together
    long window                   = 0; // their window in tokens
    long native_context           = 0; // the model's trained context (0: unknown)
    long layers                   = 0;
    long full_layers              = 0; // full-attention layers that keep KV
    double kv_head_dim            = 0; // their key size: int8/fp8/nvfp4 KV only at 256
    double state_bytes            = 0; // recurrent (GatedDeltaNet, Mamba2) state of one sequence
                                       // under --activations 16 (fp32)
    double state_bytes_auto       = 0; // the same under auto, 8 and 4 (the family's state)
    double workspace_bytes        = 0; // activations of one prefill chunk
    double params                 = 0;
    double active_params          = 0;
    double routed_share           = 0; // routed experts' parameters / all parameters
    double experts_used           = 0; // experts a token takes / routed experts (0: dense)
};

// The parameter counts and the workspace of KvModel from shapes; exposed for the tests. Layers
// are counted by kind: attention, recurrent mixers (GatedDeltaNet, Mamba2) and feed-forward
// (dense or experts) each where the model has them (NemotronH interleaves the three; a Qwen3.5
// layer is attention or GatedDeltaNet, then feed-forward). Zero counts mean every layer.
struct ModelShape {
    double layers = 0, hidden = 0, heads = 0, kv_heads = 0, head_dim = 0;
    double attention_layers = 0, mixer_layers = 0, ffn_layers = 0;
    double ffn     = 0; // a dense layer's feed-forward width
    double experts = 0, experts_used = 0, expert_ffn = 0, shared_ffn = 0;
    double ssm_inner = 0, ssm_state = 0, ssm_groups = 0, ssm_heads = 0, ssm_conv = 0;
    double ffn_matrices = 3; // gate, up, down; NemotronH's squared-ReLU experts have no gate
    double vocab = 0;
    bool tied    = false; // the output head is the embedding table
};

void count_params(KvModel& kv, const ModelShape& shape);

[[nodiscard]] KvModel kv_model_of_gguf(const nlohmann::json& metadata);
// The same from a Hugging Face config.json (its text_config when it has one): layer_types
// (full / sliding / linear attention), else hybrid_override_pattern (NemotronH: '*' attention),
// else full_attention_interval, else every layer full; a global head geometry
// (global_head_dim, num_global_key_value_heads) for the full layers where it differs. Unknown for
// latent attention (kv_lora_rank), whose cache this estimate does not model.
[[nodiscard]] KvModel kv_model_of_config(const nlohmann::json& config);
// Bytes of KV one request of `context` tokens holds.
[[nodiscard]] double kv_bytes(const KvModel& kv, long context);
[[nodiscard]] nlohmann::json to_json(const KvModel& kv);

[[nodiscard]] ModelEstimate estimate(const Catalog& catalog, const CatalogModel& model,
                                     const TargetGpu& gpu, const SizeTable& sizes);
[[nodiscard]] nlohmann::json to_json(const ModelEstimate& estimate);

} // namespace ninferno::ui
