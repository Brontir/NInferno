#include "core/device_traits.h"
#include "runtime/engine/model_instance.h"
#include "runtime/engine/family_registry.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_ladder.h"
#include "runtime/engine/warmup_widths.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "artifact/routed_experts.h"
#include "artifact/source_identity.h"
#include "core/arena.h"
#include "core/device_class.h"
#include "core/hex.h"
#include "core/host_memory.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/program/program.h"
#include "ninferno/ops/route_table.h"
#include "ninferno/ops/softmax_attention.h"
#include "ninferno/types.h"
#include "ops/generic/generic_gemm.h"
#include "ops/route_options.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <compare>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace ninferno::runtime {
namespace {
using Clock = std::chrono::steady_clock;

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// The driver model and the display: core/device_traits.h (the plan without the weights reads the
// same functions).
using ninferno::drives_display;
using ninferno::paravirtualized_gpu;

// The CUDA stack limit: the local memory per thread the driver keeps for every resident thread. It
// starts at the driver's default and grows at the first launch of a kernel with a larger frame.
std::size_t current_stack_limit_bytes() {
    std::size_t bytes = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&bytes, cudaLimitStackSize));
    return bytes;
}

// The kernel warm-up's prompts repeat this text's tokens to each width.
constexpr std::string_view kWarmUpText =
    "Before it measures the free memory, the engine runs every kernel its plan can choose. ";
// Checks after the prefill probe before the rung and the Host tier's pinned sizes stop moving
// (each probe Program and each pinning measurement only settles them); a bound this far above
// three means a fault.
constexpr std::uint32_t kMaximumWarmUpPasses = 8;
// The first step of the reserve a probe whose graphs did not fit adds to the check (doubled per
// retry).
constexpr std::size_t kGraphShortfallStepBytes = std::size_t{128} << 20;

// One request of `prompt` through the Program's request path, as the Engine runs it: admission,
// the prefill (with its speculative and sampling work), the first token and, with `decode_round`,
// one decode round of up to `round_tokens`; then the sequence is aborted.
void warm_up_request(models::qwen3_5::Program& program, const models::qwen3_5::Frontend& frontend,
                     std::vector<TokenId> prompt_ids, std::uint32_t round_tokens,
                     bool decode_round) {
    models::qwen3_5::PreparedPrompt prompt =
        frontend.prepare_tokens(std::move(prompt_ids), false);
    ResolvedExecutionOptions execution;
    execution.sampling.temperature     = 1.0F;
    execution.requested_output_tokens = 1U + round_tokens;
    execution.allow_prefix_reuse      = false;
    const models::qwen3_5::RequestBasePlan base = program.plan_request(prompt, execution);
    const std::optional<models::qwen3_5::AdmissionCandidate> candidate =
        program.inspect_admission(prompt, base, LaneId{0}, nullptr, nullptr, std::nullopt, false);
    if (!candidate) { throw std::logic_error("the kernel warm-up request was not admitted"); }
    std::optional<models::qwen3_5::ResourcePlan> resources =
        program.seal_identity(*candidate, prompt, {});
    if (!resources) { throw std::logic_error("the kernel warm-up request was not sealed"); }
    if (program.start_resource_transaction(std::move(*resources), std::move(prompt), {}) !=
        ContextTransactionReserveStatus::Reserved) {
        throw std::logic_error("the kernel warm-up request was not reserved");
    }
    std::optional<models::qwen3_5::MaterializationResult> published;
    while (!published) {
        auto progress = program.progress_context_transaction({});
        if (std::holds_alternative<ContextTransactionInProgress>(progress)) { continue; }
        if (!std::holds_alternative<models::qwen3_5::MaterializationResult>(progress)) {
            program.finalize_context_transaction();
            throw std::logic_error("the kernel warm-up request's transaction is not a start");
        }
        published.emplace(std::get<models::qwen3_5::MaterializationResult>(std::move(progress)));
    }
    program.finalize_context_transaction();
    if (published->status != ContextTransactionStatus::Published || !published->published) {
        throw std::logic_error("the kernel warm-up request was not published");
    }
    const models::qwen3_5::SequenceHandle sequence = published->published->sequence;
    try {
        std::optional<models::qwen3_5::PrefillProgress> prefill;
        do {
            if (prefill && prefill->capture) {
                throw std::logic_error("the kernel warm-up request offered a capture");
            }
            prefill.emplace(program.advance_prefill(sequence));
        } while (!prefill->complete);
        if (!prefill->pending) { throw std::logic_error("the kernel warm-up prefill is pending"); }
        const std::array<CommitDecision, 1> first{CommitDecision{.accepted_tokens = 1}};
        (void)program.commit(std::move(*prefill->pending), first);
        if (decode_round) {
            const std::array<models::qwen3_5::SequenceHandle, 1> sequences{sequence};
            const std::array<RoundBudget, 1> budgets{
                RoundBudget{.generated_tokens_remaining = round_tokens}};
            models::qwen3_5::PendingBatch round = program.decode(sequences, budgets);
            const auto produced = static_cast<std::uint32_t>(
                round.row_counts().empty() ? 1 : round.row_counts().front());
            const std::array<CommitDecision, 1> accepted{
                CommitDecision{.accepted_tokens = produced}};
            (void)program.commit(std::move(round), accepted);
        }
    } catch (...) {
        (void)program.abort(sequence);
        throw;
    }
    if (program.abort(sequence).status != ConsumeStatus::Consumed) {
        throw std::logic_error("the kernel warm-up request could not be released");
    }
}

} // namespace

void validate_engine_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninferno") {
        throw std::invalid_argument("NInferno accepts only .ninferno artifacts");
    }
    if (options.max_context == 0 && !options.max_context_automatic) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    // Every family's full-attention layers see the whole context, and attention reads at most
    // this many keys per row: refused here rather than as an execution envelope at load.
    if (!options.max_context_automatic && options.max_context > ops::kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument("--max-context " + std::to_string(options.max_context) +
                                    " exceeds the " +
                                    std::to_string(ops::kCausalAttentionMaximumVisibleKeys) +
                                    " keys attention reads per token");
    }
    if (!(options.rope_yarn_factor >= 1.0F) || !std::isfinite(options.rope_yarn_factor) ||
        options.rope_yarn_factor > 64.0F) {
        throw std::invalid_argument("--rope-yarn-factor must be in [1, 64]");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    if (options.recurrent_state != RecurrentStateStorage::FP32 &&
        options.activations == ActivationMode::Exact) {
        throw std::invalid_argument(
            "--state-dtype bf16|f16 rounds the recurrent state wherever a call stores it, so an "
            "answer depends on where the prompt is cut and how speculation groups the tokens, and "
            "--activations exact asks for bytes that do not: drop one of the two");
    }
}

EngineOptions admit_engine_options(EngineOptions options) {
    // A --route-dir or --route-table that cannot be read (or a malformed table) is refused before
    // the device is touched (ops/route_options.h): read as no table, it would run the heuristic.
    // The table's GPU class is checked when it loads on the device (engine.cpp).
    ops::check_route_options(options.maintainer.route_table, options.maintainer.route_dir);
    return normalize_engine_options(std::move(options));
}

EngineOptions normalize_engine_options(EngineOptions options) {
    if (options.recurrent_state_automatic) {
        options.recurrent_state =
            options.activations == ActivationMode::Exact
                ? RecurrentStateStorage::FP32
                : automatic_recurrent_state(artifact_model_type(options.artifact_path));
        options.recurrent_state_automatic = false;
    }
    if (options.kv_cache_automatic) {
        // The device's compute capability decides FP8 rows (automatic_kv_cache); a host without
        // the device keeps the family's storage, which the load then reports or refuses.
        int compute_capability = 1000;
        try {
            compute_capability = device_class(options.device).compute_capability();
        } catch (const std::exception&) {}
        options.kv_cache =
            automatic_kv_cache(artifact_model_type(options.artifact_path), compute_capability);
        options.kv_cache_automatic = false;
    }
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        if (options.max_context_automatic) {
            throw std::invalid_argument("causal scoring needs an explicit max_context");
        }
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    // An unconfigured cached Device state count stays unset here: the plan ladder steps it
    // (construct_or_plan, plan_ladder.h), and the Engine reports the rung's count.
    // Room for eight conversations beside the active ones: agents rotate more sessions than
    // requests run at once, and a descriptor costs a catalog entry and an address space, the KV
    // and states staying bounded by their own capacities (four rotating agents at C1 re-prefilled
    // 17 % less with 9 than with 2).
    const std::uint64_t default_private =
        std::max<std::uint64_t>(2ULL * concurrency, static_cast<std::uint64_t>(concurrency) + 8U);
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    // An automatic Host tier keeps two StateImages a conversation (its endpoint and the turn an
    // edit returns to) rather than a fixed eight: with four rotating agents eight states left each
    // only an old anchor between its turns. The Host fit still scales it to what the host pins.
    if (cache.host_tier_automatic) {
        cache.host_state_slots = std::max(cache.host_state_slots,
                                          2U * *cache.max_private_continuations);
    }
    cache.max_shared_prefixes               = cache.max_shared_prefixes.value_or(std::max(
        concurrency, static_cast<std::uint32_t>(kMaximumPreparedPromptCacheCandidatesPerRequest)));
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) +
        cache.device_state_slots.value_or(automatic_cached_device_states(concurrency));
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

namespace {

// The Qwen 3.5 model and the execution parameters prepared from it, which its execution reads.
class Qwen35LoadedModel final : public LoadedModel {
public:
    explicit Qwen35LoadedModel(std::unique_ptr<models::qwen3_5::Model> source)
        : model(std::move(source)), parameters(*model) {}

    const std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
};

} // namespace

FamilyLoad load_qwen3_5(const EngineOptions& options, DeviceContext& device) {
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto loaded = std::make_unique<Qwen35LoadedModel>(
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer));
    device.synchronize();
    const models::qwen3_5::Model& model = *loaded->model;
    FamilyLoad out;
    if (const auto* moe = std::get_if<models::qwen3_5::MoeConfig>(&model.config().text.ffn)) {
        out.routed_experts    = moe->num_experts;
        out.experts_per_token = moe->num_experts_per_tok;
    }
    out.execution = models::qwen3_5::detail::make_qwen3_5_execution(loaded->parameters);
    const models::qwen3_5::FrontendOptions frontend_options{
        .chat_template_path       = options.chat_template_path,
        .architecture             = model.config().text.architecture,
        .vision_enabled           = options.enable_vision,
        .vision_component         = reader.directory().components.contains("vision"),
        .media_cache_bytes        = options.media_cache_bytes,
        .media_live_bytes         = options.media_live_bytes,
        .media_preprocess_threads = options.media_preprocess_threads};
    out.make_frontend = [resources = &model.resources(),
                         frontend_options](std::uint32_t max_context) {
        models::qwen3_5::FrontendOptions sized = frontend_options;
        sized.max_context                      = max_context;
        return models::qwen3_5::make_frontend(*resources, std::move(sized));
    };
    const auto& stats         = model.storage_stats();
    LoadSummary& summary      = out.summary;
    summary.architecture      = models::architecture_name(model.config().text.architecture);
    // The name convert --name gave the artifact, else its file name, as the other families
    // serve it (the architecture names every Qwen artifact alike).
    summary.model_name        = reader.directory().metadata.contains("name")
                                    ? model.info().name
                                    : options.artifact_path.stem().string();
    summary.prefill_signature = models::qwen3_5::prefill_signature(model);
    std::set<std::string> formats;
    for (const auto& weight : model.weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    out.model                    = std::move(loaded);
    return out;
}

ModelInstance::ModelInstance(FamilyLoad&& load, models::qwen3_5::Frontend frontend_,
                             std::uint32_t capacity_)
    : model(std::move(load.model)), execution(std::move(load.execution)),
      frontend(std::move(frontend_)), capacity(capacity_) {
    if (!model || !execution) {
        throw std::logic_error("a family load must carry its model and execution");
    }
}

ModelInstance::~ModelInstance() = default;

namespace {

// The static /v1/models facts of any family: the identity and dimensions from the loaded model's
// traits, the parameters, weight bytes and weights profile from the artifact's tensor inventory
// (every component, as llama.cpp counts a whole GGUF).
// `artifact_id`: the artifact's id (hex), `source_identity` its source identity
// (artifact/source_identity.h; "none:<artifact id>" when it has none), and
// `routed_expert_bytes`: the bytes of its routed-expert objects (artifact/routed_experts.h; the
// decode-round price's MoE share), from the same read.
ModelMetadata model_metadata(const std::filesystem::path& artifact_path, const LoadSummary& summary,
                             const models::qwen3_5::detail::ModelTraits& traits,
                             std::string& artifact_id, std::string& source_identity,
                             std::uint64_t& routed_expert_bytes) {
    ModelMetadata metadata;
    metadata.model_id       = summary.model_name;
    metadata.vocab_size     = static_cast<std::uint64_t>(traits.public_tokens);
    metadata.embedding_size = static_cast<std::uint64_t>(traits.hidden);
    metadata.native_context = traits.max_positions;
    const artifact::Reader reader(artifact_path);
    artifact_id     = hex_encode(reader.artifact_id().data(), reader.artifact_id().size());
    source_identity = artifact::source_identity(reader.directory()).value_or("none:" + artifact_id);
    std::set<std::string> formats;
    const auto expert_layers = artifact::routed_expert_object_layers(reader.directory());
    routed_expert_bytes      = 0;
    for (std::size_t index = 0; index < reader.directory().objects.size(); ++index) {
        const auto& object = reader.directory().objects[index];
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; } // resources: tokenizer, templates
        if (expert_layers[index]) { routed_expert_bytes += tensor->bytes; }
        std::uint64_t elements = 1;
        for (const auto dimension : tensor->shape) { elements *= dimension; }
        metadata.parameters += elements;
        metadata.weight_bytes += tensor->bytes;
        formats.emplace(tensor->format);
    }
    for (const auto& format : formats) {
        if (!metadata.weights_id.empty()) { metadata.weights_id += "+"; }
        metadata.weights_id += format;
    }
    const auto& provenance = reader.directory().provenance;
    if (provenance.is_object() && provenance.contains("published_activations") &&
        provenance.at("published_activations").is_object()) {
        for (const auto& [key, value] : provenance.at("published_activations").items()) {
            if (!value.is_number_unsigned()) { continue; }
            if (key == "below") {
                metadata.below_published_activations = value.get<std::uint64_t>();
            } else if (key == "below_int8") {
                metadata.below_published_activations_int8 = value.get<std::uint64_t>();
            } else if (key == "16" || key == "8" || key == "4" || key == "unknown") {
                metadata.published_activations[key == "unknown" ? 0 : std::stoi(key)] =
                    value.get<std::uint64_t>();
            }
        }
    }
    return metadata;
}

} // namespace

ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device) {
    validate_engine_options(options);
    return construct_model(options, device,
                           family_loader(artifact_model_type(options.artifact_path)));
}

namespace {

std::string gib_text(std::uint64_t bytes) {
    // Hundredths of a GiB, rounded up: a shortfall is never shown smaller than it is.
    const std::uint64_t hundredths = (bytes * 100ULL + (std::uint64_t{1} << 30) - 1ULL) >> 30;
    const std::string cents        = std::to_string(hundredths % 100);
    return std::to_string(hundredths / 100) + "." + (cents.size() < 2 ? "0" : "") + cents + " GiB";
}

std::string grouped_count(std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return digits;
}

// Builds the Engine's model, or with `dry_run` only its plan: the family load, the ladder and the
// rung the free memory picks (no tuner, no kernel warm-up, no Program).
ConstructedModel construct_or_plan(const EngineOptions& options, DeviceContext& device,
                                   FamilyLoader load, PlanSummary* dry_run) {
    validate_engine_options(options);
    const auto start    = Clock::now();
    FamilyLoad loaded   = load(options, device);
    LoadSummary summary = std::move(loaded.summary);
    if (!loaded.execution || !loaded.make_frontend) {
        throw std::logic_error("a family load must carry its execution and its frontend");
    }
    const models::qwen3_5::detail::ModelExecution& execution = *loaded.execution;
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    // Keyed by the device's class and board facts, never its driver name.
    auto context_cost = resolve_context_machine_cost(
        context_cost_identity(device.device_class(), summary.prefill_signature),
        options.context_cost.preset_path);
    EngineOptions resolved = options;
    ContextResolution context{
        .automatic = options.max_context_automatic,
        .native    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            execution.traits().max_positions, std::numeric_limits<std::uint32_t>::max())),
        .tokens    = options.max_context,
    };
    // An automatic max_context starts from the model's own (under the cap, and within an explicit
    // KV capacity, which bounds one sequence): plan_dry_run.h, plan_context_ceiling.
    if (context.automatic && context.native == 0) {
        throw std::logic_error("the model reports no position capacity");
    }
    resolved.max_context = plan_context_ceiling(options, context.native);
    // A configured cached Device state count (none with the context cache off) pins that axis.
    const std::optional<std::uint32_t> configured_states =
        options.context_cache.enabled ? options.context_cache.device_state_slots
                                      : std::optional<std::uint32_t>(0U);
    // The options a candidate plan is built from: the resolved ones with its context, cached
    // Device states, concurrency and prefill chunk.
    const auto planned_at = [&](std::uint32_t max_context, std::uint32_t cached,
                                std::uint32_t concurrency, std::uint32_t chunk,
                                const KvCapacityPolicy& kv) {
        EngineOptions planned                     = resolved;
        planned.kv_capacity                      = kv;
        planned.max_context                      = max_context;
        planned.context_cache.device_state_slots = cached;
        planned.max_concurrency                  = concurrency;
        planned.prefill_chunk                    = chunk;
        return planned;
    };
    // An automatic Host tier is fitted to what the host can pin before anything is planned over it
    // (its KV pages widen the logical page tables).
    ContextCacheOptions& cache = resolved.context_cache;
    HostTierResolution host_tier{.automatic        = cache.enabled && cache.host_tier_automatic,
                                 .default_slots    = cache.host_state_slots,
                                 .default_kv_bytes = cache.host_kv_capacity_bytes};
    if (host_tier.automatic && device.props.integrated != 0) {
        // A GPU that shares the host's memory (DGX Spark's GB10, Jetson): a Host tier would pin
        // the memory the Device pool draws on and copy pages within it, so an automatic one is
        // off; an explicit one is kept.
        cache.host_state_slots       = 0;
        cache.host_kv_capacity_bytes = 0;
    }
    // The pinning a rung declares (rung_cost): the Host tier as the options and the platform
    // configure it. The fit below sizes an automatic tier to what the host can pin now
    // (MemAvailable, observed); were its page tables the rung's, host memory pressure would move
    // the ladder and the fingerprint with it.
    const std::uint64_t declared_host_kv_bytes    = cache.host_kv_capacity_bytes;
    const std::uint32_t declared_host_state_slots = cache.host_state_slots;
    if (host_tier.automatic && (cache.host_state_slots != 0 || cache.host_kv_capacity_bytes != 0)) {
        if (const std::optional<HostMemory> memory = read_host_memory()) {
            EngineOptions probe                     = resolved;
            probe.context_cache.device_state_slots =
                probe.context_cache.device_state_slots.value_or(
                    automatic_cached_device_states(options.max_concurrency));
            const std::size_t image_bytes =
                models::qwen3_5::make_sequence_planner(execution, device, probe)
                    .host_state_image_bytes();
            const auto budget = static_cast<std::uint64_t>(
                static_cast<long double>(pinned_host_limit(*memory)) * kAutomaticHostTierShare);
            const HostTierFit fit = fit_host_tier(image_bytes, cache.host_state_slots,
                                                  cache.host_kv_capacity_bytes, budget);
            cache.host_state_slots       = fit.state_slots;
            cache.host_kv_capacity_bytes = static_cast<std::size_t>(fit.kv_bytes);
            host_tier.reduced            = fit.reduced;
            host_tier.budget_bytes       = static_cast<std::size_t>(budget);
        }
    }
    // The cached Device states the ladder steps down to: none when the Host tier keeps StateImages
    // (the context cache then captures checkpoints to Host memory and restores them from there),
    // else one.
    // Whether a Host tier keeps StateImages is what the options and the platform configure (an
    // automatic tier's default slots, off on a GPU sharing the host's memory), not what the host
    // can pin now: the fit to MemAvailable sizes the tier (observed) and never moves the ladder,
    // so the fingerprint does not follow host memory pressure. A host that can pin no StateImage
    // then keeps no cached state beyond the active lanes on a rung without Device states, as
    // --device-state-slots 0 --host-state-slots 0 would.
    const bool host_states_configured =
        cache.enabled &&
        (host_tier.automatic ? host_tier.default_slots != 0 && device.props.integrated == 0
                             : cache.host_state_slots != 0);
    const std::uint32_t states_minimum = host_states_configured ? 0U : 1U;
    // The ladder's rungs set these from here on.
    const EngineOptions unfitted = resolved;
    // Whether the driver holds the plan's CUDA graph memory (the kernel warm-up below).
    bool graph_memory_held = false;
    // The longest decode round a request asks for: the widest launch window and its target
    // (plan_dry_run.h, the definition the dry run's axes share).
    const std::uint32_t round_tokens = plan_round_tokens(resolved.speculative);
    const bool kv_automatic          = resolved.kv_capacity.mode == KvCapacityMode::Automatic;
    // Whether a kernel warm-up measures what the plan's kernels take before the plan is final.
    const bool automatic_fit = context.automatic || resolved.prefill_chunk_automatic ||
                               kv_automatic || !configured_states.has_value();

    // The plan ladder (plan_ladder.h) of a concurrency: from the artifact (the context ceiling),
    // the options (each explicit one pins its axis), the family's planner (each candidate's
    // bytes) and nothing measured. `pins` false: every sizing axis but the chunk automatic (what
    // a refusal's advice fits instead).
    struct Ladder {
        std::vector<PlanRung> rungs;
        std::uint32_t concurrency = 1;
        KvCapacityPolicy kv;
        bool main = false; // the served ladder: an option its top rung cannot plan is refused
    };

    // A rung's planner (no device state is read). `held`: the driver holds the graphs' memory.
    const auto planner_at = [&](const Ladder& ladder, const PlanRung& rung, bool held) {
        return models::qwen3_5::make_sequence_planner(
            execution, device,
            planned_at(rung.max_context, rung.cached_states, ladder.concurrency, rung.prefill_chunk,
                       ladder.kv),
            held);
    };
    const auto pages_in = [](const Ladder& ladder, const PlanRung& rung,
                             const SequenceCapacityCurve& curve) {
        // An explicit --kv-capacity outside the plan's curve is refused by reservation_bytes,
        // as before; an automatic pool's pages are the curve's by construction.
        if (ladder.kv.mode == KvCapacityMode::Explicit) { return rung.kv_pages; }
        return std::clamp(rung.kv_pages, curve.minimum_main_page_groups,
                          curve.maximum_main_page_groups);
    };

    // Every rung cost computed once, by the shape it plans (the ladders share their rungs).
    struct CostKey {
        std::uint32_t context, pages, states, chunk, concurrency;
        bool kv_explicit, held;
        auto operator<=>(const CostKey&) const = default;
    };

    struct CachedCost {
        std::optional<PlanRungCost> cost;
        bool held_known = false; // cost->held_bytes planned (a second planner: only when read)
    };

    std::map<CostKey, CachedCost> rung_costs;
    // `graphs_held`: the driver holds the graphs' memory (no allowance); `with_pin`: the Host
    // tier's pinning is declared in the cost rather than measured; `top`: a shape the family
    // refuses is the user's option, refused as before (a lower rung's own shape is skipped);
    // `with_held`: plan held_bytes too (only a WarmUp admission reads it, of the last rung).
    const auto rung_cost = [&](const Ladder& ladder, const PlanRung& rung, bool graphs_held,
                               bool with_pin, bool top,
                               bool with_held) -> std::optional<PlanRungCost> {
        const CostKey key{rung.max_context,   rung.kv_pages,
                          rung.cached_states, rung.prefill_chunk,
                          ladder.concurrency, ladder.kv.mode == KvCapacityMode::Explicit,
                          graphs_held};
        auto found = rung_costs.find(key);
        if (found == rung_costs.end()) {
            std::optional<PlanRungCost> cost;
            try {
                const models::qwen3_5::SequencePlanner planner =
                    planner_at(ladder, rung, graphs_held);
                const SequenceCapacityCurve& curve = planner.capacity_curve();
                const std::uint32_t pages          = pages_in(ladder, rung, curve);
                PlanRungCost out;
                out.bytes      = curve.reservation_bytes(pages);
                out.held_bytes = graphs_held ? out.bytes : 0;
                if (cache.enabled) {
                    // The declared tier, never less than the fitted one: an automatic tier the
                    // host cannot pin in full pins less than the rung reserved for.
                    const std::uint64_t pinned =
                        declared_host_kv_bytes +
                        (declared_host_state_slots != 0
                             ? static_cast<std::uint64_t>(planner.host_state_image_bytes()) *
                                   declared_host_state_slots
                             : 0ULL);
                    out.pin_bytes = pinned >> 9; // 2 MiB of page tables per pinned GiB
                }
                cost = out;
            } catch (const std::invalid_argument&) { // CapacityError among them
                if (top) { throw; }
            }
            found =
                rung_costs.emplace(key, CachedCost{.cost = cost, .held_known = graphs_held}).first;
        }
        if (with_held && found->second.cost && !found->second.held_known) {
            PlanRungCost& cost = *found->second.cost;
            try {
                const models::qwen3_5::SequencePlanner held = planner_at(ladder, rung, true);
                const SequenceCapacityCurve& curve          = held.capacity_curve();
                cost.held_bytes = curve.reservation_bytes(pages_in(ladder, rung, curve));
            } catch (const std::invalid_argument&) {
                cost.held_bytes = cost.bytes; // never less than held: the graphs' bound
            }
            found->second.held_known = true;
        }
        std::optional<PlanRungCost> cost = found->second.cost;
        if (cost && !with_pin) { cost->pin_bytes = 0; }
        return cost;
    };
    const auto cost_of = [&](const Ladder& ladder, bool graphs_held, bool with_pin) {
        return PlanRungCostFn([&, graphs_held, with_pin, ladder_ptr = &ladder](
                                  std::size_t index) -> std::optional<PlanRungCost> {
            return rung_cost(*ladder_ptr, ladder_ptr->rungs.at(index), graphs_held, with_pin,
                             ladder_ptr->main && index == 0,
                             index + 1U == ladder_ptr->rungs.size());
        });
    };
    // What the selection keeps beside a rung (plan_ladder.h, PlanMargins). The class reserve of a
    // GPU Windows shares and of one sharing the host's memory applies to an automatic pool only;
    // an explicit --kv-capacity keeps none, as before.
    const bool paravirtual            = paravirtualized_gpu(device.device);
    const bool with_display           = drives_display(device.device);
    const std::uint64_t class_reserve = plan_class_reserve(
        device.props.totalGlobalMem, paravirtual, with_display, device.props.integrated != 0);
    // The ladder's axes and margins (plan_dry_run.h): the seams the dry run without the weights
    // (ninferno-serve --plan --no-load) shares with a start.
    const PlanAxes axes =
        plan_axes(unfitted, context.native, host_states_configured, class_reserve);
    if (axes.round_tokens != round_tokens || axes.kv_automatic != kv_automatic ||
        axes.states_pinned != configured_states || axes.states_minimum != states_minimum ||
        plan_margins(axes).allowance != (automatic_fit ? kPlanWarmUpAllowanceBytes : 0)) {
        throw std::logic_error("the plan axes disagree with the start's options");
    }
    const PlanMargins margins        = plan_margins(axes);
    const PlanMargins advice_margins = plan_advice_margins(axes);
    const auto ladder_for = [&](std::uint32_t concurrency, bool pins, bool chunk_pinned) {
        Ladder ladder{.concurrency = concurrency,
                      .kv          = pins ? resolved.kv_capacity
                                          : KvCapacityPolicy::automatic(
                                       kv_automatic ? resolved.kv_capacity.automatic_headroom_bytes
                                                    : kDefaultKvCapacityHeadroomBytes),
                      .main        = pins};
        const PlanLadderInputs inputs = plan_ladder_inputs(axes, concurrency, pins, chunk_pinned);
        const std::vector<PlanRung> candidates = plan_ladder(inputs);
        ladder.rungs = prune_plan_ladder(candidates, [&](std::size_t index) {
            const std::optional<PlanRungCost> cost = rung_cost(
                ladder, candidates[index], false, false, ladder.main && index == 0, false);
            return cost ? std::optional<std::uint64_t>(cost->bytes) : std::nullopt;
        });
        return ladder;
    };

    // The planning's own time (the ladder's planners and the selection), for the log only: no
    // decision reads it.
    const auto ladder_started = Clock::now();
    const Ladder served =
        ladder_for(resolved.max_concurrency, true, !resolved.prefill_chunk_automatic);
    Clock::duration planning_time = Clock::now() - ladder_started;
    if (served.rungs.empty()) { throw std::logic_error("the plan ladder has no rung"); }
    const auto apply_rung = [&](std::size_t index) {
        const PlanRung& rung   = served.rungs.at(index);
        resolved               = unfitted;
        resolved.max_context   = rung.max_context;
        resolved.prefill_chunk = rung.prefill_chunk;
        context.tokens         = rung.max_context;
    };
    // The longest context a concurrency's automatic ladder (every axis but the chunk automatic)
    // fits in `free_bytes`; nullopt: none.
    const auto longest_context = [&](std::uint32_t concurrency, std::uint64_t free_bytes,
                                     bool chunk_pinned) -> std::optional<std::uint32_t> {
        const Ladder ladder                    = ladder_for(concurrency, false, chunk_pinned);
        const std::optional<PlanSelection> fit = select_plan_rung(
            ladder.rungs, cost_of(ladder, graph_memory_held, true), free_bytes, advice_margins);
        if (!fit) { return std::nullopt; }
        return ladder.rungs[fit->rung].max_context;
    };
    // longest_context in `free_bytes`, as the advice functions the dry run shares read it
    // (plan_dry_run.h, plan_fits_instead).
    const auto longest_in = [&](std::uint64_t free_bytes) {
        return PlanLongestContextFn([&, free_bytes](std::uint32_t concurrency, bool chunk_kept) {
            return longest_context(concurrency, free_bytes, chunk_kept);
        });
    };
    // Each lower concurrency that fits a longer automatic context than `served_context` (nullopt:
    // nothing fits at the served concurrency), the highest first.
    const auto lower_concurrency_fits = [&](std::optional<std::uint32_t> served_context,
                                            std::uint64_t free_bytes, bool chunk_pinned) {
        return plan_lower_concurrency_fits(resolved.max_concurrency, served_context,
                                           unfitted.max_context, chunk_pinned,
                                           longest_in(free_bytes));
    };
    // No rung fits: the refusal names the last rung's need, the free memory and what fits
    // instead (a CapacityError: ninferno-serve's fallbacks recover by that type).
    const auto refuse = [&](std::uint64_t free_bytes) {
        const std::size_t last                = served.rungs.size() - 1U;
        const std::optional<PlanRungCost> low = cost_of(served, graph_memory_held, true)(last);
        const PlanReserve tier = margins.allowance != 0 ? PlanReserve::WarmUp : PlanReserve::Soft;
        const std::uint64_t needed = low ? plan_needed_bytes(*low, margins, tier) : 0;
        std::string message =
            "no rung of the " + std::to_string(served.rungs.size()) +
            "-rung plan ladder fits the " + gib_text(free_bytes) +
            " of Device memory free after the weights (" + gib_text(summary.host_to_device_bytes) +
            "): its last rung, " +
            describe_plan_rung(served.rungs[last], resolved.max_concurrency) +
            (low ? ", needs " + gib_text(needed) + ", " +
                       gib_text(needed > free_bytes ? needed - free_bytes : 0) + " more"
                 : std::string(", cannot be planned for this model")) +
            ".";
        // What fits instead: the longest context at this concurrency (when the context, the pool
        // or the states were asked for) and at each lower one that fits longer.
        const bool pinned = !context.automatic || !kv_automatic || configured_states.has_value();
        const bool chunk_pinned = !resolved.prefill_chunk_automatic;
        const PlanFitsInstead instead_fits =
            plan_fits_instead(resolved.max_concurrency, unfitted.max_context, pinned, chunk_pinned,
                              longest_in(free_bytes));
        std::vector<ConcurrencyFit> fits = instead_fits.fits;
        std::vector<std::uint32_t> chunks{unfitted.prefill_chunk};
        if (!chunk_pinned) {
            for (std::uint32_t chunk = unfitted.prefill_chunk / 2;
                 chunk >= kNarrowestAutomaticPrefillChunk; chunk /= 2) {
                if (chunk % kPrefillChunkQuantum == 0) { chunks.push_back(chunk); }
            }
        }
        std::uint32_t narrowed_from = 0;
        if (instead_fits.narrowed) {
            // An explicit --prefill-chunk nothing fits with: what the narrower chunks fit.
            narrowed_from = unfitted.prefill_chunk;
            chunks        = {kNarrowestAutomaticPrefillChunk};
        }
        const bool explicit_plan = !context.automatic && !resolved.prefill_chunk_automatic &&
                                   !kv_automatic && configured_states.has_value();
        const std::string instead = capacity_advice({
            .fits                    = fits,
            .chunks                  = chunks,
            .narrowed_from           = narrowed_from,
            .prefill_chunk_automatic = resolved.prefill_chunk_automatic,
            .kv_capacity_explicit    = !kv_automatic,
            .context_automatic       = context.automatic,
            .ceiling                 = unfitted.max_context,
            .max_concurrency         = resolved.max_concurrency,
            .graph_allowance = explicit_plan && resolved.use_cuda_graph && !graph_memory_held,
        });
        throw_capacity_shortage(CapacityError(message), instead);
    };
    // The resolution of a rung (its curve at its pages), against the free memory it was picked
    // in and the margin it keeps.
    const auto resolution_of = [&](std::size_t index, std::uint64_t free_bytes,
                                   PlanReserve reserve) {
        const PlanRung& rung = served.rungs.at(index);
        const models::qwen3_5::SequencePlanner planner =
            planner_at(served, rung, graph_memory_held);
        const SequenceCapacityCurve& curve = planner.capacity_curve();
        const std::uint32_t pages          = pages_in(served, rung, curve);
        const std::size_t reservation      = curve.reservation_bytes(pages);
        return KvCapacityResolution{
            .mode                                 = resolved.kv_capacity.mode,
            .main_page_groups                     = pages,
            .maximum_main_page_groups             = curve.maximum_main_page_groups,
            .resolved_tokens                      = curve.resolved_tokens(pages),
            .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
            .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
            .runtime_reservation_bytes            = reservation,
            .available_after_weights_bytes        = free_bytes,
            .automatic_headroom_bytes             = reserve == PlanReserve::Kept
                                                        ? std::max(margins.base, margins.reserve)
                                                        : margins.base,
            .planned_slack_bytes          = free_bytes > reservation ? free_bytes - reservation : 0,
            .cached_device_states         = rung.cached_states,
            .cached_device_states_default = configured_states.value_or(
                automatic_cached_device_states(resolved.max_concurrency)),
            .cached_device_states_automatic = !configured_states.has_value(),
        };
    };
    // The rung summary the logs, /v1/models and the dry run report.
    const auto rung_summary = [&](std::size_t index, const PlanRungCostFn& cost) {
        const PlanRung& rung = served.rungs.at(index);
        PlanRungSummary out{
            .max_context          = rung.max_context,
            .kv_tokens            = rung.kv_pages * kAutomaticContextGrain,
            .cached_device_states = rung.cached_states,
            .prefill_chunk        = rung.prefill_chunk,
            // A survival rung keeps the axis it stepped ("survival chunk").
            .step     = rung.survival && rung.step != PlanStep::Survival &&
                            rung.step != PlanStep::LastResort
                            ? "survival " + std::string(plan_step_name(rung.step))
                            : std::string(plan_step_name(rung.step)),
            .survival = rung.survival,
        };
        if (const std::optional<PlanRungCost> c = cost(index)) {
            out.bytes = c->bytes;
            // A survival rung gives the class reserve up (plan_ladder.h, select_plan_rung).
            out.needed_bytes = plan_needed_bytes(
                *c, margins, rung.survival ? PlanReserve::Soft : PlanReserve::Kept);
        }
        return out;
    };
    // `after_warm_up`: the free memory the warm-up's last check measured, when it moved the plan
    // down; the shortfalls are then the measured ones (the graphs held, the pinning taken) and the
    // class reserve is not blamed for the move.
    const auto plan_summary = [&](PlanSelection chosen, std::size_t selected_index,
                                  std::uint64_t free_bytes, bool with_ladder,
                                  std::optional<std::uint64_t> after_warm_up) {
        const PlanRungCostFn cost = cost_of(served, false, true);
        const PlanRungCostFn measured_cost = after_warm_up ? cost_of(served, true, false) : cost;
        const std::uint64_t measured_free  = after_warm_up.value_or(free_bytes);
        PlanSummary out;
        out.rung               = static_cast<std::uint32_t>(chosen.rung + 1U);
        out.rungs              = static_cast<std::uint32_t>(served.rungs.size());
        out.selected_rung      = static_cast<std::uint32_t>(selected_index + 1U);
        out.reserve            = std::string(plan_reserve_name(chosen.reserve));
        out.chosen             = rung_summary(chosen.rung, cost);
        out.device_total_bytes = device.props.totalGlobalMem;
        out.weight_bytes       = summary.host_to_device_bytes;
        out.free_bytes         = free_bytes;
        out.base_margin_bytes  = margins.base;
        out.reserve_bytes      = margins.reserve;
        out.allowance_bytes    = margins.allowance;
        out.planning_milliseconds = static_cast<std::uint32_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(planning_time).count());
        const auto shortfall = [&](std::size_t index) -> std::uint64_t {
            const std::optional<PlanRungCost> c = measured_cost(index);
            if (!c) { return 0; }
            PlanMargins checked = margins;
            if (after_warm_up) { checked.allowance = 0; } // what it was for is measured
            const std::uint64_t needed = plan_needed_bytes(
                *c, checked, served.rungs[index].survival ? PlanReserve::Soft : PlanReserve::Kept);
            return needed > measured_free ? needed - measured_free : 0;
        };
        // The rungs above that fit without the class reserve (the floor admitted Soft and below
        // never keep it): the first is the longest context the reserve alone holds back.
        std::optional<std::size_t> reserve_limited;
        if (chosen.rung > 0 && chosen.reserve == PlanReserve::Kept && !after_warm_up &&
            margins.reserve > margins.base) {
            const auto fits_soft = [&](std::size_t index) {
                const std::optional<PlanRungCost> c = cost(index);
                return c && plan_needed_bytes(*c, margins, PlanReserve::Soft) <= free_bytes;
            };
            if (fits_soft(chosen.rung - 1U)) {
                for (std::size_t index = 0; index < chosen.rung; ++index) {
                    if (fits_soft(index)) {
                        reserve_limited = index;
                        break;
                    }
                }
            }
        }
        // What the class reserve is kept for, in the user's words, and what giving it up risks.
        const bool shares_host        = device.props.integrated != 0;
        const std::string reserve_for = shares_host    ? "the host (the GPU shares its memory)"
                                        : with_display ? "the Windows desktop"
                                                       : "Windows";
        const std::string reserve_risk =
            shares_host ? "the host may run short of memory"
                        : "the desktop may stall when Windows pages the card out";
        if (chosen.rung > 0) {
            out.top_shortfall_bytes  = shortfall(0);
            out.next_shortfall_bytes = shortfall(chosen.rung - 1U);
        }
        if (reserve_limited) {
            const PlanRung& longer           = served.rungs[*reserve_limited];
            out.reserve_limited_context      = longer.max_context;
            out.reserve_limited_kv_tokens    = longer.kv_pages * kAutomaticContextGrain;
            const std::string context_option = "--max-context " + grouped_count(longer.max_context);
            const std::string kv_option =
                "--kv-capacity " + grouped_count(out.reserve_limited_kv_tokens);
            // At one lane an explicit --max-context makes the pool explicit (that context) and
            // keeps no reserve; with more lanes the pool must be asked for too.
            const std::string override_options = resolved.max_concurrency == 1
                                                     ? context_option + " (or " + kv_option + ")"
                                                     : context_option + " " + kv_option;
            const PlanRung& served_rung        = served.rungs[chosen.rung];
            const std::string limit =
                longer.max_context != served_rung.max_context
                    ? "the context to " + grouped_count(served_rung.max_context)
                    : "the plan to rung " + std::to_string(chosen.rung + 1U) + " (" +
                          describe_plan_rung(served_rung, resolved.max_concurrency) + ")";
            out.advice.push_back("the " + gib_text(margins.reserve) + " kept for " + reserve_for +
                                 " limits " + limit + "; " + override_options +
                                 " gives it up, and " + reserve_risk);
        } else if (chosen.rung > 0 && out.next_shortfall_bytes != 0) {
            out.advice.push_back(
                gib_text(out.next_shortfall_bytes) + " more free memory would serve rung " +
                std::to_string(chosen.rung) + " (" +
                describe_plan_rung(served.rungs[chosen.rung - 1U], resolved.max_concurrency) +
                "), " + gib_text(out.top_shortfall_bytes) + " the top rung (" +
                describe_plan_rung(served.rungs[0], resolved.max_concurrency) + ")");
        }
        if (chosen.reserve != PlanReserve::Kept && margins.reserve > margins.base) {
            out.advice.push_back("the " + gib_text(margins.reserve) + " this GPU keeps for " +
                                 reserve_for + " is not kept: " + reserve_risk +
                                 "; a shorter --max-context or a lower --max-concurrency keeps it");
        }
        if (served.rungs[chosen.rung].survival &&
            served.rungs[chosen.rung].max_context <
                std::min(unfitted.max_context, kPlanContextFloor)) {
            out.advice.push_back("the context of " +
                                 grouped_count(served.rungs[chosen.rung].max_context) +
                                 " tokens is below the " +
                                 grouped_count(std::min(unfitted.max_context, kPlanContextFloor)) +
                                 "-token floor: long prompts are refused with HTTP 400");
        }
        if (with_ladder) {
            for (std::size_t index = 0; index < served.rungs.size(); ++index) {
                out.ladder.push_back(rung_summary(index, cost));
            }
        }
        return out;
    };

    // The warm-up below starts the per-thread stack from zero and measures what the plan's kernels
    // raise it to. It is dropped before the measurement too: loading raised it (the driver's
    // default times every resident thread, ~120 MiB on an RTX 5080 against the plan's 48 B per
    // thread), and a selection that counted it as used would pick a lower rung than the warm-up
    // then finds room for. The load-time route tuner times on buffers of its own and frees them:
    // before the stack limit and the free memory are read. Exact activations run the invariant
    // routes, which the tuner does not touch; the dry run does not tune.
    if (options.activations != ActivationMode::Exact && dry_run == nullptr) {
        summary.moe_tune = execution.tune_routes(device.stream);
    }
    // The generic path's dense tuner times each dense problem at its first eager call (the
    // warm-up's prefill), before a decode graph is captured with it (ops/generic/generic_tune.cpp).
    if (dry_run == nullptr) { ops::detail::generic::enable_dense_tune(true); }
    const std::size_t stack_limit_at_load = current_stack_limit_bytes();
    if (automatic_fit) { CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 0)); }
    // The one observation: the free Device memory after the weights (the target's and a separate
    // drafter's, both uploaded by the family load) picks the first rung that fits.
    const std::size_t free_after_weights = current_free_device_bytes();
    const auto select_started            = Clock::now();
    const std::optional<PlanSelection> selected =
        select_plan_rung(served.rungs, cost_of(served, false, true), free_after_weights, margins);
    planning_time += Clock::now() - select_started;
    if (dry_run != nullptr) {
        if (!selected) {
            *dry_run =
                plan_summary({.rung = served.rungs.size() - 1U, .reserve = PlanReserve::WarmUp},
                             served.rungs.size() - 1U, free_after_weights, true, std::nullopt);
            dry_run->rung          = 0; // nothing fits
            dry_run->selected_rung = 0;
            dry_run->reserve.clear();
            // The refusal alone: no reserve advice for a plan that is not served.
            dry_run->advice.clear();
            try {
                refuse(free_after_weights);
            } catch (const CapacityError& refusal) { dry_run->advice.push_back(refusal.what()); }
        } else {
            *dry_run =
                plan_summary(*selected, selected->rung, free_after_weights, true, std::nullopt);
            if (served.rungs[selected->rung].max_context <
                    std::min(unfitted.max_context, kPlanContextFloor) &&
                resolved.max_concurrency > 1 && context.automatic) {
                for (const ConcurrencyFit& fit : lower_concurrency_fits(
                         served.rungs[selected->rung].max_context, free_after_weights,
                         !resolved.prefill_chunk_automatic)) {
                    dry_run->advice.push_back("--max-concurrency " +
                                              std::to_string(fit.max_concurrency) + " fits " +
                                              grouped_count(fit.max_context) + " tokens");
                }
            }
        }
        if (automatic_fit) {
            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack_limit_at_load));
        }
        return {};
    }
    if (!selected) { refuse(free_after_weights); }
    PlanSelection chosen = *selected;
    // A WarmUp admission was made on the held bytes: the plan holds the graphs from here on, and
    // the warm-up's probe measures them (plan_ladder.h, plan_graphs_held). Otherwise every check
    // before the probe keeps the planner's graph allowance.
    graph_memory_held = plan_graphs_held(chosen, false);
    apply_rung(chosen.rung);
    KvCapacityResolution resolution =
        resolution_of(chosen.rung, free_after_weights, chosen.reserve);
    // The kernel warm-up, before the plan is final. CUDA loads a kernel's code into Device memory
    // at its first launch (lazy module loading), and the driver grows the local memory it keeps for
    // every resident thread (cudaLimitStackSize) at the first launch of a kernel with a larger
    // frame; neither is given back. The warm-up starts the stack limit from zero, so it ends at
    // the largest frame of the plan's kernels. Which kernels run is what the routes choose for this
    // model and plan, so construction runs them: a probe Program with the rung's concurrency,
    // speculation, KV formats and prefill chunk takes one request of every prefill width 1..chunk
    // through the request path (admission, the prefill with its speculative and sampling work, the
    // first token and, once, a decode round), and a probe Program at the rung's context captures
    // and launches every decode graph class of it. The selection kept kPlanWarmUpAllowanceBytes and
    // the planned graph allowance for this; the free memory left is then measured, and the rung
    // stays when it still fits, else the plan moves to the first lower rung that does (never to a
    // higher one: re-measuring cannot move the plan above the declared selection). The probes hold
    // one request's KV and no cached states or Host tier, so they fit wherever the plan does. The
    // driver keeps a destroyed graph's memory for the next graph, so once the probe built the
    // rung's graph set, the served Program's identical set takes that memory again and the plan
    // reserves no graph allowance for it.
    KernelWarmUpResolution warm_up;
    // The free memory the warm-up's last check measured (the plan summary's shortfalls after a
    // move).
    std::uint64_t last_measured = free_after_weights;
    if (automatic_fit) {
        StartupPhaseScope warming(options.startup_observer, StartupPhase::KernelWarmUp);
        const auto warm_started            = Clock::now();
        const StartupObserver unobserved   = {};
        warm_up.ran                        = true;
        warm_up.stack_limit_before_bytes   = stack_limit_at_load;
        // The warm-up starts from no per-thread stack (the limit is the process's, for the device):
        // the driver raises it at the first launch of each larger frame, so after the warm-up it is
        // the largest frame among the kernels this plan launched (S), and the local memory kept for
        // it is S times every resident thread instead of the driver's default times every thread.
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 0));
        std::size_t stack_limit            = current_stack_limit_bytes();
        const std::size_t free_before_warm = current_free_device_bytes();
        const auto note_stack_limit        = [&](std::string step) {
            const std::size_t now = current_stack_limit_bytes();
            if (now > stack_limit) {
                stack_limit                = now;
                warm_up.stack_limit_source = std::move(step);
            }
        };
        std::size_t probe_buffer_bytes   = 0;
        const auto probe_program = [&](std::uint32_t max_context) {
            const std::uint32_t kv_tokens = std::max(
                max_context,
                resolved.max_concurrency * static_cast<std::uint32_t>(kPagedKVPageSize));
            EngineOptions probe = planned_at(max_context, 0, resolved.max_concurrency,
                                             resolved.prefill_chunk,
                                             KvCapacityPolicy::explicit_capacity(kv_tokens));
            probe.context_cache = ContextCacheOptions{.enabled = false};
            probe               = normalize_engine_options(std::move(probe));
            // Planned as with the graphs held: its reservation is the probe's buffers alone.
            models::qwen3_5::SequencePlanner planner =
                models::qwen3_5::make_sequence_planner(execution, device, probe, true);
            const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
            probe_buffer_bytes        = planner.capacity_curve().reservation_bytes(pages);
            return models::qwen3_5::create_program(execution, std::move(planner).finalize(pages),
                                                   device, unobserved);
        };
        {
            const std::uint32_t probe_context =
                std::min(resolved.max_context, resolved.prefill_chunk + round_tokens + 1U);
            if (probe_context <= round_tokens + 1U) {
                // The rung left too short a context for the widest round: plain decode (or a
                // shallower speculation) needs less, so this is a capacity refusal.
                throw CapacityError(
                    "the plan's context of " + std::to_string(probe_context) +
                    " tokens does not hold one prompt token and one decode round of " +
                    std::to_string(round_tokens) + " tokens");
            }
            const std::uint32_t widths =
                std::min(resolved.prefill_chunk, probe_context - round_tokens - 1U);
            std::unique_ptr<models::qwen3_5::Program> program = probe_program(probe_context);
            note_stack_limit("the decode graphs at a " + std::to_string(probe_context) +
                             "-token context");
            const models::qwen3_5::Frontend frontend = loaded.make_frontend(probe_context);
            const std::vector<TokenId> text          = frontend.tokenize_text(kWarmUpText);
            if (text.empty()) { throw std::logic_error("the kernel warm-up text has no tokens"); }
            // One request per kernel class, not per width: at 1..48 every T is its own class
            // (the per-exact-T schedules); above that the band edges, the full- and partial-tile
            // widths inside each band and the exported plans' classes cover every kernel variant
            // the plan can launch at any width in 1..chunk (warmup_widths.h).
            const std::vector<std::uint32_t> width_set =
                widths <= detail::kWarmUpExactWidths
                    ? [&] {
                          std::vector<std::uint32_t> all(widths);
                          for (std::uint32_t w = 1; w <= widths; ++w) { all[w - 1] = w; }
                          return all;
                      }()
                    : detail::kernel_warm_up_widths(widths);
            warm_up.widths_run.reserve(width_set.size());
            for (const std::uint32_t width : width_set) {
                std::vector<TokenId> prompt(width);
                for (std::uint32_t i = 0; i < width; ++i) { prompt[i] = text[i % text.size()]; }
                warm_up_request(*program, frontend, std::move(prompt), round_tokens, width == 1);
                note_stack_limit("a prefill of " + std::to_string(width) + " tokens");
                warm_up.widths_run.push_back(width);
            }
            warm_up.prefill_widths = widths;
            program.reset();
            device.synchronize();
        }
        // The context cache's Host tier is pinned host memory, and the driver maps every pinned
        // byte for the device: the page tables take device memory (2 MiB per pinned GiB),
        // after the plan, when the Program pins. The selection declared it; here it is measured by
        // pinning buffers of the rung's two Host tier sizes (its StateImages, its KV), and a rung
        // whose sizes differ from the measured ones is checked again.
        const auto host_tier_sizes = [&]() -> std::pair<std::size_t, std::size_t> {
            const ContextCacheOptions& tier = resolved.context_cache;
            const std::size_t states =
                tier.enabled && tier.host_state_slots != 0
                    ? planner_at(served, served.rungs[chosen.rung], graph_memory_held)
                              .host_state_image_bytes() *
                          tier.host_state_slots
                    : 0;
            return {states, tier.enabled ? tier.host_kv_capacity_bytes : 0};
        };
        const auto pinning_device_bytes = [&](std::pair<std::size_t, std::size_t> sizes) {
            const std::size_t before = current_free_device_bytes();
            std::optional<PinnedHostBuffer> states;
            std::optional<PinnedHostBuffer> kv;
            if (sizes.first != 0) { states.emplace(sizes.first); }
            if (sizes.second != 0) { kv.emplace(sizes.second); }
            const std::size_t after = current_free_device_bytes();
            return before > after ? before - after : std::size_t{0};
        };
        std::optional<std::pair<std::size_t, std::size_t>> pinned_sizes;
        std::uint32_t probed = 0;
        // The graph memory the last probe's destruction gave back beyond its buffers: a driver
        // that keeps it gives back none; WDDM (WSL 2) gave back all of it, and the served
        // Program then took it again past the plan (Qwen3.8-27B DFlash2 C=4: 0.8 GiB, the card
        // filled to 200 MiB of a planned 1 GiB headroom).
        std::size_t returned_graph_bytes = 0;
        // The graph memory a probe's capture needed beyond the plan's allowance. The allowance is
        // a bound measured on 4090/5080-class cards; an RTX 2080 Ti's DFlash2 graphs at C=8 took
        // more (Qwen3.8-27B IQ2_XXS, fp32 GDN state: 4 MiB left at the probe's instantiation, after
        // the four-minute width loop). A probe that does not fit grows this reserve, and the check
        // below moves the plan to a lower rung under it, so only the probe is repeated.
        std::size_t graph_shortfall_bytes = 0;
        for (std::uint32_t pass = 0;; ++pass) {
            // A capacity outcome, not a program error: ninferno-serve's fallbacks (a shallower or
            // no drafter, a smaller pool) recover from a CapacityError only.
            if (pass > kMaximumWarmUpPasses) {
                throw CapacityError(
                    "the kernel warm-up's plan did not settle in " + std::to_string(pass) +
                    " passes (rung " + std::to_string(chosen.rung + 1U) + " of " +
                    std::to_string(served.rungs.size()) + ", " + gib_text(last_measured) +
                    " of Device memory free beside its reserves)");
            }
            const std::size_t free_now = current_free_device_bytes();
            const std::size_t taken    = warm_up.host_pin_device_bytes + returned_graph_bytes +
                                      graph_shortfall_bytes;
            const std::uint64_t measured = free_now > taken ? free_now - taken : 0;
            last_measured                = measured;
            // The chosen rung keeps its admission tier (a WarmUp rung: its held bytes and the
            // base margin), so a probe's 6-40 MiB never refuse it before a graph is measured.
            const std::optional<PlanSelection> kept =
                plan_warm_up_check(served.rungs, cost_of(served, graph_memory_held, false),
                                   free_now, taken, margins, chosen);
            if (!kept) { refuse(measured); }
            chosen = *kept;
            // A move to the last rung on its held bytes holds the graphs from here on.
            graph_memory_held = plan_graphs_held(chosen, probed != 0);
            apply_rung(chosen.rung);
            resolution = resolution_of(chosen.rung, measured, chosen.reserve);
            if (const auto sizes = host_tier_sizes(); sizes != pinned_sizes) {
                warm_up.host_pin_device_bytes = pinning_device_bytes(sizes);
                pinned_sizes                  = sizes;
                continue;
            }
            // A context no longer than the probed one is settled: its graph classes are the
            // probed context's, cut at the same ranges, and take no more memory.
            if (probed != 0 && resolved.max_context <= probed) { break; }
            ++warm_up.graph_passes;
            const std::size_t free_before_probe = current_free_device_bytes();
            std::unique_ptr<models::qwen3_5::Program> program;
            try {
                program = probe_program(resolved.max_context);
            } catch (const CapacityError&) {
                // Only a ladder with a lower rung can make room; the last rung refuses as before.
                if (chosen.rung + 1U >= served.rungs.size()) { throw; }
                device.synchronize();
                graph_shortfall_bytes = graph_shortfall_bytes == 0 ? kGraphShortfallStepBytes
                                                                   : 2 * graph_shortfall_bytes;
                warm_up.graph_shortfall_bytes = graph_shortfall_bytes;
                continue;
            }
            note_stack_limit("the decode graphs at a " + std::to_string(resolved.max_context) +
                             "-token context");
            device.synchronize();
            const std::size_t free_with_probe = current_free_device_bytes();
            program.reset();
            device.synchronize();
            const std::size_t free_after_probe = current_free_device_bytes();
            const std::size_t given_back =
                free_after_probe > free_with_probe ? free_after_probe - free_with_probe : 0;
            // Never more than the probe took; the largest of the probes', so the plan only
            // moves down as the probes go.
            const std::size_t returned = std::min(
                given_back > probe_buffer_bytes ? given_back - probe_buffer_bytes : std::size_t{0},
                free_before_probe > free_with_probe ? free_before_probe - free_with_probe
                                                    : std::size_t{0});
            returned_graph_bytes         = std::max(returned_graph_bytes, returned);
            warm_up.graph_returned_bytes = returned_graph_bytes;
            probed = resolved.max_context;
            // The driver keeps the destroyed graphs' memory, and the served Program's graphs (this
            // context's set) take it again: the checks from here on reserve no graph allowance.
            graph_memory_held = plan_graphs_held(chosen, true);
        }
        const std::size_t free_after = current_free_device_bytes();
        warm_up.lasting_bytes = free_before_warm > free_after ? free_before_warm - free_after : 0;
        warm_up.stack_limit_bytes = current_stack_limit_bytes();
        warm_up.seconds = std::chrono::duration<double>(Clock::now() - warm_started).count();
        warming.complete();
    }
    // The lower concurrencies that fit a longer automatic context, for the operator, when the
    // rung's context is below the floor.
    if (context.automatic && kv_automatic && resolved.max_concurrency > 1 &&
        resolved.max_context < std::min(unfitted.max_context, kPlanContextFloor)) {
        context.lower_concurrency_fits = lower_concurrency_fits(
            resolved.max_context, free_after_weights, !resolved.prefill_chunk_automatic);
    }
    PlanSummary plan = plan_summary(
        chosen, selected->rung, free_after_weights, false,
        chosen.rung != selected->rung ? std::optional<std::uint64_t>(last_measured) : std::nullopt);
    for (const ConcurrencyFit& fit : context.lower_concurrency_fits) {
        plan.advice.push_back("--max-concurrency " + std::to_string(fit.max_concurrency) +
                              " fits " + grouped_count(fit.max_context) + " tokens");
    }
    auto sequence = planner_at(served, served.rungs[chosen.rung], graph_memory_held)
                        .finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("the plan's KV capacity does not match the finalized Program plan");
    }
    planning.complete();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    models::qwen3_5::Frontend served_frontend = loaded.make_frontend(resolved.max_context);
    loaded.summary.chat_template_source       = served_frontend.chat_template_source();
    frontend.complete();
    const DecodeRoundShape round_shape{.experts           = loaded.routed_experts,
                                       .experts_per_token = loaded.experts_per_token};
    auto instance = std::make_unique<ModelInstance>(std::move(loaded), std::move(served_frontend),
                                                    resolved.max_context);
    instance->decode_round_shape     = round_shape;
    instance->kv_capacity_resolution = resolution;
    instance->host_tier              = host_tier;
    instance->context                = context;
    instance->kernel_warm_up         = warm_up;
    instance->plan                   = std::move(plan);
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(*instance->execution, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const models::qwen3_5::DecodePacingFacts pacing     = instance->program->decode_pacing();
    summary.decode_steps_per_round                      = pacing.steps_per_round;
    summary.cpu_quota_us                                = pacing.cpu_quota_us;
    summary.cpu_quota_period_us                         = pacing.cpu_period_us;
    instance->decode_round_shape.steps_per_round        = pacing.steps_per_round;
    instance->decode_round_shape.verify_tokens_per_lane = pacing.verify_tokens_per_lane;
    for (auto& line : ops::detail::generic::dense_tune_report()) {
        summary.moe_tune.push_back(std::move(line));
    }
    summary.gpu_class    = ops::route_gpu_class();
    summary.device_facts = device_fact_notes(device_class(options.device));
    summary.route_table  = ops::route_table_status().message;
    summary.load_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    summary.context_cost = std::move(context_cost.summary);
    summary.metadata = model_metadata(options.artifact_path, summary, instance->execution->traits(),
                                      summary.artifact_id, summary.source_identity,
                                      instance->decode_round_shape.routed_expert_bytes);
    instance->decode_round_shape.weight_bytes = summary.metadata.weight_bytes;
    return {std::move(instance), std::move(summary), std::move(context_cost.model),
            std::move(resolved)};
}

} // namespace

ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device,
                                 FamilyLoader load) {
    return construct_or_plan(options, device, load, nullptr);
}

PlanSummary plan_model(const EngineOptions& options, DeviceContext& device) {
    validate_engine_options(options);
    PlanSummary plan;
    (void)construct_or_plan(options, device,
                            family_loader(artifact_model_type(options.artifact_path)), &plan);
    return plan;
}

} // namespace ninferno::runtime
