#include "runtime/engine/model_instance.h"
#include "runtime/engine/family_registry.h"
#include "runtime/engine/warmup_widths.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/arena.h"
#include "core/device_class.h"
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
#include "ops/host_experts/host_experts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
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

// The GPU is shared through WDDM, whose paging spills an over-full device to host memory instead
// of failing: natively on Windows a card in WDDM mode (not TCC), and under WSL 2 every GPU (it
// reaches the Windows host's through /dev/dxg).
bool paravirtualized_gpu(int device) {
#ifdef _WIN32
    int tcc = 0;
    return cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, device) != cudaSuccess || tcc == 0;
#else
    (void)device;
    static const bool present = [] {
        std::error_code error;
        return std::filesystem::exists("/dev/dxg", error);
    }();
    return present;
#endif
}

// The GPU drives a display: the host's kernel-run watchdog is on (a headless card has it off).
bool drives_display(int device) {
    int watchdog = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&watchdog, cudaDevAttrKernelExecTimeout, device));
    return watchdog != 0;
}

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
// Fits after the prefill probe before the fitted context and the Host tier's pinned sizes stop
// moving (each probe Program and each pinning measurement only settles them); a bound this far
// above three means a fault.
constexpr std::uint32_t kMaximumWarmUpPasses = 8;
// The first step of the reserve a probe whose graphs did not fit adds to the fit (doubled per retry).
constexpr std::size_t kGraphShortfallStepBytes = std::size_t{128} << 20;
// What a refusal made before the kernel warm-up keeps back for it when it names what fits instead:
// the warm-up's lasting memory (kernel code, the per-thread stack, the decode graphs; 6-40 MiB
// measured on an RTX 5080) is not yet in the free memory the advice is fitted to. An advised plan
// fitted without it was refused 18 MiB short once warmed (Qwen3.6-35B IQ3_XXS at 4 x 16,960).
constexpr std::size_t kAdviceWarmUpReserveBytes = 64ULL << 20;

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
    if (options.moe_offload.mode == MoeOffloadMode::Layers && options.moe_offload.layers == 0) {
        throw std::invalid_argument("Engine moe_offload of a layer count needs at least one layer");
    }
    if (options.moe_offload.mode == MoeOffloadMode::Off &&
        (options.moe_offload.compute != MoeOffloadCompute::Automatic ||
         options.moe_offload.threads != 0)) {
        throw std::invalid_argument(
            "Engine moe_offload compute and threads apply only with moe_offload on");
    }
    if (options.moe_offload.compute == MoeOffloadCompute::Gpu &&
        options.moe_offload.threads != 0) {
        throw std::invalid_argument(
            "Engine moe_offload threads apply only to CPU expert compute, not gpu");
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

    // An unconfigured cached Device state count stays unset here: construct_model chooses it with
    // the KV capacity from the free memory (resolve_runtime_capacity), and the Engine reports it.
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
    auto plan =
        models::qwen3_5::plan_load(reader, models::load_options(options), options.moe_offload);
    binding.complete();
    auto loaded = std::make_unique<Qwen35LoadedModel>(
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer));
    device.synchronize();
    const models::qwen3_5::Model& model = *loaded->model;
    FamilyLoad out;
    out.expert_residency     = model.expert_residency();
    out.device_expert_bytes  = model.storage_stats().expert_device_bytes;
    out.device_expert_layers = static_cast<std::uint32_t>(model.storage_stats().expert_device_layers);
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
ModelMetadata model_metadata(const std::filesystem::path& artifact_path, const LoadSummary& summary,
                             const models::qwen3_5::detail::ModelTraits& traits) {
    ModelMetadata metadata;
    metadata.model_id       = summary.model_name;
    metadata.vocab_size     = static_cast<std::uint64_t>(traits.public_tokens);
    metadata.embedding_size = static_cast<std::uint64_t>(traits.hidden);
    metadata.native_context = traits.max_positions;
    const artifact::Reader reader(artifact_path);
    std::set<std::string> formats;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; } // resources: tokenizer, templates
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

ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device,
                                 FamilyLoader load) {
    validate_engine_options(options);
    const auto start    = Clock::now();
    FamilyLoad loaded   = load(options, device);
    LoadSummary summary = std::move(loaded.summary);
    if (!loaded.execution || !loaded.make_frontend) {
        throw std::logic_error("a family load must carry its execution and its frontend");
    }
    const models::qwen3_5::detail::ModelExecution& execution = *loaded.execution;
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    auto context_cost = resolve_context_machine_cost(
        {.hardware_class =
             context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
         .prefill_signature = summary.prefill_signature,
         .sms               = static_cast<std::uint32_t>(device.props.multiProcessorCount)},
        options.context_cost.preset_path);
    EngineOptions resolved = options;
    resolved.kv_capacity = with_unified_memory_headroom(
        with_paravirtual_headroom(resolved.kv_capacity, paravirtualized_gpu(device.device),
                                  drives_display(device.device), device.props.totalGlobalMem),
        device.props.integrated != 0, device.props.totalGlobalMem);
    ContextResolution context{
        .automatic = options.max_context_automatic,
        .native    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            execution.traits().max_positions, std::numeric_limits<std::uint32_t>::max())),
        .tokens    = options.max_context,
    };
    // An automatic max_context starts from the model's own (under the cap, and within an explicit
    // KV capacity, which bounds one sequence).
    if (context.automatic) {
        if (context.native == 0) {
            throw std::logic_error("the model reports no position capacity");
        }
        resolved.max_context = context.native;
        if (options.max_context != 0) {
            resolved.max_context = std::min(resolved.max_context, options.max_context);
        }
        if (options.kv_capacity.mode == KvCapacityMode::Explicit) {
            resolved.max_context =
                std::min(resolved.max_context, options.kv_capacity.explicit_tokens);
        }
    }
    const CachedStatePolicy cached_states{
        .configured = options.context_cache.enabled ? options.context_cache.device_state_slots
                                                    : std::optional<std::uint32_t>(0U),
        .automatic_default = automatic_cached_device_states(options.max_concurrency),
    };
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
    const auto planned_options = [&](std::uint32_t max_context, std::uint32_t cached) {
        return planned_at(max_context, cached, resolved.max_concurrency, resolved.prefill_chunk,
                          resolved.kv_capacity);
    };
    // MoE expert offload: the loaded groups are placed before anything else is planned or pinned
    // (the Host tier below fits what the host can pin after them). Automatic keeps the fewest
    // leading MoE layers in Host memory with which the plan at the target context fits: the
    // automatic context's floor, or an explicit max_context.
    MoeOffloadResolution offload{.mode = options.moe_offload.mode};
    std::unique_ptr<ops::HostExperts> host_experts;
    if (ExpertResidency* experts = loaded.expert_residency) {
        const std::size_t groups = experts->group_count();
        std::size_t offloaded    = groups; // Layers: exactly the groups the load placed
        if (options.moe_offload.mode == MoeOffloadMode::Automatic) {
            const std::uint32_t target = context.automatic
                                             ? std::min(resolved.max_context, kAutomaticContextFloor)
                                             : resolved.max_context;
            offload.target_context      = target;
            const std::size_t available = current_free_device_bytes();
            const auto fits             = [&](std::size_t count) {
                const std::uint64_t placed = experts->device_bytes(count);
                return placed < available &&
                       try_resolve_runtime_capacity(
                           resolved.kv_capacity, cached_states,
                           [&](std::uint32_t states) {
                               return models::qwen3_5::make_sequence_planner(
                                          execution, device, planned_options(target, states))
                                   .capacity_curve();
                           },
                           static_cast<std::size_t>(available - placed))
                           .has_value();
            };
            std::optional<std::size_t> chosen;
            for (std::size_t count = 0; count <= groups && !chosen; ++count) {
                if (fits(count)) { chosen = count; }
            }
            if (!chosen) {
                // A free-memory refusal like the KV fit's: a caller may retry with less (a plain
                // engine without its drafter, a shorter context).
                throw CapacityError(
                    "MoE expert offload: the model does not fit even with the experts of all " +
                    std::to_string(groups) + " MoE layers (" + format_gib(experts->total_bytes()) +
                    ") in Host memory: " + format_gib(available) +
                    " of Device memory is free after the other weights, the expert staging "
                    "slots take " + format_gib(experts->staging_bytes(groups)) +
                    ", and the rest does not hold the plan for " + std::to_string(target) +
                    " tokens of context (KV, cached states, activations). Ask for a shorter "
                    "context (--max-context) or a lower concurrency.");
            }
            offloaded = *chosen;
        }
        experts->resolve(offloaded, device.device);
        offload.moe_layers     = static_cast<std::uint32_t>(groups);
        offload.offloaded      = static_cast<std::uint32_t>(offloaded);
        offload.pinned_bytes   = experts->pinned_bytes(offloaded);
        offload.staging_bytes  = experts->staging_bytes(offloaded);
        offload.resident_bytes = experts->device_bytes(offloaded) - offload.staging_bytes;

        // Where decode calls compute the offloaded experts: the CPU (the fast mode) or the GPU
        // over PCIe (byte-equal to no offload). An explicit choice that cannot hold is refused.
        const MoeOffloadCompute asked = options.moe_offload.compute;
        const bool exact              = options.activations == ActivationMode::Exact;
        const bool family             = static_cast<bool>(loaded.attach_host_experts);
        std::string unreadable;
        const bool readable =
            family && offloaded > 0 && loaded.host_experts_readable(&unreadable);
        if (asked == MoeOffloadCompute::Cpu) {
            if (exact) {
                throw std::invalid_argument(
                    "--moe-offload-compute cpu is the fast mode (the CPU's results are close to "
                    "the GPU's, not byte-equal), and --activations exact asks for bytes that do "
                    "not depend on where the experts are: drop one of the two");
            }
            if (!family) {
                throw std::invalid_argument(
                    "--moe-offload-compute cpu: this model family computes offloaded experts on "
                    "the GPU only (--moe-offload-compute gpu or auto)");
            }
            if (offloaded > 0 && !readable) {
                throw std::invalid_argument("--moe-offload-compute cpu: the CPU cannot compute "
                                            "these experts (" + unreadable +
                                            "); --moe-offload-compute gpu reads them over PCIe");
            }
        }
        bool on_cpu = false;
        if (offloaded == 0) {
            offload.compute_reason = "no MoE layer offloaded";
        } else if (asked == MoeOffloadCompute::Gpu) {
            offload.compute_reason = "--moe-offload-compute gpu";
        } else if (asked == MoeOffloadCompute::Cpu) {
            on_cpu                 = true;
            offload.compute_reason = "--moe-offload-compute cpu";
        } else if (exact) {
            offload.compute_reason = "auto: --activations exact keeps them on the GPU, byte-equal "
                                     "to no offload";
        } else if (!family) {
            offload.compute_reason = "auto: this model family has no CPU expert path";
        } else if (!readable) {
            offload.compute_reason = "auto: the CPU cannot compute them (" + unreadable + ")";
        } else {
            on_cpu                 = true;
            offload.compute_reason = "auto: --activations fast";
        }
        if (!on_cpu && options.moe_offload.threads != 0) {
            throw std::invalid_argument("--moe-offload-threads sets the CPU's expert threads, "
                                        "but the GPU computes the offloaded experts (" +
                                        offload.compute_reason + ")");
        }
        if (on_cpu) {
            host_experts = std::make_unique<ops::HostExperts>(
                loaded.host_experts_hidden, loaded.host_experts_top_k,
                static_cast<int>(options.moe_offload.threads), ops::host::detect_isa());
            loaded.attach_host_experts(host_experts.get());
            offload.compute     = MoeOffloadCompute::Cpu;
            offload.cpu_threads = static_cast<std::uint32_t>(host_experts->threads());
            offload.cpu_isa     = ops::host::isa_name(host_experts->isa());
        }
    }
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
    // The cached Device states an automatic count may go down to: none when the Host tier keeps
    // StateImages (the context cache then captures checkpoints to Host memory and restores them
    // from there), else one.
    CachedStatePolicy sizing_states = cached_states;
    sizing_states.automatic_minimum = cache.enabled && cache.host_state_slots != 0 ? 0U : 1U;
    // The fit starts from these each time it is made (after the kernel warm-up, once more).
    const EngineOptions unfitted             = resolved;
    const ContextResolution unfitted_context = context;
    // One planner per cached Device state count the resolution asks about; the chosen one plans.
    std::map<std::uint32_t, models::qwen3_5::SequencePlanner> planners;
    // Whether the driver holds the fitted plan's CUDA graph memory (the kernel warm-up below).
    bool graph_memory_held = false;
    const auto planner_for = [&](std::uint32_t count) -> models::qwen3_5::SequencePlanner& {
        auto found = planners.find(count);
        if (found == planners.end()) {
            found = planners
                        .emplace(count, models::qwen3_5::make_sequence_planner(
                                            execution, device,
                                            planned_options(resolved.max_context, count),
                                            graph_memory_held))
                        .first;
        }
        return found->second;
    };
    // A shortage names what MoE expert offload could still take off the device.
    const auto offload_hint = [&]() -> std::string {
        if (loaded.device_expert_bytes == 0 || options.moe_offload.mode == MoeOffloadMode::Automatic) {
            return {};
        }
        return " The routed experts of " + std::to_string(loaded.device_expert_layers) +
               " MoE layers take " + format_gib(loaded.device_expert_bytes) +
               " of the weights on the GPU: MoE expert offload (ninferno-serve --moe-offload auto) "
               "keeps the fewest of them in pinned Host memory that let the plan fit.";
    };
    // The fit: every choice below depends on the free Device memory and the plans only.
    // `warmed`: the free memory is measured after the kernel warm-up and net of the Host tier's
    // pinning (the fits after the warm-up); before it, a refusal's advice keeps both back.
    const auto fit_plan = [&](const std::size_t free_bytes, bool warmed) -> KvCapacityResolution {
        resolved = unfitted;
        context  = unfitted_context;
        planners.clear();
        // The free memory the fits below are made against; a refusal's advice lowers it.
        std::size_t fit_free = free_bytes;
        // The prefill chunks a choice may take: the configured one, and when the Engine chooses it,
        // each half of it down to the narrowest (a narrower chunk takes less workspace and, for a
        // model with sliding rings, less StateImage per slot; it prefills more slowly).
        std::vector<std::uint32_t> chunks{resolved.prefill_chunk};
        if (resolved.prefill_chunk_automatic) {
            for (std::uint32_t chunk = resolved.prefill_chunk / 2;
                 chunk >= kNarrowestAutomaticPrefillChunk; chunk /= 2) {
                if (chunk % kPrefillChunkQuantum == 0) { chunks.push_back(chunk); }
            }
        }
        // Whether the plan at a max_context fits, for a concurrency, chunk, cached-state policy and
        // KV capacity policy.
        const auto fits_at = [&](std::uint32_t concurrency, std::uint32_t chunk,
                                 const CachedStatePolicy& states, const KvCapacityPolicy& kv) {
            return [&, concurrency, chunk, states, kv](std::uint32_t max_context) {
                return try_resolve_runtime_capacity(
                           kv, states,
                           [&](std::uint32_t count) {
                               return models::qwen3_5::make_sequence_planner(
                                          execution, device,
                                          planned_at(max_context, count, concurrency, chunk, kv),
                                          graph_memory_held)
                                   .capacity_curve();
                           },
                           fit_free)
                    .has_value();
            };
        };
        // An automatic context takes an automatic KV capacity: the resolved one when it is
        // automatic, else (the context was asked for; the lower concurrencies a refusal names) the
        // default one.
        const KvCapacityPolicy automatic_kv =
            resolved.kv_capacity.mode == KvCapacityMode::Automatic
                ? resolved.kv_capacity
                : with_unified_memory_headroom(
                      with_paravirtual_headroom(KvCapacityPolicy::automatic(),
                                                paravirtualized_gpu(device.device), drives_display(device.device),
                                                device.props.totalGlobalMem),
                      device.props.integrated != 0, device.props.totalGlobalMem);
        // An automatic context leaves the best-effort headroom too (a GPU Windows shares): at the
        // longest context the pool is its minimum and has no pages to give up for it, so the card
        // filled and Windows paged it out (Qwen3.8-27B at its own 262,144 tokens: "free 0 B", a
        // decode round of 53.7 s). Below the floor the context takes the headroom back.
        const KvCapacityPolicy context_kv = [&] {
            KvCapacityPolicy policy = automatic_kv;
            policy.automatic_headroom_bytes += policy.best_effort_headroom_bytes;
            policy.best_effort_headroom_bytes = 0;
            return policy;
        }();
        const bool context_headroom =
            context_kv.automatic_headroom_bytes != automatic_kv.automatic_headroom_bytes;
        // The cached-state policies for a concurrency: the preferred count (configured, else one
        // per slot), and the automatic one that may go down to the minimum.
        const auto states_for = [&](std::uint32_t concurrency) {
            CachedStatePolicy sizing    = sizing_states;
            sizing.automatic_default    = concurrency;
            CachedStatePolicy preferred = sizing;
            preferred.configured        = cached_states.configured.value_or(concurrency);
            return std::pair{preferred, sizing};
        };
        const std::uint32_t ceiling = resolved.max_context;
        const std::uint32_t floor   = std::min(ceiling, kAutomaticContextFloor);
        // The longest automatic context a concurrency fits: with the preferred cached states, then,
        // below the floor, with fewer of them; at the widest chunk that reaches the floor, else at
        // the chunk with the longest context.
        struct ContextFit {
            std::optional<std::uint32_t> context;
            std::uint32_t chunk = 0;
        };
        const auto longest_context = [&](std::uint32_t concurrency) {
            const auto [preferred, sizing] = states_for(concurrency);
            ContextFit best{.context = std::nullopt, .chunk = chunks.back()};
            for (const std::uint32_t chunk : chunks) {
                std::optional<std::uint32_t> chosen = largest_fitting_context(
                    ceiling, fits_at(concurrency, chunk, preferred, context_kv));
                if (context_headroom && (!chosen || *chosen < floor)) {
                    chosen = largest_fitting_context(
                        ceiling, fits_at(concurrency, chunk, preferred, automatic_kv));
                }
                if (!cached_states.configured && (!chosen || *chosen < floor)) {
                    const std::optional<std::uint32_t> fewer_states =
                        largest_fitting_context(ceiling,
                                                fits_at(concurrency, chunk, sizing, automatic_kv));
                    if (fewer_states && (!chosen || *fewer_states > *chosen)) {
                        chosen = fewer_states;
                    }
                }
                if (chosen && (!best.context || *chosen > *best.context)) {
                    best = {.context = chosen, .chunk = chunk};
                }
                if (best.context && *best.context >= floor) { break; }
            }
            return best;
        };
        // Each lower concurrency that fits a longer automatic context than `served` (nullopt:
        // nothing fits at the served concurrency), the highest first.
        const auto lower_concurrency_fits = [&](std::optional<std::uint32_t> served) {
            std::vector<ConcurrencyFit> out;
            for (std::uint32_t concurrency = resolved.max_concurrency - 1U; concurrency >= 1U;
                 --concurrency) {
                const ContextFit fit = longest_context(concurrency);
                if (fit.context && (!served || *fit.context > *served)) {
                    out.push_back({.max_concurrency = concurrency, .max_context = *fit.context});
                    served = fit.context;
                }
                if (served && *served >= ceiling) { break; }
            }
            return out;
        };
        // With an automatic KV capacity, an automatic max_context is the longest candidate whose
        // plan fits (longest_context); below the floor with more than one slot, the lower
        // concurrencies that fit longer are recorded for the operator. An explicit max_context (or
        // one bounded by an explicit KV capacity) takes the widest chunk it fits with. When nothing
        // fits, the narrowest chunk and the shortest candidate are resolved below and refused with
        // their numbers.
        if (context.automatic && resolved.kv_capacity.mode == KvCapacityMode::Automatic) {
            const ContextFit fit   = longest_context(resolved.max_concurrency);
            resolved.prefill_chunk = fit.chunk;
            resolved.max_context   = fit.context.value_or(std::min(ceiling, kAutomaticContextMinimum));
            if (fit.context && *fit.context < floor && resolved.max_concurrency > 1) {
                context.lower_concurrency_fits = lower_concurrency_fits(fit.context);
            }
        } else if (chunks.size() > 1) {
            const auto [preferred, sizing] = states_for(resolved.max_concurrency);
            resolved.prefill_chunk         = chunks.back();
            for (const std::uint32_t chunk : chunks) {
                if (fits_at(resolved.max_concurrency, chunk, preferred,
                            resolved.kv_capacity)(resolved.max_context) ||
                    (!cached_states.configured &&
                     fits_at(resolved.max_concurrency, chunk, sizing,
                             resolved.kv_capacity)(resolved.max_context))) {
                    resolved.prefill_chunk = chunk;
                    break;
                }
            }
        }
        context.tokens = resolved.max_context;
        std::optional<KvCapacityResolution> resolved_capacity;
        try {
            resolved_capacity = resolve_runtime_capacity(
                resolved.kv_capacity, sizing_states,
                [&](std::uint32_t count) { return planner_for(count).capacity_curve(); },
                free_bytes);
        } catch (const CapacityError& error) {
            // An advised plan starts with the kernel warm-up and pins the Host tier, and both take
            // Device memory past the fit (the page tables: 2 MiB per pinned GiB). Before the
            // warm-up the advice is fitted to the free memory without them; after it, they are
            // measured and already taken off.
            if (!warmed) {
                const ContextCacheOptions& tier = resolved.context_cache;
                const std::size_t pinned =
                    tier.enabled ? tier.host_kv_capacity_bytes +
                                       (tier.host_state_slots != 0
                                            ? planner_for(sizing_states.automatic_minimum)
                                                      .host_state_image_bytes() *
                                                  tier.host_state_slots
                                            : 0)
                                 : 0;
                const std::size_t reserve = kAdviceWarmUpReserveBytes + (pinned >> 9);
                fit_free                  = free_bytes > reserve ? free_bytes - reserve : 0;
            }
            // What fits instead: the longest context at this concurrency (when the context was not
            // automatic) and at each lower one that fits longer.
            const auto what_fits = [&] {
                std::vector<ConcurrencyFit> out;
                if (!context.automatic || resolved.kv_capacity.mode != KvCapacityMode::Automatic) {
                    if (const auto here = longest_context(resolved.max_concurrency).context) {
                        out.push_back(
                            {.max_concurrency = resolved.max_concurrency, .max_context = *here});
                    }
                }
                if (resolved.max_concurrency > 1) {
                    const std::vector<ConcurrencyFit> lower = lower_concurrency_fits(
                        out.empty() ? std::nullopt
                                    : std::optional<std::uint32_t>(out.back().max_context));
                    out.insert(out.end(), lower.begin(), lower.end());
                }
                return out;
            };
            std::vector<ConcurrencyFit> fits = what_fits();
            // An explicit --prefill-chunk nothing fits with: the widest of the chunks an automatic
            // one narrows to (each half down to the narrowest) that fits, and what it fits.
            std::uint32_t narrowed_from = 0;
            if (fits.empty() && !resolved.prefill_chunk_automatic) {
                const std::uint32_t asked = chunks.front();
                for (std::uint32_t chunk = asked / 2; chunk >= kNarrowestAutomaticPrefillChunk;
                     chunk /= 2) {
                    if (chunk % kPrefillChunkQuantum != 0) { continue; }
                    chunks = {chunk};
                    fits   = what_fits();
                    if (!fits.empty()) {
                        narrowed_from = asked;
                        break;
                    }
                }
                if (fits.empty()) { chunks = {asked}; }
            }
            // What the fits assume: the chunks the fit may take, and the automatic KV capacity the
            // longest contexts are measured at (an explicit one is replaced). Every sizing choice
            // explicit, no kernel warm-up measures the CUDA graphs' memory: the plan reserves a
            // bound for it instead (SequencePlanner's graph allowance).
            const bool explicit_plan = !context.automatic && !resolved.prefill_chunk_automatic &&
                                       resolved.kv_capacity.mode != KvCapacityMode::Automatic &&
                                       cached_states.configured.has_value();
            const bool graph_allowance =
                explicit_plan && resolved.use_cuda_graph && !graph_memory_held;
            const std::string instead = capacity_advice({
                .fits                    = fits,
                .chunks                  = chunks,
                .narrowed_from           = narrowed_from,
                .prefill_chunk_automatic = resolved.prefill_chunk_automatic,
                .kv_capacity_explicit    = resolved.kv_capacity.mode != KvCapacityMode::Automatic,
                .context_automatic       = context.automatic,
                .ceiling                 = ceiling,
                .max_concurrency         = resolved.max_concurrency,
                .graph_allowance         = graph_allowance,
            });
            throw_capacity_shortage(error, instead + offload_hint());
        }
        return *resolved_capacity;
    };
    // The kernel warm-up, before the free memory an automatic fit depends on is measured for it.
    // CUDA loads a kernel's code into Device memory at its first launch (lazy module loading), and
    // the driver grows the local memory it keeps for every resident thread (cudaLimitStackSize) at
    // the first launch of a kernel with a larger frame; neither is given back. The warm-up starts
    // the stack limit from zero, so it ends at the largest frame of the plan's kernels. Which
    // kernels run is what the routes choose for this model and plan, so construction runs them: a
    // probe Program with the fitted concurrency, speculation, KV formats and prefill chunk takes
    // one request of every prefill width 1..chunk through the request path (admission, the prefill
    // with its speculative and sampling work, the first token and, once, a decode round), and a
    // probe Program at the fitted context captures and launches every decode graph class of it. The
    // fit is then made again from the free memory left, until the fitted context stops moving (a
    // shorter context's graph classes are the longer one's, cut at the same ranges). The probes
    // hold one request's KV and no cached states or Host tier, so they fit wherever the plan does.
    // The driver keeps a destroyed graph's memory for the next graph, so once the probe
    // built the fitted context's graph set, the served Program's identical set takes that memory
    // again and the fit reserves no graph allowance for it.
    const bool automatic_fit = unfitted_context.automatic || unfitted.prefill_chunk_automatic ||
                               unfitted.kv_capacity.mode == KvCapacityMode::Automatic ||
                               !cached_states.configured;
    // The warm-up below starts the per-thread stack from zero and measures what the plan's kernels
    // raise it to. It is dropped before the first fit too: loading raised it (the driver's default
    // times every resident thread, ~120 MiB on an RTX 5080 against the plan's 48 B per thread), and
    // a fit that counted it as used would refuse an asked context the warm-up then finds room for.
    // The load-time route tuner times on buffers of its own and frees them: before the stack limit
    // and the free memory are read.
    // Exact activations run the invariant routes, which the tuner does not touch: no family times
    // them (only the Nemotron family skipped it; the others tuned for nothing).
    if (options.activations != ActivationMode::Exact) {
        summary.moe_tune = execution.tune_routes(device.stream);
    }
    // The generic path's dense tuner times each dense problem at its first eager call (the warm-up's
    // prefill), before a decode graph is captured with it (ops/generic/generic_tune.cpp).
    ops::detail::generic::enable_dense_tune(true);
    const std::size_t stack_limit_at_load = current_stack_limit_bytes();
    if (automatic_fit) { CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 0)); }
    const std::size_t free_after_weights = current_free_device_bytes();
    // Before the warm-up the fit reserves the graph allowance, a bound about twice the largest
    // measured class. An automatic context shrinks under it and grows back once the warm-up
    // measured the graphs; an asked context cannot shrink, so a refusal here would stand on the
    // bound alone (Gemma 4 31B IQ3_XXS + MTP at 4 x 4,096 on an RTX 5080: 0.1 GiB short of the
    // bound, while the automatic context reached the same 4,096 with the graphs measured). When
    // the warm-up runs, such a plan is made without the allowance and the warm-up's probe measures
    // the graphs: the fit after it decides, from the free memory the graphs left.
    KvCapacityResolution resolution;
    try {
        resolution = fit_plan(free_after_weights, false);
    } catch (const CapacityError&) {
        if (!automatic_fit || !unfitted.use_cuda_graph || graph_memory_held) { throw; }
        graph_memory_held = true;
        resolution        = fit_plan(free_after_weights, false);
    }
    KernelWarmUpResolution warm_up;
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
        // The longest decode round a request asks for: the widest launch window and its target.
        std::uint32_t window = resolved.speculative.draft_tokens;
        for (const std::uint32_t k : resolved.speculative.request_draft_tokens) {
            window = std::max(window, k);
        }
        const std::uint32_t round_tokens = window + 1U;
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
                // The memory fit left too short a context for the widest round: plain decode (or
                // a shallower speculation) needs less, so this is a capacity refusal.
                throw CapacityError(
                    "the fitted context of " + std::to_string(probe_context) +
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
        // after the fit, when the Program pins. The fit reserves what pinning the fitted plan's two
        // Host tier buffers (its StateImages, its KV) takes, measured by pinning buffers of those
        // sizes; a fit whose sizes differ from the measured ones is made again.
        const auto host_tier_sizes = [&]() -> std::pair<std::size_t, std::size_t> {
            const ContextCacheOptions& tier = resolved.context_cache;
            const std::size_t states =
                tier.enabled && tier.host_state_slots != 0
                    ? planner_for(resolution.cached_device_states).host_state_image_bytes() *
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
        // Program then took it again past the fit (Qwen3.8-27B DFlash2 C=4: 0.8 GiB, the card
        // filled to 200 MiB of a planned 1 GiB headroom).
        std::size_t returned_graph_bytes = 0;
        // The graph memory a probe's capture needed beyond the fit's allowance. The allowance is a
        // bound measured on 4090/5080-class cards; an RTX 2080 Ti's DFlash2 graphs at C=8 took
        // more (Qwen3.8-27B IQ2_XXS, fp32 GDN state: 4 MiB left at the probe's instantiation, after
        // the four-minute width loop). A probe that does not fit grows this reserve and the fit
        // shrinks the automatic KV pool under it, so only the probe is repeated.
        std::size_t graph_shortfall_bytes = 0;
        for (std::uint32_t pass = 0;; ++pass) {
            if (pass > kMaximumWarmUpPasses) {
                throw std::logic_error("the kernel warm-up's fitted context did not settle");
            }
            const std::size_t free_now = current_free_device_bytes();
            const std::size_t taken    = warm_up.host_pin_device_bytes + returned_graph_bytes +
                                      graph_shortfall_bytes;
            resolution                 = fit_plan(free_now > taken ? free_now - taken : 0, true);
            if (const auto sizes = host_tier_sizes(); sizes != pinned_sizes) {
                warm_up.host_pin_device_bytes = pinning_device_bytes(sizes);
                pinned_sizes                  = sizes;
                continue;
            }
            // A context no longer than the probed one is settled: its graph classes are the
            // probed context's, cut at the same ranges, and take no more memory. Requiring the
            // same context let free-memory noise of a page move it each pass (Nemotron 3 30B at
            // C=4: 634,048 / 635,520 / 635,392 ... tokens, never settled).
            if (probed != 0 && resolved.max_context <= probed) { break; }
            ++warm_up.graph_passes;
            const std::size_t free_before_probe = current_free_device_bytes();
            std::unique_ptr<models::qwen3_5::Program> program;
            try {
                program = probe_program(resolved.max_context);
            } catch (const CapacityError&) {
                // Only an automatic plan can shrink to make room; an asked one refuses as before.
                if (unfitted.kv_capacity.mode != KvCapacityMode::Automatic) { throw; }
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
            // Never more than the probe took; the largest of the probes', so the fit only
            // shortens as the probes go.
            const std::size_t returned = std::min(
                given_back > probe_buffer_bytes ? given_back - probe_buffer_bytes : std::size_t{0},
                free_before_probe > free_with_probe ? free_before_probe - free_with_probe
                                                    : std::size_t{0});
            returned_graph_bytes         = std::max(returned_graph_bytes, returned);
            warm_up.graph_returned_bytes = returned_graph_bytes;
            probed = resolved.max_context;
            // The driver keeps the destroyed graphs' memory, and the served Program's graphs (this
            // context's set) take it again: the fits from here on reserve no graph allowance.
            graph_memory_held = true;
        }
        const std::size_t free_after = current_free_device_bytes();
        warm_up.lasting_bytes = free_before_warm > free_after ? free_before_warm - free_after : 0;
        warm_up.stack_limit_bytes = current_stack_limit_bytes();
        warm_up.seconds = std::chrono::duration<double>(Clock::now() - warm_started).count();
        warming.complete();
    }
    auto sequence =
        std::move(planner_for(resolution.cached_device_states)).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    planning.complete();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    models::qwen3_5::Frontend served_frontend = loaded.make_frontend(resolved.max_context);
    frontend.complete();
    auto instance = std::make_unique<ModelInstance>(std::move(loaded), std::move(served_frontend),
                                                    resolved.max_context);
    instance->kv_capacity_resolution = resolution;
    instance->host_tier              = host_tier;
    instance->moe_offload            = offload;
    instance->host_experts           = std::move(host_experts);
    instance->context                = context;
    instance->kernel_warm_up         = warm_up;
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(*instance->execution, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    for (auto& line : ops::detail::generic::dense_tune_report()) {
        summary.moe_tune.push_back(std::move(line));
    }
    summary.gpu_class    = ops::route_gpu_class();
    summary.route_table  = ops::route_table_status().message;
    summary.load_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    summary.context_cost = std::move(context_cost.summary);
    summary.metadata =
        model_metadata(options.artifact_path, summary, instance->execution->traits());
    return {std::move(instance), std::move(summary), std::move(context_cost.model),
            std::move(resolved)};
}

} // namespace ninferno::runtime
