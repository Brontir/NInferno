#pragma once

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "models/qwen3_5/program/model_family.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "core/expert_residency.h"
#include "runtime/engine/kv_capacity.h"
#include "ninferno/types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::ops {
class HostExperts;
} // namespace ninferno::ops

namespace ninferno::runtime {

class Backend;

// Rejects Engine options no model could be loaded under (the artifact path, context, KV capacity,
// concurrency and queue bounds) or that contradict each other; family-independent.
void validate_engine_options(const EngineOptions& options);

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// A family's loaded model as the Engine keeps it: opaque here, alive while its execution runs.
class LoadedModel {
public:
    virtual ~LoadedModel() = default;
};

// What a family's loader hands the Engine: the model, the execution over it, the frontend over its
// tokenizer and chat template, and the load facts it reports. load_seconds and the context cost
// are filled in by the Engine.
struct FamilyLoad {
    std::unique_ptr<LoadedModel> model;
    std::unique_ptr<const models::qwen3_5::detail::ModelExecution> execution;
    // The model's Frontend for a max_context: the Engine creates it once the context is resolved
    // (an automatic one depends on the plan over the loaded model).
    std::function<models::qwen3_5::Frontend(std::uint32_t max_context)> make_frontend;
    LoadSummary summary;
    // The loaded model's routed experts in Host groups (MoE expert offload), which the Engine
    // places before planning; null when the model has none offloaded.
    ExpertResidency* expert_residency = nullptr;
    // Routed experts the load left in the device arena, for the refusal of a plan that does not
    // fit (MoE expert offload could take them).
    std::uint64_t device_expert_bytes  = 0;
    std::uint32_t device_expert_layers = 0;
    // MoE expert offload's CPU decode path (src/ops/host_experts), when the family has one:
    // whether every offloaded layer's experts are in a format the CPU computes (else why; asked
    // after the groups are placed), the widest hidden size and top-k of its MoE layers (the
    // staging sizes), and the hand-over of the executor to the model before anything runs.
    std::function<bool(std::string* why)> host_experts_readable;
    std::int32_t host_experts_hidden = 0;
    std::int32_t host_experts_top_k  = 0;
    std::function<void(ops::HostExperts*)> attach_host_experts;
};

// Loads and materializes one artifact of a family (the phases ArtifactInspect through
// TargetPlan, and the tokenizer); the Engine plans and creates the Frontend and the Program.
using FamilyLoader = FamilyLoad (*)(const EngineOptions& options, DeviceContext& device);

// The context cache's Host tier as construction resolved it (ContextCacheOptions).
struct HostTierResolution {
    bool automatic                 = false;
    bool reduced                   = false;
    std::uint32_t default_slots    = 0;
    std::size_t default_kv_bytes   = 0;
    std::size_t budget_bytes       = 0;
};

// MoE expert offload as construction resolved it (EngineOptions::moe_offload).
struct MoeOffloadResolution {
    MoeOffloadMode mode          = MoeOffloadMode::Off;
    std::uint32_t moe_layers     = 0; // layers whose experts the load placed in Host groups
    std::uint32_t offloaded      = 0; // of those, read from pinned Host memory
    std::uint64_t pinned_bytes   = 0;
    std::uint64_t resident_bytes = 0; // the rest, copied to Device memory
    std::uint64_t staging_bytes  = 0; // the two Device staging slots
    std::uint32_t target_context = 0; // Automatic: the context the choice fits
    // Where decode calls compute the offloaded experts; the CPU's threads and instruction set;
    // why, when MoeOffloadCompute::Automatic chose.
    MoeOffloadCompute compute    = MoeOffloadCompute::Gpu;
    std::uint32_t cpu_threads    = 0;
    std::string cpu_isa;
    std::string compute_reason;
};

// max_context as construction resolved it (EngineOptions::max_context_automatic).
struct ContextResolution {
    bool automatic       = false;
    std::uint32_t native = 0; // the model's position capacity (ModelTraits::max_positions)
    std::uint32_t tokens = 0; // the max_context the Engine serves
    // Lower concurrencies and the longer automatic contexts they fit, when the served one is
    // below the automatic floor (MemorySummary::lower_concurrency_fits).
    std::vector<ConcurrencyFit> lower_concurrency_fits;
};

// The kernel warm-up construction ran before it measured the free memory for an automatic fit
// (construct_model): the prefill widths it ran (at least one per kernel class of 1..N; at 1..48
// every width is its own class, above that the classes come from the route layer's own choice
// points - the compiled C++ band edges, the loaded route table's breakpoints and the tile variants,
// warmup_widths.h), the probe Programs that captured every decode graph class at the fitted
// context, its time, the free Device memory it took for good (kernel code and the stack's local
// memory, from a zero stack limit), and the CUDA stack limit before it (the driver's default) and
// after it (S, the largest frame the plan launched) with the warm-up step that last raised it.
struct KernelWarmUpResolution {
    bool ran                             = false;
    std::uint32_t prefill_widths         = 0; // the covered range's upper edge (1..N)
    std::vector<std::uint32_t> widths_run;    // the widths actually run, ascending
    std::uint32_t graph_passes           = 0;
    double seconds                       = 0.0;
    std::size_t lasting_bytes            = 0;
    std::size_t stack_limit_before_bytes = 0;
    std::size_t stack_limit_bytes        = 0;
    std::string stack_limit_source;
    // Device memory the Host tier's pinning takes (page tables for the pinned bytes), measured and
    // reserved in the fit.
    std::size_t host_pin_device_bytes    = 0;
    // Graph memory the driver gave back when the decode-graph probe was destroyed (WDDM; a
    // driver that keeps it gives back none), reserved in the fit.
    std::size_t graph_returned_bytes = 0;
    // Reserve added after a probe's graphs did not fit under the allowance (0 when they did).
    std::size_t graph_shortfall_bytes = 0;
};

// One served model: whatever its family, one execution under the one Program.
struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<LoadedModel> model;
    // The CPU executor of the offloaded experts (null when the GPU computes them). Declared before
    // the execution and the Program, so it outlives every graph whose host nodes call it.
    std::unique_ptr<ops::HostExperts> host_experts;
    const std::unique_ptr<const models::qwen3_5::detail::ModelExecution> execution;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    HostTierResolution host_tier;
    MoeOffloadResolution moe_offload;
    ContextResolution context;
    KernelWarmUpResolution kernel_warm_up;
    const std::uint32_t capacity; // the longest admitted prompt: the resolved max_context
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(FamilyLoad&& load, models::qwen3_5::Frontend frontend, std::uint32_t capacity);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;

    // The public token domain: the grammar vocabulary of constrained decoding covers exactly it.
    [[nodiscard]] std::uint32_t token_domain() const noexcept {
        return static_cast<std::uint32_t>(execution->traits().public_tokens);
    }
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
    // The options the model was constructed under, with an automatic max_context resolved and an
    // automatic Host tier fitted.
    EngineOptions options;
};

// Loads the artifact with the family its model_type selects (family_registry.h), then plans and
// creates the Program over it.
[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);
[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device,
                                               FamilyLoader load);

// The family loaders of this build (family_registry.cpp lists them by model_type).
// Qwen 3.5 / 3.6: dense and MoE GDN hybrids with MTP, DFlash and Vision.
[[nodiscard]] FamilyLoad load_qwen3_5(const EngineOptions& options, DeviceContext& device);
// Gemma 4 text targets (gemma4_family.cpp).
[[nodiscard]] FamilyLoad load_gemma4(const EngineOptions& options, DeviceContext& device);
// Dense decoders of the Llama lineage and their MoE variants (dense_family.cpp).
[[nodiscard]] FamilyLoad load_dense(const EngineOptions& options, DeviceContext& device);
// NemotronH hybrids: Mamba2, attention and relu^2 MoE blocks (nemotron_h_family.cpp).
[[nodiscard]] FamilyLoad load_nemotron_h(const EngineOptions& options, DeviceContext& device);

} // namespace ninferno::runtime
