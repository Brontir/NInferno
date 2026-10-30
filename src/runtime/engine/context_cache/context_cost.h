#pragma once

#include "core/device_class.h"
#include "core/transfer_work.h"
#include "ninferno/types.h"
#include "runtime/contract/resources.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::runtime {

inline constexpr std::uint64_t kContextCostQ32One = 1ULL << 32U;

// One machine-level transfer model. Resource type, model, KV dtype, and speculative backend do
// not participate: their physical byte and copy-operation counts already capture the work.
struct ContextTransferCost {
    std::uint64_t batch_ns        = 0;
    std::uint64_t operation_ns    = 0;
    std::uint64_t ns_per_byte_q32 = 0;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferCost,
                                                   ContextTransferCost) noexcept = default;
};

struct ContextPrefillCost {
    std::uint64_t chunk_ns              = 0;
    std::uint64_t token_ns_q32          = 0;
    std::uint64_t attention_pair_ns_q32 = 0;
    std::uint64_t vision_item_ns        = 0;
    std::uint64_t vision_patch_ns_q32   = 0;

    [[nodiscard]] friend constexpr bool operator==(ContextPrefillCost,
                                                   ContextPrefillCost) noexcept = default;
};

enum class MaterializationCopyPhase : std::uint8_t {
    PressureToHost,
    Candidate,
};

// Work already coalesced for one ordered copy phase and one direction. Distinct entries are serial;
// callers merge work that shares both fields before asking the machine model to price it.
struct TransferBatchWork {
    MaterializationCopyPhase phase     = MaterializationCopyPhase::Candidate;
    ContextTransferDirection direction = ContextTransferDirection::HostToDevice;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(TransferBatchWork,
                                                   TransferBatchWork) noexcept = default;
};

struct ContextMachineCostModel {
    // Direction order is DeviceToHost, HostToDevice, DeviceToDevice.
    std::array<ContextTransferCost, 3> transfer{};
    ContextPrefillCost prefill;

    // max(batch + copy_operations * operation, payload_bytes * ns_per_byte_q32 / 2^32)
    [[nodiscard]] std::uint64_t transfer_ns(ContextTransferDirection direction,
                                            TransferWork work) const noexcept;
    [[nodiscard]] std::uint64_t
    transfer_batches_ns(std::span<const TransferBatchWork> batches) const noexcept;
    [[nodiscard]] std::uint64_t prefill_ns(PrefillWork work) const noexcept;

    [[nodiscard]] friend constexpr bool
    operator==(const ContextMachineCostModel&, const ContextMachineCostModel&) noexcept = default;
};

struct PricedMaterializationMachineWork {
    std::uint64_t optimistic_request_ns = 0;
    std::uint64_t immediate_ns          = 0;
    std::uint64_t transferred_bytes     = 0;
    std::uint32_t copy_operations       = 0;
};

[[nodiscard]] PricedMaterializationMachineWork
price_materialization_machine_work(const ContextMachineCostModel& model,
                                   const MaterializationMachineWork& work) noexcept;

[[nodiscard]] std::uint64_t price_checkpoint_recovery_work(
    const ContextMachineCostModel& model,
    std::span<const CheckpointRecoveryAlternativeWork> alternatives) noexcept;

[[nodiscard]] std::uint64_t price_context_transfer_requirements(
    const ContextMachineCostModel& model,
    std::span<const ContextTransferRequirement> requirements) noexcept;

// What the cost model of a device is chosen by: device facts, never the driver's name for it.
struct ContextCostIdentity {
    // The device's class keys, closest first (DeviceClass::class_chain(): id, chip, architecture,
    // "*"): a preset keyed by one of them is the device's own.
    std::vector<std::string> class_chain;
    // The device as measured data sees it (board_of): how far a compiled machine carries to it.
    Board board;
    std::string prefill_signature;
};

// The identity of `device` running a model of `prefill_signature`.
[[nodiscard]] ContextCostIdentity context_cost_identity(const DeviceClass& device,
                                                        std::string prefill_signature);

struct ContextPrefillPreset {
    std::string prefill_signature;
    ContextPrefillCost cost;
};

struct ContextCostMachinePreset {
    // The class key a device finds it by along its class chain. A compiled machine names the class
    // id of the board it was measured on ("gb202:170sm"): the boards of one chip differ in speed.
    std::string gpu_class;
    std::optional<std::array<ContextTransferCost, 3>> transfer;
    std::vector<ContextPrefillPreset> prefill;
    // A measured prefill of this machine that stands in for bindings it has no entry for. Without
    // it an unmeasured binding on this machine is priced with the generic (RTX 5090) prefill,
    // which misstates recomputation by the machine's own speed ratio.
    std::optional<ContextPrefillCost> class_prefill;
    // The board it was measured on (core/measured_boards.h). A device whose class chain names no
    // compiled machine takes the one measured on the board nearest it within
    // measured::kCarryDistance (board_distance), as measured: no rule turns an SM count into a
    // speed. 0 SMs (an external entry): never carried.
    Board board;
};

struct ResolvedContextMachineCost {
    ContextMachineCostModel model;
    ContextCostSummary summary;
};

[[nodiscard]] ContextMachineCostModel generic_context_machine_cost_model();

[[nodiscard]] std::vector<ContextCostMachinePreset>
parse_context_cost_presets(std::string_view json, std::string_view source_name);

// Transfer and prefill are resolved independently. Generic numerical defaults always exist; the
// compiled machine and then the external entry of the first key of the device's class chain that
// has one override them. A device whose class chain names no compiled machine takes the one of the
// nearest measured board within measured::kCarryDistance, else the generic defaults
// (ContextCostSummary::machine says which: "own <key>", "carried from <key> at <distance>",
// "generic"). A binding a machine has no prefill entry for takes that machine's class_prefill
// before the generic one.
[[nodiscard]] ResolvedContextMachineCost
resolve_context_machine_cost(const ContextCostIdentity& identity,
                             const std::filesystem::path& external_preset_path = {});
// The same resolution over an explicit compiled table; the overload above passes
// compiled_context_cost_defaults().
[[nodiscard]] ResolvedContextMachineCost
resolve_context_machine_cost(const std::vector<ContextCostMachinePreset>& compiled,
                             const ContextCostIdentity& identity,
                             const std::filesystem::path& external_preset_path = {});

// Calibration updates one independently measurable component of the machine keyed `gpu_class`
// (the measuring device's DeviceClass::id) and preserves every other component.
void upsert_context_transfer_cost_atomic(const std::filesystem::path& path,
                                         std::string_view gpu_class,
                                         const std::array<ContextTransferCost, 3>& transfer,
                                         std::string_view provenance_json);
void upsert_context_prefill_cost_atomic(const std::filesystem::path& path,
                                        std::string_view gpu_class,
                                        std::string_view prefill_signature,
                                        const ContextPrefillCost& prefill,
                                        std::string_view provenance_json);

} // namespace ninferno::runtime
