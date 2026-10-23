#include "runtime/engine/context_cache/context_cost.h"
#include "core/checked.h"

#include "core/cache_file.h"
#include "core/device_class.h"
#include "core/file_io.h"
#include "core/measured_boards.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace ninferno::runtime {

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost();
const ContextPrefillCost& generic_context_prefill_cost();
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults();

namespace {

using Json = nlohmann::json;

constexpr std::size_t direction_index(ContextTransferDirection direction) noexcept {
    return static_cast<std::size_t>(direction);
}

std::uint64_t saturating_product(std::uint64_t left, std::uint64_t right) noexcept {
    const Wide product = mul_wide(left, right);
    return product.high != 0 ? std::numeric_limits<std::uint64_t>::max() : product.low;
}

std::uint64_t q32_product_ns(std::uint64_t coefficient, std::uint64_t units) noexcept {
    if (coefficient == 0 || units == 0) { return 0; }
    // ceil(product / 2^32), saturated where it passes 64 bits: product >= (2^64 - 1) * 2^32.
    const Wide product = mul_wide(coefficient, units);
    if (product.high > 0xFFFFFFFFULL ||
        (product.high == 0xFFFFFFFFULL && product.low >= 0xFFFFFFFF00000000ULL)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    const std::uint64_t rounded = product.low + (kContextCostQ32One - 1U);
    const std::uint64_t high    = product.high + (rounded < product.low ? 1U : 0U);
    return (high << 32U) | (rounded >> 32U);
}

void require_object(const Json& value, std::string_view context) {
    if (!value.is_object()) {
        throw std::invalid_argument(std::string(context) + " must be an object");
    }
}

void require_exact_members(const Json& value, std::initializer_list<std::string_view> required,
                           std::initializer_list<std::string_view> optional,
                           std::string_view context) {
    require_object(value, context);
    for (const std::string_view member : required) {
        if (!value.contains(std::string(member))) {
            throw std::invalid_argument(std::string(context) + " is missing '" +
                                        std::string(member) + "'");
        }
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        const std::string_view member = it.key();
        const bool known = std::find(required.begin(), required.end(), member) != required.end() ||
                           std::find(optional.begin(), optional.end(), member) != optional.end();
        if (!known) {
            throw std::invalid_argument(std::string(context) + " contains unknown member '" +
                                        std::string(member) + "'");
        }
    }
}

std::uint64_t require_u64(const Json& value, std::string_view context) {
    if (value.is_number_unsigned()) { return value.get<std::uint64_t>(); }
    if (value.is_number_integer()) {
        const std::int64_t parsed = value.get<std::int64_t>();
        if (parsed >= 0) { return static_cast<std::uint64_t>(parsed); }
    }
    throw std::invalid_argument(std::string(context) + " must be an unsigned integer");
}

std::string require_nonempty_string(const Json& value, std::string_view context) {
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        throw std::invalid_argument(std::string(context) + " must be a nonempty string");
    }
    return value.get<std::string>();
}

void validate_transfer(const std::array<ContextTransferCost, 3>& transfer,
                       std::string_view context) {
    for (std::size_t index = 0; index < transfer.size(); ++index) {
        const ContextTransferCost& value = transfer[index];
        if (value.batch_ns == 0 && value.operation_ns == 0 && value.ns_per_byte_q32 == 0) {
            throw std::invalid_argument(std::string(context) + "[" + std::to_string(index) +
                                        "] has no measurable cost");
        }
    }
}

void validate_prefill(const ContextPrefillCost& prefill, std::string_view context) {
    if (prefill.chunk_ns == 0 && prefill.token_ns_q32 == 0 && prefill.attention_pair_ns_q32 == 0) {
        throw std::invalid_argument(std::string(context) + " has no text cost");
    }
    if (prefill.vision_item_ns == 0 && prefill.vision_patch_ns_q32 == 0) {
        throw std::invalid_argument(std::string(context) + " has no Vision cost");
    }
}

ContextTransferCost parse_transfer_direction(const Json& value, std::string_view context) {
    require_exact_members(value, {"batch_ns", "operation_ns", "ns_per_byte_q32"}, {}, context);
    ContextTransferCost result{
        .batch_ns = require_u64(value.at("batch_ns"), std::string(context) + ".batch_ns"),
        .operation_ns =
            require_u64(value.at("operation_ns"), std::string(context) + ".operation_ns"),
        .ns_per_byte_q32 =
            require_u64(value.at("ns_per_byte_q32"), std::string(context) + ".ns_per_byte_q32"),
    };
    if (result.batch_ns == 0 && result.operation_ns == 0 && result.ns_per_byte_q32 == 0) {
        throw std::invalid_argument(std::string(context) + " has no measurable cost");
    }
    return result;
}

std::array<ContextTransferCost, 3> parse_transfer(const Json& value, std::string_view context) {
    require_exact_members(value, {"d2h", "h2d", "d2d"}, {}, context);
    return {
        parse_transfer_direction(value.at("d2h"), std::string(context) + ".d2h"),
        parse_transfer_direction(value.at("h2d"), std::string(context) + ".h2d"),
        parse_transfer_direction(value.at("d2d"), std::string(context) + ".d2d"),
    };
}

ContextPrefillCost parse_prefill(const Json& value, std::string_view context) {
    require_exact_members(value,
                          {"chunk_ns", "token_ns_q32", "attention_pair_ns_q32", "vision_item_ns",
                           "vision_patch_ns_q32"},
                          {}, context);
    ContextPrefillCost result{
        .chunk_ns = require_u64(value.at("chunk_ns"), std::string(context) + ".chunk_ns"),
        .token_ns_q32 =
            require_u64(value.at("token_ns_q32"), std::string(context) + ".token_ns_q32"),
        .attention_pair_ns_q32 = require_u64(value.at("attention_pair_ns_q32"),
                                             std::string(context) + ".attention_pair_ns_q32"),
        .vision_item_ns =
            require_u64(value.at("vision_item_ns"), std::string(context) + ".vision_item_ns"),
        .vision_patch_ns_q32 = require_u64(value.at("vision_patch_ns_q32"),
                                           std::string(context) + ".vision_patch_ns_q32"),
    };
    validate_prefill(result, context);
    return result;
}

Json transfer_direction_json(const ContextTransferCost& value) {
    return Json{{"batch_ns", value.batch_ns},
                {"operation_ns", value.operation_ns},
                {"ns_per_byte_q32", value.ns_per_byte_q32}};
}

Json transfer_json(const std::array<ContextTransferCost, 3>& value) {
    return Json{{"d2h", transfer_direction_json(value[0])},
                {"h2d", transfer_direction_json(value[1])},
                {"d2d", transfer_direction_json(value[2])}};
}

Json prefill_json(const ContextPrefillCost& value) {
    return Json{{"chunk_ns", value.chunk_ns},
                {"token_ns_q32", value.token_ns_q32},
                {"attention_pair_ns_q32", value.attention_pair_ns_q32},
                {"vision_item_ns", value.vision_item_ns},
                {"vision_patch_ns_q32", value.vision_patch_ns_q32}};
}

Json parse_document(std::string_view text, std::string_view source_name) {
    try {
        return Json::parse(text.begin(), text.end());
    } catch (const Json::exception& error) {
        throw std::invalid_argument("invalid context-cost preset JSON in " +
                                    std::string(source_name) + ": " + error.what());
    }
}

const ContextCostMachinePreset* find_machine(const std::vector<ContextCostMachinePreset>& presets,
                                             std::string_view gpu_class) noexcept {
    const auto found = std::find_if(presets.begin(), presets.end(), [&](const auto& preset) {
        return preset.gpu_class == gpu_class;
    });
    return found == presets.end() ? nullptr : &*found;
}

// The machine of the first key of `class_chain` that has one; null when none does.
const ContextCostMachinePreset*
find_along_chain(const std::vector<ContextCostMachinePreset>& presets,
                 const std::vector<std::string>& class_chain) noexcept {
    for (const std::string& key : class_chain) {
        if (const ContextCostMachinePreset* machine = find_machine(presets, key)) {
            return machine;
        }
    }
    return nullptr;
}

const ContextPrefillPreset* find_prefill(const ContextCostMachinePreset& machine,
                                         std::string_view prefill_signature) noexcept {
    const auto found =
        std::find_if(machine.prefill.begin(), machine.prefill.end(), [&](const auto& preset) {
            return preset.prefill_signature == prefill_signature;
        });
    return found == machine.prefill.end() ? nullptr : &*found;
}

// The compiled machine a device resolves to.
struct MachineChoice {
    const ContextCostMachinePreset* machine = nullptr; // null: the generic defaults
    bool carried                            = false;
    double distance                         = 0.0; // board_distance of a carried machine
};

// The device's own machine (the first key of its class chain a compiled machine names), else the
// one measured on the board nearest it within measured::kCarryDistance (ties: the earlier), as
// measured, else none. Board distance is infinite between instruction sets and FP32-accumulate
// rates, so a machine never carries to another architecture's kind.
MachineChoice choose_machine(const std::vector<ContextCostMachinePreset>& presets,
                             const ContextCostIdentity& identity) noexcept {
    if (const ContextCostMachinePreset* own = find_along_chain(presets, identity.class_chain)) {
        return {.machine = own};
    }
    const ContextCostMachinePreset* nearest =
        nearest_measured(std::span<const ContextCostMachinePreset>(presets), identity.board,
                         measured::kCarryDistance);
    if (nearest == nullptr) { return {}; }
    return {.machine  = nearest,
            .carried  = true,
            .distance = board_distance(identity.board, nearest->board)};
}

// ContextCostSummary::machine: "own <key>", "carried from <key> at <distance>" (two decimals, as
// measured_boards.h quotes distances; context_cost_sources_of drops them), or "generic".
std::string machine_label(const MachineChoice& choice) {
    if (choice.machine == nullptr) { return "generic"; }
    if (!choice.carried) { return "own " + choice.machine->gpu_class; }
    char distance[32];
    std::snprintf(distance, sizeof(distance), "%.2f", choice.distance);
    return "carried from " + choice.machine->gpu_class + " at " + distance;
}

void validate_presets(const std::vector<ContextCostMachinePreset>& presets,
                      std::string_view context) {
    for (std::size_t machine_index = 0; machine_index < presets.size(); ++machine_index) {
        const ContextCostMachinePreset& machine = presets[machine_index];
        const std::string machine_context =
            std::string(context) + ".machines[" + std::to_string(machine_index) + "]";
        if (machine.gpu_class.empty()) {
            throw std::invalid_argument(machine_context + ".gpu_class is empty");
        }
        if (machine.transfer) {
            validate_transfer(*machine.transfer, machine_context + ".transfer");
        }
        if (machine.class_prefill) {
            validate_prefill(*machine.class_prefill, machine_context + ".class_prefill");
        }
        if (!machine.transfer && machine.prefill.empty() && !machine.class_prefill) {
            throw std::invalid_argument(machine_context + " contains no cost data");
        }
        for (std::size_t prior = 0; prior < machine_index; ++prior) {
            if (presets[prior].gpu_class == machine.gpu_class) {
                throw std::invalid_argument("duplicate context-cost gpu_class: " +
                                            machine.gpu_class);
            }
        }
        for (std::size_t prefill_index = 0; prefill_index < machine.prefill.size();
             ++prefill_index) {
            const ContextPrefillPreset& prefill = machine.prefill[prefill_index];
            if (prefill.prefill_signature.empty()) {
                throw std::invalid_argument(machine_context + ".prefill has an empty identity");
            }
            validate_prefill(prefill.cost, machine_context + ".prefill[" +
                                               std::to_string(prefill_index) + "].coefficients");
            for (std::size_t prior = 0; prior < prefill_index; ++prior) {
                if (machine.prefill[prior].prefill_signature == prefill.prefill_signature) {
                    throw std::invalid_argument(
                        "duplicate context-cost prefill identity: " + machine.gpu_class + "/" +
                        prefill.prefill_signature);
                }
            }
        }
    }
}

Json parse_provenance(std::string_view provenance_json) {
    Json provenance;
    try {
        provenance = Json::parse(provenance_json.begin(), provenance_json.end());
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("invalid context-cost provenance JSON: ") +
                                    error.what());
    }
    if (!provenance.is_object()) {
        throw std::invalid_argument("context-cost provenance must be an object");
    }
    return provenance;
}

Json load_or_create_document(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return Json{{"schema_version", 4},
                    {"artifact_type", "ninferno_context_cost_presets"},
                    {"machines", Json::array()}};
    }
    const std::string text = read_file(path, "context-cost preset file");
    (void)parse_context_cost_presets(text, path.string());
    return parse_document(text, path.string());
}

Json& find_or_append_machine(Json& document, std::string_view gpu_class) {
    for (Json& machine : document.at("machines")) {
        if (machine.at("gpu_class").get_ref<const std::string&>() == gpu_class) { return machine; }
    }
    document.at("machines")
        .push_back(
            Json{{"gpu_class", gpu_class}, {"transfer", nullptr}, {"prefill", Json::array()}});
    return document.at("machines").back();
}

void write_document_atomic(const std::filesystem::path& path, const Json& document) {
    const std::string serialized = document.dump(2) + '\n';
    (void)parse_context_cost_presets(serialized, "generated context-cost presets");
    create_parent_directories(path);
    write_file_atomic(path, serialized);
}

} // namespace

std::uint64_t ContextMachineCostModel::transfer_ns(ContextTransferDirection direction,
                                                   TransferWork work) const noexcept {
    if (work.payload_bytes == 0 && work.copy_operations == 0) { return 0; }
    const std::size_t index = direction_index(direction);
    if (index >= transfer.size()) { return std::numeric_limits<std::uint64_t>::max(); }
    const ContextTransferCost& cost = transfer[index];
    const std::uint64_t operation_limited =
        saturating_add(cost.batch_ns, saturating_product(cost.operation_ns, work.copy_operations));
    const std::uint64_t bandwidth_limited =
        q32_product_ns(cost.ns_per_byte_q32, work.payload_bytes);
    return std::max(operation_limited, bandwidth_limited);
}

std::uint64_t ContextMachineCostModel::transfer_batches_ns(
    std::span<const TransferBatchWork> batches) const noexcept {
    constexpr std::size_t kPhaseCount =
        static_cast<std::size_t>(MaterializationCopyPhase::Candidate) + 1U;
    constexpr std::size_t kDirectionCount = 3;
    std::array<TransferWork, kPhaseCount * kDirectionCount> coalesced{};
    for (const TransferBatchWork& batch : batches) {
        const std::size_t phase     = static_cast<std::size_t>(batch.phase);
        const std::size_t direction = static_cast<std::size_t>(batch.direction);
        if (phase >= kPhaseCount || direction >= kDirectionCount) { continue; }
        TransferWork& work = coalesced[phase * kDirectionCount + direction];
        work.payload_bytes = saturating_add(work.payload_bytes, batch.work.payload_bytes);
        const std::uint64_t operations =
            static_cast<std::uint64_t>(work.copy_operations) + batch.work.copy_operations;
        work.copy_operations = operations > std::numeric_limits<std::uint32_t>::max()
                                   ? std::numeric_limits<std::uint32_t>::max()
                                   : static_cast<std::uint32_t>(operations);
    }
    std::uint64_t total = 0;
    for (std::size_t phase = 0; phase < kPhaseCount; ++phase) {
        for (std::size_t direction = 0; direction < kDirectionCount; ++direction) {
            total =
                saturating_add(total, transfer_ns(static_cast<ContextTransferDirection>(direction),
                                                  coalesced[phase * kDirectionCount + direction]));
        }
    }
    return total;
}

std::uint64_t ContextMachineCostModel::prefill_ns(PrefillWork work) const noexcept {
    std::uint64_t result = saturating_product(prefill.chunk_ns, work.chunks);
    result = saturating_add(result, q32_product_ns(prefill.token_ns_q32, work.tokens));
    result =
        saturating_add(result, q32_product_ns(prefill.attention_pair_ns_q32, work.attention_pairs));
    result = saturating_add(result, saturating_product(prefill.vision_item_ns, work.vision_items));
    result =
        saturating_add(result, q32_product_ns(prefill.vision_patch_ns_q32, work.vision_patches));
    return result;
}

PricedMaterializationMachineWork
price_materialization_machine_work(const ContextMachineCostModel& model,
                                   const MaterializationMachineWork& work) noexcept {
    constexpr std::array directions{
        ContextTransferDirection::DeviceToHost,
        ContextTransferDirection::HostToDevice,
        ContextTransferDirection::DeviceToDevice,
    };
    const auto phase_ns = [&](const CoalescedTransferWork& phase) {
        std::uint64_t result = 0;
        for (std::size_t index = 0; index < phase.size(); ++index) {
            result = saturating_add(result, model.transfer_ns(directions[index], phase[index]));
        }
        return result;
    };
    const auto accumulate_shape = [](const CoalescedTransferWork& phase, std::uint64_t& bytes,
                                     std::uint64_t& operations) {
        for (const TransferWork item : phase) {
            bytes      = saturating_add(bytes, item.payload_bytes);
            operations = saturating_add(operations, item.copy_operations);
        }
    };

    const std::uint64_t prefill_ns = model.prefill_ns(work.remaining_prefill_work);
    PricedMaterializationMachineWork result;
    result.optimistic_request_ns =
        saturating_add(prefill_ns, phase_ns(work.optimistic_candidate_transfers));
    result.immediate_ns = saturating_add(prefill_ns, phase_ns(work.pressure_transfers));
    result.immediate_ns = saturating_add(result.immediate_ns, phase_ns(work.candidate_transfers));
    std::uint64_t operations = 0;
    accumulate_shape(work.pressure_transfers, result.transferred_bytes, operations);
    accumulate_shape(work.candidate_transfers, result.transferred_bytes, operations);
    result.copy_operations = operations > std::numeric_limits<std::uint32_t>::max()
                                 ? std::numeric_limits<std::uint32_t>::max()
                                 : static_cast<std::uint32_t>(operations);
    return result;
}

std::uint64_t price_checkpoint_recovery_work(
    const ContextMachineCostModel& model,
    std::span<const CheckpointRecoveryAlternativeWork> alternatives) noexcept {
    constexpr std::array directions{
        ContextTransferDirection::DeviceToHost,
        ContextTransferDirection::HostToDevice,
        ContextTransferDirection::DeviceToDevice,
    };
    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
    for (const CheckpointRecoveryAlternativeWork& alternative : alternatives) {
        std::uint64_t cost = model.prefill_ns(alternative.prefill);
        for (std::size_t index = 0; index < alternative.transfers.size(); ++index) {
            cost = saturating_add(
                cost, model.transfer_ns(directions[index], alternative.transfers[index]));
        }
        best = std::min(best, cost);
    }
    return best;
}

std::uint64_t price_context_transfer_requirements(
    const ContextMachineCostModel& model,
    std::span<const ContextTransferRequirement> requirements) noexcept {
    CoalescedTransferWork coalesced{};
    for (const ContextTransferRequirement& requirement : requirements) {
        const std::size_t index = direction_index(requirement.direction);
        if (index >= coalesced.size()) { return std::numeric_limits<std::uint64_t>::max(); }
        coalesced[index].payload_bytes =
            saturating_add(coalesced[index].payload_bytes, requirement.work.payload_bytes);
        const std::uint64_t operations =
            static_cast<std::uint64_t>(coalesced[index].copy_operations) +
            requirement.work.copy_operations;
        coalesced[index].copy_operations = operations > std::numeric_limits<std::uint32_t>::max()
                                               ? std::numeric_limits<std::uint32_t>::max()
                                               : static_cast<std::uint32_t>(operations);
    }
    constexpr std::array directions{
        ContextTransferDirection::DeviceToHost,
        ContextTransferDirection::HostToDevice,
        ContextTransferDirection::DeviceToDevice,
    };
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < coalesced.size(); ++index) {
        total = saturating_add(total, model.transfer_ns(directions[index], coalesced[index]));
    }
    return total;
}

ContextCostIdentity context_cost_identity(const DeviceClass& device,
                                          std::string prefill_signature) {
    return {.class_chain       = device.class_chain(),
            .board             = board_of(device),
            .prefill_signature = std::move(prefill_signature)};
}

ContextMachineCostModel generic_context_machine_cost_model() {
    return ContextMachineCostModel{.transfer = generic_context_transfer_cost(),
                                   .prefill  = generic_context_prefill_cost()};
}

std::vector<ContextCostMachinePreset> parse_context_cost_presets(std::string_view json,
                                                                 std::string_view source_name) {
    const Json document = parse_document(json, source_name);
    require_exact_members(document, {"schema_version", "artifact_type", "machines"}, {},
                          source_name);
    if (require_u64(document.at("schema_version"), std::string(source_name) + ".schema_version") !=
        4) {
        throw std::invalid_argument("unsupported context-cost preset schema_version in " +
                                    std::string(source_name));
    }
    if (require_nonempty_string(document.at("artifact_type"),
                                std::string(source_name) + ".artifact_type") !=
        "ninferno_context_cost_presets") {
        throw std::invalid_argument("invalid context-cost artifact_type in " +
                                    std::string(source_name));
    }
    const Json& machines = document.at("machines");
    if (!machines.is_array()) {
        throw std::invalid_argument(std::string(source_name) + ".machines must be an array");
    }

    std::vector<ContextCostMachinePreset> result;
    result.reserve(machines.size());
    for (std::size_t machine_index = 0; machine_index < machines.size(); ++machine_index) {
        const Json& machine = machines[machine_index];
        const std::string context =
            std::string(source_name) + ".machines[" + std::to_string(machine_index) + "]";
        require_exact_members(machine, {"gpu_class", "transfer", "prefill"},
                              {"transfer_provenance"}, context);
        if (machine.contains("transfer_provenance") &&
            !machine.at("transfer_provenance").is_object()) {
            throw std::invalid_argument(context + ".transfer_provenance must be an object");
        }
        ContextCostMachinePreset parsed{
            .gpu_class = require_nonempty_string(machine.at("gpu_class"), context + ".gpu_class"),
        };
        if (!machine.at("transfer").is_null()) {
            parsed.transfer = parse_transfer(machine.at("transfer"), context + ".transfer");
        }
        const Json& prefill = machine.at("prefill");
        if (!prefill.is_array()) {
            throw std::invalid_argument(context + ".prefill must be an array");
        }
        parsed.prefill.reserve(prefill.size());
        for (std::size_t prefill_index = 0; prefill_index < prefill.size(); ++prefill_index) {
            const Json& value = prefill[prefill_index];
            const std::string prefill_context =
                context + ".prefill[" + std::to_string(prefill_index) + "]";
            require_exact_members(value, {"prefill_signature", "coefficients"}, {"provenance"},
                                  prefill_context);
            if (value.contains("provenance") && !value.at("provenance").is_object()) {
                throw std::invalid_argument(prefill_context + ".provenance must be an object");
            }
            parsed.prefill.push_back(ContextPrefillPreset{
                .prefill_signature = require_nonempty_string(
                    value.at("prefill_signature"), prefill_context + ".prefill_signature"),
                .cost = parse_prefill(value.at("coefficients"), prefill_context + ".coefficients"),
            });
        }
        result.push_back(std::move(parsed));
    }
    validate_presets(result, source_name);
    return result;
}

ResolvedContextMachineCost
resolve_context_machine_cost(const ContextCostIdentity& identity,
                             const std::filesystem::path& external_preset_path) {
    return resolve_context_machine_cost(compiled_context_cost_defaults(), identity,
                                        external_preset_path);
}

ResolvedContextMachineCost
resolve_context_machine_cost(const std::vector<ContextCostMachinePreset>& compiled,
                             const ContextCostIdentity& identity,
                             const std::filesystem::path& external_preset_path) {
    if (identity.class_chain.empty() || identity.class_chain.front().empty() ||
        identity.prefill_signature.empty()) {
        throw std::invalid_argument("context-cost identity contains an empty component");
    }

    ContextMachineCostModel model{.transfer = generic_context_transfer_cost(),
                                  .prefill  = generic_context_prefill_cost()};
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;

    validate_presets(compiled, "compiled context-cost defaults");
    const MachineChoice choice = choose_machine(compiled, identity);
    if (const ContextCostMachinePreset* machine = choice.machine) {
        const ContextCostPresetSource source = choice.carried
                                                   ? ContextCostPresetSource::CarriedMachine
                                                   : ContextCostPresetSource::CompiledDefault;
        if (machine->transfer) {
            model.transfer  = *machine->transfer;
            transfer_source = source;
        }
        if (const ContextPrefillPreset* prefill =
                find_prefill(*machine, identity.prefill_signature)) {
            model.prefill  = prefill->cost;
            prefill_source = source;
        } else if (machine->class_prefill) {
            model.prefill  = *machine->class_prefill;
            prefill_source = source;
        }
    }

    if (!external_preset_path.empty()) {
        const auto external =
            parse_context_cost_presets(read_file(external_preset_path, "context-cost preset file"),
                                       external_preset_path.string());
        if (const ContextCostMachinePreset* machine =
                find_along_chain(external, identity.class_chain)) {
            if (machine->transfer) {
                model.transfer  = *machine->transfer;
                transfer_source = ContextCostPresetSource::External;
            }
            if (const ContextPrefillPreset* prefill =
                    find_prefill(*machine, identity.prefill_signature)) {
                model.prefill  = prefill->cost;
                prefill_source = ContextCostPresetSource::External;
            }
        }
    }

    return ResolvedContextMachineCost{
        .model = model,
        .summary =
            {
                .transfer_source   = transfer_source,
                .prefill_source    = prefill_source,
                .gpu_class         = identity.class_chain.front(),
                .machine           = machine_label(choice),
                .prefill_signature = identity.prefill_signature,
                .preset_path       = external_preset_path,
            },
    };
}

void upsert_context_transfer_cost_atomic(const std::filesystem::path& path,
                                         std::string_view gpu_class,
                                         const std::array<ContextTransferCost, 3>& transfer,
                                         std::string_view provenance_json) {
    if (path.empty()) { throw std::invalid_argument("context-cost preset output is empty"); }
    if (gpu_class.empty()) { throw std::invalid_argument("context-cost gpu_class is empty"); }
    validate_transfer(transfer, "context-cost transfer");
    Json document                  = load_or_create_document(path);
    Json& machine                  = find_or_append_machine(document, gpu_class);
    machine["transfer"]            = transfer_json(transfer);
    machine["transfer_provenance"] = parse_provenance(provenance_json);
    write_document_atomic(path, document);
}

void upsert_context_prefill_cost_atomic(const std::filesystem::path& path,
                                        std::string_view gpu_class,
                                        std::string_view prefill_signature,
                                        const ContextPrefillCost& prefill,
                                        std::string_view provenance_json) {
    if (path.empty()) { throw std::invalid_argument("context-cost preset output is empty"); }
    if (gpu_class.empty() || prefill_signature.empty()) {
        throw std::invalid_argument("context-cost identity contains an empty component");
    }
    validate_prefill(prefill, "context-cost prefill");
    Json document = load_or_create_document(path);
    Json& machine = find_or_append_machine(document, gpu_class);
    Json replacement{{"prefill_signature", prefill_signature},
                     {"coefficients", prefill_json(prefill)},
                     {"provenance", parse_provenance(provenance_json)}};
    Json& entries    = machine.at("prefill");
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const Json& value) {
        return value.at("prefill_signature").get_ref<const std::string&>() == prefill_signature;
    });
    if (found == entries.end()) {
        entries.push_back(std::move(replacement));
    } else {
        *found = std::move(replacement);
    }
    write_document_atomic(path, document);
}

} // namespace ninferno::runtime
