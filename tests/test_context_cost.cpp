#include "runtime/engine/context_cache/context_cost.h"

#include "core/host_kv_arena.h"

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "support/posix.h"
#include "support/files.h"

namespace {

using Json = nlohmann::json;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <class Function>
void expect_throw(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) { return; }
    expect(false, message);
}

std::array<ninferno::runtime::ContextTransferCost, 3> transfer(std::uint64_t batch_ns = 10) {
    std::array<ninferno::runtime::ContextTransferCost, 3> result;
    for (auto& direction : result) {
        direction = {.batch_ns        = batch_ns,
                     .operation_ns    = 3,
                     .ns_per_byte_q32 = ninferno::runtime::kContextCostQ32One / 2};
    }
    return result;
}

ninferno::runtime::ContextPrefillCost prefill(std::uint64_t chunk_ns = 3) {
    return {
        .chunk_ns              = chunk_ns,
        .token_ns_q32          = ninferno::runtime::kContextCostQ32One / 2,
        .attention_pair_ns_q32 = ninferno::runtime::kContextCostQ32One / 4,
        .vision_item_ns        = 5,
        .vision_patch_ns_q32   = 2 * ninferno::runtime::kContextCostQ32One,
    };
}

Json direction_json(const ninferno::runtime::ContextTransferCost& value) {
    return Json{{"batch_ns", value.batch_ns},
                {"operation_ns", value.operation_ns},
                {"ns_per_byte_q32", value.ns_per_byte_q32}};
}

Json transfer_json(const std::array<ninferno::runtime::ContextTransferCost, 3>& value) {
    return Json{{"d2h", direction_json(value[0])},
                {"h2d", direction_json(value[1])},
                {"d2d", direction_json(value[2])}};
}

Json prefill_json(const ninferno::runtime::ContextPrefillCost& value) {
    return Json{{"chunk_ns", value.chunk_ns},
                {"token_ns_q32", value.token_ns_q32},
                {"attention_pair_ns_q32", value.attention_pair_ns_q32},
                {"vision_item_ns", value.vision_item_ns},
                {"vision_patch_ns_q32", value.vision_patch_ns_q32}};
}

Json document(Json machines) {
    return Json{{"schema_version", 3},
                {"artifact_type", "ninferno_context_cost_presets"},
                {"machines", std::move(machines)}};
}

Json machine(std::string hardware, Json transfer_value, Json prefill_values) {
    return Json{{"hardware_class", std::move(hardware)},
                {"transfer", std::move(transfer_value)},
                {"prefill", std::move(prefill_values)}};
}

Json prefill_entry(std::string signature, const ninferno::runtime::ContextPrefillCost& value) {
    return Json{{"prefill_signature", std::move(signature)}, {"coefficients", prefill_json(value)}};
}

void test_exact_evaluation() {
    expect(ninferno::runtime::context_cost_hardware_class("NVIDIA GeForce RTX 5090", 12, 0) ==
               "nvidia-geforce-rtx-5090-sm120",
           "hardware class duplicated or lost the NVIDIA vendor");
    expect(ninferno::runtime::context_cost_hardware_class("Example Accelerator", 9, 0) ==
               "nvidia-example-accelerator-sm90",
           "hardware class did not canonicalize a vendorless CUDA device name");

    ninferno::runtime::ContextMachineCostModel model{.transfer = transfer(), .prefill = prefill()};
    expect(model.transfer_ns(ninferno::runtime::ContextTransferDirection::DeviceToHost,
                             {.payload_bytes = 4, .copy_operations = 2}) == 16,
           "transfer operation roofline changed");
    expect(model.transfer_ns(ninferno::runtime::ContextTransferDirection::DeviceToHost,
                             {.payload_bytes = 100, .copy_operations = 1}) == 50,
           "transfer bandwidth roofline changed");
    expect(model.transfer_ns(ninferno::runtime::ContextTransferDirection::DeviceToHost, {}) == 0,
           "empty transfer has a nonzero cost");
    const std::array same_phase{
        ninferno::runtime::TransferBatchWork{
            .phase     = ninferno::runtime::MaterializationCopyPhase::PressureToHost,
            .direction = ninferno::runtime::ContextTransferDirection::DeviceToHost,
            .work      = {.payload_bytes = 4, .copy_operations = 1}},
        ninferno::runtime::TransferBatchWork{
            .phase     = ninferno::runtime::MaterializationCopyPhase::PressureToHost,
            .direction = ninferno::runtime::ContextTransferDirection::DeviceToHost,
            .work      = {.payload_bytes = 4, .copy_operations = 1}},
    };
    expect(model.transfer_batches_ns(same_phase) == 16,
           "same-phase transfers paid the batch intercept more than once");
    auto serial_phases     = same_phase;
    serial_phases[1].phase = ninferno::runtime::MaterializationCopyPhase::Candidate;
    expect(model.transfer_batches_ns(serial_phases) == 26,
           "serial transfer phases were incorrectly coalesced");
    auto independent_directions         = same_phase;
    independent_directions[1].direction = ninferno::runtime::ContextTransferDirection::HostToDevice;
    expect(model.transfer_batches_ns(independent_directions) == 26,
           "independent transfer directions were incorrectly coalesced");

    const ninferno::runtime::PrefillWork work =
        ninferno::runtime::make_prefill_work(100, 10, 2, 7, 8);
    expect(work.chunks == 2 && work.tokens == 10 && work.attention_pairs == 1055 &&
               work.vision_items == 2 && work.vision_patches == 7,
           "prefill feature construction changed");
    expect(ninferno::runtime::make_prefill_work(std::numeric_limits<std::uint64_t>::max(),
                                              std::numeric_limits<std::uint64_t>::max(), 0, 0, 1)
                   .attention_pairs == std::numeric_limits<std::uint64_t>::max(),
           "prefill attention work did not saturate");
    expect(model.prefill_ns(work) == 299, "prefill formula or Q32 rounding changed");

    ninferno::runtime::MaterializationMachineWork materialization;
    materialization.pressure_transfers[0]             = {.payload_bytes = 4, .copy_operations = 1};
    materialization.candidate_transfers[0]            = {.payload_bytes = 4, .copy_operations = 1};
    materialization.optimistic_candidate_transfers[0] = {.payload_bytes   = 100,
                                                         .copy_operations = 1};
    materialization.remaining_prefill_work.tokens     = 2;
    const auto priced =
        ninferno::runtime::price_materialization_machine_work(model, materialization);
    expect(priced.immediate_ns == 27 && priced.optimistic_request_ns == 51 &&
               priced.transferred_bytes == 8 && priced.copy_operations == 2,
           "materialization work was not priced once across its serial phases");

    const std::array recovery{
        ninferno::runtime::CheckpointRecoveryAlternativeWork{
            .prefill = {.tokens = 100},
        },
        ninferno::runtime::CheckpointRecoveryAlternativeWork{
            .transfers = {ninferno::TransferWork{},
                          ninferno::TransferWork{.payload_bytes = 40, .copy_operations = 1},
                          ninferno::TransferWork{}},
        },
    };
    expect(ninferno::runtime::price_checkpoint_recovery_work(model, recovery) == 20,
           "recovery pricing did not select the cheapest supported physical recipe");
    const std::array requirements{
        ninferno::runtime::ContextTransferRequirement{
            .direction = ninferno::runtime::ContextTransferDirection::DeviceToHost,
            .work      = {.payload_bytes = 4, .copy_operations = 1},
        },
        ninferno::runtime::ContextTransferRequirement{
            .direction = ninferno::runtime::ContextTransferDirection::DeviceToHost,
            .work      = {.payload_bytes = 4, .copy_operations = 1},
        },
    };
    expect(ninferno::runtime::price_context_transfer_requirements(model, requirements) == 16,
           "capture transfer requirements were not coalesced before pricing");

    model.prefill.chunk_ns = std::numeric_limits<std::uint64_t>::max();
    expect(model.prefill_ns({.chunks = 2}) == std::numeric_limits<std::uint64_t>::max(),
           "prefill cost did not saturate");
}

void test_kv_physical_work() {
    const ninferno::HostKVPageLayout page_major = ninferno::plan_host_kv_page_layout({
        .page_tokens        = 3,
        .device_plane_order = ninferno::PagedKVPlaneOrder::PageMajor,
        .planes =
            {
                {ninferno::DType::BF16, 8, 4, 256},
                {ninferno::DType::I8, 16, 2, 256},
            },
    });
    const std::uint64_t page_payload =
        page_major.planes[0].page_payload_bytes + page_major.planes[1].page_payload_bytes;
    expect(page_payload < page_major.page_stride,
           "KV work fixture did not contain alignment padding");
    expect(ninferno::plan_host_kv_transfer_work(page_major, 3, 2) ==
               ninferno::TransferWork{.payload_bytes = 3 * page_payload, .copy_operations = 4},
           "PageMajor Host KV work counts padding or the wrong CUDA operations");
    expect(ninferno::plan_device_kv_copy_work(page_major, 3) ==
               ninferno::TransferWork{.payload_bytes = 3 * page_payload, .copy_operations = 6},
           "PageMajor Device KV work does not count one copy per plane and page");

    const ninferno::HostKVPageLayout head_major = ninferno::plan_host_kv_page_layout({
        .page_tokens        = 3,
        .device_plane_order = ninferno::PagedKVPlaneOrder::HeadMajor,
        .planes =
            {
                {ninferno::DType::I8, 8, 3, 256},
                {ninferno::DType::BF16, 4, 5, 256},
            },
    });
    const std::uint64_t head_payload =
        head_major.planes[0].page_payload_bytes + head_major.planes[1].page_payload_bytes;
    expect(ninferno::plan_host_kv_transfer_work(head_major, 7, 2) ==
               ninferno::TransferWork{.payload_bytes = 7 * head_payload, .copy_operations = 16},
           "HeadMajor Host KV work does not count one copy per head and run");
    expect(ninferno::plan_device_kv_copy_work(head_major, 7) ==
               ninferno::TransferWork{.payload_bytes = 7 * head_payload, .copy_operations = 14},
           "HeadMajor Device KV work should remain one copy per plane and page");
}

void test_schema_validation() {
    Json entries = Json::array();
    entries.push_back(prefill_entry("config-and-formats", prefill()));
    const Json valid =
        document(Json::array({machine("machine", transfer_json(transfer()), entries)}));
    const auto parsed = ninferno::runtime::parse_context_cost_presets(valid.dump(), "test");
    expect(parsed.size() == 1 && parsed[0].transfer == transfer() &&
               parsed[0].prefill.size() == 1 && parsed[0].prefill[0].cost == prefill(),
           "valid layered preset did not parse");

    Json duplicate_machine = valid;
    duplicate_machine["machines"].push_back(duplicate_machine["machines"][0]);
    expect_throw(
        [&] {
            (void)ninferno::runtime::parse_context_cost_presets(duplicate_machine.dump(),
                                                              "duplicate-machine");
        },
        "duplicate hardware entry was accepted");

    Json duplicate_prefill = valid;
    duplicate_prefill["machines"][0]["prefill"].push_back(
        duplicate_prefill["machines"][0]["prefill"][0]);
    expect_throw(
        [&] {
            (void)ninferno::runtime::parse_context_cost_presets(duplicate_prefill.dump(),
                                                              "duplicate-prefill");
        },
        "duplicate model prefill entry was accepted");

    Json invalid                                              = valid;
    invalid["machines"][0]["transfer"]["d2h"]["operation_ns"] = -1;
    expect_throw(
        [&] { (void)ninferno::runtime::parse_context_cost_presets(invalid.dump(), "negative"); },
        "negative transfer coefficient was accepted");

    Json prefill_only                       = valid;
    prefill_only["machines"][0]["transfer"] = nullptr;
    expect(
        ninferno::runtime::parse_context_cost_presets(prefill_only.dump(), "prefill-only").size() ==
            1,
        "independent prefill-only machine entry was rejected");
    prefill_only["machines"][0]["prefill"] = Json::array();
    expect_throw(
        [&] { (void)ninferno::runtime::parse_context_cost_presets(prefill_only.dump(), "empty"); },
        "empty machine cost entry was accepted");
}

// A machine no preset names takes the measured machine of its architecture nearest by SM count:
// its transfer as measured, its prefill for the binding (else its class prefill) scaled by the SM
// ratio. Another architecture, an unknown SM count, or a named machine never do.
void test_similar_machine() {
    using ninferno::ContextCostPresetSource;
    const std::vector<ninferno::runtime::ContextCostMachinePreset> compiled{
        {
            .hardware_class = "nvidia-big-sm120",
            .transfer       = transfer(5),
            .prefill        = {{"measured-binding", prefill(1'700)}},
            .class_prefill  = prefill(3'400),
            .sms            = 170,
        },
        {
            .hardware_class = "nvidia-small-sm120",
            .class_prefill  = prefill(8'400),
            .sms            = 84,
        },
        {.hardware_class = "nvidia-ada-sm89", .class_prefill = prefill(1'280), .sms = 128},
    };
    const auto resolve = [&](std::string hardware_class, std::uint32_t sms) {
        return ninferno::runtime::resolve_context_machine_cost(
            compiled, {.hardware_class    = std::move(hardware_class),
                       .prefill_signature = "measured-binding",
                       .sms               = sms});
    };

    // 160 SMs: the 170-SM card, its measured binding scaled by 170/160; its transfer as is.
    const auto near_big = resolve("nvidia-unmeasured-sm120", 160);
    expect(near_big.summary.prefill_source == ContextCostPresetSource::SimilarMachine &&
               near_big.summary.transfer_source == ContextCostPresetSource::SimilarMachine &&
               near_big.summary.similar_to == "nvidia-big-sm120" &&
               near_big.model.prefill.chunk_ns == 1'806 &&
               near_big.model.prefill.vision_item_ns == 5 &&
               near_big.model.transfer[0].batch_ns == 5,
           "a machine near the measured card took its binding's prefill, scaled, and its transfer");

    // 85 SMs: the 84-SM card, which has no entry for the binding: its class prefill, scaled.
    const auto near_small = resolve("nvidia-unmeasured-sm120", 85);
    expect(near_small.summary.prefill_source == ContextCostPresetSource::SimilarMachine &&
               near_small.summary.transfer_source == ContextCostPresetSource::GenericDefault &&
               near_small.summary.similar_to == "nvidia-small-sm120" &&
               near_small.model.prefill.chunk_ns == 8'301,
           "a machine near a card of a class prefill only took it, scaled");

    // Halfway on a log scale (120 SMs between 84 and 170 is nearer 170; 119.5 is the midpoint):
    // ties go to the larger card, and the nearest wins otherwise.
    expect(resolve("nvidia-unmeasured-sm120", 120).summary.similar_to == "nvidia-big-sm120" &&
               resolve("nvidia-unmeasured-sm120", 100).summary.similar_to == "nvidia-small-sm120",
           "the nearest card by SM count on a log scale stands in");

    const std::vector<std::tuple<std::string, std::uint32_t, std::string>> unplaced{
        {"nvidia-ampere-sm86", 82, "another architecture"},
        {"nvidia-unmeasured-sm120", 0, "an unknown SM count"},
    };
    for (const auto& [hardware_class, sms, what] : unplaced) {
        const auto none = resolve(hardware_class, sms);
        expect(none.summary.prefill_source == ContextCostPresetSource::GenericDefault &&
                   none.summary.transfer_source == ContextCostPresetSource::GenericDefault &&
                   none.summary.similar_to.empty(),
               (what + " took a similar machine").c_str());
    }
    const auto named = resolve("nvidia-small-sm120", 170);
    expect(named.summary.prefill_source == ContextCostPresetSource::CompiledDefault &&
               named.summary.similar_to.empty() && named.model.prefill.chunk_ns == 8'400,
           "a machine a preset names took another's costs");
}

// An architecture without a preset of its own takes a sibling architecture's (the same major and
// feature row: sm_80 and sm_86), its prefill scaled by SM count and the generic transfer; an
// architecture whose features differ takes none.
void test_sibling_machine() {
    using ninferno::ContextCostPresetSource;
    const std::vector<ninferno::runtime::ContextCostMachinePreset> compiled{
        {
            .hardware_class = "nvidia-geforce-rtx-3090-sm86",
            .transfer       = transfer(7),
            .class_prefill  = prefill(2'000),
            .sms            = 82,
        },
        {.hardware_class = "nvidia-ada-sm89", .class_prefill = prefill(1'280), .sms = 128},
    };
    const auto resolve = [&](std::string hardware_class, std::uint32_t sms) {
        return ninferno::runtime::resolve_context_machine_cost(
            compiled,
            {.hardware_class = std::move(hardware_class), .prefill_signature = "any", .sms = sms});
    };
    const auto a100 = resolve("nvidia-a100-sxm4-80gb-sm80", 108);
    expect(
        a100.summary.prefill_source == ContextCostPresetSource::SimilarMachine &&
            a100.summary.transfer_source == ContextCostPresetSource::GenericDefault &&
            a100.summary.similar_to == "nvidia-geforce-rtx-3090-sm86" &&
            a100.model.prefill.chunk_ns == 1'519,
        "sm_80 did not take the sm_86 card's prefill, scaled by 82/108, with the generic transfer");
    const auto v100 = resolve("tesla-v100-sxm2-32gb-sm70", 80);
    expect(v100.summary.prefill_source == ContextCostPresetSource::GenericDefault &&
               v100.summary.similar_to.empty(),
           "sm_70 took a preset of an architecture with other features");
}

void test_class_prefill_fallback() {
    const std::vector<ninferno::runtime::ContextCostMachinePreset> compiled{
        {
            .hardware_class = "measured-machine",
            .prefill        = {{"measured-binding", prefill(11)}},
            .class_prefill  = prefill(22),
        },
        {
            .hardware_class = "binding-only-machine",
            .prefill        = {{"measured-binding", prefill(33)}},
        },
    };
    const auto exact = ninferno::runtime::resolve_context_machine_cost(
        compiled, {.hardware_class = "measured-machine", .prefill_signature = "measured-binding"});
    expect(exact.model.prefill.chunk_ns == 11 &&
               exact.summary.prefill_source == ninferno::ContextCostPresetSource::CompiledDefault,
           "a measured binding did not take its own prefill over the class prefill");

    const auto miss = ninferno::runtime::resolve_context_machine_cost(
        compiled, {.hardware_class = "measured-machine", .prefill_signature = "other-binding"});
    expect(miss.model.prefill.chunk_ns == 22 &&
               miss.summary.prefill_source == ninferno::ContextCostPresetSource::CompiledDefault &&
               miss.summary.transfer_source == ninferno::ContextCostPresetSource::GenericDefault,
           "an unmeasured binding did not take its machine's class prefill");

    const auto no_class = ninferno::runtime::resolve_context_machine_cost(
        compiled, {.hardware_class = "binding-only-machine", .prefill_signature = "other-binding"});
    expect(no_class.summary.prefill_source == ninferno::ContextCostPresetSource::GenericDefault &&
               no_class.model.prefill.chunk_ns ==
                   ninferno::runtime::generic_context_machine_cost_model().prefill.chunk_ns,
           "a machine without a class prefill did not fall back to the generic prefill");

    const std::vector<ninferno::runtime::ContextCostMachinePreset> class_only{
        {.hardware_class = "class-only-machine", .class_prefill = prefill(44)},
    };
    const auto only = ninferno::runtime::resolve_context_machine_cost(
        class_only, {.hardware_class = "class-only-machine", .prefill_signature = "any"});
    expect(only.model.prefill.chunk_ns == 44,
           "a class prefill alone was not accepted as machine cost data");

    std::vector<ninferno::runtime::ContextCostMachinePreset> invalid = class_only;
    invalid[0].class_prefill->vision_item_ns                       = 0;
    invalid[0].class_prefill->vision_patch_ns_q32                  = 0;
    expect_throw(
        [&] {
            (void)ninferno::runtime::resolve_context_machine_cost(
                invalid, {.hardware_class = "class-only-machine", .prefill_signature = "any"});
        },
        "an invalid class prefill was accepted");
}

void test_resolution_and_atomic_upserts() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("ninferno-context-cost-test-" + std::to_string(static_cast<long long>(::getpid())));
    std::filesystem::create_directories(directory);
    const std::filesystem::path path = directory / "presets.json";
    try {
        // The RTX 3090: its own transfer, and for a binding it did not measure its class prefill
        // (the ga102 27B prefill). Transfer and prefill resolving independently of each other is
        // test_class_prefill_fallback's (every compiled machine now carries a class prefill).
        const auto compiled = ninferno::runtime::resolve_context_machine_cost(
            {.hardware_class    = "nvidia-geforce-rtx-3090-sm86",
             .prefill_signature = "unmeasured-config-and-formats"});
        const auto ga102 = ninferno::runtime::resolve_context_machine_cost(
            {.hardware_class = "nvidia-geforce-rtx-3090-sm86", .prefill_signature = "other"});
        expect(compiled.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   compiled.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   compiled.model.prefill.token_ns_q32 == ga102.model.prefill.token_ns_q32 &&
                   compiled.model.prefill.token_ns_q32 !=
                       ninferno::runtime::generic_context_machine_cost_model().prefill.token_ns_q32,
               "the RTX 3090 did not take its own transfer and its class prefill");

        const ninferno::runtime::ContextCostIdentity compiled_identity{
            .hardware_class    = "nvidia-geforce-rtx-5090-sm120",
            .prefill_signature = "unmeasured-config-and-formats",
        };
        const auto class_only = ninferno::runtime::resolve_context_machine_cost(compiled_identity);
        expect(class_only.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   class_only.model.prefill.token_ns_q32 == 944'785'862'335'153,
               "an unmeasured binding on the RTX 5090 did not take its class prefill");

        const auto measured = ninferno::runtime::resolve_context_machine_cost({
            .hardware_class    = compiled_identity.hardware_class,
            .prefill_signature = "e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a",
        });
        expect(measured.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   measured.model.prefill.token_ns_q32 == 375'800'765'711'778,
               "measured bindings did not select their compiled prefill cost");

        const ninferno::runtime::ContextCostIdentity unknown{
            .hardware_class    = "unmeasured-machine",
            .prefill_signature = "unmeasured-config-and-formats",
        };
        const auto generic = ninferno::runtime::resolve_context_machine_cost(unknown);
        expect(
            generic.summary.transfer_source == ninferno::ContextCostPresetSource::GenericDefault &&
                generic.summary.prefill_source ==
                    ninferno::ContextCostPresetSource::GenericDefault &&
                generic.model.transfer_ns(ninferno::runtime::ContextTransferDirection::DeviceToHost,
                                          {.payload_bytes = 1024, .copy_operations = 1}) > 0 &&
                generic.model.prefill_ns({.tokens = 1}) > 0,
            "unknown identity did not retain a numerical cost model");

        ninferno::runtime::upsert_context_transfer_cost_atomic(path, unknown.hardware_class,
                                                             transfer(77), R"({"run":1})");
        auto resolved = ninferno::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 &&
                   resolved.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::GenericDefault,
               "external transfer did not layer over generic prefill");

        ninferno::runtime::upsert_context_prefill_cost_atomic(path, unknown, prefill(91),
                                                            R"({"run":2})");
        resolved = ninferno::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 && resolved.model.prefill.chunk_ns == 91 &&
                   resolved.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source == ninferno::ContextCostPresetSource::External,
               "independent external transfer/prefill entries did not compose");

        const ninferno::runtime::ContextCostIdentity other_model{
            .hardware_class    = unknown.hardware_class,
            .prefill_signature = "other-config-and-formats",
        };
        const auto layered_miss =
            ninferno::runtime::resolve_context_machine_cost(other_model, path);
        expect(layered_miss.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::External &&
                   layered_miss.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::GenericDefault,
               "external model miss incorrectly discarded the matching machine transfer");

        ninferno::runtime::upsert_context_transfer_cost_atomic(path, unknown.hardware_class,
                                                             transfer(123), R"({"run":3})");
        const auto parsed = ninferno::runtime::parse_context_cost_presets(read_file(path), "file");
        expect(parsed.size() == 1 && parsed[0].transfer->at(0).batch_ns == 123 &&
                   parsed[0].prefill.size() == 1 && parsed[0].prefill[0].cost.chunk_ns == 91,
               "transfer replacement did not preserve the model prefill list");

        const std::string before = read_file(path);
        expect_throw(
            [&] {
                ninferno::runtime::upsert_context_prefill_cost_atomic(path, unknown, prefill(),
                                                                    "not-json");
            },
            "invalid provenance was accepted");
        expect(read_file(path) == before, "failed upsert modified the preset file");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    test_exact_evaluation();
    test_kv_physical_work();
    test_schema_validation();
    test_class_prefill_fallback();
    test_similar_machine();
    test_sibling_machine();
    test_resolution_and_atomic_upserts();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
