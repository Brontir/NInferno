#include "runtime/engine/context_cache/context_cost.h"

#include "core/device_class.h"
#include "core/host_kv_arena.h"
#include "core/measured_boards.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    return Json{{"schema_version", 4},
                {"artifact_type", "ninferno_context_cost_presets"},
                {"machines", std::move(machines)}};
}

Json machine(std::string gpu_class, Json transfer_value, Json prefill_values) {
    return Json{{"gpu_class", std::move(gpu_class)},
                {"transfer", std::move(transfer_value)},
                {"prefill", std::move(prefill_values)}};
}

// A device of class keys `chain` (closest first) on `board` (none: never carried to) running
// `signature`.
ninferno::runtime::ContextCostIdentity identity(std::vector<std::string> chain,
                                                std::string signature, ninferno::Board board = {}) {
    return {
        .class_chain = std::move(chain), .board = board, .prefill_signature = std::move(signature)};
}

// A board as the device query would build it, with the PCI-ID table's facts (board_of reads its
// bandwidth from them).
ninferno::DeviceClass board_class(std::uint32_t pci_device_id, int major, int minor, int sms,
                                  double l2_mib) {
    ninferno::DeviceClass d =
        ninferno::synthetic_device_class(pci_device_id, major, minor, sms,
                                         static_cast<std::size_t>(l2_mib * (1 << 20)), 24ULL << 30);
    d.facts = ninferno::machine_facts_of(d);
    return d;
}

Json prefill_entry(std::string signature, const ninferno::runtime::ContextPrefillCost& value) {
    return Json{{"prefill_signature", std::move(signature)}, {"coefficients", prefill_json(value)}};
}

void test_exact_evaluation() {
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

// Each board the compiled machines were measured on finds its own by its class id, whatever the
// driver calls it: its transfer where one was measured (the RTX 4090's D2D copies did not fit:
// the generic transfer) and its class prefill for a binding it did not measure.
void test_measured_boards() {
    using ninferno::ContextCostPresetSource;
    namespace measured = ninferno::measured;

    struct MeasuredMachine {
        const char* what;
        std::uint32_t pci_device_id;
        ninferno::Board board;
        const char* gpu_class;
        std::uint64_t class_token_ns_q32;
        std::uint64_t d2h_batch_ns; // the generic transfer's 100 000 without its own
    };

    const std::array machines{
        MeasuredMachine{"RTX 5090", 0x2B85, measured::kRtx5090, "gb202:170sm", 944'785'862'335'153,
                        48'655},
        MeasuredMachine{"RTX 4090", 0x2684, measured::kRtx4090, "ad102:128sm",
                        1'219'524'115'385'335, 100'000},
        MeasuredMachine{"RTX 3090", 0x2204, measured::kRtx3090, "ga102:82sm", 2'964'978'981'039'522,
                        27'120},
        MeasuredMachine{"RTX 2080 Ti", 0x1E07, measured::kRtx2080Ti, "tu102:68sm",
                        20'155'682'553'363'628, 3'177},
    };
    for (const MeasuredMachine& m : machines) {
        ninferno::DeviceClass device =
            board_class(m.pci_device_id, m.board.major, m.board.minor, m.board.sms, m.board.l2_mib);
        const auto own = ninferno::runtime::resolve_context_machine_cost(
            ninferno::runtime::context_cost_identity(device, "unmeasured-binding"));
        const ContextCostPresetSource transfer_source =
            m.d2h_batch_ns != 100'000 ? ContextCostPresetSource::CompiledDefault
                                      : ContextCostPresetSource::GenericDefault;
        expect(device.id() == m.gpu_class && own.summary.gpu_class == m.gpu_class &&
                   own.summary.machine == "own " + std::string(m.gpu_class) &&
                   own.summary.prefill_source == ContextCostPresetSource::CompiledDefault &&
                   own.summary.transfer_source == transfer_source &&
                   own.model.prefill.token_ns_q32 == m.class_token_ns_q32 &&
                   own.model.transfer[0].batch_ns == m.d2h_batch_ns,
               (std::string(m.what) + " did not find its own machine by its class id").c_str());

        // The driver's name is never a key: another board's name changes nothing.
        ninferno::DeviceClass renamed = device;
        renamed.name                  = "NVIDIA GeForce RTX 5090";
        const auto same               = ninferno::runtime::resolve_context_machine_cost(
            ninferno::runtime::context_cost_identity(renamed, "unmeasured-binding"));
        expect(same.summary.machine == own.summary.machine && same.model == own.model,
               (std::string(m.what) + " under another driver name took another machine").c_str());

        // The board each machine records is the one its key names: a device of no listed class on
        // that very board carries it at distance 0.
        const auto twin = ninferno::runtime::resolve_context_machine_cost(
            identity({"*"}, "unmeasured-binding", m.board));
        expect(twin.summary.machine == "carried from " + std::string(m.gpu_class) + " at 0.00" &&
                   twin.model == own.model,
               (std::string(m.what) + "'s machine does not record its board").c_str());
    }
    // A binding the RTX 2080 Ti measured takes its own entry over the class prefill.
    const auto turing_9b =
        ninferno::runtime::resolve_context_machine_cost(ninferno::runtime::context_cost_identity(
            board_class(0x1E07, 7, 5, 68, 5.5),
            "ed7fbdac7d2c4155ff33b80b97adfd30519180f94695a70aeedd589b9ad1abfe"));
    expect(turing_9b.summary.prefill_source == ContextCostPresetSource::CompiledDefault &&
               turing_9b.model.prefill.token_ns_q32 == 2'027'672'431'877'711,
           "the RTX 2080 Ti's measured 9B binding did not take its own prefill");
}

// A board no compiled machine names takes the one measured on the nearest board within the carry
// distance (core/measured_boards.h), as measured: no SM-count ratio rescales it. Farther, or of
// another instruction set or FP32-accumulate rate, it takes the declared generic defaults.
void test_carried_and_generic() {
    using ninferno::ContextCostPresetSource;
    const auto resolve = [](const ninferno::DeviceClass& device) {
        return ninferno::runtime::resolve_context_machine_cost(
            ninferno::runtime::context_cost_identity(device, "unmeasured-binding"));
    };
    const auto rtx4090 = resolve(board_class(0x2684, 8, 9, 128, 72));
    const auto rtx5090 = resolve(board_class(0x2B85, 12, 0, 170, 96));

    // RTX 4090 D (114 SMs): the RTX 4090's machine, its prefill unscaled; no transfer to carry.
    const auto d = resolve(board_class(0x2685, 8, 9, 114, 72));
    expect(d.summary.gpu_class == "ad102:114sm" &&
               d.summary.machine == "carried from ad102:128sm at 0.17" &&
               d.summary.prefill_source == ContextCostPresetSource::CarriedMachine &&
               d.summary.transfer_source == ContextCostPresetSource::GenericDefault &&
               d.model == rtx4090.model,
           ("the RTX 4090 D did not carry the RTX 4090's machine as measured: " + d.summary.machine)
               .c_str());
    // RTX PRO 6000 (188 SMs, 128 MiB L2): the RTX 5090's, transfer and prefill.
    const auto pro = resolve(board_class(0x2BB1, 12, 0, 188, 128));
    expect(
        pro.summary.machine == "carried from gb202:170sm at 0.35" &&
            pro.summary.transfer_source == ContextCostPresetSource::CarriedMachine &&
            pro.summary.prefill_source == ContextCostPresetSource::CarriedMachine &&
            pro.model == rtx5090.model,
        ("the RTX PRO 6000 did not carry the RTX 5090's machine: " + pro.summary.machine).c_str());
    // RTX 3090 Ti: the RTX 3090's.
    const auto ti = resolve(board_class(0x2203, 8, 6, 84, 6));
    expect(ti.summary.machine == "carried from ga102:82sm at 0.14",
           ("the RTX 3090 Ti did not carry the RTX 3090's machine: " + ti.summary.machine).c_str());

    const auto generic = ninferno::runtime::generic_context_machine_cost_model();

    struct Far {
        const char* what;
        ninferno::DeviceClass device;
        const char* gpu_class;
    };

    const std::array far{
        // 2.2 from the RTX 5090: once its prefill scaled by 170/84, now the declared default.
        Far{"RTX 5080", board_class(0x2C02, 12, 0, 84, 64), "gb203:84sm"},
        // The RTX 3090's instruction set, another FP32-accumulate rate: once its prefill scaled.
        Far{"A100", board_class(0x20B0, 8, 0, 108, 40), "ga100:108sm"},
        // An architecture of no measured board and an id no table lists.
        Far{"an unknown sm_100 board", board_class(0, 10, 0, 148, 126), "sm100:148sm"},
    };
    for (const Far& f : far) {
        const auto none = resolve(f.device);
        expect(none.summary.gpu_class == f.gpu_class && none.summary.machine == "generic" &&
                   none.summary.transfer_source == ContextCostPresetSource::GenericDefault &&
                   none.summary.prefill_source == ContextCostPresetSource::GenericDefault &&
                   none.model == generic,
               (std::string(f.what) +
                " did not take the generic defaults, labelled: " + none.summary.machine)
                   .c_str());
    }
    // A class no table knows on a board of no facts.
    const auto unknown = ninferno::runtime::resolve_context_machine_cost(
        identity({"unmeasured-machine", "*"}, "unmeasured-binding"));
    expect(unknown.summary.machine == "generic" && unknown.model == generic,
           "an unknown class did not take the generic defaults");

    // The class chain, closest key first: a key of a measured board's own chain is its own, and
    // a nearer key that names nothing does not stop the search.
    const std::vector<ninferno::runtime::ContextCostMachinePreset> compiled{
        {.gpu_class     = "gb202",
         .class_prefill = prefill(170),
         .board         = ninferno::measured::kRtx5090},
        {.gpu_class     = "gb202:188sm",
         .class_prefill = prefill(188),
         .board         = {12, 0, 188, 128, 1792}},
    };
    const auto chip = ninferno::runtime::resolve_context_machine_cost(
        compiled, identity({"gb202:170sm", "gb202", "sm120", "*"}, "any"));
    expect(chip.summary.machine == "own gb202" && chip.model.prefill.chunk_ns == 170,
           "a chip key of the class chain did not stand for the board");
    const auto variant = ninferno::runtime::resolve_context_machine_cost(
        compiled, identity({"gb202:188sm", "gb202", "sm120", "*"}, "any"));
    expect(variant.summary.machine == "own gb202:188sm" && variant.model.prefill.chunk_ns == 188,
           "the variant key did not come before the chip's");
}

void test_class_prefill_fallback() {
    const std::vector<ninferno::runtime::ContextCostMachinePreset> compiled{
        {
            .gpu_class     = "measured-machine",
            .prefill       = {{"measured-binding", prefill(11)}},
            .class_prefill = prefill(22),
        },
        {
            .gpu_class = "binding-only-machine",
            .prefill   = {{"measured-binding", prefill(33)}},
        },
    };
    const auto exact = ninferno::runtime::resolve_context_machine_cost(
        compiled, identity({"measured-machine"}, "measured-binding"));
    expect(exact.model.prefill.chunk_ns == 11 &&
               exact.summary.prefill_source == ninferno::ContextCostPresetSource::CompiledDefault,
           "a measured binding did not take its own prefill over the class prefill");

    const auto miss = ninferno::runtime::resolve_context_machine_cost(
        compiled, identity({"measured-machine"}, "other-binding"));
    expect(miss.model.prefill.chunk_ns == 22 &&
               miss.summary.prefill_source == ninferno::ContextCostPresetSource::CompiledDefault &&
               miss.summary.transfer_source == ninferno::ContextCostPresetSource::GenericDefault,
           "an unmeasured binding did not take its machine's class prefill");

    const auto no_class = ninferno::runtime::resolve_context_machine_cost(
        compiled, identity({"binding-only-machine"}, "other-binding"));
    expect(no_class.summary.prefill_source == ninferno::ContextCostPresetSource::GenericDefault &&
               no_class.model.prefill.chunk_ns ==
                   ninferno::runtime::generic_context_machine_cost_model().prefill.chunk_ns,
           "a machine without a class prefill did not fall back to the generic prefill");

    const std::vector<ninferno::runtime::ContextCostMachinePreset> class_only{
        {.gpu_class = "class-only-machine", .class_prefill = prefill(44)},
    };
    const auto only = ninferno::runtime::resolve_context_machine_cost(
        class_only, identity({"class-only-machine"}, "any"));
    expect(only.model.prefill.chunk_ns == 44,
           "a class prefill alone was not accepted as machine cost data");

    std::vector<ninferno::runtime::ContextCostMachinePreset> invalid = class_only;
    invalid[0].class_prefill->vision_item_ns                       = 0;
    invalid[0].class_prefill->vision_patch_ns_q32                  = 0;
    expect_throw(
        [&] {
            (void)ninferno::runtime::resolve_context_machine_cost(
                invalid, identity({"class-only-machine"}, "any"));
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
        const std::vector<std::string> rtx3090{"ga102:82sm", "ga102", "sm86", "*"};
        const auto compiled = ninferno::runtime::resolve_context_machine_cost(
            identity(rtx3090, "unmeasured-config-and-formats"));
        const auto ga102 =
            ninferno::runtime::resolve_context_machine_cost(identity(rtx3090, "other"));
        expect(compiled.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   compiled.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   compiled.model.prefill.token_ns_q32 == ga102.model.prefill.token_ns_q32 &&
                   compiled.model.prefill.token_ns_q32 !=
                       ninferno::runtime::generic_context_machine_cost_model().prefill.token_ns_q32,
               "the RTX 3090 did not take its own transfer and its class prefill");

        const ninferno::runtime::ContextCostIdentity compiled_identity =
            identity({"gb202:170sm", "gb202", "sm120", "*"}, "unmeasured-config-and-formats");
        const auto class_only = ninferno::runtime::resolve_context_machine_cost(compiled_identity);
        expect(class_only.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   class_only.model.prefill.token_ns_q32 == 944'785'862'335'153,
               "an unmeasured binding on the RTX 5090 did not take its class prefill");

        const auto measured = ninferno::runtime::resolve_context_machine_cost(
            identity(compiled_identity.class_chain,
                     "e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a"));
        expect(measured.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::CompiledDefault &&
                   measured.model.prefill.token_ns_q32 == 375'800'765'711'778,
               "measured bindings did not select their compiled prefill cost");

        const ninferno::runtime::ContextCostIdentity unknown =
            identity({"unmeasured-machine"}, "unmeasured-config-and-formats");
        const auto generic = ninferno::runtime::resolve_context_machine_cost(unknown);
        expect(
            generic.summary.transfer_source == ninferno::ContextCostPresetSource::GenericDefault &&
                generic.summary.prefill_source ==
                    ninferno::ContextCostPresetSource::GenericDefault &&
                generic.model.transfer_ns(ninferno::runtime::ContextTransferDirection::DeviceToHost,
                                          {.payload_bytes = 1024, .copy_operations = 1}) > 0 &&
                generic.model.prefill_ns({.tokens = 1}) > 0,
            "unknown identity did not retain a numerical cost model");

        ninferno::runtime::upsert_context_transfer_cost_atomic(path, unknown.class_chain.front(),
                                                               transfer(77), R"({"run":1})");
        auto resolved = ninferno::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 &&
                   resolved.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::GenericDefault,
               "external transfer did not layer over generic prefill");

        ninferno::runtime::upsert_context_prefill_cost_atomic(path, unknown.class_chain.front(),
                                                              unknown.prefill_signature,
                                                              prefill(91), R"({"run":2})");
        resolved = ninferno::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 && resolved.model.prefill.chunk_ns == 91 &&
                   resolved.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source == ninferno::ContextCostPresetSource::External,
               "independent external transfer/prefill entries did not compose");

        const ninferno::runtime::ContextCostIdentity other_model =
            identity(unknown.class_chain, "other-config-and-formats");
        const auto layered_miss =
            ninferno::runtime::resolve_context_machine_cost(other_model, path);
        expect(layered_miss.summary.transfer_source ==
                       ninferno::ContextCostPresetSource::External &&
                   layered_miss.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::GenericDefault,
               "external model miss incorrectly discarded the matching machine transfer");

        ninferno::runtime::upsert_context_transfer_cost_atomic(path, unknown.class_chain.front(),
                                                               transfer(123), R"({"run":3})");
        const auto parsed = ninferno::runtime::parse_context_cost_presets(read_file(path), "file");
        expect(parsed.size() == 1 && parsed[0].transfer->at(0).batch_ns == 123 &&
                   parsed[0].prefill.size() == 1 && parsed[0].prefill[0].cost.chunk_ns == 91,
               "transfer replacement did not preserve the model prefill list");

        const std::string before = read_file(path);
        expect_throw(
            [&] {
                ninferno::runtime::upsert_context_prefill_cost_atomic(
                    path, unknown.class_chain.front(), unknown.prefill_signature, prefill(),
                    "not-json");
            },
            "invalid provenance was accepted");
        expect(read_file(path) == before, "failed upsert modified the preset file");

        // An external entry is found along the class chain too: one keyed by the chip serves a
        // board the compiled table carries to, over the carried component it measured.
        const std::filesystem::path chip_path = directory / "chip.json";
        ninferno::runtime::upsert_context_transfer_cost_atomic(chip_path, "ad102", transfer(55),
                                                               R"({"run":4})");
        const auto chip = ninferno::runtime::resolve_context_machine_cost(
            identity({"ad102:114sm", "ad102", "sm89", "*"}, "unmeasured-config-and-formats",
                     {8, 9, 114, 72, 1008}),
            chip_path);
        expect(chip.model.transfer[0].batch_ns == 55 &&
                   chip.summary.transfer_source == ninferno::ContextCostPresetSource::External &&
                   chip.summary.prefill_source ==
                       ninferno::ContextCostPresetSource::CarriedMachine &&
                   chip.summary.machine == "carried from ad102:128sm at 0.17",
               "an external entry keyed by the chip did not layer over the carried machine");
        Json stale              = Json::parse(read_file(chip_path));
        stale["schema_version"] = 3;
        expect_throw(
            [&] { (void)ninferno::runtime::parse_context_cost_presets(stale.dump(), "schema-3"); },
            "a preset file of the name-keyed schema 3 was accepted");
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
    test_measured_boards();
    test_carried_and_generic();
    test_resolution_and_atomic_upserts();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
