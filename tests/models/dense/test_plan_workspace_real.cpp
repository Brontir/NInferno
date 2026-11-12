// The plan without the weights against a real start (#615 W3), for the dense decoder, moe_decoder
// and NemotronH: each family's artifact is loaded as ninferno-serve loads it (load_dense,
// load_nemotron_h: its profile, its tokenizer's public ids), and the arena the start's execution
// plans (ModelExecution::workspace_plan) is checked equal to the byte to what --plan sizes for the
// same rung from the artifact's facts alone (product::plan_facts_of_artifact, then
// runtime::plan_rung_bytes with the workspace query ninferno-serve registers), with "start" as its
// basis, under --activations fast and exact, on this card's routes. The tokenizer's count the
// facts carry is checked against the one the start's frontend takes.
//
// The artifacts: NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS (paths joined by ':'), else the converted
// artifacts in the Hugging Face cache (convert::default_hub_cache: models--*/ninfer/*/<file>): the
// smallest of each family whose file fits this card's free memory. A family without one is skipped
// and said; none at all skips the test (77). GPU.

#include "artifact/reader.h"
#include "convert/fetch.h"
#include "core/device.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "product/plan/plan_facts.h"
#include "runtime/engine/hf_text_frontend.h"
#include "runtime/engine/model_instance.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_workspace.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

namespace {

using namespace ninferno;
namespace detail = ninferno::models::qwen3_5::detail;
namespace fs     = std::filesystem;
namespace rt     = ninferno::runtime;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

// The text component's model_type of a readable artifact ("" where it is not one this build
// reads: another container version, no text component).
std::string model_type_of(const fs::path& path) {
    try {
        const artifact::Reader reader(path);
        const auto& components = reader.directory().components;
        const auto text        = components.find("text");
        return text == components.end() ? std::string()
                                        : text->second.config.value("model_type", std::string());
    } catch (const std::exception&) { return {}; }
}

bool family(const std::string& type) {
    return type == "dense_decoder" || type == "moe_decoder" || type == "nemotron_h";
}

// The artifacts to check: the environment's list, else the smallest cached one of each family
// that fits `free_bytes`.
std::vector<fs::path> artifacts(std::size_t free_bytes) {
    std::vector<fs::path> out;
    if (const char* env = std::getenv("NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS"); env && *env) {
        std::stringstream paths(env);
        std::string path;
        while (std::getline(paths, path, ':')) {
            if (!path.empty()) { out.emplace_back(path); }
        }
        return out;
    }
    const fs::path hub = convert::default_hub_cache();
    std::map<std::string, std::pair<std::uintmax_t, fs::path>> smallest;
    std::error_code error;
    for (const auto& repo : fs::directory_iterator(hub, error)) {
        if (!repo.is_directory() || !repo.path().filename().string().starts_with("models--")) {
            continue;
        }
        std::error_code inner;
        for (const auto& revision : fs::directory_iterator(repo.path() / "ninfer", inner)) {
            std::error_code files_error;
            for (const auto& file : fs::directory_iterator(revision.path(), files_error)) {
                const std::string ext = file.path().extension().string();
                if (!file.is_regular_file() || (ext != ".ninfer" && ext != ".ninferno")) {
                    continue;
                }
                std::error_code size_error;
                const auto bytes = fs::file_size(file.path(), size_error);
                // The weights and the arena: leave a fifth of the free memory to the rest.
                if (size_error || bytes > free_bytes / 5 * 4) { continue; }
                const std::string type = model_type_of(file.path());
                if (!family(type)) { continue; }
                auto& best = smallest[type];
                if (best.second.empty() || bytes < best.first) { best = {bytes, file.path()}; }
            }
        }
    }
    for (const char* type : {"dense_decoder", "moe_decoder", "nemotron_h"}) {
        if (const auto it = smallest.find(type); it != smallest.end()) {
            out.push_back(it->second.second);
        } else {
            std::cout << "skip " << type << ": no cached artifact of it fits this card (" << hub
                      << ")\n";
        }
    }
    return out;
}

// The plan a start's planner lays out for `options` (make_sequence_planner_impl: the options the
// family validated, its copy window, the smallest candidate), its workspace not yet planned.
detail::SequencePlanImpl plan_of(const EngineOptions& options, models::StateFamily state,
                                 std::uint64_t positions, std::int32_t sms) {
    detail::validate_program_options(options, positions, 0);
    const detail::SequencePlanningInputs inputs =
        detail::sequence_planning_inputs(options, detail::family_copy_rounds(state), sms, false);
    const auto range = detail::sequence_page_range(inputs.capacity, inputs.max_concurrency);
    return std::move(*detail::sequence_plan_shell(inputs, range.minimum));
}

EngineOptions options_of(const fs::path& path, std::uint32_t context, std::uint32_t chunk,
                         std::uint32_t concurrency, ActivationMode activations) {
    EngineOptions o;
    o.artifact_path   = path;
    o.max_context     = context;
    o.prefill_chunk   = chunk;
    o.max_concurrency = concurrency;
    o.kv_capacity     = KvCapacityPolicy::explicit_capacity(context * concurrency);
    o.kv_cache        = KvCacheStorage::BFloat16;
    o.activations     = activations;
    return o;
}

void run(const fs::path& path, DeviceContext& device) {
    const std::string name = path.filename().string();
    const auto facts       = product::plan_facts_of_artifact(path).model;
    check(facts.artifact_directory != nullptr && facts.tokenizer_tokens != 0,
          name + ": --plan's facts keep the directory and the tokenizer's ids");
    if (facts.artifact_directory == nullptr || facts.tokenizer_tokens == 0) { return; }
    const bool nemotron = facts.model_type == "nemotron_h";
    const auto state =
        nemotron ? models::StateFamily::NemotronH : models::StateFamily::DenseDecoder;
    const std::int32_t sms     = device.device_class().sms;
    const std::uint32_t native = facts.native_context;
    for (const ActivationMode activations : {ActivationMode::Fast, ActivationMode::Exact}) {
        const char* mode = activations == ActivationMode::Exact ? "exact" : "fast";
        // The start: the family's load, its execution over the tokenizer's public ids.
        const EngineOptions loaded =
            options_of(path, std::min<std::uint32_t>(32768, native), 2048, 1, activations);
        rt::FamilyLoad start =
            nemotron ? rt::load_nemotron_h(loaded, device) : rt::load_dense(loaded, device);
        check(start.execution->traits().public_tokens ==
                  static_cast<std::int32_t>(facts.tokenizer_tokens),
              name + ": the facts' tokenizer ids " + std::to_string(facts.tokenizer_tokens) +
                  " are the start's " + std::to_string(start.execution->traits().public_tokens));
        for (const auto& [context, chunk, concurrency] :
             {std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{32768, 2048, 1},
              {65536, 4096, 4}}) {
            const EngineOptions options =
                options_of(path, std::min(context, native), chunk, concurrency, activations);
            const auto plan   = plan_of(options, state, native, sms);
            const auto arenas = start.execution->workspace_plan(plan);
            rt::PlanRung rung;
            rung.max_context   = options.max_context;
            rung.prefill_chunk = options.prefill_chunk;
            const auto dry =
                rt::plan_rung_bytes(facts, rung, options, options.max_concurrency, false);
            const std::string what =
                name + " (" + mode + ", ctx " + std::to_string(options.max_context) + ", chunk " +
                std::to_string(chunk) + ", " + std::to_string(concurrency) + " streams)";
            std::cout << what << ": start " << arenas.capacity << ", --plan " << dry.workspace
                      << " (" << dry.workspace_basis << ")\n";
            check(dry.workspace_basis.rfind("start", 0) == 0,
                  what + ": --plan sizes the workspace by the start's planner (" +
                      dry.workspace_basis + ")");
            check(dry.workspace == arenas.capacity,
                  what + ": --plan's workspace " + std::to_string(dry.workspace) +
                      " is the start's " + std::to_string(arenas.capacity));
        }
    }
}

} // namespace

int main() {
    try {
        rt::register_plan_workspace_queries();
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }
        DeviceContext device(0);
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            std::cout << "SKIP: the device's memory is not known\n";
            return 77;
        }
        const auto paths = artifacts(free_bytes);
        if (paths.empty()) {
            std::cout << "SKIP: no dense_decoder, moe_decoder or nemotron_h artifact (set "
                         "NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS or convert one into the Hugging "
                         "Face cache)\n";
            return 77;
        }
        for (const auto& path : paths) { run(path, device); }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "plan workspace real: pass\n";
    return 0;
}
