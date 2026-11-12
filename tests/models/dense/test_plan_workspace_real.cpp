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
// artifacts in the Hugging Face cache (convert::default_hub_cache, ModelScope's repos under its
// modelscope/ too): those ninferno-ui keeps (models--*/ninferno/<commit>/*.ninferno) and the
// converter's cache (models--*/snapshots/<commit>/.ninferno-cache/*.ninferno); the smallest of
// each family whose file fits this card's free memory. The scanned files and the picks are
// printed. A family without one is skipped and said; none at all skips the test (77). A file in
// this build's container (its "NInferno" magic) that the reader refuses fails the test: no silent
// skip where artifacts exist. The scanner is checked first on a temporary layout (no GPU).
// GPU.

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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

bool family(const std::string& type) {
    return type == "dense_decoder" || type == "moe_decoder" || type == "nemotron_h";
}

// The .ninferno files of `dir` (none where it is not a directory).
void ninferno_files(const fs::path& dir, std::vector<fs::path>& out) {
    std::error_code error;
    for (const auto& file : fs::directory_iterator(dir, error)) {
        std::error_code kind;
        if (file.is_regular_file(kind) && file.path().extension() == ".ninferno") {
            out.push_back(file.path());
        }
    }
}

// Every converted artifact the hub cache `hub` holds, sorted: per repository (models--*, and
// ModelScope's under modelscope/), ninferno-ui's kept conversions (ninferno/<commit>/, its
// product/plan/plan_cached.cpp) and the converter's cache (snapshots/<commit>/.ninferno-cache/,
// convert/cache.cpp).
std::vector<fs::path> cached_artifacts(const fs::path& hub) {
    std::vector<fs::path> out;
    for (const fs::path& root : {hub, hub / "modelscope"}) {
        std::error_code error;
        for (const auto& repo : fs::directory_iterator(root, error)) {
            std::error_code kind;
            if (!repo.is_directory(kind) ||
                !repo.path().filename().string().starts_with("models--")) {
                continue;
            }
            std::error_code inner;
            for (const auto& commit : fs::directory_iterator(repo.path() / "ninferno", inner)) {
                ninferno_files(commit.path(), out);
            }
            for (const auto& commit : fs::directory_iterator(repo.path() / "snapshots", inner)) {
                ninferno_files(commit.path() / ".ninferno-cache", out);
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// What a cached file is: in this build's container (it starts with the "NInferno" magic; an older
// container, "NINFER", is not one this build reads), its text model_type, or why the reader
// refused it.
struct Probe {
    bool current = false;
    std::string type;
    std::string error;
};

Probe probe(const fs::path& path) {
    Probe out;
    {
        std::ifstream in(path, std::ios::binary);
        char magic[8] = {};
        in.read(magic, sizeof magic);
        out.current = in.gcount() == sizeof magic && std::string(magic, 8) == "NInferno";
    }
    if (!out.current) { return out; }
    try {
        const artifact::Reader reader(path);
        const auto& components = reader.directory().components;
        const auto text        = components.find("text");
        if (text == components.end()) {
            out.error = "no text component";
        } else {
            out.type = text->second.config.value("model_type", std::string());
        }
    } catch (const std::exception& error) { out.error = error.what(); }
    return out;
}

// The picks among `files`: the smallest of each family whose file fits `free_bytes` (the weights
// and the arena: a fifth of the free memory left to the rest). `broken`: a file in this build's
// container the reader refused (a bug path, not a skip). Says what it scanned and picked.
struct Picks {
    std::vector<fs::path> paths;
    bool broken = false;
};

Picks pick(const std::vector<fs::path>& files, std::size_t free_bytes) {
    Picks out;
    std::map<std::string, std::pair<std::uintmax_t, fs::path>> smallest;
    std::map<std::string, std::uintmax_t> too_large;
    for (const fs::path& file : files) {
        const Probe p = probe(file);
        std::error_code size_error;
        const auto bytes = fs::file_size(file, size_error);
        std::cout << "  scanned " << file << ": ";
        if (!p.current) {
            std::cout << "another container version, not read\n";
            continue;
        }
        if (!p.error.empty() || size_error) {
            std::cout << "REFUSED: " << (p.error.empty() ? size_error.message() : p.error) << '\n';
            out.broken = true;
            continue;
        }
        std::cout << p.type << ", " << bytes << " bytes\n";
        if (!family(p.type)) { continue; }
        if (bytes > free_bytes / 5 * 4) {
            too_large[p.type] = bytes;
            continue;
        }
        auto& best = smallest[p.type];
        if (best.second.empty() || bytes < best.first) { best = {bytes, file}; }
    }
    for (const char* type : {"dense_decoder", "moe_decoder", "nemotron_h"}) {
        if (const auto it = smallest.find(type); it != smallest.end()) {
            std::cout << "picked " << type << ": " << it->second.second << '\n';
            out.paths.push_back(it->second.second);
        } else if (too_large.contains(type)) {
            std::cout << "skip " << type << ": its cached artifacts do not fit this card's "
                      << free_bytes << " free bytes\n";
        } else {
            std::cout << "skip " << type << ": no cached artifact of it\n";
        }
    }
    return out;
}

// The scanner over a temporary layout: both locations found, only .ninferno files, ModelScope's
// repositories included; a file in this build's container the reader refuses is broken, an older
// container's is not read.
bool scanner_checks() {
    const fs::path root =
        fs::temp_directory_path() /
        ("ninferno_plan_scan_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(root);
    const auto touch = [](const fs::path& path, const std::string& bytes) {
        fs::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << bytes;
    };
    const fs::path kept   = root / "models--org--a" / "ninferno" / "c1" / "a.Q4_K_M.ninferno";
    const fs::path cached = root / "models--org--a" / "snapshots" / "c1" / ".ninferno-cache" /
                            "a-Q8_0.dense_bf16.ninferno";
    const fs::path scope =
        root / "modelscope" / "models--org--b" / "ninferno" / "c2" / "b.ninferno";
    touch(kept, "NINFER, an older container");
    touch(cached, "NInferno and then garbage");
    touch(scope, "NINFER, an older container");
    touch(root / "models--org--a" / "snapshots" / "c1" / "a-Q8_0.gguf", "GGUF");
    touch(root / "models--org--a" / "ninferno" / "c1" / "a.ninfer", "NINFER");
    touch(root / "models--org--a" / "snapshots" / "c1" / "a.ninferno", "not in the cache dir");
    touch(root / "other--x" / "ninferno" / "c" / "x.ninferno", "not a repository");
    const auto found           = cached_artifacts(root);
    std::vector<fs::path> want = {cached, kept, scope};
    std::sort(want.begin(), want.end());
    bool ok = true;
    if (found != want) {
        std::cerr << "FAIL: the scanner found " << found.size() << " file(s), not the 3 laid out\n";
        ok = false;
    }
    const Picks picks = pick(found, std::size_t{1} << 40);
    if (!picks.broken || !picks.paths.empty()) {
        std::cerr << "FAIL: a refused file in this build's container is not reported broken\n";
        ok = false;
    }
    const Picks older = pick({kept, scope}, std::size_t{1} << 40);
    if (older.broken || !older.paths.empty()) {
        std::cerr << "FAIL: an older container's file is not skipped\n";
        ok = false;
    }
    fs::remove_all(root);
    return ok;
}

// The artifacts to check: the environment's list, else the cache's picks.
Picks artifacts(std::size_t free_bytes) {
    if (const char* env = std::getenv("NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS"); env && *env) {
        Picks out;
        std::stringstream paths(env);
        std::string path;
        while (std::getline(paths, path, ':')) {
            if (!path.empty()) { out.paths.emplace_back(path); }
        }
        return out;
    }
    const fs::path hub = convert::default_hub_cache();
    std::cout << "scanning " << hub << '\n';
    return pick(cached_artifacts(hub), free_bytes);
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

int main(int argc, char** argv) {
    try {
        // The scanner first, on the host: a broken scan must not pass as "no artifact".
        if (!scanner_checks()) { return 1; }
        if (argc > 1 && std::string(argv[1]) == "--scan-only") {
            // The host's own cache scanned and picked as the GPU run would, any size: no device.
            const fs::path hub = convert::default_hub_cache();
            std::cout << "scanning " << hub << '\n';
            const Picks picks =
                pick(cached_artifacts(hub), std::numeric_limits<std::size_t>::max());
            return picks.broken ? 1 : 0;
        }
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
        const Picks picks = artifacts(free_bytes);
        check(!picks.broken, "every cached artifact in this build's container is read (REFUSED "
                             "above: a bug path, not a skip)");
        const std::vector<fs::path>& paths = picks.paths;
        if (paths.empty() && !picks.broken) {
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
