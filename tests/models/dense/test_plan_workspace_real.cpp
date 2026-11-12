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
// modelscope/ too): those ninferno-ui keeps (models--*/ninferno/<commit>/*.ninferno, and a
// conversion under options, ninferno/<commit>/options-<key>/, convert/acquire.cpp) and the
// converter's cache (a .ninferno-cache/ anywhere under models--*/snapshots/). Of each family the
// candidates that fit this card's free memory are tried smallest first: one whose load or plan says
// this build or card cannot run it (an sm_120a-only format, a missing instruction) is reported and
// the next is tried; any other error fails. The scanned files, the picks and the outcomes are
// printed. A file in this build's container (its "NInferno" magic) that the reader refuses is
// reported (the shared cache may hold another branch's conversion): a warning where an artifact
// was tested, a failure where none was (no silent skip where artifacts exist). A family without a
// candidate is skipped and said; nothing at all skips the test (77). The scanner and the verdict
// are checked first on a temporary layout (no GPU). GPU.

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

// Calls `f` on each entry of `dir`, advancing with increment(error): a cache cleanup running beside
// the scan ends the walk of that directory, never the test.
template <class F>
void each_entry(const fs::path& dir, F&& f) {
    std::error_code error;
    for (fs::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
        f(*it);
    }
}

bool is_ninferno(const fs::directory_entry& entry) {
    std::error_code kind;
    return entry.is_regular_file(kind) && entry.path().extension() == ".ninferno";
}

// Every converted artifact the hub cache `hub` holds, sorted: per repository (models--*, and
// ModelScope's under modelscope/), ninferno-ui's kept conversions (ninferno/<commit>/, and
// ninferno/<commit>/options-<key>/: convert::conversion_directory) and the converter's cache (a
// .ninferno-cache/ anywhere under snapshots/, as product/plan/plan_cached.cpp finds it).
std::vector<fs::path> cached_artifacts(const fs::path& hub) {
    std::vector<fs::path> out;
    const auto take = [&](const fs::directory_entry& entry) {
        if (is_ninferno(entry)) { out.push_back(entry.path()); }
    };
    for (const fs::path& root : {hub, hub / "modelscope"}) {
        each_entry(root, [&](const fs::directory_entry& repo) {
            std::error_code kind;
            if (!repo.is_directory(kind) ||
                !repo.path().filename().string().starts_with("models--")) {
                return;
            }
            each_entry(repo.path() / "ninferno", [&](const fs::directory_entry& commit) {
                each_entry(commit.path(), [&](const fs::directory_entry& entry) {
                    std::error_code dir_kind;
                    if (entry.is_directory(dir_kind) &&
                        entry.path().filename().string().starts_with("options-")) {
                        each_entry(entry.path(), take);
                    } else {
                        take(entry);
                    }
                });
            });
            std::error_code error;
            for (fs::recursive_directory_iterator it(repo.path() / "snapshots", error), end;
                 !error && it != end; it.increment(error)) {
                if (is_ninferno(*it) && it->path().parent_path().filename() == ".ninferno-cache") {
                    out.push_back(it->path());
                }
            }
        });
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

// The scan's candidates: each family's readable files that fit `free_bytes` (the weights and the
// arena: a fifth of the free memory left to the rest), smallest first; the files of this build's
// container the reader refused (their family unknown). Says what it scanned.
struct Candidates {
    std::map<std::string, std::vector<fs::path>> family;
    std::map<std::string, std::uintmax_t> too_large; // a family's smallest that does not fit
    std::vector<fs::path> refused;
};

Candidates candidates(const std::vector<fs::path>& files, std::size_t free_bytes) {
    Candidates out;
    std::map<std::string, std::vector<std::pair<std::uintmax_t, fs::path>>> sized;
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
            out.refused.push_back(file);
            continue;
        }
        std::cout << p.type << ", " << bytes << " bytes\n";
        if (!family(p.type)) { continue; }
        if (bytes > free_bytes / 5 * 4) {
            auto& smallest = out.too_large[p.type];
            smallest       = smallest == 0 ? bytes : std::min<std::uintmax_t>(smallest, bytes);
            continue;
        }
        sized[p.type].push_back({bytes, file});
    }
    for (auto& [type, files_of] : sized) {
        std::sort(files_of.begin(), files_of.end());
        for (const auto& [bytes, file] : files_of) { out.family[type].push_back(file); }
    }
    return out;
}

// Whether `error` says this build or this card cannot run the artifact (its formats or the
// instructions it needs), not that the start or the plan went wrong: linear.cpp's and
// generic_gemm.cpp's sm_120a-only NVFP4 paths, a format this build has no kernels for, and
// require_device_support's missing instructions.
bool unsupported_here(const std::string& error) {
    for (const char* says : {"sm_120a path", "this build has no kernels", "cannot run on"}) {
        if (error.find(says) != std::string::npos) { return true; }
    }
    return false;
}

// The run's outcome where nothing else failed: 1 where files of this build's container were
// refused and no artifact was tested (whatever families they hold, none was read and tested); 77
// where nothing was tested and nothing refused; else 0, the refusals warnings beside a test.
int verdict(std::size_t tested, std::size_t refused) {
    if (tested == 0) { return refused != 0 ? 1 : 77; }
    return 0;
}

// The scanner, the classifier and the verdict over a temporary layout: every location found
// (a kept conversion, one under options, the converter's cache however deep, ModelScope's), only
// .ninferno files; a file of this build's container the reader refuses is listed refused, an older
// container's is not read; refusals fail only where nothing was tested.
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
    const fs::path repo    = root / "models--org--a";
    const fs::path kept    = repo / "ninferno" / "c1" / "a.Q4_K_M.ninferno";
    const fs::path options = repo / "ninferno" / "c1" / "options-0123abcd" / "a.Q4_K_M.ninferno";
    const fs::path cached = repo / "snapshots" / "c1" / ".ninferno-cache" / "a-Q8_0.dense.ninferno";
    const fs::path nested = repo / "snapshots" / "c1" / "q8" / ".ninferno-cache" / "a.ninferno";
    const fs::path scope =
        root / "modelscope" / "models--org--b" / "ninferno" / "c2" / "b.ninferno";
    for (const fs::path& older : {kept, options, nested, scope}) {
        touch(older, "NINFER, an older container");
    }
    touch(cached, "NInferno and then garbage");
    touch(repo / "snapshots" / "c1" / "a-Q8_0.gguf", "GGUF");
    touch(repo / "ninferno" / "c1" / "a.ninfer", "NINFER");
    touch(repo / "ninferno" / "c1" / "a.ninferno.relayout-journal", "NInferno");
    touch(repo / "snapshots" / "c1" / "a.ninferno", "not in the cache dir");
    touch(root / "other--x" / "ninferno" / "c" / "x.ninferno", "not a repository");
    const auto found           = cached_artifacts(root);
    std::vector<fs::path> want = {cached, kept, nested, options, scope};
    std::sort(want.begin(), want.end());
    bool ok          = true;
    const auto fails = [&](bool bad, const std::string& what) {
        if (bad) {
            std::cerr << "FAIL: scanner self-check: " << what << '\n';
            ok = false;
        }
    };
    fails(found != want, "found " + std::to_string(found.size()) + " file(s), not the 5 laid out");
    const Candidates all = candidates(found, std::size_t{1} << 40);
    fails(all.refused != std::vector<fs::path>{cached} || !all.family.empty(),
          "the garbage in this build's container is not the one refused file");
    const Candidates older = candidates({kept, scope}, std::size_t{1} << 40);
    fails(!older.refused.empty() || !older.family.empty(), "an older container's file is read");
    fails(verdict(0, 1) != 1, "refused files with nothing tested do not fail");
    fails(verdict(1, 1) != 0, "refused files beside a tested artifact are not warnings");
    fails(verdict(0, 0) != 77, "nothing to test does not skip");
    fails(!unsupported_here("NVFP4 linear is a sm_120a path; this build has no kernels") ||
              !unsupported_here("nemotron_h runtime cannot run on RTX 2080 Ti (sm_75): missing "
                                "BF16 mma.sync") ||
              unsupported_here("kv_capacity is outside the usable range"),
          "the unsupported-here classifier");
    fs::remove_all(root);
    return ok;
}

// The candidates to check: the environment's list (each probed the same way), else the cache's.
Candidates artifacts(std::size_t free_bytes) {
    if (const char* env = std::getenv("NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS"); env && *env) {
        std::vector<fs::path> files;
        std::stringstream paths(env);
        std::string path;
        while (std::getline(paths, path, ':')) {
            if (!path.empty()) { files.emplace_back(path); }
        }
        std::cout << "the artifacts NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS names\n";
        return candidates(files, free_bytes);
    }
    const fs::path hub = convert::default_hub_cache();
    std::cout << "scanning " << hub << '\n';
    return candidates(cached_artifacts(hub), free_bytes);
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

// The families' candidates in order, each family's first that this build and card can run
// tested; returns how many were tested.
std::size_t test_families(const Candidates& found, DeviceContext* device, std::size_t free_bytes) {
    std::size_t tested = 0;
    for (const char* type : {"dense_decoder", "moe_decoder", "nemotron_h"}) {
        const auto it = found.family.find(type);
        if (it == found.family.end()) {
            if (const auto large = found.too_large.find(type); large != found.too_large.end()) {
                std::cout << "skip " << type << ": its smallest cached artifact (" << large->second
                          << " bytes) does not fit this card's " << free_bytes << " free bytes\n";
            } else {
                std::cout << "skip " << type << ": no cached artifact of it\n";
            }
            continue;
        }
        bool done = false;
        for (const fs::path& path : it->second) {
            std::cout << "picked " << type << ": " << path << '\n';
            if (device == nullptr) { // --scan-only: the pick alone
                done = true;
                break;
            }
            try {
                run(path, *device);
                done = true;
            } catch (const std::exception& error) {
                if (unsupported_here(error.what())) {
                    std::cout << "  not runnable on this build or card (" << error.what()
                              << "): the next candidate\n";
                    continue;
                }
                check(false, path.string() + ": " + error.what());
                done = true; // failed, said: not a skip
            }
            break;
        }
        if (done) {
            ++tested;
        } else {
            std::cout << "skip " << type << ": none of its cached artifacts runs on this build "
                      << "or card\n";
        }
    }
    return tested;
}

// Says the outcome of the scan: a refusal is a warning where an artifact was tested, a failure
// where none was.
int outcome(const Candidates& found, std::size_t tested) {
    const int v = verdict(tested, found.refused.size());
    for (const fs::path& file : found.refused) {
        std::cout << (v == 1 ? "FAIL: " : "WARNING: ") << "refused " << file
                  << (v == 1 ? " and nothing was tested\n" : " (another artifact was tested)\n");
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    try {
        // The scanner first, on the host: a broken scan must not pass as "no artifact".
        if (!scanner_checks()) { return 1; }
        if (argc > 1 && std::string(argv[1]) == "--scan-only") {
            // The host's own cache scanned and picked as the GPU run would, any size: no device.
            const std::size_t any  = std::numeric_limits<std::size_t>::max();
            const Candidates found = artifacts(any);
            const int v            = outcome(found, test_families(found, nullptr, any));
            return v == 77 ? 0 : v;
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
        const Candidates found = artifacts(free_bytes);
        const int v            = outcome(found, test_families(found, &device, free_bytes));
        if (failures == 0 && v == 77) {
            std::cout << "SKIP: no dense_decoder, moe_decoder or nemotron_h artifact this build "
                         "and card run (set NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS or convert one "
                         "into the Hugging Face cache)\n";
            return 77;
        }
        if (v == 1) { ++failures; }
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
