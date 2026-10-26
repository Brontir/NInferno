// Every setup the lab started fits the plan without the weights. A setup is a row of
// docs/performance/data/bench.jsonl measured through ninferno-serve (it has `flags`: ninferno-bench
// started the server with them and measured a value, so the start succeeded): the card, the
// model and quantization (`recipe`), --max-context, --max-concurrency, the KV codec and the
// speculation the start served (`spec`, `draft_tokens`). Each one is planned as
// `ninferno-serve <recipe> --plan --no-load --plan-device <board>` plans it
// (serve/plan_offline.cpp): the named board (product::plan_device_of_board), the server's options
// from the same flags (parse_serve_options, engine_options_of), the subject of the planned target
// (product::planned_subject) and product::plan_model. The speculation is the explicit one the start
// ended on (--spec <backend> --draft-tokens <k>), so the plan is judged for what was served.
//
// Offline: the models' facts are tests/fixtures/plan/bench_models.json, what
// product::plan_facts_of_gguf (plan_facts_of_config for a safetensors repository) read from each
// recipe's GGUF header and listing on 22.10 (the generator was read_hub's code; the fields are
// PlanModelFacts and PlanTargetShape). A bench row of a recipe without facts fails: add its facts.
//
// A row that does not fit is a failure unless it is a known exception below (each names why);
// an exception that fits now fails too, so the list only shrinks. CPU, no network, no GPU.

#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "product/plan/plan_speculation.h"
#include "runtime/engine/plan_dry_run.h"
#include "serve/generation_service.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#ifndef NINFERNO_SOURCE_DIR
#    error "NINFERNO_SOURCE_DIR must name the source tree"
#endif

namespace {

namespace rt = ninferno::runtime;
namespace pr = ninferno::product;
using Json   = nlohmann::json;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// The board a bench row's card is in core/data/gpu_facts.json, where its reported name is not the
// table's (a hyphenated datacenter name; the 16 GB RTX 4060 Ti, which the shortest-name rule would
// take for the 8 GB one).
std::string board_name(const std::string& gpu_name, double vram_gb) {
    if (gpu_name == "NVIDIA A100-SXM4-40GB") { return "A100 SXM4 40GB"; }
    if (gpu_name == "Tesla V100-SXM2-32GB") { return "Tesla V100 SXM2 32GB"; }
    if (gpu_name == "NVIDIA GeForce RTX 4060 Ti" && vram_gb > 12) { return "RTX 4060 Ti 16GB"; }
    return gpu_name;
}

// The driver model of the lab's machine. The rows do not record the OS: the one WDDM machine is
// the owner's RTX 5090 desktop under WSL 2, driving its display (driver 595.84, the Windows
// driver's version WSL reports, and 31.8 GiB, what WDDM leaves of 32); every other card is a
// rented or local Linux stand, taken as headless.
bool wddm_machine(const std::string& gpu_name) { return gpu_name == "NVIDIA GeForce RTX 5090"; }

std::vector<std::string> words(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> out;
    for (std::string w; in >> w;) { out.push_back(w); }
    return out;
}

ninferno::serve::ServeOptions parse(std::vector<std::string> args) {
    args.insert(args.begin(), {"ninferno-serve", "model.ninferno"});
    std::vector<char*> argv;
    for (std::string& a : args) { argv.push_back(a.data()); }
    return ninferno::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

rt::WeightBasis basis_of(const std::string& name) {
    if (name == "artifact") { return rt::WeightBasis::Artifact; }
    if (name == "gguf_tensors") { return rt::WeightBasis::GgufTensors; }
    if (name == "file_size") { return rt::WeightBasis::FileSize; }
    return rt::WeightBasis::Unknown;
}

pr::PlanSourceFacts facts_of(const Json& j) {
    pr::PlanSourceFacts f;
    rt::PlanModelFacts& m      = f.model;
    m.model_type               = j.at("model_type").get<std::string>();
    m.architecture             = j.at("architecture").get<std::string>();
    m.native_context           = j.at("native_context").get<std::uint32_t>();
    m.full_layers              = j.at("full_layers").get<std::uint32_t>();
    m.full_kv_heads_total      = j.at("full_kv_heads_total").get<std::uint32_t>();
    m.full_key_dim             = j.at("full_key_dim").get<std::uint32_t>();
    m.full_value_dim           = j.at("full_value_dim").get<std::uint32_t>();
    m.sliding_layers           = j.at("sliding_layers").get<std::uint32_t>();
    m.sliding_kv_heads_total   = j.at("sliding_kv_heads_total").get<std::uint32_t>();
    m.sliding_key_dim          = j.at("sliding_key_dim").get<std::uint32_t>();
    m.sliding_value_dim        = j.at("sliding_value_dim").get<std::uint32_t>();
    m.window                   = j.at("window").get<std::uint32_t>();
    m.recurrent_layers         = j.at("recurrent_layers").get<std::uint32_t>();
    m.recurrent_state_elements = j.at("recurrent_state_elements").get<std::uint64_t>();
    m.recurrent_conv_bytes     = j.at("recurrent_conv_bytes").get<std::uint64_t>();
    m.hidden                   = j.at("hidden").get<std::uint32_t>();
    m.ffn_active               = j.at("ffn_active").get<std::uint32_t>();
    m.vocab                    = j.at("vocab").get<std::uint32_t>();
    m.params                   = j.at("params").get<double>();
    m.weight_bytes             = j.at("weight_bytes").get<std::uint64_t>();
    m.weight_basis             = basis_of(j.at("weight_basis").get<std::string>());
    const Json& s              = j.at("shape");
    pr::PlanTargetShape& t     = f.shape;
    t.hidden_size              = s.at("hidden_size").get<std::uint64_t>();
    t.num_hidden_layers        = s.at("num_hidden_layers").get<std::uint64_t>();
    t.vocab_size               = s.at("vocab_size").get<std::uint64_t>();
    t.experts                  = s.at("experts").get<int>();
    t.experts_per_token        = s.at("experts_per_token").get<int>();
    t.active_params            = s.at("active_params").get<double>();
    t.routed_expert_params     = s.at("routed_expert_params").get<double>();
    t.builtin_mtp              = s.at("builtin_mtp").get<bool>();
    t.weight_formats           = s.at("weight_formats").get<std::vector<std::string>>();
    if (const auto base = s.at("base_model").get<std::string>(); !base.empty()) {
        t.base_models = {base};
    }
    return f;
}

struct Setup {
    std::string gpu, board, recipe, flags, kv, spec;
    double vram_gb   = 0;
    int draft_tokens = 0;

    std::string label() const {
        return gpu + " | " + recipe + " | " + flags + " | kv " + kv + " | spec " + spec +
               (draft_tokens != 0 ? ":" + std::to_string(draft_tokens) : std::string());
    }

    auto key() const { return std::tie(gpu, recipe, flags, kv, spec, draft_tokens); }

    bool operator<(const Setup& o) const { return key() < o.key(); }
};

// Started setups the plan without the weights does not fit yet, and why. Each is a gap of the
// estimate, not of the start (the start is the authority): the list only shrinks. An empty field
// matches every setup; an exception every setup of which fits is stale and fails.
struct Exception {
    std::string gpu, recipe, flags, spec;
    const char* why;
};

const std::vector<Exception> kKnownExceptions = {
    // None: gpt-oss-20b's listing counted original/model.safetensors beside the served shards
    // until list_quants took the top-level weights only, as the fetch does.
};

const Exception* known_exception(const Setup& s) {
    const auto matches = [](const std::string& want, const std::string& have) {
        return want.empty() || want == have;
    };
    for (const Exception& e : kKnownExceptions) {
        if (matches(e.gpu, s.gpu) && matches(e.recipe, s.recipe) && matches(e.flags, s.flags) &&
            matches(e.spec, s.spec)) {
            return &e;
        }
    }
    return nullptr;
}

std::string gib(std::uint64_t bytes) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(2);
    out << static_cast<double>(bytes) / kGiB << " GiB";
    return out.str();
}

} // namespace

int main() {
    const std::string root = NINFERNO_SOURCE_DIR;
    std::ifstream models_in(root + "/tests/fixtures/plan/bench_models.json");
    check(models_in.good(), "tests/fixtures/plan/bench_models.json opens");
    const Json models = models_in.good() ? Json::parse(models_in) : Json::object();

    std::ifstream bench(root + "/docs/performance/data/bench.jsonl");
    check(bench.good(), "docs/performance/data/bench.jsonl opens");
    std::set<Setup> setups;
    std::size_t rows = 0, started = 0;
    for (std::string line; std::getline(bench, line);) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) { continue; }
        const Json row = Json::parse(line);
        ++rows;
        // A server measurement: ninferno-bench started ninferno-serve with `flags` and measured a
        // value. The rows without flags are ninferno-perplexity's quality runs (a fixed 4,096-token
        // window, not a server start).
        if (!row.contains("flags") || !row["flags"].is_string() || !row["value"].is_number()) {
            continue;
        }
        ++started;
        const Json& machine = row.at("machine");
        Setup s;
        s.gpu     = machine.at("gpu_name").get<std::string>();
        s.vram_gb = machine.at("vram_gb").get<double>();
        s.board   = board_name(s.gpu, s.vram_gb);
        s.recipe  = row.at("recipe").get<std::string>();
        // The flags without --spec: the setup's speculation is what the start served.
        const std::vector<std::string> given = words(row.at("flags").get<std::string>());
        std::string flags;
        for (std::size_t i = 0; i < given.size(); ++i) {
            if (given[i] == "--spec") {
                ++i;
                continue;
            }
            flags += (flags.empty() ? "" : " ") + given[i];
        }
        s.flags = flags;
        s.kv    = row.value("kv", std::string("auto"));
        s.spec  = row.contains("spec") && row["spec"].is_string() ? row["spec"].get<std::string>()
                                                                  : std::string("none");
        if (row.contains("draft_tokens") && row["draft_tokens"].is_number_integer()) {
            s.draft_tokens = row["draft_tokens"].get<int>();
        }
        setups.insert(s);
    }
    check(started > 0, "bench.jsonl has started server rows");

    std::size_t fit = 0, exceptions = 0;
    std::set<const Exception*> used; // the exceptions a setup that does not fit needs
    for (const Setup& s : setups) {
        if (!models.contains(s.recipe)) {
            check(false, "no facts for " + s.recipe +
                             " in tests/fixtures/plan/bench_models.json (add them)");
            continue;
        }
        const pr::PlanSourceFacts facts = facts_of(models.at(s.recipe));

        // The device: the named board, with the lab machine's driver model and display.
        const bool wddm         = wddm_machine(s.gpu);
        const std::string board = s.board + (wddm ? ",wddm,display" : ",linux,headless");
        std::string error;
        const std::optional<rt::PlanDeviceFacts> device = pr::plan_device_of_board(board, &error);
        if (!device) {
            check(false, s.label() + ": no board: " + error);
            continue;
        }
        // The board is the card (within a tenth of what the lab's driver reported).
        check(std::abs(static_cast<double>(device->total_bytes) / kGiB - s.vram_gb) <
                  0.1 * s.vram_gb,
              s.label() + ": board " + device->name + " is not the card");
        const ninferno::spec::GpuKey gpu = pr::gpu_key_of(*device);

        // The server's options from the same flags, with the speculation the start served.
        std::vector<std::string> args = words(s.flags);
        const bool drafts             = s.spec != "none" && s.spec != "None" && !s.spec.empty();
        args.insert(args.end(), {"--spec", drafts ? s.spec : "none"});
        if (drafts && s.draft_tokens != 0) {
            args.insert(args.end(), {"--draft-tokens", std::to_string(s.draft_tokens)});
        }
        if (s.kv != "auto") { args.insert(args.end(), {"--kv-dtype", s.kv}); }
        const ninferno::serve::ServeOptions serve = parse(args);

        // planning_of (serve/plan_offline.cpp).
        pr::DryRunInputs inputs;
        inputs.device          = *device;
        inputs.gpu             = gpu;
        inputs.options         = ninferno::serve::engine_options_of(serve);
        const std::string repo = words(s.recipe).front();
        inputs.ask             = pr::SpeculationAsk{.automatic          = serve.speculation_from_profile,
                                                    .given              = serve.speculative,
                                                    .thinking           = serve.enable_thinking.value_or(true),
                                                    .per_request        = serve.per_request_speculation,
                                                    .per_request_from_profile =
                                                        serve.speculation_from_profile &&
                                                        !serve.per_request_given,
                                                    .depth_from_profile = serve.draft_tokens_from_model};
        const pr::SpeculationSubject subject = pr::planned_subject(
            pr::PlannedTarget{.repo          = repo,
                              .facts         = facts,
                              .gguf          = s.recipe.find(" --quant ") != std::string::npos,
                              .with_drafters = serve.with_drafters},
            gpu);
        inputs.subject                       = &subject;
        const pr::PlannedSpeculation planned = pr::plan_model(facts, inputs);
        const rt::DryRunPlan& plan           = planned.plan;

        std::ostringstream line;
        line << (plan.fits ? "fits   " : "NO FIT ") << s.label() << " | board " << device->name
             << (wddm ? " (WDDM, display)" : " (Linux, headless)") << " | free "
             << gib(plan.free_bytes) << ", weights " << gib(plan.weight_bytes);
        if (plan.fits) {
            line << " | rung " << plan.rung + 1 << " of " << plan.ladder.size() << " ("
                 << rt::plan_reserve_name(plan.reserve) << ")";
        } else {
            line << " | " << (plan.reason.empty() ? std::string("no reason") : plan.reason);
        }
        std::cout << line.str() << '\n';

        const Exception* known = known_exception(s);
        if (known != nullptr && !plan.fits) {
            used.insert(known);
            ++exceptions;
            std::cout << "       known exception: " << known->why << '\n';
            continue;
        }
        check(plan.planned, s.label() + ": not planned: " + plan.reason);
        check(plan.fits, s.label() + ": started by the lab, but the plan does not fit");
        if (plan.fits) { ++fit; }
    }
    for (const Exception& e : kKnownExceptions) {
        check(used.contains(&e), "the known exception " + e.gpu + " | " + e.recipe + " | " +
                                     e.flags + " | " + e.spec +
                                     " is stale: every setup it matches fits (remove it)");
    }

    std::cout << rows << " bench rows, " << started << " server measurements, " << setups.size()
              << " setups: " << fit << " fit, " << exceptions << " known exceptions\n";
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ninferno_plan_bench_fits_test: OK\n";
    return 0;
}
