#include "ninferno_bench_support.h"

#include "core/device_class.h"
#include "ninferno/engine.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i != 0) { out << ' '; }
        out << argv[i];
    }
    return out.str();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

void fill_cuda_environment(ninferno::bench::BenchEnvironment& env, int device) {
    env.device_id       = device;
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        env.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        env.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        env.gpu_name = properties.name;
    }
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool has_decode_tests(const std::vector<ninferno::bench::BenchTest>& tests) {
    for (const auto& test : tests) {
        if (test.has_decode()) { return true; }
    }
    return false;
}

ninferno::RequestOptions benchmark_request(const ninferno::bench::BenchTest& test) {
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens = test.requested_output_tokens();
    options.execution.allow_prefix_reuse      = false;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    options.output.preserve_special_tokens    = true;
    return options;
}

ninferno::bench::RepTiming run_repetition(ninferno::Engine& engine,
                                        const ninferno::bench::BenchTest& test,
                                        const std::vector<ninferno::TokenId>& corpus) {
    const int prompt_tokens = test.kind == ninferno::bench::TestKind::Decode
                                  ? ninferno::bench::kDecodeSeedTokens
                                  : test.n_prompt;
    auto prompt =
        engine.prepare_tokens(ninferno::bench::prompt_slice(corpus, prompt_tokens), false);
    ninferno::GenerationResult generated =
        engine.generate(std::move(prompt), benchmark_request(test));

    const std::uint32_t expected = test.requested_output_tokens();
    if (generated.generated_token_ids.size() != expected) {
        throw std::runtime_error(test.label + " generated " +
                                 std::to_string(generated.generated_token_ids.size()) +
                                 " tokens; expected " + std::to_string(expected));
    }
    if (generated.finish_reason != ninferno::FinishReason::OutputLimit) {
        throw std::runtime_error(test.label + " did not finish at the requested output limit");
    }

    ninferno::bench::RepTiming timing;
    timing.timings                 = generated.timings;
    timing.speculative             = std::move(generated.speculative);
    timing.generated_output_tokens = expected;
    return timing;
}

void prime_decode_graph(ninferno::Engine& engine, ninferno::bench::BenchEnvironment& env,
                        const std::vector<ninferno::TokenId>& corpus) {
    if (!env.use_cuda_graph || env.decode_graph_prime_output_tokens == 0) { return; }
    const int decode_tokens = static_cast<int>(env.decode_graph_prime_output_tokens - 1);
    const ninferno::bench::BenchTest prime{ninferno::bench::TestKind::Decode, 0, decode_tokens,
                                         "decode-graph-prime"};
    (void)run_repetition(engine, prime, corpus);
    env.decode_graph_primed = true;
}

void write_output(const ninferno::bench::BenchOptions& options, const std::string& text) {
    if (options.output_file.empty()) {
        std::cout << text;
        return;
    }
    const std::filesystem::path path(options.output_file);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open output file: " + options.output_file); }
    output << text;
    std::cout << "wrote " << options.output_file << '\n';
}

} // namespace

int main(int argc, char** argv) {
    ninferno::bench::BenchOptions options;
    try {
        options = ninferno::bench::parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninferno_bench: " << error.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << ninferno::bench::usage_text(argc > 0 ? argv[0] : "ninferno_bench");
        return 0;
    }
    // As ninferno-serve takes --device: N counts GPUs in PCI bus order.
    try {
        options.device = ninferno::select_device(options.device_selector);
    } catch (const std::invalid_argument& error) {
        std::cerr << "ninferno_bench: " << error.what() << '\n';
        return 2;
    }

    try {
        std::vector<ninferno::TokenId> corpus =
            ninferno::bench::load_corpus_ids(options.corpus_path);
        const std::vector<ninferno::bench::BenchTest> tests =
            ninferno::bench::expand_tests(options);
        if (options.profile_measured && (tests.size() != 1 || options.repetitions != 1)) {
            throw std::invalid_argument(
                "--profile-measured requires exactly one benchmark test and -r 1");
        }
        ninferno::bench::validate_prompt_lengths(tests, corpus.size());
        const std::uint32_t max_context = ninferno::bench::resolve_max_context(
            tests, options.max_context, options.speculative, options.use_cuda_graph);

        ninferno::EngineOptions engine_options;
        engine_options.artifact_path = options.artifact_path;
        engine_options.device        = options.device;
        engine_options.max_context   = max_context;
        engine_options.kv_capacity   = ninferno::KvCapacityPolicy::explicit_capacity(max_context);
        engine_options.prefill_chunk = options.prefill_chunk;
        engine_options.kv_cache      = options.kv_cache;
        engine_options.activations   = options.activations;
        engine_options.context_cache.enabled = false;
        engine_options.speculative           = options.speculative;
        engine_options.use_cuda_graph        = options.use_cuda_graph;

        ninferno::bench::BenchEnvironment env;
        env.artifact_path            = options.artifact_path;
        env.artifact_file_size_bytes = ninferno::bench::file_size_or_zero(options.artifact_path);
        env.max_context              = max_context;
        env.prefill_chunk            = options.prefill_chunk;
        env.kv_cache                 = options.kv_cache;
        env.activations              = options.activations;
        env.speculative              = options.speculative;
        env.use_cuda_graph           = options.use_cuda_graph;
        env.repetitions              = options.repetitions;
        env.warmup                   = options.warmup;
        env.corpus_path              = options.corpus_path;
        env.corpus_tokens            = corpus.size();
        if (options.use_cuda_graph && has_decode_tests(tests)) {
            env.decode_graph_prime_output_tokens =
                ninferno::bench::decode_graph_prime_output_tokens(options.speculative);
        }

        std::cerr << "[ninferno_bench] loading " << options.artifact_path
                  << " (max_context=" << max_context
                  << ", kv_cache=" << ninferno::bench::kv_cache_name(options.kv_cache) << ")\n";
        ninferno::Engine engine(std::move(engine_options));
        fill_cuda_environment(env, options.device);
        // The shipped corpus is Qwen token ids; a model with a smaller vocabulary (Nemotron's
        // 131072) reads them folded into its own, so every artifact runs the same lengths.
        if (const std::uint64_t vocab = engine.model_metadata().vocab_size; vocab > 0) {
            std::size_t folded = 0;
            for (auto& id : corpus) {
                if (static_cast<std::uint64_t>(id) < vocab) { continue; }
                id = static_cast<ninferno::TokenId>(static_cast<std::uint64_t>(id) % vocab);
                ++folded;
            }
            if (folded > 0) {
                std::cerr << "[ninferno_bench] " << folded << " corpus ids folded into the model's "
                          << vocab << "-token vocabulary\n";
            }
        }
        env.load   = engine.load_summary();
        env.memory = engine.memory_summary();

        prime_decode_graph(engine, env, corpus);

        std::vector<ninferno::bench::TestResult> results;
        results.reserve(tests.size());
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto& test = tests[i];
            std::cerr << "[ninferno_bench] test " << (i + 1) << '/' << tests.size() << ' '
                      << test.label << ": warmup=" << options.warmup
                      << " reps=" << options.repetitions << '\n';

            ninferno::bench::TestResult result;
            result.test = test;
            engine.reset_memory_peaks();
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                (void)run_repetition(engine, test, corpus);
            }
            result.reps.reserve(static_cast<std::size_t>(options.repetitions));
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile pre-boundary synchronize");
                require_cuda(cudaProfilerStart(), "cudaProfilerStart");
            }
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                result.reps.push_back(run_repetition(engine, test, corpus));
            }
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile post-boundary synchronize");
                require_cuda(cudaProfilerStop(), "cudaProfilerStop");
            }
            const ninferno::MemorySummary memory    = engine.memory_summary();
            result.workspace_peak_bytes           = memory.workspace_logical_peak_bytes;
            result.workspace_allocator_peak_bytes = memory.workspace.peak_used_bytes;
            results.push_back(std::move(result));
        }

        std::string report;
        switch (options.output) {
        case ninferno::bench::OutputFormat::Table:
            report = ninferno::bench::format_table(env, results);
            break;
        case ninferno::bench::OutputFormat::Json:
            report = ninferno::bench::format_json(env, command_line(argc, argv), results);
            break;
        case ninferno::bench::OutputFormat::Csv:
            report = ninferno::bench::format_csv(env, results);
            break;
        }
        write_output(options, report);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninferno_bench: " << error.what() << '\n';
        return 1;
    }
}
