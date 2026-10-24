#include "core/removed_environment.h"
#include "core/cache_file.h"
#include "core/clock.h"
#include "core/command_line.h"
#include "core/sys.h"
#include "corpus.h"
#include "evaluation.h"
#include "kld.h"

#include "ninferno/engine.h"
#include "product/activation_options.h"
#include "product/engine_flags.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"
#include "product/numeric_options.h"

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using ninferno::perplexity::CorpusSelection;
using ninferno::perplexity::KldAggregate;
using ninferno::perplexity::ScoreAggregate;
using ninferno::perplexity::WindowPlan;

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::uint32_t context               = 4096;
    std::uint32_t stride                = 2048;
    float rope_yarn_factor              = 1.0F;
    int device                          = 0;
    ninferno::KvCacheStorage kv           = ninferno::KvCacheStorage::Fp8E4M3Row256;
    ninferno::ActivationMode activations  = ninferno::kDefaultActivationMode;
    ninferno::RecurrentStateStorage state = ninferno::RecurrentStateStorage::FP32;
    bool quick                          = false;
    ninferno::product::LogLevel log_level = ninferno::product::LogLevel::Info;
    // Quantization quality: record a reference (--kld-out) or measure against one (--kld).
    std::optional<std::filesystem::path> kld_out;
    std::optional<std::filesystem::path> kld;
    std::uint32_t kld_top_k = ninferno::perplexity::kKldDefaultTopK;
    // --step-tokens N: the scored targets run N tokens per call (ScoreDetailRequest::step_tokens).
    std::uint32_t step_tokens = 0;
    ninferno::DecodePacingOptions pacing;   // --cuda-sync (the decode flags do not apply: scoring)
    ninferno::MaintainerOptions maintainer; // --route-table ... --debug-poison-skip
};

std::string usage_text() {
    return "usage: ninferno-perplexity --version\n"
           "       ninferno-perplexity <model.ninferno> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--device N] [--rope-yarn-factor F]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--activations exact|fast|int8]\n"
           "       [--state-dtype f32|bf16|f16] [--step-tokens N]\n"
           "       [--output <directory>]\n"
           "       [--kld-out <file> [--kld-top-k N] | --kld <file>]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n" +
           std::string(ninferno::product::kScoringEngineFlagsUsage) +
           "  --kld-out  record this (reference) model's top-K tokens per position\n"
           "  --kld      KL divergence and top-1 agreement against such a reference, scored\n"
           "             on the same corpus, context and stride\n"
           "  --step-tokens  run the scored targets N tokens per call, as decode (1) or a\n"
           "             speculative round (N) runs them\n";
}

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

// A count as ninferno and ninferno-serve parse one ("invalid <label>: <text>").
std::uint32_t parse_count(std::string_view text, const char* label) {
    return ninferno::product::parse_u32(std::string(text).c_str(), label, /*allow_zero=*/true);
}

Options parse_options(int argc, char** argv) {
    if (ninferno::help_requested(argc, argv)) { return Options{.help_requested = true}; }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    bool kld_top_k_given = false;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context = parse_count(value("--context"), "context");
        } else if (option == "--rope-yarn-factor") {
            const std::string text(value("--rope-yarn-factor"));
            try { // as ninferno and ninferno-serve parse it
                out.rope_yarn_factor = ninferno::product::parse_float_in(
                    text.c_str(), "rope-yarn-factor", 1.0F, 64.0F);
            } catch (const std::invalid_argument& error) { usage_error(error.what()); }
        } else if (option == "--stride") {
            out.stride = parse_count(value("--stride"), "stride");
        } else if (option == "--device") {
            out.device = ninferno::product::parse_nonnegative_int(
                std::string(value("--device")).c_str(), "device");
        } else if (option == "--kv-dtype") {
            const auto storage = ninferno::parse_kv_cache_storage(value("--kv-dtype"));
            if (!storage) {
                usage_error("--kv-dtype must be " + std::string(ninferno::kKvCacheStorageNames));
            }
            out.kv = *storage;
        } else if (option == "--state-dtype") {
            const auto storage = ninferno::parse_recurrent_state_storage(value("--state-dtype"));
            if (!storage) { usage_error("--state-dtype must be f32, bf16 or f16"); }
            out.state = *storage;
        } else if (option == "--activations") {
            try {
                out.activations = ninferno::product::parse_activation_mode(value("--activations"));
            } catch (const std::invalid_argument& error) { usage_error(error.what()); }
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--log-level") {
            out.log_level = ninferno::product::parse_log_level(value("--log-level"));
        } else if (option == "--kld-out") {
            out.kld_out = std::filesystem::path(value("--kld-out"));
        } else if (option == "--kld") {
            out.kld = std::filesystem::path(value("--kld"));
        } else if (option == "--step-tokens") {
            out.step_tokens = parse_count(value("--step-tokens"), "step-tokens");
        } else if (option == "--kld-top-k") {
            out.kld_top_k   = parse_count(value("--kld-top-k"), "kld-top-k");
            kld_top_k_given = true;
        } else if (ninferno::product::is_decode_pacing_flag(option)) {
            // Scoring runs no decode rounds: the pacing would change nothing.
            usage_error(std::string(option) + " does not apply: ninferno-perplexity scores, it "
                                              "does not decode");
        } else if (ninferno::product::parse_engine_flag(
                       option, [&] { return value(argv[i]); }, out.pacing, out.maintainer)) {
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    if (out.kld_out && out.kld) { usage_error("--kld-out and --kld are exclusive"); }
    if (out.kld_top_k == 0 || out.kld_top_k > 64) { usage_error("--kld-top-k must be 1..64"); }
    if (kld_top_k_given && !out.kld_out) {
        usage_error("--kld-top-k goes with --kld-out (--kld reads K from the reference)");
    }
    return out;
}

std::string kv_name(ninferno::KvCacheStorage value) {
    switch (value) {
    case ninferno::KvCacheStorage::BFloat16:
        return "bf16";
    case ninferno::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case ninferno::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case ninferno::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninferno::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    ninferno::sys::gmtime(now, utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const ninferno::LoadSummary& load,
                                               const CorpusSelection& corpus) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_name(options.kv) /
        safe_component(corpus.corpus_id) / safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

json kld_json(const KldAggregate& value) {
    return json{{"positions", value.positions()},           {"mean_kld", value.mean()},
                {"median_kld", value.quantile(0.5)},        {"p90_kld", value.quantile(0.9)},
                {"p99_kld", value.quantile(0.99)},          {"max_kld", value.quantile(1.0)},
                {"top1_agreement", value.top1_agreement()}, {"ppl_ratio", value.ppl_ratio()}};
}

struct EvaluationStream {
    ninferno::perplexity::CorpusStream source;
    std::vector<ninferno::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        ninferno::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<ninferno::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    ninferno::EngineOptions engine_options;
    engine_options.artifact_path    = options.artifact;
    engine_options.purpose          = ninferno::EnginePurpose::CausalScoring;
    engine_options.device           = options.device;
    engine_options.max_context      = options.context;
    engine_options.kv_cache         = options.kv;
    engine_options.activations      = options.activations;
    engine_options.recurrent_state  = options.state;
    engine_options.rope_yarn_factor = options.rope_yarn_factor;
    engine_options.pacing           = options.pacing;
    engine_options.maintainer       = options.maintainer;
    engine_options.startup_observer = startup_log.observer();
    ninferno::Engine engine(std::move(engine_options));
    const ninferno::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? ninferno::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninferno::perplexity::load_custom_text(*options.text);
    // A model whose tokenizer starts a document with a token (Gemma 4's <bos>) scores every window
    // from it in place of the window's first token, as llama.cpp's perplexity and ninferno-gemma4
    // score do: a Gemma 4 window without it degrades along its length (26B at W 4096: ppl 954).
    const std::optional<ninferno::TokenId> document_start = engine.document_start_token();
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninferno::TokenId> tokens = engine.tokenize_text(source.text);
        if (const char* dump = std::getenv("NINFERNO_DUMP_TOKENS"); dump != nullptr && *dump) {
            std::ofstream out(dump, std::ios::binary);
            out.write(reinterpret_cast<const char*>(tokens.data()),
                      static_cast<std::streamsize>(tokens.size() * sizeof(tokens[0])));
        }
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            ninferno::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    // --kld: the reference must have scored these very tokens; --kld-out: a reference of them.
    std::optional<ninferno::perplexity::KldReference> reference;
    if (options.kld) {
        reference = ninferno::perplexity::read_kld_reference(*options.kld);
        if (reference->context != options.context || reference->stride != options.stride) {
            throw std::runtime_error(
                "KLD reference was scored at context/stride " + std::to_string(reference->context) +
                '/' + std::to_string(reference->stride) + ", this run at " +
                std::to_string(options.context) + '/' + std::to_string(options.stride));
        }
        const std::uint64_t vocabulary = load.metadata.vocab_size;
        if (vocabulary > 0) {
            for (const auto& r : reference->streams) {
                for (const ninferno::TokenId id : r.ids) {
                    if (id >= 0 && static_cast<std::uint64_t>(id) >= vocabulary) {
                        throw std::runtime_error("KLD reference lists a token past this model's "
                                                 "vocabulary (another tokenizer)");
                    }
                }
            }
        }
        if (reference->streams.size() != streams.size()) {
            throw std::runtime_error("KLD reference has another stream count than the corpus");
        }
        for (std::size_t s = 0; s < streams.size(); ++s) {
            const auto& r = reference->streams[s];
            if (r.id != streams[s].source.id ||
                r.token_hash != ninferno::perplexity::token_hash(streams[s].tokens) ||
                r.targets != streams[s].tokens.size() - 1) {
                throw std::runtime_error("KLD reference stream " + r.id +
                                         " holds other tokens (another tokenizer or corpus)");
            }
        }
    }
    ninferno::perplexity::KldReference recorded;
    if (options.kld_out) {
        recorded.top_k             = options.kld_top_k;
        recorded.context           = options.context;
        recorded.stride            = options.stride;
        recorded.model             = load.model_name;
        recorded.prefill_signature = load.prefill_signature;
    }
    const double preflight_seconds = ninferno::seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 ninferno::product::format_pretty_count(streams.size()),
                 ninferno::product::format_pretty_count(total_input_tokens),
                 ninferno::product::format_pretty_count(total_scored_tokens),
                 ninferno::product::format_pretty_count(total_windows),
                 ninferno::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory = prepare_output_directory(options, load, corpus);
    const Clock::time_point scoring_started      = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 ninferno::product::format_pretty_count(streams.size()),
                 ninferno::product::format_pretty_count(total_scored_tokens),
                 ninferno::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    KldAggregate kld_overall;
    std::map<std::string, KldAggregate> kld_domains;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << ninferno::product::format_pretty_text(stream.source.id) << " | "
                      << ninferno::product::format_pretty_count(stream.tokens.size())
                      << " tokens | "
                      << ninferno::product::format_pretty_count(stream.windows.size())
                      << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        KldAggregate stream_kld;
        ninferno::perplexity::KldStream recorded_stream;
        if (options.kld_out) {
            recorded_stream.id         = stream.source.id;
            recorded_stream.token_hash = ninferno::perplexity::token_hash(stream.tokens);
            recorded_stream.targets    = stream.tokens.size() - 1;
        }
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninferno::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            if (document_start) { input.front() = *document_start; }
            const Clock::time_point window_started = Clock::now();
            const std::size_t expected             = window.target_end - window.target_begin;
            // Target token i is the stream's target i - 1.
            const std::size_t first = window.target_begin - 1;
            ninferno::ScoreDetailRequest request;
            request.step_tokens = options.step_tokens;
            if (options.kld_out) { request.top_k = options.kld_top_k; }
            if (reference) {
                const std::size_t k = reference->top_k;
                const auto& ids     = reference->streams[stream_index].ids;
                request.top_k       = 1; // this model's most probable token
                request.given_k     = reference->top_k;
                request.given_ids.assign(ids.begin() + static_cast<std::ptrdiff_t>(first * k),
                                         ids.begin() +
                                             static_cast<std::ptrdiff_t>((first + expected) * k));
            }
            ninferno::ScoreDetail detail;
            std::vector<float> logprobs;
            try {
                logprobs =
                    engine.score_tokens(std::move(input), window.first_target, request, detail);
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            if (options.kld_out) {
                auto& r = recorded_stream;
                r.ids.insert(r.ids.end(), detail.top_ids.begin(), detail.top_ids.end());
                r.logprobs.insert(r.logprobs.end(), detail.top_logprobs.begin(),
                                  detail.top_logprobs.end());
                r.target_logprobs.insert(r.target_logprobs.end(), logprobs.begin(), logprobs.end());
            }
            if (reference) {
                const std::size_t k = reference->top_k;
                const auto& r       = reference->streams[stream_index];
                KldAggregate window_kld;
                for (std::size_t j = 0; j < expected; ++j) {
                    const std::size_t t = first + j;
                    const std::span<const float> p(r.logprobs.data() + t * k, k);
                    const std::span<const float> q(detail.given_logprobs.data() + j * k, k);
                    window_kld.add(ninferno::perplexity::position_kld(p, q),
                                   detail.top_ids[j] == r.ids[t * k], r.target_logprobs[t],
                                   logprobs[j]);
                }
                stream_kld.add(window_kld);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = window.first_target;
            window_report["seconds"]      = ninferno::seconds_since(window_started);
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed          = ninferno::seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | "
                     << ninferno::product::format_pretty_count(overall.scored_tokens) << '/'
                     << ninferno::product::format_pretty_count(total_scored_tokens) << " tokens | "
                     << completed_windows << '/' << total_windows << " windows | PPL " << std::fixed
                     << std::setprecision(4) << overall.ppl() << " | "
                     << ninferno::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << ninferno::product::format_pretty_duration(elapsed) << " | ETA "
                     << ninferno::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = ninferno::seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), ninferno::product::format_pretty_text(stream.source.id),
                     ninferno::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), ninferno::product::format_pretty_duration(stream_seconds));
        json stream_report               = aggregate_json(stream_score);
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["input_tokens"]    = stream.tokens.size();
        stream_report["unscored_tokens"] = 1;
        stream_report["seconds"]         = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        if (reference) {
            stream_report["kld"] = kld_json(stream_kld);
            kld_domains[stream.source.domain].add(stream_kld);
            kld_overall.add(stream_kld);
        }
        if (options.kld_out) { recorded.streams.push_back(std::move(recorded_stream)); }
        stream_reports.push_back(std::move(stream_report));
    }
    if (options.kld_out) { ninferno::perplexity::write_kld_reference(*options.kld_out, recorded); }

    const double scoring_seconds = ninferno::seconds_since(scoring_started);
    progress->clear();
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 ninferno::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), ninferno::product::format_pretty_duration(scoring_seconds),
                 ninferno::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }

    json report{
        {"schema_version", 2},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"device", options.device},
          {"context_tokens", options.context},
          {"stride_tokens", options.stride},
          {"prefill_chunk_tokens", 1024},
          {"step_tokens", options.step_tokens},
          {"score_tile_tokens", 1024},
          {"window_start_token", document_start ? json(*document_start) : json(nullptr)},
          {"kv_dtype", kv_name(options.kv)},
          {"activations", ninferno::product::activation_mode_name(options.activations)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", ninferno::seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", aggregate_json(overall)},
    };
    if (options.kld_out) {
        report["kld_reference"] = {
            {"path", std::filesystem::absolute(*options.kld_out).lexically_normal().string()},
            {"top_k", options.kld_top_k}};
    }
    if (reference) {
        json kld_domain_reports = json::array();
        for (const auto& [domain, aggregate] : kld_domains) {
            json item      = kld_json(aggregate);
            item["domain"] = domain;
            kld_domain_reports.push_back(std::move(item));
        }
        report["kld"] = {
            {"reference",
             {{"path", std::filesystem::absolute(*options.kld).lexically_normal().string()},
              {"model", reference->model},
              {"prefill_signature", reference->prefill_signature},
              {"top_k", reference->top_k}}},
            {"metric", "KL(P_ref || Q) over the reference's top-K tokens and the unlisted rest, "
                       "natural log, per scored position"},
            {"domains", std::move(kld_domain_reports)},
            {"overall", kld_json(kld_overall)}};
    }

    const std::filesystem::path final = output_directory / "report.json";
    ninferno::write_file_atomic(final, report.dump(2) + '\n');

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode << ", context/stride: " << options.context << '/' << options.stride
              << "\n\n";
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n"
              << "score rate: " << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n";
    if (reference) {
        std::cout << "\nagainst " << reference->model << " (top-" << reference->top_k << ")\n"
                  << std::setprecision(6) << "mean KLD: " << kld_overall.mean()
                  << ", median: " << kld_overall.quantile(0.5)
                  << ", p99: " << kld_overall.quantile(0.99) << '\n'
                  << std::setprecision(4)
                  << "top-1 agreement: " << 100.0 * kld_overall.top1_agreement()
                  << " %, PPL ratio: " << kld_overall.ppl_ratio() << '\n';
    }
    if (options.kld_out) { std::cout << "KLD reference: " << *options.kld_out << '\n'; }
    std::cout << "report: " << final << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (ninferno::version_requested(argc, argv)) {
        std::cout << "NInferno " << NINFERNO_VERSION_STRING << '\n';
        return 0;
    }
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninferno-perplexity: " << error.what() << '\n';
        std::cerr << usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }
    // A removed NINFERNO_* decision variable changes nothing: say so once.
    ninferno::warn_removed_environment("ninferno-perplexity");

    ninferno::product::LoggingRuntime logging(
        {.logger_name  = "ninferno-perplexity",
         .level        = options.log_level,
         .presentation = ninferno::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninferno::product::StartupLogRenderer startup_log(logging);
    try {
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", ninferno::product::format_pretty_text(error.what()));
        return 1;
    }
}
