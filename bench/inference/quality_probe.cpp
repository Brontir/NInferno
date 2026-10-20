// ninferno_quality_probe: token-level outputs of a Qwen3.5-family artifact for comparison against
// an independent reference (for example Hugging Face transformers on the source checkpoint).
//
//   ninferno_quality_probe <artifact> score --tokens T.bin --first-target F --out O.bin
//                        [--context N] [--kv-dtype bf16|int8|fp8|nvfp4|k8v4]
//       Scores tokens[F..] teacher-forced from an empty state. O.bin holds the FP32
//       log-probabilities; with NINFERNO_SCORE_ARGMAX=<file> set, the engine appends the argmax id
//       of every scored column to that file.
//   ninferno_quality_probe <artifact> generate --tokens T.bin --max-new N --out O.bin [--context N]
//                        [--spec mtp|dflash|dflash2 --draft-tokens K]
//       Greedy continuation of the raw prompt ids (no chat template); O.bin holds the generated
//       ids. With --spec the continuation is speculative (greedy, so it must equal the plain one)
//       and the speculation statistics are printed.
//
// Token files are little-endian int32 ids produced by the model's own tokenizer.
#include "ninferno/engine.h"
#include "product/activation_options.h"
#include "product/speculative_options.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<ninferno::TokenId> read_tokens(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<ninferno::TokenId> out;
    std::int32_t v = 0;
    while (in.read(reinterpret_cast<char*>(&v), sizeof(v))) out.push_back(v);
    return out;
}

template <class T>
void write_all(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) throw std::runtime_error("cannot write " + path);
}

ninferno::KvCacheStorage kv_storage(const std::string& name) {
    if (name == "bf16") return ninferno::KvCacheStorage::BFloat16;
    if (name == "int8") return ninferno::KvCacheStorage::Int8Group64;
    if (name == "fp8") return ninferno::KvCacheStorage::Fp8E4M3Row256;
    if (name == "nvfp4") return ninferno::KvCacheStorage::Nvfp4Group16;
    if (name == "k8v4") return ninferno::KvCacheStorage::Fp8KeyNvfp4Value;
    throw std::invalid_argument("unknown --kv-dtype " + name);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: ninferno_quality_probe <artifact> score|generate --tokens T.bin "
                         "--out O.bin [--first-target F] [--max-new N] [--context N] "
                         "[--kv-dtype D] [--activations exact|fast]\n";
            return 2;
        }
        const std::string artifact = argv[1];
        const std::string command  = argv[2];
        std::map<std::string, std::string> options;
        for (int i = 3; i + 1 < argc; i += 2) options[argv[i]] = argv[i + 1];
        const auto option = [&](const std::string& key, const std::string& fallback) {
            const auto it = options.find(key);
            return it == options.end() ? fallback : it->second;
        };
        const auto tokens = read_tokens(option("--tokens", ""));
        const auto out    = option("--out", "");
        if (tokens.size() < 2 || out.empty()) throw std::invalid_argument("--tokens and --out");

        ninferno::EngineOptions engine_options;
        engine_options.artifact_path = artifact;
        engine_options.kv_cache      = kv_storage(option("--kv-dtype", "bf16"));
        engine_options.activations   = ninferno::product::parse_activation_mode(
            option("--activations",
                     ninferno::product::activation_mode_name(ninferno::kDefaultActivationMode)));
        const auto max_new           = static_cast<std::uint32_t>(std::stoul(option("--max-new", "64")));
        const auto context           = static_cast<std::uint32_t>(std::stoul(
            option("--context", std::to_string(tokens.size() + max_new + 64))));
        engine_options.max_context   = context;
        engine_options.kv_capacity =
            ninferno::KvCapacityPolicy::explicit_capacity(((context + 63U) / 64U) * 64U);
        if (command == "score") {
            engine_options.purpose = ninferno::EnginePurpose::CausalScoring;
            ninferno::Engine engine(std::move(engine_options));
            const auto first = static_cast<std::uint32_t>(std::stoul(option("--first-target", "1")));
            const auto logprobs = engine.score_tokens(tokens, first);
            write_all(out, logprobs);
            double nll = 0;
            for (const float v : logprobs) nll -= v;
            std::cout << "{\"scored_tokens\":" << logprobs.size()
                      << ",\"mean_nll\":" << nll / static_cast<double>(logprobs.size()) << "}\n";
            return 0;
        }
        if (command == "generate") {
            if (options.count("--spec")) {
                engine_options.speculative.backend =
                    ninferno::product::parse_speculative_backend(option("--spec", ""));
                engine_options.speculative.draft_tokens =
                    static_cast<std::uint32_t>(std::stoul(option("--draft-tokens", "3")));
                ninferno::product::validate_speculative_cli_options(engine_options.speculative);
            }
            ninferno::Engine engine(std::move(engine_options));
            ninferno::PreparedPrompt prompt = engine.prepare_tokens(tokens, false);
            ninferno::RequestOptions request;
            request.execution.sampling.temperature = 0.0F;
            request.execution.requested_output_tokens = max_new;
            request.execution.allow_prefix_reuse     = false;
            const auto result = engine.generate(std::move(prompt), request);
            std::vector<std::int32_t> ids(result.generated_token_ids.begin(),
                                          result.generated_token_ids.end());
            write_all(out, ids);
            const auto& spec = result.speculative;
            std::cout << "{\"generated\":" << ids.size();
            if (spec.enabled) {
                std::cout << ",\"rounds\":" << spec.rounds << ",\"accepted_tokens\":"
                          << spec.accepted_tokens << ",\"fallback_steps\":" << spec.fallback_steps
                          << ",\"drafted_tokens\":" << spec.drafted_tokens;
            }
            std::cout << "}\n";
            return 0;
        }
        throw std::invalid_argument("unknown command " + command);
    } catch (const std::exception& error) {
        std::cerr << "ninferno_quality_probe: " << error.what() << '\n';
        return 1;
    }
}
