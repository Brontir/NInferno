// ninferno-dense: run a dense-decoder artifact (Llama, Mistral/Ministral, Granite, OLMo, Seed-OSS,
// Qwen2/3 dense, ...) directly through its model family.
//
//   ninferno-dense <artifact> bench    --tokens F --prompt N --decode M [--repeat R]
//   ninferno-dense <artifact> score    --tokens F [--window W] [--dump O]
//   ninferno-dense <artifact> generate --tokens F --decode M [--out O] [--cuts P1,P2,...]
//   ninferno-dense <artifact> logits   --tokens F --out O
//   ninferno-dense <artifact> chat     --prompt TEXT [--system TEXT] [--decode M]
//                                    [--temperature T --top-p P --top-k K --seed S]
//   ninferno-dense <artifact> spec     --tokens F --decode M [--prompt N] [--repeat R] [--out O]
//
// Speculative decoding (greedy): --dflash D (a DFlash drafter artifact paired with this target)
// or --ngram (prompt lookup), --draft-tokens K (drafts verified per round), --min-draft-tokens J
// (the depth adapts within [J, K]). chat then generates by speculative rounds. spec prefills the
// prompt (the whole token file, or its first N tokens) and decodes M tokens twice from the same
// state, plainly and by speculative rounds, and reports both speeds, the rounds' acceptance and
// how far the two greedy continuations agree (--out O writes the speculative one).
//
// chat renders the artifact's own chat template over the messages, tokenizes with its own
// tokenizer, generates until an end-of-generation id (generation_config.json) or M tokens, and
// prints the decoded answer on stderr (the JSON summary on stdout, as every command).
// generate --cuts prefills the prompt in calls that end at the given positions, as ninferno-serve
// cuts a chat prompt at its execution boundaries (where the assistant turn begins, so that a
// cached prefix and a cold prompt compute alike). A call's result depends on its length (a short
// call takes another GEMM route), so a greedy parity check against the server uses its cuts.
// Token files are little-endian int32 ids produced by the model's own tokenizer (special tokens
// included by the producer). Common options: --context L, --chunk C, --a16 (every activation in
// BF16 whatever the artifact recorded), --activations exact (ninferno-serve's exact: --a16 and
// every op on its invariant route profile, ops/prefill_routes.h; tokens equal the Engine's byte for
// byte), --eager-decode (no decode graph; each step under the
// attention envelope of its own frontier, as ninferno-serve --no-cuda-graph decodes), --kv-dtype
// bf16|fp8 (keys and values in BF16, the default, or FP8 E4M3 with a scale per row: half the KV
// bytes). Every result is one JSON object on stdout.
//
// <artifact> may also be a Hugging Face checkpoint directory: it is converted once, in-process,
// with the official recipe for its architecture into <dir>/.ninferno-cache/<recipe>.ninferno, and
// that artifact is loaded; later runs load it directly.

#include "convert/cache.h"
#include "core/clock.h"
#include "core/device.h"
#include "models/dense/frontend.h"
#include "models/dense/session.h"
#include "ninferno/ops/prefill_routes.h"
#include "product/activation_options.h"
#include "models/speculation/dflash.h"
#include "product/numeric_options.h"
#include "product/tool_args.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::models::dense;

using ninferno::product::join_ids;
using ninferno::product::option_list;
using ninferno::product::read_tokens;

namespace {

using Arguments = product::ToolArguments;

Arguments parse(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument("usage: ninferno-dense <artifact> "
                                    "<bench|score|generate|chat|logits|spec> [--key value ...]");
    }
    Arguments out{argv[1], argv[2], {}};
    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) { throw std::invalid_argument("unexpected " + key); }
        key = key.substr(2);
        if (key == "a16" || key == "eager-decode" || key == "ngram") {
            out.options[key] = "1";
        } else {
            if (i + 1 >= argc) { throw std::invalid_argument("--" + key + " needs a value"); }
            out.options[key] = argv[++i];
        }
    }
    return out;
}

template <class T>
void write_all(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) { throw std::runtime_error("cannot write " + path); }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args   = parse(argc, argv);
        const auto cached = convert::cached_artifact(args.artifact, {}, {}, false);
        if (cached.converted) {
            std::fprintf(stderr, "ninferno-dense: converted %s with %s to %s in %.1f s\n",
                         args.artifact.c_str(), cached.recipe.c_str(), cached.path.c_str(),
                         cached.seconds);
        }
        DeviceContext device(0);
        const auto load_start = std::chrono::steady_clock::now();
        const bool exact = product::parse_activation_mode(args.text("activations", "fast")) ==
                           ActivationMode::Exact;
        // For the rest of the run: this thread's ops resolve their routes under it.
        const ops::ScopedInvariantRoutes invariant_routes(exact);
        const auto profile    = exact || args.flag("a16") ? ActivationProfile::A16
                                                          : ActivationProfile::Artifact;
        auto model            = Model::load(cached.path, device, profile);
        const double load_s   = ninferno::seconds_since(load_start);
        SessionOptions options;
        options.max_context   = static_cast<std::int32_t>(args.number("context", 32768));
        options.prefill_chunk = static_cast<std::int32_t>(args.number("chunk", 2048));
        options.eager_decode  = args.flag("eager-decode");
        const std::string kv_dtype = args.text("kv-dtype", "bf16");
        if (kv_dtype != "bf16" && kv_dtype != "fp8") {
            throw std::invalid_argument("--kv-dtype must be bf16 or fp8");
        }
        options.kv_fp8 = kv_dtype == "fp8";
        std::unique_ptr<models::speculation::DFlashDrafter> dflash;
        if (args.flag("dflash")) {
            dflash = models::speculation::DFlashDrafter::load(args.text("dflash"), device,
                                                              model->config());
            options.dflash = dflash.get();
        }
        options.ngram            = args.flag("ngram");
        options.draft_tokens     = static_cast<std::int32_t>(
            args.number("draft-tokens", dflash ? dflash->config().block_size - 1 : 4));
        options.min_draft_tokens = static_cast<std::int32_t>(args.number("min-draft-tokens", 0));
        const bool speculative   = options.dflash != nullptr || options.ngram;
        std::optional<Frontend> frontend;
        std::vector<std::int32_t> tokens;
        if (args.command == "chat") {
            frontend.emplace(Frontend::load(cached.path));
            nlohmann::ordered_json messages = nlohmann::ordered_json::array();
            if (args.flag("system")) {
                messages.push_back({{"role", "system"}, {"content", args.text("system")}});
            }
            messages.push_back({{"role", "user"}, {"content", args.text("prompt")}});
            tokens = frontend->render_chat(messages);
            options.temperature = std::stof(args.text("temperature", "0"));
            options.top_p       = std::stof(args.text("top-p", "1"));
            options.top_k       = static_cast<std::int32_t>(args.number("top-k", 20));
            options.seed        = static_cast<unsigned long long>(args.number("seed", 0));
        } else {
            tokens = read_tokens(args.text("tokens"));
        }
        Session session(*model, device, options);
        if (speculative) { session.warm_up(); }
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        std::printf("{\"load_seconds\":%.2f,\"weights_gib\":%.3f,\"session_gib\":%.3f,"
                    "\"kv_gib\":%.3f,\"kv_dtype\":\"%s\","
                    "\"device_used_gib\":%.3f,\"profile\":\"%s\",\"chunk\":%d,\"context\":%d,",
                    load_s, model->device_bytes() / 1073741824.0,
                    session.device_bytes() / 1073741824.0, session.kv_bytes() / 1073741824.0,
                    kv_dtype.c_str(), (total_bytes - free_bytes) / 1073741824.0,
                    exact ? "exact" : args.flag("a16") ? "a16" : "artifact", options.prefill_chunk,
                    options.max_context);
        if (speculative) {
            std::printf("\"drafter\":\"%s\",\"drafter_gib\":%.3f,\"draft_tokens\":%d,"
                        "\"min_draft_tokens\":%d,",
                        dflash ? "dflash" : "ngram",
                        dflash ? dflash->device_bytes() / 1073741824.0 : 0.0,
                        options.draft_tokens, options.min_draft_tokens);
        }

        if (args.command == "bench") {
            const auto prompt = static_cast<std::size_t>(args.number("prompt", 512));
            const auto decode = static_cast<std::int32_t>(args.number("decode", 128));
            const auto repeat = static_cast<int>(args.number("repeat", 3));
            if (tokens.size() < prompt) { throw std::runtime_error("token file shorter than prompt"); }
            const std::span<const std::int32_t> prompt_tokens(tokens.data(), prompt);
            std::vector<double> prefill_s, decode_s;
            for (int r = 0; r < repeat + 1; ++r) {
                session.reset();
                auto start = std::chrono::steady_clock::now();
                session.prefill(prompt_tokens);
                const double p = ninferno::seconds_since(start);
                // The first decode call captures the graph; time a separate window after it.
                (void)session.decode(4);
                start = std::chrono::steady_clock::now();
                (void)session.decode(decode);
                const double d = ninferno::seconds_since(start);
                if (r == 0) { continue; } // the first round warms every route
                prefill_s.push_back(p);
                decode_s.push_back(d);
            }
            std::sort(prefill_s.begin(), prefill_s.end());
            std::sort(decode_s.begin(), decode_s.end());
            const double p_med = prefill_s[prefill_s.size() / 2];
            const double d_med = decode_s[decode_s.size() / 2];
            std::printf("\"prompt_tokens\":%zu,\"decode_tokens\":%d,\"repeats\":%d,"
                        "\"prefill_seconds_median\":%.4f,\"prefill_tok_s\":%.1f,"
                        "\"prefill_tok_s_best\":%.1f,\"decode_seconds_median\":%.4f,"
                        "\"decode_tok_s\":%.2f,\"decode_tok_s_best\":%.2f}\n",
                        prompt, decode, repeat, p_med, prompt / p_med, prompt / prefill_s.front(),
                        d_med, decode / d_med, decode / decode_s.front());
        } else if (args.command == "score") {
            // Non-overlapping windows of W tokens, each scored from an empty cache.
            const auto window = static_cast<std::size_t>(args.number("window", 2048));
            double nll        = 0;
            std::size_t count = 0;
            std::vector<float> all_logprobs;
            std::vector<std::int32_t> all_predictions;
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t begin = 0; begin + 1 < tokens.size(); begin += window) {
                const auto n = std::min(window, tokens.size() - begin);
                if (n < 2) { break; }
                session.reset();
                const auto lp = session.score({tokens.data() + begin, n}, &all_predictions);
                for (const float v : lp) {
                    nll -= v;
                    ++count;
                }
                all_logprobs.insert(all_logprobs.end(), lp.begin(), lp.end());
            }
            if (args.flag("dump")) {
                // Two raw planes, in scoring order: FP32 log-probabilities, then I32 argmax ids.
                std::ofstream dump(args.text("dump"), std::ios::binary);
                dump.write(reinterpret_cast<const char*>(all_logprobs.data()),
                           static_cast<std::streamsize>(all_logprobs.size() * sizeof(float)));
                dump.write(reinterpret_cast<const char*>(all_predictions.data()),
                           static_cast<std::streamsize>(all_predictions.size() * 4));
            }
            std::printf("\"scored_tokens\":%zu,\"window\":%zu,\"mean_nll\":%.6f,"
                        "\"perplexity\":%.4f,\"seconds\":%.2f}\n",
                        count, window, nll / count, std::exp(nll / count),
                        ninferno::seconds_since(start));
        } else if (args.command == "generate") {
            const auto decode = static_cast<std::int32_t>(args.number("decode", 64));
            std::size_t begin = 0;
            for (const auto& cut : option_list(args.text("cuts"), ',')) {
                const auto end = static_cast<std::size_t>(std::stoul(cut));
                if (end <= begin || end >= tokens.size()) {
                    throw std::invalid_argument(
                        "--cuts must be increasing positions inside the prompt");
                }
                session.prefill(std::span<const std::int32_t>(tokens).subspan(begin, end - begin));
                begin = end;
            }
            session.prefill(std::span<const std::int32_t>(tokens).subspan(begin));
            const auto out = session.decode(decode);
            if (args.flag("out")) { write_all(args.text("out"), out); }
            std::printf("\"prompt_tokens\":%zu,\"generated\":[%s]}\n", tokens.size(),
                        join_ids(out).c_str());
        } else if (args.command == "chat") {
            const auto max_new = static_cast<std::int32_t>(args.number("decode", 512));
            const auto& stops  = frontend->stop_ids();
            const auto start   = std::chrono::steady_clock::now();
            session.prefill(tokens);
            const double prefill = ninferno::seconds_since(start);
            std::vector<std::int32_t> out;
            bool stopped = false;
            if (speculative) {
                // Rounds until an end-of-generation id or M tokens; the id itself is not part of
                // the answer.
                for (const auto id : session.speculate_until(max_new + 1, stops)) {
                    if (std::find(stops.begin(), stops.end(), id) != stops.end()) {
                        stopped = true;
                        break;
                    }
                    if (static_cast<std::int32_t>(out.size()) >= max_new) { break; }
                    out.push_back(id);
                }
            }
            // Blocks of 16 steps; the answer ends at the first end-of-generation id.
            while (!speculative && !stopped && static_cast<std::int32_t>(out.size()) < max_new &&
                   session.position() + 16 < options.max_context) {
                for (const auto id : session.decode(16)) {
                    if (std::find(stops.begin(), stops.end(), id) != stops.end()) {
                        stopped = true;
                        break;
                    }
                    out.push_back(id);
                    if (static_cast<std::int32_t>(out.size()) >= max_new) { break; }
                }
            }
            const double total = ninferno::seconds_since(start);
            // The answer split by the template's reasoning markup (conventions data), read from the
            // output with its special tokens, which carry that markup.
            nlohmann::json split;
            if (const auto& reasoning = frontend->conventions().reasoning) {
                const auto parts =
                    ninferno::chat::split_reasoning(frontend->decode(out, false), *reasoning);
                split["reasoning_markup"] = reasoning->name;
                split["reasoning_closed"] = parts.closed;
                split["reasoning"]        = parts.opened ? nlohmann::json(parts.reasoning) : nlohmann::json();
                split["content"]          = parts.content;
            }
            if (speculative) {
                const auto stats = session.speculation_stats();
                std::printf("\"rounds\":%lld,\"accepted\":%lld,\"tokens_per_round\":%.3f,",
                            static_cast<long long>(stats.rounds),
                            static_cast<long long>(stats.accepted), stats.tokens_per_round());
            }
            std::printf("\"prompt_tokens\":%zu,\"generated_tokens\":%zu,\"stopped\":%s,"
                        "\"prefill_seconds\":%.4f,\"decode_tok_s\":%.2f%s%s}\n",
                        tokens.size(), out.size(), stopped ? "true" : "false", prefill,
                        out.size() / std::max(total - prefill, 1e-9), split.empty() ? "" : ",",
                        split.empty() ? "" : split.dump().substr(1, split.dump().size() - 2).c_str());
            std::fflush(stdout);
            std::cerr << frontend->decode(out, true) << '\n';
        } else if (args.command == "spec") {
            if (!speculative) { throw std::invalid_argument("spec needs --dflash or --ngram"); }
            const auto decode = static_cast<std::int32_t>(args.number("decode", 256));
            const auto prompt =
                static_cast<std::size_t>(args.number("prompt", static_cast<std::int64_t>(tokens.size())));
            const auto repeat = static_cast<int>(args.number("repeat", 3));
            if (tokens.size() < prompt) { throw std::runtime_error("token file shorter than prompt"); }
            const std::span<const std::int32_t> prompt_tokens(tokens.data(), prompt);
            std::vector<double> plain_s, spec_s;
            std::vector<std::int32_t> plain_out, spec_out;
            models::speculation::SpeculationStats stats{};
            for (int r = 0; r < repeat + 1; ++r) {
                session.reset();
                session.prefill(prompt_tokens);
                auto start = std::chrono::steady_clock::now();
                plain_out  = session.decode(decode);
                const double p = ninferno::seconds_since(start);
                session.reset();
                session.prefill(prompt_tokens);
                const auto before = session.speculation_stats();
                start             = std::chrono::steady_clock::now();
                spec_out          = session.speculate_until(decode);
                const double s    = ninferno::seconds_since(start);
                const auto after  = session.speculation_stats();
                if (r == 0) { continue; } // the first round warms every route
                plain_s.push_back(p);
                spec_s.push_back(s);
                stats.rounds += after.rounds - before.rounds;
                stats.drafted += after.drafted - before.drafted;
                stats.accepted += after.accepted - before.accepted;
            }
            std::sort(plain_s.begin(), plain_s.end());
            std::sort(spec_s.begin(), spec_s.end());
            std::size_t agree = 0;
            while (agree < plain_out.size() && agree < spec_out.size() &&
                   plain_out[agree] == spec_out[agree]) {
                ++agree;
            }
            if (args.flag("out")) { write_all(args.text("out"), spec_out); }
            const double p_med = plain_s[plain_s.size() / 2];
            const double s_med = spec_s[spec_s.size() / 2];
            std::printf("\"prompt_tokens\":%zu,\"decode_tokens\":%d,\"repeats\":%d,"
                        "\"plain_tok_s\":%.2f,\"spec_tok_s\":%.2f,\"speedup\":%.3f,"
                        "\"rounds\":%lld,\"drafted\":%lld,\"accepted\":%lld,"
                        "\"acceptance\":%.4f,\"tokens_per_round\":%.3f,"
                        "\"agree_prefix\":%zu,\"spec_tokens\":%zu}\n",
                        prompt, decode, repeat, decode / p_med, spec_out.size() / s_med,
                        (spec_out.size() / s_med) / (decode / p_med),
                        static_cast<long long>(stats.rounds), static_cast<long long>(stats.drafted),
                        static_cast<long long>(stats.accepted),
                        stats.drafted ? double(stats.accepted) / double(stats.drafted) : 0.0,
                        stats.tokens_per_round(), agree, spec_out.size());
        } else if (args.command == "logits") {
            session.prefill(tokens);
            write_all(args.text("out"), session.last_logits());
            std::printf("\"prompt_tokens\":%zu}\n", tokens.size());
        } else {
            throw std::invalid_argument("unknown command " + args.command);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fflush(stdout);
        std::cerr << "ninferno-dense: " << error.what() << '\n';
        return 1;
    }
}
