// ninferno-gemma4: run a Gemma 4 artifact directly through its model family.
//
//   ninferno-gemma4 <artifact> bench    --tokens F --prompt N --decode M [--repeat R]
//   ninferno-gemma4 <artifact> score    --tokens F [--window W]
//   ninferno-gemma4 <artifact> generate --tokens F --decode M [--cuts P1,P2,...]
//   ninferno-gemma4 <artifact> logits   --tokens F --out O
//   ninferno-gemma4 <artifact> batch    --tokens F --decode M [sampling and --stop options]
//   ninferno-gemma4 <artifact> spec-bench --tokens F --prompt N --decode M
//                 (--drafter A | --dflash A) [--draft K] [--draft-min K0] [--dump-tokens O]
//                 [--stop IDS]  (both streams end at the plain stream's first stop id)
//   ninferno-gemma4 <artifact> spec-audit --list L --out O [--drafter A] [--dflash B]
//                 [--configs mtp:K[:K0],dflash:K,...] [--decode M] [--repeat R] [--tag T]
//   ninferno-gemma4 <artifact> spec-tune [--drafter A] [--dflash B | --drafter-variant V]
//                 [--configs LIST]
//                 [--budget-seconds S] [--repeat R] [--out PROFILE] [--corpus FILE]
//                 [--publish FILE --applies-to REPO,... --measured-on TEXT --gpu-class C]
//
// generate and batch speculate by default when the model came with speculation (spec/
// manifest.h): a maintainer's tune named by --spec-profile for this artifact and GPU, else the
// profile the model's <artifact>.speculation.json publishes for this GPU (its chip variant, chip,
// architecture, then "*"; spec-tune publishes for the chip unless --gpu-class says), else prompt
// lookup when that manifest says no drafter came with the model; an artifact without either decodes
// plainly, as before. --spec off|auto|<config> (e.g. dflash:7, mtp:8:2, ngram:4) overrides.
// With --drafter / --dflash they speculate with that drafter: --spec names the configuration,
// else --draft (and --draft-min) set it; --spec off decodes plainly.
// The profile's choice is its default for the prompt's thinking marker. batch picks per prompt:
// its prompts are grouped by the configuration their thinking flags choose, each group runs in its
// own session (one at a time), and the results are printed in the prompts' order, each with the
// configuration it ran. --acceptance-floor F / --no-watchdog set
// the runtime watchdog (default: the chosen configuration's break-even tokens per round). generate
// and batch capture every CUDA graph they can replay at load (Session::warm_up), so the first
// request decodes at the speed of the later ones; --no-warm-up leaves the capture to the first
// request (the old behaviour, for A/B measurements). spec-tune is a maintainer tool: it measures
// every configuration the drafters allow (drafter flags, else the manifest's drafters: as
// published, or with --drafter-variant V their variants "<name>.V", the q8/q4/nvfp4 builds
// ninferno-convert attaches beside them) plus prompt lookup on the built-in corpus and writes the
// profile under the drafters' manifest names. --drafter-variant V also picks the manifest's drafter
// for an explicit --spec.
//
// generate --cuts prefills the prompt in calls that end at the given positions, as ninferno-serve
// cuts a chat prompt at its execution boundaries (where the assistant turn begins, so that a
// cached prefix and a cold prompt compute alike). A call's result depends on its length (a short
// call takes another GEMM route), so a greedy parity check against the server uses its cuts.
//
// --draft defaults to 4 (MTP) and block_size - 1 (DFlash); the measured best drafter and depth per
// model and GPU are the profiles of src/spec/data/drafter_registry.json (what --spec auto runs).
//
// Token files are little-endian int32 ids, produced by the model's own tokenizer outside this
// program (BOS included by the producer). Common options: --context L, --chunk C (without them
// the session is sized to the free device memory: the largest context up to 32768, spec-tune 4096,
// with the widest chunk up to 2048 that keeps it; see fit_session), --kv-dtype bf16|fp8|fp8-full
// (keys and values in BF16, the default, or FP8 E4M3 with a scale per row: about half the KV bytes;
// fp8-full on the full layers alone, as ninferno-serve keeps Gemma 4), and the
// activation profile: none (the artifact's recorded W4A4 policy), --a16 (dense projections in
// BF16; on 26B-A4B the expert bank's prefill stays W4A4, so that profile is mixed) or --a16-full
// (every activation in BF16, experts included). --activations exact is ninferno-serve's exact: the
// --a16-full weights, and every op on its invariant route profile (ops/prefill_routes.h), the
// prefill profile on every prefill call; its tokens equal the Engine's byte for byte. Every result
// is one JSON object on stdout.
//
// <artifact>, --drafter and --dflash also accept a Hugging Face checkpoint directory. It is
// converted once, in-process, with the official recipe for its architecture, into
// <dir>/.ninferno-cache/<recipe>.ninferno, and that artifact is loaded; later runs load it
// directly. A drafter checkpoint without tokenizer files takes them from a checkpoint-directory
// target.
// --consume-source lets that conversion release the checkpoint's space as it goes (the
// checkpoint's weights are destroyed; the artifact then lives on in .ninferno-cache).

#include "convert/cache.h"
#include "core/clock.h"
#include "core/device.h"
#include "models/gemma4/gemma4.h"
#include "ninferno/ops/prefill_routes.h"
#include "product/activation_options.h"
#include "spec/config.h"
#include "spec/corpus.h"
#include "spec/engine_support.h"
#include "spec/fingerprint.h"
#include "spec/gpu_identity.h"
#include "spec/manifest.h"
#include "spec/profile.h"
#include "spec/registry.h"
#include "product/numeric_options.h"
#include "product/tool_args.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninferno;
using namespace ninferno::models::gemma4;

using ninferno::product::join_ids;
using ninferno::product::option_list;
using ninferno::product::read_tokens;

namespace {

using Arguments = product::ToolArguments;

Arguments parse(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument("usage: ninferno-gemma4 <artifact> "
                                    "<bench|score|generate|logits|batch|spec-tune|spec-audit|spec-"
                                    "bench> [--key value ...]");
    }
    Arguments out{argv[1], argv[2], {}};
    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) { throw std::invalid_argument("unexpected " + key); }
        key = key.substr(2);
        // Every option some command reads; anything else is refused, not ignored (an unread
        // --activations exact once ran the default profile unnoticed).
        static const std::set<std::string> known = {
            "a16",         "a16-full",    "acceptance-floor",
            "applies-to",  "block",       "budget-seconds",
            "chunk",       "configs",     "consume-source",
            "context",     "corpus",      "cuts",
            "decode",      "dflash",      "draft",
            "draft-min",   "drafter",     "drafter-variant",
            "dump",        "dump-tokens", "gpu-class",
            "kv-dtype",    "list",        "measured-on",
            "no-warm-up",  "no-watchdog", "out",
            "prefill-split",
            "profile",     "prompt",      "publish",
            "repeat",      "rounds-out",  "seed",
            "spec",        "spec-profile", "stop",       "tag",
            "temperature", "tokens",      "top-k",
            "top-p",       "window"};
        if (!known.contains(key)) {
            throw std::invalid_argument("unknown option --" + key +
                                        " (activation profiles: --a16, --a16-full)");
        }
        if (key == "a16" || key == "a16-full" || key == "consume-source" || key == "no-watchdog" ||
            key == "no-warm-up") {
            out.options[key] = "1";
        } else {
            if (i + 1 >= argc) { throw std::invalid_argument("--" + key + " needs a value"); }
            out.options[key] = argv[++i];
        }
    }
    return out;
}

std::size_t free_device_bytes() {
    std::size_t free_bytes = 0, total_bytes = 0;
    return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess ? free_bytes : 0;
}

// An artifact path, or the cached conversion of a checkpoint directory (converting it first).
std::string
artifact_path(const std::string& path, bool consume_source,
              const std::vector<std::pair<std::string, std::filesystem::path>>& resources = {}) {
    const auto cached = convert::cached_artifact(path, resources, {}, consume_source);
    if (cached.converted) {
        std::fprintf(stderr, "ninferno-gemma4: converted %s with %s to %s in %.1f s\n",
                     path.c_str(), cached.recipe.c_str(), cached.path.c_str(), cached.seconds);
    }
    return cached.path.string();
}

// Frontend files a drafter checkpoint may lack, taken from a checkpoint-directory target.
std::vector<std::pair<std::string, std::filesystem::path>>
target_resources(const std::string& target) {
    std::vector<std::pair<std::string, std::filesystem::path>> out;
    if (!convert::is_checkpoint_directory(target)) { return out; }
    for (const char* role : {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                             "generation_config.json"}) {
        out.emplace_back(role, std::filesystem::path(target) / role);
    }
    return out;
}

// --stop ids (comma-separated; default 1,106,50: Gemma 4's end of text, end of turn and <turn|>).
std::vector<std::int32_t> stop_ids(const Arguments& args) {
    std::vector<std::int32_t> out;
    for (const auto& part : option_list(args.text("stop", "1,106,50"), ',')) {
        out.push_back(std::stoi(part));
    }
    return out;
}

// The two largest logits of one column and the first index of each.
struct TopTwo {
    float first = -INFINITY, second = -INFINITY;
    std::int32_t first_id = -1, second_id = -1;
};

TopTwo top_two(const std::vector<float>& logits) {
    TopTwo out;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float v = logits[i];
        if (v > out.first) {
            out.second    = out.first;
            out.second_id = out.first_id;
            out.first     = v;
            out.first_id  = static_cast<std::int32_t>(i);
        } else if (v > out.second) {
            out.second    = v;
            out.second_id = static_cast<std::int32_t>(i);
        }
    }
    return out;
}

// A soft-capped logit back on the model's raw logit scale (identity without a cap).
double uncap(float value, std::optional<float> cap) {
    if (!cap) { return value; }
    const double x = std::clamp(double(value) / *cap, -0.999999, 0.999999);
    return *cap * std::atanh(x);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values.empty() ? 0.0 : values[values.size() / 2];
}

// spec-audit: plain greedy decode against every speculative configuration over a prompt list, in
// one process. Per prompt, plain decode runs step by step once (its top-2 gap at every position)
// and timed; the answer ends at the first stop id or --decode tokens. Per configuration, the
// speculative continuation of the same length is timed, then traced round by round (drafts,
// target argmax, phase device times), and wherever it leaves plain decode it is replayed
// teacher-forced through plain decode: every emitted token plain decode would not have chosen on
// the same prefix is reported with its deficit and the top-2 gap there, on the raw logit scale.
int spec_audit(const Arguments& args, const Model& model, DeviceContext& device,
               SessionOptions base, const Drafter* drafter, const DFlashDrafter* dflash) {
    struct Prompt {
        std::string name;
        std::vector<std::int32_t> tokens;
    };
    std::vector<Prompt> prompts;
    {
        std::ifstream list(args.text("list"));
        if (!list) { throw std::runtime_error("spec-audit needs --list NAME PATH lines"); }
        std::string name, path;
        while (list >> name >> path) { prompts.push_back({name, read_tokens(path)}); }
    }
    struct Config {
        std::string label;
        bool dflash;
        std::int32_t k, k_min;
    };
    std::vector<Config> configs;
    for (const auto& item : option_list(args.text("configs", "mtp:4"), ',')) {
        const auto parts = option_list(item, ':');
        Config c{item, parts.at(0) == "dflash", std::stoi(parts.at(1)),
                 parts.size() > 2 ? std::stoi(parts[2]) : 0};
        if ((c.dflash && dflash == nullptr) || (!c.dflash && drafter == nullptr)) {
            throw std::invalid_argument("spec-audit: " + item + " needs its drafter");
        }
        configs.push_back(c);
    }
    std::vector<std::int32_t> stops;
    for (const auto& s : option_list(args.text("stop", "1,106,50"), ',')) {
        stops.push_back(std::stoi(s));
    }
    const auto want    = static_cast<std::int32_t>(args.number("decode", 256));
    const auto repeat  = static_cast<int>(args.number("repeat", 3));
    const bool profile = args.number("profile", 1) != 0;
    const auto tag     = args.text("tag", "model");
    const auto cap     = model.config().softcap;
    std::ofstream out(args.text("out"), std::ios::app);
    std::ofstream rounds_out(args.text("rounds-out", args.text("out") + ".rounds"), std::ios::app);
    if (!out || !rounds_out) { throw std::runtime_error("spec-audit: cannot open --out"); }

    base.drafter = nullptr;
    base.dflash  = nullptr;
    Session plain(model, device, base);
    struct PlainRun {
        std::vector<std::int32_t> tokens;
        std::vector<TopTwo> top; // top[i]: the logits after consuming tokens[i]
        std::int32_t length = 0;
        double rate         = 0;
    };
    std::vector<PlainRun> plain_runs(prompts.size());
    for (std::size_t p = 0; p < prompts.size(); ++p) {
        auto& run = plain_runs[p];
        plain.reset();
        plain.prefill(prompts[p].tokens);
        for (std::int32_t i = 0; i < want; ++i) {
            run.tokens.push_back(plain.decode(1).front());
            run.top.push_back(top_two(plain.last_logits()));
        }
        run.length = want;
        for (std::int32_t i = 0; i < want; ++i) {
            if (std::find(stops.begin(), stops.end(), run.tokens[i]) != stops.end()) {
                run.length = i + 1;
                break;
            }
        }
        std::vector<double> rates;
        for (int r = 0; r < repeat; ++r) {
            plain.reset();
            plain.prefill(prompts[p].tokens);
            const auto start = std::chrono::steady_clock::now();
            const auto again = plain.decode(run.length);
            rates.push_back(run.length / ninferno::seconds_since(start));
            if (!std::equal(again.begin(), again.end(), run.tokens.begin())) {
                std::fprintf(stderr, "spec-audit: %s plain decode is not deterministic\n",
                             prompts[p].name.c_str());
            }
        }
        run.rate = median(rates);
    }

    for (const auto& c : configs) {
        SessionOptions o     = base;
        o.drafter            = c.dflash ? nullptr : drafter;
        o.dflash             = c.dflash ? dflash : nullptr;
        o.draft_tokens       = c.k;
        o.min_draft_tokens   = c.k_min;
        Session spec(model, device, o);
        const auto continue_to = [&](std::int32_t length, std::vector<SpeculationRoundTrace>* t) {
            std::vector<std::int32_t> tokens;
            while (static_cast<std::int32_t>(tokens.size()) < length) {
                const auto missing = length - static_cast<std::int32_t>(tokens.size());
                const auto rounds  = std::max<std::int32_t>(1, (missing + c.k) / (c.k + 1));
                const auto part    = t != nullptr ? spec.speculate_traced(rounds, *t, profile)
                                                  : spec.speculate(rounds);
                tokens.insert(tokens.end(), part.begin(), part.end());
            }
            return tokens;
        };
        for (std::size_t p = 0; p < prompts.size(); ++p) {
            const auto& run = plain_runs[p];
            if (p == 0) { // warm every route and capture the graphs the first prompt reaches
                spec.reset();
                spec.prefill(prompts[p].tokens);
                (void)continue_to(run.length, nullptr);
            }
            std::vector<double> rates;
            std::vector<std::int32_t> timed;
            SpeculationStats before{}, after{};
            bool repeat_equal = true;
            for (int r = 0; r < repeat; ++r) {
                spec.reset();
                spec.prefill(prompts[p].tokens);
                before           = spec.speculation_stats();
                const auto start = std::chrono::steady_clock::now();
                auto tokens      = continue_to(run.length, nullptr);
                const double s   = ninferno::seconds_since(start);
                after            = spec.speculation_stats();
                rates.push_back(tokens.size() / s);
                if (r > 0 && tokens != timed) { repeat_equal = false; }
                timed = std::move(tokens);
            }
            std::vector<SpeculationRoundTrace> trace;
            spec.reset();
            spec.prefill(prompts[p].tokens);
            const auto traced = continue_to(run.length, &trace);
            // Where the continuation first leaves plain decode, and every later token plain
            // decode would not have chosen after the same prefix.
            const auto n = std::min<std::size_t>(traced.size(), run.tokens.size());
            std::size_t same = 0;
            while (same < n && traced[same] == run.tokens[same]) { ++same; }
            std::string mismatches;
            std::size_t checked = 0, mismatch_count = 0;
            double worst_deficit = 0;
            if (same < n && same > 0) {
                plain.reset();
                plain.prefill(prompts[p].tokens);
                if (same > 1) { (void)plain.decode(static_cast<std::int32_t>(same - 1)); }
                for (std::size_t i = same - 1; i + 1 < traced.size(); ++i) {
                    plain.set_pending_token(traced[i]);
                    (void)plain.decode(1);
                    const auto logits = plain.last_logits();
                    const auto top    = top_two(logits);
                    ++checked;
                    const auto next = traced[i + 1];
                    if (next != top.first_id) {
                        const double deficit = uncap(top.first, cap) - uncap(logits[next], cap);
                        const double gap     = uncap(top.first, cap) - uncap(top.second, cap);
                        worst_deficit        = std::max(worst_deficit, deficit);
                        ++mismatch_count;
                        // Also the soft-capped BF16 values plain decode compared: its own
                        // resolution at that magnitude decides whether this is a near-tie.
                        char item[200];
                        std::snprintf(item, sizeof(item), "%s[%zu,%.4f,%.4f,%d,%d,%.6g,%.6g,%.6g]",
                                      mismatches.empty() ? "" : ",", i + 1, deficit, gap, next,
                                      top.first_id, top.first, top.second, logits[next]);
                        mismatches += item;
                    }
                }
            }
            const double first_gap =
                same < n && same > 0
                    ? uncap(run.top[same - 1].first, cap) - uncap(run.top[same - 1].second, cap)
                    : -1.0;
            double phase[4] = {0, 0, 0, 0};
            for (const auto& t : trace) {
                phase[0] += t.draft_ms;
                phase[1] += t.forward_ms;
                phase[2] += t.head_ms;
                phase[3] += t.commit_ms;
                rounds_out << "{\"tag\":\"" << tag << "\",\"prompt\":\"" << prompts[p].name
                           << "\",\"config\":\"" << c.label << "\",\"position\":" << t.position
                           << ",\"depth\":" << t.depth << ",\"accepted\":" << t.accepted
                           << ",\"drafts\":[" << join_ids(t.drafts) << "],\"target\":["
                           << join_ids(t.target) << "],\"ms\":[" << t.draft_ms << ','
                           << t.forward_ms << ',' << t.head_ms << ',' << t.commit_ms << "]}\n";
            }
            const double rounds   = double(after.rounds - before.rounds);
            const double accepted = double(after.accepted - before.accepted);
            const double drafted  = double(after.drafted - before.drafted);
            const double nt       = trace.empty() ? 1.0 : double(trace.size());
            char head[1024];
            std::snprintf(
                head, sizeof(head),
                "{\"tag\":\"%s\",\"prompt\":\"%s\",\"prompt_tokens\":%zu,\"config\":\"%s\","
                "\"k\":%d,\"k_min\":%d,\"plain_tokens\":%d,\"plain_stopped\":%s,"
                "\"plain_tok_s\":%.2f,\"spec_tok_s\":%.2f,\"speedup\":%.4f,\"spec_tokens\":%zu,"
                "\"rounds\":%.0f,\"drafted\":%.0f,\"accepted\":%.0f,\"tokens_per_round\":%.4f,"
                "\"acceptance\":%.4f,\"repeat_equal\":%s,\"traced_equal\":%s,"
                "\"identical_prefix\":%zu,\"compared\":%zu,\"first_gap\":%.4f,"
                "\"replay_checked\":%zu,\"replay_mismatches\":%zu,\"worst_deficit\":%.4f,"
                "\"phase_ms\":[%.4f,%.4f,%.4f,%.4f],",
                tag.c_str(), prompts[p].name.c_str(), prompts[p].tokens.size(), c.label.c_str(),
                c.k, c.k_min, run.length, run.length < want ? "true" : "false", run.rate,
                median(rates), median(rates) / run.rate, timed.size(), rounds, drafted, accepted,
                (accepted + rounds) / rounds, accepted / std::max(drafted, 1.0),
                repeat_equal ? "true" : "false", traced == timed ? "true" : "false", same, n,
                first_gap, checked, mismatch_count, worst_deficit, phase[0] / nt, phase[1] / nt,
                phase[2] / nt, phase[3] / nt);
            out << head << "\"mismatches\":[" << mismatches << "],\"plain\":["
                << join_ids(std::vector<std::int32_t>(run.tokens.begin(),
                                                      run.tokens.begin() + run.length))
                << "],\"spec\":[" << join_ids(traced) << "]}\n";
            out.flush();
            rounds_out.flush();
            std::fprintf(stderr, "%s %s %s: x%.3f tok/round %.3f prefix %zu/%zu mism %zu\n",
                         tag.c_str(), prompts[p].name.c_str(), c.label.c_str(),
                         median(rates) / run.rate, (accepted + rounds) / rounds, same, n,
                         mismatch_count);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Speculation: drafters, spec-tune (maintainers) and the start-up choice (spec/manifest.h).

// The drafters a session can run, by the names profiles give them.
struct Drafters {
    std::unique_ptr<Drafter> mtp;
    std::unique_ptr<DFlashDrafter> dflash;
    std::string mtp_path, dflash_path;
    std::string mtp_name = "assistant", dflash_name = "dflash";

    void load_mtp(const std::string& path, DeviceContext& device, const Config& target) {
        if (mtp) { return; }
        mtp      = Drafter::load(path, device, target);
        mtp_path = path;
    }

    void load_dflash(const std::string& path, DeviceContext& device, const Config& target) {
        if (dflash) { return; }
        dflash      = DFlashDrafter::load(path, device, target);
        dflash_path = path;
    }
};

// Session options running one configuration.
SessionOptions configured(const spec::SpecConfig& c, SessionOptions o, const Drafters& d) {
    o.drafter          = nullptr;
    o.dflash           = nullptr;
    o.ngram            = false;
    o.min_draft_tokens = 0;
    switch (c.mode) {
    case spec::Mode::None:
        break;
    case spec::Mode::Mtp:
        if (!d.mtp) { throw std::invalid_argument(c.label() + " needs the MTP assistant"); }
        o.drafter          = d.mtp.get();
        o.draft_tokens     = c.k;
        o.min_draft_tokens = c.k_min;
        break;
    case spec::Mode::DFlash:
        if (!d.dflash) { throw std::invalid_argument(c.label() + " needs the DFlash drafter"); }
        o.dflash       = d.dflash.get();
        o.draft_tokens = c.k;
        break;
    case spec::Mode::Ngram:
        o.ngram        = true;
        o.draft_tokens = c.k;
        break;
    case spec::Mode::DFlash2:
        throw std::invalid_argument("Gemma 4 has no DFlash2 drafter");
    }
    return o;
}

// Token ids a thinking prompt carries and a plain one does not (the corpus records them per
// vocabulary); empty when unknown.
std::vector<std::int32_t> thinking_ids(const std::string& fingerprint) {
    static const std::map<std::string, std::vector<std::int32_t>> known = {
        {"v1:9a7777576d37749e", {98}}, // Gemma 4: <|think|> in the system turn
    };
    const auto it = known.find(fingerprint);
    return it == known.end() ? std::vector<std::int32_t>{} : it->second;
}

std::optional<bool> is_thinking_prompt(const std::vector<std::int32_t>& prompt,
                                       const std::string& fingerprint) {
    const auto markers = thinking_ids(fingerprint);
    if (markers.empty()) { return std::nullopt; }
    const auto head = prompt.begin() + std::min<std::size_t>(prompt.size(), 32);
    for (const auto id : markers) {
        if (std::find(prompt.begin(), head, id) == head) { return false; }
    }
    return true;
}

// spec-tune: the maintainer's measurement of every configuration the available drafters allow
// over the built-in corpus (its sections make one mixed load), within a time budget; writes the
// profile and,
// with --publish, its registry form.
int spec_tune(const Arguments& args, const Model& model, DeviceContext& device, SessionOptions base,
              const std::string& artifact, Drafters& drafters) {
    const auto started = std::chrono::steady_clock::now();
    const auto fp      = spec::artifact_vocab_fingerprint(artifact);
    std::optional<spec::Corpus> own;
    if (args.flag("corpus")) { own = spec::load_corpus(args.text("corpus")); }
    const auto& corpus = own ? *own : spec::builtin_corpus();
    const auto* rendered = fp ? corpus.tokens_for(fp->value) : nullptr;
    if (rendered == nullptr) {
        throw std::runtime_error(
            "spec-tune: the corpus has no rendered prompts for vocabulary " +
            (fp ? fp->value : std::string("(none)")) +
            "; add them with tools/spec/tokenize_corpus.py --tokenizer <checkpoint>");
    }
    std::vector<std::string> labels;
    if (args.flag("configs")) {
        labels = option_list(args.text("configs"), ',');
    } else {
        // Fixed MTP windows across what the serving Engine runs (1..15, spec/engine_support.h)
        // are measured too, so the served choices come from the same tune.
        for (const char* l : {"dflash:7", "dflash:11", "dflash:15", "mtp:1", "mtp:2", "mtp:3",
                              "mtp:4", "mtp:5", "mtp:6", "mtp:8", "mtp:10", "mtp:15", "mtp:8:2",
                              "mtp:15:2", "ngram:4", "ngram:8"}) {
            labels.emplace_back(l);
        }
    }
    std::vector<spec::SpecConfig> configs{spec::SpecConfig{}};
    for (const auto& label : labels) {
        const auto c = spec::SpecConfig::parse(label);
        if ((c.mode == spec::Mode::Mtp && !drafters.mtp) ||
            (c.mode == spec::Mode::DFlash && !drafters.dflash) || c.mode == spec::Mode::DFlash2 ||
            !c.enabled()) {
            continue;
        }
        configs.push_back(c);
    }
    const double budget = double(args.number("budget-seconds", 900));
    const int repeat    = static_cast<int>(args.number("repeat", 1));
    const auto& stops   = rendered->stops;
    const auto gpu      = spec::gpu_identity(device.device);

    spec::Profile profile;
    profile.artifact    = artifact;
    profile.created         = ninferno::utc_now();
    profile.key.artifact_id = spec::artifact_id_hex(artifact);
    profile.key.gpu     = gpu.id;
    std::vector<std::string> ids;
    if (drafters.mtp) {
        profile.drafters.push_back({drafters.mtp_name, "gemma4_assistant", "mtp", drafters.mtp_path,
                                    spec::artifact_id_hex(drafters.mtp_path), "", "", "", 0, 0});
        ids.push_back(profile.drafters.back().artifact_id);
    }
    if (drafters.dflash) {
        profile.drafters.push_back({drafters.dflash_name, "dflash", "dflash", drafters.dflash_path,
                                    spec::artifact_id_hex(drafters.dflash_path), "", "", "", 0, 0});
        ids.push_back(profile.drafters.back().artifact_id);
    }
    profile.drafters.push_back({"ngram", "ngram", "ngram", "", "", "", "", "", 0, 0});
    profile.key.drafters = spec::drafter_key(ids);
    char settings[160];
    std::snprintf(settings, sizeof(settings), "context %d, chunk %d, greedy, %s activations",
                  base.max_context, base.prefill_chunk, args.flag("a16") ? "A16" : "artifact");
    profile.settings = settings;

    std::map<std::string, std::vector<std::int32_t>> reference; // plain output per prompt
    for (const auto& config : configs) {
        const double elapsed = ninferno::seconds_since(started);
        if (config.enabled() && elapsed > budget) {
            profile.skipped.push_back(config.label());
            continue;
        }
        const auto options = configured(config, base, drafters);
        Session session(model, device, options);
        const auto run = [&](const spec::CorpusPrompt& p, bool timed, double* seconds) {
            const auto& prompt = rendered->prompts.at(p.id);
            session.reset();
            session.prefill(prompt);
            std::vector<std::int32_t> out;
            const auto t0 = std::chrono::steady_clock::now();
            if (!config.enabled()) {
                // Plain decode to the first stop id: one pass finds the length, the timed pass
                // decodes exactly that many tokens in one call.
                if (!timed) {
                    out = session.decode(p.max_tokens);
                    for (std::size_t i = 0; i < out.size(); ++i) {
                        if (std::find(stops.begin(), stops.end(), out[i]) != stops.end()) {
                            out.resize(i + 1);
                            break;
                        }
                    }
                } else {
                    out = session.decode(static_cast<std::int32_t>(reference.at(p.id).size()));
                }
            } else {
                out = session.speculate_until(p.max_tokens, stops);
            }
            if (seconds) { *seconds = ninferno::seconds_since(t0); }
            return out;
        };
        // Warm every route and capture the graphs on the first prompt.
        (void)run(corpus.prompts.front(), false, nullptr);
        if (!config.enabled()) {
            for (const auto& p : corpus.prompts) { reference[p.id] = run(p, false, nullptr); }
        }
        for (const char* family : spec::kFamilies) {
            spec::Measurement m;
            m.config = config;
            m.family = family;
            for (const auto& p : corpus.prompts) {
                if (p.family != family) { continue; }
                for (int r = 0; r < repeat; ++r) {
                    const auto before = session.speculation_stats();
                    double seconds    = 0;
                    const auto out    = run(p, true, &seconds);
                    const auto after  = session.speculation_stats();
                    m.prompts += 1;
                    m.tokens += static_cast<std::int64_t>(out.size());
                    m.seconds += seconds;
                    m.rounds += after.rounds - before.rounds;
                    m.drafted += after.drafted - before.drafted;
                    m.accepted += after.accepted - before.accepted;
                    const auto& ref = reference.at(p.id);
                    const auto n    = std::min(ref.size(), out.size());
                    std::size_t same = 0;
                    while (same < n && ref[same] == out[same]) { ++same; }
                    m.compared += static_cast<std::int64_t>(n);
                    m.identical += static_cast<std::int64_t>(same);
                }
            }
            if (m.prompts == 0) { continue; }
            std::fprintf(stderr, "spec-tune %-9s %-8s %7.1f tok/s  %5.2f tok/round  acc %.3f  "
                                 "identical %lld/%lld\n",
                         config.label().c_str(), family, m.tok_s(), m.tokens_per_round(),
                         m.acceptance(), static_cast<long long>(m.identical),
                         static_cast<long long>(m.compared));
            profile.measurements.push_back(m);
        }
    }
    // The Session runs more than the serving Engine does for Gemma 4 (DFlash, adaptive depths, MTP
    // windows past 5): the profile also carries the best choices the Engine runs.
    spec::select(profile, {.served = spec::engine_support("gemma4_text")});
    profile.tune_seconds = ninferno::seconds_since(started);
    const auto out_path  = args.flag("out") ? std::filesystem::path(args.text("out"))
                                            : spec::default_profile_path(artifact);
    spec::write_profile(out_path, profile);
    if (args.flag("publish")) {
        std::vector<std::string> applies;
        for (const auto& r : option_list(args.text("applies-to"), ',')) { applies.push_back(r); }
        const auto published = spec::published_from_profile(
            profile, args.text("gpu-class", gpu.chip), gpu.name, applies,
            args.text("measured-on", std::filesystem::path(artifact).filename().string()));
        std::ofstream(args.text("publish")) << published.dump(1) << '\n';
    }
    const auto show = [](const char* what, const spec::Choice& c) {
        std::fprintf(stderr, "choice %-12s %-10s x%.3f  %.1f tok/s  floor %.2f tok/round\n", what,
                     c.config.label().c_str(), c.speedup(), c.tok_s, c.floor_tokens_per_round);
    };
    show("non_thinking", profile.non_thinking);
    show("thinking", profile.thinking);
    show("default", profile.overall);
    if (profile.served) {
        show("served non_thinking", profile.served->non_thinking);
        show("served thinking", profile.served->thinking);
        show("served default", profile.served->overall);
    }
    for (const auto& d : profile.drafters) {
        std::fprintf(stderr, "drafter %-10s %s %s\n", d.name.c_str(), d.status.c_str(),
                     d.reason.c_str());
    }
    std::printf("{\"profile\":\"%s\",\"tune_seconds\":%.1f,\"gpu_class\":\"%s\",\"default\":\"%s\","
                "\"skipped\":%zu}\n",
                out_path.c_str(), profile.tune_seconds, gpu.id.c_str(),
                profile.overall.config.label().c_str(), profile.skipped.size());
    return 0;
}

// What generate and batch run: explicit flags, else the resolved profile (a maintainer's tune or
// the model's speculation manifest), else prompt lookup when the manifest says no drafter came
// with the model, else plain decode (an artifact without a manifest decodes as it always did).
struct StartPlan {
    spec::SpecConfig config;
    double floor = 0;
    std::string source;
    std::string mtp_path, dflash_path;
    std::vector<std::string> notes;
};

// `thinking`: the prompt's thinking marker (the profile's default for it; nullopt: the overall).
StartPlan start_plan(const Arguments& args, const std::string& artifact,
                     std::optional<bool> thinking, int device) {
    StartPlan plan;
    const auto flag = args.text("spec", "auto");
    if (flag == "off" || flag == "plain") { return plan; }
    if (flag != "auto") {
        plan.config = spec::SpecConfig::parse(flag);
        plan.source = "--spec";
        // An explicit configuration still uses the drafter that came with the model.
        const bool mtp = plan.config.mode == spec::Mode::Mtp;
        if (mtp || plan.config.mode == spec::Mode::DFlash) {
            const auto variant = args.text("drafter-variant", "");
            if (const auto m = spec::read_manifest(spec::manifest_path(artifact))) {
                for (const auto& d : m->drafters) {
                    if (!d.usable() || d.artifact.empty() ||
                        spec::split_variant(d.name).second != variant) {
                        continue;
                    }
                    if (mtp && d.kind == "gemma4_assistant" && plan.mtp_path.empty()) {
                        plan.mtp_path = d.artifact;
                    } else if (!mtp && d.kind == "dflash" && plan.dflash_path.empty()) {
                        plan.dflash_path = d.artifact;
                    }
                }
            }
        }
    } else {
        const auto gpu       = spec::gpu_identity(device);
        // A local tune only when named (--spec-profile): one beside the artifact is not read.
        const auto resolved  = spec::resolve_speculation(
            artifact, spec::artifact_id_hex(artifact), gpu.key,
            args.flag("spec-profile") ? std::filesystem::path(args.text("spec-profile"))
                                      : std::filesystem::path());
        spec::require_named_tune(resolved);
        plan.notes = resolved.notes;
        if (resolved.found()) {
            const auto& choice = spec::pick(resolved.profile, thinking);
            plan.config        = choice.config;
            plan.floor         = choice.floor_tokens_per_round;
            plan.source        = resolved.source;
            if (plan.config.enabled() && plan.config.mode != spec::Mode::Ngram) {
                const auto it = resolved.drafters.find(choice.drafter);
                if (it == resolved.drafters.end() || it->second.starts_with("component:")) {
                    plan.notes.push_back("drafter '" + choice.drafter + "' unavailable: plain decode");
                    plan.config = {};
                } else if (plan.config.mode == spec::Mode::Mtp) {
                    plan.mtp_path = it->second;
                } else {
                    plan.dflash_path = it->second;
                }
            }
        } else if (resolved.fallback == "ngram") {
            plan.config = spec::SpecConfig::parse("ngram:4");
            plan.source = "fallback (no drafter came with the model)";
        }
    }
    if (plan.config.enabled() && plan.floor <= 1.0) { plan.floor = 1.2; } // a wrong drafter's mark
    if (args.flag("acceptance-floor")) { plan.floor = std::stod(args.text("acceptance-floor")); }
    if (args.flag("no-watchdog")) { plan.floor = 0; }
    return plan;
}

// generate and batch with --drafter / --dflash: --spec names the configuration (of a loaded
// drafter's kind, or ngram:K), --spec off decodes plainly, and without --spec the loaded drafter
// runs at --draft (--draft-min: adaptive MTP depth). The watchdog floor follows start_plan.
StartPlan drafter_plan(const Arguments& args, bool mtp, bool dflash, std::int32_t k,
                       std::int32_t k_min) {
    StartPlan plan;
    const auto flag = args.text("spec", "auto");
    if (flag == "off" || flag == "plain") { return plan; }
    if (flag != "auto") {
        plan.config = spec::SpecConfig::parse(flag);
        plan.source = "--spec";
    } else {
        if (mtp && dflash) {
            throw std::invalid_argument(
                "--drafter and --dflash both loaded: name the configuration with --spec");
        }
        plan.config.mode  = mtp ? spec::Mode::Mtp : spec::Mode::DFlash;
        plan.config.k     = k;
        plan.config.k_min = mtp ? k_min : 0;
        plan.source       = mtp ? "--drafter" : "--dflash";
    }
    if ((plan.config.mode == spec::Mode::Mtp && !mtp) ||
        (plan.config.mode == spec::Mode::DFlash && !dflash)) {
        throw std::invalid_argument("--spec " + plan.config.label() +
                                    " needs its drafter loaded (--drafter for mtp, --dflash for "
                                    "dflash)");
    }
    if (plan.config.enabled() && plan.floor <= 1.0) { plan.floor = 1.2; }
    if (args.flag("acceptance-floor")) { plan.floor = std::stod(args.text("acceptance-floor")); }
    if (args.flag("no-watchdog")) { plan.floor = 0; }
    return plan;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse(argc, argv);
        // Checkpoint directories are converted before the device is touched.
        const bool consume  = args.flag("consume-source");
        const auto artifact = artifact_path(args.artifact, consume);
        const auto drafter_artifact =
            args.flag("drafter")
                ? artifact_path(args.text("drafter"), consume, target_resources(args.artifact))
                : std::string();
        const auto dflash_artifact =
            args.flag("dflash")
                ? artifact_path(args.text("dflash"), consume, target_resources(args.artifact))
                : std::string();
        if (args.flag("a16") && args.flag("a16-full")) {
            throw std::invalid_argument("--a16 and --a16-full are exclusive");
        }
        const bool exact = product::parse_activation_mode(args.text("activations", "fast")) ==
                           ActivationMode::Exact;
        if (exact && (args.flag("a16") || args.flag("a16-full"))) {
            throw std::invalid_argument(
                "--activations exact sets its own profile; --a16 and --a16-full are fast-mode studies");
        }
        // For the rest of the run: this thread's ops resolve their routes under it.
        const ops::ScopedInvariantRoutes invariant_routes(exact);
        const auto profile       = exact || args.flag("a16-full") ? ActivationProfile::A16Full
                                   : args.flag("a16")             ? ActivationProfile::A16
                                                                  : ActivationProfile::Artifact;
        const char* profile_name = exact ? "exact"
                                   : profile == ActivationProfile::A16Full ? "a16_full"
                                   : profile == ActivationProfile::A16   ? "a16"
                                                                         : "artifact";
        DeviceContext device(0);
        const auto load_start = std::chrono::steady_clock::now();
        auto model            = Model::load(artifact, device, profile);
        const double load_s   = ninferno::seconds_since(load_start);
        SessionOptions options;
        const bool tuning     = args.command == "spec-tune";
        options.max_context   = static_cast<std::int32_t>(args.number("context", tuning ? 4096 : 32768));
        options.prefill_chunk = static_cast<std::int32_t>(args.number("chunk", 2048));
        const std::string kv_dtype = args.text("kv-dtype", "bf16");
        if (kv_dtype != "bf16" && kv_dtype != "fp8" && kv_dtype != "fp8-full") {
            throw std::invalid_argument("--kv-dtype must be bf16, fp8 or fp8-full");
        }
        options.kv_fp8      = kv_dtype == "fp8";
        options.kv_fp8_full = kv_dtype == "fp8-full";
        Drafters drafters;
        if (args.flag("drafter")) {
            drafters.load_mtp(drafter_artifact, device, model->config());
            options.drafter      = drafters.mtp.get();
            options.draft_tokens = static_cast<std::int32_t>(args.number("draft", 4));
        }
        if (args.flag("dflash")) {
            drafters.load_dflash(dflash_artifact, device, model->config());
            options.dflash = drafters.dflash.get();
            options.draft_tokens = static_cast<std::int32_t>(
                args.number("draft", drafters.dflash->config().block_size - 1));
        }
        options.min_draft_tokens = static_cast<std::int32_t>(args.number("draft-min", 0));
        // Only score keeps the full scoring head slab; the rest need the widest verify.
        if (args.command != "score") { options.score_columns = kMaxVerifyColumns; }
        // Without --context and --chunk the session is sized to the memory free now
        // (fit_session): on a card with room this is the old default (32768 / 2048; spec-tune 4096
        // / 2048), on a smaller one the largest context that fits, with the widest chunk that
        // keeps it. `probe` carries the heaviest drafter configuration the command will run.
        const auto sized = [&](SessionOptions o, SessionOptions probe) {
            if (args.flag("context") && args.flag("chunk")) { return o; }
            SessionFitRequest request;
            request.context     = args.flag("context") ? o.max_context : 0;
            request.chunk       = args.flag("chunk") ? o.prefill_chunk : 0;
            request.max_context = tuning ? 4096 : 32768;
            const auto fit      = fit_session(*model, device, probe, request);
            o.max_context       = fit.context;
            o.prefill_chunk     = fit.chunk;
            std::fprintf(stderr,
                         "ninferno-gemma4: sized to the device: context %d, chunk %d (%.0f MiB "
                         "free, %.0f MiB kept for CUDA graphs, %d trial sessions)\n",
                         fit.context, fit.chunk, fit.free_bytes / 1048576.0,
                         fit.reserve_bytes / 1048576.0, fit.trials);
            return o;
        };
        if (args.command == "spec-tune") {
            // Loaded before the session is sized, so the probe below carries them: sized
            // without them, the session took the memory the drafters then needed (Gemma 4
            // 31B on the RTX 3090: 7.4 GiB free sized for the target alone, then out of
            // memory at the first drafter trial).
            // Without drafter flags: the drafters attached to the model (its manifest).
            // As published, or with --drafter-variant V the variants "<name>.V" beside them;
            // one drafter of each kind (the first the manifest lists).
            if (!args.flag("drafter") && !args.flag("dflash")) {
                const auto variant = args.text("drafter-variant", "");
                if (const auto m = spec::read_manifest(spec::manifest_path(artifact))) {
                    for (const auto& d : m->drafters) {
                        if (!d.usable() || d.artifact.empty() ||
                            spec::split_variant(d.name).second != variant) {
                            continue;
                        }
                        if (d.kind == "gemma4_assistant" && !drafters.mtp) {
                            drafters.load_mtp(d.artifact, device, model->config());
                            drafters.mtp_name = d.name;
                        } else if (d.kind == "dflash" && !drafters.dflash) {
                            drafters.load_dflash(d.artifact, device, model->config());
                            drafters.dflash_name = d.name;
                        }
                    }
                }
                if (!variant.empty() && !drafters.mtp && !drafters.dflash) {
                    throw std::runtime_error("spec-tune --drafter-variant " + variant +
                                             ": the manifest attaches no drafter of that variant");
                }
            } else if (args.flag("drafter-variant")) {
                throw std::invalid_argument(
                    "--drafter-variant picks the manifest's drafters; --drafter/--dflash name them");
            }
        }
        if (args.command == "spec-audit" || tuning) {
            // The heaviest configuration these commands may run: the DFlash drafter's workspace
            // and rings, else the deepest MTP schedule (one graph per depth, 1..15). A session
            // runs one drafter, so the probe carries only that one (both may be loaded).
            SessionOptions probe = options;
            if (drafters.dflash) {
                probe.drafter      = nullptr;
                probe.dflash       = drafters.dflash.get();
                probe.draft_tokens = drafters.dflash->config().block_size - 1;
            } else if (drafters.mtp) {
                probe.drafter          = drafters.mtp.get();
                probe.draft_tokens     = kMaxVerifyColumns - 1;
                probe.min_draft_tokens = 1;
            }
            options = sized(options, probe);
        }
        if (args.command == "spec-audit") {
            return spec_audit(args, *model, device, options, drafters.mtp.get(),
                              drafters.dflash.get());
        }
        if (args.command == "spec-tune") {
            SessionOptions base = options;
            base.drafter        = nullptr;
            base.dflash         = nullptr;
            return spec_tune(args, *model, device, base, artifact, drafters);
        }
        const auto tokens = read_tokens(args.text("tokens"));
        // batch files hold [count][len0][ids0...][len1][ids1...]...
        std::vector<std::span<const std::int32_t>> batch_prompts;
        if (args.command == "batch") {
            std::size_t cursor = 0;
            const auto next    = [&]() {
                if (cursor >= tokens.size()) { throw std::runtime_error("truncated prompt file"); }
                return tokens[cursor++];
            };
            const std::int32_t count = next();
            for (std::int32_t index = 0; index < count; ++index) {
                const std::int32_t length = next();
                if (length <= 0 || cursor + length > tokens.size()) {
                    throw std::runtime_error("bad prompt length");
                }
                batch_prompts.emplace_back(tokens.data() + cursor, static_cast<std::size_t>(length));
                cursor += static_cast<std::size_t>(length);
            }
        }
        // generate and batch speculate by default when the model came with a profile: each prompt
        // runs its default for the prompt's thinking marker. A batch runs its prompts grouped by
        // that choice.
        const SessionOptions base_options = options;
        struct BatchGroup {
            StartPlan plan;
            std::vector<std::size_t> prompts;
        };
        std::vector<BatchGroup> batch_groups;
        StartPlan plan;
        if ((args.command == "generate" || args.command == "batch") && !args.flag("drafter") &&
            !args.flag("dflash")) {
            std::optional<std::string> fingerprint;
            if (std::filesystem::exists(spec::manifest_path(artifact)) ||
                args.flag("spec-profile")) {
                // The artifact tokenizer's fingerprint: the thinking marker's id.
                if (const auto tokenizer = spec::artifact_tokenizer_json(artifact)) {
                    fingerprint = spec::vocab_fingerprint(*tokenizer).value;
                }
            }
            // The plan of one prompt: its thinking marker, then the profile's default for it.
            std::map<int, StartPlan> plans;
            const auto plan_of = [&](std::span<const std::int32_t> prompt) {
                std::optional<bool> thinking;
                if (fingerprint) {
                    thinking = is_thinking_prompt(
                        std::vector<std::int32_t>(prompt.begin(), prompt.end()), *fingerprint);
                }
                const int key = thinking ? int(*thinking) : -1;
                auto it       = plans.find(key);
                if (it == plans.end()) {
                    it = plans.emplace(key, start_plan(args, artifact, thinking, device.device))
                             .first;
                }
                return it->second;
            };
            if (args.command == "batch") {
                for (std::size_t index = 0; index < batch_prompts.size(); ++index) {
                    StartPlan prompt_plan = plan_of(batch_prompts[index]);
                    const auto same = [&](const BatchGroup& g) {
                        return g.plan.config == prompt_plan.config && g.plan.floor == prompt_plan.floor;
                    };
                    auto group = std::find_if(batch_groups.begin(), batch_groups.end(), same);
                    if (group == batch_groups.end()) {
                        batch_groups.push_back({std::move(prompt_plan), {}});
                        group = batch_groups.end() - 1;
                    }
                    group->prompts.push_back(index);
                }
                if (!batch_groups.empty()) { plan = batch_groups.front().plan; }
            } else {
                plan = plan_of(tokens);
            }
            for (const auto& note : plan.notes) {
                std::fprintf(stderr, "ninferno-gemma4: speculation: %s\n", note.c_str());
            }
            // Every drafter a prompt needs, loaded once.
            const auto load = [&](const StartPlan& p) {
                if (!p.mtp_path.empty()) { drafters.load_mtp(p.mtp_path, device, model->config()); }
                if (!p.dflash_path.empty()) {
                    drafters.load_dflash(p.dflash_path, device, model->config());
                }
            };
            load(plan);
            for (const auto& group : batch_groups) { load(group.plan); }
        } else if (args.command == "generate" || args.command == "batch") {
            // A drafter named on the command line is the one these commands speculate with
            // (before, they loaded it and decoded plainly without a word).
            plan = drafter_plan(args, drafters.mtp != nullptr, drafters.dflash != nullptr,
                                options.draft_tokens, options.min_draft_tokens);
        }
        // The session options a plan runs with.
        const auto planned = [&](const StartPlan& p) {
            SessionOptions o = base_options;
            if (p.config.enabled()) {
                o                  = configured(p.config, o, drafters);
                o.acceptance_floor = p.floor;
                o.log              = [](const std::string& line) {
                    std::fprintf(stderr, "ninferno-gemma4: %s\n", line.c_str());
                };
                std::fprintf(stderr, "ninferno-gemma4: speculation %s (%s), watchdog floor %.2f\n",
                             p.config.label().c_str(), p.source.c_str(), p.floor);
            }
            return sized(o, o);
        };
        if (plan.config.enabled() || !batch_groups.empty()) {
            options = planned(plan);
        } else {
            options = sized(options, options);
        }
        std::optional<Session> session_holder(std::in_place, *model, device, options);
        Session& session = *session_holder;
        // Requests (generate, batch) find their graphs captured: the load pays for them, not the
        // first request's decode. The device memory they take is reported next to the session's.
        const bool warm = (args.command == "generate" || args.command == "batch") &&
                          !args.flag("no-warm-up");
        double warm_s = 0, warm_mib = 0;
        if (warm) {
            std::size_t free_before = 0, free_after = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
            const auto warm_start = std::chrono::steady_clock::now();
            session.warm_up();
            warm_s = ninferno::seconds_since(warm_start);
            CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
            warm_mib = (double(free_before) - double(free_after)) / 1048576.0;
        }
        // The CUDA synchronization schedule (NINFERNO_CUDA_SYNC) belongs to the measurement: the
        // adaptive draft depth waits on the device once per round.
        std::printf("{\"load_seconds\":%.2f,\"weights_gib\":%.3f,\"session_gib\":%.3f,"
                    "\"kv_gib\":%.3f,\"kv_dtype\":\"%s\",\"free_gib\":%.3f,"
                    "\"profile\":\"%s\",\"chunk\":%d,\"context\":%d,\"cuda_sync\":\"%s\",",
                    load_s, model->device_bytes() / 1073741824.0,
                    session.device_bytes() / 1073741824.0, session.kv_bytes() / 1073741824.0,
                    kv_dtype.c_str(), free_device_bytes() / 1073741824.0, profile_name,
                    options.prefill_chunk, options.max_context, device.sync_mode());
        if (warm) {
            std::printf("\"warm_up_seconds\":%.3f,\"warm_up_mib\":%.1f,", warm_s, warm_mib);
        }

        if (args.command == "bench") {
            const auto prompt = static_cast<std::size_t>(args.number("prompt", 512));
            const auto decode = static_cast<std::int32_t>(args.number("decode", 128));
            const auto repeat = static_cast<int>(args.number("repeat", 3));
            if (tokens.size() < prompt) {
                throw std::runtime_error("token file shorter than prompt");
            }
            const std::span<const std::int32_t> prompt_tokens(tokens.data(), prompt);
            std::vector<double> prefill_s, decode_s;
            std::vector<std::int32_t> generated;
            for (int r = 0; r < repeat + 1; ++r) {
                session.reset();
                auto start = std::chrono::steady_clock::now();
                session.prefill(prompt_tokens);
                const double p = ninferno::seconds_since(start);
                // The first decode call captures the graph; time a separate window after it.
                (void)session.decode(4);
                start          = std::chrono::steady_clock::now();
                const auto out = session.decode(decode);
                const double d = ninferno::seconds_since(start);
                if (r == 0) {
                    generated = out;
                    continue;
                } // the first round warms every route
                prefill_s.push_back(p);
                decode_s.push_back(d);
            }
            std::sort(prefill_s.begin(), prefill_s.end());
            std::sort(decode_s.begin(), decode_s.end());
            const double p_med = prefill_s[prefill_s.size() / 2];
            const double d_med = decode_s[decode_s.size() / 2];
            std::printf(
                "\"prompt_tokens\":%zu,\"decode_tokens\":%d,\"repeats\":%d,"
                "\"prefill_seconds_median\":%.4f,\"prefill_tok_s\":%.1f,"
                "\"prefill_tok_s_best\":%.1f,\"decode_seconds_median\":%.4f,"
                "\"decode_tok_s\":%.2f,\"decode_tok_s_best\":%.2f,"
                "\"decode_start_position\":%zu,\"generated_head\":[%s]}\n",
                prompt, decode, repeat, p_med, prompt / p_med, prompt / prefill_s.front(), d_med,
                decode / d_med, decode / decode_s.front(), prompt + 4,
                join_ids(std::vector<std::int32_t>(generated.begin(),
                                                   generated.begin() +
                                                       std::min<std::size_t>(16, generated.size())))
                    .c_str());
        } else if (args.command == "spec-bench") {
            // Plain greedy decode against greedy speculation from the same prompt, in one process.
            // The first round of each warms routes and captures graphs; later rounds are timed.
            // Both streams end at the plain stream's first stop id (--stop, as batch), at most
            // --decode tokens: the plain decode of the warm round fixes that length, and the
            // speculation runs speculate_until(length, stops), whose rounds are sized to the
            // tokens still missing - past a stop it drafts at most one round's worth, not
            // continuations nobody asked for (which inflated the acceptance).
            if (!drafters.mtp && !drafters.dflash) {
                throw std::invalid_argument("spec-bench needs --drafter or --dflash");
            }
            const auto prompt = static_cast<std::size_t>(args.number("prompt", 512));
            const auto want   = static_cast<std::int32_t>(args.number("decode", 256));
            const auto repeat = static_cast<int>(args.number("repeat", 3));
            if (tokens.size() < prompt) {
                throw std::runtime_error("token file shorter than prompt");
            }
            const auto stops = stop_ids(args);
            const std::span<const std::int32_t> prompt_tokens(tokens.data(), prompt);
            const auto k       = options.draft_tokens;
            std::int32_t length = want;
            bool stopped        = false;
            std::vector<double> plain_rate, spec_rate;
            std::vector<std::int32_t> plain_out, spec_out;
            SpeculationStats before{}, after{};
            for (int r = 0; r < repeat + 1; ++r) {
                session.reset();
                session.prefill(prompt_tokens);
                auto start = std::chrono::steady_clock::now();
                auto plain = session.decode(length);
                const double p = ninferno::seconds_since(start);
                if (r == 0) {
                    const auto stop = std::find_first_of(plain.begin(), plain.end(),
                                                         stops.begin(), stops.end());
                    stopped = stop != plain.end();
                    if (stopped) { length = static_cast<std::int32_t>(stop - plain.begin()) + 1; }
                }
                session.reset();
                session.prefill(prompt_tokens);
                if (r == 0) {
                    (void)session.speculate(1);
                    session.reset();
                    session.prefill(prompt_tokens);
                }
                before          = session.speculation_stats();
                start           = std::chrono::steady_clock::now();
                const auto spec = session.speculate_until(length, stops);
                const double q  = ninferno::seconds_since(start);
                after           = session.speculation_stats();
                if (r == 0) { continue; }
                plain_rate.push_back(plain.size() / p);
                spec_rate.push_back(spec.size() / q);
                plain_out = std::move(plain);
                spec_out  = spec;
            }
            std::sort(plain_rate.begin(), plain_rate.end());
            std::sort(spec_rate.begin(), spec_rate.end());
            if (args.flag("dump-tokens")) {
                // Two lines, plain decode then speculation: the streams a refactor must keep.
                std::ofstream dump(args.text("dump-tokens"));
                dump << join_ids(plain_out) << '\n' << join_ids(spec_out) << '\n';
            }
            std::size_t same = 0;
            while (same < plain_out.size() && same < spec_out.size() &&
                   plain_out[same] == spec_out[same]) {
                ++same;
            }
            const double r_rounds   = double(after.rounds - before.rounds);
            const double r_accepted = double(after.accepted - before.accepted);
            std::printf("\"prompt_tokens\":%zu,\"draft_tokens\":%d,\"length\":%d,\"stopped\":%s,"
                        "\"rounds\":%.0f,"
                        "\"plain_tok_s\":%.2f,\"spec_tok_s\":%.2f,\"speedup\":%.3f,"
                        "\"tokens_per_round\":%.3f,\"accepted_per_round\":%.3f,"
                        "\"acceptance_rate\":%.4f,\"spec_tokens\":%zu,"
                        "\"identical_prefix\":%zu,\"compared\":%zu}\n",
                        prompt, k, length, stopped ? "true" : "false", r_rounds,
                        plain_rate[plain_rate.size() / 2], spec_rate[spec_rate.size() / 2],
                        spec_rate[spec_rate.size() / 2] / plain_rate[plain_rate.size() / 2],
                        (r_accepted + r_rounds) / r_rounds, r_accepted / r_rounds,
                        r_accepted / (r_rounds * k), spec_out.size(), same,
                        std::min(plain_out.size(), spec_out.size()));
        } else if (args.command == "score") {
            // Non-overlapping windows of W tokens, each scored from an empty cache: the standard
            // fixed-window perplexity, comparable across engines when W and the text match. Every
            // window starts at the file's first token (its BOS) in place of its own first token, as
            // llama.cpp's perplexity does: a Gemma 4 window without BOS degrades along its length
            // (26B, Pride and Prejudice: ppl 954 at W 4096 and 1420 at W 8192 without it).
            const auto window = static_cast<std::size_t>(args.number("window", 2048));
            std::vector<std::int32_t> slice;
            double nll        = 0;
            std::size_t count = 0;
            std::vector<float> all_logprobs;
            std::vector<std::int32_t> all_predictions;
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t begin = 0; begin + 1 < tokens.size(); begin += window) {
                const auto n = std::min(window, tokens.size() - begin);
                if (n < 2) { break; }
                session.reset();
                slice.assign(tokens.begin() + begin, tokens.begin() + begin + n);
                slice.front() = tokens.front();
                const auto lp = session.score(slice, &all_predictions);
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
            std::printf(
                "\"scored_tokens\":%zu,\"window\":%zu,\"mean_nll\":%.6f,\"perplexity\":%.4f,"
                "\"seconds\":%.2f}\n",
                count, window, nll / count, std::exp(nll / count), ninferno::seconds_since(start));
        } else if (args.command == "generate") {
            const auto requested = static_cast<std::int32_t>(args.number("decode", 64));
            std::size_t begin    = 0;
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
            // A --decode that runs past the context is cut at the context end, with a warning,
            // like a server's length finish; the session refuses only a prompt that does not fit.
            const std::int32_t room   = options.max_context - session.position();
            const std::int32_t decode = std::min(requested, room);
            if (decode < requested) {
                std::fprintf(stderr,
                             "ninferno-gemma4: warning: --decode %d runs past the context (%d "
                             "positions, %zu-token prompt); decoding %d\n",
                             requested, options.max_context, tokens.size(), std::max(decode, 0));
            }
            const char* truncated = decode < requested ? ",\"truncated_by_context\":true" : "";
            if (decode <= 0) {
                std::printf("\"prompt_tokens\":%zu,\"generated\":[]%s}\n", tokens.size(), truncated);
            } else if (!plan.config.enabled()) {
                const auto out = session.decode(decode);
                std::printf("\"prompt_tokens\":%zu,\"generated\":[%s]%s}\n", tokens.size(),
                            join_ids(out).c_str(), truncated);
            } else {
                const auto start = std::chrono::steady_clock::now();
                const auto out   = session.speculate_until(decode);
                const double s   = ninferno::seconds_since(start);
                const auto stats = session.speculation_stats();
                std::printf("\"prompt_tokens\":%zu,\"generated\":[%s],\"speculation\":{"
                            "\"config\":\"%s\",\"source\":\"%s\",\"tok_s\":%.2f,\"rounds\":%lld,"
                            "\"tokens_per_round\":%.3f,\"plain_steps\":%lld,\"latched\":%s}%s}\n",
                            tokens.size(), join_ids(out).c_str(), plan.config.label().c_str(),
                            plan.source.c_str(), out.size() / s,
                            static_cast<long long>(stats.rounds), stats.tokens_per_round(),
                            static_cast<long long>(session.plain_steps()),
                            session.speculation_latched() ? "true" : "false", truncated);
            }
        } else if (args.command == "batch") {
            // --tokens holds [count][len0][ids0...][len1][ids1...]...; each prompt is prefilled
            // from an empty cache and continued until a stop id or --decode tokens.
            const auto max_new = static_cast<std::int32_t>(args.number("decode", 1024));
            const auto block   = static_cast<std::int32_t>(args.number("block", 16));
            const std::vector<std::int32_t> stops = stop_ids(args);
            const float temperature = std::stof(args.text("temperature", "0"));
            const float top_p       = std::stof(args.text("top-p", "1"));
            const auto top_k        = static_cast<std::int32_t>(args.number("top-k", 20));
            const auto seed         = static_cast<unsigned long long>(args.number("seed", 0));
            std::printf("\"command\":\"batch\",\"temperature\":%g,\"top_p\":%g,\"top_k\":%d,"
                        "\"seed\":%llu}\n",
                        temperature, top_p, top_k, seed);
            std::fflush(stdout);
            const auto count       = static_cast<std::int32_t>(batch_prompts.size());
            double generated_total = 0, decode_seconds = 0;
            if (batch_groups.empty()) {
                std::vector<std::size_t> all(batch_prompts.size());
                for (std::size_t i = 0; i < all.size(); ++i) { all[i] = i; }
                batch_groups.push_back({plan, std::move(all)});
            }
            std::vector<std::string> lines(batch_prompts.size());
            for (std::size_t g = 0; g < batch_groups.size(); ++g) {
                const BatchGroup& group = batch_groups[g];
                SessionOptions group_options = options;
                if (g > 0) {
                    // The previous group's session goes first: one session's memory at a time.
                    session_holder.reset();
                    group_options = planned(group.plan);
                    session_holder.emplace(*model, device, group_options);
                    if (warm) { session_holder->warm_up(); }
                }
                Session& s = *session_holder;
                const std::string config =
                    group.plan.config.enabled() ? group.plan.config.label() : std::string("plain");
                for (const std::size_t index : group.prompts) {
                    const std::span<const std::int32_t> prompt = batch_prompts[index];
                    s.reset();
                    s.set_sampling(temperature, top_p, top_k, seed + index);
                    const auto start = std::chrono::steady_clock::now();
                    s.prefill(prompt);
                    const double prefill = ninferno::seconds_since(start);
                    // decode() returns the tokens each step consumed, so its first element is
                    // the one the prefill drew: the continuation starts there with nothing to
                    // prepend.
                    std::vector<std::int32_t> out;
                    const auto decode_start = std::chrono::steady_clock::now();
                    bool stopped            = false;
                    const bool speculative  = group.plan.config.enabled() && temperature <= 0;
                    if (speculative) {
                        // Greedy speculation: the same tokens up to verify near-ties, faster.
                        out     = s.speculate_until(std::min(max_new, group_options.max_context -
                                                                          s.position() - 1),
                                                    stops);
                        stopped = !out.empty() &&
                                  std::find(stops.begin(), stops.end(), out.back()) != stops.end();
                    }
                    while (!speculative && !stopped &&
                           static_cast<std::int32_t>(out.size()) < max_new &&
                           s.position() + block < group_options.max_context) {
                        const auto part = s.decode(block);
                        for (const auto id : part) {
                            out.push_back(id);
                            if (std::find(stops.begin(), stops.end(), id) != stops.end()) {
                                stopped = true;
                                break;
                            }
                            if (static_cast<std::int32_t>(out.size()) >= max_new) { break; }
                        }
                    }
                    const double decode = ninferno::seconds_since(decode_start);
                    generated_total += static_cast<double>(out.size());
                    decode_seconds += decode;
                    char head[256];
                    std::snprintf(head, sizeof(head),
                                  "{\"index\":%zu,\"prompt_tokens\":%zu,\"prefill_seconds\":%.4f,"
                                  "\"decode_seconds\":%.4f,\"stopped\":%s,\"config\":\"%s\",",
                                  index, prompt.size(), prefill, decode,
                                  stopped ? "true" : "false", config.c_str());
                    lines[index] = std::string(head) + "\"tokens\":[" + join_ids(out) + "]}";
                }
            }
            for (const auto& line : lines) { std::printf("%s\n", line.c_str()); }
            std::fflush(stdout);
            std::fprintf(stderr, "batch: %d prompts, %.0f tokens, %.1f tok/s decode, %zu configurations\n",
                         count, generated_total, generated_total / decode_seconds,
                         batch_groups.size());
        } else if (args.command == "logits") {
            // --prefill-split P1,P2,...: the prompt is prefilled in separate calls cut at these
            // absolute positions (each call still chunks by --chunk), which is how a restored or
            // continued turn meets the cache. --decode M then continues greedily.
            std::vector<std::size_t> cuts;
            {
                std::string item;
                for (const char ch : args.text("prefill-split") + ",") {
                    if (ch == ',') {
                        if (!item.empty()) { cuts.push_back(std::stoull(item)); }
                        item.clear();
                    } else {
                        item += ch;
                    }
                }
            }
            cuts.push_back(tokens.size());
            std::size_t begin = 0;
            for (const std::size_t cut : cuts) {
                if (cut <= begin || cut > tokens.size()) {
                    throw std::invalid_argument("--prefill-split: cuts must rise inside the prompt");
                }
                session.prefill({tokens.data() + begin, cut - begin});
                begin = cut;
            }
            const auto logits = session.last_logits();
            std::ofstream out(args.text("out"), std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()),
                      static_cast<std::streamsize>(logits.size() * sizeof(float)));
            std::printf("\"prompt_tokens\":%zu,\"written\":%zu", tokens.size(), logits.size());
            if (args.flag("decode")) {
                const auto generated =
                    session.decode(static_cast<std::int32_t>(args.number("decode", 64)));
                std::printf(",\"generated\":[%s]", join_ids(generated).c_str());
            }
            std::printf("}\n");
        } else {
            throw std::invalid_argument("unknown command " + args.command);
        }
        std::fflush(stdout);
        std::fprintf(stderr, "ninferno-gemma4: device memory free after the run: %.3f GiB\n",
                     free_device_bytes() / 1073741824.0);
        return 0;
    } catch (const std::exception& error) {
        std::fflush(stdout);
        std::cerr << "ninferno-gemma4: " << error.what() << '\n';
        return 1;
    }
}
