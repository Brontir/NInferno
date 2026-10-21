// ninferno spec-tune: the maintainer's measurement of an artifact's speculation configurations (its
// built-in drafters, and the ones that came beside it in <artifact>.speculation.json) on the
// built-in corpus (spec/corpus.h), a mix of math, code, prose and thinking prompts judged as
// one load, within a time budget. One engine per
// configuration (the speculative backend and its K are fixed at engine start); prompts are
// rendered by the engine's own chat template. Writes <artifact>.spec_profile.json (spec/
// profile.h) and, with --publish, the registry form of the profile for this GPU class.
//
// ninferno spec-tune --load measures instead the profile's bounds under decode load
// (spec::Profile::load) for any model the Engine serves with a profile: the server's speculation
// (--spec auto, its default for every request) against plain decode at each concurrency.

#include "spec_tune.h"

#include "artifact/reader.h"
#include "core/clock.h"
#include "core/sha256.h"
#include "core/paged_kv_cache.h"
#include "ninferno/engine.h"
#include "spec/config.h"
#include "spec/corpus.h"
#include "spec/engine_support.h"
#include "spec/fingerprint.h"
#include "spec/gpu_identity.h"
#include "spec/load_tune.h"
#include "spec/manifest.h"
#include "spec/profile.h"
#include "spec/registry.h"
#include "product/speculation/speculation.h"
#include "product/speculative_options.h"
#include "product/numeric_options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ninferno::cli {
using ninferno::product::option_list;
using ninferno::product::parse_float_in;
using ninferno::product::parse_nonnegative_int;
using ninferno::product::parse_u32;

namespace {

KvCacheStorage kv_storage(const std::string& text) {
    if (const auto storage = parse_kv_cache_storage(text)) { return *storage; }
    throw std::invalid_argument("spec-tune: --kv-dtype " + std::string(kKvCacheStorageNames));
}

PromptInput user_prompt(const spec::CorpusPrompt& p) {
    PromptInput in;
    ChatMessage message;
    message.role = ChatRole::User;
    MessagePart part;
    part.kind = MessagePartKind::Text;
    part.text = p.text;
    message.parts.push_back(std::move(part));
    in.messages.push_back(std::move(message));
    in.options.enable_thinking = p.thinking;
    return in;
}

RequestOptions greedy(std::uint32_t budget) {
    RequestOptions r;
    r.execution.requested_output_tokens    = budget;
    r.execution.allow_prefix_reuse         = false;
    r.execution.sampling.temperature       = 0.0F;
    r.execution.sampling.presence_penalty  = 0.0F;
    r.execution.sampling.frequency_penalty = 0.0F;
    r.execution.sampling.top_k             = 0;
    r.execution.sampling.top_p             = 1.0F;
    r.execution.sampling.min_p             = 0.0F;
    r.execution.sampling.seed              = 1;
    return r;
}

// A request's speculation under a bound of `bound` drafts, as a step of the server's engine bounded
// at `bound` runs it: at its own depth when that is no deeper (with its watchdog), else at the
// bound with no watchdog (a load-capped round is not judged); 0: plain decode.
RequestSpeculation bounded(RequestSpeculation speculation, std::uint32_t bound) {
    if (bound == 0) { return RequestSpeculation{}; }
    if (speculation.draft_tokens <= bound) { return speculation; }
    return RequestSpeculation{.draft_tokens = bound, .acceptance_floor = 0.0, .step_down = {}};
}

// The corpus as the cells run it: every prompt's text, thinking flag and budget.
std::string corpus_digest(const spec::Corpus& corpus) {
    std::string bytes;
    for (const auto& p : corpus.prompts) {
        bytes += p.id + '\n' + p.family + '\n' + (p.thinking ? "1" : "0") + '\n' +
                 std::to_string(p.max_tokens) + '\n' + p.text + '\0';
    }
    return ninferno::sha256_hex(bytes);
}

// What the tune's engines run under besides the cell's own configuration.
// - How they wait for their decode rounds (--decode-wait, DecodePacingOptions::wait). Each round
//   of a stalling host chooses between spinning and blocking by a probe (auto, round_pacing.h),
//   so a busy host would change what a tune measures and how; the tune waits as a quiet host does
//   (spin) unless --decode-wait says otherwise.
// - --cuda-sync, and the route tables under test (--route-table FILE, --route-dir DIR): a lab
//   tunes speculation against the tables it is about to publish.
struct TuneEngine {
    DecodePacingOptions pacing{.wait = DecodeWait::Spin};
    MaintainerOptions maintainer;

    void apply(EngineOptions& options) const {
        options.pacing     = pacing;
        options.maintainer = maintainer;
    }
};

TuneEngine tune_engine(const std::map<std::string, std::string>& flags) {
    TuneEngine engine;
    if (const auto it = flags.find("decode-wait"); it != flags.end()) {
        engine.pacing.wait = parse_decode_wait(it->second);
    }
    if (const auto it = flags.find("cuda-sync"); it != flags.end()) {
        engine.pacing.cuda_sync = parse_cuda_sync_schedule(it->second);
    }
    if (const auto it = flags.find("route-table"); it != flags.end()) {
        engine.maintainer.route_table = it->second;
    }
    if (const auto it = flags.find("route-dir"); it != flags.end()) {
        engine.maintainer.route_dir = it->second;
    }
    return engine;
}

// How a profile names an explicit route table: by its content ("sha256:<hex>"), never by a path
// local to the machine that measured it.
std::string route_table_label(const std::filesystem::path& table) {
    const auto digest = file_sha256_hex(table);
    return digest ? "sha256:" + *digest : std::string("unreadable");
}

// The tune engine's settings beside the wait, named only when given (an --out measured before
// they existed still resumes): an explicit route table by its content, a route directory as "set".
void name_tune_engine(nlohmann::json& settings, const TuneEngine& engine) {
    if (engine.pacing.cuda_sync != CudaSyncSchedule::Spin) {
        settings["cuda_sync"] = cuda_sync_schedule_name(engine.pacing.cuda_sync);
    }
    if (!engine.maintainer.route_table.empty()) {
        settings["route_table"] = route_table_label(engine.maintainer.route_table);
    }
    if (!engine.maintainer.route_dir.empty()) { settings["route_dir"] = "set"; }
}

// The artifact's id, or its path's text when it has none readable (a drafter directory).
std::string artifact_id_or_path(const std::filesystem::path& artifact) {
    try {
        return spec::artifact_id_hex(artifact);
    } catch (const std::exception&) { return artifact.string(); }
}

// ninferno spec-tune <artifact> --load 2,4,8 [--load-depths 1,2,3,5] [--temperature 1.0]
//   [--top-p 0.95] [--thinking on|off|corpus] [--max-tokens N] [--repeat R] [--min-gain 0.03]
//   [--tolerance 0.03] [--draft-width block|drafts]
//   [--max-context N] [--prefill-chunk N] [--kv-dtype bf16] [--corpus FILE] [--out FILE.json]
//   [--resume FILE.json] [--profile TUNE.json (default: the tune beside the artifact)]
//   [--publish FILE --gpu-class CLASS --applies-to REPO,... --requires a,b --measured-on TEXT]
//   [--device N]
// Engines as `ninferno-serve --spec auto` builds them for this GPU (the profile's drafter at its
// default), one per candidate bound (0: plain decode, each of
// --load-depths below the engine's window, and the window itself: no bound), each bounded there
// under load (spec::load_cell_table) with the served menu's depths and its own bound as its
// launch windows, so it captures what a server with that bound does rather than every bound's
// windows at every batch (an engine capturing them all did not fit Gemma 4 31B at C=8 on an RTX
// 4090, nor at C=4 on an RTX 5080). Each engine loads the model again (one load per bound, and
// per narrower engine a concurrency needs). An engine is as wide as the widest concurrency; one
// its KV pool does not hold, or that does not fit, is tried on an engine sized for it
// (spec::measure_load_cells). Every cell runs one prefill chunk: --prefill-chunk, else the
// automatic one, which each engine may narrow from the chunk so far (as ninferno-serve narrows it
// to fit); when one narrows it, the cells measured at the wider chunk are measured again at the
// narrower. --out is rewritten after every cell (a temporary file
// renamed over it) with the settings the cells were measured with, and --resume takes an earlier
// --out's cells when every setting is the same and measures only the rest. For each
// concurrency C, every candidate bound runs the corpus with C requests in flight, each request
// routed as the server routes it (the profile's default for every request; the prompt's thinking
// flag on, unless --thinking off or corpus: each prompt's own flag) and capped at the bound. Per C
// the shallowest bound within --tolerance of the fastest wins (a deeper bound that measures the
// same only risks the depths this mix did not exercise, and served answers, longer than the
// corpus's, accept less: Qwen3.6 35B-A3B at C=8 on an RTX 5090 measured bounds 3, 5 and 7
// within 1.3 % here, while ninferno-serve gained over plain decode at 5 and lost with rows at 7
// and 11), speculation only when it beats plain decode by --min-gain (spec::decide_load); a C
// without plain decode, or without any speculative bound, measured is not decided, and the table
// says it is partial ("load_partial"). Bounds apply to steps of at least the lanes they are
// measured at; between two measured concurrencies a step takes the higher one's (lanes = the
// lower + 1), so a step never speculates deeper than measured for its load, and above the
// widest decided C, when a wider one was not decided, no deeper than the shallowest decided
// (spec::decided_load_table). One request is never bounded. --draft-width runs every cell's DFlash
// rounds at that width (SpeculativeOptions::draft_width; refused when the served speculation does
// not run it, product::fit_draft_widths) and pins it on the table's speculating bounds, so a
// server runs the width the bounds were measured at; without it every launch drafts at the
// Engine's default and the table names none.
int run_spec_tune_load(const std::string& artifact, const std::map<std::string, std::string>& flags,
                       const TuneEngine& engine) {
    const std::string decode_wait = decode_wait_name(engine.pacing.wait);
    const auto text = [&](const std::string& key, std::string fallback) {
        const auto it = flags.find(key);
        return it == flags.end() ? fallback : it->second;
    };
    const auto numbers = [&](const std::string& key, std::string fallback) {
        std::vector<std::uint32_t> out;
        for (const auto& item : option_list(text(key, fallback), ',')) {
            out.push_back(parse_u32(item.c_str(), key, /*allow_zero=*/true));
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };
    const int device        = parse_nonnegative_int(text("device", "0").c_str(), "device");
    const auto concurrency  = numbers("load", "2,4,8");
    const auto wanted       = numbers("load-depths", "1,2,3,5");
    const float temperature = parse_float_in(text("temperature", "1.0").c_str(), "temperature",
                                             static_cast<float>(kTemperatureRange.low),
                                             static_cast<float>(kTemperatureRange.high));
    const float top_p       = parse_float_in(text("top-p", "0.95").c_str(), "top-p", 0.0F, 1.0F);
    const int repeat        = static_cast<int>(parse_u32(text("repeat", "1").c_str(), "repeat"));
    const double min_gain =
        parse_float_in(text("min-gain", "0.03").c_str(), "min-gain", 0.0F, 1.0F);
    const double tolerance =
        parse_float_in(text("tolerance", "0.03").c_str(), "tolerance", 0.0F, 1.0F);
    if (!(tolerance >= 0 && tolerance < 1) || !(min_gain >= 0)) {
        throw std::invalid_argument("spec-tune --load: --tolerance in [0,1), --min-gain >= 0");
    }
    const std::string thinking_mode = text("thinking", "on");
    if (thinking_mode != "on" && thinking_mode != "off" && thinking_mode != "corpus") {
        throw std::invalid_argument("spec-tune --load: --thinking on|off|corpus");
    }
    const auto max_tokens = parse_u32(text("max-tokens", "0").c_str(), "max-tokens", true);
    if (concurrency.empty() || concurrency.front() < 2 || concurrency.back() > 8) {
        throw std::invalid_argument("spec-tune --load: concurrencies in [2,8]");
    }
    std::optional<spec::Corpus> own;
    if (flags.contains("corpus")) { own = spec::load_corpus(flags.at("corpus")); }
    const auto& corpus = own ? *own : spec::builtin_corpus();
    const auto gpu     = spec::gpu_identity(device);
    const auto started = std::chrono::steady_clock::now();

    // As ninferno-serve builds it by default (--spec-select fixed): every request at the default.
    // The lab flow: the tune named by --profile (one that does not serve refuses), else the one
    // spec-tune wrote beside the artifact when it would serve (a stale one is passed over with a
    // note), else the published profile.
    std::filesystem::path tuned;
    std::error_code beside_error; // unreadable status: the published profile runs
    if (flags.contains("profile")) {
        tuned = flags.at("profile");
    } else if (const auto beside = spec::default_profile_path(artifact);
               std::filesystem::exists(beside, beside_error)) {
        if (const auto why = product::tune_refusal(artifact, device, beside); why.empty()) {
            tuned = beside;
        } else {
            std::fprintf(stderr,
                         "spec-tune --load: the tune beside the model is stale (%s); the "
                         "published profile runs\n",
                         why.c_str());
        }
    }
    auto served = product::served_speculation(artifact, device, true, false, tuned);
    if (!served || served->menu.mode == spec::Mode::None) {
        throw std::runtime_error("spec-tune --load: the model has no speculation that runs on this "
                                 "GPU (serve --spec auto would decode plainly)");
    }
    SpeculativeOptions speculative = served->engine;
    if (speculative.drafter.empty()) {
        speculative.drafter = product::model_drafter(artifact, speculative.backend);
    }
    speculative.load.clear();
    if (flags.contains("draft-width")) {
        speculative.draft_width = product::parse_draft_width(flags.at("draft-width"));
        const auto dropped =
            product::fit_draft_widths(speculative, product::engine_support(artifact));
        if (!dropped.empty()) { throw std::invalid_argument("spec-tune --load: " + dropped[0]); }
    }
    spec::DraftWidth width = spec::DraftWidth::Default;
    if (speculative.draft_width == DFlashDraftWidth::Block) { width = spec::DraftWidth::Block; }
    if (speculative.draft_width == DFlashDraftWidth::Drafts) { width = spec::DraftWidth::Drafts; }
    const std::uint32_t window = product::speculative_window(speculative);
    // Deepest first (no bound, then the depths descending, plain decode last).
    std::vector<std::uint32_t> bounds{window};
    for (auto d = wanted.rbegin(); d != wanted.rend(); ++d) {
        if (*d >= 1 && *d < window) { bounds.push_back(*d); }
    }
    bounds.push_back(0);
    const auto name = [&](std::uint32_t bound) { return spec::load_bound_name(bound, window); };

    const auto max_context     = parse_u32(text("max-context", "4096").c_str(), "max-context");
    const std::string kv_dtype = text("kv-dtype", "bf16");
    // What the cells are measured with: --resume takes an --out's cells only when every field is
    // the same (the concurrencies, --tolerance and --min-gain only decide over the cells).
    nlohmann::json bound_names = nlohmann::json::array();
    for (const std::uint32_t b : bounds) { bound_names.push_back(name(b)); }
    nlohmann::json settings{
        {"gpu", gpu.id},
        {"gpu_name", gpu.name},
        {"artifact", artifact},
        {"artifact_id", artifact_id_or_path(artifact)},
        {"drafter", speculative.drafter.string()},
        {"drafter_id",
         speculative.drafter.empty() ? std::string() : artifact_id_or_path(speculative.drafter)},
        {"source", served->source},
        {"default", served->menu.primary.config.label()},
        {"window", window},
        {"bounds", bound_names},
        {"temperature", text("temperature", "1.0")},
        {"top_p", text("top-p", "0.95")},
        {"thinking", thinking_mode},
        {"corpus", own ? flags.at("corpus") : std::string("built-in")},
        {"corpus_sha256", corpus_digest(corpus)},
        {"repeat", repeat},
        {"max_tokens", max_tokens},
        {"max_context", max_context},
        {"kv_dtype", kv_dtype},
        {"prefill_chunk", text("prefill-chunk", "auto")},
        {"load_depths", text("load-depths", "1,2,3,5")},
        {"decode_wait", decode_wait}};
    // Only when named, so an --out measured before widths existed still resumes.
    if (width != spec::DraftWidth::Default) {
        settings["draft_width"] = std::string(spec::draft_width_name(width));
    }
    name_tune_engine(settings, engine);

    // The cells measured so far (--resume: those of an earlier --out measured with the same
    // settings), and the ones no engine holds. --out is rewritten after every cell, so a tune
    // stopped part way keeps its cells and a later run with --resume measures only the rest.
    spec::LoadCells state;
    state.settings = settings;
    state.window   = window;
    if (flags.contains("resume")) {
        const std::string& path = flags.at("resume");
        std::ifstream in(path);
        if (!in) { throw std::runtime_error("spec-tune --load: cannot read --resume " + path); }
        nlohmann::json previous;
        try {
            previous = nlohmann::json::parse(in);
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error("spec-tune --load: --resume " + path + " is not JSON: " +
                                     error.what());
        }
        state = spec::load_cells_from_json(previous, settings);
        if (state.window != window) {
            throw std::runtime_error("spec-tune --load: --resume " + path + " has window " +
                                     std::to_string(state.window) + ", this run " +
                                     std::to_string(window));
        }
        std::fprintf(stderr, "spec-tune --load: %zu cells from %s (prefill chunk %u)\n",
                     state.cells.size(), path.c_str(), state.prefill_chunk);
    }
    const bool chunk_fixed = flags.contains("prefill-chunk");
    if (chunk_fixed) {
        state.prefill_chunk = parse_u32(flags.at("prefill-chunk").c_str(), "prefill-chunk");
    }

    // A cell engine: `bound` under load (spec::load_cell_table), up to `widest` rows, its launch
    // windows the served menu's depths and its own bound. Sized as ninferno-serve sizes it
    // (--max-context with several slots and no --kv-capacity): an automatic KV capacity, every
    // slot's full context when that fits, else the most that does (at least one context). Without
    // --prefill-chunk the chunk is automatic from the tune's chunk so far (1024 at first): the
    // engine narrows it as far as its plan needs (down to 128), never widens it.
    const auto cell_options = [&](std::uint32_t widest, std::uint32_t bound) {
        const std::uint32_t chunk = state.prefill_chunk == 0 ? 1024U : state.prefill_chunk;

        EngineOptions options;
        options.artifact_path                        = artifact;
        engine.apply(options);
        options.device                               = device;
        options.max_context                          = max_context;
        options.max_concurrency                      = widest;
        options.kv_capacity                          = KvCapacityPolicy::automatic();
        options.prefill_chunk                        = chunk;
        options.prefill_chunk_automatic              = !chunk_fixed;
        options.kv_cache                             = kv_storage(kv_dtype);
        options.context_cache.disable();
        options.speculative                          = speculative;
        // The bound is a launch window from 2 lanes on (the table's own); not one of a single row,
        // which a server with this table does not capture either.
        for (const spec::LoadBound& b : spec::load_cell_table(static_cast<std::int32_t>(bound),
                                                              static_cast<std::int32_t>(window))) {
            options.speculative.load.push_back(
                {.lanes        = static_cast<std::uint32_t>(b.lanes),
                 .draft_tokens = static_cast<std::uint32_t>(b.depth)});
        }
        return options;
    };
    std::fprintf(stderr, "spec-tune --load: %s | %s | window %u | default %s | bounds",
                 served->source.c_str(), product::speculative_backend_name(speculative.backend),
                 window, served->menu.primary.config.label().c_str());
    for (const std::uint32_t b : bounds) { std::fprintf(stderr, " %s", name(b).c_str()); }
    std::fprintf(stderr, "\n");

    // The corpus in the server's routing, each prompt with its speculation.
    struct Routed {
        PromptInput input;
        RequestSpeculation speculation;
        std::uint32_t max_tokens = 0;
    };
    std::vector<Routed> routed;
    for (int r = 0; r < repeat; ++r) {
        for (const auto& p : corpus.prompts) {
            const bool thinking = thinking_mode == "corpus" ? p.thinking : thinking_mode == "on";
            Routed item{user_prompt(p), {},
                        max_tokens != 0 ? max_tokens : static_cast<std::uint32_t>(p.max_tokens)};
            item.input.options.enable_thinking = thinking;
            item.speculation = product::request_speculation(served->menu, served->menu.primary);
            routed.push_back(std::move(item));
        }
    }
    // A cell runs C wide only when its engine's KV pool holds C requests at once; a cell the pool
    // held to fewer lanes would publish a bound measured at a lower load. The largest request's
    // reservation sets the widest concurrency a pool holds, counted in whole KV pages as the
    // engine reserves it (request_plan.cpp): its prompt and outputs less one, the outputs capped by
    // the context, and for MTP the drafter's KV of window - 1 more (counted whether or not the
    // drafter keeps its own KV; Gemma 4's assistant does not).
    constexpr auto kPage = static_cast<std::uint64_t>(kPagedKVPageSize);
    const bool mtp       = speculative.backend == SpeculativeBackend::Mtp;
    std::optional<std::uint64_t> largest_request;
    const auto largest = [&](const Engine& engine) {
        if (!largest_request) {
            std::uint64_t out = 0;
            for (const Routed& item : routed) {
                const std::uint64_t prompt =
                    std::min<std::uint64_t>(engine.count_tokens(item.input), max_context);
                const std::uint64_t outputs =
                    std::min<std::uint64_t>(item.max_tokens, max_context - prompt + 1U);
                const std::uint64_t reserved = prompt + (outputs == 0 ? 0U : outputs - 1U);
                const std::uint64_t tokens =
                    mtp ? std::min<std::uint64_t>(reserved + window - 1U, max_context) : reserved;
                out = std::max(out, (tokens + kPage - 1U) / kPage * kPage);
            }
            largest_request = out;
        }
        return *largest_request;
    };

    // One cell: `c` requests in flight over the first `limit` prompts (0: all of them).
    const auto run_cell = [&](Engine& engine, std::uint32_t c, std::uint32_t bound,
                              std::size_t limit) {
        const std::size_t count = limit == 0 ? routed.size() : std::min(limit, routed.size());
        spec::LoadCell cell{.concurrency = c, .bound = bound};
        std::atomic<std::size_t> next{0};
        std::mutex mutex;
        std::string failure;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> lanes;
        for (std::uint32_t lane = 0; lane < c; ++lane) {
            lanes.emplace_back([&] {
                for (std::size_t i = next++; i < count; i = next++) {
                    try {
                        RequestOptions request;
                        request.execution.requested_output_tokens = routed[i].max_tokens;
                        request.execution.allow_prefix_reuse     = false;
                        request.execution.sampling.temperature   = temperature;
                        request.execution.sampling.top_p         = top_p;
                        request.execution.sampling.seed          = i + 1;
                        request.execution.speculation = bounded(routed[i].speculation, bound);
                        auto handle =
                            engine.submit(engine.prepare(routed[i].input), std::move(request));
                        const auto result = handle.wait();
                        const std::lock_guard lock(mutex);
                        cell.tokens += static_cast<std::int64_t>(result.generated_token_ids.size());
                        cell.rounds += result.speculative.rounds;
                        cell.drafted += result.speculative.drafted_tokens;
                        cell.accepted += result.speculative.accepted_tokens;
                        cell.plain_steps += result.speculative.plain_steps;
                        cell.trips += result.speculative.watchdog_tripped ? 1U : 0U;
                    } catch (const std::exception& error) {
                        const std::lock_guard lock(mutex);
                        failure = error.what();
                        next    = count;
                    }
                }
            });
        }
        for (auto& lane : lanes) { lane.join(); }
        if (!failure.empty()) { throw std::runtime_error(failure); }
        cell.seconds = ninferno::seconds_since(t0);
        return cell;
    };

    const auto write_out = [&](const std::string& load, const std::string& measured_on,
                               bool partial) {
        if (!flags.contains("out")) { return; }
        nlohmann::json doc  = spec::load_cells_json(state, load, measured_on);
        doc["load_partial"] = partial;
        spec::write_json_file(flags.at("out"), doc);
    };
    const auto find_cell = [&](std::uint32_t c, std::uint32_t bound) -> const spec::LoadCell* {
        for (const auto& cell : state.cells) {
            if (cell.concurrency == c && cell.bound == bound) { return &cell; }
        }
        return nullptr;
    };
    const auto mark_unheld = [&](std::uint32_t c, std::uint32_t bound, const std::string& why) {
        state.unheld.push_back({.concurrency = c, .bound = bound});
        std::fprintf(stderr, "spec-tune --load C=%u bound %s: %s\n", c, name(bound).c_str(),
                     why.c_str());
    };

    // One engine per bound, as wide as the widest concurrency its pool holds; the concurrencies
    // left over get an engine sized for them (spec::measure_load_cells). Each engine captures what
    // a server bounded there does, not every candidate bound's windows at every batch.
    bool narrowed            = false; // an engine narrowed the prefill chunk: another pass
    const auto measure_bound = [&](std::uint32_t bound) {
        std::vector<std::uint32_t> asked_for;
        for (const std::uint32_t c : concurrency) {
            if (find_cell(c, bound) == nullptr) { asked_for.push_back(c); }
        }
        if (asked_for.empty()) { return; }
        const auto left = spec::measure_load_cells(
            asked_for, [&](std::uint32_t widest, const std::vector<std::uint32_t>& asked) {
                std::optional<Engine> engine;
                try {
                    engine.emplace(cell_options(widest, bound));
                } catch (const CapacityError& refused) {
                    std::fprintf(stderr,
                                 "spec-tune --load bound %s at C=%u does not fit (prefill chunk "
                                 "%u%s): %s\n",
                                 name(bound).c_str(), widest,
                                 state.prefill_chunk == 0 ? 1024U : state.prefill_chunk,
                                 chunk_fixed ? "" : ", automatic: down to 128", refused.what());
                    return std::vector<std::uint32_t>{};
                } catch (const std::exception& error) {
                    // Not a memory shortage: a CUDA failure can leave the device unusable for the
                    // next, narrower engine (sc_v7: a graph update failure, then an abort). Stop
                    // with it; --out keeps the cells measured so far for --resume.
                    throw std::runtime_error("spec-tune --load bound " + name(bound) + " at C=" +
                                             std::to_string(widest) +
                                             ": the engine failed to build: " + error.what());
                }
                const std::uint32_t chunk = engine->memory_summary().prefill_chunk;
                if (state.prefill_chunk != 0 && chunk > state.prefill_chunk) {
                    std::fprintf(stderr,
                                 "spec-tune --load bound %s at C=%u runs prefill chunk %u, not the "
                                 "tune's %u\n",
                                 name(bound).c_str(), widest, chunk, state.prefill_chunk);
                    return std::vector<std::uint32_t>{};
                }
                if (state.prefill_chunk != 0 && chunk < state.prefill_chunk) {
                    // Every cell runs one chunk: those measured at the wider one run again.
                    std::fprintf(stderr,
                                 "spec-tune --load bound %s at C=%u narrows the prefill chunk from "
                                 "%u to %u: the %zu cells measured at %u are measured again\n",
                                 name(bound).c_str(), widest, state.prefill_chunk, chunk,
                                 state.cells.size(), state.prefill_chunk);
                    state.cells.clear();
                    narrowed = true;
                }
                state.prefill_chunk = chunk;
                std::size_t free_bytes = 0, total_bytes = 0;
                if (cudaSetDevice(device) != cudaSuccess ||
                    cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                    free_bytes = total_bytes = 0;
                }
                const std::uint64_t pool = engine->memory_summary().kv_capacity;
                std::vector<std::uint32_t> held;
                for (const std::uint32_t c : asked) {
                    if (pool >= c * largest(*engine)) { held.push_back(c); }
                }
                std::fprintf(stderr,
                             "spec-tune --load bound %s engine: up to %u rows | chunk %u | KV %llu "
                             "tokens, %llu requests of up to %llu at once | free %.0f of %.0f "
                             "MiB\n",
                             name(bound).c_str(), widest, chunk,
                             static_cast<unsigned long long>(pool),
                             static_cast<unsigned long long>(pool / largest(*engine)),
                             static_cast<unsigned long long>(largest(*engine)),
                             free_bytes / 1048576.0, total_bytes / 1048576.0);
                if (held.empty()) { return held; }
                // Warm the routes and graphs: two prompts per lane at the widest held. An engine
                // that fails it holds none of them (a narrower one is tried).
                try {
                    (void)run_cell(*engine, held.back(), bound, std::size_t{2} * held.back());
                } catch (const std::exception& error) {
                    std::fprintf(stderr, "spec-tune --load bound %s at C=%u: warm-up failed: %s\n",
                                 name(bound).c_str(), held.back(), error.what());
                    return std::vector<std::uint32_t>{};
                }
                // A cell that fails is settled as unheld; the rest still run.
                for (const std::uint32_t c : held) {
                    try {
                        const spec::LoadCell cell = run_cell(*engine, c, bound, 0);
                        state.cells.push_back(cell);
                        std::fprintf(stderr,
                                     "spec-tune --load C=%u bound %-5s %8.1f tok/s  %5.2f "
                                     "tok/round  plain steps %llu  trips %llu\n",
                                     c, name(bound).c_str(), cell.tok_s(),
                                     cell.rounds ? double(cell.accepted + cell.rounds) /
                                                       double(cell.rounds)
                                                 : 1.0,
                                     static_cast<unsigned long long>(cell.plain_steps),
                                     static_cast<unsigned long long>(cell.trips));
                    } catch (const std::exception& error) {
                        mark_unheld(c, bound, std::string("failed: ") + error.what());
                    }
                    write_out("", "", true);
                }
                return held;
            });
        for (const std::uint32_t c : left) { mark_unheld(c, bound, "no engine holds it"); }
        write_out("", "", true);
    };
    // A pass over the bounds, and another for the cells dropped when an engine narrowed the
    // prefill chunk (at most one per halving of it); the cells no engine held are tried again.
    do {
        narrowed = false;
        state.unheld.clear();
        for (const std::uint32_t bound : bounds) { measure_bound(bound); }
    } while (narrowed);

    spec::LoadDecision decision =
        spec::decide_load(state.cells, concurrency, bounds, tolerance, min_gain);
    for (spec::LoadChoice& choice : decision.choices) {
        if (choice.depth != 0) { choice.width = width; }
    }
    for (const spec::LoadChoice& choice : decision.choices) {
        const auto c              = static_cast<std::uint32_t>(choice.concurrency);
        const spec::LoadCell* got = find_cell(c, static_cast<std::uint32_t>(choice.depth));
        std::fprintf(stderr, "spec-tune --load C=%u -> %s (x%.3f over plain)\n", c,
                     name(static_cast<std::uint32_t>(choice.depth)).c_str(),
                     choice.depth != 0 ? got->tok_s() / find_cell(c, 0)->tok_s() : 1.0);
    }
    for (const std::uint32_t c : decision.undecided) {
        std::fprintf(stderr,
                     "spec-tune --load C=%u not decided: plain decode or every speculative bound "
                     "not measured\n",
                     c);
    }
    if (decision.choices.empty()) {
        throw std::runtime_error("spec-tune --load: no concurrency measured with plain decode and "
                                 "a speculative bound");
    }
    const std::vector<spec::LoadBound> merged =
        spec::decided_load_table(decision, static_cast<std::int32_t>(window));
    const bool partial = !decision.undecided.empty();

    std::string widths_text;
    for (const spec::LoadChoice& choice : decision.choices) {
        widths_text += (widths_text.empty() ? "" : ",") + std::to_string(choice.concurrency);
    }
    // A concurrency asked for and not decided makes the table partial ("load_partial"): its steps
    // take the next decided concurrency's bound, and above the widest decided one no deeper than
    // the shallowest decided (spec::decided_load_table).
    std::string undecided;
    for (const std::uint32_t c : decision.undecided) {
        undecided += (undecided.empty() ? "" : ",") + std::to_string(c);
    }
    if (partial) { widths_text += " (partial: C " + undecided + " not measured)"; }
    const std::string measured_on =
        text("measured-on", std::filesystem::path(artifact).filename().string()) + ", " + gpu.name +
        ", C " + widths_text + ", temperature " + text("temperature", "1.0") + ", thinking " +
        thinking_mode + ", " + (own ? flags.at("corpus") : "built-in corpus") + " x" +
        std::to_string(repeat) +
        (width != spec::DraftWidth::Default
             ? ", draft width " + std::string(spec::draft_width_name(width))
             : std::string()) +
        ", wait " + decode_wait + ", " + ninferno::utc_now();
    write_out(spec::load_label(merged), measured_on, partial);
    if (flags.contains("publish")) {
        // What tools/spec/registry_add.py --load merges into the profile with the same GPU class,
        // applies_to and requires.
        nlohmann::json published{{"gpu_class", text("gpu-class", gpu.chip)},
                                 {"gpu", gpu.name},
                                 {"applies_to", option_list(text("applies-to", ""), ',')},
                                 {"load", nlohmann::json::array()},
                                 {"load_measured_on", measured_on},
                                 {"load_partial", partial}};
        std::vector<std::string> required;
        if (flags.contains("requires")) { required = option_list(flags.at("requires"), ','); }
        // The bounds of a drafter variant's profile ("<name>.<variant>", which requires it:
        // spec::published_from_profile) go to that profile, not to the published drafter's.
        if (const auto m = spec::read_manifest(spec::manifest_path(artifact));
            m && !speculative.drafter.empty()) {
            for (const auto& d : m->drafters) {
                if (!d.artifact.empty() && d.artifact == speculative.drafter.string() &&
                    !spec::split_variant(d.name).second.empty() &&
                    std::find(required.begin(), required.end(), d.name) == required.end()) {
                    required.push_back(d.name);
                }
            }
        }
        if (!required.empty()) { published["requires"] = required; }
        for (const auto& bound : merged) {
            nlohmann::json entry{{"lanes", bound.lanes}, {"depth", bound.depth}};
            if (bound.width != spec::DraftWidth::Default) {
                entry["draft_width"] = std::string(spec::draft_width_name(bound.width));
            }
            published["load"].push_back(std::move(entry));
        }
        spec::write_json_file(flags.at("publish"), published);
    }
    std::printf("{\"load\":\"%s\",\"window\":%u,\"partial\":%s,\"seconds\":%.1f,"
                "\"gpu_class\":\"%s\"}\n",
                spec::load_label(merged).c_str(), window, partial ? "true" : "false",
                ninferno::seconds_since(started), gpu.id.c_str());
    return 0;
}

} // namespace

int run_spec_tune(int argc, char** argv) {
    // ninferno spec-tune <artifact> [--kv-dtype bf16] [--max-context N] [--prefill-chunk N]
    //   [--configs LIST] [--budget-seconds S] [--repeat R] [--out PROFILE] [--corpus FILE]
    //   [--publish FILE --applies-to REPO,... --measured-on TEXT --gpu-class CLASS] [--device N]
    if (argc < 3) {
        throw std::invalid_argument("usage: ninferno spec-tune <model.ninferno> [...]");
    }
    const std::string artifact = argv[2];
    std::map<std::string, std::string> flags;
    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (!key.starts_with("--") || i + 1 >= argc) {
            throw std::invalid_argument("spec-tune: expected --key value, got " + key);
        }
        flags[key.substr(2)] = argv[++i];
    }
    static const std::set<std::string, std::less<>> kKnown = {
        "applies-to",  "budget-seconds", "configs",     "corpus",           "cuda-sync",
        "decode-wait", "device",         "draft-width", "drafter-registry", "gpu-class",
        "kv-dtype",    "load",           "load-depths", "max-context",      "max-tokens",
        "measured-on", "min-gain",       "out",         "prefill-chunk",    "profile",
        "publish",     "repeat",         "requires",    "resume",           "route-dir",
        "route-table", "temperature",    "thinking",    "tolerance",        "top-p"};
    for (const auto& [key, value] : flags) {
        if (!kKnown.contains(key)) {
            throw std::invalid_argument("spec-tune: unknown flag --" + key);
        }
    }
    const auto text = [&](const std::string& key, std::string fallback) {
        const auto it = flags.find(key);
        return it == flags.end() ? fallback : it->second;
    };
    // --drafter-registry: a registry file in place of the compiled-in one, as ninferno takes it.
    if (const auto it = flags.find("drafter-registry"); it != flags.end()) {
        spec::Registry::use_file(it->second);
    }
    const TuneEngine engine       = tune_engine(flags);
    const std::string decode_wait = decode_wait_name(engine.pacing.wait);
    if (flags.contains("load")) { return run_spec_tune_load(artifact, flags, engine); }
    const auto started = std::chrono::steady_clock::now();
    const int device   = parse_nonnegative_int(text("device", "0").c_str(), "device");
    std::optional<spec::Corpus> own;
    if (flags.contains("corpus")) { own = spec::load_corpus(flags["corpus"]); }
    const auto& corpus = own ? *own : spec::builtin_corpus();

    // The backends the artifact carries.
    std::set<std::string> components;
    {
        const artifact::Reader reader(artifact);
        for (const auto& [name, component] : reader.directory().components) {
            components.insert(name);
        }
    }
    // The drafters that came beside the target (gpt-oss's DFlash, the NemotronH MTP head), by the
    // mode their configurations use: a built-in drafter of the same mode wins.
    const auto manifest = spec::read_manifest(spec::manifest_path(artifact));
    std::map<std::string, spec::ManifestDrafter> separate;
    for (const spec::Mode mode : {spec::Mode::Mtp, spec::Mode::DFlash, spec::Mode::DFlash2}) {
        const std::string name = spec::mode_name(mode);
        const auto path        = product::model_drafter(artifact, product::backend_of(mode));
        if (components.contains(name) || path.empty() || !manifest) { continue; }
        for (const auto& d : manifest->drafters) {
            if (std::filesystem::path(d.artifact) == path) { separate.emplace(name, d); }
        }
    }
    std::vector<std::string> labels;
    if (flags.contains("configs")) {
        labels = option_list(flags["configs"], ',');
    } else {
        for (const char* l : {"mtp:2+head", "mtp:3+head", "mtp:4+head", "mtp:5+head",
                              "dflash:3+head", "dflash:5+head", "dflash:7+head", "dflash:11+head",
                              "dflash2:5+head", "dflash2:7+head", "dflash2:11+head",
                              "dflash2:15+head"}) {
            labels.emplace_back(l);
        }
    }
    // The default menu drafts through the optimized proposal head where the artifact has one (a
    // GGUF conversion has none): without it, the same depths with the full head.
    const bool head = flags.contains("configs") || product::has_proposal_head(artifact);
    std::vector<spec::SpecConfig> configs{spec::SpecConfig{}};
    for (const auto& label : labels) {
        auto c = spec::SpecConfig::parse(label);
        c.proposal_head = c.proposal_head && head;
        const std::string mode = spec::mode_name(c.mode);
        if (c.enabled() && (components.contains(mode) || separate.contains(mode))) {
            configs.push_back(c);
        }
    }
    const double budget =
        parse_float_in(text("budget-seconds", "900").c_str(), "budget-seconds", 0.0F, 1.0e7F);
    const int repeat = static_cast<int>(parse_u32(text("repeat", "1").c_str(), "repeat"));
    const auto gpu = spec::gpu_identity(device);

    spec::Profile profile;
    profile.artifact        = artifact;
    profile.created         = ninferno::utc_now();
    profile.key.artifact_id = spec::artifact_id_hex(artifact);
    profile.key.gpu         = gpu.id;
    std::vector<std::string> ids; // a built-in drafter has no artifact of its own
    for (const char* name : {"mtp", "dflash", "dflash2"}) {
        if (components.contains(name)) {
            profile.drafters.push_back({name, name, name, "", "", "", "", "", 0, 0});
        } else if (const auto it = separate.find(name); it != separate.end()) {
            const auto& d = it->second;
            profile.drafters.push_back({d.name, d.kind, name, d.artifact,
                                        spec::artifact_id_hex(d.artifact), d.repo, "", "", 0, 0});
            ids.push_back(profile.drafters.back().artifact_id);
        }
    }
    profile.key.drafters   = spec::drafter_key(ids);
    const auto max_context = parse_u32(text("max-context", "4096").c_str(), "max-context");
    char settings[160];
    std::snprintf(settings, sizeof(settings),
                  "max context %u, prefill chunk %s, KV %s, greedy, wait %s", max_context,
                  text("prefill-chunk", "1024").c_str(), text("kv-dtype", "bf16").c_str(),
                  decode_wait.c_str());
    profile.settings = settings;
    if (engine.pacing.cuda_sync != CudaSyncSchedule::Spin) {
        profile.settings +=
            std::string(", CUDA sync ") + cuda_sync_schedule_name(engine.pacing.cuda_sync);
    }
    if (!engine.maintainer.route_table.empty()) {
        profile.settings += ", route table " + route_table_label(engine.maintainer.route_table);
    }
    if (!engine.maintainer.route_dir.empty()) { profile.settings += ", route dir set"; }

    std::map<std::string, std::vector<TokenId>> reference;
    for (const auto& config : configs) {
        if (config.enabled() && ninferno::seconds_since(started) > budget) {
            profile.skipped.push_back(config.label());
            continue;
        }
        EngineOptions options;
        options.artifact_path  = artifact;
        engine.apply(options);
        options.device         = device;
        options.max_context    = max_context;
        options.kv_capacity    = KvCapacityPolicy::explicit_capacity(max_context);
        options.prefill_chunk  = parse_u32(text("prefill-chunk", "1024").c_str(), "prefill-chunk");
        // The KV storage serve runs (EngineOptions: BF16) unless asked; the dense family and
        // nemotron_h keep BF16 only.
        options.kv_cache       = kv_storage(text("kv-dtype", "bf16"));
        options.context_cache.disable();
        if (config.enabled()) {
            options.speculative.backend       = product::backend_of(config.mode);
            options.speculative.draft_tokens  = static_cast<std::uint32_t>(config.k);
            options.speculative.proposal_head =
                config.proposal_head ? ProposalHead::Optimized : ProposalHead::Full;
            if (const auto it = separate.find(spec::mode_name(config.mode)); it != separate.end()) {
                options.speculative.drafter = it->second.artifact;
            }
        }
        std::optional<Engine> engine;
        try {
            engine.emplace(std::move(options));
        } catch (const std::exception& error) {
            std::fprintf(stderr, "spec-tune %s: engine refused: %s\n", config.label().c_str(),
                         error.what());
            profile.skipped.push_back(config.label() + " (" + error.what() + ")");
            continue;
        }
        const auto run = [&](const spec::CorpusPrompt& p) {
            auto prepared = engine->prepare(user_prompt(p));
            return engine->generate(std::move(prepared),
                                    greedy(static_cast<std::uint32_t>(p.max_tokens)));
        };
        (void)run(corpus.prompts.front()); // warm routes and graphs
        for (const char* family : spec::kFamilies) {
            spec::Measurement m;
            m.config = config;
            m.family = family;
            for (const auto& p : corpus.prompts) {
                if (p.family != family) { continue; }
                for (int r = 0; r < repeat; ++r) {
                    const auto result = run(p);
                    const auto& ids   = result.generated_token_ids;
                    // The first token comes from the prefill; decode time covers the rest.
                    m.prompts += 1;
                    m.tokens += ids.empty() ? 0 : static_cast<std::int64_t>(ids.size() - 1);
                    m.seconds += result.timings.decode_seconds;
                    m.rounds += static_cast<std::int64_t>(result.speculative.rounds);
                    m.drafted += static_cast<std::int64_t>(result.speculative.drafted_tokens);
                    m.accepted += static_cast<std::int64_t>(result.speculative.accepted_tokens);
                    if (!config.enabled() && r == 0) { reference[p.id] = ids; }
                    const auto& ref = reference[p.id];
                    const auto n    = std::min(ref.size(), ids.size());
                    std::size_t same = 0;
                    while (same < n && ref[same] == ids[same]) { ++same; }
                    m.compared += static_cast<std::int64_t>(n);
                    m.identical += static_cast<std::int64_t>(same);
                }
            }
            if (m.prompts == 0) { continue; }
            std::fprintf(stderr, "spec-tune %-15s %-8s %7.1f tok/s  %5.2f tok/round  acc %.3f  "
                                 "identical %lld/%lld\n",
                         config.label().c_str(), family, m.tok_s(), m.tokens_per_round(),
                         m.acceptance(), static_cast<long long>(m.identical),
                         static_cast<long long>(m.compared));
            profile.measurements.push_back(m);
        }
    }
    spec::select(profile);
    profile.tune_seconds = ninferno::seconds_since(started);
    const auto out_path  = flags.contains("out") ? std::filesystem::path(flags["out"])
                                                 : spec::default_profile_path(artifact);
    spec::write_profile(out_path, profile);
    if (flags.contains("publish")) {
        std::vector<std::string> applies = option_list(text("applies-to", ""), ',');
        // Published for the chip unless the maintainer names a variant ("gb202:170sm") or a
        // wider class.
        const auto published = spec::published_from_profile(
            profile, text("gpu-class", gpu.chip), gpu.name, applies,
            text("measured-on", std::filesystem::path(artifact).filename().string()));
        spec::write_json_file(flags["publish"], published);
    }
    const auto show = [](const char* what, const spec::Choice& c) {
        std::fprintf(stderr, "choice %-12s %-15s x%.3f  %.1f tok/s  floor %.2f tok/round\n", what,
                     c.config.label().c_str(), c.speedup(), c.tok_s, c.floor_tokens_per_round);
    };
    show("non_thinking", profile.non_thinking);
    show("thinking", profile.thinking);
    show("default", profile.overall);
    for (const auto& d : profile.drafters) {
        std::fprintf(stderr, "drafter %-8s %s %s\n", d.name.c_str(), d.status.c_str(),
                     d.reason.c_str());
    }
    std::printf("{\"profile\":\"%s\",\"tune_seconds\":%.1f,\"gpu_class\":\"%s\",\"default\":\"%s\","
                "\"skipped\":%zu}\n",
                out_path.c_str(), profile.tune_seconds, gpu.id.c_str(),
                profile.overall.config.label().c_str(), profile.skipped.size());
    return 0;
}

int run_spec_reselect(int argc, char** argv) {
    // ninferno spec-reselect <profile.json> --publish FILE --gpu-class CLASS --gpu NAME
    //   --applies-to REPO,... --measured-on TEXT [--engine MODEL_TYPE] [--out PROFILE]
    // Selects again over a saved tune's measurements (spec-tune --out, ninferno-gemma4 spec-tune
    // --out) with this build's spec::select, and publishes the result as spec-tune --publish
    // would have: a profile published before the selection learned something new (the portable
    // choices of the "*" fallback) is brought up to date without measuring again. --engine names
    // the serving Engine's model type (spec/engine_support.h: gemma4_text for a Gemma 4 tune,
    // whose Session runs more than the Engine) for the served and portable choices, needed when
    // the saved profile has served choices (spec::reselect); --gpu-class, --gpu, --applies-to
    // (empty: any variant) and --measured-on repeat the original publication's. --publish is
    // written first, then --out, each through a temporary file.
    if (argc < 3) {
        throw std::invalid_argument(
            "usage: ninferno spec-reselect <profile.json> [--key value ...]");
    }
    const std::set<std::string> known = {"publish",     "gpu-class", "gpu", "applies-to",
                                         "measured-on", "engine",    "out", "drafter-registry"};
    std::map<std::string, std::string> flags;
    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (!key.starts_with("--") || i + 1 >= argc) {
            throw std::invalid_argument("spec-reselect: expected --key value, got " + key);
        }
        if (!known.contains(key.substr(2))) {
            throw std::invalid_argument("spec-reselect: unknown option " + key);
        }
        flags[key.substr(2)] = argv[++i];
    }
    for (const char* needed : {"publish", "gpu-class", "gpu", "applies-to", "measured-on"}) {
        if (!flags.contains(needed)) {
            throw std::invalid_argument(std::string("spec-reselect: --") + needed + " is needed");
        }
    }
    if (const auto it = flags.find("drafter-registry"); it != flags.end()) {
        spec::Registry::use_file(it->second);
    }
    std::string error;
    std::optional<spec::Profile> profile = spec::read_profile(argv[2], &error);
    if (!profile) { throw std::runtime_error("spec-reselect: " + error); }
    spec::ReselectOptions options{.gpu_class   = flags["gpu-class"],
                                  .gpu_name    = flags["gpu"],
                                  .applies_to  = option_list(flags["applies-to"], ','),
                                  .measured_on = flags["measured-on"]};
    if (flags.contains("engine")) {
        options.served = spec::engine_support(flags["engine"]);
        if (!options.served->any()) {
            throw std::invalid_argument("spec-reselect: no serving Engine speculates for model "
                                        "type " + flags["engine"]);
        }
    }
    const nlohmann::json published = spec::reselect(*profile, options);
    spec::write_json_file(flags["publish"], published);
    if (flags.contains("out")) { spec::write_profile(flags["out"], *profile); }
    const auto show = [](const char* what, const spec::Choice& c) {
        std::fprintf(stderr, "%-24s %-12s %.1f tok/s  floor %.2f tok/round\n", what,
                     c.config.label().c_str(), c.tok_s, c.floor_tokens_per_round);
    };
    const spec::ServedChoices& chosen =
        profile->served ? *profile->served
                        : spec::ServedChoices{.overall      = profile->overall,
                                              .thinking     = profile->thinking,
                                              .non_thinking = profile->non_thinking};
    show("default", chosen.overall);
    show("non_thinking", chosen.non_thinking);
    show("thinking", chosen.thinking);
    if (profile->portable) { show("portable default", profile->portable->overall); }
    return 0;
}

} // namespace ninferno::cli
