#include "models/gemma4/session.h"

#include "models/common/token_ids.h"
#include "models/gemma4/mtp_round.h"
#include "models/gemma4/target.h"
#include "models/qwen3_5/execution/prefill_calls.h"
#include "models/speculation/decoder.h"
#include "models/speculation/dflash_round.h"
#include "models/speculation/driver.h"
#include "models/speculation/ngram_round.h"
#include "models/speculation/session_steps.h"

#include "core/decode_graph.h"
#include "core/nvtx.h"
#include "core/units.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/position.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/target_logprobs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::gemma4 {
namespace {

// Appends the token this step consumes to the output list. A step's input is the previous draw
// (the prefill's, for the first step), so decode(n) returns exactly the n tokens it fed.
__global__ void record_input(const std::int32_t* token, std::int32_t* out, std::int32_t* step) {
    out[*step] = *token;
    *step += 1;
}

constexpr std::int32_t kMaxDecode = 1 << 16;

// Rejects the options no allocation could be sized from, before any is made.
const SessionOptions& checked(const SessionOptions& options) {
    if (options.prefill_chunk <= 0 || options.max_context <= 0) {
        throw std::invalid_argument("gemma4 session: chunk and context must be positive");
    }
    if (options.score_columns < kMaxVerifyColumns) {
        throw std::invalid_argument("gemma4 session: score_columns below the widest verify (" +
                                    std::to_string(kMaxVerifyColumns) + ")");
    }
    return options;
}

// The drafter of the session's rounds: Google's MTP assistant, a DFlash drafter or prompt lookup.
std::unique_ptr<speculation::DraftRound> make_round(Target& target, DeviceContext& device,
                                                    DeviceArena& work,
                                                    const SessionOptions& options) {
    if (int(options.drafter != nullptr) + int(options.dflash != nullptr) + int(options.ngram) > 1) {
        throw std::invalid_argument("gemma4 session: choose one drafter");
    }
    const auto k = options.draft_tokens;
    // The verify width k+1 must stay on A16 routes (plain decode never runs the W4A4 ones) and
    // inside the expert bank's small-T route, which covers T <= 16 for both drafters. DFlash
    // also verifies at most the drafts of one block.
    const std::int32_t widest =
        options.dflash != nullptr
            ? std::min(options.dflash->config().block_size, kMaxVerifyColumns)
            : kMaxVerifyColumns;
    if (k < 1 || k + 1 > widest) {
        throw std::invalid_argument("gemma4 session: draft_tokens must be in [1, " +
                                    std::to_string(widest - 1) + "]");
    }
    // The verify pass is one forward over k+1 columns: the workspace and the target's sliding
    // rings (window + prefill_chunk slots) are sized for a chunk, so it must fit in one.
    if (k + 1 > options.prefill_chunk) {
        throw std::invalid_argument("gemma4 session: draft_tokens + 1 exceeds the prefill chunk");
    }
    if (options.dflash != nullptr) {
        return std::make_unique<speculation::DFlashRound>(*options.dflash, target, device, work,
                                                          options.max_context,
                                                          options.prefill_chunk);
    }
    if (options.ngram) { return std::make_unique<speculation::NgramRound>(device, k); }
    return std::make_unique<MtpRound>(*options.drafter, target, device, work, options.max_context,
                                      k);
}

} // namespace

struct Session::Impl {
    const Model& model;
    const Config& c;
    DeviceContext& device;
    SessionOptions options;
    DeviceArena work;
    Target target;
    DeviceBuffer ids, positions, token, position, step, out_tokens, logits, logprobs, targets;
    DeviceBuffer sampling;
    DecodeGraphExecutable decode_graph;
    // Speculation, only with a drafter: its rounds' drafter (one of the three kinds), the rounds
    // themselves, and the per-sequence driver of speculate_until() (reset by reset()).
    std::unique_ptr<speculation::DraftRound> round;
    MtpRound* mtp                    = nullptr;
    speculation::DFlashRound* dflash = nullptr;
    speculation::NgramRound* ngram   = nullptr;
    std::optional<speculation::Decoder> speculation;
    speculation::SequenceDriver driver;

    Impl(const Model& m, DeviceContext& d, SessionOptions o)
        : model(m), c(m.config()), device(d), options(checked(o)),
          work(workspace_bytes(m, options)),
          target(m, d, work, options.max_context, options.prefill_chunk, options.kv_fp8,
                 options.kv_fp8_full),
          driver(speculation::DriverOptions::of(options)) {
        const auto chunk = options.prefill_chunk;
        ids = DeviceBuffer(sizeof(std::int32_t) * std::max(chunk, options.score_columns));
        positions        = DeviceBuffer(sizeof(std::int32_t) * chunk);
        token            = DeviceBuffer(sizeof(std::int32_t));
        position         = DeviceBuffer(sizeof(std::int32_t));
        step             = DeviceBuffer(sizeof(std::int32_t));
        out_tokens       = DeviceBuffer(sizeof(std::int32_t) * kMaxDecode);
        logits           = DeviceBuffer(bf16_bytes(std::int64_t(c.vocab_size)));
        logprobs         = DeviceBuffer(sizeof(float) * options.score_columns);
        targets          = DeviceBuffer(sizeof(std::int32_t) * options.score_columns);
        sampling         = DeviceBuffer(sizeof(ops::SamplingConfig));
        upload_sampling(options.temperature, options.top_p, options.top_k, options.seed);
        if (options.drafter != nullptr || options.dflash != nullptr || options.ngram) {
            round  = make_round(target, device, work, options);
            mtp    = dynamic_cast<MtpRound*>(round.get());
            dflash = dynamic_cast<speculation::DFlashRound*>(round.get());
            ngram  = dynamic_cast<speculation::NgramRound*>(round.get());
            const speculation::DecodeState state{
                static_cast<std::int32_t*>(token.p), static_cast<std::int32_t*>(position.p),
                static_cast<std::int32_t*>(out_tokens.p), static_cast<std::int32_t*>(step.p)};
            speculation.emplace(target, device, work,
                                speculation::DecoderOptions{options.max_context,
                                                            options.draft_tokens,
                                                            options.min_draft_tokens},
                                state, *round);
        }
    }

    // The target's chunk and a scoring head, plus what a DFlash drafter holds on top of it.
    static std::size_t workspace_bytes(const Model& m, const SessionOptions& o) {
        std::size_t bytes =
            Target::workspace_bytes(m, o.max_context, o.prefill_chunk, o.score_columns);
        if (o.dflash != nullptr) {
            bytes += speculation::DFlashRound::workspace_bytes(*o.dflash, o.max_context,
                                                                o.prefill_chunk);
        }
        return bytes;
    }

    void upload_sampling(float temperature, float top_p, std::int32_t top_k,
                         unsigned long long seed) {
        ops::SamplingConfig config;
        config.temperature = temperature;
        config.top_p       = top_p;
        config.top_k       = top_k;
        config.seed        = seed;
        sampling.copy_from_host(&config, sizeof(config));
    }

    const ops::SamplingConfig* sampling_config() const {
        return static_cast<const ops::SamplingConfig*>(sampling.p);
    }

    Tensor device_i32(DeviceBuffer& buffer, std::int32_t count) {
        return Tensor(buffer.p, DType::I32, {count});
    }

    // The decode step as a graph, captured once; every input it reads (token, position, KV,
    // weights) sits at a fixed address, and the attention launch is sized by the context
    // envelope, not the position.
    void capture_decode_graph() {
        if (decode_graph.ready()) { return; }
        const cudaStream_t s = device.stream;
        Tensor pos           = device_i32(position, 1);
        device.synchronize();
        work.reset();
        {
            // One eager pass first, so every Op's one-time host setup (function attributes, route
            // tables) happens outside the capture. It rewrites this position's K/V with the values
            // the first replay will write again, and leaves the input token untouched.
            Tensor ids = device_i32(token, 1);
            const ops::Gemma4AttentionEnvelope envelope{
                static_cast<std::uint32_t>(options.max_context), 1};
            Tensor x    = target.forward(ids, pos, envelope);
            Tensor last = Tensor(logits.p, DType::BF16, {c.vocab_size, 1});
            target.head(x, last);
            device.synchronize();
            work.reset();
        }
        DecodeGraphDefinition definition;
        definition.capture(s, [&] {
            Tensor ids = device_i32(token, 1);
            record_input<<<1, 1, 0, s>>>(static_cast<const std::int32_t*>(token.p),
                                         static_cast<std::int32_t*>(out_tokens.p),
                                         static_cast<std::int32_t*>(step.p));
            CUDA_CHECK(cudaGetLastError());
            const ops::Gemma4AttentionEnvelope envelope{
                static_cast<std::uint32_t>(options.max_context), 1};
            Tensor x    = target.forward(ids, pos, envelope);
            Tensor last = Tensor(logits.p, DType::BF16, {c.vocab_size, 1});
            target.head(x, last);
            // The next input is written over this step's input only after every reader of it
            // (the embedding, at the start of this graph) has run: the draw is the last node.
            // Its RNG key is this step's position, so a seed reproduces a whole continuation.
            ops::sample(last, ids, c.vocab_size, sampling_config(), pos,
                        ops::kSamplePurposeDecode, work, s);
            ops::increment_i32_scalar(pos, s);
        });
        decode_graph.instantiate(definition);
        work.reset();
    }
};

Session::Session(const Model& model, DeviceContext& device, SessionOptions options)
    : impl_(std::make_unique<Impl>(model, device, options)) {}

Session::~Session() = default;

void Session::reset() noexcept {
    position_ = 0;
    impl_->target.poison_kv();
    if (impl_->speculation) { impl_->speculation->reset_schedule(); }
    impl_->driver.reset();
}

void Session::warm_up() {
    auto& m = *impl_;
    if (position_ != 0) {
        throw std::logic_error("gemma4 warm_up: call it before the first prefill");
    }
    // The graphs read the pending token and its position from the device. Token 0 at position 0
    // keeps every warm-up pass defined: each column attends only to keys the pass itself wrote.
    Tensor token = m.device_i32(m.token, 1);
    Tensor pos   = m.device_i32(m.position, 1);
    ops::set_i32_scalar(token, 0, m.device.stream);
    ops::set_i32_scalar(pos, 0, m.device.stream);
    m.capture_decode_graph();
    if (m.speculation) { m.speculation->capture_rounds(); }
    m.device.synchronize();
    m.work.reset();
}

bool Session::speculation_latched() const noexcept { return impl_->driver.latched(); }

std::int64_t Session::plain_steps() const noexcept { return impl_->driver.plain_steps(); }

void Session::set_sampling(float temperature, float top_p, std::int32_t top_k,
                           unsigned long long seed) {
    impl_->device.synchronize();
    impl_->upload_sampling(temperature, top_p, top_k, seed);
}

std::uint64_t Session::device_bytes() const noexcept {
    return impl_->work.capacity() + impl_->target.kv_bytes();
}

std::uint64_t Session::kv_bytes() const noexcept { return impl_->target.kv_bytes(); }

void Session::prefill(std::span<const std::int32_t> tokens) {
    auto& m = *impl_;
    if (tokens.empty()) { throw std::invalid_argument("gemma4 prefill: no tokens"); }
    check_token_ids(tokens, m.c.vocab_size, "gemma4 prefill");
    if (position_ + static_cast<std::int64_t>(tokens.size()) > m.options.max_context) {
        throw std::invalid_argument("gemma4 prefill: prompt exceeds the session context");
    }
    const cudaStream_t s      = m.device.stream;
    MtpRound* const mtp                    = m.mtp;
    speculation::DFlashRound* const dflash = m.dflash;
    std::size_t done          = 0;
    // A prefill call takes the prefill route profile, as the Engine's prefill
    // (program_family.cpp prefill_text_chunk) does: the same bytes for a column at every width.
    const ops::ScopedPrefillRoutes prefill_routes(true);
    while (done < tokens.size()) {
        const auto t =
            qwen3_5::execution::session_call_tokens(tokens.size() - done, m.options.prefill_chunk);
        m.work.reset();
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(t));
        Tensor ids = m.device_i32(m.ids, t);
        CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + done, sizeof(std::int32_t) * t,
                                   cudaMemcpyHostToDevice, s));
        Tensor pos = m.device_i32(m.positions, t);
        ops::fill_i32_positions(pos, position_, s);
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(position_ + t), t};
        Tensor x = dflash != nullptr ? dflash->prefill(ids, pos, envelope)
                                     : m.target.forward(ids, pos, envelope);
        done += static_cast<std::size_t>(t);
        position_ += t;
        if (done == tokens.size()) {
            Tensor last = x.slice(1, t - 1, 1);
            if (mtp != nullptr) {
                // The drafter's first step reads the state that predicted the pending token.
                CUDA_CHECK(cudaMemcpyAsync(mtp->carry().data, last.data, last.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
            Tensor logits = Tensor(m.logits.p, DType::BF16, {m.c.vocab_size, 1});
            m.target.head(last, logits);
            Tensor next = m.device_i32(m.token, 1);
            // The drawn token is keyed by its own absolute position, as every decode draw is.
            Tensor key = m.device_i32(m.position, 1);
            ops::set_i32_scalar(key, position_, s);
            ops::sample(logits, next, m.c.vocab_size, m.sampling_config(), key,
                        ops::kSamplePurposePrefill, m.work, s);
        }
        m.device.synchronize();
    }
    m.work.reset();
    m.driver.record(tokens);
}

std::vector<float> Session::score(std::span<const std::int32_t> tokens,
                                  std::vector<std::int32_t>* predictions,
                                  std::vector<float>* logits_out) {
    auto& m = *impl_;
    if (tokens.size() < 2) { throw std::invalid_argument("gemma4 score: need two tokens"); }
    check_token_ids(tokens, m.c.vocab_size, "gemma4 score");
    if (position_ + static_cast<std::int64_t>(tokens.size()) > m.options.max_context) {
        throw std::invalid_argument("gemma4 score: sequence exceeds the session context");
    }
    const cudaStream_t s = m.device.stream;
    std::vector<float> out;
    out.reserve(tokens.size() - 1);
    std::size_t done = 0;
    // Scoring runs the prompt as a prefill does (the prefill route profile, as Session::prefill and
    // the dense session's score): a column's bytes, and so its log-probability, are the ones serving
    // computes for that token at any chunk width.
    const ops::ScopedPrefillRoutes prefill_routes(true);
    while (done < tokens.size()) {
        const auto t =
            qwen3_5::execution::session_call_tokens(tokens.size() - done, m.options.prefill_chunk);
        m.work.reset();
        Tensor ids = m.device_i32(m.ids, t);
        CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + done, sizeof(std::int32_t) * t,
                                   cudaMemcpyHostToDevice, s));
        Tensor pos = m.device_i32(m.positions, t);
        ops::fill_i32_positions(pos, position_, s);
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(position_ + t), t};
        Tensor x = m.target.forward(ids, pos, envelope);
        // Column j predicts tokens[done + j + 1]; the final token of the sequence predicts nothing.
        const auto scored =
            static_cast<std::int32_t>(std::min<std::size_t>(t, tokens.size() - 1 - done));
        const std::int32_t columns = m.options.score_columns;
        for (std::int32_t begin = 0; begin < scored; begin += columns) {
            const std::int32_t n = std::min(columns, scored - begin);
            auto scope           = m.work.scope();
            Tensor logits        = m.work.alloc(DType::BF16, {m.c.vocab_size, n});
            m.target.head(x.slice(1, begin, n), logits);
            Tensor target = m.device_i32(m.targets, n);
            CUDA_CHECK(cudaMemcpyAsync(target.data, tokens.data() + done + begin + 1,
                                       sizeof(std::int32_t) * n, cudaMemcpyHostToDevice, s));
            Tensor lp(m.logprobs.p, DType::FP32, {n});
            ops::target_logprobs(logits, target, m.c.vocab_size, lp, s);
            std::vector<float> host(n);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), lp.data, sizeof(float) * n,
                                       cudaMemcpyDeviceToHost, s));
            std::vector<std::int32_t> best(predictions != nullptr ? n : 0);
            if (predictions != nullptr) {
                Tensor top = m.work.alloc(DType::I32, {n});
                ops::argmax(logits, top, m.c.vocab_size, s);
                CUDA_CHECK(cudaMemcpyAsync(best.data(), top.data, sizeof(std::int32_t) * n,
                                           cudaMemcpyDeviceToHost, s));
            }
            std::vector<std::uint16_t> words(logits_out != nullptr ? logits.numel() : 0);
            if (logits_out != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(words.data(), logits.data, logits.bytes(),
                                           cudaMemcpyDeviceToHost, s));
            }
            m.device.synchronize();
            out.insert(out.end(), host.begin(), host.end());
            if (predictions != nullptr) {
                predictions->insert(predictions->end(), best.begin(), best.end());
            }
            for (const std::uint16_t word : words) { // BF16 widened exactly, column by column
                const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
                float value;
                std::memcpy(&value, &bits, sizeof(value));
                logits_out->push_back(value);
            }
        }
        done += static_cast<std::size_t>(t);
        position_ += t;
    }
    m.work.reset();
    return out;
}

std::vector<std::int32_t> Session::decode(std::int32_t count) {
    auto& m = *impl_;
    if (count <= 0 || count > kMaxDecode) {
        throw std::invalid_argument("gemma4 decode: count out of range");
    }
    if (position_ + count > m.options.max_context) {
        throw std::invalid_argument("gemma4 decode: exceeds the session context");
    }
    const cudaStream_t s = m.device.stream;
    Tensor pos           = m.device_i32(m.position, 1);
    Tensor step          = m.device_i32(m.step, 1);
    ops::set_i32_scalar(pos, position_, s);
    ops::set_i32_scalar(step, 0, s);
    m.capture_decode_graph(); // on first use, unless warm_up() captured it
    for (std::int32_t i = 0; i < count; ++i) { m.decode_graph.launch(s); }
    std::vector<std::int32_t> out(count);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), m.out_tokens.p, sizeof(std::int32_t) * count,
                               cudaMemcpyDeviceToHost, s));
    m.device.synchronize();
    position_ += count;
    m.driver.record(out);
    return out;
}

std::vector<std::int32_t> Session::speculate(std::int32_t rounds) {
    auto& m = *impl_;
    if (!m.speculation) { throw std::logic_error("gemma4 speculate: the session has no drafter"); }
    speculation::check_speculate_rounds("gemma4", rounds, m.options.draft_tokens, position_,
                                        m.options.max_context, kMaxDecode);
    return m.speculation->speculate(rounds, position_);
}

std::vector<std::int32_t> Session::speculate_until(std::int32_t max_tokens,
                                                   std::span<const std::int32_t> stops) {
    auto& m = *impl_;
    if (!m.speculation) { throw std::logic_error("gemma4 speculate: the session has no drafter"); }
    if (max_tokens <= 0 || max_tokens > kMaxDecode) {
        throw std::invalid_argument("gemma4 speculate_until: max_tokens out of range");
    }
    // The driver runs this session's rounds, plain decode and prompt-lookup rounds.
    speculation::SessionSteps<Session> steps(*this, *m.speculation, m.ngram, position_, m.token,
                                             m.device);
    return m.driver.run(max_tokens, stops, steps);
}

std::vector<std::int32_t> Session::speculate_traced(std::int32_t rounds,
                                                    std::vector<SpeculationRoundTrace>& trace,
                                                    bool profile) {
    auto& m = *impl_;
    if (!m.speculation) { throw std::logic_error("gemma4 speculate: the session has no drafter"); }
    speculation::check_speculate_rounds("gemma4", rounds, m.options.draft_tokens, position_,
                                        m.options.max_context, kMaxDecode);
    return m.speculation->speculate(rounds, position_, &trace, profile);
}

void Session::set_pending_token(std::int32_t token) {
    auto& m = *impl_;
    Tensor pending = m.device_i32(m.token, 1);
    ops::set_i32_scalar(pending, token, m.device.stream);
    m.device.synchronize();
}

SpeculationStats Session::speculation_stats() const {
    return impl_->speculation ? impl_->speculation->stats() : SpeculationStats{};
}

std::vector<float> Session::last_logits() {
    auto& m = *impl_;
    std::vector<std::uint16_t> raw(m.c.vocab_size);
    CUDA_CHECK(cudaMemcpyAsync(raw.data(), m.logits.p, raw.size() * 2, cudaMemcpyDeviceToHost,
                               m.device.stream));
    m.device.synchronize();
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const std::uint32_t bits = std::uint32_t(raw[i]) << 16;
        float value;
        std::memcpy(&value, &bits, 4);
        out[i] = value;
    }
    return out;
}

namespace {

// Device memory a captured graph keeps, per target layer. Measured on the RTX 5080 with 26B-A4B
// (30 layers): the decode graph took 4 MiB, decode plus one MTP round 9-10 MiB, decode plus seven
// round graphs (mtp:8:2) 42 MiB: about 180 KiB per layer and graph; rounded up. The slack covers
// the allocator's granularity.
// NINFERNO-SPECIAL(workaround:graph-memory-estimate): the graphs are captured after the session is
// sized, so their memory is estimated; capturing them inside the trial session would make it
// exact and remove both constants.
constexpr std::uint64_t kGraphBytesPerLayer = std::uint64_t{192} << 10;
constexpr std::uint64_t kSlackBytes         = std::uint64_t{16} << 20;
constexpr std::int32_t kContextStep         = 1024;
constexpr std::int32_t kChunks[]            = {2048, 1024, 512, 256, 128};

std::uint64_t free_device_bytes() {
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

std::uint64_t graph_reserve(const Model& model, const SessionOptions& o) {
    std::int64_t graphs = 1; // decode
    if (o.drafter != nullptr || o.dflash != nullptr || o.ngram) {
        const std::int32_t deepest = std::max(o.draft_tokens, 1);
        const std::int32_t shallowest =
            o.min_draft_tokens > 0 && o.min_draft_tokens < deepest ? o.min_draft_tokens : deepest;
        graphs += deepest - shallowest + 1;
    }
    return static_cast<std::uint64_t>(graphs) * model.config().layers * kGraphBytesPerLayer +
           kSlackBytes;
}

bool out_of_memory(const std::exception& e) {
    const std::string what = e.what();
    return what.find("out of memory") != std::string::npos ||
           what.find("cudaErrorMemoryAllocation") != std::string::npos;
}

} // namespace

SessionFit fit_session(const Model& model, DeviceContext& device, SessionOptions options,
                       const SessionFitRequest& request) {
    SessionFit fit;
    device.synchronize();
    fit.free_bytes    = free_device_bytes();
    fit.reserve_bytes = graph_reserve(model, options);
    const auto fits = [&](std::int32_t context, std::int32_t chunk) {
        ++fit.trials;
        options.max_context   = context;
        options.prefill_chunk = chunk;
        try {
            const Session trial(model, device, options);
            device.synchronize();
            return free_device_bytes() >= fit.reserve_bytes;
        } catch (const std::runtime_error& e) {
            if (!out_of_memory(e)) { throw; }
            (void)cudaGetLastError(); // allocation failures are not sticky
            return false;
        }
    };
    const auto describe = [&] {
        char text[160];
        std::snprintf(text, sizeof(text), "%.0f MiB free, %.0f MiB kept for CUDA graphs",
                      to_mib(fit.free_bytes), to_mib(fit.reserve_bytes));
        return std::string(text);
    };
    std::vector<std::int32_t> chunks;
    if (request.chunk > 0) {
        chunks.push_back(request.chunk);
    } else {
        for (const std::int32_t c : kChunks) { chunks.push_back(c); }
    }
    if (request.context > 0) {
        for (const std::int32_t chunk : chunks) {
            if (chunk > request.context) { continue; }
            if (fits(request.context, chunk)) {
                fit.context = request.context;
                fit.chunk   = chunk;
                return fit;
            }
        }
        throw std::runtime_error("gemma4 session: context " + std::to_string(request.context) +
                                 " does not fit the device (" + describe() +
                                 "); use a smaller --context");
    }
    const std::int32_t cap    = std::max(request.max_context, kContextStep);
    const std::int32_t wanted = std::min(request.preferred_context, cap);
    for (const std::int32_t chunk : chunks) {
        // The largest multiple of kContextStep at most `cap` that fits. A context shorter than a
        // sliding window plus a chunk would clip the sliding layers' rings below what one chunk
        // attends to, so the search starts above it.
        const std::int32_t floor = model.config().sliding_window + chunk;
        std::int32_t lo = std::max(1, (floor + kContextStep - 1) / kContextStep);
        std::int32_t hi = cap / kContextStep;
        if (lo > hi) { continue; }
        std::int32_t best = 0;
        if (fits(hi * kContextStep, chunk)) {
            best = hi;
        } else if (fits(lo * kContextStep, chunk)) {
            best = lo;
            ++lo;
            --hi;
            while (lo <= hi) {
                const std::int32_t mid = lo + (hi - lo) / 2;
                if (fits(mid * kContextStep, chunk)) {
                    best = mid;
                    lo   = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        if (best == 0) { continue; }
        if (best * kContextStep > fit.context) {
            fit.context = best * kContextStep;
            fit.chunk   = chunk;
        }
        if (best * kContextStep >= wanted) { break; }
    }
    if (fit.context == 0) {
        throw std::runtime_error("gemma4 session: no context fits the device (" + describe() +
                                 "); free device memory or use a smaller drafter configuration");
    }
    return fit;
}

} // namespace ninferno::models::gemma4
