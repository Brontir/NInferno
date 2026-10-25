#include "models/dense/session.h"
#include "models/common/token_ids.h"
#include "models/dense/target.h"
#include "models/qwen3_5/execution/prefill_calls.h"
#include "models/speculation/decoder.h"
#include "models/speculation/dflash_round.h"
#include "models/speculation/driver.h"
#include "models/speculation/ngram_round.h"
#include "models/speculation/session_steps.h"

#include "core/decode_graph.h"
#include "core/nvtx.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/position.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/target_logprobs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::dense {
namespace {

// Appends the token this step consumes to the output list. A step's input is the previous draw
// (the prefill's, for the first step), so decode(n) returns exactly the n tokens it fed.
__global__ void record_input(const std::int32_t* token, std::int32_t* out, std::int32_t* step) {
    out[*step] = *token;
    *step += 1;
}

// Logit rows scored per head launch in teacher-forced scoring.
constexpr std::int32_t kScoreColumns = 256;
constexpr std::int32_t kMaxDecode    = 1 << 16;

// The prefill chunk bounds the tokens of one pass, and no pass holds more tokens than the context:
// a chunk above the context (the 2048 default with a short --context) is the context. Sizing the
// attention envelope with the larger chunk would ask for more queries than visible keys.
SessionOptions checked(SessionOptions options) {
    if (options.prefill_chunk <= 0 || options.max_context <= 0) {
        throw std::invalid_argument("dense session: chunk and context must be positive");
    }
    options.prefill_chunk = std::min(options.prefill_chunk, options.max_context);
    if (options.dflash != nullptr || options.ngram) {
        if (options.dflash != nullptr && options.ngram) {
            throw std::invalid_argument("dense session: choose one drafter");
        }
        const auto k = options.draft_tokens;
        // A DFlash round verifies at most the drafts of one block; every verify stays inside the
        // attention's decode route and inside one chunk (the sliding rings keep window + chunk).
        const std::int32_t widest =
            options.dflash != nullptr
                ? std::min(options.dflash->config().block_size, kMaxVerifyColumns)
                : kMaxVerifyColumns;
        if (k < 1 || k + 1 > widest) {
            throw std::invalid_argument("dense session: draft_tokens must be in [1, " +
                                        std::to_string(widest - 1) + "]");
        }
        if (k + 1 > options.prefill_chunk) {
            throw std::invalid_argument("dense session: draft_tokens + 1 exceeds the prefill chunk");
        }
    }
    return options;
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
    // Speculation, only with a drafter: the rounds' drafter, the rounds, and the per-sequence
    // driver of speculate_until().
    std::unique_ptr<speculation::DraftRound> round;
    speculation::DFlashRound* dflash = nullptr;
    speculation::NgramRound* ngram   = nullptr;
    std::optional<speculation::Decoder> speculation;
    speculation::SequenceDriver driver;

    // The target's chunk and a scoring head, plus what a DFlash drafter holds on top of it.
    static std::size_t workspace_bytes(const Model& m, const SessionOptions& o) {
        std::size_t bytes =
            Target::workspace_bytes(m, o.max_context, o.prefill_chunk, kScoreColumns);
        if (o.dflash != nullptr) {
            bytes += speculation::DFlashRound::workspace_bytes(*o.dflash, o.max_context,
                                                                o.prefill_chunk);
        }
        return bytes;
    }

    Impl(const Model& m, DeviceContext& d, SessionOptions o)
        : model(m), c(m.config()), device(d), options(checked(o)),
          work(workspace_bytes(m, options)),
          target(m, d, work, options.max_context, options.prefill_chunk, options.kv_fp8),
          driver(speculation::DriverOptions::of(options)) {
        const auto chunk = options.prefill_chunk;
        ids              = DeviceBuffer(sizeof(std::int32_t) * std::max(chunk, kScoreColumns));
        positions        = DeviceBuffer(sizeof(std::int32_t) * chunk);
        token            = DeviceBuffer(sizeof(std::int32_t));
        position         = DeviceBuffer(sizeof(std::int32_t));
        step             = DeviceBuffer(sizeof(std::int32_t));
        out_tokens       = DeviceBuffer(sizeof(std::int32_t) * kMaxDecode);
        logits           = DeviceBuffer(std::size_t(c.vocab_size) * 2);
        logprobs         = DeviceBuffer(sizeof(float) * kScoreColumns);
        targets          = DeviceBuffer(sizeof(std::int32_t) * kScoreColumns);
        sampling         = DeviceBuffer(sizeof(ops::SamplingConfig));
        upload_sampling(options.temperature, options.top_p, options.top_k, options.seed);
        if (options.dflash != nullptr) {
            auto owned = std::make_unique<speculation::DFlashRound>(
                *options.dflash, target, d, work, options.max_context, options.prefill_chunk);
            dflash = owned.get();
            round  = std::move(owned);
        } else if (options.ngram) {
            auto owned = std::make_unique<speculation::NgramRound>(d, options.draft_tokens);
            ngram      = owned.get();
            round      = std::move(owned);
        }
        if (round) {
            const speculation::DecodeState state{
                static_cast<std::int32_t*>(token.p), static_cast<std::int32_t*>(position.p),
                static_cast<std::int32_t*>(out_tokens.p), static_cast<std::int32_t*>(step.p)};
            speculation.emplace(target, d, work,
                                speculation::DecoderOptions{options.max_context,
                                                            options.draft_tokens},
                                state, *round);
        }
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

    Tensor i32(DeviceBuffer& buffer, std::int32_t count) {
        return Tensor(buffer.p, DType::I32, {count});
    }

    // The decode step as a graph, captured once: every input it reads (token, position, KV,
    // weights) sits at a fixed address, and the attention launch is sized by the context
    // envelope, not the position.
    void capture_decode_graph() {
        if (decode_graph.ready()) { return; }
        const cudaStream_t s = device.stream;
        Tensor pos           = i32(position, 1);
        const ops::Gemma4AttentionEnvelope envelope{
            static_cast<std::uint32_t>(options.max_context), 1};
        device.synchronize();
        work.reset();
        {
            // One eager pass first, so every Op's one-time host setup (function attributes, route
            // tables) happens outside the capture. It rewrites this position's K/V with the values
            // the first replay writes again and leaves the input token untouched.
            Tensor ids         = i32(token, 1);
            Tensor x           = target.forward(ids, pos, envelope);
            Tensor last_logits = Tensor(logits.p, DType::BF16, {c.vocab_size, 1});
            target.head(x, last_logits);
            device.synchronize();
            work.reset();
        }
        DecodeGraphDefinition definition;
        definition.capture(s, [&] {
            Tensor ids = i32(token, 1);
            record_input<<<1, 1, 0, s>>>(static_cast<const std::int32_t*>(token.p),
                                         static_cast<std::int32_t*>(out_tokens.p),
                                         static_cast<std::int32_t*>(step.p));
            CUDA_CHECK(cudaGetLastError());
            Tensor x           = target.forward(ids, pos, envelope);
            Tensor last_logits = Tensor(logits.p, DType::BF16, {c.vocab_size, 1});
            target.head(x, last_logits);
            // The next input overwrites this step's only after every reader of it has run: the
            // draw is the last node. Its RNG key is this step's position.
            ops::sample(last_logits, ids, c.vocab_size, sampling_config(), pos,
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
    impl_->driver.reset();
}

void Session::warm_up() {
    auto& m = *impl_;
    if (position_ != 0) { throw std::logic_error("dense warm_up: call it before the first prefill"); }
    // The graphs read the pending token and its position from the device. Token 0 at position 0
    // keeps every warm-up pass defined: each column attends only to keys the pass itself wrote.
    Tensor token = m.i32(m.token, 1);
    Tensor pos   = m.i32(m.position, 1);
    ops::set_i32_scalar(token, 0, m.device.stream);
    ops::set_i32_scalar(pos, 0, m.device.stream);
    if (!m.options.eager_decode) { m.capture_decode_graph(); }
    if (m.speculation) { m.speculation->capture_rounds(); }
    m.device.synchronize();
    m.work.reset();
}

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
    if (tokens.empty()) { throw std::invalid_argument("dense prefill: no tokens"); }
    check_token_ids(tokens, m.c.vocab_size, "dense prefill");
    if (position_ + static_cast<std::int64_t>(tokens.size()) > m.options.max_context) {
        throw std::invalid_argument("dense prefill: prompt exceeds the session context");
    }
    const cudaStream_t s = m.device.stream;
    std::size_t done     = 0;
    while (done < tokens.size()) {
        const auto t =
            qwen3_5::execution::session_call_tokens(tokens.size() - done, m.options.prefill_chunk);
        m.work.reset();
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(t));
        Tensor ids = m.i32(m.ids, t);
        CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + done, sizeof(std::int32_t) * t,
                                   cudaMemcpyHostToDevice, s));
        Tensor pos = m.i32(m.positions, t);
        ops::fill_i32_positions(pos, position_, s);
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(position_ + t), t};
        // The prefill route profile: a token's result does not depend on the chunk that carries it.
        const ops::ScopedPrefillRoutes route_profile(true);
        // With a DFlash drafter every prompt token is committed context: the forward captures the
        // tapped layers and the drafter appends the chunk's keys and values.
        Tensor x = m.dflash != nullptr ? m.dflash->prefill(ids, pos, envelope)
                                       : m.target.forward(ids, pos, envelope);
        done += static_cast<std::size_t>(t);
        position_ += t;
        if (done == tokens.size()) {
            Tensor last   = x.slice(1, t - 1, 1);
            Tensor logits = Tensor(m.logits.p, DType::BF16, {m.c.vocab_size, 1});
            m.target.head(last, logits);
            Tensor next = m.i32(m.token, 1);
            // The drawn token is keyed by its own absolute position, as every decode draw is.
            Tensor key = m.i32(m.position, 1);
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
                                  std::vector<std::int32_t>* predictions) {
    auto& m = *impl_;
    if (tokens.size() < 2) { throw std::invalid_argument("dense score: need two tokens"); }
    check_token_ids(tokens, m.c.vocab_size, "dense score");
    if (position_ + static_cast<std::int64_t>(tokens.size()) > m.options.max_context) {
        throw std::invalid_argument("dense score: sequence exceeds the session context");
    }
    const cudaStream_t s = m.device.stream;
    std::vector<float> out;
    out.reserve(tokens.size() - 1);
    std::size_t done = 0;
    while (done < tokens.size()) {
        const auto t =
            qwen3_5::execution::session_call_tokens(tokens.size() - done, m.options.prefill_chunk);
        m.work.reset();
        Tensor ids = m.i32(m.ids, t);
        CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + done, sizeof(std::int32_t) * t,
                                   cudaMemcpyHostToDevice, s));
        Tensor pos = m.i32(m.positions, t);
        ops::fill_i32_positions(pos, position_, s);
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(position_ + t), t};
        // The prefill route profile: a token's result does not depend on the chunk that carries it.
        const ops::ScopedPrefillRoutes route_profile(true);
        Tensor x = m.target.forward(ids, pos, envelope);
        // Column j predicts tokens[done + j + 1]; the final token of the sequence predicts nothing.
        const auto scored =
            static_cast<std::int32_t>(std::min<std::size_t>(t, tokens.size() - 1 - done));
        for (std::int32_t begin = 0; begin < scored; begin += kScoreColumns) {
            const std::int32_t n = std::min(kScoreColumns, scored - begin);
            auto scope           = m.work.scope();
            Tensor logits        = m.work.alloc(DType::BF16, {m.c.vocab_size, n});
            m.target.head(x.slice(1, begin, n), logits);
            Tensor target = m.i32(m.targets, n);
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
            m.device.synchronize();
            out.insert(out.end(), host.begin(), host.end());
            if (predictions != nullptr) {
                predictions->insert(predictions->end(), best.begin(), best.end());
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
        throw std::invalid_argument("dense decode: count out of range");
    }
    if (position_ + count > m.options.max_context) {
        throw std::invalid_argument("dense decode: exceeds the session context");
    }
    const cudaStream_t s = m.device.stream;
    Tensor pos           = m.i32(m.position, 1);
    Tensor step          = m.i32(m.step, 1);
    ops::set_i32_scalar(pos, position_, s);
    ops::set_i32_scalar(step, 0, s);
    if (m.options.eager_decode) {
        m.device.synchronize();
        for (std::int32_t i = 0; i < count; ++i) {
            m.work.reset();
            const ops::Gemma4AttentionEnvelope step_envelope{
                static_cast<std::uint32_t>(position_ + i + 1), 1};
            Tensor ids = m.i32(m.token, 1);
            record_input<<<1, 1, 0, s>>>(static_cast<const std::int32_t*>(m.token.p),
                                         static_cast<std::int32_t*>(m.out_tokens.p),
                                         static_cast<std::int32_t*>(m.step.p));
            CUDA_CHECK(cudaGetLastError());
            Tensor x      = m.target.forward(ids, pos, step_envelope);
            Tensor logits = Tensor(m.logits.p, DType::BF16, {m.c.vocab_size, 1});
            m.target.head(x, logits);
            ops::sample(logits, ids, m.c.vocab_size, m.sampling_config(), pos,
                        ops::kSamplePurposeDecode, m.work, s);
            ops::increment_i32_scalar(pos, s);
            m.device.synchronize();
        }
        m.work.reset();
        std::vector<std::int32_t> out(count);
        CUDA_CHECK(cudaMemcpyAsync(out.data(), m.out_tokens.p, sizeof(std::int32_t) * count,
                                   cudaMemcpyDeviceToHost, s));
        m.device.synchronize();
        position_ += count;
        m.driver.record(out);
        return out;
    }
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
    if (!m.speculation || m.ngram != nullptr) {
        throw std::logic_error("dense speculate: the session has no drafter");
    }
    speculation::check_speculate_rounds("dense", rounds, m.options.draft_tokens, position_,
                                        m.options.max_context, kMaxDecode);
    return m.speculation->speculate(rounds, position_);
}

std::vector<std::int32_t> Session::speculate_traced(std::int32_t rounds,
                                                    std::vector<SpeculationRoundTrace>& trace,
                                                    bool profile) {
    auto& m = *impl_;
    if (!m.speculation || m.ngram != nullptr) {
        throw std::logic_error("dense speculate: the session has no drafter");
    }
    speculation::check_speculate_rounds("dense", rounds, m.options.draft_tokens, position_,
                                        m.options.max_context, kMaxDecode);
    return m.speculation->speculate(rounds, position_, &trace, profile);
}

std::vector<std::int32_t> Session::speculate_until(std::int32_t max_tokens,
                                                   std::span<const std::int32_t> stops) {
    auto& m = *impl_;
    if (!m.speculation) { throw std::logic_error("dense speculate: the session has no drafter"); }
    if (max_tokens <= 0 || max_tokens > kMaxDecode) {
        throw std::invalid_argument("dense speculate_until: max_tokens out of range");
    }
    // The driver runs this session's rounds, plain decode and prompt-lookup rounds.
    speculation::SessionSteps<Session> steps(*this, *m.speculation, m.ngram, position_, m.token,
                                             m.device);
    return m.driver.run(max_tokens, stops, steps);
}

SpeculationStats Session::speculation_stats() const {
    return impl_->speculation ? impl_->speculation->stats() : SpeculationStats{};
}

bool Session::speculation_latched() const noexcept { return impl_->driver.latched(); }

std::int64_t Session::plain_steps() const noexcept { return impl_->driver.plain_steps(); }

std::vector<float> Session::last_logits() {
    auto& m = *impl_;
    std::vector<std::uint16_t> raw(m.c.vocab_size);
    CUDA_CHECK(cudaMemcpyAsync(raw.data(), m.logits.p, raw.size() * 2, cudaMemcpyDeviceToHost,
                               m.device.stream));
    m.device.synchronize();
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const std::uint32_t bits = std::uint32_t(raw[i]) << 16;
        std::memcpy(&out[i], &bits, 4);
    }
    return out;
}

} // namespace ninferno::models::dense
