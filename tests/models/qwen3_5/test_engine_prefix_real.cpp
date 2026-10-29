#include "ninferno/engine.h"
#include "real_artifact.h"
#include "kv_cache_storage.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

ninferno::EngineOptions engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninferno::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 1024;
    options.speculative.backend              = ninferno::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.enable_vision                    = ninferno::test::vision(artifact, true);
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 4;
    return options;
}

ninferno::EngineOptions host_restore_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninferno::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninferno::EngineOptions shared_replacement_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninferno::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 4;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninferno::EngineOptions anthropic_prefix_regression_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 2048;
    options.kv_capacity                      = ninferno::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk                    = 512;
    options.speculative.backend              = ninferno::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 1;
    options.context_cache.host_state_slots   = 4;
    options.context_cache.host_kv_capacity_bytes            = 512ULL << 20;
    options.context_cache.max_private_continuations         = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninferno::EngineOptions shared_rewrite_materialization_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 100000;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(100000);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninferno::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninferno::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 2;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninferno::EngineOptions private_long_anchor_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninferno::SpeculativeBackend::None;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 4;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 1;
    return options;
}

ninferno::EngineOptions last_alias_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninferno::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 3;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninferno::EngineOptions concurrent_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 512;
    options.kv_capacity                      = ninferno::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninferno::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.max_concurrency                  = 8;
    options.max_pending_requests             = 8;
    options.context_cache.device_state_slots = 16;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 8;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninferno::EngineOptions pressure_resume_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 8192;
    options.kv_capacity                      = ninferno::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninferno::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninferno::SpeculativeBackend::None;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 8ULL << 30;
    options.context_cache.max_private_continuations         = 4;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninferno::EngineOptions private_checkpoint_pressure_engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 8192;
    options.kv_capacity                      = ninferno::KvCapacityPolicy::explicit_capacity(16384);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninferno::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninferno::SpeculativeBackend::None;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 4;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

// The automatic topology of a large model on a small card: a Host tier and no cached Device
// states, so the Device StateImage pool holds exactly one image per lane.
ninferno::EngineOptions host_tier_no_cached_state_options(const char* artifact,
                                                        std::uint32_t concurrency) {
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 2048;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(2048 * concurrency);
    options.prefill_chunk                    = 512;
    options.speculative.backend              = ninferno::SpeculativeBackend::None;
    options.max_concurrency                  = concurrency;
    options.max_pending_requests             = concurrency;
    options.context_cache.device_state_slots = 0;
    options.context_cache.host_state_slots   = 8;
    options.context_cache.host_kv_capacity_bytes            = 512ULL << 20;
    options.context_cache.max_private_continuations         = 2 * concurrency;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

std::vector<std::uint8_t> gradient_ppm(int width = 64, int height = 64) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninferno::PromptInput chinese_chat(bool enable_thinking) {
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(ninferno::MessagePart{.kind  = ninferno::MessagePartKind::Text,
                                                  .text  = "你好，简单介绍一下你自己。",
                                                  .media = {}});
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

// The registered tokenizer and chat template of each supported artifact family render the Chinese
// greeting to fixed token counts: {thinking, no-thinking}.
enum class TemplateFamily : std::uint8_t { Qwen3_6, Qwen3_8 };

std::optional<TemplateFamily> registered_template_family(const ninferno::Engine& engine) {
    struct PromptGolden {
        TemplateFamily family;
        std::uint32_t thinking    = 0;
        std::uint32_t no_thinking = 0;
    };
    constexpr std::array<PromptGolden, 2> goldens{{
        {TemplateFamily::Qwen3_6, 16, 18},
        {TemplateFamily::Qwen3_8, 58, 18},
    }};
    const std::uint32_t thinking    = engine.count_tokens(chinese_chat(true));
    const std::uint32_t no_thinking = engine.count_tokens(chinese_chat(false));
    for (const PromptGolden& golden : goldens) {
        if (golden.thinking == thinking && golden.no_thinking == no_thinking) {
            return golden.family;
        }
    }
    std::cerr << "registered tokenizer/chat template changed the prompt goldens: thinking="
              << thinking << " no_thinking=" << no_thinking
              << " model=" << engine.load_summary().model_name << '\n';
    return std::nullopt;
}

int exercise_registered_frontend(const ninferno::Engine& engine) {
    return registered_template_family(engine) ? 0 : 1;
}

class ObservationSink final : public ninferno::OutputSink {
public:
    void start(ninferno::GenerationStart start) override {
        if (started_) { valid_ = false; }
        started_ = true;
        start_   = start;
    }

    void progress(ninferno::PromptProgress progress) override {
        if (!started_ || timing_seen_ ||
            progress.total_prompt_tokens != start_.prompt.prompt_tokens ||
            progress.reused_prompt_tokens != start_.reused_prompt_tokens ||
            progress.processed_prompt_tokens < last_processed_ ||
            progress.processed_prompt_tokens > progress.total_prompt_tokens ||
            progress.elapsed_ns < last_progress_elapsed_ns_) {
            valid_ = false;
        }
        last_processed_           = progress.processed_prompt_tokens;
        last_progress_elapsed_ns_ = progress.elapsed_ns;
    }

    void timing(ninferno::GenerationTimingObservation timing) override {
        if (!started_ || last_processed_ != start_.prompt.prompt_tokens ||
            (timing_seen_ && (timing.generated_tokens < last_timing_.generated_tokens ||
                              timing.prompt_elapsed_ns != last_timing_.prompt_elapsed_ns ||
                              timing.generation_elapsed_ns < last_timing_.generation_elapsed_ns))) {
            valid_ = false;
        }
        timing_seen_ = true;
        last_timing_ = timing;
    }

    void publish(ninferno::OutputDelta) override {
        if (!timing_seen_) { valid_ = false; }
    }

    [[nodiscard]] bool valid_for(const ninferno::GenerationResult& result) const {
        return valid_ && started_ && timing_seen_ && start_.reused_prompt_tokens == 0 &&
               last_processed_ == start_.prompt.prompt_tokens &&
               last_timing_.generated_tokens == result.generated_token_ids.size() &&
               result.timings.prompt_wall_seconds > 0.0 &&
               result.timings.generation_wall_seconds >= 0.0;
    }

private:
    ninferno::GenerationStart start_;
    ninferno::GenerationTimingObservation last_timing_;
    std::uint32_t last_processed_           = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    bool started_                           = false;
    bool timing_seen_                       = false;
    bool valid_                             = true;
};

int exercise_stream_observations(ninferno::Engine& engine) {
    std::vector<ninferno::TokenId> prompt(2050, 198);
    ninferno::RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const ninferno::GenerationObservationOptions observation{
        .phase_timings = true, .live_timings = true, .prompt_progress = true};

    ObservationSink sink;
    ninferno::GenerationHandle generation =
        engine.submit(engine.prepare_tokens(std::move(prompt)), std::move(request),
                      ninferno::OutputConsumerMode::Streaming, observation);
    const ninferno::GenerationResult result = generation.wait(&sink);
    if (result.generated_token_ids.size() != 3 || !sink.valid_for(result)) {
        std::cerr
            << "stream observations lost prompt progress, commit timing, or publication order\n";
        return 1;
    }
    return 0;
}

int exercise_full_prefill_chunk(ninferno::Engine& engine) {
    constexpr std::size_t kChunkTokens = 1024;
    std::vector<ninferno::TokenId> prompt(kChunkTokens, 198);
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;

    const ninferno::GenerationResult result =
        engine.generate(engine.prepare_tokens(std::move(prompt)), options);
    if (result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninferno::FinishReason::OutputLimit) {
        std::cerr << "full-chunk prefill did not complete through the planned workspace\n";
        return 1;
    }
    return 0;
}

int exercise_abandoned_handle_capacity(ninferno::Engine& engine) {
    const std::vector<ninferno::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    ninferno::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    {
        auto abandoned = engine.submit(engine.prepare_tokens(prompt), request);
        if (!abandoned) {
            std::cerr << "abandonment fixture did not create a generation handle\n";
            return 1;
        }
    }
    const auto crossed = engine.generate(engine.prepare_tokens(prompt), request);
    if (crossed.generated_token_ids.size() != 1) {
        std::cerr << "request after an abandoned handle did not complete\n";
        return 1;
    }

    auto first      = engine.submit(engine.prepare_tokens(prompt), request);
    auto second     = engine.submit(engine.prepare_tokens(prompt), request);
    bool overloaded = false;
    try {
        auto third = engine.submit(engine.prepare_tokens(prompt), request);
        (void)third;
    } catch (const ninferno::RequestError& error) {
        overloaded = error.kind() == ninferno::RequestErrorKind::Overloaded;
    }
    if (!overloaded) {
        std::cerr << "outstanding capacity was released twice or not enforced\n";
        return 1;
    }
    if (first.wait().generated_token_ids.size() != 1 ||
        second.wait().generated_token_ids.size() != 1) {
        std::cerr << "requests retained after the overload check did not complete\n";
        return 1;
    }
    return 0;
}

int exercise_zero_suffix_reuse(ninferno::Engine& engine,
                               const std::vector<ninferno::TokenId>& prompt) {
    ninferno::RequestOptions baseline_options;
    baseline_options.execution.requested_output_tokens = 8;
    baseline_options.execution.sampling.temperature    = 0.0F;
    baseline_options.execution.allow_prefix_reuse      = true;
    baseline_options.stop.include_model_defaults       = false;
    const ninferno::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), baseline_options);
    if (baseline.generated_token_ids.size() != 8) {
        std::cerr << "zero-suffix baseline did not generate eight tokens\n";
        return 1;
    }

    std::vector<ninferno::TokenId> exact_frontier = prompt;
    exact_frontier.insert(exact_frontier.end(), baseline.generated_token_ids.begin(),
                          baseline.generated_token_ids.end() - 1);

    ninferno::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 2;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninferno::GenerationResult reused =
        engine.generate(engine.prepare_tokens(exact_frontier), reuse_options);
    if (reused.reused_prompt_tokens != exact_frontier.size()) {
        std::cerr << "zero-suffix reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << exact_frontier.size() << '\n';
        return 1;
    }
    if (reused.generated_token_ids.size() != 2 ||
        reused.finish_reason != ninferno::FinishReason::OutputLimit) {
        std::cerr << "zero-suffix reuse did not resume from the retained target frontier\n";
        return 1;
    }
    return 0;
}

int exercise_prefix(ninferno::Engine& engine) {
    ninferno::RequestOptions first_options;
    first_options.execution.requested_output_tokens = 5;
    first_options.execution.sampling.temperature    = 0.0F;
    first_options.stop.include_model_defaults       = false;

    const std::vector<ninferno::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninferno::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), first_options);
    if (first.generated_token_ids.size() != 5) {
        std::cerr << "first request did not generate five tokens\n";
        return 1;
    }

    std::vector<ninferno::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    ninferno::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 5;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninferno::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), reuse_options);

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "append reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }

    if (const int result = exercise_zero_suffix_reuse(engine, prompt); result != 0) {
        return result;
    }

    return 0;
}

int exercise_host_restore(const char* artifact) {
    ninferno::Engine engine(host_restore_engine_options(artifact));
    auto options = [](std::uint32_t outputs, bool reuse) {
        ninferno::RequestOptions request;
        request.execution.requested_output_tokens = outputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = reuse;
        request.stop.include_model_defaults       = false;
        return request;
    };

    const auto retained_input = [] {
        std::string text;
        text.reserve(6U * 300U);
        for (std::uint32_t index = 0; index < 300; ++index) { text += "alpha "; }
        ninferno::ChatMessage message;
        message.role = ninferno::ChatRole::User;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
        ninferno::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "host-restore-real";
        input.context_cache.retention   = ninferno::CacheRetentionHint::LiveSession;
        return input;
    };

    const ninferno::GenerationResult retained =
        engine.generate(engine.prepare(retained_input()), options(5, true));
    if (retained.prompt.prompt_tokens <= 256 || retained.generated_token_ids.size() != 5) {
        std::cerr << "Host-restore source request did not complete\n";
        return 1;
    }

    // The follow-up carries an answer edited at its start, so that it resumes from the turn
    // closure (the complete MTP checkpoint under test) on every template: Qwen3.8 renders an
    // unedited answer exactly as generated and would resume from the endpoint instead, whose
    // state stays on Device.
    ninferno::PromptInput continuation = retained_input();
    ninferno::ChatMessage assistant;
    assistant.role              = ninferno::ChatRole::Assistant;
    assistant.reasoning_content = retained.reasoning;
    assistant.parts.push_back(ninferno::MessagePart{.kind  = ninferno::MessagePartKind::Text,
                                                  .text  = "Edited: " + retained.content,
                                                  .media = {}});
    continuation.messages.push_back(std::move(assistant));
    ninferno::ChatMessage followup;
    followup.role = ninferno::ChatRole::User;
    followup.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    continuation.messages.push_back(std::move(followup));

    const ninferno::RuntimeStats before_pressure = engine.runtime_stats();
    const ninferno::GenerationResult pressure_result =
        engine.generate(engine.prepare(continuation), options(2, false));
    const ninferno::RuntimeStats after_pressure = engine.runtime_stats();
    if (pressure_result.generated_token_ids.size() != 2 ||
        after_pressure.state_d2h_count <= before_pressure.state_d2h_count ||
        after_pressure.main_kv_d2h_pages <= before_pressure.main_kv_d2h_pages ||
        after_pressure.backend_kv_d2h_pages <= before_pressure.backend_kv_d2h_pages) {
        std::cerr << "Host pressure did not demote the complete MTP checkpoint: state="
                  << after_pressure.state_d2h_count << " main=" << after_pressure.main_kv_d2h_pages
                  << " backend=" << after_pressure.backend_kv_d2h_pages
                  << " degraded=" << after_pressure.pressure_private_owners_degraded
                  << " evicted=" << after_pressure.pressure_private_owners_evicted << '\n';
        return 1;
    }

    const ninferno::GenerationResult restored =
        engine.generate(engine.prepare(std::move(continuation)), options(2, true));
    const ninferno::RuntimeStats after_restore = engine.runtime_stats();
    if (restored.generated_token_ids.size() != 2 ||
        restored.prefix_reuse_path != ninferno::PrefixReusePath::PrivateTurnClosure ||
        restored.reused_prompt_tokens == 0 ||
        after_restore.state_h2d_count <= after_pressure.state_h2d_count ||
        after_restore.main_kv_h2d_pages <= after_pressure.main_kv_h2d_pages ||
        after_restore.backend_kv_h2d_pages <= after_pressure.backend_kv_h2d_pages) {
        std::cerr << "Complete MTP checkpoint was not materialized from Host: path="
                  << static_cast<int>(restored.prefix_reuse_path)
                  << " reused=" << restored.reused_prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size()
                  << " state=" << after_restore.state_h2d_count
                  << " main=" << after_restore.main_kv_h2d_pages
                  << " backend=" << after_restore.backend_kv_h2d_pages
                  << " degraded=" << after_restore.pressure_private_owners_degraded
                  << " evicted=" << after_restore.pressure_private_owners_evicted << '\n';
        return 1;
    }

    // The uncached pressure request and checkpoint resume use different valid prefill splits, so
    // the pressure result is a completion and transfer trigger rather than an exact-token oracle.
    return 0;
}

int exercise_shared_replacement_and_full_capacity_reuse(const char* artifact) {
    ninferno::EngineOptions engine_options = shared_replacement_engine_options(artifact);
    engine_options.max_context           = 1024;
    engine_options.kv_capacity           = ninferno::KvCapacityPolicy::explicit_capacity(1024);
    engine_options.context_cache.max_private_continuations = 1;
    ninferno::Engine engine(std::move(engine_options));
    ninferno::RequestOptions capture_request;
    capture_request.execution.requested_output_tokens = 1;
    capture_request.execution.sampling.temperature    = 0.0F;
    capture_request.execution.allow_prefix_reuse      = true;
    capture_request.stop.include_model_defaults       = false;

    const auto plain_prompt = [](std::string text) {
        ninferno::PromptInput input;
        ninferno::ChatMessage user;
        user.role = ninferno::ChatRole::User;
        user.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.context_cache.retention = ninferno::CacheRetentionHint::Disposable;
        return input;
    };
    const auto tool_prompt = [](std::string tool_json, std::string question) {
        ninferno::PromptInput input;
        ninferno::ChatMessage user;
        user.role = ninferno::ChatRole::User;
        user.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(question), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.options.tool_jsons.push_back(std::move(tool_json));
        input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
            .kind             = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninferno::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninferno::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = 1,
        });
        input.context_cache.retention = ninferno::CacheRetentionHint::Disposable;
        return input;
    };

    std::string observed_text;
    for (std::uint32_t index = 0; index < 4; ++index) { observed_text += "observed-prefix "; }
    const ninferno::GenerationResult observed_first =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    const ninferno::RuntimeStats after_observed_first = engine.runtime_stats();
    const ninferno::GenerationResult observed_second =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    const ninferno::RuntimeStats after_observed_second = engine.runtime_stats();
    if (observed_first.generated_token_ids.size() != 1 ||
        observed_second.generated_token_ids.size() != 1 ||
        after_observed_first.active_captures_completed != 1 ||
        after_observed_second.active_captures_completed !=
            after_observed_first.active_captures_completed + 1U) {
        std::cerr << "observed-prefix private/shared capture sequence changed: "
                  << after_observed_first.active_captures_completed << '/'
                  << after_observed_second.active_captures_completed << '\n';
        return 1;
    }

    const ninferno::GenerationResult observed_filler = engine.generate(
        engine.prepare(plain_prompt("Unrelated private endpoint.")), capture_request);
    const ninferno::GenerationResult observed_reuse =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    if (observed_filler.generated_token_ids.size() != 1 ||
        observed_reuse.generated_token_ids.size() != 1 ||
        observed_reuse.prefix_reuse_path != ninferno::PrefixReusePath::SharedStablePrefix ||
        observed_reuse.reused_prompt_tokens == 0) {
        std::cerr << "promoted shared prefix was not reusable after private eviction: path="
                  << static_cast<int>(observed_reuse.prefix_reuse_path)
                  << " reused=" << observed_reuse.reused_prompt_tokens << '\n';
        return 1;
    }

    std::string long_description;
    for (std::uint32_t index = 0; index < 240; ++index) { long_description += "stable-schema "; }
    const std::string bravo_tool =
        std::string(R"({"type":"function","function":{"name":"bravo","description":")") +
        long_description +
        R"(","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})";
    const ninferno::GenerationResult replacement = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), capture_request);
    const ninferno::RuntimeStats after_replacement = engine.runtime_stats();
    if (replacement.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not produce its deterministic stop token\n";
        return 1;
    }
    // Remove the exact private endpoint without publishing another shared marker. The following
    // identical Bravo prompt must therefore materialize from the retained shared prefix.
    const ninferno::GenerationResult filled =
        engine.generate(engine.prepare(plain_prompt("Another private endpoint.")), capture_request);
    const ninferno::RuntimeStats after_filler = engine.runtime_stats();
    if (filled.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not displace the exact private endpoint\n";
        return 1;
    }
    ninferno::RequestOptions full_capacity_request = capture_request;
    full_capacity_request.execution.requested_output_tokens =
        std::numeric_limits<std::uint32_t>::max();
    full_capacity_request.stop.token_ids          = {replacement.generated_token_ids.front()};
    full_capacity_request.stop.publish_stop_token = true;
    const ninferno::GenerationResult reused         = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), full_capacity_request);
    const ninferno::RuntimeStats after_reuse = engine.runtime_stats();

    if (reused.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not complete all requests\n";
        return 1;
    }
    // A shared frontier moves to its neutral prefill point (the 64-token grid here), so that a
    // restore executes the cold decomposition exactly.
    if (reused.prefix_reuse_path != ninferno::PrefixReusePath::SharedStablePrefix ||
        reused.reused_prompt_tokens == 0 || reused.reused_prompt_tokens % 64U != 0) {
        std::cerr << "single-slot shared replacement was not reusable at its neutral frontier: "
                  << "path=" << static_cast<int>(reused.prefix_reuse_path)
                  << " reused=" << reused.reused_prompt_tokens
                  << " captures=" << after_replacement.active_captures_completed << '/'
                  << after_filler.active_captures_completed << '/'
                  << after_reuse.active_captures_completed
                  << " shared_evicted=" << after_replacement.pressure_shared_owners_evicted << '/'
                  << after_filler.pressure_shared_owners_evicted << '/'
                  << after_reuse.pressure_shared_owners_evicted
                  << " shared_degraded=" << after_replacement.pressure_shared_owners_degraded << '/'
                  << after_filler.pressure_shared_owners_degraded << '/'
                  << after_reuse.pressure_shared_owners_degraded
                  << " private_evicted=" << after_replacement.pressure_private_owners_evicted << '/'
                  << after_filler.pressure_private_owners_evicted << '/'
                  << after_reuse.pressure_private_owners_evicted
                  << " refs=" << after_replacement.shared_active_references << '/'
                  << after_filler.shared_active_references << '/'
                  << after_reuse.shared_active_references
                  << " targets=" << reused.materialization.targets_evaluated
                  << " degradation=" << reused.materialization.selected_degradation_units
                  << " maximal=" << reused.materialization.selected_maximal_fallback << " stop="
                  << ninferno::materialization_stop_reason_name(reused.materialization.stop_reason)
                  << '\n';
        return 1;
    }
    if (after_replacement.active_captures_completed < 2 ||
        after_reuse.active_captures_completed < after_replacement.active_captures_completed) {
        std::cerr << "shared replacement capture was skipped under full State/KV capacity: "
                  << after_replacement.active_captures_completed << '/'
                  << after_reuse.active_captures_completed << '\n';
        return 1;
    }
    if (after_replacement.shared_active_references != 0 ||
        after_filler.shared_active_references != 0 || after_reuse.shared_active_references != 0) {
        std::cerr << "completed shared-prefix requests leaked active references: "
                  << after_replacement.shared_active_references << '/'
                  << after_filler.shared_active_references << '/'
                  << after_reuse.shared_active_references << '\n';
        return 1;
    }
    return 0;
}

int exercise_anthropic_prefix_regression(const char* artifact) {
    ninferno::Engine engine(anthropic_prefix_regression_engine_options(artifact));
    if (!engine.options().context_cache.max_shared_prefixes ||
        *engine.options().context_cache.max_shared_prefixes !=
            ninferno::kMaximumExplicitPromptCacheMarkers) {
        std::cerr << "single-concurrency Engine did not expose four default shared prefixes\n";
        return 1;
    }

    const auto conversation = [](bool followup, const ninferno::GenerationResult* first = nullptr) {
        ninferno::PromptInput input;
        input.options.enable_thinking   = true;
        input.options.preserve_thinking = true;
        input.context_cache.session_key = "anthropic-prefix-regression";
        input.context_cache.retention   = ninferno::CacheRetentionHint::LiveSession;

        ninferno::ChatMessage user;
        user.role = ninferno::ChatRole::User;
        user.parts.push_back(ninferno::MessagePart{.kind  = ninferno::MessagePartKind::Text,
                                                 .text  = "Reply with exactly the word blue.",
                                                 .media = {}});
        input.messages.push_back(std::move(user));
        if (!followup) { return input; }
        if (first == nullptr) { throw std::logic_error("followup fixture has no source result"); }

        ninferno::ChatMessage assistant;
        assistant.role              = ninferno::ChatRole::Assistant;
        assistant.reasoning_content = first->reasoning;
        if (!first->content.empty()) {
            assistant.parts.push_back(ninferno::MessagePart{
                .kind = ninferno::MessagePartKind::Text, .text = first->content, .media = {}});
        }
        input.messages.push_back(std::move(assistant));
        ninferno::ChatMessage next;
        next.role = ninferno::ChatRole::User;
        next.parts.push_back(ninferno::MessagePart{.kind  = ninferno::MessagePartKind::Text,
                                                 .text  = "Now reply with exactly the word green.",
                                                 .media = {}});
        input.messages.push_back(std::move(next));
        return input;
    };
    const auto generation_options = [](std::uint32_t outputs, bool model_stops) {
        ninferno::RequestOptions options;
        options.execution.requested_output_tokens = outputs;
        options.execution.sampling.temperature    = 0.0F;
        options.execution.allow_prefix_reuse      = true;
        options.stop.include_model_defaults       = model_stops;
        return options;
    };

    const ninferno::GenerationResult first =
        engine.generate(engine.prepare(conversation(false)), generation_options(192, true));
    if (first.reasoning.empty() || first.content.empty() ||
        first.finish_reason != ninferno::FinishReason::StopToken) {
        std::cerr << "endpoint regression fixture did not produce a closed reasoning response: "
                  << "reasoning=" << first.reasoning.size() << " content=" << first.content.size()
                  << " finish=" << static_cast<int>(first.finish_reason) << '\n';
        return 1;
    }
    const ninferno::RuntimeStats before_followup = engine.runtime_stats();
    const ninferno::GenerationResult followup =
        engine.generate(engine.prepare(conversation(true, &first)), generation_options(1, false));
    const ninferno::RuntimeStats after_followup = engine.runtime_stats();
    const std::uint64_t followup_prefill =
        after_followup.computed_prefill_tokens - before_followup.computed_prefill_tokens;
    if (followup.generated_token_ids.size() != 1 ||
        followup.prefix_reuse_path != ninferno::PrefixReusePath::PrivateEndpoint ||
        followup.reused_prompt_tokens == 0 ||
        followup_prefill != followup.prompt.prompt_tokens - followup.reused_prompt_tokens) {
        std::cerr << "model-output reasoning frontier did not resume from PrivateEndpoint: path="
                  << static_cast<int>(followup.prefix_reuse_path)
                  << " reused=" << followup.reused_prompt_tokens
                  << " prompt=" << followup.prompt.prompt_tokens << " computed=" << followup_prefill
                  << '\n';
        return 1;
    }

    const auto tool_definition = [](std::string name, std::string word) {
        std::string description;
        for (std::uint32_t index = 0; index < 160; ++index) {
            description += word;
            description.push_back(' ');
        }
        return std::string(R"({"type":"function","function":{"name":")") + name +
               R"(","description":")" + description +
               R"(","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}})";
    };
    const std::string alpha   = tool_definition("alpha", "stable-alpha");
    const std::string bravo   = tool_definition("bravo", "stable-bravo");
    const std::string charlie = tool_definition("charlie", "branch-charlie");
    const auto tool_prompt    = [](std::vector<std::string> tools, std::string question) {
        ninferno::PromptInput input;
        input.options.enable_thinking = false;
        input.options.tool_jsons      = std::move(tools);
        input.context_cache.retention = ninferno::CacheRetentionHint::Disposable;
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        for (std::uint32_t count = 1; count <= input.options.tool_jsons.size(); ++count) {
            input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                   .kind             = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
                   .evidence         = ninferno::SharedCandidateEvidence::ExplicitBoundary,
                   .location         = ninferno::PromptCacheMarkerLocation::ToolBoundary,
                   .after_tool_count = count,
            });
        }
        ninferno::ChatMessage user;
        user.role = ninferno::ChatRole::User;
        user.parts.push_back(ninferno::MessagePart{
               .kind = ninferno::MessagePartKind::Text, .text = std::move(question), .media = {}});
        input.messages.push_back(std::move(user));
        return input;
    };
    const auto filler_prompt = [](std::string text) {
        ninferno::PromptInput input;
        input.options.enable_thinking = false;
        input.context_cache.retention = ninferno::CacheRetentionHint::Disposable;
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        ninferno::ChatMessage user;
        user.role = ninferno::ChatRole::User;
        user.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        return input;
    };

    const ninferno::RequestOptions one_token = generation_options(1, false);
    const ninferno::GenerationResult seed    = engine.generate(
        engine.prepare(tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninferno::GenerationResult first_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private seed continuation.")), one_token);
    if (seed.generated_token_ids.size() != 1 || first_filler.generated_token_ids.size() != 1) {
        std::cerr << "shared compact fixture did not establish its seed and filler\n";
        return 1;
    }

    const ninferno::RuntimeStats before_late = engine.runtime_stats();
    const ninferno::GenerationResult late    = engine.generate(
        engine.prepare(tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninferno::RuntimeStats after_late = engine.runtime_stats();
    const std::uint64_t late_prefill =
        after_late.computed_prefill_tokens - before_late.computed_prefill_tokens;
    const ninferno::GenerationResult second_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private late continuation.")), one_token);
    const ninferno::RuntimeStats before_branch = engine.runtime_stats();
    const ninferno::GenerationResult branch    = engine.generate(
        engine.prepare(tool_prompt({alpha, charlie}, "Use one listed function.")), one_token);
    const ninferno::RuntimeStats after_branch = engine.runtime_stats();
    const std::uint64_t branch_prefill =
        after_branch.computed_prefill_tokens - before_branch.computed_prefill_tokens;

    if (late.prefix_reuse_path != ninferno::PrefixReusePath::SharedStablePrefix ||
        branch.prefix_reuse_path != ninferno::PrefixReusePath::SharedStablePrefix ||
        late.generated_token_ids.size() != 1 || branch.generated_token_ids.size() != 1 ||
        late.reused_prompt_tokens <= branch.reused_prompt_tokens ||
        branch.reused_prompt_tokens == 0 || second_filler.generated_token_ids.size() != 1 ||
        late_prefill != late.prompt.prompt_tokens - late.reused_prompt_tokens ||
        branch_prefill != branch.prompt.prompt_tokens - branch.reused_prompt_tokens) {
        std::cerr << "default shared catalog did not retain nested compact frontiers: late_path="
                  << static_cast<int>(late.prefix_reuse_path)
                  << " late_reused=" << late.reused_prompt_tokens
                  << " late_prompt=" << late.prompt.prompt_tokens
                  << " late_computed=" << late_prefill
                  << " branch_path=" << static_cast<int>(branch.prefix_reuse_path)
                  << " branch_reused=" << branch.reused_prompt_tokens
                  << " branch_prompt=" << branch.prompt.prompt_tokens
                  << " branch_computed=" << branch_prefill << '\n';
        return 1;
    }
    return 0;
}

int exercise_shared_rewrite_materialization(const char* artifact) {
    ninferno::Engine engine(shared_rewrite_materialization_engine_options(artifact));

    std::vector<std::string> tools;
    tools.reserve(40);
    tools.push_back(
        R"({"type":"function","function":{"name":"read_chunk","description":"Read the next diagnostic chunk. Always use this tool until told done.","parameters":{"type":"object","properties":{"chunk":{"type":"integer"}},"required":["chunk"]}}})");
    for (std::uint32_t index = 1; index < 40; ++index) {
        tools.push_back(
            std::string(R"({"type":"function","function":{"name":"unused_tool_)") +
            std::to_string(index) +
            R"(","description":"Unused diagnostic tool.","parameters":{"type":"object","properties":{"value":{"type":"string"}}}}})");
    }

    constexpr std::string_view system_text =
        "You are testing a tool loop. On every turn call read_chunk exactly once with the next "
        "integer chunk number. Do not finish or answer in prose.";
    std::vector<ninferno::ChatMessage> messages;
    ninferno::ChatMessage system;
    system.role = ninferno::ChatRole::System;
    system.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = std::string(system_text), .media = {}});
    messages.push_back(std::move(system));
    ninferno::ChatMessage user;
    user.role = ninferno::ChatRole::User;
    user.parts.push_back(ninferno::MessagePart{
        .kind  = ninferno::MessagePartKind::Text,
        .text  = "Start by calling read_chunk with chunk 1.",
        .media = {},
    });
    messages.push_back(std::move(user));

    const auto input = [&](bool latest_is_tool) {
        ninferno::PromptInput prompt;
        prompt.messages                  = messages;
        prompt.options.enable_thinking   = true;
        prompt.options.preserve_thinking = true;
        prompt.options.tool_jsons        = tools;
        prompt.context_cache.markers.push_back(ninferno::PromptCacheMarker{
            .kind     = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninferno::SharedCandidateEvidence::ExplicitBoundary,
            .location = ninferno::PromptCacheMarkerLocation::LeadingInstructionBoundary,
            .leading_instruction_bytes = static_cast<std::uint32_t>(system_text.size()),
        });
        prompt.context_cache.markers.push_back(ninferno::PromptCacheMarker{
            .kind             = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninferno::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninferno::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(tools.size()),
        });
        if (latest_is_tool) {
            prompt.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(messages.size()),
                .kind                = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence            = ninferno::SharedCandidateEvidence::ExplicitBoundary,
                .location            = ninferno::PromptCacheMarkerLocation::MessageBoundary,
            });
        } else {
            prompt.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(messages.size()),
                .kind                = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence            = ninferno::SharedCandidateEvidence::ExplicitBoundary,
                .location            = ninferno::PromptCacheMarkerLocation::MessagePartBoundary,
                .after_message_part_count = 1,
            });
        }
        return prompt;
    };
    const auto request_options = [] {
        ninferno::RequestOptions request;
        request.execution.requested_output_tokens = 16384;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.thinking.budget         = 1024;
        request.execution.allow_prefix_reuse      = true;
        return request;
    };
    const auto append_result = [&](const ninferno::GenerationResult& result, std::uint32_t turn) {
        if (result.tool_calls.size() != 1) {
            throw std::logic_error("shared rewrite fixture did not produce one tool call");
        }
        ninferno::ChatMessage assistant;
        assistant.role              = ninferno::ChatRole::Assistant;
        assistant.reasoning_content = result.reasoning + "\nHistory normalized by the client.";
        if (!result.content.empty()) {
            assistant.parts.push_back(ninferno::MessagePart{
                .kind = ninferno::MessagePartKind::Text, .text = result.content, .media = {}});
        }
        const std::string call_id = "call_" + std::to_string(turn);
        assistant.tool_calls.push_back(ninferno::ToolCall{
            .id             = call_id,
            .name           = result.tool_calls.front().name,
            .arguments_json = result.tool_calls.front().arguments_json,
        });
        messages.push_back(std::move(assistant));

        std::string diagnostic;
        diagnostic.reserve(64000);
        for (std::uint32_t line = 0; line < 500; ++line) {
            diagnostic += "chunk=" + std::to_string(turn) + " line=" + std::to_string(line) +
                          " key=value abcdefghijklmnopqrstuvwxyz0123456789 "
                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ9876543210\n";
        }
        ninferno::ChatMessage tool_result;
        tool_result.role         = ninferno::ChatRole::Tool;
        tool_result.tool_call_id = call_id;
        tool_result.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(diagnostic), .media = {}});
        messages.push_back(std::move(tool_result));
    };

    const ninferno::GenerationResult first =
        engine.generate(engine.prepare(input(false)), request_options());
    append_result(first, 1);
    const ninferno::GenerationResult second =
        engine.generate(engine.prepare(input(true)), request_options());
    append_result(second, 2);
    const ninferno::RuntimeStats before_third = engine.runtime_stats();
    const ninferno::GenerationResult third =
        engine.generate(engine.prepare(input(true)), request_options());
    const ninferno::RuntimeStats after_third = engine.runtime_stats();

    if (first.prefix_reuse_path != ninferno::PrefixReusePath::Root ||
        second.reused_prompt_tokens == 0 ||
        third.prefix_reuse_path != ninferno::PrefixReusePath::PrivateResponseReplay ||
        third.reused_prompt_tokens == 0 || third.generated_token_ids.empty() ||
        after_third.historical_fork_hits <= before_third.historical_fork_hits) {
        std::cerr << "shared/private rewrite alias did not materialize through its active Fork: "
                  << "first_path=" << static_cast<int>(first.prefix_reuse_path)
                  << " second_path=" << static_cast<int>(second.prefix_reuse_path)
                  << " second_reused=" << second.reused_prompt_tokens
                  << " third_path=" << static_cast<int>(third.prefix_reuse_path)
                  << " third_reused=" << third.reused_prompt_tokens
                  << " third_outputs=" << third.generated_token_ids.size()
                  << " forks=" << before_third.historical_fork_hits << '/'
                  << after_third.historical_fork_hits << '\n';
        return 1;
    }
    return 0;
}

int exercise_private_long_anchor_capture_and_replacement(const char* artifact) {
    ninferno::Engine engine(private_long_anchor_engine_options(artifact));

    const auto input = [](std::vector<std::string> turns,
                          std::optional<std::uint32_t> marker_after) {
        ninferno::PromptInput prompt;
        for (std::string& text : turns) {
            ninferno::ChatMessage message;
            message.role = ninferno::ChatRole::User;
            message.parts.push_back(ninferno::MessagePart{
                .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking = false;
        if (marker_after) {
            prompt.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                .after_message_count = *marker_after,
                .kind                = ninferno::PromptCacheMarkerKind::PrivateLongAnchor,
                .location            = ninferno::PromptCacheMarkerLocation::MessageBoundary,
            });
        }
        return prompt;
    };
    ninferno::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    // A long anchor moves down to its neutral prefill point on the 64-token grid: each message is
    // long enough that the two anchors below land on different grid points.
    std::string stable;
    std::string replacement_branch;
    for (int sentence = 0; sentence < 8; ++sentence) {
        stable += "This is the stable conversation prefix retained for a later branch. ";
        replacement_branch += "Follow the replacement branch through its own long preamble. ";
    }
    const ninferno::GenerationResult source = engine.generate(
        engine.prepare(input({stable, "Follow the original branch."}, 1)), request);
    if (source.generated_token_ids.size() != 1 ||
        source.prefix_reuse_path != ninferno::PrefixReusePath::Root) {
        std::cerr << "first private long-anchor capture did not complete from Root\n";
        return 1;
    }

    ninferno::PromptInput replacement_input =
        input({stable, replacement_branch, "This suffix belongs only to the replacement source."},
              2);
    // Name the replacement lineage so the final request selects this continuation rather than
    // legitimately preferring the older anonymous branch when the private catalog is full.
    replacement_input.context_cache.session_key = "private-long-anchor-replacement";
    replacement_input.context_cache.retention   = ninferno::CacheRetentionHint::LiveSession;
    const ninferno::GenerationResult replacement =
        engine.generate(engine.prepare(std::move(replacement_input)), request);
    if (replacement.generated_token_ids.size() != 1 ||
        replacement.prefix_reuse_path != ninferno::PrefixReusePath::PrivateLongAnchor ||
        replacement.reused_prompt_tokens == 0 ||
        replacement.reused_prompt_tokens >= replacement.prompt.prompt_tokens) {
        std::cerr << "private long anchor was not selected before full-capacity replacement: path="
                  << static_cast<int>(replacement.prefix_reuse_path)
                  << " reused=" << replacement.reused_prompt_tokens
                  << " prompt=" << replacement.prompt.prompt_tokens << '\n';
        return 1;
    }

    ninferno::PromptInput replaced_input =
        input({stable, replacement_branch, "Continue through a different branch suffix."},
              std::nullopt);
    replaced_input.context_cache.session_key = "private-long-anchor-replacement";
    replaced_input.context_cache.retention   = ninferno::CacheRetentionHint::LiveSession;
    const ninferno::GenerationResult replaced =
        engine.generate(engine.prepare(std::move(replaced_input)), request);
    if (replaced.generated_token_ids.size() != 1 ||
        replaced.prefix_reuse_path != ninferno::PrefixReusePath::PrivateLongAnchor ||
        replaced.reused_prompt_tokens <= replacement.reused_prompt_tokens ||
        replaced.reused_prompt_tokens >= replaced.prompt.prompt_tokens) {
        const ninferno::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "replacement private long anchor was not reusable: path="
                  << static_cast<int>(replaced.prefix_reuse_path)
                  << " first_reused=" << replacement.reused_prompt_tokens
                  << " replaced_reused=" << replaced.reused_prompt_tokens
                  << " prompt=" << replaced.prompt.prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    return 0;
}

int exercise_last_private_alias_eviction(const char* artifact) {
    ninferno::Engine engine(last_alias_engine_options(artifact));
    std::string prompt_text;
    prompt_text.reserve(6U * 300U + 8U);
    for (std::uint32_t index = 0; index < 300; ++index) { prompt_text += "alpha "; }

    const auto input = [&](std::string session, ninferno::CacheRetentionHint retention) {
        ninferno::PromptInput prompt;
        ninferno::ChatMessage user;
        user.role = ninferno::ChatRole::User;
        user.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = prompt_text, .media = {}});
        prompt.messages.push_back(std::move(user));
        prompt.options.enable_thinking   = false;
        prompt.context_cache.session_key = std::move(session);
        prompt.context_cache.retention   = retention;
        return prompt;
    };
    if (engine.count_tokens(input("probe", ninferno::CacheRetentionHint::Disposable)) % 64U == 0) {
        prompt_text += "beta";
    }

    ninferno::RequestOptions one_token;
    one_token.execution.requested_output_tokens = 1;
    one_token.execution.sampling.temperature    = 0.0F;
    one_token.execution.allow_prefix_reuse      = true;
    one_token.stop.include_model_defaults       = false;

    const ninferno::GenerationResult source = engine.generate(
        engine.prepare(input("last-alias-source", ninferno::CacheRetentionHint::LiveSession)),
        one_token);
    const ninferno::GenerationResult branch = engine.generate(
        engine.prepare(input("last-alias-branch", ninferno::CacheRetentionHint::Disposable)),
        one_token);
    if (source.generated_token_ids.size() != 1 || branch.generated_token_ids.size() != 1 ||
        branch.reused_prompt_tokens == 0 ||
        (branch.prefix_reuse_path != ninferno::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninferno::PrefixReusePath::PrivateEndpoint)) {
        std::cerr << "last-alias fixture did not establish two private prefix aliases: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens << '\n';
        return 1;
    }

    ninferno::RequestOptions full_capacity            = one_token;
    full_capacity.execution.requested_output_tokens = std::numeric_limits<std::uint32_t>::max();
    full_capacity.stop.token_ids                    = {source.generated_token_ids.front()};
    full_capacity.stop.publish_stop_token           = true;
    const ninferno::RuntimeStats before               = engine.runtime_stats();
    const ninferno::GenerationResult consumed         = engine.generate(
        engine.prepare(input("last-alias-source", ninferno::CacheRetentionHint::LiveSession)),
        full_capacity);
    const ninferno::RuntimeStats after = engine.runtime_stats();
    if (consumed.generated_token_ids.size() != 1 || consumed.reused_prompt_tokens == 0 ||
        (consumed.prefix_reuse_path != ninferno::PrefixReusePath::PrivateResponseReplay &&
         consumed.prefix_reuse_path != ninferno::PrefixReusePath::PrivateEndpoint) ||
        after.pressure_private_owners_evicted <= before.pressure_private_owners_evicted ||
        after.device_main_kv_occupied_pages == 0 || after.device_main_kv_occupied_pages > 8 ||
        after.device_backend_kv_occupied_pages == 0 || after.device_backend_kv_occupied_pages > 8) {
        std::cerr << "last private prefix alias did not transfer into the active entitlement: path="
                  << static_cast<int>(consumed.prefix_reuse_path)
                  << " reused=" << consumed.reused_prompt_tokens
                  << " evictions=" << before.pressure_private_owners_evicted << '/'
                  << after.pressure_private_owners_evicted
                  << " main=" << after.device_main_kv_occupied_pages
                  << " backend=" << after.device_backend_kv_occupied_pages << '\n';
        return 1;
    }
    return 0;
}

enum class RewriteCheckpointCacheTopology : std::uint8_t {
    PrivateOnly,
    SharedAlias,
};

int exercise_rewrite_checkpoints(ninferno::Engine& engine,
                                 RewriteCheckpointCacheTopology topology) {
    const bool shared_alias = topology == RewriteCheckpointCacheTopology::SharedAlias;
    const ninferno::RuntimeStats initial_stats = engine.runtime_stats();

    auto text_message = [](ninferno::ChatRole role, std::string text) {
        ninferno::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto assistant_call = [&](std::string reasoning, std::string id, std::string key) {
        ninferno::ChatMessage message = text_message(ninferno::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(ninferno::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    auto input_with_history = [&](int completed_responses, bool preserve_thinking) {
        ninferno::PromptInput input;
        input.messages.push_back(text_message(
            ninferno::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (completed_responses >= 1) {
            input.messages.push_back(
                assistant_call("The first lookup should be alpha.", "call_alpha", "alpha"));
            ninferno::ChatMessage tool =
                text_message(ninferno::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
            tool.tool_call_id = "call_alpha";
            input.messages.push_back(std::move(tool));
        }
        if (completed_responses >= 2) {
            input.messages.push_back(
                assistant_call("The alpha result requests beta.", "call_beta", "beta"));
            ninferno::ChatMessage tool = text_message(ninferno::ChatRole::Tool, "{\"value\":25}");
            tool.tool_call_id        = "call_beta";
            input.messages.push_back(std::move(tool));
        }
        input.options.preserve_thinking = preserve_thinking;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return input;
    };
    auto options = [](bool reuse) {
        ninferno::RequestOptions result;
        result.execution.requested_output_tokens = 4;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    const ninferno::GenerationResult first =
        engine.generate(engine.prepare(input_with_history(0, true)), options(true));
    if (first.generated_token_ids.size() != 4 ||
        first.prefix_reuse_path != ninferno::PrefixReusePath::Root) {
        std::cerr << "response-checkpoint source request did not complete from a cold lane\n";
        return 1;
    }

    const ninferno::GenerationResult exact_replay =
        engine.generate(engine.prepare(input_with_history(0, false)), options(true));
    if (exact_replay.generated_token_ids.size() != 4 ||
        exact_replay.prefix_reuse_path != ninferno::PrefixReusePath::PrivateResponseReplay ||
        exact_replay.reused_prompt_tokens == 0) {
        const ninferno::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "pre-generation response checkpoint was not restored on an exact replay: "
                  << "path=" << static_cast<int>(exact_replay.prefix_reuse_path)
                  << " reused=" << exact_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    const ninferno::GenerationResult exact_baseline =
        engine.generate(engine.prepare(input_with_history(0, false)), options(false));
    // Replay can rebuild the MTP bridge at T=1; capture can also split the source prefill.
    // Each route must finish within its own budget and publish a valid reuse frontier.
    if (exact_baseline.generated_token_ids.size() != 4 ||
        exact_baseline.prefix_reuse_path != ninferno::PrefixReusePath::Root ||
        exact_baseline.reused_prompt_tokens != 0) {
        std::cerr << "uncached response-checkpoint baseline did not complete from Root: path="
                  << static_cast<int>(exact_baseline.prefix_reuse_path)
                  << " reused=" << exact_baseline.reused_prompt_tokens
                  << " outputs=" << exact_baseline.generated_token_ids.size() << '\n';
        return 1;
    }

    const ninferno::RuntimeStats before_first_replay = engine.runtime_stats();
    const ninferno::GenerationResult first_replay =
        engine.generate(engine.prepare(input_with_history(1, true)), options(true));
    const ninferno::RuntimeStats after_first_replay = engine.runtime_stats();
    const ninferno::PrefixReusePath expected_first_replay =
        shared_alias ? ninferno::PrefixReusePath::SharedStablePrefix
                     : ninferno::PrefixReusePath::PrivateResponseReplay;
    if (first_replay.generated_token_ids.size() != 4 ||
        first_replay.prefix_reuse_path != expected_first_replay ||
        first_replay.reused_prompt_tokens == 0 ||
        (shared_alias && first_replay.reused_prompt_tokens <= exact_replay.reused_prompt_tokens)) {
        std::cerr << "normalized first response selected the wrong cache frontier: path="
                  << static_cast<int>(first_replay.prefix_reuse_path)
                  << " expected=" << static_cast<int>(expected_first_replay)
                  << " reused=" << first_replay.reused_prompt_tokens << '\n';
        return 1;
    }
    if (shared_alias &&
        after_first_replay.historical_fork_hits <= before_first_replay.historical_fork_hits &&
        after_first_replay.state_restores <= before_first_replay.state_restores) {
        std::cerr << "shared rewrite source had no StateImage materialization transition: forks="
                  << before_first_replay.historical_fork_hits << '/'
                  << after_first_replay.historical_fork_hits
                  << " restores=" << before_first_replay.state_restores << '/'
                  << after_first_replay.state_restores << '\n';
        return 1;
    }

    const ninferno::GenerationResult second_replay =
        engine.generate(engine.prepare(input_with_history(2, true)), options(true));
    if (second_replay.generated_token_ids.size() != 4 ||
        second_replay.prefix_reuse_path != ninferno::PrefixReusePath::PrivateResponseReplay ||
        second_replay.reused_prompt_tokens <= first_replay.reused_prompt_tokens) {
        const ninferno::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "rolling response checkpoint did not advance across the tool loop: first="
                  << first_replay.reused_prompt_tokens
                  << " second=" << second_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }

    const ninferno::GenerationResult mode_change =
        engine.generate(engine.prepare(input_with_history(2, false)), options(true));
    if (mode_change.generated_token_ids.size() != 4 ||
        mode_change.prefix_reuse_path != ninferno::PrefixReusePath::PrivateResponseReplay ||
        mode_change.reused_prompt_tokens == 0) {
        std::cerr << "preserve-thinking policy change discarded a compatible response checkpoint: "
                  << "path=" << static_cast<int>(mode_change.prefix_reuse_path)
                  << " reused=" << mode_change.reused_prompt_tokens << '\n';
        return 1;
    }

    if (shared_alias) {
        const ninferno::RuntimeStats final_stats   = engine.runtime_stats();
        const std::uint64_t initial_degradations = initial_stats.pressure_private_owners_degraded +
                                                   initial_stats.pressure_shared_owners_degraded;
        const std::uint64_t final_degradations = final_stats.pressure_private_owners_degraded +
                                                 final_stats.pressure_shared_owners_degraded;
        if (final_degradations <= initial_degradations ||
            final_stats.pressure_private_owners_evicted !=
                initial_stats.pressure_private_owners_evicted ||
            final_stats.pressure_shared_owners_evicted !=
                initial_stats.pressure_shared_owners_evicted ||
            final_stats.pressure_checkpoints_dropped !=
                initial_stats.pressure_checkpoints_dropped ||
            final_stats.active_captures_aborted != initial_stats.active_captures_aborted) {
            std::cerr << "shared/rewrite rotation did not preserve both cache owners: degraded="
                      << initial_degradations << '/' << final_degradations
                      << " private_evicted=" << initial_stats.pressure_private_owners_evicted << '/'
                      << final_stats.pressure_private_owners_evicted
                      << " shared_evicted=" << initial_stats.pressure_shared_owners_evicted << '/'
                      << final_stats.pressure_shared_owners_evicted
                      << " checkpoint_drops=" << initial_stats.pressure_checkpoints_dropped << '/'
                      << final_stats.pressure_checkpoints_dropped
                      << " capture_aborts=" << initial_stats.active_captures_aborted << '/'
                      << final_stats.active_captures_aborted << '\n';
            return 1;
        }
    }

    return 0;
}

int exercise_rewrite_branch(const char* artifact) {
    auto text_message = [](ninferno::ChatRole role, std::string text) {
        ninferno::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    const auto input = [&](bool branch) {
        ninferno::PromptInput value;
        value.messages.push_back(text_message(
            ninferno::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (branch) {
            value.messages.push_back(text_message(ninferno::ChatRole::User,
                                                  "Summarize the conversation before answering."));
        }
        value.options.preserve_thinking = true;
        value.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return value;
    };
    const auto options = [](bool reuse) {
        ninferno::RequestOptions value;
        value.execution.requested_output_tokens = 4;
        value.execution.sampling.temperature    = 0.0F;
        value.execution.allow_prefix_reuse      = reuse;
        value.stop.include_model_defaults       = false;
        return value;
    };

    ninferno::EngineOptions configured             = engine_options(artifact);
    configured.context_cache.device_state_slots  = 2;
    configured.context_cache.max_shared_prefixes = 0;
    ninferno::Engine engine(std::move(configured));
    const ninferno::GenerationResult source =
        engine.generate(engine.prepare(input(false)), options(true));
    if (source.generated_token_ids.size() != 4 ||
        source.prefix_reuse_path != ninferno::PrefixReusePath::Root) {
        std::cerr << "rewrite branch source did not establish a response checkpoint\n";
        return 1;
    }
    const ninferno::GenerationResult branch =
        engine.generate(engine.prepare(input(true)), options(true));
    const ninferno::GenerationResult branch_baseline =
        engine.generate(engine.prepare(input(true)), options(false));
    if (branch.generated_token_ids.size() != 4 || branch.reused_prompt_tokens == 0 ||
        branch.reused_prompt_tokens >= branch.prompt.prompt_tokens ||
        (branch.prefix_reuse_path != ninferno::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninferno::PrefixReusePath::PrivateTurnClosure) ||
        branch_baseline.generated_token_ids.size() != 4 ||
        branch_baseline.reused_prompt_tokens != 0) {
        std::cerr << "replacement user suffix did not reuse the stable conversation prefix: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens
                  << " prompt=" << branch.prompt.prompt_tokens
                  << " target_count=" << branch.materialization.targets_evaluated
                  << " degradation=" << branch.materialization.selected_degradation_units
                  << " maximal=" << branch.materialization.selected_maximal_fallback << " stop="
                  << ninferno::materialization_stop_reason_name(branch.materialization.stop_reason)
                  << " now_ns=" << branch.materialization.predicted_now_ns
                  << " future_ns=" << branch.materialization.predicted_future_loss_ns << '\n';
        return 1;
    }
    return 0;
}

int exercise_vision(ninferno::Engine& engine) {
    const auto image_bytes = gradient_ppm();
    auto image_part        = [](const std::vector<std::uint8_t>& bytes, std::string name) {
        ninferno::MessagePart image;
        image.kind              = ninferno::MessagePartKind::Media;
        image.media.kind        = ninferno::MediaKind::Image;
        image.media.bytes       = bytes;
        image.media.media_type  = "image/x-portable-pixmap";
        image.media.source_name = std::move(name);
        return image;
    };
    auto assistant_message = [](const ninferno::GenerationResult& result) {
        ninferno::ChatMessage message;
        message.role              = ninferno::ChatRole::Assistant;
        message.reasoning_content = result.reasoning;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = result.content, .media = {}});
        return message;
    };
    auto first_input = [&](const std::vector<std::uint8_t>& bytes) {
        ninferno::ChatMessage message;
        message.role = ninferno::ChatRole::User;
        message.parts.push_back(image_part(bytes, "inline.ppm"));
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = "What is visible?", .media = {}});
        ninferno::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "vision-prefix-real";
        input.context_cache.retention   = ninferno::CacheRetentionHint::LiveSession;
        return input;
    };
    auto followup_input = [&](const std::vector<std::uint8_t>& bytes,
                              const ninferno::GenerationResult& first) {
        ninferno::PromptInput input = first_input(bytes);
        input.messages.push_back(assistant_message(first));
        ninferno::ChatMessage followup;
        followup.role = ninferno::ChatRole::User;
        followup.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = "Give one more detail.", .media = {}});
        input.messages.push_back(std::move(followup));
        return input;
    };
    auto appended_media_input =
        [&](const std::vector<std::uint8_t>& old_bytes, const ninferno::GenerationResult& first,
            const ninferno::GenerationResult& second, const std::vector<std::uint8_t>& new_bytes) {
            ninferno::PromptInput input = followup_input(old_bytes, first);
            input.messages.push_back(assistant_message(second));
            ninferno::ChatMessage followup;
            followup.role = ninferno::ChatRole::User;
            followup.parts.push_back(image_part(new_bytes, "second.ppm"));
            followup.parts.push_back(ninferno::MessagePart{.kind  = ninferno::MessagePartKind::Text,
                                                           .text  = "Compare the images.",
                                                           .media = {}});
            input.messages.push_back(std::move(followup));
            return input;
        };

    auto options = [](bool reuse) {
        ninferno::RequestOptions result;
        result.execution.requested_output_tokens = 2;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    // The 1024 merged Vision columns begin after the chat prefix, so the same item necessarily
    // crosses a 1024-token prefill boundary. Its host payload may be released after the first
    // encode, while later chunks must continue to reuse the resident Vision transient.
    ninferno::RequestOptions cross_chunk_options            = options(false);
    cross_chunk_options.execution.requested_output_tokens = 1;
    const ninferno::GenerationResult cross_chunk =
        engine.generate(engine.prepare(first_input(gradient_ppm(1024, 1024))), cross_chunk_options);
    if (!cross_chunk.prompt.has_media || cross_chunk.generated_token_ids.size() != 1) {
        std::cerr << "cross-chunk Vision item did not complete after releasing its host payload\n";
        return 1;
    }

    const ninferno::GenerationResult first =
        engine.generate(engine.prepare(first_input(image_bytes)), options(true));
    if (!first.prompt.has_media || first.generated_token_ids.size() != 2 ||
        first.finish_reason != ninferno::FinishReason::OutputLimit) {
        std::cerr << "real Vision request did not complete through the public Engine\n";
        return 1;
    }

    const ninferno::GenerationResult reused =
        engine.generate(engine.prepare(followup_input(image_bytes, first)), options(true));
    if (reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0.0 ||
        reused.generated_token_ids.size() != 2) {
        std::cerr << "same-media continuation did not reuse the resident Vision prefix: reused="
                  << reused.reused_prompt_tokens << " vision=" << reused.timings.vision_seconds
                  << '\n';
        return 1;
    }
    // The reused continuation prefills a text-only suffix after the cached multimodal prefix: its
    // positions keep the image's RoPE delta, so it decodes what an uncached prefill decodes.
    // A wide 512x256 image leaves a large delta (its merged grid is far from square in tokens).
    {
        const auto wide = gradient_ppm(512, 256);
        ninferno::RequestOptions long_reuse = options(true);
        ninferno::RequestOptions long_fresh = options(false);
        long_reuse.execution.requested_output_tokens = 16;
        long_fresh.execution.requested_output_tokens = 16;
        const ninferno::GenerationResult wide_first =
            engine.generate(engine.prepare(first_input(wide)), options(true));
        const ninferno::GenerationResult cached =
            engine.generate(engine.prepare(followup_input(wide, wide_first)), long_reuse);
        const ninferno::GenerationResult fresh =
            engine.generate(engine.prepare(followup_input(wide, wide_first)), long_fresh);
        if (cached.reused_prompt_tokens == 0 || cached.timings.vision_seconds != 0.0 ||
            cached.generated_token_ids != fresh.generated_token_ids) {
            std::cerr << "a text-only suffix after a cached multimodal prefix decoded unlike an "
                         "uncached prefill (RoPE delta lost?): reused="
                      << cached.reused_prompt_tokens << '\n';
            return 1;
        }
    }

    std::vector<std::uint8_t> second_image = image_bytes;
    second_image.back() ^= 0x5aU;
    const ninferno::GenerationResult appended = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(true));
    if (appended.reused_prompt_tokens == 0 || !(appended.timings.vision_seconds > 0.0) ||
        appended.generated_token_ids.size() != 2) {
        std::cerr << "new-media suffix did not preserve the old multimodal prefix: reused="
                  << appended.reused_prompt_tokens << " vision=" << appended.timings.vision_seconds
                  << '\n';
        return 1;
    }

    const ninferno::GenerationResult baseline = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(false));
    if (baseline.generated_token_ids.size() != 2 || baseline.reused_prompt_tokens != 0 ||
        !(baseline.timings.vision_seconds > 0.0)) {
        std::cerr << "uncached multimodal prefill did not recompute its media\n";
        return 1;
    }

    std::vector<std::uint8_t> changed_prefix = image_bytes;
    changed_prefix[changed_prefix.size() - 2] ^= 0x33U;
    const ninferno::GenerationResult miss = engine.generate(
        engine.prepare(appended_media_input(changed_prefix, first, reused, second_image)),
        options(true));
    if (miss.reused_prompt_tokens != 0) {
        std::cerr << "changed media content incorrectly reused placeholder-token KV\n";
        return 1;
    }

    ninferno::RequestOptions mtp_options            = options(false);
    mtp_options.execution.requested_output_tokens = 5;
    const ninferno::GenerationResult mtp_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), mtp_options);
    if (mtp_baseline.generated_token_ids.size() != 5 ||
        mtp_baseline.generated_token_ids[0] == mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal stop fixture did not produce distinct leading tokens\n";
        return 1;
    }
    ninferno::RequestOptions stop_options       = mtp_options;
    stop_options.execution.allow_prefix_reuse = true;
    stop_options.stop.token_ids.push_back(mtp_baseline.generated_token_ids[1]);
    const ninferno::GenerationResult stopped =
        engine.generate(engine.prepare(first_input(image_bytes)), stop_options);
    if (stopped.finish_reason != ninferno::FinishReason::StopToken ||
        stopped.generated_token_ids.empty() || stopped.speculative.rounds == 0 ||
        stopped.generated_token_ids.back() != stop_options.stop.token_ids.front()) {
        std::cerr << "multimodal custom stop did not terminate at the selected token\n";
        return 1;
    }
    const ninferno::GenerationResult stopped_reuse =
        engine.generate(engine.prepare(followup_input(image_bytes, stopped)), options(true));
    if (stopped_reuse.reused_prompt_tokens == 0 || stopped_reuse.timings.vision_seconds != 0.0) {
        std::cerr << "multimodal stop discarded its reusable boundary: reused="
                  << stopped_reuse.reused_prompt_tokens
                  << " vision=" << stopped_reuse.timings.vision_seconds << '\n';
        return 1;
    }

    // Exact registered rendering prefix before the first image-pad column:
    // <|im_start|>user\n<|vision_start|>. Reusing it places the MTP bridge directly on the first
    // Vision merger column rather than on an ordinary token embedding.
    const std::vector<ninferno::TokenId> visual_prefix{248045, 846, 198, 248053};
    ninferno::RequestOptions source_options            = options(true);
    source_options.execution.requested_output_tokens = 1;
    const ninferno::GenerationResult bridge_source =
        engine.generate(engine.prepare_tokens(visual_prefix), source_options);
    ninferno::RequestOptions bridge_options            = options(true);
    bridge_options.execution.requested_output_tokens = 5;
    const ninferno::GenerationResult visual_bridge =
        engine.generate(engine.prepare(first_input(image_bytes)), bridge_options);
    if (bridge_source.generated_token_ids.size() != 1 ||
        visual_bridge.reused_prompt_tokens != visual_prefix.size() ||
        !(visual_bridge.timings.vision_seconds > 0.0) || visual_bridge.speculative.rounds == 0) {
        std::cerr << "visual MTP bridge did not append the prefix and enter speculative decode: "
                  << "source_outputs=" << bridge_source.generated_token_ids.size()
                  << " reused=" << visual_bridge.reused_prompt_tokens
                  << " vision=" << visual_bridge.timings.vision_seconds
                  << " rounds=" << visual_bridge.speculative.rounds
                  << " fallbacks=" << visual_bridge.speculative.fallback_steps << '\n';
        return 1;
    }
    if (visual_bridge.generated_token_ids.size() != 5 ||
        visual_bridge.finish_reason != ninferno::FinishReason::OutputLimit) {
        std::cerr << "visual MTP bridge did not commit the requested output budget\n";
        return 1;
    }
    const auto bridge_followup =
        engine.generate(engine.prepare(followup_input(image_bytes, visual_bridge)), options(true));
    if (bridge_followup.reused_prompt_tokens == 0 ||
        bridge_followup.timings.vision_seconds != 0.0 ||
        bridge_followup.generated_token_ids.size() != 2) {
        std::cerr << "visual MTP bridge lost its retained continuation\n";
        return 1;
    }
    return 0;
}

ninferno::PromptInput session_turn(std::string session, std::string question) {
    ninferno::PromptInput input;
    ninferno::ChatMessage user;
    user.role = ninferno::ChatRole::User;
    user.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = std::move(question), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = std::move(session);
    input.context_cache.retention   = ninferno::CacheRetentionHint::LiveSession;
    return input;
}

ninferno::PromptInput pressure_turn(std::string text, std::string session,
                                  ninferno::CacheRetentionHint retention) {
    ninferno::PromptInput input;
    ninferno::ChatMessage user;
    user.role = ninferno::ChatRole::User;
    user.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    if (!session.empty()) { input.context_cache.session_key = std::move(session); }
    input.context_cache.retention = retention;
    return input;
}

std::optional<std::string> exact_repeated_prompt_text(const ninferno::Engine& engine,
                                                      std::uint32_t target_tokens,
                                                      std::string_view word) {
    const auto text = [word](std::uint32_t repetitions) {
        std::string value;
        value.reserve(static_cast<std::size_t>(repetitions) * (word.size() + 1U));
        for (std::uint32_t index = 0; index < repetitions; ++index) {
            value.push_back(' ');
            value.append(word);
        }
        return value;
    };
    const auto count = [&](std::uint32_t repetitions) {
        return engine.count_tokens(
            pressure_turn(text(repetitions), "", ninferno::CacheRetentionHint::Disposable));
    };

    std::uint32_t low  = 0;
    std::uint32_t high = target_tokens;
    while (low <= high) {
        const std::uint32_t middle = low + (high - low) / 2U;
        const std::uint32_t tokens = count(middle);
        if (tokens == target_tokens) { return text(middle); }
        if (tokens < target_tokens) {
            low = middle + 1U;
        } else {
            if (middle == 0) { break; }
            high = middle - 1U;
        }
    }
    return std::nullopt;
}

ninferno::RequestOptions fixed_output(std::uint32_t tokens, bool reuse = true) {
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int exercise_pressure_partial_spill_and_resume(const char* artifact) {
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 31;
    constexpr std::uint32_t kShortPromptTokens = 350;
    ninferno::Engine engine(pressure_resume_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_a_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    const std::optional<std::string> short_b_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "charlie");
    if (!long_text || !short_a_text || !short_b_text) {
        std::cerr << "pressure-resume fixture could not construct exact prompt geometry\n";
        return 1;
    }

    const ninferno::GenerationResult long_result = engine.generate(
        engine.prepare(pressure_turn(*long_text, "", ninferno::CacheRetentionHint::Disposable)),
        fixed_output(kLongOutputTokens));
    if (long_result.prompt.prompt_tokens != kLongPromptTokens ||
        long_result.generated_token_ids.size() != kLongOutputTokens) {
        std::cerr << "pressure-resume long source did not establish its 121-page endpoint: prompt="
                  << long_result.prompt.prompt_tokens
                  << " output=" << long_result.generated_token_ids.size() << '\n';
        return 1;
    }

    const ninferno::GenerationResult short_a =
        engine.generate(engine.prepare(pressure_turn(*short_a_text, "pressure-short-a",
                                                     ninferno::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    if (short_a.prompt.prompt_tokens != kShortPromptTokens ||
        short_a.generated_token_ids.size() != 1) {
        std::cerr << "pressure-resume short source did not establish its six-page reservation\n";
        return 1;
    }

    const ninferno::RuntimeStats before_pressure = engine.runtime_stats();
    const ninferno::GenerationResult short_b =
        engine.generate(engine.prepare(pressure_turn(*short_b_text, "pressure-short-b",
                                                     ninferno::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    const ninferno::RuntimeStats after_pressure = engine.runtime_stats();
    const std::uint64_t pressure_main_pages =
        after_pressure.main_kv_d2h_pages - before_pressure.main_kv_d2h_pages;
    const std::uint64_t pressure_spill_pages =
        after_pressure.pressure_spill_pages - before_pressure.pressure_spill_pages;
    const std::uint64_t pressure_drops =
        after_pressure.pressure_checkpoints_dropped - before_pressure.pressure_checkpoints_dropped;
    const std::uint64_t pressure_degraded = after_pressure.pressure_private_owners_degraded -
                                            before_pressure.pressure_private_owners_degraded;
    const std::uint64_t pressure_evicted = after_pressure.pressure_private_owners_evicted -
                                           before_pressure.pressure_private_owners_evicted;
    const std::uint64_t pressure_captures =
        after_pressure.active_captures_completed - before_pressure.active_captures_completed;
    // short_b's materialization drops the long source's endpoint and spills four pages (one owner
    // degraded). Its own turn closure then fits no free State slot: by the capture rule it takes
    // the slot of short_a's endpoint, a one-token duplicate of short_a's retained turn closure (a
    // second owner degraded, a second checkpoint dropped); nothing is evicted.
    if (short_b.generated_token_ids.size() != 1 || pressure_main_pages != 4 ||
        pressure_spill_pages != 4 || pressure_drops != 2 || pressure_degraded != 2 ||
        pressure_captures != 1 || pressure_evicted != 0 ||
        after_pressure.state_d2h_count != before_pressure.state_d2h_count ||
        short_b.materialization.selected_maximal_fallback) {
        std::cerr << "pressure-resume did not select endpoint-drop plus four-page spill: main="
                  << pressure_main_pages << " spill=" << pressure_spill_pages
                  << " drops=" << pressure_drops << " degraded=" << pressure_degraded
                  << " captures=" << pressure_captures << " evicted=" << pressure_evicted
                  << " state=" << (after_pressure.state_d2h_count - before_pressure.state_d2h_count)
                  << " device_pages=" << before_pressure.device_main_kv_occupied_pages << '/'
                  << after_pressure.device_main_kv_occupied_pages
                  << " maximal=" << short_b.materialization.selected_maximal_fallback
                  << " budget=" << short_b.materialization.budget_exhausted << '\n';
        return 1;
    }
    const ninferno::RuntimeStats before_resume = engine.runtime_stats();
    const ninferno::GenerationResult resumed   = engine.generate(
        engine.prepare(pressure_turn(*long_text, "", ninferno::CacheRetentionHint::Disposable)),
        fixed_output(1));
    const ninferno::RuntimeStats after_resume = engine.runtime_stats();
    const std::uint64_t restored_pages =
        after_resume.main_kv_h2d_pages - before_resume.main_kv_h2d_pages;
    const std::uint32_t reused_pages = (resumed.reused_prompt_tokens + 63U) / 64U;
    if (resumed.generated_token_ids.size() != 1 ||
        resumed.prefix_reuse_path != ninferno::PrefixReusePath::PrivateTurnClosure ||
        reused_pages != 120 || restored_pages != 4) {
        std::cerr << "pressure-resume did not restore the retained turn closure: path="
                  << static_cast<int>(resumed.prefix_reuse_path)
                  << " reused=" << resumed.reused_prompt_tokens << " reused_pages=" << reused_pages
                  << " restored=" << restored_pages << '\n';
        return 1;
    }
    // Both short sessions kept their turn closures: short_a's survived short_b's capture, and
    // short_b's was captured under pressure instead of being skipped.
    for (const auto& [text, session] :
         {std::pair<const std::string*, const char*>{&*short_a_text, "pressure-short-a"},
          std::pair<const std::string*, const char*>{&*short_b_text, "pressure-short-b"}}) {
        const ninferno::GenerationResult again =
            engine.generate(engine.prepare(pressure_turn(
                                *text, session, ninferno::CacheRetentionHint::LiveSession)),
                            fixed_output(1));
        if (again.generated_token_ids.size() != 1 ||
            again.prefix_reuse_path != ninferno::PrefixReusePath::PrivateTurnClosure ||
            again.reused_prompt_tokens == 0 ||
            again.reused_prompt_tokens >= again.prompt.prompt_tokens) {
            std::cerr << "pressure-resume lost the turn closure of " << session
                      << ": path=" << static_cast<int>(again.prefix_reuse_path)
                      << " reused=" << again.reused_prompt_tokens << '\n';
            return 1;
        }
    }
    return 0;
}

int exercise_materialization_source_pressure_protection(const char* artifact) {
    // The source occupies 121 pages and the second owner occupies six, leaving one free page. The
    // branch reuses the source's 120-page turn closure but needs two suffix pages. Under the old
    // guided closure, the source's unprotected endpoint tail was selected as the one-page Host KV
    // victim even though the same continuation was the materialization source.
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 31;
    constexpr std::uint32_t kShortPromptTokens = 350;
    ninferno::Engine engine(pressure_resume_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    if (!long_text || !short_text) {
        std::cerr << "source-pressure fixture could not construct exact prompt geometry\n";
        return 1;
    }

    const ninferno::GenerationResult source =
        engine.generate(engine.prepare(pressure_turn(*long_text, "source-pressure-origin",
                                                     ninferno::CacheRetentionHint::LiveSession)),
                        fixed_output(kLongOutputTokens));
    const ninferno::GenerationResult resident =
        engine.generate(engine.prepare(pressure_turn(*short_text, "source-pressure-resident",
                                                     ninferno::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    const ninferno::RuntimeStats before_branch = engine.runtime_stats();
    if (source.prompt.prompt_tokens != kLongPromptTokens ||
        source.generated_token_ids.size() != kLongOutputTokens ||
        resident.prompt.prompt_tokens != kShortPromptTokens ||
        resident.generated_token_ids.size() != 1 ||
        before_branch.device_main_kv_occupied_pages != 127) {
        std::cerr << "source-pressure fixture did not establish 127 resident pages: source="
                  << source.prompt.prompt_tokens << '+' << source.generated_token_ids.size()
                  << " resident=" << resident.prompt.prompt_tokens << '+'
                  << resident.generated_token_ids.size()
                  << " pages=" << before_branch.device_main_kv_occupied_pages << '\n';
        return 1;
    }

    ninferno::PromptInput branch = pressure_turn(*long_text, "source-pressure-branch",
                                               ninferno::CacheRetentionHint::LiveSession);
    std::string suffix;
    for (std::uint32_t index = 0; index < 96; ++index) { suffix += " delta"; }
    ninferno::ChatMessage followup;
    followup.role = ninferno::ChatRole::User;
    followup.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = std::move(suffix), .media = {}});
    branch.messages.push_back(std::move(followup));

    const ninferno::GenerationResult branched =
        engine.generate(engine.prepare(std::move(branch)), fixed_output(1));
    const ninferno::RuntimeStats after_branch = engine.runtime_stats();
    const std::uint64_t demoted_pages =
        after_branch.main_kv_d2h_pages - before_branch.main_kv_d2h_pages;
    const std::uint64_t degraded = after_branch.pressure_private_owners_degraded -
                                   before_branch.pressure_private_owners_degraded;
    const bool private_partial_source =
        branched.prefix_reuse_path == ninferno::PrefixReusePath::PrivateResponseReplay ||
        branched.prefix_reuse_path == ninferno::PrefixReusePath::PrivateTurnClosure;
    if (branched.generated_token_ids.size() != 1 || !private_partial_source ||
        branched.reused_prompt_tokens == 0 ||
        branched.reused_prompt_tokens >= branched.prompt.prompt_tokens || demoted_pages == 0 ||
        degraded == 0 || branched.materialization.selected_maximal_fallback) {
        std::cerr << "source-pressure branch did not preserve its source under guided Host KV "
                     "pressure: path="
                  << static_cast<int>(branched.prefix_reuse_path)
                  << " reused=" << branched.reused_prompt_tokens
                  << " prompt=" << branched.prompt.prompt_tokens << " demoted=" << demoted_pages
                  << " degraded=" << degraded
                  << " maximal=" << branched.materialization.selected_maximal_fallback << " stop="
                  << ninferno::materialization_stop_reason_name(
                         branched.materialization.stop_reason)
                  << '\n';
        return 1;
    }
    return 0;
}

int exercise_private_checkpoint_pressure_retention(const char* artifact) {
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 16;
    constexpr std::uint32_t kShortPromptTokens = 350;
    constexpr std::uint32_t kShortOutputTokens = 256;
    ninferno::Engine engine(private_checkpoint_pressure_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_b_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    const std::optional<std::string> short_c_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "charlie");
    if (!long_text || !short_b_text || !short_c_text) {
        std::cerr << "private-checkpoint pressure fixture could not construct prompt geometry\n";
        return 1;
    }

    const std::string session             = "private-checkpoint-pressure-source";
    const ninferno::GenerationResult source =
        engine.generate(engine.prepare(pressure_turn(*long_text, session,
                                                     ninferno::CacheRetentionHint::LiveSession)),
                        fixed_output(kLongOutputTokens));
    if (source.prompt.prompt_tokens != kLongPromptTokens ||
        source.generated_token_ids.size() != kLongOutputTokens) {
        std::cerr << "private-checkpoint pressure source did not establish its long session\n";
        return 1;
    }

    const ninferno::RuntimeStats before_pressure = engine.runtime_stats();
    auto short_b                               = engine.submit(
        engine.prepare(pressure_turn(*short_b_text, "", ninferno::CacheRetentionHint::Disposable)),
        fixed_output(kShortOutputTokens));
    auto short_c = engine.submit(
        engine.prepare(pressure_turn(*short_c_text, "", ninferno::CacheRetentionHint::Disposable)),
        fixed_output(kShortOutputTokens));
    const ninferno::GenerationResult short_b_result = short_b.wait();
    const ninferno::GenerationResult short_c_result = short_c.wait();
    const ninferno::RuntimeStats after_pressure     = engine.runtime_stats();
    if (short_b_result.generated_token_ids.size() != kShortOutputTokens ||
        short_c_result.generated_token_ids.size() != kShortOutputTokens ||
        after_pressure.pressure_checkpoints_dropped <=
            before_pressure.pressure_checkpoints_dropped ||
        after_pressure.pressure_private_owners_degraded <=
            before_pressure.pressure_private_owners_degraded ||
        after_pressure.pressure_private_owners_evicted !=
            before_pressure.pressure_private_owners_evicted) {
        std::cerr << "private-checkpoint pressure did not produce a retained checkpoint "
                     "degradation: drops="
                  << before_pressure.pressure_checkpoints_dropped << '/'
                  << after_pressure.pressure_checkpoints_dropped
                  << " degraded=" << before_pressure.pressure_private_owners_degraded << '/'
                  << after_pressure.pressure_private_owners_degraded
                  << " evicted=" << before_pressure.pressure_private_owners_evicted << '/'
                  << after_pressure.pressure_private_owners_evicted << '\n';
        return 1;
    }

    ninferno::PromptInput resume =
        pressure_turn(*long_text, session, ninferno::CacheRetentionHint::LiveSession);
    ninferno::ChatMessage assistant;
    assistant.role              = ninferno::ChatRole::Assistant;
    assistant.reasoning_content = source.reasoning;
    assistant.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = source.content, .media = {}});
    resume.messages.push_back(std::move(assistant));
    ninferno::ChatMessage followup;
    followup.role = ninferno::ChatRole::User;
    followup.parts.push_back(ninferno::MessagePart{
        .kind  = ninferno::MessagePartKind::Text,
        .text  = "Return the retained answer in one line.",
        .media = {},
    });
    resume.messages.push_back(std::move(followup));
    const ninferno::GenerationResult resumed =
        engine.generate(engine.prepare(std::move(resume)), fixed_output(1));
    if (resumed.generated_token_ids.size() != 1 ||
        resumed.prefix_reuse_path != ninferno::PrefixReusePath::PrivateTurnClosure ||
        resumed.reused_prompt_tokens == 0) {
        std::cerr << "private checkpoint pressure discarded the reusable turn closure: path="
                  << static_cast<int>(resumed.prefix_reuse_path)
                  << " reused=" << resumed.reused_prompt_tokens
                  << " future_ns=" << resumed.materialization.predicted_future_loss_ns << '\n';
        return 1;
    }
    return 0;
}

// The path a follow-up turn resumes its session's private lineage by: the endpoint when the
// template renders the previous answer exactly as generated (Qwen3.8), otherwise the turn closure
// before it (the template renders the answer differently from the generated tokens).
std::optional<ninferno::PrefixReusePath> session_resume_path(const ninferno::Engine& engine) {
    const std::optional<TemplateFamily> family = registered_template_family(engine);
    if (!family) { return std::nullopt; }
    return *family == TemplateFamily::Qwen3_8 ? ninferno::PrefixReusePath::PrivateEndpoint
                                              : ninferno::PrefixReusePath::PrivateTurnClosure;
}

ninferno::PromptInput followup_turn(ninferno::PromptInput turn,
                                    const ninferno::GenerationResult& answer,
                                    std::string question) {
    ninferno::ChatMessage assistant;
    assistant.role              = ninferno::ChatRole::Assistant;
    assistant.reasoning_content = answer.reasoning;
    assistant.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = answer.content, .media = {}});
    turn.messages.push_back(std::move(assistant));
    ninferno::ChatMessage user;
    user.role = ninferno::ChatRole::User;
    user.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = std::move(question), .media = {}});
    turn.messages.push_back(std::move(user));
    return turn;
}

// C=4 with a Host tier and no cached Device states: four sessions submitted together must all be
// admitted before any of them finishes. A prompt checkpoint captured on Device while the others
// wait would hold the image the last one needs (active lineages are not pressure victims), so
// the fourth would wait for the first to finish. The checkpoints still serve the next turn.
int exercise_concurrent_wave_host_tier_admission(const char* artifact) {
    constexpr std::uint32_t kRequests     = 4;
    constexpr std::uint32_t kOutputTokens = 96;
    ninferno::Engine engine(host_tier_no_cached_state_options(artifact, kRequests));

    const std::array<std::string, kRequests> questions{
        "Explain optimistic and pessimistic locking in one paragraph.",
        "Summarize the trade-offs between REST and gRPC in one paragraph.",
        "Describe how a B-tree index speeds up range queries in one paragraph.",
        "Explain why attention cost grows quadratically with sequence length.",
    };
    std::vector<ninferno::PreparedPrompt> prepared;
    prepared.reserve(kRequests);
    for (std::uint32_t row = 0; row < kRequests; ++row) {
        prepared.push_back(engine.prepare(
            session_turn("host-tier-wave-" + std::to_string(row), questions[row])));
    }
    std::vector<ninferno::GenerationHandle> handles;
    handles.reserve(kRequests);
    for (ninferno::PreparedPrompt& request : prepared) {
        handles.push_back(engine.submit(std::move(request), fixed_output(kOutputTokens)));
    }
    std::vector<ninferno::GenerationResult> results;
    results.reserve(kRequests);
    for (ninferno::GenerationHandle& handle : handles) { results.push_back(handle.wait()); }

    double last_admission = 0.0;
    double first_finish   = std::numeric_limits<double>::max();
    for (const ninferno::GenerationResult& result : results) {
        if (result.generated_token_ids.size() != kOutputTokens) {
            std::cerr << "host-tier wave request did not reach its output limit\n";
            return 1;
        }
        last_admission = std::max(last_admission, result.engine_timing.queue_wait_seconds);
        first_finish   = std::min(first_finish,
                                  result.timings.total_seconds - result.timings.prepare_seconds);
    }
    if (last_admission >= first_finish) {
        const ninferno::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "host-tier wave admitted a request only after another finished: last "
                     "admission="
                  << last_admission << " s first finish=" << first_finish
                  << " s device_states=" << stats.device_state_occupied_slots
                  << " host_states=" << stats.host_state_occupied_slots << '\n';
        return 1;
    }

    const ninferno::GenerationResult next = engine.generate(
        engine.prepare(followup_turn(session_turn("host-tier-wave-0", questions[0]), results[0],
                                     "Now give the same answer in one sentence.")),
        fixed_output(4));
    const std::optional<ninferno::PrefixReusePath> resume = session_resume_path(engine);
    if (next.generated_token_ids.size() != 4 || !resume || next.prefix_reuse_path != *resume ||
        next.reused_prompt_tokens == 0) {
        std::cerr << "host-tier wave session lost its private checkpoint: path="
                  << static_cast<int>(next.prefix_reuse_path)
                  << " reused=" << next.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

// C=2 with a Host tier and no cached Device states under light load (one session, nothing
// waiting): the turn closure is still captured on Device (no StateImage Device-to-Host copy) and
// the next turn reuses it without a Host-to-Device restore.
int exercise_light_load_host_capture_reuse(const char* artifact) {
    ninferno::Engine engine(host_tier_no_cached_state_options(artifact, 2));
    const std::string session  = "light-load-session";
    const std::string question = "Explain the CAP theorem with one concrete example.";

    const ninferno::RuntimeStats before_first = engine.runtime_stats();
    const ninferno::GenerationResult first =
        engine.generate(engine.prepare(session_turn(session, question)), fixed_output(32));
    const ninferno::RuntimeStats after_first = engine.runtime_stats();
    if (first.generated_token_ids.size() != 32 ||
        after_first.active_captures_completed <= before_first.active_captures_completed ||
        after_first.state_d2h_count != before_first.state_d2h_count) {
        std::cerr << "light-load turn closure was not captured on Device: captures="
                  << before_first.active_captures_completed << '/'
                  << after_first.active_captures_completed
                  << " state_d2h=" << before_first.state_d2h_count << '/'
                  << after_first.state_d2h_count << '\n';
        return 1;
    }

    const ninferno::GenerationResult next = engine.generate(
        engine.prepare(followup_turn(session_turn(session, question), first,
                                     "Give a second example in one sentence.")),
        fixed_output(4));
    const ninferno::RuntimeStats after_next = engine.runtime_stats();
    const std::optional<ninferno::PrefixReusePath> resume = session_resume_path(engine);
    if (next.generated_token_ids.size() != 4 || !resume || next.prefix_reuse_path != *resume ||
        next.reused_prompt_tokens == 0 ||
        after_next.state_h2d_count != after_first.state_h2d_count) {
        std::cerr << "light-load private checkpoint was not reused from Device: path="
                  << static_cast<int>(next.prefix_reuse_path)
                  << " reused=" << next.reused_prompt_tokens
                  << " state_h2d=" << after_first.state_h2d_count << '/'
                  << after_next.state_h2d_count << '\n';
        return 1;
    }
    return 0;
}

int exercise_concurrent_resource_settlement(const char* artifact) {
    ninferno::Engine engine(concurrent_engine_options(artifact));

    constexpr std::string_view kSession = "publication-order-real";
    constexpr std::string_view kOlderQuestion =
        "Describe deterministic scheduling using exactly one concise paragraph.";
    constexpr std::string_view kNewerQuestion =
        "Describe prefix caching using exactly one concise paragraph.";
    auto older = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kOlderQuestion))),
        fixed_output(24));
    auto newer = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    const ninferno::GenerationResult newer_result = newer.wait();
    const ninferno::GenerationResult older_result = older.wait();
    if (newer_result.generated_token_ids.size() != 2 ||
        older_result.generated_token_ids.size() != 24) {
        std::cerr << "concurrent session requests did not reach staggered terminal boundaries\n";
        return 1;
    }

    for (std::uint32_t index = 0; index < 6; ++index) {
        const std::string suffix              = std::to_string(index);
        const ninferno::GenerationResult filler = engine.generate(
            engine.prepare(session_turn("publication-filler-" + suffix,
                                        "Give one deterministic token for filler " + suffix + '.')),
            fixed_output(1));
        if (filler.generated_token_ids.size() != 1) {
            std::cerr << "session-order catalog filler did not complete\n";
            return 1;
        }
    }
    const ninferno::RuntimeStats before_pressure = engine.runtime_stats();
    const ninferno::GenerationResult pressure    = engine.generate(
        engine.prepare(session_turn("publication-pressure",
                                       "Give one deterministic token for the pressure request.")),
        fixed_output(1));
    const ninferno::RuntimeStats after_pressure = engine.runtime_stats();
    if (pressure.generated_token_ids.size() != 1 ||
        after_pressure.pressure_private_owners_evicted <=
            before_pressure.pressure_private_owners_evicted) {
        std::cerr << "full session catalog did not execute its canonical eviction\n";
        return 1;
    }

    const ninferno::GenerationResult replay = engine.generate(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    if (replay.generated_token_ids.size() != 2 || replay.reused_prompt_tokens == 0 ||
        replay.prefix_reuse_path == ninferno::PrefixReusePath::Root) {
        std::cerr << "late older finish exposed the newer session binding to pressure: path="
                  << static_cast<int>(replay.prefix_reuse_path)
                  << " reused=" << replay.reused_prompt_tokens << '\n';
        return 1;
    }

    {
        std::vector<ninferno::TokenId> long_prompt(400, 198);
        auto cancelled =
            engine.submit(engine.prepare_tokens(std::move(long_prompt)), fixed_output(32, false));
        if (!cancelled) {
            std::cerr << "materialization cancellation fixture did not create a handle\n";
            return 1;
        }
    }
    const ninferno::GenerationResult after_cancel = engine.generate(
        engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198}), fixed_output(1, false));
    if (after_cancel.generated_token_ids.size() != 1) {
        std::cerr << "request after materialization cancellation did not complete\n";
        return 1;
    }

    std::vector<ninferno::GenerationHandle> handles;
    handles.reserve(8);
    for (std::uint32_t row = 0; row < 8; ++row) {
        std::vector<ninferno::TokenId> prompt{
            248045, 846, 198, static_cast<ninferno::TokenId>(1000 + row), 248046, 198};
        handles.push_back(
            engine.submit(engine.prepare_tokens(std::move(prompt)), fixed_output(row + 1, false)));
    }
    for (std::uint32_t row = 0; row < handles.size(); ++row) {
        const ninferno::GenerationResult result = handles[row].wait();
        if (result.generated_token_ids.size() != row + 1 ||
            result.finish_reason != ninferno::FinishReason::OutputLimit) {
            std::cerr << "C=8 staggered row " << row << " did not terminate independently\n";
            return 1;
        }
    }
    const ninferno::RuntimeStats settled = engine.runtime_stats();
    if (settled.running_requests != 0 || settled.materializing_requests != 0 ||
        settled.prefilling_requests != 0 || settled.decode_ready_requests != 0 ||
        settled.capture_pending_requests != 0 || settled.terminal_pending_requests != 0) {
        std::cerr << "C=8 terminal settlement left live logical membership: running="
                  << settled.running_requests << " materializing=" << settled.materializing_requests
                  << " prefill=" << settled.prefilling_requests
                  << " decode=" << settled.decode_ready_requests
                  << " capture=" << settled.capture_pending_requests
                  << " terminal=" << settled.terminal_pending_requests << '\n';
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninferno::Engine& engine, bool with_vision) {
    const ninferno::LoadSummary load = engine.load_summary();
    // A Qwen3.5-family artifact loads as the dense or the mixture-of-experts text model.
    if ((load.architecture != "Qwen3_5ForCausalLM" &&
         load.architecture != "Qwen3_5MoeForCausalLM") ||
        load.model_name.empty() ||
        load.weight_formats.empty() || load.host_to_device_bytes == 0 ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.architecture
                  << " weights=" << load.prefill_signature << '\n';
        return 1;
    }
    const ninferno::MemorySummary memory = engine.memory_summary();
    const auto* vision = memory.vision_workspace ? &*memory.vision_workspace : nullptr;
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || (vision != nullptr) != with_vision ||
        (vision != nullptr &&
         (vision->aggregate_prompt_tokens != 4096 || vision->max_item_tokens != 4096 ||
          vision->general_capacity_bytes == 0 || vision->encode_peak_bytes == 0 ||
          vision->handoff_offset_bytes > memory.workspace.capacity_bytes ||
          vision->handoff_capacity_bytes == 0 ||
          vision->handoff_capacity_bytes >
              memory.workspace.capacity_bytes - vision->handoff_offset_bytes ||
          vision->handoff_active_bytes != 0)) ||
        memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int exercise_artifact(const char* artifact) {
    // Own engines, independent of the shared-engine cases below.
    if (const int result = exercise_concurrent_wave_host_tier_admission(artifact); result != 0) {
        return result;
    }
    if (const int result = exercise_light_load_host_capture_reuse(artifact); result != 0) {
        return result;
    }
    {
        ninferno::EngineOptions options             = engine_options(artifact);
        options.context_cache.device_state_slots  = 2;
        options.context_cache.max_shared_prefixes = 0;
        ninferno::Engine engine(std::move(options));
        if (const int result =
                verify_loaded_product(engine, ninferno::test::real_artifact(artifact).vision);
            result != 0) {
            return result;
        }
        if (const int result = exercise_registered_frontend(engine); result != 0) { return result; }
        if (const int result = exercise_stream_observations(engine); result != 0) { return result; }
        if (const int result = exercise_full_prefill_chunk(engine); result != 0) { return result; }
        if (const int result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::PrivateOnly);
            result != 0) {
            return result;
        }
        if (const int result = exercise_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_abandoned_handle_capacity(engine); result != 0) {
            return result;
        }
    }
    if (const int result = exercise_rewrite_branch(artifact); result != 0) { return result; }
    if (ninferno::test::real_artifact(artifact).vision) {
        ninferno::Engine engine(engine_options(artifact));
        if (const int result = exercise_vision(engine); result != 0) { return result; }
    } else {
        ninferno::test::skip_case("vision prefix cases",
                                  "no vision component (text-only artifact)");
    }
    if (const int result = exercise_host_restore(artifact); result != 0) { return result; }
    {
        // Production C=1/H=1 topology: repeated exact use promotes the shared prefix under one
        // cache Device slot; its Fork/Restore and the later ResponseReplay must then rotate
        // without a session identity or dropping either owner.
        ninferno::Engine engine(shared_replacement_engine_options(artifact));
        if (const int result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::SharedAlias);
            result != 0) {
            return result;
        }
    }
    if (const int result = exercise_shared_replacement_and_full_capacity_reuse(artifact);
        result != 0) {
        return result;
    }
    if (const int result = exercise_private_long_anchor_capture_and_replacement(artifact);
        result != 0) {
        return result;
    }
    if (const int result = exercise_last_private_alias_eviction(artifact); result != 0) {
        return result;
    }
    if (const int result = exercise_concurrent_resource_settlement(artifact); result != 0) {
        return result;
    }
    return 0;
}

int exercise_attention_integration(const char* artifact) {
    const auto setting = [](const char* name, const char* fallback) {
        const char* value = std::getenv(name);
        return std::string_view(value && *value ? value : fallback);
    };
    const auto storage =
        ninferno::test::parse_kv_cache_storage(setting("NINFERNO_TEST_KV_DTYPE", "bf16"));
    const auto backend_name = setting("NINFERNO_TEST_SPECULATIVE", "mtp");
    ninferno::SpeculativeBackend backend;
    if (backend_name == "none")
        backend = ninferno::SpeculativeBackend::None;
    else if (backend_name == "mtp")
        backend = ninferno::SpeculativeBackend::Mtp;
    else if (backend_name == "dflash")
        backend = ninferno::SpeculativeBackend::DFlash;
    else if (backend_name == "dflash2")
        backend = ninferno::SpeculativeBackend::DFlash2;
    else
        throw std::invalid_argument("unknown attention integration backend");
    const auto batch =
        static_cast<std::uint32_t>(std::stoul(std::string(setting("NINFERNO_TEST_BATCH", "2"))));
    const auto drafts = static_cast<std::uint32_t>(std::stoul(std::string(setting(
        "NINFERNO_TEST_DRAFT_TOKENS", backend == ninferno::SpeculativeBackend::Mtp ? "3" : "7"))));
    ninferno::EngineOptions options;
    options.artifact_path                    = artifact;
    options.kv_cache                         = storage;
    options.max_context                      = 65536;
    options.kv_capacity                      = ninferno::KvCapacityPolicy::explicit_capacity(65536);
    options.prefill_chunk                    = 1024;
    options.max_concurrency                  = batch;
    options.max_pending_requests             = batch;
    options.context_cache.device_state_slots = batch + 2;
    options.context_cache.max_private_continuations = batch + 2;
    options.speculative.backend                     = backend;
    options.speculative.draft_tokens  = backend == ninferno::SpeculativeBackend::None ? 0 : drafts;
    options.speculative.proposal_head = ninferno::ProposalHead::Full;
    ninferno::Engine engine(options);
    if (engine.memory_summary().kv_cache != storage)
        throw std::runtime_error("attention integration selected the wrong KV dtype");

    const auto validate = [backend](const ninferno::GenerationResult& result, std::uint32_t count) {
        if (result.generated_token_ids.size() != count ||
            result.finish_reason != ninferno::FinishReason::OutputLimit)
            throw std::runtime_error("attention integration did not finish its output budget");
        if (count > 1 && backend != ninferno::SpeculativeBackend::None &&
            result.speculative.rounds == 0)
            throw std::runtime_error(
                "attention integration did not exercise speculative verification");
    };
    const auto seed =
        engine.tokenize_text("Explain how a sequence continues from its stored prefix. ");
    const auto visible_offset = backend == ninferno::SpeculativeBackend::Mtp ? 2 * drafts : 0;
    std::vector<ninferno::TokenId> prefix(8182 - visible_offset - 3 * (batch - 1));
    for (std::size_t i = 0; i < prefix.size(); ++i) prefix[i] = seed[i % seed.size()];
    const auto primed = engine.generate(engine.prepare_tokens(prefix), fixed_output(8));
    validate(primed, 8);
    prefix.insert(prefix.end(), primed.generated_token_ids.begin(),
                  primed.generated_token_ids.end());

    // A private continuation need not be retained for every competing row. Long output budgets
    // keep rows active together even when another row prefills, across a Graph resource tier.
    const auto before = engine.runtime_stats();
    std::vector<ninferno::GenerationHandle> handles;
    for (std::uint32_t row = 0; row < batch; ++row) {
        auto prompt = prefix;
        prompt.insert(prompt.end(), row * 3, seed.back());
        handles.push_back(
            engine.submit(engine.prepare_tokens(prompt), fixed_output(128 + row * 5)));
    }
    std::vector<ninferno::TokenId> continuation = prefix;
    std::uint64_t reused_tokens               = 0;
    for (std::uint32_t row = 0; row < batch; ++row) {
        const auto result = handles[row].wait();
        validate(result, 128 + row * 5);
        reused_tokens += result.reused_prompt_tokens;
        if (row == 0)
            continuation.insert(continuation.end(), result.generated_token_ids.begin(),
                                result.generated_token_ids.end());
    }
    if (reused_tokens == 0)
        throw std::runtime_error("attention integration did not reuse the primed continuation");
    const auto after = engine.runtime_stats();
    if (batch > 1 && after.decode_row_rounds - before.decode_row_rounds <=
                         after.decode_rounds - before.decode_rounds)
        throw std::runtime_error("attention integration did not execute a multi-row decode round");
    continuation.insert(continuation.end(), 33, seed.back());
    const auto reused = engine.generate(engine.prepare_tokens(continuation), fixed_output(8));
    const auto fresh = engine.generate(engine.prepare_tokens(continuation), fixed_output(8, false));
    validate(reused, 8);
    validate(fresh, 8);
    if (reused.reused_prompt_tokens == 0 || fresh.reused_prompt_tokens != 0)
        throw std::runtime_error(
            "attention integration prefix continuation did not follow reuse policy");
    const auto appended = reused.prompt.prompt_tokens - reused.reused_prompt_tokens;
    if (appended <= 16 || appended > 256)
        throw std::runtime_error("attention integration did not exercise small prefix append");
    const auto memory = engine.memory_summary();
    if (memory.workspace_logical_peak_bytes == 0 ||
        memory.workspace_logical_peak_bytes > memory.workspace.capacity_bytes)
        throw std::runtime_error("attention integration exceeded its planned workspace");
    std::cout << "attention integration KV=" << setting("NINFERNO_TEST_KV_DTYPE", "bf16")
              << " backend=" << backend_name << " B=" << batch
              << " workspace_peak=" << memory.workspace_logical_peak_bytes
              << " graph_allowance=" << memory.cuda_graph_allowance_bytes
              << " appended=" << appended
              << " append_prefill_ms=" << reused.timings.prefill_seconds * 1000 << '\n';
    return 0;
}

int main() {
    const char* artifact = std::getenv("NINFERNO_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFERNO_TEST_ARTIFACT is not set\n";
        return 77;
    }
    // Every prefix case drafts with MTP; an artifact without it has nothing to run here.
    if (!ninferno::test::real_artifact(artifact).mtp) {
        std::cout << "skip: the artifact has no MTP component\n";
        return 77;
    }
    const char* selected            = std::getenv("NINFERNO_PREFIX_REAL_SCENARIO");
    const std::string_view scenario = selected ? selected : "all";
    int result                      = 0;
    if (scenario == "attention") {
        try {
            result = exercise_attention_integration(artifact);
        } catch (const std::exception& error) {
            std::cerr << "attention integration failed: " << error.what() << '\n';
            return 1;
        }
    } else if (scenario == "vision") {
        if (!ninferno::test::real_artifact(artifact).vision) {
            std::cout << "skip: no vision component (text-only artifact)\n";
            return 77;
        }
        ninferno::Engine engine(engine_options(artifact));
        result = exercise_vision(engine);
    } else if (scenario == "all") {
        result = exercise_artifact(artifact);
    } else if (scenario == "concurrent") {
        result = exercise_concurrent_resource_settlement(artifact);
    } else if (scenario == "host-restore") {
        result = exercise_host_restore(artifact);
    } else if (scenario == "last-alias-eviction") {
        result = exercise_last_private_alias_eviction(artifact);
    } else if (scenario == "rewrite-branch") {
        result = exercise_rewrite_branch(artifact);
    } else if (scenario == "host-tier-wave") {
        result = exercise_concurrent_wave_host_tier_admission(artifact);
    } else if (scenario == "light-load-capture") {
        result = exercise_light_load_host_capture_reuse(artifact);
    } else if (scenario == "anthropic-prefix-regression") {
        result = exercise_anthropic_prefix_regression(artifact);
    } else if (scenario == "shared-rewrite-materialization") {
        result = exercise_shared_rewrite_materialization(artifact);
    } else if (scenario == "pressure-resume") {
        result = exercise_pressure_partial_spill_and_resume(artifact);
    } else if (scenario == "private-checkpoint-pressure") {
        result = exercise_private_checkpoint_pressure_retention(artifact);
    } else if (scenario == "source-pressure-protection") {
        result = exercise_materialization_source_pressure_protection(artifact);
    } else if (scenario == "shared-replacement") {
        result = exercise_shared_replacement_and_full_capacity_reuse(artifact);
    } else if (scenario == "private-long-anchor") {
        result = exercise_private_long_anchor_capture_and_replacement(artifact);
    } else if (scenario == "rewrite-checkpoint-shared") {
        ninferno::Engine engine(shared_replacement_engine_options(artifact));
        result = exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::SharedAlias);
    } else if (scenario == "rewrite-checkpoint") {
        auto options                              = engine_options(artifact);
        options.context_cache.device_state_slots  = 2;
        options.context_cache.max_shared_prefixes = 0;
        ninferno::Engine engine(std::move(options));
        result = exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::PrivateOnly);
    } else if (scenario == "stream-observations") {
        auto options          = engine_options(artifact);
        options.enable_vision = false;
        options.context_cache = ninferno::ContextCacheOptions{.enabled = false};
        ninferno::Engine engine(std::move(options));
        result = exercise_stream_observations(engine);
    } else {
        throw std::invalid_argument("unknown prefix integration scenario");
    }
    if (result == 0) { std::cout << "ok\n"; }
    return result;
}
