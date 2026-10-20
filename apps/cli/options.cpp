#include "options.h"
#include "core/command_line.h"
#include "ninferno/types.h"
#include "product/activation_options.h"
#include "product/numeric_options.h"
#include "product/sampling_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::cli {
namespace {

using product::parse_float_in;
using product::parse_kv_capacity;
using product::parse_kv_dtype;
using product::parse_u32;
using product::parse_u64;

int parse_device(const char* text) { return product::parse_nonnegative_int(text, "device"); }


ReasoningEffort parse_reasoning_effort_option(std::string_view text) {
    if (const auto effort = ninferno::parse_reasoning_effort(text)) { return *effort; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 + " --version\n       " + argv0 +
           " <model.ninferno> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--device N]\n"
           "       [--kv-dtype auto|bf16|int8|fp8|nvfp4|k8v4] [--sliding-kv-dtype bf16|fp8]\n"
           "       [--state-dtype f32|bf16|f16]\n"
           "       [--activations exact|fast|int8] [--rope-yarn-factor F]\n"
           "       [--spec auto|none|mtp|dflash|dflash2 "
           "--draft-tokens N]\n"
           "       [--ngram-draft-tokens N [--ngram-min-match M]]  copy rounds beside MTP\n"
           "       [--lm-head-draft] [--acceptance-floor TOKENS_PER_ROUND]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--chat-template FILE]\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-effort none|minimal|low|medium|high|xhigh|max] [--vision]\n"
           "       [--no-cuda-graph]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "\n"
           "Streams answer content to stdout and reasoning plus diagnostics to stderr.\n"
           "Structured message content accepts text, image/image_url, and video/video_url parts;\n"
           "media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n"
           "--vision enables image/video input and loads the fixed Vision GPU allocations.\n"
           "Without --spec, speculation follows the profile the model came with\n"
           "(<model>.ninferno.speculation.json, or a maintainer's "
           "<model>.ninferno.spec_profile.json)\n"
           "for this GPU: its default for the thinking flag. --spec none decodes plainly. `" +
           std::string(argv0) +
           " spec-tune <model.ninferno>` measures a profile, `spec-reselect <profile.json>` "
           "selects\n"
           "again over its measurements (maintainers; docs/cli.md lists their flags).\n"
           "--thinking-budget caps model-origin thinking tokens; inserted control tokens count "
           "toward --max-new.\n"
           "--kv-capacity auto plans every runtime byte and leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB for the CUDA driver's own allocations (under WSL 2, where Windows shares the "
           "card and pages it out to Host memory when full, a 32nd of it, at most 1 GiB, as far "
           "as the free memory allows).\n"
           "Sampling defaults come from the loaded model and thinking mode; flags override "
           "individual fields.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (help_requested(argc, argv)) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument(".ninferno model path is required"); }
    options.artifact_path     = argv[1];
    bool kv_capacity_explicit = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--chat-template") {
            options.chat_template_path = value(arg);
        } else if (arg == "--messages") {
            options.messages_path = value(arg);
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device = parse_device(value(arg));
        } else if (arg == "--kv-dtype") {
            const std::string_view text = value(arg);
            options.kv_cache = text == "auto" ? std::nullopt : std::optional(parse_kv_dtype(text));
        } else if (arg == "--sliding-kv-dtype") {
            const std::string_view text = value(arg);
            if (text != "bf16" && text != "fp8") {
                throw std::invalid_argument("--sliding-kv-dtype must be bf16 or fp8");
            }
            options.sliding_kv_cache = parse_kv_dtype(text);
        } else if (arg == "--state-dtype") {
            const auto storage = parse_recurrent_state_storage(value(arg));
            if (!storage) { throw std::invalid_argument("--state-dtype must be f32, bf16 or f16"); }
            options.recurrent_state = *storage;
        } else if (arg == "--activations") {
            options.activations = product::parse_activation_mode(value(arg));
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor = parse_float_in(value(arg), "rope-yarn-factor", 1.0F, 64.0F);
        } else if (arg == "--spec") {
            const std::string_view mode = value(arg);
            if (mode == "auto") {
                options.spec_explicit = false;
            } else {
                options.speculative.backend = product::parse_speculative_backend(mode);
                options.spec_explicit       = true;
            }
        } else if (arg == "--acceptance-floor") {
            options.acceptance_floor =
                parse_float_in(value(arg), "acceptance-floor", 0.0F, 1000.0F);
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens =
                parse_u32(value(arg), "ngram-draft-tokens", /*allow_zero=*/true);
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = parse_u32(value(arg), "ngram-min-match");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value(arg), "thinking-budget");
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort_option(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (product::parse_sampling_flag(
                       arg, [&] { return value(arg); }, options.sampling)) {
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(value(arg));
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.enable_thinking == false && options.reasoning_effort &&
        *options.reasoning_effort != ReasoningEffort::None) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (options.reasoning_effort == ReasoningEffort::None) options.enable_thinking = false;
    if (options.enable_thinking == false && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }
    return options;
}

} // namespace ninferno::cli
