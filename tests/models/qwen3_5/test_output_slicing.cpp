// The multi-token delta harness.
//
// Speculation (MTP/DFlash) commits 1..k+1 tokens per round, so every piece of output presentation
// behind OutputSession -- stop tokens, stop strings and their hold-back, the incremental UTF-8
// decoder, the reasoning splitter, the thinking budget and the Qwen tool-call decoder -- must give
// the same result for every way a token sequence is cut into rounds. This test feeds a corpus of
// generated-token sequences through OutputSession under every slicing into rounds of 1..8 tokens
// (exhaustively for short sequences; otherwise the uniform slicings and 1000 random ones), plus
// the whole sequence as one round, emulating the Engine loop. The assembled content, reasoning,
// tool calls, parse diagnostics, finish reason, accepted-token count, matched stop, reasoning-token
// count, thinking-budget statistics and execution splits must equal one token per round and the
// stated expectation. Every published delta must be valid UTF-8, and no excluded stop string may
// ever be published.
#include "models/qwen3_5/frontend/frontend.h"

#include "core/utf8.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/unicode.h"
#include "support/files.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninferno::FinishReason;
using ninferno::OutputChannel;
using ninferno::TokenId;
using Frontend       = ninferno::models::qwen3_5::Frontend;
using PreparedPrompt = ninferno::models::qwen3_5::PreparedPrompt;
namespace fi         = ninferno::models::qwen3_5::frontend;

constexpr TokenId kByteBase       = 1'000;
constexpr TokenId kEos            = 248046; // <|im_end|>
constexpr std::uint32_t kMaxRound = 8;
constexpr std::uint32_t kAll      = std::numeric_limits<std::uint32_t>::max();

// Multi-byte model tokens. Each is an added (non-special) token, so a script can place a stop
// string, a reasoning close or a tool-call marker across or inside token boundaries at will.
const std::vector<std::string>& piece_vocabulary() {
    static const std::vector<std::string> pieces = {
        "helloST",
        "OPtail",
        "STOP",
        "ST",
        "OP",
        "Ж",
        "中文",
        "😀",
        "</think>",
        "</thi",
        "nk>\n\nans",
        "wer",
        "<tool_call>",
        "</tool_call>",
        "<parameter=",
        "</parameter>",
        "d.\n\n<tool",
        "_call>\n<function=get_weather>\n",
        "</tool_call>\n<tool_call>",
        "\"a</parameter>b\"",
        "ok</think>\n\nHi",
        "\n</parameter>\n",
        "</function>\n</tool_call>",
        " <tool_call>",
        "<tool",
        "_call>",
        "<<tool_call>",
        "\n<function=search>\n",
        "\n<function=search>\n</function>\n",
        "\n<function=get_weather>\n",
    };
    return pieces;
}

TokenId piece_id(std::size_t index) { return static_cast<TokenId>(40 + index); }

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninferno::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level symbol is outside one byte");
}

nlohmann::json added(int id, std::string content, bool special) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

Frontend
make_fixture_frontend(const std::string& generation_config = R"({"eos_token_id":[248046]})",
                      const std::string& template_path     = NINFERNO_SOURCE_DIR
                      "/tools/chat_templates/qwen3_6.jinja") {
    nlohmann::json tokens = nlohmann::json::array();
    tokens.push_back(added(kEos, "<|im_end|>", true));
    tokens.push_back(added(248045, "<|im_start|>", true));
    tokens.push_back(added(248053, "<|vision_start|>", true));
    tokens.push_back(added(248054, "<|vision_end|>", true));
    tokens.push_back(added(248056, "<|image_pad|>", true));
    tokens.push_back(added(248057, "<|video_pad|>", true));
    tokens.push_back(added(248068, "<think>", false));
    tokens.push_back(added(248044, "<|endoftext|>", true));
    const auto& pieces = piece_vocabulary();
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        tokens.push_back(added(piece_id(i), pieces[i], false));
    }
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value <= 255; ++value) {
        vocab[byte_level_symbol(static_cast<std::uint8_t>(value))] = kByteBase + value;
    }
    const std::string tokenizer_json =
        nlohmann::json{{"model",
                        {{"type", "BPE"}, {"vocab", vocab}, {"merges", nlohmann::json::array()}}},
                       {"added_tokens", tokens}}
            .dump();
    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    const std::string chat_template = read_file(template_path);
    const std::string tokenizer_config = nlohmann::json{{"add_bos_token", false},
                                                        {"add_prefix_space", false},
                                                        {"pad_token", "<|endoftext|>"},
                                                        {"chat_template", chat_template},
                                                        {"added_tokens_decoder", decoder}}
                                             .dump();

    ninferno::models::qwen3_5::FrontendResources resources{
        tokenizer_json, tokenizer_config, chat_template, generation_config, "", ""};
    resources.tokenizer = std::make_shared<const fi::Tokenizer>(
        fi::TokenizerResources{tokenizer_json, tokenizer_config, generation_config});
    resources.public_token_count = static_cast<std::uint32_t>(resources.tokenizer->vocab_size());
    ninferno::models::qwen3_5::FrontendOptions options;
    options.vision_enabled = false;
    options.max_context    = std::numeric_limits<std::uint32_t>::max();
    return ninferno::models::qwen3_5::make_frontend(resources, options);
}

// A generated-token script. A Piece is one multi-byte token when it names a vocabulary piece, the
// EOS token when it is kEosPiece, and otherwise one byte token per byte.
struct Piece {
    std::string text;
    bool as_bytes = false; // force one token per byte even for a vocabulary piece
};
const std::string kEosPiece = "\x01<EOS>";

std::vector<TokenId> script(const std::vector<Piece>& pieces) {
    const auto& vocabulary = piece_vocabulary();
    std::vector<TokenId> out;
    for (const Piece& piece : pieces) {
        if (piece.text == kEosPiece) {
            out.push_back(kEos);
            continue;
        }
        const auto found = std::find(vocabulary.begin(), vocabulary.end(), piece.text);
        if (!piece.as_bytes && found != vocabulary.end()) {
            out.push_back(piece_id(static_cast<std::size_t>(found - vocabulary.begin())));
            continue;
        }
        for (const char byte : piece.text) {
            out.push_back(kByteBase + static_cast<unsigned char>(byte));
        }
    }
    return out;
}

enum class PromptKind : std::uint8_t { Plain, Thinking, Tools, ThinkingTools };

struct Case {
    std::string name;
    std::vector<TokenId> tokens;
    PromptKind prompt = PromptKind::Plain;
    std::vector<ninferno::StopString> stops;
    std::uint32_t max_tokens = 0; // 0: the script length
    std::optional<std::uint32_t> thinking_budget;
    std::uint32_t repetition_window    = 0;
    bool repetition_concludes_thinking = false;
    // Whole-sequence semantics, checked on the one-token-per-round reference.
    std::optional<std::string> content;
    std::optional<std::string> reasoning;
    std::optional<FinishReason> finish;
    std::optional<std::uint32_t> accepted;
    std::optional<std::size_t> tool_calls;
    std::vector<std::string> call_arguments; // expected arguments_json per call, when given
};

struct Outcome {
    std::string content;
    std::string reasoning;
    std::vector<std::pair<std::string, std::string>> calls;
    ninferno::ToolCallParseDiagnostics diagnostics;
    FinishReason finish         = FinishReason::None;
    std::uint32_t accepted      = 0; // model and control tokens committed
    std::uint32_t reasoning_tok = 0;
    std::optional<std::string> matched_stop;
    std::uint32_t thinking_model_tokens = 0;
    std::uint32_t thinking_injected     = 0;
    std::vector<std::uint32_t> splits; // absolute execution-split positions
    std::string violation;             // first per-delta invariant violation, if any

    bool operator==(const Outcome& other) const {
        return content == other.content && reasoning == other.reasoning && calls == other.calls &&
               diagnostics == other.diagnostics && finish == other.finish &&
               accepted == other.accepted && reasoning_tok == other.reasoning_tok &&
               matched_stop == other.matched_stop &&
               thinking_model_tokens == other.thinking_model_tokens &&
               thinking_injected == other.thinking_injected && splits == other.splits;
    }
};

std::string describe(const Outcome& o) {
    std::ostringstream s;
    s << "content=" << nlohmann::json(o.content).dump()
      << " reasoning=" << nlohmann::json(o.reasoning).dump()
      << " finish=" << static_cast<int>(o.finish) << " accepted=" << o.accepted
      << " reasoning_tokens=" << o.reasoning_tok << " thinking=" << o.thinking_model_tokens << "+"
      << o.thinking_injected << " calls=" << o.calls.size()
      << " fallback=" << static_cast<int>(o.diagnostics.fallback_reason)
      << " stop=" << (o.matched_stop ? *o.matched_stop : "-") << " splits=[";
    for (const std::uint32_t split : o.splits) { s << split << ' '; }
    s << "]";
    for (const auto& [name, arguments] : o.calls) { s << " [" << name << " " << arguments << "]"; }
    return s.str();
}

class Harness {
public:
    Harness()
        : frontend_(make_fixture_frontend()), plain_(frontend_.prepare_tokens({kByteBase + 'x'})),
          thinking_(prepare(true, false)), tools_(prepare(false, true)),
          thinking_tools_(prepare(true, true)) {}

    [[nodiscard]] const PreparedPrompt& prompt(PromptKind kind) const {
        switch (kind) {
        case PromptKind::Plain:
            return plain_;
        case PromptKind::Thinking:
            return thinking_;
        case PromptKind::Tools:
            return tools_;
        case PromptKind::ThinkingTools:
            return thinking_tools_;
        }
        throw std::logic_error("unknown prompt kind");
    }

    // Emulates the Engine loop: a round is the next slice, capped by the thinking-budget licence
    // and by the script; a terminal decision ends the request; a pending thinking control is
    // committed as its own round; an exhausted script ends like a between-round cancellation.
    [[nodiscard]] Outcome run(const Case& c, const std::vector<std::uint32_t>& slices) const {
        ninferno::StopPolicy stop;
        stop.strings                       = c.stops;
        stop.repetition_window             = c.repetition_window;
        stop.repetition_concludes_thinking = c.repetition_concludes_thinking;
        auto session = frontend_.make_output_session(
            prompt(c.prompt), stop, ninferno::OutputOptions{.tool_name_max_length = 64},
            ninferno::ThinkingControlOptions{.budget = c.thinking_budget});
        Outcome out;
        std::uint32_t remaining =
            c.max_tokens == 0 ? static_cast<std::uint32_t>(c.tokens.size()) : c.max_tokens;
        std::size_t position = 0;
        std::size_t slice    = 0;
        for (int guard = 0; guard < 1'000'000; ++guard) {
            const std::span<const TokenId> control = session.pending_control_tokens();
            ninferno::runtime::OutputDecision decision;
            if (!control.empty()) {
                decision = session.preview_control(control, remaining);
            } else {
                const std::uint32_t licensed = session.model_token_budget_remaining(remaining);
                if (licensed == 0 || position == c.tokens.size()) {
                    decision = session.preview_terminal(FinishReason::Cancelled);
                } else {
                    std::uint32_t size = std::min(slices[slice++ % slices.size()], licensed);
                    size               = std::min<std::uint32_t>(
                        size, static_cast<std::uint32_t>(c.tokens.size() - position));
                    decision = session.preview_model(
                        std::span<const TokenId>(c.tokens).subspan(position, size), remaining,
                        FinishReason::OutputLimit);
                    position += decision.accepted_tokens;
                }
            }
            if (decision.prefix_execution_split_after) {
                out.splits.push_back(out.accepted + *decision.prefix_execution_split_after);
            }
            out.accepted += decision.accepted_tokens;
            remaining -= decision.accepted_tokens;
            for (const ninferno::OutputDelta& delta : session.commit_preview()) {
                if (out.violation.empty() && !ninferno::valid_utf8(delta.text)) {
                    out.violation = "a published delta is not valid UTF-8";
                }
                (delta.channel == OutputChannel::Content ? out.content : out.reasoning) +=
                    delta.text;
            }
            if (decision.finished()) {
                out.finish = decision.finish_reason;
                break;
            }
        }
        for (ninferno::GeneratedToolCall& call : session.take_tool_calls()) {
            out.calls.emplace_back(std::move(call.name), std::move(call.arguments_json));
        }
        out.diagnostics           = session.tool_call_parse_diagnostics();
        out.reasoning_tok         = session.reasoning_tokens();
        out.matched_stop          = session.matched_stop_string();
        const auto stats          = session.thinking_stats();
        out.thinking_model_tokens = stats.model_thinking_tokens;
        out.thinking_injected     = stats.injected_tokens;
        // The first match terminates, so an excluded stop string is never published, and an
        // included one only as the very end of its channel.
        for (const ninferno::StopString& s : c.stops) {
            const std::string& text =
                s.channel == OutputChannel::Content ? out.content : out.reasoning;
            const std::size_t found = text.find(s.text);
            if (found != std::string::npos && out.violation.empty() &&
                !(s.include_in_output && found + s.text.size() == text.size())) {
                out.violation = "stop string '" + s.text + "' was published";
            }
        }
        return out;
    }

private:
    [[nodiscard]] PreparedPrompt prepare(bool thinking, bool tools) const {
        ninferno::ChatMessage message;
        message.role = ninferno::ChatRole::User;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
        ninferno::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.continuation    = ninferno::PromptContinuationMode::NewAssistantTurn;
        input.options.enable_thinking = thinking;
        if (tools) {
            input.options.tool_jsons.push_back(
                R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}})");
            input.options.tool_jsons.push_back(
                R"({"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"q":{"type":"string"},"n":{"type":"integer"}}}}})");
        }
        return frontend_.prepare(std::move(input));
    }

    Frontend frontend_;
    PreparedPrompt plain_;
    PreparedPrompt thinking_;
    PreparedPrompt tools_;
    PreparedPrompt thinking_tools_;
};

// All compositions of n into rounds of 1..kMaxRound when there are few; otherwise the uniform
// slicings and 1000 random ones. Always also the whole script as one round.
std::vector<std::vector<std::uint32_t>> slicings(std::size_t n, std::uint64_t seed) {
    std::vector<std::vector<std::uint32_t>> out;
    if (n <= 13) {
        std::vector<std::uint32_t> current;
        std::function<void(std::size_t)> compose = [&](std::size_t left) {
            if (left == 0) {
                if (!current.empty()) { out.push_back(current); }
                return;
            }
            for (std::uint32_t part = 1; part <= kMaxRound && part <= left; ++part) {
                current.push_back(part);
                compose(left - part);
                current.pop_back();
            }
        };
        compose(n);
    } else {
        for (std::uint32_t k = 1; k <= kMaxRound; ++k) { out.push_back({k}); }
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<std::uint32_t> part(1, kMaxRound);
        for (int i = 0; i < 1000; ++i) {
            std::vector<std::uint32_t> s;
            for (std::size_t left = n; left != 0;) {
                const std::uint32_t p = std::min<std::uint32_t>(part(rng), left);
                s.push_back(p);
                left -= p;
            }
            out.push_back(std::move(s));
        }
    }
    out.push_back({kAll});
    return out;
}

const std::string kFffd = "\xef\xbf\xbd";

ninferno::StopString stop(std::string text, bool include = false,
                        OutputChannel channel = OutputChannel::Content) {
    return ninferno::StopString{
        .text = std::move(text), .channel = channel, .include_in_output = include};
}

std::vector<Case> corpus() {
    using P = Piece;
    std::vector<Case> cases;

    // A stop string across tokens and rounds, inside one token, leftmost match first.
    cases.push_back({.name     = "stop across tokens",
                     .tokens   = script({P{"say "}, P{"helloST"}, P{"OPtail"}, P{" more"}}),
                     .stops    = {stop("STOP")},
                     .content  = "say hello",
                     .finish   = FinishReason::StopString,
                     .accepted = 6});
    cases.push_back({.name     = "stop inside one token",
                     .tokens   = script({P{"a"}, P{"helloST"}, P{"b"}}),
                     .stops    = {stop("ll")},
                     .content  = "ahe",
                     .finish   = FinishReason::StopString,
                     .accepted = 2});
    cases.push_back({.name     = "leftmost of several stops",
                     .tokens   = script({P{"x"}, P{"OPtail"}, P{"helloST"}, P{"STOP"}}),
                     .stops    = {stop("STOP"), stop("tail"), stop("OP")},
                     .content  = "x",
                     .finish   = FinishReason::StopString,
                     .accepted = 2});
    cases.push_back({.name     = "stop included in output",
                     .tokens   = script({P{"ab"}, P{"ST"}, P{"OP"}, P{"cd"}}),
                     .stops    = {stop("STOP", true)},
                     .content  = "abSTOP",
                     .finish   = FinishReason::StopString,
                     .accepted = 4});
    // A held stop prefix is released, never lost, at EOS and at the length limit.
    cases.push_back({.name     = "held prefix flushed at EOS",
                     .tokens   = script({P{"q"}, P{"helloST"}, P{kEosPiece}, P{"zz"}}),
                     .stops    = {stop("STOP")},
                     .content  = "qhelloST",
                     .finish   = FinishReason::StopToken,
                     .accepted = 3});
    cases.push_back({.name       = "held prefix flushed at length",
                     .tokens     = script({P{"q"}, P{"helloST"}, P{"OP"}}),
                     .stops      = {stop("STOP")},
                     .max_tokens = 2,
                     .content    = "qhelloST",
                     .finish     = FinishReason::OutputLimit,
                     .accepted   = 2});
    cases.push_back(
        {.name     = "false stop prefix released",
         .tokens   = script({P{"ST"}, P{"x"}, P{"S"}, P{"T"}, P{"O"}, P{"x"}, P{kEosPiece}}),
         .stops    = {stop("STOP")},
         .content  = "STxSTOx",
         .finish   = FinishReason::StopToken,
         .accepted = 7});
    // EOS and a stop string at every position of an 8-token window, tokens after it.
    for (std::uint32_t at = 0; at < 8; ++at) {
        std::vector<P> eos_script;
        std::vector<P> stop_script;
        std::string prefix;
        for (std::uint32_t i = 0; i < 8; ++i) {
            if (i == at) {
                eos_script.push_back(P{kEosPiece});
                stop_script.push_back(P{"STOP"});
                continue;
            }
            const std::string letter(1, static_cast<char>('a' + i));
            eos_script.push_back(P{letter});
            stop_script.push_back(P{letter});
            if (i < at) { prefix += letter; }
        }
        eos_script.push_back(P{"tail"});
        stop_script.push_back(P{"tail"});
        cases.push_back({.name     = "EOS at window position " + std::to_string(at),
                         .tokens   = script(eos_script),
                         .content  = prefix,
                         .finish   = FinishReason::StopToken,
                         .accepted = at + 1});
        cases.push_back({.name     = "stop string at window position " + std::to_string(at),
                         .tokens   = script(stop_script),
                         .stops    = {stop("STOP")},
                         .content  = prefix,
                         .finish   = FinishReason::StopString,
                         .accepted = at + 1});
    }
    // Code points split over rounds are published whole. A limit inside a code
    // point publishes U+FFFD. Invalid bytes are replaced without losing the text after them.
    cases.push_back({.name     = "utf8 split across tokens",
                     .tokens   = script({P{"é", true}, P{"中文", true}, P{"😀", true}, P{"Ж"},
                                         P{"中文"}, P{kEosPiece}}),
                     .content  = "é中文😀Ж中文",
                     .finish   = FinishReason::StopToken,
                     .accepted = 2 + 6 + 4 + 1 + 1 + 1});
    cases.push_back({.name       = "limit inside a code point",
                     .tokens     = script({P{"a"}, P{"😀", true}}),
                     .max_tokens = 3,
                     .content    = "a" + kFffd,
                     .finish     = FinishReason::OutputLimit,
                     .accepted   = 3});
    cases.push_back({.name = "invalid bytes then text",
                     .tokens =
                         script({P{"\xe4\xb8", true}, P{"ok"}, P{"\xf5", true}, P{"!"}, P{kEosPiece}}),
                     .content  = kFffd + "ok" + kFffd + "!",
                     .finish   = FinishReason::StopToken,
                     .accepted = 7});
    cases.push_back({.name     = "multibyte stop string",
                     .tokens   = script({P{"a"}, P{"中文", true}, P{"b"}}),
                     .stops    = {stop("文")},
                     .content  = "a中",
                     .finish   = FinishReason::StopString,
                     .accepted = 7});
    // The reasoning start comes from the prompt; the close, content and a call can share
    // one round or one token.
    cases.push_back({.name   = "reasoning close split over tokens",
                     .tokens = script({P{"th"}, P{"</thi"}, P{"nk>\n\nans"}, P{"wer"}, P{kEosPiece}}),
                     .prompt = PromptKind::Thinking,
                     .content   = "answer",
                     .reasoning = "th",
                     .finish    = FinishReason::StopToken,
                     .accepted  = 6});
    cases.push_back({.name   = "canonical reasoning close",
                     .tokens = script({P{"hm"}, P{"</think>"}, P{"\n\n"}, P{"Hi"}, P{kEosPiece}}),
                     .prompt = PromptKind::Thinking,
                     .content   = "Hi",
                     .reasoning = "hm",
                     .finish    = FinishReason::StopToken});
    cases.push_back({.name     = "no reasoning split without a thinking prompt",
                     .tokens   = script({P{"a"}, P{"</think>"}, P{"b"}, P{kEosPiece}}),
                     .content  = "a</think>b",
                     .finish   = FinishReason::StopToken,
                     .accepted = 4});
    cases.push_back({.name           = "reasoning close, content and a call in one run",
                     .tokens         = script({P{"ok</think>\n\nHi"}, P{"d.\n\n<tool"},
                                               P{"_call>\n<function=get_weather>\n"}, P{"<parameter="},
                                               P{"city>\nOslo"}, P{"\n</parameter>\n"},
                                               P{"</function>\n</tool_call>"}, P{kEosPiece}}),
                     .prompt         = PromptKind::ThinkingTools,
                     .content        = "Hid.",
                     .reasoning      = "ok",
                     .finish         = FinishReason::StopToken,
                     .tool_calls     = 1,
                     .call_arguments = {R"({"city":"Oslo"})"}});
    // A tool-call marker inside unterminated reasoning stays reasoning text, not lost. A
    // complete call that ends the reasoning at the stop token is the turn's call too;
    // one quoted before more reasoning is not.
    cases.push_back(
        {.name      = "tool call inside unterminated reasoning",
         .tokens    = script({P{"plan "}, P{"<tool_call>"}, P{"\n<function=search>\n</function>\n"},
                              P{"</tool_call>"}, P{kEosPiece}}),
         .prompt    = PromptKind::ThinkingTools,
         .content   = "",
         .reasoning = "plan <tool_call>\n<function=search>\n</function>\n</tool_call>",
         .finish    = FinishReason::StopToken,
         .tool_calls = 1});
    cases.push_back(
        {.name      = "tool call quoted inside unterminated reasoning",
         .tokens    = script({P{"plan "}, P{"<tool_call>"}, P{"\n<function=search>\n</function>\n"},
                              P{"</tool_call>"}, P{" then think"}, P{kEosPiece}}),
         .prompt    = PromptKind::ThinkingTools,
         .content   = "",
         .reasoning = "plan <tool_call>\n<function=search>\n</function>\n</tool_call> then think",
         .finish    = FinishReason::StopToken,
         .tool_calls = 0});
    // Several calls in one delta; brackets and quotes inside string arguments.
    cases.push_back({.name           = "two calls in one delta",
                     .tokens         = script({P{"<tool_call>"}, P{"\n<function=search>\n"},
                                               P{"<parameter="}, P{"q>\n{\"x\": [1, \"]\"]}\n"},
                                               P{"</parameter>"}, P{"\n"},
                                               P{"</function>\n</tool_call>"}, P{"\n"},
                                               P{"<tool_call>"}, P{"\n<function=search>\n"},
                                               P{"<parameter="}, P{"n>\n3\n"}, P{"</parameter>"},
                                               P{"\n"}, P{"</function>\n</tool_call>"}, P{kEosPiece}}),
                     .prompt         = PromptKind::Tools,
                     .content        = "",
                     .finish         = FinishReason::StopToken,
                     .tool_calls     = 2,
                     .call_arguments = {R"({"q":"{\"x\": [1, \"]\"]}"})", R"({"n":3})"}});
    cases.push_back(
        {.name       = "adjacent calls joined in one token",
         .tokens     = script({P{"<tool_call>"}, P{"\n<function=search>\n</function>\n"},
                               P{"</tool_call>\n<tool_call>"}, P{"\n<function=get_weather>\n"},
                               P{"<parameter="}, P{"city>\n\"a<b>\""}, P{"\n</parameter>\n"},
                               P{"</function>\n</tool_call>"}, P{kEosPiece}}),
         .prompt         = PromptKind::Tools,
         .content        = "",
         .finish         = FinishReason::StopToken,
         .tool_calls     = 2,
         .call_arguments = {"{}", R"({"city":"\"a<b>\""})"}});
    // A closing tag inside a value is ambiguous markup: the whole region is restored verbatim.
    // A closer the grammar cannot continue from (here "b" follows it) is value text quoting the
    // markup: the value keeps it and the call is made.
    cases.push_back({.name           = "closing tag quoted inside a value stays in the value",
                     .tokens         = script({P{"<tool_call>"}, P{"\n<function=get_weather>\n"},
                                               P{"<parameter="}, P{"city>\n"},
                                               P{"\"a</parameter>b\""}, P{"\n</parameter>\n"},
                                               P{"</function>\n</tool_call>"}, P{kEosPiece}}),
                     .prompt         = PromptKind::Tools,
                     .content        = "",
                     .finish        = FinishReason::StopToken,
                     .tool_calls     = 1,
                     .call_arguments = {R"({"city":"\"a</parameter>b\""})"}});
    cases.push_back({.name           = "text, whitespace and a split marker",
                     .tokens         = script({P{"Sure"}, P{" "}, P{"<tool"}, P{"_call>"},
                                               P{"\n<function=search>\n"},
                                               P{"</function>\n</tool_call>"}, P{kEosPiece}}),
                     .prompt         = PromptKind::Tools,
                     .content        = "Sure",
                     .finish         = FinishReason::StopToken,
                     .tool_calls     = 1,
                     .call_arguments = {"{}"}});
    cases.push_back(
        {.name       = "double angle before the marker",
         .tokens     = script({P{"a"}, P{"<<tool_call>"}, P{"\n<function=search>\n</function>\n"},
                               P{"</tool_call>"}, P{kEosPiece}}),
         .prompt     = PromptKind::Tools,
         .content    = "a<",
         .finish     = FinishReason::StopToken,
         .tool_calls = 1});
    cases.push_back({.name       = "partial marker at EOS is content",
                     .tokens     = script({P{"x "}, P{"<tool"}, P{kEosPiece}}),
                     .prompt     = PromptKind::Tools,
                     .content    = "x <tool",
                     .finish     = FinishReason::StopToken,
                     .tool_calls = 0});
    // A length limit inside a call returns the markup verbatim as content.
    cases.push_back({.name       = "limit inside a call",
                     .tokens     = script({P{"Hi"}, P{" <tool_call>"}, P{"\n<function=get_weather>\n"},
                                           P{"<parameter="}, P{"city>\nOs"}}),
                     .prompt     = PromptKind::Tools,
                     .max_tokens = 2 + 1 + 1 + 1,
                     .content    = "Hi <tool_call>\n<function=get_weather>\n<parameter=",
                     .finish     = FinishReason::OutputLimit,
                     .tool_calls = 0});
    cases.push_back({.name       = "stop string inside a call region",
                     .tokens     = script({P{"<tool_call>"}, P{"\n<function=search>\n"}, P{"STOP"},
                                           P{"</function>\n</tool_call>"}}),
                     .prompt     = PromptKind::Tools,
                     .stops      = {stop("STOP")},
                     .content    = "<tool_call>\n<function=search>\n",
                     .finish     = FinishReason::StopString,
                     .tool_calls = 0});
    // A reasoning-channel stop string, and the thinking budget with its injected control.
    cases.push_back({.name      = "reasoning-channel stop",
                     .tokens    = script({P{"we"}, P{"helloST"}, P{"OPtail"}}),
                     .prompt    = PromptKind::Thinking,
                     .stops     = {stop("STOP", false, OutputChannel::Reasoning)},
                     .reasoning = "wehello",
                     .finish    = FinishReason::StopString});
    cases.push_back({.name            = "thinking budget control",
                     .tokens          = script({P{"abcdefgh"}, P{"Hi"}, P{kEosPiece}}),
                     .prompt          = PromptKind::Thinking,
                     .max_tokens      = 400,
                     .thinking_budget = 5,
                     .content         = "fghHi",
                     .reasoning       = "abcde\n\n Considering the limited time by the user, I have "
                                        "to give the solution based on the thinking directly now.\n",
                     .finish          = FinishReason::StopToken});
    return cases;
}

// The stop set is generation_config.json plus the template's assistant-turn terminator. A
// fine-tune that lists only <|endoftext|> must still stop at <|im_end|>, the token its own template
// closes assistant turns with, and constrained decoding must accept it as a stop.
int test_turn_terminator_joins_default_stops() {
    const Frontend frontend = make_fixture_frontend(R"({"eos_token_id":[248044]})");
    const auto& stops       = frontend.default_stop_policy().token_ids;
    const auto has          = [&](TokenId id) {
        return std::find(stops.begin(), stops.end(), id) != stops.end();
    };
    int failures = 0;
    if (!has(248044) || !has(kEos) || stops.size() != 2) {
        std::cerr << "default stops are not generation_config plus the turn terminator\n";
        ++failures;
    }
    auto prompt                       = frontend.prepare_tokens({kByteBase + 'x'});
    auto session                      = frontend.make_output_session(prompt, {});
    const std::vector<TokenId> tokens = script({Piece{"ok"}, Piece{kEosPiece}, Piece{"more"}});
    const auto decision               = session.preview_model(
        tokens, static_cast<std::uint32_t>(tokens.size()), FinishReason::OutputLimit);
    if (decision.finish_reason != FinishReason::StopToken || decision.accepted_tokens != 3) {
        std::cerr << "generation did not stop at the template's turn terminator\n";
        ++failures;
    }
    const auto vocabulary = frontend.grammar_vocabulary(static_cast<std::uint32_t>(kEos + 1));
    if (std::find(vocabulary.stop_token_ids.begin(), vocabulary.stop_token_ids.end(), kEos) ==
        vocabulary.stop_token_ids.end()) {
        std::cerr << "constrained decoding does not take the turn terminator as a stop\n";
        ++failures;
    }
    return failures;
}

// The reasoning parser's initial state comes from the tail of the rendered prompt, not
// from a flag. Matrix: both maintained templates x enable_thinking {unset, true, false} x the
// final assistant prefix {none, "<think>", "<think>\n", "<think>\n\n", "<think></think>",
// "plain"} (assistant continuation), each output "мысль</think>\n\nответ" committed one token per
// round and as one round. A template-opened thinking block (no prefix, thinking not disabled)
// splits it into reasoning "мысль" and content "ответ"; with thinking disabled the reasoning is
// empty; a prefix is user text, so its "<think>" never opens a block, and a prefix with thinking
// enabled is refused. No committed delta is an empty reasoning delta.
int test_reasoning_start_state_matrix() {
    int failures = 0;
    const std::vector<std::string> prefixes = {"", "<think>", "<think>\n", "<think>\n\n",
                                               "<think></think>", "plain"};
    const std::vector<TokenId> output =
        script({Piece{"мысль"}, Piece{"</think>"}, Piece{"\n\nответ"}, Piece{kEosPiece}});
    for (const char* name : {"qwen3_6", "qwen3_8"}) {
        const Frontend frontend = make_fixture_frontend(
            R"({"eos_token_id":[248046]})",
            std::string(NINFERNO_SOURCE_DIR "/tools/chat_templates/") + name + ".jinja");
        for (const std::optional<bool> thinking :
             {std::optional<bool>{}, std::optional<bool>{true}, std::optional<bool>{false}}) {
            for (const std::string& prefix : prefixes) {
                const std::string label =
                    std::string(name) + " thinking=" +
                    (thinking ? (*thinking ? "true" : "false") : "unset") +
                    " prefix=" + nlohmann::json(prefix).dump();
                const auto fail = [&](const std::string& what) {
                    std::cerr << "[" << label << "] " << what << "\n";
                    ++failures;
                };
                ninferno::PromptInput input;
                ninferno::ChatMessage user;
                user.role = ninferno::ChatRole::User;
                user.parts.push_back(
                    {.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
                input.messages.push_back(std::move(user));
                if (!prefix.empty()) {
                    ninferno::ChatMessage assistant;
                    assistant.role = ninferno::ChatRole::Assistant;
                    assistant.parts.push_back(
                        {.kind = ninferno::MessagePartKind::Text, .text = prefix, .media = {}});
                    input.messages.push_back(std::move(assistant));
                    input.options.continuation =
                        ninferno::PromptContinuationMode::ContinueFinalAssistant;
                }
                input.options.enable_thinking = thinking;
                std::optional<PreparedPrompt> prompt;
                try {
                    prompt.emplace(frontend.prepare(std::move(input)));
                } catch (const std::exception&) {
                    if (!(prefix.empty() || thinking != true)) { continue; } // refused as required
                    fail("prompt refused");
                    continue;
                }
                if (!prefix.empty() && thinking == true) {
                    fail("an assistant prefix was accepted with thinking enabled");
                    continue;
                }
                const bool opened = prefix.empty() && thinking != false;
                if (prompt->summary().starts_in_reasoning != opened) {
                    fail(std::string("starts_in_reasoning is ") +
                         (opened ? "false" : "true") + " for this prompt tail");
                }
                for (const std::uint32_t round : {1U, kAll}) {
                    auto session = frontend.make_output_session(*prompt, {});
                    std::string content;
                    std::string reasoning;
                    bool first_delta = true;
                    std::size_t position = 0;
                    std::uint32_t remaining = 64;
                    for (int guard = 0; guard < 64; ++guard) {
                        const std::uint32_t size = std::min<std::uint32_t>(
                            round, static_cast<std::uint32_t>(output.size() - position));
                        const auto decision = session.preview_model(
                            std::span<const TokenId>(output).subspan(position, size), remaining,
                            FinishReason::OutputLimit);
                        position += decision.accepted_tokens;
                        remaining -= decision.accepted_tokens;
                        for (const ninferno::OutputDelta& delta : session.commit_preview()) {
                            if (first_delta && delta.channel == OutputChannel::Reasoning &&
                                delta.text.empty()) {
                                fail("the first delta is an empty reasoning delta");
                            }
                            first_delta = false;
                            (delta.channel == OutputChannel::Content ? content : reasoning) +=
                                delta.text;
                        }
                        if (decision.finished() || position == output.size()) { break; }
                    }
                    const std::string want_reasoning = opened ? "мысль" : "";
                    const std::string want_content =
                        opened ? "ответ" : "мысль</think>\n\nответ";
                    if (reasoning != want_reasoning || content != want_content) {
                        fail("round " + std::to_string(round == kAll ? 0 : round) +
                             ": reasoning=" + nlohmann::json(reasoning).dump() +
                             " content=" + nlohmann::json(content).dump());
                    }
                }
            }
        }
    }
    return failures;
}

int run_case(const Harness& harness, const Case& c, std::size_t& runs) {
    const Outcome reference = harness.run(c, {1});
    int failures            = 0;
    const auto fail         = [&](const std::string& what) {
        std::cerr << "[" << c.name << "] " << what << "\n";
        ++failures;
    };
    if (!reference.violation.empty()) { fail("one per round: " + reference.violation); }
    const auto quoted = [](const std::string& text) { return nlohmann::json(text).dump(); };
    if (c.content && reference.content != *c.content) {
        fail("content " + quoted(reference.content) + " != expected " + quoted(*c.content));
    }
    if (c.reasoning && reference.reasoning != *c.reasoning) {
        fail("reasoning " + quoted(reference.reasoning) + " != expected " + quoted(*c.reasoning));
    }
    if (c.finish && reference.finish != *c.finish) {
        fail("finish " + std::to_string(static_cast<int>(reference.finish)) + " != expected " +
             std::to_string(static_cast<int>(*c.finish)));
    }
    if (c.accepted && reference.accepted != *c.accepted) {
        fail("accepted " + std::to_string(reference.accepted) + " != expected " +
             std::to_string(*c.accepted));
    }
    if (c.tool_calls && reference.calls.size() != *c.tool_calls) {
        fail("tool calls " + std::to_string(reference.calls.size()) + " != expected " +
             std::to_string(*c.tool_calls) + " (" + describe(reference) + ")");
    }
    for (std::size_t i = 0; i < c.call_arguments.size() && i < reference.calls.size(); ++i) {
        if (reference.calls[i].second != c.call_arguments[i]) {
            fail("call " + std::to_string(i) + " arguments " + reference.calls[i].second +
                 " != expected " + c.call_arguments[i]);
        }
    }
    // Slice the consumed prefix plus one maximal overrun round; the rest is never read.
    const std::size_t length =
        std::min<std::size_t>(c.tokens.size(), std::max<std::uint32_t>(reference.accepted, 1) +
                                                   kMaxRound);
    int mismatches = 0;
    for (const auto& slicing : slicings(length, std::hash<std::string>{}(c.name))) {
        const Outcome got = harness.run(c, slicing);
        ++runs;
        if (got.violation.empty() && got == reference) { continue; }
        if (++mismatches > 3) { continue; }
        std::string cut;
        for (const std::uint32_t s : slicing) {
            cut += (cut.empty() ? "" : ",") + (s == kAll ? std::string("all") : std::to_string(s));
        }
        fail("slicing {" + cut + "}: " + (got.violation.empty() ? "" : got.violation + "; ") +
             describe(got) + "  vs one per round: " + describe(reference));
    }
    if (mismatches > 3) { fail(std::to_string(mismatches) + " slicings differ in total"); }
    return failures;
}

} // namespace

// --repetition-stop: an output whose last `window` tokens repeat one short period ends with
// FinishReason::Repetition at the token that completes the window; varied output does not, nor
// does a period too long to repeat four times in the window. Off, a loop runs to the limit.
int test_repetition_stop() {
    const Frontend frontend = make_fixture_frontend(R"({"eos_token_id":[248044]})");
    auto prompt             = frontend.prepare_tokens({kByteBase + 'x'});
    std::vector<TokenId> loop;
    for (int i = 0; i < 6; ++i) { loop.push_back(kByteBase + 'a' + i); }
    for (int i = 0; i < 30; ++i) {
        loop.push_back(kByteBase + 'p');
        loop.push_back(kByteBase + 'q');
    }
    const auto run = [&](std::uint32_t window, std::span<const TokenId> tokens) {
        ninferno::StopPolicy stop;
        stop.repetition_window = window;
        auto session           = frontend.make_output_session(prompt, stop);
        return session.preview_model(tokens, static_cast<std::uint32_t>(tokens.size()) + 1U,
                                     FinishReason::OutputLimit);
    };
    int failures   = 0;
    const auto hit = run(32, loop);
    // Six varied tokens, then "pq" from the 7th: 32 periodic tokens end at token 6 + 32.
    if (hit.finish_reason != FinishReason::Repetition || hit.accepted_tokens != 38) {
        std::cerr << "a two-token loop did not stop at its window\n";
        ++failures;
    }
    if (run(0, loop).finish_reason != FinishReason::None) {
        std::cerr << "repetition stop ran while off\n";
        ++failures;
    }
    std::vector<TokenId> varied;
    for (int i = 0; i < 64; ++i) { varied.push_back(kByteBase + 'A' + (i * 7) % 26); }
    // Period 26 needs 104 tokens in the window to count; a window of 64 sees fewer than four.
    if (run(64, varied).finish_reason != FinishReason::None) {
        std::cerr << "a period too long for the window was taken as a loop\n";
        ++failures;
    }
    return failures;
}

// --repetition-thinking conclude: a loop inside the thinking closes the thinking with the budget's
// control suffix and the answer follows; stop (the default) ends the output at the loop.
int test_repetition_concludes_thinking(const Harness& harness) {
    std::string loop = "ab";
    for (int i = 0; i < 16; ++i) { loop += "pq"; }
    Case c{.name              = "thinking loop",
           .tokens            = script({Piece{loop, true}, Piece{"Hi"}, Piece{kEosPiece}}),
           .prompt            = PromptKind::Thinking,
           .max_tokens        = 400,
           .repetition_window = 32};
    int failures      = 0;
    const Outcome cut = harness.run(c, {1});
    if (cut.finish != FinishReason::Repetition || !cut.content.empty()) {
        std::cerr << "a thinking loop under stop did not end the output: " << describe(cut) << "\n";
        ++failures;
    }
    c.repetition_concludes_thinking = true;
    const Outcome closed            = harness.run(c, {1});
    if (closed.finish != FinishReason::StopToken || closed.content != "Hi" ||
        closed.thinking_injected == 0 || closed.reasoning.rfind("abpq", 0) != 0) {
        std::cerr << "a thinking loop under conclude did not close the thinking: "
                  << describe(closed) << "\n";
        ++failures;
    }
    return failures;
}

int main() {
    try {
        const Harness harness;
        int failures     = 0;
        std::size_t runs = 0;
        const auto cases = corpus();
        for (const Case& c : cases) { failures += run_case(harness, c, runs); }
        failures += test_turn_terminator_joins_default_stops();
        failures += test_reasoning_start_state_matrix();
        failures += test_repetition_stop();
        failures += test_repetition_concludes_thinking(harness);
        std::cout << "output slicing: " << cases.size() << " scripts, " << runs
                  << " slicings, failures=" << failures << "\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "output slicing harness error: " << error.what() << "\n";
        return 1;
    }
}
