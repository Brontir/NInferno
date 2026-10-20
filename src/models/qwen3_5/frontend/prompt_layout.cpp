#include "models/qwen3_5/frontend/prompt_layout.h"

#include <algorithm>
#include <stdexcept>

namespace ninferno::models::qwen3_5::frontend {
namespace {

constexpr std::string_view kStart       = "<|im_start|>";
constexpr std::string_view kVisionStart = "<|vision_start|>";
constexpr std::string_view kVisionEnd   = "<|vision_end|>";

bool template_bytes(const text::TemplateOutput& output, std::size_t begin, std::size_t end) {
    return !text::overlaps(output.literal_spans, begin, end);
}

std::size_t find_marker(const text::TemplateOutput& output, std::string_view marker,
                        std::size_t begin) {
    for (auto pos = output.text.find(marker, begin); pos != std::string::npos;
         pos      = output.text.find(marker, pos + marker.size())) {
        if (template_bytes(output, pos, pos + marker.size())) return pos;
    }
    return std::string::npos;
}

} // namespace

std::optional<text::ByteSpan> unique_output_region(const text::TemplateOutput& output,
                                                   std::uint32_t tag) {
    std::optional<text::ByteSpan> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag) continue;
        if (result) return std::nullopt;
        result = text::ByteSpan{region.begin, region.end};
    }
    return result;
}

std::optional<std::size_t> source_boundary(const text::TemplateOutput& output, std::uint32_t tag,
                                           std::size_t offset) {
    std::optional<std::size_t> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag || !region.source_offset) continue;
        const auto begin = *region.source_offset;
        if (offset < begin || offset - begin > region.end - region.begin) continue;
        const auto mapped = region.begin + offset - begin;
        if (result && *result != mapped) return std::nullopt;
        result = mapped;
    }
    return result;
}

PromptLayout inspect_prompt_layout(const text::TemplateOutput& output,
                                   std::span<const Modality> media,
                                   const chat::ChatConventions& conventions) {
    PromptLayout result;
    const std::string_view source = output.text;
    const auto& turns             = conventions.turns;
    const auto& reasoning         = conventions.reasoning;
    const std::string_view start  = turns ? std::string_view(turns->open) : std::string_view{};
    // Whether a request could end with the turn before this one, so that its generation prompt
    // (and its TurnClosure or ResponseReplay checkpoint) opens at this turn's begin.
    bool request_could_end_before = false;
    for (auto pos = turns ? find_marker(output, start, 0) : std::string::npos;
         pos != std::string::npos;) {
        auto header_end = source.find(turns->header_end, pos + start.size());
        if (header_end == std::string::npos) {
            // A generation prompt may end inside the header ("<|start|>assistant" of harmony):
            // the model writes the rest of it (a channel) itself. That turn is the open assistant
            // turn at the end of the prompt, its body empty.
            header_end = source.size();
        }
        const auto body_begin = std::min(source.size(), header_end + turns->header_end.size());
        const auto role =
            turns->role(source.substr(pos + start.size(), header_end - pos - start.size()));
        if (!role) {
            // A turn the conventions do not name (a harmony tool call or tool response addressed
            // to a recipient) counts as a tool turn for the replacement-suffix boundary below.
            request_could_end_before = true;
            pos                      = find_marker(output, start, body_begin);
            continue;
        }
        const auto next  = find_marker(output, start, body_begin);
        const auto close = find_marker(output, turns->close, body_begin);
        const bool closed =
            close != std::string::npos && (next == std::string::npos || close < next);
        const auto content_end = closed ? close : next == std::string::npos ? source.size() : next;
        auto end               = closed ? close + turns->close.size() : content_end;
        if (closed && !turns->after_close.empty() &&
            source.substr(end).starts_with(turns->after_close))
            end += turns->after_close.size();
        if (pos > 0 && request_could_end_before && *role != ChatRole::Assistant &&
            *role != ChatRole::Tool) {
            // A request ending with the previous turn holds its TurnClosure or ResponseReplay
            // checkpoint at this byte, where its generation prompt opened. A request that appends
            // a user, system or developer message there instead (a replacement suffix) must split
            // its prefill here too, or its cold decomposition differs from that checkpoint's and
            // the checkpoint can never be restored for it. An assistant turn splits here anyway
            // (below). A tool turn does not: consecutive tool results (one turn each in ChatML,
            // Gemma 4 and Llama 3 ipython) would each pay a short extra prefill call for a branch
            // that replaces a tool result, which almost never reuses. Every rule reads only the
            // turns up to this one, so each prompt sharing the prefix splits it alike.
            result.execution_boundaries.push_back(pos);
        }
        // A user or tool turn can end a request, and so can a system or developer message after
        // the conversation began (a mid-conversation instruction). The leading instructions do
        // not: every system and developer message before the first other turn (harmony renders
        // a system then a developer message ahead of the first user turn), as no request ends
        // there.
        const bool leading_instruction =
            std::all_of(result.messages.begin(), result.messages.end(), [](const auto& message) {
                return message.role == ChatRole::System || message.role == ChatRole::Developer;
            });
        request_could_end_before =
            *role == ChatRole::User || *role == ChatRole::Tool ||
            ((*role == ChatRole::System || *role == ChatRole::Developer) && !leading_instruction);
        result.messages.push_back({*role, pos, body_begin, content_end, end, closed});
        if (*role == ChatRole::Assistant) {
            // The message begin is where a TurnClosure or ResponseReplay checkpoint lies. It is an
            // execution boundary of every rendering of the prompt, so a prefill splits there
            // whether or not the cache captures or restores it, and a restore there is exact.
            if (pos > 0) result.execution_boundaries.push_back(pos);
            result.execution_boundaries.push_back(body_begin);
            const auto body = source.substr(body_begin, content_end - body_begin);
            if (reasoning) {
                const std::string open = reasoning->open + reasoning->open_suffix;
                const std::string_view canonical = reasoning->canonical_close;
                if (body.starts_with(open) &&
                    template_bytes(output, body_begin, body_begin + open.size())) {
                    result.execution_boundaries.push_back(body_begin + open.size());
                    const auto reasoning_close =
                        canonical.empty() ? std::string::npos
                                          : find_marker(output, canonical, body_begin + open.size());
                    if (reasoning_close != std::string::npos && reasoning_close < content_end) {
                        result.execution_boundaries.push_back(reasoning_close + canonical.size());
                    }
                }
            }
            if (!closed && next == std::string::npos) {
                // Only the final assistant prefix controls the initial output channel.
                if (reasoning) {
                    const std::string_view marker = reasoning->open;
                    const auto open_end =
                        body.starts_with(marker) &&
                                template_bytes(output, body_begin, body_begin + marker.size())
                            ? body_begin + marker.size()
                            : std::string::npos;
                    const auto close_think = open_end == std::string::npos
                                                 ? std::string::npos
                                                 : find_marker(output, reasoning->close, open_end);
                    result.starts_in_reasoning =
                        open_end != std::string::npos && close_think == std::string::npos;
                    result.reasoning_open_allowed =
                        body.find(marker) == std::string_view::npos;
                }
            }
        }
        pos = next;
    }

    // Each media input contributes one complete placeholder in input order: Qwen's pad inside its
    // vision wrapper, or Gemma 4's bare <|image|> (the processor adds its own delimiters).
    for (std::size_t pos = 0; pos < source.size();) {
        const auto image = find_marker(output, "<|image_pad|>", pos);
        const auto video = find_marker(output, "<|video_pad|>", pos);
        const auto gemma = find_marker(output, "<|image|>", pos);
        if (gemma != std::string::npos && gemma < std::min(image, video)) {
            constexpr std::string_view token = "<|image|>";
            if (!template_bytes(output, gemma, gemma + token.size())) {
                throw std::invalid_argument("chat template media marker is not template text");
            }
            const auto index = result.media_placeholders.size();
            if (index >= media.size() || media[index] != Modality::Image) {
                throw std::invalid_argument(
                    "chat template media placeholders must match input count, type and order");
            }
            result.media_placeholders.push_back(
                {{gemma, gemma + token.size()}, Modality::Image, index, VisionStyle::Gemma4});
            pos = gemma + token.size();
            continue;
        }
        const auto pad = std::min(image, video);
        if (pad == std::string::npos) break;
        const auto modality = pad == image ? Modality::Image : Modality::Video;
        const std::string_view token =
            modality == Modality::Image ? "<|image_pad|>" : "<|video_pad|>";
        if (pad < kVisionStart.size() ||
            source.substr(pad - kVisionStart.size(), kVisionStart.size()) != kVisionStart ||
            source.substr(pad + token.size(), kVisionEnd.size()) != kVisionEnd ||
            !template_bytes(output, pad - kVisionStart.size(),
                            pad + token.size() + kVisionEnd.size())) {
            throw std::invalid_argument(
                "chat template media pad requires a complete Qwen vision wrapper");
        }
        const auto index = result.media_placeholders.size();
        if (index >= media.size() || media[index] != modality) {
            throw std::invalid_argument(
                "chat template media placeholders must match input count, type and order");
        }
        result.media_placeholders.push_back({{pad, pad + token.size()}, modality, index});
        pos = pad + token.size();
    }
    if (result.media_placeholders.size() != media.size()) {
        throw std::invalid_argument("chat template omitted an input media placeholder");
    }
    return result;
}

} // namespace ninferno::models::qwen3_5::frontend
