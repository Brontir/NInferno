#pragma once

// The role of an artifact parameter that --activations reads, by its canonical name (the names the
// converter writes, docs/reference/artifact-format.md): a header with no dependency, so the
// converter's publisher-validated count and the engine's loaders share it.

#include <array>
#include <string_view>

namespace ninferno::models {

// Whether a parameter is a head or an embedding: a component's own matrix (one name segment after
// the component, never a layer's) named as a logits head or a token, patch or position embedding.
// --activations 8|4 leave these at a recorded BF16. "text/output_head" also serves the Qwen MTP
// head (bound with input "mtp/final_hidden"); "proposal/head" is the DFlash2 proposal head.
[[nodiscard]] constexpr bool logits_or_embedding(std::string_view parameter) noexcept {
    const auto slash = parameter.find('/');
    if (slash == std::string_view::npos) { return false; }
    const auto role = parameter.substr(slash + 1);
    constexpr std::array<std::string_view, 6> kRoles{"output_head",     "head",
                                                     "token_embedding", "embed_tokens",
                                                     "patch_embedding", "position_embedding"};
    for (const auto known : kRoles) {
        if (role == known) { return true; }
    }
    return false;
}

static_assert(logits_or_embedding("text/output_head") && logits_or_embedding("text/embed_tokens") &&
              logits_or_embedding("text/token_embedding") && logits_or_embedding("proposal/head") &&
              logits_or_embedding("vision/patch_embedding"));
static_assert(!logits_or_embedding("text/layers/0/mlp/down") &&
              !logits_or_embedding("text/layers/0/attention/head_gate") &&
              !logits_or_embedding("text/feature_projection") && !logits_or_embedding("head"));

} // namespace ninferno::models
