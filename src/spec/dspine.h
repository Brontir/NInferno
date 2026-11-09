#pragma once

// DSpine (Tencent, arXiv 2609.36173): a DFlash drafter whose layers pass each position's predicted
// token code to the next position, and whose last write is redone per candidate predecessor into a
// transition table. This is the host side that needs no GPU: reading and validating a checkpoint's
// config, its tensor contract, the facts a round's cost rests on and a reference of the arithmetic
// (docs/maintainer/dspine.md owns the mathematics). The engine does not run it yet; the converter
// recognizes it so that no DSpine checkpoint is read as a plain DFlash drafter.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninferno::spec {

struct DSpineConfig {
    std::uint32_t hidden_size       = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t layers            = 0;
    std::uint32_t heads             = 0;
    std::uint32_t kv_heads          = 0;
    std::uint32_t head_dim          = 0;
    std::uint32_t vocab_size        = 0;
    std::uint32_t block_size        = 0; // rows of one draft block, the anchor included
    std::uint32_t sliding_window    = 0; // 0: every layer attends fully
    std::uint32_t mask_token_id     = 0;
    std::uint32_t code_size         = 0; // C: width of a token code
    std::uint32_t rank              = 0; // R: width of a message
    std::uint32_t head_topk         = 0; // K: candidates per row whose last write is redone
    float rms_norm_eps              = 0;
    float rope_theta                = 0;
    float bridge_eps                = 0;
    std::vector<std::uint32_t> target_layer_ids;

    // Proposals one block carries: every row after the anchor.
    [[nodiscard]] std::uint32_t proposals() const noexcept { return block_size - 1; }
};

// True when a draft config.json carries the code bridge (also a "DFlashDraftModel" by name).
[[nodiscard]] bool is_dspine_config(const nlohmann::json& config);

// The config of a DSpine draft, refusing every variant the arithmetic below does not cover (a
// terminal-corrected bridge, next-token rows, DFlash2 selector keys, a partial bridge, ...).
// Throws std::runtime_error naming the field.
[[nodiscard]] DSpineConfig parse_dspine_config(const nlohmann::json& config);

struct DSpineTensor {
    std::string name; // as the Hugging Face checkpoint stores it
    std::vector<std::uint64_t> shape;
    bool bridge = false; // part of the code bridge (kept apart from a plain DFlash layer)

    [[nodiscard]] std::uint64_t elements() const noexcept;
};

// Every floating tensor of the checkpoint (the four int64 version markers are not listed).
[[nodiscard]] std::vector<DSpineTensor> dspine_tensors(const DSpineConfig& config);
[[nodiscard]] std::uint64_t dspine_parameter_count(const DSpineConfig& config);

// What one draft round of `width` rows (the block, 2..block_size) of one request reads and
// computes, in the model's own terms; the caller prices the bytes at its format.
struct DSpineRoundFacts {
    std::uint64_t backbone_params = 0; // fc + layers + norms: read whole each round
    std::uint64_t bridge_matrices = 0; // read/write/gate matrices: read whole each round
    std::uint64_t codebook_values = 0; // rows touched: the anchor's + one per candidate lane
    std::uint64_t head_values     = 0; // the target head's rows of the candidates
    std::uint64_t multiply_adds   = 0; // per request, the dense parts of the pass
};

[[nodiscard]] DSpineRoundFacts dspine_round_facts(const DSpineConfig& config, std::uint32_t width);

// ------------------------------------------------------------------------------------------
// Reference arithmetic (float, one row at a time): the contract the device kernels match.

// z -> sqrt(C) z / (|z| + eps): the code a hidden state reads.
void dspine_unit_code(std::span<const float> z, float eps, std::span<float> out);

// The anchor row's code: sqrt(C) times its token's codebook row.
void dspine_anchor_code(std::span<const float> codebook_row, std::span<float> out);

// sigmoid(query + sender + bias) * value, per rank lane: row t's message from row t-1's code.
void dspine_message(std::span<const float> query, std::span<const float> sender,
                    std::span<const float> value, std::span<const float> bias,
                    std::span<float> out);

// raw + rms * delta, the write of one row (the anchor row of a block is never written).
void dspine_write(std::span<const float> raw, std::span<const float> delta, float rms,
                  std::span<float> out);

// The walk over transition tables. edges[(i * k + p) * k + c] is the score of candidate lane c of
// proposal i when proposal i-1 took lane p; proposal 0's predecessor is the anchor, so only
// p = 0 is read for i = 0. Greedy: the lowest lane attaining the maximum. Returns the lane of
// every proposal.
[[nodiscard]] std::vector<std::uint32_t>
dspine_walk_greedy(std::span<const float> edges, std::uint32_t k, std::uint32_t proposals);

// The same walk drawing from softmax(edge / temperature), with `uniforms` in [0,1) one per
// proposal. `q` (proposals x k) receives the distribution of every proposal along the walk.
[[nodiscard]] std::vector<std::uint32_t>
dspine_walk_sample(std::span<const float> edges, std::uint32_t k, std::uint32_t proposals,
                   float temperature, std::span<const float> uniforms, std::span<float> q);

} // namespace ninferno::spec
