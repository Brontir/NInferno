#pragma once

// Quantization quality against a reference model (ninferno-perplexity --kld-out / --kld): the
// reference run records, for every scored position, its top-K tokens with their log-probabilities
// and the target's log-probability; the measured run reads its own log-probabilities of the same
// tokens. Per position the KL divergence KL(P_ref || Q) is taken over the K listed tokens plus one
// outcome for everything else (the unlisted mass of each), a lower bound of the full-vocabulary
// divergence that is tight when the K tokens hold nearly all of P (K = 32 holds > 0.99 of a
// language model's next-token mass at almost every position).

#include "ninferno/types.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace ninferno::perplexity {

inline constexpr std::uint32_t kKldDefaultTopK = 32;

// One stream of the reference: its tokens' hash (the measured run must score the same tokens) and
// per target position (token 1 .. n-1) top_k ids and log-probabilities and the target's
// log-probability.
struct KldStream {
    std::string id;
    std::uint64_t token_hash = 0;
    std::uint64_t targets    = 0;
    std::vector<TokenId> ids;           // [targets][top_k]
    std::vector<float> logprobs;        // [targets][top_k]
    std::vector<float> target_logprobs; // [targets]
};

struct KldReference {
    std::uint32_t top_k   = 0;
    std::uint32_t context = 0;     // the windows the reference scored: the measured run
    std::uint32_t stride  = 0;     // must condition on the same histories
    std::string model;             // the reference artifact's model name
    std::string prefill_signature; // and its weight formats, for the report
    std::vector<KldStream> streams;
};

[[nodiscard]] std::uint64_t token_hash(std::span<const TokenId> tokens);

void write_kld_reference(const std::filesystem::path& path, const KldReference& reference);
[[nodiscard]] KldReference read_kld_reference(const std::filesystem::path& path);

// KL(P || Q) over the listed tokens and the unlisted rest: p and q are the two models'
// log-probabilities of the same listed tokens (-inf for an empty slot, in both).
[[nodiscard]] double position_kld(std::span<const float> p, std::span<const float> q);

// The measured model against the reference over many positions.
struct KldAggregate {
    std::vector<double> kld;       // per position
    std::uint64_t top1_same = 0;   // positions whose most probable token is the reference's
    double reference_nll    = 0.0; // sum over positions
    double nll              = 0.0;

    void add(double position_kld, bool same_top1, double reference_logprob, double logprob);
    void add(const KldAggregate& other);

    [[nodiscard]] std::uint64_t positions() const { return kld.size(); }

    [[nodiscard]] double mean() const;
    // The q-quantile (0..1) of the per-position divergence.
    [[nodiscard]] double quantile(double q) const;
    [[nodiscard]] double top1_agreement() const;
    // PPL(measured) / PPL(reference) over the same positions.
    [[nodiscard]] double ppl_ratio() const;
};

} // namespace ninferno::perplexity
