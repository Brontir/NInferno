#pragma once

// What every architecture adapter shares: the checks config.json fields pass
// (tools/convert/config_checks.py), the frontend resources and token domain (resources.py), and
// the frequency shortlist (proposal.py).

#include "convert/files.h"
#include "convert/model.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::convert {

// Tokenizer resources hold objects of 262k members; ordered_json inserts in linear time, so they
// are parsed unordered (token_domain does not depend on member order).
using Json = nlohmann::json;

template <class J>
const J* member(const J& object, std::string_view key) {
    if (!object.is_object()) { return nullptr; }
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &*it;
}

[[nodiscard]] bool present(const OJson* value);
// A positive integer (a bool is not one).
[[nodiscard]] std::uint64_t positive(const OJson* value, const std::string& name);
// A positive finite value representable as FP32; returns the FP32 value as a Python float.
[[nodiscard]] double f32(const OJson* value, const std::string& name);
[[nodiscard]] double f32_value(const OJson* value, double fallback, const std::string& name);
// Python `a == b` for JSON values.
[[nodiscard]] bool py_equal(const OJson& a, const OJson& b);
// `key`, when present, must equal `value` (with a bool `value`, it must be a bool).
void fixed(const OJson& config, std::string_view key, const OJson& value, const std::string& label);
// Whether a source's config.json describes a model rather than only its bookkeeping.
[[nodiscard]] bool has_model_config(const OJson& config);
// An additional source must agree with the model on every field its site depends on.
void check_source_fields(const std::string& name, const OJson& actual, const OJson& expected,
                         const std::set<std::string>& fields);

// The one architecture a checkpoint's config declares; throws otherwise.
[[nodiscard]] std::string declared_architecture(const OJson& config);

// proposal.add_official_proposal: an indexed Q4 proposal head gathered from the text head's
// default source, with its int32 token ids.
// `drafters`: the components whose final hidden states it takes (empty: every one the model has
// of mtp, dflash, dflash2).
void add_official_proposal(Model& model, Recipe& recipe, const std::filesystem::path& ranking,
                           const std::string& ranking_text, std::uint64_t rows,
                           const std::vector<std::string>& drafters = {});

using ResourceOverrides = std::vector<std::pair<std::string, std::filesystem::path>>;
using ResourceFallbacks = std::map<std::string, std::string>;

// The text resources (and, with a Vision config, the preprocessor configs) as artifact
// resources, their references in the components, and the tokenizer's public token domain.
// A role neither overridden nor present in `root` takes its bytes from `fallbacks`, when there
// (recorded in model.resource_notes); so do the tokenizer roles when `root`'s tokenizer.json has
// another vocabulary than the fallback's (a GGUF in a folder with another model's files).
void load_resources(Model& model, const std::filesystem::path& root, std::uint64_t vocab_size,
                    const ResourceOverrides& overrides, const OJson* vision = nullptr,
                    const ResourceFallbacks& fallbacks = {});

// Resources a checkpoint can stand in for when its directory lacks them: those a GGUF file's
// metadata rebuilds, and a generation_config.json derived from config.json and
// tokenizer_config.json (end-of-sequence ids) for a checkpoint published without one.
[[nodiscard]] ResourceFallbacks resource_fallbacks(const Checkpoint& checkpoint);

// The ranking a proposal head is restricted by when --ranking names none
// (tools/freq_corpus/fixtures/ranking, installed with the registry's rankings).
inline constexpr const char* kDefaultProposalRanking = "ranking.train.counts.i64";

// The proposal head a Qwen target with MTP layers or a DFlash drafter gets without --proposal
// (spec/draft_head.h), for those drafters (DFlash2's fused top-k runs only on the 131072-row Q4
// head, so it keeps the full head): for each, the rows from the ranking's counts and the share of
// its round's stored bytes the full draft head reads (MTP: k heads over the target's pass and k
// MTP passes; DFlash: one over the target's pass, the drafter's component and the head), at its
// default depth; the head keeps the most rows any of them asks for. rows 0: the target has no
// such drafter, its facts are unreadable, or no restriction pays. `note` says what the choice
// rested on.
struct DefaultProposal {
    std::uint64_t rows = 0;
    std::vector<std::string> drafters; // the components it serves ("mtp", "dflash")
    std::string note;
};
[[nodiscard]] DefaultProposal default_proposal(const Model& model, const Recipe& recipe,
                                               const std::filesystem::path& ranking);
// The bytes the proposal head default_proposal gives a target take, from what a plan knows before
// the conversion: the head's stored bytes (0: unknown, an unquantized checkpoint's: stored as its
// default recipe stores it, qwen3_5_recipe_head_format of qwen3_5_checkpoint_recipe(`moe`), its
// rows gathered to the recipe's Q4), the target's weights, its MTP layers' (0: none) and its DFlash drafter's (0:
// none), the vocabulary and width. The rows from the default ranking as default_proposal finds
// it (the most MTP 3 and DFlash 7 ask for), with their int32 token ids; 0 when it is not found
// or nothing pays.
[[nodiscard]] std::uint64_t planned_proposal_bytes(double head_bytes, double target_bytes,
                                                   double mtp_bytes, double drafter_bytes,
                                                   std::uint64_t vocab, std::uint64_t hidden,
                                                   bool moe = false);

// select_shortlist(counts, n, force_include).
[[nodiscard]] std::vector<std::int64_t> select_shortlist(const std::vector<std::int64_t>& counts,
                                                         std::uint64_t n,
                                                         const std::vector<std::int64_t>& forced);
// shortlist_ids: the `rows` most frequent ids of the model's token domain by a ranking file.
[[nodiscard]] std::vector<std::int64_t>
shortlist_ids(const Model& model, const std::filesystem::path& ranking, std::uint64_t rows);

} // namespace ninferno::convert
