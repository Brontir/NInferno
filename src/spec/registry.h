#pragma once

// The official drafter registry: a curated JSON the project publishes (src/spec/data/
// drafter_registry.json is compiled in; --drafter-registry FILE names a newer copy on disk). It
// maps model ids (a base model, its quantized variants and known fine-tunes) to the drafters
// NInferno verified for them and to the speculation profiles measured per GPU class. Lookup is
// deterministic: an exact repository id, then the base-model lineage the checkpoint itself
// declares (its model card), then the nearest registered model by what the checkpoint is (its
// engine, vocabulary and geometry: Registry::nearest), never a search.
//
// A model bundle can declare the same thing itself: ninferno_speculation.json in the checkpoint
// (a "model" object of this schema without "match"). A bundle declaration wins over the registry.

#include "spec/pairs.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::spec {

struct DrafterDecl {
    std::string name;      // referenced by profiles ("assistant", "dflash", "mtp", ...)
    std::string kind;      // mtp, gemma4_assistant, dflash, dflash2
    std::string repo;      // hub repository ("" for the target's own MTP layers)
    std::string revision;  // pinned commit ("" = the hub's default)
    std::string recipe;    // converter recipe of a separately packaged drafter
    std::string component; // artifact component of a builtin or companion drafter
    std::string license;
    std::string ranking;   // shortlist ranking file name
    // A drafter published only as a GGUF file (llama.cpp's DFlash drafter, a NemotronH MTP
    // sidecar) in `repo`, beside other GGUF models: the file selector (a case-insensitive part of
    // its name, as ninferno-convert --quant takes it). Its metadata, read as the checkpoint config
    // llama.cpp converted it from, stands in for the config.json such a repository lacks.
    std::string gguf;
    std::uint64_t proposal_rows = 0;
    // NInferno's own quantizations of this drafter (q8, q4, nvfp4: ninferno-convert
    // --drafter-quant) that the converter builds from the same download beside it, each attached as
    // the drafter
    // "<name>.<variant>". A drafter only proposes tokens and the target verifies every one, so a
    // variant changes the speed, never the output; the profiles of each GPU class name the one
    // measured fastest there. Separately packaged drafters only (not a companion component).
    std::vector<std::string> variants;
    // A variant drafter's quantization (plan_drafters sets it on "<name>.<variant>"); "" for the
    // drafter as its recipe leaves it.
    std::string quant;
};

// The drafter "<name>.<variant>" stands for; the variant is "" for a plain name.
[[nodiscard]] std::pair<std::string, std::string> split_variant(std::string_view name);

struct Geometry {
    std::uint64_t hidden_size = 0, num_hidden_layers = 0, vocab_size = 0;
};

// How far apart two model shapes are: |log2| of the hidden-size ratio plus |log2| of the
// layer-count ratio (0: the same shape; 1: twice as wide or twice as deep). Throws on an
// incomplete geometry.
[[nodiscard]] double shape_distance(const Geometry& a, const Geometry& b);

// The engine family the registry names for a checkpoint architecture (its config's
// "architectures"[0]): "gemma4", "qwen3_5", "nemotron_h" or "dense"; empty for one no NInferno
// engine runs.
[[nodiscard]] std::string engine_for_architecture(std::string_view architecture);

// Past it, a registered model says nothing about another one's speculation: Qwen3.5 4B and 9B
// (0.68) are near, 0.8B and 9B (2.0) are not.
inline constexpr double kMaxModelDistance = 1.0;

struct ModelEntry {
    std::string id;
    std::string engine; // gemma4 | qwen3_5 | nemotron_h | dense
    std::vector<std::string> repos;   // exact ids: the model, quantized variants, fine-tunes
    std::vector<std::string> lineage; // base models whose declared descendants this entry covers
    std::string vocab_fingerprint;
    Geometry geometry;
    std::vector<DrafterDecl> drafters;
    nlohmann::json profiles = nlohmann::json::array(); // see spec/profile.h
    std::string evidence;
};

// A drafter kind's rule, for every drafter of that kind that does not name its own: the converter
// recipe, NInferno's quantization of its plain projections (a drafter only proposes and the target
// verifies every token, so it changes the speed, never the output; a published encoding is kept),
// and the shortlist its head is restricted to (a Gemma 4 assistant's own head).
struct KindInfo {
    std::string name;
    std::vector<std::string> architectures;
    std::string recipe;
    std::string quant;               // q8 | q4 | nvfp4; "" as the recipe leaves it
    std::uint64_t proposal_rows = 0; // 0: the full head
    std::string ranking;             // the shortlist ranking proposal_rows reads
};

// A separately packaged drafter (no component) takes its kind's rule where it names none: the
// registry's own entries when it is read, a bundle's declared ones when they are planned.
void apply_kind_rule(DrafterDecl& d, const std::vector<KindInfo>& kinds);

class Registry {
public:
    [[nodiscard]] static Registry parse(const std::string& text);
    // The compiled-in registry, or the file use_file() named.
    [[nodiscard]] static const Registry& builtin();
    // The products' --drafter-registry FILE: builtin() reads this file instead of the compiled-in
    // registry. Called before the first builtin(); a later call naming another file throws
    // std::logic_error (the registry already answered from the other one).
    static void use_file(const std::filesystem::path& path);
    // The file use_file() named; empty: the compiled-in registry.
    [[nodiscard]] static std::filesystem::path file();

    [[nodiscard]] const std::vector<ModelEntry>& models() const noexcept { return models_; }
    [[nodiscard]] const std::vector<KindInfo>& kinds() const noexcept { return kinds_; }
    [[nodiscard]] const std::string& published() const noexcept { return published_; }
    // Bumped by tools/spec/registry_add.py on every change (0: a registry without one), so a
    // newer registry can be told from an older one; each profile carries its "measured_at" date
    // and "revision" (a hash of its content).
    [[nodiscard]] std::uint64_t version() const noexcept { return version_; }

    // SHA-256 (lowercase hex) of the text builtin() parsed (the compiled-in registry or the file
    // use_file() named); empty for a registry parse() built.
    [[nodiscard]] const std::string& source_sha256() const noexcept { return source_sha256_; }

    // The entry naming `repo` exactly (case-insensitive).
    [[nodiscard]] const ModelEntry* by_repo(std::string_view repo) const;
    // The entry covering a model whose declared base-model lineage contains one of its lineage
    // repositories.
    [[nodiscard]] const ModelEntry* by_lineage(std::string_view base) const;
    [[nodiscard]] const ModelEntry* by_id(std::string_view id) const;

    // The registered model nearest to one no repository or lineage names: the same engine and
    // vocabulary (vocab_size, and the fingerprint where both carry one), a complete geometry, the
    // smallest shape_distance within kMaxModelDistance (ties: the first listed). A model of the
    // same shape is the entry's model under another name (a requantization, an undeclared
    // fine-tune): every drafter of the entry applies to it. One of another shape shares only what
    // does not depend on the shape, the choices of the MTP layers it carries itself, so it is
    // placed only when `with_mtp` (the target carries MTP layers) and only by an entry that
    // declares a drafter of kind "mtp". nullptr: no entry is near enough.
    [[nodiscard]] const ModelEntry* nearest(std::string_view engine, const Geometry& geometry,
                                            std::string_view fingerprint, bool with_mtp) const;

    // The pairs the lab measured (spec/pairs.h: a target's weights and its drafters, one card each).
    [[nodiscard]] const std::vector<MeasuredPair>& pairs() const noexcept { return pairs_; }
    // The measurements of the target whose weights are `target` (MeasuredPair::target), every card.
    [[nodiscard]] std::vector<const MeasuredPair*> measured(std::string_view target) const;

    // The drafter kind a checkpoint architecture belongs to (nullopt: not a known drafter).
    [[nodiscard]] std::optional<std::string> kind_for_architecture(std::string_view arch) const;
    // Whether an engine family (gemma4, qwen3_5, dense, nemotron_h) serves drafters of this kind:
    // the serving Engine's answer (spec/engine_support.h engine_runs_kind), for a kind listed here.
    [[nodiscard]] bool supported(std::string_view kind, std::string_view engine) const;

private:
    std::vector<ModelEntry> models_;
    std::vector<KindInfo> kinds_;
    std::vector<MeasuredPair> pairs_;
    std::string published_;
    std::uint64_t version_ = 0;
    std::string source_sha256_;
};

// A model entry from a bundle declaration or a registry "models" item.
[[nodiscard]] ModelEntry parse_model_entry(const nlohmann::json& item);
[[nodiscard]] nlohmann::json drafter_json(const DrafterDecl& d);
[[nodiscard]] DrafterDecl parse_drafter(const nlohmann::json& d);


} // namespace ninferno::spec
