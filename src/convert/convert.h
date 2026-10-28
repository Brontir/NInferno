#pragma once

// Native conversion: a Hugging Face checkpoint or a GGUF model to a .ninferno v3 artifact and its
// .conversion.json report. For Gemma 4 and Qwen3.5 recipes the artifact is byte-identical to
// `python -m tools.convert` with the same arguments (apart from the random artifact id and
// timing fields); the other families convert natively only.

#include "convert/expert_layout.h"
#include "convert/files.h"
#include "convert/gguf_request.h"
#include "convert/pyjson.h"
#include "core/units.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::convert {

struct ConvertOptions {
    std::filesystem::path model;     // --model
    std::string recipe;              // --recipe (official name)
    // --vocab native|q8 (dense and MoE recipes): the token embedding and output head as the
    // checkpoint publishes them, or quantized to q8_g32_fp16; empty: the recipe's default.
    std::string vocab;
    // --drafter-quant q8|q4|nvfp4: a drafter's plain projections in NInferno's own quantization
    // (drafters only: the target verifies every drafted token, so the output is unchanged);
    // empty: as the recipe leaves them.
    std::string drafter_quant;
    // drafter_quant is a default (a registry drafter's "quant"), not --drafter-quant: one that
    // changes nothing (the drafter is published quantized, and kept) is expected and not warned of.
    bool drafter_quant_declared = false;
    std::filesystem::path out;       // --out
    std::optional<std::string> name; // --name
    std::vector<std::pair<std::string, std::filesystem::path>> resources; // --resource ROLE=PATH
    std::vector<std::pair<std::string, std::filesystem::path>> sources;   // --source NAME=PATH
    std::string components      = "text";                                 // --components
    // --components not given: also keep the drafters the model carries (a checkpoint's or GGUF's
    // MTP layers) and those named by --source, instead of converting the text alone.
    bool components_auto        = false;
    bool proposal               = false;                                  // --proposal
    std::uint64_t proposal_rows = 131072;
    // --no-proposal clears it: without --proposal, a Qwen target converted with a DFlash drafter
    // gets the proposal head default_proposal chooses for DFlash (checks.h).
    bool proposal_default = true;
    std::filesystem::path ranking;         // --ranking (defaults to the repository's ranking)
    std::string ranking_label;             // provenance spelling of the ranking ("": its path)
    std::uint64_t rows_per_chunk = 512;    // recorded in the report; chunking is internal
    std::uint64_t max_file_bytes = kDefaultMaxFileBytes;
    unsigned threads             = 0;            // 0: hardware concurrency, at most 16
    bool keep_source_cache       = false;        // leave read checkpoint pages in the page cache
    std::function<void(const std::string&)> log; // progress lines; empty for silence

    // --consume-source: free the checkpoint's space while converting (it is destroyed). The
    // artifact is still a new file; a crash leaves a journal and the run resumes.
    bool consume_source = false;
    // Output bytes of uncommitted objects in flight when consuming (bounds the peak).
    std::uint64_t consume_window = kGiB;
    // --assume-free-bytes: the free space the checks assume (tests); unset: statvfs.
    std::optional<std::uint64_t> assume_free_bytes;
    // --maintainer-quantize: allow what a conversion otherwise refuses, because the artifact
    // would no longer be the published model: a quantized checkpoint's matrices dequantized
    // (and quantized again, or kept with rounded scales), or the projections of a checkpoint
    // without a published quantization quantized by NInferno's own recipe.
    bool maintainer_quantize = false;
    // --round-tiny-scales: an integer site whose only inexact binary16 scales are below binary16's
    // normal range is kept with them rounded instead of decoded (changes those weights).
    bool round_tiny_scales = false;
    // --layout: the routed-expert banks of ggml block formats in ggml_frag_v1 where it measured
    // not slower (auto), as published, or in ggml_frag_v1 wherever a format has a map (frag).
    ExpertLayout expert_layout = ExpertLayout::Auto;
};

struct ConvertResult {
    OJson report;
    double seconds        = 0.0;
    std::uint64_t objects = 0;
    std::uint64_t files   = 0;
    // Wall time per phase: model+recipe, plan, weights, finish (fdatasync and publish).
    double model_seconds  = 0.0;
    double plan_seconds   = 0.0;
    double write_seconds  = 0.0;
    double finish_seconds = 0.0;
    // Source-consuming runs.
    bool resumed                     = false;
    std::uint64_t resumed_objects    = 0;
    std::uint64_t consume_peak_bound = 0; // planned peak bytes above the starting checkpoint
    std::uint64_t punched_bytes      = 0;
};

[[nodiscard]] ConvertResult convert(const ConvertOptions& options);

struct RecipeChoice {
    std::string recipe;
    // Named sources the recipe reads (--source NAME=PATH), e.g. quantized=<the checkpoint>.
    std::vector<std::pair<std::string, std::filesystem::path>> sources;
    // Why the obvious recipe was passed over (empty when it was not).
    std::string note;
};

// The official recipe for a checkpoint (a Hugging Face directory, a GGUF file or a directory of
// one GGUF model), by its declared architecture, MoE-ness and, for Qwen3.5, whether its
// projections are NVFP4 (the _nvfp4 recipes, with the checkpoint as their quantized source).
// A checkpoint without a published quantization gets its family's BF16 recipe (dense_bf16,
// moe_published, qwen3_5_bf16); a quantized one the recipe that keeps its encoding. What a
// recipe cannot keep natively is refused by convert(), never requantized silently.
// A GGUF quantizes tensor by tensor, so an NVFP4 recipe is chosen for one only when it plans:
// a file mixing NVFP4 with other types takes the plain recipe (note says why).
// A directory is judged by its config.json alone, so a consumed checkpoint still answers. An
// architecture without a recipe throws naming the supported ones.
// `mtp`: the conversion keeps the checkpoint's MTP layers (converts_mtp), so a recipe is planned
// with them; without, a recipe the text alone fits is chosen whatever the MTP layers hold.
[[nodiscard]] RecipeChoice choose_recipe(const std::filesystem::path& checkpoint,
                                         GgufRequest request = GgufRequest::Model, bool mtp = true);

// Whether a conversion with `options` keeps a checkpoint's MTP layers: --components names mtp, or
// none were named (components_auto) and the checkpoint carries them.
[[nodiscard]] bool converts_mtp(const ConvertOptions& options);

// The architectures the native converter has adapters and official recipes for.
[[nodiscard]] const std::string& supported_architectures_text();

} // namespace ninferno::convert
