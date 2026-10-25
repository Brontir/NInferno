#pragma once

// The facts the plan without the weights reads (runtime/engine/plan_dry_run.h), from what a model
// says about itself before it is loaded or downloaded, and from what a device or a named board is:
//
//   a GGUF header       general.architecture and its keys (block_count, context_length, attention
//                       head counts and widths, the sliding window and pattern, the recurrent
//                       ssm.* sizes, feed-forward and expert widths), and its tensor table: the
//                       weight bytes when the table is the whole model's (one file), else the
//                       listed file sizes (WeightBasis::FileSize, "estimated from file size");
//   a config.json       a Hugging Face checkpoint's (its text_config when it has one), with the
//                       weights' listed sizes;
//   an artifact         a .ninferno file's text component config (the converter's own spelling of
//                       the same keys) and its tensors' bytes;
//   a device            the CUDA device's total memory, compute capability, driver model (WDDM)
//                       and display, or a named board of core/data/gpu_facts.json (a what-if: its
//                       marketed VRAM through plan_board_total_bytes).
//
// The family (PlanModelFacts::model_type) is the model_type the converted artifact would declare
// (qwen3_5_text, qwen3_5_moe_text, gemma4_text, dense_decoder, moe_decoder, nemotron_h), from the
// GGUF architecture or the checkpoint's architectures[0]; "" when no family serves it.
//
// The state the family's planner lays out (PlanModelFacts::state, offline_bytes) is read from the
// same keys: the full layers' KV heads and widths (one geometry; a model whose full layers differ
// is approximate), the sliding layers' (a ring decoder keeps the full layers' geometry in its
// rings), the recurrent mixers' (GatedDeltaNet: ssm.time_step_rank value heads of
// ssm.inner_size / heads, ssm.group_count key heads of ssm.state_size; Mamba2: heads, head width,
// groups, state), the residual width and the vocabulary (the ids a request may carry are taken as
// the vocabulary: the planner sizes its token counts by the tokenizer's, at most the vocabulary).
// An artifact's DFlash component names its drafter (state.drafter); a separate drafter's config is
// plan_drafter_state_of_config's.
// Pure but for plan_device_of_cuda (the CUDA runtime) and plan_facts_of_artifact (the file).

#include "runtime/engine/plan_dry_run.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::convert {
struct GgufFile;
}

namespace ninferno::product {

// What the configuration choice and the drafter plan read of the target beside the plan facts:
// its layers, experts, weight formats and whether it carries MTP layers of its own.
struct PlanTargetShape {
    std::uint64_t hidden_size = 0, num_hidden_layers = 0, vocab_size = 0;
    int experts = 0, experts_per_token = 0;
    double active_params        = 0;
    double routed_expert_params = 0;
    bool builtin_mtp = false; // MTP / NextN layers in the model (GGUF nextn_predict_layers)
    bool tied_embeddings = false; // the output head shares the input table (no separate rows to read)
    // The quantized weight formats, as artifact::format_name spells them where known
    // ("ggml_q4_k", "mxfp4", "nvfp4", "bf16"); empty when unknown.
    std::vector<std::string> weight_formats;
    std::string base_model; // general.base_model.0.repo_url's org/name, or "" (a GGUF's lineage)
};

struct PlanSourceFacts {
    runtime::PlanModelFacts model;
    PlanTargetShape shape;
};

// The family a GGUF architecture converts to ("" when none): qwen35 -> qwen3_5_text, qwen35moe ->
// qwen3_5_moe_text, gemma4 -> gemma4_text, gpt-oss -> moe_decoder, nemotron_h_moe -> nemotron_h,
// the dense architectures (llama, qwen2, qwen3, ...) -> dense_decoder, or moe_decoder when the
// metadata names experts.
[[nodiscard]] std::string plan_model_type_of_gguf(const nlohmann::json& metadata);
// The same for a config.json (architectures[0], else model_type), or an artifact's text config
// (its own model_type).
[[nodiscard]] std::string plan_model_type_of_config(const nlohmann::json& config);

// From a GGUF header (convert::parse_gguf_head). `file_bytes`: every file of the model together
// (the split parts as listed, or the file's size); the weights are the tensor table's bytes when
// the header is the whole model's (no split.count above 1), else `file_bytes` (FileSize).
[[nodiscard]] PlanSourceFacts plan_facts_of_gguf(const convert::GgufFile& head,
                                                 std::uint64_t file_bytes);
// The same from the metadata alone (ninferno-ui keeps it as JSON): weights `file_bytes`, FileSize.
[[nodiscard]] PlanSourceFacts plan_facts_of_gguf_metadata(const nlohmann::json& metadata,
                                                          std::uint64_t file_bytes);
// From a config.json and the weights' bytes (WeightBasis as the caller knows them).
[[nodiscard]] PlanSourceFacts plan_facts_of_config(const nlohmann::json& config,
                                                   std::uint64_t weight_bytes,
                                                   runtime::WeightBasis basis);
// From a .ninferno artifact: its text config and its tensors (Artifact). Throws
// std::runtime_error when the file is no artifact.
[[nodiscard]] PlanSourceFacts plan_facts_of_artifact(const std::filesystem::path& artifact);

// A DFlash drafter's state from its config.json (a Qwen artifact's dflash / dflash2 component, or a
// separate drafter's: z-lab's DFlash and DFlash2 checkpoints, a converted gemma4_dflash / dflash
// artifact): its full and sliding layers, window, KV heads and width, and the target features one
// column holds (`target_hidden` x its taps; a drafter whose feature projection is split by tap
// holds the hidden width a prefill column, which its config does not say: the unsplit rows, the
// larger). nullopt: the config names no DFlash drafter.
[[nodiscard]] std::optional<models::DrafterStateShape>
plan_drafter_state_of_config(const nlohmann::json& config, std::uint32_t target_hidden);

// The same model at another quantization of `bytes` (a hub listing's row): the shape kept, the
// weights replaced (basis as given), the bits-per-weight of the rule from the parameter count.
[[nodiscard]] PlanSourceFacts with_weights(PlanSourceFacts facts, std::uint64_t bytes,
                                           runtime::WeightBasis basis);

// Whether this host shares its GPUs through WDDM (native Windows; WSL 2, /dev/dxg).
[[nodiscard]] bool host_paravirtual();

// A named board: a gpu_facts.json row by its name (case and the "NVIDIA "/"GeForce " prefixes
// ignored; "4090" finds "GeForce RTX 4090", the shortest name containing it wins) or PCI id
// ("0x2684"), then optional comma-separated facts: "headless" | "display", "linux" | "wddm". The
// defaults: this host's driver model, and a display under WDDM (the conservative side), none on
// Linux. nullopt with `error` set when nothing matches (the error lists close names).
[[nodiscard]] std::optional<runtime::PlanDeviceFacts> plan_device_of_board(std::string_view spec,
                                                                           std::string* error);

// The CUDA device `device`: its name, total memory, compute capability, WDDM (as the Engine reads
// it), display (the kernel watchdog) and integrated flag; with `measure_free` the free memory now
// (cudaMemGetInfo). Throws std::runtime_error when the device cannot be read.
// `stack_reset`: read the free memory with the CUDA stack limit at 0, as a start does when its
// plan has an automatic axis (runtime::plan_automatic_fit).
[[nodiscard]] runtime::PlanDeviceFacts plan_device_of_cuda(int device, bool measure_free,
                                                           bool stack_reset = true);

// The device from what NVML says (ninferno-ui): total bytes, name, compute capability; the driver
// model this host's; `display` as NVML reports it (nullopt: unknown, taken as driving one under
// WDDM); the PCI product id and the multiprocessors when the caller read them (0: unknown).
[[nodiscard]] runtime::PlanDeviceFacts
plan_device_of_stats(const std::string& name, std::uint64_t total_bytes, int cc_major, int cc_minor,
                     std::optional<bool> display, std::uint32_t pci_device_id = 0, int sms = 0);

} // namespace ninferno::product
