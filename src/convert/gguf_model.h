#pragma once

// A GGUF model read as the Hugging Face checkpoint llama.cpp converted it from: config.json
// rebuilt from the GGUF metadata, and every tensor the architecture adapters read exposed under
// its Hugging Face name, with llama.cpp's conversion undone on read.
//
// Supported general.architecture values and what their conversion changed (llama.cpp
// conversion/qwen.py, conversion/gemma.py, gguf-py tensor_mapping.py):
//
//   qwen35, qwen35moe (Qwen3.5 / 3.6 / 3.8 text):
//     RMSNorm weights stored as 1 + w (all but linear_attn.norm)   -> w = stored - 1
//     A_log stored as -exp(A_log)                                  -> A_log = log(-stored)
//     dt_bias renamed dt_proj.bias; conv1d squeezed to [C, K]      -> renamed, [C, 1, K]
//     GDN value heads reordered from grouped-by-key-head to tiled
//       (in_proj_qkv V rows, in_proj_z, in_proj_a/b, A_log, dt_bias,
//        conv1d V channels, out_proj columns)                      -> the inverse permutation
//     experts: gate_up_proj split into gate/up banks               -> rebuilt as one bank, and
//                                                                     every expert also as a 2-D
//                                                                     matrix per role
//     MTP: mtp.* moved to block N (nextn.*)                         -> mtp.* again
//   gemma4 (Gemma 4 text):
//     expert banks [E, rows, cols]                                 -> per-expert 2-D matrices
//     router.per_expert_scale moved to ffn_down_exps.scale (folded into the NVFP4 global scale
//       when the experts are NVFP4)                                -> the scale again, or ones
//                                                                     with the fold kept in the
//                                                                     down matrix's divisor
//     token table trimmed to the tokenizer's vocabulary            -> vocab_size follows it
//   nemotron_h_moe, when it carries only the MTP head (nextn_predict_layers set, no blk.0; the
//     folded NextN block at block_count - 1)                     -> mtp.layers.0 / mtp.layers.1
//       names (enorm / hnorm / eh_proj, the attention, the MoE,
//       final_layernorm); the re-shipped token_embd / output /
//       output_norm of the target are not mapped
//   clip with projector type qwen3vl_merger (a Qwen3.5 / 3.6 / 3.8 vision projector, mmproj-*.gguf):
//     Conv3D patch kernel split along time into two Conv2D kernels
//       (v.patch_embd.weight, .weight.1)                           -> one [h, 3, 2, p, p] kernel
//     model.visual.* renamed v.blk.* / v.post_ln / mm.0 / mm.2     -> model.visual.* again;
//                                                                     vision_config from clip.vision.*
//   llama, qwen2, qwen3 (dense decoders, LlamaForCausalLM / Qwen2ForCausalLM / Qwen3ForCausalLM):
//     llama: Q/K rows of each head interleaved for ggml's NORM rope
//       (LlamaModel.permute)                                       -> rows de-interleaved
//     llama 3.x: rope scaling written as the rope_freqs tensor     -> per-pair frequency factors
//     tied embeddings: no output.weight                            -> tie_word_embeddings
//
// NVFP4 GGUF tensors keep their codes and E4M3 block scales (a nibble permutation); the global
// scale (<name>.scale) becomes the divisor 1/scale and <name>.input_scale the activation divisor.
// MXFP4 GGUF tensors map onto NVFP4 exactly when their exponents fit one power-of-two divisor.

#include "convert/gguf.h"
#include "convert/pyjson.h"
#include "convert/safetensors.h"

#include <functional>
#include <string>
#include <vector>

namespace ninferno::convert {

struct GgufMapping {
    OJson config;
    std::vector<TensorInfo> views;  // Hugging Face-named, each with a view
    std::vector<std::string> notes; // one line per kind of rename or inverse transform
    // The mapping's version where it changed what a converted model holds ("" before any change):
    // recorded in the artifact's provenance, so an artifact of an earlier mapping is told apart.
    std::string mapper;
};

// Stored tensors by GGUF name (null when absent), and a pinned plan-time read of a stored
// tensor's bytes.
using StoredLookup = std::function<const TensorInfo*(const std::string&)>;
using StoredRead   = std::function<void(const TensorInfo&, std::uint64_t, std::uint64_t, void*)>;

[[nodiscard]] GgufMapping map_gguf_model(const GgufFile& file, const StoredLookup& stored,
                                         const StoredRead& read, const std::string& label,
                                         GgufRequest request = GgufRequest::Model);

// general.architecture values map_gguf_model reads.
[[nodiscard]] const std::vector<std::string>& supported_gguf_architectures();

} // namespace ninferno::convert
