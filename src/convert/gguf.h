#pragma once

// GGUF container parsing (versions 2 and 3, little-endian): the header, the metadata key/value
// pairs, the tensor infos and the aligned data section. Parsing reads a byte range the caller
// owns (a memory mapping); nothing is copied but the metadata.
//
// Metadata values become JSON: integers stay integers, BOOL a boolean, STRING a string, ARRAY a
// JSON array. An F32 value becomes the double with the shortest decimal that rounds back to the
// same binary32 (1e-06f reads as 1e-06, as a Hugging Face config spells it); F64 is exact.

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::convert {

enum class GgufValueType : std::uint32_t {
    U8     = 0,
    I8     = 1,
    U16    = 2,
    I16    = 3,
    U32    = 4,
    I32    = 5,
    F32    = 6,
    BOOL   = 7,
    STRING = 8,
    ARRAY  = 9,
    U64    = 10,
    I64    = 11,
    F64    = 12,
};

struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> ne; // ggml order: ne[0] is the contiguous dimension
    std::uint32_t type  = 0;       // ggml type id
    std::uint64_t offset = 0;      // from the start of the data section
    std::uint64_t bytes  = 0;

    // Row-major shape (ne reversed), as safetensors and PyTorch spell it.
    [[nodiscard]] std::vector<std::uint64_t> shape() const;
    [[nodiscard]] std::uint64_t elements() const noexcept;
};

struct GgufFile {
    std::uint32_t version   = 0;
    std::uint64_t alignment = 32;
    std::uint64_t data_offset = 0; // start of the tensor data section in the file
    nlohmann::json metadata = nlohmann::json::object();
    // Value type of each key (for an ARRAY, the element type) and whether it is an array.
    std::map<std::string, std::pair<GgufValueType, bool>> types;
    std::vector<GgufTensor> tensors;

    [[nodiscard]] bool has(const std::string& key) const { return metadata.contains(key); }

    [[nodiscard]] const nlohmann::json* find(const std::string& key) const {
        const auto it = metadata.find(key);
        return it == metadata.end() ? nullptr : &*it;
    }
};

// Parse a whole GGUF file held in memory. `label` names it in errors. Checks every tensor's type,
// byte size, alignment and extent against the file.
[[nodiscard]] GgufFile parse_gguf(const std::byte* data, std::size_t size, const std::string& label);

// The same from the first `size` bytes of a file of `file_size` (its metadata before its tensors
// are downloaded): tensor extents are checked against `file_size`. Throws GgufTruncated when the
// header, metadata or tensor infos run past `size` (a longer head reads them).
[[nodiscard]] GgufFile parse_gguf_head(const std::byte* data, std::size_t size,
                                       std::uint64_t file_size, const std::string& label);

// The Hugging Face repository (org/name) a GGUF's metadata names as the model's source: the first
// of general.base_model.0.repo_url, general.source.repo_url, general.source.huggingface.repository
// and general.source.url
// that is a Hugging Face address or an org/name ("https://huggingface.co/Qwen/Qwen3.5-9B/tree/main"
// -> "Qwen/Qwen3.5-9B"); nullopt when none is. The model's lineage, one reading for the plan and the
// converter's drafter lookup alike.
[[nodiscard]] std::optional<std::string> gguf_metadata_base_repository(const nlohmann::json& metadata);

struct GgufTruncated : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A GGUF file beside a model that is not the model: a vision projector (mmproj-*), an importance
// matrix (imatrix*.gguf, *-imatrix.gguf: not a model named "iMatrix" in the middle, cdiamond's
// Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf), an MTP-only sidecar (a4lg's <model>-MTP-ONLY-<quant>.gguf,
// mtp-*.gguf: the NextN block of a model whose GGUF dropped it, read as a target's mtp source), a
// drafter published beside the model (dflash-*.gguf, eagle*.gguf: meta-models'
// dflash-Muse-Glimmer-30B-Q4_K_M.gguf next to the Q4_K_M model). By file name, case ignored.
[[nodiscard]] bool gguf_auxiliary_file(const std::string& file_name);

// The binary32 value `value` as the double of its shortest round-trip decimal.
[[nodiscard]] double f32_shortest(float value);

} // namespace ninferno::convert
