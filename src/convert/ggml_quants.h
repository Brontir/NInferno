#pragma once

// ggml tensor types as GGUF files store them: block geometry, and dequantization to binary32
// bit-exact with ggml's reference dequantize_row_* (ggml/src/ggml-quants.c; the converter is
// built without FMA contraction, so every product and sum rounds as ggml's C does).
//
// NVFP4 (ggml type 40) and MXFP4 (39) blocks also convert to the engine's NVFP4 words without
// touching a value: ggml keeps E2M1 codes in both, only the nibble order and scale packing differ.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ninferno::convert::ggml {

enum Type : std::uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39,
    NVFP4   = 40,
    Q1_0    = 41,
    Q2_0    = 42,
};

struct Traits {
    const char* name;     // ggml_type_name spelling, e.g. "q4_K"
    std::uint32_t block;  // elements per block
    std::uint32_t bytes;  // bytes per block
    bool dequantizable;   // the converter decodes it to binary32
};

// Traits of a GGUF type id, or nullopt for an id ggml does not define.
[[nodiscard]] std::optional<Traits> traits(std::uint32_t type) noexcept;

// Lowercase type name ("q4_k"), or "type<id>" for an unknown id.
[[nodiscard]] std::string type_name(std::uint32_t type);

// Whether a type holds plain values (F32, F16, BF16, F64 and the integer types).
[[nodiscard]] bool is_plain(std::uint32_t type) noexcept;

// Bytes of `elements` consecutive values; throws unless they fill whole blocks.
[[nodiscard]] std::uint64_t bytes_for(std::uint32_t type, std::uint64_t elements);

// Dequantize `elements` values (whole blocks) starting at a block boundary. Throws for a type
// that is not dequantizable.
void dequantize(std::uint32_t type, const std::byte* data, std::uint64_t elements, float* out);

// One NVFP4 block (4 UE4M3 scales, then 32 bytes where byte j of each 16-element sub-block holds
// elements j and j+8) as the engine's natural words: 32 code bytes with element 2i in the low
// nibble, and 4 E4M3 scale words (ggml reads 0x7f as zero; it becomes the zero word).
void nvfp4_block_natural(const std::uint8_t* block, std::uint8_t* codes, std::uint8_t* scales);

// One MXFP4 block (an E8M0 exponent, then 16 bytes where byte j holds elements j and j+16) as
// natural codes (16 bytes); returns the exponent.
std::uint8_t mxfp4_block_natural(const std::uint8_t* block, std::uint8_t* codes);

} // namespace ninferno::convert::ggml
