"""Closed registry of persistent NInferno tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class Nvfp4Format:
    """E2M1 weights with one E4M3FN scale word per K-axis group."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Fp8RowFormat:
    """E4M3FN weights with one BF16 multiplier per logical row."""

    name: str


@dataclass(frozen=True, slots=True)
class Fp8RowF32Format:
    """E4M3FN weights with one binary32 multiplier per logical row."""

    name: str


@dataclass(frozen=True, slots=True)
class Fp8BlockFormat:
    """E4M3FN weights with one binary32 multiplier per block_rows x block_k block."""

    name: str
    block_rows: int
    block_k: int


@dataclass(frozen=True, slots=True)
class U4ZpFormat:
    """Unsigned 4-bit codes with one zero point byte and one binary16 scale per group of K values:
    w = (code - zero) * scale."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class GgmlKFormat:
    """ggml blocks along K (ggml semantics), kept as published: the K-quant, i-quant and ternary
    (TQ1_0, TQ2_0) superblocks of 256 values, the 32-value blocks of Q4_0, Q4_1, Q5_0, Q5_1 and
    IQ4_NL, Q1_0 (128 values) and Q2_0 (64)."""

    name: str
    block_bytes: int
    values: int = 256


@dataclass(frozen=True, slots=True)
class Mxfp4Format:
    """OCP MXFP4: E2M1 weights with one E8M0 exponent per 32 K values, no global scale."""

    name: str
    group_size: int


NumericFormat: TypeAlias = (
    DirectFormat
    | QuantFormat
    | Nvfp4Format
    | Fp8RowFormat
    | Fp8RowF32Format
    | Fp8BlockFormat
    | Mxfp4Format
    | U4ZpFormat
    | GgmlKFormat
)


BF16 = DirectFormat("bf16", 2)
FP32 = DirectFormat("fp32", 4)
INT32 = DirectFormat("int32", 4)

Q4_G64_FP16 = QuantFormat("q4_g64_fp16", 4, 64, -8, 7)
Q5_G64_FP16 = QuantFormat("q5_g64_fp16", 5, 64, -16, 15)
Q6_G64_FP16 = QuantFormat("q6_g64_fp16", 6, 64, -32, 31)
Q8_G32_FP16 = QuantFormat("q8_g32_fp16", 8, 32, -127, 127)
NVFP4 = Nvfp4Format("nvfp4", 16)
FP8_E4M3FN_ROW_BF16 = Fp8RowFormat("fp8_e4m3fn_row_bf16")
MXFP4 = Mxfp4Format("mxfp4", 32)
FP8_E4M3FN_ROW_F32 = Fp8RowF32Format("fp8_e4m3fn_row_f32")
FP8_E4M3FN_B128_F32 = Fp8BlockFormat("fp8_e4m3fn_b128_f32", 128, 128)
U4_G128_ZP_FP16 = U4ZpFormat("u4_g128_zp_fp16", 128)
U4_G32_ZP_FP16 = U4ZpFormat("u4_g32_zp_fp16", 32)
U4_G64_ZP_FP16 = U4ZpFormat("u4_g64_zp_fp16", 64)
GGML_Q4_K = GgmlKFormat("ggml_q4_k", 144)
GGML_Q5_K = GgmlKFormat("ggml_q5_k", 176)
GGML_Q6_K = GgmlKFormat("ggml_q6_k", 210)
GGML_Q3_K = GgmlKFormat("ggml_q3_k", 110)
GGML_Q2_K = GgmlKFormat("ggml_q2_k", 84)
GGML_IQ4_XS = GgmlKFormat("ggml_iq4_xs", 136)
GGML_IQ3_XXS = GgmlKFormat("ggml_iq3_xxs", 98)
GGML_IQ3_S = GgmlKFormat("ggml_iq3_s", 110)
GGML_Q4_0 = GgmlKFormat("ggml_q4_0", 18, 32)
GGML_Q4_1 = GgmlKFormat("ggml_q4_1", 20, 32)
GGML_Q5_0 = GgmlKFormat("ggml_q5_0", 22, 32)
GGML_Q5_1 = GgmlKFormat("ggml_q5_1", 24, 32)
GGML_IQ4_NL = GgmlKFormat("ggml_iq4_nl", 18, 32)
GGML_IQ2_XXS = GgmlKFormat("ggml_iq2_xxs", 66)
GGML_IQ2_XS = GgmlKFormat("ggml_iq2_xs", 74)
GGML_IQ2_S = GgmlKFormat("ggml_iq2_s", 82)
GGML_IQ1_S = GgmlKFormat("ggml_iq1_s", 50)
GGML_IQ1_M = GgmlKFormat("ggml_iq1_m", 56)
GGML_TQ1_0 = GgmlKFormat("ggml_tq1_0", 54)
GGML_TQ2_0 = GgmlKFormat("ggml_tq2_0", 66)
GGML_Q1_0 = GgmlKFormat("ggml_q1_0", 18, 128)
GGML_Q2_0 = GgmlKFormat("ggml_q2_0", 18, 64)
GGML_BLOCK_FORMATS = (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q3_K, GGML_Q2_K, GGML_IQ4_XS,
                      GGML_IQ3_XXS, GGML_IQ3_S, GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1,
                      GGML_IQ4_NL, GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ1_S,
                      GGML_IQ1_M, GGML_TQ1_0, GGML_TQ2_0, GGML_Q1_0, GGML_Q2_0)


DIRECT_FORMATS = MappingProxyType({item.name: item for item in (BF16, FP32, INT32)})
QUANT_FORMATS = MappingProxyType(
    {item.name: item for item in (Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, Q8_G32_FP16)}
)
NVFP4_FORMATS = MappingProxyType({NVFP4.name: NVFP4})
FP8_ROW_FORMATS = MappingProxyType({FP8_E4M3FN_ROW_BF16.name: FP8_E4M3FN_ROW_BF16})
MXFP4_FORMATS = MappingProxyType({MXFP4.name: MXFP4})
FP8_F32_FORMATS = MappingProxyType(
    {item.name: item for item in (FP8_E4M3FN_ROW_F32, FP8_E4M3FN_B128_F32)}
)
NUMERIC_FORMATS = MappingProxyType(
    {
        **DIRECT_FORMATS,
        **QUANT_FORMATS,
        **NVFP4_FORMATS,
        **FP8_ROW_FORMATS,
        **MXFP4_FORMATS,
        **FP8_F32_FORMATS,
        U4_G128_ZP_FP16.name: U4_G128_ZP_FP16,
        U4_G32_ZP_FP16.name: U4_G32_ZP_FP16,
        U4_G64_ZP_FP16.name: U4_G64_ZP_FP16,
        GGML_Q4_K.name: GGML_Q4_K,
        GGML_Q5_K.name: GGML_Q5_K,
        GGML_Q6_K.name: GGML_Q6_K,
        GGML_Q3_K.name: GGML_Q3_K,
        GGML_Q2_K.name: GGML_Q2_K,
        GGML_IQ4_XS.name: GGML_IQ4_XS,
        GGML_IQ3_XXS.name: GGML_IQ3_XXS,
        GGML_IQ3_S.name: GGML_IQ3_S,
        GGML_Q4_0.name: GGML_Q4_0,
        GGML_Q4_1.name: GGML_Q4_1,
        GGML_Q5_0.name: GGML_Q5_0,
        GGML_Q5_1.name: GGML_Q5_1,
        GGML_IQ4_NL.name: GGML_IQ4_NL,
        GGML_IQ2_XXS.name: GGML_IQ2_XXS,
        GGML_IQ2_XS.name: GGML_IQ2_XS,
        GGML_IQ2_S.name: GGML_IQ2_S,
        GGML_IQ1_S.name: GGML_IQ1_S,
        GGML_IQ1_M.name: GGML_IQ1_M,
        GGML_TQ1_0.name: GGML_TQ1_0,
        GGML_TQ2_0.name: GGML_TQ2_0,
        GGML_Q1_0.name: GGML_Q1_0,
        GGML_Q2_0.name: GGML_Q2_0,
    }
)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1_word(word: int) -> float:
    """Decode one exact four-bit E2M1 word, including signed zero."""

    if type(word) is not int or not 0 <= word <= 0xF:
        raise ValueError("E2M1 word must be an integer in [0, 15]")
    magnitude = _E2M1_MAGNITUDES[word & 0x7]
    return math.copysign(magnitude, -1.0 if word & 0x8 else 1.0)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def valid_nvfp4_scale_word(word: int) -> bool:
    """Return whether *word* is an admitted nonnegative finite E4M3FN scale."""

    return type(word) is int and 0 <= word <= 0xFF and word & 0x80 == 0 and word != 0x7F


def valid_fp8_weight_word(word: int) -> bool:
    """Return whether *word* is a finite E4M3FN weight code."""

    return type(word) is int and 0 <= word <= 0xFF and (word & 0x7F) != 0x7F


def valid_fp8_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite BF16 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFF or word & 0x8000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return math.isfinite(value)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "FP32",
    "INT32",
    "Q4_G64_FP16",
    "Q5_G64_FP16",
    "Q6_G64_FP16",
    "Q8_G32_FP16",
    "NVFP4",
    "FP8_E4M3FN_ROW_BF16",
    "MXFP4",
    "MXFP4_FORMATS",
    "FP8_E4M3FN_ROW_F32",
    "FP8_E4M3FN_B128_F32",
    "FP8_F32_FORMATS",
    "U4_G128_ZP_FP16",
    "U4_G32_ZP_FP16",
    "U4_G64_ZP_FP16",
    "U4ZpFormat",
    "GGML_Q4_K",
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_Q3_K",
    "GGML_Q2_K",
    "GGML_IQ4_XS",
    "GGML_IQ3_XXS",
    "GGML_IQ3_S",
    "GGML_Q4_0",
    "GGML_Q4_1",
    "GGML_Q5_0",
    "GGML_Q5_1",
    "GGML_IQ4_NL",
    "GGML_IQ2_XXS",
    "GGML_IQ2_XS",
    "GGML_IQ2_S",
    "GGML_IQ1_S",
    "GGML_IQ1_M",
    "GGML_TQ1_0",
    "GGML_TQ2_0",
    "GGML_Q1_0",
    "GGML_Q2_0",
    "GGML_BLOCK_FORMATS",
    "GgmlKFormat",
    "Fp8RowF32Format",
    "Fp8BlockFormat",
    "Mxfp4Format",
    "DIRECT_FORMATS",
    "QUANT_FORMATS",
    "NVFP4_FORMATS",
    "FP8_ROW_FORMATS",
    "NUMERIC_FORMATS",
    "DirectFormat",
    "QuantFormat",
    "Nvfp4Format",
    "Fp8RowFormat",
    "NumericFormat",
    "get_format",
    "decode_e2m1_word",
    "decode_e4m3fn_word",
    "valid_fp8_row_scale_word",
    "valid_fp8_weight_word",
    "valid_nvfp4_scale_word",
    "valid_positive_fp32_word",
]
