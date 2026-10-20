"""Persistent tensor layouts shared by NInferno converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import operator
from types import MappingProxyType
from typing import Sequence


from .formats import (
    DirectFormat,
    Fp8BlockFormat,
    Fp8RowF32Format,
    Fp8RowFormat,
    Mxfp4Format,
    Nvfp4Format,
    NumericFormat,
    QuantFormat,
    U4ZpFormat,
    GgmlKFormat,
    get_format,
)

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    scale_rows: int  # align_up(n, 128): the scale plane holds whole 128-row tiles
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    weight_divisor_offset: int
    weight_divisor_count: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


CONTIGUOUS_LE_V1 = Layout("contiguous_le_v1", 256, frozenset(("bf16", "fp32", "int32")))
ROW_SPLIT_K128_V1 = Layout(
    "row_split_k128_v1",
    256,
    frozenset(("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")),
)
BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "block_scale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4",)),
)
ROW_SCALE_V1 = Layout(
    "row_scale_v1",
    256,
    frozenset(("fp8_e4m3fn_row_bf16", "fp8_e4m3fn_row_f32")),
)
BLOCK_SCALE_128X128_F32_V1 = Layout(
    "block_scale_128x128_f32_v1",
    256,
    frozenset(("fp8_e4m3fn_b128_f32",)),
)
BLOCK_SCALE_K32_E8M0_V1 = Layout(
    "block_scale_k32_e8m0_v1",
    256,
    frozenset(("mxfp4",)),
)
ROW_GROUP_ZP_V1 = Layout(
    "row_group_zp_v1",
    256,
    frozenset(("u4_g128_zp_fp16", "u4_g32_zp_fp16", "u4_g64_zp_fp16")),
)
# row_group_zp_v1 with the K values in a stored order: the same planes, then at a plane boundary K
# little-endian int32 input-column indices (order[j] is the column stored at position j).
ROW_GROUP_ZP_ORDERED_V1 = Layout(
    "row_group_zp_ordered_v1",
    256,
    frozenset(("u4_g128_zp_fp16", "u4_g32_zp_fp16", "u4_g64_zp_fp16")),
)
GGML_BLOCKS_V1 = Layout(
    "ggml_blocks_v1",
    256,
    frozenset(("ggml_q4_k", "ggml_q5_k")),
)
GGML_PLANES_V1 = Layout(
    "ggml_planes_v1",
    256,
    frozenset(("ggml_q6_k", "ggml_q3_k", "ggml_q2_k", "ggml_iq4_xs", "ggml_iq3_xxs", "ggml_iq3_s",
               "ggml_q4_0", "ggml_q4_1", "ggml_q5_0", "ggml_q5_1", "ggml_iq4_nl", "ggml_iq2_xxs",
               "ggml_iq2_xs", "ggml_iq2_s", "ggml_iq1_s", "ggml_iq1_m", "ggml_tq1_0", "ggml_tq2_0",
               "ggml_q1_0", "ggml_q2_0")),
)
# ggml_frag_v1: a routed-expert bank's published planes (ggml_blocks_v1 for ggml_q4_k,
# ggml_planes_v1 otherwise) with each row's code bytes in fragment order (core/ggml_frag.h,
# docs/maintainer/storage-layouts.md); the same geometry.
GGML_FRAG_V1 = Layout(
    "ggml_frag_v1",
    256,
    frozenset(("ggml_q4_0", "ggml_q4_k", "ggml_iq3_xxs", "ggml_iq2_s")),
)
# ggml_planes_v1: per block (code, high, scale) bytes (core/weight.h ggml_plane_bytes).
GGML_PLANE_BYTES = MappingProxyType({
    "ggml_q6_k": (192, 16, 2), "ggml_q3_k": (96, 12, 2), "ggml_q2_k": (64, 16, 4),
    "ggml_iq4_xs": (128, 4, 4), "ggml_iq3_xxs": (64, 32, 2), "ggml_iq3_s": (104, 4, 2),
    "ggml_q4_0": (16, 0, 2), "ggml_q4_1": (16, 0, 4), "ggml_q5_0": (16, 4, 2),
    "ggml_q5_1": (16, 4, 4), "ggml_iq4_nl": (16, 0, 2),
    "ggml_iq2_xxs": (64, 0, 2),
    "ggml_iq2_xs": (64, 8, 2),
    "ggml_iq2_s": (72, 8, 2),
    "ggml_iq1_s": (32, 16, 2),
    "ggml_iq1_m": (32, 16, 8),
    "ggml_tq1_0": (52, 0, 2),
    "ggml_tq2_0": (64, 0, 2),
    "ggml_q1_0": (16, 0, 2),
    "ggml_q2_0": (16, 0, 2),
})

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCK_SCALE_K16_M128X4_V1,
            ROW_SCALE_V1,
            BLOCK_SCALE_K32_E8M0_V1,
            BLOCK_SCALE_128X128_F32_V1,
            ROW_GROUP_ZP_V1,
            ROW_GROUP_ZP_ORDERED_V1,
            GGML_BLOCKS_V1,
            GGML_PLANES_V1,
            GGML_FRAG_V1,
        )
    }
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row_split_k128_v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    base_bytes_per_group = spec.group_size if spec.bits == 8 else spec.group_size // 2
    high_bytes_per_group = (
        0 if spec.bits in (4, 8) else spec.group_size * (spec.bits - 4) // 8
    )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int], divisors: int = 1
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
    n, k = _shape(shape, rank=2)
    if k % 64 != 0:
        raise ValueError("block_scale_k16_m128x4_v1 requires K divisible by 64")
    if divisors < 1 or n % divisors:
        raise ValueError("NVFP4 divisor count must divide the plane's rows")
    # Each divisor covers one source matrix, and a 128-row scale tile may not span two of them:
    # the swizzled scale plane is addressed in whole tiles and the engine takes the divisor from
    # the row.
    if divisors > 1 and (n // divisors) % 128:
        raise ValueError(
            "each NVFP4 divisor must cover a whole number of 128-row scale tiles"
        )
    # The scale plane is addressed in whole 128-row tiles: a last tile N does not fill is
    # completed, the scale words of its rows past N zero.
    scale_rows = align_up(n, 128)
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = scale_rows * k // spec.group_size
    weight_divisor_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        scale_rows=scale_rows,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=weight_divisor_offset,
        weight_divisor_count=divisors,
        payload_bytes=weight_divisor_offset + 4 * divisors,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat | Fp8RowF32Format, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, (Fp8RowFormat, Fp8RowF32Format)):
        raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * (4 if isinstance(spec, Fp8RowF32Format) else 2)
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def mxfp4_geometry(format: str | Mxfp4Format, shape: Sequence[int]) -> RowScaleGeometry:
    """Codes [N, K/2], then at a plane boundary the E8M0 exponents [N, K/32] row-major."""

    spec = _format(format)
    if not isinstance(spec, Mxfp4Format):
        raise ValueError("block_scale_k32_e8m0_v1 requires MXFP4")
    n, k = _shape(shape, rank=2)
    if k % spec.group_size:
        raise ValueError("block_scale_k32_e8m0_v1 requires K divisible by 32")
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * k // spec.group_size
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def fp8_block_geometry(format: str | Fp8BlockFormat, shape: Sequence[int]) -> RowScaleGeometry:
    """Codes [N, K], then at a plane boundary one binary32 multiplier per 128x128 block,
    [ceil(N/128), ceil(K/128)] row-major."""

    spec = _format(format)
    if not isinstance(spec, Fp8BlockFormat):
        raise ValueError("block_scale_128x128_f32_v1 requires block-scaled FP8")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = (
        4 * (-(-n // spec.block_rows)) * (-(-k // spec.block_k))
    )
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def u4_zp_geometry(format: str | U4ZpFormat, shape: Sequence[int]) -> RowSplitGeometry:
    """Codes [N, K/2] (element 2i in the low nibble), then at a plane boundary one zero point byte
    per group [N, K/group], then at a plane boundary the binary16 group scales [N, K/group]."""

    spec = _format(format)
    if not isinstance(spec, U4ZpFormat):
        raise ValueError("row_group_zp_v1 requires a u4 zero-point format")
    n, k = _shape(shape, rank=2)
    if k % spec.group_size:
        raise ValueError("row_group_zp_v1 requires K divisible by the group size")
    groups_per_row = k // spec.group_size
    base_bytes = n * k // 2
    high_bytes = n * groups_per_row
    scale_bytes = n * groups_per_row * 2
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k,
        groups_per_row=groups_per_row,
        base_bytes_per_group=spec.group_size // 2,
        high_bytes_per_group=1,
        base_row_bytes=k // 2,
        high_row_bytes=groups_per_row,
        scale_row_bytes=groups_per_row * 2,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def ggml_k_geometry(format: str | GgmlKFormat, shape: Sequence[int]) -> RowSplitGeometry:
    """ggml_q4_k / ggml_q5_k: row r is its K/256 superblocks unchanged (one plane).
    The ggml_planes_v1 formats: per row its blocks' code bytes, then at a plane boundary their
    high bytes, then at a plane boundary their binary16 d (and dmin, m or IQ4_XS scales_h),
    GGML_PLANE_BYTES per block of spec.values values."""

    spec = _format(format)
    if not isinstance(spec, GgmlKFormat):
        raise ValueError("ggml layouts require a ggml block format")
    n, k = _shape(shape, rank=2)
    if k % spec.values:
        raise ValueError(f"ggml layouts require K divisible by {spec.values}")
    blocks = k // spec.values
    if spec.name not in GGML_PLANE_BYTES:
        base_bytes = n * blocks * spec.block_bytes
        return RowSplitGeometry(
            n=n, k=k, k_pad=k, groups_per_row=blocks, base_bytes_per_group=spec.block_bytes,
            high_bytes_per_group=0, base_row_bytes=blocks * spec.block_bytes, high_row_bytes=0,
            scale_row_bytes=0, base_offset=0, base_bytes=base_bytes, high_offset=base_bytes,
            high_bytes=0, scale_offset=base_bytes, scale_bytes=0, payload_bytes=base_bytes,
        )
    code, high, scale = GGML_PLANE_BYTES[spec.name]
    base_bytes = n * blocks * code
    high_bytes = n * blocks * high
    scale_bytes = n * blocks * scale
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n, k=k, k_pad=k, groups_per_row=blocks, base_bytes_per_group=code,
        high_bytes_per_group=high, base_row_bytes=blocks * code, high_row_bytes=blocks * high,
        scale_row_bytes=blocks * scale, base_offset=0, base_bytes=base_bytes, high_offset=high_offset,
        high_bytes=high_bytes, scale_offset=scale_offset, scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
    divisors: int = 1,
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if divisors != 1 and layout_spec is not BLOCK_SCALE_K16_M128X4_V1:
        raise ValueError(
            f"layout {layout_spec.name!r} stores one divisor, not {divisors}"
        )
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous_le_v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous_le_v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row_split_k128_v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_K16_M128X4_V1:
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
        return block_scale_geometry(numeric_spec, shape, divisors).payload_bytes
    if layout_spec is ROW_SCALE_V1:
        if not isinstance(numeric_spec, (Fp8RowFormat, Fp8RowF32Format)):
            raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_128X128_F32_V1:
        return fp8_block_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_K32_E8M0_V1:
        return mxfp4_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_GROUP_ZP_V1:
        return u4_zp_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_GROUP_ZP_ORDERED_V1:
        geometry = u4_zp_geometry(numeric_spec, shape)
        return align_up(geometry.payload_bytes, PLANE_ALIGNMENT) + 4 * geometry.k
    if layout_spec in (GGML_BLOCKS_V1, GGML_PLANES_V1, GGML_FRAG_V1):
        return ggml_k_geometry(numeric_spec, shape).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")
