"""Interpret the current compressed-tensors FP8/NVFP4 fields and scale semantics.

The matrix resolver also accepts direct tensors. Resolution remains lazy so a
recipe can replace an unused checkpoint source before its weights are inspected.
"""

from __future__ import annotations

from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource, tensor_source


def _divisor_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    tensor = store.read_flat(name)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return raw


def _reciprocal_word(store: SafetensorsSource, name: str) -> bytes:
    """A divisor word from a multiplier scalar, rounded once in FP32.

    NVIDIA Model Optimizer stores NVFP4's per-matrix factor as a multiplier (`weight_scale_2`,
    `input_scale` = amax / (6 * 448)); compressed-tensors and the artifact store its reciprocal
    as a divisor. The conversion is one FP32 rounding, a relative change of at most 2^-24.
    """
    multiplier = struct.unpack("<f", _divisor_word(store, name))[0]
    word = struct.pack("<f", float(torch.tensor(1.0, dtype=torch.float32) / multiplier))
    if not valid_positive_fp32_word(struct.unpack("<I", word)[0]):
        raise ValueError(f"{name}: reciprocal divisor is not finite and positive")
    return word


def nvfp4_layout(store: SafetensorsSource, prefix: str) -> str | None:
    """Which publisher's NVFP4 field names a matrix uses, or None if it is not NVFP4.

    compressed-tensors (RedHat, Unsloth): `weight_packed`, `weight_scale`, `weight_global_scale`
    (a divisor), `input_global_scale`. Model Optimizer (NVIDIA): `weight` (packed U8),
    `weight_scale`, `weight_scale_2` (a multiplier), `input_scale`. Codes pack two E2M1 values
    per byte, low nibble first, in both; block scales are E4M3 over 16 columns in both.
    """
    if store.has(prefix + ".weight_packed"):
        return "compressed_tensors"
    if store.has(prefix + ".weight_scale_2") and store.describe(prefix + ".weight").dtype == "U8":
        return "modelopt"
    return None


def compressed_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Interpret the current compressed-tensors or Model Optimizer NVFP4, or per-row FP8."""
    if format not in ("nvfp4", "fp8_e4m3fn_row_bf16"):
        raise ValueError(f"unsupported encoded source format {format}")
    n, k = shape
    if format == "nvfp4" and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")
    layout = nvfp4_layout(store, prefix) if format == "nvfp4" else None
    if format == "nvfp4" and layout is None:
        raise ValueError(f"{prefix}: no NVFP4 representation in this checkpoint")
    packed_field = "weight_packed" if layout == "compressed_tensors" else "weight"

    def weight_divisor() -> bytes:
        if layout == "modelopt":
            return _reciprocal_word(store, f"{prefix}.weight_scale_2")
        return _divisor_word(store, f"{prefix}.weight_global_scale")

    def input_divisor() -> bytes:
        if layout == "modelopt":
            return _reciprocal_word(store, f"{prefix}.input_scale")
        return _divisor_word(store, f"{prefix}.input_global_scale")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        if format == "nvfp4":
            packed, scale = f"{prefix}.{packed_field}", f"{prefix}.weight_scale"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(
                end - begin, k // 2
            )
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((scales > 0x7E).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(format, codes, scales, weight_divisor())
        weight, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
        signature(weight, shape, "F8_E4M3")
        info = store.describe(scale)
        if info.dtype != "BF16" or prod(info.shape) != n:
            raise ValueError(f"{scale}: expected one BF16 scale per row")
        codes = (
            store.read_flat(weight, begin * k, end * k)
            .view(torch.uint8)
            .reshape(end - begin, k)
        )
        scales = store.read_flat(scale, begin, end)
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        if format == "fp8_e4m3fn_row_bf16":
            values = (
                words.codes.view(torch.float8_e4m3fn).float()
                * words.scales.float()[:, None]
            )
        else:
            codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
                last - first, k
            )
            values_table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            values = values_table[codes.long()]
            scales = (
                words.scales.view(torch.float8_e4m3fn)
                .float()
                .repeat_interleave(16, dim=1)
            )
            values = values * scales / struct.unpack("<f", words.weight_divisor)[0]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} ({format})",
        read,
        encoded,
        weight_divisor if format == "nvfp4" else None,
        input_divisor if format == "nvfp4" else None,
        ((store, prefix),),
    )


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve the selected matrix's encoding lazily, after recipe source overrides."""
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            actual = format
            if actual is None and nvfp4_layout(store, prefix) is not None:
                actual = "nvfp4"
            if actual is None and store.describe(name).dtype == "F8_E4M3":
                actual = "fp8_e4m3fn_row_bf16"
            resolved = (
                tensor_source(store, name, shape)
                if actual is None
                else compressed_matrix_source(store, prefix, shape, actual)
            )
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
        ((store, name),),
    )


_E4M3_MAX = 448.0


def _divisor_value(word: bytes) -> float:
    return struct.unpack("<f", word)[0]


def _refold_scales(scales: torch.Tensor, ratio: float) -> torch.Tensor:
    """Re-express E4M3 block scales against a different per-matrix divisor.

    A value is `code * block_scale / divisor`, so moving a matrix from divisor `d` to divisor
    `D` means `block_scale * D/d`, rounded back onto the E4M3 grid. The codes never move. The
    whole error of the fold lives in that one rounding, and it is bounded by half an E4M3 step.
    """
    if ratio == 1.0:
        return scales
    values = scales.view(torch.float8_e4m3fn).float() * ratio
    if bool((values > _E4M3_MAX).any()):
        raise ValueError(
            f"refolded NVFP4 block scale overflows E4M3 (divisor ratio {ratio!r}, "
            f"largest source scale {float(scales.view(torch.float8_e4m3fn).float().max())})"
        )
    folded = values.to(torch.float8_e4m3fn)
    lost = (folded.float() == 0) & (values != 0)
    if bool(lost.any()):
        raise ValueError("refolded NVFP4 block scale underflows E4M3 to zero")
    return folded.view(torch.uint8)


def divisor_scaled_source(source: LogicalSource, factor, label: str) -> LogicalSource:
    """The same encoded matrix read against its weight divisor times `factor()`.

    Codes and block scales are untouched; every value comes out divided by the factor, and the
    activation divisor of its input is divided by it too, because the input this matrix will be
    fed is expected to arrive multiplied by the same factor (see stacked_matrix_source's
    `adopt_first`). The factor is read lazily; both divisors round once to FP32.
    """
    if source.weight_divisor is None:
        raise ValueError(f"{label}: only an encoded matrix can take a scaled divisor")

    def scaled(word: bytes, multiply: bool) -> bytes:
        value = _divisor_value(word)
        f = float(factor())
        result = struct.pack("<f", value * f if multiply else value / f)
        if not valid_positive_fp32_word(struct.unpack("<I", result)[0]):
            raise ValueError(f"{label}: scaled divisor is not finite and positive")
        return result

    def encoded(begin: int, end: int) -> EncodedRows:
        words = source.read_encoded(begin, end)
        return EncodedRows(words.format, words.codes, words.scales,
                           scaled(words.weight_divisor, True))

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        k = source.shape[1]
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        pairs = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(last - first, k)
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        )
        block = words.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        values = table[pairs.long()] * block / _divisor_value(words.weight_divisor)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        source.shape,
        f"divisor×factor({source.label})",
        read,
        encoded,
        lambda: scaled(source.weight_divisor(), True),
        (lambda: scaled(source.input_divisor(), False)) if source.input_divisor else None,
        source.origin,
    )


def stacked_matrix_source(
    parts: list[LogicalSource], label: str, *, adopt_first: bool = False
) -> LogicalSource:
    """Stack whole NVFP4 matrices into one plane under a single per-matrix divisor.

    Gemma 4 publishes each expert's gate and up as separate 704-row matrices with separate
    divisors, and the Gemma 4 MoE kernels take one divisor per expert (its fused gate_up), so one
    of the two divisors has to give way. The survivor is the larger, because scaling block scales
    up toward E4M3's 448 ceiling is the direction that can be checked exactly — and is checked, in
    `_refold_scales`.

    Values are read through the same fold, so a dequantized read of this source and the encoded
    rows it hands the writer describe the same numbers.

    `adopt_first` replaces the fold with no rounding at all: every code and block scale is kept
    and the whole plane takes the first matrix's divisor, so the rows of a later matrix k come out
    multiplied by exactly divisor_k / divisor_0. That is a different matrix, and it is only right
    when the caller removes the factor downstream — Gemma 4's GeGLU does, because `up` enters the
    product linearly and `down` is linear (divisor_scaled_source on that expert's down).
    """
    if not parts:
        raise ValueError("a stacked source needs at least one matrix")
    if any(len(part.shape) != 2 or part.shape[1] != parts[0].shape[1] for part in parts):
        raise ValueError(f"{label}: stacked matrices must share their column count")
    if any(part.weight_divisor is None for part in parts):
        raise ValueError(f"{label}: stacked NVFP4 matrices must state their divisors")
    k = parts[0].shape[1]
    rows = [part.shape[0] for part in parts]
    shape = (sum(rows), k)

    def common_divisor() -> bytes:
        if adopt_first:
            return parts[0].weight_divisor()
        return max((part.weight_divisor() for part in parts), key=_divisor_value)

    def spans(begin: int, end: int):
        cursor = 0
        for part, count in zip(parts, rows):
            low, high = max(begin, cursor), min(end, cursor + count)
            if low < high:
                yield part, low - cursor, high - cursor
            cursor += count

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= shape[0]:
            raise ValueError(f"{label}: invalid encoded rows [{begin},{end})")
        divisor = common_divisor()
        target = _divisor_value(divisor)
        codes, scales = [], []
        for part, low, high in spans(begin, end):
            words = part.read_encoded(low, high)
            if words.format != "nvfp4":
                raise ValueError(f"{label}: stacked sources must all be NVFP4")
            codes.append(words.codes)
            scales.append(
                words.scales
                if adopt_first
                else _refold_scales(words.scales, target / _divisor_value(words.weight_divisor))
            )
        return EncodedRows(
            "nvfp4",
            torch.cat(codes) if len(codes) > 1 else codes[0],
            torch.cat(scales) if len(scales) > 1 else scales[0],
            divisor,
        )

    def read(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        pairs = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(last - first, k)
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        )
        values = table[pairs.long()]
        block = words.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        values = values * block / _divisor_value(words.weight_divisor)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"stack({label}, {len(parts)} matrices, {shape[0]} rows)",
        read,
        encoded,
        common_divisor,
        parts[0].input_divisor,
        tuple(pair for part in parts for pair in part.origin),
    )
