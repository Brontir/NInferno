"""Checks every architecture adapter applies to a checkpoint's config.json fields.

Each one returns the validated value or raises ValueError naming the field; the strictest of the
per-family variants they replace: a bool is never a number, and a fixed bool field must be a bool.
"""

from __future__ import annotations

from math import isfinite
import struct

from .sources.safetensors import SafetensorsSource

# config.json keys a weights-only source may carry without describing a model.
_BOOKKEEPING_KEYS = {
    "quantization_config",
    "_name_or_path",
    "transformers_version",
    "torch_dtype",
    "dtype",
}


def positive(value, name):
    """A positive integer (a bool is not one)."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def f32(value, name):
    """A positive finite value representable as FP32 (a bool is not one)."""
    if type(value) not in (float, int) or isinstance(value, bool):
        raise ValueError(f"{name}: expected a real value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name}: not representable as F32") from error
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name}: expected positive finite F32")
    return result


def fixed(config, key, value, label):
    """`key`, when present, must equal `value` (with a bool `value`, it must be a bool)."""
    if key in config and (
        config[key] != value or isinstance(value, bool) and type(config[key]) is not bool
    ):
        raise ValueError(f"{label}.{key}: expected {value!r}, got {config[key]!r}")


def has_model_config(source: SafetensorsSource) -> bool:
    """Whether a source's config.json describes a model rather than only its bookkeeping."""
    return bool(source.config.keys() - _BOOKKEEPING_KEYS)


def check_source_fields(name, actual, expected, fields):
    """An additional source must agree with the model on every field its site depends on."""
    for field in sorted(fields):
        if actual.get(field) != expected.get(field):
            raise ValueError(
                f"{name}: additional source {field} differs; provide an explicit logical source"
            )
