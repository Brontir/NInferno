"""The native GGUF reader and ggml decoders against gguf-py, bit for bit.

    python gguf_pyref.py NINFERNO_GGUF_DUMP

gguf-py (llama.cpp's own Python package) writes a GGUF file holding one tensor of random blocks
for every ggml type both sides decode, plus metadata of every value type. gguf-py's GGUFReader
and its numpy dequantizers are the reference: the native parse must report the same metadata and
tensor infos, and the native dequantization the same binary32 words. Blocks whose reference
values are not finite are redrawn, so every compared word is a number.

Exits 77 (skipped) without numpy or gguf.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import gguf
    from gguf import GGMLQuantizationType as Q
    from gguf.quants import dequantize
except ImportError:
    sys.exit(77)

TYPES = [
    Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0, Q.Q2_K, Q.Q3_K, Q.Q4_K,
    Q.Q5_K, Q.Q6_K, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S, Q.IQ3_XXS, Q.IQ3_S, Q.IQ1_S, Q.IQ1_M,
    Q.IQ4_NL, Q.IQ4_XS, Q.TQ1_0, Q.TQ2_0, Q.MXFP4, Q.NVFP4,
]
ROWS = 3


def random_blocks(rng, qtype, blocks):
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    if qtype in (Q.F32, Q.F16, Q.BF16):
        values = rng.standard_normal(blocks).astype(np.float32)
        if qtype == Q.F32:
            return values.view(np.uint8)
        if qtype == Q.F16:
            return values.astype(np.float16).view(np.uint8)
        return (values.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    raw = rng.integers(0, 256, (blocks, size), dtype=np.uint8)
    for b in range(blocks):
        for _ in range(1000):
            values = dequantize(raw[b:b + 1].reshape(-1), qtype)
            if np.all(np.isfinite(values)):
                break
            raw[b] = rng.integers(0, 256, size, dtype=np.uint8)
        else:
            raise RuntimeError(f"{qtype.name}: no finite block drawn")
    return raw.reshape(-1)


def main() -> int:
    dump = Path(sys.argv[1]).resolve()
    rng = np.random.default_rng(20260923)
    errors = []
    with tempfile.TemporaryDirectory(prefix="ninferno-gguf-pyref-") as tmp:
        work = Path(tmp)
        path = work / "all.gguf"
        writer = gguf.GGUFWriter(path, "qwen35")
        writer.add_uint32("test.u32", 7)
        writer.add_int32("test.i32", -7)
        writer.add_uint64("test.u64", 1 << 50)
        writer.add_int64("test.i64", -(1 << 50))
        writer.add_float32("test.f32", 0.1)
        writer.add_float64("test.f64", 0.1)
        writer.add_bool("test.bool", False)
        writer.add_uint8("test.u8", 200)
        writer.add_int8("test.i8", -100)
        writer.add_uint16("test.u16", 60000)
        writer.add_int16("test.i16", -30000)
        writer.add_string("test.string", "héllo")
        writer.add_array("test.strings", ["a", "", "ü"])
        writer.add_array("test.ints", [1, -2, 3])
        writer.add_array("test.floats", [0.5, 1e-6])
        reference = {}
        for qtype in TYPES:
            block, size = gguf.GGML_QUANT_SIZES[qtype]
            per_row = 2 * 256 if block > 1 else 512
            blocks = ROWS * per_row // block
            raw = random_blocks(rng, qtype, blocks)
            name = f"t.{qtype.name.lower()}"
            if block == 1:
                dtype = {Q.F32: np.float32, Q.F16: np.float16}.get(qtype)
                if dtype is not None:
                    writer.add_tensor(name, raw.view(dtype).reshape(ROWS, per_row))
                else:
                    writer.add_tensor(name, raw.reshape(ROWS, per_row * 2), raw_dtype=qtype)
            else:
                writer.add_tensor(name, raw.reshape(ROWS, per_row // block * size), raw_dtype=qtype)
            reference[name] = (qtype, dequantize(raw, qtype).astype(np.float32).reshape(-1))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        out = work / "dump"
        out.mkdir()
        result = subprocess.run([str(dump), str(path), str(out)], capture_output=True, text=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return 1
        native = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

        reader = gguf.GGUFReader(path)
        for field in reader.fields.values():
            if field.name.startswith("GGUF."):
                continue
            value = field.contents()
            got = native["metadata"].get(field.name)
            main_type = field.types[0]
            if main_type == gguf.GGUFValueType.FLOAT32:
                value = float(np.float32(value))
                ok = got is not None and np.float32(got) == np.float32(value)
            elif main_type == gguf.GGUFValueType.ARRAY and field.types[-1] == gguf.GGUFValueType.FLOAT32:
                ok = got is not None and [np.float32(v) for v in got] == [np.float32(v) for v in value]
            else:
                ok = got == value
            if not ok:
                errors.append(f"metadata {field.name}: native {got!r}, gguf-py {value!r}")
        by_name = {t["name"]: (i, t) for i, t in enumerate(native["tensors"])}
        for tensor in reader.tensors:
            index, info = by_name.get(tensor.name, (None, None))
            if info is None:
                errors.append(f"{tensor.name}: missing from the native parse")
                continue
            if info["type"] != int(tensor.tensor_type) or info["ne"] != [int(n) for n in tensor.shape]:
                errors.append(f"{tensor.name}: tensor info differs")
            if info["offset"] + native["data_offset"] != int(tensor.data_offset):
                errors.append(f"{tensor.name}: data offset differs")
        for name, (qtype, expected) in reference.items():
            index, info = by_name[name]
            got = np.fromfile(out / f"{index}.f32", dtype=np.float32)
            if got.shape != expected.shape:
                errors.append(f"{name}: {got.size} values, expected {expected.size}")
                continue
            same = got.view(np.uint32) == expected.view(np.uint32)
            zeros = (got == 0) & (expected == 0)
            differ = ~(same | zeros)
            signed_zero = int(np.count_nonzero(zeros & ~same))
            status = "ok" if not differ.any() else "FAILED"
            note = f" ({signed_zero} zeros differ in sign)" if signed_zero else ""
            print(f"{qtype.name}: {got.size} values {status}{note}", flush=True)
            if differ.any():
                first = int(np.flatnonzero(differ)[0])
                errors.append(f"{name}: {int(differ.sum())} values differ, first at {first}: "
                              f"native {got[first]!r}, gguf-py {expected[first]!r}")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
