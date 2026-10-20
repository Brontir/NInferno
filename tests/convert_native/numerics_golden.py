"""Golden encodings for test_numerics.cpp, produced by the Python codecs.

Run from the repository root with the converter's Python environment:

    PYTHONPATH=. python tests/convert_native/numerics_golden.py

and paste the printed table into test_numerics.cpp when a codec changes on purpose. Inputs come
from the same 64-bit LCG as the C++ test and are exact in binary32 on both sides.
"""

from __future__ import annotations

import struct

import numpy as np
import torch

from tools.artifact.codecs.nvfp4 import encode_nvfp4
from tools.artifact.codecs.row_split import encode_row_split
from tools.convert.quantization.groupwise import quantize_matrix


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        return self.state >> 40  # 24 bits

    def unit(self) -> float:  # [-0.5, 0.5), exact in binary32
        return self.next() * 2.0**-24 - 0.5


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & (2**64 - 1)
    return h


def grouped_values(rows: int, k: int, seed: int, qmax: int) -> np.ndarray:
    lcg = Lcg(seed)
    values = np.array(
        [[lcg.unit() * 8.0 for _ in range(k)] for _ in range(rows)], dtype=np.float32
    )
    values[0, :32] = 0.0  # an all-zero group
    values[1, :32] = values[1, :32] * np.float32(2.0**-110)  # scale underflows binary16
    ties = [float(qmax), 2.5, -3.5, 0.5, -0.5, 1.5, -1.5, 4.5]
    values[2, : len(ties)] = ties  # scale 1: codes are the ties themselves
    values[2, len(ties) : 32] = 0.25
    return values


def grouped_case(fmt: str, rows: int, k: int, seed: int, qmax: int) -> int:
    values = torch.from_numpy(grouped_values(rows, k, seed, qmax))
    encoded = quantize_matrix(values, fmt, device="cpu")
    return fnv1a(encode_row_split(encoded.codes, encoded.scales, fmt, (rows, k)))


def nvfp4_case(rows: int, k: int, seed: int) -> int:
    lcg = Lcg(seed)
    codes = torch.tensor(
        [[lcg.next() & 0xFF for _ in range(k // 2)] for _ in range(rows)], dtype=torch.uint8
    )
    scales = torch.tensor(
        [[lcg.next() % 0x7F for _ in range(k // 16)] for _ in range(rows)], dtype=torch.uint8
    )
    divisor = struct.pack("<f", 0.123)
    return fnv1a(encode_nvfp4(codes, scales, divisor, (rows, k)))


CASES = [
    ("q8_g32_fp16", 37, 200, 1, 127),
    ("q6_g64_fp16", 19, 200, 2, 31),
    ("q5_g64_fp16", 11, 256, 3, 15),
    ("q4_g64_fp16", 7, 130, 4, 7),
]

if __name__ == "__main__":
    for fmt, rows, k, seed, qmax in CASES:
        print(f'    {{"{fmt}", {rows}, {k}, {seed}, 0x{grouped_case(fmt, rows, k, seed, qmax):016x}ULL}},')
    print(f"    nvfp4 256x128 seed 5: 0x{nvfp4_case(256, 128, 5):016x}ULL")
