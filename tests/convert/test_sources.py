from __future__ import annotations

import struct

import torch
from safetensors.torch import save_file

from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.sources.compressed_tensors import (
    compressed_matrix_source,
    matrix_source,
)
from tools.convert.sources.logical import select_rows


def test_nvfp4_source_preserves_words_and_decodes_independently(tmp_path):
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight_packed": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_global_scale": torch.tensor([2.0], dtype=torch.float32),
            "proj.input_global_scale": torch.tensor([1.5], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        expected = torch.tensor(
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
        expected = torch.stack((expected / 2, expected))
        assert torch.equal(source.values().reshape(2, 16), expected)
        assert source.input_divisor() == struct.pack("<f", 1.5)
        assert source.values(16, 16).numel() == 0


def test_modelopt_nvfp4_reads_as_the_compressed_tensors_divisor(tmp_path):
    """NVIDIA Model Optimizer names the fields differently and stores the per-matrix factors as
    multipliers. The same codes and block scales, with weight_scale_2 = 1/2 and input_scale =
    1/1.5, must read back as the compressed-tensors matrix above: identical words, identical
    values, divisors 2.0 and 1.5."""
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_scale_2": torch.tensor([0.5], dtype=torch.float32),
            "proj.input_scale": torch.tensor([1.0 / 1.5], dtype=torch.float32),
            "dense.weight": torch.ones(2, 16, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        assert source.input_divisor() == struct.pack("<f", 1.5)
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        )
        assert torch.equal(source.values().reshape(2, 16), torch.stack((table / 2, table)))
        # A BF16 matrix beside it stays a plain tensor: a U8 `weight` alone is not NVFP4.
        plain = matrix_source(store, "dense.weight", (2, 16))
        assert torch.equal(plain.values(), torch.ones(32))


def test_adopted_divisor_is_undone_by_the_scaled_down_projection(tmp_path):
    """A GeGLU expert whose gate and up carry different divisors, stacked without a fold.

    The stacked plane keeps every code and scale and takes gate's divisor, so its up rows are
    wrong by d_up/d_gate on their own; down read against its divisor times that ratio must give
    back exactly the expert the separate matrices describe. The control arm (plain down) must
    miss by the ratio."""
    from tools.convert.sources.compressed_tensors import (
        divisor_scaled_source,
        stacked_matrix_source,
    )

    generator = torch.Generator().manual_seed(7)

    def encoded_matrix(prefix, rows, cols, divisor, tensors):
        tensors[f"{prefix}.weight_packed"] = torch.randint(
            0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator
        )
        tensors[f"{prefix}.weight_scale"] = (
            torch.randint(0x30, 0x48, (rows, cols // 16), dtype=torch.uint8, generator=generator)
            .view(torch.float8_e4m3fn)
        )
        tensors[f"{prefix}.weight_global_scale"] = torch.tensor([divisor], dtype=torch.float32)
        tensors[f"{prefix}.input_global_scale"] = torch.tensor([3.0], dtype=torch.float32)

    tensors = {}
    encoded_matrix("gate", 16, 32, 2.0, tensors)
    encoded_matrix("up", 16, 32, 2.925, tensors)
    encoded_matrix("down", 32, 16, 5.0, tensors)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with SafetensorsSource(tmp_path) as store:
        gate = matrix_source(store, "gate.weight", (16, 32), "nvfp4")
        up = matrix_source(store, "up.weight", (16, 32), "nvfp4")
        down = matrix_source(store, "down.weight", (32, 16), "nvfp4")
        stacked = stacked_matrix_source([gate, up], "gate_up", adopt_first=True)
        ratio = 2.925 / 2.0
        scaled = divisor_scaled_source(down, lambda: ratio, "down")

        def geglu(g, u, d, x):
            z = g @ x
            h = 0.5 * z * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (z + 0.044715 * z**3)))
            return d @ (h * (u @ x))

        x = torch.randn(32, 3, dtype=torch.float64, generator=generator)
        g, u, d = (s.values().reshape(s.shape).double() for s in (gate, up, down))
        gu = stacked.values().reshape(32, 32).double()
        truth = geglu(g, u, d, x)
        built = geglu(gu[:16], gu[16:], scaled.values().reshape(32, 16).double(), x)
        control = geglu(gu[:16], gu[16:], d, x)
        assert torch.equal(gu[:16], g)
        assert torch.allclose(built, truth, rtol=1e-6, atol=0)
        assert not torch.allclose(control, truth, rtol=1e-2, atol=0)
        # The encoded rows the writer receives describe the same numbers as the value read.
        words = scaled.read_encoded(0, 32)
        assert struct.unpack("<f", words.weight_divisor)[0] == struct.unpack("<f", struct.pack("<f", 5.0 * ratio))[0]
        assert torch.equal(words.codes, down.read_encoded(0, 32).codes)
        assert struct.unpack("<f", scaled.input_divisor())[0] == struct.unpack("<f", struct.pack("<f", 3.0 / ratio))[0]


def test_row_fp8_source_and_reordered_encoded_rows(tmp_path):
    codes = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)
    scales = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    save_file(
        {
            "proj.weight": codes.view(torch.float8_e4m3fn),
            "proj.weight_scale": scales,
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), "fp8_e4m3fn_row_bf16")
        assert torch.equal(
            source.values().reshape(2, 3),
            torch.tensor([[2.0, -2.0, 4.0], [0.25, -0.25, -0.0]]),
        )
        reordered = select_rows(source, ((1, 2), (0, 1)))
        words = reordered.read_encoded(0, 2)
        assert torch.equal(words.codes, codes.flip(0))
        assert torch.equal(words.scales, scales.flatten().flip(0))
