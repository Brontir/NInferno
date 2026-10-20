"""Pins the logical layout the DFlash builders produce, for both families.

A manifest — every logical parameter's name, shape, uses, direct format and residency, plus the
packing groups and the component configs — fully determines an artifact for a given recipe. The
expected manifests were taken from the builders the engine was validated with; a refactor of the
builders must reproduce them exactly.

Regenerate a fixture only for an intended layout change:
  python -m tests.convert.test_draft_layouts --write
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

from tests.convert.test_qwen3_5 import _checkpoint, _config
from tools.convert import gemma4_dflash, qwen3_5

FIXTURES = Path(__file__).parent / "fixtures" / "draft_layouts"


def _manifest(model) -> dict:
    return {
        "parameters": [
            [p.name, list(p.shape), list(p.inputs), p.direct_format, p.residency]
            for p in sorted(model.parameters.values(), key=lambda p: p.name)
        ],
        "packing_groups": [list(group) for group in model.packing_groups],
        "configs": {name: record.get("config") for name, record in sorted(model.components.items())},
    }


def _qwen_draft(architecture: str) -> dict:
    return {
        "architectures": [architecture],
        "hidden_size": 16,
        "vocab_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 24,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 32,
        "dflash_config": {
            "target_layer_ids": [1, 0],
            "mask_token_id": 7,
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 2,
        },
    }


GEMMA_DFLASH = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "final_logit_softcapping": 30.0,
    "hidden_size": 16,
    "intermediate_size": 24,
    "num_hidden_layers": 2,
    "num_target_layers": 3,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 32,
    "rope_theta": 1000000,
    "rope_scaling": None,
    "rms_norm_eps": 1e-6,
    "block_size": 4,
    "vocab_size": 8,
    "dflash_config": {"mask_token_id": 1, "target_layer_ids": [0, 2]},
}


def _build(case: str, tmp_path: Path):
    if case in ("qwen_dflash", "qwen_dflash2"):
        backend = "dflash2" if case == "qwen_dflash2" else "dflash"
        architecture = "DFlash2DraftModel" if backend == "dflash2" else "DFlashDraftModel"
        base = _checkpoint(tmp_path / "text", _config(), {"unused": torch.ones(1)})
        draft = _checkpoint(tmp_path / "draft", _qwen_draft(architecture), {"unused": torch.ones(1)})
        return qwen3_5.build_model(base, components=("text", backend), companions={backend: draft})
    base = _checkpoint(tmp_path / "gemma-dflash", GEMMA_DFLASH, {"unused": torch.ones(1)})
    return gemma4_dflash.build_model(base)


CASES = ("qwen_dflash", "qwen_dflash2", "gemma_dflash")


@pytest.mark.parametrize("case", CASES)
def test_draft_layout_is_unchanged(case, tmp_path):
    expected = json.loads((FIXTURES / f"{case}.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(_manifest(_build(case, tmp_path)))) == expected


if __name__ == "__main__" and "--write" in sys.argv:
    import tempfile

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(_build(case, Path(tmp)))
        (FIXTURES / f"{case}.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        print(case, len(manifest["parameters"]), "parameters")
