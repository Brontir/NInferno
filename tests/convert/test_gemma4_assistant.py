from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.convert.gemma4_assistant import assistant_config, restrict_head
from tools.convert.model import Model, Parameter
from tools.convert.sources.logical import array_source


def _source_config(**changes):
    text = {
        "model_type": "gemma4_text",
        "hidden_activation": "gelu_pytorch_tanh",
        "enable_moe_block": False,
        "hidden_size_per_layer_input": 0,
        "num_hidden_layers": 2,
        "num_kv_shared_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "num_attention_heads": 8,
        "head_dim": 256,
        "num_key_value_heads": 4,
        "global_head_dim": 512,
        "num_global_key_value_heads": 2,
        "hidden_size": 1024,
        "vocab_size": 16,
        "max_position_embeddings": 4096,
        "intermediate_size": 4096,
        "sliding_window": 1024,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    }
    text.update(changes)
    return {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "backbone_hidden_size": 2816,
        "text_config": text,
    }


def test_assistant_config_reads_geometry_and_backbone():
    config = assistant_config(_source_config())
    assert config["model_type"] == "gemma4_assistant"
    assert config["backbone_hidden_size"] == 2816
    assert config["attention_geometry"]["full_attention"] == {
        "head_dim": 512,
        "num_key_value_heads": 2,
    }


@pytest.mark.parametrize(
    "change, message",
    [
        ({"num_kv_shared_layers": 1}, "key/value cache"),
        ({"enable_moe_block": True}, "enable_moe_block"),
        ({"layer_types": ["sliding_attention"]}, "layer_types"),
    ],
)
def test_assistant_config_rejects_other_drafters(change, message):
    with pytest.raises(ValueError, match=message):
        assistant_config(_source_config(**change))


def test_restrict_head_keeps_ranked_rows_and_their_ids(tmp_path):
    head = torch.arange(8 * 4).reshape(8, 4).to(torch.bfloat16)
    model = Model(
        {"text": {"config": {"vocab_size": 8}}}, token_count=6, special_token_ids=(5,)
    )
    model.add(
        Parameter(
            "text/output_head", (8, 4), array_source(head, "head"),
            inputs=("text/final_hidden",),
        )
    )
    ranking = tmp_path / "ranking.i64"
    # Rows 6 and 7 are outside the token domain; the special token 5 is always kept.
    np.array([9, 5, 9, 5, 0, 0, 999, 999], dtype="<i8").tofile(ranking)
    restrict_head(model, ranking=ranking, rows=3)
    ids = model.parameters["text/head_token_ids"].source.values()
    kept = model.parameters["text/output_head"]
    assert ids.dtype == torch.int32 and ids.tolist() == [0, 2, 5]
    assert kept.shape == (3, 4) and kept.inputs == ("text/final_hidden",)
    assert torch.equal(kept.source.values().reshape(3, 4), head[[0, 2, 5]])
    assert model.parameters["text/head_token_ids"].direct_format == "int32"
    assert model.config["head_rows"] == 3


def test_restrict_head_rejects_a_partial_ranking(tmp_path):
    model = Model({"text": {"config": {"vocab_size": 8}}}, token_count=8)
    model.add(Parameter("text/output_head", (8, 4), array_source(torch.zeros(8, 4), "head")))
    ranking = tmp_path / "ranking.i64"
    np.zeros(5, dtype="<i8").tofile(ranking)
    with pytest.raises(ValueError, match="complete int64"):
        restrict_head(model, ranking=ranking, rows=3)
