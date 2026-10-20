from __future__ import annotations

import pytest

from tools.convert.gemma4_dflash import dflash_config


def _source(**changes):
    config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "final_logit_softcapping": 30.0,
        "hidden_size": 2816,
        "intermediate_size": 5632,
        "num_hidden_layers": 5,
        "num_target_layers": 30,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
        "sliding_window": 2048,
        "rope_theta": 1000000,
        "rope_scaling": None,
        "rms_norm_eps": 1e-6,
        "block_size": 16,
        "vocab_size": 262144,
        "dflash_config": {"mask_token_id": 4, "target_layer_ids": [1, 6, 11, 17, 22, 27]},
    }
    config.update(changes)
    return config


def test_dflash_config_keeps_what_the_engine_reads():
    config = dflash_config(_source())
    assert config["model_type"] == "gemma4_dflash"
    assert config["dflash_config"] == {
        "target_layer_ids": [1, 6, 11, 17, 22, 27],
        "mask_token_id": 4,
    }
    assert config["block_size"] == 16 and config["num_target_layers"] == 30
    # The two published drafters disagree on this flag; it carries no meaning here.
    assert dflash_config(_source(tie_word_embeddings=True))["hidden_size"] == 2816


@pytest.mark.parametrize(
    "change, message",
    [
        ({"final_logit_softcapping": None}, "not a Gemma 4 drafter"),
        ({"input_embedding_scale": 1.0}, "DFlash2"),
        ({"dflash_config": {"mask_token_id": 4, "target_layer_ids": [1, 30]}}, "target_layer_ids"),
        ({"dflash_config": {"mask_token_id": 262144, "target_layer_ids": [1]}}, "mask token"),
        ({"num_key_value_heads": 5}, "divide"),
        ({"rope_scaling": {"type": "yarn"}}, "rope_scaling"),
    ],
)
def test_dflash_config_rejects_what_the_engine_would_misread(change, message):
    with pytest.raises(ValueError, match=message):
        dflash_config(_source(**change))
