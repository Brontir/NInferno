"""Pins the Gemma 4 config readers on the published checkpoints' own config.json.

The fixtures are the real configs (quantization_config dropped: no reader uses it) and the
normalized configs the converter wrote into the artifacts the engine was validated on. A
refactor of the readers must reproduce them exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.convert.gemma4 import text_config
from tools.convert.gemma4_assistant import assistant_config
from tools.convert.gemma4_dflash import dflash_config

FIXTURES = Path(__file__).parent / "fixtures" / "gemma4"
READERS = {
    "rh-26b-a4b": text_config,
    "rh-31b": text_config,
    "assistant-26b": assistant_config,
    "dflash-31b": dflash_config,
}


@pytest.mark.parametrize("name", sorted(READERS))
def test_config_readers_reproduce_the_validated_artifacts(name):
    source = json.loads((FIXTURES / f"{name}.source.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / f"{name}.expected.json").read_text(encoding="utf-8"))
    # JSON round trip: the artifact stores the config as JSON, so compare in that form.
    assert json.loads(json.dumps(READERS[name](source))) == expected


def test_text_config_rejects_a_changed_geometry():
    source = json.loads((FIXTURES / "rh-26b-a4b.source.json").read_text(encoding="utf-8"))
    config = source.get("text_config", source)
    config["num_attention_heads"] = 0
    with pytest.raises(ValueError):
        text_config(source)
