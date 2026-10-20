"""An artifact's JSON directory as the synthetic converter tests compare it, read with the
container's own framing (tools/artifact/framing.py)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.framing import HEADER, PAYLOAD_ALIGNMENT  # noqa: E402


def directory(path: Path):
    """(the directory, its objects by id, the file's bytes, the payload's offset in them)."""
    data = path.read_bytes()
    _, length, _ = HEADER.unpack_from(data)
    value = json.loads(data[HEADER.size:HEADER.size + length].decode("utf-8"))
    offset = -(-(HEADER.size + length) // PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT
    return value, {o["id"]: o for o in value["objects"]}, data, offset
