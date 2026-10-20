"""A converted artifact uploaded into a poisoned weight arena.

    python poisoned_upload.py NINFERNO_CONVERT MATERIALIZATION_TEST [REPOSITORY]

Converts the synthetic Gemma 4 MoE checkpoints of tests/convert_native (compressed-tensors and
Model Optimizer NVFP4: packed q/k/v, stacked gate_up planes with one divisor per expert,
compensated downs, Q8 and BF16 objects) and materializes each with NINFERNO_DEBUG_POISON=1: the
device arena is filled with 0xFF (NaN in every float format) before the upload. The test binary
then requires every bound object's device bytes to equal the file's, every alignment gap between
objects to still hold the poison (so the poison was there to be overwritten), and every logical
view to be covered. Before that, every NVFP4 object's scale words and divisors are decoded on the
host: none may be NaN or non-positive (the converter wrote them).

Exits 77 without torch (the synthetic checkpoints need it) or without a CUDA device.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import torch  # noqa: F401
except ImportError:
    sys.exit(77)


def main() -> int:
    native, test = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    repository = Path(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).resolve().parents[2])
    sys.path.insert(0, str(repository / "tests" / "convert_native"))
    sys.path.insert(0, str(repository))
    import synthetic_gemma4 as gemma
    from synthetic_scales import Artifact

    failures = []
    with tempfile.TemporaryDirectory(prefix="ninferno-poisoned-upload-") as tmp:
        work = Path(tmp)
        for layout in ("ct", "modelopt"):
            root = work / layout
            gemma.text_checkpoint(root, moe=True, layout=layout, nested=True, tied=True, seed=17)
            out = work / f"{layout}.ninferno"
            run = subprocess.run([str(native), f"--model={root}", "--recipe=gemma4_26b_a4b_nvfp4",
                                  "--quiet", f"--out={out}"], cwd=repository, capture_output=True,
                                 text=True)
            if run.returncode:
                failures.append(f"{layout}: conversion failed: {run.stderr[-500:]}")
                continue
            artifact = Artifact(out)
            words = 0
            for oid, o in artifact.objects.items():
                if o.get("format") == "nvfp4":
                    try:
                        values, scales, divisors = artifact.nvfp4_object(oid)
                    except Exception as error:  # noqa: BLE001 - reported with the object id
                        failures.append(f"{layout}: {oid} does not decode: {error}")
                        continue
                    words += scales.numel()
                    if not bool(torch.isfinite(values).all()) or not bool((divisors > 0).all()):
                        failures.append(f"{layout}: {oid} holds a non-finite value or divisor")
            env = dict(os.environ, NINFERNO_DEBUG_POISON="1")
            run = subprocess.run([str(test), "--artifact", str(out)], capture_output=True, text=True,
                                 env=env)
            print(f"{layout}: {words} NVFP4 scale words decoded; {run.stdout.strip()}", flush=True)
            if run.returncode == 77:
                return 77
            if run.returncode or "0 of them not poison" not in run.stdout:
                failures.append(f"{layout}: poisoned upload failed (rc {run.returncode}): "
                                f"{run.stderr[-800:]}")
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
