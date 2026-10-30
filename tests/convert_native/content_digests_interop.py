"""The content digests and source identity agree between the Python and the native writer.

tools/artifact/writer.py (content_digests=True, as tools/convert/pipeline.py writes) and
src/convert/writer.h (as convert() writes) seal the SHA-256 of every object's bytes into the
directory. The same description and payload written by both give the same files apart from the
random artifact id, and so the same source identity. No torch.

    content_digests_interop.py TEST_CONTENT_DIGESTS
"""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.schema import TensorSpec  # noqa: E402
from tools.artifact.writer import ArtifactWriter  # noqa: E402


def main() -> int:
    executable = sys.argv[1]
    failures = 0
    data = bytes((index * 37 + 11) % 251 for index in range(130 * 130 * 2))
    tail = bytes(range(128))
    with tempfile.TemporaryDirectory(prefix="ninferno-digests-interop-") as temporary:
        for label, limit in (("single", 1_000_000), ("sharded", 12288)):
            python_out = Path(temporary) / f"{label}.python.ninferno"
            native_out = Path(temporary) / f"{label}.native.ninferno"
            provenance = {
                "converter": "ninferno-v3",
                "recipe": "synthetic",
                "inputs": [{"role": "base", "files": [{"sha256": "a" * 64}]}],
            }
            with ArtifactWriter(
                python_out,
                [TensorSpec("matrix", (130, 130), "bf16", "contiguous_le_v1"),
                 TensorSpec("tail", (64,), "bf16", "contiguous_le_v1")],
                components={"text": {"config": {}}},
                bindings={"whole": {"object": "matrix"}, "tail": {"object": "tail"}},
                provenance=provenance,
                max_file_bytes=limit,
                content_digests=True,
            ) as writer:
                writer.write_object("matrix", data)
                writer.write_object("tail", tail)
            digests = provenance["content_sha256"]
            if digests != {"matrix": hashlib.sha256(data).hexdigest(),
                           "tail": hashlib.sha256(tail).hexdigest()}:
                print(f"FAILED {label}: Python digests {digests}")
                failures += 1
            subprocess.run([executable, "--rewrite", str(python_out), str(native_out),
                            str(limit)], check=True)
            for py_file in sorted(Path(temporary).glob(f"{label}.python.ninferno*")):
                native_file = Path(str(py_file).replace(".python.", ".native."))
                # The entry directory names the part files ("native" and "python" are of one
                # length), and bytes 16..32 of a header are the random artifact id.
                a = py_file.read_bytes()
                b = native_file.read_bytes().replace(b".native.", b".python.")
                if a[:16] != b[:16] or a[32:] != b[32:]:
                    print(f"FAILED {label}: {py_file.name} and {native_file.name} differ")
                    failures += 1
            identities = [
                subprocess.run([executable, "--identity", str(path)], check=True,
                               capture_output=True, text=True).stdout.strip()
                for path in (python_out, native_out)
            ]
            if identities[0] != identities[1] or len(identities[0]) != 64:
                print(f"FAILED {label}: identities {identities}")
                failures += 1
    if failures == 0:
        print("content digests interop: all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
