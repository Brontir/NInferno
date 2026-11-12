#!/usr/bin/env python3
"""Add a published release's engines to the kit's release manifest (tools/kit/releases.json), the
list validate.py holds a user measurement's build to (P1, docs/maintainer/validation-protocol.md).

    python3 tools/kit/releases.py add --version 0.1.0 --commit SHA --platform linux-x86_64 BIN_DIR
        [--manifest tools/kit/releases.json]

BIN_DIR is the unpacked release's bin/ (the files as users install them): every engine
ninferno-serve-sm<tag>[.exe] in it is hashed. SHA is the commit the release was built at
(`git rev-parse v<VERSION>^{commit}`). Run once per platform archive. A version's platform that is
already listed is refused: a published release never changes.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate  # noqa: E402

ENGINE = re.compile(r"ninferno-serve-sm(\d+[a-z]?)(\.exe)?")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def engines_of(bin_dir):
    """{tag: sha256} of the engines in `bin_dir`."""
    out = {}
    for f in sorted(Path(bin_dir).iterdir()):
        m = ENGINE.fullmatch(f.name)
        if m and f.is_file():
            out[m.group(1)] = sha256_of(f)
    return out


def add(manifest, version, commit, platform, engines):
    """The manifest with the release's platform added; ValueError when it cannot be."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{commit!r} is not a full commit hash")
    if not engines:
        raise ValueError("no ninferno-serve-sm<tag> engine in the directory")
    entry = next((r for r in manifest["releases"] if r["version"] == version), None)
    if entry is None:
        entry = {"version": version, "commit": commit, "engines": {}}
        manifest["releases"].append(entry)
    elif entry["commit"] != commit:
        raise ValueError(f"release {version} is listed at commit {entry['commit']}")
    if platform in entry["engines"]:
        raise ValueError(f"release {version} already lists {platform} (a published release never changes)")
    entry["engines"][platform] = dict(sorted(engines.items()))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="add one platform archive's engines of a release")
    a.add_argument("--version", required=True)
    a.add_argument("--commit", required=True)
    a.add_argument("--platform", required=True, help="e.g. linux-x86_64, windows-x86_64")
    a.add_argument("--manifest", default=str(validate.RELEASES))
    a.add_argument("bin_dir")
    args = ap.parse_args()
    path = Path(args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate.load_releases(path)  # the manifest's own shape, before it is changed
    try:
        add(manifest, args.version, args.commit, args.platform, engines_of(args.bin_dir))
    except ValueError as e:
        sys.exit(f"releases.py: {e}")
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    validate.load_releases(path)
    entry = next(r for r in manifest["releases"] if r["version"] == args.version)
    print(f"{path}: release {args.version} {args.platform}, {len(entry['engines'][args.platform])} engines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
