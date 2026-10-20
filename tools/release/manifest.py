#!/usr/bin/env python3
"""Write a release's manifest (latest.json) from its files.

    manifest.py --version 0.1.0 --base https://ninferno.brontir.com/releases/v0.1.0 \
        [--notes URL] [--min-version 0.1.0] DIST_DIR > latest.json

Every file of DIST_DIR that has a .sha256 beside it is listed under "assets" with its URL (BASE/
NAME), size and SHA-256. ninferno-ui and ninferno-serve read "version", "notes" and
"min_version" (product/update/update_check.h); install.sh (and ninferno-update through it) takes
the Linux archive's asset, NInferno-Setup.exe its own and the engine for the GPU. Clients act on
the manifest only with a valid signature of the release key (sign.sh writes latest.json.sig).
Publish the release's files under BASE first and latest.json last: a client that reads the new
manifest finds everything it names.
"""

import argparse
import datetime
import json
import pathlib
import re
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--base", required=True, help="URL the release's files are served under")
    parser.add_argument("--notes", default="", help="what changed (a page)")
    parser.add_argument("--min-version", default="",
                        help="installs older than this are told they must update")
    parser.add_argument("dist", type=pathlib.Path)
    args = parser.parse_args()
    release = re.compile(r"^v?\d+(\.\d+){1,3}(-[0-9A-Za-z.]+)?$")
    for value in (args.version, args.min_version):
        if value and (not release.match(value) or "-g" in value):
            parser.error(f"{value!r} is not a release version (0.1.0, 0.1.0-rc1)")
    base = args.base.rstrip("/")
    assets = {}
    for checksum in sorted(args.dist.glob("*.sha256")):
        target = checksum.with_suffix("")
        if not target.is_file():
            continue
        digest = checksum.read_text(encoding="utf-8").split()[0].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error(f"{checksum}: no SHA-256")
        assets[target.name] = {"url": f"{base}/{target.name}", "size": target.stat().st_size,
                               "sha256": digest}
    if not assets:
        parser.error(f"no files with a .sha256 in {args.dist}")
    manifest = {"version": args.version, "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                "notes": args.notes, "min_version": args.min_version, "assets": assets}
    # install.sh reads the assets line by line from this layout (one key per line, indent 1): the
    # bytes are signed, so it cannot be changed under it, but changing it here breaks the installer.
    json.dump(manifest, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
