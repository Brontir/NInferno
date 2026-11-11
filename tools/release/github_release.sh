#!/usr/bin/env bash
# github_release.sh VERSION DIST_DIR [--publish]: the GitHub release of a version the release
# server already serves (server/publish.sh), as a draft unless --publish.
#
# The README sends Windows users to the GitHub release for NInferno-Setup.exe; the Windows zip and
# the WSL root file system it installs are there too. The files come from DIST_DIR, the directory
# publish.sh published, and each must match both its .sha256 and the SHA-256 the signed manifest
# (DIST_DIR/latest.json) gives for it, so GitHub carries the bytes the release server does. The tag
# v<VERSION> must already be on origin (gh --verify-tag), the notes are
# docs/release-notes/v<VERSION>.md, and a version with a pre-release label (0.2.0-rc1) is marked a
# pre-release. A release that is already there is never changed: a fix is a new version.
#
# Needs gh (signed in, with write access to the repository), git and python3.
set -euo pipefail
usage="usage: github_release.sh VERSION DIST_DIR [--publish]"
VERSION=${1:?$usage}
DIST=${2:?$usage}
PUBLISH=no
case ${3:-} in
  --publish) PUBLISH=yes ;;
  '') ;;
  *) echo "github_release.sh: unknown argument $3 ($usage)" >&2; exit 1 ;;
esac
[ $# -le 3 ] || { echo "github_release.sh: $usage" >&2; exit 1; }
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
die() { echo "github_release.sh: $*" >&2; exit 1; }

printf '%s' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.]+)?$' ||
  die "$VERSION is not a release version (0.1.0, 0.1.0-rc1)"
TAG=v$VERSION
FILES=(NInferno-Setup.exe ninferno-windows-x86_64.zip ninferno-wsl-rootfs.tar.gz)
NOTES=$ROOT/docs/release-notes/$TAG.md
[ -f "$NOTES" ] || die "no release notes $NOTES"
[ -f "$DIST/latest.json" ] || die "no $DIST/latest.json: give the directory publish.sh published"
for f in "${FILES[@]}"; do
  [ -f "$DIST/$f" ] && [ -f "$DIST/$f.sha256" ] || die "$DIST has no $f with its .sha256"
  have=$(sha256sum "$DIST/$f" | cut -d' ' -f1)
  [ "$have" = "$(cut -d' ' -f1 "$DIST/$f.sha256")" ] || die "$f does not match its .sha256"
  signed=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["assets"].get(sys.argv[2], {}).get("sha256", ""))' \
    "$DIST/latest.json" "$f")
  [ "$have" = "$signed" ] || die "$f is not the file latest.json names (SHA-256 ${signed:-missing})"
done
manifest=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$DIST/latest.json")
[ "$manifest" = "$VERSION" ] || die "$DIST/latest.json is the manifest of $manifest, not $VERSION"
git -C "$ROOT" ls-remote --exit-code --tags origin "refs/tags/$TAG" > /dev/null ||
  die "the tag $TAG is not on origin: push it first (git push origin $TAG)"
if gh release view "$TAG" > /dev/null 2>&1; then
  die "the GitHub release $TAG already exists; a published release is not changed (a fix is a new version)"
fi

args=(release create "$TAG" --verify-tag --title "NInferno $VERSION" --notes-file "$NOTES")
[ "$PUBLISH" = yes ] || args+=(--draft)
case $VERSION in *-*) args+=(--prerelease) ;; esac
for f in "${FILES[@]}"; do args+=("$DIST/$f" "$DIST/$f.sha256"); done
gh "${args[@]}"
if [ "$PUBLISH" = yes ]; then
  echo "published the GitHub release $TAG"
else
  echo "made the draft GitHub release $TAG: check it, then publish it (gh release edit $TAG --draft=false)"
fi
