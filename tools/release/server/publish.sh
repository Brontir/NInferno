#!/usr/bin/env bash
# publish.sh VERSION DIST_DIR [--notes URL] [--min-version V] [--host USER@HOST] [--root /srv/ninferno]
#
# Publishes a release to the release server (README.md here): the files of DIST_DIR that have a
# .sha256 go to <root>/releases/v<VERSION>/ and are checked there, then latest.json (signed here by
# sign.sh with the release key, which never leaves this machine) with its .sig and the
# releases/latest link are switched to it, each by an atomic rename, so a client never sees a
# manifest naming a file that is not there yet. A version is published once: a published release
# is served as immutable, so a fix is a new version. NINFERNO_RELEASE_HOST is the default --host;
# NINFERNO_SIGNING_KEY the key (sign.sh).
set -euo pipefail
VERSION=${1:?usage: publish.sh VERSION DIST_DIR [--notes URL] [--min-version V] [--host USER@HOST] [--root DIR]}
DIST=$(realpath "${2:?DIST_DIR}")
shift 2
NOTES=""; MIN=""; HOST=${NINFERNO_RELEASE_HOST:-releases@ninferno.brontir.com}; ROOT=/srv/ninferno
while [ $# -gt 0 ]; do
  case $1 in
    --notes) NOTES=$2; shift 2 ;;
    --min-version) MIN=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --root) ROOT=$2; shift 2 ;;
    *) echo "publish.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
# The version goes into remote shell commands: letters, digits, dots, dashes and underscores only.
[[ ${VERSION#v} =~ ^[0-9A-Za-z._-]+$ ]] || { echo "publish.sh: invalid version $VERSION" >&2; exit 2; }
TAG=v${VERSION#v}
BASE=https://ninferno.brontir.com/releases/$TAG

files=()
for sum in "$DIST"/*.sha256; do
  file=${sum%.sha256}
  [ -f "$file" ] || continue
  ( cd "$DIST" && sha256sum -c "$(basename "$sum")" > /dev/null ) || { echo "publish.sh: $file does not match its .sha256" >&2; exit 1; }
  files+=("$file" "$sum")
done
[ ${#files[@]} -gt 0 ] || { echo "publish.sh: nothing with a .sha256 in $DIST" >&2; exit 1; }

args=(--version "${VERSION#v}" --base "$BASE")
[ -n "$NOTES" ] && args+=(--notes "$NOTES")
[ -n "$MIN" ] && args+=(--min-version "$MIN")
# Installers and updaters act only on a manifest the release key signed. The key stays on the
# machine that signs: where it is not (a build machine), the manifest is written and this stops;
# signing it there (sign.sh DIST/latest.json) and bringing latest.json.sig back resumes here.
verify() {
  openssl dgst -sha256 -verify "$HERE/../signing.pub.pem" -signature <(base64 -d "$DIST/latest.json.sig") \
    "$DIST/latest.json" > /dev/null 2>&1
}
if [ -f "$DIST/latest.json.sig" ] && verify; then
  grep -q "\"version\": \"${VERSION#v}\"" "$DIST/latest.json" ||
    { echo "publish.sh: the signed latest.json is not $VERSION's" >&2; exit 1; }
else
  python3 "$HERE/../manifest.py" "${args[@]}" "$DIST" > "$DIST/latest.json"
  rm -f "$DIST/latest.json.sig"
  if [ -f "${NINFERNO_SIGNING_KEY:-$HOME/.ninferno-release/signing-key.pem}" ]; then
    bash "$HERE/../sign.sh" "$DIST/latest.json"
  else
    echo "publish.sh: $DIST/latest.json written; sign it where the release key is" \
      "(tools/release/sign.sh latest.json), put latest.json.sig beside it and run this again" >&2
    exit 3
  fi
fi

# published: latest names it; uploaded: its directory is there but the switch did not happen (a
# run that stopped after the move), which this run completes; new: neither.
state=$(ssh "$HOST" "cd '$ROOT/releases' && if [ \"\$(readlink latest 2>/dev/null)\" = '$TAG' ]; then echo published;\
  elif [ -e '$TAG' ]; then echo uploaded; else echo new; fi")
[ "$state" != published ] || { echo "publish.sh: $TAG is already published" >&2; exit 1; }
if [ "$state" = new ]; then
  ssh "$HOST" "rm -rf '$ROOT/releases/$TAG.partial' && mkdir -p '$ROOT/releases/$TAG.partial'"
  rsync -a --partial "${files[@]}" "$HOST:$ROOT/releases/$TAG.partial/"
  # Checked where they will be served from, then the release directory appears whole.
  ssh "$HOST" "cd '$ROOT/releases/$TAG.partial' && sha256sum -c --quiet *.sha256 && \
    mv '../$TAG.partial' '../$TAG'"
fi
scp "$DIST/latest.json" "$HOST:$ROOT/releases/latest.json.new"
scp "$DIST/latest.json.sig" "$HOST:$ROOT/releases/latest.json.sig.new"
# The signature and the manifest switch back to back: a client that reads them in between finds
# them not matching and acts on neither (it asks again later).
ssh "$HOST" "cd '$ROOT/releases' && ln -sfn '$TAG' latest.new && mv -T latest.new latest && \
  mv latest.json.sig.new latest.json.sig && mv latest.json.new latest.json"
echo "published $TAG: $BASE (manifest https://ninferno.brontir.com/releases/latest.json)"
