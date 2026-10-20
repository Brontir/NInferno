#!/usr/bin/env bash
# sign.sh MANIFEST [KEY]: signs a release manifest (latest.json) with the release key, writing
# MANIFEST.sig (base64 of the DER ECDSA P-256 / SHA-256 signature of its exact bytes), and checks the
# signature against tools/release/signing.pub.pem, the key every installer and updater carries.
#
# KEY defaults to $NINFERNO_SIGNING_KEY, else ~/.ninferno-release/signing-key.pem. The private key
# never goes to the release server or into the repository: a server that is broken into cannot
# sign, so it cannot make an installation take a file the release did not name.
set -euo pipefail
MANIFEST=${1:?usage: sign.sh MANIFEST [KEY]}
KEY=${2:-${NINFERNO_SIGNING_KEY:-$HOME/.ninferno-release/signing-key.pem}}
PUB=$(cd "$(dirname "$0")" && pwd)/signing.pub.pem
[ -f "$KEY" ] || { echo "sign.sh: no signing key at $KEY" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
openssl dgst -sha256 -sign "$KEY" -out "$tmp/sig.der" "$MANIFEST"
openssl dgst -sha256 -verify "$PUB" -signature "$tmp/sig.der" "$MANIFEST" > /dev/null ||
  { echo "sign.sh: $KEY is not the release key (tools/release/signing.pub.pem)" >&2; exit 1; }
base64 -w0 "$tmp/sig.der" > "$MANIFEST.sig"
echo "signed $MANIFEST"
