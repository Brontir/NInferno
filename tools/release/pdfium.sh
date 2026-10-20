#!/usr/bin/env bash
# pdfium.sh PLATFORM LIB_DIR LICENSE_DIR: the pdfium build a release carries for PDF documents
# (serve reads a PDF's text through it, loaded at run time), from bblanchon/pdfium-binaries at a
# pinned version, its archive checked against the SHA-256 recorded in pdfium.pin. PLATFORM is linux-x64 or
# win-x64. NINFERNO_PDFIUM_ARCHIVE names a local copy of the archive (an offline build).
set -euo pipefail
PLATFORM=${1:?usage: pdfium.sh PLATFORM LIB_DIR LICENSE_DIR} LIB_DIR=${2:?usage: pdfium.sh PLATFORM LIB_DIR LICENSE_DIR}
LICENSE_DIR=${3:?usage: pdfium.sh PLATFORM LIB_DIR LICENSE_DIR}
PIN="$(dirname "$0")/pdfium.pin"
TAG=$(sed -n 's/^tag //p' "$PIN")  # chromium/8076: pdfium 156.0.8076.0, published 2026-09-29
SHA=$(sed -n "s/^$PLATFORM //p" "$PIN")
case $PLATFORM in
  linux-x64) LIB=lib/libpdfium.so ;;
  win-x64)   LIB=bin/pdfium.dll ;;
  *) echo "pdfium.sh: unknown platform $PLATFORM" >&2; exit 1 ;;
esac
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
archive=$tmp/pdfium-$PLATFORM.tgz
if [ -n "${NINFERNO_PDFIUM_ARCHIVE:-}" ]; then
  cp "$NINFERNO_PDFIUM_ARCHIVE" "$archive"
else
  curl -fsSL -o "$archive" "https://github.com/bblanchon/pdfium-binaries/releases/download/${TAG/\//%2F}/pdfium-$PLATFORM.tgz"
fi
echo "$SHA  $archive" | sha256sum -c --quiet - || { echo "pdfium.sh: checksum mismatch for $TAG $PLATFORM" >&2; exit 1; }
mkdir -p "$tmp/x" "$LIB_DIR" "$LICENSE_DIR"
tar -xzf "$archive" -C "$tmp/x"
install -m 0755 "$tmp/x/$LIB" "$LIB_DIR/"
cp "$tmp/x/LICENSE" "$LICENSE_DIR/LICENSE"
[ -d "$tmp/x/licenses" ] && cp -r "$tmp/x/licenses" "$LICENSE_DIR/third_party"
echo "pdfium $TAG ($PLATFORM) -> $LIB_DIR/$(basename "$LIB")"
