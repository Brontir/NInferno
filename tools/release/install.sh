#!/bin/sh
# NInferno installer for Linux and WSL 2: checks the NVIDIA GPUs and driver, downloads the release
# (one archive with an engine for every GPU generation) and installs it for the current user.
# Nothing is compiled.
#
#   curl -fsSL https://raw.githubusercontent.com/Brontir/NInferno/master/tools/release/install.sh | sh
#   sh install.sh --verify MANIFEST    exit 0 when MANIFEST (MANIFEST.sig beside it) is signed
#                                      by the release key (ninferno-update asks this)
#
# What is installed is what the release manifest (latest.json) names, and the manifest counts only
# with a valid signature of the release key below: a release server that is broken into cannot
# sign, so it cannot make this script install anything else. The archive is checked against the
# SHA-256 the signed manifest gives for it.
#
# Environment:
#   NINFERNO_HOME         where releases are unpacked (default ~/.local/share/ninferno)
#   NINFERNO_BIN          where the commands are linked (default ~/.local/bin)
#   NINFERNO_UPDATE_URL   the manifest (default the release server's latest.json); its signature
#                         is checked all the same
#   NINFERNO_MANIFEST     a manifest file already downloaded (MANIFEST.sig beside it), instead of
#                         reading one
#   NINFERNO_ARCHIVE      install this archive file instead of downloading one (checked against
#                         the signed manifest too)
set -eu
MANIFEST_URL=${NINFERNO_UPDATE_URL:-https://ninferno.brontir.com/releases/latest.json}
HOME_DIR=${NINFERNO_HOME:-$HOME/.local/share/ninferno}
BIN_DIR=${NINFERNO_BIN:-$HOME/.local/bin}
# The release key (tools/release/signing.pub.pem; tests check every copy against it).
RELEASE_KEY='-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE1R5JganVYknHjhX9nN2fYLqxwe2G
R9Zeg0QL/WoCKgTm4fpmwOM4ASQ+0Av3ASV+RPHQH1rTaXBSHyWCPZokZg==
-----END PUBLIC KEY-----'

say() { printf '%s\n' "$*"; }
die() { printf 'NInferno install: %s\n' "$*" >&2; exit 1; }

# verify_manifest FILE: 0 when FILE.sig is the release key's signature of FILE's exact bytes.
verify_manifest() {
  command -v openssl > /dev/null 2>&1 ||
    die "openssl is needed to check the release's signature (install the openssl package)"
  work=$(mktemp -d)
  printf '%s\n' "$RELEASE_KEY" > "$work/release.pub"
  ok=1
  if base64 -d "$1.sig" > "$work/sig.der" 2> /dev/null &&
    openssl dgst -sha256 -verify "$work/release.pub" -signature "$work/sig.der" "$1" > /dev/null 2>&1; then
    ok=0
  fi
  rm -rf "$work"
  return $ok
}

# asset_field NAME FIELD MANIFEST: the asset's "url" or "sha256" as manifest.py writes them (one
# key per line; the manifest is signed, so its layout is the one manifest.py gives it).
asset_field() {
  awk -v name="\"$1\": {" -v key="\"$2\":" '
    index($0, name) { on = 1; next }
    on && index($0, "}") { exit }
    on && index($0, key) { sub(/^[^:]*: *"/, ""); sub(/".*$/, ""); print; exit }' "$3"
}

if [ "${1:-}" = --verify ]; then
  [ -f "${2:-}" ] || die "usage: install.sh --verify MANIFEST"
  verify_manifest "$2" || die "$2 is not signed by the NInferno release key"
  exit 0
fi

[ "$(uname -s)" = Linux ] || die "this installer is for Linux and WSL 2; on Windows run NInferno-Setup.exe"
[ "$(uname -m)" = x86_64 ] || die "the release builds are for x86-64 Linux"
command -v nvidia-smi > /dev/null 2>&1 ||
  die "nvidia-smi not found: install the NVIDIA driver (on WSL 2, the Windows driver provides it)"

# Every GPU; the release runs on each from compute capability 7.0 (an engine per generation, PTX
# for later ones): at least one must qualify.
gpus=$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
  --format=csv,noheader,nounits) || die "nvidia-smi cannot read the GPUs"
usable=0; min_driver=0; driver=""
while IFS=, read -r index name cap mem drv; do
  name=$(printf '%s' "$name" | sed 's/^ *//; s/ *$//'); cap=$(printf '%s' "$cap" | tr -d ' ')
  mem=$(printf '%s' "$mem" | tr -d ' '); driver=$(printf '%s' "$drv" | tr -d ' ')
  major_cc=${cap%%.*}; minor_cc=${cap#*.}
  case $major_cc$minor_cc in
    '' | *[!0-9]*) say "GPU$index: $name: compute capability unknown ($cap), skipped"; continue ;;
  esac
  if [ "$major_cc" -ge 7 ]; then
    usable=$((usable + 1))
    # CUDA 13 (driver 580) for every engine but Volta's, CUDA 12.8 (570).
    if [ "$major_cc" = 7 ] && [ "$minor_cc" -lt 5 ]; then need=570; else need=580; fi
    [ "$need" -gt "$min_driver" ] && min_driver=$need
    say "GPU$index: $name, compute capability $cap, $((mem / 1024)) GB"
  else
    say "GPU$index: $name, compute capability $cap: older than NInferno supports (7.0 and newer)"
  fi
done <<EOF
$gpus
EOF
[ "$usable" -gt 0 ] || die "no GPU here is supported (compute capability 7.0 and newer: RTX 20 series, V100 and later)"
major=${driver%%.*}
# A CUDA runtime does not run on a driver of an older major release (minor-version compatibility
# holds within one major only): installing goes on, so the driver can be updated afterwards.
[ "$major" -ge "$min_driver" ] ||
  say "WARNING: NVIDIA driver $driver is too old: NInferno needs driver $min_driver or newer and will not start until it is updated (nvidia.com/drivers)."

archive=ninferno-linux-x86_64.tar.gz
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ -n "${NINFERNO_MANIFEST:-}" ]; then
  cp "$NINFERNO_MANIFEST" "$tmp/latest.json" && cp "$NINFERNO_MANIFEST.sig" "$tmp/latest.json.sig" ||
    die "no $NINFERNO_MANIFEST with its .sig"
else
  command -v curl > /dev/null 2>&1 || die "curl not found"
  curl -fsSL --max-time 60 -o "$tmp/latest.json" "$MANIFEST_URL" &&
    curl -fsSL --max-time 60 -o "$tmp/latest.json.sig" "$MANIFEST_URL.sig" ||
    die "cannot read the release manifest $MANIFEST_URL"
fi
verify_manifest "$tmp/latest.json" ||
  die "the release manifest is not signed by the NInferno release key: nothing installed"
sha=$(asset_field "$archive" sha256 "$tmp/latest.json")
url=$(asset_field "$archive" url "$tmp/latest.json")
printf '%s' "$sha" | grep -Eq '^[0-9a-f]{64}$' || die "the release manifest names no $archive"
if [ -n "${NINFERNO_ARCHIVE:-}" ]; then
  cp "$NINFERNO_ARCHIVE" "$tmp/$archive"
else
  case $url in https://*) ;; *) die "the release manifest gives no https address for $archive" ;; esac
  say "Downloading $archive ..."
  curl -fL --proto =https --progress-bar -o "$tmp/$archive" "$url" || die "download failed: $url"
fi
printf '%s  %s\n' "$sha" "$tmp/$archive" | sha256sum -c - > /dev/null 2>&1 ||
  die "$archive does not match the signed release manifest: nothing installed"

mkdir -p "$HOME_DIR" "$BIN_DIR"
top=$(tar -tzf "$tmp/$archive" | head -1 | cut -d/ -f1)
# An unreadable archive lists nothing (sh has no pipefail): never remove the install home itself.
case $top in "" | . | ..) die "not a NInferno archive: $archive" ;; esac
rm -rf "${HOME_DIR:?}/$top"
tar -xzf "$tmp/$archive" -C "$HOME_DIR"
ln -sfn "$HOME_DIR/$top" "$HOME_DIR/current"
# The commands (the engines, bin/*-sm<tag>, are started by ninferno-serve itself).
for exe in "$HOME_DIR/current/bin/"*; do
  case $(basename "$exe") in *-sm[0-9]*) continue ;; esac
  ln -sf "$exe" "$BIN_DIR/$(basename "$exe")"
done
if ! "$BIN_DIR/ninferno-serve" --version; then
  die "the installed build cannot run on this machine (see above); a newer NVIDIA driver usually fixes it"
fi

say "Installed $top into $HOME_DIR, commands in $BIN_DIR."
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "Add $BIN_DIR to PATH, e.g.: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc" ;; esac
say ""
say "Start NInferno:"
say "  ninferno-ui                  the local page: pick a model, download and run it (opens in your browser)"
say "  ninferno-serve ORG/NAME ...  a model straight from the command line, e.g."
say "      ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters"
