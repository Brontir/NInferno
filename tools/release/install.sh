#!/bin/sh
# NInferno installer for Linux and WSL 2: checks the NVIDIA GPUs and driver, downloads the release
# (one archive with an engine for every GPU generation) and installs it for the current user.
# Nothing is compiled.
#
#   curl -fsSL https://ninferno.brontir.com/releases/latest/download/install.sh | sh
#
# Environment:
#   NINFERNO_HOME      where releases are unpacked (default ~/.local/share/ninferno)
#   NINFERNO_BIN       where the commands are linked (default ~/.local/bin)
#   NINFERNO_URL       where the archives are (default the latest release on ninferno.brontir.com)
#   NINFERNO_ARCHIVE   install this archive file instead of downloading one
set -eu
URL=${NINFERNO_URL:-https://ninferno.brontir.com/releases/latest/download}
HOME_DIR=${NINFERNO_HOME:-$HOME/.local/share/ninferno}
BIN_DIR=${NINFERNO_BIN:-$HOME/.local/bin}

say() { printf '%s\n' "$*"; }
die() { printf 'NInferno install: %s\n' "$*" >&2; exit 1; }

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
if [ -n "${NINFERNO_ARCHIVE:-}" ]; then
  cp "$NINFERNO_ARCHIVE" "$tmp/$archive"
  [ -f "$NINFERNO_ARCHIVE.sha256" ] && cp "$NINFERNO_ARCHIVE.sha256" "$tmp/$archive.sha256"
else
  command -v curl > /dev/null 2>&1 || die "curl not found"
  say "Downloading $archive ..."
  curl -fL --progress-bar -o "$tmp/$archive" "$URL/$archive" || die "download failed: $URL/$archive"
  curl -fsSL -o "$tmp/$archive.sha256" "$URL/$archive.sha256" || die "checksum download failed"
fi
if [ -f "$tmp/$archive.sha256" ]; then
  ( cd "$tmp" && sha256sum -c "$archive.sha256" > /dev/null ) || die "checksum mismatch: $archive"
fi

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
