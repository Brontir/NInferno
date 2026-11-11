#!/bin/sh
# NInferno installer for Linux and WSL 2: checks the NVIDIA GPUs and driver, downloads the release
# (one archive with an engine for every GPU generation) and installs it for the current user.
# Nothing is compiled.
#
#   curl -fsSL https://raw.githubusercontent.com/Brontir/NInferno/master/tools/release/install.sh | sh
#   sh install.sh                      install the newest release (on a GPU NInferno does not
#                                      support, such as the GTX 16 series, it asks first)
#   sh install.sh --yes                the same without asking (ninferno-update)
#   sh install.sh --uninstall [--yes]  remove what it installed (asks first unless --yes)
#   sh install.sh --verify MANIFEST    exit 0 when MANIFEST (MANIFEST.sig beside it) is signed
#                                      by the release key and names an archive this script
#                                      installs (ninferno-update asks this)
#   sh install.sh --gpu-check ID CAP   exit 0 when NInferno supports the GPU of that PCI product
#                                      id and compute capability, else say why (ninferno-start)
#   sh install.sh --help               this text
#
# The release needs x86-64 Linux with glibc 2.38 or newer (Ubuntu 24.04, Debian 13, Fedora 39 and
# later): the C library is the system's, and the archive is built against that version.
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
# The commands' directory an install here last used (BIN_FILE): an update, a rollback or an
# uninstall finds it without NINFERNO_BIN given again.
BIN_FILE=$HOME_DIR/.bin-dir
recorded_bin=$(sed -n 1p "$BIN_FILE" 2> /dev/null || true)
BIN_DIR=${NINFERNO_BIN:-${recorded_bin:-$HOME/.local/bin}}
archive=ninferno-linux-x86_64.tar.gz
# The oldest glibc the archive runs on: the build machine's (tools/release/package.sh checks it).
GLIBC_MIN=2.38
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

# installable MANIFEST: 0 when MANIFEST names this platform's archive with a SHA-256 and an https
# address (or NINFERNO_ARCHIVE stands in for the download), the install's own checks; else it says
# why. --verify asks it too, so ninferno-update --check never offers what --yes then refuses.
installable() {
  printf '%s' "$(asset_field "$archive" sha256 "$1")" | grep -Eq '^[0-9a-f]{64}$' ||
    { say "the release manifest names no $archive"; return 1; }
  [ -n "${NINFERNO_ARCHIVE:-}" ] && return 0
  case $(asset_field "$archive" url "$1") in https://*) return 0 ;; esac
  say "the release manifest gives no https address for $archive (a rehearsal without TLS sets NINFERNO_ARCHIVE)"
  return 1
}

# no_tensor_cores PCI_ID CAP: 0 when the GPU's chip has no tensor cores, by its PCI product id
# (nvidia-smi's pci.device_id, 0xDDDDVVVV: device, then vendor 10DE) and compute capability, as
# the engine's GPU facts say: TU117 (0x1F80-0x1FFF) and TU116 (0x2180-0x21FF) at 7.5
# (chip_mma_slow, core/device_class_host.cpp), and the GTX 16 boards on TU106 (0x1F09, 0x1F0A),
# whose rows in core/data/gpu_facts.json carry mma_slow. This script runs before anything is
# downloaded, so it holds the ids itself; tests/release/signed_install_test.sh checks them against
# both sources.
no_tensor_cores() {
  [ "$2" = 7.5 ] || return 1
  case $(printf '%s' "$1" | tr a-fx A-FX) in
    0X1F[89A-F][0-9A-F]10DE | 0X21[89A-F][0-9A-F]10DE | 0X1F0910DE | 0X1F0A10DE) return 0 ;;
  esac
  return 1
}
NO_TENSOR_CORES="its chip has no tensor cores (the GTX 16 series: TU116, TU117)"

# gpu_unsupported PCI_ID CAP: 0, printing why, when NInferno does not support that GPU.
gpu_unsupported() {
  case $2 in
    [0-9].[0-9] | [0-9][0-9].[0-9]) ;;
    *) say "its compute capability is unknown ($2)"; return 0 ;;
  esac
  if [ "${2%%.*}" -lt 7 ]; then
    say "compute capability $2 is older than NInferno supports (7.0 and newer)"; return 0
  fi
  if no_tensor_cores "$1" "$2"; then
    say "$NO_TENSOR_CORES; NInferno may not run on it, or may run slowly"; return 0
  fi
  return 1
}

usage() {
  say "NInferno installer for Linux and WSL 2."
  say "  install.sh                      install the newest release"
  say "  install.sh --yes                the same, without asking on a GPU NInferno does not support"
  say "  install.sh --uninstall [--yes]  remove what it installed from $HOME_DIR and $BIN_DIR"
  say "  install.sh --verify MANIFEST    exit 0 when MANIFEST.sig is the release key's signature"
  say "                                  and MANIFEST names an archive this script installs"
  say "  install.sh --gpu-check ID CAP   exit 0 when NInferno supports that GPU, else say why"
  say "  install.sh --help               this text"
  say "Through a pipe: curl -fsSL .../install.sh | sh -s -- --uninstall"
}

# Staging directories (HOME_DIR/.install.XXXXXX) carry STAGING_MARK with the PID of the install
# that made them. remove_stale_staging removes those whose install is gone: the PID is not alive,
# or the mark is a day old (a PID used again; no install takes that long), or, when the mark holds
# no PID, six hours old. An unmarked .install.* is not ours, and a live one is another install's
# (ninferno-ui's Update and a manual run at once): both stay.
STAGING_MARK=.ninferno-staging
remove_stale_staging() {
  for old in "$HOME_DIR"/.install.*; do
    [ -d "$old" ] && [ ! -L "$old" ] && [ -f "$old/$STAGING_MARK" ] || continue
    pid=$(head -n 1 "$old/$STAGING_MARK" 2> /dev/null || true)
    case $pid in
      '' | *[!0-9]*) age=360 ;;
      *) if kill -0 "$pid" 2> /dev/null; then age=1440; else age=0; fi ;;
    esac
    if [ "$age" = 0 ] || [ -n "$(find "$old/$STAGING_MARK" -mmin +"$age" 2> /dev/null)" ]; then
      rm -rf "$old"
    fi
  done
}

# uninstall yes|ask: only what install.sh and ninferno-update make, so a custom NINFERNO_HOME
# (even ~/.local or ~) loses nothing else: the release directories (each holds
# share/ninferno/release.json), `current` when it is a link to one of them, staging copies left
# by an install that is gone, and the links in BIN_DIR to a release's bin/. A link may name the home as given or resolved
# (ninferno-update passes the resolved one). Downloaded models and settings live elsewhere.
uninstall() {
  real=$(readlink -f "$HOME_DIR" 2> /dev/null || printf '%s' "$HOME_DIR")
  [ -d "$HOME_DIR" ] || die "nothing is installed in $HOME_DIR"
  say "This removes NInferno from $HOME_DIR and its commands from $BIN_DIR (downloaded models stay)."
  if [ "$1" != yes ]; then
    printf 'Remove it? [y/N] '
    { read -r answer < /dev/tty; } 2> /dev/null || die "no terminal to ask: run it with --uninstall --yes"
    case $answer in [yY]*) ;; *) say "Nothing removed."; exit 0 ;; esac
  fi
  for link in "$BIN_DIR"/*; do
    [ -L "$link" ] || continue
    case $(readlink "$link") in
      "$HOME_DIR"/current/bin/* | "$HOME_DIR"/ninferno-*-linux-x86_64*/bin/* | \
        "$real"/current/bin/* | "$real"/ninferno-*-linux-x86_64*/bin/*) rm -f "$link" ;;
    esac
  done
  if [ -L "$HOME_DIR/current" ]; then
    case $(readlink "$HOME_DIR/current") in
      "$HOME_DIR"/ninferno-*-linux-x86_64* | "$real"/ninferno-*-linux-x86_64*) rm -f "$HOME_DIR/current" ;;
    esac
  fi
  for dir in "$HOME_DIR"/ninferno-*-linux-x86_64*; do
    [ -d "$dir" ] && [ ! -L "$dir" ] || continue
    if [ -f "$dir/share/ninferno/release.json" ]; then rm -rf "$dir"; fi
  done
  remove_stale_staging
  rm -f "$BIN_FILE"
  if rmdir "$HOME_DIR" 2> /dev/null; then
    say "NInferno is removed."
  else
    say "NInferno is removed; $HOME_DIR stays, it holds files NInferno did not put there."
  fi
}

assume_yes=no
case ${1:-} in
  --verify)
    [ $# = 2 ] && [ -f "$2" ] || die "usage: install.sh --verify MANIFEST"
    verify_manifest "$2" || die "$2 is not signed by the NInferno release key"
    why=$(installable "$2") || die "$why"
    exit 0 ;;
  --gpu-check)
    [ $# = 3 ] || die "usage: install.sh --gpu-check PCI_DEVICE_ID COMPUTE_CAP"
    if gpu_unsupported "$2" "$3"; then exit 1; fi
    exit 0 ;;
  -h | --help) [ $# = 1 ] || die "--help takes no arguments"; usage; exit 0 ;;
  --uninstall)
    case $#:${2:-} in 1:) uninstall ask ;; 2:--yes | 2:-y) uninstall yes ;; *) die "usage: install.sh --uninstall [--yes]" ;; esac
    exit 0 ;;
  --yes | -y) [ $# = 1 ] || die "usage: install.sh [--yes]"; assume_yes=yes ;;
  '') [ $# = 0 ] || die "unknown argument '' (see --help)" ;;
  *) die "unknown argument $1 (see --help)" ;;
esac

[ "$(uname -s)" = Linux ] || die "this installer is for Linux and WSL 2; on Windows run NInferno-Setup.exe"
[ "$(uname -m)" = x86_64 ] || die "the release builds are for x86-64 Linux"
# version_at_least HAVE NEED: 0 when the dotted version HAVE is NEED or later.
version_at_least() {
  printf '%s\n%s\n' "$1" "$2" | awk -F. 'NR == 1 { for (i = 1; i <= 4; i++) h[i] = $i + 0 }
    NR == 2 { for (i = 1; i <= 4; i++) { n = $i + 0; if (h[i] != n) exit (h[i] > n) ? 0 : 1 } }'
}
# The C library is the system's: an older glibc cannot load the release's binaries.
glibc=$(getconf GNU_LIBC_VERSION 2> /dev/null | sed -n 's/^glibc \([0-9][0-9.]*\).*/\1/p')
[ -n "$glibc" ] || die "this system's C library is not glibc; the release needs glibc $GLIBC_MIN or newer"
version_at_least "$glibc" "$GLIBC_MIN" ||
  die "glibc $glibc is too old: the release needs glibc $GLIBC_MIN or newer (Ubuntu 24.04, Debian 13, Fedora 39 and later)"
command -v nvidia-smi > /dev/null 2>&1 ||
  die "nvidia-smi not found: install the NVIDIA driver (on WSL 2, the Windows driver provides it)"

# Every GPU; the release runs on each from compute capability 7.0 (an engine per generation, PTX
# for later ones): at least one must qualify. A chip without tensor cores (GTX 16) is not
# supported either: when no other GPU qualifies, the user is asked whether to install anyway.
gpus=$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version,pci.device_id \
  --format=csv,noheader,nounits) || die "nvidia-smi cannot read the GPUs"
usable=0; no_tc=0; min_driver=0; driver=""
while IFS=, read -r index name cap mem drv pci; do
  name=$(printf '%s' "$name" | sed 's/^ *//; s/ *$//'); cap=$(printf '%s' "$cap" | tr -d ' ')
  mem=$(printf '%s' "$mem" | tr -d ' '); driver=$(printf '%s' "$drv" | tr -d ' ')
  pci=$(printf '%s' "$pci" | tr -d ' ')
  major_cc=${cap%%.*}; minor_cc=${cap#*.}
  case $major_cc$minor_cc in
    '' | *[!0-9]*) say "GPU$index: $name: compute capability unknown ($cap), skipped"; continue ;;
  esac
  if [ "$major_cc" -ge 7 ]; then
    # CUDA 13 (driver 580) for every engine but Volta's, CUDA 12.8 (570).
    if [ "$major_cc" = 7 ] && [ "$minor_cc" -lt 5 ]; then need=570; else need=580; fi
    [ "$need" -gt "$min_driver" ] && min_driver=$need
    if no_tensor_cores "$pci" "$cap"; then
      no_tc=$((no_tc + 1))
      say "GPU$index: $name, compute capability $cap, $((mem / 1024)) GB: NOT SUPPORTED, $NO_TENSOR_CORES"
    else
      usable=$((usable + 1))
      say "GPU$index: $name, compute capability $cap, $((mem / 1024)) GB"
    fi
  else
    say "GPU$index: $name, compute capability $cap: older than NInferno supports (7.0 and newer)"
  fi
done <<EOF
$gpus
EOF
if [ "$usable" = 0 ] && [ "$no_tc" -gt 0 ]; then
  say "WARNING: NInferno does not support a GPU without tensor cores: it may not run, or may run slowly."
  if [ "$assume_yes" != yes ]; then
    printf 'Install it anyway? [y/N] '
    { read -r answer < /dev/tty; } 2> /dev/null ||
      die "no terminal to ask: run it with --yes to install on this GPU anyway (curl ... | sh -s -- --yes)"
    case $answer in [yY]*) ;; *) say "Nothing installed."; exit 0 ;; esac
  fi
  usable=$no_tc
fi
[ "$usable" -gt 0 ] || die "no GPU here is supported (compute capability 7.0 and newer: RTX 20 series, V100 and later)"
major=${driver%%.*}
# A CUDA runtime does not run on a driver of an older major release (minor-version compatibility
# holds within one major only): installing goes on, so the driver can be updated afterwards.
[ "$major" -ge "$min_driver" ] ||
  say "WARNING: NVIDIA driver $driver is too old: NInferno needs driver $min_driver or newer and will not start until it is updated (nvidia.com/drivers)."

tmp=$(mktemp -d)
stage=""  # the unpacked release before it is switched in, inside HOME_DIR (a move, not a copy)
top=""
# A same-version copy moved aside (below) goes back when the new one never got its place.
cleanup() {
  if [ -n "$stage" ] && [ -n "$top" ] && { [ -e "$stage/replaced" ] || [ -L "$stage/replaced" ]; } &&
    [ ! -e "$HOME_DIR/$top" ] && [ ! -L "$HOME_DIR/$top" ]; then
    mv -T "$stage/replaced" "$HOME_DIR/$top" || true
  fi
  rm -rf "$tmp" ${stage:+"$stage"}
}
# dash runs no EXIT trap when a signal kills it: a stopped install would leave its staging copy.
trap cleanup EXIT
trap 'cleanup; trap - EXIT; exit 129' HUP
trap 'cleanup; trap - EXIT; exit 130' INT
trap 'cleanup; trap - EXIT; exit 143' TERM
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
why=$(installable "$tmp/latest.json") || die "$why"
sha=$(asset_field "$archive" sha256 "$tmp/latest.json")
url=$(asset_field "$archive" url "$tmp/latest.json")
if [ -n "${NINFERNO_ARCHIVE:-}" ]; then
  cp "$NINFERNO_ARCHIVE" "$tmp/$archive"
else
  say "Downloading $archive ..."
  curl -fL --proto =https --progress-bar -o "$tmp/$archive" "$url" || die "download failed: $url"
fi
printf '%s  %s\n' "$sha" "$tmp/$archive" | sha256sum -c - > /dev/null 2>&1 ||
  die "$archive does not match the signed release manifest: nothing installed"

mkdir -p "$HOME_DIR" "$BIN_DIR"
# As an absolute path: it is recorded for later updates, which run from another directory.
BIN_DIR=$(cd "$BIN_DIR" && pwd)
top=$(tar -tzf "$tmp/$archive" | head -1 | cut -d/ -f1)
# An unreadable archive lists nothing (sh has no pipefail): never remove the install home itself.
case $top in "" | . | .. | .install.*) die "not a NInferno archive: $archive" ;; esac
# Unpacked and run beside the installed versions first: `current` and the commands move only to a
# release that starts here, so a failed install leaves the one before in place.
# A staging copy an install killed hard left is up to 3 GB: removed first, when its install is gone.
remove_stale_staging
stage=$(mktemp -d "$HOME_DIR/.install.XXXXXX")
printf '%s\n' "$$" > "$stage/$STAGING_MARK"
tar -xzf "$tmp/$archive" -C "$stage" || die "cannot unpack $archive: nothing installed"
if ! version=$("$stage/$top/bin/ninferno-serve" --version 2>&1); then
  printf '%s\n' "$version" >&2
  case $version in
    *GLIBC_*) die "this system's glibc ($glibc) is older than the release needs; nothing installed" ;;
  esac
  die "the release cannot run on this machine (see above), so nothing was installed; a newer NVIDIA driver usually fixes it"
fi
say "$version"
if [ -e "$HOME_DIR/current" ] && [ ! -L "$HOME_DIR/current" ]; then
  die "$HOME_DIR/current is not NInferno's link (a file or directory of its own): move it away or set NINFERNO_HOME; nothing installed"
fi
# The switch is two renames: the same version installed again is moved aside into the staging
# directory (removed on exit) rather than deleted before the new copy is in place, and `current`
# is made under another name and renamed over the old link (GNU mv -T: no window without it).
if [ -e "$HOME_DIR/$top" ] || [ -L "$HOME_DIR/$top" ]; then mv "$HOME_DIR/$top" "$stage/replaced"; fi
# -T: never into a directory another install of this version made meanwhile; that one stands.
mv -T "$stage/$top" "$HOME_DIR/$top" ||
  die "another install of $top finished while this one ran; it stays, nothing switched"
ln -sfn "$HOME_DIR/$top" "$stage/current"
mv -fT "$stage/current" "$HOME_DIR/current"
# The commands (the engines, bin/*-sm<tag>, are started by ninferno-serve itself).
for exe in "$HOME_DIR/current/bin/"*; do
  case $(basename "$exe") in *-sm[0-9]*) continue ;; esac
  ln -sf "$exe" "$BIN_DIR/$(basename "$exe")"
done
printf '%s\n' "$BIN_DIR" > "$BIN_FILE"

say "Installed $top into $HOME_DIR, commands in $BIN_DIR."
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "Add $BIN_DIR to PATH, e.g.: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc" ;; esac
say ""
say "Start NInferno:"
say "  ninferno-ui                  the local page: pick a model, download and run it (opens in your browser)"
say "  ninferno-serve ORG/NAME ...  a model straight from the command line, e.g."
say "      ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters"
