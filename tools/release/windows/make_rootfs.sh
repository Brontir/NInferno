#!/usr/bin/env bash
# make_rootfs.sh [OUT_DIR]: the root file system of NInferno's own WSL 2 distribution
# (ninferno-wsl-rootfs.tar.gz + .sha256), which NInferno-Setup.exe imports as "NInferno".
#
# Ubuntu's minimal base image plus what the release archive does not carry: CA certificates (the
# engine downloads models over https) and curl (ninferno-update). Root is the default user: the
# distribution exists only to run the engine, nobody logs into it. Needs root (chroot) and network.
#
# UBUNTU must not be newer than the distribution the release archives are built on: their binaries
# need its C library or an older one (package.sh).
set -euo pipefail
OUT=$(realpath -m "${1:-dist}")
UBUNTU=${UBUNTU:-24.04}
[ "$(id -u)" = 0 ] || { echo "make_rootfs.sh: run as root (it uses chroot)" >&2; exit 1; }
WORK=$(mktemp -d)
ROOT=$WORK/root
cleanup() {
  for m in proc sys dev; do umount -l "$ROOT/$m" 2>/dev/null || true; done
  rm -rf --one-file-system "$WORK"
}
trap cleanup EXIT

release="https://cdimage.ubuntu.com/ubuntu-base/releases/$UBUNTU/release"
curl -fsSL -o "$WORK/SHA256SUMS" "$release/SHA256SUMS"
# The directory carries the newest point release only (24.04.3, then .4, ...).
base=$(grep -o "ubuntu-base-$UBUNTU[.0-9]*-base-amd64\.tar\.gz" "$WORK/SHA256SUMS" | head -n 1 || true)
[ -n "$base" ] || { echo "make_rootfs.sh: no amd64 base image in $release/SHA256SUMS" >&2; exit 1; }
curl -fL --progress-bar -o "$WORK/$base" "$release/$base"
( cd "$WORK" && grep " \*\?$base\$" SHA256SUMS | sed 's/ \*/  /' | sha256sum -c - )
mkdir -p "$ROOT"
tar -xzf "$WORK/$base" -C "$ROOT"

cp /etc/resolv.conf "$ROOT/etc/resolv.conf"
mount -t proc proc "$ROOT/proc"
mount --rbind /sys "$ROOT/sys"
mount --rbind /dev "$ROOT/dev"
chroot "$ROOT" /bin/sh -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q --no-install-recommends ca-certificates curl
  apt-get clean
  rm -rf /var/lib/apt/lists/* /var/log/* /tmp/* /root/.cache
'
for m in proc sys dev; do umount -l "$ROOT/$m"; done
: > "$ROOT/etc/resolv.conf" # WSL writes its own

cat > "$ROOT/etc/wsl.conf" <<'EOF'
# NInferno's distribution: it runs the engine for NInferno.exe and is not meant for logging in.
[user]
default = root

[boot]
systemd = false

[automount]
root = /mnt/

[interop]
enabled = true
# The engine needs none of Windows' PATH; leaving it out keeps command lookup fast.
appendWindowsPath = false
EOF
# WSL's GPU libraries and nvidia-smi.
echo 'export PATH=/usr/lib/wsl/lib:$PATH' > "$ROOT/etc/profile.d/wsl-gpu.sh"
echo "NInferno engine distribution (Ubuntu $POINT base)" > "$ROOT/etc/ninferno-distribution"

mkdir -p "$OUT"
tar -C "$ROOT" --numeric-owner -czf "$OUT/ninferno-wsl-rootfs.tar.gz" .
( cd "$OUT" && sha256sum ninferno-wsl-rootfs.tar.gz > ninferno-wsl-rootfs.tar.gz.sha256 )
du -h "$OUT/ninferno-wsl-rootfs.tar.gz"
