#!/usr/bin/env bash
# signed_install_test.sh REPO: the installer and the updater act only on a release manifest signed
# by the release key, and install only the archive the signed manifest names. A copy of
# tools/release/install.sh carries a key made here (the release key's private half is never in the
# repository); nvidia-smi is a stand-in and the archives are fakes whose ninferno-serve prints a
# version. Needs openssl and python3; skipped (77) without them.
set -euo pipefail
REPO=$(cd "${1:?usage: signed_install_test.sh REPO}" && pwd)
command -v openssl > /dev/null && command -v python3 > /dev/null || exit 77
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
expect() { # what  status-wanted  command...
  local what=$1 want=$2 got=0
  shift 2
  "$@" > "$work/out.txt" 2>&1 || got=$?
  if { [ "$want" = 0 ] && [ "$got" != 0 ]; } || { [ "$want" != 0 ] && [ "$got" = 0 ]; }; then
    echo "FAIL: $what (exit $got)"; sed 's/^/  /' "$work/out.txt"; failures=$((failures + 1))
  fi
}

openssl ecparam -name prime256v1 -genkey -noout -out "$work/key.pem" 2> /dev/null
openssl ec -in "$work/key.pem" -pubout -out "$work/pub.pem" 2> /dev/null
sign() { openssl dgst -sha256 -sign "$work/key.pem" "$1" | base64 -w0 > "$1.sig"; }
# The installer and the updater with the test key in place of the release key.
python3 - "$REPO/tools/release/install.sh" "$work/pub.pem" "$work/install.sh" << 'PY'
import re, sys
text = open(sys.argv[1]).read()
key = open(sys.argv[2]).read().strip()
new, n = re.subn(r"RELEASE_KEY='[^']*'", lambda m: "RELEASE_KEY='" + key + "'", text)
assert n == 1
open(sys.argv[3], "w").write(new)
PY

mkdir -p "$work/bin"
cat > "$work/bin/nvidia-smi" << 'EOF'
#!/bin/sh
echo "0, Test GPU, 8.9, 24564, 580.95.05"
EOF
chmod +x "$work/bin/nvidia-smi"
export PATH="$work/bin:$PATH"

# release VERSION: dist/<VERSION>/ with the archive, its .sha256 and a signed latest.json.
release() {
  local v=$1 top="ninferno-$1-linux-x86_64" stage="$work/stage-$1" dist="$work/dist-$1"
  mkdir -p "$stage/$top/bin" "$stage/$top/share/ninferno" "$dist"
  printf '#!/bin/sh\necho "ninferno %s"\n' "$v" > "$stage/$top/bin/ninferno-serve"
  chmod +x "$stage/$top/bin/ninferno-serve"
  install -m 0755 "$REPO/tools/release/ninferno-update" "$stage/$top/bin/ninferno-update"
  install -m 0644 "$work/install.sh" "$stage/$top/share/ninferno/install.sh"
  printf '{\n "version": "%s"\n}\n' "$v" > "$stage/$top/share/ninferno/release.json"
  tar -czf "$dist/ninferno-linux-x86_64.tar.gz" -C "$stage" "$top"
  (cd "$dist" && sha256sum ninferno-linux-x86_64.tar.gz > ninferno-linux-x86_64.tar.gz.sha256)
  python3 "$REPO/tools/release/manifest.py" --version "$v" \
    --base "https://releases.invalid/v$v" "$dist" > "$dist/latest.json"
  sign "$dist/latest.json"
}
release 0.1.0
release 0.2.0
d1=$work/dist-0.1.0
d2=$work/dist-0.2.0

expect "a signed manifest verifies" 0 sh "$work/install.sh" --verify "$d1/latest.json"
cp "$d1/latest.json" "$work/tampered.json" && cp "$d1/latest.json.sig" "$work/tampered.json.sig"
sed -i 's/"version": "0.1.0"/"version": "9.0.0"/' "$work/tampered.json"
expect "a changed manifest is refused" 1 sh "$work/install.sh" --verify "$work/tampered.json"
cp "$d1/latest.json" "$work/unsigned.json"
expect "a manifest without a signature is refused" 1 sh "$work/install.sh" --verify "$work/unsigned.json"
expect "the release key does not verify a test signature" 1 \
  sh "$REPO/tools/release/install.sh" --verify "$d1/latest.json"

home=$work/home; bin=$work/userbin
I=(env NINFERNO_HOME="$home" NINFERNO_BIN="$bin")  # where every install and update goes
expect "an unsigned manifest installs nothing" 1 "${I[@]}" NINFERNO_MANIFEST="$work/unsigned.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
[ ! -e "$home/current" ] || { echo "FAIL: something was installed from an unsigned manifest"; failures=$((failures + 1)); }
expect "an archive the manifest does not name is refused" 1 "${I[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d2/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
[ ! -e "$home/current" ] || { echo "FAIL: a foreign archive was installed"; failures=$((failures + 1)); }
expect "the signed release installs" 0 "${I[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
[ "$("$bin/ninferno-serve")" = "ninferno 0.1.0" ] || { echo "FAIL: 0.1.0 not installed"; failures=$((failures + 1)); }

# The updater reads the manifest (here a file:// address) and acts on it only when signed.
expect "the updater refuses a changed manifest" 1 "${I[@]}" NINFERNO_UPDATE_URL="file://$work/tampered.json" \
  "$bin/ninferno-update" --yes
expect "the updater installs a signed newer release" 0 "${I[@]}" NINFERNO_UPDATE_URL="file://$d2/latest.json" \
  NINFERNO_ARCHIVE="$d2/ninferno-linux-x86_64.tar.gz" "$bin/ninferno-update" --yes
[ "$("$bin/ninferno-serve")" = "ninferno 0.2.0" ] || { echo "FAIL: 0.2.0 not installed"; failures=$((failures + 1)); }
expect "the updater does not go back to an older release" 0 "${I[@]}" NINFERNO_UPDATE_URL="file://$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" "$bin/ninferno-update" --yes
[ "$("$bin/ninferno-serve")" = "ninferno 0.2.0" ] || { echo "FAIL: went back to 0.1.0"; failures=$((failures + 1)); }

[ "$failures" = 0 ] && echo "signed install: ok" || { echo "$failures failure(s)"; exit 1; }
