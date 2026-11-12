#!/usr/bin/env bash
# signed_install_test.sh REPO: the installer and the updater act only on a release manifest signed
# by the release key, and install only the archive the signed manifest names. A copy of
# tools/release/install.sh carries a key made here (the release key's private half is never in the
# repository); nvidia-smi and getconf are stand-ins and the archives are fakes whose ninferno-serve
# prints a version. Also: a release that does not start is never switched in (install, update,
# rollback), the glibc minimum, the version order (a pre-release below its release, as
# update_check.cpp), --help / --uninstall / unknown arguments, ninferno-kit through its link, and a
# GPU NInferno does not support (a GTX 16 card: install.sh asks, --yes goes on; ninferno-start
# --list says so), known by the ids the engine's GPU facts give its chip.
# Needs openssl and python3; skipped (77) without them.
set -euo pipefail
REPO=$(cd "${1:?usage: signed_install_test.sh REPO}" && pwd)
command -v openssl > /dev/null && command -v python3 > /dev/null || exit 77
work=$(mktemp -d)
alive=""  # a background process standing in for a running install (shared-home case below)
trap 'if [ -n "$alive" ]; then kill "$alive" 2> /dev/null || true; fi; rm -rf "$work"' EXIT
failures=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }
expect() { # what  status-wanted  command...
  local what=$1 want=$2 got=0
  shift 2
  "$@" > "$work/out.txt" 2>&1 < /dev/null || got=$?
  if { [ "$want" = 0 ] && [ "$got" != 0 ]; } || { [ "$want" != 0 ] && [ "$got" = 0 ]; }; then
    fail "$what (exit $got)"; sed 's/^/  /' "$work/out.txt"
  fi
}
said() { grep -qF -- "$1" "$work/out.txt" || { fail "the output does not say '$1'"; sed 's/^/  /' "$work/out.txt"; }; }

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
# One GPU, FAKE_GPU "name|compute_cap|memory MiB|driver|pci.device_id" (default an AD102 board),
# answering the fields --query-gpu asks for, in its order.
cat > "$work/bin/nvidia-smi" << 'EOF'
#!/bin/sh
for a; do case $a in --query-gpu=*) query=${a#--query-gpu=} ;; esac; done
IFS='|'; set -- ${FAKE_GPU:-Test GPU|8.9|24564|580.95.05|0x268410DE}; IFS=' '
out=""
for field in $(printf '%s' "${query:?}" | tr ',' ' '); do
  case $field in
    index) v=0 ;; name) v=$1 ;; compute_cap) v=$2 ;; memory.total) v=$3 ;;
    driver_version) v=$4 ;; pci.device_id) v=$5 ;; *) echo "no field $field" >&2; exit 2 ;;
  esac
  out="${out:+$out, }$v"
done
printf '%s\n' "$out"
EOF
# The C library's version: FAKE_GLIBC, "none" for a system without glibc.
cat > "$work/bin/getconf" << 'EOF'
#!/bin/sh
[ "$1" = GNU_LIBC_VERSION ] || exit 1
[ "${FAKE_GLIBC:-2.39}" = none ] && exit 1
echo "glibc ${FAKE_GLIBC:-2.39}"
EOF
chmod +x "$work/bin/nvidia-smi" "$work/bin/getconf"
export PATH="$work/bin:$PATH"

# release VERSION [BROKEN]: dist-<VERSION>/ with the archive, its .sha256 and a signed latest.json.
# BROKEN: the release's ninferno-serve fails the way a missing driver library (driver) or an old
# glibc (glibc) makes it fail.
release() {
  local v=$1 broken=${2:-} top="ninferno-$1-linux-x86_64" stage="$work/stage-$1" dist="$work/dist-$1"
  mkdir -p "$stage/$top/bin" "$stage/$top/share/ninferno/kit/tools/kit" "$dist"
  case $broken in
    driver) printf '#!/bin/sh\necho "ninferno-serve: libcuda.so.1: cannot open shared object file" >&2\nexit 127\n' ;;
    glibc) printf '#!/bin/sh\necho "ninferno-serve: libc.so.6: version GLIBC_2.40 not found" >&2\nexit 1\n' ;;
    slow) printf '#!/bin/sh\nsleep 2\necho "ninferno %s"\n' "$v" ;;
    *) printf '#!/bin/sh\necho "ninferno %s"\n' "$v" ;;
  esac > "$stage/$top/bin/ninferno-serve"
  chmod +x "$stage/$top/bin/ninferno-serve"
  install -m 0755 "$REPO/tools/release/ninferno-update" "$stage/$top/bin/ninferno-update"
  install -m 0755 "$REPO/tools/release/ninferno-kit" "$stage/$top/bin/ninferno-kit"
  printf 'import sys\nprint("measure", " ".join(sys.argv[1:]))\n' > "$stage/$top/share/ninferno/kit/tools/kit/measure.py"
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
release 0.3.0 driver
release 0.4.0 glibc
release 0.5.0-rc1
release 0.5.0
release 0.6.0 slow
d1=$work/dist-0.1.0
d2=$work/dist-0.2.0
d3=$work/dist-0.3.0
d4=$work/dist-0.4.0
drc=$work/dist-0.5.0-rc1
d5=$work/dist-0.5.0
d6=$work/dist-0.6.0

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
serves() { # what  version-wanted: the linked ninferno-serve runs that version
  local got
  got=$("$bin/ninferno-serve" 2>&1 || true)
  [ "$got" = "ninferno $2" ] || fail "$1: ninferno-serve says '$got', not 'ninferno $2'"
}

# Arguments: only the documented ones; --help and an unknown one install nothing.
expect "--help" 0 "${I[@]}" sh "$work/install.sh" --help
said "--uninstall"
expect "an unknown argument is refused" 1 "${I[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh" --bogus
said "unknown argument --bogus"
expect "an extra argument is refused" 1 "${I[@]}" sh "$work/install.sh" --help --bogus
[ ! -e "$home" ] || fail "--help or an unknown argument installed something"

# glibc: refused before anything is downloaded.
expect "an old glibc is refused" 1 "${I[@]}" FAKE_GLIBC=2.35 NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
said "glibc 2.35 is too old: the release needs glibc 2.39"
expect "a system without glibc is refused" 1 "${I[@]}" FAKE_GLIBC=none NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
said "not glibc"
[ ! -e "$home/current" ] || fail "something was installed on an old glibc"

expect "an unsigned manifest installs nothing" 1 "${I[@]}" NINFERNO_MANIFEST="$work/unsigned.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
[ ! -e "$home/current" ] || fail "something was installed from an unsigned manifest"
expect "an archive the manifest does not name is refused" 1 "${I[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d2/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
[ ! -e "$home/current" ] || fail "a foreign archive was installed"
expect "a release that does not start is not installed" 1 "${I[@]}" NINFERNO_MANIFEST="$d3/latest.json" \
  NINFERNO_ARCHIVE="$d3/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
[ ! -e "$home/current" ] && [ ! -e "$bin/ninferno-serve" ] || fail "a release that does not start was switched in"
# A killed install's leftover: marked with the PID of an install that is gone.
sh -c 'exit 0' & dead=$!; wait "$dead"
mkdir -p "$home/.install.stale" && echo x > "$home/.install.stale/big" && echo "$dead" > "$home/.install.stale/.ninferno-staging"
expect "the signed release installs (glibc 2.39 is enough)" 0 "${I[@]}" FAKE_GLIBC=2.39 NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
serves "install 0.1.0" 0.1.0

# A failed install leaves the one before in place, links included.
expect "a release that misses the driver is not switched in" 1 "${I[@]}" NINFERNO_MANIFEST="$d3/latest.json" \
  NINFERNO_ARCHIVE="$d3/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
said "nothing was installed"
serves "after the failed 0.3.0" 0.1.0
expect "a release that needs a newer glibc is not switched in" 1 "${I[@]}" NINFERNO_MANIFEST="$d4/latest.json" \
  NINFERNO_ARCHIVE="$d4/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
said "glibc (2.39) is older than the release needs"
serves "after the failed 0.4.0" 0.1.0
[ ! -e "$home/ninferno-0.3.0-linux-x86_64" ] && [ ! -e "$home/ninferno-0.4.0-linux-x86_64" ] ||
  fail "a failed release was left in $home"
[ -z "$(find "$home" -maxdepth 1 -name '.install.*')" ] ||
  fail "a staging directory was left in $home (or an old one not removed)"

# An install stopped by a signal removes its staging copy (dash runs no EXIT trap on a signal).
"${I[@]}" NINFERNO_MANIFEST="$d6/latest.json" NINFERNO_ARCHIVE="$d6/ninferno-linux-x86_64.tar.gz" \
  sh "$work/install.sh" > "$work/killed.txt" 2>&1 < /dev/null &
pid=$!
for _ in $(seq 100); do
  [ -n "$(find "$home" -maxdepth 1 -name '.install.*')" ] && break
  sleep 0.1
done
kill -TERM "$pid"
got=0; wait "$pid" || got=$?
[ "$got" = 143 ] || { fail "a stopped install exits $got, not 143"; sed 's/^/  /' "$work/killed.txt"; }
[ -z "$(find "$home" -maxdepth 1 -name '.install.*')" ] || fail "a stopped install left its staging copy"
serves "after the stopped install" 0.1.0

# ninferno-kit through its link in the bin directory finds the kit beside the real file.
expect "ninferno-kit through its link" 0 "${I[@]}" "$bin/ninferno-kit" --budget-minutes 5
said "measure --bin $home/ninferno-0.1.0-linux-x86_64/bin --budget-minutes 5"

# The updater reads the manifest (here a file:// address) and acts on it only when signed.
expect "the updater refuses a changed manifest" 1 "${I[@]}" NINFERNO_UPDATE_URL="file://$work/tampered.json" \
  "$bin/ninferno-update" --yes
expect "the updater installs a signed newer release" 0 "${I[@]}" NINFERNO_UPDATE_URL="file://$d2/latest.json" \
  NINFERNO_ARCHIVE="$d2/ninferno-linux-x86_64.tar.gz" "$bin/ninferno-update" --yes
serves "update to 0.2.0" 0.2.0
expect "the updater does not go back to an older release" 0 "${I[@]}" NINFERNO_UPDATE_URL="file://$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" "$bin/ninferno-update" --yes
serves "update offered 0.1.0" 0.2.0
expect "the updater does not switch in a release that does not start" 1 "${I[@]}" \
  NINFERNO_UPDATE_URL="file://$d3/latest.json" NINFERNO_ARCHIVE="$d3/ninferno-linux-x86_64.tar.gz" \
  "$bin/ninferno-update" --yes
serves "after the failed update to 0.3.0" 0.2.0
expect "an unknown updater argument is refused" 1 "${I[@]}" "$bin/ninferno-update" --bogus

# Rollback goes back only to a version that starts.
old_serve=$home/ninferno-0.1.0-linux-x86_64/bin/ninferno-serve
cp "$old_serve" "$work/serve-0.1.0"
printf '#!/bin/sh\nexit 127\n' > "$old_serve"
expect "rollback to a version that does not start is refused" 1 "${I[@]}" "$bin/ninferno-update" --rollback
serves "after the refused rollback" 0.2.0
cp "$work/serve-0.1.0" "$old_serve"
expect "rollback" 0 "${I[@]}" "$bin/ninferno-update" --rollback
serves "after the rollback" 0.1.0
# Asked with no terminal (a new session has none): it says so and installs nothing.
if command -v setsid > /dev/null; then
  expect "the updater with no terminal to ask on" 0 "${I[@]}" NINFERNO_UPDATE_URL="file://$d2/latest.json" \
    NINFERNO_ARCHIVE="$d2/ninferno-linux-x86_64.tar.gz" setsid -w "$bin/ninferno-update"
  said "No terminal to ask on: not installed"
  ! grep -q "/dev/tty" "$work/out.txt" || fail "the updater printed the shell's /dev/tty error"
  serves "after the update with no terminal" 0.1.0
fi
# The same version again: switched by renames, `current` stays a link.
expect "0.1.0 installs over itself" 0 "${I[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
serves "after installing 0.1.0 again" 0.1.0
[ -L "$home/current" ] && [ "$(readlink "$home/current")" = "$home/ninferno-0.1.0-linux-x86_64" ] ||
  fail "current is not the link to 0.1.0"
[ -z "$(find "$home" -maxdepth 1 -name '.install.*')" ] || fail "the reinstall left its staging copy"
# Stopped between its two renames (the old copy moved aside, the new one not yet in place): the
# old copy goes back. A stand-in mv sends the install TERM at that rename.
mkdir -p "$work/mvkill"
printf '#!/bin/sh\nfor a; do case $a in */.install.*/ninferno-*) kill -TERM $PPID; sleep 1; exit 1 ;; esac; done\nexec %s "$@"\n' \
  "$(command -v mv)" > "$work/mvkill/mv"
chmod +x "$work/mvkill/mv"
expect "a reinstall stopped between its renames" 1 "${I[@]}" PATH="$work/mvkill:$PATH" \
  NINFERNO_MANIFEST="$d1/latest.json" NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
serves "after the reinstall stopped between its renames" 0.1.0
[ -z "$(find "$home" -maxdepth 1 -name '.install.*')" ] || fail "the stopped reinstall left its staging copy"
# Another install of the same version finishes between this one's two renames: this one stops and
# leaves that copy as it is, never moving its own copy inside it.
mkdir -p "$work/mvrace"
printf '#!/bin/sh\nfor a; do case $a in */.install.*/ninferno-*) mkdir -p "%s" && : > "%s/other";; esac; done\nexec %s "$@"\n' \
  "$home/ninferno-0.1.0-linux-x86_64" "$home/ninferno-0.1.0-linux-x86_64" "$(command -v mv)" > "$work/mvrace/mv"
chmod +x "$work/mvrace/mv"
expect "a reinstall raced by another of the same version" 1 "${I[@]}" PATH="$work/mvrace:$PATH" \
  NINFERNO_MANIFEST="$d1/latest.json" NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
said "another install of ninferno-0.1.0-linux-x86_64"
[ -e "$home/ninferno-0.1.0-linux-x86_64/other" ] || fail "the raced reinstall replaced the other copy"
[ ! -e "$home/ninferno-0.1.0-linux-x86_64/ninferno-0.1.0-linux-x86_64" ] ||
  fail "the raced reinstall moved its copy inside the other one"
rm -rf "$home/ninferno-0.1.0-linux-x86_64"
expect "0.1.0 installs again after the race" 0 "${I[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
serves "after the raced reinstall" 0.1.0

# The version order is update_check.cpp's (tests/test_update_check.cpp has the same cases).
eval "$(sed -n '/^version_newer() {$/,/^}$/p' "$REPO/tools/release/ninferno-update")"
order() { # a b want: want is ">" "<" "=" or "bad" (a is no release version)
  local got=0 back=0 want
  version_newer "$1" "$2" || got=$?
  version_newer "$2" "$1" || back=$?
  case $got$back in 01) want=">" ;; 10) want="<" ;; 11) want="=" ;; 2*) want=bad ;; *) want="?$got$back" ;; esac
  [ "$want" = "$3" ] || fail "version order: $1 vs $2 is '$want', expected '$3'"
}
order 0.2.0 0.1.9 ">"
order 0.10.0 0.9.0 ">"
order v0.2.0 0.2.0 "="
order 0.2 0.2.0 "="
order 0.2.0-rc1 0.2.0 "<"
order 0.1.0-rc1 0.1.0 "<"
order 0.2.0-rc2 0.2.0-rc10 "<"
order 0.2.0-beta 0.2.0-rc1 "<"
order 0.2.0-rc2 0.2.0-rc1 ">"
order 0.2.0-rc1 0.1.5 ">"
order 0.2.0-rc.1 0.2.0-rc.1.1 "<"
order 0.1.0-10000000000 0.1.0-9999999999 "<"  # 10+ digits compare as text, as in C++
order dev 0.1.0 bad
order 0.1.0-12-g5c50953 0.1.0 bad
order 1 0.1.0 bad
version_newer 0.1.0 0.1.0-3-gabcdef0 || fail "a build between releases does not update to a release"

# rc to final through the updater, and never back to the rc.
home2=$work/home-rc; bin2=$work/userbin-rc
J=(env NINFERNO_HOME="$home2" NINFERNO_BIN="$bin2")
expect "the rc installs" 0 "${J[@]}" NINFERNO_MANIFEST="$drc/latest.json" \
  NINFERNO_ARCHIVE="$drc/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
expect "the updater sees the final release" 0 "${J[@]}" NINFERNO_UPDATE_URL="file://$d5/latest.json" \
  "$bin2/ninferno-update" --check
expect "the updater installs the final release over its rc" 0 "${J[@]}" NINFERNO_UPDATE_URL="file://$d5/latest.json" \
  NINFERNO_ARCHIVE="$d5/ninferno-linux-x86_64.tar.gz" "$bin2/ninferno-update" --yes
[ "$("$bin2/ninferno-serve")" = "ninferno 0.5.0" ] || fail "rc to final: 0.5.0 not installed"
expect "the final release sees no update in its rc" 1 "${J[@]}" NINFERNO_UPDATE_URL="file://$drc/latest.json" \
  "$bin2/ninferno-update" --check

# A custom commands directory given once: updates, rollbacks and the uninstall keep using it (the
# home records it), with HOME's default left empty. Rollback goes to the newest kept version older
# than this one, whatever the directories' times (a release directory's time is its build's).
home7=$work/home-bin; bin7=$work/custom-bin; fakehome=$work/fake-home
mkdir -p "$fakehome"
H=(env HOME="$fakehome" NINFERNO_HOME="$home7")
# Given relative (from $work): recorded as the absolute path.
expect "0.1.0 into a custom commands directory" 0 "${H[@]}" NINFERNO_BIN="custom-bin" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh -c 'cd "$1" && sh "$1/install.sh"' sh "$work"
[ "$(cat "$home7/.bin-dir")" = "$bin7" ] || fail "the recorded commands directory is '$(cat "$home7/.bin-dir")', not $bin7"
for d in "$d2" "$d5"; do
  expect "$(basename "$d") over it without NINFERNO_BIN" 0 "${H[@]}" NINFERNO_MANIFEST="$d/latest.json" \
    NINFERNO_ARCHIVE="$d/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
done
[ "$("$bin7/ninferno-serve")" = "ninferno 0.5.0" ] || fail "the custom commands directory does not run 0.5.0"
[ ! -e "$fakehome/.local/bin" ] || fail "an install without NINFERNO_BIN linked into HOME's default"
touch -d '2000-01-01' "$home7/ninferno-0.2.0-linux-x86_64" "$home7/ninferno-0.5.0-linux-x86_64"
touch "$home7/ninferno-0.1.0-linux-x86_64"
# A kept directory with no version is neither a rollback target nor removed.
mkdir -p "$home7/ninferno-x-linux-x86_64/share/ninferno"
echo '{}' > "$home7/ninferno-x-linux-x86_64/share/ninferno/release.json"
# The commands directory was removed meanwhile: rollback makes it again.
rm -rf "$bin7"
expect "rollback by version" 0 "${H[@]}" "$home7/current/bin/ninferno-update" --rollback
[ "$("$bin7/ninferno-serve")" = "ninferno 0.2.0" ] || fail "rollback from 0.5.0 did not go to 0.2.0"
expect "an update after the rollback" 0 "${H[@]}" NINFERNO_UPDATE_URL="file://$d6/latest.json" \
  NINFERNO_ARCHIVE="$d6/ninferno-linux-x86_64.tar.gz" "$bin7/ninferno-update" --yes
[ "$("$bin7/ninferno-serve")" = "ninferno 0.6.0" ] || fail "the update after the rollback did not link 0.6.0"
kept=$(cd "$home7" && ls -d ninferno-*-linux-x86_64 | grep -v '^ninferno-x-' | tr '\n' ' ')
[ -d "$home7/ninferno-x-linux-x86_64" ] || fail "the update removed a kept directory with no version"
rm -rf "$home7/ninferno-x-linux-x86_64"
# 0.5.0 for --rollback, 0.2.0 the update ran from; 0.1.0 goes.
[ "$kept" = "ninferno-0.2.0-linux-x86_64 ninferno-0.5.0-linux-x86_64 ninferno-0.6.0-linux-x86_64 " ] ||
  fail "after the update the home keeps '$kept', not 0.2.0, 0.5.0 and 0.6.0"
[ ! -e "$fakehome/.local/bin" ] || fail "an update or rollback linked into HOME's default"
expect "uninstall from the recorded commands directory" 0 "${H[@]}" sh "$work/install.sh" --uninstall --yes
[ -z "$(ls -A "$bin7")" ] || fail "the uninstall left links in the custom commands directory: $(ls "$bin7")"
[ ! -e "$home7" ] || fail "the uninstall left the home: $(ls -A "$home7")"

# --check offers only what --yes installs: a signed manifest with no https address for the
# archive is refused by both (a rehearsal without TLS gives the archive as a file).
dh=$work/dist-http
mkdir -p "$dh"
cp "$d6/ninferno-linux-x86_64.tar.gz" "$d6/ninferno-linux-x86_64.tar.gz.sha256" "$dh/"
python3 "$REPO/tools/release/manifest.py" --version 0.6.0 --base "http://releases.invalid/v0.6.0" "$dh" > "$dh/latest.json"
sign "$dh/latest.json"
expect "--check refuses an archive without https" 1 "${J[@]}" NINFERNO_UPDATE_URL="file://$dh/latest.json" \
  "$bin2/ninferno-update" --check
said "gives no https address for ninferno-linux-x86_64.tar.gz"
! grep -q "is available" "$work/out.txt" || fail "--check offered a release --yes refuses"
expect "--check with the archive given as a file" 0 "${J[@]}" NINFERNO_UPDATE_URL="file://$dh/latest.json" \
  NINFERNO_ARCHIVE="$d6/ninferno-linux-x86_64.tar.gz" "$bin2/ninferno-update" --check

# sign.sh says why it does not sign, without OpenSSL's raw output.
cp "$d1/latest.json" "$work/to-sign.json"
expect "sign.sh with a key that is not the release key" 1 bash "$REPO/tools/release/sign.sh" \
  "$work/to-sign.json" "$work/key.pem"
said "is not the release key"
[ "$(wc -l < "$work/out.txt")" = 1 ] || { fail "sign.sh printed more than its own line"; sed 's/^/  /' "$work/out.txt"; }
[ ! -e "$work/to-sign.json.sig" ] || fail "sign.sh wrote a signature with the wrong key"
echo "not a key" > "$work/not-a-key.pem"
expect "sign.sh with a file that is no key" 1 bash "$REPO/tools/release/sign.sh" "$work/to-sign.json" "$work/not-a-key.pem"
said "sign.sh: cannot sign"
[ "$(wc -l < "$work/out.txt")" = 1 ] || { fail "sign.sh printed more than its own line"; sed 's/^/  /' "$work/out.txt"; }

# Uninstall: what install.sh placed goes, nothing else; it asks unless --yes.
ln -s /bin/true "$bin/not-ninferno"
echo keep > "$bin/notes.txt"
# A custom home such as ~/.local holds other programs: a pipx-like tool with its link in the bin
# directory stays, and so does the home itself.
mkdir -p "$home/share/pipx/venvs/black/bin"
printf '#!/bin/sh\n' > "$home/share/pipx/venvs/black/bin/black"
ln -s "$home/share/pipx/venvs/black/bin/black" "$bin/black"
if command -v setsid > /dev/null; then
  expect "uninstall without a terminal and without --yes" 1 "${I[@]}" setsid -w sh "$work/install.sh" --uninstall
  said "--uninstall --yes"
  [ -e "$home/current" ] || fail "uninstall without an answer removed something"
fi
expect "uninstall with an argument it does not know" 1 "${I[@]}" sh "$work/install.sh" --uninstall --bogus
expect "uninstall --yes" 0 "${I[@]}" sh "$work/install.sh" --uninstall --yes
said "$home stays"
! grep -q rmdir "$work/out.txt" || fail "uninstall printed rmdir's complaint"
[ "$(ls -A "$home")" = share ] || fail "uninstall left or removed the wrong things in $home: $(ls -A "$home")"
for f in "$bin"/ninferno-*; do [ ! -e "$f" ] && [ ! -L "$f" ] || fail "uninstall left $f"; done
[ -L "$bin/not-ninferno" ] && [ -f "$bin/notes.txt" ] || fail "uninstall removed a file it did not install"
[ -L "$bin/black" ] && [ -f "$home/share/pipx/venvs/black/bin/black" ] ||
  fail "uninstall removed a foreign link into the home or its target"
rm -rf "$home"
expect "uninstall when nothing is installed" 1 "${I[@]}" sh "$work/install.sh" --uninstall --yes

# A home such as ~ with a directory of the user's own named current: kept.
home4=$work/home-user
mkdir -p "$home4/current" && echo mine > "$home4/current/notes"
expect "uninstall beside a real current directory" 0 env NINFERNO_HOME="$home4" NINFERNO_BIN="$work/bin4" \
  sh "$work/install.sh" --uninstall --yes
[ -f "$home4/current/notes" ] || fail "uninstall removed a current directory it did not make"
expect "an install beside a real current directory stops plainly" 1 env NINFERNO_HOME="$home4" \
  NINFERNO_BIN="$work/bin4" NINFERNO_MANIFEST="$d1/latest.json" NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" \
  sh "$work/install.sh"
said "current is not NInferno's link"
! grep -q "^mv:" "$work/out.txt" || fail "the install stopped on mv's own error"
[ -f "$home4/current/notes" ] && [ ! -e "$home4/ninferno-0.1.0-linux-x86_64" ] ||
  fail "an install beside a real current directory changed something"

# Staging directories in a shared home: only marked ones whose install is gone are removed.
home5=$work/home-shared; bin5=$work/userbin-shared
L=(env NINFERNO_HOME="$home5" NINFERNO_BIN="$bin5")
sleep 60 & alive=$!
sh -c 'exit 0' & dead=$!; wait "$dead"
staging() { # name mark [age]: a .install.<name> directory with that mark ("" = no mark file)
  mkdir -p "$home5/.install.$1" && echo data > "$home5/.install.$1/big"
  if [ -n "$2" ]; then
    printf '%s\n' "$2" > "$home5/.install.$1/.ninferno-staging"
    [ -z "${3:-}" ] || touch -d "$3" "$home5/.install.$1/.ninferno-staging"
  fi
}
make_staging() {
  staging user ""                     # the user's own: no mark
  staging live "$alive"               # another install, still running
  staging dead "$dead"                # an install that is gone
  staging nopid-new x                 # a mark without a PID, recent
  staging nopid-old x '7 hours ago'   # ... and old
  staging reused "$alive" '2 days ago'  # a PID used again
}
kept() { # what  names...: exactly these .install.* stay
  local what=$1 got want
  shift
  got=$(cd "$home5" && { ls -1d .install.* 2> /dev/null || true; } | sort | tr '\n' ' ')
  want=$(printf '.install.%s\n' "$@" | sort | tr '\n' ' ')
  [ "$got" = "$want" ] || fail "$what: staging left '$got', expected '$want'"
}
make_staging
expect "install in a shared home" 0 "${L[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
kept "install" user live nopid-new
make_staging
expect "uninstall in a shared home" 0 "${L[@]}" sh "$work/install.sh" --uninstall --yes
kept "uninstall" user live nopid-new
said "$home5 stays"
kill "$alive" 2> /dev/null || true; alive=""

# Links through the resolved home (ninferno-update passes it) go too.
realh=$work/home-real; homel=$work/home-link; bin3=$work/userbin-link
mkdir -p "$realh" && ln -s "$realh" "$homel"
K=(env NINFERNO_HOME="$homel" NINFERNO_BIN="$bin3")
expect "install through a linked home" 0 "${K[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh"
ln -sf "$realh/current/bin/ninferno-serve" "$bin3/ninferno-serve"
ln -s "$realh/ninferno-0.1.0-linux-x86_64/bin/ninferno-update" "$bin3/ninferno-update-old"
expect "uninstall through a linked home" 0 "${K[@]}" sh "$work/install.sh" --uninstall --yes
[ -z "$(ls -A "$bin3")" ] || fail "uninstall left links through the resolved home: $(ls -A "$bin3")"
[ -z "$(ls -A "$realh")" ] || fail "uninstall left $realh: $(ls -A "$realh")"

# A GPU without tensor cores is known by its chip's facts: install.sh's ids against the engine's
# (the TU116 / TU117 PCI blocks of core/device_class_host.cpp at 7.5, and every board row of
# core/data/gpu_facts.json from 7.0 on, by its mma_slow), the blocks' edges included.
python3 - "$REPO/src/core/device_class_host.cpp" "$REPO/src/core/data/gpu_facts.json" > "$work/gpu_cases.txt" << 'PY'
import json, re, sys
blocks = re.findall(r'\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+), "(tu11[67])"\}', open(sys.argv[1]).read())
assert len(blocks) == 2, blocks
rows = {int(r["id"], 16): r for r in json.load(open(sys.argv[2]))["gpus"]}
def case(dev, cc, slow): print(f"0x{dev:04X}10DE {cc // 10}.{cc % 10} {'no' if slow else 'yes'}")
for first, last, _ in blocks:
    first, last = int(first, 16), int(last, 16)
    for dev in (first, (first + last) // 2, last):
        case(dev, 75, True)
        case(dev, 86, False)  # a block names its chip only at that chip's compute capability
    for dev in (first - 1, last + 1):
        case(dev, 75, rows.get(dev, {}).get("mma_slow", False))
for dev, r in rows.items():
    if r["cc"] >= 70:
        case(dev, r["cc"], r.get("mma_slow", False))
PY
while read -r pci cap want; do
  got=yes; sh "$work/install.sh" --gpu-check "$pci" "$cap" > "$work/out.txt" 2>&1 || got=no
  [ "$got" = "$want" ] || fail "--gpu-check $pci $cap: supported '$got', the GPU facts say '$want'"
done < "$work/gpu_cases.txt"
[ "$(wc -l < "$work/gpu_cases.txt")" -gt 100 ] || fail "the GPU facts gave only $(wc -l < "$work/gpu_cases.txt") cases"
expect "--gpu-check on a TU116 board" 1 sh "$work/install.sh" --gpu-check 0x218210de 7.5
said "no tensor cores"
expect "--gpu-check on a Pascal board" 1 sh "$work/install.sh" --gpu-check 0x1B0610DE 6.1
said "older than NInferno supports"

# A GTX 16 card (a TU116 board): the installer warns and asks; with no terminal it refuses unless
# --yes, which installs; ninferno-update goes on without asking (the card was accepted).
gtx16="GeForce GTX 1660 Ti|7.5|6144|580.95.05|0x218210DE"
home6=$work/home-gtx16; bin6=$work/userbin-gtx16
G=(env FAKE_GPU="$gtx16" NINFERNO_HOME="$home6" NINFERNO_BIN="$bin6")
if command -v setsid > /dev/null; then
  expect "a GTX 16 card without a terminal and without --yes" 1 "${G[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
    NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" setsid -w sh "$work/install.sh"
  said "NOT SUPPORTED, its chip has no tensor cores (the GTX 16 series: TU116, TU117)"
  said "run it with --yes"
  [ ! -e "$home6/current" ] || fail "a GTX 16 card was installed for without an answer"
fi
expect "a GTX 16 card with --yes" 0 "${G[@]}" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" sh "$work/install.sh" --yes
said "WARNING: NInferno does not support a GPU without tensor cores"
[ "$("$bin6/ninferno-serve")" = "ninferno 0.1.0" ] || fail "--yes on a GTX 16 card did not install 0.1.0"
expect "--yes takes nothing after it" 1 "${G[@]}" sh "$work/install.sh" --yes --bogus
if command -v setsid > /dev/null; then
  expect "the updater on a GTX 16 card" 0 "${G[@]}" NINFERNO_UPDATE_URL="file://$d2/latest.json" \
    NINFERNO_ARCHIVE="$d2/ninferno-linux-x86_64.tar.gz" setsid -w "$bin6/ninferno-update" --yes
  [ "$("$bin6/ninferno-serve")" = "ninferno 0.2.0" ] || fail "the updater on a GTX 16 card did not install 0.2.0"
fi
# A Turing chip with tensor cores (TU102) installs without asking.
detach=(); if command -v setsid > /dev/null; then detach=(setsid -w); fi
expect "an RTX 2080 Ti" 0 env FAKE_GPU="GeForce RTX 2080 Ti|7.5|11264|580.95.05|0x1E0710DE" \
  NINFERNO_HOME="$work/home-rtx" NINFERNO_BIN="$work/userbin-rtx" NINFERNO_MANIFEST="$d1/latest.json" \
  NINFERNO_ARCHIVE="$d1/ninferno-linux-x86_64.tar.gz" "${detach[@]}" sh "$work/install.sh"
! grep -q "NOT SUPPORTED\|WARNING" "$work/out.txt" || fail "an RTX 2080 Ti was warned about"

# ninferno-start --list: an unsupported GPU is said to be one (with the reason), not given the
# numbers of a measured card of its architecture.
start=$work/start-root
mkdir -p "$start/bin" "$start/share/ninferno"
install -m 0755 "$REPO/tools/release/ninferno-start" "$start/bin/ninferno-start"
install -m 0644 "$work/install.sh" "$start/share/ninferno/install.sh"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  rtx2080ti "RTX 2080 Ti" "GeForce RTX 2080 Ti" 75 11 "Qwen3 8B" 100 300 8 "ninferno-serve m" - \
  rtx4090 "RTX 4090" "Test GPU" 89 24 "Qwen3 32B" 50 200 8 "ninferno-serve m" - > "$start/share/ninferno/recommend.tsv"
expect "ninferno-start --list on a GTX 16 card" 1 env FAKE_GPU="$gtx16" "$start/bin/ninferno-start" --list
said "GeForce GTX 1660 Ti is not supported: its chip has no tensor cores"
expect "ninferno-start --list on a Pascal card" 1 env FAKE_GPU="GeForce GTX 1080 Ti|6.1|11264|580.95.05|0x1B0610DE" \
  "$start/bin/ninferno-start" --list
said "is not supported: compute capability 6.1 is older"
expect "ninferno-start --list on a supported card" 0 "$start/bin/ninferno-start" --list
said "Qwen3 32B"

[ "$failures" = 0 ] && echo "signed install: ok" || { echo "$failures failure(s)"; exit 1; }
