#!/usr/bin/env bash
# package_checks_test.sh REPO: the checks tools/release/package.sh makes before it builds, and the
# live kernel check's reading of an engine's exit. Nothing is built: getconf, patchelf and the
# engines are stand-ins, and nvcc is pointed at a path that does not exist, so every run stops
# at the latest at the nvcc check.
#   - 70 in NINFERNO_ARCHS without CUDA 12.8 stops the run (unless a resumed run holds the engine);
#     leaving 70 out is the way to build without Volta.
#   - A build machine whose glibc is newer than install.sh's GLIBC_MIN stops the run.
#   - `selector-facts` exiting 127 because libcuda.so.1 is missing (no NVIDIA driver) counts as
#     "no GPU", like exit 2; any other failure stops the run.
set -euo pipefail
REPO=$(cd "${1:?usage: package_checks_test.sh REPO}" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
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
said() { grep -qF -- "$1" "$work/out.txt" || { fail "$2: the output does not say '$1'"; sed 's/^/  /' "$work/out.txt"; }; }
not_said() { ! grep -qF -- "$1" "$work/out.txt" || { fail "$2: the output says '$1'"; sed 's/^/  /' "$work/out.txt"; }; }

mkdir -p "$work/bin"
cat > "$work/bin/getconf" << 'EOF'
#!/bin/sh
echo "glibc ${FAKE_GLIBC:-2.38}"
EOF
printf '#!/bin/sh\nexit 0\n' > "$work/bin/patchelf"
chmod +x "$work/bin/getconf" "$work/bin/patchelf"
P=(env -u NINFERNO_ARCHS -u NINFERNO_PACKAGE_RESUME PATH="$work/bin:$PATH" CUDACXX="$work/no-nvcc" NINFERNO_CUDA70="$work/no-cuda-12.8" NINFERNO_VERSION=9.9.9)

expect "Volta without CUDA 12.8 stops" 1 "${P[@]}" NINFERNO_ARCHS="89 70" bash "$REPO/tools/release/package.sh" "$work/dist"
said "leave 70 out of NINFERNO_ARCHS" "Volta without CUDA 12.8"
expect "the default list holds Volta" 1 "${P[@]}" bash "$REPO/tools/release/package.sh" "$work/dist"
said "no CUDA 12.8 for the Volta engine" "the default list"
expect "Volta left out goes on" 1 "${P[@]}" NINFERNO_ARCHS="89" bash "$REPO/tools/release/package.sh" "$work/dist"
not_said "Volta" "Volta left out"
said "nvcc not found" "Volta left out"
# A resumed run goes on only where the build loop resumes too: the engine and the dispatcher staged.
staged=$work/dist/ninferno-9.9.9-linux-x86_64/bin
mkdir -p "$staged"
printf '#!/bin/sh\n' > "$staged/ninferno-serve-sm70"
chmod +x "$staged/ninferno-serve-sm70"
expect "a resumed run with the Volta engine but no dispatcher stops" 1 "${P[@]}" NINFERNO_ARCHS="89 70" \
  NINFERNO_PACKAGE_RESUME=1 bash "$REPO/tools/release/package.sh" "$work/dist"
said "no CUDA 12.8 for the Volta engine" "resumed without the dispatcher"
printf '#!/bin/sh\n' > "$staged/ninferno-serve.dispatch"
expect "a resumed run that holds the Volta engine goes on" 1 "${P[@]}" NINFERNO_ARCHS="89 70" NINFERNO_PACKAGE_RESUME=1 \
  bash "$REPO/tools/release/package.sh" "$work/dist"
not_said "Volta" "resumed with the Volta engine"
said "nvcc not found" "resumed with the Volta engine"

expect "a newer glibc than install.sh checks stops" 1 "${P[@]}" FAKE_GLIBC=2.41 NINFERNO_ARCHS="89" \
  bash "$REPO/tools/release/package.sh" "$work/dist"
said "glibc 2.41, newer than the 2.38" "a newer glibc"
expect "an older glibc goes on" 1 "${P[@]}" FAKE_GLIBC=2.36 NINFERNO_ARCHS="89" \
  bash "$REPO/tools/release/package.sh" "$work/dist"
said "nvcc not found" "an older glibc"

# The live check's reading of the engine's exit, on stand-in engines.
fn=$(sed -n '/^check_selector_facts() {$/,/^}$/p' "$REPO/tools/release/package.sh")
[ -n "$fn" ] || fail "package.sh has no check_selector_facts"
engine() { # name status stderr-text
  printf '#!/bin/sh\necho "%s" >&2\nexit %s\n' "$3" "$2" > "$work/$1"
  chmod +x "$work/$1"
}
engine no-driver 127 "ninferno-serve-sm86: error while loading shared libraries: libcuda.so.1: cannot open shared object file: No such file or directory"
engine no-other-lib 127 "ninferno-serve-sm86: error while loading shared libraries: libavcodec.so.60: cannot open shared object file: No such file or directory"
engine no-gpu 2 "no GPU runs the sm_86 kernels"
engine passes 0 ""
engine differs 1 "the kernels differ"
live() { bash -c "$fn"'; check_selector_facts 86 "$1"' live "$work/$1"; }
expect "no NVIDIA driver counts as no GPU" 0 live no-driver
said "no NVIDIA driver on this machine" "no driver"
expect "another missing library stops the run" 1 live no-other-lib
said "libavcodec.so.60" "another missing library"
expect "no GPU goes on" 0 live no-gpu
expect "a passing check goes on" 0 live passes
expect "a failing check stops the run" 1 live differs
said "the kernels differ" "a failing check"

[ "$failures" = 0 ] && echo "package checks: ok" || { echo "$failures failure(s)"; exit 1; }
