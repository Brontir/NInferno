#!/usr/bin/env bash
# readonly_cache_test.sh DENSE GGUF: a conversion as an ordinary user against a model directory it
# cannot write (#975, #1047). DENSE is a built ninferno-dense, GGUF a small dense GGUF file (the
# ctest passes NINFERNO_TEST_GGUF; without it the test is skipped, exit 77).
#   1. The model directory is made unwritable (chmod a-w); the run is an ordinary user's (as root:
#      setpriv to nobody, else the caller itself), with XDG_CACHE_HOME in a fresh directory.
#   2. The first start converts into $XDG_CACHE_HOME/ninferno/converted/<key>/, writes nothing
#      beside the model, and leaves no lock probe behind.
#   3. A second start converts nothing and serves the same artifact, byte for byte untouched.
# The first run of this test (RTX 2080 Ti stand, as nobody, a build under /root) stopped before any
# conversion: resolve_ranking's exists() threw on the build tree the binary embeds, which nobody
# cannot read; it is now false there (drafters.cpp). The start runs from a directory the user can
# read, as a user's does, so an unreadable working directory is not what this test covers.
# No GPU is needed: ninferno-dense converts before it opens the device (CUDA_VISIBLE_DEVICES is
# empty, so the start then stops at the device; only the conversion is checked).
set -euo pipefail
DENSE=${1:?usage: readonly_cache_test.sh DENSE GGUF}
GGUF=${2:-${NINFERNO_TEST_GGUF:-}}
if [ -z "$GGUF" ] || [ ! -f "$GGUF" ]; then
  echo "SKIP: no GGUF (set NINFERNO_TEST_GGUF to a small dense GGUF file)"; exit 77
fi
work=$(mktemp -d /tmp/ninferno-rocache.XXXXXX)
trap 'chmod -R u+w "$work" 2>/dev/null; rm -rf "$work"' EXIT
chmod 755 "$work"
mkdir "$work/model" "$work/cache"
cp "$GGUF" "$work/model/model.gguf"
chmod 555 "$work/model"
chmod 444 "$work/model/model.gguf"
as_user=()
if [ "$(id -u)" = 0 ]; then
  command -v setpriv > /dev/null || { echo "SKIP: running as root without setpriv"; exit 77; }
  chown 65534:65534 "$work/cache"
  as_user=(setpriv --reuid=65534 --regid=65534 --clear-groups)
  # nobody must reach the binary and the work directory (a build under /root, mode 700, is not).
  if ! "${as_user[@]}" test -x "$DENSE" || ! "${as_user[@]}" test -w "$work/cache"; then
    echo "SKIP: nobody cannot run $DENSE or write $work/cache (put the build where other users can read it)"; exit 77
  fi
fi
failures=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }
start() { # log; from the user's own directory, as an ordinary user starts it
  cd "$work/cache"
  env CUDA_VISIBLE_DEVICES= XDG_CACHE_HOME="$work/cache" HOME="$work/cache" \
    ${as_user[@]+"${as_user[@]}"} "$DENSE" "$work/model/model.gguf" chat --prompt hi --decode 1 > "$1" 2>&1 < /dev/null || true
}
start "$work/first.log"
own=$(find "$work/cache/ninferno/converted" -name '*.ninferno' 2>/dev/null | head -1 || true)
[ -n "$own" ] || { fail "first start: no artifact under XDG_CACHE_HOME/ninferno/converted"; sed 's/^/  /' "$work/first.log"; }
grep -q "converted .* to $work/cache/ninferno/converted/" "$work/first.log" || fail "first start: no 'converted ... to <own cache>' line"
[ ! -e "$work/model/.ninferno-cache" ] || fail "first start: wrote beside the model"
if find "$work/cache" -name '.convert.probe.*' 2>/dev/null | grep -q .; then fail "first start: a lock probe was left"; fi
if [ -n "$own" ]; then
  sum=$(sha256sum "$own" | cut -d' ' -f1); mtime=$(stat -c %Y "$own")
  start "$work/second.log"
  if grep -q "converted " "$work/second.log"; then fail "second start converted again"; sed 's/^/  /' "$work/second.log"; fi
  [ "$(sha256sum "$own" | cut -d' ' -f1)" = "$sum" ] && [ "$(stat -c %Y "$own")" = "$mtime" ] || fail "second start changed the artifact"
  [ "$(find "$work/cache/ninferno/converted" -name '*.ninferno' 2>/dev/null | wc -l || true)" = 1 ] || fail "second start made another artifact"
fi
[ "$failures" = 0 ] && echo "readonly_cache_test: passed" || exit 1
