#!/bin/bash
# ninferno-gemma4 generate near the context end (Gemma 4 26B-A4B, sliding window 1024):
#   * a --decode that runs past the context is cut at the context end with a warning and
#     "truncated_by_context", exit 0, for plain decode and MTP; a --decode that fits is untouched;
#   * a context shorter than window + chunk (1024 with a 512 chunk) prefills and decodes.
# NINFERNO_TEST_GEMMA4_DIR names the artifact directory; without it the test skips (77).
#   cli_generate_context_real.sh <ninferno-gemma4>
set -u
BIN=$1
DIR=${NINFERNO_TEST_GEMMA4_DIR:-}
if [ -z "$DIR" ]; then echo "skip: NINFERNO_TEST_GEMMA4_DIR is not set"; exit 77; fi
TARGET=$DIR/gemma4_26b_a4b_nvfp4.ninferno
MTP=$DIR/gemma4_26b_assistant_q8_sl98304r2.ninferno
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
prompt() {  # prompt <length> <file>: <bos>, then a cycle of ordinary text tokens
  python3 -c "
import struct, sys
n = int(sys.argv[1])
ids = [2] + [236743 + (i * 7919) % 5000 for i in range(n - 1)]
sys.stdout.buffer.write(struct.pack('<%di' % n, *ids))" "$1" > "$2"
}
prompt 2000 "$WORK/p2000.i32"
prompt 1000 "$WORK/p1000.i32"
fail=0
run() {  # run <label> <prompt file> <context> <expected count or le:N> <truncated 0|1> <args...>
  local label=$1 file=$2 context=$3 want=$4 trunc=$5; shift 5
  "$BIN" "$TARGET" generate --tokens "$file" --context "$context" --chunk 512 "$@" \
    > "$WORK/out.json" 2> "$WORK/err.txt"
  local rc=$?
  local n
  n=$(python3 -c "
import json, sys
lines = [l for l in open(sys.argv[1]) if '\"generated\"' in l]
print(len(json.loads(lines[-1])['generated']) if lines else -1)" "$WORK/out.json" 2>/dev/null)
  local has=0 warned=0
  grep -q '"truncated_by_context":true' "$WORK/out.json" && has=1
  grep -q 'runs past the context' "$WORK/err.txt" && warned=1
  local ok=1
  [ $rc -eq 0 ] || ok=0
  case "$want" in
    le:*) [ -n "$n" ] && [ "$n" -ge 1 ] && [ "$n" -le "${want#le:}" ] || ok=0 ;;
    *) [ "$n" = "$want" ] || ok=0 ;;
  esac
  [ $has -eq "$trunc" ] && [ $warned -eq "$trunc" ] || ok=0
  echo "$label: rc=$rc generated=$n truncated=$has warned=$warned -> $([ $ok = 1 ] && echo ok || echo FAIL)"
  [ $ok = 1 ] || { tail -3 "$WORK/err.txt"; fail=1; }
}
run "plain, --decode 100 past the end of 2048" "$WORK/p2000.i32" 2048 48 1 --decode 100 --spec off
run "plain, --decode 48 fits exactly" "$WORK/p2000.i32" 2048 48 0 --decode 48 --spec off
run "MTP, --decode 100 past the end of 2048" "$WORK/p2000.i32" 2048 le:48 1 --decode 100 --drafter "$MTP" --draft 4
run "context 1024 below window + chunk" "$WORK/p1000.i32" 1024 24 0 --decode 24 --spec off
run "context 1024, MTP" "$WORK/p1000.i32" 1024 le:24 0 --decode 20 --drafter "$MTP" --draft 4
exit $fail
