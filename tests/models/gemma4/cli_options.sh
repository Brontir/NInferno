#!/bin/bash
# ninferno-gemma4 refuses an option no command reads instead of ignoring it (an ignored
# --activations exact once ran the default activation profile unnoticed). No device needed:
# the options are parsed before the artifact is opened.
#   cli_options.sh <ninferno-gemma4>
# set -u without -e: the refusal's nonzero exit is what is tested.
set -u
BIN=$1
err=$(mktemp)
trap 'rm -f "$err"' EXIT
"$BIN" /nonexistent.ninferno score --tokens /nonexistent.i32 --activations exact > /dev/null 2> "$err"
rc=$?
if [ $rc -ne 0 ] && grep -q "unknown option --activations" "$err"; then
  echo "unknown option refused: ok"; exit 0
fi
echo "unknown option not refused (rc=$rc): $(cat "$err")"; exit 1
