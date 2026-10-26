#!/bin/bash
# Again with each failure's stderr kept (5 of 51 gave no plan once, all planned on a retry).
# master after #131-#142: every shipped pair's target planned on a card of its class with no flags (the
# drafter axis decides): each should stay "measured" and fit.
set -u
cd /root/claude
. /root/claude/lab.sh
cuda_env
BRN=master; B=/root/claude/build_master_89
mkdir -p /root/claude/shipm_err; i=0
S=$B/apps/ninferno-serve
python3 -c '
import json
for p in json.load(open("/root/claude/wt_master/src/spec/data/drafter_registry.json"))["pairs"]:
    repo, _, quant = p["target_repo"].partition(" ")
    print(repo.split("@")[0] + "|" + quant + "|" + p["tune"]["key"]["gpu"])
' | while IFS='|' read -r R Q C; do
  case ${C%%:*} in ad104) DEV="NVIDIA GeForce RTX 4070 Ti";; ad102) DEV="NVIDIA GeForce RTX 4090";; gb203) DEV="NVIDIA GeForce RTX 5070 Ti";; *) continue;; esac
  QF=(); [ -n "$Q" ] && QF=(--quant "$Q")
  i=$((i+1)); out=$($S $R "${QF[@]}" --plan --no-load --plan-device "$DEV" --json 2>/root/claude/shipm_err/$i.err | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print("no plan"); sys.exit()
s=(d.get("plan") or {}).get("speculation") or {}
print(s.get("basis"), "|", s.get("served"), "| fits", (d.get("plan") or {}).get("fits"))')
  echo "$R $Q ($C): $out"; [ "$out" = "no plan" ] && echo "  stderr: $(tail -2 /root/claude/shipm_err/$i.err | tr '
' ' ')"
done
echo SHIPM2DONE
