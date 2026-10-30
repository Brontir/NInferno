#!/bin/bash
# det/gemma-layer-params: the plan's prefill estimate (master: Gemma's parameters ~18 % high; the
# branch: per layer) against a measured 22K-token prefill on the RTX 4090, Qwen 9B as the control.
set -u
. /root/claude/lab.sh
cuda_env
cd /root/claude
M=/root/claude/build_master_89/apps/ninferno-serve; BRS=/root/claude/build_det_gemma-layer-params_89/apps/ninferno-serve
D=/root/claude/gpre; P=18973; mkdir -p $D
PM=/root/claude/wt_det_gemma-layer-params/tools/bench/perf_matrix.py
wait_up() { local i; for i in $(seq 1 360); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$1/health 2>/dev/null | grep -q 200 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
for mq in "unsloth/gemma-4-12b-it-GGUF|Q4_K_M" "unsloth/Qwen3.5-9B-GGUF|Q4_K_M" "unsloth/gemma-4-26B-A4B-it-GGUF|UD-Q4_K_XL"; do
  repo=${mq%%|*}; q=${mq#*|}; tag=$(basename $repo)_$q
  for s in $M $BRS; do
    $s $repo --quant $q --spec none --max-context 32768 --plan --json --no-load 2>/dev/null | python3 -c '
import json,sys
c=json.load(sys.stdin)["choice"]["chosen"]; print("  plan", sys.argv[1], "prefill", c.get("prefill_tok_s"), "decode", c.get("decode_tok_s"), "chunk", c.get("chunk"))' "$(echo $s | cut -d/ -f4)"
  done
  for i in 1 2 3; do $M $repo --quant $q --prepare-only > $D/prep_$tag.log 2>&1 && break; sleep 60; done
  lock gpre
  $M $repo --quant $q --spec none --port $P --max-context 32768 --max-concurrency 1 > $D/srv_$tag.log 2>&1 & pid=$!
  if wait_up $P $pid; then
    python3 $PM --port $P --concurrency 1 > $D/pm_$tag.log 2>&1
    echo "$tag measured: $(grep -E 'aggregate|decode=' $D/pm_$tag.log | tr '\n' ' ' | cut -c1-260)"
  else echo "$tag SRV_FAIL $(grep -m1 -iE 'error|fatal' $D/srv_$tag.log | cut -c1-200)"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; unlock; sleep 3
done
echo GPREDONE
