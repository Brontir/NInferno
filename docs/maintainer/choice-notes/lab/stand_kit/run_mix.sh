#!/bin/bash
# The mixed workload (mix_tasks.py) on several configurations, --spec none, to check the choice model's speed
# predictions. usage: run_mix.sh SERVE_BIN "label|repo|quant|C|flags" ...
set -u
. /root/claude/lab.sh
cuda_env
S=$1; shift
D=/root/claude/mix; P=18972; mkdir -p $D
lock mix
trap unlock EXIT
wait_up() { local i; for i in $(seq 1 720); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$1/health 2>/dev/null | grep -q 200 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
for cfg in "$@"; do
  IFS='|' read -r label repo q c flags <<< "$cfg"
  if [ "${repo:0:1}" = / ]; then src="$repo"; else
    for i in 1 2 3; do $S $repo --quant $q --prepare-only > $D/prep_$label.log 2>&1 && break; sleep 60; done
    src="$repo --quant $q"; fi
  $S $src --spec none --port $P --max-context 32768 --max-concurrency $c $flags > $D/srv_$label.log 2>&1 & pid=$!
  if wait_up $P $pid; then
    python3 /root/claude/mix_tasks.py --port $P --messages /root/claude/messages --concurrency $c > $D/$label.jsonl 2> $D/$label.err
    echo "$label: $(tail -1 $D/$label.jsonl) | $(grep -m1 -oE 'rung [0-9]+ of [0-9]+: [^|]*' $D/srv_$label.log | cut -c1-120)"
  else echo "$label SRV_FAIL $(grep -m1 -iE 'error|fatal|refus' $D/srv_$label.log | cut -c1-200)"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
done
echo DONE
