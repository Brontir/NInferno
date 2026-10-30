#!/bin/bash
# Ampere (RTX 3060 12 GB): the 9B GGUF Q4_K_M with its DFlash drafter (attached by q_amq9), spec-tune:
# the first lab pair of this generation (the plan transferred an Ada sibling: x2.31 without thinking,
# the task mix measured x1.59).
set -u
cd /root/claude
. /root/claude/lab.sh
cuda_env
BR=master ./run_cb.sh 86 ninferno > run_cb_t9a.out 2>&1; grep -E "^at|built|failed" run_cb_t9a.out
B=/root/claude/build_master_86; OUT=/root/claude/pairs_9b; mkdir -p $OUT
A=$(ls /root/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/ninferno/*/Qwen3.5-9B-Q4_K_M.qwen3_5_bf16.ninferno | head -1)
echo "artifact $A"; ls $(dirname $A) | grep -i -E "speculation|dflash" | head
lock tune9a
$B/apps/ninferno spec-tune "$A" --budget-seconds 3600 --out $OUT/9b_q4km_3060.json > $OUT/tune.log 2>&1
echo "tune rc=$?"
unlock
python3 - $OUT/9b_q4km_3060.json <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for k in ("default", "thinking", "non_thinking"):
    c = p.get(k, {}); print(f"  {k}: {c.get('config')} x{c.get('speedup',0):.2f} acc {c.get('acceptance',0):.3f} tok/round {c.get('tokens_per_round',0):.2f}")
PY
echo TUNE9ADONE
