#!/bin/bash
# Ampere (RTX 3060): the second lab pair of this generation. The 9B MTP GGUF Q4_K_M with its MTP layers
# and DFlash drafter (Ada has this pair: 4070 Ti). With 9b_q4km_3060 it tells whether the round cost an
# Ampere card pays over an Ada one (x1.165 for DFlash k=7 on the 9B) holds across pairs and drafters.
set -u
cd /root/claude
. /root/claude/lab.sh
cuda_env
B=/root/claude/build_master_86; OUT=/root/claude/pairs_9b; R=unsloth/Qwen3.5-9B-MTP-GGUF
for i in 1 2 3; do $B/apps/ninferno-serve $R --quant Q4_K_M --with-drafters --prepare-only > $OUT/prep_9m.log 2>&1 && break; sleep 60; done
tail -2 $OUT/prep_9m.log | cut -c1-200
A=$(ls /root/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/ninferno/*/*Q4_K_M*.ninferno | head -1)
echo "artifact $A"; ls $(dirname $A) | head
lock tune9m
$B/apps/ninferno spec-tune "$A" --budget-seconds 3600 --out $OUT/9bmtp_q4km_3060.json > $OUT/tune9m.log 2>&1
echo "tune rc=$?"
unlock
python3 - $OUT/9bmtp_q4km_3060.json <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for k in ("default", "thinking", "non_thinking"):
    c = p.get(k, {}); print(f"  {k}: {c.get('config')} x{c.get('speedup',0):.2f} acc {c.get('acceptance',0):.3f} tok/round {c.get('tokens_per_round',0):.2f}")
for m in p.get("measurements", [])[:0]: pass
PY
echo TUNE9MDONE
