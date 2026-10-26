#!/bin/bash
# det/pairs-ampere with generation round costs (770ae5b2): spec tests, the plan of four targets on the
# RTX 3060 (master: an Ada transfer as measured), and the 9B MTP Q6_K (no Ampere pair: Ada's at
# Ampere's round costs) without flags against --spec none on the task mix.
set -u
cd /root/claude
. /root/claude/lab.sh
cuda_env
BR=det/pairs-ampere ./run_cb.sh 86 ninferno-serve ninferno_spec_pairs_test ninferno_spec_test ninferno_plan_dry_run_test ninferno_plan_drafter_kinds_test > run_cb_pamp2.out 2>&1; head -1 run_cb_pamp2.out; grep -E "built|failed" run_cb_pamp2.out | head -3
B=/root/claude/build_det_pairs-ampere_86
(cd $B && ctest -R "spec_pairs|spec_test|plan_dry_run|plan_drafter_kinds" 2>&1 | grep -E "tests passed|Failed")
for a in "unsloth/Qwen3.5-9B-GGUF Q4_K_M" "unsloth/Qwen3.5-9B-GGUF Q5_K_M" "unsloth/Qwen3.5-9B-MTP-GGUF Q4_K_M" "unsloth/Qwen3.5-9B-MTP-GGUF Q6_K"; do
  set -- $a
  for s in /root/claude/build_master_86 $B; do
    $s/apps/ninferno-serve $1 --quant $2 --max-context 32768 --plan --json --no-load 2>/dev/null | python3 /root/claude/plan_spec_row.py "$1" "$2" "$(basename $s)"
  done
done
S=$B/apps/ninferno-serve; R=unsloth/Qwen3.5-9B-MTP-GGUF
sed 's/--spec none/--spec $SPEC/; s/seq 1 720/seq 1 1440/' run_mix.sh > run_mix_df.sh
for spec in auto none; do
  SPEC=$spec bash run_mix_df.sh $S "am_m6_$spec|$R|Q6_K|1|" | grep -v DONE
  grep -E "speculation \| (serves|registry)" /root/claude/mix/srv_am_m6_$spec.log | head -2 | cut -c1-200
  python3 - /root/claude/mix/am_m6_$spec.jsonl <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{"task"')]
G=sum(x["G"] for x in r); T=sum(x["s"] for x in r)
print(f"   {len(r)} req, {G} tok out, {T:.2f} s, {G/T:.1f} tok/s overall")
PY
done
echo PAMP2DONE
