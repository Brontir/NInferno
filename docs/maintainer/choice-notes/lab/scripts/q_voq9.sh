#!/bin/bash
# Volta (Quadro GV100 32 GB, no lab pair of its generation, no int8 MMA), master: a Qwen3.5-9B GGUF without MTP layers, its DFlash drafter an axis of the
# choice. The plan's row, then the start without flags (prepare, then start, as ninferno-ui runs it)
# against --spec none on the task mix: the start attaches what the plan chose.
set -u
cd /root/claude
. /root/claude/lab.sh
BR=master ./run_cb.sh 70 ninferno-serve > run_cb_m70.out 2>&1; head -1 run_cb_m70.out; grep -E "built|error" run_cb_m70.out | head -3
B=/root/claude/build_master_70/apps/ninferno-serve
R=unsloth/Qwen3.5-9B-GGUF
$B $R --quant Q4_K_M --max-context 32768 --plan --json --no-load 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin); p=d["plan"]; s=p.get("speculation") or {}; e={}
print("plan:", s.get("served"), "|", s.get("basis"), "|", s.get("source"), "| drafter_bytes", p.get("drafter_bytes"), "| decode", e.get("decode_tok_s"))'
sed 's/--spec none/--spec $SPEC/; s/seq 1 720/seq 1 1440/' run_mix.sh > run_mix_df.sh
for spec in auto none; do
  SPEC=$spec bash run_mix_df.sh $B "vo_q9_$spec|$R|Q4_K_M|1|" | grep -v DONE
  grep -E "drafters \||speculation \| (serves|registry)" /root/claude/mix/srv_vo_q9_$spec.log | head -3 | cut -c1-230
  python3 - /root/claude/mix/vo_q9_$spec.jsonl <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{"task"')]
G=sum(x["G"] for x in r); T=sum(x["s"] for x in r)
print(f"   {len(r)} req, {G} tok out, {T:.2f} s, {G/T:.1f} tok/s overall")
PY
done
echo VOQ9DONE
