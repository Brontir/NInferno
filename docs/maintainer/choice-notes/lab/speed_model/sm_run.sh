#!/bin/bash
# Speed-model verification at a commit: sm_run.sh ARCH SHA TAG [BRANCH]
# Builds ninferno-serve of SHA (after any other build on this stand), writes the plan's estimates,
# then under the GPU lock measures plain decode 1/2/4/8 streams, prefill 1K-16K (Qwen3.5-9B Q4_K_M,
# Q8_0) and Gemma 4 12B Q4_K_M plain and mtp:5. Output: /root/claude/sm/TAG/.
set -u
. /root/claude/lab.sh
cuda_env; [ "${1:-}" = 70 ] && export PATH=/usr/local/cuda-12.8/bin:$PATH
ARCH=$1; SHA=$2; TAG=$3; BR=${4:-}
D=/root/claude/sm; OUT=$D/$TAG; mkdir -p $OUT; L=/root/claude/logs; mkdir -p $L
W=/root/claude/wt_at_$SHA; B=/root/claude/build_at_${SHA}_$ARCH
export CCACHE_DIR=/root/.ccache CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1
git -C /root/claude/wt_tst fetch -q https://github.com/Brontir/NInferno "+refs/heads/*:refs/remotes/gh/*" || exit 1
[ -d $W ] || git -C /root/claude/wt_tst worktree add -q --detach $W $SHA || exit 1
if [ -n "$BR" ]; then git -C $W checkout -q -f --detach gh/$BR; fi
echo "at $(git -C $W log --oneline -1)"
while pgrep -x ninja > /dev/null || pgrep -x nvcc > /dev/null; do sleep 30; done
[ -f $B/build.ninja ] || cmake -S $W -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DBUILD_TESTING=OFF > $L/cmake_sm_$SHA.log 2>&1 || { tail -20 $L/cmake_sm_$SHA.log; exit 1; }
( cd $B && nice ninja -j8 ninferno-serve > $L/ninja_sm_$SHA.log 2>&1 ) || { grep -m12 -E "error|FAILED" $L/ninja_sm_$SHA.log; exit 1; }
echo built
S=$B/apps/ninferno-serve
Q9=unsloth/Qwen3.5-9B-MTP-GGUF; G12=unsloth/gemma-4-12b-it-GGUF
DEC=(--spec none --no-prefix-reuse --max-concurrency 8 --max-context 4096)
PRE=(--spec none --no-prefix-reuse --max-concurrency 1 --max-context 17408)
MTP=(--spec mtp --draft-tokens 5 --no-prefix-reuse --max-concurrency 1 --max-context 4096)
PLAIN1=(--spec none --no-prefix-reuse --max-concurrency 1 --max-context 4096)
# name repo quant mode flags...
CASES=(
  "q4_dec $Q9 Q4_K_M decode ${DEC[*]}"
  "q4_pre $Q9 Q4_K_M prefill ${PRE[*]}"
  "q8_dec $Q9 Q8_0 decode ${DEC[*]}"
  "q8_pre $Q9 Q8_0 prefill ${PRE[*]}"
  "g12_dec $G12 Q4_K_M decode ${DEC[*]}"
  "g12_plain $G12 Q4_K_M spec ${PLAIN1[*]}"
  "g12_mtp5 $G12 Q4_K_M spec ${MTP[*]}"
)
for c in "${CASES[@]}"; do
  set -- $c; N=$1; R=$2; Q=$3; shift 4
  $S $R --quant $Q --plan --no-load --json "$@" > $OUT/$N.plan.json 2> $OUT/$N.plan.err
  $S $R --quant $Q --prepare-only --with-drafters > $OUT/$N.prep.log 2>&1
done
echo planned
lock speedmodel
trap "unlock" EXIT
P=18350
for c in "${CASES[@]}"; do
  set -- $c; N=$1; R=$2; Q=$3; M=$4; shift 4
  ss -ltn | grep -q ":$P " && { echo "port $P busy"; exit 1; }
  $S $R --quant $Q --port $P --model-id m "$@" > $OUT/$N.serve.log 2>&1 &
  SP=$!
  for i in $(seq 1 900); do curl -sf http://127.0.0.1:$P/health > /dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  if kill -0 $SP 2>/dev/null; then
    python3 $D/sm_bench.py $P $M $OUT/$N.json > $OUT/$N.out 2>&1 || echo "$N bench failed: $(tail -2 $OUT/$N.out)"
  else echo "$N: serve died: $(tail -2 $OUT/$N.serve.log)"; fi
  kill $SP; wait $SP 2>/dev/null
  echo "$N done"
done
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm --format=csv,noheader > $OUT/gpu.txt
echo SMDONE
