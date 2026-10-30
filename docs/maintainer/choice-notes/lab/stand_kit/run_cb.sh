#!/bin/bash
# Build and run the choice test on the det/choice-model branch (CPU only, no GPU lock).
# usage: run_cb.sh ARCH [TARGETS...]
set -u
. /root/claude/lab.sh
cuda_env
ARCH=$1; shift; T=${*:-ninferno_choice_test}
BR=${BR:-det/choice-model}; N=$(echo $BR | tr / _)
W=/root/claude/wt_$N; B=/root/claude/build_${N}_$ARCH; L=/root/claude/logs; mkdir -p $L
export CCACHE_DIR=/root/.ccache CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1
git -C /root/claude/wt_tst fetch -q https://github.com/Brontir/NInferno "+refs/heads/$BR:refs/remotes/gh/$BR" || { echo "fetch failed"; exit 1; }
[ -d $W ] || git -C /root/claude/wt_tst worktree add -q --detach $W gh/$BR
git -C $W checkout -q -f --detach gh/$BR
echo "at $(git -C $W log --oneline -1)"
CFG="-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DBUILD_TESTING=ON -DNINFERNO_BUILD_BENCHMARKS=OFF"
[ -f $B/build.ninja ] || cmake -S $W -B $B $CFG > $L/cmake_cb.log 2>&1 || { echo "cmake failed"; tail -20 $L/cmake_cb.log; exit 1; }
t0=$(date +%s); (cd $B && nice ninja -j8 $T > $L/ninja_cb_$N.log 2>&1) || { echo "build failed"; grep -m12 -E "error|FAILED" $L/ninja_cb_$N.log; exit 1; }
echo "built $(( $(date +%s) - t0 ))s"
for t in $T; do f=$(find $B -name $t -type f -executable | head -1); case $t in *_test) [ -n "$f" ] && echo "== $t" && $f 2>&1 | tail -15;; esac; done
echo DONE
