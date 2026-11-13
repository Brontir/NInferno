#!/usr/bin/env bash
# #850 (a), docs/maintainer/kernel-proposals.md entry P: three sweeps of every R8 decode schedule on
# Qwen3.5-9B's decode problems on one idle card, then a table to paste and the arguments of the
# R8 decode table's merge. Nothing here chooses a route; the bench times the schedules it compiles.
#
#   tools/bench/run_850_r8_sweep.sh BENCH [OUT]
#   tools/bench/run_850_r8_sweep.sh --summarize OUT          (no GPU: the table again)
#
# BENCH is ninferno_r8_schedule_bench of a build of this card's architecture (configure with
# -DNINFERNO_BUILD_BENCHMARKS=ON; it lands in <build>/bench/). Master's binary and this branch's
# time the same kernels and print the same rules mark: either serves. OUT defaults to
# r8_850_<class>_<MMDD_HHMM> in the current directory. With several cards, pick one
# (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i>). A compute process already on the GPUs
# stops the script (ALLOW_BUSY=1 runs anyway; the figures are then not the protocol's).
#
# Protocol (the bench's own, its header comment): copies of one random weight covering four times
# the L2, a CUDA graph of 60 calls cycling through them per schedule, the schedules interleaved
# within each sample, the median of 15 replays divided by 60; every schedule's output bytes are
# compared (same=1). Three sweeps, each two bench runs:
#   down_<i>.txt     the down projections, the merge's input: Q6_K 4096x12288 store and add (the
#                    candidate), Q4_K and Q5_K 4096x12288 (the same call in the other mixes);
#   context_<i>.txt  the rest of the 9B step (attn_qkv, attn_gate, ssm_out, gate+up, q, k/v) and the
#                    K >= 8192 shapes the rule check reads (Qwen3.5-4B's 2560x9216, Qwen3.8's
#                    5120x17408, the goldens' 28672x8192): reported, never merged.
# A run takes a few minutes; the card must stay idle for all six.
#
# After the sweeps, on a dev tree with the OUT directories of every card copied in (the merge
# keeps every entry of a card it was not given and every problem the sweeps did not time):
#   python3 tools/bench/r8_decode_table.py --merge src/ops/generic/r8_decode_table.h \
#     --out src/ops/generic/r8_decode_table.h <class>=<OUT>/down_1.txt,<OUT>/down_2.txt,<OUT>/down_3.txt ...
# (the summary prints each card's argument). Then regenerate nothing else: no selector golden has
# a 4096x12288 row, so a merge of the down files moves none (check with
# tools/audit/golden_diff.py --range origin/master...HEAD).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
summary="$here/r8_sweep_summary.py"
py=${PYTHON:-python3}

summarize() {
    local out=$1
    local files=()
    for i in 1 2 3; do
        for part in down context; do
            [[ -f "$out/${part}_$i.txt" ]] && files+=("$out/${part}_$i.txt")
        done
    done
    [[ ${#files[@]} -gt 0 ]] || { echo "no sweep files in $out" >&2; exit 2; }
    "$py" "$summary" "${files[@]}" | tee "$out/summary.md"
    local class
    class=$(sed -n 's/^device .* class \([^,]*\),.*/\1/p' "$out/down_1.txt" | head -1)
    echo
    echo "merge argument: ${class}=$out/down_1.txt,$out/down_2.txt,$out/down_3.txt" |
        tee -a "$out/summary.md"
}

if [[ ${1:-} == --summarize ]]; then
    [[ $# -eq 2 ]] || { echo "usage: $0 --summarize OUT" >&2; exit 2; }
    summarize "$2"
    exit 0
fi
[[ $# -ge 1 && $# -le 2 ]] || { sed -n '6,7p' "$0" >&2; exit 2; }
bench=$1
[[ -x "$bench" ]] || { echo "$bench: not an executable" >&2; exit 2; }

if command -v nvidia-smi >/dev/null; then
    busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || true)
    if [[ -n "$busy" && ${ALLOW_BUSY:-0} != 1 ]]; then
        echo "compute processes on the GPUs (ALLOW_BUSY=1 to run anyway):" >&2
        echo "$busy" >&2
        exit 3
    fi
fi

down=(
    --problem q6k,store,4096,12288 --problem q6k,add,4096,12288
    --problem q4k,store,4096,12288 --problem q4k,add,4096,12288
    --problem q5k,store,4096,12288
)
context=(
    --problem q6k,store,8192,4096 --problem q4k,store,8192,4096
    --problem q4k,store,4096,4096 --problem q5k,store,4096,4096
    --problem q4k,swiglu,12288,4096
    --problem q4k,store,1024,4096 --problem q6k,store,1024,4096
    --problem q6k,store,2560,9216 --problem q4k,store,2560,9216
    --problem q6k,store,5120,17408 --problem q4k,store,5120,17408
    --problem q6k,store,28672,8192 --problem q4k,store,28672,8192
)
protocol=(--samples 15 --launches 60)

probe=$("$bench" --problem q4k,store,1024,4096 --samples 1 --launches 1)
class=$(sed -n 's/^device .* class \([^,]*\),.*/\1/p' <<<"$probe" | head -1)
[[ -n "$class" ]] || { echo "the bench printed no device line:" >&2; echo "$probe" >&2; exit 1; }
out=${2:-r8_850_${class/:/_}_$(date +%m%d_%H%M)}
mkdir -p "$out"
{
    echo "bench $bench"
    echo "tree $(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "date $(date -Is)"
    command -v nvidia-smi >/dev/null &&
        nvidia-smi --query-gpu=name,pci.device_id,driver_version,clocks.max.sm,clocks.max.mem \
            --format=csv,noheader
    grep '^device\|^board' <<<"$probe"
} >"$out/meta.txt"
cat "$out/meta.txt"

for i in 1 2 3; do
    echo "sweep $i: down"
    "$bench" "${down[@]}" "${protocol[@]}" >"$out/down_$i.txt"
    echo "sweep $i: context"
    "$bench" "${context[@]}" "${protocol[@]}" >"$out/context_$i.txt"
done
echo
summarize "$out"
