#!/bin/bash
# Cross-check of isa_probe with Nsight Compute. Needs ncu on PATH and a built ./isa_probe (see
# isa_probe.cu). A few minutes; a profiled kernel is replayed per metric pass.
#   tools/bench/isa_probe/isa_probe_ncu.sh ./isa_probe > isa_probe_ncu.txt
#
# 1. The tensor figure: the tensor pipe's active cycles and the MMA instructions of the throughput
#    kernel, from which cycles per MMA per SM follow as
#    pipe_tensor_cycles_active / (inst_executed_pipe_tensor / SMs).
# 2. The DRAM levels: the L2's read sectors, hit and missed, and the DRAM bytes of one chase. A
#    chase whose links are first touches shows about as many missed sectors as links (2 x
#    kDramLinks per chasing warp, warm-up and timed) and hits only from what else the kernel
#    reads.
#
# --launch-skip counts the launches of the kernels the name matches, in the probe's order (main()
# of isa_probe.cu):
#   mma_throughput: 0 warm, 1 timed (SMs x 4 CTAs of 4 warps), then the warm / timed pairs of the
#                   one-CTA-per-SM runs at 4, 8 and 16 warps (2..7).
#   chase:          four launches a level (1, 2, 4, 8 warps per SM), the levels in the order l1,
#                   l2, dram_tlbwarm, dram: 0..3, 4..7, 8..11, 12..15 (chase_shared follows).
# Under the profiler the probe's own timings and self-check are no measurement (the profiler
# changes how the launches overlap): only ncu's counters are read here, and a non-zero exit is
# printed, not acted on.
set -u
EXE=${1:-./isa_probe}
TENSOR=sm__cycles_elapsed.avg,sm__pipe_tensor_cycles_active.avg,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed
L2=lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,dram__bytes_read.sum

echo "== mma_throughput, the timed launch (SMs x 4 CTAs of 4 warps)"
ncu --kernel-name regex:mma_throughput --launch-skip 1 --launch-count 1 --metrics "$TENSOR" "$EXE" ||
    echo "ncu exited $?"
echo "== chase, dram_tlbwarm at 1 warp per SM"
ncu --kernel-name regex:chase --launch-skip 8 --launch-count 1 --metrics "$L2" "$EXE" ||
    echo "ncu exited $?"
echo "== chase, dram at 1 warp per SM"
ncu --kernel-name regex:chase --launch-skip 12 --launch-count 1 --metrics "$L2" "$EXE" ||
    echo "ncu exited $?"
