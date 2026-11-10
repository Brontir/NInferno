#!/bin/bash
# Cross-check of isa_probe's tensor figure with Nsight Compute: the tensor pipe's active cycles
# and the MMA instructions of the throughput kernel, from which cycles per MMA per SM follow as
# pipe_tensor_cycles_active / (inst_executed_pipe_tensor / SMs). Needs ncu on PATH and a built
# ./isa_probe (see isa_probe.cu). About 2 minutes; the kernel is replayed per metric pass.
#   tools/bench/isa_probe/isa_probe_ncu.sh ./isa_probe > isa_probe_ncu.txt
set -eu
EXE=${1:-./isa_probe}
ncu --kernel-name regex:mma_throughput --launch-skip 1 --launch-count 1 \
    --metrics sm__cycles_elapsed.avg,sm__pipe_tensor_cycles_active.avg,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    "$EXE"
ncu --kernel-name regex:chase --launch-skip 8 --launch-count 1 \
    --metrics lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,dram__bytes_read.sum \
    "$EXE"
