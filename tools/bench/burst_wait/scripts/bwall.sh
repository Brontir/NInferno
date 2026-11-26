#!/bin/bash
# bwall.sh ARCH TAG : the burst_wait probe set (r38): N 1-4 x C 5-8 with 128 B chase rows, N 2 / C 5-8 with 512 B rows.
# Out bw_TAG_*.txt, then BWDONE in bwall_TAG.out.
cd /root/claude; . lab.sh; cuda_env; [ "$1" = 70 ] && export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc -O3 -arch=sm_$1 -o bwp_$1 burst_wait.cu && nvcc -O3 -arch=sm_$1 -DROW16 -o bwp16_$1 burst_wait.cu || { echo BWFAIL > bwall_$2.out; exit 1; }
for n in 1 2 3 4; do ./bwp_$1 $n 5 6 7 8 > bw_$2_N$n.txt 2>&1; done
./bwp16_$1 2 5 6 7 8 > bw_$2_N2r512.txt 2>&1
echo BWDONE > bwall_$2.out
