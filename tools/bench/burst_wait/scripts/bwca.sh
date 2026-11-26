#!/bin/bash
# bwca.sh ARCH TAG : burst_wait with vec-like activation rows (512 B a warp, default caching, a CTA's warps on one chain), N 2.
cd /root/claude; . lab.sh; cuda_env; [ "$1" = 70 ] && export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc -O3 -arch=sm_$1 -DROW16 -DCA -o bwpca_$1 burst_wait.cu || { echo BWFAIL > bwca_$2.out; exit 1; }
./bwpca_$1 2 5 6 7 8 > bw_$2_N2r512ca.txt 2>&1; echo BWDONE > bwca_$2.out
