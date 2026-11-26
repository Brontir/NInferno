#!/bin/bash
# bwtag.sh ARCH TAG : burst_wait -DACT -DNARROW -DPITCHW=32u (r46: 4 B a lane, a 128 B line a lane = 32 tags a request,
# 128 B of data: the tag-bound step, clocks per tag), N 2, C 5-8.
cd /root/claude; . lab.sh; cuda_env; [ "$1" = 70 ] && export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc -O3 -arch=sm_$1 -DACT -DNARROW -DPITCHW=32u -o bwptag_$1 burst_wait.cu || { echo BWFAIL > bwtag_$2.out; exit 1; }
./bwptag_$1 2 5 6 7 8 > bw_$2_N2tag.txt 2>&1; echo BWDONE > bwtag_$2.out
