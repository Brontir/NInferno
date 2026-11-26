#!/bin/bash
# bwnar.sh ARCH TAG : burst_wait -DACT -DNARROW (r42: 4 B a lane at a 36 B pitch), N 2, C 5-8.
cd /root/claude; . lab.sh; cuda_env; [ "$1" = 70 ] && export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc -O3 -arch=sm_$1 -DACT -DNARROW -o bwpnar_$1 burst_wait.cu || { echo BWFAIL > bwnar_$2.out; exit 1; }
./bwpnar_$1 2 5 6 7 8 > bw_$2_N2nar.txt 2>&1; echo BWDONE > bwnar_$2.out
