#!/bin/bash
# bwact.sh ARCH TAG : burst_wait -DACT (r39: vec's activation request shape on an L1-resident shared table), N 2 and 3.
cd /root/claude; . lab.sh; cuda_env; [ "$1" = 70 ] && export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc -O3 -arch=sm_$1 -DACT -o bwpact_$1 burst_wait.cu || { echo BWFAIL > bwact_$2.out; exit 1; }
./bwpact_$1 2 5 6 7 8 > bw_$2_N2act.txt 2>&1; ./bwpact_$1 3 5 6 7 8 > bw_$2_N3act.txt 2>&1; echo BWDONE > bwact_$2.out
