#!/bin/bash
# bwr47.sh ARCH TAG : r47 probes: ACT2 (two interleaved 16 B loads a step), NARROW to E 256, NARROW4 (4 B in ACT's layout),
# TAG with a 48 KB table (ACT_ROWS 12). N 2, C 5 6 8. Out bw_TAG_N2{act2,nar256,nar4,tag48}.txt, BWDONE in bwr47_TAG.out.
cd /root/claude; . lab.sh; cuda_env; [ "$1" = 70 ] && export PATH=/usr/local/cuda-12.8/bin:$PATH
A=$1; T=$2
nvcc -O3 -arch=sm_$A -DACT -DACT2 -o bwp_act2_$A burst_wait.cu && ./bwp_act2_$A 2 5 6 8 > bw_${T}_N2act2.txt 2>&1
nvcc -O3 -arch=sm_$A -DACT -DNARROW -DLONG256 -o bwp_nar256_$A burst_wait.cu && ./bwp_nar256_$A 2 5 6 8 > bw_${T}_N2nar256.txt 2>&1
nvcc -O3 -arch=sm_$A -DACT -DNARROW4 -o bwp_nar4_$A burst_wait.cu && ./bwp_nar4_$A 2 5 6 8 > bw_${T}_N2nar4.txt 2>&1
nvcc -O3 -arch=sm_$A -DACT -DNARROW -DPITCHW=32u -DACT_ROWS=12u -o bwp_tag48_$A burst_wait.cu && ./bwp_tag48_$A 2 5 6 8 > bw_${T}_N2tag48.txt 2>&1
echo BWDONE > bwr47_$T.out
