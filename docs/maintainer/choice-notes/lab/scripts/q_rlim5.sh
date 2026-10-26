#!/bin/bash
# det/hub-rate-limit after the Codex fix (exhausted quotas only): the whole fetch_test, the unit test.
set -u
cd /root/claude
. /root/claude/lab.sh
BR=det/hub-rate-limit ./run_cb.sh 75 ninferno-convert fetch_head_probe ninferno_convert_model_support_test > run_cb_rlim5.out 2>&1; head -1 run_cb_rlim5.out; grep -E "built|failed" run_cb_rlim5.out | head -3
B=/root/claude/build_det_hub-rate-limit_75; W=/root/claude/wt_det_hub-rate-limit
(cd $B && ctest -R model_support 2>&1 | grep -E "tests passed|Failed")
timeout 2400 venv-torch/bin/python -B $W/tests/convert_native/fetch_test.py $B/apps/ninferno-convert $W $B/tests/fetch_head_probe > fetch_rlim5.out 2>&1
echo "fetch_test rc=$? failed=$(grep -c FAILED fetch_rlim5.out) | $(grep 'rate limit' fetch_rlim5.out)"
echo RLIM5DONE
