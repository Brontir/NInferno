#!/bin/bash
# Cases a ctest run skipped inside passing tests ("207/207" on a card without FP4 or FP8
# MMA is not full coverage). A test that cannot run one of its cases on this build or device prints
# "SKIP <case>: <reason>" and passes; a test skipped whole exits 77 and ctest reports it itself.
# This reads the last run's log and prints the skipped cases by reason, then by test.
#
#   ctest_skips.sh <build dir>
set -u
LOG=${1:-.}/Testing/Temporary/LastTest.log
[ -f "$LOG" ] || { echo "ctest_skips: no $LOG (run ctest first)" >&2; exit 2; }
awk '
  / Testing: / { test = $NF; next }
  /^SKIP/ {
    reason = $0
    sub(/^SKIP:? */, "", reason)
    if (index(reason, ": ") > 0) reason = substr(reason, index(reason, ": ") + 2)
    total++; by_reason[reason]++; by_test[test]++
  }
  END {
    printf "ctest skips: %d cases in %d tests\n", total + 0, length(by_test)
    if (total == 0) exit
    print "by reason:"; fflush()
    for (r in by_reason) printf "  %4d  %s\n", by_reason[r], r | "sort -rn"
    close("sort -rn")
    print "by test:"; fflush()
    for (t in by_test) printf "  %4d  %s\n", by_test[t], t | "sort -rn"
    close("sort -rn")
  }' "$LOG"
