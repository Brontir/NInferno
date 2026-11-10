#!/bin/bash
# Every registered CTest test carries a label from the set host, gpu, real, golden, slow
# (tests/cmake/NInfernoTests.cmake explains them), and a test labeled neither host, gpu nor real
# says nothing about what it needs. A test added with a bare add_test() and no
# ninferno_label_tests() fails here, so `ctest -L host` stays the set that runs with no GPU.
#
#   ctest_labels.sh [build dir]     (default: the current directory; CTEST overrides ctest, PYTHON python3)
#
# Exit 0: every test is labeled. 1: a test is not, or the build registers no test at all (an empty
# list is a broken build, not a clean one). 2: the list could not be read.
set -u -o pipefail
BUILD=${1:-.}
CTEST_BIN=${CTEST:-ctest}
[ -f "$BUILD/CTestTestfile.cmake" ] || { echo "ctest_labels: $BUILD is not a configured build dir" >&2; exit 2; }
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || { echo "ctest_labels: $PY not found" >&2; exit 2; }
LIST=$(cd "$BUILD" && "$CTEST_BIN" --show-only=json-v1) ||
  { echo "ctest_labels: $CTEST_BIN --show-only=json-v1 failed in $BUILD" >&2; exit 2; }
printf '%s' "$LIST" | "$PY" -I -c '
import json, sys
known = {"host", "gpu", "real", "golden", "slow"}
needs = {"host", "gpu", "real"}
try:
    tests = json.load(sys.stdin).get("tests", [])
except ValueError as e:
    print("ctest_labels: the test list is not JSON: %s" % e, file=sys.stderr)
    sys.exit(2)
if not tests:
    print("ctest_labels: the build registers no test")
    sys.exit(1)
bad = []
count = {}
for t in tests:
    labels = set()
    for p in t.get("properties", []):
        if p["name"] == "LABELS":
            v = p["value"]
            labels = set(v if isinstance(v, list) else [v])
    for l in labels:
        count[l] = count.get(l, 0) + 1
    if not labels:
        bad.append((t["name"], "no LABELS"))
    elif labels - known:
        bad.append((t["name"], "unknown label " + ",".join(sorted(labels - known))))
    elif not labels & needs:
        bad.append((t["name"], "none of host, gpu, real"))
    elif "host" in labels and labels & {"gpu", "real"}:
        bad.append((t["name"], "host with gpu or real"))
for name, why in sorted(bad):
    print("ctest_labels: %s: %s" % (name, why))
print("ctest_labels: %d tests, %s" % (len(tests), ", ".join("%s %d" % (k, count[k]) for k in sorted(count))))
sys.exit(1 if bad else 0)
'
