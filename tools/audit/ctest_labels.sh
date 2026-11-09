#!/bin/bash
# Every registered CTest test carries a label from the set host, gpu, real, golden, slow
# (tests/cmake/NInfernoTests.cmake explains them), and a test labeled neither host, gpu nor real
# says nothing about what it needs. A test added with a bare add_test() and no
# ninferno_label_tests() fails here, so `ctest -L host` stays the set that runs with no GPU.
#
#   ctest_labels.sh [build dir]     (default: the current directory; CTEST overrides ctest)
set -u
BUILD=${1:-.}
CTEST_BIN=${CTEST:-ctest}
[ -f "$BUILD/CTestTestfile.cmake" ] || { echo "ctest_labels: $BUILD is not a configured build dir" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "ctest_labels: python3 not found" >&2; exit 2; }
(cd "$BUILD" && "$CTEST_BIN" --show-only=json-v1) | python3 -I -c '
import json, sys
known = {"host", "gpu", "real", "golden", "slow"}
needs = {"host", "gpu", "real"}
tests = json.load(sys.stdin).get("tests", [])
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
