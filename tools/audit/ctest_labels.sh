#!/bin/bash
# Every registered CTest test carries a label from the set host, gpu, real, golden, slow
# (tests/cmake/NInfernoTests.cmake explains them), and a test labeled neither host, gpu nor real
# says nothing about what it needs. A test added with a bare add_test() and no
# ninferno_label_tests() fails here, so `ctest -L host` stays the set that runs with no GPU.
#
#   ctest_labels.sh [build dir [config]]
#
# The build dir defaults to the current directory. A multi-config generator (Visual Studio, Xcode,
# Ninja Multi-Config) lists no test without a configuration: pass it as the second argument or in
# CTEST_CONFIG. CTEST overrides the ctest command, PYTHON python3.
#
# Exit 0: every test is labeled. 1: a test is not, or the build registers no test at all (an empty
# list is a broken build, not a clean one). 2: the list could not be read or has an unexpected shape.
set -u -o pipefail
BUILD=${1:-.}
CONFIG=${2:-${CTEST_CONFIG:-}}
CTEST_BIN=${CTEST:-ctest}
[ -f "$BUILD/CTestTestfile.cmake" ] || { echo "ctest_labels: $BUILD is not a configured build dir" >&2; exit 2; }
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || { echo "ctest_labels: $PY not found" >&2; exit 2; }
"$PY" -I -c 'import json' >/dev/null 2>&1 || { echo "ctest_labels: $PY cannot run" >&2; exit 2; }
CONFIG_ARGS=()
[ -z "$CONFIG" ] || CONFIG_ARGS=(-C "$CONFIG")
LIST=$("$CTEST_BIN" --test-dir "$BUILD" ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} --show-only=json-v1) ||
  { echo "ctest_labels: $CTEST_BIN --show-only=json-v1 failed in $BUILD" >&2; exit 2; }
printf '%s' "$LIST" | "$PY" -I -c '
import json, sys
known = {"host", "gpu", "real", "golden", "slow"}
needs = {"host", "gpu", "real"}
try:
    tests = json.load(sys.stdin)["tests"]
    if not isinstance(tests, list):
        raise TypeError("\"tests\" is not a list")
    bad = []
    count = {}
    for t in tests:
        labels = set()
        for p in t.get("properties") or []:
            if p["name"] == "LABELS":
                v = p["value"]
                labels = set(v if isinstance(v, list) else [v])
                if not all(isinstance(l, str) for l in labels):
                    raise TypeError("a LABELS value is not a string")
        for l in labels:
            count[l] = count.get(l, 0) + 1
        name = t["name"]
        if not labels:
            bad.append((name, "no LABELS"))
        elif labels - known:
            bad.append((name, "unknown label " + ",".join(sorted(labels - known))))
        elif not labels & needs:
            bad.append((name, "none of host, gpu, real"))
        elif "host" in labels and labels & {"gpu", "real"}:
            bad.append((name, "host with gpu or real"))
    bad.sort()
except Exception as e:
    print("ctest_labels: unexpected test list: %s: %s" % (type(e).__name__, e), file=sys.stderr)
    sys.exit(2)
if not tests:
    print("ctest_labels: the build registers no test")
    sys.exit(1)
for name, why in bad:
    print("ctest_labels: %s: %s" % (name, why))
summary = "".join(", %s %d" % (k, count[k]) for k in sorted(count))
print("ctest_labels: %d test%s%s" % (len(tests), "" if len(tests) == 1 else "s", summary))
sys.exit(1 if bad else 0)
'
