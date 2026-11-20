"""routes/pick (tools/route_pick/gen.py): no predicted band names ggml_mmq on rows MMQ does not take
(whole tiles of ggml_mmq::kRowMultiple, generic_select.cpp mmq_call), which the launch would demote (#1077).
No GPU."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "route_pick"))

import pick  # noqa: E402  (its kRowMultiple is read from ggml_mmq.h)


def main():
    bad = []
    files = sorted(glob.glob(os.path.join(ROOT, "routes", "pick", "*.json")))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            entries = json.load(fh).get("entries", [])
        for e in entries:
            for lo, hi, route in e.get("bands", []):
                if route == "ggml_mmq" and not pick.mmq_takes(e):
                    bad.append("%s: %s %dx%d T %d-%d ggml_mmq on rows not in %d-row tiles"
                               % (os.path.basename(f), e["format"], e["n"], e["k"], lo, hi,
                                  pick.MMQ_ROW_MULTIPLE))
    for b in bad:
        print("FAIL:", b)
    if not files:
        print("FAIL: no routes/pick files")
        return 1
    print("route pick routes: %d files, %d bad bands" % (len(files), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
