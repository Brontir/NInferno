"""routes/pick (tools/route_pick/gen.py): no predicted band names a route the launch would demote for its
shape (#1077): ggml_mmq only on rows in whole 128-row tiles (generic_select.cpp mmq_call). No GPU."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMQ_ROW_MULTIPLE = 128  # ggml_mmq::kRowMultiple


def main():
    bad = []
    files = sorted(glob.glob(os.path.join(ROOT, "routes", "pick", "*.json")))
    for f in files:
        for e in json.load(open(f, encoding="utf-8")).get("entries", []):
            for lo, hi, route in e.get("bands", []):
                if route == "ggml_mmq" and e["n"] % MMQ_ROW_MULTIPLE:
                    bad.append("%s: %s %dx%d T %d-%d ggml_mmq on rows not in 128-row tiles"
                               % (os.path.basename(f), e["format"], e["n"], e["k"], lo, hi))
    for b in bad:
        print("FAIL:", b)
    if not files:
        print("FAIL: no routes/pick files")
        return 1
    print("route pick routes: %d files, %d bad bands" % (len(files), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
