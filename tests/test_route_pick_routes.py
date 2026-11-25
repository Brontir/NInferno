"""routes/pick (tools/route_pick/gen.py): no predicted band and no ranked list names ggml_mmq on rows MMQ does not
take (whole tiles of ggml_mmq::kRowMultiple, generic_select.cpp mmq_call), which the launch would demote (#1077); every
ranked list (T 1-8) is non-empty and ends at its first register-MMA schedule (an A16 route every policy admits), or holds
none. No GPU."""
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
            for lo, hi, routes in e.get("ranked", []):
                where = "%s: %s %dx%d T %d-%d" % (os.path.basename(f), e["format"], e["n"], e["k"], lo, hi)
                if not 1 <= lo <= hi <= 8 or not routes:
                    bad.append(where + " ranked list empty or outside T 1-8")
                    continue
                if "ggml_mmq" in routes and not pick.mmq_takes(e):
                    bad.append(where + " ranked ggml_mmq on rows not in whole tiles")
                mma = [i for i, r in enumerate(routes) if r.startswith("rmma.")]
                if mma and mma[0] != len(routes) - 1:
                    bad.append(where + " ranked list past its first rmma schedule")
    for b in bad:
        print("FAIL:", b)
    if not files:
        print("FAIL: no routes/pick files")
        return 1
    print("route pick routes: %d files, %d bad bands" % (len(files), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
