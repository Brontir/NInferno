"""tools/route_table_retime.py (#1073): one route over the range, untimed table routes kept."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "route_table_retime.py")
FIELDS = ["format", "compute", "epilogue", "n", "k", "T", "route", "median_us"]


def run(tmp_path, bands, rows, margin="0.03"):
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"entries": [{"format": "q4k", "compute": "a16", "epilogue": "store",
                                              "n": 4096, "k": 4096, "bands": bands}]}, indent=1) + "\n")
    grid = tmp_path / "routes.csv"
    with grid.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for t, route, us in rows:
            # route_tune --a8-candidates 1 labels an a16 shape's rows a8.
            w.writerow(["q4k", "a8", "store", 4096, 4096, t, route, us])
    r = subprocess.run([sys.executable, TOOL, "--table", str(table), "--routes", str(grid),
                        "--tokens", "9-11", "--margin", margin], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(table.read_text())["entries"][0]["bands"], r


def test_stale_band_takes_one_route_over_the_range(tmp_path):
    # n2 wins T9 and T11, n4 wins T10 by noise: the range takes n2 (least worst regret), not a checker.
    rows = [(9, "n1", 200), (9, "n2", 100), (9, "n4", 101),
            (10, "n1", 200), (10, "n2", 101), (10, "n4", 100),
            (11, "n1", 200), (11, "n2", 100), (11, "n4", 102)]
    bands, _ = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]


def test_route_within_the_margin_is_kept(tmp_path):
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 102), ("n2", 100))]
    bands, _ = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows)
    assert bands == [[1, 11, "n1"], [12, 16777216, "n2"]]


def test_untimed_table_route_is_kept_and_named(tmp_path):
    rows = [(t, "n2", 100) for t in (9, 10, 11)]
    bands, r = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows)
    assert bands == [[1, 11, "n1"], [12, 16777216, "n2"]]
    assert "n1 not timed" in r.stderr


if __name__ == "__main__":
    import pathlib
    import tempfile

    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                test(pathlib.Path(d))
            print("ok", name)
