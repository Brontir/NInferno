"""tools/route_table_retime.py (#1073): one route over the range, untimed table routes kept, no W4A8 route
in a table, a moved count's evidence row dropped."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "route_table_retime.py")
FIELDS = ["format", "compute", "epilogue", "n", "k", "T", "route", "median_us"]


def run(tmp_path, bands, rows, margin="0.03", evidence=None):
    table = tmp_path / "t.json"
    entry = {"format": "q4k", "compute": "a16", "epilogue": "store", "n": 4096, "k": 4096, "bands": bands}
    if evidence is not None:
        entry["evidence_us"] = evidence
    table.write_text(json.dumps({"entries": [entry]}, indent=1) + "\n")
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
    out = json.loads(table.read_text())["entries"][0]
    return (out["bands"], r) if evidence is None else (out["bands"], out["evidence_us"])


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


def test_w4a8_routes_are_not_candidates(tmp_path):
    # The W4A8 heuristic answers those routes, not a table: the fastest A16 route takes the range.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 150), ("w4a8.32x128", 100))]
    bands, _ = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]


def test_a_tie_keeps_the_table_route(tmp_path):
    # n1 is far behind at T9, so the range moves; n2 and n4 tie on worst regret, and the table's own
    # n4 (at T10-11) is kept rather than the alphabetically first n2.
    rows = [(9, "n1", 300), (9, "n2", 100), (9, "n4", 100),
            (10, "n1", 100), (10, "n2", 110), (10, "n4", 110),
            (11, "n1", 100), (11, "n2", 110), (11, "n4", 110)]
    bands, _ = run(tmp_path, [[1, 9, "n1"], [10, 16777216, "n4"]], rows)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n4"]]


def test_a_moved_count_drops_its_evidence_row(tmp_path):
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    evidence = [[8, 3.0, 2.0], [9, 3.0, 2.0], [10, 3.0, 2.0], [12, 3.0, 2.0]]
    bands, ev = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows, evidence=evidence)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]
    assert ev == [[8, 3.0, 2.0], [12, 3.0, 2.0]]


def test_counts_no_band_covers_are_left_alone(tmp_path):
    # The entry's bands stop at T 9; T 10-11 are not the table's answer and stay uncovered.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    bands, _ = run(tmp_path, [[1, 9, "n1"]], rows)
    assert bands == [[1, 8, "n1"], [9, 9, "n2"]]


if __name__ == "__main__":
    import pathlib
    import tempfile

    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                test(pathlib.Path(d))
            print("ok", name)
