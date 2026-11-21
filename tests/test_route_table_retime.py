"""tools/route_table_retime.py (#1073): one route over the range, untimed table routes kept, no W4A8 route
in a table, a moved count's evidence row dropped or written again from a second grid (--evidence), an
entry kept where the range route would slow a count past the margin."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "route_table_retime.py")
FIELDS = ["format", "compute", "epilogue", "n", "k", "T", "route", "median_us"]


def run(tmp_path, bands, rows, margin="0.03", evidence=None, second=None):
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
    extra = []
    if second is not None:
        # A second grid of the card: its timings and the ranker's answer per count.
        times, rankers = second
        ev = tmp_path / "ev_routes.csv"
        with ev.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for t, route, us in times:
                w.writerow(["q4k", "a8", "store", 4096, 4096, t, route, us])
        sel = tmp_path / "ev_selected.csv"
        with sel.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["format", "compute", "epilogue", "n", "k", "T", "ranker"])
            for t, route in rankers:
                w.writerow(["q4k", "a8", "store", 4096, 4096, t, route])
        extra = ["--evidence", str(ev), str(sel)]
    r = subprocess.run([sys.executable, TOOL, "--table", str(table), "--routes", str(grid),
                        "--tokens", "9-11", "--margin", margin] + extra, capture_output=True, text=True)
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


def test_second_grid_without_the_old_route_keeps_the_entry(tmp_path):
    # n2 wins T 9-11 in the first grid; the second grid times n2 and the ranker n4 at T 10, not n1.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(9, "n1", 200), (9, "n2", 100), (10, "n2", 108), (10, "n4", 100),
               (11, "n1", 200), (11, "n2", 100)],
              [(9, "n1"), (10, "n4"), (11, "n1")])
    bands, evidence = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows, evidence=[],
                          second=second)
    assert bands == [[1, 11, "n1"], [12, 16777216, "n2"]]
    assert evidence == []


def test_second_grid_that_reverses_the_move_keeps_the_entry(tmp_path):
    # n2 wins T 9-11 in the first grid; the second grid has n2 8 % slower than n1 at T 10: kept.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(9, "n1", 200), (9, "n2", 100), (10, "n1", 100), (10, "n2", 108),
               (11, "n1", 200), (11, "n2", 100)], [(t, "n1") for t in (9, 10, 11)])
    bands, evidence = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows, evidence=[],
                          second=second)
    assert bands == [[1, 11, "n1"], [12, 16777216, "n2"]]
    assert evidence == []


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


def test_r8_routes_are_not_candidates(tmp_path):
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 150), ("r8.w4r1", 100))]
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


def test_second_grid_writes_the_moved_counts_evidence(tmp_path):
    # Moved T9-11 n1 -> n2; the second grid's ranker answers n4 there (rk us, n2 us); T12's row stays.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(t, r, us) for t in (9, 10, 11)
               for r, us in (("n1", 200), ("n2", 101.234), ("n4", 150), ("w4a8.32x128", 90))],
              [(t, "n4") for t in (9, 10, 11)])
    bands, ev = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows,
                    evidence=[[9, 300, 200], [12, 300, 100]], second=second)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]
    assert ev == [[9, 150, 101.23], [10, 150, 101.23], [11, 150, 101.23], [12, 300, 100]]


def test_second_grid_without_a_row_keeps_the_entry(tmp_path):
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(t, "n2", 100) for t in (9, 10)], [(t, "n4") for t in (9, 10, 11)])
    bands, ev = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows, evidence=[], second=second)
    assert bands == [[1, 11, "n1"], [12, 16777216, "n2"]] and ev == []


def test_a_range_route_slower_at_a_kept_count_keeps_the_entry(tmp_path):
    # a is 5 % off c at T9-10 (past the margin), so the range is re-timed; a has the least worst
    # regret, but at T11 it is 6 % slower than the table's own b: the entry is kept (#1125, GV100).
    rows = [(9, "a", 100), (9, "c", 95), (9, "b", 200),
            (10, "a", 100), (10, "c", 95), (10, "b", 200),
            (11, "b", 100), (11, "a", 106), (11, "c", 300)]
    bands, r = run(tmp_path, [[1, 10, "a"], [11, 16777216, "b"]], rows)
    assert bands == [[1, 10, "a"], [11, 16777216, "b"]]
    assert "would slow T11 past the margin, kept" in r.stderr


if __name__ == "__main__":
    import pathlib
    import tempfile

    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                test(pathlib.Path(d))
            print("ok", name)
