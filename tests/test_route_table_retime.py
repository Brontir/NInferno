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


def run(tmp_path, bands, rows, margin="0.03", evidence=None, second=None, repeat=False, tuned=None,
        tokens="9-11"):
    table = tmp_path / "t.json"
    entry = {"format": "q4k", "compute": "a16", "epilogue": "store", "n": 4096, "k": 4096, "bands": bands}
    if evidence is not None:
        entry["evidence_us"] = evidence
    if tuned is not None:
        entry["tuned_on"] = tuned
    table.write_text(json.dumps({"gpu": {"class": "ga106"}, "entries": [entry]}, indent=1) + "\n")
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
    if repeat:
        extra.append("--repeat")
    r = subprocess.run([sys.executable, TOOL, "--table", str(table), "--routes", str(grid),
                        "--tokens", tokens, "--margin", margin] + extra, capture_output=True, text=True)
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


def test_untimed_count_between_timed_ones_takes_the_range_route(tmp_path):
    # #1149: the grid timed T 9 and T 11, not T 10; n1's band spans all three, n2 wins both timed
    # counts: T 10 is n2 too (one band), not n1 left between them.
    rows = [(9, "n1", 200), (9, "n2", 100), (11, "n1", 200), (11, "n2", 100)]
    bands, r = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]
    assert "T10: n1 -> n2 (untimed" in r.stdout


def test_a_gap_across_a_round_edge_is_not_filled(tmp_path):
    # Timed T 16 and T 33 only: n4's 32 edge lies between them (T 17-32 one pass, T 33 two), so
    # the untimed T 17-32 keep n4; the timed counts move.
    rows = [(16, "n4", 100), (16, "staged", 102), (33, "n4", 160), (33, "staged", 100)]
    bands, _ = run(tmp_path, [[1, 16777216, "n4"]], rows, tokens="16-33")
    assert bands == [[1, 15, "n4"], [16, 16, "staged"], [17, 32, "n4"], [33, 33, "staged"],
                     [34, 16777216, "n4"]]


def test_repeat_without_evidence_is_refused(tmp_path):
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"entries": []}) + "\n")
    grid = tmp_path / "routes.csv"
    grid.write_text("format,compute,epilogue,n,k,T,route,median_us\n")
    r = subprocess.run([sys.executable, str(TOOL), "--table", str(table), "--routes", str(grid),
                        "--tokens", "9-11", "--repeat"], capture_output=True, text=True)
    assert r.returncode != 0 and "--repeat needs --evidence" in r.stderr


def test_untimed_band_the_grid_never_timed_is_left_alone(tmp_path):
    # T 10's band holds n4, which the grid timed at no count of the range: left as it is.
    rows = [(9, "n1", 200), (9, "n2", 100), (11, "n1", 200), (11, "n2", 100)]
    bands, _ = run(tmp_path, [[1, 9, "n1"], [10, 10, "n4"], [11, 11, "n1"], [12, 16777216, "n2"]], rows)
    assert bands == [[1, 8, "n1"], [9, 9, "n2"], [10, 10, "n4"], [11, 16777216, "n2"]]


def test_filled_count_takes_its_evidence_row_from_the_second_grid(tmp_path):
    # The first grid timed T 9 and 11; the second times T 9-11 and the ranker's answer: T 10 is
    # filled (no first-grid row) and its evidence row comes from the second grid.
    rows = [(9, "n1", 200), (9, "n2", 100), (11, "n1", 200), (11, "n2", 100)]
    second = ([(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 201), ("n2", 99), ("n4", 150))],
              [(t, "n4") for t in (9, 10, 11)])
    bands, evidence = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows, evidence=[],
                          second=second)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]
    assert evidence == [[9, 150.0, 99.0], [10, 150.0, 99.0], [11, 150.0, 99.0]]


def test_repeat_keeps_an_entry_the_second_grid_does_not_repeat(tmp_path):
    # n2 wins T 9-11 in the first grid. In the second, n2 ties n1 at T 10 and n4 (a table route)
    # is 10 % ahead of both: no reversal past the margin, so without --repeat the move is written;
    # with it, n2 is neither within the margin of that grid's fastest nor ahead of n1: kept.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(9, "n1", 200), (9, "n2", 100), (10, "n1", 100), (10, "n2", 100), (10, "n4", 90),
               (11, "n1", 200), (11, "n2", 100)], [(t, "n1") for t in (9, 10, 11)])
    table = [[1, 11, "n1"], [12, 20, "n2"], [21, 16777216, "n4"]]
    bands, _ = run(tmp_path, table, rows, evidence=[], second=second, repeat=True)
    assert bands == table
    bands, _ = run(tmp_path, table, rows, evidence=[], second=second)
    assert bands == [[1, 8, "n1"], [9, 20, "n2"], [21, 16777216, "n4"]]


def test_repeat_keeps_a_large_gain_the_second_grid_puts_a_little_behind(tmp_path):
    # The second grid has n4 3 % ahead of n2 at T 10, but n2 still beats the old n1 by 2x there: the
    # move repeats as a gain and is written.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(9, "n1", 200), (9, "n2", 100), (10, "n1", 200), (10, "n2", 100), (10, "n4", 95),
               (11, "n1", 200), (11, "n2", 100)], [(t, "n1") for t in (9, 10, 11)])
    bands, _ = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows, evidence=[],
                   second=second, repeat=True)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]


def test_a_borrowed_entry_moves_but_keeps_its_borrowed_rows(tmp_path):
    # tuned_on another board's class: the move is written, this board's rows are not.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = (rows, [(t, "n1") for t in (9, 10, 11)])
    bands, evidence = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows,
                          evidence=[[8, 5.0, 4.0]], second=second, tuned="ga102:82sm")
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]
    assert evidence == [[8, 5.0, 4.0]]


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


def test_a_move_that_does_not_lower_the_worst_regret_is_kept(tmp_path):
    # T 9-10 n1 the fastest; T 11 n2 4 % behind n4, n1 7 % behind: n1 over the range would trade
    # T 11's 4 % for 7 %, inside the slow-down margin (104 -> 107 us).
    rows = ([(t, r, us) for t in (9, 10) for r, us in (("n1", 100), ("n2", 108), ("n4", 110))]
            + [(11, "n1", 107), (11, "n2", 104), (11, "n4", 100)])
    bands, _ = run(tmp_path, [[1, 10, "n1"], [11, 16777216, "n2"]], rows)
    assert bands == [[1, 10, "n1"], [11, 16777216, "n2"]]


def test_a_count_the_move_leaves_alone_does_not_block_it(tmp_path):
    # T 9-15 n1 10 % behind n2; T 16 n2 8 % behind n4 (n1 far behind there): n2 over the range
    # clears T 9-15 and leaves T 16 as it is. T 16's 8 % is on both sides: the move stands.
    rows = ([(t, r, us) for t in range(9, 16) for r, us in (("n1", 110), ("n2", 100), ("n4", 300))]
            + [(16, "n1", 300), (16, "n2", 108), (16, "n4", 100)])
    bands, r = run(tmp_path, [[1, 15, "n1"], [16, 16777216, "n2"]], rows, tokens="9-16")
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]], r.stderr


def test_the_table_route_already_least_worst_is_left_alone(tmp_path):
    # n1 5 % behind at T 9, far ahead at T 10-11: n1 is the range's route, nothing moves (rc 0).
    rows = [(9, "n1", 105), (9, "n2", 100), (10, "n1", 100), (10, "n2", 120),
            (11, "n1", 100), (11, "n2", 120)]
    bands, _ = run(tmp_path, [[1, 16777216, "n1"]], rows)
    assert bands == [[1, 16777216, "n1"]]


def test_second_grid_with_the_default_far_ahead_keeps_the_entry(tmp_path):
    # n2 wins T 9-11 in the first grid; in the second, n2 beats n1 (150 vs 200) but the ranker's
    # n4 takes 100 at T 10: the moved entry would ship 1.5x the default there, kept.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(9, "n1", 200), (9, "n2", 100), (9, "n4", 100), (10, "n1", 200), (10, "n2", 150),
               (10, "n4", 100), (11, "n1", 200), (11, "n2", 100), (11, "n4", 100)],
              [(t, "n4") for t in (9, 10, 11)])
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
