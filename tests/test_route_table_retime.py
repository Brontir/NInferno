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


def test_a_filled_count_the_second_grid_reverses_keeps_the_entry(tmp_path):
    # #1157 review: T 9 and 11 move n1 -> n2 and T 10 is filled; the second grid times T 10 with
    # n2 at 200 against n1's and the ranker's 90. The move would write that regression into the
    # table: the entry is kept.
    rows = [(9, "n1", 200), (9, "n2", 100), (11, "n1", 200), (11, "n2", 100)]
    second = ([(9, "n1", 201), (9, "n2", 99), (11, "n1", 201), (11, "n2", 99),
               (10, "n1", 90), (10, "n2", 200)],
              [(t, "n1") for t in (9, 10, 11)])
    table = [[1, 11, "n1"], [12, 16777216, "n2"]]
    bands, evidence = run(tmp_path, table, rows, evidence=[], second=second)
    assert bands == table and evidence == []


def test_a_filled_count_behind_the_default_keeps_the_entry(tmp_path):
    # As above, but the old route n1 is not timed at T 10 in the second grid: the ranker n4 (90)
    # alone reverses the move there.
    rows = [(9, "n1", 200), (9, "n2", 100), (11, "n1", 200), (11, "n2", 100)]
    second = ([(9, "n1", 201), (9, "n2", 99), (9, "n4", 150), (11, "n1", 201), (11, "n2", 99),
               (11, "n4", 150), (10, "n2", 200), (10, "n4", 90)],
              [(t, "n4") for t in (9, 10, 11)])
    table = [[1, 11, "n1"], [12, 16777216, "n2"]]
    bands, evidence = run(tmp_path, table, rows, evidence=[], second=second)
    assert bands == table and evidence == []


def test_a_filled_counts_dead_old_route_is_no_rival(tmp_path):
    # T 17 and 19 move a dead ggml_mmq band (W4A8 timed: an A16 call cannot run it) to n2; T 18 is
    # filled. The second grid times ggml_mmq at T 18 faster than n2, but there it is dead too, and
    # the W4A8 ranker gives way to the fastest A16 route: the move stands.
    rows = [(t, r, us) for t in (17, 19)
            for r, us in (("n1", 200), ("n2", 100), ("ggml_mmq", 60), ("w4a8.64x64", 50))]
    second = ([(t, r, us) for t in (17, 18, 19)
               for r, us in (("n1", 200), ("n2", 100), ("ggml_mmq", 60), ("w4a8.64x64", 50))],
              [(t, "w4a8.64x64") for t in (17, 18, 19)])
    bands, evidence = run(tmp_path, [[1, 19, "ggml_mmq"], [20, 16777216, "n2"]], rows,
                          evidence=[], second=second, tokens="17-19")
    assert bands == [[1, 16, "ggml_mmq"], [17, 16777216, "n2"]]
    assert evidence == [[17, 100, 100], [18, 100, 100], [19, 100, 100]]


def test_a_dead_count_needs_no_old_route_row_in_the_second_grid(tmp_path):
    # #1157 review: neither grid times the dead ggml_mmq band (W4A8 timed at T 17-19); the move to
    # n2 needs n2's and the default's rows in the second grid, not the old route's.
    rows = [(t, r, us) for t in (17, 19) for r, us in (("n1", 200), ("n2", 100), ("w4a8.64x64", 50))]
    second = ([(t, r, us) for t in (17, 18, 19)
               for r, us in (("n1", 200), ("n2", 100), ("w4a8.64x64", 50))],
              [(t, "w4a8.64x64") for t in (17, 18, 19)])
    bands, evidence = run(tmp_path, [[1, 19, "ggml_mmq"], [20, 16777216, "n2"]], rows,
                          evidence=[], second=second, tokens="17-19")
    assert bands == [[1, 16, "ggml_mmq"], [17, 16777216, "n2"]]
    assert evidence == [[17, 100, 100], [18, 100, 100], [19, 100, 100]]


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


def test_repeat_keeps_an_entry_the_second_grid_shows_as_a_tie(tmp_path):
    # #1205: n2 is 2 % ahead of n1 at T 9-11 in the first grid (past a 1 % margin), but 0.3 % in
    # the second. n2 is within the margin of the second grid's fastest at every count, so each
    # count passes; the move as a whole lowers the second grid's worst regret by 0.3 %, under the
    # margin: kept. Without --repeat it is written.
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 102.0), ("n2", 100.0))]
    second = ([(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 100.3), ("n2", 100.0))],
              [(t, "n1") for t in (9, 10, 11)])
    table = [[1, 11, "n1"], [12, 16777216, "n2"]]
    bands, _ = run(tmp_path, table, rows, margin="0.01", evidence=[], second=second, repeat=True)
    assert bands == table
    bands, _ = run(tmp_path, table, rows, margin="0.01", evidence=[], second=second)
    assert bands == [[1, 8, "n1"], [9, 16777216, "n2"]]


def test_repeat_dead_counts_stay_dead_in_the_second_grid(tmp_path):
    # #1206 review: ggml_mmq is dead at T17 (the first grid timed W4A8 there: the shape admits
    # it). The second grid timed no W4A8, yet ggml_mmq (99.2) is no A16 rival there either. The
    # live move at T 18-19 (n1 1.5 % behind n2 in the second grid, margin 1 %) goes through, and
    # the dead T17 entry moves with it.
    rows = [(17, r, us) for r, us in (("n1", 200), ("n2", 100), ("ggml_mmq", 60), ("w4a8.64x64", 50))]
    rows += [(t, r, us) for t in (18, 19) for r, us in (("n1", 103), ("n2", 100), ("w4a8.64x64", 50))]
    second = ([(17, "n2", 100.0), (17, "ggml_mmq", 99.2)]
              + [(t, r, us) for t in (18, 19) for r, us in (("n1", 101.5), ("n2", 100.0))],
              [(t, "n2") for t in (17, 18, 19)])
    table = [[1, 17, "ggml_mmq"], [18, 19, "n1"], [20, 16777216, "n2"]]
    bands, _ = run(tmp_path, table, rows, margin="0.01", evidence=[], second=second, repeat=True,
                   tokens="17-19")
    assert bands == [[1, 16, "ggml_mmq"], [17, 16777216, "n2"]]


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


def test_an_a8_default_is_no_rival_of_the_table(tmp_path):
    # #1154: the ranker answers W4A8 (100) at T 17-19, faster than any table route; the table
    # stands aside where W4A8 may run, so the move to n2 (the fastest table route) stands and its
    # rows hold n2 against the fastest table route, not against the A8 tile.
    rows = [(t, r, us) for t in (17, 18, 19) for r, us in (("n1", 200), ("n2", 150))]
    second = ([(t, r, us) for t in (17, 18, 19)
               for r, us in (("n1", 200), ("n2", 150), ("w4a8.64x64", 100))],
              [(t, "w4a8.64x64") for t in (17, 18, 19)])
    bands, evidence = run(tmp_path, [[1, 19, "n1"], [20, 16777216, "n2"]], rows, evidence=[],
                          second=second, tokens="17-19")
    assert bands == [[1, 16, "n1"], [17, 16777216, "n2"]]
    assert evidence == [[17, 150, 150], [18, 150, 150], [19, 150, 150]]


def test_ggml_mmq_is_no_candidate_where_w4a8_may_run(tmp_path):
    # #1149 review: at T >= 17 where the grid timed W4A8 (the format and K admit it), an A8 policy
    # never reads the measured entry and an A16 call cannot run ggml_mmq (int8 activations): the
    # range takes the fastest A16 route, n2, not ggml_mmq.
    rows = [(t, r, us) for t in (17, 18, 19)
            for r, us in (("n1", 200), ("n2", 150), ("ggml_mmq", 100), ("w4a8.64x64", 90))]
    bands, _ = run(tmp_path, [[1, 19, "n1"], [20, 16777216, "n2"]], rows, tokens="17-19")
    assert bands == [[1, 16, "n1"], [17, 16777216, "n2"]]


def test_ggml_mmq_stays_a_candidate_without_w4a8(tmp_path):
    # No W4A8 timed (the shape does not admit it): an A8 policy reads the measured entry, and
    # ggml_mmq is its fastest route.
    rows = [(t, r, us) for t in (17, 18, 19) for r, us in (("n1", 200), ("n2", 150), ("ggml_mmq", 100))]
    bands, _ = run(tmp_path, [[1, 19, "n1"], [20, 16777216, "n2"]], rows, tokens="17-19")
    assert bands == [[1, 16, "n1"], [17, 19, "ggml_mmq"], [20, 16777216, "n2"]]


def test_a_count_without_a_ranker_is_a_gap(tmp_path):
    # The second grid timed T 9-11, its selected file names a ranker at T 9 and 11 only: T 10 is a
    # gap and the entry is kept (before, a KeyError).
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))],
              [(9, "n2"), (11, "n2")])
    bands, evidence = run(tmp_path, [[1, 11, "n1"], [12, 16777216, "n2"]], rows, evidence=[],
                          second=second)
    assert bands == [[1, 11, "n1"], [12, 16777216, "n2"]]
    assert evidence == []


def test_a_ranker_the_second_grid_did_not_time_is_a_gap(tmp_path):
    # The ranker names n4, a table route the second grid did not time: no default to hold the move
    # to, the entry is kept (only an A8 ranker is swapped for the fastest table route).
    rows = [(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))]
    second = ([(t, r, us) for t in (9, 10, 11) for r, us in (("n1", 200), ("n2", 100))],
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


def test_round_width_reads_the_token_tile_from_the_name():
    import importlib.util
    spec = importlib.util.spec_from_file_location("route_table_retime", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.round_width("rmma.m1.n4.w1x4") == 32
    assert tool.round_width("rmma.m2.n1.w4x1.c2") == 8
    assert tool.round_width("staged.64x128.w4") == 128
    assert tool.round_width("staged.128x64.w8") == 64
    assert tool.round_width("w4a4.32x64") == 32  # #1157 review: W4A4 names its token tile too
    assert tool.round_width("nvfp4t.c2.128x128") == 128
    assert tool.round_width("i8.64x128") == 64
    assert tool.round_width("tiled.128x64") == 128
    assert tool.round_width("w4a8.32x128") == 32
    assert tool.round_width("ggml_mmq") == 8


def test_a_dead_ggml_mmq_entry_is_replaced(tmp_path):
    # The table's band names ggml_mmq at T 17-19 where the grid timed W4A8: no policy reads it
    # (an A16 call cannot run it, an A8 one takes the predicted band), so the fastest A16 route
    # takes the range, whatever ggml_mmq's own time.
    rows = [(t, r, us) for t in (17, 18, 19)
            for r, us in (("n1", 200), ("n2", 150), ("ggml_mmq", 80), ("w4a8.64x64", 90))]
    bands, _ = run(tmp_path, [[1, 16, "n1"], [17, 19, "ggml_mmq"], [20, 16777216, "n2"]], rows,
                   tokens="17-19")
    assert bands == [[1, 16, "n1"], [17, 16777216, "n2"]]


def test_a_dead_entry_the_grid_never_timed_is_replaced(tmp_path):
    # #1149 re-review P2: the band names ggml_mmq, the grid timed W4A8 but no ggml_mmq row.
    rows = [(t, r, us) for t in (17, 18, 19)
            for r, us in (("n2", 150), ("n4", 160), ("w4a8.64x64", 90))]
    bands, _ = run(tmp_path, [[1, 16, "n1"], [17, 19, "ggml_mmq"], [20, 16777216, "n2"]], rows,
                   tokens="17-19")
    assert bands == [[1, 16, "n1"], [17, 16777216, "n2"]]
