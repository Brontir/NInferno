import contextlib
import io
import json
import random
import re
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import mmq_width_grid as g


TEMPLATE = 12  # the chat template's own tokens for Qwen3 with thinking off (user turn 5, header 3, empty think block 4)
SALT_TOKENS = 7  # "w" + six digits, one token per digit


class FakeServe(BaseHTTPRequestHandler):
    """A stand-in for ninferno-serve: counts a prompt as 12 template tokens + 7 for the salt word + 1 per
    word + 1 for a trailing space, answers with usage and timings like the non-streaming chat response."""

    skew = 0  # tokens added on top of the true count (a mismatching size) ...
    skew_from = 0  # ... from this POST on (0 = every request; a later value = after the calibration)
    cache_n = 0
    prompt_n_delta = 0  # prompt_n differs from usage.prompt_tokens by this
    drop_timings = False
    prompt_ms = None
    fail = False
    seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.reply({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        index = len(type(self).seen)
        type(self).seen.append(body)
        if type(self).fail:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"ggml mmq: no tile fits this device")
            return
        content = body["messages"][0]["content"]
        words = content.split()
        n = TEMPLATE + SALT_TOKENS + len(words) - 1 + (1 if content.endswith(" ") else 0)
        if index >= type(self).skew_from:
            n += type(self).skew
        out = {"usage": {"prompt_tokens": n}}
        if not type(self).drop_timings:
            ms = type(self).prompt_ms if type(self).prompt_ms is not None else 10.0 + n / 10
            out["timings"] = {"cache_n": type(self).cache_n, "prompt_n": n + type(self).prompt_n_delta,
                              "prompt_ms": ms}
        self.reply(out)

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows) if not isinstance(rows, str) else rows)
        return str(path)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        old = sys.argv
        sys.argv = ["mmq_width_grid.py", *argv]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = g.main()
        finally:
            sys.argv = old
        return rc, out.getvalue(), err.getvalue()


def row(label, n, ms, pn=None, k=7, **extra):
    return {"label": label, "n": n, "k": k, "ms": ms, "prompt_n": pn if pn is not None else [n] * len(ms), **extra}


def skipped(label, n, k=7):
    return {"label": label, "n": n, "k": k, "skipped": "cannot build"}


class Measure(Base):
    def setUp(self):
        super().setUp()
        FakeServe.skew, FakeServe.skew_from, FakeServe.cache_n, FakeServe.prompt_n_delta = 0, 0, 0, 0
        FakeServe.drop_timings, FakeServe.prompt_ms, FakeServe.fail, FakeServe.seen = False, None, False, []
        self.srv = HTTPServer(("127.0.0.1", 0), FakeServe)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.port = self.srv.server_address[1]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        super().tearDown()

    def measure(self, out, sizes="24,80,128", *extra):
        return self.run_main("measure", "--port", str(self.port), "--label", "J80", "--out", str(out),
                             "--sizes", sizes, "--reps", "2", "--warmup", "1", *extra)

    def test_exact_sizes_thinking_off_and_prompt_n_recorded(self):
        out = self.root / "m.jsonl"
        rc, stdout, _ = self.measure(out)
        self.assertEqual(rc, 0, stdout)
        rows = [json.loads(s) for s in out.read_text().splitlines()]
        self.assertEqual([r["n"] for r in rows], [24, 80, 128])
        for r in rows:
            self.assertEqual(r["prompt_n"], [r["n"]] * 2)
            self.assertEqual(r["k"], 7)
        self.assertTrue(all(b["enable_thinking"] is False and b["max_tokens"] == 1 and "stream" not in b
                            for b in FakeServe.seen))

    def test_warmup_comes_first_and_the_floor_is_measured_on_an_empty_body(self):
        # --warmup 1: one 120-word request first; then the calibration of the first size starts with the
        # salt word alone (no body word), so the template's own count is measured, not inferred
        self.assertEqual(self.measure(self.root / "m.jsonl", "80")[0], 0)
        words = [len(b["messages"][0]["content"].split()) for b in FakeServe.seen]
        self.assertEqual(words[0], 1 + 120)
        self.assertEqual(words[1], 1)
        self.assertEqual(words[2], 1 + 80 - (TEMPLATE + SALT_TOKENS + 1))  # first guess: n - the empty prompt (20)

    def test_salt_is_seven_tokens_wide_and_never_repeats(self):
        # literal widths, not g.SALT_DIGITS: the fake counts one token a digit, a narrower salt must fail here
        self.assertEqual(self.measure(self.root / "m.jsonl")[0], 0)
        salts = [b["messages"][0]["content"].split()[0] for b in FakeServe.seen]
        self.assertTrue(all(len(s) == SALT_TOKENS and s[0] == "w" and s[1:].isdigit() for s in salts), salts)
        self.assertEqual(len(set(salts)), len(salts), "a salt was used twice")

    def test_salt_stays_fixed_width_when_it_starts_at_zero(self):
        with mock.patch.object(g.time, "time", return_value=0.0):
            rc, out, err = self.measure(self.root / "m.jsonl")
        self.assertEqual(rc, 0, err)
        self.assertTrue(all(len(b["messages"][0]["content"].split()[0]) == SALT_TOKENS for b in FakeServe.seen))

    def test_unbuildable_size_is_skipped_and_reported_not_fatal(self):
        # 12 + 7 + 1 = 20 tokens is the least a prompt can have: 16 cannot be built
        out = self.root / "m.jsonl"
        rc, stdout, err = self.measure(out, "16,80")
        self.assertEqual(rc, 0, err)
        self.assertIn("N=16: skipped", err)
        self.assertIn("1 skipped [16]", stdout)
        rows = [json.loads(s) for s in out.read_text().splitlines()]
        self.assertEqual([("skipped" in r, r["n"]) for r in rows], [(True, 16), (False, 80)])
        self.assertEqual(rows[1]["prompt_n"], [80, 80])

    def test_grid_sizes_are_all_buildable_with_the_real_template(self):
        self.assertGreaterEqual(min(g.SIZES), TEMPLATE + SALT_TOKENS + 1)
        rc, _, err = self.measure(self.root / "m.jsonl", "grid", "--k", "7")
        self.assertEqual(rc, 0, err)

    def test_refuses_a_non_empty_output(self):
        out = self.root / "m.jsonl"
        out.write_text("stale\n")
        rc, _, err = self.measure(out)
        self.assertEqual(rc, 1)
        self.assertIn("not empty", err)
        self.assertEqual(out.read_text(), "stale\n")

    def test_server_error_is_readable(self):
        FakeServe.fail = True
        rc, _, err = self.measure(self.root / "m.jsonl")
        self.assertEqual(rc, 1)
        self.assertIn("no tile fits this device", err)
        self.assertNotIn("Traceback", err)

    def test_a_count_that_skews_after_the_calibration_is_rejected(self):
        # 1 warmup + 2 calibration requests, then the server starts to count one token more
        FakeServe.skew, FakeServe.skew_from = 1, 3
        out = self.root / "m.jsonl"
        rc, _, err = self.measure(out, "80")
        self.assertEqual(rc, 1)
        self.assertIn("had 81 prompt tokens, rejected", err)
        self.assertIn("size not written", err)
        self.assertFalse(out.exists() and out.read_text())

    def test_a_cache_hit_is_refused(self):
        FakeServe.cache_n = 5  # prompt_n stays equal to prompt_tokens: only cache_n tells
        rc, _, err = self.measure(self.root / "m.jsonl")
        self.assertEqual(rc, 1)
        self.assertIn("prefix cache", err)

    def test_a_prompt_n_that_differs_from_the_usage_is_refused(self):
        FakeServe.prompt_n_delta = -3  # cache_n stays 0: only prompt_n tells
        rc, _, err = self.measure(self.root / "m.jsonl")
        self.assertEqual(rc, 1)
        self.assertIn("prefix cache", err)

    def test_missing_timings_are_refused(self):
        FakeServe.drop_timings = True
        rc, _, err = self.measure(self.root / "m.jsonl")
        self.assertEqual(rc, 1)
        self.assertIn("no usage.prompt_tokens / timings", err)
        self.assertNotIn("Traceback", err)

    def test_a_zero_prompt_ms_is_refused(self):
        FakeServe.prompt_ms = 0.0
        rc, _, err = self.measure(self.root / "m.jsonl")
        self.assertEqual(rc, 1)
        self.assertIn("no first token", err)

    def test_k_and_f_are_recorded(self):
        out = self.root / "m.jsonl"
        self.assertEqual(self.measure(out, "80", "--k", "9", "--f", "0.2")[0], 0)
        r = json.loads(out.read_text())
        self.assertEqual((r["k"], r["f"]), (9, 0.2))


class Readers(Base):
    def grid_files(self, sizes=(96, 128, 160), skip=None, k=7):
        files = []
        for j in g.WIDTHS:
            rows = [row(f"J{j}", n, [1.0 + n / 100 + abs(j - 96) / 500] * 4, k=k) for n in sizes
                    if (j, n) != skip]
            files.append(self.write(f"grid_J{j}.jsonl", rows))
        return files

    def model_files(self, sizes, k=7, f=0.2, nudge=None, base=5.0):
        """Times that follow the work rule at f exactly, in columns = N - k."""
        files = []
        for j in g.WIDTHS:
            rows = [row(f"J{j}", n, [base + g.work(n - k, j, f) + (nudge or {}).get((j, n), 0.0)] * 4, k=k)
                    for n in sizes]
            files.append(self.write(f"grid_J{j}.jsonl", rows))
        return files

    def test_grid_reports_counts_and_a_least(self):
        rc, out, err = self.run_main("grid", *self.grid_files())
        self.assertEqual(rc, 0, err)
        self.assertIn("J96: 3 sizes, 3 visits, 12 requests", out)
        self.assertIn("regret at the assumed f 0.20", out)

    def test_grid_prints_assumption_criteria_buckets_and_the_prediction_beside_the_measurement(self):
        rc, out, err = self.run_main("grid", *self.grid_files(sizes=g.SIZES))
        self.assertEqual(rc, 0, err)
        for text in ("ASSUMPTION under test: f = 0.20", "columns = tokens - k, k = 7", "REFUTATION CRITERIA",
                     "R1 grid", "R2 grid", "R3 grid", "<- ASSUMED f 0.20", "lower edge decided by",
                     "fewest tiles", "difference", "spread over the buckets", "differs from the measured best in",
                     "material mismatch", "verdict on the criteria", "predicted  best"):
            self.assertIn(text, out)

    def test_the_prediction_uses_columns_not_tokens(self):
        # times that follow the rule in N - 7 columns: the prediction at f 0.20 matches everywhere, sizes at
        # and just above a tile edge included (136 tokens = 129 columns = two tiles of 80 / 96 / 128)
        sizes = [96, 128, 130, 136, 142, 160, 262, 264, 270]
        rc, out, err = self.run_main("grid", *self.model_files(sizes))
        self.assertEqual(rc, 0, err)
        self.assertIn("differs from the measured best in 0 of 9 sizes", out)
        self.assertIn("material mismatch): 0", out)
        self.assertIn("R5: not met", out)

    def test_a_card_whose_f_is_far_from_the_assumption_refutes_it(self):
        # times that follow the rule at f 0.05: the fit measures 0.05; the regret report and its mean pinned
        rc, out, err = self.run_main("grid", *self.model_files(g.SIZES, f=0.05, base=0.0))
        self.assertEqual(rc, 0, err)
        self.assertIn("R5: MET", out)
        self.assertIn("is REFUTED", out)
        self.assertIn("regret at the assumed f 0.20: 0.87 %", out)  # the mean over the sizes, not the worst
        self.assertIn("R1 (report, not a verdict): difference 0.87 pp", out)

    def test_a_wrong_pick_shows_as_a_mismatch(self):
        # at 142 tokens the width predicted at f 0.20 is far slower than another: 1 of 4 sizes is exactly
        # the 25 % share, which R2 does not exceed
        files = self.model_files([96, 128, 142, 160], nudge={(80, 142): 40.0, (96, 142): 40.0})
        rc, out, _ = self.run_main("grid", *files)
        self.assertEqual(rc, 0)
        self.assertIn("MISMATCH", out)
        self.assertIn("material mismatch): 1", out)
        self.assertIn("R2: not met  (1 of 4 sizes against 1.0)", out)

    # --- the law itself, over the kit's own sizes (the reviewer's synthetic grids) -----------------------
    def law_files(self, f, sizes=None, C=42.0, L=53.0, k=7, noise=0.0, seed=1, grows=0.0, visits=1, drift=0.0,
                  shape=None, outlier=False, lin=0.0, offset=0.0):
        """T = C + grows x N + L x tiles x (f + (1 - f) J / 128) at every width (computed here, not with
        the module's functions), 4 requests a cell, each times (1 + uniform(-noise / 2, noise / 2)); one
        file per width visit, every visit's times scaled by its own (1 + uniform(-drift, drift)) (a server
        run that drifted), plus `lin` x (the run's place in the script's order, 16..128 then 128..16) / (the
        number of runs - 1) (a linear drift in time), and an offset of uniform(-offset, offset) ms per run;
        `shape(n, j, work_ms)` replaces the law's work term (a misfit); `outlier`: the last request of
        every third cell at 5x (a median does not see it, a mean does)."""
        rng = random.Random(seed)
        files = []
        nw = len(g.WIDTHS)
        for v in range(visits):
            for p, j in enumerate(g.WIDTHS):
                place = v * nw + (p if v % 2 == 0 else nw - 1 - p)
                run = 1 + rng.uniform(-drift, drift) + (lin * place / (visits * nw - 1) if lin else 0.0)
                off = rng.uniform(-offset, offset) if offset else 0.0
                rows = []
                for i, n in enumerate(sizes or g.SIZES):
                    w = L * -(-(n - k) // j) * (f + (1 - f) * j / 128)
                    t = (C + grows * n + (shape(n, j, w) if shape else w)) * run + off
                    ms = [t * (1 + rng.uniform(-noise / 2, noise / 2)) for _ in range(4)]
                    if outlier and i % 3 == 0:
                        ms[3] *= 5
                    rows.append(row(f"J{j}", n, ms, k=k))
                files.append(self.write(f"grid_J{j}_v{v}.jsonl", rows))
        return files

    def fitted(self, out):
        m = re.search(r"measured f = a / b = ([0-9.]+), 95 % bootstrap interval \[([0-9.]+), ([0-9.]+)\]", out)
        self.assertIsNotNone(m, out)
        return tuple(float(x) for x in m.groups())

    def test_the_exact_law_at_the_assumed_f_over_the_kit_sizes_stands(self):
        # the old reader REFUTED the truth here: J 48 / J 112 beat the six production widths at 6 sizes
        rc, out, err = self.run_main("grid", *self.law_files(0.20))
        self.assertEqual(rc, 0, err)
        self.assertIn("f 0.20 stands", out)
        self.assertNotIn("REFUTED", out)
        self.assertIn("over 168 cells (21 sizes x 8 widths x 1 visits", out)  # J 48 / J 112 are fitted too
        self.assertIn("R2: not met  (0 of 21 sizes", out)
        self.assertIn("R5: not met", out)
        self.assertEqual(self.fitted(out), (0.2, 0.2, 0.2))
        self.assertIn("residual rms 0.00 % of the size's time, lack-of-fit ratio 0.00", out)  # the law, with tiles in columns, fits exactly
        # the two extra widths are shown, as information only
        self.assertIn("J 48 / J 112 beat the best production width at 7 sizes", out)

    def test_the_exact_law_away_from_the_assumption_is_refuted_with_the_true_f(self):
        # 0.10 and 0.30 were "stands" or refuted for the wrong reason before; the fit measures them
        # outside the predicted range 0.15..0.25: the whole (zero-width) interval lies outside it
        for f in (0.05, 0.10, 0.12, 0.14, 0.26, 0.30, 0.43):
            rc, out, err = self.run_main("grid", *self.law_files(f))
            self.assertEqual(rc, 0, err)
            self.assertIn("f 0.20 is REFUTED", out, f)
            self.assertIn("R5: MET", out, f)
            fit, lo, hi = self.fitted(out)
            self.assertAlmostEqual(fit, f, delta=0.01)
            self.assertTrue(lo <= f + 1e-3 and f - 1e-3 <= hi, (f, lo, hi))
            self.assertIn("residual rms 0.00 % of the size's time, lack-of-fit ratio 0.00", out)

    def test_a_noisy_law_recovers_f(self):
        # 2 % request noise (uniform), median of four, fixed seed: f recovered, the interval holds it
        for f, verdict in ((0.12, "f 0.20 is REFUTED"), (0.20, "f 0.20 stands"), (0.43, "f 0.20 is REFUTED")):
            rc, out, err = self.run_main("grid", *self.law_files(f, noise=0.02, seed=7))
            self.assertEqual(rc, 0, err)
            fit, lo, hi = self.fitted(out)
            self.assertAlmostEqual(fit, f, delta=0.01)
            self.assertTrue(lo <= f <= hi, (f, lo, hi))
            self.assertLess(hi - lo, 0.05)
            self.assertIn(verdict, out)

    def test_a_width_independent_part_that_grows_with_n_does_not_move_f(self):
        # the rest of the prefill grows with N; one constant per size keeps it out of b
        rc, out, err = self.run_main("grid", *self.law_files(0.12, grows=0.3))
        self.assertEqual(rc, 0, err)
        self.assertAlmostEqual(self.fitted(out)[0], 0.12, delta=1e-3)

    def test_the_fit_counts_tiles_in_columns(self):
        # at k 7 the kit's sizes have the same tile counts in tokens and in columns (the margins); at k 40
        # they do not, and only a fit in columns = N - k recovers the law exactly
        sizes = [n for n in g.SIZES if n >= 64]
        rc, out, err = self.run_main("grid", *self.law_files(0.12, sizes=sizes, k=40))
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fitted(out), (0.12, 0.12, 0.12))
        self.assertIn("residual rms 0.00 % of the size's time, lack-of-fit ratio 0.00", out)

    def test_the_fit_is_deterministic(self):
        # the same cells give the same interval to the last bit (a fixed bootstrap seed)
        rng = random.Random(3)
        cells = [(n, *g.law_terms(n - 7, j), (42 + 53 * -(-(n - 7) // j) * (0.2 + 0.8 * j / 128))
                  * (1 + rng.uniform(-0.01, 0.01))) for n in g.SIZES for j in g.WIDTHS]
        blocks = [j for n in g.SIZES for j in g.WIDTHS]
        rel = [1.0] * len(cells)
        first, second = g.fit_f(cells, blocks, rel), g.fit_f(cells, blocks, rel)
        self.assertTrue(first["ok"])
        self.assertEqual(first, second)
        self.assertLess(first["lo"], first["hi"])

    def test_widths_that_do_not_matter_give_no_gain_and_no_f(self):
        # every width as fast as every other: the table gains nothing over the fewest tiles (R3, equal
        # counts as no gain) and the time does not grow with the width (no f)
        files = [self.write(f"grid_J{j}.jsonl", [row(f"J{j}", n, [50.0 + n / 10] * 4) for n in g.SIZES])
                 for j in g.WIDTHS]
        rc, out, err = self.run_main("grid", *files)
        self.assertEqual(rc, 0, err)
        self.assertIn("R3: MET", out)
        self.assertIn("R5: NO VERDICT on f  (no f:", out)
        self.assertIn("is REFUTED", out)

    def test_noise_is_the_median_range_of_a_cell(self):
        # 13 of 24 cells spread 1 %, 11 spread 50 %: the median is 1.0 %, a mean would be about 23 %
        files, i = [], 0
        for j in g.WIDTHS:
            rows = []
            for n in (96, 128, 160):
                s = 0.01 if i < 13 else 0.5
                i += 1
                rows.append(row(f"J{j}", n, [10.0, 10.0, 10.0 * (1 + s), 10.0]))
            files.append(self.write(f"grid_J{j}.jsonl", rows))
        rc, out, err = self.run_main("grid", *files)
        self.assertEqual(rc, 0, err)
        self.assertIn("noise (median range of one cell's requests) 1.0 %", out)

    def test_buckets_have_one_table_each_and_an_edge_belongs_to_the_bucket_below(self):
        cols = [n - 7 for n in g.SIZES]
        bk = g.buckets(cols, g.ASSUMED_F)
        tables = [b["table"] for b in bk]
        self.assertEqual(len(set(tables)), len(tables))
        self.assertEqual(bk[0]["lo"], 0.0)
        self.assertEqual(bk[-1]["hi"], 1.0)
        assumed = [b for b in bk if b["assumed"]]
        self.assertEqual(len(assumed), 1)
        self.assertEqual(assumed[0]["rep"], g.ASSUMED_F)
        self.assertAlmostEqual(assumed[0]["hi"], 0.2, places=6)  # the 5 x 96 / 4 x 128 tie sits at f 0.20
        for b in bk:
            self.assertEqual(g.pick_table(cols, b["rep"]), b["table"])

    def test_a_tie_goes_to_the_narrower_width(self):
        self.assertEqual(g.pick(450, 0.2, g.PRODUCTION), 96)  # 5 x 96 and 4 x 128 cost the same at f 0.20
        self.assertEqual(g.pick(450, 0.21, g.PRODUCTION), 128)
        self.assertEqual(g.pick(129, None, g.OLD), 128)
        self.assertEqual(g.pick(80, 0.2, g.PRODUCTION), 80)

    def test_every_size_keeps_a_margin_from_a_tile_edge(self):
        for n in set(g.SIZES + g.AB_SIZES):
            self.assertGreaterEqual(g.margins(n, g.K_ASSUMED), g.MARGIN_WARN, n)
        self.assertEqual(g.margins(136, 7), 0)  # the old size, as the reviewer found it
        self.assertEqual(g.margins(264, 7), 0)
        self.assertNotIn(136, g.SIZES + g.AB_SIZES)
        self.assertNotIn(264, g.AB_SIZES)

    def test_the_mmq_gate_is_a_margin_too(self):
        # 24 tokens = 17 columns sits on the 17-column gate: one more token of k leaves ggml's matrix kernel
        self.assertEqual(g.margins(24, 7), 0)
        self.assertEqual(g.margins(32, 7), 8)  # 25 columns: 8 above the gate, far from 64
        self.assertLess(g.margins(23, 7), 0)
        self.assertNotIn(24, g.SIZES)
        self.assertEqual(len(g.SIZES), 21)
        self.assertGreaterEqual(min(g.SIZES) - g.K_ASSUMED - g.MMQ_GATE, g.MARGIN_WARN)

    def test_predicted_ranges_hold_their_edges(self):
        self.assertIsNone(g.pred_range(128))
        for c, want in ((129, (-19.0, -14.0)), (168, (-19.0, -14.0)), (169, (-13.0, -9.0)), (200, (-13.0, -9.0)),
                        (201, (-0.5, 0.5)), (260, (-0.5, 0.5))):
            self.assertEqual(g.pred_range(c), want, c)
        self.assertIsNone(g.pred_range(261))

    def test_the_hook_default_f_is_the_assumed_f(self):
        patch = (Path(__file__).resolve().parents[1] / "tools" / "bench" / "524_grid_hook.patch").read_text()
        m = re.search(r'getenv\("NINFERNO_MMQ_F"\)\) : ([0-9.]+);', patch)
        self.assertIsNotNone(m)
        self.assertEqual(float(m.group(1)), g.ASSUMED_F)

    def test_grid_refuses_stale_rows(self):
        files = self.grid_files()
        files.append(self.write("stale.jsonl", [row("J80", 96, [9.0] * 4)]))
        rc, _, err = self.run_main("grid", *files)
        self.assertEqual(rc, 1)
        self.assertIn("stale rows", err)

    def test_grid_refuses_a_missing_width(self):
        files = [f for f in self.grid_files() if "J80" not in f]
        rc, _, err = self.run_main("grid", *files)
        self.assertEqual(rc, 1)
        self.assertIn("no rows for ['J80']", err)

    def test_grid_refuses_a_size_missing_at_one_width(self):
        rc, _, err = self.run_main("grid", *self.grid_files(skip=(80, 128)))
        self.assertEqual(rc, 1)
        self.assertIn("incomplete campaign", err)

    def test_grid_takes_a_size_skipped_at_every_width(self):
        files = self.grid_files()
        files += [self.write(f"sk{j}.jsonl", [skipped(f"J{j}", 24)]) for j in g.WIDTHS]
        rc, out, err = self.run_main("grid", *files)
        self.assertEqual(rc, 0, err)
        self.assertIn("skipped sizes (could not be built): [24]", out)

    def test_grid_refuses_a_size_skipped_at_one_width_only(self):
        files = self.grid_files(sizes=(96, 128, 160, 24), skip=(80, 24))
        files.append(self.write("sk.jsonl", [skipped("J80", 24)]))
        rc, _, err = self.run_main("grid", *files)
        self.assertEqual(rc, 1)
        self.assertIn("incomplete campaign", err)

    def test_files_with_different_k_or_without_k_are_refused(self):
        files = self.grid_files()
        files.append(self.write("k.jsonl", [row("J16", 24, [1.0], k=8)]))
        rc, _, err = self.run_main("grid", *files)
        self.assertEqual(rc, 1)
        self.assertIn("different k", err)
        nok = {"label": "J16", "n": 16, "ms": [1.0], "prompt_n": [16]}
        rc, _, err = self.run_main("grid", self.write("nok.jsonl", [nok]))
        self.assertEqual(rc, 1)
        self.assertIn("not a measurement row", err)

    def test_empty_partial_and_missing_files(self):
        for name, text, word in (("empty.jsonl", "", "empty"),
                                 ("partial.jsonl", '{"label": "J16", "n": 16, "ms": [1.0, ', "partial"),
                                 ("nokey.jsonl", '{"label": "J16"}\n', "not a measurement row")):
            path = self.write(name, text)
            for cmd in ("grid", "ab"):
                rc, _, err = self.run_main(cmd, path)
                self.assertEqual(rc, 1, (cmd, name))
                self.assertIn(word, err)
                self.assertNotIn("Traceback", err)
        rc, _, err = self.run_main("grid", str(self.root / "nope.jsonl"))
        self.assertEqual((rc, "no such file" in err), (1, True))

    def test_mismatched_prompt_n_is_rejected(self):
        path = self.write("bad.jsonl", [row("J16", 128, [1.0, 1.0], pn=[128, 129])])
        rc, _, err = self.run_main("grid", path)
        self.assertEqual(rc, 1)
        self.assertIn("not N tokens", err)
        old = self.write("old.jsonl", [{"label": "J16", "n": 16, "k": 7, "ms": [1.0]}])
        rc, _, err = self.run_main("grid", old)
        self.assertIn("no prompt_n", err)

    def ab_files(self, drop_new=None, visits=2, new_f=0.2, k=7, scale=0.9, sizes=None):
        files = []
        for v in range(visits):
            for lab, kk in (("old", 1.0), ("new", scale)):
                extra = {"f": new_f} if lab == "new" else {}
                rows = [row(lab, n, [kk * (n / 10 + v * 0.1)] * 2, k=k, **extra) for n in (sizes or g.MIX)
                        if n != drop_new or lab == "old"]
                files.append(self.write(f"ab_{v}_{lab}.jsonl", rows))
        return files

    def test_ab_prints_spread_mix_and_the_prediction_beside_it(self):
        rc, out, err = self.run_main("ab", *self.ab_files())
        self.assertEqual(rc, 0, err)
        for text in ("[", "per visit", "-10.0 %", "predicted range", "OUTSIDE", "measured change outside",
                     "predicted mix change", "R4:", "model work"):
            self.assertIn(text, out)

    def test_ab_size_missing_from_new_is_refused(self):
        rc, _, err = self.run_main("ab", *self.ab_files(drop_new=160))
        self.assertEqual(rc, 1)
        self.assertIn("incomplete campaign", err)

    def test_ab_without_the_whole_mix_is_refused(self):
        rc, _, err = self.run_main("ab", *self.ab_files(sizes=[144, 160, 176]))
        self.assertEqual(rc, 1)
        self.assertIn("mix sizes", err)

    def test_ab_unequal_visits_refused(self):
        files = self.ab_files()
        rc, _, err = self.run_main("ab", *files[:-1])
        self.assertEqual(rc, 1)
        self.assertIn("visits", err)

    def test_ab_at_another_f_gives_no_verdict(self):
        rc, out, _ = self.run_main("ab", *self.ab_files(new_f=0.1))
        self.assertEqual(rc, 0)
        self.assertIn("do NOT apply", out)
        self.assertNotIn("R4:", out)

    def test_kprobe_finds_the_step(self):
        # the call crosses 128 columns between N=134 and N=135, so k = 6
        rows = [row("J128", n, [10.0 if n <= 134 else 14.0 + (n - 135) * 0.01] * 3) for n in range(128, 147)]
        rc, out, err = self.run_main("kprobe", self.write("p.jsonl", rows))
        self.assertEqual(rc, 0, err)
        self.assertIn("k lies in [6, 6]", out)
        self.assertIn("exact", out)
        self.assertEqual(out.strip().splitlines()[-1], "KPROBE k_lo=6 k_hi=6 trusted=1")

    def test_kprobe_gives_an_interval_for_spaced_sizes(self):
        # sizes two apart: the step between 134 and 136 puts the crossing at 135 or 136, k in [6, 7]
        rows = [row("J128", n, [10.0 if n <= 134 else 14.0] * 3) for n in range(128, 147, 2)]
        rc, out, err = self.run_main("kprobe", self.write("p.jsonl", rows))
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.strip().splitlines()[-1], "KPROBE k_lo=6 k_hi=7 trusted=1")

    def test_kprobe_does_not_trust_a_step_among_steps(self):
        # two steps of 10 % and 9 %: neither is an edge
        ms = {n: 10.0 for n in range(128, 147)}
        for n in range(135, 147):
            ms[n] = 11.0
        for n in range(140, 147):
            ms[n] = 11.0 * 1.09
        rows = [row("J128", n, [ms[n]] * 3) for n in range(128, 147)]
        rc, out, err = self.run_main("kprobe", self.write("p.jsonl", rows))
        self.assertEqual(rc, 0, err)
        self.assertIn("do not trust this interval", out)
        self.assertEqual(out.strip().splitlines()[-1], "KPROBE k_lo=6 k_hi=6 trusted=0")

    # --- the prediction is a range; drift, misfit and the bootstrap (the second review) -----------------
    def ratio(self, out):
        m = re.search(r"residual rms ([0-9.]+) % of the size's time, lack-of-fit ratio ([0-9.]+|inf)", out)
        self.assertIsNotNone(m, out)
        return float(m.group(1)), float(m.group(2))

    def test_the_exact_law_inside_the_predicted_range_stands(self):
        for f in (0.15, 0.18, 0.22, 0.25):
            rc, out, err = self.run_main("grid", *self.law_files(f))
            self.assertEqual(rc, 0, err)
            self.assertIn("f 0.20 stands", out, f)
            self.assertIn("R5: not met", out, f)
            self.assertAlmostEqual(self.fitted(out)[0], f, delta=1e-3)

    def test_a_drift_of_every_width_run_does_not_refute_the_truth(self):
        # the law at f 0.20, 2 % request noise, each of the 16 width runs (two visits) off by up to +-1 %:
        # the point test refuted 0.20 in most of such grids; the range test must not
        for seed in range(4):
            rc, out, err = self.run_main("grid", *self.law_files(0.20, noise=0.02, visits=2, drift=0.01, seed=seed))
            self.assertEqual(rc, 0, err)
            self.assertIn("visits per width: 2", out)
            self.assertIn("(1000 draws over the 16 width visits", out)  # the unit is the width run
            self.assertIn("f 0.20 stands", out, seed)
            self.assertLessEqual(self.ratio(out)[1], g.LOF_MAX)

    def test_a_larger_drift_between_server_runs_is_no_misfit_of_the_law(self):
        # the third review: the law at exactly f 0.20 with a drift BETWEEN the server runs (each width visit
        # is one server start) read "the work law does not describe this card" in most seeds, because the
        # ratio divided a run's drift by the noise within a run. With one fitted offset per run it stands.
        cases = (("per-run scale +-2 %", dict(drift=0.02)), ("linear +3 % over the 16 runs", dict(lin=0.03)),
                 ("1 % request noise, per-run +-1 %", dict(noise=0.01, drift=0.01)))
        for name, kw in cases:
            for seed in range(4):
                args = dict(noise=0.02, visits=2, seed=seed)
                args.update(kw)
                rc, out, err = self.run_main("grid", *self.law_files(0.20, **args))
                self.assertEqual(rc, 0, err)
                self.assertIn("f 0.20 stands", out, (name, seed))
                self.assertNotIn("NO VERDICT", out, (name, seed))
                self.assertLessEqual(self.ratio(out)[1], 2.5, (name, seed))  # 1.4-2.0 on these seeds
                self.assertAlmostEqual(self.fitted(out)[0], 0.20, delta=0.01)
                self.assertIn("run offsets D_r (FITTED nuisance terms, not facts about the card", out)

    def test_a_per_run_offset_is_taken_out_exactly(self):
        # the exact law plus one constant per server run (up to +-3 ms): the fitted offsets take it all, so
        # f is the truth and nothing is left; the same cells without the offsets in the fit read a misfit
        files = self.law_files(0.20, visits=2, offset=3.0, seed=4)
        rc, out, err = self.run_main("grid", *files)
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fitted(out), (0.2, 0.2, 0.2))
        self.assertIn("residual rms 0.00 % of the size's time, lack-of-fit ratio 0.00", out)
        self.assertIn("f 0.20 stands", out)
        data = g.load(files)
        cells, blocks = [], []
        for j in g.WIDTHS:
            for n in g.SIZES:
                for v, fi in zip(data.d[f"J{j}"][n], data.src[f"J{j}"][n]):
                    cells.append((n, *g.law_terms(n - 7, j), v[0]))
                    blocks.append((j, fi))
        with_runs = g.fit_f(cells, blocks, [1.0] * len(cells))
        bare = g.fit_f(cells, blocks, [1.0] * len(cells), per_run=False)
        self.assertAlmostEqual(with_runs["f"], 0.2, places=9)
        self.assertEqual(with_runs["rms"], 0.0)
        self.assertGreater(bare["rms"], 1e-3)
        self.assertGreater(abs(bare["f"] - 0.2), 1e-4)
        # the offsets come back (around their median): they are the generator's per-run constants
        self.assertEqual(len(with_runs["offsets"]), 16)

    def test_the_degrees_of_freedom_follow_the_counts(self):
        # cells - size constants - (runs - the one constant runs and sizes share) - a, b, from the counts
        for kw, text in ((dict(visits=2), "degrees of freedom 298 = 336 cells - 21 size constants - 15 run offsets "
                                          "(16 runs, 1 shared with the size constants) - 2 (a, b)"),
                         (dict(visits=1), "degrees of freedom 138 = 168 cells - 21 size constants - 7 run offsets "
                                          "(8 runs, 1 shared with the size constants) - 2 (a, b)"),
                         (dict(visits=1, sizes=[n for n in g.SIZES if n >= 64], k=40),
                          "degrees of freedom 117 = 144 cells - 18 size constants - 7 run offsets")):
            rc, out, err = self.run_main("grid", *self.law_files(0.20, noise=0.02, **kw))
            self.assertEqual(rc, 0, err)
            self.assertIn(text, out)
        cells, blocks, rel = self.noisy_cells()
        r = g.fit_f(cells, blocks, rel)
        self.assertEqual((r["cells"], r["sizes"], r["runs"], r["dof"]), (336, 21, 16, 336 - 21 - 15 - 2))
        self.assertEqual(g.fit_f(cells, blocks, rel, per_run=False)["dof"], 336 - 21 - 2)
        # runs that share no size with each other: every run offset is a size constant already
        part = [(n, *g.law_terms(n - 7, j), 42 + 53 * -(-(n - 7) // j) * (0.2 + 0.8 * j / 128))
                for n in (96, 128, 160, 192) for j in g.WIDTHS]
        runs = ["A" if n < 150 else "B" for n in (96, 128, 160, 192) for j in g.WIDTHS]
        self.assertEqual(g.fit_law(part, None, runs)[4], 4)  # 4 sizes + 2 runs - 2 components
        self.assertEqual(g._components([96, 128, 160, 192], ["A", "A", "B", "B"]), 2)

    def test_the_run_drift_is_printed_beside_the_ratio(self):
        # visit 2 (the files *_v1) of every width 2 % slower than visit 1, J 16's 1 % faster instead
        files = self.law_files(0.20, visits=2)
        for p in files:
            if "_v1." in p:
                fac = 0.99 if "grid_J16_" in p else 1.02
                rows = [json.loads(s) for s in Path(p).read_text().splitlines()]
                for r in rows:
                    r["ms"] = [x * fac for x in r["ms"]]
                Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows))
        rc, out, err = self.run_main("grid", *files)
        self.assertEqual(rc, 0, err)
        self.assertIn("run drift (observed, not fitted): per width the median over the sizes of visit 2 / visit 1: "
                      "J16 0.990, J32 1.020, J48 1.020, J64 1.020, J80 1.020, J96 1.020, J112 1.020, J128 1.020; over the "
                      "widths median 1.020, range 0.990..1.020", out)
        # one visit: no drift to show
        rc, out, err = self.run_main("grid", *self.law_files(0.20))
        self.assertNotIn("run drift", out)

    def test_one_visit_per_width_warns(self):
        rc, out, err = self.run_main("grid", *self.law_files(0.20))
        self.assertEqual(rc, 0, err)
        self.assertIn("ONE visit per width", out)

    def test_the_noisy_law_fits_to_its_noise(self):
        # 2 % request noise: the residual is the noise, the ratio near 1 (an rms or a ratio printed as 0
        # would be wrong here)
        rc, out, err = self.run_main("grid", *self.law_files(0.20, noise=0.02, visits=2, seed=5))
        self.assertEqual(rc, 0, err)
        rms, lof = self.ratio(out)
        self.assertTrue(0.2 <= rms <= 0.8, rms)
        self.assertTrue(0.7 <= lof <= 1.4, lof)
        # pinned for this seed: 336 cells, 298 degrees of freedom (21 size constants, 15 run offsets, a, b)
        self.assertIn("residual rms 0.37 % of the size's time, lack-of-fit ratio 1.02", out)
        self.assertNotIn("does not fit", out.split("verdict on the criteria")[1])

    def test_a_law_that_does_not_hold_gives_no_measurement_of_f(self):
        # the review's misfit shapes at a true f 0.20: an occupancy step (tiles of J >= 80 cost x1.3) and a
        # per-tile tail (x(1 + 0.02 (tiles - 1))); the fit would print a confident f for both
        step = self.law_files(0.20, noise=0.02, visits=2, shape=lambda n, j, w: w * (1.3 if j >= 80 else 1.0))
        rc, out, err = self.run_main("grid", *step)
        self.assertEqual(rc, 0, err)
        text = ("the residual exceeds the request noise after the run drift is taken out: the work law does not "
                "fit this card (see the visit ratios for the drift)")
        self.assertIn("R5: NO VERDICT on f  (ratio 9.", out)
        self.assertIn(text, out)
        self.assertNotIn("wave quantisation", out)  # a ratio of 9 is no 68-SM wave effect
        self.assertIn("is not a measurement of f", out)
        self.assertIn("R2: MET", out)  # the table itself loses there: R2 still speaks about it
        self.assertGreater(self.ratio(out)[1], g.LOF_MAX)
        tail = self.law_files(0.20, noise=0.02, visits=2, shape=lambda n, j, w: w * (1 + 0.02 * (-(-(n - 7) // j) - 1)))
        rc, out, err = self.run_main("grid", *tail)
        self.assertEqual(rc, 0, err)
        self.assertIn("no verdict on f 0.20: " + text, out)
        # a milder misfit: the tiles of N > 256 cost x1.25 (ratio about 5.5, above the a-priori 3)
        self.assertEqual((g.LOF_MAX, g.LOF_WAVE), (3.0, 5.0))
        late = self.law_files(0.20, noise=0.02, visits=2, shape=lambda n, j, w: w * (1.25 if n > 256 else 1.0))
        rc, out, err = self.run_main("grid", *late)
        self.assertEqual(rc, 0, err)
        self.assertIn("R5: NO VERDICT on f  (ratio 5.50", out)

    def test_a_misfit_still_reads_no_verdict_under_a_run_drift(self):
        # the run offsets take a drift of the server runs, not a misfit of the law: under +-1 % per run the
        # cost step at N > 256 (an L2 effect, ratio about 5.5) and the per-tile tail (about 11) still give no
        # verdict on f, and a step of x1.18 (ratio 3-5) names the wave quantisation as a candidate too
        shapes = (("L2 x1.25 at N > 256", lambda n, j, w: w * (1.25 if n > 256 else 1.0), 5.0, False),
                  ("tail 2 % a tile", lambda n, j, w: w * (1 + 0.02 * (-(-(n - 7) // j) - 1)), 8.0, False),
                  ("L2 x1.18 at N > 256", lambda n, j, w: w * (1.18 if n > 256 else 1.0), 3.0, True))
        for name, shape, least, wave in shapes:
            for seed in range(2):
                rc, out, err = self.run_main("grid", *self.law_files(0.20, noise=0.02, visits=2, drift=0.01,
                                                                     seed=seed, shape=shape))
                self.assertEqual(rc, 0, err)
                self.assertIn("R5: NO VERDICT on f", out, (name, seed))
                self.assertGreater(self.ratio(out)[1], least, (name, seed))
                self.assertEqual("wave quantisation of the stream-k launch" in out, wave, (name, seed))
                if wave:
                    self.assertLessEqual(self.ratio(out)[1], g.LOF_WAVE, (name, seed))
                    self.assertIn("(68 on the RTX 2080 Ti)", out)

    def test_an_interval_across_an_edge_is_undecided(self):
        rc, out, err = self.run_main("grid", *self.law_files(0.15, noise=0.02, visits=2, seed=2))
        self.assertEqual(rc, 0, err)
        fit, lo, hi = self.fitted(out)
        self.assertTrue(lo < 0.15 < hi, (lo, hi))
        self.assertIn("R5: UNDECIDED", out)
        self.assertIn("f 0.20 is UNDECIDED", out)

    def test_the_fit_reads_cell_medians(self):
        rc, out, err = self.run_main("grid", *self.law_files(0.12, outlier=True))
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fitted(out), (0.12, 0.12, 0.12))

    def test_the_table_at_the_measured_f_is_printed(self):
        rc, out, err = self.run_main("grid", *self.law_files(0.05))
        self.assertEqual(rc, 0, err)
        self.assertIn("the table of picks at the measured f 0.050 against f 0.20: differs at 5 sizes "
                      "(40t/33c:J64->J16, 48t/41c:J64->J16, 208t/201c:J128->J80, 224t/217c:J128->J80, "
                      "448t/441c:J96->J64)", out)
        self.assertIn("the table of picks at the interval's high edge 0.050", out)

    def noisy_cells(self, widths=None, f=0.2, seed=3):
        rng = random.Random(seed)
        cells, blocks = [], []
        for v in range(2):
            for j in widths or g.WIDTHS:
                run = 1 + rng.uniform(-0.01, 0.01)
                for n in g.SIZES:
                    t = (42 + 53 * -(-(n - 7) // j) * (f + (1 - f) * j / 128)) * run
                    cells.append((n, *g.law_terms(n - 7, j), t * (1 + rng.uniform(-0.01, 0.01))))
                    blocks.append((j, v))
        return cells, blocks, [1.0] * len(cells)

    def test_the_interval_is_the_central_95_percent_of_1000_draws(self):
        self.assertEqual((g.FIT_LEVEL, g.BOOT_N), (0.95, 1000))
        r = g.fit_f(*self.noisy_cells())
        self.assertEqual(r["draws"] + r["bad"], 1000)
        self.assertEqual(r["blocks"], 16)  # the bootstrap unit is the width visit, not the cell
        self.assertTrue(r["dmin"] < r["lo"] < r["f"] < r["hi"] < r["dmax"], r)

    def test_undetermined_draws_are_dropped(self):
        # two widths, one visit each: a draw of one width alone determines nothing; it must not count as f 0
        cells, blocks, rel = self.noisy_cells(widths=[16, 128], f=0.3)
        cells = [(n, x1, x2, 42 + 53 * (x1 * 0.3 + x2)) for (n, x1, x2, _), bl in zip(cells, blocks)]
        keep = [i for i, bl in enumerate(blocks) if bl[1] == 0]
        r = g.fit_f([cells[i] for i in keep], [blocks[i] for i in keep], [rel[i] for i in keep])
        self.assertGreater(r["bad"], 100)
        self.assertAlmostEqual(r["lo"], 0.3, places=9)
        self.assertAlmostEqual(r["hi"], 0.3, places=9)

    def test_grid_refuses_widths_with_different_visits(self):
        files = self.law_files(0.2, visits=2)
        files = [p for p in files if not p.endswith("grid_J80_v1.jsonl")]
        rc, _, err = self.run_main("grid", *files)
        self.assertEqual(rc, 1)
        self.assertIn("different number of visits", err)

    def test_grid_refuses_a_visit_without_a_size(self):
        files = self.law_files(0.2, visits=2)
        path = Path(next(p for p in files if p.endswith("grid_J80_v1.jsonl")))
        path.write_text("".join(s + "\n" for s in path.read_text().splitlines() if '"n": 96,' not in s))
        rc, _, err = self.run_main("grid", *files)
        self.assertEqual(rc, 1)
        self.assertIn("incomplete visit", err)

    # --- the A/B reader's verdict path on the law (old = fewest tiles, new = the table at 0.20) ----------
    # Picks written out by hand (not with g.pick) for c = N - 7 at the mix sizes and the judged ones:
    # old: 2 tiles of 128 at 129..256 columns; new at f 0.20: 80 at 135..153, 96 at 169..185, 128 at 201..249.
    NEW_AT_020 = {142: 80, 144: 80, 160: 80, 176: 96, 192: 96, 208: 128, 224: 128, 240: 128, 256: 128}

    def law_ab_files(self, f=0.20, C=42.0, L=53.0, k=7, same=False, outliers=(), grows=0.0):
        """Three rounds of old / new on the law at f; `same`: the new arm runs the old widths; `outliers`:
        (arm, visit, N) cells that get one request at 5x (the median is unchanged, the mean is not)."""
        def t(n, j):
            return C + grows * n + L * -(-(n - k) // j) * (f + (1 - f) * j / 128)

        files = []
        for v in range(3):
            for lab in ("old", "new"):
                rows = []
                for n in g.AB_SIZES:
                    c = n - k
                    jo = 128 if c > 64 else (64 if c > 32 else (32 if c > 16 else 16))
                    if lab == "old" or same:
                        j = jo
                    else:
                        j = self.NEW_AT_020.get(n) or g.pick(c, 0.20, g.PRODUCTION)
                    ms = [t(n, j)] * 4
                    if (lab, v, n) in outliers:
                        ms[3] *= 5
                    rows.append(row(lab, n, ms, k=k, **({"f": 0.20} if lab == "new" else {})))
                files.append(self.write(f"ab_{v}_{lab}.jsonl", rows))
        return files

    def test_ab_on_the_law_pins_outside_mix_and_r4(self):
        # the law at f 0.20, C 42 / L 53 ms: mix old 148, new (2 x 116.2 + 2 x 126.8 + 4 x 148) / 8 = 134.75,
        # -8.95 %; predicted mix (-19 -19 -13 -13 -0.5 x4) / 8 .. (-14 -14 -9 -9 +0.5 x4) / 8 = -8.25 .. -5.5,
        # R4 band -10.25 .. -3.5: not met. Per size: 135..153 columns -21.5 % (outside -19..-14), 169..185
        # -14.3 % (outside -13..-9), 201..249 0 % (inside): 5 of the 9 judged sizes outside.
        rc, out, err = self.run_main("ab", *self.law_ab_files(outliers=(("new", 1, 144), ("old", 2, 160))))
        self.assertEqual(rc, 0, err)
        self.assertIn("measured change outside the predicted range: 5 of 9 sizes that have a prediction", out)
        self.assertIn("mix [144, 160, 176, 192, 208, 224, 240, 256]: old 148.00 new 134.75 ms -9.0 %  "
                      "(per visit -9.0 .. -9.0 %, 3 visits)", out)
        self.assertIn("is): -8.2 .. -5.5 %", out)
        self.assertIn("R4: not met (measured -9.0 % against -10.2 .. -3.5 %)", out)
        self.assertEqual(out.count("OUTSIDE"), 5)

    def test_ab_weights_the_predicted_mix_like_the_measured_one(self):
        # the old times grow with N (148 + N ms at the mix sizes): the measured mix change is a change of
        # means, so the predicted range is weighted by the old times: lo = (-19 (292 + 308) - 13 (324 + 340)
        # - 0.5 (356 + 372 + 388 + 404)) / 2784 = -7.47, hi = (-14 x 600 - 9 x 664 + 0.5 x 1520) / 2784 = -4.89
        # (unweighted: -8.25 .. -5.5)
        rc, out, err = self.run_main("ab", *self.law_ab_files(grows=1.0))
        self.assertEqual(rc, 0, err)
        self.assertIn("is): -7.5 .. -4.9 %", out)

    def test_ab_with_no_change_meets_r4(self):
        rc, out, err = self.run_main("ab", *self.law_ab_files(same=True))
        self.assertEqual(rc, 0, err)
        self.assertIn("R4: MET (refutes f 0.20) (measured +0.0 %", out)

    def test_ab_picks_on_columns_not_tokens(self):
        # k 40: 142 tokens are 102 columns, one tile of 128 for both arms (on tokens the new arm would be 80)
        rc, out, err = self.run_main("ab", *self.law_ab_files(k=40))
        self.assertEqual(rc, 0, err)
        self.assertRegex(out, r"N=142  102 J128->J128 ")
        self.assertRegex(out, r"N= 96   56 J64 ->J64  ")


if __name__ == "__main__":
    unittest.main()
