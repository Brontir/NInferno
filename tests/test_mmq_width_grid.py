import contextlib
import io
import json
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
        self.assertIn("R1: not met", out)

    def test_a_card_whose_f_is_far_from_the_assumption_refutes_it(self):
        # times that follow the rule at f 0.05: the assumed table loses much more than 0.5 pp to the least
        rc, out, err = self.run_main("grid", *self.model_files(g.SIZES, f=0.05, base=0.0))
        self.assertEqual(rc, 0, err)
        self.assertIn("R1: MET", out)
        self.assertIn("is REFUTED", out)

    def test_a_wrong_pick_shows_as_a_mismatch(self):
        # at 142 tokens the width predicted at f 0.20 is far slower than another
        files = self.model_files([96, 128, 142, 160], nudge={(80, 142): 40.0, (96, 142): 40.0})
        rc, out, _ = self.run_main("grid", *files)
        self.assertEqual(rc, 0)
        self.assertIn("MISMATCH", out)
        self.assertIn("material mismatch): 1", out)

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


if __name__ == "__main__":
    unittest.main()
