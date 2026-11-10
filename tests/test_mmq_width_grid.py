import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import mmq_width_grid as g


class FakeServe(BaseHTTPRequestHandler):
    """A stand-in for ninferno-serve: counts a prompt as 4 template tokens + 1 per word (+ the salt's
    digits), answers with usage and timings like the non-streaming chat response."""

    skew = 0  # tokens added on top of the true count (a mismatching size)
    fail = False
    seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.reply({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        if type(self).fail:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"ggml mmq: no tile fits this device")
            return
        words = body["messages"][0]["content"].split()
        # the salt word is 7 tokens ("w" + 6 digits), the other words 1 each
        n = 4 + 7 + len(words) - 1 + type(self).skew
        self.reply({"usage": {"prompt_tokens": n},
                    "timings": {"cache_n": 0, "prompt_n": n, "prompt_ms": 10.0 + n / 10}})

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


def row(label, n, ms, pn=None):
    return {"label": label, "n": n, "ms": ms, "prompt_n": pn if pn is not None else [n] * len(ms)}


class Measure(Base):
    def setUp(self):
        super().setUp()
        FakeServe.skew, FakeServe.fail, FakeServe.seen = 0, False, []
        self.srv = HTTPServer(("127.0.0.1", 0), FakeServe)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.port = self.srv.server_address[1]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        super().tearDown()

    def measure(self, out, sizes="16,80,128"):
        return self.run_main("measure", "--port", str(self.port), "--label", "J80", "--out", str(out),
                             "--sizes", sizes, "--reps", "2", "--warmup", "1")

    def test_exact_sizes_thinking_off_and_prompt_n_recorded(self):
        out = self.root / "m.jsonl"
        rc, stdout, _ = self.measure(out)
        self.assertEqual(rc, 0, stdout)
        rows = [json.loads(s) for s in out.read_text().splitlines()]
        self.assertEqual([r["n"] for r in rows], [16, 80, 128])
        for r in rows:
            self.assertEqual(r["prompt_n"], [r["n"]] * 2)
        self.assertTrue(all(b["enable_thinking"] is False and b["max_tokens"] == 1 and "stream" not in b
                            for b in FakeServe.seen))
        salts = {b["messages"][0]["content"].split()[0] for b in FakeServe.seen}
        self.assertTrue(all(len(s) == 1 + g.SALT_DIGITS for s in salts), salts)

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


class Readers(Base):
    def grid_files(self, shift=0.0):
        files = []
        for j in g.WIDTHS:
            rows = [row(f"J{j}", n, [1.0 + n / 100 + abs(j - 96) / 500 + shift] * 4) for n in (96, 128, 160)]
            files.append(self.write(f"grid_J{j}.jsonl", rows))
        return files

    def test_grid_reports_counts_and_a_least(self):
        rc, out, err = self.run_main("grid", *self.grid_files())
        self.assertEqual(rc, 0, err)
        self.assertIn("J96: 3 sizes, 3 visits, 12 requests", out)
        self.assertIn("least", out)

    def test_grid_refuses_stale_rows(self):
        files = self.grid_files()
        files.append(self.write("stale.jsonl", [row("J80", 96, [9.0] * 4)]))
        rc, _, err = self.run_main("grid", *files)
        self.assertEqual(rc, 1)
        self.assertIn("stale rows", err)

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
        old = self.write("old.jsonl", [{"label": "J16", "n": 16, "ms": [1.0]}])
        rc, _, err = self.run_main("grid", old)
        self.assertIn("no prompt_n", err)

    def ab_files(self, drop_new=None, visits=2):
        files = []
        for v in range(visits):
            for lab, k in (("old", 1.0), ("new", 0.9)):
                rows = [row(lab, n, [k * (n / 10 + v * 0.1)] * 2) for n in g.MIX if n != drop_new or lab == "old"]
                files.append(self.write(f"ab_{v}_{lab}.jsonl", rows))
        return files

    def test_ab_prints_spread_and_mix(self):
        rc, out, err = self.run_main("ab", *self.ab_files())
        self.assertEqual(rc, 0, err)
        self.assertIn("[", out)
        self.assertIn("per visit", out)
        self.assertIn("-10.0 %", out)

    def test_ab_size_missing_from_new_is_handled(self):
        rc, out, err = self.run_main("ab", *self.ab_files(drop_new=160))
        self.assertEqual(rc, 0, err)
        self.assertIn("warning: sizes [160]", out)

    def test_ab_unequal_visits_refused(self):
        files = self.ab_files()
        rc, _, err = self.run_main("ab", *files[:-1])
        self.assertEqual(rc, 1)
        self.assertIn("visits", err)

    def test_pick_keeps_the_production_rule(self):
        self.assertEqual(g.pick(80, 0.2, g.PRODUCTION), 80)
        self.assertEqual(g.pick(129, None, g.OLD), 128)


if __name__ == "__main__":
    unittest.main()
