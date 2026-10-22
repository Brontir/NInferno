import contextlib
import datetime as dt
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import spec_gate  # noqa: E402

SINCE = dt.datetime(2026, 10, 20, tzinfo=dt.timezone.utc)
NEW, OLD = "2026-10-21T10:00:00Z", "2026-10-19T10:00:00Z"


def row(value, spread=None, auto=True, served="dflash", date=NEW, metric="decode", extra="",
        **fields):
    out = {"machine": {"gpu_name": "NVIDIA GeForce RTX 4090", "sm": "89", "vram_gb": 24.0,
                       "driver": "580.95.05"},
           "model": "org/m", "recipe": "org/m-GGUF --quant Q4", "kv": "auto", "sampling": "greedy",
           "metric": metric, "prompt_set": "short", "concurrency": 1, "date": date,
           "spec": served if auto else "none", "draft_tokens": 7 if auto else None,
           "flags": f"--max-context 32768 --max-concurrency 1 {extra}"
                    + ("" if auto else " --spec none"),
           "value": value, "spread": spread}
    out.update(fields)
    return out


def run_main(args, rows=None, text=None):
    """main() over a temporary bench file; returns (exit code, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as d:
        bench = Path(d) / "bench.jsonl"
        if rows is not None:
            bench.write_text("\n".join(json.dumps(x) for x in rows))
        elif text is not None:
            bench.write_text(text)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = spec_gate.main(["--bench", str(bench), *args])
        return code, out.getvalue(), err.getvalue()


class SpecGateTest(unittest.TestCase):
    def judge(self, *rows):
        return spec_gate.gate(list(rows), SINCE)

    def test_real_loss_fails(self):
        r = self.judge(row(80, [78, 82]), row(100, [98, 102], auto=False))
        self.assertEqual(r["counts"]["paired"], 1)
        self.assertEqual(r["counts"]["fail"], 1)
        self.assertEqual(r["cells"][0]["verdict"], "fail")

    def test_missing_spread_is_the_single_value(self):
        r = self.judge(row(80, metric="decode_aggregate"),
                       row(100, auto=False, metric="decode_aggregate"))
        self.assertEqual(r["counts"]["fail"], 1)

    def test_overlap_passes(self):
        r = self.judge(row(90, [85, 99]), row(100, [96, 104], auto=False))
        self.assertEqual((r["counts"]["overlap"], r["counts"]["fail"]), (1, 0))

    def test_stale_row_passes(self):
        r = self.judge(row(80, [78, 82], date=OLD), row(100, [98, 102], auto=False))
        self.assertEqual((r["counts"]["stale"], r["counts"]["fail"]), (1, 0))

    def test_commit_answer_wins_over_date(self):
        rows = [row(80, [78, 82], date=OLD, commit="abcdef1"), row(100, [98, 102], auto=False)]
        r = spec_gate.gate(rows, SINCE, newer=lambda x: True if spec_gate.row_commit(x) else None)
        self.assertEqual(r["counts"]["fail"], 1)

    def test_stale_plain_twin_is_not_a_fail(self):
        r = self.judge(row(80, [78, 82]), row(100, [98, 102], auto=False, date=OLD))
        self.assertEqual((r["counts"]["stale_plain"], r["counts"]["fail"]), (1, 0))

    def test_unknown_commit_falls_back_to_date_and_is_counted(self):
        rows = [row(80, [78, 82], commit="abcdef1"), row(100, [98, 102], auto=False)]
        r = spec_gate.gate(rows, SINCE, newer=lambda x: None)
        self.assertEqual((r["stats"]["commit_unknown"], r["counts"]["fail"]), (1, 1))

    def test_real_git_ancestry(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            def g(*args):
                return subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t",
                                       *args], check=True, capture_output=True,
                                      text=True).stdout.strip()

            g("init", "-q")
            commits = {}
            for name, path in (("root", "a.txt"), ("before", "a.txt"),
                               ("decision", "src/spec/x.h"), ("after", "a.txt")):
                (Path(d) / path).parent.mkdir(parents=True, exist_ok=True)
                (Path(d) / path).write_text(name)
                g("add", "-A")
                g("commit", "-q", "-m", name)
                commits[name] = g("rev-parse", "HEAD")
            plain = row(100, [98, 102], auto=False, commit=commits["after"])
            cases = (("after", OLD, 1), ("before", "2030-01-01T00:00:00Z", 0))
            for name, date, fails in cases:  # the commit decides, whatever the date says
                with mock.patch.object(spec_gate, "REPO", Path(d)), \
                        mock.patch.object(spec_gate, "DECISION_FILES", ("src/spec",)):
                    code, out, _ = run_main(["--json"], [row(80, [78, 82], date=date,
                                                             commit=commits[name]), plain])
                result = json.loads(out)
                self.assertTrue(result["since"].startswith(commits["decision"][:12]))
                self.assertEqual((code, result["counts"]["fail"]), (fails, fails), name)
                self.assertEqual(result["stats"]["commit_unknown"], 0)

    def test_memory_size_and_engine_in_the_card(self):
        small = {"gpu_name": "NVIDIA GeForce RTX 4060 Ti", "sm": "89", "vram_gb": 7.99}
        big = {"gpu_name": "NVIDIA GeForce RTX 4060 Ti", "sm": "89", "vram_gb": 15.99}
        r = self.judge(row(80, machine=small), row(100, auto=False, machine=big))
        self.assertEqual(r["counts"]["paired"], 0)
        r = self.judge(row(80, engine="ninferno"), row(100, auto=False))
        self.assertEqual(r["counts"]["paired"], 1)
        r = self.judge(row(80, engine="llama.cpp"), row(100, auto=False))
        self.assertEqual(r["counts"]["paired"], 0)

    def test_row_commit_from_commit_or_sha_version(self):
        self.assertEqual(spec_gate.row_commit({"commit": "84d92a362"}), "84d92a362")
        self.assertEqual(spec_gate.row_commit({"version": "84d92a362"}), "84d92a362")
        self.assertIsNone(spec_gate.row_commit({"version": "0.1.0"}))

    def test_plain_served_and_small_loss_pass(self):
        r = self.judge(row(80, served="none"), row(100, auto=False))
        self.assertEqual((r["counts"]["plain_served"], r["counts"]["fail"]), (1, 0))
        r = self.judge(row(98), row(100, auto=False))
        self.assertEqual((r["counts"]["pass"], r["counts"]["losses"]), (1, 0))

    def test_pairs_on_every_non_spec_flag(self):
        plain = row(100, auto=False)
        for auto in (row(80, extra="--activations exact"), row(80, kv="fp8"),
                     row(80, concurrency=4), row(80, prompt_set="long"), row(80, prompt_len=9),
                     row(80, machine={"gpu_name": "NVIDIA GeForce RTX 5090", "sm": "120"}),
                     row(80, machine={"gpu_name": "NVIDIA GeForce RTX 4090", "sm": "90"}),
                     row(80, recipe="org/m-GGUF --quant Q8")):
            r = self.judge(auto, plain)
            self.assertEqual((r["counts"]["paired"], r["counts"]["unpaired_auto"],
                              r["counts"]["unpaired_plain"]), (0, 1, 1), auto)
        # flag order and the drafter depth are not part of the key
        r = self.judge(row(80, extra="--x 1 --y 2"), row(100, auto=False, extra="--y 2 --x 1"))
        self.assertEqual(r["counts"]["paired"], 1)

    def test_card_is_gpu_name_and_sm_only(self):
        plain = row(100, auto=False, version="0.1.0", warmup=1)
        auto = row(80, version="84d92a362", warmup=2, decision_fingerprint="x",
                   machine={"gpu_name": "NVIDIA GeForce RTX 4090", "sm": "89", "vram_gb": 23.9,
                            "driver": "590.1", "cuda": "13.1", "power_limit_w": 400.0})
        self.assertEqual(self.judge(auto, plain)["counts"]["paired"], 1)

    def test_spec_auto_is_auto_and_forced_rows_are_excluded(self):
        plain = row(100, auto=False)
        r = self.judge(row(80, extra="--spec auto"), plain)
        self.assertEqual(r["counts"]["paired"], 1)
        for extra in ("--draft-tokens 3", "--spec-load 4:2", "--spec auto --draft-tokens=3"):
            r = self.judge(row(80, extra=extra), plain)
            self.assertEqual((r["stats"]["forced_spec"], r["counts"]["paired"]), (1, 0), extra)

    def test_prefill_and_explicit_spec_excluded(self):
        r = self.judge(row(10, metric="prefill"), row(100, auto=False, metric="prefill"),
                       row(50, extra="--spec mtp"))
        self.assertEqual(r["stats"]["prefill_excluded"], 2)
        self.assertEqual(r["stats"]["explicit_spec"], 1)
        self.assertEqual(r["counts"]["paired"], 0)

    def test_newest_row_of_a_side_is_used(self):
        newer = row(120, date="2026-10-21T12:00:00Z")
        for order in ((row(80), newer), (newer, row(80))):  # the older one first, then second
            r = self.judge(*order, row(100, auto=False))
            self.assertEqual((r["stats"]["superseded"], r["counts"]["pass"]), (1, 1))

    def test_main_exit_code_and_json(self):
        rows = (row(80, [78, 82]), row(100, [98, 102], auto=False))
        code, out, _ = run_main(["--since-date", SINCE.isoformat(), "--json"], rows)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["counts"]["fail"], 1)
        self.assertEqual(run_main(["--since-date", NEW], rows)[0], 0)

    def test_nothing_paired_fails_and_unpaired_auto_warns(self):
        rows = (row(80, concurrency=4), row(100, auto=False))
        code, _, err = run_main(["--since-date", NEW], rows)
        self.assertEqual(code, 1)
        self.assertIn("no auto row pairs", err)
        self.assertIn("WARNING: 1 auto rows have no plain twin", err)
        rows = (row(99), row(100, auto=False), row(80, concurrency=4))
        code, _, err = run_main(["--since-date", NEW], rows)
        self.assertEqual(code, 0)
        self.assertIn("RTX 4090 m Q4 short C4 decode", err)

    def test_missing_or_malformed_bench_exits_2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(spec_gate.main(["--bench", "/nonexistent/bench.jsonl",
                                             "--since-date", NEW]), 2)
        self.assertEqual(run_main(["--since-date", NEW], text="{not json")[0], 2)
        self.assertEqual(run_main(["--since-date", NEW], text="[1, 2]")[0], 2)
        bad = row(80)
        del bad["value"]
        self.assertEqual(run_main(["--since-date", NEW], [bad, row(100, auto=False)])[0], 2)

    def test_shallow_clone_or_root_decision_commit_exits_2(self):
        rows = (row(99), row(100, auto=False))
        with mock.patch.object(spec_gate, "git", return_value="true"):
            code, _, err = run_main([], rows)
        self.assertEqual(code, 2)
        self.assertIn("shallow", err)

        def root_only(*args):
            if args[0] == "rev-parse":
                return "false"
            if "--format=%H" in args:
                return "a" * 40
            return "a" * 40 + " 2026-10-21T22:13:54+02:00"  # no parents: the root commit

        with mock.patch.object(spec_gate, "git", side_effect=root_only):
            code, _, err = run_main([], rows)
        self.assertEqual(code, 2)
        self.assertIn("root commit", err)


if __name__ == "__main__":
    unittest.main()
