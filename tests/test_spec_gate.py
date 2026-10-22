import contextlib
import datetime as dt
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import spec_gate  # noqa: E402

SINCE = dt.datetime(2026, 10, 20, tzinfo=dt.timezone.utc)
NEW, OLD = "2026-10-21T10:00:00Z", "2026-10-19T10:00:00Z"


def row(value, spread=None, auto=True, served="dflash", date=NEW, metric="decode", extra="",
        **fields):
    out = {"machine": {"gpu_name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0},
           "model": "org/m", "recipe": "org/m-GGUF --quant Q4", "kv": "auto", "sampling": "greedy",
           "metric": metric, "prompt_set": "short", "concurrency": 1, "date": date,
           "spec": served if auto else "none", "draft_tokens": 7 if auto else None,
           "flags": f"--max-context 32768 --max-concurrency 1 {extra}"
                    + ("" if auto else " --spec none"),
           "value": value, "spread": spread}
    out.update(fields)
    return out


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
        rows = [row(80, [78, 82], date=OLD, commit="abc"), row(100, [98, 102], auto=False)]
        r = spec_gate.gate(rows, SINCE, newer=lambda x: x.get("commit") == "abc")
        self.assertEqual(r["counts"]["fail"], 1)

    def test_plain_served_and_small_loss_pass(self):
        r = self.judge(row(80, served="none"), row(100, auto=False))
        self.assertEqual((r["counts"]["plain_served"], r["counts"]["fail"]), (1, 0))
        r = self.judge(row(98), row(100, auto=False))
        self.assertEqual((r["counts"]["pass"], r["counts"]["losses"]), (1, 0))

    def test_pairs_on_every_non_spec_flag(self):
        plain = row(100, auto=False)
        for auto in (row(80, extra="--activations exact"), row(80, kv="fp8"),
                     row(80, concurrency=4), row(80, prompt_set="long"),
                     row(80, machine={"gpu_name": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0}),
                     row(80, recipe="org/m-GGUF --quant Q8")):
            r = self.judge(auto, plain)
            self.assertEqual((r["counts"]["paired"], r["counts"]["unpaired_auto"],
                              r["counts"]["unpaired_plain"]), (0, 1, 1), auto)
        # flag order and the drafter depth are not part of the key
        r = self.judge(row(80, extra="--x 1 --y 2"), row(100, auto=False, extra="--y 2 --x 1"))
        self.assertEqual(r["counts"]["paired"], 1)

    def test_prefill_and_explicit_spec_excluded(self):
        explicit = row(50)
        explicit["flags"] += " --spec mtp"
        r = self.judge(row(10, metric="prefill"), row(100, auto=False, metric="prefill"), explicit)
        self.assertEqual(r["stats"]["prefill_excluded"], 2)
        self.assertEqual(r["stats"]["explicit_spec"], 1)
        self.assertEqual(r["counts"]["paired"], 0)

    def test_newest_row_of_a_side_is_used(self):
        r = self.judge(row(80, date=NEW), row(120, date="2026-10-21T12:00:00Z"),
                       row(100, auto=False))
        self.assertEqual((r["stats"]["superseded"], r["counts"]["pass"]), (1, 1))

    def test_main_exit_code_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            bench = Path(d) / "bench.jsonl"
            bench.write_text("\n".join(json.dumps(x) for x in
                                       (row(80, [78, 82]), row(100, [98, 102], auto=False))))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = spec_gate.main(["--bench", str(bench), "--since-date", SINCE.isoformat(),
                                       "--json"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out.getvalue())["counts"]["fail"], 1)
            with contextlib.redirect_stdout(io.StringIO()):
                code = spec_gate.main(["--bench", str(bench), "--since-date", NEW])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
