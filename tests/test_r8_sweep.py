"""tools/bench/r8_decode_table.py and r8_sweep_summary.py read ninferno_r8_schedule_bench's
output as the bench prints it (t=, decode=, a16pick and h= since #155), synthetic sweeps, no GPU.
Also the #850 kits around them: run_850_r8_sweep.sh --summarize refuses an incomplete protocol, and
step_ab.py summarizes the whole-step A/B's rounds."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import r8_decode_table as tool  # noqa: E402
import r8_sweep_summary as summary  # noqa: E402
import step_ab  # noqa: E402

HEAD = ("device NVIDIA GeForce RTX 4090 class ad102:128sm, L2 72 MB, 15 samples x 60 launches\n"
        "board cc 89 sms 128 l2_mib 72 dram_gbps 1008 pci 0x2684\n")


def row(fmt, epi, n, k, sched, us, marks="", t=1, same=1):
    return (f"r8 {fmt} {epi} n={n} k={k} t={t} r8.{sched} {us:.2f} us 850.0 GB/s same={same} "
            f"decode=1{marks} h=0123456789abcdef\n")


def sweep(w8r1, w8r2=48.70, extra="", add_w8r1=None):
    text = HEAD
    for epi in ("store", "add"):
        text += row("q6k", epi, 4096, 12288, "w4r1", 47.10)
        text += row("q6k", epi, 4096, 12288, "w4r2", 49.00)
        text += row("q6k", epi, 4096, 12288, "w8r1",
                    add_w8r1 if epi == "add" and add_w8r1 is not None else w8r1)
        text += row("q6k", epi, 4096, 12288, "w8r2", w8r2, " picked rules")
        text += (f"r8 q6k {epi} n=4096 k=12288 t=1 rmma.m1.n1.w4x1 60.00 us 700.0 GB/s same=0 "
                 "decode=0 a16pick h=0123456789abcdef\n")
        text += f"pick q6k {epi} n=4096 k=12288 t=1 picked=r8.w8r2 48.70 us best=r8.w8r1 45.0 us\n"
    # A verify width is not a decode row: it must not enter the table (its schedules may differ).
    text += row("q6k", "store", 4096, 12288, "w8r1", 10.0, t=2, same=0)
    return text + extra + "done\n"


class Reader(unittest.TestCase):
    def files(self, texts):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = []
        for i, text in enumerate(texts):
            p = Path(tmp.name) / f"down_{i}.txt"
            p.write_text(text)
            paths.append(str(p))
        return paths

    def test_current_format_rows_and_entry(self):
        paths = self.files([sweep(45.00), sweep(45.30), sweep(44.90)])
        run, base, board, adds = tool.read_run(paths[0], "ad102:128sm")
        key = ("q6k", "store", 4096, 12288)
        self.assertEqual(board, "0x2684")
        self.assertEqual(base[key], (8, 2))
        self.assertEqual(sorted(run[key]), [(4, 1), (4, 2), (8, 1), (8, 2)])
        self.assertEqual(run[key][(8, 1)], 45.00)  # the t=2 row did not overwrite it
        self.assertEqual(adds[key][1], (8, 2))  # the add row checks the store entry
        entries = tool.entries_for("ad102:128sm", paths, 0.005)
        self.assertEqual([(e[0], e[1]) for e in entries], [(key, (8, 1))])

    def test_unknown_format_is_refused(self):
        old = HEAD + "r8 q6k store 4096 12288 w8r2 48.70\ndone\n"
        paths = self.files([old])
        with self.assertRaises(SystemExit):
            tool.read_run(paths[0], "ad102:128sm")

    def test_unequal_bytes_are_refused(self):
        bad = HEAD + row("q6k", "store", 4096, 12288, "w8r1", 45.0, same=0) + "done\n"
        paths = self.files([bad])
        with self.assertRaises(SystemExit):
            tool.read_run(paths[0], "ad102:128sm")

    def summarize(self, paths):
        out = io.StringIO()
        argv = sys.argv
        sys.argv = ["r8_sweep_summary.py", *paths]
        try:
            with contextlib.redirect_stdout(out):
                summary.main()
        finally:
            sys.argv = argv
        return out.getvalue()

    def test_add_check_drops_an_entry_slower_on_the_add_call(self):
        # The store row writes w8r1 (45.50 < 48.70); on the add call w8r1 is 49.00, above w8r2.
        paths = self.files([sweep(45.50, add_w8r1=49.0), sweep(45.60, add_w8r1=49.1),
                            sweep(45.40, add_w8r1=48.9)])
        text = self.summarize(paths)
        self.assertIn("- q6k 4096x12288: store entry w8r1; add w8r2 48.70 -> w8r1 49.00 us "
                      "(+0.30, faster in every sweep: no): DROP (slower on the add call)", text)

    def test_table_drops_an_entry_slower_on_the_add_call(self):
        # The generator applies the add check the summary reports (codex on #874): w8r1 wins the
        # store call but not the add call, so no entry is written.
        paths = self.files([sweep(45.50, add_w8r1=49.0), sweep(45.60, add_w8r1=49.1),
                            sweep(45.40, add_w8r1=48.9)])
        self.assertEqual(tool.entries_for("ad102:128sm", paths, 0.005), [])

    def test_add_check_needs_every_sweep(self):
        # Faster on the add call in the median (47.0 < 48.7) but not in the third sweep (48.8):
        # the summary drops it and the generator writes no entry.
        paths = self.files([sweep(45.50, add_w8r1=47.0), sweep(45.60, add_w8r1=46.9),
                            sweep(45.40, add_w8r1=48.8)])
        self.assertEqual(tool.entries_for("ad102:128sm", paths, 0.005), [])
        self.assertIn("faster in every sweep: no): DROP (faster in the median, not in every sweep)",
                      self.summarize(paths))

    def test_summary_verdicts(self):
        paths = self.files([sweep(45.50), sweep(45.60), sweep(45.40)])
        text = self.summarize(paths)
        self.assertIn("- q6k 4096x12288: store entry w8r1; add w8r2 48.70 -> w8r1 45.50 us "
                      "(-3.20, faster in every sweep: yes): keep", text)
        # 45.50 - 48.70 = -3.20 us: inside the RTX 4090's frozen band -2.0..-4.0.
        self.assertIn("- store: w8r2 48.70 -> w8r1 45.50 us, -3.20 us", text)
        self.assertIn("inside the band: accepted", text)
        self.assertIn("| q6k add 4096x12288 | 3 | w8r2 |", text)
        self.assertIn("no (looked up as store)", text)
        self.assertIn("w8r1 is never slower than w8r2", text)

    def test_summary_refutation(self):
        paths = self.files([sweep(48.60), sweep(48.40), sweep(48.50)])
        out = io.StringIO()
        argv = sys.argv
        sys.argv = ["r8_sweep_summary.py", *paths]
        try:
            with contextlib.redirect_stdout(out):
                summary.main()
        finally:
            sys.argv = argv
        self.assertIn("under half the lower edge: refuted on this card", out.getvalue())


class SweepKit(unittest.TestCase):
    def test_summarize_refuses_an_incomplete_protocol(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("no bash")
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("down_1.txt", "context_1.txt", "down_2.txt"):
                (Path(tmp) / name).write_text(sweep(45.0))
            kit = Path(__file__).resolve().parents[1] / "tools" / "bench" / "run_850_r8_sweep.sh"
            done = subprocess.run([bash, str(kit), "--summarize", tmp], capture_output=True,
                                  text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("missing context_2.txt down_3.txt context_3.txt", done.stderr)


class StepAB(unittest.TestCase):
    """tools/bench/step_ab.py summarize on rounds as run_850_step_ab.sh writes them."""

    def run_summary(self, a, b, same=True, order="ABBAABBAABBA", rounds=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        values = {"A": list(a), "B": list(b)}
        lines = []
        for i, arm in enumerate(order, start=1):
            v = values[arm].pop(0)
            for req in (1, 2):
                lines.append(json.dumps({
                    "arm": arm, "round": i, "request": req, "decode_tok_s": v,
                    "prompt_tokens": 22184, "completion_tokens": 512,
                    "output_sha256": "same" if same or arm == "A" else "other"}))
            (out / f"clocks_{i}.csv").write_text("2520, 10501, 61, 300.5\n2505, 10501, 63, [N/A]\n")
        (out / "rounds.jsonl").write_text("\n".join(lines) + "\n")
        (out / "meta.txt").write_text("gpu NVIDIA GeForce RTX 4090\nrounds per build 6\n")
        argv = sys.argv
        sys.argv = ["step_ab.py", "summarize", str(out)] + (
            ["--rounds", str(rounds)] if rounds else [])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                step_ab.main()
        finally:
            sys.argv = argv
        return (out / "summary.md").read_text()

    def test_no_slower_within_noise(self):
        text = self.run_summary([100.0, 101.0, 100.5, 99.5, 100.0, 100.2],
                                [100.1, 100.0, 99.8, 100.6, 100.3, 99.9])
        self.assertIn("Verdict: no slower", text)
        self.assertIn("Noise (half the larger arm spread): 0.75 %", text)
        self.assertIn("| 1 | A | 100.00 | 100.00, 100.00 | 2505 / 2512 / 2520 | 63 |", text)
        self.assertIn("identical in every request of both arms", text)

    def test_slower_beyond_noise(self):
        text = self.run_summary([100.0, 100.2, 100.1, 99.9, 100.0, 100.1],
                                [97.0, 97.1, 96.9, 97.0, 97.2, 97.0], same=False)
        self.assertIn("Verdict: SLOWER", text)
        self.assertIn("NOT identical", text)

    def test_fewer_than_six_rounds_refused(self):
        # Proposal P's six repeats are frozen: four rounds a build give no verdict, even without
        # --rounds.
        with self.assertRaises(SystemExit) as raised:
            self.run_summary([100.0, 100.1, 99.9, 100.0], [100.2, 100.0, 100.1, 99.8],
                             order="ABBAABBA")
        self.assertIn("6 each needed (frozen)", str(raised.exception))

    def test_more_than_six_rounds_refused(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_summary([100.0] * 8, [100.1] * 8, order="ABBA" * 4)
        self.assertIn("6 each needed (frozen)", str(raised.exception))

    def test_rounds_argument_as_the_kit_passes_it(self):
        text = self.run_summary([100.0, 101.0, 100.5, 99.5, 100.0, 100.2],
                                [100.1, 100.0, 99.8, 100.6, 100.3, 99.9], rounds=6)
        self.assertIn("Verdict: no slower", text)


if __name__ == "__main__":
    unittest.main()
