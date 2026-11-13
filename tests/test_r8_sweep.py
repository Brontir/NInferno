"""tools/bench/r8_decode_table.py and r8_sweep_summary.py read ninferno_r8_schedule_bench's
output as the bench prints it (t=, decode=, a16pick and h= since #155), synthetic sweeps, no GPU."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import r8_decode_table as tool  # noqa: E402
import r8_sweep_summary as summary  # noqa: E402

HEAD = ("device NVIDIA GeForce RTX 4090 class ad102:128sm, L2 72 MB, 15 samples x 60 launches\n"
        "board cc 89 sms 128 l2_mib 72 dram_gbps 1008 pci 0x2684\n")


def row(fmt, epi, n, k, sched, us, marks="", t=1, same=1):
    return (f"r8 {fmt} {epi} n={n} k={k} t={t} r8.{sched} {us:.2f} us 850.0 GB/s same={same} "
            f"decode=1{marks} h=0123456789abcdef\n")


def sweep(w8r1, w8r2=48.70, extra=""):
    text = HEAD
    for epi in ("store", "add"):
        text += row("q6k", epi, 4096, 12288, "w4r1", 47.10)
        text += row("q6k", epi, 4096, 12288, "w4r2", 49.00)
        text += row("q6k", epi, 4096, 12288, "w8r1", w8r1)
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
        run, base, board = tool.read_run(paths[0], "ad102:128sm")
        key = ("q6k", "store", 4096, 12288)
        self.assertEqual(board, "0x2684")
        self.assertEqual(base[key], (8, 2))
        self.assertEqual(sorted(run[key]), [(4, 1), (4, 2), (8, 1), (8, 2)])
        self.assertEqual(run[key][(8, 1)], 45.00)  # the t=2 row did not overwrite it
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

    def test_summary_verdicts(self):
        paths = self.files([sweep(45.50), sweep(45.60), sweep(45.40)])
        out = io.StringIO()
        argv = sys.argv
        sys.argv = ["r8_sweep_summary.py", *paths]
        try:
            with contextlib.redirect_stdout(out):
                summary.main()
        finally:
            sys.argv = argv
        text = out.getvalue()
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


if __name__ == "__main__":
    unittest.main()
