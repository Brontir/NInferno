import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import golden_diff as gd

BASE = """# v100 gv100:80sm sm70
bf16 256x2048 not packed: row-split test packer: unsupported qtype
q4 p0 store 256x2048 speed T=1..16 rmma.m1.n2.w4x1 ws=0
q4 p0 store 256x2048 speed T=17..2048 tiled.128x128 ws=0
q4 p1 store 256x2048 speed T=1 r8.w4r1 ws=0
q4 p1 store 256x2048 speed T=2 r8.w4r1 ws=5120
q4 p1 store 256x2048 prefill T=17..2048 tiled.128x128 ws=0
"""

HEAD = """# v100 gv100:80sm sm70
bf16 256x2048 not packed: row-split test packer: unsupported qtype
q4 p0 store 256x2048 speed T=1..16 rmma.m1.n2.w4x1 ws=0
q4 p0 store 256x2048 speed T=17..64 tiled.64x128 ws=0
q4 p0 store 256x2048 speed T=65..2048 tiled.128x128 ws=0
q4 p1 store 256x2048 speed T=1 r8.w4r1 ws=0
q4 p1 store 256x2048 speed T=2 r8.w4r1 ws=5120
q4 p1 store 256x2048 prefill T=17..2048 tiled.128x128 ws=0
"""


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = gd.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def tree(root, files):
    for rel, text in files.items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


class ParseTest(unittest.TestCase):
    def test_parses_all_line_kinds(self):
        g = gd.parse_golden("# facts abc123\n" + BASE)
        self.assertEqual(g.facts, "abc123")
        self.assertEqual(g.header, "v100 gv100:80sm sm70")
        self.assertEqual(len(g.not_packed), 1)
        self.assertEqual(g.runs[("q4", "p0", "store", "256x2048", "speed")][1], (17, 2048, "tiled.128x128 ws=0"))

    def test_route_text_keeps_annotations(self):
        g = gd.parse_golden("q4 p0 add 8x8 speed T=3 x ws=0 runs=y (demoted)\n")
        self.assertEqual(g.runs[("q4", "p0", "add", "8x8", "speed")], [(3, 3, "x ws=0 runs=y (demoted)")])

    def test_unknown_line_fails(self):
        with self.assertRaises(gd.GoldenParseError):
            gd.parse_golden("# v100 a b\nwhat is this\n")
        with self.assertRaises(gd.GoldenParseError):
            gd.parse_golden("# one two\n")

    def test_overlap_fails(self):
        with self.assertRaises(gd.GoldenParseError):
            gd.parse_golden("q4 p0 add 8x8 speed T=1..8 a\nq4 p0 add 8x8 speed T=8..9 b\n")


class DiffTest(unittest.TestCase):
    def test_runs_collapse_on_sampled_widths(self):
        grid = [17, 24, 32, 64, 96, 128]
        segs = gd.diff_runs([(17, 128, "a")], [(17, 32, "b"), (64, 64, "b"), (96, 128, "a")], grid)
        self.assertEqual(segs, [(17, 64, "a", "b", 4)])

    def test_unsampled_gap_is_not_a_change(self):
        # T=48..128 vs T=48..192: widths 129..191 are not sampled, so only 192 differs.
        grid = [48, 128, 192, 256]
        segs = gd.diff_runs([(48, 128, "m"), (192, 192, "w"), (256, 256, "w")],
                            [(48, 192, "m"), (256, 256, "w")], grid)
        self.assertEqual(segs, [(192, 192, "w", "m", 1)])

    def test_absent_and_merge(self):
        grid = [1, 2, 3]
        segs = gd.diff_runs([(1, 1, "a")], [(1, 2, "b"), (3, 3, "c")], grid)
        self.assertEqual(segs, [(1, 1, "a", "b", 1), (2, 2, None, "b", 1), (3, 3, None, "c", 1)])

    def test_pairing_by_board_when_digest_moves(self):
        pairs = gd.pair_files(["rtx3090.aaaaaaaaaaaa.txt", "v100.txt"], ["rtx3090.bbbbbbbbbbbb.txt", "v100.txt"])
        self.assertIn(("rtx3090.aaaaaaaaaaaa.txt", "rtx3090.bbbbbbbbbbbb.txt"), pairs)
        self.assertIn(("v100.txt", "v100.txt"), pairs)


class CliTest(unittest.TestCase):
    def test_equal_trees(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/full/v100.txt": BASE, "b/full/v100.txt": BASE})
            code, out, _ = run(f"{d}/a", f"{d}/b", "--fail-on-change")
            self.assertEqual((code, out.strip()), (0, "no golden changes"))

    def test_summary_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/full/v100.txt": BASE, "b/full/v100.txt": HEAD})
            code, out, _ = run(f"{d}/a", f"{d}/b")
            self.assertEqual(code, 0)
            self.assertIn("full/v100.txt: 2 decisions changed in 1 problems", out)
            self.assertIn("T 17..64: tiled.128x128 ws=0  ->  tiled.64x128 ws=0", out)
            self.assertIn("q4 p0 store 256x2048 speed", out)
            self.assertEqual(run(f"{d}/a", f"{d}/b", "--fail-on-change")[0], 1)

    def test_added_file_and_meta(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/full/v100.txt": BASE, "b/full/v100.txt": BASE.replace("sm70", "sm71"), "b/full/l4.txt": BASE})
            code, out, _ = run(f"{d}/a", f"{d}/b", "--fail-on-change")
            self.assertEqual(code, 1)
            self.assertIn("full/l4.txt: ADDED", out)
            self.assertIn("header:", out)

    def test_unknown_line_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/full/v100.txt": BASE, "b/full/v100.txt": BASE + "garbage\n"})
            code, _, err = run(f"{d}/a", f"{d}/b")
            self.assertEqual(code, 2)
            self.assertIn("unknown line", err)


if __name__ == "__main__":
    unittest.main()
