import io
import re
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
q4 p0 store 256x2048 speed T=96..2048 tiled.128x128 ws=0
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


def changes(old_text, new_text):
    res = gd.diff_golden(gd.parse_golden(old_text), gd.parse_golden(new_text))
    return sum(gd.widths(v) for v in res["segments"].values()), res


class DiffTest(unittest.TestCase):
    def test_sampled_widths_match_the_test_source(self):
        src = (Path(__file__).resolve().parent / "ops" / "linear" / "test_selector_golden.cpp").read_text(
            encoding="utf-8"
        )
        m = re.search(r"kTokens\[\]\s*=\s*\{([^}]*)\}", src)
        self.assertIsNotNone(m)
        self.assertEqual(tuple(int(x) for x in m.group(1).split(",")), gd.KTOKENS)

    def test_run_end_moves_over_unsampled_widths_is_no_change(self):
        n, _ = changes("q4 p1 add 8x8 speed T=48..128 m\nq4 p1 add 8x8 speed T=192 w\n",
                       "q4 p1 add 8x8 speed T=48..192 m\n")
        self.assertEqual(n, 1)  # only T=192 changed (129..191 are not sampled)

    def test_counts_every_sampled_width(self):
        n, res = changes("q4 p1 add 8x8 speed T=1..2048 A\n", "q4 p1 add 8x8 speed T=1 A\nq4 p1 add 8x8 speed T=2..2048 B\n")
        self.assertEqual(n, len(gd.KTOKENS) - 1)
        (segs,) = res["segments"].values()
        self.assertEqual(segs, [(2, 2048, "A", "B", len(gd.KTOKENS) - 1)])

    def test_speed_c1_hole_is_read_as_speed(self):
        speed = "q4 p1 add 8x8 speed T=1..16 C\n"
        old = speed + "q4 p1 add 8x8 speed-c1 T=1..16 C\n"
        new = speed + "q4 p1 add 8x8 speed-c1 T=1 C\nq4 p1 add 8x8 speed-c1 T=16 C\n"
        self.assertEqual(changes(old, new)[0], 0)  # the holes answer as speed = C
        # the speed route moves: the omitted c1 widths move with it, only what is written is a hole
        n, _ = changes(old, speed.replace(" C", " D") + "q4 p1 add 8x8 speed-c1 T=1 C\nq4 p1 add 8x8 speed-c1 T=16 C\n")
        self.assertGreater(n, 0)

    def test_speed_c1_without_speed_row_stays_absent(self):
        n, _ = changes("q4 p1 add 8x8 speed-c1 T=1..16 C\n", "q4 p1 add 8x8 speed-c1 T=1 C\n")
        self.assertGreater(n, 0)

    def test_unsampled_width_fails(self):
        with self.assertRaises(gd.GoldenParseError):
            gd.check_widths(gd.parse_golden("q4 p1 add 8x8 speed T=7 A\n"), "x")

    def test_split_name(self):
        self.assertEqual(gd.split_name("rtx3090.0ae580162b29.txt"), ("rtx3090", "0ae580162b29"))
        self.assertEqual(gd.split_name("v100.txt"), ("v100", None))


INDEX = """# comment
release 13.4.92
base/a10 0697b549a22f 0697b549a22f 12.8.93
base/a10 c180c3754514 7b7cafcde150 13.4.92
base/a10 7b7cafcde150 7b7cafcde150 13.5.37
"""
OLD_A = "q4 p1 add 8x8 speed T=1 A\n"
NEW_A = "q4 p1 add 8x8 speed T=1 B\n"
FILES = {
    "base": {
        "a10.0697b549a22f.txt": OLD_A,
        "a10.7b7cafcde150.txt": NEW_A,
        "v100.txt": OLD_A,
    }
}


class IndexTest(unittest.TestCase):
    def test_release_file_chosen_and_other_files_ignored(self):
        r = gd.resolve(FILES, INDEX)
        self.assertEqual(r[("base", "a10")][0], "a10.7b7cafcde150.txt")
        self.assertEqual(r[("base", "v100")][0], "v100.txt")

    def test_several_files_without_row_fail(self):
        with self.assertRaises(gd.GoldenParseError):
            gd.resolve(FILES, None)

    def test_repoint_is_a_change_without_any_txt_diff(self):
        out = []
        moved = INDEX.replace("c180c3754514 7b7cafcde150 13.4.92", "c180c3754514 0697b549a22f 13.4.92")
        changed, meta, files = gd.summarize(FILES, FILES, 20, out, INDEX, moved)
        self.assertEqual((files > 0), False)
        self.assertGreater(changed + meta, 0)
        self.assertIn("a10.0697b549a22f.txt", "\n".join(out))

    def test_same_release_file_digest_change_pairs_without_added_removed(self):
        new = {"base": {"a10.7b7cafcde150.txt": NEW_A, "v100.txt": OLD_A}}
        new_index = "release 13.4.92\nbase/a10 c180c3754514 7b7cafcde150 13.4.92\n"
        changed, meta, files = gd.summarize(FILES, new, 20, [], INDEX, new_index)
        self.assertEqual((changed, files), (0, 0))

    def test_post_629_layout_pairs_with_release_file(self):
        new = {"base": {"a10.txt": NEW_A, "v100.txt": OLD_A}}
        changed, meta, files = gd.summarize(FILES, new, 20, [], INDEX, None)
        self.assertEqual((changed, files), (0, 0))
        self.assertEqual(meta, 0)  # the facts_index note vanished: information only


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
            self.assertIn("full/v100: 5 decisions changed in 1 problems", out)
            self.assertIn("T 17..64: tiled.128x128 ws=0  ->  tiled.64x128 ws=0", out)
            self.assertIn("q4 p0 store 256x2048 speed", out)
            self.assertEqual(run(f"{d}/a", f"{d}/b", "--fail-on-change")[0], 1)

    def test_added_file_and_meta(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/full/v100.txt": BASE, "b/full/v100.txt": BASE.replace("sm70", "sm71"), "b/full/l4.txt": BASE})
            code, out, _ = run(f"{d}/a", f"{d}/b", "--fail-on-change")
            self.assertEqual(code, 1)
            self.assertIn("full/l4: ADDED", out)
            self.assertIn("header:", out)

    def test_unknown_line_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/full/v100.txt": BASE, "b/full/v100.txt": BASE + "garbage\n"})
            code, _, err = run(f"{d}/a", f"{d}/b")
            self.assertEqual(code, 2)
            self.assertIn("unknown line", err)


    def test_bad_detail_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            run("a", "b", "--detail", "x")

    def test_dirs_with_facts_index(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/facts_index.txt": INDEX, "b/facts_index.txt": INDEX,
                     "a/base/a10.0697b549a22f.txt": OLD_A, "a/base/a10.7b7cafcde150.txt": OLD_A,
                     "b/base/a10.0697b549a22f.txt": OLD_A, "b/base/a10.7b7cafcde150.txt": NEW_A})
            code, out, _ = run(f"{d}/a", f"{d}/b", "--fail-on-change")
            self.assertEqual(code, 1)
            self.assertIn("base/a10: 1 decisions changed", out)


if __name__ == "__main__":
    unittest.main()
