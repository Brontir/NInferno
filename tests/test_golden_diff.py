import io
import re
import subprocess
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

    def test_c1_key_on_one_side_only(self):
        speed = "q4 p1 add 8x8 speed T=1..2048 A\n"
        n, _ = changes(speed, speed + "q4 p1 add 8x8 speed-c1 T=1 X\n")
        self.assertEqual(n, 1)  # A -> X at T=1 only

    def test_speed_change_with_c1_pinned_to_old_route(self):
        old = "q4 p1 add 8x8 speed T=1..2048 A\n"
        new = "q4 p1 add 8x8 speed T=1..2048 B\nq4 p1 add 8x8 speed-c1 T=1..16 A\n"
        self.assertEqual(changes(old, new)[0], len(gd.KTOKENS))  # the pinned c1 is no change of its own

    def test_speed_change_counted_once_with_c1_present(self):
        old = "q4 p1 add 8x8 speed T=1..2048 A\nq4 p1 add 8x8 speed-c1 T=1 A\n"
        new = "q4 p1 add 8x8 speed T=1..2048 B\nq4 p1 add 8x8 speed-c1 T=1 A\n"
        self.assertEqual(changes(old, new)[0], len(gd.KTOKENS))

    def test_c1_max_matches_the_test_source(self):
        src = (Path(__file__).resolve().parent / "ops" / "linear" / "test_selector_golden.cpp").read_text(
            encoding="utf-8"
        )
        m = re.search(r"Profile::SpeedC1 && t > (\d+)\) continue", src)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), gd.C1_MAX)

    def test_duplicate_release_rows_fail(self):
        dup = INDEX + "base/a10 111111111111 7b7cafcde150 13.4.92\n"
        with self.assertRaises(gd.GoldenParseError):
            gd.resolve(FILES, dup)

    def test_digest_only_change_prints_an_info_line(self):
        old = {"base": {"a10.7b7cafcde150.txt": "# a10 x y\n# facts aaa\n" + NEW_A}}
        new = {"base": {"a10.7b7cafcde150.txt": "# a10 x y\n# facts bbb\n" + NEW_A}}
        index = "release 13.4.92\nbase/a10 c180c3754514 7b7cafcde150 13.4.92\n"
        out = []
        self.assertEqual(gd.summarize(old, new, 20, out, index, index), (0, 0, 0))
        self.assertIn("no decision changes (facts digest: aaa -> bbb)", "\n".join(out))

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
            self.assertTrue(out.rstrip("\n").splitlines()[-1].startswith("total: 1 decisions changed"))

    def test_an_info_line_alone_still_ends_with_no_golden_changes(self):
        # The CI gate takes exit 0 as equal goldens only with this line: a facts digest that moved
        # with no decision change must not read as a tool failure.
        index = "release 13.4.92\nbase/a10 c180c3754514 7b7cafcde150 13.4.92\n"
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/facts_index.txt": index, "b/facts_index.txt": index,
                     "a/base/a10.7b7cafcde150.txt": "# a10 x y\n# facts aaa\n" + NEW_A,
                     "b/base/a10.7b7cafcde150.txt": "# a10 x y\n# facts bbb\n" + NEW_A})
            code, out, _ = run(f"{d}/a", f"{d}/b", "--fail-on-change")
            self.assertEqual(code, 0)
            lines = out.rstrip("\n").splitlines()
            self.assertIn("no decision changes (facts digest: aaa -> bbb)", out)
            self.assertEqual(lines[-1], "no golden changes")


def split_board(text):
    """A single-file dump as the per-format layout writes it: {file name: text}."""
    files = {gd.HEADER_FILE: ""}
    for line in text.splitlines(keepends=True):
        name = gd.HEADER_FILE if line.startswith("#") else line.split(" ", 1)[0] + ".txt"
        files[name] = files.get(name, "") + line
    return files


def git_repo(d, *commits):
    """A repository at d with one commit per {path: text or None (deleted)}; returns the shas."""

    def g(*args):
        return subprocess.run(["git", "-C", d, *args], check=True, capture_output=True, text=True).stdout

    g("init", "-q")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "t")
    g("config", "commit.gpgsign", "false")
    shas = []
    for files in commits:
        for rel, text in files.items():
            p = Path(d) / rel
            if text is None:
                p.unlink()
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8", newline="\n")
        g("add", "-A")
        g("commit", "-q", "-m", "c")
        shas.append(g("rev-parse", "HEAD").strip())
    return shas


class LayoutTest(unittest.TestCase):
    """Both layouts are read (#620 step 8): one file per board, and a board directory with
    header.txt and one file per format."""

    def test_header_file_matches_the_test_source(self):
        src = (Path(__file__).resolve().parent / "ops" / "linear" / "test_selector_golden.cpp").read_text(
            encoding="utf-8"
        )
        m = re.search(r'kHeaderFile\s*=\s*"([^"]+)"', src)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), gd.HEADER_FILE)

    def test_directory_reads_as_the_single_file(self):
        files = split_board(BASE)
        self.assertEqual(sorted(files), ["bf16.txt", "header.txt", "q4.txt"])
        a, b = gd.parse_checked(BASE, "v100.txt"), gd.parse_checked(files, "v100/")
        self.assertEqual((a.header, a.not_packed, dict(a.runs)), (b.header, b.not_packed, dict(b.runs)))

    def test_layout_change_alone_is_no_change(self):
        out = []
        old, new = {"full": {"v100.txt": BASE}}, {"full": {"v100/": split_board(BASE)}}
        self.assertEqual(gd.summarize(old, new, 20, out), (0, 0, 0))
        self.assertEqual(out, [])
        self.assertEqual(gd.summarize(new, old, 20, out), (0, 0, 0))

    def test_change_across_the_layouts_is_counted(self):
        out = []
        old, new = {"full": {"v100.txt": BASE}}, {"full": {"v100/": split_board(HEAD)}}
        self.assertEqual(gd.summarize(old, new, 20, out)[0], 5)
        self.assertIn("full/v100 (file v100.txt -> v100/): 5 decisions changed in 1 problems", out)

    def test_bad_directories_fail(self):
        files = split_board(BASE)
        cases = {
            "no header": {k: v for k, v in files.items() if k != gd.HEADER_FILE},
            "misplaced line": dict(files, **{"bf16.txt": files["bf16.txt"] + files["q4.txt"]}),
            "comment in a format file": dict(files, **{"q4.txt": "# v100 a b\n" + files["q4.txt"]}),
            "decision in the header": dict(files, **{gd.HEADER_FILE: files[gd.HEADER_FILE] + files["q4.txt"]}),
            "overlap": dict(files, **{"q4.txt": files["q4.txt"] + "q4 p0 store 256x2048 speed T=1 x\n"}),
        }
        for what, case in cases.items():
            with self.subTest(what=what), self.assertRaises(gd.GoldenParseError):
                gd.parse_checked(case, "full/v100/")

    def test_both_layouts_of_one_board_fail(self):
        with self.assertRaises(gd.GoldenParseError):
            gd.resolve({"full": {"v100.txt": BASE, "v100/": split_board(BASE)}}, None)

    def test_dirs_cli_across_the_layouts(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, {"a/full/v100.txt": BASE, "a/full/l4.txt": BASE})
            tree(d, {f"b/full/v100/{k}": v for k, v in split_board(BASE).items()})
            tree(d, {f"c/full/v100/{k}": v for k, v in split_board(HEAD).items()})
            code, out, _ = run(f"{d}/a", f"{d}/b", "--fail-on-change")
            self.assertEqual(code, 1)
            self.assertIn("full/l4: REMOVED board file (5 runs)", out)
            self.assertNotIn("full/v100", out)
            tree(d, {"b/full/l4.txt": BASE})
            self.assertEqual(run(f"{d}/a", f"{d}/b", "--fail-on-change")[:2], (0, "no golden changes\n"))
            code, out, _ = run(f"{d}/b", f"{d}/c")
            self.assertIn("full/v100: 5 decisions changed in 1 problems", out)

    def test_revisions_across_the_layout_change(self):
        rel = gd.GOLDEN_REL + "/full/"
        split = {rel + "v100/" + k: v for k, v in split_board(BASE).items()}
        moved = {rel + "v100/" + k: v for k, v in split_board(HEAD).items()}
        with tempfile.TemporaryDirectory() as d:
            one, two, three = git_repo(d, {rel + "v100.txt": BASE}, dict(split, **{rel + "v100.txt": None}), moved)
            code, out, err = run("--repo", d, "--base", one, "--head", two, "--fail-on-change")
            self.assertEqual((code, out, err), (0, "no golden changes\n", ""))
            code, out, _ = run("--repo", d, "--base", one, "--head", three, "--fail-on-change")
            self.assertEqual(code, 1)
            self.assertIn("full/v100 (file v100.txt -> v100/): 5 decisions changed in 1 problems", out)
            code, out, _ = run("--repo", d, "--range", f"{two}...{three}")
            self.assertIn("full/v100: 5 decisions changed in 1 problems", out)

    def test_missing_blob_is_a_tool_failure(self):
        with tempfile.TemporaryDirectory() as d:
            (one,) = git_repo(d, {"x.txt": "x\n"})
            with self.assertRaises(gd.GoldenParseError):
                gd.read_blobs(one, ["no/such/file.txt"], d)


if __name__ == "__main__":
    unittest.main()
