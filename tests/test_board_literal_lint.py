#!/usr/bin/env python3
"""Unit tests of tools/audit/board_literal_lint.py, and the tree is clean."""
import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "board_literal_lint", ROOT / "tools" / "audit" / "board_literal_lint.py")
lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint)

# Every spelling of a hand-written board the lint must see (one flagged line each).
FLAGGED = {
    "plain": "auto b = Board{12, 0, 170, 96, 1792};",
    "namespaced": "const ninferno::Board b{8, 9, 128, 72, 1008};",
    "global_qualified": "::ninferno::Board b{12, 0, 170, 96, 1792};",
    "global_qualified_named_first": "::ninferno::Board b{kMajor, 0, 170, 96, 1792};",
    "global_qualified_table": "::ninferno::Board k[] = {{8, 6, 82}};",
    "equals": "const Board b = {8, 9, 128, 72, 1008};",
    "space": "x(Board {7, 5, 68, 5.5, 616})",
    "multiline": "inline constexpr Board kMine{\n    12, 0, 170, 96, 1792};",
    "multiline_comment_inside": "Board b{12, /* cc */ 0,\n  170, 96, 1792};",
    "designated": "Board k{.major = 12, .minor = 0, .sms = 170, .l2_mib = 96, .dram_gbps = 1792};",
    "parens": "Board(12, 0, 170, 96, 1792)",
    "parens_decl": "Board b(12, 0, 170, 96, 1792);",
    "array": "Board kBoards[] = {{12, 0, 170, 96, 1792}, {8, 9, 128, 72, 1008}};",
    "std_array": "std::array<Board, 1> k{{{12, 0, 170, 96, 1792}}};",
    "vector": "std::vector<Board> v = {{12, 0, 170, 96, 1792}};",
    "optional": "std::optional<Board> o = Board{12, 0, 170, 96, 1792};",
    "optional_ctor": "std::optional<Board> o{8, 6, 82, 6, 936};",
    "return": "Board f() { return {12, 0, 170, 96, 1792}; }",
    "argument": "g(1, {12, 0, 170, 96, 1792});",
    "assign": "b = {12, 0, 170, 96, 1792};",
    "const_after": "Board const b{12, 0, 170, 96, 1792};",
    "const_ref": "const Board& b = {12, 0, 170, 96, 1792};",
    "alias": "using B = Board;\nB b{12, 0, 170, 96, 1792};",
    "alias_ns": "using B = ninferno::Board;\nB b(12, 0, 170, 96, 1792);",
    "typedef": "typedef Board B;\nB b{12, 0, 170, 96, 1792};",
    "paren_first": "Board b{(12), 0, 170, 96, 1792};",
    "named_first": "Board b{kMajor, 0, 170, 96, 1792};",
    "named_field": "Board b{8, 0, kSms, 96, 1792};",
    "expression_first": "Board b{6 + 6, 0, 170, 96, 1792};",
    "table_row": '{"ga100:108sm", {8, 0, 108, 40, 1555}, kTable},',
    "field": '{.gpu_class = "gb202", .board = {12, 0, 188, 128, 1792}},',
    "field_named": "{.board = {kMajor, 0, 188, 128, 1792}},",
    "field_table": "{.boards = Board{kMajor, 0, 188, 128, 1792}},",
    "float_l2": "x{7, 5, 68, 5.5, 616}",
    "digit_separator": "Board b{12, 0, 170, 96, 1'792};",
    "mma_slow": "Board b{7, 5, 68, 5.5, 616, true};",
    "macro": "#define MINE Board{12, 0, \\\n  170, 96, 1792}",
    "hex": "Board b{0xC, 0, 170, 96, 1792};",
    "after_string_glob": 'const char* kGlob = "/*.gguf";\nBoard b{12, 0, 170, 96, 1792};',
    "after_char_quote": "char c = '\"'; Board b{12, 0, 170, 96, 1792};",
    "after_raw_string": 'auto s = R"x(/* )x";\nBoard b{12, 0, 170, 96, 1792};',
    # Tables of boards: every row of a typed table is tested.
    "table_named_first": "Board kBoards[] = {{kMajor, 0, 170, 96, 1792}};",
    "table_named_field": "Board kBoards[] = {{8, 0, kSms, 96, 1792}};",
    "table_designated": "const Board k[] = {{.major = 8, .minor = 6, .sms = 82, .l2_mib = 6,"
                        " .dram_gbps = 936}};",
    "table_short_rows": "std::vector<Board> v = {{8, 6, 82}, {8, 9, 128}};",
    "table_two_fields": "Board k[] = {{12, 0}};",
    "table_second_row_only": "Board k[] = {measured::kRtx5090, {8, 9, 128, 72, 1008}};",
    # Integer suffixes on a field: the tuple is still a board.
    "tuple_suffixed_hex": "Board b = f({0xCu, 0, 170, 96, 1792});",
    "tuple_suffixed_hex_ull": "x = {12, 0, 0xAAul, 96, 0x700ULL};",
    "tuple_suffixed_decimal": "x = {12u, 0, 170L, 96, 1792ull};",
    # A function returning Board: its braced returns are board literals.
    "return_named_fields": "Board f() { return {kMajor, 0, 170, 96, kBw}; }",
    "return_const_ref_qualified": "const Board& g(int a) const noexcept"
                                  " { return {8, 6, kS, 6, 9}; }",
    "return_through_alias": "using B = Board; B f(int x) { if (x) return {kM, 0, 1, 2, 3};"
                            " return {}; }",
    # A template argument with a call in it.
    "std_array_call_size": "std::array<Board, board_count()> a = {{{kMajor, 0, 170, 96, kBw}}};",
    # Qualified constants and the spellings of an alias.
    "qualified_constant": "Board b{facts::kMajor, 0, 170, 96, kBw};",
    "globally_qualified_constant": "Board b{::ns::facts::kMajor, 0, 170, 96, kBw};",
    "typedef_global_qualified": "typedef ::ninferno::Board B; B b{kMajor, 0, 170, 96, kBw};",
    "using_const_alias": "using B = const Board; B b{kMajor, 0, 170, 96, kBw};",
    "typedef_const_after": "typedef Board const B; B b{kMajor, 0, 170, 96, kBw};",
    "std_array_short_rows": "std::array<Board, 2> a{{{8, 6, 82}, {8, 9, 128}}};",
    "std_array_designated": "std::array<Board, 1> a{{{.major = 8, .minor = 6, .sms = 82}}};",
    "std_array_size_name": "std::array<Board, kN> a = {{{8, 6, 82}}};",
    "table_multiline": "const Board k[] = {\n    {8, 6, 82,\n     6, kBw},\n};",
    "global_table_named": "::ninferno::Board k[] = {{kMajor, 0, 170, 96, 1792}};",
    # A trailing comma, as clang-format writes a long row.
    "tuple_trailing_comma": "{8, 6, 82, 6, 936,}",
    "tuple_trailing_comma_multiline": "push({\n    8, 6, 82, 6, 936,\n});",
    "tuple_bool_trailing_comma": "{7, 5, 68, 5.5, 616, true,}",
}

CLEAN = {
    "declaration": "struct Board {",
    "declaration_body": "struct Board{ int major = 0; };",
    "conversion": "table->board = Board{cc / 10, cc % 10, sms};",
    "row_fields": "return Board{row->major, row->minor};",
    "name_prefix": "FactsBoard{1, 2}",
    "name_prefix_board": "BoardRow{0x2B85, 12, 0, 170, 96}",
    "count": "int BoardCount{12};",
    "index": "std::size_t BoardIndex = {0};",
    "other_namespace": "chess::Board b{8, 8};",
    "other_namespace_global": "::chess::Board b{8, 8};",
    "other_namespace_template": "std::vector<chess::Board> v{{8, 8}};",
    "member": "x.Board{8, 8};",
    "registry": "kRtx5090 = board_registry::board(0x2B85);",
    "table_of_registry_boards": "std::vector<Board> v = {measured::kRtx5090, board_of(1, 2)};",
    "table_of_names": "Board k[] = {{a, b}, {c, d}};",
    "empty_vector": "std::vector<Board> v{};",
    "not_a_board_array": "std::array<int, 2> a{{1, 2}};",
    "list_of_sizes": "for (int t : {8, 256, 4096, 49152, 65536}) {",
    "ids": "std::vector<int> v{1, 2, 46, 47, 768};",
    "pdf_header": "header.insert(header.end(), {8, 2, 0, 0, 0});",
    "sizes_trailing_comma": "for (int t : {4, 2, 16, 64, 256,}) {",
    "return_from_fields": "Board f(int cc) { return {cc / 10, cc % 10, 3}; }",
    "return_of_another_type": "int h() { return {kA, 1}; }",
    "alias_of_another_type": "using B = chess::Board; B b{kMajor, 0, 1, 2, 3};",
    "list_of_qualified_boards": "std::vector<Board> v = {measured::kA, measured::kB,"
                                " measured::kC};",
    "comment_line": "// was Board{12, 0, 170, 96, 1792}",
    "comment_block": "/* Board{12, 0, 170}\n still */ int x;",
    "comment_block_tuple": "/* {12, 0, 170, 96, 1792}\n */ int x;",
    "string": 'puts("Board{12, 0, 170, 96, 1792}");',
    "string_tuple": 'puts("{12, 0, 170, 96, 1792}");',
    "string_escaped_quote": 'puts("a\\" {12, 0, 170, 96, 1792}");',
}


def tree(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def pin(text):
    """The sorted literals of `text`, as an ALLOW entry holds them."""
    return tuple(sorted(lit for _n, lit, _l in lint.literals(text)))


class Violations(unittest.TestCase):
    def test_flags_every_spelling(self):
        for name, code in FLAGGED.items():
            with self.subTest(name):
                self.assertEqual(len(lint.violations(code)), 1, code)

    def test_ignores_non_boards(self):
        for name, code in CLEAN.items():
            with self.subTest(name):
                self.assertEqual(lint.violations(code), [], code)

    def test_code_after_a_comment_is_seen(self):
        code = "int x; /* a */ Board{1, 2}"
        self.assertEqual(lint.violations(code), [(1, code)])

    def test_line_numbers(self):
        code = "int a;\n// c\nBoard b{12,\n 0, 170, 96, 1792};\n"
        self.assertEqual([n for n, _ in lint.violations(code)], [3])

    def test_every_row_of_a_table_is_a_literal(self):
        code = "Board k[] = {{8, 6, 82}, {kMajor, 9, 128, 72, 1008}, measured::kX, {12, 0}};"
        self.assertEqual([lit for _n, lit, _l in lint.literals(code)],
                         ["{8,6,82}", "{kMajor,9,128,72,1008}", "{12,0}"])

    def test_a_nested_literal_counts_once(self):
        code = "Board b{12, 0, 170, 96, 1792};\nstd::vector<Board> v = {{8, 6, 82, 6, 936}};"
        self.assertEqual([lit for _n, lit, _l in lint.literals(code)],
                         ["Board b{12,0,170,96,1792}", "{8,6,82,6,936}"])

    def test_literal_text_ignores_layout_and_comments(self):
        a = "Board b{12, 0, 170, 96, 1792};"
        b = "Board  b{ 12,   // cc\n 0, 170,\n 96, /* GB/s */ 1792 };"
        self.assertEqual(pin(a), pin(b))


class Scan(unittest.TestCase):
    def test_every_directory_and_suffix_is_scanned(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {
                "src/a.cpp": lit, "tests/b.cpp": lit, "tools/c.cu": lit, "apps/cli/d.cpp": lit,
                "bench/ops/e.cpp": lit, "include/f.h": lit, "examples/g.hpp": lit,
                "src/core/h.inc": lit, "third_party/x/i.cpp": lit, "src/j.txt": lit})
            bad = {r for r, _, _ in lint.scan(root, allow={})}
            self.assertEqual(bad, {"src/a.cpp", "tests/b.cpp", "tools/c.cu", "apps/cli/d.cpp",
                                   "bench/ops/e.cpp", "include/f.h", "examples/g.hpp",
                                   "src/core/h.inc"})

    def test_only_the_third_party_prefixes_are_skipped(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {
                "third_party/spdlog/a.cpp": lit, "src/ui/third_party/x/b.cpp": lit,
                "src/third_party/c.cpp": lit, "src/ops/generic/third_party/d.h": lit,
                "src/my_third_party/e.cpp": lit, "third_party_copy/f.cpp": lit})
            bad = {r for r, _, _ in lint.scan(root, allow={})}
            self.assertEqual(bad, {"src/third_party/c.cpp", "src/ops/generic/third_party/d.h",
                                   "src/my_third_party/e.cpp", "third_party_copy/f.cpp"})

    def test_the_skipped_prefixes_exist_in_the_tree(self):
        self.assertTrue((ROOT / "third_party").is_dir())
        self.assertIn("third_party/", lint.THIRD_PARTY)

    def test_tracked_files_only_in_a_checkout(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            tree(root, {"src/tracked.cpp": lit, "src/untracked.cpp": lit})
            subprocess.run(["git", "-C", str(root), "add", "src/tracked.cpp"], check=True)
            self.assertEqual({r for r, _, _ in lint.scan(root, allow={})}, {"src/tracked.cpp"})

    def test_a_checkout_without_git_fails_with_a_message(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / ".git").mkdir()
            with mock.patch.object(lint.subprocess, "run", side_effect=FileNotFoundError("git")):
                with self.assertRaises(SystemExit) as cm:
                    lint.files(root)
            self.assertIn("git command is not installed", str(cm.exception))


class Pin(unittest.TestCase):
    """The allow-list pins the literals of a file, not only how many lines hold one."""
    R8 = ('const Row kRows[] = {\n  {"a100", {8, 0, 108, 40, 1555}, kT},\n'
          '  {"a40", {8, 6, 82, 6, 936}, kU},\n};\n')

    def run_scan(self, text, pinned=None):
        pinned = pin(self.R8) if pinned is None else pinned
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {"src/r8.h": text})
            return lint.scan(root, allow={"src/r8.h": (pinned, "x")})

    def test_unchanged_file_passes_whatever_its_layout(self):
        self.assertEqual(self.run_scan(self.R8), [])
        spaced = self.R8.replace("{8, 0,", "{ 8, // cc\n 0,")
        self.assertEqual(self.run_scan(spaced), [])

    def test_a_replaced_board_fails(self):  # plant 1: the row regenerated from another board
        bad = self.run_scan(self.R8.replace("{8, 0, 108, 40, 1555}", "{8, 0, 108, 80, 2039}"))
        self.assertEqual(len(bad), 2)
        self.assertEqual({n > 0 for _r, n, _t in bad}, {True, False})

    def test_a_second_board_on_a_flagged_line_fails(self):  # plant 2
        bad = self.run_scan(self.R8.replace(
            '{"a100", {8, 0, 108, 40, 1555}, kT},',
            '{"a100", {8, 0, 108, 40, 1555}, kT}, {"b", {8, 9, 128, 72, 1008}, kT},'))
        self.assertEqual([(r, n > 0) for r, n, _t in bad], [("src/r8.h", True)])

    def test_a_hypothetical_swapped_for_a_real_board_fails(self):  # plant 3
        hypo = "Board h{12, 0, 170, 0, 1792};\n"
        pinned = pin(hypo)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {"tests/x.cpp": "Board h{8, 6, 82, 6, 936};\n"})
            bad = lint.scan(root, allow={"tests/x.cpp": (pinned, "x")})
        self.assertEqual(sorted(n > 0 for _r, n, _t in bad), [False, True])

    def test_a_removed_literal_and_an_added_one_elsewhere_fail(self):  # plant 4: same count
        text = self.R8.replace('  {"a40", {8, 6, 82, 6, 936}, kU},\n', "") + "Board b{8, 9, 34, 32, 288};\n"
        bad = self.run_scan(text)
        self.assertEqual(sorted(n > 0 for _r, n, _t in bad), [False, True])

    def test_a_removed_literal_fails(self):
        bad = self.run_scan(self.R8.replace('  {"a40", {8, 6, 82, 6, 936}, kU},\n', ""))
        self.assertEqual([(r, n) for r, n, _t in bad], [("src/r8.h", 0)])
        self.assertIn("{8,6,82,6,936}", bad[0][2])

    def test_a_new_line_fails_with_its_line_number(self):
        bad = self.run_scan(self.R8 + "Board c{12, 0, 170, 96, 1792};\n")
        self.assertEqual([(r, n) for r, n, _t in bad], [("src/r8.h", 5)])

    def test_a_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {"src/other.cpp": "int x;\n"})
            bad = lint.scan(root, allow={"src/missing.h": (("{8,6,82,6,936}",), "x")})
        self.assertEqual([(r, n) for r, n, _t in bad], [("src/missing.h", 0)])

    def test_a_duplicate_literal_is_pinned_twice(self):
        two = "Board a{8, 6, 82, 6, 936};\nBoard a{8, 6, 82, 6, 936};\n"
        self.assertEqual(len(pin(two)), 2)
        self.assertEqual(self.run_scan(two, pin(two)), [])
        self.assertEqual(len(self.run_scan(two, pin(two)[:1])), 1)


class Main(unittest.TestCase):
    def run_main(self, files, allow=None):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, files)
            with contextlib.redirect_stdout(out):
                code = lint.main(root, {} if allow is None else allow)
        return code, out.getvalue()

    def test_exit_0_on_a_clean_tree(self):
        code, out = self.run_main({"src/a.cpp": "int x;\n"})
        self.assertEqual((code, out), (0, ""))

    def test_exit_1_on_a_new_literal(self):
        code, out = self.run_main({"src/a.cpp": "Board b{8, 6, 82, 6, 936};\n"})
        self.assertEqual(code, 1)
        self.assertIn("src/a.cpp:1: board literal outside the registry", out)
        self.assertIn("name the board by PCI id", out)
        self.assertNotIn("pin is stale", out)

    def test_exit_1_on_a_stale_pin_with_its_own_message(self):
        code, out = self.run_main({"src/a.cpp": "int x;\n"},
                                  {"src/a.cpp": (("{8,6,82,6,936}",), "reason of the pin")})
        self.assertEqual(code, 1)
        self.assertIn("pin is stale", out)
        self.assertNotIn("name the board by PCI id", out)


class AllowList(unittest.TestCase):
    def test_entries_exist_with_literals_and_a_reason(self):
        for rel, (lits, reason) in lint.ALLOW.items():
            self.assertTrue((ROOT / rel).is_file(), rel)
            self.assertGreater(len(lits), 0, rel)
            self.assertEqual(list(lits), sorted(lits), rel)
            self.assertGreater(len(reason), 20, rel)

    def test_tree_is_clean(self):
        bad = lint.scan(ROOT)
        self.assertEqual(bad, [], "\n".join(f"{r}:{n}: {l}" for r, n, l in bad))


if __name__ == "__main__":
    unittest.main()
