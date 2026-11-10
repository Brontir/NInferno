#!/usr/bin/env python3
"""Unit tests of tools/audit/board_literal_lint.py, and the tree is clean."""
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "board_literal_lint", ROOT / "tools" / "audit" / "board_literal_lint.py")
lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint)

# Every spelling of a hand-written board the lint must see (one flagged line each).
FLAGGED = {
    "plain": "auto b = Board{12, 0, 170, 96, 1792};",
    "namespaced": "const ninferno::Board b{8, 9, 128, 72, 1008};",
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
    "expression_first": "Board b{6 + 6, 0, 170, 96, 1792};",
    "table_row": '{"ga100:108sm", {8, 0, 108, 40, 1555}, kTable},',
    "field": '{.gpu_class = "gb202", .board = {12, 0, 188, 128, 1792}},',
    "field_named": "{.board = {kMajor, 0, 188, 128, 1792}},",
    "float_l2": "x{7, 5, 68, 5.5, 616}",
    "digit_separator": "Board b{12, 0, 170, 96, 1'792};",
    "mma_slow": "Board b{7, 5, 68, 5.5, 616, true};",
    "macro": "#define MINE Board{12, 0, \\\n  170, 96, 1792}",
    "hex": "Board b{0xC, 0, 170, 96, 1792};",
    "after_string_glob": 'const char* kGlob = "/*.gguf";\nBoard b{12, 0, 170, 96, 1792};',
    "after_char_quote": "char c = '\"'; Board b{12, 0, 170, 96, 1792};",
    "after_raw_string": 'auto s = R"x(/* )x";\nBoard b{12, 0, 170, 96, 1792};',
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
    "registry": "kRtx5090 = board_registry::board(0x2B85);",
    "list_of_sizes": "for (int t : {8, 256, 4096, 49152, 65536}) {",
    "ids": "std::vector<int> v{1, 2, 46, 47, 768};",
    "pdf_header": "header.insert(header.end(), {8, 2, 0, 0, 0});",
    "comment_line": "// was Board{12, 0, 170, 96, 1792}",
    "comment_block": "/* Board{12, 0, 170}\n still */ int x;",
    "comment_block_tuple": "/* {12, 0, 170, 96, 1792}\n */ int x;",
    "string": 'puts("Board{12, 0, 170, 96, 1792}");',
    "string_tuple": 'puts("{12, 0, 170, 96, 1792}");',
    "string_escaped_quote": 'puts("a\\" {12, 0, 170, 96, 1792}");',
}


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


class Scan(unittest.TestCase):
    def tree(self, root, files):
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def test_every_directory_and_suffix_is_scanned(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.tree(root, {
                "src/a.cpp": lit, "tests/b.cpp": lit, "tools/c.cu": lit, "apps/cli/d.cpp": lit,
                "bench/ops/e.cpp": lit, "include/f.h": lit, "examples/g.hpp": lit,
                "src/core/h.inc": lit, "third_party/x/i.cpp": lit, "src/j.txt": lit})
            bad = {r for r, _, _ in lint.scan(root, allow={})}
            self.assertEqual(bad, {"src/a.cpp", "tests/b.cpp", "tools/c.cu", "apps/cli/d.cpp",
                                   "bench/ops/e.cpp", "include/f.h", "examples/g.hpp",
                                   "src/core/h.inc"})

    def test_tracked_files_only_in_a_checkout(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            self.tree(root, {"src/tracked.cpp": lit, "src/untracked.cpp": lit})
            subprocess.run(["git", "-C", str(root), "add", "src/tracked.cpp"], check=True)
            self.assertEqual({r for r, _, _ in lint.scan(root, allow={})}, {"src/tracked.cpp"})

    def test_allow_list_is_pinned_by_count(self):
        two = "Board a{8, 6, 82, 6, 936};\nBoard b{8, 9, 128, 72, 1008};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.tree(root, {"tests/b.cpp": two, "src/other.cpp": "int x;\n"})
            self.assertEqual(lint.scan(root, allow={"tests/b.cpp": (2, "x")}), [])
            more = lint.scan(root, allow={"tests/b.cpp": (1, "x")})  # a new literal
            self.assertEqual([(r, n) for r, n, _ in more], [("tests/b.cpp", 0)])
            fewer = lint.scan(root, allow={"tests/b.cpp": (3, "x")})  # one removed
            self.assertEqual([(r, n) for r, n, _ in fewer], [("tests/b.cpp", 0)])
            gone = lint.scan(root, allow={"tests/b.cpp": (2, "x"), "src/missing.h": (1, "x")})  # entry of a missing file
            self.assertEqual([(r, n) for r, n, _ in gone], [("src/missing.h", 0)])

    def test_allow_list_entries_exist_with_a_reason(self):
        for rel, (count, reason) in lint.ALLOW.items():
            self.assertTrue((ROOT / rel).is_file(), rel)
            self.assertGreater(count, 0, rel)
            self.assertGreater(len(reason), 20, rel)

    def test_tree_is_clean(self):
        bad = lint.scan(ROOT)
        self.assertEqual(bad, [], "\n".join(f"{r}:{n}: {l}" for r, n, l in bad))


if __name__ == "__main__":
    unittest.main()
