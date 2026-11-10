#!/usr/bin/env python3
"""Unit tests of tools/audit/board_literal_lint.py, and the tree itself is clean."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "board_literal_lint", ROOT / "tools" / "audit" / "board_literal_lint.py")
lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint)


class Violations(unittest.TestCase):
    def hit(self, code):
        return len(lint.violations(code)) == 1

    def test_flags_literals(self):
        self.assertTrue(self.hit("auto b = Board{12, 0, 170, 96, 1792};"))
        self.assertTrue(self.hit("const ninferno::Board b{8, 9, 128, 72, 1008};"))
        self.assertTrue(self.hit("const Board b = {8, 9, 128, 72, 1008};"))
        self.assertTrue(self.hit("x(Board {7, 5, 68, 5.5, 616})"))

    def test_ignores_non_literals(self):
        for code in ("struct Board {", "struct Board{ int major = 0; };",
                     "table->board = Board{cc / 10, cc % 10, sms};",
                     "return Board{row->major, row->minor};",
                     "FactsBoard{1, 2}", "kRtx5090 = board_registry::board(0x2B85);"):
            self.assertEqual(lint.violations(code), [], code)

    def test_ignores_comments(self):
        self.assertEqual(lint.violations("// was Board{12, 0, 170, 96, 1792}"), [])
        self.assertEqual(lint.violations("/* Board{12, 0, 170}\n still */ int x;"), [])
        self.assertEqual(lint.violations("int x; /* a */ Board{1, 2}"), [(1, "int x; /* a */ Board{1, 2}")])


class Scan(unittest.TestCase):
    def test_allow_list_and_dirs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "tools").mkdir()
            (root / "src" / "a.cpp").write_text("Board b{8, 6, 82, 6, 936};\n")
            (root / "tests" / "b.cpp").write_text("Board b{8, 6, 82, 6, 936};\n")
            (root / "src" / "c.txt").write_text("Board b{8, 6, 82, 6, 936};\n")
            bad = lint.scan(root, allow={"tests/b.cpp": "x"})
            self.assertEqual([(r, n) for r, n, _ in bad], [("src/a.cpp", 1)])

    def test_tree_is_clean(self):
        bad = lint.scan(ROOT)
        self.assertEqual(bad, [], "\n".join(f"{r}:{n}: {l}" for r, n, l in bad))


if __name__ == "__main__":
    unittest.main()
