"""tools/dev/qc.sh: the early-stop dyndep message pattern must be a real regex, not control bytes."""
import os
import re
import unittest

QC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "dev", "qc.sh")


def text():
    with open(QC, encoding="utf-8") as f:
        return f.read()


class QcShTest(unittest.TestCase):
    def test_no_control_bytes(self):
        bad = sorted({hex(ord(c)) for c in text() if ord(c) < 32 and c not in "\n\t\r"})
        self.assertEqual(bad, [])

    def test_dyndep_pattern_recognises_ninja_messages(self):
        m = re.search(r'not aborted\(r\) and not re\.search\(r"([^"]+)", txt\)', text())
        self.assertIsNotNone(m)
        pat = re.compile(m.group(1))
        for msg in ("ninja: error: loading 'CMakeFiles/foo.dir/CXX.dd': No such file or directory",
                    "ninja: error: 'src/new.cpp', needed by 'x.o', is not mentioned in its dyndep file "
                    "'CMakeFiles/foo.dir/CXX.dd'"):
            self.assertIsNotNone(pat.search(msg), msg)
        self.assertIsNone(pat.search("ninja: error: unknown target 'foo'"))


if __name__ == "__main__":
    unittest.main()
