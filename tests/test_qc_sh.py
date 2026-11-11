"""tools/dev/qc.sh: the early-stop dyndep message pattern must be a real regex, not control bytes;
the ninja-abort recovery (dependency reset, then fresh build dir) must trigger and stay inside the build dir."""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

QC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "dev", "qc.sh")
NL = chr(10)
STATE = (".ninja_deps", ".ninja_log", "keepme", "CMakeFiles/x.dir/CXX.dd", "CMakeFiles/x.dir/a.o.ddi",
         "CMakeFiles/x.dir/CXXDependInfo.json", "CMakeFiles/x.dir/a.o")


def text():
    with open(QC, encoding="utf-8") as f:
        return f.read()


def recover_block():
    m = re.search(r"^# --- qc recover begin.*?^# --- qc recover end$", text(), re.M | re.S)
    assert m, "recover block markers missing"
    return m.group(0)


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

    def test_abort_pattern(self):
        m = re.search(r"^QC_ABORT_RE='([^']+)'$", text(), re.M)
        self.assertIsNotNone(m)
        pat = re.compile(m.group(1))
        for msg in ("ninja: build.cc:435: void Plan::NodeFinished(): Assertion `edge && !edge->outputs_ready()' failed.",
                    "ninja: Assertion failed in Plan::RefreshDyndepDependents (CMakeFiles/a.dir/CXX.dd)"):
            self.assertIsNotNone(pat.search(msg), msg)
        self.assertIsNone(pat.search("FAILED: foo.o error: expected ';'"))

    def test_reset_names_only_dependency_state(self):
        blk = recover_block()
        for name in (".ninja_deps", ".ninja_log", "*.dd", "*.ddi", "CXXDependInfo.json"):
            self.assertIn(name, blk)


def find_bash():
    """A bash that can run a script from a temp path (on Windows prefer Git's, not the WSL launcher)."""
    cands = [os.path.join(os.environ.get("ProgramFiles", ""), "Git", "bin", "bash.exe"), shutil.which("bash")]
    for c in cands:
        if not c or not os.path.exists(c):
            continue
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "probe.sh")
            with open(p, "w", newline=NL) as f:
                f.write("echo ok" + NL)
            try:
                r = subprocess.run([c, p.replace(chr(92), "/")], capture_output=True, text=True, timeout=30)
            except OSError:
                continue
            if r.stdout.strip() == "ok":
                return c
    return None


BASH = find_bash()


@unittest.skipUnless(BASH, "needs bash")
class QcRecoverTest(unittest.TestCase):
    """Runs the extracted recovery block with a fake ninja/cmake on PATH and a temp build dir."""

    def run_case(self, ninja_body):
        with tempfile.TemporaryDirectory() as d:
            d = d.replace(chr(92), "/")
            bindir, root = d + "/bin", d + "/root"
            b = root + "/build_qc_89"
            for p in (bindir, b, root + "/wt_qc", root + "/logs"):
                os.makedirs(p)
            cmake_body = 'echo "cmake $*" >> "$QC_TRACE"' + NL + 'mkdir -p "$B"' + NL + 'touch "$B/build.ninja"' + NL
            for name, body in (("ninja", ninja_body), ("cmake", cmake_body)):
                with open(f"{bindir}/{name}", "w", newline=NL) as f:
                    f.write("#!/bin/bash" + NL + body)
                os.chmod(f"{bindir}/{name}", 0o755)
            for rel in STATE:
                os.makedirs(os.path.dirname(f"{b}/{rel}"), exist_ok=True)
                open(f"{b}/{rel}", "w").close()
            open(root + "/outside", "w").close()
            script = NL.join(["set -Eeuo pipefail",
                              f"ROOT={root}; B={b}; W={root}/wt_qc; L={root}/logs; ARCH=89; JOBS=1",
                              f'LOW=""; TARGETS="t"; LOG=$L/ninja.log; export B QC_TRACE={d}/trace',
                              recover_block(), "qc_build", 'echo "RC=$RC"', ""])
            with open(d + "/run.sh", "w", newline=NL) as f:   # a file, not bash -c: no argv quoting layer
                f.write(script)
            r = subprocess.run([BASH, d + "/run.sh"], capture_output=True, text=True,
                               env=dict(os.environ, PATH=bindir + os.pathsep + os.environ["PATH"]))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            trace = ""
            if os.path.exists(d + "/trace"):
                with open(d + "/trace") as f:
                    trace = f.read()
            left = {rel: os.path.exists(f"{b}/{rel}") for rel in STATE}
            return r.stdout, trace, left, os.path.exists(root + "/outside")

    @staticmethod
    def ninja(*lines):
        return NL.join(['case "$*" in *build.ninja) exit 0;; esac'] + list(lines)) + NL

    def test_assertion_takes_dependency_reset_only(self):
        # aborts while stale dyndep state exists, succeeds once .ninja_deps is gone
        out, trace, left, outside = self.run_case(self.ninja(
            'if [ -e .ninja_deps ]; then echo "ninja: build.cc:435: Assertion edge failed." >&2; exit 134; fi',
            "echo ok"))
        self.assertIn("RC=0", out)
        self.assertIn("resetting its dependency state", out)
        self.assertNotIn("removing", out)
        gone = {".ninja_deps", ".ninja_log", "CMakeFiles/x.dir/CXX.dd", "CMakeFiles/x.dir/a.o.ddi",
                "CMakeFiles/x.dir/CXXDependInfo.json"}
        self.assertEqual(left, {rel: rel not in gone for rel in STATE})   # objects and the rest stay
        self.assertEqual(len(trace.splitlines()), 1)                      # cmake re-run once
        self.assertNotIn("-G", trace)                                     # a regenerate, not a fresh configure
        self.assertTrue(outside)

    def test_second_abort_removes_build_dir(self):
        out, trace, left, outside = self.run_case(self.ninja('echo "Assertion failed" >&2', "exit 134"))
        self.assertIn("RC=134", out)
        self.assertIn("removing", out)
        self.assertFalse(left["keepme"])            # whole build dir was wiped and configured fresh
        self.assertIn("-G Ninja", trace)            # fresh configure ran
        self.assertTrue(outside)                    # nothing outside the build dir touched

    def test_ordinary_failure_does_not_reset(self):
        out, trace, left, outside = self.run_case(self.ninja('echo "FAILED: x.o" >&2', "exit 1"))
        self.assertIn("RC=1", out)
        self.assertNotIn("resetting", out)
        self.assertTrue(all(left.values()))
        self.assertEqual(trace, "")


if __name__ == "__main__":
    unittest.main()
