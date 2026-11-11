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


RECOVER = os.path.join(os.path.dirname(QC), "qc_recover.sh")


def text(path=QC):
    with open(path, encoding="utf-8") as f:
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
    """Runs the extracted recovery block with a fake ninja/cmake/pgrep on PATH and a temp build dir."""

    def run_case(self, ninja_body, pgrep_pid=None, cmake_fail=False, expect_rc=0):
        with tempfile.TemporaryDirectory() as d:
            d = d.replace(chr(92), "/")
            bindir, root = d + "/bin", d + "/root"
            b = root + "/build_qc_89"
            for p in (bindir, b, root + "/wt_qc", root + "/logs", d + "/proc"):
                os.makedirs(p)
            cmake_body = ('[ -z "${QC_CMAKE_FAIL:-}" ] || { echo "CMake Error: bad" >&2; exit 1; }' + NL
                          + 'echo "cmake $*" >> "$QC_TRACE"' + NL + 'mkdir -p "$B"' + NL + 'touch "$B/build.ninja"' + NL)
            pgrep_body = (f"echo {pgrep_pid}" if pgrep_pid else "exit 1") + NL
            for name, body in (("ninja", ninja_body), ("cmake", cmake_body), ("pgrep", pgrep_body)):
                with open(f"{bindir}/{name}", "w", newline=NL) as f:
                    f.write("#!/bin/bash" + NL + body)
                os.chmod(f"{bindir}/{name}", 0o755)
            if pgrep_pid:   # a live ninja whose command line says -C <build dir>
                os.makedirs(f"{d}/proc/{pgrep_pid}")
                with open(f"{d}/proc/{pgrep_pid}/cmdline", "wb") as f:
                    f.write(b"ninja\0-C\0" + b.encode() + b"\0")
            for rel in STATE:
                os.makedirs(os.path.dirname(f"{b}/{rel}"), exist_ok=True)
                open(f"{b}/{rel}", "w").close()
            open(root + "/outside", "w").close()
            RECOVER_SH = RECOVER.replace(chr(92), "/")
            script = NL.join(["set -Eeuo pipefail",
                              f"ROOT={root}; B={b}; W={root}/wt_qc; L={root}/logs; ARCH=89; JOBS=1",
                              f'LOW=""; TARGETS="t"; LOG=$L/ninja.log; export B QC_TRACE={d}/trace',
                              f'. "{RECOVER_SH}"', "qc_build", 'echo "RC=$RC"', ""])
            with open(d + "/run.sh", "w", newline=NL) as f:   # a file, not bash -c: no argv quoting layer
                f.write(script)
            env = dict(os.environ, PATH=bindir + os.pathsep + os.environ["PATH"], QC_PROC=d + "/proc")
            if cmake_fail:
                env["QC_CMAKE_FAIL"] = "1"
            r = subprocess.run([BASH, d + "/run.sh"], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, expect_rc, r.stdout + r.stderr)
            trace = ""
            if os.path.exists(d + "/trace"):
                with open(d + "/trace") as f:
                    trace = f.read()
            left = {rel: os.path.exists(f"{b}/{rel}") for rel in STATE}
            self.assertEqual([n for n in os.listdir(root) if ".del." in n], [])   # no half-removed dir left behind
            return r.stdout + r.stderr, trace, left, os.path.exists(root + "/outside")

    def test_abort_detection(self):
        """qc_aborted: ninja's own assert line or rc >= 128 only; compiler text with the word does not count."""
        cases = [(1, "ninja: build.cc:435: void Plan::NodeFinished(): Assertion `edge && !edge->outputs_ready()' failed.", True),
                 (1, "ninja: ../src/build.cc:900: void Plan::RefreshDyndepDependents(DependencyScan*, Node*): "
                     "Assertion `edge' failed.", True),
                 (134, "anything", True), (139, "", True),
                 (1, "In member function 'void GrammarBuilder::UpdateLookaheadAssertion(int)':", False),
                 (1, "x.cc:3:1: error: static assertion failed: size", False),
                 (1, "FAILED: foo.o error: expected ';'", False),
                 (0, "ninja: x: Assertion `y' failed.", False)]
        with tempfile.TemporaryDirectory() as d:
            d = d.replace(chr(92), "/")
            for i, (rc, log, want) in enumerate(cases):
                with open(f"{d}/log{i}", "w", newline=NL) as f:
                    f.write(log + NL)
                with open(f"{d}/run.sh", "w", newline=NL) as f:
                    f.write(NL.join([". " + RECOVER.replace(chr(92), "/"), f"qc_aborted {rc} {d}/log{i}", ""]))
                r = subprocess.run([BASH, d + "/run.sh"], capture_output=True, text=True)
                self.assertEqual(r.returncode == 0, want, (rc, log))

    @staticmethod
    def ninja(*lines):
        return NL.join(['case "$*" in *build.ninja) exit 0;; esac'] + list(lines)) + NL

    ASSERT = 'echo "ninja: build.cc:435: void Plan::NodeFinished(): Assertion edge failed." >&2; exit 134'

    def test_step1_deletes_dyndep_files_only(self):
        # aborts while a dyndep file exists; .ninja_log/.ninja_deps and objects must survive
        out, trace, left, outside = self.run_case(self.ninja(
            f"if [ -e CMakeFiles/x.dir/CXX.dd ]; then {self.ASSERT}; fi", "echo ok"))
        self.assertIn("RC=0", out)
        self.assertNotIn("step 2", out)
        gone = {"CMakeFiles/x.dir/CXX.dd", "CMakeFiles/x.dir/a.o.ddi", "CMakeFiles/x.dir/CXXDependInfo.json"}
        self.assertEqual(left, {rel: rel not in gone for rel in STATE})
        self.assertEqual(len(trace.splitlines()), 1)     # cmake re-run once
        self.assertNotIn("-G", trace)                    # a regenerate, not a fresh configure
        self.assertTrue(outside)

    def test_step2_drops_ninja_log_and_deps_when_retry_aborts(self):
        out, trace, left, outside = self.run_case(self.ninja(
            f"if [ -e .ninja_deps ]; then {self.ASSERT}; fi", "echo ok"))
        self.assertIn("RC=0", out)
        self.assertIn("step 2", out)
        self.assertNotIn("step 3", out)
        gone = {".ninja_deps", ".ninja_log", "CMakeFiles/x.dir/CXX.dd", "CMakeFiles/x.dir/a.o.ddi",
                "CMakeFiles/x.dir/CXXDependInfo.json"}
        self.assertEqual(left, {rel: rel not in gone for rel in STATE})
        self.assertEqual(len(trace.splitlines()), 2)
        self.assertTrue(outside)

    def test_step3_fresh_build_dir_after_two_aborts(self):
        out, trace, left, outside = self.run_case(self.ninja(self.ASSERT))
        self.assertIn("RC=134", out)
        self.assertIn("removing", out)
        self.assertIn("ninja aborted after a fresh build dir", out)
        self.assertFalse(left["keepme"])            # whole build dir was wiped and configured fresh
        self.assertIn("-G Ninja", trace)            # fresh configure ran
        self.assertTrue(outside)                    # nothing outside the build dir touched

    def test_compiler_text_with_assertion_does_not_reset(self):
        for log in ("In member function 'void GrammarBuilder::UpdateLookaheadAssertion(int)':",
                    "x.cc:3:1: error: static assertion failed: size"):
            out, trace, left, outside = self.run_case(self.ninja(f'echo "FAILED: x.o" >&2; echo "{log}" >&2', "exit 1"))
            self.assertIn("RC=1", out)
            self.assertNotIn("recovery", out)
            self.assertTrue(all(left.values()))
            self.assertEqual(trace, "")

    def test_ordinary_failure_does_not_reset(self):
        out, trace, left, outside = self.run_case(self.ninja('echo "FAILED: x.o" >&2', "exit 1"))
        self.assertIn("RC=1", out)
        self.assertNotIn("recovery", out)
        self.assertTrue(all(left.values()))
        self.assertEqual(trace, "")

    def test_cmake_error_is_not_escalated(self):
        out, trace, left, outside = self.run_case(self.ninja(
            f"if [ -e CMakeFiles/x.dir/CXX.dd ]; then {self.ASSERT}; fi"), cmake_fail=True)
        self.assertIn("RC=1", out)
        self.assertIn("not escalating", out)
        self.assertNotIn("step 2", out)
        self.assertTrue(left[".ninja_log"] and left[".ninja_deps"] and left["keepme"])

    def test_live_ninja_in_build_dir_blocks_reset(self):
        out, trace, left, outside = self.run_case(self.ninja(self.ASSERT), pgrep_pid=4242, expect_rc=3)
        self.assertIn("another ninja is running", out)
        self.assertTrue(all(left.values()))        # nothing deleted
        self.assertEqual(trace, "")


if __name__ == "__main__":
    unittest.main()
