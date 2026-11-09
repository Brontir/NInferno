"""tools/audit/diff_lint.py on planted diffs: each rule fires on the mistake and stays quiet on the
look-alikes. Plain git diff -U0 text and an in-memory tree; no git, no GPU."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import diff_lint as dl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def diff(path, removed=(), added=(), old_start=1, new_start=1):
    """A one-hunk -U0 diff of one file."""
    rows = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}",
            f"@@ -{old_start},{len(removed)} +{new_start},{len(added)} @@"]
    return "\n".join(rows + ["-" + r for r in removed] + ["+" + a for a in added]) + "\n"


def rules(found):
    return [f[2] for f in found]


BLOCK = ["    const auto plan = make_plan(model, card);", '    check(plan.valid, "plan is valid");',
         '    check(plan.bytes > 0, "plan has bytes");', '    check(plan.steps == 3, "three steps");']


LONG = BLOCK + ['    check(plan.steps[0].ok, "first step");', '    check(plan.steps[1].ok, "second step");',
                '    check(plan.steps[2].ok, "third step");', '    check(plan.total_ms < budget, "within budget");',
                '    log_line("plan checked for " + card.name());', '    release(plan);']


class DupBlock(unittest.TestCase):
    def test_pasted_twice_is_flagged_on_the_second_copy(self):
        body = ["void test_a() {"] + LONG + ["}", "", "void test_b() {"] + LONG + ["}"]
        text = "\n".join(body) + "\n"
        found = dl.lint(diff("t.cpp", added=body), dl.DictTree({"t.cpp": text}))
        self.assertEqual(rules(found), ["dup-block"])
        self.assertEqual(found[0][1], 15)  # first line of the second copy

    def test_copy_of_old_content_is_flagged(self):
        old = LONG + ["}", "int other();"]
        new = old + ["", "void again() {"] + LONG + ["}"]
        added = new[len(old):]
        d = diff("t.cpp", added=added, new_start=len(old) + 1)
        self.assertEqual(rules(dl.lint(d, dl.DictTree({"t.cpp": "\n".join(new) + "\n"}))), ["dup-block"])

    def test_short_copy_inside_a_bigger_new_run_is_quiet(self):
        a = ["void f() {"] + BLOCK + ["    return 1;", "}"]
        b = ["void g() {"] + BLOCK + ["    return 2;", "}"]
        text = "\n".join(a + [""] + b) + "\n"
        d = diff("t.cpp", added=a + [""] + b)
        self.assertEqual(dl.lint(d, dl.DictTree({"t.cpp": text})), [])

    def test_trivial_closers_are_quiet(self):
        closers = ["    }", "  }", "}", "};"] * 2
        d = diff("t.cpp", added=closers)
        self.assertEqual(dl.lint(d, dl.DictTree({"t.cpp": "\n".join(closers) + "\n"})), [])

    def test_allow_marker(self):
        marked = BLOCK[:2] + [BLOCK[2] + " // diff-lint: allow"] + BLOCK[3:]
        body = ["void test_a() {"] + BLOCK + ["}", "", "void test_b() {"] + marked + ["}"]
        text = "\n".join(body) + "\n"
        # the marker also makes the second copy differ from the first, so nothing repeats
        self.assertEqual(dl.lint(diff("t.cpp", added=body), dl.DictTree({"t.cpp": text})), [])

    def test_data_and_vendored_files_are_skipped(self):
        body = BLOCK + ["x"] + BLOCK
        for path in ("routes/a.json", "third_party/x/y.cpp"):
            self.assertEqual(dl.lint(diff(path, added=body), dl.DictTree({path: "\n".join(body) + "\n"})), [])


DEF = ["double DrafterOption::measured_at(int k) const {", "    return table_[k];", "}"]


class RemovedDefinition(unittest.TestCase):
    def test_removed_and_still_called_is_flagged_at_the_call(self):
        t = dl.DictTree({"a.h": "struct DrafterOption {};\n",
                         "b.cpp": "void f(DrafterOption o) {\n    use(o.measured_at(3));\n}\n"})
        found = dl.lint(diff("a.cpp", removed=DEF), t)
        self.assertEqual(rules(found), ["removed-def"])
        self.assertEqual(found[0][:2], ("b.cpp", 2))

    def test_inline_member_in_a_header(self):
        t = dl.DictTree({"b.cu": "void f() { x = widget.area(); }\n"})
        d = diff("w.h", removed=["    [[nodiscard]] int area() const {", "        return w_ * h_;", "    }"])
        self.assertEqual(rules(dl.lint(d, t)), ["removed-def"])

    def test_multi_line_signature(self):
        t = dl.DictTree({"b.cpp": "int g() { return pick_route(1, 2); }\n"})
        d = diff("a.cpp", removed=["static int pick_route(int a,", "                      int b) {",
                                   "    return a;", "}"])
        self.assertEqual(rules(dl.lint(d, t)), ["removed-def"])

    def test_removed_and_no_call_is_quiet(self):
        t = dl.DictTree({"b.cpp": "void f() { other(); }\n// measured_at(3) was removed\n"})
        self.assertEqual(dl.lint(diff("a.cpp", removed=DEF), t), [])

    def test_name_still_defined_elsewhere_is_quiet(self):
        for left in ("double Other::measured_at(int k) const {", "double measured_at(int k);",
                     "    double measured_at(double k) const;"):
            t = dl.DictTree({"b.cpp": "void f() { x.measured_at(3); }\n", "c.h": left + "\n"})
            self.assertEqual(dl.lint(diff("a.cpp", removed=DEF), t), [], left)

    def test_moved_definition_is_quiet(self):
        t = dl.DictTree({"new.cpp": "\n".join(DEF) + "\n", "b.cpp": "void f() { x.measured_at(3); }\n"})
        self.assertEqual(dl.lint(diff("old.cpp", removed=DEF), t), [])

    def test_calls_and_control_flow_are_not_definitions(self):
        removed = ["    if (check(x)) {", "        return run_all(x);", "    }", "    while (pending(x)) {",
                   "    }", "    : pool_(threads, 0) {}"]
        t = dl.DictTree({"b.cpp": "void f() { run_all(1); check(2); pending(3); pool_(4); }\n"})
        self.assertEqual(dl.lint(diff("a.cpp", removed=removed), t), [])

    def test_python_files_are_not_checked(self):
        t = dl.DictTree({"b.cpp": "void f() { x.measured_at(3); }\n"})
        self.assertEqual(dl.lint(diff("a.py", removed=DEF), t), [])

    def test_deleted_file(self):
        t = dl.DictTree({"b.cpp": "void f() { x.measured_at(3); }\n"})
        d = ("diff --git a/gone.cpp b/gone.cpp\ndeleted file mode 100644\n--- a/gone.cpp\n+++ /dev/null\n"
             "@@ -1,3 +0,0 @@\n" + "".join("-" + r + "\n" for r in DEF))
        self.assertEqual(rules(dl.lint(d, t)), ["removed-def"])

    def test_allow_marker_on_the_removed_line_or_the_call(self):
        t = dl.DictTree({"b.cpp": "void f() { x.measured_at(3); }\n"})
        marked = [DEF[0] + " // diff-lint: allow"] + DEF[1:]
        self.assertEqual(dl.lint(diff("a.cpp", removed=marked), t), [])
        t = dl.DictTree({"b.cpp": "void f() { x.measured_at(3); } // diff-lint: allow\n"})
        self.assertEqual(dl.lint(diff("a.cpp", removed=DEF), t), [])


class RemovedInclude(unittest.TestCase):
    HEADER = "#pragma once\nstruct Widget {\n    int size;\n};\n"

    def tree(self, user_text, **more):
        files = {"src/widget.h": self.HEADER, "src/use.cpp": user_text}
        files.update(more)
        return dl.DictTree(files)

    def removed(self, inc='#include "widget.h"'):
        return diff("src/use.cpp", removed=[inc])

    def test_type_still_used_and_no_other_path_is_flagged(self):
        t = self.tree("#include <vector>\nint f(const Widget& w) { return w.size; }\n")
        found = dl.lint(self.removed(), t)
        self.assertEqual(rules(found), ["removed-include"])
        self.assertEqual(found[0][:2], ("src/use.cpp", 2))

    def test_still_reachable_through_another_include_is_quiet(self):
        t = self.tree('#include "mid.h"\nint f(const Widget& w) { return w.size; }\n',
                      **{"src/mid.h": '#pragma once\n#include "widget.h"\n'})
        self.assertEqual(dl.lint(self.removed(), t), [])

    def test_symbol_not_used_is_quiet(self):
        t = self.tree('int f(int w) { return w; }\n// Widget is gone\nconst char* s = "Widget";\n')
        self.assertEqual(dl.lint(self.removed(), t), [])

    def test_type_defined_in_the_file_itself_is_quiet(self):
        t = self.tree("struct Widget { int size; };\nint f(const Widget& w) { return w.size; }\n")
        self.assertEqual(dl.lint(self.removed(), t), [])

    def test_system_and_unresolved_includes_are_skipped(self):
        t = self.tree("int f(const Widget& w) { return w.size; }\n")
        self.assertEqual(dl.lint(self.removed("#include <widget.h>"), t), [])
        self.assertEqual(dl.lint(self.removed('#include "nowhere.h"'), t), [])

    def test_allow_marker(self):
        t = self.tree("int f(const Widget& w) { return w.size; } // diff-lint: allow\n")
        self.assertEqual(dl.lint(self.removed(), t), [])
        t = self.tree("int f(const Widget& w) { return w.size; }\n")
        self.assertEqual(dl.lint(self.removed('#include "widget.h" // diff-lint: allow'), t), [])


def real_commit(sha):
    ok = subprocess.run(["git", "cat-file", "-e", sha + "^{commit}"], cwd=REPO, capture_output=True).returncode == 0
    return ok


def lint_commit(sha):
    d = subprocess.run(["git", "diff", "-U0", "-M", sha + "^", sha], cwd=REPO, capture_output=True,
                       text=True, encoding="utf-8", errors="replace").stdout
    return dl.lint(d, dl.GitTree(sha, cwd=REPO))


@unittest.skipUnless(real_commit("bd51efb1") and real_commit("7275afba") and real_commit("ec402086"),
                     "the real commits are not in this clone")
class RealBugs(unittest.TestCase):
    def test_599_deleted_measured_at_still_called(self):
        found = [f for f in lint_commit("bd51efb1") if f[2] == "removed-def"]
        self.assertEqual(len(found), 1)
        self.assertIn("`measured_at(`", found[0][3])
        self.assertIn("src/product/choice/choice.h", found[0][3])

    def test_600_header_tail_pasted_twice_warns(self):
        found = lint_commit("ec402086")
        self.assertTrue(found and all(f[2] == "dup-block" for f in found))
        self.assertTrue(all(f[0] == "src/product/choice/choice.h" for f in found))

    def test_602_block_pasted_twice_warns_only_below_the_long_copy_threshold(self):
        # The pasted test block is 5 lines, under the 10-line threshold: silent by default.
        self.assertEqual(lint_commit("7275afba"), [])
        old, dl.LONG_COPY = dl.LONG_COPY, 4
        try:
            found = lint_commit("7275afba")
        finally:
            dl.LONG_COPY = old
        self.assertEqual([(f[0], f[2]) for f in found],
                         [("tests/ops/linear/test_scratch_lifetime.cpp", "dup-block")])


class FalsePositives(unittest.TestCase):
    """Correct code that an earlier version flagged."""

    def lint(self, removed, **files):
        return dl.lint(diff("a.cpp", removed=removed), dl.DictTree(files))

    def test_member_init_is_not_a_call(self):
        w = ["    int area() const {", "        return w_ * h_;", "    }"]
        t = {"b.h": "struct B {\n    B(int w) : area(w) {}\n    int a() : x(1), area(2) {}\n};\n"}
        self.assertEqual(self.lint(w, **t), [])
        t = {"b.h": "struct B {\n    B(int w)\n        : area(w) {}\n};\n"}
        self.assertEqual(self.lint(w, **t), [])
        # an argument is still a call
        t = {"b.h": "void f() {\n    g(1, area(w));\n}\n"}
        self.assertEqual(rules(self.lint(w, **t)), ["removed-def"])

    def test_declaration_with_the_return_type_on_the_line_above(self):
        t = {"b.h": "struct B {\n    [[nodiscard]] double\n    measured_at(int k) const;\n};\n",
             "c.cpp": "void f(B b) { b.measured_at(1); }\n"}
        self.assertEqual(self.lint(DEF, **t), [])

    def test_a_call_on_a_continuation_line_is_still_a_call(self):
        t = {"c.cpp": "void f(B b) {\n    use(1,\n        measured_at(2));\n}\n"}
        self.assertEqual(rules(self.lint(DEF, **t)), ["removed-def"])

    def test_name_passed_to_a_macro_may_be_generated(self):
        t = {"m.h": "DEFINE_ACCESSOR(measured_at)\n", "c.cpp": "void f(B b) { b.measured_at(1); }\n"}
        self.assertEqual(self.lint(DEF, **t), [])
        t = {"m.h": "DEFINE_ACCESSOR(int, measured_at)\n", "c.cpp": "void f(B b) { b.measured_at(1); }\n"}
        self.assertEqual(self.lint(DEF, **t), [])

    def test_forward_declaration_replaces_the_include(self):
        header = {"src/widget.h": "#pragma once\nstruct Widget {\n    int size;\n};\n"}
        for fwd in ("struct Widget;", "class Widget;"):
            user = fwd + "\nint f(const Widget& w);\n"
            t = dl.DictTree({**header, "src/use.cpp": user})
            self.assertEqual(dl.lint(diff("src/use.cpp", removed=['#include "widget.h"']), t), [], fwd)

    def test_member_alias_is_not_a_header_type(self):
        header = {"src/vec.h": "#pragma once\nstruct Vec {\n    using value_type = int;\n    struct Inner {\n    };\n};\n"}
        user = "int f(value_type v, Inner i);\n"
        t = dl.DictTree({**header, "src/use.cpp": user})
        self.assertEqual(dl.lint(diff("src/use.cpp", removed=['#include "vec.h"']), t), [])

    def test_unresolvable_include_makes_removed_include_skip_the_file(self):
        header = {"src/widget.h": "#pragma once\nstruct Widget {\n    int size;\n};\n"}
        user = '#include "generated_elsewhere.h"\nint f(const Widget& w);\n'
        t = dl.DictTree({**header, "src/use.cpp": user})
        self.assertEqual(dl.lint(diff("src/use.cpp", removed=['#include "widget.h"']), t), [])


def git_repo_main(before, after, message="change", extra_message=None):
    """Commit two states of a tiny repo and run main() on the second; (exit code, stdout)."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

    def git(*args):
        subprocess.run(["git", *args], check=True, capture_output=True, env=env)

    with tempfile.TemporaryDirectory() as tmp:
        old = os.getcwd()
        os.chdir(tmp)
        try:
            git("init", "-q")
            for state, msg in ((before, "base"), (after, message)):
                for path, text in state.items():
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_text(text, encoding="utf-8", newline="\n")
                git("add", "-A")
                git("commit", "-q", "-m", msg)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = dl.main(["--base", "HEAD~1", "--head", "HEAD", "--exact"])
            return code, out.getvalue()
        finally:
            os.chdir(old)


class Main(unittest.TestCase):
    CALLER = "void f(DrafterOption o) { o.measured_at(3); }\n"
    LONGTEXT = "\n".join(["void a() {"] + LONG + ["}", "", "void b() {"] + LONG + ["}"]) + "\n"

    def test_clean_diff_exits_0_silently(self):
        code, out = git_repo_main({"a.cpp": "int x;\n"}, {"a.cpp": "int x;\nint y;\n"})
        self.assertEqual((code, out), (0, ""))

    def test_removed_def_fails(self):
        code, out = git_repo_main({"a.cpp": "\n".join(DEF) + "\n", "b.cpp": self.CALLER}, {"a.cpp": "", "b.cpp": self.CALLER})
        self.assertEqual(code, 1)
        self.assertIn("b.cpp:1: [removed-def]", out)

    def test_removed_def_allowed_by_the_commit_message(self):
        code, out = git_repo_main({"a.cpp": "\n".join(DEF) + "\n", "b.cpp": self.CALLER},
                                  {"a.cpp": "", "b.cpp": self.CALLER},
                                  message="drop it\n\ndiff-lint: allow removed-def measured_at")
        self.assertEqual((code, out), (0, ""))

    def test_removed_def_allowed_at_the_call_site(self):
        caller = "void f(DrafterOption o) { o.measured_at(3); } // diff-lint: allow\n"
        code, out = git_repo_main({"a.cpp": "\n".join(DEF) + "\n", "b.cpp": caller}, {"a.cpp": "", "b.cpp": caller})
        self.assertEqual((code, out), (0, ""))

    def test_dup_block_warns_with_an_annotation_and_exits_0(self):
        code, out = git_repo_main({"t.cpp": "int x;\n"}, {"t.cpp": self.LONGTEXT})
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("::warning file=t.cpp,line=15,title=diff-lint dup-block::"), out)

    def test_removed_include_warns_and_exits_0(self):
        before = {"src/widget.h": "#pragma once\nstruct Widget {\n    int size;\n};\n",
                  "src/use.cpp": '#include "widget.h"\nint f(const Widget& w);\n'}
        after = {"src/widget.h": before["src/widget.h"], "src/use.cpp": "int f(const Widget& w);\n"}
        code, out = git_repo_main(before, after)
        self.assertEqual(code, 0)
        self.assertIn("::warning file=src/use.cpp,line=1,title=diff-lint removed-include::", out)


if __name__ == "__main__":
    unittest.main()
