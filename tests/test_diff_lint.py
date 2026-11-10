"""tools/audit/diff_lint.py on planted diffs: each rule fires on the mistake and stays quiet on the
look-alikes. Mostly plain git diff -U0 text and an in-memory tree; the Main tests and the constructor
skip run through a scratch git repository. No GPU."""
import contextlib
import io
import json
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


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "diff_lint"


def lint_fixture(name):
    """Lint a recorded real change: its diff against a tree built from the recorded head files
    (`text`: the whole file; `sparse`: only the listed lines, at their line numbers)."""
    data = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    files = {}
    for path, f in data["files"].items():
        if "text" in f:
            files[path] = f["text"]
        else:
            rows = [""] * max(int(n) for n in f["sparse"])
            for n, text in f["sparse"].items():
                rows[int(n) - 1] = text
            files[path] = "\n".join(rows) + "\n"
    return dl.lint(data["diff"], dl.DictTree(files))


class RealBugs(unittest.TestCase):
    """Three real mistakes that reached review (#599, #600, #602), recorded as fixtures so that they
    are checked wherever the tests run, with or without the history."""

    def test_599_deleted_measured_at_still_called(self):
        found = [f for f in lint_fixture("pr599_removed_def") if f[2] == "removed-def"]
        self.assertEqual(len(found), 1)
        self.assertIn("`measured_at(`", found[0][3])
        self.assertIn("src/product/choice/choice.h", found[0][3])
        self.assertIn("called at 5 site(s)", found[0][3])

    def test_600_header_tail_pasted_twice_warns(self):
        found = lint_fixture("pr600_dup_block")
        self.assertTrue(found and all(f[2] == "dup-block" for f in found))
        self.assertTrue(all(f[0] == "src/product/choice/choice.h" for f in found))

    def test_602_block_pasted_twice_warns_only_below_the_long_copy_threshold(self):
        # The pasted test block is 5 lines, under the 10-line threshold: silent by default.
        self.assertEqual(lint_fixture("pr602_dup_block"), [])
        old, dl.LONG_COPY = dl.LONG_COPY, 4
        try:
            found = lint_fixture("pr602_dup_block")
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

    LIVE_FOO = {"c.cpp": "int pick();\nint g() { return foo(1); }\n"}

    def test_removed_block_comment_with_a_signature_is_not_a_definition(self):
        removed = ["/*", " * old version:", " * int foo(int x) {", " *     return x;", " * }", " */"]
        self.assertEqual(self.lint(removed, **self.LIVE_FOO), [])
        self.assertEqual(self.lint(["/* int foo(int x) { return x; } */"], **self.LIVE_FOO), [])

    def test_removed_line_comment_with_a_signature_is_not_a_definition(self):
        removed = ["// int foo(int x) {", "//     return x;", "// }"]
        self.assertEqual(self.lint(removed, **self.LIVE_FOO), [])

    def test_removed_string_with_a_signature_is_not_a_definition(self):
        removed = ['const char* sample = "int foo(int x) {";', 'const char* more = R"(', "int foo(int x) {", ')";']
        self.assertEqual(self.lint(removed, **self.LIVE_FOO), [])

    def test_a_real_removed_definition_of_the_same_shape_is_still_flagged(self):
        found = self.lint(["int foo(int x) {", "    return x;", "}"], **self.LIVE_FOO)
        self.assertEqual(rules(found), ["removed-def"])
        self.assertEqual(found[0][:2], ("c.cpp", 2))

    def test_a_definition_removed_with_a_comment_above_it_is_still_flagged(self):
        removed = ["/* int foo(int x) { fake } */", "// foo(int x) { also fake }", "int foo(int x) {", "    return x;", "}"]
        self.assertEqual(rules(self.lint(removed, **self.LIVE_FOO)), ["removed-def"])

    def test_function_like_text_in_a_string_is_not_a_call(self):
        files = {"b.cpp": 'const char* text = "foo(1)";\n'}
        self.assertEqual(self.lint(["int foo(int x) {", "    return x;", "}"], **files), [])
        files = {"b.cpp": 'const char* text = R"(\nfoo(1)\n)";\n'}
        self.assertEqual(self.lint(["int foo(int x) {", "    return x;", "}"], **files), [])
        files = {"b.cpp": 'const char* text = "foo(1)";\nint h() { return foo(2); }\n'}
        found = self.lint(["int foo(int x) {", "    return x;", "}"], **files)
        self.assertEqual([f[:2] for f in found], [("b.cpp", 2)])

    def test_a_commented_out_declaration_does_not_hide_a_live_call(self):
        files = {"b.h": "// int foo(int x);\n", "c.cpp": "int h() { return foo(2); }\n"}
        self.assertEqual(rules(self.lint(["int foo(int x) {", "    return x;", "}"], **files)), ["removed-def"])

    def test_commented_out_include_is_not_a_removed_include(self):
        user = '#include "widget.h"\nint f(const Widget& w);\n'
        files = {"src/widget.h": "#pragma once\nstruct Widget {\n    int size;\n};\n", "src/use.cpp": user.split("\n", 1)[1]}
        t = dl.DictTree(files)
        d = diff("src/use.cpp", removed=["/*", '#include "widget.h"', "*/"])
        self.assertEqual(dl.lint(d, t), [])

    def test_mask_keeps_length_and_newlines(self):
        src = 'int a; // x \\\n"cont\n/* b\n"c" */ const char* s = "q\\"r"; char c = \'"\'; int n = 1\'000;\n'
        out = dl.mask_cxx(src)
        self.assertEqual(len(out), len(src))
        self.assertEqual([i for i, c in enumerate(out) if c == "\n"], [i for i, c in enumerate(src) if c == "\n"])
        self.assertNotIn("cont", out)
        self.assertIn("int a;", out)
        self.assertIn("1'000", out)
        self.assertIn('#include "x.h"', dl.mask_cxx('#include "x.h" // y\n'))

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

    def test_removed_def_warns_and_exits_0(self):
        # On the history of master most removed-def findings were false: it must never fail a check.
        code, out = git_repo_main({"a.cpp": "\n".join(DEF) + "\n", "b.cpp": self.CALLER}, {"a.cpp": "", "b.cpp": self.CALLER})
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("::warning file=b.cpp,line=1,title=diff-lint removed-def::"), out)
        self.assertNotIn("error(s)", out)

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


def lits(path, text, added=None):
    """bad-literal findings for a file; `added` = 1-based added line numbers (default: all lines)."""
    n = len(text.split("\n")) if added is None else None
    rows = text.split("\n")
    added = added or list(range(1, n + 1))
    d = diff(path, added=[rows[i - 1] for i in added], new_start=added[0])
    return [f[1] for f in dl.lint(d, dl.DictTree({path: text})) if f[2] == "bad-literal"]


class BadLiteral(unittest.TestCase):
    def test_newline_inside_a_cpp_string_is_flagged_at_the_first_line(self):
        text = 'void f() {\n    std::cout << "SKIP: no usable CUDA device\n                 ";\n}\n'
        self.assertEqual(lits("t.cpp", text, [2, 3]), [2])

    def test_only_added_lines_are_reported(self):
        text = 'void f() {\n    std::cout << "SKIP\n                 ";\n}\n'
        self.assertEqual(lits("t.cpp", text, [4]), [])

    def test_all_cpp_extensions(self):
        for name in ("a.c", "a.cc", "a.cu", "a.h", "a.cuh", "a.hpp", "a.inl"):
            self.assertEqual(lits(name, 'const char* s = "abc\n";\n', [1]), [1], name)

    def test_char_literal_with_a_quote_is_valid(self):
        text = "char q = '\"';\nconst char* s = \"a\" '\"' \"b\";\nchar a = '\\'';\n"
        self.assertEqual(lits("t.cpp", text), [])

    def test_escaped_quote_is_valid(self):
        text = 'const char* s = "say \\"hi\\"";\nconst char* t = "backslash \\\\";\n'
        self.assertEqual(lits("t.cpp", text), [])

    def test_escaped_quote_at_the_end_is_unterminated(self):
        self.assertEqual(lits("t.cpp", 'const char* s = "say \\"\n'), [1])

    def test_line_continuation_inside_a_string_is_valid(self):
        self.assertEqual(lits("t.cpp", '#define MSG "first \\\n second"\n'), [])

    def test_raw_string_spanning_lines_is_valid(self):
        text = ('const char* k = R"cuda(\n__global__ void f() { printf("x\\n"); }\nit\'s "quoted\n)cuda";\n'
                'const char* m = R"(a\nb)" "c";\nint after = 1;\n')
        self.assertEqual(lits("t.cu", text), [])

    def test_text_after_a_raw_string_is_still_checked(self):
        self.assertEqual(lits("t.cpp", 'auto a = R"x(\nbody)x"; auto b = "oops\n;\n'), [2])

    def test_line_comment_with_a_quote_is_valid(self):
        text = '// it\'s "quoted\nint x = 1;  // don\'t "\n/* a " b\n still " comment */ int y;\n'
        self.assertEqual(lits("t.cpp", text), [])

    def test_quote_inside_a_string_after_a_comment_marker(self):
        self.assertEqual(lits("t.cpp", 'const char* u = "http://x"; int z;\n'), [])

    def test_digit_separator_is_not_a_char_literal(self):
        self.assertEqual(lits("t.cpp", "int n = 1'000'000;\nunsigned m = 0xFF'FF;\n"), [])

    def test_broken_literal_is_isolated_to_its_line(self):
        text = 'a = "x\nb = 1;\nc = "y\n'
        self.assertEqual(lits("t.cpp", text), [1, 3])

    def test_allow_marker(self):
        self.assertEqual(lits("t.cpp", 'a = "x // diff-lint: allow\n"\n', [1]), [])

    def test_python_single_quoted_string_with_a_newline_is_flagged(self):
        text = "print('no usable CUDA device\n      ')\n"
        self.assertEqual(lits("t.py", text), [1])

    def test_python_double_quoted_is_flagged_and_valid_forms_are_not(self):
        self.assertEqual(lits("t.py", 'x = "abc\n"\n'), [1])
        ok = ("a = 'it\\'s'\nb = \"say \\\"hi\\\"\"  # it's \"quoted\nc = 'one \\\n two'\nd = '\"' + \"'\"\n")
        self.assertEqual(lits("t.py", ok), [])

    def test_python_triple_quoted_string_may_span_lines(self):
        text = ('"""Doc string\nwith it\'s "quotes" and one more "\n"""\nx = \'\'\'a\nb "c\n\'\'\'\n'
                'y = f"""\n{1}\n"""\nz = 1\n')
        self.assertEqual(lits("t.py", text), [])

    def test_python_triple_quote_then_broken_line(self):
        self.assertEqual(lits("t.py", 'x = """a\nb"""\ny = \'oops\n\'\n'), [3])

    def test_line_comment_ending_in_a_backslash_continues_onto_the_next_line(self):
        self.assertEqual(lits("t.cpp", '// docs \\\n"unterminated-looking text\nint x;\n'), [])
        self.assertEqual(lits("t.cpp", '// a \\\n// b \\\n"c\nint x;\n'), [])
        self.assertEqual(lits("t.cpp", 'int x; // a \\\n"c\nint y;\n'), [])

    def test_the_comment_ends_where_the_backslash_chain_ends(self):
        self.assertEqual(lits("t.cpp", '// a \\\n// b\nconst char* s = "oops\n;\n'), [3])
        self.assertEqual(lits("t.cpp", '// a \\ b\nconst char* s = "oops\n;\n'), [2])

    def test_other_files_are_not_checked(self):
        self.assertEqual(lits("t.sh", 'echo "a\nb"\n'), [])

    def test_cli_exits_1_and_names_file_and_line(self):
        code, out = git_repo_main({"t.cpp": "int x;\n"}, {"t.cpp": 'int x;\nconst char* s = "a\n";\n'})
        self.assertEqual(code, 1)
        self.assertIn("t.cpp:2: [bad-literal]", out)


def lint_old(old_text, removed_range, new_text, other_files, path="a.cpp", rename_to=None):
    """Lint a hunk that removes lines `removed_range` (1-based, inclusive) of `old_text`, with the old
    file given as the base tree."""
    old_rows = old_text.split("\n")
    first, last = removed_range
    new_path = rename_to or path
    d = (f"diff --git a/{path} b/{new_path}\n--- a/{path}\n+++ b/{new_path}\n"
         f"@@ -{first},{last - first + 1} +{first - 1},0 @@\n"
         + "".join("-" + r + "\n" for r in old_rows[first - 1:last]))
    head = dl.DictTree({new_path: new_text, **other_files})
    return dl.lint(d, head, base=dl.DictTree({path: old_text}))


class WholeOldFileMask(unittest.TestCase):
    """A -U0 hunk shows only the removed lines; comments and literals are masked in the old file."""
    CALLER = {"b.cpp": "int g() { return lib.foo_helper(1) + foo_helper(2); }\n"}

    def test_tail_of_a_block_comment_is_not_a_definition(self):
        old = "/*\nold version:\nint foo_helper(int x) {\n    return x;\n}\n*/\nint keep;\n"
        self.assertEqual(lint_old(old, (3, 5), "/*\nold version:\n*/\nint keep;\n", self.CALLER), [])

    def test_comment_lines_without_a_leading_star(self):
        old = "int a;\n/* the former body\nint foo_helper(int x) {\n    return x + 1;\n}\nend of note */\n"
        self.assertEqual(lint_old(old, (3, 5), "int a;\n/* the former body\nend of note */\n", self.CALLER), [])

    def test_middle_of_a_raw_string_is_not_a_definition(self):
        old = 'const char* k = R"cuda(\nint foo_helper(int x) {\n    return x;\n}\n)cuda";\n'
        self.assertEqual(lint_old(old, (2, 4), 'const char* k = R"cuda(\n)cuda";\n', self.CALLER), [])

    def test_commented_out_include_in_a_block_is_not_a_removed_include(self):
        old = '/* disabled:\n#include "widget.h"\n*/\nint f(const Widget& w);\n'
        files = {"widget.h": "#pragma once\nstruct Widget {\n    int size;\n};\n"}
        self.assertEqual(lint_old(old, (2, 2), "/* disabled:\n*/\nint f(const Widget& w);\n", files), [])

    def test_a_definition_after_the_end_of_a_comment_on_the_same_line_is_found(self):
        old = '/* see "x\n*/ int foo_helper(int x) {\n    return x;\n}\n'
        found = lint_old(old, (2, 4), "/* see \"x\n*/\n", self.CALLER)
        self.assertEqual(rules(found), ["removed-def"])

    def test_a_real_definition_after_a_comment_tail_is_found(self):
        old = "/*\nnote\n*/\nint foo_helper(int x) {\n    return x;\n}\n"
        found = lint_old(old, (3, 6), "/*\nnote\n", self.CALLER)
        self.assertEqual(rules(found), ["removed-def"])

    def test_a_renamed_file_is_masked_by_its_old_content(self):
        old = "/*\nold version:\nint foo_helper(int x) {\n    return x;\n}\n*/\n"
        self.assertEqual(lint_old(old, (3, 5), "/*\nold version:\n*/\n", self.CALLER, rename_to="c.cpp"), [])
        old = "int foo_helper(int x) {\n    return x;\n}\n"
        found = lint_old(old, (1, 3), "", self.CALLER, rename_to="c.cpp")
        self.assertEqual(rules(found), ["removed-def"])

    def test_a_base_that_does_not_match_the_hunk_is_not_trusted(self):
        # the old text at those line numbers is another text: the hunk is masked on its own
        old = "// unrelated\n// unrelated\n// unrelated\n"
        d = diff("a.cpp", removed=["int foo_helper(int x) {", "    return x;", "}"])
        found = dl.lint(d, dl.DictTree({"a.cpp": "", **self.CALLER}), base=dl.DictTree({"a.cpp": old}))
        self.assertEqual(rules(found), ["removed-def"])

    def test_parse_diff_keeps_the_old_path(self):
        d = "diff --git a/o.cpp b/n.cpp\n--- a/o.cpp\n+++ b/n.cpp\n@@ -1 +1,0 @@\n-int x;\n"
        self.assertEqual(dl.parse_diff(d)["n.cpp"]["old"], "o.cpp")


class ParseDiffCounts(unittest.TestCase):
    def test_rows_that_look_like_file_headers_are_content(self):
        d = ("diff --git a/a.cpp b/a.cpp\n--- a/a.cpp\n+++ b/a.cpp\n@@ -4,2 +4,2 @@\n--- gone\n-- also gone\n"
             "+++ new\n+++ b/fake.cpp\n")
        files = dl.parse_diff(d)
        self.assertEqual(list(files), ["a.cpp"])
        self.assertEqual(files["a.cpp"]["removed"], [[(4, "-- gone"), (5, "- also gone")]])
        self.assertEqual(files["a.cpp"]["added"], [(4, "++ new"), (5, "++ b/fake.cpp")])

    def test_deleted_file_rows_that_look_like_headers(self):
        d = "diff --git a/a.cpp b/a.cpp\n--- a/a.cpp\n+++ /dev/null\n@@ -1,2 +0,0 @@\n--- x\n-int y;\n"
        self.assertEqual(dl.parse_deleted(d), {"a.cpp": [(1, "-- x"), (2, "int y;")]})


class RemovedDefFalseAlarms(unittest.TestCase):
    FREE = ["int digest_of(int x) {", "    return x;", "}"]
    MEMBER = ["    int digest_of(int x) const {", "        return x;", "    }"]

    def lint(self, removed, **files):
        return dl.lint(diff("a.cpp", removed=removed), dl.DictTree(files))

    def test_a_member_call_is_not_a_call_of_a_removed_free_function(self):
        files = {"b.cpp": "int g(S s) { return s.digest_of(1) + p->digest_of(2) + std::digest_of(3); }\n"}
        self.assertEqual(self.lint(self.FREE, **files), [])

    def test_a_member_call_still_counts_for_a_removed_member(self):
        files = {"b.cpp": "int g(S s) { return s.digest_of(1); }\n"}
        self.assertEqual(rules(self.lint(self.MEMBER, **files)), ["removed-def"])
        qualified = ["int S::digest_of(int x) const {", "    return x;", "}"]
        self.assertEqual(rules(self.lint(qualified, **files)), ["removed-def"])

    def test_a_namespace_qualified_call_is_still_a_call(self):
        files = {"b.cpp": "int g() { return detail::digest_of(1); }\n"}
        self.assertEqual(rules(self.lint(self.FREE, **files)), ["removed-def"])

    def test_std_and_global_qualified_calls_are_other_functions(self):
        files = {"b.cpp": "int g() { return std::digest_of(1) + ::digest_of(2); }\n"}
        self.assertEqual(self.lint(self.FREE, **files), [])

    def test_a_lambda_or_variable_of_the_same_name_is_a_declaration(self):
        for line in ("    const auto digest_of = [&](int x) { return x; };", "    std::function<int(int)> digest_of;",
                     "    int digest_of{3};", "int g(int digest_of, int y) { return y; }"):
            files = {"b.cpp": f"{line}\nint h() {{ return digest_of(2); }}\n"}
            self.assertEqual(self.lint(self.FREE, **files), [], line)

    def test_a_plain_use_of_the_name_is_not_a_declaration(self):
        files = {"b.cpp": "int h(int y) {\n    return digest_of(y);\n}\nint k = 3 + digest_of;\n"}
        self.assertEqual(rules(self.lint(self.FREE, **files)), ["removed-def"])

    def test_the_constructor_of_a_class_of_that_name_is_not_a_removed_function(self):
        ctor = ["    explicit RequestError(int m) {", "    }"]
        caller = "void f() { throw RequestError(\"x\"); }\n"
        for head in ("struct RequestError : std::exception {", "class RequestError final : public Base {"):
            self.assertEqual(self.lint(ctor, **{"e.h": head + "\n};\n", "b.cpp": caller}), [], head)
        # a name that is not a class stays a finding
        self.assertEqual(rules(self.lint(ctor, **{"b.cpp": caller})), ["removed-def"])

    def test_the_constructor_skip_works_with_git_grep(self):
        # GitTree.grep is `git grep -E`: the class lookup must be written in that dialect
        ctor = "    explicit RequestError(int m) {\n    }\n"
        caller = 'void f() { throw RequestError("x"); }\n'
        code, out = git_repo_main({"e.h": "struct RequestError : std::exception {\n" + ctor + "};\n", "b.cpp": caller},
                                  {"e.h": "struct RequestError : std::exception {\n};\n", "b.cpp": caller})
        self.assertEqual((code, out), (0, ""))

    def test_the_name_filter(self):
        caller = {"b.cpp": "int g() { return run(1) + RUN(2) + main(3); }\n"}
        for sig in ("int RUN(int x) {", "int main(int x) {"):
            self.assertEqual(self.lint([sig, "}"], **caller), [], sig)
        files = {"b.cpp": "int g() { return ab(1); }\n"}
        self.assertEqual(self.lint(["int ab(int x) {", "}"], **files), [])  # shorter than 3 characters

    def test_vendored_and_generated_trees_are_not_read(self):
        files = {"third_party/x/b.cpp": "int g() { return digest_of(1); }\n", "build/c.cpp": "int h() { return digest_of(2); }\n"}
        self.assertEqual(self.lint(self.FREE, **files), [])

    def test_a_call_that_starts_a_multi_line_statement_is_a_call(self):
        # not a member initializer: the line above ends a statement
        files = {"b.cpp": "void f() {\n    int a = 1;\n    digest_of(\n        a);\n}\n"}
        found = self.lint(self.FREE, **files)
        self.assertEqual([(f[2], f[1]) for f in found], [("removed-def", 3)])
        files = {"b.cpp": "void f() {\n    if (x) {\n    }\n    digest_of(\n        1);\n}\n"}
        self.assertEqual(rules(self.lint(self.FREE, **files)), ["removed-def"])

    def test_a_member_initializer_on_its_own_line_is_still_an_initializer(self):
        files = {"b.h": "struct B {\n    B(int w) :\n        other_(1),\n        digest_of(w) {}\n};\n"}
        self.assertEqual(self.lint(self.MEMBER, **files), [])
        files = {"b.h": "struct B {\n    B(int w)\n        : other_(1),\n          digest_of(w) {}\n};\n"}
        self.assertEqual(self.lint(self.MEMBER, **files), [])


class RemovedIncludeFixes(unittest.TestCase):
    HEADER = {"src/widget.h": "#pragma once\nstruct Widget {\n    int size;\n};\n"}

    def lint(self, user, removed=('#include "widget.h"',), added=(), **files):
        t = dl.DictTree({**self.HEADER, "src/use.cpp": user, **files})
        return dl.lint(diff("src/use.cpp", removed=list(removed), added=list(added)), t)

    def test_an_unresolved_generated_table_does_not_skip_the_file(self):
        user = '#include "core/rows_generated.inc"\nint f(const Widget& w);\n'
        self.assertEqual(rules(self.lint(user)), ["removed-include"])

    def test_an_unresolved_header_still_skips_the_file(self):
        user = '#include "core/rows_generated.h"\nint f(const Widget& w);\n'
        self.assertEqual(self.lint(user), [])

    def test_a_function_of_the_file_with_the_name_of_a_header_type_is_not_needed(self):
        header = {"src/err.h": "#pragma once\nstruct parse_error {\n};\n"}
        user = "int parse_error(int x) {\n    return x;\n}\nint g() { return parse_error(1); }\n"
        t = dl.DictTree({**header, "src/use.cpp": user})
        self.assertEqual(dl.lint(diff("src/use.cpp", removed=['#include "err.h"']), t), [])

    def test_a_quoted_include_swapped_for_an_angle_include_is_not_removed(self):
        user = "#include <widget.h>\nint f(const Widget& w);\n"
        self.assertEqual(self.lint(user, added=["#include <widget.h>"]), [])
        self.assertEqual(rules(self.lint("int f(const Widget& w);\n", added=["#include <other.h>"])), ["removed-include"])

    def test_a_relative_quoted_include_swapped_for_a_search_path_angle_include(self):
        user = "#include <nlohmann/json.hpp>\nint f(const Json& w);\n"
        files = {"third_party/nlohmann/json.hpp": "struct Json {\n};\n"}
        d = diff("src/use.cpp", removed=['#include "../third_party/nlohmann/json.hpp"'], added=["#include <nlohmann/json.hpp>"])
        self.assertEqual(dl.lint(d, dl.DictTree({"src/use.cpp": user, **files})), [])


class MaskAndLiteralFixes(unittest.TestCase):
    def test_a_u8_char_literal_is_not_a_digit_separator(self):
        out = dl.mask_cxx("auto c = u8'a'; int after = foo(1);\n")
        self.assertEqual(out, "auto c = u8' '; int after = foo(1);\n")

    def test_the_include_path_after_a_block_comment_is_kept(self):
        self.assertIn('#include "x.h"', dl.mask_cxx('/* c */ #include "x.h"\nint x;\n'))

    def test_error_and_warning_text_may_hold_an_apostrophe(self):
        self.assertEqual(lits("t.cpp", "#error don't build this\nint x;\n"), [])
        self.assertEqual(lits("t.cpp", "  #  warning this isn't supported\nint x;\n"), [])
        self.assertEqual(dl.mask_cxx("#error don't\nint h = foo(1); char c = 'x';\n"),
                         "#error don't\nint h = foo(1); char c = ' ';\n")
        # a real broken literal on the next line is still found
        self.assertEqual(lits("t.cpp", "#error don't\nconst char* s = \"oops\n;\n"), [2])

    def test_if_0_groups_may_hold_prose(self):
        self.assertEqual(lits("t.cpp", "#if 0\ndon't \"do this\n#endif\nint i;\n"), [])
        self.assertEqual(lits("t.cpp", "#if 0 // old\nit's\n#if X\nnot ok'\n#endif\nstill off'\n#endif\nint i;\n"), [])
        # #else ends the dead group; code after it is checked again
        self.assertEqual(lits("t.cpp", "#if 0\nit's\n#else\nconst char* s = \"oops\n;\n#endif\n"), [4])
        # a group that is not `#if 0` is code
        self.assertEqual(lits("t.cpp", "#if 1\nconst char* s = \"oops\n;\n#endif\n"), [2])
        self.assertEqual(lits("t.cpp", "#if 0 || X\nconst char* s = \"oops\n;\n#endif\n"), [2])

    def test_inc_files_are_checked(self):
        self.assertEqual(lits("rows.inc", 'const char* s = "oops\n;\n'), [1])

    def test_elif_0_keeps_the_group_dead(self):
        self.assertEqual(lits("t.cpp", "#if 0\nit's\n#elif 0\nstill off'\n#else\nconst char* s = \"oops\n;\n#endif\n"), [6])
        self.assertEqual(lits("t.cpp", "#if 0\nit's\n#elif X\nconst char* s = \"oops\n;\n#endif\n"), [4])

    def test_alternatives_after_a_taken_literal_branch_are_dead(self):
        # `#elif 1` (or `#if 1`) takes its branch: the `#else` after it is not compiled.
        self.assertEqual(lits("t.cpp", "#if 0\nit's\n#elif 1\nint a;\n#else\ndon't\n#endif\nint b;\n"), [])
        self.assertEqual(lits("t.cpp", "#if 1\nint a;\n#else\ndon't\n#endif\nint b;\n"), [])
        self.assertEqual(lits("t.cpp", "#if 1\nint a;\n#elif X\ndon't\n#endif\nint b;\n"), [])
        # the taken branch itself, and code after the group, are still checked
        self.assertEqual(lits("t.cpp", "#if 0\nx'\n#elif 1\nconst char* s = \"oops\n;\n#else\ny'\n#endif\n"), [4])
        self.assertEqual(lits("t.cpp", "#if 1\n#else\ny'\n#endif\nconst char* s = \"oops\n;\n"), [5])
        # a condition it cannot evaluate keeps every branch checked, nested groups included
        self.assertEqual(lits("t.cpp", "#if X\nint a;\n#else\nconst char* s = \"oops\n;\n#endif\n"), [4])
        self.assertEqual(lits("t.cpp", "#ifdef A\n#if 1\nint a;\n#else\nz'\n#endif\n#else\nconst char* s = \"oops\n;\n#endif\n"), [8])

    def test_other_spellings_of_a_literal_zero_or_one(self):
        # `(0)`, `00`, `0L`, `0x0`, `0'0` are a zero the compiler drops; `(1)`, `1u`, `0x1` are a one.
        for zero in ("(0)", "00", "0L", "0ull", "0x0", "0b0", "0'0", "((0))", "0 /* off */",
                     "/* disabled */ 0", "(0 /* why */)", "( /* a */ 0 /* b */ )"):
            with self.subTest(zero=zero):
                self.assertEqual(lits("t.cpp", f"#if {zero}\nit's\n#endif\nint b;\n"), [])
                self.assertEqual(lits("t.cpp", f"#if X\n#elif {zero}\nit's\n#endif\n"), [])
        for one in ("(1)", "1u", "0x1", "2L", "010"):
            with self.subTest(one=one):
                self.assertEqual(lits("t.cpp", f"#if {one}\nint a;\n#else\ndon't\n#endif\n"), [])
                self.assertEqual(lits("t.cpp", f"#if {one}\nconst char* s = \"oops\n;\n#endif\n"), [2])
        # not one literal: evaluated by nobody here, so every branch stays checked
        for other in ("0 || X", "(0) + X", "08", "0x", "1.0", "(0"):
            with self.subTest(other=other):
                self.assertEqual(lits("t.cpp", f"#if {other}\nconst char* s = \"oops\n;\n#endif\n"), [2])

    def test_a_comment_before_the_directive_keyword(self):
        for opener in ("#/**/if 0", "/* note */ #if 0", "# /* a */ if /* b */ 0"):
            with self.subTest(opener=opener):
                self.assertEqual(lits("t.cpp", f"{opener}\nit's\n#endif\nint b;\n"), [])
        # a comment that only looks like a directive opener is not one
        self.assertEqual(lits("t.cpp", "/* #if 0 */\nconst char* s = \"oops\n;\n"), [2])

    def test_an_angle_header_name_opens_no_literal(self):
        self.assertEqual(lits("t.cpp", "#include <don't.h>\nint a;\n"), [])
        self.assertEqual(lits("t.cpp", "#  include_next <it's/a.h> // ok\nint a;\n"), [])
        # after the header name the line is code again, and a quoted name is still a literal
        self.assertEqual(lits("t.cpp", "#include <a.h> /* c\n*/ const char* s = \"oops\n;\n"), [2])
        self.assertEqual(lits("t.cpp", "#include \"don't.h\nint a;\n"), [1])


class QuotedPaths(unittest.TestCase):
    BROKEN = 'const char* s = "oops'

    def quoted_diff(self, field_minus, field_plus):
        return "\n".join(["diff --git x y", "--- " + field_minus, "+++ " + field_plus, "@@ -0,0 +1,2 @@",
                          "+" + self.BROKEN, "+;"]) + "\n"

    def test_a_name_with_a_blank_or_a_non_ascii_character_is_read(self):
        # git ends a name with a blank with a tab and C-quotes one with a non-ASCII character.
        for minus, plus, path in (("/dev/null", "b/src/my file.cpp\t", "src/my file.cpp"),
                                  ("/dev/null", '"b/\\303\\274nit.cpp"', "ünit.cpp"),
                                  ('"a/q\\"t.cpp"', '"b/q\\"t.cpp"', 'q"t.cpp')):
            with self.subTest(path=path):
                self.assertIn(path, dl.parse_diff(self.quoted_diff(minus, plus)))
                tree = dl.DictTree({path: self.BROKEN + "\n;\n"})
                found = dl.lint(self.quoted_diff(minus, plus), tree)
                self.assertEqual([(f[0], f[2]) for f in found], [(path, "bad-literal")])

    def test_a_file_the_head_tree_cannot_read_fails(self):
        found = dl.lint(diff("gone.cpp", added=["int x = 1;"]), dl.DictTree({}))
        self.assertEqual(rules(found), ["unread-file"])
        self.assertNotIn("unread-file", dl.WARN_RULES)

    def test_a_short_octal_escape_does_not_stop_the_lint(self):
        self.assertEqual(dl.diff_path('"b/x\\1y.cpp"', "b/"), "x\x01y.cpp")


if __name__ == "__main__":
    unittest.main()
