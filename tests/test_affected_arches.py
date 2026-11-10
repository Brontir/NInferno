import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import affected_arches as aa

ALL = list(aa.ARCHES)
EXAMPLE = Path(__file__).resolve().parents[1] / "tools" / "audit" / "stands.example.json"


class Classify(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.saved_env = os.environ.get(aa.STANDS_ENV)
        os.environ[aa.STANDS_ENV] = str(EXAMPLE)

    def tearDown(self):
        if self.saved_env is None:
            os.environ.pop(aa.STANDS_ENV, None)
        else:
            os.environ[aa.STANDS_ENV] = self.saved_env
        self.dir.cleanup()

    def tree(self, files):
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def run_diff(self, paths, files=None):
        self.tree(files or {})
        return aa.classify(paths, aa.Repo(self.root))

    def test_all_for_data_and_build_files(self):
        for path in ("routes/ga106.json", "src/core/data/gpu_facts.json", "CMakeLists.txt",
                     "src/ops/CMakeLists.txt", "tests/convert_native/tests.cmake",
                     "tests/ops/linear/selector_golden/base/a10.txt"):
            with self.subTest(path=path):
                res = self.run_diff([path])
                self.assertEqual(res["arches"], ALL)
                self.assertTrue(res["all"])

    def test_arch_macros_answer_all(self):
        for macro in ("NINFERNO_BUILD_ARCH", "WITHOUT_NVFP4", "__CUDA_ARCH__"):
            with self.subTest(macro=macro):
                res = self.run_diff(["src/ops/x.cpp"], {"src/ops/x.cpp": f"#if {macro}\n#endif\n"})
                self.assertEqual(res["arches"], ALL)

    def test_header_of_a_cu_file_is_all_transitively(self):
        files = {
            "src/ops/k.cu": '#include "mid.h"\n',
            "src/ops/mid.h": '#include "deep/leaf.h"\n',
            "src/ops/deep/leaf.h": "int leaf();\n",
            "src/ops/host_only.h": "int host();\n",
        }
        self.assertEqual(self.run_diff(["src/ops/deep/leaf.h"], files)["arches"], ALL)
        self.assertEqual(self.run_diff(["src/ops/mid.h"])["arches"], ALL)
        res = self.run_diff(["src/ops/host_only.h"])
        self.assertEqual(res["arches"], ALL)  # no host prefix, no rule: conservative
        self.assertIn("no rule", res["reasons"]["src/ops/host_only.h"])

    def test_per_arch_names(self):
        cases = {
            "profiles/gv100.json": ["sm_70"],
            "profiles/rtx4090.json": ["sm_89"],
            "tools/x_sm75_table.txt": ["sm_75"],
        }
        for path, arches in cases.items():
            with self.subTest(path=path):
                res = self.run_diff([path])
                self.assertEqual(res["arches"], arches)
                self.assertFalse(res["all"])

    def test_volta_cu_built_everywhere_is_all(self):
        files = {"src/ops/sources.cmake": 'target_sources(ops PRIVATE "${D}/volta_a.cu")\n',
                 "src/ops/volta_a.cu": "int takes() { return 0; }\n"}
        res = self.run_diff(["src/ops/volta_a.cu"], files)
        self.assertEqual(res["arches"], ALL)

    def test_volta_define_and_header_answer_all(self):
        files = {"src/ops/volta_b.cu": "#ifdef NINFERNO_VOLTA_SPLITD\n#endif\n",
                 "src/ops/volta_h.cuh": "",
                 "src/ops/k.cu": '#include "volta_h.cuh"\n'}
        self.assertEqual(self.run_diff(["src/ops/volta_b.cu"], files)["arches"], ALL)
        self.assertEqual(self.run_diff(["src/ops/volta_h.cuh"])["arches"], ALL)

    def test_cu_listed_only_under_an_arch_guard_is_that_arch(self):
        files = {"src/ops/sources.cmake":
                 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n  "${D}/volta_c.cu"\nendif()\n',
                 "src/ops/volta_c.cu": ""}
        self.assertEqual(self.run_diff(["src/ops/volta_c.cu"], files)["arches"], ["sm_70"])

    def test_negated_or_ranged_guards_are_all(self):
        for cond in ('NOT NINFERNO_BUILD_ARCH STREQUAL "70"', "NINFERNO_BUILD_ARCH GREATER_EQUAL 70",
                     'CMAKE_CUDA_ARCHITECTURES MATCHES "70;75"'):
            with self.subTest(cond=cond):
                files = {"src/ops/sources.cmake": f"if({cond})\n  volta_d.cu\nendif()\n",
                         "src/ops/volta_d.cu": ""}
                self.assertEqual(self.run_diff(["src/ops/volta_d.cu"], files)["arches"], ALL)

    def test_ui_serve_and_pdl_allowlists_are_all(self):
        for path in ("src/ui/catalog.cpp", "src/serve/request_log.cpp", "apps/cli/main.cpp",
                     "tests/ops/pdl_audit_sabotage_allow.txt", "tools/pdl_prewait_loads.txt"):
            with self.subTest(path=path):
                self.assertTrue(self.run_diff([path])["all"])

    def test_include_dir_and_dotdot_includes(self):
        files = {"src/ops/k.cu": '#include "../util/../api/h.h"\n#include "ninferno/pub.h"\n',
                 "src/api/h.h": "", "include/ninferno/pub.h": ""}
        self.assertTrue(self.run_diff(["src/api/h.h"], files)["all"])
        self.assertIn("header", self.run_diff(["include/ninferno/pub.h"])["reasons"]["include/ninferno/pub.h"])

    def test_two_arch_files_union(self):
        self.tree({"src/ops/sources.cmake":
                   'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'
                   'if(NINFERNO_BUILD_ARCH STREQUAL "89")\n b_sm89.cu\nendif()\n'})
        res = self.run_diff(["src/ops/volta_a.cu", "src/ops/b_sm89.cu"])
        self.assertEqual(res["arches"], ["sm_70", "sm_89"])
        self.assertEqual(res["stands"]["sm_70"], ["stand-a"])
        self.assertEqual(res["stands"]["sm_89"], ["stand-e", "stand-f", "stand-g"])

    def test_no_stands_file_names_no_machine(self):
        os.environ[aa.STANDS_ENV] = str(self.root / "missing.json")
        with self.assertRaises(SystemExit):
            self.run_diff(["src/chat/tool_calls.cpp"])
        del os.environ[aa.STANDS_ENV]
        saved = aa.STANDS_LOCAL
        aa.STANDS_LOCAL = self.root / "none.json"
        try:
            res = self.run_diff(["src/chat/tool_calls.cpp"])
            self.assertEqual(res["stands"], {"any": []})
            res = self.run_diff(["src/ops/other.cu"], {"src/ops/other.cu": ""})
            self.assertEqual(res["stands"]["sm_70"], [])
        finally:
            aa.STANDS_LOCAL = saved

    def test_malformed_stands_file_lists_no_machine(self):
        bad = self.root / "bad.json"
        for text in ("{not json", '{"stands": ["sm_70"]}', '{"stands": {"sm_70": "stand-a"}}',
                     '{"stands": {"sm_99": ["x"]}, "any_one": 3}', "[]"):
            with self.subTest(text=text):
                bad.write_text(text, encoding="utf-8")
                os.environ[aa.STANDS_ENV] = str(bad)
                try:
                    res = self.run_diff(["src/ops/volta_a.cu"],
                                        {"src/ops/sources.cmake":
                                         'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'})
                finally:
                    del os.environ[aa.STANDS_ENV]
                self.assertEqual(res["arches"], ["sm_70"])
                self.assertEqual(res["stands"]["sm_70"], [])

    def test_host_only_is_any_one_arch(self):
        res = self.run_diff(["src/chat/tool_calls.cpp", "src/convert/a.cpp", "docs/x.md",
                             "tests/chat/test_chat.cpp"])
        self.assertEqual(res["arches"], [])
        self.assertTrue(res["host"])
        self.assertEqual(list(res["stands"]), ["any"])
        # no test carries a LABELS property in this tree: no label filter is recommended
        self.assertTrue(res["ctest"].startswith("full ctest on one stand"), res["ctest"])
        self.assertIn("0 of 0", res["ctest"])

    LABELLED = ("add_test(NAME t1 COMMAND x)\nset_tests_properties(t1 PROPERTIES LABELS host)\n"
                "add_test(NAME t2 COMMAND x)\nset_tests_properties(t2 PROPERTIES LABELS gpu)\n")

    def test_label_filter_only_when_every_test_is_labelled(self):
        partial = "add_test(NAME t3 COMMAND x)\n"
        res = self.run_diff(["src/chat/a.cpp"], {"tests/CMakeLists.txt": self.LABELLED + partial})
        self.assertTrue(res["ctest"].startswith("full ctest on one stand"), res["ctest"])
        self.assertIn("2 of 3", res["ctest"])
        res = self.run_diff(["src/chat/a.cpp"], {"tests/CMakeLists.txt": self.LABELLED})
        self.assertEqual(res["ctest"], "ctest -L host on one stand")
        res = self.run_diff(["profiles/gv100.json"], {"tests/CMakeLists.txt": self.LABELLED})
        self.assertEqual(res["ctest"], "ctest -L 'host|gpu|golden' on the listed stands")
        both = ("add_test(NAME a COMMAND x)\nadd_test(NAME b COMMAND x)\n"
                "set_tests_properties(a b PROPERTIES LABELS host)\n")
        res = self.run_diff(["src/chat/a.cpp"], {"tests/CMakeLists.txt": both})
        self.assertEqual(res["ctest"], "ctest -L host on one stand")

    def test_ignored_build_trees_do_not_take_part_in_cmake_discovery(self):
        guarded = 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'
        unguarded = "add_library(x volta_a.cu)\n"
        files = {"src/ops/volta_a.cu": "", "src/ops/sources.cmake": guarded}
        for d in ("build", "build-debug", "cmake-build-release"):
            files[d + "/CMakeFiles/ops.dir/DependInfo.cmake"] = unguarded
            files[d + "/sub/CMakeLists.txt"] = unguarded
        res = self.run_diff(["src/ops/volta_a.cu"], files)
        self.assertEqual(res["arches"], ["sm_70"])
        self.assertFalse(res["all"])

    def git(self, *args):
        import shutil
        import subprocess
        if shutil.which("git") is None:
            self.skipTest("git not available")
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=t", "-c", "user.email=t@t",
                        *args], check=True, capture_output=True)

    def commit_all(self, msg, tag=None):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        if tag:
            self.git("tag", tag)

    def test_head_ref_is_read_from_git_not_from_the_checkout(self):
        import contextlib
        import io
        import json
        self.git("init", "-q")
        self.tree({"src/chat/tool_calls.cpp": "int f() { return 1; }\n"})
        self.commit_all("base", "base")
        self.tree({"src/chat/tool_calls.cpp": "#if NINFERNO_BUILD_ARCH == 70\nint f();\n#endif\n"})
        self.commit_all("adds an arch macro", "feature")
        self.git("checkout", "-q", "base")          # the checkout stays at the base
        paths = aa.git_diff_paths("base", "feature", self.root)
        self.assertEqual(paths, ["src/chat/tool_calls.cpp"])
        self.assertFalse(aa.classify(paths, aa.Repo(self.root))["all"])   # the stale checkout
        res = aa.classify(paths, aa.Repo(self.root, ref="feature"), base=aa.Repo(self.root, ref="base"))
        self.assertTrue(res["all"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            aa.main(["--root", str(self.root), "--base", "base", "--head", "feature", "--json"])
        self.assertTrue(json.loads(out.getvalue())["all"])

    def test_include_graph_of_the_head_ref(self):
        self.git("init", "-q")
        self.tree({"src/ops/k.cu": "// none\n", "src/ops/h.h": "int x;\n"})
        self.commit_all("base", "base")
        self.tree({"src/ops/k.cu": '#include "h.h"\n'})
        self.commit_all("kernel now includes h.h", "feature")
        self.git("checkout", "-q", "base")
        res = aa.classify(["src/ops/h.h"], aa.Repo(self.root, ref="feature"),
                          base=aa.Repo(self.root, ref="base"))
        self.assertTrue(res["all"])

    def test_removing_the_last_arch_macro_still_answers_all(self):
        self.git("init", "-q")
        self.tree({"src/chat/tool_calls.cpp": "#ifdef __CUDA_ARCH__\nint f();\n#endif\n"})
        self.commit_all("base")
        self.tree({"src/chat/tool_calls.cpp": "int f();\n"})
        self.commit_all("removes the macro")
        paths = ["src/chat/tool_calls.cpp"]
        self.assertFalse(aa.classify(paths, aa.Repo(self.root, ref="HEAD"))["all"])   # head alone
        res = aa.classify(paths, aa.Repo(self.root, ref="HEAD"), base=aa.Repo(self.root, ref="HEAD~1"))
        self.assertTrue(res["all"])
        self.assertIn("before or after", res["reasons"][paths[0]])

    def test_other_arch_and_unknown_paths_answer_all(self):
        self.assertTrue(self.run_diff(["src/ops/hopper_fp8.cu"])["all"])
        self.assertTrue(self.run_diff(["src/ops/unmapped.cpp"])["all"])

    def test_empty_diff(self):
        res = self.run_diff([])
        self.assertEqual(res["arches"], [])
        self.assertEqual(res["ctest"], "no test needed")

    def test_ninja_listing_maps_objects_to_sources(self):
        self.tree({"src/ops/volta_a.cu": "", "src/chat/c.cpp": "",
                   "src/ops/sources.cmake": 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'})
        repo = aa.Repo(self.root)
        text = (
            "[1/3] Building CUDA object src/ops/CMakeFiles/ops.dir/volta_a.cu.o\n"
            "[2/3] Building CXX object src/chat/CMakeFiles/chat.dir/c.cpp.o\n"
            "[3/3] Linking CXX executable bin/x\n")
        self.assertEqual(aa.sources_from_ninja(text, repo), ["src/chat/c.cpp", "src/ops/volta_a.cu"])
        res = aa.classify(aa.sources_from_ninja(text, repo), repo)
        self.assertEqual(res["arches"], ["sm_70"])
        self.assertTrue(res["host"])

    NINJA_TREE = {"src/ops/volta_a.cu": "", "src/chat/c.cpp": "", "src/my dir/sp.cpp": "",
                  "src/ops/sources.cmake": 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'}

    def test_ninja_windows_listing_selects_same_arches_as_linux(self):
        self.tree(self.NINJA_TREE)
        repo = aa.Repo(self.root)
        bs = chr(92)
        linux = ("[1/2] Building CUDA object src/ops/CMakeFiles/ops.dir/volta_a.cu.o\n"
                 "[2/2] Building CXX object src/chat/CMakeFiles/chat.dir/c.cpp.o\n")
        win = ("[1/2] Building CUDA object " + bs.join(["src", "ops", "CMakeFiles", "ops.dir", "volta_a.cu.obj"])
               + "\r\n[2/2] Building CXX object src/chat/CMakeFiles/chat.dir/c.cpp.obj\r\n")
        self.assertEqual(aa.sources_from_ninja(win, repo), aa.sources_from_ninja(linux, repo))
        res = aa.classify(aa.sources_from_ninja(win, repo), repo)
        self.assertEqual(res["arches"], ["sm_70"])
        self.assertNotEqual(res["ctest"], "no test needed")

    def test_ninja_windows_odd_spellings(self):
        self.tree(self.NINJA_TREE)
        repo = aa.Repo(self.root)
        bs = chr(92)
        for line in ('Building CXX object "src/my dir/CMakeFiles/t.dir/sp.cpp.obj"',
                     "Building CXX object " + bs.join(["src", "my dir", "CMakeFiles", "t.dir", "sp.cpp.obj"]) + "\r",
                     "Building CXX object src/my dir/CMakeFiles/t.dir/sp.cpp.o"):
            self.assertEqual(aa.sources_from_ninja(line + "\n", repo), ["src/my dir/sp.cpp"], line)
        self.assertEqual(aa.sources_from_ninja(
            "Building CUDA object src/ops/CMakeFiles/ops.dir/volta_a.cuda.obj\n", repo),
            ["src/ops/volta_a.cu"])

    def test_ninja_unparsed_build_lines_fail_loudly(self):
        self.tree(self.NINJA_TREE)
        repo = aa.Repo(self.root)
        with self.assertRaises(SystemExit) as cm:
            aa.sources_from_ninja("[1/1] Building CXX object weird/thing.cpp.gch\n", repo)
        self.assertIn("not understood", str(cm.exception))
        # a dry run with no build lines at all is a legitimate "nothing to rebuild"
        self.assertEqual(aa.sources_from_ninja("ninja: no work to do.\n", repo), [])
        # through the CLI path the exit is non-zero
        saved = aa.subprocess.run
        aa.subprocess.run = lambda *a, **k: aa.subprocess.CompletedProcess(
            a, 0, "[1/1] Building CXX object x.cpp.weird\n", "")
        try:
            with self.assertRaises(SystemExit) as cm:
                aa.main(["--root", str(self.root), "--ninja", "build"])
            self.assertTrue(cm.exception.code)
        finally:
            aa.subprocess.run = saved

    def test_cli_files_json(self):
        import contextlib
        import io
        import json
        self.tree({})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            aa.main(["--root", str(self.root), "--files", "profiles/gv100.json", "--json"])
        self.assertEqual(json.loads(out.getvalue())["arches"], ["sm_70"])


if __name__ == "__main__":
    unittest.main()
