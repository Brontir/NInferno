import os
import re
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

    def test_a_chip_name_in_a_data_file_name_does_not_narrow(self):
        # a file's name says whose board it describes, not which build set reads it
        for path in ("profiles/gv100.json", "profiles/rtx4090.json", "tools/x_sm75_table.txt",
                     "tests/ops/moe_plan_golden/sm75/rtx3090.txt",
                     "src/ops/generic/data/kernel_facts/sm_86.txt"):
            with self.subTest(path=path):
                res = self.run_diff([path])
                self.assertEqual(res["arches"], ALL)
                self.assertTrue(res["all"])

    def test_golden_directories_answer_all(self):
        for path in ("tests/ops/moe_plan_golden/sm75/rtx3090.txt", "tests/ops/x_golden/a/b.txt",
                     "src/ops/generic/data/kernel_facts/sm_70.txt"):
            with self.subTest(path=path):
                self.assertTrue(self.run_diff([path])["all"])

    def test_arch_hint_is_a_whole_token(self):
        # "conv1000" holds "v100" and "data100" holds "a100" but neither is a chip name
        files = {"src/ops/sources.cmake": 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n conv1000.cu\nendif()\n'}
        self.assertEqual(self.run_diff(["src/ops/conv1000.cu"], files)["arches"], ALL)
        for name in ("conv1000", "data100", "xvolta", "gv1000", "rtx40900", "sm_700"):
            with self.subTest(name=name):
                self.assertEqual([a for a, pat in aa.ARCH_NAME_PATTERNS if re.search(pat, name + ".json")], [])
        self.assertIsNone(re.search(aa.OTHER_ARCH_NAME, "data100.json"))
        for name in ("rtx4090", "gv100", "x_sm75", "sm_86", "tu104", "volta_a"):
            with self.subTest(name=name):
                self.assertEqual(len([a for a, pat in aa.ARCH_NAME_PATTERNS if re.search(pat, name + ".json")]), 1)

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
        self.assertIn("included by a .cu", self.run_diff(["include/ninferno/pub.h"])["reasons"]["include/ninferno/pub.h"])

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
        self.assertEqual(res["ctest"], "full ctest on one stand")

    def test_ctest_line_is_always_the_full_suite(self):
        # the labels of the suite are not read, so no label filter is ever named
        labelled = ("add_test(NAME t1 COMMAND x)\nset_tests_properties(t1 PROPERTIES LABELS host)\n"
                    "add_test(NAME t2 COMMAND x)\nset_tests_properties(t2 PROPERTIES LABELS gpu)\n")
        cmake = {"tests/CMakeLists.txt": labelled}
        self.assertEqual(self.run_diff(["src/chat/a.cpp"], cmake)["ctest"], "full ctest on one stand")
        self.tree({"src/ops/sources.cmake":
                   'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'})
        res = self.run_diff(["src/ops/volta_a.cu"], cmake)
        self.assertEqual(res["ctest"], "full ctest on the listed stands")
        self.assertEqual(self.run_diff(["routes/a.json"], cmake)["ctest"], "full ctest on every arch's stand")
        for res in (self.run_diff(["src/chat/a.cpp"]), self.run_diff(["profiles/gv100.json"])):
            self.assertNotIn("-L", res["ctest"])

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
        self.tree({"src/ops/volta_a.cu": "",
                   "src/ops/sources.cmake": 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            aa.main(["--root", str(self.root), "--files", "src/ops/volta_a.cu", "--json"])
        self.assertEqual(json.loads(out.getvalue())["arches"], ["sm_70"])

    # ---- helpers for the tests that run the CLI over a git repository

    def git_out(self, *args):
        import subprocess
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    def cli(self, *argv):
        import contextlib
        import io
        import json
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            aa.main(["--root", str(self.root), "--json", *argv])
        return json.loads(out.getvalue())

    def cli_ninja(self, listing, *argv):
        saved = aa.run_ninja
        aa.run_ninja = lambda build_dir: listing
        try:
            return self.cli("--base", "base", "--head", "HEAD", "--ninja", "build", *argv)
        finally:
            aa.run_ninja = saved

    # ---- --ninja: the dry run is joined with the diff and never says "no test needed" by silence

    def ninja_repo(self):
        self.git("init", "-q")
        self.tree({"src/chat/c.cpp": "int c;\n", "src/ops/k.cu": "int k;\n", "routes/ga106.json": "{}\n"})
        self.commit_all("base", "base")

    def test_ninja_manifest_step_answers_all(self):
        self.ninja_repo()
        for listing in ("[0/1] Re-running CMake...\n",
                        "[0/1] Re-checking globbed directories...\n[0/1] Re-running CMake...\n",
                        "[0/1] Re-checking globbed directories...\r\n"):
            with self.subTest(listing=listing):
                res = self.cli_ninja(listing)
                self.assertTrue(res["all"])
                self.assertEqual(res["arches"], ALL)
                self.assertIn("manifest step", res["reasons"]["(ninja -n)"])

    def test_ninja_listing_without_build_lines_or_no_work_is_all(self):
        self.ninja_repo()
        for listing in ("", "[1/1] Linking CXX executable bin/x\n", "ninja: warning: something\n"):
            with self.subTest(listing=listing):
                res = self.cli_ninja(listing)
                self.assertTrue(res["all"])
                self.assertIn("no build line", res["reasons"]["(ninja -n)"])

    def test_ninja_no_work_to_do_on_a_clean_tree_is_none(self):
        self.ninja_repo()
        res = self.cli_ninja("ninja: no work to do.\n")
        self.assertEqual(res["arches"], [])
        self.assertEqual(res["ctest"], "no test needed")

    def test_ninja_no_work_to_do_still_sees_a_file_read_at_run_time(self):
        self.ninja_repo()
        # committed: the dry run cannot know that the tests read routes/ga106.json
        self.tree({"routes/ga106.json": '{"a": 1}\n'})
        self.commit_all("route table")
        res = self.cli_ninja("ninja: no work to do.\n")
        self.assertTrue(res["all"])
        self.assertIn("routes/ga106.json", res["reasons"])

    def test_ninja_sees_uncommitted_and_untracked_files(self):
        self.ninja_repo()
        self.tree({"routes/ga106.json": '{"a": 1}\n'})          # edited, not committed
        self.assertTrue(self.cli_ninja("ninja: no work to do.\n")["all"])
        self.git("checkout", "-q", "--", "routes/ga106.json")
        self.assertFalse(self.cli_ninja("ninja: no work to do.\n")["all"])
        self.tree({"routes/new.json": "{}\n"})                  # untracked
        self.assertTrue(self.cli_ninja("ninja: no work to do.\n")["all"])

    def test_ninja_sees_the_old_path_of_an_uncommitted_move(self):
        self.ninja_repo()
        (self.root / "docs").mkdir()
        self.git("mv", "routes/ga106.json", "docs/ga106.json")
        res = self.cli_ninja("ninja: no work to do.\n")
        self.assertTrue(res["all"])
        self.assertIn("routes/ga106.json", res["reasons"])

    def test_ninja_objects_are_joined_with_the_diff(self):
        self.ninja_repo()
        host_obj = "[1/2] Building CXX object src/chat/CMakeFiles/chat.dir/c.cpp.o\n[2/2] Linking CXX executable x\n"
        res = self.cli_ninja(host_obj)
        self.assertFalse(res["all"])
        self.assertTrue(res["host"])
        self.assertEqual(res["arches"], [])
        res = self.cli_ninja("[1/1] Building CUDA object src/ops/CMakeFiles/ops.dir/k.cu.o\n")
        self.assertTrue(res["all"])

    def test_ninja_listing_lines_as_a_real_dry_run_prints_them(self):
        self.tree({"src/chat/c.cpp": "", "src/ops/k.cu": ""})
        repo = aa.Repo(self.root)
        bs = chr(92)
        text = ("[0/3] Re-checking globbed directories...\r\n"
                "[1/3] Building CXX object " + bs.join(["src", "chat", "CMakeFiles", "chat.dir", "c.cpp.obj"]) + "\r\n"
                "[2/3] Building CUDA object src/ops/CMakeFiles/ops.dir/k.cu.o\n"
                "[3/3] Linking CXX executable bin" + bs + "x.exe\r\n")
        sources, why = aa.ninja_listing(text, repo)
        self.assertEqual(sources, ["src/chat/c.cpp", "src/ops/k.cu"])
        self.assertIn("manifest step", why)
        sources, why = aa.ninja_listing(text.split("\r\n", 1)[1], repo)
        self.assertEqual(sources, ["src/chat/c.cpp", "src/ops/k.cu"])
        self.assertIsNone(why)

    def test_ninja_resource_object_is_not_an_error(self):
        self.tree({"src/chat/c.cpp": ""})
        repo = aa.Repo(self.root)
        self.assertEqual(aa.sources_from_ninja("[1/1] Building RC object src/app.rc.res\n", repo), ["src/app.rc"])

    def test_ninja_candidates_are_normalised(self):
        self.tree({"src/ops/k.cu": "", "src/chat/k.cu": ""})
        repo = aa.Repo(self.root)
        self.assertEqual(aa.sources_from_ninja(
            "Building CUDA object src/ops/../ops/CMakeFiles/ops.dir/k.cu.o\n", repo), ["src/ops/k.cu"])
        bs = chr(92)
        self.assertEqual(aa.sources_from_ninja(
            "Building CUDA object " + bs.join(["src", "ops", "..", "ops", "CMakeFiles", "ops.dir", "k.cu.obj"]) + "\n",
            repo), ["src/ops/k.cu"])
        out = aa.sources_from_ninja("Building CXX object ../ext/CMakeFiles/e.dir/x.cpp.o\n", repo)
        self.assertEqual(out, ["../ext/x.cpp"])
        res = aa.classify(out, repo)
        self.assertTrue(res["all"])
        self.assertIn("outside the source tree", res["reasons"]["../ext/x.cpp"])

    # ---- the real tree

    def test_no_tracked_file_alone_answers_a_single_arch(self):
        import shutil
        import subprocess
        root = Path(__file__).resolve().parents[1]
        if shutil.which("git") is None:
            self.skipTest("git not available")
        cp = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
        files = [n for n in cp.stdout.decode("utf-8", errors="replace").split("\0") if n]
        if cp.returncode or len(files) < 100:
            self.skipTest("not a git checkout")
        repo = aa.Repo(root)
        narrow = {}
        for f in files:
            res = aa.classify([f], repo)
            if not res["all"] and len(res["arches"]) == 1:
                narrow[f] = res["reasons"][f]
        # A file read by every build set must not answer one arch because of its name. The allowlist
        # is empty today: no tracked file is a source listed under one arch's guard alone.
        allowed = {}
        self.assertEqual({k: v for k, v in narrow.items() if k not in allowed}, {})
        for f, why in (("src/convert/ggml_tables.inc", "included by a .cu"),
                       ("tests/ops/moe_plan_golden/sm75/rtx3090.txt", "golden"),
                       ("src/ops/generic/data/kernel_facts/sm_86.txt", "data"),
                       ("tools/kernel_facts/kernel_facts.py", "no rule")):
            if f in files:
                with self.subTest(path=f):
                    res = aa.classify([f], repo)
                    self.assertTrue(res["all"], res)
                    self.assertIn(why, res["reasons"][f])

    # ---- the include graph holds every file an #include resolves to

    def test_any_file_a_cu_includes_is_a_node_of_the_graph(self):
        files = {"tests/ops/t.cu": '#include "q.h"\n',
                 "tests/ops/q.h": '#include "convert/tables.inc"\n#include "convert/table.txt"\n',
                 "src/convert/tables.inc": "int t[1];\n", "src/convert/table.txt": "1\n",
                 "src/convert/other.txt": "2\n"}
        for path in ("src/convert/tables.inc", "src/convert/table.txt"):
            with self.subTest(path=path):
                res = self.run_diff([path], files)
                self.assertTrue(res["all"])
                self.assertIn("included by a .cu", res["reasons"][path])
        # host text that no .cu reaches stays host
        self.assertTrue(self.run_diff(["src/convert/other.txt"])["host"])
        self.assertFalse(self.run_diff(["src/convert/other.txt"])["all"])

    def test_include_graph_of_a_git_ref_holds_every_extension(self):
        self.git("init", "-q")
        self.tree({"tests/ops/t.cu": '#include "convert/table.txt"\n', "src/convert/table.txt": "1\n"})
        self.commit_all("base", "base")
        res = aa.classify(["src/convert/table.txt"], aa.Repo(self.root, ref="HEAD"))
        self.assertTrue(res["all"])

    def test_resolve_follows_every_file_of_that_name(self):
        for first, second in (("src/chat/util.h", "src/zzz/util.h"), ("src/aaa/util.h", "src/chat/util.h")):
            with self.subTest(first=first):
                res = self.run_diff(["src/chat/util.h"], {"src/ops/k.cu": '#include "util.h"\n',
                                                          first: "", second: ""})
                self.assertTrue(res["all"])
                self.root = Path(tempfile.mkdtemp(dir=self.dir.name))

    def test_include_after_a_block_comment_is_seen(self):
        files = {"src/ops/k.cu": '/* note */ #include "host_h.h"\n', "src/chat/host_h.h": ""}
        self.assertTrue(self.run_diff(["src/chat/host_h.h"], files)["all"])

    def test_a_computed_include_answers_all(self):
        files = {"src/chat/pick.h": "#define ARCH_HEADER <arch.h>\n#include ARCH_HEADER\n"}
        res = self.run_diff(["src/chat/pick.h"], files)
        self.assertTrue(res["all"])
        self.assertIn("computed include", res["reasons"]["src/chat/pick.h"])
        self.assertTrue(self.run_diff(["src/chat/ok.h"], {"src/chat/ok.h": '#include <vector>\n#include "a.h"\n'})["host"])

    # ---- renames: old and new path are both in the diff

    def rename_repo(self, old, new, old_text, new_text):
        self.git("init", "-q")
        self.tree({old: old_text})
        self.commit_all("base", "base")
        (self.root / old).unlink()
        self.tree({new: new_text})
        self.commit_all("moves")

    def test_diff_lists_both_paths_of_a_rename(self):
        body = "".join(f"line {i}\n" for i in range(20))
        self.rename_repo("routes/q.json", "docs/q.json", body, body + "one more\n")
        self.assertEqual(sorted(aa.git_diff_paths("base", "HEAD", self.root)), ["docs/q.json", "routes/q.json"])
        self.assertTrue(self.cli("--base", "base", "--head", "HEAD")["all"])

    def test_a_moved_header_loses_its_macro_and_still_answers_all(self):
        body = "".join(f"int f{i}();\n" for i in range(20))
        self.rename_repo("src/ops/arch.h", "src/chat/arch.h", body + "#ifdef __CUDA_ARCH__\n#endif\n", body)
        self.assertTrue(self.cli("--base", "base", "--head", "HEAD")["all"])

    def test_a_kernel_moved_to_docs_answers_all(self):
        body = "".join(f"// line {i}\n" for i in range(20))
        self.rename_repo("src/ops/k2.cu", "docs/k2.md", body, body + "text\n")
        res = self.cli("--base", "base", "--head", "HEAD")
        self.assertTrue(res["all"])
        self.assertIn("src/ops/k2.cu", res["reasons"])

    def test_paths_with_spaces_and_non_ascii_names_come_through_unquoted(self):
        self.git("init", "-q")
        self.tree({"docs/a.md": "x\n"})
        self.commit_all("base", "base")
        self.tree({"routes/café table.json": "{}\n"})
        self.commit_all("adds")
        self.assertEqual(aa.git_diff_paths("base", "HEAD", self.root), ["routes/café table.json"])
        self.assertTrue(self.cli("--base", "base", "--head", "HEAD")["all"])

    # ---- the host class

    def test_tools_fixtures_and_artifact_answer_all(self):
        for path in ("tools/kernel_facts/kernel_facts.py", "tools/bench/x.py", "tests/fixtures/plan/bench_models.json",
                     "src/artifact/materializer.cpp", "tests/artifact/test_materialization.cpp"):
            with self.subTest(path=path):
                self.assertTrue(self.run_diff([path])["all"])
        for path in ("tools/README.md", "docs/reference/serving.md", "src/chat/tool_calls.cpp"):
            with self.subTest(path=path):
                res = self.run_diff([path])
                self.assertTrue(res["host"] and not res["all"])

    def test_host_directory_files_that_are_not_host_code_answer_all(self):
        for path in ("src/serve/defaults.yaml", "src/convert/config.h.in", "src/convert/gen.sh",
                     "src/convert/tables.def", "src/chat/data.bin", ".github/workflows/x.yml.in"):
            with self.subTest(path=path):
                self.assertTrue(self.run_diff([path])["all"])
        self.assertTrue(self.run_diff([".github/workflows/ci.yml"])["host"])
        self.assertTrue(self.run_diff(["src/chat/a.cpp"])["host"])

    # ---- --files paths

    def test_files_spellings_are_normalised(self):
        self.tree({"routes/ga106.json": "{}\n", "src/core/data/gpu_facts.json": "{}\n"})
        bs = chr(92)
        spellings = ["routes/ga106.json", "routes" + bs + "ga106.json", "./routes/ga106.json",
                     "." + bs + "routes" + bs + "ga106.json", "docs/../routes/ga106.json",
                     str(self.root / "routes" / "ga106.json"),
                     (self.root / "routes" / "ga106.json").as_posix()]
        for sp in spellings:
            with self.subTest(path=sp):
                res = self.cli("--files", sp)
                self.assertTrue(res["all"])
                self.assertEqual(list(res["reasons"]), ["routes/ga106.json"])
        self.assertEqual(self.cli("--files", "docs/../src/core/data/gpu_facts.json")["reasons"].keys(),
                         {"src/core/data/gpu_facts.json"})

    def test_files_outside_the_root_are_rejected(self):
        outside = str(self.root.parent / "other.cpp")
        for sp in ("../x.cpp", "docs/../../x.cpp", ".", "./", "..", outside, "/definitely/not/here.cpp"):
            with self.subTest(path=sp):
                with self.assertRaises(SystemExit) as cm:
                    aa.main(["--root", str(self.root), "--files", sp])
                self.assertIn("repository root", str(cm.exception))

    def test_normalise_path_unit(self):
        n = lambda p: aa.normalise_path(p, self.root)
        self.assertEqual(n("a//b/./c/../d.h"), "a/b/d.h")
        self.assertEqual(n("a" + chr(92) + "b.h"), "a/b.h")
        if os.name == "nt":
            self.assertEqual(n(str(self.root / "a" / "b.h").lower()), "a/b.h")   # drive paths: case-blind

    # ---- ref mode reads the tree of the ref, not the checkout, and keeps tracked build-named paths

    def test_build_trees_are_excluded_by_the_path_below_the_root(self):
        self.root = self.root / "build-x" / "repo"        # a root that itself lives below a build-like name
        guarded = 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'
        files = {"src/ops/volta_a.cu": "", "src/ops/sources.cmake": guarded,
                 "build/CMakeFiles/ops.dir/DependInfo.cmake": "add_library(x volta_a.cu)\n",
                 "out/CMakeCache.txt": "", "out/sub/CMakeLists.txt": "add_library(x volta_a.cu)\n"}
        res = self.run_diff(["src/ops/volta_a.cu"], files)
        self.assertEqual(res["arches"], ["sm_70"])

    def test_tracked_paths_named_build_take_part_in_a_ref(self):
        self.git("init", "-q")
        self.tree({"src/convert/build/plan.h": "int plan();\n", "tests/ops/t.cu": '#include "build/plan.h"\n',
                   "src/chat/build-info.cpp": "#ifdef __CUDA_ARCH__\n#endif\n"})
        self.commit_all("base", "base")
        res = aa.classify(["src/convert/build/plan.h", "src/chat/build-info.cpp"], aa.Repo(self.root, ref="HEAD"))
        self.assertTrue(res["all"])
        self.assertIn("included by a .cu", res["reasons"]["src/convert/build/plan.h"])
        self.assertIn("arch macro", res["reasons"]["src/chat/build-info.cpp"])
        res = aa.classify(["src/chat/build-info.cpp"], aa.Repo(self.root, ref="HEAD"))
        self.assertTrue(res["all"])

    def test_cmake_listing_of_a_ref_is_read_from_git(self):
        self.git("init", "-q")
        unguarded = "add_library(x volta_a.cu)\n"
        self.tree({"src/ops/volta_a.cu": "", "src/ops/sources.cmake": unguarded})
        self.commit_all("base", "base")
        self.tree({"src/ops/sources.cmake": 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_a.cu\nendif()\n'})
        self.commit_all("guards it", "feature")
        self.git("checkout", "-q", "base")                 # the checkout lists it unguarded
        res = aa.classify(["src/ops/volta_a.cu"], aa.Repo(self.root, ref="feature"))
        self.assertEqual(res["arches"], ["sm_70"])

    # ---- the base side of a diff, through main

    def test_main_judges_the_base_side_of_a_removed_macro(self):
        self.git("init", "-q")
        self.tree({"src/chat/y.cpp": "#ifdef __CUDA_ARCH__\nint f();\n#endif\n"})
        self.commit_all("base", "base")
        self.tree({"src/chat/y.cpp": "int f();\n"})
        self.commit_all("removes the macro")
        res = self.cli("--base", "base", "--head", "HEAD")
        self.assertTrue(res["all"])
        self.assertIn("before or after", res["reasons"]["src/chat/y.cpp"])

    def test_main_uses_the_merge_base_not_the_tip_of_the_base_ref(self):
        self.git("init", "-q")
        self.tree({"src/chat/y.cpp": "#ifdef __CUDA_ARCH__\nint f();\n#endif\n"})
        self.commit_all("fork point", "fork")
        self.git("checkout", "-q", "-b", "feature")
        self.tree({"src/chat/y.cpp": "int f();\nint g();\n"})
        self.commit_all("feature removes the macro", "feature")
        self.git("checkout", "-q", "fork")
        self.git("checkout", "-q", "-b", "tip")
        self.tree({"src/chat/y.cpp": "int f();\nint h();\n", "routes/a.json": "{}\n"})   # the base moved on
        self.commit_all("base moved", "tip")
        res = self.cli("--base", "tip", "--head", "feature")
        self.assertTrue(res["all"])
        self.assertIn("before or after", res["reasons"]["src/chat/y.cpp"])

    def test_diff_is_three_dot(self):
        self.git("init", "-q")
        self.tree({"src/chat/z.cpp": "int z;\n"})
        self.commit_all("fork point", "fork")
        self.git("checkout", "-q", "-b", "feature")
        self.tree({"src/chat/z.cpp": "int z2;\n"})
        self.commit_all("feature", "feature")
        self.git("checkout", "-q", "fork")
        self.git("checkout", "-q", "-b", "tip")
        self.tree({"routes/a.json": "{}\n"})
        self.commit_all("base moved", "tip")
        self.assertEqual(aa.git_diff_paths("tip", "feature", self.root), ["src/chat/z.cpp"])
        res = self.cli("--base", "tip", "--head", "feature")
        self.assertFalse(res["all"])
        self.assertTrue(res["host"])

    def test_base_device_headers_are_part_of_the_answer(self):
        # the kernel's include of the header goes away in the head: the base side still reaches it
        self.git("init", "-q")
        self.tree({"src/ops/k.cu": '#include "chat/h.h"\n', "src/chat/h.h": "int h();\n"})
        self.commit_all("base", "base")
        self.tree({"src/ops/k.cu": "int k;\n"})
        self.commit_all("kernel drops the include", "feature")
        head, base = aa.Repo(self.root, ref="feature"), aa.Repo(self.root, ref="base")
        self.assertFalse(aa.classify(["src/chat/h.h"], head)["all"])
        res = aa.classify(["src/chat/h.h"], head, base=base)
        self.assertTrue(res["all"])
        self.assertIn("included by a .cu", res["reasons"]["src/chat/h.h"])

    # ---- reading a ref

    def test_a_gitlink_does_not_shift_the_blobs_after_it(self):
        self.git("init", "-q")
        self.tree({"src/chat/arch.cpp": "#ifdef __CUDA_ARCH__\n#endif\n", "src/chat/plain.cpp": "int p;\n",
                   "src/zz.cpp": "int z;\n"})
        self.commit_all("base")
        sha = self.git_out("rev-parse", "HEAD")
        self.git("update-index", "--add", "--cacheinfo", f"160000,{sha},src/a_link.cpp")
        self.git("commit", "-q", "-m", "adds a gitlink")
        repo = aa.Repo(self.root, ref="HEAD")
        self.assertIn("__CUDA_ARCH__", repo._text["src/chat/arch.cpp"])
        self.assertEqual(repo._text["src/chat/plain.cpp"], "int p;\n")
        self.assertEqual(repo._text["src/zz.cpp"], "int z;\n")
        self.assertTrue(aa.classify(["src/chat/arch.cpp"], repo)["all"])

    def test_utf16_source_with_a_macro_answers_all(self):
        self.git("init", "-q")
        path = self.root / "src" / "chat" / "w.cpp"
        path.parent.mkdir(parents=True)
        path.write_bytes("#ifdef __CUDA_ARCH__\n#endif\n".encode("utf-16"))
        self.commit_all("utf16", "base")
        self.assertTrue(aa.classify(["src/chat/w.cpp"], aa.Repo(self.root, ref="HEAD"))["all"])
        self.assertTrue(aa.classify(["src/chat/w.cpp"], aa.Repo(self.root))["all"])

    # ---- cmake guards

    def test_multi_line_guard_over_two_arches_is_not_one_arch(self):
        cmake = ('if(NINFERNO_BUILD_ARCH STREQUAL "70"\n    OR NINFERNO_BUILD_ARCH STREQUAL "75")\n'
                 '  volta_k.cu\nendif()\n')
        res = self.run_diff(["src/ops/volta_k.cu"], {"src/ops/sources.cmake": cmake, "src/ops/volta_k.cu": ""})
        self.assertEqual(res["arches"], ALL)

    def test_guarded_header_included_by_a_cu_is_all(self):
        files = {"src/ops/sources.cmake": 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n volta_tiles.h\nendif()\n',
                 "src/ops/volta_tiles.h": "", "src/ops/k.cu": '#include "volta_tiles.h"\n'}
        self.assertTrue(self.run_diff(["src/ops/volta_tiles.h"], files)["all"])


if __name__ == "__main__":
    unittest.main()
