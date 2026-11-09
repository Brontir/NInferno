import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import affected_arches as aa

ALL = list(aa.ARCHES)


class Classify(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
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
            "src/ops/generic/generic_volta_hmma.cu": ["sm_70"],
            "src/ops/foo_sm75_tile.cu": ["sm_75"],
            "src/ops/gemma4/turing_decode.cu": ["sm_75"],
            "profiles/rtx4090.json": ["sm_89"],
            "src/ops/blackwell_fp4.cu": ["sm_120a"],
        }
        for path, arches in cases.items():
            with self.subTest(path=path):
                res = self.run_diff([path])
                self.assertEqual(res["arches"], arches)
                self.assertFalse(res["all"])

    def test_two_arch_files_union(self):
        res = self.run_diff(["src/ops/volta_a.cu", "src/ops/b_sm89.cu"])
        self.assertEqual(res["arches"], ["sm_70", "sm_89"])
        self.assertEqual(res["stands"]["sm_70"], ["lgv100"])
        self.assertEqual(res["stands"]["sm_89"], ["l4060", "l4070", "l4090"])

    def test_host_only_is_any_one_arch(self):
        res = self.run_diff(["src/chat/tool_calls.cpp", "src/serve/a.cpp", "docs/x.md",
                             "tests/chat/test_chat.cpp"])
        self.assertEqual(res["arches"], [])
        self.assertTrue(res["host"])
        self.assertEqual(list(res["stands"]), ["any"])
        self.assertEqual(res["ctest"], "ctest -L host on one stand")

    def test_other_arch_and_unknown_paths_answer_all(self):
        self.assertTrue(self.run_diff(["src/ops/hopper_fp8.cu"])["all"])
        self.assertTrue(self.run_diff(["src/ops/unmapped.cpp"])["all"])

    def test_empty_diff(self):
        res = self.run_diff([])
        self.assertEqual(res["arches"], [])
        self.assertEqual(res["ctest"], "no test needed")

    def test_ninja_listing_maps_objects_to_sources(self):
        self.tree({"src/ops/volta_a.cu": "", "src/chat/c.cpp": ""})
        repo = aa.Repo(self.root)
        text = (
            "[1/3] Building CUDA object src/ops/CMakeFiles/ops.dir/volta_a.cu.o\n"
            "[2/3] Building CXX object src/chat/CMakeFiles/chat.dir/c.cpp.o\n"
            "[3/3] Linking CXX executable bin/x\n")
        self.assertEqual(aa.sources_from_ninja(text, repo), ["src/chat/c.cpp", "src/ops/volta_a.cu"])
        res = aa.classify(aa.sources_from_ninja(text, repo), repo)
        self.assertEqual(res["arches"], ["sm_70"])
        self.assertTrue(res["host"])

    def test_cli_files_json(self):
        import contextlib
        import io
        import json
        self.tree({})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            aa.main(["--root", str(self.root), "--files", "src/ops/volta_a.cu", "--json"])
        self.assertEqual(json.loads(out.getvalue())["arches"], ["sm_70"])


if __name__ == "__main__":
    unittest.main()
