import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import golden_gate as gate
import regret_due as due

ROOT = Path(__file__).resolve().parents[1]
CHIPS = {"ga100": 80, "ga102": 86, "ga106": 86, "ad102": 89, "ad104": 89, "gb202": 120,
         "gv100": 70, "tu102": 75}
PATTERNS = gate.load_patterns("\n".join(["src/ops/generic/generic_cost.cpp", "routes/**",
                                         "src/core/data/gpu_facts.json", *due.EXTRA_PATTERNS]))


def boards():
    reports = {
        "rtx3090.json": {"board": "rtx3090", "card": "RTX 3090",
                         "regret_report": {"device": "ga102:82sm"}},
        "rtx4090.json": {"board": "rtx4090", "card": "RTX 4090",
                         "regret_report": {"device": "ad102:128sm"}},
        "rtx5090.json": {"board": "rtx5090", "card": "RTX 5090",
                         "regret_report": {"device": "gb202:170sm"}},
        "gv100.json": {"board": "gv100", "card": "Quadro GV100",
                       "regret_report": {"device": "gv100:80sm"}},
        "policy_evidence_moves.json": {"schema": "x"},
    }
    return due.lab_boards(reports, CHIPS)


def table(cls):
    return json.dumps({"schema": "ninferno-route-table/1", "gpu": {"class": cls}, "entries": []})


def run(changed, tables=None, arches=(), kernels=()):
    """answer() with a stub classification: every path answers `arches` (all five: "all";
    none: host code)."""
    calls = []

    def classify_one(path):
        calls.append(path)
        everything = set(arches) == set(due.aa.ARCHES)
        return {"arches": list(arches), "all": everything, "host": everything or not arches}

    res = due.answer(changed, PATTERNS, boards(), lambda p: (tables or {}).get(p), classify_one,
                     CHIPS, kernels)
    return res, calls


class AnswerTest(unittest.TestCase):
    def test_nothing_relevant_is_not_due(self):
        res, calls = run(["docs/x.md", "tools/bench/selection_regret.py", "src/product/a.cpp"])
        self.assertFalse(res["due"])
        self.assertEqual(res["boards"], {})
        self.assertEqual(calls, [])

    def test_a_route_table_moves_its_own_architecture(self):
        res, calls = run(["routes/ad102.json"], {"routes/ad102.json": table("ad102")})
        self.assertEqual(res["boards"], {"rtx4090": ["routes/ad102.json"]})
        self.assertEqual(calls, [])  # a readable table is not sent to affected_arches
        res, _ = run(["routes/ga106.json"], {"routes/ga106.json": table("ga106:28sm")})
        self.assertEqual(list(res["boards"]), ["rtx3090"])

    def test_a_table_without_a_lab_board_moves_its_major(self):
        res, _ = run(["routes/ga100.json"], {"routes/ga100.json": table("ga100")})
        self.assertEqual(sorted(res["boards"]), ["rtx3090", "rtx4090"])
        self.assertIn("sibling", res["reasons"]["routes/ga100.json"])

    def test_an_unreadable_table_goes_to_affected_arches(self):
        res, calls = run(["routes/new.json"], {}, arches=due.aa.ARCHES)
        self.assertEqual(calls, ["routes/new.json"])
        self.assertEqual(len(res["boards"]), 4)

    def test_the_grid_moves_every_board(self):
        res, calls = run([due.GRID])
        self.assertEqual(len(res["boards"]), 4)
        self.assertEqual(calls, [])

    def test_the_job_moves_every_board(self):
        res, calls = run([due.JOB])
        self.assertEqual(len(res["boards"]), 4)
        self.assertEqual(calls, [])
        self.assertIn("job", res["reasons"][due.JOB])

    def test_affected_arches_narrows_only_by_arch(self):
        res, _ = run(["src/ops/generic/generic_cost.cpp"], arches=["sm_89"])
        self.assertEqual(list(res["boards"]), ["rtx4090"])
        # host code (no arch) is every board's selector
        res, _ = run(["src/ops/generic/generic_cost.cpp"], arches=[])
        self.assertEqual(len(res["boards"]), 4)
        self.assertIn("host code", res["reasons"]["src/ops/generic/generic_cost.cpp"])

    def test_kernel_files_trigger(self):
        kernels = {"src/ops/generic/ggml_mmq/mmq.cu", "src/ops/common/mma.cuh"}
        res, calls = run(["src/ops/generic/ggml_mmq/mmq.cu", "src/ops/common/mma.cuh",
                          "src/ops/common/unused.cuh"], arches=["sm_120a"], kernels=kernels)
        self.assertEqual(list(res["boards"]), ["rtx5090"])
        self.assertEqual(calls, ["src/ops/common/mma.cuh", "src/ops/generic/ggml_mmq/mmq.cu"])
        self.assertEqual(res["triggers"], sorted(kernels))

    def test_each_board_lists_its_triggers(self):
        res, _ = run(["routes/ad102.json", due.GRID], {"routes/ad102.json": table("ad102")})
        self.assertEqual(res["boards"]["rtx4090"], ["routes/ad102.json", due.GRID])
        self.assertEqual(res["boards"]["gv100"], [due.GRID])
        self.assertEqual(res["cards"]["gv100"], "Quadro GV100")

    def test_lab_boards_of_the_committed_reports(self):
        facts = json.loads((ROOT / due.GPU_FACTS).read_text(encoding="utf-8"))
        reports = {p.name: json.loads(p.read_text(encoding="utf-8"))
                   for p in (ROOT / due.REPORTS).glob("*.json")}
        found = due.lab_boards(reports, due.chip_cc(facts))
        self.assertIn("rtx4070ti", found)
        for board, v in found.items():
            with self.subTest(board=board):
                self.assertIn(v["arch"], due.aa.ARCHES)

    def test_table_cc_reads_the_class(self):
        self.assertEqual(due.table_cc(table("ad104:60sm"), CHIPS), 89)
        self.assertEqual(due.table_cc(table("sm_120"), CHIPS), 120)
        self.assertIsNone(due.table_cc(table("zz999"), CHIPS))
        self.assertIsNone(due.table_cc("not json", CHIPS))
        self.assertIsNone(due.table_cc(None, CHIPS))


class RealTreeTest(unittest.TestCase):
    """affected_arches' own classification and include closure, on a small tree."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.saved = os.environ.get(due.aa.STANDS_ENV)
        os.environ[due.aa.STANDS_ENV] = str(ROOT / "tools/audit/stands.example.json")
        files = {
            # an arch-guarded unit: built on the sm_70 tree alone
            "src/ops/sources.cmake": 'if(NINFERNO_BUILD_ARCH STREQUAL "70")\n  volta_c.cu\nendif()\n',
            "src/ops/volta_c.cu": "",
            # host code
            "src/chat/template.cpp": "int f() { return 1; }\n",
            # a generic kernel and the common headers it includes
            "src/ops/generic/k.cu": '#include "../common/mma.cuh"\n',
            "src/ops/common/mma.cuh": '#include "math.cuh"\n',
            "src/ops/common/math.cuh": "",
            "src/ops/common/unused.cuh": "",
        }
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.repo = due.aa.Repo(self.root)

    def tearDown(self):
        if self.saved is None:
            os.environ.pop(due.aa.STANDS_ENV, None)
        else:
            os.environ[due.aa.STANDS_ENV] = self.saved
        self.dir.cleanup()

    def classify_one(self, path):
        return due.aa.classify([path], self.repo)

    def test_a_host_file_beside_an_arch_guarded_one_keeps_every_board(self):
        paths = ["src/chat/template.cpp", "src/ops/volta_c.cu"]
        together = due.aa.classify(paths, self.repo)
        # one call over both: the arch list alone would read "sm_70 only"
        self.assertEqual(together["arches"], ["sm_70"])
        self.assertTrue(together["host"])
        patterns = gate.load_patterns("\n".join(paths))
        res = due.answer(paths, patterns, boards(), lambda p: None, self.classify_one, CHIPS)
        self.assertEqual(sorted(res["boards"]), ["gv100", "rtx3090", "rtx4090", "rtx5090"])
        self.assertEqual(res["boards"]["gv100"], sorted(paths))
        self.assertEqual(res["boards"]["rtx4090"], ["src/chat/template.cpp"])
        self.assertIn("host code", res["reasons"]["src/chat/template.cpp"])
        self.assertEqual(res["reasons"]["src/ops/volta_c.cu"], "affected_arches.py: sm_70")

    def test_the_kernel_include_closure_triggers(self):
        kernels = due.kernel_files(self.repo)
        self.assertEqual(kernels, {"src/ops/generic/k.cu", "src/ops/common/mma.cuh",
                                   "src/ops/common/math.cuh"})
        res = due.answer(["src/ops/common/math.cuh", "src/ops/common/unused.cuh"],
                         gate.load_patterns(""), boards(), lambda p: None, self.classify_one,
                         CHIPS, kernels)
        self.assertEqual(res["triggers"], ["src/ops/common/math.cuh"])
        self.assertEqual(len(res["boards"]), 4)


if __name__ == "__main__":
    unittest.main()
