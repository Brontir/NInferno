import io
import re
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import golden_gate as gg

PATHS = gg.load_patterns(gg.PATHS_FILE.read_text(encoding="utf-8"))
ROOT = Path(__file__).resolve().parents[1]


def event(*labels):
    return {"pull_request": {"labels": [{"name": n} for n in labels]}}


def run(changed, diff, labels=()):
    """diff: (rc, text) golden_diff answers; None means it must not be run."""
    calls = []

    def fake(rng):
        calls.append(rng)
        assert diff is not None, "golden_diff must not run"
        return diff

    with tempfile.TemporaryDirectory() as d:
        ev = Path(d) / "event.json"
        ev.write_text(json.dumps(event(*labels)), encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = gg.main(["--base-ref", "master", "--event", str(ev)], run_diff=fake, changed_files=changed)
    return rc, out.getvalue(), err.getvalue(), calls


class PathMatching(unittest.TestCase):
    def match(self, f):
        return bool(gg.decision_files([f], PATHS))

    def test_decision_files(self):
        for f in ["src/ops/generic/generic_cost.cpp", "src/ops/generic/selector_facts.h", "routes/gb202.json",
                  "routes/tuner/dense_tune.json", "src/core/data/gpu_facts.json", "src/product/choice/choice.cpp",
                  "src/ops/generic/data/kernel_facts/a/b.json", "src/ops/generic/my_route_authorities.cpp"]:
            self.assertTrue(self.match(f), f)

    def test_other_files(self):
        for f in ["README.md", "src/ops/generic/generic_tiled.cu", "tests/ops/linear/test_selector_golden.cpp",
                  "src/core/data/other.json", "routesx/a.json"]:
            self.assertFalse(self.match(f), f)

    def test_double_star(self):
        rx = gg.glob_to_regex("a/**/b.txt")
        self.assertTrue(rx.match("a/b.txt"))
        self.assertTrue(rx.match("a/x/y/b.txt"))
        self.assertFalse(gg.glob_to_regex("a/*.txt").match("a/x/b.txt"))

    def test_listed_files_exist(self):
        # a literal entry must name a tracked file, so a rename cannot drop out unseen
        for text, _ in PATHS:
            if not any(c in text for c in "*?"):
                self.assertTrue((ROOT / text).exists(), text)


# Files in the selector's include closure that do not decide (types, I/O, containers).
NOT_DECIDING = {
    "include/ninferno/ops/linear.h", "include/ninferno/ops/moe.h", "include/ninferno/option_enums.h",
    "src/core/arena.h", "src/core/cache_file.h", "src/core/cuda_error.h", "src/core/device.h",
    "src/core/dtype.h", "src/core/file_io.h", "src/core/layout.h", "src/core/per_device.h",
    "src/core/sha256.h", "src/core/sys.h", "src/core/tensor.h", "src/core/tune_timer.h",
    "src/core/units.h", "src/core/weight.h", "src/core/weight_view.h", "src/ops/moe/moe_internal.h",
}
SEEDS = ["src/ops/generic/generic_gemm.cpp", "src/ops/generic/generic_cost.cpp",
         "src/ops/generic/generic_workspace.cpp", "src/ops/generic/generic_coverage.cpp",
         "src/ops/generic/generic_ggml.cpp", "src/ops/generic/generic_tune.cpp",
         "src/ops/generic/route_query.cpp", "src/ops/generic/selector_facts.cpp",
         "src/ops/route_table.cpp", "src/ops/generic/route_authorities.cpp",
         "src/ops/generic/kernel_roofline.h"]
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def include_closure():
    seen, todo = set(), [s for s in SEEDS if (ROOT / s).exists()]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        for name in INCLUDE.findall(text):
            for base in (Path(f).parent, Path("src"), Path("include")):
                cand = (base / name).as_posix()
                if (ROOT / cand).exists():
                    todo.append(cand)
                    break
    return seen


class Closure(unittest.TestCase):
    def test_every_included_file_is_listed_or_named_not_deciding(self):
        closure = include_closure()
        self.assertGreater(len(closure), 20)
        missing = sorted(f for f in closure if f not in NOT_DECIDING and not gg.decision_files([f], PATHS))
        self.assertEqual(missing, [], "add to tools/audit/decision_paths.txt, or to NOT_DECIDING if it decides nothing")


class Labels(unittest.TestCase):
    def test_rename_listing_uses_nul_separated_paths(self):
        self.assertEqual(gg.decision_files(["routes/a b.json"], PATHS), ["routes/a b.json"])

    def test_labels(self):
        self.assertEqual(gg.labels_of(event("a", "goldens-equal")), {"a", "goldens-equal"})
        self.assertEqual(gg.labels_of({}), set())
        self.assertEqual(gg.labels_of({"pull_request": {"labels": None}}), set())


class Gate(unittest.TestCase):
    def test_no_decision_path_skips_diff(self):
        rc, out, _, calls = run(["README.md"], None)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])

    def test_goldens_changed_passes_and_prints_summary(self):
        rc, out, _, calls = run(["routes/gb202.json"], (1, "SUMMARY-LINE\ntotal: 3 decisions changed"))
        self.assertEqual(rc, 0)
        self.assertIn("SUMMARY-LINE", out)
        self.assertEqual(calls, ["origin/master...HEAD"])

    def test_equal_without_label_fails(self):
        rc, out, _, _ = run(["routes/gb202.json"], (0, "no golden changes"))
        self.assertEqual(rc, 1)
        self.assertIn("goldens-equal", out)

    def test_equal_with_label_passes_and_prints_claim(self):
        rc, out, _, _ = run(["routes/gb202.json"], (0, "no golden changes"), labels=["goldens-equal"])
        self.assertEqual(rc, 0)
        self.assertIn("claims", out)

    def test_other_label_does_not_help(self):
        rc, _, _, _ = run(["routes/gb202.json"], (0, "no golden changes"), labels=["enhancement"])
        self.assertEqual(rc, 1)

    def test_golden_diff_failure_is_an_error(self):
        rc, _, err, _ = run(["routes/gb202.json"], (2, "bad golden"))
        self.assertEqual(rc, 2)
        self.assertIn("failed", err)


if __name__ == "__main__":
    unittest.main()
