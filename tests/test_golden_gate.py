import io
import re
import subprocess
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
        for f in ["README.md", "src/ops/generic/generic_tiled.cu", "src/core/data/other.json", "routesx/a.json"]:
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


# Files in the include closure that decide nothing: memory, I/O, hashing, timing, containers.
NOT_DECIDING = {
    "src/core/arena.cu", "src/core/arena.h", "src/core/cache_file.h", "src/core/checked.h",
    "src/core/cuda_error.h", "src/core/device.cu", "src/core/device.h", "src/core/file_io.h",
    "src/core/hex.h", "src/core/per_device.h", "src/core/sha256.cpp", "src/core/sha256.h",
    "src/core/sys.cpp", "src/core/sys.h", "src/core/tensor.cpp", "src/core/tensor.h",
    "src/core/tune_timer.cu", "src/core/tune_timer.h", "src/core/units.h",
    "src/core/weight_view.cpp", "src/core/weight_view.h",
    # device code reached through the listed ggml_mmq.cu / generic_gemm.cpp: it computes, the
    # selector reads none of its constants
    "src/ops/common/e2m1.cuh", "src/ops/common/fp8_e4m3.cuh", "src/ops/common/memory.cuh",
    "src/ops/common/mma.cuh", "src/ops/generic/generic_gemm.cuh", "src/ops/generic/ggml_iq_tables.cuh",
    "src/ops/linear/fp8/fp8_a16_codec.cuh",
    # per-op API headers and helpers pulled in beside the selector: they decide nothing
    "include/ninferno/ops/gemma4_attention.h", "include/ninferno/ops/gemma4_fused.h",
    "include/ninferno/ops/gemma4_moe.h", "include/ninferno/ops/linear_add.h",
    "include/ninferno/ops/rmsnorm.h", "include/ninferno/ops/silu_mul.h", "include/ninferno/ops/sparse_moe.h",
    "src/artifact/file_io.cpp", "src/artifact/file_io.h", "src/artifact/framing.h",  # artifact reading
    "src/artifact/reader.cpp", "src/artifact/reader.h", "src/artifact/source_identity.cpp",
    "src/artifact/source_identity.h",
    "src/artifact/schema.cpp", "src/artifact/schema.h",  # file layout, not format facts
    "src/core/ascii.h", "src/ops/common/math.cuh", "src/ops/common/math.h", "src/ops/common/warp.cuh",
    "src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_simt.cuh", "src/ops/gdn_input_proj/gdn_conv.cuh",
    "src/ops/gdn_input_proj/gdn_projected_conv.cu", "src/ops/gdn_input_proj/gdn_projected_conv.h",
    "src/ops/generic/segmented_gating.cu",  # kernel
    # the per-format dispatcher consumes a route; the generic selector (listed) makes the golden choice
    "src/ops/linear/linear.cpp", "src/ops/linear/linear_registry.h",
    # consumers of a route or a board fact (found by the reverse check): they ask the selector,
    # route_pins or machine_facts and act on the answer; the golden is made in the listed files
    "src/models/gemma4/load_target.cpp", "src/models/qwen3_5/execution/parameters.cpp",  # model load
    "src/ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp",  # per-op plans
    "src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp",
    "src/ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp",
    "src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp",
    "src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp",
    "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp",
    "src/ops/linear/fp8/fp8_dispatch.cpp", "src/ops/linear/nvfp4/nvfp4_dispatch.cpp",  # dispatch
    "src/ops/moe/moe.cpp", "src/ops/wrapper/attn_input_proj.cpp",  # op wrappers
    "src/ops/wrapper/candidate_selector.cpp", "src/ops/wrapper/embedding.cpp",
    "src/ops/wrapper/gdn_input_proj.cpp", "src/ops/wrapper/linear_add.cpp",
    "src/ops/wrapper/linear_swiglu.cpp",
    # plan_facts also restates the MoE W4A8 bank refusal (ExpertRoutes); no selector golden reads it.
    "src/product/plan/plan_facts.cpp", "src/spec/round_cost.cpp",  # plan, spec costs
    "src/runtime/engine/engine.cpp", "src/runtime/engine/model_instance.cpp",  # engine setup
    "src/runtime/engine/warmup_widths.cpp",
}
# Includes that resolve to no tracked file: generated at build time (name -> the generator input,
# which must be listed) or third-party code.
GENERATED = {
    "core/board_registry_rows.inc": "cmake/BoardRegistry.cmake",
    "core/embedded_gpu_facts.h": "src/core/embedded_gpu_facts.h.in",
}
THIRD_PARTY = ("ggml-cuda/", "nlohmann/")
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
SOURCE_EXT = (".h", ".hpp", ".cuh", ".cpp", ".cu", ".cc")


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f]


def include_closure(files):
    """Files reachable from every listed file: its #includes and the .cpp/.cu beside each header."""
    known = set(files)
    seen, todo, unresolved = set(), gg.decision_files(files, PATHS), set()
    todo.append("tests/ops/linear/test_selector_golden.cpp")  # the golden producer
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        if not f.endswith(SOURCE_EXT):
            continue
        stem = Path(f).with_suffix("").as_posix()
        todo += [stem + e for e in SOURCE_EXT if stem + e in known]
        if f.startswith("include/ninferno/"):  # public header -> its implementation under src/
            impl = "src/" + stem[len("include/ninferno/"):]
            todo += [impl + e for e in SOURCE_EXT if impl + e in known]
        for name in INCLUDE.findall((ROOT / f).read_text(encoding="utf-8", errors="replace")):
            for base in (Path(f).parent.as_posix(), "src", "include", "tests", ""):
                cand = (Path(base) / name).as_posix()
                if cand in known:
                    todo.append(cand)
                    break
            else:
                unresolved.add(name)
    return seen, unresolved


class Closure(unittest.TestCase):
    def test_every_reached_file_is_listed_or_named_not_deciding(self):
        closure, _ = include_closure(tracked())
        self.assertGreater(len(closure), 50)
        missing = sorted(f for f in closure if f not in NOT_DECIDING and not gg.decision_files([f], PATHS))
        self.assertEqual(missing, [], "add to tools/audit/decision_paths.txt, or to NOT_DECIDING if it decides nothing")

    def test_not_deciding_files_exist_and_are_unlisted(self):
        files = set(tracked())
        for f in NOT_DECIDING:
            self.assertIn(f, files)
            self.assertFalse(gg.decision_files([f], PATHS), f)

    def test_unresolved_includes_are_generated_listed_or_third_party(self):
        _, unresolved = include_closure(tracked())
        for name in sorted(unresolved):
            if name in GENERATED:
                self.assertTrue(gg.decision_files([GENERATED[name]], PATHS), GENERATED[name])
        for name, gen in GENERATED.items():
            self.assertTrue((ROOT / gen).exists(), gen)
        # a quoted project include nothing resolves would be a hole in the walk
        quoted = set()
        for f in include_closure(tracked())[0]:
            if f.endswith(SOURCE_EXT):
                text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
                quoted |= set(re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.M))
        for name in sorted(quoted & unresolved):
            self.assertTrue(name in GENERATED or name.startswith(THIRD_PARTY), f"unresolved include {name}")


# The reverse check asks of the files that read the selector's own headers: is this source part of
# the decision? Consumers of a route (model programs, per-op dispatch and plan files, kernels, the
# engine) read a decision, they do not move the selector goldens; they are out of its scope.
REVERSE_HEADERS = {
    "src/ops/generic/selector_facts.h", "src/ops/generic/route_query.h", "src/ops/generic/route_table.h",
    "include/ninferno/ops/route_table.h", "src/ops/generic/route_cost_constants.h",
    "src/ops/generic/generic_select_internal.h", "src/ops/generic/format_traits.h",
    "src/ops/generic/generic_gemm.h", "src/ops/route_pins.h", "src/core/mmq_columns.h",
    "src/core/board_registry.h", "src/core/cost_constants.h", "src/core/machine_facts.h",
}
REVERSE_SCOPE = [gg.glob_to_regex(g) for g in ("src/**/*.cpp", "src/**/*.cu")]


def reverse_includers(files):
    """In-scope sources that include one of REVERSE_HEADERS (by path tail)."""
    tails = {}
    for h in REVERSE_HEADERS:
        tails[h.split("/", 1)[1]] = h
        tails[Path(h).name] = h
    out = set()
    for f in files:
        if not any(rx.match(f) for rx in REVERSE_SCOPE):
            continue
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        for name in INCLUDE.findall(text):
            h = tails.get(name)
            if h and (name != Path(h).name or Path(f).parent == Path(h).parent):
                out.add(f)
    return out


class Reverse(unittest.TestCase):
    def test_sources_including_a_listed_header_are_listed_or_not_deciding(self):
        files = tracked()
        missing = sorted(f for f in reverse_includers(files)
                         if f not in NOT_DECIDING and not gg.decision_files([f], PATHS))
        self.assertEqual(missing, [], "list them in decision_paths.txt, or in NOT_DECIDING with a reason")


class Labels(unittest.TestCase):
    def test_changed_files_use_nul_separated_no_rename_listing(self):
        calls = []

        def fake_git(args, cwd):
            calls.append(args)
            return "a b.json\0old.h\0"

        orig = gg.git
        gg.git = fake_git
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = gg.main(["--base-ref", "master", "--paths", str(gg.PATHS_FILE)],
                             run_diff=lambda rng: (0, "no golden changes"))
        finally:
            gg.git = orig
        diff = next(c for c in calls if c[0] == "diff")
        self.assertIn("-z", diff)
        self.assertIn("--no-renames", diff)
        self.assertEqual(rc, 0)  # neither "a b.json" nor "old.h" is on the list

    def test_base_list_is_unioned_so_a_dropped_line_does_not_escape(self):
        head_without = gg.load_patterns("# nothing\n")
        self.assertEqual(gg.decision_files(["routes/x.json"], head_without), [])
        base = gg.base_patterns("master", ".", text="routes/**\n")
        self.assertEqual(gg.decision_files(["routes/x.json"], head_without + base), ["routes/x.json"])
        with tempfile.TemporaryDirectory() as d:
            paths = Path(d) / "paths.txt"
            paths.write_text("# empty head list\n", encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                rc = gg.main(["--base-ref", "master", "--paths", str(paths)],
                             run_diff=lambda rng: (0, "no golden changes"),
                             changed_files=["routes/x.json"], base_list="routes/**\n")
        self.assertEqual(rc, 1)  # still gated, no label

    def test_missing_base_list_is_empty(self):
        orig = gg.git

        def boom(args, cwd):
            raise gg.subprocess.CalledProcessError(128, "git")

        gg.git = boom
        try:
            self.assertEqual(gg.base_patterns("master", "."), [])
        finally:
            gg.git = orig

    def test_path_with_space_still_matches(self):
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
