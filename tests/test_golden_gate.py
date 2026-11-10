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
    "core/arch_registry_rows.inc": "cmake/BoardRegistry.cmake",  # rows of the "archs" array of gpu_facts.json
    # which sm_*.txt snapshots are embedded is decided by sources.cmake (listed too)
    "ops/generic/embedded_kernel_facts.h": "src/ops/generic/embedded_kernel_facts.h.in",
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


# The completeness walk. The include closure above finds the headers the selector compiles in; this
# finds the rest of what a golden depends on: the sources the linked libraries are built from
# (src/core and src/ops, as their CMake lists name them) and the non-code files the build or the
# run reads (CMake files, generator templates, JSON tables). Each candidate is listed in
# decision_paths.txt, or named here with the reason it cannot move a selector golden. A new
# decision file that is in neither place fails with its name: decide, then list it in
# decision_paths.txt or add a line below.
#
# Scope: the golden is tests/ops/linear/selector_golden, made by ninferno_selector_golden_test,
# which links ninferno_core, ninferno_ops and ninferno_ggml_mmq. Other libraries (models, runtime,
# serve, spec, chat, ui, product plan) are not linked into it. Offline tools that write listed data
# (tools/kernel_facts, tools/bench/r8_decode_table.py, route_table_merge.py, moe_tune_to_table.py)
# are run by hand, not by the build; what they write is listed, so a change shows in that file.
EXCLUDED = [
    ("**/*.cu", "device code: it computes the route it is given; what a kernel costs the selector "
                "(registers, spills) is read from the kernel-facts snapshots, which are listed, and a kernel "
                "change that moves them fails ninferno_kernel_facts_test until they are regenerated"),
    ("**/*.cuh", "device code, as .cu"),
    ("src/core/cyclic_kv_cache.cpp", "KV cache state"), ("src/core/paged_kv_cache.cpp", "KV cache state"),
    ("src/core/host_kv_arena.cpp", "KV cache state"), ("src/core/linear_attention_state.cpp", "state"),
    ("src/core/host_memory.cpp", "host memory"), ("src/core/host_worker_pool.cpp", "host threads"),
    ("src/core/decode_graph.cpp", "CUDA graph capture"), ("src/core/debug_poison.cpp", "debug"),
    ("src/core/gdn_replay_records.cpp", "state records"),
    ("src/core/device_traits.cpp", "reads the driver (display, WDDM) for the plan, not the selector"),
    ("src/core/gate_atlas.cpp", "audit table that calls the gate; the gate's rule is in a listed header"),
    ("src/core/kernel_occupancy.cpp", "occupancy for launch code and the kernel-facts test; the selector "
                                      "prices from the snapshots, not from it"),
    ("src/ops/*/*/shapes/*.cpp", "per-shape tables of the per-format dispatch; the generic selector, "
                                 "which makes the golden choice, does not call them"),
    ("src/ops/**/*_dispatch.cpp", "per-format dispatch: it runs a route"),
    ("src/ops/**/*_format.cpp", "weight geometry checks"),
    ("src/ops/**/*plan.cpp", "per-op plans: they ask the selector and act on the answer"),
    ("src/ops/wrapper/*.cpp", "op wrappers"),
    ("src/ops/context_kv_materialize/*.cpp", "not the linear projection"),
    ("src/ops/kv_cache/**", "not the linear projection"),
    ("src/ops/linear_attention/**", "not the linear projection"),
    ("src/ops/linear_topk/*.cpp", "not the linear projection"),
    ("src/ops/rmsnorm_rope/*.cpp", "not the linear projection"),
    ("src/ops/softmax_attention/**", "attention, not the linear projection"),
    ("src/ops/moe/moe_tune.cpp", "load-time MoE tuner (times plans); no selector golden reads it"),
    ("cmake/Dependencies.cmake", "third-party fetch"), ("cmake/GitCommit.cmake", "version string"),
    ("cmake/Windows.cmake", "Windows toolchain"), ("cmake/windows-utf8.manifest", "Windows manifest"),
    ("cmake/NInfernoTargets.cmake", "compile flags (-lineinfo, archive kind); a flag that moves SASS shows "
                                    "in the kernel-facts snapshot test"),
    ("src/CMakeLists.txt", "target wiring"),
    ("src/*/CMakeLists.txt", "build files of libraries the golden test does not link"),
    ("src/models/**", "model programs: not linked into the golden test"),
    ("src/ops/**/sources.cmake", "source lists of the per-op libraries: they run a route"),
    ("src/ops/CMakeLists.txt", "target wiring of ninferno_ops (the selector's own lists are listed)"),
    ("src/ops/basic_sources.cmake", "source list of the elementwise ops"),
    ("src/chat/**", "chat conventions: not linked into the golden test"),
    ("src/spec/**", "speculation data: not linked into the golden test"),
    ("src/ui/**", "web UI: not linked into the golden test"),
]
EXCLUDED_RX = [(g, gg.glob_to_regex(g), why) for g, why in EXCLUDED]
NON_CODE_EXT = (".json", ".inc", ".in", ".cmake", ".txt", ".html", ".manifest")
SOURCE_LIST_ROOTS = ("src/core", "src/ops")
LISTED_SOURCE = re.compile(r'["\s]((?:\$\{CMAKE_CURRENT_LIST_DIR\}/)?[\w/.\-]+\.(?:cpp|cu))\b')


def library_sources(files):
    """The .cpp/.cu files that the CMake lists of src/core and src/ops name."""
    known, out = set(files), set()
    for cm in files:
        base = Path(cm).parent.as_posix()
        if not (cm.startswith(SOURCE_LIST_ROOTS) and (cm.endswith(".cmake") or cm.endswith("CMakeLists.txt"))):
            continue
        text = (ROOT / cm).read_text(encoding="utf-8", errors="replace")
        for name in LISTED_SOURCE.findall(text):
            cand = f"{base}/{name.replace('${CMAKE_CURRENT_LIST_DIR}/', '')}"
            if cand in known:
                out.add(cand)
    return out


def decision_candidates(files):
    """Library sources and non-code build/run inputs that can matter to a golden."""
    cands = set(library_sources(files))
    for f in files:
        if f.startswith("src/ui/third_party/"):
            continue
        in_tree = f.startswith(("src/", "cmake/", "include/")) or f == "CMakeLists.txt"
        if in_tree and (f.endswith(NON_CODE_EXT) or Path(f).name == "CMakeLists.txt"):
            cands.add(f)
    return cands


def unaccounted(files):
    return sorted(f for f in decision_candidates(files)
                  if f not in NOT_DECIDING and not gg.decision_files([f], PATHS)
                  and not any(rx.match(f) for _, rx, _ in EXCLUDED_RX))


class Completeness(unittest.TestCase):
    def test_every_decision_candidate_is_listed_or_excluded_with_a_reason(self):
        missing = unaccounted(tracked())
        self.assertEqual(missing, [], "these files can feed a selector golden and are neither in "
                         "tools/audit/decision_paths.txt nor excluded (with a reason) in EXCLUDED: "
                         + ", ".join(missing))

    def test_a_new_decision_file_is_caught_by_name(self):
        new = ["src/ops/generic/data/new_table.json", "src/core/data/other_facts.json",
               "src/ops/generic/new_table.inc"]
        self.assertEqual(unaccounted(tracked() + new), sorted(new))

    def test_exclusions_match_something_and_name_a_reason(self):
        files = tracked()
        for glob, rx, why in EXCLUDED_RX:
            self.assertTrue(why.strip(), glob)
            self.assertTrue(any(rx.match(f) for f in files), f"{glob} matches no tracked file")

    def test_the_gate_guards_itself(self):
        # a PR that edits the manifest, the gate, its diff tool, its workflow or its tests is a
        # decision-path change: it needs a golden change or the label, and the reviewer sees it
        for f in ("tools/audit/decision_paths.txt", "tools/audit/golden_gate.py", "tools/audit/golden_diff.py",
                  ".github/workflows/golden-gate.yml", "tests/test_golden_gate.py"):
            self.assertTrue(gg.decision_files([f], PATHS), f)

    def test_build_inputs_of_the_embedded_kernel_facts_and_the_direct_selector_inputs(self):
        for f in ("src/ops/generic/sources.cmake", "src/ops/generic/embedded_kernel_facts.h.in",
                  "src/ops/generic/data/kernel_facts/sm_120a.txt", "CMakeLists.txt",
                  "src/core/mmq_columns.h", "src/core/measured_boards.h"):
            self.assertTrue(gg.decision_files([f], PATHS), f)

    def test_workflow_reruns_on_an_edited_description(self):
        text = (ROOT / ".github/workflows/golden-gate.yml").read_text(encoding="utf-8")
        types = re.search(r"types:\s*\[([^\]]*)\]", text).group(1)
        for t in ("opened", "synchronize", "reopened", "labeled", "unlabeled", "edited"):
            self.assertIn(t, types)


class ToolFailuresExitTwo(unittest.TestCase):
    """1 means "the gate refused", 0 "ok"; every tool failure is 2."""

    def gate(self, argv, **kw):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            return gg.main(argv, **kw), err.getvalue()

    def test_unreadable_event_and_manifest_and_git_failure(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "event.json"
            bad.write_text("{not json", encoding="utf-8")
            rc, _ = self.gate(["--base-ref", "master", "--event", str(bad)], changed_files=[])
            self.assertEqual(rc, 2)
            rc, _ = self.gate(["--base-ref", "master", "--event", str(Path(d) / "missing.json")], changed_files=[])
            self.assertEqual(rc, 2)
            rc, _ = self.gate(["--base-ref", "master", "--paths", str(Path(d) / "missing.txt")], changed_files=[])
            self.assertEqual(rc, 2)
            binary = Path(d) / "paths.txt"
            binary.write_bytes(b"\xff\xfe\x00")
            rc, _ = self.gate(["--base-ref", "master", "--paths", str(binary)], changed_files=[])
            self.assertEqual(rc, 2)
            rc, _ = self.gate(["--base-ref", "no-such-branch", "--repo", d], base_list="")  # git diff fails
            self.assertEqual(rc, 2)

    def test_bad_arguments_exit_2(self):
        with self.assertRaises(SystemExit) as cm:
            self.gate([])
        self.assertEqual(cm.exception.code, 2)

    def test_a_traceback_with_exit_1_is_a_tool_failure(self):
        rc, out, err, _ = run(["routes/gb202.json"], (1, "Traceback (most recent call last):\n  ...\nUnicodeDecodeError"))
        self.assertEqual(rc, 2)

    def test_an_unexpected_exception_exits_2(self):
        orig = gg.main

        def boom(*a, **k):
            raise RuntimeError("boom")

        gg.main = boom
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(gg.run(), 2)
        finally:
            gg.main = orig
        self.assertIn("boom", err.getvalue())

    def test_golden_diff_tool_failures_exit_2(self):
        import golden_diff as gd
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a" / "full", Path(d) / "b" / "full"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            (a / "x.txt").write_bytes(b"\xff\xfe\xfd")
            (b / "x.txt").write_bytes(b"\xff\xfe\xfd")
            for argv in ([str(Path(d) / "a"), str(Path(d) / "b")], [str(Path(d) / "nope"), str(Path(d) / "b")]):
                err = io.StringIO()
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    self.assertEqual(gd.main(argv + ["--fail-on-change"]), 2, argv)
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(gd.main(["--range", "a...b", "--repo", str(Path(d) / "no-repo")]), 2)


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
