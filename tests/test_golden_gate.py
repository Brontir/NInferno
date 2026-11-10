import io
import os
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
                  "routes/tuner/dense_tune.json", "src/core/data/gpu_facts.json",
                  "src/ops/generic/data/kernel_facts/a/b.json", "src/ops/generic/my_route_authorities.cpp",
                  "tests/ops/linear/tests.cmake"]:
            self.assertTrue(self.match(f), f)

    def test_other_files(self):
        for f in ["README.md", "src/ops/generic/generic_gemm_q8.cu", "src/core/data/other.json", "routesx/a.json",
                  "src/product/choice/choice.cpp"]:
            self.assertFalse(self.match(f), f)

    def test_device_files_with_host_decision_functions_are_decision_files(self):
        for f in ["src/ops/generic/generic_tiled.cu", "src/ops/generic/generic_staged.cu",
                  "src/ops/generic/generic_volta_hmma.cu", "src/ops/generic/generic_volta_cutlass.cu",
                  "src/ops/generic/generic_volta_gemv.cu", "src/ops/generic/ggml_mmq/ggml_vec.cu",
                  # the files these functions move to (generic_routes.cpp, ggml_mmq_host.cpp) match the
                  # src/ops/generic/*.cpp glob, whichever change lands first
                  "src/ops/generic/generic_routes.cpp", "src/ops/generic/ggml_mmq_host.cpp"]:
            self.assertTrue(self.match(f), f)

    def test_glob_ends_are_anchored(self):
        rx = gg.glob_to_regex("src/*.h")
        self.assertTrue(rx.match("src/a.h"))
        self.assertFalse(rx.match("src/a.hpp"))  # not end-anchored: would match
        self.assertFalse(rx.match("xsrc/a.h"))  # a search instead of a match: would match
        self.assertFalse(gg.decision_files(["old/routes/a.json"], PATHS))

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
    "src/ops/generic/generic_staged.cuh", "src/ops/generic/generic_tiled.cuh",
    "src/ops/generic/generic_volta_hmma.cuh",  # kernel templates of the listed generic_*.cu files
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
    # the configuration choice (product code): the golden test links core and ops only, so nothing here
    # can move a selector golden
    "src/product/choice/choice.cpp",
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
THIRD_PARTY = ("ggml-cuda/", "nlohmann/", "cutlass/")
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
    ("**/*.cu", "device code: kernels compute the route they are given; what a kernel costs the selector "
                "(registers, spills) is read from the kernel-facts snapshots, which are listed, and a kernel "
                "change that moves them fails ninferno_kernel_facts_test until they are regenerated. A .cu "
                "file that also defines a host function the selector calls is NOT covered by this line: it "
                "is listed in decision_paths.txt, and HostFunctions below finds a new one by name"),
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
NON_CODE_EXT = (".json", ".inc", ".in", ".cmake", ".txt", ".html", ".manifest", ".csv", ".tsv", ".def", ".yaml",
                ".yml", ".toml")
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
        for f in gg.GATE_FILES:
            self.assertTrue(gg.decision_files([f], PATHS), f)
            self.assertTrue((ROOT / f).exists(), f)

    def test_build_inputs_of_the_embedded_kernel_facts_and_the_direct_selector_inputs(self):
        for f in ("src/ops/generic/sources.cmake", "src/ops/generic/embedded_kernel_facts.h.in",
                  "src/ops/generic/data/kernel_facts/sm_120a.txt", "CMakeLists.txt",
                  "src/core/mmq_columns.h", "src/core/measured_boards.h"):
            self.assertTrue(gg.decision_files([f], PATHS), f)

    def test_the_shared_test_target_helpers_and_the_files_wiring_the_golden_in_are_decision_paths(self):
        # ninferno_add_test / ninferno_link_ops shape every test target, the golden producer
        # included; the parent files include them and add the golden's directory to the build
        for f in ("tests/cmake/NInfernoTests.cmake", "tests/cmake/ops_shared_anchor.cpp",
                  "tests/CMakeLists.txt", "tests/ops/tests.cmake"):
            self.assertTrue(gg.decision_files([f], PATHS), f)
            self.assertTrue((ROOT / f).exists(), f)
        # the other test lists under tests/cmake declare unrelated targets
        self.assertFalse(gg.decision_files(["tests/cmake/CoreTests.cmake"], PATHS))

    def test_workflow_reruns_on_label_events_and_on_edit_of_the_description_or_base(self):
        text = (ROOT / ".github/workflows/golden-gate.yml").read_text(encoding="utf-8")
        types = re.search(r"types:\s*\[([^\]]*)\]", text).group(1)
        for t in ("opened", "synchronize", "reopened", "labeled", "unlabeled", "edited"):
            self.assertIn(t, types)


# Host functions in device files. A .cu file in the selector's own directory can define host
# functions the selector calls (is this codec tiled, does this schedule fit, how big is the
# workspace); they are decisions, and the golden moves when they change. They are found by name,
# not by file: a function declared in a listed header under src/ops/generic and defined in a .cu
# file that is not listed fails the test, wherever it lives today and wherever it moves to.
# Launchers (launch_*) run a route and are not decisions.
HOST_FN_SCOPE = gg.glob_to_regex("src/ops/generic/**/*.cu*")  # .cu and .cuh
HOST_FN_HEADERS = gg.glob_to_regex("src/ops/generic/**/*.h")
HOST_FN_DECL = re.compile(r"^[ \t]*(?:\[\[nodiscard\]\]\s*)?(?:inline\s+|constexpr\s+|static\s+)*[\w:<>,*&\s]+?[\s*&]"
                          r"(\w+)\s*\([^;{]*\)\s*(?:const\s*)?(?:noexcept\s*)?;", re.M)
HOST_FN_DEF = re.compile(r"^(?![ \t#/])(?!static\b|template\b|__)[\w:<>,*&\s]*?[\s*&](\w+)\s*\([^;{]*\)\s*"
                         r"(?:const\s*)?(?:noexcept\s*)?\{", re.M)
RUNS_A_ROUTE = "launch_"


def read_text(f, extra):
    return extra[f] if f in extra else (ROOT / f).read_text(encoding="utf-8", errors="replace")


def host_decision_functions(files, extra=None):
    """{unlisted .cu file: sorted names of host functions it defines that a listed header declares}."""
    extra = extra or {}
    declared = set()
    for f in files:
        if f.endswith(".h") and HOST_FN_HEADERS.match(f) and gg.decision_files([f], PATHS):
            declared |= set(HOST_FN_DECL.findall(read_text(f, extra)))
    declared -= {"if", "for", "while", "switch", "return", "sizeof", "decltype"}  # not functions
    out = {}
    for f in sorted(set(files) | set(extra)):
        if not (f.endswith((".cu", ".cuh")) and HOST_FN_SCOPE.match(f)) or gg.decision_files([f], PATHS):
            continue
        found = sorted(n for n in set(HOST_FN_DEF.findall(read_text(f, extra)))
                       if n in declared and not n.startswith(RUNS_A_ROUTE))
        if found:
            out[f] = found
    return out


class HostFunctions(unittest.TestCase):
    def test_no_unlisted_device_file_defines_a_host_function_the_selector_calls(self):
        self.assertEqual(host_decision_functions(tracked()), {},
                         "list these .cu files in tools/audit/decision_paths.txt (or move the host function "
                         "into a listed .cpp file)")

    def test_a_new_host_decision_function_in_a_device_file_is_caught(self):
        files = tracked()
        new = {"src/ops/generic/generic_new_kernel.cu":
               "namespace ninferno {\nbool tiled_fits(int schedule, WeightCodec codec) {\n    return true;\n}\n}\n"}
        self.assertEqual(host_decision_functions(files, new),
                         {"src/ops/generic/generic_new_kernel.cu": ["tiled_fits"]})

    def test_the_finder_sees_the_functions_of_the_listed_files(self):
        # the same finder, with the manifest ignored, must find each function named in the review of #649
        files = tracked()
        saved = list(PATHS)
        PATHS[:] = [p for p in saved if not p[0].endswith(".cu")]
        try:
            found = host_decision_functions(files)
        finally:
            PATHS[:] = saved
        for f, name in [("src/ops/generic/generic_tiled.cu", "tiled_fits"),
                        ("src/ops/generic/generic_tiled.cu", "tiled_supports"),
                        ("src/ops/generic/generic_staged.cu", "staged_fits"),
                        ("src/ops/generic/generic_volta_hmma.cu", "volta_hmma_takes"),
                        ("src/ops/generic/generic_volta_cutlass.cu", "volta_cutlass_workspace_bytes"),
                        ("src/ops/generic/generic_volta_cutlass.cu", "volta_cutlass_chunk_bytes"),
                        ("src/ops/generic/ggml_mmq/ggml_vec.cu", "vec_supported")]:
            self.assertIn(name, found.get(f, []), f"{f}: {name}")

    def test_a_launcher_is_not_a_decision(self):
        new = {"src/ops/generic/generic_new_kernel.cu":
               "namespace ninferno {\nvoid launch_rmma_q4(int schedule, Epilogue e, const GemmArgs& args,\n"
               "                    cudaStream_t stream) {\n}\n}\n"}
        self.assertEqual(host_decision_functions(tracked(), new), {})


class Workflow(unittest.TestCase):
    """The workflow file is the only thing that wires the two scripts to the event: pin the wiring."""

    @classmethod
    def setUpClass(cls):
        raw = (ROOT / ".github/workflows/golden-gate.yml").read_text(encoding="utf-8")
        cls.text = "\n".join(l for l in raw.splitlines() if not l.lstrip().startswith("#"))

    def test_runs_on_pull_request_never_pull_request_target(self):
        self.assertRegex(self.text, r"(?m)^  pull_request:\s*$")
        self.assertNotIn("pull_request_target", self.text)
        self.assertRegex(self.text, r"(?m)^  contents: read\s*$")

    def test_full_history_so_the_merge_base_exists(self):
        self.assertRegex(self.text, r"fetch-depth:\s*0\b")

    def test_gate_gets_the_event_the_base_ref_and_the_manifest(self):
        run = re.search(r'python3 -B "\$gate/golden_gate.py"[^\n]*', self.text).group(0)
        for part in ('--base-ref "$BASE_REF"', '--event "$GITHUB_EVENT_PATH"', "--paths tools/audit/decision_paths.txt"):
            self.assertIn(part, run)
        self.assertIn("BASE_REF: ${{ github.base_ref }}", self.text)

    def test_scripts_come_from_the_base_branch_when_it_has_them(self):
        self.assertIn('gate="$RUNNER_TEMP/gate"', self.text)
        for name in ("golden_gate.py", "golden_diff.py"):
            self.assertIn(f'git show "origin/$BASE_REF:tools/audit/{name}" > "$RUNNER_TEMP/gate/{name}"', self.text)

    def test_the_gate_cannot_fail_open(self):
        self.assertNotIn("|| true", self.text)
        self.assertNotIn("continue-on-error", self.text)
        self.assertNotIn("if: always()", self.text)
        self.assertRegex(self.text, r'(?m)^\s*exit "\$rc"\s*$')  # the gate's own code is the step's code

    def test_exit_2_prints_a_hint(self):
        self.assertRegex(self.text, r'\[ "\$rc" -eq 2 \]')
        self.assertIn("::error::", self.text)
        self.assertIn("a PR of its own", self.text)

    def test_both_test_files_run(self):
        for name in ("tests/test_golden_gate.py", "tests/test_golden_diff.py"):
            self.assertRegex(self.text, r"python3 -B " + re.escape(name))


GOLDEN_BASE = ("# v100 gv100:80sm sm70\n"
               "q4 p0 store 256x2048 speed T=1..16 rmma.m1.n2.w4x1 ws=0\n"
               "q4 p0 store 256x2048 speed T=17..2048 tiled.128x128 ws=0\n")
GOLDEN_MOVED = GOLDEN_BASE.replace("T=17..2048 tiled.128x128", "T=17..2048 tiled.64x128")
GOLDEN_FILE = "tests/ops/linear/selector_golden/full/v100.txt"
MISSING = object()  # gate(tool=MISSING): the diff tool is not there


class EndToEnd(unittest.TestCase):
    """Both scripts run as processes, beside each other, over a small git repository (as in CI)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.repo, cls.scripts = root / "repo", root / "gate"
        cls.repo.mkdir()
        cls.scripts.mkdir()
        for name in ("golden_gate.py", "golden_diff.py"):
            (cls.scripts / name).write_text((ROOT / "tools/audit" / name).read_text(encoding="utf-8"), encoding="utf-8")
        cls.paths = root / "paths.txt"
        cls.paths.write_text("routes/**\n", encoding="utf-8")
        cls.git("init", "-q", "-b", "master")
        cls.git("config", "user.email", "t@example.com")
        cls.git("config", "user.name", "t")
        cls.git("config", "core.autocrlf", "false")
        cls.write({"routes/a.json": "{}\n", "README.md": "x\n", GOLDEN_FILE: GOLDEN_BASE})
        cls.git("add", "-A")
        cls.git("commit", "-q", "-m", "base")
        cls.git("update-ref", "refs/remotes/origin/master", "HEAD")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def git(cls, *args):
        return subprocess.run(["git", *args], cwd=cls.repo, check=True, capture_output=True, text=True).stdout

    @classmethod
    def write(cls, files):
        for rel, text in files.items():
            p = cls.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")

    def gate(self, files, labels=(), tool=None, base="master", remove=()):
        """Commit `files` on a branch off origin/master, run golden_gate.py as a process: (rc, stdout, stderr)."""
        self.git("checkout", "-q", "-B", "pr", "refs/remotes/origin/master")
        self.write(files)
        for rel in remove:
            (self.repo / rel).unlink()
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "change")
        ev = Path(self.tmp.name) / "event.json"
        ev.write_text(json.dumps(event(*labels)), encoding="utf-8")
        tool_path = self.scripts / "golden_diff.py"
        original = tool_path.read_text(encoding="utf-8")
        if tool == MISSING:
            tool_path.unlink()
        elif tool is not None:
            tool_path.write_text(tool, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-B", str(self.scripts / "golden_gate.py"), "--base-ref", base,
                                "--event", str(ev), "--paths", str(self.paths), "--repo", str(self.repo)],
                               capture_output=True, text=True)
        finally:
            tool_path.write_text(original, encoding="utf-8")
        return p.returncode, p.stdout, p.stderr

    def test_decision_change_with_equal_goldens_is_refused_and_the_label_passes_it(self):
        rc, out, _ = self.gate({"routes/a.json": '{"x": 1}\n'})
        self.assertEqual(rc, 1, out)
        self.assertIn("goldens-equal", out)
        rc, out, _ = self.gate({"routes/a.json": '{"x": 1}\n'}, labels=["goldens-equal"])
        self.assertEqual(rc, 0, out)
        self.assertIn("claims", out)

    def test_moved_goldens_pass_without_the_label_and_show_the_summary(self):
        rc, out, _ = self.gate({"routes/a.json": '{"x": 1}\n', GOLDEN_FILE: GOLDEN_MOVED})
        self.assertEqual(rc, 0, out)
        self.assertIn("total: ", out)
        self.assertIn("tiled.64x128", out)

    def test_an_added_board_file_counts_as_moved(self):
        rc, out, _ = self.gate({"routes/a.json": '{"x": 1}\n',
                                "tests/ops/linear/selector_golden/full/v101.txt": GOLDEN_BASE})
        self.assertEqual(rc, 0, out)
        self.assertIn("1 board files added/removed", out)

    def test_a_meta_only_change_counts_as_moved(self):
        rc, out, _ = self.gate({"routes/a.json": '{"x": 1}\n',
                                GOLDEN_FILE: GOLDEN_BASE.replace("# v100 gv100:80sm sm70", "# v100 gv100:80sm sm71")})
        self.assertEqual(rc, 0, out)
        self.assertIn("1 meta changes", out)

    def test_a_removed_board_file_counts_as_moved(self):
        rc, out, _ = self.gate({"routes/a.json": '{"x": 1}\n'}, remove=[GOLDEN_FILE])
        self.assertEqual(rc, 0, out)
        self.assertIn("1 board files added/removed", out)

    def test_no_decision_file_changed_passes_without_running_the_diff(self):
        rc, out, _ = self.gate({"README.md": "y\n"}, tool="raise SystemExit('must not run')\n")
        self.assertEqual(rc, 0, out)
        self.assertIn("nothing to check", out)

    def test_goldens_only_says_so(self):
        rc, out, _ = self.gate({GOLDEN_FILE: GOLDEN_MOVED})
        self.assertEqual(rc, 0, out)
        self.assertIn("reviewer reads the golden diff", out)

    def test_moe_plan_goldens_are_named_and_not_read(self):
        rc, out, _ = self.gate({"routes/a.json": '{"x": 1}\n', "tests/ops/moe_plan_golden/base/a.txt": "x\n"})
        self.assertEqual(rc, 1, out)  # selector goldens are equal: still refused
        self.assertIn("1 MoE plan golden file(s) changed; the gate does not read them", out)
        self.assertIn("MoE plan goldens are not read", out)

    def test_editing_the_gate_itself_is_called_out(self):
        rc, out, _ = self.gate({"tools/audit/golden_gate.py": "# edited\n"})
        self.assertIn("edits the gate itself", out)
        self.assertIn("tools/audit/golden_gate.py", out)

    def test_a_broken_diff_tool_is_exit_2_whatever_its_own_code(self):
        for name, tool in [("syntax error", "def broken(:\n"),
                           ("sys.exit with a message", "import sys\nsys.exit('boom')\n"),
                           ("empty file", ""),
                           ("uncaught exception", "raise RuntimeError('boom')\n"),
                           ("exit 0 with other output", "print('hello')\n")]:
            rc, out, err = self.gate({"routes/a.json": '{"x": 1}\n'}, tool=tool)
            self.assertEqual(rc, 2, f"{name}: {out}{err}")
        rc, _, _ = self.gate({"routes/a.json": '{"x": 1}\n'}, labels=["goldens-equal"], tool="def broken(:\n")
        self.assertEqual(rc, 2)  # the label does not help
        rc, _, _ = self.gate({"routes/a.json": '{"x": 1}\n'}, labels=["goldens-equal"], tool="print('hello')\n")
        self.assertEqual(rc, 2)  # nor does it make other output read as "goldens equal"

    def test_a_hung_diff_tool_is_exit_2(self):
        os.environ["NINFERNO_GOLDEN_DIFF_TIMEOUT"] = "2"
        try:
            rc, out, err = self.gate({"routes/a.json": '{"x": 1}\n'}, tool="import time\ntime.sleep(30)\n")
        finally:
            del os.environ["NINFERNO_GOLDEN_DIFF_TIMEOUT"]
        self.assertEqual(rc, 2, out + err)
        self.assertIn("longer than 2 s", err)

    def test_a_bad_timeout_value_is_exit_2_with_its_name(self):
        for raw in ("abc", "0", "-5", "inf", "nan"):
            os.environ["NINFERNO_GOLDEN_DIFF_TIMEOUT"] = raw
            try:
                rc, out, err = self.gate({"routes/a.json": '{"x": 1}\n'}, tool="print('hello')\n")
            finally:
                del os.environ["NINFERNO_GOLDEN_DIFF_TIMEOUT"]
            self.assertEqual(rc, 2, f"{raw!r}: {out}{err}")
            self.assertIn("NINFERNO_GOLDEN_DIFF_TIMEOUT", err)

    def test_a_missing_diff_tool_is_exit_2(self):
        rc, out, err = self.gate({"routes/a.json": '{"x": 1}\n'}, tool=MISSING)
        self.assertEqual(rc, 2, out + err)

    def test_an_unknown_base_ref_is_exit_2(self):
        rc, out, err = self.gate({"routes/a.json": '{"x": 1}\n'}, base="no-such-branch")
        self.assertEqual(rc, 2, out + err)


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
    "src/ops/generic/generic_gemm_internal.h", "src/ops/generic/ggml_mmq/ggml_mmq.h",
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
        # device files of the generic directory are judged by function (HostFunctions), not by include
        missing = sorted(f for f in reverse_includers(files)
                         if f not in NOT_DECIDING and not gg.decision_files([f], PATHS)
                         and not (f.endswith(".cu") and HOST_FN_SCOPE.match(f)))
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
