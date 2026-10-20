# Build system

NInferno builds from its source tree with CMake 3.28+ and C++/CUDA 20.
Each build targets one architecture: `120a` (Blackwell, with NVFP4 kernels; `121a` and `120f`
too), `89`, `86`, `80`, `75` or `70`; any other number from 70 on builds the generic engine
(the release archives carry `100` and `90` this way). CUDA 12.8, 13.2 and 13.4 are the validated
toolkits ([building](../building.md)).
Product commands and prerequisites are in the
[README](../../README.md#quick-start); test and measurement workflows live in
[tests](../../tests/README.md) and [benchmarks](../../bench/README.md).

## Configuration

The root `CMakeLists.txt` owns language/toolchain constraints, build options and top-level
composition. Architecture selection and validation happen before `project()` detects CUDA.
The default configuration is Release. Ninja links and archives share the single-slot
`ninferno_link` pool; compilation uses the build command's parallelism.

| Cache option | Default | Scope |
|---|---|---|
| `NINFERNO_BUILD_APPS` | ON | the executables of `apps/` (CLI, server, page, converter, perplexity, dispatcher, family CLIs) |
| `BUILD_TESTING` | OFF | C++ tests and registered Python interoperability tests |
| `NINFERNO_BUILD_BENCHMARKS` | OFF | Op, model, Engine and context-cost benchmarks |
| `NINFERNO_UI_CHAT` | ON | download llama-ui's chat bundle for `ninferno-ui` (see below) |
| `NINFERNO_VOLTA_CUTLASS` | ON | CUTLASS sm_70 GEMM for Volta prefill (`sm_70` trees; fetched) |
| `NINFERNO_CUDA_EMBED_PTX` | OFF | embed each architecture's PTX beside its SASS (JIT on a later GPU) |
| `NINFERNO_CUDA_LINEINFO` | ON in NVFP4 (`sm_120`) trees that embed tree paths, else OFF | compile CUDA with `-lineinfo` (for profiling) |
| `NINFERNO_SOURCE_TREE_PATHS` | ON | embed the source and build tree paths, so a binary run from the tree finds `routes/`, the drafters' rankings and the chat page there; the release scripts turn it off (those files ship beside the binaries) and source paths in CUDA errors are then relative to the tree |
| `NINFERNO_VERSION_STRING` | the project version | the version the executables report |
| `NINFERNO_SHARED_OP_TESTS` | OFF | link the op tests against one shared `ninferno_ops` |

Apps or tests enable the internal product support components: media acquisition, prompt input,
logging and serving. This is one derived condition, not a separate user option. FFmpeg belongs
to model media decoding and remains required when apps are disabled. Curl and OpenSSL's libcrypto
are required in every tree (the converter and the drafter files use them); spdlog is needed only
for product support. A Python 3 interpreter is found when tests are configured. The larger Python
pytest suites and conversion/evaluation scripts run separately from CMake and CTest.

The CUDA versions and Python 3.11 describe the maintained environment, not configuration version
gates.
CMake also discovers FFmpeg without imposing library version floors. Actual language/API support
is exercised by compilation and tests. The libcurl 7.85 minimum has a concrete API basis:
media acquisition uses `CURLOPT_PROTOCOLS_STR` and `CURLOPT_REDIR_PROTOCOLS_STR`, introduced in
[that release](https://curl.se/libcurl/c/CURLOPT_PROTOCOLS_STR.html).

The checked-in `CMakePresets.json` provides:

- `release`: Release product binaries, tests and benchmarks disabled;
- `dev`: Release products, tests and benchmarks enabled;
- matching build presets and a `dev` test preset with failure output enabled.

Both configure presets use `build/`. Switching between them explicitly resets all three build
options; it does not create independent build trees. Build with `cmake --build build -j`.
For a separate configuration, override the binary directory with `-B build-<name>` and use that
directory in subsequent build/test commands.

Keep machine paths in the ignored `CMakeUserPresets.json`. For example, replacing the interpreter
placeholder with the interpreter from the selected environment:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "dev",
      "cacheVariables": {
        "Python3_EXECUTABLE": "/path/to/python3"
      }
    }
  ]
}
```

Configure that environment with `cmake --preset local`. Compiler paths may also be supplied here;
changing compilers requires a fresh build directory. There is no tools option, installed SDK,
package export or configure-time dependency download, except two: CUTLASS for an `sm_70` tree
(`NINFERNO_VOLTA_CUTLASS`, on by default: header-only, fetched with git at tag v4.4.2, or taken
from `FETCHCONTENT_SOURCE_DIR_CUTLASS`), and llama.cpp's prebuilt llama-ui bundle for
`ninferno-ui`'s chat window (`NINFERNO_UI_CHAT`, on by default: pinned by version and SHA-256;
offline, or off, the page has no chat window). `cmake --install` installs the product
executables, with the route tables (`routes/`), the token rankings (`tools/spec/rankings/` and the
default ranking) and the chat window they look up beside themselves in `share/ninferno/`, and
the licenses in `share/licenses/`.

## Targets and dependencies

`cmake/Dependencies.cmake` discovers system CUDA, Threads, FFmpeg and curl (OpenSSL is found by
the converter and speculation components that link it), and
exposes the repository-pinned JSON/HTTP headers and conditional spdlog library. External include
requirements follow their consuming targets. `cmake/NInfernoTargets.cmake` provides private
project includes and the two CUDA archive policies.

`src/CMakeLists.txt` explicitly enters the component directories. Each component's
`CMakeLists.txt` owns its targets, dependency visibility and compile properties:

| Component | Targets |
|---|---|
| `core/` | `ninferno_core` |
| `artifact/` | `ninferno_artifact` |
| `text/` | `ninferno_text`, including bundled utf8proc C source |
| `media/` | `ninferno_media_decode` |
| `ops/` | `ninferno_ops`, `ninferno_nvfp4_non_rdc` |
| `models/` | `ninferno_model_loading`, `ninferno_model_runtime` |
| `runtime/` | `ninferno_runtime_support`, `ninferno_engine` and its `ninferno::engine` alias |
| `product/` | media acquisition, prompt input and logging libraries |
| `serve/` | `ninferno_serve` |
| `spec/` | `ninferno_spec` |
| `convert/` | `ninferno_convert` |
| `chat/` | `ninferno_chat` |
| `structured/` | `ninferno_structured` |
| `ui/` | `ninferno_ui` |

Component declaration order is not the execution dependency graph: CMake resolves named target
links during generation. Model runtime can refer to the runtime support target declared later;
support links core, while Engine links model runtime. The ownership and dependency contracts are
defined by [Engine architecture](engine-architecture.md) and [Op development](op-development.md).

## Explicit source ownership

Small components list their sources directly. Larger components include local `sources.cmake`
manifests which use `target_sources(<owner> PRIVATE ...)`. Included manifests anchor file paths
with `CMAKE_CURRENT_LIST_DIR`; `include()` does not change `CMAKE_CURRENT_SOURCE_DIR`.

- Every implementation under `src/` has one target owner and one explicit registration.
- Ops register simple horizontal implementations in `basic_sources.cmake` and larger semantic
  families in their own manifests. A family also registers its files under `wrapper/`.
- Linear includes one manifest for each of BF16, FP8, NVFP4, Q4, Q5, Q6 and Q8. Each manifest lists
  all its shape filenames and their actual `.cpp`/`.cu` extensions.
- Models separate loading, frontend, execution and Program source lists while retaining the
  distinct immutable-loading and mutable-runtime targets. Checkpoints/recipes do not define
  additional compile owners.
- New files require an explicit manifest edit. New families require an explicit parent include.
  There is no recursive glob, directory discovery or generated shape/dispatch registry.

Adding a source manifest does not require another library. Add a target only for a meaningful
dependency, reuse or compilation boundary. Shape semantics, supported domains and finite
dispatch remain in C++; CMake only selects translation units.

## CUDA compilation boundaries

`ninferno_core` and `ninferno_ops` go through `ninferno_cuda_archive` (`cmake/NInfernoTargets.cmake`):
a generic tree builds them with separable compilation (RDC) and resolves device symbols in the
static archive; a tree with NVFP4 (`120a`, `121a`, `120f`) compiles each translation unit whole,
which measured faster there. No library links a device symbol of another, so either links.

`ninferno_nvfp4_non_rdc` disables separable compilation and device-symbol resolution. It contains
the warp-specialized Linear, LinearSwiGLU and causal-attention NVFP4 sources that depend on
`setmaxnreg` register transfer. Their owning family manifests register these sources into the
non-RDC target; host launchers connect them to the normal Ops. Every CUDA library (the archives
above, `ninferno_ggml_mmq` and the model families) compiles with `-lineinfo` where
`NINFERNO_CUDA_LINEINFO` is on.

Source-list maintenance must preserve language, architecture, RDC mode, device-link ownership
and numerical compiler options. Splitting a manifest does not reduce kernel instantiation work
or justify combining shape translation units or enabling unity builds.

## Consumers and verification

`apps/CMakeLists.txt` explicitly defines every executable. Tests and benchmarks
include their domain registrations without creating new CMake subdirectories, preserving
`build/tests/`, `build/bench/` and CTest working directories. The helpers are local to those
consumers. Tests explicitly name their linked libraries; public-header and pure-host checks keep
their isolated include/link requirements.

Linear, LinearAdd and LinearSwiGLU reuse compiled test support libraries. Oracle options apply
to those libraries themselves as well as the test executables. Test variants, skip return codes,
interop commands and special linker options belong with the affected registrations.

For build-organization changes, compare configured target/source ownership, compile flags,
device links and CTest commands/properties; check the relevant option combinations, build the
affected targets and run their behavioral tests. A shape-source edit should rebuild its own
translation unit plus necessary downstream links. Build-time or inference-speed claims require
measurements at that scope.
