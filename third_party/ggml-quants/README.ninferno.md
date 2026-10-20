# NInferno ggml block-quant sources

These files come from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`a25c9865fe03c954c93fd755b5d79ae86ba99750`](https://github.com/ggml-org/llama.cpp/commit/a25c9865fe03c954c93fd755b5d79ae86ba99750),
`ggml/include/` and `ggml/src/`, under the MIT license; see [LICENSE](LICENSE). Only the headers
NInferno's code includes are kept.

NInferno compiles ggml's integer tensor-core matrix multiplication (`src/ggml-cuda/mmq.cuh`) in its
own archive, `ninferno_ggml_mmq` (`src/ops/generic/ggml_mmq/`), which ninfer-all's ggml bridge first
showed how to drive without ggml's backend. Products of up to eight tokens use NInferno's own vector
kernel (`ggml_vec.cu`), which reads the grids and block structs from `ggml-common.h`.

## Changes made in NInferno

Their declarations carry `NInferno:` comments, and the git history holds the full diff against the
copy as vendored (commit `e200e9d9a`).

- `src/ggml-cuda/mmq-load-tiles.cuh`, `src/ggml-cuda/mmq.cuh`: every tile loader takes a
  `ggml_planes` argument, so a weight stored with its blocks split into a code plane, a high-bit
  plane and a scale plane (NInferno's `GgmlPlanes` layout) is read in place; a weight kept as whole
  blocks passes no planes and reads as before.
- `src/ggml-cuda/mmq.cuh`: `NINFERNO_MMQ_PDL_ENTRY()` at the entry of the two matrix kernels, empty
  unless the includer defines it, so the kernels can run as programmatic dependents.
- `src/ggml-cuda/common.cuh`, `src/ggml-cuda/vendors/cuda.h`: the cuBLAS includes and helpers are
  left out when `NINFERNO_GGML_NO_CUBLAS` is defined; the matrix kernels call no cuBLAS.
