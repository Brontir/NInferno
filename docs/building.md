# Building from source

The [release builds](../README.md#quick-start) need no compiler. Build from source to change the
code, to target a GPU architecture the releases do not carry, or to make a Docker image.

## Build

Requirements: x86-64 Linux (Windows: [Windows](windows.md)), an NVIDIA GPU and driver, a CUDA
toolkit that knows your GPU (CUDA 12.8, 13.2 and 13.4 are the validated ones; `121a` and `120f` need
12.9 or later, and a `70` build needs 12.8, since CUDA 13 no longer compiles for Volta), CMake
3.28+, a C++20 compiler, Ninja, `pkg-config`, FFmpeg development libraries (`libavformat`,
`libavcodec`, `libavutil`, `libswscale`), `libcurl` 7.85+ and OpenSSL's `libcrypto` (on Debian and
Ubuntu: `libssl-dev`). A `70` (V100) build also needs `git` and network access while configuring: it
fetches NVIDIA's header-only CUTLASS (or point `-DFETCHCONTENT_SOURCE_DIR_CUTLASS` at a local
checkout of tag v4.4.2). Configuring also downloads the page's chat window (llama-ui) from Hugging
Face; offline, it warns after the download times out, and `-DNINFERNO_UI_CHAT=OFF` skips it.

Build for your GPU (`120a` RTX 50, `89` RTX 40, `86` RTX 30, `80` A100, `75` RTX 20, `70` V100;
without `-DCMAKE_CUDA_ARCHITECTURES` the tree is `120a`, which runs on RTX 50 cards only):

```bash
git clone https://github.com/Brontir/NInferno.git
cd NInferno
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

The binaries run from `build/apps/`. `cmake --install build --prefix ~/.local` installs them
with the per-GPU route tables and drafter rankings they read from `share/ninferno/`. PDF documents
need pdfium beside them (`tools/release/pdfium.sh`, or `NINFERNO_PDFIUM`; the releases carry it).

## Docker image

```bash
docker build --build-arg CUDA_ARCH=89 --tag ninferno:local .
# V100: --build-arg CUDA_ARCH=70 --build-arg CUDA_VERSION=12.8.1
docker run --rm --gpus '"device=0"' --publish 8080:8080 \
  --volume "$HOME/.cache/huggingface:/home/ubuntu/.cache/huggingface" \
  ninferno:local ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL --host 0.0.0.0
```

The build arguments: `CUDA_ARCH` (default `120a`), `CUDA_VERSION` (the `nvidia/cuda` image, default
`13.4.3`) and `NINFERNO_VERSION` (the version the binaries report, default the project's). The image
holds `ninferno` and `ninferno-serve` with their route tables and drafter rankings, and pdfium.

## Release archives

`tools/release/package.sh [OUT_DIR]` builds the architectures `NINFERNO_ARCHS` lists (default
every generation; `NINFERNO_ARCHS=89` for one) the way the releases are built (no `-lineinfo`,
the CUDA runtime linked statically; `70` with CUDA 12.8, since CUDA 13 no longer compiles for
Volta) and packs one self-contained archive: the
binaries (`ninferno-ui` among them), the shared libraries they load (found beside them), the route
tables, the drafter rankings, the page's chat window and the model list `ninferno-start` offers.
It needs `patchelf` (it points the bundled libraries' run path at their own directory) and network
access (PDFium and the chat page are downloaded). `tools/release/install.sh` installs such an
archive (`NINFERNO_ARCHIVE=FILE`) or downloads the one for the GPU it finds.
