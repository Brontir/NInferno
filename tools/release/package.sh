#!/usr/bin/env bash
# package.sh [OUT_DIR]: build NInferno for every GPU generation and pack the one self-contained
# archive that runs on any x86-64 Linux (WSL 2 included) with an NVIDIA driver, whatever its GPUs:
#
#   ninferno-<version>-linux-x86_64/
#     bin/    ninferno-serve (apps/dispatch): starts the engine for the GPU a command runs on
#             ninferno-serve-sm<tag>: one engine per generation (120a Blackwell RTX/PRO; 100
#             datacenter Blackwell; 90 Hopper; 89 Ada, with PTX for any later GPU; 86 and 80
#             Ampere; 75 Turing; 70 Volta)
#             ninferno-ui (the local page), ninferno-convert, ninferno-start, ninferno-update
#             (installs a newer release, share/ninferno/install.sh)
#     lib/    the shared libraries the binaries link (FFmpeg, curl, OpenSSL, ...), found through
#             the binaries' $ORIGIN/../lib run path; the CUDA runtime is linked statically;
#             libpdfium.so, which ninferno-serve loads for PDF documents
#     share/ninferno/   engines.json (which engine runs where), route tables, drafter rankings,
#                       the page's chat window (llama-ui), the recommendations ninferno-start shows
#
# NINFERNO_ARCHS lists the generations (default "120a 100 90 89 86 80 75 70"); each builds in its own
# tree (NINFERNO_BUILD_ROOT/build-release-<arch>). Volta needs CUDA 12.8 (CUDA 13 dropped it):
# NINFERNO_CUDA70=/usr/local/cuda-12.8, else the archive has no Volta engine. NINFERNO_PACKAGE_CLEAN=1
# removes each tree once its engine is copied (six trees at once outgrow a small disk);
# NINFERNO_PACKAGE_RESUME=1 keeps the engines an interrupted run already put in OUT_DIR; NINFERNO_JOBS
# caps the compile jobs (nvcc on sm_75's largest kernels needs memory). NINFERNO_VERSION overrides
# the version (default `git describe`, without a leading v), NINFERNO_CMAKE_ARGS adds configure
# arguments and NINFERNO_PACKAGE_TARGETS more targets for the first tree. The C library, the
# dynamic loader and the NVIDIA driver libraries stay the user's; build on the oldest
# distribution the archive should support (the glibc it needs is the build machine's).
set -euo pipefail
OUT=$(realpath -m "${1:-dist}")
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VERSION=${NINFERNO_VERSION:-$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo dev)}
VERSION=${VERSION#v}  # the manifest's and the installers' spelling: 0.1.0, not v0.1.0
ARCHS=${NINFERNO_ARCHS:-120a 100 90 89 86 80 75 70}
BUILD_ROOT=${NINFERNO_BUILD_ROOT:-$ROOT}
NAME=ninferno-$VERSION-linux-x86_64
STAGE=$OUT/$NAME
ARCHIVE=ninferno-linux-x86_64.tar.gz  # a stable name: releases/latest/download/<name>
CUDA70=${NINFERNO_CUDA70:-$(ls -d /usr/local/cuda-12.8 2>/dev/null || true)}
# Checked before hours of building: the run paths below need it.
command -v patchelf > /dev/null || { echo "package.sh: patchelf is needed (the binaries' run path)" >&2; exit 1; }
# Every tree but Volta's builds with this nvcc, not the one CMake finds first (/usr/local/cuda can be
# an older CUDA without the NVFP4 instructions sm_120a needs).
NVCC=${CUDACXX:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
[ -x "$NVCC" ] || { echo "package.sh: nvcc not found (CUDACXX, PATH or /usr/local/cuda/bin)" >&2; exit 1; }
CUDA=$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
# The selector's choices follow the registers the release's nvcc compiled for the register-MMA
# kernels, committed per architecture as kernel-facts snapshots (src/ops/generic/data/kernel_facts/,
# #612; #556): the price reads the snapshot, so the toolkit is pinned to the patch level the
# snapshots record (nvcc's V13.4.92), and each engine built here must equal its snapshot.
FACTS=$ROOT/src/ops/generic/data/kernel_facts
nvcc_version() { "$1" --version | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | head -n1; }
NVCC_V=$(nvcc_version "$NVCC")
PINNED=$(sed -n 's/^nvcc \(.*\)$/\1/p' "$FACTS/sm_120a.txt")
[ -n "$PINNED" ] || { echo "package.sh: $FACTS/sm_120a.txt names no nvcc" >&2; exit 1; }
[ "$NVCC_V" = "$PINNED" ] || { echo "package.sh: nvcc $NVCC_V, the release is pinned to $PINNED (kernel-facts snapshots, #612)" >&2; exit 1; }
for arch in $ARCHS; do
  [ -f "$FACTS/sm_$arch.txt" ] || { echo "package.sh: no kernel-facts snapshot $FACTS/sm_$arch.txt" >&2; exit 1; }
  want=$PINNED
  if [ "$arch" = 70 ]; then want=$([ -n "$CUDA70" ] && nvcc_version "$CUDA70/bin/nvcc"); fi
  have=$(sed -n 's/^nvcc \(.*\)$/\1/p' "$FACTS/sm_$arch.txt")
  [ -z "$want" ] || [ "$have" = "$want" ] || { echo "package.sh: $FACTS/sm_$arch.txt was made with nvcc $have, this release builds sm_$arch with $want" >&2; exit 1; }
done

# The kernels an engine compiled against its snapshot: the linked binary read with cuobjdump
# (tools/kernel_facts/kernel_facts.py check: no GPU), and, where a GPU of the generation is on
# this machine, the running kernels' attributes against it too (`ninferno-serve selector-facts`,
# the probe the selector goldens are priced by; exit 2 without such a GPU: the release goes on,
# the first check having passed). A flag, a source or a compiler build that changed the kernels is
# caught here, not by the toolkit's name.
check_kernel_facts() {
  local arch=$1 bin=$2 nvcc_bin=$3
  python3 -B "$ROOT/tools/kernel_facts/kernel_facts.py" check --binary "$bin" --expected "$FACTS/sm_$arch.txt" \
    --nvcc "$(nvcc_version "$nvcc_bin")" --cuda-bin "$(dirname "$nvcc_bin")" \
    || { echo "package.sh: the sm_$arch kernels differ from their snapshot (kernel-facts snapshots, #612)" >&2; exit 1; }
}
check_selector_facts() {
  local arch=$1 bin=$2 out status
  out=$("$bin" selector-facts) && status=0 || status=$?
  if [ "$status" = 2 ]; then
    echo "package.sh: 'ninferno-serve selector-facts' found no GPU that runs the sm_$arch kernels on this machine; the live check was not run" >&2
    return 0
  fi
  [ "$status" = 0 ] || { echo "$out" >&2; echo "package.sh: 'ninferno-serve selector-facts' failed for sm_$arch (exit $status)" >&2; exit 1; }
}

first=""
engines=""
jobs=()
[ -n "${NINFERNO_JOBS:-}" ] && jobs=(-j "$NINFERNO_JOBS")
for arch in $ARCHS; do
  ptx=false; [ "$arch" = 89 ] && ptx=true
  specific=false; [[ $arch =~ [af]$ ]] && specific=true
  if [ "${NINFERNO_PACKAGE_RESUME:-0}" = 1 ] && [ -x "$STAGE/bin/ninferno-serve-sm$arch" ] && [ -f "$STAGE/bin/ninferno-serve.dispatch" ]; then
    echo "package.sh: sm_$arch already packed (resume)"
    # Its facts again: the tree, a flag or the index may have changed since it was staged.
    arch_nvcc=$NVCC; [ "$arch" = 70 ] && arch_nvcc=$CUDA70/bin/nvcc
    check_kernel_facts "$arch" "$STAGE/bin/ninferno-serve-sm$arch" "$arch_nvcc"
    if [ "$arch" = 86 ] || [ "$arch" = 120a ]; then check_selector_facts "$arch" "$STAGE/bin/ninferno-serve-sm$arch"; fi
    first=${first:-$STAGE}
    engines+="${engines:+, }{\"tag\": \"$arch\", \"arch\": ${arch//[!0-9]/}, \"specific\": $specific, \"ptx\": $ptx}"
    continue
  fi
  extra=(-DCMAKE_CUDA_COMPILER="$NVCC")
  if [ "$arch" = 70 ]; then
    [ -n "$CUDA70" ] || { echo "package.sh: no CUDA 12.8 (NINFERNO_CUDA70): no Volta engine" >&2; continue; }
    extra=(-DCMAKE_CUDA_COMPILER="$CUDA70/bin/nvcc")
  fi
  # Ada's engine carries its PTX too: the driver compiles it, once, for a GPU no engine names.
  build=$BUILD_ROOT/build-release-$arch
  cmake -S "$ROOT" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" -DNINFERNO_CUDA_EMBED_PTX=$([ $ptx = true ] && echo ON || echo OFF) \
    -DNINFERNO_CUDA_LINEINFO=OFF -DNINFERNO_BUILD_APPS=ON -DBUILD_TESTING=OFF \
    -DNINFERNO_BUILD_BENCHMARKS=OFF -DNINFERNO_SOURCE_TREE_PATHS=OFF -DCMAKE_INSTALL_LIBDIR=lib -DNINFERNO_VERSION_STRING="$VERSION" \
    "${extra[@]}" ${NINFERNO_CMAKE_ARGS:-}
  # The first tree builds the programs every machine shares; the others their engine alone.
  if [ -z "$first" ]; then
    targets="ninferno-ui ninferno-serve ninferno-convert ninferno-dispatch ${NINFERNO_PACKAGE_TARGETS:-}"
    # ninferno-connect where the tree has it (libnghttp2 and OpenSSL found): its install rule
    # expects it built.
    if grep -q '^NINFERNO_HAS_CONNECT:INTERNAL=ON$' "$build/CMakeCache.txt"; then
      targets="$targets ninferno-connect"
    fi
  else
    targets=ninferno-serve
  fi
  # shellcheck disable=SC2086  # a list of target names
  cmake --build "$build" --target $targets -- "${jobs[@]}"
  if [ -z "$first" ]; then
    first=$build
    rm -rf "$STAGE"
    cmake --install "$build" --prefix "$STAGE" --strip
    install -m 0755 -s "$build/apps/ninferno-dispatch" "$STAGE/bin/ninferno-serve.dispatch"
  fi
  check_kernel_facts "$arch" "$build/apps/ninferno-serve" "${extra[0]#-DCMAKE_CUDA_COMPILER=}"
  if [ "$arch" = 86 ] || [ "$arch" = 120a ]; then check_selector_facts "$arch" "$build/apps/ninferno-serve"; fi
  install -m 0755 -s "$build/apps/ninferno-serve" "$STAGE/bin/ninferno-serve-sm$arch"
  [ "${NINFERNO_PACKAGE_CLEAN:-0}" = 1 ] && rm -rf "$build"
  engines+="${engines:+, }{\"tag\": \"$arch\", \"arch\": ${arch//[!0-9]/}, \"specific\": $specific, \"ptx\": $ptx}"
done
[ -n "$first" ] || { echo "package.sh: no architecture built" >&2; exit 1; }
mv "$STAGE/bin/ninferno-serve.dispatch" "$STAGE/bin/ninferno-serve"
# The dispatcher and the engines come from their build trees, not `cmake --install`: they get the
# installed programs' run path here, so the libraries in lib/ are found before the system's.
for exe in "$STAGE/bin/ninferno-serve" "$STAGE"/bin/ninferno-serve-sm*; do patchelf --set-rpath '$ORIGIN/../lib' "$exe"; done
echo "{\"engines\": [$engines]}" > "$STAGE/share/ninferno/engines.json"

# The shared libraries the executables load, except the C library family, the loader and the
# driver's, which must come from the user's system.
system_lib='^(linux-vdso|ld-linux|libc|libm|libdl|librt|libpthread|libresolv|libutil|libcuda|libnvidia)[.-]'
mkdir -p "$STAGE/lib"
for exe in "$STAGE"/bin/*; do
  ldd "$exe" 2> /dev/null | awk '$2 == "=>" && $3 ~ /^\// {print $3}'
done | sort -u | while read -r lib; do
  base=$(basename "$lib")
  [[ $base =~ $system_lib ]] && continue
  cp -L "$lib" "$STAGE/lib/$base"
done
for exe in "$STAGE"/bin/ninferno-serve-sm*; do
  if ldd "$exe" | grep -q 'libcudart'; then
    echo "package.sh: $(basename "$exe") links the CUDA runtime dynamically; expected it static" >&2
    exit 1
  fi
done
# pdfium for PDF documents: loaded at run time from lib/, not linked (tools/release/pdfium.sh).
bash "$ROOT/tools/release/pdfium.sh" linux-x64 "$STAGE/lib" "$STAGE/share/licenses/pdfium"
# The bundled libraries resolve each other from lib/ as well.
for lib in "$STAGE"/lib/*.so*; do patchelf --set-rpath '$ORIGIN' "$lib"; done
# Their licenses, from the distribution's packages (Debian and Ubuntu: /usr/share/doc/<package>).
for lib in "$STAGE"/lib/*.so*; do
  package=$(dpkg -S "$(basename "$lib")" 2> /dev/null | head -n 1 | cut -d: -f1) || true
  if [ -n "$package" ] && [ -f "/usr/share/doc/$package/copyright" ]; then
    install -D -m 0644 "/usr/share/doc/$package/copyright" "$STAGE/share/licenses/$package/copyright"
  fi
done
[ -f "$STAGE/share/ninferno/llama-ui/index.html" ] ||
  { echo "package.sh: the chat window (llama-ui) was not fetched at configure time" >&2; exit 1; }

install -m 0755 "$ROOT/tools/release/ninferno-start" "$STAGE/bin/ninferno-start"
# The user measurement kit (tools/kit, docs/maintainer/user-measurement-kit.md) under the source
# tree's own layout, which measure.py resolves its data from; bin/ninferno-kit runs it on this
# release's ninferno-serve.
for f in tools/kit/measure.py tools/kit/validate.py tools/kit/gpu.py tools/kit/models.json \
         tools/kit/hard50_reference.json tools/bench/hard_set.py eval/corpora/hard50/hard50.json \
         examples/cli/messages/long_niah_256k.json src/spec/data/drafter_registry.json; do
  install -D -m 0644 "$ROOT/$f" "$STAGE/share/ninferno/kit/$f"
done
install -m 0755 "$ROOT/tools/release/ninferno-kit" "$STAGE/bin/ninferno-kit"
# ninferno-update installs a newer release with the installer this archive carries.
install -m 0755 "$ROOT/tools/release/ninferno-update" "$STAGE/bin/ninferno-update"
install -m 0644 "$ROOT/tools/release/install.sh" "$STAGE/share/ninferno/install.sh"
python3 "$ROOT/tools/release/recommend.py" --out "$STAGE/share/ninferno/recommend.tsv"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$STAGE/"
# cuda: the toolkit of every engine but Volta's, which CUDA 12.8 builds.
cat > "$STAGE/share/ninferno/release.json" <<JSON
{"version": "$VERSION", "archs": "$ARCHS", "cuda": "$CUDA",
 "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
"$STAGE/bin/ninferno-serve" --version || true

( cd "$OUT" && tar -czf "$ARCHIVE" "$NAME" && sha256sum "$ARCHIVE" > "$ARCHIVE.sha256" )
# The installer is a release file of its own (releases/latest/download/install.sh, NInferno-Setup.exe).
install -m 0644 "$ROOT/tools/release/install.sh" "$OUT/install.sh"
( cd "$OUT" && sha256sum install.sh > install.sh.sha256 )
du -sh "$STAGE" "$OUT/$ARCHIVE"
echo "$OUT/$ARCHIVE"
