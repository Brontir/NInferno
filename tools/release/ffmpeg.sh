#!/usr/bin/env bash
# ffmpeg.sh PREFIX: the FFmpeg a Linux release carries, built from source at the version
# ffmpeg.pin names (each tarball checked against the SHA-256 recorded there) under the LGPL 2.1 or
# later: decoders and demuxers only, no encoder, no muxer, no protocol, no GPL or nonfree part.
# It installs into PREFIX:
#   lib/                    libavcodec, libavformat, libavutil, libswscale (shared; dav1d linked
#                           into libavcodec)
#   include/, lib/pkgconfig the headers and .pc files a build finds through PKG_CONFIG_PATH
#   share/licenses/ffmpeg/  the licence texts, BUILD.txt (ffmpeg_check.py: the licence each
#                           library reports, its configure line, its decoders and demuxers), and
#                           source/ (the two tarballs, this script and ffmpeg.pin)
# A second run with the same pin, script and options keeps PREFIX as it is.
#
# Needs a C compiler, make, pkg-config, python3, nasm, and meson with ninja for dav1d.
# NINFERNO_FFMPEG_ARCHIVE and NINFERNO_DAV1D_ARCHIVE name local copies of the tarballs (an offline
# build). NINFERNO_DAV1D_PREFIX links an installed dav1d (its lib/pkgconfig/dav1d.pc) instead of
# building one. NINFERNO_JOBS caps the compile jobs. NINFERNO_FFMPEG_NO_ASM=1 builds without nasm,
# with slow decoders: for tests only, package.sh refuses such a build.
set -euo pipefail
PREFIX=$(realpath -m "${1:?usage: ffmpeg.sh PREFIX}")
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=$HERE/ffmpeg.pin
read -r _ FFMPEG_VERSION FFMPEG_SHA < <(grep '^ffmpeg ' "$PIN")
read -r _ DAV1D_VERSION DAV1D_SHA < <(grep '^dav1d ' "$PIN")
NO_ASM=${NINFERNO_FFMPEG_NO_ASM:-0}
DAV1D_PREFIX=${NINFERNO_DAV1D_PREFIX:-}
JOBS=${NINFERNO_JOBS:-$(nproc)}

# What src/media/decode/decode.cpp needs, nothing more. It hands FFmpeg one in-memory buffer
# through its own AVIOContext, lets the demuxers probe it, and refuses every nested open (a
# playlist's segments, a reference to another file): no protocol is needed. A format left out
# here is one the server refuses ("failed to open media").
#   mov: MP4, MOV, M4V and 3GP (phones, screen recorders, the web), and AVIF images;
#   matroska: MKV and WebM;
#   gif: GIF, animated or not;
#   image_{jpeg,png,webp}_pipe: one JPEG, PNG or WebP image (with GIF, the image formats the
#   OpenAI and Anthropic APIs document, and the PNG pages the PDF reader renders).
DEMUXERS=mov,matroska,gif,image_jpeg_pipe,image_png_pipe,image_webp_pipe
# The codecs those carry: H.264, HEVC, VP8 and VP9; AV1 through libdav1d (FFmpeg's own av1
# decoder decodes only with a hardware accelerator, and a release enables none); MPEG-4 Part 2
# (older phones and cameras; it brings the H.263 decoder with it); Motion JPEG, which decodes
# JPEG; PNG; WebP (it brings VP8); GIF. No audio decoder: decode.cpp decodes the best video stream
# and drops every other packet, so a file's sound track is read past, not decoded.
DECODERS=h264,hevc,vp8,vp9,libdav1d,mpeg4,mjpeg,png,webp,gif
# The image pipes, and the containers for some streams, split their bytes into frames with these.
PARSERS=h264,hevc,vp8,vp9,av1,mpeg4video,mjpeg,png,webp,gif

need() { command -v "$1" > /dev/null || { echo "ffmpeg.sh: $1 is needed ($2)" >&2; exit 1; }; }
need make "the FFmpeg build"
need cc "the FFmpeg build"
need pkg-config "the FFmpeg build finds dav1d"
need python3 "ffmpeg_check.py"
need curl "the source downloads"
[ "$NO_ASM" = 1 ] || need nasm "FFmpeg's and dav1d's x86 assembly; NINFERNO_FFMPEG_NO_ASM=1 builds without, for tests"
if [ -z "$DAV1D_PREFIX" ]; then
  need meson "the dav1d build"
  need ninja "the dav1d build"
fi

# The build is reused only when everything that shapes it is the same.
want=$(cat "$PIN" "$HERE/ffmpeg.sh" "$HERE/ffmpeg_check.py" | sha256sum | cut -d' ' -f1)
want="$want no_asm=$NO_ASM dav1d_prefix=$DAV1D_PREFIX"
stamp=$PREFIX/.ninferno-ffmpeg
if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
  echo "ffmpeg.sh: $PREFIX already holds FFmpeg $FFMPEG_VERSION"
  exit 0
fi
if [ -d "$PREFIX" ] && [ ! -f "$stamp" ] && [ -n "$(ls -A "$PREFIX")" ]; then
  echo "ffmpeg.sh: $PREFIX is not empty and holds no build of ffmpeg.sh" >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fetch() { # name version sha256 url local-copy
  local out=$tmp/$1-$2.tar.xz
  if [ -n "$5" ]; then cp "$5" "$out"; else curl -fsSL -o "$out" "$4"; fi
  echo "$3  $out" | sha256sum -c --quiet - || { echo "ffmpeg.sh: checksum mismatch for $1 $2 (ffmpeg.pin)" >&2; exit 1; }
  tar -xJf "$out" -C "$tmp"
}
fetch ffmpeg "$FFMPEG_VERSION" "$FFMPEG_SHA" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz" \
  "${NINFERNO_FFMPEG_ARCHIVE:-}"
fetch dav1d "$DAV1D_VERSION" "$DAV1D_SHA" \
  "https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz" \
  "${NINFERNO_DAV1D_ARCHIVE:-}"

# dav1d (BSD 2-Clause) as a static, position-independent library, linked into libavcodec: one
# library fewer to bundle, and never the build machine's.
asm=true
[ "$NO_ASM" = 1 ] && asm=false
if [ -z "$DAV1D_PREFIX" ]; then
  DAV1D_PREFIX=$tmp/dav1d
  meson setup "$tmp/dav1d-build" "$tmp/dav1d-$DAV1D_VERSION" --prefix="$DAV1D_PREFIX" --libdir=lib \
    --buildtype=release --default-library=static -Db_staticpic=true -Denable_asm=$asm \
    -Denable_tools=false -Denable_examples=false -Denable_tests=false -Denable_docs=false
  ninja -C "$tmp/dav1d-build" -j "$JOBS" install
fi

args=(
  --prefix="$PREFIX" --libdir="$PREFIX/lib" --shlibdir="$PREFIX/lib" --incdir="$PREFIX/include"
  --enable-shared --disable-static --enable-pic
  # The licence: LGPL 2.1 or later. configure stops on any component that would need more.
  --disable-gpl --disable-nonfree
  # Nothing the build machine happens to have: no VA-API, VDPAU, OpenCL, CUDA, X11, iconv, bzip2
  # or lzma. Threads (frame and slice threading of the decoders) and zlib (PNG, compressed MP4
  # headers, compressed Matroska tracks) are asked for by name.
  --disable-autodetect --enable-pthreads --enable-zlib
  --disable-everything --disable-programs --disable-doc --disable-network --disable-debug
  --disable-avdevice --disable-avfilter --disable-swresample
  --enable-libdav1d --pkg-config-flags=--static
  --enable-decoder="$DECODERS" --enable-demuxer="$DEMUXERS" --enable-parser="$PARSERS"
)
[ "$NO_ASM" = 1 ] && args+=(--disable-x86asm)
( cd "$tmp/ffmpeg-$FFMPEG_VERSION" &&
  PKG_CONFIG_PATH="$DAV1D_PREFIX/lib/pkgconfig" ./configure "${args[@]}" > "$tmp/configure.log" ) ||
  { tail -n 40 "$tmp/configure.log" "$tmp/ffmpeg-$FFMPEG_VERSION/ffbuild/config.log" >&2 || true; echo "ffmpeg.sh: configure failed" >&2; exit 1; }
make -C "$tmp/ffmpeg-$FFMPEG_VERSION" -j "$JOBS" > "$tmp/make.log" 2>&1 ||
  { tail -n 40 "$tmp/make.log" >&2; echo "ffmpeg.sh: the build failed" >&2; exit 1; }
# Installed into a staging copy of PREFIX first: an old build stays until the new one passes.
stage=$tmp/stage$PREFIX
make -C "$tmp/ffmpeg-$FFMPEG_VERSION" install DESTDIR="$tmp/stage" > /dev/null
rm -rf "$stage/share/ffmpeg"  # the API examples

# The licence notes the release carries, with the sources: LGPL 2.1 section 6 asks for the
# library's complete source and the scripts that built it beside the binary.
lic=$stage/share/licenses/ffmpeg
mkdir -p "$lic/source"
cp "$tmp/ffmpeg-$FFMPEG_VERSION/COPYING.LGPLv2.1" "$tmp/ffmpeg-$FFMPEG_VERSION/LICENSE.md" "$lic/"
cp "$tmp/dav1d-$DAV1D_VERSION/COPYING" "$lic/dav1d-COPYING"
cp "$tmp/ffmpeg-$FFMPEG_VERSION.tar.xz" "$tmp/dav1d-$DAV1D_VERSION.tar.xz" "$HERE/ffmpeg.sh" "$PIN" \
  "$HERE/ffmpeg_check.py" "$lic/source/"
cat > "$lic/README" << EOF
FFmpeg $FFMPEG_VERSION (https://ffmpeg.org), under the GNU Lesser General Public License 2.1 or
later (COPYING.LGPLv2.1, LICENSE.md), with dav1d $DAV1D_VERSION (https://code.videolan.org/videolan/dav1d,
BSD 2-Clause, dav1d-COPYING) linked into libavcodec. lib/libavcodec, libavformat, libavutil and
libswscale are this build. It holds decoders and demuxers only; BUILD.txt lists them with the
configure line.

source/ holds the unmodified source tarballs and the scripts that built them (ffmpeg.sh, with
ffmpeg.pin and ffmpeg_check.py): "bash ffmpeg.sh PREFIX", with NINFERNO_FFMPEG_ARCHIVE and
NINFERNO_DAV1D_ARCHIVE naming the tarballs, rebuilds the libraries; put the new ones in lib/ to
replace these.
EOF
python3 "$HERE/ffmpeg_check.py" "$stage/lib" > "$lic/BUILD.txt" ||
  { cat "$lic/BUILD.txt" >&2; echo "ffmpeg.sh: the build is not the LGPL, decoders-only FFmpeg it should be" >&2; exit 1; }
echo "$want" > "$stage/.ninferno-ffmpeg"
rm -rf "$PREFIX"  # empty, or an older build of this script
mkdir -p "$(dirname "$PREFIX")"
mv "$stage" "$PREFIX"
echo "ffmpeg.sh: FFmpeg $FFMPEG_VERSION with dav1d $DAV1D_VERSION -> $PREFIX"
