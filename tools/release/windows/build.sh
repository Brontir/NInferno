#!/usr/bin/env bash
# build.sh VERSION [OUT_DIR]: NInferno-Setup.exe (+ .sha256), cross-compiled on Linux with
# mingw-w64 (apt install g++-mingw-w64-x86-64-posix): one static executable, no runtime DLLs.
set -euo pipefail
VERSION=${1:?usage: build.sh VERSION [OUT_DIR]}
OUT=$(realpath -m "${2:-dist}")
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
CXX=${CXX:-x86_64-w64-mingw32-g++-posix}
WINDRES=${WINDRES:-x86_64-w64-mingw32-windres}

# 0.2.0-rc1 -> 0,2,0,0 for the version resource.
numbers=$(printf '%s' "${VERSION#v}" | sed 's/-.*//' | awk -F. '{printf "%d,%d,%d,%d", $1, $2, $3, $4}')
mkdir -p "$OUT"
python3 "$HERE/make_icon.py" "$WORK/ninferno.ico"
cp "$HERE/setup.manifest" "$WORK/"
"$WINDRES" -I "$WORK" -DNINFERNO_VERSION="\\\"$VERSION\\\"" -DNINFERNO_VERSION_NUMBERS="$numbers" \
  -i "$HERE/setup.rc" -o "$WORK/setup.res.o"
"$CXX" -std=c++20 -O2 -municode -mwindows -static -static-libgcc -static-libstdc++ \
  -DUNICODE -D_UNICODE -DNOMINMAX -D_WIN32_WINNT=0x0A00 -DNINFERNO_VERSION="\"$VERSION\"" \
  -Wall -Wextra -Wno-cast-function-type -Wno-missing-field-initializers \
  "$HERE/main.cpp" "$HERE/engine.cpp" "$HERE/winutil.cpp" "$WORK/setup.res.o" \
  -o "$OUT/NInferno-Setup.exe" \
  -lwinhttp -lbcrypt -lcomctl32 -lshell32 -lole32 -luuid -ladvapi32 -lgdi32 -luser32
x86_64-w64-mingw32-strip "$OUT/NInferno-Setup.exe"
( cd "$OUT" && sha256sum NInferno-Setup.exe > NInferno-Setup.exe.sha256 )
ls -la "$OUT/NInferno-Setup.exe"
