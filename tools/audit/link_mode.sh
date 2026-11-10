#!/usr/bin/env bash
# The shared test tree's link modes, read off the built binaries (NINFERNO_SHARED_OP_TESTS,
# docs/maintainer/build-system.md). <build>/ninferno_link_modes.txt, written at configure, names
# every test executable, app and bench with its mode; this checks each built binary against it:
# - shared (a test on the shared library): needs libninferno_ops_shared.so and the shared CUDA
#   runtime, and defines no CUDA runtime entry point and no function of a member archive (a second
#   copy of their state);
# - static (a test that keeps the product's link) and product (an app or bench): needs neither the
#   library nor the shared runtime, and takes no CUDA runtime entry point from a shared library;
# - library (ninferno_ops_shared): defines the member archives' functions.
# Every ELF executable under <build>/tests must be listed. Prints each binary's mode.
# Usage: link_mode.sh <build directory>
set -u
build=${1:?usage: link_mode.sh <build directory>}
list="$build/ninferno_link_modes.txt"
if [ ! -f "$list" ]; then
  echo "no $list: not a NINFERNO_SHARED_OP_TESTS tree" >&2
  exit 1
fi
for tool in readelf nm; do
  command -v "$tool" > /dev/null || { echo "$tool not found" >&2; exit 1; }
done

# One function of each member archive that holds state or code a test must not copy:
# ninferno_core_host, ninferno_core, ninferno_ops.
anchors='^ninferno::machine_facts\(ninferno::DriverReport const&\)$|^ninferno::device_class\(int\)$|^ninferno::ops::detail::generic::route_query\('

fail=0
bad() {
  echo "FAIL $1: $2"
  fail=1
}
needs() { readelf -d "$1" 2> /dev/null | sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p'; }
defines_cuda_malloc() { nm --defined-only "$1" 2> /dev/null | grep -qE ' [TtWw] cudaMalloc$'; }
dynamic_cuda_malloc() { nm -D --undefined-only "$1" 2> /dev/null | grep -qE ' cudaMalloc$'; }
anchor_hits() { nm -C --defined-only "$1" 2> /dev/null | sed 's/^[0-9a-f]* [A-Za-z] //' | grep -E "$anchors"; }

declare -A listed=()
counts=""
while read -r mode name path; do
  [ -n "${mode:-}" ] || continue
  listed["$path"]=1
  if [ ! -f "$path" ]; then
    bad "$name" "not built ($path)"
    continue
  fi
  needed=$(needs "$path")
  case "$mode" in
    shared)
      grep -qx 'libninferno_ops_shared.so' <<< "$needed" || bad "$name" "does not need libninferno_ops_shared.so"
      grep -q '^libcudart\.so' <<< "$needed" || bad "$name" "does not need the shared CUDA runtime"
      defines_cuda_malloc "$path" && bad "$name" "defines cudaMalloc: a static CUDA runtime beside the shared one"
      hits=$(anchor_hits "$path")
      [ -z "$hits" ] || bad "$name" "defines member functions beside the shared library: $(tr '\n' ' ' <<< "$hits")"
      ;;
    static | product)
      grep -qx 'libninferno_ops_shared.so' <<< "$needed" && bad "$name" "needs libninferno_ops_shared.so"
      grep -q '^libcudart\.so' <<< "$needed" && bad "$name" "needs the shared CUDA runtime"
      dynamic_cuda_malloc "$path" && bad "$name" "takes cudaMalloc from a shared library"
      ;;
    library)
      [ "$(anchor_hits "$path" | wc -l)" -ge 3 ] || bad "$name" "lacks a member archive's functions"
      ;;
    *)
      bad "$name" "unknown mode $mode"
      ;;
  esac
  echo "$mode $name"
  counts="$counts$mode"$'\n'
done < "$list"

# Every test executable is one the configure decided.
while IFS= read -r -d '' f; do
  head -c 4 "$f" 2> /dev/null | grep -q 'ELF' || continue
  [ -n "${listed[$f]:-}" ] || bad "$(basename "$f")" "an executable under tests/ that ninferno_link_modes.txt does not list"
done < <(find "$build/tests" -maxdepth 1 -type f -perm -u+x -print0)

echo "modes: $(sort <<< "$counts" | uniq -c | awk 'NF == 2 {printf "%s %s  ", $2, $1}')"
if [ "$fail" -ne 0 ]; then
  echo "link_mode: FAILED"
  exit 1
fi
echo "link_mode: ok"
