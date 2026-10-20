#!/usr/bin/env bash
# Lists every tagged specialization in the tree: code that holds only for one GPU chip or
# architecture, one memory tier, one model, one weight format, or that works around a defect. The
# tag is
#
#   // NINFERNO-SPECIAL(<axis>:<value>): <why it exists; what would let us remove or generalise it>
#
# with <axis> one of
#   chip        a chip, optionally with its SM variant: chip:gb203, chip:gb202:170sm
#   arch        a whole architecture: arch:sm120
#   mem         a memory range, when a choice truly depends on the tier: mem:>=31gib, mem:<16gib
#   model       a model family or shape: model:gemma4-26b-a4b
#   format      a weight format: format:nvfp4
#   workaround  a temporary fix, with the condition for removing it
# Anything hardware- or model-dependent should first be a runtime device query
# (src/core/device_class.h), then data (a per-GPU-class table or the model's config), then a
# parametric generic path; only what is left carries the tag.
#
# Usage: tools/specializations.sh [axis]      e.g. tools/specializations.sh chip
# Output: file:line<TAB>axis:value<TAB>reason, sorted by axis, then a count per axis.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
axis=${1:-}
cd "$root"
pattern='NINFERNO-SPECIAL\((chip|arch|mem|model|format|workaround):[^)]*\)'
if [[ -n "$axis" ]]; then pattern="NINFERNO-SPECIAL\\(${axis}:[^)]*\\)"; fi
hits=$(grep -rnE "$pattern" src include apps bench tests tools \
         --include='*.h' --include='*.hpp' --include='*.cuh' --include='*.cu' --include='*.cpp' \
         --include='*.cmake' --include='CMakeLists.txt' --include='*.py' 2>/dev/null |
       sed -E 's/^([^:]+:[0-9]+):.*NINFERNO-SPECIAL\(([^)]*)\):?[[:space:]]*(.*)$/\1\t\2\t\3/' |
       sort -t$'\t' -k2,2 -k1,1V || true)
if [[ -z "$hits" ]]; then
  echo "no NINFERNO-SPECIAL tags${axis:+ for axis $axis}"
  exit 0
fi
printf '%s\n' "$hits"
echo
printf '%s\n' "$hits" | cut -f2 | cut -d: -f1 | sort | uniq -c |
  awk '{printf "%s: %d\n", $2, $1}'
