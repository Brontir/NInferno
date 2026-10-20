#!/usr/bin/env bash
# Fails when a tracked file is committed executable (mode 100755) without being a script. Only a
# file whose first line is a #! interpreter line may carry the executable bit: C/C++/CUDA sources
# and headers, CMake, JSON and Markdown never do. The modes checked are the index's (what a commit
# records), so the result is the same on filesystems without an executable bit.
#
# Fix a hit with: git update-index --chmod=-x <file> (and chmod -x <file> on POSIX).
#
# Usage: tools/check_file_modes.sh
# Exit:  0 clean; 1 offenders listed; 77 not a git checkout (skipped).
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
if ! command -v git > /dev/null 2>&1 || ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "not a git checkout: skipped"
  exit 77
fi
bad=0
executables=0
# git ls-files -s: "<mode> <object> <stage>\t<path>", NUL-terminated with -z.
while IFS= read -r -d '' entry; do
  meta=${entry%%$'\t'*}
  path=${entry#*$'\t'}
  read -r mode object _ <<< "$meta"
  [[ $mode == 100755 ]] || continue
  executables=$((executables + 1))
  first=$(git cat-file blob "$object" | head -c 2 || true)
  if [[ $first != '#!' ]]; then
    echo "executable but not a script (no #! line): $path"
    bad=1
  fi
done < <(git ls-files -s -z)
if [[ $bad != 0 ]]; then
  echo "fix with: git update-index --chmod=-x <file>"
  exit 1
fi
echo "file modes clean: $executables executable files, all scripts"
