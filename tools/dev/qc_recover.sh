# qc_recover.sh: ninja-abort recovery for qc.sh (sourced by it; tests/test_qc_sh.py sources it too).
# Needs from the caller: B (build dir) ROOT W L ARCH JOBS LOW TARGETS LOG; sets RC.
# ninja 1.13 can abort (rc >= 128, "ninja: ...: Assertion `...' failed.") on a stale dyndep graph after sources
# moved between targets; a plain retry reuses the same state and fails again. Recovery runs under the lock taken
# above (fd 9 stays open until exit), inside $B only, escalating only when the retry itself aborted:
#   1. delete the dyndep files (*.dd, *.ddi, CXXDependInfo.json), re-run cmake, retry (objects stay valid);
#   2. also delete .ninja_deps and .ninja_log (dirties every object: ccache absorbs it), re-run cmake, retry;
#   3. move $B aside, remove it, configure fresh, retry.
# A plain compile error, or a cmake error that is not an abort, never resets anything further.
QC_ABORT_RE='^ninja: .*Assertion .* failed\.$'   # ninja's own glibc assert line; a compiler's "In member function ...Assertion..." does not match
QC_PROC=${QC_PROC:-/proc}
qc_aborted() {   # qc_aborted RC LOGFILE : ninja was killed or exited >= 128, or failed printing its own assertion line
  [ "$1" -ge 128 ] || { [ "$1" -ne 0 ] && grep -qE "$QC_ABORT_RE" "$2" 2>/dev/null; }
}
qc_busy() {   # another ninja works in $B (cwd or -C), so a reset would pull files from under it
  local p
  for p in $(pgrep -x ninja 2>/dev/null || true); do
    [ "$(readlink "$QC_PROC/$p/cwd" 2>/dev/null || true)" != "$B" ] || return 0
    if tr '\0' ' ' < "$QC_PROC/$p/cmdline" 2>/dev/null | grep -qF -- "-C $B"; then return 0; fi
  done
  return 1
}
qc_configure() {   # fresh configure of $B (Release, tests on, ccache launchers)
  cmake -S "$W" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DBUILD_TESTING=ON \
    > "$L/qc_cmake.log" 2>&1 9>&-
}
qc_regen() {   # ninja build.ninja; 0 ok, 2 ninja aborted, 1 another failure (a real cmake error)
  local g=0
  ( cd "$B" && $LOW ninja -j"$JOBS" build.ninja > "$L/qc_regen.log" 2>&1 9>&- ) || g=$?
  [ $g -eq 0 ] && return 0
  if qc_aborted "$g" "$L/qc_regen.log"; then return 2; fi
  tail -15 "$L/qc_regen.log"; return 1
}
qc_reset() {   # qc_reset STEP(1|2|3) : returns qc_regen's codes (0 ok, 2 aborted, 1 real failure). Refuses (exit 3) while another ninja runs in $B.
  if qc_busy; then echo "qc.sh: another ninja is running in $B; refusing to reset its state, rerun later"; exit 3; fi
  local step=$1
  if [ "$step" -ge 3 ]; then
    [[ "$ARCH" =~ ^[0-9a-z]+$ ]] && [ -n "$ROOT" ] && [ "$(basename "$B")" = "build_qc_$ARCH" ] \
      && [ "$(realpath "$(dirname "$B")")" = "$(realpath "$ROOT")" ] || { echo "  qc: refusing to remove '$B'"; return 1; }
    echo "  qc: ninja aborted again; removing $B and configuring fresh (full rebuild)"
    mv "$B" "$B.del.$$" && rm -rf "$B.del.$$"
    qc_configure || { tail -15 "$L/qc_cmake.log"; return 1; }
    qc_regen; return $?
  fi
  if [ "$step" -eq 1 ]; then
    echo "  qc: ninja aborted; deleting dyndep files (*.dd *.ddi CXXDependInfo.json) in $B, re-running cmake"
    find "$B/" -type f \( -name '*.dd' -o -name '*.ddi' -o -name CXXDependInfo.json \) -delete
  else
    echo "  qc: ninja aborted again; also deleting .ninja_deps and .ninja_log in $B (objects rebuild via ccache), re-running cmake"
    find "$B/" -type f \( -name .ninja_deps -o -name .ninja_log -o -name '*.dd' -o -name '*.ddi' -o -name CXXDependInfo.json \) -delete
  fi
  cmake -S "$W" -B "$B" > "$L/qc_reset_cmake.log" 2>&1 9>&- || { tail -5 "$L/qc_reset_cmake.log"; return 1; }
  qc_regen; return $?
}
qc_try() { RC=0; ( cd "$B" && $LOW ninja -k 0 -j"$JOBS" $TARGETS > "$LOG" 2>&1 9>&- ) || RC=$?; }
qc_build() {   # qc_build : ninja on $TARGETS into $LOG; sets RC. Recovery as described above.
  local step g
  qc_try
  for step in 1 2 3; do
    qc_aborted "$RC" "$LOG" || break
    echo "  ninja aborted (rc=$RC), recovery step $step"; tail -2 "$LOG" | cut -c1-200
    g=0; qc_reset "$step" || g=$?
    if [ $g -eq 1 ]; then RC=1; echo "  qc: reset step $step failed (see logs); not escalating"; return 0; fi
    if [ $g -eq 2 ]; then RC=134; else qc_try; fi
  done
  if qc_aborted "$RC" "$LOG"; then echo "  qc: ninja aborted after a fresh build dir"; fi
}
