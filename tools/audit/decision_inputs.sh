#!/usr/bin/env bash
# Fails when decision code can read a clock or the monitor. Observations (timings, step latencies,
# NVML clocks, temperature, power) may be logged and exported, never fed back into a decision
# (docs/maintainer/deterministic-decisions.md, principle 3). The files below choose routes, depth,
# speculation, admission, placement and cache plans; they may not include a clock or timing header
# nor name a clock, a timer or a clock read outside comments.
#
# The monitor side of planning is context_cache/planning_stopwatch.h (not listed). Its one user is
# context_cache/resource_manager.h, which may include it and nothing else of the monitor: it
# stamps a plan's wall-time diagnostics after the planner returns.
#
# The check reads each listed file's own text, not what its includes pull in: the Qwen3.5 planning
# files reach program_impl.h, whose CUDA event timers time rounds for telemetry. Decision code
# outside the list that still reads a timing is a known gap of the page above:
# models/qwen3_5/program/round_pacing.h and decode.cpp (the decode wait, mask fill order,
# constrained DFlash drafting, the 30 s draft-arrival deadline), structured/request_constraint.cpp
# (measured mask-row time) and ops/kernel/sampling_device.cuh (the mask-release wait bound).
#
# Every entry must name at least one file: a pattern that matches nothing fails, so a renamed or
# moved decision file cannot drop out of the check unseen. A self-test of planted violations runs
# first.
#
# Usage: tools/audit/decision_inputs.sh
# Exit:  0 clean; 1 offenders listed.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

# One entry per decision file or per directory pattern (relative to the repository root).
decision_patterns=(
  src/core/host_cpu_quota.h
  src/core/machine_facts.cpp
  src/runtime/engine/admission_policy.cpp
  src/runtime/engine/admission_policy.h
  src/runtime/engine/decision_fingerprint.cpp
  src/runtime/engine/decision_fingerprint.h
  src/runtime/engine/generation_budget.h
  src/runtime/engine/interleave_cost.h
  src/runtime/engine/kv_capacity.cpp
  src/runtime/engine/kv_capacity.h
  src/runtime/engine/scheduler.h
  src/runtime/engine/warmup_widths.cpp
  src/runtime/engine/warmup_widths.h
  'src/runtime/engine/context_cache/*.h'
  'src/runtime/engine/context_cache/*.cpp'
  'src/models/qwen3_5/program/planning/*.h'
  'src/models/qwen3_5/program/planning/*.cpp'
  src/ops/prefill_routes.cpp
  src/ops/route_table.cpp
  src/runtime/engine/process_options_lease.h
  src/product/engine_flags.h
  src/ops/generic/route_table.h
  src/ops/wrapper/candidate_selector.cpp
  src/ops/candidate_selector/bf16/candidate_selector_path_plan.cpp
  src/ops/candidate_selector/bf16/candidate_selector_path_plan.h
  'src/ops/common/*route*.h'
  'src/ops/linear/*/*_dispatch.cpp'
  'src/ops/linear/*/*_policy.h'
  src/ops/softmax_attention/dense/causal_cache/plan_rules.h
  'src/ops/softmax_attention/dense/causal_cache/*/*_policy.h'
  'src/ops/sparse_moe/*/*_plan.cpp'
  'src/spec/*.h'
  'src/spec/*.cpp'
)
monitor_file=src/runtime/engine/context_cache/planning_stopwatch.h
monitor_user=src/runtime/engine/context_cache/resource_manager.h

# Clock and monitor headers, matched at the end of any include path; then clock types and reads
# (std::chrono clocks, the Program's Clock alias, CUDA events, the engine's timing helpers, POSIX,
# Win32 and device clocks) and NVML.
clock_headers='chrono|ctime|time\.h|sys/time\.h|core/clock\.h|core/startup\.h|core/tune_timer\.h'
clock_headers+='|runtime/contract/timing\.h|qwen3_5/program/context_clock\.h|ui/gpu_stats\.h|nvml\.h'
monitor_header='context_cache/planning_stopwatch\.h'
include_of() { printf '#include[[:space:]]*[<"]([^">]*/)?(%s)[>"]' "$1"; }
names='steady_clock|system_clock|high_resolution_clock|Clock::now|CudaEventTimer'
names+='|cudaEventElapsedTime|elapsed_ms\(|elapsed_ns\(|seconds_since|utc_now|clock_gettime'
names+='|QueryPerformanceCounter|GetTickCount|timeGetTime|globaltimer|clock64|(^|[^_[:alnum:]])clock\('
names+='|nvml[A-Z]'
forbidden="$(include_of "$clock_headers|$monitor_header")|$names"
forbidden_for_monitor_user="$(include_of "$clock_headers")|$names"

# The file's code lines as "line:text": `//` comment lines and the lines of /* */ blocks are left
# out (a comment may name what the monitor measures); every other line is read, whatever it
# starts with.
code_lines() {
  awk '
    {
      line = $0
      if (in_block) {
        end = index(line, "*/")
        if (end == 0) { next }
        line = substr(line, end + 2)
        in_block = 0
      }
      while ((start = index(line, "/*")) > 0) {
        rest = substr(line, start + 2)
        end = index(rest, "*/")
        if (end == 0) { line = substr(line, 1, start - 1); in_block = 1; break }
        line = substr(line, 1, start - 1) substr(rest, end + 2)
      }
      if (line ~ /^[[:space:]]*\/\//) { next }
      print NR ":" line
    }' "$1"
}

# Prints the offending lines of `file` against `pattern`; returns 1 when there are any.
check_file() {
  local file=$1 pattern=$2 hits
  if hits=$(code_lines "$file" | grep -E "$pattern"); then
    while IFS= read -r hit; do echo "$file:$hit"; done <<< "$hits"
    return 1
  fi
  return 0
}

# Self-test: planted violations are caught, comments and clean code are not.
self_test() {
  local dir
  dir=$(mktemp -d)
  trap 'rm -rf "$dir"' RETURN
  printf '%s\n' '    *out = Clock::now();' > "$dir/deref.h"
  printf '%s\n' '#include "runtime/engine/context_cache/planning_stopwatch.h"' > "$dir/monitor.h"
  printf '%s\n' '#include "src/models/qwen3_5/program/context_clock.h"' > "$dir/clock.h"
  printf '%s\n' 'auto t = clock();' > "$dir/libc.h"
  printf '%s\n' '/* a block' ' * that names steady_clock' ' */ auto x = 1;' 'auto y = GetTickCount();' \
    > "$dir/after_block.h"
  printf '%s\n' '// Clock::now() in a comment' '/* steady_clock */ int a = 0;' '#include <cstdint>' \
    'std::uint64_t clocks = 0; // a word, not a read' '#include "runtime/contract/execution_timing.h"' \
    > "$dir/clean.h"
  local planted
  for planted in deref monitor clock libc after_block; do
    if check_file "$dir/$planted.h" "$forbidden" > /dev/null; then
      echo "self-test: a planted violation passed ($planted)"
      return 1
    fi
  done
  if ! check_file "$dir/clean.h" "$forbidden"; then
    echo "self-test: clean code was flagged"
    return 1
  fi
  if ! check_file "$dir/monitor.h" "$forbidden_for_monitor_user" > /dev/null; then
    echo "self-test: the monitor's own user may not include the stopwatch"
    return 1
  fi
}

self_test

bad=0
decision_files=()
for pattern in "${decision_patterns[@]}"; do
  # shellcheck disable=SC2206 # the pattern is expanded here on purpose
  matches=($pattern)
  if [[ ${#matches[@]} -eq 0 || ! -e ${matches[0]} ]]; then
    echo "decision entry names no file: $pattern"
    bad=1
    continue
  fi
  for file in "${matches[@]}"; do
    [[ $file == "$monitor_file" ]] || decision_files+=("$file")
  done
done

for file in "${decision_files[@]}"; do
  pattern=$forbidden
  [[ $file == "$monitor_user" ]] && pattern=$forbidden_for_monitor_user
  check_file "$file" "$pattern" || bad=1
done
if [[ $bad -ne 0 ]]; then
  echo "decision code reads a clock or the monitor (deterministic-decisions.md, principle 3)"
  exit 1
fi
echo "${#decision_files[@]} decision files read no clock"
