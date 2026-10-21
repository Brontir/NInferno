#!/usr/bin/env bash
# Fails when decision code can read a clock or the monitor. Observations (timings, step latencies,
# NVML clocks, temperature, power) may be logged and exported, never fed back into a decision
# (docs/maintainer/deterministic-decisions.md, principle 3). The files below choose routes, depth,
# speculation, admission, placement and cache plans; they may not include a clock or timing header
# nor name a clock, a timer or a clock read. Their wall-time diagnostics are filled in by the
# monitor side (context_cache/planning_stopwatch.h, which is therefore not listed).
#
# The check reads each listed file's own text, not what its includes pull in: the Qwen3.5 planning
# files reach program_impl.h, whose CUDA event timers time rounds for telemetry. Decision code
# outside the list that still reads a timing is a known gap of the page above:
# models/qwen3_5/program/round_pacing.h and decode.cpp (the decode wait, mask fill order,
# constrained DFlash drafting, the 30 s draft-arrival deadline), structured/request_constraint.cpp
# (measured mask-row time) and ops/kernel/sampling_device.cuh (the mask-release wait bound).
#
# Usage: tools/audit/decision_inputs.sh
# Exit:  0 clean; 1 offenders listed.
set -euo pipefail
shopt -s nullglob
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

decision_files=(
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
  src/ops/prefill_routes.cpp
  src/ops/route_table.cpp
  src/ops/generic/route_table.h
  src/ops/wrapper/candidate_selector.cpp
  src/ops/candidate_selector/bf16/candidate_selector_path_plan.cpp
  src/ops/candidate_selector/bf16/candidate_selector_path_plan.h
  src/ops/common/*route*.h
  src/ops/linear/*/*_dispatch.cpp
  src/ops/linear/*/*_policy.h
  src/ops/softmax_attention/dense/causal_cache/plan_rules.h
  src/ops/softmax_attention/dense/causal_cache/*/*_policy.h
  src/ops/sparse_moe/*/*_plan.cpp
  src/spec/*.h
  src/spec/*.cpp
)
# Every context-cache planning file but the stopwatch (the monitor side), and every planning file
# of the Qwen3.5 Program.
for file in src/runtime/engine/context_cache/*.h src/runtime/engine/context_cache/*.cpp \
  src/models/qwen3_5/program/planning/*.h src/models/qwen3_5/program/planning/*.cpp; do
  [[ $file == */planning_stopwatch.h ]] || decision_files+=("$file")
done

# Clock and monitor headers; clock types and reads (std::chrono clocks, the Program's Clock alias,
# CUDA events, the engine's timing helpers, POSIX, Win32 and device clocks); NVML.
headers='chrono|ctime|time\.h|sys/time\.h|core/clock\.h|core/startup\.h|core/tune_timer\.h'
headers+='|runtime/contract/timing\.h|models/qwen3_5/program/context_clock\.h'
headers+='|context_cache/planning_stopwatch\.h|ui/gpu_stats\.h|nvml\.h'
forbidden="#include[[:space:]]*[<\"]($headers)[>\"]"
forbidden+='|steady_clock|system_clock|high_resolution_clock|Clock::now|CudaEventTimer'
forbidden+='|cudaEventElapsedTime|elapsed_ms\(|elapsed_ns\(|seconds_since|utc_now|clock_gettime'
forbidden+='|QueryPerformanceCounter|GetTickCount|timeGetTime|globaltimer|clock64|[^_[:alnum:]]clock\('
forbidden+='|nvml[A-Z]'

bad=0
for file in "${decision_files[@]}"; do
  if [[ ! -f $file ]]; then
    echo "listed decision file is missing: $file"
    bad=1
    continue
  fi
  # Comment lines are skipped: a comment may name what the monitor measures.
  if hits=$(grep -nE "$forbidden" "$file" | grep -vE '^[0-9]+:[[:space:]]*(//|\*|/\*)'); then
    while IFS= read -r hit; do echo "$file:$hit"; done <<< "$hits"
    bad=1
  fi
done
if [[ $bad -ne 0 ]]; then
  echo "decision code reads a clock or the monitor (deterministic-decisions.md, principle 3)"
  exit 1
fi
echo "${#decision_files[@]} decision files read no clock"
