#!/usr/bin/env bash
# Fails when decision code can read a clock or the monitor. Observations (timings, step latencies,
# NVML clocks, temperature, power) may be logged and exported, never fed back into a decision
# (docs/maintainer/deterministic-decisions.md, principle 3). The files below choose routes, depth,
# speculation, admission, placement and cache plans; they may not include a clock or timing header
# nor name a clock, a CUDA event timer or NVML. Their wall-time diagnostics are filled in by the
# caller (the ResourceManager stamps a plan's planning_elapsed_ns).
#
# The check reads each file's own text, not what its includes pull in. Decision code outside the
# list that still reads a timing is a known gap of the page above (round_pacing.h: the decode
# wait, spin or block, still follows measured host lag; it cannot change bytes or batches).
#
# Usage: tools/audit/decision_inputs.sh
# Exit:  0 clean; 1 offenders listed.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

decision_files=(
  src/core/host_cpu_quota.h
  src/runtime/engine/context_cache/capture_planner.h
  src/runtime/engine/context_cache/context_portfolio_value.h
  src/runtime/engine/context_cache/materialization_budget.h
  src/runtime/engine/context_cache/materialization_planner.h
  src/runtime/engine/context_cache/resource_search.h
  src/runtime/engine/admission_policy.cpp
  src/runtime/engine/admission_policy.h
  src/runtime/engine/generation_budget.h
  src/runtime/engine/interleave_cost.h
  src/runtime/engine/kv_capacity.cpp
  src/runtime/engine/kv_capacity.h
  src/runtime/engine/scheduler.h
  src/runtime/engine/warmup_widths.cpp
  src/runtime/engine/warmup_widths.h
  src/spec/config.cpp
  src/spec/config.h
  src/spec/load_model.cpp
  src/spec/load_model.h
  src/spec/manifest.cpp
  src/spec/manifest.h
  src/spec/menu.cpp
  src/spec/menu.h
  src/spec/profile.cpp
  src/spec/profile.h
  src/spec/watchdog.h
  src/ops/prefill_routes.cpp
  src/ops/route_table.cpp
  src/ops/candidate_selector/bf16/candidate_selector_path_plan.cpp
  src/ops/candidate_selector/bf16/candidate_selector_path_plan.h
)
# Every planning file of the Qwen3.5 Program.
while IFS= read -r -d '' file; do
  decision_files+=("$file")
done < <(find src/models/qwen3_5/program/planning -type f \( -name '*.h' -o -name '*.cpp' \) -print0 | sort -z)

# Clock and monitor headers, then clock, event-timer and NVML names.
forbidden='#include[[:space:]]*[<"](chrono|ctime|core/clock\.h|core/startup\.h|core/tune_timer\.h|runtime/contract/timing\.h|ui/gpu_stats\.h|nvml\.h)[>"]'
forbidden+='|steady_clock|system_clock|high_resolution_clock|cudaEventElapsedTime|nvml[A-Z]'

bad=0
for file in "${decision_files[@]}"; do
  if [[ ! -f $file ]]; then
    echo "listed decision file is missing: $file"
    bad=1
    continue
  fi
  if hits=$(grep -nE "$forbidden" "$file"); then
    while IFS= read -r hit; do echo "$file:$hit"; done <<< "$hits"
    bad=1
  fi
done
if [[ $bad -ne 0 ]]; then
  echo "decision code reads a clock or the monitor (deterministic-decisions.md, principle 3)"
  exit 1
fi
echo "${#decision_files[@]} decision files read no clock"
