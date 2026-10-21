#pragma once

// How the server reports the plan ladder's rung (src/runtime/engine/plan_ladder.h): the startup
// lines, the server-start record and the --plan dry run's text and JSON. Pure: formatting only.

#include "serve/generation_service.h"
#include "serve/operational_log.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ninferno::serve {

// The startup lines: "plan | rung k of N: ..." (INFO), and a WARNING when the rung is below what
// the options ask for (an asked --max-context whose pool stepped below every slot's context, a
// survival rung, the class reserve not kept, a rung the warm-up moved down, or the class reserve
// alone keeping the rung above out, where the INFO line says so instead of a shortfall and the
// warning names the override), naming the shortfall in GiB and what would get more.
[[nodiscard]] std::vector<OperationalRecord> render_plan(const ninferno::MemorySummary& memory,
                                                         bool context_explicit,
                                                         std::uint32_t max_concurrency);

// The plan summary as JSON (with every rung when `ladder` and the summary carries them).
[[nodiscard]] nlohmann::json plan_summary_json(const ninferno::PlanSummary& plan, bool ladder);

// The --plan dry run: a table of the ladder with the chosen rung marked, and its advice.
[[nodiscard]] std::string render_plan_text(const ServePlan& plan, std::uint32_t max_concurrency);
// The same as one JSON document ("schema": "ninferno-plan/1").
[[nodiscard]] std::string render_plan_json(const ServePlan& plan);

} // namespace ninferno::serve
