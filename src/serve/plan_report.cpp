#include "serve/plan_report.h"

#include "product/logging/pretty_format.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>

namespace ninferno::serve {

namespace {

using Json = nlohmann::json;

std::string gib(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f GiB", static_cast<double>(bytes) / 1073741824.0);
    return text;
}

// "131,072 tokens x 2 contexts (262,144 KV tokens), 4 cached states, chunk 1024".
std::string describe(const ninferno::PlanRungSummary& rung) {
    std::string pool;
    if (rung.max_context != 0) {
        const std::uint64_t context_tokens = (rung.max_context + 63ULL) / 64ULL * 64ULL;
        const std::uint64_t tenths         = rung.kv_tokens * 10ULL / context_tokens;
        pool                               = tenths % 10 == 0 ? std::to_string(tenths / 10)
                                                              : std::to_string(tenths / 10) + "." + std::to_string(tenths % 10);
        pool += tenths == 10 ? " context" : " contexts";
    }
    return product::format_pretty_count(rung.max_context) + " tokens x " + pool + " (" +
           product::format_pretty_count(rung.kv_tokens) + " KV tokens), " +
           std::to_string(rung.cached_device_states) +
           (rung.cached_device_states == 1 ? " cached state" : " cached states") + ", chunk " +
           std::to_string(rung.prefill_chunk);
}

Json rung_json(const ninferno::PlanRungSummary& rung) {
    return Json{{"max_context", rung.max_context},
                {"kv_tokens", rung.kv_tokens},
                {"cached_device_states", rung.cached_device_states},
                {"prefill_chunk", rung.prefill_chunk},
                {"step", rung.step},
                {"survival", rung.survival},
                {"bytes", rung.bytes},
                {"needed_bytes", rung.needed_bytes}};
}

} // namespace

std::vector<OperationalRecord> render_plan(const ninferno::MemorySummary& memory,
                                           bool context_explicit, std::uint32_t max_concurrency) {
    const ninferno::PlanSummary& plan = memory.plan;
    std::vector<OperationalRecord> out;
    if (plan.rung == 0) { return out; }
    std::string line = "plan | rung " + std::to_string(plan.rung) + " of " +
                       std::to_string(plan.rungs) + ": " + describe(plan.chosen) + " for " +
                       std::to_string(max_concurrency) +
                       (max_concurrency == 1 ? " lane" : " lanes") + " (step " + plan.chosen.step +
                       ", reserve " + plan.reserve + ") | free " + gib(plan.free_bytes) +
                       " after " + gib(plan.weight_bytes) + " of weights";
    if (plan.reserve_limited_context != 0) {
        // The memory is there: the class reserve alone keeps the rung above out.
        line += " | rung " + std::to_string(plan.rung - 1) + " fits but for the " +
                gib(plan.reserve_bytes) + " class reserve";
    } else if (plan.rung > 1 && plan.next_shortfall_bytes != 0) {
        line += " | " + gib(plan.next_shortfall_bytes) + " more would serve rung " +
                std::to_string(plan.rung - 1) + ", " + gib(plan.top_shortfall_bytes) + " the top";
    }
    line += " | planned in " + std::to_string(plan.planning_milliseconds) + " ms";
    line += "; the rungs are declared, the free memory picks one";
    out.push_back({.severity = OperationalSeverity::Info, .message = std::move(line)});

    std::vector<std::string> why;
    // An asked context keeps its length: a lower rung there gave up pool contexts, states or
    // prefill width the options asked for.
    if (context_explicit && plan.rung > 1) {
        why.push_back("below the top rung of the asked --max-context " +
                      product::format_pretty_count(plan.chosen.max_context));
    }
    if (plan.chosen.survival) { why.push_back("a survival rung"); }
    if (plan.reserve != "kept" && plan.reserve_bytes > plan.base_margin_bytes) {
        why.push_back("the " + gib(plan.reserve_bytes) + " class reserve is not kept");
    }
    if (plan.reserve_limited_context != 0) {
        why.push_back("the " + gib(plan.reserve_bytes) + " class reserve limits the plan; " +
                      product::format_pretty_count(plan.reserve_limited_context) +
                      " tokens fit without it");
    }
    if (plan.selected_rung != 0 && plan.selected_rung != plan.rung) {
        why.push_back("the kernel warm-up took more than its declared allowance and moved the plan "
                      "from rung " +
                      std::to_string(plan.selected_rung) + " down to rung " +
                      std::to_string(plan.rung));
    }
    if (!why.empty()) {
        std::string warning = "plan | rung " + std::to_string(plan.rung) + ": ";
        for (std::size_t i = 0; i < why.size(); ++i) { warning += (i == 0 ? "" : "; ") + why[i]; }
        for (const std::string& advice : plan.advice) { warning += " | " + advice; }
        out.push_back({.severity = OperationalSeverity::Warning, .message = std::move(warning)});
    }
    return out;
}

Json plan_summary_json(const ninferno::PlanSummary& plan, bool ladder) {
    Json out{{"rung", plan.rung},
             {"rungs", plan.rungs},
             {"selected_rung", plan.selected_rung},
             {"reserve", plan.reserve},
             {"chosen", rung_json(plan.chosen)},
             {"device_total_bytes", plan.device_total_bytes},
             {"weight_bytes", plan.weight_bytes},
             {"free_bytes", plan.free_bytes},
             {"base_margin_bytes", plan.base_margin_bytes},
             {"reserve_bytes", plan.reserve_bytes},
             {"allowance_bytes", plan.allowance_bytes},
             {"top_shortfall_bytes", plan.top_shortfall_bytes},
             {"next_shortfall_bytes", plan.next_shortfall_bytes},
             {"reserve_limited_context", plan.reserve_limited_context},
             {"reserve_limited_kv_tokens", plan.reserve_limited_kv_tokens},
             {"advice", plan.advice}};
    if (ladder) {
        Json rungs = Json::array();
        for (const ninferno::PlanRungSummary& rung : plan.ladder) {
            rungs.push_back(rung_json(rung));
        }
        out["ladder"] = std::move(rungs);
    }
    return out;
}

std::string render_plan_text(const ServePlan& served, std::uint32_t max_concurrency) {
    const auto table = [&](std::ostringstream& out, const ninferno::PlanSummary& plan) {
        out << "device " << gib(plan.device_total_bytes) << " | weights " << gib(plan.weight_bytes)
            << " | free after the weights " << gib(plan.free_bytes) << " | kept beside a rung: "
            << gib(std::max(plan.base_margin_bytes, plan.reserve_bytes)) << " (driver "
            << gib(plan.base_margin_bytes) << ", class reserve " << gib(plan.reserve_bytes)
            << ") + warm-up allowance " << gib(plan.allowance_bytes) << "\n";
        out << "     rung  context      KV tokens  states  chunk  step         bytes        "
               "needs\n";
        for (std::size_t i = 0; i < plan.ladder.size(); ++i) {
            const ninferno::PlanRungSummary& rung = plan.ladder[i];
            const bool chosen                     = plan.rung == i + 1U;
            char row[160];
            std::snprintf(row, sizeof(row), "%s %4zu  %-11s  %-10s %6u  %5u  %-11s %-12s %s\n",
                          chosen ? " >" : "  ", i + 1U,
                          product::format_pretty_count(rung.max_context).c_str(),
                          product::format_pretty_count(rung.kv_tokens).c_str(),
                          rung.cached_device_states, rung.prefill_chunk, rung.step.c_str(),
                          rung.bytes == 0 ? "-" : gib(rung.bytes).c_str(),
                          rung.needed_bytes == 0 ? "cannot be planned"
                                                 : gib(rung.needed_bytes).c_str());
            out << row;
        }
        if (plan.rung == 0) {
            out << "no rung fits\n";
        } else {
            out << "picked: rung " << plan.rung << " of " << plan.rungs << " (reserve "
                << plan.reserve << "): " << describe(plan.chosen) << " for " << max_concurrency
                << (max_concurrency == 1 ? " lane" : " lanes") << "\n";
        }
        for (const std::string& advice : plan.advice) { out << "  - " << advice << "\n"; }
    };
    std::ostringstream out;
    out << "plan | speculation " << served.speculation << "\n";
    if (served.speculation_classes.is_array()) {
        for (const auto& c : served.speculation_classes) {
            out << "plan | speculation class " << c.value("class", std::string()) << ": depth "
                << c.value("depth", 0) << " (" << c.value("basis", std::string()) << ") "
                << c.value("reason", std::string()) << "\n";
        }
    }
    table(out, served.plan);
    if (served.plain) {
        out << "plan | --spec auto would try the profile's fit fallback, shallower drafts and the "
               "model's MTP head before plain decode (not planned here); plain decode:\n";
        table(out, *served.plain);
    }
    out << "the rungs are declared (artifact, options, device facts); the free memory picks the "
           "first that fits, and a real start's kernel warm-up may only move it down\n";
    return out.str();
}

std::string render_plan_json(const ServePlan& served) {
    Json out{{"schema", "ninferno-plan/1"},
             {"speculation", served.speculation},
             {"speculation_classes", served.speculation_classes},
             {"plan", plan_summary_json(served.plan, true)}};
    out["plain"] = served.plain ? plan_summary_json(*served.plain, true) : Json(nullptr);
    return out.dump(2);
}

} // namespace ninferno::serve
