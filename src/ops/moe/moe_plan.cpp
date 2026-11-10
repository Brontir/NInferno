// The MoE plan names and schedules (#620): the host part of moe.cpp (no CUDA), which the route table
// parses a band's plan with and the selector's golden reads without a device. The launch and the
// plan's choice stay in moe.cpp.
#include "ninferno/ops/moe.h"

#include "ops/moe/moe_internal.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninferno::ops::detail::moe {

namespace {

constexpr std::array<ScheduleInfo, 9> kSchedules{{
#define NINFERNO_MOE_INFO(I, WM, WK, MT, NT) {WM, WK, MT, NT},
    NINFERNO_MOE_SCHEDULES(NINFERNO_MOE_INFO)
#undef NINFERNO_MOE_INFO
}};

constexpr std::array<R8ScheduleInfo, 4> kR8Schedules{{
#define NINFERNO_MOE_R8_INFO(I, W, R) {W, R},
    NINFERNO_MOE_R8_SCHEDULES(NINFERNO_MOE_R8_INFO)
#undef NINFERNO_MOE_R8_INFO
}};

} // namespace

int schedule_count() { return static_cast<int>(kSchedules.size()); }

ScheduleInfo schedule_info(int index) { return kSchedules.at(static_cast<std::size_t>(index)); }

int r8_schedule_count() { return static_cast<int>(kR8Schedules.size()); }

R8ScheduleInfo r8_schedule_info(int index) {
    return kR8Schedules.at(static_cast<std::size_t>(index));
}

bool r8_expert_format(QType format) {
    switch (format) {
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q3_K:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::MXFP4:
    // The native formats on the codecs generic R8 reads them with (moe_r8_native.cu).
    case QType::Q8_G32_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        return true;
    // NVFP4 experts decode on R8 only without the native NVFP4 kernels (moe_r8_ggml4.cu).
    case QType::NVFP4:
        return !kNativeNvfp4;
    default: return false;
    }
}

// An R8 stage sums a slot's dot products in one order whatever T is, but not in the register
// MMA's: a plan with one is not the prefill profile's.
bool plan_is_token_invariant(const MoePlan& plan) {
    return plan.gate_up_r8 < 0 && plan.down_r8 < 0 &&
           schedule_info(plan.router_schedule).warps_k == 1 &&
           schedule_info(plan.gate_up_schedule).warps_k == 1 &&
           schedule_info(plan.down_schedule).warps_k == 1;
}

int streaming_schedule(std::int32_t k) {
    if (k >= 1024) { return 0; } // 1 x 8 warps, 16 rows
    if (k >= 512) { return 1; }  // 1 x 4
    return 3;                    // 4 x 1
}

// The plan with R8 schedules gate_up_r8 / down_r8 (-1: the stage's register MMA). A call of at most
// kR8Tokens tokens has no job of more slots whatever the job width (a token selects an expert
// once), so an R8 stage adds no job-width constraint.
MoePlan with_r8(MoePlan p, int gate_up_r8, int down_r8) {
    p.gate_up_r8 = gate_up_r8;
    p.down_r8    = down_r8;
    return p;
}

std::string schedule_name(int index) {
    const auto s = schedule_info(index);
    return "rmma.m" + std::to_string(s.m_tiles) + ".n" + std::to_string(s.n_tiles) + ".w" +
           std::to_string(s.warps_m) + "x" + std::to_string(s.warps_k);
}

std::string r8_schedule_name(int index) {
    const auto s = r8_schedule_info(index);
    return "r8.w" + std::to_string(s.warps) + "r" + std::to_string(s.rows);
}

namespace {

// A grouped stage's route: a register-MMA schedule, or (r8) an R8 schedule.
struct StageRoute {
    int schedule = 0;
    bool r8      = false;
};

std::optional<StageRoute> parse_stage(std::string_view name) {
    for (int i = 0; i < schedule_count(); ++i) {
        if (schedule_name(i) == name) { return StageRoute{i, false}; }
    }
    for (int i = 0; i < r8_schedule_count(); ++i) {
        if (r8_schedule_name(i) == name) { return StageRoute{i, true}; }
    }
    return std::nullopt;
}

} // namespace

std::optional<MoePlan> parse_plan(std::string_view name, std::int32_t hidden) {
    // "|a16": a band measured on A16 (register-MMA stages only).
    constexpr std::string_view kA16 = "|a16";
    const bool a16 = name.size() > kA16.size() && name.ends_with(kA16);
    if (a16) { name.remove_suffix(kA16.size()); }
    const auto bar = name.find('|');
    if (bar == std::string_view::npos) { return std::nullopt; }
    const auto gu = parse_stage(name.substr(0, bar));
    const auto dn = parse_stage(name.substr(bar + 1));
    if (!gu || !dn || (a16 && (gu->r8 || dn->r8))) { return std::nullopt; }
    MoePlan p;
    // An R8 stage keeps the streaming schedule's slot (never launched) and adds no job width.
    p.gate_up_schedule = gu->r8 ? 0 : gu->schedule;
    p.down_schedule    = dn->r8 ? 0 : dn->schedule;
    p.job_columns      = 8;
    for (const auto& s : {*gu, *dn}) {
        if (!s.r8) {
            p.job_columns = std::max(p.job_columns, schedule_info(s.schedule).tokens_per_cta());
        }
    }
    p = with_r8(p, gu->r8 ? gu->schedule : -1, dn->r8 ? dn->schedule : -1);
    // The router follows: token-invariant with a token-invariant plan, streaming otherwise.
    const bool invariant = !gu->r8 && !dn->r8 && schedule_info(gu->schedule).warps_k == 1 &&
                           schedule_info(dn->schedule).warps_k == 1;
    p.router_schedule = invariant ? kPrefillSchedule : streaming_schedule(hidden);
    p.a16             = a16;
    return p;
}

} // namespace ninferno::ops::detail::moe
