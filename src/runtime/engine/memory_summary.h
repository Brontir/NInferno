#pragma once

#include "ninferno/types.h"

namespace ninferno::runtime {

// The memory summary of a served instance: the program's own figures plus what startup resolved
// (KV capacity, Host tier, context, kernel warm-up). The caller holds the lock that
// keeps the instance still.
template <class Instance>
[[nodiscard]] MemorySummary summarize_memory(const Instance& instance) {
    MemorySummary out                     = instance.program->memory_summary();
    const auto& resolution                = instance.kv_capacity_resolution;
    out.kv_capacity_mode                  = resolution.mode;
    out.kv_capacity_page_groups           = resolution.main_page_groups;
    out.kv_capacity_max_page_groups       = resolution.maximum_main_page_groups;
    out.minimum_runtime_reservation_bytes = resolution.minimum_runtime_reservation_bytes;
    out.kv_capacity_increment_bytes       = resolution.bytes_per_additional_main_page_group;
    out.runtime_reservation_bytes         = resolution.runtime_reservation_bytes;
    out.available_after_weights_bytes     = resolution.available_after_weights_bytes;
    out.available_after_startup_bytes     = resolution.available_after_startup_bytes;
    out.kv_capacity_headroom_bytes        = resolution.automatic_headroom_bytes;
    out.planned_slack_bytes               = resolution.planned_slack_bytes;
    out.cached_device_states              = resolution.cached_device_states;
    out.cached_device_states_default      = resolution.cached_device_states_default;
    out.cached_device_states_automatic    = resolution.cached_device_states_automatic;
    const auto& host                      = instance.host_tier;
    out.host_tier_automatic               = host.automatic;
    out.host_tier_reduced                 = host.reduced;
    out.host_state_slots_default          = host.default_slots;
    out.host_kv_capacity_default_bytes    = host.default_kv_bytes;
    out.host_tier_budget_bytes            = host.budget_bytes;
    out.max_context_automatic             = instance.context.automatic;
    out.max_context_native                = instance.context.native;
    out.lower_concurrency_fits            = instance.context.lower_concurrency_fits;
    const auto& warm_up                   = instance.kernel_warm_up;
    out.kernel_warm_up                    = warm_up.ran;
    out.kernel_warm_up_prefill_widths     = warm_up.prefill_widths;
    out.kernel_warm_up_widths_run         = warm_up.widths_run;
    out.kernel_warm_up_graph_passes       = warm_up.graph_passes;
    out.kernel_warm_up_seconds            = warm_up.seconds;
    out.kernel_warm_up_lasting_bytes      = warm_up.lasting_bytes;
    out.stack_limit_before_bytes          = warm_up.stack_limit_before_bytes;
    out.stack_limit_bytes                 = warm_up.stack_limit_bytes;
    out.stack_limit_source                = warm_up.stack_limit_source;
    out.host_pin_device_bytes             = warm_up.host_pin_device_bytes;
    out.graph_returned_bytes              = warm_up.graph_returned_bytes;
    return out;
}

} // namespace ninferno::runtime
