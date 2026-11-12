#pragma once

// A named board as the plan without the weights reads it (#615 W4): --plan-device's facts of a
// gpu_facts.json row (product::plan_device_of_board) and the device class the plan builds from
// them (runtime::plan_device_class, which the dry run, product::gpu_key_of and the MoE route query
// read). For the tests that run a capacity function under a named board's substituted class.

#include "core/device_class.h"
#include "core/machine_facts.h"
#include "runtime/engine/plan_dry_run.h"

// The facts of a row that the class reads, as plan_device_of_board sets them: the PCI id, the
// compute capability, the multiprocessors and the driver's view of the VRAM.
inline ninferno::runtime::PlanDeviceFacts named_board_facts(const ninferno::GpuFactsRow& row) {
    ninferno::runtime::PlanDeviceFacts out;
    out.name               = row.name;
    out.source             = "named board " + row.name;
    out.total_bytes        = ninferno::runtime::plan_board_total_bytes(row.vram_gib);
    out.compute_capability = row.major * 10 + row.minor;
    out.pci_device_id      = row.pci_device_id;
    out.sms                = row.sms;
    return out;
}

// The class the plan reads for a named board (runtime::plan_device_class, #829).
inline ninferno::DeviceClass named_board_class(const ninferno::runtime::PlanDeviceFacts& device) {
    return ninferno::runtime::plan_device_class(device);
}
