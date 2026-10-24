#pragma once

// What a start reads of a CUDA device beside its properties, in one place: the Engine's plan
// (runtime/engine/model_instance.cpp) and the plan without the weights
// (product/plan/plan_facts.cpp plan_device_of_cuda) call these. No context is created.

namespace ninferno {

// The GPU is shared through WDDM, whose paging spills an over-full device to host memory instead
// of failing: natively on Windows a card in WDDM mode (not TCC), and under WSL 2 every GPU (it
// reaches the Windows host's through /dev/dxg).
[[nodiscard]] bool paravirtualized_gpu(int device);

// The GPU drives a display: the host's kernel-run watchdog is on (a headless card has it off).
// Throws std::runtime_error when the attribute cannot be read.
[[nodiscard]] bool drives_display(int device);

} // namespace ninferno
