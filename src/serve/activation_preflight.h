#pragma once

// ninferno-serve MODEL --activations-preflight MAJOR.MINOR[:0xPCIID]: what each of --activations
// auto, 16, 8 and 4 resolves to on a GPU of that compute capability, before any load (the desktop
// UI's /api/precision asks it, so it never enables a mode the server would refuse). CPU only: the
// B1 resolution (runtime/engine/activation_precision.h) over the artifact's directory, with the
// device features of that compute capability this build runs. Nothing is fetched or converted: a
// hub id or a checkpoint not converted yet is answered "pending" (checked after the download).
//
// One JSON object on stdout:
//   {"model", "cc", "status": "checked" | "pending", "pending": why (pending only),
//    "experts_settle_at_load": the routed experts' counts are the linear path's until the load,
//    "modes": [{"mode": "auto" | "16" | "8" | "4", "refused", "refusal",
//               "resolved": {"4", "8", "16"}, "drafter": {"4", "8", "16"} (an in-artifact drafter),
//               "kept_wider": [...], "notes": [...], "same_as_auto",
//               "faster_than_auto": "8" (auto only: 8 computes some matrices narrower here)}]}

#include <nlohmann/json.hpp>

#include <string>

namespace ninferno::serve {

// `cc`: "MAJOR.MINOR" ("12.0"), or "MAJOR.MINOR:0xPCIID" ("7.5:0x2182") naming the board, whose
// PCI-ID table row says what the compute capability cannot (no tensor cores on TU116 / TU117);
// std::invalid_argument for any other text.
[[nodiscard]] nlohmann::json activation_preflight(const std::string& model, const std::string& cc);

} // namespace ninferno::serve
