#pragma once

// The relu2 MoE layer's weights, shared by the trunk's loader (load.cpp) and the MTP block's
// (load_mtp.cpp): both build the layer the same way.

#include "models/common/loading.h"

#include "ninferno/ops/moe.h"

#include <string>
#include <vector>

namespace ninferno::models::nemotron_h {

struct MoeIds {
    loading::Id router = 0, selection_bias = 0, shared_up = 0, shared_down = 0;
    std::vector<loading::Id> up, down; // one per routed expert
};

// One complete parent per expert bank (`where` prefixes the error). The routed experts and the
// shared expert each take A8 (R8 at decode) only where all their projections allow it.
ops::MoeWeights prepare_moe(const loading::Prepare& prepare, const MoeIds& ids,
                            const std::string& where);

} // namespace ninferno::models::nemotron_h
