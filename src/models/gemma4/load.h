#pragma once

// What the family's three loaders share, internal to them: the loading skeleton of every model
// family (models/common/loading.h), with Gemma 4's activation profiles.

#include "models/common/loading.h"
#include "models/gemma4/model.h"

namespace ninferno::models::gemma4::loading {

using models::loading::Demands;
using models::loading::Id;
using models::loading::open_text_component;
using models::loading::Pending;

class Prepare : public models::loading::Prepare {
public:
    // The dense projections run under the profile's mode (activation_mode_of); the native expert
    // bank's prefill route reads recorded_input() whatever the profile.
    Prepare(const Demands& demands, const ArtifactBacking& backing, ActivationProfile profile)
        : models::loading::Prepare(demands, backing, activation_mode_of(profile)) {}
};

} // namespace ninferno::models::gemma4::loading
