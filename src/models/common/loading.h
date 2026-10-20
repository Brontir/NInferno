#pragma once

// The loading skeleton the dense, Gemma 4 and Nemotron-H families and the DFlash drafters run
// (Qwen 3.5 has its own, qwen3_5/load.cpp), and the pieces it shares:
//
//   artifact::Reader reader(path);
//   config = parse_...(open_text_component(reader, "family ..."));  // then pairing checks
//   loading::Demands demands(reader);                               // bind every parameter
//   backing = demands.materialize(device);                          // upload, bind the views
//   const loading::Prepare prepare(demands, *backing, recorded);    // Op-ready parameters
//
// A family differs only in the config it parses, its pairing checks, and the parameters it
// demands and prepares.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight_view.h"
#include "models/common/weight_use.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/weight_input.h"

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninferno::models {

// Everything a loaded model's prepared weights point into, freed together with the model: the
// materialized artifact, the view of every demanded parameter, and device buffers the loader
// derives from those parameters.
struct ArtifactBacking {
    artifact::MaterializedArtifact materialized;
    // Views are referenced by ops::WeightInput while parameters are prepared; a deque keeps every
    // element's address stable as the bank grows.
    std::deque<WeightView> views;
    std::deque<DeviceBuffer> derived;
};

namespace loading {

// The config of the artifact's text component, where every family keeps its model. `who`
// prefixes the error when there is none.
[[nodiscard]] const artifact::Json& open_text_component(const artifact::Reader& reader,
                                                        const char* who);

struct Pending {
    artifact::ParameterReference reference;
    std::optional<WeightUsePolicy> use; // matrices only
};

// A demand's index: its Pending entry, and after materialization its view.
using Id = std::size_t;

// Collects every demand before materialization: the Binder is consumed by materialize(), so each
// Use's policy and activation divisor has to be read while it is still alive.
class Demands {
public:
    explicit Demands(const artifact::Reader& reader) : reader_(reader), binder_(reader) {}

    Id matrix(const std::string& name, artifact::Shape shape, const std::string& input);
    // `format`: the exact format required, or (nullopt) whichever format the artifact stored.
    Id direct(const std::string& name, artifact::Shape shape, std::optional<QType> format);
    // A direct demand kept in pinned host memory the device reads in place: a gathered table of
    // which each token reads one row, so its bytes stay out of device memory.
    Id mapped(const std::string& name, artifact::Shape shape, std::optional<QType> format);
    float scalar(const std::string& name);

    // The host copy of a direct demand's values, for the loader's own checks.
    [[nodiscard]] artifact::HostValues values(Id id, QType format);

    bool has(const std::string& name) const { return binder_.contains(name); }

    // Consumes the binder: uploads every demanded parameter and binds each one's view at its Id.
    // No demand may follow. `offload`: the routed experts it places in Host groups
    // (artifact/expert_groups.h; MaterializedArtifact::expert_residency).
    [[nodiscard]] std::unique_ptr<ArtifactBacking>
    materialize(DeviceContext& device, const MoeOffloadOptions& offload = {});

    [[nodiscard]] const std::vector<Pending>& items() const noexcept { return items_; }

private:
    const artifact::Reader& reader_;
    artifact::Binder binder_;
    std::vector<Pending> items_;
};

using Projection = ops::SingleProjectionWeight;

class Prepare {
public:
    // `recorded`: every projection keeps the activation policy its Use recorded (W4A4 where the
    // checkpoint was calibrated for it); otherwise every activation stays BF16.
    Prepare(const Demands& demands, const ArtifactBacking& backing, bool recorded)
        : items_(demands.items()), views_(backing.views), recorded_(recorded) {}

    const WeightView& view(Id id) const { return views_.at(id); }

    ops::WeightInput input(Id id) const;

    // The artifact's own policy, whatever the profile.
    ops::WeightInput recorded_input(Id id) const;

    Tensor tensor(Id id) const;
    Projection projection(Id id) const;

    // Rows of one parent become one projection when they cover that parent completely; the
    // NVFP4 native Weight addresses whole parents only. A W4A4 route quantizes the shared input
    // once, so members that recorded different activation divisors are given the largest one:
    // the divisor only keeps block scales inside E4M3, and the largest keeps every member's.
    std::optional<Projection> packed(std::span<const Id> ids) const;

private:
    const std::vector<Pending>& items_;
    const std::deque<WeightView>& views_;
    bool recorded_;
};

// The policy of a MoE's routed experts: A8 (the R8 route of codebook banks at decode) only where
// both banks allow it, A4 where both allow that (an NVFP4 bank's grouped W4A8 asks), as
// ops::prepare_moe_weights decides. MoeWeights defaults to A16Only, so a loader that builds its
// MoeWeights by hand sets it from here.
inline ops::LinearPolicy expert_policy(const Projection& gate_up, const Projection& down) {
    if (ops::allows_a4(gate_up.policy) && ops::allows_a4(down.policy)) {
        return ops::LinearPolicy::AllowA4;
    }
    return ops::allows_a8(gate_up.policy) && ops::allows_a8(down.policy) ? ops::LinearPolicy::AllowA8
                                                                         : ops::LinearPolicy::A16Only;
}

} // namespace loading
} // namespace ninferno::models
