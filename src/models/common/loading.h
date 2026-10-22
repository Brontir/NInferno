#pragma once

// The loading skeleton the dense, Gemma 4 and Nemotron-H families and the DFlash drafters run
// (Qwen 3.5 has its own, qwen3_5/load.cpp), and the pieces it shares:
//
//   artifact::Reader reader(path);
//   config = parse_...(open_text_component(reader, "family ..."));  // then pairing checks
//   loading::Demands demands(reader);                               // bind every parameter
//   backing = demands.materialize(device);                          // upload, bind the views
//   const loading::Prepare prepare(demands, *backing, activations); // Op-ready parameters
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
#include "ninferno/ops/moe.h"
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
    // No demand may follow.
    [[nodiscard]] std::unique_ptr<ArtifactBacking> materialize(DeviceContext& device);

    [[nodiscard]] const std::vector<Pending>& items() const noexcept { return items_; }

private:
    const artifact::Reader& reader_;
    artifact::Binder binder_;
    std::vector<Pending> items_;
};

using Projection = ops::SingleProjectionWeight;

class Prepare {
public:
    // `activations`: the policy every projection runs under (effective_policy): Auto keeps the
    // one its Use recorded (W4A4 where the checkpoint was calibrated for it), Bits16 keeps every
    // activation BF16, Bits8 and Bits4 allow eight and four bits (a recorded-BF16 head keeps it).
    Prepare(const Demands& demands, const ArtifactBacking& backing, ActivationMode activations)
        : items_(demands.items()), views_(backing.views), activations_(activations) {}

    const WeightView& view(Id id) const { return views_.at(id); }

    // The parameter's artifact name.
    const std::string& name(Id id) const { return items_.at(id).reference.name; }

    ActivationMode activations() const noexcept { return activations_; }

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
    ActivationMode activations_;
};

// The policy of a MoE's routed experts: A8 (the R8 route of codebook banks at decode) only where
// both banks allow it, A4 where one allows that (an NVFP4 bank's grouped W4A8 asks), as
// ops::prepare_moe_weights decides (ops::moe_policy). MoeWeights defaults to A16Only, so a loader
// that builds its MoeWeights by hand sets it from here.
inline ops::LinearPolicy expert_policy(const Projection& gate_up, const Projection& down) {
    return ops::moe_policy(gate_up.policy, down.policy);
}

// A layer the generic MoE Op runs, as --activations prepares it: under Bits4 an expert or shared
// policy that allows A4 reads as AllowA8, since the Op has no four-bit route and its NVFP4 banks
// would otherwise stay on BF16 where the build has the W4A4 kernels (moe_policy); eight bits are
// the narrowest it has. The layer's widths then go to this thread's activation census
// (activation_census.h): `experts` both banks' matrices, `shared_gate_up` and `shared_down` the
// shared expert's.
void generic_moe(ops::MoeWeights& weights, const Prepare& prepare, std::span<const Id> experts,
                 std::span<const Id> shared_gate_up, std::span<const Id> shared_down);

} // namespace loading
} // namespace ninferno::models
