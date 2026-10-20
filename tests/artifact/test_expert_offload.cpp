// MoE expert offload: routed-expert objects grouped by layer out of the device arena, and the
// ExpertResidency placements a MoE call reads them through (resident, pinned Host, staged).

#include "artifact/binder.h"
#include "artifact/expert_groups.h"
#include "artifact/fixture.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/expert_residency.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::artifact;
using namespace ninferno::test::artifact_fixture;

constexpr const char* kExpert = "text/layers/0/mlp/experts/0/gate_up";
constexpr const char* kRouter = "text/layers/0/mlp/router";

void names() {
    require(routed_expert_layer("text/layers/3/mlp/experts/7/gate_up") == "text/layers/3",
            "a routed expert matrix names its layer");
    require(routed_expert_layer("text/layers/12/moe/experts/gate_up_bias") == "text/layers/12",
            "a stacked expert bias names its layer");
    require(routed_expert_layer("mtp/layers/0/moe/experts/2/down") == "mtp/layers/0",
            "an MTP expert names its component's layer");
    require(!routed_expert_layer("text/layers/3/mlp/shared/gate_up"), "a shared expert is not routed");
    require(!routed_expert_layer("text/layers/3/moe/router"), "the router is not an expert");
    require(!routed_expert_layer("text/players/3/experts/1/up"), "layers must be a whole segment");
    require(!routed_expert_layer("text/layers/x/experts/1/up"), "a layer index is a number");
    require(!routed_expert_layer("text/layers/3"), "a layer is not an expert");
}

// The fixture with its q5 parent bound as layer 0's expert and its divisors as the router.
std::vector<std::byte> expert_fixture(Fixture& fixture) {
    // The q5 parent is read by the expert alone (the fixture's own bindings of it and their
    // Uses would make it shared, and shared objects stay in the arena).
    fixture.root["bindings"].erase("matrix");
    fixture.root["bindings"].erase("row");
    fixture.root["uses"]              = Json::array();
    fixture.root["bindings"][kExpert] = {{"object", "q5"}};
    fixture.root["bindings"][kRouter] = {{"object", "divisors"}};
    fixture.write(true);
    return {fixture.payload.begin() + 256, fixture.payload.begin() + 256 + 528};
}

std::vector<std::byte> download(const void* device, std::size_t bytes) {
    std::vector<std::byte> out(bytes);
    CUDA_CHECK(cudaMemcpy(out.data(), device, bytes, cudaMemcpyDeviceToHost));
    return out;
}

struct Loaded {
    std::optional<MaterializedArtifact> backing;
    const std::byte* expert = nullptr; // the expert parent's bound bytes
    const std::byte* router = nullptr;
};

Loaded load(const Fixture& fixture, DeviceContext& device, std::optional<std::size_t> count) {
    Loaded out;
    Reader reader(fixture.entry);
    Binder binder(reader);
    const auto expert = binder.parameter(kExpert, {2, 130});
    const auto router = binder.parameter(kRouter, {2}, Residency::Device, QType::FP32);
    auto plan         = std::move(binder).finish();
    const auto layers = place_expert_groups(plan, reader, count);
    require(layers == 1, "one MoE layer");
    require(plan.expert_groups.size() == 1 && plan.expert_groups[0].name == "text/layers/0" &&
                plan.expert_groups[0].objects.size() == 1 && plan.expert_groups[0].bytes == 528,
            "the expert parent is layer 0's group");
    require(plan.device_objects.size() == 1 && plan.device_capacity_bytes == 8,
            "the arena keeps only the router, re-laid from offset 0");
    out.backing.emplace(materialize(reader, std::move(plan), device));
    out.expert = bind_view(expert, *out.backing).parts.at(0).parent->data;
    out.router = bind_view(router, *out.backing).parts.at(0).parent->data;
    return out;
}

void offloaded(DeviceContext& device) {
    Fixture fixture;
    const auto bytes = expert_fixture(fixture);
    Loaded loaded    = load(fixture, device, std::nullopt);
    ExpertResidency* experts = loaded.backing->expert_residency();
    require(experts && experts->group_count() == 1 && experts->group_bytes(0) == 528,
            "the residency holds the group");
    require(std::equal(bytes.begin(), bytes.end(), loaded.expert),
            "the Host group holds the expert's bytes across the file boundary");
    require(loaded.backing->stats().expert_host_bytes == 528 &&
                loaded.backing->stats().expert_group_count == 1,
            "stats report the Host group");
    require(experts->pinned_bytes(1) == 528 && experts->staging_bytes(1) == 2 * 4096 &&
                experts->device_bytes(1) == 2 * 4096 && experts->device_bytes(0) == 4096 &&
                experts->staging_bytes(0) == 0,
            "placement arithmetic");
    experts->resolve(1, device.device);
    // Zero-copy: the device reads the pinned Host bytes through their alias.
    {
        const auto lease = experts->bind(loaded.expert + 17, false, device.stream);
        require(lease.source() == ExpertResidency::Lease::Source::Host, "a few experts read Host");
        require(lease.map(loaded.router) == loaded.router, "a pointer outside the group is kept");
        const std::byte* alias = lease.map(loaded.expert);
        require(download(alias, bytes.size()) == bytes, "the alias reads the expert bytes");
    }
    // Streamed: the whole group lands in a staging slot before the call's stream reads it.
    const std::byte* slot = nullptr;
    {
        const auto lease = experts->bind(loaded.expert, true, device.stream);
        require(lease.source() == ExpertResidency::Lease::Source::Staged, "a whole group is staged");
        slot = lease.map(loaded.expert);
        require(slot != loaded.expert, "a staged group is read from the slot");
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        require(download(slot, bytes.size()) == bytes, "the slot holds the expert bytes");
    }
    {
        const auto again = experts->bind(loaded.expert, true, device.stream);
        require(again.map(loaded.expert) == slot, "a group still in its slot is not copied again");
    }
    // A captured call reads the Host bytes: slots move between replays.
    {
        cudaStream_t capture = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&capture, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(capture, cudaStreamCaptureModeThreadLocal));
        ExpertResidency::Lease::Source source{};
        {
            const auto lease = experts->bind(loaded.expert, true, capture);
            source           = lease.source();
        }
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(capture, &graph));
        if (graph) { CUDA_CHECK(cudaGraphDestroy(graph)); }
        CUDA_CHECK(cudaStreamDestroy(capture));
        require(source == ExpertResidency::Lease::Source::Host, "a captured call reads Host");
    }
    require(download(loaded.router, 8) ==
                std::vector<std::byte>(fixture.payload.begin() + 1024, fixture.payload.begin() + 1032),
            "the router stayed in the arena");
}

void resident(DeviceContext& device) {
    Fixture fixture;
    const auto bytes = expert_fixture(fixture);
    Loaded loaded    = load(fixture, device, std::optional<std::size_t>(1));
    ExpertResidency* experts = loaded.backing->expert_residency();
    const std::byte* host    = loaded.expert;
    experts->resolve(0, device.device);
    const auto lease = experts->bind(host, true, device.stream);
    require(lease.source() == ExpertResidency::Lease::Source::Resident,
            "a group kept on the device is resident");
    const std::byte* copy = lease.map(host + 3) - 3;
    require(copy != host && download(copy, bytes.size()) == bytes,
            "the resident copy holds the expert bytes");
}

void refusals(DeviceContext&) {
    Fixture fixture;
    (void)expert_fixture(fixture);
    Reader reader(fixture.entry);
    const auto plan_for = [&] {
        Binder binder(reader);
        (void)binder.parameter(kExpert, {2, 130});
        return std::move(binder).finish();
    };
    rejects([&] {
        auto plan = plan_for();
        (void)place_expert_groups(plan, reader, std::optional<std::size_t>(2));
    }, "offloading more layers than the model has");
    rejects([&] {
        auto plan = plan_for();
        (void)place_expert_groups(plan, reader, std::optional<std::size_t>(0));
    }, "offloading zero layers");
    // Automatic on a model without routed experts places nothing; an explicit count is refused.
    Fixture plain;
    plain.write(true);
    Reader plain_reader(plain.entry);
    Binder binder(plain_reader);
    (void)binder.parameter("matrix", {2, 130});
    auto plan = std::move(binder).finish();
    place_expert_groups(plan, plain_reader, MoeOffloadOptions{.mode = MoeOffloadMode::Automatic});
    require(plan.expert_groups.empty() && plan.device_objects.size() == 1,
            "automatic offload of a dense model places nothing");
    rejects([&] {
        (void)place_expert_groups(plan, plain_reader, std::optional<std::size_t>(1));
    }, "an explicit offload of a dense model");
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    try {
        names();
        DeviceContext device(0);
        offloaded(device);
        resident(device);
        refusals(device);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "expert offload: all checks passed\n";
    return 0;
}
