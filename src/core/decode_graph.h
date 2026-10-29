#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

namespace ninferno {

// cudaGraphInstantiate ran out of Device memory. A Program built at startup reports it as a
// capacity shortage (CapacityError), so a caller may plan again with less.
class GraphMemoryError final : public std::runtime_error {
public:
    explicit GraphMemoryError(const std::string& message) : std::runtime_error(message) {}
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;
    // The kernel nodes the captured graph holds (0 before a capture): the kernels a replay
    // launches, each paying the generation's launch time (spec/round_cost.h LaunchFacts).
    [[nodiscard]] std::uint32_t kernel_nodes() const noexcept { return kernel_nodes_; }

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_          = nullptr;
    std::uint32_t kernel_nodes_ = 0;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    // Installs a definition of the same topology; throws when CUDA rejects it.
    void update(const DecodeGraphDefinition& definition);
    // As update(), but false when CUDA rejects the definition (another topology). Either way no
    // CUDA error is left pending; after a rejection the installed parameters are unspecified, so
    // the caller installs a definition again before launching.
    [[nodiscard]] bool try_update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_       = nullptr;
    std::uint32_t kernel_nodes_ = 0; // the installed definition's
};

// The kernel nodes every DecodeGraphExecutable::launch of this process has replayed so far (each
// launch adds its graph's kernel_nodes): a measurement's delta counts the kernels its decode
// launched through the graphs (spec-tune's graph_kernels).
[[nodiscard]] std::uint64_t graph_kernels_launched() noexcept;

} // namespace ninferno
