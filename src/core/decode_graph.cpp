#include "core/decode_graph.h"
#include "core/cuda_error.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/units.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno {
namespace {

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

std::atomic<std::uint64_t> g_graph_kernels{0};

// The kernel nodes of a captured graph (0 when CUDA cannot list them).
std::uint32_t count_kernel_nodes(cudaGraph_t graph) noexcept {
    std::size_t n = 0;
    if (graph == nullptr) { return 0; }
    if (cudaGraphGetNodes(graph, nullptr, &n) != cudaSuccess) {
        (void)cudaGetLastError(); // its own failure only
        std::fprintf(stderr, "warning: a decode graph's nodes could not be listed (not counted)\n");
        return 0;
    }
    if (n == 0) { return 0; }
    std::vector<cudaGraphNode_t> nodes(n);
    if (cudaGraphGetNodes(graph, nodes.data(), &n) != cudaSuccess) {
        (void)cudaGetLastError();
        std::fprintf(stderr, "warning: a decode graph's nodes could not be listed (not counted)\n");
        return 0;
    }
    std::uint32_t kernels = 0;
    for (std::size_t i = 0; i < n; ++i) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        if (type == cudaGraphNodeTypeKernel) { ++kernels; }
    }
    return kernels;
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_), kernel_nodes_(other.kernel_nodes_) {
    other.graph_        = nullptr;
    other.kernel_nodes_ = 0;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_        = other.graph_;
    kernel_nodes_ = other.kernel_nodes_;

    other.graph_        = nullptr;
    other.kernel_nodes_ = 0;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_        = graph;
    kernel_nodes_ = count_kernel_nodes(graph_);
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept {
    destroy_graph(graph_);
    kernel_nodes_ = 0;
}

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_), kernel_nodes_(other.kernel_nodes_) {
    other.exec_         = nullptr;
    other.kernel_nodes_ = 0;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_               = other.exec_;
    kernel_nodes_       = other.kernel_nodes_;
    other.exec_         = nullptr;
    other.kernel_nodes_ = 0;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange instantiate_range(nvtx::Name::CudaGraphInstantiate, nvtx::Category::Graph);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    // The nodes' own priorities (an urgent launch's, pdl::launch), not the launch stream's.
    const cudaError_t err = cudaGraphInstantiateWithFlags(&exec, definition.graph_,
                                                          cudaGraphInstantiateFlagUseNodePriority);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        // Instantiation failures are not sticky: clear the error so the context stays usable and
        // the caller can report or recover (a smaller context, chunk or draft depth).
        (void)cudaGetLastError();
        std::string message = std::string("CUDA Graph instantiation failed: ") +
                              cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
        if (err == cudaErrorMemoryAllocation) {
            std::size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                char memory[96];
                std::snprintf(memory, sizeof(memory), " (%.0f MiB free of %.0f MiB)",
                              to_mib(free_bytes), to_mib(total_bytes));
                message += memory;
            }
            message += ". The session left too little device memory for its captured graphs: "
                       "use a smaller context or prefill chunk, or fewer draft tokens.";
            throw GraphMemoryError(message);
        }
        throw std::runtime_error(message);
    }
    exec_         = exec;
    kernel_nodes_ = definition.kernel_nodes_;
}

namespace {

// One cudaGraphExecUpdate. A rejected update is recorded as the thread's last CUDA error
// (cudaErrorGraphExecUpdateFailure) although the context stays usable: clear it here, or the next
// CUDA_CHECK(cudaGetLastError()) after any launch reports it as that launch's failure.
cudaError_t exec_update(cudaGraphExec_t exec, cudaGraph_t graph,
                        cudaGraphExecUpdateResultInfo& result) noexcept {
    const cudaError_t err = cudaGraphExecUpdate(exec, graph, &result);
    if (err == cudaErrorGraphExecUpdateFailure) { (void)cudaGetLastError(); }
    return err;
}

bool rejected_update(cudaError_t err, const cudaGraphExecUpdateResultInfo& result) noexcept {
    return err == cudaErrorGraphExecUpdateFailure ||
           (err == cudaSuccess && result.result != cudaGraphExecUpdateSuccess);
}

} // namespace

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = exec_update(exec_, definition.graph_, result);
    if (err != cudaSuccess && !rejected_update(err, result)) { // the context's failure
        throw CudaError(err, "CUDA Graph executable update failed: " +
                                 std::string(cudaGetErrorName(err)));
    }
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
    }
    kernel_nodes_ = definition.kernel_nodes_;
}

bool DecodeGraphExecutable::try_update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = exec_update(exec_, definition.graph_, result);
    if (rejected_update(err, result)) { return false; }
    if (err != cudaSuccess) { // the context's failure, not the definition's
        throw CudaError(err, "CUDA Graph executable update failed: " +
                                 std::string(cudaGetErrorName(err)));
    }
    kernel_nodes_ = definition.kernel_nodes_;
    return true;
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    nvtx::ScopedRange upload_range(nvtx::Name::CudaGraphUpload, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    // This range executes for every replay; ranges in the captured body execute only at capture.
    nvtx::ScopedRange launch_range(nvtx::Name::CudaGraphLaunch, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    g_graph_kernels.fetch_add(kernel_nodes_, std::memory_order_relaxed);
}

std::uint64_t graph_kernels_launched() noexcept {
    return g_graph_kernels.load(std::memory_order_relaxed);
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept {
    destroy_graph_exec(exec_);
    kernel_nodes_ = 0;
}

} // namespace ninferno
