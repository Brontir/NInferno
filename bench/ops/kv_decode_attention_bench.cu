// One decode step of the KV-grouped causal attention, outside the server, for ncu (#849, #853,
// #856).
//
// ncu cannot profile inside ninferno-serve ("Failed to create profiler module"). This bench runs
// the call the Qwen3.5 text decoder makes for each full-attention layer of an ordinary decode
// step (models/qwen3_5/execution/text.cpp, the batched branch with B=1): the public
// ops::causal_softmax_attention append entry with W=1, no valid-column mask, one KV table row and
// the output gate, at the geometry of Qwen3.5-4B/9B (D256, 16 query heads over 4 KV heads:
// CausalGeometry<256,16,4>), over --layers separate caches. The Op makes its own plan
// (make_<codec>_kv_causal_plan from the device's SM count and the envelope) and launches the
// production kernels and merges, so the launch configuration is the engine's at that context;
// the bench only prints it. Each cache is filled through ops::kv_cache_append, the codec's own
// quantizer, with varied BF16 values.
//
// Timing: CUDA events around each step (all layers), median and minimum over --iters steps after
// --warmup steps, L2 evicted before each timed step outside the events (--no-flush keeps it). The
// read rate is the census's (#618): the slope between two or more depths (--context A,B, default
// 7843,21961) of the call time against the codec's KV bytes (visible keys x bytes per token),
// which drops the fixed part of a call (launch, Q, partials, merge, gate); the intercept is that
// fixed part. Each depth's call average (bytes / whole call) is printed beside it. The timed steps
// run between cudaProfilerStart and cudaProfilerStop.

#include "ninferno/ops/kv_cache_append.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/softmax_attention.h"

#include "core/device.h"
#include "core/device_class.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ninferno_bench_common.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#if __has_include(<cxxabi.h>)
#    include <cxxabi.h>
#    define NINFERNO_KV_BENCH_DEMANGLE 1
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninferno;

namespace {

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQueryHeads = 16;
constexpr std::int32_t kKvHeads    = 4;
constexpr float kScale             = 0.0625F; // 1/sqrt(256), the Op's only scale

enum class Execution : std::uint8_t { Eager, Graph };

struct Options {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    // Visible keys of the step at each depth; two or more give the slope rate.
    std::vector<std::int32_t> contexts{7843, 21961};
    std::int32_t layers = 8;
    int iters           = 50;
    int warmup          = 5;
    Execution execution = Execution::Eager;
    bool exact          = false;
    bool flush          = true;
    // The session's KV capacity: the server's last ordinary graph profile ends at capacity - 1.
    std::uint32_t capacity = 262144;
    // 0: the engine's envelope for the execution: {N, N} eager (decode.cpp without a CUDA graph),
    // the server's ordinary graph profile around N with --execution graph.
    std::uint32_t envelope_min = 0;
    std::uint32_t envelope_max = 0;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(
        stderr,
        "error: %s\n"
        "usage: ninferno_kv_decode_attention_bench [--codec bf16|int8|fp8|nvfp4|k8v4] "
        "[--context N[,N...]] [--layers L] [--iters N] [--warmup N] "
        "[--execution eager|graph] [--activations fast|exact] [--no-flush] [--capacity C] "
        "[--envelope MIN,MAX]\n"
        "  --context N,...  visible keys of the step at each depth: the new token sits at "
        "position N-1 (default 7843,21961; two or more depths give the slope rate)\n"
        "  --layers L       full-attention layers a step runs, each over its own cache "
        "(default 8, Qwen3.5-9B)\n"
        "  --execution      eager: envelope {N,N}, as the engine without CUDA graphs; graph: the "
        "step replayed as a graph over the server's ordinary graph profile around N\n"
        "  --no-flush       keep L2 between timed steps (default: evict it before each step)\n"
        "  --capacity C     session KV capacity, where the server's last graph profile ends "
        "(default 262144)\n"
        "  --envelope       an explicit execution envelope (one depth only)\n",
        message);
    std::exit(2);
}

long parse_long(const char* text, long minimum, long maximum, const char* flag) {
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < minimum || parsed > maximum)
        usage(flag);
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    const long max_keys = static_cast<long>(ops::kCausalAttentionMaximumVisibleKeys);
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) usage(flag);
            return argv[index];
        };
        if (argument == "--codec" || argument == "--kv-dtype") {
            const std::string_view value(next("--codec requires a value"));
            bool found = false;
            for (const KvCacheStorage storage :
                 {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                  KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                  KvCacheStorage::Fp8KeyNvfp4Value}) {
                if (value == kv_cache_storage_name(storage)) {
                    options.storage = storage;
                    found           = true;
                }
            }
            if (!found) usage("--codec expects bf16, int8, fp8, nvfp4 or k8v4");
        } else if (argument == "--context") {
            const std::string value(next("--context requires a value"));
            options.contexts.clear();
            std::size_t begin = 0;
            while (begin <= value.size()) {
                const std::size_t comma = std::min(value.find(',', begin), value.size());
                options.contexts.push_back(static_cast<std::int32_t>(parse_long(
                    value.substr(begin, comma - begin).c_str(), 1, max_keys, "--context")));
                begin = comma + 1;
            }
        } else if (argument == "--layers") {
            options.layers = static_cast<std::int32_t>(
                parse_long(next("--layers requires a value"), 1, 64, "--layers"));
        } else if (argument == "--iters") {
            options.iters = static_cast<int>(
                parse_long(next("--iters requires a value"), 1, 100000, "--iters"));
        } else if (argument == "--warmup") {
            options.warmup = static_cast<int>(
                parse_long(next("--warmup requires a value"), 0, 100000, "--warmup"));
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else
                usage("--execution expects eager or graph");
        } else if (argument == "--activations") {
            const std::string_view value(next("--activations requires a value"));
            if (value == "fast")
                options.exact = false;
            else if (value == "exact")
                options.exact = true;
            else
                usage("--activations expects fast or exact");
        } else if (argument == "--no-flush") {
            options.flush = false;
        } else if (argument == "--capacity") {
            options.capacity = static_cast<std::uint32_t>(
                parse_long(next("--capacity requires a value"), 1, max_keys, "--capacity"));
        } else if (argument == "--envelope") {
            const std::string value(next("--envelope requires MIN,MAX"));
            const std::size_t comma = value.find(',');
            if (comma == std::string::npos) usage("--envelope expects MIN,MAX");
            options.envelope_min = static_cast<std::uint32_t>(
                parse_long(value.substr(0, comma).c_str(), 1, max_keys, "--envelope MIN"));
            options.envelope_max = static_cast<std::uint32_t>(
                parse_long(value.substr(comma + 1).c_str(), 1, max_keys, "--envelope MAX"));
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    for (const std::int32_t context : options.contexts) {
        if (static_cast<std::uint32_t>(context) > options.capacity)
            usage("every --context must fit the --capacity");
    }
    if (options.envelope_max != 0) {
        const auto context = static_cast<std::uint32_t>(options.contexts.front());
        if (options.contexts.size() != 1) usage("--envelope takes one --context");
        if (options.envelope_min > options.envelope_max || context < options.envelope_min ||
            context > options.envelope_max)
            usage("--envelope must satisfy MIN <= context <= MAX");
    }
    return options;
}

// The server's ordinary decode graph profile whose frontiers hold position N-1, as the envelope
// its captured step runs over (program/graphs.cpp: {min + 1, max + 1}).
ops::CausalAttentionExecutionEnvelope server_graph_envelope(std::int32_t visible,
                                                            std::uint32_t capacity) {
    const auto frontier = static_cast<std::uint32_t>(visible - 1);
    for (const auto& profile : models::qwen3_5::detail::ordinary_graph_profiles(capacity)) {
        if (profile.min <= frontier && frontier <= profile.max)
            return {profile.min + 1, profile.max + 1};
    }
    throw std::invalid_argument("no ordinary graph profile holds the context");
}

__global__ void initialize_values(__nv_bfloat16* data, std::size_t count, unsigned seed,
                                  float scale) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned value = static_cast<unsigned>(i) * 0x9e3779b9U + seed;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    data[i] = __float2bfloat16_rn((float(value >> 8) * (2.f / 16777216.f) - 1.f) * scale);
}

void fill_values(DeviceBuffer& buffer, std::size_t count, unsigned seed, float scale,
                 cudaStream_t stream) {
    if (buffer.bytes < count * sizeof(__nv_bfloat16))
        throw std::logic_error("value buffer is smaller than its fill");
    const auto blocks = static_cast<unsigned>((count + 255) / 256);
    initialize_values<<<blocks, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(buffer.p), count,
                                                  seed, scale);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t plane_bytes(const PagedKVVectorLayout& vector, std::int32_t pages, bool scale) {
    const std::size_t extent = scale ? vector.scale_leading_extent : vector.data_leading_extent;
    const DType dtype        = scale ? vector.scale_dtype : vector.data_dtype;
    return extent * kKvHeads * kPagedKVPageSize * static_cast<std::size_t>(pages) *
           dtype_size(dtype);
}

Tensor plane(DeviceBuffer& buffer, const PagedKVVectorLayout& vector, std::int32_t pages,
             bool scale) {
    if (scale && !vector.has_scale()) return Tensor();
    return Tensor(buffer.p, scale ? vector.scale_dtype : vector.data_dtype,
                  {scale ? vector.scale_leading_extent : vector.data_leading_extent,
                   kPagedKVPageSize, kKvHeads, pages});
}

// One full-attention layer: its paged cache (identity block table, one row) and the decode
// step's inputs, as the engine binds them for an ordinary decode of one sequence.
class Layer {
public:
    // `pages` physical pages hold the visible keys; the block table spans `logical_pages` (the
    // envelope's capacity), its pages past the physical ones pointing at page 0, never read.
    Layer(const PagedKVStorageLayout& layout, std::int32_t pages, std::int32_t logical_pages,
          std::int32_t position, unsigned seed, cudaStream_t stream)
        : pages_(pages), k_pages_(plane_bytes(layout.key, pages, false)),
          v_pages_(plane_bytes(layout.value, pages, false)),
          k_scales_(layout.key.has_scale() ? plane_bytes(layout.key, pages, true) : 1),
          v_scales_(layout.value.has_scale() ? plane_bytes(layout.value, pages, true) : 1),
          table_(static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t)),
          q_(std::size_t{kHeadDim} * kQueryHeads * 2),
          gate_(std::size_t{kHeadDim} * kQueryHeads * 2), k_(std::size_t{kHeadDim} * kKvHeads * 2),
          v_(std::size_t{kHeadDim} * kKvHeads * 2), out_(std::size_t{kHeadDim} * kQueryHeads * 2),
          position_(sizeof(std::int32_t)), table_row_(sizeof(std::int32_t)) {
        k_pages_.fill();
        v_pages_.fill();
        k_scales_.fill();
        v_scales_.fill();
        out_.fill();
        std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages), 0);
        for (std::int32_t page = 0; page < std::min(pages, logical_pages); ++page)
            table[page] = page;
        table_.copy_from_host(table.data(), table_.bytes);
        const std::int32_t row = 0;
        position_.copy_from_host(&position, sizeof(position));
        table_row_.copy_from_host(&row, sizeof(row));
        fill_values(q_, std::size_t{kHeadDim} * kQueryHeads, seed + 1, .25f, stream);
        fill_values(gate_, std::size_t{kHeadDim} * kQueryHeads, seed + 2, 1.f, stream);
        fill_values(k_, std::size_t{kHeadDim} * kKvHeads, seed + 3, .25f, stream);
        fill_values(v_, std::size_t{kHeadDim} * kKvHeads, seed + 4, 1.f, stream);
        cache_ = PagedKVLayerView{
            .k_pages       = plane(k_pages_, layout.key, pages, false),
            .v_pages       = plane(v_pages_, layout.value, pages, false),
            .k_scale_pages = plane(k_scales_, layout.key, pages, true),
            .v_scale_pages = plane(v_scales_, layout.value, pages, true),
            .block_table   = Tensor(table_.p, DType::I32, {logical_pages}),
            .head_dim      = kHeadDim,
            .num_kv_heads  = kKvHeads,
            .storage       = layout.storage,
        };
    }

    // Every slot of the cache through the codec's own quantizer (ops::kv_cache_append).
    void populate(DeviceBuffer& keys, DeviceBuffer& values, const Tensor& positions, unsigned seed,
                  cudaStream_t stream) {
        const std::int32_t slots = pages_ * kPagedKVPageSize;
        const std::size_t count  = std::size_t{kHeadDim} * kKvHeads * slots;
        fill_values(keys, count, seed + 11, .25f, stream);
        fill_values(values, count, seed + 13, 1.f, stream);
        const Tensor k(keys.p, DType::BF16, {kHeadDim, kKvHeads, slots});
        const Tensor v(values.p, DType::BF16, {kHeadDim, kKvHeads, slots});
        ops::kv_cache_append(k, v, positions, cache_, stream);
    }

    void launch(ops::CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                DeviceExecutionView execution) {
        const Tensor q(q_.p, DType::BF16, {kHeadDim, kQueryHeads, 1, 1});
        const Tensor gate(gate_.p, DType::BF16, {kHeadDim, kQueryHeads, 1, 1});
        const Tensor k(k_.p, DType::BF16, {kHeadDim, kKvHeads, 1, 1});
        const Tensor v(v_.p, DType::BF16, {kHeadDim, kKvHeads, 1, 1});
        const Tensor positions(position_.p, DType::I32, {1, 1});
        const Tensor rows(table_row_.p, DType::I32, {1});
        Tensor out(out_.p, DType::BF16, {kHeadDim, kQueryHeads, 1, 1});
        ops::causal_softmax_attention(
            q, k, v, positions, Tensor{}, rows, {kHeadDim, kQueryHeads, kKvHeads}, kScale,
            single_row_paged_kv_batch_view(cache_), envelope, workspace, out, execution, &gate);
    }

private:
    std::int32_t pages_;
    DeviceBuffer k_pages_, v_pages_, k_scales_, v_scales_, table_;
    DeviceBuffer q_, gate_, k_, v_, out_, position_, table_row_;
    PagedKVLayerView cache_{};
};

std::string demangle(const char* name) {
    if (name == nullptr) return "?";
#ifdef NINFERNO_KV_BENCH_DEMANGLE
    int status = 0;
    std::unique_ptr<char, decltype(&std::free)> text(
        abi::__cxa_demangle(name, nullptr, nullptr, &status), &std::free);
    if (status == 0 && text) return text.get();
#endif
    return name;
}

// The kernel's name without its namespace and template arguments ("bf16_kv_grouped_mma_kernel").
std::string short_name(const std::string& full) {
    std::string name = full.substr(0, full.find_first_of("<("));
    if (const std::size_t colon = name.rfind("::"); colon != std::string::npos)
        name = name.substr(colon + 2);
    const std::size_t space = name.rfind(' ');
    return space == std::string::npos ? name : name.substr(space + 1);
}

struct LaunchRecord {
    std::string name;
    std::string full;
    cudaKernelNodeParams params{};
};

// The kernels one layer's call launches, read back from a capture of it (nothing runs).
std::vector<LaunchRecord> capture_launches(Layer& layer,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           WorkspaceArena& workspace,
                                           DeviceExecutionView execution) {
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(execution.stream, cudaStreamCaptureModeThreadLocal));
    try {
        layer.launch(envelope, workspace, execution);
    } catch (...) {
        cudaStreamEndCapture(execution.stream, &graph);
        if (graph != nullptr) cudaGraphDestroy(graph);
        throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(execution.stream, &graph));
    std::size_t count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
    // The nodes of one stream's capture, in the order the driver lists them (their creation
    // order). The edge queries are not used: their signature differs between CUDA 12 and 13.
    std::vector<LaunchRecord> records;
    for (const cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        LaunchRecord record;
        if (type != cudaGraphNodeTypeKernel) {
            record.name = type == cudaGraphNodeTypeMemset   ? "(memset)"
                          : type == cudaGraphNodeTypeMemcpy ? "(memcpy)"
                                                            : "(other node)";
            records.push_back(record);
            continue;
        }
        CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &record.params));
        const char* mangled = nullptr;
#if CUDART_VERSION >= 12030
        if (cudaFuncGetName(&mangled, record.params.func) != cudaSuccess) mangled = nullptr;
#endif
        record.full = demangle(mangled);
        record.name = short_name(record.full);
        records.push_back(record);
    }
    CUDA_CHECK(cudaGraphDestroy(graph));
    return records;
}

void print_launches(const std::vector<LaunchRecord>& records) {
    for (const LaunchRecord& record : records) {
        if (record.params.func == nullptr) {
            std::printf("  %s\n", record.name.c_str());
            continue;
        }
        const cudaKernelNodeParams& p = record.params;
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, p.func));
        int resident = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, p.func, static_cast<int>(p.blockDim.x * p.blockDim.y * p.blockDim.z),
            p.sharedMemBytes));
        std::printf("  %s grid=(%u,%u,%u) block=(%u,%u,%u) CTAs=%u dynamic_smem=%u B "
                    "static_smem=%zu B regs=%d local=%zu B resident_CTAs/SM=%d\n",
                    record.name.c_str(), p.gridDim.x, p.gridDim.y, p.gridDim.z, p.blockDim.x,
                    p.blockDim.y, p.blockDim.z, p.gridDim.x * p.gridDim.y * p.gridDim.z,
                    p.sharedMemBytes, attributes.sharedSizeBytes, attributes.numRegs,
                    attributes.localSizeBytes, resident);
        std::printf("    %s\n", record.full.c_str());
    }
}

const char* bf16_instance_name(ops::detail::Bf16KvInstance instance) {
    using ops::detail::Bf16KvInstance;
    switch (instance) {
    case Bf16KvInstance::GroupedDecode:
        return "GroupedDecode";
    case Bf16KvInstance::Grouped32:
        return "Grouped32";
    case Bf16KvInstance::Grouped64:
        return "Grouped64";
    case Bf16KvInstance::Tiled64:
        return "Tiled64";
    case Bf16KvInstance::Tiled128:
        return "Tiled128";
    }
    return "?";
}

template <class Family>
const char* family_name(Family family) {
    switch (static_cast<int>(family)) {
    case 0:
        return "Grouped";
    case 1:
        return "ParallelGrouped";
    case 2:
        return "Tiled";
    }
    return "?";
}

void print_causal_partition(const char* family, const ops::detail::CausalKvPartition& partition,
                            int visible) {
    using ops::detail::CausalKvPartition;
    const int splits =
        partition.aligned ? CausalKvPartition::aligned_active(visible) : partition.active(visible);
    std::printf("plan: family=%s partition capacity=%d target=%d key_shift=%d aligned=%d -> "
                "%d splits at %d keys (~%d keys a split)\n",
                family, partition.capacity, partition.target,
                partition.aligned ? CausalKvPartition::aligned_key_shift(visible)
                                  : partition.key_shift,
                partition.aligned ? 1 : 0, splits, visible, (visible + splits - 1) / splits);
}

// The plan the Op makes for this call (the same function and inputs as its launch).
void print_plan(KvCacheStorage storage, ops::CausalAttentionExecutionEnvelope envelope, int sms,
                int visible) {
    namespace d = ops::detail;
    switch (storage) {
    case KvCacheStorage::BFloat16: {
        const auto plan = d::make_bf16_kv_causal_plan(kQueryHeads, kKvHeads, 1, 1, envelope, sms);
        const auto live = plan.partition.aligned ? d::Bf16KvPartition::aligned_live(visible)
                                                 : plan.partition.live(visible);
        std::printf("plan: instance=%s partition capacity=%d normal_target=%d long_target=%d "
                    "key_rows=%d aligned=%d -> %d splits at %d keys (%d keys a split)\n",
                    bf16_instance_name(plan.instance), plan.partition.capacity,
                    plan.partition.normal_target, plan.partition.long_target,
                    plan.partition.key_rows, plan.partition.aligned ? 1 : 0, live.splits, visible,
                    live.keys_per_split);
        return;
    }
    case KvCacheStorage::Int8Group64: {
        const auto plan = d::make_int8_kv_causal_plan(kQueryHeads, kKvHeads, 1, 1, envelope, sms);
        return print_causal_partition(family_name(plan.family), plan.partition, visible);
    }
    case KvCacheStorage::Fp8E4M3Row256: {
        const auto plan = d::make_fp8_kv_causal_plan(kQueryHeads, kKvHeads, 1, 1, envelope, sms);
        return print_causal_partition(family_name(plan.family), plan.partition, visible);
    }
    case KvCacheStorage::Nvfp4Group16: {
        const auto plan = d::make_nvfp4_kv_causal_plan(kQueryHeads, kKvHeads, 1, 1, envelope, sms);
        return print_causal_partition(family_name(plan.family), plan.partition, visible);
    }
    case KvCacheStorage::Fp8KeyNvfp4Value: {
        const auto plan = d::make_k8v4_kv_causal_plan(kQueryHeads, kKvHeads, 1, 1, envelope, sms);
        return print_causal_partition(family_name(plan.family), plan.partition, visible);
    }
    }
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    return n % 2 == 1 ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

// The attention kernels of a call: every kernel but the merges, the gate and the appends.
bool attention_kernel(const LaunchRecord& record) {
    if (record.params.func == nullptr) return false;
    for (const char* other : {"merge", "sigmoid_mul", "append"}) {
        if (record.name.find(other) != std::string::npos) return false;
    }
    return true;
}

// An ncu -k regex for the attention kernels: "kv_grouped" when it names them all (the census
// spelling), else their names (Volta's INT8 key-major kernel, a tiled or chunked call).
std::string attention_regex(const std::vector<LaunchRecord>& launches) {
    std::vector<std::string> names;
    bool grouped = true;
    for (const LaunchRecord& record : launches) {
        if (!attention_kernel(record)) continue;
        grouped = grouped && record.name.find("kv_grouped") != std::string::npos;
        if (std::find(names.begin(), names.end(), record.name) == names.end())
            names.push_back(record.name);
    }
    if (names.empty()) return "kv";
    if (grouped) return "kv_grouped";
    std::string regex;
    for (const std::string& name : names) regex += (regex.empty() ? "" : "|") + name;
    return regex;
}

struct Depth {
    std::int32_t visible = 0;
    double kv_bytes      = 0;
    double call_us       = 0;
};

double share(double gbs, double roof) { return roof > 0 ? 100.0 * gbs / roof : 0.0; }

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        std::optional<ops::ScopedInvariantRoutes> invariant;
        if (options.exact) invariant.emplace();

        DeviceContext device;
        const DeviceExecutionView execution = device.execution_view();
        const cudaStream_t stream           = execution.stream;
        const DeviceClass& gpu              = device_class();
        const double dram_gbs               = gpu.dram_gbps();
        const double read_gbs               = bench::device_roofs().read_gbs;
        const double l2_mib                 = static_cast<double>(gpu.l2_bytes) / 1048576.0;

        const PagedKVStorageLayout layout = paged_kv_storage_layout(options.storage, kHeadDim);
        const std::size_t token_bytes =
            (layout.key.physical_bytes() + layout.value.physical_bytes()) * kKvHeads;
        // Evicting L2 reads a preinitialized buffer of twice its size (bench::L2FlushBuffer: a
        // read leaves no dirty lines whose writeback the next step would pay).
        std::optional<bench::L2FlushBuffer> flush;
        if (options.flush)
            flush.emplace(std::max<std::size_t>(2 * gpu.l2_bytes, std::size_t{64} << 20));

        std::printf("device: %s (%s, %s), %d SMs, L2 %.1f MiB, DRAM %.1f GB/s (facts), read "
                    "ceiling %.1f GB/s\n",
                    gpu.name.c_str(), gpu.id().c_str(), gpu.arch().c_str(), gpu.sms, l2_mib,
                    dram_gbs, read_gbs);
        std::printf(
            "call: causal_softmax_attention append, D%d H%d/KV%d (Qwen3.5-9B full "
            "attention), B=1 W=1, gate, %d layers; activations %s, execution %s, L2 %s "
            "before each timed step\n",
            kHeadDim, kQueryHeads, kKvHeads, options.layers, options.exact ? "exact" : "fast",
            options.execution == Execution::Eager ? "eager" : "graph", flush ? "evicted" : "kept");
        std::printf("codec: %s, %zu B a token a layer (key %zu + value %zu per KV head), %zu B a "
                    "token over %d layers\n",
                    kv_cache_storage_name(options.storage).data(), token_bytes,
                    layout.key.physical_bytes(), layout.value.physical_bytes(),
                    token_bytes * options.layers, options.layers);

        std::vector<Depth> depths;
        // Attention-kernel launches so far, for ncu's --launch-skip.
        long launched = 0;
        for (const std::int32_t visible : options.contexts) {
            const auto graph_envelope = server_graph_envelope(visible, options.capacity);
            ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(visible),
                                                           static_cast<std::uint32_t>(visible)};
            if (options.envelope_max != 0)
                envelope = {options.envelope_min, options.envelope_max};
            else if (options.execution == Execution::Graph)
                envelope = graph_envelope;
            std::printf("\ndepth: visible keys %d (position %d), envelope [%u,%u]; the server's "
                        "ordinary graph profile here: [%u,%u] (capacity %u)\n",
                        visible, visible - 1, envelope.min_visible_keys, envelope.max_visible_keys,
                        graph_envelope.min_visible_keys, graph_envelope.max_visible_keys,
                        options.capacity);
            print_plan(options.storage, envelope, execution.multiprocessor_count, visible);

            const std::int32_t pages = (visible + kPagedKVPageSize - 1) / kPagedKVPageSize;
            const auto logical_pages = static_cast<std::int32_t>(
                (envelope.max_visible_keys + kPagedKVPageSize - 1) / kPagedKVPageSize);
            const std::int32_t slots = pages * kPagedKVPageSize;
            // The caches, filled through the codec's quantizer; the staging values are freed
            // after.
            std::vector<std::unique_ptr<Layer>> layers;
            {
                DeviceBuffer keys(std::size_t{kHeadDim} * kKvHeads * slots * 2);
                DeviceBuffer values(std::size_t{kHeadDim} * kKvHeads * slots * 2);
                std::vector<std::int32_t> host_positions(static_cast<std::size_t>(slots));
                for (std::int32_t slot = 0; slot < slots; ++slot) host_positions[slot] = slot;
                DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
                positions.copy_from_host(host_positions.data(), positions.bytes);
                const Tensor position_tensor(positions.p, DType::I32, {slots});
                for (std::int32_t index = 0; index < options.layers; ++index) {
                    const unsigned seed = 1000U * static_cast<unsigned>(index + 1);
                    layers.push_back(std::make_unique<Layer>(layout, pages, logical_pages,
                                                             visible - 1, seed, stream));
                    layers.back()->populate(keys, values, position_tensor, seed, stream);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            const std::size_t workspace_bytes =
                ops::causal_softmax_attention_workspace_capacity_bytes(
                    {kHeadDim, kQueryHeads, kKvHeads}, options.storage, envelope, 1, 1, 1,
                    execution);
            WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
            const auto step = [&](cudaStream_t target) {
                for (const auto& layer : layers)
                    layer->launch(envelope, workspace, execution.on_stream(target));
            };

            // One step primes the kernels' attributes; then one layer's call is captured to read
            // its launches back.
            step(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto launches = capture_launches(*layers.front(), envelope, workspace, execution);
            std::printf("launches a call (%zu B workspace):\n", workspace_bytes);
            print_launches(launches);
            const auto per_call = static_cast<long>(
                std::count_if(launches.begin(), launches.end(), attention_kernel));
            const double working_set = static_cast<double>(token_bytes) * visible * options.layers;
            std::printf("KV read a step: %.1f MiB over %d layers (L2 %.1f MiB)\n",
                        working_set / 1048576.0, options.layers, l2_mib);
            if (working_set < 2.0 * static_cast<double>(gpu.l2_bytes))
                std::printf("warning: the KV read of a step is below 2x L2; %s\n",
                            flush ? "L2 is evicted before each timed step, but a step's later "
                                    "layers may still hit lines its earlier ones left"
                                  : "with --no-flush the steps read partly from L2");
            // The attention launches before this depth's timed steps: the earlier depths, this
            // depth's priming step and its warmup.
            launched += per_call * options.layers * (1 + options.warmup);
            std::printf("ncu: %ld attention kernel(s) a call; this depth's timed steps start "
                        "after %ld of them: -k regex:%s --launch-skip %ld (or "
                        "--profile-from-start off with one --context)\n",
                        per_call, launched, attention_regex(launches).c_str(), launched);
            launched += per_call * options.layers * options.iters;

            bench::TimedGraph graph;
            if (options.execution == Execution::Graph) graph.capture(stream, step);
            const auto run = [&] {
                if (options.execution == Execution::Graph)
                    graph.launch(stream);
                else
                    step(stream);
            };
            for (int index = 0; index < options.warmup; ++index) run();
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // A start and a stop event a step; the eviction runs between steps, untimed.
            const auto steps = static_cast<std::size_t>(options.iters);
            std::vector<cudaEvent_t> events(2 * steps);
            for (cudaEvent_t& event : events) CUDA_CHECK(cudaEventCreate(&event));
            CUDA_CHECK(cudaProfilerStart());
            for (std::size_t index = 0; index < steps; ++index) {
                if (flush) bench::flush_l2(*flush, stream);
                CUDA_CHECK(cudaEventRecord(events[2 * index], stream));
                run();
                CUDA_CHECK(cudaEventRecord(events[2 * index + 1], stream));
            }
            CUDA_CHECK(cudaEventSynchronize(events.back()));
            CUDA_CHECK(cudaProfilerStop());
            std::vector<double> samples;
            for (std::size_t index = 0; index < steps; ++index) {
                float milliseconds = 0.F;
                CUDA_CHECK(
                    cudaEventElapsedTime(&milliseconds, events[2 * index], events[2 * index + 1]));
                samples.push_back(static_cast<double>(milliseconds) * 1000.0);
            }
            for (cudaEvent_t event : events) CUDA_CHECK(cudaEventDestroy(event));

            const double step_us  = median(samples);
            const double min_us   = *std::min_element(samples.begin(), samples.end());
            const double call_us  = step_us / options.layers;
            const double kv_bytes = static_cast<double>(token_bytes) * visible;
            const double gbs      = kv_bytes / (call_us * 1.0e3);
            std::printf("result: codec=%s context=%d layers=%d step_us=%.2f (min %.2f) "
                        "call_us=%.3f kv_bytes_per_call=%.0f call_average_GB/s=%.1f "
                        "(%.1f%% of DRAM, fixed part included)\n",
                        kv_cache_storage_name(options.storage).data(), visible, options.layers,
                        step_us, min_us, call_us, kv_bytes, gbs, share(gbs, dram_gbs));
            depths.push_back({visible, kv_bytes, call_us});
        }

        // Least squares of the call time against the KV bytes: the slope is the read rate, the
        // intercept the fixed part of a call (for two depths, the two-point slope).
        std::printf("\n");
        double mean_x = 0, mean_y = 0;
        for (const Depth& depth : depths) {
            mean_x += depth.kv_bytes / static_cast<double>(depths.size());
            mean_y += depth.call_us / static_cast<double>(depths.size());
        }
        double sxy = 0, sxx = 0;
        for (const Depth& depth : depths) {
            sxy += (depth.kv_bytes - mean_x) * (depth.call_us - mean_y);
            sxx += (depth.kv_bytes - mean_x) * (depth.kv_bytes - mean_x);
        }
        if (sxx <= 0) {
            std::printf("rate: one depth only, no slope; the call average above includes the "
                        "fixed part\n");
            return 0;
        }
        const double us_per_byte = sxy / sxx;
        const double fixed_us    = mean_y - us_per_byte * mean_x;
        const double slope_gbs   = us_per_byte > 0 ? 1.0 / (us_per_byte * 1.0e3) : 0.0;
        std::string over;
        for (const Depth& depth : depths)
            over += (over.empty() ? "" : ",") + std::to_string(depth.visible);
        std::printf("rate: codec=%s slope over contexts %s: read_rate_GB/s=%.1f of_dram=%.1f%% "
                    "of_read_ceiling=%.1f%%; fixed part (intercept) %.3f us a call, %.2f us a "
                    "step\n",
                    kv_cache_storage_name(options.storage).data(), over.c_str(), slope_gbs,
                    share(slope_gbs, dram_gbs), share(slope_gbs, read_gbs), fixed_us,
                    fixed_us * options.layers);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninferno_kv_decode_attention_bench: %s\n", error.what());
        return 1;
    }
}
