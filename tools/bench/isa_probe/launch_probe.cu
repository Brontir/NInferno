// launch_probe: what a launch costs on one card, the launch facts of its architecture (#613
// PR-8b). Four kinds, each the cost of one more launch in a long back-to-back series of empty
// kernels, so the series' fixed parts (the graph launch, the first node, the events) cancel:
//   node_ns      an empty kernel of one CTA (one warp) as a node of a CUDA graph, each node
//                depending on the one before (a captured stream's linear chain): graph replay,
//                how the engine runs decode;
//   node_pdl_ns  the same node launched with the programmatic stream serialization attribute,
//                its kernel waiting on the one before (griddepcontrol.wait) at its start and
//                releasing the next (griddepcontrol.launch_dependents) at its end: the Blackwell
//                decode chain. On sm_90+ the captured graph's edges are checked to be
//                programmatic (a chain of N nodes has N - 1 of them), or there is no fact.
//                griddepcontrol is sm_90+; below it the kernel is empty and the attribute alone
//                is timed (core/pdl.cuh: the engine does not set it there), and where the driver
//                refuses the attribute, or will not let the host queue such a graph ahead of
//                earlier work, the fact is not printed (a note says why);
//   stream_ns    the one-CTA kernel launched on a stream (<<<>>>) from a backlog: the host has
//                queued them all before the first may start, so the figure is the GPU's own
//                cost per stream launch, not the host's submit rate (that is the machine's CPU
//                and only a note);
//   floor_ns_C   an empty grid of C CTAs of 128 threads (the decode linears' four warps) as a
//                graph node of a chain, C = 1, SMs, 4 x SMs, 1024 and 4096: what a call of C
//                empty CTAs costs, so floor_ns_C - floor_ns_1 is the CTA dispatch ramp of a
//                C-CTA grid (#850's grids are 256 to 3072 CTAs, 1024 on 170 SMs the typical one;
//                27 % of the selector goldens' routes are past 1024 CTAs, so 4096 is read, not
//                extrapolated: #886; from launch_probe 2).
//
// Timer. CUDA events, whose tick is 1.024 us on the consumer boards and the GV100
// (route_tune_tool.cpp): one launch of about 1 us cannot be timed by them, a series can. Each
// timing is a series of N launches between two events. A repeat times the series of N = 1, the
// middle and the longest one after another; its slope is (t_N - t_1) / (N - 1), N - 1 = 1023
// graph nodes or 255 stream launches, and the fact is the median of kRepeats such slopes (two
// ticks over that many launches are 2 ns or 8 ns of a slope). The middle series (256 nodes, 64
// launches) gives the same slope over a shorter span, and a slope that moves with N is refused
// (self-check): the cost of a launch is a constant of the series or it is no fact.
//
// Every series runs behind a hold: a kernel of FFMAs on every SM that ends a fixed time after it
// starts (20 ms), the first event recorded after it, the series queued while it runs. So the
// series starts with the whole series already queued, and the timing is the GPU's alone; a
// series the host had not finished queuing when the hold ended is refused (self-check). A longer
// hold (200 ms) comes before the first kind. That the chip runs the series at its working clock
// is the intent of the holds, not a reading: the clocks are what `isa_probe.py run`'s nvidia-smi
// note saw over the run. The repeats alternate the series, so a drift of the clock over the run
// reaches all of them alike.
//
// Build (CUDA 13.x; sm_70 needs CUDA 12.8's nvcc), as `isa_probe.py run --section launch` does:
//   nvcc -O3 -std=c++17 -arch=sm_<arch> -o launch_probe tools/bench/isa_probe/launch_probe.cu
// Run (about 10 s; the GPU otherwise idle): `fact <name> <value> ns` lines; exit 0 with
// `selfcheck ok`, 3 when a result fails the self-check (its reasons printed), 2 on a CUDA error.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
            std::exit(2);                                                                          \
        }                                                                                          \
    } while (0)
#define LAUNCHED() CHECK(cudaGetLastError())

namespace {

constexpr int kRepeats        = 15;
constexpr long long kHoldNs   = 20'000'000;  // the hold before every timed series
constexpr long long kWarmNs   = 200'000'000; // the hold before the first kind
constexpr int kNodeThreads    = 32;          // the node kernels: one warp
constexpr int kFloorThreads   = 128;         // the floor grids: four warps a CTA
constexpr int kGraphSeries[]  = {1, 256, 1024};
constexpr int kStreamSeries[] = {1, 64, 256};
// A slope of the middle series within this of the longest series' (the larger of the two).
constexpr double kLinearShare = 0.10;
constexpr double kLinearNs    = 20.0;
// A fact outside this is no launch of an empty kernel on any card the probe knows.
constexpr double kFactMinNs = 10.0, kFactMaxNs = 100'000.0;

__device__ __forceinline__ long long globaltimer_ns() {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void empty_node() {}

// The PDL node: waits on the grid before it at its start, releases the next at its end, as the
// engine's Store linears do (generic_gemm.h).
__global__ void pdl_node() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The hold: FFMAs on every SM until `ns` after each CTA started; touches no memory but the sink
// on a chance match.
__global__ void hold(long long ns, float* sink) {
    const long long end = globaltimer_ns() + ns;
    float x             = threadIdx.x * 1e-3f;
    do {
#pragma unroll
        for (int i = 0; i < 256; ++i) x = fmaf(x, 0.999f, 0.001f);
    } while (globaltimer_ns() < end);
    if (x == 12345.f) sink[0] = x;
}

double median(std::vector<double> d) {
    std::sort(d.begin(), d.end());
    return d[d.size() / 2];
}

double us_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<std::string> g_failures;

void check(bool ok, const std::string& what) {
    if (!ok) g_failures.push_back(what);
}

void fact(const char* name, double value) { std::printf("fact %s %.3f ns\n", name, value); }

// A series of N launches, queued by `enqueue` on the probe's stream.
using Enqueue = std::function<void()>;

struct Probe {
    int sms             = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    float* sink    = nullptr;
    double host_us = 0; // host time of the last enqueue

    void hold_for(long long ns) {
        hold<<<sms, kFloorThreads, 0, stream>>>(ns, sink);
        LAUNCHED();
    }

    // One timing of a series (us): the hold, the first event, the series, the second event; false
    // in `ahead` when the hold had ended before the series was queued.
    double time_once(const Enqueue& enqueue, bool& ahead) {
        hold_for(kHoldNs);
        CHECK(cudaEventRecord(e0, stream));
        const auto h0 = std::chrono::steady_clock::now();
        enqueue();
        host_us = us_since(h0);
        CHECK(cudaEventRecord(e1, stream));
        const cudaError_t queued = cudaEventQuery(e0);
        (void)cudaGetLastError(); // not-ready is the answer, not an error of the next call
        ahead = queued == cudaErrorNotReady;
        CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        return ms * 1e3;
    }
};

// A kind's slope (ns a launch), or no fact and why. `host_behind`: the reason is that the host
// had not queued a series when its hold ended.
struct Slope {
    double per = -1.0;
    std::string why;
    bool host_behind = false;
};

// The slope of a kind over its series. `series[i]` launches each, `enqueue[i]` queues them.
Slope slope(Probe& pr, const char* name, const int (&series)[3],
            const std::vector<Enqueue>& enqueue) {
    for (int i = 0; i < 3; ++i) { // warm: the code, the graph's upload, the clocks
        enqueue[i]();
        CHECK(cudaStreamSynchronize(pr.stream));
    }
    std::vector<double> t[3], mid_of, per_of;
    double host_max[3] = {0, 0, 0};
    bool all_ahead     = true;
    for (int r = 0; r < kRepeats; ++r) {
        for (int i = 0; i < 3; ++i) {
            bool ahead = false;
            t[i].push_back(pr.time_once(enqueue[i], ahead));
            all_ahead   = all_ahead && ahead;
            host_max[i] = std::max(host_max[i], pr.host_us);
        }
        mid_of.push_back((t[1][r] - t[0][r]) * 1e3 / (series[1] - series[0]));
        per_of.push_back((t[2][r] - t[0][r]) * 1e3 / (series[2] - series[0]));
    }
    const double per = median(per_of), mid = median(mid_of);
    const auto span = [](const std::vector<double>& v) {
        return std::to_string(*std::min_element(v.begin(), v.end())) + ".." +
               std::to_string(*std::max_element(v.begin(), v.end()));
    };
    std::printf("note %s: medians over %d timings %.3f us (N = %d; %s), %.3f us (N = %d), %.3f us "
                "(N = %d; %s); median slope %.1f ns over N = %d, %.1f over %d (%s); the host "
                "queued the series in at most %.1f, %.1f, %.1f us\n",
                name, kRepeats, median(t[0]), series[0], span(t[0]).c_str(), median(t[1]),
                series[1], median(t[2]), series[2], span(t[2]).c_str(), mid, series[1], per,
                series[2], span(per_of).c_str(), host_max[0], host_max[1], host_max[2]);
    Slope s;
    if (!all_ahead) {
        s.why = "the host had not queued a series when its hold ended (at most " +
                std::to_string(host_max[2]) + " us for the longest, the hold " +
                std::to_string(kHoldNs / 1000) + " us)";
        s.host_behind = true;
    } else if (per <= 0) {
        s.why = "the longest series took no longer than one launch (median slope " +
                std::to_string(per) + " ns)";
    } else if (std::abs(mid - per) > std::max(kLinearShare * per, kLinearNs)) {
        s.why = "the slope moves with the series (" + std::to_string(mid) + " ns over " +
                std::to_string(series[1]) + ", " + std::to_string(per) + " over " +
                std::to_string(series[2]) + ")";
    } else if (per < kFactMinNs || per > kFactMaxNs) {
        s.why = std::to_string(per) + " ns is outside the probe's range";
    } else {
        s.per = per;
    }
    return s;
}

// The edges of a graph, with their data (CUDA 12.3+: the _v2 call before CUDA 13).
cudaError_t graph_edges(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to,
                        cudaGraphEdgeData* data, size_t* count) {
#if CUDART_VERSION >= 13000
    return cudaGraphGetEdges(g, from, to, data, count);
#else
    return cudaGraphGetEdges_v2(g, from, to, data, count);
#endif
}

// A graph of `n` launches captured from the probe's stream (a linear chain), shown to `inspect`
// (where given) before it is instantiated and uploaded; nullptr with the error in `why` when the
// capture or the instantiation is refused.
cudaGraphExec_t capture(Probe& pr, int n, const std::function<cudaError_t()>& launch,
                        const std::function<void(cudaGraph_t, int)>& inspect, std::string& why) {
    cudaGraph_t graph = nullptr;
    CHECK(cudaStreamBeginCapture(pr.stream, cudaStreamCaptureModeThreadLocal));
    cudaError_t err = cudaSuccess;
    for (int i = 0; i < n && err == cudaSuccess; ++i) err = launch();
    const cudaError_t end = cudaStreamEndCapture(pr.stream, &graph);
    (void)cudaGetLastError();
    if (err == cudaSuccess) err = end;
    if (err == cudaSuccess && inspect) inspect(graph, n);
    cudaGraphExec_t exec = nullptr;
    if (err == cudaSuccess) err = cudaGraphInstantiate(&exec, graph, 0);
    if (err == cudaSuccess) err = cudaGraphUpload(exec, pr.stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(pr.stream);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    (void)cudaGetLastError();
    if (err != cudaSuccess) {
        why = cudaGetErrorString(err);
        if (exec != nullptr) (void)cudaGraphExecDestroy(exec);
        return nullptr;
    }
    return exec;
}

// How long cudaGraphLaunch of a graph keeps the host, on an idle stream and behind a hold: a
// driver that returns only when the stream's earlier work has ended shows the hold here.
void host_note(Probe& pr, const char* name, cudaGraphExec_t exec, int n) {
    CHECK(cudaStreamSynchronize(pr.stream));
    auto h0 = std::chrono::steady_clock::now();
    CHECK(cudaGraphLaunch(exec, pr.stream));
    const double idle = us_since(h0);
    CHECK(cudaStreamSynchronize(pr.stream));
    pr.hold_for(kHoldNs);
    h0 = std::chrono::steady_clock::now();
    CHECK(cudaGraphLaunch(exec, pr.stream));
    const double behind = us_since(h0);
    CHECK(cudaStreamSynchronize(pr.stream));
    std::printf("note %s: cudaGraphLaunch of the %d-node graph kept the host %.1f us on an idle "
                "stream, %.1f us behind a %lld ms hold\n",
                name, n, idle, behind, kHoldNs / 1'000'000);
}

// The slope of a graph chain whose node `launch` queues.
Slope graph_slope(Probe& pr, const char* name, const std::function<cudaError_t()>& launch,
                  const std::function<void(cudaGraph_t, int)>& inspect = {}) {
    std::vector<cudaGraphExec_t> execs;
    for (int n : kGraphSeries) {
        Slope refused;
        cudaGraphExec_t e = capture(pr, n, launch, inspect, refused.why);
        if (e == nullptr) {
            for (cudaGraphExec_t x : execs) CHECK(cudaGraphExecDestroy(x));
            refused.why = "the graph of " + std::to_string(n) + " nodes: " + refused.why;
            return refused;
        }
        execs.push_back(e);
    }
    host_note(pr, name, execs.back(), kGraphSeries[2]);
    std::vector<Enqueue> enqueue;
    for (cudaGraphExec_t e : execs)
        enqueue.push_back([&pr, e] { CHECK(cudaGraphLaunch(e, pr.stream)); });
    const Slope s = slope(pr, name, kGraphSeries, enqueue);
    for (cudaGraphExec_t e : execs) CHECK(cudaGraphExecDestroy(e));
    return s;
}

// A kind every card owes: its fact, or a failed self-check.
void owed(const char* name, const Slope& s) {
    if (s.per > 0) {
        fact(name, s.per);
    } else {
        check(false, std::string(name) + ": " + s.why);
    }
}

} // namespace

int main() {
    // Before the first CUDA call: every kernel of the module is loaded when the context is made,
    // so no capture or timed series meets a first-launch module load (as isa_probe.cu does).
#ifdef _WIN32
    _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
#endif
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int driver = 0, runtime = 0;
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    const int cc = p.major * 10 + p.minor;
    char pci[32] = {};
    CHECK(cudaDeviceGetPCIBusId(pci, sizeof pci, 0));
    std::printf("probe launch_probe 2\n");
    std::printf("card %s\n", p.name);
    std::printf("cc %d\n", cc);
    std::printf("sms %d\n", p.multiProcessorCount);
    std::printf("driver %d\n", driver);
    std::printf("runtime %d\n", runtime);
    std::printf("nvcc %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
                __CUDACC_VER_BUILD__);
    std::printf("note device 0 is the GPU at PCI %s\n", pci);
    std::printf("note timer: CUDA events (a 1.024 us tick on the consumer boards and the GV100); "
                "a fact is the median over %d repeats of the slope between series of %d and %d "
                "graph nodes, or %d and %d stream launches, each series queued behind a %lld ms "
                "hold of FFMAs on every SM, a %lld ms hold before the first kind\n",
                kRepeats, kGraphSeries[0], kGraphSeries[2], kStreamSeries[0], kStreamSeries[2],
                kHoldNs / 1'000'000, kWarmNs / 1'000'000);

    Probe pr;
    pr.sms = p.multiProcessorCount;
    CHECK(cudaStreamCreateWithFlags(&pr.stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&pr.e0));
    CHECK(cudaEventCreate(&pr.e1));
    CHECK(cudaMalloc(&pr.sink, sizeof(float)));
    pr.hold_for(kWarmNs);
    CHECK(cudaStreamSynchronize(pr.stream));

    // node_ns: one warp's empty kernel a node.
    owed("node_ns", graph_slope(pr, "node_ns", [&] {
             empty_node<<<1, kNodeThreads, 0, pr.stream>>>();
             return cudaGetLastError();
         }));

    // node_pdl_ns: the same node with the programmatic serialization attribute. Asked for once
    // on the stream first: a driver that refuses the attribute here would refuse the capture.
    const auto pdl_launch = [&] {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{};
        config.gridDim  = dim3(1);
        config.blockDim = dim3(kNodeThreads);
        config.stream   = pr.stream;
        config.attrs    = &attribute;
        config.numAttrs = 1;
        return cudaLaunchKernelEx(&config, pdl_node);
    };
    // The longest graph's edges: a chain of N nodes has N - 1, programmatic where the capture
    // made them so; on sm_90+ all of them must be, or the series is no PDL chain.
    std::string edge_fault;
    const auto edges = [&](cudaGraph_t g, int n) {
        if (n != kGraphSeries[2]) return;
        size_t count = 0;
        if (graph_edges(g, nullptr, nullptr, nullptr, &count) != cudaSuccess) {
            (void)cudaGetLastError();
            edge_fault = "the graph's edges could not be read";
            return;
        }
        std::vector<cudaGraphNode_t> from(count), to(count);
        std::vector<cudaGraphEdgeData> data(count);
        if (graph_edges(g, from.data(), to.data(), data.data(), &count) != cudaSuccess) {
            (void)cudaGetLastError();
            edge_fault = "the graph's edges could not be read";
            return;
        }
        size_t programmatic = 0;
        for (size_t i = 0; i < count; ++i)
            programmatic += data[i].type == cudaGraphDependencyTypeProgrammatic;
        std::printf("note node_pdl_ns: the %d-node graph has %zu edges, %zu programmatic\n", n,
                    count, programmatic);
        if (count != static_cast<size_t>(n - 1) || programmatic != count)
            edge_fault = "the " + std::to_string(n) + "-node graph has " + std::to_string(count) +
                         " edges, " + std::to_string(programmatic) + " programmatic, not " +
                         std::to_string(n - 1) + " programmatic ones";
    };
    cudaError_t accepted = pdl_launch();
    if (accepted == cudaSuccess) accepted = cudaStreamSynchronize(pr.stream);
    (void)cudaGetLastError();
    Slope pdl;
    if (accepted != cudaSuccess) {
        pdl.why = std::string("the driver refuses the attribute: ") + cudaGetErrorString(accepted);
    } else {
        pdl = graph_slope(pr, "node_pdl_ns", pdl_launch, edges);
        if (cc >= 90 && !edge_fault.empty()) {
            pdl.per = -1.0;
            pdl.why = edge_fault;
        }
    }
    if (pdl.per > 0) {
        fact("node_pdl_ns", pdl.per);
        std::printf("note node_pdl_ns: %s\n",
                    cc >= 90 ? "griddepcontrol.wait at the node's start, launch_dependents at its "
                               "end"
                             : "the attribute alone (griddepcontrol is sm_90+; the engine does "
                               "not set the attribute below it)");
    } else {
        std::printf("note node_pdl_ns not measured: %s\n", pdl.why.c_str());
        // The engine's decode chain on sm_90+ is this node: there it is owed. Below sm_90 a
        // driver that refuses the attribute, or keeps the host until the stream's earlier work
        // has ended (no series can be queued ahead), leaves the fact unknown; any other refusal
        // is a fault of the run.
        const bool unmeasurable =
            accepted != cudaSuccess || pdl.host_behind || pdl.why.rfind("the graph of ", 0) == 0;
        check(cc < 90 && unmeasurable, "node_pdl_ns: " + pdl.why);
    }

    // stream_ns: the one-warp kernel on the stream, from a backlog.
    {
        std::vector<Enqueue> enqueue;
        for (int n : kStreamSeries) {
            enqueue.push_back([&pr, n] {
                for (int i = 0; i < n; ++i) empty_node<<<1, kNodeThreads, 0, pr.stream>>>();
                LAUNCHED();
            });
        }
        owed("stream_ns", slope(pr, "stream_ns", kStreamSeries, enqueue));
        std::printf("note stream_ns: the host queued %d launches in %.1f us (the machine's CPU, "
                    "not a fact of the card)\n",
                    kStreamSeries[2], pr.host_us);
    }

    // floor_ns_C: empty grids of C CTAs of four warps, graph nodes of a chain.
    const struct {
        const char* name;
        int ctas;
    } floors[] = {{"floor_ns_1", 1},
                  {"floor_ns_sms", pr.sms},
                  {"floor_ns_4sms", 4 * pr.sms},
                  {"floor_ns_1024", 1024},
                  {"floor_ns_4096", 4096}};

    for (const auto& f : floors) {
        const int ctas = f.ctas;
        owed(f.name, graph_slope(pr, f.name, [&pr, ctas] {
                 empty_node<<<ctas, kFloorThreads, 0, pr.stream>>>();
                 return cudaGetLastError();
             }));
        std::printf("note %s: %d CTAs of %d threads\n", f.name, ctas, kFloorThreads);
    }

    CHECK(cudaFree(pr.sink));
    CHECK(cudaEventDestroy(pr.e0));
    CHECK(cudaEventDestroy(pr.e1));
    CHECK(cudaStreamDestroy(pr.stream));
    if (g_failures.empty()) {
        std::printf("selfcheck ok\n");
    } else {
        for (const std::string& f : g_failures) std::printf("selfcheck FAIL %s\n", f.c_str());
    }
    std::printf("end\n");
    return g_failures.empty() ? 0 : 3;
}
