// launch_probe: what a launch costs on one card, the launch facts of its architecture (#613
// PR-8b). Four kinds, each the cost of one more launch in a long back-to-back series of empty
// kernels, so the series' fixed parts (the graph launch, the first node, the events) cancel:
//   node_ns      an empty kernel of one CTA (one warp) as a node of a CUDA graph, each node
//                depending on the one before (a captured stream's linear chain): graph replay,
//                how the engine runs decode;
//   node_pdl_ns  the same node launched with the programmatic stream serialization attribute,
//                its kernel waiting on the one before (griddepcontrol.wait) at its start and
//                releasing the next (griddepcontrol.launch_dependents) at its end: the Blackwell
//                decode chain. griddepcontrol is sm_90+; below it the kernel is empty and the
//                attribute alone is timed (core/pdl.cuh: the engine does not set it there), and
//                where the driver refuses the attribute the fact is not printed (a note says
//                why);
//   stream_ns    the one-CTA kernel launched on a stream (<<<>>>) from a backlog: the host has
//                queued them all before the first may start, so the figure is the GPU's own
//                cost per stream launch, not the host's submit rate (that is the machine's CPU
//                and only a note);
//   floor_ns_C   an empty grid of C CTAs of 128 threads (the decode linears' four warps) as a
//                graph node of a chain, C = 1, SMs, 4 x SMs and 1024: what a call of C empty
//                CTAs costs, so floor_ns_C - floor_ns_1 is the CTA dispatch ramp of a C-CTA grid
//                (#850's grids are 256 to 3072 CTAs, 1024 on 170 SMs the typical one).
//
// Timer. CUDA events, whose tick is 1.024 us on the consumer boards and the GV100
// (route_tune_tool.cpp): one launch of about 1 us cannot be timed by them, a series can. Each
// timing is a series of N launches between two events; a fact is the slope between the medians
// of kRepeats timings of N = 1 and of the longest series, (t_N - t_1) / (N - 1), N - 1 = 1023
// graph nodes or 255 stream launches: two ticks over that many launches is 2 ns (graphs) or
// 8 ns (stream) of a fact. The middle series (256 nodes, 64 launches) gives the same slope over a
// shorter span, and a slope that moves with N is refused (self-check): the cost of a launch is a
// constant of the series or it is no fact.
//
// Every series runs behind a hold: a kernel of FFMAs on every SM that ends a fixed time after it
// starts (20 ms), the first event recorded after it, the series queued while it runs. So the
// series starts on a chip at its working clock with the whole series already queued, and the
// timing is the GPU's alone; a series the host had not finished queuing when the hold ended is
// refused (self-check). The repeats of the series of every N alternate, so a drift of the clock
// over the run reaches all of them alike.
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
constexpr long long kHoldNs   = 20'000'000; // the hold kernel's span
constexpr int kNodeThreads    = 32;         // the node kernels: one warp
constexpr int kFloorThreads   = 128;        // the floor grids: four warps a CTA
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

    // One timing of a series (us): the hold, the first event, the series, the second event; false
    // in `ahead` when the hold had ended before the series was queued.
    double time_once(const Enqueue& enqueue, bool& ahead) {
        hold<<<sms, kFloorThreads, 0, stream>>>(kHoldNs, sink);
        LAUNCHED();
        CHECK(cudaEventRecord(e0, stream));
        const auto h0 = std::chrono::steady_clock::now();
        enqueue();
        host_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0)
                      .count();
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

// The slope of a kind over its series (ns a launch), or a negative number when a series was not
// queued ahead of its hold. `series[i]` launches each, `enqueue[i]` queues them.
double slope(Probe& pr, const char* name, const int (&series)[3],
             const std::vector<Enqueue>& enqueue) {
    for (int i = 0; i < 3; ++i) { // warm: the code, the graph's upload, the clocks
        enqueue[i]();
        CHECK(cudaStreamSynchronize(pr.stream));
    }
    std::vector<double> t[3];
    double host_last = 0;
    bool all_ahead   = true;
    for (int r = 0; r < kRepeats; ++r) {
        for (int i = 0; i < 3; ++i) {
            bool ahead = false;
            t[i].push_back(pr.time_once(enqueue[i], ahead));
            all_ahead = all_ahead && ahead;
            if (i == 2) host_last = pr.host_us;
        }
    }
    check(all_ahead, std::string(name) + ": the host had not queued a series when its hold ended");
    const double m[3] = {median(t[0]), median(t[1]), median(t[2])};
    const double per  = (m[2] - m[0]) * 1e3 / (series[2] - series[0]);
    const double mid  = (m[1] - m[0]) * 1e3 / (series[1] - series[0]);
    check(std::abs(mid - per) <= std::max(kLinearShare * per, kLinearNs),
          std::string(name) + ": the slope moves with the series (" + std::to_string(mid) +
              " ns over " + std::to_string(series[1]) + ", " + std::to_string(per) + " over " +
              std::to_string(series[2]) + ")");
    check(per >= kFactMinNs && per <= kFactMaxNs, std::string(name) + " outside its range");
    const auto span = [](const std::vector<double>& v) {
        return std::to_string(*std::min_element(v.begin(), v.end())) + ".." +
               std::to_string(*std::max_element(v.begin(), v.end()));
    };
    std::printf("note %s: medians over %d timings %.3f us (N = %d; %s), %.3f us (N = %d), %.3f us "
                "(N = %d; %s); slope %.1f ns over N = %d, %.1f over %d; host queued the longest "
                "series in %.1f us\n",
                name, kRepeats, m[0], series[0], span(t[0]).c_str(), m[1], series[1], m[2],
                series[2], span(t[2]).c_str(), mid, series[1], per, series[2], host_last);
    return all_ahead ? per : -1.0;
}

// A graph of `n` launches captured from the probe's stream (a linear chain), instantiated and
// uploaded; nullptr with the error in `why` when the capture or the instantiation is refused.
cudaGraphExec_t capture(Probe& pr, int n, const std::function<cudaError_t()>& launch,
                        std::string& why) {
    cudaGraph_t graph = nullptr;
    CHECK(cudaStreamBeginCapture(pr.stream, cudaStreamCaptureModeThreadLocal));
    cudaError_t err = cudaSuccess;
    for (int i = 0; i < n && err == cudaSuccess; ++i) err = launch();
    const cudaError_t end = cudaStreamEndCapture(pr.stream, &graph);
    (void)cudaGetLastError();
    if (err == cudaSuccess) err = end;
    cudaGraphExec_t exec = nullptr;
    if (err == cudaSuccess) err = cudaGraphInstantiate(&exec, graph, 0);
    if (err == cudaSuccess) err = cudaGraphUpload(exec, pr.stream);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    (void)cudaGetLastError();
    if (err != cudaSuccess) {
        why = cudaGetErrorString(err);
        if (exec != nullptr) (void)cudaGraphExecDestroy(exec);
        return nullptr;
    }
    return exec;
}

// The slope of a graph chain whose node `launch` queues; a negative number when refused (`why`).
double graph_slope(Probe& pr, const char* name, const std::function<cudaError_t()>& launch,
                   std::string& why) {
    std::vector<cudaGraphExec_t> execs;
    for (int n : kGraphSeries) {
        cudaGraphExec_t e = capture(pr, n, launch, why);
        if (e == nullptr) {
            for (cudaGraphExec_t x : execs) CHECK(cudaGraphExecDestroy(x));
            return -1.0;
        }
        execs.push_back(e);
    }
    std::vector<Enqueue> enqueue;
    for (cudaGraphExec_t e : execs)
        enqueue.push_back([&pr, e] { CHECK(cudaGraphLaunch(e, pr.stream)); });
    const double per = slope(pr, name, kGraphSeries, enqueue);
    for (cudaGraphExec_t e : execs) CHECK(cudaGraphExecDestroy(e));
    if (per < 0) why = "the host had not queued a series when its hold ended";
    return per;
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
    std::printf("probe launch_probe 1\n");
    std::printf("card %s\n", p.name);
    std::printf("cc %d\n", cc);
    std::printf("sms %d\n", p.multiProcessorCount);
    std::printf("driver %d\n", driver);
    std::printf("runtime %d\n", runtime);
    std::printf("nvcc %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
                __CUDACC_VER_BUILD__);
    std::printf("note device 0 is the GPU at PCI %s\n", pci);
    std::printf("note timer: CUDA events (a 1.024 us tick on the consumer boards and the GV100); "
                "a fact is the slope between the medians of %d timings of series of %d and %d "
                "graph nodes, or %d and %d stream launches, each series queued behind a %lld ms "
                "hold of FFMAs on every SM\n",
                kRepeats, kGraphSeries[0], kGraphSeries[2], kStreamSeries[0], kStreamSeries[2],
                kHoldNs / 1'000'000);

    Probe pr;
    pr.sms = p.multiProcessorCount;
    CHECK(cudaStreamCreateWithFlags(&pr.stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&pr.e0));
    CHECK(cudaEventCreate(&pr.e1));
    CHECK(cudaMalloc(&pr.sink, sizeof(float)));
    std::string why;

    // node_ns: one warp's empty kernel a node.
    const double node = graph_slope(
        pr, "node_ns",
        [&] {
            empty_node<<<1, kNodeThreads, 0, pr.stream>>>();
            return cudaGetLastError();
        },
        why);
    check(node > 0, "node_ns: " + why);
    if (node > 0) fact("node_ns", node);

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
    cudaError_t accepted = pdl_launch();
    if (accepted == cudaSuccess) accepted = cudaStreamSynchronize(pr.stream);
    (void)cudaGetLastError();
    double node_pdl = -1.0;
    if (accepted != cudaSuccess) {
        why = cudaGetErrorString(accepted);
    } else {
        node_pdl = graph_slope(pr, "node_pdl_ns", pdl_launch, why);
    }
    if (node_pdl > 0) {
        fact("node_pdl_ns", node_pdl);
        std::printf("note node_pdl_ns: %s\n",
                    cc >= 90 ? "griddepcontrol.wait at the node's start, launch_dependents at its "
                               "end"
                             : "the attribute alone (griddepcontrol is sm_90+; the engine does "
                               "not set the attribute below it)");
    } else {
        std::printf("note node_pdl_ns not measured: %s\n", why.c_str());
        // The engine's decode chain on sm_90+ is this node: there it is owed.
        check(cc < 90, "node_pdl_ns: " + why);
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
        const double stream = slope(pr, "stream_ns", kStreamSeries, enqueue);
        if (stream > 0) fact("stream_ns", stream);
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
                  {"floor_ns_1024", 1024}};

    for (const auto& f : floors) {
        const int ctas   = f.ctas;
        const double per = graph_slope(
            pr, f.name,
            [&pr, ctas] {
                empty_node<<<ctas, kFloorThreads, 0, pr.stream>>>();
                return cudaGetLastError();
            },
            why);
        check(per > 0, std::string(f.name) + ": " + why);
        if (per > 0) fact(f.name, per);
        std::printf("note %s: %d CTAs of %d threads\n", f.name, ctas, kFloorThreads);
    }

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
