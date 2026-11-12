// The launch law of ggml's matrix kernel on one card (#785; DECISIONS 78; core/generation.h
// MmqLaunchLaw): what a unit, a stream-k split and a launch cost, measured with the production
// kernel and its production launch template, the column tile and the CTA count forced
// (mmq_launch_microbench_launch.cuh; no kernel patch). BENCH ONLY: built with
// -DNINFERNO_BUILD_BENCHMARKS=ON as ninferno_mmq_launch_microbench; run it through
// tools/bench/mmq_launch_law.py run, which refuses a run with another compute process on the GPU
// and records the clocks nvidia-smi sees through the run.
//
// What it measures, per format (q4_K, q5_K, q6_K) and column tile J (16 32 64 80 96 128), on a card
// of S SMs (a unit: one 128-row x J-column tile over 256 K values, mmq_columns::kUnitK):
//   law       a tile a CTA, floor(S / 2) row tiles x 2 column tiles: one wave at one CTA a SM, the
//             launch DECISIONS 78 measured Turing's unit_clocks on (two column tiles, so each
//             weight tile is read by two CTAs);
//   wave1     a tile a CTA, S row tiles x 1 column tile: one wave, every weight tile read once;
//   full      a tile a CTA, 2 S row tiles x 2 column tiles: four waves (the law's waves x steps);
//   streamk   floor(S / 2) row tiles x 1 column tile on S CTAs: stream-k, tiles % CTAs != 0, so
//             the fixup kernel runs (the launch the law prices with stream_k_unit and its fixup);
//   halfwave  the streamk shape as a tile a CTA (floor(S / 2) CTAs): what the split is against.
// Each at K = 2048, 4096, 8192 (full: 4096, 8192), so that a unit is the slope of a call's time
// over its longest CTA's units and the launch's fixed part the intercept (mmq_launch_law.py
// derive). Beside them: the fixup kernel alone (the streamk launch at the largest K), and the
// per-launch node cost of an empty kernel, back to back on a stream and as the nodes of a graph,
// each with and without the programmatic-serialization attribute (core/pdl.cuh).
//
// Protocol: one sample is CALLS back-to-back calls between two events after an L2 flush (a
// cudaMemset of 2 x the L2), the calls cycling through copies of the weights that together hold at
// least 3 x the L2, so a call reads its weights from DRAM as a prefill linear does; CALLS is sized
// so a sample takes about --sample-ms. Every configuration runs --samples samples in each of
// --passes passes (every second pass in reverse order, so a drift in clock or temperature does not
// line up with J or K), after a warm-up of --warmup-ms on the largest launch. The JSON keeps every
// sample; mmq_launch_law.py takes the medians. Clocks are not locked: the law is stated at the
// board's reference clock (core/machine_facts.h), and the clocks the card ran at are recorded by
// the driver script, as tools/bench/isa_probe and warp_inflight do.
#include "mmq_launch_microbench.h"

#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/mmq_columns.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace mb = ninferno::bench::mmq_launch;
using ninferno::ops::ggml_mmq::Planes;

namespace {

constexpr const char* kSchema = "ninferno-mmq-launch-microbench/1";
constexpr int kTileRows       = ninferno::mmq_columns::kTile;  // I of every K-quant tile
constexpr int kUnitK          = ninferno::mmq_columns::kUnitK; // MMQ_ITER_K

#define CK(x)                                                                                      \
    do {                                                                                           \
        const cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,                         \
                         cudaGetErrorString(e_));                                                  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// --- options ---

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::size_t at = 0;
    while (at <= s.size()) {
        const std::size_t comma = s.find(',', at);
        std::string item =
            s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (!item.empty()) out.push_back(item);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return out;
}

std::vector<int> split_ints(const std::string& s) {
    std::vector<int> out;
    for (const std::string& item : split(s)) out.push_back(std::atoi(item.c_str()));
    return out;
}

const char* const kGrids[] = {"law", "wave1", "full", "streamk", "halfwave"};

struct Options {
    std::string out;
    std::vector<std::string> formats{"q4_K", "q5_K", "q6_K"};
    std::vector<int> widths{16, 32, 64, 80, 96, 128};
    std::vector<std::string> grids{std::begin(kGrids), std::end(kGrids)};
    std::vector<int> ks{2048, 4096, 8192};
    std::vector<int> full_ks{4096, 8192};
    int samples       = 9;
    int passes        = 2;
    double sample_ms  = 40;
    double warmup_ms  = 3000;
    int node_launches = 1000;
    bool dry_run      = false;
    int sms           = 0;   // --dry-run: the SMs of the board planned for
    double l2_mib     = 0.0; // --dry-run: its L2
};

void usage(std::FILE* to) {
    std::fprintf(
        to,
        "usage: ninferno_mmq_launch_microbench [--out FILE] [--formats q4_K,q5_K,q6_K]\n"
        "    [--widths 16,32,64,80,96,128] [--grids law,wave1,full,streamk,halfwave]\n"
        "    [--ks 2048,4096,8192] [--full-ks 4096,8192] [--samples 9] [--passes 2]\n"
        "    [--sample-ms 40] [--warmup-ms 3000] [--node-launches 1000]\n"
        "    [--dry-run [--sms N] [--l2-mib M]] [--help]\n"
        "\n"
        "Times ggml's matrix kernel with the column tile and the CTA count forced, for the launch\n"
        "law of core/generation.h (MmqLaunchLaw, DECISIONS 78), and writes one JSON document\n"
        "(stdout, or --out; the plan and the progress go to stderr). Run it through\n"
        "tools/bench/mmq_launch_law.py run on an idle GPU: the script refuses a run with another\n"
        "compute process on the GPU and records the clocks.\n"
        "--dry-run prints the plan (launches, buffers, expected run time) without touching a GPU,\n"
        "for a board of --sms SMs (default 60) and --l2-mib MiB of L2 (default 48).\n");
}

void parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value          = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage(stdout);
            std::exit(0);
        } else if (a == "--out") {
            o.out = value();
        } else if (a == "--formats") {
            o.formats = split(value());
        } else if (a == "--widths") {
            o.widths = split_ints(value());
        } else if (a == "--grids") {
            o.grids = split(value());
        } else if (a == "--ks") {
            o.ks = split_ints(value());
        } else if (a == "--full-ks") {
            o.full_ks = split_ints(value());
        } else if (a == "--samples") {
            o.samples = std::atoi(value().c_str());
        } else if (a == "--passes") {
            o.passes = std::atoi(value().c_str());
        } else if (a == "--sample-ms") {
            o.sample_ms = std::atof(value().c_str());
        } else if (a == "--warmup-ms") {
            o.warmup_ms = std::atof(value().c_str());
        } else if (a == "--node-launches") {
            o.node_launches = std::atoi(value().c_str());
        } else if (a == "--dry-run") {
            o.dry_run = true;
        } else if (a == "--sms") {
            o.sms = std::atoi(value().c_str());
        } else if (a == "--l2-mib") {
            o.l2_mib = std::atof(value().c_str());
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    for (const std::string& g : o.grids) {
        if (std::find_if(std::begin(kGrids), std::end(kGrids),
                         [&](const char* n) { return g == n; }) == std::end(kGrids)) {
            throw std::invalid_argument("unknown grid " + g);
        }
    }
    for (const int j : o.widths) {
        if (ninferno::mmq_columns::work_index(j) < 0) {
            throw std::invalid_argument("width " + std::to_string(j) +
                                        " is not one of the work table's (16 32 64 80 96 128)");
        }
    }
    for (const std::vector<int>* ks : {&o.ks, &o.full_ks}) {
        for (const int k : *ks) {
            if (k <= 0 || k % kUnitK != 0) {
                throw std::invalid_argument("K must be a positive multiple of 256");
            }
        }
    }
    if (o.samples < 3 || o.passes < 1 || !(o.sample_ms > 0) || o.node_launches < 10) {
        throw std::invalid_argument("--samples >= 3, --passes >= 1, --sample-ms > 0, "
                                    "--node-launches >= 10");
    }
}

// --- formats ---

struct Format {
    const mb::FormatOps* ops = nullptr;
    // Bytes of one 256-value block in each plane: the blocks kept whole (Q4_K, Q5_K) have one.
    int code_bytes  = 0;
    int high_bytes  = 0;
    int scale_bytes = 0;

    [[nodiscard]] int block_bytes() const { return code_bytes + high_bytes + scale_bytes; }
};

Format format_of(const std::string& name) {
    using ninferno::QType;
    if (name == "q4_K") return {&mb::kQ4K, 144, 0, 0}; // sizeof(block_q4_K)
    if (name == "q5_K") return {&mb::kQ5K, 176, 0, 0}; // sizeof(block_q5_K)
    if (name == "q6_K") {
        // Stored as ggml_planes_v1 (core/format_facts.h): a code, a high and a scale plane.
        constexpr ninferno::GgmlPlaneBytes p = ninferno::ggml_plane_bytes(QType::GGML_Q6_K);
        static_assert(p.values == 256 && p.code + p.high + p.scale == 210);
        return {&mb::kQ6K, int(p.code), int(p.high), int(p.scale)};
    }
    throw std::invalid_argument("unknown format " + name + " (q4_K, q5_K, q6_K)");
}

// --- the launches ---

struct Grid {
    std::string name;
    int row_tiles = 0, column_tiles = 0, blocks = 0;
};

Grid grid_of(const std::string& name, int sms) {
    const int half = std::max(1, sms / 2);
    if (name == "law") return {name, half, 2, 2 * half};
    if (name == "wave1") return {name, sms, 1, sms};
    if (name == "full") return {name, 2 * sms, 2, 4 * sms};
    if (name == "streamk") return {name, half, 1, sms};
    if (name == "halfwave") return {name, half, 1, half};
    throw std::invalid_argument("unknown grid " + name);
}

// ggml's partition of a forced launch (mmq.cuh mul_mat_q; mmq_columns::launch_of with the CTAs
// given): `blocks` equal to the tiles is a tile a CTA, in waves of one CTA a SM; any other count
// (at most one CTA a SM) shares the W = tiles x steps units, CTA b taking [b W / B, (b + 1) W / B).
struct Partition {
    int tiles = 0, blocks = 0, steps = 0, units = 0;
    bool split = false, fixup = false;
    int partials = 0, split_tiles = 0;
    double fixup_bytes = 0;
};

Partition partition_of(int rows, int k, int columns, int j, int blocks, int sms) {
    Partition p;
    p.tiles  = rows / kTileRows * ((columns + j - 1) / j);
    p.steps  = k / kUnitK;
    p.blocks = blocks;
    if (blocks == p.tiles) {
        p.units = (p.tiles + sms - 1) / sms * p.steps;
        return p;
    }
    if (blocks > sms) throw std::invalid_argument("a split launch takes at most one CTA a SM");
    const long long work = static_cast<long long>(p.tiles) * p.steps;
    p.units              = static_cast<int>((work + blocks - 1) / blocks);
    p.split              = true;
    p.fixup              = p.tiles % blocks != 0;
    long long last       = -1;
    for (long long b = 1; b < blocks; ++b) {
        const long long end = b * work / blocks; // CTA b - 1's
        if (end % p.steps == 0) continue;
        ++p.partials;
        if (end / p.steps != last) {
            ++p.split_tiles;
            last = end / p.steps;
        }
    }
    // mmq_columns::fixup_bytes: every partial tile read, every output tile it lands in read and
    // written.
    if (p.fixup) p.fixup_bytes = (p.partials + 2.0 * p.split_tiles) * kTileRows * j * 4.0;
    return p;
}

struct Config {
    std::string format;
    Format fmt;
    Grid grid;
    int j = 0, k = 0, rows = 0, columns = 0;
    Partition part;
    std::size_t weight_bytes = 0;
};

std::vector<Config> plan_of(const Options& o, int sms) {
    std::vector<Config> plan;
    for (const std::string& f : o.formats) {
        const Format fmt = format_of(f);
        for (const int j : o.widths) {
            for (const std::string& g : o.grids) {
                const Grid grid = grid_of(g, sms);
                for (const int k : g == "full" ? o.full_ks : o.ks) {
                    Config c;
                    c.format       = f;
                    c.fmt          = fmt;
                    c.grid         = grid;
                    c.j            = j;
                    c.k            = k;
                    c.rows         = grid.row_tiles * kTileRows;
                    c.columns      = grid.column_tiles * j;
                    c.part         = partition_of(c.rows, k, c.columns, j, grid.blocks, sms);
                    c.weight_bytes = std::size_t(c.rows) * (k / kUnitK) * fmt.block_bytes();
                    plan.push_back(c);
                }
            }
        }
    }
    return plan;
}

// The fixup-alone timings: the streamk launch at its largest K, per format and width.
std::vector<std::size_t> fixup_alone_of(const std::vector<Config>& plan) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        const Config& c = plan[i];
        if (c.grid.name != "streamk" || !c.part.fixup) continue;
        bool largest = true;
        for (const Config& d : plan) {
            if (d.grid.name == c.grid.name && d.format == c.format && d.j == c.j && d.k > c.k) {
                largest = false;
            }
        }
        if (largest) out.push_back(i);
    }
    return out;
}

struct Buffers {
    std::size_t pool_bytes = 0, flush_bytes = 0, out_bytes = 0, fixup_bytes = 0;
    std::size_t x_values = 0;
    int max_k = 0, max_columns = 0;
};

Buffers buffers_of(const std::vector<Config>& plan, std::size_t l2_bytes, int sms) {
    Buffers b;
    std::size_t widest = 0;
    for (const Config& c : plan) {
        widest        = std::max(widest, c.weight_bytes);
        b.out_bytes   = std::max(b.out_bytes, std::size_t(c.rows) * c.columns * sizeof(float));
        b.max_k       = std::max(b.max_k, c.k);
        b.max_columns = std::max(b.max_columns, c.columns);
    }
    // The pool holds the largest product's weights and 3 x the L2 more, so floor(pool / w) copies
    // of w bytes hold at least 3 x the L2 at every size.
    b.pool_bytes  = widest + 3 * l2_bytes;
    b.flush_bytes = std::max<std::size_t>(2 * l2_bytes, std::size_t(8) << 20);
    b.fixup_bytes = std::size_t(sms) * 128 * kTileRows * sizeof(float); // CTAs x J 128 x I
    b.x_values    = std::size_t(b.max_k) * b.max_columns;
    return b;
}

int copies_of(const Config& c, const Buffers& b) {
    return static_cast<int>(std::max<std::size_t>(1, b.pool_bytes / c.weight_bytes));
}

double expected_seconds(const Options& o, const std::vector<Config>& plan, std::size_t alone) {
    // Per configuration and pass: the calibration and --samples samples of about --sample-ms,
    // with the flush and the host's own work; the warm-up and the launch-node kinds once.
    const double per = (o.samples + 1) * (o.sample_ms + 1.0) / 1e3 + 0.05;
    return o.warmup_ms / 1e3 + o.passes * (plan.size() + alone) * per + 5.0;
}

void print_plan(std::FILE* to, const Options& o, const std::vector<Config>& plan, const Buffers& b,
                int sms, std::size_t l2) {
    const std::size_t alone = fixup_alone_of(plan).size();
    std::fprintf(to,
                 "plan for %d SMs, %.1f MiB of L2: %zu launches x %d passes x %d samples, %zu "
                 "fixup-alone, 4 launch-node kinds x %d launches\n",
                 sms, l2 / 1048576.0, plan.size(), o.passes, o.samples, alone, o.node_launches);
    std::fprintf(to, "buffers: weights %.1f MiB, flush %.1f MiB, out %.1f MiB, fixup %.1f MiB\n",
                 b.pool_bytes / 1048576.0, b.flush_bytes / 1048576.0, b.out_bytes / 1048576.0,
                 b.fixup_bytes / 1048576.0);
    for (const Config& c : plan) {
        std::fprintf(to,
                     "%-5s J %3d %-8s k %5d: %6d x %4d, %4d tiles on %4d CTAs, %4d units%s%s, "
                     "weights %.1f MiB x %d copies\n",
                     c.format.c_str(), c.j, c.grid.name.c_str(), c.k, c.rows, c.columns,
                     c.part.tiles, c.part.blocks, c.part.units, c.part.split ? ", stream-k" : "",
                     c.part.fixup ? " with fixup" : "", c.weight_bytes / 1048576.0,
                     copies_of(c, b));
    }
    std::fprintf(to, "expected run time: about %.1f min (the build not included)\n",
                 expected_seconds(o, plan, alone) / 60.0);
}

// --- the measurement ---

__global__ void fill_words(std::uint32_t* p, std::size_t words, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < words;
         i += std::size_t(gridDim.x) * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        p[i] = h;
    }
}

__global__ void fill_bf16(__nv_bfloat16* p, std::size_t n) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u;
        h ^= h >> 16;
        p[i] = __float2bfloat16((h & 0xffff) / 32768.0f - 1.0f);
    }
}

__global__ void empty_kernel() {}

double now_s() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

struct Device {
    std::uint8_t* pool  = nullptr;
    std::uint8_t* flush = nullptr;
    float* out          = nullptr;
    void* fixup         = nullptr;
    __nv_bfloat16* x    = nullptr;
    void* act           = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t a = nullptr, b = nullptr;
};

Planes planes_of(const Config& c, const Device& d, int copy) {
    const std::size_t blocks = std::size_t(c.rows) * (c.k / kUnitK);
    std::uint8_t* base       = d.pool + std::size_t(copy) * c.weight_bytes;
    Planes p;
    p.codes = base;
    if (c.fmt.high_bytes > 0) p.high = base + blocks * c.fmt.code_bytes;
    if (c.fmt.scale_bytes > 0) p.scales = base + blocks * (c.fmt.code_bytes + c.fmt.high_bytes);
    return p;
}

struct Timed {
    std::vector<std::vector<double>> pass_us; // per pass, per sample: microseconds a call
    int calls  = 0;
    int copies = 0;
    std::vector<double> t_begin, t_end; // per pass, epoch seconds
};

template <class Call>
std::vector<double> sample(Device& d, const Buffers& b, int samples, int calls, Call&& call,
                           long long& counter) {
    std::vector<double> us;
    for (int s = 0; s < samples; ++s) {
        CK(cudaMemsetAsync(d.flush, s & 0xff, b.flush_bytes, d.stream));
        CK(cudaEventRecord(d.a, d.stream));
        for (int i = 0; i < calls; ++i) call(counter++);
        CK(cudaEventRecord(d.b, d.stream));
        CK(cudaEventSynchronize(d.b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, d.a, d.b));
        us.push_back(1e3 * ms / calls);
    }
    CK(cudaGetLastError());
    return us;
}

template <class Call>
int calibrate(Device& d, double sample_ms, Call&& call, long long& counter) {
    for (int i = 0; i < 2; ++i) call(counter++);
    CK(cudaStreamSynchronize(d.stream));
    CK(cudaEventRecord(d.a, d.stream));
    for (int i = 0; i < 3; ++i) call(counter++);
    CK(cudaEventRecord(d.b, d.stream));
    CK(cudaEventSynchronize(d.b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, d.a, d.b));
    const double per = std::max(1e-4, ms / 3.0);
    return static_cast<int>(std::clamp(std::lround(sample_ms / per), 3L, 5000L));
}

void quantize(const Config& c, Device& d) {
    ninferno::ops::ggml_mmq::quantize_activation(c.fmt.ops->type, d.x, c.k, c.k, c.columns, d.act,
                                                 d.stream);
}

// The per-launch node cost of an empty kernel (one CTA of 32 threads), N launches back to back on
// the stream, or the N nodes of one graph replayed; each with and without the
// programmatic-serialization attribute.
struct NodeCost {
    std::string name;
    std::vector<double> us; // per sample: microseconds a launch
    std::string error;
};

cudaError_t launch_empty(cudaStream_t s, bool pdl) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim  = dim3(1, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.stream   = s;
    cudaLaunchAttribute attr[1]{};
    if (pdl) {
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs                                          = attr;
        cfg.numAttrs                                       = 1;
    }
    return cudaLaunchKernelEx(&cfg, empty_kernel);
}

NodeCost node_cost(Device& d, const Options& o, bool graph, bool pdl) {
    NodeCost r;
    r.name               = std::string(graph ? "graph" : "stream") + (pdl ? "_pdl" : "");
    const int n          = o.node_launches;
    cudaGraphExec_t exec = nullptr;
    if (graph) {
        cudaGraph_t g = nullptr;
        CK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
        cudaError_t status = cudaSuccess;
        for (int i = 0; i < n && status == cudaSuccess; ++i) status = launch_empty(d.stream, pdl);
        const cudaError_t end = cudaStreamEndCapture(d.stream, &g);
        if (status == cudaSuccess) status = end;
        if (status == cudaSuccess) status = cudaGraphInstantiate(&exec, g, 0);
        if (g != nullptr) (void)cudaGraphDestroy(g);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            r.error = cudaGetErrorString(status);
            return r;
        }
    }
    auto batch = [&]() -> cudaError_t {
        if (graph) return cudaGraphLaunch(exec, d.stream);
        for (int i = 0; i < n; ++i) {
            const cudaError_t e = launch_empty(d.stream, pdl);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    };
    cudaError_t status = batch();
    if (status == cudaSuccess) status = cudaStreamSynchronize(d.stream);
    for (int s = 0; s < o.samples && status == cudaSuccess; ++s) {
        CK(cudaEventRecord(d.a, d.stream));
        status = batch();
        CK(cudaEventRecord(d.b, d.stream));
        CK(cudaEventSynchronize(d.b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, d.a, d.b));
        r.us.push_back(1e3 * ms / n);
    }
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        r.error = cudaGetErrorString(status);
        r.us.clear();
    }
    if (exec != nullptr) (void)cudaGraphExecDestroy(exec);
    return r;
}

// --- JSON ---

std::string json_str(const std::string& s) {
    std::string out = "\"";
    for (const char ch : s) {
        if (ch == '"' || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (static_cast<unsigned char>(ch) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
            out += buf;
        } else {
            out += ch;
        }
    }
    return out + "\"";
}

std::string json_num(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.12g", v); // byte counts exact (a 32 GiB board: 11 digits)
    return buf;
}

std::string json_list(const std::vector<double>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + json_num(v[i]);
    return out + "]";
}

std::string json_passes(const Timed& t) {
    std::string out = "[";
    for (std::size_t p = 0; p < t.pass_us.size(); ++p) {
        out += (p ? ", " : "") + json_list(t.pass_us[p]);
    }
    return out + "]";
}

std::string config_json(const Config& c, const Timed& t, int resident, bool skipped) {
    std::string s = "{\"format\": " + json_str(c.format) + ", \"grid\": " + json_str(c.grid.name) +
                    ", \"j\": " + std::to_string(c.j) + ", \"k\": " + std::to_string(c.k) +
                    ", \"rows\": " + std::to_string(c.rows) +
                    ", \"columns\": " + std::to_string(c.columns) +
                    ", \"tiles\": " + std::to_string(c.part.tiles) +
                    ", \"blocks\": " + std::to_string(c.part.blocks) +
                    ", \"steps\": " + std::to_string(c.part.steps) +
                    ", \"units\": " + std::to_string(c.part.units) +
                    ", \"split\": " + (c.part.split ? "true" : "false") +
                    ", \"fixup\": " + (c.part.fixup ? "true" : "false") +
                    ", \"partials\": " + std::to_string(c.part.partials) +
                    ", \"split_tiles\": " + std::to_string(c.part.split_tiles) +
                    ", \"fixup_bytes\": " + json_num(c.part.fixup_bytes) +
                    ", \"weight_bytes\": " + json_num(double(c.weight_bytes));
    if (skipped) return s + ", \"skipped\": \"no tile of this width on this device\"}";
    return s + ", \"resident_ctas_per_sm\": " + std::to_string(resident) +
           ", \"copies\": " + std::to_string(t.copies) + ", \"calls\": " + std::to_string(t.calls) +
           ", \"t_begin\": " + json_list(t.t_begin) + ", \"t_end\": " + json_list(t.t_end) +
           ", \"samples_us\": " + json_passes(t) + "}";
}

template <class Call>
void time_config(Device& d, const Buffers& b, const Options& o, Timed& t, Call&& call) {
    long long counter = 0;
    t.t_begin.push_back(now_s());
    const int calls = calibrate(d, o.sample_ms, call, counter);
    t.calls         = std::max(t.calls, calls);
    t.pass_us.push_back(sample(d, b, o.samples, calls, call, counter));
    t.t_end.push_back(now_s());
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int run(const Options& o) {
    const ninferno::DeviceClass& dc = ninferno::device_class();
    const int sms                   = dc.sms;
    const std::size_t l2            = dc.l2_bytes;
    if (dc.major * 10 + dc.minor < 75) {
        std::fprintf(stderr, "mmq_launch_microbench: ggml's tensor-core kernel needs sm_75 on\n");
        return 2;
    }
    const std::vector<Config> plan = plan_of(o, sms);
    const Buffers b                = buffers_of(plan, l2, sms);
    print_plan(stderr, o, plan, b, sms, l2);

    Device d;
    CK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
    CK(cudaEventCreate(&d.a));
    CK(cudaEventCreate(&d.b));
    CK(cudaMalloc(&d.pool, b.pool_bytes));
    CK(cudaMalloc(&d.flush, b.flush_bytes));
    CK(cudaMalloc(&d.out, b.out_bytes));
    CK(cudaMalloc(&d.fixup, b.fixup_bytes));
    CK(cudaMalloc(&d.x, b.x_values * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d.act, ninferno::ops::ggml_mmq::activation_bytes(b.max_k, b.max_columns)));
    // Random weights: the time of an int8 MMA, a decode and an FP32 scale does not depend on the
    // values (an FP16 scale that reads as a NaN or an infinity included).
    fill_words<<<1024, 256, 0, d.stream>>>(reinterpret_cast<std::uint32_t*>(d.pool),
                                           b.pool_bytes / 4, 0x9e3779b9u);
    fill_bf16<<<256, 256, 0, d.stream>>>(d.x, b.x_values);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(d.stream));

    // A width this device has no tile of is recorded as skipped, not assumed away.
    std::vector<int> resident(plan.size(), 0);
    std::vector<bool> skipped(plan.size(), false);
    for (std::size_t i = 0; i < plan.size(); ++i) {
        skipped[i] = !plan[i].fmt.ops->config(plan[i].j).ok;
    }
    auto forced_of = [](const Config& c) {
        return mb::Forced{c.rows, c.k, c.columns, c.j, c.part.blocks};
    };

    // Warm-up: the largest launch, until --warmup-ms has passed.
    std::size_t big = plan.size();
    for (std::size_t i = 0; i < plan.size(); ++i) {
        if (skipped[i]) continue;
        if (big == plan.size() || double(plan[i].part.units) * plan[i].part.tiles >
                                      double(plan[big].part.units) * plan[big].part.tiles) {
            big = i;
        }
    }
    if (big < plan.size()) {
        const Config& c    = plan[big];
        const int copies   = copies_of(c, b);
        const mb::Forced f = forced_of(c);
        quantize(c, d);
        const double until = now_s() + o.warmup_ms / 1e3;
        long long n        = 0;
        while (now_s() < until) {
            for (int i = 0; i < 20; ++i, ++n) {
                c.fmt.ops->product(planes_of(c, d, int(n % copies)), c.k / kUnitK, f, d.act, d.out,
                                   d.fixup, d.stream);
            }
            CK(cudaStreamSynchronize(d.stream));
        }
    }

    std::vector<Timed> timed(plan.size());
    const std::vector<std::size_t> alone = fixup_alone_of(plan);
    std::vector<Timed> alone_timed(alone.size());
    for (int pass = 0; pass < o.passes; ++pass) {
        std::vector<std::size_t> order(plan.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        if (pass % 2 == 1) std::reverse(order.begin(), order.end());
        for (const std::size_t i : order) {
            if (skipped[i]) continue;
            const Config& c    = plan[i];
            Timed& t           = timed[i];
            const mb::Forced f = forced_of(c);
            quantize(c, d);
            t.copies  = copies_of(c, b);
            auto call = [&](long long n) {
                c.fmt.ops->product(planes_of(c, d, int(n % t.copies)), c.k / kUnitK, f, d.act,
                                   d.out, d.fixup, d.stream);
            };
            time_config(d, b, o, t, call);
            if (resident[i] == 0) resident[i] = c.fmt.ops->resident_ctas(c.j);
            std::fprintf(stderr, "pass %d %-5s J %3d %-8s k %5d: %9.2f us a call (%d calls)\n",
                         pass + 1, c.format.c_str(), c.j, c.grid.name.c_str(), c.k,
                         median(t.pass_us.back()), t.calls);
        }
        for (std::size_t a = 0; a < alone.size(); ++a) {
            if (skipped[alone[a]]) continue;
            const Config& c    = plan[alone[a]];
            Timed& t           = alone_timed[a];
            const mb::Forced f = forced_of(c);
            quantize(c, d);
            // The product once, so the fixup plane holds this launch's partial tiles.
            c.fmt.ops->product(planes_of(c, d, 0), c.k / kUnitK, f, d.act, d.out, d.fixup,
                               d.stream);
            t.copies = 1;
            time_config(d, b, o, t,
                        [&](long long) { c.fmt.ops->fixup_only(f, d.out, d.fixup, d.stream); });
        }
    }
    std::vector<NodeCost> nodes;
    for (const bool graph : {false, true}) {
        for (const bool pdl : {false, true}) nodes.push_back(node_cost(d, o, graph, pdl));
    }

    // The document. The card is named by the driver (no host name); the bus id lets the driver
    // script match nvidia-smi's samples to this GPU.
    int driver = 0, runtime = 0;
    CK(cudaDriverGetVersion(&driver));
    CK(cudaRuntimeGetVersion(&runtime));
    char bus[32] = {};
    CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), dc.device));
    const bool roofline = dc.facts.has_roofline();
    std::string doc     = "{\n\"schema\": " + json_str(kSchema) + ",\n";
    doc += "\"card\": {\"name\": " + json_str(dc.name) + ", \"chip\": " + json_str(dc.chip) +
           ", \"cc\": " + json_str(std::to_string(dc.major) + "." + std::to_string(dc.minor)) +
           ", \"sms\": " + std::to_string(sms) + ", \"l2_bytes\": " + json_num(double(l2)) +
           ", \"global_bytes\": " + json_num(double(dc.global_bytes)) +
           ", \"pci_bus_id\": " + json_str(bus) +
           ", \"reference_clock_hz\": " + json_num(roofline ? dc.facts.reference_clock_hz() : 0.0) +
           ", \"dram_bytes_per_s\": " + json_num(roofline ? dc.facts.dram_bytes_per_s() : 0.0) +
           ", \"driver_sm_clock_khz\": " + std::to_string(dc.sm_clock_khz) +
           ", \"driver_memory_clock_khz\": " + std::to_string(dc.memory_clock_khz) +
           ", \"driver\": " + std::to_string(driver) + ", \"runtime\": " + std::to_string(runtime) +
           "},\n";
    doc += "\"protocol\": {\"samples\": " + std::to_string(o.samples) +
           ", \"passes\": " + std::to_string(o.passes) +
           ", \"sample_ms\": " + json_num(o.sample_ms) +
           ", \"warmup_ms\": " + json_num(o.warmup_ms) +
           ", \"flush_bytes\": " + json_num(double(b.flush_bytes)) +
           ", \"weight_pool_bytes\": " + json_num(double(b.pool_bytes)) +
           ", \"unit_k\": " + std::to_string(kUnitK) +
           ", \"tile_rows\": " + std::to_string(kTileRows) +
           ", \"clocks\": \"not locked; recorded by the driver script\"},\n";
    doc += "\"configs\": [\n";
    for (std::size_t i = 0; i < plan.size(); ++i) {
        doc += "  " + config_json(plan[i], timed[i], resident[i], skipped[i]) +
               (i + 1 < plan.size() ? ",\n" : "\n");
    }
    doc += "],\n\"fixup_alone\": [\n";
    for (std::size_t a = 0; a < alone.size(); ++a) {
        const Config& c = plan[alone[a]];
        doc += "  {\"format\": " + json_str(c.format) + ", \"j\": " + std::to_string(c.j) +
               ", \"k\": " + std::to_string(c.k) + ", \"grid\": " + json_str(c.grid.name) +
               ", \"fixup_bytes\": " + json_num(c.part.fixup_bytes) +
               ", \"calls\": " + std::to_string(alone_timed[a].calls) +
               ", \"t_begin\": " + json_list(alone_timed[a].t_begin) +
               ", \"t_end\": " + json_list(alone_timed[a].t_end) +
               ", \"samples_us\": " + json_passes(alone_timed[a]) + "}" +
               (a + 1 < alone.size() ? ",\n" : "\n");
    }
    doc += "],\n\"launch_node\": [\n";
    for (std::size_t n = 0; n < nodes.size(); ++n) {
        doc += "  {\"kind\": " + json_str(nodes[n].name) +
               ", \"launches\": " + std::to_string(o.node_launches) +
               ", \"samples_us\": " + json_list(nodes[n].us) +
               (nodes[n].error.empty() ? "" : ", \"error\": " + json_str(nodes[n].error)) + "}" +
               (n + 1 < nodes.size() ? ",\n" : "\n");
    }
    doc += "]\n}\n";
    if (o.out.empty()) {
        std::fputs(doc.c_str(), stdout);
        return 0;
    }
    std::FILE* f = std::fopen(o.out.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "mmq_launch_microbench: cannot write %s\n", o.out.c_str());
        return 1;
    }
    std::fputs(doc.c_str(), f);
    std::fclose(f);
    std::fprintf(stderr, "mmq_launch_microbench: wrote %s\n", o.out.c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        parse(argc, argv, o);
        if (o.dry_run) {
            // No CUDA call: the plan for the board named by --sms and --l2-mib.
            const int sms        = o.sms > 0 ? o.sms : 60;
            const std::size_t l2 = std::size_t((o.l2_mib > 0 ? o.l2_mib : 48.0) * 1048576.0);
            const std::vector<Config> plan = plan_of(o, sms);
            print_plan(stdout, o, plan, buffers_of(plan, l2, sms), sms, l2);
            return 0;
        }
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mmq_launch_microbench: %s\n", e.what());
        return 2;
    }
}
