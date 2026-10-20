// ninferno_r8_schedule_bench: every compiled R8 schedule of the generic projection path at decode
// (T = 1) on the given problems, next to the schedule the dispatcher picks, with the check that
// all schedules give the same bytes.
//
//   ninferno_r8_schedule_bench [--shapes FILE] [--problem FORMAT,EPILOGUE,N,K ...]
//                            [--samples 15] [--launches 60]
//
// FILE is the NINFERNO_ROUTE_RECORD / ninferno_route_tune format, "<format> <compute> <epilogue>
// <n> <k>" per line (the compute column is ignored: R8 is the A8 decode route); --problem adds one
// ("iq3xxs,store,43008,5376"). N is the op's output rows (SwiGLU: the gate rows). A problem whose
// format has no R8 codec, or whose activation does not fit the CTA, is reported and skipped.
//
// Protocol: copies of one random weight covering four times the L2, a CUDA graph of --launches
// calls cycling through them per schedule, the schedules interleaved within each sample (the first
// one rotating), the median of --samples graph replays divided by --launches.
//
// Output, per problem: one line per schedule,
//   r8 <format> <epilogue> n=<n> k=<k> <route> <us> us <GB/s> GB/s same=<0|1> [picked] [rules]
// where same=1 means the output bytes equal the first schedule's, "picked" marks the dispatcher's
// choice (select_route at T = 1, AllowA8: with this GPU's r8_decode_table.h entries) and "rules"
// the shape-class rules' choice without them (r8_rules_route, what tools/bench/r8_decode_table.py
// measures entries against); then
//   pick <format> <epilogue> n=<n> k=<k> picked=<route> <us> rules=<route> <us> best=<route> <us>
#include "core/arena.h"
#include "core/device.h"
#include "core/device_class.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"
#include "random_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;

struct Problem {
    QType qtype;
    generic::Epilogue epilogue;
    std::int32_t n;
    std::int32_t k;
};

QType parse_format(const std::string& s) {
    for (int v = 0; v <= 0xff; ++v) {
        const auto q = static_cast<QType>(v);
        if (s == generic::qtype_name(q)) return q;
    }
    throw std::invalid_argument("unknown format " + s);
}

generic::Epilogue parse_epilogue(const std::string& s) {
    if (s == "store") return generic::Epilogue::Store;
    if (s == "add") return generic::Epilogue::AddResidual;
    if (s == "swiglu") return generic::Epilogue::SwiGlu;
    throw std::invalid_argument("unknown epilogue " + s);
}

double median(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

void run_problem(const Problem& p, int samples, int launches, std::size_t l2_bytes) {
    const char* fmt = generic::qtype_name(p.qtype);
    const char* epi = generic::epilogue_name(p.epilogue);
    const bool swiglu       = p.epilogue == generic::Epilogue::SwiGlu;
    const std::int32_t rows = swiglu ? 2 * p.n : p.n;
    if (!generic::r8_supports(p.qtype) || (p.k % 128) != 0 || !generic::r8_fits(p.qtype, p.k, 1)) {
        std::printf("skip %s %s n=%d k=%d: no R8 decode for this format or K\n", fmt, epi, p.n,
                    p.k);
        return;
    }

    // Weight copies (one random weight at several addresses): four times the L2 in all, at least
    // two, at most 3 GB.
    std::vector<std::unique_ptr<bench::RandomWeight>> copies;
    copies.push_back(std::make_unique<bench::RandomWeight>(
        bench::random_weight(p.qtype, rows, p.k, 7U + p.n + p.k)));
    const std::size_t bytes = copies[0]->parent.geometry.bytes;
    std::size_t count       = std::max<std::size_t>(2, (4 * l2_bytes + bytes - 1) / bytes);
    count = std::min<std::size_t>(count, std::max<std::size_t>(2, (3ULL << 30) / bytes));
    for (std::size_t c = 1; c < count; ++c) {
        copies.push_back(std::make_unique<bench::RandomWeight>(bench::clone_weight(*copies[0])));
    }

    DeviceBuffer x(static_cast<std::size_t>(p.k) * 2);
    {
        std::vector<std::uint16_t> host(static_cast<std::size_t>(p.k));
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        for (auto& v : host) {
            const float f = nd(rng);
            std::uint32_t b;
            std::memcpy(&b, &f, 4);
            v = static_cast<std::uint16_t>(b >> 16);
        }
        x.copy_from_host(host.data(), host.size() * 2);
    }
    DeviceBuffer y(static_cast<std::size_t>(p.n) * 2);
    y.fill(0);
    DeviceArena workspace(256);
    Tensor tx(x.p, DType::BF16, {p.k, 1});
    Tensor ty(y.p, DType::BF16, {p.n, 1});
    const auto make_call = [&](const Weight& w) {
        generic::Call call;
        call.x               = &tx;
        call.weight          = &w;
        call.rows            = p.n;
        call.up_offset       = swiglu ? p.n : 0;
        call.epilogue        = p.epilogue;
        call.out[0]          = &ty;
        call.segment_rows[0] = p.n;
        return call;
    };

    const int schedules = generic::r8_schedule_count();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Bytes: every schedule against the first, on copy 0.
    std::vector<std::vector<std::uint8_t>> outputs(schedules);
    for (int i = 0; i < schedules; ++i) {
        const generic::Route r{generic::Family::R8Mma, static_cast<std::uint8_t>(i)};
        y.fill(0);
        generic::run_route(r, make_call(copies[0]->weight), &workspace, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        outputs[i].resize(y.bytes);
        y.copy_to_host(outputs[i].data(), y.bytes);
    }

    std::vector<cudaGraphExec_t> graphs(schedules);
    for (int i = 0; i < schedules; ++i) {
        const generic::Route r{generic::Family::R8Mma, static_cast<std::uint8_t>(i)};
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < launches; ++l) {
            const auto& copy = *copies[static_cast<std::size_t>(l) % copies.size()];
            generic::run_route(r, make_call(copy.weight), &workspace, stream);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphs[i], graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        for (int w = 0; w < 2; ++w) CUDA_CHECK(cudaGraphLaunch(graphs[i], stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<std::vector<double>> us(schedules);
    for (int s = 0; s < samples; ++s) {
        for (int j = 0; j < schedules; ++j) {
            const int i = (s + j) % schedules;
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaGraphLaunch(graphs[i], stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            us[i].push_back(static_cast<double>(ms) * 1000.0 / launches);
        }
    }

    const generic::Problem problem{p.qtype, ops::LinearPolicy::AllowA8, p.n, p.k, p.epilogue};
    const generic::Route picked = generic::select_route(problem, copies[0]->weight, 1, false);
    const generic::Route rules  = generic::r8_rules_route(problem, 1);
    int best = 0;
    std::vector<double> med(schedules);
    for (int i = 0; i < schedules; ++i) {
        med[i] = median(us[i]);
        if (med[i] < med[best]) best = i;
    }
    for (int i = 0; i < schedules; ++i) {
        const generic::Route r{generic::Family::R8Mma, static_cast<std::uint8_t>(i)};
        std::printf("r8 %s %s n=%d k=%d %s %.2f us %.1f GB/s same=%d%s%s\n", fmt, epi, p.n, p.k,
                    generic::route_name(r).c_str(), med[i],
                    static_cast<double>(bytes) / (med[i] * 1e3),
                    outputs[i] == outputs[0] ? 1 : 0, r == picked ? " picked" : "",
                    r == rules ? " rules" : "");
    }
    const bool picked_r8 = picked.family == generic::Family::R8Mma;
    const generic::Route fastest{generic::Family::R8Mma, static_cast<std::uint8_t>(best)};
    std::printf("pick %s %s n=%d k=%d picked=%s %.2f us rules=%s %.2f us best=%s %.2f us\n", fmt,
                epi, p.n, p.k, generic::route_name(picked).c_str(),
                picked_r8 ? med[picked.schedule] : 0.0, generic::route_name(rules).c_str(),
                med[rules.schedule], generic::route_name(fastest).c_str(), med[best]);
    std::fflush(stdout);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (auto g : graphs) CUDA_CHECK(cudaGraphExecDestroy(g));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<Problem> problems;
        int samples  = 15;
        int launches = 60;
        std::set<std::tuple<int, int, int, int>> seen;
        const auto add = [&](const std::string& fmt, const std::string& epi, std::int32_t n,
                             std::int32_t k) {
            const Problem p{parse_format(fmt), parse_epilogue(epi), n, k};
            if (seen.insert({static_cast<int>(p.qtype), static_cast<int>(p.epilogue), n, k})
                    .second) {
                problems.push_back(p);
            }
        };
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            const std::string v = argv[++i];
            if (a == "--samples") {
                samples = std::stoi(v);
            } else if (a == "--launches") {
                launches = std::stoi(v);
            } else if (a == "--problem") {
                std::stringstream in(v);
                std::string fmt, epi, n, k;
                if (!std::getline(in, fmt, ',') || !std::getline(in, epi, ',') ||
                    !std::getline(in, n, ',') || !std::getline(in, k, ',')) {
                    throw std::invalid_argument("--problem FORMAT,EPILOGUE,N,K");
                }
                add(fmt, epi, std::stoi(n), std::stoi(k));
            } else if (a == "--shapes") {
                std::ifstream in(v);
                if (!in) throw std::runtime_error("cannot open " + v);
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty() || line[0] == '#') continue;
                    std::istringstream fields(line);
                    std::string fmt, comp, epi;
                    std::int32_t n = 0, k = 0;
                    if (fields >> fmt >> comp >> epi >> n >> k) add(fmt, epi, n, k);
                }
            } else {
                throw std::invalid_argument("unknown option " + a);
            }
        }
        if (problems.empty() || samples <= 0 || launches <= 0) {
            std::cerr << "usage: ninferno_r8_schedule_bench [--shapes FILE] "
                         "[--problem FORMAT,EPILOGUE,N,K ...] [--samples N] [--launches N]\n";
            return 2;
        }
        const auto& device = device_class();
        std::printf("device %s class %s, L2 %zu MB, %d samples x %d launches\n",
                    device.name.c_str(), device.id().c_str(), device.l2_bytes >> 20, samples,
                    launches);
        {
            const Board b = board_of(device);
            std::printf("board cc %d sms %d l2_mib %g dram_gbps %g\n", b.major * 10 + b.minor,
                        b.sms, b.l2_mib, b.dram_gbps);
        }
        for (const Problem& p : problems) run_problem(p, samples, launches, device.l2_bytes);
        std::printf("done\n");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ninferno_r8_schedule_bench: " << e.what() << '\n';
        return 1;
    }
}
