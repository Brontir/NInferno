// ninferno_r8_schedule_bench: every compiled R8 schedule of the generic projection path at decode
// (T = 1) on the given problems, next to the schedule the dispatcher picks, with the check that
// all schedules give the same bytes. With --tokens T (speculative verify widths), the R8
// schedules that take T, ggml's vector kernel where it reads the format, and the dispatcher's
// route at T, each with whether its first token's output equals the decode route's (T = 1) bytes:
// where it does, greedy speculation reproduces plain decode.
//
//   ninferno_r8_schedule_bench [--shapes FILE] [--problem FORMAT,EPILOGUE,N,K ...]
//                            [--samples 15] [--launches 60] [--tokens 1,2,4,8] [--rmma 1]
//                            [--span-mib M]
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

// --rmma 1: also time every launchable register-MMA schedule of one token tile (T <= 8), so a
// K-quant verify width compares the dispatcher's pick with the other tiles.
bool g_rmma = false;

// --span-mib M: the weight copies cover at least M MiB (default: four times the L2), so that a run
// of launches also walks as many distinct pages as a model's step does.
std::size_t g_span_bytes = 0;

double median(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

void run_problem(const Problem& p, int samples, int launches, std::size_t l2_bytes,
                 std::int32_t tokens) {
    const char* fmt         = generic::qtype_name(p.qtype);
    const char* epi         = generic::epilogue_name(p.epilogue);
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
    const std::size_t span  = std::max(4 * l2_bytes, g_span_bytes);
    std::size_t count       = std::max<std::size_t>(2, (span + bytes - 1) / bytes);
    count = std::min<std::size_t>(count, std::max<std::size_t>(2, (3ULL << 30) / bytes));
    for (std::size_t c = 1; c < count; ++c) {
        copies.push_back(std::make_unique<bench::RandomWeight>(bench::clone_weight(*copies[0])));
    }

    DeviceBuffer x(static_cast<std::size_t>(p.k) * tokens * 2);
    {
        std::vector<std::uint16_t> host(static_cast<std::size_t>(p.k) * tokens);
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
    DeviceBuffer y(static_cast<std::size_t>(p.n) * tokens * 2);
    y.fill(0);
    // The verify widths quantize the activation into the workspace (R8, the vector kernel).
    DeviceArena workspace(64ULL << 20);
    Tensor tx(x.p, DType::BF16, {p.k, tokens});
    Tensor ty(y.p, DType::BF16, {p.n, tokens});
    Tensor tx1(x.p, DType::BF16, {p.k, 1});
    Tensor ty1(y.p, DType::BF16, {p.n, 1});
    const auto make_call = [&](const Weight& w, bool decode = false) {
        generic::Call call;
        call.x               = decode ? &tx1 : &tx;
        call.weight          = &w;
        call.rows            = p.n;
        call.up_offset       = swiglu ? p.n : 0;
        call.epilogue        = p.epilogue;
        call.out[0]          = decode ? &ty1 : &ty;
        call.segment_rows[0] = p.n;
        return call;
    };

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const generic::Problem problem{p.qtype, ops::LinearPolicy::AllowA8, p.n, p.k, p.epilogue};
    const generic::Route picked = generic::select_route(problem, copies[0]->weight, tokens, false);
    // The dispatcher's route for a BF16-activation call (A16Only: a DFlash drafter's projections).
    const generic::Problem problem16{p.qtype, ops::LinearPolicy::A16Only, p.n, p.k, p.epilogue};
    const generic::Route picked16 =
        generic::select_route(problem16, copies[0]->weight, tokens, false);

    // The candidates at T: the R8 schedules, ggml's vector kernel and the dispatcher's route; one
    // the build or the call refuses is left out.
    std::vector<generic::Route> routes;
    const auto offer = [&](generic::Route r) {
        if (std::find(routes.begin(), routes.end(), r) != routes.end()) return;
        if (!generic::route_admits(r, copies[0]->weight, ops::LinearPolicy::AllowA8)) return;
        try {
            generic::run_route(r, make_call(copies[0]->weight), &workspace, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            routes.push_back(r);
        } catch (const std::invalid_argument&) {}
    };
    for (int i = 0; i < generic::r8_schedule_count(); ++i) {
        offer(generic::Route{generic::Family::R8Mma, static_cast<std::uint8_t>(i)});
    }
    if (tokens > 1) offer(generic::Route{generic::Family::GgmlVec, 0});
    offer(picked);
    offer(picked16);
    if (g_rmma && tokens <= 8) {
        for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
            if (generic::rmma_schedule_launchable(i) &&
                generic::rmma_schedule_info(i).n_tiles == 1) {
                offer(generic::Route{generic::Family::RegisterMma, static_cast<std::uint8_t>(i)});
            }
        }
    }
    const int schedules = static_cast<int>(routes.size());

    // The decode route's bytes of the first token (T = 1), against every candidate's first token.
    const std::size_t column = static_cast<std::size_t>(p.n) * 2;
    std::vector<std::uint8_t> decoded(column);
    {
        const generic::Route decode = generic::select_route(problem, copies[0]->weight, 1, false);
        y.fill(0);
        generic::run_route(decode, make_call(copies[0]->weight, true), &workspace, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        y.copy_to_host(decoded.data(), column);
    }
    std::vector<std::vector<std::uint8_t>> outputs(schedules);
    for (int i = 0; i < schedules; ++i) {
        y.fill(0);
        generic::run_route(routes[i], make_call(copies[0]->weight), &workspace, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        outputs[i].resize(y.bytes);
        y.copy_to_host(outputs[i].data(), y.bytes);
    }

    std::vector<cudaGraphExec_t> graphs(schedules);
    for (int i = 0; i < schedules; ++i) {
        const generic::Route r = routes[i];
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

    const generic::Route rules = generic::r8_rules_route(problem, 1);
    int best = 0, at_picked = 0;
    std::vector<double> med(schedules);
    for (int i = 0; i < schedules; ++i) {
        med[i] = median(us[i]);
        if (med[i] < med[best]) best = i;
        if (routes[i] == picked) at_picked = i;
    }
    for (int i = 0; i < schedules; ++i) {
        const generic::Route r = routes[i];
        const bool same_decode = std::equal(decoded.begin(), decoded.end(), outputs[i].begin());
        // FNV-1a of the whole output: one route's bytes compared across builds.
        std::uint64_t hash = 14695981039346656037ULL;
        for (const std::uint8_t c : outputs[i]) hash = (hash ^ c) * 1099511628211ULL;
        std::printf(
            "r8 %s %s n=%d k=%d t=%d %s %.2f us %.1f GB/s same=%d decode=%d%s%s%s h=%016llx\n", fmt,
            epi, p.n, p.k, tokens, generic::route_name(r).c_str(), med[i],
            static_cast<double>(bytes) / (med[i] * 1e3), outputs[i] == outputs[0] ? 1 : 0,
            same_decode ? 1 : 0, r == picked ? " picked" : "", r == rules ? " rules" : "",
            r == picked16 ? " a16pick" : "", static_cast<unsigned long long>(hash));
    }
    std::printf("pick %s %s n=%d k=%d t=%d picked=%s %.2f us best=%s %.2f us\n", fmt, epi, p.n, p.k,
                tokens, generic::route_name(picked).c_str(), med[at_picked],
                generic::route_name(routes[best]).c_str(), med[best]);
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
        std::vector<std::int32_t> widths{1};
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
            } else if (a == "--span-mib") {
                g_span_bytes = static_cast<std::size_t>(std::stoll(v)) << 20;
            } else if (a == "--rmma") {
                g_rmma = std::stoi(v) != 0;
            } else if (a == "--tokens") {
                widths.clear();
                std::stringstream in(v);
                std::string item;
                while (std::getline(in, item, ',')) widths.push_back(std::stoi(item));
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
        if (problems.empty() || samples <= 0 || launches <= 0 || widths.empty() ||
            std::any_of(widths.begin(), widths.end(), [](std::int32_t t) { return t <= 0; })) {
            std::cerr << "usage: ninferno_r8_schedule_bench [--shapes FILE] "
                         "[--problem FORMAT,EPILOGUE,N,K ...] [--samples N] [--launches N] "
                         "[--tokens T,...]\n";
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
        for (const Problem& p : problems) {
            for (const std::int32_t t : widths)
                run_problem(p, samples, launches, device.l2_bytes, t);
        }
        std::printf("done\n");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ninferno_r8_schedule_bench: " << e.what() << '\n';
        return 1;
    }
}
