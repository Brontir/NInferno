// Cold-L2 operator benchmark and route sweep of the generic MoE Op (ninferno/ops/moe.h).
//
// One layer of any geometry and expert format, random weights (a constant fill would make the
// kernels look faster) and random inputs with a random BF16 router, so routing spreads over the
// experts roughly uniformly. Every sample flushes L2, then times one Op call with events.
//
// Usage: ninferno_moe_bench --format F [--down-format F] --experts E --topk K
//                         --hidden H --inter F [--act swiglu|geglu|clamped|relu2]
//                         [--layout concatenated|interleaved] [--tokens 1,2,4,...]
//                         [--plans auto|sweep] [--prefill] [--repeat 20] [--passes 3]
//                         [--flush-mib 256] [--graph 0|1] [--stages 0|1] [--a8 0|1]
//                         [--frag 0|1]
//
// Formats: mxfp4 nvfp4 bf16 fp8 q4 q5 q6 q8, and ggml's blocks as published: q4_k q5_k q6_k q3_k
// q2_k iq4_xs iq3_xxs iq3_s iq2_xxs iq2_xs iq2_s and the 32-value q4_0 q4_1 q5_0 q5_1 iq4_nl.
// --down-format gives the down bank its own format (a GGUF mixture keeps gate/up and down in
// different blocks); default: --format. A ggml bank has random code bytes and FP16 scale words in
// a small range.
//
// --frag 1 stores the expert banks of a format with a ggml_frag_v1 map (q4_0, q4_k, iq3_xxs,
// iq2_s) in that layout, as the converter does by default: their random code bytes are read in
// fragment order (the timing does not depend on the values; the digest is the frag layout's).
//
// --graph 1 captures each plan's call once and times graph replays, as the engine's decode graph
// runs it; the default times eager launches, where small calls measure the host's launch rate.
// --stages 1 (implies --graph 1) also times the call's gate/up and down launches alone, each
// replayed from the captured call on the job list that call leaves, with L2 flushed before each.
//
// Every row ends with a digest of the plan's output (one eager call): two builds whose digests
// match computed bit-identical outputs.
//
// --plans auto times the route the Op selects (under the prefill profile with --prefill); sweep
// times every (gate/up schedule, down schedule) pair the compiled list allows, the router on its
// selected schedule. Output: one CSV row per (T, plan): median and minimum microseconds, and the
// weight bytes the call streams (selected experts' distinct rows) as a read-bandwidth figure.

#include "core/device.h"
#include "core/ggml_frag.h"
#include "core/weight_view.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/moe/moe_internal.h"
#include "core/fnv.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninferno;
namespace moe = ninferno::ops::detail::moe;

namespace {

__global__ void fill_random(std::uint8_t* p, std::uint64_t n, std::uint64_t seed) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    std::uint64_t v = i * 0x9e3779b97f4a7c15ULL + seed;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    p[i] = static_cast<std::uint8_t>(v ^ (v >> 31));
}

// Fills [p, p + n) with valid scale words of `format`, from random bits.
__global__ void fix_scales(std::uint8_t* p, std::uint64_t n, int format) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const std::uint8_t r = p[i];
    switch (format) {
    case 0: p[i] = static_cast<std::uint8_t>(118 + (r & 7)); break;         // MXFP4 E8M0
    case 1: p[i] = static_cast<std::uint8_t>(0x20 + (r & 15)); break;       // NVFP4 E4M3
    case 2: // FP16 group scales (two bytes, little endian): 2^-8 .. 2^-6
        p[i] = (i & 1) ? static_cast<std::uint8_t>(0x1c + (r & 3)) : r;
        break;
    case 3: // BF16 row scales / BF16 weights: small magnitudes
        p[i] = (i & 1) ? static_cast<std::uint8_t>(0x3c + (r & 1) - ((r >> 1) & 1) * 0x80) : r;
        break;
    }
}

void random_bytes(void* p, std::uint64_t n, std::uint64_t seed) {
    if (n == 0) return;
    fill_random<<<static_cast<unsigned>((n + 255) / 256), 256>>>(static_cast<std::uint8_t*>(p), n, seed);
    CUDA_CHECK(cudaGetLastError());
}

void scale_words(void* p, std::uint64_t n, int format) {
    if (n == 0) return;
    fix_scales<<<static_cast<unsigned>((n + 255) / 256), 256>>>(static_cast<std::uint8_t*>(p), n, format);
    CUDA_CHECK(cudaGetLastError());
}

QType parse_format(const std::string& s) {
    if (s == "mxfp4") return QType::MXFP4;
    if (s == "nvfp4") return QType::NVFP4;
    if (s == "bf16") return QType::BF16;
    if (s == "fp8") return QType::FP8_E4M3FN_ROW_BF16;
    if (s == "q4") return QType::Q4_G64_FP16;
    if (s == "q5") return QType::Q5_G64_FP16;
    if (s == "q6") return QType::Q6_G64_FP16;
    if (s == "q8") return QType::Q8_G32_FP16;
    if (s == "q4_k") return QType::GGML_Q4_K;
    if (s == "q5_k") return QType::GGML_Q5_K;
    if (s == "q6_k") return QType::GGML_Q6_K;
    if (s == "q3_k") return QType::GGML_Q3_K;
    if (s == "q2_k") return QType::GGML_Q2_K;
    if (s == "iq4_xs") return QType::GGML_IQ4_XS;
    if (s == "iq3_xxs") return QType::GGML_IQ3_XXS;
    if (s == "iq3_s") return QType::GGML_IQ3_S;
    if (s == "iq2_xxs") return QType::GGML_IQ2_XXS;
    if (s == "iq2_xs") return QType::GGML_IQ2_XS;
    if (s == "iq2_s") return QType::GGML_IQ2_S;
    if (s == "q4_0") return QType::GGML_Q4_0;
    if (s == "q4_1") return QType::GGML_Q4_1;
    if (s == "q5_0") return QType::GGML_Q5_0;
    if (s == "q5_1") return QType::GGML_Q5_1;
    if (s == "iq4_nl") return QType::GGML_IQ4_NL;
    throw std::invalid_argument("unknown format " + s);
}

QuantLayout layout_of(QType q) {
    switch (q) {
    case QType::MXFP4: return QuantLayout::BlockScaleK32E8M0;
    case QType::NVFP4: return QuantLayout::BlockScaleK16M128x4;
    case QType::BF16: return QuantLayout::Contiguous;
    case QType::FP8_E4M3FN_ROW_BF16: return QuantLayout::RowScale;
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K: return QuantLayout::GgmlBlocks;
    default: return is_ggml_block(q) ? QuantLayout::GgmlPlanes : QuantLayout::RowSplit;
    }
}

// A device matrix of one format with random codes and valid scales.
struct DeviceMatrix {
    std::unique_ptr<WeightParent> parent;
    DeviceBuffer buffer;
    Weight weight;
};

DeviceMatrix make_matrix(QType q, std::uint64_t n, std::uint64_t k, std::uint64_t seed) {
    DeviceMatrix m;
    const std::array<std::uint64_t, 2> shape{n, k};
    const auto g = weight_geometry(q, layout_of(q), shape);
    m.buffer     = DeviceBuffer(g.bytes);
    random_bytes(m.buffer.p, g.bytes, seed);
    auto* base = static_cast<std::uint8_t*>(m.buffer.p);
    switch (q) {
    case QType::MXFP4: scale_words(base + g.scale_offset, g.scale_bytes, 0); break;
    case QType::NVFP4: {
        scale_words(base + g.scale_offset, g.scale_bytes, 1);
        const float one = 1.0F;
        CUDA_CHECK(cudaMemcpy(base + g.divisor_offset, &one, 4, cudaMemcpyHostToDevice));
        break;
    }
    case QType::BF16: scale_words(base, g.bytes, 3); break;
    case QType::FP8_E4M3FN_ROW_BF16:
        scale_words(base + g.scale_offset, g.scale_bytes, 3);
        break;
    default:
        // GgmlBlocks keep each block's FP16 words inside its bytes: left random (the timing does
        // not depend on the values). Every other layout has a scale plane of FP16 words.
        if (layout_of(q) != QuantLayout::GgmlBlocks) {
            scale_words(base + g.scale_offset, g.scale_bytes, 2);
        }
        break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    m.parent = std::make_unique<WeightParent>(WeightParent{g, reinterpret_cast<const std::byte*>(base),
                                                           q == QType::NVFP4 ? 1.0F : 0.0F});
    const WeightView view{{n, k}, {{m.parent.get(), 0, g.elements}}};
    m.weight = native_weight(view, 1.0F);
    return m;
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) out.push_back(std::stoi(item));
    return out;
}

// One kernel node of a captured call, alone in a graph of its own: the call's arguments as
// captured (its workspace pointers included), so a replay reads what the whole call left there.
struct StageGraph {
    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec  = nullptr;
};

StageGraph stage_graph(cudaGraphNode_t node) {
    cudaKernelNodeParams params{};
    CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params));
    StageGraph s;
    CUDA_CHECK(cudaGraphCreate(&s.graph, 0));
    cudaGraphNode_t copy = nullptr;
    CUDA_CHECK(cudaGraphAddKernelNode(&copy, s.graph, nullptr, 0, &params));
    CUDA_CHECK(cudaGraphInstantiate(&s.exec, s.graph, 0));
    return s;
}

// The gate/up and down launches of a captured call: its kernel nodes are the stream's launches
// in order, and every call ends with gate/up, down, combine. The grid width checks the pick.
std::array<cudaGraphNode_t, 2> grouped_nodes(cudaGraph_t g, unsigned gate_up_x, unsigned down_x) {
    std::size_t n = 0;
    CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));
    std::vector<cudaGraphNode_t> kernels;
    for (auto node : nodes) {
        cudaGraphNodeType type{};
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeKernel) kernels.push_back(node);
    }
    if (kernels.size() < 4) throw std::runtime_error("stages: too few kernel nodes");
    const std::array<cudaGraphNode_t, 2> picked{kernels[kernels.size() - 3],
                                                kernels[kernels.size() - 2]};
    cudaKernelNodeParams gu{}, dn{};
    CUDA_CHECK(cudaGraphKernelNodeGetParams(picked[0], &gu));
    CUDA_CHECK(cudaGraphKernelNodeGetParams(picked[1], &dn));
    if (gu.gridDim.x != gate_up_x || dn.gridDim.x != down_x) {
        throw std::runtime_error("stages: the captured nodes are not gate/up and down");
    }
    return picked;
}

// FNV-1a over the bytes: equal digests of two builds' outputs mark them bit-identical.
std::uint64_t digest(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t h = ninferno::fnv::kOffset;
    for (const std::uint8_t b : bytes) h = ninferno::fnv::mix(h, b);
    return h;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> a{{"format", "mxfp4"}, {"experts", "32"}, {"topk", "4"},
                                             {"hidden", "2880"}, {"inter", "2880"},
                                             {"act", "swiglu"}, {"layout", "concatenated"},
                                             {"tokens", "1,2,4,8,16,32,64,128,512,2048"},
                                             {"plans", "auto"}, {"repeat", "20"}, {"passes", "3"},
                                             {"flush-mib", "256"}};
        bool prefill = false;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k == "--prefill") { prefill = true; continue; }
            if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::invalid_argument("bad argument " + k);
            a[k.substr(2)] = argv[++i];
        }
        const QType q   = parse_format(a["format"]);
        const std::string down_format = a.count("down-format") ? a["down-format"] : a["format"];
        const QType qd  = parse_format(down_format);
        const bool stages = a.count("stages") && a["stages"] == "1";
        const bool graph  = stages || (a.count("graph") && a["graph"] == "1");
        const int E     = std::stoi(a["experts"]);
        const int K     = std::stoi(a["topk"]);
        const int H     = std::stoi(a["hidden"]);
        const int F     = std::stoi(a["inter"]);
        const int reps  = std::stoi(a["repeat"]);
        const int passes = std::stoi(a["passes"]);
        const auto tokens = parse_list(a["tokens"]);
        const int t_max = *std::max_element(tokens.begin(), tokens.end());

        DeviceContext device(0);
        auto router  = make_matrix(QType::BF16, E, H, 1);
        // relu2 (NemotronH): a non-gated expert, gate_up holds its F up rows only.
        const int gate_up_rows = a["act"] == "relu2" ? 1 : 2;
        auto gate_up = make_matrix(q, static_cast<std::uint64_t>(E) * gate_up_rows * F, H, 2);
        auto down    = make_matrix(qd, static_cast<std::uint64_t>(E) * H, F, 3);
        ops::MoeWeights w;
        w.router  = router.weight;
        w.gate_up = gate_up.weight;
        w.down    = down.weight;
        if (a.count("a8") && a["a8"] == "1") w.expert_policy = ops::LinearPolicy::AllowA8;
        if (a.count("frag") && a["frag"] == "1") {
            for (Weight* bank : {&w.gate_up, &w.down}) {
                if (ggml_frag_format(bank->qtype)) bank->layout = QuantLayout::GgmlFrag;
            }
        }
        ops::MoeConfig c;
        c.top_k = K;
        c.activation = a["act"] == "geglu"     ? ops::MoeActivation::GeGluTanh
                       : a["act"] == "clamped" ? ops::MoeActivation::ClampedSwiGlu
                       : a["act"] == "relu2"   ? ops::MoeActivation::Relu2
                                               : ops::MoeActivation::SwiGlu;
        c.gate_up_layout = a["layout"] == "interleaved" ? ops::MoeGateUpLayout::Interleaved
                                                        : ops::MoeGateUpLayout::Concatenated;

        DeviceBuffer x(static_cast<std::size_t>(t_max) * H * 2);
        random_bytes(x.p, x.bytes, 7);
        scale_words(x.p, x.bytes, 3);
        DeviceBuffer out(x.bytes);
        DeviceBuffer flush(static_cast<std::size_t>(std::stoi(a["flush-mib"])) << 20);
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        const ops::ScopedPrefillRoutes profile(prefill);

        const double expert_bytes =
            static_cast<double>(gate_up.weight.payload_bytes + down.weight.payload_bytes) / E;
        std::printf("format,down_format,graph,E,K,H,F,T,prefill,router,plan,job,median_us,"
                    "min_us,expert_weight_gbs,gate_up_us,down_us,out_digest\n");
        for (const int t : tokens) {
            std::vector<moe::MoePlan> plans;
            const auto selected = moe::select_plan(w, c, t, prefill);
            if (a["plans"] == "sweep") {
                // A stage's candidates: every register-MMA schedule, and every R8 schedule where
                // the layer admits R8 at this T (--a8 1).
                struct Candidate {
                    int schedule, r8;
                };
                const auto candidates = [&](const Weight& bank) {
                    std::vector<Candidate> out;
                    for (int s = 0; s < moe::schedule_count(); ++s) out.push_back({s, -1});
                    if (moe::r8_admits(w, bank, t)) {
                        for (int s = 0; s < moe::r8_schedule_count(); ++s) out.push_back({0, s});
                    }
                    return out;
                };
                for (const auto gu : candidates(w.gate_up)) {
                    for (const auto dn : candidates(w.down)) {
                        moe::MoePlan p = selected;
                        p.gate_up_schedule = gu.schedule;
                        p.down_schedule    = dn.schedule;
                        p.gate_up_r8       = gu.r8;
                        p.down_r8          = dn.r8;
                        p.job_columns      = 8;
                        for (const auto s : {gu, dn}) {
                            if (s.r8 < 0) {
                                p.job_columns = std::max(
                                    p.job_columns, moe::schedule_info(s.schedule).tokens_per_cta());
                            }
                        }
                        if (prefill && t >= 17 && !moe::plan_is_token_invariant(p)) continue;
                        if (!moe::plan_compiled(w, p)) continue;
                        plans.push_back(p);
                    }
                }
            } else {
                plans.push_back(selected);
            }
            std::size_t ws_bytes = 0;
            for (const auto& p : plans) ws_bytes = std::max(ws_bytes, moe::plan_workspace_bytes(w, c, t, p));
            DeviceArena workspace(std::max<std::size_t>(ws_bytes, 256));
            Tensor tx(x.p, DType::BF16, {H, t});
            Tensor tout(out.p, DType::BF16, {H, t});
            // Distinct experts a call touches, for the bandwidth column (uniform routing estimate).
            const double touched = E * (1.0 - std::pow(1.0 - static_cast<double>(K) / E, t));
            // Per plan: the whole call, then (--stages 1) its gate/up and its down launch alone.
            constexpr int kSeries = 3;
            std::vector<std::array<std::vector<float>, kSeries>> samples(plans.size());
            // One captured graph per plan, its workspace carved from the same arena position.
            std::vector<cudaGraph_t> captured(plans.size(), nullptr);
            std::vector<cudaGraphExec_t> execs(plans.size(), nullptr);
            std::vector<std::array<StageGraph, 2>> stage_graphs(plans.size());
            if (graph) {
                for (std::size_t i = 0; i < plans.size(); ++i) {
                    workspace.reset();
                    CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
                    moe::moe_with_plan(tx, w, c, ops::MoeEpilogue::Store, tout, workspace,
                                       device.stream, plans[i]);
                    CUDA_CHECK(cudaStreamEndCapture(device.stream, &captured[i]));
                    CUDA_CHECK(cudaGraphInstantiate(&execs[i], captured[i], 0));
                    if (stages) {
                        const auto rows = [](int schedule, int r8, const Weight& bank) {
                            return r8 >= 0 ? moe::r8_schedule_info(r8).rows_per_cta() * 32 /
                                                 moe::r8_lanes(bank)
                                           : moe::schedule_info(schedule).rows_per_cta();
                        };
                        const int gu =
                            rows(plans[i].gate_up_schedule, plans[i].gate_up_r8, w.gate_up);
                        const int dn = rows(plans[i].down_schedule, plans[i].down_r8, w.down);
                        const auto nodes =
                            grouped_nodes(captured[i], static_cast<unsigned>((F + gu - 1) / gu),
                                          static_cast<unsigned>((H + dn - 1) / dn));
                        stage_graphs[i] = {stage_graph(nodes[0]), stage_graph(nodes[1])};
                    }
                }
            }
            const auto time_once = [&](std::size_t i, int series) {
                CUDA_CHECK(cudaEventRecord(e0, device.stream));
                if (series > 0) {
                    CUDA_CHECK(cudaGraphLaunch(stage_graphs[i][series - 1].exec, device.stream));
                } else if (graph) {
                    CUDA_CHECK(cudaGraphLaunch(execs[i], device.stream));
                } else {
                    moe::moe_with_plan(tx, w, c, ops::MoeEpilogue::Store, tout, workspace,
                                       device.stream, plans[i]);
                }
                CUDA_CHECK(cudaEventRecord(e1, device.stream));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                return ms * 1000.0F;
            };
            const int series = stages ? kSeries : 1;
            for (int pass = 0; pass < passes; ++pass) {
                for (std::size_t i = 0; i < plans.size(); ++i) {
                    for (int sr = 0; sr < series; ++sr) {
                        for (int r = 0; r < reps + (pass == 0 ? 2 : 0); ++r) {
                            // A stage replays on the job list and inputs its whole call leaves.
                            if (sr > 0 && r == 0) {
                                CUDA_CHECK(cudaGraphLaunch(execs[i], device.stream));
                            }
                            CUDA_CHECK(cudaMemsetAsync(flush.p, r & 0xff, flush.bytes,
                                                       device.stream));
                            workspace.reset();
                            const float us = time_once(i, sr);
                            if (pass == 0 && r < 2) continue; // warm-up
                            samples[i][sr].push_back(us);
                        }
                    }
                }
            }
            for (std::size_t i = 0; i < plans.size(); ++i) {
                for (auto& s : stage_graphs[i]) {
                    if (s.exec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(s.exec));
                    if (s.graph != nullptr) CUDA_CHECK(cudaGraphDestroy(s.graph));
                }
                if (execs[i] != nullptr) CUDA_CHECK(cudaGraphExecDestroy(execs[i]));
                if (captured[i] != nullptr) CUDA_CHECK(cudaGraphDestroy(captured[i]));
            }
            const auto median = [](std::vector<float>& s) {
                if (s.empty()) return 0.0;
                std::sort(s.begin(), s.end());
                return static_cast<double>(s[s.size() / 2]);
            };
            std::vector<std::uint8_t> host(static_cast<std::size_t>(t) * H * 2);
            for (std::size_t i = 0; i < plans.size(); ++i) {
                auto& s = samples[i][0];
                const double med = median(s);
                // The plan's output of one eager call.
                workspace.reset();
                moe::moe_with_plan(tx, w, c, ops::MoeEpilogue::Store, tout, workspace,
                                   device.stream, plans[i]);
                CUDA_CHECK(cudaMemcpyAsync(host.data(), tout.data, host.size(),
                                           cudaMemcpyDeviceToHost, device.stream));
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%.2f,%.2f,%.1f,%.2f,%.2f,"
                            "%016llx\n",
                            a["format"].c_str(), down_format.c_str(), graph ? 1 : 0, E, K, H, F, t,
                            prefill ? 1 : 0, plans[i].router_schedule,
                            moe::plan_name(plans[i]).c_str(), plans[i].job_columns, med, s.front(),
                            touched * expert_bytes / (med * 1e3), median(samples[i][1]),
                            median(samples[i][2]),
                            static_cast<unsigned long long>(digest(host)));
            }
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninferno_moe_bench: %s\n", e.what());
        return 1;
    }
}
