// The route analyzer (route_analyzer.h). Every formula is tools/route_pick/analyzer.py's, term
// for term, so ninferno_route_analyzer_parity_test can hold the two equal; the comments name the
// source each input mirrors.
#include "ops/generic/route_analyzer.h"

#include "core/arch_registry.h"
#include "core/device_class.h"
#include "core/isa_facts.h"
#include "core/machine_facts.h"
#include "core/mmq_columns.h"
#include "core/sm_rate_facts.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_rmma_decoders.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_table.h"
#include "ops/route_pins.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace ninferno::ops::route_analyzer {
namespace {

namespace kf = detail::generic::kernel_facts;
using detail::generic::Epilogue;
using detail::generic::WeightCodec;
using kf::KernelFacts;
using kf::LoopCounts;
using kf::PipeCounts;
using kf::PrologueCounts;
using kf::StaticSchedule;

constexpr double kL1MinKib   = 28.0; // SET: the L1 an SM keeps beside its shared memory
constexpr double kPartitions = 4.0;  // tensor partitions of an SM
// Counted: the SASS instructions of one emulated MMA call (sm_70, nvcc 12.8.93,
// generic_gemm_bf16.cu.o mma_bf16_fma_call; #1165).
constexpr double kEmuCall = 551.0;
// Measured (few/bw/emu.cu, Quadro GV100, 24.11): a warp's ns a call with 1 / 2 / 4 independent
// accumulator chains, and the issue cost a call (4 partitions x 1627 MHz / 18.86 calls a us).
constexpr double kEmuLatNs[5] = {0.0, 563.2, 437.0, 437.0, 377.6};
constexpr double kEmuIssue    = 4.0 * 1627.0 / 18.86;
constexpr double kMmqWarps    = 8.0;
constexpr int kMmqTile        = 128;
constexpr int kR8StaticSmem   = 48 * 1024;           // arch_limits.h kStaticSharedBytesPerBlock
constexpr int kVecWarps       = 4;                   // ggml_vec.cu kVecWarps
constexpr int kVecTwinBlocks  = 7;                   // ggml_vec.cu kVecBoundedBlocks
constexpr int kVecKs[]        = {5120, 6144, 17408}; // ggml_vec.cu vec_launch_k
constexpr int kVoltaCtaRows   = 32;                  // generic_volta_hmma.cuh kCtaRows

// The narrow-load facts per architecture (probe narrow.cu, #1183): a warp load's issue cap at
// (4 B a lane, 4 B stride), a wavefront's clocks, the bank width whose wavefront count matches.
// Equal on every card of an architecture.
struct Narrow {
    std::string_view arch;
    double cap, wf;
    int bank;
};

constexpr Narrow kNarrow[] = {{"70", 1.621, 1.027, 8},
                              {"75", 2.012, 2.000, 8},
                              {"86", 2.000, 1.019, 4},
                              {"89", 2.000, 1.019, 4},
                              {"120a", 2.000, 1.008, 4}};

// The bits a weight value of a format takes (analyzer.py FORMATS), its rmma latency-variant
// threshold class and the formats the law prices.
struct FormatRow {
    QType q;
    std::string_view name;
    double bits;
};

constexpr FormatRow kFormats[] = {
    {QType::GGML_Q4_K, "q4k", 4.5},
    {QType::GGML_Q5_K, "q5k", 5.5},
    {QType::GGML_Q6_K, "q6k", 6.5625},
    {QType::Q8_G32_FP16, "q8", 8.5},
    {QType::GGML_Q3_K, "q3k", 3.4375},
    {QType::GGML_Q2_K, "q2k", 2.625},
    {QType::GGML_IQ4_XS, "iq4xs", 4.25},
    {QType::GGML_IQ4_NL, "iq4nl", 4.5},
    {QType::GGML_IQ3_S, "iq3s", 3.4375},
    {QType::GGML_IQ3_XXS, "iq3xxs", 3.0625},
    {QType::GGML_IQ2_S, "iq2s", 2.5625},
    {QType::GGML_IQ2_XS, "iq2xs", 2.3125},
    {QType::GGML_IQ2_XXS, "iq2xxs", 2.0625},
    {QType::GGML_IQ1_S, "iq1s", 1.5625},
    {QType::GGML_IQ1_M, "iq1m", 1.75},
    {QType::GGML_Q4_0, "q40", 4.5},
    {QType::GGML_Q4_1, "q41", 5.0},
    {QType::GGML_Q5_0, "q50", 5.5},
    {QType::GGML_Q5_1, "q51", 6.0},
    {QType::BF16, "bf16", 16.0},
    {QType::FP8_E4M3FN_ROW_BF16, "fp8", 8.0},
    {QType::FP8_E4M3FN_ROW_F32, "fp8r32", 8.0},
    {QType::NVFP4, "nvfp4", 4.5},
    {QType::Q4_G64_FP16, "q4", 4.25},
    {QType::Q5_G64_FP16, "q5", 5.25},
    {QType::Q6_G64_FP16, "q6", 6.25},
};

const FormatRow* format_row(QType q) {
    for (const FormatRow& f : kFormats)
        if (f.q == q) return &f;
    return nullptr;
}

// supports(): the K a weight of the format must span to be loaded at all.
int format_k(QType q) {
    switch (q) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
        return 32;
    case QType::NVFP4:
        return 64;
    default:
        return format_facts(q).block_values == 256 ? 256 : 1;
    }
}

std::optional<WeightCodec> codec_of(QType q) { return format_facts(q).codec; }

std::string rmma_decoder(QType q, int k) {
    const auto c = codec_of(q);
    if (!c) return {};
    const auto d = detail::generic::rmma_decoder_of(*c, k);
    return d.name ? std::string(d.name) : std::string();
}

std::string r8_codec(QType q) {
    const auto c = codec_of(q);
    if (!c) return {};
    return std::string(detail::generic::w4a8_decoder_name(*c));
}

int ggml_type(QType q) { return format_facts(q).ggml_type; }

int epi_of(Epilogue e) { return static_cast<int>(e); }

double lat_bytes(Epilogue e) { return e == Epilogue::SwiGlu ? 4.5 : 5.5; }

// ---- the kernel-facts reads (kf_row: a loop row needs loops > 0) ----
bool has_loop(const KernelFacts& f) { return !f.loop || f.loop->loops > 0; }

double loop_i(const KernelFacts& f, int LoopCounts::* m) { return f.loop ? (*f.loop).*m : 0.0; }

double loop_z(const KernelFacts& f, std::size_t LoopCounts::* m) {
    return f.loop ? static_cast<double>((*f.loop).*m) : 0.0;
}

double sched(const KernelFacts& f, int StaticSchedule::* m) {
    return f.loop && f.loop->schedule ? (*f.loop->schedule).*m : 0.0;
}

double pipe(const KernelFacts& f, int PipeCounts::* m) {
    return f.loop && f.loop->pipes ? (*f.loop->pipes).*m : 0.0;
}

double pro(const KernelFacts& f, int PrologueCounts::* m) {
    return f.prologue ? (*f.prologue).*m : 0.0;
}

std::string mma_shape(const KernelFacts& f) { return f.loop ? f.loop->mma_shape : std::string(); }

std::optional<KernelFacts> rmma_row(const Card& c, const std::string& decoder, int A, int B, int M,
                                    int N, int C, Epilogue ep, bool latency) {
    kf::RmmaKernelKey key;
    key.decoder  = decoder;
    key.warps_m  = A;
    key.warps_k  = B;
    key.m_tiles  = M;
    key.n_tiles  = N;
    key.cluster  = C;
    key.epilogue = epi_of(ep);
    key.latency  = latency;
    key.sums     = 0;
    auto r       = kf::rmma_facts(c.arch, key);
    if (r && (!has_loop(*r) || r->registers <= 0 || !r->loop || r->loop->k_iter <= 0)) r.reset();
    return r;
}

std::optional<KernelFacts> r8_row(const Card& c, const std::string& codec, int Wp, int R, int nt,
                                  Epilogue ep, bool pre) {
    if (codec.empty()) return std::nullopt;
    auto r = kf::r8_facts(c.arch, kf::R8KernelKey{codec, Wp, R, nt, epi_of(ep), pre});
    if (r && (!has_loop(*r) || r->registers <= 0)) r.reset();
    return r;
}

// ---- the board ----
std::optional<Card> card_from(int cc, int sms, double mhz, double bus_bytes_ns) {
    const IsaRow* isa = nullptr;
    std::optional<IsaRow> isa_r;
    std::string arch;
    for (const std::string a : {std::to_string(cc), std::to_string(cc) + "a"}) {
        for (const IsaRow& r : isa_facts_table()) {
            if (r.arch == a) {
                isa_r = r;
                arch  = a;
                break;
            }
        }
        if (isa_r) break;
    }
    if (!isa_r) return std::nullopt;
    isa             = &*isa_r;
    const auto rate = sm_rate_row(cc / 10, cc % 10);
    bool snapshot   = false;
    for (const std::string_view a : kf::snapshot_archs()) snapshot = snapshot || a == arch;
    if (!rate || rate->arch != arch || !isa->launch || !isa->launch->node_ns || !snapshot) {
        return std::nullopt;
    }
    const arch_registry::ArchRow& lim = arch_registry::find(cc);
    Card c;
    c.cc                                = cc;
    c.arch                              = arch;
    c.sms                               = sms;
    c.clk                               = mhz * 1e-3;
    c.bus                               = bus_bytes_ns;
    c.bpc                               = c.bus / (c.sms * c.clk);
    c.regs_sm                           = lim.registers_per_sm;
    c.thr_sm                            = lim.max_threads_per_sm;
    c.blk_sm                            = lim.max_blocks_per_sm;
    c.smem_sm                           = static_cast<double>(lim.smem_per_sm);
    c.l1                                = c.smem_sm + kL1MinKib * 1024.0;
    c.smem_res                          = lim.smem_reserved_per_block;
    const IsaLaunch& la                 = *isa->launch;
    c.l0                                = rate->dram_stream_latency_ns;
    c.node                              = *la.node_ns;
    c.stream                            = la.stream_ns;
    c.pdl                               = la.node_pdl_ns.has_value();
    c.imma_n                            = isa->imma_narrow_cycles_per_partition;
    c.imma_w                            = isa->imma_wide_cycles_per_partition;
    const std::optional<double>* need[] = {&isa->mma_dependent_latency,
                                           &isa->mma_cycles_per_partition,
                                           &isa->lat_imad,
                                           &isa->l1_latency.w1,
                                           &isa->l2_latency_ns.w1,
                                           &isa->int32_cycles_per_warp_instruction,
                                           &isa->fp32_cycles_per_warp_instruction,
                                           &isa->lat_shfl,
                                           &isa->shared_latency,
                                           &isa->l1_ldg32_cycles_per_sm,
                                           &isa->l2_bytes_per_sm_clock,
                                           &isa->dram_latency.w1,
                                           &isa->l2_latency.w1};
    for (const auto* n : need)
        if (!n->has_value()) return std::nullopt;
    if (!rate->warp_fill_bytes_per_clock) return std::nullopt;
    c.mma_lat  = *isa->mma_dependent_latency;
    c.hmma     = *isa->mma_cycles_per_partition;
    c.lat      = *isa->lat_imad;
    c.fill     = *rate->warp_fill_bytes_per_clock;
    c.ring_free = rate->ring2_linear_to_w_free * rate->ring2_b_w_w1_free;
    c.l1lat    = *isa->l1_latency.w1;
    c.l2lat_ns = *isa->l2_latency_ns.w1;
    c.int_cyc  = *isa->int32_cycles_per_warp_instruction;
    c.fp_cyc   = *isa->fp32_cycles_per_warp_instruction;
    c.shfl     = *isa->lat_shfl;
    c.smem_lat = *isa->shared_latency;
    c.l1cyc    = *isa->l1_ldg32_cycles_per_sm;
    c.l2bpc    = *isa->l2_bytes_per_sm_clock;
    c.dramlat  = *isa->dram_latency.w1;
    // The L2 round trip of four warps (the stream-k fixup's walk), the one-warp trip where the
    // probe has no w4 row.
    c.l2trip       = isa->l2_latency.w4 ? *isa->l2_latency.w4 : *isa->l2_latency.w1;
    const int lsms = la.sms;
    const std::pair<int, const std::optional<double>*> pts[] = {{1, &la.floor_ns_1},
                                                                {lsms, &la.floor_ns_sms},
                                                                {4 * lsms, &la.floor_ns_4sms},
                                                                {1024, &la.floor_ns_1024},
                                                                {4096, &la.floor_ns_4096}};
    for (const auto& [n, v] : pts)
        if (n > 0 && v->has_value()) c.floor.emplace_back(n, **v);
    std::sort(c.floor.begin(), c.floor.end());
    if (c.floor.size() < 2) return std::nullopt;
    const Narrow* nr = nullptr;
    for (const Narrow& x : kNarrow)
        if (x.arch == arch) nr = &x;
    if (nr == nullptr) return std::nullopt;
    c.nr_cap  = nr->cap;
    c.nr_wf   = nr->wf;
    c.nr_bank = nr->bank;
    if (isa->cluster && isa->cluster->sms > 0) {
        for (const int C : {2, 8}) {
            const IsaClusterSize* f = isa->cluster->size(C);
            if (!f || !f->sync_ns || !f->rank_ns_s1 || !f->rank_ns_s2 || !f->slots) continue;
            (C == 2 ? c.cluster2 : c.cluster8) = Card::ClusterSize{
                *f->sync_ns, *f->rank_ns_s1, *f->rank_ns_s2, *f->slots / isa->cluster->sms};
        }
    }
    return c;
}

// ---- shared terms ----
double occupancy(const Card& c, double warps, double regs, double smem = 0.0) {
    const long long unit = static_cast<long long>(std::ceil(regs * 32.0 / 256.0) * 256.0);
    const long long by_regs =
        unit > 0 ? (c.regs_sm / unit) / static_cast<long long>(warps) : 1LL << 30;
    double lim = static_cast<double>(
        std::min<long long>({by_regs, c.thr_sm / static_cast<long long>(warps * 32),
                             static_cast<long long>(c.blk_sm)}));
    if (smem > 0) lim = std::min(lim, std::floor(c.smem_sm / (smem + c.smem_res)));
    return std::max(lim, 1.0);
}

double launch_ns(const Card& c, const Settings& s, double ctas) {
    const auto& fl = c.floor;
    const double n = std::max(ctas, 1.0);
    const auto& a  = fl[fl.size() - 2];
    const auto& b  = fl.back();
    double f       = b.second + (n - b.first) * (b.second - a.second) / (b.first - a.first);
    for (std::size_t i = 0; i + 1 < fl.size(); ++i) {
        if (n <= fl[i + 1].first) {
            f = fl[i].second + (fl[i + 1].second - fl[i].second) * (n - fl[i].first) /
                                   (fl[i + 1].first - fl[i].first);
            break;
        }
    }
    const double base = s.launch == LaunchMode::Stream && c.stream ? *c.stream : c.node;
    return base + std::max(f - fl.front().second, 0.0);
}

double second_kernel(const Card& c, const Settings& s) {
    return s.second_kernel ? (c.stream ? *c.stream : c.node) : 0.0;
}

double window(const Card& c, double bw) {
    return c.l0 + std::max(bw - 512.0, 0.0) / c.fill / c.clk;
}

bool latency_launched(const Card& c, Epilogue ep, int A, int B, int M, double n,
                      const KernelFacts& plain, const std::optional<KernelFacts>& lat, int C) {
    if (!lat || c.bpc < lat_bytes(ep) || c.cc < 75) return false;
    if (ep == Epilogue::SwiGlu && c.cc < 80 && A * B * 32 > 128) return false;
    const double warps = A * B;
    const double var   = occupancy(c, warps, lat->registers),
                 pla   = occupancy(c, warps, plain.registers);
    if (var >= pla) return true;
    const double ctas = std::ceil(n / (16.0 * M * A)) * C;
    return std::ceil(ctas / (var * c.sms)) <= std::ceil(ctas / (pla * c.sms));
}

double price_rmma(const Card& c, const Settings& s, const KernelFacts& r, bool latency_row,
                  double n, double k, double T, double pitch, int M, int N, int A, int B, int C,
                  double sets) {
    const double sms = c.sms, clk = c.clk, bus = c.bus;
    const double passes   = std::ceil(T / (8.0 * N));
    const double rows_cta = 16.0 * M * A, warps = A * B;
    const double ctas  = std::ceil(n / rows_cta) * passes * C;
    // A launch of C-CTA clusters holds fewer CTAs than a plain one (a cluster lives in one GPC; 2
    // an SM on GB20x whatever the threads): the cluster facts' slots an SM over this board's SMs,
    // in whole clusters (the nearest: the RTX 5090's 328 CTAs at C 8 give the RTX 5070 Ti 17
    // clusters, 136 CTAs, as its own launch holds), as CTAs an SM (#1228, #1234).
    const Card::ClusterSize* cl = C > 1 ? c.cluster(C) : nullptr;
    const double whole =
        cl ? C * std::max(1.0, std::floor(cl->slots_sm * sms / C + 0.5)) / sms : 0.0;
    const double occ =
        cl ? std::min(occupancy(c, warps, r.registers), whole) : occupancy(c, warps, r.registers);
    const double kiter = loop_i(r, &LoopCounts::k_iter), slots = sms * occ;
    const double trips = k / (B * C * kiter);
    const double nfull = std::floor(ctas / slots);
    const double tail  = ctas - nfull * slots;
    const double cp    = ctas / passes;
    const double share =
        passes > 1 ? std::min(std::max((slots - cp) / std::max(ctas - cp, 1e-9), 0.0), 1.0) : 1.0;
    const double bcta  = n * pitch * (1.0 + (passes - 1.0) * (1.0 - share)) / ctas;
    // The cN epilogue (generic_gemm.cuh): two cluster.sync() and rank 0's reads of the C - 1
    // other ranks' partials, one rank after another (#pragma unroll 1), of `sets` weight sets.
    const double out = c.lat * sched(r, &StaticSchedule::outside_stall) / clk +
                       (cl ? 2.0 * cl->sync + (C - 1) * (sets > 1.5 ? cl->rank2 : cl->rank1) : 0.0);
    const double red   = (passes - 1.0) / passes;
    const double steps = 2.0 - (A * B <= 8 && latency_row ? 1.0 : 0.0);
    const double bw    = loop_z(r, &LoopCounts::ldg_bytes) + loop_z(r, &LoopCounts::ld_bytes) +
                      loop_z(r, &LoopCounts::ldgsts_bytes);
    // A codebook format's lookups in shared memory (the loop's LDS) each expose the shared
    // latency on the chain, as ggml_mmq's chain charges its shared waits (round 24, #1236).
    const double w_dram =
        steps * window(c, bw / steps) + loop_i(r, &LoopCounts::lds) * c.smem_lat / clk;
    const double st         = sched(r, &StaticSchedule::stall_sum) / clk;
    const double tot        = loop_i(r, &LoopCounts::total);
    const std::string shape = mma_shape(r);
    const bool emulated     = shape.size() >= 4 && shape.compare(shape.size() - 4, 4, ".EMU") == 0;
    const double mma        = loop_i(r, &LoopCounts::mma);
    const double emu        = emulated ? mma * kEmuIssue : 0.0;
    const double w_lat =
        emulated ? mma * kEmuLatNs[std::min(4, static_cast<int>(M * N * sets))] : 0.0;
    const double own  = w_dram + st + w_lat;
    const double iint = loop_i(r, &LoopCounts::int_alu);

    const auto rnd = [&](double cps, double nc, double o) {
        const double dram =
            nc * bcta / bus * std::pow(std::max(1.0, cps * rows_cta * pitch / c.l1), s.g);
        const double cm = std::max(nc / sms, 1.0);
        const double tau =
            std::max({own, red * cps * warps * iint / kPartitions / clk + (1.0 - red) * own,
                      // the issue bound: the warps of the busiest SM's CTAs, an instruction a
                      // clock a partition
                      cps * warps * tot / kPartitions / clk,
                      emu > 0 ? cm * warps * (tot + emu) / kPartitions / clk : 0.0});
        const double sm = trips * tau + o + nc * bcta / bus / static_cast<double>(B);
        return std::sqrt(sm * sm + dram * dram);
    };
    const auto part = [&](double x) { return std::max(std::min(std::ceil(x / sms), occ), 1.0); };
    const double ln = launch_ns(c, s, ctas);
    if (nfull < 1) return ln + rnd(part(ctas), tail, out);
    return ln + rnd(occ, slots, out) + (nfull - 1.0) * rnd(occ, slots, out) +
           (tail > 0 ? rnd(part(tail), tail, out) : 0.0);
}

int mmq_j(double T) { return T <= 16 ? 16 : T <= 32 ? 32 : T <= 64 ? 64 : 128; }

bool mmq_takes(const Call& call) {
    return call.n % ggml_mmq::kRowMultiple == 0 && call.k % ggml_mmq::kKMultiple == 0;
}

double mmq_fixup_ns(const Card& c, const Settings& s, const mmq_columns::ColumnLaunch& l, int j) {
    if (!l.fixup) return 0.0;
    const double ctas =
        std::min(std::max(l.split_tiles * (kMmqTile / 32), 1), static_cast<int>(c.sms));
    const double bytes_ns =
        (l.partials + 2.0 * l.split_tiles) * kMmqTile * j * 4.0 / (c.l2bpc * ctas * c.clk);
    const double walk_ns = l.rounds > 0 ? (l.rounds + 1) * c.l2trip / c.clk : 0.0;
    return launch_ns(c, s, static_cast<double>(l.blocks) * (kMmqTile / 32)) +
           std::max(bytes_ns, walk_ns);
}

double price_mmq(const Card& c, const Settings& s, const KernelFacts& r, double n, double k,
                 double T, double bits, double sets, Epilogue ep) {
    const double sms = c.sms, clk = c.clk, bus = c.bus;
    const auto quant = [&](double ctas, double out_per_value) {
        return launch_ns(c, s, ctas) + c.l0 + second_kernel(c, s) +
               T * k * (2.0 + out_per_value) / (c.l2bpc * std::min(ctas, sms) * clk);
    };
    const double out   = c.lat * sched(r, &StaticSchedule::outside_stall) / clk;
    const double pre   = quant(T * std::ceil(k / 512.0), 288.0 / 256.0);
    const double nt    = std::ceil(n / 128.0);
    const int j        = mmq_j(T);
    const auto launch  = mmq_columns::launch_of(static_cast<int>(n), static_cast<int>(k),
                                                static_cast<int>(T), j, static_cast<int>(sms));
    const double ctas  = launch.blocks;
    const double per   = launch.units * sets / std::ceil(ctas / sms);
    const double cyc   = mma_shape(r).find("16832") != std::string::npos ? *c.imma_w : *c.imma_n;
    const double chain = (sched(r, &StaticSchedule::stall_sum) +
                          pipe(r, &PipeCounts::sb_waits_shared) * c.smem_lat) /
                         clk;
    const double issue =
        std::max(kMmqWarps * loop_i(r, &LoopCounts::mma) * cyc / kPartitions / clk, chain);
    const double resident = launch.split ? ctas : std::min<double>(launch.tiles, std::floor(sms));
    const double weight_tiles =
        std::min(resident, std::ceil(nt * std::max(1.0, std::ceil(k / 256.0)) / launch.units));
    const double l2_ns     = resident * (4096.0 * bits + j * 288.0) / (c.l2bpc * resident * clk);
    const double load      = std::max(4096.0 * bits * weight_tiles / bus, l2_ns) + c.dramlat / clk;
    const double unit      = per * (issue + load);
    const double epi_bytes = n * T * (4.0 * sets + 2.0 + (ep == Epilogue::AddResidual ? 2.0 : 0.0));
    const double epi =
        launch_ns(c, s, std::ceil(n / 1024.0) * T) + epi_bytes / (c.l2bpc * sms * clk);
    const double fix = mmq_fixup_ns(c, s, launch, j);
    return pre +
           sets * (launch_ns(c, s, ctas) + fix + std::ceil(ctas / sms) * (unit / sets + out)) + epi;
}

double price_r8(const Card& c, const Settings& s, const KernelFacts& r, double n, double k,
                double T, double bits, double sets, int Wp, int R, int nt, double act,
                double smem) {
    const double sms = c.sms, clk = c.clk, bus = c.bus, pitch = sets * k * bits / 8.0;
    const auto quant = [&](double ctas, double out_per_value) {
        return launch_ns(c, s, ctas) + c.l0 + second_kernel(c, s) +
               T * k * (2.0 + out_per_value) / (c.l2bpc * std::min(ctas, sms) * clk);
    };
    const double pre   = quant(std::ceil(T * k / 8192.0), 1.0 + 8.0 / 32.0);
    const double out   = c.lat * sched(r, &StaticSchedule::outside_stall) / clk;
    const double ctas  = std::ceil(n / static_cast<double>(Wp * R));
    const double slots = sms * occupancy(c, Wp, r.registers, smem);
    const double trips = k / std::max(loop_i(r, &LoopCounts::k_iter), 1.0);
    const double gw    = pro(r, &PrologueCounts::loop_global_waits);
    const double trip  = sched(r, &StaticSchedule::stall_sum) / clk +
                        gw * window(c, loop_z(r, &LoopCounts::ldg_bytes) / std::max(gw, 1.0));
    double chain = trips * trip + out;
    if (T > 1) chain *= T / std::max(nt, 1); // an NT-token kernel runs T of its NT slots (round 24)
    if (T <= 1) {
        chain += k / 1024.0 / Wp *
                 (pro(r, &PrologueCounts::stall_sum) / clk +
                  pro(r, &PrologueCounts::global_waits) * c.l2lat_ns);
    }
    const double rounds = std::ceil(ctas / slots);
    const double last   = ctas - (rounds - 1) * slots;
    const auto rnd      = [&](double nc) {
        return std::hypot(chain,
                               std::max(nc * Wp * R * pitch / bus,
                                        nc * Wp * trips * act * c.nr_wf / (std::min(nc, sms) * clk)));
    };
    return launch_ns(c, s, ctas) + (T <= 1 ? 0.0 : pre) + (rounds - 1) * rnd(slots) + rnd(last);
}

// ---- ggml_vec ----
int vec_ks(Epilogue ep, int k, std::string_view inst) {
    const bool gated  = ep == Epilogue::SwiGlu;
    const bool listed = std::find(std::begin(kVecKs), std::end(kVecKs), k) != std::end(kVecKs);
    return inst != "loop" && listed && (!gated || k == 5120) ? k : 0;
}

struct VecRow {
    KernelFacts f;
    int ks = 0;
};

std::optional<VecRow> vec_row(const Card& c, QType q, Epilogue ep, int k, int T, bool bounded,
                              std::string_view inst) {
    const FormatFacts& ff = format_facts(q);
    if (ff.ggml_type < 0) return std::nullopt; // analyzer.py MMQ_TYPE: the format's ggml type
    const int ks = vec_ks(ep, k, inst);
    const auto r = kf::vec_facts(
        c.arch, kf::VecKernelKey{ff.ggml_type, T, ks, ep == Epilogue::SwiGlu, bounded});
    if (!r || r->registers <= 0) return std::nullopt;
    return VecRow{*r, ks};
}

int vec_occ(const Card& c, const KernelFacts& r) {
    const long long regs = static_cast<long long>(std::ceil(r.registers * 32.0 / 256.0) * 256.0);
    long long occ =
        std::min<long long>({(c.regs_sm / regs) / kVecWarps, c.thr_sm / (kVecWarps * 32),
                             static_cast<long long>(c.blk_sm)});
    const double sh = static_cast<double>(r.static_shared_bytes);
    if (sh > 0)
        occ = std::min<long long>(occ, static_cast<long long>(c.smem_sm / (sh + c.smem_res)));
    return static_cast<int>(std::max<long long>(occ, 1));
}

struct VecLaunch {
    VecRow row;
    int ctas = 0, groups = 0;
};

std::optional<VecLaunch> vec_launch(const Card& c, QType q, Epilogue ep, double n, int k, int T,
                                    std::string_view inst) {
    auto r = vec_row(c, q, ep, k, T, false, inst);
    if (!r) return std::nullopt;
    const int groups =
        static_cast<int>(std::ceil(n / (kVecWarps * (ep == Epilogue::SwiGlu ? 1.0 : 2.0))));
    int slots = vec_occ(c, r->f) * static_cast<int>(c.sms);
    if (T == 4 && ep != Epilogue::SwiGlu && r->ks == 0) {
        const auto tw = vec_row(c, q, ep, k, T, true, inst);
        if (tw && tw->f.local_bytes == 0 && tw->f.stack_bytes == 0) {
            const int ts = std::min(vec_occ(c, tw->f), kVecTwinBlocks) * static_cast<int>(c.sms);
            if (std::ceil(static_cast<double>(groups) / ts) <
                std::ceil(static_cast<double>(groups) / slots)) {
                r     = tw;
                slots = ts;
            }
        }
    }
    return VecLaunch{*r, std::min(groups, slots), groups};
}

// ggml_vec.cu Decoder<type>::decode, one row of one 1024-value slice: (bytes a lane, bytes a
// block in that plane, offsets by u = lane % (lanes a block)). The plane letter is not priced.
struct VecLoad {
    int width, block;
    std::array<int, 8> offs;
};

struct VecLoads {
    int per;
    std::vector<VecLoad> loads;
};

std::array<int, 8> offsets(int (*f)(int)) {
    std::array<int, 8> a{};
    for (int u = 0; u < 8; ++u) a[static_cast<std::size_t>(u)] = f(u);
    return a;
}

const VecLoads* vec_loads(QType q) {
    static const std::map<QType, VecLoads> table = [] {
        std::map<QType, VecLoads> t;
        const auto zero = offsets([](int) { return 0; });
        const auto k8   = [](int base, int blk) {
            return VecLoad{
                1,
                blk,
                  {base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7}};
        };
        const auto k4 = [](int base, int blk) {
            return VecLoad{
                1, blk, {base, base + 1, base + 2, base + 3, base, base + 1, base + 2, base + 3}};
        };
        const auto pair32 = [](int base, int blk) {
            std::array<int, 8> a{};
            for (int u = 0; u < 8; ++u) a[static_cast<std::size_t>(u)] = base + 32 * (u / 2);
            return VecLoad{16, blk, a};
        };
        const auto quad32 = [](int base, int blk) {
            std::array<int, 8> a{};
            for (int u = 0; u < 8; ++u) a[static_cast<std::size_t>(u)] = base + 32 * (u / 4);
            return VecLoad{16, blk, a};
        };
        const auto scale_h = [](int e, int blk) {
            std::array<int, 8> a{};
            for (int u = 0; u < 8; ++u)
                a[static_cast<std::size_t>(u)] = 8 * (u / 4) + 2 * (u % 4) + e;
            return VecLoad{1, blk, a};
        };
        const auto stride = [](int width, int blk, int step, int base) {
            std::array<int, 8> a{};
            for (int u = 0; u < 8; ++u) a[static_cast<std::size_t>(u)] = base + step * u;
            return VecLoad{width, blk, a};
        };
        t[QType::GGML_Q4_K] = {8,
                               {{4, 144, zero},
                                k8(4, 144),
                                k8(8, 144),
                                k4(4, 144),
                                k4(8, 144),
                                pair32(16, 144),
                                pair32(32, 144)}};
        t[QType::GGML_Q5_K] = {8,
                               {{4, 176, zero},
                                k8(4, 176),
                                k8(8, 176),
                                k4(4, 176),
                                k4(8, 176),
                                {16, 176, offsets([](int) { return 16; })},
                                {16, 176, offsets([](int) { return 32; })},
                                pair32(48, 176),
                                pair32(64, 176)}};
        t[QType::GGML_Q2_K] = {
            8, {{4, 4, zero}, scale_h(0, 16), scale_h(1, 16), quad32(0, 64), quad32(16, 64)}};
        {
            VecLoads q6{8, {{2, 2, zero}, scale_h(0, 16), scale_h(1, 16)}};
            for (int i = 0; i < 8; ++i)
                for (const int d : {0, 2}) {
                    std::array<int, 8> a{};
                    for (int u = 0; u < 8; ++u)
                        a[static_cast<std::size_t>(u)] = 64 * (u / 4) + 32 * (u & 1) + 4 * i + d;
                    q6.loads.push_back({2, 192, a});
                }
            for (int i = 0; i < 8; ++i)
                for (const int d : {0, 2}) {
                    std::array<int, 8> a{};
                    for (int u = 0; u < 8; ++u)
                        a[static_cast<std::size_t>(u)] = 128 + 32 * (u / 4) + 4 * i + d;
                    q6.loads.push_back({2, 192, a});
                }
            t[QType::GGML_Q6_K] = q6;
        }
        t[QType::GGML_IQ3_XXS] = {8, {stride(8, 64, 8, 0), stride(4, 32, 4, 0), {2, 2, zero}}};
        t[QType::GGML_IQ3_S]   = {8,
                                  {stride(8, 104, 8, 0),
                                   stride(1, 104, 1, 64),
                                   stride(4, 104, 4, 72),
                                   {1, 4, offsets([](int u) { return u / 2; })},
                                   {2, 2, zero}}};
        {
            VecLoads q3{8, {{2, 2, zero}}};
            for (const int e : {0, 1}) {
                std::array<int, 8> a{};
                for (int u = 0; u < 8; ++u)
                    a[static_cast<std::size_t>(u)] = (8 * (u / 4) + 2 * (u % 4) + e) % 8;
                q3.loads.push_back({1, 12, a});
            }
            for (const int e : {0, 1}) {
                std::array<int, 8> a{};
                for (int u = 0; u < 8; ++u)
                    a[static_cast<std::size_t>(u)] = 8 + (8 * (u / 4) + 2 * (u % 4) + e) % 4;
                q3.loads.push_back({1, 12, a});
            }
            q3.loads.push_back({16, 96, offsets([](int) { return 0; })});
            q3.loads.push_back({16, 96, offsets([](int) { return 16; })});
            q3.loads.push_back(quad32(32, 96));
            q3.loads.push_back(quad32(48, 96));
            t[QType::GGML_Q3_K] = q3;
        }
        t[QType::GGML_IQ2_XXS] = {8, {stride(8, 64, 8, 0), {2, 2, zero}}};
        t[QType::GGML_IQ2_XS]  = {8, {stride(8, 64, 8, 0), {2, 2, zero}, stride(1, 8, 1, 0)}};
        t[QType::GGML_IQ2_S]   = {8,
                                  {stride(4, 72, 4, 0),
                                   stride(4, 72, 4, 32),
                                   stride(1, 72, 1, 64),
                                   {2, 2, zero},
                                   stride(1, 8, 1, 0)}};
        t[QType::GGML_IQ1_S]   = {8, {stride(4, 32, 4, 0), stride(2, 16, 2, 0), {2, 2, zero}}};
        t[QType::GGML_IQ1_M]   = {8,
                                  {{2, 8, zero},
                                   {2, 8, offsets([](int) { return 2; })},
                                   {2, 8, offsets([](int) { return 4; })},
                                   {2, 8, offsets([](int) { return 6; })},
                                   stride(4, 32, 4, 0),
                                   stride(2, 16, 2, 0)}};
        t[QType::GGML_IQ4_XS]  = {8,
                                  {{2, 4, zero},
                                   {2, 4, offsets([](int) { return 2; })},
                                   {1, 4, offsets([](int u) { return u / 2; })},
                                   stride(8, 128, 16, 0),
                                   stride(8, 128, 16, 8)}};
        t[QType::GGML_Q4_0]    = {1, {{16, 16, zero}, {2, 2, zero}}};
        t[QType::GGML_IQ4_NL]  = {1, {{16, 16, zero}, {2, 2, zero}}};
        t[QType::GGML_Q4_1]    = {1, {{16, 16, zero}, {4, 4, zero}}};
        t[QType::GGML_Q5_0]    = {1, {{16, 16, zero}, {4, 4, zero}, {2, 2, zero}}};
        t[QType::GGML_Q5_1]    = {1, {{16, 16, zero}, {4, 4, zero}, {4, 4, zero}}};
        return t;
    }();
    const auto it = table.find(q);
    return it == table.end() ? nullptr : &it->second;
}

double load_clocks(const Card& c, const std::vector<std::pair<int, int>>& lanes) {
    std::map<int, std::set<int>> by;
    const int bank = c.nr_bank;
    for (const auto& [a, w] : lanes) {
        for (int b = a / bank; b <= (a + w - 1) / bank; ++b)
            by[b % (128 / bank)].insert(b * bank / 128);
    }
    std::size_t most = 0;
    for (const auto& [b, v] : by) most = std::max(most, v.size());
    return std::max(c.nr_cap, static_cast<double>(most) * c.nr_wf);
}

std::optional<std::pair<double, double>> vec_load_unit(const Card& c, QType q) {
    static std::mutex mu;
    static std::map<std::tuple<double, double, int, QType>, std::pair<double, double>> cache;
    const auto key = std::make_tuple(c.nr_cap, c.nr_wf, c.nr_bank, q);
    {
        std::lock_guard lock(mu);
        if (const auto it = cache.find(key); it != cache.end()) return it->second;
    }
    const VecLoads* L = vec_loads(q);
    if (L == nullptr) return std::nullopt;
    double uw = 0.0;
    for (const VecLoad& ld : L->loads) {
        double tot = 0.0;
        for (int al = 0; al < 128; al += 4) {
            std::vector<std::pair<int, int>> lanes;
            for (int l = 0; l < 32; ++l)
                lanes.emplace_back(al + (l / L->per) * ld.block +
                                       ld.offs[static_cast<std::size_t>(l % L->per)],
                                   ld.width);
            tot += load_clocks(c, lanes);
        }
        uw += tot / 32.0;
    }
    // A token's activation loads: ds, the two int4 of qs.
    double ua                       = 0.0;
    const std::pair<int, int> act[] = {{4, 4}, {16, 32}, {16, 32}};
    for (int i = 0; i < 3; ++i) {
        std::vector<std::pair<int, int>> lanes;
        for (int l = 0; l < 32; ++l)
            lanes.emplace_back(act[i].second * l + (i == 2 ? 16 : 0), act[i].first);
        ua += load_clocks(c, lanes);
    }
    std::lock_guard lock(mu);
    return cache[key] = {uw, ua};
}

double sm_side(const Card& c, bool og, bool ib, bool lb, double tot, double stall, double o_tot,
               double o_stall, double trips, double cps, double dram_trips, double bw,
               double ld_clk, double warps = kVecWarps, double extra = 0.0) {
    const double clk = c.clk;
    double sm = (trips * stall + (og ? o_stall : 0.0)) / clk + dram_trips * window(c, bw) + extra;
    if (ib) sm = std::max(sm, cps * (trips * tot + (og ? o_tot : 0.0)) / clk);
    if (lb) sm = std::max(sm, cps * warps * ld_clk / clk);
    return sm;
}

double combine(RoundForm f, double sm, double d) {
    return f == RoundForm::Hypot ? std::hypot(sm, d)
           : f == RoundForm::Sum ? sm + d
                                 : std::max(sm, d);
}

std::optional<double> price_vec(const Card& c, const Settings& s, QType q, Epilogue ep, double n,
                                int k, int T, double bits, double sets, std::string_view inst) {
    const auto L = vec_launch(c, q, ep, n, k, T, inst);
    if (!L) return std::nullopt;
    VecRow qrow = L->row;
    if (loop_i(qrow.f, &LoopCounts::total) <= 0) { // unknown flow: the plain instance's counts
        const auto p = vec_row(c, q, Epilogue::Store, k, T, false, inst);
        if (!p) return std::nullopt;
        qrow = *p;
    }
    const double bg      = n * sets * k * bits / 8.0 / L->groups;
    const double rounds  = std::ceil(static_cast<double>(L->groups) / L->ctas);
    const double last    = L->groups - (rounds - 1) * L->ctas;
    const KernelFacts& r = qrow.f;
    const double tot = loop_i(r, &LoopCounts::total), stall = sched(r, &StaticSchedule::stall_sum);
    const double o_tot   = sched(r, &StaticSchedule::outside_total),
                 o_stall = sched(r, &StaticSchedule::outside_stall);
    const bool unrolled  = qrow.ks != 0;
    const double slices  = k / 1024.0;
    const double trips   = unrolled ? 1.0 : slices;
    const double nld     = loop_i(r, &LoopCounts::ldg) + loop_i(r, &LoopCounts::lds);
    double ld;
    const auto lu = s.narrow_loads ? vec_load_unit(c, q) : std::nullopt;
    if (!lu) {
        ld = trips * nld * c.l1cyc;
    } else {
        ld = slices * (2.0 * lu->first + T * lu->second) +
             trips * loop_i(r, &LoopCounts::lds) * c.nr_cap;
    }
    const double bw = bg / kVecWarps / (k / 1024.0);
    // The trip's waits on global loads beyond those its loads in flight cover (the slices it
    // spans x ldg_peak): each re-reads a load's bytes (round 25, #1236; the T 6 / 7 gated
    // instances wait 22-23 times a trip with 12-13 loads in flight, T 8 13 times with 16).
    const double peak = std::max(pipe(r, &PipeCounts::ldg_peak), 1.0);
    const double excess =
        std::max(0.0, pipe(r, &PipeCounts::sb_waits_global) - peak * slices / trips);
    const double n_ld = loop_i(r, &LoopCounts::ldg) + loop_i(r, &LoopCounts::ld);
    const double load_bytes =
        (loop_z(r, &LoopCounts::ldg_bytes) + loop_z(r, &LoopCounts::ld_bytes)) /
        std::max(n_ld, 1.0);
    const double act = static_cast<double>(T) * k * 36.0 / 32.0; // the q8_1 activation
    const double rb  = k * bits / 8.0; // a row's bytes in 128-B L1 lines, one more where unaligned
    const double row_lines =
        (std::ceil(rb / 128.0) + (std::fmod(rb, 128.0) != 0.0 ? 1.0 : 0.0)) * 128.0;
    const auto rnd = [&](double nc) {
        const double d   = nc * bg / c.bus;
        const double cps = std::ceil(nc / c.sms);
        double sm = sm_side(c, s.vec_outside, s.vec_issue, s.vec_load, tot, stall, o_tot, o_stall,
                            trips, cps, k / 1024.0, bw, ld, kVecWarps, 0.0);
        // The SM's L2 link carries every resident warp's trip bytes and, where the rows its CTAs
        // keep open and the activation exceed L1, the re-reads: the share of them past the bytes
        // the SM keeps in flight at a linear rate (ring_free) queues there.
        const double opn = cps * bg / rb * row_lines;
        double reread    = opn + act > c.l1 ? excess * load_bytes : 0.0;
        reread *= std::max(0.0, 1.0 - c.ring_free / (cps * kVecWarps * bw));
        sm = std::max(sm,
                      cps * kVecWarps * trips * (bw * slices / trips + reread) / (c.l2bpc * c.clk));
        return combine(s.vec_form, sm, d);
    };
    return launch_ns(c, s, L->ctas) + launch_ns(c, s, std::ceil(k / 256.0) * T) +
           second_kernel(c, s) + c.l0 + (rounds - 1) * rnd(L->ctas) + rnd(last);
}

std::optional<double> price_vec_mma(const Card& c, const Settings& s, QType q, Epilogue ep,
                                    double n, double k, int T, double bits, double sets) {
    if ((q != QType::GGML_Q4_K && q != QType::GGML_Q5_K && q != QType::GGML_Q6_K) || c.cc < 80) {
        return std::nullopt;
    }
    const auto row = kf::vec_mma_facts(c.arch, ggml_type(q), ep == Epilogue::SwiGlu);
    if (!row || row->registers <= 0 || loop_i(*row, &LoopCounts::mma) <= 0) return std::nullopt;
    const KernelFacts& r  = *row;
    const double rows_cta = ep == Epilogue::SwiGlu ? 8.0 : 16.0;
    const double ctas     = std::ceil(n / rows_cta);
    const double occ      = vec_occ(c, r);
    const double slots    = c.sms * occ;
    const double trips    = k / 256.0 / kVecWarps;
    const double bw       = loop_z(r, &LoopCounts::ldg_bytes) + loop_z(r, &LoopCounts::ld_bytes);
    const double bc       = rows_cta * sets * k * bits / 8.0;
    const double rounds   = std::ceil(ctas / slots);
    const double last     = ctas - (rounds - 1) * slots;
    const auto rnd        = [&](double nc) {
        const double d = nc * bc / c.bus;
        const double sm =
            sm_side(c, s.mma_outside, s.mma_issue, false, loop_i(r, &LoopCounts::total),
                           sched(r, &StaticSchedule::stall_sum), sched(r, &StaticSchedule::outside_total),
                           sched(r, &StaticSchedule::outside_stall), trips,
                           std::min(std::ceil(nc / c.sms), occ), trips, bw,
                           trips * (loop_i(r, &LoopCounts::ldg) + loop_i(r, &LoopCounts::lds)) * c.l1cyc);
        return s.vec_form == RoundForm::Hypot ? std::hypot(sm, d) : sm + d;
    };
    return launch_ns(c, s, ctas) + launch_ns(c, s, std::ceil(k / 256.0) * T) + second_kernel(c, s) +
           c.l0 + (rounds - 1) * rnd(slots) + rnd(last);
}

// ---- Volta's substitutes (generic_routes.cpp volta_hmma_takes / volta_gemv_takes) ----
enum class VoltaKind { None, Hmma, Gemv };

VoltaKind volta_takes(QType q, int k, int T) {
    int hk = 0, gk = 0;
    switch (q) {
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
        hk = 256, gk = 256;
        break;
    case QType::Q8_G32_FP16:
        hk = 64, gk = 256;
        break;
    case QType::BF16:
        gk = 8;
        break;
    default:
        break;
    }
    if (hk > 0 && T >= 1 && T <= 16 && k % hk == 0) return VoltaKind::Hmma;
    if (gk > 0 && T >= 1 && T <= 4 && k % gk == 0) return VoltaKind::Gemv;
    return VoltaKind::None;
}

std::pair<std::string_view, std::string_view> volta_fmt(QType q) {
    switch (q) {
    case QType::GGML_Q4_K:
        return {"HK45<4>", "K45<4>"};
    case QType::GGML_Q5_K:
        return {"HK45<5>", "K45<5>"};
    case QType::GGML_Q6_K:
        return {"HK6", "K6"};
    case QType::Q8_G32_FP16:
        return {"HQ8", "Q8G32"};
    case QType::BF16:
        return {"", "Bf16Rows"};
    default:
        return {"", ""};
    }
}

std::optional<KernelFacts> section_row(const Card& c, std::string_view family,
                                       const std::string& key) {
    static std::mutex mu;
    static std::map<std::pair<std::string, std::string>, std::map<std::string, KernelFacts>> cache;
    std::lock_guard lock(mu);
    auto [it, fresh] = cache.try_emplace({c.arch, std::string(family)});
    if (fresh)
        for (auto& [k, f] : kf::section_rows(c.arch, family)) it->second.emplace(k, f);
    const auto r = it->second.find(key);
    if (r == it->second.end()) return std::nullopt;
    return r->second;
}

std::optional<double> volta_sub_price(const Card& c, const Settings& s, double n, double k,
                                      double T, double bits, double sets, QType q, Epilogue ep,
                                      VoltaKind kind) {
    const std::string E = std::to_string(epi_of(ep));
    const auto [hf, gf] = volta_fmt(q);
    double warps = 0, rows_cta = 0, ctas = 0, depth = 1.0;
    std::optional<KernelFacts> r;
    if (kind == VoltaKind::Hmma) {
        warps = 4, rows_cta = kVoltaCtaRows;
        ctas                    = std::ceil(n / rows_cta);
        const std::string ahead = ctas >= 2 * c.sms ? "1" : "2";
        r                       = section_row(c, "vhmma",
                                              std::string(hf) + ' ' + E + ' ' + (T <= 8 ? "1" : "2") + ' ' + ahead);
        depth                   = ahead == "1" ? 2.0 : 3.0;
    } else {
        const int R = ep == Epilogue::Store && std::ceil(n / 32) >= 4 * c.sms ? 4 : 1;
        warps = 8, rows_cta = 8.0 * R;
        ctas                 = std::ceil(n / rows_cta);
        const std::string nt = T <= 1 ? "1" : T <= 2 ? "2" : "4";
        r                    = R > 1 ? section_row(c, "vgemvr",
                                                   std::string(gf) + ' ' + E + ' ' + nt + ' ' + std::to_string(R))
                                     : section_row(c, "vgemv", std::string(gf) + ' ' + E + ' ' + nt);
    }
    if (!r) return std::nullopt;
    double occ          = occupancy(c, warps, r->registers);
    const double shared = static_cast<double>(r->static_shared_bytes);
    if (shared > 0)
        occ = std::max(std::min(occ, std::floor(c.smem_sm / (shared + c.smem_res))), 1.0);
    const double tot        = loop_i(*r, &LoopCounts::total),
                 stall      = sched(*r, &StaticSchedule::stall_sum);
    const double rows_warp  = kind == VoltaKind::Hmma ? 8.0 : rows_cta / warps;
    const double bytes_iter = rows_warp * 256.0 * bits / 8.0 * sets;
    const double iters      = std::ceil(k / 256.0);
    const double kloops =
        kind == VoltaKind::Hmma ? 1.0 : std::max(loop_i(*r, &LoopCounts::loops), 1.0);
    double t    = launch_ns(c, s, ctas) + c.l0;
    double left = ctas;
    while (left > 0) {
        const double nc    = std::min(left, occ * c.sms);
        const double w_sm  = std::ceil(nc / c.sms) * warps;
        const double issue = w_sm * tot / kPartitions / c.clk;
        const double chain = (tot + stall) / c.clk;
        const double dram  = nc * warps * bytes_iter / c.bus;
        t += iters * kloops * std::max({issue, chain, c.l0 / depth + dram / kloops});
        left -= nc;
    }
    return t;
}

// ---- route names ----
struct RmmaName {
    int M = 1, N = 1, A = 1, B = 1, C = 1;
};

std::optional<RmmaName> parse_rmma(std::string_view r) {
    // rmma.m<M>.n<N>.w<A>x<B>[.c<C>]
    if (r.substr(0, 6) != "rmma.m") return std::nullopt;
    RmmaName x;
    std::size_t i  = 6;
    const auto num = [&](int& out) {
        const std::size_t b = i;
        int v               = 0;
        while (i < r.size() && r[i] >= '0' && r[i] <= '9') v = v * 10 + (r[i++] - '0');
        out = v;
        return i > b;
    };
    const auto lit = [&](std::string_view s) {
        if (r.substr(i, s.size()) != s) return false;
        i += s.size();
        return true;
    };
    if (!num(x.M) || !lit(".n") || !num(x.N) || !lit(".w") || !num(x.A) || !lit("x") || !num(x.B)) {
        return std::nullopt;
    }
    if (i < r.size()) {
        if (!lit(".c") || !num(x.C) || i != r.size()) return std::nullopt;
    }
    return x;
}

std::optional<std::pair<int, int>> parse_r8(std::string_view r) {
    if (r.substr(0, 4) != "r8.w") return std::nullopt;
    std::size_t i = 4;
    int w = 0, rows = 0;
    const std::size_t b = i;
    while (i < r.size() && r[i] >= '0' && r[i] <= '9') w = w * 10 + (r[i++] - '0');
    if (i == b || i >= r.size() || r[i] != 'r') return std::nullopt;
    const std::size_t c = ++i;
    while (i < r.size() && r[i] >= '0' && r[i] <= '9') rows = rows * 10 + (r[i++] - '0');
    if (i == c || i != r.size()) return std::nullopt;
    return std::make_pair(w, rows);
}

std::optional<double> price(const Card& c, const Call& call, std::string_view route,
                            const Settings& s) {
    const FormatRow* fr = format_row(call.qtype);
    if (fr == nullptr) return std::nullopt;
    const double n = call.n, k = call.k, T = std::max<double>(call.tokens, 1.0);
    const int ik = call.k, iT = static_cast<int>(T);
    const Epilogue ep = call.epilogue;
    const double bits = fr->bits, sets = ep == Epilogue::SwiGlu ? 2.0 : 1.0;
    if (ik % format_k(call.qtype) != 0) return std::nullopt;
    if (const auto m = parse_rmma(route)) {
        const auto codec = codec_of(call.qtype);
        if (!codec || ik % detail::generic::k_multiple(*codec) != 0) return std::nullopt;
        if (s.volta_substitutes && c.cc == 70) {
            const VoltaKind vk = volta_takes(call.qtype, ik, iT);
            if (vk != VoltaKind::None)
                return volta_sub_price(c, s, n, k, T, bits, sets, call.qtype, ep, vk);
        }
        // Clusters are sm_90+, and a cN schedule is priced from its cluster facts.
        if (m->C > 1 && (c.cc < 90 || c.cluster(m->C) == nullptr)) return std::nullopt;
        const std::string dec = rmma_decoder(call.qtype, ik);
        if (dec.empty()) return std::nullopt;
        auto r = rmma_row(c, dec, m->A, m->B, m->M, m->N, m->C, ep, false);
        if (!r) return std::nullopt;
        bool lat_row = false;
        if (m->N == 1) {
            const auto lr = rmma_row(c, dec, m->A, m->B, m->M, m->N, m->C, ep, true);
            if (latency_launched(c, ep, m->A, m->B, m->M, n, *r, lr, m->C)) {
                r       = lr;
                lat_row = true;
            }
        }
        return price_rmma(c, s, *r, lat_row, n, k, T, sets * k * bits / 8.0, m->M, m->N, m->A, m->B,
                          m->C, sets);
    }
    if (const auto m = parse_r8(route)) {
        const auto [Wp, R] = *m;
        if (ik % detail::generic::r8_k_values(call.qtype) != 0) return std::nullopt;
        if (T > 8) return std::nullopt;
        const std::string codec = r8_codec(call.qtype);
        const int nt            = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
        const auto r            = r8_row(c, codec, Wp, R, nt, ep, T > 1);
        double act              = 0.0;
        if (T > 1) {
            int order[4] = {1, 2, 4, 8};
            std::stable_sort(std::begin(order), std::end(order),
                             [&](int a, int b) { return std::abs(a - nt) < std::abs(b - nt); });
            for (const int nt2 : order) {
                const auto a1 = nt2 == nt ? r : r8_row(c, codec, Wp, R, nt2, ep, true);
                const auto a0 = r8_row(c, codec, Wp, R, nt2, ep, false);
                if (a1 && a0) {
                    if (r) {
                        const double scale = nt / static_cast<double>(nt2);
                        const double n_add =
                            std::max(loop_i(*a1, &LoopCounts::ldg) - loop_i(*a0, &LoopCounts::ldg),
                                     0.0) *
                            scale;
                        const double b_add = std::max(loop_z(*a1, &LoopCounts::ldg_bytes) -
                                                          loop_z(*a0, &LoopCounts::ldg_bytes),
                                                      0.0) *
                                             scale;
                        const double codes =
                            std::max(std::min((b_add - 128.0 * n_add) / 384.0, n_add), 0.0);
                        const std::string base = codec.substr(0, codec.find('<'));
                        const double grp =
                            base == "W4a8Q45K"                                              ? 2.0
                            : (base == "W4a8Q6K" || base == "W4a8Q3K" || base == "W4a8Q2K") ? 4.0
                                                                                            : 1.0;
                        const double lines = std::min(32.0, 8.0 * grp);
                        act                = codes * lines + (n_add - codes) * std::min(grp, 32.0);
                    }
                    break;
                }
            }
        }
        // Its dynamic shared memory: at T 1 the CTA's quantized activation; below sm_80 a codebook
        // codec's grid.
        const bool sums = codec == "W4a8Q45K<4>" || codec == "W4a8Q45K<5>" ||
                          codec == "W4a8Ggml32<1>" || codec == "W4a8Ggml32<3>";
        double smem = T <= 1 ? std::ceil(k / 32.0 * (sums ? 40.0 : 36.0) / 8.0) * 8.0 : 0.0;
        if (c.cc < 80) {
            const std::string base                        = codec.substr(0, codec.find('<'));
            const std::pair<std::string_view, int> grid[] = {
                {"W4a8Iq3Xxs", 256}, {"W4a8Iq3S", 512},   {"W4a8Iq1S", 2048}, {"W4a8Iq1M", 2048},
                {"W4a8Iq2Xxs", 512}, {"W4a8Iq2Xs", 1024}, {"W4a8Iq2S", 2048}};
            for (const auto& [name, words] : grid)
                if (name == base) smem += 4.0 * words + 4.0;
        }
        if (T <= 1 && smem > kR8StaticSmem) return std::nullopt;
        if (!r) return std::nullopt;
        return price_r8(c, s, *r, n, k, T, bits, sets, Wp, R, nt, act, smem);
    }
    if (route == "ggml_mmq") {
        const FormatFacts& ff = format_facts(call.qtype);
        if (!c.imma_n || !c.imma_w || !mmq_takes(call) || ff.ggml_type < 0) return std::nullopt;
        auto r = kf::mmq_facts(c.arch, kf::MmqKernelKey{ff.ggml_type, mmq_j(T), false, 30});
        if (r && !has_loop(*r)) r.reset();
        if (!r) return std::nullopt;
        return price_mmq(c, s, *r, n, k, T, bits, sets, ep);
    }
    if (route == "ggml_vec.mma") {
        return T <= 8 ? price_vec_mma(c, s, call.qtype, ep, n, k, iT, bits, sets) : std::nullopt;
    }
    if (route == "ggml_vec" || route == "ggml_vec.ks" || route == "ggml_vec.loop") {
        const std::string_view inst = route == "ggml_vec.loop" ? "loop"
                                      : route == "ggml_vec.ks" ? "ks"
                                                               : "";
        if (!vec_row(c, call.qtype, ep, ik, iT, false, inst)) return std::nullopt;
        if (inst == "ks" && vec_ks(ep, ik, inst) == 0) return std::nullopt;
        return price_vec(c, s, call.qtype, ep, n, ik, iT, bits, sets, inst);
    }
    return std::nullopt;
}

} // namespace

std::string_view format_name(QType qtype) {
    const FormatRow* f = format_row(qtype);
    return f ? f->name : std::string_view{};
}

std::optional<QType> format_of_name(std::string_view name) {
    for (const FormatRow& f : kFormats)
        if (f.name == name) return f.q;
    return std::nullopt;
}

std::optional<Card> card_of(const GpuFactsRow& row) {
    const int mhz =
        row.form == FormFactor::Laptop && row.boost_min_mhz > 0 ? row.boost_min_mhz : row.boost_mhz;
    return card_from(row.major * 10 + row.minor, row.sms, mhz, row.bus_bits * row.gbps / 8.0);
}

std::optional<Card> card_of(const DeviceClass& device) {
    // The board's resolved facts (machine_facts: the row of its id and subsystem, a laptop's clock,
    // the driver's values on the facts grid for an unknown id): for a stock board the row's.
    const MachineFacts& f = device.facts;
    if (f.reference_clock_khz <= 0 || f.dram_gbps() <= 0) return std::nullopt;
    return card_from(device.compute_capability(), device.sms, f.reference_clock_khz * 1e-3,
                     f.dram_gbps());
}

std::optional<double> price_ns(const Card& card, const Call& call, std::string_view route,
                               const Settings& s) {
    const auto t = price(card, call, route, s);
    // A price the facts cannot give (a zero count in a row) is no price, never inf or NaN.
    return t && std::isfinite(*t) && *t > 0 ? t : std::nullopt;
}

std::vector<std::string> candidates(const Card& card, const Call& call, const Settings& s) {
    std::set<std::string> names;
    const double T = std::max<double>(call.tokens, 1.0);
    if (T >= 5 && mmq_takes(call)) names.insert("ggml_mmq");
    if (T <= 8) names.insert("ggml_vec");
    const std::string dec = rmma_decoder(call.qtype, call.k);
    // The register-MMA rows of the snapshot (key: decoder w_m w_k m n cluster epilogue latency
    // sums).
    for (const std::string& key : kf::rmma_row_keys(card.arch)) {
        std::istringstream in(key);
        std::string d;
        int wm = 0, wk = 0, m = 0, nn = 0, cl = 0, epi = 0, lat = 0, sums = 0;
        in >> d >> wm >> wk >> m >> nn >> cl >> epi >> lat >> sums;
        if (cl != 1 && card.cc < 90) continue;
        if (!dec.empty() && d == dec && lat == 0 && sums == 0) {
            std::string name = "rmma.m" + std::to_string(m) + ".n" + std::to_string(nn) + ".w" +
                               std::to_string(wm) + "x" + std::to_string(wk);
            if (cl != 1) name += ".c" + std::to_string(cl);
            names.insert(name);
        }
    }
    if (T <= 8) {
        const std::string codec = r8_codec(call.qtype);
        for (const auto& [key, f] : kf::section_rows(card.arch, "r8")) {
            (void)f;
            std::istringstream in(key);
            std::string cd;
            int w = 0, rows = 0;
            in >> cd >> w >> rows;
            if (!codec.empty() && cd == codec)
                names.insert("r8.w" + std::to_string(w) + "r" + std::to_string(rows));
        }
    }
    std::vector<std::string> out;
    for (const std::string& r : names)
        if (price_ns(card, call, r, s)) out.push_back(r);
    return out;
}

std::vector<detail::generic::Route> device_routes(const detail::generic::Problem& problem,
                                                  std::int32_t tokens) {
    if (tokens < 1 || tokens > kMaxTokens || route_policy() == RoutePolicy::Legacy) return {};
    if (!format_row(problem.qtype)) return {};
    // The analyzer prices A16 calls (the exam grids'); an A4 call (NVFP4 under --activations a4)
    // keeps its measured A4 bands and the rules.
    if (detail::generic::compute_class(problem.qtype, problem.policy) !=
        detail::generic::ComputeClass::A16)
        return {};
    const DeviceClass* dp = nullptr;
    try {
        dp = &device_class();
    } catch (const std::exception&) {
        return {}; // no device and no substitute (a CPU-only context): the older authorities answer
    }
    const DeviceClass& d = *dp;
    for (const std::string_view chip : kHeldChips)
        if (d.chip == chip) return {};
    // The architectures the switch scored (sm_75, sm_86, sm_89, sm_120a) on boards with tensor
    // cores. Volta keeps its HMMA authority: the analyzer prices its substitutes, but they are not
    // scored yet (the GV100's HMMA priced 11.8 us where it takes 20, the sealed exam 25.11).
    const int cc = d.compute_capability();
    if ((cc != 75 && cc != 86 && cc != 89 && cc != 120) || d.facts.mma_slow) return {};
    const auto kernels = kf::sass_arch_of(d, substituted_device_class() != nullptr);
    // Keyed by every number the price reads of the board (its SMs, clock and bus), not its name:
    // two classes of one chip and SM count on other clocks or memory price apart.
    std::string key =
        d.id() + '|' + std::to_string(d.sms) + '|' + std::to_string(d.facts.reference_clock_khz) +
        '|' + std::to_string(d.facts.dram_gbps()) + '|' + kernels.value_or("-") + '|' +
        std::to_string(static_cast<int>(problem.qtype)) + '|' +
        std::to_string(static_cast<int>(problem.epilogue)) + '|' + std::to_string(problem.n) + '|' +
        std::to_string(problem.k) + '|' + std::to_string(tokens);
    static std::mutex mu;
    static std::unordered_map<std::string, std::vector<detail::generic::Route>> memo;
    {
        std::lock_guard lock(mu);
        if (const auto it = memo.find(key); it != memo.end()) return it->second;
    }
    std::vector<detail::generic::Route> out;
    const auto card = card_of(d);
    if (card && kernels && *kernels == card->arch) {
        for (const Priced& p :
             rank(*card, Call{problem.qtype, problem.epilogue, problem.n, problem.k, tokens})) {
            if (const auto r = detail::generic::parse_route(p.route)) out.push_back(*r);
            if (p.route.rfind("rmma.", 0) == 0) break;
        }
    }
    std::lock_guard lock(mu);
    return memo.emplace(std::move(key), std::move(out)).first->second;
}

std::vector<Priced> rank(const Card& card, const Call& call, const Settings& s) {
    std::vector<Priced> v;
    for (const std::string& r : candidates(card, call, s))
        if (const auto t = price_ns(card, call, r, s)) v.push_back({r, *t});
    std::stable_sort(v.begin(), v.end(), [](const Priced& a, const Priced& b) {
        return a.ns < b.ns || (a.ns == b.ns && a.route < b.route);
    });
    return v;
}

} // namespace ninferno::ops::route_analyzer
