// The register-MMA kernel a call would run, described from the committed kernel facts with no CUDA
// (#620): rmma_resources, the price's read of a launch (generic_cost.cpp). It decides which kernel
// instance launch_rmma_schedule (generic_gemm_launch.cuh) would take (the add-sums, latency or wide
// variant, by the same rules and the same facts), reads that instance's registers and shared memory
// from the snapshot (kernel_facts.h), and gives the launch's geometry. The launch itself stays in
// the CUDA library, which describes the same by running in describe mode
// (rmma_resources_by_launch, generic_describe.cpp); ninferno_rmma_describe_test holds the two equal
// for every codec, schedule and epilogue on every golden board. Under the maintainer's live check
// (kernel_facts::live_check_target) rmma_resources describes through the launch itself, the only
// path that reads the running GPU's kernels.

#include "core/device_class.h"
#include "core/kernel_occupancy.h"
#include "core/per_device.h"
#include "core/tensor.h"
#include "ops/common/arch_limits.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_rmma_decoders.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/kernel_facts.h"

#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ninferno::ops::detail::generic {
namespace {

using Facts = std::optional<kernel_facts::KernelFacts>;

// Whether this build's MXFP4 register-MMA decoder is the byte-permute spread one (a build with
// NINFERNO_WITHOUT_NVFP4) rather than the FP4 converter's (Mxfp4RmmaDecoder, generic_gemm.cuh).
#ifdef NINFERNO_WITHOUT_NVFP4
constexpr bool kMxfp4Spread = true;
#else
constexpr bool kMxfp4Spread = false;
#endif

// The committed facts of the kernel instances of the kernels this thread's device runs, read once
// per device (a substituted board's, once per substitute's content: PerDevice) and keyed by
// integers (decoder, variant, schedule, epilogue): after the first read of an instance a priced
// call builds no string and searches no snapshot. The one table of a device holds its SASS label
// (kernel_facts::sass_arch, empty where the device runs PTX the driver compiled: no facts) and
// the instances read so far, each as the snapshot has it (nullopt: the snapshot lacks it).
struct FactsMemo {
    std::optional<std::string> arch;
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::optional<kernel_facts::KernelFacts>> facts; // by key
};

// The calling thread's table, or nullptr where there is no current device and no substitute (no
// kernels run: nothing is described).
FactsMemo* facts_memo() {
    static PerDevice<std::shared_ptr<FactsMemo>> memos;
    try {
        return memos
            .get([](int) {
                auto memo  = std::make_shared<FactsMemo>();
                memo->arch = kernel_facts::sass_arch();
                return memo;
            })
            .get();
    } catch (const std::exception&) { return nullptr; }
}

// The kernel instances of one decoder a launch chooses between (launch_rmma_schedule).
enum class Variant : int { Plain, Latency, AddSums, Wide };

// One call's decoder, schedule and epilogue, with the device's table.
struct Instance {
    FactsMemo* memo;
    WeightCodec codec;
    bool quad; // K a multiple of 128: the decoder of rmma_decoder_of for U4Zp's G32 and G64
    int schedule;
    const RmmaScheduleInfo& s;
    Epilogue epilogue;
};

// The committed facts of an instance (rmma_kernel_facts, generic_gemm_launch.cuh): nullopt where
// none are known. The decoder is a function of the codec and of K a multiple of 128
// (rmma_decoder_of), the wide one of the variant, so the integers key it; the schedule is below
// 4096 (rmma_schedule_count).
const std::optional<kernel_facts::KernelFacts>& facts_of(const Instance& in, const RmmaDecoder& dec,
                                                         Variant variant) {
    static const std::optional<kernel_facts::KernelFacts> none;
    if (in.memo == nullptr || !in.memo->arch) return none;
    const std::uint64_t key = (((static_cast<std::uint64_t>(in.codec) * 2 + (in.quad ? 1 : 0)) * 4 +
                                static_cast<std::uint64_t>(variant)) *
                                   4096 +
                               static_cast<std::uint64_t>(in.schedule)) *
                                  8 +
                              static_cast<std::uint64_t>(in.epilogue);
    const std::lock_guard lock(in.memo->mutex);
    auto it = in.memo->facts.find(key);
    if (it == in.memo->facts.end()) {
        const RmmaScheduleInfo& s = in.s;
        it                        = in.memo->facts
                 .emplace(key,
                          kernel_facts::rmma_facts(
                              *in.memo->arch,
                              {variant == Variant::Wide ? dec.wide : dec.name, s.warps_m, s.warps_k,
                               s.m_tiles, s.n_tiles, s.cluster, static_cast<int>(in.epilogue),
                               variant == Variant::Latency, variant == Variant::AddSums ? 1 : 0}))
                 .first;
    }
    return it->second; // node-based map: the reference outlives the lock
}

std::int64_t grid_ctas(const RmmaScheduleInfo& s, const GemmArgs& args) {
    const std::int64_t rows_per_cta   = static_cast<std::int64_t>(s.warps_m) * s.m_tiles * 16;
    const std::int64_t tokens_per_cta = static_cast<std::int64_t>(s.n_tiles) * 8;
    const std::int64_t x              = (args.rows + rows_per_cta - 1) / rows_per_cta;
    const std::int64_t y              = (args.tokens + tokens_per_cta - 1) / tokens_per_cta;
    return x * y * s.cluster;
}

// rmma_occupancy_kept (generic_gemm_launch.cuh): the variant's resident CTAs per SM against the
// plain kernel's, from their committed facts, and the call's waves at either.
bool occupancy_kept(const RmmaScheduleInfo& s, const Facts& variant, const Facts& plain,
                    const GemmArgs& args) {
    const auto smem     = static_cast<std::size_t>(args.spread_smem);
    const int threads   = s.warps_m * s.warps_k * 32;
    const auto& d       = device_class();
    const auto resident = [&](const Facts& f) {
        if (!f) return 0;
        return resident_ctas_per_sm(
            d, KernelShape{f->registers, f->static_shared_bytes, threads, smem});
    };
    return rmma_variant_occupancy_rule(grid_ctas(s, args), d.sms, resident(variant),
                                       resident(plain));
}

// kRmmaAheadForm (generic_gemm.cuh): the K-quant decoder of one token tile whose activations ride
// a ring a step ahead.
bool ahead_form(const RmmaDecoder& dec, const RmmaScheduleInfo& s, Epilogue epilogue) {
    return dec.stage_acts && dec.spread && s.n_tiles == 1 && !dec.finish && !dec.group_tail &&
           s.warps_m * s.warps_k <= 8 &&
           (epilogue != Epilogue::SwiGlu || s.warps_m * s.warps_k * 32 <= 128);
}

// kRmmaLatencyVariant: one token tile of a decoder that unpacks late, or SwiGLU of one that stages
// its activations.
bool has_latency_variant(const RmmaDecoder& dec, const RmmaScheduleInfo& s, Epilogue epilogue) {
    return s.n_tiles == 1 && (dec.unpack || (dec.stage_acts && epilogue == Epilogue::SwiGlu));
}

// rmma_latency_board: where the board is latency-bound for the epilogue, has the MMA the variant
// needs, and (without cp.async) the activation ring's form.
bool latency_board(const RmmaDecoder& dec, const RmmaScheduleInfo& s, Epilogue epilogue) {
    const bool swiglu = epilogue == Epilogue::SwiGlu;
    if (dram_bytes_per_sm_cycle() < (swiglu ? kRmmaLatencySwiGluBytes : kRmmaLatencyOneSetBytes)) {
        return false;
    }
    const auto& d = device_class();
    // Volta's register MMA is emulated on FMA (no m16n8k16): its kernel is compute-bound, and the
    // variant only cost there (GV100, T 5..8: 0-5 % slower, never picked).
    if (d.major * 10 + d.minor < 75) return false;
    // Without cp.async (sm_75) SwiGLU's variant is the activation ring's form, or none.
    return !(swiglu && d.major < 8 && !ahead_form(dec, s, epilogue));
}

bool latency_pays(const Instance& in, const RmmaDecoder& dec, const GemmArgs& args) {
    return latency_board(dec, in.s, in.epilogue) &&
           occupancy_kept(in.s, facts_of(in, dec, Variant::Latency),
                          facts_of(in, dec, Variant::Plain), args);
}

// rmma_wide_pays: the wide-load kernel's spills and waves against the plain kernel's; a kernel
// without facts never pays.
bool wide_pays(const Instance& in, const RmmaDecoder& dec, const GemmArgs& args) {
    const Facts& plain = facts_of(in, dec, Variant::Plain);
    const Facts& wide  = facts_of(in, dec, Variant::Wide);
    if (!plain || !wide) return false;
    const auto& d = device_class();
    // The board's facts first: the occupancy only where they admit the wide kernel.
    if (!rmma_wide_rule(dram_bytes_per_sm_cycle(), d.major * 10 + d.minor, true,
                        wide->spill_bytes(), plain->spill_bytes())) {
        return false;
    }
    return occupancy_kept(in.s, wide, plain, args);
}

} // namespace

std::optional<KernelResources> rmma_resources(int schedule, const Weight& w, Epilogue epilogue,
                                              std::int32_t tokens, std::int32_t rows) {
    // The live check reads the running GPU's kernels, which only the launch reaches: under it the
    // description is the launch's own (the facts it checks are the ones the launch read).
    if (kernel_facts::live_check_target() != nullptr) {
        return rmma_resources_by_launch(schedule, w, epilogue, tokens, rows);
    }
    if (schedule < 0 || schedule >= rmma_schedule_count() || !rmma_schedule_launchable(schedule) ||
        tokens < 1 || !supports(w)) {
        return std::nullopt;
    }
    const WeightCodec codec = codec_of(w);
    const auto s            = rmma_schedule_info(schedule);
    if (!rmma_tile_compiled(codec, s.m_tiles, s.n_tiles, w.k) || (w.k % k_multiple(codec)) != 0) {
        return std::nullopt;
    }
    GemmArgs args    = weight_args(w);
    args.rows        = rows > 0 ? rows : (epilogue == Epilogue::SwiGlu ? w.n / 2 : w.n);
    args.tokens      = tokens;
    args.ldx         = w.k;
    args.lane_local  = invariant_routes_active() ? 0 : 1;
    args.add_sums    = rmma_add_sums() ? 1 : 0;
    args.spread_smem = spread_smem(Route{Family::RegisterMma, static_cast<std::uint8_t>(schedule)},
                                   codec, epilogue, args);
    // Without tensor mma.sync run_route hands such calls to Volta's own HMMA and GEMV kernels
    // (the call's x aligned as the engine passes it): no register-MMA kernel runs to describe.
    if (!current_features().mma_tensor &&
        (volta_hmma_takes(codec, args) || volta_gemv_takes(codec, args))) {
        return std::nullopt;
    }
    const RmmaDecoder dec = rmma_decoder_of(codec, args.k, kMxfp4Spread);
    if (dec.name == nullptr) return std::nullopt; // no register-MMA launch for the codec
    const Instance instance{facts_memo(), codec, args.k % 128 == 0, schedule, s, epilogue};

    // launch_rmma_schedule's choice of the kernel instance: the call's add-sums form, else the
    // latency variant where it pays, else the wide-load one, else the plain kernel. A failed read
    // of the device's facts takes the plain kernel, as the launch does.
    Variant variant = Variant::Plain;
    if (dec.min_term && s.n_tiles >= 4 && args.add_sums != 0) {
        variant = Variant::AddSums;
    } else {
        bool pays = false;
        if (has_latency_variant(dec, s, epilogue)) {
            try {
                pays = latency_pays(instance, dec, args);
            } catch (const std::exception&) {}
            if (pays) variant = Variant::Latency;
        }
        if (!pays && s.n_tiles == 1 && dec.wide[0] != '\0') {
            try {
                pays = wide_pays(instance, dec, args);
            } catch (const std::exception&) {}
            if (pays) variant = Variant::Wide;
        }
    }

    // launch_rmma_kernel in describe mode: the kernel's committed facts and the launch's geometry.
    const std::int64_t rows_per_cta   = static_cast<std::int64_t>(s.warps_m) * s.m_tiles * 16;
    const std::int64_t tokens_per_cta = static_cast<std::int64_t>(s.n_tiles) * 8;
    const std::int64_t grid_x         = (args.rows + rows_per_cta - 1) / rows_per_cta;
    const std::int64_t grid_y         = (args.tokens + tokens_per_cta - 1) / tokens_per_cta;
    if (grid_y > kMaxGridYZ) {
        throw std::invalid_argument("generic linear: token extent exceeds the grid");
    }
    const Facts& facts = facts_of(instance, dec, variant);
    if (!facts) return std::nullopt; // nothing described: the caller prices by threads alone
    KernelResources out;
    describe_from_facts(out, *facts, static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                        static_cast<unsigned>(s.cluster), s.warps_m * s.warps_k * 32,
                        static_cast<std::size_t>(args.spread_smem), variant == Variant::Latency);
    // The price's instance: the MMA with ones or the adds (RmmaSums 0 / 1), as the launch's own
    // description says; never the group-sums plane's (rmma_plane_taken, the device's launch).
    out.sums = variant == Variant::AddSums ? 1 : 0;
    return out;
}

} // namespace ninferno::ops::detail::generic
