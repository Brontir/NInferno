// The register-MMA kernel a call would run, described from the committed kernel facts with no CUDA
// (#620): rmma_resources, the price's read of a launch (generic_cost.cpp). It decides which kernel
// instance launch_rmma_schedule (generic_gemm_launch.cuh) would take (the add-sums, latency or wide
// variant, by the same rules and the same facts), reads that instance's registers and shared memory
// from the snapshot (kernel_facts.h), and gives the launch's geometry. The launch itself stays in
// the CUDA library, which describes the same by running in describe mode
// (rmma_resources_by_launch, generic_describe.cpp); ninferno_rmma_describe_test holds the two equal
// for every codec, schedule and epilogue on every golden board.

#include "core/device_class.h"
#include "core/kernel_occupancy.h"
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
#include <optional>
#include <stdexcept>
#include <string>

namespace ninferno::ops::detail::generic {
namespace {

using Facts = std::optional<kernel_facts::KernelFacts>;

// The committed facts of a kernel instance for the kernels this thread's device runs
// (rmma_kernel_facts, generic_gemm_launch.cuh): nullopt where none are known.
Facts facts_of(const RmmaScheduleInfo& s, const char* decoder, Epilogue epilogue, bool latency,
               bool add_sums) {
    const auto arch = kernel_facts::sass_arch();
    if (!arch) return std::nullopt;
    return kernel_facts::rmma_facts(*arch,
                                    {decoder, s.warps_m, s.warps_k, s.m_tiles, s.n_tiles, s.cluster,
                                     static_cast<int>(epilogue), latency, add_sums});
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

bool latency_pays(const RmmaDecoder& dec, const RmmaScheduleInfo& s, Epilogue epilogue,
                  const GemmArgs& args) {
    return latency_board(dec, s, epilogue) &&
           occupancy_kept(s, facts_of(s, dec.name, epilogue, true, false),
                          facts_of(s, dec.name, epilogue, false, false), args);
}

// rmma_wide_pays: the wide-load kernel's spills and waves against the plain kernel's; a kernel
// without facts never pays.
bool wide_pays(const RmmaDecoder& dec, const RmmaScheduleInfo& s, Epilogue epilogue,
               const GemmArgs& args) {
    const Facts plain = facts_of(s, dec.name, epilogue, false, false);
    const Facts wide  = facts_of(s, dec.wide, epilogue, false, false);
    if (!plain || !wide) return false;
    const auto& d = device_class();
    // The board's facts first: the occupancy only where they admit the wide kernel.
    if (!rmma_wide_rule(dram_bytes_per_sm_cycle(), d.major * 10 + d.minor, true,
                        wide->spill_bytes(), plain->spill_bytes())) {
        return false;
    }
    return occupancy_kept(s, wide, plain, args);
}

} // namespace

std::optional<KernelResources> rmma_resources(int schedule, const Weight& w, Epilogue epilogue,
                                              std::int32_t tokens, std::int32_t rows) {
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
        (volta_hmma_takes(codec, args) || ((args.ldx % 8) == 0 && volta_gemv_takes(codec, args)))) {
        return std::nullopt;
    }
    const RmmaDecoder dec = rmma_decoder_of(codec, args.k);
    if (dec.name == nullptr) return std::nullopt; // no register-MMA launch for the codec

    // launch_rmma_schedule's choice of the kernel instance: the call's add-sums form, else the
    // latency variant where it pays, else the wide-load one, else the plain kernel. A failed read
    // of the device's facts takes the plain kernel, as the launch does.
    bool latency        = false;
    bool add_sums       = false;
    const char* decoder = dec.name;
    if (dec.min_term && s.n_tiles >= 4 && args.add_sums != 0) {
        add_sums = true;
    } else {
        bool pays = false;
        if (has_latency_variant(dec, s, epilogue)) {
            try {
                pays = latency_pays(dec, s, epilogue, args);
            } catch (const std::exception&) {}
            latency = pays;
        }
        if (!pays && s.n_tiles == 1 && dec.wide[0] != '\0') {
            try {
                pays = wide_pays(dec, s, epilogue, args);
            } catch (const std::exception&) {}
            if (pays) decoder = dec.wide;
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
    const Facts facts = facts_of(s, decoder, epilogue, latency, add_sums);
    if (!facts) return std::nullopt; // nothing described: the caller prices by threads alone
    KernelResources out;
    describe_from_facts(out, *facts, static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                        static_cast<unsigned>(s.cluster), s.warps_m * s.warps_k * 32,
                        static_cast<std::size_t>(args.spread_smem), latency);
    return out;
}

} // namespace ninferno::ops::detail::generic
