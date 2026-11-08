// The generic path's kernels described instead of run (#298): the registers, shared memory and
// geometry of the kernel a route would launch for a call (rmma_resources, r8_resources,
// route_resources; generic_gemm.h).

#include "core/device_class.h"
#include "core/tensor.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include <algorithm>
#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::generic {

namespace {
thread_local KernelResources* t_describe = nullptr;
} // namespace

KernelResources* kernel_describe_target() noexcept { return t_describe; }
ScopedKernelDescribe::ScopedKernelDescribe(KernelResources& out) noexcept : previous_(t_describe) {
    t_describe = &out;
}
ScopedKernelDescribe::~ScopedKernelDescribe() { t_describe = previous_; }

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
    GemmArgs args = weight_args(w);
    args.rows       = rows > 0 ? rows : (epilogue == Epilogue::SwiGlu ? w.n / 2 : w.n);
    args.tokens   = tokens;
    args.ldx      = w.k;
    args.lane_local = invariant_routes_active() ? 0 : 1;
    args.add_sums   = rmma_add_sums() ? 1 : 0;
    args.spread_smem =
        spread_smem(Route{Family::RegisterMma, static_cast<std::uint8_t>(schedule)}, codec, epilogue, args);
    // Without tensor mma.sync run_route hands such calls to Volta's own HMMA and GEMV kernels
    // (the call's x aligned as the engine passes it): no register-MMA kernel runs to describe.
    if (!current_features().mma_tensor &&
        (volta_hmma_takes(codec, args) || ((args.ldx % 8) == 0 && volta_gemv_takes(codec, args)))) {
        return std::nullopt;
    }
    KernelResources out;
    const ScopedKernelDescribe describe(out);
    launch_rmma_codec(codec, schedule, epilogue, args, nullptr);
    if (out.threads == 0) return std::nullopt; // nothing described: no register-MMA launch ran
    return out;
}

std::optional<R8Resources> r8_resources(int schedule, const Weight& w, Epilogue epilogue,
                                        std::int32_t tokens) {
    const Route route{Family::R8Mma, static_cast<std::uint8_t>(schedule)};
    if (schedule < 0 || schedule >= r8_schedule_count() || tokens < 1 ||
        tokens > kR8VerifyTokens || !supports(w) || !route_admits(route, w, LinearPolicy::AllowA8)) {
        return std::nullopt;
    }
    const std::int32_t rows = epilogue == Epilogue::SwiGlu ? w.n / 2 : w.n;
    // The call as the engine passes it, with the workspace its pre-quantized form needs.
    const ResolvedRoute resolved =
        resolve_route(route, w,
                      CallLayout{.x_aligned16    = true,
                                 .ldx            = w.k,
                                 .tokens         = tokens,
                                 .rows           = rows,
                                 .segments       = 1,
                                 .epilogue       = epilogue,
                                 .workspace_free = route_workspace_bytes(route, w.k, tokens)});
    if (!(resolved.executed == route)) return std::nullopt;
    GemmArgs args = weight_args(w);
    args.rows     = rows;
    args.tokens   = tokens;
    args.ldx      = w.k;
    // Never read: the described launch takes only their presence (the pre-quantized form).
    static const std::int8_t codes = 0;
    static const float planes      = 0.0F;
    const W4a8Activation pre =
        resolved.r8_prequantized ? W4a8Activation{&codes, &planes, &planes} : W4a8Activation{};
    R8Resources out;
    out.prequantized = resolved.r8_prequantized;
    {
        const ScopedKernelDescribe describe(out.kernel);
        launch_r8(schedule, codec_of(w), epilogue, args, nullptr, pre);
    }
    if (out.kernel.threads == 0) return std::nullopt;
    return out;
}

namespace {

// ggml's matrix or vector kernel for the call, as run_route launches it: the matrix kernel on the
// first kMmqPrefillSlice tokens (a wider call repeats it), the vector kernel on the first
// kVecMaxColumns (likewise); the gate's product of a SwiGLU pair (the up's is the same kernel).
// The activation quantizers and the stream-K fixup are not described (ggml_mmq::ScopedDescribe).
std::optional<KernelResources> ggml_resources(Route route, const Weight& w, Epilogue epilogue,
                                              const GemmArgs& args) {
    const auto type = mmq_type(w);
    if (!type) return std::nullopt;
    const int bpr   = w.k / static_cast<int>(ggml_block_values(w.qtype));
    const auto rows = ggml_rows_at(args, w.qtype, bpr, 0);
    // Never read: a described launch takes only the pointers' presence.
    void* const stand_in = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    ggml_mmq::LaunchShape shape;
    {
        const ggml_mmq::ScopedDescribe describe(shape);
        if (route.family == Family::GgmlMmq) {
            const int slice = std::min(args.tokens, kMmqPrefillSlice);
            ggml_mmq::product(*type, rows, bpr, args.rows, w.k, stand_in, slice,
                              static_cast<float*>(stand_in), args.rows, stand_in, nullptr);
        } else {
            ggml_mmq::VecOut out;
            out.epilogue = epilogue == Epilogue::SwiGlu ? ggml_mmq::VecOut::Epilogue::SwiGlu
                           : epilogue == Epilogue::AddResidual
                               ? ggml_mmq::VecOut::Epilogue::AddResidual
                               : ggml_mmq::VecOut::Epilogue::Store;
            out.out[0]         = stand_in;
            out.segment_end[0] = args.rows;
            out.ld[0]          = args.rows;
            const auto up      = epilogue == Epilogue::SwiGlu
                                     ? ggml_rows_at(args, w.qtype, bpr, args.rows)
                                     : ggml_mmq::Planes{};
            ggml_mmq::vec_product(*type, rows, up, bpr, args.rows, w.k, stand_in,
                                  std::min(args.tokens, ggml_mmq::kVecMaxColumns), out, nullptr);
        }
    }
    if (shape.threads == 0) return std::nullopt;
    KernelResources out;
    out.registers             = shape.registers;
    out.local_bytes           = shape.local_bytes;
    out.static_smem           = shape.static_smem;
    out.dynamic_smem          = shape.dynamic_smem;
    out.threads               = shape.threads;
    out.max_threads_per_block = shape.max_threads_per_block;
    out.ptx_version           = shape.ptx_version;
    out.binary_version        = shape.binary_version;
    out.grid_x                = shape.grid_x;
    out.grid_y                = shape.grid_y;
    out.grid_z                = shape.grid_z;
    return out;
}

} // namespace

std::optional<KernelResources> route_resources(Route route, const Weight& w, Epilogue epilogue,
                                               std::int32_t tokens) {
    switch (route.family) {
    case Family::RegisterMma: return rmma_resources(route.schedule, w, epilogue, tokens);
    case Family::R8Mma: {
        const auto r8 = r8_resources(route.schedule, w, epilogue, tokens);
        return r8 ? std::optional<KernelResources>(r8->kernel) : std::nullopt;
    }
    case Family::TiledMma:
    case Family::StagedMma:
    case Family::W4a8Mma:
    case Family::I8Mma:
    case Family::W4a4Mma:
    case Family::Nvfp4TiledMma:
    case Family::GgmlMmq:
    case Family::GgmlVec: break;
    default: return std::nullopt;
    }
    // W4A4 computes A4 activations: the policy that admits it.
    const LinearPolicy policy =
        route.family == Family::W4a4Mma ? LinearPolicy::AllowA4 : LinearPolicy::AllowA8;
    if (tokens < 1 || !supports(w) || !route_admits(route, w, policy)) {
        return std::nullopt;
    }
    const std::int32_t rows = epilogue == Epilogue::SwiGlu ? w.n / 2 : w.n;
    // ggml's matrix kernel holds its planes (both of a SwiGLU pair's) for one prefill slice.
    const std::size_t workspace =
        route.family == Family::GgmlMmq
            ? mmq_workspace_bytes(w.k, epilogue == Epilogue::SwiGlu ? 2 * rows : rows,
                                  std::min(tokens, kMmqPrefillSlice))
            : route_workspace_bytes(route, w.k, tokens);
    const ResolvedRoute resolved =
        resolve_route(route, w,
                      CallLayout{.x_aligned16    = true,
                                 .ldx            = w.k,
                                 .tokens         = tokens,
                                 .rows           = rows,
                                 .segments       = 1,
                                 .epilogue       = epilogue,
                                 .workspace_free = workspace});
    if (!(resolved.executed == route)) return std::nullopt;
    const WeightCodec codec = codec_of(w);
    if (route.family == Family::TiledMma &&
        (w.k % 64 != 0 || volta_cutlass_only(codec) ||
         volta_cutlass_runs(volta_cutlass_only(codec), tokens))) {
        // Scalar, or Volta's CUTLASS copy. Where the copy does not fit the call's workspace (or
        // there is none) run_route falls back to the tile kernel: described here as none, never
        // as another kernel.
        return std::nullopt;
    }
    GemmArgs args = weight_args(w);
    args.rows     = rows;
    args.tokens   = tokens;
    args.ldx      = w.k;
    if (route.family == Family::GgmlMmq || route.family == Family::GgmlVec) {
        return ggml_resources(route, w, epilogue, args);
    }
    KernelResources out;
    {
        const ScopedKernelDescribe describe(out);
        switch (route.family) {
        case Family::TiledMma: launch_tiled(route.schedule, codec, epilogue, args, nullptr); break;
        case Family::StagedMma: launch_staged(route.schedule, epilogue, args, nullptr); break;
        case Family::W4a8Mma:
            launch_w4a8(route.schedule, codec, epilogue, args, nullptr, nullptr, nullptr,
                        kW4a8TokenMajor, nullptr);
            break;
        case Family::I8Mma:
            launch_i8(route.schedule, i8_bits(w.qtype), epilogue, args, nullptr, nullptr, nullptr,
                      nullptr);
            break;
#ifndef NINFERNO_WITHOUT_NVFP4
        case Family::W4a4Mma:
            launch_w4a4(route.schedule, epilogue, args, w.input_scale_divisor, nullptr, nullptr,
                        nullptr);
            break;
        case Family::Nvfp4TiledMma:
            launch_nvfp4_tiled(nvfp4_tiled_schedule_info(route.schedule).classes, epilogue, args,
                               nullptr);
            break;
#endif
        default: break;
        }
    }
    if (out.threads == 0) return std::nullopt;
    return out;
}

} // namespace ninferno::ops::detail::generic
