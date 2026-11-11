#include "ops/generic/generic_gemm.h"
#include "ops/generic/format_traits.h"
#include "core/per_device.h"
#include "ops/common/native_nvfp4.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/tensor.h"
#include "core/layout.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/route_table.h"
#include "ops/route_pins.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <tuple>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::ops::detail::generic {

namespace {

GemmArgs make_args(const Call& call) {
    GemmArgs a = weight_args(*call.weight);
    a.x         = call.x->data;
    a.ldx       = call.x->ne[0];
    a.tokens    = call.x->ne[1];
    a.row_begin = call.row_begin;
    a.rows      = call.rows;
    a.up_offset = call.up_offset;
    a.gelu_gate = call.gelu_gate ? 1 : 0;
    a.relu2     = call.relu2 ? 1 : 0;
    a.segments  = call.segments;
    int end     = 0;
    for (int s = 0; s < call.segments; ++s) {
        end += call.segment_rows[s];
        a.out[s]         = static_cast<std::uint16_t*>(call.out[s]->data) + call.out_row_offset[s];
        a.ld[s]          = call.out[s]->ne[0];
        a.segment_end[s] = end;
    }
    for (int s = call.segments; s < kMaxSegments; ++s) a.segment_end[s] = end;
    return a;
}

void validate_call(const Call& call) {
    if (call.x == nullptr || call.weight == nullptr) {
        throw std::invalid_argument("generic linear: missing input or weight");
    }
    const Weight& w = *call.weight;
    refuse_frag(w, "generic linear");
    const Tensor& x = *call.x;
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != w.k || x.ne[1] <= 0 ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("generic linear: x must be contiguous BF16 [K,T]");
    }
    if (call.segments < 1 || call.segments > kMaxSegments) {
        throw std::invalid_argument("generic linear: invalid output segment count");
    }
    int total = 0;
    for (int s = 0; s < call.segments; ++s) {
        const Tensor* out = call.out[s];
        if (out == nullptr || out->dtype != DType::BF16 || !out->is_contiguous() ||
            call.out_row_offset[s] < 0 ||
            out->ne[0] < call.out_row_offset[s] + call.segment_rows[s] ||
            static_cast<std::int64_t>(out->ne[1]) * out->ne[2] * out->ne[3] != x.ne[1]) {
            throw std::invalid_argument("generic linear: output segment must be BF16 [rows,T]");
        }
        total += call.segment_rows[s];
    }
    if (total != call.rows || call.rows <= 0 || call.row_begin < 0) {
        throw std::invalid_argument("generic linear: output segments do not cover the rows");
    }
    const int sets = call.epilogue == Epilogue::SwiGlu ? 2 : 1;
    const std::int64_t last =
        static_cast<std::int64_t>(call.row_begin) + call.rows + (sets == 2 ? call.up_offset : 0);
    if (last > w.n || (sets == 2 && call.up_offset < call.rows)) {
        throw std::invalid_argument("generic linear: rows exceed the weight");
    }
}

// The bytes free in the rest of the arena from its next 256-byte address (nullopt: no arena; 0
// where that address is already past its end: no route's planes are empty), against
// which mmq_workspace_bytes' layout is held (resolve_route). Its blocks are 256-aligned, so from
// there they land exactly as the layout counted them: an arena the planner sized to the route
// holds it with nothing to spare.
std::optional<std::size_t> workspace_free(const DeviceArena* workspace) {
    if (workspace == nullptr) return std::nullopt;
    const auto base = reinterpret_cast<std::uintptr_t>(workspace->base());
    const auto start =
        ((base + workspace->used() + 255U) & ~static_cast<std::uintptr_t>(255U)) - base;
    return start <= workspace->capacity() ? workspace->capacity() - start : 0;
}

std::atomic<std::uint64_t> mmq_fallbacks{0};

std::atomic<std::uint64_t> route_demotions{0};

std::atomic<std::uint64_t> joined_launches{0};

// A demotion that keeps the bytes: the staged MMA and the NVFP4 tiled route hand a call they
// cannot read to the register MMA of the same summation order.
bool same_bytes(Route asked, Route ran) {
    return ran.family == Family::RegisterMma &&
           (asked.family == Family::StagedMma || asked.family == Family::Nvfp4TiledMma);
}

// A call the asked route could not take ran another route, with other bytes and speed than a
// route table or the heuristic chose: said once per pair of routes, with the call's facts, and
// counted (route_demotion_count). No product run demoted in the lab (perplexity, generation and
// serving at 1 and 4 streams on Ada, Blackwell and Turing); one names a caller that sized or
// aligned its call short of its route.
void note_demotion(Route asked, Route ran, int rows, int tokens, int k, bool x16, std::int64_t ldx,
                   std::size_t workspace) {
    route_demotions.fetch_add(1, std::memory_order_relaxed);
    static std::mutex mutex;
    static std::set<std::pair<std::string, std::string>> said;
    const std::lock_guard lock(mutex);
    if (!said.emplace(route_name(asked), route_name(ran)).second) { return; }
    std::fprintf(stderr,
                 "ninferno: warning: route %s cannot take a call of %d rows x %d tokens (k %d, x "
                 "%s16-byte aligned, ldx %lld, workspace %zu bytes): it ran %s, with other bytes "
                 "and speed\n",
                 route_name(asked).c_str(), rows, tokens, k, x16 ? "" : "not ",
                 static_cast<long long>(ldx), workspace, route_name(ran).c_str());
}

// Every distinct problem that reached the generic path on a thread collecting them (seen_problems:
// an Engine's load, on its own thread, so another live Engine's calls stay out), and under
// NINFERNO_ROUTE_RECORD=<file> each distinct (format, compute, epilogue, N, K) appended to the file
// in ninferno_route_tune's --shapes format, so a model's shapes come from running it.
using SeenSet = std::set<std::tuple<int, int, int, std::int32_t, std::int32_t, std::int32_t, int>>;

struct ThreadSeen {
    bool collecting = false;
    SeenSet seen;
};

ThreadSeen& thread_seen() {
    thread_local ThreadSeen state;
    return state;
}

struct RecordedProblems {
    std::mutex mutex;
    std::set<std::tuple<int, int, int, int, int>> recorded;
};

RecordedProblems& recorded_state() {
    static RecordedProblems state;
    return state;
}

void record_problem(const Problem& p, const Weight& w) {
    static const char* path = std::getenv("NINFERNO_ROUTE_RECORD");
    if (ThreadSeen& t = thread_seen(); t.collecting) {
        t.seen.insert({static_cast<int>(p.qtype), static_cast<int>(p.policy),
                       static_cast<int>(p.epilogue), p.n, p.k, w.n, static_cast<int>(w.layout)});
    }
    if (path == nullptr || *path == 0) return;
    RecordedProblems& s = recorded_state();
    std::lock_guard lock(s.mutex);
    const auto compute = compute_class(p.qtype, p.policy);
    if (!s.recorded.insert({static_cast<int>(p.qtype), static_cast<int>(compute),
                            static_cast<int>(p.epilogue), p.n, p.k})
             .second) {
        return;
    }
    std::ofstream(path, std::ios::app) << qtype_name(p.qtype) << ' ' << compute_class_name(compute)
                                       << ' ' << epilogue_name(p.epilogue) << ' ' << p.n << ' '
                                       << p.k << '\n';
}

struct RecordedProjections {
    std::mutex mutex;
    std::set<std::tuple<std::string, std::string, std::int32_t, std::int32_t, std::int32_t>> seen;
};

RecordedProjections& recorded_projections() {
    static RecordedProjections state;
    return state;
}


} // namespace

void record_projection_route(const char* op, const char* route, std::int32_t tokens, std::int32_t n,
                             std::int32_t k) {
    static const char* path = std::getenv("NINFERNO_PROJECTION_RECORD");
    if (path == nullptr || *path == 0) return;
    RecordedProjections& s = recorded_projections();
    std::lock_guard lock(s.mutex);
    if (!s.seen.insert({op, route, tokens, n, k}).second) return;
    std::ofstream(path, std::ios::app)
        << op << ' ' << route << ' ' << tokens << ' ' << n << ' ' << k << '\n';
}

ggml_mmq::Planes ggml_rows_at(const GemmArgs& args, QType qtype, int bpr, int row) {
    const auto bytes  = mmq_plane_bytes(qtype);
    const auto blocks = static_cast<std::int64_t>(row) * bpr;
    ggml_mmq::Planes p;
    p.codes = args.codes + blocks * bytes.codes;
    if (bytes.high > 0) p.high = args.high + blocks * bytes.high;
    if (bytes.scales > 0) p.scales = args.scales + blocks * bytes.scales;
    return p;
}

void clear_seen_problems() {
    ThreadSeen& t = thread_seen();
    t.collecting  = true;
    t.seen.clear();
}

std::vector<SeenProblem> seen_problems() {
    const ThreadSeen& t = thread_seen();
    std::vector<SeenProblem> out;
    out.reserve(t.seen.size());
    for (const auto& [qtype, policy, epilogue, n, k, parent_rows, layout] : t.seen) {
        out.push_back({Problem{static_cast<QType>(qtype), static_cast<LinearPolicy>(policy), n, k,
                               static_cast<Epilogue>(epilogue)},
                       parent_rows, static_cast<QuantLayout>(layout)});
    }
    return out;
}

void refuse_frag(const Weight& w, const char* op) {
    if (w.layout != QuantLayout::GgmlFrag) return;
    throw std::invalid_argument(
        std::string(op) +
        ": the weight is a ggml_frag_v1 routed-expert bank, which only the grouped MoE kernel "
        "reads; a projection needs its published layout (ninferno-convert --relayout published)");
}

namespace {

// A registered shape's own kernels of these formats compute BF16 activations only (the int8
// tiers and R8 are the generic path's): a call such a gate keeps on its table is A16. The NVFP4
// and FP8 shapes choose their quantized kernels themselves (their dispatch notes them).
bool owned_tables_a16(QType qtype) {
    return qtype != QType::NVFP4 && qtype != QType::FP8_E4M3FN_ROW_BF16;
}


} // namespace

// These two ask about the whole weight: the call reads all its rows.
bool r8_takes_block(const Weight& weight) {
    const Problem whole{weight.qtype, LinearPolicy::AllowA8, weight.n, weight.k, Epilogue::Store};
    return supports(weight) && r8_call(whole, weight, kR8VerifyTokens);
}

bool owned_decode_r8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    const Problem whole{weight.qtype, policy, weight.n, weight.k, Epilogue::Store};
    const bool r8 = tokens <= kR8DecodeTokens && supports(weight) && r8_call(whole, weight, tokens);
    // --decode-route: R8 keeps the rule (a registered shape hands R8 only the decode calls its
    // workspace query covers; the generic path notes the call); any other pin keeps the call on
    // the shape's own kernel, which reads BF16 activations.
    const DecodeRoute pin = route_pins().decode;
    if (pin == DecodeRoute::Auto || pin == DecodeRoute::R8 || tokens < 1 ||
        tokens > kDecodePinMaxTokens) {
        return r8;
    }
    if (owned_tables_a16(weight.qtype)) {
        note_decode_pin(pin == DecodeRoute::A16,
                        family_bit(DecodeRoute::A16) | (r8 ? family_bit(DecodeRoute::R8) : 0U));
    }
    return false;
}

bool owned_prefill_w4a8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    // Without tensor mma.sync the tiled route owns such a call, as in select_route.
    const bool w4a8 = tokens >= kQ8PrefillA8Tokens && prefill_routes_active() &&
                      weight.qtype == QType::NVFP4 && supports(weight) &&
                      allows_w4a8(weight, policy) && !volta_tiled_route(weight, policy, tokens);
    // --prefill-route a16: the shape's own plan keeps the call on its A16 route.
    if (w4a8 && route_pins().prefill == PrefillRoute::A16) {
        note_prefill_pin(true, family_bit(PrefillRoute::A16) | family_bit(PrefillRoute::W4a8));
        return false;
    }
    return w4a8;
}

namespace {

// Q4/Q5/Q6 G64 weights take int8 activations at prefill wherever the route admits W4A8 (the
// artifact's A16Only uses included) on the cards where they pay (qx_prefill_a8_pays);
// --activations exact keeps A16 (invariant routes). Qwen3.8-27B book 8K KLD 0.0047, top-1 97.1 %.
// Int8 activations pay where the card's int8 mma runs well ahead of its bf16 mma with fp32
// accumulation: the GeForce and Ada/Ampere inference parts (half-rate fp32-accumulate HMMA, a
// quarter of int8). Qwen3.8-27B Q4/Q5 G64 24K prefill, MTP 3: RTX 4090
// +18.7 %, L4 +14 %, A10 +7.8 %. The HBM training parts run fp32-accumulate HMMA at full rate,
// and the activation quantization pass then costs more than the faster mma wins: A100 -3.5 %.
bool qx_prefill_a8_pays() {
    try {
        const auto& d = device_class();
        // Not on a chip whose mma.sync is emulated (mma_slow: TU116 / TU117), as for ggml's mmq.
        // A generation without a row of its own has no measured kind: A16 (#300).
        const auto full_rate = fp32acc_full_rate_of(d.major, d.minor);
        return d.features.mma_tensor && !d.features.mma_slow && full_rate && !*full_rate;
    } catch (const std::exception&) { return false; }
}

bool qx_format(QType qtype) { return format_traits(qtype).qx_g64; }

// Whether the Q4/Q5/Q6 G64 weights take the int8 tiers at prefill: where int8 activations pay
// (qx_prefill_a8_pays, `pays`), or as a route pin asks (--prefill-route w4a8 or i8: on every card;
// a16: on none).
bool qx_prefill_a8_takes(bool pays) {
    switch (route_pins().prefill) {
    case PrefillRoute::W4a8:
    case PrefillRoute::I8:
        return true;
    case PrefillRoute::A16:
        return false;
    default:
        return pays;
    }
}


} // namespace

LinearPolicy prefill_a8_policy(QType qtype, LinearPolicy policy) {
    if (policy == LinearPolicy::A16Only && qx_format(qtype) &&
        qx_prefill_a8_takes(qx_prefill_a8_pays()) && !invariant_routes_active()) {
        return LinearPolicy::AllowA8;
    }
    return policy;
}

std::size_t prefill_a8_workspace_bytes(const Problem& problem, std::int32_t min_tokens,
                                       std::int32_t max_tokens) {
    // NVFP4 under a policy with A8 and without W4A4 (owned_prefill_w4a8): the W4A8 planes of its
    // calls of 17 tokens or more, which the shape's own plan does not run.
    if (nvfp4_prefill_a8_policy(problem.qtype, problem.policy)) {
        if (max_tokens < kPrefillBandMinTokens) return 0;
        return workspace_capacity_bytes(problem, nullptr,
                                        std::max(min_tokens, kPrefillBandMinTokens), max_tokens);
    }
    const LinearPolicy a8_policy = prefill_a8_policy(problem.qtype, problem.policy);
    if (!prefill_a8_format(problem.qtype) || !allows_a8(a8_policy) ||
        max_tokens < kQ8PrefillA8Tokens) {
        return 0;
    }
    Problem a8 = problem;
    a8.policy  = a8_policy;
    return workspace_capacity_bytes(a8, nullptr, min_tokens, max_tokens);
}

bool prefill_a8_format(QType qtype) {
    static PerDevice<bool> pays;
    const bool qx = qx_prefill_a8_takes(pays.get([](int) { return qx_prefill_a8_pays(); }));
    switch (qtype) {
    case QType::Q8_G32_FP16:
        // --prefill-route a16 keeps Q8 on its shape tables' A16 kernels too.
        return route_pins().prefill != PrefillRoute::A16;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        return qx;
    default:
        return false;
    }
}

bool q8_prefill_a8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    const bool band = tokens >= kQ8PrefillA8Tokens && prefill_routes_active();
    bool tier = band && prefill_a8_format(weight.qtype) && allows_a8(policy) && supports(weight);
    // --prefill-route w4a8 / i8: the int8 tier of that family only, where it reads the weight;
    // a16 has no tier (prefill_a8_format). A call kept on its shape table runs A16 there.
    const PrefillRoute pin = route_pins().prefill;
    if (pin == PrefillRoute::W4a8) tier = tier && allows_w4a8(weight, policy);
    if (pin == PrefillRoute::I8) tier = tier && allows_i8(weight, policy);
    if (band && !tier && pin != PrefillRoute::Auto && owned_tables_a16(weight.qtype)) {
        const bool a8 = allows_a8(policy) && supports(weight);
        note_prefill_pin(
            pin == PrefillRoute::A16,
            family_bit(PrefillRoute::A16) |
                (a8 && allows_w4a8(weight, policy) ? family_bit(PrefillRoute::W4a8) : 0U) |
                (a8 && allows_i8(weight, policy) ? family_bit(PrefillRoute::I8) : 0U));
    }
    return tier;
}

bool q8_prefill_a8_fused(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    return q8_prefill_a8(weight, policy, tokens);
}

LinearPolicy fixed_parent_prefill_a8_policy(QType qtype) {
    return prefill_a8_policy(qtype, LinearPolicy::A16Only);
}

bool fixed_pair_prefill_a8(const Weight& first, const Weight& second, std::int32_t tokens) {
    return q8_prefill_a8(first, fixed_parent_prefill_a8_policy(first.qtype), tokens) &&
           q8_prefill_a8(second, fixed_parent_prefill_a8_policy(second.qtype), tokens);
}

std::size_t fixed_parents_prefill_a8_workspace_bytes(std::initializer_list<FixedParent> parents,
                                                     std::int32_t k, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    std::size_t bytes = 0;
    for (const FixedParent& parent : parents) {
        const LinearPolicy policy = fixed_parent_prefill_a8_policy(parent.qtype);
        if (!prefill_a8_format(parent.qtype) || !allows_a8(policy) ||
            max_tokens < kQ8PrefillA8Tokens) {
            return 0;
        }
        bytes = std::max(bytes, workspace_capacity_bytes(
                                    {parent.qtype, policy, parent.rows, k, Epilogue::Store},
                                    nullptr, std::max(min_tokens, kQ8PrefillA8Tokens), max_tokens));
    }
    return bytes;
}

bool volta_tiled_format(QType qtype) {
    return !current_features().mma_tensor && volta_tiled_qtype(qtype);
}

std::string_view schedule_refusal_name(ScheduleRefusal why) noexcept {
    switch (why) {
    case ScheduleRefusal::None:
        return "none";
    case ScheduleRefusal::NotDeclared:
        return "not_declared";
    case ScheduleRefusal::NotCompiled:
        return "not_compiled";
    case ScheduleRefusal::Format:
        return "format";
    case ScheduleRefusal::Instruction:
        return "instruction";
    case ScheduleRefusal::Alignment:
        return "alignment";
    case ScheduleRefusal::Extent:
        return "extent";
    case ScheduleRefusal::CodecTile:
        return "codec_tile";
    case ScheduleRefusal::Policy:
        return "policy";
    case ScheduleRefusal::Resource:
        return "resource";
    }
    return "none";
}

std::string_view r8_call_refusal_name(R8CallRefusal why) noexcept {
    switch (why) {
    case R8CallRefusal::None:
        return "none";
    case R8CallRefusal::NotAdmitted:
        return "not_admitted";
    case R8CallRefusal::DecodeFit:
        return "decode_fit";
    case R8CallRefusal::VerifyWidth:
        return "verify_width";
    case R8CallRefusal::VerifyShape:
        return "verify_shape";
    }
    return "none";
}

bool ggml_mmq_admits(const Weight& w, LinearPolicy policy) { return mmq_admits(w, policy); }

bool ggml_vec_admits(const Weight& w, LinearPolicy policy) { return vec_admits(w, policy); }

std::optional<Route> volta_fixed_route(const Problem& problem, const Weight& weight,
                                       std::int32_t tokens) {
    if (const auto r = volta_tiled_route(weight, problem.policy, tokens)) return r;
    return volta_hmma_route(problem, weight, tokens);
}

namespace {

bool family_on_device(Family family) {
    return has_features(current_features(), route_requires(Route{family, 0}));
}


} // namespace

std::optional<Route> pinned_family_route(PrefillRoute family, const Problem& problem,
                                         const Weight& weight, std::int32_t tokens, bool prefill) {
    if (tokens < kPrefillPinMinTokens) return std::nullopt;
    return prefill_family_route(family, problem, weight, tokens, prefill);
}

std::optional<Route> pinned_family_route(DecodeRoute family, const Problem& problem,
                                         const Weight& weight, std::int32_t tokens, bool prefill) {
    if (tokens < 1 || tokens > kDecodePinMaxTokens) return std::nullopt;
    return decode_family_route(family, problem, weight, tokens, prefill);
}

bool pin_family_on_device(PrefillRoute family) {
    switch (family) {
    case PrefillRoute::Auto:
    case PrefillRoute::A16:
        return true;
    case PrefillRoute::W4a8:
        return w4a8_schedule_count() > 0 && family_on_device(Family::W4a8Mma);
    case PrefillRoute::I8:
        return i8_schedule_count() > 0 && family_on_device(Family::I8Mma);
    case PrefillRoute::Mmq:
        return ggml_mmq::device_supported() && family_on_device(Family::GgmlMmq);
    case PrefillRoute::W4a4:
        return w4a4_schedule_count() > 0 && family_on_device(Family::W4a4Mma);
    }
    return false;
}

bool pin_family_on_device(DecodeRoute family) {
    switch (family) {
    case DecodeRoute::Auto:
    case DecodeRoute::A16:
        return true;
    case DecodeRoute::R8:
        return r8_schedule_count() > 0 && family_on_device(Family::R8Mma);
    case DecodeRoute::Vec:
        return family_on_device(Family::GgmlVec);
    case DecodeRoute::Mmq:
        return ggml_mmq::device_supported() && family_on_device(Family::GgmlMmq);
    }
    return false;
}

bool r8_tunable(const Problem& problem, const Weight& w, std::int32_t tokens) {
    return r8_verify_tunable(w, problem.policy, tokens) ||
           (vec_rule(w, problem.policy, tokens) && r8_call(problem, w, tokens));
}

bool prepared_rules_match(const PreparedRule& a, const PreparedRule& b) noexcept {
    return !a.weight_bound && !b.weight_bound && a.representation == b.representation &&
           a.ld == b.ld && a.variant == b.variant;
}

bool prepared_serves(const PreparedActivation* prepared, PreparedRepresentation representation,
                     const void* source, std::int32_t ldx, std::int32_t k,
                     std::int32_t tokens) noexcept {
    return prepared != nullptr && prepared->data != nullptr &&
           prepared->representation == representation && prepared->source == source &&
           prepared->ldx == ldx && prepared->k == k && prepared->tokens == tokens;
}

std::uint64_t ggml_mmq_fallback_count() noexcept {
    return mmq_fallbacks.load(std::memory_order_relaxed);
}

std::uint64_t instrumentation_joined_launches() noexcept {
    return joined_launches.load(std::memory_order_relaxed);
}

std::uint64_t route_demotion_count() noexcept {
    return route_demotions.load(std::memory_order_relaxed);
}

std::string_view overlap_refusal_name(OverlapRefusal why) noexcept {
    switch (why) {
    case OverlapRefusal::None: return "none";
    case OverlapRefusal::NotAsked: return "not_asked";
    case OverlapRefusal::Family: return "family";
    case OverlapRefusal::Epilogue: return "epilogue";
    case OverlapRefusal::PrequantizedR8: return "prequantized_r8";
    case OverlapRefusal::NoPdl: return "no_pdl";
    case OverlapRefusal::Wide: return "wide";
    case OverlapRefusal::Joined: return "joined";
    }
    return "none";
}

// FUSION-SITE: pdl_overlap
LaunchOverlap launch_overlap(const ResolvedRoute& resolved, Epilogue epilogue, int tokens,
                             bool join_asked, bool release_asked, bool pdl) {
    const Family family = resolved.executed.family;
    const bool r8_pre   = family == Family::R8Mma && resolved.r8_prequantized;
    // Only a register-MMA or R8 Store launch joins or releases; every other route runs in stream
    // order.
    const auto joinable = [&](bool asked) {
        if (!asked) return OverlapRefusal::NotAsked;
        if (family != Family::RegisterMma && family != Family::R8Mma) return OverlapRefusal::Family;
        return epilogue == Epilogue::Store ? OverlapRefusal::None : OverlapRefusal::Epilogue;
    };
    LaunchOverlap out;
    // The pre-quantized R8 form does not join either: its quantizer and kernel launch in stream
    // order (generic_r8.cuh kPre), so a join flag would only claim an overlap that never happens.
    out.join = joinable(join_asked);
    if (out.join == OverlapRefusal::None && r8_pre) out.join = OverlapRefusal::PrequantizedR8;
    // The pre-quantized R8 form does not release: the next call's quantizer could take the
    // workspace it still reads (its kernel ignores the flag too, generic_r8.cuh kPre). Without
    // PDL nothing starts early (run_route keeps the kernel's flag, a no-op there).
    out.release = joinable(release_asked);
    if (out.release == OverlapRefusal::None) {
        out.release = r8_pre ? OverlapRefusal::PrequantizedR8 : !pdl ? OverlapRefusal::NoPdl
                                                                     : OverlapRefusal::None;
    }
    // A decode-width register-MMA call that does not join starts its weight stream under the
    // previous kernel, where the device has programmatic dependent launch (sm_90 and up; Ada
    // launches in stream order as before); R8 does the same where its activation is quantized in
    // the kernel (not the pre-quantized form, whose quantizer runs before it).
    if (r8_pre) {
        out.prefetch = OverlapRefusal::PrequantizedR8;
    } else if (family != Family::RegisterMma && family != Family::R8Mma) {
        out.prefetch = OverlapRefusal::Family;
    } else if (out.joins()) {
        out.prefetch = OverlapRefusal::Joined;
    } else if (tokens > kPdlPrefetchMaxTokens) {
        out.prefetch = OverlapRefusal::Wide;
    } else {
        out.prefetch = pdl ? OverlapRefusal::None : OverlapRefusal::NoPdl;
    }
    return out;
}

R8Coverage r8_coverage(const Problem& problem, const Weight& w, std::int32_t tokens) {
    R8Coverage r;
    FamilyCoverage& c = r.family;
    c.declared        = r8_schedule_count();
    const bool reads  = r8_supports(w.qtype);
    for (int i = 0; i < c.declared; ++i) {
        const ScheduleRefusal why = r8_refusal(i, w, problem.policy);
        c.schedules.push_back({i, why});
        c.compiled += reads;
        c.legal += why == ScheduleRefusal::None;
    }
    c.heuristic             = heuristic_r8(problem, tokens);
    c.heuristic_refusal     = r8_refusal(c.heuristic, w, problem.policy);
    r.call                  = r8_call_refusal(problem, w, tokens);
    r.prequantized_compiled = r8_verify_compiled(w.qtype);
    return r;
}

std::vector<FamilyAlternative> family_alternatives(const Problem& problem, const Weight& w,
                                                   std::int32_t tokens, bool prefill) {
    const bool band = prefill && tokens >= kPrefillBandMinTokens;
    // The calls heuristic_route prices (cost_route): 17 tokens or more outside the prefill profile,
    // NVFP4 apart. The times are cost_route's own, so a priced alternative compares as it would.
    std::vector<std::pair<Route, double>> priced;
    if (!band && tokens >= kPrefillBandMinTokens && problem.qtype != QType::NVFP4) {
        price_routes(problem, w, tokens,
                     [&](Route r, double time) { priced.emplace_back(r, time); });
    }
    std::vector<FamilyAlternative> out;
    const auto take = [&](Family family, const FamilyCoverage& c, R8CallRefusal call) {
        FamilyAlternative a;
        a.family   = family;
        a.compiled = c.compiled;
        a.call     = call;
        for (const auto& [r, time] : priced) {
            if (r.family == family && (!a.seconds || time < *a.seconds)) {
                a.route   = r;
                a.seconds = time;
            }
        }
        if (!a.route && call == R8CallRefusal::None) {
            if (c.heuristic >= 0 && c.heuristic_refusal == ScheduleRefusal::None) {
                a.route = Route{family, static_cast<std::uint8_t>(c.heuristic)};
            } else {
                for (const ScheduleCoverage& s : c.schedules) {
                    if (s.refusal != ScheduleRefusal::None) continue;
                    a.route = Route{family, static_cast<std::uint8_t>(s.schedule)};
                    break;
                }
            }
        }
        if (!a.route) {
            a.refusal = c.heuristic >= 0      ? c.heuristic_refusal
                        : c.schedules.empty() ? ScheduleRefusal::NotDeclared
                                              : c.schedules.front().refusal;
        }
        out.push_back(a);
    };
    take(Family::RegisterMma, rmma_coverage(problem, w, tokens, band), R8CallRefusal::None);
    const R8Coverage r8 = r8_coverage(problem, w, tokens);
    take(Family::R8Mma, r8.family, r8.call);
    take(Family::W4a8Mma, w4a8_coverage(problem, w, tokens), R8CallRefusal::None);
    take(Family::I8Mma, i8_coverage(problem, w, tokens), R8CallRefusal::None);
    take(Family::TiledMma, tiled_coverage(w), R8CallRefusal::None);
    take(Family::StagedMma, staged_coverage(w), R8CallRefusal::None);
    for (const Family f :
         {Family::W4a4Mma, Family::Nvfp4TiledMma, Family::GgmlMmq, Family::GgmlVec}) {
        take(f, other_family_coverage(f, w, problem.policy), R8CallRefusal::None);
    }
    return out;
}

void launch_rmma_codec(WeightCodec codec, int schedule, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::Bf16:
        launch_rmma_bf16(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Fp8:
        launch_rmma_fp8(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Nvfp4:
        launch_rmma_nvfp4(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q4:
        launch_rmma_q4(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q5:
        launch_rmma_q5(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q6:
        launch_rmma_q6(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q3G128:
        launch_rmma_q3g128(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q3G64:
        launch_rmma_q3g64(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q8:
        launch_rmma_q8(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Fp8RowF32:
        launch_rmma_fp8_row_f32(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Fp8B128:
        launch_rmma_fp8_b128(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4Zp:
        launch_rmma_u4zp(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4ZpG32:
        launch_rmma_u4zp_g32(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4ZpG64:
        launch_rmma_u4zp_g64(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ4K:
        launch_rmma_ggml_q4k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ5K:
        launch_rmma_ggml_q5k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ6K:
        launch_rmma_ggml_q6k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ3K:
        launch_rmma_ggml_q3k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ2K:
        launch_rmma_ggml_q2k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq4Xs:
        launch_rmma_ggml_iq4xs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq3Xxs:
        launch_rmma_ggml_iq3xxs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq3S:
        launch_rmma_ggml_iq3s(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ40:
        launch_rmma_ggml_q40(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ41:
        launch_rmma_ggml_q41(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ50:
        launch_rmma_ggml_q50(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ51:
        launch_rmma_ggml_q51(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq4Nl:
        launch_rmma_ggml_iq4nl(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq2Xxs:
        launch_rmma_ggml_iq2xxs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq2Xs:
        launch_rmma_ggml_iq2xs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq2S:
        launch_rmma_ggml_iq2s(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq1S:
        launch_rmma_ggml_iq1s(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq1M:
        launch_rmma_ggml_iq1m(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlTq10:
        launch_rmma_ggml_tq10(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlTq20:
        launch_rmma_ggml_tq20(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ10:
        launch_rmma_ggml_q10(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ20:
        launch_rmma_ggml_q20(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Mxfp4:
        launch_rmma_mxfp4(schedule, epilogue, args, stream);
        return;
    }
}

void run_route(Route route, const Call& ordered, WorkspaceArena* workspace, cudaStream_t stream,
               bool* released) {
    if (released != nullptr) *released = false;
    validate_call(ordered);
    const Weight& w = *ordered.weight;
    if (!supports(w)) throw std::invalid_argument("generic linear: unsupported weight layout");
    // An explicit route parses on every build; one whose kernels this build lacks is refused as
    // route_admits refuses it (the Ada build's W4A4).
    if (route.family == Family::W4a4Mma && route.schedule >= w4a4_schedule_count()) {
        throw std::invalid_argument("generic linear: route " + route_name(route) +
                                    " is not compiled in this build (NVFP4 MMA, sm_120a)");
    }
    // A weight stored in another column order multiplies the input gathered into that order.
    Call call = ordered;
    std::optional<WorkspaceArena::Scope> scope;
    Tensor gathered;
    if (w.k_order != nullptr) {
        if (workspace == nullptr) {
            throw std::invalid_argument(
                "generic linear: a weight stored in another column order needs caller workspace");
        }
        scope.emplace(workspace->scope());
        const int tokens = call.x->ne[1];
        gathered         = workspace->alloc(DType::BF16, {w.k, tokens});
        launch_gather_k(call.x->data, call.x->ne[0], w.k_order, w.k, tokens, gathered.data,
                        stream);
        call.x = &gathered;
    }
    GemmArgs args     = make_args(call);
    const Route asked = route;
    const ResolvedRoute resolved =
        resolve_route(route, w,
                      CallLayout{.x_aligned16    = aligned_to(args.x, 16),
                                 .ldx            = args.ldx,
                                 .tokens         = args.tokens,
                                 .rows           = call.rows,
                                 .segments       = args.segments,
                                 .epilogue       = call.epilogue,
                                 .workspace_free = workspace_free(workspace)});
    route             = resolved.executed;
    const bool r8_pre = resolved.r8_prequantized;
    // A caller that sized its workspace short of ggml's route (said below, with every other
    // demotion) is counted too (ggml_mmq_fallback_count).
    if (resolved.ggml_fallback) mmq_fallbacks.fetch_add(1, std::memory_order_relaxed);
    const bool gated     = call.epilogue == Epilogue::SwiGlu;
    const int plane_rows = gated ? 2 * call.rows : call.rows;
    if (route.family != asked.family && !same_bytes(asked, route)) {
        note_demotion(asked, route, call.rows, args.tokens, w.k, aligned_to(args.x, 16), args.ldx,
                      workspace != nullptr ? workspace->capacity() : 0);
    }
    // The launch's overlap with its neighbours (launch_overlap, #309). A joinable launch keeps its
    // release flag without PDL (the trigger is then a no-op), as before; the pre-quantized R8
    // form's is cleared, which its kernel ignored (kPre).
    const LaunchOverlap overlap = launch_overlap(resolved, call.epilogue, args.tokens,
                                                 call.join_previous, call.release_next,
                                                 pdl_device());
    args.join_previous = overlap.joins() ? 1 : 0;
    if (overlap.joins()) joined_launches.fetch_add(1, std::memory_order_relaxed);
    args.release_next  = overlap.releases() || overlap.release == OverlapRefusal::NoPdl ? 1 : 0;
    if (released != nullptr) *released = overlap.releases();
    args.pdl_prefetch = overlap.prefetches() ? 1 : 0;
    const WeightCodec codec = codec_of(w);
    args.spread_smem        = spread_smem(route, codec, call.epilogue, args);
    switch (route.family) {
    case Family::GgmlMmq: {
        auto scope       = workspace->scope();
        const auto type  = *mmq_type(w);
        const int slice  = std::min(args.tokens, kMmqPrefillSlice);
        const int bpr    = w.k / static_cast<int>(ggml_block_values(w.qtype));
        void* activation = workspace->alloc_bytes(ggml_mmq::activation_bytes(w.k, slice)).data;
        auto* plane      = static_cast<float*>(
            workspace->alloc_bytes(static_cast<std::size_t>(plane_rows) * slice * sizeof(float))
                .data);
        void* fixup        = workspace->alloc_bytes(ggml_mmq::fixup_bound(slice)).data;
        const auto rows_at = [&](int row) { return ggml_rows_at(args, w.qtype, bpr, row); };
        // A prefill call kMmqPrefillSlice tokens at a time through the one activation and plane.
        for (int t0 = 0; t0 < args.tokens; t0 += slice) {
            const auto first = static_cast<std::size_t>(t0);
            GemmArgs part    = args;
            part.tokens      = std::min(slice, args.tokens - t0);
            part.x           = static_cast<const __nv_bfloat16*>(args.x) + first * args.ldx;
            for (int seg = 0; seg < args.segments; ++seg) {
                part.out[seg] = static_cast<__nv_bfloat16*>(args.out[seg]) + first * args.ld[seg];
            }
            ggml_mmq::quantize_activation(type, static_cast<const __nv_bfloat16*>(part.x),
                                          args.ldx, w.k, part.tokens, activation, stream);
            ggml_mmq::product(type, rows_at(call.row_begin), bpr, call.rows, w.k, activation,
                              part.tokens, plane, plane_rows, fixup, stream);
            if (gated) {
                ggml_mmq::product(type, rows_at(call.row_begin + call.up_offset), bpr, call.rows,
                                  w.k, activation, part.tokens, plane + call.rows, plane_rows,
                                  fixup, stream);
            }
            launch_mmq_epilogue(call.epilogue, part, plane, plane_rows, stream);
        }
        return;
    }
    case Family::GgmlVec: {
        // Wider than the kernel's columns (vec_max_tokens): the call runs eight tokens at a time,
        // each part quantizing its own activation.
        if (args.tokens > ggml_mmq::kVecMaxColumns) {
            for (int begin = 0; begin < args.tokens; begin += ggml_mmq::kVecMaxColumns) {
                const int n     = std::min(ggml_mmq::kVecMaxColumns, args.tokens - begin);
                auto scope      = workspace->scope();
                const auto type = *mmq_type(w);
                const int bpr   = w.k / static_cast<int>(ggml_block_values(w.qtype));
                const auto* x   = static_cast<const __nv_bfloat16*>(args.x) +
                                static_cast<std::int64_t>(begin) * args.ldx;
                void* activation =
                    workspace->alloc_bytes(ggml_mmq::vec_activation_bytes(w.k, n)).data;
                ggml_mmq::vec_quantize_activation(x, args.ldx, w.k, n, activation, stream);
                ggml_mmq::VecOut out;
                out.epilogue  = gated ? ggml_mmq::VecOut::Epilogue::SwiGlu
                                : call.epilogue == Epilogue::AddResidual
                                    ? ggml_mmq::VecOut::Epilogue::AddResidual
                                    : ggml_mmq::VecOut::Epilogue::Store;
                out.gelu_gate = args.gelu_gate;
                out.relu2     = args.relu2;
                out.segments  = args.segments;
                for (int s = 0; s < args.segments; ++s) {
                    out.out[s] = static_cast<__nv_bfloat16*>(args.out[s]) +
                                 static_cast<std::int64_t>(begin) * args.ld[s];
                    out.segment_end[s] = args.segment_end[s];
                    out.ld[s]          = args.ld[s];
                }
                ggml_mmq::vec_product(
                    type, ggml_rows_at(args, w.qtype, bpr, call.row_begin),
                    gated ? ggml_rows_at(args, w.qtype, bpr, call.row_begin + call.up_offset)
                          : ggml_mmq::Planes{},
                    bpr, call.rows, w.k, activation, n, out, stream);
            }
            return;
        }
        auto scope       = workspace->scope();
        const auto type  = *mmq_type(w);
        const int tokens = args.tokens;
        const int bpr    = w.k / static_cast<int>(ggml_block_values(w.qtype));
        // The shared q8_1 where it was made from the x this call reads (after any column gather),
        // at its stride and extent; else this call's own.
        const void* activation =
            prepared_serves(call.prepared, PreparedRepresentation::GgmlQ8_1, args.x, args.ldx, w.k,
                            tokens)
                ? call.prepared->data
                : nullptr;
        if (activation == nullptr) {
            void* own = workspace->alloc_bytes(ggml_mmq::vec_activation_bytes(w.k, tokens)).data;
            ggml_mmq::vec_quantize_activation(static_cast<const __nv_bfloat16*>(args.x), args.ldx,
                                              w.k, tokens, own, stream);
            activation = own;
        }
        ggml_mmq::VecOut out;
        out.epilogue  = gated ? ggml_mmq::VecOut::Epilogue::SwiGlu
                        : call.epilogue == Epilogue::AddResidual
                            ? ggml_mmq::VecOut::Epilogue::AddResidual
                            : ggml_mmq::VecOut::Epilogue::Store;
        out.gelu_gate = args.gelu_gate;
        out.relu2     = args.relu2;
        out.segments  = args.segments;
        for (int s = 0; s < args.segments; ++s) {
            out.out[s]         = args.out[s];
            out.segment_end[s] = args.segment_end[s];
            out.ld[s]          = args.ld[s];
        }
        ggml_mmq::vec_product(
            type, ggml_rows_at(args, w.qtype, bpr, call.row_begin),
            gated ? ggml_rows_at(args, w.qtype, bpr, call.row_begin + call.up_offset)
                  : ggml_mmq::Planes{},
            bpr, call.rows, w.k, activation, tokens, out, stream);
        return;
    }
    case Family::Scalar:
        launch_scalar(codec, call.epilogue, args, stream);
        return;
    case Family::RegisterMma:
        // K is on the kernel's step here: resolve_route took a call off it to Scalar (KStep).
        args.lane_local = invariant_routes_active() ? 0 : 1;
        args.add_sums   = rmma_add_sums() ? 1 : 0;
        if (!current_features().mma_tensor && volta_hmma_takes(codec, args)) {
            launch_volta_hmma(codec, call.epilogue, args, stream);
            return;
        }
        if (!current_features().mma_tensor && volta_gemv_takes(codec, args) &&
            (args.ldx % 8) == 0 && (reinterpret_cast<std::uintptr_t>(args.x) % 16) == 0) {
            launch_volta_gemv(codec, call.epilogue, args, stream);
            return;
        }
        launch_rmma_codec(codec, route.schedule, call.epilogue, args, stream);
        return;
    case Family::TiledMma:
        if ((w.k % 64) != 0) {
            throw std::invalid_argument("generic tiled: K must be whole 64-value tiles");
        }
        // The prefill band on Volta (kVoltaCutlassMinTokens): a CUTLASS-only format has no tile of
        // its own (V100, gpt-oss BF16 attention at T = 79 ran scalar, 7.3 ms a call, 360 -> 140
        // tok/s), and a tiled one's grid seldom fills the SMs' waves (volta_copy_pays).
        {
            const auto cutlass_free = workspace_free(workspace);
            if (volta_cutlass_call_runs(codec, call.rows, w.k, args.tokens,
                                        call.epilogue == Epilogue::SwiGlu, cutlass_free) &&
                launch_volta_cutlass(codec, call.epilogue, args, *workspace, *cutlass_free,
                                     stream)) {
                return;
            }
        }
        // A call the decoded copy does not fit runs scalar: a demotion (other bytes and speed);
        // volta_tiled_runs_scalar says it for a description.
        if (volta_cutlass_only(codec)) {
            note_demotion(route, Route{Family::Scalar, 0}, call.rows, args.tokens, w.k,
                          aligned_to(args.x, 16), args.ldx,
                          workspace != nullptr ? workspace->capacity() : 0);
            launch_scalar(codec, call.epilogue, args, stream);
            return;
        }
        launch_tiled(route.schedule, codec, call.epilogue, args, stream);
        return;
    case Family::StagedMma:
        launch_staged(route.schedule, call.epilogue, args, stream);
        return;
    case Family::Nvfp4TiledMma:
#ifdef NINFERNO_WITHOUT_NVFP4
        throw std::invalid_argument("generic NVFP4 tiled GEMM is an sm_120a path on this build");
#else
        launch_nvfp4_tiled(nvfp4_tiled_schedule_info(route.schedule).classes, call.epilogue, args,
                           stream);
#endif
        return;
    case Family::W4a4Mma:
#ifdef NINFERNO_WITHOUT_NVFP4
        throw std::logic_error(
            "generic W4A4 needs the NVFP4 MMA (sm_120a); route selection must not offer it on Ada");
#else
    {
        if (workspace == nullptr) {
            throw std::invalid_argument("generic W4A4 requires caller workspace");
        }
        auto scope         = workspace->scope();
        const auto planes  = activation_planes(route, args.k, args.tokens);
        const DeviceSpan c = workspace->alloc_bytes(planes.codes, 256);
        const DeviceSpan s = workspace->alloc_bytes(planes.scales, 256);
        launch_w4a4(route.schedule, call.epilogue, args, w.input_scale_divisor,
                    static_cast<std::uint8_t*>(c.data), static_cast<std::uint8_t*>(s.data),
                    stream);
        return;
    }
#endif
    case Family::W4a8Mma: {
        if (workspace == nullptr) {
            throw std::invalid_argument("generic W4A8 requires caller workspace");
        }
        auto scope          = workspace->scope();
        const auto planes   = activation_planes(route, args.k, args.tokens);
        const DeviceSpan c  = workspace->alloc_bytes(planes.codes, 256);
        const DeviceSpan sc = workspace->alloc_bytes(planes.scales, 256);
        const DeviceSpan su = workspace->alloc_bytes(planes.sums, 256);
        launch_w4a8(route.schedule, codec, call.epilogue, args, static_cast<std::int8_t*>(c.data),
                    static_cast<float*>(sc.data), static_cast<float*>(su.data), kW4a8TokenMajor,
                    stream);
        return;
    }
    case Family::I8Mma: {
        if (workspace == nullptr) {
            throw std::invalid_argument("generic I8 requires caller workspace");
        }
        auto scope          = workspace->scope();
        const auto planes   = activation_planes(route, args.k, args.tokens);
        const DeviceSpan c  = workspace->alloc_bytes(planes.codes, 256);
        const DeviceSpan sc = workspace->alloc_bytes(planes.scales, 256);
        const DeviceSpan su = workspace->alloc_bytes(planes.sums, 256);
        launch_i8(route.schedule, i8_bits(w.qtype), call.epilogue, args,
                  static_cast<std::int8_t*>(c.data),
                  static_cast<float*>(sc.data), static_cast<int*>(su.data), stream);
        return;
    }
    case Family::R8Mma:
        if (r8_pre) {
            auto scope          = workspace->scope();
            const auto planes   = activation_planes(route, args.k, args.tokens);
            const DeviceSpan c  = workspace->alloc_bytes(planes.codes, 256);
            const DeviceSpan sc = workspace->alloc_bytes(planes.scales, 256);
            const DeviceSpan su = workspace->alloc_bytes(planes.sums, 256);
            auto* codes  = static_cast<std::int8_t*>(c.data);
            auto* scales = static_cast<float*>(sc.data);
            auto* sums   = static_cast<float*>(su.data);
            launch_w4a8_quantize(args, codes, scales, sums, stream);
            launch_r8(route.schedule, codec, call.epilogue, args, stream,
                      W4a8Activation{codes, scales, sums});
            return;
        }
        launch_r8(route.schedule, codec, call.epilogue, args, stream);
        return;
    }
}

void run(const Call& call, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream,
         bool* released) {
    const Problem problem{call.weight->qtype, policy, call.rows, call.weight->k, call.epilogue};
    record_problem(problem, *call.weight);
    tune_dense(problem, call, stream);
    const Route route =
        workspace_free_route(select_route(problem, *call.weight, call.x->ne[1], prefill_routes_active()),
                             problem, *call.weight, call.x->ne[1], prefill_routes_active(),
                             workspace != nullptr);
    run_route(route, call, workspace, stream, released);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::Store;
    call.out[0]          = &out;
    call.segment_rows[0] = w.n;
    run(call, policy, workspace, stream);
}

void linear_relu2(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::Store;
    call.relu2           = true;
    call.out[0]          = &out;
    call.segment_rows[0] = w.n;
    run(call, policy, workspace, stream);
}

bool linear_release_next(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                         WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::Store;
    call.out[0]          = &out;
    call.segment_rows[0] = w.n;
    call.release_next    = true;
    bool released        = false;
    run(call, policy, workspace, stream, &released);
    return released;
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::AddResidual;
    call.out[0]          = &residual;
    call.segment_rows[0] = w.n;
    run(call, policy, workspace, stream);
}

namespace {

void run_gated(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
               WorkspaceArena* workspace, cudaStream_t stream, bool gelu) {
    if ((gate_up.n % 2) != 0) {
        throw std::invalid_argument("generic linear_swiglu: gate/up rows must be even");
    }
    Call call;
    call.x               = &x;
    call.weight          = &gate_up;
    call.rows            = gate_up.n / 2;
    call.up_offset       = gate_up.n / 2;
    call.epilogue        = Epilogue::SwiGlu;
    call.gelu_gate       = gelu;
    call.out[0]          = &out;
    call.segment_rows[0] = gate_up.n / 2;
    run(call, policy, workspace, stream);
}


} // namespace

void linear_swiglu(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream) {
    run_gated(x, gate_up, out, policy, workspace, stream, false);
}

void linear_geglu(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    run_gated(x, gate_up, out, policy, workspace, stream, true);
}

} // namespace ninferno::ops::detail::generic
