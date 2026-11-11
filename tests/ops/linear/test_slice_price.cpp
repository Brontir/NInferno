// A slice of a fused weight is priced by its own rows (#640), host only on the substituted boards
// of the selector goldens (selector_facts.h; the register-MMA price reads the committed
// kernel-facts snapshots, as the golden does). The engine hands the selector a slice as the whole
// parent with the call's rows in Problem::n (segmented_projection, ggml's gated pair halves); the
// call reads its rows' share of the parent's bytes (under SwiGLU its gate and its up rows), so its
// price, its traffic and the route the selector takes must be those of a standalone weight of the
// slice's own shape, packed alone: the same format, rows and K. The pure query agrees: a call
// described as a slice of its parent (route_query, describe_call with parent_rows) answers as
// the own shape at every width, the widths and band the cost model does not price included.
//
// The same rule for a row view (#704; native_weight of part of a parent: its own rows in n, the
// parent's payload) and for the R8 verify gate, which asks the call's own rows: a Q4_0 slice below
// the gate's bound of a parent above it answers as its own shape at T = 2..4. A row view is priced
// by the bytes of a standalone weight of its shape, its planes' 256-byte padding included, to the
// bit: the price and the route the decision manifest's replay of the recorded call gives
// (route_query with the view's rows and layout), small padded views included.
//
// Complementary to the whole-weight identity in test_route_traffic.cpp: this test compares a slice
// with its own shape through the same bytes_per_weight, so a factor applied to both sides alike
// (dropping the SwiGLU `* sets`, flooring the stored rows at 1 instead of at the call's rows)
// passes here and only that identity catches it; a revert at any one pricing site (a8_candidates,
// price_routes, route_traffic, describe_resolved) is caught here.
//
// Without a GPU ggml's matrix kernel cannot read the device's properties for its column tiles, so
// the pairs that choose between MMQ and the register MMA throw on both sides and are counted, not
// compared (113 on a sm_89 build); with a device every pair is compared.
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

int checks = 0, failures = 0, moved = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// Within the plane paddings: each packed plane is padded to 256 bytes, so a slice's share of its
// parent's bytes is the standalone weight's to a few parts in a million.
bool near(double a, double b) {
    return std::fabs(a - b) <= 1e-4 * std::max(std::fabs(a), std::fabs(b));
}

bool near(const std::optional<double>& a, const std::optional<double>& b) {
    return a.has_value() == b.has_value() && (!a || near(*a, *b));
}

std::string text(const std::optional<generic::Route>& r) {
    return r ? generic::route_name(*r) : std::string("none");
}

// The selector's answer, or what it threw.
std::string selected(const generic::Problem& p, const Weight& w, std::int32_t t, bool prefill) {
    try {
        return generic::route_name(generic::select_route(p, w, t, prefill));
    } catch (const std::exception& error) { return std::string("threw: ") + error.what(); }
}

// Whether `f` throws. A host without a GPU cannot answer every call (ggml's matrix kernel reads
// the device's properties for its column tiles): the slice must then throw as its own shape does,
// and the pair is counted as not compared.
int not_compared = 0;

template <class F>
bool throws(F&& f) {
    try {
        f();
        return false;
    } catch (const std::exception&) { return true; }
}

bool both_answer(bool own_threw, bool slice_threw, const std::string& at) {
    check(own_threw == slice_threw, at + ": the slice throws where its own shape does");
    if (own_threw || slice_threw) ++not_compared;
    return !own_threw && !slice_threw;
}

constexpr std::int32_t kK = 4096, kOwnRows = 1024, kParentRows = 8192;
constexpr QType kFormats[]             = {QType::Q8_G32_FP16, QType::GGML_Q4_K, QType::GGML_Q6_K,
                                          QType::Q4_G64_FP16, QType::FP8_E4M3FN_ROW_BF16};
constexpr std::int32_t kPricedTokens[] = {17, 32, 64, 128, 512};

// A matrix laid out as the converter stores it, its bytes never read (an aligned address stands
// in), and a weight read through `view` of it: the whole matrix, or a row view of `rows` rows from
// `begin` (native_weight of part of a parent: its own rows in n, the parent's payload). Filled in
// place: the view and the weight point into the parent.
struct Described {
    WeightParent parent;
    WeightView view;
    Weight weight;

    Described(QType qtype, std::uint64_t parent_rows, std::uint64_t begin, std::uint64_t rows,
              std::uint64_t k = kK) {
        parent.data                  = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
        const std::uint64_t shape[2] = {parent_rows, k};
        parent.geometry              = weight_geometry(qtype, default_weight_layout(qtype), shape);
        view.shape                   = {rows, k};
        view.parts                   = {WeightRegion{&parent, begin * k, (begin + rows) * k}};
        weight                       = native_weight(view);
    }

    Described(const Described&)            = delete;
    Described& operator=(const Described&) = delete;
};

// The formats whose layout has row views (the rows are intrinsic; FP8 and NVFP4 have none).
constexpr QType kRowViewFormats[] = {
    QType::BF16,      QType::Q8_G32_FP16, QType::Q4_G64_FP16, QType::U4_G128_ZP_FP16, QType::MXFP4,
    QType::GGML_Q4_K, QType::GGML_Q6_K,   QType::GGML_Q4_0,   QType::GGML_IQ4_XS};

int view_moved = 0;

// A row view of kOwnRows rows from the middle of a kParentRows parent is priced and routed as a
// standalone weight of its shape (#704): the same format, rows and K, its payload its own.
void check_row_views() {
    for (const QType qtype : kRowViewFormats) {
        const std::string fmt(generic::qtype_name(qtype));
        std::optional<Described> own, view;
        try {
            own.emplace(qtype, kOwnRows, 0, kOwnRows);
            view.emplace(qtype, kParentRows, 3 * kOwnRows, kOwnRows);
        } catch (const std::invalid_argument& error) {
            std::cout << fmt << ": no row view (" << error.what() << ")\n";
            continue;
        }
        if (!generic::supports(own->weight) || !generic::supports(view->weight)) {
            std::cout << fmt << ": not read by the generic path\n";
            continue;
        }
        check(view->weight.n == kOwnRows && view->weight.payload_bytes > own->weight.payload_bytes,
              fmt + ": the view has its own rows and its parent's payload");
        for (const generic::FactsBoard& board : generic::kFactsBoards) {
            const ScopedDeviceClass scoped(generic::board_device_class(board));
            for (const LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
                for (const generic::Epilogue epilogue :
                     {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                    const bool swiglu       = epilogue == generic::Epilogue::SwiGlu;
                    const std::int32_t rows = swiglu ? kOwnRows / 2 : kOwnRows;
                    const generic::Problem p{qtype, policy, rows, kK, epilogue};
                    const std::string call = std::string(board.name) + " " + fmt + " view p" +
                                             std::to_string(static_cast<int>(policy)) + " " +
                                             generic::epilogue_name(epilogue);
                    // To the bit: the view's bytes are its own shape's (Weight::priced_bytes).
                    check(generic::bytes_per_weight(own->weight, p) ==
                              generic::bytes_per_weight(view->weight, p),
                          call + ": bytes per weight " +
                              std::to_string(generic::bytes_per_weight(view->weight, p)) +
                              ", its own shape's " +
                              std::to_string(generic::bytes_per_weight(own->weight, p)));
                    for (const std::int32_t t : kPricedTokens) {
                        const std::string at = call + " T=" + std::to_string(t);
                        for (const bool prefill : {false, true}) {
                            const std::string o     = selected(p, own->weight, t, prefill);
                            const std::string s     = selected(p, view->weight, t, prefill);
                            const std::string where = at + (prefill ? " prefill" : " speed");
                            if (!both_answer(o.starts_with("threw"), s.starts_with("threw"),
                                             where)) {
                                continue;
                            }
                            if (o != s) ++view_moved;
                            check(o == s, where + ": the view takes " + s + ", its own shape " + o);
                        }
                        std::vector<generic::FamilyAlternative> fo, fs;
                        const bool fo_threw = throws(
                            [&] { fo = generic::family_alternatives(p, own->weight, t, false); });
                        const bool fs_threw = throws(
                            [&] { fs = generic::family_alternatives(p, view->weight, t, false); });
                        if (!both_answer(fo_threw, fs_threw, at + " families")) continue;
                        check(fo.size() == fs.size(), at + ": the same families");
                        for (std::size_t i = 0; i < std::min(fo.size(), fs.size()); ++i) {
                            check(fo[i].family == fs[i].family && fo[i].route == fs[i].route &&
                                      near(fo[i].seconds, fs[i].seconds),
                                  at + " family " + std::to_string(i) + ": the view's " +
                                      text(fs[i].route) + " " +
                                      std::to_string(fs[i].seconds.value_or(-1) * 1e6) +
                                      " us, its own shape's " + text(fo[i].route) + " " +
                                      std::to_string(fo[i].seconds.value_or(-1) * 1e6) + " us");
                        }
                        for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                            const generic::Route r{generic::Family::RegisterMma,
                                                   static_cast<std::uint8_t>(i)};
                            const auto to = generic::route_traffic(r, own->weight, p, t);
                            const auto ts = generic::route_traffic(r, view->weight, p, t);
                            check(to.has_value() == ts.has_value() &&
                                      (!to || near(to->weight_bytes, ts->weight_bytes)),
                                  at + ": traffic of " + generic::route_name(r));
                        }
                    }
                }
            }
        }
    }
}

int gate_differs = 0;

int replay_differs = 0;

// Small row views of formats whose planes are padded to 256 bytes (#704): the parent's padding is
// not spread over the view. A view is priced by the bytes of a standalone weight of its shape, to
// the bit, which is what the decision manifest's replay prices: the run records the call with the
// view's rows and layout (record_problem: Problem, w.n, w.layout), and replay describes that shape
// (route_query, described_bytes_per_weight). The selector's answer on the live view and the
// replay's agree at every priced width, in the speed profile and the prefill band.
void check_padded_view_replay() {
    struct Case {
        QType qtype;
        std::uint64_t k, parent_rows, begin, rows;
    };

    constexpr Case kCases[] = {
        {QType::Q4_G64_FP16, 128, 3, 0, 1}, // one row: 260 bytes; its parent of three, 268
        {QType::Q4_G64_FP16, 128, 3, 1, 1},     {QType::Q8_G32_FP16, 128, 5, 2, 2},
        {QType::U4_G128_ZP_FP16, 128, 3, 0, 1}, {QType::MXFP4, 128, 3, 1, 2},
        {QType::GGML_Q6_K, 256, 3, 0, 1},       {QType::GGML_Q4_0, 128, 3, 2, 1},
        {QType::GGML_IQ4_XS, 256, 4, 1, 2},     {QType::BF16, 128, 3, 1, 1},
        {QType::Q4_G64_FP16, 4096, 48, 16, 16}, {QType::GGML_Q6_K, 4096, 40, 0, 24},
    };
    for (const Case& c : kCases) {
        const std::string fmt = std::string(generic::qtype_name(c.qtype)) + " " +
                                std::to_string(c.rows) + "x" + std::to_string(c.k) + " of " +
                                std::to_string(c.parent_rows) + " from " + std::to_string(c.begin);
        const Described view(c.qtype, c.parent_rows, c.begin, c.rows, c.k);
        const Described own(c.qtype, c.rows, 0, c.rows, c.k);
        const Weight& w               = view.weight;
        const std::uint64_t shape[2]  = {c.rows, c.k};
        const std::uint64_t own_bytes = weight_geometry(c.qtype, w.layout, shape).bytes;
        check(w.n == static_cast<std::int32_t>(c.rows) &&
                  w.payload_bytes == view.parent.geometry.bytes && w.priced_bytes == own_bytes &&
                  own.weight.priced_bytes == 0 && own.weight.payload_bytes == own_bytes,
              fmt + ": the view keeps its parent's payload and prices its own shape's " +
                  std::to_string(own_bytes) + " bytes, not " + std::to_string(w.priced_bytes));
        const bool routed = generic::supports(w) && generic::supports(own.weight);
        if (!routed) std::cout << fmt << ": not read by the generic path (bytes only)\n";
        for (const LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
            for (const generic::Epilogue epilogue :
                 {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                const bool swiglu = epilogue == generic::Epilogue::SwiGlu;
                if (swiglu && c.rows % 2) continue;
                const auto rows = static_cast<std::int32_t>(swiglu ? c.rows / 2 : c.rows);
                const auto k    = static_cast<std::int32_t>(c.k);
                const generic::Problem p{c.qtype, policy, rows, k, epilogue};
                const std::string call = fmt + " p" + std::to_string(static_cast<int>(policy)) +
                                         " " + generic::epilogue_name(epilogue);
                const double live   = generic::bytes_per_weight(w, p);
                const double replay = generic::described_bytes_per_weight(c.qtype, policy, epilogue,
                                                                          rows, k, w.n, w.layout);
                check(live == replay && live == generic::bytes_per_weight(own.weight, p),
                      call + ": bytes per weight " + std::to_string(live) + ", replayed " +
                          std::to_string(replay) + ", its own shape's " +
                          std::to_string(generic::bytes_per_weight(own.weight, p)));
                if (c.qtype == QType::Q4_G64_FP16 && c.k == 128 && c.rows == 1 && !swiglu) {
                    check(live == 260.0 / 128.0,
                          call + ": bytes per weight " + std::to_string(live) + ", not 260/128");
                }
                if (!routed) continue;
                for (const generic::FactsBoard& board : generic::kFactsBoards) {
                    const DeviceClass d = generic::board_device_class(board);
                    const ScopedDeviceClass scoped(d);
                    for (const std::int32_t t : kPricedTokens) {
                        for (const bool prefill : {false, true}) {
                            const std::string at = std::string(board.name) + " " + call +
                                                   " T=" + std::to_string(t) +
                                                   (prefill ? " prefill" : " speed");
                            const std::string s = selected(p, w, t, prefill);
                            std::optional<generic::Route> q;
                            const bool q_threw = throws([&] {
                                q = generic::route_query(d, c.qtype, policy, epilogue, rows, k, t,
                                                         prefill, w.n, w.layout);
                            });
                            if (!both_answer(q_threw, s.starts_with("threw"), at + " replay")) {
                                continue;
                            }
                            if (text(q) != s) ++replay_differs;
                            check(text(q) == s, at + ": the view takes " + s +
                                                    ", the replay of its line " + text(q));
                        }
                    }
                }
            }
        }
    }
}

// The R8 verify gate (Q4_0 takes R8 at T = 2..4 only on a matrix of kR8VerifyQ40MinValues values
// or more, 24M) asks the call's own rows (#704): a slice answers as a standalone weight of its
// shape, below the bound (2048 x 4096 = 8M of an 8192-row parent, 32M) and above it (8192 x 4096
// of a 16384-row parent), in the selector, the coverage it reports and the pure query.
void check_r8_verify_gate() {
    constexpr QType qtype = QType::GGML_Q4_0;
    for (const auto& [own_rows, parent_rows] :
         {std::pair<std::int32_t, std::int32_t>{2048, 8192},
          std::pair<std::int32_t, std::int32_t>{8192, 16384}}) {
        const Described own(qtype, own_rows, 0, own_rows);
        const Described parent(qtype, parent_rows, 0, parent_rows);
        for (const generic::FactsBoard& board : generic::kFactsBoards) {
            const DeviceClass d = generic::board_device_class(board);
            const ScopedDeviceClass scoped(d);
            for (const generic::Epilogue epilogue :
                 {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                const bool swiglu       = epilogue == generic::Epilogue::SwiGlu;
                const std::int32_t rows = swiglu ? own_rows / 2 : own_rows;
                const generic::Problem p{qtype, LinearPolicy::AllowA8, rows, kK, epilogue};
                for (std::int32_t t = 2; t <= 4; ++t) {
                    const std::string at =
                        std::string(board.name) + " Q4_0 " + generic::epilogue_name(epilogue) +
                        " " + std::to_string(rows) + "x" + std::to_string(kK) + " of " +
                        std::to_string(parent_rows) + " T=" + std::to_string(t);
                    const auto co = generic::r8_coverage(p, own.weight, t).call;
                    const auto cs = generic::r8_coverage(p, parent.weight, t).call;
                    if (co != cs) ++gate_differs;
                    check(co == cs, at + ": the slice's R8 call " +
                                        std::string(generic::r8_call_refusal_name(cs)) +
                                        ", its own shape's " +
                                        std::string(generic::r8_call_refusal_name(co)));
                    const std::string o = selected(p, own.weight, t, false);
                    const std::string s = selected(p, parent.weight, t, false);
                    if (both_answer(o.starts_with("threw"), s.starts_with("threw"), at)) {
                        check(o == s, at + ": the slice takes " + s + ", its own shape " + o);
                    }
                    std::optional<generic::Route> qo, qs;
                    const bool qo_threw = throws([&] {
                        qo = generic::route_query(d, qtype, LinearPolicy::AllowA8, epilogue, rows,
                                                  kK, t, false, 0);
                    });
                    const bool qs_threw = throws([&] {
                        qs = generic::route_query(d, qtype, LinearPolicy::AllowA8, epilogue, rows,
                                                  kK, t, false, parent_rows);
                    });
                    if (both_answer(qo_threw, qs_threw, at + " query")) {
                        check(qo == qs, at + ": query of the slice " + text(qs) +
                                            ", of its own shape " + text(qo));
                    }
                }
            }
        }
    }
}

} // namespace

int main() {
    // Never read: an aligned address stands in for the payload.
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        const std::string fmt(generic::qtype_name(qtype));
        // The slice's own weight (1024 rows) and a parent of 8192 rows the slice is 1024 of: a k
        // slice of a stacked qkv, an eighth of a stacked gate/up bank under SwiGLU (512 output
        // rows reading 512 gate and 512 up rows).
        std::optional<quantized_weight::PackedWeight> own_host, parent_host;
        try {
            own_host.emplace(quantized_weight::make_patterned_weight(qtype, kOwnRows, kK, 7U));
            parent_host.emplace(
                quantized_weight::make_patterned_weight(qtype, kParentRows, kK, 11U));
        } catch (const std::invalid_argument& error) {
            std::cout << fmt << ": not packed (" << error.what() << ")\n";
            continue;
        }
        const Weight own    = own_host->device_weight(payload);
        const Weight parent = parent_host->device_weight(payload);
        if (!generic::supports(own) || !generic::supports(parent)) {
            std::cout << fmt << ": not read by the generic path\n";
            continue;
        }
        for (const generic::FactsBoard& board : generic::kFactsBoards) {
            const DeviceClass d = generic::board_device_class(board);
            const ScopedDeviceClass scoped(d);
            for (const LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
                for (const generic::Epilogue epilogue :
                     {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                    const bool swiglu       = epilogue == generic::Epilogue::SwiGlu;
                    const std::int32_t rows = swiglu ? kOwnRows / 2 : kOwnRows;
                    const generic::Problem p{qtype, policy, rows, kK, epilogue};
                    const std::string call = std::string(board.name) + " " + fmt + " p" +
                                             std::to_string(static_cast<int>(policy)) + " " +
                                             generic::epilogue_name(epilogue) + " " +
                                             std::to_string(rows) + "x" + std::to_string(kK);
                    for (const std::int32_t t : kPricedTokens) {
                        const std::string at = call + " T=" + std::to_string(t);
                        // The selector, at the speed profile and in the prefill band.
                        for (const bool prefill : {false, true}) {
                            const std::string o     = selected(p, own, t, prefill);
                            const std::string s     = selected(p, parent, t, prefill);
                            const std::string where = at + (prefill ? " prefill" : " speed");
                            if (!both_answer(o.starts_with("threw"), s.starts_with("threw"),
                                             where)) {
                                continue;
                            }
                            if (o != s) ++moved;
                            check(o == s,
                                  where + ": the slice takes " + s + ", its own shape " + o);
                        }
                        // Every family's cheapest route and its price.
                        std::vector<generic::FamilyAlternative> fo, fs;
                        const bool fo_threw =
                            throws([&] { fo = generic::family_alternatives(p, own, t, false); });
                        const bool fs_threw =
                            throws([&] { fs = generic::family_alternatives(p, parent, t, false); });
                        if (!both_answer(fo_threw, fs_threw, at + " families")) continue;
                        check(fo.size() == fs.size(), at + ": the same families");
                        for (std::size_t i = 0; i < std::min(fo.size(), fs.size()); ++i) {
                            check(fo[i].family == fs[i].family && fo[i].route == fs[i].route &&
                                      near(fo[i].seconds, fs[i].seconds),
                                  at + " family " + std::to_string(i) + ": the slice's " +
                                      text(fs[i].route) + " " +
                                      std::to_string(fs[i].seconds.value_or(-1) * 1e6) +
                                      " us, its own shape's " + text(fo[i].route) + " " +
                                      std::to_string(fo[i].seconds.value_or(-1) * 1e6) + " us");
                        }
                        // The traffic of every register-MMA schedule: the slice's rows' bytes.
                        for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                            const generic::Route r{generic::Family::RegisterMma,
                                                   static_cast<std::uint8_t>(i)};
                            const auto to = generic::route_traffic(r, own, p, t);
                            const auto ts = generic::route_traffic(r, parent, p, t);
                            check(to.has_value() == ts.has_value() &&
                                      (!to || (near(to->weight_bytes, ts->weight_bytes) &&
                                               to->activation_bytes == ts->activation_bytes)),
                                  at + ": traffic of " + generic::route_name(r));
                        }
                    }
                    // The pure query (the decision manifest replays it): a call described as a
                    // slice of its parent answers as the own shape at every width; at 1..16 and in
                    // the prefill band nothing prices, so those answers are what they were.
                    for (std::int32_t t = 1; t <= 512; t = t < 16 ? t + 1 : t * 2) {
                        for (const bool prefill : {false, true}) {
                            if (prefill && t < 32) continue;
                            const std::string at = call + " T=" + std::to_string(t) +
                                                   (prefill ? " prefill" : " speed");
                            std::optional<generic::Route> qo, qs;
                            const bool qo_threw = throws([&] {
                                qo = generic::route_query(d, qtype, policy, epilogue, rows, kK, t,
                                                          prefill, 0);
                            });
                            const bool qs_threw = throws([&] {
                                qs = generic::route_query(d, qtype, policy, epilogue, rows, kK, t,
                                                          prefill, kParentRows);
                            });
                            if (!both_answer(qo_threw, qs_threw, at + " query")) continue;
                            check(qo == qs, at + ": query of the slice " + text(qs) +
                                                ", of its own shape " + text(qo));
                        }
                    }
                    for (const std::int32_t t : kPricedTokens) {
                        const std::string at   = call + " T=" + std::to_string(t);
                        const std::size_t room = std::size_t{1} << 30;
                        std::optional<generic::RouteDescription> own_d, slice_d;
                        const bool od_threw = throws([&] {
                            own_d = generic::describe_call(d, qtype, policy, epilogue, rows, kK, t,
                                                           room, false, 0);
                        });
                        const bool sd_threw = throws([&] {
                            slice_d = generic::describe_call(d, qtype, policy, epilogue, rows, kK,
                                                             t, room, false, kParentRows);
                        });
                        if (!both_answer(od_threw, sd_threw, at + " described")) continue;
                        check(own_d.has_value() == slice_d.has_value(), at + ": described");
                        if (!own_d || !slice_d) continue;
                        check(own_d->resolved.asked == slice_d->resolved.asked &&
                                  own_d->resolved.executed == slice_d->resolved.executed,
                              at + ": the description's route " +
                                  generic::route_name(slice_d->resolved.asked) + ", its own " +
                                  generic::route_name(own_d->resolved.asked));
                        check(near(own_d->cost_seconds, slice_d->cost_seconds),
                              at + ": the description's price");
                        check(own_d->traffic.has_value() == slice_d->traffic.has_value() &&
                                  (!own_d->traffic || near(own_d->traffic->weight_bytes,
                                                           slice_d->traffic->weight_bytes)),
                              at + ": the description's traffic");
                    }
                }
            }
        }
    }
    check_row_views();
    check_padded_view_replay();
    check_r8_verify_gate();
    const bool device = !cuda_unavailable();
    check(moved == 0, std::to_string(moved) +
                          " selections where the slice takes another route than its own shape");
    std::cout << "row views: " << view_moved
              << " selections where a view takes another route than its own shape; padded views: "
              << replay_differs
              << " selections where the replay answers otherwise than the view; R8 verify gate: "
              << gate_differs << " calls where a slice answers otherwise than its own shape\n";
    // With a device every pair answers; without one the MMQ-or-register-MMA pairs need it.
    if (device) {
        check(not_compared == 0, std::to_string(not_compared) + " pairs threw with a device");
    }
    std::cout << "slice price: " << checks << " checks, " << failures << " failed, " << moved
              << " selections where the slice takes another route than its own shape, "
              << not_compared << " pairs not compared"
              << (device ? "" : " (no device: the MMQ-or-register-MMA pairs need one)") << '\n';
    if (checks == 0) {
        std::cerr << "FAIL: nothing checked\n";
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
