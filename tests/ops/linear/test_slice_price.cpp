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
#include "ops/generic/generic_gemm.h"
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
    const bool device = !cuda_unavailable();
    check(moved == 0, std::to_string(moved) +
                          " selections where the slice takes another route than its own shape");
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
