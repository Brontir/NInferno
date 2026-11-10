// A route's description (#298, describe_route / describe_call), host only, on a substituted RTX
// 4090 class (one board per process: the selector caches what it reads of the device). The
// description is assembled from the functions the launch and the selector ask, so each field must
// agree with them: the choice with route_query, the resolved route with resolved_route_query, the
// workspace with route_workspace_bytes, the tile with route_tile, the traffic with route_traffic on
// an independently built weight (the patterned fixture, not the query's described one). A pinned
// call pins the rendered line.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool near(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b)); }

} // namespace

int main() {
    const DeviceClass device =
        synthetic_device_class(0x2684, 8, 9, 128, 72 * kMiB, 24 * 1024 * kMiB);
    // Never read: an aligned address stands in for the payload.
    void* const payload      = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    constexpr std::int32_t n = 4096, k = 2048;

    // A pinned call: the register MMA's first schedule, a Q8 G32 weight, 100 tokens, no workspace.
    {
        const auto host = quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, n, k, 7U, {});
        const Weight w  = host.device_weight(payload);
        const generic::Route route{generic::Family::RegisterMma, 0};
        const auto d =
            generic::describe_route(device, route, QType::Q8_G32_FP16, LinearPolicy::A16Only,
                                    generic::Epilogue::Store, n, k, 100, std::nullopt);
        check(d.has_value(), "pinned call is described");
        if (d) {
            const auto s = generic::rmma_schedule_info(0);
            check(d->resolved.asked == route && d->resolved.executed == route &&
                      d->resolved.reason == generic::Demotion::None,
                  "pinned call runs the route asked");
            check(d->workspace_bytes == 0 && !d->prepared, "A16 register MMA: no workspace");
            check(d->tile && d->tile->out_rows == 16 * s.m_tiles * s.warps_m &&
                      d->tile->tokens == 8 * s.n_tiles && d->tile->k_split == s.cluster,
                  "pinned tile is the schedule's declared geometry");
            const generic::Problem p{QType::Q8_G32_FP16, LinearPolicy::A16Only, n, k,
                                     generic::Epilogue::Store};
            const auto tr = generic::route_traffic(route, w, p, 100);
            check(tr && d->traffic && near(d->traffic->weight_bytes, tr->weight_bytes) &&
                      near(d->traffic->activation_bytes, tr->activation_bytes) &&
                      near(d->traffic->output_bytes, tr->output_bytes),
                  "traffic agrees with route_traffic on an independently built weight");
            check(d->traffic && near(d->traffic->output_bytes, double{n} * 100 * 2),
                  "the output is written once in bf16");
            const std::string text = generic::describe_text(*d);
            const std::string head = "asked=" + generic::route_name(route) +
                                     " runs=" + generic::route_name(route) +
                                     " workspace=0 tile=" + std::to_string(d->tile->out_rows) +
                                     "x" + std::to_string(d->tile->tokens);
            check(text.rfind(head, 0) == 0, "pinned line starts: " + text);
            // The whole line, every key literal: a format change anywhere fails.
            std::string full = "asked=" + generic::route_name(route) +
                               " runs=" + generic::route_name(route) +
                               " workspace=0 tile=" + std::to_string(s.m_tiles * s.warps_m * 16) +
                               "x" + std::to_string(8 * s.n_tiles);
            if (s.cluster > 1) full += " ksplit=" + std::to_string(s.cluster);
            full += " weight_bytes=" +
                    std::to_string(static_cast<std::uint64_t>(d->traffic->weight_bytes)) +
                    " activation_bytes=" +
                    std::to_string(static_cast<std::uint64_t>(d->traffic->activation_bytes)) +
                    " output_bytes=" + std::to_string(n * 100 * 2);
            if (d->cost_seconds) {
                char us[64];
                std::snprintf(us, sizeof(us), "%.3f", *d->cost_seconds * 1e6);
                full += std::string(" cost_us=") + us;
            }
            check(text == full, "pinned full line: " + text + " vs " + full);
            check(text.find("weight_bytes=") != std::string::npos &&
                      text.find("output_bytes=" + std::to_string(n * 100 * 2)) != std::string::npos,
                  "pinned line carries the traffic: " + text);
        }
        // A route the format refuses is not described.
        check(!generic::describe_route(device, generic::Route{generic::Family::W4a4Mma, 0},
                                       QType::Q8_G32_FP16, LinearPolicy::A16Only,
                                       generic::Epilogue::Store, n, k, 100, std::nullopt),
              "a route route_admits refuses has no description");
    }

    // The selected call: every field agrees with the function the launch or selector asks.
    for (const QType qtype : {QType::Q8_G32_FP16, QType::GGML_Q4_K, QType::GGML_Q6_K}) {
        const auto host = quantized_weight::make_patterned_weight(qtype, n, k, 7U, {});
        const Weight w  = host.device_weight(payload);
        for (const LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
            for (const auto epilogue : {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                for (const std::int32_t t : {1, 4, 16, 64, 512}) {
                    const std::string at =
                        std::string(generic::qtype_name(qtype)) +
                        (policy == LinearPolicy::A16Only ? " a16 " : " a8 ") +
                        (epilogue == generic::Epilogue::SwiGlu ? "swiglu" : "store") +
                        " T=" + std::to_string(t);
                    const std::size_t free = std::size_t{1} << 30;
                    const auto d =
                        generic::describe_call(device, qtype, policy, epilogue, n, k, t, free);
                    const auto rq = generic::resolved_route_query(device, qtype, policy, epilogue,
                                                                  n, k, t, free);
                    check(d.has_value() == rq.has_value(), at + ": described where resolved");
                    if (!d || !rq) continue;
                    check(d->resolved.asked == rq->asked && d->resolved.executed == rq->executed &&
                              d->resolved.r8_prequantized == rq->r8_prequantized &&
                              d->resolved.reason == rq->reason,
                          at + ": resolved route is resolved_route_query's");
                    const auto q = generic::route_query(device, qtype, policy, epilogue, n, k, t);
                    check(q && *q == d->resolved.asked, at + ": the route asked is route_query's");
                    const generic::Route ran = d->resolved.executed;
                    if (ran.family != generic::Family::GgmlMmq &&
                        ran.family != generic::Family::GgmlVec &&
                        ran.family != generic::Family::R8Mma) {
                        check(d->workspace_bytes == generic::route_workspace_bytes(ran, k, t),
                              at + ": workspace is route_workspace_bytes'");
                    }
                    const auto tile = generic::route_tile(ran, epilogue, t);
                    check(d->tile.has_value() == tile.has_value() &&
                              (!tile || (d->tile->out_rows == tile->out_rows &&
                                         d->tile->tokens == tile->tokens &&
                                         d->tile->k_split == tile->k_split)),
                          at + ": tile is route_tile's");
                    // The traffic on the independently built weight: SwiGLU's has 2 n rows.
                    if (epilogue == generic::Epilogue::Store) {
                        const generic::Problem p{qtype, policy, n, k, epilogue};
                        const auto tr = generic::route_traffic(ran, w, p, t);
                        check(
                            d->traffic.has_value() == tr.has_value() &&
                                (!tr || (near(d->traffic->weight_bytes, tr->weight_bytes) &&
                                         near(d->traffic->activation_bytes, tr->activation_bytes) &&
                                         near(d->traffic->output_bytes, tr->output_bytes))),
                            at + ": traffic is route_traffic's");
                    }
                    check(d->prepared.has_value() ==
                              generic::prepared_rule(d->resolved, w).has_value(),
                          at + ": prepared rule is prepared_rule's");
                    check(!generic::describe_text(*d).empty(), at + ": renders");
                }
            }
        }
    }

    // A slice of a fused parent (#298): the traffic is of the call's rows, not the parent's, so a
    // call reading n of 2n stored rows describes what the same call on its own n-row weight does.
    for (const QType qtype : {QType::Q8_G32_FP16, QType::GGML_Q4_K}) {
        const generic::Route route{generic::Family::RegisterMma, 0};
        const auto own = generic::describe_route(device, route, qtype, LinearPolicy::A16Only,
                                                 generic::Epilogue::Store, n, k, 64, std::nullopt);
        const auto slice =
            generic::describe_route(device, route, qtype, LinearPolicy::A16Only,
                                    generic::Epilogue::Store, n, k, 64, std::nullopt, 2 * n);
        const std::string at = std::string("slice ") + std::string(generic::qtype_name(qtype));
        check(own && slice && own->traffic && slice->traffic &&
                  near(own->traffic->weight_bytes, slice->traffic->weight_bytes) &&
                  near(own->traffic->activation_bytes, slice->traffic->activation_bytes),
              at + ": traffic of a slice is the call's rows'");
        check(own && slice && own->cost_seconds.has_value() == slice->cost_seconds.has_value() &&
                  (!own->cost_seconds || near(*own->cost_seconds, *slice->cost_seconds)),
              at + ": price of a slice is the call's rows'");
    }

    // R8 at verify widths takes workspace only in its pre-quantized form: without any workspace
    // (not pre-quantized) the description reports none; with it, the planes it reads.
    for (const QType qtype : {QType::GGML_Q4_K, QType::GGML_Q6_K, QType::Q8_G32_FP16}) {
        for (const std::int32_t t : {1, 2, 4, 8}) {
            const std::string at = std::string("R8 ") + std::string(generic::qtype_name(qtype)) +
                                   " T=" + std::to_string(t);
            const generic::Route r8{generic::Family::R8Mma, 0};
            const auto bare =
                generic::describe_route(device, r8, qtype, LinearPolicy::AllowA8,
                                        generic::Epilogue::Store, n, k, t, std::nullopt);
            if (bare && bare->resolved.executed.family == generic::Family::R8Mma) {
                check(!bare->resolved.r8_prequantized && bare->workspace_bytes == 0,
                      at + ": no workspace, none reported");
            }
            const auto roomy =
                generic::describe_route(device, r8, qtype, LinearPolicy::AllowA8,
                                        generic::Epilogue::Store, n, k, t, std::size_t{1} << 30);
            if (roomy && roomy->resolved.executed.family == generic::Family::R8Mma) {
                check(roomy->workspace_bytes == (roomy->resolved.r8_prequantized
                                                     ? generic::route_workspace_bytes(r8, k, t)
                                                     : std::size_t{0}),
                      at + ": workspace only in the pre-quantized form");
            }
        }
    }

    if (failures == 0) std::cout << "route description: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
