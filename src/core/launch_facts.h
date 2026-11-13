#pragma once

// The launch facts per architecture (#613 PR-8b): what one launch costs, as seconds, from the
// launch entry of the architecture's ISA row (core/isa_facts.h; measured by
// tools/bench/isa_probe/launch_probe.cu on one card of the architecture, owner decision 52: one
// card is enough for an ISA fact). No second table: the launch probe's runs are merged into the
// ISA rows by `isa_probe.py regenerate`.
//
// A fact the row does not carry is unknown (nullopt), never a default number: an architecture
// without a launch run, and a kind its run did not measure (node_pdl below sm_90 where the driver
// refused the attribute or kept the host until earlier work had ended), stand aside. Nothing in the
// engine decides by this table yet: the speed models keep core/cost_constants.h's kLaunchNode (the
// Ada census record) until a DECISIONS entry moves them (PR-8e), tests/test_launch_node.cpp checks
// the Ada row against that census, and the selector's completeness terms (kernel_roofline.h, #613
// PR-8a) read the node and the ramp below with their switch off.
//
// What each kind is (launch_probe.cu's head): the cost of one more launch in a long back-to-back
// series of empty kernels, so the series' fixed parts cancel and the CUDA event's 1.024 us tick is
// spread over 255 or 1023 launches.
//   GraphNode     an empty one-warp CTA as a node of a captured graph's linear chain: graph
//                 replay, how the engine runs decode;
//   GraphNodePdl  the same node with the programmatic stream serialization attribute, waiting on
//                 the node before it at its start and releasing the next at its end. A launch of
//                 this kind exists from sm_90 on only (core/pdl.cuh sets the attribute there):
//                 launch_seconds gives nullopt below sm_90 whatever the row holds, and the row's
//                 node_pdl_s keeps the probe's reading of the attribute alone for the record;
//   Stream        the one-warp kernel launched on a stream from a backlog: the GPU's own cost per
//                 stream launch (the host's submit rate is the machine's CPU, not a fact).
// The floors: an empty grid of floor_ctas[i] CTAs of 128 threads (1, the probe card's SMs, 4 x its
// SMs, 1024, 4096; the last unknown on a row launch_probe 1 measured, #886) as a graph-chain node,
// so floor_s[i] - floor_s[0] is the CTA dispatch ramp of that grid on the probe card. It is not
// rounds of resident CTAs (the RTX 5070 Ti's equals the RTX 5090's at 1024 CTAs), and it is not one
// figure a CTA either: the slope between the points falls with the CTA count on Ada, Ampere and
// Turing (the RTX 4090: 1.13, 0.81, 0.44 ns a CTA), rises on gb20x (0.23, 0.40, 0.43) and is flat
// only on Volta (1.5; #870). A law that prices another grid reads the points as they are (between
// them, not past the last known one without saying so); that is the law's business, not this
// table's.

#include "core/isa_facts.h"

#include <array>
#include <optional>
#include <string>

namespace ninferno {

enum class LaunchKind { GraphNode, GraphNodePdl, Stream };

struct LaunchRow {
    std::string arch; // the ISA row's key ("120a")
    int major = 0, minor = 0;
    std::string card;  // the card the launch probe ran on
    std::string probe; // "launch_probe 2" (a launch_probe 1 row has no 4096-CTA floor)
    int sms = 0;       // that card's SMs
    std::optional<double> node_s, node_pdl_s, stream_s;
    std::array<int, 5> floor_ctas{};
    std::array<std::optional<double>, 5> floor_s{};
};

namespace launch_facts_detail {
inline std::optional<double> seconds(const std::optional<double>& ns) {
    return ns ? std::optional<double>(*ns * 1e-9) : std::nullopt;
}
} // namespace launch_facts_detail

// The launch row of compute capability major.minor; nullopt when the architecture has no ISA row
// or its row no launch entry.
[[nodiscard]] inline std::optional<LaunchRow> launch_row(int major, int minor) {
    const std::optional<IsaRow> isa = isa_row(major, minor);
    if (!isa || !isa->launch) return std::nullopt;
    const IsaLaunch& l = *isa->launch;
    using launch_facts_detail::seconds;
    LaunchRow r;
    r.arch       = isa->arch;
    r.major      = isa->major;
    r.minor      = isa->minor;
    r.card       = l.card;
    r.probe      = l.probe;
    r.sms        = l.sms;
    r.node_s     = seconds(l.node_ns);
    r.node_pdl_s = seconds(l.node_pdl_ns);
    r.stream_s   = seconds(l.stream_ns);
    r.floor_ctas = {1, l.sms, 4 * l.sms, 1024, 4096};
    r.floor_s    = {seconds(l.floor_ns_1), seconds(l.floor_ns_sms), seconds(l.floor_ns_4sms),
                    seconds(l.floor_ns_1024), seconds(l.floor_ns_4096)};
    return r;
}

// One launch of `kind` on compute capability major.minor, in seconds; nullopt when unknown, and
// for GraphNodePdl below sm_90, where the engine launches no such node.
[[nodiscard]] inline std::optional<double> launch_seconds(int major, int minor, LaunchKind kind) {
    const std::optional<LaunchRow> r = launch_row(major, minor);
    if (!r) return std::nullopt;
    if (kind == LaunchKind::GraphNodePdl && major < 9) return std::nullopt;
    switch (kind) {
    case LaunchKind::GraphNode:
        return r->node_s;
    case LaunchKind::GraphNodePdl:
        return r->node_pdl_s;
    case LaunchKind::Stream:
        return r->stream_s;
    }
    return std::nullopt;
}

// The CTA dispatch ramp's measured points (#613 PR-8a): the floors' CTA counts (1, the probe
// card's SMs, 4 x its SMs, 1024, 4096) and their durations in seconds, as the launch entry holds
// them (a launch_probe 1 row has no 4096 point).
struct RampPoints {
    std::array<double, 5> ctas{};
    std::array<std::optional<double>, 5> seconds{};
};

[[nodiscard]] inline RampPoints ramp_points(const IsaLaunch& l) {
    using launch_facts_detail::seconds;
    RampPoints p;
    p.ctas    = {1.0, static_cast<double>(l.sms), 4.0 * l.sms, 1024.0, 4096.0};
    p.seconds = {seconds(l.floor_ns_1), seconds(l.floor_ns_sms), seconds(l.floor_ns_4sms),
                 seconds(l.floor_ns_1024), seconds(l.floor_ns_4096)};
    return p;
}

// The dispatch ramp of a grid: its seconds beyond a one-CTA grid's, and whether they were
// extrapolated past the last measured point.
struct Ramp {
    double seconds    = 0;
    bool extrapolated = false;
};

// The ramp of a grid of `ctas` CTAs from the measured points: piecewise linear between them (each
// point reproduced exactly: floor_s[i] - floor_s[0]), 0 at one CTA and below; past the last point
// (4096 CTAs, 1024 on a launch_probe 1 row) the last segment's slope carries on and the answer says
// it is extrapolated. The
// slopes are not one figure a generation (#870: the RTX 4090's fall 1.13, 0.81, 0.44 ns a CTA,
// gb20x's rise 0.23, 0.40, 0.43), so no single rate stands in for the points. A point the entry
// lacks is left out; nullopt without the one-CTA floor and at least one more point.
[[nodiscard]] inline std::optional<Ramp> ramp_from(const RampPoints& p, double ctas) {
    std::array<double, 5> x{}, y{};
    int m = 0;
    for (int i = 0; i < 5; ++i) {
        // The one-CTA floor first; then the points in increasing CTA count (a probe card with 256
        // SMs or more would put 4 x SMs at or past 1024: such a point is left out).
        if (!p.seconds[i] || (m > 0 && !(p.ctas[i] > x[m - 1]))) continue;
        if (m == 0 && i != 0) return std::nullopt;
        x[m] = p.ctas[i];
        y[m] = *p.seconds[i];
        ++m;
    }
    if (m < 2) return std::nullopt;
    if (ctas <= x[0]) return Ramp{};
    for (int i = 0; i + 1 < m; ++i) {
        if (ctas > x[i + 1]) continue;
        if (ctas == x[i + 1]) return Ramp{y[i + 1] - y[0], false};
        return Ramp{y[i] + (ctas - x[i]) * ((y[i + 1] - y[i]) / (x[i + 1] - x[i])) - y[0], false};
    }
    const int l = m - 1;
    return Ramp{y[l] + (ctas - x[l]) * ((y[l] - y[l - 1]) / (x[l] - x[l - 1])) - y[0], true};
}

// The dispatch ramp of a grid of `ctas` CTAs on compute capability major.minor (ramp_from on its
// launch entry's points); nullopt where the architecture has no launch entry or too few floors.
[[nodiscard]] inline std::optional<Ramp> ramp_seconds(int major, int minor, double ctas) {
    const std::optional<IsaRow> isa = isa_row(major, minor);
    if (!isa || !isa->launch) return std::nullopt;
    return ramp_from(ramp_points(*isa->launch), ctas);
}

} // namespace ninferno
