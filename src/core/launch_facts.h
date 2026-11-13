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
// engine reads this table yet: the speed models keep core/cost_constants.h's kLaunchNode (the Ada
// census record) until a DECISIONS entry moves them (PR-8e), and tests/test_launch_node.cpp checks
// the Ada row against that census.
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
// SMs, 1024) as a graph-chain node, so floor_s[i] - floor_s[0] is the CTA dispatch ramp of that
// grid on the probe card. A ramp speaks for its card's SM count; a law that prices another card's
// grid reads it per wave or per CTA, which is the law's business, not this table's.

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
    std::string probe; // "launch_probe 1"
    int sms = 0;       // that card's SMs
    std::optional<double> node_s, node_pdl_s, stream_s;
    std::array<int, 4> floor_ctas{};
    std::array<std::optional<double>, 4> floor_s{};
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
    r.floor_ctas = {1, l.sms, 4 * l.sms, 1024};
    r.floor_s    = {seconds(l.floor_ns_1), seconds(l.floor_ns_sms), seconds(l.floor_ns_4sms),
                    seconds(l.floor_ns_1024)};
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

} // namespace ninferno
