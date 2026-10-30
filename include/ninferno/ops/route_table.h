#pragma once

// ninferno::ops - the per-GPU-class route table (ninferno-route-table/1).
//
// One data file per GPU class steers every tuned route choice that differs between cards: the
// chip ("gb202"), or where boards of a chip tune apart the variant "gb202:170sm"; an architecture
// ("sm120") or "*" as broader fallbacks. Memory size is never part of a key. One loader reads it,
// once, and serves its two consumers:
//
// - "bands" / "prefill" steer the generic projection path (shapes no C++ table registers,
//   ops/generic/route_table.h): per exact problem (weight format, compute class, epilogue, N, K)
//   the compiled schedule per token band, plus one prefill route for every call of 17 tokens or
//   more under the prefill route profile, optionally narrowed by "prefill_bands" (per T band of
//   17 tokens or more, a route that sums in the prefill route's order: the same bytes, a better
//   schedule for that width). Written by `ninferno_route_tune`.
// - "route_as" re-asks a shape-owned Linear table (src/ops/linear/<format>/shapes) at a wider
//   width T' for the calls of a token band (ops/common/shape_routes.h): the same compiled routes,
//   crossovers moved for this card. Written by tools/route_as_table.py from
//   ninferno_linear_route_sweep_bench runs; only overrides bit-identical to the table's own choice.
//
// Precedence at dispatch: a shape-owned table serves every shape it registers (route_as may widen
// the width it is asked about); otherwise the generic path takes the table's band; otherwise the
// shape-class heuristic.
//
// Chunk invariance (prefill bytes must not depend on T for T >= 17) holds whatever a table says:
// the loader refuses a prefill route that is not token-invariant and a prefill band whose route
// sums in another order than the prefill route, bands never apply to a call of
// 17 tokens or more under the prefill profile, and a route_as band never asks a table about fewer
// tokens than the call has, so a prefill width is only ever re-asked at another prefill width of
// the same table. Refused as well: a route_as band narrower than its own widths, route_as on a
// fused epilogue, bands on "a8" (the generic path has no A8 schedule), a compute class the weight
// format does not have, duplicate problems, overlapping bands, unknown formats, compute classes,
// epilogues and route names.
//
// Loading: explicitly through load_route_table() (the engine's --route-table), or on first use
// from `<dir>/<key>.json` for the first key of the device's class chain
// that has a file ("gb202:170sm", "gb202", "sm120", "*"; a ':' in a key is '-' in the file name,
// "gb202-170sm.json", and stays ':' inside the file), where <dir> is route_table_dir(). A table
// whose class is not in the device's chain is not applied. A malformed default table is not
// applied either (every consumer answers as untuned) but never silently: route_table_status()
// names the file and the reason, and a warning goes to stderr when it is refused; an explicit
// load of a malformed file throws. Keys the
// loader does not know are ignored (evidence, notes), so one file can carry the tuning record.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninferno::ops {

struct RouteTableStatus {
    bool loaded = false;
    std::string path; // UTF-8
    std::string gpu_class;
    std::size_t entries          = 0; // problems in the file
    std::size_t generic_entries  = 0; // problems with bands or a prefill route
    std::size_t route_as_entries = 0; // problems with at least one route_as band
    std::size_t borrowed_entries = 0; // generic problems taken from similar chips' tables
    std::string borrowed_from;        // which chips lent them, nearest first ("gb206 41, gb202 9")
    // SHA-256 (lowercase hex) of the file's bytes as parsed, and of each lending table's, nearest
    // first ("gb206 <hex>, gb202 <hex>"; empty when nothing was borrowed): what the decision
    // fingerprint (runtime/engine/decision_fingerprint.h) names the route data by.
    std::string content_sha256;
    std::string lender_sha256;
    std::string message;              // why nothing is loaded, or what was loaded
};

// The current device's full class id (core/device_class.h), e.g. "gb202:170sm".
[[nodiscard]] std::string route_gpu_class();

// The directory the default table is looked up in: the one set_route_table_dir() named (the
// engine's --route-dir), else the routes/ shipped beside the running binary (route_dir_beside),
// else the compiled default (the repository's routes/, which exists only where the binary was
// built); empty when none is set. In UTF-8, as every path this API takes or reports
// (load_route_table, RouteTableStatus::path).
[[nodiscard]] std::string route_table_dir();

// Names the default lookup's directory (empty: the shipped one) and returns every device to the
// default lookup: a change of directory drops the tables found so far, and a table put in place
// explicitly (load_route_table, clear_route_table) is dropped whatever the directory, so the next
// lookup reads `dir`. The engine calls it for each construction that runs with no other engine
// alive, before an explicit --route-table loads.
void set_route_table_dir(const std::filesystem::path& dir);

// The routes/ directory shipped with a binary at `exe`: <exe dir>/routes, else
// <exe dir>/../share/ninferno/routes, the first that holds a table (a .json file); empty when
// neither does. `exe` and the result in UTF-8.
[[nodiscard]] std::string route_dir_beside(const std::string& exe);

// Replaces the active table. `path` is UTF-8 (the status reports it so). Throws std::runtime_error
// on a malformed file; a GPU-class mismatch is returned as an unloaded status and the active table
// stays.
RouteTableStatus load_route_table(const std::string& path);

// Every token count at or above `from` where the loaded route table's width-dependent choices
// change, in range: the "bands" / "prefill_bands" edges and the route_as bands' t_min and "as"
// widths (a call re-asked at T' runs the shape table's T' band, so its own edges matter there).
// Below the first prefill band's t_min (17) the prefill route is constant. The engine's kernel
// warm-up uses this to run one width per width class instead of every width 1..chunk; the C++
// choosers' own edges (the shape tables and the fused-op plans) are enumerated beside it
// (src/runtime/engine/warmup_widths.cpp). Only width-dependent choice points enter; a table
// without bands changes nothing at any width.
[[nodiscard]] std::vector<std::int32_t> route_table_breakpoints(std::int32_t from);

// Parses and validates a table file (`path` in UTF-8) without applying it (same refusals as
// load_route_table).
[[nodiscard]] RouteTableStatus inspect_route_table(const std::string& path);

// Drops the active table and skips the default lookup: every consumer answers as untuned.
void clear_route_table();

[[nodiscard]] RouteTableStatus route_table_status();

// Whether the table at `path` (UTF-8) is this device's explicit table (load_route_table), without
// running the default lookup (which refuses a broken directory the explicit table stands in for).
[[nodiscard]] bool route_table_loaded(const std::string& path);

} // namespace ninferno::ops
