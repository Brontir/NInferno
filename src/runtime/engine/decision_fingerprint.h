#pragma once

// The decision fingerprint, v1 (docs/maintainer/deterministic-decisions.md, Fingerprint): a hash
// of every input the engine's decisions read, so that two runs with the same fingerprint are known
// to have been asked the same thing on the same kind of machine.
//
// The inputs are six sections of key -> value text:
//   facts    the device's facts (DeviceClass, minus its name and ordinal; of its clocks the
//            machine facts the decisions read, and the driver's only where they fed a fact), the
//            build's feature row and the driver version;
//   build    the release, its commit, the architecture it compiled for, PTX, NVFP4, the CUDA
//            runtime it was built against;
//   options  the options as the caller asked for them (before normalize_engine_options), the
//            decode pacing and the maintainer options among them, the artifact's id and its
//            prefill signature;
//   sources  the data files decisions were read from, by content (route table and its lenders,
//            dense- and MoE-tune files, context-cost presets, speculation profile, drafter
//            registry), and a memory-driven --spec auto fallback ("spec_fit_fallback");
//   plan     what startup resolved: KV and state storage, speculative graph windows, route
//            breakpoints and table, the decode steps per round, and the plan ladder's rung the
//            plan runs at (its index and the ladder's length) with the rung's sizes (context,
//            chunk, KV tokens, cached states);
//   taint    the pillar violations present (a maintainer option, the user's tune cache, floor
//            calibration, no roofline, borrowed or foreign route data, a local spec-tune, a
//            --spec auto fallback the free memory forced).
// and one section that is not part of the fingerprint:
//   observed what startup measured beside the plan: the rung the selection picked before the
//            kernel warm-up and the tier it admitted it at (rung_selected, rung_reserve), what it
//            sized from the Host memory (the Host tier), and the driver's clocks where they fed
//            no fact (MachineFacts::driver_read), so no decision reads them; they enter the
//            manifest only.
//
// The canonical text is "schema\tninferno-fingerprint/1\n" followed by one
// "<section>.<key>\t<value>\n" line per entry in byte order (tab, newline, carriage return and
// backslash escaped as \t, \n, \r and \\), without the observed section. The fingerprint is "nf1-"
// and the first 16 hex digits of its SHA-256; the manifest is "nm1-" and the first 16 hex digits
// of the SHA-256 of the schema line, the plan lines and the observed lines, so a plan change moves
// both, an option change only the fingerprint and an observed size only the manifest.
//
// Canonical values: integers in decimal, booleans 0/1, enumerations by name, floating-point
// numbers as the hex of their bits, sequences joined by commas, an unset optional as "unset".
//
// Everything here is pure: no CUDA, no environment, no files. The Engine passes the facts in
// (src/runtime/engine/engine.cpp), and a CPU can recompute a fingerprint from a canonical text.

#include "core/device_class.h"
#include "ninferno/types.h"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ninferno::runtime {

inline constexpr std::string_view kDecisionFingerprintSchema = "ninferno-fingerprint/1";

using DecisionSection = std::map<std::string, std::string>;

struct DecisionInputs {
    DecisionSection facts;
    DecisionSection build;
    DecisionSection options;
    DecisionSection sources;
    DecisionSection plan;
    DecisionSection taint;
    DecisionSection observed; // in the manifest, not the fingerprint
};

struct DecisionFingerprint {
    std::string value;     // "nf1-" + 16 hex digits
    std::string manifest;  // "nm1-" + 16 hex digits
    std::string canonical; // the canonical text value hashes
    std::string observed;  // the observed lines (manifest only)
};

[[nodiscard]] std::string canonical_text(const DecisionInputs& inputs);
[[nodiscard]] DecisionFingerprint decision_fingerprint(const DecisionInputs& inputs);

// The host's facts as the Engine reads them: logical CPU count, the RAM the process may use
// (0: unknown) and the process's CPU bandwidth quota (host_cpu_quota.h; 0 and 0: none), which the
// decode round's default step count follows (round_pacing.h).
struct HostFacts {
    unsigned logical_cores      = 0;
    std::uint64_t ram_bytes     = 0;
    std::uint64_t cpu_quota_us  = 0;
    std::uint64_t cpu_period_us = 0;
};

// The device's facts (no name, no ordinal), `build` (build_features()), the driver version
// (cudaDriverGetVersion; 0 when unknown) and the host's: "host.cpu_cores", "host.cpu_quota"
// ("QUOTA/PERIOD" in microseconds, or "none") and "host.ram_gib_class", the largest power of two
// of GiB at or below the RAM (0 below 1 GiB or unknown), so that a few MiB the kernel keeps do not
// move it.
// Of the clocks, "machine.*" are the facts the decisions read (DeviceClass::facts: source,
// reference and memory clock, bus width, share, mma_slow); "driver.*" (the driver's clocks and
// bus width) enter only when they fed a fact (MachineFacts::driver_read: an unknown id, a laptop
// row without the bottom of its range).
[[nodiscard]] DecisionSection facts_of(const DeviceClass& device, const DeviceFeatures& build,
                                       int driver_version, const HostFacts& host);

// This binary's build, from its compile definitions.
[[nodiscard]] DecisionSection build_of();

// The options as asked (a copy taken before normalize_engine_options), with the artifact's id
// (hex), its prefill signature (LoadSummary::prefill_signature) and a separate drafter's artifact
// id (empty: none). Paths, the device ordinal, queue limits, media budgets, observers and
// SpeculativeOptions::decision_sources do not enter (the drafter enters by its id, the sources
// in their own section): a maintainer option naming a file or directory (--route-table,
// --route-dir, --tune-cache) enters as "set" or "unset", what it holds by content in sources.
[[nodiscard]] DecisionSection options_of(const EngineOptions& requested,
                                         std::string_view artifact_id,
                                         std::string_view prefill_signature,
                                         std::string_view drafter_id);

// What startup resolved without the free memory: `resolved` is Engine::options() (after the plan),
// `memory` the MemorySummary (with the plan ladder's rung, MemorySummary::plan),
// `route_breakpoints` ops::route_table_breakpoints(17), `route_table` the effective route table's
// content (its hash and its lenders'; empty: none) and `decode_steps_per_round` the steps an
// ordinary decode round runs (LoadSummary).
[[nodiscard]] DecisionSection plan_of(const EngineOptions& resolved, const MemorySummary& memory,
                                      std::span<const std::int32_t> route_breakpoints,
                                      std::string_view route_table,
                                      std::uint32_t decode_steps_per_round);

// What startup observed beside the plan (the observed section): the rung the selection picked
// before the kernel warm-up and the tier it admitted the rung at (rung_selected, rung_reserve: the
// warm-up's measurement, not a decision of its own), and what it sized from the Host memory
// (host_state_slots, host_kv_bytes).
[[nodiscard]] DecisionSection observed_of(const MemorySummary& memory);

// The device's observed values: the driver's clocks and bus width ("driver.*") when they fed no
// fact (!MachineFacts::driver_read), so no decision reads them; empty otherwise (facts_of holds
// them).
[[nodiscard]] DecisionSection observed_of(const DeviceClass& device);

// The environment variables the products read. None steers a decision (an option does:
// docs/maintainer/deterministic-decisions.md); the inventory test fails on a variable read in
// src/ or apps/ that is not on this list.
[[nodiscard]] std::span<const std::string_view> non_decision_environment_variables();

// The taint of the other sections: "maintainer" (the maintainer options given, by name:
// route_table, route_dir, dense_tune, moe_tune, tune_cache while a tuner is on,
// debug_prefill_cuts, debug_poison, debug_poison_skip, and drafter_registry for a registry file
// read in place of the compiled-in one), "user_tune_cache" (which tuners: "dense", "moe"),
// "clocks" (no roofline: a machine clock or bus width of 0), "borrowed_routes" (route entries lent
// by another chip's table), "foreign_route_table" (a table keyed by a class not in the device's
// own chain), "local_spec_tune", "spec_fit_fallback" (the --spec auto fallback a capacity refusal
// forced, sources.spec_fit_fallback).
[[nodiscard]] DecisionSection taint_of(const DecisionInputs& inputs);

} // namespace ninferno::runtime
