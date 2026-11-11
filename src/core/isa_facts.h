#pragma once

// ISA facts: what one architecture's SM does per clock, measured by the ISA probe on one card of
// it (tools/bench/isa_probe/, its rows in core/data/isa_facts.json, compiled in). The tensor pipe's
// rate and cycles per MMA, the dependent MMA latency, the INT32 / FP32 cycles per warp instruction,
// and the dependent L1 / L2 / shared / DRAM load latencies alone and under 2, 4, 8 chasing warps
// per SM. A cost law reads these instead of a constant; nothing in the engine reads them yet
// (#613).
//
// A fact the row does not carry is unknown (an empty optional), never a default number: a row of
// an older probe lacks the facts a later probe added, and the law that needs one stands aside.
// The table is keyed by the row's compute capability (the "cc" field; the key is the build's
// arch, "120a"). An architecture without a row is unknown (isa_row() gives nullopt): a row speaks
// for its own architecture only, sm_80 is not sm_86.
//
// Units are the table's and are checked on parse: clk (SM clocks), flop, ns. The mma_cycles_*
// facts are per PTX mma.sync of the row's mma_shape: on sm_70 one mma.sync.m8n8k4 is four SASS
// HMMA.884 STEPs. A law that counts SASS MMAs prices them as flops over tensor_flops_per_sm_clock,
// which holds on every architecture. This header and its source need no CUDA.

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

// A latency measured with 1, 2, 4 and 8 chasing warps on every SM (the facts "<name>_w1" ..
// "<name>_w8").
struct IsaByWarps {
    std::optional<double> w1, w2, w4, w8;
};

// One architecture's row of the ISA facts table.
struct IsaRow {
    // Where the row comes from.
    std::string arch; // the table's key: the build's arch ("86", "120a")
    int major = 0, minor = 0;
    std::string card;       // the card the probe ran on
    std::string probe;      // the probe version that measured it ("isa_probe 16")
    std::string provenance; // "Measured"
    std::string mma_shape;  // the PTX mma.sync the MMA facts are of ("hmma_16816_f32_bf16")
    int mma_flops = 0;      // flops of one such mma.sync
    // SASS MMA instructions of one such mma.sync in the probe's build (4 on sm_70); rows from
    // probe 17 on carry it.
    std::optional<int> sass_per_mma;
    int sms = 0; // the probe card's SMs

    // Tensor pipe (flop, clk).
    std::optional<double> tensor_flops_per_sm_clock; // dense, per SM and SM clock
    std::optional<double> mma_cycles_per_sm;         // throughput, one mma.sync, whole SM
    std::optional<double> mma_cycles_per_partition;  // the same per SM sub-partition (scheduler)
    // The period of a chain of mma.sync through the accumulators (on sm_70 of a group of four
    // STEPs, each chained to the same STEP of the next mma.sync).
    std::optional<double> mma_dependent_latency;

    // ALU throughput, cycles per warp instruction at one SM sub-partition (clk).
    std::optional<double> int32_cycles_per_warp_instruction;
    std::optional<double> fp32_cycles_per_warp_instruction;

    // Dependent load latencies on a chip running nothing else (clk unless _ns).
    IsaByWarps l1_latency;
    IsaByWarps l2_latency;
    std::optional<double> shared_latency;
    IsaByWarps dram_latency;            // links anywhere in a 2 GiB buffer
    IsaByWarps dram_tlbwarm_latency;    // each warp in a 512 KB range whose pages are walked
    IsaByWarps dram_tlbwarm_latency_ns; // the same in wall nanoseconds
};

// The table, parsed once from the embedded JSON. Throws std::runtime_error when the embedded text
// is malformed (a build defect: the host test parses the same text).
[[nodiscard]] std::span<const IsaRow> isa_facts_table();

// The row of compute capability major.minor, nullopt when the table has none.
[[nodiscard]] std::optional<IsaRow> isa_row(int major, int minor);

// Parses a table text (the embedded one, or a test's). Throws std::runtime_error naming the row
// and the fact: a schema other than 1, a key that is not the row's cc, a cc twice, a fact in
// another unit than its name's or not a positive number.
[[nodiscard]] std::vector<IsaRow> parse_isa_facts(std::string_view json_text);

// The names of the facts the reader takes from a row, in the table's spelling ("l1_latency_w4").
[[nodiscard]] std::span<const std::string> isa_fact_names();

} // namespace ninferno
