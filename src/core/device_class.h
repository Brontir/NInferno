#pragma once

// The device class: every hardware quantity a heuristic, a launch shape or a data table depends
// on, queried once per CUDA ordinal from the runtime. Code reads these instead of writing literal
// SM counts, cache sizes or memory sizes; per-GPU-class data (route tables, tuning presets) is
// keyed by id().
//
// Each ordinal is queried once and cached for the life of the process; a lookup is a
// cudaGetDevice() and an index, cheap enough for a launch path. The cache is per ordinal, so a
// multi-GPU host sees each card's own facts as long as the caller's device is current
// (DeviceContext binds it, and Ops run on the bound device). An ordinal is this process's name for
// a board only; what outlives the process or chooses among GPUs reads DeviceClass::identity
// (core/device_identity.h).
//
// Capability gates compare the feature set, never a compute-capability inequality: sm_120 is
// numerically above sm_100 and lacks its tensor-core instructions, and "== 120" misses sm_121.
// A feature names the instruction family the kernels actually issue.
//
// GPU identity. The class of a GPU is its CHIP ("gb202": RTX 5090 and RTX PRO 6000; "gb203": RTX
// 5080 and 5070 Ti; "ad102", "ga102", ...), found from the PCI product id through a data table
// (device_class.cpp), else from the architecture and the SM count. Boards of one chip can enable
// different SM counts (the 5090 170, the PRO 6000 188), so the full id adds that as a variant,
// "gb202:170sm". Per-GPU-class data (speculation profiles in the drafter registry, route tables,
// bench roofs) is keyed by chip, by the variant only where boards of a chip truly tune apart, and
// looked up along class_chain(): "gb202:170sm" -> "gb202" -> "sm120" (the architecture) -> "*".
// Memory size is never part of an id: the same RTX 5090 reports 31.36 GiB under a native Linux
// driver and 31.84 GiB under WDDM. A choice that depends on memory reads global_bytes (or free
// memory) and compares ranges.

#include "core/machine_facts.h"
#include "core/device_identity.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

struct DeviceFeatures {
    bool mma_bf16 = false;     // mma.sync m16n8k16 BF16/FP16 -> FP32 and m16n8k32 S8 -> S32
                               // (composed from Turing's m16n8k8 / m8n8k16 in the sm_75 build)
    bool mma_bf16_native = false; // BF16 operands in the MMA itself (sm_80 on); Turing's build
                                  // rounds them to f16 and issues two k8 MMAs per k16
    bool mma_tensor = false;      // mma.sync on tensor cores (sm_75 on); Volta's build computes the
                                  // same fragments in FP32 FMAs and dp4a (ops/common/mma.cuh)
    bool mma_fp8      = false; // mma.sync m16n8k32 E4M3 -> FP32, the un-prefixed form (sm_89 on)
    bool mma_f8f6f4   = false; // mma.sync .kind::f8f6f4 (FP8 E4M3 operands)
    bool mma_mxf4nvf4 = false; // mma.sync .kind::mxf4nvf4 block-scaled NVFP4
    bool tma          = false; // cp.async.bulk with mbarrier transaction counts
    bool pdl          = false; // griddepcontrol (programmatic dependent launch)
    bool clusters     = false; // thread-block clusters and distributed shared memory (sm_90 on)
    bool mma_slow = false; // mma.sync without tensor cores: TU116 / TU117 (GTX 16xx, T400..T1200)
                           // run it on their FP16 units, as do the GTX 16 boards on TU106;
                           // board_mma_slow, never the compute capability
};

// Whether a chip of this compute capability has mma.sync but no tensor cores (TU116, TU117): the
// fact of a chip, for a board the PCI-ID table lacks (board_mma_slow) and for the chips themselves
// (the similar-chip lists, a measured table's board keyed by its chip).
[[nodiscard]] bool chip_mma_slow(int major, int minor, std::string_view chip) noexcept;

// The board's mma_slow, from one source: the PCI-ID table's row where the table knows the id
// (`facts.sku` set, the row of the device's architecture), else the chip's (chip_mma_slow of the
// chip the id's block names, or the SM-count guess for an unknown id). The table marks every
// TU116 / TU117 board and the GTX 16 boards on TU106, whose chip has tensor cores.
[[nodiscard]] bool board_mma_slow(const MachineFacts& facts, int major, int minor,
                                  std::string_view chip) noexcept;

// The architecture's feature set by compute capability: a table, not an ordering.
[[nodiscard]] DeviceFeatures device_features_for(int major, int minor) noexcept;

// Every feature in `required` is present in `available`.
[[nodiscard]] bool has_features(const DeviceFeatures& available,
                                const DeviceFeatures& required) noexcept;

// The features both sets have.
[[nodiscard]] DeviceFeatures common_features(const DeviceFeatures& a,
                                             const DeviceFeatures& b) noexcept;

// What this build's device code carries: the row of the architecture it was compiled for
// (NINFERNO_BUILD_ARCH), without the block-scaled kinds unless the target was architecture- or
// family-specific ("120a", "120f": a plain sm_120 build compiles them out). A GPU newer than the
// build runs the build's code, its PTX compiled by the driver (NINFERNO_CUDA_EMBED_PTX), so
// DeviceClass::features is the device's row and this one in common: an sm_89 build on an RTX
// 5090 takes no PDL or TMA path, whose kernels it compiled empty.
[[nodiscard]] DeviceFeatures build_features() noexcept;

struct DeviceClass {
    int device                       = 0;
    int major                        = 0;
    int minor                        = 0;
    int sms                          = 0; // multiprocessors
    std::size_t l2_bytes             = 0;
    std::size_t smem_per_block_optin = 0;
    std::size_t smem_per_sm          = 0;
    std::size_t global_bytes         = 0;
    std::string name; // informational only; never a key
    std::uint32_t pci_device_id = 0; // PCI product id (0x2B85: RTX 5090); 0 when unreadable
    std::string chip;                // "gb202"; the architecture ("sm120") when the chip is unknown
    bool chip_from_pci = false;      // chip named by the PCI id's block, not guessed by SM count
    DeviceIdentity identity; // which board (PCI address, UUID): logs and ordering, never a key

    int max_threads_per_sm              = 0;
    int max_blocks_per_sm               = 0;
    int registers_per_sm                = 0;
    std::size_t persisting_l2_max_bytes = 0;
    // What the driver reports: informational, and the input of `facts`. No decision reads these
    // (docs/maintainer/deterministic-decisions.md): they are 0 on MIG and some WSL drivers and
    // carry a board's factory overclock.
    int memory_bus_bits                 = 0;
    int memory_clock_khz                = 0;
    int sm_clock_khz                    = 0;
    DeviceFeatures features;
    // The board's canonical facts (core/machine_facts.h): the PCI-ID table's row, else the driver
    // values above on the 2^(1/16) grid. What the cost model, the board distance and the
    // speculation load model read.
    MachineFacts facts;

    // "<chip>:<sms>sm", for example "gb202:170sm" (RTX 5090) and "gb203:84sm" (RTX 5080).
    [[nodiscard]] std::string id() const;
    // "sm<major><minor>", for example "sm120".
    [[nodiscard]] std::string arch() const;
    // The keys a per-GPU-class table is searched by, closest first: id(), chip, arch(), "*".
    [[nodiscard]] std::vector<std::string> class_chain() const;
    // The other known chips of this architecture, most similar first (gpu_similar_chips): where
    // per-class data is borrowed from when no key of class_chain() has any.
    [[nodiscard]] std::vector<std::string> similar_chips() const;
    // class_chain() widened by similarity, for data that carries across architectures (a
    // speculation profile measures the model's acceptance and the card's relative verify cost):
    // gpu_similarity_chain.
    [[nodiscard]] std::vector<std::string> similarity_chain() const;

    [[nodiscard]] int compute_capability() const noexcept { return major * 10 + minor; }
    // CTAs the whole device holds at `ctas_per_sm` resident CTAs per SM: one wave.
    [[nodiscard]] std::int64_t wave(std::int64_t ctas_per_sm = 1) const noexcept {
        return static_cast<std::int64_t>(sms) * ctas_per_sm;
    }
    // Nominal DRAM bandwidth in GB/s from the facts (MachineFacts::dram_gbps); 0 when unknown.
    [[nodiscard]] double dram_gbps() const noexcept { return facts.dram_gbps(); }
};

// The driver's values of `d` as machine_facts() takes them.
[[nodiscard]] DriverReport driver_report_of(const DeviceClass& d) noexcept;

// machine_facts(driver_report_of(d)): what the query stores in DeviceClass::facts. A test that
// sets a substitute's driver clocks calls it after them.
[[nodiscard]] MachineFacts machine_facts_of(const DeviceClass& d);

// A board as per-card measured data sees it: its instruction set and the quantities a kernel's
// best launch shape scales with. Data measured on one card (a route table, an R8 decode table, a
// measured gate) records the board it was measured on, and a device takes the data of the nearest
// measured board of its own instruction set (board_distance), never a chip name or an exact SM
// count: one chip ships as boards of 46 to 128 SMs and 300 to 1008 GB/s (AD102 4070 Ti SUPER,
// RTX 4090; AD104 L4, RTX 4070 Ti), and one product as two chips (RTX 4070 on AD104 and AD103).
struct Board {
    int major        = 0;
    int minor        = 0;
    int sms          = 0;
    double l2_mib    = 0; // 0: unknown, not compared
    double dram_gbps = 0; // 0: unknown, not compared
    bool mma_slow    = false; // DeviceFeatures::mma_slow: no tensor cores (board_mma_slow)
};

[[nodiscard]] Board board_of(const DeviceClass& d) noexcept;

// How far measured data of board `b` carries to board `a`: |log2| of the SM ratio, plus |log2| of
// the bandwidth ratio, plus half |log2| of the L2 ratio (a quantity either side does not know is
// not compared). Infinite between instruction sets (device_features_for rows that differ either
// way: routes and schedules are kernels of one) and between FP32-accumulate HMMA rates
// (GenerationRow::fp32acc_full_rate: an A30 is the A100's kind, not the A10's). Symmetric; 0 for the
// same board.
[[nodiscard]] double board_distance(const Board& a, const Board& b) noexcept;

// board_distance without the FP32-accumulate HMMA rate: how far a memory-bound choice carries (a
// decode or verify width, whose kernels stream weights at any rate). Infinite only between
// instruction sets.
[[nodiscard]] double board_memory_distance(const Board& a, const Board& b) noexcept;

// The entry of `measured` (anything with a `board`) nearest `board`, ties to the earlier; null
// when none is within `max_distance` (by default: any board of the same instruction set).
template <class Measured, std::size_t Extent>
[[nodiscard]] const Measured* nearest_measured(
    std::span<const Measured, Extent> measured, const Board& board,
    double max_distance = std::numeric_limits<double>::max()) noexcept {
    const Measured* nearest = nullptr;
    for (const Measured& m : measured) {
        const double d = board_distance(board, m.board);
        if (d <= max_distance && (nearest == nullptr || d < board_distance(board, nearest->board))) {
            nearest = &m;
        }
    }
    return nearest;
}

// "sm<major><minor>".
[[nodiscard]] std::string gpu_arch_id(int major, int minor);

// The chip of a GPU: by its PCI product id when the table knows it, else the smallest chip of its
// architecture whose full SM count holds `sms`, else the architecture ("sm<major><minor>").
[[nodiscard]] std::string gpu_chip(std::uint32_t pci_device_id, int major, int minor, int sms);
// Whether the PCI id's block names a chip of this architecture (gpu_chip then needs no SM count).
[[nodiscard]] bool gpu_chip_by_pci(std::uint32_t pci_device_id, int major, int minor);

// The facts this class lacks or guessed, each with the conservative choice taken in its place:
// the chip from the SM count (no PCI id, or one no block lists; the board's clocks may still be
// the PCI-ID table's), the clocks or bus width a virtualized or MIG GPU leaves unreported (the
// table's facts stand in for a listed board; otherwise no roofline: the A16 routes stand alone),
// and facts the table only approximates (MachineFacts::source extrapolated: an unlisted id, a
// laptop, a MIG slice). Empty for a fully known GPU. Logged at startup, so a degraded fact is
// never silent.
[[nodiscard]] std::vector<std::string> device_fact_notes(const DeviceClass& d);

// The architecture ("sm86") of a chip ("ga102") or GPU class key ("ga102:84sm") by the
// full-chip table; an architecture key ("sm86") is its own when the table has a chip of it. Empty
// for a chip the table does not know, an unknown architecture, or an architecture with an SM
// count ("sm86:84sm": the count belongs to a chip). Lower-case keys only.
[[nodiscard]] std::string gpu_chip_arch(std::string_view gpu_class);
// The full-chip SM count of a chip or GPU class key by the same table; 0 when unknown.
[[nodiscard]] int gpu_chip_full_sms(std::string_view gpu_class);

// "<chip>:<sms>sm".
[[nodiscard]] std::string gpu_class_id(std::string_view chip, int sms);

// The known chips of architecture <major>.<minor> other than `own_chip`, most similar first: by the
// distance of the chip's full SM count from `sms` on a log scale, so a 46-SM board is nearer a
// 60-SM chip than a 144-SM one, and ties go to the larger chip. After them the chips of a sibling
// architecture (the same major and the same feature row: sm_80 and sm_86, sm_121 and sm_120), in
// the same order; an architecture of other instructions is never similar. A board without tensor
// cores (mma_slow) and one with them are never similar: `mma_slow` is the board's own (the
// device's DeviceFeatures::mma_slow, board_mma_slow: also the GTX boards of TU106),
// chip_mma_slow(own_chip) when not given.
[[nodiscard]] std::vector<std::string> gpu_similar_chips(int major, int minor, int sms,
                                                         std::string_view own_chip,
                                                         std::optional<bool> mma_slow = {});

// The class chain widened by similarity: id, chip, arch (gpu_class_chain without "*"), the similar
// chips of the architecture, "*", then the chips of every other architecture, nearest first by
// compute capability and then by SM count as gpu_similar_chips orders them. Data that holds on any
// card walks it and takes the first key that has an entry: a card's own measurement, its
// architecture's, a sibling chip's, the curated fallback, and only then the most similar card
// measured at all.
// `mma_slow` is the board's own, as gpu_similar_chips takes it.
[[nodiscard]] std::vector<std::string> gpu_similarity_chain(std::string_view chip, int sms,
                                                            int major, int minor,
                                                            std::optional<bool> mma_slow = {});

// id, chip, arch, "*", without repeats (the chip is the arch when it is unknown).
[[nodiscard]] std::vector<std::string> gpu_class_chain(std::string_view chip, int sms,
                                                       std::string_view arch);

// The current device's class, cached per ordinal. Throws std::runtime_error when CUDA cannot
// report the device or the ordinal is invalid.
[[nodiscard]] const DeviceClass& device_class();
[[nodiscard]] const DeviceClass& device_class(int device);

// The identity of ordinal `device`: cudaDeviceGetPCIBusId and cudaGetDeviceProperties' uuid,
// without the rest of the class (no NVML, no kernel probe). Throws std::runtime_error when CUDA
// cannot report it.
[[nodiscard]] DeviceIdentity query_device_identity(int device);

// Every ordinal's identity, entry i for ordinal i. A board CUDA cannot report is sorted last
// (domain 0xFFFFFFFF), so it is never the default.
[[nodiscard]] std::vector<DeviceIdentity> visible_device_identities();

// The ordinal EngineOptions::device names (resolve_device in core/device_identity.h over the
// visible boards): an explicit ordinal as given, kDefaultDevice the first visible board in PCI
// order, whatever CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER made of the enumeration. 0 when CUDA
// reports no device, which DeviceContext then refuses as it refuses any ordinal there.
[[nodiscard]] int resolve_device(int requested);

// The ordinal of the visible board a --device selector names (select_device in
// core/device_identity.h). 0 when CUDA reports no device (DeviceContext refuses it, as above);
// throws std::invalid_argument listing the visible GPUs when none matches.
[[nodiscard]] int select_device(const DeviceSelector& selector);

// The board the current device's facts were read from, for the decision fingerprint to print
// beside its hash (never hashed: two boards of one chip type share a fingerprint).
[[nodiscard]] const DeviceIdentity& device_identity();

// A class built from a board's facts alone, as the query would build it (no CUDA call): the chip
// from the PCI id or the SM count, the architecture's features, and the per-block and per-SM
// limits of the architecture. Its driver clocks are 0 and its `facts` empty (FactSource::None):
// a test that wants the table's facts sets `facts = machine_facts_of(d)`. For host tests of the heuristics and route tables that read a
// class, on GPUs this machine lacks.
[[nodiscard]] DeviceClass synthetic_device_class(std::uint32_t pci_device_id, int major, int minor,
                                                 int sms, std::size_t l2_bytes,
                                                 std::size_t global_bytes);

// While alive, device_class() (the current device's form) answers `substitute` on this thread;
// device_class(int) still queries the ordinal. For tests: a consumer that cached what it read of
// the real class keeps it, so a test runs one substitute per process.
class ScopedDeviceClass {
public:
    explicit ScopedDeviceClass(DeviceClass substitute);
    ~ScopedDeviceClass();
    ScopedDeviceClass(const ScopedDeviceClass&)            = delete;
    ScopedDeviceClass& operator=(const ScopedDeviceClass&) = delete;

private:
    DeviceClass substitute_;
    const DeviceClass* previous_;
};

// Whether this binary carries a kernel image device `device` can load (SASS or PTX for its
// architecture), probed once per ordinal with an empty kernel compiled with the build. Kept out of
// DeviceClass so reading the class never launches the probe: that is safe before main (a static
// initializer can run before this binary's kernels are registered with the runtime).
[[nodiscard]] bool device_has_kernel_image(int device);

// A kernel of a separately compiled CUDA library, registered under `library` so that
// require_device_support also refuses a device that library has no image for. The build compiles
// every library for one architecture list, which the core probe above stands for; a library with
// its own list (or a plugin) registers a kernel of its own. Call from ordinary code, not a static
// initializer (the runtime may not have registered the kernel yet).
void register_kernel_image_probe(const char* library, const void* kernel);

// Whether the device can load `kernel` (an image of its architecture, SASS or PTX, is present).
[[nodiscard]] bool kernel_image_loadable(const void* kernel, int device);

// Resident CTAs per SM of one compiled kernel at `threads` threads and `dynamic_shared_bytes` of
// dynamic shared memory on the current device (cudaOccupancyMaxActiveBlocksPerMultiprocessor),
// queried once per (kernel, launch shape, device). Registers and shared memory are properties of
// the compiled object, so a cooperative grid must be sized from this, never from a table.
[[nodiscard]] int kernel_resident_ctas_per_sm(const void* kernel, int threads,
                                              std::size_t dynamic_shared_bytes);

// Throws std::runtime_error naming `consumer` when this build has no kernel image for the device
// or the device lacks a feature in `required`; the message lists every missing piece.
void require_device_support(const DeviceClass& device, const DeviceFeatures& required,
                            const char* consumer);

} // namespace ninferno
