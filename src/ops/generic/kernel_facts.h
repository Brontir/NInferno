#pragma once

// The kernel-facts snapshots (#612): what ptxas made of the generic path's register-MMA kernels,
// per build architecture, as committed data (src/ops/generic/data/kernel_facts/sm_<arch>.txt,
// written by tools/kernel_facts/kernel_facts.py from the linked binary, all of them embedded in
// every build). The selector's price and the launch's variant rules read these facts, never the
// host GPU's kernels (cudaFuncGetAttributes): a decision is then a function of data in git, the
// same on every machine whatever its nvcc or driver (#556), any board of any architecture is
// priced on a CPU host, and the selector goldens are kept per board again. The live query stays a
// maintainer check that the table equals the build (ninferno-serve selector-facts,
// ninferno_kernel_facts_live_test).
//
// A kernel is keyed as its symbol spells it: the decoder type, the schedule's template arguments
// (NINFERNO_GENERIC_RMMA_SCHEDULES), the epilogue, its latency and add-sums variants. The launch
// builds the key from the template arguments it instantiates (rmma_kernel_key), the tool from the
// demangled symbol, in one canonical spelling (decoder_name): nothing is retyped.

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::ops::detail::generic::kernel_facts {

struct RmmaKernelKey {
    std::string decoder; // canonical type name: "GgmlQ45KQuadDecoder<4>", "Bf16Decoder"
    int warps_m = 0, warps_k = 0, m_tiles = 0, n_tiles = 0, cluster = 1;
    int epilogue  = 0; // Epilogue as an integer
    bool latency  = false;
    bool add_sums = false;
};

// The instructions of the kernel's K loop, counted once per loop iteration off the SASS
// (tools/kernel_facts/sass_counts.py, #613 PR-3: how the loop is found and what a static count
// means). One iteration of rmma_kernel's loop is two K steps; k_iter is the K values a warp's
// weight MMAs cover in it (2 * 4 * Decoder::kRun where the loop was found), so a count per K
// value is count / k_iter. Bytes are what the warp's instructions request (32 lanes times the
// width), not unique memory traffic. Facts for the price's terms (kernel_roofline.h): mma and
// mma_shape feed the tensor-pipe term, the loads the LSU and spill terms, int_alu and fp_alu the
// issue term. Nothing reads them yet.
struct LoopCounts {
    int loops  = 0; // candidate K loops found: 1 normally; 0 none (every count is 0); 2+ ambiguous
    int k_iter = 0; // K values per warp per iteration from the weight MMAs; 0 unknown
    int mma    = 0; // tensor-core MMAs, the min-term ones included
    int mma_ones = 0; // of them, the min-term MMAs with a 1.0 operand (Q4_K / Q5_K / Q4_1 / Q5_1)
    std::string mma_shape;        // the MMA opcode ("HMMA.16816.F32.BF16"), "-" without one
    int ldg                  = 0; // global loads
    std::size_t ldg_bytes    = 0;
    int ld                   = 0; // generic loads (the activations)
    std::size_t ld_bytes     = 0;
    int ldgsts               = 0; // asynchronous global-to-shared copies
    std::size_t ldgsts_bytes = 0;
    int lds                  = 0; // shared loads
    std::size_t lds_bytes    = 0;
    int sts                  = 0; // shared stores
    std::size_t sts_bytes    = 0;
    int local                = 0; // local loads and stores: the spills in the loop
    std::size_t local_bytes  = 0;
    int shfl                 = 0;
    int int_alu              = 0;
    int fp_alu               = 0;
    int uniform              = 0; // uniform-datapath instructions
    int other                = 0; // branches, barriers, convergence, constant loads, NOP
    int total                = 0;
};

struct KernelFacts {
    int registers                   = 0; // per thread
    std::size_t stack_bytes         = 0; // per thread: the stack frame (spills and stack arrays)
    std::size_t local_bytes         = 0; // per thread: other local memory
    std::size_t static_shared_bytes = 0; // per CTA
    std::optional<LoopCounts> loop;      // schema 2 snapshots; nullopt in a resources-only one

    // Per thread: every local-memory byte (the driver's localSizeBytes), the one spill measure
    // the price, the wide rule and the live check all read.
    [[nodiscard]] std::size_t spill_bytes() const noexcept { return stack_bytes + local_bytes; }
};

// A committed snapshot's header, as the file records it.
struct SnapshotHeader {
    std::string arch;   // the build architecture label: "89", "120a"
    std::string nvcc;   // the nvcc that compiled the kernels
    std::string flags;  // the CUDA flags of that build
    std::string source; // sha256 of the device sources it was made from
    std::size_t kernels = 0;
};

// The build architecture labels with a committed snapshot, in file order.
[[nodiscard]] std::vector<std::string_view> snapshot_archs();
// The header of a label's snapshot; nullopt where none is committed.
[[nodiscard]] std::optional<SnapshotHeader> snapshot_header(std::string_view arch);
// The facts of one kernel of a label's snapshot; nullopt where the label has no snapshot or the
// snapshot holds no such kernel (the build then prices by threads alone).
[[nodiscard]] std::optional<KernelFacts> rmma_facts(std::string_view arch,
                                                    const RmmaKernelKey& key);

// The label of the release engine whose SASS a board of this compute capability runs, by the
// dispatcher's rule over the committed snapshots (apps/dispatch): an architecture-specific
// snapshot of the exact capability, else a family one of the same major, else the newest plain
// one of the same major not above it. Empty where no engine's SASS runs there (the driver would
// compile the PTX: no facts).
[[nodiscard]] std::string release_arch(int major, int minor);

// The label of the kernels this thread's device_class() runs, where their facts are known: a
// substituted board's release engine (release_arch: what the board would run, whatever this
// build is), else this build's own label where the current device runs its SASS (an exact match
// for an architecture-specific build, the family for a family build, the same major and an equal
// or higher minor for a plain one). nullopt where the device would run PTX the driver compiled,
// or there is no device: the price is then by threads alone and no variant pays.
[[nodiscard]] std::optional<std::string> sass_arch();

// This build's architecture label (CMAKE_CUDA_ARCHITECTURES: "120a", "89").
[[nodiscard]] std::string_view build_arch();

// The maintainer's live check (ninferno-serve selector-facts, ninferno_kernel_facts_live_test):
// while a sink is installed on this thread, every register-MMA kernel the launch describes is also
// read from the running GPU (cudaFuncGetAttributes, cudaOccupancyMaxActiveBlocksPerMultiprocessor)
// and both readings are recorded here for the caller to compare. The decisions themselves never
// read the live values.
struct LiveRead {
    RmmaKernelKey key;
    std::string arch;                     // the snapshot the price read (sass_arch), "" for none
    std::optional<KernelFacts> snapshot;  // what the price read
    KernelFacts live;                     // what the running GPU's kernel has
    int threads                      = 0; // the launch's
    std::size_t dynamic_shared_bytes = 0; // the launch's
    int snapshot_ctas                = 0; // resident CTAs per SM from the snapshot (0: no facts)
    int live_ctas                    = 0; // the same from the driver

    [[nodiscard]] bool agrees() const noexcept {
        return snapshot && snapshot->registers == live.registers &&
               snapshot->spill_bytes() == live.stack_bytes &&
               snapshot->static_shared_bytes == live.static_shared_bytes &&
               snapshot_ctas == live_ctas;
    }
};

using LiveCheckSink = std::vector<LiveRead>;

class ScopedLiveCheck {
public:
    explicit ScopedLiveCheck(LiveCheckSink& sink) noexcept;
    ~ScopedLiveCheck();
    ScopedLiveCheck(const ScopedLiveCheck&)            = delete;
    ScopedLiveCheck& operator=(const ScopedLiveCheck&) = delete;

private:
    LiveCheckSink* previous_;
};

[[nodiscard]] LiveCheckSink* live_check_target() noexcept;

// One line naming a read: its key, the snapshot's facts and the live ones.
[[nodiscard]] std::string describe(const LiveRead& read);

// The canonical spelling of a type name as the snapshots key it: the generic namespace dropped,
// no `struct`/`class`, no spaces, integers bare, bools true/false (the tool spells cu++filt's
// output the same way).
[[nodiscard]] std::string canonical_type_name(std::string_view raw);

// The compiler's spelling of T, from the enclosing function's signature string.
template <class T>
constexpr std::string_view raw_type_name() {
#if defined(_MSC_VER) && !defined(__clang__)
    constexpr std::string_view sig  = __FUNCSIG__; // "... raw_type_name<struct X<4>>(void)"
    constexpr std::string_view head = "raw_type_name<";
    const auto start                = sig.find(head) + head.size();
    const auto end                  = sig.rfind(">(void)");
    return sig.substr(start, end - start);
#else
    constexpr std::string_view sig = __PRETTY_FUNCTION__; // "[with T = X<4>; ...]" or "[T = X<4>]"
    const auto start               = sig.find("T = ") + 4;
    auto end                       = sig.find(';', start);
    if (end == std::string_view::npos) end = sig.find(']', start);
    return sig.substr(start, end - start);
#endif
}

// The canonical name of a decoder type, once per type. A decoder whose template argument is an
// enumerator declares kKeyArgument, its value: compilers spell an enum argument by name
// (Ggml32::Q4_0), cu++filt by value ((Ggml32)0), and the snapshot keeps the value.
// kKeyArgument stands for the one template argument of the decoder itself: a decoder with more
// arguments, or one derived from such a decoder, must not carry or inherit it.
template <class T>
const std::string& decoder_name() {
    static const std::string name = [] {
        std::string n = canonical_type_name(raw_type_name<T>());
        if constexpr (requires { T::kKeyArgument; }) {
            n = n.substr(0, n.find('<')) + '<' + std::to_string(int{T::kKeyArgument}) + '>';
        }
        return n;
    }();
    return name;
}

} // namespace ninferno::ops::detail::generic::kernel_facts
