#include "ops/generic/selector_facts.h"

#include "core/generation.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_query.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops::detail::generic {

DeviceClass board_device_class(const FactsBoard& b, double sm_scale, double memory_scale) {
    DeviceClass d      = synthetic_device_class(b.pci_device_id, b.major, b.minor, b.sms,
                                                b.l2_mib * kMiB, b.memory_gib * 1024 * kMiB);
    d.sm_clock_khz     = static_cast<int>(b.sm_clock_mhz * 1000 * sm_scale);
    d.memory_clock_khz = static_cast<int>(b.memory_clock_mhz * 1000 * memory_scale);
    d.memory_bus_bits  = b.bus_bits;
    d.facts            = machine_facts_of(d); // as the query derives them from these values
    return d;
}

bool board_prices_from_facts(const FactsBoard& b) { return rmma_terms_measured(b.major, b.minor); }

std::size_t selector_facts_probe(const FactsBoard& board) {
    // Every format the register-MMA terms are kept for (has_subblock_scales), a new one included.
    std::vector<QType> formats;
    for (int v = 0; v < 256; ++v) {
        if (has_subblock_scales(static_cast<QType>(v))) formats.push_back(static_cast<QType>(v));
    }

    struct Shape {
        std::int32_t n, k;
    };

    constexpr Shape kShapes[]        = {{1024, 4096}, {4096, 4096}, {17408, 5120}, {5120, 17408}};
    constexpr std::int32_t kTokens[] = {17, 32, 64, 128, 256, 1024};
    const DeviceClass device         = board_device_class(board);
    std::vector<RmmaFactRead> reads;
    {
        const ScopedRmmaFactsSink sink(reads);
        for (const QType qtype : formats) {
            for (const Shape shape : kShapes) {
                for (const Epilogue epilogue :
                     {Epilogue::Store, Epilogue::AddResidual, Epilogue::SwiGlu}) {
                    for (const std::int32_t t : kTokens) {
                        const std::int32_t rows =
                            epilogue == Epilogue::SwiGlu ? shape.n / 2 : shape.n;
                        try {
                            (void)route_query(device, qtype, ops::LinearPolicy::A16Only, epilogue,
                                              rows, shape.k, t);
                        } catch (const std::exception&) {} // a shape the format does not take
                    }
                }
            }
        }
    }
    if (reads.empty()) {
        throw std::runtime_error(std::string(board.name) +
                                 ": the price read no kernel facts (no snapshot for sm_" +
                                 kernel_facts::release_arch(board.major, board.minor) + ")");
    }
    return reads.size();
}

LiveCheckResult selector_facts_live_check(const FactsBoard& board) {
    kernel_facts::LiveCheckSink reads;
    {
        const kernel_facts::ScopedLiveCheck live(reads);
        (void)selector_facts_probe(board);
    }
    LiveCheckResult result;
    result.checked = reads.size();
    for (const auto& r : reads) {
        if (!r.agrees()) result.mismatches.push_back(r);
    }
    return result;
}

} // namespace ninferno::ops::detail::generic

namespace ninferno::tools {

int selector_facts_main(int, char**) {
    namespace generic = ninferno::ops::detail::generic;
    // The kernels are this build's (the SASS of NINFERNO_BUILD_ARCH): their boards are those of
    // that compute capability, whatever GPU reads them.
    constexpr int kMajor = NINFERNO_BUILD_ARCH / 10, kMinor = NINFERNO_BUILD_ARCH % 10;
#ifdef NINFERNO_BUILD_ARCH_SPECIFIC
    constexpr bool kExactMinor = true; // an "a"/"f" target runs on its own minor only
#else
    constexpr bool kExactMinor = false;
#endif
    try {
        bool any_board = false;
        for (const auto& board : generic::kFactsBoards) {
            any_board = any_board || (generic::board_prices_from_facts(board) &&
                                      board.major == kMajor && board.minor == kMinor);
        }
        if (!any_board) {
            std::fprintf(stderr, "selector-facts: this build (sm_%d) prices no board from facts\n",
                         NINFERNO_BUILD_ARCH);
            return 2;
        }
        // The first GPU that runs this build's kernels.
        int count = 0, chosen = -1;
        if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
        for (int d = 0; d < count && chosen < 0; ++d) {
            int major = 0, minor = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d) !=
                    cudaSuccess ||
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, d) !=
                    cudaSuccess) {
                continue;
            }
            if (major == kMajor && (kExactMinor ? minor == kMinor : minor >= kMinor)) chosen = d;
        }
        if (chosen < 0) {
            std::fprintf(stderr, "selector-facts: no GPU runs this build's sm_%d kernels\n",
                         NINFERNO_BUILD_ARCH);
            return 2;
        }
        if (cudaSetDevice(chosen) != cudaSuccess) {
            throw std::runtime_error("cannot select GPU " + std::to_string(chosen));
        }
        int failed = 0;
        for (const auto& board : generic::kFactsBoards) {
            if (!generic::board_prices_from_facts(board) || board.major != kMajor ||
                board.minor != kMinor) {
                continue;
            }
            const auto result = generic::selector_facts_live_check(board);
            if (result.mismatches.empty()) {
                std::printf("%s ok %zu kernels\n", board.name, result.checked);
                continue;
            }
            ++failed;
            std::printf("%s MISMATCH %zu of %zu kernels\n", board.name, result.mismatches.size(),
                        result.checked);
            for (const auto& r : result.mismatches) {
                std::printf("  %s\n", generic::kernel_facts::describe(r).c_str());
            }
        }
        if (failed != 0) {
            std::fprintf(stderr,
                         "selector-facts: this build's sm_%s kernels differ from the committed "
                         "snapshot (src/ops/generic/data/kernel_facts/): a kernel, flag or "
                         "compiler change; regenerate it (tools/kernel_facts/kernel_facts.py)\n",
                         std::string(generic::kernel_facts::build_arch()).c_str());
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "selector-facts: %s\n", error.what());
        return 1;
    }
}

} // namespace ninferno::tools
