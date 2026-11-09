// The maintainer's live check (#612): on a GPU that runs this build's SASS, every register-MMA
// kernel the price describes is read from the running GPU too (cudaFuncGetAttributes, the
// driver's occupancy) and must equal the committed snapshot and the pure occupancy on the
// device's facts. The decisions never read the live values; this test is what keeps the table
// honest against the build. Skips (77) without such a GPU, and where this nvcc is not the
// snapshot's (ptxas then differs by construction; the price still reads the snapshot).
#include "core/device_class.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_query.h"
#include "ops/generic/selector_facts.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace generic = ninferno::ops::detail::generic;
namespace kf      = generic::kernel_facts;
using namespace ninferno;

constexpr int kSkip = 77;

int report(const std::string& where, const generic::LiveCheckResult& r) {
    if (r.mismatches.empty()) {
        std::cout << where << ": " << r.checked << " kernels agree with the snapshot\n";
        return 0;
    }
    std::cerr << where << ": " << r.mismatches.size() << " of " << r.checked
              << " kernels differ from the snapshot\n";
    std::size_t shown = 0;
    for (const auto& m : r.mismatches) {
        if (shown++ < 40) std::cerr << "  " << kf::describe(m) << '\n';
    }
    return 1;
}

// The host device's own kernels (no substitute): the register-MMA kernels described for a few
// formats, shapes and widths, under the live check.
generic::LiveCheckResult host_device_check() {
    constexpr QType kFormats[] = {QType::GGML_Q4_K,      QType::GGML_Q6_K, QType::GGML_Q2_K,
                                  QType::GGML_IQ2_S,     QType::BF16,      QType::Q4_G64_FP16,
                                  QType::U4_G32_ZP_FP16, QType::GGML_Q4_0};
    void* const payload        = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    kf::LiveCheckSink reads;
    {
        const kf::ScopedLiveCheck live(reads);
        for (const QType qtype : kFormats) {
            std::optional<test::quantized_weight::PackedWeight> host;
            try {
                host.emplace(test::quantized_weight::make_patterned_weight(qtype, 4096, 4096, 7U));
            } catch (const std::invalid_argument&) { continue; }
            const Weight w = host->device_weight(payload);
            if (!generic::supports(w)) continue;
            for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                for (const generic::Epilogue e :
                     {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                      generic::Epilogue::SwiGlu}) {
                    for (const std::int32_t t : {1, 8, 32}) {
                        (void)generic::rmma_resources(i, w, e, t);
                    }
                }
            }
        }
    }
    generic::LiveCheckResult r;
    r.checked = reads.size();
    for (const auto& read : reads) {
        if (!read.agrees()) r.mismatches.push_back(read);
    }
    return r;
}

} // namespace

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::cout << "kernel facts live: no CUDA device, skipped\n";
            return kSkip;
        }
        const DeviceClass& d = device_class();
        const auto arch      = kf::sass_arch();
        if (!arch) {
            std::cout << "kernel facts live: " << d.name << " (cc " << d.major << "." << d.minor
                      << ") does not run this build's sm_" << kf::build_arch()
                      << " SASS, skipped\n";
            return kSkip;
        }
        const auto header = kf::snapshot_header(*arch);
        if (!header) {
            std::cerr << "kernel facts live: no snapshot for sm_" << *arch << '\n';
            return 1;
        }
        if (header->nvcc != NINFERNO_NVCC_VERSION) {
            std::cout << "kernel facts live: this build's nvcc " << NINFERNO_NVCC_VERSION
                      << " is not the snapshot's " << header->nvcc << ", skipped\n";
            return kSkip;
        }
        int failed = report(std::string(d.name) + " (sm_" + *arch + ")", host_device_check());
        // The facts-priced golden boards of this build's capability: the probe the release runs.
        for (const auto& board : generic::kFactsBoards) {
            if (!generic::board_prices_from_facts(board) || board.major != d.major ||
                board.minor != d.minor) {
                continue;
            }
            failed += report(board.name, generic::selector_facts_live_check(board));
        }
        if (failed != 0) {
            std::cerr << "kernel facts live: the snapshot differs from this build's kernels; "
                         "regenerate it (tools/kernel_facts/kernel_facts.py)\n";
            return 1;
        }
        std::cout << "kernel facts live: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kernel facts live: " << error.what() << '\n';
        return 1;
    }
}
