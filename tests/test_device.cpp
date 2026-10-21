#include "core/device.h"
#include "support/cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int expect_throws_device(int device_id) {
    try {
        ninferno::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninferno::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.transfer_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.compute_capability() <= 0) {
        std::cerr << label << " compute capability is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

int expect(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "device class: " << what << '\n'; }
    return ok ? 0 : 1;
}

int check_device_class(const ninferno::DeviceContext& ctx) {
    int failures   = 0;
    const auto& t  = ctx.device_class();
    const auto get = [&](cudaDeviceAttr attr) {
        int value = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, ctx.device));
        return value;
    };
    failures += expect(t.device == ctx.device, "device ordinal");
    failures += expect(t.sms == get(cudaDevAttrMultiProcessorCount), "SM count");
    failures += expect(t.l2_bytes == static_cast<std::size_t>(get(cudaDevAttrL2CacheSize)), "L2");
    failures += expect(t.smem_per_block_optin ==
                           static_cast<std::size_t>(get(cudaDevAttrMaxSharedMemoryPerBlockOptin)),
                       "shared memory per block (opt-in)");
    failures += expect(t.global_bytes == ctx.total_vram(), "total memory");
    failures += expect(t.compute_capability() == ctx.compute_capability(), "compute capability");
    failures += expect(t.memory_bus_bits > 0 && t.memory_clock_khz > 0 && t.dram_gbps() > 0.0,
                       "memory bus and clock");
    failures += expect(&t == &ninferno::device_class(ctx.device), "one cached record per device");
    failures += expect(&t == &ninferno::device_class(), "current device lookup");
    failures += expect(t.chip == ninferno::gpu_chip(t.pci_device_id, t.major, t.minor, t.sms) &&
                           t.id() == ninferno::gpu_class_id(t.chip, t.sms) &&
                           t.arch() == ninferno::gpu_arch_id(t.major, t.minor) &&
                           t.class_chain() == ninferno::gpu_class_chain(t.chip, t.sms, t.arch()) &&
                           t.class_chain().front() == t.id() && t.class_chain().back() == "*",
                       "class id " + t.id());
    // An NVIDIA board answers with its PCI product id (sysfs natively, NVML under WSL2).
    failures += expect(t.pci_device_id != 0, "PCI product id");
    // The running test binary launched kernels above, so its image loads on this device.
    failures += expect(ninferno::device_has_kernel_image(t.device), "kernel image probe");
    const auto expected = ninferno::common_features(ninferno::device_features_for(t.major, t.minor),
                                                  ninferno::build_features());
    failures += expect(ninferno::has_features(t.features, expected) &&
                           ninferno::has_features(expected, t.features),
                       "feature set: the architecture table's row and the build's in common");
    failures += expect(ninferno::has_features(ninferno::build_features(), t.features),
                       "no feature the build carries no code for");
    try {
        ninferno::require_device_support(t, t.features, "device test");
    } catch (const std::exception& e) { failures += expect(false, e.what()); }
    std::cout << "device " << t.name << " pci 0x" << std::hex << t.pci_device_id << std::dec
              << " class " << t.id() << " SMs " << t.sms
              << " L2 " << (t.l2_bytes >> 20) << " MiB DRAM " << t.dram_gbps() << " GB/s\n";
    return failures;
}

int check_feature_table() {
    int failures = 0;
    // Chips by PCI product id (0x2B85 RTX 5090, 0x2BB1 RTX PRO 6000, 0x2C02 RTX 5080, 0x2684
    // RTX 4090), and by SM count when the id is unknown.
    failures += expect(ninferno::gpu_chip(0x2B85, 12, 0, 170) == "gb202" &&
                           ninferno::gpu_chip(0x2BB1, 12, 0, 188) == "gb202" &&
                           ninferno::gpu_chip(0x2C02, 12, 0, 84) == "gb203" &&
                           ninferno::gpu_chip(0x2684, 8, 9, 128) == "ad102",
                       "chip by PCI id");
    // A block names its chip only on that chip's architecture: Jetson Thor (sm_110) is 0x2B00,
    // inside GB202's block.
    failures += expect(ninferno::gpu_chip(0x2B00, 11, 0, 20) == "sm110",
                       "a PCI id block of another architecture names no chip");
    // Measured data carries by board, not chip: the RTX 5070 Ti (70 SMs, 48 MiB L2, 896 GB/s) is
    // nearest the RTX 5080 of the measured sm_120 boards, and no board of another instruction set
    // is near at all.
    {
        const ninferno::Board ti{12, 0, 70, 48, 896}, b5080{12, 0, 84, 64, 960},
            b5060{12, 0, 36, 32, 448}, b5090{12, 0, 170, 96, 1792}, l4{8, 9, 58, 48, 300};
        const double d80 = ninferno::board_distance(ti, b5080);
        failures += expect(ninferno::board_distance(ti, ti) == 0 &&
                               d80 == ninferno::board_distance(b5080, ti) &&
                               d80 < ninferno::board_distance(ti, b5060) &&
                               d80 < ninferno::board_distance(ti, b5090) &&
                               !std::isfinite(ninferno::board_distance(ti, l4)),
                           "board distance");
        const ninferno::Board a30{8, 0, 56, 24, 933}, a100{8, 0, 108, 40, 1555}, a10{8, 6, 72, 6, 600};
        failures += expect(std::isfinite(ninferno::board_distance(a30, a100)) &&
                               !std::isfinite(ninferno::board_distance(a30, a10)),
                           "board distance keeps FP32-accumulate HMMA rates apart");
        failures += expect(std::isfinite(ninferno::board_memory_distance(a30, a10)) &&
                               ninferno::board_memory_distance(a100, a100) == 0.0,
                           "board memory distance ignores the HMMA rate");
    }
    failures += expect(ninferno::gpu_chip(0, 12, 0, 170) == "gb202" &&
                           ninferno::gpu_chip(0, 12, 0, 84) == "gb203" &&
                           ninferno::gpu_chip(0, 12, 0, 70) == "gb203" &&
                           ninferno::gpu_chip(0, 12, 0, 48) == "gb205" &&
                           ninferno::gpu_chip(0, 12, 0, 36) == "gb206" &&
                           ninferno::gpu_chip(0, 8, 6, 82) == "ga102" &&
                           ninferno::gpu_chip(0, 12, 1, 48) == "sm121",
                       "chip by architecture and SM count");
    failures += expect(ninferno::gpu_class_id("gb202", 170) == "gb202:170sm" &&
                           ninferno::gpu_class_id("gb203", 84) == "gb203:84sm",
                       "class ids");
    failures += expect(ninferno::gpu_class_chain("gb202", 170, "sm120") ==
                               std::vector<std::string>{"gb202:170sm", "gb202", "sm120", "*"} &&
                           ninferno::gpu_class_chain("sm121", 48, "sm121") ==
                               std::vector<std::string>{"sm121:48sm", "sm121", "*"},
                       "class chains");
    failures +=
        expect(ninferno::gpu_similar_chips(8, 9, 46, "ad104") ==
                       std::vector<std::string>{"ad106", "ad103", "ad107", "ad102"} &&
                   ninferno::gpu_similar_chips(8, 6, 82, "ga102") ==
                       std::vector<std::string>{"ga103", "ga104", "ga106", "ga107", "ga100"} &&
                   ninferno::gpu_similar_chips(7, 5, 68, "tu102") ==
                       std::vector<std::string>{"tu104", "tu106"} &&
                   ninferno::gpu_similar_chips(7, 5, 22, "tu116") ==
                       std::vector<std::string>{"tu117"} &&
                   ninferno::gpu_similar_chips(8, 0, 108, "") ==
                       std::vector<std::string>{"ga100", "ga102", "ga103", "ga104", "ga106",
                                                "ga107"} &&
                   ninferno::gpu_similar_chips(12, 1, 48, "sm121") ==
                       std::vector<std::string>{"gb205", "gb206", "gb203", "gb207", "gb202"} &&
                   ninferno::gpu_similar_chips(8, 9, 46, "ad104").size() == 4,
               "similar chips: the same architecture by SM count on a log scale, then a "
               "sibling architecture of the same feature row, never another");
    {
        // GTX 1660 Super (TU116 by PCI ID), the same board with its PCI ID unknown (the SM-count
        // fallback names TU116 too), an RTX 2080 Ti (TU102) and a V100.
        const auto slow = [](std::uint32_t id, int major, int minor, int sms) {
            return ninferno::synthetic_device_class(id, major, minor, sms, 1536u << 10, 6ull << 30)
                .features.mma_slow;
        };
        const auto a = ninferno::board_of(ninferno::synthetic_device_class(0x21C4, 7, 5, 22,
                                                                           1536u << 10, 6ull << 30));
        const auto b = ninferno::board_of(ninferno::synthetic_device_class(0x1F08, 7, 5, 30,
                                                                           3u << 20, 6ull << 30));
        failures += expect(slow(0x21C4, 7, 5, 22) && slow(0, 7, 5, 22) && slow(0x1F82, 7, 5, 14) &&
                               !slow(0x1E07, 7, 5, 68) && !slow(0x1F08, 7, 5, 30) &&
                               !slow(0x1DB5, 7, 0, 80) && a.mma_slow && !b.mma_slow &&
                               std::isinf(ninferno::board_distance(a, b)),
                           "mma_slow: TU116 / TU117 by chip (PCI ID, else SM count), no other "
                           "board; boards either side of it are apart");
        // The PCI-ID table decides where it has the id: the GTX 16 boards on TU106 (0x1F09 GTX
        // 1660 SUPER, 0x1F0A GTX 1650) are slow though their chip is not, an RTX board on TU106
        // (0x1F07) is not. An id the table lacks takes its block's chip: 0x1F99 (TU117, a laptop
        // GTX 1650 not in the table) is slow by the chip.
        failures += expect(slow(0x1F09, 7, 5, 22) && slow(0x1F0A, 7, 5, 14) &&
                               !slow(0x1F07, 7, 5, 36) && slow(0x1F99, 7, 5, 16),
                           "mma_slow: the table's row where it has the id, else the chip's");
    }
    {
        const auto ampere = ninferno::gpu_similarity_chain("ga102", 82, 8, 6);
        const auto at     = [&](const std::string& key) {
            return std::find(ampere.begin(), ampere.end(), key) - ampere.begin();
        };
        failures += expect(ampere.size() > 8 && ampere[0] == "ga102:82sm" && ampere[1] == "ga102" &&
                               ampere[2] == "sm86" && ampere[3] == "ga103" &&
                               at("ga107") < at("*") && at("*") < at("ad103") &&
                               at("ad102") < at("tu102") && at("tu102") < at("gb202"),
                           "similarity chain: own keys, the architecture's chips, \"*\", then "
                           "the nearest compute capability's chips first");
    }
    {
        // A guessed chip and unreported clocks are labelled; a known board has no note.
        auto known       = ninferno::synthetic_device_class(0x2B85, 12, 0, 170, 96u << 20, 32ull << 30);
        known.sm_clock_khz = known.memory_clock_khz = known.memory_bus_bits = 1;
        auto guessed     = ninferno::synthetic_device_class(0, 12, 0, 170, 96u << 20, 32ull << 30);
        guessed.sm_clock_khz = guessed.memory_clock_khz = guessed.memory_bus_bits = 1;
        const auto mig   = ninferno::synthetic_device_class(0x2B85, 12, 0, 170, 96u << 20, 32ull << 30);
        const auto none  = ninferno::device_fact_notes(known);
        const auto chip  = ninferno::device_fact_notes(guessed);
        const auto clock = ninferno::device_fact_notes(mig);
        const auto h100 = ninferno::device_fact_notes(
            ninferno::synthetic_device_class(0x2330, 9, 0, 132, 50u << 20, 80ull << 30));
        failures += expect(!h100.empty() && h100[0].find("chip unknown") != std::string::npos &&
                               h100[0].find("sm90") != std::string::npos,
                           "a chip the table lacks: keyed by the architecture, not 'guessed'");
        failures += expect(known.chip_from_pci && none.empty() && !guessed.chip_from_pci &&
                               guessed.chip == "gb202" && chip.size() == 1 &&
                               chip[0].find("from the SM count") != std::string::npos &&
                               clock.size() == 1 &&
                               clock[0].find("SM clock, memory clock") != std::string::npos,
                           "degraded device facts are labelled, a known board has none");
    }
    {
        // The labels agree with the machine facts (as the query derives them): a listed board
        // whose driver reports no clocks has the table's roofline, not "the cost model is off";
        // a MIG slice and a laptop are labelled extrapolated; a chip without a block but a row in
        // the table names the row.
        const auto with_facts = [](ninferno::DeviceClass d) {
            d.facts = ninferno::machine_facts_of(d);
            return d;
        };
        const auto joined = [](const std::vector<std::string>& notes) {
            std::string all;
            for (const auto& n : notes) all += n + "\n";
            return all;
        };
        const auto slice = joined(ninferno::device_fact_notes(with_facts(
            ninferno::synthetic_device_class(0x20B0, 8, 0, 14, 40u << 20, 10ull << 30))));
        failures += expect(
            slice.find("reference table's facts stand in") != std::string::npos &&
                slice.find("are off") == std::string::npos &&
                slice.find("a slice of 14 SMs") != std::string::npos,
            "MIG A100 slice, no clocks: the table's facts, labelled extrapolated:\n" + slice);
        auto laptop_class =
            ninferno::synthetic_device_class(0x2717, 8, 9, 76, 64u << 20, 16ull << 30);
        laptop_class.sm_clock_khz = laptop_class.memory_clock_khz = laptop_class.memory_bus_bits =
            1;
        const auto laptop = joined(ninferno::device_fact_notes(with_facts(laptop_class)));
        failures += expect(laptop.find("a laptop: priced at the bottom") != std::string::npos,
                           "a laptop's facts are labelled extrapolated:\n" + laptop);
        auto h100_class =
            ninferno::synthetic_device_class(0x2330, 9, 0, 132, 50u << 20, 80ull << 30);
        h100_class.sm_clock_khz = h100_class.memory_clock_khz = h100_class.memory_bus_bits = 1;
        const auto h100 = joined(ninferno::device_fact_notes(with_facts(h100_class)));
        failures +=
            expect(h100.find("chip unknown") != std::string::npos &&
                       h100.find("reference table's H100 SXM5 80GB (gh100)") != std::string::npos &&
                       h100.find("extrapolated") == std::string::npos,
                   "H100: chip keyed by sm90, facts verified from its row:\n" + h100);
        auto full = with_facts(
            ninferno::synthetic_device_class(0x2B85, 12, 0, 170, 96u << 20, 32ull << 30));
        full.sm_clock_khz = full.memory_clock_khz = full.memory_bus_bits = 1;
        failures += expect(ninferno::device_fact_notes(full).empty(),
                           "a verified board with clocks has no note");
    }
    const auto blackwell = ninferno::device_features_for(12, 0);
    const auto spark     = ninferno::device_features_for(12, 1);
    const auto hopper    = ninferno::device_features_for(9, 0);
    const auto b200      = ninferno::device_features_for(10, 0);
    const auto ada       = ninferno::device_features_for(8, 9);
    const auto unknown   = ninferno::device_features_for(42, 0);
    failures += expect(blackwell.mma_mxf4nvf4 && blackwell.mma_f8f6f4 && blackwell.tma &&
                           blackwell.pdl && blackwell.mma_bf16,
                       "sm_120 feature row");
    failures += expect(ninferno::has_features(spark, blackwell), "sm_121 is sm_120's family");
    failures += expect(!b200.mma_mxf4nvf4 && !b200.mma_f8f6f4 && b200.tma,
                       "sm_100 is numerically above and lacks the mma.sync FP4/FP8 forms");
    failures += expect(!hopper.mma_f8f6f4 && hopper.tma && hopper.pdl, "sm_90 feature row");
    failures += expect(ada.mma_bf16 && !ada.tma && !ada.pdl && !ada.mma_f8f6f4, "sm_89 row");
    const auto ampere = ninferno::device_features_for(8, 6);
    failures += expect(ampere.mma_bf16 && !ampere.mma_fp8 && !ampere.tma && !ampere.pdl,
                       "sm_86 row: BF16 MMA, no E4M3 MMA");
    failures += expect(!ninferno::device_features_for(7, 0).mma_tensor &&
                           ninferno::device_features_for(7, 5).mma_tensor && ampere.mma_tensor &&
                           ada.mma_tensor && blackwell.mma_tensor,
                       "tensor mma.sync from sm_75 on; Volta emulates it");
    {
        // The selector reads BF16-native MMA as implying tensor mma.sync (a8_candidates).
        bool implied = true;
        for (int major = 7; major <= 13; ++major) {
            for (int minor = 0; minor <= 9; ++minor) {
                const auto f = ninferno::device_features_for(major, minor);
                implied      = implied && (!f.mma_bf16_native || f.mma_tensor);
            }
        }
        failures += expect(implied, "BF16-native MMA implies tensor mma.sync on every row");
    }
    failures +=
        expect(ada.mma_fp8 && hopper.mma_fp8 && b200.mma_fp8 && blackwell.mma_fp8 && spark.mma_fp8,
               "the un-prefixed E4M3 mma.sync from sm_89 on");
    failures +=
        expect(unknown.mma_bf16 && unknown.mma_bf16_native && unknown.mma_fp8 && unknown.tma &&
                   unknown.pdl && !unknown.mma_f8f6f4 && !unknown.mma_mxf4nvf4,
               "a later generation keeps the portable families, no block-scaled kind");
    // An older build on a newer GPU (an sm_89 tree's PTX on an RTX 5090): the paths the build
    // compiled, no PDL or TMA it compiled empty.
    const auto old_build_new_gpu = ninferno::common_features(blackwell, ada);
    failures += expect(ninferno::has_features(old_build_new_gpu, ada) &&
                           ninferno::has_features(ada, old_build_new_gpu),
                       "features in common: an sm_89 build on sm_120 is sm_89");
    failures += expect(!ninferno::common_features(unknown, blackwell).mma_mxf4nvf4 &&
                           ninferno::common_features(unknown, blackwell).pdl,
                       "an sm_120 build on a later GPU keeps PDL, not the block-scaled kinds");
    const auto pascal = ninferno::device_features_for(6, 1);
    failures +=
        expect(!pascal.mma_bf16 && !pascal.tma, "an older unknown architecture assumes nothing");
    // A device lacking a feature is refused, and the message names it.
    ninferno::DeviceClass without = ninferno::device_class(0);
    without.features            = {};
    try {
        ninferno::require_device_support(without, {.mma_f8f6f4 = true}, "rejection test");
        failures += expect(false, "a device without the required feature was admitted");
    } catch (const std::runtime_error& e) {
        failures += expect(std::string_view(e.what()).find(".kind::f8f6f4") != std::string_view::npos,
                           "rejection names the missing feature");
    }
    return failures;
}

// A failed checked call throws CudaError, and a layer's context added on the way up keeps the
// type, so the Engine still answers the failure as Unavailable.
int check_cuda_error_context() {
    int failures = 0;
    try {
        try {
            ninferno::cuda_check(cudaErrorInvalidValue, "probe()", "test_device.cpp", 1);
        } catch (const std::exception&) { ninferno::rethrow_with_context("text/layers/3 T=1"); }
        failures += fail("cuda_check did not throw");
    } catch (const ninferno::CudaError& error) {
        failures += expect(error.code() == cudaErrorInvalidValue &&
                               std::string_view(error.what()).starts_with("text/layers/3 T=1: "),
                           "CudaError keeps its code and type through rethrow_with_context");
    } catch (...) { failures += fail("CudaError lost its type"); }
    try {
        try {
            throw std::logic_error("plan");
        } catch (const std::exception&) { ninferno::rethrow_with_context("text/layers/0"); }
    } catch (const ninferno::CudaError&) {
        failures += fail("a host error became a CudaError");
    } catch (const std::runtime_error& error) {
        failures += expect(std::string_view(error.what()) == "text/layers/0: plan",
                           "a host error is rethrown as runtime_error with its context");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--invalid-sync") {
        try {
            ninferno::DeviceContext ctx(0);
        } catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).find("NINFERNO_CUDA_SYNC") !=
                           std::string_view::npos
                       ? 0
                       : fail("invalid sync setting has no configuration diagnostic");
        }
        return fail("invalid sync setting did not fail before CUDA initialization");
    }
    const unsigned int expected_flags =
        argc == 2 ? static_cast<unsigned int>(std::stoul(argv[1])) : cudaDeviceScheduleSpin;
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;
    failures += check_cuda_error_context();

    ninferno::DeviceContext ctx(0);
    unsigned int actual_flags = 0;
    CUDA_CHECK(cudaGetDeviceFlags(&actual_flags));
    if ((actual_flags & cudaDeviceScheduleMask) != expected_flags) {
        return fail("CUDA did not apply the requested synchronization schedule");
    }
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    failures += check_device_class(ctx);
    failures += check_feature_table();
    int* device_value = nullptr;
    int* host_value   = nullptr;
    CUDA_CHECK(cudaMalloc(&device_value, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_value, sizeof(int)));
    *host_value = 0;
    CUDA_CHECK(cudaMemsetAsync(device_value, 0x5a, sizeof(int), ctx.stream));
    CUDA_CHECK(
        cudaMemcpyAsync(host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    ctx.synchronize();
    const bool transfer_complete = *host_value == 0x5a5a5a5a;
    CUDA_CHECK(cudaFreeHost(host_value));
    CUDA_CHECK(cudaFree(device_value));
    if (!transfer_complete) {
        return fail("stream synchronization returned before transfer completed");
    }

    const cudaStream_t original_stream = ctx.stream;
    ninferno::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.transfer_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);

    ninferno::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
