// The register-MMA kernel's static shared memory, as the host's constexpr geometry
// (ops/generic/rmma_geometry.h, which generic_gemm.cuh static_asserts its declarations against)
// computes it, against every row of every committed kernel-facts snapshot
// (src/ops/generic/data/kernel_facts/sm_<arch>.txt, what ptxas made of the kernels): the schedule-
// dependent part must equal the snapshot's exactly, for every listed instance on every
// architecture. A kernel whose shared arrays change without the geometry (or the geometry without
// the kernel: the static_assert) fails here, and so does a snapshot regenerated over a change.
//
// The decoders' own tables, staging and group tails are rows of geometry::kDecoderGeometry,
// static_asserted against the decoders by generic_gemm.cuh; the build's capabilities are the
// compute capability (cp.async from sm_80) and CMakeLists.txt's NINFERNO_FULL_CUDA_ARCHS (NVFP4).
// CPU, no GPU.

#include "ops/generic/rmma_geometry.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace geo = ninferno::ops::detail::generic::geometry;

int failures     = 0;
int rows_checked = 0;

void fail(const std::string& message) {
    if (++failures <= 20) std::cerr << "FAIL: " << message << '\n';
}

// The architectures of the full build (the NVFP4 kernels and, with them, the cluster hand-over):
// CMakeLists.txt's NINFERNO_FULL_CUDA_ARCHS, comma-joined into a compile definition: one list, not
// restated or parsed here.
std::set<std::string> full_build_archs() {
    std::set<std::string> archs;
    const std::string list = NINFERNO_FULL_CUDA_ARCHS;
    for (std::size_t at = 0; at <= list.size();) {
        std::size_t end = list.find(',', at);
        if (end == std::string::npos) end = list.size();
        if (end > at) archs.insert(list.substr(at, end - at));
        at = end + 1;
    }
    return archs;
}

void check_file(const std::filesystem::path& path, const std::set<std::string>& full_archs) {
    std::ifstream in(path);
    if (!in) {
        fail("cannot read " + path.string());
        return;
    }
    std::string arch;
    int rows = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream f(line);
        std::string first;
        f >> first;
        if (first == "arch") {
            f >> arch;
            continue;
        }
        if (first != "rmma") continue;
        std::string decoder;
        int wm, wk, mt, nt, cl, ep, lat, add, regs, stack, local;
        std::size_t shared;
        f >> decoder >> wm >> wk >> mt >> nt >> cl >> ep >> lat >> add >> regs >> stack >> local >>
            shared;
        if (!f) {
            fail(path.string() + ": malformed row: " + line);
            continue;
        }
        ++rows;
        ++rows_checked;
        const std::size_t want =
            geo::rmma_decoder_static_shared_bytes(decoder, wm, wk, mt, nt, cl, ep == 2, lat != 0,
                                                  std::stoi(arch), full_archs.count(arch) != 0);
        if (want != shared) {
            fail("sm_" + arch + " " + decoder + " " + std::to_string(wm) + "x" +
                 std::to_string(wk) + " m" + std::to_string(mt) + " n" + std::to_string(nt) +
                 " cl" + std::to_string(cl) + " ep" + std::to_string(ep) + " lat" +
                 std::to_string(lat) + ": snapshot " + std::to_string(shared) +
                 " B shared, geometry " + std::to_string(want));
        }
    }
    if (rows == 0) fail(path.string() + ": no rows");
}

} // namespace

int main() {
    // The threads of a CTA are the warps' lanes.
    static_assert(geo::rmma_threads(1, 8) == 256 && geo::rmma_threads(4, 1) == 128);
    const std::set<std::string> full = full_build_archs();
    if (full.count("120a") == 0) fail("NINFERNO_FULL_CUDA_ARCHS does not name 120a");
    int files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(
             std::filesystem::path(NINFERNO_SOURCE_DIR) / "src/ops/generic/data/kernel_facts")) {
        if (entry.path().extension() != ".txt") continue;
        check_file(entry.path(), full);
        ++files;
    }
    if (files < 8) fail("fewer snapshots than architectures: " + std::to_string(files));
    std::cout << "rmma geometry: " << rows_checked << " snapshot rows in " << files
              << " files checked\n";
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    return 0;
}
