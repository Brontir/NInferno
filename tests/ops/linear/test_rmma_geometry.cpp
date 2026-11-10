// The register-MMA kernel's static shared memory, as the host's constexpr geometry
// (ops/generic/rmma_geometry.h, which generic_gemm.cuh static_asserts its declarations against)
// computes it, against every row of every committed kernel-facts snapshot
// (src/ops/generic/data/kernel_facts/sm_<arch>.txt, what ptxas made of the kernels): the schedule-
// dependent part must equal the snapshot's exactly, for every listed instance on every
// architecture. A kernel whose shared arrays change without the geometry (or the geometry without
// the kernel: the static_assert) fails here, and so does a snapshot regenerated over a change.
//
// What the geometry does not own yet, stated as data below: each decoder's own table (the IQ
// codebooks) and which decoders stage their activations; both are checked against the snapshots
// here, not exported from the decoders.
// CPU, no GPU.

#include "ops/generic/rmma_geometry.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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

// The decoders' own shared tables (generic_gemm.cuh kTableWords x 4 bytes), by the snapshot's name.
const std::map<std::string, std::size_t>& decoder_tables() {
    static const std::map<std::string, std::size_t> m = {
        {"GgmlIq1MDecoder", 8192},  {"GgmlIq1SDecoder", 8192},   {"GgmlIq2SDecoder", 8192},
        {"GgmlIq2XsDecoder", 4096}, {"GgmlIq2XxsDecoder", 2048}, {"GgmlIq3SDecoder", 2048},
        {"GgmlIq3XxsDecoder", 1024}};
    return m;
}

// The decoders that copy a SwiGLU step's activations ahead with cp.async (StageActsOf): J = 8 MMAs
// a K step.
bool stages_activations(const std::string& decoder) {
    return decoder == "GgmlQ3KDecoder" || decoder == "GgmlQ45KQuadDecoder<4>" ||
           decoder == "GgmlQ45KQuadDecoder<5>" || decoder == "GgmlQ6KDecoder";
}

constexpr int kStageJ = 8;

// The register-MMA kernel's shared memory for one snapshot row, from the geometry.
std::size_t expected_shared(const std::string& arch, const std::string& decoder, int warps_m,
                            int warps_k, int m_tiles, int n_tiles, int cluster, int epilogue,
                            bool latency) {
    const int sets = geo::rmma_sets(epilogue == 2);
    // The cluster hand-over exists in the sm_120a build (the NVFP4 kernels) only.
    const bool with_nvfp4 = arch == "120a";
    // cp.async: sm_80 and up (sm_70 and sm_75 copy synchronously, no stage).
    const bool cp_async = arch != "70" && arch != "75";
    const bool staged   = cp_async && latency && epilogue == 2 && n_tiles == 1 &&
                        warps_m * warps_k <= 8 && stages_activations(decoder);
    const auto table = decoder_tables().find(decoder);
    return geo::rmma_static_shared_bytes(warps_m, warps_k, sets, m_tiles, n_tiles, cluster,
                                         with_nvfp4, staged ? kStageJ : 0,
                                         table == decoder_tables().end() ? 0 : table->second);
}

void check_file(const std::filesystem::path& path) {
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
        const std::size_t want = expected_shared(arch, decoder, wm, wk, mt, nt, cl, ep, lat != 0);
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
    int files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(NINFERNO_KERNEL_FACTS_DIR)) {
        if (entry.path().extension() != ".txt") continue;
        check_file(entry.path());
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
