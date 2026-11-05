// A route's tile and logical traffic (#298, route_tile and route_traffic), host only: each tile is its
// schedule's declared geometry (SwiGLU halves the output rows of a tile that stages gate and up
// rows together), the families priced without a tile have none, and the traffic follows the tile:
// the whole weight once for every column of CTAs, the whole activation once for every row block in
// the route's representation, the output once. ggml's kernels have no declared tile.
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace g = ninferno::ops::detail::generic;
using ninferno::ops::LinearPolicy;
using ninferno::QType;
using ninferno::Weight;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool near(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b)); }

} // namespace

int main() {
    using F = g::Family;
    for (const auto epilogue : {g::Epilogue::Store, g::Epilogue::SwiGlu}) {
        const int sets = epilogue == g::Epilogue::SwiGlu ? 2 : 1;
        for (int i = 0; i < g::rmma_schedule_count(); ++i) {
            const auto s = g::rmma_schedule_info(i);
            const auto t = g::route_tile(g::Route{F::RegisterMma, static_cast<std::uint8_t>(i)},
                                         epilogue, 64);
            check(t && t->out_rows == 16 * s.m_tiles * s.warps_m && t->tokens == 8 * s.n_tiles &&
                      t->k_split == s.cluster,
                  "rmma tile " + std::to_string(i));
        }
        for (int i = 0; i < g::tiled_schedule_count(); ++i) {
            const auto s = g::tiled_schedule_info(i);
            const auto t =
                g::route_tile(g::Route{F::TiledMma, static_cast<std::uint8_t>(i)}, epilogue, 64);
            check(t && t->out_rows * sets == s.block_n && t->tokens == s.block_m,
                  "tiled tile " + std::to_string(i));
        }
        for (int i = 0; i < g::w4a8_schedule_count(); ++i) {
            const auto s = g::w4a8_schedule_info(i);
            const auto t =
                g::route_tile(g::Route{F::W4a8Mma, static_cast<std::uint8_t>(i)}, epilogue, 64);
            check(t && t->out_rows * sets == s.block_n && t->tokens == s.block_m,
                  "w4a8 tile " + std::to_string(i));
        }
        check(!g::route_tile(g::Route{F::GgmlMmq, 0}, epilogue, 64),
              "ggml's matrix kernel picks its tile at launch");
        check(!g::route_tile(g::Route{F::R8Mma, 0}, epilogue, 4), "R8 has no tile");
        check(!g::route_tile(g::Route{F::GgmlVec, 0}, epilogue, 4), "the vector kernel has no tile");
        check(!g::route_tile(g::Route{F::Scalar, 0}, epilogue, 4), "scalar has no tile");
    }

    // The traffic of one rmma schedule: a weight of one byte a value, n = 4096 rows, K = 2048.
    Weight w;
    w.qtype         = QType::Q8_G32_FP16;
    w.payload_bytes = std::uint64_t{4096} * 2048;
    const g::Problem p{QType::Q8_G32_FP16, LinearPolicy::A16Only, 4096, 2048, g::Epilogue::Store};
    for (int i = 0; i < g::rmma_schedule_count(); ++i) {
        const g::Route r{F::RegisterMma, static_cast<std::uint8_t>(i)};
        const auto tile = *g::route_tile(r, p.epilogue, 100);
        const auto tr   = g::route_traffic(r, w, p, 100);
        const double cols = std::ceil(100.0 / tile.tokens);
        const double rows = std::ceil(4096.0 / tile.out_rows);
        check(tr && near(tr->weight_bytes, 4096.0 * 2048 * cols) &&
                  near(tr->activation_bytes, 100.0 * 2048 * 2 * rows) &&
                  near(tr->output_bytes, 4096.0 * 100 * 2),
              "rmma traffic " + std::to_string(i));
    }
    // The int8 tier reads 36 bytes a 32-value block of the activation.
    if (g::w4a8_schedule_count() > 0) {
        const g::Route r{F::W4a8Mma, 0};
        const auto tile = *g::route_tile(r, p.epilogue, 256);
        const auto tr   = g::route_traffic(r, w, p, 256);
        check(tr && near(tr->activation_bytes,
                         256.0 * 2048 * 36 / 32 * std::ceil(4096.0 / tile.out_rows)),
              "w4a8 activation traffic");
    }
    // I8: an FP32 scale and an int32 sum a 64-value group, tokens padded to 256.
    if (g::i8_schedule_count() > 0) {
        const g::Route r{F::I8Mma, 0};
        const auto tile = *g::route_tile(r, p.epilogue, 100);
        const auto tr   = g::route_traffic(r, w, p, 100);
        check(tr && near(tr->activation_bytes,
                         256.0 * 2048 * (1.0 + 8.0 / 64.0) * std::ceil(4096.0 / tile.out_rows)),
              "i8 activation traffic");
    }
    check(!g::route_traffic(g::Route{F::R8Mma, 0}, w, p, 4), "R8 has no traffic description");

    if (failures == 0) std::cout << "route tiles and traffic: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
