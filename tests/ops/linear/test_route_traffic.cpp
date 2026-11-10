// A route's tile and logical traffic (#298, route_tile and route_traffic), host only: each tile is
// its schedule's declared geometry (SwiGLU halves the output rows of a tile that stages gate and up
// rows together), the families priced without a tile have none, and the traffic follows the tile:
// the whole weight once for every column of CTAs, the whole activation once for every row block in
// the route's representation, the output once. ggml's kernels have no declared tile. The bytes a
// weight value takes are the call's share of the weight's rows (bytes_per_weight, #640).
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_select_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

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
                  near(tr->activation_bytes, 100.0 * 2048 * 2 * rows * g::rmma_schedule_info(i).warps_m) &&
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

    // A whole weight (Weight::n the call's rows, twice them under SwiGLU) or one of no stored rows
    // keeps payload / (n K) to the bit, so no whole-weight price moves; a slice of a fused parent
    // takes its rows' share. Complementary to test_slice_price.cpp, which compares a slice with its
    // own shape through this same helper and so cannot see a factor wrong on both sides (dropping
    // the SwiGLU `* sets`, flooring the stored rows at 1): only this identity catches those, while
    // the slice test catches a revert at any one pricing site.
    for (const auto epilogue : {g::Epilogue::Store, g::Epilogue::SwiGlu}) {
        const std::int32_t sets = epilogue == g::Epilogue::SwiGlu ? 2 : 1;
        for (const auto& [n, k] :
             {std::pair{256, 2048}, std::pair{1024, 4096}, std::pair{6144, 5120},
              std::pair{17408, 5120}, std::pair{5120, 17408}, std::pair{28672, 8192}}) {
            const std::int32_t rows = n / sets;
            const g::Problem q{QType::GGML_Q4_K, LinearPolicy::A16Only, rows, k, epilogue};
            const std::string at =
                std::to_string(rows) + "x" + std::to_string(k) + (sets == 2 ? " swiglu" : " store");
            Weight whole;
            whole.n             = n;
            whole.payload_bytes = static_cast<std::uint64_t>(n) * k * 9 / 16 + 777;
            const double today =
                static_cast<double>(whole.payload_bytes) / (static_cast<double>(rows) * k);
            check(g::bytes_per_weight(whole, q) == today, at + ": a whole weight keeps its bpw");
            Weight unsized = whole;
            unsized.n      = 0;
            check(g::bytes_per_weight(unsized, q) == today, at + ": no stored rows keeps its bpw");
            Weight parent        = whole;
            parent.n             = 8 * n;
            parent.payload_bytes = 8 * whole.payload_bytes;
            check(std::fabs(g::bytes_per_weight(parent, q) - today) <= 1e-12 * today,
                  at + ": a slice of an 8x parent takes its rows' share");
        }
    }

    if (failures == 0) std::cout << "route tiles and traffic: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
