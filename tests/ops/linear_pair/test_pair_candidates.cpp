// ops::linear_pair's candidates for one call (#310, ops/linear/linear_registry.h
// linear_pair_candidates): the fused Q8 pair kernel and the two separate A16 linear() calls, one
// selected as linear_pair runs, each refused with why. Weights are described, never read.
#include "core/weight_view.h"
#include "ops/linear/linear_registry.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ops = ninferno::ops;
using ninferno::QType;
using ninferno::Weight;
using Status = ops::detail::LinearCandidate::Status;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// A weight as the converter lays it out, at an aligned stand-in address (route_query.cpp's
// description).
struct Described {
    ninferno::WeightParent parent;
    ninferno::WeightView view;
    Weight weight;
    Described(QType qtype, std::int32_t n, std::int32_t k, std::uintptr_t at) {
        parent.data = reinterpret_cast<const std::byte*>(at);
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)};
        parent.geometry = ninferno::weight_geometry(qtype, ninferno::default_weight_layout(qtype), shape);
        view.shape      = {shape[0], shape[1]};
        view.parts      = {ninferno::WeightRegion{&parent, 0, shape[0] * shape[1]}};
        weight          = ninferno::native_weight(view, 0.0F);
    }
    Described(const Described&)            = delete;
    Described& operator=(const Described&) = delete;
};

// 1024 rows of a Q8_G32 row-split parent from row `first` (the DFlash drafter's K or V view of its
// fused QKV plane), as the loader binds them; a second view shares the first's parent.
struct Rows {
    std::shared_ptr<ninferno::WeightParent> parent;
    ninferno::WeightView view;
    Weight weight;
    Rows(std::uint64_t rows, std::uint64_t k, std::uint64_t first, std::uintptr_t at)
        : parent(std::make_shared<ninferno::WeightParent>()) {
        parent->data = reinterpret_cast<const std::byte*>(at);
        const std::uint64_t shape[2] = {rows, k};
        parent->geometry = ninferno::weight_geometry(
            QType::Q8_G32_FP16, ninferno::default_weight_layout(QType::Q8_G32_FP16), shape);
        bind(first, k);
    }
    Rows(const Rows& other, std::uint64_t first) : parent(other.parent) { bind(first, other.view.shape[1]); }
    void bind(std::uint64_t first, std::uint64_t k) {
        view.shape  = {1024, k};
        view.parts  = {ninferno::WeightRegion{parent.get(), first * k, (first + 1024) * k}};
        weight      = ninferno::native_weight(view, 0.0F);
    }
};

std::string describe(const std::vector<ops::detail::LinearPairCandidate>& c) {
    std::string out;
    for (const auto& x : c) {
        out += " [" + x.id + " " + std::to_string(static_cast<int>(x.status)) +
               (x.why.empty() ? "" : " " + x.why) + " parts " + std::to_string(x.parts.size()) + "]";
    }
    return out;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "CUDA unavailable; skipping linear_pair candidates test\n";
        return 77;
    }
    const std::uintptr_t a = std::uintptr_t{1} << 32, b = std::uintptr_t{1} << 34;
    for (const std::int32_t t : {1, 4, 17, 512}) {
        const std::string at = " T=" + std::to_string(t);
        // The fused kernel's geometry: the chain with fewer requested bytes runs (a tie keeps the
        // fused kernel); the other stays listed, not taken.
        {
            const Described f(QType::Q8_G32_FP16, 1024, 5120, a), s(QType::Q8_G32_FP16, 1024, 5120, b);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            const auto r     = ops::detail::linear_pair_requested_bytes(f.weight, s.weight, t);
            const bool fused = !(r.fused && r.separate) || *r.fused <= *r.separate;
            check(c.size() == 2 && c[0].id == "fused:q8_pair" &&
                      c[0].status == (fused ? Status::Selected : Status::Dominated) &&
                      c[1].id == "separate" &&
                      c[1].status == (fused ? Status::Inadmissible : Status::Selected) &&
                      c[1].parts.size() == 4,
                  "1024x5120 Q8:" + describe(c) + at);
        }
        // Another shape: no fused kernel; the two calls run, each with its own candidates.
        {
            const Described f(QType::Q8_G32_FP16, 1024, 4096, a), s(QType::Q8_G32_FP16, 1024, 4096, b);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            check(c.size() == 2 && c[0].status == Status::Unavailable &&
                      c[0].why == "no fused kernel for this [N,K]" &&
                      c[1].status == Status::Selected && c[1].parts.size() == 4,
                  "1024x4096 Q8:" + describe(c) + at);
            int selected = 0;
            for (const auto& p : c.size() == 2 ? c[1].parts : std::vector<ops::detail::LinearCandidate>{}) {
                selected += p.status == Status::Selected ? 1 : 0;
            }
            check(selected == 2, "each separate call selects one implementation" + at);
        }
        // The fused geometry in another format: not the kernel's layout.
        {
            const Described f(QType::BF16, 1024, 5120, a), s(QType::BF16, 1024, 5120, b);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            check(c.size() == 2 && c[0].status == Status::Unavailable &&
                      c[0].why.find("layout") != std::string::npos && c[1].status == Status::Selected,
                  "1024x5120 BF16:" + describe(c) + at);
        }
        // The fused kernel's K/V geometry (1024 x 2048): only the drafter's adjacent K/V row
        // views of one 6144-row parent (rows 4096 and 5120); two weights of their own run apart.
        {
            const Described f(QType::Q8_G32_FP16, 1024, 2048, a), s(QType::Q8_G32_FP16, 1024, 2048, b);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            check(c.size() == 2 && c[0].status == Status::Unavailable &&
                      c[0].why.find("row views") != std::string::npos &&
                      c[1].status == Status::Selected,
                  "1024x2048 Q8, two parents:" + describe(c) + at);
        }
        {
            const Rows f(6144, 2048, 4096, a), s(f, 5120);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            const auto r     = ops::detail::linear_pair_requested_bytes(f.weight, s.weight, t);
            const bool fused = !(r.fused && r.separate) || *r.fused <= *r.separate;
            check(c.size() == 2 && c[0].status == (fused ? Status::Selected : Status::Dominated) &&
                      c[1].status == (fused ? Status::Inadmissible : Status::Selected),
                  "1024x2048 Q8, K/V row views:" + describe(c) + at);
        }
        // A fragment-order pair: linear() refuses each weight, so neither path runs.
        {
            Described f(QType::Q8_G32_FP16, 1024, 4096, a), s(QType::Q8_G32_FP16, 1024, 4096, b);
            f.weight.layout = ninferno::QuantLayout::GgmlFrag;
            s.weight.layout = ninferno::QuantLayout::GgmlFrag;
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            check(c.size() == 2 && c[0].status != Status::Selected && c[1].status == Status::Unavailable,
                  "ggml_frag_v1 pair:" + describe(c) + at);
        }
        // Two shapes: linear_pair refuses them; nothing is selected.
        {
            const Described f(QType::Q8_G32_FP16, 1024, 4096, a), s(QType::Q8_G32_FP16, 2048, 4096, b);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            check(c.size() == 2 && c[0].status != Status::Selected && c[1].status == Status::Unavailable,
                  "two shapes:" + describe(c) + at);
        }
    }
    // The chain prices over the widths, for the record (MB the CTAs request; * the chain that
    // runs).
    for (const std::int32_t k : {5120, 2048}) {
        const Rows f(6144, static_cast<std::uint64_t>(k), 4096, a), s(f, 5120);
        for (const std::int32_t t :
             {1, 2, 4, 8, 16, 17, 32, 48, 64, 85, 86, 96, 128, 192, 256, 512, 1024, 2048}) {
            const auto r  = ops::detail::linear_pair_requested_bytes(f.weight, s.weight, t);
            const auto c  = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            const auto mb = [](const std::optional<double>& v) {
                return v ? std::to_string(*v / 1.0e6) : std::string("n/a");
            };
            std::cout << "K=" << k << " T=" << t << " fused " << mb(r.fused) << " separate "
                      << mb(r.separate) << " runs "
                      << (c[0].status == Status::Selected ? "fused" : "separate") << '\n';
        }
    }
    if (failures != 0) {
        std::cerr << "linear_pair candidates: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "linear_pair candidates: the fused pair or the two calls, one selected as linear_pair runs\n";
    return 0;
}
