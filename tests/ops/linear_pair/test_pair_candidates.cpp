// ops::linear_pair's candidates for one call (#310, ops/linear/linear_registry.h
// linear_pair_candidates): the fused Q8 pair kernel and the two separate A16 linear() calls, one
// selected as linear_pair runs, each refused with why. Weights are described, never read.
#include "core/weight_view.h"
#include "ops/linear/linear_registry.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
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
        // The fused kernel's geometry: it runs; the separate calls stay listed, not taken.
        {
            const Described f(QType::Q8_G32_FP16, 1024, 5120, a), s(QType::Q8_G32_FP16, 1024, 5120, b);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            check(c.size() == 2 && c[0].id == "fused:q8_pair" && c[0].status == Status::Selected &&
                      c[1].id == "separate" && c[1].status == Status::Inadmissible &&
                      c[1].why == "fused" && c[1].parts.size() == 4,
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
        // The fused kernel's K/V geometry (1024 x 2048).
        {
            const Described f(QType::Q8_G32_FP16, 1024, 2048, a), s(QType::Q8_G32_FP16, 1024, 2048, b);
            const auto c = ops::detail::linear_pair_candidates(f.weight, s.weight, t);
            check(c.size() == 2 && c[0].status == Status::Selected && c[1].status == Status::Inadmissible,
                  "1024x2048 Q8:" + describe(c) + at);
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
    if (failures != 0) {
        std::cerr << "linear_pair candidates: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "linear_pair candidates: the fused pair or the two calls, one selected as linear_pair runs\n";
    return 0;
}
