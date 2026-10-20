// ninferno-gemma4-inspect <artifact> [--a16|--a16-full]: load a Gemma 4 artifact through its model
// family and print what every layer resolved to — the projection each Op will receive, its format,
// shape, policy and divisors. Needs the device (weights are materialized) but runs no kernels.

#include "core/device.h"
#include "models/gemma4/gemma4.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::models::gemma4;

namespace {

const char* format(QType q) {
    switch (q) {
    case QType::Q4_G64_FP16: return "q4";
    case QType::Q5_G64_FP16: return "q5";
    case QType::Q6_G64_FP16: return "q6";
    case QType::Q8_G32_FP16: return "q8";
    case QType::BF16: return "bf16";
    case QType::FP32: return "fp32";
    case QType::INT32: return "i32";
    case QType::NVFP4: return "nvfp4";
    case QType::FP8_E4M3FN_ROW_BF16: return "fp8";
    }
    return "?";
}

const char* policy(ops::LinearPolicy p) {
    switch (p) {
    case ops::LinearPolicy::A16Only: return "A16";
    case ops::LinearPolicy::AllowA8: return "A8";
    case ops::LinearPolicy::AllowA4: return "A4";
    }
    return "?";
}

std::string describe(const char* role, const Weight& w, ops::LinearPolicy p) {
    char line[256];
    std::snprintf(line, sizeof(line), "%s=%s[%d,%d]/%s wdiv=%g in=%g stack=%d", role, format(w.qtype),
                  w.n, w.k, policy(p), w.weight_scale_divisor, w.input_scale_divisor,
                  w.weight_divisor_rows ? w.n / w.weight_divisor_rows : 0);
    return line;
}

// Reads a stacked plane's per-source divisors back from the device, to show they are distinct.
std::vector<float> divisors(const Weight& w, int count) {
    std::vector<float> out(count);
    if (w.weight_divisors != nullptr) {
        cudaMemcpy(out.data(), w.weight_divisors, sizeof(float) * count, cudaMemcpyDeviceToHost);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            throw std::invalid_argument(
                "usage: ninferno-gemma4-inspect <artifact> [--a16|--a16-full]");
        }
        const std::string mode = argc > 2 ? argv[2] : "";
        const auto profile     = mode == "--a16-full" ? ActivationProfile::A16Full
                                 : mode == "--a16"    ? ActivationProfile::A16
                                                      : ActivationProfile::Artifact;
        DeviceContext device(0);
        const auto start = std::chrono::steady_clock::now();
        auto model       = Model::load(argv[1], device, profile);
        const double s   = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto& c    = model->config();
        std::printf("loaded in %.1f s, %.3f GiB on device (upload %.1f s)\n", s,
                    model->device_bytes() / 1073741824.0, model->upload_seconds());
        std::printf("hidden=%d vocab=%d layers=%d heads=%d window=%d inter=%d experts=%d/%d/%d "
                    "softcap=%g k_eq_v=%d scale=%g\n",
                    c.hidden_size, c.vocab_size, c.layers, c.query_heads, c.sliding_window,
                    c.intermediate_size, c.experts, c.experts_per_token, c.expert_intermediate,
                    c.softcap.value_or(0.0F), int(c.k_eq_v), c.attention_scale);
        std::printf("sliding D=%d kv=%d theta=%g pairs=%d | full D=%d kv=%d theta=%g pairs=%d\n",
                    c.sliding.head_dim, c.sliding.kv_heads, c.sliding.rope_theta,
                    c.sliding.rotary_pairs, c.full.head_dim, c.full.kv_heads, c.full.rope_theta,
                    c.full.rotary_pairs);
        const auto& w = model->weights();
        std::printf("embedding %s[%d,%d]  %s\n", format(w.token_embedding.qtype), w.token_embedding.n,
                    w.token_embedding.k,
                    describe("head", w.output_head.weight, w.output_head.policy).c_str());
        for (int i = 0; i < c.layers; ++i) {
            const auto& l = w.layers[i];
            std::string line = std::string(l.kind == LayerKind::Full ? "F" : "S") + " " +
                               std::to_string(i) + " scalar=" + std::to_string(l.output_scale) + " ";
            if (l.qkv) {
                line += describe("qkv", l.qkv->weight, l.qkv->policy);
            } else {
                line += describe("q", l.query.weight, l.query.policy) + " " +
                        describe("k", l.key.weight, l.key.policy);
                if (l.value.weight.n) { line += " " + describe("v", l.value.weight, l.value.policy); }
            }
            line += " " + describe("o", l.output.weight, l.output.policy) + " " +
                    describe("gu", l.gate_up.weight, l.gate_up.policy) + " " +
                    describe("dn", l.down.weight, l.down.policy);
            if (l.moe) {
                // The prefill route's encoding; decode and small-T are A16 under every profile.
                const auto experts = l.moe_activations == ops::Gemma4MoeActivations::A16Only
                                         ? ops::LinearPolicy::A16Only
                                         : ops::LinearPolicy::AllowA4;
                line += " " + describe("egu", l.moe->gate_up, experts) + " " +
                        describe("edn", l.moe->down, experts);
            }
            std::printf("%s\n", line.c_str());
            if (l.moe && i == 0) {
                const auto gd = divisors(l.moe->gate_up, 4);
                const auto dd = divisors(l.moe->down, 4);
                std::printf("  layer0 expert gate_up divisors %g %g %g %g | down %g %g %g %g\n",
                            gd[0], gd[1], gd[2], gd[3], dd[0], dd[1], dd[2], dd[3]);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninferno-gemma4-inspect: " << error.what() << '\n';
        return 1;
    }
}
