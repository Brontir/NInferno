#include "ninferno/ops/gemma4_vision.h"


#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::invalid_argument(what); }
}

void check_launch(const char* op) {
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(status));
    }
}

bool contiguous_bf16(const Tensor& t) {
    return t.dtype == DType::BF16 && t.data != nullptr && t.is_contiguous();
}

constexpr int kQkvWarps = 4;

struct QkvParams {
    const __nv_bfloat16* src[3];
    std::int64_t src_stride[3]; // elements between tokens
    const __nv_bfloat16* weight[2];
    const std::int32_t* positions;
    __nv_bfloat16* out[3];
    int head_dim;
    int heads;
    int tokens;
    float eps;
    float theta;
    float q_scale;
};

// One warp per (token, head, q|k|v) vector; each lane holds pairs (i, i + D/2).
__global__ void __launch_bounds__(kQkvWarps * 32) vision_qkv_kernel(QkvParams p) {
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (warp >= p.tokens * p.heads * 3) { return; }
    const int which = warp % 3;
    const int rest  = warp / 3;
    const int h     = rest % p.heads;
    const int t     = rest / p.heads;
    const int d     = p.head_dim;
    const int half  = d / 2;
    const int quart = d / 4;
    const __nv_bfloat16* x =
        p.src[which] + static_cast<std::int64_t>(t) * p.src_stride[which] +
        static_cast<std::int64_t>(h) * d;
    float ss = 0.0F;
    for (int i = lane; i < d; i += 32) {
        const float v = __bfloat162float(x[i]);
        ss += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffU, ss, o); }
    const float inv = rsqrtf(ss / static_cast<float>(d) + p.eps);
    __nv_bfloat16* y = p.out[which] + (static_cast<std::int64_t>(t) * p.heads + h) * d;
    if (which == 2) {
        for (int i = lane; i < d; i += 32) { y[i] = __float2bfloat16(__bfloat162float(x[i]) * inv); }
        return;
    }
    const __nv_bfloat16* w = p.weight[which];
    const float scale      = which == 0 ? p.q_scale : 1.0F;
    for (int i = lane; i < half; i += 32) {
        const float a    = __bfloat162float(x[i]) * inv * __bfloat162float(w[i]);
        const float b    = __bfloat162float(x[i + half]) * inv * __bfloat162float(w[i + half]);
        const int axis   = i < quart ? 0 : 1;
        const int j      = i < quart ? i : i - quart;
        const float pos  = static_cast<float>(p.positions[2 * t + axis]);
        const float freq = powf(p.theta, -4.0F * static_cast<float>(j) / static_cast<float>(d));
        float s, c;
        sincosf(pos * freq, &s, &c);
        y[i]        = __float2bfloat16((a * c - b * s) * scale);
        y[i + half] = __float2bfloat16((b * c + a * s) * scale);
    }
}

constexpr int kPoolThreads = 256;

// One CTA per soft token: the pooled, scaled, standardized row in shared memory, then its norm.
__global__ void __launch_bounds__(kPoolThreads)
    vision_pool_kernel(const __nv_bfloat16* __restrict__ x, int hidden, int width, int kernel,
                       int out_width, const __nv_bfloat16* __restrict__ bias,
                       const __nv_bfloat16* __restrict__ scale, float root, float eps,
                       __nv_bfloat16* __restrict__ out) {
    extern __shared__ float row[];
    __shared__ float partial[kPoolThreads / 32];
    const int n        = static_cast<int>(blockIdx.x);
    const int ny       = n / out_width;
    const int nx       = n - ny * out_width;
    const float inv_k2 = 1.0F / static_cast<float>(kernel * kernel);
    float ss           = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kPoolThreads) {
        float acc = 0.0F;
        for (int dy = 0; dy < kernel; ++dy) {
            const std::int64_t base =
                static_cast<std::int64_t>(kernel * ny + dy) * width + kernel * nx;
            for (int dx = 0; dx < kernel; ++dx) {
                acc += __bfloat162float(x[(base + dx) * hidden + d]);
            }
        }
        const float m = __bfloat162float(__float2bfloat16(acc * inv_k2));
        const float s = __bfloat162float(__float2bfloat16(
            (m * root - __bfloat162float(bias[d])) * __bfloat162float(scale[d])));
        row[d] = s;
        ss += s * s;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffU, ss, o); }
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = ss; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int i = 0; i < kPoolThreads / 32; ++i) { total += partial[i]; }
    const float inv = rsqrtf(total / static_cast<float>(hidden) + eps);
    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kPoolThreads) {
        out[static_cast<std::int64_t>(n) * hidden + d] = __float2bfloat16(row[d] * inv);
    }
}

} // namespace

void gemma4_vision_qkv_prepare(const Tensor& q_src, const Tensor& k_src, const Tensor& v_src,
                               std::int32_t head_dim, const Tensor& positions,
                               const Tensor& q_weight, const Tensor& k_weight, float eps,
                               float theta, float q_scale, Tensor& q_out, Tensor& k_out,
                               Tensor& v_out, cudaStream_t stream) {
    constexpr const char* op = "gemma4_vision_qkv_prepare";
    require(head_dim >= 4 && head_dim <= 256 && head_dim % 4 == 0,
            std::string(op) + ": head_dim must be a multiple of 4 in [4, 256]");
    require(std::isfinite(eps) && eps > 0 && std::isfinite(theta) && theta > 0 &&
                std::isfinite(q_scale),
            std::string(op) + ": eps and theta must be positive and finite, q_scale finite");
    const std::int32_t rows   = q_src.ne[0];
    const std::int32_t tokens = q_src.ne[1];
    require(rows > 0 && rows % head_dim == 0 && tokens > 0,
            std::string(op) + ": q_src must be [heads*D, T]");
    const std::int32_t heads = rows / head_dim;
    const Tensor* sources[3] = {&q_src, &k_src, &v_src};
    QkvParams p{};
    for (int i = 0; i < 3; ++i) {
        const Tensor& s = *sources[i];
        require(s.dtype == DType::BF16 && s.data != nullptr && s.ne[0] == rows &&
                    s.ne[1] == tokens && s.ne[2] == 1 && s.ne[3] == 1 && s.nb[0] == 2 &&
                    s.nb[1] % 2 == 0 && s.nb[1] >= static_cast<std::int64_t>(rows) * 2,
                std::string(op) + ": sources must be BF16 [heads*D, T] with contiguous features");
        p.src[i]        = static_cast<const __nv_bfloat16*>(s.data);
        p.src_stride[i] = s.nb[1] / 2;
    }
    Tensor* outputs[3] = {&q_out, &k_out, &v_out};
    for (int i = 0; i < 3; ++i) {
        const Tensor& o = *outputs[i];
        require(contiguous_bf16(o) && o.ne[0] == head_dim && o.ne[1] == heads &&
                    o.ne[2] == tokens && o.ne[3] == 1,
                std::string(op) + ": outputs must be contiguous BF16 [D, heads, T]");
        p.out[i] = static_cast<__nv_bfloat16*>(o.data);
    }
    require(contiguous_bf16(q_weight) && q_weight.numel() == head_dim &&
                contiguous_bf16(k_weight) && k_weight.numel() == head_dim,
            std::string(op) + ": q_weight and k_weight must be contiguous BF16 [D]");
    require(positions.dtype == DType::I32 && positions.data != nullptr &&
                positions.is_contiguous() && positions.numel() == 2 * std::int64_t(tokens),
            std::string(op) + ": positions must be contiguous I32 [2, T]");
    p.weight[0] = static_cast<const __nv_bfloat16*>(q_weight.data);
    p.weight[1] = static_cast<const __nv_bfloat16*>(k_weight.data);
    p.positions = static_cast<const std::int32_t*>(positions.data);
    p.head_dim  = head_dim;
    p.heads     = heads;
    p.tokens    = tokens;
    p.eps       = eps;
    p.theta     = theta;
    p.q_scale   = q_scale;
    const std::int64_t warps = std::int64_t(tokens) * heads * 3;
    const auto blocks        = static_cast<unsigned>((warps + kQkvWarps - 1) / kQkvWarps);
    vision_qkv_kernel<<<blocks, kQkvWarps * 32, 0, stream>>>(p);
    check_launch(op);
}

void gemma4_vision_pool(const Tensor& x, std::int32_t width, std::int32_t kernel,
                        const Tensor& std_bias, const Tensor& std_scale, float root, float eps,
                        Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gemma4_vision_pool";
    require(contiguous_bf16(x) && x.ne[2] == 1 && x.ne[3] == 1,
            std::string(op) + ": x must be contiguous BF16 [H, patches]");
    const std::int32_t hidden  = x.ne[0];
    const std::int32_t patches = x.ne[1];
    require(hidden > 0 && hidden <= 8192, std::string(op) + ": H must be in [1, 8192]");
    require(kernel > 0 && width > 0 && width % kernel == 0 && patches % width == 0 &&
                (patches / width) % kernel == 0,
            std::string(op) + ": the patch grid must be whole k x k cells");
    const std::int32_t out_width = width / kernel;
    const std::int32_t tokens    = out_width * (patches / width / kernel);
    require(contiguous_bf16(out) && out.ne[0] == hidden && out.ne[1] == tokens && out.ne[2] == 1 &&
                out.ne[3] == 1,
            std::string(op) + ": out must be contiguous BF16 [H, soft tokens]");
    require(contiguous_bf16(std_bias) && std_bias.numel() == hidden &&
                contiguous_bf16(std_scale) && std_scale.numel() == hidden,
            std::string(op) + ": std_bias and std_scale must be contiguous BF16 [H]");
    require(std::isfinite(root) && root > 0 && std::isfinite(eps) && eps > 0,
            std::string(op) + ": root and eps must be positive and finite");
    const auto shared = static_cast<std::size_t>(hidden) * sizeof(float);
    vision_pool_kernel<<<static_cast<unsigned>(tokens), kPoolThreads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), hidden, width, kernel, out_width,
        static_cast<const __nv_bfloat16*>(std_bias.data),
        static_cast<const __nv_bfloat16*>(std_scale.data), root, eps,
        static_cast<__nv_bfloat16*>(out.data));
    check_launch(op);
}

} // namespace ninferno::ops
