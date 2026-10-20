#include "convert/ggml_quants.h"
#include "core/ascii.h"

#include "convert/numerics.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninferno::convert::ggml {
namespace {

#include "convert/ggml_tables.inc"

constexpr int kQK = 256; // QK_K
constexpr float kIq1Delta = 0.125F;

// Little-endian field reads (GGUF is little-endian; the converter runs on x86-64).
inline float half(const std::uint8_t* p) {
    std::uint16_t w;
    std::memcpy(&w, p, 2);
    return f16_to_f32(w);
}

inline std::uint16_t u16(const std::uint8_t* p) {
    std::uint16_t w;
    std::memcpy(&w, p, 2);
    return w;
}

inline std::uint32_t u32(const std::uint8_t* p) {
    std::uint32_t w;
    std::memcpy(&w, p, 4);
    return w;
}

// ggml_e8m0_to_fp32_half: 2^(e-127) / 2 (e = 0 and e = 1 are the denormal special cases).
inline float e8m0_half(std::uint8_t e) {
    std::uint32_t bits = e < 2 ? 0x00200000U << e : static_cast<std::uint32_t>(e - 1) << 23;
    return f32_from_bits(bits);
}

// ggml_ue4m3_to_fp32: an unsigned E4M3 word times 0.5 (0x7f reads as zero).
inline float ue4m3_half(std::uint8_t x) {
    if (x == 0 || x == 0x7F) { return 0.0F; }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = std::ldexp(static_cast<float>(man), -9);
    } else {
        raw = std::ldexp(1.0F + static_cast<float>(man) / 8.0F, exp - 7);
    }
    return raw * 0.5F;
}

inline void scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

inline float sign_of(std::uint8_t signs, int j) { return signs & kmask_iq2xs[j] ? -1.0F : 1.0F; }

// Each function decodes one block to y (block elements).

void q1_0(const std::uint8_t* x, float* y) {
    const float d     = half(x);
    const float neg_d = -d;
    for (int j = 0; j < 128; ++j) { y[j] = ((x[2 + j / 8] >> (j % 8)) & 1) ? d : neg_d; }
}

void q2_0(const std::uint8_t* x, float* y) {
    const float d = half(x);
    for (int j = 0; j < 64; ++j) {
        const int q = (x[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
        y[j]        = static_cast<float>(q - 1) * d;
    }
}

void q4_0(const std::uint8_t* x, float* y) {
    const float d = half(x);
    for (int j = 0; j < 16; ++j) {
        const int x0 = (x[2 + j] & 0x0F) - 8;
        const int x1 = (x[2 + j] >> 4) - 8;
        y[j]         = static_cast<float>(x0) * d;
        y[j + 16]    = static_cast<float>(x1) * d;
    }
}

void q4_1(const std::uint8_t* x, float* y) {
    const float d = half(x);
    const float m = half(x + 2);
    for (int j = 0; j < 16; ++j) {
        const int x0 = x[4 + j] & 0x0F;
        const int x1 = x[4 + j] >> 4;
        y[j]         = static_cast<float>(x0) * d + m;
        y[j + 16]    = static_cast<float>(x1) * d + m;
    }
}

void q5_0(const std::uint8_t* x, float* y) {
    const float d          = half(x);
    const std::uint32_t qh = u32(x + 2);
    const std::uint8_t* qs = x + 6;
    for (int j = 0; j < 16; ++j) {
        const std::uint8_t xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
        const std::uint8_t xh_1 = static_cast<std::uint8_t>(((qh >> (j + 12))) & 0x10);
        const std::int32_t x0   = ((qs[j] & 0x0F) | xh_0) - 16;
        const std::int32_t x1   = ((qs[j] >> 4) | xh_1) - 16;
        y[j]                    = static_cast<float>(x0) * d;
        y[j + 16]               = static_cast<float>(x1) * d;
    }
}

void q5_1(const std::uint8_t* x, float* y) {
    const float d          = half(x);
    const float m          = half(x + 2);
    const std::uint32_t qh = u32(x + 4);
    const std::uint8_t* qs = x + 8;
    for (int j = 0; j < 16; ++j) {
        const std::uint8_t xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
        const std::uint8_t xh_1 = static_cast<std::uint8_t>(((qh >> (j + 12))) & 0x10);
        const int x0            = (qs[j] & 0x0F) | xh_0;
        const int x1            = (qs[j] >> 4) | xh_1;
        y[j]                    = static_cast<float>(x0) * d + m;
        y[j + 16]               = static_cast<float>(x1) * d + m;
    }
}

void q8_0(const std::uint8_t* x, float* y) {
    const float d = half(x);
    for (int j = 0; j < 32; ++j) {
        y[j] = static_cast<float>(static_cast<std::int8_t>(x[2 + j])) * d;
    }
}

void mxfp4(const std::uint8_t* x, float* y) {
    const float d = e8m0_half(x[0]);
    for (int j = 0; j < 16; ++j) {
        const int x0 = kvalues_fp4[x[1 + j] & 0x0F];
        const int x1 = kvalues_fp4[x[1 + j] >> 4];
        y[j]         = static_cast<float>(x0) * d;
        y[j + 16]    = static_cast<float>(x1) * d;
    }
}

void nvfp4(const std::uint8_t* x, float* y) {
    for (int s = 0; s < 4; ++s) {
        const float d = ue4m3_half(x[s]);
        float* yb     = y + s * 16;
        for (int j = 0; j < 8; ++j) {
            const int v0 = kvalues_fp4[x[4 + s * 8 + j] & 0x0F];
            const int v1 = kvalues_fp4[x[4 + s * 8 + j] >> 4];
            yb[j]        = static_cast<float>(v0) * d;
            yb[j + 8]    = static_cast<float>(v1) * d;
        }
    }
}

void q2_k(const std::uint8_t* x, float* y) {
    const std::uint8_t* scales = x;
    const std::uint8_t* q      = x + 16;
    const float d              = half(x + 80);
    const float min            = half(x + 82);
    int is                     = 0;
    for (int n = 0; n < kQK; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            std::uint8_t sc = scales[is++];
            float dl        = d * static_cast<float>(sc & 0xF);
            float ml        = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * static_cast<float>(static_cast<std::int8_t>((q[l] >> shift) & 3)) - ml;
            }
            sc = scales[is++];
            dl = d * static_cast<float>(sc & 0xF);
            ml = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *y++ =
                    dl * static_cast<float>(static_cast<std::int8_t>((q[l + 16] >> shift) & 3)) -
                    ml;
            }
            shift += 2;
        }
        q += 32;
    }
}

void q3_k(const std::uint8_t* x, float* y) {
    constexpr std::uint32_t kmask1 = 0x03030303;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    const std::uint8_t* hm         = x;
    const std::uint8_t* q          = x + 32;
    const float d_all              = half(x + 108);
    std::uint32_t aux[4];
    std::memcpy(aux, x + 96, 12);
    const std::uint32_t tmp = aux[2];
    aux[2]                  = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3]                  = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0]                  = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1]                  = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::int8_t scales[16];
    std::memcpy(scales, aux, 16);
    std::uint8_t m = 1;
    int is         = 0;
    for (int n = 0; n < kQK; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * static_cast<float>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                const int v = static_cast<std::int8_t>((q[l + 0] >> shift) & 3) -
                              ((hm[l + 0] & m) ? 0 : 4);
                *y++ = dl * static_cast<float>(v);
            }
            dl = d_all * static_cast<float>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                const int v = static_cast<std::int8_t>((q[l + 16] >> shift) & 3) -
                              ((hm[l + 16] & m) ? 0 : 4);
                *y++ = dl * static_cast<float>(v);
            }
            shift += 2;
            m = static_cast<std::uint8_t>(m << 1);
        }
        q += 32;
    }
}

void q4_k(const std::uint8_t* x, float* y) {
    const float d              = half(x);
    const float min            = half(x + 2);
    const std::uint8_t* scales = x + 4;
    const std::uint8_t* q      = x + 16;
    int is                     = 0;
    std::uint8_t sc, m;
    for (int j = 0; j < kQK; j += 64) {
        scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(m);
        scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(m);
        for (int l = 0; l < 32; ++l) { *y++ = d1 * static_cast<float>(q[l] & 0xF) - m1; }
        for (int l = 0; l < 32; ++l) { *y++ = d2 * static_cast<float>(q[l] >> 4) - m2; }
        q += 32;
        is += 2;
    }
}

void q5_k(const std::uint8_t* x, float* y) {
    const float d              = half(x);
    const float min            = half(x + 2);
    const std::uint8_t* scales = x + 4;
    const std::uint8_t* qh     = x + 16;
    const std::uint8_t* ql     = x + 48;
    int is                     = 0;
    std::uint8_t sc, m;
    std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < kQK; j += 64) {
        scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(m);
        scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(m);
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 = static_cast<std::uint8_t>(u1 << 2);
        u2 = static_cast<std::uint8_t>(u2 << 2);
    }
}

void q6_k(const std::uint8_t* x, float* y) {
    const std::uint8_t* ql = x;
    const std::uint8_t* qh = x + 128;
    const auto* sc         = reinterpret_cast<const std::int8_t*>(x + 192);
    const float d          = half(x + 208);
    for (int n = 0; n < kQK; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const auto q1 =
                static_cast<std::int8_t>(((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            const auto q2 =
                static_cast<std::int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            const auto q3 =
                static_cast<std::int8_t>(((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            const auto q4 =
                static_cast<std::int8_t>(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            y[l + 0]  = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

void tq1_0(const std::uint8_t* x, float* y) {
    static constexpr std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const std::uint8_t* qs                = x;      // 48 bytes
    const std::uint8_t* qh                = x + 48; // 4 bytes
    const float d                         = half(x + 52);
    constexpr std::size_t kQs             = 48;
    for (std::size_t j = 0; j < kQs - kQs % 32; j += 32) {
        for (std::size_t n = 0; n < 5; ++n) {
            for (std::size_t m = 0; m < 32; ++m) {
                const auto q        = static_cast<std::uint8_t>(qs[j + m] * pow3[n]);
                const std::int16_t xi = static_cast<std::int16_t>((static_cast<std::uint16_t>(q) * 3) >> 8);
                *y++                = static_cast<float>(xi - 1) * d;
            }
        }
    }
    for (std::size_t j = kQs - kQs % 32; j < kQs; j += 16) {
        for (std::size_t n = 0; n < 5; ++n) {
            for (std::size_t m = 0; m < 16; ++m) {
                const auto q        = static_cast<std::uint8_t>(qs[j + m] * pow3[n]);
                const std::int16_t xi = static_cast<std::int16_t>((static_cast<std::uint16_t>(q) * 3) >> 8);
                *y++                = static_cast<float>(xi - 1) * d;
            }
        }
    }
    for (std::size_t n = 0; n < 4; ++n) {
        for (std::size_t j = 0; j < 4; ++j) {
            const auto q          = static_cast<std::uint8_t>(qh[j] * pow3[n]);
            const std::int16_t xi = static_cast<std::int16_t>((static_cast<std::uint16_t>(q) * 3) >> 8);
            *y++                  = static_cast<float>(xi - 1) * d;
        }
    }
}

void tq2_0(const std::uint8_t* x, float* y) {
    const float d = half(x + 64);
    for (std::size_t j = 0; j < 64; j += 32) {
        for (std::size_t l = 0; l < 4; ++l) {
            for (std::size_t m = 0; m < 32; ++m) {
                const auto q = static_cast<std::int8_t>((x[j + m] >> (l * 2)) & 3);
                *y++         = static_cast<float>(q - 1) * d;
            }
        }
    }
}

void iq2_xxs(const std::uint8_t* x, float* y) {
    const float d = half(x);
    for (int ib32 = 0; ib32 < kQK / 32; ++ib32) {
        std::uint32_t aux32[2];
        std::memcpy(aux32, x + 2 + 8 * ib32, 8);
        const auto* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
        const float db = d * (0.5F + static_cast<float>(aux32[1] >> 28)) * 0.25F;
        for (int l = 0; l < 4; ++l) {
            const auto* grid = reinterpret_cast<const std::uint8_t*>(iq2xxs_grid + aux8[l]);
            const std::uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
            for (int j = 0; j < 8; ++j) {
                y[j] = db * static_cast<float>(grid[j]) * sign_of(signs, j);
            }
            y += 8;
        }
    }
}

void iq2_xs(const std::uint8_t* x, float* y) {
    const float d              = half(x);
    const std::uint8_t* qs     = x + 2; // 32 uint16
    const std::uint8_t* scales = x + 66;
    for (int ib32 = 0; ib32 < kQK / 32; ++ib32) {
        float db[2];
        db[0] = d * (0.5F + static_cast<float>(scales[ib32] & 0xf)) * 0.25F;
        db[1] = d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F;
        for (int l = 0; l < 4; ++l) {
            const std::uint16_t q = u16(qs + 2 * (4 * ib32 + l));
            const auto* grid = reinterpret_cast<const std::uint8_t*>(iq2xs_grid + (q & 511));
            const std::uint8_t signs = ksigns_iq2xs[q >> 9];
            for (int j = 0; j < 8; ++j) {
                y[j] = db[l / 2] * static_cast<float>(grid[j]) * sign_of(signs, j);
            }
            y += 8;
        }
    }
}

void iq2_s(const std::uint8_t* x, float* y) {
    const float d              = half(x);
    const std::uint8_t* qs     = x + 2;
    const std::uint8_t* qh     = x + 66;
    const std::uint8_t* scales = x + 74;
    const std::uint8_t* signs  = qs + kQK / 8;
    for (int ib32 = 0; ib32 < kQK / 32; ++ib32) {
        float db[2];
        db[0] = d * (0.5F + static_cast<float>(scales[ib32] & 0xf)) * 0.25F;
        db[1] = d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F;
        for (int l = 0; l < 4; ++l) {
            const float dl = db[l / 2];
            const auto* grid = reinterpret_cast<const std::uint8_t*>(
                iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
            for (int j = 0; j < 8; ++j) {
                y[j] = dl * static_cast<float>(grid[j]) * sign_of(signs[l], j);
            }
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

void iq3_xxs(const std::uint8_t* x, float* y) {
    const float d                    = half(x);
    const std::uint8_t* qs           = x + 2;
    const std::uint8_t* scales_signs = qs + kQK / 4;
    for (int ib32 = 0; ib32 < kQK / 32; ++ib32) {
        const std::uint32_t aux32 = u32(scales_signs + 4 * ib32);
        const float db            = d * (0.5F + static_cast<float>(aux32 >> 28)) * 0.5F;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
            const auto* grid1 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 0]);
            const auto* grid2 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * static_cast<float>(grid1[j]) * sign_of(signs, j + 0);
                y[j + 4] = db * static_cast<float>(grid2[j]) * sign_of(signs, j + 4);
            }
            y += 8;
        }
        qs += 8;
    }
}

void iq3_s(const std::uint8_t* x, float* y) {
    const float d              = half(x);
    const std::uint8_t* qs     = x + 2;
    const std::uint8_t* qh     = x + 66;
    const std::uint8_t* signs  = x + 74;
    const std::uint8_t* scales = x + 106;
    for (int ib32 = 0; ib32 < kQK / 32; ib32 += 2) {
        const float db1 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] & 0xf));
        const float db2 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] >> 4));
        for (int l = 0; l < 4; ++l) {
            const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
            const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db1 * static_cast<float>(grid1[j]) * sign_of(signs[l], j + 0);
                y[j + 4] = db1 * static_cast<float>(grid2[j]) * sign_of(signs[l], j + 4);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; ++l) {
            const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
            const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db2 * static_cast<float>(grid1[j]) * sign_of(signs[l], j + 0);
                y[j + 4] = db2 * static_cast<float>(grid2[j]) * sign_of(signs[l], j + 4);
            }
            y += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
    }
}

void iq1_s(const std::uint8_t* x, float* y) {
    const float d          = half(x);
    const std::uint8_t* qs = x + 2;
    const std::uint8_t* qh = x + 34; // 8 uint16
    for (int ib = 0; ib < kQK / 32; ++ib) {
        const std::uint16_t h = u16(qh + 2 * ib);
        const float dl        = d * static_cast<float>(2 * ((h >> 12) & 7) + 1);
        const float delta     = h & 0x8000 ? -kIq1Delta : kIq1Delta;
        for (int l = 0; l < 4; ++l) {
            const auto* grid = reinterpret_cast<const std::int8_t*>(
                iq1s_grid + (qs[l] | (((h >> 3 * l) & 7) << 8)));
            for (int j = 0; j < 8; ++j) { y[j] = dl * (static_cast<float>(grid[j]) + delta); }
            y += 8;
        }
        qs += 4;
    }
}

void iq1_m(const std::uint8_t* x, float* y) {
    const std::uint8_t* qs = x;
    const std::uint8_t* qh = x + 32;
    std::uint16_t sc[4];
    std::memcpy(sc, x + 48, 8);
    const std::uint16_t scale = static_cast<std::uint16_t>(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = f16_to_f32(scale);
    for (int ib = 0; ib < kQK / 32; ++ib) {
        const float dl1 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        const float dl2 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        std::uint16_t idx[4];
        float delta[4];
        idx[0]   = static_cast<std::uint16_t>(qs[0] | ((qh[0] << 8) & 0x700));
        idx[1]   = static_cast<std::uint16_t>(qs[1] | ((qh[0] << 4) & 0x700));
        idx[2]   = static_cast<std::uint16_t>(qs[2] | ((qh[1] << 8) & 0x700));
        idx[3]   = static_cast<std::uint16_t>(qs[3] | ((qh[1] << 4) & 0x700));
        delta[0] = qh[0] & 0x08 ? -kIq1Delta : kIq1Delta;
        delta[1] = qh[0] & 0x80 ? -kIq1Delta : kIq1Delta;
        delta[2] = qh[1] & 0x08 ? -kIq1Delta : kIq1Delta;
        delta[3] = qh[1] & 0x80 ? -kIq1Delta : kIq1Delta;
        for (int l = 0; l < 2; ++l) {
            const auto* grid = reinterpret_cast<const std::int8_t*>(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) { y[j] = dl1 * (static_cast<float>(grid[j]) + delta[l]); }
            y += 8;
        }
        for (int l = 2; l < 4; ++l) {
            const auto* grid = reinterpret_cast<const std::int8_t*>(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) { y[j] = dl2 * (static_cast<float>(grid[j]) + delta[l]); }
            y += 8;
        }
        qs += 4;
        qh += 2;
    }
}

void iq4_nl(const std::uint8_t* x, float* y) {
    const float d = half(x);
    for (int j = 0; j < 16; ++j) {
        y[j]      = d * static_cast<float>(kvalues_iq4nl[x[2 + j] & 0xf]);
        y[j + 16] = d * static_cast<float>(kvalues_iq4nl[x[2 + j] >> 4]);
    }
}

void iq4_xs(const std::uint8_t* x, float* y) {
    const float d                = half(x);
    const std::uint16_t scales_h = u16(x + 2);
    const std::uint8_t* scales_l = x + 4;
    const std::uint8_t* qs       = x + 8;
    for (int ib = 0; ib < kQK / 32; ++ib) {
        const int ls   = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j + 0]  = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0xf]);
            y[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
        }
        y += 32;
        qs += 16;
    }
}

using BlockFn = void (*)(const std::uint8_t*, float*);

struct Entry {
    std::uint32_t id;
    Traits traits;
    BlockFn decode; // null: plain values (converted directly) or not decodable
};

// ggml_type_traits: names, block sizes and type sizes (ggml/src/ggml.c).
constexpr Entry kTypes[] = {
    {F32, {"f32", 1, 4, true}, nullptr},
    {F16, {"f16", 1, 2, true}, nullptr},
    {Q4_0, {"q4_0", 32, 18, true}, q4_0},
    {Q4_1, {"q4_1", 32, 20, true}, q4_1},
    {Q5_0, {"q5_0", 32, 22, true}, q5_0},
    {Q5_1, {"q5_1", 32, 24, true}, q5_1},
    {Q8_0, {"q8_0", 32, 34, true}, q8_0},
    {Q8_1, {"q8_1", 32, 36, false}, nullptr},
    {Q2_K, {"q2_K", 256, 84, true}, q2_k},
    {Q3_K, {"q3_K", 256, 110, true}, q3_k},
    {Q4_K, {"q4_K", 256, 144, true}, q4_k},
    {Q5_K, {"q5_K", 256, 176, true}, q5_k},
    {Q6_K, {"q6_K", 256, 210, true}, q6_k},
    {Q8_K, {"q8_K", 256, 292, false}, nullptr},
    {IQ2_XXS, {"iq2_xxs", 256, 66, true}, iq2_xxs},
    {IQ2_XS, {"iq2_xs", 256, 74, true}, iq2_xs},
    {IQ3_XXS, {"iq3_xxs", 256, 98, true}, iq3_xxs},
    {IQ1_S, {"iq1_s", 256, 50, true}, iq1_s},
    {IQ4_NL, {"iq4_nl", 32, 18, true}, iq4_nl},
    {IQ3_S, {"iq3_s", 256, 110, true}, iq3_s},
    {IQ2_S, {"iq2_s", 256, 82, true}, iq2_s},
    {IQ4_XS, {"iq4_xs", 256, 136, true}, iq4_xs},
    {I8, {"i8", 1, 1, true}, nullptr},
    {I16, {"i16", 1, 2, true}, nullptr},
    {I32, {"i32", 1, 4, true}, nullptr},
    {I64, {"i64", 1, 8, true}, nullptr},
    {F64, {"f64", 1, 8, true}, nullptr},
    {IQ1_M, {"iq1_m", 256, 56, true}, iq1_m},
    {BF16, {"bf16", 1, 2, true}, nullptr},
    {TQ1_0, {"tq1_0", 256, 54, true}, tq1_0},
    {TQ2_0, {"tq2_0", 256, 66, true}, tq2_0},
    {MXFP4, {"mxfp4", 32, 17, true}, mxfp4},
    {NVFP4, {"nvfp4", 64, 36, true}, nvfp4},
    {Q1_0, {"q1_0", 128, 18, true}, q1_0},
    {Q2_0, {"q2_0", 64, 18, true}, q2_0},
};

const Entry* entry(std::uint32_t type) noexcept {
    for (const auto& e : kTypes) {
        if (e.id == type) { return &e; }
    }
    return nullptr;
}

} // namespace

std::optional<Traits> traits(std::uint32_t type) noexcept {
    const auto* e = entry(type);
    return e ? std::optional<Traits>(e->traits) : std::nullopt;
}

std::string type_name(std::uint32_t type) {
    const auto* e = entry(type);
    if (!e) { return "type" + std::to_string(type); }
    return ascii_lower(e->traits.name);
}

bool is_plain(std::uint32_t type) noexcept {
    switch (type) {
    case F32:
    case F16:
    case BF16:
    case F64:
    case I8:
    case I16:
    case I32:
    case I64:
        return true;
    default:
        return false;
    }
}

std::uint64_t bytes_for(std::uint32_t type, std::uint64_t elements) {
    const auto* e = entry(type);
    if (!e) { throw std::runtime_error("unknown ggml type " + std::to_string(type)); }
    if (elements % e->traits.block) {
        throw std::runtime_error(type_name(type) + ": " + std::to_string(elements) +
                                 " values do not fill whole blocks of " +
                                 std::to_string(e->traits.block));
    }
    return elements / e->traits.block * e->traits.bytes;
}

void dequantize(std::uint32_t type, const std::byte* data, std::uint64_t elements, float* out) {
    const auto* e = entry(type);
    if (!e || !e->traits.dequantizable) {
        throw std::runtime_error("the converter does not decode ggml type " + type_name(type));
    }
    const auto* p = reinterpret_cast<const std::uint8_t*>(data);
    switch (type) {
    case F32:
        std::memcpy(out, p, elements * 4);
        return;
    case F16:
        for (std::uint64_t i = 0; i < elements; ++i) { out[i] = half(p + 2 * i); }
        return;
    case BF16:
        for (std::uint64_t i = 0; i < elements; ++i) { out[i] = bf16_to_f32(u16(p + 2 * i)); }
        return;
    case F64:
        for (std::uint64_t i = 0; i < elements; ++i) {
            double v;
            std::memcpy(&v, p + 8 * i, 8);
            out[i] = static_cast<float>(v);
        }
        return;
    case I8:
        for (std::uint64_t i = 0; i < elements; ++i) {
            out[i] = static_cast<float>(static_cast<std::int8_t>(p[i]));
        }
        return;
    case I16:
        for (std::uint64_t i = 0; i < elements; ++i) {
            out[i] = static_cast<float>(static_cast<std::int16_t>(u16(p + 2 * i)));
        }
        return;
    case I32:
        for (std::uint64_t i = 0; i < elements; ++i) {
            out[i] = static_cast<float>(static_cast<std::int32_t>(u32(p + 4 * i)));
        }
        return;
    case I64:
        for (std::uint64_t i = 0; i < elements; ++i) {
            std::int64_t v;
            std::memcpy(&v, p + 8 * i, 8);
            out[i] = static_cast<float>(v);
        }
        return;
    default:
        break;
    }
    const auto block = e->traits.block;
    if (elements % block) {
        throw std::runtime_error(type_name(type) + ": partial block");
    }
    for (std::uint64_t b = 0; b < elements / block; ++b) {
        e->decode(p + b * e->traits.bytes, out + b * block);
    }
}

void nvfp4_block_natural(const std::uint8_t* block, std::uint8_t* codes, std::uint8_t* scales) {
    for (int s = 0; s < 4; ++s) {
        scales[s]              = block[s] == 0x7F ? 0 : static_cast<std::uint8_t>(block[s] & 0x7F);
        const std::uint8_t* qs = block + 4 + 8 * s;
        std::uint8_t element[16];
        for (int j = 0; j < 8; ++j) {
            element[j]     = qs[j] & 0x0F;
            element[j + 8] = qs[j] >> 4;
        }
        for (int i = 0; i < 8; ++i) {
            codes[8 * s + i] = static_cast<std::uint8_t>(element[2 * i] | (element[2 * i + 1] << 4));
        }
    }
}

std::uint8_t mxfp4_block_natural(const std::uint8_t* block, std::uint8_t* codes) {
    std::uint8_t element[32];
    for (int j = 0; j < 16; ++j) {
        element[j]      = block[1 + j] & 0x0F;
        element[j + 16] = block[1 + j] >> 4;
    }
    for (int i = 0; i < 16; ++i) {
        codes[i] = static_cast<std::uint8_t>(element[2 * i] | (element[2 * i + 1] << 4));
    }
    return block[0];
}

} // namespace ninferno::convert::ggml
