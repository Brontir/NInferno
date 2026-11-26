// burst_wait.cu (lab r38): what one extra dependent L2 round trip costs a warp's trip while every SM streams
// weights over the bus, the vec kernel's shape (#1236): a warp issues a burst of weight loads (N LDG.128 a lane,
// fresh addresses: DRAM), consumes it, then makes E dependent L2-resident loads (a pointer chase, one coalesced 128 B row a warp a step, read
// with ld.global.cg), then the next trip. Every SM holds C CTAs of 4 warps. The slope of the trip time
// over E is the extra burst's cost under load; 0 where other warps' bytes hide it.
// Build: nvcc -O3 -arch=sm_XX -o burst_wait burst_wait.cu. Run: ./burst_wait [N] [C...]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CHASE_ROWS 4096u
#ifndef ACT_ROWS  // -DACT: 28 rows of 1 KB = 28 KB, about vec's T 6 activation (4.6 KB x T); -DACT_ROWS=12u: a smaller table
#define ACT_ROWS 28u
#endif
#ifndef PITCHW  // -DNARROW: the lane pitch in words (9: vec's 36 B q8_1 block; 32: a line a lane, the tag-bound step)
#define PITCHW 9u
#endif
#define ACT_ROW_WORDS (PITCHW * 32u > 256u ? PITCHW * 32u : 256u)
#ifdef ROW16  // -DROW16: a chase step reads a 512 B row a warp (16 B a lane), as vec's activation loads do
#define ROW_WORDS 128u
#else
#define ROW_WORDS 32u
#endif
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint32_t ld_cg(const uint32_t* p) {
#if defined(ROW16) && defined(CA)  // -DCA: the default caching (L1), as vec's activation loads
  uint32_t a, b, c, d;
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p));
  return a | ((b ^ c ^ d) & 0u);
#elif defined(ROW16)
  uint32_t a, b, c, d;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p));
  return a | ((b ^ c ^ d) & 0u);
#else
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
#endif
}

template <int N>
__global__ void __launch_bounds__(128) burst_kernel(const int4* __restrict__ w, size_t w_int4_per_warp, int trips,
                                                    const uint32_t* __restrict__ chase, int extra, uint32_t* out) {
  const int lane = threadIdx.x & 31;
  const size_t warp = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int4* p = w + warp * w_int4_per_warp + lane;
#if defined(ACT)
  uint32_t acc = 0, row = (uint32_t)(warp % ACT_ROWS);  // its own chain over the shared table
#elif defined(CA)  // the CTA's 4 warps walk one chain (vec's warps share the activation rows: L1 hits after the first)
  uint32_t acc = 0, row = (uint32_t)(blockIdx.x * 97) & (CHASE_ROWS - 1);
#else
  uint32_t acc = 0, row = (uint32_t)(warp * 97) & (CHASE_ROWS - 1);  // warp-uniform: a coalesced 128 B row
#endif
  for (int t = 0; t < trips; ++t) {
    int4 v[N];
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = __ldcs(p + i * 32);  // one burst: N x 512 B a warp, streamed (evict-first)
    p += N * 32;
#pragma unroll
    for (int i = 0; i < N; ++i) acc += v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    uint32_t zero;  // 0, opaque to the compiler: the first chase load waits for the burst's values
    asm volatile("and.b32 %0, %1, 0;" : "=r"(zero) : "r"(acc));
    row += zero;
#if defined(ACT) && defined(ACT2)  // -DACT -DACT2 (r47): two interleaved 16 B loads a step over the same 8 lines
    // (offsets 16 and 0 at a 32 B pitch: the 1 KB span read whole); a line charged once a burst costs ACT's step here
    for (int e = 0; e < extra; ++e) {
      uint32_t a, b, c, d, f, g, h, k;
      asm("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
          : "l"(chase + row * ACT_ROW_WORDS + 4u + lane * 8u));
      asm("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(f), "=r"(g), "=r"(h), "=r"(k)
          : "l"(chase + row * ACT_ROW_WORDS + lane * 8u));
      row = a;
      acc += a + (b ^ c ^ d) + (f ^ g ^ h ^ k);
    }
#elif defined(ACT) && defined(NARROW4)  // -DACT -DNARROW4 (r47): a 4 B load in ACT's 32 B layout (8 lines, 128 B)
    for (int e = 0; e < extra; ++e) {
      uint32_t a;
      asm("ld.global.ca.u32 %0, [%1];" : "=r"(a) : "l"(chase + row * ACT_ROW_WORDS + 4u + lane * 8u));
      row = a;
      acc += a;
    }
#elif defined(ACT) && defined(NARROW)  // -DACT -DNARROW (r42): 4 B a lane at vec's 36 B q8_1 pitch from offset 4
    // (9-10 lines a request), the scale / narrow-load shape TU102's queue may serve at its own rate
    for (int e = 0; e < extra; ++e) {
      uint32_t a;
      asm("ld.global.ca.u32 %0, [%1];" : "=r"(a) : "l"(chase + row * ACT_ROW_WORDS + 1u + lane * PITCHW));
      row = a;
      acc += a;
    }
#elif defined(ACT)  // -DACT (r39): vec's activation request shape on an L1-resident table every warp of the SM shares: a
    // step is 32 lanes x 16 B at a 32 B pitch from offset 16 (8 lines, 32 sectors a request), each warp on its own chain
    for (int e = 0; e < extra; ++e) {
      uint32_t a, b, c, d;
      asm("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
          : "l"(chase + row * ACT_ROW_WORDS + 4u + lane * 8u));
      row = a;
      acc += a + (b ^ c ^ d);  // every word used: the load stays 16 B a lane
    }
#else
    for (int e = 0; e < extra; ++e) {  // E dependent L2 round trips, one coalesced 128 B row a warp each
      row = ld_cg(chase + row * ROW_WORDS + lane * (ROW_WORDS / 32));  // every lane of row r holds the same next row
      acc += row;
    }
#endif
  }
  if (acc == 0x12345678u) out[warp] = acc;
}

template <int N>
float run(int sms, int ctas_sm, int extra, int trips, const int4* w, size_t per_warp, const uint32_t* chase,
          uint32_t* out) {
  const int grid = sms * ctas_sm;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  burst_kernel<N><<<grid, 128>>>(w, per_warp, trips, chase, extra, out);  // warm
  CK(cudaEventRecord(a));
  const int reps = 3;
  for (int r = 0; r < reps; ++r) burst_kernel<N><<<grid, 128>>>(w, per_warp, trips, chase, extra, out);
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  CK(cudaGetLastError());
  return ms / reps;
}

int main(int argc, char** argv) {
  const int n_arg = argc > 1 ? atoi(argv[1]) : 2;
  std::vector<int> cs;
  for (int i = 2; i < argc; ++i) cs.push_back(atoi(argv[i]));
  if (cs.empty()) cs = {5, 6, 7, 8};
  cudaDeviceProp pr;
  CK(cudaGetDeviceProperties(&pr, 0));
  const int sms = pr.multiProcessorCount;
  int clk_khz = 0, mem_khz = 0, bus_bits = 0;
  CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  CK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
  CK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
  printf("card %s\nsms %d\nl2_bytes %d\nprobe burst_wait N %d (burst %d B a warp)\n", pr.name, sms, pr.l2CacheSize,
         n_arg, n_arg * 512);
  printf("columns ctas_sm warps_sm extra trips us gbps us_per_trip clk_per_trip_at_boost\n");
  const int max_c = 8;
  const int trips = 256;
  const size_t per_warp = size_t(trips) * n_arg * 32 + 64;  // int4 a warp
  const size_t warps = size_t(sms) * max_c * 4;
  int4* w;
  CK(cudaMalloc(&w, warps * per_warp * sizeof(int4)));
  CK(cudaMemset(w, 1, warps * per_warp * sizeof(int4)));
#ifdef ACT
  std::vector<uint32_t> h(ACT_ROWS * ACT_ROW_WORDS);
  for (uint32_t r = 0; r < ACT_ROWS; ++r)
    for (uint32_t l = 0; l < ACT_ROW_WORDS; ++l) h[r * ACT_ROW_WORDS + l] = (r * 11u + 5u) % ACT_ROWS;
#else
  std::vector<uint32_t> h(CHASE_ROWS * ROW_WORDS);  // 512 KB (2 MB with ROW16): L2-resident on every card
  for (uint32_t r = 0; r < CHASE_ROWS; ++r)
    for (uint32_t l = 0; l < ROW_WORDS; ++l) h[r * ROW_WORDS + l] = (uint32_t)((r * 2654435761u) >> 20) & (CHASE_ROWS - 1);
#endif
  uint32_t *chase, *out;
  CK(cudaMalloc(&chase, h.size() * 4));
  CK(cudaMemcpy(chase, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&out, warps * 4));
#if defined(LONG256)  // r47: to E 256 (NARROW's knee)
  const int extras[] = {0, 32, 64, 96, 128, 192, 256};
#elif defined(ACT) || defined(LONGE)  // LONGE: the cg chase to E 64 as well (r43)
  const int extras[] = {0, 4, 8, 16, 24, 32, 48, 64};  // r40: to the knee of the L1 queue
#else
  const int extras[] = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16};
#endif
  for (int c : cs) {
    if (c > max_c) continue;
    for (int e : extras) {
      float ms = n_arg == 1   ? run<1>(sms, c, e, trips, w, per_warp, chase, out)
                 : n_arg == 2 ? run<2>(sms, c, e, trips, w, per_warp, chase, out)
                 : n_arg == 3 ? run<3>(sms, c, e, trips, w, per_warp, chase, out)
                              : run<4>(sms, c, e, trips, w, per_warp, chase, out);
      const double bytes = double(sms) * c * 4 * trips * n_arg * 512.0;
      const double us = ms * 1e3;
      printf("row %d %d %d %d %.2f %.1f %.4f %.0f\n", c, c * 4, e, trips, us, bytes / (us * 1e3), us / trips,
             us / trips * clk_khz / 1e3);
    }
  }
  printf("note boost clock %d MHz (cudaDevAttrClockRate), peak bus %.0f GB/s\n", clk_khz / 1000,
         2.0 * mem_khz * 1e3 * bus_bits / 8 / 1e9);
  return 0;
}
