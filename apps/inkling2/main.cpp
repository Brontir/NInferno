// ninferno-inkling2 <artifact> --ids 1,2,3 [--generate N] [--dump]: run an Inkling 2 artifact on
// a token sequence through the family's plain forward (models/inkling2/target.h), print the top
// logits of the last token, greedily continue N tokens, and (--dump) every visit's output stream
// summary (sum and the first values of the last token) for a comparison with llama.cpp's
// llama-eval-callback (l_out-<slot>, result_norm).

#include "core/device.h"
#include "models/inkling2/model.h"
#include "models/inkling2/target.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::models::inkling2;

namespace {

std::vector<std::int32_t> parse_ids(const std::string& text) {
    std::vector<std::int32_t> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (!item.empty()) { out.push_back(std::stoi(item)); }
    }
    return out;
}

void top(const std::vector<float>& logits, int k) {
    std::vector<int> index(logits.size());
    std::iota(index.begin(), index.end(), 0);
    std::partial_sort(index.begin(), index.begin() + k, index.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("top:");
    for (int i = 0; i < k; ++i) { std::printf(" %d:%.4f", index[i], logits[index[i]]); }
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: ninferno-inkling2 <artifact> --ids 1,2,3 [--generate N] [--dump]\n");
        return 2;
    }
    std::vector<std::int32_t> ids;
    int generate = 0;
    bool dump    = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ids" && i + 1 < argc) {
            ids = parse_ids(argv[++i]);
        } else if (a == "--generate" && i + 1 < argc) {
            generate = std::stoi(argv[++i]);
        } else if (a == "--dump") {
            dump = true;
        }
    }
    try {
        DeviceContext device(0);
        auto model    = Model::load(argv[1], device);
        const auto& c = model->config();
        std::printf("loaded: H %d layers %d visits %d vocab %d/%d experts %d top %d\n",
                    c.hidden_size, c.layers, c.visits(), c.vocab_size, c.vocab_rows, c.experts,
                    c.experts_per_tok);
        Target target(*model, 4096,
                      std::max<std::int32_t>(64, static_cast<std::int32_t>(ids.size())));
        if (dump) {
            const auto& visits = target.visits();
            target.observe     = [&](const char* name, std::int32_t visit, const Tensor& value) {
                const std::size_t n = std::size_t(value.numel());
                std::vector<std::uint16_t> bits(n);
                cudaMemcpy(bits.data(), value.data, n * 2, cudaMemcpyDeviceToHost);
                double sum = 0;
                std::vector<float> v(n);
                for (std::size_t i = 0; i < n; ++i) {
                    const std::uint32_t w = std::uint32_t(bits[i]) << 16;
                    std::memcpy(&v[i], &w, 4);
                    sum += v[i];
                }
                const std::size_t last = n - std::size_t(value.ne[0]);
                std::printf(
                    "%s visit %d slot %d layer %d sum %.6f last[0..3] %.5f %.5f %.5f %.5f\n", name,
                    visit, visit >= 0 ? visits[std::size_t(visit)].slot : -1,
                    visit >= 0 ? visits[std::size_t(visit)].layer : -1, sum, v[last], v[last + 1],
                    v[last + 2], v[last + 3]);
            };
        }
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        auto logits = target.forward(ids, s);
        top(logits, 10);
        target.observe = nullptr;
        std::printf("generated:");
        const auto start = std::chrono::steady_clock::now();
        for (int g = 0; g < generate; ++g) {
            const auto next = static_cast<std::int32_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::printf(" %d", next);
            std::fflush(stdout);
            const std::int32_t one[1] = {next};
            logits                    = target.forward(one, s);
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::printf("\n");
        if (generate > 0) {
            std::printf("decode: %d tokens, %.3f ms a token (%.1f tok/s), one sequence, no graph\n",
                        generate, ms / generate, 1000.0 * generate / ms);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninferno-inkling2: %s\n", e.what());
        return 1;
    }
    return 0;
}
