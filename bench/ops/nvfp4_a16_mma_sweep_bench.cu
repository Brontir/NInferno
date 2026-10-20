// Sweep of the A16 tensor-core schedules on the Qwen3.5-27B NVFP4 shapes (nvfp4_a16_mma_sweep.cuh).
//
//   ninferno_nvfp4_a16_mma_sweep_bench [--shape n34816_k5120|...|all] [--tokens 1,2,...]
//                                    [--warmup N] [--repeat N]
//
// Output: CSV on stdout, one row per (shape, candidate, T); candidate linear_a16 is the Linear A16
// shape route at the same T.

#include "nvfp4_a16_mma_sweep.cuh"

#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace ninferno::bench::a16_sweep;

int main(int argc, char** argv) {
    try {
        Options options;
        std::string shape = "all";
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument("missing value"); }
                return argv[++i];
            };
            if (argument == "--shape") {
                shape = next();
            } else if (argument == "--tokens") {
                options.tokens.clear();
                std::stringstream list(next());
                for (std::string item; std::getline(list, item, ',');) {
                    options.tokens.push_back(std::stoi(item));
                }
            } else if (argument == "--warmup") {
                options.warmup = std::stoi(next());
            } else if (argument == "--repeat") {
                options.repeat = std::stoi(next());
            } else {
                throw std::invalid_argument("unknown argument " + std::string(argument));
            }
        }
        print_header();
        const bool all = shape == "all";
        if (all || shape == "n14336_k5120") { sweep_n14336_k5120(options); }
        if (all || shape == "n16384_k5120") { sweep_n16384_k5120(options); }
        if (all || shape == "n34816_k5120") { sweep_n34816_k5120(options); }
        if (all || shape == "n5120_k6144") { sweep_n5120_k6144(options); }
        if (all || shape == "n5120_k17408") { sweep_n5120_k17408(options); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
