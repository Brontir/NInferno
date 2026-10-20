// fetch_gguf_head() for fetch_test.py: the first bytes of the GGUF file a fetch would choose.
//
//     fetch_head_probe SPEC QUANT BYTES CACHE
//
// Prints {"commit", "path", "size", "bytes", "cached", "offline"} as one JSON line; an error on
// stderr, exit 1.
#include "convert/fetch.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: fetch_head_probe SPEC QUANT BYTES CACHE\n");
        return 2;
    }
    try {
        ninferno::convert::FetchOptions options;
        options.quant   = argv[2];
        options.cache   = argv[4];
        const auto head =
            ninferno::convert::fetch_gguf_head(argv[1], options, std::stoull(argv[3]));
        const nlohmann::json out{{"commit", head.commit},        {"path", head.path},
                                 {"size", head.size},            {"bytes", head.bytes.size()},
                                 {"cached", head.cached},        {"offline", head.offline}};
        std::printf("%s\n", out.dump().c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "fetch_head_probe: %s\n", error.what());
        return 1;
    }
}
