// fetch_gguf_head() for fetch_test.py: the first bytes of the GGUF file a fetch would choose.
//
//     fetch_head_probe SPEC QUANT BYTES CACHE [BEGIN]
//
// (BEGIN: the window from that offset, as gguf_quants_head_bytes reads it; "first" then holds its
// first 8 bytes in hex.)
// Prints {"commit", "path", "size", "bytes", "cached", "offline", "parts"} as one JSON line (QUANT
// "--include=GLOB,GLOB": the files those globs choose instead of a quantization); an error on
// stderr, exit 1. With QUANT --list, prints list_quants(SPEC) instead: [{"quant", "format",
// "bytes", "mtp_bytes", "drafter", "files"}]. With QUANT --holds or --holds-vision (BYTES the
// quantization, CACHE a snapshot directory), prints whether snapshot_holds_model() finds the model
// there (with its Vision projector).
#include "convert/fetch.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: fetch_head_probe SPEC QUANT BYTES CACHE\n");
        return 2;
    }
    try {
        ninferno::convert::FetchOptions options;
        options.cache = argv[4];
        if (std::string(argv[2]) == "--list") {
            nlohmann::json rows = nlohmann::json::array();
            for (const auto& q : ninferno::convert::list_quants(argv[1], options).quants) {
                rows.push_back({{"quant", q.quant},
                                {"format", q.format},
                                {"bytes", q.bytes},
                                {"mtp_bytes", q.mtp_bytes},
                                {"drafter", q.drafter},
                                {"files", q.paths()}});
            }
            std::printf("%s\n", rows.dump().c_str());
            return 0;
        }
        // --heads=Q,Q TABLE_END CACHE HIDDEN:VOCAB[:tied]: gguf_quants_head_bytes as {quant: bytes}.
        if (const std::string heads = argv[2]; heads.rfind("--heads=", 0) == 0 && argc == 6) {
            std::vector<std::string> quants;
            for (std::size_t at = 8; at <= heads.size();) {
                const std::size_t comma = heads.find(',', at);
                const std::size_t end   = comma == std::string::npos ? heads.size() : comma;
                quants.push_back(heads.substr(at, end - at));
                at = end + 1;
            }
            const std::string shape = argv[5];
            const auto colon        = shape.find(':');
            const auto hidden       = std::stoull(shape.substr(0, colon));
            const auto vocab        = std::stoull(shape.substr(colon + 1));
            const bool tied         = shape.find(":tied") != std::string::npos;
            nlohmann::json out = ninferno::convert::gguf_quants_head_bytes(
                argv[1], options, quants, std::stoull(argv[3]), hidden, vocab, tied);
            std::printf("%s\n", out.dump().c_str());
            return 0;
        }
        if (std::string(argv[2]) == "--holds" || std::string(argv[2]) == "--holds-vision") {
            options.quant    = argv[3];
            const bool holds = ninferno::convert::snapshot_holds_model(
                argv[4], options, argv[1], std::string(argv[2]) == "--holds-vision");
            std::printf("%s\n", holds ? "true" : "false");
            return 0;
        }
        if (const std::string quant = argv[2]; quant.rfind("--include=", 0) == 0) {
            std::string globs = quant.substr(10);
            for (std::size_t at = 0; at <= globs.size();) {
                const std::size_t comma = globs.find(',', at);
                const std::size_t end   = comma == std::string::npos ? globs.size() : comma;
                options.include.push_back(globs.substr(at, end - at));
                at = end + 1;
            }
        } else {
            options.quant = quant;
        }
        const std::uint64_t begin = argc == 6 ? std::stoull(argv[5]) : 0;
        const auto head =
            ninferno::convert::fetch_gguf_head(argv[1], options, std::stoull(argv[3]), 0, begin);
        std::string first;
        for (std::size_t i = 0; i < head.bytes.size() && i < 8; ++i) {
            char hex[3];
            std::snprintf(hex, sizeof hex, "%02x", static_cast<unsigned char>(head.bytes[i]));
            first += hex;
        }
        const nlohmann::json out{{"commit", head.commit},        {"path", head.path},
                                 {"size", head.size},            {"bytes", head.bytes.size()},
                                 {"cached", head.cached},        {"offline", head.offline},
                                 {"parts", head.parts},          {"first", first}};
        std::printf("%s\n", out.dump().c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "fetch_head_probe: %s\n", error.what());
        return 1;
    }
}
