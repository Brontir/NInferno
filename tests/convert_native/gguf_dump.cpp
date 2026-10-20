// Test helper: parse a GGUF file and dump what the converter reads from it.
//
//   ninferno_gguf_dump FILE OUTDIR
//
// OUTDIR/metadata.json holds the parsed metadata and tensor infos; OUTDIR/<index>.f32 the
// dequantized values of every tensor the converter decodes, as little-endian binary32.

#include "convert/ggml_quants.h"
#include "convert/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <vector>

using namespace ninferno::convert;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: ninferno_gguf_dump FILE OUTDIR\n");
        return 2;
    }
    try {
        const int fd = open(argv[1], O_RDONLY);
        if (fd < 0) { throw std::runtime_error("cannot open file"); }
        struct stat st{};
        fstat(fd, &st);
        const auto size = static_cast<std::size_t>(st.st_size);
        void* base      = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) { throw std::runtime_error("cannot map file"); }
        const auto* data = static_cast<const std::byte*>(base);
        const auto file  = parse_gguf(data, size, argv[1]);
        nlohmann::json out;
        out["version"]     = file.version;
        out["alignment"]   = file.alignment;
        out["data_offset"] = file.data_offset;
        out["metadata"]    = file.metadata;
        out["tensors"]     = nlohmann::json::array();
        for (std::size_t i = 0; i < file.tensors.size(); ++i) {
            const auto& t = file.tensors[i];
            out["tensors"].push_back({{"name", t.name},
                                      {"ne", t.ne},
                                      {"type", t.type},
                                      {"offset", t.offset},
                                      {"bytes", t.bytes}});
            const auto traits = ggml::traits(t.type);
            if (!traits || !traits->dequantizable) { continue; }
            std::vector<float> values(t.elements());
            ggml::dequantize(t.type, data + file.data_offset + t.offset, t.elements(),
                             values.data());
            std::ofstream f(std::string(argv[2]) + "/" + std::to_string(i) + ".f32",
                            std::ios::binary);
            f.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * 4));
        }
        std::ofstream(std::string(argv[2]) + "/metadata.json") << out.dump();
        munmap(base, size);
        close(fd);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninferno_gguf_dump: %s\n", error.what());
        return 1;
    }
    return 0;
}
