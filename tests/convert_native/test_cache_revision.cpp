// Which cached artifacts revision 4 converts again: a qwen3_5_published artifact whose DFlash
// drafter kept its attention projections unquantized; not a text-only one, not one whose drafter
// is already Q8, not another recipe's.

#include "convert/cache.h"
#include "core/sys.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << std::endl;
        ++failures;
    }
}

// An artifact at `revision` whose report binds `query` (a parameter name) in `format`.
std::filesystem::path artifact(const std::filesystem::path& dir, const std::string& name,
                               int revision, const std::string& query, const std::string& format) {
    const auto path = dir / (name + ".ninferno");
    std::ofstream(path) << "x";
    std::ofstream(path.string() + ".revision") << revision << "\n";
    std::ofstream(path.string() + ".conversion.json")
        << R"({"components": {"text": {}}, "methods": [)"
        << R"({"parameters": ["text/layers/0/attention/query"], "format": "nvfp4"},)"
        << R"({"parameters": [")" << query << R"("], "format": ")" << format << R"("}]})";
    return path;
}

} // namespace

int main() {
    namespace fs   = std::filesystem;
    const auto dir = fs::temp_directory_path() /
                     ("ninferno_cache_revision_" + std::to_string(ninferno::sys::pid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    using ninferno::convert::cache_current;
    const std::string published = "qwen3_5_published";
    check(cache_current(artifact(dir, "text", 3, "text/output_head", "bf16"), published),
          "a text-only qwen3_5_published artifact at revision 3 is current");
    const auto bf16 = artifact(dir, "bf16", 3, "dflash/layers/0/attention/query", "bf16");
    check(!cache_current(bf16, published) && ninferno::convert::unquantized_drafter_attention(bf16),
          "a BF16 DFlash drafter at revision 3 is converted again");
    check(!cache_current(artifact(dir, "bf16-2", 3, "dflash2/layers/1/attention/query", "bf16"),
                         published),
          "a BF16 DFlash2 drafter too");
    check(cache_current(artifact(dir, "q8", 3, "dflash2/layers/0/attention/query", "q8_g32_fp16"),
                        published),
          "a drafter already in Q8 is current");
    check(cache_current(bf16, "qwen3_6_35b_a3b_nvfp4"), "another recipe's artifact is current");
    check(cache_current(artifact(dir, "new", 4, "dflash/layers/0/attention/query", "bf16"),
                        published),
          "revision 4 is current");
    check(!cache_current(artifact(dir, "old", 2, "text/output_head", "bf16"), published),
          "revision 2 is not");
    fs::remove_all(dir);
    if (failures == 0) { std::cout << "cache revision: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
