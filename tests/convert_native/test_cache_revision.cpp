// Which cached artifacts revisions 4 and 5 convert again: an artifact of any recipe whose DFlash
// companion drafter kept its attention projections unquantized (revision 4: qwen3_5_published;
// revision 5: qwen3_5_bf16, the dense Qwen3.5 GGUF default, too); not a text-only one, not one
// whose drafter is already Q8.

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
    const std::string bf16_recipe = "qwen3_5_bf16";
    check(cache_current(artifact(dir, "text", 3, "text/output_head", "bf16"), published),
          "a text-only qwen3_5_published artifact at revision 3 is current");
    check(cache_current(artifact(dir, "text-bf16", 4, "text/output_head", "bf16"), bf16_recipe),
          "a text-only qwen3_5_bf16 artifact at revision 4 is current");
    const auto bf16 = artifact(dir, "bf16", 3, "dflash/layers/0/attention/query", "bf16");
    check(!cache_current(bf16, published) && ninferno::convert::unquantized_drafter_attention(bf16),
          "a BF16 DFlash drafter at revision 3 is converted again");
    check(!cache_current(artifact(dir, "bf16-2", 3, "dflash2/layers/1/attention/query", "bf16"),
                         published),
          "a BF16 DFlash2 drafter too");
    check(cache_current(artifact(dir, "q8", 3, "dflash2/layers/0/attention/query", "q8_g32_fp16"),
                        published),
          "a drafter already in Q8 is current");
    // Revision 5: the GGUF dense default (qwen3_5_bf16) left the companion BF16 through
    // revision 4 (unsloth/Qwen3.6-27B-GGUF + z-lab/Qwen3.6-27B-DFlash).
    check(!cache_current(bf16, bf16_recipe), "a qwen3_5_bf16 BF16 DFlash drafter at revision 3");
    check(!cache_current(artifact(dir, "gguf4", 4, "dflash/layers/0/attention/query", "bf16"),
                         bf16_recipe),
          "a qwen3_5_bf16 BF16 DFlash drafter at revision 4 is converted again");
    check(cache_current(
              artifact(dir, "gguf4-q8", 4, "dflash/layers/0/attention/query", "q8_g32_fp16"),
              bf16_recipe),
          "a Q8 drafter at revision 4 is current");
    check(cache_current(artifact(dir, "new", 5, "dflash/layers/0/attention/query", "bf16"),
                        published),
          "revision 5 is current");
    check(!cache_current(artifact(dir, "old", 2, "text/output_head", "bf16"), published),
          "revision 2 is not");
    fs::remove_all(dir);
    if (failures == 0) { std::cout << "cache revision: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
