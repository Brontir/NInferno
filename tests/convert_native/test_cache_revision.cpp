// Which cached artifacts revisions 4 and 5 convert again: an artifact of any recipe whose DFlash
// companion drafter kept its attention projections unquantized (revision 4: qwen3_5_published;
// revision 5: qwen3_5_bf16, the dense Qwen3.5 GGUF default, too); not a text-only one, not one
// whose drafter is already Q8. A current older artifact is stamped current, so its report is read
// once. The stamp of what a conversion read tells a checkpoint replaced at the same path.

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
    using ninferno::convert::cached_revision;
    using ninferno::convert::kCacheRevision;
    const auto text = artifact(dir, "text", 3, "text/output_head", "bf16");
    check(cache_current(text), "a text-only artifact at revision 3 is current");
    check(cached_revision(text) == kCacheRevision, "and is stamped the current revision");
    check(cache_current(artifact(dir, "text-4", 4, "text/output_head", "bf16")),
          "a text-only artifact at revision 4 is current");
    const auto bf16 = artifact(dir, "bf16", 3, "dflash/layers/0/attention/query", "bf16");
    check(!cache_current(bf16) && ninferno::convert::unquantized_drafter_attention(bf16),
          "a BF16 DFlash drafter at revision 3 is converted again");
    check(cached_revision(bf16) == 3, "and keeps its stamp");
    check(!cache_current(artifact(dir, "bf16-2", 3, "dflash2/layers/1/attention/query", "bf16")),
          "a BF16 DFlash2 drafter too");
    check(cache_current(artifact(dir, "q8", 3, "dflash2/layers/0/attention/query", "q8_g32_fp16")),
          "a drafter already in Q8 is current");
    // Revision 5: the GGUF dense default (qwen3_5_bf16) left the companion BF16 through
    // revision 4 (unsloth/Qwen3.6-27B-GGUF + z-lab/Qwen3.6-27B-DFlash).
    check(!cache_current(artifact(dir, "gguf4", 4, "dflash/layers/0/attention/query", "bf16")),
          "a BF16 DFlash drafter at revision 4 is converted again");
    check(cache_current(
              artifact(dir, "gguf4-q8", 4, "dflash/layers/0/attention/query", "q8_g32_fp16")),
          "a Q8 drafter at revision 4 is current");
    check(cache_current(artifact(dir, "new", 5, "dflash/layers/0/attention/query", "bf16")),
          "revision 5 is current");
    check(!cache_current(artifact(dir, "old", 2, "text/output_head", "bf16")), "revision 2 is not");

    // A checkpoint replaced at the same path changes the stamp of what its conversion read; an
    // artifact converted before stamps were kept is stamped on first sight, not converted again.
    using ninferno::convert::source_changed;
    using ninferno::convert::source_stamp;
    const auto weights = dir / "model.safetensors";
    std::ofstream(weights) << "weights";
    std::ofstream(dir / "config.json") << "{}";
    const auto before = source_stamp(dir, {weights, dir / "config.json"});
    check(!source_changed(text, before) && fs::exists(text.string() + ".source"),
          "an unstamped artifact is stamped, not changed");
    check(!source_changed(text, source_stamp(dir, {dir / "config.json", weights})),
          "the same files in another order are the same source");
    fs::create_directories(dir / "sub");
    const auto spelled = dir / "sub" / "..";
    check(!source_changed(text, source_stamp(spelled, {spelled / "model.safetensors",
                                                       spelled / "config.json"})),
          "another spelling of the directory is the same source");
    fs::remove(weights);
    check(!source_changed(text, source_stamp(dir, {weights, dir / "config.json"})),
          "weights removed (--consume-source) are no change");
    std::ofstream(weights) << "other weights";
    check(source_changed(text, source_stamp(dir, {weights, dir / "config.json"})),
          "weights of another size are another source");
    fs::remove(weights);
    std::ofstream(dir / "model-00001-of-00002.safetensors") << "shard";
    check(source_changed(text, source_stamp(dir, {dir / "model-00001-of-00002.safetensors",
                                                  dir / "config.json"})),
          "a new file is another source");
    fs::remove_all(dir);
    if (failures == 0) { std::cout << "cache revision: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
