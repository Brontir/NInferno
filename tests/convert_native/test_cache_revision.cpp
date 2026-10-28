// Which cached artifacts revisions 4 to 7 convert again: an artifact of any recipe whose DFlash
// companion drafter kept its attention projections unquantized (revision 4: qwen3_5_published;
// revision 5: qwen3_5_bf16, the dense Qwen3.5 GGUF default, too), and one with a DFlash /
// DFlash2 companion and no proposal head (revision 6), and one with MTP layers and no head or
// revision 6's DFlash-only head (revision 7); not a text-only one. A current older artifact is stamped current, so its report is read
// once. The stamp of what a conversion read tells a checkpoint replaced at the same path.

#include "convert/acquire.h"
#include "convert/cache.h"
#include "convert/checks.h"
#include "convert/fetch.h"
#include "convert/qwen3_5.h"
#include "core/sys.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
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
        << R"({"components": {"text": {"config": {"model_type": "qwen3_5_text"}}}, "methods": [)"
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
    // Revision 6: a Qwen DFlash companion without a proposal head is converted again (it takes
    // one, the ranking found in this tree); DFlash2 takes none.
    check(!cache_current(artifact(dir, "q8", 3, "dflash/layers/0/attention/query", "q8_g32_fp16")),
          "a Q8 DFlash drafter without a proposal head at revision 3 is converted again");
    check(cache_current(artifact(dir, "q8-2", 3, "dflash2/layers/0/attention/query", "q8_g32_fp16")),
          "a Q8 DFlash2 drafter at revision 3 is current");
    check(cache_current(artifact(dir, "q8-head", 3, "proposal/head", "ggml_q6_k")) ,
          "an artifact with a proposal head and no companion at revision 3 is current");
    // Revision 5: the GGUF dense default (qwen3_5_bf16) left the companion BF16 through
    // revision 4 (unsloth/Qwen3.6-27B-GGUF + z-lab/Qwen3.6-27B-DFlash).
    check(!cache_current(artifact(dir, "gguf4", 4, "dflash/layers/0/attention/query", "bf16")),
          "a BF16 DFlash drafter at revision 4 is converted again");
    check(!cache_current(
              artifact(dir, "gguf4-q8", 4, "dflash/layers/0/attention/query", "q8_g32_fp16")),
          "a Q8 drafter without a proposal head at revision 4 is converted again");
    check(!cache_current(artifact(dir, "rev5", 5, "dflash/layers/0/attention/query", "q8_g32_fp16")),
          "a companion without a proposal head at revision 5 is converted again");
    check(cache_current(artifact(dir, "new", 7, "dflash/layers/0/attention/query", "bf16")),
          "revision 7 is current");
    // Revision 7: MTP layers take the default head too; revision 6's head was DFlash's alone.
    check(!cache_current(artifact(dir, "mtp5", 5, "mtp/layers/0/attention/query", "q8_g32_fp16")),
          "MTP layers without a head at revision 5 are converted again");
    check(!cache_current(artifact(dir, "rev6", 6, "dflash/layers/0/attention/query", "q8_g32_fp16")),
          "a DFlash companion without a head at revision 6 is converted again");
    check(!cache_current(artifact(dir, "old", 2, "text/output_head", "bf16")), "revision 2 is not");

    // The pre-conversion budget of the default proposal head (convert::planned_proposal_bytes):
    // Qwen3.5-9B Q4_K_M with z-lab's DFlash drafter keeps 147,456 rows of its Q6_K head (3,360 B a
    // row) and their int32 token ids (Codex on #170).
    check(ninferno::convert::planned_proposal_bytes(834355200.0, 5117970432.0, 1372703744.0,
                                                    248320, 4096) ==
              147456ULL * 3360 + 147456ULL * 4,
          "the planned proposal head counts its rows and their token ids");
    // A BF16 checkpoint's head (no stored bytes known) as its default recipe stores it: the MoE's
    // (qwen3_6_35b_a3b) Q6, the dense one's (qwen3_5_bf16) BF16 (Codex on #170). The MoE's smaller
    // head is a smaller share of the round: more rows than a BF16 head would get.
    {
        using namespace ninferno::convert;
        check(qwen3_5_recipe_head_format(qwen3_5_checkpoint_recipe(true)) == ninferno::QType::Q6_G64_FP16 &&
                  qwen3_5_recipe_head_format(qwen3_5_checkpoint_recipe(false)) == ninferno::QType::BF16,
              "the default recipes' head formats");
        const auto moe   = planned_proposal_bytes(0, 5e9, 0.9e9, 248320, 2048, true);
        const auto dense = planned_proposal_bytes(0, 5e9, 0.9e9, 248320, 2048, false);
        check(dense != 0 && moe > dense, "the MoE checkpoint's head priced at its recipe's Q6");
    }

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
    fs::remove(dir / "model-00001-of-00002.safetensors");
    fs::remove(dir / "config.json");
    check(source_changed(text, source_stamp(dir, {})), "a configuration file gone is a change");
    fs::remove_all(dir);

    // A local commit: a commit id itself, else the one a ref records (refs/<revision>, "main"
    // when none is asked); nothing for a missing ref or a recorded path.
    {
        using ninferno::convert::local_commit;
        const fs::path repo = fs::temp_directory_path() / "ninferno_local_commit_test";
        fs::remove_all(repo);
        fs::create_directories(repo / "refs");
        const std::string id(40, 'a');
        std::ofstream(repo / "refs" / "main") << "c0ffee\n";
        std::ofstream(repo / "refs" / "bad") << "../x";
        check(local_commit(repo, "") == std::optional<std::string>("c0ffee") &&
                  local_commit(repo, "main") == std::optional<std::string>("c0ffee"),
              "the ref's commit, main by default");
        check(local_commit(repo, id) == std::optional<std::string>(id), "a commit id names itself");
        check(!local_commit(repo, "dev") && !local_commit(repo, "bad"),
              "no ref, or a recorded path: no commit");
        fs::remove_all(repo);
    }
    // A conversion's name: <gguf stem>[.vision].<recipe>.ninferno.
    {
        using ninferno::convert::converted_name;
        const auto text = converted_name("m-UD-Q4_K_M.qwen3_5_bf16.ninferno");
        const auto vis  = converted_name("dir/m-Q8_0.vision.qwen3_5_q8.ninferno");
        check(text && text->stem == "m-UD-Q4_K_M" && !text->vision && text->recipe == "qwen3_5_bf16" &&
                  text->label == "UD-Q4_K_M",
              "a text conversion's stem, recipe and label");
        check(vis && vis->stem == "m-Q8_0" && vis->vision && vis->recipe == "qwen3_5_q8" &&
                  vis->label == "Q8_0",
              "a --vision conversion");
        check(!converted_name("m.gguf") && !converted_name("plain.ninferno"),
              "another name is no conversion's");
    }
    if (failures == 0) { std::cout << "cache revision: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
