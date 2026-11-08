// The drafter plan's rules that need no hub: a GGUF drafter that records no num_target_layers is
// held to the registry entry's layer count and its taps to the target's layers; a declared base
// model (lineage) matches only a target of the entry's geometry; the dense engine drafts only for
// the models it was validated on (a registry entry for it whose geometry the target has); a target
// of a listed repository with another layer count than the entry's (converted wrong) takes no
// drafter, not even a given one; the built-in Nemotron 3.5 Lightning entry by repository and by
// lineage; a Nemotron-H GGUF conversion that records no mapping version, at a depth no registry
// model has, takes no drafter, not even its built-in MTP layers, whatever its repository; a model
// no repository or lineage names takes the nearest registered model of its engine and vocabulary:
// all of it at the same shape, only the choices of its own MTP layers at a nearby one; a
// checkpoint the dense converter reads (gpt-oss) is described as the dense engine's; attaching
// is still to do while the manifest is missing or lists a published drafter as failed.
#include "convert/acquire.h"
#include "convert/drafters.h"

#include "artifact/fixture.h"
#include "spec/fingerprint.h"
#include "spec/manifest.h"
#include "support/temp_dir.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace cv   = ninferno::convert;
namespace spec = ninferno::spec;
using Json     = nlohmann::json;

namespace {

int failures = 0;

bool line_starts_warning(const std::string& line) { return line.starts_with("warning: "); }

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void expect(bool condition, const std::string& what, const std::string& detail = "") {
    if (!condition) {
        std::cerr << "FAIL: " << what << (detail.empty() ? "" : ": " + detail) << '\n';
        ++failures;
    }
}

const spec::Registry& registry() {
    static const spec::Registry out = spec::Registry::parse(R"({
      "schema": 1, "published": "test",
      "kinds": {"dflash": {"architectures": ["DFlashDraftModel", "MuseGlimmerAssistantModel"],
                           "recipe": "dflash_published"}},
      "models": [{"id": "muse", "engine": "dense",
                  "match": {"repos": ["org/muse", "org/muse-GGUF"], "lineage": ["org/muse"],
                            "geometry": {"hidden_size": 256, "num_hidden_layers": 4,
                                         "vocab_size": 1000}},
                  "drafters": [{"name": "dflash", "kind": "dflash", "repo": "org/muse-GGUF",
                                "revision": "70bf1b61ac09f91b24d39038091b41c582bc5d7a",
                                "gguf": "dflash-muse-Q4_K_M", "recipe": "dflash_published"}],
                  "profiles": [{"gpu_class": "gb203", "applies_to": ["org/muse-GGUF"],
                                "default": {"config": "dflash:3", "drafter": "dflash"},
                                "load": [{"lanes": 3, "depth": 0}]}]}]
    })");
    return out;
}

cv::DrafterTarget muse(std::uint64_t layers = 4) {
    cv::DrafterTarget t;
    t.repo         = "someone/muse-quants";
    t.engine       = "dense";
    t.architecture = "DenseDecoderForCausalLM";
    t.geometry =
        spec::Geometry{.hidden_size = 256, .num_hidden_layers = layers, .vocab_size = 1000};
    return t;
}

// A DFlash GGUF's metadata as the converter maps it: no num_target_layers.
Json gguf_config(Json taps) {
    return Json{{"architectures", Json::array({"DFlashDraftModel"})},
                {"model_type", "dflash"},
                {"hidden_size", 256},
                {"vocab_size", 1000},
                {"block_size", 16},
                {"dflash_config", {{"target_layer_ids", std::move(taps)}, {"mask_token_id", 999}}}};
}

bool has_check(const cv::PlannedDrafter& d, const std::string& start) {
    for (const auto& check : d.checks) {
        if (check.rfind(start, 0) == 0) { return true; }
    }
    return false;
}

void num_target_layers() {
    auto plan = cv::plan_drafters(
        [] {
            auto t = muse();
            t.repo = "org/muse-GGUF";
            return t;
        }(),
        registry());
    expect(plan.matched_by == "repo" && plan.drafters.size() == 1, "the entry by repository");
    expect(plan.drafters[0].supported && plan.drafters[0].entry_geometry.num_hidden_layers == 4,
           "a validated dense target runs the entry's drafter");
    auto d = plan.drafters[0];
    cv::check_drafter(plan.target, gguf_config({0, 3}), "", d, registry());
    expect(d.compatible, "taps up to L-1 and the entry's layer count pass", d.reason);
    expect(has_check(d, "ok num_target_layers unrecorded (a GGUF drafter)"),
           "the layer count is checked against the entry's");

    d = plan.drafters[0];
    cv::check_drafter(plan.target, gguf_config({0, 4}), "", d, registry());
    expect(!d.compatible && d.reason.find("target_layer_ids") != std::string::npos,
           "a tap at L is refused", d.reason);

    // A target of another depth: the drafter was published for the entry's.
    auto deeper = cv::plan_drafters(
        [] {
            auto t = muse(6);
            t.repo = "org/muse-GGUF";
            return t;
        }(),
        registry());
    expect(deeper.matched_by == "repo" && deeper.drafters.size() == 1 &&
               !deeper.drafters[0].supported &&
               deeper.stale_target.find("reconvert") != std::string::npos &&
               deeper.stale_target.find("--model-id") != std::string::npos &&
               deeper.drafters[0].stale_target == deeper.stale_target &&
               !deeper.notes.empty() && deeper.notes.back() == deeper.stale_target,
           "a target of a listed repository with other than the entry's layers takes no drafter",
           deeper.drafters[0].reason);
    d = deeper.drafters[0];
    cv::check_drafter(deeper.target, gguf_config({0, 3}), "", d, registry());
    expect(!d.compatible && !d.supported && d.reason == deeper.drafters[0].stale_target,
           "nor does the check pass it", d.reason);
    // (Refused before the check: the check's own hold on the layer count alone here.)
    d              = deeper.drafters[0];
    d.stale_target = "";
    d.supported    = true;
    d.unvalidated  = false;
    cv::check_drafter(deeper.target, gguf_config({0, 3}), "", d, registry());
    expect(!d.compatible && d.reason.find("num_hidden_layers") != std::string::npos,
           "a target with other than the entry's layers is refused", d.reason);

    // A checkpoint drafter (no gguf file declared) must record num_target_layers.
    d           = plan.drafters[0];
    d.decl.gguf = "";
    cv::check_drafter(plan.target, gguf_config({0, 3}), "", d, registry());
    expect(!d.compatible && d.reason.find("num_target_layers") != std::string::npos,
           "a checkpoint drafter without num_target_layers is refused", d.reason);
}

void lineage() {
    auto t    = muse();
    t.lineage = {"org/muse"};
    auto plan = cv::plan_drafters(t, registry());
    expect(plan.matched_by == "lineage" && plan.entry == "muse" && plan.drafters.size() == 1 &&
               plan.drafters[0].supported,
           "a descendant of the entry's geometry matches through its base model");

    auto small    = muse(2);
    small.lineage = {"org/muse"};
    plan          = cv::plan_drafters(small, registry());
    expect(plan.source == "none" && plan.entry.empty() && plan.drafters.empty() &&
               plan.profiles.empty(),
           "a descendant of another geometry is not the entry's");
}

void dense_scope() {
    // A dense model the registry does not know, declaring a DFlash drafter itself.
    auto t        = muse();
    t.repo        = "someone/other-dense";
    t.declaration = Json{
        {"drafters", Json::array({Json{{"name", "dflash"}, {"kind", "dflash"}, {"repo", "a/b"}}})}};
    const auto plan = cv::plan_drafters(t, registry());
    expect(plan.source == "bundle" && plan.drafters.size() == 1 && !plan.drafters[0].supported &&
               plan.drafters[0].unvalidated,
           "the dense engine does not draft for a model it was not validated on");
    auto d = plan.drafters[0];
    cv::check_drafter(plan.target, gguf_config({0, 3}), "", d, registry());
    expect(!d.compatible && !d.supported, "nor does the check pass it");

    // A drafter the user gives (--drafter-artifact) for a dense model no entry names: a converted
    // GGUF drafter records no num_target_layers, so its taps alone are checked.
    const auto given_drafter = [] {
        cv::PlannedDrafter d;
        d.decl.name = "dflash";
        d.taps_only = true; // attach_drafters() marks every given drafter so
        return d;
    };
    auto given = given_drafter();
    cv::check_drafter(t, gguf_config({0, 3}), "", given, registry());
    expect(given.compatible && given.supported &&
               has_check(given, "note num_target_layers unrecorded: the taps alone are checked"),
           "a given drafter without num_target_layers runs on a model the registry does not name",
           given.reason);
    given = given_drafter();
    cv::check_drafter(t, gguf_config({0, 4}), "", given, registry());
    expect(!given.compatible && given.reason.find("target_layer_ids") != std::string::npos,
           "its taps are still held to the target's layers", given.reason);

    // The same kind for the Qwen engine is not limited.
    auto qwen         = t;
    qwen.engine       = "qwen3_5";
    qwen.architecture = "Qwen3_5ForCausalLM";
    const auto other  = cv::plan_drafters(qwen, registry());
    expect(other.drafters.size() == 1 && other.drafters[0].supported &&
               !other.drafters[0].unvalidated && other.drafters[0].taps_only,
           "other engines run the kinds they run; a bundle's drafter is held to its taps");
    auto bundled = other.drafters[0];
    cv::check_drafter(other.target, gguf_config({0, 3}), "", bundled, registry());
    expect(bundled.compatible, "a bundle's drafter without num_target_layers passes",
           bundled.reason);
}

// Nemotron 3.5 Lightning in the built-in registry: a target converted before the trunk mapper
// left out the MTP block counts 53 layers (unsloth's GGUF folds it into block_count). Its
// repository names the entry, but it takes no drafter and is told to be converted again; a base
// model alone does not name the entry for another depth.
void nemotron_geometry() {
    const auto nemotron = [](std::uint64_t layers) {
        cv::DrafterTarget t;
        t.engine       = "nemotron_h";
        t.architecture = "NemotronHForCausalLM";
        t.geometry =
            spec::Geometry{.hidden_size = 2688, .num_hidden_layers = layers, .vocab_size = 131072};
        return t;
    };
    const auto& builtin = spec::Registry::builtin();
    auto by_repo        = nemotron(53);
    by_repo.repo        = "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF";
    auto plan           = cv::plan_drafters(by_repo, builtin);
    expect(plan.matched_by == "repo" && plan.entry == "nemotron-3.5-lightning-30b-a3b" &&
               plan.drafters.size() == 1 && !plan.drafters[0].supported &&
               plan.drafters[0].reason.find("reconvert") != std::string::npos,
           "a 53-layer target of a listed repository takes no drafter: reconvert it",
           plan.drafters.empty() ? "" : plan.drafters[0].reason);
    by_repo.geometry.num_hidden_layers = 52;
    plan                               = cv::plan_drafters(by_repo, builtin);
    expect(plan.matched_by == "repo" && plan.drafters.size() == 1 && plan.drafters[0].supported,
           "the 52-layer target of a listed repository takes the entry's MTP drafter");

    auto descendant    = nemotron(53);
    descendant.repo    = "someone/nemotron-finetune-GGUF";
    descendant.lineage = {"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"};
    plan               = cv::plan_drafters(descendant, builtin);
    expect(plan.source == "none" && plan.entry.empty() && plan.drafters.empty(),
           "a 53-layer descendant is refused through its base model alone");

    descendant.geometry.num_hidden_layers = 52;
    plan                                  = cv::plan_drafters(descendant, builtin);
    expect(plan.matched_by == "lineage" && plan.entry == "nemotron-3.5-lightning-30b-a3b" &&
               plan.drafters.size() == 1 && plan.drafters[0].supported,
           "a 52-layer descendant matches through its base model");
}

// A Nemotron-H artifact converted from a GGUF before nemotron_h_moe/2 records no gguf_mapper. At a
// depth a registry model has (52 layers at hidden size 2688) it is that model's trunk and runs its
// drafters; at another (53: the MTP block counted as a layer) no drafter runs for it, with or
// without a repository.
void nemotron_mapper() {
    const auto gguf = [](std::string repo, std::string mapper, std::uint64_t layers) {
        cv::DrafterTarget t;
        t.repo         = std::move(repo);
        t.engine       = "nemotron_h";
        t.architecture = "NemotronHForCausalLM";
        t.geometry =
            spec::Geometry{.hidden_size = 2688, .num_hidden_layers = layers, .vocab_size = 131072};
        t.gguf_source = true;
        t.gguf_mapper = std::move(mapper);
        return t;
    };
    const auto& builtin    = spec::Registry::builtin();
    const std::string repo = "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF";

    // 52 layers without a mapper: the registry's depth, by repository or by its shape.
    auto plan = cv::plan_drafters(gguf("", "", 52), builtin);
    expect(plan.matched_by == "similar" && plan.entry == "nemotron-3.5-lightning-30b-a3b" &&
               plan.drafters.size() == 1 && plan.drafters[0].supported && plan.stale_target.empty(),
           "an unversioned conversion at the registry's depth without a repository is not stale: "
           "its shape names the entry",
           plan.stale_target);
    plan = cv::plan_drafters(gguf(repo, "", 52), builtin);
    expect(plan.matched_by == "repo" && plan.stale_target.empty() && plan.drafters.size() == 1 &&
               plan.drafters[0].supported,
           "an unversioned conversion at the registry's depth takes the entry's drafter",
           plan.stale_target);

    // 53 layers without a mapper: stale, by repository or none, built-in MTP layers too.
    plan = cv::plan_drafters(gguf("", "", 53), builtin);
    expect(plan.source == "none" && plan.drafters.empty() &&
               plan.stale_target.find("gguf_mapper") != std::string::npos &&
               plan.stale_target.find("reconvert") != std::string::npos &&
               !plan.notes.empty() && plan.notes.back() == plan.stale_target,
           "an unversioned 53-layer conversion without a repository is stale", plan.stale_target);
    auto builtin_mtp        = gguf("", "", 53);
    builtin_mtp.builtin_mtp = true;
    plan                    = cv::plan_drafters(builtin_mtp, builtin);
    expect(plan.drafters.size() == 1 && plan.drafters[0].decl.kind == "mtp" &&
               !plan.drafters[0].supported && !plan.drafters[0].compatible &&
               plan.drafters[0].reason == plan.stale_target && !plan.stale_target.empty(),
           "nor do its built-in MTP layers draft for it",
           plan.drafters.empty() ? "" : plan.drafters[0].reason);
    plan = cv::plan_drafters(gguf(repo, "", 53), builtin);
    expect(plan.matched_by == "repo" && plan.drafters.size() == 1 &&
               !plan.drafters[0].supported && plan.drafters[0].reason == plan.stale_target &&
               plan.stale_target.find("gguf_mapper") != std::string::npos,
           "an unversioned 53-layer conversion of a listed repository is stale",
           plan.drafters.empty() ? "" : plan.drafters[0].reason);

    // With the mapper: its layer count is the trunk's; 53 is refused only by the repository's.
    plan = cv::plan_drafters(gguf("", "nemotron_h_moe/2", 53), builtin);
    expect(plan.stale_target.empty(), "a versioned conversion of another depth is not stale",
           plan.stale_target);
    plan = cv::plan_drafters(gguf(repo, "nemotron_h_moe/2", 53), builtin);
    expect(plan.drafters.size() == 1 && !plan.drafters[0].supported &&
               plan.stale_target.find("--model-id") != std::string::npos,
           "a versioned conversion of a listed repository at another depth is refused by it",
           plan.stale_target);
    plan = cv::plan_drafters(gguf(repo, "nemotron_h_moe/2", 52), builtin);
    expect(plan.stale_target.empty() && plan.drafters.size() == 1 && plan.drafters[0].supported,
           "a conversion by nemotron_h_moe/2 takes the entry's drafter", plan.stale_target);

    // A Hugging Face checkpoint has no GGUF mapping to record: at a depth no registry model has
    // and without a repository, nothing refuses it.
    auto checkpoint        = gguf("", "", 53);
    checkpoint.gguf_source = false;
    plan                   = cv::plan_drafters(checkpoint, builtin);
    expect(plan.stale_target.empty(),
           "a conversion of a Hugging Face checkpoint is not held to the GGUF mapping's version",
           plan.stale_target);
}

// Qwen3.5-family checkpoints the built-in registry names by no repository and no lineage.
void similar() {
    const auto qwen = [](std::uint64_t hidden, std::uint64_t layers, bool mtp,
                         std::uint64_t vocab = 248320) {
        cv::DrafterTarget t;
        t.repo         = "someone/private-finetune";
        t.engine       = "qwen3_5";
        t.architecture = "Qwen3_5ForCausalLM";
        t.geometry =
            spec::Geometry{.hidden_size = hidden, .num_hidden_layers = layers, .vocab_size = vocab};
        t.builtin_mtp = mtp;
        return t;
    };
    const auto& builtin = spec::Registry::builtin();
    const auto names    = [](const cv::DrafterPlan& plan) {
        std::string out;
        for (const auto& d : plan.drafters) { out += d.decl.name + " "; }
        return out;
    };

    // The shape of Qwen3.6 35B-A3B: that model under another name, all of its drafters.
    auto plan = cv::plan_drafters(qwen(2048, 40, true), builtin);
    expect(plan.matched_by == "similar" && plan.source == "registry" &&
               plan.entry == "qwen3.6-35b-a3b" && plan.drafters.size() == 2 &&
               !plan.profiles.empty(),
           "a model of a registered shape takes that entry whole", names(plan));

    // Between Qwen3.5 4B and 9B, nearer 9B (0.39 against 0.68): its MTP choices, nothing else.
    plan = cv::plan_drafters(qwen(3584, 28, true), builtin);
    const bool noted =
        !plan.notes.empty() && plan.notes.back().find("shape distance 0.39") != std::string::npos;
    expect(plan.matched_by == "similar" && plan.entry == "qwen3.5-9b" &&
               plan.drafters.size() == 1 && plan.drafters[0].decl.kind == "mtp" &&
               plan.drafters[0].compatible && !plan.profiles.empty() && noted,
           "a model of another shape takes the nearest entry's MTP choices",
           plan.notes.empty() ? names(plan) : plan.notes.back());

    // Near Qwen3.6 27B's shape, which also has a DFlash drafter: only the MTP layers carry over.
    plan = cv::plan_drafters(qwen(5120, 48, true), builtin);
    expect(plan.matched_by == "similar" && plan.drafters.size() == 1 &&
               plan.drafters[0].decl.kind == "mtp",
           "a drafter trained for another shape does not carry over", names(plan));

    // Without MTP layers of its own a model of another shape shares nothing.
    plan = cv::plan_drafters(qwen(3584, 28, false), builtin);
    expect(plan.source == "none" && plan.entry.empty() && plan.drafters.empty(),
           "without MTP layers another shape takes nothing", names(plan));

    // Too far (2.4 times as wide and 1.5 times as deep as the 27Bs) or another vocabulary: nothing.
    plan = cv::plan_drafters(qwen(12288, 96, true), builtin);
    expect(plan.source == "builtin" && plan.entry.empty() && plan.profiles.empty(),
           "past kMaxModelDistance no entry stands in", plan.entry);
    plan = cv::plan_drafters(qwen(4096, 32, true, 151936), builtin);
    expect(plan.source == "builtin" && plan.entry.empty(),
           "another vocabulary is another tokenizer: no entry stands in", plan.entry);

    expect(std::abs(spec::shape_distance({2560, 32, 0}, {4096, 32, 0}) - 0.678) < 1e-3 &&
               spec::shape_distance({4096, 32, 0}, {4096, 32, 0}) == 0.0,
           "shape distance: |log2| of the width and depth ratios");
}

// The engine of a checkpoint directory comes from its config.json architecture.
void checkpoint_engine() {
    const TempDir scratch("ninferno_drafter_plan_ckpt");
    const auto& dir      = scratch.path();
    const auto engine_of = [&](const std::string& architecture) {
        std::ofstream(dir / "config.json") << Json{{"architectures", Json::array({architecture})},
                                                   {"hidden_size", 256},
                                                   {"num_hidden_layers", 4},
                                                   {"vocab_size", 1000}}
                                                  .dump();
        return cv::describe_checkpoint(dir, "org/model").engine;
    };
    expect(engine_of("GptOssForCausalLM") == "dense", "a gpt-oss checkpoint is the dense engine's");
    expect(engine_of("Qwen3MoeForCausalLM") == "dense",
           "a Qwen3 MoE checkpoint is the dense engine's");
    expect(engine_of("NemotronHForCausalLM") == "nemotron_h", "Nemotron-H keeps its engine");
    expect(engine_of("NemotronHMtpForCausalLM").empty(), "a Nemotron-H MTP sidecar is no target");
    expect(engine_of("UnknownForCausalLM").empty(), "an unknown architecture has no engine");
}

// Built-in MTP layers are what the weight index lists: a fine-tune saved without them keeps the
// declaration in its config.json and has none. Without an index, the declaration.
void checkpoint_mtp() {
    const TempDir scratch("ninferno_drafter_plan_mtp");
    const auto& dir = scratch.path();
    std::ofstream(dir / "config.json") << Json{{"architectures", Json::array({"Qwen3_5ForCausalLM"})},
                                               {"hidden_size", 256},
                                               {"num_hidden_layers", 4},
                                               {"vocab_size", 1000},
                                               {"mtp_num_hidden_layers", 1}}
                                              .dump();
    expect(cv::describe_checkpoint(dir, "org/model").builtin_mtp,
           "no weight listing: the declared MTP layers");
    const auto index = [&](const Json& names) {
        Json map = Json::object();
        for (const auto& name : names) { map[name.get<std::string>()] = "model-1.safetensors"; }
        std::ofstream(dir / "model.safetensors.index.json") << Json{{"weight_map", map}}.dump();
        return cv::describe_checkpoint(dir, "org/model").builtin_mtp;
    };
    expect(!index({"model.embed_tokens.weight", "lm_head.weight"}),
           "declared MTP layers absent from the index: none");
    expect(index({"model.embed_tokens.weight", "mtp.fc.weight"}), "MTP layers in the index");
    // NextN layers (DeepSeek, GLM) are numbered past the decoder's: their declaration stands.
    const Json nextn  = {{"num_nextn_predict_layers", 1}};
    const Json listed = {{"weight_map", {{"model.layers.4.eh_proj.weight", "model-2.safetensors"}}}};
    expect(cv::sharded_checkpoint_mtp(nextn, listed), "declared NextN layers");
    expect(cv::sharded_checkpoint_mtp({{"text_config", nextn}}, listed),
           "NextN layers declared in text_config");
    expect(!cv::sharded_checkpoint_mtp({{"mtp_num_hidden_layers", 1}}, listed),
           "declared MTP layers the index does not list");
    expect(cv::sharded_checkpoint_mtp(Json::object(),
                                      {{"weight_map", {{"mtp.fc.weight", "model-2.safetensors"}}}}),
           "MTP layers the index lists, the config silent");
}

// A failed drafter is retried by the next run; a failed own quantization is not.
void retry_failed_drafter() {
    const TempDir scratch("ninferno_drafter_plan_manifest");
    const auto& dir     = scratch.path();
    const auto artifact = dir / "model.ninferno";
    const auto with     = [&](const std::string& text) {
        std::ofstream(ninferno::spec::manifest_path(artifact)) << text;
        return cv::drafters_incomplete(artifact);
    };
    const std::string registry =
        R"("registry_sha256": ")" + ninferno::spec::Registry::builtin().source_sha256() + R"(", )";
    expect(cv::drafters_incomplete(artifact), "no manifest: attaching is to do");
    expect(with("{" + registry +
                R"("drafters": [{"name": "dflash", "status": "failed: SSL connect error"}]})"),
           "a failed published drafter is retried");
    expect(!with("{" + registry + R"("drafters": [{"name": "dflash", "status": "converted"},
                                   {"name": "dflash.q8", "quant": "q8", "status": "failed: x"}]})"),
           "a failed own quantization is not retried");
    // Planned from another registry (before an update that names a new drafter), or before the
    // manifest recorded one: planned again.
    expect(with(R"({"registry_sha256": "0000", "drafters": [{"name": "mtp", "status": "embedded"}]})"),
           "a manifest of another registry: attaching is to do");
    expect(with(R"({"drafters": [{"name": "mtp", "status": "embedded"}]})"),
           "a manifest that names no registry: attaching is to do");
    expect(with("{"), "an unreadable manifest: attaching is to do");
    // An attached drafter whose artifact is gone, or is no longer the artifact the manifest
    // recorded, is attached again (otherwise every start serves plain: "changed since attach").
    const auto gone = (dir / "gone.ninferno").string();
    expect(with("{" + registry + R"("drafters": [{"name": "assistant", "status": "converted",
                "artifact": ")" +
                gone + R"(", "artifact_id": "00"}]})"),
           "an attached drafter whose artifact is gone: attaching is to do");
    // A drafter the user gave (--drafter-artifact) is not planned again when its file is gone: a
    // plan without that flag would drop the entry.
    expect(!with("{" + registry + R"("drafters": [{"name": "mine", "status": "attached",
                "artifact": ")" +
                 gone + R"(", "artifact_id": "00"}]})"),
           "a given drafter whose file is gone keeps its entry: nothing to plan");
    // ... and is named for a start that requires its drafters, which stops on it; a drafter the
    // registry planned is not (it is planned and attached again).
    const auto given_gone = cv::given_drafters_changed(artifact);
    expect(given_gone.size() == 1 && given_gone[0] == "mine",
           "a given drafter whose file is gone is named for a required start");
    with("{" + registry + R"("drafters": [{"name": "assistant", "status": "converted",
                "artifact": ")" +
         gone + R"(", "artifact_id": "00"}]})");
    expect(cv::given_drafters_changed(artifact).empty(),
           "a planned drafter whose file is gone is not a given one");
    const auto other = dir / "other.ninferno";
    std::ofstream(other) << "not an artifact";
    expect(with("{" + registry + R"("drafters": [{"name": "assistant", "status": "converted",
                "artifact": ")" +
                other.string() + R"(", "artifact_id": "00"}]})"),
           "an attached drafter whose artifact cannot be read: attaching is to do");
    expect(cv::refresh_changed_drafters(artifact).empty(),
           "a drafter file that cannot be read is not refreshed (an attach plans it again)");
    // A drafter file here written again since (its id is no longer the recorded one) takes its new
    // id, whatever the start asks of drafters: nothing is fetched, the other entries are kept.
    // A drafter whose vocabulary cannot be read (the fixture's tokenizer.json is not a tokenizer)
    // is left to an attach, and the others are refreshed all the same.
    const ninferno::test::artifact_fixture::Fixture unreadable;
    unreadable.write();
    std::ofstream(unreadable.entry.string() + ".conversion.json") << "{}";
    // A vocabulary the fingerprint can read.
    ninferno::test::artifact_fixture::Fixture drafter;
    const std::string tokenizer = R"({"model": {"vocab": {"a": 0, "b": 1}}})";
    for (std::size_t i = 0; i < tokenizer.size(); ++i) {
        drafter.payload[i] = std::byte(tokenizer[i]);
    }
    drafter.root["objects"][0]["bytes"] = tokenizer.size();
    drafter.write();
    std::ofstream(drafter.entry.string() + ".conversion.json") << "{}";
    const auto id    = ninferno::spec::artifact_id_hex(drafter.entry);
    const auto entry = [&](const std::string& status, const std::string& recorded) {
        return R"({"name": "assistant", "status": ")" + status + R"(", "artifact": ")" +
               drafter.entry.string() + R"(", "artifact_id": ")" + recorded + R"("})";
    };
    const auto manifest = [&](const std::string& target, const std::string& drafters) {
        return "{" + registry + R"("target": )" + target + R"(, "drafters": [)" + drafters + "]}";
    };
    const std::string companion = R"({"name": "dflash2", "status": "failed: SSL connect error"})";
    // The attach's checks, stood in by the vocabulary they compare (check_drafter itself is
    // tested apart); the production one describes the target, which this test has none of.
    std::string wanted = ninferno::spec::artifact_vocab_fingerprint(drafter.entry).value().value;
    int checked        = 0;
    const cv::DrafterCheck passes = [&](const Json&, const std::filesystem::path& file) {
        ++checked;
        return ninferno::spec::artifact_vocab_fingerprint(file).value().value == wanted;
    };
    const auto refresh = [&] { return cv::refresh_changed_drafters(artifact, passes); };
    expect(!with(manifest("{}", entry("cached", id))), "the recorded id: nothing to do");
    expect(refresh().empty() && checked == 0, "the recorded id: nothing checked or refreshed");
    expect(with(manifest("{}", entry("converted", "00") + ", " + companion)),
           "a drafter written again since: attaching is to do");
    const auto refreshed = refresh();
    expect(refreshed.size() == 1 && refreshed[0] == "assistant",
           "the changed drafter is refreshed");
    const auto after = Json::parse(read_text(ninferno::spec::manifest_path(artifact)));
    expect(after["drafters"][0]["artifact_id"] == id &&
               after["drafters"][0]["status"] == "converted",
           "the refreshed drafter takes the file's id and keeps its status");
    expect(after["drafters"][1]["status"] == "failed: SSL connect error",
           "another drafter's failure is kept for its retry");
    expect(refresh().empty(), "refreshed once: nothing more to do");
    // One drafter whose vocabulary cannot be read does not stop the others' refresh.
    const std::string bad = R"({"name": "other", "status": "cached", "artifact": ")" +
                            unreadable.entry.string() + R"(", "artifact_id": "00"})";
    with(manifest("{}", bad + ", " + entry("cached", "00")));
    const auto past_bad = refresh();
    expect(past_bad.size() == 1 && past_bad[0] == "assistant",
           "a drafter of an unreadable vocabulary is skipped, the next one refreshed");
    // Checks that cannot run on the file refuse it like checks that fail: its id is kept, so an
    // attach does not take it for a cache.
    const auto past = Json::parse(read_text(ninferno::spec::manifest_path(artifact)));
    expect(past["drafters"][0].value("refused_artifact_id", std::string()) ==
               ninferno::spec::artifact_id_hex(unreadable.entry),
           "a file the checks cannot read is recorded as refused");
    // A replacement the checks refuse keeps the entry as it was: the next start looks at the file
    // again, and takes a later replacement that passes.
    wanted = "v1:another";
    with(manifest("{}", entry("cached", "00")));
    expect(refresh().empty(), "a replacement the checks refuse is not refreshed");
    const auto refused = Json::parse(read_text(ninferno::spec::manifest_path(artifact)));
    expect(refused["drafters"][0]["status"] == "cached" &&
               refused["drafters"][0]["artifact_id"] == "00",
           "a refused replacement leaves the entry as it was");
    // The refused file is named on the entry, so an attach converts it again instead of taking it
    // for its cache.
    expect(refused["drafters"][0].value("refused_artifact_id", std::string()) == id,
           "the refused file's id is kept on the entry");
    wanted = ninferno::spec::artifact_vocab_fingerprint(drafter.entry).value().value;
    expect(refresh().size() == 1, "the next start takes a replacement that passes");
    const auto taken = Json::parse(read_text(ninferno::spec::manifest_path(artifact)));
    expect(!taken["drafters"][0].contains("refused_artifact_id"),
           "a replacement taken clears the refusal");
    // The production checks describe the target: one that cannot be read refreshes nothing.
    with(manifest("{}", entry("cached", "00")));
    expect(cv::refresh_changed_drafters(artifact).empty(),
           "no target to check against: nothing refreshed");
    // Never attached (failed), or without its conversion report (a conversion stopped half way):
    // left to an attach.
    with(manifest("{}", entry("failed: x", "00")));
    expect(refresh().empty(), "a failed drafter was never attached");
    std::filesystem::remove(drafter.entry.string() + ".conversion.json");
    with(manifest("{}", entry("cached", "00")));
    expect(refresh().empty(), "no conversion report: not refreshed");
    // A file the user gave (--drafter-artifact) has no conversion report and stays theirs.
    with(manifest("{}", entry("attached", "00")));
    expect(refresh().size() == 1, "a given drafter file is refreshed");
    const auto given = Json::parse(read_text(ninferno::spec::manifest_path(artifact)));
    expect(given["drafters"][0]["status"] == "attached" &&
               given["drafters"][0]["artifact_id"] == id,
           "a given drafter keeps its status and takes the file's id");
}

// A drafter that cannot be had stops a start that asked for it (--with-drafters) and is a warning
// for a start whose automatic choice drafts: served without it, the line naming the drafter, the
// reason and what it costs, and, only for a failed fetch, how to get the drafter. A drafter that
// is fine says nothing.
void automatic_drafter_missing() {
    cv::DrafterPlan plan;
    cv::PlannedDrafter missing;
    missing.decl.name     = "dflash";
    missing.decl.repo     = "z-lab/Qwen3.5-4B-DFlash";
    missing.decl.revision = "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf";
    missing.decl.gguf     = "dflash-muse-Q4_K_M";
    missing.status        = "failed: HF_HUB_OFFLINE is set; no snapshot in the cache.";
    missing.fetch_failed  = true;
    cv::PlannedDrafter disk;
    disk.decl.name = "assistant";
    disk.decl.repo = "google/gemma-4-12B-it-assistant";
    disk.status    = "failed: needs 3.1 GiB more in the hub cache";
    cv::PlannedDrafter nowhere;
    nowhere.decl.name    = "nowhere";
    nowhere.status       = "failed: no repository declared";
    nowhere.fetch_failed = true; // whatever the flag says, no repository means no advice
    cv::PlannedDrafter odd;
    odd.decl.name = "odd";
    odd.decl.repo = "org/odd";
    odd.status    = "incompatible: vocabulary differs";
    cv::PlannedDrafter lacking;
    lacking.decl.name = "lacking";
    lacking.status    = "not in the artifact: convert the target with --components";
    cv::PlannedDrafter fine;
    fine.decl.name = "mtp";
    fine.status    = "embedded";
    plan.drafters  = {fine, missing, disk, nowhere, odd, lacking};

    std::vector<std::string> lines;
    cv::AcquireOptions options;
    options.log = [&](const std::string& line) { lines.push_back(line); };

    options.drafters = cv::Drafters::Require;
    bool threw       = false;
    try {
        cv::check_attached_drafters(plan, options);
    } catch (const std::exception& error) {
        threw = std::string(error.what()).find("dflash") != std::string::npos;
    }
    expect(threw && lines.empty(), "--with-drafters: the missing drafter is an error");

    options.drafters = cv::Drafters::Attach;
    try {
        cv::check_attached_drafters(plan, options);
    } catch (const std::exception& error) {
        expect(false, "the automatic choice's missing drafter threw", error.what());
    }
    expect(lines.size() == 5, "one warning per drafter left out", std::to_string(lines.size()));
    if (lines.size() == 5) {
        const auto has = [&](const std::string& line, const std::string& part) {
            return line.find(part) != std::string::npos;
        };
        expect(line_starts_warning(lines[0]) && has(lines[0], "drafter dflash") &&
                   has(lines[0], "HF_HUB_OFFLINE") && has(lines[0], "Unset HF_HUB_OFFLINE") &&
                   has(lines[0], "hf download z-lab/Qwen3.5-4B-DFlash --revision 9a1996cc") &&
                   has(lines[0], "--include \"dflash-muse-Q4_K_M*\"") &&
                   has(lines[0], "speculative decoding off") && !has(lines[0], ".."),
               "the fetch failure's warning names the drafter, the reason, the cure and the cost",
               lines[0]);
        expect(has(lines[1], "needs 3.1 GiB") && !has(lines[1], "hf download") &&
                   !has(lines[1], "REPO_ID"),
               "a full disk gets no download advice", lines[1]);
        expect(!has(lines[2], "hf download") && !has(lines[2], "REPO_ID"),
               "no repository: no download advice", lines[2]);
        expect(has(lines[3], "incompatible") && !has(lines[3], "hf download"),
               "an incompatible drafter gets no download advice", lines[3]);
        expect(has(lines[4], "drafter lacking") && has(lines[4], "not in the artifact"),
               "a companion the artifact lacks is warned of too", lines[4]);
    }

    // Required by the selected backend, not by the flag: the error says so and offers the ways
    // out, and does not blame --with-drafters.
    options.drafters    = cv::Drafters::Require;
    options.required_by = "--spec dflash";
    std::string message;
    try {
        cv::check_attached_drafters(plan, options);
    } catch (const std::exception& error) { message = error.what(); }
    expect(message.find("the selected --spec dflash needs its drafter") != std::string::npos &&
               message.find("--spec none") != std::string::npos &&
               message.find("--spec mtp") != std::string::npos &&
               message.find("--with-drafters") == std::string::npos,
           "a --spec backend's error names the backend, not --with-drafters", message);
    options.required_by.clear();
    message.clear();
    try {
        cv::check_attached_drafters(plan, options);
    } catch (const std::exception& error) { message = error.what(); }
    expect(message.find("--with-drafters requires every drafter") != std::string::npos,
           "the flag's error is as before", message);

    lines.clear();
    options.drafters = cv::Drafters::Default;
    cv::check_attached_drafters(plan, options);
    options.drafters = cv::Drafters::Off;
    cv::check_attached_drafters(plan, options);
    expect(lines.empty(), "a checkpoint's default and --no-drafters say nothing here");
}

// A GGUF model's companion drafter whose fetch failed is listed as failed in the manifest, so
// the next run retries it; left to the plan made from the artifact it would read "not in the
// artifact", which a GGUF never retries.
void companion_failure_is_retried() {
    const TempDir scratch("ninferno_drafter_plan_companion");
    const auto artifact = scratch.path() / "model.ninferno";
    cv::DrafterPlan plan;
    cv::PlannedDrafter companion;
    companion.decl.name      = "dflash";
    companion.decl.component = "dflash";
    plan.drafters            = {companion};
    plan.target.gguf_source  = true;

    cv::PlannedDrafter failed = companion;
    failed.status             = "failed: HF_HUB_OFFLINE is set";
    failed.fetch_failed       = true;
    const auto retried        = [&](const cv::DrafterPlan& made) {
        auto manifest               = made.to_json();
        manifest["registry_sha256"] = ninferno::spec::Registry::builtin().source_sha256();
        std::ofstream(ninferno::spec::manifest_path(artifact)) << manifest.dump(1);
        return cv::drafters_incomplete(artifact);
    };

    auto lacking               = plan;
    lacking.drafters[0].status = "not in the artifact: convert the target with --components";
    expect(!retried(lacking), "a GGUF's missing companion alone is not retried (the cause)");

    cv::carry_failures(plan, {failed});
    expect(plan.drafters[0].status == failed.status && plan.drafters[0].fetch_failed,
           "the failure is carried into the plan made from the artifact");
    expect(retried(plan), "the carried failure is retried by the next run");

    cv::DrafterPlan other       = lacking; // a drafter of another name keeps its own status
    other.drafters[0].decl.name = "other";
    cv::carry_failures(other, {failed});
    expect(other.drafters[0].status.starts_with("not in the artifact"),
           "a failure carries only to the drafter of its name");
}

// A drafter the user gave on an earlier start is kept when the drafters are planned again without
// that flag (another drafter's retry, a registry update); a start that gives the name again takes
// the new mapping instead.
void keeps_given_drafters() {
    const ninferno::test::artifact_fixture::Fixture target;
    target.write();
    const auto manifest_path = ninferno::spec::manifest_path(target.entry);
    const std::string old    = R"({"drafters": [
        {"name": "mine", "status": "attached", "artifact": "/models/mine.ninferno", "artifact_id": "ab"},
        {"name": "assistant", "status": "failed: SSL connect error"}]})";
    std::ofstream(manifest_path) << old;
    (void)cv::attach_drafters(cv::DrafterPlan{}, target.entry, cv::AttachOptions{});
    const auto kept  = Json::parse(read_text(manifest_path));
    const auto names = [](const Json& m) {
        std::vector<std::string> out;
        for (const auto& d : m.at("drafters")) { out.push_back(d.value("name", std::string())); }
        return out;
    };
    expect(names(kept) == std::vector<std::string>{"mine"} &&
               kept["drafters"][0]["artifact_id"] == "ab",
           "a given drafter the new plan does not name is kept as it was");
    std::ofstream(manifest_path) << old;
    cv::AttachOptions again;
    again.drafter_artifacts["mine"] = "/models/other.ninferno";
    (void)cv::attach_drafters(cv::DrafterPlan{}, target.entry, again);
    // The new mapping is this start's (here a file that is not there), never the earlier one.
    const auto given = Json::parse(read_text(manifest_path));
    expect(
        std::none_of(given.at("drafters").begin(), given.at("drafters").end(),
                     [](const Json& d) { return d.value("artifact_id", std::string()) == "ab"; }),
        "a start that gives the name again does not keep the old mapping");
}

} // namespace

int main() {
    num_target_layers();
    keeps_given_drafters();
    lineage();
    dense_scope();
    nemotron_geometry();
    nemotron_mapper();
    similar();
    checkpoint_engine();
    checkpoint_mtp();
    retry_failed_drafter();
    automatic_drafter_missing();
    companion_failure_is_retried();
    if (failures == 0) { std::cout << "drafter plan: ok\n"; }
    return failures == 0 ? 0 : 1;
}
