// Loads the Gemma 4 26B-A4B target with its MTP drafter and its DFlash drafter from real
// artifacts, and checks that both drafters refuse the 31B target. NINFERNO_TEST_GEMMA4_DIR names
// the directory that holds the artifacts; without it the test skips.

#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "models/gemma4/model.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::models::gemma4;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// Reading the directory touches no weight, so the 31B target's config costs no device memory.
Config target_config(const std::filesystem::path& path) {
    const artifact::Reader reader(path);
    return parse_config(reader.directory().components.at("text").config);
}

template <typename Load>
void expect_artifact_error(Load load, const std::string& what) {
    try {
        load();
        check(false, what + ": loaded");
    } catch (const artifact::ArtifactError&) {
    } catch (const std::exception& error) {
        check(false, what + ": threw a non-artifact error: " + error.what());
    }
}

void check_target(const Model& model) {
    const auto& c = model.config();
    const auto& w = model.weights();
    check(c.moe(), "the 26B-A4B target has an expert bank");
    check(c.kinds.size() == std::size_t(c.layers) && w.layers.size() == std::size_t(c.layers),
          "the target has one kind and one prepared layer per layer");
    check(w.token_embedding.n == c.vocab_size && w.token_embedding.k == c.hidden_size,
          "the embedding covers the vocabulary at the hidden width");
    check(w.output_head.weight.n == c.vocab_size, "the tied head scores the whole vocabulary");
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const auto& layer = w.layers[i];
        const auto name   = "target layer " + std::to_string(i);
        check(layer.kind == c.kinds[i], name + " keeps its config's kind");
        check(layer.moe.has_value(), name + " has its expert bank");
        check(layer.qkv.has_value() || layer.query.weight.n > 0, name + " has its query rows");
        check(layer.gate_up.weight.n == 2 * c.intermediate_size,
              name + " packs gate and up into one plane");
    }
    check(model.device_bytes() > 0, "the target's weights are on the device");
}

void check_mtp(const Drafter& drafter, const Config& target) {
    const auto& c = drafter.config();
    const auto& w = drafter.weights();
    check(c.backbone_size == target.hidden_size, "the MTP drafter reads the target's width");
    check(w.layers.size() == std::size_t(c.layers), "the MTP drafter prepared every layer");
    check(w.head.weight.n == c.head_rows, "the MTP head has one row per shortlisted token");
    // This artifact restricts the head to a shortlist; each row names the token it scores.
    check(c.head_rows < c.vocab_size && w.head_token_ids.numel() == c.head_rows,
          "the MTP shortlist carries one token id per head row");
    check(drafter.device_bytes() > 0, "the MTP drafter's weights are on the device");
}

void check_dflash(const DFlashDrafter& drafter, const Config& target) {
    const auto& c = drafter.config();
    const auto& w = drafter.weights();
    check(c.hidden_size == target.hidden_size && c.target_layers == target.layers,
          "the DFlash drafter was trained on this target's width and depth");
    check(!c.taps.empty(), "the DFlash drafter taps the target");
    for (const auto tap : c.taps) {
        check(tap >= 0 && tap < target.layers, "every DFlash tap is a target layer");
    }
    check(w.layers.size() == std::size_t(c.layers), "the DFlash drafter prepared every layer");
    check(w.feature_projection.weight.k == std::int32_t(c.taps.size()) * c.hidden_size,
          "the DFlash feature projection reads every tap");
    check(drafter.device_bytes() > 0, "the DFlash drafter's weights are on the device");
}

} // namespace

int main() {
    const char* dir = std::getenv("NINFERNO_TEST_GEMMA4_DIR");
    if (dir == nullptr || *dir == '\0') {
        std::cout << "skip: NINFERNO_TEST_GEMMA4_DIR is not set\n";
        return 77;
    }
    const std::filesystem::path artifacts(dir);
    const auto moe_target = artifacts / "gemma4_26b_a4b_nvfp4.ninferno";
    const auto mtp        = artifacts / "gemma4_26b_assistant_q8_sl98304r2.ninferno";
    const auto dflash     = artifacts / "gemma4_26b_dflash_q8.ninferno";
    const auto dense      = artifacts / "gemma4_31b_nvfp4.ninferno";

    try {
        DeviceContext device(0);
        {
            // The pairing: each drafter is checked against the loaded target's config. Freed at
            // the end of the scope, before the mismatched loads below.
            const auto target = Model::load(moe_target, device, ActivationProfile::Artifact);
            check_target(*target);
            const auto mtp_drafter = Drafter::load(mtp, device, target->config());
            check_mtp(*mtp_drafter, target->config());
            const auto dflash_drafter = DFlashDrafter::load(dflash, device, target->config());
            check_dflash(*dflash_drafter, target->config());
        }

        // The mismatch: the 26B drafters against the 31B target, whose config alone suffices.
        // The loaders reject the pairing before binding a single weight.
        const Config dense_config = target_config(dense);
        expect_artifact_error([&] { (void)Drafter::load(mtp, device, dense_config); },
                              "the 26B MTP drafter on the 31B target");
        expect_artifact_error([&] { (void)DFlashDrafter::load(dflash, device, dense_config); },
                              "the 26B DFlash drafter on the 31B target");

        // Strict load: every artifact in the directory loads under the loaders' refusal
        // of bindings they never bind - each target (26B and 31B, every publisher's), and each
        // drafter against the target of its size. One at a time, freed before the next.
        if (std::getenv("NINFERNO_TEST_GEMMA4_EVERY") != nullptr) {
            // A drafter's target: the published target of its size.
            const auto target_of = [&](const std::string& name) {
                return name.find("31b") != std::string::npos ? dense : moe_target;
            };
            std::vector<std::filesystem::path> all;
            for (const auto& entry : std::filesystem::directory_iterator(artifacts)) {
                if (entry.path().extension() == ".ninferno") { all.push_back(entry.path()); }
            }
            std::sort(all.begin(), all.end());
            for (const auto& path : all) {
                const auto name = path.filename().string();
                try {
                    if (name.find("assistant") != std::string::npos) {
                        const auto config = target_config(target_of(name));
                        (void)Drafter::load(path, device, config);
                    } else if (name.find("dflash") != std::string::npos) {
                        const auto config = target_config(target_of(name));
                        (void)DFlashDrafter::load(path, device, config);
                    } else {
                        (void)Model::load(path, device, ActivationProfile::Artifact);
                    }
                    std::cout << "strict load: " << name << " loaded\n";
                } catch (const std::exception& error) {
                    check(false, "strict load: " + name + ": " + error.what());
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "gemma4 real loading: 26B target, MTP and DFlash drafters paired; 31B refused\n";
    return 0;
}
