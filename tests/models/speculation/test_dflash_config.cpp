// The rows a DFlash drafter's full layers read (DFlashConfig::full_window, #549): a YaRN rope
// records the context length it was scaled from (original_max_position_embeddings); an artifact
// converted before that is told by YaRN's default attention factor (0.1 ln(factor) + 1) and the
// positions the drafter addresses; a rope that is not YaRN has no window.
#include "artifact/schema.h"
#include "models/speculation/dflash.h"

#include <cmath>
#include <iostream>
#include <string>

using ninferno::artifact::Json;
using ninferno::models::speculation::parse_dflash_config;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << std::endl;
        ++failures;
    }
}

// gpt-oss's drafter: eight full layers, D=64, 64 query heads, a rope of 32 resolved frequencies.
// `positions`: max_position_embeddings (0: not recorded); `sliding`: the last layer slides.
Json config(const Json& rope, int positions = 131072, bool sliding = false) {
    Json layers = Json::array();
    for (int i = 0; i < 8; ++i) {
        layers.push_back(sliding && i == 7 ? "sliding_attention" : "full_attention");
    }
    Json freq = Json::array();
    for (int i = 0; i < 32; ++i) {
        freq.push_back(std::pow(150000.0, -static_cast<double>(i) / 32));
    }
    Json r          = rope;
    r["rotary_dim"] = 64;
    r["inv_freq"]   = freq;
    Json out        = {{"model_type", "dflash"},
                       {"hidden_size", 2880},
                       {"num_hidden_layers", 8},
                       {"num_target_layers", 24},
                       {"num_attention_heads", 64},
                       {"num_key_value_heads", 8},
                       {"head_dim", 64},
                       {"intermediate_size", 7680},
                       {"block_size", 8},
                       {"rope_theta", 150000.0},
                       {"rms_norm_eps", 1e-5},
                       {"layer_types", layers},
                       {"rope", r},
                       {"dflash_config",
                        {{"target_layer_ids", {1, 6, 11, 16, 21}},
                         {"mask_token_id", 200000},
                         {"sliding_attention_mask", "block"}}}};
    if (positions != 0) { out["max_position_embeddings"] = positions; }
    if (sliding) { out["sliding_window"] = 2048; }
    return out;
}

} // namespace

int main() {
    const double scaling = 0.1 * std::log(32.0) + 1.0;
    check(parse_dflash_config(config({{"rope_type", "yarn"},
                                      {"attention_scaling", scaling},
                                      {"original_max_position_embeddings", 4096}}))
                  .full_window == 4096,
          "a recorded original length is the window");
    check(parse_dflash_config(config({{"rope_type", "yarn"}, {"attention_scaling", scaling}}))
                  .full_window == 4096,
          "an artifact without it is told by the attention factor (gpt-oss: 32 -> 4096)");
    const Json recorded = {{"rope_type", "yarn"},
                           {"attention_scaling", scaling},
                           {"original_max_position_embeddings", 4096}};
    const Json told     = {{"rope_type", "yarn"}, {"attention_scaling", scaling}};
    // The gate: original below the positions, a drafter without a sliding layer.
    Json equal                                = recorded;
    equal["original_max_position_embeddings"] = 131072;
    check(parse_dflash_config(config(equal)).full_window == 0, "original == max: no window");
    check(parse_dflash_config(config(recorded, 0)).full_window == 0,
          "a recorded length without max_position_embeddings: no window");
    check(parse_dflash_config(config(told, 0)).full_window == 0,
          "the derived length without max_position_embeddings: no window");
    check(parse_dflash_config(config(recorded, 131072, true)).full_window == 0,
          "a drafter with a sliding layer: no window");
    // An attention_factor set in the checkpoint gives no whole factor: no derived length.
    check(parse_dflash_config(config({{"rope_type", "yarn"}, {"attention_scaling", 1.2}}))
                  .full_window == 0,
          "an explicit attention factor: no derived length");
    check(parse_dflash_config(config({{"rope_type", "default"}, {"attention_scaling", scaling}}))
                  .full_window == 0,
          "a rope that is not YaRN has no window");
    check(parse_dflash_config(config({{"rope_type", "yarn"}, {"attention_scaling", 1.0}}))
                  .full_window == 0,
          "a YaRN rope of factor 1 has none");
    if (failures == 0) {
        std::cout << "dflash config: PASS
                     "; }
                    return failures ==
                0
            ? 0
            : 1;
    }
