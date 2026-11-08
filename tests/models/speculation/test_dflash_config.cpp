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
Json config(const Json& rope) {
    Json layers = Json::array();
    for (int i = 0; i < 8; ++i) { layers.push_back("full_attention"); }
    Json freq = Json::array();
    for (int i = 0; i < 32; ++i) { freq.push_back(std::pow(150000.0, -static_cast<double>(i) / 32)); }
    Json r        = rope;
    r["rotary_dim"] = 64;
    r["inv_freq"]   = freq;
    return {{"model_type", "dflash"},
            {"hidden_size", 2880},
            {"num_hidden_layers", 8},
            {"num_target_layers", 24},
            {"num_attention_heads", 64},
            {"num_key_value_heads", 8},
            {"head_dim", 64},
            {"intermediate_size", 7680},
            {"block_size", 8},
            {"max_position_embeddings", 131072},
            {"rope_theta", 150000.0},
            {"rms_norm_eps", 1e-5},
            {"layer_types", layers},
            {"rope", r},
            {"dflash_config",
             {{"target_layer_ids", {1, 6, 11, 16, 21}},
              {"mask_token_id", 200000},
              {"sliding_attention_mask", "block"}}}};
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
    check(parse_dflash_config(config({{"rope_type", "default"}, {"attention_scaling", 1.0}}))
                  .full_window == 0,
          "a rope that is not YaRN has no window");
    check(parse_dflash_config(config({{"rope_type", "yarn"}, {"attention_scaling", 1.0}}))
                  .full_window == 0,
          "a YaRN rope of factor 1 has none");
    if (failures == 0) { std::cout << "dflash config: PASS\n"; }
    return failures == 0 ? 0 : 1;
}
