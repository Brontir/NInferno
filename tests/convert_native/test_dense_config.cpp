// The dense decoder adapter against transformers: every rotary table it resolves must equal
// the inv_freq transformers computes for that checkpoint (fixtures/dense_rope.json, written by
// transformers 5.17 from the real config.json files of Ministral 3 3B, OLMo 3 7B, Granite 4.2 8B
// and Llama 3.2 1B), and the resolved conventions must be the ones each model's modeling code uses.
#include "convert/dense.h"
#include "convert/numerics.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using ninferno::convert::OJson;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::uint32_t bits(double value) {
    const float f = static_cast<float>(value);
    std::uint32_t out;
    std::memcpy(&out, &f, 4);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_dense_config <fixtures/dense_rope.json>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    const auto fixtures = OJson::parse(in);
    for (auto it = fixtures.begin(); it != fixtures.end(); ++it) {
        const auto& name = it.key();
        const auto c     = ninferno::convert::dense_text_config(it.value().at("config"));
        const auto& rope = it.value().at("rope");
        std::size_t compared = 0, ulps = 0;
        for (auto r = rope.begin(); r != rope.end(); ++r) {
            const auto& got = c.at("rope_groups").at(r.key());
            // LongRoPE: the long-context frequencies too, and the position count they start past.
            for (const auto& [ours, theirs] : {std::pair{"inv_freq", "inv_freq_bits"},
                                               std::pair{"long_inv_freq", "long_inv_freq_bits"}}) {
                if (!r.value().contains(theirs)) {
                    expect(!got.contains(ours), name + " " + r.key() + ": unexpected " + ours);
                    continue;
                }
                const auto& want = r.value().at(theirs);
                expect(got.contains(ours) && got.at(ours).size() == want.size(),
                       name + " " + r.key() + " " + ours + " size");
                if (!got.contains(ours)) { continue; }
                for (std::size_t i = 0; i < want.size() && i < got.at(ours).size(); ++i) {
                    const auto a = bits(got.at(ours)[i].get<double>());
                    const auto b = want[i].get<std::uint32_t>();
                    ulps         = std::max<std::size_t>(ulps, a > b ? a - b : b - a);
                    ++compared;
                }
            }
            if (r.value().contains("long_after")) {
                expect(got.value("long_after", 0) == r.value().at("long_after").get<int>(),
                       name + " " + r.key() + " long_after");
            }
            expect(std::abs(got.at("attention_scaling").get<double>() -
                            r.value().at("attention_scaling").get<double>()) < 1e-6,
                   name + " " + r.key() + " attention_scaling");
        }
        std::cout << name << ": " << compared << " frequencies, max " << ulps
                  << " ulp from transformers\n";
        // One binary32 rounding of pow may differ between libm and torch.
        expect(ulps <= 1, name + ": rotary frequencies differ from transformers by " +
                              std::to_string(ulps) + " ulp");
        if (name == "ministral3_3b") {
            expect(c.at("query_scale").at("period") == 16384, "ministral3 query temperature");
            expect(c.at("tie_word_embeddings") == true, "ministral3 3B ties its head");
            expect(c.at("norm_placement") == "pre" && c.at("qk_norm") == "none", "ministral3");
        } else if (name == "olmo3_7b") {
            expect(c.at("norm_placement") == "post" && c.at("qk_norm") == "projection", "olmo3");
            expect(c.at("sliding_window") == 4096, "olmo3 window");
            expect(c.at("layer_rope")[0] == "sliding_attention" &&
                       c.at("layer_rope")[3] == "full_attention",
                   "olmo3: sliding layers take the unscaled table, full layers YaRN");
            expect(c.at("rope_groups").at("sliding_attention").at("rope_type") == "default",
                   "olmo3 sliding rope");
        } else if (name == "granite42_8b") {
            expect(std::abs(c.at("attention_scale").get<double>() - 0.0078125) < 1e-9,
                   "granite attention_multiplier");
        } else if (name == "llama32_1b") {
            expect(c.at("rope_groups").at("full_attention").at("rope_type") == "llama3",
                   "llama 3.2 rope type");
        } else if (name == "phi4_mini" || name == "phi35_mini") {
            // Phi-3's sliding_window (262144) covers every position: full attention throughout.
            expect(!c.contains("sliding_window") &&
                       c.at("layer_types")[0] == "full_attention",
                   name + ": a window past max_position_embeddings is no window");
            expect(c.at("rope_groups").at("full_attention").at("rotary_dim") ==
                       96,
                   name + " rotary width");
        } else if (name == "gemma3_1b" || name == "gemma3_4b") {
            expect(c.at("norm_placement") == "sandwich" && c.at("norm_unit_offset") == true &&
                       c.at("qk_norm") == "head" && c.at("hidden_act") == "gelu_pytorch_tanh",
                   name + ": sandwich norms with gain 1 + w, per-head q/k norm, GELU-tanh");
            expect(std::abs(c.at("attention_scale").get<double>() - 0.0625) < 1e-9,
                   name + ": query_pre_attn_scalar 256");
            // sqrt(hidden) rounded to BF16 as transformers multiplies it: 34 (1152), 50.5 (2560).
            expect(c.at("embedding_multiplier") == (name == "gemma3_1b" ? 34.0 : 50.5),
                   name + " embedding multiplier");
            expect(c.at("layer_types")[0] == "sliding_attention" &&
                       c.at("layer_types")[5] == "full_attention" &&
                       c.at("layer_rope")[0] == "sliding_attention",
                   name + ": every sixth layer full (sliding_window_pattern 6)");
        }
    }
    // A control: a wrong theta must be caught by the same comparison.
    {
        auto config = fixtures.at("granite42_8b").at("config");
        config["rope_theta"]                    = 1e6;
        config["rope_parameters"]["rope_theta"] = 1e6;
        const auto c    = ninferno::convert::dense_text_config(config);
        const auto a    = bits(c.at("rope_groups").at("full_attention").at("inv_freq")[10]);
        const auto want = fixtures.at("granite42_8b")
                              .at("rope")
                              .at("full_attention")
                              .at("inv_freq_bits")[10]
                              .get<std::uint32_t>();
        expect(a != want, "control: a wrong theta is not detected");
    }
    // Muse Glimmer (meta-models/Muse-Glimmer-30B@a4e59da5 text_config, cut to 8 layers): the
    // sandwich of 1+w norms with a plain final norm, weightless q/k and embedding norms, the
    // attention output gate, qk_scale_factor in the score scale, NoPE full layers from
    // layer_rope_theta, the logit multiplier before the soft cap.
    {
        OJson t = {{"model_type", "muse_glimmer_text"}, {"hidden_size", 6656},
                   {"intermediate_size", 19968}, {"num_hidden_layers", 8},
                   {"num_attention_heads", 32}, {"num_key_value_heads", 2}, {"head_dim", 128},
                   {"hidden_activation", "silu"}, {"max_position_embeddings", 131072},
                   {"rms_norm_eps", 1e-5}, {"post_norm_eps", 1e-8}, {"vocab_size", 202048},
                   {"tie_word_embeddings", false}, {"sliding_window", 2048},
                   {"final_logit_softcapping", 20.0}, {"qk_scale_factor", 3.87},
                   {"output_multiplier", 0.19611613513818404}, {"attention_bias", false},
                   {"rope_parameters", {{"rope_theta", 500000.0}, {"rope_type", "default"}}}};
        OJson kinds = OJson::array(), thetas = OJson::array();
        for (int i = 0; i < 8; ++i) {
            const bool full = (7 - i) % 4 == 0;
            kinds.push_back(full ? "full_attention" : "sliding_attention");
            thetas.push_back(full ? 0.0 : 500000.0);
        }
        t["layer_types"]      = kinds;
        t["layer_rope_theta"] = thetas;
        const OJson source = {{"architectures", {"MuseGlimmerForConditionalGeneration"}},
                              {"model_type", "muse_glimmer"}, {"text_config", t}};
        const auto c = ninferno::convert::dense_text_config(source, true);
        expect(c.at("norm_placement") == "sandwich" && c.at("norm_unit_offset") == true &&
                   c.at("final_norm_unit_offset") == false,
               "muse: sandwich of 1+w norms, plain final norm");
        expect(c.at("qk_norm") == "head_unscaled" && c.at("embedding_norm") == true &&
                   c.at("attention_output_gate") == true,
               "muse: weightless q/k and embedding norms, output gate");
        expect(std::abs(c.at("post_norm_eps").get<double>() - 1e-8) < 1e-15, "muse: post_norm_eps");
        expect(std::abs(c.at("attention_scale").get<double>() - 3.87 / std::sqrt(128.0)) < 1e-6,
               "muse: qk_scale_factor in the score scale");
        expect(std::abs(c.at("logit_multiplier").get<double>() - 0.19611613513818404) < 1e-7 &&
                   c.at("final_logit_softcapping") == 20.0,
               "muse: logit multiplier before the cap");
        expect(c.at("embedding_multiplier") == 1.0, "muse: no embedding multiplier");
        expect(c.at("layer_rope")[3].is_null() && c.at("layer_rope")[7].is_null() &&
                   c.at("layer_rope")[0] == "sliding_attention" &&
                   !c.at("rope_groups").contains("full_attention"),
               "muse: NoPE full layers");
        expect(c.at("sliding_window") == 2048, "muse: window");
        auto bad                   = source;
        bad["text_config"]["layer_rope_theta"][0] = 10000.0;
        try {
            (void)ninferno::convert::dense_text_config(bad, true);
            expect(false, "muse: a layer_rope_theta off its type's theta accepted");
        } catch (const std::exception& error) {
            expect(std::string(error.what()).find("layer_rope_theta") != std::string::npos,
                   "muse: layer_rope_theta message");
        }
    }
    // Refusals name their field.
    try {
        auto config        = fixtures.at("granite42_8b").at("config");
        config["mlp_bias"] = true;
        (void)ninferno::convert::dense_text_config(config);
        expect(false, "mlp_bias accepted");
    } catch (const std::exception& error) {
        expect(std::string(error.what()).find("mlp_bias") != std::string::npos, "mlp_bias message");
    }
    // tie_word_embeddings: a declared value stands whatever the checkpoint holds; unset (absent or
    // null) it follows the checkpoint, tied when there is no output head tensor.
    {
        using ninferno::convert::dense_text_config;
        auto config  = fixtures.at("granite42_8b").at("config");
        auto tie     = [&](bool has_head) {
            return dense_text_config(config, has_head).at("tie_word_embeddings").get<bool>();
        };
        config["tie_word_embeddings"] = true;
        expect(tie(true) && tie(false), "declared true");
        config["tie_word_embeddings"] = false;
        expect(!tie(true) && !tie(false), "declared false");
        config["tie_word_embeddings"] = nullptr;
        expect(!tie(true) && tie(false), "null follows the checkpoint");
        config.erase("tie_word_embeddings");
        expect(!tie(true) && tie(false), "absent follows the checkpoint");
        config["tie_word_embeddings"] = "yes";
        try {
            (void)tie(true);
            expect(false, "a string tie_word_embeddings accepted");
        } catch (const std::exception& error) {
            expect(std::string(error.what()).find("tie_word_embeddings") != std::string::npos,
                   "tie_word_embeddings message");
        }
    }
    // Round-to-nearest NVFP4 (dense_nvfp4): every value decodes within half a code step of its
    // block's scale, and values already on the grid come back exactly.
    {
        constexpr std::size_t rows = 3, k = 64;
        std::vector<float> values(rows * k);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = std::sin(0.37F * static_cast<float>(i)) * (1.0F + static_cast<float>(i % 7));
        }
        float max_abs = 0;
        for (const float v : values) { max_abs = std::max(max_abs, std::fabs(v)); }
        const float divisor = ninferno::convert::nvfp4_divisor(max_abs);
        std::vector<std::uint8_t> codes(rows * k / 2), scales(rows * k / 16);
        ninferno::convert::quantize_nvfp4_rows(values.data(), rows, k, divisor, codes.data(),
                                             scales.data());
        std::vector<float> back(k);
        double worst = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            ninferno::convert::dequantize_nvfp4_row(codes.data() + r * k / 2,
                                                  scales.data() + r * k / 16, divisor, k,
                                                  back.data());
            for (std::size_t j = 0; j < k; ++j) {
                const float scale = ninferno::convert::kE4m3Values[scales[r * k / 16 + j / 16]];
                // The widest E2M1 step is 2 (4 -> 6), so half a step is one block unit.
                worst = std::max(worst, double(std::fabs(back[j] - values[r * k + j]) * divisor / scale));
            }
        }
        std::cout << "nvfp4 rtn: worst error " << worst << " block units\n";
        expect(worst <= 1.0 + 1e-6, "nvfp4 rtn: a value decodes more than half a step away");
        // On the grid: e2m1 values times one block scale.
        std::vector<float> grid(16);
        for (std::size_t j = 0; j < 16; ++j) {
            grid[j] = ninferno::convert::kE2m1Values[j % 16] * 0.5F;
        }
        const float d = ninferno::convert::nvfp4_divisor(3.0F);
        std::vector<std::uint8_t> c(8), s(1);
        ninferno::convert::quantize_nvfp4_rows(grid.data(), 1, 16, d, c.data(), s.data());
        std::vector<float> g(16);
        ninferno::convert::dequantize_nvfp4_row(c.data(), s.data(), d, 16, g.data());
        for (std::size_t j = 0; j < 16; ++j) {
            expect(std::fabs(g[j] - grid[j]) <= 1e-6F * 3.0F, "nvfp4 rtn: a grid value moved");
        }
    }
    if (failures != 0) {
        std::cerr << "dense_config: " << failures << " checks failed\n";
        return 1;
    }
    std::cout << "dense_config: all checks passed\n";
    return 0;
}
