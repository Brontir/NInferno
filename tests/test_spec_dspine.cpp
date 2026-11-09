// DSpine (spec/dspine.h): the published Qwen3.8-27B draft's config (tests/fixtures/dspine) parsed
// and its variants refused, its tensor contract against the checkpoint's safetensors header, and
// the reference arithmetic's invariants. No GPU.

#include "spec/dspine.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sp = ninferno::spec;
using nlohmann::json;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }

std::string fixture_path(const std::string& name) {
    return std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/dspine/" + name;
}

json published() {
    std::ifstream in(fixture_path("config.json"));
    return json::parse(in);
}

bool refused(const json& config) {
    try {
        (void)sp::parse_dspine_config(config);
    } catch (const std::runtime_error&) { return true; }
    return false;
}

// "name dtype 5120x25600" lines of the checkpoint's safetensors header.
std::map<std::string, std::string> header() {
    std::ifstream in(fixture_path("tensors.txt"));
    std::map<std::string, std::string> out;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') { continue; }
        std::istringstream words(line);
        std::string name, dtype, shape;
        words >> name >> dtype >> shape;
        if (dtype == "BF16") { out[name] = shape; }
    }
    return out;
}

std::string shape_text(const std::vector<std::uint64_t>& shape) {
    std::string out;
    for (const auto d : shape) { out += (out.empty() ? "" : "x") + std::to_string(d); }
    return out;
}

void test_published_config() {
    const auto config = published();
    expect(sp::is_dspine_config(config), "the published config carries the bridge");
    const auto c = sp::parse_dspine_config(config);
    expect(c.hidden_size == 5120 && c.layers == 5 && c.block_size == 16, "geometry");
    expect(c.code_size == 1024 && c.rank == 512 && c.head_topk == 16, "bridge geometry");
    expect(c.sliding_window == 8192 && c.proposals() == 15, "window and proposals");
    expect(c.target_layer_ids == std::vector<std::uint32_t>({1, 16, 31, 46, 61}), "taps");
    expect(c.mask_token_id == 248070 && c.vocab_size == 248320, "mask and vocabulary");
    // A plain DFlash config is no DSpine.
    auto plain = config;
    plain["dflash_config"].erase("selected_prefix_code_bridge");
    expect(!sp::is_dspine_config(plain), "no bridge, no DSpine");
}

void test_refusals() {
    const auto base = published();
    const auto edit = [&](auto&& change) {
        auto config = base;
        change(config);
        return config;
    };
    expect(!refused(base), "the published config is accepted");
    expect(refused(edit([](json& c) { c["dflash_config"]["selected_prefix_terminal"] = true; })),
           "terminal-corrected bridge");
    expect(refused(edit([](json& c) { c["dflash_config"].erase("selected_prefix_terminal"); })),
           "terminal flag absent (the serving default is terminal)");
    expect(refused(edit([](json& c) { c["is_causal"] = false; })), "non-causal block");
    expect(refused(edit([](json& c) { c["dflash_config"]["draft_block_causal"] = false; })),
           "non-causal block, dflash_config");
    expect(refused(edit([](json& c) {
               c["dflash_config"]["selected_prefix_feature_next_token_rows"] = true;
           })),
           "next-token rows");
    expect(refused(edit([](json& c) { c["dflash_config"]["selector_rank"] = 256; })),
           "DFlash2 selector keys");
    expect(refused(edit(
               [](json& c) { c["dflash_config"]["selected_prefix_feature_layers"] = {1, 5}; })),
           "a partial bridge");
    expect(refused(edit([](json& c) { c["code_bridge_second_pass"]["mode"] = "parallel"; })),
           "another second pass");
    expect(refused(edit([](json& c) { c["code_bridge_second_pass"]["head_topk"] = 8; })),
           "disagreeing refinement widths");
    expect(refused(edit([](json& c) {
               c["code_bridge_second_pass"]["head_topk"]                            = 12;
               c["dflash_config"]["selected_prefix_last_write_refine"]["head_topk"] = 12;
           })),
           "a refinement width that is no power of two");
    expect(refused(edit([](json& c) { c["rope_parameters"]["rope_theta"] = 10000; })),
           "rope theta");
    expect(refused(edit([](json& c) { c["rope_scaling"] = {{"rope_type", "yarn"}}; })),
           "rope scaling");
    expect(refused(edit([](json& c) { c["block_size"] = 17; })), "block over 16");
    expect(refused(edit([](json& c) { c["architectures"] = {"DFlash2DraftModel"}; })),
           "architecture");
    expect(refused(edit([](json& c) { c["layer_types"][2] = "full_attention"; })),
           "a mix of local and full layers");
    // Null is the same as absent for the optional keys.
    expect(!refused(edit([](json& c) {
        c["has_embed_tokens"]      = nullptr;
        c["partial_rotary_factor"] = nullptr;
    })),
           "null optional keys");
}

void test_tensor_contract() {
    const auto c       = sp::parse_dspine_config(published());
    const auto stored  = header();
    const auto tensors = sp::dspine_tensors(c);
    expect(tensors.size() == stored.size(), "as many tensors as the checkpoint stores");
    for (const auto& tensor : tensors) {
        const auto it = stored.find(tensor.name);
        expect(it != stored.end(), "stored: " + tensor.name);
        if (it != stored.end()) {
            expect(it->second == shape_text(tensor.shape), "shape " + tensor.name);
        }
    }
    // The model card's header: 2,025,912,064 BF16 parameters.
    expect(sp::dspine_parameter_count(c) == 2'025'912'064ULL, "parameter count");
}

void test_round_facts() {
    const auto c     = sp::parse_dspine_config(published());
    const auto full  = sp::dspine_round_facts(c, 16);
    const auto small = sp::dspine_round_facts(c, 8);
    // The codebook is gathered, not streamed: 1 anchor row and K sender rows per later proposal.
    expect(full.codebook_values == (1 + 16ULL * 14) * 1024, "codebook rows touched");
    expect(full.head_values == 16ULL * 15 * 5120, "head rows of the candidates");
    // 5 layers of attention + FFN weights, the feature projection, 3 norms.
    const std::uint64_t layer = 5120ULL * (4096 + 2 * 1024) + 4096ULL * 5120 + 3ULL * 5120 * 17408;
    expect(full.backbone_params == 5120ULL * 25600 + 2 * 5120 + 5 * (layer + 2 * 5120 + 2 * 128),
           "backbone parameters");
    // read (4 early + last) and write per layer, the three gate matrices and the bias.
    expect(full.bridge_matrices == 5ULL * 1024 * 5120 + 5ULL * 5120 * 512 + 3ULL * 512 * 1024 + 512,
           "bridge matrices");
    expect(small.multiply_adds < full.multiply_adds && small.codebook_values < full.codebook_values,
           "a narrower round does less");
    bool threw = false;
    try {
        (void)sp::dspine_round_facts(c, 17);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "a round wider than the trained block is refused");
}

void test_arithmetic() {
    // A unit code has norm sqrt(C) (up to eps), whatever the scale of the hidden state.
    std::vector<float> z(64), a(64), b(64);
    for (std::size_t i = 0; i < z.size(); ++i) { z[i] = std::sin(0.37F * float(i)) + 0.1F; }
    auto scaled = z;
    for (auto& v : scaled) { v *= 1000; }
    sp::dspine_unit_code(z, 1e-6F, a);
    sp::dspine_unit_code(scaled, 1e-6F, b);
    double norm = 0, drift = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        norm += double(a[i]) * a[i];
        drift = std::max(drift, std::abs(double(a[i]) - b[i]));
    }
    expect(near(std::sqrt(norm), 8.0, 1e-4), "unit code norm is sqrt(C)");
    expect(drift < 1e-4, "the code is scale-invariant");

    // The anchor's code is the codebook row times sqrt(C).
    const std::vector<float> row(16, 0.25F);
    std::vector<float> anchor(16);
    sp::dspine_anchor_code(row, anchor);
    expect(near(anchor[3], 1.0, 1e-6), "anchor code");

    // A message: gate 0.5 at zero input; saturates with a large input.
    std::vector<float> zero(4, 0.F), value{1, 2, 3, 4}, out(4);
    sp::dspine_message(zero, zero, value, zero, out);
    expect(near(out[1], 1.0, 1e-6), "sigmoid(0) = 0.5");
    const std::vector<float> big(4, 40.F);
    sp::dspine_message(big, zero, value, zero, out);
    expect(near(out[3], 4.0, 1e-4), "a saturated gate passes the value");

    std::vector<float> raw{1, 1}, delta{2, -2}, written(2);
    sp::dspine_write(raw, delta, 0.5F, written);
    expect(near(written[0], 2.0, 1e-6) && near(written[1], 0.0, 1e-6), "write adds rms * delta");
}

void test_walk() {
    // K=3, 3 proposals. edges[(i*3 + p)*3 + c].
    std::vector<float> edges(3 * 3 * 3, 0.F);
    const auto at = [&](int i, int p, int c) -> float& { return edges[(i * 3 + p) * 3 + c]; };
    at(0, 0, 1)   = 5;  // proposal 0: lane 1 whatever the (anchor) predecessor
    at(0, 1, 2)   = 99; // never read: proposal 0 has no predecessor lane
    at(1, 1, 2)   = 4;  // after lane 1, lane 2
    at(1, 0, 0)   = 50; // the walk did not take lane 0
    at(2, 2, 0)   = 3;  // after lane 2, lane 0
    at(2, 1, 1)   = 60;
    auto lanes    = sp::dspine_walk_greedy(edges, 3, 3);
    expect(lanes == std::vector<std::uint32_t>({1, 2, 0}), "greedy walk follows the taken lanes");
    // Ties go to the lowest lane.
    std::vector<float> tie(3, 1.F);
    expect(sp::dspine_walk_greedy(tie, 3, 1)[0] == 0, "lowest lane on a tie");

    // Sampling: each proposal's distribution sums to one; a cold temperature is the argmax.
    std::vector<float> q(9), u{0.99F, 0.99F, 0.99F};
    lanes = sp::dspine_walk_sample(edges, 3, 3, 0.01F, u, q);
    expect(lanes == std::vector<std::uint32_t>({1, 2, 0}), "cold sampling is the greedy walk");
    for (int i = 0; i < 3; ++i) {
        expect(near(q[i * 3] + q[i * 3 + 1] + q[i * 3 + 2], 1.0, 1e-5), "q sums to one");
    }
    // A hot temperature spreads the mass; uniform 0 draws lane 0.
    std::vector<float> flat(3, 0.F), qf(3), zero{0.F};
    lanes = sp::dspine_walk_sample(flat, 3, 1, 1.F, zero, qf);
    expect(lanes[0] == 0 && near(qf[2], 1.0 / 3, 1e-6), "equal scores, equal mass");
}

} // namespace

int main() {
    test_published_config();
    test_refusals();
    test_tensor_contract();
    test_round_facts();
    test_arithmetic();
    test_walk();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
