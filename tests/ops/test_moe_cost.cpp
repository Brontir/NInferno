// The MoE stage price (ops/moe/moe_cost.h, #616 step 4): the law's own terms on small cases, the
// candidate list on a layer of the lab's models, and the replica: every MoE row of the sweeps on
// disk (routes/<class>.json "moe": evidence_us / evidence_decode_us rows [T, default us, chosen
// us], measured by ninferno_moe_bench --plans sweep on the board of the table), priced by
// price_moe_stage on that board's facts with the table's own plan, predicted against measured:
// the ratio (a floor: it must not exceed 1) and the rank agreement of predicted and measured times
// across the rows of a board (Kendall's concordance; a tie in the prediction is counted apart, the
// law having no term to order two plans of the same bytes). Host only, no GPU; it changes no route
// (the moe plan goldens are the check of that).
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/moe.h"
#include "ops/generic/selector_facts.h"
#include "ops/moe/moe_cost.h"
#include "ops/moe/moe_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
namespace moe     = ninferno::ops::detail::moe;
namespace qw      = ninferno::test::quantized_weight;
using Json        = nlohmann::json;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

const generic::FactsBoard& board_named(const std::string& name) {
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        if (name == b.name) return b;
    }
    throw std::runtime_error("no facts board " + name);
}

std::optional<QType> qtype_of(const std::string& s) {
    static const std::map<std::string, QType> names = {
        {"nvfp4", QType::NVFP4},         {"mxfp4", QType::MXFP4},
        {"iq3s", QType::GGML_IQ3_S},     {"iq2s", QType::GGML_IQ2_S},
        {"iq4xs", QType::GGML_IQ4_XS},   {"iq4nl", QType::GGML_IQ4_NL},
        {"iq3xxs", QType::GGML_IQ3_XXS}, {"q8", QType::Q8_G32_FP16}};
    const auto it = names.find(s);
    return it == names.end() ? std::nullopt : std::optional<QType>(it->second);
}

// A layer's weights as descriptors (no payload), the way the plan goldens build them.
struct Built {
    ops::MoeWeights weights;
    ops::MoeConfig config;
    std::optional<qw::PackedWeight> gate_up, down;
};

std::unique_ptr<Built> build(QType gate_up, QType down, std::int32_t experts, std::int32_t top_k,
                             std::int32_t hidden, std::int32_t inter, bool relu2) {
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    auto b              = std::make_unique<Built>();
    const auto options  = [](QType q) {
        qw::PatternedWeightOptions o;
        o.descriptor_only = true;
        if (q == QType::NVFP4) o.weight_scale_divisor = o.input_scale_divisor = 1.0F;
        return o;
    };
    const std::int32_t sets = relu2 ? 1 : 2;
    // A hidden size that is not whole blocks of 256 is a padded GGUF: gate/up columns to the next
    // 256 (Nemotron-H, H 2688, K 2816), as the plan goldens build it.
    const std::int32_t pad = hidden % 256 == 0 ? 0 : (hidden + 255) / 256 * 256 - hidden;
    b->gate_up.emplace(qw::make_patterned_weight(gate_up, experts * sets * inter, hidden + pad, 7U,
                                                 options(gate_up)));
    b->down.emplace(qw::make_patterned_weight(down, experts * hidden, inter, 9U, options(down)));
    b->config.top_k = top_k;
    if (relu2) b->config.activation = ops::MoeActivation::Relu2;
    Weight r{};
    r.qtype  = QType::BF16;
    r.layout = QuantLayout::Contiguous;
    r.qdata = r.payload = payload;
    r.payload_bytes     = static_cast<std::size_t>(experts) * hidden * 2;
    r.n = r.shape[0] = r.padded_shape[0] = experts;
    r.k = r.shape[1] = r.padded_shape[1] = hidden;
    r.ndim                               = 2;
    b->weights.router                    = r;
    b->weights.gate_up                   = b->gate_up->device_weight(payload);
    b->weights.down                      = b->down->device_weight(payload);
    b->weights.expert_policy             = ops::LinearPolicy::AllowA8;
    return b;
}

// The law's own terms on shapes whose answers are read off by hand.
void law_checks() {
    check(std::abs(moe::expected_experts_touched(256, 8, 1) - 8.0) < 1e-9,
          "one token touches top-k experts");
    check(moe::expected_experts_touched(256, 8, 4096) > 255.99, "a wide call touches them all");
    const double two = moe::expected_experts_touched(128, 8, 2);
    check(two > 8.0 && two < 16.0, "two tokens touch between k and 2k");
    check(moe::expected_experts_touched(0, 8, 1) == 0.0 &&
              moe::expected_experts_touched(8, 8, 0) == 0.0,
          "no experts or no tokens: none");

    const moe::MoeBoardFacts board{1e12, 100, 2e9, 64.0, 2048.0};
    moe::MoeStageShape s;
    s.bank_bytes      = 1e9;
    s.experts         = 100;
    s.top_k           = 10;
    s.rows            = 1000;
    s.out_rows        = 500;
    s.cols            = 1000;
    const auto decode = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 1, board);
    check(decode && decode->dram_seconds > decode->compute_seconds,
          "a one-token stage is bandwidth-bound");
    const double bank_part = 1e9 * 10 / 100; // k of E experts' bytes at T = 1
    check(decode && decode->bytes >= bank_part && decode->bytes < bank_part * 1.01,
          "its bytes are the touched experts' (+ the small activation and output)");
    const auto wide = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 4096, board);
    check(wide && wide->compute_seconds > wide->dram_seconds, "a wide stage is compute-bound");
    check(wide && wide->seconds == wide->compute_seconds, "seconds is the larger term");
    const auto r8 = moe::price_moe_stage(s, moe::MoeStageKernel::R8, 4096, board);
    check(r8 && !r8->compute_priced && r8->seconds == r8->dram_seconds,
          "R8's work has no term: its price is the DRAM floor, said so");
    const auto w4 = moe::price_moe_stage(s, moe::MoeStageKernel::W4a8, 4096, board);
    check(w4 && w4->compute_seconds < wide->compute_seconds, "the int8 tier computes faster");
    check(!moe::price_moe_stage(s, moe::MoeStageKernel::W4a8, 1,
                                moe::MoeBoardFacts{1e12, 100, 2e9, 1536.0, 0.0}),
          "no int8 tensor cores: no W4A8 price");
    check(!moe::price_moe_stage(moe::MoeStageShape{}, moe::MoeStageKernel::R8, 1, board),
          "an empty shape has no price");
}

// The candidates of a layer on boards of three generations: the statuses the law gives.
void candidate_checks() {
    const auto layer = build(QType::GGML_IQ3_XXS, QType::GGML_IQ4_XS, 256, 8, 2048, 512, false);
    for (const char* name : {"rtx4090", "rtx2080ti", "v100"}) {
        const generic::FactsBoard& fb = board_named(name);
        const DeviceClass d           = generic::board_device_class(fb);
        const ScopedDeviceClass scoped(d);
        for (const std::int32_t t : {1, 4, 64}) {
            for (const moe::Stage stage : {moe::Stage::GateUp, moe::Stage::Down}) {
                const auto candidates =
                    moe::moe_candidates(layer->weights, layer->config, t, false, stage);
                int selected = 0;
                for (const auto& c : candidates) {
                    selected += c.status == moe::MoeCandidate::Status::Selected;
                    if (c.status == moe::MoeCandidate::Status::Inadmissible) {
                        check(!c.why.empty() && !c.seconds,
                              std::string(name) + ": an inadmissible candidate says why");
                    }
                    if (c.status == moe::MoeCandidate::Status::Dominated ||
                        c.status == moe::MoeCandidate::Status::PricedBelow) {
                        check(c.kernel != moe::MoeStageKernel::R8,
                              "R8 is never ranked by a law that does not price it");
                    }
                }
                check(selected == 1, std::string(name) + " T=" + std::to_string(t) +
                                         ": exactly one candidate is the selected kind");
                check(candidates.size() == 4, "one entry per kind");
            }
        }
    }
}

struct Row {
    double predicted, measured;
};

double geomean_ratio(const std::vector<Row>& rows) {
    double sum = 0;
    for (const Row& r : rows) sum += std::log(r.predicted / r.measured);
    return std::exp(sum / static_cast<double>(rows.size()));
}

// The sweeps on disk, one board at a time.
void replica(const std::filesystem::path& routes_dir) {
    struct Source {
        const char* file;
        const char* board;
    };

    const Source sources[] = {{"ad106.json", "rtx4060ti"},
                              {"gb202.json", "rtx5090"},
                              {"gb203.json", "rtx5080"},
                              {"gb203-70sm.json", "rtx5070ti"},
                              {"gb206.json", "rtx5060ti"}};
    std::printf("replica: price_moe_stage (bytes law + roofline, no fitted constant) against the "
                "sweeps in routes/*.json\n");
    std::printf("%-10s %5s %12s %8s %8s %12s %9s\n", "board", "rows", "geomean p/m", "min", "max",
                "concordance", "pred ties");
    for (const Source& source : sources) {
        std::ifstream in(routes_dir / source.file);
        check(static_cast<bool>(in), std::string("cannot read ") + source.file);
        const Json table    = Json::parse(in);
        const DeviceClass d = generic::board_device_class(board_named(source.board));
        const ScopedDeviceClass scoped(d);
        const auto facts = moe::moe_board_facts(d);
        check(facts.has_value(), std::string(source.board) + ": the board has a roofline");
        if (!facts) continue;
        std::vector<Row> rows;
        for (const Json& e : table.value("moe", Json::array())) {
            const auto q  = qtype_of(e.at("format").get<std::string>());
            const auto dq = qtype_of(e.value("down_format", e.at("format").get<std::string>()));
            if (!q || !dq) continue;
            const std::int32_t experts = e.at("experts"), top_k = e.at("top_k"),
                               hidden = e.at("hidden"), inter = e.at("inter");
            // Nemotron's mixer is the one non-gated (relu^2) layer of the tables.
            const bool relu2           = hidden == 2688;
            const auto layer           = build(*q, *dq, experts, top_k, hidden, inter, relu2);
            const moe::MoeRouteKey key = moe::route_key(layer->weights, layer->config);
            for (const auto& [field, prefill] :
                 {std::pair{"evidence_us", true}, std::pair{"evidence_decode_us", false}}) {
                if (!e.contains(field)) continue;
                for (const Json& row : e.at(field)) {
                    const std::int32_t t   = row.at(0);
                    const double chosen_us = row.at(2).get<double>();
                    const auto plan        = moe::table_plan(key, t, prefill);
                    if (!plan || chosen_us <= 0) continue;
                    const auto seconds =
                        moe::moe_layer_seconds(layer->weights, layer->config, *plan, t, *facts);
                    if (!seconds) continue;
                    rows.push_back({*seconds, chosen_us * 1e-6});
                }
            }
        }
        if (rows.empty()) {
            std::printf("%-10s %5d (no rows)\n", source.board, 0);
            continue;
        }
        double lo = 1e9, hi = 0;
        for (const Row& r : rows) {
            lo = std::min(lo, r.predicted / r.measured);
            hi = std::max(hi, r.predicted / r.measured);
        }
        long concordant = 0, discordant = 0, ties = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            for (std::size_t j = i + 1; j < rows.size(); ++j) {
                const double dp = rows[i].predicted - rows[j].predicted;
                const double dm = rows[i].measured - rows[j].measured;
                if (std::abs(dp) <= 1e-12 * std::abs(rows[i].predicted)) {
                    ++ties;
                } else if (dm == 0) {
                    continue;
                } else if ((dp > 0) == (dm > 0)) {
                    ++concordant;
                } else {
                    ++discordant;
                }
            }
        }
        const double concordance =
            concordant + discordant > 0
                ? static_cast<double>(concordant) / static_cast<double>(concordant + discordant)
                : 0.0;
        std::printf("%-10s %5zu %12.3f %8.3f %8.3f %12.3f %9ld\n", source.board, rows.size(),
                    geomean_ratio(rows), lo, hi, concordance, ties);
        // A floor: no stage runs faster than its bytes and its roofline, so the prediction never
        // exceeds the measurement by more than the uniform-routing assumption's slack.
        check(hi <= 1.10, std::string(source.board) +
                              ": a row's predicted floor exceeds its "
                              "measured time by more than 10 % (" +
                              std::to_string(hi) + ")");
        check(concordance > 0.5, std::string(source.board) + ": the prediction orders the rows "
                                                             "no better than chance");
    }
}

} // namespace

int main() {
    try {
        law_checks();
        candidate_checks();
        replica(std::filesystem::path(NINFERNO_ROUTES_DIR));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe cost: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "moe cost: %d checks failed\n", g_failures);
        return 1;
    }
    std::printf("moe cost: PASS\n");
    return 0;
}
