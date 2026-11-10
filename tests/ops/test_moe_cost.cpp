// The MoE stage price (ops/moe/moe_cost.h, #616 step 4): the law's own terms on small cases, the
// candidate list on a layer of the lab's models, and the replica: every MoE row of the sweeps on
// disk (routes/<class>.json "moe": rows [T, default us, chosen us]; "evidence_us" comes from the
// MoE sweep of the bench, "evidence_decode_us" from the load-time MoE tuner, the minimum over
// the candidates' medians), priced by price_moe_stage on that board's facts with the table's own
// plan, predicted against measured. Only the boards whose table holds that board's own sweep are
// replayed: RTX 4060 Ti, RTX 5090, RTX 5080, RTX 5060 Ti (the RTX 5070 Ti table's MoE section is
// the RTX 5080's sweep, so it is not a source). Every row of those tables is either replayed or
// counted under the reason it was not, the counts are printed, an unknown expert format fails
// the test, and the number of replayed rows per board is asserted exactly (a renamed field or a
// new format cannot empty the replica silently).
//
// Two checks, both verification thresholds chosen for this test (not measured facts of a chip,
// not published numbers):
//   - predicted / measured of every replayed row is at most 1.10. The price is the bytes and the
//     unit rates, a lower bound of a call that reads x and the weights once. It is not a physical
//     floor in the tuner's regime: the same x is fed every repetition, so the L2 serves its
//     re-reads (and part of the weights on a small bank), and a measured time can sit below the
//     price. 10 % is the slack the checked boards need over that effect (the largest ratio is
//     1.005, the RTX 4060 Ti).
//   - the rank agreement of predicted and measured times across the rows of a board (Kendall's
//     concordance over the pairs the prediction orders) is above 0.8; a tie in the prediction is
//     counted apart, the law having no term to order two plans of the same bytes. It is high
//     mostly because both times grow with T.
// Host only, no GPU; it changes no route (the moe plan goldens are the check of that).
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/moe.h"
#include "ops/generic/selector_facts.h"
#include "ops/moe/moe_cost.h"
#include "ops/moe/moe_internal.h"
#include "runtime/engine/interleave_cost.h"

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

    // The terms by hand, exactly: at T = 1 a gate/up stage touches 10 of 100 experts (1e8 bytes),
    // reads one row of x (1000 values, BF16) and writes 10 assignments x 500 rows (BF16).
    const auto close = [](double a, double b) { return std::abs(a - b) <= 1e-9 * std::abs(b); };
    check(decode && close(decode->bytes, 1e8 + 1000 * 2.0 + 10 * 500 * 2.0),
          "decode bytes: touched experts + x once + the outputs");
    check(decode && close(decode->dram_seconds, decode->bytes / 1e12), "DRAM seconds = bytes / bw");
    // 2 x 4096 x 10 assignments x 1000 rows x 1000 cols over 64 x 100 x 2e9 ops/s.
    check(wide && close(wide->compute_seconds, 2.0 * 4096 * 10 * 1000 * 1000 / (64.0 * 100 * 2e9)),
          "compute seconds = 2 x assignments x rows x cols / (rate x SMs x clock)");
    check(wide && w4 && close(wide->compute_seconds / w4->compute_seconds, 2048.0 / 64.0),
          "the int8 tier's compute time is the rate ratio, no other factor");
    // A down stage reads its T x k activation rows (10 x as many as a gate/up's x).
    moe::MoeStageShape down = s;
    down.stage              = moe::Stage::Down;
    const auto down_price = moe::price_moe_stage(down, moe::MoeStageKernel::RegisterMma, 1, board);
    check(down_price && close(down_price->bytes, 1e8 + 10 * 1000 * 2.0 + 10 * 500 * 2.0),
          "a down stage reads one activation row per assignment");
    // The int8 activation: 36 bytes a 32 values, 40 with a block sum (a format with a min).
    moe::MoeStageShape k4 = s;
    k4.qtype              = QType::GGML_Q4_K;
    const auto a8_plain   = moe::price_moe_stage(s, moe::MoeStageKernel::W4a8, 1, board);
    const auto a8_min     = moe::price_moe_stage(k4, moe::MoeStageKernel::W4a8, 1, board);
    check(a8_plain && a8_min && close(a8_min->bytes - a8_plain->bytes, 1000 * 4.0 / 32.0),
          "a format with a min adds the block sum to the int8 activation");

    // Where the law has no rate fact the compute term is absent and says why.
    moe::MoeStageShape mx = s;
    mx.mx8                = true;
    const auto mx8        = moe::price_moe_stage(mx, moe::MoeStageKernel::W4a8, 4096, board);
    check(mx8 && !mx8->compute_priced && !mx8->compute_note.empty() &&
              mx8->seconds == mx8->dram_seconds,
          "the block-scaled MXFP8 stage is not priced at the int8 rate");
    check(w4 && w4->compute_priced && w4->compute_note.empty(), "the plain W4A8 stage is priced");
    moe::MoeBoardFacts emulated = board;
    emulated.bf16_native        = false;
    for (const moe::MoeStageKernel kind :
         {moe::MoeStageKernel::RegisterMma, moe::MoeStageKernel::VoltaHmma}) {
        const auto p = moe::price_moe_stage(s, kind, 4096, emulated);
        check(p && !p->compute_priced && !p->compute_note.empty() && p->seconds == p->dram_seconds,
              "no native BF16 MMA: the fitted aggregate rate does not price the compute");
    }

    // The ranking of two prices: an unpriced side is not ranked, equal prices tie.
    std::string why;
    using Status     = moe::MoeCandidate::Status;
    const auto cheap = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 4096, board);
    const auto same  = moe::price_moe_stage(s, moe::MoeStageKernel::VoltaHmma, 4096, board);
    check(cheap && same && moe::rank_against_selected(same, cheap, why) == Status::Tied &&
              !why.empty(),
          "two kinds of equal price are a tie, not one dominated");
    check(cheap && w4 && moe::rank_against_selected(w4, cheap, why) == Status::PricedBelow &&
              moe::rank_against_selected(cheap, w4, why) == Status::Dominated,
          "a cheaper kind is priced below, a dearer one dominated");
    const auto emulated_a =
        moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 4096, emulated);
    const auto emulated_b = moe::price_moe_stage(s, moe::MoeStageKernel::VoltaHmma, 4096, emulated);
    check(emulated_a && emulated_b &&
              moe::rank_against_selected(emulated_a, emulated_b, why) == Status::Unpriced &&
              !why.empty(),
          "Volta's HMMA and the emulated register MMA are not ranked by the fitted rate");
    check(moe::rank_against_selected(std::nullopt, cheap, why) == Status::Unpriced,
          "no price, no rank");
}

// The expert-coverage law has several copies; this one is bound to the runtime's (fixed point,
// runtime/engine/interleave_cost.h) on a grid, so that they cannot drift apart unseen.
void coverage_binding_checks() {
    for (const std::uint32_t experts : {8U, 32U, 64U, 128U, 256U, 512U}) {
        for (const std::uint32_t top_k : {1U, 2U, 4U, 8U, 22U}) {
            if (top_k > experts) continue;
            for (const std::uint32_t tokens : {0U, 1U, 2U, 3U, 7U, 16U, 64U, 300U}) {
                const double here = moe::expected_experts_touched(
                    static_cast<std::int32_t>(experts), static_cast<std::int32_t>(top_k),
                    static_cast<std::int32_t>(tokens));
                const double there =
                    static_cast<double>(
                        ninferno::runtime::touched_expert_fraction_q32(experts, top_k, tokens)) /
                    4294967296.0 * experts;
                check(std::abs(here - there) <= 1e-6 * experts,
                      "expert coverage E=" + std::to_string(experts) +
                          " k=" + std::to_string(top_k) + " T=" + std::to_string(tokens) + ": " +
                          std::to_string(here) + " vs the runtime's " + std::to_string(there));
            }
        }
    }
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
                    const bool ranked = c.status == moe::MoeCandidate::Status::Dominated ||
                                        c.status == moe::MoeCandidate::Status::PricedBelow ||
                                        c.status == moe::MoeCandidate::Status::Tied;
                    if (ranked) {
                        check(c.kernel != moe::MoeStageKernel::R8,
                              "R8 is never ranked by a law that does not price it");
                        // Volta and Turing have no native BF16 MMA: the generation's BF16 rate is
                        // a fitted aggregate there, so no kind is ranked by the law on them.
                        check(d.features.mma_bf16_native,
                              std::string(name) + ": a kind ranked without a native BF16 MMA");
                    }
                    if (c.status == moe::MoeCandidate::Status::Unpriced) {
                        check(!c.why.empty(), std::string(name) + ": an unpriced kind says why");
                    }
                }
                check(selected == 1, std::string(name) + " T=" + std::to_string(t) +
                                         ": exactly one candidate is the selected kind");
                check(candidates.size() == 4, "one entry per kind");
            }
        }
    }
}

// The verification thresholds of the header (chosen for this test, not measured facts).
constexpr double kFloorTolerance = 1.10;
constexpr double kMinConcordance = 0.8;
// The tables carry no activation field; the one non-gated (relu^2) layer is Nemotron's.
constexpr std::int32_t kRelu2Hidden = 2688;

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
        std::size_t expected_rows; // the rows of its table the replica replays, exactly
    };

    const Source sources[] = {{"ad106.json", "rtx4060ti", 19},
                              {"gb202.json", "rtx5090", 9},
                              {"gb203.json", "rtx5080", 23},
                              {"gb206.json", "rtx5060ti", 24}};
    std::printf("replica: price_moe_stage (bytes law + roofline) against the sweeps in "
                "routes/*.json\n");
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
        // The rows of the table not replayed, by reason.
        std::map<std::string, int> skipped;
        for (const Json& e : table.value("moe", Json::array())) {
            const std::string format = e.at("format").get<std::string>();
            const std::string down   = e.value("down_format", format);
            const auto q             = qtype_of(format);
            const auto dq            = qtype_of(down);
            if (!q || !dq) {
                check(false,
                      std::string(source.board) +
                          ": an expert format the test cannot build: " + (q ? down : format));
                continue;
            }
            const std::int32_t experts = e.at("experts"), top_k = e.at("top_k"),
                               hidden = e.at("hidden"), inter = e.at("inter");
            // The tables carry no activation field: Nemotron's mixer (hidden 2688) is the one
            // non-gated (relu^2) layer of them.
            const bool relu2 = hidden == kRelu2Hidden;
            std::unique_ptr<Built> layer;
            try {
                layer = build(*q, *dq, experts, top_k, hidden, inter, relu2);
            } catch (const std::invalid_argument& error) {
                ++skipped[std::string("layer the fixture cannot pack: ") + error.what()];
                continue;
            }
            const moe::MoeRouteKey key = moe::route_key(layer->weights, layer->config);
            for (const auto& [field, prefill] :
                 {std::pair{"evidence_us", true}, std::pair{"evidence_decode_us", false}}) {
                if (!e.contains(field)) continue;
                for (const Json& row : e.at(field)) {
                    const std::int32_t t   = row.at(0);
                    const double chosen_us = row.at(2).get<double>();
                    const auto plan        = moe::table_plan(key, t, prefill);
                    if (chosen_us <= 0) {
                        ++skipped["no measured time"];
                        continue;
                    }
                    if (!plan) {
                        ++skipped["no table plan at that T"];
                        continue;
                    }
                    const auto seconds =
                        moe::moe_layer_seconds(layer->weights, layer->config, *plan, t, *facts);
                    if (!seconds) {
                        ++skipped["no price"];
                        continue;
                    }
                    rows.push_back({*seconds, chosen_us * 1e-6});
                }
            }
        }
        for (const auto& [reason, count] : skipped) {
            std::printf("%-10s skipped %d row(s): %s\n", source.board, count, reason.c_str());
        }
        check(rows.size() == source.expected_rows,
              std::string(source.board) + ": replayed " + std::to_string(rows.size()) +
                  " rows, expected " + std::to_string(source.expected_rows));
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
        // The verification thresholds of the header: the price is not a physical floor where the
        // same x is read every repetition, so a row may exceed 1 by the L2's share.
        check(hi <= kFloorTolerance, std::string(source.board) +
                                         ": a row's predicted time exceeds its measured time by "
                                         "more than the tolerance (" +
                                         std::to_string(hi) + " > " +
                                         std::to_string(kFloorTolerance) + ")");
        check(concordance > kMinConcordance,
              std::string(source.board) + ": the prediction orders the rows no better than " +
                  std::to_string(kMinConcordance) + " of the pairs (" +
                  std::to_string(concordance) + ")");
    }
}

} // namespace

int main() {
    try {
        law_checks();
        candidate_checks();
        coverage_binding_checks();
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
