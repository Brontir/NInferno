// The choice's prefill estimate asks the selector's route (#613 PR-9; product/choice
// ModelFacts::prefill_routes, filled by product/plan with_prefill_routes from the registered
// runtime::register_plan_prefill_route_query, the generic selector's route_query):
//   - every measured prefill point (tests/fixtures/choice/prefill_measured.json, the lab's cards)
//     priced with the query against without it: a point moves only where the selector runs a
//     linear on another kernel than the tile laws assume, and the moves are the ones listed here
//     (a Q8_0 linear in a K-quant file: the W4A8 int8 tile, not MMQ's tile or the K-quant W4A8
//     tile at the file's K step);
//   - every board of the registry (core/data/gpu_facts.json) at the fixtures' models and
//     quantizations: every estimate that moves is printed and counted by generation, and none moves
//     on a quantization without such a linear.
// Host only, no GPU.

#include "choice_fixture.h"
#include "core/board_registry.h"
#include "core/device_class.h"
#include "product/choice/choice.h"
#include "product/plan/plan_choice.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_prefill_routes.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#ifndef NINFERNO_SOURCE_DIR
#    error "NINFERNO_SOURCE_DIR must name the source tree"
#endif

namespace {

namespace ch = ninferno::product::choice;
namespace pr = ninferno::product;
namespace rt = ninferno::runtime;
using choice_fixture::card_of;
using choice_fixture::model_at;
using choice_fixture::quant_of;
using ninferno::KvCacheStorage;
using nlohmann::json;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

json load(const std::string& path) {
    std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/" + path);
    return json::parse(in);
}

// The planned device of a registry row, as a named board's plan gives it.
rt::PlanDeviceFacts device_of(const ninferno::board_registry::BoardRow& row) {
    rt::PlanDeviceFacts d;
    d.pci_device_id      = row.pci_device_id;
    d.compute_capability = row.major * 10 + row.minor;
    d.sms                = row.sms;
    d.total_bytes        = rt::plan_board_total_bytes(row.vram_gib);
    return d;
}

// The fixture's cards by the board their rows were measured on.
const std::map<std::string, std::uint32_t> kFixtureBoards = {
    {"RTX 4060 Ti", 0x2805}, {"RTX 4070 Ti", 0x2782}, {"RTX 4090", 0x2684},
    {"RTX 3060", 0x2504},    {"RTX 2080 Ti", 0x1E07}, {"RTX 5070 Ti", 0x2C05},
    {"RTX 5090", 0x2B85},    {"Quadro GV100", 0x1DBA}};

// Whether a quantization of the fixture names a Q8_0 linear among its roles' formats.
bool names_q8_linear(const json& model, const std::string& quant) {
    const json& q = model.at("quants").at(quant);
    if (!q.contains("role_formats")) { return false; }
    for (const auto& [role, shares] : q.at("role_formats").items()) {
        if (shares.contains("q8_g32_fp16")) { return true; }
    }
    return false;
}

struct Estimate {
    double prefill       = 0;
    ch::SpeedBasis basis = ch::SpeedBasis::Own;
};

Estimate estimate(const ch::ModelFacts& model, const ch::CardFacts& card,
                  const ch::QuantOption& quant, std::uint32_t prompt) {
    ch::Candidate c;
    c.kv              = KvCacheStorage::Int8Group64;
    const ch::Speed s = ch::speed_of(model, card, quant, c, 1, prompt, prompt);
    return {s.prefill, s.basis};
}

bool moved(const Estimate& a, const Estimate& b) {
    return a.basis != b.basis || std::abs(b.prefill - a.prefill) > 1e-12 * std::abs(a.prefill);
}

// The measured points: a point moves only on a quantization with a Q8_0 linear (the W4A8 int8
// tile the selector runs it on), Volta never (its CUTLASS route is not asked), and the moves are
// these (card, model, quantization).
void measured_points(const json& fixture) {
    const std::set<std::tuple<std::string, std::string, std::string>> expected = {
        {"RTX 2080 Ti", "Qwen3.5-9B", "Q6_K"}, {"RTX 2080 Ti", "Qwen3.5-9B", "UD-Q4_K_XL"},
        {"RTX 3060", "Qwen3.5-9B", "Q6_K"},    {"RTX 3060", "Qwen3.5-9B", "UD-Q4_K_XL"},
        {"RTX 4060 Ti", "Qwen3.5-9B", "Q6_K"}, {"RTX 4060 Ti", "Qwen3.5-9B", "UD-Q4_K_XL"},
        {"RTX 4070 Ti", "Qwen3.5-9B", "Q6_K"}, {"RTX 4070 Ti", "Qwen3.5-9B", "UD-Q4_K_XL"},
        {"RTX 4090", "Qwen3.5-9B", "Q6_K"},    {"RTX 4090", "Qwen3.5-9B", "UD-Q4_K_XL"},
        {"RTX 5070 Ti", "Qwen3.5-9B", "Q6_K"}, {"RTX 5070 Ti", "Qwen3.5-9B", "UD-Q4_K_XL"},
        {"RTX 5090", "Qwen3.5-9B", "Q6_K"},    {"RTX 5090", "Qwen3.5-9B", "UD-Q4_K_XL"},
    };
    std::set<std::tuple<std::string, std::string, std::string>> seen;
    double sq_before = 0, sq_after = 0, sq_moved_before = 0, sq_moved_after = 0;
    int points = 0, moves = 0;
    for (const json& row : fixture.at("rows")) {
        const std::string card_name  = row.at("card").get<std::string>();
        const std::string model_name = row.at("model").get<std::string>();
        const std::string q          = row.at("quant").get<std::string>();
        const json& m                = fixture.at("models").at(model_name);
        const ch::CardFacts card     = card_of(fixture.at("cards").at(card_name));
        const ch::QuantOption quant  = quant_of(q, m.at("quants").at(q));
        const std::uint32_t prompt   = row.at("prompt").get<std::uint32_t>();
        const double measured        = row.at("prefill").get<double>();
        ch::ModelFacts model         = model_at(m, q);
        const Estimate before        = estimate(model, card, quant, prompt);
        pr::with_prefill_routes(
            model, device_of(ninferno::board_registry::row(kFixtureBoards.at(card_name))));
        check(static_cast<bool>(model.prefill_routes), "the query is asked on " + card_name);
        const Estimate after = estimate(model, card, quant, prompt);
        const double e0 = before.prefill / measured - 1, e1 = after.prefill / measured - 1;
        sq_before += e0 * e0, sq_after += e1 * e1, ++points;
        if (!moved(before, after)) { continue; }
        ++moves;
        sq_moved_before += e0 * e0, sq_moved_after += e1 * e1;
        seen.insert({card_name, model_name, q});
        std::cout << "  moved: " << card_name << ' ' << model_name << ' ' << q << ' ' << prompt
                  << ": " << before.prefill << " -> " << after.prefill << " tok/s ("
                  << ch::speed_basis_name(before.basis) << " -> "
                  << ch::speed_basis_name(after.basis) << "), measured " << measured << ": "
                  << std::lround(e0 * 1000) / 10.0 << " % -> " << std::lround(e1 * 1000) / 10.0
                  << " %\n";
        check(names_q8_linear(m, q),
              "a measured point moves only where a Q8_0 linear changes kernel: " + card_name + ' ' +
                  model_name + ' ' + q);
    }
    std::cout << "measured prefill points: " << points << ", moved " << moves << "; rms "
              << std::sqrt(sq_before / points) * 100 << " % -> "
              << std::sqrt(sq_after / points) * 100 << " %; the moved points' rms "
              << std::sqrt(sq_moved_before / std::max(moves, 1)) * 100 << " % -> "
              << std::sqrt(sq_moved_after / std::max(moves, 1)) * 100 << " %\n";
    check(seen == expected, "the measured points that move are the listed ones");
}

// Every board of the registry with tensor cores at the fixtures' models and quantizations, a
// 7K-token prompt: the estimates that move, by generation; none on a quantization without a Q8_0
// linear.
void every_board(const json& fixture) {
    std::map<std::string, int> moves_by_generation;
    int estimates = 0, moves = 0;
    for (const ninferno::board_registry::BoardRow& row : ninferno::board_registry::kRows) {
        if (row.mma_slow) { continue; }
        const rt::PlanDeviceFacts device = device_of(row);
        ninferno::DeviceClass d          = ninferno::synthetic_device_class(
            row.pci_device_id, row.major, row.minor, row.sms, 0, device.total_bytes);
        d.facts                  = ninferno::machine_facts_of(d);
        const ch::CardFacts card = ch::card_facts_of(d);
        for (const auto& [model_name, m] : fixture.at("models").items()) {
            for (const auto& [q, qj] : m.at("quants").items()) {
                const ch::QuantOption quant = quant_of(q, qj);
                ch::ModelFacts model        = model_at(m, q);
                const Estimate before       = estimate(model, card, quant, 7000);
                pr::with_prefill_routes(model, device);
                const Estimate after = estimate(model, card, quant, 7000);
                ++estimates;
                if (!moved(before, after)) { continue; }
                ++moves;
                ++moves_by_generation[std::string(ch::generation_of(card))];
                std::cout << "  board " << std::hex << row.pci_device_id << std::dec << " ("
                          << ch::generation_of(card) << ") " << model_name << ' ' << q << ": "
                          << before.prefill << " -> " << after.prefill << " tok/s ("
                          << ch::speed_basis_name(before.basis) << " -> "
                          << ch::speed_basis_name(after.basis) << ")\n";
                check(names_q8_linear(m, q), "a board's estimate moves only where a Q8_0 linear "
                                             "changes kernel: " +
                                                 model_name + ' ' + q);
            }
        }
    }
    std::cout << "every board: " << estimates << " estimates, " << moves << " moved;";
    for (const auto& [generation, n] : moves_by_generation) {
        std::cout << ' ' << generation << ' ' << n;
    }
    std::cout << '\n';
}

} // namespace

int main() {
    const json fixture = load("tests/fixtures/choice/prefill_measured.json");
    // No query registered (a program that does not link the Ops: ninferno-ui): not asked.
    {
        const json& m          = fixture.at("models").at("Qwen3.5-9B");
        ch::ModelFacts unasked = model_at(m, "Q6_K");
        pr::with_prefill_routes(unasked, device_of(ninferno::board_registry::row(0x2684)));
        check(!unasked.prefill_routes, "no query registered: the prefill routes are not asked");
    }
    rt::register_plan_prefill_route_query();
    // The selector's own answers on the RTX 4090 at a 1024-token chunk (the probe of the issue):
    // Q8_0 on the W4A8 tiles, Q4_K and Q6_K on MMQ; a gated pair asked as SwiGLU on its half.
    {
        const auto& query                 = rt::plan_prefill_route_query();
        const rt::PlanDeviceFacts rtx4090 = device_of(ninferno::board_registry::row(0x2684));
        const auto fast                   = ninferno::ActivationMode::Fast;
        check(query(rtx4090, "q8_g32_fp16", 4096, 4096, false, 1024, fast) ==
                      rt::PlanPrefillKernel::W4a8 &&
                  query(rtx4090, "q8_g32_fp16", 24576, 4096, true, 1024, fast) ==
                      rt::PlanPrefillKernel::W4a8 &&
                  query(rtx4090, "ggml_q4_k", 4096, 4096, false, 1024, fast) ==
                      rt::PlanPrefillKernel::Mmq &&
                  query(rtx4090, "ggml_q6_k", 4096, 12288, false, 1024, fast) ==
                      rt::PlanPrefillKernel::Mmq,
              "the selector's prefill routes on the RTX 4090");
        check(query(rtx4090, "ggml_q4_k", 4096, 4096, false, 1024,
                    ninferno::ActivationMode::Exact) == rt::PlanPrefillKernel::A16 &&
                  query(rtx4090, "not_a_format", 4096, 4096, false, 1024, fast) ==
                      rt::PlanPrefillKernel::Unknown,
              "exact activations: a 16-bit kernel; a format the facts do not name: no answer");
    }
    measured_points(fixture);
    every_board(fixture);
    if (failures == 0) { std::cout << "choice prefill routes: OK\n"; }
    return failures == 0 ? 0 : 1;
}
