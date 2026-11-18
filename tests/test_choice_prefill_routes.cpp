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
#include "product/choice/choice_replay.h"
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
    // #613 PR-9's gate: the measured points' rms with the selector's routes at most 6.4 %.
    check(points > 0 && std::sqrt(sq_after / points) <= 0.064,
          "the prefill estimate on the selector's routes within #613 PR-9's 6.4 % rms");
}

// #931: these are the production callback cases that either changed twelve candidate prices
// or threw ReplayIncomplete when the untouched record was read back. No model weights or GPU.
void production_choices_replay(const json& fixture) {
    for (const auto& [name, label, id, prompt] :
         {std::tuple{"Qwen3.5-9B", "Q6_K", 0x2684U, 2048U},
          std::tuple{"Qwen3.5-4B", "Q4_K_M", 0x2C05U, 22000U}}) {
        rt::register_plan_prefill_route_query();
        const auto& m     = fixture.at("models").at(name);
        auto model        = model_at(m, label);
        const auto quant  = quant_of(label, m.at("quants").at(label));
        const auto device = device_of(ninferno::board_registry::row(id));
        auto d            = ninferno::synthetic_device_class(id, device.compute_capability / 10,
                                                             device.compute_capability % 10, device.sms, 0,
                                                             device.total_bytes);
        d.facts           = ninferno::machine_facts_of(d);
        const auto card   = ch::card_facts_of(d);
        pr::with_prefill_routes(model, device);
        const auto allowed = ch::allowed_axes(model, KvCacheStorage::Int8Group64);
        const auto own     = [](const ch::Candidate&) { return 32768U; };
        ch::Task task;
        task.contexts     = {32768};
        task.prompt       = prompt;
        const auto direct = ch::choose(model, card, quant, allowed, {}, task, own);
        const auto record = ch::record_choice(model, card, quant, allowed, {}, task, own);
        check(ch::choice_summary(record.choice) == ch::choice_summary(direct),
              std::string(name) + ": recording leaves the production choice unchanged");
        // Neither the original callback nor a new registration is available for replay.
        model.prefill_routes = {};
        rt::set_plan_prefill_route_query({});
        try {
            const auto again =
                ch::replay_choice(nlohmann::ordered_json::parse(record.inputs.dump()));
            check(ch::choice_summary(again) == ch::choice_summary(direct),
                  std::string(name) + ": every production candidate and explanation replays");
        } catch (const std::exception& e) {
            check(false, std::string(name) + ": its own complete record was refused: " + e.what());
        }
    }
    rt::register_plan_prefill_route_query();
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

// #847: a final rest under 17 tokens. A request's prompt runs as whole chunks and a partial one,
// each priced at its own route (single_request_seconds); at an 8-token rest the selector may take
// R8 or the vector kernel (the int8 share) where the chunks run on the W4A8 tiles. The rest alone
// is a 1032-token request less a 1024-token one in 1024-token chunks: attention, the tail calls
// and decode read no prefill route, so the difference is the rest's linears. Per fixture board:
// the selector's answer for a Q8_0 linear at 8 and at 1024 tokens (printed), the rests that move
// once the routes are asked (listed), none on a quantization without a Q8_0 linear, none on Volta
// (its prefill is its CUTLASS route's, which the choice does not ask: selected_kernel), and at
// least one move over the boards, so the check is not vacuous.
void short_rest(const json& fixture) {
    const auto& query = rt::plan_prefill_route_query();
    const auto fast   = ninferno::ActivationMode::Fast;
    int rests = 0, moves = 0;
    for (const auto& [card_name, id] : kFixtureBoards) {
        const rt::PlanDeviceFacts device = device_of(ninferno::board_registry::row(id));
        const bool volta                 = device.compute_capability == 70;
        const rt::PlanPrefillKernel rest_kernel =
            query(device, "q8_g32_fp16", 4096, 4096, false, 8, fast);
        const rt::PlanPrefillKernel chunk_kernel =
            query(device, "q8_g32_fp16", 4096, 4096, false, 1024, fast);
        std::cout << "  " << card_name << ": a Q8_0 4096 x 4096 call on kernel "
                  << static_cast<int>(rest_kernel) << " at 8 tokens, "
                  << static_cast<int>(chunk_kernel)
                  << " at 1024 (0 unknown, 1 MMQ, 2 W4A8, 3 other A8, 4 A16)\n";
        check(volta || rest_kernel != rt::PlanPrefillKernel::Unknown,
              card_name + ": the selector answers an 8-token Q8_0 call");
        const ch::CardFacts card = card_of(fixture.at("cards").at(card_name));
        for (const auto& [model_name, m] : fixture.at("models").items()) {
            for (const auto& [q, qj] : m.at("quants").items()) {
                const ch::QuantOption quant = quant_of(q, qj);
                ch::ModelFacts model        = model_at(m, q);
                ch::Candidate c;
                c.kv    = KvCacheStorage::Int8Group64;
                c.chunk = 1024;
                ch::Task task;
                task.answer     = 64;
                const auto rest = [&] {
                    return ch::single_request_seconds(model, card, quant, c, task, 1032) -
                           ch::single_request_seconds(model, card, quant, c, task, 1024);
                };
                const double before = rest();
                pr::with_prefill_routes(model, device);
                const double after = rest();
                ++rests;
                if (!(std::abs(after - before) > 1e-9 * std::abs(before))) { continue; }
                ++moves;
                std::cout << "  rest moved: " << card_name << ' ' << model_name << ' ' << q << ": "
                          << before * 1e3 << " -> " << after * 1e3 << " ms\n";
                check(!volta, card_name +
                                  ": a Volta rest moves (the choice asks no route there): " +
                                  model_name + ' ' + q);
                // A Q8_0 file names no role formats: its linears take the format from the label
                // (selected_kernel), and on the int8 tiles any 8-token answer but W4A8 moves them.
                check(names_q8_linear(m, q) || q == "Q8_0",
                      "an 8-token rest moves only where a Q8_0 linear changes kernel: " +
                          card_name + ' ' + model_name + ' ' + q);
            }
        }
    }
    std::cout << "short rest: " << rests << " 8-token rests, " << moves << " moved\n";
    check(moves > 0, "some 8-token rest moves with the selector's routes");
}

// #930: a real board's selector is useful for the displayed estimate, but it is not the
// synthetic generation reference's selector. In particular, the 4B/22K case used to choose
// INT8 KV on the 5090 and FP8 on the 5070 Ti with identical capacity.
void reference_prices(const json& fixture) {
    for (const auto& [name, label] :
         {std::pair{"Qwen3.5-4B", "Q4_K_M"}, std::pair{"Qwen3.5-9B", "Q6_K"}}) {
        const auto& m    = fixture.at("models").at(name);
        const auto quant = quant_of(label, m.at("quants").at(label));
        for (const auto& ids : {std::pair{0x2B85U, 0x2C05U}, std::pair{0x2684U, 0x2782U}}) {
            for (const std::uint32_t prompt : {2048U, 2065U, 22000U}) {
                ch::Task task;
                task.contexts         = {32768};
                task.prompt           = prompt;
                const auto own        = [](const ch::Candidate&) { return 32768U; };
                const auto base_model = model_at(m, label);
                const auto allowed    = ch::allowed_axes(base_model, KvCacheStorage::Int8Group64);
                std::optional<ch::Choice> reference;
                for (const auto id : {ids.first, ids.second}) {
                    const auto device = device_of(ninferno::board_registry::row(id));
                    auto d  = ninferno::synthetic_device_class(id, device.compute_capability / 10,
                                                               device.compute_capability % 10,
                                                               device.sms, 0, device.total_bytes);
                    d.facts = ninferno::machine_facts_of(d);
                    const auto card = ch::card_facts_of(d);
                    const auto baseline =
                        ch::choose(base_model, card, quant, allowed, {}, task, own);
                    auto model = base_model;
                    pr::with_prefill_routes(model, device);
                    check(bool(model.prefill_routes),
                          "reference regression uses the production query");
                    const auto actual    = ch::choose(model, card, quant, allowed, {}, task, own);
                    const auto& expected = reference ? *reference : baseline;
                    const std::string at = std::string(name) + '/' + label +
                                           " prompt=" + std::to_string(prompt) +
                                           " board=" + std::to_string(id);
                    check(actual.chosen && actual.chosen == expected.chosen &&
                              actual.by_worth == expected.by_worth &&
                              actual.resolution_basis == expected.resolution_basis &&
                              actual.scored.size() == expected.scored.size(),
                          at + ": production routes do not change the reference choice or band");
                    if (actual.scored.size() == expected.scored.size()) {
                        for (std::size_t i = 0; i < actual.scored.size(); ++i) {
                            const auto& a = actual.scored[i];
                            const auto& e = expected.scored[i];
                            check(a.decision_seconds == e.decision_seconds &&
                                      a.decision_basis == e.decision_basis && a.worth == e.worth,
                                  at + ": reference price/basis of candidate " + std::to_string(i));
                            check(a.seconds ==
                                      ch::request_seconds(model, card, quant, a.candidate, task),
                                  at + ": displayed seconds keep the board's production routes");
                        }
                    }
                    reference = baseline;
                }
            }
        }
    }
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
        // A BF16 linear asked at AllowA8 (Fast) answers as its recorded A16Only would: no A8 route
        // reads a 16-bit weight.
        check(query(rtx4090, "bf16", 4096, 4096, false, 1024, fast) == rt::PlanPrefillKernel::A16,
              "a BF16 linear under fast activations: a 16-bit kernel");
    }
    production_choices_replay(fixture);
    measured_points(fixture);
    every_board(fixture);
    short_rest(fixture);
    reference_prices(fixture);
    if (failures == 0) { std::cout << "choice prefill routes: OK\n"; }
    return failures == 0 ? 0 : 1;
}
