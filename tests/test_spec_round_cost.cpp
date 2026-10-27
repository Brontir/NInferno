// A speculation round's cost from facts (spec/round_cost.h): the formula on the registry's own
// pairs (golden numbers from choice-notes/lab/round_cost/fit_eta.py), the cards' rates from the
// PCI-ID table, and a held-out card: the plan's speedup for a card whose measurement is left out,
// against that measurement. No GPU.

#include "spec/pairs.h"
#include "spec/registry.h"
#include "spec/round_cost.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace sp = ninferno::spec;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool near(double a, double b, double rel) { return std::abs(a / b - 1) <= rel; }

const sp::MeasuredPair& pair_of(std::string_view repo_prefix, std::string_view gpu) {
    for (const auto& p : sp::Registry::builtin().pairs()) {
        if (std::string_view(p.target_repo).starts_with(repo_prefix) && p.tune.key.gpu == gpu) {
            return p;
        }
    }
    std::cerr << "no registry pair " << repo_prefix << " on " << gpu << '\n';
    std::exit(1);
}

// A card as a start sees it (spec::gpu_identity): its class chain and its rates.
sp::GpuKey card(const char* id, const char* chip, const char* arch, int cc, int sms, int clock_mhz,
                double dram_gbps) {
    sp::GpuKey g;
    g.chain              = {id, chip, arch, "*"};
    g.own_keys           = 3;
    g.memory_bytes       = 12ull << 30;
    g.sms                = sms;
    g.compute_capability = cc;
    g.sm_clock_khz       = clock_mhz * 1000;
    g.dram_gbps          = dram_gbps;
    g.nvfp4              = cc >= 120;
    return g;
}

constexpr const char* kQ9 = "unsloth/Qwen3.5-9B-MTP-GGUF@9716a636ee4bddc3fed678220b7a33dd2a4160ae Q4_K_M";

} // namespace

int main() {
    // The facts read back as written; malformed facts are refused.
    {
        const auto& q9 = pair_of(kQ9, "ad104:60sm");
        expect(q9.round_facts.has_value() && q9.lab_board == 0x2782,
               "the registry's pairs carry their round facts and lab board");
        const auto again = sp::parse_round_facts(sp::round_facts_json(*q9.round_facts));
        expect(again == *q9.round_facts, "round facts round-trip through their JSON");
        const auto pair_again = sp::parse_pair(sp::pair_json(q9));
        expect(pair_again.round_facts == q9.round_facts && pair_again.lab_board == 0x2782,
               "a pair's round facts and lab board round-trip");
        auto bad                       = sp::round_facts_json(*q9.round_facts);
        bad["drafters"]["ngram"]       = bad["drafters"]["mtp"];
        bool refused                   = false;
        try {
            (void)sp::parse_round_facts(bad);
        } catch (const std::runtime_error&) { refused = true; }
        expect(refused, "round facts of no drafting mode are refused");
    }
    // An MoE pass's experts: k/E for one row, towards k/E + 0.30 (1 - k/E) for many.
    {
        sp::PassFacts moe{.bytes = 1, .params = 1, .expert_bytes = 1, .expert_params = 1,
                          .experts = 256, .experts_used = 8};
        expect(std::abs(sp::moe_read_fraction(moe, 1) - 8.0 / 256) < 1e-12 &&
                   std::abs(sp::moe_read_fraction(moe, 100000) - (8.0 / 256 + 0.30 * (1 - 8.0 / 256))) <
                       1e-9,
               "routed experts read: k/E a row, 0.30 of independent routing for many");
        expect(sp::moe_read_fraction(sp::PassFacts{.bytes = 1, .params = 1}, 8) == 1.0,
               "a dense pass reads all its weights");
    }
    // The cards' rates (PCI-ID table) and kappa = BW / (eta x int8 peak).
    {
        const auto rtx4070ti = sp::card_rates_of_board(0x2782);
        const auto gv100     = sp::card_rates_of_board(0x1DBA);
        const auto rtx3060   = sp::card_rates_of_board(0x2503);
        expect(rtx4070ti.generation == "ada" && near(rtx4070ti.kappa(), 0.00255941, 1e-5),
               "RTX 4070 Ti: kappa " + std::to_string(rtx4070ti.kappa()));
        expect(gv100.generation == "volta" && near(gv100.kappa(), 0.0553426, 1e-5),
               "Quadro GV100 (dp4a's rate): kappa " + std::to_string(gv100.kappa()));
        expect(near(rtx3060.kappa(), 0.00849246, 1e-5),
               "RTX 3060: kappa " + std::to_string(rtx3060.kappa()));
        const auto from_key = sp::card_rates(card("ga106:28sm", "ga106", "sm86", 86, 28, 1777, 360));
        expect(near(from_key.kappa(), rtx3060.kappa(), 1e-9), "a GpuKey's rates as the board's");
        expect(!sp::card_rates(80, 108, 1410000, 1555).known() &&
                   !sp::card_rates_of_board(0x1234).known() && !sp::card_rates(89, 0, 0, 0).known(),
               "no estimate for a generation without eta (A100), an unknown board, or no rates");
    }
    // R(k) golden numbers (fit_eta.py on the same facts).
    {
        struct Golden {
            const char* repo;
            const char* gpu;
            const char* config;
            double r4070ti, r5070ti, rgv100;
        };
        const Golden golden[] = {
            {kQ9, "ad104:60sm", "mtp:3", 1.610161, 1.703991, 2.010646},
            {kQ9, "ad104:60sm", "dflash:7", 1.754403, 2.111242, 3.277462},
            {"unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "gb203:70sm", "dflash:5", 1.898877, 2.167194, 3.054983},
            {"unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "gb203:70sm", "mtp:2", 1.567344, 1.621905, 1.802434},
            {"ggml-org/gpt-oss-20b-GGUF", "gb203:70sm", "dflash:3", 2.247096, 2.478775, 3.246199},
        };
        const double k4070 = sp::card_rates_of_board(0x2782).kappa();
        const double k5070 = sp::card_rates_of_board(0x2C05).kappa();
        const double kgv   = sp::card_rates_of_board(0x1DBA).kappa();
        for (const auto& g : golden) {
            const auto& p     = pair_of(g.repo, g.gpu);
            const auto config = sp::SpecConfig::parse(g.config);
            const auto a = sp::round_cost(*p.round_facts, config, k4070);
            const auto b = sp::round_cost(*p.round_facts, config, k5070);
            const auto c = sp::round_cost(*p.round_facts, config, kgv);
            expect(a && b && c && near(*a, g.r4070ti, 1e-5) && near(*b, g.r5070ti, 1e-5) &&
                       near(*c, g.rgv100, 1e-5),
                   std::string("R of ") + g.repo + " " + g.config);
        }
        expect(!sp::round_cost(*pair_of(kQ9, "ad104:60sm").round_facts, sp::SpecConfig{}, k4070),
               "plain decode has no round");
    }
    // A card nobody measured the pair on: the facts price its rounds, the lab card's measurement
    // raising them where it found rounds dearer than their bytes, and say so.
    {
        const auto& gemma = pair_of("unsloth/gemma-4-12b-it-GGUF@fc034cfff751157913579611efad8462ac1be606 Q4_K_M",
                                    "ad104:60sm");
        const auto rtx4060ti = card("ad106:34sm", "ad106", "sm89", 89, 34, 2535, 288);
        const auto est       = sp::round_cost_estimate(gemma, rtx4060ti);
        expect(est.known && est.note.find("conservative") != std::string::npos &&
                   est.cost.contains("mtp:3"),
               "Gemma 4's verify costs more than its bytes on the lab card: raised, said: " + est.note);
        const auto q9         = pair_of(kQ9, "ad104:60sm");
        const auto deep       = sp::round_cost_estimate(q9, rtx4060ti);
        expect(deep.known && !deep.cost.contains("dflash:11") && deep.cost.contains("dflash:7"),
               "deeper than kPortableDepth is not estimated away from the lab card");
        auto no_facts        = q9;
        no_facts.round_facts.reset();
        expect(!sp::round_cost_estimate(no_facts, rtx4060ti).known &&
                   !sp::round_cost_estimate(q9, sp::GpuKey{{"*"}, 0, false}).known,
               "no facts, or no rates: no estimate (the measured ratios stand)");
    }
    // Held out: the plan's speedup on a card with its own measurement left out, against that
    // measurement (the same target, its drafters as a start carries them).
    {
        const auto& registry = sp::Registry::builtin();
        struct Held {
            const char* gpu;
            sp::GpuKey key;
        };
        const Held held[] = {
            {"ga106:28sm", card("ga106:28sm", "ga106", "sm86", 86, 28, 1777, 360)},
            {"ad102:128sm", card("ad102:128sm", "ad102", "sm89", 89, 128, 2520, 1008)},
            {"gv100:80sm", card("gv100:80sm", "gv100", "sm70", 70, 80, 1627, 870.4)},
        };
        const auto& own_pair = pair_of(kQ9, "ad104:60sm");
        const std::vector<std::string> kinds = {"mtp", "dflash"};
        for (const auto& h : held) {
            std::vector<sp::MeasuredPair> without;
            for (const auto& p : registry.pairs()) {
                if (p.tune.key.gpu != h.gpu) { without.push_back(p); }
            }
            const std::string generation = h.key.chain[2];
            const auto measured  = sp::pair_choice(registry.pairs(), own_pair.target, "qwen3_5", kinds,
                                                   4.0, generation, {}, own_pair.drafters, {}, &h.key);
            const auto predicted = sp::pair_choice(without, own_pair.target, "qwen3_5", kinds, 4.0,
                                                   generation, {}, own_pair.drafters, {}, &h.key);
            if (!measured || !predicted) {
                expect(false, std::string("no choice for ") + h.gpu);
                continue;
            }
            const double m = measured->profile.overall.speedup();
            const double p = predicted->profile.overall.speedup();
            std::printf("held out %-12s measured %s x%.3f  predicted %s x%.3f  (%+.1f %%)\n  %s\n",
                        h.gpu, measured->profile.overall.config.label().c_str(), m,
                        predicted->profile.overall.config.label().c_str(), p, 100 * (p / m - 1),
                        predicted->source.c_str());
            expect(predicted->source.find("rounds priced from the weights' bytes") != std::string::npos,
                   std::string("the held-out card's rounds are priced by the facts: ") + h.gpu);
            expect(near(p, m, 0.10), std::string("held-out speedup within 10 %: ") + h.gpu);
        }
    }
    if (failures == 0) { std::cout << "round cost: all passed\n"; }
    return failures == 0 ? 0 : 1;
}
