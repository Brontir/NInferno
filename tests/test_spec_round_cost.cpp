// A speculation round's cost from facts (spec/round_cost.h): the formula on the registry's own
// pairs (golden numbers from choice-notes/lab/round_cost/fit_eta.py), the cards' rates from the
// PCI-ID table, and a pair-removal consistency check: the plan's speedup for a card whose own pair
// is left out, against that pair's measurement. It is no hold-out of the fit (kSmallTEta rests on
// these cards' rounds); the hold-out evidence is fit_eta.py's cross-fit (DECISIONS 53). No GPU.

#include "artifact/schema.h"
#include "spec/manifest.h"
#include "spec/pairs.h"
#include "spec/registry.h"
#include "spec/round_cost.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
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
        bad["drafters"]["eagle9"]      = bad["drafters"]["mtp"];
        expect(sp::parse_round_facts(bad).drafters.size() == q9.round_facts->drafters.size(),
               "a drafter of a mode this build does not know is left out, the facts kept");
        auto no_used                              = sp::round_facts_json(*q9.round_facts);
        no_used["target"]["expert_bytes"]          = 10.0;
        no_used["target"]["expert_params"]         = 10.0;
        no_used["target"]["experts"]               = 8;
        no_used["target"]["experts_used"]          = 0;
        bool refused                              = false;
        try {
            (void)sp::parse_round_facts(no_used);
        } catch (const std::runtime_error&) { refused = true; }
        expect(refused, "experts with none used are refused");
        expect(q9.round_facts->drafters.at("dflash").head.has_value() &&
                   q9.round_facts->drafters.at("mtp").head.has_value() &&
                   q9.round_facts->drafters.at("mtp").head->linears ==
                       q9.round_facts->drafters.at("dflash").head->linears,
               "the target's head apart: DFlash's over the drafted rows, MTP's per draft");
        // The linears: the target's matrices hold nearly all its parameters (norms, convolutions
        // and gates left), one output head among them, the one DFlash runs over the drafted rows.
        const auto& facts = *q9.round_facts;
        double held       = 0;
        int heads         = 0;
        for (const auto& g : facts.target.linears) {
            held += double(g.n) * double(g.k) * g.count;
            heads += g.head ? 1 : 0;
        }
        const double all = facts.target.params + facts.target.expert_params;
        expect(held <= all && held >= 0.97 * all && heads == 1,
               "the target's linears hold its parameters, one head among them");
        std::vector<sp::LinearGroup> head;
        for (const auto& g : facts.target.linears) {
            if (g.head) { head.push_back(g); }
        }
        expect(facts.drafters.at("dflash").head->linears == head &&
                   !facts.drafters.at("mtp").pass.linears.empty(),
               "DFlash's head is the target's; the MTP pass lists its linears");
        auto zero                             = sp::round_facts_json(facts);
        zero["target"]["linears"][0]["count"] = 0;
        bool refused_linear                   = false;
        try {
            (void)sp::parse_round_facts(zero);
        } catch (const std::runtime_error&) { refused_linear = true; }
        expect(refused_linear, "a linear counted zero times is refused");
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
        expect(rtx4070ti.generation == "ada" && near(rtx4070ti.kappa(), 0.0106181, 1e-5),
               "RTX 4070 Ti: kappa " + std::to_string(rtx4070ti.kappa()));
        // A generation with one lab card has no eta (a constant rests on two cards): no estimate.
        expect(gv100.generation == "volta" && near(gv100.int8_ops_per_s, 512.0 * 80 * 1627e6, 1e-9) &&
                   !gv100.known() && !rtx3060.known() && !sp::card_rates_of_board(0x1E04).known(),
               "Volta (dp4a's rate), Ampere, Turing: one lab card each, no eta yet");
        // Blackwell's two lab cards (RTX 5070 Ti, RTX 5090: DECISIONS 56).
        const auto rtx5070ti = sp::card_rates_of_board(0x2C05);
        const auto rtx5090   = sp::card_rates_of_board(0x2B85);
        expect(rtx5070ti.generation == "gb20x" && rtx5070ti.known() && rtx5070ti.eta_cards == 2 &&
                   rtx5090.known() && near(rtx5090.kappa(), 1792e9 / (0.165 * 2048.0 * 170 * 2407e6), 1e-9),
               "Blackwell: eta from two lab cards, kappa " + std::to_string(rtx5090.kappa()));
        const auto from_key = sp::card_rates(card("ad104:60sm", "ad104", "sm89", 89, 60, 2610, 504));
        expect(near(from_key.kappa(), rtx4070ti.kappa(), 1e-9), "a GpuKey's rates as the board's");
        expect(!sp::card_rates(80, 108, 1410000, 1555).known() &&
                   !sp::card_rates_of_board(0x1234).known() && !sp::card_rates(89, 0, 0, 0).known(),
               "no estimate for a generation without eta (A100), an unknown board, or no rates");
    }
    // R(k) golden numbers (fit_eta.py on the same facts): at the RTX 4070 Ti's kappa, at none
    // (bytes alone) and at 0.05.
    {
        struct Golden {
            const char* repo;
            const char* gpu;
            const char* config;
            double r4070ti, r_bytes, r_k005;
        };
        const Golden golden[] = {
            {kQ9, "ad104:60sm", "mtp:3", 1.679516, 1.587408, 1.975274},
            {kQ9, "ad104:60sm", "dflash:7", 1.752298, 1.431237, 2.783223},
            {"unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "gb203:70sm", "dflash:5", 1.873191, 1.660462, 2.564459},
            {"unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "gb203:70sm", "mtp:2", 1.607643, 1.554161, 1.781435},
            {"ggml-org/gpt-oss-20b-GGUF", "gb203:70sm", "dflash:3", 2.030166, 1.905428, 2.435942},
        };
        const double k4070 = sp::card_rates_of_board(0x2782).kappa();
        for (const auto& g : golden) {
            const auto& p     = pair_of(g.repo, g.gpu);
            const auto config = sp::SpecConfig::parse(g.config);
            const auto a = sp::round_cost(*p.round_facts, config, k4070);
            const auto b = sp::round_cost(*p.round_facts, config, 0.0);
            const auto c = sp::round_cost(*p.round_facts, config, 0.05);
            expect(a && b && c && near(*a, g.r4070ti, 1e-5) && near(*b, g.r_bytes, 1e-5) &&
                       near(*c, g.r_k005, 1e-5),
                   std::string("R of ") + g.repo + " " + g.config);
        }
        expect(!sp::round_cost(*pair_of(kQ9, "ad104:60sm").round_facts, sp::SpecConfig{}, k4070),
               "plain decode has no round");
    }
    // The draft head over the rows an artifact's proposal head carries (#205: MTP reads its own
    // 57,344-row prefix of the 147,456 rows it read before, Qwen3.5-9B Q4_K_M, Q6_K head). The
    // rounds/s gain R(147,456) / R(57,344) - 1 against hard50's on the RTX 4060 Ti (l4060):
    // +9.4 / +12.9 / +16.5 / +19.6 % at mtp:2..5.
    {
        const auto& facts = *pair_of(kQ9, "ad104:60sm").round_facts;
        const double k4060 = sp::card_rates_of_board(0x2805).kappa();
        const sp::ProposalRows before{{{"mtp", 147456}}}, after{{{"mtp", 57344}}};
        const double measured[] = {0.094, 0.129, 0.165, 0.196};
        for (int k = 2; k <= 5; ++k) {
            const auto config = sp::SpecConfig::parse("mtp:" + std::to_string(k));
            const auto r0     = sp::round_cost(facts, config, k4060, before);
            const auto r1     = sp::round_cost(facts, config, k4060, after);
            expect(r0 && r1 && std::abs(*r0 / *r1 - 1 - measured[k - 2]) < 0.007,
                   "the MTP head's rows: rounds/s gain within 0.7 pp of hard50's at mtp:" +
                       std::to_string(k));
        }
        const auto config = sp::SpecConfig::parse("mtp:3");
        expect(sp::round_cost(facts, config, k4060) ==
                       sp::round_cost(facts, config, k4060, sp::ProposalRows{{{"dflash", 1024}}}) &&
                   sp::round_cost(facts, config, k4060) ==
                       sp::round_cost(facts, config, k4060, sp::ProposalRows{{{"mtp", 1u << 30}}}),
               "another mode's rows, or rows past the head's, leave the full head");
        const auto dflash = sp::SpecConfig::parse("dflash:7");
        expect(*sp::round_cost(facts, dflash, k4060, sp::ProposalRows{{{"dflash", 147456}}}) <
                   *sp::round_cost(facts, dflash, k4060),
               "DFlash's head over the k drafts reads the proposal rows too");
        // A checkpoint's head (BF16 here): its proposal rows are stored Q4_G64_FP16 (4.25 bits a
        // value), a GGUF's in the head's own format (its bytes' share, above).
        {
            sp::RoundFacts f;
            f.target = sp::PassFacts{.bytes = 1000, .params = 1000};
            sp::DraftFacts d;
            d.pass = sp::PassFacts{.bytes = 100, .params = 100};
            d.head = sp::PassFacts{.bytes = 2.0 * 1000 * 64, .params = 1000.0 * 64};
            d.head->linears = {{.format = "bf16", .n = 1000, .k = 64, .count = 1, .head = true}};
            f.drafters["mtp"] = d;
            const auto c = sp::SpecConfig::parse("mtp:2");
            const auto r = sp::round_cost(f, c, 0.0, sp::ProposalRows{{{"mtp", 100}}});
            expect(r && near(*r, (1000 + 2 * (100 + 100 * 64 * 4.25 / 8)) / 1000, 1e-12),
                   "a checkpoint's proposal rows at Q4_G64_FP16's 4.25 bits");
        }
        // Each kernel a round's graphs launch, and a plain step's, at the card's launch time:
        // R = (C_round + L n_round) / (C_step + L n_step), L = t0 x eta_dram x BW.
        {
            sp::RoundFacts f;
            f.target          = sp::PassFacts{.bytes = 1000, .params = 1000};
            f.drafters["mtp"] = sp::DraftFacts{.pass = sp::PassFacts{.bytes = 100, .params = 100}};
            const auto c      = sp::SpecConfig::parse("mtp:2");
            const auto r = sp::round_cost(f, c, 0.0, {}, 2.0, sp::RoundLaunches{.round = 50, .step = 30});
            expect(r && near(*r, (1000 + 200 + 2.0 * 50) / (1000 + 2.0 * 30), 1e-12),
                   "a round's kernel launches and a plain step's");
            expect(sp::round_cost(f, c, 0.0, {}, 2.0, sp::RoundLaunches{.round = 50}) ==
                       sp::round_cost(f, c, 0.0),
                   "launches not counted for the plain step: none added");
            const auto ada = sp::card_rates_of_board(0x2782);
            expect(near(ada.launch_bytes(), 1.5e-6 * 0.946 * ada.dram_bytes_per_s, 1e-12) &&
                       sp::card_rates_of_board(0x2503).launch_bytes() == 0 &&
                       sp::card_rates_of_board(0x2C05).launch_bytes() == 0,
                   "Ada's launch time; Ampere and gb20x none (not fitted)");
        }
        // A tune's graph kernels round-trip, and an estimate prices them: more kernels a round,
        // a dearer round.
        {
            auto counted = pair_of(kQ9, "ad104:60sm");
            for (auto& m : counted.tune.measurements) {
                m.graph_rounds  = std::max<std::int64_t>(m.config.enabled() ? m.rounds : m.tokens, 1);
                m.graph_kernels = (m.config.enabled() ? 500 : 311) * m.graph_rounds;
            }
            const auto again = sp::parse_pair(sp::pair_json(counted));
            expect(again.tune.measurements.front().graph_kernels ==
                           counted.tune.measurements.front().graph_kernels &&
                       again.tune.measurements.front().graph_rounds ==
                           counted.tune.measurements.front().graph_rounds,
                   "graph kernels and their launches round-trip");
            const auto rtx4060 = card("ad106:34sm", "ad106", "sm89", 89, 34, 2535, 288);
            const auto with    = sp::round_cost_estimate(counted, rtx4060);
            const auto without = sp::round_cost_estimate(pair_of(kQ9, "ad104:60sm"), rtx4060);
            expect(with.known && without.known && with.cost.at("mtp:3") != without.cost.at("mtp:3"),
                   "counted launches enter the estimate");
            // Each section counts once, as in its measured cost: a section of other kernel
            // counts that ran ten times the rounds moves nothing.
            const std::string first = counted.tune.measurements.front().family;
            auto mixed              = counted;
            for (auto& m : mixed.tune.measurements) {
                if (m.family != first) { m.graph_kernels += 200 * m.graph_rounds; }
            }
            auto longer = mixed;
            for (auto& m : longer.tune.measurements) {
                if (m.family == first) {
                    m.graph_kernels *= 10;
                    m.graph_rounds *= 10;
                }
            }
            const auto a = sp::round_cost_estimate(mixed, rtx4060);
            const auto b = sp::round_cost_estimate(longer, rtx4060);
            expect(std::any_of(mixed.tune.measurements.begin(), mixed.tune.measurements.end(),
                               [&](const auto& m) { return m.family != first; }) &&
                       a.known && b.known && near(a.cost.at("mtp:3"), b.cost.at("mtp:3"), 1e-12),
                   "launch counts average over the sections, not their rounds");
        }
        // The lab's rows ride in its tune (spec-tune records them): they round-trip; zero rows are
        // refused.
        auto pair                            = pair_of(kQ9, "ad104:60sm");
        pair.tune.proposal_rows              = after;
        const auto again                     = sp::parse_pair(sp::pair_json(pair));
        expect(again.tune.proposal_rows == after, "a tune's proposal rows round-trip");
        auto zero                            = sp::pair_json(pair);
        zero["tune"]["proposal_rows"]["mtp"] = 0;
        bool refused                         = false;
        try {
            (void)sp::parse_pair(zero);
        } catch (const std::runtime_error&) { refused = true; }
        expect(refused, "zero proposal rows are refused");
        // The lab card's own rounds at its rows, the card's at its artifact's: a lab artifact of
        // the full head and a card's of MTP's prefix price the card's MTP rounds cheaper.
        const auto rtx4060ti = card("ad106:34sm", "ad106", "sm89", 89, 34, 2535, 288);
        const auto lab       = sp::round_cost_estimate(pair_of(kQ9, "ad104:60sm"), rtx4060ti);
        const auto mine      = sp::round_cost_estimate(pair_of(kQ9, "ad104:60sm"), rtx4060ti, &after);
        expect(lab.known && mine.known && mine.cost.at("mtp:3") < lab.cost.at("mtp:3") &&
                   mine.cost.at("dflash:7") == lab.cost.at("dflash:7"),
               "the card's artifact rows price its MTP rounds; DFlash's (not named) stay");
        // A tune's "mtp:3" drafted with the full head whatever rows its artifact had: its own
        // rounds priced so (only a "+head" label drafted through the rows).
        const auto plain_labels = sp::round_cost_estimate(pair, rtx4060ti);
        expect(plain_labels.known && plain_labels.cost.at("mtp:3") == lab.cost.at("mtp:3"),
               "a configuration without +head was measured with the full head");
        // An artifact's rows: its proposal head where a Use binds it to the drafter, MTP its
        // own prefix (drafter_rows), DFlash all of them; DFlash2 none (the whole domain).
        namespace art = ninferno::artifact;
        art::Directory dir;
        for (const char* c : {"text", "mtp", "dflash", "dflash2"}) { dir.components[c] = {}; }
        dir.components["text"].proposal =
            art::Proposal{.indexed = true, .rows = 147456, .drafter_rows = {{"mtp", 57344}}};
        for (const char* c : {"mtp", "dflash", "dflash2"}) {
            dir.uses[{"proposal/head", std::string(c) + "/final_hidden"}] = {};
        }
        expect(sp::artifact_proposal_rows(dir) == sp::ProposalRows{{{"mtp", 57344}, {"dflash", 147456}}},
               "an artifact's proposal rows: MTP its prefix, DFlash all, DFlash2 none");
        // A head without row views (row-scaled FP8 here): MTP reads all its rows.
        art::TensorObject fp8;
        fp8.id     = "head";
        fp8.layout = "row_scale_v1";
        dir.objects.push_back(fp8);
        dir.bindings["proposal/head"].parts.push_back({.object = {.index = 0}, .begin = 0, .end = 1});
        expect(sp::artifact_proposal_rows(dir).of("mtp") == 147456,
               "a head with no row views: MTP reads all its rows");
        dir.bindings.clear();
        dir.objects.clear();
        dir.uses.erase({"proposal/head", "mtp/final_hidden"});
        expect(sp::artifact_proposal_rows(dir).of("mtp") == 0, "a head not bound to MTP: its full head");
        dir.components["text"].proposal.reset();
        expect(sp::artifact_proposal_rows(dir).empty(), "no proposal head: the full head");
    }
    // A card nobody measured the pair on: the facts price its rounds, the lab card's measurement
    // raising them where it found rounds dearer than their bytes, and say so.
    {
        const auto& gemma = pair_of("unsloth/gemma-4-12b-it-GGUF@fc034cfff751157913579611efad8462ac1be606 Q4_K_M",
                                    "ad104:60sm");
        const auto rtx4060ti = card("ad106:34sm", "ad106", "sm89", 89, 34, 2535, 288);
        const auto est       = sp::round_cost_estimate(gemma, rtx4060ti);
        expect(est.known && est.cost.contains("mtp:3"), "Gemma 4's rounds priced for another card");
        // A lab card that measured its rounds dearer than their bytes raises the estimate and says
        // so (the fresh 4070 Ti pair measures them at their bytes: a copy made 50 % dearer).
        auto dearer = gemma;
        for (auto& m : dearer.tune.measurements) {
            if (m.config.enabled()) { m.seconds *= 1.5; }
        }
        const auto raised = sp::round_cost_estimate(dearer, rtx4060ti);
        expect(raised.known && raised.note.find("conservative") != std::string::npos &&
                   raised.cost.at("mtp:3") > est.cost.at("mtp:3"),
               "rounds dearer than their bytes on the lab card: raised, said: " + raised.note);
        const auto q9         = pair_of(kQ9, "ad104:60sm");
        const auto deep       = sp::round_cost_estimate(q9, rtx4060ti);
        expect(deep.known && !deep.cost.contains("dflash:11") && deep.cost.contains("dflash:7"),
               "deeper than kPortableDepth is not estimated away from the lab card");
        auto no_facts        = q9;
        no_facts.round_facts.reset();
        expect(!sp::round_cost_estimate(no_facts, rtx4060ti).known &&
                   !sp::round_cost_estimate(q9, sp::GpuKey{{"*"}, 0, false}).known,
               "no facts, or no rates: no estimate (the measured ratios stand)");
        // Two lab cards of its generation (RTX 4090, RTX 4070 Ti): the nearer by rates prices it,
        // whichever class id sorts first (the 4060 Ti is the 4070 Ti's size, not the 4090's).
        const auto& gemma_4090 = pair_of(
            "unsloth/gemma-4-12b-it-GGUF@fc034cfff751157913579611efad8462ac1be606 Q4_K_M", "ad102:128sm");
        std::string both_note, near_note;
        (void)sp::profile_for_target({&gemma_4090, &gemma}, "sm89", {}, &rtx4060ti, &both_note);
        (void)sp::profile_for_target({&gemma}, "sm89", {}, &rtx4060ti, &near_note);
        expect(!both_note.empty() && both_note == near_note,
               "the nearest lab board's pair prices a card of its generation: " + both_note);
    }
    // The card that measured a pair takes its own measurement, its tune keyed by class id or chip.
    {
        const auto rtx4070ti = card("ad104:60sm", "ad104", "sm89", 89, 60, 2610, 504);
        auto chip_keyed      = pair_of(kQ9, "ad104:60sm");
        chip_keyed.tune.key.gpu = "ad104";
        for (const auto* p : {&pair_of(kQ9, "ad104:60sm"), static_cast<const sp::MeasuredPair*>(&chip_keyed)}) {
            std::string note;
            (void)sp::profile_for_target({p}, "sm89", {}, &rtx4070ti, &note);
            expect(note.empty(), "the measuring card's own record (" + p->tune.key.gpu + "), not facts");
        }
        // Another board of the lab card's class (RTX 4500 Ada: 432 GB/s against the 4070 Ti's 504):
        // not the lab card, priced by the facts.
        const auto rtx4500 = card("ad104:60sm", "ad104", "sm89", 89, 60, 2580, 432);
        std::string other_note;
        (void)sp::profile_for_target({&pair_of(kQ9, "ad104:60sm")}, "sm89", {}, &rtx4500,
                                     &other_note);
        expect(!other_note.empty(), "another board of the lab card's class: priced by the facts");
        auto arch_keyed          = chip_keyed;
        arch_keyed.tune.key.gpu  = "sm89";
        std::string note;
        (void)sp::profile_for_target({&arch_keyed}, "sm89", {}, &rtx4070ti, &note);
        expect(!note.empty(), "an architecture key is no card's own: priced by the facts");
        // A generation without eta (one lab card): another board keeps the lab-measured profile.
        const auto rtx3060_8gb = card("ga106:28sm", "ga106", "sm86", 86, 28, 1777, 240);
        std::string ampere_note;
        (void)sp::profile_for_target({&pair_of(kQ9, "ga106:28sm")}, "sm86", {}, &rtx3060_8gb,
                                     &ampere_note);
        expect(ampere_note.empty(), "Ampere has no eta yet: the lab-measured profile, no facts");
        // Turing: no eta and no lab pair of this target on Turing. The median R of every lab card
        // that measured it, said as a transfer; never the registry's first pair (the RTX 4070
        // Ti's own R, DECISIONS 57).
        const auto rtx2080ti = card("tu102:68sm", "tu102", "sm75", 75, 68, 1545, 616);
        std::vector<const sp::MeasuredPair*> q9;
        std::map<std::string, std::vector<double>> each;
        for (const auto& p : sp::Registry::builtin().pairs()) {
            if (std::string_view(p.target_repo) != kQ9) { continue; }
            q9.push_back(&p);
            for (const auto& [label, r] : sp::profile_round_costs(p.tune)) { each[label].push_back(r); }
        }
        std::string turing_note;
        const auto turing = sp::profile_for_target(q9, "sm75", {}, &rtx2080ti, &turing_note);
        const auto cost   = sp::profile_round_costs(turing);
        const auto ada_r  = sp::profile_round_costs(pair_of(kQ9, "ad104:60sm").tune);
        bool median = q9.size() >= 2 && cost.contains("mtp:2") && cost.contains("mtp:3");
        for (const auto& [label, r] : cost) {
            auto v = each.at(label);
            std::sort(v.begin(), v.end());
            const double m = v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
            median = median && near(r, m, 1e-9);
        }
        expect(median && !near(cost.at("mtp:2"), ada_r.at("mtp:2"), 0.01),
               "Turing takes the lab cards' median R, not the RTX 4070 Ti's");
        expect(turing_note == "transferred: median of " + std::to_string(q9.size()) +
                                  " lab cards of other generations (no facts for turing)",
               "the transfer is said: " + turing_note);
        auto reversed = q9;
        std::reverse(reversed.begin(), reversed.end());
        expect(sp::to_json(sp::profile_for_target(reversed, "sm75", {}, &rtx2080ti)).dump() ==
                   sp::to_json(turing).dump(),
               "the registry's order never moves Turing's profile");
    }
    // Two lab cards of the generation that agree: their deep depths and load table are theirs; a
    // card that measured nothing is priced by the facts, no deeper than kPortableDepth (Codex P2
    // on #161); a lab board keeps the measurement.
    {
        const auto& ada   = pair_of(kQ9, "ad104:60sm");
        auto twin         = ada; // the same measurement on a second Ada card: they agree
        twin.tune.key.gpu = "ad102:128sm";
        twin.lab_board    = 0x2684;
        const std::vector<const sp::MeasuredPair*> both = {&ada, &twin};
        const auto agreed    = sp::profile_for_target(both, "sm89");
        const auto rtx4060ti = card("ad106:34sm", "ad106", "sm89", 89, 34, 2535, 288);
        std::string note;
        const auto priced  = sp::profile_for_target(both, "sm89", {}, &rtx4060ti, &note);
        const auto shallow = [](const sp::Profile& p) {
            return p.overall.config.k <= sp::kPortableDepth &&
                   p.thinking.config.k <= sp::kPortableDepth &&
                   p.non_thinking.config.k <= sp::kPortableDepth;
        };
        expect(!shallow(agreed), "the agreeing lab cards choose a deep depth: " +
                                     agreed.thinking.config.label());
        expect(!note.empty() && shallow(priced),
               "an unmeasured Ada card: facts, no deeper than kPortableDepth: " +
                   priced.thinking.config.label());
        // What the plan reports as "round_cost" (spec::profile_round_costs of the resolved
        // profile) is, for a profile priced by the facts, the facts' R(k) itself: the user kit
        // holds a measured R(k) to it (tools/kit/validate.py).
        {
            std::string one_note;
            const auto one   = sp::profile_for_target({&ada}, "sm89", {}, &rtx4060ti, &one_note);
            const auto facts = sp::round_cost_estimate(ada, rtx4060ti);
            const auto costs = sp::profile_round_costs(one);
            bool same = !one_note.empty() && facts.known && costs.size() == facts.cost.size();
            for (const auto& [label, cost] : costs) {
                const auto it = facts.cost.find(label);
                same = same && it != facts.cost.end() && near(cost, it->second, 1e-9);
            }
            expect(same, "the priced profile's round costs are the facts' estimate");
        }
        const auto rtx4070ti = card("ad104:60sm", "ad104", "sm89", 89, 60, 2610, 504);
        std::string own_note;
        const auto own = sp::profile_for_target(both, "sm89", {}, &rtx4070ti, &own_note);
        expect(own_note == "rounds: the median of 2 lab cards of ada" &&
                   own.thinking.config == agreed.thinking.config &&
                   own.overall.config == agreed.overall.config,
               "a lab board takes its generation's agreeing cards' median rounds: " + own_note);
    }
    // One card's two tunes (spec-tune measures one drafter at a time), each with its own mode's
    // facts: pooled, both drafters are priced, whatever the records' order.
    {
        const auto& whole = pair_of(kQ9, "ad104:60sm");
        auto only = [&](sp::Mode mode, const char* name) {
            auto p = whole;
            std::erase_if(p.tune.measurements, [&](const sp::Measurement& m) {
                return m.config.enabled() && m.config.mode != mode;
            });
            std::erase_if(p.round_facts->drafters, [&](const auto& d) { return d.first != name; });
            return p;
        };
        const auto mtp    = only(sp::Mode::Mtp, "mtp");
        const auto dflash = only(sp::Mode::DFlash, "dflash");
        const auto rtx4060ti = card("ad106:34sm", "ad106", "sm89", 89, 34, 2535, 288);
        const std::vector<std::string> kinds = {"mtp", "dflash"};
        const auto choose = [&](std::vector<sp::MeasuredPair> records) {
            return sp::pair_choice(records, whole.target, "qwen3_5", kinds, 4.0, "sm89", {},
                                   whole.drafters, {}, &rtx4060ti);
        };
        const auto ab = choose({mtp, dflash});
        const auto ba = choose({dflash, mtp});
        const auto modes = [](const sp::Profile& p) {
            std::set<sp::Mode> out;
            for (const auto& m : p.measurements) {
                if (m.config.enabled()) { out.insert(m.config.mode); }
            }
            return out;
        };
        expect(ab && ba && modes(ab->profile) == std::set<sp::Mode>{sp::Mode::Mtp, sp::Mode::DFlash} &&
                   modes(ba->profile) == modes(ab->profile),
               "two tunes of one card, each with its mode's facts: both drafters priced");
        expect(ab && ba && ab->profile.overall.config == ba->profile.overall.config &&
                   ab->profile.thinking.config == ba->profile.thinking.config &&
                   std::abs(ab->profile.overall.speedup() - ba->profile.overall.speedup()) < 1e-9,
               "the pooled choice does not depend on the records' order");
        // Every order of the records gives the same plan. One card (the RTX 4070 Ti) has four tunes
        // that an order could tell apart: two MTP tunes whose mode's round facts differ, plain
        // decode measured at other speeds in each (sums of three or more terms), two DFlash tunes
        // (pair_group keeps one of them), one naming no lab board; two other cards.
        const auto tuned = [](sp::MeasuredPair p, double plain, double spec, std::uint32_t board) {
            for (auto& m : p.tune.measurements) { m.seconds *= m.config.enabled() ? spec : plain; }
            p.lab_board = board;
            return p;
        };
        auto mtp_b = tuned(mtp, 1.1, 1.05, 0x2782);
        mtp_b.round_facts->drafters.at("mtp").rows += 1;
        const std::vector<sp::MeasuredPair> records = {
            tuned(mtp, 1.0, 1.0, 0x2782), mtp_b, tuned(dflash, 1.2, 1.0, 0),
            tuned(dflash, 1.3, 1.03, 0x2782), pair_of(kQ9, "ad102:128sm"), pair_of(kQ9, "ga106:28sm")};
        const auto rtx4070ti = card("ad104:60sm", "ad104", "sm89", 89, 60, 2610, 504);
        const auto rtx2080ti = card("tu102:68sm", "tu102", "sm75", 75, 68, 1545, 616);
        const std::string repo(std::string_view(kQ9).substr(0, std::string_view(kQ9).find(' ')));
        const auto plan_of = [&](const std::vector<sp::MeasuredPair>& in) {
            std::string out;
            const auto add = [&](const std::optional<sp::PairChoice>& c) {
                out += c ? c->basis + "|" + c->source + "|" + sp::to_json(c->profile).dump() + "\n"
                         : std::string("none\n");
            };
            for (const auto* gpu : {&rtx4060ti, &rtx4070ti, &rtx2080ti}) {
                add(sp::pair_choice(in, whole.target, "qwen3_5", kinds, 4.0,
                                    gpu == &rtx2080ti ? "sm75" : "sm89", {}, whole.drafters, {}, gpu));
            }
            add(sp::pair_choice(in, "another-quantization", "qwen3_5", kinds, 4.0, "sm89", {},
                                whole.drafters, repo, &rtx4060ti));
            const auto group = sp::pair_group(in, "qwen3_5", "dflash", 4.0);
            for (const auto* p : group) { out += sp::pair_json(*p).dump() + ","; }
            const auto grouped = sp::profile_for_group(group, "dflash", nullptr);
            out += "|" + (grouped ? sp::to_json(*grouped).dump() : std::string("none")) + "\n";
            return out;
        };
        const auto reference = plan_of(records);
        std::vector<std::size_t> idx = {0, 1, 2, 3, 4, 5};
        bool same = !reference.empty();
        int orders = 0;
        do {
            std::vector<sp::MeasuredPair> shuffled;
            for (const auto i : idx) { shuffled.push_back(records[i]); }
            same = same && plan_of(shuffled) == reference;
            ++orders;
        } while (same && std::next_permutation(idx.begin(), idx.end()));
        expect(same && orders == 720, "all 720 orders of the records give the same plans");
        // Two boards of one class (an RTX 4500 Ada's tune beside the 4070 Ti's, both ad104:60sm) are
        // two cards (each finds its own record, not the facts) taking the median R of the two.
        const std::vector<sp::MeasuredPair> boards = {tuned(mtp, 1.0, 1.0, 0x2782),
                                                      tuned(mtp, 1.0, 1.3, 0x27B1)};
        const auto rtx4500 = card("ad104:60sm", "ad104", "sm89", 89, 60, 2580, 432);
        const auto two     = sp::pair_choice(boards, whole.target, "qwen3_5", kinds, 4.0, "sm89", {},
                                             whole.drafters, {}, &rtx4070ti);
        const auto other   = sp::pair_choice(boards, whole.target, "qwen3_5", kinds, 4.0, "sm89", {},
                                             whole.drafters, {}, &rtx4500);
        const auto r = [](const std::optional<sp::PairChoice>& c) {
            return c ? sp::profile_round_costs(c->profile) : std::map<std::string, double>{};
        };
        expect(two && other && two->source.find("(2 lab card(s))") != std::string::npos &&
                   !r(two).empty() && !r(other).empty() &&
                   std::abs(r(other).at("mtp:3") / r(two).at("mtp:3") - 1) < 1e-9,
               "two boards of one class are two cards, priced by their median: " +
                   (two ? two->source : std::string()));
        // So in a group: the two boards are two cards (enough for a group's choice); a record naming
        // no board is its class's card's.
        const std::vector<sp::MeasuredPair> group_boards = {tuned(dflash, 1.0, 1.0, 0x2782),
                                                            tuned(dflash, 1.0, 1.1, 0x27B1),
                                                            tuned(dflash, 1.0, 1.2, 0)};
        expect(sp::pair_group(group_boards, "qwen3_5", "dflash", 4.0).size() == 2,
               "two boards of one class are a group of two cards; an unnamed board's record is one of them");
        // Two targets on one card, one record naming its board and one naming none: one card, no
        // group (a group's choice rests on two cards).
        auto unnamed_other   = tuned(dflash, 1.0, 1.0, 0);
        unnamed_other.target = "other-weights";
        expect(sp::pair_group({tuned(dflash, 1.0, 1.0, 0x2782), unnamed_other}, "qwen3_5", "dflash", 4.0)
                   .empty(),
               "a record naming no board is its class's card in a group's count");
        const auto one = sp::pair_choice({tuned(mtp, 1.0, 1.0, 0), tuned(dflash, 1.0, 1.0, 0x2782)},
                                         whole.target, "qwen3_5", kinds, 4.0, "sm89", {}, whole.drafters,
                                         {}, &rtx4070ti);
        expect(one && one->source.find("(1 lab card(s))") != std::string::npos && one->basis == "measured",
               "a tune naming no board joins its class's card: " + (one ? one->source : std::string()));
    }
    // Pair removal: the plan's speedup on a card with its own pair left out (the facts price the
    // other cards' measurements for it), against that pair's measurement (the same target, its
    // drafters as a start carries them). A consistency check, not a hold-out of eta.
    {
        const auto& registry = sp::Registry::builtin();
        struct Removed {
            const char* gpu;
            sp::GpuKey key;
        };
        const Removed removed[] = {
            {"ad102:128sm", card("ad102:128sm", "ad102", "sm89", 89, 128, 2520, 1008)},
            {"ga106:28sm", card("ga106:28sm", "ga106", "sm86", 86, 28, 1777, 360)},
            {"gb202:170sm", card("gb202:170sm", "gb202", "sm120", 120, 170, 2407, 1792)},
            {"gb203:70sm", card("gb203:70sm", "gb203", "sm120", 120, 70, 2452, 896)},
        };
        const auto& own_pair = pair_of(kQ9, "ad104:60sm");
        const std::vector<std::string> kinds = {"mtp", "dflash"};
        for (const auto& h : removed) {
            std::vector<sp::MeasuredPair> without;
            for (const auto& p : registry.pairs()) {
                if (p.tune.key.gpu != h.gpu) { without.push_back(p); }
            }
            const std::string generation = h.key.chain[2];
            // The card's own measurement: its pairs alone (with every card's, two agreeing lab
            // cards of the generation would answer with the first card's profile).
            std::vector<sp::MeasuredPair> own_only;
            for (const auto& p : registry.pairs()) {
                if (p.tune.key.gpu == h.gpu) { own_only.push_back(p); }
            }
            const auto measured  = sp::pair_choice(own_only, own_pair.target, "qwen3_5", kinds, 4.0,
                                                   generation, {}, own_pair.drafters, {}, &h.key);
            const auto predicted = sp::pair_choice(without, own_pair.target, "qwen3_5", kinds, 4.0,
                                                   generation, {}, own_pair.drafters, {}, &h.key);
            if (!measured || !predicted) {
                expect(false, std::string("no choice for ") + h.gpu);
                continue;
            }
            const double m = measured->profile.overall.speedup();
            const double p = predicted->profile.overall.speedup();
            std::printf("pair removed %-12s measured %s x%.3f  predicted %s x%.3f  (%+.1f %%)\n  %s\n",
                        h.gpu, measured->profile.overall.config.label().c_str(), m,
                        predicted->profile.overall.config.label().c_str(), p, 100 * (p / m - 1),
                        predicted->source.c_str());
            // Ada and Blackwell have eta (two lab cards); Ampere not yet (one): it keeps the
            // previous transfer.
            const bool facts = h.key.compute_capability == 89 || h.key.compute_capability == 120;
            expect((predicted->source.find("rounds priced from the weights' bytes") !=
                    std::string::npos) == facts,
                   std::string("the card's rounds are priced by the facts only with eta: ") + h.gpu);
            // The RTX 5070 Ti's 9B rounds verifying 6-8 rows are cheaper than their facts
            // (-7..-12 %, DECISIONS 56): its plan's speedup stays low by up to 15 %, the pick right.
            const double within = h.key.compute_capability == 120 && h.key.sms == 70 ? 0.15 : 0.10;
            if (facts) {
                expect(near(p, m, within) && predicted->profile.overall.config ==
                                                 measured->profile.overall.config,
                       std::string("speedup without its pair within the bound, the same pick: ") +
                           h.gpu);
            }
        }
    }
    if (failures == 0) { std::cout << "round cost: all passed\n"; }
    return failures == 0 ? 0 : 1;
}
