#include "evaluation.h"
#include "kld.h"
#include "support/temp_dir.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

int check_partition(std::size_t tokens, std::uint32_t context, std::uint32_t stride) {
    const auto windows   = ninferno::perplexity::plan_windows(tokens, context, stride);
    int failures         = 0;
    std::size_t frontier = 1;
    for (const auto& window : windows) {
        failures += require(window.input_begin < window.input_end, "window input is nonempty");
        failures +=
            require(window.input_end - window.input_begin <= context, "window respects context");
        failures += require(window.target_begin == frontier, "target ranges are contiguous");
        failures +=
            require(window.target_end <= window.input_end, "targets are inside input window");
        failures += require(window.first_target == window.target_begin - window.input_begin,
                            "local target matches global range");
        frontier = window.target_end;
    }
    failures += require(frontier == tokens, "targets cover every token after the first");
    return failures;
}

// KL(P || Q) over listed tokens and the unlisted rest, and the reference file's round trip.
int check_kld() {
    namespace pp     = ninferno::perplexity;
    int failures     = 0;
    const float none = -std::numeric_limits<float>::infinity();
    const auto near  = [](double a, double b) { return std::abs(a - b) < 1e-6; };
    // The same distribution: 0.
    const std::vector<float> p{std::log(0.5F), std::log(0.25F), none};
    failures += require(near(pp::position_kld(p, p), 0.0), "KLD of a distribution with itself");
    // P = (0.5, 0.25, rest 0.25), Q = (0.25, 0.25, rest 0.5): 0.5 ln 2 + 0.25 ln 0.5.
    const std::vector<float> q{std::log(0.25F), std::log(0.25F), none};
    failures += require(near(pp::position_kld(p, q), 0.5 * std::log(2.0) + 0.25 * std::log(0.5)),
                        "KLD over listed tokens and the rest");
    // Everything listed in P: no rest term even where Q leaves mass out.
    const std::vector<float> full{std::log(0.5F), std::log(0.5F)};
    const std::vector<float> partial{std::log(0.5F), std::log(0.25F)};
    failures += require(near(pp::position_kld(full, partial), 0.5 * std::log(2.0)),
                        "KLD without unlisted mass");
    // A peaked Q whose listed mass rounds to 1: its rest is floored at FP32 noise, not 1e-30
    // (P's 1 % rest against 1e-5 costs about 0.06 nats, against 1e-30 it would be 0.6).
    const std::vector<float> wide{std::log(0.99F), none};
    const std::vector<float> peaked{0.0F, none};
    const double rest = pp::position_kld(wide, peaked);
    failures += require(rest > 0.05 && rest < 0.2, "KLD rest against a peaked Q");

    pp::KldAggregate a, b;
    a.add(0.1, true, -1.0, -1.5);
    a.add(0.3, false, -2.0, -2.0);
    b.add(0.2, true, -1.0, -1.0);
    a.add(b);
    failures += require(a.positions() == 3, "KLD aggregate counts positions");
    failures += require(near(a.mean(), 0.2), "KLD aggregate mean");
    failures += require(near(a.quantile(0.5), 0.2) && near(a.quantile(1.0), 0.3),
                        "KLD aggregate quantiles");
    failures += require(near(a.top1_agreement(), 2.0 / 3.0), "top-1 agreement");
    failures += require(near(a.ppl_ratio(), std::exp(0.5 / 3.0)), "PPL ratio");

    const std::vector<ninferno::TokenId> tokens{5, 9, 2};
    failures +=
        require(pp::token_hash(tokens) != pp::token_hash(std::vector<ninferno::TokenId>{5, 9}),
                "token hash sees every token");
    pp::KldReference written;
    written.top_k             = 2;
    written.context           = 4096;
    written.stride            = 2048;
    written.model             = "reference";
    written.prefill_signature = "bf16";
    written.streams.push_back({"wiki",
                               pp::token_hash(tokens),
                               2,
                               {9, 1, 2, -1},
                               {-0.5F, -1.5F, -0.1F, none},
                               {-0.5F, -0.1F}});
    const TempDir scratch("ninferno_kld_reference_test");
    const auto path = scratch.path() / "reference.bin";
    pp::write_kld_reference(path, written);
    const pp::KldReference read = pp::read_kld_reference(path);
    failures += require(read.top_k == 2 && read.context == 4096 && read.stride == 2048 &&
                            read.model == "reference" && read.prefill_signature == "bf16" &&
                            read.streams.size() == 1 && read.streams[0].id == "wiki" &&
                            read.streams[0].token_hash == written.streams[0].token_hash &&
                            read.streams[0].ids == written.streams[0].ids &&
                            read.streams[0].target_logprobs == written.streams[0].target_logprobs &&
                            std::isinf(read.streams[0].logprobs[3]),
                        "KLD reference round trip");
    // A listed slot with a positive log-probability, or an empty one with a finite value, is
    // rejected on read.
    for (const auto& [ids, logprobs] :
         std::vector<std::pair<std::vector<ninferno::TokenId>, std::vector<float>>>{
             {{9, 1, 2, -1}, {-0.5F, 0.5F, -0.1F, none}},
             {{9, 1, 2, -1}, {-0.5F, -1.5F, -0.1F, -3.0F}},
             {{9, 1, -7, -1}, {-0.5F, -1.5F, -0.1F, none}}}) {
        pp::KldReference bad    = written;
        bad.streams[0].ids      = ids;
        bad.streams[0].logprobs = logprobs;
        pp::write_kld_reference(path, bad);
        bool rejected = false;
        try {
            (void)pp::read_kld_reference(path);
        } catch (const std::runtime_error&) { rejected = true; }
        failures += require(rejected, "KLD reference with an invalid slot is rejected");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_partition(2, 4096, 2048);
    failures += check_partition(4096, 4096, 2048);
    failures += check_partition(4097, 4096, 2048);
    failures += check_partition(12001, 4096, 2048);

    const std::array<ninferno::perplexity::WindowPlan, 3> expected{{
        {.input_begin = 0, .input_end = 6, .target_begin = 1, .target_end = 6, .first_target = 1},
        {.input_begin = 2, .input_end = 8, .target_begin = 6, .target_end = 8, .first_target = 4},
        {.input_begin = 4, .input_end = 10, .target_begin = 8, .target_end = 10, .first_target = 4},
    }};
    const auto planned = ninferno::perplexity::plan_windows(10, 6, 2);
    failures += require(planned.size() == expected.size(), "window count matches the protocol");
    for (std::size_t index = 0; index < std::min(planned.size(), expected.size()); ++index) {
        const auto& actual = planned[index];
        const auto& wanted = expected[index];
        failures += require(actual.input_begin == wanted.input_begin &&
                                actual.input_end == wanted.input_end &&
                                actual.target_begin == wanted.target_begin &&
                                actual.target_end == wanted.target_end &&
                                actual.first_target == wanted.first_target,
                            "window boundaries match the protocol");
    }

    const std::vector<float> first{-1.0F, -2.0F};
    const std::vector<float> second{-3.0F};
    ninferno::perplexity::ScoreAggregate a;
    ninferno::perplexity::ScoreAggregate b;
    a.add(first);
    b.add(second);
    a.add(b);
    failures += require(a.scored_tokens == 3, "aggregate counts target tokens");
    failures += require(std::abs(a.total_nll - 6.0) < 1e-12, "aggregate accumulates FP64 NLL");
    failures += require(std::abs(a.mean_nll() - 2.0) < 1e-12, "aggregate computes mean NLL");
    failures += require(std::abs(a.ppl() - std::exp(2.0)) < 1e-12, "aggregate computes perplexity");

    failures += check_kld();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_evaluation\n";
    return failures == 0 ? 0 : 1;
}
