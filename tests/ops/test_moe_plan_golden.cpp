// The generic MoE op's route decisions, frozen (#616 step 1): for each board of the selector
// goldens (a substituted DeviceClass, core/device_class.h) the plan select_plan gives for each MoE
// layer of the lab's models, under the speed, prefill and invariant profiles at T = 1..8, 16, 17,
// 32, 64, 128, 512 and 2048, and which authority of its cascade answered (moe_plan_query). Nothing
// runs: the weights are descriptors over an aligned stand-in address, as in the selector golden.
// The dump is compared with moe_plan_golden/<build set>/<board>.txt; a refactor of the cascade
// (Roadmap steps 2-4) must leave every file as it was. Regenerate with --update after a change
// that means to move routes.
//
// The build set decides what a board can pick (the same four sets as the selector goldens: sm70,
// sm75, base, full), so each keeps its own files. The decision code reads no compiled-kernel
// fact: only the substituted class, the build's features and the route tables (per class).
//
// The lab's MoE models: Qwen3.6-35B-A3B (E256 k8 H2048 F512), Gemma 4 26B-A4B (E128 k8 H2816
// F704), gpt-oss-20b (E32 k4 H2880 F2880) and, for the fourth shape, Nemotron 3.5 Lightning
// 30B-A3B (E128 k6 H2688 F1856, relu^2): Muse Glimmer 30B is a dense decoder.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"
#include "ops/moe/moe_internal.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
namespace moe     = ninferno::ops::detail::moe;
namespace qw      = ninferno::test::quantized_weight;
using ops::LinearPolicy;

// The selector goldens' boards and the ones whose routes/ tables carry MoE entries that those
// lack (this golden's own, so the selector goldens do not move): the RTX 3060 (ga106), the RTX
// 4070 (ad104, 46 SMs) and the RTX 4070 Ti (ad104-60sm).
constexpr generic::FactsBoard kExtraBoards[] = {
    {"rtx3060", 0x2503, 8, 6, 28, 2, 12, 1777, 7501, 192},
    {"rtx4070", 0x2786, 8, 9, 46, 36, 12, 2475, 10501, 192},
    {"rtx4070ti", 0x2782, 8, 9, 60, 48, 12, 2610, 10501, 192},
};

const std::vector<generic::FactsBoard>& boards() {
    static const std::vector<generic::FactsBoard> all = [] {
        std::vector<generic::FactsBoard> v(std::begin(generic::kFactsBoards),
                                           std::end(generic::kFactsBoards));
        v.insert(v.end(), std::begin(kExtraBoards), std::end(kExtraBoards));
        return v;
    }();
    return all;
}

constexpr const char* kBuildSet =
#if NINFERNO_BUILD_ARCH == 70
    "sm70";
#elif NINFERNO_BUILD_ARCH == 75
    "sm75";
#elif defined(NINFERNO_WITHOUT_NVFP4)
    "base";
#else
    "full";
#endif

struct Layer {
    const char* model;
    std::int32_t experts, top_k, hidden, inter;
    QType gate_up, down;
    bool relu2 = false;
    // The banks in ggml_frag_v1, as expert_bank_layout's Auto makes shipped artifacts (Q4_K,
    // IQ3_XXS, IQ2_S; Q4_0 from K = 1024): the weight descriptor says so, and nothing here reads
    // the payload.
    bool frag_gate_up = false, frag_down = false;
    // Columns of the gate/up bank past H (a padded GGUF: Nemotron-H, H 2688, K 2816).
    std::int32_t pad = 0;
};

// Each model in the formats its checkpoints ship in (GGUF K-quants and IQ, published int4,
// NVFP4, MXFP4), and the ones the route tables carry entries for.
const Layer kLayers[] = {
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_Q4_K, QType::GGML_Q6_K},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_Q3_K, QType::GGML_Q4_K},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_IQ2_S, QType::GGML_IQ4_XS},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_IQ2_S, QType::GGML_IQ3_XXS},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_IQ3_XXS, QType::GGML_IQ4_XS},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::U4_G128_ZP_FP16, QType::U4_G128_ZP_FP16},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::NVFP4, QType::NVFP4},
    {"gemma-4-26b-a4b", 128, 8, 2816, 704, QType::GGML_IQ4_NL, QType::GGML_IQ4_NL},
    {"gemma-4-26b-a4b", 128, 8, 2816, 704, QType::GGML_IQ3_S, QType::Q8_G32_FP16},
    {"gemma-4-26b-a4b", 128, 8, 2816, 704, QType::GGML_IQ2_S, QType::GGML_IQ4_NL},
    {"gemma-4-26b-a4b", 128, 8, 2816, 704, QType::GGML_Q4_0, QType::GGML_Q4_0},
    {"gemma-4-26b-a4b", 128, 8, 2816, 704, QType::NVFP4, QType::NVFP4},
    {"gpt-oss-20b", 32, 4, 2880, 2880, QType::MXFP4, QType::MXFP4},
    {"gpt-oss-20b", 32, 4, 2880, 2880, QType::GGML_Q4_0, QType::GGML_Q4_0},
    {"nemotron-3.5-30b-a3b", 128, 6, 2688, 1856, QType::NVFP4, QType::NVFP4, true},
    {"nemotron-3.5-30b-a3b", 128, 6, 2688, 1856, QType::GGML_Q5_1, QType::GGML_Q5_0, true},
    // The layers the route tables key (gb202 / gb203 / gb203-70sm: Qwen3-30B-A3B NVFP4;
    // gb203: Nemotron IQ3_XXS, padded, frag).
    {"qwen3-30b-a3b", 128, 8, 2048, 768, QType::NVFP4, QType::NVFP4},
    {"nemotron-3.5-30b-a3b", 128, 6, 2688, 2048, QType::GGML_IQ3_XXS, QType::GGML_IQ3_XXS, true,
     true, true, 128},
    {"nemotron-3.5-30b-a3b", 128, 6, 2688, 2048, QType::GGML_IQ3_XXS, QType::GGML_IQ3_XXS, true,
     false, false, 128},
    // The same banks in fragment order (frag at Auto): the R8 fragment rules, the codec refusals,
    // Volta's HMMA layout rule and the W4A8 fragment admission.
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_Q4_K, QType::GGML_Q6_K, false, true, false},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_IQ2_S, QType::GGML_IQ4_XS, false, true,
     false},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_IQ2_S, QType::GGML_IQ3_XXS, false, true,
     true},
    {"qwen3.6-35b-a3b", 256, 8, 2048, 512, QType::GGML_IQ3_XXS, QType::GGML_IQ4_XS, false, true,
     false},
    {"gemma-4-26b-a4b", 128, 8, 2816, 704, QType::GGML_IQ2_S, QType::GGML_IQ4_NL, false, true,
     false},
    {"gemma-4-26b-a4b", 128, 8, 2816, 704, QType::GGML_Q4_0, QType::GGML_Q4_0, false, true, false},
    {"gpt-oss-20b", 32, 4, 2880, 2880, QType::GGML_Q4_0, QType::GGML_Q4_0, false, true, true},
};

constexpr std::int32_t kTokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 32, 64, 128, 512, 2048};

enum class Profile { Speed, Prefill, Invariant };

const char* profile_name(Profile p) {
    return p == Profile::Speed ? "speed" : p == Profile::Prefill ? "prefill" : "invariant";
}

int g_evidence_mismatches = 0;

// The authority list's name behind each label the goldens print.
std::string_view source_of(moe::MoeAuthority a) {
    using A = moe::MoeAuthority;
    switch (a) {
    case A::Tuner:
        return "tuner";
    case A::Invariant:
        return "invariant-profile";
    case A::TableAsMeasured:
    case A::TableW4a8:
    case A::TableClassA8:
    case A::TableR8Refused:
        return "route-table";
    case A::VoltaHmma:
        return "volta-hmma";
    case A::PrefillW4a8:
        return "prefill-w4a8";
    case A::PrefillSchedule:
        return "prefill-schedule";
    case A::SpeedW4a8:
        return "speed-w4a8";
    case A::SpeedSchedule:
    case A::DecodeR8:
    case A::DecodeStreaming:
        return "decode";
    }
    return "?";
}

// Whether an answer of this label is a table's or the tuner's (measured or carried), not a rule's.
bool measured_label(moe::MoeAuthority a) {
    using A = moe::MoeAuthority;
    return a == A::Tuner || a == A::TableAsMeasured || a == A::TableW4a8 || a == A::TableClassA8 ||
           a == A::TableR8Refused;
}

const char* refusal(moe::BankRefusal why) { return moe::bank_refusal_name(why); }

// What one call decides: the plan, its authority and each stage's kernel with the banks'
// refusals of R8 and of the grouped W4A8 where they are not None.
std::string describe(const ops::MoeWeights& weights, const ops::MoeConfig& config, std::int32_t t,
                     bool prefill) {
    const moe::MoePlanQuery q = moe::moe_plan_query(weights, config, t, prefill);
    // The evidence names the authority that answered; a table's or the tuner's is measured or
    // carried, a rule's a ranker's (the decode R8 route carries a board's when one is near).
    if (q.evidence.source != source_of(q.authority) ||
        (measured_label(q.authority) && q.evidence.kind == moe::Evidence::Kind::Ranker) ||
        (!measured_label(q.authority) && q.authority != moe::MoeAuthority::DecodeR8 &&
         q.evidence.kind != moe::Evidence::Kind::Ranker)) {
        ++g_evidence_mismatches;
    }
    const moe::MoeStageRoutes stages = moe::stage_routes(weights, q.plan);
    std::ostringstream out;
    out << moe::plan_name(q.plan) << " jobs=" << q.plan.job_columns << " "
        << moe::authority_name(q.authority) << " stages=" << stages.gate_up.name << ","
        << stages.down.name;
    for (const auto& [name, s] : {std::pair{"gate_up", &stages.gate_up}, {"down", &stages.down}}) {
        if (s->r8 != moe::BankRefusal::None) out << " " << name << ".r8=" << refusal(s->r8);
        if (s->w4a8 != moe::BankRefusal::None) out << " " << name << ".w4a8=" << refusal(s->w4a8);
    }
    return out.str();
}

std::string layer_name(const Layer& l) {
    return std::string(l.model) + " " + generic::qtype_name(l.gate_up) + "/" +
           generic::qtype_name(l.down) + " E" + std::to_string(l.experts) + " k" +
           std::to_string(l.top_k) + " H" + std::to_string(l.hidden) + " F" +
           std::to_string(l.inter) + (l.relu2 ? " relu2" : "") +
           (l.pad != 0 ? " pad" + std::to_string(l.pad) : "") +
           (l.frag_gate_up || l.frag_down
                ? std::string(" frag:") + (l.frag_gate_up ? "gate_up" : "") +
                      (l.frag_gate_up && l.frag_down ? "+" : "") + (l.frag_down ? "down" : "")
                : "");
}

// Appends the layer's lines to each board's dump.
void dump_layer(const Layer& l, std::map<std::string, std::string>& dumps) {
    void* const payload     = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    const std::int32_t sets = l.relu2 ? 1 : 2;
    const std::string name  = layer_name(l);
    std::optional<qw::PackedWeight> gate_up, down;
    std::string unpacked;
    try {
        const auto options_for = [&](QType q) {
            qw::PatternedWeightOptions o;
            if (q == QType::NVFP4) o.weight_scale_divisor = o.input_scale_divisor = 1.0F;
            return o;
        };

        gate_up.emplace(qw::make_patterned_weight(l.gate_up, l.experts * sets * l.inter,
                                                  l.hidden + l.pad, 7U, options_for(l.gate_up)));
        down.emplace(qw::make_patterned_weight(l.down, l.experts * l.hidden, l.inter, 9U,
                                               options_for(l.down)));
    } catch (const std::invalid_argument& error) {
        unpacked = name + " not packed: " + error.what() + "\n";
    }
    ops::MoeConfig config;
    config.top_k = l.top_k;
    if (l.relu2) config.activation = ops::MoeActivation::Relu2;
    ops::MoeWeights weights;
    if (gate_up) {
        // The BF16 router: a contiguous descriptor, as the op tests build it.
        Weight r{};
        r.qtype  = QType::BF16;
        r.layout = QuantLayout::Contiguous;
        r.qdata = r.payload = payload;
        r.payload_bytes     = static_cast<std::size_t>(l.experts) * l.hidden * 2;
        r.n = r.shape[0] = r.padded_shape[0] = l.experts;
        r.k = r.shape[1] = r.padded_shape[1] = l.hidden;
        r.ndim                               = 2;
        weights.router                       = r;
        weights.gate_up                      = gate_up->device_weight(payload);
        weights.down                         = down->device_weight(payload);
        if (l.frag_gate_up) weights.gate_up.layout = QuantLayout::GgmlFrag;
        if (l.frag_down) weights.down.layout = QuantLayout::GgmlFrag;
    }
    for (const generic::FactsBoard& b : boards()) {
        std::string& out = dumps[b.name];
        if (!gate_up) {
            out += unpacked;
            continue;
        }
        const DeviceClass d = generic::board_device_class(b);
        const ScopedDeviceClass scoped(d);
        for (const LinearPolicy policy :
             {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
            if (policy == LinearPolicy::AllowA4 && l.gate_up != QType::NVFP4 &&
                l.down != QType::NVFP4) {
                continue;
            }
            weights.expert_policy = policy;
            for (const Profile profile : {Profile::Speed, Profile::Prefill, Profile::Invariant}) {
                std::optional<ops::ScopedInvariantRoutes> invariant;
                if (profile == Profile::Invariant) invariant.emplace();
                std::string last;
                std::int32_t first_t = 0, last_t = 0;
                const auto flush = [&] {
                    if (last.empty()) return;
                    out += name + " p" + std::to_string(static_cast<int>(policy)) + " " +
                           profile_name(profile) + " T=" + std::to_string(first_t);
                    if (last_t != first_t) out += ".." + std::to_string(last_t);
                    out += " " + last + "\n";
                };
                for (const std::int32_t t : kTokens) {
                    std::string what;
                    try {
                        what = describe(weights, config, t, profile == Profile::Prefill);
                    } catch (const std::exception& error) {
                        what = std::string("threw: ") + error.what();
                    }
                    if (what != last) {
                        flush();
                        last    = what;
                        first_t = t;
                    }
                    last_t = t;
                }
                flush();
            }
        }
    }
}

std::filesystem::path golden_dir() { return std::filesystem::path(NINFERNO_MOE_PLAN_GOLDEN_DIR); }

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool update = false;
        for (int i = 1; i < argc; ++i) update = update || std::string_view(argv[i]) == "--update";
        std::map<std::string, std::string> dumps;
        for (const generic::FactsBoard& b : boards()) {
            dumps[b.name] = std::string("# ") + b.name + " " + kBuildSet + "\n";
        }
        for (const Layer& l : kLayers) dump_layer(l, dumps);
        int failed                                         = 0;
        const std::vector<std::string_view> expected_order = {
            "tuner",        "invariant-profile", "route-table", "volta-hmma",
            "prefill-w4a8", "prefill-schedule",  "speed-w4a8",  "decode"};
        if (moe::authority_order() != expected_order) {
            std::cerr << "moe plan golden: the authority order moved
                         ";
                ++failed;
        }
        if (g_evidence_mismatches != 0) {
            std::cerr << "moe plan golden: " << g_evidence_mismatches
                      << " answers whose evidence does not match their authority
                         ";
                ++failed;
        }
        for (const generic::FactsBoard& b : boards()) {
            const auto path        = golden_dir() / kBuildSet / (std::string(b.name) + ".txt");
            const std::string& now = dumps[b.name];
            if (update) {
                std::filesystem::create_directories(path.parent_path());
                std::ofstream file(path, std::ios::binary);
                file << now;
                file.flush();
                if (!file.good()) {
                    std::cerr << b.name << ": cannot write " << path.string() << "\n";
                    ++failed;
                    continue;
                }
                std::cout << b.name << ": wrote " << path.string() << '\n';
                continue;
            }
            const std::string was = read_file(path);
            if (was == now) {
                std::cout << b.name << ": plans as the golden file\n";
                continue;
            }
            std::istringstream a(was), c(now);
            std::string la, lc;
            int shown = 0, line = 0;
            while (shown < 20) {
                const bool ga = static_cast<bool>(std::getline(a, la));
                const bool gc = static_cast<bool>(std::getline(c, lc));
                if (!ga && !gc) break;
                ++line;
                if (!ga) la = "<end>";
                if (!gc) lc = "<end>";
                if (la != lc) {
                    std::cerr << b.name << ":" << line << "\n  golden: " << la
                              << "\n  now:    " << lc << '\n';
                    ++shown;
                }
            }
            std::cerr << b.name << ": plans differ from " << path.string()
                      << (was.empty() ? " (missing; run with --update)" : "") << '\n';
            ++failed;
        }
        if (failed != 0) {
            std::cerr << "moe plan golden: " << failed << " boards differ\n";
            return 1;
        }
        if (!update) std::cout << "moe plan golden: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "moe plan golden: " << error.what() << '\n';
        return 1;
    }
}
