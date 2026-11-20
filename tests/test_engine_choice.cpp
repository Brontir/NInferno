// Which release engine runs a GPU (apps/dispatch/engine_choice.h, #991): the most specific,
// newest match; a plain engine runs a newer minor of its major only with the same KV row codec.
// No GPU.
#include "dispatch/engine_choice.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using ninferno::dispatch::Engine;
using ninferno::dispatch::choose;

int g_failed = 0;

void expect_tag(const std::vector<Engine>& engines, int cc, const std::optional<std::string>& tag,
                const std::string& what) {
    const std::optional<Engine> e        = choose(engines, cc);
    const std::optional<std::string> got = e ? std::optional<std::string>(e->tag) : std::nullopt;
    if (got != tag) {
        ++g_failed;
        std::cerr << "FAIL: " << what << ": sm_" << cc << " got " << got.value_or("none")
                  << ", expected " << tag.value_or("none") << '\n';
    }
}

} // namespace

int main() {
    // The default release (tools/release/package.sh): 120a 100 90 89 (PTX) 86 80 75 70.
    const std::vector<Engine> release = {{"120a", 120, true, false}, {"100", 100, false, false},
                                         {"90", 90, false, false},   {"89", 89, false, true},
                                         {"86", 86, false, false},   {"80", 80, false, false},
                                         {"75", 75, false, false},   {"70", 70, false, false}};
    expect_tag(release, 120, "120a", "an arch-specific engine on its own arch");
    expect_tag(release, 89, "89", "Ada on its own engine");
    expect_tag(release, 87, "86", "a newer minor of the same codec (Orin)");
    expect_tag(release, 86, "86", "Ampere on its own engine");
    expect_tag(release, 75, "75", "Turing on its own engine");
    expect_tag(release, 121, "89", "an unknown newer GPU runs the PTX engine");

    // A release without the 89 engine: the 86 engine writes int8 rows, Ada's plan prices E4M3, so
    // an sm_89 card gets none (main.cpp refuses it with no_engine_reason).
    const std::vector<Engine> no89 = {
        {"120a", 120, true, false}, {"86", 86, false, false}, {"80", 80, false, false}};
    expect_tag(no89, 89, std::nullopt, "an 86 engine on an sm_89 card (another KV row codec)");
    expect_tag(no89, 86, "86", "Ampere still runs the 86 engine");
    expect_tag(no89, 87, "86", "the 86 engine still runs sm_87");
    // The 80 engine (int8 rows below sm_89) still runs sm_86 where no 86 engine ships.
    expect_tag({{"80", 80, false, false}}, 86, "80", "the 80 engine on sm_86 (same codec)");
    // PTX keeps the codec too: an int8 engine with PTX does not run an E4M3 card.
    expect_tag({{"86", 86, false, true}}, 90, std::nullopt, "an 86 PTX engine on sm_90");
    expect_tag({{"89", 89, false, true}}, 90, "89", "the 89 PTX engine on sm_90");

    // The refusal names its cause (#991 review).
    using ninferno::dispatch::no_engine_reason;
    const auto expect_reason = [](bool ok, const std::string& what) {
        if (!ok) {
            ++g_failed;
            std::cerr << "FAIL: " << what << '\n';
        }
    };
    expect_reason(no_engine_reason(no89, 89).find("KV row codes") != std::string::npos,
                  "an sm_89 card under an 86 engine is refused for its row codec");
    expect_reason(no_engine_reason(release, 60) == "is older than this release supports",
                  "a Pascal card is older than the release");
    expect_reason(no_engine_reason({{"86", 86, false, false}}, 120) ==
                      "is not supported by this release",
                  "sm_120 under an 86-only release (no PTX) is not supported");

    if (g_failed != 0) { return 1; }
    std::cout << "engine choice: ok\n";
    return 0;
}
