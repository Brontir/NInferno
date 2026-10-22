#include "serve/activation_preflight.h"

#include "artifact/reader.h"
#include "core/machine_facts.h"
#include "ops/linear/activation_width.h"
#include "product/activation_options.h"
#include "runtime/engine/activation_precision.h"

#include <charconv>
#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace ninferno::serve {

namespace {

std::pair<int, int> parse_cc(const std::string& cc) {
    int major = 0, minor = 0;
    const auto dot = cc.find('.');
    const auto ok  = [&](std::string_view text, int& out) {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
        return error == std::errc() && end == text.data() + text.size() && !text.empty();
    };
    if (dot == std::string::npos || !ok(std::string_view(cc).substr(0, dot), major) ||
        !ok(std::string_view(cc).substr(dot + 1), minor) || major < 5 || major > 20 || minor < 0 ||
        minor > 9) {
        throw std::invalid_argument("--activations-preflight takes a compute capability "
                                    "MAJOR.MINOR such as 12.0, not " +
                                    cc);
    }
    return {major, minor};
}

struct Card {
    int major     = 0;
    int minor     = 0;
    bool mma_slow = false; // the PCI-ID table's row: TU116 / TU117 and the GTX 16 boards
};

// "MAJOR.MINOR" or "MAJOR.MINOR:0xPCIID": the board's PCI product id says what the compute
// capability cannot (mma.sync without tensor cores on TU116 / TU117, where no int8 prefill runs).
Card parse_card(const std::string& text) {
    Card card;
    const auto colon = text.find(':');
    if (colon != std::string::npos) {
        const std::string_view id = std::string_view(text).substr(colon + 1);
        unsigned value            = 0;
        const bool hex            = id.size() == 6 && id.starts_with("0x");
        const auto parsed = hex ? std::from_chars(id.data() + 2, id.data() + id.size(), value, 16)
                                : std::from_chars_result{id.data(), std::errc::invalid_argument};
        if (!hex || parsed.ec != std::errc() || parsed.ptr != id.data() + id.size()) {
            throw std::invalid_argument("--activations-preflight: a PCI product id is 0x and four "
                                        "hex digits, not " +
                                        std::string(id));
        }
        if (const auto* row = find_gpu_facts(value); row != nullptr) {
            card.mma_slow = row->mma_slow;
        }
    }
    const auto [major, minor] = parse_cc(text.substr(0, colon));
    card.major                = major;
    card.minor                = minor;
    return card;
}

nlohmann::json counts(std::uint64_t four, std::uint64_t eight, std::uint64_t sixteen) {
    return {{"4", four}, {"8", eight}, {"16", sixteen}};
}

} // namespace

nlohmann::json activation_preflight(const std::string& model, const std::string& cc) {
    const Card card = parse_card(cc);
    nlohmann::json out{{"model", model}, {"cc", cc}};
    std::error_code ec;
    const std::filesystem::path path(model);
    if (path.extension() != ".ninferno" || !std::filesystem::is_regular_file(path, ec)) {
        out["status"]  = "pending";
        out["pending"] = "not a converted .ninferno artifact on this machine yet: the server "
                         "checks the modes after the download and conversion";
        out["modes"]   = nlohmann::json::array();
        return out;
    }
    const artifact::Reader reader(path);
    const auto device             = ops::activation_device(card.major, card.minor, card.mma_slow);
    out["status"]                 = "checked";
    out["experts_settle_at_load"] = runtime::has_routed_experts(reader.directory());
    nlohmann::json modes          = nlohmann::json::array();
    for (const auto& row : runtime::preflight_activation_modes(reader.directory(), device)) {
        const auto& p = row.precision;
        nlohmann::json mode{{"mode", product::activation_mode_name(row.mode)},
                            {"refused", row.refused},
                            {"refusal", row.refusal}};
        if (!row.refused) {
            mode["resolved"]     = counts(p.bits4, p.bits8, p.bits16);
            mode["kept_wider"]   = p.kept_wider;
            mode["notes"]        = p.notes;
            mode["same_as_auto"] = p.same_as_auto;
            if (!p.faster_than_auto.empty()) { mode["faster_than_auto"] = p.faster_than_auto; }
            if (p.drafter_bits4 + p.drafter_bits8 + p.drafter_bits16 > 0) {
                mode["drafter"] = counts(p.drafter_bits4, p.drafter_bits8, p.drafter_bits16);
            }
        }
        modes.push_back(std::move(mode));
    }
    out["modes"] = std::move(modes);
    return out;
}

} // namespace ninferno::serve
