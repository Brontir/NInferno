#include "spec/config.h"

#include <stdexcept>
#include <vector>

namespace ninferno::spec {
namespace {

std::vector<std::string> split(std::string_view text, char separator) {
    std::vector<std::string> out(1);
    for (const char c : text) {
        if (c == separator) {
            out.emplace_back();
        } else {
            out.back() += c;
        }
    }
    return out;
}

std::int32_t number(const std::string& text, std::string_view label) {
    if (text.empty() || text.size() > 3 ||
        text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("invalid speculation config '" + std::string(label) + "'");
    }
    return static_cast<std::int32_t>(std::stoi(text));
}

} // namespace

const char* mode_name(Mode mode) noexcept {
    switch (mode) {
    case Mode::None:
        return "plain";
    case Mode::Mtp:
        return "mtp";
    case Mode::DFlash:
        return "dflash";
    case Mode::DFlash2:
        return "dflash2";
    case Mode::Ngram:
        return "ngram";
    }
    return "unknown";
}

std::string SpecConfig::label() const {
    if (mode == Mode::None) { return "plain"; }
    std::string out = std::string(mode_name(mode)) + ":" + std::to_string(k);
    if (adaptive()) { out += ":" + std::to_string(k_min); }
    if (proposal_head) { out += "+head"; }
    return out;
}

SpecConfig SpecConfig::parse(std::string_view label) {
    if (label == "plain" || label == "none" || label == "off") { return {}; }
    std::string_view body = label;
    SpecConfig out;
    if (body.ends_with("+head")) {
        out.proposal_head = true;
        body.remove_suffix(5);
    }
    const auto parts = split(body, ':');
    if (parts.size() < 2 || parts.size() > 3) {
        throw std::invalid_argument("invalid speculation config '" + std::string(label) + "'");
    }
    if (parts[0] == "mtp") {
        out.mode = Mode::Mtp;
    } else if (parts[0] == "dflash") {
        out.mode = Mode::DFlash;
    } else if (parts[0] == "dflash2") {
        out.mode = Mode::DFlash2;
    } else if (parts[0] == "ngram") {
        out.mode = Mode::Ngram;
    } else {
        throw std::invalid_argument("invalid speculation config '" + std::string(label) + "'");
    }
    out.k = number(parts[1], label);
    if (parts.size() == 3) {
        if (out.mode != Mode::Mtp) {
            throw std::invalid_argument("adaptive depth is an MTP option: '" + std::string(label) +
                                        "'");
        }
        out.k_min = number(parts[2], label);
    }
    if (out.k < 1 || out.k > 16 || (parts.size() == 3 && (out.k_min < 1 || out.k_min >= out.k))) {
        throw std::invalid_argument("invalid speculation depth in '" + std::string(label) + "'");
    }
    return out;
}

} // namespace ninferno::spec
