#include "spec/context_class.h"

#include "core/ascii.h"

#include "spec/json_fields.h"

#include <regex>
#include <set>
#include <stdexcept>

namespace ninferno::spec {
namespace {

using Json = nlohmann::json;

[[noreturn]] void refuse(const std::string& what) {
    throw std::invalid_argument("context_classes: " + what);
}

} // namespace

ContextClass class_of(std::int64_t prompt_tokens) noexcept {
    if (prompt_tokens < kContextBounds[0]) { return ContextClass::Short; }
    if (prompt_tokens < kContextBounds[1]) { return ContextClass::Mid; }
    if (prompt_tokens < kContextBounds[2]) { return ContextClass::Long; }
    return ContextClass::Huge;
}

const char* class_name(ContextClass c) noexcept {
    switch (c) {
    case ContextClass::Short:
        return "short";
    case ContextClass::Mid:
        return "mid";
    case ContextClass::Long:
        return "long";
    case ContextClass::Huge:
        return "huge";
    }
    return "unknown";
}

ContextFactors parse_context_factors(const Json& object) {
    if (!object.is_object()) { refuse("not an object"); }
    ContextFactors out;
    out.present       = true;
    const auto bounds = object.find("bounds");
    if (bounds == object.end() || !bounds->is_array() || bounds->size() != out.bounds.size()) {
        refuse("bounds must be three prompt lengths");
    }
    for (std::size_t i = 0; i < out.bounds.size(); ++i) {
        const auto& b = (*bounds)[i];
        if (!b.is_number_integer() || b.get<std::int64_t>() <= 0 ||
            (i > 0 && b.get<std::int64_t>() <= out.bounds[i - 1])) {
            refuse("bounds must be strictly increasing positive integers");
        }
        out.bounds[i] = b.get<std::int64_t>();
    }
    const auto stamp = object.value("stamp", Json::object());
    if (!stamp.is_object()) { refuse("stamp must be an object"); }
    out.engine       = json_text(stamp, "engine");
    out.source       = json_text(stamp, "source");
    if (out.engine.empty()) { refuse("stamp.engine is missing"); }
    if (const auto it = stamp.find("measured_on"); it != stamp.end()) {
        if (!it->is_array()) { refuse("stamp.measured_on must be a list of GPU classes"); }
        // A GPU class key as the profiles' gpu_class names it: a chip, optionally its SM count.
        static const std::regex gpu_class("[a-z]+[0-9]+(:[0-9]+sm)?");
        for (const auto& card : *it) {
            const auto name = card.is_string() ? ascii_lower(card.get<std::string>()) : "";
            if (!std::regex_match(name, gpu_class)) {
                refuse("stamp.measured_on: '" + (card.is_string() ? name : card.dump()) +
                       "' is not a GPU class (ga102, gb203:84sm)");
            }
            out.measured_on.push_back(name);
        }
    }
    const auto factors = object.value("factors", Json::object());
    if (!factors.is_object()) { refuse("factors must be an object"); }
    for (const auto& [kind, values] : factors.items()) {
        if (!values.is_array() || values.size() != kContextClasses) {
            refuse("factors." + kind + " must hold 4 values (short, mid, long, huge)");
        }
        std::array<std::optional<double>, kContextClasses> row;
        for (std::size_t i = 0; i < kContextClasses; ++i) {
            const auto& v = values[i];
            if (v.is_null()) { continue; }
            if (!v.is_number() ||
                !(v.get<double>() > 0.0 && v.get<double>() <= kMaxContextFactor)) {
                refuse("factors." + kind + ": each factor must be in (0, 1.5] or null");
            }
            row[i] = v.get<double>();
        }
        if (row[0] != 1.0) { refuse("factors." + kind + ": the short factor must be 1"); }
        out.factors.emplace(kind, row);
    }
    return out;
}

ClassFactor factor(const ContextFactors& factors, std::string_view kind, ContextClass c,
                   std::string_view engine) {
    const auto unknown = [](std::string reason) {
        return ClassFactor{std::nullopt, std::move(reason)};
    };
    if (!factors.present) { return unknown("the registry has no context_classes"); }
    if (factors.bounds != kContextBounds) {
        return unknown("context_classes measured for other class bounds");
    }
    if (factors.engine != engine) {
        return unknown("context_classes stamped " + factors.engine + ", engine is " +
                       std::string(engine));
    }
    const std::set<std::string> cards(factors.measured_on.begin(), factors.measured_on.end());
    if (cards.size() < kMinContextCards) {
        return unknown("context_classes measured on " + std::to_string(cards.size()) +
                       " card(s), fewer than " + std::to_string(kMinContextCards) + " cards");
    }
    const auto row = factors.factors.find(kind);
    if (row == factors.factors.end()) {
        return unknown("no context_classes factors for " + std::string(kind));
    }
    const auto& value = row->second[static_cast<std::size_t>(c)];
    if (!value) {
        return unknown("no " + std::string(class_name(c)) + " context factor for " +
                       std::string(kind));
    }
    return ClassFactor{value, {}};
}

} // namespace ninferno::spec
