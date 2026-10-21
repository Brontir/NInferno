#include "spec/context_class.h"

#include "core/ascii.h"
#include "core/device_class.h"

#include "spec/json_fields.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <regex>
#include <stdexcept>

namespace ninferno::spec {
namespace {

using Json = nlohmann::json;

[[noreturn]] void refuse(const std::string& what) {
    throw std::invalid_argument("context_classes: " + what);
}

std::string required_text(const Json& object, const char* key, const std::string& where) {
    auto text = json_text(object, key);
    if (text.empty()) { refuse(where + ": " + key + " is missing"); }
    return text;
}

std::int32_t required_count(const Json& object, const char* key, const std::string& where) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_number_integer() || it->get<std::int64_t>() < 1 ||
        it->get<std::int64_t>() > 1024) {
        refuse(where + ": " + key + " must be a positive integer");
    }
    return it->get<std::int32_t>();
}

bool fast(const Json& byte_class) {
    return json_text(byte_class, "activations").starts_with("fast");
}

// A key value that stands for something not recorded: never matched to a running server.
bool placeholder(std::string_view value) {
    return value == "auto" || value.starts_with("placeholder:");
}

std::string percent(double fraction) {
    char text[32];
    std::snprintf(text, sizeof text, "%.1f %%", 100.0 * fraction);
    return text;
}

// The fields of a key in lookup order; the first that differs names why an entry does not apply.
struct Field {
    const char* name;
    bool same;
    std::string entry, wanted;
};

std::vector<Field> compare(const AcceptanceKey& entry, const AcceptanceKey& wanted) {
    const bool lanes_matter = fast(entry.byte_class) || fast(wanted.byte_class);
    return {
        {"target_artifact", entry.target_artifact == wanted.target_artifact, entry.target_artifact,
         wanted.target_artifact},
        {"drafter", entry.drafter == wanted.drafter, entry.drafter, wanted.drafter},
        {"drafter_artifact", entry.drafter_artifact == wanted.drafter_artifact,
         entry.drafter_artifact, wanted.drafter_artifact},
        {"byte_class", entry.byte_class == wanted.byte_class, entry.byte_class.dump(),
         wanted.byte_class.dump()},
        {"sampling", entry.sampling == wanted.sampling, entry.sampling, wanted.sampling},
        {"depth", entry.depth == wanted.depth, std::to_string(entry.depth),
         std::to_string(wanted.depth)},
        {"lanes", !lanes_matter || entry.lanes == wanted.lanes, std::to_string(entry.lanes),
         std::to_string(wanted.lanes)},
        {"corpus", entry.corpus == wanted.corpus, entry.corpus, wanted.corpus},
    };
}

} // namespace

bool known_route(std::string_view route) {
    static const std::regex name("[a-z0-9_]+(\\.[a-z0-9]+)*");
    if (!std::regex_match(route.begin(), route.end(), name)) { return false; }
    const auto family = route.substr(0, route.find('.'));
    for (const char* known : {"scalar", "rmma", "w4a4", "tiled", "w4a8", "w4a8ws", "staged",
                              "nvfp4t", "r8", "ggml_mmq", "ggml_vec", "i8"}) {
        if (family == known) { return true; }
    }
    return false;
}

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
    const auto entries = object.value("entries", Json::array());
    if (!entries.is_array()) { refuse("entries must be a list"); }
    // A GPU class key as the profiles' gpu_class names it: a chip, optionally its SM count.
    static const std::regex gpu_class("[a-z]+[0-9]+(:[0-9]+sm)?");
    for (const auto& item : entries) {
        if (!item.is_object()) { refuse("an entry must be an object"); }
        ContextEntry entry;
        auto& key            = entry.key;
        key.model            = required_text(item, "model", "entry");
        const auto where     = "entry " + key.model;
        key.target_artifact  = required_text(item, "target_artifact", where);
        key.drafter          = required_text(item, "drafter", where);
        key.drafter_artifact = required_text(item, "drafter_artifact", where);
        key.sampling         = required_text(item, "sampling", where);
        key.corpus           = required_text(item, "corpus", where);
        key.depth            = required_count(item, "depth", where);
        key.lanes            = required_count(item, "lanes", where);
        entry.schedule       = required_text(item, "schedule", where);
        entry.source         = json_text(item, "source");
        const auto tolerance = item.find("tolerance");
        if (tolerance == item.end() || !tolerance->is_number() ||
            tolerance->get<double>() != kContextTolerance) {
            refuse(where + ": tolerance must be " + std::to_string(kContextTolerance));
        }
        entry.tolerance = tolerance->get<double>();
        key.byte_class  = item.value("byte_class", Json());
        if (!key.byte_class.is_object()) { refuse(where + ": byte_class must be an object"); }
        for (const char* field : {"activations", "kv", "summation", "parity"}) {
            (void)required_text(key.byte_class, field, where + " byte_class");
        }
        for (const auto& [field, value] : key.byte_class.items()) {
            if (!value.is_string()) { refuse(where + ": byte_class." + field + " must be text"); }
        }
        const auto activations = json_text(key.byte_class, "activations");
        if (activations != "exact" && activations != "fast-a8" && activations != "fast-a4") {
            refuse(where + ": byte_class.activations '" + activations +
                   "' is not exact, fast-a8 or fast-a4");
        }
        const auto cards = item.value("cards", Json());
        if (!cards.is_array() || cards.empty()) { refuse(where + ": cards must be a list"); }
        for (const auto& c : cards) {
            if (!c.is_object()) { refuse(where + ": a card must be an object"); }
            CardFactors card;
            card.gpu_class = ascii_lower(json_text(c, "gpu_class"));
            card.arch      = gpu_chip_arch(card.gpu_class);
            card.route     = ascii_lower(json_text(c, "route"));
            if (!std::regex_match(card.gpu_class, gpu_class) || card.arch.empty()) {
                refuse(where + ": '" + card.gpu_class +
                       "' is not a known GPU class (ga102, gb203:84sm, sm86)");
            }
            if (const auto colon = card.gpu_class.find(':'); colon != std::string::npos) {
                const int sms  = std::atoi(card.gpu_class.c_str() + colon + 1);
                const int full = gpu_chip_full_sms(card.gpu_class);
                if (sms < 1 || (full > 0 && sms > full)) {
                    refuse(where + ": " + card.gpu_class + " names more SMs than its chip has (" +
                           std::to_string(full) + ")");
                }
            }
            if (c.contains("route") && !known_route(card.route)) {
                refuse(where + " " + card.gpu_class + ": unknown route '" + card.route + "'");
            }
            if (json_text(c, "schedule") != entry.schedule) {
                refuse(where + " " + card.gpu_class + ": schedule '" + json_text(c, "schedule") +
                       "' is not the entry's '" + entry.schedule + "'");
            }
            const auto same = [&](const CardFactors& o) {
                return o.gpu_class == card.gpu_class && o.route == card.route;
            };
            if (std::any_of(entry.cards.begin(), entry.cards.end(), same)) {
                refuse(where + ": card " + card.gpu_class + " is listed twice");
            }
            const auto values = c.value("factors", Json());
            if (!values.is_array() || values.size() != kContextClasses) {
                refuse(where + " " + card.gpu_class +
                       ": factors must hold 4 values (short, mid, long, huge)");
            }
            for (std::size_t i = 0; i < kContextClasses; ++i) {
                const auto& v = values[i];
                if (v.is_null()) { continue; }
                if (!v.is_number() ||
                    !(v.get<double>() > 0.0 && v.get<double>() <= kMaxContextFactor)) {
                    refuse(where + " " + card.gpu_class +
                           ": each factor must be in (0, 1.5] or null");
                }
                card.factors[i] = v.get<double>();
            }
            if (card.factors[0] != 1.0) {
                refuse(where + " " + card.gpu_class + ": the short factor must be 1");
            }
            entry.cards.push_back(std::move(card));
        }
        for (const auto& other : out.entries) {
            const auto fields = compare(other.key, key);
            const bool same =
                other.key.model == key.model &&
                std::all_of(fields.begin(), fields.end(), [](const Field& f) { return f.same; });
            if (same) { refuse(where + ": two entries of the same key"); }
        }
        out.entries.push_back(std::move(entry));
    }
    return out;
}

ClassFactor factor(const ContextFactors& factors, const AcceptanceKey& key, ContextClass c) {
    const auto unknown = [](std::string reason) {
        return ClassFactor{std::nullopt, std::move(reason)};
    };
    if (!factors.present) { return unknown("the registry has no context_classes"); }
    if (factors.bounds != kContextBounds) {
        return unknown("context_classes measured for other class bounds");
    }
    // The entry of this model that matches every field, else the one matching the longest run of
    // fields in lookup order, whose first differing field is the reason.
    const ContextEntry* best = nullptr;
    std::size_t best_run     = 0;
    std::string why;
    for (const auto& entry : factors.entries) {
        if (entry.key.model != key.model) { continue; }
        const auto fields = compare(entry.key, key);
        std::size_t run   = 0;
        while (run < fields.size() && fields[run].same) { ++run; }
        if (best == nullptr || run > best_run) {
            best     = &entry;
            best_run = run;
            why      = run < fields.size()
                           ? std::string(fields[run].name) + " differs (measured " + fields[run].entry +
                            ", running " + fields[run].wanted + ")"
                           : std::string();
        }
    }
    if (best == nullptr) { return unknown("no context_classes entry for model " + key.model); }
    if (!why.empty()) { return unknown(why); }
    // A key with a placeholder matched only another placeholder: what it was measured for is not
    // known, whatever the cards say.
    const auto& k = best->key;
    for (const auto& [field, value] : std::initializer_list<std::pair<std::string, std::string>>{
             {"target_artifact", k.target_artifact},
             {"drafter_artifact", k.drafter_artifact},
             {"sampling", k.sampling}}) {
        if (placeholder(value)) { return unknown("placeholder key: " + field + "=" + value); }
    }
    for (const auto& [field, value] : k.byte_class.items()) {
        if (placeholder(value.get<std::string>())) {
            return unknown("placeholder key: byte_class." + field + "=" + value.get<std::string>());
        }
    }
    if (!k.corpus.starts_with("sha256:")) { return unknown("placeholder key: corpus=" + k.corpus); }

    const auto i = static_cast<std::size_t>(c);
    std::vector<const CardFactors*> measured;
    for (const auto& card : best->cards) {
        if (card.factors[i]) { measured.push_back(&card); }
    }
    if (measured.empty()) {
        return unknown("no " + std::string(class_name(c)) + " context factor for " + key.drafter);
    }
    bool differ = false;
    for (const auto* a : measured) {
        for (const auto* b : measured) {
            differ = differ || a->arch != b->arch ||
                     (!a->route.empty() && !b->route.empty() && a->route != b->route);
        }
    }
    if (!differ) {
        std::string cards;
        for (const auto* card : measured) {
            cards += (cards.empty() ? "" : ", ") + card->gpu_class;
        }
        return unknown("measured on " + std::to_string(measured.size()) + " card(s) (" + cards +
                       "): needs 2 cards of different generations or routes");
    }
    double sum = 0.0;
    for (const auto* card : measured) { sum += *card->factors[i]; }
    const double mean = sum / static_cast<double>(measured.size());
    double spread     = 0.0; // the farthest card from the mean, relative
    for (const auto* card : measured) {
        spread = std::max(spread, std::abs(*card->factors[i] - mean) / mean);
    }
    // (A card exactly at the tolerance stands: 1e-12 absorbs the rounding of the division.)
    if (spread > best->tolerance + 1e-12) {
        return unknown("unreliable: hardware-dependent (cards disagree by " + percent(spread) +
                       " from their mean, tolerance " + percent(best->tolerance) + ")");
    }
    return ClassFactor{mean, {}};
}

} // namespace ninferno::spec
