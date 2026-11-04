#include "runtime/engine/decision_routes.h"

#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::runtime {

namespace {

namespace generic = ops::detail::generic;
using ops::LinearPolicy;

const char* policy_name(LinearPolicy p) {
    switch (p) {
    case LinearPolicy::A16Only:
        return "a16only";
    case LinearPolicy::AllowA8:
        return "allowa8";
    case LinearPolicy::AllowA4:
        return "allowa4";
    }
    return "a16only";
}

std::string key_of(const generic::SeenProblem& s) {
    const generic::Problem& p = s.problem;
    return std::string(generic::qtype_name(p.qtype)) + ' ' + policy_name(p.policy) + ' ' +
           generic::epilogue_name(p.epilogue) + ' ' + std::to_string(p.n) + 'x' +
           std::to_string(p.k) + " of " + std::to_string(s.parent_rows) + " layout " +
           std::to_string(static_cast<unsigned>(s.layout));
}

std::string name_or_none(const std::optional<generic::Route>& r) {
    return r ? generic::route_name(*r) : std::string("none");
}

std::string line_of(const DeviceClass& device, const generic::SeenProblem& s,
                    std::int32_t prefill_chunk) {
    const generic::Problem& p = s.problem;
    const auto ask = [&](std::int32_t tokens, bool prefill) {
        return name_or_none(generic::route_query(device, p.qtype, p.policy, p.epilogue, p.n, p.k,
                                                 tokens, prefill, s.parent_rows, s.layout));
    };
    std::string out;
    std::int32_t from = 1;
    std::string run   = ask(1, false);
    for (std::int32_t t = 2; t <= kDecisionRouteWidths + 1; ++t) {
        const std::string at = t <= kDecisionRouteWidths ? ask(t, false) : std::string();
        if (at == run) continue;
        out += (out.empty() ? "" : ", ") + std::to_string(from) + '-' + std::to_string(t - 1) + ' ' +
               run;
        from = t;
        run  = at;
    }
    return out + "; prefill " + std::to_string(prefill_chunk) + ' ' + ask(prefill_chunk, true);
}

[[noreturn]] void malformed(const std::string& key, const char* why) {
    throw std::invalid_argument("decision route line \"" + key + "\": " + why);
}

template <typename E, typename Name>
E parse_enum(const std::string& key, const std::string& word, unsigned count, Name name) {
    for (unsigned v = 0; v < count; ++v) {
        if (word == name(static_cast<E>(v))) return static_cast<E>(v);
    }
    malformed(key, "an unknown name");
}

generic::SeenProblem parse_key(const std::string& key) {
    std::istringstream in(key);
    std::string format, policy, epilogue, shape, of, layout_word;
    long long parent = -1, layout = -1;
    if (!(in >> format >> policy >> epilogue >> shape >> of >> parent >> layout_word >> layout) ||
        of != "of" || layout_word != "layout" || parent < 0 || layout < 0 || layout > 0xFFFF) {
        malformed(key, "not \"<format> <policy> <epilogue> <N>x<K> of <rows> layout <n>\"");
    }
    std::string rest;
    if (in >> rest) malformed(key, "trailing text");
    const auto x = shape.find('x');
    if (x == std::string::npos) malformed(key, "no NxK");
    generic::SeenProblem s;
    try {
        s.problem.n = std::stoi(shape.substr(0, x));
        s.problem.k = std::stoi(shape.substr(x + 1));
    } catch (const std::exception&) { malformed(key, "N or K not a number"); }
    s.problem.qtype = parse_enum<QType>(key, format, 0x10000, [](QType q) {
        return std::string_view(generic::qtype_name(q)) == "unknown" ? std::string()
                                                                      : std::string(generic::qtype_name(q));
    });
    s.problem.policy =
        parse_enum<LinearPolicy>(key, policy, 3, [](LinearPolicy p) { return std::string(policy_name(p)); });
    s.problem.epilogue = parse_enum<generic::Epilogue>(
        key, epilogue, 3, [](generic::Epilogue e) { return std::string(generic::epilogue_name(e)); });
    s.parent_rows = static_cast<std::int32_t>(parent);
    s.layout      = static_cast<QuantLayout>(layout);
    return s;
}

std::int32_t chunk_of(const std::string& key, const std::string& value) {
    const auto at = value.find("; prefill ");
    if (at == std::string::npos) malformed(key, "no prefill chunk");
    try {
        return std::stoi(value.substr(at + 10));
    } catch (const std::exception&) { malformed(key, "the prefill chunk is not a number"); }
}

} // namespace

DecisionSection routes_of(const DeviceClass& device,
                          std::span<const generic::SeenProblem> problems,
                          std::int32_t prefill_chunk) {
    DecisionSection out;
    for (const auto& s : problems) out[key_of(s)] = line_of(device, s, prefill_chunk);
    return out;
}

DecisionSection replay_routes(const DeviceClass& device, const DecisionSection& recorded) {
    DecisionSection out;
    for (const auto& [key, value] : recorded) {
        const generic::SeenProblem s = parse_key(key);
        if (key_of(s) != key) malformed(key, "not in its canonical form");
        out[key] = line_of(device, s, chunk_of(key, value));
    }
    return out;
}

} // namespace ninferno::runtime
