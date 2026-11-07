// The route cost model's constants (ops/generic/route_cost_constants.h), read only through their
// one descriptor, kRouteCostConstants: every entry has a unique name, a provenance and values in
// the range of its kind (a fraction in (0, 1], every other kind positive and finite). `--list`
// prints them, the fitted ones first: the numbers the model still takes from route_tune timings
// instead of deriving them. No GPU.

#include "ops/generic/route_cost_constants.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <string_view>

namespace rc = ninferno::ops::detail::generic::route_cost;

namespace {

const char* provenance_name(rc::Provenance p) {
    switch (p) {
    case rc::Provenance::Fact:
        return "fact";
    case rc::Provenance::Counted:
        return "counted";
    case rc::Provenance::Fitted:
        return "fitted";
    }
    return "?";
}

bool in_range(rc::Kind kind, double v) {
    if (!std::isfinite(v) || v <= 0.0) return false;
    return kind != rc::Kind::Fraction || v <= 1.0;
}

} // namespace

int main(int argc, char** argv) {
    int failures = 0;
    std::set<std::string_view> names;
    std::size_t fitted = 0;
    for (const auto& c : rc::kRouteCostConstants) {
        if (c.name.empty() || !names.insert(c.name).second) {
            std::fprintf(stderr, "constant name '%.*s' empty or listed twice\n",
                         static_cast<int>(c.name.size()), c.name.data());
            ++failures;
        }
        if (c.values.empty()) {
            std::fprintf(stderr, "%.*s: no values\n", static_cast<int>(c.name.size()),
                         c.name.data());
            ++failures;
        }
        for (const double v : c.values) {
            if (in_range(c.kind, v)) continue;
            std::fprintf(stderr, "%.*s: %g is out of its kind's range\n",
                         static_cast<int>(c.name.size()), c.name.data(), v);
            ++failures;
        }
        fitted += c.from == rc::Provenance::Fitted;
    }
    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (auto want : {rc::Provenance::Fitted, rc::Provenance::Counted, rc::Provenance::Fact}) {
            for (const auto& c : rc::kRouteCostConstants) {
                if (c.from != want) continue;
                std::printf("%-8s %-32.*s", provenance_name(c.from),
                            static_cast<int>(c.name.size()), c.name.data());
                for (const double v : c.values) std::printf(" %g", v);
                std::printf("\n");
            }
        }
    }
    std::printf("route cost constants: %zu listed, %zu fitted\n",
                std::size(rc::kRouteCostConstants), fitted);
    return failures == 0 ? 0 : 1;
}
