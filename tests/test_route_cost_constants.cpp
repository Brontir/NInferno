// The route cost model's constants (ops/generic/route_cost_constants.h), read only through their
// one descriptor, kRouteCostConstants: every entry has a unique name, a provenance and values in
// the range of its kind (a fraction in (0, 1], every other kind positive and finite). `--list`
// prints them, the fitted ones first: the numbers the model still takes from route_tune timings
// instead of deriving them. The header's own text is read too: every constant it defines must be
// in the descriptor and every listed one defined, so a new one cannot bypass the checks by being
// left out. No GPU.

#include "ops/generic/route_cost_constants.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
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

// The header's text: the constants it defines (`constexpr Scalar kX`, `constexpr Constant<N> kX`
// or `constexpr auto kX = Scalar{...}`, any spacing or line breaks) and the ones its descriptor
// lists (`listed(kX, ...)`), as two sets of names. Each defined constant must be listed and each
// listed one defined: a missing entry is named, not hidden behind a count.
struct HeaderNames {
    std::set<std::string> defined, listed;
    bool read = false;
};

HeaderNames header_names() {
    HeaderNames out;
    const std::filesystem::path header =
        std::filesystem::path(NINFERNO_SOURCE_DIR) / "src/ops/generic/route_cost_constants.h";
    std::ifstream in(header);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", header.string().c_str());
        return out;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // Comments may name constants too; drop them first.
    text = std::regex_replace(text, std::regex(R"(//[^\n]*)"), "");
    const std::regex definition(
        R"(constexpr\s+(?:Scalar|Constant\s*<\s*\d+\s*>)\s+(k\w+)\s*[{(=])"
        R"(|constexpr\s+auto\s+(k\w+)\s*=\s*(?:Scalar|Constant\s*<\s*\d+\s*>)\s*[{(])");
    for (std::sregex_iterator it(text.begin(), text.end(), definition), end; it != end; ++it) {
        out.defined.insert((*it)[1].matched ? (*it)[1].str() : (*it)[2].str());
    }
    const std::regex listing(R"(\blisted\s*\(\s*(k\w+))");
    for (std::sregex_iterator it(text.begin(), text.end(), listing), end; it != end; ++it) {
        out.listed.insert((*it)[1].str());
    }
    out.read = true;
    return out;
}

bool in_range(rc::Kind kind, double v) {
    if (!std::isfinite(v) || v <= 0.0) return false;
    return kind != rc::Kind::Fraction || v <= 1.0;
}

} // namespace

int main(int argc, char** argv) {
    int failures = 0;
    const HeaderNames header = header_names();
    if (!header.read) ++failures;
    for (const auto& name : header.defined) {
        if (header.listed.count(name) != 0) continue;
        std::fprintf(stderr, "%s is defined but not listed in kRouteCostConstants\n", name.c_str());
        ++failures;
    }
    for (const auto& name : header.listed) {
        if (header.defined.count(name) != 0) continue;
        std::fprintf(stderr, "%s is listed but not defined as a Scalar or Constant\n", name.c_str());
        ++failures;
    }
    if (header.listed.size() != std::size(rc::kRouteCostConstants)) {
        std::fprintf(stderr, "the header lists %zu names, kRouteCostConstants holds %zu\n",
                     header.listed.size(), std::size(rc::kRouteCostConstants));
        ++failures;
    }
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
