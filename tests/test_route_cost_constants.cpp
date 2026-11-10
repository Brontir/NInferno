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

// The header's text: the constants it declares (any `const` or `constexpr` declaration of a name
// kX, whatever its type is spelled as, over any spacing or line breaks; the descriptor itself
// aside) and the ones its descriptor lists (`listed(kX, ...)` inside kRouteCostConstants), as two
// sets of names. Each declared constant must be listed and each listed one declared: a missing
// entry is named, not hidden behind a count. The header names every constant kX for this.
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
    // Comments may name constants too; drop them first (block comments, then line comments).
    text = std::regex_replace(text, std::regex(R"(/\*[\s\S]*?\*/)"), "");
    text = std::regex_replace(text, std::regex(R"(//[^\n]*)"), "");
    const std::regex definition(R"(\b(?:constexpr|const)\b[^;{}=()]*?\b(k[A-Z]\w*)\s*[{(=\[])");
    for (std::sregex_iterator it(text.begin(), text.end(), definition), end; it != end; ++it) {
        if ((*it)[1] != "kRouteCostConstants" && (*it)[1] != "kPendingFits") {
            out.defined.insert((*it)[1].str());
        }
    }
    const std::string::size_type start = text.find("kRouteCostConstants[]");
    if (start == std::string::npos) {
        std::fprintf(stderr, "no kRouteCostConstants[] in %s\n", header.string().c_str());
        return out;
    }
    const std::regex listing(R"(\blisted\s*\(\s*(k\w+))");
    for (std::sregex_iterator
             it(text.begin() + static_cast<std::ptrdiff_t>(start), text.end(), listing),
         end;
         it != end; ++it) {
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
    int failures             = 0;
    const HeaderNames header = header_names();
    if (!header.read) ++failures;
    for (const auto& name : header.defined) {
        if (header.listed.count(name) != 0) continue;
        std::fprintf(stderr, "%s is defined but not listed in kRouteCostConstants\n", name.c_str());
        ++failures;
    }
    for (const auto& name : header.listed) {
        if (header.defined.count(name) != 0) continue;
        std::fprintf(stderr, "%s is listed but not declared in the header\n", name.c_str());
        ++failures;
    }
    if (header.listed.size() != std::size(rc::kRouteCostConstants)) {
        std::fprintf(stderr, "the header lists %zu names, kRouteCostConstants holds %zu\n",
                     header.listed.size(), std::size(rc::kRouteCostConstants));
        ++failures;
    }
    for (const auto& pending : rc::kPendingFits) {
        if (pending.from != rc::Provenance::Fitted || pending.value.has_value() ||
            pending.name.empty()) {
            std::fprintf(
                stderr,
                "pending fit '%.*s' must be Fitted, named and empty (a calibrated one moves "
                "into kRouteCostConstants with its Holdout)\n",
                static_cast<int>(pending.name.size()), pending.name.data());
            ++failures;
        }
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
