// The literal rules of the generic path (ops/generic/route_rules.h), held to the sources they
// describe: every descriptor has a unique name, a literal that is in its file (whitespace aside), a
// recorded boards / cards / tolerance field ("unknown" or "none recorded" where the record is
// silent, never empty) and an empty kernel hash (none is recorded); a named constant's value is the
// one the source defines; and every constant of generic_gemm.h's widths block has a descriptor, so a
// rule cannot change, move or appear without its record. `--list` prints them. No GPU.

#include "ops/generic/route_rules.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <string>
#include <string_view>

namespace rr = ninferno::ops::detail::generic::route_rules;

namespace {

std::string read(const std::string& relative) {
    const std::filesystem::path path =
        std::filesystem::path(NINFERNO_SOURCE_DIR) / "src/ops/generic" / relative;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", path.string().c_str());
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Runs of whitespace as one space: clang-format may realign a definition's `=`.
std::string squeezed(const std::string& text) {
    return std::regex_replace(text, std::regex(R"(\s+)"), " ");
}

} // namespace

int main(int argc, char** argv) {
    int failures = 0;
    const auto fail = [&](const std::string& message) {
        std::fprintf(stderr, "%s\n", message.c_str());
        ++failures;
    };
    std::set<std::string> names;
    std::set<std::string> sources;
    for (const auto& rule : rr::kRouteRules) {
        const std::string name(rule.name);
        if (name.empty() || !names.insert(name).second) {
            fail("rule name '" + name + "' empty or listed twice");
        }
        if (rule.literal.empty()) fail(name + ": no literal");
        for (const auto field : {rule.boards, rule.cards, rule.tolerance}) {
            if (field.empty()) fail(name + ": a boards / cards / tolerance field is empty");
        }
        if (!rule.kernel_hash.empty()) {
            fail(name + ": a kernel hash is set, but none is recorded anywhere to check it against");
        }
        const std::string file = squeezed(read(std::string(rule.file)));
        if (file.empty()) {
            fail(name + ": cannot read " + std::string(rule.file));
            continue;
        }
        sources.insert(std::string(rule.file));
        const std::string literal = squeezed(std::string(rule.literal));
        if (file.find(literal) == std::string::npos) {
            fail(name + ": the literal '" + std::string(rule.literal) + "' is not in " +
                 std::string(rule.file));
        }
        // A named constant: the number the source defines is the descriptor's.
        static const std::regex definition(R"(constexpr [^=;]*\b(k[A-Z]\w*) = ([0-9.]+);)");
        std::smatch match;
        if (std::regex_search(literal, match, definition)) {
            if (match[1].str() != name) {
                fail(name + ": its literal defines " + match[1].str());
            }
            if (std::fabs(std::stod(match[2].str()) - rule.value) > 1e-12) {
                fail(name + ": the source says " + match[2].str() + ", the descriptor " +
                     std::to_string(rule.value));
            }
        }
    }
    // The widths block of generic_gemm.h: every constant there is a described rule.
    const std::string header = read("generic_gemm.h");
    const auto begin         = header.find("The widths where the speed profile's heuristic choices");
    const auto end           = header.find("enum class Epilogue", begin);
    if (begin == std::string::npos || end == std::string::npos) {
        fail("generic_gemm.h: the widths block is not where this test expects it");
    } else {
        const std::string block = header.substr(begin, end - begin);
        static const std::regex constant(R"(constexpr [^=;]*\b(k[A-Z]\w*)\s*=)");
        std::size_t found = 0;
        for (std::sregex_iterator it(block.begin(), block.end(), constant), stop; it != stop; ++it) {
            ++found;
            if (names.count((*it)[1].str()) == 0) {
                fail((*it)[1].str() + " is a constant of generic_gemm.h's widths block without a "
                                      "descriptor in kRouteRules");
            }
        }
        if (found == 0) fail("no constants found in the widths block");
    }
    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (const auto& rule : rr::kRouteRules) {
            std::printf("%-44.*s %-10g %.*s | %.*s\n", static_cast<int>(rule.name.size()),
                        rule.name.data(), rule.value, static_cast<int>(rule.cards.size()),
                        rule.cards.data(), static_cast<int>(rule.tolerance.size()),
                        rule.tolerance.data());
        }
    }
    std::printf("route rules: %zu described, kernel hash recorded for none\n",
                std::size(rr::kRouteRules));
    return failures == 0 ? 0 : 1;
}
