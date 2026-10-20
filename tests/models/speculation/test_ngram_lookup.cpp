// Prompt lookup's draft search (models/speculation/ngram_round.h), on the host.

#include "models/speculation/ngram_round.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAILED: " << what << '\n'; }
    return ok ? 0 : 1;
}

using V = std::vector<std::int32_t>;

} // namespace

int main() {
    using ninferno::models::speculation::ngram_lookup;
    int failures = 0;
    // "1 2 3 4 5 ... 1 2 3": the suffix 1 2 3 recurred; its continuation was 4 5 6.
    const V history{1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3};
    failures += check(ngram_lookup(history, 2, 4, 3) == V{4, 5, 6}, "continuation of the suffix");
    failures += check(ngram_lookup(history, 2, 4, 2) == V{4, 5}, "the limit caps the drafts");
    // The longest suffix wins: "7 2 3" occurs once with 8 after it, "2 3" more recently with 5.
    const V longest{7, 2, 3, 8, 2, 3, 5, 0, 7, 2, 3};
    failures += check(ngram_lookup(longest, 2, 3, 1) == V{8}, "longest match first");
    failures += check(ngram_lookup(longest, 2, 2, 1) == V{5}, "most recent occurrence");
    // No recurring suffix of at least min_n tokens: no drafts.
    failures += check(ngram_lookup(V{1, 2, 3, 4, 5}, 2, 4, 8).empty(), "no match");
    failures += check(ngram_lookup(V{1, 2, 3, 9, 3}, 2, 4, 8).empty(), "a single token is too short");
    // A run repeats itself: the continuation may overlap the suffix.
    failures += check(ngram_lookup(V{5, 5, 5, 5}, 2, 3, 4) == V{5}, "overlapping repeat");
    failures += check(ngram_lookup(V{}, 2, 4, 4).empty() && ngram_lookup(V{1}, 2, 4, 4).empty(),
                      "short histories");
    if (failures == 0) { std::cout << "ngram lookup: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
