#include "models/speculation/draft_target.h"

#include <cstdint>
#include <iostream>

namespace {

using ninferno::models::speculation::next_draft_depth;

int check(std::int32_t actual, std::int32_t expected, const char* message) {
    if (actual == expected) { return 0; }
    std::cerr << message << ": got " << actual << ", expected " << expected << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    // A fully accepted round deepens the next one by two, up to the deepest depth.
    failures += check(next_draft_depth(4, 4, 2, 8), 6, "full acceptance deepens by two");
    failures += check(next_draft_depth(7, 7, 2, 8), 8, "deepening stops at the deepest depth");
    failures += check(next_draft_depth(8, 8, 2, 8), 8, "the deepest depth holds");
    // Any rejected draft makes the next round one shallower, down to the shallowest depth.
    failures += check(next_draft_depth(6, 5, 2, 8), 5, "a rejected last draft shallows by one");
    failures += check(next_draft_depth(6, 0, 2, 8), 5, "no accepted draft shallows by one");
    failures += check(next_draft_depth(2, 1, 2, 8), 2, "shallowing stops at the shallowest depth");
    // A one-wide range never moves.
    failures += check(next_draft_depth(3, 3, 3, 3), 3, "a fixed depth ignores full acceptance");
    failures += check(next_draft_depth(3, 0, 3, 3), 3, "a fixed depth ignores rejection");
    return failures == 0 ? 0 : 1;
}
