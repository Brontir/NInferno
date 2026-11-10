# Route rules: the literals of the generic path

`src/ops/generic/route_rules.h` declares one descriptor for each literal rule of the generic path's
route selection: the thresholds `rules_r8` compares a call against, `w4a8_ws_wins` and its
neighbour, `kVecRuleMaxTokens`, the speed profile's width bands (`kRmmaMidMinTokens`,
`kTiledSpeedMinTokens`, `kStagedSpeedMinTokens`, `kW4a8WideMinTokens`, `kSpeedFill`), the verify
widths (`kR8VerifyTokens`, `kMmqMinTokens`, `kMmqVerifyMinTokens`), Volta's routes and the byte
thresholds of the register MMA. Each entry says what the rule is (its literal, as written in the
source), what it rests on (the boards and cards it was measured on) and how much margin the record
gives (the tolerance); the kernel hash is empty, because the repository records one digest per
kernel-facts file, not one per kernel, and no rule records the tree it was timed on.

It is a record, not a mechanism. The code keeps its own literals and nothing reads the descriptor
at run time. `ninferno_route_rules_test` (`--list` prints the table) holds the two to each other:
every literal text must stand in its source file outside comments exactly as many times as the
descriptor says, a named constant's number must equal the source's, the boards, cards and tolerance
fields are never empty (`unknown` and `none recorded` say that the record is silent, they are not
guesses), and every `const` or `constexpr` definition or `#define` under `src/ops/generic`
(subdirectories and `.cu`/`.cuh` included) whose name says it bounds something (`Min`, `Max`,
`Tokens`, `Fill`, `Distance`, `Values`, `Bytes`, `Gbps`, `Margin`, `Slack`, `Columns`, `Rows`,
`Limit`) has a descriptor, or an entry of the test's list of non-rules with its reason (tile shapes,
layout facts, sizes). That is the extent of the guard: an enumerator, a constant whose name has none
of those words, and a threshold written as a bare literal in an expression are not found by the
scan; the width of the prefill band, which stood as a bare `17` in sixteen comparisons, is now one
named constant (`kPrefillBandMinTokens`) with a descriptor. A rule therefore cannot change, move
or appear as a named constant without its record changing in the same diff.

## The rule that follows

A literal stays until its derivation PR replaces it with a law: a rule over the card's facts
(SMs, DRAM and L2 rates, the generation's tensor rates), checked against the boards the descriptor
names and held out on the others (`test_selector_golden`, the regret tools of #614). When that PR
lands it deletes the descriptor with the literal; it does not edit the numbers. A literal is never
retuned in place on one board's timings: a new measurement is a new descriptor field (more boards,
a tighter tolerance) or a derivation, not a new number. A descriptor whose boards say `unknown` is
the first candidate for a derivation or a re-measurement, because nothing can be held out against
it.

Where the provenance of a rule was found: the comments at its use in `generic_gemm.cpp` and
`generic_gemm.h` (quoted in the tolerance where they state a margin), `decisions.md` 58 for the
verify widths, and `deterministic-decisions.md` for the rules that rest on few boards (the R8 rule's
SM-count edge, placed between the two measured boards, 84 and 170 SMs). The history before the
0.1.0 squash is not in the repository.
