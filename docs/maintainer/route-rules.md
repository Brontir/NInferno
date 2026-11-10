# Route rules: the literals of the generic path

`src/ops/generic/route_rules.h` declares one descriptor for each literal threshold of the generic
path's route selection that this list covers (the next section says what it does not): the
thresholds `rules_r8` compares a call against (the verify branch's two-token edge, the Blackwell
branch's gate and its rules, the Turing and Volta rules), `w4a8_ws_wins` and its neighbour,
`kVecRuleMaxTokens` and the R8 and vector verify edges, the speed profile's width bands
(`kRmmaMidMinTokens`, `kTiledSpeedMinTokens`, `kStagedSpeedMinTokens`, `kW4a8WideMinTokens`,
`kSpeedFill`), the literals inside `heuristic_rmma`, `heuristic_w4a8`, `heuristic_w4a4`,
`staged_speed_route`, `tiled_speed_pays`, `nvfp4_tiled_route`, `auto_route`'s half-the-L2 rule for
NVFP4 and `spread_smem` with its table of measured boards, the verify widths (`kR8VerifyTokens`,
`kMmqMinTokens`, `kMmqVerifyMinTokens`), Volta's routes and the byte thresholds of the register
MMA. Each entry says what the rule is (its literal, as written in the source), what it rests on
(the boards and cards it was measured on, as device class ids and products) and how much margin
the record gives (the tolerance). A number that was fitted to measurements says `FITTED` in its
tolerance and names the boards; where the source gives no record the entry says `unknown` or
`none recorded`, it never invents one. The kernel hash is empty, because the repository records
one digest per kernel-facts file, not one per kernel, and no rule records the tree it was timed
on.

It is a record, not a mechanism. The code keeps its own literals and nothing reads the descriptor
at run time. `ninferno_route_rules_test` (`--list` prints the table) holds the two to each other:

- every literal text must stand in its source file, outside comments and strings, exactly as many
  times as the descriptor says, whole (a number is not the prefix of a longer number, a name not
  the tail of a longer one); a comparison's literal carries its action where the action matters;
- a literal that defines a constant is the whole definition (to its `;`; an array with every
  element), defines the descriptor's name, is that name's only definition in its file (the test
  reads text, not the preprocessor, so a second definition under `#if` fails), and its number is
  computed as C++ computes it (integers stay integers; `0b`, octal, `'` separators and qualified
  names are read; what the evaluator does not implement is refused, never skipped) and must equal
  the descriptor's `value`; a condition's `value` is 0 or a number its literal writes;
- the boards, cards and tolerance fields are never empty;
- every definition under `src/ops/generic` (subdirectories; `.h`, `.hpp`, `.cpp`, `.cu`, `.cuh`)
  of a name `kX` that says it bounds something (`Min`, `Max`, `Tokens`, `Fill`, `Distance`,
  `Values`, `Bytes`, `Gbps`, `Margin`, `Slack`, `Columns`, `Rows`, `Limit`, `Measured`,
  `Threshold`), as a `const`, `constexpr`, `static` or `inline` variable or array (initialised by
  `=`, braces or parentheses, later declarators of one declaration included) or a `#define`, has a
  descriptor or an entry of the test's list of non-rules, keyed by file and name, with the reason
  and the number of definitions the file holds, so a second definition of a listed name fails.

## What is not covered

The guard is a text scan, and the list is not every literal comparison in `generic_gemm.cpp`. Left
out knowingly:

- names that are not camel-case `kX` constants or carry none of the words above (snake_case
  variables, `kMmqPrefillSlice` in `generic_ggml_internal.h`), enumerators and template
  parameters: found only where a descriptor's literal names them;
- thresholds written as a bare literal in an expression that no descriptor names. Known ones: the
  decode block of `heuristic_rmma` (`tokens <= 8`, `ctas <= 2 * sms`, `ctas > 2 * sms`), which
  shares one comment with the K-split rule; the `tokens >= 2` and `max_tokens >= 2` edges in
  `generic_workspace.cpp`, which mirror the verify edge described for `r8_verify_tunable` (its
  descriptor says so); the shape tables' own bands, which are data (`routes/`); other literal
  `17`s outside the generic path's selection (the shape-owned linear plans, the Qwen prefill
  planner's own `kPrefillMinTokens`) and `kVoltaCutlassMinTokens` in `src/product/choice` (a
  model's copy of Volta's rule): they are not defined from the one constant and equal it today;
- the preprocessor: two definitions of a name under `#if` / `#else` count as two definitions;
- the text scan's own limits: a macro that expands to a comment or to a declaration is not seen,
  and a declaration whose type text holds a `=`, `;`, `{`, `(` or `[` other than `decltype(...)`
  is read from the first of them.

The width of the prefill band is one definition: `kPrefillPinMinTokens` in `ops/route_pins.h`
(the widest speculative verify, `kDecodePinMaxTokens` = 16, plus one), which is below the generic
path in the include order. `kPrefillBandMinTokens` (`generic_gemm.h`), `kQ8PrefillA8Tokens`,
`kVoltaCutlassMinTokens` and `route_table.cpp`'s `kPrefillMinTokens` are defined from it because
their comments name the same width; the tuner's band grid and the selector facts' token grid start
at it, held by `static_assert`s, and the tune tool and the workspace plan read it by name. Other
constants equal to 16 or 8 (`kDenseTuneMaxTokens`, `kPdlPrefetchMaxTokens`, `kMmqMaxTokens`) are
separate facts that happen to match, and each descriptor says so.

## The rule that follows

A literal stays until its derivation PR replaces it with a law: a rule over the card's facts
(SMs, DRAM and L2 rates, the generation's tensor rates), checked against the boards the descriptor
names and held out on the others (`test_selector_golden`, the regret tools of #614). When that PR
lands it deletes the descriptor with the literal; it does not edit the numbers. A literal is never
retuned in place on one board's timings: a new measurement is a new descriptor field (more boards,
a tighter tolerance) or a derivation, not a new number. A descriptor whose boards say `unknown` is
the first candidate for a derivation or a re-measurement, because nothing can be held out against
it, and a `FITTED` one is a fit, not a derivation.

Where the provenance of a rule was found: the comments at its use in `generic_gemm.cpp` and
`generic_gemm.h` (quoted in the tolerance where they state a margin), `decisions.md` 58 for the
verify widths, and `deterministic-decisions.md` for the rules that rest on few boards (the R8 rule's
SM-count edge, placed between the two measured boards, 84 and 170 SMs). The history before the
0.1.0 squash is not in the repository.
