# Route rules: the literals of the generic path

`src/ops/generic/route_rules.h` declares one descriptor for each literal threshold of the generic
path's route selection that this list covers (the next section says what it does not): the
thresholds `rules_r8` compares a call against (the verify branch's two-token edge, the Blackwell
branch's gate and its rules, the Turing and Volta rules), `w4a8_ws_wins` and its neighbour,
`kVecRuleMaxTokens` and the R8 and vector verify edges, the speed profile's width bands
(`kRmmaMidMinTokens`, `kTiledSpeedMinTokens`, `kStagedSpeedMinTokens`, `kW4a8WideMinTokens`,
`kSpeedFill`), the literals inside `heuristic_rmma` (its speed ladder included),
`rmma_k_split_fit`, `rmma_wide_rule`, `heuristic_w4a8`, `heuristic_w4a4`, `staged_speed_route`,
`tiled_speed_pays`, `nvfp4_tiled_route`, the half-the-L2 rule for NVFP4 (`nvfp4_half_l2` in
`route_authorities.cpp`, with its weight-bytes factor) and, in the same file, the R8 table's
one-token gate, the narrow register-MMA fallback tiles and the source of the lane count,
`spread_smem` with its table of measured boards, the verify widths (`kR8VerifyTokens`,
`kMmqMinTokens`, `kMmqVerifyMinTokens`), Volta's routes and the byte thresholds of the register
MMA. Each entry says what the rule is (its literal, as written in the source), what it rests on
(a basis class and the boards and cards it was measured on, as device class ids and products) and
how much margin the record gives (the tolerance). Where the source gives no record the entry says
`unknown` or `none recorded`, it never invents one. The kernel hash is empty, because the
repository records one digest per kernel-facts file, not one per kernel, and no rule records the
tree it was timed on.

The basis is a typed field (`Basis`, as `route_cost_constants.h` types its `Provenance`), read from
the source comments, `decisions.md` and `deterministic-decisions.md` by a person, and `Unknown`
where they do not say. `Structural`: follows from a kernel's shape, a contract, the hardware's
layout or another constant, not from a timing. `Measured`: a timing or product run in the record
names the value or the margin it was held to, on the boards the entry names.
`PlacedBetweenMeasured`: the record has measured points on both sides and the value was put
between them, so the line itself has no measurement. `Fitted`: chosen to make a rule match
timings of several boards, not derived from a fact; the entry names those boards.
`ScaledFromOneBoard`: one board's measured margin carried to other boards by a stated model, no
second board measured (`kVoltaCopyMinDramGbps`). `Unknown`: the record does not say what the edge
rests on (the rule's action may still be measured); a choice nobody timed is `Unknown`, not
`Structural`. The class is judgement and the test cannot check it is right: where an edge is only
the widest speculative verify (16 tokens) while the timings sit inside it, the tolerance says so. A
`Measured` rule may rest on one board and still apply to every board: 19 of the 34 `Measured`
entries name a single card (count them from `--list`), and for example `kR8MaxTokens` and
`kR8DecodeTokens` (RTX 5080), the single-sequence rule of mmq verify (RTX 5070 Ti), the
narrow-through-16 rule (RTX 5090) and `w4a8_ws_wins: ampere` (RTX 3090, taken for sm_80 and sm_86)
decide on every card. The class says what the timing was, not how far the rule reaches; the
`boards` field is where to look. A step that holds fitted or placed constants out of a derivation
reads this field, not the prose; `--list` prints it and the test prints the count per class. Where a descriptor only says where a
value comes from (the engine's round width at the call, the Ask's lane count handed on), it is
`Structural` with unknown boards and its tolerance names the entry that holds the timing; several
descriptors then share one record, and none of them is a second measurement.

It is a record, not a mechanism. The code keeps its own literals and nothing reads the descriptor
at run time. `ninferno_route_rules_test` (`--list` prints the table) holds the two to each other:

- every descriptor has a basis class (the field has no default, so a missing one does not compile);
  a `Measured` one names a board (a class id such as `gb202:170sm`); a `Fitted`,
  `PlacedBetweenMeasured` or `ScaledFromOneBoard` one names its boards (class ids) or says none are
  recorded; a `Structural` constant that is a bare number in the code is defined from another
  constant, is named in a `static_assert` of the scanned sources, or is in the test's
  `structural_roots` list with its reason (a comment that calls a number structural is not enough);
- every literal text must stand in its source file, outside comments and outside the contents of
  string and character literals (the reader blanks them, quotes kept, and has a self-check),
  exactly as many times as the descriptor says, whole (a number is not the prefix of a longer
  number, a name not the tail of a longer one); whitespace between tokens does not matter (a
  space between two identifier characters stays one), nor a trailing comma before a `}`; a
  literal that opens with `if (` is not found behind an `else`; a comparison's literal carries its
  action where the action matters, and where a rule is one statement of a cascade (`rules_r8`,
  `w4a8_ws_wins`) the literal carries the end of the statement before it, so moving the statement
  or putting a case label above it breaks the match;
- a literal that defines a constant is the whole definition (to its `;`; an array with every
  element), defines the descriptor's name, is that name's only definition in its file (the test
  reads text, not the preprocessor, so a second definition under `#if` fails), and its number is
  computed by the test's own evaluator (integers stay integers; `0b`, octal, `'` separators and
  qualified names are read; what the evaluator does not implement is refused, never skipped; the
  conversion to the constant's declared type is not applied, and no constant in the sources needs
  it) and
  must equal the descriptor's `value`; a condition's `value` is 0 or a number its literal writes;
- the boards, cards and tolerance fields are never empty;
- every definition under `src/ops/generic` (subdirectories; `.h`, `.hpp`, `.hh`, `.cpp`, `.cc`,
  `.cxx`, `.inc`, `.inl`, `.ipp`, `.tpp`, `.cu`, `.cuh`), in the files directly in `src/ops` (route
  pins, route table, prefill routes, a selector header put there) and in
  `src/core/measured_boards.h` of a name `kX` that says it bounds something (`Min`, `Max`,
  `Tokens`, `Fill`, `Distance`, `Values`, `Bytes`, `Gbps`, `Margin`, `Slack`, `Columns`, `Rows`,
  `Limit`, `Measured`, `Threshold`), as a `const`, `constexpr`, `static` or `inline` variable or
  array (initialised by `=`, braces or parentheses, later declarators of one declaration included)
  or a `#define`, has a descriptor or an entry of the test's list of non-rules, keyed by file and
  name, with the reason and the number of definitions the file holds, so a second definition of a
  listed name fails.

## What is not covered

The guard is a text scan, and the list is not every literal comparison in `generic_gemm.cpp` or
in `route_authorities.cpp` (the authority cascade: `nvfp4_half_l2`, `r8_schedule`,
`rmma_narrow_fallback` and the rest). Left out knowingly:

- the comparators at the uses of a named constant: the constant's value is held, but whether a
  site writes `>=` or `<=` against it, or a bare 17 or 33 in its place, is not: for example the
  prefill-band sites in `route_authorities.cpp` and `route_authorities.h` (`kPrefillBandMinTokens`,
  `kPrefillPinMinTokens`), `<= measured::kCarryDistance` and `kStagedSpeedMinTokens` there;
- the order of the three authority arrays in `route_authorities.cpp` (`kSelectAuthorities`,
  `kAutoAuthorities`, `kRuleAuthorities`): the first authority that answers wins, so swapping two
  entries changes decisions, and only the selector goldens (`test_selector_golden`) hold the
  order, not this record;
- a rule switched off by an early return, and a new bare-literal gate added inside an authority;
- what an authority does with a route after it has built it (whether it admits it, which of two
  it returns) and the token argument it hands to a rule: a change there moves decisions that only
  the selector goldens hold;
- the format membership of the measured rules: `heuristic_w4a8: 65..127 formats` holds the
  `tokens < kW4a8WideMinTokens` edge, not the `wide_format` switch in `generic_gemm.cpp` that says
  which formats take the wider tile, and the per-format flags in `format_traits.h` (`r8_verify`,
  `mmq_verify`, `light_step`) are not held at all: adding or removing a format in either passes
  the test;
- the bare kernel counts of the cost model in `generic_cost.cpp` (the 384, 128 and 256 threads a
  CTA, the stage factors 3 and 2): they are not constants the guard names, and the cost model's
  constants registered in `route_cost_constants.h` are not these;
- names that are not camel-case `kX` constants or carry none of the words above (snake_case
  variables, `kMmqPrefillSlice` in `generic_ggml_internal.h`), enumerators and template
  parameters: found only where a descriptor's literal names them;
- thresholds written as a bare literal in an expression that no descriptor names. Known ones: the
  decode block of `heuristic_rmma` (`tokens <= 8`, `ctas <= 2 * sms`, `ctas > 2 * sms`; the
  one-wave edge `ctas <= sms` at its two sites is described, and `split == 4` / `split == 8`
  dispatch on the K split's own result and are not thresholds), which shares one comment with the
  K-split rule; the `tokens >= 2` and `max_tokens >= 2` edges in `generic_workspace.cpp`, which
  mirror the verify edge described for `r8_verify_tunable` (its descriptor says so); the shape
  tables' own bands, which are data (`routes/`); other literal `17`s outside the generic path's
  selection (the bare prefill-band 17s in `src/ops/moe/moe.cpp`, `src/ops/wrapper/linear_swiglu.cpp`
  and `gdn_input_proj.cpp`, `src/ops/sparse_moe/prefill/sparse_moe_prefill.h` and the NVFP4 and FP8
  input plans, the shape-owned linear plans, the Qwen prefill planner's own `kPrefillMinTokens`,
  `PREFILL_MIN` in `tools/bench/mmq_transfer_audit.py`) and `kVoltaCutlassMinTokens` in
  `src/product/choice` (a model's copy of Volta's rule): they are not defined from the one
  constant and equal it today;
- values that are data or admission, not a selection threshold: the per-board tables in
  `r8_decode_table.h` (data, keyed by board), `ggml_mmq::kRowMultiple` and `kKMultiple` (the
  alignment the ggml matrix kernel requires of N and K: a call that fails it cannot take the
  route), and `generic_cost.cpp`'s `s.n_tiles == 8` (the cost model skips the eight-tile
  schedules where the BF16 MMA is emulated; the cost model's own constants are registered in
  `route_cost_constants.h`);
- the capability predicates in `src/core/generation.h` (`turing_cc`, `ampere_cc`, `ada_cc`,
  `gb20x_cc`, `consumer_blackwell_major`): they name which cards a rule applies to (a reach, not a
  threshold on a quantity), and what the rule was measured on is said by the descriptor of each
  rule that calls them (for example `w4a8_ws_wins: ampere`: measured on the RTX 3090 only, taken
  for sm_80 and sm_86 alike). Widening one of them moves a rule onto cards nobody measured, and
  the test does not see it;
- a launch bound in a form the scan cannot place: the only two found are described (`kVecWarps`,
  `kVecBoundedBlocks`); the conservative rule that decides when the bounded twin is taken
  (`kVecHasBoundedTwin` and the launch's round count in `ggml_vec.cu`) is not described;
- the preprocessor: two definitions of a name under `#if` / `#else` count as two definitions;
- the text scan's own limits: a macro that expands to a comment or to a declaration is not seen,
  and a declaration whose type text holds a `=`, `;`, `{`, `(` or `[` other than `decltype(...)`
  is read from the first of them;
- scope: constants are looked for in `src/ops/generic`, in the files directly in `src/ops` and in
  `src/core/measured_boards.h`. A selector rule put in another directory (`src/ops/<op>/`,
  `src/core`, `src/product`) is not scanned, and the walk does not follow an `#include` from a
  scanned file into one.

What a described literal's guard does not catch (each was run against the test and passes it, or
follows from how the test reads text):

- arithmetic or logic added at a token boundary of a condition literal: `&& false &&` after a
  literal that ends in `&&`, `>= kSpeedFill * 2`, a factor written in front of a literal that
  starts mid-expression;
- a use of a named constant: the constants' values are held, and the use sites are held only where
  a descriptor's literal contains the use (`volta_copy_pays`, the uses of `kRmmaMidMinTokens` and
  `kSpeedFill` inside the `heuristic_rmma` literals). A comparison changed at any other use (`>=` to
  `<=`), a use scaled or replaced by a bare literal, or a wrapper function of the constant
  changed, passes. A variable shadowed or changed before a described condition passes;
- a described statement moved, or an enclosing gate changed, where the descriptor's literal does
  not carry the neighbour that pins it: the cascades named above do, the rest of
  `heuristic_rmma`, `heuristic_w4a8` and `staged_speed_route` do not (a moved `if` or an inverted
  enclosing `if` there passes);
- `if (false)` before, or `#if 0` around, a described statement;
- a threshold in a form the scan does not read: a function (`constexpr ... kNewFnMaxTokens()
  { return 96; }`, the form `vec_rule_max_tokens()` already has), a function-like macro, a default
  argument, `constinit`, a global without `const` / `constexpr` / `static` / `inline`, a member's
  default initialiser;
- a descriptor deleted together with its literal: the doc says a derivation PR does that, and the
  test only requires that the scan finds enough constants;
- the class of a descriptor: nothing checks that `Measured` is true, only that a board is named.

A literal is also frozen against harmless edits it cannot tell from changes: a renamed local inside
a described condition, `8` written `0x8`, and one more layout `kMin` in a file whose count is
listed all fail the test and are updated with the descriptor.

The width of the prefill band is one definition: `kPrefillPinMinTokens` in `ops/route_pins.h`
(the widest speculative verify, `kDecodePinMaxTokens` = 16, plus one), which is below the generic
path in the include order. `kPrefillBandMinTokens` (`generic_gemm.h`), `kQ8PrefillA8Tokens`,
`kVoltaCutlassMinTokens` and `route_table.cpp`'s `kPrefillMinTokens` are defined from it because
their comments name the same width; the tuner's band grid and the selector facts' token grid start
at it, held by `static_assert`s, and the tune tool and the workspace plan read it by name. Other
constants equal to 16 or 8 (`kDenseTuneMaxTokens`, `kPdlPrefetchMaxTokens`, `kMmqMaxTokens`) are
not defined from it, and each descriptor says so: `kMmqMaxTokens`' comment names its own timings,
the comments of the other two name none (`kDenseTuneMaxTokens` and `kPdlPrefetchMaxTokens` call
the same decode and speculative-verify band, but nothing ties them to it).

## The rule that follows

A literal stays until its derivation PR replaces it with a law: a rule over the card's facts
(SMs, DRAM and L2 rates, the generation's tensor rates), checked against the boards the descriptor
names and held out on the others (`test_selector_golden`, the regret tools of #614). When that PR
lands it deletes the descriptor with the literal; it does not edit the numbers. A literal is never
retuned in place on one board's timings: a new measurement is a new descriptor field (more boards,
a tighter tolerance) or a derivation, not a new number. A descriptor whose boards say `unknown` is
the first candidate for a derivation or a re-measurement, because nothing can be held out against
it, and a `Fitted` one is a fit, not a derivation; a `PlacedBetweenMeasured` one is the same
question about a line nobody measured.

Where the provenance of a rule was found: the comments at its use in `generic_gemm.cpp`,
`generic_gemm.h` and `route_authorities.cpp` (quoted in the tolerance where they state a margin),
`decisions.md` 58 for the verify widths, and `deterministic-decisions.md` for the rules that rest
on few boards (the R8 rule's SM-count edge, placed between the two measured boards, 84 and 170
SMs). The history before the 0.1.0 squash is not in the repository.
