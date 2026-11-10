# Gap map

`tools/bench/gap_map.py` puts every measured kernel time the repository already commits next to the
time a counted law allows, and prints the ratio. It is step 1 of the kernel roadmap (#619, decision
K4): the map names a *regime*; a kernel is built only when a counted law explains the current
kernel's time there. It needs no GPU and only the standard library.

```bash
python tools/bench/gap_map.py                       # Markdown to stdout
python tools/bench/gap_map.py --md gap.md --json gap.json --top 25
```

## The two laws

| Law | Counted from | Source |
|---|---|---|
| DRAM | weight elements x bits per weight, plus the activations in and the bf16 output, over the board's bandwidth | bits: the tables of [tensor-formats.md](tensor-formats.md) (parsed, not copied); bandwidth: `bus_bits x gbps / 8` of `src/core/data/gpu_facts.json` |
| tensor | 2 x rows x k x T flops over SMs x boost clock x flops per SM-clock | SMs and clock: `gpu_facts.json`; per-SM rate: the generation row of `src/core/generation.h` |

The law time is the larger of the two and the `binds` column says which. A SwiGLU call reads two
weight rows per output row; an Add call also reads its bf16 residual. Nothing else is a constant of
the tool: a row whose law needs a number the repository does not carry is kept and printed as
`no law` with the reason.

Rates that are fitted, not counted, are not used. The Volta and Turing bf16 rows of
`generation.h` (1536) sit above what the tensor cores physically issue (Volta: 8 cores x 64 FMA =
1024 flops per SM-clock), so Volta prefill, which has no int8 rate either, and Turing A16 linears
get `no law (fitted rate)` for the tensor side. Consumer Blackwell's `mma.sync` takes 8.5 cycles
per m16n8k16 against Ada's 4.0 (#619), so the gb20x bf16 row is about 2x high for those kernels: a
gb20x gap that the tensor law binds is a lower bound. Both belong in the generation facts (step 3),
not in a constant here.

## Measured sources

| Rows | From | Measured time |
|---|---|---|
| `linear` | `routes/*.json` entries with `evidence_us` | last column of `[T, default us, shipped us]` |
| `decode` | `tests/fixtures/choice/decode_measured.json` | one step = 1 / per-stream tokens a second, at C streams |
| `prefill` | `tests/fixtures/choice/prefill_measured.json` | prompt / prompt tokens a second |

Attribution rules, each a trap found while writing the tool:

- Evidence belongs to the board it was measured on. `ga106.json` keeps the RTX 3090's evidence
  (`tuned_on: ga102:82sm`, tracked in #634); priced on the 3060's bandwidth it ran at 0.6 of the
  DRAM law, which is impossible. The row takes the `tuned_on` board, or the table's own board when
  chip and SM count are the same. A chip and SM count that match several `gpu_facts` rows with
  different clock, bandwidth or L2 (`gv100:80sm`, `ad104:60sm`) is `no law: ambiguous board`.
- The RTX 5070 Ti's `route_as` evidence is the RTX 5080's (its table says so); identical evidence
  copied between tables is counted once, for the table with the shortest name. Each board has one
  name in the output, its `gpu_facts` name.
- Decode bytes follow `tools/fits/speed_model/fit_decode.py`: weights less the input embedding
  table and the speculation layers (the runs are `--spec none`), plus per stream the KV, the
  sliding-window ring (`min(context, window)` tokens) and the recurrent state read and written.
  MoE models and a KV or ring dtype missing from the fixture get no law.
- Whole-model prefill is 2 x `active_params` x prompt tokens. `active_params` already leaves out an
  untied input table and includes the output head, which a `max_tokens 1` prefill runs for the
  last token only; the head's size is not in the fixture, so the law is slightly high and the gap a
  lower bound, tied or untied. It is priced at the int8 rate where the generation has one; attention
  and norms are not in the law, so those gaps are upper bounds on the linears' own.
- A time under its law (gap below 1) is an attribution or byte-count bug, never a result: those rows
  are listed in their own section and left out of every statistic. Rows whose weight bytes fit the
  board's L2 are counted in the JSON (`l2_resident`); their DRAM law bounds a warm call from above.
- Rows whose law time is under `--min-law-us` (default 5) are counted and not ranked: a call that
  short is a launch, not a law (the event timer ticks at 1.024 us).

## Reading the output

Per card, format and T band (T1, 2-4, 5-8, 9-16, 17-64, 65-256, 257+): rows, rows scored, rows with
no law, median and worst gap, and the binding law. A gap of 1.0 is the law; 2.0 means the kernel
takes twice what the counted bytes or flops need. A band whose gap is not explained by a counted
law (launch cost, a missing term) is where a prediction has to come first (K4 rule 3).

## What the committed data says (master 35ebc031)

Route-table evidence exists for two boards: RTX 3090 and RTX 5080 (the 5070 Ti's copy is the 5080's).

- **Linears, RTX 3090**: median gap 1.7 at T1, 1.9 at T2-4, 2.0 at T5-8, 2.2 at T9-16, all DRAM
  bound. Worst: Q5_0 2688x3712 flat at 29.7 us over T11-15 (gap 4.0, a call-cost shape, not a
  bandwidth one); Q4_K / IQ4_XS 17408x5120 SwiGLU about 2.1-2.2 at T9-16.
- **Linears, RTX 5080**: 1.3-1.4 at T2-8, 3.0 at T17-64, 1.8 at T65-256, tensor-bound from about
  T40. Worst: the shape-owned Q5 1024x5120 at T96-128 (53-61 us against 8.9-11.9 us, gap 4.5-6.0),
  and Q4 4096x5120 at T40-48 (72-74 us, gap 4.1-4.8).
- **Whole decode step** (T = streams, short context): median 1.10 (4060 Ti) to 1.35 (2080 Ti);
  Quadro GV100 2.0, worst 3.2 at 8 streams, taken on a hot board. At one stream the DRAM share is
  0.92 (4060 Ti), 0.89 (4070 Ti), 0.88 (3060), 0.87 (5070 Ti), 0.84 (4090), 0.81 (2080 Ti), 0.77
  (5090), 0.57 (GV100).
- **Whole prefill**: median 3.3 (5070 Ti) to 3.9 (5090) of the int8 rate on Ampere/Ada/Blackwell
  and 6.3 on the 2080 Ti (worst 14 at 82K tokens, attention included). This is the "one third of
  int8 peak" and "Turing at 2.5x Ada's clocks" regime of #619. GV100 prefill has no law.

## No law yet

- MoE layers (17 layers in 5 route tables) and MoE models: needs the gating (gated or Relu2), the
  distinct experts a batch touches and the down format (step 11).
- A4 / `nvfp4` activations in route-table linears (161 rows): activation bytes with block scales
  are not counted.
- Volta prefill and Turing / Volta A16 tensor law (fitted rate), Volta int8 (no rate), Hopper and
  GB100 (lent rows).
- `u4zp*`, `q3g*` and any format missing from the format tables.
- Boards without route-table evidence (RTX 4090, 4070 Ti, 4060 Ti, 5090, 5060 Ti, 2080 Ti, GV100,
  A100): only the whole-model rows exist for them. The linear map needs their `ninferno_route_tune`
  evidence committed.
