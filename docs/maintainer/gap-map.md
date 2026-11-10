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
weight rows per output row. Nothing else is a constant of the tool: a row whose law needs a number
the repository does not carry is kept and printed as `no law` with the reason.

The tensor rate is the cost model's lent, analytic one. #619 records that consumer Blackwell's
`mma.sync` takes 8.5 cycles per m16n8k16 against Ada's 4.0, so the `gb20x` bf16 row is about 2x
high for those kernels: a gb20x gap that the tensor law binds is a lower bound. That is a finding
the map exists to make visible, and it belongs in the generation facts (step 3), not in a fitted
constant here.

## Measured sources

| Rows | From | Measured time |
|---|---|---|
| `linear` | `routes/*.json` entries with `evidence_us` | last column of `[T, default us, shipped us]` |
| `decode` | `tests/fixtures/choice/decode_measured.json` | one step = 1 / per-stream tokens a second, at C streams |
| `prefill` | `tests/fixtures/choice/prefill_measured.json` | prompt / prompt tokens a second |

Attribution rules, each a trap found while writing the tool:

- Evidence belongs to the board it was measured on. `ga106.json` keeps the RTX 3090's evidence
  (`tuned_on: ga102:82sm`); priced on the 3060's bandwidth it ran at 0.6 of the DRAM law, which is
  impossible. The row now takes the `tuned_on` board.
- The RTX 5070 Ti's `route_as` evidence is the RTX 5080's (its table says so); identical evidence
  copied between tables is counted once, for the table with the shortest name.
- Decode bytes: weights less the input embedding table, plus per stream the KV and recurrent state
  read. MoE models, sliding-window models at long context and `nvfp4` (A4 activations) get no law.
- Whole-model prefill is priced at the int8 rate where the generation has one, else bf16; attention
  and norms are not in the law, so those gaps are upper bounds on the linears' own.
- Rows whose law time is under `--min-law-us` (default 5) are counted and not ranked: a call that
  short is a launch, not a law.

## Reading the output

Per card, format and T band (T1, 2-4, 5-8, 9-16, 17-64, 65-256, 257+): rows, rows scored, rows with
no law, median and worst gap, and the binding law. A gap of 1.0 is the law; 2.0 means the kernel
takes twice what the counted bytes or flops need. A band whose gap is not explained by a counted
law (launch cost, a missing term) is where a prediction has to come first (K4 rule 3).

## What the committed data says (master 35ebc031)

Route-table evidence exists for three boards only: RTX 3090, RTX 5080 (and the 5070 Ti's copy).

- **Linears, RTX 3090**: median gap 1.7 at T1, 1.9 at T2-4, 2.0 at T5-8, 2.2 at T9-16, all DRAM
  bound. Worst: Q4_K / IQ4_XS 17408x5120 SwiGLU near 2.2 at T9-16; Q5_0 2688x3712 constant at 29.7
  us over T11-15 (gap 3.9-4.0, a call-cost shape, not a bandwidth one).
- **Linears, RTX 5080**: 1.3-1.4 at T2-8, 2.9 at T17-64, 1.8 at T65-256. The worst rows are the
  shape-owned Q5 1024x5120 call at T96-128 (53-61 us against 11.5-11.9, gap 4.5-5.3).
- **Whole decode step** (T = streams, short context): 1.09 (4060 Ti) to 1.31 (2080 Ti) median;
  Quadro GV100 1.92, worst 3.4 at 8 streams, and its rows were taken on a hot board. At one stream
  the DRAM share is 0.95 (4060 Ti), 0.92 (4070 Ti), 0.90 (3060), 0.89 (5070 Ti), 0.86 (4090),
  0.83 (2080 Ti), 0.80 (5090).
- **Whole prefill**: median 3.7 (5070 Ti) to 4.2 (5090) of the int8 rate on Ampere/Ada/Blackwell,
  5.7 on GV100 and 6.8 on the 2080 Ti (worst 14 at 82K tokens, attention included). This is the
  "one third of int8 peak" and "Turing at 2.5x Ada's clocks" regime of #619.

## No law yet

- MoE layers and MoE models (5 route-table layers, all MoE fixture rows): needs the gating
  (gated or Relu2), the distinct experts a batch touches and the down format (step 11).
- A4 / `nvfp4` activations (161 rows): activation bytes with block scales are not counted.
- Sliding-window models at long context (44 decode rows): window KV bytes.
- `u4zp` and any format missing from the format tables.
- Boards without route-table evidence (RTX 4090, 4070 Ti, 4060 Ti, 5090, 5060 Ti, 2080 Ti, GV100,
  A100): only the whole-model rows above exist for them. The linear gap map needs their
  `ninferno_route_tune` evidence committed.
- Tensor law for Volta int8 (the generation row has no s8 rate) and for Hopper/GB100 (lent rows).
