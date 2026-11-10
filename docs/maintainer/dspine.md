# DSpine

DSpine is a parallel drafter for speculative decoding from Tencent (arXiv 2609.36173, code
`github.com/Tencent/DSpine`, weights `tencent/Qwen3.8-27B-DSpine-b16`). It is DFlash with a
causal conditioning chain carried through the depth of the draft network. This page records what
the method is, what was checked, how it maps onto NInferno and what is implemented. The engine
does not run it yet: [`spec/dspine.h`](../../src/spec/dspine.h) reads and validates the checkpoint,
and the converter refuses to read a DSpine draft as a plain DFlash draft.

## The method

`b16` is the block width: one anchor plus 15 proposals per round. `DSpine` is the drafter. The
checkpoint's config says `DFlashDraftModel`, `model_type=qwen3`; only
`dflash_config.selected_prefix_code_bridge` tells it from DFlash.

Backbone (as [DFlash](dflash.md), with these differences): five sliding-window layers (8192),
taps at target blocks `[1,16,31,46,61]`, `fc` `[5120, 5*5120]` and `hidden_norm`, no dynamic
convolution, and the block attends **causally** (`is_causal`, `draft_block_causal`): row t sees the
context and rows 0..t of its block.

Code bridge, after every draft layer l (weights `selected_prefix_bridge.*`):

```text
raw, rms = x, sqrt(mean(x^2))                          (x = the residual stream after layer l)
z_t      = read_l . (raw_t / (rms_t + eps))            read_l [C=1024, H]   (read_weight for the last)
c_t      = sqrt(C) z_t / (|z_t| + eps)                 the code row t predicts for itself
c_0      = sqrt(C) codebook[anchor token]              the anchor row is exact
q,s,v    = gate c_t, predecessor_gate c_(t-1), prev c_(t-1)   each [R=512]; gate has a bias
m_t      = sigmoid(q_t + s_t + b) * v_t                m_0 = 0
x_t     <- raw_t + rms_t * write_l m_t                 write_l [H, R]; row 0 is never written
```

So a row learns its predecessor's *predicted* token (a code in the target's output-embedding space,
the codebook is `[248320, 1024]`) at every depth, while all rows still run in one parallel pass.

Last-write refinement (`code_bridge_second_pass`: `serial_table`, `head_topk` 16): the final hidden
gives the top-16 candidate tokens of each row from the target's LM head. For each row and each
candidate *predecessor* p among the previous row's 16 candidates, the last write is redone with p's
exact codebook row, and the 16 candidates c are scored again:

```text
E_i[p, c] = lm_head[c] . norm(raw_i + rms_i * write_last(sigmoid(q_i + gate_p + b) * val_p))
```

The chain is a left-to-right walk of these tables: proposal 0 takes the argmax of the table with
the anchor as predecessor (every p equal), proposal i the argmax of `E_i[lane of i-1, .]`; at
temperature > 0 it draws from softmax(E/T) and keeps that q. This is the same object as DFlash2's
conditional selector (`E_i[p,c]` over top-16, `docs/maintainer/dflash.md`), computed differently.

Serving adds "dynamic verification" (DSpark's SPS): the target verifies fewer than 16 rows when the
measured cost says so (`verification/policy.py`: a width per batch size). It only lowers the
number of proposals checked; it cannot change an output.

## Checked against the sources

| Claim | Check | Result |
|---|---|---|
| 2,025,912,064 BF16 parameters | tensor shapes of the safetensors header summed in `test_spec_dspine` | exact; no hidden tensor |
| Lossless | `sglang_dspine/sampling/rejection.py`: token-wise rejection `coin*q < p`, correction from `max(p-q,0)`, bonus from p | exact in distribution (greedy: the target's argmax chain). Same status as DFlash/DFlash2: the target is the authority |
| Acceptance 27B (card, T=0, vs DFlash2) | GSM8K 7.20 vs 5.69 (+27 %), MATH 6.49/5.49 (+18 %), HumanEval 5.71/4.87 (+17 %), MBPP 6.14/5.04 (+22 %), MT-Bench 4.48/4.11 (+9 %) | a different block (16 vs 8) and a different runtime; not our measurement |
| Paper headline (+27.8 % acceptance, +23.3 % throughput) | abstract | Qwen3-4B/8B against **DFlash**, not the 27B against DFlash2 |
| Throughput 27B (2.7-3.8x AR) | card table, SGLang, concurrency 8 | not reproduced; the GGUF fork's own claim (125 vs 52 tok/s DFlash2, RTX 3090) is third-party |

Cost per round (W=16, one request): the weights streamed are the backbone (1.73 B parameters) and
the bridge matrices (41 M), 3.5 GB in BF16, about what DFlash2's five layers read. The codebook
(508 MB, a quarter of the file) is gathered: 1 anchor row plus 16 sender rows per later proposal, and
the lattice reads the LM head rows of the candidates (240 rows). The exact lattice costs
`15 * 16 * (2*512*1024 + 512*5120)` multiply-adds plus `15 * 16 * 16 * 5120` for the logits, small
against the backbone. Memory: the 4.05 GB file (3.54 GB without the codebook rows, which a 4.5-bit
file shrinks from 508 MB to ~143 MB), a context K/V ring of five sliding layers as DFlash2's.
`dspine_round_facts` holds these numbers; the draft is weight-bound at small T, so a W=16 pass costs
about one W=8 pass.

Not verified here: acceptance and speed in NInferno (no forward pass exists yet), and the
quality of a quantized codebook (see below).

## Other engines

SGLang 0.5.22 runs it with the `sglang_dspine` plugin (Triton kernels `prepare_read`,
`normalize_codes`, `feature_message`, `feature_finish`, a two-stage top-k, `_serial_table`,
`_serial_chain`; tables `TG[v]=bf16(lm_head[v]*norm) @ W_last`, `TSV[v]=[gate_p;prev] @ code(v)` of
`[V,512]` and `[V,1024]` kept per vocabulary row). llama.cpp: no upstream support. The fork
`Ajay9o9/llama.cpp` branch `dspine` (+1 commit, `src/models/dflash.cpp` +179 lines) builds the same
graph from ggml ops and packs the lattice as DFlash2 does. vLLM: no native support; the community plugin in `JRamirez-UAB/Qwen3.8-27B-DSpine-vLLM-quants` (`plugin/`) runs it on vLLM's DFlash path and reports +21..49 % over the target's MTP 3 with 1-2 requests and up to 10 % slower with 8 (RTX PRO 6000 and A40, its own measurement).

## Mapping to NInferno

| Part | Today | DSpine |
|---|---|---|
| Config, kind | `dflash` / `dflash2` | new kind `dspine` (same architecture string, told by the bridge key) |
| Context features, K/V materializer | DFlash | reused unchanged (taps `[1,16,31,46,61]`) |
| Layer attention | non-causal, symmetric window | **new mask**: causal in the block, window over the context |
| Dynamic convolution | DFlash2 only | none |
| Code bridge, 5 writes | - | **new Ops**: read (norm + GEMV + code normalize), message (3 GEMVs + gate), write (GEMV + residual + next norm), fused as `feature_*` |
| Final proposal | DFlash2: unary scores + two codebooks | **new Op** `lattice`: top-16 from the head, the exact last write per predecessor lane, `E[p,c]` |
| The walk, `proposal_q` | `candidate_selector_path` | **new small Op** `candidate_table_path`: same walk and RNG purpose over a given `E`; q is the actual softmax |
| Verification | rejection with the actual q | unchanged |
| Round cost, spec profile | `round_cost.h`, `pairs.cpp` rows per measured pair | rows only from measured acceptance (below) |
| Convert | `drafters.cpp`, `qwen3_5.cpp draft_config`, GGUF `map_dflash2` | map the bridge tensors as written in the checkpoint (`early_read_weights` / `feature_write_weights` stay stacked or split per layer), keep codebook, read/write, gates in BF16 or Q8 |

The engine's draft count K is at most 15, so a block of 16 fits without a wider K. The lattice
does not need the `[V,512]`/`[V,1024]` tables SGLang builds: the rows it needs are the candidates'
(`15*16` of them), computed on the fly from the head and codebook, which keeps the 760 MB of tables
out of the VRAM budget of a 24 GB card.

Per [deterministic-decisions](deterministic-decisions.md), no profile row is added without a measured
acceptance for the pair (target quant, drafter quant, card); the registry entry for
`tencent/Qwen3.8-27B-DSpine-b16` follows the first measurement, and the kind stays unsupported
(`engine_support.h`) until the forward pass matches a reference.

## Quantized variants (Hugging Face, 2026-11-09)

| Repo | Format | Note |
|---|---|---|
| `aj9o9/Qwen3.8-27B-DSpine-b16-GGUF` | GGUF BF16 4.06 GB, Q8_0 2.20 GB, Q4_K_M 1.26 GB | llama.cpp fork `dspine`. In the Q4_K_M file the codebook is **Q4_K**, the other bridge tensors BF16 (the card says the bridge stays BF16) |
| `generalinference/Qwen3.8-27B-DSpine-b16-GGUF` | GGUF BF16 | same tensor names (`dspine_codebook`, `blk.N.dspine_read`, `blk.N.dspine_write`, `dspine_gate`, `dspine_pred_gate`, `dspine_prev`) |
| `JRamirez-UAB/Qwen3.8-27B-DSpine-vLLM-quants` | W4A16, W8A16, FP8, NVFP4, EXL3 (compressed-tensors) | vLLM plugin; bridge, codebook, norms and version tensors stay BF16 in every branch; 4 bit costs no measurable acceptance (its claim) |
| `Avifenesh/Qwen3.8-27B-DSpine-b16-FP8` | safetensors, FP8 backbone | SGLang plugin on a GB10 beside the NVIDIA NVFP4 target |
| `haiping/...-nvfp4` (velox `.vi`), `qwegnai/...-hq4g32` (Qwegn `QWP1`), `netprojects/...-MLX` | engine-private containers and MLX | cannot be read by NInferno; not inspected beyond the cards |

The upstream SGLang plugin refuses quantized draft weights (`DSpineDraftModel`: "does not support quantized weights").

NInferno's own quantization of a draft (`q8_g32_fp16`, as for DFlash) applies to the backbone
matrices; the bridge matrices steer every row and stay BF16 until a measured acceptance says
otherwise (`companion_drafter_bytes` needs a `dspine` row then). The GGUF reader
(`gguf_model.cpp`) refuses a DSpine file with a clear message until the tensors are mapped: it
used to be read as plain DFlash without its bridge.

## Implemented in this change

- `spec/dspine.{h,cpp}`: config parse with the refusals of the reference implementations, tensor
  inventory and parameter count, round facts, the reference arithmetic and the greedy/sampled walk.
- `check_drafter`: a DSpine draft is recognized, validated and reported unsupported; the GGUF
  mapper refuses DSpine files.
- `tests/test_spec_dspine.cpp` on the published config and header (fixtures in
  `tests/fixtures/dspine`).

## Left

1. Device Ops (read/message/write, lattice, table walk, causal block mask), a binding
   (`dspine/...` artifact names) and the draft program next to `execution/draft.cpp`.
2. A reference run to compare the Ops against (the checkpoint's own Triton kernels) and the
   greedy-exactness test on a real pair.
3. Converter mapping for the safetensors and GGUF forms; a registry entry and `engine_support.h`
   row; measured acceptance per card and pair for the profiles; the cost model row.
