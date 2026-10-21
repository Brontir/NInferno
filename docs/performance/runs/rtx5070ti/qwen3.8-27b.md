# Qwen3.8-27B on the RTX 5070 Ti

[RTX 5070 Ti](../../gpus/rtx5070ti.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 120 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ3_XXS --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 261 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS | dflash2 ×7 | auto | exact | 517 | 109 | 106 | – | – | 2026-10-21 |
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS | dflash2 ×7 | auto | fast | 2,051 | 122 | 120 | – | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS | mtp ×3 | auto | exact | – | – | – | – | 191 | 2026-10-21 |
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS | mtp ×3 | auto | fast | – | – | – | 248 | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS | off | auto | fast | 2,070 | 61.1 | 64.0 | 196 | 261 | 2026-10-19 |

## Quality

What each `--activations` mode gives: its speed (the fastest setting measured in it), perplexity (lower is better) on `ninferno-ppl-1m-v1/quick`, and how far it moves the next-token distribution from `exact`: mean KL divergence and the share of positions with the same top token.

| Weights | Mode | One stream tok/s | Prefill tok/s | Perplexity | KL from exact | Same top token | Measured |
|---|---|---:|---:|---:|---:|---:|---|
| UD-IQ3_XXS | exact | 106 | 517 | 4.4192 | reference |  | 2026-10-20 |
| UD-IQ3_XXS | fast = int8 | 120 | 2,070 | 4.4188 | 0.00198 | 98.38 % | 2026-10-20 |

**fast = int8**: on these weights `fast` already runs 8-bit activations, so `int8` takes the same kernels and gives the same numbers.


## Measurement

- NVIDIA GeForce RTX 5070 Ti: driver 615.71.09, CUDA None, power limit 300 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --activations exact
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --activations exact
--max-context 32768 --max-concurrency 8 --spec none
```
