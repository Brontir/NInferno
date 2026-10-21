# Qwen3.8-27B on the RTX 4090

[RTX 4090](../../gpus/rtx4090.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 122 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 366 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 3 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | dflash2 ×7 | auto | exact | 789 | 117 | 111 | – | – | 2026-10-21 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | dflash2 ×7 | auto | fast | 3,345 | 127 | 122 | 277 | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | mtp ×3 | auto | exact | – | – | – | – | 251 | 2026-10-21 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | mtp ×3 | auto | fast | – | – | – | – | 366 | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | off | auto | fast | 3,376 | 48.7 | 50.5 | 156 | 234 | 2026-10-19 |

## Quality

What each `--activations` mode gives: its speed (the fastest setting measured in it), perplexity (lower is better) on `ninferno-ppl-1m-v1/quick`, and how far it moves the next-token distribution from `exact`: mean KL divergence and the share of positions with the same top token.

| Weights | Mode | One stream tok/s | Prefill tok/s | Perplexity | KL from exact | Same top token | Measured |
|---|---|---:|---:|---:|---:|---:|---|
| UD-Q4_K_XL | exact | 111 | 789 | 4.2663 | reference |  | 2026-10-20 |
| UD-Q4_K_XL | fast = int8 | 122 | 3,376 | 4.2675 | 0.00256 | 98.02 % | 2026-10-20 |

**fast = int8**: on these weights `fast` already runs 8-bit activations, so `int8` takes the same kernels and gives the same numbers.


## Measurement

- NVIDIA GeForce RTX 4090: driver 580.95.05, CUDA 13.0, power limit 450 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --activations exact
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --activations exact
--max-context 32768 --max-concurrency 8 --spec none
```
