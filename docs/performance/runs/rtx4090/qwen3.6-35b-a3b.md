# Qwen3.6-35B-A3B on the RTX 4090

[RTX 4090](../../gpus/rtx4090.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 398 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

4 concurrent chats, 843 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7 --max-concurrency 4
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | dflash ×7 | auto | fast | 13,550 | 412 | 398 | 843 | – | 2026-10-19 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | off | auto | fast | 13,777 | 235 | 258 | 612 | 799 | 2026-10-19 |

## Quality

What each `--activations` mode gives: its speed (the fastest setting measured in it), perplexity (lower is better) on `ninferno-ppl-1m-v1/quick`, and how far it moves the next-token distribution from `exact`: mean KL divergence and the share of positions with the same top token.

| Weights | Mode | One stream tok/s | Prefill tok/s | Perplexity | KL from exact | Same top token | Measured |
|---|---|---:|---:|---:|---:|---:|---|
| UD-Q4_K_M | exact | – | – | 4.3359 | reference |  | 2026-10-20 |
| UD-Q4_K_M | fast = int8 | 398 | 13,777 | 4.3371 | 0.00503 | 97.28 % | 2026-10-20 |

**fast = int8**: on these weights `fast` already runs 8-bit activations, so `int8` takes the same kernels and gives the same numbers.


## Measurement

- NVIDIA GeForce RTX 4090: driver 580.95.05, CUDA 13.0, power limit 450 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
