# Gemma 4 26B-A4B on the L4

[L4](../../gpus/l4.md) · [Gemma 4 26B-A4B](../../models/gemma-4-26b-a4b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 92.9 tok/s:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 5
```

4 concurrent chats, 205 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 5 --max-concurrency 4
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | mtp ×5 | auto | 3,092 | 86.1 | 92.9 | 205 | – | 2026-10-20 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | off | auto | 3,033 | 58.0 | 62.9 | 139 | 180 | 2026-10-20 |

## Measurement

- NVIDIA L4: driver 615.71.09, CUDA None, power limit 72 W.
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
