# Gemma 4 26B-A4B on the Tesla V100 32 GB

[Tesla V100 32 GB](../../gpus/v100-32gb.md) · [Gemma 4 26B-A4B](../../models/gemma-4-26b-a4b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 106 tok/s:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 3
```

8 concurrent chats, 141 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 3 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | mtp ×3 | auto | auto | 861 | 90.6 | 106 | 133 | 141 | 2026-10-20 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | off | auto | auto | 862 | 69.7 | 78.0 | 123 | 132 | 2026-10-20 |

## Measurement

- Tesla V100-SXM2-32GB: driver 580.159.03, CUDA 13.0, power limit 250 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --spec none
```
