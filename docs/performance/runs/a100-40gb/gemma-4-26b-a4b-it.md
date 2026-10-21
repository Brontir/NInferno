# Gemma 4 26B-A4B on the A100 40 GB

[A100 40 GB](../../gpus/a100-40gb.md) · [Gemma 4 26B-A4B](../../models/gemma-4-26b-a4b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 268 tok/s:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 6
```

8 concurrent chats, 683 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 6 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | mtp ×6 | auto | fast | 5,948 | 252 | 268 | 543 | 683 | 2026-10-20 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | off | auto | fast | 5,882 | 182 | 198 | 394 | 498 | 2026-10-20 |

## Measurement

- NVIDIA A100-SXM4-40GB: driver 595.99.02, CUDA 13.2, power limit 300 W.
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
