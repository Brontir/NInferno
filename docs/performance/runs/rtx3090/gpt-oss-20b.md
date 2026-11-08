# gpt-oss-20b on the RTX 3090

[RTX 3090](../../gpus/rtx3090.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 234 tok/s:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --with-drafters --spec dflash --draft-tokens 5
```

8 concurrent chats, 581 tok/s over all streams:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --with-drafters --spec dflash --draft-tokens 5 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `ggml-org/gpt-oss-20b-GGUF` | dflash ×5 | auto | fast | 7,108 | 183 | 234 | 439 | 581 | 2026-11-08 |
| `ggml-org/gpt-oss-20b-GGUF` | off | auto | fast | 7,316 | 210 | 234 | 431 | 516 | 2026-11-08 |
| `openai/gpt-oss-20b` | dflash ×7 | auto | fast | 6,400 | 144 | 201 | 424 | – | 2026-10-19 |
| `openai/gpt-oss-20b` | off | auto | fast | 6,476 | 160 | 170 | 361 | 453 | 2026-10-19 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Prefill 24K | 6,695 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Decode @24K | 180 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=1 | 200 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=4 | 360 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=8 | 358 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 3090: driver 580.159.03, CUDA 13.0, power limit 350 W.
- NVIDIA GeForce RTX 3090: driver 615.71.09, CUDA None, power limit 390 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec dflash --draft-tokens 5 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 --spec dflash --draft-tokens 5 
--max-context 32768 --max-concurrency 8 --spec none
--max-context 32768 --max-concurrency 8 --spec none 
```
