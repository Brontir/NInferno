# gpt-oss-20b on the RTX 5070 Ti

[RTX 5070 Ti](../../gpus/rtx5070ti.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 299 tok/s:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 654 tok/s over all streams:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `ggml-org/gpt-oss-20b-GGUF` | dflash ×7 | auto | fast | 11,179 | 232 | 299 | 521 | – | 2026-11-08 |
| `ggml-org/gpt-oss-20b-GGUF` | off | auto | fast | 11,115 | 239 | 263 | 511 | 654 | 2026-11-08 |
| `openai/gpt-oss-20b` | dflash ×7 | auto | fast | 10,548 | 174 | 250 | 496 | 565 | 2026-11-08 |
| `openai/gpt-oss-20b` | off | auto | fast | 10,838 | 180 | 191 | 426 | 569 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Prefill 24K | 9,927 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Decode @24K | 195 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=1 | 224 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=4 | 382 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=8 | 379 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 5070 Ti: driver 615.71.09, CUDA None, power limit 300 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec dflash --draft-tokens 7 
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --spec dflash --draft-tokens 7 
--max-context 32768 --max-concurrency 8 --spec none 
```
