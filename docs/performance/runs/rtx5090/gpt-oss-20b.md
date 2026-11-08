# gpt-oss-20b on the RTX 5090

[RTX 5090](../../gpus/rtx5090.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 533 tok/s:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --with-drafters --spec dflash --draft-tokens 5
```

8 concurrent chats, 1,455 tok/s over all streams:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --with-drafters --spec dflash --draft-tokens 5 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `ggml-org/gpt-oss-20b-GGUF` | dflash ×5 | auto | fast | 21,893 | 400 | 533 | 1,102 | 1,455 | 2026-11-08 |
| `ggml-org/gpt-oss-20b-GGUF` | off | auto | fast | 22,671 | 430 | 458 | 897 | 1,191 | 2026-11-08 |
| `openai/gpt-oss-20b` | dflash ×7 | auto | fast | 19,274 | 298 | 409 | 952 | 1,281 | 2026-10-20 |
| `openai/gpt-oss-20b` | off | auto | fast | 20,164 | 317 | 329 | 710 | 979 | 2026-10-20 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Prefill 24K | 18,287 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Decode @24K | 310 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=1 | 353 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=4 | 660 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=8 | 668 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 5090: driver 595.84, CUDA 13.2, power limit 575 W.
- NVIDIA GeForce RTX 5090: driver 615.71.09, CUDA None, power limit 575 W.
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
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --spec dflash --draft-tokens 5 
--max-context 32768 --max-concurrency 8 --spec none
--max-context 32768 --max-concurrency 8 --spec none 
```
