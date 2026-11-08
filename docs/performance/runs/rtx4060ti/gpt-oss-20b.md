# gpt-oss-20b on the RTX 4060 Ti 16 GB

[RTX 4060 Ti 16 GB](../../gpus/rtx4060ti.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 109 tok/s:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 281 tok/s over all streams:

```bash
ninferno-serve ggml-org/gpt-oss-20b-GGUF --with-drafters --spec dflash --draft-tokens 7 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `ggml-org/gpt-oss-20b-GGUF` | dflash ×7 | auto | fast | 4,829 | 86.3 | 109 | 211 | 281 | 2026-11-08 |
| `ggml-org/gpt-oss-20b-GGUF` | off | auto | fast | 4,925 | 88.4 | 97.2 | 181 | 233 | 2026-11-08 |
| `openai/gpt-oss-20b` | dflash ×7 | auto | fast | 4,318 | 59.4 | 88.8 | 187 | 256 | 2026-11-08 |
| `openai/gpt-oss-20b` | off | auto | fast | 4,412 | 64.8 | 69.3 | 152 | 201 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Prefill 24K | 3,833 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | Decode @24K | 77.2 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=1 | 91.2 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=4 | 153 | 2026-11-08 |
| llama.cpp b12198 | gpt-oss-20b-GGUF | off | C=8 | 154 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 4060 Ti: driver 615.71.09, CUDA None, power limit 165 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 --spec dflash --draft-tokens 7 
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --spec dflash --draft-tokens 7 
```
