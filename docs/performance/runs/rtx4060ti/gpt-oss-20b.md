# gpt-oss-20b on the RTX 4060 Ti 16 GB

[RTX 4060 Ti 16 GB](../../gpus/rtx4060ti.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 86.6 tok/s:

```bash
ninferno-serve openai/gpt-oss-20b --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 208 tok/s over all streams:

```bash
ninferno-serve openai/gpt-oss-20b --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `openai/gpt-oss-20b` | dflash ×7 | auto | auto | 4,187 | 60.9 | 86.6 | – | – | 2026-10-19 |
| `openai/gpt-oss-20b` | off | auto | auto | 4,279 | 66.0 | 69.8 | 155 | 208 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4060 Ti: driver 570.211.01, CUDA 12.8, power limit 165 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
