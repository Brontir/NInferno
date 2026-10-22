# gpt-oss-20b on the RTX 4090

[RTX 4090](../../gpus/rtx4090.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 261 tok/s:

```bash
ninferno-serve openai/gpt-oss-20b --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 865 tok/s over all streams:

```bash
ninferno-serve openai/gpt-oss-20b --with-drafters --spec dflash --draft-tokens 7 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `openai/gpt-oss-20b` | dflash ×7 | auto | auto | 13,748 | 184 | 261 | 592 | 865 | 2026-10-19 |
| `openai/gpt-oss-20b` | off | auto | auto | 14,131 | 197 | 205 | 470 | 637 | 2026-10-19 |

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
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --spec none
```
