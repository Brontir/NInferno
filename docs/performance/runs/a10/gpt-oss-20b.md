# gpt-oss-20b on the A10

[A10](../../gpus/a10.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 120 tok/s:

```bash
ninferno-serve openai/gpt-oss-20b --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 274 tok/s over all streams:

```bash
ninferno-serve openai/gpt-oss-20b --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `openai/gpt-oss-20b` | dflash ×7 | auto | auto | 4,133 | 87.2 | 120 | 256 | – | 2026-10-19 |
| `openai/gpt-oss-20b` | off | auto | auto | 4,231 | 94.4 | 101 | 222 | 274 | 2026-10-19 |

## Measurement

- NVIDIA A10: driver 570.144, CUDA 12.8, power limit 150 W.
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
