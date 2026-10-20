# gpt-oss-20b on the A100 40 GB

[A100 40 GB](../../gpus/a100-40gb.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 225 tok/s:

```bash
ninferno-serve openai/gpt-oss-20b --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 627 tok/s over all streams:

```bash
ninferno-serve openai/gpt-oss-20b --with-drafters --spec dflash --draft-tokens 7 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `openai/gpt-oss-20b` | dflash ×7 | auto | 7,005 | 172 | 225 | 456 | 627 | 2026-10-20 |
| `openai/gpt-oss-20b` | off | auto | 7,081 | 186 | 195 | 389 | 474 | 2026-10-20 |

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
