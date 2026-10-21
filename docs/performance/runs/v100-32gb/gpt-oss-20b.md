# gpt-oss-20b on the Tesla V100 32 GB

[Tesla V100 32 GB](../../gpus/v100-32gb.md) · [gpt-oss-20b](../../models/gpt-oss-20b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 93.6 tok/s:

```bash
ninferno-serve openai/gpt-oss-20b --spec none
```

8 concurrent chats, 188 tok/s over all streams:

```bash
ninferno-serve openai/gpt-oss-20b --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `openai/gpt-oss-20b` | dflash ×7 | auto | fast | 995 | 52.2 | 85.1 | 132 | 187 | 2026-10-20 |
| `openai/gpt-oss-20b` | off | auto | fast | 1,000 | 68.2 | 93.6 | 171 | 188 | 2026-10-20 |

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
