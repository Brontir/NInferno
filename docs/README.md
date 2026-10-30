# NInferno documentation

Start with the [project README](../README.md): install NInferno for your GPU, then pick a model in
the [local page](ui.md) or serve one straight from its Hugging Face checkpoint.

## User guides

| Document | Purpose |
|---|---|
| [Performance](performance/README.md) | measured prefill and decode speed for every GPU × model, with a page per GPU, per model and per GPU × model |
| [The local page](ui.md) | `ninferno-ui`: pick, download and run models in the browser, its options |
| [Building from source](building.md) | your own build, Docker images and release archives |
| [Methodology](performance/methodology.md) | how the performance numbers are measured and how to reproduce them |
| [HTTP serving](serving.md) | start a server, the main options, endpoints and examples, speculation, memory, authentication |
| [Serving reference](reference/serving.md) | every request field, compatibility rule, state, log line and error code |
| [CLI](cli.md) | text, chat history, image and video input, sampling, speculative decoding, runtime options |
| [Weight conversion](weight-conversion.md) | official recipes per architecture, custom formats and sources, drafters, artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode and long-context inputs |

The executable `--help` output is the exact source for option spelling and defaults.

## Repository guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The references under [`maintainer/`](maintainer/) record the architecture, model, artifact and
maintenance contracts. [Engine architecture](maintainer/engine-architecture.md) is the top-level
reference; the others own narrower contracts:

| Document | Responsibility |
|---|---|
| [Deterministic decisions](maintainer/deterministic-decisions.md) | the pillar: decisions from chip facts, the artifact and options; tiers B1/B2/S, the fingerprint, what is promised and forbidden |
| [Configuration choice](maintainer/configuration-choice.md) | the KV codec, activations and quantization the engine picks when left automatic: memory, speed and accuracy from facts, the trade, what a decision reads, the explanation |
| [GPU facts table](maintainer/gpu-facts.md) | the PCI-ID reference table: fields, the reference clock, how facts are derived, adding a SKU |
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | the `.ninferno` directory, objects, logical bindings, uses, resources and file framing |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes and scales, conversion arithmetic |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state |
| [Gemma 4 model](maintainer/gemma4-model.md) | Gemma 4 units, mathematics, KV pools, fused ops, drafters and the verify rule |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and device/host checkpoint policy |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Maintainer options and environment variables](maintainer/environment.md) | route tables, lab tuners and debug options; the output-only variables for numerics dumps and fault injection |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch trade-offs and report format |
