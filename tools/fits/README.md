# Fits

Each constant the configuration choice could not derive from chip facts alone comes from one of these scripts, run on the measurement rows in `data/` and in `tests/fixtures/choice/` (lab hostnames in them name the lab's cards). Rerun a script after new rows land to see whether its constant still holds. [`docs/maintainer/decisions.md`](../../docs/maintainer/decisions.md) records why each one exists.

| Script | Constant | Inputs |
|---|---|---|
| `round_cost/fit_eta.py` | `kSmallTEta`, the share of the int8 peak small-T kernels reach (`src/spec/round_cost.h`) | the shipped drafter registry, `src/core/data/gpu_facts.json` |
| `round_cost/eta_facts.py` | checks `kSmallTEta` against two fact hypotheses (decision 54) | the same, through `fit_eta.py` |
| `speed_model/fit_decode.py` | decode rows per generation (`src/product/choice/choice_constants.h`) | `tests/fixtures/choice/decode_measured.json`, `decode_census.json` |
| `speed_model/fit_moe_decode.py` | the routed-expert share of an MoE decode step (decision 65) | `tests/fixtures/choice/speed_measured.json` |
| `prefill/fit_prefill.py` | prefill linears and attention per generation | `data/prefill/rows.json` |

Run each from the repository root, for example `python tools/fits/round_cost/fit_eta.py`. They need only Python 3 and the repository; no GPU.
