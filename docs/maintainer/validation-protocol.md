# Validation protocol for user measurements

The engine decides from the card's published facts and the kernels' counted facts; the lab's
measurements only verify that theory. A user's measurement, made with the kit
([user-measurement-kit.md](user-measurement-kit.md)), is one more check of it on a card the lab
does not have. This page says what a user can measure, how a submission is checked, and what it
may and may not change.

## The rules

1. **Reproduced before it counts.** A user's result becomes "Measured on this board" in the
   plan's provenance only after a maintainer reproduces it on a lab or rented card. Until then it
   is evidence: the validator's verdict says which estimate it confirms or contradicts.
2. **Its own board class only.** A reproduced user result may raise a constant's `cards` count
   only for the board class it was measured on. It is never served as a lab card: it is not a
   generation's median, a template card or a transfer source. The kit page states the same for a
   user's speculation pairs ([A user's tune into the registry](user-measurement-kit.md#a-users-tune-into-the-registry-maintainer)).
3. **Nothing feeds a decision at run time.** The engine embeds its constants. A user's number
   changes a decision only when a maintainer changes a constant from facts and that change ships
   in a release. A contradiction is fixed in the model, never by fitting a constant to it.
4. **The lab set for a new generation is two cards**, each measured with
   `measure.py --full --switch-factors`: decode and prefill of every model of the set that fits,
   hard50, the speculation tune of the registry's anchors, the route sweep, the KV read factors
   and the exact route's factor. Two cards are what principle 4 asks of a hardware constant
   ([deterministic-decisions.md](deterministic-decisions.md)). The decode census under load and
   prefill at several lengths are not in the kit yet (open points below).
5. **A new board of a known generation needs no measurement.** Its facts price it. The kit only
   confirms or contradicts that price.

## What a user can measure

| Kit option | What it measures | What it checks |
|---|---|---|
| (always) | plain decode and prefill of each model that fits | the speed model's estimate for this board |
| `--hard50` | the hard question set | a gross failure: a broken kernel, template or download |
| `--spec-tune S` | speculation acceptance and round cost | the registry's pairs and the plan's round cost |
| `--routes` | every route of each projection | the route selection on this chip |
| `--switch-factors` | Qwen3.5-9B Q4_K_M at each KV codec, and exact against fast | the KV read factors and the exact route's factor of the card's generation |
| `--kld` | each switch's KL divergence from the exact reference | the model's KLD terms |

The last two measure the quantities the configuration choice is gated on.

**`--switch-factors`** repeats the lab's switch-factor run on one model (Qwen3.5-9B Q4_K_M, the
kit's first entry). It starts the server with each of INT8, BF16, FP8, K8V4 and NVFP4 KV under the
fast route, and INT8 KV under the exact route. Each start is measured as a plain run (two passes,
the same rules R1 to R7), at a context whose plan decodes at a 22K-token depth (the lab's depth;
the model's own context where the card cannot hold it). The validator divides each
configuration's decode by the INT8 fast base's, measured and predicted. A card's absolute error
cancels in that ratio, and the ratio is what the configuration choice compares between two
candidates on one card. It tests the generation's KV read factor of that codec, or the exact
route's compute factor (`kKvReadFactor`, `kExactComputeEta` in `choice_constants.h`).

**`--kld`** scores each switch alone with `ninferno-perplexity` against the exact reference (the
exact route, BF16 KV, an FP32 state) on the lab's corpus (`perplexity-1m`, one stream a domain):
INT8, FP8, K8V4 and NVFP4 KV, and the fast activations. A KLD term is a model constant: it depends
on the model's file, not on the card's speed. It is not identical across cards, though: two lab
cards' terms of one switch differ by up to 13 % (the KV codecs of Qwen3.5-9B on the RTX 5070 Ti and
the RTX 4070 Ti). A term the lab has measured is reported as confirmed within ±15 % of it
(`KLD_TOLERANCE`, just above that spread) and as contradicted outside it. These are reporting
words only: no term changes from them. For a model whose family has no lab term, the value is a
candidate constant until a maintainer reproduces it.

**`--kld` is a lab tool for now.** It needs `ninferno-perplexity` and the source tree's corpus, so
it runs only from a build, and P1 refuses a build's engine. A `--kld` file passes the validator
only with `--any-build`, on a maintainer's or a lab's run. It cannot pass the CI job for
`docs/performance/user-runs/` until a release ships both (open points).

## Provenance: which build measured

A result file (schema `ninferno-user-run/1.4`) records the build that measured:
- `ninferno.build`: the build section of `ninferno-serve --build-manifest`, read from the engine
  that runs on the measured card: the release version and the commit it was built at, the
  architecture and the CUDA runtime;
- `ninferno.engine`: the engine's tag and the SHA-256 of its executable, the file that holds this
  card's kernels (`bin/ninferno-serve-sm<tag>` of a release, a build's own `ninferno-serve`).

`tools/kit/releases.json` lists every published release: its commit and the SHA-256 of each
engine, by platform and tag. The validator accepts a file only when its version, its commit and
its engine's SHA-256 are one release's entry (rule P1). The entry is added after each release
([release.md](release.md), step 11) and never changed.

P1 catches honest mistakes: a build of one's own, a modified engine, a `-dirty` commit, a file from
an older kit. It is not proof of anything. The file is written by the user, who can copy a
release's hashes into it. What guards the constants is rule 1: nothing counts until a maintainer
reproduces it.

## Untrusted input

A submitted file is data from someone we do not know. The validator:
- reads it as JSON only; nothing in it is executed, imported or evaluated;
- refuses, before parsing, a file over the size cap (64 MiB; a full run of the whole model set is
  estimated at about 40 MB, most of it telemetry, not yet measured) or anything but a regular file;
- refuses a file that is not UTF-8, not JSON, nested too deep, or holds `NaN`, `Infinity`, a number
  too large to be finite (`1e400`) or a key given twice;
- escapes every control character of a string it prints (an id, an error, the schema), so that a
  submitted string never starts a line of the CI log, where the runner would read it as a command;
- refuses a schema version it does not know (major 1, minors 0 to 4) and any key `measure.py` does
  not write, at the file's top, in `kit`, in `ninferno` and in each run (by its mode);
- refuses a value of another type than the kit writes, where a rule reads it;
- refuses a build that is not a release's (P1).

The CI job [`.github/workflows/user-runs.yml`](../../.github/workflows/user-runs.yml) runs on
every pull request that touches `docs/performance/user-runs/`:
- on `pull_request`, never `pull_request_target`: no secret reaches it, and its token can only read;
- the validator and the release manifest come from the base branch, and the pull request's files
  are read as data only;
- a pull request that adds or changes a result file may change nothing outside
  `docs/performance/user-runs/`;
- the folder may hold only `NAME.json` files and this README: no link, no subfolder;
- every JSON file in the folder is validated, and a rejected one fails the job;
- the runner reads no workflow command from the validator's output (`::stop-commands::`).

The job is a check, not a guard. On `pull_request` GitHub runs the pull request's own copy of the
workflow, so a pull request that edits the workflow can change the check. The check therefore
means something only when the pull request touches nothing outside `docs/performance/user-runs/`.
The step above states that rule, though such a pull request could edit it too. The real guard is
a maintainer's review. `.github/CODEOWNERS` names a maintainer for the folder, the validator, the
release manifest, the workflow and itself. The review is required only when the default branch
requires a code owner's review (open points).

## From a pull request to evidence

1. The user runs the kit and sends the file as a pull request (or an issue). The kit never sends
   anything.
2. CI validates it. A file enters the folder only as `accepted`. Within it, single runs and
   metrics may still be rejected by their rules.
3. A maintainer reads the report. A contradiction names the estimate it disagrees with. The
   maintainer reproduces it on a lab or rented card of the same board class, or closes it with
   the reason.
4. A reproduced contradiction is fixed in the model from facts, in a pull request of its own.
   Only then may the board class's `cards` count rise (rule 2). The change reaches users in the
   next release (rule 3).

## Changing this page

A threshold or rule in `validate.py` (the size cap, the known minors, the tolerances
`KLD_TOLERANCE` and the R/S/H/RT/W/K rules) changes here and on the kit page first. A new key in
the result file is a minor schema bump, added to `measure.py` and to `validate.py`'s known keys in
the same pull request.

## Open points

- **Branch protection.** CODEOWNERS asks for a maintainer's review, but GitHub enforces it only
  when the default branch's protection or a ruleset requires a code owner's review. The repository
  has neither yet: that is a repository setting for the owner.
- **`--kld` is lab-only.** A release does not ship `ninferno-perplexity` or the `perplexity-1m`
  corpus, so `--kld` runs only from a build of the source tree. A build's file is refused by P1:
  it passes only with `validate.py --any-build`, never the CI job. Users can contribute KLD terms
  once both ship in the release's kit.
- **The quantization curve.** `--kld` measures the switch terms, not a model's quantization curve
  (each quantization against its Q8_0, `kQuantKld`). That needs the Q8_0 and every quantization of
  the model downloaded, and a kit option of its own.
- **The decode depth.** The plan prices decode at a quarter of the asked context, so
  `--switch-factors` asks for an 88K context to reach the lab's 22K depth. A card that cannot hold
  it falls back to the model's context and a shorter depth, where the KV share of a step, and the
  test, is smaller. `ninferno-serve --plan` has no option to price another depth.
- **The census under load.** The new-generation set of the roadmap (#618) also asks for decode at
  2, 4 and 8 streams and prefill at 2K, 7K and 21K tokens (`kRowsEta`, `kKernelCost`,
  `kPrefillEta`). The kit measures one stream, and prefill at the plan's typical prompt only.
- **No release yet.** `releases.json` is empty until the first release. Until then every
  submission is refused by P1; a lab's own run is checked with `validate.py --any-build`.
