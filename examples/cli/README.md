# Stable CLI examples

Committed, offline `--messages` inputs for exercising the product CLI ([CLI guide](../../docs/cli.md))
from the repository root. They cover short text, chat history, images, video, mixed multimodal
history, hard thinking problems, long decode, and four long-context lengths. This is an
operator-facing example set, not a second correctness framework.

[`manifest.json`](manifest.json) lists the frozen comparison cases, their intended observations,
recommended runtime budgets and prepared-prompt token counts. Standalone exploratory prompts may sit
beside them when they deliberately have no frozen output oracle.

## Quick start

Run from the repository root; media paths in the JSON files are repository-relative.

```bash
CLI=./build/apps/ninferno
MODEL=models/qwen3_6_27b.ninferno   # a model converted from the Hugging Face checkpoint

$CLI "$MODEL" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8
```

Expected stdout is exactly `42`. `--no-thinking --greedy` is the normal comparison mode for simple
cases. Reasoning, progress, timings, memory and MTP statistics go to stderr; answer content goes to
stdout.

With MTP:

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

## Text and multimodal cases

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_chat_history.json \
  --no-thinking --greedy --max-new 32

$CLI "$MODEL" --messages examples/cli/messages/text_code_review.json \
  --no-thinking --greedy --max-new 256

for CASE in image_chart image_natural video_temporal multi_image_compare \
            mixed_image_video mixed_multiturn; do
  $CLI "$MODEL" --messages "examples/cli/messages/${CASE}.json" \
    --max-context 8192 --no-thinking --greedy --max-new 128 --vision
done
```

Expected observations:

| Case | Expected observation |
|---|---|
| `text_chat_history` | `Cedar|2041-09-17|37` |
| `text_code_review` | empty input divides by zero; add an explicit empty-input policy |
| `image_chart` | `NIFER VISION 731`; three red circles; blue square on the left |
| `image_natural` | mailbox `24`; sun on the right |
| `video_temporal` | red circle moves; green `3`; square still visible when `3` appears; ending `9` |
| `multi_image_compare` | two circles, three circles, and a new yellow star |
| `mixed_image_video` | `NIFER-9` |
| `mixed_multiturn` | `24-9` |

## Thinking cases

Do not pass `--no-thinking`. Give reasoning enough room to finish and move to answer content; the
model may stop before the maximum.

```bash
$CLI "$MODEL" --messages examples/cli/messages/thinking_logic_grid.json \
  --greedy --max-context 16384 --max-new 8192

$CLI "$MODEL" --messages examples/cli/messages/thinking_multimodal_checksum.json \
  --greedy --max-context 8192 --max-new 4096 --vision

$CLI "$MODEL" --messages examples/cli/messages/reasoning_jacobian_counterexample_3d.json \
  --greedy --max-context 32768 --max-new 16384
```

- The logic grid has one solution and must end with `CHECK=4606`.
- The multimodal case reads independent facts from two images and one video, and must end with
  `CHECKSUM=2238`.
- The Jacobian prompt is a standalone exact-proof stress case without a frozen output oracle, so it
  is not in the comparison manifest.

## Long decode

The three competition-style math problems were written for this repository, so no model has seen
them or their answers. They get generous budgets and run until the model's stop token; they do not
search for the smallest `max-new` that fits one output.

```bash
for CASE in 01 02 03; do
  $CLI "$MODEL" --messages "examples/cli/messages/long_decode_math_${CASE}.json" \
    --greedy --max-context 262144 --kv-dtype int8 --max-new 65536
done
```

The boxed integer answers for cases 01, 02 and 03 are `3543`, `744` and `741`. Greedy decoding
keeps runs comparable, but a heavily quantized model can fall into a repetition loop under it
(Qwen3.8-27B IQ3_S on case 01 reached its answer, then repeated a phrase to the budget); without
`--greedy` the model's own sampling defaults apply, as its model card recommends for thinking.

## Long context

Prompt lengths include the chat template with thinking disabled. Each input is one long document
with the same retrieval needle at 50 % depth. The document is Federal Register rules and proposed
rules published between 15 September and 16 October 2026: public-domain works of the United States
Government, newer than any model's training data, so a model can neither recall the text nor spot
the needle as a change to a text it knows. Run the four lengths `8k`, `64k`, `128k` and `256k`:

```bash
for LEN in 8k 64k 128k 256k; do
  $CLI "$MODEL" --messages "examples/cli/messages/long_niah_${LEN}.json" \
    --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
    --no-thinking --greedy --max-new 128
done
```

All four must output:

```text
ORCHID=493817; COLOR=COBALT
```

[`tools/bench/build_text_examples.py`](../../tools/bench/build_text_examples.py) builds the
long-context and long-decode files; its `--check` verifies them against the manifest without a
tokenizer. The committed prompt token counts come from the Qwen3.6 tokenizer and chat template,
which Qwen3.6-27B and Qwen3.6-35B-A3B share. Qwen3.8-27B uses the same CLI
surface but has its own tokenizer and chat-template resources; check its prepared token count when
using these fixtures.

## Send the same fixture to Serve

`send_to_serve` submits one CLI messages file to an already-running OpenAI Chat Completions endpoint.
Run it as a repository module so it uses the maintained Serve client.

```bash
python3 -m examples.cli.send_to_serve \
  examples/cli/messages/image_chart.json \
  --base-url http://127.0.0.1:8080 \
  --model qwen3.6-27b --no-thinking --max-tokens 64
```

- Local image and video paths are read on the client and encoded as data URIs. The server never
  receives a local filesystem path.
- Start `ninferno-serve` with `--vision` for media fixtures.
- Relative media paths use the current working directory by default. Pass `--media-root` when
  running elsewhere.
- `--dry-run` prints the exact converted request without contacting a server.

## Fixture construction

All PNG and MP4 media are project-authored deterministic scenes. The video is five seconds at 8 FPS
with forty H.264 frames. Runtime tests never depend on a network URL or mutable external content.

The committed files are the canonical inputs; there is no in-tree regeneration script. A deliberate
replacement must update the file, its manifest hash, prompt-token count and oracle together.
