# NInferno hard question set (hard50)

`hard50.json` holds 50 questions on which the lab's earlier EvalScope runs disagreed: each one was
answered right by some of the runs and wrong by others (Qwen3.5-9B UD-IQ2, Q4_K_M and Q8_0 and
repeats of them, Gemma 4 12B Q4_K_M and Q8_0, 24-25.10.2026, eval/ with temperature 0). They sit
at the edge of what these models solve, so one run of the set shows both what a configuration
answers and how fast it decodes, speculation included.

| Dataset | Questions | Source split | Answer |
|---|---:|---|---|
| `gsm8k` | 20 | GSM8K main, test (`gsm8k_main:N`, N the row) | the number after `####` |
| `math_500` | 20 | MATH-500 by level (`math_500_Level L:N`) | the reference answer |
| `mmlu_pro` | 10 | MMLU-Pro by category (`mmlu_pro_<category>:N`) | the option letter |

Every item keeps the prompt exactly as those runs sent it (few-shot examples and the answer format
included), the reference answer and the output budget they used (`max_tokens`: 1024, 2048 for
MATH-500). The 50 were drawn with a fixed seed (20261027) from the questions with mixed outcomes:
179 of GSM8K, 127 of MATH-500, 5356 of MMLU-Pro.

`tools/bench/hard_set.py` runs it against an OpenAI-compatible server (one request at a time,
greedy, thinking off) and grades it; its grader agrees with EvalScope's on 96-98 % of these
questions (the rest are answers without `\boxed{}`, counted wrong for every configuration alike).

Being at the edge, the set is sensitive: one configuration's score moved by up to 6 of 50 between
cards and between speculation depths (Qwen3.5-9B Q4_K_M plain 23-29, its answers diverging after
the first token where two cards' arithmetic differs). A difference of a few questions is numerical
noise, not a change of quality; the quality of a quantization is its KLD (eval/corpora/perplexity-1m).

See `THIRD_PARTY_NOTICES.md` before redistributing the set.
