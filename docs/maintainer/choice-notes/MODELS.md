# Строгий список моделей (утверждён владельцем 24.10)

Отбор: каждое семейство движка покрыто (Qwen 3.5 dense / MoE, Gemma 4 dense / MoE, gpt-oss, Nemotron-H,
dense decoder); модель влезает в 8–24 ГБ хотя бы одним 4-битным квантом; есть драфтер; популярна на HF.
Варианты — до 5 самых скачиваемых (HF API, 24.10) **в форматах, которые движок принимает**
(GGUF, исходные safetensors, NVFP4 modelopt / compressed-tensors, AWQ, GPTQ, AutoRound, FP8, MXFP4);
MLX, bnb, ONNX, NPU отброшены; варианты, не влезающие в 24 ГБ, отброшены.

| # | Модель (база) | Семейство | Драфтер | Стенды (12/16/24 ГБ) |
|---|---|---|---|---|
| 1 | Qwen/Qwen3.5-9B | qwen3_5 dense | MTP в весах | все |
| 2 | Qwen/Qwen3.5-4B | qwen3_5 dense | MTP в весах | все |
| 3 | Qwen/Qwen3.8-27B | qwen3_5 dense | MTP, DFlash2 | 16 ГБ (IQ3), 24 ГБ |
| 4 | Qwen/Qwen3.6-35B-A3B | qwen3_5 MoE | MTP, DFlash, DFlash2 | 16 ГБ (IQ3), 24 ГБ |
| 5 | google/gemma-4-12B-it | gemma4 dense | ассистент (MTP) | все |
| 6 | google/gemma-4-26B-A4B-it | gemma4 MoE | ассистент | 12 ГБ (IQ2), 16, 24 |
| 7 | google/gemma-4-31B-it | gemma4 dense | ассистент | 24 ГБ (16 — IQ2/IQ3) |
| 8 | openai/gpt-oss-20b | moe_decoder | DFlash | 16, 24 |
| 9 | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B | nemotron_h | MTP (отд. файл) | только 24 ГБ |
| 10 | meta-models/Muse-Glimmer-30B | dense_decoder | DFlash / DFlash2 | 16 ГБ (IQ3), 24 |

## Варианты (до 5, по скачиваниям; каждый проверен по карточке HF: чат-модель, формат движка, ≤24 ГБ в плане)

1. **Qwen3.5-9B**: unsloth/Qwen3.5-9B-GGUF (1,36 M) · HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
   (GGUF, файнтюн, 0,65 M) · bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF (GGUF глубокого чат-файнтюна Xiaomi, 0,96 M) · lmstudio-community/Qwen3.5-9B-GGUF
   (0,50 M) · AxionML/Qwen3.5-9B-NVFP4 (0,24 M)
2. **Qwen3.5-4B**: unsloth/Qwen3.5-4B-GGUF (1,25 M) · sanskar003/Qwen3.5-4B-AWQ (0,48 M) ·
   lmstudio-community/Qwen3.5-4B-GGUF (0,16 M) · bartowski/Qwen_Qwen3.5-4B-GGUF (0,12 M) ·
   unsloth/Qwen3.5-4B-MTP-GGUF (0,10 M)
3. **Qwen3.8-27B**: unsloth/Qwen3.8-27B-GGUF (5,80 M) · cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF (2,51 M) ·
   mradermacher/Huihui-Qwen3.8-27B-abliterated-GGUF (GGUF файнтюна, 0,02 M; оригинал 38 k) ·
   ReadyArt/Dark-Scarlett-27B-v2.0-GGUF (GGUF файнтюна, 0,01 M) · mradermacher/Qwen3.8-27B-Uncensored-GGUF
   (GGUF файнтюна JonathanColetti, 0,02 M)
4. **Qwen3.6-35B-A3B**: nvidia/Qwen3.6-35B-A3B-NVFP4 (4,14 M) · unsloth/Qwen3.6-35B-A3B-GGUF (1,30 M) ·
   unsloth/Qwen3.6-35B-A3B-MTP-GGUF (1,14 M) · RedHatAI/Qwen3.6-35B-A3B-NVFP4 (0,97 M) ·
   HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive (GGUF, 0,95 M)
5. **Gemma 4 12B**: unsloth/gemma-4-12b-it-GGUF (1,53 M) · HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced
   (GGUF, 0,80 M) · RedHatAI/gemma-4-12B-it-NVFP4 · lmstudio-community/gemma-4-12B-it-GGUF (0,10 M) ·
   google/gemma-4-12B-it (оригинал BF16, 24 ГБ — только как эталон)
6. **Gemma 4 26B-A4B**: nvidia/Gemma-4-26B-A4B-NVFP4 (1,11 M) · unsloth/gemma-4-26B-A4B-it-GGUF (0,99 M) ·
   HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP (GGUF, 0,38 M) · RedHatAI/gemma-4-26B-A4B-it-NVFP4
   (0,37 M) · unsloth/gemma-4-26B-A4B-it-qat-GGUF (каталог)
7. **Gemma 4 31B**: unsloth/gemma-4-31B-it-GGUF (0,26 M) · HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP
   (GGUF Q4_K_M 18,7 ГБ, 84 k) · LilaRest/gemma-4-31B-it-NVFP4-turbo (19,3 ГБ, 76 k) · ggml-org/gemma-4-31B-it-GGUF
   (Q4_0 18 ГБ, 36 k) · unsloth/gemma-4-31B-it-qat-GGUF (каталог)
8. **gpt-oss-20b**: openai/gpt-oss-20b (MXFP4, 5,76 M) · unsloth/gpt-oss-20b-GGUF (0,54 M) ·
   ggml-org/gpt-oss-20b-GGUF (0,23 M) · lmstudio-community/gpt-oss-20b-GGUF (0,09 M) ·
   MuXodious/gpt-oss-20b-RichardErkhov-heresy (чат-файнтюн, MXFP4, 0,03 M)
9. **Nemotron 3.5 Lightning 30B-A3B**: ggml-org/…-GGUF (0,19 M) · nvidia/…-NVFP4 (0,95 M; заменил …-NVFP4-DSpark — это драфтер DSpark, не модель, DECISIONS 37) ·
   unsloth/…-GGUF (0,04 M) · bartowski/…-GGUF (0,04 M) · useful-quants/…-W4A16 (0,03 M)
10. **Muse Glimmer 30B**: unsloth/Muse-Glimmer-30B-GGUF (0,62 M) · RadixArk/Muse-Glimmer-NVFP4 (0,21 M) ·
    meta-models/Muse-Glimmer-30B-GGUF (0,18 M) · 0bserverx/Muse-Glimmer-30B-Heretic-Uncensored-GGUF (0,02 M) ·
    bartowski/Muse-Glimmer-30B-GGUF (0,02 M)

## Не вошли (и почему)

- Qwen3.5 0.8B / 2B — малы, мало кому нужны на GPU; при желании — один из них как «самая слабая карта».
- Qwen3.6-27B — тот же размер, что Qwen3.8-27B, вытеснен им (2,4 M против 7,9 M).
- Qwen3-8B (8,2 M, dense_decoder) — на нём сняты термы точности dense; можно держать как лабораторную,
  а не каталожную.
- FP8 и BF16 оригиналы 26–31B, FP8 Qwen3.8-27B — не влезают в 24 ГБ.

## Что это даёт

- Матрица прогонов конечна: 10 моделей × ≤5 вариантов × 4 стенда (где влезает) — каждая клетка
  проверяется: выбор, план, скорость, точность, спекуляция.
- Реестр спекуляции и замеры точности — только для этих 10 (+ варианты); остальное — «не проверено»,
  с честной подписью в GUI.
- Nemotron проверяем только на 4090 (меньшего файла, влезающего в 16 ГБ, нет).
