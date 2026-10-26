# Поиск драфтера для неизвестной модели (эмуляция ответа сервера)

Задача (владелец, 2026-10-25): для модели, которой нет в реестре, по странице HF и списку файлов
быстро ответить, каким драфтером её ускорять. Сервер мог бы делать это одним вызовом малой
LLM. Эмулировал так: на каждую модель один вызов Haiku с curl к публичному HF API, а ответ
сверял с реестром (`src/spec/data/drafter_registry.json` v118) и с ручной проверкой curl.

Промпт: `choice/drafter_lookup_prompt.md` (v4, ≈5 КБ). Ответ — JSON:
`original`, `relation` (self / quantized / finetune), `engine`, `candidates[]`
(kind, repo, file, supported, variant), `recommendation` (kind, repo, file, sha), `requests`.
LLM даёт кандидатов, а не вердикт: каждый кандидат NInferno проверяет сам по config
драфтера (архитектура → kind, `registry.supported(kind, engine)`).

## Результаты по версиям

| Версия | Моделей | Верная рекомендация | Запросов | Что сломалось |
|---|---|---|---|---|
| v1 | 10 | 8 | 5–8 | Gemma GGUF: прошёл от -it к pretrained-базе, ассистент для -it отброшен, «finetune». RedHat NVFP4: выбрал сторонний incoai DFlash2 вместо z-lab DFlash. Списки по 17 сторонних EAGLE-3. Nemotron: рекомендовал NVFP4-DFlash, который движок nemotron_h не запускает |
| v2 | 10 | 8 | 2–5 | Добавлены таблица «движок → kinds», доверенные издатели и правило «стоп на вендоре». Новая ошибка: поиск HF без фильтра по автору (limit 100) **не возвращает z-lab**, проверено curl: 100 результатов, z-lab нет. Поэтому RedHat и Huihui остались без DFlash, а Huihui получил MTP из чужого репо |
| v3 | 10 | **10** | 4–6 | Два поиска с фильтром `author=z-lab` и `author=<вендор>`. MTP берётся только из самой цели. Вендор определяется по семейству (nvidia/Gemma-NVFP4 → google) |
| v4 | 6 новых, вне реестра | **6** (1 — ошибка доступа, обработана) | 3–5 | Мой пробел в таблице: Gemma 4 запускает и DFlash (`gemma4_dflash`, z-lab), добавлено. mlx-конвертация: config заявляет MTP, но в индексе весов нет `mtp.*`, проверено curl. Ложный кандидат mtp, рекомендация при этом верна. Добавлено правило: для чужой конвертации MTP требует ключ `mtp.` в индексе. Повтор: mlx — без mtp, Huihui (веса MTP в индексе есть) — mtp сохранён, обе рекомендации верны |

Время одного вызова Haiku: 14–35 с, ≈65–75 тыс. токенов контекста (системный промпт агента
плюс ответы API).

## Проверенные модели (v3/v4, все совпали с реестром или ручной проверкой)

- **Qwen.**
  - RedHatAI/Qwen3.6-35B-A3B-NVFP4 → z-lab/Qwen3.6-35B-A3B-DFlash (+ mtp в цели).
  - nvidia/Qwen3.6-35B-A3B-NVFP4 → тот же DFlash.
  - unsloth/Qwen3.6-35B-A3B-GGUF → DFlash, без MTP.
  - Qwen/Qwen3.6-27B-FP8 → z-lab/Qwen3.6-27B-DFlash (+ mtp.safetensors).
  - mlx-community/Qwen3.6-27B-4bit → DFlash.
  - huihui-ai/Huihui-Qwen3.6-27B-abliterated → DFlash, finetune.
  - unsloth/Qwen3.5-9B-MTP-GGUF → z-lab/Qwen3.5-9B-DFlash (+ mtp).
  - HauhauCS/Qwen3.5-9B-…-Aggressive → DFlash, finetune.
  - Qwen/Qwen3.5-2B → mtp (DFlash для 2B нет).
- **Gemma.**
  - unsloth/gemma-4-12b-it-GGUF → google/gemma-4-12B-it-assistant (в репо ещё лежит mtp-*.gguf, это ассистент в GGUF).
  - nvidia/Gemma-4-26B-A4B-NVFP4 → google/gemma-4-26B-A4B-it-assistant.
  - google/gemma-4-26B-A4B-it → тот же ассистент.
  - bartowski/google_gemma-4-31B-it-GGUF → google/gemma-4-31B-it-assistant.
  - mradermacher/Huihui-gemma-4-31B-it-abliterated-v2-i1-GGUF → тот же ассистент, finetune, цепочка из двух шагов.
- **Прочие.**
  - bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF → nemotron_h_mtp, файл mtp-…-Q8_0.gguf в цели.
  - ggml-org/gpt-oss-20b-GGUF → z-lab/gpt-oss-20b-DFlash; eagle3-*.gguf в репо помечен как неподдерживаемый.
  - meta-models/Muse-Glimmer-30B-GGUF → dflash-*.gguf в самом репо.
  - LiquidAI/LFM2-1.2B → none.
  - mistralai/Mistral-Small-3.2-24B-Instruct-2506 → none.
  - microsoft/Phi-5-mini-instruct → 401 (закрыт или не существует) → ошибка.

## Что из этого следует для продукта

- **Детерминированные шаги не требуют LLM:**
  - цепочка `base_model`;
  - список файлов (mtp-*.gguf, dflash-*.gguf, mtp.safetensors, ключи индекса);
  - два поиска по автору (z-lab и вендор семейства);
  - разбор kind по суффиксу имени.

  Каждая ошибка v1–v2 была ошибкой именно этих шагов, и после того как правило записано явно,
  Haiku выполняет их без сбоев. Значит, их можно написать кодом рядом с
  `convert::hub_base_model`. LLM остаётся только для неоднозначного: какой вендор у семейства,
  если `base_model` нет, и является ли переименованная модель fine-tune.
- **Порядок kinds — из реестра:** для неизвестной модели берётся `pool_rule` по движку, без
  замера. Помечается как `unmeasured`, как сейчас, пока spec-tune не замерит пару.
- **Открытый вопрос владельцу:** для Gemma 4 есть и ассистент, и z-lab DFlash. Реестр сейчас
  ставит ассистента (замерено), DFlash для Gemma не замерен. Порядок «ассистент > DFlash»
  в промпте взят из реестра, а не из замера.
