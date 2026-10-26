# Драфтеры для утверждённых моделей (рабочая таблица, 25.10)

Источник — ручной разбор: карточки моделей и драфтеров на HF, списки файлов репозиториев, реестр
v112, возможности движка (`spec/engine_support.h`: Qwen 3.5 — MTP / DFlash / DFlash2; Gemma 4 —
только ассистент (MTP); Nemotron-H — только MTP; dense / MoE decoder — только DFlash) и конвертера
(ассистент Gemma и MTP Qwen берутся и из GGUF; MTP-файлы `mtp-*.gguf` — тоже).
«Движок» — запускает ли наш движок этот вид для семейства. Всё «да» — кандидаты на замер E(k).

## Qwen3.5-9B
| Драфтер | Откуда | Размер | Лицензия | Движок | Реестр |
|---|---|---|---|---|---|
| MTP (в весах) | `Qwen/Qwen3.5-9B`, `unsloth/Qwen3.5-9B-MTP-GGUF` (те же кванты с MTP внутри) | в весах | Apache-2.0 | да | есть |
| DFlash | `z-lab/Qwen3.5-9B-DFlash` @5fc3b3d4 (обучен на 40K, скользящее окно) | 2,4 ГБ (BF16) | Apache-2.0 | да | **новый** |

Варианты: `unsloth/Qwen3.5-9B-GGUF`, `lmstudio`, `HauhauCS` — **MTP в файле нет**: предлагать
MTP-GGUF-репозиторий unsloth или DFlash. `AxionML NVFP4`, `JEV-9B` (safetensors) — MTP в весах.

## Qwen3.5-4B
| MTP (в весах) | `Qwen/Qwen3.5-4B`, `unsloth/Qwen3.5-4B-MTP-GGUF` | в весах | Apache-2.0 | да | есть |
| DFlash | `z-lab/Qwen3.5-4B-DFlash` @9a1996cc | 1,2 ГБ | Apache-2.0 | да | **новый** |
| DFlash2 | `nchapman/Qwen3.5-4B-DFlash2` (неофициальный) | ? | ? | да | новый, проверить |

## Qwen3.8-27B
| MTP | в весах HF; для GGUF — `unsloth/Qwen3.8-27B-GGUF: MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1,3 ГБ | Apache-2.0 | да | есть |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2` = зеркало `incoai/…` (одни веса) | 3,6 ГБ | Apache-2.0 | да | есть |

## Qwen3.6-35B-A3B
| MTP | в весах HF; `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (MTP внутри) | в весах | Apache-2.0 | да | есть |
| DFlash | `z-lab/Qwen3.6-35B-A3B-DFlash` @f181eece | 0,7 ГБ | Apache-2.0 | да | есть |
| DFlash2 | `incoai/Qwen3.6-35B-A3B-DFlash2` @51ef7b69 | 1,0 ГБ | Apache-2.0 | да | **новый** |

## Gemma 4 12B / 26B-A4B / 31B
| Ассистент (MTP) | `google/gemma-4-{12B,26B-A4B,31B}-it-assistant` (safetensors, 0,8–0,9 ГБ) | | Apache-2.0 | да | есть |
| Ассистент в GGUF | в самих репозиториях unsloth: `mtp-gemma-4-*-it.gguf` (Q8_0 0,44 ГБ, Q4_0 0,24 ГБ); у HauhauCS — тоже | | Apache-2.0 | да | **новый источник** |
| DFlash | `z-lab/gemma-4-*-DFlash` | | Apache / ? | **нет** (Gemma — только MTP) | есть, не запускается |
| EAGLE3 | `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` | | | нет | — |

Ошибка продукта: профили Gemma 12B (и двух 26B) ставят DFlash на thinking → спекуляция пропадает.

## gpt-oss-20b
| DFlash | `z-lab/gpt-oss-20b-DFlash` @d53f6551 | 1,5 ГБ | MIT | да | есть |
| EAGLE3 / P-EAGLE | `zhuyksir/EAGLE3-gpt-oss-20b-bf16` (70K скач.), `amazon/GPT-OSS-20B-P-EAGLE` | | | **нет** (вид не поддержан) | — |

## Nemotron 3.5 Lightning 30B-A3B
| MTP (отдельный файл) | `ggml-org/…-GGUF: mtp-…-{Q4_0 1,1 ГБ, Q8_0 2,0 ГБ}`, `bartowski/…: mtp-…-Q4_0` | | NVIDIA Open | да | есть (bartowski) |
| DFlash | `nvidia/…-NVFP4-DFlash` (от NVIDIA) | | | **нет** (Nemotron — только MTP) | — |

## Muse Glimmer 30B
| DFlash (GGUF) | `meta-models/…-GGUF: dflash-Muse-Glimmer-30B-Q4_K_M.gguf`; `unsloth/…-GGUF: dflash-kquant.gguf` | 1,5 ГБ | Apache-2.0 | да | есть (meta-models) |
| DFlash2 | `z-lab/Muse-Glimmer-30B-DFlash2` = `incoai/…` | | | **нет** (dense — только DFlash) | — |

## Что дальше

1. Замер E(k) для всех «да»: по модели — на 2 картах разных поколений; варианты — на одной.
   Новые драфтеры (z-lab Qwen3.5-9B/4B DFlash, incoai Qwen3.6-35B DFlash2, GGUF-ассистенты Gemma) —
   сначала проверка совместимости конвертером, потом замер.
2. Запуски, которые движок не умеет (DFlash для Gemma и Nemotron, DFlash2 для dense, EAGLE3) —
   отдельный вопрос развития движка; в реестр не попадают, пока движок их не запускает.
