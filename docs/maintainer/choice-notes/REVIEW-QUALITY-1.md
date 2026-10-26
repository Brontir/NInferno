# Ревью качества 1 (26.10): недавние изменения (#126–#130, #127, #128) и ветки

Ревью сделал Sonnet (только чтение), каждую находку проверяю в коде перед правкой. Владелец: «код
красивый, переиспользуемый, понятный; периодические ревью и рефакторинг — сотни разработчиков».

| # | Находка | Что делаем | Где | Статус |
|---|---|---|---|---|
| 1 | рецепт субъекта плана дважды (plan_offline, app), уже разошёлся (веса, коммит) | product::HubSubjects — один план субъектов хаба | det/gguf-drafter-axis bffe4a30 | сделано, тесты идут |
| 2 | hub_quant_subject строит целый субъект ради attached_drafters | declared_drafters(target) (+ own_components, drafter_target); субъект конвертации берёт только attached | **в master #139 (41bd8556)** | сделано, планы 28/28 = база |
| 3 | «ref → коммит» пять раз, разные правила; разбор имени конвертации трижды | convert::local_commit, ConvertedName | #136 (db315bd2) | сделано, тесты, Codex 👍 |
| 4 | cached_conversion_with_drafters ~75 строк, третье определение «драфтер служит» | ManifestDrafter::present(); conversion_carries_drafters отдельно | #136 (db315bd2) | сделано, тесты, Codex 👍 |
| 5 | комментарии не у своих объявлений (fetch.h, drafters.h, plan_speculation.h, plan_offline.h/.cpp) | вернуть на место | plan_offline, plan_speculation.h — det/gguf-drafter-axis (bffe4a30, f8ba43ed); fetch.h, drafters.h, fetch.cpp — det/refactor-review-1 (dbf7a5d6) | сделано, тесты |
| 6 | ветка драфтер-оси и #127 решали одно двумя путями | слито через HubSubjects | bffe4a30 | сделано |
| 7 | backend_of_drafter_kind — копия backend_of_kind | одна функция в product/speculation; label → backend без try/catch | bffe4a30 | сделано |
| 8 | pair_choice ~165 строк, 3 лямбды, 9 параметров | PairQuery; carried_of/pool_cards/own_cards/nearest_siblings; try_mode_of_kind; same_bit_class; same_repo_revision | det/refactor-pair-choice dc5edd71 (поверх #131) | сделано, тесты + сравнение планов |
| 9 | идентичность хаба строкой «repo@commit»; копия hub_repo_of_directory | core/hub_revision.h: HubRevision (parse, id, same_as), hub_repo_directory/hub_repo_of_directory; комментарии на местах | **в master #143 (84743071)** | сделано |
| 10 | RepoQuant: параллельные массивы, commit в каждой записи | RemoteFile (запись листинга хаба, публичная) вместо трёх массивов; RepoListing{commit, quants} | **в master #144 (9d360406)** | сделано, планы 0 изменений |
| 11 | conversions_stamp — I/O на каждый plan_json, не видит каталог загрузки | convert::conversions_stamp рядом с правилом; + snapshots (дефект: GUI держал устаревший план после удаления загрузки) | **в master #140 (2ef95f43)** | сделано |
| 12 | PlannedTarget::lineage мёртвое; два чтения base_model GGUF; лишний HTTP карточки | удалить (dbf7a5d6); одно чтение base_model GGUF — convert::gguf_metadata_base_repository (#138, 87275c5a); base_models в SupportFacts — одна карточка на план (#144) | #138, #144 | сделано |
| 13 | cache_served / cache_current — двойное чтение, неясные имена | cache_serves + serves_at, одно чтение (dbf7a5d6) | det/refactor-review-1 | сделано, тесты |
| 14 | поиск брата: два прохода, сборка имён отдельно | nearest_siblings — один проход | dc5edd71 | сделано |
| 15 | group-guarded-gain: индексы tables/group, литерал 2, tok_s = выигрыш; bpw: ручной ne-продукт, свободная функция, история в комментарии | группа: GroupMember, kGroupCards, label_gain (#131, 0ebe0ff0); биты: PlanModelFacts::weight_elements, метод bits_per_weight (#133) | #131, #133 | сделано, тесты |
| + | две функции «метка → биты» (spec::label_bits / convert::quant_label_bits) с разными ответами | одна ninferno::quant_label_bits (core/quant_label.h, optional), умолчание у вызова | **в master #137 (45fcc29c)** | сделано |
| + | GUI читал геометрию модели второй копией (ui/catalog.cpp KvModel ~300 строк: KV, состояния, параметры), уже разошлась с plan_facts; странице нужны 4 поля | kv_summary(PlanSourceFacts) — одно чтение | **в master #141 (25433fd5)** | сделано, вживую 10 репо = master |

Ревью второго круга (Sonnet по #137–#140, 26.10): не собрались бы #137 (include) и #140 (перевод строки в тесте); #138 давал ложный репозиторий из ссылок datasets/spaces/другого хоста — всё исправлено (8830f1b1, dde25225, febc21e2, 2f4bcdb3).
