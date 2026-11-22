# HW4: проверка сдвига и три дополнительных контроля после Public

Исследование проведено после результатов11=.34642 и12=.34242. Все новые
target-сравнения ограничены прежними2023Q3/Q4/2024Q1.2024Q2/Q3 повторно не
оцениваются, опубликованные CSV и87 зафиксированных parent-файлов не меняются.
Это адаптивное исследование после Public, а не независимая проверка обобщения.

## Что узнал domain-classifier

Бинарная метка здесь означает принадлежность к test, **не fail инспекции**.
Competition target был физически удалён из metadata перед обучением.
Фиксированы CatBoost200/d3/lr.05/L2=10, domain-balanced class weights и четыре
StratifiedGroupKFold поsite_id; один сайт целиком в одном fold. Оценка —
OOF domain AUROC. Вход не содержит rawID/forecast date/year/quarter/cutoff gap.

| Представление | Все train против test | Только2022/2023Q4 против2024Q4 |
|---|---:|---:|
|Static type/risk/area,3колонки|0.49217|0.49074|
|Own burden/static,21колонка|0.53230|0.53916|
|Own history без explicitcalendar,271колонка|0.52867|0.52312|
|Own21+peer72,93колонки|0.99999987|Не запускалось|

Наиболее сильные признаки распознавания snapshot в compact93:
`ctx_facility_365d_visits_load_smooth`, `ctx_risk_all_visits_fail_smooth`,
`ctx_risk_all_routine_n` и близкие risk/facility aggregates. В own21 самые
заметные одномерные KS относятся к sparse recent90 burden, максимум0.0871;
различимость own/static остаётся слабой даже при сравнении одинакового сезона.

Это обосновывает диагностику context-clock, но **не доказывает причину
провала смеси на Public**. Domain-classifier видит часть того же test-cohort
при обучении, а непрерывные групповые статистики могут идентифицировать любую
конкретную четверть. Сам лучший Public CatBoost тоже использует эти признаки.
Условный сдвиг target или шум небольшой Public-части этим анализом не исключаются.

Adversarial importance weights не предложены для task-модели: own-domain
отношения почти единичны, ESS97–98%; в context нет наблюдаемого overlap.
Высокий ESS после clipping context-отношений не означает хорошее совпадение:
почти все source probabilities малы, нижний clip просто выравнивает веса.

## Непрерывные ranks не решили проблему

Для72 context-полей внутри каждого **unlabeled** query-cohort рассчитан
average percentile rank. NaN и все own/legacycalendar значения сохраняются.
Это явно batch-transductive обработка: roster всех прогнозируемых строк
конкретной четверти используется до просмотра их исходов. Повторный domain
classifier после такой обработки далAUROC0.999979: отличия tie-позиций и
состава групп всё ещё почти идеально указывают на snapshot.

Несмотря на это ограничение, задача проверена отдельной фиксированной
CatBoost-абляцией: компактные94 и полные346 столбцов,500/d5/.04/L2=10,
Plain/MVS.8/rsm1/border254, без подбора параметров. Полные literal feature lists
и параметры до AP находятся в `hw4-post-public-fixed-plans-20261122.json`.

Для компактной ветки дополнительно обучен точный raw94-control: прежний
compact96 с year/cutoff-gap не являлся бы идентичным контролем.

| Seed42 OLD-прогноз | Средний AP |
|---|---:|
|Raw full346, переиспользованный matched checkpoint|0.268277|
|Rank-context full346|0.260406|
|Raw compact94, новый точно matched control|0.272355|
|Rank-context compact94|0.270669|

Нормализация ухудшила полный контроль во всех кварталах, компактный — в двух
из трёх. Формальная75%context/25%rank-compact смесь прошла исходный mean gate
и подтверждение трёх seed: OLD AP0.269289, Δ кcontext+0.001783,
номинальный95%CI[−0.000707,+0.004575]. Но в target-like2023Q4 дельта−0.003194,
а у standalone−0.019991. Это **слабый резерв, не рекомендация для двух попыток**.

Только технически готовый CSV:
`submission_rankctx_compact94__blend75.csv`, SHA
`0de59c0c68cf5248f072d4d61f38fe88777b5f53e0d0aae5c69792e16555ebc6`.
Public неизвестен, файл не загружался. Подтверждены2246ID/sampleorder и точная
формула75%старого context04+25%mean3 новых rank-compact моделей.

## Два остальных новых контроля

**Фиксированные same-season веса3:1.** Пастовые строки той же календарной
четверти, что query, получили вес3, остальные1; среднее веса нормировано к1.
Это не прежние half-life/recency варианты. Все346 исходных признаков и параметры
CatBoost сохранены. До AP добавлено требование неотрицательной дельты в старом
Q4 наряду с общими mean/2-of-3/worst-gate. RawAP0.262608, все три нижеcontext.
СмесьAP0.266728, Δ−0.000778; Q4Δ−0.001262. Подтверждения/full/CSV не делались.

**Own first-routine по завершённым календарным кварталам.** К346 добавлено
ровно11числовых полей:count/failcount/failmean,lastfail/recordedload/age,
last2/4failmean,same-seasoncount/mean/lastfail. Это собственная история,
в отличие от прежних12 peer-seasonal признаков. Использован существующий
консервативный `_first_routines`: exactsnapshot,week<cutoff,quarterend<=cutoff,
левая граница наблюдаемого envelope, unknown/conflicting/boundary-first
исключают site-quarter без замены последующим известным событием. Отсутствие
наблюдения не подменяется отсутствием реальных нарушений.

В test94.9% строк имеют eligible прошлый first,42.2% — same-season ownfirst,
средняя поддержка1.58квартала. Но fixedCB357 rawAP0.262026 и blendAP0.266274
нижеcontext во всех трёх кварталах. Подтверждения/full/CSV отсутствуют.

## Точный объём и проверка

| Работа | Реальные fit |
|---|---:|
|Domain membership:7cases×4fold|28|
|Domain после rank-context:1case×4fold|4|
|Rank forecasting:6screen+3matchedraw+6confirmation+3full|18|
|Same-season forecasting screen|3|
|Own first-quarter forecasting screen|3|
|Всего|**56**|

Таким образом,32 модели использовали только domain membership,24 — task
labels для OLD forecasting. Все24 forecasting-checkpoint заново загружены и
повторили forward EXACT; source/weight SHA и weak-CSV formula проверены.
Четыре persistent feature-unit проверки и одна inline same-season weight/past
guard проверка прошли. Все87 исходных frozen parent/evidence хешей повторно
совпали без refit или повторного AP. Новых recent AP и Kaggle uploads —0.

Артефакты: `analysis/hw4-post-public-20261122/{shift-audit,quarter-rank-context,
same-season,own-first-quarter}/`. Итог — полезные отрицательные контроли,
а не основание обещать новый Public-выигрыш или продолжать перебор параметров
этих же гипотез. Новых экспериментов после данных отрицательных результатов
в этой ветке не запускается.
