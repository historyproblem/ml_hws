# Зафиксированный EBM: прогнозы последних кварталов без метрик

`scripts/prepare_hw4_recent_ebm.py` готовит только прогнозы для будущей
однократной диагностики. Параметры взяты из уже завершённого старого поиска:
`ebm_canonical_pairs10`, выбранный по среднему трёх OLD INNER AP. На2024Q2/Q3
не выбираются ни параметры, ни веса, ни число взаимодействий.

Зафиксированы compact94, десять заданных пар, greedy_ratio10,
learning_rate.015, max_bins1024, max_interaction_bins64, min_samples_leaf4,
reg_lambda0, max_rounds2000, восемь native bags и ES50.15% случайной validation
для каждой bag находится **внутри переданных прошлых fit-строк**. Внешняя
предсказываемая четверть и её target в fit не передаются.

Перед запуском проверены parent freeze4a82ec…, old-inner selection,
три полных parent-модели, текущие исходники, RAW/cache и изолированный
interpret-core0.7.8 вместе с native library SHA. Готовых recent-моделей
не было. Обучены ровно шесть моделей:42/17/2026 для каждого из двух кварталов,
не более двух CPU jobs. Full-train модель не используется для holdout.

| Прогноз | Fit-строки | Максимальный fit-период | Query-строки |
|---|---:|---|---:|
|2024Q2|16916|2024Q1|2054|
|2024Q3|18970|2024Q2|2047|

Готовый файл на текущем Маке:

```text
/Users/khlystov/Documents/ChatGPT/ML ДЗ/analysis/hw4-six-hour-20261122/recent-ebm/predictions.csv
```

Колонки: `row_id,site_id,forecast_period,target,ebm_fixed_pairs_mean3`.
Всего4101 уникальных официальных строк в порядке Q2, затем Q3.
`components.csv` дополнительно содержит три отдельные вероятности seed.
Итог — их равное арифметическое среднее. Смеси здесь не создаются.

SHA256 прогноза:
`970b2f91a5447d074b25230e50c81cdc6e44af8431b3a878a954d24ada696ddc`.
**AP, сравнения и иные внешние метрики этих строк не рассчитывались.**

Каждый checkpoint после загрузки воспроизвёл forward побитно. Независимая
проверка повторила все шесть forward, mean3 и официальный порядок ID.
`--resume` под запретом любого fit вернул неизменный manifest. Источник,
параметры, байты модели/вероятностей или родительских артефактов при изменении
отклоняют resume. Доказательства сохранены в
`recent-ebm/{freeze,manifest,verification}.json` и markers моделей.

Для повторной проверки:

```bash
python scripts/prepare_hw4_recent_ebm.py \
  --raw data --feature-cache analysis/final-features/features.joblib \
  --parent-run analysis/ebm-canonical --runtime analysis/ebm-runtime \
  --out analysis/recent-ebm --resume
```

На существующих текущих артефактах надо использовать их фактические пути,
записанные в freeze. Synthetic native QA проверил strict-past fit без eval,
сохранение/загрузку, отсутствие нового fit на resume и отказ при порче payload.
