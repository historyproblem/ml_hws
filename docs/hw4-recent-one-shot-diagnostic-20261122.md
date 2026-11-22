# Однократная диагностика зафиксированных финалистов

2024Q2/Q3 уже использовались20 ноября. Пользователь поручил дополнительно
проверить новые заранее зафиксированные модели. Этот инструмент реализует
одну описательную временную проверку и не называет её новым независимым holdout.
Параметры и веса по её результатам не настраиваются.

## Готовые базовые прогнозы без метрик

`scripts/prepare_hw4_recent_baselines.py` подготовил4101 строку:2054 за2024Q2
и2047 за2024Q3. Файл на текущем Маке:

```text
/Users/khlystov/Documents/ChatGPT/ML ДЗ/analysis/hw4-six-hour-20261122/recent-baselines/predictions.csv
```

После `row_id,site_id,forecast_period,target` в нём находятся:

| Колонка | Рецепт |
|---|---|
| `context_3seed` | Среднее вероятностей прежнего context CatBoost42/17/2026 |
| `old_gam_core` | Прежний фиксированный GAM C=.1 |
| `previous07` | .75×context+.25×old_gam_core |
| `fixed_gam_interactions_c001` | Зафиксированный полный рецепт нового GAM: взаимодействия, C=.01 |

Шесть context checkpoint из `outputs/hw4/final-20261120/holdout_<quarter>/`
повторно использованы без fit. Проверены исходная подпись38eaf418…, RAW/source
SHA,500 деревьев, native-параметры, feature schema и точный forward на текущем
кеше. Последовательности train/predict равны16916/2054 и18970/2047.

Две фиксированные E-модели взяты из `additive/gam-baseline-residual/teacher/`:
исходная подписьc3b094…, те же данные и строго предыдущие кварталы. Saved IDs,
raw logits и forward проверены точно. `expit(raw)` совпал с нативными
вероятностями модели. Полный21-тысячный fit для holdout не используется.

Только отсутствовавшие два старых GAM C=.1 обучены заново. Сохранены исходные
transform/classifier-параметры,1500-итерационный предел и два CPU-потока.
Все собственные checkpoints точно воспроизводятся после загрузки. Повторный
запуск под запретом новых fit вернул тот же manifest байтово.

SHA `predictions.csv`:
`25b995e3997907673e1a2ae60bcb03f83b588d76172f31926fcc5e66449f0ae5`.
Подробности — `recent-baselines/{freeze,manifest,reuse-proof,verification}.json`.
**AP и другие метрики этих4101 строк пока не рассчитывались.**

## Формат зафиксированного списка

Root сначала выбирает конечный список по старому development и готовит CSV
с точными официальными recent-строками. Помимо четырёх metadata-колонок CSV
должен содержать ровно перечисленные в JSON колонки вероятностей. Формулы
framework не исполняет и не перебирает: он фиксирует уже полученные значения.

Минимальный пример request:

```json
{
  "format": "hw4-recent-finalists-1",
  "reference": "context_3seed",
  "formulas_frozen": true,
  "hyperparameter_search": false,
  "weight_search": false,
  "selection_explanation": "Финалисты выбраны по прежнему development до этой проверки",
  "finalists": [
    {
      "id": "context_3seed",
      "recipe": {"trees": 500, "depth": 5, "seeds": [42, 17, 2026]},
      "formula": "mean_probability_three_seeds",
      "sources": [{"path": "/absolute/path/to/source.py", "sha256": "source_digest"}],
      "old_selection_evidence": ["/absolute/path/to/earlier-report.json"]
    },
    {
      "id": "frozen_gam",
      "recipe": {"id": "core_interactions_C0.01"},
      "formula": "fixed_recipe_probability",
      "sources": ["/absolute/path/to/additive_search.py"],
      "old_selection_evidence": ["/absolute/path/to/final-recipe.json"]
    }
  ]
}
```

При указанном SHA он проверяется; без него вычисляется текущий SHA. Пути
без начального `/` разрешаются относительно request-файла. `recipe` описывает
конкретный полный рецепт, а не приблизительный CV алгоритма с разными рецептами.
Для GAM нужно различать0.276997 у последовательного алгоритма и зафиксированный
`core_interactions_C0.01`, который реально использован в полном fit.

## Два отдельных шага

```bash
python scripts/audit_hw4_recent_finalists.py freeze \
  --raw data --recipes final-recipes.json --predictions final-recent.csv \
  --out analysis/recent-finalist-diagnostic
```

Этот шаг не вычисляет AP. Он связывает подписью четыре RAW SHA, точные ID,
список рецептов/формул, source и evidence SHA, исходный и канонический prediction
SHA, версии scorer и его окружения. Изменившиеся входы отклоняются.

Только после проверки замороженного списка:

```bash
python scripts/audit_hw4_recent_finalists.py score \
  --out analysis/recent-finalist-diagnostic
```

Результат: `results/quarter-metrics.csv`, `summary.csv`, `comparison.json`,
`complete.json`. AP рассчитывается отдельно на каждом квартале; среднее имеет
равные квартальные веса.2000 парных bootstrap поsite_id используют одни и те же
кратности площадок для всех моделей и обоих кварталов. Интервалы номинальные,
без поправки на предшествующее адаптивное исследование. Автоматического winner,
гейта, настройки моделей или весов и загрузки в Kaggle здесь нет.

Повторная команда `score --resume` только проверяет хеши и возвращает уже
завершённые результаты, без AP/bootstrap. Изменение входов, рецептов, исходников
или результатов запрещает такой resume. Если scoring уже начат, но атомарная
публикация не завершилась, автоматическое повторное вычисление блокируется;
прерванные артефакты требуется сначала разобрать явно.

Synthetic QA: четыре проверки framework и одна native-проверка подготовки
baseline прошли, включая неизменный resume, запрет подмены snapshot, corrupted
output и fail-closed прерывание. Реальная однократная диагностика ещё не запущена.
