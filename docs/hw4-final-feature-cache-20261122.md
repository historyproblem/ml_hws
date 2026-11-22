# Воспроизведение признаков HW4

Из корня checkout четыре официальных файла должны находиться прямо в `data/`:
`train.csv`, `test_student.csv`, `inspection_history.csv`, `sample_submission.csv`.
В существующем Python-окружении:

```bash
python scripts/build_hw4_final_feature_cache.py \
  --raw data --out analysis/final-features
```

Результат: `analysis/final-features/features.joblib` и `features-meta.json`.
Это построение признаков без официального target текущей строки. Известные
исторические исходы используются строго до cutoff в соответствующем snapshot.
Обучений, AP и новых метрик Q2/Q3 здесь нет.

Сохранённый cache не перезаписывается. Для проверки и повторного использования:

```bash
python scripts/build_hw4_final_feature_cache.py \
  --raw data --out analysis/final-features --resume
```

Изменение CSV, исходников, окружения или байтов cache отклоняет resume.
Для другого набора входов требуется новая выходная папка.

Четырёх CSV достаточно для cache-builder. Модельным runner дополнительно нужны
проверенные локальные `baseline_oof.csv` и `submission_04_context_cb.csv`,
передаваемые через `--baseline-cv` и `--context-test`. Builder их не создаёт;
чистый сервер только с четырьмя CSV ещё не готов к полному runner.

На текущем Маке Python расположен в
`/Users/khlystov/Documents/ChatGPT/ML ДЗ/.venv/bin/python`, данные — в
`/Users/khlystov/Documents/ChatGPT/ML ДЗ/data/hw4/raw`. Проверенные anchor-файлы:

- `/Users/khlystov/Documents/ChatGPT/ML ДЗ/analysis/hw4-next-round-20261121/baseline_oof.csv`;
- `/Users/khlystov/Documents/ChatGPT/ML ДЗ/outputs/hw4/final-20261120/submission_04_context_cb.csv`.

Для нативных OpenMP-моделей на Маке, если runner не находит `libomp.dylib`,
используется уже установленная библиотека Torch:

```bash
export HW4_OPENMP="/Users/khlystov/Documents/ChatGPT/ML ДЗ/.venv/lib/python3.12/site-packages/torch/lib"
export DYLD_LIBRARY_PATH="$HW4_OPENMP${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Сам cache-builder CatBoost/XGBoost не импортирует и OpenMP-настройки не требует.

QA от 22 ноября: отдельный rebuild в `qa/cache-rebuild` занял 5.73 секунды.
Все значения, dtype, индексы, порядок колонок трёх raw frames и трёх views,
а также десять категориальных имён EXACT совпали с исследовательским cache:
23 263 строки; base346, transition353, seasonal358. Байты сериализации joblib
отличаются; фактические данные совпадают точно. Resume не перестраивает
признаки, изменённые fingerprints и перезапись существующего cache отвергаются.
Исходный исследовательский cache не изменён. Аудит:
`analysis/hw4-deep-research-20261122/qa/cache-rebuild-verification.json`.
