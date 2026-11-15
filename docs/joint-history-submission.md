# Отправка версии перед эмбеддингами

> Архив прежней исследовательской версии. После ограничения семинарами1–10
> этот Joint+mean3 не входит в строгий финальный рецепт. Текущий маршрут:
> [course10-only.md](course10-only.md). Старый запускатор отключён.

Выбран **Joint+history mean3**, исходные HP: lr=0.001, L2=0.0001,
dropout=0, batch1024, dense128/64, seed42/123/2026. Его проверочный
pooled Sep–Oct RMSLE — **2.624691**. Вариант lr=0.002 не улучшил подтверждение;
эмбеддинги не входят в эту отправку.

Для тестовой даты требуется отдельный GPU-рефит: присланные архивы содержали
validation-прогнозы. Финальный файл строится по всей доступной к 2018-11-01
истории, после проверки происхождения всех трёх seed.

## Одна команда на V100

Из корня того же клона с `outputs/history-tuning-v1`, `outputs/behavior-v1`
и исходными CSV в `data/raw`:

```bash
git switch codex/full-experiment-suite
git pull --ff-only
python -m pip install -e .
bash scripts/make_joint_history_submission.sh
```

Команда выполняет только финальный refit baseline Joint+history: эпохи
выбираются на октябре, затем три seed обучаются на всех доступных к
2018-11-01 метках. Повторный вызов использует завершённые проверенные refit
jobs. HP-подбор и embedding-эксперименты не запускаются.

Затем создаётся файл:

```text
outputs/history-tuning-v1/submission_joint_history_mean3.csv
```

Скрипт проверяет полный CUDA-режим без smoke, исходные параметры, SHA кода,
данных/артефактов, все три seed, temporal cutoff и выбранные эпохи. Усреднение
делается в log1p-пространстве, затем `maximum(expm1(z),0)`. Проверяются все
3188 тестовых Id, customer keys, точный порядок `sample_submission.csv`,
две колонки `Id,spend_30d`, конечность и неотрицательность. Рядом сохраняется
JSON с происхождением файла и SHA256.

Если refit уже выполнен, можно только экспортировать CSV:

```bash
python scripts/export_joint_history_submission.py
```

Для других output/reference путей они передаются первым/вторым аргументами
shell-скрипта. Конвертер также принимает `--run-dir`, `--reference`,
`--data-dir`, `--output`.

## Загрузка

В файловом браузере Jupyter откройте `outputs/history-tuning-v1`, скачайте
`submission_joint_history_mean3.csv`. На [странице соревнования](https://www.kaggle.com/competitions/hw-3-ml-mipt-autumn-2026)
выберите Submit Predictions, загрузите этот CSV и отправьте один файл.
Описание: `Joint+history mean3; seeds 42/123/2026; lr=0.001; no embeddings`.
Это стандартная [загрузка CSV в Kaggle](https://www.kaggle.com/docs/competitions).
API-токен и установка Kaggle CLI для этого не нужны.

Скрипт формирует и проверяет файл; сам upload не выполняет. После загрузки
надо дождаться обработки и записать фактический Kaggle score. Локальный
RMSLE не является обещанием leaderboard score.
