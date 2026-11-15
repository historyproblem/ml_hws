# Отправка Joint+history 75% + item32 25%

> Архив ранее выбранного исследовательского варианта. Новое обязательное
> ограничение семинарами1–10 заменяет этот рецепт для финальной сдачи.
> Экспорт отключён; текущий маршрут — [course10-only.md](course10-only.md).

Пользователь выбрал эту смесь 15 ноября после проверки первого финального
рефита. Его Kaggle score со слов пользователя — 2.77883; ошибки экспорта,
усреднения seed, порядка Id или временного split при аудите не найдены.
Последний сообщённый остаток — четыре попытки. Эта команда не расходует их.

Выбранный вариант имеет локальный pooled Sep–Oct RMSLE 2.621432 против
2.624691 у Joint+history; чистый item32 — 2.629341. Это не прогноз November
score. Вес item32 фиксирован на 25%, по leaderboard он не подбирается.

На V100 из того же клона с готовыми `outputs/history-tuning-v1`,
`outputs/item-embeddings-v1`, `outputs/behavior-v1` и данными в `data/raw`:

```bash
git switch codex/full-experiment-suite
git pull --ff-only
bash scripts/make_item32_submission.sh
```

Скрипт проверяет и переиспользует готовые три Joint+history refit. Обучается
только item32: три seed42/123/2026, выбор эпох на October, затем fit на всех
метках с target_end <= 2018-11-01. Каждый seed включает selection и refit.
Готовые item32 refit также переиспользуются при повторном запуске. Исходники,
конфиги обучения и fingerprints прежних прогонов не меняются.

Для каждого варианта сначала берётся среднее трёх z-прогнозов. Затем
`z = 0.75*z_history + 0.25*z_item32`, ответ `maximum(expm1(z), 0)`.
Проверяются CUDA/full, HP, SHA кода/данных/артефактов, общие cache/reference,
временные границы, выбранные эпохи, все seed и соответствие сохранённых средних.
Id/customer keys сверяются с полным test; CSV — с порядком sample.

Результаты в `outputs/item-embeddings-v1/`:

* `submission_joint_history_item32_75_25.csv` — загрузить в Kaggle.
* `submission_joint_history_item32_75_25.json` — происхождение и SHA файла.
* `submission_joint_history_item32_75_25_audit.tar.gz` — прогнозы всех шести
  seed, кривые, manifests и CSV/JSON для независимой проверки; без весов/кэшей.

Через Jupyter скачайте CSV и архив. Один CSV отправляется в то же соревнование,
где уже была отправлена первая модель. Описание:
`Joint-history mean3 75% + item32 mean3 25%; log1p blend; seeds 42/123/2026`.
Скрипт не выполняет upload. Фактический score и оставшиеся попытки фиксируются
после обработки отправки. Прямого подключения к пользовательской V100 здесь нет.

Если refit готов, только экспорт:

```bash
python scripts/export_item32_submission.py
```

Первые три аргумента shell-скрипта меняют embedding output, history output,
reference соответственно. `baseline_validation.csv` — проверочная копия
старой модели, а не выбранный файл для новой отправки.
