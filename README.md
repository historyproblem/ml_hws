# ДЗ4: прогноз исхода проверки

Ветка `codex/hw4-transformers` в репозитории [historyproblem/ml_hws](https://github.com/historyproblem/ml_hws)
содержит решение [HW4 ML MIPT autumn 2026](https://www.kaggle.com/competitions/hw-4-ml-mipt-autumn-2026).
Код и результаты HW3 сохранены в ветке `codex/full-experiment-suite`.

Цель — предсказать fail первой routine-проверки следующего квартала.
Метрика — Average Precision класса1. Текущий подтверждённый Public context-CatBoost:
**0.33982**, Private неизвестен.

## Данные

Четыре официальных CSV положить в `data/hw4/raw/`:

```text
train.csv
test_student.csv
inspection_history.csv
sample_submission.csv
```

История сопоставляется по `forecast_period + site_id`; используются только
проверки с `inspection_week < history_before` соответствующей строки.
Повторные квартальные снимки не объединяются. Идентификаторы не подаются в модель.
Исходные CSV, веса и построчные прогнозы сохраняются локально вне Git.

## Установка

Python3.12, отдельное виртуальное окружение:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r configs/hw4-neural-requirements.txt
```

На Linux-сервере с V100:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Для CPU-проверки вместо CUDA-wheel можно установить `torch==2.5.1` из обычного
индекса. Для скриптов CatBoost дополнительно:

```bash
python -m pip install -r configs/hw4-initial-requirements.txt
```

## Тесты и короткий прогон

```bash
python scripts/test_hw4.py
python scripts/run_hw4_neural.py --smoke --max-epochs 2 \
  --quarters 2023-07-01 --models mlp visit_transformer feature_transformer \
  --width 16 --batch-size 32 --out analysis/hw4-smoke
```

Тесты проверяют временные границы, принадлежность snapshot, независимость от
таргета, обработку пустой истории/padding, неизвестных категорий и train-only
preprocessing. Для smoke нужны официальные CSV; для12 тестов они не нужны.

## Запуск на V100

Следующий ещё не измеренный полный эксперимент — FT-подобный трансформер
табличных признаков:

```bash
bash scripts/run_hw4_neural_v100.sh features
```

Повторить на GPU уже выполненный CPU-пилот MLP и трансформера истории:

```bash
bash scripts/run_hw4_neural_v100.sh history
```

Wrapper сначала выводит модель GPU и VRAM и проверяет CUDA. Feature Transformer
использует batch32; история/MLP — batch256. Каждый запуск требует новой выходной
папки; её можно передать вторым аргументом wrapper. Для прямого запуска, другого
seed или иной папки доступны параметры:

```bash
python scripts/run_hw4_neural.py --help
```

Выбор эпох — на предыдущем внутреннем квартале, затем refit с нуля перед
оценкой внешнего. Dev2023Q3/Q4/2024Q1; 2024Q2/Q3 остаются закрытой проверкой.
Сохраняются кривые, OOF, веса, preprocessing, версии и SHA. Загрузка сохранённой
модели проверяется точным повторением прогноза на том же устройстве.
Скрипты нейросетевого пилота не отправляют результаты в Kaggle и не создают
финальный test-refit.

## Первые результаты

Среднее AP трёх dev-кварталов; один заранее заданный seed42, CPU:

| Модель | AP |
| --- | ---: |
| Context-CatBoost | 0.268277 |
| MLP | 0.267556 |
| Visit Transformer | 0.270649 |
| CatBoost + MLP50/50 | 0.271963 |
| CatBoost + Visit Transformer50/50 | 0.271733 |

Visit Transformer:187193 параметра,2 слоя,width64,4 головы,до16 визитов.
Прирост к CatBoost +0.002372, site-bootstrap95%CI[-0.007971,+0.013205].
Убедительное преимущество пока не установлено. Полной CV Feature Transformer
ещё нет; выполнен короткий сквозной smoke. Public0.33982 относится к CatBoost,
новые нейросетевые варианты на Kaggle не измерялись.

Агрегированные результаты: [CSV по кварталам](docs/hw4-neural-cv.csv),
[средние и bootstrap CI](docs/hw4-neural-comparison.json).

## Следующие опыты

Маскированное предобучение разных полей и связанных групп признаков: категории,
численные характеристики, контекст и последовательность проверок. Коды
нарушений — одна из групп для абляций. Это план; self-supervised обучение ещё
не реализовано и не запускалось. Supervised embeddings для downstream-модели
требуют временного cross-fitting.

Подробности: [план и результаты трансформеров](docs/hw4-transformer-plan.md),
[анализ признаков](docs/hw4-feature-analysis.md),
[первые бустинги](docs/hw4-initial-results.md),
[план соревнования](docs/hw4-competition-plan.md).
