# ДЗ4: прогноз исхода проверки

Ветка `codex/hw4-transformers` в репозитории [historyproblem/ml_hws](https://github.com/historyproblem/ml_hws)
содержит решение [HW4 ML MIPT autumn 2026](https://www.kaggle.com/competitions/hw-4-ml-mipt-autumn-2026).
Код и результаты HW3 сохранены в ветке `codex/full-experiment-suite`.

Цель — предсказать fail первой routine-проверки следующего квартала.
Метрика — Average Precision класса1. Текущий подтверждённый Public context-CatBoost:
**0.33982**, Private неизвестен.

## Данные

Четыре официальных CSV положить непосредственно в `data/`:

```text
train.csv
test_student.csv
inspection_history.csv
sample_submission.csv
```

История сопоставляется по `forecast_period + site_id`; используются только
проверки с `inspection_week < history_before` соответствующей строки.
Повторные квартальные снимки не объединяются. row_id не подаётся в модель;
site_id допускается только в отдельной
регуляризованной ветке нового классического экрана.
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

Проверки включают временные границы, принадлежность snapshot, независимость от
таргета, обработку пустой истории/padding, неизвестных категорий и train-only
preprocessing, маскирование и предобучение. Для smoke нужны официальные CSV;
для тестов они не нужны.

## Новый GPU-экран: residual MLP и embeddings (19 ноября)

Четыре варианта: matched MLP/ResNet, ResNet с числовыми PLE-эмбеддингами,
тот же ResNet с кодовыми/type/result-эмбеддингами и Transformer истории.
Все используют96 табличных признаков. Полного результата пока нет.

```bash
python -m pip install -r configs/hw4-residual-requirements.txt
bash scripts/run_hw4_residual_v100.sh \
  analysis/hw4-residual-v100-01 data analysis/hw4-classical-01
```

Использует V100; читает готовые OOF классического экрана без его переобучения.
Данные должны быть непосредственно в `data/`. После завершения вернуть
`analysis/hw4-residual-v100-01/return_results.tar.gz`.
[Архитектуры, проверки и фоновый запуск](docs/hw4-residual-embeddings.md).

## Классический экран (18 ноября)

Завершён по логу пользователя: лучший CB+compact rank50 AP0.273134,
reference0.268611,95%CI дельты[-0.000916,+0.009894]. Архив для независимой
проверки ещё не получен. CatBoost/LightGBM/ExtraTrees, компактные признаки,
ID/давность и фиксированные ансамбли.
На сервере он использует CPU; свободная V100 в этом запуске ожидаема.

```bash
python -m pip install -r configs/hw4-classical-requirements.txt
bash scripts/run_hw4_classical_server.sh analysis/hw4-classical-01 data
```

9 моделей,3 временных фолда, проверяемый resume. По завершении вернуть
`analysis/hw4-classical-01/return_results.tar.gz`.
[Полный протокол, фоновый запуск и порядок следующих этапов](docs/hw4-classical-ensemble.md).
Текущий Public0.33982 сохраняется; последние2024Q2/Q3 не открываются.

## Запуск на V100

Завершённые опыты FT/истории можно воспроизвести следующими командами;
для нового экрана использовать раздел выше. FT-подобный трансформер:

```bash
bash scripts/run_hw4_neural_v100.sh features
```

Повторить на GPU уже выполненный CPU-пилот MLP и трансформера истории:

```bash
bash scripts/run_hw4_neural_v100.sh history
```

Wrapper сначала выводит модель GPU и VRAM и проверяет CUDA. Feature Transformer
использует batch32; история/MLP — batch256. Каждый запуск требует новой выходной
папки; её можно передать вторым аргументом wrapper. Папка CSV по умолчанию —
`data/`, другую можно передать третьим аргументом:

```bash
bash scripts/run_hw4_neural_v100.sh features analysis/hw4-ft-v100-02 data
```

При прямом запуске Python путь задаётся через `--raw`. Для другого
seed или иной папки доступны параметры:

```bash
python scripts/run_hw4_neural.py --help
```

В логе сначала идут `cuda_ready` и этапы CPU-подготовки CSV/признаков.
Visit sequences строятся только для Visit Transformer. Затем `fit_start`
показывает устройства модели/тензоров, а `first_batch_complete` — устройство
реального результата обучения. Для каждой эпохи выводятся время и GPU-память.
После итоговой таблицы процесс завершён, поэтому GPU снова свободна.

Выбор эпох — на предыдущем внутреннем квартале, затем refit с нуля перед
оценкой внешнего. Dev2023Q3/Q4/2024Q1; 2024Q2/Q3 остаются закрытой проверкой.
Сохраняются кривые, OOF, веса, preprocessing, версии и SHA. Загрузка сохранённой
модели проверяется точным повторением прогноза на том же устройстве.
Скрипты нейросетевого пилота не отправляют результаты в Kaggle и не создают
финальный test-refit.

## Первые результаты

Среднее AP трёх dev-кварталов; один заранее заданный seed42.
MLP/Visit Transformer измерены на CPU, Feature Transformer — на V100:

| Модель | AP |
| --- | ---: |
| Context-CatBoost | 0.268277 |
| MLP | 0.267556 |
| Visit Transformer | 0.270649 |
| Feature Transformer, V100 | 0.259700 |
| CatBoost + MLP50/50 | 0.271963 |
| CatBoost + Visit Transformer50/50 | 0.271733 |

Visit Transformer:187193 параметра,2 слоя,width64,4 головы,до16 визитов.
Прирост к CatBoost +0.002372, site-bootstrap95%CI[-0.007971,+0.013205].
Убедительное преимущество Visit Transformer пока не установлено.
Feature Transformer завершил3 фолда на V100:0.266667/0.257944/0.254488,
выбраны эпохи1/9/4; текущая конфигурация слабее context-CatBoost.
Это результаты предоставленного пользователем лога; построчные прогнозы GPU
ещё не проверены отдельно. Public0.33982 относится к CatBoost,
новые нейросетевые варианты на Kaggle не измерялись.

Агрегированные результаты: [CSV по кварталам](docs/hw4-neural-cv.csv),
[средние и bootstrap CI](docs/hw4-neural-comparison.json).
Отдельно: [логовые результаты FT на V100](docs/hw4-ft-v100-results.json).

## Следующие опыты

Маскированное предобучение разных групп уже реализовано и проверено на smoke.
Полная CV завершена: supervised0.262575, masked0.258064, CB+SSL64=0.261880.
Эта конфигурация не улучшила context-CatBoost0.268277.
[Полный разбор и интервалы](docs/hw4-masked-results.md).
Для воспроизведения один запуск сравнивает групповую сеть с нуля,
ту же сеть с предобучением и CatBoost с64 замороженными SSL-признаками:

```bash
python -m pip install -r configs/hw4-masked-requirements.txt
bash scripts/run_hw4_masked_v100.sh analysis/hw4-masked-v100-01 data
```

Подробный протокол, запуск в фоне и resume:
[маскированное предобучение](docs/hw4-masked-pretraining.md).
Предобучение ограничено прошлым fit; SSL-признаки фиксируются до обучения
на таргете. Для supervised embeddings downstream-модели нужен cross-fitting.

Подробности: [план и результаты трансформеров](docs/hw4-transformer-plan.md),
[анализ признаков](docs/hw4-feature-analysis.md),
[первые бустинги](docs/hw4-initial-results.md),
[план соревнования](docs/hw4-competition-plan.md).
