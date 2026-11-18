# HW4: residual MLP, числовые эмбеддинги и история

19 ноября 2026. Новый GPU-экран подготовлен; полный результат ещё не получен.
50 проверок прошли. Small-smoke всех12 внешних job и ещё4 job с реальной
архитектурой прошли; прогнозы и embeddings после reload EXACT. Resume12job
без fit, изменённый config и изменённые reference-прогнозы отклонены.
Интерфейс классического источника дополнительно проверен на явно синтетической
QA-опоре; она не является новым результатом качества.
[Машинный QA-протокол](hw4-residual-qa.json).
Его цель — проверить представление признаков и residual-связи, затем оценить
добавку к уже обученным классическим моделям.

## Четыре контролируемых варианта

Все модели используют один и тот же компактный набор96 признаков: точные
24 static/calendar/burden плюс72 context. Статические категории получают
обучаемые8-мерные embeddings. Идентификаторы row_id/site_id не подаются.

| Модель | Представление и архитектура | Что сравниваем |
| --- | --- | --- |
| mlp_control | стандартизованные числа + missing flags + category embeddings;3 блока width128 | новый сопоставимый контроль |
| resnet | те же входы, параметры и начальные веса; в каждом блоке добавляется skip | эффект residual-связей |
| ple_resnet | каждый числовой признак получает обучаемый8-мерный embedding кусочно-линейного представления; тот же ResNet | эффект numeric embeddings |
| ple_visit_transformer | тот же PLE-ResNet + отдельный Transformer прошлых визитов | дополнительный сигнал истории |

В MLP/ResNet блок: LayerNorm→Linear128→256→GELU→Dropout.15→Linear256→128→Dropout.15.
ResNet прибавляет выход блока к его входу. Plain-контроль делает те же вычисления
без этого прибавления; тест подтверждает совпадение исходных параметров.
Это новая пара контролей, не повтор старого MLP346.

Числовые embeddings: до16 сегментов с границами по квантилям только fit-данных.
После удаления повторных границ каждый сегмент кодируется clip((x-left)/(right-left),0,1).
Далее отдельная обучаемая проекция16→8 для каждого признака. Natural missing
получает нулевой basis и отдельный missing flag. Константы и all-missing поля
не создают деление на ноль; неизвестная категория имеет индекс0.

История: до16 визитов, строго до cutoff и в точном forecast snapshot.
44 непрозрачных кода нарушений получают16-мерные embeddings, внутри визита
усредняются; пустой набор имеет отдельный вектор. Тип/результат проверки получают
8-мерные embeddings, возраст/gap/count/blank — отдельную проекцию. Это подаётся
в Transformer width64,2 слоя,4 головы,pre-norm,FFN128,CLS и padding mask.
Padding зануляется до проекций; даже NaN в исключённых слотах не влияет на ответ.
Смысл кодов из внешнего словаря не предполагается.

В прежних Transformer residual-связи уже были. Отличия нового опыта —
контролируемая MLP/ResNet-пара, numeric PLE и представление визитов через
кодовые/type/result embeddings. Masked-pretraining не повторяется.

Это собственные компактные адаптации идей из
[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)
и [On Embeddings for Numerical Features](https://arxiv.org/abs/2203.05556),
а не точное воспроизведение их benchmark-конфигураций. Их результаты
не доказывают заранее прирост на HW4.

## Обучение и сохранение

Seed42, batch256, AdamW lr.001/weight_decay.01, обычная BCE, clip_norm1.
Внутренний максимум60 эпох,patience10. Dev прежние2023Q3/Q4/2024Q1.
Число эпох выбирается по AP предыдущего inner-квартала; затем новая модель
обучается на всех более ранних строках ровно выбранное число эпох.
Outer-labels не передаются в training-функцию. Всего12 внешних+12 внутренних fit.
Словари, normalization и числовые границы заново fit только на раннем train.
Последние2024Q2/Q3 отбрасываются до признаков/выбора/оценки.

CUDA проверяется явно, модель и тензоры размещаются на GPU. Используется
math-only attention и deterministic algorithms; silent CPU fallback нет.
Лог содержит стадии CPU-признаков, cuda_ready, fit_start, first_batch,
каждую эпоху с loss/AP и CUDA memory, затем fit_complete и complete.

Каждый job сохраняет веса, preprocessing, inner/outer curves, OOF и
validation_embeddings.npz с row_id. Это supervised embeddings: их нельзя
просто подать в CatBoost на тех же обучающих строках. Для отдельного
обучения поверх них нужен temporal cross-fitting; в этом экране downstream-fit нет.
Reload модели+preprocessing обязан точно повторить outer-прогноз.
Завершённые job пропускаются при проверенном --resume, изменённые SHA/config/
версии/источник классических прогнозов отклоняются.

## Использование готового классического результата

По предоставленному пользователем логу классический экран закончен за485.69с:
27 job,без test-refit/upload. Лучший — CB+compact rank50/50 AP.273134,
против нового reference.268611,delta+.004523,nominalCI[-.000916,+.009894].
Полный архив ещё не получен для независимого аудита. Старый reference.268277
не совпадает с новым; причина пока не установлена. Проверка кода признаков,
схемы и старого Q3-forward показала EXACT, но не объяснила серверное обучение.
Нельзя напрямую объявлять прежний CB+VisitTF.273290 лучше новой смеси.

Wrapper читает существующий analysis/hw4-classical-01 на том же сервере.
Он проверяет completed/non-smoke protocol, input SHA, исходники/спецификации
reference/compact и SHA их job-прогнозов, сверяет официальные ID/targets.
Классические модели не переобучаются и pickle-файлы не загружаются.
К новым OOF применяются предзаданные50/50 probability/rank смеси с тем же
серверным reference, поэтому сравнение будет согласованным внутри нового отчёта.
Состав/веса по Public не подбираются; stacking отсутствует.

## Запуск на V100

На прежнем сервере:

```bash
cd ~/ml_hws_hw4
source .venv/bin/activate
git fetch origin
git merge --ff-only origin/codex/hw4-transformers
python -m pip install -r configs/hw4-residual-requirements.txt
python scripts/test_hw4.py
mkdir -p logs
nohup bash scripts/run_hw4_residual_v100.sh \
  analysis/hw4-residual-v100-01 data analysis/hw4-classical-01 \
  > logs/hw4-residual-v100-01.log 2>&1 &
```

Имеющийся torch2.5.1+cu121 переустанавливать не нужно. CSV остаются прямо вdata/.
Последний аргумент — папка уже завершённого классического экрана, не архив.
Для просмотра: `tail -n 50 -f logs/hw4-residual-v100-01.log`.
Ctrl-C останавливает tail, обучение продолжается.

После завершения вернуть **analysis/hw4-residual-v100-01/return_results.tar.gz**.
Он содержит protocol, OOF, фолдовые и ансамблевые метрики, диагностику и сравнение.
Веса/embeddings остаются на сервере в job-подпапках. Дополнительно нужен прежний
**analysis/hw4-classical-01/return_results.tar.gz** для независимого разбора опоры.

CPU smoke без классических файлов:

```bash
python scripts/run_hw4_residual.py --device cpu --smoke \
  --max-epochs 2 --width 16 --blocks 1 --bins 4 --batch-size 32 \
  --bootstrap 10 --out analysis/hw4-residual-smoke
```

Small-smoke AP не является оценкой качества. После полноценного результата
подтверждаем только полезные ветки дополнительными seed и фиксируем ансамбль
до открытия2024Q2/Q3; конечные CSV и Kaggle-upload в этом экране отсутствуют.
