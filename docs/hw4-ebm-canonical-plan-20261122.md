# EBM: greedy scheduler и bounded canonical bagging, 22 ноября 2026

Отдельное продолжение после отрицательного cyclic8-screen. Его source, freeze и результаты не меняются. Новая проверка закрывает две конкретные недопроверенные части настоящего EBM: greedy scheduler и внутренний bagging с early stopping. Это заранее согласованный один алгоритм выбора среди четырёх рецептов, а не четыре алгоритма с последующим выбором по outer AP.

## Источник и runtime

Используется официальный `interpret-core==0.7.8`, установленный wheel-only и `--no-deps` в изолированный `analysis/hw4-six-hour-20261122/ebm-runtime`. Основные NumPy/SciPy/sklearn/Torch не изменены. Native arm64-библиотека загружается и хешируется. API и defaults проверены в [конструкторе upstream 0.7.8](https://github.com/interpretml/interpret/blob/v0.7.8/python/interpret-core/interpret/glassbox/_ebm/_ebm.py), а поведение scheduler — в [upstream training loop](https://github.com/interpretml/interpret/blob/v0.7.8/python/interpret-core/interpret/glassbox/_ebm/_boost.py). `greedy_ratio` управляет обучающим gain при выборе обновляемого терма; он не означает автоматический отбор interactions.

## Четыре рецепта, замороженные до HW4-fit

Все используют compact94: точные static + burden24 + ctx72, без `forecast_year` и `cutoff_gap_days`; квартал — номинальная категория. Raw ID, текущий target и даты не входят. Категориальный словарь обучается только на FIT; неизвестная категория становится missing, числовой ноль сохраняется. Missing обрабатывает native EBM.

| Рецепт | Термы | Scheduler | Обучение |
| --- | --- | --- | --- |
| A: `ebm_match_main_300` | 94 main effects | greedy10/cyclic_progress=False | прежний controlled recipe, 300 cycles, 1 bag, без internal validation/ES |
| B: `ebm_match_pairs10_300` | 94 main + те же 10 фиксированных пар | тот же | тот же |
| C: `ebm_canonical_main` | 94 main effects | тот же | LR .015, bins1024, min_leaf4, L2=0, 8 bags, validation .15, cap2000, ES50, tolerance1e-5 |
| D: `ebm_canonical_pairs10` | 94 main + те же 10 пар | тот же | как C, interaction bins64 |

A/B отличаются от прежних соответствующих compact300-рецептов **только greedy_ratio=10**. Остальные параметры A/B: LR .03, main bins128, interaction bins16, min_leaf20, L2=1, max_leaves2, smoothing75 в обеих стадиях. C/D — bounded canonical-style adaptation: upstream defaults LR .015 / bins1024 / interaction64 / min_leaf4 / lambda0 / max_leaves2 / smoothing75; явно ограничены bags8, cap2000 и ES50. Это не заявление о полном совпадении с upstream default (там bags14, cap50000, ES100 и другой scope interactions).

Фиксированные пары те же, что в предыдущем screen: own burden с risk/facility/area/quarter и own history mean с `ctx_area_365d_routine_load_smooth`. Автоматического поиска пар нет.

## Время, выбор и запреты

Native SYNTHETIC probe до первого HW4-fit: 1000 строк × 94 selected features, C/D со всеми реальными cap2000/bags8/ES50: main 3.07 с, pairs 2.43 с; finite probabilities и reload EXACT. Это проверка совместимости и ограниченности бюджета; не обещание линейного времени на HW4. Результат и SHA — `analysis/hw4-six-hour-20261122/ebm-canonical-native-probe.json`.

Двенадцать INNER-fit (4 рецепта × 3 строго предыдущих квартала) заканчиваются до первого OUTER-fit. Для old outer 2023Q3/Q4/2024Q1 выбирается рецепт по AP предыдущего квартала; ties: ниже logloss, ниже cap, затем recipe ID. Далее один алгоритм имеет 3 seed42 OUTER-fit. Максимум 24 fit: 12 inner + 3 outer + при допуске 6 подтверждающих seed17/2026 + 3 full. Все FIT имеют даты строго раньше внешнего оцениваемого квартала. Финальный full recipe выбирается по среднему трёх OLD INNER AP; никаких новых Q2/Q3-метрик для настройки.

Внутренний random15%-split C/D находится **только внутри уже прошлых FIT-строк**. Это часть разрешённого алгоритма, не временная валидация. Native bins видят весь этот past FIT pool, как в upstream. Внешний inner/outer DataFrame и его target в native fit не передаются. При разрешённом полном fit доступны все train labels, включая Q2/Q3; internal ES на mixed-past pool нельзя выдавать за отдельную оценку этих периодов.

Прогнозы заранее фиксированы: raw EBM и `.75 * previous07 + .25 * EBM`, усреднение вероятностей seed42/17/2026. Gate к previous07: mean ΔAP > 0, минимум 2/3 кварталов неотрицательны, worst ΔAP ≥ −.005. После трёх seed тот же gate. Если и raw, и fixed blend проигрывают новому GAM во всех трёх кварталах и каждый средний gap не меньше .002, full fit запрещён. OOF сохраняется независимо от допуска.

`best_iteration_[stage,bag]` сохраняется как actual attempted update counts; это не число деревьев и не доступный score-prefix. Модели сохраняются атомарно; предсказания после joblib reload обязаны совпасть EXACT. Freeze включает code/runtime/native/raw/cache/anchor fingerprints; strict resume не принимает несовпадение или повреждённые артефакты. Это adaptive исследование на многократно использованных old dev, интервалы nominal; independent validation gain не заявляется. Kaggle uploads: 0.
