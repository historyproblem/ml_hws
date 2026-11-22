# HW4: первый ограниченный EBM-экран

22 ноября 2026. Прогон восьми заранее зафиксированных циклическихEBM-рецептов завершён наMac за180.8 секунды. **27 реальныхfit:24 внутренних и3внешних.** Gate против прежней07 не пройден; подтверждений,полныхfit,новыхCSV иотправок нет.

Изолированно установлен только официальныйwheel `interpret-core==0.7.8` в `analysis/hw4-six-hour-20261122/ebm-runtime/`,с`--no-deps`. Версии numpy/pandas/scipy/scikit-learn/Torch/joblib основного окружения не менялись. ДоHW4 проверены настоящийMacARMnativefit,категории/NaN/0,точныйreload инеизменностьbins при будущих экстремумах. Девятьsyntheticпроверок backend/runner прошли.

## Зафиксированная область

Дваview,compact94 иno-code168,не содержатyear/cutoff-gap;quarter —nominal. Main effects либо10 **конкретных заранее заданных** пар;никакогоautomaticinteractionsearch.300/800циклов —отдельныеfixed fits,а не префиксы. Validation0,early_stopping0,outer_bags1,inner_bags0,learning_rate.03,max_bins128,interactionbins16,minleaf20,reg_lambda1,maxleaves2,greedy_ratio0. Для каждойouter2023Q3/Q4/2024Q1 конфигурация выбиралась только на предыдущемinner. Все24внутреннихfit завершились до первого внешнего.

В0.7.8 `greedy_ratio=0` внутренне принудительно включает циклическийprogress,даже если публичныйаргумент `cyclic_progress=False`. Это отмечено вbundle/protocol. Числа `best_iteration_[stage,bag]` —активныеterm-update steps,**не число деревьев и не API префикса**. [Исходныйалгоритм](https://github.com/interpretml/interpret/blob/v0.7.8/python/interpret-core/interpret/glassbox/_ebm/_boost.py).

## Результат на старых development-кварталах

| Predictor | MeanAP | Δ к07 | Кварталы сΔ≥0 |
|---|---:|---:|---:|
|НовыйGAM другойветки,диагностическаяопора|0.276997|+0.006157|3/3|
|Прежняя07,основнаяопора|0.270840|0|3|
|75%07+25%EBM,вероятности|0.268954|−0.001886|1/3|
|EBM algorithm raw|0.251703|−0.019138|0/3|

Raw AP по кварталам0.261766/0.274812/0.218530. Усмеси дельты к07−0.001493/+0.002099/−0.006264. НовыйGAMraw превосходит обаEBM-вектора во всех трёх кварталах. Граница worst-quarter≥−.005 у смеси также нарушена. Ни стандартный07gate,нисодержательнаяконкурентоспособность относительноGAM не подтверждены.

Внешнийалгоритм выбрал compact94/main300 дляQ3 иQ1,compact94/pairs300 дляQ4. Внутренне агрегированныйfull-рецепт был быcompact94/main300,ноgate не разрешилfull-fit. Сохранены всеOOF для независимыхдиагностик/ансамблей;исходныйprotocol не переписывался.

24inner+3outerjob проверены поSHA,reload-flags;все15поквартальныхAP независимо воспроизведены с ошибкой<1e−14. Смесь07/EBM реконструирована EXACT. Это reusedolddev с номинальнымиsite-bootstrap2000интервалами,не свежая независимаявалидация. Новых2024Q2/Q3-метрик нет.

## Что это не проверило

Этот экран **не исчерпал официальныйEBM**. Upstreamdefault `greedy_ratio=10` выбираетобновляемыйтерм поregularizedtraininggain;он не ищетновыеinteractionпары и не обращается к будущимinner/outer-меткам. При300циклах пределактивныхобновлений тотже,ноихраспределение ивычислительныеиздержки отличаются. Поэтомуnegativecyclic0 не доказываетнепригодностьgreedyEBM. [ТренировочныйgainвC++](https://github.com/interpretml/interpret/blob/v0.7.8/shared/libebm/GenerateTermUpdate.cpp#L1046-L1064).

Послеотрицательногорезультата отдельно разрешено новоеисследование:matchedgreedy10 main/pairs300 иcanonical-stylebaggedES main/pairs. Это будутдругиеsource/freeze/output;они **не добавляются** вданныйзавершённыйэкран и не считаются уже рассчитанными. Внутреннийrandomholdout изуже прошедшихfit-строк разрешён какчастьfitalgorithm,но не называется временнойвалидацией;внешние chronologicalinner/outerпо-прежнемуразделены.

Артефакты: `analysis/hw4-six-hour-20261122/ebm/`:freeze,inner-selection,screenOOF/summary/quarter-metrics,forecast-manifest,`audit/verification.json`. Реальныйсчётклассическихветокэтогошестичасовогоокна сейчас **117fit=81boosting+9TD+27EBM**,безстарых108 ибезsyntheticQA.
