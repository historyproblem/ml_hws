# Последнее новое семейство: RBF-SVC, 22 ноября 2026

После закрытия EBM согласован ровно один ограниченный RBF-screen. Новых архитектур и расширений не добавляется. Данные, old3 chronology, previous07 gate и strong-GAM domination guard наследуются как явно зафиксированные правила; старые source/freeze не меняются.

## Источник и смысл

[Официальный SVC scikit-learn1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html), [SVM guide](https://scikit-learn.org/1.7/modules/svm.html) и [LIBSVM FAQ](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html) задают RBF kernel, зависимость от масштаба признаков, вычислительную стоимость и встроенную вероятностную калибровку. Здесь проверяется другое предположение о зависимости риска: гладкая локальная близость полностью обработанных прошлых профилей вместо суммы сплайнов или ветвей дерева. Это гипотеза, а не обещание прироста.

## Ровно восемь рецептов

`{own22, compact94} × C{0.5,5} × gamma_multiplier{0.25,1}`. Constant C не нормируется по числу строк. Full gamma = multiplier / `(D * Var(Z_FIT))`, после всех обученных только на FIT преобразований; это multiplier к стандартному sklearn `gamma='scale'`. При нулевой variance fit отвергается. Все восемь config оцениваются только на previous inner, не восемь раз на outer.

Common: `SVC(kernel='rbf', probability=True, class_weight=None, shrinking=True, cache_size=512, tol=1e-3, max_iter=100000, random_state=seed)`. Tol1e-3 совпадает с документированным default; конечный cap ограничивает зависание, он не тюнинговая ось. `ConvergenceWarning` или `fit_status_ != 0` означает **INVALID**, частичная модель не принимается и не сравнивается как кандидат. Oversampling и class weighting отсутствуют. Версии и native LIBSVM/source SHA фиксируются, установки и обновления пакетов не нужны.

## Fit-only геометрия

Own22 — точные static+burden компоненты compact96 без year/gap; compact94 добавляет все ctx72. Quarter и facility/risk/area — nominal. Raw row/site IDs, target и даты запрещены. Числовой ноль сохраняется. У неотрицательных count/load/burden полей `log1p`; signed-log только для `violation_change_last_vs_mean`; ctx fail rates остаются без log. Затем FIT-медиана (all-missing→0), FIT RobustScaler с IQR25–75 для numeric, отдельные raw missing flags для **каждого** numeric, dense OHE `handle_unknown='ignore'`. Clipping отсутствует. Категориальные словари, medians, scale и gamma не видят внешний eval.

Известные разные OHE-категории дают квадрат расстояния2 независимо от числа уровней; неизвестная категория даёт allzero. Коррелированные ctx и повторяющиеся маски всё же могут перевешивать own-профиль: две заранее заданные view проверяют именно это. При текущих категориях ориентировочные размерности own22≈83 и compact94≈227; фактические FIT-размерности и gamma сохраняются в bundle.

`probability=True` использует встроенную five-fold Platt calibration **только на FIT-past**. Это разрешённая часть estimator, не temporal validation. Финальный продукт — class1 probability; decision score не будет альтернативой, выбираемой после outer scoring. Seeds17/2026 меняют calibration folds, а не случайную инициализацию самой margin-задачи. Счётчик estimator.fit отдельно объясняет внутренние calibration subfits.

## Бюджет и протокол

До HW4 — native synthetic probes2000 и5000 строк с реальным дорогим compact94/C5/mult1/probabilityTrue/cache512/max_iter100000. Записываются время, support/iterations/status и memory; по ним сообщается ETA. LIBSVM не имеет n_jobs: один fit около одногоCPU; thread pools ограничены2, гипотезы запускаются последовательно. Cache512MB ограничивает kernel-cache, полная Gram-матрица20k² не материализуется нами.

24 INNER-fit полностью завершаются до первого OUTER-fit. Для outer2023Q3/Q4/2024Q1 recipe выбирается только AP предыдущего квартала; ties: logloss, меньшие C/multiplier, ID. Full recipe — highest mean трёх OLD INNER AP с теми же ties. После3seed42outer проходят два заранее фиксированных продукта: raw и `.75*previous07+.25*RBF`. Gate: mean delta07>0, ≥2/3 неотрицательных quarters, worst≥−.005. При допуске6 confirmation17/2026; повторный gate. Затем full только если не сработал заранее заданный guard: **оба** raw/blend хуже нового GAM во всех трёх quarters и каждый mean gap≥.002.

Максимум36 estimator-fit:24inner+3outer+6confirmation+3full; у каждого native five-fold calibration — не скрытый дополнительный recipe-search. Все модели/параметры и predictions атомарно сохраняются; reload обязан быть EXACT. Freeze включает source/runtime/raw/cache/anchors/config SHA, resume отвергает любые изменения. OOF сохраняется при любом результате. Новых внешних Q2/Q3-метрик и Kaggle uploads нет. Old dev reused; bootstrap CI nominal, независимое подтверждение не заявляется. После этого семейства поиск заканчивается, root замораживает global shortlist до разрешённого one-shot recent audit.

## Проверки до первого HW4-fit

Native probe завершён: compact94/C5/gamma multiplier1 с настоящими probability=True/max_iter100000 — 2000 строк1.07с,5000 строк6.36с; final-margin n_iter4340/11621, fit_status0, точный reload, finite probabilities. Peak process RSS до435MB. Артефакт `analysis/hw4-six-hour-20261122/rbf-native-probe.json` содержит source SHA. Это синтетика; HW4 время не измерено, предварительный ориентир sequential run20–35мин с уточнением по первому planned INNER-fit. Бюджет не изменён.

Дополнительно проверен [native sklearn1.7.2 LIBSVM source](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/svm/src/libsvm/svm.cpp): `svm_binary_svc_probability` передаёт один status pointer во все пять calibration train calls, `svm_train_one` объединяет timeout в этот status. Поэтому iteration-cap failure calibration-fold тоже делает общий fit invalid. `n_iter_` описывает только final-margin optimization, его нельзя выдавать за сумму всех native calibration iterations.
