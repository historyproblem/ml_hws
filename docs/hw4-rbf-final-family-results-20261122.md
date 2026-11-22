# HW4: RBF-SVC, последнее новое семейство раунда

22 ноября 2026. Финальный CAP1M run завершён за663.8с: **27 estimator-fit =24 INNER+3 OUTER**, все численно valid. Gate к прежней07 не пройден; seed17/2026, full-fit, test-предсказания и CSV не создавались. После этого новые семейства/лимиты в раунде не добавляются.

[Исходный план восьми рецептов](hw4-rbf-final-family-plan-20261122.md): own22/compact94 × C{.5,5} × gamma multiplier{.25,1}. Только FIT обучает log/median/RobustScaler/missing flags/OHE и gamma scale. Quarter — nominal; year/gap/raw IDs/target исключены. Kernel RBF, probability=True, tol1e-3, cache512MB, class_weight=None. Внутренний five-fold Platt CV находится только в прошлых FIT-строках; продукт всегда class1 probability, без выбора между margin/rank/probability после оценки.

Первый cap100k корректно остановился на девятом INNER-fit:8 успешных+1 failed attempt, OUTER0. [Отдельный согласованный pre-outer amendment](hw4-rbf-convergence-amendment-20261122.md) увеличил только convergence cap до1M. Старый source/freeze/output сохранены. Все24 INNER выполнены заново. Прежний failed case сошёлся при140507 final-margin iterations. Восемь повторно обученных старых complete моделей имеют EXACT одинаковые native margin/Platt/preprocessing/gamma компоненты; изменён только лимит. Повышение cap не было дополнительным AP-поиском.

## Внешний результат одного nested алгоритма

Все24 INNER закончены до первого OUTER. Выбор на previous-inner AP: own22/C5/g.25 для2023Q3 и2024Q1, compact94/C5/g.25 для2023Q4. Агрегированный OLD INNER full recipe был бы own22/C5/g.25 (mean INNER AP .26875047), но gate full не разрешил.

| Вектор, seed42 | Mean AP | Δ к07 | Δ≥0 | Worst Δ |
| --- | ---: | ---: | ---: | ---: |
| Новый GAM, диагностическая опора | .27699741 | +.00615707 | 3/3 | +.00405323 |
| Прежняя07 | .27084035 | 0 | 3/3 | 0 |
| `.75*07+.25*SVM` | .26805157 | −.00278877 | 2/3 | −.00987251 |
| Context04 | .26750641 | −.00333393 | 0/3 | −.00588998 |
| SVM raw | .23138479 | −.03945556 | 0/3 | −.05490193 |

Raw AP по кварталам .25141156/.22091303/.22182978. У смеси delta к07 +.00094272/−.00987251/+.00056347: два небольших положительных квартала не компенсируют сильный провалQ4. Mean и worst нарушают gate. Оба SVM-вектора ниже нового GAM во всех трёх кварталах. Source/численная корректность не превращают эту гипотезу в хороший финалист; результат отрицательный для конкретного ограниченного алгоритма, не теорема обо всех kernel-моделях.

## Протокол, учёт и артефакты

`analysis/hw4-six-hour-20261122/rbf-convergence/` содержит freeze,24 immutable INNER jobs,inner-selection,3 OUTER jobs,screenOOF/summary/quarter-metrics и COMPLETE manifest. В `screen-oof.csv` — `svm_raw`, `svm_blend75`, `previous07`, `context04`, `new_gam_raw` плюс official metadata. Это **один seed42**, не mean3. `confirmed-oof.csv` после отказа содержит только reference-векторы; его нельзя выдавать за подтверждённый SVM.

Native convergence failures при1M получили бы явный budget-invalid record без модели и AP; каждый квартал сохраняется, full aggregate допускает только valid-all3 recipes. На реальных24 INNER таких failures0. Ни один config не удалялся за низкий AP. Последующий OUTER/full failure остановил бы run без замены рецепта. Повторного повышения cap нет.

5 backend+7 runner synthetic QA PASS; native probes нового cap2000/5000 PASS. Независимый review проверил границы данных, запись invalid, resume/atomic/reload и отсутствие изменений C/gamma/tol/RNG. Old dev многократно использован; bootstrap интервалы nominal, не independent confirmation. Основные пакеты не обновлялись; внешних Q2/Q3-метрик и uploads0.

Общий расход SVM: **36 estimator-fit attempts =9 первоначальных (8 completed+1 numerical-invalid)+27 финальных completed**. Внутренние five-fold Platt optimizations входят в стоимость каждого estimator.fit и не являются дополнительными HPO recipes. Общий расход данной классической ветки шестичасового раунда — **177 attempts,176 completed**:81boosting+9TD+27cyclicEBM+24canonicalEBM+36SVM. Старые108, root GAM/events и GPU-ветки сюда не входят.

Источники: [SVC1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html), [SVM guide](https://scikit-learn.org/1.7/modules/svm.html), [native solver/status propagation](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/svm/src/libsvm/svm.cpp).

`audit/verification.json`: independent loaded-model forwards для всех27 job EXACT; gamma пересчитана из каждого FIT-design; INNER selection независимо воспроизведён, все inner AP/logloss и old quarterAP имеют ошибку0, смесь07 реконструирована EXACT, official5815 IDs/site/quarter/target совпали. Signature `07496753311e767a97e837cd42b5ca02ec70fa107647bff6e0a4cc7014b1f49f`. Это согласованность сохранённых артефактов, не независимая запись исполнения до их сохранения.
