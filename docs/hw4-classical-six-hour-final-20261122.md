# HW4: завершение классической ветки шестичасового раунда

22 ноября 2026, CPU-поиск закончен к15:01 UTC, все итоговые проверки — к15:05. **177 estimator-fit attempts:176 завершённых и1 numerical-invalid.** Новых методов после последнего RBF-screen нет. Старые108 fit, root GAM/events и GPU-прогоны в этот счёт не включены. Kaggle uploads и новых внешних2024Q2/Q3-метрик у этой ветки0.

Единственный новый допущенный CSV — canonical EBM, заранее фиксированная probability-смесь75% прежней07+25% EBM. Она улучшила mean old AP на.00117794, но уступает новому GAM raw во всех трёх кварталах. Поэтому это компонент/резерв для общего зафиксированного shortlist, а не основание объявить победу над Public baseline или расходовать попытку вслепую.

## Что реально проверено

В каждом поиске конфигурация выбиралась на строго предыдущем INNER-квартале, после чего оценивался один внешний алгоритм выбора. Все соответствующие INNER-fit закончены до первого OUTER-fit. Ниже mean AP на old2023Q3/Q4/2024Q1; это не Public score.

| Алгоритм | Raw AP | Fixed75/25 AP | Опора смеси/gate | Решение |
| --- | ---: | ---: | --- | --- |
| LightGBM GBDT tuner | .260778 | .267140 | context04 | fail |
| LightGBM DART tuner | .258977 | .267100 | context04 | fail |
| CatBoost rsm/bootstrap/L2 tuner | .264656 | .266462 | context04 | fail |
| Pooled previous-inner selector | .264817 | .267054 | context04 | fail |
| Author CB_TD + matched L2 control tuner | .268097 | .270796 | previous07 | fail |
| Controlled cyclic EBM8 selector | .251703 | .268954 | previous07 | fail |
| Canonical-style greedy/bagged EBM4 selector, mean3 | .273297 | .272018 | previous07 | только blend pass |
| RBF-SVC8 selector, seed42 | .231385 | .268052 | previous07 | fail |

Опоры: context04=.26750641; previous07=.27084035; новый GAM другой ветки=.27699741. Четыре boosted-selector получили общий81-fit budget (72inner+9outer); pooled переиспользовал готовые outer, не добавлял3 fit. TD9,cyclic EBM27,canonical EBM24,RBF36 attempts. Ни одно отрицательное семейство не получило необоснованный full-fit.

LightGBM-поиск действительно расширил слабую прежнюю область: full346/no-code170, leaves7–63,minleaf20–80,L2/featurefraction/extra_trees; DART использовал реальные checkpoints в моментN, а не неверные префиксы поздней модели. CB проверял новые rsm/Bernoulli/Bayesian/L2 комбинации. TD опирался на фактический author preset с отдельно согласованным L2-контролем. Это ограниченные source-grounded тесты, а не исчерпание всех возможных моделей.

EBM после отрицательного controlled screen получил отдельную честную проверку настоящего upstream-style greedy/bagged алгоритма. Именно этот заранее согласованный составной рецепт был выбран на всех INNER и дал допущенный blend. Из результата нельзя отдельно причинно вывести эффект greedy, bins или bagging. Исходный controlled screen и его freeze не переписаны.

RBF применял fit-only log/median/RobustScaler/OHE/missing masks, nominal quarter и gamma scale к двум фиксированным view. Cap100k остановился до первого outer; согласованный численный amendment1M сохранил все8 recipes и запустил24 INNER заново. Восемь старых успешных моделей после refit совпали EXACT по native margin/Platt/preprocessing/gamma; бывший failing case потребовал140507 iterations. Ни один квартал или низкий AP не исключался. Финальные27 fit численно корректны, но предсказательно слабы; seed/full/CSV не запускались. Расход36 attempts включает8 старых complete+1failed+27новых, а не только успешный rerun.

## Результат, который можно передать дальше

Canonical EBM mean3 raw AP=.27329656; raw worst delta к07=−.00522265 нарушает gate. Fixed blend AP=.27201829,meanΔ=.00117794,2/3 nonnegative,worst−.00041583 проходит. Номинальный95% site-bootstrap CI meanΔ=[−.00107810,+.00335989]. Предзаданный BOTH-domination guard не запретил full, потому что raw EBM слегка обогнал GAM вQ4; смесь сама ниже GAM во всех3. Это reused old development, не независимое подтверждение gain.

Папка `analysis/hw4-six-hour-20261122/ebm-canonical/`:

- `confirmed-oof.csv`: raw mean3 в `ebm_raw`, фиксированная смесь в `ebm_blend75`, aligned official metadata и reference-векторы.
- `models/ebm_canonical_pairs10__full__s{42,17,2026}/prediction.npy`: raw full-test components; averaging не требует fit.
- `submission_ebm_canonical_blend75.csv`:2246 строк, sample order, SHA256 `4f852438dc1546fa2b93a49dec062439b0e00a3721b6fccde150b19d7159f0fd`.
- `audit/verification.json` и `audit/own-readonly-review.json`: независимые SHA/IDs/mean/blend/AP проверки; reloaded forwards EXACT.

Full recipe frozen по OLD INNER: `CANONICAL_RECIPES['ebm_canonical_pairs10']`; новое внешнее count/term/weight tuning по recent не разрешается. Root отдельно формирует фиксированные общие смеси и замораживает shortlist до пользовательского one-shot recent diagnostic.

RBF OOF толькоseed42 находится в `rbf-convergence/screen-oof.csv`, `svm_raw`/`svm_blend75`; это не подтверждённый mean3. Его independent audit подтвердил все27 forward, FIT-only gamma и scores с ошибкой0. Конкурентного полного RBF CSV нет.

## Проверяемость и границы

Каждый run содержит source/raw/cache/anchor/runtime/config fingerprints, freeze до fit, immutable atomic job artifacts, точный joblib reload и strict resume. Пакеты основного окружения не обновлялись; официальный interpret-core0.7.8 установлен отдельно `--no-deps`. Старые training modules/runners не изменены; новые файлы отдельные. Сводка точных metric rows, counters, signatures и manifest SHA — `analysis/hw4-six-hour-20261122/classical-final-branch-summary.json`.

Подробности: [boosting24](hw4-boosted-six-hour-results-20261122.md), [CB_TD](hw4-td-six-hour-results-20261122.md), [cyclic EBM](hw4-ebm-cyclic-results-20261122.md), [canonical EBM](hw4-ebm-canonical-results-20261122.md), [RBF и численный amendment](hw4-rbf-final-family-results-20261122.md). Использование source-provenance не является независимым доказательством исполнения до сохранения моделей. Победа над последним известным Public baseline.35212 пока не получена; автоматических отправок нет.
