# OOF GAM baseline и неглубокая CatBoost-коррекция

План до расчётов: 22 ноября 13:46:11 UTC,
SHA `08584d6bf504f53a39d26e7bf4fcd54ec4ed7593bf5c53d3fc1a7e90515aadfd`.
Один фиксированный teacher: unweighted interaction-GAM/C=.01. Каждая строка
meta-training получает raw decision function teacher, обученного строго на
предыдущих официальных кварталах. Первые 1 860 строк 2022Q1 исключены из
meta-training, поскольку для них нет предшествующего teacher-fit.

Шесть отсутствовавших teacher-фолдов обучены; четыре прежних OOF checkpoint
переиспользованы. Для 2024Q2/Q3 рассчитаны только обычные chronological OOF
входы последующего обучения, **без AP**. Full teacher не переобучался.
Принадлежность каждой строки и cutoff проверены по model/raw/row-ID SHA.

CatBoost получает исходные бинарные labels и `Pool.baseline=teacherOOFraw`,
а не искусственные непрерывные residual targets. Рецепт: depth3, lr.03,
L2=20, rsm.8, Bernoulli.8, boost_from_average=False, CPU2. По предыдущему
inner AP выбирается только число деревьев 0/50/100/200/400, earliest tie=0.
На inference используется `expit(teacherRaw + CB.RawFormulaVal(X))`, без
baseline в inference Pool. Native `ntree_end=0` означает все деревья, поэтому
нулевой correction реализован отдельным sentinel, EXACT равным teacher.
[Официальный baseline API](https://catboost.ai/docs/en/concepts/python-usages-examples#using-pre-training-results-baseline).

Нативный Pool1.2.10 хранит baseline в float32; caller/inference logits остаются
float64. Tiny numeric probe проверил точное сложение residual Raw и native
baseline. Предсказания вероятностей не складываются в raw-logit композиции.

| Модель | 2023Q3 | 2023Q4 | 2024Q1 | Среднее AP |
|---|---:|---:|---:|---:|
| Fixed matched teacher E/C=.01 | .281863744 | .285657957 | .267830091 | **.278450598** |
| Предыдущий sequential GAM алгоритм | .282710945 | .280451200 | .267830091 | .276997412 |
| Residual | .281863744 | .272613341 | .267830091 | .274102392 |
| 50% matched teacher + 50% residual | .281863744 | .279290464 | .267830091 | .276328100 |

Выбраны counts0/200/0, медиана0. Correction ухудшил Q4; два остальных outer
остались EXACT teacher. К matched teacher raw-дельта−.004348205,
nominal95%site-CI[−.008454384,−.000722165]; половинная смесь−.002122498,
CI[−.004070609,−.000201165]. Оба gate отвергнуты. Полных residual-fit и CSV нет.
Это отрицательный результат конкретной коррекции; исходный fixed teacher
сохранён. Отдельно выявлен его старый matched AP.278450598: он сравнивается
с residual как заранее фиксированный контроль, а не выбирается из 24 outer.

Риск модели: маленькие ранние OOF teacher и большой full teacher имеют разное
качество и coverage. Fit/preprocessing причинны по времени; исследовательский
выбор самой E-рецептуры ранее использовал этот старый dev. Нельзя называть
интервалы независимым доказательством будущего Private.

Аудит PASS: 10 teacher raw checkpoint и 6 meta checkpoint переиграны EXACT,
четыре native CB fit и два настоящих zero sentinel, все inner-grid решения и
15 внешних AP воспроизведены. Resume не обучает модели. Артефакты:
`analysis/hw4-six-hour-20261122/additive/gam-baseline-residual/audit/verification.json`.
