# Дополнительные проверки после Public11/12

Пользователь явно продолжил вычислительную работу после результатов TabICL94
0.34642 и GAM75/TabICL94 0.34242. Лучший подтверждённый Public остаётся у
context CatBoost04:0.34646; baseline0.35212 пока не побит. Последняя известная
история оставляет две попытки. Отправки и финальные checkbox агент не менял.

Новые опыты находятся отдельно в `analysis/hw4-post-public-20261122/`.
Они исследовательские, выполнены после просмотра Public и используют уже
многократно просмотренные OLD3:2023Q3,2023Q4,2024Q1. Новых метрик2024Q2/Q3
нет. Старые87 frozen source/evidence, рецепты и CSV не изменены.

Итог завершённых вычислений:151 обучение для failure/auxiliary задач и32
domain-membership обучения, всего183 завершённые модели. В151 входят четыре
реальных двухэпоховых измерения скорости neural-моделей без AP. Отдельно один
CatBoostMAP fit прерван по первоначальному30-секундному ограничению до
первой модели или AP; неизменные параметры затем получили заранее объявленный
бюджет90сек/fit,15мин всего. Синтетические QA и foundation-контексты в183
не входят. Всего184 реальные optimizer-попытки:183 завершённых и одна прерванная.

| Проверка | OLD mean AP | Вывод |
|---|---:|---|
| Context CatBoost04, mean3 | .267506 | Опора |
| TabICL94 native4 | .281069 | Сильнейшая OLD опора |
| C04/ICL94 equal50 odds | .276651 | Ниже ICL во всех3 кварталах |
| C04/ICL94 equal50 probability | .276528 | Ниже ICL во всех3 кварталах |
| C04/ICL94 equal50 ranks | .275743 | Ниже ICL во всех3 кварталах |
| Own22 CatBoost, seed42 | .266549 | 3fit,0full |
| Sqrt class-weight CatBoost, mean2 | .265557 | Ниже matched unweighted mean2=.266536 |
| XGBoost MAP inner-selected algorithm | .263452 | 30fits с matched logistic;0full |
| XGBoost matched logistic algorithm | .257775 | Отрицательный контроль |
| CatBoost native MAP algorithm | .261745 | 9fits;0full |
| CatBoost MAP / context rank50 | .266242 | Mean-gate не пройден |
| Cached LR code teacher + CatBoost | .264469 | Отрицательно |
| Nonlinear code teacher + CatBoost | .267448 | Worst delta−.00861; отбор не пройден |
| Context rank-normalized compact94 blend75 | .269289 | Слабый формальный резерв; Q4 delta−.003194 |
| Same-season weighted CatBoost | .262608 | Ниже опоры во всех3 кварталах |
| Own first-routine-per-quarter features + CB | .262026 | Ниже опоры во всех3 кварталах |
| Own22 matched MLP | .263684 | Ниже context |
| Own22 retrieval-MLP | .265173 | Лучше matched MLP, но ниже context |
| Compact94 matched MLP | .255426 | Ниже context |
| Compact94 retrieval-MLP | .245904 | Ниже matched MLP и context |

В каждой строке зафиксированы собственные seed/алгоритм выбора; это не
одинаково независимые оценки. Полные параметры, matched controls и split
находятся в соответствующих freeze и отчётах. Положительный общий результат
rank-normalized смеси не доказывает пользу преобразования: matched
seed42 rawcompact94=.272355, rankcompact94=.270669. Для двух оставшихся
попыток этот резерв не рекомендуется.

Domain-классификаторы без failure-target дали AUROC около.49 для static3,
.53 для собственных признаков и почти1 для peer-context. После rank-нормализации
внутри квартала AUROC всё ещё.999979. Это различимость временных когорт,
а не доказательство причины переноса качества. Контрфактические matched
абляции пользу нормализации не подтвердили. Density weights по собственным
признакам почти единичны; по context отсутствует overlap — их не применяли.

В nonlinear code teacher известный исход уже завершённой routine предсказывается
по44 code-presence плюс count. Все визиты query-site hash-бакета исключены
из соответствующего teacher fit; ровно45 heads для9 прошлых snapshot.
Три агрегата этих прогнозов добавлены к fixed CatBoost. Это не использование
будущего официального target. Независимый causal/alignment audit прошёл.

Приоритетный внешний расчёт — тот же compact94 TabICLv2 с32 native-компонентами
вместо4. Код опубликован коммитом3b993593caae229ef4cc06093f8e61800e8fac2d.
Четыре jobs:OLD3 и полный test; ни prediction, ни score Q2/Q3 нет. Запуск
V100 пользователем на момент этого документа не подтверждён. Публичные веса,
пакеты, признаки и весь исторический контекст сохранены. Реальный native32
synthetic forward проверен с actual32 transformed views и EXACT replay.

Отдельно зафиксирован **исследовательский резерв15**: equal50 logit-смесь
именно прежнего CatBoost04 и **native4** TabICL94. Строгий OLD gate против
ICL остаётся FAIL и не переписан. Дополнительный допуск основан на практически
равных Public двух компонентов и служит проверкой разнообразия при несовпадении
локального/Public порядка моделей. Это адаптивное решение после Public,
а не подтверждённое улучшение. Коэффициенты50/50 не подбирались по Public;
из трёх заранее вычисленных OLD форм выбрана odds, её преимущество над
probability лишь.000123. Подстановка native32 в эту формулу не разрешена.
CSV SHA3a64705549d64f3cc729ee32af6f522b8d8f61ce47bc9fa5effe9a2284851bef;
freeze SHA58e39482fb2a94d16b29699ee0803649bf8aa143f1a8b9ea4d798a1d9f119ced.
Независимый math replay всех2246 test и5815 OLD значений сошёлся до1.11e-16.
Скрипт `export_hw4_duo_reserve.py` воспроизводит CSV побайтно. Пока это резерв
до завершения новых neural-проверок, не автоматическая рекомендация отправить.

У двух исследовательских runner остались неисполняемые поля базового шаблона
provenance (старые seed/logloss/blend descriptions). Явные новые поля, исполненный
код и независимые receipts однозначно описывают фактические seed42/AP/rank50
либо ICL50. Исходные freeze/source сохранены, пересчётов для сокрытия этой
неточности нет. Детали в MAP и code-teacher независимых аудитах.

Источники методов: [XGBoost ranking](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html),
[CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking),
[TabR, официальный код](https://github.com/yandex-research/tabular-dl-tabr).
Neural-сравнение использовало один64-wide backbone, k16, AdamW.001/.0003,
seed42 и эпохи, выбранные на предыдущем inner-квартале. Соседи исключали
то же физическое site и все кварталы не раньше query. Это упрощённая
retrieval-MLP по идее TabR. Ни один raw или fixed50ICL вариант не прошёл gate.
Все28 сохранённых neural-forward повторились EXACT; девять целевых проверок
включали пустой контекст, маски, инвариантность к запрещённым меткам и градиенты.

Подробные отчёты: `hw4-post-public-shift-and-controls-20261122.md`,
`hw4-catboost-map-results-20261122.md` и
`hw4-post-public-retrieval-mlp-20261122.md`. Параметры и исследовательские данные
сохраняются локально; построчные данные и веса в Git не публикуются.
